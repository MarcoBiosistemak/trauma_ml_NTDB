"""Audit a parquet for the NTDB columns the pipeline expects.

Usage:
    python -m trauma_ml.cli.audit_parquet path/to/unified_train.parquet

Reports for each expected column:
    PRESENT and POPULATED (✓)  — column exists, < 30% NaN
    PRESENT but SPARSE (⚠)     — column exists, 30-99% NaN
    PRESENT but EMPTY (✗)      — column exists, ≥ 99% NaN
    MISSING from schema (✗)    — column not in parquet at all

For 100%-NaN columns derived from joins (NISS, TRAUMATYPE, etc.) it
points the user at the most likely cause in the build pipeline.

Designed to be run as a sanity check after ``trauma-build`` and before
``trauma-train`` — every ✗ line is a build problem the user should fix
before training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Columns the pipeline cares about, grouped for diagnosis
_REQUIRED = {
    "Demographics": [
        ("AGEYEARS",
         "Patient age in years. Used for TRISS age cutoff (≥55) and "
         "stratification.  If missing, check raw CSV for 'AGEyears' "
         "(mixed case, NTDB AY 2019) or AGE_YEARS / PATIENTAGE — the "
         "alias resolver in harmonise_year handles case variants."),
        ("SEX",  "Patient sex (NTDB code 1=M, 2=F).  Subgroup analysis."),
    ],
    "On-scene physiology (also TRISS RTS inputs)": [
        ("GCSTOTAL", "Glasgow Coma Scale total. Aliases: TOTALGCS, GCS_TOTAL, GCS."),
        ("SBPFIRST", "First systolic blood pressure.  Aliases: SBP, FIRSTSBP, INITIALSBP."),
        ("RRFIRST",  "First respiratory rate.  Aliases: RESPIRATORYRATE, RR, FIRSTRR."),
    ],
    "ED-arrival physiology (extends ed_complete cohort)": [
        ("TEMPERATURE",   "ED-arrival temperature."),
        ("PULSEOXIMETRY", "ED-arrival oxygen saturation."),
        ("PULSERATE",     "ED-arrival pulse / heart rate."),
    ],
    "Anatomy scores (ISS / NISS baselines)": [
        ("ISS",  "Injury Severity Score (also ISS_05 in AY 2019 → renamed ISS)."),
        ("NISS", "New Injury Severity Score.  ⚠ DERIVED by the build from "
                 "PUF_AISDIAGNOSIS.csv — if 100% NaN, the AIS file was not "
                 "found at the path expected by ntdb_tables['aisdiagnosis']."),
    ],
    "Mechanism (TRISS blunt/penetrating split)": [
        # PRIMARYECODEICD10 is the JOIN KEY used during the build to derive
        # TRAUMATYPE/MECHANISM/INTENT from PUF_ECODE_LOOKUP.csv.  It's a
        # high-cardinality string that's not useful as a feature and lives
        # in NON_PREDICTOR_COLUMNS — the build can drop it from the parquet
        # after the join.  Don't flag it as MISSING here; the derived columns
        # below are what actually matters for downstream training.
        ("TRAUMATYPE",
         "1=Blunt, 2=Penetrating, 3=Burn, 4=Other.  ⚠ JOINED from "
         "PUF_ECODE_LOOKUP.csv via PRIMARYECODEICD10 — if 100% NaN, the "
         "lookup CSV was not found at ntdb_tables['ecode_lookup'].  TRISS "
         "will fall back to blunt coefficients for every row."),
        ("MECHANISM", "Same join as TRAUMATYPE."),
        ("INTENT",    "Same join as TRAUMATYPE."),
    ],
    "Outcome variables": [
        ("HOSPDISCHARGEDISPOSITION", "5 = deceased.  Mortality outcome."),
        ("EDDISCHARGEDISPOSITION",   "5 = deceased in ED.  Mortality outcome."),
    ],
    "Comorbidities (Tran 2022 S1 Table)": [
        ("COPD",                                "0/1 flag."),
        ("CHF",                                 "0/1 flag."),
        ("HYPERTENSION",                        "0/1 flag."),
        ("DIABETESMELLITUS",                    "0/1 flag."),
        ("DISSEMINATEDCANCER",                  "0/1 flag."),
        ("ADVANCEDDIRECTIVELIMITINGCARE",       "0/1 flag."),
        ("FUNCTIONALLYDEPENDENTHEALTHSTATUS",   "0/1 flag."),
    ],
    "Barell injury features (round 27)": [
        ("BARELL_TBI",
         "0/1 flag — derived by ntdb_loader.merge_barell_features from "
         "PUF_AISDIAGNOSIS ICDDIAGNOSISCODE column.  If 100% NaN/missing, "
         "either (a) the diagnosis table wasn't loaded, or (b) the ICD-10 "
         "code column has a non-standard name.  Check the build log."),
        ("BARELL_THORAX",        "0/1 flag — see BARELL_TBI."),
        ("BARELL_ABDOMEN_PELVIS", "0/1 flag — see BARELL_TBI."),
        ("BARELL_LOWER_EXTREMITY", "0/1 flag — see BARELL_TBI."),
        ("INJ_SUBDURAL_HEMORRHAGE", "0/1 specific-injury flag."),
        ("INJ_FEMUR_FRACTURE",      "0/1 specific-injury flag."),
    ],
    "Partitioning": [
        ("__admission_year", "Added by harmonise_year — used for temporal holdout."),
    ],
}


def audit_parquet(parquet_path: Path) -> int:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    schema_cols = set(pf.schema.names)

    # First pass: figure out which expected cols are even in the schema
    needed = [c for group in _REQUIRED.values() for c, _ in group]
    present = [c for c in needed if c in schema_cols]

    # Read only the present columns to compute NaN rates
    if present:
        df = pf.read(columns=present).to_pandas()
        nan_pct = {c: 100 * df[c].isna().mean() for c in present}
        del df
    else:
        nan_pct = {}

    print(f"\nAuditing parquet: {parquet_path}")
    print(f"  Rows: {pf.metadata.num_rows:,}")
    print(f"  Cols: {pf.metadata.num_columns}")
    print()

    n_missing = 0
    n_empty   = 0
    n_sparse  = 0

    for group_name, cols in _REQUIRED.items():
        print(f"== {group_name} ==")
        for col, description in cols:
            if col not in schema_cols:
                print(f"  [MISSING]  {col}")
                print(f"      {description}")
                # Suggest case-insensitive variants if any
                lower = col.lower()
                near = [c for c in schema_cols if c.lower() == lower]
                if near:
                    print(f"      Note: schema has {near[0]!r} — possibly a case mismatch?")
                n_missing += 1
            else:
                pct = nan_pct.get(col, 0)
                if pct >= 99:
                    print(f"  [EMPTY]    {col}: {pct:.1f}% NaN")
                    print(f"      {description}")
                    n_empty += 1
                elif pct >= 30:
                    print(f"  [SPARSE]   {col}: {pct:.1f}% NaN")
                    n_sparse += 1
                else:
                    print(f"  [OK]       {col}: {pct:.1f}% NaN")
        print()

    print("=" * 70)
    print(f"Summary: {n_missing} MISSING, {n_empty} EMPTY, {n_sparse} SPARSE")
    print("=" * 70)
    if n_missing == 0 and n_empty == 0:
        print("All expected columns are present and populated. Good to go.")
        return 0
    print()
    print("Action items:")
    if n_missing > 0:
        print(f"  • {n_missing} column(s) MISSING from schema. Check that")
        print(f"    your raw CSV has these columns (possibly under an alias")
        print(f"    name) and that the build whitelist isn't dropping them.")
    if n_empty > 0:
        print(f"  • {n_empty} column(s) EMPTY (≥99% NaN). For derived columns")
        print(f"    (NISS, TRAUMATYPE, MECHANISM, INTENT) the most common")
        print(f"    cause is the corresponding lookup CSV not being found")
        print(f"    at the path expected in ntdb_tables config.")
    print()
    return 1 if (n_missing > 0 or n_empty > 0) else 0


def main():
    parser = argparse.ArgumentParser(
        description="Audit a parquet for the NTDB columns trauma_ml expects."
    )
    parser.add_argument("parquet", type=Path,
                        help="Path to the unified parquet to audit.")
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"[ERROR] file not found: {args.parquet}", file=sys.stderr)
        sys.exit(2)

    sys.exit(audit_parquet(args.parquet))


if __name__ == "__main__":
    main()
