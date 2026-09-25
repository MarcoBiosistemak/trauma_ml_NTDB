"""NTDB loader — read PUF CSVs across admission years and build a unified dataset.

The main entry point is :func:`build_unified_dataset`, which:

1. Reads PUF_TRAUMA.csv for each requested admission year.
2. Harmonises the known year-to-year breaking changes (ISS rename, EMS* removal,
   AIS version, ECODE split — see sheet 4 of the mapping xlsx).
3. Joins PUF_ECODE_LOOKUP (TRAUMATYPE, MECHANISM, INTENT) to PUF_TRAUMA via
   PRIMARYECODEICD10.
4. Derives ISS/NISS helper columns and a unified injury-count field.
5. Applies the variable whitelist from the catalogue.
6. Concatenates all years into one parquet file.
7. Optionally drops INC_KEY (`anonymize=True`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .catalogue import Catalogue, PHASE_ORDER

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Case-variant column deduplication (round 35)
# ---------------------------------------------------------------------------
def _dedupe_case_variants(df: pd.DataFrame, *, context: str = "") -> pd.DataFrame:
    """Merge columns sharing a name modulo case, coalescing their values.

    NTDB CSVs across years ship the same logical column under different
    casings: e.g. AY 2019 has ``AgeYears``, AY 2021 has ``AGEYEARS``, AY
    2022 has ``AGEyears``.  After per-year alias resolution and the
    pd.concat across years, the unified frame ends up with all three as
    SEPARATE columns, each filled only for the years that originally
    used that casing.  Downstream predictor selection treats them as
    distinct, picks one (the dict-order winner), finds it mostly-NaN
    (because it only covers a subset of years), and drops it for
    exceeding the missingness threshold — so the column effectively
    vanishes from the model.

    This helper does the right thing in one pass:
      1. Group columns by their lowercase name.
      2. For each group with >1 member, coalesce values row-by-row
         (first non-null wins).
      3. Promote the all-upper-case variant as canonical (or the first
         variant if none is upper).  Drop the others.

    Idempotent — safe to call on already-clean frames.

    Round 35 places this at THREE points:
      - end of ``load_year`` (so per-year frames are clean post-alias)
      - after ``pd.concat`` in ``build_unified_dataset`` (catches
        cross-year collisions)
      - in the trainer (round 34) for already-built parquets
    """
    seen_lower: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for c in df.columns:
        cl = c.lower()
        if cl in seen_lower:
            groups.setdefault(cl, [seen_lower[cl]]).append(c)
        else:
            seen_lower[cl] = c

    if not groups:
        return df

    for cl, variants in groups.items():
        before = {v: int(df[v].notna().sum()) for v in variants}
        coalesced = df[variants[0]]
        for v in variants[1:]:
            coalesced = coalesced.fillna(df[v])
        canonical_name = next((v for v in variants if v.isupper()), variants[0])
        df[canonical_name] = coalesced
        for v in variants:
            if v != canonical_name and v in df.columns:
                df = df.drop(columns=[v])
        log.info(
            "%s_dedupe_case_variants: %s -> %s (was %s; now %d non-null)",
            f"[{context}] " if context else "",
            variants, canonical_name,
            ", ".join(f"{k}={v}" for k, v in before.items()),
            int(df[canonical_name].notna().sum()),
        )
    return df


# ---------------------------------------------------------------------------
# Year-to-year harmonisation (see sheet 4 of the xlsx)
# ---------------------------------------------------------------------------
def harmonise_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Resolve the known breaking schema changes across AY 2019-2024.

    Changes handled (see sheet 4 of the NTDB mapping xlsx)
    -------------------------------------------------------
    1. ISS renamed: ISS_05 (2019)  →  ISS (2020+).
    2. DEATHINED (2019-only) mapped to EDDISCHARGEDISPOSITION==5.
    3. ADDITIONALECODEICD10 (2019) → ADDITIONALECODE1/2 (2020+).
    4. EMS* prehospital physiology ONLY in 2019 and 2020; absent from 2021+.
       NaN-filled stub columns are NOT added — absence in df signals missingness.
    5. LOWESTSBP removed in 2022 and 2024 — noted in log only.
    6. A stable ``__admission_year`` column is always added.
    7. Vital-sign column aliases mapped to canonical names used downstream
       (GCSTOTAL / SBPFIRST / RRFIRST). NTDB has used several spellings
       across years: TOTALGCS↔GCSTOTAL, SBP↔SBPFIRST, RESPIRATORYRATE↔RRFIRST.
       We rename in-place so the catalogue / baseline / TRISS code can rely
       on the canonical names without per-year branching.
    """
    out = df.copy()
    out["__admission_year"] = year

    # ── 1. ISS rename ────────────────────────────────────────────────────
    if "ISS" not in out.columns and "ISS_05" in out.columns:
        out["ISS"] = out["ISS_05"]
        log.debug("AY %d: created ISS column from ISS_05", year)

    # ── 2. DEATHINED → EDDISCHARGEDISPOSITION ────────────────────────────
    if year == 2019 and "DEATHINED" in out.columns and "EDDISCHARGEDISPOSITION" not in out.columns:
        out["EDDISCHARGEDISPOSITION"] = np.where(out["DEATHINED"] == 1, 5, np.nan)
        log.debug("AY 2019: mapped DEATHINED → EDDISCHARGEDISPOSITION=5")

    # ── 3. ECODE split compatibility ─────────────────────────────────────
    if "ADDITIONALECODEICD10" in out.columns and "ADDITIONALECODE1" not in out.columns:
        out["ADDITIONALECODE1"] = out["ADDITIONALECODEICD10"]
        out["ADDITIONALECODE2"] = np.nan
        log.debug("AY %d: split ADDITIONALECODEICD10 into ADDITIONALECODE1/2", year)

    # ── 4. EMS physiology only in 2019 and 2020 ─────────────────────────
    ems_cols = [c for c in out.columns if c.startswith("EMS")]
    if year >= 2021 and ems_cols:
        log.warning(
            "AY %d: unexpected EMS* columns present (%s …) — "
            "NTDB sheet 4 states EMS physiology was removed from 2021+. "
            "Check your CSV source files.",
            year, ems_cols[:4],
        )
    if year >= 2021 and not ems_cols:
        log.debug(
            "AY %d: EMS* prehospital physiology not present (expected per sheet 4 caveat #2). "
            "On-scene physiology unavailable for this year; ED-arrival values are the closest proxy.",
            year,
        )

    # ── 5. LOWESTSBP removed in 2022 and 2024 ───────────────────────────
    if year in (2022, 2024) and "LOWESTSBP" in out.columns:
        log.warning(
            "AY %d: LOWESTSBP present but sheet 4 caveat #3 states it was removed "
            "from 2022 and 2024. Verify CSV source.",
            year,
        )

    # ── 7. Canonical names for vitals AND demographics ────────────────────
    # Map any known alias to the canonical name expected by the catalogue
    # and the baseline/TRISS code. Only renames if the canonical is absent
    # AND an alias exists, to avoid clobbering real data.
    #
    # CASE-INSENSITIVE matching is critical: NTDB AY 2019 uses ``AGEyears``
    # (mixed case), AY 2021/2022 use ``AGEYEARS`` (upper).  We build a
    # lower-case → actual-name index over the dataframe's columns and look
    # up aliases in lowercase to catch all variants.
    canonical_aliases = {
        "AGEYEARS": ("AGEYEARS", "AGE_YEARS", "AGE_YRS", "AGE_IN_YEARS",
                     "AGEINYEARS", "AGE", "PATIENTAGE", "PT_AGE_YR"),
        "GCSTOTAL": ("TOTALGCS", "GCS_TOTAL", "GCS"),
        "SBPFIRST": ("SBP", "FIRSTSBP", "SBP_FIRST", "INITIALSBP"),
        "RRFIRST":  ("RESPIRATORYRATE", "RR", "FIRSTRR", "RR_FIRST",
                     "INITIALRR", "RESP_RATE"),
    }
    # Build lowercase -> actual column name index (handles AGEyears, etc.)
    # Round 34: when MULTIPLE columns collapse to the same lowercase key
    # (e.g. NTDB ships AGEYEARS, AGEyears, AgeYears in the same year's CSV),
    # we must consolidate them into ONE canonical column.  The naive dict
    # comprehension overwrites silently and the trainer's predictor-selector
    # later misses the column entirely.  We coalesce by picking the variant
    # with the most non-null cells per row (i.e. .bfill across the variant
    # group), then drop the duplicates.
    col_lower_to_actual: dict[str, str] = {}
    duplicate_groups: dict[str, list[str]] = {}
    for c in out.columns:
        cl = c.lower()
        if cl in col_lower_to_actual:
            duplicate_groups.setdefault(cl, [col_lower_to_actual[cl]]).append(c)
        else:
            col_lower_to_actual[cl] = c

    # Consolidate duplicate-cased columns
    for cl, variants in duplicate_groups.items():
        # Coalesce: take first non-null value across variants per row
        coalesced = out[variants[0]]
        for v in variants[1:]:
            coalesced = coalesced.fillna(out[v])
        # Prefer the all-upper-case variant as the canonical, else the
        # first variant we encountered
        canonical_name = next((v for v in variants if v.isupper()), variants[0])
        out[canonical_name] = coalesced
        # Drop the others
        for v in variants:
            if v != canonical_name and v in out.columns:
                out = out.drop(columns=[v])
        log.info(
            "AY %d: coalesced duplicate-cased columns %s -> %s "
            "(%d non-null after merge, was %s)",
            year, variants, canonical_name,
            int(out[canonical_name].notna().sum()),
            ", ".join(f"{v}={int(out[v].notna().sum())}" for v in variants
                       if v in out.columns or v == canonical_name),
        )
        # Re-index col_lower_to_actual
        col_lower_to_actual[cl] = canonical_name

    for canonical, aliases in canonical_aliases.items():
        if canonical in out.columns:
            continue
        # First, look for the canonical name itself in any case variant
        if canonical.lower() in col_lower_to_actual:
            actual = col_lower_to_actual[canonical.lower()]
            if actual != canonical:
                out[canonical] = out[actual]
                # Round 35: drop the source column to prevent it persisting
                # into the unified parquet as a duplicate of `canonical`.
                if actual in out.columns:
                    out = out.drop(columns=[actual])
                log.info(
                    "AY %d: aliased %s -> %s and dropped source "
                    "(case-insensitive match for canonical name)",
                    year, actual, canonical,
                )
            continue
        # Otherwise look for any of the alias variants (case-insensitive)
        for alias in aliases:
            if alias.lower() in col_lower_to_actual:
                actual = col_lower_to_actual[alias.lower()]
                out[canonical] = out[actual]
                # Round 35: drop source after aliasing
                if actual in out.columns and actual != canonical:
                    out = out.drop(columns=[actual])
                log.info(
                    "AY %d: renamed %s -> %s and dropped source "
                    "(canonical name expected by catalogue / baselines)",
                    year, actual, canonical,
                )
                break

    # Final sanity check — log if AGEYEARS still missing
    if "AGEYEARS" not in out.columns:
        log.warning(
            "AY %d: AGEYEARS not present in CSV after alias resolution. "
            "Looked for aliases: %s. Actual cols starting with 'A': %s. "
            "Cohort eval and TRISS will be degraded.",
            year, list(canonical_aliases["AGEYEARS"]),
            sorted([c for c in out.columns if c.upper().startswith("A")])[:10],
        )

    # Round 35: belt-and-braces — clean up any remaining case-variant
    # duplicates that the alias resolver didn't catch (e.g. columns
    # not in `canonical_aliases` that still ship under multiple casings).
    out = _dedupe_case_variants(out, context=f"AY {year}")

    return out


# ---------------------------------------------------------------------------
# Ecode lookup join (mechanism, intent, trauma type)
# ---------------------------------------------------------------------------
def join_ecode_lookup(
    trauma: pd.DataFrame,
    ecode_lookup: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-join PUF_ECODE_LOOKUP onto PUF_TRAUMA via PRIMARYECODEICD10.

    Brings in the clinically important columns TRAUMATYPE, MECHANISM, INTENT.
    If the lookup is not available, the columns are filled with NaN and a
    warning is logged.

    Column-name candidates for the join key (checked in priority order):
        ICD10ECODE        — used in NTDB PUF AY 2021, 2022, 2024
        ICDECODE          — used in some older NTDB releases
        ECODE             — alternative older name
        PRIMARYECODEICD10 — if the lookup was already keyed to PUF_TRAUMA naming
    """
    out = trauma.copy()
    wanted_lookup_cols = ["TRAUMATYPE", "MECHANISM", "INTENT"]

    if ecode_lookup is None or "PRIMARYECODEICD10" not in out.columns:
        for col in wanted_lookup_cols:
            if col not in out.columns:
                out[col] = np.nan
        if ecode_lookup is None:
            log.warning("No PUF_ECODE_LOOKUP provided — TRAUMATYPE/MECHANISM/INTENT will be NaN")
        else:
            log.warning("PRIMARYECODEICD10 not found in PUF_TRAUMA — cannot join ecode lookup")
        return out

    lookup = ecode_lookup.copy()
    lookup.columns = lookup.columns.str.strip()   # guard against whitespace in CSV headers

    # Build a lower-case → actual-name index so we can match column names
    # case-insensitively.  Real NTDB lookup files use 'ECode' (mixed
    # case) for AY 2021+, not 'ECODE'.  This is the same pattern used
    # for harmonising vital-sign / age column names in harmonise_year.
    lookup_cols_lower = {c.lower(): c for c in lookup.columns}

    # Priority order — try each candidate name in any case variant
    join_key = None
    for candidate in ("ICD10ECODE", "ICDECODE", "ECODE", "PRIMARYECODEICD10"):
        actual = lookup_cols_lower.get(candidate.lower())
        if actual is not None:
            join_key = actual
            break

    if join_key is None:
        log.warning(
            "PUF_ECODE_LOOKUP has no recognised join key "
            "(looked for ICD10ECODE, ICDECODE, ECODE, PRIMARYECODEICD10 "
            "in any case). Actual columns: %s. "
            "TRAUMATYPE/MECHANISM/INTENT will be NaN — TRISS "
            "blunt/penetrating split will default to blunt.",
            list(lookup.columns),
        )
        for col in wanted_lookup_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out

    log.info("PUF_ECODE_LOOKUP join key: %r (%d rows)", join_key, len(lookup))

    # Resolve the wanted lookup cols case-insensitively too.  Build a
    # rename map so we end up with canonical UPPERCASE names regardless
    # of what the file used.
    rename_map: dict[str, str] = {}
    available_wanted: list[str] = []
    for canonical in wanted_lookup_cols:
        actual = lookup_cols_lower.get(canonical.lower())
        if actual is not None:
            available_wanted.append(actual)
            if actual != canonical:
                rename_map[actual] = canonical

    if not available_wanted:
        log.warning(
            "PUF_ECODE_LOOKUP found (join key=%r) but none of %s are present "
            "(in any case). Columns in lookup: %s",
            join_key, wanted_lookup_cols, list(lookup.columns),
        )
        for col in wanted_lookup_cols:
            if col not in out.columns:
                out[col] = np.nan
        return out

    keep = [join_key] + available_wanted
    lookup_small = lookup[keep].drop_duplicates(subset=[join_key])
    # Build the rename used for the merge: join_key → PRIMARYECODEICD10,
    # plus any case-mismatched wanted cols → canonical UPPERCASE.
    merge_rename = {join_key: "PRIMARYECODEICD10", **rename_map}
    before_rows = len(out)
    out = out.merge(
        lookup_small.rename(columns=merge_rename),
        on="PRIMARYECODEICD10",
        how="left",
    )
    assert len(out) == before_rows, \
        f"Row count changed after ECODE join ({before_rows} → {len(out)}); check for duplicate keys"

    # Fill any columns that were absent in the lookup
    for col in wanted_lookup_cols:
        if col not in out.columns:
            out[col] = np.nan

    matched = out["TRAUMATYPE"].notna().sum()
    log.info(
        "ECODE join complete: %d / %d rows matched (%.1f%%) — "
        "TRAUMATYPE/MECHANISM/INTENT populated",
        matched, len(out), 100 * matched / max(len(out), 1),
    )
    return out


# ---------------------------------------------------------------------------
# NISS derivation from PUF_AISDIAGNOSIS
# ---------------------------------------------------------------------------
def derive_niss(
    trauma: pd.DataFrame,
    aisdiag: pd.DataFrame | None,
    inc_key_col: str = "INC_KEY",
) -> pd.DataFrame:
    """Add a NISS column — sum of squares of the three highest AISSEVERITY values
    per patient, regardless of body region.

    Silently fills with NaN and logs a WARNING if PUF_AISDIAGNOSIS is
    missing or has no recognisable severity / id columns.

    Real NTDB AY 2021+ AIS files use mixed-case column names — e.g.
    ``inc_key`` (lowercase) and ``AISSeverity`` (mixed case).  This
    function resolves both case-insensitively against the trauma
    dataframe (which has been harmonised to uppercase by ``harmonise_year``).
    """
    out = trauma.copy()
    if aisdiag is None:
        log.warning(
            "PUF_AISDIAGNOSIS not provided — NISS will be 100%% NaN. "
            "Check ntdb_tables['aisdiagnosis'] path in your build config."
        )
        out["NISS"] = np.nan
        return out

    # Case-insensitive lookup for the AIS-side column names
    ais_cols_lower = {c.lower(): c for c in aisdiag.columns}
    ais_inc_key = ais_cols_lower.get(inc_key_col.lower())
    ais_severity = ais_cols_lower.get("aisseverity")

    if ais_inc_key is None or ais_severity is None:
        log.warning(
            "PUF_AISDIAGNOSIS missing required columns. "
            "Wanted (case-insensitive): %r and 'AISSEVERITY'. "
            "Actual columns: %s. NISS will be 100%% NaN.",
            inc_key_col, list(aisdiag.columns),
        )
        out["NISS"] = np.nan
        return out

    log.info(
        "PUF_AISDIAGNOSIS join keys: AIS=(%r, %r), trauma=%r",
        ais_inc_key, ais_severity, inc_key_col,
    )

    # Trauma dataframe might also use mixed-case inc_key — resolve there too
    trauma_cols_lower = {c.lower(): c for c in out.columns}
    trauma_inc_key = trauma_cols_lower.get(inc_key_col.lower())
    if trauma_inc_key is None:
        log.warning(
            "Trauma dataframe lacks inc_key column %r (case-insensitive). "
            "NISS will be 100%% NaN.",
            inc_key_col,
        )
        out["NISS"] = np.nan
        return out

    severities = aisdiag[[ais_inc_key, ais_severity]].dropna()
    # Rename to canonical names locally for the join
    severities = severities.rename(columns={
        ais_inc_key: trauma_inc_key,
        ais_severity: "AISSEVERITY",
    })
    # NTDB convention: severity 9 = unknown (drop), 6 = max (keep)
    severities = severities[severities["AISSEVERITY"].between(1, 6)]
    severities = severities.sort_values(
        [trauma_inc_key, "AISSEVERITY"], ascending=[True, False],
    )
    top3 = severities.groupby(trauma_inc_key).head(3)
    niss = (
        top3.assign(sq=top3["AISSEVERITY"] ** 2)
        .groupby(trauma_inc_key)["sq"]
        .sum()
        .rename("NISS")
        .reset_index()
    )
    before_rows = len(out)
    out = out.merge(niss, on=trauma_inc_key, how="left")
    assert len(out) == before_rows, (
        f"Row count changed after NISS join ({before_rows} → {len(out)})"
    )
    n_populated = out["NISS"].notna().sum()
    log.info(
        "NISS derivation complete: %d / %d rows populated (%.1f%%)",
        n_populated, len(out), 100 * n_populated / max(len(out), 1),
    )
    return out


# ---------------------------------------------------------------------------
# Whitelist application
# ---------------------------------------------------------------------------
def apply_whitelist(
    df: pd.DataFrame,
    catalogue: Catalogue,
    phase_cutoff: str = "In-hospital (a posteriori)",
    registries: Iterable[str] | None = None,
    always_keep: Iterable[str] = (
        # Row identity & partitioning
        "INC_KEY", "__admission_year",
        # Outcome / target variables
        "HOSPDISCHARGEDISPOSITION", "EDDISCHARGEDISPOSITION", "DEATHINED",
        # Anatomy scores — needed for ISS and NISS baselines
        "ISS", "ISS_05", "NISS",
        # TRISS physiology inputs (Champion 1989) — must survive whitelist
        "GCSTOTAL", "SBPFIRST", "RRFIRST",
        # TRISS covariates
        "AGEYEARS", "TRAUMATYPE",
        # Demographics for subgroup analysis + Tran S1 Table.
        # There is no single native RACE column (NTDB ships RACE_* category
        # flags).  PRIMARYMETHODPAYMENT (insurance/payer) is kept as an
        # admission-time economic feature + sociodemographic subgroup axis.
        "SEX", "ETHNICITY", "PRIMARYMETHODPAYMENT",
        # Anthropometry (L1), all NTDB years
        "HEIGHT", "WEIGHT",
        # Raw ICD-10 diagnosis code list (faithful Doshi FFNN input; L3).
        "ICD_DIAG_CODES",
        # Mechanism / intent (from ECODE join)
        "MECHANISM", "INTENT",
        # First-recorded vitals beyond TRISS (these ARE the ED-arrival values;
        # NTDB has no separate "ED*"-prefixed or "*HIGHEST/*LOWEST" columns,
        # so those placeholders were removed in round 51).
        "TEMPERATURE", "PULSEOXIMETRY", "PULSERATE",
        # Round 51: new predictors (all native, all NTDB years 2019-2024)
        # L1 (on-scene): transport mode, prehospital cardiac arrest
        "TRANSPORTMODE", "PREHOSPITALCARDIACARREST",
        # L2 (ED arrival): incident->arrival time, pupils, alcohol screen
        # (NTDB has NO drug-screen field — only alcohol)
        "HOSPITALARRIVALHRS", "TBIPUPILLARYRESPONSE",
        "ALCOHOLSCREEN", "ALCOHOLSCREENRESULT",
        # (EDDISCHARGEHRS removed in round 56 — leakage; see catalogue
        #  NON_PREDICTOR_COLUMNS)
        # Comorbidities — Tran 2022 S1 Table (round 26)
        "SMOKINGSTATUS", "COPD", "CHF", "MI", "HYPERTENSION",
        "PERIPHERALVASCULARDISEASE", "ESRD", "CIRRHOSIS",
        "DIABETESMELLITUS", "BLEEDINGDISORDER", "DISSEMINATEDCANCER",
        "ALCOHOLUSEDISORDER", "MENTALPERSONALITYDISORDER",
        "SUBSTANCEABUSEDISORDERDRUG", "ATTENTIONDEFICITDISORDER",
        "DEMENTIA", "ADVANCEDDIRECTIVELIMITINGCARE",
        "FUNCTIONALLYDEPENDENTHEALTHSTATUS",
    ),
) -> pd.DataFrame:
    """Restrict the dataframe to variables in the catalogue whitelist.

    The whitelist is the union of
      * variables whose phase is <= phase_cutoff, AND
      * variables used by any listed registry, AND
      * always_keep (hard-wired must-haves for identifying, outcomes,
        stratification, comorbidities, ED-arrival vitals), AND
      * **all Barell-derived injury features** (BARELL_*, INJ_*) — these
        are added by ``merge_barell_features()`` after load_year and must
        survive whitelisting unconditionally.

    Round 28: case-insensitive matching (so 'race' in the parquet matches
    'RACE' in the whitelist).  This was the build-time analogue of the
    round-20 trainer fix.
    """
    # Round 27: ALWAYS keep the 22 Barell injury features
    from .barell import ALL_BARELL_FEATURES

    registries = list(registries) if registries else []
    whitelist = set(catalogue.variables_for(
        phase_cutoff=phase_cutoff,
        registries=registries or None,
        include_target_derivers=True,
    ))
    whitelist.update(always_keep)
    whitelist.update(ALL_BARELL_FEATURES)

    # Round 28: case-insensitive intersection.  Build a lowered-key index
    # of the whitelist; for every column in df, keep it if its lowercase
    # form appears in the lowered whitelist.  We KEEP the column under its
    # parquet-side casing (don't rename) so downstream code that resolves
    # case-insensitively (round-20 trainer logic) still works.
    whitelist_lower = {w.lower() for w in whitelist}
    keep = [c for c in df.columns if c.lower() in whitelist_lower]
    dropped = [c for c in df.columns if c.lower() not in whitelist_lower]
    log.info(
        "Whitelist kept %d / %d columns (dropped %d). "
        "Sample dropped: %s",
        len(keep), df.shape[1], len(dropped), dropped[:8],
    )
    # Diagnostic: which Barell / comorbidity columns survived?
    barell_kept = [c for c in keep if c.upper().startswith(("BARELL_", "INJ_"))]
    comorb_kept = [c for c in keep if c.upper() in {
        "COPD", "CHF", "MI", "HYPERTENSION", "DIABETESMELLITUS",
        "DISSEMINATEDCANCER", "ADVANCEDDIRECTIVELIMITINGCARE",
        "ALCOHOLUSEDISORDER", "DEMENTIA", "ESRD", "CIRRHOSIS",
        "BLEEDINGDISORDER", "PERIPHERALVASCULARDISEASE",
        "FUNCTIONALLYDEPENDENTHEALTHSTATUS", "SMOKINGSTATUS",
        "MENTALPERSONALITYDISORDER", "SUBSTANCEABUSEDISORDERDRUG",
        "ATTENTIONDEFICITDISORDER",
    }]
    log.info("  → %d Barell features kept: %s", len(barell_kept), barell_kept[:6])
    log.info("  → %d comorbidity columns kept: %s", len(comorb_kept), comorb_kept[:6])
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Per-year builder
# ---------------------------------------------------------------------------
def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Read a CSV if it exists, tolerating minor parsing issues."""
    if not path.exists():
        log.warning("Missing: %s", path)
        return None
    try:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")
    except Exception as exc:                                      # noqa: BLE001
        log.warning("Could not read %s (%s); trying utf-8", path, exc)
        try:
            return pd.read_csv(path, low_memory=False, encoding="utf-8")
        except Exception as exc2:                                 # noqa: BLE001
            log.error("Failed to read %s: %s", path, exc2)
            return None


def _find_comorbidity_csv(year_dir: Path) -> Path | None:
    """Locate a long-format comorbidity table in a year subdir.

    NTDB renames this file across years — common forms:
      PUF_PREEXISTINGCONDITION.csv
      PUF_PRE_EXISTING_CONDITION.csv
      RFPREEXISTINGCONDITION.csv
      PUF_COMORBIDITY.csv
    """
    candidates = (
        "PUF_PREEXISTINGCONDITION.csv", "PUF_PRE_EXISTING_CONDITION.csv",
        "PUF_PRE-EXISTING_CONDITION.csv",
        "RFPREEXISTINGCONDITION.csv", "RF_PREEXISTINGCONDITION.csv",
        "PUF_COMORBIDITY.csv", "RF_COMORBIDITY.csv",
    )
    for name in candidates:
        for p in (year_dir / name, year_dir / name.lower()):
            if p.exists():
                return p
    # Fallback — glob anything that matches *PRE*EXIST* or *COMORBID*
    for p in year_dir.iterdir():
        if not p.is_file():
            continue
        nl = p.name.lower()
        if (nl.endswith(".csv") and
                ("preexist" in nl or "pre_exist" in nl or "pre-exist" in nl or
                 "comorbid" in nl)):
            return p
    return None


def load_year(
    year: int,
    year_dir: Path,
    ntdb_tables: dict[str, str],
    inc_key_col: str = "INC_KEY",
) -> pd.DataFrame | None:
    """Load one admission year: PUF_TRAUMA + Ecode + NISS + Barell + comorbidities."""
    trauma = _read_csv_safe(year_dir / ntdb_tables["trauma"])
    if trauma is None:
        return None

    trauma = harmonise_year(trauma, year)
    ecode_lookup = _read_csv_safe(year_dir / ntdb_tables["ecode_lookup"])
    trauma = join_ecode_lookup(trauma, ecode_lookup)
    aisdiag = _read_csv_safe(year_dir / ntdb_tables["aisdiagnosis"])
    trauma = derive_niss(trauma, aisdiag, inc_key_col=inc_key_col)

    # Round 50 FIX: Barell-style injury features must come from the
    # ICD-10-CM diagnosis table (PUF_ICDDIAGNOSIS), NOT PUF_AISDIAGNOSIS.
    # PUF_AISDIAGNOSIS carries AIS predot codes (AISPreDot, e.g. 140202),
    # which are an entirely different coding system from ICD-10-CM.  Feeding
    # it to compute_barell_features (which matches ICD-10-CM S/T prefixes)
    # produced all-zero Barell columns for every patient.  The ICD-10-CM
    # codes live in PUF_ICDDIAGNOSIS (ICDDIAGNOSISCODE, e.g. S06.5X9A).
    # NISS still uses aisdiag above — that table genuinely holds AIS severities.
    icddiag = _read_csv_safe(year_dir / ntdb_tables["icddiagnosis"])
    from .barell import merge_barell_features, merge_icd_code_list
    trauma = merge_barell_features(trauma, icddiag, inc_key_col=inc_key_col)
    # Faithful Doshi FFNN input: per-patient raw ICD-10 code list (one compact
    # string column; the doshi_ffnn model vectorises it to a multi-hot itself).
    trauma = merge_icd_code_list(trauma, icddiag, inc_key_col=inc_key_col)

    # Round 29: comorbidity features (Tran 2022 S1 Table set, 18 binary
    # flags).  NTDB stores these differently per year — sometimes as wide
    # columns on PUF_TRAUMA, sometimes as long-format
    # PUF_PREEXISTINGCONDITION.csv.  The extractor tries both; missing
    # comorbidities get filled with 0.
    from .comorbidities import merge_comorbidity_features
    comorbidity_path = _find_comorbidity_csv(year_dir)
    comorbidity_long = (
        _read_csv_safe(comorbidity_path) if comorbidity_path else None
    )
    if comorbidity_path:
        log.info("Loaded comorbidity table: %s", comorbidity_path.name)
    else:
        log.info(
            "No long-format comorbidity table found in %s — "
            "will rely on wide columns of PUF_TRAUMA only.",
            year_dir,
        )
    trauma = merge_comorbidity_features(
        trauma, comorbidity_long, inc_key_col=inc_key_col,
    )

    log.info("Loaded AY %d: %d rows × %d columns", year, len(trauma), trauma.shape[1])
    return trauma


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------
def warn_missing_predictors(
    df: pd.DataFrame,
    year: int,
    expected_cols: Iterable[str],
) -> list[str]:
    """Round 51: log a WARNING for every expected predictor column that is
    NOT present in this admission year's frame (checked AFTER load_year, so
    derived columns — NISS, BARELL_*/INJ_*, aliased ISS/GCSTOTAL/… — are
    already in place).

    This is a diagnostic, not a hard failure: a column may be legitimately
    absent in some years (e.g. NTDB dropped EMS physiology after 2020), and
    seeing the warning tells you whether to adjust the catalogue / phase
    lists or accept the per-year missingness.  Returns the missing list.
    """
    present_lower = {c.lower() for c in df.columns}
    missing = sorted({c for c in expected_cols if c.lower() not in present_lower})
    if missing:
        log.warning(
            "AY %d availability check: %d expected predictor column(s) MISSING "
            "after load — %s. (If a column should exist, check the NTDB CSV / "
            "alias map; if it is legitimately year-specific, this is expected.)",
            year, len(missing), missing,
        )
    else:
        log.info(
            "AY %d availability check: all %d expected predictor columns present.",
            year, len(list(expected_cols)),
        )
    return missing


def build_unified_dataset(
    catalogue: Catalogue,
    ntdb_root: Path,
    ntdb_year_subdirs: dict[int, str],
    ntdb_tables: dict[str, str],
    years: Iterable[int] | None = None,
    phase_cutoff: str = "In-hospital (a posteriori)",
    registries: Iterable[str] | None = None,
    anonymize: bool = False,
    output_path: Path | None = None,
    inc_key_col: str = "INC_KEY",
) -> pd.DataFrame:
    """Build the unified multi-year NTDB dataset.

    Parameters
    ----------
    catalogue         : Catalogue instance (already loaded from sheet 6)
    ntdb_root         : Path to the root of the PUF data (contains "PUF AY 2019" etc.)
    ntdb_year_subdirs : {year: relative subdir_with_/CSV} mapping
    ntdb_tables       : {logical_name: csv_filename}
    years             : subset of years to load (defaults to all in ntdb_year_subdirs)
    phase_cutoff      : max phase for the whitelist
    registries        : registries whose variables are always kept
    anonymize         : if True, drop INC_KEY from the final dataframe
    output_path       : optional parquet path to save the result

    Returns
    -------
    Unified DataFrame with a `__admission_year` column added.
    """
    years_list = list(years) if years is not None else sorted(ntdb_year_subdirs.keys())

    # Round 51: the full predictor union we expect to be available, used to
    # warn (per year) about any column that did not materialise after load.
    try:
        expected_predictors = catalogue.variables_for(
            phase_cutoff="On-scene + ED arrival + In-hospital",
            include_target_derivers=False,
            include_injury_features=True,
        )
    except Exception as exc:  # noqa: BLE001 — never let the diagnostic break a build
        log.warning("Could not compute expected-predictor list for availability "
                    "check: %s", exc)
        expected_predictors = []

    frames: list[pd.DataFrame] = []
    for year in years_list:
        subdir = ntdb_year_subdirs.get(year)
        if subdir is None:
            log.warning("Year %d not in ntdb_year_subdirs; skipping", year)
            continue
        year_dir = ntdb_root / subdir
        df_year = load_year(year, year_dir, ntdb_tables, inc_key_col=inc_key_col)
        if df_year is None:
            log.warning("No data for AY %d", year)
            continue
        if expected_predictors:
            warn_missing_predictors(df_year, year, expected_predictors)
        frames.append(df_year)

    if not frames:
        raise RuntimeError("No admission years could be loaded; check paths and CSV names.")

    # Concatenate, letting pandas align columns (missing -> NaN)
    unified = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Merged %d years into shape %s", len(frames), unified.shape)

    # Round 35: cross-year case-variant cleanup.
    # Even with per-year dedup in load_year, concat across years can
    # produce duplicates if year A normalised AgeYears -> AGEYEARS while
    # year B's CSV happens to have a third casing the alias resolver
    # didn't anticipate.  Run a final dedup pass on the unified frame.
    unified = _dedupe_case_variants(unified, context="unified")

    # Apply whitelist
    unified = apply_whitelist(
        unified,
        catalogue=catalogue,
        phase_cutoff=phase_cutoff,
        registries=registries,
    )

    if anonymize and inc_key_col in unified.columns:
        unified = unified.drop(columns=[inc_key_col])
        log.info("Anonymised: dropped %s", inc_key_col)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Round 13: coerce mixed-type object columns to a clean dtype before
        # writing.  pyarrow refuses to write an `object` column that mixes
        # strings with NaN (floats) — it raises
        # "Expected bytes, got a 'float' object" mid-write.  Several NTDB
        # categorical columns (e.g. TEACHINGSTATUS, HOSPITALTYPE) hit this
        # because most rows are strings but a few hospitals leave them blank.
        # We convert those columns to pandas' string dtype, which preserves
        # NaN as pd.NA and serializes cleanly to parquet.
        n_coerced = 0
        for col in unified.columns:
            if unified[col].dtype != "object":
                continue
            sample = unified[col].dropna()
            if sample.empty:
                continue
            # If the non-null values are mostly strings, coerce to string dtype
            n_str = sum(isinstance(v, str) for v in sample.head(1000))
            if n_str > 0:
                try:
                    unified[col] = unified[col].astype("string")
                    n_coerced += 1
                except Exception as exc:
                    log.warning(
                        "Cannot coerce %s to string dtype: %s. "
                        "pyarrow may fail to write this column.",
                        col, exc,
                    )
        if n_coerced > 0:
            log.info("Coerced %d object columns to pandas string dtype "
                     "before parquet write", n_coerced)
        unified.to_parquet(output_path, index=False)
        log.info("Saved unified dataset to %s (%d rows, %d cols)",
                 output_path, *unified.shape)

    return unified
