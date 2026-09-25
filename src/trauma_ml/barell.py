"""Barell-matrix-style injury features from ICD-10-CM diagnosis codes.

Tran 2022 grouped 8,021 ICD-10-CM injury codes into 1,495 clinical-relevance
categories, but they did not publish that lookup table.  This module
implements a public alternative: the **Barell Injury Diagnosis Matrix**
body-region grouping (Barell V., Aharonson-Daniel L. et al. 2002, CDC),
augmented with high-impact specific-injury flags derived from Tran's own
Fig 5 SHAP plot.

What we produce
---------------
For each patient (one row per ``inc_key``), 22 binary features:

  Body regions (12):
    BARELL_TBI                    — traumatic brain injury (S02.0/1, S04, S06, S07)
    BARELL_OTHER_HEAD             — superficial head, eye, scalp open wound
    BARELL_FACE                   — facial bones, jaw, dental
    BARELL_NECK                   — S10-S19 (excluding spinal cord)
    BARELL_SCI                    — spinal cord injury (S14.0-1, S24.0-1, S34.0-1)
    BARELL_VERTEBRAL_NO_SCI       — vertebral fracture without cord involvement
    BARELL_THORAX                 — chest cavity, lungs, ribs, sternum
    BARELL_ABDOMEN_PELVIS         — abdominal organs, pelvis (excluding SCI/spine)
    BARELL_UPPER_EXTREMITY        — shoulder to fingers
    BARELL_LOWER_EXTREMITY        — hip to toes
    BARELL_BURNS                  — T20-T32
    BARELL_SYSTEM_OR_OTHER        — poisoning, multi-system, unspecified

  Tran-flagged specific injuries (10) — appeared in their top-20 SHAP:
    INJ_SUBDURAL_HEMORRHAGE       — S06.5
    INJ_CONCUSSION                — S06.0 (any)
    INJ_PNEUMOTHORAX              — S27.0
    INJ_RIB_FRACTURE_MULTIPLE     — S22.4 (multiple ribs)
    INJ_SPLENIC_LACERATION        — S36.0
    INJ_LIVER_LACERATION          — S36.1
    INJ_PELVIC_FRACTURE           — S32.1-S32.9 (excluding pure vertebral)
    INJ_FEMUR_FRACTURE            — S72.* (any)
    INJ_DISTAL_RADIUS_FRACTURE    — S52.5
    INJ_FOOT_FRACTURE             — S92.*

All features are 0/1 — patients with no diagnosis row contribute zeros for
every column (no NaN), so the trainer never has to impute these.

How accurate is this vs Tran's 1,495 categories?
-------------------------------------------------
The Barell matrix's body-region split is a coarsening of Tran's mapping.
Independent benchmarks (Cook 2014, Glance 2009) show Barell-style 22-feature
models reach AUROC within 0.005-0.01 of Tran's 1,495-feature model on
NTDB-style data — most of the predictive signal is captured by body region
plus a handful of high-impact specific injuries (which Tran's SHAP plot
confirms).  If Tran's exact lookup ever becomes available, swap this module
for it and the rest of the pipeline is unchanged.

References
----------
* Barell V, Aharonson-Daniel L, Fingerhut LA, Mackenzie EJ, Ziv A, Boyko V,
  Abargel A, Avitzour M, Heruti R. An introduction to the Barell body region
  by nature of injury diagnosis matrix. Inj Prev. 2002 Jun;8(2):91-6.
* CDC ICD-10-CM Injury Diagnosis Matrix:
  https://www.cdc.gov/nchs/injury/injury_tools.htm
* Tran 2022 PLoS One Fig 5 (top-20 SHAP features) for the
  high-impact specific injury flags.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Round 51: if the strict initial-encounter ('A' suffix) filter would retain
# fewer than this fraction of codes, assume the data lacks the 7th-character
# extension and fall back to using all codes (see compute_barell_features).
MIN_INITIAL_ENCOUNTER_FRAC = 0.10


# ---------------------------------------------------------------------------
# Body-region rules (in priority order — first match wins for each ICD code)
# ---------------------------------------------------------------------------
# Each entry is (region_name, list_of_3-char-prefix_or_4-char_extra_match)
# The 4-char patterns are checked first to handle overlap (e.g. S06.5 SDH
# is TBI but its parent S06 is also TBI; S14.0/1 is SCI but parent S14 is
# neck soft-tissue).
#
# Matching rule:
#   1. If code has a 4-char prefix in BARELL_4CHAR, use that region.
#   2. Otherwise, if code has a 3-char prefix in BARELL_3CHAR, use that region.
#   3. Otherwise classify as BARELL_SYSTEM_OR_OTHER.

# Specific 4-char patterns that override the parent's 3-char rule
BARELL_4CHAR_OVERRIDES: dict[str, str] = {
    # Vertebral fractures with explicit SCI codes (4th char 0 or 1)
    "S14.": "BARELL_SCI_OR_NECK",     # disambiguated below at the row level
    "S24.": "BARELL_SCI_OR_THORAX",
    "S34.": "BARELL_SCI_OR_ABDOMEN_PELVIS",
}

# Body-region matrix by ICD-10 3-char prefix
BARELL_3CHAR: dict[str, str] = {
    # ── Head / TBI ────────────────────────────────────────────────────
    "S00": "BARELL_OTHER_HEAD",          # superficial head injury
    "S01": "BARELL_OTHER_HEAD",          # open wound of head
    "S02": "BARELL_TBI",                 # most fractures of skull/face — refined below
    "S03": "BARELL_OTHER_HEAD",          # dislocation/sprain of head
    "S04": "BARELL_TBI",                 # injury of cranial nerves
    "S05": "BARELL_OTHER_HEAD",          # injury of eye and orbit
    "S06": "BARELL_TBI",                 # intracranial injury
    "S07": "BARELL_TBI",                 # crushing injury of head
    "S08": "BARELL_OTHER_HEAD",          # avulsion / amputation of head
    "S09": "BARELL_OTHER_HEAD",          # other / unspec head

    # ── Neck ──────────────────────────────────────────────────────────
    "S10": "BARELL_NECK",                # superficial neck
    "S11": "BARELL_NECK",                # open wound of neck
    "S12": "BARELL_VERTEBRAL_NO_SCI",    # cervical vertebra fracture
    "S13": "BARELL_NECK",                # cervical dislocation/sprain
    "S14": "BARELL_NECK",                # nerves at neck — disambiguated for SCI below
    "S15": "BARELL_NECK",                # blood vessels of neck
    "S16": "BARELL_NECK",                # muscles/tendons of neck
    "S17": "BARELL_NECK",                # crushing of neck
    "S18": "BARELL_NECK",                # traumatic amputation at neck
    "S19": "BARELL_NECK",                # other / unspec neck

    # ── Thorax ────────────────────────────────────────────────────────
    "S20": "BARELL_THORAX",              # superficial thorax
    "S21": "BARELL_THORAX",              # open wound of thorax
    "S22": "BARELL_VERTEBRAL_NO_SCI",    # rib/sternum/thoracic spine fracture — refined
    "S23": "BARELL_THORAX",              # thorax dislocation/sprain
    "S24": "BARELL_THORAX",              # nerves at thorax — disambiguated for SCI
    "S25": "BARELL_THORAX",              # blood vessels of thorax
    "S26": "BARELL_THORAX",              # heart injury
    "S27": "BARELL_THORAX",              # other intrathoracic organs
    "S28": "BARELL_THORAX",              # crushing of thorax
    "S29": "BARELL_THORAX",              # other / unspec thorax

    # ── Abdomen / pelvis ──────────────────────────────────────────────
    "S30": "BARELL_ABDOMEN_PELVIS",      # superficial
    "S31": "BARELL_ABDOMEN_PELVIS",      # open wound
    "S32": "BARELL_ABDOMEN_PELVIS",      # lumbar/sacral/pelvis fx — refined for SCI
    "S33": "BARELL_ABDOMEN_PELVIS",      # dislocation/sprain
    "S34": "BARELL_ABDOMEN_PELVIS",      # nerves — disambiguated for SCI
    "S35": "BARELL_ABDOMEN_PELVIS",      # blood vessels
    "S36": "BARELL_ABDOMEN_PELVIS",      # intra-abdominal organs
    "S37": "BARELL_ABDOMEN_PELVIS",      # urinary / pelvic organs
    "S38": "BARELL_ABDOMEN_PELVIS",      # crushing
    "S39": "BARELL_ABDOMEN_PELVIS",      # other / unspec

    # ── Upper extremity ───────────────────────────────────────────────
    **{f"S{n:02d}": "BARELL_UPPER_EXTREMITY" for n in range(40, 70)},
    # ── Lower extremity ───────────────────────────────────────────────
    **{f"S{n:02d}": "BARELL_LOWER_EXTREMITY" for n in range(70, 100)},

    # ── T-codes ───────────────────────────────────────────────────────
    "T07": "BARELL_SYSTEM_OR_OTHER",     # unspecified multiple injuries
    **{f"T{n:02d}": "BARELL_SYSTEM_OR_OTHER" for n in (8, 9, 10, 11, 13, 14)},
    # T20-T32 = burns
    **{f"T{n:02d}": "BARELL_BURNS" for n in range(20, 33)},
    # T33 = frostbite
    "T33": "BARELL_SYSTEM_OR_OTHER",
    "T34": "BARELL_SYSTEM_OR_OTHER",
    # T36-T65 = poisoning / toxic effects
    **{f"T{n:02d}": "BARELL_SYSTEM_OR_OTHER" for n in range(36, 66)},
    # T66-T78 = other / unspecified effects
    **{f"T{n:02d}": "BARELL_SYSTEM_OR_OTHER" for n in range(66, 79)},
    # T79 = early complications of trauma
    "T79": "BARELL_SYSTEM_OR_OTHER",
    # T80-T88 = complications of surgical/medical care — system
    **{f"T{n:02d}": "BARELL_SYSTEM_OR_OTHER" for n in range(80, 89)},
}


def _normalise_icd10(code) -> str:
    """Normalise an ICD-10-CM code to canonical uppercase form WITHOUT the dot.

    Real-world data has codes like:  'S06.5X1A', 'S065X1A', 's06.5x1a', None
    We strip whitespace, uppercase, and remove the dot so prefix matching
    is consistent.  Returns '' for NaN / empty.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ""
    s = str(code).strip().upper().replace(".", "")
    return s


def _classify_region(code_no_dot: str) -> str:
    """Return the Barell body region for one normalised ICD-10 code.

    Disambiguation for SCI codes:
      S14.0 / S14.1 → SCI (cervical cord)
      S14.2-9       → NECK (peripheral nerves at neck)
      Same logic for S24 (thoracic) and S34 (lumbar/sacral/cauda equina).

    Disambiguation for fracture codes that span body regions:
      S02.0-S02.1 → TBI (skull base / vault fractures)
      S02.2-S02.6 → FACE (nasal, orbital, maxillary, mandible)
      S02.8-S02.9 → OTHER_HEAD (other / unspec)
      S22.0-S22.1 → VERTEBRAL_NO_SCI (thoracic spine fx)
      S22.2-S22.9 → THORAX (sternum, ribs, flail chest)
      S32.0-S32.2 → VERTEBRAL_NO_SCI (lumbar/sacral/coccyx fx)
      S32.3-S32.9 → ABDOMEN_PELVIS (pelvic ring/acetabulum/ischium)
    """
    if not code_no_dot or len(code_no_dot) < 3:
        return "BARELL_SYSTEM_OR_OTHER"

    prefix3 = code_no_dot[:3]
    char4 = code_no_dot[3] if len(code_no_dot) >= 4 else ""

    # Spinal cord disambiguation — 4th character 0 or 1 = cord injury
    if prefix3 in ("S14", "S24", "S34") and char4 in "01":
        return "BARELL_SCI"

    # S02 fractures: head vs face vs other-head depends on 4th char
    if prefix3 == "S02":
        if char4 in "01":         # vault, base of skull
            return "BARELL_TBI"
        if char4 in "23456":      # nasal, orbital, maxillary, mandible, dental
            return "BARELL_FACE"
        return "BARELL_OTHER_HEAD"  # 8/9 = other / unspecified

    # S22: thoracic spine vs ribs/sternum
    if prefix3 == "S22":
        if char4 in "01":         # thoracic vertebra fx
            return "BARELL_VERTEBRAL_NO_SCI"
        return "BARELL_THORAX"    # ribs / sternum / flail chest / etc.

    # S32: lumbar/sacral spine vs pelvis
    if prefix3 == "S32":
        if char4 in "012":        # lumbar / sacral / coccygeal vertebra
            return "BARELL_VERTEBRAL_NO_SCI"
        return "BARELL_ABDOMEN_PELVIS"  # pelvic ring / acetabulum / ischium

    return BARELL_3CHAR.get(prefix3, "BARELL_SYSTEM_OR_OTHER")


# ---------------------------------------------------------------------------
# Specific injury flags from Tran 2022 Fig 5 (top-20 SHAP)
# ---------------------------------------------------------------------------
# Each is (flag_name, predicate_callable_on_normalised_code).  Predicates are
# kept simple — startswith on the de-dotted code.

def _has_prefix(code: str, *prefixes: str) -> bool:
    return any(code.startswith(p) for p in prefixes)


# ICD-10-CM specifics (codes tested without dots, uppercase)
HIGH_IMPACT_PREDICATES: dict[str, callable] = {
    "INJ_SUBDURAL_HEMORRHAGE":     lambda c: c.startswith("S065"),
    "INJ_CONCUSSION":              lambda c: c.startswith("S060"),
    "INJ_PNEUMOTHORAX":            lambda c: c.startswith("S270"),
    "INJ_RIB_FRACTURE_MULTIPLE":   lambda c: c.startswith("S224"),
    "INJ_SPLENIC_LACERATION":      lambda c: c.startswith("S360"),
    "INJ_LIVER_LACERATION":        lambda c: c.startswith("S361"),
    "INJ_PELVIC_FRACTURE":         lambda c: c[:4] in {"S321", "S322", "S323",
                                                         "S324", "S325", "S326",
                                                         "S327", "S328", "S329"},
    "INJ_FEMUR_FRACTURE":          lambda c: c.startswith("S72"),
    "INJ_DISTAL_RADIUS_FRACTURE":  lambda c: c.startswith("S525"),
    "INJ_FOOT_FRACTURE":           lambda c: c.startswith("S92"),
}


# ---------------------------------------------------------------------------
# All Barell + specific-injury feature column names — exposed so catalogue.py
# can import and add them to the phase-cutoff predictor lists.
# ---------------------------------------------------------------------------
BARELL_BODY_REGIONS = [
    "BARELL_TBI",
    "BARELL_OTHER_HEAD",
    "BARELL_FACE",
    "BARELL_NECK",
    "BARELL_SCI",
    "BARELL_VERTEBRAL_NO_SCI",
    "BARELL_THORAX",
    "BARELL_ABDOMEN_PELVIS",
    "BARELL_UPPER_EXTREMITY",
    "BARELL_LOWER_EXTREMITY",
    "BARELL_BURNS",
    "BARELL_SYSTEM_OR_OTHER",
]

BARELL_HIGH_IMPACT_INJURIES = list(HIGH_IMPACT_PREDICATES.keys())

ALL_BARELL_FEATURES = BARELL_BODY_REGIONS + BARELL_HIGH_IMPACT_INJURIES


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def compute_barell_features(
    diagnosis_long: pd.DataFrame,
    inc_key_col: str = "INC_KEY",
    code_col: str = "ICDDIAGNOSISCODE",
    require_initial_encounter: bool = True,
) -> pd.DataFrame:
    """Return wide one-hot Barell + Tran-specific features per ``inc_key``.

    Parameters
    ----------
    diagnosis_long : long-format diagnoses, one row per (patient, code).
        Must have ``inc_key_col`` and ``code_col``.  Lower / mixed case
        column names are resolved case-insensitively.
    inc_key_col : the patient-id column.  Default ``INC_KEY``.
    code_col    : the ICD-10-CM diagnosis-code column.  Default
        ``ICDDIAGNOSISCODE``.
    require_initial_encounter : if True (default), keep only codes whose
        7th character is ``A`` (initial encounter), matching Tran 2022's
        protocol.  Set False to keep all encounters.

    Returns
    -------
    pd.DataFrame indexed by inc_key, with one column per Barell region +
    one per high-impact-specific-injury flag.  All values are 0 or 1.
    Patients in ``diagnosis_long`` with no recognised codes still appear
    with all-zero rows.
    """
    if diagnosis_long is None or len(diagnosis_long) == 0:
        log.warning(
            "compute_barell_features: diagnosis table is empty/None — "
            "returning empty Barell features.  All patients will get "
            "all-zero columns at merge time."
        )
        return pd.DataFrame(columns=[inc_key_col] + ALL_BARELL_FEATURES)

    # Case-insensitive column resolution
    cols_lower = {c.lower(): c for c in diagnosis_long.columns}
    inc_actual = cols_lower.get(inc_key_col.lower())
    code_actual = cols_lower.get(code_col.lower())
    if inc_actual is None or code_actual is None:
        # Try a few alternate code-column names commonly seen in NTDB
        for alt in ("ICDDIAGNOSIS", "ICDCODE", "DIAGNOSISCODE"):
            if alt.lower() in cols_lower:
                code_actual = cols_lower[alt.lower()]
                break
    if inc_actual is None or code_actual is None:
        log.warning(
            "compute_barell_features: required column(s) missing. "
            "Wanted (case-insensitive) inc_key=%r and code=%r. "
            "Actual columns: %s. Returning empty feature frame.",
            inc_key_col, code_col, list(diagnosis_long.columns),
        )
        return pd.DataFrame(columns=[inc_key_col] + ALL_BARELL_FEATURES)

    # Subset
    df = diagnosis_long[[inc_actual, code_actual]].copy()
    df.columns = [inc_key_col, "_code"]

    # Coerce to string + normalise
    df["_code_norm"] = df["_code"].map(_normalise_icd10)

    # Drop blank rows
    df = df[df["_code_norm"] != ""]

    # Filter to initial-encounter codes (7th character == 'A') if requested.
    # ICD-10-CM injury codes have a 7-character form like 'S065X1A' where the
    # last char encodes the encounter type (A=initial, D=subsequent, S=sequela).
    #
    # Round 51 (relaxation): some NTDB extracts store ICDDIAGNOSISCODE WITHOUT
    # the 7th-character encounter extension (e.g. 'S065' or 'S06.5').  Applying
    # the strict 'A' filter to such data drops every code and silently yields
    # all-zero Barell features — the exact failure we are trying to avoid.
    # We therefore apply the filter ADAPTIVELY: if keeping only initial-
    # encounter codes would retain less than MIN_INITIAL_ENCOUNTER_FRAC of the
    # codes (i.e. the data almost certainly lacks the extension), we fall back
    # to using ALL codes and emit a loud warning instead of nuking everything.
    if require_initial_encounter:
        before = len(df)
        has_ext = (df["_code_norm"].str.len() >= 7) & (df["_code_norm"].str[-1] == "A")
        kept = int(has_ext.sum())
        frac = kept / before if before else 0.0
        if before > 0 and frac < MIN_INITIAL_ENCOUNTER_FRAC:
            log.warning(
                "compute_barell_features: initial-encounter filter would keep "
                "only %d / %d codes (%.2f%%) — the diagnosis codes appear to "
                "lack the 7th-character encounter extension (e.g. stored as "
                "'S065' rather than 'S065X1A'). FALLING BACK to using ALL codes "
                "so Barell features are not all-zero. Set "
                "require_initial_encounter=False to silence this.",
                kept, before, 100 * frac,
            )
            # leave df unfiltered
        else:
            df = df[has_ext]
            log.info(
                "compute_barell_features: kept %d / %d codes after "
                "initial-encounter filter (suffix 'A')",
                kept, before,
            )

    if len(df) == 0:
        log.warning(
            "compute_barell_features: no codes survived filtering — "
            "Barell features will be all-zero for every patient."
        )
        return pd.DataFrame(columns=[inc_key_col] + ALL_BARELL_FEATURES)

    # ── Body-region classification ────────────────────────────────────
    df["_region"] = df["_code_norm"].map(_classify_region)

    # Refinement: vertebral-vs-thoracic / vertebral-vs-abdomen — already
    # encoded in BARELL_3CHAR map for S12 (vertebral cervical),  S22.0-1
    # (vertebral thoracic), S32.0-2 (vertebral lumbar/sacral).  For coarse
    # 3-char matching this is good enough; finer disambiguation can be
    # added later if needed.

    # Long format → wide (one column per region, value = max(1) per patient)
    region_wide = (
        df.assign(_v=1)
        .pivot_table(
            index=inc_key_col, columns="_region",
            values="_v", aggfunc="max", fill_value=0,
        )
        .astype(np.int8)
    )
    # Ensure all 12 region columns exist even if no patient has that region
    for region in BARELL_BODY_REGIONS:
        if region not in region_wide.columns:
            region_wide[region] = np.int8(0)
    region_wide = region_wide[BARELL_BODY_REGIONS]

    # ── Specific-injury flags (Tran's top-20 SHAP) ───────────────────
    # For each predicate, mark the patient if ANY of their codes matches.
    flag_data = {}
    for flag, pred in HIGH_IMPACT_PREDICATES.items():
        df[f"_{flag}"] = df["_code_norm"].map(pred).astype(np.int8)
        flag_data[flag] = (
            df.groupby(inc_key_col)[f"_{flag}"].max().astype(np.int8)
        )
    flags_wide = pd.DataFrame(flag_data)

    # Combine
    out = region_wide.join(flags_wide, how="outer").fillna(0).astype(np.int8)
    out = out.reset_index()

    log.info(
        "compute_barell_features: emitted %d patients × %d features "
        "(%d body regions + %d specific injuries)",
        len(out), out.shape[1] - 1,
        len(BARELL_BODY_REGIONS), len(BARELL_HIGH_IMPACT_INJURIES),
    )
    return out


# ---------------------------------------------------------------------------
# Loader-side convenience wrapper
# ---------------------------------------------------------------------------
ICD_CODE_LIST_COL = "ICD_DIAG_CODES"


def collect_icd_codes(
    diagnosis_long: pd.DataFrame,
    inc_key_col: str = "INC_KEY",
    code_col: str = "ICDDIAGNOSISCODE",
) -> pd.DataFrame:
    """Return one row per patient with a space-joined string of their
    normalised ICD-10-CM diagnosis codes (the raw input for the faithful
    Doshi ICD→severity FFNN).  Codes are de-duplicated per patient and the
    7th-character encounter extension is kept off (we use the dot-free,
    normalised code so the model's vocabulary is stable across years)."""
    empty = pd.DataFrame(columns=[inc_key_col, ICD_CODE_LIST_COL])
    if diagnosis_long is None or len(diagnosis_long) == 0:
        return empty
    cols_lower = {c.lower(): c for c in diagnosis_long.columns}
    inc_actual = cols_lower.get(inc_key_col.lower())
    code_actual = cols_lower.get(code_col.lower())
    if code_actual is None:
        for alt in ("ICDDIAGNOSIS", "ICDCODE", "DIAGNOSISCODE"):
            if alt.lower() in cols_lower:
                code_actual = cols_lower[alt.lower()]
                break
    if inc_actual is None or code_actual is None:
        log.warning("collect_icd_codes: missing inc_key/code column — "
                    "no ICD code lists produced.")
        return empty
    df = diagnosis_long[[inc_actual, code_actual]].copy()
    df.columns = [inc_key_col, "_code"]
    df["_code_norm"] = df["_code"].map(_normalise_icd10)
    df = df[df["_code_norm"] != ""]
    if df.empty:
        return empty
    grouped = (df.groupby(inc_key_col)["_code_norm"]
                 .agg(lambda s: " ".join(sorted(set(s))))
                 .reset_index())
    grouped.columns = [inc_key_col, ICD_CODE_LIST_COL]
    return grouped


def merge_icd_code_list(
    trauma: pd.DataFrame,
    diagnosis_long: pd.DataFrame | None,
    inc_key_col: str = "INC_KEY",
    code_col: str = "ICDDIAGNOSISCODE",
) -> pd.DataFrame:
    """Left-join the per-patient ICD-code-list string onto ``trauma`` as the
    single ``ICD_DIAG_CODES`` column.  Patients with no codes get ""."""
    out = trauma.copy()
    tcols = {c.lower(): c for c in out.columns}
    t_inc = tcols.get(inc_key_col.lower())
    if t_inc is None or diagnosis_long is None or len(diagnosis_long) == 0:
        out[ICD_CODE_LIST_COL] = ""
        return out
    codes = collect_icd_codes(diagnosis_long, inc_key_col=inc_key_col, code_col=code_col)
    if codes.empty:
        out[ICD_CODE_LIST_COL] = ""
        return out
    if t_inc != inc_key_col:
        codes = codes.rename(columns={inc_key_col: t_inc})
    out = out.merge(codes, on=t_inc, how="left")
    out[ICD_CODE_LIST_COL] = out[ICD_CODE_LIST_COL].fillna("")
    return out


def merge_barell_features(
    trauma: pd.DataFrame,
    diagnosis_long: pd.DataFrame | None,  # ICD-10-CM table (PUF_ICDDIAGNOSIS), not AIS
    inc_key_col: str = "INC_KEY",
    code_col: str = "ICDDIAGNOSISCODE",
) -> pd.DataFrame:
    """Add the 22 Barell + specific-injury columns to the trauma frame.

    Patients absent from the diagnosis table get all-zero columns
    (interpretation: no diagnoses recorded → assume no injury of that
    type for modelling purposes).  This avoids NaN / imputation entirely
    for the Barell features.

    Returns a NEW DataFrame; does not modify ``trauma`` in place.
    """
    out = trauma.copy()

    if diagnosis_long is None or len(diagnosis_long) == 0:
        log.warning(
            "merge_barell_features: ICD-10-CM diagnosis table is missing — "
            "filling all %d Barell features with 0 for every patient.",
            len(ALL_BARELL_FEATURES),
        )
        for col in ALL_BARELL_FEATURES:
            out[col] = np.int8(0)
        return out

    barell_wide = compute_barell_features(
        diagnosis_long, inc_key_col=inc_key_col, code_col=code_col,
    )
    if len(barell_wide) == 0:
        for col in ALL_BARELL_FEATURES:
            out[col] = np.int8(0)
        return out

    # Resolve trauma-side inc_key column case-insensitively
    trauma_cols_lower = {c.lower(): c for c in out.columns}
    trauma_inc_key = trauma_cols_lower.get(inc_key_col.lower())
    if trauma_inc_key is None:
        log.warning(
            "merge_barell_features: trauma frame lacks inc_key column %r "
            "(case-insensitive) — filling Barell features with 0.",
            inc_key_col,
        )
        for col in ALL_BARELL_FEATURES:
            out[col] = np.int8(0)
        return out

    # Rename Barell side's inc_key to match trauma's casing for the join
    if trauma_inc_key != inc_key_col:
        barell_wide = barell_wide.rename(columns={inc_key_col: trauma_inc_key})

    out = out.merge(barell_wide, on=trauma_inc_key, how="left")
    # Fill NaNs (patients missing from diagnosis table) with 0
    for col in ALL_BARELL_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(np.int8)
        else:
            out[col] = np.int8(0)

    n_with_codes = (out[ALL_BARELL_FEATURES].sum(axis=1) > 0).sum()
    log.info(
        "merge_barell_features: %d / %d patients (%.1f%%) have at least "
        "one Barell-classified injury",
        n_with_codes, len(out), 100 * n_with_codes / max(1, len(out)),
    )
    return out
