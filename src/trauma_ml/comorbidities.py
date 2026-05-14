"""Comorbidity feature extraction from NTDB PUF data.

NTDB PUF stores comorbidities in different shapes across admission years:

  Pattern A — wide flag columns on PUF_TRAUMA itself
              (column names like ``CHF``, ``COPD``, or longer aliases like
               ``CONGESTIVEHEARTFAILURE``).

  Pattern B — separate long-format file ``PUF_PREEXISTINGCONDITION.csv``
              (or ``RFPREEXISTINGCONDITION.csv``, varies by year), with
              one row per (inc_key, condition_string).

This module tries A first.  Anything not found in A is then sought in B.
Anything still missing is filled with 0 (assume condition absent).

Output: 18 binary columns matching the canonical names used by
``trauma_ml.catalogue`` and Tran 2022's S1 Table:

    COPD, CHF, MI, HYPERTENSION,
    PERIPHERALVASCULARDISEASE, ESRD, CIRRHOSIS,
    DIABETESMELLITUS, BLEEDINGDISORDER, DISSEMINATEDCANCER,
    ALCOHOLUSEDISORDER, MENTALPERSONALITYDISORDER,
    SUBSTANCEABUSEDISORDERDRUG, ATTENTIONDEFICITDISORDER,
    DEMENTIA, ADVANCEDDIRECTIVELIMITINGCARE,
    FUNCTIONALLYDEPENDENTHEALTHSTATUS, SMOKINGSTATUS

If a comorbidity is genuinely absent in the source year (e.g.
ATTENTIONDEFICITDISORDER added only in AY 2017+), its column will be all
zeros — this is correct and matches Tran 2022's treatment.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


CANONICAL_COMORBIDITIES = [
    "COPD", "CHF", "MI", "HYPERTENSION",
    "PERIPHERALVASCULARDISEASE", "ESRD", "CIRRHOSIS",
    "DIABETESMELLITUS", "BLEEDINGDISORDER", "DISSEMINATEDCANCER",
    "ALCOHOLUSEDISORDER", "MENTALPERSONALITYDISORDER",
    "SUBSTANCEABUSEDISORDERDRUG", "ATTENTIONDEFICITDISORDER",
    "DEMENTIA", "ADVANCEDDIRECTIVELIMITINGCARE",
    "FUNCTIONALLYDEPENDENTHEALTHSTATUS", "SMOKINGSTATUS",
]


# ---------------------------------------------------------------------------
# NTDB INTEGER-CODED comorbidity lookup
# ---------------------------------------------------------------------------
# The PUF_PREEXISTINGCONDITION table doesn't store free-text descriptions in
# AY 2017+ — it uses an integer code that maps to a row in the PUF data
# dictionary's "Co-Morbid Condition" table.  The mapping is published in
# the NTDB User Manual; the numeric→canonical map below is the merged
# AY 2019-2024 set.  Multiple integer codes can map to the same canonical
# (the OR of their flags is taken at merge time).
#
# Reference: ACS NTDB PUF User Manual,
# "PUF_PREEXISTINGCONDITION" table, "PreExistingConditionCode" field.
NTDB_INT_TO_CANONICAL: dict[int, str] = {
    1:  "ADVANCEDDIRECTIVELIMITINGCARE",
    2:  "ALCOHOLUSEDISORDER",
    # 3:  Angina Pectoris — no Tran-S1 equivalent, skipped
    4:  "BLEEDINGDISORDER",   # Anticoagulant Therapy → bleeds risk
    5:  "ATTENTIONDEFICITDISORDER",
    6:  "BLEEDINGDISORDER",   # Bleeding Disorder
    7:  "DISSEMINATEDCANCER", # Chemotherapy <30d → active cancer surrogate
    8:  "CIRRHOSIS",
    9:  "CHF",
    10: "COPD",
    11: "DISSEMINATEDCANCER", # Currently receiving chemo
    # 12: CVA — skipped (no Tran-S1 equivalent)
    13: "COPD",                # Currently requiring O2 → severe pulmonary
    14: "DEMENTIA",
    15: "DIABETESMELLITUS",
    16: "DISSEMINATEDCANCER",
    # 17: DVT/Thromboembolism — outcome, not pre-existing
    18: "SUBSTANCEABUSEDISORDERDRUG",
    19: "ESRD",
    20: "FUNCTIONALLYDEPENDENTHEALTHSTATUS",
    21: "MI",                  # History of MI
    # 22: History of Angina — skipped
    23: "HYPERTENSION",
    24: "MENTALPERSONALITYDISORDER",  # Major Psychiatric Illness
    25: "MENTALPERSONALITYDISORDER",
    26: "MI",                  # Recent MI
    # 27: Pacemaker — skipped
    28: "PERIPHERALVASCULARDISEASE",
    # 29: Pregnancy — skipped
    # 30: Prematurity — skipped
    # 31: Respirator Dependent — skipped
    32: "SMOKINGSTATUS",       # Smoking history
    # 33: Steroid Use — skipped
    34: "SUBSTANCEABUSEDISORDERDRUG",
    35: "SMOKINGSTATUS",       # Currently smoker
    # 36-38: Other Cardiac/Pulmonary/Other Comorbidity — too vague, skipped
}


# ---------------------------------------------------------------------------
# Keyword/alias patterns — applied case-insensitively, with substring match
# ---------------------------------------------------------------------------
# Order matters within each list: more specific patterns first.  The matcher
# returns the FIRST canonical that has any pattern matching the input string.

# Patterns that need a word-boundary (to avoid matching as substring of
# unrelated words — e.g. "MI" must not match "Mike"):
WORD_BOUNDARY_PATTERNS: dict[str, list[str]] = {
    "MI":   ["mi"],
    "CHF":  ["chf"],
    "COPD": ["copd"],
    "ESRD": ["esrd"],
    "DM":   ["dm"],   # alias for DIABETESMELLITUS — handled below
    "ADD":  ["add"],
    "ADHD": ["adhd"],
    "DNR":  ["dnr"],
    "HTN":  ["htn"],
}

# Substring patterns — match anywhere in the input.
SUBSTRING_PATTERNS: dict[str, list[str]] = {
    "COPD": ["chronic obstructive", "chronicobstructive", "obstructivepulmonary"],
    "CHF":  ["congestive heart", "congestiveheart", "heart failure", "heartfailure",
              "congheart"],
    "MI":   ["myocardial infarction", "myocardialinfarction", "history of mi",
              "prior mi"],
    "HYPERTENSION": ["hypertension", "high blood pressure"],
    "PERIPHERALVASCULARDISEASE": ["peripheral vascular", "peripheralvascular",
                                    "peripheral arterial", "peripheralarterial"],
    "ESRD": ["endstagerenal", "end stage renal", "end-stage renal",
              "dialysis", "renal failure", "renalfailure",
              "chronickidney", "chronic kidney"],
    "CIRRHOSIS": ["cirrhosis", "liver failure", "liverfailure",
                   "hepatic failure", "chronic liver", "chronicliver"],
    "DIABETESMELLITUS": ["diabetes", "diabetesmellitus"],
    "BLEEDINGDISORDER": ["bleeding disorder", "bleedingdisorder",
                          "coagulopathy", "anticoagulation",
                          "anticoagulant", "warfarin", "currentlyrequiringtherapy"],
    "DISSEMINATEDCANCER": ["disseminated cancer", "disseminatedcancer",
                            "metastatic", "metastasis"],
    "ALCOHOLUSEDISORDER": ["alcohol use disorder", "alcoholusedisorder",
                            "alcoholism", "alcohol abuse", "alcoholabuse"],
    "MENTALPERSONALITYDISORDER": ["mental disorder", "mentaldisorder",
                                    "personality disorder", "personalitydisorder",
                                    "psychiatric"],
    "SUBSTANCEABUSEDISORDERDRUG": ["substance abuse", "substanceabuse",
                                    "drug abuse", "drugabuse",
                                    "drug dependence", "drugdependence",
                                    "drug use disorder", "druguse"],
    "ATTENTIONDEFICITDISORDER": ["attention deficit", "attentiondeficit"],
    "DEMENTIA": ["dementia", "alzheimer"],
    "ADVANCEDDIRECTIVELIMITINGCARE": ["advanced directive", "advanceddirective",
                                        "advance directive", "advancedirective",
                                        "limit care", "limitingcare", "limiting care"],
    "FUNCTIONALLYDEPENDENTHEALTHSTATUS": ["functionally dependent",
                                            "functionallydependent",
                                            "functional dependence",
                                            "functionaldependence",
                                            "functional status",
                                            "functionalstatus"],
    "SMOKINGSTATUS": ["smoking", "current smoker", "currentsmoker",
                       "tobacco", "smoker"],
}

# Aliases that should map to a different canonical
ALIAS_TO_CANONICAL: dict[str, str] = {
    "DM":   "DIABETESMELLITUS",
    "ADD":  "ATTENTIONDEFICITDISORDER",
    "ADHD": "ATTENTIONDEFICITDISORDER",
    "DNR":  "ADVANCEDDIRECTIVELIMITINGCARE",
    "HTN":  "HYPERTENSION",
}


def _match_canonical(s: str) -> str | None:
    """Map a string (column name or condition description) to one of the
    18 canonical comorbidity names.  Returns None if no match.

    Case-insensitive, with both word-boundary and substring matching.
    """
    if not s or pd.isna(s):
        return None
    s_lower = str(s).lower()
    # Normalise punctuation/separators so 'pre_existing' == 'pre existing' etc.
    s_norm = re.sub(r"[_\-/]", " ", s_lower)
    s_norm = re.sub(r"\s+", " ", s_norm).strip()

    # First pass: word-boundary matches (for short tokens like CHF, MI, COPD)
    for canonical, patterns in WORD_BOUNDARY_PATTERNS.items():
        for p in patterns:
            if re.search(rf"\b{re.escape(p)}\b", s_norm):
                # Resolve aliases
                return ALIAS_TO_CANONICAL.get(canonical, canonical)

    # Second pass: substring matches
    for canonical, patterns in SUBSTRING_PATTERNS.items():
        for p in patterns:
            if p in s_norm:
                return canonical

    return None


# ---------------------------------------------------------------------------
# Pattern A: wide columns on PUF_TRAUMA
# ---------------------------------------------------------------------------
def _coerce_to_01(series: pd.Series) -> pd.Series:
    """NTDB flags arrive as 'Yes'/'No', 1/0/NaN, 'Y'/'N', etc.  Coerce to 0/1 int8."""
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce") > 0) \
            .fillna(False).astype(np.int8)
    s = series.astype(str).str.strip().str.upper()
    is_yes = s.isin({"1", "Y", "YES", "TRUE", "T", "1.0"})
    return is_yes.astype(np.int8)


def extract_from_wide(trauma_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Find comorbidity columns directly on the trauma frame.

    Returns dict {canonical_name: 0/1 series indexed like trauma_df}.

    When multiple columns map to the same canonical (e.g. ``HYPERTENSION``
    and ``HYPERTENSIONREQUIRINGMEDICATION``), prefers the column whose name
    exactly equals or starts with the canonical, otherwise the OR of all
    matching columns (so any positive match counts).
    """
    candidates_by_canon: dict[str, list[str]] = {}
    for col in trauma_df.columns:
        canonical = _match_canonical(col)
        if canonical is not None:
            candidates_by_canon.setdefault(canonical, []).append(col)

    found: dict[str, pd.Series] = {}
    for canonical, cols in candidates_by_canon.items():
        if len(cols) == 1:
            found[canonical] = _coerce_to_01(trauma_df[cols[0]])
        else:
            # Prefer exact-name match; otherwise OR all matching columns
            exact = [c for c in cols if c.upper() == canonical]
            if exact:
                found[canonical] = _coerce_to_01(trauma_df[exact[0]])
            else:
                # OR across all matching columns
                ored = pd.Series(0, index=trauma_df.index, dtype=np.int8)
                for c in cols:
                    ored = (ored | _coerce_to_01(trauma_df[c])).astype(np.int8)
                found[canonical] = ored
    return found


# ---------------------------------------------------------------------------
# Pattern B: long-format PUF_PREEXISTINGCONDITION-style table
# ---------------------------------------------------------------------------
def _detect_condition_column(df: pd.DataFrame, inc_actual: str) -> tuple[str | None, str]:
    """Heuristically pick the condition column AND figure out its dtype.

    Returns (column_name, kind) where kind is one of:
        'integer'   — the column holds NTDB integer codes (PUF_PREEXISTINGCONDITION)
        'text'      — free-text condition descriptions
        'unknown'   — neither pattern matched
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # PUF_PREEXISTINGCONDITION canonical name in AY 2017+
    for candidate in (
        "preexistingconditioncode", "pre_existing_condition_code",
        "preexistingcondition", "pmh", "comorbidity", "comorbidcode",
    ):
        if candidate in cols_lower:
            col = cols_lower[candidate]
            # Check if numeric or text
            sample = df[col].dropna().head(50)
            try:
                pd.to_numeric(sample, errors="raise")
                return col, "integer"
            except (ValueError, TypeError):
                return col, "text"

    # Older free-text formats
    for candidate in (
        "cdescr", "comorbidity_descr", "condition", "comorbid_name",
        "description", "conditionname", "precondition_name",
    ):
        if candidate in cols_lower:
            return cols_lower[candidate], "text"

    # Heuristic fallback: pick the only non-key column that looks plausible
    candidate_cols = [c for c in df.columns if c != inc_actual]
    # Prefer numeric columns first since PUF_PREEXISTINGCONDITION uses int codes
    for c in candidate_cols:
        sample = df[c].dropna().head(100)
        if len(sample) == 0:
            continue
        try:
            nums = pd.to_numeric(sample, errors="raise")
            # Sanity: integer codes should be small positives
            if nums.min() >= 0 and nums.max() < 100:
                return c, "integer"
        except (ValueError, TypeError):
            pass
    # Then try text columns
    for c in candidate_cols:
        if df[c].dtype == object and df[c].astype(str).str.len().median() > 3:
            return c, "text"

    return None, "unknown"


def extract_from_long(
    comorbidity_df: pd.DataFrame,
    inc_key_col: str = "INC_KEY",
) -> pd.DataFrame:
    """Pivot a long-format comorbidity table to wide.

    Tries TWO formats automatically:
      1. PUF_PREEXISTINGCONDITION integer-code format (AY 2017+) — uses
         NTDB_INT_TO_CANONICAL to map integer codes to canonical flags.
      2. Free-text description format (older AY) — uses _match_canonical
         keyword matching to map descriptions to canonical flags.

    Returns DataFrame with `inc_key_col` + 18 canonical columns (0/1 int8).
    Patients in the long table with no recognised conditions still appear
    with all-zero rows.  Patients absent from the long table get 0s at
    merge time.
    """
    empty = pd.DataFrame(columns=[inc_key_col] + CANONICAL_COMORBIDITIES)
    if comorbidity_df is None or len(comorbidity_df) == 0:
        return empty

    cols_lower = {c.lower(): c for c in comorbidity_df.columns}
    inc_actual = cols_lower.get(inc_key_col.lower())
    if inc_actual is None:
        log.warning("extract_from_long: inc_key column not found in %s",
                    list(comorbidity_df.columns))
        return empty

    cond_col, kind = _detect_condition_column(comorbidity_df, inc_actual)
    if cond_col is None:
        log.warning(
            "extract_from_long: cannot identify condition column. "
            "Available: %s", list(comorbidity_df.columns),
        )
        return empty

    log.info("extract_from_long: using inc_key=%r, condition=%r (kind=%s)",
             inc_actual, cond_col, kind)

    df = comorbidity_df[[inc_actual, cond_col]].copy()
    df.columns = [inc_key_col, "_cond"]

    # Map to canonical based on detected kind
    if kind == "integer":
        # Coerce to int and look up in NTDB_INT_TO_CANONICAL
        df["_int"] = pd.to_numeric(df["_cond"], errors="coerce")
        df["_canonical"] = df["_int"].map(NTDB_INT_TO_CANONICAL)
        # Diagnostic: how many integer codes were unmapped?
        unmapped_ints = (df["_int"].notna() & df["_canonical"].isna())
        if unmapped_ints.any():
            seen_codes = (
                df.loc[unmapped_ints, "_int"]
                .astype(int).value_counts().head(15)
            )
            log.info(
                "extract_from_long: %d rows had integer codes NOT in "
                "NTDB_INT_TO_CANONICAL (these don't map to Tran S1 set, "
                "so they're skipped — this is normal). Top codes:",
                int(unmapped_ints.sum()),
            )
            for code, count in seen_codes.items():
                log.info("    code=%-3d occurrences=%d", int(code), int(count))
    else:
        # Free-text matching path
        df["_canonical"] = df["_cond"].astype(str).map(_match_canonical)

    # Diagnostics — counts per canonical and a sample of unmatched
    matched = df.dropna(subset=["_canonical"])
    by_canon = matched.groupby("_canonical").size().to_dict()
    log.info(
        "extract_from_long: matched %d / %d rows to canonical comorbidities. "
        "Per canonical:", len(matched), len(df),
    )
    for canon in CANONICAL_COMORBIDITIES:
        n = by_canon.get(canon, 0)
        log.info("    %-40s %d", canon, n)

    if kind == "text":
        # Show top unmatched strings only when in text mode (not useful for ints)
        unmatched = df[df["_canonical"].isna()]["_cond"].astype(str).str.strip()
        if len(unmatched) > 0:
            unique_unmatched = unmatched.value_counts().head(20)
            log.info("extract_from_long: top 20 unmatched condition strings:")
            for cond, count in unique_unmatched.items():
                log.info("    %-50s %d", cond[:50], count)

    if len(matched) == 0:
        return empty

    wide = (
        matched.assign(_v=1)
        .pivot_table(index=inc_key_col, columns="_canonical",
                     values="_v", aggfunc="max", fill_value=0)
        .astype(np.int8)
    )
    for canonical in CANONICAL_COMORBIDITIES:
        if canonical not in wide.columns:
            wide[canonical] = np.int8(0)
    return wide[CANONICAL_COMORBIDITIES].reset_index()


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------
def merge_comorbidity_features(
    trauma: pd.DataFrame,
    comorbidity_long: pd.DataFrame | None = None,
    inc_key_col: str = "INC_KEY",
) -> pd.DataFrame:
    """Add the 18 canonical comorbidity columns to the trauma frame.

    Strategy
    --------
    1. Look for already-present wide comorbidity columns on ``trauma``
       (case-insensitive, alias matching).  Each canonical found this way
       gets its value from the trauma frame directly.
    2. For comorbidities NOT found in step 1, try to extract them from
       ``comorbidity_long`` (a long-format table loaded by the caller).
    3. Anything still missing → fill with 0 (condition absent).

    Returns NEW DataFrame; does not modify ``trauma`` in place.
    """
    out = trauma.copy()

    # Step 1: wide columns on trauma
    found_wide = extract_from_wide(out)
    log.info(
        "merge_comorbidity_features: %d/%d comorbidities recovered from wide "
        "columns on PUF_TRAUMA: %s",
        len(found_wide), len(CANONICAL_COMORBIDITIES),
        sorted(found_wide.keys()),
    )

    # Step 2: pull missing ones from long-format
    missing_after_wide = [c for c in CANONICAL_COMORBIDITIES if c not in found_wide]
    found_long: pd.DataFrame | None = None
    if missing_after_wide and comorbidity_long is not None and len(comorbidity_long) > 0:
        long_wide = extract_from_long(comorbidity_long, inc_key_col=inc_key_col)
        if len(long_wide) > 0:
            # Resolve trauma-side inc_key column case-insensitively
            trauma_cols_lower = {c.lower(): c for c in out.columns}
            trauma_inc_key = trauma_cols_lower.get(inc_key_col.lower())
            if trauma_inc_key is None:
                log.warning(
                    "merge_comorbidity_features: trauma frame lacks inc_key "
                    "column — long-format merge skipped.",
                )
            else:
                if trauma_inc_key != inc_key_col:
                    long_wide = long_wide.rename(columns={inc_key_col: trauma_inc_key})
                # Only pull canonicals that aren't already populated from wide
                pull_cols = [trauma_inc_key] + missing_after_wide
                pull_cols = [c for c in pull_cols if c in long_wide.columns]
                found_long = long_wide[pull_cols]
                out = out.merge(found_long, on=trauma_inc_key, how="left")

    # Step 3: assign wide-extracted columns + fill remaining NaNs with 0
    for canonical, ser in found_wide.items():
        # If long_wide also added a column (shouldn't happen — we excluded those),
        # the wide-extraction wins
        if canonical in out.columns:
            out[canonical] = ser.values
        else:
            out[canonical] = ser.values
    for canonical in CANONICAL_COMORBIDITIES:
        if canonical not in out.columns:
            out[canonical] = np.int8(0)
        else:
            out[canonical] = pd.to_numeric(out[canonical], errors="coerce") \
                .fillna(0).astype(np.int8)

    # Final coverage report — what fraction of patients have at least one
    # recorded comorbidity
    n_with_any = (out[CANONICAL_COMORBIDITIES].sum(axis=1) > 0).sum()
    log.info(
        "merge_comorbidity_features: %d / %d patients (%.1f%%) have ≥1 "
        "comorbidity recorded after wide+long extraction",
        n_with_any, len(out), 100 * n_with_any / max(1, len(out)),
    )
    # Per-canonical coverage
    log.info("Per-canonical coverage:")
    for canonical in CANONICAL_COMORBIDITIES:
        pct = 100 * out[canonical].sum() / max(1, len(out))
        log.info("    %-40s %5.1f%%", canonical, pct)

    return out
