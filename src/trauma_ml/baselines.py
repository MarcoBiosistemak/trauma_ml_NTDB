"""Clinical scoring baselines — ISS, NISS, TRISS.

Computes binary classification metrics for the three canonical trauma scores
using their medically accepted mortality-risk thresholds:

  * ISS  ≥ 16  →  predicted positive (major trauma / increased mortality risk)
  * NISS ≥ 16  →  predicted positive (same convention as ISS)
  * TRISS < 0.50 → predicted positive (survival probability below 50 %)

TRISS derivation
----------------
The Revised Trauma Score (RTS) is derived from GCS, SBP and RR using
Champion's 1989 coefficients.  TRISS = 1 / (1 + exp(-(b0 + b1*RTS + b2*ISS
+ b3*age_flag))) with the standard penetrating/blunt coefficients.

References
----------
* Baker SP et al. (1974) — ISS
* Osler T et al. (1997)  — NISS
* Boyd CR et al. (1987)  — TRISS
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import binary_metrics

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TRISS coefficients (blunt / penetrating)  — Boyd 1987
# ---------------------------------------------------------------------------
_TRISS_COEF = {
    "blunt": {
        "b0": -1.2470, "b1": 0.9544, "b2": -0.0768, "b3": -1.9052,
    },
    "penetrating": {
        "b0": -0.6029, "b1": 1.1430, "b2": -0.1516, "b3": -2.6676,
    },
}

# GCS → GCS-coded  (Table 1, Champion 1989)
_GCS_CODED = [(14, 15, 4), (11, 13, 3), (8, 10, 2), (5, 7, 1), (3, 4, 0)]
_SBP_CODED = [(90, 999, 4), (76, 89, 3), (50, 75, 2), (1, 49, 1), (0, 0, 0)]
_RR_CODED  = [(10, 29, 4), (30, 35, 3), (6, 9, 2), (1, 5, 1), (0, 0, 0), (36, 999, 3)]


def _code(value: float, table: list[tuple]) -> float:
    if pd.isna(value):
        return np.nan
    for lo, hi, code in table:
        if lo <= value <= hi:
            return float(code)
    return 0.0


def _rts(gcs: float, sbp: float, rr: float) -> float:
    g = _code(gcs, _GCS_CODED)
    s = _code(sbp, _SBP_CODED)
    r = _code(rr, _RR_CODED)
    if any(pd.isna(v) for v in (g, s, r)):
        return np.nan
    return 0.9368 * g + 0.7326 * s + 0.2908 * r


def compute_triss(
    df: pd.DataFrame,
    gcs_col: str = "GCSTOTAL",
    sbp_col: str = "SBPFIRST",
    rr_col: str = "RRFIRST",
    iss_col: str = "ISS",
    age_col: str = "AGEYEARS",
    mechanism_col: str = "TRAUMATYPE",
) -> pd.Series:
    """Return TRISS survival probability (0-1) aligned with df.index.

    TRAUMATYPE encoding (NTDB PUF via ECODE_LOOKUP):
        1 = Blunt, 2 = Penetrating, 3 = Burn, 4 = Other

    When TRAUMATYPE is NaN (e.g. because the ECODE join failed), blunt
    coefficients are used as the conservative fallback and a WARNING is logged
    reporting the fraction affected.  Burn/Other (3/4) also fall back to blunt
    as they are clinically closer to blunt than penetrating.

    Robust to missing columns: if any required input column (gcs, sbp, rr,
    iss, age) is absent or entirely NaN, returns an all-NaN Series and logs
    a warning. The caller (``baseline_metrics``) will then skip TRISS via
    its ``n_valid`` check rather than crashing the whole pipeline.
    """
    def _series_or_nan(col: str) -> pd.Series:
        """Return df[col] coerced to numeric, or an all-NaN Series if absent."""
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index, name=col, dtype=float)

    gcs  = _series_or_nan(gcs_col)
    sbp  = _series_or_nan(sbp_col)
    rr   = _series_or_nan(rr_col)
    iss  = _series_or_nan(iss_col)
    age  = _series_or_nan(age_col)
    mech = df[mechanism_col] if mechanism_col in df.columns else None

    # Hard guard — if any of the five required inputs is absent or completely
    # missing, TRISS cannot be computed for any row.  Return all-NaN.
    required = [(gcs_col, gcs), (sbp_col, sbp), (rr_col, rr),
                (iss_col, iss), (age_col, age)]
    unusable = [name for name, s in required
                if name not in df.columns or s.isna().all()]
    if unusable:
        log.warning(
            "compute_triss: required input(s) absent or all-NaN: %s "
            "— TRISS will be all-NaN. To enable TRISS, ensure these columns "
            "are present and populated in the parquet.",
            unusable,
        )
        return pd.Series(np.nan, index=df.index, name="TRISS", dtype=float)

    rts_vals = np.array([_rts(g, s, r) for g, s, r in zip(gcs, sbp, rr)])
    age_flag = (age >= 55).astype(float)

    # Determine blunt vs penetrating per row
    if mech is not None:
        mech_num = pd.to_numeric(mech, errors="coerce")
        is_pen = (mech_num == 2)          # NTDB code 2 = penetrating
        n_nan = mech_num.isna().sum()
        n_total = len(mech_num)
        if n_nan > 0:
            log.warning(
                "TRAUMATYPE is NaN for %d / %d rows (%.1f%%) — "
                "likely caused by a failed PUF_ECODE_LOOKUP join "
                "(check for ICD10ECODE column name mismatch). "
                "Blunt TRISS coefficients used as fallback for all NaN rows.",
                n_nan, n_total, 100 * n_nan / max(n_total, 1),
            )
    else:
        is_pen = pd.Series(False, index=df.index)
        log.warning(
            "TRAUMATYPE column absent — using blunt TRISS coefficients for all rows. "
            "TRISS penetrating-injury predictions will be biased."
        )

    triss_vals = np.full(len(df), np.nan)
    for i, (rts_v, iss_v, age_f, pen) in enumerate(
            zip(rts_vals, iss, age_flag, is_pen)):
        if any(pd.isna(v) for v in (rts_v, iss_v, age_f)):
            continue
        kind = "penetrating" if pen else "blunt"
        c = _TRISS_COEF[kind]
        b = c["b0"] + c["b1"] * rts_v + c["b2"] * iss_v + c["b3"] * age_f
        triss_vals[i] = 1.0 / (1.0 + np.exp(-b))

    return pd.Series(triss_vals, index=df.index, name="TRISS")


def baseline_metrics(
    df: pd.DataFrame,
    y: np.ndarray,
    iss_col: str = "ISS",
    niss_col: str = "NISS",
    iss_threshold: int = 16,
    niss_threshold: int = 16,
    triss_threshold: float = 0.50,
    **triss_kwargs,
) -> dict[str, dict[str, Any]]:
    """Compute binary classification metrics for ISS, NISS and TRISS baselines.

    Robust to missing columns: if any of ISS / NISS / TRISS-input columns
    is absent or entirely NaN, the corresponding baseline is silently
    skipped (a warning is logged).  Returns a nested dict keyed by
    baseline name.  Each value has the same shape as
    :func:`~trauma_ml.evaluation.binary_metrics`.

    Reporting convention:
    * ISS / NISS are threshold rules (≥16) → no probability → AUROC,
      AUPRC, Brier, log-loss are NaN.  Only thresholded metrics are
      meaningful (accuracy, recall, precision, specificity, F1,
      confusion matrix).
    * TRISS is a probability → all metrics including AUROC, AUPRC,
      Brier are reported, using ``1 - TRISS`` as the death-class score.
    """
    def _series_or_nan(col: str) -> pd.Series:
        """Return df[col] coerced to numeric, or all-NaN if absent/None."""
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            # Ensure index is preserved in case of any pandas weirdness
            return pd.Series(s.to_numpy(), index=df.index, name=col)
        return pd.Series(np.nan, index=df.index, name=col, dtype=float)

    results: dict[str, dict[str, Any]] = {}
    y_series_isna = pd.Series(pd.isna(y), index=df.index)

    # --- ISS ---
    # ISS is used in TWO ways:
    # 1. Threshold rule (>=16) for accuracy/recall/precision/F1/specificity
    # 2. Continuous risk score for AUROC/AUPRC/Brier — higher ISS means
    #    higher mortality probability, so we pass ISS itself as the
    #    positive-class score (rescaled to 0..1 by dividing by 75, the
    #    theoretical max).  This makes ISS comparable to TRISS in Pareto.
    iss = _series_or_nan(iss_col)
    valid = iss.notna() & ~y_series_isna
    if valid.sum() >= 10:
        iss_vals = iss[valid].to_numpy()
        iss_pred = (iss_vals >= iss_threshold).astype(int)
        # Continuous score for AUROC/AUPRC: ISS / 75.0 (max possible = 75)
        iss_score = np.clip(iss_vals / 75.0, 0.0, 1.0)
        y_valid  = y[valid.to_numpy()]
        results["ISS"] = binary_metrics(y_valid, iss_pred, iss_score)
        results["ISS"]["threshold"] = iss_threshold
        results["ISS"]["n_valid"]   = int(valid.sum())
        log.info("ISS baseline (>=%d): n=%d, AUROC=%.3f, recall=%.3f, precision=%.3f",
                 iss_threshold, valid.sum(),
                 results["ISS"].get("AUROC", float("nan")),
                 results["ISS"]["recall"], results["ISS"]["precision"])
    else:
        log.warning(
            "ISS baseline skipped: only %d valid rows (need >=10). "
            "Column %s present=%s, all-NaN=%s",
            int(valid.sum()), iss_col, iss_col in df.columns,
            iss.isna().all() if iss_col in df.columns else "n/a",
        )

    # --- NISS ---
    # Same treatment as ISS: threshold rule for binary metrics + continuous
    # score (NISS/75) for AUROC/AUPRC.
    niss = _series_or_nan(niss_col)
    valid_n = niss.notna() & ~y_series_isna
    if valid_n.sum() >= 10:
        niss_vals = niss[valid_n].to_numpy()
        niss_pred = (niss_vals >= niss_threshold).astype(int)
        niss_score = np.clip(niss_vals / 75.0, 0.0, 1.0)
        y_valid_n = y[valid_n.to_numpy()]
        results["NISS"] = binary_metrics(y_valid_n, niss_pred, niss_score)
        results["NISS"]["threshold"] = niss_threshold
        results["NISS"]["n_valid"]   = int(valid_n.sum())
        log.info("NISS baseline (>=%d): n=%d, AUROC=%.3f, recall=%.3f, precision=%.3f",
                 niss_threshold, valid_n.sum(),
                 results["NISS"].get("AUROC", float("nan")),
                 results["NISS"]["recall"], results["NISS"]["precision"])
    else:
        log.warning(
            "NISS baseline skipped: only %d valid rows (need >=10). "
            "Column %s present=%s, all-NaN=%s",
            int(valid_n.sum()), niss_col, niss_col in df.columns,
            niss.isna().all() if niss_col in df.columns else "n/a",
        )

    # --- TRISS ---
    triss_proba = compute_triss(df, **triss_kwargs)
    valid_t = triss_proba.notna() & ~y_series_isna
    if valid_t.sum() >= 10:
        tp = triss_proba[valid_t].to_numpy()
        # TRISS is a *survival* probability → death = TRISS < threshold
        triss_pred = (tp < triss_threshold).astype(int)
        # For AUROC / AUPRC / Brier we pass 1 - TRISS as the positive-class score
        y_valid_t  = y[valid_t.to_numpy()]
        results["TRISS"] = binary_metrics(y_valid_t, triss_pred, 1.0 - tp)
        results["TRISS"]["threshold"] = triss_threshold
        results["TRISS"]["n_valid"]   = int(valid_t.sum())
        log.info(
            "TRISS baseline (<%.2f survival): n=%d, AUROC=%s, recall=%.3f",
            triss_threshold, valid_t.sum(),
            f"{results['TRISS'].get('AUROC', float('nan')):.3f}",
            results['TRISS']['recall'],
        )
    else:
        log.warning(
            "TRISS baseline skipped: only %d valid rows (need >=10). "
            "Inputs: GCSTOTAL/SBPFIRST/RRFIRST/ISS/AGEYEARS must be present "
            "AND non-NaN for at least 10 rows.",
            int(valid_t.sum()),
        )

    return results


def compute_baseline_scores(
    df: pd.DataFrame,
    iss_col: str = "ISS",
    niss_col: str = "NISS",
    iss_threshold: int = 16,
    niss_threshold: int = 16,
    triss_threshold: float = 0.50,
    **triss_kwargs,
) -> dict[str, dict[str, Any]]:
    """Return a per-baseline dict with score / pred / valid-mask arrays.

    Output structure
    ----------------
    {
        "ISS":   {"score": np.ndarray (continuous, ISS/75 clipped 0..1),
                   "pred":  np.ndarray (binary, ISS >= threshold),
                   "valid": pd.Series (bool mask aligned with df.index),
                   "threshold": int (ISS threshold used)},
        "NISS":  same shape,
        "TRISS": {"score": 1 - TRISS_proba (death probability),
                   "pred":  TRISS_proba < threshold (binary death prediction),
                   "valid": pd.Series,
                   "threshold": float},
    }

    A baseline is omitted from the dict if it cannot be computed
    (column absent or fewer than 10 valid rows).

    These arrays are used by ``Trainer._evaluate_baselines`` for subgroup
    analysis, by ``Trainer._calibrate_baselines`` for Platt/isotonic
    fitting and threshold tuning, and at evaluation time on every
    partition.
    """
    def _series_or_nan(col: str) -> pd.Series:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            return pd.Series(s.to_numpy(), index=df.index, name=col)
        return pd.Series(np.nan, index=df.index, name=col, dtype=float)

    out: dict[str, dict[str, Any]] = {}

    # ISS
    iss = _series_or_nan(iss_col)
    valid_i = iss.notna()
    if valid_i.sum() >= 10:
        score = np.clip(iss.to_numpy() / 75.0, 0.0, 1.0)
        # Set to NaN where invalid, keep array length = len(df)
        score = np.where(valid_i.to_numpy(), score, np.nan)
        pred = np.where(valid_i.to_numpy(),
                         (iss.to_numpy() >= iss_threshold).astype(int),
                         -1)
        out["ISS"] = {
            "score": score, "pred": pred, "valid": valid_i,
            "threshold": iss_threshold,
        }

    # NISS
    niss = _series_or_nan(niss_col)
    valid_n = niss.notna()
    if valid_n.sum() >= 10:
        score = np.clip(niss.to_numpy() / 75.0, 0.0, 1.0)
        score = np.where(valid_n.to_numpy(), score, np.nan)
        pred = np.where(valid_n.to_numpy(),
                         (niss.to_numpy() >= niss_threshold).astype(int),
                         -1)
        out["NISS"] = {
            "score": score, "pred": pred, "valid": valid_n,
            "threshold": niss_threshold,
        }

    # TRISS
    triss_proba = compute_triss(df, **triss_kwargs)
    valid_t = triss_proba.notna()
    if valid_t.sum() >= 10:
        # Death probability = 1 - TRISS_survival
        death_score = 1.0 - triss_proba.to_numpy()
        death_score = np.where(valid_t.to_numpy(), death_score, np.nan)
        pred = np.where(valid_t.to_numpy(),
                         (triss_proba.to_numpy() < triss_threshold).astype(int),
                         -1)
        out["TRISS"] = {
            "score": death_score, "pred": pred, "valid": valid_t,
            "threshold": triss_threshold,
        }

    return out
