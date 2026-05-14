"""Stratified train / calibration / test splitting.

Stratification combines (target, gender, age_group, **admission_year**) so
that the year distribution is preserved across partitions.  The holdout set is
a *temporally-held-out* year slice: the most recent year(s) passed in
``holdout_years`` are excluded from the train/calibration/test pool and
evaluated independently as a temporal holdout.

Key guarantees
--------------
* Year stratification is always included when ``__admission_year`` is present.
* Rare strata (n < 2) are merged into an '__other__' bucket.
* The returned dict has keys ``'train' | 'calibration' | 'test'``.
* :func:`extract_holdout_years` splits the DataFrame into a non-holdout pool
  (used for train/cal/test) and the temporal holdout slice.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def assign_age_group(
    ages: pd.Series,
    edges: Iterable[int] = (0, 18, 45, 65, 75, 200),
    labels: Iterable[str] = ("pediatric", "young_adult", "middle_aged",
                             "early_elderly", "elderly"),
) -> pd.Series:
    """Bin ages into labelled categories. Defaults follow ECTrauma age bands."""
    edges = list(edges)
    labels = list(labels)
    if len(labels) != len(edges) - 1:
        raise ValueError("labels must have one fewer element than edges")
    return pd.cut(ages, bins=edges, labels=labels, right=False,
                  include_lowest=True).astype("object")


def _combine_stratum(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Join stratification columns row-wise into '|'-delimited keys.

    NaN cells are coerced to the literal string ``"NaN"`` before joining
    so the row-wise ``"|".join(...)`` never sees a float.  Without this,
    pandas' StringDtype preserves NaN as ``float('nan')`` even after
    ``.astype(str)``, and the join raises
    ``TypeError: sequence item N: expected str instance, float found``.

    Stratification cells that are NaN end up grouped together under the
    "NaN" token; the rare-bucket collapse downstream folds them into
    "__other__" if the group is below the 2-row minimum.
    """
    # Use object dtype + fillna so NaNs become real "NaN" strings rather
    # than pandas-StringArray missing values.
    return (
        df[cols]
        .astype(object)
        .fillna("NaN")
        .astype(str)
        .agg("|".join, axis=1)
    )


def extract_holdout_years(
    df: pd.DataFrame,
    holdout_years: list[int],
    year_col: str = "__admission_year",
) -> tuple[pd.Index, pd.Index]:
    """Return (non_holdout_index, holdout_index).

    Rows in *holdout_years* are the temporal holdout and are excluded from the
    random train/cal/test pool entirely.  Pass the returned indices to the
    Trainer so it knows which rows belong to each bucket.
    """
    if year_col not in df.columns or not holdout_years:
        return df.index, pd.Index([], dtype=df.index.dtype)

    is_holdout = df[year_col].isin(holdout_years)
    holdout_idx = df.index[is_holdout]
    non_holdout_idx = df.index[~is_holdout]
    log.info(
        "Temporal holdout: %d rows (years %s) | non-holdout pool: %d rows",
        len(holdout_idx), sorted(holdout_years), len(non_holdout_idx),
    )
    return non_holdout_idx, holdout_idx


def stratified_split(
    df: pd.DataFrame,
    y: pd.Series,
    stratify_by: list[str] | None = None,
    train_frac: float = 0.70,
    calibration_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
    id_col: str | None = "INC_KEY",
    year_col: str = "__admission_year",
) -> dict[str, pd.Index]:
    """Return ``{'train', 'calibration', 'test'}`` index dicts.

    ``year_col`` is automatically added to the stratification columns when
    present, preserving the year distribution across all three partitions.
    Call this only on the non-holdout pool (after :func:`extract_holdout_years`).
    """
    if not np.isclose(train_frac + calibration_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    if stratify_by is None:
        stratify_by = []

    # Always stratify by year when available
    extra = [year_col] if year_col in df.columns else []
    stratify_cols = list(stratify_by) + extra + ["__y"]

    work = df.copy()
    work["__y"] = y.values
    strat_key = _combine_stratum(work, stratify_cols)
    counts = strat_key.value_counts()
    rare = counts[counts < 2].index
    strat_key = strat_key.where(~strat_key.isin(rare), other="__other__")

    idx_all = work.index.to_numpy()
    idx_train, idx_temp, _, strat_temp = train_test_split(
        idx_all,
        strat_key,
        test_size=(1.0 - train_frac),
        stratify=strat_key,
        random_state=random_state,
    )
    test_prop_in_temp = test_frac / (calibration_frac + test_frac)
    temp_counts = pd.Series(strat_temp).value_counts()
    stratify_temp = strat_temp if (temp_counts >= 2).all() else None

    idx_cal, idx_test = train_test_split(
        idx_temp,
        test_size=test_prop_in_temp,
        stratify=stratify_temp,
        random_state=random_state,
    )

    log.info(
        "Split sizes: train=%d, calibration=%d, test=%d (year-stratified=%s)",
        len(idx_train), len(idx_cal), len(idx_test),
        year_col in df.columns,
    )
    return {
        "train":       pd.Index(idx_train),
        "calibration": pd.Index(idx_cal),
        "test":        pd.Index(idx_test),
    }
