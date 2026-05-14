"""Imputation bank — wraps several imputers behind a uniform interface and
offers a holdout evaluation that masks known values in the test set, imputes
them, and reports MAE / accuracy per variable.

Supported methods
-----------------
none           : no imputation (passthrough — requires a model that handles NaN)
median_mode    : SimpleImputer(median) for numeric, mode for categorical
knn            : KNNImputer(k=10)
mice           : IterativeImputer (BayesianRidge default regressor)
missforest     : miceforest (optional, needs miceforest)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

log = logging.getLogger(__name__)


@dataclass
class Imputer:
    """Uniform imputer facade.  ``.fit(df[num_cols])`` then ``.transform(df[num_cols])``."""
    method: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    _num_imputer: object | None = None
    _cat_imputer: object | None = None

    # -------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame) -> "Imputer":
        method = self.method.lower()

        # Numeric branch
        num_df = df[self.numeric_cols]
        if method == "none":
            self._num_imputer = None
        elif method == "median_mode":
            self._num_imputer = SimpleImputer(strategy="median").fit(num_df)
        elif method == "knn":
            self._num_imputer = KNNImputer(n_neighbors=10, weights="uniform",
                                            metric="nan_euclidean").fit(num_df)
        elif method == "mice":
            self._num_imputer = IterativeImputer(
                max_iter=10, tol=1e-3, random_state=42
            ).fit(num_df)
        elif method == "missforest":
            try:
                import miceforest as mf
            except ImportError as e:
                raise RuntimeError(
                    "miceforest not installed; `pip install miceforest` or use "
                    "`pip install 'trauma_ml[imputers]'`"
                ) from e
            kds = mf.ImputationKernel(num_df, save_all_iterations=False, random_state=42)
            kds.mice(5)
            self._num_imputer = kds
        else:
            raise ValueError(f"Unknown imputer method {self.method!r}")

        # Categorical branch: simple mode imputation (stored even when method=='none'
        # because most models need categoricals non-null)
        if self.categorical_cols and method != "none":
            self._cat_imputer = SimpleImputer(strategy="most_frequent").fit(
                df[self.categorical_cols]
            )
        return self

    # -------------------------------------------------------------- #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._num_imputer is not None:
            if self.method == "missforest":
                out[self.numeric_cols] = self._num_imputer.complete_data(
                    dataset=0
                )[self.numeric_cols].values
            else:
                out[self.numeric_cols] = self._num_imputer.transform(
                    out[self.numeric_cols]
                )
        if self._cat_imputer is not None and self.categorical_cols:
            out[self.categorical_cols] = self._cat_imputer.transform(
                out[self.categorical_cols]
            )
        return out

    # -------------------------------------------------------------- #
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Holdout imputation evaluation
# ---------------------------------------------------------------------------
def evaluate_imputation(
    df_test: pd.DataFrame,
    imputer: Imputer,
    mask_fraction: float = 0.10,
    random_state: int = 42,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Mask a fraction of known values in ``df_test`` per variable, run the imputer
    that is already fitted on train, and record the per-variable error.

    Returns a dataframe with one row per (variable, method) including:
      n_masked, MAE (numeric) or accuracy (categorical).
    """
    rng = np.random.default_rng(random_state)
    results = []

    numeric_set = set(imputer.numeric_cols)
    categorical_set = set(imputer.categorical_cols)

    # Choose which cells to mask
    masked_df = df_test.copy()
    mask_records: dict[str, np.ndarray] = {}
    for col in imputer.numeric_cols + imputer.categorical_cols:
        known_idx = df_test.index[df_test[col].notna()]
        if len(known_idx) == 0:
            continue
        n_mask = max(1, int(len(known_idx) * mask_fraction))
        mask_idx = rng.choice(known_idx, size=n_mask, replace=False)
        mask_records[col] = mask_idx
        masked_df.loc[mask_idx, col] = np.nan

    imputed_df = imputer.transform(masked_df)

    for col, mask_idx in mask_records.items():
        truth = df_test.loc[mask_idx, col].to_numpy()
        pred = imputed_df.loc[mask_idx, col].to_numpy()
        if col in numeric_set:
            truth = truth.astype(float)
            pred = pred.astype(float)
            mae = float(np.mean(np.abs(truth - pred)))
            rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
            results.append({
                "variable": col, "type": "numeric",
                "n_masked": int(len(mask_idx)),
                "MAE": mae, "RMSE": rmse,
                "method": imputer.method,
            })
        elif col in categorical_set:
            acc = float(np.mean(truth == pred))
            results.append({
                "variable": col, "type": "categorical",
                "n_masked": int(len(mask_idx)),
                "accuracy": acc,
                "method": imputer.method,
            })

    df_out = pd.DataFrame(results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"imputation_eval_{imputer.method}.csv"
        df_out.to_csv(csv_path, index=False)
        log.info("Wrote imputation evaluation to %s", csv_path)

    return df_out


def make_imputer(
    method: str,
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
) -> Imputer:
    """Factory that enforces the method naming convention."""
    return Imputer(
        method=method,
        numeric_cols=list(numeric_cols),
        categorical_cols=list(categorical_cols),
    )
