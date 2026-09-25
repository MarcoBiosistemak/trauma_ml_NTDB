"""Imputation bank — wraps several imputers behind a uniform interface and
offers a holdout evaluation that masks known values in the test set, imputes
them, and reports MAE / accuracy per variable.

Supported methods
-----------------
none           : no imputation (passthrough — requires a model that handles NaN)
median_mode    : SimpleImputer(median) for numeric, mode for categorical
knn            : KNNImputer(k=10)
mice           : IterativeImputer (BayesianRidge default regressor)
bagged_trees   : MissForest-style — IterativeImputer with BAGGED DECISION TREES
                 (BaggingRegressor) for numeric, and a per-column BaggingClassifier
                 (bagged trees) for categorical.  sklearn-only, no extra deps.
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
from sklearn.metrics import balanced_accuracy_score

log = logging.getLogger(__name__)

# Bagged-trees ("bagged_trees") imputer hyper-parameters.  Kept modest +
# sub-sampled so a MissForest-style imputer is tractable on the ~1.4M-row NTDB
# training pool.  Tune here if you want deeper/forest-ier behaviour.
_BT_N_ESTIMATORS = 10          # trees per bagged ensemble
_BT_MAX_DEPTH = 16             # cap tree depth (None = unbounded, much slower)
_BT_NUM_MAX_ITER = 5           # IterativeImputer round-robin passes (numeric)
_BT_MAX_SAMPLES = 0.5          # bootstrap fraction per tree
_BT_CAT_FIT_ROWS = 100_000     # cap observed rows used to fit each categorical model

# Gradient-boosting imputer (HistGradientBoosting — fast, NaN-native).
_GB_N_ITER = 100               # boosting iterations per HistGBM
_GB_MAX_DEPTH = None           # None -> use max_leaf_nodes (default 31)
_GB_LR = 0.1                   # learning rate
_GB_NUM_MAX_ITER = 5           # IterativeImputer round-robin passes (numeric)

# MissForest (miceforest) — number of MICE iterations.
_MF_ITERATIONS = 5


@dataclass
class Imputer:
    """Uniform imputer facade.  ``.fit(df[num_cols])`` then ``.transform(df[num_cols])``."""
    method: str
    numeric_cols: list[str]
    categorical_cols: list[str]
    _num_imputer: object | None = None
    _cat_imputer: object | None = None
    # missforest state
    _missforest_medians_: object | None = None   # Series of numeric medians (fallback)
    # bagged_trees state
    _cat_bagged: dict | None = None        # col -> ("model", clf, feat_cols) | ("const", value)
    _bt_num_median: object | None = None   # Series of numeric medians (predictor fill)
    _bt_cat_mode: object | None = None      # Series of categorical modes (predictor fill)

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
        elif method == "bagged_trees":
            # MissForest-style: round-robin IterativeImputer whose per-column
            # regressor is a bag of decision trees.
            from sklearn.ensemble import BaggingRegressor
            from sklearn.tree import DecisionTreeRegressor
            self._num_imputer = IterativeImputer(
                estimator=BaggingRegressor(
                    estimator=DecisionTreeRegressor(max_depth=_BT_MAX_DEPTH),
                    n_estimators=_BT_N_ESTIMATORS,
                    max_samples=_BT_MAX_SAMPLES,
                    random_state=42, n_jobs=-1,
                ),
                max_iter=_BT_NUM_MAX_ITER, tol=1e-3, random_state=42,
            ).fit(num_df)
            # Categorical columns get their OWN bagged-tree classifiers.
            self._fit_bagged_categorical(df)
        elif method == "gradient_boosting":
            # Gradient-boosting (HistGradientBoosting) MICE-style imputer:
            # round-robin IterativeImputer whose per-column regressor is a
            # histogram gradient-boosting machine.  Fast and NaN-native.
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._num_imputer = IterativeImputer(
                estimator=HistGradientBoostingRegressor(
                    max_iter=_GB_N_ITER, max_depth=_GB_MAX_DEPTH,
                    learning_rate=_GB_LR, random_state=42),
                max_iter=_GB_NUM_MAX_ITER, tol=1e-3, random_state=42,
            ).fit(num_df)
            # Categoricals: per-column gradient-boosting classifiers.
            self._fit_bagged_categorical(df)
        elif method == "missforest":
            try:
                import miceforest as mf
            except ImportError as e:
                raise RuntimeError(
                    "miceforest not installed; `pip install miceforest` or use "
                    "`pip install 'trauma_ml[imputers]'`"
                ) from e
            # miceforest renamed constructor args between majors:
            #   v5: save_all_iterations / datasets
            #   v6: save_all_iterations_data / num_datasets
            # Pass only what the installed version actually accepts.
            import inspect
            try:
                supported = set(
                    inspect.signature(mf.ImputationKernel.__init__).parameters
                ) - {"self", "args", "kwargs"}
            except (TypeError, ValueError):      # pragma: no cover
                supported = set()
            # NOTE: these MUST be True.  miceforest can only impute a *new*
            # frame (impute_new_data) if the MICE iteration data was retained;
            # with False, transform() on test/holdout silently degrades to a
            # median fill, making "missforest" identical to median_mode.
            candidates = {
                "save_all_iterations_data": True,    # v6+
                "save_all_iterations": True,         # v5
                "num_datasets": 1,                   # v6+
                "datasets": 1,                       # v5
                "random_state": 42,
            }
            kw = {k: v for k, v in candidates.items() if k in supported}
            if "save_all_iterations_data" in kw:
                kw.pop("save_all_iterations", None)
            if "num_datasets" in kw:
                kw.pop("datasets", None)
            log.info("missforest: miceforest kernel kwargs=%s", sorted(kw))
            # miceforest 6.x asserts a clean 0..n-1 RangeIndex ("Please reset
            # the index on the dataframe").  Upstream row filtering (inclusion
            # strategy, dropped-gender rows, train/test split) leaves gaps in
            # the index, so reset it before handing the frame over.  We only
            # ever use positional (.values) assignment afterwards, so dropping
            # the original index is safe.
            num_df = num_df.reset_index(drop=True)
            try:
                kds = mf.ImputationKernel(num_df, **kw)
                kds.mice(_MF_ITERATIONS)
            except TypeError as exc:
                # miceforest's mean-matching reaches into LightGBM internals
                # (Booster.__inner_predict), whose signature changed in
                # LightGBM >= 4.6.  Disabling mean matching avoids that path
                # and uses the model predictions directly.
                log.warning(
                    "missforest: mean matching failed (%s) - likely a "
                    "miceforest/LightGBM version clash; retrying with "
                    "mean_match_candidates=0", exc)
                if "mean_match_candidates" in supported:
                    kw["mean_match_candidates"] = 0
                kds = mf.ImputationKernel(num_df, **kw)
                kds.mice(_MF_ITERATIONS)
            self._num_imputer = kds
            # Safety net for transform(): if impute_new_data ever fails on a
            # new frame we still need a deterministic numeric fill.
            self._missforest_medians_ = num_df.median(numeric_only=True)
        else:
            raise ValueError(f"Unknown imputer method {self.method!r}")

        # Categorical branch: simple mode imputation (stored even when method=='none'
        # because most models need categoricals non-null).  For 'bagged_trees' the
        # categoricals are handled by per-column bagged classifiers (fit above), so
        # skip the mode imputer here.
        if self.categorical_cols and method not in ("none", "bagged_trees", "gradient_boosting"):
            self._cat_imputer = SimpleImputer(strategy="most_frequent").fit(
                df[self.categorical_cols]
            )
        return self

    # -------------------------------------------------------------- #
    def _fit_bagged_categorical(self, df: pd.DataFrame) -> None:
        """Fit a bagged-tree CLASSIFIER per categorical column (MissForest-style).

        Each categorical is predicted from all numeric + the other categoricals.
        Predictor cells are filled with train medians / modes so the classifier
        always sees a complete feature matrix.  Fitting is sub-sampled to
        ``_BT_CAT_FIT_ROWS`` observed rows per column to stay tractable on the
        ~1.4M-row training pool.  Columns with <50 observed rows or a single
        class fall back to a constant (mode) fill.
        """
        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier

        def _make_cat_clf():
            # gradient_boosting uses a HistGBM classifier; bagged_trees uses a
            # bag of decision trees.  Both populate self._cat_bagged and share
            # the same transform path.
            if self.method == "gradient_boosting":
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier(
                    max_iter=_GB_N_ITER, max_depth=_GB_MAX_DEPTH,
                    learning_rate=_GB_LR, random_state=42)
            return BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=_BT_MAX_DEPTH),
                n_estimators=_BT_N_ESTIMATORS,
                max_samples=_BT_MAX_SAMPLES,
                random_state=42, n_jobs=-1,
            )

        self._bt_num_median = df[self.numeric_cols].median() if self.numeric_cols \
            else pd.Series(dtype=float)
        self._bt_cat_mode = (
            df[self.categorical_cols].mode().iloc[0]
            if self.categorical_cols and len(df) else pd.Series(dtype=float)
        )
        self._cat_bagged = {}

        for col in self.categorical_cols:
            obs = df[df[col].notna()]
            if len(obs) < 50 or obs[col].nunique() < 2:
                fallback = (obs[col].mode().iloc[0] if len(obs) else 0)
                self._cat_bagged[col] = ("const", fallback)
                continue
            if len(obs) > _BT_CAT_FIT_ROWS:
                obs = obs.sample(_BT_CAT_FIT_ROWS, random_state=42)
            feat_cols = self.numeric_cols + [c for c in self.categorical_cols if c != col]
            Xf = self._bt_fill_features(obs, feat_cols)
            yf = obs[col].astype(int)
            clf = _make_cat_clf().fit(Xf, yf)
            self._cat_bagged[col] = ("model", clf, feat_cols)
        log.info("%s: fit %d categorical classifiers (+%d constant)",
                 self.method,
                 sum(1 for v in self._cat_bagged.values() if v[0] == "model"),
                 sum(1 for v in self._cat_bagged.values() if v[0] == "const"))

    def _bt_fill_features(self, frame: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
        """Complete the predictor matrix for the bagged categorical models with
        stored medians/modes.  A regular method (NOT a closure) so the fitted
        Imputer stays picklable for ModelArtifact persistence / deployment."""
        f = frame[feat_cols].copy()
        num_here = [c for c in feat_cols if c in self.numeric_cols]
        cat_here = [c for c in feat_cols if c in self.categorical_cols]
        if num_here:
            f[num_here] = f[num_here].fillna(self._bt_num_median[num_here])
        for c in cat_here:
            f[c] = f[c].fillna(self._bt_cat_mode.get(c, 0))
        return f

    def _transform_bagged_categorical(self, out: pd.DataFrame) -> None:
        for col, spec in (self._cat_bagged or {}).items():
            miss = out[col].isna()
            if not miss.any():
                continue
            if spec[0] == "const":
                out.loc[miss, col] = spec[1]
            else:
                _, clf, feat_cols = spec
                Xf = self._bt_fill_features(out.loc[miss], feat_cols)
                out.loc[miss, col] = clf.predict(Xf)

    # -------------------------------------------------------------- #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._num_imputer is not None:
            if self.method == "missforest":
                # IMPORTANT: complete_data() returns the *training* frame the
                # kernel was fitted on.  For any new frame (calibration/test/
                # holdout/serving) we must run impute_new_data() first,
                # otherwise we'd write training values into `out` (wrong rows,
                # and a shape mismatch whenever len(df) != len(train)).
                # Same RangeIndex requirement as at fit time.
                sub = out[self.numeric_cols].reset_index(drop=True)
                try:
                    newk = self._num_imputer.impute_new_data(new_data=sub)
                    completed = newk.complete_data(dataset=0)
                except Exception as exc:
                    log.warning(
                        "missforest impute_new_data failed (%s); falling back "
                        "to train-median fill for numeric columns", exc)
                    completed = sub.fillna(self._missforest_medians_)
                out[self.numeric_cols] = completed[self.numeric_cols].values
            else:
                out[self.numeric_cols] = self._num_imputer.transform(
                    out[self.numeric_cols]
                )
        if self._cat_imputer is not None and self.categorical_cols:
            out[self.categorical_cols] = self._cat_imputer.transform(
                out[self.categorical_cols]
            )
        # bagged_trees: per-column bagged classifiers for categoricals
        if self.method in ("bagged_trees", "gradient_boosting") and self._cat_bagged:
            self._transform_bagged_categorical(out)
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
    train_std: dict[str, float] | None = None,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Mask a fraction of known values in ``df_test`` per variable, run the imputer
    that is already fitted on train, and record the per-variable error.

    Round 37: when ``train_std`` and ``thresholds`` are supplied, the
    returned frame also includes:
      - ``std_train`` — standard deviation of the column on the training
        set (used to scale numeric MAE)
      - ``relative_MAE`` — MAE / std_train (for numeric variables)
      - ``keep`` — boolean indicating the imputer's reconstruction passes
        the threshold supplied in ``thresholds``.  Used downstream when
        ``imputer_check=True`` to filter out features the imputer can't
        recover reliably.

    Returns a dataframe with one row per (variable, method).
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

    thresholds = thresholds or {}
    num_thresh = float(thresholds.get("numeric_relative_mae_max", float("inf")))
    cat_thresh = float(thresholds.get("categorical_balanced_accuracy_min", -1.0))

    for col, mask_idx in mask_records.items():
        truth = df_test.loc[mask_idx, col].to_numpy()
        pred = imputed_df.loc[mask_idx, col].to_numpy()
        if col in numeric_set:
            try:
                truth = truth.astype(float)
                pred = pred.astype(float)
            except (ValueError, TypeError):
                continue
            mae = float(np.mean(np.abs(truth - pred)))
            rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
            std = (train_std or {}).get(col)
            if std is None or std <= 0 or not np.isfinite(std):
                std = float(np.std(truth)) if len(truth) > 1 else float("nan")
            rel_mae = (mae / std) if std and np.isfinite(std) and std > 0 else float("nan")
            keep = bool(np.isfinite(rel_mae) and rel_mae <= num_thresh) if thresholds else True
            results.append({
                "variable": col, "type": "numeric",
                "n_masked": int(len(mask_idx)),
                "MAE": mae, "RMSE": rmse,
                "std_train": float(std) if np.isfinite(std) else None,
                "relative_MAE": rel_mae if np.isfinite(rel_mae) else None,
                "keep": keep,
                "method": imputer.method,
            })
        elif col in categorical_set:
            acc = float(np.mean(truth == pred))
            # Round 54: assess categorical reconstruction with BALANCED ACCURACY
            # (mean per-class recall), which is chance-corrected and not fooled by
            # imbalanced flags — plain accuracy lets a majority/mode imputer score
            # high without ever recovering the minority category.  We still record
            # plain `accuracy` for reference, but the keep-gate uses balanced acc.
            try:
                t_lab = truth.astype(int)
                p_lab = pred.astype(int)
            except (ValueError, TypeError):
                t_lab = truth.astype(str)
                p_lab = pred.astype(str)
            if len(np.unique(t_lab)) < 2:
                # only one class present in the masked sample — balanced acc is
                # undefined/degenerate, fall back to plain accuracy
                bal_acc = acc
            else:
                bal_acc = float(balanced_accuracy_score(t_lab, p_lab))
            keep = bool(bal_acc >= cat_thresh) if thresholds else True
            results.append({
                "variable": col, "type": "categorical",
                "n_masked": int(len(mask_idx)),
                "balanced_accuracy": bal_acc,
                "accuracy": acc,
                "keep": keep,
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


def select_good_features(eval_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split eval_df['variable'] into (kept, dropped) by the 'keep' column.

    Designed to be called when ``imputer_check=True``.  Variables with
    ``keep == True`` survive; the rest are reported so the caller can log
    and drop them from the predictor set.
    """
    if eval_df.empty or "keep" not in eval_df.columns:
        return list(eval_df.get("variable", [])), []
    kept = eval_df.loc[eval_df["keep"] == True, "variable"].tolist()
    dropped = eval_df.loc[eval_df["keep"] != True, "variable"].tolist()
    return kept, dropped


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
