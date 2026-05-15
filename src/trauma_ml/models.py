"""Model family wrappers — one factory per family from the CEIm memoria.

Each factory returns an sklearn-compatible estimator already configured with
sensible defaults.  Heavy / optional dependencies (XGBoost, LightGBM,
CatBoost, FLAML, TabPFN, TabNet, FT-Transformer, scikit-survival) are
imported lazily so that installing the base package does not require them.

The ``build_model`` dispatcher accepts a ``family`` name, the task type (binary,
multiclass, or survival), and returns an estimator ready to call ``.fit(X, y)``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task dispatcher
# ---------------------------------------------------------------------------
_TASK_KINDS = {"binary", "multiclass", "survival"}


def _require(module_name: str, extras_hint: str):
    """Raise a helpful ImportError if an optional dependency is missing."""
    def _inner():
        raise ImportError(
            f"{module_name!r} is required for this model family. "
            f"Install with: pip install 'trauma_ml[{extras_hint}]'"
        )
    return _inner


# ---------------------------------------------------------------------------
# GPU detection (round 12)
# ---------------------------------------------------------------------------
def _gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is present and torch can see it.

    Used as the default 'auto' policy across model factories.  Falls back
    to False on any failure (no torch installed, no CUDA driver, etc.).
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device(use_gpu: str | bool | None = "auto") -> str:
    """Resolve the GPU policy keyword to a concrete device string.

    Accepts:
      * ``"auto"`` (default) — use GPU if available, else CPU.
      * ``"force"`` / ``True`` — use GPU; raise if unavailable.
      * ``"never"`` / ``False`` / ``None`` — always CPU.

    Returns ``"cuda"`` or ``"cpu"``.
    """
    if use_gpu in (None, False, "never", "no", "off"):
        return "cpu"
    if use_gpu in (True, "force", "yes", "on", "gpu", "cuda"):
        if not _gpu_available():
            raise RuntimeError(
                "use_gpu='force' but no CUDA-capable GPU detected via torch. "
                "Install torch with CUDA support, or use use_gpu='auto'."
            )
        return "cuda"
    # auto
    return "cuda" if _gpu_available() else "cpu"


# ---------------------------------------------------------------------------
# Linear penalised regressions
# ---------------------------------------------------------------------------
def logistic_l1(task: str, n_jobs: int = 4, **kwargs) -> Any:
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        C=kwargs.get("C", 1.0),
        n_jobs=n_jobs,
        random_state=42,
    )


def logistic_elasticnet(task: str, n_jobs: int = 4, **kwargs) -> Any:
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=kwargs.get("l1_ratio", 0.5),
        max_iter=5000,
        C=kwargs.get("C", 1.0),
        n_jobs=n_jobs,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Tree-based ensembles
# ---------------------------------------------------------------------------
def random_forest(task: str, n_jobs: int = 4, **kwargs) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 500),
        max_depth=kwargs.get("max_depth", None),
        n_jobs=n_jobs,
        random_state=42,
        class_weight="balanced",
    )


def xgboost(task: str, n_jobs: int = 4, use_gpu: str = "auto", **kwargs) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return _require("xgboost", "boosting")()
    objective = "binary:logistic" if task == "binary" else "multi:softprob"
    device = _resolve_device(use_gpu)
    log.info("xgboost factory: device=%s (use_gpu=%r)", device, use_gpu)
    return XGBClassifier(
        n_estimators=kwargs.get("n_estimators", 500),
        max_depth=kwargs.get("max_depth", 6),
        learning_rate=kwargs.get("learning_rate", 0.05),
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        # tree_method='hist' is the recommended GPU-compatible histogram method;
        # 'device=cuda' (XGBoost >=2.0) routes histogram to the GPU.
        tree_method="hist",
        device=device,
        n_jobs=n_jobs,
        random_state=42,
        eval_metric="auc" if task == "binary" else "mlogloss",
    )


def lightgbm(task: str, n_jobs: int = 4, use_gpu: str = "auto", **kwargs) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return _require("lightgbm", "boosting")()
    device = _resolve_device(use_gpu)
    # LightGBM uses 'gpu' as the keyword (not 'cuda') and requires the
    # GPU-compiled wheel; the standard `pip install lightgbm` ships CPU-only,
    # so we silently fall back to CPU if a GPU build isn't available.
    if device == "cuda":
        try:
            # Quick sanity probe: build a tiny model with device='gpu' to
            # check the binary supports it.  If it doesn't, fall back.
            from lightgbm import LGBMClassifier as _Probe
            _Probe(device="gpu", n_estimators=1).fit(
                [[0.0]], [0],
            )
            lgbm_device = "gpu"
            log.info("lightgbm factory: device=gpu (CUDA build detected)")
        except Exception as exc:
            log.warning(
                "lightgbm GPU build NOT available (%s) — falling back to CPU. "
                "Reinstall with the CUDA-enabled wheel: "
                "`pip install --no-binary=lightgbm lightgbm` or use the conda "
                "package `lightgbm-gpu` to enable GPU.",
                exc,
            )
            lgbm_device = "cpu"
    else:
        lgbm_device = "cpu"
    return LGBMClassifier(
        n_estimators=kwargs.get("n_estimators", 500),
        max_depth=kwargs.get("max_depth", -1),
        learning_rate=kwargs.get("learning_rate", 0.05),
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary" if task == "binary" else "multiclass",
        n_jobs=n_jobs,
        device=lgbm_device,
        random_state=42,
        verbose=-1,
    )


def catboost(task: str, n_jobs: int = 4, use_gpu: str = "auto", **kwargs) -> Any:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return _require("catboost", "boosting")()
    device = _resolve_device(use_gpu)
    task_type = "GPU" if device == "cuda" else "CPU"
    log.info("catboost factory: task_type=%s (use_gpu=%r)", task_type, use_gpu)
    return CatBoostClassifier(
        iterations=kwargs.get("n_estimators", 500),
        depth=kwargs.get("max_depth", 6),
        learning_rate=kwargs.get("learning_rate", 0.05),
        loss_function="Logloss" if task == "binary" else "MultiClass",
        thread_count=n_jobs,
        task_type=task_type,
        random_seed=42,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# AutoML
# ---------------------------------------------------------------------------
def flaml(task: str, n_jobs: int = 4, time_budget: int = 300, **kwargs) -> Any:
    try:
        from flaml import AutoML
    except ImportError:
        return _require("flaml", "automl")()
    automl = AutoML()
    # FLAML is trained differently: return a wrapper with .fit/.predict_proba.
    return _FLAMLWrapper(
        automl=automl,
        task="classification",
        time_budget=time_budget,
        metric="roc_auc" if task == "binary" else "accuracy",
        n_jobs=n_jobs,
    )


class _FLAMLWrapper:
    """sklearn-like facade around FLAML's AutoML."""
    def __init__(self, automl, task, time_budget, metric, n_jobs):
        self.automl = automl
        self.task = task
        self.time_budget = time_budget
        self.metric = metric
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X, y):
        # FLAML 2.1.x's bundled XGBoost integration calls XGBClassifier.fit
        # with a `callbacks=` kwarg.  XGBoost 3.x removed that argument →
        # FLAML crashes with TypeError on every xgb trial, killing the run.
        # Workaround: tell FLAML which estimators to search and OMIT xgboost.
        # FLAML still has lightgbm, RF, extra_tree, l1/l2 logreg, catboost,
        # kneighbor — plenty of model space to explore.
        # When FLAML pins to XGBoost 3.x compatibility (issue #1485), we can
        # restore xgboost here.
        estimator_list = [
            "lgbm",          # LightGBM
            "rf",            # RandomForest
            "extra_tree",    # ExtraTrees
            "lrl1", "lrl2",  # LogisticRegression L1 / L2
            "catboost",      # CatBoost (if installed)
            "kneighbor",     # KNN
        ]
        try:
            self.automl.fit(
                X_train=X, y_train=y,
                task=self.task,
                time_budget=self.time_budget,
                metric=self.metric,
                seed=42,
                n_jobs=self.n_jobs,
                estimator_list=estimator_list,
            )
        except Exception as exc:
            # If catboost or some other estimator isn't installed in the
            # search space, fall back to a smaller list.
            log.warning(
                "FLAML fit raised %s — retrying with a minimal estimator list",
                type(exc).__name__,
            )
            self.automl.fit(
                X_train=X, y_train=y,
                task=self.task,
                time_budget=self.time_budget,
                metric=self.metric,
                seed=42,
                n_jobs=self.n_jobs,
                estimator_list=["lgbm", "rf", "lrl2"],
            )
        try:
            self.model = self.automl.model.estimator
        except Exception:
            self.model = self.automl.model
        return self

    def predict(self, X):
        return self.automl.predict(X)

    def predict_proba(self, X):
        return self.automl.predict_proba(X)


# ---------------------------------------------------------------------------
# AutoML — TPOT (genetic-programming pipeline search)
# ---------------------------------------------------------------------------
def tpot(
    task: str,
    n_jobs: int = 4,
    generations: int = 5,
    population_size: int = 20,
    max_time_mins: int | None = 15,   # round 35: was 5
    cv: int = 3,                       # round 35: was 5
    subsample: float = 0.2,            # round 35: was 1.0 (full data)
    early_stop: int = 3,               # round 35: halt if no improvement
    **kwargs,
) -> Any:
    """TPOT classifier wrapper.

    TPOT searches scikit-learn pipelines via genetic programming; the
    object exposes ``.fit`` / ``.predict`` / ``.predict_proba`` once
    fit, but the underlying search is `population_size * generations`
    pipeline evaluations, which is expensive.

    Round 35 — speed tuning for NTDB-scale data (1.44M training rows):
      - ``subsample=0.2``: GA search runs on 20% of data (~288k rows).
        Genetic search only needs relative rankings between candidate
        pipelines, not perfect quality estimates; subsampling preserves
        the rankings while reducing each candidate's fit cost ~5×.
        The winning pipeline is then re-fit on the FULL training set
        at the end of TPOT's search, so the deployed model sees all
        the data.
      - ``cv=3``: 3-fold cross-validation inside the GA instead of 5.
        Loses a small amount of evaluation noise robustness; gains
        1.67× search speed.  With 1.44M rows even 3-fold is plenty.
      - ``max_time_mins=15``: bumped from 5.  Since each candidate
        is now cheaper (subsample + 3-fold), 15 min explores ~3× more
        candidates than the previous 5-min cap explored at full data.
      - ``early_stop=3``: halt the GA if the best CV score doesn't
        improve across 3 consecutive generations.  Saves time when
        the search has converged; rarely hurts because TPOT's GA
        plateaus quickly.

    Pass overrides as kwargs to the factory if you want different
    behaviour for a specific run, e.g. ``max_time_mins=None`` for
    unlimited search on a small grid.

    Survival is not supported by TPOT.
    """
    try:
        from tpot import TPOTClassifier
    except ImportError:
        return _require("tpot", "automl")()
    if task == "survival":
        raise NotImplementedError(
            "TPOT does not support survival targets; use cox_ph or "
            "random_survival_forest instead."
        )
    log.info(
        "tpot factory: generations=%d, population_size=%d, "
        "max_time_mins=%s, cv=%d, subsample=%.2f, early_stop=%d, n_jobs=%d",
        generations, population_size, max_time_mins, cv, subsample,
        early_stop, n_jobs,
    )
    return _TPOTWrapper(
        generations=generations,
        population_size=population_size,
        max_time_mins=max_time_mins,
        cv=cv,
        subsample=subsample,
        early_stop=early_stop,
        n_jobs=n_jobs,
        scoring="roc_auc" if task == "binary" else "accuracy",
    )


class _TPOTWrapper:
    """sklearn-like facade around TPOTClassifier.

    Why a wrapper: TPOT's TPOTClassifier IS sklearn-compatible after
    fit(), but exposing it directly would force the trainer to call
    .export() to get a usable artifact.  The wrapper saves the best
    fitted_pipeline_ (a normal sklearn Pipeline) into ``self.model``
    so ``predict_proba`` / pickling work like any other estimator and
    the persistence layer doesn't need TPOT-specific code.
    """
    def __init__(self, generations, population_size, max_time_mins,
                  cv, subsample, early_stop,
                  n_jobs, scoring):
        self.generations     = generations
        self.population_size = population_size
        self.max_time_mins   = max_time_mins
        self.cv              = cv
        self.subsample       = subsample
        self.early_stop      = early_stop
        self.n_jobs          = n_jobs
        self.scoring         = scoring
        self.tpot_           = None
        self.model           = None   # the best fitted sklearn Pipeline

    def fit(self, X, y):
        import numpy as np
        from tpot import TPOTClassifier

        # ── Round 35: subsample for the GA search ────────────────────────
        # TPOT 1.x has no native subsample param.  We do it manually:
        # pass a stratified subsample to TPOT for the genetic search,
        # then re-fit the WINNING pipeline on the full data at the end.
        # This preserves pipeline rankings (what GA cares about) while
        # making each candidate evaluation ~1/subsample faster.
        if 0 < self.subsample < 1.0 and len(X) > 10_000:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=self.subsample, random_state=42,
            )
            try:
                idx, _ = next(sss.split(X, y))
                X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
                y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
                log.info(
                    "TPOT subsample %.0f%%: GA will search on %d rows "
                    "(full %d) — winner re-fit on full data at end",
                    100 * self.subsample, len(idx), len(X),
                )
            except ValueError as exc:
                # Stratification can fail if a target class is too rare for
                # the subsample to contain at least 2 examples; fall back.
                log.warning("Subsample stratification failed (%s); using "
                            "full data for GA search", exc)
                X_sub, y_sub = X, y
        else:
            X_sub, y_sub = X, y

        # ── TPOT API detection (0.12.x vs 1.x) ───────────────────────────
        common = dict(
            n_jobs=self.n_jobs,
            random_state=42,
            verbose=2,
        )
        if self.max_time_mins is not None:
            common["max_time_mins"] = self.max_time_mins
        # early_stop: only TPOT 0.12.x supports this directly; 1.x uses
        # its own GA convergence criteria.  We try to pass it; if rejected,
        # the fallback path will omit it.
        try:
            self.tpot_ = TPOTClassifier(
                scorers=[self.scoring],
                scorers_weights=[1.0],
                search_space="linear",
                cv=self.cv,
                **common,
            )
            log.info("Using TPOT 1.x API (scorers=, search_space=, cv=%d)",
                     self.cv)
        except TypeError:
            log.info("Falling back to TPOT 0.12.x API (scoring=, generations=, "
                     "cv=%d, early_stop=%d)", self.cv, self.early_stop)
            kwargs = dict(
                generations=self.generations,
                population_size=self.population_size,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=42,
                verbosity=2,
                cv=self.cv,
                early_stop=self.early_stop,
            )
            if self.max_time_mins is not None:
                kwargs["max_time_mins"] = self.max_time_mins
            self.tpot_ = TPOTClassifier(**kwargs)

        self.tpot_.fit(X_sub, y_sub)
        # Both API generations expose the best fitted pipeline as
        # ``fitted_pipeline_``; this is a regular sklearn Pipeline that
        # works for predict_proba / pickling.
        best_pipeline = self.tpot_.fitted_pipeline_

        # ── Round 35: re-fit the winning pipeline on the FULL data ───────
        # The GA's `fitted_pipeline_` was fit only on the subsample.  For
        # final inference we want the same pipeline architecture trained
        # on all rows, so its decisions reflect the full data
        # distribution (not just 20% of it).  Clone the architecture and
        # re-fit on (X, y).  This adds one model fit's wall-clock but
        # preserves the full benefit of the data we have.
        if 0 < self.subsample < 1.0 and len(X) > 10_000:
            from sklearn.base import clone
            try:
                refit = clone(best_pipeline)
                log.info("Re-fitting winning pipeline on full data "
                         "(was fit on %d-row subsample, now %d rows)",
                         len(X_sub), len(X))
                refit.fit(X, y)
                self.model = refit
            except Exception as exc:
                log.warning("Re-fit on full data failed (%s); keeping "
                            "subsample-fit pipeline", exc)
                self.model = best_pipeline
        else:
            self.model = best_pipeline
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ---------------------------------------------------------------------------
# Foundation model — TabPFN (+ TabPFN 2.5 alias)
# ---------------------------------------------------------------------------
def tabpfn(task: str, n_jobs: int = 4, use_gpu: str = "auto", **kwargs) -> Any:
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return _require("tabpfn", "full")()
    if task != "binary":
        log.warning("TabPFN works best in binary settings; multiclass supported up to 10 classes")
    device = _resolve_device(use_gpu)
    log.info("tabpfn factory: device=%s", device)
    # TabPFN expects N <= 10k for v2.x; caller is responsible for subsampling
    return TabPFNClassifier(
        device=kwargs.get("device", device),
        N_ensemble_configurations=kwargs.get("ensembles", 32),
    )


# ---------------------------------------------------------------------------
# DL tabular — TabNet
# ---------------------------------------------------------------------------
def tabnet(task: str, n_jobs: int = 4, use_gpu: str = "auto", **kwargs) -> Any:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        return _require("pytorch_tabnet", "full")()
    device = _resolve_device(use_gpu)
    log.info("tabnet factory: device=%s", device)
    raw = TabNetClassifier(
        n_d=kwargs.get("n_d", 16),
        n_a=kwargs.get("n_a", 16),
        n_steps=kwargs.get("n_steps", 3),
        gamma=1.3,
        n_independent=2, n_shared=2,
        device_name=device,
        seed=42,
    )
    return _TabNetDataFrameAdapter(raw)


class _TabNetDataFrameAdapter:
    """Round 33: wrap TabNetClassifier so the rest of the pipeline can pass
    DataFrames without thinking about TabNet's numpy-only constraint.

    pytorch_tabnet rejects DataFrames in .fit / .predict / .predict_proba;
    every other family in our zoo (sklearn, xgboost, lightgbm, catboost,
    flaml) accepts both.  This adapter calls .to_numpy() on input and is
    otherwise transparent — exposes `.classes_`, `.feature_importances_`,
    etc. via __getattr__ so SHAP / persistence / evaluation just work.

    Also handles label dtype: TabNet requires int64 labels.
    """
    _estimator_type = "classifier"   # so sklearn utilities accept it

    def __init__(self, wrapped):
        self._w = wrapped

    @staticmethod
    def _to_np(arr):
        if hasattr(arr, "to_numpy"):
            return arr.to_numpy()
        import numpy as _np
        return _np.asarray(arr)

    def fit(self, X, y, *args, **kwargs):
        import numpy as _np
        Xn = self._to_np(X)
        yn = self._to_np(y).astype(_np.int64)
        return self._w.fit(Xn, yn, *args, **kwargs)

    def predict(self, X):
        return self._w.predict(self._to_np(X))

    def predict_proba(self, X):
        return self._w.predict_proba(self._to_np(X))

    def __getattr__(self, name):
        # Forward unknown attribute access to the wrapped model so things
        # like `.classes_`, `.history`, `.feature_importances_` still work
        return getattr(self._w, name)


def ft_transformer(task: str, n_jobs: int = 4, **kwargs) -> Any:
    # Placeholder — FT-Transformer needs a custom training loop; to be wired up later.
    raise NotImplementedError("FT-Transformer scaffold not yet implemented in this release")


# ---------------------------------------------------------------------------
# Survival families (stubs; require scikit-survival / lifelines)
# ---------------------------------------------------------------------------
def cox_ph(task: str, **kwargs) -> Any:
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
    except ImportError:
        return _require("scikit-survival", "survival")()
    return CoxPHSurvivalAnalysis(alpha=kwargs.get("alpha", 1e-4))


def random_survival_forest(task: str, **kwargs) -> Any:
    try:
        from sksurv.ensemble import RandomSurvivalForest
    except ImportError:
        return _require("scikit-survival", "survival")()
    return RandomSurvivalForest(
        n_estimators=kwargs.get("n_estimators", 300),
        min_samples_leaf=kwargs.get("min_samples_leaf", 15),
        n_jobs=kwargs.get("n_jobs", 4),
        random_state=42,
    )


def deep_surv(task: str, **kwargs) -> Any:
    raise NotImplementedError(
        "DeepSurv scaffold: integrate pycox.models.CoxPH or sksurv.DeepSurv here"
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
FACTORIES = {
    "logistic_l1":           logistic_l1,
    "logistic_elasticnet":   logistic_elasticnet,
    "random_forest":         random_forest,
    "xgboost":               xgboost,
    "lightgbm":              lightgbm,
    "catboost":              catboost,
    "flaml":                 flaml,
    "tpot":                  tpot,
    "tabpfn":                tabpfn,
    "tabnet":                tabnet,
    "ft_transformer":        ft_transformer,
    "cox_ph":                cox_ph,
    "random_survival_forest": random_survival_forest,
    "deep_surv":             deep_surv,
}


def build_model(family: str, task: str, **kwargs) -> Any:
    """Instantiate a model from its family name.

    Parameters
    ----------
    family : one of FACTORIES
    task   : 'binary' | 'multiclass' | 'survival'
    **kwargs : forwarded to the factory
    """
    if task not in _TASK_KINDS:
        raise ValueError(f"Unknown task {task!r}. Must be one of {_TASK_KINDS}")
    if family not in FACTORIES:
        raise ValueError(f"Unknown model family {family!r}. Known: {sorted(FACTORIES)}")
    return FACTORIES[family](task=task, **kwargs)
