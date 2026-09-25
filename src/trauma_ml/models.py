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
    """Random forest with memory-efficient defaults.

    Round 36 — added ``max_samples=0.5``.  At 2.75M training rows and 500
    trees, the full-bootstrap RF held ~275 GB once fully built — well over
    the slurm allocation, leading to OOM-kills mid-training on the band
    targets.  Bootstrapping each tree on 50% of rows halves the per-tree
    memory footprint and halves the per-worker data copy without
    materially affecting the model's discrimination (variance reduction
    from bagging saturates by ~50% sample size on N=2.75M).

    Keep ``n_estimators=500`` to match Tran 2022's scale.  If you want to
    explore the variance-vs-walltime tradeoff, override via
    ``--n-estimators`` from the CLI (kwarg passthrough).
    """
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 500),
        max_depth=kwargs.get("max_depth", None),
        max_samples=kwargs.get("max_samples", 0.5),    # round 36
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
        # Round 36: for multiclass targets (ISS_band/NISS_band) the class
        # distribution is imbalanced (1.7M / 1.5M / 261k / 404k for
        # ISS_band).  Plain accuracy lets TPOT win by predicting the
        # majority class.  balanced_accuracy averages recall across
        # classes — closer to what we actually care about.
        scoring="roc_auc" if task == "binary" else "balanced_accuracy",
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
    """Build a TabPFN classifier that works across TabPFN versions.

    TabPFN's constructor changed between major versions: v1 took
    ``N_ensemble_configurations``, while v2+ (incl. 2.5 / 8.x) renamed it to
    ``n_estimators`` and added ``ignore_pretraining_limits``.  Passing a
    parameter the installed version doesn't know raises TypeError, so instead
    of hard-coding names we introspect the signature and pass only what is
    actually supported.

    TabPFN is also a *small-data* foundation model (pretraining limit ~10k
    rows).  Trauma cohorts are far larger, so the returned estimator is
    wrapped in an adapter that stratified-subsamples the training set down to
    ``tabpfn_max_train`` rows (default 10000) before fitting.  This keeps the
    family runnable and comparable; report the subsample size in the paper.
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return _require("tabpfn", "full")()
    if task != "binary":
        log.warning("TabPFN works best in binary settings; multiclass supported up to 10 classes")
    device = _resolve_device(use_gpu)

    import inspect
    try:
        supported = set(
            inspect.signature(TabPFNClassifier.__init__).parameters
        ) - {"self", "args", "kwargs"}
    except (TypeError, ValueError):       # pragma: no cover - exotic builds
        supported = set()

    ensembles = kwargs.get("ensembles", 32)
    # Candidate params, newest-first names; only supported ones are passed.
    candidates = {
        "device": kwargs.get("device", device),
        "n_estimators": ensembles,                 # TabPFN v2+
        "N_ensemble_configurations": ensembles,    # TabPFN v1
        "ignore_pretraining_limits": True,         # v2+: allow >10k rows
        "random_state": kwargs.get("random_state", 42),
        "n_jobs": n_jobs,
    }
    init_kwargs = {k: v for k, v in candidates.items() if k in supported}
    # Don't send both spellings of the ensemble-size parameter.
    if "n_estimators" in init_kwargs:
        init_kwargs.pop("N_ensemble_configurations", None)

    log.info("tabpfn factory: device=%s, init kwargs=%s",
             device, sorted(init_kwargs))
    clf = TabPFNClassifier(**init_kwargs)

    max_train = int(kwargs.get("tabpfn_max_train", 10000))
    return _TabPFNSubsampleAdapter(clf, max_train=max_train,
                                   random_state=kwargs.get("random_state", 42))


class _TabPFNSubsampleAdapter:
    """Cap TabPFN's training-set size (stratified) before delegating to it.

    TabPFN performs in-context inference and degrades / OOMs well before the
    millions of rows in NTDB.  We keep the full pipeline intact (imputation,
    augmentation, calibration all happen upstream) and only shrink what is
    handed to ``fit``.  Prediction is unaffected.
    """

    def __init__(self, estimator, max_train: int = 10000, random_state: int = 42):
        self.estimator = estimator
        self.max_train = max_train
        self.random_state = random_state

    # sklearn-compat plumbing -------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        return {"estimator": self.estimator, "max_train": self.max_train,
                "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def classes_(self):
        return self.estimator.classes_

    def __getattr__(self, item):
        # Delegate anything we don't define (e.g. fitted attributes).
        return getattr(self.__dict__["estimator"], item)

    # core API ---------------------------------------------------------------
    def fit(self, X, y, **fit_kwargs):
        import numpy as np
        n = len(y)
        if self.max_train and n > self.max_train:
            try:
                from sklearn.model_selection import train_test_split
                idx = np.arange(n)
                keep, _ = train_test_split(
                    idx, train_size=self.max_train,
                    random_state=self.random_state, stratify=y,
                )
            except Exception:                     # tiny/degenerate classes
                rng = np.random.default_rng(self.random_state)
                keep = rng.choice(n, size=self.max_train, replace=False)
            X = X.iloc[keep] if hasattr(X, "iloc") else X[keep]
            y = y.iloc[keep] if hasattr(y, "iloc") else y[keep]
            log.info("TabPFN: subsampled training set %d -> %d rows "
                     "(stratified, seed=%d)", n, len(y), self.random_state)
        self.estimator.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


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
        """Fit with pytorch_tabnet defaults tuned for NTDB-scale data.

        Round 37 — TabNet's default fit() runs max_epochs=100 with
        batch_size=1024 and no early stopping (because no eval_set is
        provided).  On 2.75M training rows that's ~95s/epoch × 100 = 2.5h
        per combo, leaving the loss plateaued for the last 40 epochs.
        With 144 combos in the mortality grid the slurm hits walltime
        before completing 40 models.

        Three changes:
          1. Carve a 5% stratified validation slice from the training
             data and pass it as ``eval_set``.  This activates
             pytorch_tabnet's patience-based early stopping.
          2. ``patience=10`` — halt if the eval metric (AUC for binary,
             accuracy for multiclass) doesn't improve in 10 consecutive
             epochs.  Empirically most combos plateau by epoch ~30-50.
          3. ``max_epochs=50`` — hard cap.  Even without patience
             triggering, 50 epochs is past the observed plateau point
             (epoch ~60 in the round-33 logs at the previous settings).
          4. ``batch_size=4096`` (was 1024).  The RTX 3090 was running
             at 16% utilisation with the small default — bigger batches
             keep the GPU pipeline full.  4× larger batch → ~4× fewer
             SGD steps per epoch → ~3× wall-clock speedup empirically
             after PyTorch overhead.
          5. ``virtual_batch_size=512`` (was 128) — matches the larger
             physical batch.  Affects only TabNet's GhostBatchNorm.

        Expected per-combo wall-clock: 20-30 min (was 2.5 h).  Full
        144-combo grid: ~50-70 hours, comfortably within xlong (8 d).

        Caller-supplied kwargs override these defaults (e.g. pass
        ``max_epochs=200`` for a single deep-train experiment).
        """
        import numpy as _np
        Xn = self._to_np(X)
        yn = self._to_np(y).astype(_np.int64)

        fit_kwargs: dict = dict(
            max_epochs=50,
            patience=10,
            batch_size=4096,
            virtual_batch_size=512,
            drop_last=False,
        )

        # Try to set up an eval_set for early stopping (only meaningful
        # if we have enough data + can stratify on the target).
        if len(Xn) > 50_000:
            try:
                from sklearn.model_selection import train_test_split
                X_tr, X_val, y_tr, y_val = train_test_split(
                    Xn, yn, test_size=0.05, stratify=yn, random_state=42,
                )
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["eval_metric"] = (
                    ["auc"] if len(_np.unique(yn)) == 2 else ["accuracy"]
                )
                Xn, yn = X_tr, y_tr
                log.info(
                    "TabNet: carved %d-row eval_set (5%%) for patience-based "
                    "early stopping; training on %d rows; "
                    "max_epochs=%d, patience=%d, batch_size=%d",
                    len(X_val), len(X_tr),
                    fit_kwargs["max_epochs"], fit_kwargs["patience"],
                    fit_kwargs["batch_size"],
                )
            except ValueError as exc:
                # Stratification can fail if a target class is too rare
                log.warning(
                    "TabNet eval_set stratification failed (%s); training "
                    "without early stopping at max_epochs=%d",
                    exc, fit_kwargs["max_epochs"],
                )

        # Caller overrides (rare; the trainer doesn't pass extra kwargs)
        fit_kwargs.update(kwargs)
        return self._w.fit(Xn, yn, *args, **fit_kwargs)

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
# Doshi-style feed-forward neural network (Round 52)
# ---------------------------------------------------------------------------
# Doshi et al. (2024) used a feed-forward network to map injury codes to ISS
# (ISS>=16 binary or exact ISS).  They did not publish a full architecture, so
# this is a faithful, reasonable reconstruction: a multi-layer perceptron with
# ReLU + BatchNorm + Dropout and early stopping.  Rather than feeding raw
# ICD-10-CM multi-hot vectors (which would need a separate feature pipeline),
# we feed the SAME tabular predictor matrix every other family receives — i.e.
# the Barell-based injury features plus physiology/demographics.  This is the
# pragmatic "go-around" the user asked for and keeps the FFNN comparable, on
# identical inputs, to the boosting/linear families for BOTH mortality and the
# ISS/NISS-band tasks.
#
# NOT NaN-native — requires an imputer upstream (do not pair with imputer=none).
class _DoshiFFNNClassifier:
    """Minimal sklearn-style classifier wrapping a PyTorch MLP.

    Exposes fit / predict / predict_proba / classes_ so it slots into the
    pipeline exactly like the TabNet adapter.  Works for binary and
    multiclass via a softmax output + cross-entropy loss with inverse-
    frequency class weights (helps the rare mortality positive class).
    """
    _estimator_type = "classifier"

    def __init__(self, task: str = "binary", hidden_dims=(256, 128, 64),
                 dropout: float = 0.3, lr: float = 1e-3, weight_decay: float = 1e-5,
                 max_epochs: int = 100, patience: int = 10, batch_size: int = 4096,
                 use_gpu: str = "auto", seed: int = 42, l1_lambda: float = 1e-4,
                 icd_col: str = "ICD_DIAG_CODES", icd_features: str = "off",
                 icd_max_vocab: int | None = None, icd_min_count: int = 1,
                 **kwargs):
        self.task = task
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.seed = seed
        self.l1_lambda = l1_lambda
        # Faithful-Doshi ICD options:
        #   icd_features = "off"  -> ignore ICD codes, use the numeric matrix
        #                            (legacy behaviour);
        #                  "only" -> use ONLY the multi-hot ICD vector (the
        #                            faithful Doshi ICD->severity FFNN);
        #                  "plus" -> concatenate ICD multi-hot WITH the other
        #                            (L3) numeric features (ablation for gain).
        self.icd_col = icd_col
        self.icd_features = icd_features
        self.icd_max_vocab = icd_max_vocab
        self.icd_min_count = icd_min_count
        self._icd_vocab = None          # list[str]; set on fit when ICD used
        self._extra = kwargs
        self.classes_ = None
        self._net = None
        self._n_features = None

    @staticmethod
    def _to_np(arr):
        if hasattr(arr, "to_numpy"):
            return arr.to_numpy()
        return np.asarray(arr)

    # ---- Faithful-Doshi ICD handling --------------------------------- #
    def _uses_icd(self, X) -> bool:
        return (self.icd_features in ("only", "plus")
                and hasattr(X, "columns") and self.icd_col in X.columns)

    def _fit_icd_vocab(self, code_lists) -> None:
        """Build the ICD code vocabulary from the training code-list strings:
        keep codes seen in >= icd_min_count patients, then the top
        icd_max_vocab by document frequency."""
        from collections import Counter
        cnt = Counter()
        for s in code_lists:
            if isinstance(s, str) and s:
                cnt.update(set(s.split()))
        kept = [(c, n) for c, n in cnt.items() if n >= self.icd_min_count]
        kept.sort(key=lambda kv: (-kv[1], kv[0]))
        if self.icd_max_vocab and len(kept) > self.icd_max_vocab:
            kept = kept[:self.icd_max_vocab]
        self._icd_vocab = [c for c, _ in kept]
        self._icd_index = {c: i for i, c in enumerate(self._icd_vocab)}
        log.info("doshi_ffnn[icd]: vocabulary = %d codes (min_count=%d, cap=%s)",
                 len(self._icd_vocab), self.icd_min_count, self.icd_max_vocab)

    def _icd_multihot(self, code_lists):
        """Return an (n × |vocab|) scipy CSR multi-hot matrix."""
        from scipy.sparse import csr_matrix
        idx = self._icd_index
        rows, cols = [], []
        for r, s in enumerate(code_lists):
            if isinstance(s, str) and s:
                for code in set(s.split()):
                    j = idx.get(code)
                    if j is not None:
                        rows.append(r); cols.append(j)
        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(code_lists), len(self._icd_vocab)),
                          dtype=np.float32)

    def _build_inputs(self, X, *, fit: bool):
        """Return (numeric_dense [n×d] float32 or None, icd_csr [n×K] or None).

        - icd_features='only': numeric=None, icd=multi-hot
        - icd_features='plus': numeric=other cols, icd=multi-hot
        - else                : numeric=all cols, icd=None  (legacy)
        """
        if self._uses_icd(X):
            code_lists = X[self.icd_col].astype(str).tolist()
            if fit:
                self._fit_icd_vocab(code_lists)
            icd = self._icd_multihot(code_lists)
            if self.icd_features == "only":
                return None, icd
            num = X.drop(columns=[self.icd_col])
            return self._to_np(num).astype(np.float32), icd
        # legacy numeric-only path (drop ICD col if present but unused)
        if hasattr(X, "columns") and self.icd_col in X.columns:
            X = X.drop(columns=[self.icd_col])
        return self._to_np(X).astype(np.float32), None

    @staticmethod
    def _batch_dense(num, icd, idx):
        """Assemble a dense float32 minibatch from numeric array + sparse ICD."""
        parts = []
        if num is not None:
            parts.append(num[idx])
        if icd is not None:
            parts.append(icd[idx].toarray())
        if len(parts) == 1:
            return parts[0]
        return np.hstack(parts)

    def _n_in(self, num, icd) -> int:
        return (0 if num is None else num.shape[1]) + (0 if icd is None else icd.shape[1])

    def _build_net(self, n_features: int, n_classes: int):
        import torch.nn as nn
        layers = []
        prev = n_features
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        return nn.Sequential(*layers)

    def fit(self, X, y, *args, **kwargs):
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = _resolve_device(self.use_gpu)
        self._device = device

        num, icd = self._build_inputs(X, fit=True)
        if num is not None and np.isnan(num).any():
            log.warning("doshi_ffnn: numeric input has NaN after imputation; "
                        "zero-filling as a last resort.")
            num = np.nan_to_num(num, nan=0.0)
        yn = self._to_np(y).astype(np.int64)

        self.classes_ = np.array(sorted(np.unique(yn)))
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        yn = np.array([class_to_idx[v] for v in yn], dtype=np.int64)
        self._n_features = self._n_in(num, icd)
        n = len(yn)

        # Inverse-frequency class weights (helps rare positives).
        counts = np.bincount(yn, minlength=n_classes).astype(np.float64)
        weights = (counts.sum() / np.maximum(counts, 1.0))
        weights = weights / weights.sum() * n_classes
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

        # Stratified early-stopping split on ROW INDICES (sparse-safe).
        try:
            tr_idx, val_idx = train_test_split(
                np.arange(n), test_size=0.1, stratify=yn, random_state=self.seed)
        except ValueError:
            tr_idx, val_idx = np.arange(n), np.arange(min(1, n))

        self._net = self._build_net(self._n_features, n_classes).to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)

        ytr_t = torch.tensor(yn[tr_idx], device=device)
        Xval = self._batch_dense(num, icd, val_idx)
        Xval_t = torch.tensor(Xval, device=device)
        yval_t = torch.tensor(yn[val_idx], device=device)

        n_tr = len(tr_idx)
        best_val = float("inf"); best_state = None; bad = 0
        for epoch in range(self.max_epochs):
            self._net.train()
            perm = np.random.permutation(n_tr)
            for i in range(0, n_tr, self.batch_size):
                b = perm[i:i + self.batch_size]
                if len(b) < 2:
                    continue  # BatchNorm1d needs >1 sample
                Xb = self._batch_dense(num, icd, tr_idx[b])
                Xb_t = torch.tensor(Xb, device=device)
                opt.zero_grad()
                out = self._net(Xb_t)
                loss = loss_fn(out, ytr_t[b])
                # L1 on the first linear layer -> soft input feature selection.
                if self.l1_lambda and self.l1_lambda > 0:
                    loss = loss + self.l1_lambda * self._net[0].weight.abs().sum()
                loss.backward()
                opt.step()
            self._net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self._net(Xval_t), yval_t).item())
            if vloss < best_val - 1e-4:
                best_val = vloss
                best_state = {k: v.detach().clone()
                              for k, v in self._net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    log.info("doshi_ffnn: early stop at epoch %d (val_loss=%.4f)",
                             epoch, best_val)
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        num, icd = self._build_inputs(X, fit=False)
        if num is not None and np.isnan(num).any():
            num = np.nan_to_num(num, nan=0.0)
        n = (num.shape[0] if num is not None else icd.shape[0])
        self._net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, n, self.batch_size):
                idx = np.arange(i, min(i + self.batch_size, n))
                Xb = self._batch_dense(num, icd, idx)
                logits = self._net(torch.tensor(Xb, device=self._device))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def doshi_ffnn(task: str, use_gpu: str = "auto", **kwargs) -> Any:
    """Factory for the Doshi-style feed-forward network.

    Available for 'binary' (mortality, ISS/NISS-band binary) and 'multiclass'
    (ISS/NISS-band).  Requires torch; if torch is unavailable, raises a clear
    install hint pointing at the [full] extra.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return _require("torch", "full")()
    return _DoshiFFNNClassifier(task=task, use_gpu=use_gpu, **kwargs)


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
    "doshi_ffnn":            doshi_ffnn,
    # Faithful Doshi ICD->severity FFNN variants (multi-hot ICD code input):
    #   _icd      = ICD codes ONLY (the paper's direct ICD->ISS model);
    #   _icd_plus = ICD multi-hot + the other L3 features (ablation for gain).
    "doshi_ffnn_icd":        lambda task, **kw: doshi_ffnn(task, icd_features="only", **kw),
    "doshi_ffnn_icd_plus":   lambda task, **kw: doshi_ffnn(task, icd_features="plus", **kw),
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
