"""Trainer — the orchestrator.

Changes vs v0.1
---------------
* ``holdout_years`` : most-recent NTDB year(s) are extracted as a *temporal*
  holdout **before** any random split, so the non-holdout data is what gets
  split into train/calibration/test.  The temporal holdout is evaluated
  separately after training.
* ``__admission_year`` is always added to the stratification columns so year
  distribution is preserved across the three random partitions.
* Baseline metrics (ISS ≥ 16, NISS ≥ 16, TRISS < 0.50) are computed and saved
  alongside model metrics.
* Diagnostic plots (ROC+PR, confusion matrices, SHAP) are generated per model.
* ``year`` is added to ``subgroup_axes`` so per-year metrics are always saved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, RobustScaler

from .baselines import baseline_metrics, compute_baseline_scores, compute_triss
from .catalogue import Catalogue
from .evaluation import (
    evaluate, save_metrics, save_subgroup_metrics, subgroup_metrics,
    subgroup_metrics_for_score, binary_metrics,
)
from .imputation import evaluate_imputation, make_imputer
from .inclusion import get_strategy
from .models import build_model
from .persistence import ModelArtifact
from .plots import save_model_plots
from .splitting import assign_age_group, extract_holdout_years, stratified_split
from .targets import TargetSpec, build_target, encode_classes

log = logging.getLogger(__name__)


def _vectorised_label_encode(vals: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Encode an array of string-coerced values against ``classes`` (the
    fitted LabelEncoder's classes_ array) without per-element sklearn calls.

    Unknown values map to -1 (matches the old per-row behaviour).
    Vectorised with np.searchsorted; ~30-100× faster than looping
    le.transform() per row, which is the difference between 28 minutes
    and 30 seconds on the 1.44M × 46 col user grid.
    """
    classes = np.asarray(classes)
    # Sorted-position lookup for each val (left); then verify equality
    # to detect unknowns (positions clamped to len(classes)-1 first).
    pos = np.searchsorted(classes, vals)
    pos_clipped = np.clip(pos, 0, len(classes) - 1)
    matched = (classes[pos_clipped] == vals) if len(classes) > 0 else np.zeros_like(vals, dtype=bool)
    encoded = np.where(matched, pos_clipped, -1).astype(np.int64)
    return encoded


@dataclass
class TrainerConfig:
    """Serialisable run configuration."""
    model_id: str
    dataset_path: Path
    catalogue: Catalogue
    target: TargetSpec
    predictor_type: str = "all"
    phase_cutoff: str = "On-scene + ED arrival + In-hospital"
    inclusion_strategy: str | None = None
    imputer_method: str = "median_mode"
    model_family: str = "xgboost"
    calibration: str = "none"
    missingness_threshold: float = 0.50
    data_augmentation: str | None = None
    correlation_threshold: float = 0.80
    holdout_dataset_path: Path | None = None   # external file holdout (legacy)
    holdout_years: list[int] = field(default_factory=list)  # temporal year holdout
    sample_fraction: float = 1.0
    age_bins_edges: list[int] = field(default_factory=lambda: [0, 18, 45, 65, 75, 200])
    age_bins_labels: list[str] = field(default_factory=lambda: [
        "pediatric", "young_adult", "middle_aged", "early_elderly", "elderly"
    ])
    # year axis is added automatically; keep the rest here for reference
    subgroup_axes: list[str] = field(default_factory=lambda: [
        "gender", "age_group", "__admission_year",
        "ethnicity", "race", "mechanism", "intent",
    ])
    generate_plots: bool = True
    # Round 16: SHAP can segfault on glibc<2.28 with large RF / boosting
    # ensembles.  Set enable_shap=False on heavy-grid slurms.
    enable_shap: bool = True
    random_state: int = 42
    n_jobs: int = 4
    # round 12: GPU policy for model factories that support it
    # ("auto" — use GPU if available, "force" — require GPU, "never" — CPU)
    use_gpu: str = "auto"
    # round 26: hyperparameter tuning via internal RandomizedSearchCV.
    # When True, each model fit is wrapped in a randomized search of N
    # points × cv_folds folds.  Tran 2022 did this and found "negligible
    # impact"; default is False to keep walltimes manageable.
    tune_hparams: bool = False
    cv_folds: int = 3
    n_search_iter: int = 10
    # Round 37: post-imputation feature gating.
    # When True, the trainer evaluates imputer quality per feature on the
    # CALIBRATION SET (never the test set — kept untouched for final
    # evaluation), then drops features whose imputation fails the
    # ``imputer_check_thresholds``.  If no features survive, all
    # downstream metrics are emitted as NaN and training is skipped.
    # Combined with ``imputer_method="none"`` (which lets NaN-tolerant
    # tree models consume the raw data), this gives three regimes:
    #   imputer_method != "none", imputer_check = False  → impute all
    #   imputer_method != "none", imputer_check = True   → impute, drop bad
    #   imputer_method == "none",  imputer_check = *      → skip imputation
    imputer_check: bool = False
    imputer_check_thresholds: dict = field(default_factory=lambda: {
        # numeric: keep if MAE / std(train) < 0.5  (imputer at least
        # halves uncertainty vs a "predict the variance" baseline)
        "numeric_relative_mae_max": 0.5,
        # categorical: keep if reconstruction accuracy > 0.7
        "categorical_accuracy_min": 0.7,
    })


# Mapping predictor_type label → Catalogue registry column
PREDICTOR_TYPE_TO_REGISTRY = {
    "all":                 None,
    "Tran NTDB":           "Tran NTDB",
    "Karolinska SweTrau":  "Karolinska SweTrau",
    "RETRAUCI":            "RETRAUCI",
    "ECTrauma":            "ECTrauma",
    "EIPD":                "EIPD",
}


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.data: pd.DataFrame | None = None
        self.splits: dict[str, pd.Index] | None = None
        self.holdout_idx: pd.Index | None = None   # temporal year holdout indices
        self.predictors: list[str] | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.transformers: dict[str, dict] = {"encoders": {}, "scalers": {}}
        self.imputer = None
        self.model = None
        self.calibrated_models: dict[str, Any] = {}
        self.target_inverse: dict[int, Any] = {}
        self.y_all: np.ndarray | None = None
        self.task: str = "binary"
        self.classes: list | None = None

    # ================================================================= #
    # Orchestration
    # ================================================================= #
    def run(self, outputs_root: Path) -> ModelArtifact:
        outputs_root = Path(outputs_root)
        self._load_and_filter()
        self._build_target()
        self._select_predictors()
        self._drop_high_missingness()
        self._split()                          # extracts temporal holdout first
        self._snapshot_raw_baseline_cols()     # save ISS/NISS/TRISS before scaling
        self._snapshot_raw_missingness()       # save predictor NaN mask before imputation
        self._fit_transformers_on_train()
        self._apply_transformers()
        self._fit_imputer_and_eval(outputs_root / "imputation_eval" / self.cfg.model_id)

        # Round 37: imputer_check may have dropped every feature.  If so,
        # there's nothing to model — emit NaN metrics so the resume tracker
        # treats this combo as "done" (and won't retry), then return early.
        if getattr(self, "_imputer_check_no_features", False):
            log.warning("[%s] No predictors survived imputer_check — "
                        "writing NaN metrics and skipping training.",
                        self.cfg.model_id)
            return self._emit_empty_run(outputs_root)

        self._augment_if_needed()
        self._fit_model()
        self._calibrate_if_binary()
        self._evaluate(outputs_root / "metrics" / self.cfg.model_id, outputs_root)
        return self._save_artifact(outputs_root / "models")

    # ================================================================= #
    # Steps
    # ================================================================= #
    def _load_and_filter(self):
        log.info("Loading dataset from %s", self.cfg.dataset_path)
        self.data = pd.read_parquet(self.cfg.dataset_path)
        log.info("Loaded %d rows × %d cols", *self.data.shape)

        # Round 34: coalesce columns that share a name modulo case
        # (e.g. AGEYEARS / AGEyears / AgeYears).  The build's alias resolver
        # is supposed to do this at parquet-write time, but existing
        # parquets built before this fix carry the duplicates.  Doing it
        # here makes ALL training resilient to the old build artefact,
        # avoiding a full rebuild.
        self._coalesce_duplicate_cased_columns()

        if 0 < self.cfg.sample_fraction < 1.0:
            n_keep = int(len(self.data) * self.cfg.sample_fraction)
            self.data = self.data.sample(
                n=n_keep, random_state=self.cfg.random_state
            ).reset_index(drop=True)

        if self.cfg.inclusion_strategy:
            strat = get_strategy(self.cfg.inclusion_strategy)
            self.data, excl_log = strat.apply(self.data)
            log.info("Inclusion strategy %s: %s", strat.name, excl_log)

        # Derived columns ------------------------------------------------
        if "AGEYEARS" in self.data.columns:
            self.data["age_group"] = assign_age_group(
                self.data["AGEYEARS"],
                edges=self.cfg.age_bins_edges,
                labels=self.cfg.age_bins_labels,
            )
        race_cols = [c for c in ("WHITE", "BLACK", "ASIAN", "AMERICANINDIAN",
                                  "PACIFICISLANDER", "RACEOTHER")
                     if c in self.data.columns]
        if race_cols:
            race_labels = []
            arr = self.data[race_cols].fillna(0).astype(int).to_numpy()
            for row in arr:
                active = [c for c, v in zip(race_cols, row) if v == 1]
                race_labels.append("__multi__" if len(active) > 1
                                   else active[0] if active else "__unknown__")
            self.data["race"] = race_labels

        # Round 21: case-insensitive lookup for SEX/TRAUMATYPE/INTENT
        # because the parquet may have 'Sex', 'sex', 'TraumaType' etc.
        col_lower_to_actual = {c.lower(): c for c in self.data.columns}

        if "gender" not in self.data.columns:
            sex_actual = col_lower_to_actual.get("sex")
            if sex_actual is not None:
                self.data["gender"] = self.data[sex_actual].map({1: "Male", 2: "Female"})
            else:
                # Fallback: blank gender column so downstream stratify+drop don't crash
                self.data["gender"] = pd.NA
                log.warning(
                    "Neither 'SEX' nor 'gender' present in parquet — "
                    "skipping gender stratification.  Columns starting with "
                    "'S': %s",
                    sorted(c for c in self.data.columns if c.upper().startswith("S"))[:10],
                )
        if "mechanism" not in self.data.columns:
            tt_actual = col_lower_to_actual.get("traumatype")
            if tt_actual is not None:
                self.data["mechanism"] = self.data[tt_actual]
        if "intent" not in self.data.columns:
            it_actual = col_lower_to_actual.get("intent")
            if it_actual is not None:
                self.data["intent"] = self.data[it_actual]

        mask_valid = self.data["gender"].notna()
        n_dropped = (~mask_valid).sum()
        self.data = self.data.loc[mask_valid].copy()
        if n_dropped:
            log.info("Dropped %d rows due to missing 'gender'", n_dropped)

    def _coalesce_duplicate_cased_columns(self) -> None:
        """Merge columns that share a name modulo case.

        Round 34: NTDB parquets built before round-34's loader fix can
        carry triplicated columns like ``AGEYEARS``, ``AGEyears``, and
        ``AgeYears`` — because the per-year alias resolver only checked
        whether the canonical name existed, missing other-cased duplicates.

        When ``_select_predictors`` later builds a lowercase→actual lookup
        for case-insensitive catalogue matching, only ONE of the variants
        wins (whichever the dict comprehension processed last); the others
        are invisible.  If that winner is the variant with high NaN rate,
        ``_drop_high_missingness`` then deletes it and the column appears
        nowhere in the model.

        Fix: detect duplicate-cased columns, coalesce their values per row
        (first non-null wins), promote the all-upper-case variant as
        canonical (or whichever variant we encountered first), drop the
        rest.  Idempotent — safe to call on already-clean parquets.
        """
        seen_lower: dict[str, str] = {}
        groups: dict[str, list[str]] = {}
        for c in self.data.columns:
            cl = c.lower()
            if cl in seen_lower:
                groups.setdefault(cl, [seen_lower[cl]]).append(c)
            else:
                seen_lower[cl] = c

        if not groups:
            return

        for cl, variants in groups.items():
            non_null_before = {v: int(self.data[v].notna().sum())
                               for v in variants}
            # Coalesce — first variant's non-null values win, fill from others
            coalesced = self.data[variants[0]]
            for v in variants[1:]:
                coalesced = coalesced.fillna(self.data[v])

            # Promote the all-upper-case canonical if any, else first variant
            canonical_name = next((v for v in variants if v.isupper()),
                                   variants[0])
            self.data[canonical_name] = coalesced

            for v in variants:
                if v != canonical_name and v in self.data.columns:
                    self.data = self.data.drop(columns=[v])

            log.info(
                "Coalesced duplicate-cased columns -> %s "
                "(before: %s; after: %d non-null)",
                canonical_name,
                ", ".join(f"{k}={v}" for k, v in non_null_before.items()),
                int(self.data[canonical_name].notna().sum()),
            )

    def _build_target(self):
        y = build_target(self.data, self.cfg.target)
        self.data = self.data.loc[y.notna()].reset_index(drop=True)
        y = y.loc[y.notna()].reset_index(drop=True)
        if self.cfg.target.kind == "binary":
            self.task = "binary"
            self.y_all = y.astype(int).to_numpy()
            self.target_inverse = {0: "negative", 1: "positive"}
            self.classes = [0, 1]
        elif self.cfg.target.kind == "ordinal_bands":
            self.task = "multiclass"
            y_int, inverse = encode_classes(y)
            self.y_all = y_int.astype(int)
            self.target_inverse = inverse
            self.classes = sorted(inverse.keys())
        else:
            raise ValueError(f"Unsupported target kind {self.cfg.target.kind!r}")
        log.info("Target %s built (task=%s, N=%d, class counts=%s)",
                 self.cfg.target.name, self.task, len(self.y_all),
                 dict(pd.Series(self.y_all).value_counts()))

    def _select_predictors(self):
        registry = PREDICTOR_TYPE_TO_REGISTRY.get(self.cfg.predictor_type, None)
        whitelist = self.cfg.catalogue.variables_for(
            phase_cutoff=self.cfg.phase_cutoff,
            registries=[registry] if registry else None,
            include_target_derivers=False,
        )
        target_cols = set()
        if self.cfg.target.kind == "binary":
            target_cols.update({"HOSPDISCHARGEDISPOSITION", "EDDISCHARGEDISPOSITION",
                                 "DEATHINED"})
        elif self.cfg.target.kind == "ordinal_bands":
            source = self.cfg.target.spec.get("variable") or self.cfg.target.spec.get("derive_from")
            target_cols.add(source)
            target_cols.update({"ISS", "ISS_05", "NISS"})

        # Round 18 defense-in-depth: even though NON_PREDICTOR_COLUMNS is
        # already applied inside variables_for(), some entry points (custom
        # cohort defs, future code paths) might bypass that filter.  Apply
        # the same blacklist here so the trainer NEVER trains on row IDs
        # or hospital-level admin columns regardless of how the whitelist
        # was constructed.  Round 19: case-insensitive comparison.
        from .catalogue import is_non_predictor

        # Round 20: case-insensitive match between the catalogue whitelist
        # (canonical uppercase names) and the parquet's actual column names
        # (which may be 'Sex', 'sex', 'AgeYears' depending on year/source).
        # The loader's alias-resolver handles a few clinical canonicals but
        # not every catalogue entry.  We do the case-insensitive intersection
        # here so 'SEX' from the catalogue matches 'sex' or 'Sex' in the
        # parquet, then resolve to the actual parquet column name.
        col_lower_to_actual = {c.lower(): c for c in self.data.columns}
        target_cols_lower = {c.lower() for c in target_cols}

        self.predictors = []
        for canonical in whitelist:
            if is_non_predictor(canonical):
                continue
            if canonical.lower() in target_cols_lower:
                continue
            actual = col_lower_to_actual.get(canonical.lower())
            if actual is None:
                continue
            self.predictors.append(actual)

        log.info("Selected %d predictors (whitelist=%d, parquet has %d cols)",
                 len(self.predictors), len(whitelist), len(self.data.columns))
        if len(self.predictors) == 0:
            log.error(
                "No predictors selected.  Catalogue whitelist (post-blacklist) "
                "for phase_cutoff=%r registry=%r: %s. Parquet columns starting "
                "with 'A'/'G'/'I'/'N'/'S'/'R'/'T': %s",
                self.cfg.phase_cutoff, registry,
                whitelist[:30],
                sorted(c for c in self.data.columns
                       if c.upper().startswith(("A", "G", "I", "N", "S", "R", "T")))[:30],
            )
        # Sanity check: log if any blacklisted column slipped into the data
        # so we get a loud warning rather than silent leakage.
        leaked = {c for c in (set(self.data.columns) & set(whitelist))
                   if is_non_predictor(c)}
        if leaked:
            log.warning(
                "Blacklisted columns %s were in the catalogue whitelist "
                "but EXCLUDED from predictors (round 18+19 NON_PREDICTOR_COLUMNS).",
                sorted(leaked),
            )

    def _drop_high_missingness(self):
        ratios = self.data[self.predictors].isna().mean()
        keep = ratios[ratios <= self.cfg.missingness_threshold].index.tolist()
        dropped = sorted(set(self.predictors) - set(keep))
        if dropped:
            log.info("Dropped %d high-missingness predictors: %s",
                     len(dropped), dropped[:12])
        self.predictors = keep

    def _split(self):
        """Extract temporal holdout years first, then stratify the remainder."""
        if self.cfg.holdout_years:
            non_ho_idx, ho_idx = extract_holdout_years(
                self.data, self.cfg.holdout_years
            )
            self.holdout_idx = ho_idx
            work_df = self.data.loc[non_ho_idx]
            work_y  = pd.Series(self.y_all, index=self.data.index).loc[non_ho_idx]
        else:
            work_df = self.data
            work_y  = pd.Series(self.y_all, index=self.data.index)
            self.holdout_idx = pd.Index([], dtype=self.data.index.dtype)

        stratify_by = []
        if "gender" in work_df.columns:
            stratify_by.append("gender")
        if "age_group" in work_df.columns:
            stratify_by.append("age_group")
        # year is added automatically inside stratified_split

        self.splits = stratified_split(
            work_df,
            work_y,
            stratify_by=stratify_by,
            random_state=self.cfg.random_state,
        )

    # Columns needed for ISS/NISS/TRISS — must be saved before RobustScaler
    _BASELINE_COLS = (
        "ISS", "ISS_05", "NISS",
        "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "AGEYEARS", "TRAUMATYPE",
    )

    # ----------------------------------------------------------------- #
    # Cohort definitions — explicit variable lists, NOT phase-based.
    # ----------------------------------------------------------------- #
    # Rationale: defining "on-site complete" as "every catalogue variable
    # with phase=On-scene non-NaN" is too aggressive — many such variables
    # are sparsely populated even in well-coded NTDB rows, and a single
    # rare-NaN column can drop the cohort from 70% to <1% of the dataset.
    #
    # We instead use the variable set used by Tran et al (PLoS One 2022,
    # doi:10.1371/journal.pone.0276624) — the most directly comparable
    # NTDB ML baseline study for our project.  Tran's "base" model uses
    # five inputs (SBP, RR, GCS, age, mechanism) that are also the inputs
    # to TRISS, making the cohort interpretable as "rows where the TRISS-
    # equivalent feature set is fully observed".
    #
    # ed_complete adds the next-tier ED-arrival physiology that Tran
    # references in supplementary table S1 of the same paper (temperature,
    # pulse oximetry, pulse rate / heart rate).
    #
    # baseline_complete = inputs needed to compute the clinical baselines
    # (ISS, NISS, TRISS) without any imputation.
    # ----------------------------------------------------------------- #
    _COHORT_VAR_DEFS: dict[str, tuple[str, ...]] = {
        # Tran et al "base" model variables — on-scene physiology +
        # demographics + mechanism.  All five are inputs to TRISS.
        "onsite_complete": (
            "AGEYEARS",     # demographics (always knowable)
            "TRAUMATYPE",   # mechanism (blunt vs penetrating)
            "GCSTOTAL",     # on-scene neurology
            "SBPFIRST",     # on-scene haemodynamics
            "RRFIRST",      # on-scene respiratory
        ),
        # ED-arrival physiology added on top of the on-scene set.
        # Optional vars that are missing from the parquet are dropped
        # at resolution time — see _resolve_cohort_vars().
        "ed_complete": (
            "AGEYEARS", "TRAUMATYPE", "GCSTOTAL", "SBPFIRST", "RRFIRST",
            "TEMPERATURE",   # ED-arrival temperature
            "PULSEOXIMETRY", # ED-arrival oxygen saturation
            "PULSERATE",     # ED-arrival heart rate
        ),
        # Inputs needed for ISS/NISS/TRISS without imputation
        "baseline_complete": (
            "ISS", "NISS",
            "GCSTOTAL", "SBPFIRST", "RRFIRST",
            "AGEYEARS", "TRAUMATYPE",
        ),
    }

    def _resolve_cohort_vars(
        self, available_cols: Iterable[str]
    ) -> dict[str, list[str]]:
        """Return {cohort_name: list of vars from _COHORT_VAR_DEFS that
        actually exist in ``available_cols``}.

        Variables in the canonical definition that are NOT in the parquet
        are silently dropped (a debug message is logged).  This handles
        the common case where TEMPERATURE / PULSEOXIMETRY / PULSERATE
        aren't in the user's NTDB extract — the ed_complete cohort then
        falls back to the on-scene-physiology subset and a warning is
        emitted so the user knows.
        """
        avail = set(available_cols)
        resolved: dict[str, list[str]] = {}
        for cohort_name, all_vars in self._COHORT_VAR_DEFS.items():
            kept = [v for v in all_vars if v in avail]
            missing = [v for v in all_vars if v not in avail]
            if missing:
                log.info(
                    "[%s] cohort %s: %d/%d canonical vars missing from "
                    "this dataset (%s) — using %d available vars: %s",
                    self.cfg.model_id, cohort_name,
                    len(missing), len(all_vars), missing,
                    len(kept), kept,
                )
            resolved[cohort_name] = kept
        return resolved

    def _baseline_required_cols(self, available_cols: Iterable[str]) -> list[str]:
        """Inputs that must be non-NaN for ALL computable baselines.

        Adapts to which baseline scores can actually be computed from the
        dataset.  If TRISS inputs (GCS/SBP/RR + ISS + age) are missing,
        only ISS and NISS are required (since TRISS would be skipped).
        """
        avail = set(available_cols)
        required: set[str] = set()
        if "ISS" in avail:
            required.add("ISS")
        if "NISS" in avail:
            required.add("NISS")
        triss_inputs = {"GCSTOTAL", "SBPFIRST", "RRFIRST", "ISS", "AGEYEARS"}
        if triss_inputs.issubset(avail):
            required |= triss_inputs
            if "TRAUMATYPE" in avail:
                required.add("TRAUMATYPE")
        return sorted(required)

    def _snapshot_raw_baseline_cols(self) -> None:
        """Save a copy of the raw (pre-scaling) score columns used by baselines.

        RobustScaler will overwrite their values in ``self.data``; we preserve
        the originals here so ``_evaluate_baselines`` always works with the
        clinically meaningful numbers.
        """
        cols_present = [c for c in self._BASELINE_COLS if c in self.data.columns]
        self._raw_baseline: pd.DataFrame = self.data[cols_present].copy()
        log.debug("Snapshotted %d raw baseline columns: %s", len(cols_present), cols_present)

    def _snapshot_raw_missingness(self) -> None:
        """Capture raw NaN mask of cohort-relevant variables.

        Snapshots the union of variables referenced across every entry in
        ``_COHORT_VAR_DEFS`` (∩ ``self.data.columns``) BEFORE the imputer
        runs, so cohort masks reflect the original parquet missingness.
        Cohort definitions are explicit variable lists (Tran-style), NOT
        catalogue phases — see ``_COHORT_VAR_DEFS`` for the rationale.
        """
        resolved = self._resolve_cohort_vars(self.data.columns)
        all_cohort_cols = sorted({v for cols in resolved.values() for v in cols})
        if not all_cohort_cols:
            log.warning(
                "[%s] No cohort-relevant cols found in parquet — "
                "cohort masks will be empty.  Check that AGEYEARS, "
                "GCSTOTAL, SBPFIRST, RRFIRST, TRAUMATYPE, ISS, NISS "
                "are present.",
                self.cfg.model_id,
            )
        self._raw_missingness: pd.DataFrame = (
            self.data[all_cohort_cols].isna().copy()
            if all_cohort_cols
            else pd.DataFrame(index=self.data.index)
        )
        self._cohort_resolved_vars: dict[str, list[str]] = resolved
        log.info(
            "[%s] Cohort missingness snapshot: %d rows × %d cols. "
            "Per-cohort var counts: %s",
            self.cfg.model_id, *self._raw_missingness.shape,
            {c: len(v) for c, v in resolved.items()},
        )

    def _fit_transformers_on_train(self):
        train_idx = self.splits["train"]
        train = self.data.loc[train_idx, self.predictors]

        # Round 26: import the semantic-categorical override
        from .catalogue import is_semantic_categorical

        self.numeric_cols = []
        self.categorical_cols = []
        for col in self.predictors:
            # Force semantic categoricals (SEX, TRAUMATYPE, comorbidity flags)
            # into categorical_cols regardless of stored dtype — median-imputing
            # SEX (1=Male, 2=Female) makes no sense.
            if is_semantic_categorical(col):
                self.categorical_cols.append(col)
                continue
            coerced = pd.to_numeric(train[col], errors="coerce")
            if coerced.notna().sum() > 0 and train[col].dtype not in ("object", "bool") \
                    and str(train[col].dtype) not in ("string", "str", "category", "boolean"):
                self.numeric_cols.append(col)
            else:
                if pd.api.types.is_numeric_dtype(train[col]):
                    self.numeric_cols.append(col)
                else:
                    self.categorical_cols.append(col)
        log.info("Predictors split: %d numeric, %d categorical (%d forced "
                 "by SEMANTIC_CATEGORICALS)", len(self.numeric_cols),
                 len(self.categorical_cols),
                 sum(1 for c in self.categorical_cols if is_semantic_categorical(c)))

        for col in self.numeric_cols:
            if not pd.api.types.is_float_dtype(self.data[col]):
                self.data[col] = self.data[col].astype(float)

        train = self.data.loc[train_idx, self.predictors]
        for col in self.categorical_cols:
            non_null = train[col].notna()
            if non_null.sum() == 0:
                continue
            vals = np.asarray(train.loc[non_null, col].astype(str).tolist(), dtype=object)
            self.transformers["encoders"][col] = LabelEncoder().fit(vals)

        for col in self.numeric_cols:
            non_null = train[col].notna()
            if non_null.sum() > 1:
                values = np.asarray(train.loc[non_null, col].to_numpy(),
                                     dtype=float).reshape(-1, 1)
                self.transformers["scalers"][col] = RobustScaler().fit(values)

    def _apply_transformers(self):
        # Round 31: vectorise categorical encoding.
        # OLD: per-row [le.transform([v])[0] for v in vals] — sklearn calls
        # for every cell, ~66 M calls on the user's 1.44M × 46-col grid =
        # 28 minutes wall-clock just for encoding.
        # NEW: np.searchsorted against sorted le.classes_ — pure numpy
        # vectorised, ~1-2 seconds for the same workload.
        # Unknown values (not in le.classes_) get encoded as -1, same as
        # before.
        for col, le in self.transformers["encoders"].items():
            if col not in self.data.columns:
                continue
            non_null = self.data[col].notna()
            if not non_null.any():
                continue
            vals = np.asarray(self.data.loc[non_null, col].astype(str).to_numpy(),
                              dtype=object)
            encoded = _vectorised_label_encode(vals, le.classes_)
            new_col = np.full(len(self.data), np.nan, dtype=float)
            new_col[non_null.to_numpy()] = encoded
            self.data[col] = new_col

        for col, scaler in self.transformers["scalers"].items():
            if col not in self.data.columns:
                continue
            non_null = self.data[col].notna()
            if not non_null.any():
                continue
            if not pd.api.types.is_float_dtype(self.data[col]):
                self.data[col] = self.data[col].astype(float)
            vals = np.asarray(self.data.loc[non_null, col].to_numpy(),
                               dtype=float).reshape(-1, 1)
            self.data.loc[non_null, col] = scaler.transform(vals).flatten()

    # Round 37: families that handle NaN natively can use imputer_method="none"
    _NAN_TOLERANT_FAMILIES = frozenset({"xgboost", "lightgbm", "catboost", "flaml"})

    def _fit_imputer_and_eval(self, output_dir: Path):
        # ─── Regime 1: imputer_method == "none" ──────────────────────────
        # Skip imputation entirely.  Only valid when the chosen model
        # family handles NaN natively.  Round 37.
        if self.cfg.imputer_method == "none":
            if self.cfg.model_family not in self._NAN_TOLERANT_FAMILIES:
                raise ValueError(
                    f"imputer_method='none' requires a NaN-tolerant model "
                    f"family ({sorted(self._NAN_TOLERANT_FAMILIES)}); got "
                    f"{self.cfg.model_family!r}.  Either pick a different "
                    f"imputer, a different family, or remove this combo "
                    f"from the grid."
                )
            log.info(
                "imputer_method='none': skipping all imputation; "
                "%s will consume NaN-containing data directly.",
                self.cfg.model_family,
            )
            self.imputer = None
            # Still need a baseline imputer to make ISS/NISS/TRISS
            # comparable.  Use median_mode for that — it's tangential to
            # the model's data.
            if hasattr(self, "_raw_baseline") and len(self._raw_baseline) > 0:
                self._fit_baseline_imputer(method_override="median_mode")
            return

        # ─── Regime 2/3: normal imputation, optionally with feature gating ─
        self.imputer = make_imputer(
            method=self.cfg.imputer_method,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
        )
        train = self.data.loc[self.splits["train"], self.predictors]
        log.info("Fitting imputer %s on %d training rows",
                 self.cfg.imputer_method, len(train))
        self.imputer.fit(train)

        # Compute train-set std for each numeric column — needed to
        # convert MAE into a relative MAE in the eval CSV.
        train_std = {col: float(train[col].std(skipna=True))
                     for col in self.numeric_cols if col in train.columns}

        # Round 37: evaluate imputer quality on the CALIBRATION SET, not
        # the test set.  Test must stay untouched until the final
        # evaluation step.
        calibration_df = self.data.loc[self.splits["calibration"], self.predictors]
        log.info("Evaluating imputer on calibration set (%d rows) — "
                 "test set untouched", len(calibration_df))
        eval_df = evaluate_imputation(
            calibration_df, self.imputer,
            mask_fraction=0.10,
            random_state=self.cfg.random_state,
            output_dir=output_dir,
            train_std=train_std,
            thresholds=self.cfg.imputer_check_thresholds,
        )

        # ─── Regime 3: imputer_check — filter features by reconstruction quality ─
        if self.cfg.imputer_check:
            kept, dropped = select_good_features(eval_df)
            log.info("imputer_check=True: %d features pass thresholds "
                     "(MAE/std < %.2f for numeric, accuracy > %.2f for "
                     "categorical); %d features dropped",
                     len(kept),
                     self.cfg.imputer_check_thresholds.get(
                         "numeric_relative_mae_max", 0.5),
                     self.cfg.imputer_check_thresholds.get(
                         "categorical_accuracy_min", 0.7),
                     len(dropped))
            if dropped:
                log.info("  dropped: %s", dropped)

            if not kept:
                # No features survive — caller will short-circuit
                # the run via the public method below.
                log.warning("imputer_check=True dropped ALL features; "
                            "training will be skipped and metrics emitted as NaN")
                self._imputer_check_no_features = True
                # Save the eval CSV showing zero kept rows so it's clear
                # what happened
                self.predictors = []
                self.numeric_cols = []
                self.categorical_cols = []
                # Set self.imputer to None to skip downstream imputation
                self.imputer = None
                return

            # Filter the predictor set and refit the imputer on it
            self.predictors = [p for p in self.predictors if p in kept]
            self.numeric_cols = [c for c in self.numeric_cols if c in kept]
            self.categorical_cols = [c for c in self.categorical_cols if c in kept]
            self.imputer = make_imputer(
                method=self.cfg.imputer_method,
                numeric_cols=self.numeric_cols,
                categorical_cols=self.categorical_cols,
            )
            train_kept = self.data.loc[self.splits["train"], self.predictors]
            log.info("Refitting imputer on %d kept predictors", len(self.predictors))
            self.imputer.fit(train_kept)

        # ─── Apply the (possibly filtered) imputer to all partitions ───
        for part in ("train", "calibration", "test"):
            idx = self.splits[part]
            self.data.loc[idx, self.predictors] = self.imputer.transform(
                self.data.loc[idx, self.predictors]
            ).values

        if len(self.holdout_idx) > 0:
            self.data.loc[self.holdout_idx, self.predictors] = self.imputer.transform(
                self.data.loc[self.holdout_idx, self.predictors]
            ).values
            log.debug("Imputed %d temporal holdout rows in-place", len(self.holdout_idx))

        # ---- Baseline-input imputer ----------------------------------
        if hasattr(self, "_raw_baseline") and len(self._raw_baseline) > 0:
            self._fit_baseline_imputer()

    def _fit_baseline_imputer(self, method_override: str | None = None) -> None:
        """Fit a baseline-only imputer and produce ``self._imputed_baseline``.

        Uses the same imputer method the user configured for the model
        (cfg.imputer_method) and fits on the same training rows.  The
        result is a DataFrame the same shape as ``self._raw_baseline``
        but with NO NaN cells in the baseline-input columns — so every
        row produces a defined ISS / NISS / TRISS in baseline_metrics().
        Categorical columns (TRAUMATYPE) get mode-imputed.

        Round 37: ``method_override`` lets callers force a concrete imputer
        method even when ``cfg.imputer_method == "none"`` (the model uses
        no imputer, but the baselines still need defined ISS/NISS/TRISS for
        a fair comparison — we use median_mode for those).
        """
        baseline_method = method_override or self.cfg.imputer_method
        # Drop columns that are 100% NaN on the training set — sklearn's
        # SimpleImputer silently skips them during fit but expects them
        # absent during transform too, causing a "Columns must be same
        # length as key" error.  ISS_05 is the typical offender (an
        # AY-2019-only alias that's all-NaN in 2021+ data).
        train_idx = self.splits["train"]
        train_raw_bl = self._raw_baseline.reindex(train_idx)
        all_nan_cols = [c for c in train_raw_bl.columns
                        if train_raw_bl[c].isna().all()]
        usable_cols = [c for c in train_raw_bl.columns if c not in all_nan_cols]
        if all_nan_cols:
            log.info(
                "[%s] Baseline imputer: skipping %d all-NaN cols on training "
                "set: %s",
                self.cfg.model_id, len(all_nan_cols), all_nan_cols,
            )
        if not usable_cols:
            log.warning(
                "[%s] Baseline imputer: no usable cols — all baseline cols "
                "are all-NaN on training set. Falling back to raw baselines.",
                self.cfg.model_id,
            )
            self._baseline_imputer = None
            self._usable_baseline_cols = []
            return

        bl_categorical = [c for c in ("TRAUMATYPE",) if c in usable_cols]
        bl_numeric     = [c for c in usable_cols if c not in bl_categorical]

        self._baseline_imputer = make_imputer(
            method=baseline_method,
            numeric_cols=bl_numeric,
            categorical_cols=bl_categorical,
        )
        self._usable_baseline_cols = usable_cols
        train_subset = train_raw_bl[usable_cols]
        log.info(
            "[%s] Fitting baseline imputer (%s) on %d training rows × %d cols (%s)",
            self.cfg.model_id, baseline_method,
            len(train_subset), len(usable_cols), usable_cols,
        )
        try:
            self._baseline_imputer.fit(train_subset)
        except Exception as exc:
            log.error(
                "[%s] Baseline imputer fit failed: %s. Baselines will fall "
                "back to drop-NaN behaviour (smaller n_valid).",
                self.cfg.model_id, exc,
            )
            self._baseline_imputer = None
            return

        # Transform ALL rows in _raw_baseline (train + cal + test + holdout)
        # using only the usable cols, then re-attach the all-NaN cols
        # untouched so the output schema matches the input.
        try:
            full_subset = self._raw_baseline[usable_cols]
            imputed_arr = self._baseline_imputer.transform(full_subset)
            imputed_subset = pd.DataFrame(
                imputed_arr,
                columns=usable_cols,
                index=self._raw_baseline.index,
            )
            # Reattach skipped cols (still all NaN) so downstream code
            # that expects them by name still finds them.
            self._imputed_baseline = self._raw_baseline.copy()
            self._imputed_baseline[usable_cols] = imputed_subset.values
            n_nan_before = int(self._raw_baseline[usable_cols].isna().sum().sum())
            n_nan_after  = int(self._imputed_baseline[usable_cols].isna().sum().sum())
            log.info(
                "[%s] Baseline imputation: %d NaN cells before -> %d after "
                "(across %d usable cols)",
                self.cfg.model_id, n_nan_before, n_nan_after, len(usable_cols),
            )
        except Exception as exc:
            log.error(
                "[%s] Baseline imputer transform failed: %s. "
                "Falling back to raw (drop-NaN) baselines.",
                self.cfg.model_id, exc, exc_info=True,
            )
            self._imputed_baseline = None

    def _augment_if_needed(self):
        if self.task != "binary" or self.cfg.data_augmentation is None:
            return
        train_idx = self.splits["train"]
        X = self.data.loc[train_idx, self.predictors]
        y = self.y_all[train_idx.to_numpy()]
        method = self.cfg.data_augmentation.lower()
        try:
            if method == "smote":
                from imblearn.over_sampling import SMOTE
                X2, y2 = SMOTE(random_state=self.cfg.random_state).fit_resample(X, y)
            elif method == "adasyn":
                from imblearn.over_sampling import ADASYN
                X2, y2 = ADASYN(random_state=self.cfg.random_state).fit_resample(X, y)
            else:
                log.warning("Unknown data_augmentation=%s; skipping", method)
                return
        except ImportError:
            log.warning("imbalanced-learn missing; install with `pip install 'trauma_ml[imbalance]'`")
            return
        log.info("Augmented training set %d -> %d rows via %s", len(X), len(X2), method)
        self._augmented_X = X2
        self._augmented_y = y2

    def _fit_model(self):
        if getattr(self, "_augmented_X", None) is not None:
            X = self._augmented_X
            y = self._augmented_y
        else:
            X = self.data.loc[self.splits["train"], self.predictors]
            y = self.y_all[self.splits["train"].to_numpy()]
        log.info("Training %s on %d rows, %d features",
                 self.cfg.model_family, len(X), X.shape[1])
        base_model = build_model(
            family=self.cfg.model_family,
            task=self.task,
            n_jobs=self.cfg.n_jobs,
            use_gpu=self.cfg.use_gpu,
        )
        if self.cfg.tune_hparams:
            # Round 26: lightweight hyperparameter search via sklearn's
            # RandomizedSearchCV.  Tran 2022 did this with "broadly defined
            # hyperparameter space" and 10-fold CV, but they found
            # "negligible impact of hyperparameter tuning."  We default the
            # search to 3 folds and 10 iterations to keep walltime under
            # control; bump --n-search-iter / --cv-folds to expand.
            self.model = self._wrap_with_random_search(base_model, X, y)
        else:
            self.model = base_model
            # Round 33: TabNet's `pytorch_tabnet` rejects pandas DataFrames.
            # Convert to numpy on the fly for that family only — other
            # families are happy with DataFrames (xgboost/lightgbm/etc.).
            if self.cfg.model_family == "tabnet":
                X_fit = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
                y_fit = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                # TabNet also wants integer labels (not float) for binary
                y_fit = y_fit.astype(np.int64)
                self.model.fit(X_fit, y_fit)
            else:
                self.model.fit(X, y)

    # ------------------------------------------------------------------
    def _wrap_with_random_search(self, base_model, X, y):
        """RandomizedSearchCV wrapper for tree-based families.

        Falls back to plain `.fit(X, y)` for families where defining a
        sklearn-compatible param distribution doesn't apply (FLAML, TPOT
        already do AutoML internally; tabpfn / tabnet aren't standard
        sklearn estimators).
        """
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import loguniform, randint

        family = self.cfg.model_family
        scoring = "balanced_accuracy" if self.task == "multiclass" else "average_precision"

        # Per-family search distributions.  Kept small — the goal is sanity
        # checking, not an exhaustive HPO.
        if family == "xgboost":
            param_dist = {
                "n_estimators":      randint(100, 500),
                "max_depth":         randint(3, 10),
                "learning_rate":     loguniform(1e-3, 3e-1),
                "subsample":         [0.7, 0.85, 1.0],
                "colsample_bytree":  [0.7, 0.85, 1.0],
            }
        elif family == "lightgbm":
            param_dist = {
                "n_estimators":     randint(100, 500),
                "num_leaves":       randint(15, 63),
                "learning_rate":    loguniform(1e-3, 3e-1),
                "subsample":        [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            }
        elif family == "catboost":
            param_dist = {
                "iterations":      randint(100, 500),
                "depth":           randint(4, 10),
                "learning_rate":   loguniform(1e-3, 3e-1),
                "l2_leaf_reg":     loguniform(1, 10),
            }
        elif family == "random_forest":
            param_dist = {
                "n_estimators":      randint(100, 500),
                "max_depth":         [None, 5, 10, 20],
                "min_samples_split": randint(2, 20),
                "max_features":      ["sqrt", "log2"],
            }
        elif family in ("logistic_l1", "logistic_elasticnet"):
            param_dist = {"C": loguniform(1e-3, 1e2)}
        else:
            log.info("[%s] family=%s has no tunable param distribution; "
                     "fitting with default hyperparameters",
                     self.cfg.model_id, family)
            base_model.fit(X, y)
            return base_model

        # The model factory may return a wrapper; RandomizedSearchCV needs a
        # sklearn-compatible estimator.  Most tree wrappers expose a
        # `.fit/.predict/.predict_proba` interface; assume sklearn-compat.
        #
        # Round 31: pass an EXPLICIT StratifiedKFold splitter rather than an
        # int.  Sklearn's auto-detection (int → StratifiedKFold for
        # classifiers) requires the estimator's `_estimator_type` to be
        # 'classifier'; some of our wrappers (xgboost native, lightgbm
        # native) don't always carry this, so being explicit guarantees
        # target-stratified CV folds regardless of estimator type.
        from sklearn.model_selection import StratifiedKFold
        cv_splitter = StratifiedKFold(
            n_splits=self.cfg.cv_folds,
            shuffle=True,
            random_state=42,
        )
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=self.cfg.n_search_iter,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=1,           # the inner model uses n_jobs internally
            random_state=42,
            verbose=1,
            refit=True,
            error_score="raise",
        )
        try:
            search.fit(X, y)
            log.info("[%s] best params (cv %s=%.4f): %s",
                     self.cfg.model_id, scoring, search.best_score_,
                     search.best_params_)
            return search.best_estimator_
        except Exception as exc:
            log.warning("[%s] RandomizedSearchCV failed (%s); falling back "
                        "to default hyperparameters", self.cfg.model_id, exc)
            base_model.fit(X, y)
            return base_model

    def _calibrate_if_binary(self):
        if self.task != "binary":
            return
        calibration_choice = (self.cfg.calibration or "none").lower()
        if calibration_choice == "none":
            return
        X_cal = self.data.loc[self.splits["calibration"], self.predictors]
        y_cal = self.y_all[self.splits["calibration"].to_numpy()]
        try:
            from sklearn.frozen import FrozenEstimator
            base_for_cal = FrozenEstimator(self.model)
            prefit_kwargs = {"cv": 5}
        except ImportError:
            base_for_cal = self.model
            prefit_kwargs = {"cv": "prefit"}
        method_map = {"platt": "sigmoid", "isotonic": "isotonic"}
        if calibration_choice not in method_map:
            log.warning("Unknown calibration %r; skipping", calibration_choice)
            return
        try:
            cal = CalibratedClassifierCV(
                estimator=base_for_cal,
                method=method_map[calibration_choice],
                **prefit_kwargs,
            )
            cal.fit(X_cal, y_cal)
            self.calibrated_models[calibration_choice] = cal
        except Exception as e:
            log.warning("Calibration %s failed: %s", calibration_choice, e)

    # ================================================================= #
    # Evaluation
    # ================================================================= #
    def _evaluate(self, output_dir: Path, outputs_root: Path):
        # ---- overall + calibrated metrics per partition ----------------
        for part in ("train", "calibration", "test"):
            X = self.data.loc[self.splits[part], self.predictors]
            y = self.y_all[self.splits[part].to_numpy()]
            metrics = evaluate(self.model, X, y, task=self.task, classes=self.classes)
            save_metrics(metrics, output_dir, f"overall__{part}.json")
            for cal_name, cal_model in self.calibrated_models.items():
                if cal_model is None:
                    continue
                m_cal = evaluate(cal_model, X, y, task=self.task, classes=self.classes)
                save_metrics(m_cal, output_dir, f"overall__{part}__{cal_name}.json")

        # ---- subgroup metrics on test set (gender, age_group, year …) --
        X_te = self.data.loc[self.splits["test"], self.predictors]
        y_te = self.y_all[self.splits["test"].to_numpy()]
        for axis in self.cfg.subgroup_axes:
            if axis not in self.data.columns:
                continue
            axis_series = self.data.loc[self.splits["test"], axis]
            df_sub = subgroup_metrics(
                self.model, X_te, y_te, axis_series, axis,
                task=self.task, classes=self.classes,
            )
            if not df_sub.empty:
                save_subgroup_metrics(df_sub, output_dir / "subgroups", axis)

        # ---- subgroup metrics on TRAIN and CALIBRATION partitions too ---
        # (Round 11 item 1) — fixes "subgroup metrics NaN for models on
        # train/calibration partitions". Saved to dedicated dirs so the
        # aggregator can pick them up under the right partition labels.
        for part_name, dir_name in (("train", "subgroups_train"),
                                      ("calibration", "subgroups_calibration")):
            if part_name not in self.splits or len(self.splits[part_name]) == 0:
                continue
            X_part = self.data.loc[self.splits[part_name], self.predictors]
            y_part = self.y_all[self.splits[part_name].to_numpy()]
            for axis in self.cfg.subgroup_axes:
                if axis not in self.data.columns:
                    continue
                axis_series = self.data.loc[self.splits[part_name], axis]
                df_sub = subgroup_metrics(
                    self.model, X_part, y_part, axis_series, axis,
                    task=self.task, classes=self.classes,
                )
                if not df_sub.empty:
                    save_subgroup_metrics(df_sub, output_dir / dir_name, axis)

        # ---- temporal year holdout evaluation --------------------------
        X_ho, y_ho = None, None
        if len(self.holdout_idx) > 0:
            X_ho, y_ho = self._evaluate_year_holdout(output_dir)

        # ---- external file holdout (legacy) ----------------------------
        if self.cfg.holdout_dataset_path is not None:
            X_ho_ext, y_ho_ext = self._evaluate_file_holdout(output_dir)
            if X_ho_ext is not None:
                X_ho, y_ho = X_ho_ext, y_ho_ext   # override for plots

        # ---- baselines (ISS / NISS / TRISS) ----------------------------
        # Round 23: baselines and their calibration are mortality-risk
        # scores by definition — only meaningful for the binary mortality
        # task.  Skip the entire block for multiclass severity-band targets.
        if self.task == "binary":
            self._evaluate_baselines(output_dir, X_te, y_te, partition="test")

            # Baselines on TRAIN and CALIBRATION partitions (round 10 user req)
            # — gives the user the same partitions for baselines as for models.
            if hasattr(self, "_imputed_baseline") or hasattr(self, "_raw_baseline"):
                source = (self._imputed_baseline
                          if getattr(self, "_imputed_baseline", None) is not None
                          else self._raw_baseline)
                for part in ("train", "calibration"):
                    if part not in self.splits or len(self.splits[part]) == 0:
                        continue
                    idx = self.splits[part]
                    df_raw_part = source.reindex(idx)
                    y_part = self.y_all[idx.to_numpy()]
                    sg_part = self.data.reindex(idx)
                    self._evaluate_baselines(
                        output_dir, X_model=None, y=y_part,
                        partition=part, df_raw=df_raw_part,
                        subgroup_df=sg_part,
                    )

            # ---- baseline calibration (round 10) ----------------------------
            # Fit Platt / isotonic / threshold-tuning on the CALIBRATION set's
            # baseline scores, save the calibrated outputs.  Provides the same
            # quality of probability calibration to ISS / NISS / TRISS as
            # the ML models receive.
            if "calibration" in self.splits and len(self.splits["calibration"]) > 0:
                try:
                    self._calibrate_baselines(output_dir)
                except Exception as exc:
                    log.error("[%s] _calibrate_baselines failed: %s",
                              self.cfg.model_id, exc, exc_info=True)

            if len(self.holdout_idx) > 0:
                # Use IMPUTED baseline snapshot so every row produces a value.
                source = (self._imputed_baseline
                          if getattr(self, "_imputed_baseline", None) is not None
                          else self._raw_baseline)
                df_ho_raw = (
                    source.reindex(self.holdout_idx)
                    if hasattr(self, "_raw_baseline")
                    else self.data.loc[self.holdout_idx]
                )
                y_ho_bl = self.y_all[self.holdout_idx.to_numpy()]
                subgroup_ho_df = self.data.reindex(self.holdout_idx)
                self._evaluate_baselines(output_dir, None, y_ho_bl,
                                         partition="holdout_year",
                                         df_raw=df_ho_raw,
                                         subgroup_df=subgroup_ho_df)
        else:
            log.info("[%s] task=%s — skipping baseline (ISS/NISS/TRISS) "
                     "computation and calibration (binary-only)",
                     self.cfg.model_id, self.task)

        # ---- cohort-filtered evaluation (model + baselines) ------------
        self._evaluate_cohorts(output_dir)

        # ---- diagnostic plots ------------------------------------------
        if self.cfg.generate_plots:
            class_names = [str(self.target_inverse.get(c, c)) for c in (self.classes or [])]
            # Round 26: build baseline ISS / NISS / TRISS score arrays at
            # the test (and holdout, if any) row indices for overlay on the
            # ROC/PR plots.  Multiclass tasks: skip — plot_roc_pr already
            # short-circuits for K>2 and these scores are mortality-risk by
            # definition, not severity-band predictors.
            #
            # Round 30 fix: holdout overlay needs to choose between TWO
            # baseline sources depending on which holdout path was used:
            #   * Year-stratified internal holdout — pulled from
            #     self._imputed_baseline at self.holdout_idx (these indices
            #     point into the train parquet).
            #   * External file holdout (--holdout-dataset flag) — pulled
            #     from self._ext_holdout_state["raw_baseline"], which is a
            #     SEPARATE frame indexed by the external parquet's rows.
            # Previously the file-holdout case left baseline_scores_holdout
            # as None because `len(self.holdout_idx) > 0` was False, leading
            # to the missing dashed lines in user's plots.
            baseline_scores_test = None
            baseline_scores_holdout = None
            if self.task == "binary":
                try:
                    baseline_scores_test = self._build_baseline_scores_for(
                        self.splits["test"]
                    )
                    using_external = (self.cfg.holdout_dataset_path is not None
                                       and getattr(self, "_ext_holdout_state", None)
                                           is not None)
                    if X_ho is not None and y_ho is not None:
                        if using_external:
                            baseline_scores_holdout = self._build_baseline_scores_for(
                                idx=None, use_external=True,
                            )
                            log.info(
                                "[%s] Built baseline_scores_holdout from EXTERNAL "
                                "file holdout (n=%d rows, scores=%s)",
                                self.cfg.model_id,
                                len(next(iter(baseline_scores_holdout.values())))
                                    if baseline_scores_holdout else 0,
                                list(baseline_scores_holdout.keys()),
                            )
                        elif len(self.holdout_idx) > 0:
                            baseline_scores_holdout = self._build_baseline_scores_for(
                                self.holdout_idx
                            )
                            log.info(
                                "[%s] Built baseline_scores_holdout from INTERNAL "
                                "year holdout (n=%d rows, scores=%s)",
                                self.cfg.model_id, len(self.holdout_idx),
                                list(baseline_scores_holdout.keys()),
                            )
                except Exception as exc:
                    log.warning("[%s] Could not build baseline_scores for plot "
                                "overlay: %s", self.cfg.model_id, exc, exc_info=True)
            save_model_plots(
                model=self.model,
                X_test=X_te,
                y_test=y_te,
                feature_names=self.predictors,
                output_dir=outputs_root / "plots" / self.cfg.model_id,
                model_id=self.cfg.model_id,
                X_holdout=X_ho,
                y_holdout=y_ho,
                class_names=class_names or None,
                enable_shap=self.cfg.enable_shap,
                baseline_scores_test=baseline_scores_test,
                baseline_scores_holdout=baseline_scores_holdout,
            )

    def _evaluate_year_holdout(
        self, output_dir: Path
    ) -> tuple[pd.DataFrame | None, np.ndarray | None]:
        """Evaluate the temporal year holdout.  Returns (X_ho, y_ho) for plots."""
        log.info("Evaluating temporal year holdout (%d rows)", len(self.holdout_idx))
        df_ho = self.data.loc[self.holdout_idx]
        y_ho  = self.y_all[self.holdout_idx.to_numpy()]

        # Apply imputer to the holdout (transformers already applied to all rows)
        X_ho_raw = df_ho[self.predictors]
        if self.imputer is not None:
            X_ho = pd.DataFrame(
                self.imputer.transform(X_ho_raw),
                columns=self.predictors,
                index=X_ho_raw.index,
            )
        else:
            X_ho = X_ho_raw

        metrics = evaluate(self.model, X_ho, y_ho, task=self.task, classes=self.classes)
        metrics["n_holdout_year_rows"] = int(len(df_ho))
        save_metrics(metrics, output_dir, "overall__holdout_year.json")
        for cal_name, cal_model in self.calibrated_models.items():
            if cal_model is None:
                continue
            m_cal = evaluate(cal_model, X_ho, y_ho, task=self.task, classes=self.classes)
            save_metrics(m_cal, output_dir, f"overall__holdout_year__{cal_name}.json")

        for axis in self.cfg.subgroup_axes:
            if axis not in df_ho.columns:
                continue
            df_sub = subgroup_metrics(
                self.model, X_ho, y_ho, df_ho[axis], axis,
                task=self.task, classes=self.classes,
            )
            if not df_sub.empty:
                save_subgroup_metrics(df_sub, output_dir / "subgroups_holdout_year", axis)

        return X_ho, y_ho

    def _evaluate_file_holdout(
        self, output_dir: Path
    ) -> tuple[pd.DataFrame | None, np.ndarray | None]:
        """Load an external holdout parquet and evaluate (legacy path)."""
        log.info("Loading external holdout from %s", self.cfg.holdout_dataset_path)
        df_ho = pd.read_parquet(self.cfg.holdout_dataset_path)

        if self.cfg.inclusion_strategy:
            strat = get_strategy(self.cfg.inclusion_strategy)
            df_ho, excl_log = strat.apply(df_ho)

        if "AGEYEARS" in df_ho.columns:
            df_ho["age_group"] = assign_age_group(
                df_ho["AGEYEARS"],
                edges=self.cfg.age_bins_edges,
                labels=self.cfg.age_bins_labels,
            )
        if "SEX" in df_ho.columns and "gender" not in df_ho.columns:
            df_ho["gender"] = df_ho["SEX"].map({1: "Male", 2: "Female"})
        if "TRAUMATYPE" in df_ho.columns and "mechanism" not in df_ho.columns:
            df_ho["mechanism"] = df_ho["TRAUMATYPE"]
        if "INTENT" in df_ho.columns and "intent" not in df_ho.columns:
            df_ho["intent"] = df_ho["INTENT"]

        y_ho_raw = build_target(df_ho, self.cfg.target)
        df_ho = df_ho.loc[y_ho_raw.notna()].reset_index(drop=True)
        y_ho_raw = y_ho_raw.loc[y_ho_raw.notna()].reset_index(drop=True)
        if self.cfg.target.kind == "binary":
            y_ho = y_ho_raw.astype(int).to_numpy()
        else:
            class_map = {v: k for k, v in self.target_inverse.items()}
            y_ho = y_ho_raw.map(class_map).to_numpy()

        # ── SNAPSHOT raw state BEFORE any transformation ─────────────────
        # These are needed for (a) baseline ISS/NISS/TRISS computation, which
        # requires the original clinical-score values, and (b) cohort-mask
        # construction, which requires the original NaN pattern.
        # Snapshot covers the union of vars referenced by _COHORT_VAR_DEFS
        # (Tran-style explicit lists) that exist in this parquet —
        # independent of the model's own predictor list.
        cohort_resolved = self._resolve_cohort_vars(df_ho.columns)
        all_cohort_cols = sorted({c for cols in cohort_resolved.values()
                                    for c in cols})
        ext_raw_missingness = (
            df_ho[all_cohort_cols].isna().copy()
            if all_cohort_cols
            else pd.DataFrame(index=df_ho.index)
        )
        ext_baseline_cols = [c for c in self._BASELINE_COLS if c in df_ho.columns]
        ext_raw_baseline = (
            df_ho[ext_baseline_cols].copy()
            if ext_baseline_cols else None
        )
        # Apply the trainer's baseline imputer to the external holdout
        # so every row produces ISS / NISS / TRISS values — same population
        # as the model evaluation, no drop-NaN reduction.  See round 8.
        ext_imputed_baseline: pd.DataFrame | None = None
        if (ext_raw_baseline is not None
                and getattr(self, "_baseline_imputer", None) is not None
                and getattr(self, "_usable_baseline_cols", None)):
            try:
                # Use only the columns the imputer was actually fit on
                # (sklearn skips all-NaN cols at fit time).
                usable = self._usable_baseline_cols
                aligned = ext_raw_baseline.reindex(columns=usable)
                imputed_arr = self._baseline_imputer.transform(aligned)
                imputed_subset = pd.DataFrame(
                    imputed_arr, columns=usable,
                    index=ext_raw_baseline.index,
                )
                # Reattach any non-usable cols (all-NaN) so output keeps
                # the input column set.
                ext_imputed_baseline = ext_raw_baseline.copy()
                ext_imputed_baseline[usable] = imputed_subset.values
                log.info(
                    "[%s] External holdout baseline imputation: "
                    "%d rows × %d usable cols",
                    self.cfg.model_id, len(ext_imputed_baseline), len(usable),
                )
            except Exception as exc:
                log.warning(
                    "[%s] External holdout baseline imputer transform failed: %s. "
                    "Falling back to raw (drop-NaN) baselines for external holdout.",
                    self.cfg.model_id, exc,
                )
                ext_imputed_baseline = None
        log.info(
            "[%s] External holdout raw snapshot: %d rows, "
            "cohort cols=%d, baseline cols=%s",
            self.cfg.model_id, len(df_ho), len(all_cohort_cols),
            ext_baseline_cols,
        )

        for col in self.predictors:
            if col not in df_ho.columns:
                df_ho[col] = np.nan

        for col, le in self.transformers["encoders"].items():
            if col not in df_ho.columns:
                continue
            non_null = df_ho[col].notna()
            if not non_null.any():
                continue
            vals = np.asarray(df_ho.loc[non_null, col].astype(str).to_numpy(),
                              dtype=object)
            encoded = _vectorised_label_encode(vals, le.classes_)
            new_col = np.full(len(df_ho), np.nan, dtype=float)
            new_col[non_null.to_numpy()] = encoded
            df_ho[col] = new_col

        for col, scaler in self.transformers["scalers"].items():
            if col not in df_ho.columns:
                continue
            non_null = df_ho[col].notna()
            if not non_null.any():
                continue
            if not pd.api.types.is_float_dtype(df_ho[col]):
                df_ho[col] = df_ho[col].astype(float)
            vals = np.asarray(df_ho.loc[non_null, col].to_numpy(),
                               dtype=float).reshape(-1, 1)
            df_ho.loc[non_null, col] = scaler.transform(vals).flatten()

        X_ho = df_ho[self.predictors]
        if self.imputer is not None:
            X_ho = pd.DataFrame(
                self.imputer.transform(X_ho),
                columns=self.predictors,
                index=X_ho.index,
            )

        metrics = evaluate(self.model, X_ho, y_ho, task=self.task, classes=self.classes)
        metrics["n_holdout_rows"] = int(len(df_ho))
        save_metrics(metrics, output_dir, "overall__holdout.json")
        for cal_name, cal_model in self.calibrated_models.items():
            if cal_model is None:
                continue
            m_cal = evaluate(cal_model, X_ho, y_ho, task=self.task, classes=self.classes)
            save_metrics(m_cal, output_dir, f"overall__holdout__{cal_name}.json")
        for axis in self.cfg.subgroup_axes:
            if axis not in df_ho.columns:
                continue
            df_sub = subgroup_metrics(
                self.model, X_ho, y_ho, df_ho[axis], axis,
                task=self.task, classes=self.classes,
            )
            if not df_sub.empty:
                save_subgroup_metrics(df_sub, output_dir / "subgroups_holdout", axis)

        # ── baselines on the external holdout ─────────────────────────────
        # Use the IMPUTED baseline snapshot (when available) so every row
        # produces a defined ISS / NISS / TRISS — same population as the
        # model evaluation.  Falls back to raw (drop-NaN) if imputer
        # unavailable or transform failed.
        baseline_input = (ext_imputed_baseline
                          if ext_imputed_baseline is not None
                          else ext_raw_baseline)
        if baseline_input is not None and len(baseline_input) > 0:
            # Route through _evaluate_baselines so subgroup baseline metrics
            # are also produced (round 10) — using df_ho as the source for
            # subgroup axis columns (gender, race, mechanism, intent etc.
            # which are still raw in df_ho before any encoding).
            self._evaluate_baselines(
                output_dir, X_model=None, y=y_ho,
                partition="holdout",
                df_raw=baseline_input,
                subgroup_df=df_ho,
            )

        # ── stash for _evaluate_cohorts and round-32 plot overlay ────────
        # Plot overlay (round 32) uses RAW baseline so the "valid row" mask
        # reflects whether the score was actually computable from raw data,
        # not "whether imputation filled in a value."  Cohort evaluation
        # uses IMPUTED baseline so every row produces a defined ISS/NISS.
        self._ext_holdout_state = {
            "X":               X_ho,                 # transformed, post-imputation
            "y":               y_ho,
            "raw_missingness": ext_raw_missingness,  # raw NaN pattern (cohort defs)
            "raw_baseline":    ext_raw_baseline,     # PRE-imputation snapshot (for plots)
            "imputed_baseline": baseline_input,      # IMPUTED (for cohort eval)
        }

        return X_ho, y_ho

    def _build_baseline_scores_for(self, idx=None, *, use_external: bool = False
                                    ) -> dict[str, np.ndarray]:
        """Return {ISS, NISS, TRISS} score arrays aligned with ``idx``.

        Used by round-26 ROC/PR plots to overlay clinical-baseline curves
        on the same axes as the model's curves.  Each score is converted to
        a "higher = more positive class probability" form so it plays well
        with sklearn's roc_curve / precision_recall_curve:

          - ISS / NISS: raw values (higher → more severe → more likely to die).
            sklearn's ROC is rank-based so the absolute scale doesn't matter.
          - TRISS: 1 - survival_probability (so higher → more likely to die,
            matching the positive class = mortality convention).

        Rows with NaN in a baseline column simply contribute NaN scores; the
        plotting code drops them before computing the curve.

        IMPORTANT (round 32): we deliberately read from the *raw* (pre-
        imputation) baseline snapshot, NOT from the imputed one.  The
        validity mask in the plot needs to reflect "was this score
        actually computable from raw NTDB data?"  If we used the imputed
        version, every row would have a (median-filled) value and the
        baseline-complete subset would equal the full set — which is
        exactly the bug observed in user's previous run where the two
        plots came out identical.

        Parameters
        ----------
        idx : pd.Index | None
            Indices into ``self._raw_baseline`` (the train/cal/test/year-
            holdout snapshot).  Use this for any partition that lives inside
            the unified train parquet.
        use_external : bool
            If True, ignore ``idx`` and pull from
            ``self._ext_holdout_state["raw_baseline"]`` (the un-imputed
            external-file snapshot).
        """
        from .baselines import compute_triss

        if use_external:
            ext = getattr(self, "_ext_holdout_state", None)
            if ext is None:
                log.info("No external holdout state — no baseline overlay.")
                return {}
            sub = ext.get("raw_baseline")
            if sub is None or len(sub) == 0:
                log.info("External holdout raw baseline frame is empty.")
                return {}
        else:
            # Round 32: prefer RAW snapshot for plot validity mask.  Fall
            # back to imputed only if raw is somehow missing (shouldn't
            # happen on real runs — _snapshot_raw_baseline_cols runs at
            # __init__ time).
            source = (getattr(self, "_raw_baseline", None)
                      if getattr(self, "_raw_baseline", None) is not None
                      else getattr(self, "_imputed_baseline", None))
            if source is None:
                return {}
            sub = source.reindex(idx)

        # Diagnostic: log the validity coverage so user can sanity-check
        # in the log even before opening the plot
        out: dict[str, np.ndarray] = {}
        for col, score_name in (("ISS", "ISS"), ("NISS", "NISS")):
            if col in sub.columns:
                arr = pd.to_numeric(sub[col], errors="coerce").to_numpy()
                out[score_name] = arr
                pct_valid = 100 * np.isfinite(arr).sum() / max(1, len(arr))
                log.info(
                    "[%s] %s baseline: %d/%d rows valid (%.1f%%)%s",
                    self.cfg.model_id, score_name,
                    int(np.isfinite(arr).sum()), len(arr), pct_valid,
                    " [external]" if use_external else "",
                )
        # TRISS: compute_triss returns survival probability; flip to mortality
        try:
            triss_surv = compute_triss(sub)
            triss_mort = 1.0 - triss_surv.to_numpy()
            if np.isfinite(triss_mort).any():
                out["TRISS"] = triss_mort
                pct_valid = 100 * np.isfinite(triss_mort).sum() / max(1, len(triss_mort))
                log.info(
                    "[%s] TRISS baseline: %d/%d rows valid (%.1f%%)%s",
                    self.cfg.model_id,
                    int(np.isfinite(triss_mort).sum()), len(triss_mort), pct_valid,
                    " [external]" if use_external else "",
                )
        except Exception as exc:
            log.info("Skipping TRISS overlay (compute_triss failed: %s)", exc)
        return out

    def _evaluate_baselines(
        self,
        output_dir: Path,
        X_model: pd.DataFrame | None,
        y: np.ndarray,
        partition: str,
        df_raw: pd.DataFrame | None = None,
        subgroup_df: pd.DataFrame | None = None,
    ) -> None:
        """Compute ISS / NISS / TRISS baselines and save to JSON.

        Parameters
        ----------
        df_raw : DataFrame or None
            Raw baseline-input columns (ISS, NISS, GCSTOTAL etc.) for the
            partition's rows.  If None, looked up from ``self._raw_baseline``
            for the test partition.
        subgroup_df : DataFrame or None
            Rows of the original (pre-encoding) data for this partition,
            containing the subgroup axis columns (gender, race, mechanism …).
            If None, looked up from ``self.data`` for the test partition.
            Subgroup baseline metrics are saved when this is non-None and
            at least one cfg.subgroup_axes column is present.

        Files written per baseline score (ISS, NISS, TRISS):
            ``baseline__<score>__<partition>.json``                    (overall)
            ``baseline__<score>__<partition>__subgroups_by_<axis>.csv`` (per axis)
        """
        if self.task != "binary":
            return

        if df_raw is None:
            test_idx = self.splits.get("test", pd.Index([]))
            if not hasattr(self, "_raw_baseline"):
                log.error(
                    "[%s] _raw_baseline not available — _snapshot_raw_baseline_cols() "
                    "was not called before transformers. Baselines will be missing.",
                    self.cfg.model_id,
                )
                return
            source = (self._imputed_baseline
                      if getattr(self, "_imputed_baseline", None) is not None
                      else self._raw_baseline)
            df_raw = source.reindex(test_idx).copy()
            # Default subgroup_df for test partition: original data rows
            if subgroup_df is None:
                subgroup_df = self.data.reindex(test_idx).copy()

        if len(df_raw) == 0:
            log.error(
                "[%s] df_raw has 0 rows for partition=%s — no baseline metrics written. "
                "Check that _raw_baseline indices match split indices.",
                self.cfg.model_id, partition,
            )
            return

        if len(y) != len(df_raw):
            log.error(
                "[%s] Length mismatch: y=%d rows, df_raw=%d rows for partition=%s. "
                "Skipping baselines.",
                self.cfg.model_id, len(y), len(df_raw), partition,
            )
            return

        missing_cols = [
            c for c in ("ISS", "NISS", "GCSTOTAL", "SBPFIRST", "RRFIRST")
            if c not in df_raw.columns
        ]
        if missing_cols:
            log.warning(
                "[%s] Baseline snapshot missing columns %s — "
                "add them to always_keep in apply_whitelist and re-run trauma-build. "
                "TRISS will be unavailable; ISS/NISS may still work.",
                self.cfg.model_id, missing_cols,
            )

        # ── overall baseline metrics (existing behaviour) ────────────────
        bl: dict = {}
        try:
            bl = baseline_metrics(df_raw, y)
        except Exception as exc:
            log.error(
                "[%s] baseline_metrics raised an exception for partition=%s: %s. "
                "No baseline files written for this partition.",
                self.cfg.model_id, partition, exc,
                exc_info=True,
            )
            return

        if not bl:
            log.warning(
                "[%s] baseline_metrics returned empty dict for partition=%s.",
                self.cfg.model_id, partition,
            )
            return

        for name, m in bl.items():
            save_metrics(m, output_dir, f"baseline__{name}__{partition}.json")

        # ── subgroup metrics for baselines (round 10) ────────────────────
        # Without these the baseline rows in all_metrics.csv have NaN for
        # every subgroup column → NaN for worst_*  → can't participate
        # in Pareto.  Compute one CSV per (baseline_score, axis) pair,
        # in the same shape as model subgroup CSVs so the aggregator can
        # pick them up.
        if subgroup_df is None:
            log.debug(
                "[%s] No subgroup_df for partition=%s — skipping baseline subgroups.",
                self.cfg.model_id, partition,
            )
        else:
            try:
                self._evaluate_baseline_subgroups(
                    output_dir, df_raw, y, subgroup_df, partition,
                )
            except Exception as exc:
                log.error(
                    "[%s] Baseline subgroup eval failed for partition=%s: %s",
                    self.cfg.model_id, partition, exc,
                    exc_info=True,
                )

        log.info(
            "[%s] Baseline metrics written for partition=%s: %s",
            self.cfg.model_id, partition, list(bl.keys()),
        )

    def _evaluate_baseline_subgroups(
        self,
        output_dir:    Path,
        df_raw:        pd.DataFrame,
        y:             np.ndarray,
        subgroup_df:   pd.DataFrame,
        partition:     str,
    ) -> None:
        """Compute per-subgroup metrics for ISS / NISS / TRISS baselines.

        Saved as ``baseline__<score>__<partition>__subgroups_by_<axis>.csv``
        — same row layout as the model subgroup CSVs, so the aggregator
        treats baseline subgroup numbers identically.
        """
        scores = compute_baseline_scores(df_raw)
        if not scores:
            log.warning(
                "[%s] No computable baselines for subgroup analysis "
                "(partition=%s)", self.cfg.model_id, partition,
            )
            return

        # Align subgroup_df with df_raw's index so masks align with y
        sg_aligned = subgroup_df.reindex(df_raw.index)

        for score_name, info in scores.items():
            score_arr = info["score"]
            pred_arr  = info["pred"]
            valid_arr = info["valid"].to_numpy()

            # For subgroup analysis: drop rows where score is NaN (baseline
            # not computable for that row) — those would force AUROC=NaN
            # within the subgroup.
            keep = valid_arr
            if keep.sum() < 30:
                log.debug(
                    "[%s] %s baseline: only %d valid rows for subgroup analysis "
                    "(partition=%s) — skipping",
                    self.cfg.model_id, score_name, int(keep.sum()), partition,
                )
                continue

            score_v = score_arr[keep]
            pred_v  = pred_arr[keep]
            y_v     = y[keep]
            sg_v    = sg_aligned.iloc[keep]

            for axis in self.cfg.subgroup_axes:
                if axis not in sg_v.columns:
                    continue
                df_sub = subgroup_metrics_for_score(
                    score=score_v, y_pred=pred_v, y_true=y_v,
                    subgroup_series=sg_v[axis], axis_name=axis,
                )
                if df_sub.empty:
                    continue
                out_dir = output_dir / f"subgroups_baseline__{partition}__{score_name}"
                save_subgroup_metrics(df_sub, out_dir, axis)

    def _calibrate_baselines(self, output_dir: Path) -> None:
        """Calibrate ISS / NISS / TRISS baselines on the calibration set.

        For each baseline score we produce three calibrated variants:

        1. **Platt-calibrated**: sigmoid(a + b * score) fit on calibration
           set.  Maps the score to a better-calibrated probability.
           Predictions use threshold 0.5 on the calibrated probability.

        2. **Isotonic-calibrated**: piecewise-constant monotonic mapping
           fit on calibration set.  Same threshold-0.5 rule on the
           calibrated probability.

        3. **Threshold-tuned**: original score; threshold is moved to
           the value that maximises Youden's J = TPR - FPR on the
           calibration set.  Useful when raw probability ordering is
           already well-calibrated but the default cut-off (e.g. 0.5
           for TRISS, ≥16 for ISS) is too lenient/strict for the
           population.

        Each calibrated variant is then re-evaluated on test, calibration,
        and holdout partitions, with files named:
            ``baseline__<score>__<partition>__<cal_method>.json``

        The fitted calibrators are stored on ``self._baseline_calibrators``
        for use in cohort evaluation and external holdout.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import roc_curve

        cal_idx = self.splits["calibration"]
        if len(cal_idx) < 50:
            log.info(
                "[%s] Calibration set has only %d rows — skipping baseline calibration",
                self.cfg.model_id, len(cal_idx),
            )
            return

        source = (self._imputed_baseline
                  if getattr(self, "_imputed_baseline", None) is not None
                  else getattr(self, "_raw_baseline", None))
        if source is None:
            log.warning(
                "[%s] No baseline snapshot available — skipping baseline calibration",
                self.cfg.model_id,
            )
            return

        df_cal = source.reindex(cal_idx)
        y_cal  = self.y_all[cal_idx.to_numpy()]

        # Compute baseline scores on calibration set
        cal_scores = compute_baseline_scores(df_cal)
        if not cal_scores:
            log.warning(
                "[%s] No baseline scores computable on calibration set",
                self.cfg.model_id,
            )
            return

        self._baseline_calibrators: dict[str, dict[str, Any]] = {}

        for score_name, info in cal_scores.items():
            score = info["score"]
            valid = info["valid"].to_numpy()
            keep = valid
            if keep.sum() < 30:
                log.debug(
                    "[%s] %s: only %d valid rows on calibration set — skipping",
                    self.cfg.model_id, score_name, int(keep.sum()),
                )
                continue
            s_cal = score[keep].reshape(-1, 1)
            y_c   = y_cal[keep]
            if len(set(y_c)) < 2:
                log.warning(
                    "[%s] %s calibration set is single-class — skipping",
                    self.cfg.model_id, score_name,
                )
                continue

            calibrators: dict[str, Any] = {}

            # 1. Platt (sigmoid) — fit logistic regression on the score
            try:
                platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                platt.fit(s_cal, y_c)
                calibrators["platt"] = platt
            except Exception as exc:
                log.warning("[%s] %s Platt calibration failed: %s",
                            self.cfg.model_id, score_name, exc)

            # 2. Isotonic
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(s_cal.ravel(), y_c)
                calibrators["isotonic"] = iso
            except Exception as exc:
                log.warning("[%s] %s Isotonic calibration failed: %s",
                            self.cfg.model_id, score_name, exc)

            # 3. Threshold tuning via Youden's J
            try:
                fpr, tpr, thresholds = roc_curve(y_c, s_cal.ravel())
                j_scores = tpr - fpr
                best_idx = int(np.argmax(j_scores))
                best_thresh = float(thresholds[best_idx])
                calibrators["threshold_tuned"] = {
                    "threshold":      best_thresh,
                    "youden_j":       float(j_scores[best_idx]),
                    "tpr_at_optimal": float(tpr[best_idx]),
                    "fpr_at_optimal": float(fpr[best_idx]),
                }
                log.info(
                    "[%s] %s threshold-tuned: optimal=%.4f (J=%.3f, "
                    "TPR=%.3f, FPR=%.3f) [default was %s]",
                    self.cfg.model_id, score_name, best_thresh,
                    j_scores[best_idx], tpr[best_idx], fpr[best_idx],
                    info["threshold"],
                )
            except Exception as exc:
                log.warning("[%s] %s threshold tuning failed: %s",
                            self.cfg.model_id, score_name, exc)

            self._baseline_calibrators[score_name] = calibrators

            # ---- Save calibrator artifacts ----
            cal_save = {
                "score": score_name,
                "default_threshold": info["threshold"],
            }
            if "platt" in calibrators:
                cal_save["platt"] = {
                    "coef":      float(calibrators["platt"].coef_[0][0]),
                    "intercept": float(calibrators["platt"].intercept_[0]),
                }
            if "isotonic" in calibrators:
                cal_save["isotonic"] = {
                    "X_thresholds": calibrators["isotonic"].X_thresholds_.tolist(),
                    "y_thresholds": calibrators["isotonic"].y_thresholds_.tolist(),
                }
            if "threshold_tuned" in calibrators:
                cal_save["threshold_tuned"] = calibrators["threshold_tuned"]
            save_metrics(
                cal_save, output_dir,
                f"baseline_calibrator__{score_name}.json",
            )

        # ---- Apply calibrated variants to all partitions and re-evaluate ----
        for part in ("train", "calibration", "test"):
            if part not in self.splits or len(self.splits[part]) == 0:
                continue
            idx = self.splits[part]
            df_p = source.reindex(idx)
            y_p  = self.y_all[idx.to_numpy()]
            self._save_calibrated_baselines(
                output_dir, df_p, y_p, partition=part,
            )
        if len(self.holdout_idx) > 0:
            idx = self.holdout_idx
            df_p = source.reindex(idx)
            y_p  = self.y_all[idx.to_numpy()]
            self._save_calibrated_baselines(
                output_dir, df_p, y_p, partition="holdout_year",
            )

    def _save_calibrated_baselines(
        self,
        output_dir: Path,
        df_raw:     pd.DataFrame,
        y:          np.ndarray,
        partition:  str,
    ) -> None:
        """Apply self._baseline_calibrators to a partition and save metrics.

        Writes ``baseline__<score>__<partition>__<cal_method>.json`` for
        each (score, calibration-method) pair.  Mirrors the pattern used
        by model calibration (``overall__<part>__<cal>.json``).
        """
        if not getattr(self, "_baseline_calibrators", None):
            return
        scores = compute_baseline_scores(df_raw)
        for score_name, info in scores.items():
            calibrators = self._baseline_calibrators.get(score_name, {})
            if not calibrators:
                continue
            score = info["score"]
            valid = info["valid"].to_numpy()
            keep = valid
            if keep.sum() < 10:
                continue
            s = score[keep]
            y_v = y[keep]

            # Platt
            if "platt" in calibrators:
                platt = calibrators["platt"]
                p = platt.predict_proba(s.reshape(-1, 1))[:, 1]
                pred = (p >= 0.5).astype(int)
                m = binary_metrics(y_v, pred, p)
                m["n_valid"] = int(keep.sum())
                m["calibration"] = "platt"
                save_metrics(
                    m, output_dir,
                    f"baseline__{score_name}__{partition}__platt.json",
                )

            # Isotonic
            if "isotonic" in calibrators:
                iso = calibrators["isotonic"]
                p = iso.transform(s)
                pred = (p >= 0.5).astype(int)
                m = binary_metrics(y_v, pred, p)
                m["n_valid"] = int(keep.sum())
                m["calibration"] = "isotonic"
                save_metrics(
                    m, output_dir,
                    f"baseline__{score_name}__{partition}__isotonic.json",
                )

            # Threshold-tuned: original score, new threshold
            if "threshold_tuned" in calibrators:
                tuned = calibrators["threshold_tuned"]
                t = tuned["threshold"]
                pred = (s >= t).astype(int)
                m = binary_metrics(y_v, pred, s)
                m["n_valid"]   = int(keep.sum())
                m["threshold"] = t
                m["calibration"] = "threshold_tuned"
                save_metrics(
                    m, output_dir,
                    f"baseline__{score_name}__{partition}__threshold_tuned.json",
                )

    # ================================================================= #
    # Cohort-filtered evaluation
    # ================================================================= #
    def _build_cohort_masks(
        self,
        raw_missingness: pd.DataFrame,
        raw_baseline:    pd.DataFrame | None,
    ) -> tuple[dict[str, pd.Series | None], dict[str, list[str]]]:
        """Build boolean cohort masks aligned with ``raw_missingness.index``.

        Cohorts are defined by EXPLICIT variable lists (Tran 2022 NTDB
        baseline framework — see ``_COHORT_VAR_DEFS``) — NOT by catalogue
        phase tags, because phase-tagged variable sets contain too many
        sparsely-populated columns that drop cohort sizes to <1%.

        Two-stage filtering of the canonical variable list:
        1. ``_resolve_cohort_vars()`` drops vars not in the dataset at all.
        2. This method additionally drops vars that ARE in the dataset but
           are 100% NaN (e.g. NTDB ``TRAUMATYPE`` when the ECODE_LOOKUP
           join failed in trauma-build, or ``NISS`` when AISDIAGNOSIS
           wasn't joined).  Without this, every cohort would be 0/N
           because at least one row needs to satisfy ALL required vars.
           A warning is logged so the user knows to fix their build.

        Returns
        -------
        masks : dict[cohort_name, Series | None]
            Boolean mask per cohort; ``None`` if no required vars remain
            after dropping fully-missing columns.
        resolved : dict[cohort_name, list[str]]
            The actual variable list used for each cohort.
        """
        masks: dict[str, pd.Series | None] = {}
        resolved = self._resolve_cohort_vars(raw_missingness.columns)

        # Stage 2: drop any required var that is 100% NaN in this dataset
        # (would force the cohort mask to all-False)
        for cohort_name, cols in list(resolved.items()):
            kept = []
            dropped_all_nan = []
            for c in cols:
                if c in raw_missingness.columns and raw_missingness[c].all():
                    # all() on a boolean isna mask = True ⇔ 100% NaN
                    dropped_all_nan.append(c)
                else:
                    kept.append(c)
            if dropped_all_nan:
                log.warning(
                    "[%s] cohort %s: dropping %d 100%%-NaN cols from required "
                    "set: %s. Re-running with %d remaining vars: %s. "
                    "Fix your build pipeline to populate these columns "
                    "(common causes: missing PUF_ECODE_LOOKUP for TRAUMATYPE, "
                    "missing PUF_AISDIAGNOSIS for NISS).",
                    self.cfg.model_id, cohort_name,
                    len(dropped_all_nan), dropped_all_nan, len(kept), kept,
                )
            resolved[cohort_name] = kept

        for cohort_name in self._COHORT_VAR_DEFS:
            cols = resolved[cohort_name]
            if not cols:
                masks[cohort_name] = None
                log.warning(
                    "[%s] cohort %s: no required vars left after dropping "
                    "missing/all-NaN cols — cohort skipped",
                    self.cfg.model_id, cohort_name,
                )
                continue
            # cohort = rows where every *required* variable is non-NaN
            masks[cohort_name] = ~raw_missingness[cols].any(axis=1)
            n_keep = int(masks[cohort_name].sum())
            n_total = len(masks[cohort_name])
            log.info(
                "[%s] cohort %s: required %d vars (%s) → %d / %d rows complete (%.1f%%)",
                self.cfg.model_id, cohort_name, len(cols), cols,
                n_keep, n_total, 100 * n_keep / max(n_total, 1),
            )

        # baseline_complete OVERRIDE: if raw_baseline exists, use the
        # adaptive baseline-input requirement which is more permissive
        # than the canonical _COHORT_VAR_DEFS["baseline_complete"] when
        # NISS or TRISS inputs are absent from the parquet.  This matches
        # the user's stated intent: "rows where ALL of the baseline
        # metrics can be computed without imputation".
        if raw_baseline is not None and len(raw_baseline) > 0:
            adaptive = self._baseline_required_cols(raw_baseline.columns)
            # Same all-NaN filter for adaptive set
            adaptive = [c for c in adaptive
                        if c in raw_baseline.columns
                        and not raw_baseline[c].isna().all()]
            if adaptive and set(adaptive) != set(resolved.get("baseline_complete", [])):
                bl_mask = ~raw_baseline[adaptive].isna().any(axis=1)
                full_mask = pd.Series(False, index=raw_missingness.index)
                common = raw_missingness.index.intersection(bl_mask.index)
                full_mask.loc[common] = bl_mask.loc[common].astype(bool).values
                masks["baseline_complete"] = full_mask
                resolved["baseline_complete"] = adaptive
                n_keep = int(full_mask.sum())
                log.info(
                    "[%s] cohort baseline_complete (adaptive override): "
                    "required %d vars (%s) → %d / %d rows complete",
                    self.cfg.model_id, len(adaptive), adaptive,
                    n_keep, len(full_mask),
                )

        return masks, resolved


    def _evaluate_cohorts_for_partition(
        self,
        output_dir:      Path,
        partition:       str,
        partition_idx:   pd.Index,
        get_X,
        get_y,
        raw_missingness: pd.DataFrame,
        raw_baseline:    pd.DataFrame | None,
    ) -> dict[str, Any]:
        """Run cohort-filtered evaluation for a single partition.

        ``get_X`` and ``get_y`` are callables that take a kept-row index and
        return the model-ready X and y for those rows.  This decouples the
        cohort logic from where the partition's data lives (``self.data`` for
        test / holdout_year vs. an external parquet for holdout).
        """
        cohort_masks, resolved_vars = self._build_cohort_masks(
            raw_missingness, raw_baseline,
        )

        partition_total = int(len(partition_idx))
        partition_counts: dict[str, Any] = {
            "n_total": partition_total,
            "_resolved_vars": resolved_vars,   # leaf field for inspection
        }

        for cohort_name, full_mask in cohort_masks.items():
            if full_mask is None:
                partition_counts[cohort_name] = None
                continue

            in_part  = full_mask.reindex(partition_idx, fill_value=False)
            keep_idx = partition_idx[in_part.to_numpy()]
            n_cohort = int(len(keep_idx))
            partition_counts[cohort_name] = {
                "n_cohort": n_cohort,
                "n_total":  partition_total,
                "fraction": float(n_cohort / partition_total)
                            if partition_total else 0.0,
            }

            if n_cohort < 30:
                log.warning(
                    "[%s] cohort=%s partition=%s: only %d rows — "
                    "skipping metric computation (need >= 30)",
                    self.cfg.model_id, cohort_name, partition, n_cohort,
                )
                continue

            try:
                X_sub = get_X(keep_idx)
                y_sub = get_y(keep_idx)
            except Exception as exc:
                log.error(
                    "[%s] cohort=%s partition=%s: get_X/get_y raised %s",
                    self.cfg.model_id, cohort_name, partition, exc,
                )
                continue

            # ── model metrics on cohort ────────────────────────────────
            try:
                metrics = evaluate(
                    self.model, X_sub, y_sub,
                    task=self.task, classes=self.classes,
                )
                metrics["n_cohort"]        = n_cohort
                metrics["n_total"]         = partition_total
                metrics["cohort_fraction"] = float(n_cohort / partition_total) \
                    if partition_total else 0.0
                save_metrics(
                    metrics, output_dir,
                    f"overall__{partition}__cohort_{cohort_name}.json",
                )
            except Exception as exc:
                log.error(
                    "[%s] cohort=%s partition=%s: evaluate() raised %s",
                    self.cfg.model_id, cohort_name, partition, exc,
                )
                continue

            # ── calibrated variants ────────────────────────────────────
            for cal_name, cal_model in self.calibrated_models.items():
                if cal_model is None:
                    continue
                try:
                    m_cal = evaluate(
                        cal_model, X_sub, y_sub,
                        task=self.task, classes=self.classes,
                    )
                    m_cal["n_cohort"] = n_cohort
                    save_metrics(
                        m_cal, output_dir,
                        f"overall__{partition}__cohort_{cohort_name}__{cal_name}.json",
                    )
                except Exception as exc:
                    log.warning(
                        "[%s] cohort=%s partition=%s calibrated=%s: %s",
                        self.cfg.model_id, cohort_name, partition, cal_name, exc,
                    )

            # ── baselines on cohort ────────────────────────────────────
            if raw_baseline is not None:
                df_raw_sub = raw_baseline.reindex(keep_idx)
                if len(df_raw_sub) == len(y_sub):
                    try:
                        bl = baseline_metrics(df_raw_sub, y_sub)
                    except Exception as exc:
                        log.error(
                            "[%s] baseline_metrics failed for cohort=%s "
                            "partition=%s: %s",
                            self.cfg.model_id, cohort_name, partition, exc,
                        )
                        bl = {}
                    for name, m in bl.items():
                        save_metrics(
                            m, output_dir,
                            f"baseline__{name}__{partition}__cohort_{cohort_name}.json",
                        )
                else:
                    log.warning(
                        "[%s] cohort=%s partition=%s: raw_baseline len (%d) "
                        "!= y len (%d) — baselines skipped",
                        self.cfg.model_id, cohort_name, partition,
                        len(df_raw_sub), len(y_sub),
                    )

        return partition_counts

    def _evaluate_cohorts(self, output_dir: Path) -> None:
        """Top-level dispatcher: run cohort eval on every available partition.

        Handles three partitions when configured:
        - ``test``         — random test split of the main parquet
        - ``holdout_year`` — temporal year holdout (cfg.holdout_years)
        - ``holdout``      — external file holdout (cfg.holdout_dataset_path)

        Writes one ``cohort_counts.json`` per model with ALL partitions and
        cohorts.  The aggregator (``aggregate_cohort_counts``) walks these
        into ``cohort_counts.csv``.
        """
        if self.task != "binary":
            return

        cohort_counts: dict[str, Any] = {}

        # Helpers shared by test and temporal holdout (both live in self.data)
        def _main_get_X(keep_idx: pd.Index) -> pd.DataFrame:
            return self.data.loc[keep_idx, self.predictors]

        def _main_get_y(keep_idx: pd.Index) -> np.ndarray:
            pos = self.data.index.get_indexer(keep_idx)
            return self.y_all[pos]

        # Round 8: prefer the IMPUTED baseline so cohorts evaluate baselines
        # on the SAME populations as the model — no drop-NaN reduction.
        baseline_source = (self._imputed_baseline
                           if getattr(self, "_imputed_baseline", None) is not None
                           else getattr(self, "_raw_baseline", None))

        # ── Partition 1: random test split ────────────────────────────────
        cohort_counts["test"] = self._evaluate_cohorts_for_partition(
            output_dir,
            partition="test",
            partition_idx=self.splits["test"],
            get_X=_main_get_X,
            get_y=_main_get_y,
            raw_missingness=self._raw_missingness,
            raw_baseline=baseline_source,
        )

        # ── Partition 2: temporal year holdout ────────────────────────────
        if len(self.holdout_idx) > 0:
            cohort_counts["holdout_year"] = self._evaluate_cohorts_for_partition(
                output_dir,
                partition="holdout_year",
                partition_idx=self.holdout_idx,
                get_X=_main_get_X,
                get_y=_main_get_y,
                raw_missingness=self._raw_missingness,
                raw_baseline=baseline_source,
            )

        # ── Partition 3: external file holdout ────────────────────────────
        if hasattr(self, "_ext_holdout_state"):
            st = self._ext_holdout_state

            def _ext_get_X(keep_idx: pd.Index, _X=st["X"]) -> pd.DataFrame:
                return _X.loc[keep_idx]

            def _ext_get_y(keep_idx: pd.Index, _X=st["X"], _y=st["y"]) -> np.ndarray:
                pos = _X.index.get_indexer(keep_idx)
                return _y[pos]

            cohort_counts["holdout"] = self._evaluate_cohorts_for_partition(
                output_dir,
                partition="holdout",
                partition_idx=st["X"].index,
                get_X=_ext_get_X,
                get_y=_ext_get_y,
                raw_missingness=st["raw_missingness"],
                raw_baseline=st.get("imputed_baseline", st["raw_baseline"]),
            )

        # Separate the resolved-vars info out of cohort_counts (which is the
        # numerical case-count file) into its own cohort_variables.json so
        # the user can inspect what variables were required for each cohort
        # in each partition.  Also strip them out of cohort_counts before
        # saving to keep the counts JSON purely numerical.
        cohort_variables: dict[str, Any] = {}
        for part_name, part_blob in cohort_counts.items():
            if isinstance(part_blob, dict) and "_resolved_vars" in part_blob:
                cohort_variables[part_name] = part_blob.pop("_resolved_vars")

        save_metrics(cohort_counts, output_dir, "cohort_counts.json")
        save_metrics(cohort_variables, output_dir, "cohort_variables.json")
        log.info("[%s] Cohort counts: %s", self.cfg.model_id, cohort_counts)
        log.info("[%s] Cohort variables: %s", self.cfg.model_id, cohort_variables)

    # ================================================================= #
    # Persistence
    # ================================================================= #
    @staticmethod
    def _resolve_actual_model_type(model: object, family: str) -> str:
        """Return a human-readable name for the fitted model.

        For AutoML (FLAML) the best learner name is extracted from the wrapper.
        For sklearn pipelines the final step class name is used.
        For everything else the family string is returned unchanged.
        """
        # FLAML wrapper
        if hasattr(model, "automl"):
            try:
                best = getattr(model.automl, "best_estimator", None)
                learner = getattr(model.automl, "best_learner", None)
                if learner:
                    return f"flaml:{learner}"
                if best is not None:
                    return f"flaml:{type(best).__name__}"
            except Exception:
                pass
            # model.model is set after fit()
            inner = getattr(model, "model", None)
            if inner is not None:
                return f"flaml:{type(inner).__name__}"
            return "flaml:unknown"

        # sklearn Pipeline
        if hasattr(model, "steps"):
            try:
                final_name, final_est = model.steps[-1]
                return f"pipeline:{type(final_est).__name__}"
            except Exception:
                pass

        # CalibratedClassifierCV
        if hasattr(model, "calibrated_classifiers_"):
            try:
                base = model.estimator
                return f"calibrated:{type(base).__name__}"
            except Exception:
                pass

        return type(model).__name__

    def _emit_empty_run(self, outputs_root: Path) -> "ModelArtifact":
        """Write NaN-filled metrics + a minimal artifact when imputer_check
        leaves no usable predictors.

        This makes the run "complete" from the resume tracker's point of
        view (it sees overall__test.json and won't retry the combo) while
        clearly signalling, via NaN values and a flag in config.json, that
        the combo was abandoned due to the imputer check.
        """
        import json as _json

        metrics_dir = outputs_root / "metrics" / self.cfg.model_id
        metrics_dir.mkdir(parents=True, exist_ok=True)

        nan_metrics = {
            "AUROC": None, "AUPRC": None, "Brier": None,
            "balanced_accuracy": None, "f1": None,
            "recall": None, "precision": None,
            "_skipped_reason": "imputer_check removed all predictors",
        }
        for partition in ("train", "calibration", "test", "holdout"):
            (metrics_dir / f"overall__{partition}.json").write_text(
                _json.dumps(nan_metrics, indent=2)
            )

        # Minimal config.json so downstream tools can read the combo + see
        # why it was skipped.
        models_dir = outputs_root / "models" / self.cfg.model_id
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / "config.json").write_text(_json.dumps({
            "model_id": self.cfg.model_id,
            "family": self.cfg.model_family,
            "task": self.task,
            "predictor_cols": [],
            "numeric_cols": [],
            "categorical_cols": [],
            "config": {
                "predictor_type":        self.cfg.predictor_type,
                "phase_cutoff":          self.cfg.phase_cutoff,
                "inclusion_strategy":    self.cfg.inclusion_strategy,
                "imputer_method":        self.cfg.imputer_method,
                "imputer_check":         self.cfg.imputer_check,
                "model_family":          self.cfg.model_family,
                "calibration":           self.cfg.calibration,
                "missingness_threshold": self.cfg.missingness_threshold,
                "data_augmentation":     self.cfg.data_augmentation or "none",
                "n_predictors":          0,
                "skipped":               True,
                "skipped_reason":        "imputer_check removed all predictors",
            },
        }, indent=2))

        log.info("[%s] Empty run recorded (NaN metrics + skipped flag).",
                 self.cfg.model_id)
        # Return a lightweight artifact-less sentinel; callers only use the
        # return value for logging, so None-model is fine.
        return None

    def _save_artifact(self, output_root: Path) -> ModelArtifact:
        actual_type = self._resolve_actual_model_type(self.model, self.cfg.model_family)
        log.info("[%s] Actual model type: %s", self.cfg.model_id, actual_type)

        artifact = ModelArtifact(
            model_id=self.cfg.model_id,
            family=self.cfg.model_family,
            task=self.task,
            target_spec={"name": self.cfg.target.name, "kind": self.cfg.target.kind,
                          "spec": self.cfg.target.spec},
            predictor_cols=self.predictors,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            transformers=self.transformers,
            imputer=self.imputer,
            model=self.model,
            calibrated_models=self.calibrated_models,
            target_inverse=self.target_inverse,
            config={
                "predictor_type":        self.cfg.predictor_type,
                "phase_cutoff":          self.cfg.phase_cutoff,
                "inclusion_strategy":    self.cfg.inclusion_strategy,
                "imputer_method":        self.cfg.imputer_method,
                "imputer_check":         self.cfg.imputer_check,
                "model_family":          self.cfg.model_family,
                "calibration":           self.cfg.calibration,
                "missingness_threshold": self.cfg.missingness_threshold,
                "data_augmentation":     self.cfg.data_augmentation or "none",
                "correlation_threshold": self.cfg.correlation_threshold,
                "sample_fraction":       self.cfg.sample_fraction,
                "random_state":          self.cfg.random_state,
                "holdout_years":         self.cfg.holdout_years,
                "n_predictors":          len(self.predictors) if self.predictors else 0,
            },
            extras={
                "dataset_path":         str(self.cfg.dataset_path),
                "holdout_dataset_path": str(self.cfg.holdout_dataset_path)
                                        if self.cfg.holdout_dataset_path else None,
                "holdout_years":        self.cfg.holdout_years,
            },
        )
        # Patch actual_model_type into the artifact before saving so
        # aggregate_all_metrics can read it from config.json
        artifact.extras["actual_model_type"] = actual_type
        artifact.save(output_root)
        return artifact
