"""`trauma-train` CLI — grid runner for the trauma_ml experiments.

New axes vs v0.1
----------------
* `--calibrations`     : none | platt | isotonic  (separate run per choice)
* `--holdout-dataset`  : external holdout parquet, evaluated after training
* `--sample-fraction`  : subsample the training pool for quick iteration
"""
from __future__ import annotations

import argparse
import gc
import itertools
import logging
import traceback
from pathlib import Path

from ..catalogue import Catalogue
from ..targets import TargetSpec
from ..trainer import Trainer, TrainerConfig
from ._common import load_yaml, resolve_repo_root, setup_logging

log = logging.getLogger(__name__)


# Round 22: per-target output subfolder convention
# ---------------------------------------------------------------------
# By default, each target writes to outputs/<target_subfolder>/...
# instead of dumping everything into outputs/ flat.  This avoids
# collisions when training mortality + iss_band + niss_band in the same
# project and lets the aggregator produce per-target Pareto fronts.
#
# The user can override on a per-run basis with --outputs-root, in which
# case THIS dict is bypassed and everything goes to that explicit path.
TARGET_SUBFOLDER = {
    "in_hospital_mortality": "mortality",
    # Both casings supported because config/experiment_grids.yaml has used
    # uppercase 'ISS_band' / 'NISS_band' historically.  Either YAML name
    # (or the lowercase form in the slurms) maps to the same on-disk path.
    "iss_band":               "iss_band",
    "ISS_band":               "iss_band",
    "niss_band":              "niss_band",
    "NISS_band":              "niss_band",
    # Round 52: binary (ISS>=16 / NISS>=16) band targets.
    "iss_band_binary":        "iss_band_binary",
    "ISS_band_binary":        "iss_band_binary",
    "niss_band_binary":       "niss_band_binary",
    "NISS_band_binary":       "niss_band_binary",
    # Add more here as new targets are introduced.  Default behaviour
    # (when target-name not in this dict) is to use the target name as
    # the subfolder verbatim.
}


def _outputs_subfolder_for(target_name: str) -> str:
    """Return the per-target subfolder name (e.g. 'mortality')."""
    return TARGET_SUBFOLDER.get(target_name, target_name)


# Round 52: built-in target definitions that are ALWAYS available, even if the
# user's config/experiment_grids.yaml does not list them.  They are merged in
# only when absent from the YAML (the YAML always wins on name collisions), so
# `--targets ISS_band_binary` / `NISS_band_binary` resolve out of the box and
# no manual YAML edit is required.  Add new always-on targets here.
_BUILTIN_TARGET_DEFS: list[dict] = [
    {"name": "ISS_band_binary",  "kind": "binary_threshold",
     "spec": {"variable": "ISS",  "threshold": 15}},   # y = 1 when ISS  > 15 (ISS  >= 16)
    {"name": "NISS_band_binary", "kind": "binary_threshold",
     "spec": {"variable": "NISS", "threshold": 15}},   # y = 1 when NISS > 15 (NISS >= 16)
]


def _parse_target_defs(target_cfg: list[dict]) -> dict[str, TargetSpec]:
    specs = {t["name"]: TargetSpec(name=t["name"], kind=t["kind"], spec=t["spec"])
             for t in target_cfg}
    # Merge built-ins for any name the YAML did not define (YAML wins).
    for d in _BUILTIN_TARGET_DEFS:
        if d["name"] not in specs:
            specs[d["name"]] = TargetSpec(name=d["name"], kind=d["kind"], spec=d["spec"])
            log.info("Target %r not defined in YAML — using built-in default "
                     "(kind=%s, spec=%s).", d["name"], d["kind"], d["spec"])
    return specs


def _build_grid(
    targets, predictor_types, phase_cutoffs, inclusion_strategies,
    imputers, model_families, calibrations, missingness_thresholds, augmentations,
    imputer_checks=(False,),
    aug_targets=("in_hospital_mortality",),
    calib_targets=("in_hospital_mortality",),
):
    """Enumerate the experiment grid.

    Round 37: ``imputer_checks`` is the LAST axis in itertools.product so
    that adding ``True`` to it appends new combos at the END of the model_id
    sequence, leaving every previously-assigned model_id unchanged.  This
    is what lets you keep already-run results and only train the new combos.

    Round 52/53: augmentation and calibration are gated by SEPARATE target
    sets, because they have different applicability:
      - ``calib_targets`` — probability calibration (Platt/isotonic) only
        makes sense for BINARY targets (mortality + binary_threshold bands).
      - ``aug_targets`` — SMOTE/ADASYN oversampling works for BINARY AND
        MULTICLASS, so this set also includes the 4-class ordinal_bands
        targets (ISS_band / NISS_band).  Augmentation is still skipped at
        train time when imputer='none' (NaN input) — see _augment_if_needed.

    Two skip rules keep the grid sane:
      - ``imputer_method == "none"`` is only valid for NaN-tolerant
        families (xgboost / lightgbm / catboost / flaml).  Other families
        are skipped.
      - When ``imputer_method == "none"`` there is no imputation to check,
        so the ``imputer_check`` dimension collapses — we only keep the
        ``imputer_check == False`` variant (point 5).  This prevents
        duplicate identical runs.
    """
    NAN_TOLERANT = {"xgboost", "lightgbm", "catboost", "flaml", "doshi_ffnn_icd"}
    aug_targets = set(aug_targets)
    calib_targets = set(calib_targets)
    grid = []
    for imp_check in imputer_checks:
        combos = itertools.product(
            targets, predictor_types, phase_cutoffs, inclusion_strategies,
            imputers, model_families, calibrations, missingness_thresholds,
            augmentations,
        )
        for (target, ptype, phase, incl, imp, family, calib, missing,
             aug) in combos:
            if aug is not None and target not in aug_targets:
                continue
            if calib != "none" and target not in calib_targets:
                continue
            # imputer=none only for NaN-tolerant families
            if imp == "none" and family not in NAN_TOLERANT:
                continue
            # Point 5: with no imputation there's nothing to check — keep
            # only the imputer_check=False variant to avoid duplicate runs.
            if imp == "none" and imp_check:
                continue
            grid.append({
                "target":                target,
                "predictor_type":        ptype,
                "phase_cutoff":          phase,
                "inclusion_strategy":    incl,
                "imputer_method":        imp,
                "model_family":          family,
                "calibration":           calib,
                "missingness_threshold": missing / 100.0,
                "data_augmentation":     aug,
                "imputer_check":         imp_check,
            })
    return grid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the trauma_ml experiment grid.")
    parser.add_argument("--paths-config", default="config/paths.yaml")
    parser.add_argument("--grid-config",  default="config/experiment_grids.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--holdout-dataset", default=None,
                         help="External holdout parquet (evaluated after training)")
    parser.add_argument(
        "--holdout-years",
        nargs="+",
        type=int,
        default=None,
        help="NTDB admission year(s) to use as the temporal holdout "
             "(excluded from train/cal/test pool). Example: --holdout-years 2023 2024",
    )
    parser.add_argument(
        "--pareto-metric",
        default="AUPRC",
        help="Metric used for worst-case Pareto computation (default: AUPRC). "
             "Any metric in the subgroup CSVs is valid (e.g. AUROC, f1, recall).",
    )
    parser.add_argument(
        "--pareto-partitions",
        nargs="+",
        default=["test", "holdout_year"],
        help="Partitions included in worst-case Pareto (default: test holdout_year).",
    )
    parser.add_argument(
        "--pareto-objectives",
        nargs="+",
        default=None,
        help="Explicit Pareto objective column names. If set, overrides the "
             "automatic worst-case derivation from --pareto-metric / --pareto-partitions.",
    )
    parser.add_argument("--outputs-root", default=None)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id",   type=int, default=None)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--predictor-types", nargs="+", default=None)
    parser.add_argument(
        "--phase-cutoffs", nargs="+", default=None,
        help=(
            "Phase-based: 'On-scene', 'On-scene + ED arrival', "
            "'On-scene + ED arrival + In-hospital'. "
            "Baseline-feature based (round 11): 'iss_only', 'niss_only', "
            "'triss_inputs', 'all_baseline_inputs' — pin the predictor set "
            "to the exact inputs of one clinical baseline score for direct "
            "head-to-head comparison."
        ),
    )
    parser.add_argument("--inclusion-strategies", nargs="+", default=None)
    parser.add_argument("--imputers", nargs="+", default=None,
                         help="Imputer methods. Use 'none' to skip imputation "
                              "(only valid for NaN-tolerant families: xgboost, "
                              "lightgbm, catboost, flaml).")
    parser.add_argument("--imputer-check", action="store_true",
                         help="After fitting the imputer, evaluate per-feature "
                              "reconstruction quality on the CALIBRATION set and "
                              "drop features the imputer can't recover reliably. "
                              "Adds imputer_check=True combos to the grid (the "
                              "imputer_check=False combos are always included so "
                              "your existing results stay valid).")
    parser.add_argument("--model-families", nargs="+", default=None)
    parser.add_argument("--calibrations", nargs="+", default=None,
                         choices=["none", "platt", "isotonic"])
    parser.add_argument("--missingness-thresholds", nargs="+", type=int, default=None)
    parser.add_argument("--augmentations", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--bootstrap-ci", type=int, default=0, metavar="N",
        help="If >0, attach stratified percentile bootstrap CIs (N resamples) "
             "to the test + holdout (+ cohort) metric JSONs as "
             "<metric>__ci_low/__ci_high. Off by default. 200-1000 is typical; "
             "on the full NTDB holdout this adds noticeable time.")
    parser.add_argument(
        "--ci-alpha", type=float, default=0.05,
        help="Significance level for bootstrap CIs (default 0.05 -> 95%% CI).")
    parser.add_argument(
        "--plot-ci-bootstrap", type=int, default=200, metavar="N",
        help="Resamples for the AUROC/AUPRC confidence intervals shown in the "
             "binary ROC/PR plot legends (default 200, ON by default; 0 to "
             "disable). Cheap AUC/AP-only bootstrap, separate from the heavier "
             "all-metric --bootstrap-ci. ROC/PR render only for binary targets.")
    parser.add_argument(
        "--use-gpu", default="auto", choices=["auto", "force", "never"],
        help=(
            "GPU policy for model factories that support it (xgboost, "
            "lightgbm, catboost, tabpfn, tabnet). 'auto' uses GPU if "
            "torch.cuda.is_available(), 'force' requires GPU and errors "
            "otherwise, 'never' always uses CPU. Default: auto."
        ),
    )
    parser.add_argument(
        "--model-id-prefix", default="model",
        help=(
            "Prefix for generated model_ids (default 'model' -> "
            "'model_0000', 'model_0001'…).  When running multiple Slurm "
            "jobs in parallel, set distinct prefixes per job (e.g. "
            "'lgb', 'xgb', 'logreg') so the per-model output directories "
            "don't collide.  Allowed chars: [a-zA-Z0-9_-]."
        ),
    )
    parser.add_argument(
        "--no-aggregate", action="store_true",
        help=(
            "Skip the post-loop aggregation step that writes "
            "all_metrics.csv / baseline_metrics.csv / cohort_counts.csv.  "
            "Use this when running multiple Slurm jobs in parallel — let "
            "each job train its share of the grid, then run "
            "`trauma-aggregate-metrics` once at the end against the merged "
            "outputs directory to produce the unified Pareto."
        ),
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help=(
            "Disable SHAP plotting.  Recommended for heavy-grid runs (large "
            "random forests, big tree ensembles) on glibc<2.28 systems where "
            "SHAP's TreeExplainer can segfault on the underlying native code. "
            "ROC/PR + confusion-matrix plots still produced."
        ),
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help=(
            "Disable ALL plot generation (ROC/PR + confusion matrix + SHAP). "
            "Useful for very large grids where plots dominate per-model time, "
            "or for re-running just metrics on existing models."
        ),
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help=(
            "Disable the round-26 resume tracker.  By default, combos that "
            "already wrote `<outputs_root>/metrics/<model_id>/overall__test.json` "
            "are SKIPPED, so resubmitting after a walltime hit picks up where it "
            "left off.  Pass --no-resume to retrain every combo from scratch."
        ),
    )
    parser.add_argument(
        "--no-prep-cache", action="store_true",
        help=(
            "Disable the round-61 preprocessing/model cache.  By default the "
            "runner fits the encoders+imputer ONCE per (target, phase, "
            "imputer, missingness, family) group and reuses them across every "
            "calibration/augmentation variant, and reuses the trained base "
            "model across calibration variants (binary).  Saved artifacts are "
            "identical either way; this flag forces a fresh fit per combo."
        ),
    )
    parser.add_argument(
        "--tune-hparams", action="store_true",
        help=(
            "Wrap each model's .fit in sklearn RandomizedSearchCV with the "
            "search distribution defined in trainer.py::_wrap_with_random_search. "
            "Tran 2022 did this and reported negligible impact, so default is "
            "OFF.  Multiplies per-fit cost by --n-search-iter * --cv-folds."
        ),
    )
    parser.add_argument(
        "--n-search-iter", type=int, default=10,
        help="Number of random search points when --tune-hparams is set",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=3,
        help="Number of CV folds inside RandomizedSearchCV when --tune-hparams is set",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    repo_root = resolve_repo_root()
    paths_cfg = load_yaml(repo_root / args.paths_config)
    grid_cfg  = load_yaml(repo_root / args.grid_config)

    target_specs = _parse_target_defs(grid_cfg["targets"])
    targets = args.targets or [t["name"] for t in grid_cfg["targets"]]

    # Round 16: surface a clearer error than KeyError when the user passes
    # --targets X for a name that's not defined in the YAML config.  This
    # was hitting users running `--targets survival_time` without first
    # adding survival_time to config/experiment_grids.yaml.  Survival isn't
    # yet supported end-to-end (the target builder raises NotImplementedError
    # at trainer time too), but at least give a clear error here.
    missing_targets = [t for t in targets if t not in target_specs]
    if missing_targets:
        available = sorted(target_specs.keys())
        raise SystemExit(
            f"Target(s) {missing_targets} are not defined in "
            f"config/experiment_grids.yaml.  Available: {available}.\n"
            f"To add a new target, edit config/experiment_grids.yaml and "
            f"add a `targets:` entry with `name`, `kind`, and `spec`. "
            f"NOTE: survival targets (`kind: survival`) are NOT yet supported "
            f"end-to-end — the target builder raises NotImplementedError."
        )
    predictor_types = args.predictor_types or grid_cfg["predictor_types"]
    phase_cutoffs  = args.phase_cutoffs  or grid_cfg["phase_cutoffs"]
    inclusion_strategies = args.inclusion_strategies or grid_cfg["inclusion_strategies"]
    inclusion_strategies = [None if str(s).lower() == "none" else s
                             for s in inclusion_strategies]
    imputers = args.imputers or grid_cfg["imputers"]
    model_families = args.model_families or grid_cfg["model_families"]
    calibrations = args.calibrations or grid_cfg.get(
        "calibrations", ["none", "platt", "isotonic"]
    )
    missingness_thresholds = args.missingness_thresholds or grid_cfg["missingness_thresholds"]
    augmentations_raw = args.augmentations if args.augmentations is not None \
                         else grid_cfg["data_augmentation"]
    augmentations = [None if a in (None, "none", "None", "null") else a
                      for a in augmentations_raw]

    # Round 37: imputer_check axis.  Always include False (so existing
    # results remain valid and keep their model_ids); add True only when
    # --imputer-check is passed.  True combos enumerate LAST.
    imputer_checks = (False, True) if args.imputer_check else (False,)

    # Round 52: targets eligible for augmentation + calibration = BINARY ones
    # (mortality + any binary_threshold band targets such as ISS>=16/NISS>=16).
    # Multiclass ordinal_bands targets are excluded (calib/aug don't apply).
    # Round 53: calibration only for BINARY targets; augmentation for binary
    # AND multiclass (ordinal_bands) targets — SMOTE/ADASYN handle multiclass.
    calib_targets = {
        name for name, spec in target_specs.items()
        if spec.kind in ("binary", "binary_threshold")
    }
    aug_targets = {
        name for name, spec in target_specs.items()
        if spec.kind in ("binary", "binary_threshold", "ordinal_bands")
    }

    grid = _build_grid(
        targets, predictor_types, phase_cutoffs, inclusion_strategies,
        imputers, model_families, calibrations, missingness_thresholds, augmentations,
        imputer_checks=imputer_checks,
        aug_targets=aug_targets,
        calib_targets=calib_targets,
    )
    log.info("Full grid has %d combinations", len(grid))

    if args.dry_run:
        for i, combo in enumerate(grid):
            print(f"{i:4d} | {combo}")
        return

    end_id = args.end_id if args.end_id is not None else len(grid) - 1
    grid = [(i, combo) for i, combo in enumerate(grid)
             if args.start_id <= i <= end_id]
    log.info("Running %d combinations (id %d..%d)", len(grid), args.start_id, end_id)

    catalogue = Catalogue(
        repo_root / paths_cfg["variable_catalogue"],
        sheet_name=paths_cfg["catalogue_sheet_name"],
        header_row=paths_cfg["catalogue_header_row"],
    )
    dataset_path = Path(args.dataset) if args.dataset else (
        repo_root / paths_cfg["datasets_dir"] / "unified_train.parquet"
    )
    if not dataset_path.exists():
        legacy = repo_root / paths_cfg["datasets_dir"] / "unified_ntdb.parquet"
        if legacy.exists():
            dataset_path = legacy
        else:
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}. Run `trauma-build` first."
            )
    holdout_path = Path(args.holdout_dataset) if args.holdout_dataset else None
    if holdout_path is not None and not holdout_path.exists():
        raise FileNotFoundError(f"Holdout dataset not found: {holdout_path}")
    holdout_years = args.holdout_years or []
    # Round 22: outputs_root is now PER-TARGET by default.  If the user
    # didn't pass --outputs-root, we derive it from each combo's target.
    # If they did pass --outputs-root, we honour it as the "trunk" and
    # still append the target subfolder under it (so multi-target runs
    # don't collide), unless they explicitly want flat layout.
    outputs_root_explicit = Path(args.outputs_root) if args.outputs_root else None
    outputs_root_default  = repo_root / paths_cfg["outputs_root"]

    # Validate prefix (avoid path-injection / weird chars)
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.model_id_prefix):
        raise SystemExit(
            f"Invalid --model-id-prefix={args.model_id_prefix!r}: "
            f"must match [A-Za-z0-9_-]+"
        )

    # Per-target outputs_root (used for aggregation step at end).  Track
    # which targets were actually trained so we aggregate each one.
    targets_trained: set[str] = set()

    # Round 26: resume tracker — combos that already have a completed metrics
    # output get skipped on re-submission.  This means if a slurm hits its
    # walltime mid-grid, you can resubmit the exact same slurm and it picks up
    # where it left off.  The completion sentinel is the existence of
    # `<outputs_root>/metrics/<model_id>/overall__test.json`, written by the
    # trainer at the end of evaluation.  If the file exists AND is non-empty,
    # the model is considered done and we skip it.
    n_skipped = 0
    n_attempted = 0

    def _resolve_outputs_root(target_name):
        sub = _outputs_subfolder_for(target_name)
        base = (outputs_root_explicit if outputs_root_explicit is not None
                else outputs_root_default)
        return base / sub

    def _build_cfg(model_id, combo):
        return TrainerConfig(
            model_id=model_id,
            dataset_path=dataset_path,
            catalogue=catalogue,
            target=target_specs[combo["target"]],
            predictor_type=combo["predictor_type"],
            phase_cutoff=combo["phase_cutoff"],
            inclusion_strategy=combo["inclusion_strategy"],
            imputer_method=combo["imputer_method"],
            imputer_check=combo.get("imputer_check", False),
            model_family=combo["model_family"],
            calibration=combo["calibration"],
            missingness_threshold=combo["missingness_threshold"],
            data_augmentation=combo["data_augmentation"],
            holdout_dataset_path=holdout_path,
            holdout_years=holdout_years,
            sample_fraction=args.sample_fraction,
            n_jobs=args.n_jobs,
            use_gpu=args.use_gpu,
            n_bootstrap=args.bootstrap_ci,
            ci_alpha=args.ci_alpha,
            plot_ci_bootstrap=args.plot_ci_bootstrap,
            generate_plots=not args.no_plots,
            enable_shap=not args.no_shap,
            tune_hparams=args.tune_hparams,
            n_search_iter=args.n_search_iter,
            cv_folds=args.cv_folds,
        )

    def _cleanup():
        # Round 16: aggressive cleanup between models.  Long grids were
        # segfaulting AFTER successful completion — matplotlib figure registry
        # + accumulated SHAP native arrays + sklearn fitted-attribute closures
        # pin memory.  Force-release so the next iteration starts fresh.
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass
        gc.collect()

    # ---- Round 61: preprocessing/model cache --------------------------- #
    # Group combos whose preprocessing is identical (everything EXCEPT
    # calibration + augmentation).  Within a group we fit encoders+imputer
    # once and reuse the trained base model across calibration variants.
    def _prep_key(combo):
        return (combo["target"], combo["predictor_type"], combo["phase_cutoff"],
                combo["inclusion_strategy"], combo["missingness_threshold"],
                combo["imputer_method"], combo.get("imputer_check", False),
                combo["model_family"])

    def _calib_order(item):
        _, c = item
        return (str(c["data_augmentation"]),
                0 if (c["calibration"] or "none") == "none" else 1,
                str(c["calibration"]))

    from collections import OrderedDict
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for idx, combo in grid:
        groups.setdefault(_prep_key(combo), []).append((idx, combo))

    for _pk, members in groups.items():
        members = sorted(members, key=_calib_order)
        trainer = None            # prepared Trainer, reused across the group
        prep_ok = True
        base_models: dict = {}    # data_augmentation -> uncalibrated base (binary)
        for idx, combo in members:
            model_id = f"{args.model_id_prefix}_{idx:04d}"
            outputs_root = _resolve_outputs_root(combo["target"])
            targets_trained.add(combo["target"])

            sentinel = outputs_root / "metrics" / model_id / "overall__test.json"
            if (not args.no_resume and sentinel.exists()
                    and sentinel.stat().st_size > 0):
                n_skipped += 1
                log.info("[%s] SKIP (already completed: %s)", model_id, sentinel)
                continue
            n_attempted += 1
            cfg = _build_cfg(model_id, combo)
            log.info("=" * 72)
            log.info("[%s] %s", model_id, combo)
            log.info("    -> outputs_root: %s", outputs_root)
            log.info("=" * 72)
            try:
                if args.no_prep_cache:
                    Trainer(cfg).run(outputs_root)
                    continue

                if trainer is None:
                    _t = Trainer(cfg)
                    prep_ok = _t.prepare(outputs_root)   # may raise -> trainer stays None
                    trainer = _t
                else:
                    trainer.reset_for_next_combo(cfg)

                if not prep_ok:
                    # imputer_check dropped every predictor: the preprocessing is
                    # identical for the whole group, so each combo is an empty run.
                    trainer._emit_empty_run(outputs_root)
                    continue

                aug = combo["data_augmentation"]
                is_binary = (trainer.task == "binary")
                base = base_models.get(aug) if is_binary else None
                trainer.fit_evaluate_save(outputs_root, cached_base_model=base)
                if is_binary and base is None and trainer.model is not None:
                    base_models[aug] = trainer.model     # reuse for platt/isotonic
            except Exception as exc:
                log.error("[%s] FAILED: %s", model_id, exc)
                log.debug(traceback.format_exc())
                continue
            finally:
                _cleanup()

    print(f"[OK] Grid complete. Trained {len(targets_trained)} target(s): "
          f"{sorted(targets_trained)}")
    if n_skipped > 0:
        print(f"[OK] Resume tracker: {n_skipped} combo(s) skipped (already "
              f"completed), {n_attempted} combo(s) attempted in this run.")

    # ── Auto-aggregate metrics PER TARGET ──────────────────────────────────
    # Round 22: each target wrote to its own subfolder; aggregate each one
    # independently so we get per-target unified Pareto fronts.
    if args.no_aggregate:
        for target_name in sorted(targets_trained):
            sub = _outputs_subfolder_for(target_name)
            target_root = (outputs_root_explicit or outputs_root_default) / sub
            print(f"[OK] --no-aggregate set; skipping aggregation. "
                  f"Run trauma-aggregate-metrics later against {target_root}.")
        return

    for target_name in sorted(targets_trained):
        sub = _outputs_subfolder_for(target_name)
        target_root = (outputs_root_explicit or outputs_root_default) / sub
        metrics_root = target_root / "metrics"
        models_root  = target_root / "models"
        all_metrics_path = target_root / "all_metrics.csv"
        if not metrics_root.exists():
            log.warning("[%s] metrics_root %s not found; skipping aggregation",
                        target_name, metrics_root)
            continue

        # Round 23: auto-pick the Pareto comparison metric per target kind.
        # For binary mortality, AUPRC is appropriate (rare-event sensitive).
        # For multiclass severity-band targets, balanced_accuracy is the
        # right choice (avg of per-class recall, robust to class imbalance —
        # rare 'Profound' band would be ignored by plain accuracy).
        # CLI override (--pareto-metric) still wins if user passed one.
        target_kind = target_specs[target_name].kind
        if args.pareto_metric == "AUPRC" and target_kind != "binary":
            chosen_pareto_metric = "balanced_accuracy"
            log.info(
                "[%s] target_kind=%s — using balanced_accuracy as Pareto metric "
                "(override --pareto-metric to change)",
                target_name, target_kind,
            )
        else:
            chosen_pareto_metric = args.pareto_metric

        log.info("=" * 72)
        log.info("[%s] Aggregating model metrics → %s (Pareto on %s)",
                 target_name, all_metrics_path, chosen_pareto_metric)
        log.info("=" * 72)
        try:
            from ..evaluation import (
                aggregate_all_metrics, aggregate_cohort_counts,
                aggregate_baseline_metrics_long,
            )
            df_all = aggregate_all_metrics(
                metrics_root=metrics_root,
                models_root=models_root,
                output_path=all_metrics_path,
                pareto_metric=chosen_pareto_metric,
                pareto_partitions=tuple(args.pareto_partitions),
                pareto_objectives=args.pareto_objectives or None,
            )
            n_front = int(df_all["pareto_front"].sum()) if not df_all.empty else 0
            print(f"[OK] [{target_name}] all_metrics.csv written: {len(df_all)} models, "
                  f"{n_front} on Pareto front → {all_metrics_path}")

            main_columns = list(df_all.columns) if not df_all.empty else None
            for cohort in ("onsite_complete", "ed_complete", "baseline_complete"):
                cohort_path = target_root / f"all_metrics__cohort_{cohort}.csv"
                df_c = aggregate_all_metrics(
                    metrics_root=metrics_root,
                    models_root=models_root,
                    output_path=cohort_path,
                    pareto_metric=chosen_pareto_metric,
                    pareto_partitions=tuple(args.pareto_partitions),
                    pareto_objectives=args.pareto_objectives or None,
                    cohort=cohort,
                    column_template=main_columns,
                )
                if df_c.empty:
                    log.warning(
                        "[%s] cohort=%s: no metrics found — no model wrote cohort JSONs",
                        target_name, cohort,
                    )
                    continue
                n_front_c = int(df_c["pareto_front"].sum())
                print(f"[OK] [{target_name}] all_metrics__cohort_{cohort}.csv: "
                      f"{len(df_c)} models, {n_front_c} Pareto → {cohort_path}")

            counts_path = target_root / "cohort_counts.csv"
            counts_df = aggregate_cohort_counts(
                metrics_root=metrics_root, output_path=counts_path,
            )
            if not counts_df.empty:
                print(f"[OK] [{target_name}] cohort_counts.csv: "
                      f"{len(counts_df)} rows → {counts_path}")

            # Baseline scores (ISS/NISS/TRISS) only meaningful for binary
            # mortality target — skip for multiclass.
            target_kind = target_specs[target_name].kind
            if target_kind == "binary":
                baseline_path = target_root / "baseline_metrics.csv"
                baseline_df = aggregate_baseline_metrics_long(
                    metrics_root=metrics_root, output_path=baseline_path,
                )
                if not baseline_df.empty:
                    print(f"[OK] [{target_name}] baseline_metrics.csv: "
                          f"{len(baseline_df)} rows → {baseline_path}")
        except Exception as exc:
            log.error("[%s] aggregate_all_metrics failed: %s", target_name, exc)
            log.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
