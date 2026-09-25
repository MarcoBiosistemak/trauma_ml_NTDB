#!/usr/bin/env python3
"""Free disk space by deleting model ARTIFACTS for models that are never the
top performer on ANY evaluation metric -- overall OR within any demographic
subgroup -- in either the test or the external holdout partition. Metrics
and plots are NEVER touched, and config.json is NEVER touched either --
only the ARTIFACT_FILENAME file (artifact.pkl by default) inside
outputs/<target>/models/<model_id>/ is ever deleted. This means every
deleted model's exact configuration is still on disk afterwards, so it
can be reproduced by re-running trauma-train with that exact combo.

A model_id is KEPT (artifact preserved) if it is the single best model for
at least one of:
  (a) an OVERALL metric   -- overall__test.json / overall__holdout.json
                              (and their __platt / __isotonic variants)
  (b) a SUBGROUP metric    -- metrics/<model_id>/subgroups/subgroups_by_<axis>.csv
                              (test) and .../subgroups_holdout/subgroups_by_<axis>.csv
                              (external holdout) -- one row per demographic
                              level (e.g. age_group=elderly), one column per
                              metric.
"Best" means max for metrics that should be maximized (AUROC, accuracy, F1,
precision, recall, ...) and min for metrics that should be minimized
(Brier, logloss, FPR, FNR, ...). Direction is inferred from the metric name
via keyword heuristics -- check the printed report before trusting an
unusual metric name.

Baseline-score subgroup dirs (subgroups_baseline__*) are clinical-score
comparators (TRISS/ISS/NISS), not your models, and are ignored. Train/
calibration subgroup dirs are ignored too (only test + external holdout, to
match "primary reported performance", per your instructions).

A model_id is only eligible for deletion if it has a complete base
"overall__test.json" -- incomplete/failed combos are left untouched.

USAGE
-----
    # 1. ALWAYS dry-run first -- prints what would be kept/deleted and how
    #    much space would be freed. Deletes nothing.
    python3 scripts/prune_non_best_models.py --outputs-root outputs

    # 2. Review the report (add --verbose-keep-reasons to see WHY each
    #    survivor is kept -- overall win, or which subgroup), then delete:
    python3 scripts/prune_non_best_models.py --outputs-root outputs --apply

Restrict targets or add metrics to ignore:
    python3 scripts/prune_non_best_models.py --outputs-root outputs \
        --targets mortality iss_band --exclude-metrics some_weird_metric
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

DEFAULT_TARGETS = [
    "mortality", "iss_band", "niss_band", "iss_band_binary", "niss_band_binary",
]

CALIBRATIONS = ["", "platt", "isotonic"]   # "" = base (uncalibrated) file
PARTITIONS = ["test", "holdout"]

# Subgroup directory name (under metrics/<model_id>/) -> partition label.
# Only test + external holdout are considered, matching the overall files.
SUBGROUP_DIRS = {"subgroups": "test", "subgroups_holdout": "holdout"}

# Which demographic axes may justify keeping a model on subgroup performance
# alone. Every extra axis multiplies the keep-set: each (axis, level, metric)
# tuple is its own "best model" slot, so axes with many levels (race, intent,
# mechanism, insurance) preserve far more artifacts than they are worth.
# Default = gender only; overall (whole-cohort) winners are ALWAYS kept
# regardless of this setting.
DEFAULT_SUBGROUP_AXES = ["gender"]

# Phase-of-care evaluation slices. These are NOT subgroup axes -- the trainer
# writes them as their own whole-cohort files:
#     overall__<partition>__cohort_<name>.json     (phase-completeness cohorts)
#     overall__<partition>__phase_complete.json    (all phase blocks present)
# They are central to the phase-cutoff framework (onsite = the TRISS inputs,
# then +ED arrival, then baseline), so a model that is best within a phase
# cohort is worth keeping even if it never wins overall.
DEFAULT_PHASE_COHORTS = [
    "onsite_complete", "ed_complete", "baseline_complete", "phase_complete",
]

# Keys that are counts/metadata, not performance metrics -- never ranked.
# The core set below mirrors trauma_ml.evaluation._CI_SKIP_KEYS /
# metric_cols exactly (the pipeline's own "not a rate/score" list), so a
# model can never be "kept" just because it happened to see more rows or
# have a different decision threshold. Additional row-count fields the
# trainer injects into overall__holdout*.json (n_holdout_rows,
# n_holdout_year_rows, n_cohort, n_total) are added on top, since a bigger
# evaluation cohort is not a performance win either.
NOT_A_METRIC = {
    "n", "n_valid", "support", "threshold",
    "TP", "TN", "FP", "FN", "prevalence",
    "n_holdout_rows", "n_holdout_year_rows", "n_cohort", "n_total",
    "cohort_fraction",
    "n_eval",
    "classes", "_ci_method", "subgroup_axis", "subgroup_level",
}

# Metric-name substrings that mean LOWER is better. Anything not matching
# is assumed to be maximized (AUROC, accuracy, F1, recall, precision,
# balanced_accuracy, specificity, AUPRC, ...).
MINIMIZE_KEYWORDS = (
    "brier", "logloss", "log_loss", "loss", "fpr", "fnr", "ece",
    "error", "mae", "rmse", "mse", "calibration_error",
)

MIN_SUBGROUP_ROWS_WARN = 1   # just a sanity guard, not a filter

# Only this file is deleted from a losing model_id's folder -- config.json
# (and anything else) is always kept, so the exact recipe survives.
ARTIFACT_FILENAME = "artifact.pkl"


def is_minimize(metric_name: str) -> bool:
    low = metric_name.lower()
    return any(kw in low for kw in MINIMIZE_KEYWORDS)


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PiB"


def _update_best(best: dict, tag: tuple, val: float, model_id: str) -> None:
    minimize = is_minimize(tag[-1])   # metric name is always the last element
    cur = best.get(tag)
    better = (
        cur is None
        or (minimize and val < cur[0])
        or (not minimize and val > cur[0])
    )
    if better:
        best[tag] = (val, model_id)


def process_target(target: str, outputs_root: Path, exclude_metrics: set[str],
                    apply: bool, allowed_axes: set[str] | None = None,
                    phase_cohorts: list[str] | None = None) -> dict:
    metrics_dir = outputs_root / target / "metrics"
    models_dir = outputs_root / target / "models"
    if not metrics_dir.is_dir():
        return {"target": target, "skipped": "no metrics dir"}

    model_ids = sorted(p.name for p in metrics_dir.iterdir() if p.is_dir())

    # best[tag] = (best_value, model_id)
    #   overall tag  = ("overall", partition, calibration, metric)
    #   subgroup tag = ("subgroup", partition, axis, level, metric)
    best: dict[tuple, tuple] = {}
    complete_model_ids: set[str] = set()
    n_subgroup_rows_seen = 0

    for model_id in model_ids:
        mdir = metrics_dir / model_id
        base_test = mdir / "overall__test.json"
        if not base_test.is_file():
            continue   # incomplete/failed combo -- out of scope, don't touch
        complete_model_ids.add(model_id)

        # ---- overall metrics (test/holdout x base/platt/isotonic) -------
        for part in PARTITIONS:
            for cal in CALIBRATIONS:
                suffix = f"__{cal}" if cal else ""
                fpath = mdir / f"overall__{part}{suffix}.json"
                if not fpath.is_file():
                    continue
                try:
                    data = json.loads(fpath.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                for key, val in data.items():
                    if key in NOT_A_METRIC or key in exclude_metrics:
                        continue
                    if key.endswith("__ci_low") or key.endswith("__ci_high"):
                        continue
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        continue
                    if not math.isfinite(val):
                        continue   # NaN/inf (e.g. empty-run diagnostics) must
                                   # never win OR poison a ranking tag
                    tag = ("overall", part, cal or "base", key)
                    _update_best(best, tag, val, model_id)

        # ---- phase-of-care cohort slices --------------------------------
        for part in PARTITIONS:
            for coh in (phase_cohorts or []):
                fname = (f"overall__{part}__phase_complete.json"
                         if coh == "phase_complete"
                         else f"overall__{part}__cohort_{coh}.json")
                fpath = mdir / fname
                if not fpath.is_file():
                    continue
                try:
                    data = json.loads(fpath.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(data, dict):
                    continue
                for key, val in data.items():
                    if key in NOT_A_METRIC or key in exclude_metrics:
                        continue
                    if key.endswith("__ci_low") or key.endswith("__ci_high"):
                        continue
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        continue
                    if not math.isfinite(val):
                        continue
                    tag = ("phase", part, coh, key)
                    _update_best(best, tag, val, model_id)

        # ---- subgroup metrics (test + external holdout only) ------------
        for dirname, part in SUBGROUP_DIRS.items():
            sdir = mdir / dirname
            if not sdir.is_dir():
                continue
            for csv_file in sorted(sdir.glob("subgroups_by_*.csv")):
                axis = csv_file.stem.replace("subgroups_by_", "")
                # Axis policy: only the allowed axes can win a keep-slot.
                # Others are still evaluated by the pipeline and their CSVs
                # are preserved -- they just don't protect an artifact.
                if allowed_axes is not None and axis not in allowed_axes:
                    continue
                try:
                    with open(csv_file, newline="") as f:
                        for row in csv.DictReader(f):
                            n_subgroup_rows_seen += 1
                            level = row.get("subgroup_level", "?")
                            for key, raw in row.items():
                                if key in NOT_A_METRIC or key in exclude_metrics:
                                    continue
                                if key.endswith("__ci_low") or key.endswith("__ci_high"):
                                    continue
                                try:
                                    val = float(raw)
                                except (TypeError, ValueError):
                                    continue
                                if not math.isfinite(val):
                                    continue
                                tag = ("subgroup", part, axis, level, key)
                                _update_best(best, tag, val, model_id)
                except (OSError, csv.Error):
                    continue

    keep_ids = {model_id for _, model_id in best.values()}
    delete_ids = sorted(complete_model_ids - keep_ids)

    # Why each kept model is kept (for the printed report)
    reasons: dict[str, list[str]] = {mid: [] for mid in keep_ids}
    for tag, (val, mid) in best.items():
        if tag[0] == "phase":
            _, part, coh, metric = tag
            reasons.setdefault(mid, []).append(
                f"{metric}[phase/{part}/{coh}]={val:.4g}")
        elif tag[0] == "overall":
            _, part, cal, metric = tag
            reasons.setdefault(mid, []).append(
                f"{metric}[overall/{part}/{cal}]={val:.4g}")
        else:
            _, part, axis, level, metric = tag
            reasons.setdefault(mid, []).append(
                f"{metric}[subgroup/{part}/{axis}={level}]={val:.4g}")

    freed = 0
    deletions = []
    for model_id in delete_ids:
        mpath = models_dir / model_id
        artifact_path = mpath / ARTIFACT_FILENAME
        if not artifact_path.is_file():
            continue   # nothing to delete (already gone, or non-standard layout)
        try:
            size = artifact_path.stat().st_size
        except OSError:
            continue
        freed += size
        deletions.append((model_id, size))
        if apply:
            artifact_path.unlink()   # delete ONLY the artifact, keep config.json

    return {
        "target": target,
        "n_complete": len(complete_model_ids),
        "n_kept": len(keep_ids & complete_model_ids),
        "n_deleted": len(deletions),
        "freed_bytes": freed,
        "deletions": deletions,
        "keep_reasons": reasons,
        "n_subgroup_rows_seen": n_subgroup_rows_seen,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default="outputs",
                     help="Path to the outputs/ directory (default: outputs)")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                     help=f"Targets to process (default: {DEFAULT_TARGETS})")
    ap.add_argument("--exclude-metrics", nargs="*", default=[],
                     help="Metric names to ignore when ranking")
    ap.add_argument("--subgroup-axes", nargs="+", default=DEFAULT_SUBGROUP_AXES,
                    metavar="AXIS",
                    help="Demographic axes whose subgroup winners are kept "
                         f"(default: {' '.join(DEFAULT_SUBGROUP_AXES)}). "
                         "Use 'all' for every axis (the old, much less strict "
                         "behaviour), or 'none' to keep overall winners only "
                         "(strictest). Overall winners are always kept.")
    ap.add_argument("--phase-cohorts", nargs="+", default=DEFAULT_PHASE_COHORTS,
                    metavar="COHORT",
                    help="Phase-of-care cohort slices whose winners are kept "
                         f"(default: {' '.join(DEFAULT_PHASE_COHORTS)}). "
                         "Use 'none' to ignore phase cohorts entirely.")
    ap.add_argument("--apply", action="store_true",
                     help="Actually delete. Without this flag, DRY RUN only.")
    ap.add_argument("--verbose-keep-reasons", action="store_true",
                     help="Print why each kept model is kept (overall win "
                          "and/or which subgroup(s) it won).")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    exclude_metrics = set(args.exclude_metrics)

    ph = [c.lower() for c in args.phase_cohorts]
    phase_cohorts = [] if "none" in ph else list(args.phase_cohorts)
    phase_desc = ", ".join(phase_cohorts) if phase_cohorts else "NONE"

    axes_arg = [a.lower() for a in args.subgroup_axes]
    if "all" in axes_arg:
        allowed_axes = None                      # no filtering
        axes_desc = "ALL axes (least strict)"
    elif "none" in axes_arg:
        allowed_axes = set()                     # nothing matches
        axes_desc = "NONE - overall winners only (strictest)"
    else:
        allowed_axes = set(args.subgroup_axes)
        axes_desc = ", ".join(sorted(allowed_axes))

    mode = "APPLYING (files WILL be deleted)" if args.apply else "DRY RUN (nothing deleted)"
    print(f"{'='*72}\nMODE: {mode}")
    print(f"Subgroup axes that can protect a model: {axes_desc}")
    print(f"Phase-of-care cohorts that can protect a model: {phase_desc}")
    print("Overall (whole-cohort) winners are always kept.")
    print(f"{'='*72}\n")

    grand_freed = 0
    grand_deleted = 0
    grand_kept = 0
    for target in args.targets:
        result = process_target(target, outputs_root, exclude_metrics,
                                args.apply, allowed_axes, phase_cohorts)
        if result.get("skipped"):
            print(f"[{target}] skipped: {result['skipped']}\n")
            continue
        print(f"[{target}] complete models: {result['n_complete']}  "
              f"| kept (best overall or in a subgroup): {result['n_kept']}  "
              f"| to delete: {result['n_deleted']}  "
              f"| space {'freed' if args.apply else 'to free'}: "
              f"{human(result['freed_bytes'])}  "
              f"(subgroup rows scanned: {result['n_subgroup_rows_seen']})")
        if args.verbose_keep_reasons:
            for mid, why in sorted(result["keep_reasons"].items()):
                print(f"    KEEP {mid}: best at {len(why)} metric(s) "
                      f"-> {', '.join(sorted(why)[:6])}"
                      f"{' ...' if len(why) > 6 else ''}")
        if result["deletions"]:
            biggest = sorted(result["deletions"], key=lambda t: -t[1])[:5]
            print("    largest deletions:",
                  ", ".join(f"{mid} ({human(sz)})" for mid, sz in biggest))
        print()
        grand_freed += result["freed_bytes"]
        grand_deleted += result["n_deleted"]
        grand_kept += result["n_kept"]

    print(f"{'='*72}")
    print(f"TOTAL: kept {grand_kept} model(s), "
          f"{'deleted' if args.apply else 'would delete'} {grand_deleted} model(s), "
          f"{'freed' if args.apply else 'would free'} {human(grand_freed)}")
    if not args.apply:
        print("\nThis was a DRY RUN. Re-run with --apply to actually delete.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
