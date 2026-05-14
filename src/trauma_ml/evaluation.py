"""Metrics — compute overall performance and break it down by subgroups.

Returns metrics as plain dicts (serialisable to JSON / CSV).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core metric computations
# ---------------------------------------------------------------------------
def _get_scores(model, X) -> np.ndarray | None:
    """Prefer predict_proba[:,1]; fall back to decision_function min-max normalised."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            s = model.decision_function(X)
            return (s - s.min()) / (s.max() - s.min() + 1e-12)
        except Exception:
            pass
    return None


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_proba: np.ndarray | None) -> dict[str, Any]:
    """Binary-classification metrics + calibration + confusion counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true == 1)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "FPR":       float(fp / (fp + tn)) if (fp + tn) else None,
        "FNR":       float(fn / (fn + tp)) if (fn + tp) else None,
    }
    if y_proba is not None and len(set(y_true)) > 1:
        metrics["AUROC"] = float(roc_auc_score(y_true, y_proba))
        metrics["AUPRC"] = float(average_precision_score(y_true, y_proba))
        metrics["Brier"] = float(brier_score_loss(y_true, y_proba))
        try:
            metrics["logloss"] = float(log_loss(y_true, y_proba))
        except Exception:
            metrics["logloss"] = None
    else:
        metrics["AUROC"] = metrics["AUPRC"] = metrics["Brier"] = metrics["logloss"] = None
    return metrics


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: np.ndarray | None, classes: list) -> dict[str, Any]:
    metrics = {
        "n": int(len(y_true)),
        "classes": list(map(str, classes)),
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        # Round 23: balanced_accuracy is the right top-level multiclass
        # comparison metric (avg of per-class recall, robust to class
        # imbalance — exactly what we have with rare 'Profound' bands).
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == len(classes):
        try:
            metrics["AUROC_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", labels=classes)
            )
        except Exception:
            metrics["AUROC_ovr"] = None
        try:
            metrics["logloss"] = float(log_loss(y_true, y_proba, labels=classes))
        except Exception:
            metrics["logloss"] = None
    else:
        metrics["AUROC_ovr"] = metrics["logloss"] = None
    # Per-class precision / recall / F1
    for cls_idx, cls in enumerate(classes):
        y_bin_true = (y_true == cls_idx).astype(int)
        y_bin_pred = (y_pred == cls_idx).astype(int)
        metrics[f"precision__{cls}"] = float(
            precision_score(y_bin_true, y_bin_pred, zero_division=0)
        )
        metrics[f"recall__{cls}"] = float(
            recall_score(y_bin_true, y_bin_pred, zero_division=0)
        )
        metrics[f"f1__{cls}"] = float(
            f1_score(y_bin_true, y_bin_pred, zero_division=0)
        )
    return metrics


def evaluate(model, X: pd.DataFrame, y: np.ndarray,
             task: str, classes: list | None = None) -> dict[str, Any]:
    """Compute performance metrics for one dataset partition."""
    y_pred = model.predict(X)
    y_proba = _get_scores(model, X)
    if task == "binary":
        return binary_metrics(y, y_pred, y_proba)
    if task == "multiclass":
        if classes is None:
            classes = sorted(np.unique(y).tolist())
        # If proba is 1D (binary facade returning pos-class prob), skip AUROC
        if y_proba is not None and y_proba.ndim == 1:
            full = None
            if hasattr(model, "predict_proba"):
                try:
                    full = model.predict_proba(X)
                except Exception:
                    full = None
            y_proba = full
        return multiclass_metrics(y, y_pred, y_proba, classes)
    raise ValueError(f"Unknown task {task!r}")


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------
def subgroup_metrics_for_score(
    score: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    subgroup_series: pd.Series,
    axis_name: str,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Subgroup metrics for a PRECOMPUTED score / prediction (e.g. a baseline).

    Parallel to ``subgroup_metrics``, but instead of querying a fitted
    model with X, it takes already-computed positive-class score
    (``score``) and binary prediction (``y_pred``) arrays.  This lets
    baseline scores like TRISS / ISS / NISS produce per-subgroup AUROC,
    AUPRC, recall, etc. just like ML models do — so they can compete in
    the same Pareto front.

    All three input arrays must be 1-D, of the same length, and aligned
    with ``subgroup_series.index``.
    """
    assert len(score) == len(y_pred) == len(y_true) == len(subgroup_series), (
        f"length mismatch: score={len(score)}, y_pred={len(y_pred)}, "
        f"y_true={len(y_true)}, subgroup={len(subgroup_series)}"
    )
    rows: list[dict[str, Any]] = []
    for level, mask in _group_masks(subgroup_series):
        n = int(mask.sum())
        if n < min_group_size:
            continue
        m_arr = mask.to_numpy()
        sub_score = score[m_arr]
        sub_pred  = y_pred[m_arr]
        sub_y     = y_true[m_arr]
        # Skip if all-one-class (AUROC undefined)
        if len(set(sub_y)) < 2:
            level_metrics = binary_metrics(sub_y, sub_pred, None)
        else:
            level_metrics = binary_metrics(sub_y, sub_pred, sub_score)
        level_metrics = {
            "subgroup_axis":  axis_name,
            "subgroup_level": str(level),
            **level_metrics,
        }
        rows.append(level_metrics)
    return pd.DataFrame(rows)


def subgroup_metrics(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    subgroup_series: pd.Series,
    axis_name: str,
    task: str,
    classes: list | None = None,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """Compute metrics for each level of ``subgroup_series`` (ignoring groups
    smaller than ``min_group_size``).
    """
    rows: list[dict[str, Any]] = []
    for level, mask in _group_masks(subgroup_series):
        n = int(mask.sum())
        if n < min_group_size:
            continue
        level_metrics = evaluate(
            model,
            X.loc[mask],
            y[mask.to_numpy()],
            task=task,
            classes=classes,
        )
        level_metrics = {"subgroup_axis": axis_name, "subgroup_level": str(level),
                         **level_metrics}
        rows.append(level_metrics)
    return pd.DataFrame(rows)


def _group_masks(series: pd.Series):
    """Yield (level, boolean_mask) pairs from a categorical-ish series."""
    s = series.copy()
    s = s.where(s.notna(), other="__missing__")
    for level in sorted(s.unique(), key=str):
        yield level, (s == level)


def save_metrics(
    metrics: dict[str, Any],
    output_dir: Path,
    filename: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    else:
        # flatten one-level for CSV
        pd.DataFrame([metrics]).to_csv(path, index=False)
    log.info("Wrote metrics to %s", path)
    return path


def save_subgroup_metrics(
    df: pd.DataFrame,
    output_dir: Path,
    axis_name: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"subgroups_by_{axis_name}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Combined metrics CSV + Pareto front  (aggregated across all models)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Partition display names  (internal key → human-readable label in CSV)
# ---------------------------------------------------------------------------
_PARTITION_LABELS: dict[str, str] = {
    "train":        "train",
    "calibration":  "calibration_set",
    "test":         "random_test",        # random 15% held aside from non-holdout pool
    "holdout_year": "holdout_2024",       # temporal year holdout (most recent NTDB year)
    "holdout":      "holdout_external",   # legacy external file holdout
}

# Subgroup values that represent missingness — excluded from subgroup metrics
_MISSING_LEVEL_TOKENS = frozenset({"missing", "nan", "none", "unknown", "", "na", "other"})

# Subgroup directories produced by the Trainer, mapped to partition key
_SUBGROUP_DIRS: list[tuple[str, str]] = [
    ("subgroups",              "test"),
    ("subgroups_train",        "train"),         # round 11
    ("subgroups_calibration",  "calibration"),   # round 11
    ("subgroups_holdout_year", "holdout_year"),
    ("subgroups_holdout",      "holdout"),
]

# Metric names expected inside every subgroup CSV row
# Round 32: balanced_accuracy added so multiclass targets (ISS_band /
# NISS_band) get worst_balanced_accuracy__* per-subgroup columns built;
# without this the Pareto for multiclass targets had nothing to work with.
_SUBGROUP_METRICS = (
    "AUROC", "AUPRC", "Brier", "f1", "recall", "precision",
    "balanced_accuracy",
    "n",
)

# Default Pareto settings
_DEFAULT_PARETO_METRIC     = "AUPRC"
_DEFAULT_PARETO_PARTITIONS = ("test", "holdout_year")


def _partition_label(partition: str) -> str:
    return _PARTITION_LABELS.get(partition, partition)


def _is_missing_level(level: str) -> bool:
    return level.strip("_ ").lower() in _MISSING_LEVEL_TOKENS


# ---------------------------------------------------------------------------
# Column name builders — single source of truth for CSV column naming
# ---------------------------------------------------------------------------

def _overall_col(metric: str, partition: str) -> str:
    """e.g. 'AUROC__random_test'  or  'AUPRC__holdout_2024'"""
    return f"{metric}__{_partition_label(partition)}"


def _baseline_col(score: str, metric: str, partition: str) -> str:
    """e.g. 'baseline_ISS__AUPRC__random_test'"""
    return f"baseline_{score}__{metric}__{_partition_label(partition)}"


def _subgroup_col(metric: str, partition: str, axis: str, level: str) -> str:
    """e.g. 'AUPRC__holdout_2024__gender__Female'"""
    return f"{metric}__{_partition_label(partition)}__{axis}__{level}"


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------

def _worst_case_per_group(
    df: pd.DataFrame,
    metric: str = _DEFAULT_PARETO_METRIC,
    partitions: tuple[str, ...] = _DEFAULT_PARETO_PARTITIONS,
) -> list[str]:
    """Add worst-case-per-group columns and return their names.

    For each (partition, axis) combination, takes the minimum of
    ``<metric>__<partition_label>__<axis>__*`` across all demographic levels
    and writes it into ``worst_<metric>__<partition_label>__<axis>``.
    """
    added: list[str] = []
    for partition in partitions:
        label = _partition_label(partition)
        prefix = f"{metric}__{label}__"
        subgroup_cols = [c for c in df.columns if c.startswith(prefix)]
        if not subgroup_cols:
            continue
        axes: dict[str, list[str]] = {}
        for col in subgroup_cols:
            remainder = col[len(prefix):]
            last_sep = remainder.rfind("__")
            if last_sep == -1:
                continue
            axis = remainder[:last_sep]
            axes.setdefault(axis, []).append(col)
        for axis, cols in axes.items():
            worst_col = f"worst_{metric}__{label}__{axis}"
            df[worst_col] = df[cols].min(axis=1)
            added.append(worst_col)
    return sorted(added)


def compute_pareto_front(
    df: pd.DataFrame,
    objectives: list[str] | None = None,
    pareto_metric: str = _DEFAULT_PARETO_METRIC,
    pareto_partitions: tuple[str, ...] = _DEFAULT_PARETO_PARTITIONS,
) -> pd.Series:
    """Return boolean Series: True when row is on the Pareto front.

    Maximises worst-case ``pareto_metric`` across all demographic subgroups
    in ``pareto_partitions``.  Pass ``objectives`` explicitly to override.
    NaN treated as −∞ (worst).
    """
    if objectives is not None:
        objs = [o for o in objectives if o in df.columns]
    else:
        labels = [_partition_label(p) for p in pareto_partitions]
        worst_cols = [
            c for c in df.columns
            if c.startswith(f"worst_{pareto_metric}__")
            and any(f"__{lbl}__" in c for lbl in labels)
        ]
        objs = worst_cols or [
            _overall_col(pareto_metric, p) for p in pareto_partitions
            if _overall_col(pareto_metric, p) in df.columns
        ]

    objs = [o for o in objs if o in df.columns]
    if not objs:
        log.warning("No Pareto objectives found; marking all rows False.")
        return pd.Series(False, index=df.index)

    log.info("Pareto front: metric=%s, %d objectives", pareto_metric, len(objs))
    vals = df[objs].fillna(-np.inf).to_numpy()
    n = len(df)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j or not on_front[j]:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                on_front[i] = False
                break
    return pd.Series(on_front, index=df.index)


# ---------------------------------------------------------------------------
# Config reader
# ---------------------------------------------------------------------------

def _read_config_json(model_dir: Path) -> dict:
    import json
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        blob = json.load(f)
    row: dict = {}
    for key in ("family", "task"):
        if key in blob:
            row[f"cfg_{key}"] = blob[key]
    ts = blob.get("target_spec", {})
    row["cfg_target_name"] = ts.get("name", "")
    row["cfg_target_kind"] = ts.get("kind", "")
    for k, v in blob.get("config", {}).items():
        row[f"cfg_{k}"] = str(v) if isinstance(v, (dict, list)) else v
    for k, v in blob.get("extras", {}).items():
        row[f"extra_{k}"] = str(v) if isinstance(v, (dict, list)) else v
    row["cfg_n_predictors"] = len(blob.get("predictor_cols", []))
    row["cfg_actual_model_type"] = blob.get(
        "actual_model_type", row.get("cfg_family", "")
    )
    return row


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

def aggregate_unified_competitors(
    metrics_root: Path,
    output_path: Path | None = None,
    pareto_metric: str = "AUROC",
    pareto_partitions: tuple[str, ...] = ("test", "holdout"),
) -> pd.DataFrame:
    """Build a UNIFIED Pareto-eligible CSV of ML models AND baselines.

    What this addresses
    -------------------
    The user asked for the baseline scores (TRISS, ISS, NISS) to
    "participate in the Pareto selection" — i.e. show up as rows next to
    the ML models, ranked on the same metrics, on the same Pareto front.
    The wide ``all_metrics.csv`` has baselines as columns inside each
    model's row, which is no good for ranking baselines AGAINST models.

    Output structure
    ----------------
    One row per ``(competitor, cohort)`` pair, where ``competitor`` is
    either the ML model id (e.g. ``model_0003``) or a baseline pseudo-id
    (``baseline:TRISS``, ``baseline:ISS``, ``baseline:NISS``).

    For each row, columns are:
        AUROC__<partition>, AUPRC__<partition>, Brier__<partition>,
        recall__<partition>, precision__<partition>, accuracy__<partition>,
        specificity__<partition>, f1__<partition>, n_eval__<partition>
    for partitions in {random_test, holdout_external, holdout_2024}.

    A single ``pareto_front`` column at the end marks rows on the
    Pareto front maximising ``pareto_metric`` across the requested
    ``pareto_partitions``.  The Pareto computation treats every row
    (model and baseline alike) as a single competitor, so a baseline
    can dominate an ML model and vice versa — exactly what the user
    asked for.

    NaN handling: ISS / NISS now produce real AUROC / AUPRC because
    we pass the (continuous) score / 75.0 to ``binary_metrics``.  See
    the round-7 baselines.py change.
    """
    import json
    metrics_root = Path(metrics_root)

    # The metrics we surface (for both models and baselines).  Order is
    # kept stable for downstream column expectations.  Round 23 added
    # balanced_accuracy + f1_macro/f1_weighted for multiclass scenarios.
    # Binary models simply leave the multiclass-specific cells blank.
    metric_keys = ["AUROC", "AUPRC", "Brier",
                   "accuracy", "balanced_accuracy",
                   "precision", "recall", "specificity", "f1",
                   "f1_macro", "f1_weighted",
                   "precision_macro", "recall_macro",
                   "AUROC_ovr", "logloss"]
    rows: list[dict] = []

    # ── Pass 1: ML models ─────────────────────────────────────────────────
    for model_dir in sorted(metrics_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name

        # All overall files for this model: overall__<partition>[__cohort_<n>][__<cal>].json
        for jf in sorted(model_dir.glob("overall__*.json")):
            stem_parts = jf.stem.split("__")
            # Skip calibrated variants for the unified table — keep base
            # uncalibrated metrics so each (model, cohort) appears once
            if any(p in ("platt", "isotonic", "sigmoid") for p in stem_parts):
                continue
            if len(stem_parts) == 2:                          # overall__<partition>.json
                _, partition = stem_parts
                cohort = "all"
            elif len(stem_parts) == 3 and stem_parts[2].startswith("cohort_"):
                _, partition, cohort_tag = stem_parts
                cohort = cohort_tag.removeprefix("cohort_")
            else:
                continue

            try:
                with open(jf) as f:
                    blob = json.load(f)
            except Exception as exc:
                log.warning("Cannot read %s: %s", jf, exc)
                continue

            row_key = (f"model:{model_id}", cohort)
            row = {(rk[0], rk[1]): rk for rk in [row_key]}  # dummy
            # Find/create row for this (competitor, cohort) pair
            entry = next(
                (r for r in rows
                 if r["competitor"] == row_key[0]
                 and r["cohort"] == row_key[1]),
                None,
            )
            if entry is None:
                entry = {"competitor": row_key[0],
                         "kind": "model",
                         "cohort": cohort}
                rows.append(entry)
            partition_label = _partition_label(partition)
            for mk in metric_keys:
                entry[f"{mk}__{partition_label}"] = blob.get(mk, float("nan"))
            entry[f"n_eval__{partition_label}"] = blob.get("n_cohort",
                                                            blob.get("n_eval"))

    # ── Pass 2: baselines ─────────────────────────────────────────────────
    # For each baseline JSON, we add one row per (baseline_score, cohort)
    # combination, aggregating across model_dirs (since the baseline output
    # is identical across models for the same partition+cohort — they're
    # not model-dependent).  Use the first model_dir's baseline files.
    seen_baseline: dict[tuple[str, str], dict] = {}

    for model_dir in sorted(metrics_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("baseline__*.json")):
            stem_parts = jf.stem.split("__")
            if len(stem_parts) == 3:                          # baseline__<score>__<part>.json
                _, score, partition = stem_parts
                cohort = "all"
            elif len(stem_parts) == 4 and stem_parts[3].startswith("cohort_"):
                _, score, partition = stem_parts[:3]
                cohort = stem_parts[3].removeprefix("cohort_")
            else:
                continue

            key = (score, cohort)
            if key not in seen_baseline:
                seen_baseline[key] = {
                    "competitor": f"baseline:{score}",
                    "kind":       "baseline",
                    "cohort":     cohort,
                }

            try:
                with open(jf) as f:
                    blob = json.load(f)
            except Exception as exc:
                log.warning("Cannot read %s: %s", jf, exc)
                continue

            partition_label = _partition_label(partition)
            entry = seen_baseline[key]
            for mk in metric_keys:
                # Only set if not already populated (across model_dirs the
                # baseline values are identical, so first wins)
                col = f"{mk}__{partition_label}"
                if col not in entry or pd.isna(entry.get(col)):
                    entry[col] = blob.get(mk, float("nan"))
            n_col = f"n_eval__{partition_label}"
            if n_col not in entry:
                entry[n_col] = blob.get("n_valid")

    rows.extend(seen_baseline.values())

    if not rows:
        log.warning("No competitor data found under %s", metrics_root)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Pareto front across ML models AND baselines ──────────────────────
    # Compute on rows where the chosen metric is populated for the chosen
    # partitions.  Compute SEPARATELY per cohort so we don't compare a
    # model on cohort_all against a baseline on cohort_baseline_complete.
    df["pareto_front"] = False
    partition_labels = [_partition_label(p) for p in pareto_partitions]
    pareto_objs = [f"{pareto_metric}__{lbl}" for lbl in partition_labels
                   if f"{pareto_metric}__{lbl}" in df.columns]
    if pareto_objs:
        for cohort_value in df["cohort"].unique():
            mask = (df["cohort"] == cohort_value)
            sub = df.loc[mask, pareto_objs].fillna(-np.inf).to_numpy()
            n = len(sub)
            on_front = np.ones(n, dtype=bool)
            for i in range(n):
                if not on_front[i]:
                    continue
                for j in range(n):
                    if i == j or not on_front[j]:
                        continue
                    if np.all(sub[j] >= sub[i]) and np.any(sub[j] > sub[i]):
                        on_front[i] = False
                        break
            df.loc[mask, "pareto_front"] = on_front
        log.info(
            "Unified Pareto: %d competitor rows, %d on Pareto front "
            "(per-cohort) across %d objectives: %s",
            len(df), int(df["pareto_front"].sum()), len(pareto_objs), pareto_objs,
        )
    else:
        log.warning(
            "Unified Pareto: no objective columns available "
            "(expected %s in %s)",
            pareto_metric, partition_labels,
        )

    # Reorder columns: ids first, then metrics grouped by partition, then pareto
    id_cols = ["competitor", "kind", "cohort"]
    metric_cols = sorted([c for c in df.columns
                          if c not in id_cols and c != "pareto_front"])
    df = df[id_cols + metric_cols + ["pareto_front"]]

    # Sort: pareto winners first within each cohort, then by AUROC desc
    cohort_order = pd.CategoricalDtype(
        ["all", "onsite_complete", "ed_complete", "baseline_complete"],
        ordered=True,
    )
    df["cohort"] = df["cohort"].astype(cohort_order)
    sort_metric = next(
        (c for c in df.columns if c.startswith(f"{pareto_metric}__")),
        None,
    )
    if sort_metric:
        df = df.sort_values(
            ["cohort", "pareto_front", sort_metric],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    df["cohort"] = df["cohort"].astype(str)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        n_models = (df["kind"] == "model").sum()
        n_bls    = (df["kind"] == "baseline").sum()
        n_pareto = int(df["pareto_front"].sum())
        log.info(
            "Wrote %s: %d rows (%d model-rows, %d baseline-rows), "
            "%d on Pareto front",
            output_path.name, len(df), n_models, n_bls, n_pareto,
        )

    return df


def aggregate_baseline_metrics_long(
    metrics_root: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build a LONG-FORMAT CSV of baseline (ISS / NISS / TRISS) metrics.

    Each row corresponds to one (model_id, score, partition, cohort,
    calibration) combination — the baselines themselves don't have a
    calibration variant but the column is included for shape consistency
    with the model-metrics output.

    Why this exists
    ---------------
    The wide ``all_metrics.csv`` keeps one row per model and spreads
    baseline metrics across dozens of columns named e.g.
    ``baseline_ISS__AUPRC__holdout_external``.  That layout is fine for
    Pareto computations across models but is unreadable when the user
    actually wants to compare baseline scores side-by-side.  This long
    format addresses complaint #1: "I need the metrics of the baseline
    scores as rows, not columns."

    Output columns
    --------------
    model_id, score, partition, cohort, n_valid, threshold,
    AUROC, AUPRC, Brier, logloss, accuracy, precision, recall,
    specificity, f1, FPR, FNR, prevalence, TP, FP, TN, FN

    cohort='all' means the un-cohorted (full-partition) baseline JSON.
    For ISS / NISS, AUROC/AUPRC/Brier/logloss are NaN by design — those
    scores are threshold rules, not probability scores.  For TRISS,
    all metrics are populated.
    """
    import json
    metrics_root = Path(metrics_root)
    rows: list[dict] = []

    # The metric columns we surface in this CSV (in display order)
    metric_cols = [
        "n_valid", "threshold",
        "AUROC", "AUPRC", "Brier", "logloss",
        "accuracy", "precision", "recall", "specificity", "f1",
        "FPR", "FNR", "prevalence",
        "TP", "FP", "TN", "FN",
    ]

    for model_dir in sorted(metrics_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("baseline__*.json")):
            stem_parts = jf.stem.split("__")
            calibration = "raw"   # round 11: default for non-calibrated baseline files
            # Three filename shapes (round 11 added cal-variant cases):
            #   baseline__<score>__<partition>.json                       (3 parts)
            #   baseline__<score>__<partition>__cohort_<name>.json        (4 parts)
            if len(stem_parts) == 3:
                _, score, partition = stem_parts
                cohort = "all"
            elif len(stem_parts) == 4 and stem_parts[3].startswith("cohort_"):
                _, score, partition = stem_parts[:3]
                cohort = stem_parts[3].removeprefix("cohort_")
            elif len(stem_parts) == 4 and stem_parts[3] in ("platt", "isotonic", "threshold_tuned"):
                _, score, partition, calibration = stem_parts
                cohort = "all"
            elif (len(stem_parts) == 5 and stem_parts[3].startswith("cohort_")
                  and stem_parts[4] in ("platt", "isotonic", "threshold_tuned")):
                _, score, partition = stem_parts[:3]
                cohort = stem_parts[3].removeprefix("cohort_")
                calibration = stem_parts[4]
            else:
                log.debug("Skipping baseline file with unexpected name: %s", jf.name)
                continue

            try:
                with open(jf) as f:
                    blob = json.load(f)
            except Exception as exc:
                log.warning("Cannot read %s: %s", jf, exc)
                continue

            row: dict = {
                "model_id":    model_dir.name,
                "score":       score,
                "calibration": calibration,
                "partition":   _partition_label(partition),
                "cohort":      cohort,
            }
            for m in metric_cols:
                row[m] = blob.get(m, float("nan"))
            rows.append(row)

    if not rows:
        log.warning("No baseline__*.json files found under %s", metrics_root)
        return pd.DataFrame(columns=["model_id", "score", "partition", "cohort"]
                                     + metric_cols)

    df = pd.DataFrame(rows)
    # Ordering: by model_id, then score (ISS/NISS/TRISS), partition, cohort
    score_order = pd.CategoricalDtype(["ISS", "NISS", "TRISS"], ordered=True)
    df["score"] = df["score"].astype(score_order)
    cohort_order = pd.CategoricalDtype(
        ["all", "onsite_complete", "ed_complete", "baseline_complete"],
        ordered=True,
    )
    cal_order = pd.CategoricalDtype(
        ["raw", "platt", "isotonic", "threshold_tuned"], ordered=True,
    )
    if "calibration" in df.columns:
        df["calibration"] = df["calibration"].astype(cal_order)
    df["cohort"] = df["cohort"].astype(cohort_order)
    sort_keys = ["model_id", "score", "partition", "cohort"]
    if "calibration" in df.columns:
        sort_keys.insert(2, "calibration")
    df = df.sort_values(sort_keys).reset_index(drop=True)
    df["score"]  = df["score"].astype(str)
    df["cohort"] = df["cohort"].astype(str)
    if "calibration" in df.columns:
        df["calibration"] = df["calibration"].astype(str)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        log.info(
            "Wrote %s: %d rows across %d models, %d scores, %d partitions, %d cohorts",
            output_path.name, len(df),
            df["model_id"].nunique(), df["score"].nunique(),
            df["partition"].nunique(), df["cohort"].nunique(),
        )
    return df


def aggregate_all_metrics(
    metrics_root: Path,
    models_root: Path | None = None,
    output_path: Path | None = None,
    pareto_metric: str = _DEFAULT_PARETO_METRIC,
    pareto_partitions: tuple[str, ...] = _DEFAULT_PARETO_PARTITIONS,
    pareto_objectives: list[str] | None = None,
    cohort: str | None = None,
    column_template: list[str] | None = None,
) -> pd.DataFrame:
    """Build the all_metrics CSV from all model metric directories.

    Parameters
    ----------
    cohort : str | None
        If ``None`` (default), aggregate the JSONs computed on the full
        partition (current behaviour — produces ``all_metrics.csv``).
        If a cohort name (e.g. ``"onsite_complete"``), aggregate only the
        JSONs whose filename carries the matching ``__cohort_<n>``
        suffix (produces e.g. ``all_metrics__cohort_onsite_complete.csv``).
    column_template : list[str] | None
        If given, the returned DataFrame is reindexed to exactly these
        columns (missing ones filled with NaN, extra ones dropped).
        Used to ensure that cohort CSVs share the same column structure
        as the main ``all_metrics.csv`` — addresses the complaint that
        cohort files have inconsistent shapes.

    Parameters
    ----------
    cohort : str | None
        If ``None`` (default), aggregate the JSONs computed on the full
        partition (current behaviour — produces ``all_metrics.csv``).
        If a cohort name (e.g. ``"onsite_complete"``), aggregate only the
        JSONs whose filename carries the matching ``__cohort_<name>``
        suffix (produces e.g. ``all_metrics__cohort_onsite_complete.csv``).

    Column layout (left → right)
    ----------------------------
    1.  ``model_id``
    2.  ``cfg_*`` / ``extra_*``       — full pipeline config incl. actual_model_type
    3.  Overall model metrics          — ``<metric>__<partition_label>``
          Partition labels: random_test, holdout_2024, train, calibration_set
    4.  Baseline clinical scores       — ``baseline_<ISS|NISS|TRISS>__<metric>__<partition>``
          Same partition labels as above.
    5.  Per-subgroup model metrics     — ``<metric>__<partition>__<axis>__<level>``
          Missing / unknown demographic levels are excluded.
    6.  Worst-case Pareto inputs       — ``worst_<pareto_metric>__<partition>__<axis>``
    7.  ``pareto_front``               — boolean

    Partition label key
    -------------------
    * ``random_test``      — 15 % of non-holdout rows, randomly assigned
    * ``holdout_2024``     — temporally held-out NTDB year(s) (never in training)
    * ``train``            — training rows
    * ``calibration_set``  — calibration rows (used only for Platt/isotonic scaling)
    * ``holdout_external`` — legacy external holdout parquet
    """
    import json

    metrics_root = Path(metrics_root)
    if models_root is None:
        models_root = metrics_root.parent / "models"
    else:
        models_root = Path(models_root)

    cohort_suffix = f"__cohort_{cohort}" if cohort else ""
    rows: list[dict] = []

    for model_dir in sorted(metrics_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        row: dict = {"model_id": model_id}

        # ── 1. Config ────────────────────────────────────────────────────
        row.update(_read_config_json(models_root / model_id))
        calibration = str(row.get("cfg_calibration", "none")).lower().strip()
        cal_suffix = f"__{calibration}" if calibration not in ("none", "", "nan") else ""

        # ── 2. Overall model metrics ─────────────────────────────────────
        for partition in ("train", "calibration", "test", "holdout_year", "holdout"):
            candidates = []
            if cal_suffix:
                candidates.append(
                    model_dir / f"overall__{partition}{cohort_suffix}{cal_suffix}.json"
                )
            candidates.append(model_dir / f"overall__{partition}{cohort_suffix}.json")
            for jf in candidates:
                if not jf.exists():
                    continue
                with open(jf) as f:
                    m = json.load(f)
                for k, v in m.items():
                    row[_overall_col(k, partition)] = v
                break

        # ── 3. Baseline clinical score metrics ────────────────────────────
        # Baselines are NO LONGER columns inside the per-model row.  They
        # appear as their own rows at the end of this DataFrame (added in
        # the post-loop pass below) so they participate in the same
        # Pareto selection as the ML models.  This addresses the user's
        # repeated request: "I want the baseline scores as rows, not
        # columns. Avoid unnecessary columns."  See also:
        # the now-deprecated ``models_and_baselines.csv`` aggregator.

        # ── 4. Subgroup metrics — skip missing/unknown levels ─────────────
        # Subgroup CSVs are not cohort-stratified — only included for the
        # default (no-cohort) aggregation.
        if not cohort:
            for sub_dir_name, partition in _SUBGROUP_DIRS:
                sub_dir = model_dir / sub_dir_name
                if not sub_dir.exists():
                    continue
                for csv_file in sorted(sub_dir.glob("subgroups_by_*.csv")):
                    axis = csv_file.stem.replace("subgroups_by_", "")
                    try:
                        sg = pd.read_csv(csv_file)
                    except Exception as e:
                        log.warning("Cannot read %s: %s", csv_file, e)
                        continue
                    for _, sg_row in sg.iterrows():
                        level = str(sg_row.get("subgroup_level", ""))
                        if _is_missing_level(level):
                            continue      # exclude NaN / unknown / missing groups
                        for metric in _SUBGROUP_METRICS:
                            if metric in sg_row and not pd.isna(sg_row[metric]):
                                row[_subgroup_col(metric, partition, axis, level)] = sg_row[metric]

        rows.append(row)

    # ── Append baseline rows (one per score × calibration variant) ──────
    # Same column shape as model rows so they participate in the same
    # Pareto computation.  Each combination of baseline score (ISS, NISS,
    # TRISS) and calibration method (raw, platt, isotonic, threshold_tuned)
    # appears as its OWN row, e.g. ``baseline:TRISS_platt``,
    # ``baseline:TRISS_isotonic``.  This addresses round-11 item 2.
    if metrics_root.is_dir():
        _CAL = {"platt", "isotonic", "threshold_tuned"}
        # (score_name, calibration) -> partition_label -> list of metric dicts
        per_score: dict[tuple[str, str], dict[str, list[dict]]] = {}
        for model_dir in sorted(metrics_root.iterdir()):
            if not model_dir.is_dir():
                continue
            # Iterate over ALL baseline files (raw + calibrated + cohort variants)
            for bf in sorted(model_dir.glob("baseline__*.json")):
                stem_parts = bf.stem.split("__")
                file_calibration = "raw"
                file_cohort = "all"
                if len(stem_parts) == 3:
                    _, score_name, partition = stem_parts
                elif len(stem_parts) == 4 and stem_parts[3].startswith("cohort_"):
                    _, score_name, partition = stem_parts[:3]
                    file_cohort = stem_parts[3].removeprefix("cohort_")
                elif len(stem_parts) == 4 and stem_parts[3] in _CAL:
                    _, score_name, partition, file_calibration = stem_parts
                elif (len(stem_parts) == 5
                      and stem_parts[3].startswith("cohort_")
                      and stem_parts[4] in _CAL):
                    _, score_name, partition = stem_parts[:3]
                    file_cohort = stem_parts[3].removeprefix("cohort_")
                    file_calibration = stem_parts[4]
                else:
                    continue
                # When aggregating with cohort=None we want only cohort='all' files.
                # When aggregating with a specific cohort we want only that cohort.
                expected_cohort = cohort if cohort else "all"
                if file_cohort != expected_cohort:
                    continue
                try:
                    with open(bf) as f:
                        bm = json.load(f)
                except Exception as exc:
                    log.warning("Cannot read %s: %s", bf, exc)
                    continue
                key = (score_name, file_calibration)
                per_score.setdefault(key, {}).setdefault(
                    _partition_label(partition), []
                ).append(bm)

        for (score_name, calibration) in sorted(per_score):
            # model_id format:
            #   baseline:ISS              for raw
            #   baseline:ISS_platt        for calibrated variants
            mid = f"baseline:{score_name}" if calibration == "raw" \
                  else f"baseline:{score_name}_{calibration}"
            row: dict = {"model_id": mid}
            for part_label, blob_list in per_score[(score_name, calibration)].items():
                metric_vals: dict[str, list[float]] = {}
                for blob in blob_list:
                    for k, v in blob.items():
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            continue
                        metric_vals.setdefault(k, []).append(v)
                for k, vs in metric_vals.items():
                    if not vs:
                        continue
                    try:
                        mean_v = float(np.mean(vs))
                        row[_overall_col(k, part_label)] = mean_v
                    except (TypeError, ValueError):
                        row[_overall_col(k, part_label)] = vs[0]

            # ---- Baseline SUBGROUP metrics (round 10) ----
            # Subgroup CSVs are produced only for the RAW baseline currently
            # (calibration changes the threshold/probability but the score
            # ranking on which subgroup AUROC depends is preserved by Platt
            # and isotonic — they are monotonic transforms — so subgroup
            # AUROC values are identical across raw/platt/isotonic).
            # We attach the same subgroup numbers to all calibration variants
            # so they all have populated worst_* columns and can compete in
            # the Pareto front.
            for model_dir in sorted(metrics_root.iterdir()):
                if not model_dir.is_dir():
                    continue
                for sub_dir in model_dir.glob(
                    f"subgroups_baseline__*__{score_name}"
                ):
                    name_parts = sub_dir.name.split("__")
                    if len(name_parts) < 3:
                        continue
                    sub_partition = name_parts[1]
                    sub_partition_label = _partition_label(sub_partition)
                    for csv_file in sub_dir.glob("subgroups_by_*.csv"):
                        axis = csv_file.stem.replace("subgroups_by_", "")
                        try:
                            sg = pd.read_csv(csv_file)
                        except Exception:
                            continue
                        for _, sg_row in sg.iterrows():
                            level = str(sg_row.get("subgroup_level", ""))
                            if _is_missing_level(level):
                                continue
                            for metric in _SUBGROUP_METRICS:
                                if metric not in sg_row or pd.isna(sg_row[metric]):
                                    continue
                                col = _subgroup_col(
                                    metric, sub_partition_label, axis, level,
                                )
                                if col not in row or pd.isna(row.get(col)):
                                    row[col] = sg_row[metric]
                break
            rows.append(row)

    if not rows:
        log.warning("No model directories found under %s", metrics_root)
        return pd.DataFrame()

    df = pd.DataFrame(rows).reset_index(drop=True)

    # ── 5. Worst-case Pareto input columns ────────────────────────────────
    worst_cols: list[str] = []
    if pareto_objectives is None:
        worst_cols = _worst_case_per_group(
            df, metric=pareto_metric, partitions=pareto_partitions
        )

    # ── 6. Pareto front ───────────────────────────────────────────────────
    df["pareto_front"] = compute_pareto_front(
        df,
        objectives=pareto_objectives,
        pareto_metric=pareto_metric,
        pareto_partitions=pareto_partitions,
    )

    # ── 7. Column ordering ────────────────────────────────────────────────
    # Note: ``baseline_*`` columns are no longer produced — baselines now
    # appear as ROWS (model_id starts with 'baseline:').  See user request
    # on round 9.
    id_col     = ["model_id"]
    cfg_cols   = sorted(c for c in df.columns
                        if c.startswith("cfg_") or c.startswith("extra_"))
    overall_m  = sorted(
        c for c in df.columns
        if c not in id_col + ["pareto_front"] + worst_cols
        and not c.startswith(("cfg_", "extra_", "baseline_", "worst_"))
        and "__" in c
        and c.count("__") == 1
    )
    subgroup_m = sorted(
        c for c in df.columns
        if "__" in c
        and not c.startswith(("cfg_", "extra_", "worst_", "baseline_"))
        and c not in overall_m
        and c not in id_col + ["pareto_front"] + worst_cols
    )
    ordered = id_col + cfg_cols + overall_m + subgroup_m + worst_cols + ["pareto_front"]
    df = df[[c for c in ordered if c in df.columns]]

    if column_template is not None:
        df = df.reindex(columns=list(column_template))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        n_front = int(df["pareto_front"].sum())
        n_baselines = int(
            df["model_id"].astype(str).str.startswith("baseline:").sum()
        )
        log.info(
            "Wrote %s: %d rows (%d models + %d baselines), %d cols "
            "(%d overall, %d subgroup, %d worst), %d Pareto%s",
            output_path.name, len(df),
            len(df) - n_baselines, n_baselines, len(df.columns),
            len(overall_m), len(subgroup_m), len(worst_cols), n_front,
            f"  [cohort={cohort}]" if cohort else "",
        )

    return df


# ---------------------------------------------------------------------------
# Cohort-counts aggregator
# ---------------------------------------------------------------------------

def aggregate_cohort_counts(
    metrics_root: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Walk every model's ``cohort_counts.json`` (+ ``cohort_variables.json``)
    and emit a long-form CSV.

    Output columns
    --------------
    model_id, partition, cohort, n_cohort, n_total, fraction, required_vars

    * ``required_vars`` is the semicolon-joined list of variables that
      had to be non-NaN in the original parquet for a row to be counted
      in the cohort.  Comes from ``cohort_variables.json`` written by
      the trainer; left blank if that file is absent.
    * The ``cohort = "all"`` row records the partition's total row count.
    * Defensive: the trainer-side counts JSON should no longer contain
      a ``_resolved_vars`` key (the trainer pops it out into a separate
      file), but if an older run left one in, we skip it here.
    """
    import json
    metrics_root = Path(metrics_root)
    rows: list[dict] = []

    for model_dir in sorted(metrics_root.iterdir()):
        if not model_dir.is_dir():
            continue
        cc_path = model_dir / "cohort_counts.json"
        if not cc_path.exists():
            continue
        with open(cc_path) as f:
            blob = json.load(f)

        # Load companion variable lists if present
        vars_path = model_dir / "cohort_variables.json"
        vars_blob: dict[str, dict[str, list[str]]] = {}
        if vars_path.exists():
            try:
                with open(vars_path) as f:
                    vars_blob = json.load(f)
            except Exception as exc:
                log.warning("Cannot read %s: %s", vars_path, exc)

        for partition, partition_blob in blob.items():
            if not isinstance(partition_blob, dict):
                continue
            n_total = partition_blob.get("n_total")
            partition_vars = vars_blob.get(partition, {})
            if n_total is not None:
                rows.append({
                    "model_id":      model_dir.name,
                    "partition":     _partition_label(partition),
                    "cohort":        "all",
                    "n_cohort":      int(n_total),
                    "n_total":       int(n_total),
                    "fraction":      1.0,
                    "required_vars": "",
                })
            for cohort_name, cohort_blob in partition_blob.items():
                if cohort_name in ("n_total", "_resolved_vars"):
                    continue
                vars_for_cohort = partition_vars.get(cohort_name, [])
                vars_str = ";".join(vars_for_cohort) if vars_for_cohort else ""
                if cohort_blob is None:
                    rows.append({
                        "model_id":      model_dir.name,
                        "partition":     _partition_label(partition),
                        "cohort":        cohort_name,
                        "n_cohort":      None,
                        "n_total":       int(n_total) if n_total is not None else None,
                        "fraction":      None,
                        "required_vars": vars_str,
                    })
                    continue
                rows.append({
                    "model_id":      model_dir.name,
                    "partition":     _partition_label(partition),
                    "cohort":        cohort_name,
                    "n_cohort":      cohort_blob.get("n_cohort"),
                    "n_total":       cohort_blob.get("n_total"),
                    "fraction":      cohort_blob.get("fraction"),
                    "required_vars": vars_str,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No cohort_counts.json files found under %s", metrics_root)
        return df

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        log.info(
            "Wrote %s: %d rows across %d models",
            output_path.name, len(df), df["model_id"].nunique(),
        )
    return df
