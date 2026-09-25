"""
analysis_figures.py
===================

Place in ``trauma_ml/outputs/`` (sibling of ``mortality/``, ``iss_band/``,
``niss_band/``).  Run from PyCharm or the command line.

Creates ``analysis/<target>/`` folders with paper-quality figures and
companion CSV summaries.

For each target it produces, per subset (full and baseline_complete):

  Mortality                              <target>_<subset>.png
    6 subplots (one per configuration axis: model family, calibration,
    imputer, phase cutoff, missingness, augmentation).  Within each
    subplot, four grouped bars per level showing AUROC test, AUROC holdout,
    AUPRC test, AUPRC holdout (mean ± SD across all models in that group).
    TRISS / NISS / ISS baseline values appear as dotted horizontal lines
    on the matching subset.  Each subplot title carries the result of an
    assumption test (Shapiro / Levene) plus the appropriate omnibus test
    (one-way ANOVA when normality + equal variance hold, otherwise
    Kruskal-Wallis) on the test AUROC across levels.

  ISS_band, NISS_band                    <target>_<subset>.png
    Same layout but using balanced accuracy (two bars per level: test +
    holdout), with the natural baseline (ISS or NISS) as horizontal
    lines on the matching subset.

Additional outputs per target:
  top_models.csv         the 4-row best-per-(metric × partition) table,
                          on each subset.
  stats_summary.csv      omnibus test outcome per subplot, p-value,
                          assumptions verdict, significant post-hoc pairs.

Dependencies: pandas, numpy, matplotlib, scipy.
No seaborn dependency.
"""

from __future__ import annotations
import json
import re
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Paths — relative to this file
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
OUT_DIR    = SCRIPT_DIR / "analysis"
TARGETS    = ("mortality", "iss_band", "niss_band",
              "iss_band_binary", "niss_band_binary")
# Binary-style targets (AUROC/AUPRC/TPR/FPR) vs 4-class band targets
# (balanced accuracy + per-class recall).
BINARY_TARGETS = ("mortality", "iss_band_binary", "niss_band_binary")

# Human-readable target names for figure titles. Missing entries fall back to a
# prettified version of the raw key, so an unseen target degrades gracefully
# instead of raising KeyError.
TARGET_PRETTY = {
    "mortality":        "In-hospital mortality",
    "iss_band":         "ISS band",
    "niss_band":        "NISS band",
    "iss_band_binary":  "Severe injury (ISS >= 16)",
    "niss_band_binary": "Severe injury (NISS >= 16)",
}
BAND_TARGETS   = ("iss_band", "niss_band")
BAND_BASELINE = {"iss_band": "ISS", "niss_band": "NISS",
                 "iss_band_binary": "ISS", "niss_band_binary": "NISS"}

# ─────────────────────────────────────────────────────────────────────────────
# Paper-quality matplotlib styling
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "savefig.dpi":        300,
    "figure.dpi":         120,
    "axes.facecolor":     "#fbfbfb",
    "axes.edgecolor":     "#333",
    "axes.linewidth":     0.8,
    "axes.titlesize":     10.5,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.color":         "#d8d8d8",
    "grid.alpha":         0.7,
    "grid.linestyle":     "-",
    "grid.linewidth":     0.4,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "xtick.color":        "#333",
    "ytick.color":        "#333",
    "legend.fontsize":    8,
    "legend.frameon":     False,
    "font.family":        "sans-serif",
    "font.size":          12,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.autolayout":  False,
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
})

# Color palette — academic + colorblind-conscious
PALETTE = {
    # Metric bars — AUROC/AUPRC (mortality primary)
    "auroc_test":     "#1f4e79",   # deep blue
    "auroc_holdout":  "#7fa9d8",   # light blue
    "auprc_test":     "#b22222",   # firebrick
    "auprc_holdout":  "#f4a4a4",   # light coral
    # Metric bars — TPR / FPR (mortality secondary plot)
    "tpr_test":       "#1b7837",   # dark green
    "tpr_holdout":    "#a1d99b",   # light green
    "fpr_test":       "#8b4513",   # saddle-brown
    "fpr_holdout":    "#d2b48c",   # tan
    # Metric bars — balanced accuracy (bands primary)
    "bacc_test":      "#1b7837",   # dark green
    "bacc_holdout":   "#a1d99b",   # light green
    # Per-class recall bars (bands secondary)
    "recall_0_test":  "#1f4e79",   "recall_0_holdout": "#7fa9d8",
    "recall_1_test":  "#1b7837",   "recall_1_holdout": "#a1d99b",
    "recall_2_test":  "#b22222",   "recall_2_holdout": "#f4a4a4",
    "recall_3_test":  "#8b4513",   "recall_3_holdout": "#d2b48c",
    # Baselines
    "TRISS":          "#762a83",   # purple
    "NISS":           "#e67e22",   # orange
    "ISS":            "#8e6e53",   # taupe
    "RTS":            "#2c7fb8",   # blue
    "MGAP":           "#41ab5d",   # green
    "mREMS":          "#c51b8a",   # magenta
}

# Clinical-score baselines drawn on the figures (raw / non-calibrated only).
BASELINE_NAMES = ["TRISS", "NISS", "ISS", "RTS", "MGAP", "mREMS"]

# ISS/NISS band class labels (classes "0"-"3" in the JSON)
BAND_CLASS_LABELS = {
    "iss_band":  {0: "ISS 1-8 (Minor)", 1: "ISS 9-15 (Moderate)",
                  2: "ISS 16-24 (Serious)", 3: "ISS 25+ (Critical)"},
    "niss_band": {0: "NISS 1-8 (Minor)", 1: "NISS 9-15 (Moderate)",
                  2: "NISS 16-24 (Serious)", 3: "NISS 25+ (Critical)"},
}

PRETTY_METRIC = {
    "auroc_test":    "AUROC (test)",
    "auroc_holdout": "AUROC (holdout)",
    "auprc_test":    "AUPRC (test)",
    "auprc_holdout": "AUPRC (holdout)",
    "tpr_test":      "TPR/Recall (test)",
    "tpr_holdout":   "TPR/Recall (holdout)",
    "fpr_test":      "FPR (test)",
    "fpr_holdout":   "FPR (holdout)",
    "bacc_test":     "Balanced accuracy (test)",
    "bacc_holdout":  "Balanced accuracy (holdout)",
    # Per-class recall labels are set dynamically in plot_subplot using BAND_CLASS_LABELS
    **{f"recall_{c}_test":    f"Recall class-{c} (test)"    for c in range(4)},
    **{f"recall_{c}_holdout": f"Recall class-{c} (holdout)" for c in range(4)},
}

# Configuration axes shown as subplots
CONFIG_AXES = [
    ("cfg_model_family",          "Model family"),
    ("cfg_calibration",           "Calibration method"),
    ("cfg_imputer_method",        "Imputation strategy"),
    ("cfg_phase_cutoff",          "Phase cutoff (predictor availability)"),
    ("cfg_missingness_threshold", "Missingness threshold"),
    ("cfg_data_augmentation",     "Data augmentation"),
]

# Logical orderings where meaningful
# ═════════════════════════════════════════════════════════════════════════════
# Presentation names
# ─────────────────────────────────────────────────────────────────────────────
# The internal family keys carry "doshi" for historical reasons, but only ONE
# of them actually reproduces Doshi et al.:
#
#   doshi_ffnn           -> a plain feed-forward net on TABULAR clinical
#                           predictors. Architecturally a standard MLP; it is
#                           NOT the architecture proposed in that paper, so
#                           labelling it "Doshi" in a figure would misattribute
#                           it. Presented simply as "FFNN".
#   doshi_ffnn_icd       -> multi-hot ICD-10 -> severity. This IS the faithful
#                           reproduction; cite Doshi et al. in the caption.
#   doshi_ffnn_icd_plus  -> ICD multi-hot + the L3 tabular block. A hybrid of
#                           our own, not in the paper.
#
# Applied at RENDER time only. The underlying cfg_model_family values are never
# rewritten: model_id, the resume tracker and every stored metric depend on the
# original strings.
DISPLAY_NAMES = {
    "doshi_ffnn":           "FFNN (clinical)",
    "doshi_ffnn_icd":       "FFNN (ICD)",
    "doshi_ffnn_icd_plus":  "FFNN (ICD + clinical)",
}


def display_name(value) -> str:
    """Map an internal level/family key to its presentation label."""
    s = str(value)
    return DISPLAY_NAMES.get(s, s)


LEVEL_ORDER = {
    "cfg_phase_cutoff": [
        "On-scene", "On-scene + ED arrival",
        "On-scene + ED arrival + In-hospital", "all_baseline_inputs",
    ],
    "cfg_calibration":    ["none", "platt", "isotonic"],
    "cfg_imputer_method": ["median_mode", "mice", "bagged_trees",
                           "missforest", "gradient_boosting", "none"],
    "cfg_data_augmentation": ["null", "None", "smote", "adasyn"],
}


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════
def _coerce_metric_cols(df: pd.DataFrame) -> None:
    """In-place: cast all known metric columns to float."""
    NUMERIC_PREFIXES = (
        "AUROC__", "AUROC_ovr__", "AUPRC__", "Brier__",
        "balanced_accuracy__", "accuracy__",
        "recall__", "recall_macro__", "precision__",
        "FPR__", "FNR__", "f1__", "f1_macro__", "f1_weighted__",
        "logloss__",
    )
    for c in df.columns:
        if c.startswith(NUMERIC_PREFIXES):
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_target(target_dir: Path) -> Optional[pd.DataFrame]:
    """Load all_metrics.csv if present, else walk per-model JSONs."""
    csv_path = target_dir / "all_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
        _coerce_metric_cols(df)
        print(f"  loaded {len(df)} rows from {csv_path.name}")

        # ── Cohort subsets live in SEPARATE files, not extra columns ────────
        # 99_aggregate.slurm writes all_metrics__cohort_<name>.csv alongside
        # all_metrics.csv, each using the SAME column names. Nothing merged
        # them, so detect_metric_col() found no "__baseline_complete" columns
        # and every baseline_complete figure rendered as "missing columns".
        # Merge them in here, suffixing the metric columns with the cohort.
        for cpath in sorted(target_dir.glob("all_metrics__cohort_*.csv")):
            cohort = cpath.stem.replace("all_metrics__cohort_", "")
            try:
                cdf = pd.read_csv(cpath, keep_default_na=False, na_values=[""])
            except Exception as exc:
                print(f"  [cohort] cannot read {cpath.name}: {exc}")
                continue
            if "model_id" not in cdf.columns:
                continue
            _coerce_metric_cols(cdf)
            METRIC_PFX = ("AUROC", "AUROC_ovr", "AUPRC", "Brier", "logloss",
                          "accuracy", "balanced_accuracy", "precision",
                          "recall", "specificity", "f1", "FPR", "FNR")
            keep = {"model_id": "model_id"}
            for c in cdf.columns:
                if c == "model_id":
                    continue
                if c.split("__")[0] in METRIC_PFX:
                    keep[c] = f"{c}__{cohort}"
            sub = cdf[list(keep)].rename(columns=keep)
            before = df.shape[1]
            df = df.merge(sub, on="model_id", how="left", suffixes=("", "_dup"))
            df = df[[c for c in df.columns if not c.endswith("_dup")]]
            print(f"  [cohort] merged {cpath.name}: "
                  f"+{df.shape[1]-before} columns ({cohort})")

        # ── TPOT supplement: tpt_*/iss_tpt_*/niss_tpt_* models are only written
        # by 99_aggregate.slurm AFTER all jobs finish.  When the CSV exists but
        # lacks TPOT rows, walk the metrics/ folder directly and merge any tpt_*
        # model dirs that are absent from the CSV.
        # tnt_ic_ prefix is used for the imputer_check TPOT variant (15_train_tpot.slurm)
        TPOT_PREFIXES = ("tpt_", "iss_tpt_", "niss_tpt_", "tpt_ic_", "tnt_ic_", "tnt_")
        if "model_id" in df.columns:
            in_csv = set(df["model_id"].astype(str).tolist())
            has_tpot_csv = any(m.startswith(TPOT_PREFIXES) for m in in_csv)
            metrics_root = target_dir / "metrics"
            models_root  = target_dir / "models"
            if not has_tpot_csv and metrics_root.exists():
                tpot_dirs = [d for d in metrics_root.iterdir()
                             if d.is_dir() and d.name.startswith(TPOT_PREFIXES)
                             and d.name not in in_csv]
                if tpot_dirs:
                    print(f"  [tpot] supplementing CSV with {len(tpot_dirs)} "
                          f"tpt_* model dirs found in metrics/")
                    # Build a mini metrics_root containing only tpot dirs
                    import tempfile, shutil
                    with tempfile.TemporaryDirectory() as tmp:
                        tmp_root = Path(tmp)
                        for d in tpot_dirs:
                            shutil.copytree(d, tmp_root / d.name)
                        tpot_df = _build_from_jsons(tmp_root, models_root)
                    _coerce_metric_cols(tpot_df)
                    df = pd.concat([df, tpot_df], ignore_index=True, sort=False)
                    print(f"  [tpot] merged → {len(df)} total rows")
                else:
                    print("  [warn] no TPOT models in metrics/ either "
                          "— tpt_* jobs may still be running")
        return df

    metrics_root = target_dir / "metrics"
    models_root  = target_dir / "models"
    if not metrics_root.exists():
        print(f"  no metrics/ directory at {metrics_root} — skipping")
        return None

    df = _build_from_jsons(metrics_root, models_root)
    print(f"  built {len(df)} rows from {metrics_root.parent.name}/metrics/")

    bl_path = target_dir / "baseline_metrics.csv"
    if bl_path.exists():
        bl = pd.read_csv(bl_path, keep_default_na=False, na_values=[""])
        if "model_id" not in bl.columns:
            for c in ("id", "name", "baseline_id"):
                if c in bl.columns:
                    bl = bl.rename(columns={c: "model_id"})
                    break
        bl["model_id"] = bl["model_id"].astype(str).apply(
            lambda x: x if x.startswith("baseline:") else f"baseline:{x}")
        for c in bl.columns:
            if c.startswith(("AUROC__", "AUPRC__", "balanced_accuracy__")):
                bl[c] = pd.to_numeric(bl[c], errors="coerce")
        df = pd.concat([df, bl], ignore_index=True, sort=False)
        print(f"  + appended {len(bl)} baseline rows")
    return df


def _build_from_jsons(metrics_root: Path, models_root: Path) -> pd.DataFrame:
    rows = []
    PART = {"test": "random_test", "holdout": "holdout_external",
            "holdout_year": "holdout_year",
            "train": "train", "calibration": "calibration_set"}
    for model_dir in sorted(metrics_root.iterdir()):
        if not model_dir.is_dir():
            continue
        row = {"model_id": model_dir.name}
        for jf in sorted(model_dir.glob("overall__*.json")):
            stem = jf.stem.replace("overall__", "")
            if "__" in stem:
                part_name, _, subset = stem.partition("__")
                # overall__test__cohort_baseline_complete → subset="cohort_baseline_complete"
                # Strip the cohort_ prefix so model rows use the same bare name as
                # baseline rows (e.g. "baseline_complete" not "cohort_baseline_complete").
                if subset and subset.startswith("cohort_"):
                    subset = subset[len("cohort_"):]
            else:
                part_name, subset = stem, None
            partition = PART.get(part_name, part_name)
            try:
                d = json.loads(jf.read_text())
            except Exception:
                continue
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    col = f"{k}__{partition}" + (f"__{subset}" if subset else "")
                    row[col] = float(v)
                elif isinstance(v, dict):
                    for ss, val in v.items():
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            if ss == "full":
                                row[f"{k}__{partition}"] = float(val)
                            else:
                                row[f"{k}__{partition}__{ss}"] = float(val)
        # Infer family from model_id prefix as fallback (works without models/)
        mid = row.get("model_id", "")
        if str(mid).startswith(("tpt_", "tpt_ic_", "tnt_ic_", "tnt_",
                                  "iss_tpt_", "niss_tpt_")):
            row.setdefault("cfg_model_family", "tpot")

        cfg_path = models_root / model_dir.name / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                cfg = {}
            inner = cfg.get("config") if isinstance(cfg.get("config"), dict) else {}
            for top, dst in (("family", "cfg_model_family"),
                              ("actual_model_type", "cfg_actual_model_type")):
                if top in cfg:
                    row[dst] = cfg[top]
            for src_key, dst in (
                ("phase_cutoff",          "cfg_phase_cutoff"),
                ("imputer_method",        "cfg_imputer_method"),
                ("imputer_check",         "cfg_imputer_check"),
                ("calibration",           "cfg_calibration"),
                ("data_augmentation",     "cfg_data_augmentation"),
                ("missingness_threshold", "cfg_missingness_threshold"),
                ("model_family",          "cfg_model_family"),
            ):
                if src_key in inner:
                    row[dst] = inner[src_key]
                elif src_key in cfg:
                    row[dst] = cfg[src_key]
        rows.append(row)

    # Also collect baseline rows directly from baseline__*.json files.
    # One synthetic row per scorer (ISS / NISS / TRISS); partitions and
    # cohort-subsets are spread across columns using the same naming
    # convention that detect_metric_col() and extract_baselines() expect.
    baseline_rows = _build_baselines_from_jsons(metrics_root)
    all_rows = pd.DataFrame(rows)
    if baseline_rows:
        all_rows = pd.concat(
            [all_rows, pd.DataFrame(baseline_rows)],
            ignore_index=True, sort=False,
        )
    return all_rows


def _build_baselines_from_jsons(metrics_root: Path) -> list[dict]:
    """Scan every model dir for ``baseline__<SCORER>__<partition>[__cohort_<name>].json``
    files and accumulate one synthetic 'baseline row' per scorer.

    Column name mapping (to match detect_metric_col / extract_baselines):
      • baseline__ISS__test.json         → AUROC__random_test, AUPRC__random_test …
      • baseline__ISS__holdout.json      → AUROC__holdout_external, …
      • baseline__ISS__test__cohort_baseline_complete.json
                                          → AUROC__random_test__cohort_baseline_complete, …
    Only ``test`` and ``holdout`` partitions are used (train/calibration
    are irrelevant for analysis_figures).
    Only raw (non-calibrated) files are used (no ``__platt``/``__isotonic`` suffix).
    When multiple model dirs contain the same baseline file the values
    should be identical; the last-seen value wins (safe).
    """
    PART_MAP = {
        "test":         "random_test",
        "holdout":      "holdout_external",
        "holdout_year": "holdout_year",   # year-stratified internal holdout (DIPC runs)
    }
    METRICS = ("AUROC", "AUPRC", "Brier", "logloss",
               "accuracy", "precision", "recall", "f1",
               "specificity", "FPR", "FNR", "prevalence",
               "n_valid", "threshold")
    CALIBRATION_SUFFIXES = {"platt", "isotonic", "threshold_tuned"}

    # Accumulate: {scorer_upper: {col: value}}
    accum: dict[str, dict] = {}

    for model_dir in sorted(metrics_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("baseline__*.json")):
            parts = jf.stem.split("__")
            # Expect: ["baseline", scorer, partition] or
            #         ["baseline", scorer, partition, "cohort_<name>"]
            # Skip calibrated variants (e.g. __platt, __isotonic)
            if len(parts) < 3:
                continue
            _, scorer, partition = parts[0], parts[1], parts[2]
            if partition not in PART_MAP:
                continue  # skip train / calibration
            if len(parts) == 4 and parts[3] in CALIBRATION_SUFFIXES:
                continue  # skip calibrated baseline files
            # Cohort subset (None → full partition row)
            cohort_tag: Optional[str] = None
            if len(parts) == 4 and parts[3].startswith("cohort_"):
                cohort_tag = parts[3].removeprefix("cohort_")
            elif len(parts) > 4:
                continue  # unexpected shape — skip

            try:
                blob = json.loads(jf.read_text())
            except Exception:
                continue

            scorer_upper = scorer.upper()
            model_id = f"baseline:{scorer_upper}"
            if scorer_upper not in accum:
                accum[scorer_upper] = {"model_id": model_id}

            part_key = PART_MAP[partition]
            for metric in METRICS:
                if metric not in blob:
                    continue
                v = blob[metric]
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                # Build the column name to match detect_metric_col():
                #   AUROC__random_test                  (full partition)
                #   AUROC__random_test__cohort_baseline_complete   (cohort subset)
                col_suffix = f"__{cohort_tag}" if cohort_tag else ""
                col = f"{metric}__{part_key}{col_suffix}"
                accum[scorer_upper][col] = float(v)

    return list(accum.values())


def detect_metric_col(df: pd.DataFrame, metric: str, partition: str,
                       subset: str) -> Optional[str]:
    """metric in {AUROC, AUPRC, balanced_accuracy, recall, FPR, recall__N};
    partition in {test, holdout}; subset in {full, baseline_complete}.

    For the holdout partition we try three column-name variants:
      * ``holdout_external``  — external parquet holdout (--holdout-dataset)
      * ``holdout``           — legacy label that may appear in older runs
      * ``holdout_year``      — year-stratified internal holdout (DIPC runs)
    """
    part_kws = {"test":    ("random_test", "test"),
                "holdout": ("holdout_external", "holdout", "holdout_year")}[partition]
    if subset == "full":
        for kw in part_kws:
            if f"{metric}__{kw}" in df.columns:
                return f"{metric}__{kw}"
            if f"{metric}__{kw}__full" in df.columns:
                return f"{metric}__{kw}__full"
    else:
        for kw in part_kws:
            # Try bare subset name (e.g. AUROC__random_test__baseline_complete)
            if f"{metric}__{kw}__{subset}" in df.columns:
                return f"{metric}__{kw}__{subset}"
            # Also try with the cohort_ prefix produced when _build_from_jsons
            # parses overall__<part>__cohort_<name>.json filenames
            if f"{metric}__{kw}__cohort_{subset}" in df.columns:
                return f"{metric}__{kw}__cohort_{subset}"
    return None


def extract_baselines(df: pd.DataFrame) -> dict:
    """{baseline_name: {(metric, partition, subset): value}}"""
    out: dict = {}
    mask = df["model_id"].astype(str).str.startswith("baseline:")
    for _, row in df[mask].iterrows():
        full = row["model_id"]
        bare = full.replace("baseline:", "").split("_")[0].upper()
        out.setdefault(bare, {})
        for metric in ("AUROC", "AUPRC", "balanced_accuracy"):
            for part in ("test", "holdout"):
                for ss in ("full", "baseline_complete"):
                    col = detect_metric_col(df, metric, part, ss)
                    if col and pd.notna(row.get(col)):
                        out[bare][(metric, part, ss)] = float(row[col])
    return out


def filter_to_models(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["model_id"].astype(str).str.startswith("baseline:")
    return df[mask].reset_index(drop=True)


# ── Faithful Doshi ICD->severity FFNN: analysed SEPARATELY from the main
#    per-family charts, and compared only against the other L3 approaches. ──
ICD_FAMILIES = ("doshi_ffnn_icd", "doshi_ffnn_icd_plus")
ICD_PREFIXES = ("doshi_icd", "doshi_icdplus")
L3_CUTOFF = "On-scene + ED arrival + In-hospital"
ICD_TARGETS = ("mortality", "iss_band_binary", "niss_band_binary")


def _is_icd_row(row) -> bool:
    fam = str(row.get("cfg_model_family", ""))
    mid = str(row.get("model_id", ""))
    return fam in ICD_FAMILIES or mid.startswith(ICD_PREFIXES)


def split_icd_models(models: pd.DataFrame):
    """Return (icd_models, non_icd_models)."""
    if models.empty:
        return models, models
    is_icd = models.apply(_is_icd_row, axis=1)
    return (models[is_icd].reset_index(drop=True),
            models[~is_icd].reset_index(drop=True))


def make_icd_comparison_figure(all_models: pd.DataFrame, target: str,
                                out_path: Path) -> None:
    """Compare the faithful Doshi ICD FFNN (ICD-only and ICD+L3) against every
    OTHER model family, restricted to the L3 (in-hospital) cutoff, on AUROC and
    AUPRC (test + holdout).  Only produced for the binary targets that the ICD
    models cover.  No-op if no ICD models are present for this target."""
    if target not in ICD_TARGETS:
        return
    icd, non_icd = split_icd_models(all_models)
    if icd.empty:
        print(f"    [icd] no ICD models for {target} — skipping comparison")
        return

    def _best(sub):
        col = detect_metric_col(sub, "AUROC", "test", "full")
        if col is None:
            return None
        sub = sub.dropna(subset=[col])
        if sub.empty:
            return None
        _i = pick_best(sub, col, ascending=False)
        return None if _i is None else sub.loc[_i]

    # Approaches: the two ICD variants + each other family's best L3 model.
    approaches = []
    only = icd[icd.apply(lambda r: str(r.get("cfg_model_family")) == "doshi_ffnn_icd"
                         or str(r.get("model_id")).startswith("doshi_icd_"), axis=1)]
    plus = icd[icd.apply(lambda r: str(r.get("cfg_model_family")) == "doshi_ffnn_icd_plus"
                         or str(r.get("model_id")).startswith("doshi_icdplus_"), axis=1)]
    for label, sub in [("FFNN\n(ICD)", only), ("FFNN\n(ICD +\nclinical)", plus)]:
        row = _best(sub)
        if row is not None:
            approaches.append((label, row, True))

    l3 = non_icd[non_icd["cfg_phase_cutoff"].astype(str) == L3_CUTOFF] \
        if "cfg_phase_cutoff" in non_icd.columns else non_icd
    for fam in sorted(l3.get("cfg_model_family", pd.Series(dtype=str)).dropna().unique()):
        row = _best(l3[l3["cfg_model_family"].astype(str) == fam])
        if row is not None:
            approaches.append((display_name(fam), row, False))

    if len(approaches) < 2:
        print(f"    [icd] not enough comparable approaches for {target}")
        return

    metric_specs = [("AUROC", "test"), ("AUROC", "holdout"),
                    ("AUPRC", "test"), ("AUPRC", "holdout")]
    colors = {"AUROC_test": PALETTE["auroc_test"], "AUROC_holdout": PALETTE["auroc_holdout"],
              "AUPRC_test": PALETTE["auprc_test"], "AUPRC_holdout": PALETTE["auprc_holdout"]}

    labels = [a[0] for a in approaches]
    x = np.arange(len(approaches)); w = 0.2
    fig, ax = plt.subplots(figsize=(max(9, 1.15 * len(approaches)), 6.4))
    allvals = []
    for k, (metric, part) in enumerate(metric_specs):
        vals = []
        for _, row, _ in approaches:
            col = detect_metric_col(all_models, metric, part, "full")
            v = float(row.get(col)) if (col and pd.notna(row.get(col))) else np.nan
            vals.append(v); allvals.append(v)
        ax.bar(x + (k - 1.5) * w, vals, w, label=f"{metric} ({part})",
               color=colors[f"{metric}_{part}"], edgecolor="white", linewidth=0.5)

    finite = [v for v in allvals if np.isfinite(v)]
    if finite:
        lo = max(0.0, min(finite) - 0.02)
        ax.set_ylim(lo, min(1.0, max(finite) + (max(finite) - lo) * 0.18))
    # highlight the two ICD approaches
    for i, (_, _, is_icd_app) in enumerate(approaches):
        if is_icd_app:
            ax.axvspan(i - 0.5, i + 0.5, color="#fff3cd", alpha=0.5, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    # Legend ABOVE the title (title was previously overlapped by the legend),
    # and start the y-axis at 0 so AUPRC bars are not visually truncated.
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{target}: ICD-based FFNN vs other in-hospital (L3) approaches",
                 fontsize=14, fontweight="bold", pad=34)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.10),
              frameon=False, fontsize=11)
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ wrote {out_path.name} ({len(approaches)} approaches)")




def get_metric_calibrated(df: pd.DataFrame, metric: str, partition: str,
                            subset: str) -> pd.Series:
    """Per-row metric values, swapping in __platt / __isotonic columns when
    the row's cfg_calibration matches.

    The trainer writes one metric JSON per calibration variant:
        overall__test.json              → AUROC__random_test (uncalibrated)
        overall__test__platt.json       → AUROC__random_test__platt
        overall__test__isotonic.json    → AUROC__random_test__isotonic
    (And the same pattern for the baseline_complete cohort, holdout
     partitions, recall/FPR/balanced_accuracy/per-class recalls, etc.)

    detect_metric_col() always resolves to the UNCALIBRATED base column.
    This helper returns a per-row Series where each row whose
    cfg_calibration == "platt" picks up the __platt column, "isotonic" the
    __isotonic column, and "none" (or anything else) keeps the base column.

    Important: AUROC/AUPRC are rank-preserving under monotonic calibration,
    so calibrated vs. uncalibrated columns differ only by tiny
    cross-fitting noise from CalibratedClassifierCV(cv=5).  TPR/FPR at a
    fixed threshold, on the other hand, do shift meaningfully — that is
    what calibration is for.  Brier and ECE are where calibration actually
    moves the needle visibly.
    """
    base_col = detect_metric_col(df, metric, partition, subset)
    if base_col is None:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    result = pd.to_numeric(df[base_col], errors="coerce").copy()
    if "cfg_calibration" not in df.columns:
        return result
    cal_series = df["cfg_calibration"].astype(str).str.lower()
    for cal in ("platt", "isotonic"):
        cal_col = f"{base_col}__{cal}"      # e.g. AUROC__random_test__platt
        if cal_col in df.columns:
            cal_vals = pd.to_numeric(df[cal_col], errors="coerce")
            mask = (cal_series == cal) & cal_vals.notna()
            result.loc[mask] = cal_vals.loc[mask]
    return result

# ═════════════════════════════════════════════════════════════════════════════
# Statistics
# ═════════════════════════════════════════════════════════════════════════════
def assumption_tests(groups: list[np.ndarray]) -> tuple[bool, bool, str]:
    """Check normality (Shapiro per group) and homoscedasticity (Levene).
    Returns (normal, equal_var, summary_str)."""
    normal = True
    norm_info = []
    for i, g in enumerate(groups):
        n = len(g)
        if n < 3:
            normal = False
            norm_info.append(f"g{i}: n={n} too small")
            continue
        try:
            sample = g[:min(5000, n)]   # Shapiro spec-limit
            _, p = stats.shapiro(sample)
            norm_info.append(f"g{i}: p={p:.3g}")
            if p < 0.05:
                normal = False
        except Exception as e:
            normal = False
            norm_info.append(f"g{i}: err")
    equal_var = False
    lev_str = ""
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        try:
            _, p_lev = stats.levene(*groups)
            equal_var = (p_lev >= 0.05)
            lev_str = f"Levene p={p_lev:.3g}"
        except Exception:
            lev_str = "Levene err"
    summary = f"normal={normal}, equal_var={equal_var}"
    return normal, equal_var, summary


def omnibus_test(groups: list[np.ndarray], normal: bool,
                  equal_var: bool) -> tuple[str, float]:
    """Pick the right test depending on number of groups & assumptions.

    For k = 2 groups: independent t-test (parametric) or Mann-Whitney U
    (non-parametric).  For k >= 3: ANOVA or Kruskal-Wallis.

    (Note: KW with k=2 is mathematically equivalent to MWU; labelling it
    "Mann-Whitney U" is more precise for the reader.)
    """
    valid = [g for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return "n/a", float("nan")
    try:
        if len(valid) == 2:
            if normal and equal_var:
                _, p = stats.ttest_ind(valid[0], valid[1], equal_var=True)
                return "t-test", float(p)
            else:
                _, p = stats.mannwhitneyu(valid[0], valid[1], alternative="two-sided")
                return "Mann-Whitney U", float(p)
        else:
            if normal and equal_var:
                _, p = stats.f_oneway(*valid)
                return "ANOVA", float(p)
            else:
                _, p = stats.kruskal(*valid)
                return "Kruskal-Wallis", float(p)
    except Exception:
        return "n/a", float("nan")


def posthoc_pairwise(groups: dict[str, np.ndarray], normal: bool,
                      equal_var: bool, alpha: float = 0.05
                      ) -> list[tuple[str, str, float, bool]]:
    """Pairwise post-hoc.  Returns list of (level_a, level_b, p_adj, sig).
    Uses Welch t-test (parametric) or Mann-Whitney U (non-parametric),
    Bonferroni-corrected."""
    levels = list(groups.keys())
    pairs = list(combinations(levels, 2))
    results = []
    n_comp = len(pairs)
    for a, b in pairs:
        ga, gb = groups[a], groups[b]
        if len(ga) < 2 or len(gb) < 2:
            results.append((a, b, float("nan"), False))
            continue
        try:
            if normal and equal_var:
                _, p_raw = stats.ttest_ind(ga, gb, equal_var=False)
            else:
                _, p_raw = stats.mannwhitneyu(ga, gb, alternative="two-sided")
            p_adj = min(1.0, p_raw * n_comp)  # Bonferroni
        except Exception:
            p_adj = float("nan")
        results.append((a, b, float(p_adj), bool(p_adj < alpha)))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════
def wrap_label(s: str, width: int = 14) -> str:
    import textwrap
    if not s or s in ("nan", "None", "none"):
        return s if s else ""
    s = display_name(s)          # presentation name before wrapping
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s


def order_levels(axis_col: str, present: list[str]) -> list[str]:
    pref = LEVEL_ORDER.get(axis_col, [])
    ordered = [p for p in pref if p in present]
    ordered += [lv for lv in present if lv not in ordered]
    return ordered


def aggregate_groups(df: pd.DataFrame, axis_col: str, metric_col: str,
                       levels: list[str]) -> dict[str, np.ndarray]:
    """Return {level: array of metric values} excluding NaN."""
    return {
        lv: df.loc[df[axis_col].astype(str) == lv, metric_col]
              .dropna().to_numpy()
        for lv in levels
    }


def best_in_group(df: pd.DataFrame, axis_col: str, metric_col: str,
                   level: str) -> Optional[tuple[str, float]]:
    sub = df[df[axis_col].astype(str) == level].copy()
    sub = sub.dropna(subset=[metric_col])
    if sub.empty:
        return None
    idx = pick_best(sub, metric_col, ascending=False)
    return str(sub.loc[idx, "model_id"]), float(sub.loc[idx, metric_col])


# ═════════════════════════════════════════════════════════════════════════════
# Plot a single subplot — mortality (4 bars per level) or bands (2 bars)
# ═════════════════════════════════════════════════════════════════════════════
def plot_subplot(ax, df: pd.DataFrame, baselines: dict, axis_col: str,
                  axis_label: str, subset: str, target: str,
                  metrics_mode: str = "default") -> dict:
    """Draw one subplot.  Returns a dict of statistical results for the log.

    metrics_mode:
      "default"          — AUROC/AUPRC (mortality) or balanced_accuracy (bands)
      "tpr_fpr"          — recall (TPR) and FPR for mortality
      "recall_by_class"  — per-class recall for band targets (classes 0-3)
    """
    # ── Metrics layout ──────────────────────────────────────────────────
    # lower_is_better: set of metric keys where the "best" = minimum, not maximum
    LOWER_IS_BETTER = {"fpr_test", "fpr_holdout"}

    if metrics_mode == "tpr_fpr" and target in BINARY_TARGETS:
        metrics_layout = [
            ("recall", "test",    "tpr_test"),
            ("recall", "holdout", "tpr_holdout"),
            ("FPR",    "test",    "fpr_test"),
            ("FPR",    "holdout", "fpr_holdout"),
        ]
        primary_key   = "tpr_test"
        baseline_names = list(BASELINE_NAMES)
    elif metrics_mode == "recall_by_class" and target in BAND_TARGETS:
        class_labels = BAND_CLASS_LABELS.get(target, {})
        metrics_layout = []
        for cls in range(4):
            cls_name = class_labels.get(cls, str(cls))
            # Override PRETTY_METRIC in-place for this call (local copy)
            PRETTY_METRIC[f"recall_{cls}_test"]    = f"{cls_name} (test)"
            PRETTY_METRIC[f"recall_{cls}_holdout"] = f"{cls_name} (holdout)"
            metrics_layout.append((f"recall__{cls}", "test",    f"recall_{cls}_test"))
            metrics_layout.append((f"recall__{cls}", "holdout", f"recall_{cls}_holdout"))
        primary_key   = "recall_0_test"
        baseline_names = []
    elif target in BINARY_TARGETS:
        metrics_layout = [
            ("AUROC", "test",    "auroc_test"),
            ("AUROC", "holdout", "auroc_holdout"),
            ("AUPRC", "test",    "auprc_test"),
            ("AUPRC", "holdout", "auprc_holdout"),
        ]
        primary_key   = "auroc_test"
        baseline_names = list(BASELINE_NAMES)
    else:
        metrics_layout = [
            ("balanced_accuracy", "test",    "bacc_test"),
            ("balanced_accuracy", "holdout", "bacc_holdout"),
        ]
        primary_key   = "bacc_test"
        baseline_names = [BAND_BASELINE.get(target, "")]

    # ── Resolve column names ──────────────────────────────────────────────
    metric_cols = {key: detect_metric_col(df, metric, part, subset)
                   for metric, part, key in metrics_layout}

    if axis_col not in df.columns or not all(metric_cols.values()):
        missing = [key for key, col in metric_cols.items() if not col]
        ax.text(0.5, 0.5, f"missing columns for\n{axis_label}\n({', '.join(missing)})",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                color="grey")
        ax.set_xticks([]); ax.set_yticks([])
        return {"axis": axis_col, "subset": subset, "test": None}

    sub = df.dropna(subset=[axis_col], how="any")
    sub = sub[sub[axis_col].astype(str).str.lower().isin(["nan", ""]) == False]
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return {"axis": axis_col, "subset": subset, "test": None}

    # ── Calibration-aware metric values ──────────────────────────────────
    # For each (metric, partition) in this subplot, replace the base column
    # with a per-row Series whose values come from the __platt / __isotonic
    # column when cfg_calibration matches.  This makes every downstream
    # read (bar means, KW/MWU, posthoc voting, ★ annotations) reflect the
    # ACTUAL evaluation of each row's model — not the uncalibrated proxy.
    sub = sub.copy()
    for metric, part, _key in metrics_layout:
        col = detect_metric_col(sub, metric, part, subset)
        if col is None:
            continue
        sub[col] = get_metric_calibrated(sub, metric, part, subset)

    levels = order_levels(axis_col,
                           sorted(sub[axis_col].astype(str).unique().tolist()))
    if not levels:
        ax.text(0.5, 0.5, "no levels", ha="center", transform=ax.transAxes)
        return {"axis": axis_col, "subset": subset, "test": None}

    n_metrics = len(metrics_layout)
    bar_width = 0.7 / n_metrics
    x_pos = np.arange(len(levels), dtype=float)
    centred_offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2.0) * bar_width

    # ── Draw bars ──────────────────────────────────────────────────────────
    all_means, all_stds = [], []
    for j, (metric, part, key) in enumerate(metrics_layout):
        col = metric_cols[key]
        means, stds = [], []
        for lv in levels:
            vals = sub.loc[sub[axis_col].astype(str) == lv, col].dropna().to_numpy()
            means.append(float(np.mean(vals)) if len(vals) else 0.0)
            stds.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0)
        all_means.append(means)
        all_stds.append(stds)
        ax.bar(x_pos + centred_offsets[j], means, bar_width,
               yerr=stds, capsize=3,
               color=PALETTE[key], edgecolor="black", linewidth=0.4,
               error_kw=dict(elinewidth=0.8, ecolor="#333"),
               label=PRETTY_METRIC[key], alpha=0.92)

    # ── Baseline horizontal lines ─────────────────────────────────────────
    LINE_STYLES = {k: ("-", 1.5) if k.endswith("_test") else ("--", 1.0)
                   for k in metric_cols}
    # Band-binary targets are ISS/NISS >= 16 -- an ISS/NISS "baseline" would be
    # the label itself, and TRISS/RTS/MGAP/mREMS predict mortality, not injury
    # severity. Drawing any of them here is meaningless, so suppress them.
    if target in ("iss_band_binary", "niss_band_binary"):
        baseline_names = []

    annotated_lines = []
    for bn in baseline_names:
        for metric, part, key in metrics_layout:
            v = baselines.get(bn, {}).get((metric, part, subset))
            if v is None:
                continue
            ls, lw = LINE_STYLES.get(key, ("-", 1.0))
            ax.axhline(v, color=PALETTE[bn], linestyle=ls, lw=lw, alpha=0.85)
            annotated_lines.append((bn, key, v))

    # ── Statistical tests ─────────────────────────────────────────────────
    primary_col = metric_cols.get(primary_key)
    groups_dict_primary = (aggregate_groups(sub, axis_col, primary_col, levels)
                           if primary_col else {})
    # Per-level means keyed by metric key (for direction-aware pair ranking)
    metric_level_means: dict[str, dict[str, float]] = {}
    for mk2, col_k2 in metric_cols.items():
        if col_k2 is None:
            continue
        gd2 = aggregate_groups(sub, axis_col, col_k2, levels)
        metric_level_means[mk2] = {lv: float(np.mean(g)) if len(g) else np.nan
                                    for lv, g in gd2.items()}
    level_means = metric_level_means.get(primary_key, {})

    stats_out = {
        "axis": axis_col, "subset": subset, "primary_metric": primary_key,
        "n_levels": len(levels),
        "n_models_total": int(sum(len(g) for g in groups_dict_primary.values())),
    }

    from collections import Counter, defaultdict
    # pair_vote[frozenset({a,b})] counts metrics for which the SAME member
    # consistently outperforms the other (direction-aware: for FPR, lower=better)
    pair_direction: dict[frozenset, Counter] = {}  # {pair: Counter({winner: count})}
    all_omnibus_p: list[float] = []
    headline_test_name = "Kruskal-Wallis"
    posthoc_rows: list[dict] = []      # every pairwise test, every metric
    primary_omnibus_p = None
    n_metrics_tested = 0

    for mk, col_k in metric_cols.items():
        if col_k is None:
            continue
        gd_k = aggregate_groups(sub, axis_col, col_k, levels)
        gl_k = [g for g in gd_k.values() if len(g) >= 2]
        if len(gl_k) < 2:
            continue
        norm_k, eqv_k, _ = assumption_tests(gl_k)
        tname_k, pval_k = omnibus_test(gl_k, norm_k, eqv_k)
        n_metrics_tested += 1
        if not np.isnan(pval_k):
            all_omnibus_p.append(pval_k)
        if mk == primary_key:
            primary_omnibus_p = pval_k
            headline_test_name = tname_k
            stats_out.update({"normality_ok": norm_k, "equal_var_ok": eqv_k,
                               "omnibus_test": tname_k})
        if not np.isnan(pval_k) and pval_k < 0.05:
            posthoc_k = posthoc_pairwise(gd_k, norm_k, eqv_k)
            mk_means = metric_level_means.get(mk, {})
            lower_better_k = mk in LOWER_IS_BETTER
            for a, b, p, s in posthoc_k:
                # Record EVERY pairwise test (significant or not) for the
                # companion CSV, before the `if not s` filter drops the
                # non-significant ones from the figure logic.
                posthoc_rows.append({"metric": mk, "level_a": a, "level_b": b,
                                     "p_adj": p, "significant": bool(s)})
                if not s:
                    continue
                pair = frozenset({a, b})
                if pair not in pair_direction:
                    pair_direction[pair] = Counter()
                # winner = lower mean if lower_better else higher mean
                mean_a = mk_means.get(a, 0.0)
                mean_b = mk_means.get(b, 0.0)
                if lower_better_k:
                    winner = a if mean_a <= mean_b else b
                else:
                    winner = a if mean_a >= mean_b else b
                pair_direction[pair][winner] += 1

    # The headline p MUST be the primary metric's omnibus, not max() across
    # metrics. Previously the figure showed max() (the worst p) while post-hoc
    # pairs were harvested from ANY metric with p<0.05 -- so a subplot could
    # read "KW p=0.879 ns" and still list inequalities (a direct contradiction),
    # or show a tiny p with no pairs at all.
    headline_p = primary_omnibus_p if primary_omnibus_p is not None else (
        max(all_omnibus_p) if all_omnibus_p else np.nan)
    stats_out["omnibus_p"] = headline_p

    # Standard practice: no post-hoc unless the omnibus is significant. This
    # makes the reported inequalities logically consistent with the p shown.
    if np.isnan(headline_p) or headline_p >= 0.05:
        pair_direction = {}

    # A pair is consistent if the SAME member wins across ALL tested metrics
    # A pair is reported when it is significant in a MAJORITY of the tested
    # metrics AND the direction is unanimous among those.  The previous rule
    # required significance in *every* metric, which is why subplots with a
    # tiny omnibus p often listed no pairs at all: one metric failing to reach
    # the Bonferroni-corrected threshold suppressed the entire pair.
    min_required = max(1, (n_metrics_tested + 1) // 2)   # majority
    consistent_sig = []
    for pair, cnt in pair_direction.items():
        n_sig = sum(cnt.values())
        if n_sig < min_required:
            continue
        top_winner, top_count = cnt.most_common(1)[0]
        if top_count == n_sig:            # unanimous direction where significant
            losers = [x for x in pair if x != top_winner]
            consistent_sig.append((top_winner,
                                   losers[0] if losers else top_winner))

    stats_out["posthoc_rows"] = posthoc_rows
    if consistent_sig:
        stats_out.update({"consistent_sig_pairs": consistent_sig,
                          "posthoc_method": "Mann-Whitney U (per metric)",
                          "posthoc_correction": "Bonferroni"})

    # ── Build title — single set_title string, everything inside ──────────────
    # Line 1 bold+large: category name
    # Line 2 normal: Kruskal-Wallis result
    # Guard against contradictory statements: if both "A > B" and "B > A"
    # somehow survive (possible when different metrics disagree), drop both
    # rather than print a self-contradicting caption.
    _seen = {}
    _bad = set()
    for w, l in consistent_sig:
        if (l, w) in _seen:
            _bad.add((w, l)); _bad.add((l, w))
        _seen[(w, l)] = True
    if _bad:
        consistent_sig = [x for x in consistent_sig if x not in _bad]

    # Lines 3+: significant pairs (wrapped at 55 chars per line)
    if not np.isnan(headline_p):
        p_str = f"{headline_p:.2e}" if headline_p < 0.001 else f"{headline_p:.3f}"
        sig_marker = ("***" if headline_p < 0.001 else "**" if headline_p < 0.01
                      else "*" if headline_p < 0.05 else "ns")
        # Abbreviate test name for compactness
        _abbrev = {"Kruskal-Wallis": "KW", "ANOVA": "ANOVA",
                    "Mann-Whitney U": "MWU", "t-test": "t-test"}
        stat_line = f"{_abbrev.get(headline_test_name, headline_test_name)} p={p_str} {sig_marker}"

        pairs_lines = []
        if consistent_sig:
            # Optionally abbreviate level names when they are long enough to
            # cause the title block to clip horizontally.  If the longest level
            # name is > 18 chars (e.g. "On-scene + ED arrival + In-hospital"),
            # we replace each level by a short letter code (L1, L2, ...) and
            # emit a legend mapping line so the reader can decode it.
            max_lvl_len = max((len(l) for l in levels), default=0)
            # Abbreviate only when level names are truly long (e.g. phase cutoff,
            # whose longest member is ~35 chars).  Model family names like
            # "logistic_elasticnet" (19 chars) stay verbose.
            if max_lvl_len > 25:
                level_short = {lv: f"L{i+1}" for i, lv in enumerate(levels)}
                level_legend = "; ".join(
                    f"{level_short[lv]}={display_name(lv)}" for lv in levels)
            else:
                # Presentation names (never the internal "doshi_*" keys).
                level_short = {lv: display_name(lv) for lv in levels}
                level_legend = None

            # Step 1: for each loser, collect its set of winners
            loser_to_winners: dict[str, set] = defaultdict(set)
            for winner, loser in consistent_sig:
                loser_to_winners[loser].add(winner)
            # Step 2: group losers that share the SAME winner-set onto one line.
            primary_is_lower = primary_key in LOWER_IS_BETTER
            winnerset_to_losers: dict[tuple, list[str]] = defaultdict(list)
            for lo, ws in loser_to_winners.items():
                key = tuple(sorted(ws,
                                    key=lambda x: level_means.get(x, 0),
                                    reverse=not primary_is_lower))
                winnerset_to_losers[key].append(lo)
            # Step 3: emit one "winners > losers" string per group, using the
            # short names so the line fits in the subplot width.
            group_strs = []
            def group_key(item):
                winners_tuple, losers = item
                if not winners_tuple:
                    return 0
                return level_means.get(winners_tuple[0], 0)
            for winners_tuple, losers in sorted(
                winnerset_to_losers.items(), key=group_key,
                reverse=not primary_is_lower,
            ):
                losers_sorted = sorted(losers,
                                        key=lambda x: level_means.get(x, 0),
                                        reverse=primary_is_lower)
                winners_short = [level_short[w] for w in winners_tuple]
                losers_short  = [level_short[l] for l in losers_sorted]
                group_strs.append(
                    f"{', '.join(winners_short)} > {', '.join(losers_short)}"
                )
            # Wrap groups across lines: each group on its own line if it's
            # itself > 50 chars (will already be hard-wrapped by the renderer
            # via fontsize, but we keep lines short to avoid horizontal clipping).
            # Use 50-char target per line; long single groups get their own line.
            cur, cur_len = [], 0
            MAX_PAIR_LINES = 3     # hard cap; the rest go to stats_summary.csv
            MAX_LINE = 44          # narrower: long lines were clipping the axes
            for gs in group_strs:
                if len(gs) >= MAX_LINE:
                    if cur:
                        pairs_lines.append("; ".join(cur))
                        cur, cur_len = [], 0
                    # A single group can still be far wider than the subplot
                    # (e.g. "a, b, c, d > e, f, g, h"); hard-wrap it rather
                    # than letting matplotlib clip it off the figure edge.
                    import textwrap as _tw
                    pairs_lines.extend(_tw.wrap(gs, width=MAX_LINE))
                    continue
                tok = ("; " + gs) if cur else gs
                if cur_len + len(tok) > MAX_LINE and cur:
                    pairs_lines.append("; ".join(cur))
                    cur, cur_len = [gs], len(gs)
                else:
                    cur.append(gs)
                    cur_len += len(tok)
            if cur:
                pairs_lines.append("; ".join(cur))

            # HARD CAP. With 9+ model families the post-hoc block ran to 8
            # lines and overran the figure suptitle. Keep the strongest few
            # on the plot; the complete list is already written to
            # stats_summary.csv, which is the right place for it.
            if len(pairs_lines) > MAX_PAIR_LINES:
                n_hidden = len(pairs_lines) - MAX_PAIR_LINES
                pairs_lines = pairs_lines[:MAX_PAIR_LINES]
                pairs_lines.append(
                    f"(+{n_hidden} more comparison(s) - see figure caption)")

            # If we abbreviated, append the legend line(s) at the end
            if level_legend is not None:
                # Wrap legend to MAX_LINE chars per line too
                parts = level_legend.split("; ")
                cur, cur_len = [], 0
                for p in parts:
                    tok = ("; " + p) if cur else p
                    if cur_len + len(tok) > MAX_LINE and cur:
                        pairs_lines.append("; ".join(cur))
                        cur, cur_len = [p], len(p)
                    else:
                        cur.append(p)
                        cur_len += len(tok)
                if cur:
                    pairs_lines.append("; ".join(cur))
    else:
        stat_line = "insufficient groups"
        pairs_lines = []

    # Build the title as a single multi-line string so matplotlib lays it out
    # cleanly above the axes — no overlapping artists, no transAxes text.
    # First line: category name (uppercase to visually distinguish it).
    # Remaining lines: omnibus result, then significant pairs (wrapped).
    # Single set_title: category name + stat + pairs.
    # Category name on its own line, larger via the matplotlib title artist
    # rendering all lines at the same fontsize.  We make the category name
    # itself visually larger by surrounding it with empty strings to add
    # vertical space.  ALL lines are bold to keep one weight.
    # Fold the omnibus result onto the SAME line as the category name: with up
    # to 4 pair lines this reclaims a whole line of vertical space per subplot,
    # which is what was pushing the top-row titles into the suptitle.
    head = f"{axis_label.upper()}  -  {stat_line}" if stat_line else axis_label.upper()
    title_lines = [head] + pairs_lines
    ax.set_title("\n".join(title_lines),
                  pad=26, fontsize=18, fontweight="bold",
                  linespacing=1.30, loc="center")
    # ── Axis ylim: include mean+sd AND headroom for annotation labels ─────
    # all_means[j] is per-level mean per metric; all_stds[j] is the matching sd.
    # The annotations sit at bar_top + up to ~12 % of y-range, so we reserve
    # enough headroom for that.
    means_only = [v for means in all_means for v in means]
    means_plus_sd = []
    for j in range(len(all_means)):
        for k in range(len(all_means[j])):
            means_plus_sd.append(all_means[j][k] + all_stds[j][k])
    bl_vals = [v for _, _, v in annotated_lines]
    base = means_only + means_plus_sd + bl_vals
    if base:
        # Zoom tightly to the data band so small differences are legible
        # (paper-style): lower bound just under the smallest value, modest
        # headroom for the staggered annotation labels.
        ymin_data = max(0.0, min(base) - 0.012)
        ymax_data = max(base)
        span = max(ymax_data - ymin_data, 0.05)
        ax.set_ylim(ymin_data, min(1.0, ymax_data + span * 0.34))
        ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
        ax.tick_params(axis="y", which="minor", length=2.5, color="#bbb")
        ax.grid(which="minor", axis="y", color="#ececec", linewidth=0.3)
    else:
        ymin_data, ymax_data = 0.0, 1.0
    ylo, yhi = ax.get_ylim()

    # ── Best-model annotations: ★ + label JUST above each bar ─────────────
    # For each metric we find the globally-best individual model (max for
    # maximisers, min for minimisers), place a ★ on the corresponding bar,
    # and write the model id + score directly above that bar.
    #
    # Vertical stagger is done in PIXEL coordinates (via transform offsets),
    # not data coordinates, so a label always sits visually just above its
    # bar regardless of whether the bar is at y=0.05 (FPR) or y=0.95 (AUROC).
    #
    # When two labels would land on the same bar we stagger them vertically
    # so they don't overlap.

    # ── Collect all annotation candidates first ─────────────────────────────
    raw_annots = []   # list of dicts before stagger assignment
    for j, (metric, part, key) in enumerate(metrics_layout):
        col_j = metric_cols[key]
        if col_j is None:
            continue
        sub_valid = sub.dropna(subset=[col_j])
        if sub_valid.empty:
            continue
        asc      = key in LOWER_IS_BETTER
        # sort_values(...).iloc[0] is a STABLE sort, so among models tied on
        # the metric it returned whichever came first in THIS subplot's frame.
        # Because sub_valid is filtered per axis (rows with a null level are
        # dropped), different subplots could report different winners for the
        # same tied score -- e.g. xgb_none_0038 under MODEL FAMILY but
        # xgb_none_0036 everywhere else. Route through the tie-break cascade.
        _bi = pick_best(sub_valid, col_j, ascending=asc,
                        metric=metric, subset=subset)
        if _bi is None:
            continue
        best_row = sub_valid.loc[_bi]
        best_id  = str(best_row["model_id"])
        best_val = float(best_row[col_j])
        best_lv  = str(best_row[axis_col])
        lv_idx   = levels.index(best_lv) if best_lv in levels else 0
        bar_x = x_pos[lv_idx] + centred_offsets[j]
        lv_col_vals = sub.loc[sub[axis_col].astype(str) == best_lv,
                               col_j].dropna().to_numpy()
        mean_lv = float(np.mean(lv_col_vals)) if len(lv_col_vals) else best_val
        sd_lv   = float(np.std(lv_col_vals, ddof=0)) if len(lv_col_vals) > 1 else 0.0
        raw_annots.append(dict(
            bar_x=bar_x, bar_top=mean_lv + sd_lv,
            label=f"{best_id[:16]} {best_val:.3f}",
            color=PALETTE[key], j=j,
        ))

    # ── Assign stagger slots: any two annotations whose x is within
    # ~1 bar-width get different vertical slots so labels never overlap.
    # We sort by bar_x and process left→right, tracking active labels
    # (those whose x is still "close" to the next one).
    raw_annots.sort(key=lambda a: a["bar_x"])
    annotations = []
    # Estimate label width in DATA coords:
    # an 18-char label at fontsize 13 ≈ 90 px ≈ ~1.2 data units in a 4-level subplot.
    # Use a generous proximity = 2 * one full level step (x_pos delta = 1.0)
    label_proximity = max(1.2, bar_width * (n_metrics + 2))
    used_slots = []   # list of (x_right_edge, slot) of currently active labels
    for a in raw_annots:
        # Remove expired slots: any label whose right edge is left of (a["bar_x"] - tol)
        used_slots = [s for s in used_slots if s[0] > a["bar_x"] - label_proximity]
        taken = {s[1] for s in used_slots}
        slot = 0
        while slot in taken:
            slot += 1
        a["slot"] = slot
        used_slots.append((a["bar_x"] + label_proximity, slot))
        annotations.append(a)

    # Each label offset 12 pt above its bar top in PIXEL coordinates
    # (independent of data scale), plus slot*14 pt extra to stack labels
    # that land on the same bar without overlapping.
    for a in annotations:
        ax.plot(a["bar_x"], a["bar_top"], marker="*", markersize=14,
                color=a["color"], markeredgecolor="white",
                markeredgewidth=0.7, zorder=7)
        ax.annotate(a["label"],
                    xy=(a["bar_x"], a["bar_top"]),
                    xytext=(0, 12 + a["slot"] * 20),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=13, color=a["color"], fontweight="bold",
                    clip_on=False)
    # ── Axis formatting ──────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    rot = 25 if max((len(l) for l in levels), default=0) > 8 else 0
    ax.set_xticklabels([wrap_label(l, 15) for l in levels],
                        rotation=rot, ha="right" if rot else "center",
                        fontsize=16)
    ax.set_ylabel("Score", fontsize=17)
    ax.tick_params(axis="y", labelsize=15)

    return stats_out


# ═════════════════════════════════════════════════════════════════════════════
# Per-target figure builders
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# Deterministic "best model" selection
# ─────────────────────────────────────────────────────────────────────────────
# With thousands of models, many tie on a metric once it is rounded to a
# sensible precision. `idxmax()` then returns whichever row pandas happens to
# meet first, so the reported winner can change between runs or subsets for no
# scientific reason. TIE_TOLERANCE defines "effectively equal", and the cascade
# below resolves those ties on defensible grounds, in this order:
#
#   1. PARSIMONY OF INPUTS - fewest predictors available to the model. A model
#      that matches another using less of the care pathway is the better
#      result. Ranked by actual predictor count, so L4 (8 score inputs) <
#      L1 (on-scene) < L2 (+ED) < L3 (+in-hospital).
#   2. MODEL CLASS - interpretable first, AutoML last:
#        0 interpretable (logistic_l1, logistic_elasticnet)
#        1 standard ML   (xgboost, lightgbm, catboost, random_forest)
#        2 deep learning (tabnet, tabpfn, the FFNN variants)
#        3 AutoML        (flaml, tpot) - these search over pipelines, so an
#          equal score is less informative and far less reproducible.
#   3. PIPELINE SIMPLICITY - no augmentation before SMOTE/ADASYN, no
#      calibration before Platt/isotonic, cheap imputer before model-based.
#      Fewer moving parts for the same score.
#   4. GENERALISATION GAP - smallest |train - test| for the same metric, which
#      prefers the model least likely to be overfitting. (Train vs TEST, never
#      the holdout: the holdout must stay untouched by selection.)
#   5. LOWEST MODEL INDEX - a final deterministic fallback so the choice is
#      reproducible even if everything above ties.
TIE_TOLERANCE = 1e-4        # metric differences below this count as a tie

_PHASE_NPRED = {            # smaller = fewer predictors = preferred
    "all_baseline_inputs": 0,
    "On-scene": 1,
    "On-scene + ED arrival": 2,
    "On-scene + ED arrival + In-hospital": 3,
}
_FAMILY_TIER = {
    "logistic_l1": 0, "logistic_elasticnet": 0,
    "xgboost": 1, "lightgbm": 1, "catboost": 1, "random_forest": 1,
    "tabnet": 2, "tabpfn": 2,
    "doshi_ffnn": 2, "doshi_ffnn_icd": 2, "doshi_ffnn_icd_plus": 2,
    "flaml": 3, "tpot": 3,
}
_IMPUTER_COST = {"none": 0, "median_mode": 1, "knn": 2, "mice": 3,
                 "bagged_trees": 3, "missforest": 4, "gradient_boosting": 4}


def _model_index(mid) -> int:
    import re as _re
    m = _re.search(r"(\d{4,})$", str(mid))
    return int(m.group(1)) if m else 10**9


def _tiebreak_key(row, df, metric: str, subset: str):
    """Sort key applied only among models already tied on the headline metric."""
    phase = row.get("cfg_phase_cutoff")
    fam   = row.get("cfg_model_family")
    aug   = str(row.get("cfg_data_augmentation", "none")).lower()
    cal   = str(row.get("cfg_calibration", "none")).lower()
    imp   = str(row.get("cfg_imputer_method", "none")).lower()

    # 4. generalisation gap |train - test|, if both columns exist
    gap = np.inf
    try:
        # detect_metric_col only resolves test/holdout, so locate the TRAIN
        # column directly (aggregator writes "<metric>__train", plus a
        # "__<cohort>" suffix for cohort subsets).
        cands_tr = ([f"{metric}__train__{subset}", f"{metric}__train"]
                    if subset != "full" else [f"{metric}__train"])
        c_tr = next((c for c in cands_tr if c in df.columns), None)
        c_te = detect_metric_col(df, metric, "test", subset)
        if c_tr and c_te and pd.notna(row.get(c_tr)) and pd.notna(row.get(c_te)):
            gap = abs(float(row[c_tr]) - float(row[c_te]))
    except Exception:
        pass

    return (
        _PHASE_NPRED.get(phase, 99),                       # 1 fewest predictors
        _FAMILY_TIER.get(fam, 5),                          # 2 interpretable > AutoML
        (0 if aug in ("none", "nan", "null", "") else 1),  # 3 pipeline simplicity
        (0 if cal in ("none", "nan", "") else 1),
        _IMPUTER_COST.get(imp, 5),
        gap,                                               # 4 least overfitting
        _model_index(row.get("model_id")),                 # 5 deterministic
    )


def pick_best(df: pd.DataFrame, col: str, ascending: bool = False,
              metric: str | None = None, subset: str = "full"):
    """Index of the best row in `col`, ties broken by `_tiebreak_key`.

    `ascending=True` for metrics where lower is better (Brier, FPR, ...).
    Returns None when there is nothing to choose from.
    """
    sub = df[df[col].notna()]
    if sub.empty:
        return None
    best_val = sub[col].min() if ascending else sub[col].max()
    tied = sub[np.isclose(sub[col].astype(float), float(best_val),
                          atol=TIE_TOLERANCE, rtol=0)]
    if len(tied) <= 1:
        return tied.index[0] if len(tied) == 1 else (
            sub[col].idxmin() if ascending else sub[col].idxmax())
    met = metric or col.split("__")[0]
    order = sorted(tied.index, key=lambda i: _tiebreak_key(tied.loc[i], df, met, subset))
    return order[0]


# ═════════════════════════════════════════════════════════════════════════════
# Selected models A / B / C  (best per phase cutoff, by AUPRC on the full test)
# ─────────────────────────────────────────────────────────────────────────────
PHASE_ABC = [("On-scene", "Model L1", "on-scene"),
             ("On-scene + ED arrival", "Model L2", "+ ED arrival"),
             ("On-scene + ED arrival + In-hospital", "Model L3", "full")]

# ── (1) Decode the numeric subgroup levels to clinical labels ────────────────
# MECHANISM / INTENT / TRAUMATYPE are stored as integer codes in the NTDB PUF;
# the meanings below are taken verbatim from the "PUF Dictionary by Admission
# Year" workbook. Age-group bands follow the strata used in the pipeline.
_INTENT_LABELS = {1:"Unintentional",2:"Self-inflicted",3:"Assault",
                  4:"Undetermined",5:"Other"}
_MECHANISM_LABELS = {1:"Cut/pierce",2:"Drowning",3:"Fall",4:"Fire/flame",
    5:"Hot object",6:"Firearm",7:"Machinery",8:"MVT occupant",
    9:"MVT motorcyclist",10:"MVT cyclist",11:"MVT pedestrian",
    12:"MVT unspecified",13:"MVT other",14:"Cyclist, other",
    15:"Pedestrian, other",16:"Transport, other",17:"Bites/stings",
    18:"Environmental",19:"Overexertion",20:"Poisoning",21:"Struck by/against",
    22:"Suffocation",23:"Other specified",24:"Other, NEC",25:"Unspecified",
    26:"Adverse effects, care",27:"Adverse effects, drugs"}
_TRAUMATYPE_LABELS = {1:"Blunt",2:"Penetrating",3:"Burn",4:"Other",
                      9:"Activity code"}
_AGE_LABELS = {"pediatric":"Pediatric\n(<18 y)","young_adult":"Young adult\n(18-39 y)",
               "middle_aged":"Middle-aged\n(40-64 y)","early_elderly":"Early-elderly\n(65-74 y)",
               "elderly":"Elderly\n(75+ y)"}


# The subgroup levels observed in practice are 0-INDEXED label encodings, not
# the 1-based NTDB PUF codes: `intent` appears as {0..4} against a dictionary of
# {1..5}, and `mechanism` appears as {0..3} while the trainer reports mechanism
# as perfectly correlated with TRAUMATYPE (1=Blunt..4=Other). We therefore shift
# by one. Any code outside the expected range is labelled explicitly rather than
# silently guessed, so a mis-assumption is visible in the figure.
_INTENT_0IDX = {i: _INTENT_LABELS[i + 1] for i in range(5)}
_MECHANISM_0IDX = {i: _TRAUMATYPE_LABELS[i + 1] for i in range(4)}


def decode_level(axis: str, level) -> str:
    """Map a raw subgroup level to a human-readable clinical label."""
    s = str(level)
    if axis == "age_group":
        return _AGE_LABELS.get(s, s.replace("_", " ").capitalize())
    table = {"mechanism": _MECHANISM_0IDX, "intent": _INTENT_0IDX,
             "TRAUMATYPE": _TRAUMATYPE_LABELS,
             "trauma_type": _TRAUMATYPE_LABELS}.get(axis)
    if table is None:
        return s
    try:
        k = int(float(s))
    except (TypeError, ValueError):
        return s
    return table.get(k, f"Code {k}\n(unmapped)")


def select_abc(df: pd.DataFrame, target: str):
    """Best model at each of L1/L2/L3, ranked by AUPRC on the full test set.

    Returns [(label, short, phase, model_id, row), ...] in phase order. The
    selection metric is fixed to AUPRC on the full test partition, per the
    study protocol, and ties are resolved by the standard cascade.
    """
    # The 4-class band targets have NO AUPRC/AUROC columns -- their metrics are
    # balanced_accuracy / f1_macro / AUROC_ovr. Asking for "AUPRC_ovr" and then
    # falling back to "AUROC" resolved to None for both, so select_abc returned
    # nothing and these targets silently produced no radar and "no models
    # available" in the summary. Try the right names, in order of preference.
    cands = (["balanced_accuracy", "f1_macro", "AUROC_ovr"]
             if target in BAND_TARGETS else ["AUPRC", "AUROC"])
    metric, col = None, None
    for m in cands:
        c = detect_metric_col(df, m, "test", "full")
        if c is not None:
            metric, col = m, c
            break
    if col is None or "cfg_phase_cutoff" not in df.columns:
        print(f"    [select_abc] no usable metric among {cands} for {target}")
        return []
    out = []
    for phase, label, short in PHASE_ABC:
        sub = df[(df["cfg_phase_cutoff"] == phase) & df[col].notna()]
        if sub.empty:
            continue
        i = pick_best(sub, col, ascending=False, metric=metric, subset="full")
        if i is None:
            continue
        out.append((label, short, phase, str(sub.loc[i, "model_id"]), sub.loc[i]))
    return out


def _subgroup_table(metrics_root: Path, model_id: str, axis: str,
                    partition: str = "test") -> Optional[pd.DataFrame]:
    """Read metrics/<model_id>/subgroups[_holdout]/subgroups_by_<axis>.csv."""
    d = "subgroups" if partition == "test" else "subgroups_holdout"
    f = metrics_root / model_id / d / f"subgroups_by_{axis}.csv"
    if not f.is_file():
        return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None


def make_models_vs_baselines_figure(df, baselines, target, metrics_root, out_path):
    """Fig: selected models A/B/C against the legacy scores, baseline-complete."""
    abc = select_abc(df, target)
    if not abc:
        print("    [skip] models-vs-baselines: no phase models"); return
    names = [a[0] for a in abc]
    bl = [b for b in BASELINE_NAMES if baselines.get(b)]
    if target in ("iss_band_binary", "niss_band_binary"):
        bl = []

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
        labels, vt, vh, cols = [], [], [], []
        pal = ["#2b6cb0", "#c0392b", "#1e8449"]
        for k, (label, short, phase, mid, row) in enumerate(abc):
            ct = detect_metric_col(df, metric, "test", "baseline_complete")
            ch = detect_metric_col(df, metric, "holdout", "baseline_complete")
            if ct is None:
                continue
            labels.append(label); cols.append(pal[k % 3])
            vt.append(float(row[ct]) if pd.notna(row.get(ct)) else np.nan)
            vh.append(float(row[ch]) if ch and pd.notna(row.get(ch)) else np.nan)
        nmod = len(labels)
        for b in bl:
            labels.append(b); cols.append(PALETTE.get(b, "#888"))
            vt.append((baselines.get(b) or {}).get((metric, "test", "baseline_complete"), np.nan))
            vh.append((baselines.get(b) or {}).get((metric, "holdout", "baseline_complete"), np.nan))
        x = np.arange(len(labels)); w = 0.4
        ax.bar(x - w/2, vt, w, color=cols, edgecolor="black", linewidth=.6, label="Test (2019-22)")
        ax.bar(x + w/2, vh, w, color=cols, edgecolor="black", linewidth=.6,
               alpha=.55, hatch="//", label="Holdout (2024)")
        # Zoom to the data band. Starting the axis at 0 compressed every bar
        # into the top fifth of the panel and made the value labels collide;
        # a padded min-to-max range makes the differences legible.
        finite = [v for v in vt + vh if np.isfinite(v)]
        lo = max(0.0, min(finite) - (max(finite) - min(finite)) * 0.35) if finite else 0
        hi = min(1.0, max(finite) + (max(finite) - min(finite)) * 0.28) if finite else 1
        if hi - lo < 1e-6:
            lo, hi = max(0, lo - .05), min(1, hi + .05)
        ax.set_ylim(lo, hi)
        span = hi - lo
        for xi, (a, b2) in enumerate(zip(vt, vh)):
            if np.isfinite(a):
                ax.text(xi - w/2, a + span * .018, f"{a:.3f}", ha="center",
                        fontsize=11, fontweight="bold", rotation=90, va="bottom")
            if np.isfinite(b2):
                ax.text(xi + w/2, b2 + span * .018, f"{b2:.3f}", ha="center",
                        fontsize=10, color="#555", rotation=90, va="bottom")
        if nmod and len(labels) > nmod:
            ax.axvline(nmod - .5, color="#888", linestyle="--", linewidth=1.2)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylabel(metric, fontweight="bold", fontsize=13)
        ax.set_title(f"{metric} (baseline-complete cohort)", fontweight="bold",
                     fontsize=14, pad=10)
        ax.grid(axis="y", color="#ececec", linewidth=.6)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[:2], l[:2], loc="upper right", frameon=True, fontsize=11)
    fig.suptitle("Selected ML models versus legacy severity scores",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.20, wspace=.20)
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"    OK wrote {out_path.name}")


SUBGROUP_RADAR_AXES = [("age_group", "Age group"), ("__admission_year", "Admission year"),
                       ("mechanism", "Mechanism"), ("intent", "Intent"),
                       ("race", "Race / ethnicity")]


def _subgroup_metric(target: str) -> str:
    """Balanced accuracy for the 4-class band targets, AUROC otherwise."""
    return "balanced_accuracy" if target in BAND_TARGETS else "AUROC"


def make_subgroup_radar_figure(df, target, metrics_root, out_path):
    """Fig: per-subgroup performance for models L1/L2/L3, radar + sex bar."""
    abc = select_abc(df, target)
    if not abc:
        print("    [skip] subgroup radar: no phase models"); return
    metric = _subgroup_metric(target)
    pal = ["#2b6cb0", "#c0392b", "#1e8449"]

    panels = []
    missing = set()
    for axis, nice in SUBGROUP_RADAR_AXES:
        axis_key = axis
        series = []
        for k, (label, short, phase, mid, row) in enumerate(abc):
            tb = _subgroup_table(metrics_root, mid, axis, "test")
            if tb is None:
                missing.add((label, mid, axis, "no subgroups_by_*.csv"))
                continue
            if metric not in tb.columns:
                missing.add((label, mid, axis, f"no '{metric}' column"))
                continue
            tb = tb[~tb["subgroup_level"].astype(str).str.startswith("__")]
            if tb.empty:
                continue
            series.append((label, pal[k % 3],
                           dict(zip(tb["subgroup_level"].astype(str), tb[metric]))))
        if series:
            panels.append((nice, series, axis))
    sex = []
    for k, (label, short, phase, mid, row) in enumerate(abc):
        tb = _subgroup_table(metrics_root, mid, "gender", "test")
        if tb is not None and metric in tb.columns:
            tb = tb[~tb["subgroup_level"].astype(str).str.startswith("__")]
            sex.append((label, pal[k % 3],
                        dict(zip(tb["subgroup_level"].astype(str), tb[metric]))))
    if missing:
        for label, mid, axis, why in sorted(missing)[:6]:
            print(f"    [radar] {label} ({mid}) axis={axis}: {why}")
    if not panels and not sex:
        print("    [skip] subgroup radar: no subgroup CSVs found"); return

    n = len(panels) + (1 if sex else 0)
    ncol, nrow = 2, int(np.ceil(n / 2))
    fig = plt.figure(figsize=(14.5, 5.6 * nrow))
    for i, (nice, series, axis_key) in enumerate(panels):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="polar")
        levels = sorted({k for _, _, d in series for k in d})
        ang = np.linspace(0, 2*np.pi, len(levels), endpoint=False).tolist()
        ang += ang[:1]
        for label, c, d in series:
            v = [float(d.get(l, np.nan)) for l in levels]; v += v[:1]
            ax.plot(ang, v, "-o", color=c, linewidth=1.8, markersize=4, label=label)
            ax.fill(ang, v, color=c, alpha=.06)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels([decode_level(axis_key, l) for l in levels], fontsize=14)
        vals = [v for _, _, d in series for v in d.values() if np.isfinite(v)]
        lo = max(0.0, (min(vals) - 0.04) if vals else 0.78)
        hi = min(1.0, (max(vals) + 0.02) if vals else 1.0)
        ax.set_ylim(lo, hi)
        ticks = np.round(np.linspace(lo, hi, 4), 2)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:.2f}" for v in ticks], fontsize=12, color="#666")
        ax.set_title(nice, fontweight="bold", fontsize=18, pad=16)
    if sex:
        ax = fig.add_subplot(nrow, ncol, len(panels) + 1)
        levels = sorted({k for _, _, d in sex for k in d})
        x = np.arange(len(levels)); w = 0.26
        for k, (label, c, d) in enumerate(sex):
            v = [float(d.get(l, np.nan)) for l in levels]
            ax.bar(x + (k - 1) * w, v, w, color=c, label=label, edgecolor="black", linewidth=.5)
            for xi, val in zip(x, v):
                if np.isfinite(val):
                    ax.text(xi + (k-1)*w, val + .002, f"{val:.3f}", ha="center", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=15)
        sv = [vv for _,_,d in sex for vv in d.values() if np.isfinite(vv)]
        ax.set_ylim(max(0, min(sv) - .04) if sv else .8, (max(sv) + .015) if sv else .95)
        ax.set_ylabel(f"{metric} (test)", fontweight="bold", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.set_title("Sex", fontweight="bold", fontsize=18, pad=12)
        ax.grid(axis="y", color="#ececec", linewidth=.6)
    hs = [plt.Line2D([0],[0], color=pal[i%3], marker="o", linewidth=2,
                     label=abc[i][0]) for i in range(len(abc))]
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), frameon=False,
               fontsize=17, bbox_to_anchor=(.5, .958))
    fig.suptitle(f"{TARGET_PRETTY.get(target,target)} {metric} across patient subgroups (test partition)",
                 fontsize=20, fontweight="bold", y=0.992)
    fig.subplots_adjust(top=.915, bottom=.035, hspace=.34, wspace=.10,
                        left=.055, right=.965)
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"    OK wrote {out_path.name}")


def make_temporal_stability_figure(df, target, metrics_root, out_path):
    """Fig: discrimination by admission year for models A/B/C."""
    abc = select_abc(df, target)
    if not abc:
        print("    [skip] temporal stability: no phase models"); return
    metric = _subgroup_metric(target)
    pal = ["#2b6cb0", "#c0392b", "#1e8449"]
    series = []
    for k, (label, short, phase, mid, row) in enumerate(abc):
        tb = _subgroup_table(metrics_root, mid, "__admission_year", "test")
        if tb is None or metric not in tb.columns:
            continue
        tb = tb[~tb["subgroup_level"].astype(str).str.startswith("__")]
        series.append((label, pal[k % 3],
                       dict(zip(tb["subgroup_level"].astype(str), tb[metric]))))
    if not series:
        print("    [skip] temporal stability: no admission-year subgroup CSVs"); return

    years = sorted({y for _, _, d in series for y in d})
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    x = np.arange(len(years)); w = 0.26
    for k, (label, c, d) in enumerate(series):
        v = [float(d.get(y, np.nan)) for y in years]
        ax.bar(x + (k - 1) * w, v, w, color=c, label=label, edgecolor="black", linewidth=.5)
        for xi, val in zip(x, v):
            if np.isfinite(val):
                ax.text(xi + (k-1)*w, val + .0012, f"{val:.3f}", ha="center", fontsize=8.5)
    # Set the y-range FIRST: the period labels are positioned relative to the
    # top of the axes, so placing them before set_ylim put them off-canvas.
    allv = [v for _,_,d in series for v in d.values() if np.isfinite(v)]
    if allv:
        ax.set_ylim(min(allv) - .02, max(allv) + .020)
    ylo, yhi = ax.get_ylim()
    spans = [("pre-pandemic", {"2019"}, "#e8f5e9"), ("pandemic era", {"2020","2021"}, "#fdeee7"),
             ("recovery", {"2022"}, "#e8eef9")]
    for name, ys, col in spans:
        idx = [i for i, y in enumerate(years) if y in ys]
        if idx:
            ax.axvspan(min(idx)-.5, max(idx)+.5, color=col, zorder=0)
            ax.text((min(idx)+max(idx))/2, yhi - (yhi-ylo)*0.04, name, ha="center",
                    va="top", fontsize=10.5, style="italic", color="#666")
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel(f"{metric} (test)", fontweight="bold")
    ax.set_title(f"Temporal stability of {TARGET_PRETTY.get(target,target).lower()} "
                 f"discrimination across admission years", fontweight="bold", fontsize=14, pad=22)
    ax.legend(fontsize=10, frameon=True, loc="lower center", ncol=3)
    ax.grid(axis="y", color="#ececec", linewidth=.6)
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.10)
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"    OK wrote {out_path.name}")


def write_best_by_phase_txt(per_target: dict, out_path: Path):
    """Write the best model per phase cutoff (L1/L2/L3) for every target.

    "Best" is the model maximising AUPRC on the FULL test partition, or
    balanced accuracy for the multiclass band targets, with ties resolved by
    the standard deterministic cascade. These are exactly the models plotted
    as Model L1 / L2 / L3 in the figures.
    """
    lines = ["BEST MODEL PER PHASE CUTOFF AND TARGET",
             "=" * 78,
             "Selection metric: AUPRC on the full test partition",
             "                  (balanced accuracy for the 4-class band targets).",
             "Ties broken by: fewest predictors > interpretable family >",
             "                simpler pipeline > smallest train-test gap > lowest index.",
             "=" * 78, ""]
    for target, (abc, sel_metric, rows) in per_target.items():
        lines.append(f"### {TARGET_PRETTY.get(target, target)}  [{target}]")
        lines.append(f"    selection metric: {sel_metric} (test, full cohort)")
        if not abc:
            lines.append("    (no models available)"); lines.append(""); continue
        lines.append(f"    {'phase':<6}{'model_id':<24}{'value':>9}   configuration")
        lines.append("    " + "-" * 92)
        for (label, short, phase, mid, row), val in zip(abc, rows):
            cfg = " | ".join(str(row.get(c, "?")) for c in
                             ("cfg_model_family", "cfg_imputer_method",
                              "cfg_calibration", "cfg_data_augmentation"))
            lab = label.replace("Model ", "")
            v = f"{val:.4f}" if val is not None and np.isfinite(val) else "n/a"
            lines.append(f"    {lab:<6}{mid:<24}{v:>9}   {cfg}")
            lines.append(f"    {'':<6}{'':<24}{'':>9}   phase = {phase}")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"    OK wrote {out_path.name}")


def sig_stars(pv: float) -> str:
    """Significance markers, matching the convention used in the main figures."""
    if pv is None or not np.isfinite(pv):
        return ""
    return "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"


def make_phase_progression_figure(df: pd.DataFrame, baselines: dict,
                                  subset: str, target: str, out_path: Path):
    """Performance progression across the phase-of-care cutoffs L1 -> L2 -> L3.

    L4 (`all_baseline_inputs`) is EXCLUDED by design: it is not a later phase
    of care but a restricted feature set matching the clinical scores' inputs,
    so it is not nested in the L1->L2->L3 progression and plotting it on the
    same axis invites a false reading.

    Per phase we show the grid mean +/- SD band, the best model on test and on
    holdout (labelled with its model_id), and Mann-Whitney tests between
    consecutive phases. Clinical baselines are drawn for mortality only and
    ONLY in their raw form -- the platt / isotonic / threshold_tuned variants
    are deliberately omitted (they turned the reference figure into 24
    unreadable overlapping lines).
    """
    PHASES = ["On-scene",
              "On-scene + ED arrival",
              "On-scene + ED arrival + In-hospital"]
    SHORT = ["L1\nOn-scene", "L2\n+ ED arrival", "L3\n+ In-hospital"]

    if "cfg_phase_cutoff" not in df.columns:
        print("    [skip] phase progression: no cfg_phase_cutoff column"); return
    present = [(i, ph) for i, ph in enumerate(PHASES)
               if (df["cfg_phase_cutoff"] == ph).any()]
    if len(present) < 2:
        print(f"    [skip] phase progression: only {len(present)} phase(s)"); return

    # The 4-class band targets have no AUROC/AUPRC columns -- their metrics are
    # balanced accuracy and macro-F1. Using the binary names silently produced
    # no columns, which is why no progression figure appeared for them.
    if target in BAND_TARGETS:
        metric_specs = [("balanced_accuracy", "Balanced accuracy"),
                        ("f1_macro",          "Macro-F1")]
    else:
        metric_specs = [("AUROC", "AUROC"), ("AUPRC", "AUPRC")]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.4))

    for ax, (metric, metric_lbl) in zip(axes, metric_specs):
        col_t = detect_metric_col(df, metric, "test", subset)
        col_h = detect_metric_col(df, metric, "holdout", subset)
        if col_t is None:
            ax.text(0.5, 0.5, f"no {metric_lbl} column for subset '{subset}'",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="#888")
            ax.set_title(f"{metric_lbl} across care phases", fontweight="bold")
            continue

        xs, mus, sds, best_t, best_h, best_id, groups = [], [], [], [], [], [], []
        mus_h, sds_h = [], []          # holdout grid mean / SD
        for xi, (_, ph) in enumerate(present):
            sub = df[df["cfg_phase_cutoff"] == ph]
            vals = sub[col_t].dropna()
            groups.append(vals.to_numpy())
            if not len(vals):
                continue
            xs.append(xi); mus.append(float(vals.mean())); sds.append(float(vals.std(ddof=1) or 0))
            vh = sub[col_h].dropna() if col_h else pd.Series(dtype=float)
            mus_h.append(float(vh.mean()) if len(vh) else np.nan)
            sds_h.append(float(vh.std(ddof=1) or 0) if len(vh) else np.nan)
            imax = pick_best(sub, col_t, ascending=False, metric=metric, subset=subset)
            best_t.append(float(sub.loc[imax, col_t]))
            best_id.append(str(sub.loc[imax, "model_id"]) if "model_id" in sub.columns else "")
            best_h.append(float(sub.loc[imax, col_h]) if col_h and pd.notna(sub.loc[imax, col_h]) else np.nan)

        mus, sds = np.array(mus), np.array(sds)
        mus_h, sds_h = np.array(mus_h), np.array(sds_h)
        ax.fill_between(xs, mus - sds, mus + sds, color="#cfe0f3", alpha=0.55,
                        label="grid mean +/- SD (test)", zorder=1)
        ax.plot(xs, mus, "-o", color="#7fb0e0", markersize=6,
                label="grid mean (test)", zorder=2)
        if np.isfinite(mus_h).any():
            ax.fill_between(xs, mus_h - sds_h, mus_h + sds_h, color="#f6cfcf",
                            alpha=0.45, label="grid mean +/- SD (holdout)", zorder=1)
            ax.plot(xs, mus_h, "--o", color="#d98c8c", markersize=5,
                    label="grid mean (holdout)", zorder=2)
        ax.plot(xs, best_t, "-o", color=PALETTE.get("auroc_test", "#1f4e79"),
                linewidth=2.4, markersize=9, label="best model (test)", zorder=4)
        if col_h:
            ax.plot(xs, best_h, "--s", color=PALETTE.get("auprc_test", "#b2182b"),
                    linewidth=2.0, markersize=8, label="best model (holdout)", zorder=4)
        for xi, yv, mid in zip(xs, best_t, best_id):
            ax.annotate(f"{mid}\n{yv:.3f}", (xi, yv), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=10.5,
                        fontweight="bold", color=PALETTE.get("auroc_test", "#1f4e79"))
        # Holdout value beside its marker (below the point, so it cannot
        # collide with the test label above it).
        for xi, yv in zip(xs, best_h):
            if yv is not None and np.isfinite(yv):
                ax.annotate(f"{yv:.3f}", (xi, yv), textcoords="offset points",
                            xytext=(0, -20), ha="center", fontsize=10.5,
                            fontweight="bold",
                            color=PALETTE.get("auprc_test", "#b2182b"),
                            bbox=dict(facecolor="white", edgecolor="none",
                                      alpha=.7, pad=1.2))

        # consecutive-phase Mann-Whitney tests
        bits = []
        for a in range(len(groups) - 1):
            ga, gb = groups[a], groups[a + 1]
            if len(ga) > 1 and len(gb) > 1:
                try:
                    _, pv = stats.mannwhitneyu(ga, gb, alternative="two-sided")
                    bits.append(f"L{a+1}->L{a+2} MWU p={pv:.2e} {sig_stars(pv)}")
                except Exception:
                    pass
        subtitle = "; ".join(bits) if bits else "insufficient data for tests"

        # raw clinical baselines (mortality only), with de-cluttered labels
        if target == "mortality":
            drawn = []
            for bn in BASELINE_NAMES:
                v = (baselines.get(bn) or {}).get((metric, "test", subset))
                if v is None or not np.isfinite(v):
                    continue
                vh = (baselines.get(bn) or {}).get((metric, "holdout", subset))
                drawn.append((float(v), bn, float(vh) if vh is not None
                              and np.isfinite(vh) else None))
            drawn.sort()
            ylo, yhi = ax.get_ylim()
            min_gap = (yhi - ylo) * 0.035
            last = -np.inf
            for v, bn, vh in drawn:
                c = PALETTE.get(bn, "#999")
                ax.axhline(v, color=c, linewidth=1.3, alpha=0.85, zorder=0)
                if vh is not None:      # holdout baseline, dotted
                    ax.axhline(vh, color=c, linewidth=1.1, alpha=0.75,
                               linestyle=":", zorder=0)
                # Sit the label just ABOVE its line: with va="center" the line
                # ran straight through the text. min_gap keeps near-identical
                # baselines (e.g. ISS 0.817 vs NISS 0.819) from overprinting.
                ytxt = max(v + min_gap * 0.30, last + min_gap)
                last = ytxt
                lbl = (f"{bn}  test {v:.3f} / hold {vh:.3f}" if vh is not None
                       else f"{bn} {v:.3f}")
                ax.text(-0.42, ytxt, lbl, va="bottom", ha="left",
                        fontsize=9.5, fontweight="bold", color=c, zorder=5,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.65, pad=0.8))

        # Headroom for the "model_id / value" annotations above each marker,
        # otherwise they climb into the subplot subtitle.
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.16)

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([SHORT[i] for i, _ in present], fontsize=11)
        ax.set_xlim(-0.45, len(present) - 0.5)
        ax.set_ylabel(metric_lbl, fontsize=12, fontweight="bold")
        ax.set_title(f"{metric_lbl} across care phases\n{subtitle}",
                     fontweight="bold", fontsize=12, pad=12)
        ax.grid(axis="y", color="#ececec", linewidth=0.6)
        ax.legend(fontsize=9.5, frameon=True, loc="lower right", framealpha=0.9)

    pretty = TARGET_PRETTY.get(target, target)
    sub_lbl = "full subset" if subset == "full" else subset.replace("_", "-") + " subset"
    fig.suptitle(f"{pretty} - phase-cutoff progression - {sub_lbl}",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.78, bottom=0.13, wspace=0.26)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"    OK wrote {out_path.name}")


def make_figure(df: pd.DataFrame, baselines: dict, subset: str,
                  target: str, out_path: Path,
                  metrics_mode: str = "default") -> list[dict]:
    """Create one figure with 6 subplots (3 rows × 2 cols).

    metrics_mode: "default" | "tpr_fpr" | "recall_by_class"
    """
    fig, axes = plt.subplots(3, 2, figsize=(22, 31))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    # Drop config axes that have only one level (e.g. calibration is always
    # "none" for the 4-class band targets) -- a one-bar subplot with no
    # comparison to make is wasted space.
    usable_axes = []
    for axis_col, axis_label in CONFIG_AXES:
        try:
            nlev = df[axis_col].dropna().nunique() if axis_col in df.columns else 0
        except Exception:
            nlev = 0
        if nlev >= 2:
            usable_axes.append((axis_col, axis_label))
        else:
            print(f"    [skip] {axis_label}: only {nlev} level(s)")
    for ax in axes[len(usable_axes):]:
        ax.set_visible(False)

    stats_collected = []
    for ax, (axis_col, axis_label) in zip(axes, usable_axes):
        try:
            s = plot_subplot(ax, df, baselines, axis_col, axis_label,
                              subset, target, metrics_mode=metrics_mode)
            stats_collected.append(s)
        except Exception as e:
            ax.text(0.5, 0.5, f"error: {e}", ha="center",
                    transform=ax.transAxes, fontsize=8, color="red")

    # Suptitle
    # NOTE: use .get() with a derived fallback -- this dict previously omitted
    # the band-binary targets and raised KeyError on them. A new target must
    # never be able to crash the figures.
    target_pretty = TARGET_PRETTY.get(
        target, target.replace("_", " ").capitalize())
    subset_pretty = "full subset" if subset == "full" else \
                    "baseline-complete subset (where ISS/NISS/TRISS computable)"
    mode_suffix = {"default": "", "tpr_fpr": " — TPR & FPR",
                   "recall_by_class": " — per-class recall"}[metrics_mode]
    fig.suptitle(f"{target_pretty}{mode_suffix} — {subset_pretty}",
                  fontsize=28, fontweight="bold", y=0.985)

    # Legend — collect from one subplot
    handles, labels = axes[0].get_legend_handles_labels()
    # Add baseline indicators (only for mortality default/tpr_fpr)
    baseline_handles = []
    if target in BINARY_TARGETS and metrics_mode in ("default", "tpr_fpr"):
        for bn in BASELINE_NAMES:
            if bn in baselines:
                baseline_handles.append(
                    Line2D([0], [0], color=PALETTE[bn], lw=1.6,
                            label=f"{bn} baseline (test=solid, holdout=dashed)"))
    elif target not in ("iss_band", "niss_band"):
        bn = BAND_BASELINE.get(target)
        if bn and bn in baselines:
            baseline_handles.append(
                Line2D([0], [0], color=PALETTE[bn], lw=1.6,
                        label=f"{bn} baseline (test=solid, holdout=dashed)"))
    all_handles = handles + baseline_handles
    all_labels  = labels  + [h.get_label() for h in baseline_handles]
    if all_handles:
        # Two-row legend: split metric handles and baseline handles
        # (or just halve the handle list when no baseline_handles).
        # Three rows: with 6 clinical baselines the old 2-row layout ran wider
        # than the canvas and the first/last entries were clipped.
        n_handles = len(all_handles)
        ncol = max(1, -(-n_handles // 3))      # ceil(n/3) -> 3 rows
        fig.legend(all_handles, all_labels,
                    loc="lower center",
                    ncol=ncol,
                    bbox_to_anchor=(0.5, 0.012),
                    frameon=False, fontsize=16,
                    columnspacing=1.8, handlelength=2.2)

    # Layout: bigger gap between suptitle (top=0.94) and first row,
    # smaller hspace between rows since titles already include all stat text.
    # bbox_inches="tight" is intentionally NOT used — it would expand the canvas.
    fig.subplots_adjust(left=0.05, right=0.99,
                         top=0.875, bottom=0.075,
                         hspace=0.80, wspace=0.16)
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"    ✓ wrote {out_path.name}")
    return stats_collected


# ═════════════════════════════════════════════════════════════════════════════
# Top models per (metric × partition × subset) — same idea as meta_analysis.py
# ═════════════════════════════════════════════════════════════════════════════
def write_top_models(df: pd.DataFrame, baselines: dict, target: str,
                       out_path: Path) -> None:
    metrics_of_interest = (["AUROC", "AUPRC"] if target in BINARY_TARGETS
                           else ["balanced_accuracy"])
    # Calibration-aware overwrite: each model row is evaluated under its
    # actual calibration choice (not the uncalibrated proxy).
    layout = [(m, p, s) for m in metrics_of_interest
               for p in ("test", "holdout")
               for s in ("full", "baseline_complete")]
    df = _apply_calibrated_metrics(df, layout)
    rows = []
    for metric in metrics_of_interest:
        for partition in ("test", "holdout"):
            for subset in ("full", "baseline_complete"):
                col = detect_metric_col(df, metric, partition, subset)
                if col is None:
                    continue
                sub = df.dropna(subset=[col])
                if sub.empty:
                    continue
                idx = pick_best(sub, col, ascending=False)
                w = sub.loc[idx]
                row = {
                    "metric": metric, "partition": partition, "subset": subset,
                    "winner_id": w["model_id"],
                    "winner_value": float(w[col]),
                }
                for bn in baselines:
                    v = baselines[bn].get((metric, partition, subset))
                    if v is not None:
                        row[f"delta_vs_{bn}"] = row["winner_value"] - v
                # winner's config
                for c in ("cfg_model_family", "cfg_phase_cutoff",
                           "cfg_imputer_method", "cfg_calibration",
                           "cfg_data_augmentation",
                           "cfg_missingness_threshold", "cfg_imputer_check"):
                    if c in w:
                        row[c.replace("cfg_", "")] = w[c]
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.4f")
    print(f"    ✓ wrote {out_path.name} ({len(rows)} rows)")


def write_stats_summary(stats_collected: list[list[dict]], target: str,
                          out_path: Path) -> None:
    flat = []
    for subset_stats in stats_collected:
        for s in subset_stats:
            if not s:
                continue
            row = {
                "subset":         s.get("subset"),
                "axis":           s.get("axis", "").replace("cfg_", ""),
                "primary_metric": s.get("primary_metric"),
                "n_levels":       s.get("n_levels"),
                "n_models":       s.get("n_models_total"),
                "normality_ok":   s.get("normality_ok"),
                "equal_var_ok":   s.get("equal_var_ok"),
                "omnibus_test":   s.get("omnibus_test"),
                "omnibus_p":      s.get("omnibus_p"),
                "posthoc_method": s.get("posthoc_method"),
                "posthoc_correction": s.get("posthoc_correction"),
                # NOTE: the key is "consistent_sig_pairs" (was read as
                # "sig_pairs", which never existed -> column always empty).
                "n_sig_pairs":    len(s.get("consistent_sig_pairs", [])),
                "sig_pairs":      "; ".join(
                                      f"{display_name(a)} > {display_name(b)}"
                                      for a, b in s.get("consistent_sig_pairs", [])),
                "n_posthoc_tests": len(s.get("posthoc_rows", [])),
            }
            flat.append(row)
    pd.DataFrame(flat).to_csv(out_path, index=False, float_format="%.4g")
    print(f"    ✓ wrote {out_path.name}")

    # Companion file with EVERY pairwise post-hoc test (all metrics, both
    # significant and not). stats_summary.csv only carries the consistent
    # winners; this is the complete record behind the figure captions.
    detail = []
    for subset_stats in stats_collected:
        for s in subset_stats:
            if not s:
                continue
            for r in s.get("posthoc_rows", []):
                detail.append({
                    "subset":     s.get("subset"),
                    "axis":       s.get("axis"),
                    "metric":     r["metric"],
                    "level_a":    display_name(r["level_a"]),
                    "level_b":    display_name(r["level_b"]),
                    "p_adj":      r["p_adj"],
                    "significant": r["significant"],
                    "omnibus_test": s.get("omnibus_test"),
                    "omnibus_p":    s.get("omnibus_p"),
                    "posthoc_correction": "Bonferroni",
                })
    if detail:
        dpath = out_path.with_name("posthoc_pairwise_tests.csv")
        pd.DataFrame(detail).to_csv(dpath, index=False, float_format="%.4g")
        print(f"    ✓ wrote {dpath.name} ({len(detail)} tests)")



# ═════════════════════════════════════════════════════════════════════════════
# Best-model CSV (one row per "best" winner, with full metric dict)
# ═════════════════════════════════════════════════════════════════════════════
def _all_metric_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Return the canonical list of metric columns for a target, in display order."""
    if target in BINARY_TARGETS:
        prefixes = ["AUROC", "AUPRC", "recall", "FPR"]
    elif target in BAND_TARGETS:
        prefixes = ["balanced_accuracy",
                     "recall__0", "recall__1", "recall__2", "recall__3"]
    else:
        prefixes = ["AUROC", "AUPRC"]

    cols = []
    for pref in prefixes:
        for part in ("random_test", "holdout_year", "holdout_external"):
            for ss in ("", "__baseline_complete"):
                col = f"{pref}__{part}{ss}"
                if col in df.columns:
                    cols.append(col)
    return cols


def _model_metrics_dict(row: pd.Series, target: str,
                          df_cols: list[str]) -> dict:
    """Pretty per-row dict of metric values keyed by short names like
    ``AUROC_test_full``, ``FPR_holdout_baseline_complete`` etc.
    Returns missing metrics as None (not NaN, so CSV cells stay readable)."""
    out: dict = {}

    def pretty_partition(p):
        return {"random_test": "test",
                "holdout_year": "holdout",
                "holdout_external": "holdout"}.get(p, p)

    if target in BINARY_TARGETS:
        prefixes = ["AUROC", "AUPRC", "recall", "FPR"]
    elif target in BAND_TARGETS:
        prefixes = ["balanced_accuracy",
                     "recall__0", "recall__1", "recall__2", "recall__3"]
    else:
        prefixes = ["AUROC", "AUPRC"]

    for pref in prefixes:
        for part in ("random_test", "holdout_year", "holdout_external"):
            for ss in ("", "__baseline_complete"):
                col = f"{pref}__{part}{ss}"
                if col not in df_cols:
                    continue
                v = row.get(col)
                if pd.isna(v):
                    continue
                # Build short key: AUROC_test_full, FPR_holdout_baseline_complete
                metric_short = pref.replace("__", "_")
                part_short   = pretty_partition(part)
                ss_short     = "baseline_complete" if ss else "full"
                key = f"{metric_short}_{part_short}_{ss_short}"
                out[key] = float(v)
    return out



def _apply_calibrated_metrics(df: pd.DataFrame,
                                metric_layouts: list) -> pd.DataFrame:
    """Return a copy of df where, for every (metric, part, subset) triple
    in metric_layouts, the base metric column is overwritten with its
    calibration-aware per-row Series.  Downstream code (idxmax, dropna,
    _model_metrics_dict) then works correctly without further changes.

    `metric_layouts` items can be 3- or 4-tuples; only the first three
    positions (metric, part, subset) are used."""
    out = df.copy()
    seen = set()
    for tup in metric_layouts:
        metric, part, subset = tup[0], tup[1], tup[2]
        if (metric, part, subset) in seen:
            continue
        seen.add((metric, part, subset))
        col = detect_metric_col(out, metric, part, subset)
        if col is None:
            continue
        out[col] = get_metric_calibrated(out, metric, part, subset)
    return out


def write_best_models_csv(df: pd.DataFrame, target: str,
                            out_path: Path) -> None:
    """Write a CSV with one row per "best model" winner across the figures.

    Columns:
      model_id : str
      figures  : "; "-joined list of (subplot, partition, subset) where this
                 model is the best
      best_metric_value : the metric value for the FIRST appearance
      metrics  : JSON-serialised dict of ALL its metric values
                 (every metric × partition × subset present in the data)
    """
    import json as _json

    if target in BINARY_TARGETS:
        metric_layouts = [
            # (metric, partition, subset, ascending?)
            ("AUROC", "test",    "full",              False),
            ("AUROC", "holdout", "full",              False),
            ("AUPRC", "test",    "full",              False),
            ("AUPRC", "holdout", "full",              False),
            ("AUROC", "test",    "baseline_complete", False),
            ("AUROC", "holdout", "baseline_complete", False),
            ("AUPRC", "test",    "baseline_complete", False),
            ("AUPRC", "holdout", "baseline_complete", False),
            ("recall","test",    "full",              False),
            ("recall","holdout", "full",              False),
            ("FPR",   "test",    "full",              True),
            ("FPR",   "holdout", "full",              True),
            ("recall","test",    "baseline_complete", False),
            ("recall","holdout", "baseline_complete", False),
            ("FPR",   "test",    "baseline_complete", True),
            ("FPR",   "holdout", "baseline_complete", True),
        ]
    elif target in BAND_TARGETS:
        metric_layouts = [
            ("balanced_accuracy", "test",    "full", False),
            ("balanced_accuracy", "holdout", "full", False),
        ]
        # per-class recalls
        for cls in range(4):
            metric_layouts.append((f"recall__{cls}", "test",    "full", False))
            metric_layouts.append((f"recall__{cls}", "holdout", "full", False))
    else:
        metric_layouts = []

    # Calibration-aware: overwrite metric base columns with the per-row
    # calibrated values before searching for winners.  After this, both the
    # idxmax/idxmin lookup AND the metrics_dict populated from `mrow` will
    # reflect each model's ACTUAL calibration choice.
    df = _apply_calibrated_metrics(df, metric_layouts)

    # winner_id -> list[(figure_label, value)]
    winners: dict[str, list] = {}
    for metric, part, subset, ascending in metric_layouts:
        col = detect_metric_col(df, metric, part, subset)
        if col is None:
            continue
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        idx = pick_best(sub, col, ascending=ascending)
        wid = str(sub.loc[idx, "model_id"])
        val = float(sub.loc[idx, col])
        fig_label = f"{metric}_{part}_{subset}"
        winners.setdefault(wid, []).append((fig_label, val))

    # Build rows.  Each best-model gets its own row.
    df_cols = list(df.columns)
    rows = []
    for wid, appearances in winners.items():
        model_row = df[df["model_id"].astype(str) == wid]
        if model_row.empty:
            continue
        mrow = model_row.iloc[0]
        figures = "; ".join(f for f, _ in appearances)
        values  = "; ".join(f"{f}={v:.4f}" for f, v in appearances)
        metrics_dict = _model_metrics_dict(mrow, target, df_cols)
        rows.append({
            "model_id": wid,
            "figures":  figures,
            "values_at_those_figures": values,
            "n_figures_won": len(appearances),
            "all_metrics_json": _json.dumps(metrics_dict),
            **metrics_dict,
            # Also include cfg_* fields so the user can see the setting
            **{c: mrow.get(c) for c in mrow.index
                if c.startswith("cfg_") and not pd.isna(mrow.get(c))},
        })

    # Sort by number of figures won (most-frequent winners first)
    rows.sort(key=lambda r: -r["n_figures_won"])
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.4f")
    print(f"    ✓ wrote {out_path.name} ({len(rows)} unique winners)")


def write_phase_filtered_best_models_csv(df: pd.DataFrame, target: str,
                                            out_path: Path) -> None:
    """For mortality only: best models conditioned on cfg_phase_cutoff in
    {"On-scene", "On-scene + ED arrival"}.  One row per (phase, metric) winner.
    """
    if target != "mortality" or "cfg_phase_cutoff" not in df.columns:
        return
    PHASE_TARGETS = ["On-scene", "On-scene + ED arrival"]

    metric_layouts = [
        ("AUROC", "test",    "full",              False),
        ("AUROC", "holdout", "full",              False),
        ("AUPRC", "test",    "full",              False),
        ("AUPRC", "holdout", "full",              False),
        ("AUROC", "test",    "baseline_complete", False),
        ("AUROC", "holdout", "baseline_complete", False),
        ("AUPRC", "test",    "baseline_complete", False),
        ("AUPRC", "holdout", "baseline_complete", False),
        ("recall","test",    "full",              False),
        ("recall","holdout", "full",              False),
        ("FPR",   "test",    "full",              True),
        ("FPR",   "holdout", "full",              True),
        ("recall","test",    "baseline_complete", False),
        ("recall","holdout", "baseline_complete", False),
        ("FPR",   "test",    "baseline_complete", True),
        ("FPR",   "holdout", "baseline_complete", True),
    ]

    # Calibration-aware overwrite before the per-phase winner search.
    df = _apply_calibrated_metrics(df, metric_layouts)
    df_cols = list(df.columns)
    rows = []
    for phase in PHASE_TARGETS:
        phase_df = df[df["cfg_phase_cutoff"].astype(str) == phase]
        if phase_df.empty:
            continue
        # Find the winner for each (metric, partition, subset)
        winners: dict[str, list] = {}
        for metric, part, subset, ascending in metric_layouts:
            col = detect_metric_col(phase_df, metric, part, subset)
            if col is None:
                continue
            sub = phase_df.dropna(subset=[col])
            if sub.empty:
                continue
            idx = pick_best(sub, col, ascending=ascending)
            wid = str(sub.loc[idx, "model_id"])
            val = float(sub.loc[idx, col])
            fig_label = f"{metric}_{part}_{subset}"
            winners.setdefault(wid, []).append((fig_label, val))

        for wid, appearances in winners.items():
            model_row = phase_df[phase_df["model_id"].astype(str) == wid]
            if model_row.empty:
                continue
            mrow = model_row.iloc[0]
            figures = "; ".join(f for f, _ in appearances)
            values  = "; ".join(f"{f}={v:.4f}" for f, v in appearances)
            metrics_dict = _model_metrics_dict(mrow, target, df_cols)
            rows.append({
                "phase":   phase,
                "model_id": wid,
                "figures": figures,
                "values_at_those_figures": values,
                "n_figures_won": len(appearances),
                **metrics_dict,
                **{c: mrow.get(c) for c in mrow.index
                    if c.startswith("cfg_") and not pd.isna(mrow.get(c))},
            })

    # Sort by phase then by n_figures_won
    rows.sort(key=lambda r: (r["phase"], -r["n_figures_won"]))
    pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.4f")
    print(f"    ✓ wrote {out_path.name} ({len(rows)} rows)")


def write_audit_log(df: pd.DataFrame, target: str, out_path: Path) -> None:
    """Audit: for each axis (cfg_*) and each metric (test/holdout/full/bc),
    log how many non-NaN samples each level has — verifies the stat tests
    see the full data with no group dropped.
    """
    audit_rows = []
    axes_cfg = [
        ("cfg_model_family",          "Model family"),
        ("cfg_calibration",           "Calibration method"),
        ("cfg_imputer_method",        "Imputation strategy"),
        ("cfg_phase_cutoff",          "Phase cutoff"),
        ("cfg_missingness_threshold", "Missingness threshold"),
        ("cfg_data_augmentation",     "Data augmentation"),
    ]
    if target in BINARY_TARGETS:
        metric_grid = [("AUROC", "test"), ("AUROC", "holdout"),
                        ("AUPRC", "test"), ("AUPRC", "holdout"),
                        ("recall", "test"), ("recall", "holdout"),
                        ("FPR",   "test"), ("FPR",   "holdout")]
    else:
        metric_grid = [("balanced_accuracy", "test"),
                        ("balanced_accuracy", "holdout")]
        for c in range(4):
            metric_grid.append((f"recall__{c}", "test"))
            metric_grid.append((f"recall__{c}", "holdout"))

    for axis_col, axis_label in axes_cfg:
        if axis_col not in df.columns:
            continue
        levels = sorted(df[axis_col].dropna().astype(str).unique().tolist())
        for metric, part in metric_grid:
            for ss in ("full", "baseline_complete"):
                col = detect_metric_col(df, metric, part, ss)
                if col is None:
                    continue
                for lv in levels:
                    n = int(df[(df[axis_col].astype(str) == lv) &
                                df[col].notna()].shape[0])
                    audit_rows.append({
                        "axis":      axis_label,
                        "level":     lv,
                        "metric":    metric,
                        "partition": part,
                        "subset":    ss,
                        "n_samples": n,
                    })
    pd.DataFrame(audit_rows).to_csv(out_path, index=False)
    print(f"    ✓ wrote {out_path.name} ({len(audit_rows)} rows)")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"Reading from: {SCRIPT_DIR}")
    print(f"Writing to:   {OUT_DIR}\n")
    OUT_DIR.mkdir(exist_ok=True)

    BEST_BY_PHASE = {}
    for target in TARGETS:
        target_data = SCRIPT_DIR / target
        if not target_data.exists():
            print(f"[skip] {target} — no {target}/ folder at {target_data}")
            continue
        print(f"── {target} " + "─" * (60 - len(target)))
        df = load_target(target_data)
        if df is None or df.empty:
            print(f"  no data, skipping {target}")
            continue
        baselines = extract_baselines(df)
        print(f"  baselines detected: {list(baselines.keys())}")
        all_models = filter_to_models(df)
        # Keep the faithful Doshi ICD FFNN APART from the main per-family
        # charts; it gets its own ICD-vs-L3 comparison below.
        icd_models, models = split_icd_models(all_models)
        print(f"  models (baselines excluded): {len(models)}"
              f" (+{len(icd_models)} ICD models analysed separately)")

        target_out = OUT_DIR / target
        target_out.mkdir(exist_ok=True)

        # Faithful Doshi ICD FFNN vs the other L3 approaches (binary targets).
        make_icd_comparison_figure(all_models, target,
                                   target_out / f"{target}_icd_vs_l3.png")

        all_stats = []
        for subset in ("full", "baseline_complete"):
            fig_path = target_out / f"{target}_{subset}.png"
            stats_one = make_figure(models, baselines, subset, target, fig_path)
            make_phase_progression_figure(
                models, baselines, subset, target,
                target_out / f"{target}_phase_progression_{subset}.png")
            if subset == "full":
                metrics_root = target_data / "metrics"
                _abc = select_abc(models, target)
                _m = "balanced_accuracy" if target in BAND_TARGETS else "AUPRC"
                _c = detect_metric_col(models, _m, "test", "full")
                _vals = [float(r[_c]) if _c and pd.notna(r.get(_c)) else np.nan
                         for *_x, r in _abc]
                BEST_BY_PHASE[target] = (_abc, _m, _vals)
                make_models_vs_baselines_figure(
                    models, baselines, target, metrics_root,
                    target_out / f"{target}_models_vs_baselines.png")
                make_subgroup_radar_figure(
                    models, target, metrics_root,
                    target_out / f"{target}_subgroups_radar.png")
                make_temporal_stability_figure(
                    models, target, metrics_root,
                    target_out / f"{target}_temporal_stability.png")
            all_stats.append(stats_one)

        # ── Extra plots depending on target ──────────────────────────────
        if target in BINARY_TARGETS:
            # TPR / FPR secondary figure (test + holdout, full subset only)
            # baseline_complete not produced here — the ISS/NISS/TRISS recall
            # baseline values aren't stored in the JSON, so skip that subset.
            for subset in ("full", "baseline_complete"):
                fp = target_out / f"{target}_tpr_fpr_{subset}.png"
                make_figure(models, baselines, subset, target, fp,
                            metrics_mode="tpr_fpr")

        elif target in BAND_TARGETS:
            # Per-class recall secondary figure
            for subset in ("full",):
                fp = target_out / f"{target}_recall_by_class_{subset}.png"
                make_figure(models, baselines, subset, target, fp,
                            metrics_mode="recall_by_class")

        # Top models (legacy)
        write_top_models(models, baselines, target,
                          target_out / "top_models.csv")
        # Stats summary
        write_stats_summary(all_stats, target,
                              target_out / "stats_summary.csv")
        # NEW: best-model CSV per target (Task 2)
        write_best_models_csv(models, target,
                                target_out / "best_models.csv")
        # NEW: phase-filtered best models (Task 3, mortality only)
        if target in BINARY_TARGETS:
            write_phase_filtered_best_models_csv(
                models, target,
                target_out / "best_models_by_phase.csv")
        # NEW: audit log of sample counts (Task 1 verification)
        write_audit_log(models, target,
                          target_out / "sample_counts_audit.csv")
        print()

    if BEST_BY_PHASE:
        write_best_by_phase_txt(BEST_BY_PHASE, OUT_DIR / "best_models_by_phase.txt")

    print(f"\nDone.  Browse {OUT_DIR}/")


if __name__ == "__main__":
    main()
