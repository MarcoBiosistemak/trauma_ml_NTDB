"""`trauma-aggregate` — collect all model metrics into a single CSV.

Triggered automatically by ``trauma-train`` after the grid completes, but can
also be run manually to re-compute with different Pareto settings:

    trauma-aggregate \\
        --metrics-root  outputs/metrics \\
        --models-root   outputs/models  \\
        --output        outputs/all_metrics.csv \\
        --pareto-metric AUPRC \\
        --pareto-partitions test holdout_year

Column layout (left → right)
-----------------------------
1. ``model_id``
2. ``cfg_*`` / ``extra_*`` — full pipeline config (actual_model_type for AutoML, etc.)
3. Overall model metrics   — ``<metric>_<partition>``
4. Baseline metrics        — ``baseline_<ISS|NISS|TRISS>_<metric>_<partition>``
5. Per-subgroup metrics    — ``<metric>_<partition>__<axis>__<level>``
6. Worst-case Pareto inputs — ``worst_<pareto_metric>_<partition>__<axis>``
7. ``pareto_front``        — True if no other model dominates on all Pareto inputs
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        level=level,
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate all model metrics into a single CSV with Pareto front column.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metrics-root", default="outputs/metrics",
        help="Directory with one sub-folder per model_id (default: outputs/metrics)",
    )
    parser.add_argument(
        "--models-root", default=None,
        help="Directory with per-model config.json (default: <metrics-root parent>/models)",
    )
    parser.add_argument(
        "--output", default="outputs/all_metrics.csv",
        help="Output CSV path (default: outputs/all_metrics.csv)",
    )
    parser.add_argument(
        "--pareto-metric", default="AUPRC",
        help="Metric for worst-case Pareto computation (default: AUPRC). "
             "Any subgroup metric is valid: AUROC, f1, recall, precision, Brier.",
    )
    parser.add_argument(
        "--pareto-partitions", nargs="+", default=["test", "holdout_year"],
        help="Partitions included in worst-case Pareto (default: test holdout_year).",
    )
    parser.add_argument(
        "--pareto-objectives", nargs="+", default=None,
        help="Explicit Pareto objective column names. Overrides the automatic "
             "worst-case derivation from --pareto-metric / --pareto-partitions.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    from ..evaluation import (
        aggregate_all_metrics, aggregate_cohort_counts,
        aggregate_baseline_metrics_long,
    )

    metrics_root = Path(args.metrics_root)
    models_root  = Path(args.models_root) if args.models_root else None
    output_path  = Path(args.output)

    if not metrics_root.exists():
        print(f"[ERROR] --metrics-root does not exist: {metrics_root}", file=sys.stderr)
        sys.exit(1)

    df = aggregate_all_metrics(
        metrics_root=metrics_root,
        models_root=models_root,
        output_path=output_path,
        pareto_metric=args.pareto_metric,
        pareto_partitions=tuple(args.pareto_partitions),
        pareto_objectives=args.pareto_objectives or None,
    )

    if df.empty:
        print("[WARNING] No metrics found — check --metrics-root.", file=sys.stderr)
        sys.exit(1)

    n_pareto = int(df["pareto_front"].sum())
    worst_cols = [c for c in df.columns if c.startswith("worst_")]

    print(f"[OK] {len(df)} models aggregated  →  {output_path}")
    print(f"     Pareto metric : {args.pareto_metric} (worst-case across subgroups)")
    print(f"     Partitions    : {args.pareto_partitions}")
    print(f"     Pareto inputs : {len(worst_cols)} worst-case columns")
    print(f"     Pareto front  : {n_pareto} model(s)\n")

    # ── Cohort-filtered aggregations ─────────────────────────────────────
    # Use the main file's columns as a template so cohort CSVs share the
    # same column structure (NaN-filled where the cohort lacks subgroup
    # data, etc.).
    main_columns = list(df.columns) if not df.empty else None
    cohorts = ("onsite_complete", "ed_complete", "baseline_complete")
    for cohort in cohorts:
        cohort_path = output_path.with_name(
            output_path.stem + f"__cohort_{cohort}" + output_path.suffix
        )
        df_c = aggregate_all_metrics(
            metrics_root=metrics_root,
            models_root=models_root,
            output_path=cohort_path,
            pareto_metric=args.pareto_metric,
            pareto_partitions=tuple(args.pareto_partitions),
            pareto_objectives=args.pareto_objectives or None,
            cohort=cohort,
            column_template=main_columns,
        )
        if df_c.empty:
            print(f"[WARN] cohort={cohort}: no metrics found "
                  "(no model wrote cohort JSONs — re-run trauma-train).")
            continue
        n_front_c = int(df_c["pareto_front"].sum())
        print(f"[OK] cohort={cohort:<20s} {len(df_c)} models, "
              f"{n_front_c} on Pareto  →  {cohort_path}")

    # ── Cohort counts CSV ────────────────────────────────────────────────
    counts_path = output_path.with_name("cohort_counts.csv")
    counts_df = aggregate_cohort_counts(
        metrics_root=metrics_root, output_path=counts_path,
    )
    if not counts_df.empty:
        print(f"\n[OK] Cohort case counts → {counts_path}")
        # Print a compact pivot for quick inspection
        try:
            pv = counts_df.pivot_table(
                index=["model_id", "partition"],
                columns="cohort", values="n_cohort", aggfunc="first",
            )
            print(pv.to_string())
        except Exception:
            pass

    # ── Long-format baseline metrics CSV ─────────────────────────────────
    baseline_path = output_path.with_name("baseline_metrics.csv")
    baseline_df = aggregate_baseline_metrics_long(
        metrics_root=metrics_root, output_path=baseline_path,
    )
    if not baseline_df.empty:
        print(f"\n[OK] Baseline metrics (long format) → {baseline_path}")
        print(f"     {len(baseline_df)} rows: "
              f"{baseline_df['score'].nunique()} scores × "
              f"{baseline_df['partition'].nunique()} partitions × "
              f"{baseline_df['cohort'].nunique()} cohorts")

    # NOTE: ``models_and_baselines.csv`` is no longer produced — baseline
    # scores now appear as their own rows directly inside ``all_metrics.csv``
    # (and the cohort variants) per round-9 user request.

    pareto_df = df[df["pareto_front"]].copy()
    show_cols = (
        ["model_id", "cfg_actual_model_type", "cfg_model_family",
         "cfg_target_name", "cfg_imputer_method", "cfg_predictor_type"]
        + [c for c in df.columns if c.startswith("AUPRC_test") and "__" not in c][:2]
        + [c for c in df.columns if c.startswith("AUPRC_holdout_year") and "__" not in c][:2]
        + worst_cols[:4]
    )
    show_cols = [c for c in show_cols if c in pareto_df.columns]

    if not pareto_df.empty and show_cols:
        print("Pareto-front models:")
        print(pareto_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
