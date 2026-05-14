"""Audit phase-cutoff feature sets and visualise their overlaps.

Usage
-----
    python -m trauma_ml.cli.phase_cutoff_audit \
        --catalogue path/to/catalogue.xlsx \
        --parquet outputs/datasets/unified_train.parquet \
        --output outputs/phase_cutoff_audit/

Produces in ``--output``:

* ``phase_cutoff_features.csv`` — long-format table: phase_cutoff, feature,
  is_baseline_input, baseline_score (if applicable), nan_pct (in parquet).
* ``phase_cutoff_summary.csv`` — wide pivot: rows are features, columns are
  phase_cutoffs and baseline-input groups; cells are 0/1 for membership.
* ``phase_cutoff_overlap.png`` — figure with two panels:
    A) 3-way Venn of On-scene ∩ On-scene+ED ∩ TRISS-inputs.
    B) Membership matrix — every variable shown as a row, every feature
       set as a column; filled cells indicate membership.

What this addresses
-------------------
Round-11 item 4: "double-check the correctness of the on-scene / ED
features" + "show overlap with baseline metric required features".  Helps
catch mistakes like an On-scene cutoff accidentally including a variable
recorded only later in the patient journey, or a Tran-style baseline
cutoff missing a required input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Per-phase-cutoff feature lists (canonical clinical inputs).
# These mirror the trainer's _COHORT_VAR_DEFS plus the new round-11
# baseline-feature cutoffs.  Kept here as a self-contained reference so
# the audit doesn't depend on a fully-loaded catalogue.
# ---------------------------------------------------------------------------
PHASE_CUTOFF_FEATURES: dict[str, list[str]] = {
    # Time-ordered cutoffs — each one ADDS variables to the previous.
    "On-scene": [
        "AGEYEARS", "SEX", "TRAUMATYPE",
        "GCSTOTAL", "SBPFIRST", "RRFIRST",
    ],
    "On-scene + ED arrival": [
        "AGEYEARS", "SEX", "TRAUMATYPE",
        "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "TEMPERATURE", "PULSEOXIMETRY", "PULSERATE",
    ],
    "On-scene + ED arrival + In-hospital": [
        "AGEYEARS", "SEX", "TRAUMATYPE",
        "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "TEMPERATURE", "PULSEOXIMETRY", "PULSERATE",
        "ISS", "NISS",
    ],
    # Round-11 baseline-feature cutoffs.
    "iss_only":             ["ISS", "AGEYEARS", "SEX"],
    "niss_only":            ["NISS", "AGEYEARS", "SEX"],
    "triss_inputs": [
        "ISS", "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "AGEYEARS", "SEX", "TRAUMATYPE",
    ],
    "all_baseline_inputs": [
        "ISS", "NISS",
        "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "AGEYEARS", "SEX", "TRAUMATYPE",
    ],
}

# Baseline scores → exact required input columns
BASELINE_REQUIRED: dict[str, list[str]] = {
    "ISS":   ["ISS"],
    "NISS":  ["NISS"],
    "TRISS": ["GCSTOTAL", "SBPFIRST", "RRFIRST", "ISS", "AGEYEARS", "TRAUMATYPE"],
}

# Expected NTDB phase tag for each variable — used to flag any phase-cutoff
# entry that violates the temporal ordering.  Source: NTDB PUF data
# dictionary; cross-checked against Tran 2022 supplementary tables.
EXPECTED_PHASE: dict[str, str] = {
    "AGEYEARS":      "Demographics",
    "SEX":           "Demographics",
    "TRAUMATYPE":    "Demographics",   # joined from PUF_ECODE_LOOKUP
    "GCSTOTAL":      "On-scene",
    "SBPFIRST":      "On-scene",
    "RRFIRST":       "On-scene",
    "TEMPERATURE":   "At ED arrival",
    "PULSEOXIMETRY": "At ED arrival",
    "PULSERATE":     "At ED arrival",
    "ISS":           "In-hospital (a posteriori)",
    "NISS":          "In-hospital (a posteriori)",
}


def _audit_temporal_correctness(verbose: bool = True) -> list[str]:
    """Check each phase_cutoff list for variables that arrive too late
    in the patient journey.  Returns a list of warning strings.
    """
    warnings: list[str] = []
    phase_idx = {
        "Demographics": 0,
        "On-scene": 1,
        "At ED arrival": 2,
        "In-hospital (a posteriori)": 3,
    }
    cutoff_max = {
        "On-scene":                            phase_idx["On-scene"],
        "On-scene + ED arrival":               phase_idx["At ED arrival"],
        "On-scene + ED arrival + In-hospital": phase_idx["In-hospital (a posteriori)"],
    }
    for cutoff_name, max_idx in cutoff_max.items():
        for var in PHASE_CUTOFF_FEATURES[cutoff_name]:
            if var not in EXPECTED_PHASE:
                continue
            v_idx = phase_idx[EXPECTED_PHASE[var]]
            if v_idx > max_idx:
                msg = (
                    f"[VIOLATION] cutoff {cutoff_name!r} includes "
                    f"{var!r} (phase={EXPECTED_PHASE[var]}), which arrives "
                    f"AFTER the cutoff."
                )
                warnings.append(msg)
                if verbose:
                    print(f"  {msg}")
    if not warnings and verbose:
        print("  All phase_cutoff variable lists are temporally consistent.")
    # Round-11 baseline-feature cutoffs are NOT temporally constrained
    # (they purposely include In-hospital ISS/NISS) — skip those.
    return warnings


def _measure_nan_pct(parquet: Path | None,
                      variables: Iterable[str]) -> dict[str, float | None]:
    """Return per-variable NaN percentage from the supplied parquet.
    None for variables not present in the schema; float % otherwise.
    """
    if parquet is None:
        return {v: None for v in variables}
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(parquet))
    schema = set(pf.schema.names)
    wanted = [v for v in variables if v in schema]
    if not wanted:
        return {v: None for v in variables}
    df = pf.read(columns=wanted).to_pandas()
    out: dict[str, float | None] = {}
    for v in variables:
        if v not in schema:
            out[v] = None
        else:
            out[v] = float(100 * df[v].isna().mean())
    return out


def _build_long_table(nan_pct: dict[str, float | None]) -> pd.DataFrame:
    rows = []
    # Phase-cutoff rows
    for cutoff, feats in PHASE_CUTOFF_FEATURES.items():
        for f in feats:
            baseline_for = [s for s, reqs in BASELINE_REQUIRED.items() if f in reqs]
            rows.append({
                "phase_cutoff":      cutoff,
                "feature":           f,
                "is_baseline_input": len(baseline_for) > 0,
                "baseline_scores":   ";".join(baseline_for) if baseline_for else "",
                "expected_phase":    EXPECTED_PHASE.get(f, "?"),
                "nan_pct_in_parquet": nan_pct.get(f),
            })
    return pd.DataFrame(rows)


def _build_wide_table(nan_pct: dict[str, float | None]) -> pd.DataFrame:
    """Wide pivot: rows=variables, columns=phase_cutoffs + baseline
    requirement groups, cells=1 if member else 0.  A summary 'used_in_n'
    column counts how many groups each variable belongs to.
    """
    all_vars = sorted({v for vs in PHASE_CUTOFF_FEATURES.values() for v in vs}
                       | {v for vs in BASELINE_REQUIRED.values() for v in vs})
    rows = []
    for v in all_vars:
        row = {"feature": v, "expected_phase": EXPECTED_PHASE.get(v, "?")}
        for cutoff, feats in PHASE_CUTOFF_FEATURES.items():
            row[f"cutoff:{cutoff}"] = int(v in feats)
        for score, reqs in BASELINE_REQUIRED.items():
            row[f"baseline:{score}"] = int(v in reqs)
        row["nan_pct_in_parquet"] = nan_pct.get(v)
        memberships = (
            sum(int(v in feats) for feats in PHASE_CUTOFF_FEATURES.values())
            + sum(int(v in reqs) for reqs in BASELINE_REQUIRED.values())
        )
        row["used_in_n"] = memberships
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure: 3-way Venn (matplotlib-only) + membership matrix
# ---------------------------------------------------------------------------
def _draw_venn3(ax, set_a, set_b, set_c, labels: tuple[str, str, str]) -> None:
    """Hand-drawn 3-way Venn using matplotlib patches + text annotations.
    No matplotlib_venn dependency.
    """
    from matplotlib.patches import Circle

    # Region counts (standard Venn-3 regions)
    A, B, C = set(set_a), set(set_b), set(set_c)
    only_A   = A - B - C
    only_B   = B - A - C
    only_C   = C - A - B
    AB_only  = (A & B) - C
    AC_only  = (A & C) - B
    BC_only  = (B & C) - A
    ABC      = A & B & C

    # Three overlapping circles
    radius = 1.6
    centers = {
        "A": (-1.0,  0.6),
        "B": ( 1.0,  0.6),
        "C": ( 0.0, -1.0),
    }
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    for key, (cx, cy) in centers.items():
        ax.add_patch(Circle((cx, cy), radius, alpha=0.32,
                             facecolor=colors[key], edgecolor=colors[key], linewidth=2))

    # Region annotations (count + variable list)
    def _annot(x, y, items, fontsize=8):
        if not items:
            ax.text(x, y, "0", ha="center", va="center", fontsize=fontsize+2,
                     fontweight="bold")
            return
        text = f"n={len(items)}\n" + "\n".join(sorted(items))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    _annot(-2.0,  1.4, only_A)
    _annot( 2.0,  1.4, only_B)
    _annot( 0.0, -2.4, only_C)
    _annot( 0.0,  1.5, AB_only)
    _annot(-1.4, -0.8, AC_only)
    _annot( 1.4, -0.8, BC_only)
    _annot( 0.0,  0.0, ABC, fontsize=9)

    # Set labels
    ax.text(-2.5, 2.4, labels[0], fontsize=11, fontweight="bold", color=colors["A"])
    ax.text( 1.5, 2.4, labels[1], fontsize=11, fontweight="bold", color=colors["B"])
    ax.text(-0.6, -3.2, labels[2], fontsize=11, fontweight="bold", color=colors["C"])

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 3)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_membership_matrix(ax, wide: pd.DataFrame) -> None:
    """Membership matrix — rows are features, columns are sets,
    filled cells are members.  Variables sorted by phase, then name.
    """
    phase_order = {"Demographics": 0, "On-scene": 1, "At ED arrival": 2,
                   "In-hospital (a posteriori)": 3, "?": 9}
    wide = wide.copy()
    wide["_phase_idx"] = wide["expected_phase"].map(phase_order).fillna(9)
    wide = wide.sort_values(["_phase_idx", "feature"]).reset_index(drop=True)

    set_cols = [c for c in wide.columns
                 if c.startswith("cutoff:") or c.startswith("baseline:")]
    n_rows = len(wide)
    n_cols = len(set_cols)
    matrix = wide[set_cols].to_numpy(dtype=float)

    ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1.4)
    # X labels (sets)
    short = [c.replace("cutoff:", "").replace("baseline:", "BL: ")
             for c in set_cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    # Y labels (features) — colour by phase
    phase_colors = {
        "Demographics":                 "#444444",
        "On-scene":                     "#1f77b4",
        "At ED arrival":                "#ff7f0e",
        "In-hospital (a posteriori)":   "#d62728",
        "?":                            "#888888",
    }
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(wide["feature"], fontsize=9)
    for tick, phase in zip(ax.get_yticklabels(), wide["expected_phase"]):
        tick.set_color(phase_colors.get(phase, "#000000"))

    # Black borders on filled cells
    for i in range(n_rows):
        for j in range(n_cols):
            if matrix[i, j] >= 1:
                ax.text(j, i, "●", ha="center", va="center",
                         fontsize=8, color="white")
    ax.set_title("Variable membership matrix\n"
                  "(rows colored by expected NTDB phase)", fontsize=10)
    ax.tick_params(axis="both", which="both", length=0)


def _save_figure(out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.4], wspace=0.30)
    ax_venn = fig.add_subplot(gs[0, 0])
    ax_mat  = fig.add_subplot(gs[0, 1])

    # Panel A — 3-way Venn: On-scene vs On-scene+ED vs TRISS-inputs.
    # Picked these three because they're the most clinically interesting
    # cross-section: time-ordered phases × the canonical baseline features.
    _draw_venn3(
        ax_venn,
        set(PHASE_CUTOFF_FEATURES["On-scene"]),
        set(PHASE_CUTOFF_FEATURES["On-scene + ED arrival"]),
        set(PHASE_CUTOFF_FEATURES["triss_inputs"]),
        labels=("On-scene", "On-scene + ED", "TRISS inputs"),
    )
    ax_venn.set_title("Phase-cutoff feature overlap\n"
                       "(3-way Venn — counts and variables in each region)",
                       fontsize=10)

    # Panel B — full membership matrix
    nan_pct: dict[str, float | None] = {}  # not needed for the matrix
    _draw_membership_matrix(ax_mat, _build_wide_table(nan_pct))

    fig.suptitle(
        "Phase-cutoff vs baseline feature audit "
        "(round-11 item 4)",
        fontsize=12, fontweight="bold",
    )
    out_path = out_dir / "phase_cutoff_overlap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Audit phase-cutoff feature sets and produce a Venn-style figure."
    )
    parser.add_argument("--parquet", type=Path, default=None,
                         help="Optional unified parquet to measure NaN rates against.")
    parser.add_argument("--output", type=Path,
                         default=Path("outputs/phase_cutoff_audit"),
                         help="Output directory for CSVs and the figure.")
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== Phase-cutoff temporal-correctness audit ==")
    warnings = _audit_temporal_correctness(verbose=True)

    all_vars = sorted({v for vs in PHASE_CUTOFF_FEATURES.values() for v in vs}
                       | {v for vs in BASELINE_REQUIRED.values() for v in vs})
    nan_pct = _measure_nan_pct(args.parquet, all_vars)

    long_df = _build_long_table(nan_pct)
    wide_df = _build_wide_table(nan_pct)
    long_df.to_csv(out_dir / "phase_cutoff_features.csv", index=False)
    wide_df.to_csv(out_dir / "phase_cutoff_summary.csv", index=False)
    print(f"\nWrote {out_dir / 'phase_cutoff_features.csv'} ({len(long_df)} rows)")
    print(f"Wrote {out_dir / 'phase_cutoff_summary.csv'} ({len(wide_df)} rows)")

    fig_path = _save_figure(out_dir)
    print(f"Wrote {fig_path}")

    if warnings:
        print(f"\n[!] {len(warnings)} temporal-correctness violations — "
               "see above.  Phase-cutoff variable lists need fixing.")
        sys.exit(1)
    else:
        print("\nAll phase-cutoff feature sets are temporally consistent.")


if __name__ == "__main__":
    main()
