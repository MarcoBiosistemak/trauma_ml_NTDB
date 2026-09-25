#!/usr/bin/env python3
"""Check whether the best-per-phase models still have their artifacts, and emit
exact re-run commands for the ones the pruner removed.

`analysis/best_models_by_phase.txt` names the best model at L1/L2/L3 for each
target. Some of those may have had `artifact.pkl` deleted by
`prune_non_best_models.py` (an early version deleted the whole models/<id>/
directory, later versions keep config.json). This script reports, per model:

    KEEP    artifact.pkl present  -> nothing to do
    RERUN   artifact.pkl missing  -> re-train needed

For every RERUN it prints a `trauma-train` command reconstructed from the
model's own saved configuration where available, or from the grid position
otherwise. Because `model_id` is `<prefix>_<grid_index>` and the grid is
enumerated deterministically, re-running the owning slurm regenerates exactly
the same model_id with exactly the same configuration -- provided the code and
the dataset are unchanged, the result is bit-identical for the CPU families.

USAGE
-----
    python3 scripts/check_best_models_present.py \
        --outputs-root outputs \
        --txt outputs/analysis/best_models_by_phase.txt

Add --emit-slurm to also print, for each missing model, the slurm file whose
index range covers it (so you can resubmit just that one).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ID_RE = re.compile(r"^\s*(L[123])\s+(\S+)\s+([\d.]+|n/a)\s+(.*)$")
TARGET_RE = re.compile(r"^###\s+(.*)\s+\[(\w+)\]")

# model-id prefix -> the slurm that owns it (for --emit-slurm)
PREFIX_SLURM = {
    "xgb": "12_train_xgboost.slurm", "lgb": "11_train_lightgbm.slurm",
    "cb": "13_train_catboost.slurm", "flm": "14_train_flaml.slurm",
    "lgr": "10_train_logistic.slurm", "rf": "16_train_random_forest.slurm",
    "tpt": "15_train_tpot.slurm", "tpf": "17_train_tabpfn.slurm",
    "tnt": "18{slice}_train_tabnet.slurm", "doshi": "30_train_mortality_doshi_ffnn.slurm",
    "doshi_icd": "35_train_doshi_icd.slurm", "doshi_icdplus": "35_train_doshi_icd.slurm",
    "iss_xgb": "20_train_iss_xgboost.slurm", "iss_lgb": "21_train_iss_lightgbm.slurm",
    "iss_rf": "22_train_iss_random_forest.slurm", "iss_flm": "26_train_iss_flaml.slurm",
    "iss_tpt": "27_train_iss_tpot.slurm", "iss_doshi": "33_train_iss_doshi_ffnn.slurm",
    "niss_xgb": "23_train_niss_xgboost.slurm", "niss_lgb": "24_train_niss_lightgbm.slurm",
    "niss_rf": "25_train_niss_random_forest.slurm", "niss_flm": "28_train_niss_flaml.slurm",
    "niss_tpt": "29_train_niss_tpot.slurm", "niss_doshi": "34_train_niss_doshi_ffnn.slurm",
    "iss_bin_boost": "31{slice}_train_iss_bin_boost.slurm",
    "iss_bin_other": "31{slice}_train_iss_bin_other.slurm",
    "niss_bin_boost": "32{slice}_train_niss_bin_boost.slurm",
    "niss_bin_other": "32{slice}_train_niss_bin_other.slurm",
}
SLICED = {"tnt": 45, "iss_bin_boost": 216, "niss_bin_boost": 216,
          "iss_bin_other": 225, "niss_bin_other": 225}


def parse_txt(path: Path):
    """Yield (target, phase, model_id) from best_models_by_phase.txt."""
    target = None
    for line in path.read_text().splitlines():
        m = TARGET_RE.match(line)
        if m:
            target = m.group(2)
            continue
        if target and line.startswith("    L"):
            m2 = ID_RE.match(line)
            if m2 and m2.group(2) != "":
                yield target, m2.group(1), m2.group(2)


def owning_slurm(model_id: str) -> str | None:
    m = re.match(r"^(?P<p>.+?)_(?:none_)?(?P<i>\d{4,})$", model_id)
    if not m:
        return None
    pfx, idx = m.group("p"), int(m.group("i"))
    if pfx.endswith("_none"):
        pfx = pfx[:-5]
    tmpl = PREFIX_SLURM.get(pfx)
    if tmpl is None:
        return None
    if "{slice}" in tmpl:
        step = SLICED.get(pfx, 45)
        letter = "abcdefghijkl"[min(idx // step, 11)]
        return tmpl.format(slice=letter)
    return tmpl


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--txt", default="outputs/analysis/best_models_by_phase.txt")
    ap.add_argument("--emit-slurm", action="store_true",
                    help="Also print the slurm file covering each missing model.")
    args = ap.parse_args()

    root = Path(args.outputs_root)
    txt = Path(args.txt)
    if not txt.is_file():
        print(f"ERROR: {txt} not found. Run analysis_figures.py first."); return

    rows, missing = [], []
    for target, phase, mid in parse_txt(txt):
        mdir = root / target / "models" / mid
        art = mdir / "artifact.pkl"
        cfg = mdir / "config.json"
        state = "KEEP " if art.is_file() else "RERUN"
        size = f"{art.stat().st_size/2**30:.2f} GiB" if art.is_file() else "-"
        rows.append((target, phase, mid, state, size, cfg.is_file()))
        if state == "RERUN":
            missing.append((target, phase, mid, cfg if cfg.is_file() else None))

    print("=" * 96)
    print(f"{'target':<18}{'phase':<6}{'model_id':<24}{'state':<7}{'artifact':<12}config.json")
    print("=" * 96)
    for target, phase, mid, state, size, has_cfg in rows:
        print(f"{target:<18}{phase:<6}{mid:<24}{state:<7}{size:<12}"
              f"{'yes' if has_cfg else 'MISSING'}")
    print("=" * 96)
    print(f"{len(rows) - len(missing)} present, {len(missing)} need re-training.\n")

    if not missing:
        print("All best-per-phase models still have their artifacts - nothing to do.")
        return

    print("RE-RUN PLAN")
    print("-" * 96)
    print("model_id is <prefix>_<grid_index> and the grid is enumerated")
    print("deterministically, so re-running the owning slurm regenerates the SAME")
    print("model_id with the SAME configuration. Delete the metrics sentinel first,")
    print("otherwise the resume tracker will skip the combo as already complete.\n")
    for target, phase, mid, cfgp in missing:
        print(f"# {target} {phase}  {mid}")
        if cfgp:
            try:
                cfg = json.loads(cfgp.read_text())
                keys = ("model_family", "phase_cutoff", "imputer_method",
                        "calibration", "data_augmentation", "missingness_threshold")
                desc = ", ".join(f"{k}={cfg.get(k, cfg.get('cfg_' + k, '?'))}" for k in keys)
                print(f"#   saved config: {desc}")
            except Exception:
                pass
        print(f"rm -rf {root}/{target}/metrics/{mid} {root}/{target}/models/{mid}")
        sl = owning_slurm(mid)
        if args.emit_slurm and sl:
            print(f"sbatch slurms/{sl}")
        elif args.emit_slurm:
            print(f"#   (could not infer owning slurm for prefix of {mid})")
        print()
    print("After the queue drains, re-run analysis_figures.py to confirm the")
    print("artifacts are back and the winners are unchanged.")


if __name__ == "__main__":
    main()
