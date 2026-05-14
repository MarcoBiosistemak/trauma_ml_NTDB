# slurms/ — DIPC supercomputer pipeline (round 14, validated on atlas-edr)

For the step-by-step "what to run, in what order" guide, read **`DIPC_RUNBOOK.md`** at the project root. This file is the brief reference for what's in the directory.

## Files

| File | Family / Purpose | Compute | Prefix | Walltime |
|---|---|---|---|---|
| `requirements.txt` | Python deps with all pins learned from real installs | — | — | — |
| `setup_env.slurm` | One-off venv build + pip install — alternative to interactive setup | CPU | — | 2 h |
| `00_build_dataset.slurm` | One-off `trauma-build` (AY 2019/2020/2021/2022 train + 2024 holdout) | CPU | — | 6 h |
| `10_train_logistic.slurm` | logistic_l1 + logistic_elasticnet | CPU | `lgr_*` | 1 day |
| `11_train_lightgbm.slurm` | lightgbm | GPU (rtx3090) | `lgb_*` | 1 day |
| `12_train_xgboost.slurm` | xgboost | GPU (rtx3090) | `xgb_*` | 1 day |
| `13_train_catboost.slurm` | catboost | GPU (rtx3090) | `cb_*` | 1 day |
| `14_train_flaml.slurm` | flaml (AutoML) | CPU | `flm_*` | 2 days |
| `15_train_tpot.slurm` | tpot (AutoML) | CPU | `tpt_*` | 3 days |
| `16_train_random_forest.slurm` | random_forest | CPU | `rf_*` | 2 days |
| `17_train_tabpfn.slurm` | tabpfn (sub-sampled to ~12k) | GPU (rtx3090) | `tpf_*` | 6 h |
| `18_train_tabnet.slurm` | tabnet | GPU (rtx3090) | `tnt_*` | 1 day |
| `19_train_survival.slurm` | cox_ph + random_survival_forest | CPU | `surv_*` | 2 days |
| `99_aggregate.slurm` | One-off aggregation after all training jobs return success | CPU | — | 2 h |
| `submit_all.sh` | Wrapper that submits everything with the right dependency chain | — | — | — |

## Resource sizing (validated against atlas-edr.sw.ehu.es, April 2026)

* GPU jobs request `--gres=gpu:rtx3090:1` (RTX 3090, 24 GB VRAM, 7 nodes, Ampere).
* GPU jobs use `--mem=80G` (RTX 3090 nodes have 95 GB total / ~92 GB usable).
* CPU jobs use up to `--mem=120G` (smallest CPU nodes have 128 GB / ~125 GB usable).
* Largest CPU request is `--cpus-per-task=20` (smallest GPU/CPU node has 32-48 cores).

If you adapt to a different cluster, run `sinfo -N -p <partition> -o "%n %c %m %G"` and adjust those three numbers in every slurm.

## How parallelism + Pareto work

Each training job:

1. Trains a disjoint subset of the experiment grid (different `--model-families`).
2. Uses a distinct `--model-id-prefix` so per-model output paths never collide:
   `outputs/models/lgr_NNNN/`, `outputs/models/lgb_NNNN/`, etc.
3. Passes `--no-aggregate` so it skips writing its own `all_metrics.csv`. Otherwise three jobs would race to overwrite each other.

`99_aggregate.slurm` runs *once* after every training job has succeeded (`afterok` dependency), walks the entire `outputs/metrics/` tree, and produces:

* `outputs/all_metrics.csv` — unified Pareto across every model from every job
* `outputs/all_metrics__cohort_*.csv` — per-cohort variants
* `outputs/baseline_metrics.csv` — long-format ISS/NISS/TRISS (with calibration variants)
* `outputs/cohort_counts.csv` — case counts per cohort

## Notes on specific families

* **TabPFN (`17_train_tabpfn.slurm`)** is hard-capped at ~10,000 training rows by its pretrained transformer architecture. The slurm passes `--sample-fraction 0.005` for the full NTDB pool (4.67M rows). Bump that fraction only if you've already filtered the cohort heavily.
* **Survival (`19_train_survival.slurm`)** uses `--targets survival_time` rather than `in_hospital_mortality`. Don't try to mix `cox_ph` or `random_survival_forest` into another slurm with a binary target — they error at trainer time.
* **TPOT (`15_train_tpot.slurm`)** depends on `setuptools<81` because its transitive dep `stopit` imports the legacy `pkg_resources` API removed in setuptools 81+. The pin is in `requirements.txt`.
* **`ft_transformer` and `deep_surv`** are intentionally absent — they're scaffolds in `models.py` that raise `NotImplementedError`.

## Adding more families / wider grid

Drop a new `slurms/2X_train_<family>.slurm` (copy from a similar one), change `--model-families` and `--model-id-prefix`, then add it to `TRAIN_JOBS` in `submit_all.sh`. The aggregator picks it up automatically.

To narrow a grid (faster runs), edit the `--imputers` / `--missingness-thresholds` / `--phase-cutoffs` / `--augmentations` arguments in the slurm. The grid is the cartesian product, so dropping one option from a 3-option axis cuts walltime by ~33%.

To widen a grid, do the opposite — but watch the walltime ceiling. The current grids are sized to fit within `--time` (1-3 days depending on family).
