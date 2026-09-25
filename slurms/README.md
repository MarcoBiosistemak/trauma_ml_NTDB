# slurms/ — DIPC supercomputer pipeline (round 37, validated on atlas-edr)

Brief reference for what's in this directory and how to run it. For the full
project context see the top-level project docs.

## Cluster + environment

* Cluster: `atlas-edr.sw.ehu.es`, partition `general`. Two load-balanced login
  nodes (`-01`, `-02`); environment does NOT persist between sessions, so every
  interactive session must reload the module + venv:

  ```bash
  module purge && module load Python/3.10.4-GCCcore-11.3.0
  source /scratch/$USER/trauma_ml_venv/bin/activate
  cd /scratch/$USER/trauma_ml
  ```

  (Suggested `~/.bashrc` alias `trauma_env` does all four at once.)

* QoS ceilings: `regular` = 1 day, `long` = 2 days, `xlong` = 8 days.

## Files

| File | Family / Purpose | Compute | Base prefix | qos / walltime |
|---|---|---|---|---|
| `setup_env.slurm` | One-off venv build + `pip install -e .` | CPU | — | regular / 2 h |
| `00_build_dataset.slurm` | One-off `trauma-build` (AY 2019-2022 train + 2024 holdout) | CPU | — | regular / 6 h |
| `10_train_logistic.slurm` | logistic_l1 + logistic_elasticnet | CPU | `lgr` | xlong / 8 d |
| `11_train_lightgbm.slurm` | lightgbm | GPU auto | `lgb` | xlong / 4 d |
| `12_train_xgboost.slurm` | xgboost | GPU auto | `xgb` | xlong / 4 d |
| `13_train_catboost.slurm` | catboost | GPU auto | `cb` | xlong / 4 d |
| `14_train_flaml.slurm` | flaml (AutoML) | CPU | `flm` | xlong / 4 d |
| `15_train_tpot.slurm` | tpot (AutoML) | CPU | `tpt` | xlong / 8 d |
| `16_train_random_forest.slurm` | random_forest | CPU | `rf` | xlong / 4 d |
| `17_train_tabpfn.slurm` | tabpfn (sub-sampled) | GPU force | `tpf` | xlong / 4 d |
| `18_train_tabnet.slurm` | tabnet | GPU force | `tnt` | xlong / 4 d |
| `19_train_survival.slurm` | cox_ph + random_survival_forest (NOT currently used) | CPU | `surv` | long / 2 d |
| `20_train_iss_xgboost.slurm` | xgboost on ISS_band | GPU auto | `iss_xgb` | xlong / 4 d |
| `21_train_iss_lightgbm.slurm` | lightgbm on ISS_band | GPU auto | `iss_lgb` | xlong / 4 d |
| `22_train_iss_random_forest.slurm` | random_forest on ISS_band | CPU | `iss_rf` | xlong / 4 d |
| `23_train_niss_xgboost.slurm` | xgboost on NISS_band | GPU auto | `niss_xgb` | xlong / 4 d |
| `24_train_niss_lightgbm.slurm` | lightgbm on NISS_band | GPU auto | `niss_lgb` | xlong / 4 d |
| `25_train_niss_random_forest.slurm` | random_forest on NISS_band | CPU | `niss_rf` | xlong / 4 d |
| `26_train_iss_flaml.slurm` | flaml on ISS_band | CPU | `iss_flm` | long / 2 d |
| `27_train_iss_tpot.slurm` | tpot on ISS_band | CPU | `iss_tpt` | xlong / 8 d |
| `28_train_niss_flaml.slurm` | flaml on NISS_band | CPU | `niss_flm` | long / 2 d |
| `29_train_niss_tpot.slurm` | tpot on NISS_band | CPU | `niss_tpt` | xlong / 8 d |
| `99_aggregate.slurm` | One-off aggregation after all training returns | CPU | — | regular / 4 h |

## Round 37: imputer_check + imputer=none (multiple invocations per slurm)

Each family slurm now contains **stacked `trauma-train` invocations**, each with
its OWN `--model-id-prefix` so they never renumber or collide with each other.
The resume tracker keys on `<prefix>_<idx>`: a new prefix is a fresh sequence,
and any model whose `overall__test.json` already exists is skipped instantly.

| Invocation | Prefix suffix | What it does | Which families |
|---|---|---|---|
| base | (none, e.g. `xgb`) | impute (median_mode, mice, **bagged_trees**), imputer_check=False — the original grid | all |
| (a) | `_ic` (e.g. `xgb_ic`) | impute, then DROP features the imputer can't reconstruct on the calibration set | all |
| (b) | `_none` (e.g. `xgb_none`) | no imputation; model consumes NaN directly | xgboost, lightgbm, catboost, flaml ONLY |

**Round 54/55: `bagged_trees` imputer added** to the base + `_ic` invocations of
every family (NOT `_none`). It is a MissForest-style imputer — bagged decision
trees for numeric (`IterativeImputer`+`BaggingRegressor`) and a per-column bagged
classifier for categoricals. It is slower than median_mode/mice, so walltimes
were raised (4-day slurms → 7 days; tabnet slices → 3 days).

**Why `_none` is restricted:** xgboost / lightgbm / catboost / flaml handle NaN
natively (learned default split direction). logistic, random_forest, tpot,
tabnet, tabpfn cannot consume NaN — so `imputer=none` produces zero combos for
them and they get only the base + `_ic` invocations.

**imputer_check, briefly:** with `--imputer-check`, after fitting the imputer the
trainer masks 10% of known values **on the calibration set** (the test set is
left fully untouched), measures per-feature reconstruction (relative MAE for
numeric, **balanced accuracy** for categorical), and keeps only features that
pass the thresholds (numeric MAE/std < 0.5, categorical **balanced_accuracy >=
0.6**). The `imputation_eval_<method>.csv` records `keep` per feature. If NO
feature survives, the run writes NaN metrics + a `skipped` flag and is marked
done (so it isn't retried).

**imputer=none collapses the check dimension:** when there's no imputation there's
nothing to check, so `imputer=none` only ever produces the imputer_check=False
variant — no duplicate runs.

**Backward compatibility:** models trained before round 37 have no `imputer_check`
field in their config.json. Treat a missing field as False. The standalone
`patch_imputer_check.py` (project root, NOT in the package) backfills
`imputer_check: false` into existing configs; it is idempotent and only needs to
be run once.

## Resource sizing (validated against atlas-edr.sw.ehu.es)

* GPU jobs request a GPU via `--use-gpu auto` (lgb/xgb/cb) or `force` (tabpfn/
  tabnet); the RTX 3090 has 24 GB VRAM.
* random_forest uses `--mem=250G` + `--n-jobs 12` + `max_samples=0.5` in the
  factory — round-36 fix for OOM-kills on the band targets at 2.75M rows × 500
  trees.
* tpot uses `--mem=200G`; its GA search runs on a 20% subsample (round 35) with
  the winning pipeline re-fit on full data.

## How parallelism + aggregation work

Each training job trains a disjoint slice of the grid (distinct
`--model-families` and prefix) and passes `--no-aggregate` so it never writes its
own `all_metrics.csv` (otherwise concurrent jobs would race). `99_aggregate.slurm`
runs once after all training succeeds, walks `outputs/<target>/metrics/`, and
writes the unified `all_metrics.csv`, per-cohort variants, and
`baseline_metrics.csv` (ISS / NISS / TRISS).

## Notes on specific families

* **random_forest** runs with `--no-shap` on BOTH its base and `_ic` invocations.
  SHAP's TreeExplainer segfaults (C-extension) on a 500-tree forest at this
  scale. xgb/lgb/cb keep SHAP; it's stable for them.
* **TabNet** (round 37) uses a 5% calibration eval_set for early stopping
  (`max_epochs=50`, `patience=10`, `batch_size=4096`) — was running the full 100
  epochs at batch 1024 with the GPU at 16% util, ~2.5 h/combo; now ~20-30 min.
* **TabPFN** is hard-capped at ~10k training rows by its pretrained transformer;
  the slurm sub-samples accordingly.
* **TPOT** uses `balanced_accuracy` scoring on the multiclass band targets
  (plain accuracy would let it win by predicting the majority class) and depends
  on `setuptools<81` (transitive `stopit` → legacy `pkg_resources`).
* **Survival** (`19_*`) is not currently part of the run — NTDB lacks a clean
  time-to-death; advised to skip.

## Re-running after a code change (standard cycle)

```bash
scp src.zip mcapo@atlas-edr.sw.ehu.es:/scratch/mcapo/trauma_ml/
# on the cluster, with venv active:
unzip -o src.zip
pip install -e .          # MANDATORY after any code change
# resubmit the affected slurms; completed combos skip via the resume tracker
sbatch slurms/16_train_random_forest.slurm   # etc.
```

To verify the resume tracker is skipping (not retraining) completed work:

```bash
ls -t slurms/logs/train_xgb_*.out | head -1 | xargs grep -c "SKIP (already completed"
```
