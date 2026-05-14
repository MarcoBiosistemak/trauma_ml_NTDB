# trauma_ml — pipeline options cheatsheet

This is the canonical reference for every choice you can make on the `trauma-train` / `trauma-run-experiments` command line. Run any combination via:

```bash
trauma-train \
    --targets in_hospital_mortality \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" \
    --imputers median_mode mice \
    --model-families logistic_l1 xgboost lightgbm \
    --calibrations none platt isotonic \
    --augmentations null smote \
    --missingness-thresholds 50 \
    --use-gpu auto
```

Every list-valued option is OR'd into the experiment grid. The grid size is the cartesian product of all axes.

## 1. Targets

| Value | Meaning |
|---|---|
| `in_hospital_mortality` | Binary: died during hospital stay |
| `iss_band_4` | Ordinal: ISS minor / moderate / severe / profound |
| `niss_band_4` | Same on NISS |
| `survival_time` | Time-to-event (requires survival model family) |

Override list with `--targets`. Defaults from `config/experiment_grids.yaml`.

## 2. Inclusion strategies (cohort filters)

| Value |
|---|
| `Tran NTDB` (the reference; matches Tran 2022 PLoS One exactly) |
| `ECTrauma` |
| `Karolinska SweTrau` |
| `RETRAUCI` |

Override with `--inclusion-strategies`.

## 3. Predictor types (registry-restricted variable subsets)

| Value | Effect |
|---|---|
| `all` | Every NTDB variable surviving the phase + missingness filters |
| `Tran NTDB` | Variables Tran 2022 used (about 30 with `On-scene + ED arrival + In-hospital` cutoff) |
| `Karolinska SweTrau` | Holtenius 2024 SweTrau registry feature set |
| `RETRAUCI` | Servia ICU-trauma registry |
| `ECTrauma` | EC trauma minimal data set |
| `EIPD` | EIPD trauma registry |

Override with `--predictor-types`.

## 4. Phase cutoffs (when in the patient journey are features known?)

**Time-ordered cutoffs** — each adds variables from the previous one:

| Value | Variables added |
|---|---|
| `On-scene` | EMS / on-scene physiology, demographics |
| `On-scene + ED arrival` | + ED-arrival vitals (TEMPERATURE, PULSEOXIMETRY, PULSERATE) |
| `On-scene + ED arrival + In-hospital` | + ISS, NISS, anatomy details computed in-hospital |

**Round-11 baseline-feature cutoffs** — pin the predictor set to the exact inputs of one clinical baseline so models can be compared head-to-head with that baseline:

| Value | Variables |
|---|---|
| `iss_only` | ISS, AGEYEARS, SEX |
| `niss_only` | NISS, AGEYEARS, SEX |
| `triss_inputs` | ISS, GCSTOTAL, SBPFIRST, RRFIRST, AGEYEARS, SEX, TRAUMATYPE |
| `all_baseline_inputs` | Union of ISS + NISS + TRISS inputs |

Override with `--phase-cutoffs`. Run `python -m trauma_ml.cli.phase_cutoff_audit` to render the Venn / membership diagram.

## 5. Imputers

| Value | Description |
|---|---|
| `none` | Pass-through (only works with models that accept NaN: lightgbm, xgboost) |
| `median_mode` | SimpleImputer median (numeric) + mode (categorical). Fast, robust default. |
| `knn` | KNNImputer with k=10 |
| `mice` | IterativeImputer (BayesianRidge) — Tran 2022's choice |
| `missforest` | miceforest (random-forest-based MICE; needs `[imputers]` extra) |

Override with `--imputers`.

## 6. Model families (13)

| Value | Family | GPU? |
|---|---|---|
| `logistic_l1` | LogisticRegression L1, saga | — |
| `logistic_elasticnet` | LogisticRegression elastic-net | — |
| `random_forest` | sklearn.RandomForestClassifier | — |
| `xgboost` | XGBoost | yes (`device=cuda`) |
| `lightgbm` | LightGBM | yes (needs CUDA build of lightgbm) |
| `catboost` | CatBoost | yes (`task_type=GPU`) |
| `flaml` | FLAML AutoML (5-min budget by default) | — |
| `tpot` | TPOT — genetic-programming pipeline search (no GPU; long walltime) | — |
| `tabpfn` | TabPFN — capped at ~10k rows, subsample for NTDB | yes (torch) |
| `tabnet` | pytorch-tabnet | yes (torch) |
| `ft_transformer` | FT-Transformer (scaffold; not yet wired in) | yes (torch) |
| `cox_ph` | scikit-survival CoxPH (survival target only) | — |
| `random_survival_forest` | scikit-survival RSF (survival target only) | — |
| `deep_surv` | DeepSurv (scaffold; not yet wired in) | — |

Override with `--model-families`. The GPU flag (see below) only affects families that support it.

## 7. Calibration

| Value | Method |
|---|---|
| `none` | No calibration (uses raw model probabilities) |
| `platt` | Platt scaling — sigmoid(a + b·score) |
| `isotonic` | Isotonic regression — monotonic step function |

Override with `--calibrations`. **Note:** `platt` and `isotonic` automatically also calibrate the baselines (ISS / NISS / TRISS) plus a Youden's-J-tuned threshold variant — see `baseline_calibrator__*.json`.

## 8. Data augmentation

| Value |
|---|
| `null` (or `none`) |
| `smote` |
| `adasyn` |

Override with `--augmentations`.

## 9. Missingness threshold

`--missingness-thresholds 50 60` — drop a column from the predictor pool if its NaN fraction exceeds N%. Default 50.

## 10. GPU policy (round 12)

| Value | Behaviour |
|---|---|
| `auto` (default) | Use GPU if `torch.cuda.is_available()`, else CPU |
| `force` | Use GPU; raise an error if none available |
| `never` | Always use CPU |

Override with `--use-gpu`. Affects xgboost / lightgbm / catboost / tabpfn / tabnet.

## 11. Parallel-Slurm options (round 12)

| Flag | Purpose |
|---|---|
| `--model-id-prefix lgb` | Generate model_ids `lgb_0000`, `lgb_0001`, …. **Set distinct prefixes when running multiple Slurm jobs writing to the same `outputs/` directory** to avoid collisions. |
| `--no-aggregate` | Skip the post-loop `all_metrics.csv` / Pareto step. Run `trauma-aggregate-metrics` once after every job has finished. |
| `--start-id 0 --end-id 23` | Run only grid indices 0..23 (inclusive). Used to slice a big grid across N Slurm jobs. |
| `--outputs-root outputs` | Where to write `metrics/`, `models/`, etc. Default from `config/paths.yaml`. |
