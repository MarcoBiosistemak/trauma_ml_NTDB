# Pipeline Options (run all per target)

Every option below is a `trauma-train` grid axis or a always-on pipeline stage.
The reference grid (`config/experiment_grids.reference.yaml`) runs the full
cross-product for each target.

## Targets (`--targets`)

| Name | Kind | Task | Notes |
|---|---|---|---|
| `in_hospital_mortality` | `binary` | binary | positive = died (disposition code 5, incl. ED death) |
| `ISS_band` | `ordinal_bands` | multiclass (4) | bands 0–9 / 9–16 / 16–25 / 25–75 on `ISS` |
| `NISS_band` | `ordinal_bands` | multiclass (4) | bands derived from AIS severities (`NISS`) |
| `ISS_band_binary` | `binary_threshold` | binary | `ISS ≥ 16` (severe vs not) |
| `NISS_band_binary` | `binary_threshold` | binary | `NISS ≥ 16` |

Built-in defs (`_BUILTIN_TARGET_DEFS`) mean the band-binary targets resolve
without editing the YAML.

## Phase cutoffs (`--phase-cutoffs`) — the 3 predictor families
1. `On-scene`
2. `On-scene + ED arrival`
3. `On-scene + ED arrival + In-hospital`

Feature counts (mortality / band): L1 = 32 / 32, L2 = 39 / 39, L3 = 63 / 61.

## Model families (`--model-families`)
Classification: `xgboost`, `lightgbm`, `catboost`, `flaml`, `tpot`,
`random_forest`, `logistic_l1`, `logistic_elasticnet`, `tabnet`, `tabpfn`,
`doshi_ffnn`, `ft_transformer`. Survival: `cox_ph`, `deep_surv`,
`random_survival_forest`.

**Faithful Doshi ICD FFNN (L3-only):** `doshi_ffnn_icd` feeds *only* a multi-hot
of the patient's ICD-10 diagnosis codes (the paper's direct ICD→severity model);
`doshi_ffnn_icd_plus` concatenates that multi-hot with the other L3 numeric
features (ablation for gain). Both target mortality, `ISS_band_binary`, and
`NISS_band_binary`; they require the in-hospital cutoff (ICD codes are
a-posteriori). Vocabulary = top `icd_max_vocab` codes with ≥ `icd_min_count`
patients. See `slurms/35_train_doshi_icd.slurm`.

**NaN-native** (accept `--imputers none`): `xgboost`, `lightgbm`, `catboost`,
`flaml`. All others require an imputer.

**Embedded feature selection by family:** `logistic_l1` / `logistic_elasticnet`
(L1 sparsity); boosting + RF (tree regularisation); `tabnet` (attentive masks);
`doshi_ffnn` (L1 on the first layer, `l1_lambda` default `1e-4`).

## Imputers (`--imputers`)
`none` (NaN-native models only), `median_mode`, `mice`, `bagged_trees`
(MissForest-style bagged trees), `missforest`, `gradient_boosting`
(HistGradientBoosting round-robin / MICE-style)
(MissForest-style: bagged-tree regressor for numeric + per-column bagged
classifier for categorical), `missforest`.

## Imputer-check (`--imputer-check`, the `_ic` configs)
After fitting the imputer, mask 10% of known **calibration-set** cells, re-impute,
and **keep only features that reconstruct well**:
- numeric: `relative_MAE = MAE / std(train) < 0.5`
- categorical: **balanced accuracy ≥ 0.6** (mean per-class recall; 0.5 = binary
  no-skill floor, so this requires beating chance on the minority class)
Recorded per feature in `imputation_eval_<method>.csv` (`keep` column). If no
feature survives, the run writes NaN metrics + a `skipped` flag and is marked
done.

## Pre-filter (always on; `prefilter`, all families/targets)
Computed on the **training split only**, before modelling. Drops, with a console
message naming each variable + reason:
- **near-zero variance** — constant, all-missing, or one value covering
  ≥ `nzv_dominant_frac` (default 0.999) of train rows;
- **redundancy** — one of each near-perfectly correlated numeric pair
  (`|Pearson| ≥ corr_drop_threshold`, default 0.9999) and any exact-duplicate
  column.

## Augmentation (`--augmentations`)
`null`, `smote`, `adasyn` — applied to the **minority class/band** for mortality
(binary), band-binary, and 4-class band targets. Skipped when `imputer=none`
(SMOTE/ADASYN can't run on NaN).

## Calibration (`--calibrations`)
`none`, `platt` (sigmoid), `isotonic` — binary targets only. Baselines
(ISS/NISS/TRISS) get the same calibration treatment for the mortality target.

## Missingness threshold (`--missingness-thresholds`)
e.g. `10 50` — drop predictors whose missingness exceeds the threshold (%).

## Clinical baselines (mortality only)
TRISS, RTS, MGAP, mREMS computed and overlaid on the ROC/PR plots and reported
as metric JSONs. Not produced for band targets (the score *is* the label there).

## Confidence intervals
- `--bootstrap-ci N` — heavy, all-metric stratified percentile CIs written to
  the `overall__*.json` files as `<metric>__ci_low/__ci_high`.
- `--plot-ci-bootstrap N` (default **200**, on) — light AUC/AP-only bootstrap
  that annotates the **binary ROC/PR plot legends** (model + every baseline)
  with their intervals. `0` disables. Binary-only.

## Evaluation slices (figures + metric JSON, per partition)
- **full** — every test/holdout row (`roc_pr_curves__full.png`, `confusion_matrix_test.png`).
- **baseline_complete** — rows where all clinical baselines are computable
  (apples-to-apples model vs baseline).
- **phase_complete** *(round 59)* — rows with ≥1 REAL (pre-imputation) value in
  **every** phase block up to the cutoff. Emits
  `overall__{test,holdout}__phase_complete.json` plus
  `roc_pr_curves__{test,holdout}__phase_complete.png` and
  `confusion_matrix_{test,holdout}__phase_complete.png`.

## Subgroup axes (per target, all partitions)
`gender`, `age_group`, `__admission_year`, `ethnicity`, `race`, `mechanism`,
`intent`, `insurance` (payer, labelled Medicaid/Medicare/Private/Commercial/…).
Each emits `…__subgroups_by_<axis>.csv`.

## Performance: preprocessing/model cache (default on; `--no-prep-cache` to disable)
The runner fits the encoders+imputer **once** per
`(target, predictor_type, phase, inclusion, missingness, imputer, family)` group
and reuses them across every calibration **and** augmentation variant, since none
of those change the fitted preprocessing. For **binary** targets it also trains the
base model **once** per `(…, augmentation)` and reuses it across `none/platt/isotonic`
— only the post-hoc calibrator on the calibration set differs. Net effect: a slow
imputer (mice/missforest/bagged_trees/gradient_boosting) is fit ~9× fewer times
(3 calibrations × 3 augmentations share one fit), and binary base-model training
runs ~3× fewer times. `imputation_eval/` is written once per group (under that
group's first `model_id`). Saved artifacts are **byte-identical** to a no-cache run
(verified); `--no-prep-cache` forces a fresh fit per combo if ever needed.

## Inclusion / predictor types
`--inclusion-strategies` and `--predictor-types` (e.g. `"Tran NTDB"`) select the
cohort/whitelist provenance.

## Deployment
Models are saved as a `ModelArtifact` (`artifact.pkl`) bundling **transformers →
imputer → model → calibration**. The cache above is a *training-time* optimization
only: every artifact serialises its **own** copy of the fitted imputer/transformers/
model/calibrator, so loading any single `model_id` is fully self-contained — it never
reaches back to a shared cache. `ModelArtifact.predict` applies the whole pipeline
(including imputation) to raw input, so deployments (incl. Hugging Face / the FastAPI
app) must load the artifact, not the bare estimator. `doshi_ffnn` needs `torch` in the
serving image.
