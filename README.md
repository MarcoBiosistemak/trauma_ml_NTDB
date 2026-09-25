# trauma_ml

Machine-learning pipeline for **trauma outcome prediction** on the American
College of Surgeons **National Trauma Data Bank (NTDB / TQP PUF)**. It builds a
harmonised multi-year dataset from the raw NTDB CSVs and trains, calibrates,
evaluates, and packages models for three families of outcomes, under a
**phase-of-care** framework that mirrors when information actually becomes
available to a clinician.

Developed for Biosistemak / DIPC (Bilbao). Runs locally (PyCharm) or as a SLURM
sweep on the DIPC cluster.

---

## What it predicts (targets)

| Target | Kind | Task | Definition |
|---|---|---|---|
| `in_hospital_mortality` | binary | binary | death (disposition incl. ED death) |
| `ISS_band` | ordinal bands | 4-class | Injury Severity Score 0–9 / 9–16 / 16–25 / 25–75 |
| `NISS_band` | ordinal bands | 4-class | New ISS, same band cuts |
| `ISS_band_binary` | binary threshold | binary | severe injury, `ISS ≥ 16` |
| `NISS_band_binary` | binary threshold | binary | severe injury, `NISS ≥ 16` |

Survival families (`cox_ph`, `deep_surv`, `random_survival_forest`) are also
supported for time-to-event framings.

## Phase-of-care framework

Predictors are grouped by **when they become known**, so a model can be trained
for the exact decision point it will be used at:

1. **On-scene / first contact** (L1, 32 features) — demographics, payer,
   anthropometry, mechanism/intent, on-scene GCS/SBP/RR, prehospital arrest,
   transport mode, 18 comorbidities.
2. **+ At ED arrival** (L2, +7 → 39) — ED vitals (temperature, SpO₂, pulse),
   arrival interval, pupillary response, alcohol screen.
3. **+ In-hospital** (L3) — injury coding: 22 Barell/specific-injury features,
   plus `ISS`/`NISS` for the mortality target (excluded for band targets, where
   they define the label).

`EDDISCHARGEHRS` is **blacklisted** as leakage; insurance/payer is kept as an
admission-time feature and as a fairness subgroup axis. See
`NTDB_REQUIRED_COLUMNS.md` for the full column list and provenance.

## Pipeline

```
raw NTDB CSVs ──trauma-build──▶ unified_train.parquet + unified_holdout.parquet
                                         │
                                   trauma-train
   per (target × phase × model × imputer × calibration × augmentation × missingness):
     load+filter → build target → select predictors → temporal split
     → pre-filter (NZV + correlated/duplicate drop)
     → encode/scale → impute (+optional imputer-check) → augment (SMOTE/ADASYN)
     → fit model → calibrate (binary) → evaluate → save ModelArtifact
```

Each run emits metric JSONs (train / calibration / test / temporal-holdout),
subgroup CSVs, diagnostic figures, and a self-contained deployable artifact.

The runner caches preprocessing across combos: encoders+imputer are fit once per
`(target, predictor_type, phase, inclusion, missingness, imputer, family)` group and
reused across all calibration+augmentation variants, and for binary targets the base
model is trained once per augmentation and reused across `none/platt/isotonic`
(only the post-hoc calibrator differs). This makes slow imputers ~9× cheaper and
binary training ~3× cheaper while producing **byte-identical** outputs to a
per-combo fit; disable with `--no-prep-cache`.

### Model families
Gradient boosting (`xgboost`, `lightgbm`, `catboost`), `flaml`, `tpot`,
`random_forest`, `logistic_l1`, `logistic_elasticnet`, deep tabular (`tabnet`,
`ft_transformer`, `tabpfn`), and feed-forward nets: `doshi_ffnn` (tabular) plus the faithful **`doshi_ffnn_icd`** / **`doshi_ffnn_icd_plus`** that vectorise the patient's ICD-10 code list to a multi-hot (Doshi 2024 ICD→severity, L3-only).
Boosting families consume NaN natively; the rest require an imputer.

### Imputation
`median_mode`, `mice`, `bagged_trees` (MissForest-style bagged trees),
`missforest`, `gradient_boosting` (HistGradientBoosting MICE), or `none`
(NaN-native models). Optional **imputer-check** masks
known cells and keeps only features the imputer reconstructs well (relative MAE
for numeric, balanced accuracy ≥ 0.6 for categorical).

### Feature selection
Embedded per family: L1 (logistic), tree regularisation (boosting/RF), attentive
masks (tabnet), L1 on the first layer (`doshi_ffnn`). A model-agnostic
**pre-filter** (near-zero-variance + near-perfectly-correlated/duplicate drop,
each removal logged) runs for every family and target.

### Clinical baselines
For mortality, **TRISS, RTS, MGAP, mREMS** are computed and overlaid on the
ROC/PR curves and reported as metrics, so model gains over standard scores are
explicit.

### Evaluation
Per target and partition: AUROC/AUPRC (binary) or balanced accuracy + per-class
recall (multiclass), Brier, calibration, confusion matrices, and SHAP. Three
row-slices — **full**, **baseline_complete** (all clinical baselines computable),
and **phase_complete** (≥1 real value per phase block) — each with their own
metric JSON and figures. Bootstrap confidence intervals are written to the JSONs
(`--bootstrap-ci`) and annotated on the binary ROC/PR legends by default.
Subgroup metrics are produced for gender, age, admission year, ethnicity, race,
mechanism, intent, and insurance.

### Deployment
Models are saved as a `ModelArtifact` bundling **encoders → scalers → imputer →
model → calibration**; `ModelArtifact.predict` applies the whole chain to raw
input, so serving (the FastAPI app in `app/`, or a Hugging Face Space) loads the
artifact rather than a bare estimator. `doshi_ffnn` needs `torch` in the image.

## Repository layout

```
src/trauma_ml/
  cli/            trauma-build, trauma-train, trauma-aggregate entry points
  ntdb_loader.py  per-year load, alias harmonisation, whitelist, derivations
  catalogue.py    single source of truth for phases / predictors / blacklist
  barell.py       Barell injury-matrix features
  targets.py      target construction (binary / ordinal_bands / binary_threshold)
  baselines.py    TRISS / RTS / MGAP / mREMS
  imputation.py   imputers + imputer-check (balanced-accuracy gate)
  models.py       model factories (incl. doshi_ffnn)
  trainer.py      orchestration: split, pre-filter, fit, calibrate, evaluate, slices
  evaluation.py   metrics + bootstrap CIs
  plots.py        ROC/PR, confusion, calibration, SHAP, slice figures
  persistence.py  ModelArtifact (deployable pipeline bundle)
  app/            FastAPI service + Dockerfile
slurms/           cluster jobs: 00 build, 10–34 train, 99 aggregate
config/           paths.yaml + experiment grids
analysis_figures.py   post-hoc paper-quality figures + stats across all targets
DIPC_RUNBOOK.md       operational guide (build → train → analyse → deploy)
PIPELINE_OPTIONS.md   every option/axis explained
NTDB_REQUIRED_COLUMNS.md   columns consumed + provenance
```

## Quick start

```bash
pip install -e .                    # + pip install torch for DL families
trauma-build --years 2019 2020 2021 2022 2024 --holdout-years 2024 --anonymize --verbose
trauma-train --dataset unified_train.parquet --holdout-dataset unified_holdout.parquet \
  --targets in_hospital_mortality --model-families xgboost \
  --phase-cutoffs "On-scene + ED arrival + In-hospital" \
  --imputers median_mode --calibrations isotonic --missingness-thresholds 50 --verbose
python analysis_figures.py          # from the outputs/ folder, after a sweep
```

On the cluster, submit `slurms/00_build_dataset.slurm`, then the `1x/2x/3x`
training jobs, then `slurms/99_aggregate.slurm`. See **DIPC_RUNBOOK.md** for the
full procedure and **PIPELINE_OPTIONS.md** for every configurable axis.
