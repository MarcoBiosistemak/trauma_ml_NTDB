# trauma_ml

ML pipeline for trauma **mortality** and **injury-severity-band** prediction on
the National Trauma Data Bank (NTDB), admission years 2019–2024. Benchmarks
ICD-10-based machine-learning models against the classic TRISS / ISS / NISS
scores, following the design of Tran et al. 2022 (PLoS One,
doi:10.1371/journal.pone.0276624).

## What it does

* Builds a unified NTDB dataset across admission years, harmonising the
  year-to-year schema changes (including case-variant column collapse, e.g.
  `AGEYEARS` / `AGEyears` / `AgeYears` → one column).
* Trains a large grid of models per target, varying preprocessing axes
  (phase cutoff, imputer, calibration, augmentation, missingness threshold) so
  every model family is compared on equal footing.
* Computes ISS / NISS / TRISS baselines on the matching patient subset
  (`baseline_complete`) so the "did ML beat the score?" comparison is fair.
* Evaluates on a random test split AND an external temporal holdout (AY 2024).

## Targets

| Target | Task | Notes |
|---|---|---|
| `in_hospital_mortality` | binary | primary outcome; ISS/NISS/TRISS baselines apply |
| `ISS_band` | multiclass | injury-severity band; multiclass-only grid restrictions |
| `NISS_band` | multiclass | new ISS band; same restrictions |

## Model families

xgboost, lightgbm, catboost, random_forest, logistic (L1 + elastic net),
flaml (AutoML), tpot (AutoML), tabnet, tabpfn. Survival models (cox_ph,
random_survival_forest) are scaffolded but not currently run — NTDB lacks a
clean time-to-death.

## Install

```bash
pip install -e .                 # core
pip install -e ".[boosting]"     # + xgboost, lightgbm, catboost
pip install -e ".[automl]"       # + flaml
pip install -e ".[full]"         # + tabpfn, pytorch-tabnet, torch
pip install -e ".[all]"          # everything
```

Requires Python ≥ 3.10.

## CLI entry points

| Command | Purpose |
|---|---|
| `trauma-build` | build the unified train + holdout parquet from raw NTDB |
| `trauma-train` | run the experiment grid (one or more families) |
| `trauma-predict` | score new data with a saved model bundle |
| `trauma-aggregate` | walk `outputs/` and write `all_metrics.csv` |

## Imputation strategy (rounds 35–37)

Imputers performed poorly on NTDB, so the pipeline supports three regimes,
selectable per run:

| `--imputers` | `--imputer-check` | Behaviour |
|---|---|---|
| `median_mode` / `mice` | (absent) | impute everything (original behaviour) |
| `median_mode` / `mice` | present | impute, then **drop features the imputer can't reconstruct** on the calibration set (numeric MAE/std < 0.5, categorical accuracy > 0.7). If no feature survives, metrics are emitted as NaN. |
| `none` | n/a | **no imputation** — model consumes NaN directly. Only valid for NaN-native families: xgboost, lightgbm, catboost, flaml. |

Imputer quality is always evaluated on the **calibration set**, never the test
set, so the test split stays pristine for final evaluation. Per-feature
reconstruction quality (with a `keep` flag) is written to
`outputs/<target>/imputation_eval/<model_id>/imputation_eval_<method>.csv`.

## Output layout

```
outputs/
├── datasets/
│   ├── unified_train.parquet      AY 2019-2022, ~4.67M rows
│   └── unified_holdout.parquet    AY 2024, ~1.15M rows
└── <target>/                      mortality / iss_band / niss_band
    ├── models/<model_id>/
    │   ├── config.json            full run configuration (incl. imputer_check)
    │   └── artifact.pkl           pickled ModelArtifact (the fitted bundle)
    ├── metrics/<model_id>/
    │   └── overall__<partition>[__<subset>].json
    ├── plots/<model_id>/          ROC/PR, calibration, confusion matrices, SHAP
    ├── imputation_eval/<model_id>/
    ├── all_metrics.csv            (after trauma-aggregate)
    └── baseline_metrics.csv       ISS / NISS / TRISS
```

`config.json` keys: `model_id, family, actual_model_type, task, target_spec,
predictor_cols, numeric_cols, categorical_cols, target_inverse, config{…},
extras`. The nested `config` block holds `phase_cutoff, imputer_method,
imputer_check, calibration, data_augmentation, missingness_threshold,
model_family`. Metric columns follow `<metric>__<partition>[__<subset>]`, e.g.
`AUROC__random_test`, `AUPRC__holdout_external__baseline_complete`.

## Running on the DIPC cluster

See [`slurms/README.md`](slurms/README.md) for the full slurm catalogue,
resource sizing, the round-37 multi-invocation layout, and the standard
deploy/resubmit cycle.

## Source layout

```
src/trauma_ml/
├── ntdb_loader.py    raw NTDB → unified parquet (schema harmonisation, dedup)
├── catalogue.py      variable catalogue + alias resolution
├── inclusion.py      cohort inclusion strategies
├── targets.py        target construction (mortality, ISS_band, NISS_band)
├── barell.py         Barell ICD-10 injury matrix features
├── comorbidities.py  canonical comorbidity flags
├── baselines.py      ISS / NISS / TRISS computation
├── splitting.py      train / calibration / test split (year-stratified)
├── imputation.py     imputers + imputation evaluation + feature gating
├── models.py         model factories (one per family)
├── trainer.py        the per-run orchestrator (TrainerConfig, run())
├── evaluation.py     metric computation per partition / subset
├── plots.py          ROC/PR, calibration, confusion, SHAP figures
├── persistence.py    ModelArtifact save/load
├── cli/              command-line entry points
└── app/              FastAPI serving scaffold
```

## Reference

Tran Z, et al. *Machine learning models outperform the Trauma and Injury
Severity Score (TRISS) in predicting mortality after trauma.* PLoS One 2022.
doi:10.1371/journal.pone.0276624
