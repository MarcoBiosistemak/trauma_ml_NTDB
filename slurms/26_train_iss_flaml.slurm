#!/bin/bash
#SBATCH --job-name=trauma_iss_flm
#SBATCH --output=slurms/logs/train_iss_flm_%j.out
#SBATCH --error=slurms/logs/train_iss_flm_%j.err
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train: FLAML AutoML on ISS_band (multiclass).
# Round 36: added FLAML/TPOT to band targets for completeness vs mortality.
#
# FLAML's AutoML supports task='multiclass' natively.  Each combo runs
# FLAML's internal time-budgeted search (default 300s = 5 min) which
# usually picks LightGBM or CatBoost as the winner.  Total per-combo
# wall-clock ≈ 5-10 min including preprocessing.
#
# Note FLAML does NOT save a SHAP plot through the standard pipeline —
# its sklearn-compat wrapper hides the underlying winning estimator from
# SHAP's TreeExplainer.  See round-36 notes for inspecting FLAML
# feature importances via artifact.pkl after the fact.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"

trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       ISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" \
    --imputers      median_mode mice \
    --model-families flaml \
    --calibrations  none \
    --augmentations null \
    --missingness-thresholds 25 50 85 \
    --use-gpu       never \
    --model-id-prefix iss_flm \
    --no-aggregate \
    --n-jobs        20 \
    --verbose

# ─── Round 37: imputer_check + imputer=none in SEPARATE id-namespaces ──
# Appended so they never renumber the existing iss_flm_* models above.
# (a) imputer_check=True — drop features the imputer can't reconstruct
#     on the calibration set.  Prefix iss_flm_ic.
trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       ISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" \
    --imputers      median_mode mice \
    --imputer-check \
    --model-families flaml \
    --calibrations  none \
    --augmentations null \
    --missingness-thresholds 25 50 85 \
    --use-gpu       never \
    --model-id-prefix iss_flm_ic \
    --no-aggregate \
    --n-jobs        20 \
    --verbose

# (b) imputer=none — flaml handles NaN natively; no imputation.
#     imputer_check irrelevant here.  Prefix iss_flm_none.
trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       ISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" \
    --imputers      none \
    --model-families flaml \
    --calibrations  none \
    --augmentations null \
    --missingness-thresholds 25 50 85 \
    --use-gpu       never \
    --model-id-prefix iss_flm_none \
    --no-aggregate \
    --n-jobs        20 \
    --verbose
