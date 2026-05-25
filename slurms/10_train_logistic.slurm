#!/bin/bash
#SBATCH --job-name=trauma_lgr
#SBATCH --output=slurms/logs/train_lgr_%j.out
#SBATCH --error=slurms/logs/train_lgr_%j.err
#SBATCH --partition=general
#SBATCH --qos=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=8-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# Round 33: walltime expanded from regular (1d) → xlong (8d) after
# the 1-day walltime was hit mid-MICE on lgr_0062.  Full preprocessing
# grid kept intact: 288 combos × ~10-20 min/combo with vectorised
# encoding + MICE on 2.75M rows fits well inside 8 days.  Memory raised
# 96 → 160 GB to absorb MICE's iterative chain on the larger train set.
#
# Note: --no-shap is intentionally NOT set — KernelExplainer is what
# made the original run slow, but it's still ~1 min/model and we want
# the explanations.  If a future run hits the 8-day ceiling, this is
# the first knob to flip.

# trauma-train on the LOGISTIC family.
# Model IDs prefixed lgr_*.  Each job writes to non-overlapping
# outputs paths so all training jobs can run in parallel.
# --no-aggregate: per-job all_metrics.csv is skipped; the final
# unified all_metrics.csv comes from 99_aggregate.slurm after all
# training jobs return success.
#
# Round 17: UNIFIED grid across every family — identical axes for
# fair head-to-head comparison.  See DIPC_RUNBOOK.md for the full
# explanation of why these specific values were chosen.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"

trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       in_hospital_mortality \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" all_baseline_inputs \
    --imputers      median_mode mice \
    --model-families logistic_l1 logistic_elasticnet \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix lgr \
    --no-aggregate \
    --n-jobs        16 \
    --verbose

# ─── Round 37: imputer_check + imputer=none in SEPARATE id-namespaces ──
# Appended so they never renumber the existing lgr_* models above.
# (a) imputer_check=True — drop features the imputer can't reconstruct
#     on the calibration set.  Prefix lgr_ic.
trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       in_hospital_mortality \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" all_baseline_inputs \
    --imputers      median_mode mice \
    --imputer-check \
    --model-families logistic_l1 logistic_elasticnet \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix lgr_ic \
    --no-aggregate \
    --n-jobs        16 \
    --verbose

# (b) imputer=none NOT added — logistic_l1 cannot consume NaN.
