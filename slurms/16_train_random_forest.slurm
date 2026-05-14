#!/bin/bash
#SBATCH --job-name=trauma_rf
#SBATCH --output=slurms/logs/train_rf_%j.out
#SBATCH --error=slurms/logs/train_rf_%j.err
#SBATCH --partition=general
#SBATCH --qos=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train on the RF family.
# Model IDs prefixed rf_*.  Each job writes to non-overlapping
# outputs paths so all training jobs can run in parallel.
# --no-aggregate: per-job all_metrics.csv is skipped; the final
# unified all_metrics.csv comes from 99_aggregate.slurm after all
# training jobs return success.
#
# Round 17: UNIFIED grid across every family — identical axes for
# fair head-to-head comparison.  See DIPC_RUNBOOK.md for the full
# explanation of why these specific values were chosen.
#
# Random Forest is the slowest non-AutoML family on this data
# size (~10 min per fit on 4.67M rows × 30 feats × 500 trees).
# 378 fits × 10 min = 63 h → 4-day walltime to be safe.

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
    --model-families random_forest \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix rf \
    --no-aggregate \
    --no-shap \
    --n-jobs        20 \
    --verbose

# Round 33: --no-shap added.  SHAP's TreeExplainer C extension segfaults
# on random_forest models with 100+ trees on glibc 2.17 (DIPC's libc
# version).  Symptom: "Violación de segmento" mid-grid after one model
# completes its plots — kills the slurm and every subsequent combo.
# Random forest still produces all metrics, ROC/PR plots, confusion
# matrices, and subgroup analyses; we lose only the SHAP beeswarm/bar
# for this family.  Tree-based importance is still available via
# `feature_importances_` in the saved model bundle if needed.
