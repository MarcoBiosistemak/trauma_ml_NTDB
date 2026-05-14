#!/bin/bash
#SBATCH --job-name=trauma_cb
#SBATCH --output=slurms/logs/train_cb_%j.out
#SBATCH --error=slurms/logs/train_cb_%j.err
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train on the CATBOOST family.
# Model IDs prefixed cb_*.  Each job writes to non-overlapping
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
nvidia-smi || echo "[WARN] nvidia-smi not available"

trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       in_hospital_mortality \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" all_baseline_inputs \
    --imputers      median_mode mice \
    --model-families catboost \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       auto \
    --model-id-prefix cb \
    --no-aggregate \
    --n-jobs        12 \
    --verbose
