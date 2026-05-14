#!/bin/bash
#SBATCH --job-name=trauma_tpf
#SBATCH --output=slurms/logs/train_tpf_%j.out
#SBATCH --error=slurms/logs/train_tpf_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train on the TABPFN family.
# Model IDs prefixed tpf_*.  Each job writes to non-overlapping
# outputs paths so all training jobs can run in parallel.
# --no-aggregate: per-job all_metrics.csv is skipped; the final
# unified all_metrics.csv comes from 99_aggregate.slurm after all
# training jobs return success.
#
# Round 17: UNIFIED grid across every family — identical axes for
# fair head-to-head comparison.  See DIPC_RUNBOOK.md for the full
# explanation of why these specific values were chosen.
#
# TabPFN v2.x is hard-capped at ~10,000 training rows.
# --sample-fraction 0.005 ≈ 12k rows from the 4.67M pool.
# Each fit takes ~30 s after subsampling, so the full unified
# grid (378 fits) finishes in ~6 hours — the only family for
# which the unified grid is essentially free.

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
    --model-families tabpfn \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       force \
    --sample-fraction 0.005 \
    --model-id-prefix tpf \
    --no-aggregate \
    --n-jobs        8 \
    --verbose
