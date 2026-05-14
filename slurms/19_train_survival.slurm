#!/bin/bash
#SBATCH --job-name=trauma_surv
#SBATCH --output=slurms/logs/train_surv_%j.out
#SBATCH --error=slurms/logs/train_surv_%j.err
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train on the SURVIVAL family.
# Model IDs prefixed surv_*.  Each job writes to non-overlapping
# outputs paths so all training jobs can run in parallel.
# --no-aggregate: per-job all_metrics.csv is skipped; the final
# unified all_metrics.csv comes from 99_aggregate.slurm after all
# training jobs return success.
#
# Round 17: UNIFIED grid across every family — identical axes for
# fair head-to-head comparison.  See DIPC_RUNBOOK.md for the full
# explanation of why these specific values were chosen.
#
# DISABLED in submit_all.sh — survival end-to-end requires
# (event, time) target tuples that the trainer doesn't yet
# produce.  Re-enable when targets.py::_survival_time builder
# and trainer's survival metrics path are implemented.

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
    --targets       survival_time \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" iss_only niss_only triss_inputs all_baseline_inputs \
    --imputers      median_mode mice \
    --model-families cox_ph random_survival_forest \
    --calibrations  none \
    --augmentations null \
    --missingness-thresholds 25 50 85 \
    --use-gpu       never \
    --model-id-prefix surv \
    --no-aggregate \
    --n-jobs        16 \
    --verbose
