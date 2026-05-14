#!/bin/bash
#SBATCH --job-name=trauma_tpt
#SBATCH --output=slurms/logs/train_tpt_%j.out
#SBATCH --error=slurms/logs/train_tpt_%j.err
#SBATCH --partition=general
#SBATCH --qos=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=300G
#SBATCH --time=8-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# Round 33: TPOT's Dask scheduler was OOM-killed mid-grid on the
# 120 GB / 2-day allocation ("Event loop was unresponsive in Nanny
# for 114s" → SLURM kill).  Memory raised 120 → 300 GB (uses the
# fatter nodes via the regular partition), walltime extended to
# 8 days.  Full preprocessing grid (4 cutoffs × 2 imputers × 3 cals
# × 3 augs × 2 missingness = 144 combos) kept intact so TPOT
# competes against all other families on equal preprocessing
# footing.  Per-combo time is dominated by TPOT's internal AutoML
# search; expect ~30-60 min per combo at this scale.

# trauma-train on the TPOT family.
# Model IDs prefixed tpt_*.  Each job writes to non-overlapping
# outputs paths so all training jobs can run in parallel.
# --no-aggregate: per-job all_metrics.csv is skipped; the final
# unified all_metrics.csv comes from 99_aggregate.slurm after all
# training jobs return success.
#
# Round 17: UNIFIED grid across every family — identical axes for
# fair head-to-head comparison.  See DIPC_RUNBOOK.md for the full
# explanation of why these specific values were chosen.
#
# TPOT genetic search is HARD time-budgeted at 5 min per fit
# (default is unlimited).  With the unified 378-combo grid,
# 5 min × 378 = 31.5 h ≈ fits in 2-day walltime.
# If you want deeper TPOT search per fit, narrow the grid first
# (e.g. drop missingness 25/85, only 50) before bumping the
# per-fit budget in trauma_ml/models.py::tpot factory.

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
    --model-families tpot \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix tpt \
    --no-aggregate \
    --n-jobs        20 \
    --verbose
