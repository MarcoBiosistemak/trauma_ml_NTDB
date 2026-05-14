#!/bin/bash
#SBATCH --job-name=trauma_iss_lgb
#SBATCH --output=slurms/logs/train_iss_lgb_%j.out
#SBATCH --error=slurms/logs/train_iss_lgb_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train on the ISS_LGB family for the ISS_BAND target.
# Multiclass classification — predicting ISS/NISS bands rather than mortality.
# Outputs land under outputs/iss_band/ (NOT outputs/) so they don't collide
# with the mortality run's outputs/{models,metrics,plots}/.
#
# Round 19: multiclass slurms.  Grid is narrower than the mortality slurms
# because calibration (Platt/isotonic) and augmentation (SMOTE/ADASYN) do
# not generalise cleanly to multiclass.  Baseline scores (ISS/NISS/TRISS)
# are NOT computed by the aggregator for non-mortality targets — there's
# nothing to compare against since the bands ARE derived from those scores.

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
    --targets       ISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" \
    --imputers      median_mode mice \
    --model-families lightgbm \
    --calibrations  none \
    --augmentations null \
    --missingness-thresholds 25 50 85 \
    --use-gpu       auto \
    --model-id-prefix iss_lgb \
    --no-aggregate \
    --n-jobs        12 \
    --verbose
