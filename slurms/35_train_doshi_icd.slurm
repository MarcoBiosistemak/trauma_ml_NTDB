#!/bin/bash
#SBATCH --job-name=trauma_doshi_icd
#SBATCH --output=slurms/logs/train_doshi_icd_%j.out
#SBATCH --error=slurms/logs/train_doshi_icd_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --mem=90G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# Faithful Doshi (2024) ICD->severity feed-forward network.
# Input = a MULTI-HOT vector of the patient's ICD-10 diagnosis codes (built
# inside the model from the ICD_DIAG_CODES column).  ICD codes are a-posteriori,
# so these runs are IN-HOSPITAL (L3) ONLY.
#
#   doshi_ffnn_icd       -> ICD codes ONLY (the paper's direct ICD->ISS model)
#   doshi_ffnn_icd_plus  -> ICD multi-hot + the other L3 numeric features (gain ablation)
#
# Targets: in_hospital_mortality, ISS_band_binary (ISS>=16), NISS_band_binary.
# Model IDs prefixed doshi_icd_* / doshi_icdplus_*.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi || echo "[WARN] nvidia-smi not available"

# (a) FAITHFUL: ICD codes ONLY.  No numeric features -> imputer 'none'.
# ── auto-resubmit on walltime ────────────────────────────────────────────
# The resume tracker skips any combo that already wrote overall__test.json,
# so re-running is cheap and lossless.  --signal delivers USR1 ~2 min before
# the time limit; we resubmit ONCE then exit cleanly.  Normal completion or
# scancel (SIGTERM) does NOT resubmit.
SELF="slurms/35_train_doshi_icd.slurm"
WALL_SECONDS=86400            # keep in sync with --time (1d)
_resubmitted=0
_on_timeout() {
  if [ "${_resubmitted}" -eq 0 ]; then
    _resubmitted=1
    echo "[INFO] $(date) nearing walltime -> resubmitting ${SELF} to resume"
    sbatch "${SELF}" || true
  fi
  exit 0
}
trap _on_timeout USR1
# pure-bash watchdog: signals THIS script ~4 min before the walltime so we
# resubmit (the resume tracker skips finished combos).  No --signal needed.
( sleep $(( WALL_SECONDS - 240 )); kill -USR1 $$ 2>/dev/null ) &
_watchdog=$!
trap 'kill ${_watchdog} 2>/dev/null || true' EXIT   # stop watchdog on normal finish
run() { "$@" & wait $!; }   # background+wait so the USR1 trap fires promptly

run trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       in_hospital_mortality ISS_band_binary NISS_band_binary \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene + ED arrival + In-hospital" \
    --imputers      none \
    --model-families doshi_ffnn_icd \
    --calibrations  none platt isotonic \
    --augmentations null \
    --missingness-thresholds 50 \
    --use-gpu       auto \
    --model-id-prefix doshi_icd \
    --no-aggregate \
    --n-jobs        12 \
    --verbose

# (b) ABLATION: ICD multi-hot + other L3 numeric features -> real imputation.
run trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       in_hospital_mortality ISS_band_binary NISS_band_binary \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene + ED arrival + In-hospital" \
    --imputers      median_mode mice bagged_trees \
    --model-families doshi_ffnn_icd_plus \
    --calibrations  none platt isotonic \
    --augmentations null \
    --missingness-thresholds 10 50 \
    --use-gpu       auto \
    --model-id-prefix doshi_icdplus \
    --no-aggregate \
    --n-jobs        12 \
    --verbose
