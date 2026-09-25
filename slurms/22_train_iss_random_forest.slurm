#!/bin/bash
#SBATCH --job-name=trauma_iss_rf
#SBATCH --output=slurms/logs/train_iss_rf_%j.out
#SBATCH --error=slurms/logs/train_iss_rf_%j.err
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=250G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train on the ISS_RF family for the ISS_BAND target.
# Multiclass classification — predicting ISS/NISS bands rather than mortality.
# Outputs land under outputs/iss_band/ (NOT outputs/) so they don't collide
# with the mortality run's outputs/{models,metrics,plots}/.
#
# Round 19/53: multiclass (4-band) slurms.  Calibration (Platt/isotonic) is
# binary-only so it stays 'none' here, but SMOTE/ADASYN augmentation DOES work
# for multiclass and is enabled on the main imputed block (it is auto-skipped
# for the imputer=none block, where the matrix still has NaN).  Phase cutoffs
# include On-scene + ED arrival + In-hospital (ISS/NISS are excluded as
# predictors for band targets, so the in-hospital cutoff is safe).  Baseline
# scores (ISS/NISS/TRISS/RTS/MGAP/mREMS) are NOT computed for band targets —
# the bands ARE derived from ISS/NISS, so there is nothing to compare against.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"

# ── auto-resubmit on walltime ────────────────────────────────────────────
# The resume tracker skips any combo that already wrote overall__test.json,
# so re-running is cheap and lossless.  --signal delivers USR1 ~2 min before
# the time limit; we resubmit ONCE then exit cleanly.  Normal completion or
# scancel (SIGTERM) does NOT resubmit.
SELF="slurms/22_train_iss_random_forest.slurm"
WALL_SECONDS=172800            # keep in sync with --time (2d)
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
    --targets       ISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting \
    --model-families random_forest \
    --calibrations  none \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix iss_rf \
    --no-aggregate \
    --no-shap \
    --n-jobs        12 \
    --verbose

# Round 33: --no-shap added — same SHAP segfault as the mortality
# random_forest slurm.  See 16_train_random_forest.slurm for details.

# (b) imputer=none NOT added — random_forest cannot consume NaN.
