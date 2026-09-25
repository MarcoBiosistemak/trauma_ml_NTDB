#!/bin/bash
#SBATCH --job-name=trauma_iss_tpt
#SBATCH --output=slurms/logs/train_iss_tpt_%j.out
#SBATCH --error=slurms/logs/train_iss_tpt_%j.err
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train: TPOT AutoML on ISS_band (multiclass).
# Round 36: added FLAML/TPOT to band targets.
#
# TPOT scoring is balanced_accuracy on multiclass targets (round 36
# fix in models.py) — plain accuracy would let TPOT win by predicting
# the majority class given ISS_band's 1.7M/1.5M/261k/404k imbalance.
#
# Round 35 settings carry over: GA on 20% subsample (288k rows),
# winner re-fit on full data, cv=3, max_time_mins=15, early_stop=3.
# Per-combo wall-clock ≈ 10-20 min; full 12-combo grid ≈ 4-6 h.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"

# ── auto-resubmit on walltime ────────────────────────────────────────────
# The resume tracker skips any combo that already wrote overall__test.json,
# so re-running is cheap and lossless.  --signal delivers USR1 ~2 min before
# the time limit; we resubmit ONCE then exit cleanly.  Normal completion or
# scancel (SIGTERM) does NOT resubmit.
SELF="slurms/27_train_iss_tpot.slurm"
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
    --model-families tpot \
    --calibrations  none \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix iss_tpt \
    --no-aggregate \
    --n-jobs        20 \
    --verbose

# (b) imputer=none NOT added — tpot cannot consume NaN.
