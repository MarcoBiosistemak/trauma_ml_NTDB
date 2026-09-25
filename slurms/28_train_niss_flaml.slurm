#!/bin/bash
#SBATCH --job-name=trauma_niss_flm
#SBATCH --output=slurms/logs/train_niss_flm_%j.out
#SBATCH --error=slurms/logs/train_niss_flm_%j.err
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train: FLAML AutoML on NISS_band (multiclass).
# Round 36: added FLAML/TPOT to band targets for completeness vs mortality.
#
# FLAML's AutoML supports task='multiclass' natively.  Each combo runs
# FLAML's internal time-budgeted search (default 300s = 5 min) which
# usually picks LightGBM or CatBoost as the winner.  Total per-combo
# wall-clock ≈ 5-10 min including preprocessing.
#
# Note FLAML does NOT save a SHAP plot through the standard pipeline —
# its sklearn-compat wrapper hides the underlying winning estimator from
# SHAP's TreeExplainer.  See round-36 notes for inspecting FLAML
# feature importances via artifact.pkl after the fact.

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
SELF="slurms/28_train_niss_flaml.slurm"
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
    --targets       NISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting \
    --model-families flaml \
    --calibrations  none \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix niss_flm \
    --no-aggregate \
    --n-jobs        20 \
    --verbose

# (b) imputer=none — flaml handles NaN natively; no imputation.
#     imputer_check irrelevant here.  Prefix niss_flm_none.
run trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       NISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" \
    --imputers      none \
    --model-families flaml \
    --calibrations  none \
    --augmentations null \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix niss_flm_none \
    --no-aggregate \
    --n-jobs        20 \
    --verbose
