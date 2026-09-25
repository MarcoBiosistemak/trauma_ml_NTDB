#!/bin/bash
#SBATCH --job-name=trauma_tpt
#SBATCH --output=slurms/logs/train_tpt_%j.out
#SBATCH --error=slurms/logs/train_tpt_%j.err
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# Round 35: TPOT factory now subsamples training to 20% for the GA search
# and re-fits the winner on full data (see models.py::tpot).  Combined
# with cv=3 (was 5) and early_stop=3, per-combo wall-clock should drop
# from ~30-60 min to ~5-10 min.  Memory pressure also reduced because
# Dask scheduler holds less data per worker — dropped 300G → 200G.
# Walltime kept at xlong (8d) as safety margin; expected actual
# walltime for full 144-combo grid is now 12-24 h rather than 3-5 days.

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

# ── auto-resubmit on walltime ────────────────────────────────────────────
# The resume tracker skips any combo that already wrote overall__test.json,
# so re-running is cheap and lossless.  --signal delivers USR1 ~2 min before
# the time limit; we resubmit ONCE then exit cleanly.  Normal completion or
# scancel (SIGTERM) does NOT resubmit.
SELF="slurms/15_train_tpot.slurm"
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
    --targets       in_hospital_mortality \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" all_baseline_inputs \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting \
    --model-families tpot \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       never \
    --model-id-prefix tpt \
    --no-aggregate \
    --n-jobs        20 \
    --verbose

# (b) imputer=none NOT added — tpot cannot consume NaN.
