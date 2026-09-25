#!/bin/bash
#SBATCH --job-name=trauma_tnt_f
#SBATCH --output=slurms/logs/train_tnt_f_%j.out
#SBATCH --error=slurms/logs/train_tnt_f_%j.err
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

# TabNet mortality — partition F (combos 90..107).
#
# This is one of 8 parallel slurms (18a..18h) splitting the full 144-combo
# tnt grid into 45-combo slices (360-combo main grid: 5 imputers x 4 phases x 3 calib x 2 missingness x 3 augment).  Each slurm uses the ORIGINAL --model-id-prefix
# `tnt` and slices the grid via --start-id / --end-id, so model_ids stay in
# their canonical sequence:
#    18a: tnt_0000..tnt_0017 (and tnt_ic_0000..tnt_ic_0017)
#    18b: tnt_0018..tnt_0035 (and tnt_ic_0018..tnt_ic_0035)
#    18c: tnt_0036..tnt_0053 (and tnt_ic_0036..tnt_ic_0053)
#    18d: tnt_0054..tnt_0071 (and tnt_ic_0054..tnt_ic_0071)
#    18e: tnt_0072..tnt_0089 (and tnt_ic_0072..tnt_ic_0089)
#    18f: tnt_0090..tnt_0107 (and tnt_ic_0090..tnt_ic_0107)
#    18g: tnt_0108..tnt_0125 (and tnt_ic_0108..tnt_ic_0125)
#    18h: tnt_0126..tnt_0143 (and tnt_ic_0126..tnt_ic_0143)
#
# Every parallel slurm enumerates the SAME full grid in the SAME order
# (deterministic itertools.product), then runs only its --start-id..--end-id
# slice.  No two slurms can pick the same model_id, so they write to
# disjoint directories on disk — safe to run all 8 in parallel.
#
# Existing tnt_NNNN models already on disk (from the original sequential
# 18_train_tabnet.slurm) keep their ids.  The resume tracker skips any
# combo whose overall__test.json already exists, so partial pre-existing
# work is preserved automatically.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"
echo "[INFO] Partition F: combos 90..107 (model_ids tnt_0090..tnt_0107)"
nvidia-smi || echo "[WARN] nvidia-smi not available"

# ── (a) base: impute + train, slice combos 90..107 of the FULL grid ────
# ── auto-resubmit on walltime ────────────────────────────────────────────
# The resume tracker skips any combo that already wrote overall__test.json,
# so re-running is cheap and lossless.  --signal delivers USR1 ~2 min before
# the time limit; we resubmit ONCE then exit cleanly.  Normal completion or
# scancel (SIGTERM) does NOT resubmit.
SELF="slurms/18f_train_tabnet.slurm"
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
    --targets       in_hospital_mortality \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" all_baseline_inputs \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting \
    --model-families tabnet \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       force \
    --model-id-prefix tnt \
    --start-id      225 \
    --end-id        269 \
    --no-aggregate \
    --n-jobs        8 \
    --verbose

# ── (b) imputer_check: same slice, _ic prefix ────────────────────────────────
