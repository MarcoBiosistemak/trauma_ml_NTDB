#!/bin/bash
#SBATCH --job-name=trauma_iss_bin_f
#SBATCH --output=slurms/logs/train_iss_bin_f_%j.out
#SBATCH --error=slurms/logs/train_iss_bin_f_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=250G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# ISS_band_binary — "boost" block, partition F (combos 1080..1295).
#
# This is one of 6 parallel slurms splitting the 1296-combo `iss_bin_boost`
# grid into 216-combo slices.  Every slice enumerates the SAME full grid in
# the SAME deterministic order and runs only its --start-id..--end-id range,
# so model_ids stay in their canonical sequence and no two slurms can pick the
# same id:
#    31a: iss_bin_boost_0000..iss_bin_boost_0215
#    31b: iss_bin_boost_0216..iss_bin_boost_0431
#    31c: iss_bin_boost_0432..iss_bin_boost_0647
#    31d: iss_bin_boost_0648..iss_bin_boost_0863
#    31e: iss_bin_boost_0864..iss_bin_boost_1079
#    31f: iss_bin_boost_1080..iss_bin_boost_1295
#
# Pre-existing iss_bin_boost_NNNN models on disk (from the original
# 31_train_iss_band_binary.slurm) keep their ids — the resume tracker
# skips any combo that already wrote overall__test.json, so partial work is
# preserved automatically.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"
echo "[INFO] iss_bin_boost partition F: combos 1080..1295"
echo "[INFO] CPU slice (xgboost/lightgbm/catboost/flaml fall back via --use-gpu auto)"

# ── auto-resubmit on walltime ────────────────────────────────────────────
SELF="slurms/31f_train_iss_bin_boost.slurm"
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
( sleep $(( WALL_SECONDS - 240 )); kill -USR1 $$ 2>/dev/null ) &
_watchdog=$!
trap 'kill ${_watchdog} 2>/dev/null || true' EXIT
run() { "$@" & wait $!; }

run trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       ISS_band_binary \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting none \
    --model-families xgboost lightgbm catboost flaml \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       auto \
    --model-id-prefix iss_bin_boost \
    --start-id      1080 \
    --end-id        1295 \
    --no-aggregate \
    --n-jobs        20 \
    --verbose
