#!/bin/bash
#SBATCH --job-name=rerun_iss_bin
#SBATCH --output=slurms/logs/rerun_iss_bin_%j.out
#SBATCH --error=slurms/logs/rerun_iss_bin_%j.err
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=250G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# ==========================================================================
#  Re-train the best model at L1 / L2 / L3 for iss_band_binary, but ONLY if its
#  artifact.pkl is missing (i.e. the pruner removed it).
#
#  EXACTNESS
#  ---------
#  model_id is <prefix>_<grid_index>. run_experiments.py builds the FULL grid
#  from the flags below and only then filters by --start-id/--end-id, so
#  restricting to a single index reproduces exactly the combo that produced
#  that model_id originally. The flag block below is copied verbatim from the
#  slurm that created these models, so the grid -- and therefore the index ->
#  configuration mapping -- is identical. With random_state=42 threaded
#  through the split, imputer, augmentation and estimator, the CPU families
#  reproduce bit-for-bit; GPU families (doshi_ffnn) may differ in the last
#  decimals because of cuDNN non-determinism.
#
#  The metrics sentinel is deleted first: the resume tracker keys on
#  metrics/<id>/overall__test.json, so a model whose artifact was pruned still
#  looks "complete" and would otherwise be skipped.
# ==========================================================================

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

TARGET=iss_band_binary
echo "[INFO] $(date) re-run check for ${TARGET} on $(hostname)"

rerun_if_missing () {
  local phase="$1" mid="$2" idx="$3"; shift 3
  local art="outputs/${TARGET}/models/${mid}/artifact.pkl"
  if [ -f "${art}" ]; then
    echo "[SKIP] ${phase} ${mid}: artifact present ($(du -h "${art}" | cut -f1))"
    return 0
  fi
  echo "[RERUN] ${phase} ${mid}: artifact missing -> retraining grid index ${idx}"
  rm -rf "outputs/${TARGET}/metrics/${mid}" "outputs/${TARGET}/models/${mid}"
  "$@"
  if [ -f "${art}" ]; then
    echo "[OK] ${phase} ${mid} rebuilt"
  else
    echo "[FAIL] ${phase} ${mid} not rebuilt - check the log above"
  fi
}

# ---- L1: iss_bin_boost_0360 (grid index 360) ----------------------------------
rerun_if_missing L1 iss_bin_boost_0360 360 \
  trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       auto \
    --no-aggregate \
    --verbose \
    --targets       ISS_band_binary \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting none \
    --model-families xgboost lightgbm catboost flaml \
    --model-id-prefix iss_bin_boost \
    --start-id      360 \
    --end-id        360 \
    --n-jobs        20

# ---- L2: iss_bin_boost_0792 (grid index 792) ----------------------------------
rerun_if_missing L2 iss_bin_boost_0792 792 \
  trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       auto \
    --no-aggregate \
    --verbose \
    --targets       ISS_band_binary \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting none \
    --model-families xgboost lightgbm catboost flaml \
    --model-id-prefix iss_bin_boost \
    --start-id      792 \
    --end-id        792 \
    --n-jobs        20

# ---- L3: iss_bin_boost_1209 (grid index 1209) ----------------------------------
rerun_if_missing L3 iss_bin_boost_1209 1209 \
  trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       auto \
    --no-aggregate \
    --verbose \
    --targets       ISS_band_binary \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting none \
    --model-families xgboost lightgbm catboost flaml \
    --model-id-prefix iss_bin_boost \
    --start-id      1209 \
    --end-id        1209 \
    --n-jobs        20

echo "[INFO] $(date) done for ${TARGET}"
