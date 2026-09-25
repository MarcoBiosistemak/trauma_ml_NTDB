#!/bin/bash
#SBATCH --job-name=rerun_mortality
#SBATCH --output=slurms/logs/rerun_mortality_%j.out
#SBATCH --error=slurms/logs/rerun_mortality_%j.err
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
#  Re-train the best model at L1 / L2 / L3 for mortality, but ONLY if its
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

TARGET=mortality
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

# ---- L1: lgb_none_0000 (grid index 0) ----------------------------------
rerun_if_missing L1 lgb_none_0000 0 \
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
    --targets       in_hospital_mortality \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" all_baseline_inputs \
    --imputers      none \
    --model-families lightgbm \
    --model-id-prefix lgb_none \
    --start-id      0 \
    --end-id        0 \
    --n-jobs        20

# ---- L2: xgb_none_0018 (grid index 18) ----------------------------------
rerun_if_missing L2 xgb_none_0018 18 \
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
    --targets       in_hospital_mortality \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" all_baseline_inputs \
    --imputers      none \
    --model-families xgboost \
    --model-id-prefix xgb_none \
    --start-id      18 \
    --end-id        18 \
    --n-jobs        20

# ---- L3: xgb_none_0036 (grid index 36) ----------------------------------
rerun_if_missing L3 xgb_none_0036 36 \
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
    --targets       in_hospital_mortality \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" all_baseline_inputs \
    --imputers      none \
    --model-families xgboost \
    --model-id-prefix xgb_none \
    --start-id      36 \
    --end-id        36 \
    --n-jobs        20

echo "[INFO] $(date) done for ${TARGET}"
