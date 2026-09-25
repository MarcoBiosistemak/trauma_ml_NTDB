#!/bin/bash
#SBATCH --job-name=trauma_niss_bin_i
#SBATCH --output=slurms/logs/train_niss_bin_i_%j.out
#SBATCH --error=slurms/logs/train_niss_bin_i_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=250G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# NISS_band_binary — "other" block, partition I (combos 450..674).
#
# This is one of 6 parallel slurms splitting the 1350-combo `niss_bin_other`
# grid into 225-combo slices.  Every slice enumerates the SAME full grid in
# the SAME deterministic order and runs only its --start-id..--end-id range,
# so model_ids stay in their canonical sequence and no two slurms can pick the
# same id:
#    32g: niss_bin_other_0000..niss_bin_other_0224
#    32h: niss_bin_other_0225..niss_bin_other_0449
#    32i: niss_bin_other_0450..niss_bin_other_0674
#    32j: niss_bin_other_0675..niss_bin_other_0899
#    32k: niss_bin_other_0900..niss_bin_other_1124
#    32l: niss_bin_other_1125..niss_bin_other_1349
#
# Pre-existing niss_bin_other_NNNN models on disk (from the original
# 32_train_niss_band_binary.slurm) keep their ids — the resume tracker
# skips any combo that already wrote overall__test.json, so partial work is
# preserved automatically.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"
echo "[INFO] niss_bin_other partition I: combos 450..674"
echo "[INFO] CPU-only slice (random_forest/tpot/tabnet need RAM, not GPU)"

# ── auto-resubmit on walltime ────────────────────────────────────────────
SELF="slurms/32i_train_niss_bin_other.slurm"
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
    --targets       NISS_band_binary \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital" \
    --imputers      median_mode mice bagged_trees missforest gradient_boosting \
    --model-families logistic_l1 logistic_elasticnet random_forest tpot tabnet \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       auto \
    --model-id-prefix niss_bin_other \
    --start-id      450 \
    --end-id        674 \
    --no-shap \
    --no-aggregate \
    --n-jobs        20 \
    --verbose
