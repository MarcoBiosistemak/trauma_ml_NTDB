#!/bin/bash
#SBATCH --job-name=trauma_iss_bin_k
#SBATCH --output=slurms/logs/train_iss_bin_k_%j.out
#SBATCH --error=slurms/logs/train_iss_bin_k_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=250G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# ISS_band_binary — "other" block, partition K (combos 900..1124).
#
# This is one of 6 parallel slurms splitting the 1350-combo `iss_bin_other`
# grid into 225-combo slices.  Every slice enumerates the SAME full grid in
# the SAME deterministic order and runs only its --start-id..--end-id range,
# so model_ids stay in their canonical sequence and no two slurms can pick the
# same id:
#    31g: iss_bin_other_0000..iss_bin_other_0224
#    31h: iss_bin_other_0225..iss_bin_other_0449
#    31i: iss_bin_other_0450..iss_bin_other_0674
#    31j: iss_bin_other_0675..iss_bin_other_0899
#    31k: iss_bin_other_0900..iss_bin_other_1124
#    31l: iss_bin_other_1125..iss_bin_other_1349
#
# Pre-existing iss_bin_other_NNNN models on disk (from the original
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
echo "[INFO] iss_bin_other partition K: combos 900..1124"
echo "[INFO] CPU-only slice (random_forest/tpot/tabnet need RAM, not GPU)"

# ── auto-resubmit on walltime ────────────────────────────────────────────
SELF="slurms/31k_train_iss_bin_other.slurm"
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
    --imputers      median_mode mice bagged_trees missforest gradient_boosting \
    --model-families logistic_l1 logistic_elasticnet random_forest tpot tabnet \
    --calibrations  none platt isotonic \
    --augmentations null smote adasyn \
    --missingness-thresholds 10 50 \
    --use-gpu       auto \
    --model-id-prefix iss_bin_other \
    --start-id      900 \
    --end-id        1124 \
    --no-shap \
    --no-aggregate \
    --n-jobs        20 \
    --verbose
