#!/bin/bash
#SBATCH --job-name=trauma_niss_tpt
#SBATCH --output=slurms/logs/train_niss_tpt_%j.out
#SBATCH --error=slurms/logs/train_niss_tpt_%j.err
#SBATCH --partition=general
#SBATCH --qos=xlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=8-00:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma-train: TPOT AutoML on NISS_band (multiclass).
# Round 36: added FLAML/TPOT to band targets.
#
# TPOT scoring is balanced_accuracy on multiclass targets (round 36
# fix in models.py) — plain accuracy would let TPOT win by predicting
# the majority class given NISS_band's 1.7M/1.5M/261k/404k imbalance.
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

trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       NISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" \
    --imputers      median_mode mice \
    --model-families tpot \
    --calibrations  none \
    --augmentations null \
    --missingness-thresholds 25 50 85 \
    --use-gpu       never \
    --model-id-prefix niss_tpt \
    --no-aggregate \
    --n-jobs        20 \
    --verbose

# ─── Round 37: imputer_check + imputer=none in SEPARATE id-namespaces ──
# Appended so they never renumber the existing niss_tpt_* models above.
# (a) imputer_check=True — drop features the imputer can't reconstruct
#     on the calibration set.  Prefix niss_tpt_ic.
trauma-train \
    --dataset       outputs/datasets/unified_train.parquet \
    --holdout-dataset outputs/datasets/unified_holdout.parquet \
    --targets       NISS_band \
    --inclusion-strategies "Tran NTDB" \
    --predictor-types "Tran NTDB" \
    --phase-cutoffs "On-scene" "On-scene + ED arrival" \
    --imputers      median_mode mice \
    --imputer-check \
    --model-families tpot \
    --calibrations  none \
    --augmentations null \
    --missingness-thresholds 25 50 85 \
    --use-gpu       never \
    --model-id-prefix niss_tpt_ic \
    --no-aggregate \
    --n-jobs        20 \
    --verbose

# (b) imputer=none NOT added — tpot cannot consume NaN.
