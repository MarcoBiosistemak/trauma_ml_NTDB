#!/bin/bash
#SBATCH --job-name=trauma_build
#SBATCH --output=slurms/logs/build_%j.out
#SBATCH --error=slurms/logs/build_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# Build the unified parquets (train + holdout) from the raw NTDB CSVs.
# Run once before any training jobs.

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID}"
which trauma-build
python -c "import sys; print('Python:', sys.executable)"

trauma-build \
    --anonymize \
    --verbose \
    --years 2019 2020 2021 2022 2024 \
    --holdout-years 2024
