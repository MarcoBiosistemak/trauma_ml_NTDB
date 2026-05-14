#!/bin/bash
#SBATCH --job-name=trauma_agg
#SBATCH --output=slurms/logs/aggregate_%j.out
#SBATCH --error=slurms/logs/aggregate_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=END,FAIL

# trauma_ml — final aggregation job.
#
# Runs ONCE after every training job finishes.  Walks each target's outputs
# directory (mortality / iss_band / niss_band) and produces per-target
# unified Pareto + baseline summaries.
#
# Outputs structure:
#   outputs/mortality/all_metrics.csv          (binary, with baselines)
#   outputs/mortality/baseline_metrics.csv     (ISS/NISS/TRISS comparisons)
#   outputs/mortality/cohort_counts.csv
#   outputs/mortality/all_metrics__cohort_*.csv
#   outputs/iss_band/all_metrics.csv           (multiclass — no baselines)
#   outputs/iss_band/cohort_counts.csv
#   outputs/niss_band/all_metrics.csv          (multiclass — no baselines)
#   outputs/niss_band/cohort_counts.csv

set -euo pipefail
mkdir -p slurms/logs

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source "/scratch/${USER}/trauma_ml_venv/bin/activate"

echo "[INFO] Slurm job ${SLURM_JOB_ID} on $(hostname)"

# Aggregate each target's outputs directory independently.  Empty
# directories (target wasn't run) are skipped silently.
for target_dir in outputs/mortality outputs/iss_band outputs/niss_band; do
    if [[ ! -d "${target_dir}/metrics" ]]; then
        echo "[SKIP] ${target_dir}/metrics not found — target not run."
        continue
    fi

    n_models=$(ls "${target_dir}/metrics" 2>/dev/null | wc -l)
    if [[ "${n_models}" -eq 0 ]]; then
        echo "[SKIP] ${target_dir}/metrics is empty."
        continue
    fi

    echo "──────────────────────────────────────────────────────────────────"
    echo "[INFO] Aggregating ${target_dir}/  (${n_models} model directories)"
    echo "──────────────────────────────────────────────────────────────────"
    ls "${target_dir}/metrics/" | head -10
    if [[ "${n_models}" -gt 10 ]]; then
        echo "    ... ($((n_models - 10)) more)"
    fi

    python -m trauma_ml.cli.aggregate_metrics \
        --metrics-root "${target_dir}/metrics" \
        --models-root  "${target_dir}/models" \
        --output       "${target_dir}/all_metrics.csv" \
        --pareto-metric AUROC \
        --pareto-partitions test holdout

    echo "[OK] ${target_dir} aggregation complete:"
    ls -la "${target_dir}"/*.csv 2>/dev/null || echo "(no CSV outputs)"
    echo
done

echo "[OK] All target aggregations complete."
echo "Top-level outputs/ tree:"
find outputs -maxdepth 2 -name "*.csv" -type f | sort
