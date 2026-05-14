#!/usr/bin/env bash
# trauma_ml — submit the parallel Slurm pipeline (round 14, validated on DIPC atlas-edr).
#
# Run from the repository root:
#     bash slurms/submit_all.sh           # rebuild data + train all + aggregate
#     bash slurms/submit_all.sh --skip-build   # skip the rebuild (data already exists)
#
# Submits build → 10 training jobs in parallel → aggregate.  Each training
# job uses a distinct --model-id-prefix so they never collide on outputs/.
# The aggregate job runs ONLY after every training job has finished.
#
# To remove a family from the run, comment out its line in TRAIN_JOBS.

set -euo pipefail

# Make sure logs dir exists BEFORE Slurm tries to open log files.
mkdir -p slurms/logs

SKIP_BUILD=false
for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=true ;;
    *) echo "[!] Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ── 1. Build (or skip) the unified parquets ─────────────────────────────
BUILD_DEP=""
if [[ "${SKIP_BUILD}" == "true" ]]; then
  echo "[build  ] --skip-build: assuming outputs/datasets/unified_*.parquet exist"
  if [[ ! -f outputs/datasets/unified_train.parquet ]]; then
    echo "[!] outputs/datasets/unified_train.parquet missing — cannot --skip-build" >&2
    exit 1
  fi
else
  BUILD_ID=$(sbatch --parsable slurms/00_build_dataset.slurm)
  echo "[build  ] Slurm id ${BUILD_ID}"
  BUILD_DEP="--dependency=afterok:${BUILD_ID}"
fi

# ── 2. Training jobs — submit in parallel ───────────────────────────────
# Each line below becomes a separate Slurm job that:
#   • trains a disjoint subset of model families,
#   • uses a distinct --model-id-prefix so paths never collide,
#   • runs with --no-aggregate so only 99_aggregate writes the unified CSV.
# Comment out any line you don't want to run on this submission.
TRAIN_JOBS=(
  # ── Mortality (binary) ──────────────────────────────────────────────
  "slurms/10_train_logistic.slurm"        # CPU,  prefix lgr_*  (~1 h)
  "slurms/11_train_lightgbm.slurm"        # GPU,  prefix lgb_*  (~12-24 h)
  "slurms/12_train_xgboost.slurm"         # GPU,  prefix xgb_*  (~12-24 h)
  "slurms/13_train_catboost.slurm"        # GPU,  prefix cb_*   (~12-24 h)
  "slurms/14_train_flaml.slurm"           # CPU,  prefix flm_*  (AutoML, ~24-48 h)
  "slurms/15_train_tpot.slurm"            # CPU,  prefix tpt_*  (AutoML, ~48-72 h)
  "slurms/16_train_random_forest.slurm"   # CPU,  prefix rf_*   (~24-48 h)
  "slurms/17_train_tabpfn.slurm"          # GPU,  prefix tpf_*  (sub-sampled, ~1 h)
  "slurms/18_train_tabnet.slurm"          # GPU,  prefix tnt_*  (~12-24 h)
  # "slurms/19_train_survival.slurm"      # DISABLED: survival target builder
                                            #   raises NotImplementedError.
  # ── ISS-band (multiclass) ───────────────────────────────────────────
  "slurms/20_train_iss_xgboost.slurm"        # GPU,  prefix iss_xgb_*
  "slurms/21_train_iss_lightgbm.slurm"       # GPU,  prefix iss_lgb_*
  "slurms/22_train_iss_random_forest.slurm"  # CPU,  prefix iss_rf_*
  # ── NISS-band (multiclass) ──────────────────────────────────────────
  "slurms/23_train_niss_xgboost.slurm"       # GPU,  prefix niss_xgb_*
  "slurms/24_train_niss_lightgbm.slurm"      # GPU,  prefix niss_lgb_*
  "slurms/25_train_niss_random_forest.slurm" # CPU,  prefix niss_rf_*
)

TRAIN_IDS=()
for slurm_file in "${TRAIN_JOBS[@]}"; do
  if [[ ! -f "${slurm_file}" ]]; then
    echo "[!] missing slurm file: ${slurm_file}" >&2
    continue
  fi
  if [[ -n "${BUILD_DEP}" ]]; then
    jid=$(sbatch --parsable ${BUILD_DEP} "${slurm_file}")
  else
    jid=$(sbatch --parsable "${slurm_file}")
  fi
  echo "[train  ] ${slurm_file}  ->  Slurm id ${jid}"
  TRAIN_IDS+=("${jid}")
done

if [[ ${#TRAIN_IDS[@]} -eq 0 ]]; then
  echo "No training jobs submitted; aborting." >&2
  exit 1
fi

# ── 3. Aggregate AFTER all training jobs finish ─────────────────────────
DEPS=$(IFS=:; echo "${TRAIN_IDS[*]}")
AGG_ID=$(sbatch --parsable --dependency=afterok:${DEPS} slurms/99_aggregate.slurm)
echo "[agg    ] Slurm id ${AGG_ID}  (depends on ${DEPS})"

cat <<EOF

────────────────────────────────────────────────────────────────────────
Pipeline submitted.

$([[ -n "${BUILD_DEP}" ]] && echo "  build_id    : ${BUILD_ID}")
  train_ids   : ${TRAIN_IDS[*]}
  aggregate   : ${AGG_ID}

Monitor with:
  squeue -u \$USER
  squeue -u \$USER --start
  watch -n 60 'squeue -u \$USER'
  tail -F slurms/logs/*.out
  sacct -u \$USER --starttime today --format=JobID,JobName,State,ExitCode,Elapsed

When everything's green, results land in:
  outputs/all_metrics.csv            (unified Pareto across every model)
  outputs/all_metrics__cohort_*.csv  (per-cohort variants)
  outputs/baseline_metrics.csv       (long format ISS/NISS/TRISS)
  outputs/cohort_counts.csv          (case counts per cohort)
────────────────────────────────────────────────────────────────────────
EOF
