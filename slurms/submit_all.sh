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
  "slurms/10_train_logistic.slurm"        # CPU,  prefix lgr_*
  "slurms/11_train_lightgbm.slurm"        # CPU,  prefix lgb_*
  "slurms/12_train_xgboost.slurm"         # CPU,  prefix xgb_*
  "slurms/13_train_catboost.slurm"        # CPU,  prefix cb_*
  "slurms/14_train_flaml.slurm"           # CPU,  prefix flm_*  (AutoML)
  "slurms/15_train_tpot.slurm"            # CPU,  prefix tpt_*  (AutoML, big-mem)
  "slurms/16_train_random_forest.slurm"   # CPU,  prefix rf_*   (big-mem)
  "slurms/17_train_tabpfn.slurm"          # GPU rtx3090, prefix tpf_*
  "slurms/18_train_tabnet.slurm"          # GPU rtx3090, prefix tnt_*  (main grid)
  "slurms/18a_train_tabnet.slurm"         # GPU rtx3090, tabnet slice a
  "slurms/18b_train_tabnet.slurm"         # GPU rtx3090, tabnet slice b
  "slurms/18c_train_tabnet.slurm"         # GPU rtx3090, tabnet slice c
  "slurms/18d_train_tabnet.slurm"         # GPU rtx3090, tabnet slice d
  "slurms/18e_train_tabnet.slurm"         # GPU rtx3090, tabnet slice e
  "slurms/18f_train_tabnet.slurm"         # GPU rtx3090, tabnet slice f
  "slurms/18g_train_tabnet.slurm"         # GPU rtx3090, tabnet slice g
  "slurms/18h_train_tabnet.slurm"         # GPU rtx3090, tabnet slice h
  "slurms/30_train_mortality_doshi_ffnn.slurm"  # GPU rtx3090, prefix doshi_*
  # ── ISS-band (4-class ordinal) ──────────────────────────────────────
  "slurms/20_train_iss_xgboost.slurm"        # CPU,  prefix iss_xgb_*
  "slurms/21_train_iss_lightgbm.slurm"       # CPU,  prefix iss_lgb_*
  "slurms/22_train_iss_random_forest.slurm"  # CPU,  prefix iss_rf_*  (big-mem)
  "slurms/26_train_iss_flaml.slurm"          # CPU,  prefix iss_flm_*
  "slurms/27_train_iss_tpot.slurm"           # CPU,  prefix iss_tpt_*  (big-mem)
  "slurms/33_train_iss_doshi_ffnn.slurm"     # GPU rtx3090, prefix iss_doshi_*
  # ── NISS-band (4-class ordinal) ─────────────────────────────────────
  "slurms/23_train_niss_xgboost.slurm"       # CPU,  prefix niss_xgb_*
  "slurms/24_train_niss_lightgbm.slurm"      # CPU,  prefix niss_lgb_*
  "slurms/25_train_niss_random_forest.slurm" # CPU,  prefix niss_rf_*  (big-mem)
  "slurms/28_train_niss_flaml.slurm"         # CPU,  prefix niss_flm_*
  "slurms/29_train_niss_tpot.slurm"          # CPU,  prefix niss_tpt_*  (big-mem)
  "slurms/34_train_niss_doshi_ffnn.slurm"    # GPU rtx3090, prefix niss_doshi_*
  # ── Band-binary (ISS>=16 / NISS>=16) ────────────────────────────────
  # Split into 6 parallel slices per block (--start-id/--end-id over the SAME
  # full grid, so model_ids stay canonical and partial work is preserved).
  # The old monolithic 31_/32_ slurms are superseded by these and are NOT
  # submitted; they ran both blocks sequentially, so the `_other` block
  # (1350 combos) never started until `_boost` (1296) fully finished.
  #   *_bin_boost : xgboost/lightgbm/catboost/flaml (+imputer=none), 1296 combos
  #   *_bin_other : logistic_l1/elasticnet/random_forest/tpot/tabnet, 1350 combos
  "slurms/31a_train_iss_bin_boost.slurm"     # iss_bin_boost_0000..0215
  "slurms/31b_train_iss_bin_boost.slurm"     # iss_bin_boost_0216..0431
  "slurms/31c_train_iss_bin_boost.slurm"     # iss_bin_boost_0432..0647
  "slurms/31d_train_iss_bin_boost.slurm"     # iss_bin_boost_0648..0863
  "slurms/31e_train_iss_bin_boost.slurm"     # iss_bin_boost_0864..1079
  "slurms/31f_train_iss_bin_boost.slurm"     # iss_bin_boost_1080..1295
  "slurms/31g_train_iss_bin_other.slurm"     # iss_bin_other_0000..0224
  "slurms/31h_train_iss_bin_other.slurm"     # iss_bin_other_0225..0449
  "slurms/31i_train_iss_bin_other.slurm"     # iss_bin_other_0450..0674
  "slurms/31j_train_iss_bin_other.slurm"     # iss_bin_other_0675..0899
  "slurms/31k_train_iss_bin_other.slurm"     # iss_bin_other_0900..1124
  "slurms/31l_train_iss_bin_other.slurm"     # iss_bin_other_1125..1349
  "slurms/32a_train_niss_bin_boost.slurm"    # niss_bin_boost_0000..0215
  "slurms/32b_train_niss_bin_boost.slurm"    # niss_bin_boost_0216..0431
  "slurms/32c_train_niss_bin_boost.slurm"    # niss_bin_boost_0432..0647
  "slurms/32d_train_niss_bin_boost.slurm"    # niss_bin_boost_0648..0863
  "slurms/32e_train_niss_bin_boost.slurm"    # niss_bin_boost_0864..1079
  "slurms/32f_train_niss_bin_boost.slurm"    # niss_bin_boost_1080..1295
  "slurms/32g_train_niss_bin_other.slurm"    # niss_bin_other_0000..0224
  "slurms/32h_train_niss_bin_other.slurm"    # niss_bin_other_0225..0449
  "slurms/32i_train_niss_bin_other.slurm"    # niss_bin_other_0450..0674
  "slurms/32j_train_niss_bin_other.slurm"    # niss_bin_other_0675..0899
  "slurms/32k_train_niss_bin_other.slurm"    # niss_bin_other_0900..1124
  "slurms/32l_train_niss_bin_other.slurm"    # niss_bin_other_1125..1349
  # ── Faithful Doshi ICD->severity FFNN (L3-only) ─────────────────────
  "slurms/35_train_doshi_icd.slurm"          # GPU rtx3090, prefix doshi_icd*/doshi_icdplus*
  # ── Survival (optional; enable if the survival target builder is ready)
  # "slurms/19_train_survival.slurm"         # CPU, big-mem
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
