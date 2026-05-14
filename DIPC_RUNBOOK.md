# DIPC_RUNBOOK.md

Step-by-step reproduction of the trauma_ml pipeline on DIPC's atlas-edr cluster, with every gotcha encoded that we hit on the first run. Follow this exactly the next time you set up from scratch.

Cluster reference: `atlas-edr.sw.ehu.es`, validated April 2026.

---

## Phase 0 — Cluster reconnaissance (5 min, do once)

Login and confirm the cluster's actual partition layout. **Do not skip this** even if you've used DIPC before — the values below were determined empirically and may drift.

```bash
ssh mcapo@atlas-edr.sw.ehu.es

# Partitions and what GPUs they have
sinfo -O partition,gres
# Expected output includes:
#   general*    gpu:p40:1
#   general*    gpu:p40:2
#   general*    gpu:rtx3090:1
#   general*    gpu:rtx3090:2
#   general*    (null)            ← CPU-only nodes

# Node sizes (CPUs / RAM in MB / GPU type)
sinfo -N -p general -o "%n %c %m %G" | head -30
sinfo -N -p general -o "%n %c %m %G" | grep -E "rtx3090|p40"
# Atlas-edr findings:
#   CPU-only nodes:     32 CPUs / 128–385 GB RAM
#   RTX 3090 nodes:     48 CPUs / 95 GB RAM  (only 7 nodes, contested)
#   P40 nodes:          48 CPUs / 385 GB RAM (32 nodes, faster to schedule)
```

If your cluster differs, you'll need to adjust `--partition`, `--gres`, `--mem`, and `--cpus-per-task` in the slurm files. The values shipped here target atlas-edr specifically.

Also check your QoS limits — slurms request walltimes in the 12h-4d range and need a QoS that allows them:

```bash
sacctmgr show qos format=name,maxwall,priority,maxsubmit
sacctmgr show association where user=$USER format=user,qos
```

The shipped slurms assume your account has access to these QoSes (atlas-edr defaults):

| QoS | MaxWall | Used by |
|---|---|---|
| `regular` | 1 day | most slurms (1d or less) |
| `long`    | 2 days | catboost, tpot, tabnet, survival |
| `xlong`   | 8 days | flaml, random_forest |

If your QoS landscape is different (different names, different MaxWall ceilings), edit `--qos` and `--time` in every slurm to fit. The runbook's "Per-family fit count + walltime + QoS" table later shows which QoS each slurm uses.

---

## Phase 1 — Get the project onto DIPC (5 min)

The project root layout (matches your laptop's `~/Escritorio/DIPC/trauma_ml_v0.2/trauma_ml/`):

```
trauma_ml/                         ← project root (rsync this)
├── config/                        ← paths.yaml, experiment_grids.yaml
├── data/                          ← raw NTDB CSVs
├── outputs/                       ← models/metrics/plots written here
├── slurms/                        ← Slurm scripts at PROJECT ROOT (NOT under src/)
│   ├── 00_build_dataset.slurm
│   ├── 10_train_logistic.slurm
│   ├── ...
│   ├── submit_all.sh
│   └── requirements.txt
├── src/                           ← Python package source
│   └── trauma_ml/
│       ├── __init__.py
│       ├── trainer.py
│       └── ...
├── tests/
├── DIPC_RUNBOOK.md                ← this file
├── PIPELINE_OPTIONS.md
├── NTDB_REQUIRED_COLUMNS.md
├── pyproject.toml                 ← `pip install -e .` reads this
├── README.md
├── LICENSE
└── NTDB_variable_mapping_for_trauma_ML.xlsx
```

The slurms run with the project root as CWD — every relative path inside (`outputs/datasets/...`, `slurms/logs/...`) resolves against that.

### Transfer from laptop with rsync

```bash
# From your laptop — filtered transfer (skip IDE / venv / cached files)
rsync -av \
    --exclude='.venv/' \
    --exclude='.idea/' \
    --exclude='.pytest_cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='*.zip' \
    --exclude='outputs/models/' \
    --exclude='outputs/metrics/' \
    --exclude='outputs/plots/' \
    --exclude='outputs/imputation_eval/' \
    --exclude='outputs/datasets/' \
    --exclude='slurms/logs/' \
    ~/Escritorio/DIPC/trauma_ml_v0.2/trauma_ml/ \
    mcapo@atlas-edr.sw.ehu.es:/scratch/mcapo/trauma_ml_v1/
```

Trailing slashes matter: `trauma_ml/` (with slash) on the source = "copy the *contents* into the destination". The destination `trauma_ml_v1/` (with slash) = "into a directory of that name". Together they create `/scratch/mcapo/trauma_ml_v1/` containing your project files directly.

To preview what would be transferred without actually copying, add `--dry-run` to the command — it lists every file that would be transferred. Drop the flag to commit.

### Updating just the code (without re-rsync)

If you've already transferred and just need to ship updated source:

```bash
# From laptop
scp /mnt/user-data/outputs/src.zip mcapo@atlas-edr.sw.ehu.es:/scratch/mcapo/trauma_ml_v1/

# On DIPC
ssh mcapo@atlas-edr.sw.ehu.es
cd /scratch/mcapo/trauma_ml_v1/
unzip -o src.zip                  # extracts: pyproject.toml, slurms/, src/, *.md
                                  # (top level matches the project root structure above)
```

The shipped `src.zip` extracts to **project root level** — so unzipping at `/scratch/mcapo/trauma_ml_v1/` puts:
- `pyproject.toml` and `*.md` files at `/scratch/mcapo/trauma_ml_v1/`
- `slurms/` at `/scratch/mcapo/trauma_ml_v1/slurms/`
- Python package at `/scratch/mcapo/trauma_ml_v1/src/trauma_ml/`

After updating code, ALWAYS reinstall in editable mode so the venv's console scripts pick up the new package source:

```bash
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source /scratch/$USER/trauma_ml_venv/bin/activate
cd /scratch/mcapo/trauma_ml_v1/
pip install -e .
```

### Final preflight on DIPC

```bash
cd /scratch/mcapo/trauma_ml_v1/

mkdir -p slurms/logs                # CRITICAL — Slurm writes log files HERE
                                    #   before the job script runs.
                                    #   Missing dir = ExitCode=1:0, Elapsed=00:00:00.
chmod +x slurms/submit_all.sh

# Verify project layout
ls -la
# Expect at top level:
#   config/  data/  outputs/  slurms/  src/  tests/
#   pyproject.toml  DIPC_RUNBOOK.md  PIPELINE_OPTIONS.md
#   README.md  LICENSE  NTDB_variable_mapping_for_trauma_ML.xlsx

ls src/
# Expect: trauma_ml/

ls slurms/
# Expect: 00_build_dataset.slurm  10_train_logistic.slurm  ...  submit_all.sh  requirements.txt

ls config/
# Expect: paths.yaml  experiment_grids.yaml
```

If any expected directory is missing, the rsync skipped or excluded too aggressively — re-run with the corrected exclude list.

---

## Phase 2 — Build the Python venv (one-off, ~20 min)

This is done **interactively on a login node**, not via a slurm job — so you can fix install errors as they happen.

```bash
# Step 2.1 — Load Python + CUDA modules
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# Step 2.2 — Create the venv ON /scratch (NOT $HOME)
# CRITICAL: $HOME points to /dipc/$USER which is mounted on login but
# NOT consistently visible from compute nodes.  /scratch IS visible from
# both → the venv has to live there.
python -m venv /scratch/$USER/trauma_ml_venv
source /scratch/$USER/trauma_ml_venv/bin/activate
which python
# Expect: /scratch/mcapo/trauma_ml_venv/bin/python

# Step 2.3 — Install build tools
python -m pip install -U pip wheel
# CRITICAL: setuptools<81 — tpot's transitive dep `stopit` imports the
# legacy `pkg_resources` API which was removed in setuptools 81+.
python -m pip install "setuptools<81"

# Step 2.4 — Install torch with CUDA 12.1 wheels (matches the CUDA module)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Step 2.5 — Install everything else
cd /scratch/mcapo/trauma_ml_v1/
pip install --prefer-binary -r slurms/requirements.txt
# CRITICAL: --prefer-binary is non-negotiable.  Without it pip tries to
# build pyarrow/libcst from source, which needs a Rust compiler that
# isn't on the cluster's PATH.  --prefer-binary forces the latest
# prebuilt wheel.

# Step 2.6 — Install trauma_ml in editable mode
pip install -e .

# Step 2.7 — Verify the install
which trauma-build trauma-train trauma-aggregate-metrics
# Expect: /scratch/mcapo/trauma_ml_venv/bin/trauma-{build,train,aggregate-metrics}

python -c "
import torch, sklearn, xgboost, lightgbm, pandas, pyarrow
import catboost, flaml, tpot, tabpfn, pytorch_tabnet, sksurv
import miceforest, imblearn, shap
print('torch     :', torch.__version__, '  cuda(login):', torch.cuda.is_available())
print('sklearn   :', sklearn.__version__)
print('xgboost   :', xgboost.__version__)
print('lightgbm  :', lightgbm.__version__)
print('catboost  :', catboost.__version__)
print('flaml     :', flaml.__version__)
print('tpot      :', tpot.__version__)
print('tabpfn    :', tabpfn.__version__)
print('All required packages OK.')
"
```

`torch.cuda.is_available()` will print **False** on the login node — that's expected, login nodes have no GPU. It'll be True inside a GPU slurm job.

If something says `MISSING`, install it individually with `pip install --prefer-binary <pkg>`.

---

## Phase 3 — Verify GPU works on a compute node (5 min, do once)

Worth doing because xgboost on glibc<2.28 systems may silently fall back to CPU. This catches it.

```bash
# Allocate a one-shot GPU node
srun --partition=general --gres=gpu:rtx3090:1 --cpus-per-task=4 \
     --mem=8G --time=00:05:00 \
     bash -c '
        module purge
        module load Python/3.10.4-GCCcore-11.3.0
        module load CUDA/12.1.1
        source /scratch/$USER/trauma_ml_venv/bin/activate

        python -c "
import torch
print(\"torch cuda :\", torch.cuda.is_available(), torch.cuda.get_device_name(0))

# xgboost GPU sanity
import xgboost as xgb
import numpy as np
X = np.random.RandomState(0).randn(2000, 5).astype(\"float32\")
y = (X[:, 0] > 0).astype(int)
try:
    xgb.XGBClassifier(device=\"cuda\", tree_method=\"hist\", n_estimators=20).fit(X, y)
    print(\"xgb GPU    : OK\")
except Exception as e:
    print(\"xgb GPU    : FAILED -\", str(e)[:150])

# lightgbm GPU sanity
try:
    import lightgbm as lgb
    lgb.LGBMClassifier(device=\"gpu\", n_estimators=20, verbose=-1).fit(X, y)
    print(\"lgbm GPU   : OK\")
except Exception as e:
    print(\"lgbm GPU   : FAILED -\", str(e)[:150])
"
'
```

Expected (atlas-edr is glibc 2.17 — manylinux2014 wheels):
- `torch cuda: True NVIDIA GeForce RTX 3090`
- `xgb GPU: OK` *or* `FAILED — Cuda is not enabled` (manylinux2014 limitation)
- `lgbm GPU: FAILED — GPU Tree Learner was not enabled` (default pip wheel is CPU-only)

If xgboost GPU fails, edit `slurms/12_train_xgboost.slurm`:

```bash
sed -i 's|--use-gpu       auto|--use-gpu       never|' slurms/12_train_xgboost.slurm
sed -i 's|^#SBATCH --gres=gpu:rtx3090:1$|# (no GPU; xgboost manylinux2014 wheel CPU-only)|' \
    slurms/12_train_xgboost.slurm
```

CatBoost is the safest GPU bet — its default wheel always ships GPU support.

---

## Phase 4 — Build the unified parquets (one-off, ~10 min)

```bash
cd /scratch/mcapo/trauma_ml_v1/
sbatch slurms/00_build_dataset.slurm
squeue -u $USER
tail -F slurms/logs/build_*.out
```

Wait for the job to finish. The build runs `trauma-build` with `--years 2019 2020 2021 2022 2024 --holdout-years 2024`, producing:

* `outputs/datasets/unified_train.parquet` — ~4.67 M rows (AY 2019+2020+2021+2022)
* `outputs/datasets/unified_holdout.parquet` — ~1.35 M rows (AY 2024)

When the build returns successfully (`State: COMPLETED`, log ends with two `[OK]` lines), audit:

```bash
# Reload modules in your interactive shell every fresh session
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source /scratch/$USER/trauma_ml_venv/bin/activate

python -m trauma_ml.cli.audit_parquet outputs/datasets/unified_train.parquet
python -m trauma_ml.cli.audit_parquet outputs/datasets/unified_holdout.parquet
```

Both audits should end with `Summary: 0 MISSING, 0 EMPTY, 0 SPARSE`. Some `SPARSE` warnings may appear on EMS-physiology fields for AY 2019/2020 — those years didn't collect EMS-prehospital values; my loader logs that and falls back to ED-arrival values as the closest proxy. As long as no `MISSING` or `EMPTY`, you're cleared to train.

---

## Phase 5 — Submit the full training pipeline

```bash
cd /scratch/mcapo/trauma_ml_v1/
bash slurms/submit_all.sh --skip-build
```

The wrapper submits 9 training jobs in parallel (survival is disabled — see "What gets exhaustively tested" below), plus 99_aggregate that depends on them. You'll see something like:

```
[build  ] --skip-build: assuming outputs/datasets/unified_*.parquet exist
[train  ] slurms/10_train_logistic.slurm        ->  Slurm id 36912XX  (lgr_*)
[train  ] slurms/11_train_lightgbm.slurm        ->  Slurm id 36912XX  (lgb_*)
[train  ] slurms/12_train_xgboost.slurm         ->  Slurm id 36912XX  (xgb_*)
[train  ] slurms/13_train_catboost.slurm        ->  Slurm id 36912XX  (cb_*)
[train  ] slurms/14_train_flaml.slurm           ->  Slurm id 36912XX  (flm_*)
[train  ] slurms/15_train_tpot.slurm            ->  Slurm id 36912XX  (tpt_*)
[train  ] slurms/16_train_random_forest.slurm   ->  Slurm id 36912XX  (rf_*)
[train  ] slurms/17_train_tabpfn.slurm          ->  Slurm id 36912XX  (tpf_*)
[train  ] slurms/18_train_tabnet.slurm          ->  Slurm id 36912XX  (tnt_*)
[agg    ] Slurm id 36913XX  (depends on 36912XX:36912XX:...)
```

Total wall time: ~4 days, gated by `random_forest` (4-day walltime ceiling). All other jobs finish earlier and idle waiting for the aggregator.

**If you haven't built yet**, omit `--skip-build` and the wrapper submits the build first with downstream `afterok` dependencies.

---

## Phase 6 — Monitor

```bash
# Queue overview
squeue -u $USER

# Live updating queue
watch -n 60 'squeue -u $USER'

# Estimated start times for pending jobs
squeue -u $USER --start

# Per-job logs (live tail)
tail -F slurms/logs/train_lgb_*.out
tail -F slurms/logs/aggregate_*.out

# Completed-job summary
sacct -u $USER --starttime today \
    --format=JobID,JobName%20,State,ExitCode,Elapsed,MaxRSS,NodeList
```

If a training job fails (state `FAILED`, exit code non-zero), `99_aggregate` will not run automatically. Two recovery options:

```bash
# Option A — fix the failure, resubmit just that family, then aggregate
sbatch slurms/14_train_flaml.slurm     # whichever failed
# Wait for it; then:
sbatch slurms/99_aggregate.slurm

# Option B — accept partial results, aggregate what you have
sbatch slurms/99_aggregate.slurm       # walks outputs/metrics/, picks up
                                       # whatever models exist regardless
                                       # of which family they came from
```

---

## Phase 7 — What you get

Once `99_aggregate.slurm` returns successfully:

```bash
ls -la outputs/all_metrics*.csv outputs/baseline_metrics.csv outputs/cohort_counts.csv
```

Five files:

* `all_metrics.csv` — wide table, one row per (model OR baseline calibration variant). Columns include `pareto_front` (bool) computed across all rows.
* `all_metrics__cohort_baseline_complete.csv` — the same table, restricted to cases that have ISS+NISS+TRISS inputs available.
* `all_metrics__cohort_ed_complete.csv` — restricted to cases with on-scene + ED-arrival vitals.
* `all_metrics__cohort_onsite_complete.csv` — restricted to on-scene-only cases.
* `baseline_metrics.csv` — long-format detail of every (score, calibration, partition, cohort) cell.
* `cohort_counts.csv` — case counts per cohort definition.

Pull them back to your laptop:

```bash
# From your laptop:
rsync -av mcapo@atlas-edr.sw.ehu.es:/scratch/mcapo/trauma_ml_v1/outputs/*.csv \
    ~/Escritorio/DIPC/trauma_ml_v0.2/trauma_ml/outputs/
```

---

## Phase-cutoff variable lists (what each cutoff actually selects)

Round 18+19 added a hard **non-predictor blacklist** that strips IDs (`INC_KEY`, etc.) and hospital-level admin (`TEACHINGSTATUS`, `STATEDESIGNATION`, `VERIFICATIONLEVEL`, `HOSPITALTYPE`, ...) from every predictor list. The lists below show what's left AFTER blacklist filtering.

| Phase cutoff | Variables selected (post-blacklist) | Why |
|---|---|---|
| `On-scene` | `AGEYEARS`, `SEX`, `TRAUMATYPE`, `GCSTOTAL`, `SBPFIRST`, `RRFIRST` | Earliest data — what EMS / first responders have. GCS/SBP/RR are first-recorded values. |
| `On-scene + ED arrival` | adds `TEMPERATURE`, `PULSEOXIMETRY`, `PULSERATE` | Adds vitals captured at ED triage. |
| `On-scene + ED arrival + In-hospital` | adds `ISS`, `NISS` | Adds anatomical injury severity scores derived a posteriori from AIS codes. Tran 2022's "full features" set. |
| `iss_only` | `ISS`, `AGEYEARS`, `SEX` | Pinned to ISS's exact inputs. For benchmarking ML against ISS alone. |
| `niss_only` | `NISS`, `AGEYEARS`, `SEX` | Pinned to NISS's exact inputs. |
| `triss_inputs` | `ISS`, `GCSTOTAL`, `SBPFIRST`, `RRFIRST`, `AGEYEARS`, `SEX`, `TRAUMATYPE` | Exact TRISS input set — anatomy (ISS), physiology (GCS, SBP, RR), age, mechanism. |
| `all_baseline_inputs` | `ISS`, `NISS`, `GCSTOTAL`, `SBPFIRST`, `RRFIRST`, `AGEYEARS`, `SEX`, `TRAUMATYPE` | Union of ISS + NISS + TRISS inputs. |

The first three are **time-ordered cutoffs** — each adds variables to the previous, simulating what a clinician would know at that point in the patient journey. The last four are **baseline-feature cutoffs** that pin the predictor set to a clinical score's exact inputs, useful for direct ML-vs-baseline benchmarking.

To verify the actual lists on your local catalogue (matches your xlsx exactly):

```bash
cd /scratch/mcapo/trauma_ml_v1/
python -m trauma_ml.cli.phase_cutoff_audit \
    --catalogue NTDB_variable_mapping_for_trauma_ML.xlsx \
    --parquet outputs/datasets/unified_train.parquet \
    --output outputs/phase_cutoff_audit/
ls outputs/phase_cutoff_audit/
# Three files produced: phase_cutoff_features.csv, phase_cutoff_summary.csv,
# phase_cutoff_overlap.png — inspect to confirm.
```

### Hard blacklist (NEVER predictors, regardless of catalogue tagging)

```
Patient/incident IDs:
  INC_KEY, INCKEY, INCIDENT_KEY, INCIDENTKEY (case-insensitive)

Hospital-level attributes (case-mix proxies, not patient state):
  TEACHINGSTATUS, HOSPITALTYPE, BEDSIZE, STATEDESIGNATION,
  ACSVERIFICATIONLEVEL, VERIFICATIONLEVEL,
  FACILITYID, TRAUMAFACILITYID, FACILITY_ID, TRAUMACENTERLEVEL

Raw ICD/E-code strings (used only for derivation, not as features):
  PRIMARYECODEICD10, ICDDIAGNOSISCODE, AISPREDOT

Outcome/discharge columns (target leakage):
  HOSPDISCHARGEDISPOSITION, EDDISCHARGEDISPOSITION, DEATHINED,
  DISCHARGE_DATE, DISCHARGEDATE, EDDISCHARGE, HOSPDISCHARGE,
  INTERFACILITYTRANSFER
```

The blacklist is enforced in `catalogue.py::variables_for()` AND defensively in `trainer.py::_select_predictors()`. Comparison is case-insensitive — `inc_key`, `Inc_Key`, and `INC_KEY` are all blocked equivalently.

---

## Multiclass targets (ISS-band / NISS-band) — round 19

In addition to the binary mortality target, the pipeline supports **multiclass severity-band prediction**:

| Target | Kind | Bands |
|---|---|---|
| `in_hospital_mortality` | binary | died (1) vs survived (0) |
| `iss_band` | ordinal multiclass | Minor (ISS 1-8), Moderate (9-15), Severe (16-24), Profound (25+) |
| `niss_band` | ordinal multiclass | same boundaries on NISS instead of ISS |

The bands match the standard NTDB / ATLS reporting groups.

### Adding the targets to your config

The targets MUST be defined in `config/experiment_grids.yaml` before the multiclass slurms can find them. Add this snippet under the `targets:` key:

```yaml
targets:
  - name: in_hospital_mortality
    kind: binary
    spec:
      positive_code: 5     # NTDB HOSPDISCHARGEDISPOSITION code 5 = died
      include_ed_death: true

  - name: iss_band
    kind: ordinal_bands
    spec:
      variable: ISS
      bands:
        - [0,  9,  "Minor"]      # ISS 1-8
        - [9,  16, "Moderate"]   # ISS 9-15
        - [16, 25, "Severe"]     # ISS 16-24
        - [25, 76, "Profound"]   # ISS 25-75 (75 is theoretical max)

  - name: niss_band
    kind: ordinal_bands
    spec:
      variable: NISS
      bands:
        - [0,  9,  "Minor"]
        - [9,  16, "Moderate"]
        - [16, 25, "Severe"]
        - [25, 76, "Profound"]
```

The band ranges are `[lower_inclusive, upper_exclusive)` — so `[0, 9, "Minor"]` covers ISS 1-8.

### What the multiclass slurms produce

Six new slurms (3 model families × 2 targets):

| Slurm | Target | Family | Prefix |
|---|---|---|---|
| `20_train_iss_xgboost.slurm` | iss_band | xgboost | `iss_xgb_*` |
| `21_train_iss_lightgbm.slurm` | iss_band | lightgbm | `iss_lgb_*` |
| `22_train_iss_random_forest.slurm` | iss_band | random_forest | `iss_rf_*` |
| `23_train_niss_xgboost.slurm` | niss_band | xgboost | `niss_xgb_*` |
| `24_train_niss_lightgbm.slurm` | niss_band | lightgbm | `niss_lgb_*` |
| `25_train_niss_random_forest.slurm` | niss_band | random_forest | `niss_rf_*` |

These use a NARROWER grid than mortality slurms — calibration and augmentation axes have a single value each because:

- **Calibration** (Platt/isotonic) is binary-only; doesn't generalize cleanly to multiclass.
- **SMOTE/ADASYN** also binary-only; multiclass oversampling needs different (k>2) algorithms.
- **Logistic regression** is intentionally not included — slow on multiclass, and the boosting + RF families cover the prediction space.
- **TabPFN/TabNet/FLAML/TPOT** — also not included for multiclass yet to keep the run focused; can be added later if needed.

**Per-target grid: 7 cutoffs × 2 imputers × 3 missingness × 1 cal × 1 aug = 42 fits/family.**

### Output organisation by target

Round 22 makes target subfolders **automatic** — derived from the target name. The CLI and slurms no longer need to pass `--outputs-root` explicitly:

```
outputs/
  datasets/                    # shared (parquets used by all targets)
    unified_train.parquet
    unified_holdout.parquet
  mortality/                   # binary mortality target (in_hospital_mortality → mortality)
    models/
    metrics/
    plots/
    all_metrics.csv
    baseline_metrics.csv       # ISS/NISS/TRISS comparisons
    cohort_counts.csv
  iss_band/                    # multiclass ISS-band target
    models/
    metrics/
    plots/
    all_metrics.csv            # NO baseline_metrics — bands ARE derived from ISS
    cohort_counts.csv
  niss_band/                   # multiclass NISS-band target
    ...same structure...
```

The mapping `target → subfolder` is defined in `src/trauma_ml/cli/run_experiments.py::TARGET_SUBFOLDER`. Currently:

| Target | Subfolder |
|---|---|
| `in_hospital_mortality` | `outputs/mortality/` |
| `iss_band` | `outputs/iss_band/` |
| `niss_band` | `outputs/niss_band/` |
| (any other) | `outputs/<target_name>/` |

To override the automatic layout, pass `--outputs-root <PATH>`. The target subfolder will still be appended under it. So `--outputs-root /tmp/runX --targets iss_band` writes to `/tmp/runX/iss_band/`.

The end-of-run aggregator and `99_aggregate.slurm` walk each target subfolder independently and produce per-target unified Pareto + cohort summaries. Multi-target runs (e.g. `--targets in_hospital_mortality iss_band`) emit separate aggregated files under each subfolder.

### Plots produced for multiclass

- ✅ Confusion matrix (test + holdout) — works for any K classes.
- ✅ SHAP beeswarm + bar — collapses multiclass `(samples, features, classes)` array to `mean |SHAP|` across classes for the summary plots.
- ❌ ROC / PR curves — skipped silently (binary-only metrics, one-vs-rest plots become unreadable for K>3).

Per-class metrics (precision/recall/F1 per band, macro and weighted averages) are written to `outputs/<target>/metrics/<model_id>/overall__test.json` for every multiclass model.

---

## What gets exhaustively tested in this configuration

**Round 17 update**: every family uses the **identical outer grid** — same phase cutoffs, same imputers, same calibrations, same augmentations, same missingness thresholds. This guarantees fair head-to-head comparison: the differences in `all_metrics.csv` reflect the model family alone, not different preprocessing choices.

### The unified grid (every active family)

| Axis | Values | Count |
|---|---|---|
| `--phase-cutoffs` | `On-scene`, `On-scene + ED arrival`, `On-scene + ED arrival + In-hospital`, `iss_only`, `niss_only`, `triss_inputs`, `all_baseline_inputs` | 7 |
| `--imputers` | `median_mode`, `mice` | 2 |
| `--calibrations` | `none`, `platt`, `isotonic` | 3 |
| `--augmentations` | `null`, `smote`, `adasyn` | 3 |
| `--missingness-thresholds` | `25`, `50`, `85` | 3 |

**Per-family combinations**: 7 × 2 × 3 × 3 × 3 = **378 fits per family** (756 for logistic since it bundles two sub-families: `logistic_l1` + `logistic_elasticnet`).

### Per-family fit count + walltime + QoS

| Family | Slurm | Fits | ~min/fit | Walltime | QoS |
|---|---|---|---|---|---|
| logistic (`lgr_*`) | `10_train_logistic.slurm` | 756 | ~1 | 1 day | `regular` |
| lightgbm (`lgb_*`) | `11_train_lightgbm.slurm` | 378 | ~2 | 1 day | `regular` |
| xgboost (`xgb_*`) | `12_train_xgboost.slurm` | 378 | ~2 | 1 day | `regular` |
| catboost (`cb_*`) | `13_train_catboost.slurm` | 378 | ~4 | 2 days | `long` |
| flaml (`flm_*`) | `14_train_flaml.slurm` | 378 | ~8 | 3 days | `xlong` |
| tpot (`tpt_*`) | `15_train_tpot.slurm` | 378 | 5 (HARD CAP per fit) | 2 days | `long` |
| random_forest (`rf_*`) | `16_train_random_forest.slurm` | 378 | ~10 | 4 days | `xlong` |
| tabpfn (`tpf_*`, sub-sampled) | `17_train_tabpfn.slurm` | 378 | ~1 | 12 h | `regular` |
| tabnet (`tnt_*`) | `18_train_tabnet.slurm` | 378 | ~5 | 2 days | `long` |
| iss_band — xgboost | `20_train_iss_xgboost.slurm` | 42 | ~2 | 1 day | `regular` |
| iss_band — lightgbm | `21_train_iss_lightgbm.slurm` | 42 | ~2 | 1 day | `regular` |
| iss_band — random_forest | `22_train_iss_random_forest.slurm` | 42 | ~10 | 3 days | `xlong` |
| niss_band — xgboost | `23_train_niss_xgboost.slurm` | 42 | ~2 | 1 day | `regular` |
| niss_band — lightgbm | `24_train_niss_lightgbm.slurm` | 42 | ~2 | 1 day | `regular` |
| niss_band — random_forest | `25_train_niss_random_forest.slurm` | 42 | ~10 | 3 days | `xlong` |
| **TOTAL** | | **~4,032 fits** | | gated by `rf` (4 days) | |

The `--qos` value in each slurm is matched to its `--time`:

| QoS | MaxWall | Used by |
|---|---|---|
| `regular` | 24h (1 day) | logistic, lightgbm, xgboost, tabpfn, tabnet (mortality 1d), all iss/niss xgb+lgb |
| `long`    | 48h (2 days) | catboost, tpot, tabnet, survival |
| `xlong`   | 8 days       | flaml, random_forest, all iss/niss random_forest |

If you ever change a slurm's `--time`, also update its `--qos` to the matching QoS — Slurm rejects submissions where `--time` exceeds the QoS's `MaxWall`. Run `sacctmgr show qos format=name,maxwall` if you need to check current limits.

Survival (`19_train_survival.slurm`) is **disabled** in `submit_all.sh` — the (event, time) target builder isn't yet implemented and the cox_ph / random_survival_forest factories need it.

### Why some imputers / augmentations are NOT in the universal grid

The user's request was "all imputers, all augmentations, all missingness, all phase cutoffs". Two imputers were intentionally **excluded**:

- **`knn`** is O(n²) memory — at 4.67M rows × 30 columns the pairwise distance matrix needs ~85 TB. The job OOMs in seconds. The CLI accepts it for smaller cohorts; on the full pool it cannot run.
- **`missforest`** (miceforest) takes ~1-2 hours **per call** before any model trains. Including it would multiply walltime by ~10× across every family — incompatible with the 4-day Slurm ceiling.
- **`none`** (passthrough NaN) is excluded from the universal grid because most families error on NaN inputs. Only xgboost / lightgbm accept it natively. Including `none` would create rows in `all_metrics.csv` that fail for 7 of 9 families — apples-to-oranges comparison defeats the unified-grid principle.

`median_mode` (fast, robust) and `mice` (sklearn IterativeImputer with BayesianRidge — Tran 2022's choice) are the realistic "both" that work for every active family on this data scale.

### TPOT-specific caveat

TPOT's per-fit budget is hard-capped at **5 minutes** of genetic search (set in `models.py::tpot` factory, round 17). Without that cap, TPOT would run unlimited search per fit (~60-90 min each) and blow well past any walltime ceiling. If you want deeper TPOT search per fit, narrow the grid first (e.g. drop `25` and `85` from `--missingness-thresholds`, leaving only `50`), then bump the budget in the factory.

### Customising the grid

To **narrow** any axis (cut walltime), edit the relevant slurm's `--imputers` / `--missingness-thresholds` / `--phase-cutoffs` / `--augmentations` / `--calibrations` lines. Each option dropped from a 3-option axis cuts the family's walltime by ~33%.

To **widen** any axis, do the opposite. But check the `--time` ceiling: if your widening pushes runtime past the slurm's walltime, the job is killed mid-run.

To **break the unified-grid principle deliberately** (e.g. add `none` imputer only for xgboost/lightgbm), edit those two slurm files only — the aggregator still produces a single Pareto front, but rows from different families won't be perfectly comparable.

**ICD-10 injury features (Barell matrix).** Round 27 adds 22 binary injury features derived at build time from `PUF_AISDIAGNOSIS.ICDDIAGNOSISCODE`: 12 body-region one-hots from the CDC Barell matrix (TBI, face, neck, SCI, vertebral, thorax, abdomen/pelvis, upper/lower extremity, burns, other) plus 10 specific-injury flags from Tran 2022's Fig 5 SHAP plot (subdural hemorrhage, concussion, pneumothorax, multiple rib fracture, splenic/liver laceration, pelvic/femur/distal radius/foot fracture). Available in all three time-ordered phase cutoffs because injuries are knowable on first contact. Patients absent from the diagnosis table get 0s (no NaN, no imputation). The mapping is in `src/trauma_ml/barell.py`; you can edit it to add more specific injuries or refine body-region rules. **Requires rebuilding the parquet** — run `trauma-build --years 2019 2020 2021 2022 2024 --holdout-years 2024` after deploying.

**Comorbidity features (round 29).** 18 canonical comorbidity flags from Tran 2022's S1 Table — `COPD`, `CHF`, `MI`, `HYPERTENSION`, `PERIPHERALVASCULARDISEASE`, `ESRD`, `CIRRHOSIS`, `DIABETESMELLITUS`, `BLEEDINGDISORDER`, `DISSEMINATEDCANCER`, `ALCOHOLUSEDISORDER`, `MENTALPERSONALITYDISORDER`, `SUBSTANCEABUSEDISORDERDRUG`, `ATTENTIONDEFICITDISORDER`, `DEMENTIA`, `ADVANCEDDIRECTIVELIMITINGCARE`, `FUNCTIONALLYDEPENDENTHEALTHSTATUS`, `SMOKINGSTATUS`. Built by `src/trauma_ml/comorbidities.py`, which handles BOTH NTDB storage patterns: wide flag columns directly on `PUF_TRAUMA` (older years) AND a long-format `PUF_PREEXISTINGCONDITION.csv` table (newer years), with case-insensitive matching across all known column-name variants and condition-string descriptions. Anything not found is filled with 0. Build log shows per-canonical coverage so you can see which years contributed which flags. To debug a specific year's NTDB layout, run `python -m trauma_ml.cli.inspect_ntdb data/NTDB/PUF_AY_2021/`. **Requires rebuilding the parquet.**

**Whitelist filter is now case-insensitive (round 28 fix).** The build pipeline's `apply_whitelist()` previously did case-sensitive intersection — `race` (lowercase, NTDB-shipped) didn't match `RACE` (uppercase, predictor list), so the column got silently deleted at parquet write time. Now matches case-insensitively AND unconditionally retains all 22 Barell features + 18 comorbidity flags + 6 ED-arrival vitals + demographics. Build log prints which Barell + comorbidity columns survived so silent drops become visible.

---

## Round-26 changes summary

Quick reference for the new behaviours added in round 26:

**Predictor lists expanded.** `On-scene` now includes 18 comorbidity flags (smoking, COPD, CHF, MI history, hypertension, ESRD, cirrhosis, diabetes, bleeding disorder, disseminated cancer, alcohol/substance abuse, dementia, advance directive, dependent functional status), plus race / ethnicity / insurance, plus mechanism / intent. `On-scene + ED arrival` adds the `ED*` vital columns explicitly (`EDSBP`, `EDPULSERATE`, `EDTEMPERATURE`, `EDRESPIRATORYRATE`, `EDOXYGENSATURATION`, `EDGCSTOTAL`) on top of the `*FIRST` ones. `On-scene + ED arrival + In-hospital` adds highest/lowest 24-hour vitals (`SBPHIGHEST`, `RRHIGHEST`, `GCSHIGHEST`, plus *LOWEST). All matches Tran 2022's S1 Table.

**Semantic categorical override.** `SEX`, `TRAUMATYPE`, `MECHANISM`, `INTENT`, `RACE`, `ETHNICITY`, `PRIMARYINSURANCE`, and all 18 comorbidity flags are now mode-imputed regardless of stored dtype. Median-imputing `SEX` (1=Male, 2=Female) made no statistical sense; this fixes that.

**Resume tracker.** Combos that already wrote `outputs/<target>/metrics/<model_id>/overall__test.json` are skipped on resubmission. So if a slurm hits walltime mid-grid, just resubmit the same slurm — it picks up where it left off. Pass `--no-resume` to retrain from scratch.

**Hyperparameter tuning.** Pass `--tune-hparams` to wrap each fit in `RandomizedSearchCV` with per-family search distributions (xgboost, lightgbm, catboost, random_forest, logistic_l1, logistic_elasticnet). Defaults: 10 random points × 3 CV folds. Tran 2022 reported "negligible impact"; default is OFF for walltime reasons.

**ROC/PR baseline overlay.** When the target is binary mortality, the `plots/<model_id>/roc_pr_curves.png` file now overlays ISS / NISS / TRISS curves on the same axes, computed on the same y_true. Test = solid, holdout = dashed. Each baseline shows its AUC + valid-row count in the legend.

**Subgroup balanced_accuracy worst-case Pareto for multiclass.** For `iss_band` / `niss_band` targets, the aggregator now ranks models by worst-case `balanced_accuracy` across all demographic subgroups in test + holdout (matching the binary AUPRC convention).

**SHAP runs by default.** `--no-shap` is no longer in any shipped slurm — every model produces SHAP bar + beeswarm plots. Pass `--no-shap` manually if you hit a segfault on a particular family + glibc combo.

**ISS/NISS slurms narrowed.** `slurms/2[0-5]_train_*.slurm` now request only `On-scene` and `On-scene + ED arrival` cutoffs (no `iss_only` / `triss_inputs` because those are circular for severity-band targets). Mortality slurms drop `iss_only` / `niss_only` / `triss_inputs` and use `--missingness-thresholds 10 50` (down from 25/50/85). Per-family fits drop from 378 to 144 (288 for logistic).

**FastAPI deployment scaffold.** New `src/trauma_ml/app/` subpackage with a working FastAPI service that loads ONE model bundle and exposes `GET /healthz`, `GET /metadata`, `POST /predict`, `POST /predict/batch`. Install with `pip install -e .[serve]` and run `uvicorn trauma_ml.app.main:app --host 0.0.0.0 --port 8000`. Set `TRAUMA_ML_MODEL_DIR=/path/to/outputs/<target>/models/<model_id>` to pick the bundle. A Dockerfile is included at `src/trauma_ml/app/Dockerfile`.

---

## Common errors & fixes (encoded from the first run)

| Error | Root cause | Fix |
|---|---|---|
| `Elapsed=00:00:00 ExitCode=1:0` immediately on submit | `slurms/logs/` directory missing | `mkdir -p slurms/logs` then resubmit |
| `error: invalid partition specified: gpu` | Slurm has no `gpu` partition; everything's on `general` | The shipped slurms already use `general` — but if you copy from a generic template, fix it |
| `error: QOSMaxWallDurationPerJobLimit` on submit | Slurm's `--time` exceeds the `MaxWall` of the chosen `--qos` | Match each slurm's `--qos` to its `--time`. atlas-edr defaults: `regular`=1d, `long`=2d, `xlong`=8d. The shipped slurms are already aligned (round 24); only an issue if you change `--time` and forget to update `--qos`. |
| `Requested node configuration is not available` | Asking for more RAM than any node has | RTX 3090 nodes have only 95 GB; cap GPU jobs at `--mem=80G` |
| `error while loading shared libraries: libpython3.10.so.1.0` | Lmod modules not loaded in current shell | `module purge; module load Python/...; module load CUDA/...` then activate venv |
| `/dipc/mcapo/trauma_ml_venv/bin/activate: No such file` (from compute node) | $HOME / /dipc not visible from compute | Build venv on `/scratch/$USER/` instead |
| `pyarrow.lib.ArrowTypeError: Expected bytes, got 'float'` mid-build | Mixed string-NaN columns can't be parquet-written | Already fixed in `ntdb_loader.py` round 13 |
| `pip install -r requirements.txt` fails on libcst with "can't find Rust compiler" | pyarrow 24.x has no cp310 wheel, falls back to source | `--prefer-binary` + `pyarrow<20` pin (already in requirements.txt) |
| `import tpot` fails with `ModuleNotFoundError: pkg_resources` | setuptools 81+ removed pkg_resources; tpot's `stopit` dep needs it | `pip install "setuptools<81"` (already in requirements.txt) |
| `XGBClassifier.fit() got an unexpected keyword argument 'callbacks'` (FLAML) | FLAML 2.1.x bundled XGBoost integration uses old `callbacks=` API; XGBoost 3.x removed it | `models.py::_FLAMLWrapper.fit` now passes an `estimator_list` that EXCLUDES xgboost (round 16). FLAML still searches lgbm + RF + extra_tree + LR + catboost + kneighbor. |
| `TPOTEstimator.__init__() got an unexpected keyword argument 'scoring'` | TPOT 1.x changed API: `scoring=` → `scorers=` + `scorers_weights=`, dropped `generations`/`population_size`, requires `search_space=` | `models.py::_TPOTWrapper.fit` now tries the 1.x API first, falls back to 0.12.x on TypeError (round 16). |
| `Violación de segmento` (segfault) AFTER successful first model on long grids | SHAP TreeExplainer + glibc<2.28 + many-tree forests crash on native side; matplotlib accumulates figures across models | `--no-shap` flag (round 16) skips SHAP entirely. Already enabled on the heavy-grid slurms (lightgbm/xgboost/catboost/flaml/random_forest). `plt.close("all")` between models also added in the per-model loop. |
| `KeyError: 'survival_time'` from `target_specs[combo["target"]]` | Survival end-to-end isn't implemented — target builder raises NotImplementedError, cox_ph factory needs (event,time) tuples | Survival slurm `19_train_survival.slurm` is COMMENTED OUT in `submit_all.sh` (round 16). Re-enable when survival pipeline is fully wired. The CLI now emits a clearer error pointing at the YAML config. |
| `outputs/models/` shows `model_0000`, `model_0001`, … instead of `lgr_0000`, `lgb_0000`, etc | Old src.zip without `--model-id-prefix` plumbed through; OR per-job slurm omits the flag | Every shipped slurm now passes `--model-id-prefix <prefix>`. **Wipe `outputs/models/` and `outputs/metrics/` before re-running** to avoid the mix of old and new model_ids contaminating the aggregator. |
| SHAP plot shows `INC_KEY`, `TEACHINGSTATUS`, `STATEDESIGNATION`, `VERIFICATIONLEVEL`, `INTERFACILITYTRANSFER` as top features | Predictor whitelist wasn't filtering admin/ID columns | Round 18+19: hard `NON_PREDICTOR_COLUMNS` blacklist applied (case-insensitive) in both `catalogue.py::variables_for()` AND `trainer.py::_select_predictors()`. After deploying the new src.zip, **wipe `outputs/{models,metrics,plots}/` before resubmitting** — old runs' configs persist on disk and confuse downstream consumers. |
| `KeyError: 'iss_band'` (or `'niss_band'`) when running multiclass slurms | Target spec not added to `config/experiment_grids.yaml` | See "Multiclass targets" section above for the YAML snippet to paste in. The trainer's `_parse_target_defs` looks up by name; missing entries fail loudly. |

---

## Useful one-liners

```bash
# Re-enter the working environment from any fresh shell
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1
source /scratch/$USER/trauma_ml_venv/bin/activate
cd /scratch/mcapo/trauma_ml_v1/

# Or define this once in your ~/.bashrc:
trauma_env() {
  module purge
  module load Python/3.10.4-GCCcore-11.3.0
  module load CUDA/12.1.1
  source /scratch/$USER/trauma_ml_venv/bin/activate
  cd /scratch/$USER/trauma_ml_v1/
}
# Then just type `trauma_env` anytime.

# Check a single trained model's results
ls outputs/metrics/lgb_0000/
cat outputs/metrics/lgb_0000/overall__test.json | python -m json.tool

# Re-aggregate without retraining (e.g. after editing the aggregator)
sbatch slurms/99_aggregate.slurm

# Cancel everything
squeue -u $USER -h -o "%i" | xargs -r scancel
```
