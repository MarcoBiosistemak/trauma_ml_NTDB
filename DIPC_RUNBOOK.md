# DIPC Runbook — trauma_ml

Operational guide for building, training (all options per target), and deploying
on the DIPC cluster. Commands run at the bash `$` prompt (not the Python `>>>`),
single-line (no stray `\`).

## 0. Install
The editable install **and** torch are done inside `setup_env.slurm` (it loads
the Python module, creates the venv on `/scratch`, then runs `pip install -e .`
and `pip install torch`). So normally you just:
```bash
sbatch slurms/setup_env.slurm
```
Do NOT run `pip ...` at the bare login shell — pip isn't on PATH there until the
Python module is loaded and the venv is activated (that's what the job does).
To reinstall manually after a package update, first:
```bash
module load Python/3.10.4-GCCcore-11.3.0
source /scratch/$USER/trauma_ml_venv/bin/activate
pip install -e .          # from the repo root
```

## 1. Build the unified dataset
```bash
trauma-build --years 2019 2020 2021 2022 2024 --holdout-years 2024 --anonymize --verbose
```
Writes `unified_train.parquet` + `unified_holdout.parquet`. The build harmonises
aliases, derives NISS / Barell / INJ_* / comorbidities, and applies the
predictor whitelist. Check the log for non-zero Barell rates and the target
prevalence lines.

## 2. Train (one combo, sanity check)
```bash
trauma-train --dataset unified_train.parquet --holdout-dataset unified_holdout.parquet \
  --targets in_hospital_mortality --model-families xgboost \
  --phase-cutoffs "On-scene + ED arrival + In-hospital" \
  --inclusion-strategies "Tran NTDB" --predictor-types "Tran NTDB" \
  --imputers median_mode --calibrations isotonic --augmentations null \
  --missingness-thresholds 50 --bootstrap-ci 500 --verbose
```

### Run ALL options for a target
Use the family slurms (below) or expand the axes:
```
--phase-cutoffs "On-scene" "On-scene + ED arrival" "On-scene + ED arrival + In-hospital"
--imputers median_mode mice bagged_trees missforest gradient_boosting   # (+ none for NaN-native families)
--calibrations none platt isotonic
--augmentations null smote adasyn
--missingness-thresholds 10 50
--bootstrap-ci 1000                              # heavy all-metric JSON CIs
# --plot-ci-bootstrap defaults to 200 (binary ROC/PR legend CIs, on)
# pre-filter + phase-complete slice are ALWAYS on
```
Phase cutoff strings must be exact: `On-scene`, `On-scene + ED arrival`,
`On-scene + ED arrival + In-hospital`.

**Preprocessing/model cache (on by default).** The runner fits encoders+imputer
once per `(target,predictor_type,phase,inclusion,missingness,imputer,family)` group
and reuses them across all calibration+augmentation variants; for binary targets it
also trains the base model once per augmentation and reuses it across
`none/platt/isotonic`. A slow imputer is therefore fit ~9× fewer times and binary
base-model training runs ~3× fewer times, with **byte-identical** metrics/artifacts
to a no-cache run (verified). The resume tracker still skips already-completed
combos, so this composes with auto-resubmit-on-timeout. Pass `--no-prep-cache` to
force a fresh fit per combo if you ever need to isolate a combo. Each saved
`model_id` still bundles its own imputer, so deployment is unaffected.

## 3. SLURM sweep
`slurms/` holds one file per family (mortality 10–18, band 20–29, band-binary
31–32, Doshi 30/33/34, **Doshi-ICD 35**). Each runs:
- **base** (e.g. `xgb`): all five imputers `median_mode mice bagged_trees
  missforest gradient_boosting`.
- **`_none`** (e.g. `xgb_none`): `--imputers none` — NaN-native families only.

(The imputer-check `_ic` blocks were retired; imputer-check remains available
via `--imputer-check` if you want it.)

**Submit everything (after the build finishes) with the helper, which sets the
build dependency and the final aggregate automatically:**
```bash
bash slurms/submit_all.sh
```
If you submit by hand instead, make sure the glob reaches **35** (a plain
`3[0-4]_*` misses the Doshi-ICD job):
```bash
for f in slurms/1[0-9]_*.slurm slurms/18[a-h]_*.slurm \
         slurms/2[0-9]_*.slurm slurms/3[0-5]_*.slurm; do sbatch "$f"; done
```

### atlas-edr resource notes (baked into the headers)
- **Tiered QOS**: fast families use `--qos=regular` (`--time=1-00:00:00`, 1-day
  cap); slow ones (`flaml`, `tpot`, `random_forest`, and the ISS/NISS variants
  26–29) use `--qos=long` (`--time=2-00:00:00`, 2-day cap). `xlong` is no longer
  used. Each training job carries a pure-bash **auto-resubmit-on-timeout**
  watchdog: ~4 min before the wall limit it `sbatch`es itself and exits 0, and the
  resume tracker skips already-finished combos, so a timeout is lossless. The
  build job (`00_*`) has **no** watchdog, so its `afterok` dependency is honest.
- **rtx3090 GPU nodes have only ~95 GB RAM** → GPU jobs request `--mem=90G`
  (asking for 120 G there gives *"Requested node configuration is not
  available"*). DL families (`tabnet`, `tabpfn`, `doshi_*`) use `gpu:rtx3090:1`.
- **Tree models run on CPU** (no `--gres`): `lightgbm`/`xgboost`/`catboost`
  fall back gracefully via `--use-gpu auto` and land on the 94-node CPU pool,
  freeing the 7 rtx3090 nodes for the DL families.
- **Big-memory CPU jobs** (`tpot` 200 G, `random_forest` 250 G) schedule on the
  385 GB p40 nodes (they don't use the GPU); this is expected.

`tabnet` is sliced (`18a–h`). Submit with `slurms/submit_all.sh`.

### Band-binary targets are sliced too (`31a–l`, `32a–l`)
`ISS_band_binary` / `NISS_band_binary` are the two largest sweeps (**2,646
combos each**) and were originally one slurm apiece containing *two* sequential
`trauma-train` blocks — so the second block (`*_bin_other`, 1350 combos) could
not start until the first (`*_bin_boost`, 1296) fully finished. Both blocks are
now split into 6 parallel slices each:

| slices | prefix | families | combos |
|--------|--------|----------|--------|
| `31a–f` / `32a–f` | `*_bin_boost` | xgboost, lightgbm, catboost, flaml (+`imputer=none`) | 1296 |
| `31g–l` / `32g–l` | `*_bin_other` | logistic_l1, logistic_elasticnet, random_forest, tpot, tabnet | 1350 |

Each slice enumerates the **same full grid** in the same deterministic order and
runs only its `--start-id..--end-id` range, so `model_id`s stay canonical, slices
write to disjoint directories, and any pre-existing work is skipped by the resume
tracker. The monolithic `31_`/`32_` slurms are superseded and no longer submitted
by `submit_all.sh` (kept on disk for reference only).

**TabNet: run the 8 slices, never `18_train_tabnet.slurm`.** `18a`-`18h` carry
`--start-id/--end-id` covering 0..359 contiguously and disjointly. The base
`18_train_tabnet.slurm` has no range, so it enumerates the whole grid and
overlaps all eight; the resume tracker only skips *finished* combos, so two live
jobs can start the same combo and write the same `metrics/<id>/*.json`
concurrently, producing truncated JSON. It is therefore marked SUPERSEDED and
exits immediately if submitted. The slices were also raised from 64 GB (which
caused `ExitCode 137` OOM kills after 1-2 h) to 90 GB - the practical maximum on
a ~95 GB rtx3090 node, which TabNet needs for its GPU.
```bash
for f in slurms/18[a-h]_train_tabnet.slurm; do sbatch "$f"; done
```

**The CPU tree slurms also request 250 GB.** `11_train_lightgbm`,
`12_train_xgboost`, `13_train_catboost` (mortality) and `20`/`21`/`23`/`24`
(ISS/NISS) originally asked for 90 GB and were killed with `OUT_OF_MEMORY`
after 7-9 hours of work (observed on `trauma_cb` and `trauma_xgb`). They carry
no `--gres`, so they now request `--mem=250G` / `--cpus-per-task=20` and land on
the 385 GB p40 nodes. The `doshi_*` slurms keep 90 GB **and** their
`gpu:rtx3090:1` request: rtx3090 nodes only have ~95 GB, so raising their memory
would make them unschedulable, and they have not hit OOM.

**All 24 band-binary slices are CPU/big-memory jobs (`--mem=250G`, no `--gres`).**
The band-binary cohort is ~3.94M rows (vs ~2.06M for mortality), and the load +
inclusion-filter step alone copies the frame — one observed failure was a plain
`numpy MemoryError` ("Unable to allocate 4.20 GiB for an array with shape
(143, 3941870)") inside `inclusion.apply()`, i.e. *before any model was trained*,
with MaxRSS already at 83.5 GB of a 90 GB request. 90 GB is simply not enough for
this target regardless of model family, so both the `_boost` and `_other` blocks
request 250 GB and land on the 385 GB p40 nodes.

**The `*_bin_other` slices additionally disable SHAP.** They contain
`random_forest`/`tpot`/`tabnet`, which peak well above 90 GB during evaluation
(observed: killed at MaxRSS ~75 GB with a 90 G request, i.e. the true peak spiked
past the limit between samples). They therefore carry **no `--gres` line** and
request `--mem=250G` / `--cpus-per-task=20`, matching `16_train_random_forest.slurm`,
so they land on the 385 GB p40 nodes.

Do **not** try to fix an OOM here from the command line: `--mem=250G` alone is
rejected (*"Requested node configuration is not available"*) if the script still
requests `--gres=gpu:rtx3090:1`, since no rtx3090 node has 250 GB, and this
cluster rejects both `--gres=NONE` (*"Invalid generic resource"*) and `--gres=gpu:0`
(*"You must always specify a type when requesting gpu GRES"*). A GRES request can
only be removed by editing the file — which is already done in these twelve.

They also pass **`--no-shap`**. SHAP's tree explainers are native C++ extensions and
**segfault** ("Violación de segmento" / SIGSEGV) on `random_forest`/`tpot` fitted to
millions of rows — the crash lands immediately after the holdout confusion matrix,
which is the call right before `plot_shap`. Because the failure is in native memory
it is *not* fixed by raising `--mem` (we saw SIGKILL at 90 G become SIGSEGV at 250 G).
`16_train_random_forest.slurm` / `22_` / `25_` already carry `--no-shap` for the same
reason. All other plots (ROC/PR, calibration, confusion matrices) are unaffected.

## 4. Outputs (per `model_id`)
```
metrics/<id>/overall__{train,calibration,test,holdout_year}.json
metrics/<id>/overall__{test,holdout}__phase_complete.json   # round 59 slice
metrics/<id>/overall__<part>__{platt,isotonic}.json
metrics/<id>/subgroups/…__subgroups_by_{gender,…,insurance}.csv
imputation_eval/<id>/imputation_eval_<method>.csv            # incl. balanced_accuracy + keep
plots/<id>/roc_pr_curves__{full,baseline_complete,test__phase_complete,holdout__phase_complete}.png
plots/<id>/confusion_matrix_{test,holdout,test__phase_complete,holdout__phase_complete}.png
plots/<id>/calibration_*.png, shap_*.png
models/<id>/artifact.pkl                                     # full deployable pipeline
```
Console messages name every pre-filter removal (variable + reason) and the
phase-complete slice row counts.

## 5. Aggregate

`99_aggregate.slurm` now **discovers targets dynamically** (any
`outputs/*/metrics` directory) instead of using a hardcoded list. The old list
covered only `mortality`, `iss_band` and `niss_band`, so `iss_band_binary` and
`niss_band_binary` -- 2,646 completed models each -- were silently skipped and
produced no `all_metrics.csv`. If you add a target in future, the aggregator
picks it up with no edit.
```bash
trauma-aggregate            # (99_aggregate.slurm) -> consolidated metrics table
```
**Important:** `submit_all.sh` chains `99_aggregate` with `--dependency=afterok`
on the training jobs, but because a timed-out training job auto-resubmits and
exits 0, that dependency can fire on *partial* results. So after `squeue -u $USER`
is **empty**, re-run the aggregate by hand to pick up everything:
```bash
sbatch slurms/99_aggregate.slurm      # or: trauma-aggregate
```
Note: per-model JSONs carry the CI columns; the consolidated CSV may not — read
`overall__*.json` for `<metric>__ci_low/__ci_high`.

## 5.4 Repair corrupt metrics before aggregating

Interrupted jobs (walltime kill, cancellation, a full filesystem) can leave a
metrics JSON zero-length or truncated. There are two distinct cases, and only
one heals itself:

* `overall__test.json` is the **resume sentinel**, and `run_experiments.py`
  checks `exists() AND st_size > 0` — so a zero-length sentinel is re-run
  automatically.
* **Any other JSON** (`overall__holdout.json`, `overall__test__platt.json`,
  `config.json`, `cohort_counts.json`, …) is **not** size-checked. A corrupt one
  leaves the model looking "complete", so it is **skipped forever** by the resume
  tracker, while `trauma-aggregate` dies on it with
  `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.

`scripts/repair_corrupt_metrics.py` finds both cases and invalidates the affected
models so the next run re-trains them. `model_id` is derived from the combo's
*position in the grid*, and the grid is enumerated deterministically, so a
deleted model is regenerated with **exactly the same id and configuration** —
nothing is renumbered.

Only **general** (whole-cohort) metrics justify a re-train: `overall__train.json`,
`overall__test.json`, `overall__holdout.json`. A missing or corrupt *subgroup*
file is reported as **minor** and is NOT re-trained — the pruner simply finds no
value for that (axis, level, metric) tag, i.e. it behaves as NaN, and ranking
falls back to the general metrics. Calibrated variants (`__platt`/`__isotonic`)
are not required, since they only exist when that model's calibration axis was set.

The default scan is **fast**: one `iterdir()` per model dir, never recursing into
`subgroups*/`. It prints progress every 500 dirs (`--progress-every`). Use
`--deep` only if you also want subgroup CSVs audited (much slower on BeeGFS).

**Scan locally, apply on the cluster** — useful when the cluster filesystem is
slow, or when a local copy still has JSONs the cluster lost:

```bash
# on your laptop, against a downloaded copy:
python3 scripts/repair_corrupt_metrics.py --outputs-root outputs --out bad_ids.txt
# copy bad_ids.txt to the cluster, then:
python3 scripts/repair_corrupt_metrics.py --outputs-root outputs --ids-from bad_ids.txt --apply
```

If the local copy has *good* versions of files the cluster is missing, restoring
them beats re-training — `rsync --ignore-existing` adds only what's absent:
```bash
rsync -av --ignore-existing --include='*/' --include='overall__*.json' --exclude='*' \
      outputs/ mcapo@atlas-edr-login-01:/scratch/mcapo/trauma_ml/outputs/
```

```bash
python3 scripts/repair_corrupt_metrics.py --outputs-root outputs --list-ids   # DRY RUN
python3 scripts/repair_corrupt_metrics.py --outputs-root outputs --apply      # invalidate
for f in slurms/3[12][a-l]_train_*_bin_*.slurm; do sbatch "$f"; done          # re-train
# ...then re-run the aggregator once the queue drains.
```

The aggregator is also hardened (`_safe_load_json`): a corrupt file is now warned
about and skipped rather than aborting the whole run, so one bad file can never
again discard thousands of good models.

## 5.5 Disk cleanup: prune non-best model artifacts

Trained artifacts dominate disk usage (`outputs/<target>/models/<id>/artifact.pkl`;
random-forest ones reach 80+ GiB each). `scripts/prune_non_best_models.py` deletes
**only** the `artifact.pkl` of models that are never the top performer on any
metric — overall **or** within any demographic subgroup, on test **or** external
holdout, for base/platt/isotonic. Metrics, plots and `config.json` are always
kept, so every deleted model can still be re-trained from its saved config.

```bash
module purge && module load Python/3.10.4-GCCcore-11.3.0
source /scratch/$USER/trauma_ml_venv/bin/activate
python3 scripts/prune_non_best_models.py --outputs-root outputs --verbose-keep-reasons  # DRY RUN
python3 scripts/prune_non_best_models.py --outputs-root outputs --apply                 # delete
beegfs-ctl --getquota --uid $USER --storagepoolid=1                                     # verify
```

Safe to re-run periodically: it re-evaluates against whatever is on disk, so as
new combos finish the keep-set stays genuinely "best-only". Row counts
(`n_holdout_rows`, `n_total`, …), thresholds and non-finite (NaN) values are
excluded from ranking, so a model can never be kept for a non-performance reason.

## 6. Deploy (Hugging Face / FastAPI)
Load the **`ModelArtifact`** (`models/<id>/artifact.pkl`), not the bare model —
its `predict` / `predict_proba` apply transformers → imputer → model →
calibration to raw NTDB-style input. The FastAPI app (`app/main.py`,
`app/Dockerfile`) does this; the Dockerfile installs CPU `torch` for
`doshi_ffnn`. The `bagged_trees` imputer is fully picklable (round-trips through
`artifact.pkl`).

## Gotchas
- Reinstall (`pip install -e .`) after every package update.
- `imputer=none` only with `xgboost`/`lightgbm`/`catboost`/`flaml`.
- `EDDISCHARGEHRS` is blacklisted (leakage) — do not re-add it.
- `--model-id-prefix` is a cosmetic label; it changes output folder names only,
  not results.
