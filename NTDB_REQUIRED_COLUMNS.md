# NTDB columns required by trauma_ml

This document lists every NTDB column the pipeline expects to find and what each is used for. Use it to verify your raw NTDB CSVs and check the build output before training.

The pipeline tolerates **case-insensitive** matches and **alias matches** for the canonical names — e.g. `AGEyears` (NTDB AY 2019 actual name) → `AGEYEARS`, `TOTALGCS` → `GCSTOTAL`. See `harmonise_year` in `ntdb_loader.py` for the full alias map.

## Columns the user must verify exist in raw NTDB CSVs

These are checked in `PUF_TRAUMA*.csv` for each admission year. Names below are the **canonical** names; aliases that get auto-renamed are listed in parentheses.

### Demographics (mandatory — needed for cohort definitions, stratification, TRISS)

| Canonical | Aliases auto-resolved | Used for |
|-----------|----------------------|----------|
| `AGEYEARS` | `AGEyears`, `AGE_YEARS`, `AGE_YRS`, `AGE_IN_YEARS`, `AGEINYEARS`, `AGE`, `PATIENTAGE`, `PT_AGE_YR` | TRISS age cutoff (≥55), stratification, all cohorts |
| `SEX` | (none) | Subgroup analysis |

### On-scene physiology (mandatory — needed for `onsite_complete` cohort and TRISS)

| Canonical | Aliases auto-resolved | Used for |
|-----------|----------------------|----------|
| `GCSTOTAL` | `TOTALGCS`, `GCS_TOTAL`, `GCS` | TRISS RTS component, baseline cohort |
| `SBPFIRST` | `SBP`, `FIRSTSBP`, `SBP_FIRST`, `INITIALSBP` | TRISS RTS component, baseline cohort |
| `RRFIRST` | `RESPIRATORYRATE`, `RR`, `FIRSTRR`, `RR_FIRST`, `INITIALRR`, `RESP_RATE` | TRISS RTS component, baseline cohort |

### ED-arrival physiology (optional — extends `ed_complete` cohort if present)

| Canonical | Used for |
|-----------|----------|
| `TEMPERATURE` | `ed_complete` cohort definition |
| `PULSEOXIMETRY` | `ed_complete` cohort definition |
| `PULSERATE` | `ed_complete` cohort definition |

### Anatomy scores (mandatory — needed for ISS/NISS baselines)

| Canonical | Source | Used for |
|-----------|--------|----------|
| `ISS` | `PUF_TRAUMA.csv` (also `ISS_05` → `ISS` in AY 2019) | ISS baseline (≥16 → death) |
| `NISS` | **DERIVED from `PUF_AISDIAGNOSIS.csv`** | NISS baseline (≥16 → death) |

⚠ `NISS` is **not** an NTDB raw column — it is computed by the pipeline as the sum of squares of the three highest `AISSEVERITY` values per patient. This requires `PUF_AISDIAGNOSIS.csv` to be present at the path expected by `ntdb_tables["aisdiagnosis"]` in your build config. **If this file is missing, `NISS` will be 100% NaN in the output parquet.**

### Mechanism (mandatory — needed for TRISS blunt/penetrating split)

| Canonical | Source | Used for |
|-----------|--------|----------|
| `TRAUMATYPE` | **JOINED from `PUF_ECODE_LOOKUP.csv`** via `PRIMARYECODEICD10` | TRISS coefficient selection |
| `MECHANISM` | Same join | (Future use) |
| `INTENT` | Same join | (Future use) |
| `PRIMARYECODEICD10` | `PUF_TRAUMA.csv` | Join key for the above |

⚠ `TRAUMATYPE`, `MECHANISM`, `INTENT` are **not** raw NTDB PUF_TRAUMA columns — they come from joining `PUF_ECODE_LOOKUP.csv`. **If this file is missing, all three will be 100% NaN and TRISS will fall back to "blunt" coefficients for every patient** (a warning is logged).

The lookup CSV must have a join key in one of these names (case-insensitive — the loader handles `ECode`, `ecode`, `ICD10ECode`, etc.):
1. `ICD10ECODE` (some NTDB releases)
2. `ICDECODE`
3. `ECODE` ← **actual NTDB AY 2021/2022 uses `ECode`** (mixed case, resolved automatically)
4. `PRIMARYECODEICD10`

### Outcome variables (mandatory — needed for target construction)

| Canonical | Used for |
|-----------|----------|
| `HOSPDISCHARGEDISPOSITION` | Mortality outcome (value 5 = deceased) |
| `EDDISCHARGEDISPOSITION` | Mortality outcome (value 5 = deceased in ED) |
| `DEATHINED` | AY 2019 only — mapped to `EDDISCHARGEDISPOSITION=5` |

### Identity / partitioning

| Canonical | Used for |
|-----------|----------|
| `INC_KEY` (or `inc_key`) | Patient row identifier — joins `PUF_AISDIAGNOSIS` to `PUF_TRAUMA` |
| `__admission_year` | Added by `harmonise_year` — used for temporal holdout |

## How to verify your build output

After running `trauma-build`, the parquet should contain **all** of the canonical names above with **non-zero non-NaN counts**. Run this Python snippet against the output:

```python
import pyarrow.parquet as pq
pf = pq.ParquetFile("path/to/unified_train.parquet")
df = pf.read(columns=[
    "AGEYEARS", "SEX",
    "GCSTOTAL", "SBPFIRST", "RRFIRST",
    "ISS", "NISS",
    "TRAUMATYPE", "MECHANISM", "INTENT",
    "HOSPDISCHARGEDISPOSITION", "EDDISCHARGEDISPOSITION",
]).to_pandas()

for col in df.columns:
    pct_nan = 100 * df[col].isna().mean()
    flag = "❌" if pct_nan > 99 else ("⚠" if pct_nan > 30 else "✓")
    print(f"  {flag} {col}: {pct_nan:.1f}% NaN")
```

If you see ❌ for `NISS`, your `PUF_AISDIAGNOSIS.csv` is missing or the path is wrong.
If you see ❌ for `TRAUMATYPE`, your `PUF_ECODE_LOOKUP.csv` is missing or the path is wrong.
If you see ❌ or column-not-present for `AGEYEARS`, check whether your CSV uses `AGEyears` (mixed case — fixed in current alias resolver) or some other variant.

## Build log lines to watch for

`trauma-build` should log the following INFO lines per year. If you see the WARNING variants instead, fix the corresponding CSV:

| Good (INFO)                                                                          | Bad (WARNING)                                              |
|--------------------------------------------------------------------------------------|------------------------------------------------------------|
| `ECODE join complete: <N> / <M> rows matched (<P>%) — TRAUMATYPE/MECHANISM/INTENT populated` | `No PUF_ECODE_LOOKUP provided — TRAUMATYPE/MECHANISM/INTENT will be NaN` |
| `AY <year>: aliased AGEyears -> AGEYEARS (case-insensitive match for canonical name)` | `AY <year>: AGEYEARS not present in CSV after alias resolution` |
| (NISS derivation produces no INFO log on success — verify by checking parquet)       | NISS will be silently 100% NaN if `PUF_AISDIAGNOSIS.csv` not found |
