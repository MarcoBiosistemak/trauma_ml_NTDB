# NTDB Required & Used Columns

This is the authoritative list of NTDB (TQP PUF) columns the pipeline consumes,
how they map to model features, and at which **phase** each becomes available.
It assumes the full pipeline is run for **every target** (mortality, ISS/NISS
band 4-class, ISS/NISS band binary).

All variable names are the pipeline's canonical (post-alias) names. The loader
harmonises common NTDB aliases automatically: `AgeYears→AGEYEARS`,
`TOTALGCS→GCSTOTAL`, `SBP→SBPFIRST`, `RESPIRATORYRATE→RRFIRST`.

## Required source tables (`config/paths.yaml → ntdb_tables`)

| Key | File (per AY) | Used for |
|---|---|---|
| `puf` (main) | `PUF_TRAUMA.csv` | demographics, vitals, anthropometry, payer, disposition |
| `aisdiagnosis` | `PUF_AISDIAGNOSIS.csv` | NISS derivation, AIS severities (band targets) |
| `icddiagnosis` | `PUF_ICDDIAGNOSIS.csv` | Barell matrix + specific-injury (INJ_*) features |
| `ecode` | `PUF_ECODE.csv` | TRAUMATYPE / MECHANISM / INTENT |
| `preexistingconditions` | `PUF_PREEXISTINGCONDITIONS.csv` | 18 comorbidity flags |

## Native columns by phase

### L1 — On-scene / first contact (32 features)
- **Demographics / admin:** `AGEYEARS`, `SEX`, `ETHNICITY`, `PRIMARYMETHODPAYMENT`
  (insurance/payer — also a sociodemographic **subgroup axis**)
- **Anthropometry:** `HEIGHT`, `WEIGHT`
- **Mechanism (from ECODE):** `TRAUMATYPE`, `MECHANISM`, `INTENT`
- **On-scene physiology:** `GCSTOTAL`, `SBPFIRST`, `RRFIRST`
- **On-scene status:** `PREHOSPITALCARDIACARREST`, `TRANSPORTMODE`
- **18 comorbidities:** `SMOKINGSTATUS`, `COPD`, `CHF`, `MI`, `HYPERTENSION`,
  `PERIPHERALVASCULARDISEASE`, `ESRD`, `CIRRHOSIS`, `DIABETESMELLITUS`,
  `BLEEDINGDISORDER`, `DISSEMINATEDCANCER`, `ALCOHOLUSEDISORDER`,
  `MENTALPERSONALITYDISORDER`, `SUBSTANCEABUSEDISORDERDRUG`,
  `ATTENTIONDEFICITDISORDER`, `DEMENTIA`, `ADVANCEDDIRECTIVELIMITINGCARE`,
  `FUNCTIONALLYDEPENDENTHEALTHSTATUS`

### L2 — + At ED arrival (+7 → 39 features)
`TEMPERATURE`, `PULSEOXIMETRY`, `PULSERATE`, `HOSPITALARRIVALHRS`,
`TBIPUPILLARYRESPONSE`, `ALCOHOLSCREEN`, `ALCOHOLSCREENRESULT`

### L3 — + In-hospital (a posteriori)
- **Mortality target (+24 → 63):** `ISS`, `NISS`, + the 22 injury features below.
- **ISS/NISS band targets (+22 → 61):** the 22 injury features only. `ISS`/`NISS`
  are **excluded** because they (or AIS severities) define the band label.

**22 injury features (derived, L3 only):**
`BARELL_TBI`, `BARELL_OTHER_HEAD`, `BARELL_FACE`, `BARELL_NECK`, `BARELL_SCI`,
`BARELL_VERTEBRAL_NO_SCI`, `BARELL_THORAX`, `BARELL_ABDOMEN_PELVIS`,
`BARELL_UPPER_EXTREMITY`, `BARELL_LOWER_EXTREMITY`, `BARELL_BURNS`,
`BARELL_SYSTEM_OR_OTHER`, `INJ_SUBDURAL_HEMORRHAGE`, `INJ_CONCUSSION`,
`INJ_PNEUMOTHORAX`, `INJ_RIB_FRACTURE_MULTIPLE`, `INJ_SPLENIC_LACERATION`,
`INJ_LIVER_LACERATION`, `INJ_PELVIC_FRACTURE`, `INJ_FEMUR_FRACTURE`,
`INJ_DISTAL_RADIUS_FRACTURE`, `INJ_FOOT_FRACTURE`

## Derived (not native) — produced during build
- `NISS` from `PUF_AISDIAGNOSIS`; `ISS` native or derived.
- `BARELL_*` / `INJ_*` from `PUF_ICDDIAGNOSIS` (Barell matrix + ICD patterns).
- `TRAUMATYPE`, `MECHANISM`, `INTENT` from `PUF_ECODE`.
- 18 comorbidity flags from `PUF_PREEXISTINGCONDITIONS`.

## Explicitly EXCLUDED / blacklisted
- `EDDISCHARGEHRS` — **leakage** (an ED death's "ED discharge" is the death
  event; ED length-of-stay is fixed by the disposition). Blacklisted.
- `INC_KEY` and registry IDs — identifiers.
- `HOSPDISCHARGEDISPOSITION`, `EDDISCHARGEDISPOSITION`, `DEATHINED` — outcome
  columns (used to BUILD the mortality target, never as predictors).
- Non-native placeholders that do **not** exist in the PUF: `RACE` (only
  `RACE_*` flags ship), `PRIMARYINSURANCE` (real name is `PRIMARYMETHODPAYMENT`),
  `EDSBP` / `SBPHIGHEST` / `*LOWEST`.

## `PRIMARYMETHODPAYMENT` category codes (all AYs)
`1=Medicaid`, `2=Not Billed`, `3=Self-Pay`, `4=Private/Commercial Insurance`,
`6=Medicare`, `7=Other Government`, `10=Other`.

## Availability notes
- EMS prehospital interval fields exist only for AY 2019–2020 (now included in the build).
- NTDB has **no drug-screen** field — only alcohol (`ALCOHOLSCREEN*`).
- `HOSPITALARRIVALHRS`, `ALCOHOLSCREENRESULT`, `TBIPUPILLARYRESPONSE` are often
  >50% missing and may be dropped by the missingness threshold.
