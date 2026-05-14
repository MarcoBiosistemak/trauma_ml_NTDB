# trauma_ml

End-to-end machine learning pipeline for trauma mortality and severity-band
prediction on the ACS National Trauma Data Bank (NTDB).  Reproduces and
extends the methodology of Tran et al. 2022 (PLoS ONE,
https://doi.org/10.1371/journal.pone.0276624) with a wider grid of
preprocessing and model families.

## Quickstart

```bash
pip install -e .
python -m trauma_ml.cli.build_dataset --years 2019 2020 2021 2022 2024 \
       --holdout-years 2024
trauma-train --dataset outputs/datasets/unified_train.parquet \
             --targets in_hospital_mortality \
             --model-families xgboost --calibrations none platt isotonic
```

See `DIPC_RUNBOOK.md` for the full cluster deployment guide and
`PIPELINE_OPTIONS.md` for every available CLI flag.

## Data

NTDB data is **not** committed to this repository — it's proprietary
to ACS.  Request access at
https://www.facs.org/quality-programs/trauma/quality/national-trauma-data-bank/.
Place each year's PUF folder into `data/NTDB/PUF_AY_<year>/`.

## Licence

See LICENSE.
