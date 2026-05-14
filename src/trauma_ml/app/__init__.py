"""trauma_ml.app — FastAPI deployment scaffold.

Lets you serve a trained trauma_ml model behind a small REST API.  The
service loads ONE model bundle (joblib + transformers + config.json)
and exposes:

    GET  /healthz         — liveness probe
    GET  /metadata        — model card (target, predictors, training cfg)
    POST /predict         — single-row prediction
    POST /predict/batch   — multi-row prediction

This is a STARTING POINT — the endpoints work out of the box for any
classifier the trainer produces, but you'll likely want to add auth,
rate limiting, request logging, and your own input validation rules
before exposing it to real users.

Run locally with:

    pip install -e .[serve]
    uvicorn trauma_ml.app.main:app \\
        --host 0.0.0.0 --port 8000 \\
        --env-file .env

with `.env` containing at minimum:

    TRAUMA_ML_MODEL_DIR=/path/to/outputs/mortality/models/xgb_0042

See ``trauma_ml/app/main.py`` for the full surface and
``DIPC_RUNBOOK.md`` for the deployment cookbook.
"""
