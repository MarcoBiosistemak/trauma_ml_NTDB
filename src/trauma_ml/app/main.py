"""FastAPI service for trauma_ml model serving.

One service instance loads ONE model bundle.  The bundle is a directory
produced by the trainer at:

    outputs/<target>/models/<model_id>/
        ├── model.joblib              ← the fitted estimator
        ├── transformers.joblib       ← LabelEncoder dict + imputer
        └── config.json               ← predictor list, target spec, etc.

Path is set via the TRAUMA_ML_MODEL_DIR environment variable.

Endpoints
---------
GET  /healthz             — liveness, returns {"status": "ok"}
GET  /metadata            — model card with predictors, target, cfg
POST /predict             — single-row inference, returns class + proba
POST /predict/batch       — multi-row inference, same per-row shape
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "FastAPI / pydantic not installed.  Install with `pip install "
        "'trauma_ml[serve]'` or `pip install fastapi uvicorn pydantic`."
    ) from e

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

class ModelBundle:
    """Lazily-loaded singleton holding the fitted artefacts."""
    model: Any
    transformers: dict
    config: dict
    predictor_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_inverse: dict[int, str]

    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir.resolve()
        if not self.bundle_dir.is_dir():
            raise FileNotFoundError(
                f"Model bundle directory not found: {self.bundle_dir}"
            )
        for fname in ("model.joblib", "transformers.joblib", "config.json"):
            if not (self.bundle_dir / fname).exists():
                raise FileNotFoundError(
                    f"Missing required artefact in {self.bundle_dir}: {fname}"
                )

        log.info("Loading model bundle from %s", self.bundle_dir)
        self.model = joblib.load(self.bundle_dir / "model.joblib")
        self.transformers = joblib.load(self.bundle_dir / "transformers.joblib")
        with open(self.bundle_dir / "config.json") as f:
            self.config = json.load(f)

        self.predictor_cols = self.config["predictor_cols"]
        self.numeric_cols = self.config.get("numeric_cols", [])
        self.categorical_cols = self.config.get("categorical_cols", [])
        # target_inverse keys are JSON strings ("0", "1") — coerce to int
        ti = self.config.get("target_inverse", {})
        self.target_inverse = {int(k): v for k, v in ti.items()}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the trained pipeline (encoder + imputer) to raw input."""
        # Restrict to predictor cols (extras silently dropped, missing -> NaN)
        out = pd.DataFrame(index=df.index)
        for c in self.predictor_cols:
            out[c] = df[c] if c in df.columns else np.nan

        # Apply categorical encoders
        for col, enc in self.transformers.get("encoders", {}).items():
            if col not in out.columns:
                continue
            non_null = out[col].notna()
            if non_null.any():
                vals = out.loc[non_null, col].astype(str).to_numpy()
                # Unknown categories -> a fallback (use first known class as
                # least-bad default; production deployments should add a
                # proper "unknown" bucket during training).
                known = set(enc.classes_)
                vals = np.array([v if v in known else enc.classes_[0] for v in vals])
                out.loc[non_null, col] = enc.transform(vals)
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # Apply imputer (Imputer dataclass from trauma_ml.imputation)
        imputer = self.transformers.get("imputer")
        if imputer is not None:
            out = imputer.transform(out)

        return out[self.predictor_cols]

    def predict(self, df: pd.DataFrame) -> dict[str, list]:
        X = self.transform(df)
        try:
            proba = self.model.predict_proba(X)
        except AttributeError:
            proba = None
        pred = self.model.predict(X)
        return {
            "prediction":       pred.tolist(),
            "prediction_label": [self.target_inverse.get(int(p), str(p))
                                  for p in pred],
            "probability":      proba.tolist() if proba is not None else None,
            "n_rows":           int(len(df)),
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_bundle: ModelBundle | None = None


def _get_bundle() -> ModelBundle:
    global _bundle
    if _bundle is None:
        bundle_dir = os.environ.get("TRAUMA_ML_MODEL_DIR")
        if not bundle_dir:
            raise HTTPException(
                500,
                "TRAUMA_ML_MODEL_DIR env var not set; cannot locate model.",
            )
        _bundle = ModelBundle(Path(bundle_dir))
    return _bundle


app = FastAPI(
    title="trauma_ml inference API",
    description=(
        "REST interface for serving a trained NTDB mortality / "
        "severity-band classifier produced by `trauma-train`. "
        "Set TRAUMA_ML_MODEL_DIR to point at a model bundle directory."
    ),
    version="0.1.0",
)


# ---- Request / response schemas ---------------------------------------------

class PredictRequest(BaseModel):
    """Single-row prediction.

    Field names should match the bundle's predictor_cols.  Missing fields
    are imputed using the same imputer fitted at training time.
    """
    features: dict[str, Any] = Field(
        ...,
        description="One row of features keyed by NTDB column name "
                    "(e.g. AGEYEARS, GCSTOTAL, SEX, TRAUMATYPE).",
    )


class PredictBatchRequest(BaseModel):
    """Multi-row prediction."""
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=10_000)


class PredictResponse(BaseModel):
    prediction:       list[int]
    prediction_label: list[str]
    probability:      list[list[float]] | None = None
    n_rows:           int


# ---- Endpoints --------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe — does NOT load the model."""
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """Return the model card embedded in the bundle's config.json."""
    bundle = _get_bundle()
    cfg = bundle.config
    return {
        "model_id":        cfg.get("model_id"),
        "model_family":    cfg.get("family"),
        "task":            cfg.get("task"),
        "target":          cfg.get("target_spec", {}).get("name"),
        "n_predictors":    cfg.get("config", {}).get("n_predictors", len(bundle.predictor_cols)),
        "predictor_cols":  bundle.predictor_cols,
        "numeric_cols":    bundle.numeric_cols,
        "categorical_cols": bundle.categorical_cols,
        "target_inverse":  bundle.target_inverse,
        "bundle_dir":      str(bundle.bundle_dir),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    bundle = _get_bundle()
    df = pd.DataFrame([req.features])
    result = bundle.predict(df)
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(req: PredictBatchRequest) -> PredictResponse:
    bundle = _get_bundle()
    df = pd.DataFrame(req.rows)
    result = bundle.predict(df)
    return PredictResponse(**result)
