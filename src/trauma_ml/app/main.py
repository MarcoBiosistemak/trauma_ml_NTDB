"""FastAPI service for trauma_ml — multi-model deployment.

Serves multiple trained model bundles simultaneously.  Each bundle is a
directory produced by the trainer at:

    outputs/<target>/models/<model_id>/
        ├── artifact.pkl              ← the full ModelArtifact
        └── config.json               ← human-readable summary

Set TRAUMA_ML_MODELS as a comma-separated list of model directories
(or model_ids with TRAUMA_ML_OUTPUTS_ROOT pointing at the outputs/ tree).

Endpoints
---------
GET  /                    — interactive demo page (HTML form)
GET  /healthz             — liveness probe
GET  /models              — list loaded models with their settings
GET  /models/{id}/fields  — list of expected input fields for a model
POST /predict/{id}        — single-row inference
POST /predict/{id}/batch  — multi-row inference (up to 10,000 rows)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "FastAPI / pydantic not installed.  Install with `pip install "
        "'trauma_ml[serve]'` or `pip install fastapi uvicorn pydantic`."
    ) from e

from trauma_ml.persistence import ModelArtifact

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — loaded at startup
# ---------------------------------------------------------------------------
_models: dict[str, ModelArtifact] = {}


def _load_models() -> None:
    """Populate _models from environment variables."""
    global _models

    # Option 1: TRAUMA_ML_MODELS = comma-separated list of full paths
    #   e.g.  /path/to/outputs/mortality/models/xgb_none_0000,...
    models_env = os.environ.get("TRAUMA_ML_MODELS", "")

    # Option 2: TRAUMA_ML_MODEL_IDS + TRAUMA_ML_OUTPUTS_ROOT
    #   e.g.  TRAUMA_ML_OUTPUTS_ROOT=/path/to/outputs
    #         TRAUMA_ML_MODEL_IDS=xgb_none_0000,xgb_none_0018,xgb_none_0036
    #         TRAUMA_ML_TARGET=mortality   (default)
    ids_env = os.environ.get("TRAUMA_ML_MODEL_IDS", "")
    root_env = os.environ.get("TRAUMA_ML_OUTPUTS_ROOT", "")
    target_env = os.environ.get("TRAUMA_ML_TARGET", "mortality")

    paths: list[Path] = []
    if models_env:
        paths = [Path(p.strip()) for p in models_env.split(",") if p.strip()]
    elif ids_env and root_env:
        root = Path(root_env)
        for mid in ids_env.split(","):
            mid = mid.strip()
            if mid:
                paths.append(root / target_env / "models" / mid)
    else:
        # Fallback: legacy single-model env var
        single = os.environ.get("TRAUMA_ML_MODEL_DIR", "")
        if single:
            paths = [Path(single)]

    if not paths:
        log.warning("No models configured.  Set TRAUMA_ML_MODEL_IDS + "
                    "TRAUMA_ML_OUTPUTS_ROOT, or TRAUMA_ML_MODELS.")
        return

    for p in paths:
        try:
            artifact = ModelArtifact.load(p)
            _models[artifact.model_id] = artifact
            phase = artifact.config.get("phase_cutoff",
                                         artifact.extras.get("phase_cutoff", "?"))
            log.info("Loaded %s (phase=%s, %d predictors)",
                     artifact.model_id, phase, len(artifact.predictor_cols))
        except Exception as e:
            log.error("Failed to load model from %s: %s", p, e)

    log.info("Loaded %d model(s): %s", len(_models), list(_models.keys()))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="trauma_ml — Trauma Mortality Prediction API",
    description=(
        "REST API serving trained XGBoost mortality classifiers from the "
        "National Trauma Data Bank (NTDB).  Three models are available, "
        "each trained on a different set of clinical predictors corresponding "
        "to increasing stages of hospital care.\n\n"
        "**Visit `/` for an interactive demo form.**"
    ),
    version="1.0.0",
)

# Allow CORS so the interactive page works from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    _load_models()


def _get_model(model_id: str) -> ModelArtifact:
    if model_id not in _models:
        available = list(_models.keys())
        raise HTTPException(
            404,
            f"Model '{model_id}' not found.  Available: {available}",
        )
    return _models[model_id]


# ---- Schemas -----------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single-row prediction.  Keys = NTDB column names."""
    features: dict[str, Any] = Field(
        ...,
        description="One row of features keyed by NTDB column name.",
        json_schema_extra={"example": {
            "AGEYEARS": 45, "SEX": "Male", "TRAUMATYPE": "Blunt",
            "GCSTOTAL": 14, "SBPVALUE": 120, "PULSERATE": 88,
        }},
    )
    calibrated: str | None = Field(
        None,
        description="Calibration to apply: 'platt', 'isotonic', or null "
                    "(uses the raw XGBoost score).  Only matters if the "
                    "model was trained with that calibration method.",
    )


class PredictBatchRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=10_000)
    calibrated: str | None = None


class PredictResponse(BaseModel):
    model_id:         str
    phase_cutoff:     str
    prediction:       list[int]
    prediction_label: list[str]
    probability:      list[list[float]] | None = None
    n_rows:           int


# ---- Endpoints ---------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok", "models_loaded": len(_models)}


@app.get("/models")
def list_models():
    """List all loaded models with their training configuration."""
    out = []
    for mid, art in _models.items():
        cfg = art.config
        out.append({
            "model_id":       mid,
            "family":         art.family,
            "task":           art.task,
            "phase_cutoff":   cfg.get("phase_cutoff", "?"),
            "imputer_method": cfg.get("imputer_method", "?"),
            "calibration":    cfg.get("calibration", "none"),
            "n_predictors":   len(art.predictor_cols),
            "predictor_cols": art.predictor_cols,
        })
    return out


@app.get("/models/{model_id}/fields")
def model_fields(model_id: str):
    """Return the expected input fields for a specific model."""
    art = _get_model(model_id)
    return {
        "model_id":        model_id,
        "phase_cutoff":    art.config.get("phase_cutoff", "?"),
        "numeric_fields":  art.numeric_cols,
        "categorical_fields": art.categorical_cols,
        "all_fields":      art.predictor_cols,
        "note": "You can omit fields — XGBoost handles missing values "
                "natively.  But providing more fields improves accuracy.",
    }


@app.post("/predict/{model_id}", response_model=PredictResponse)
def predict(model_id: str, req: PredictRequest):
    """Predict mortality for a single patient."""
    art = _get_model(model_id)
    df = pd.DataFrame([req.features])
    result_df = art.predict(df, calibrated=req.calibrated)

    proba_cols = [c for c in result_df.columns if c.startswith("proba__")]
    if "predicted_proba" in result_df.columns:
        proba = [[1 - float(result_df["predicted_proba"].iloc[0]),
                   float(result_df["predicted_proba"].iloc[0])]]
    elif proba_cols:
        proba = [result_df[proba_cols].iloc[0].tolist()]
    else:
        proba = None

    pred_class = result_df["predicted_class"].iloc[0]
    pred_int = 1 if str(pred_class).lower() in ("1", "dead", "died", "true", "yes") else 0

    return PredictResponse(
        model_id=model_id,
        phase_cutoff=art.config.get("phase_cutoff", "?"),
        prediction=[pred_int],
        prediction_label=[str(pred_class)],
        probability=proba,
        n_rows=1,
    )


@app.post("/predict/{model_id}/batch", response_model=PredictResponse)
def predict_batch(model_id: str, req: PredictBatchRequest):
    """Predict mortality for multiple patients."""
    art = _get_model(model_id)
    df = pd.DataFrame(req.rows)
    result_df = art.predict(df, calibrated=req.calibrated)

    proba_cols = [c for c in result_df.columns if c.startswith("proba__")]
    if "predicted_proba" in result_df.columns:
        proba = [[1 - float(v), float(v)]
                 for v in result_df["predicted_proba"]]
    elif proba_cols:
        proba = result_df[proba_cols].values.tolist()
    else:
        proba = None

    preds = []
    labels = []
    for pc in result_df["predicted_class"]:
        labels.append(str(pc))
        preds.append(1 if str(pc).lower() in ("1", "dead", "died", "true", "yes") else 0)

    return PredictResponse(
        model_id=model_id,
        phase_cutoff=art.config.get("phase_cutoff", "?"),
        prediction=preds,
        prediction_label=labels,
        probability=proba,
        n_rows=len(df),
    )


# ---- Interactive demo page ---------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def demo_page():
    """Serve an interactive HTML page for trying the models."""
    models_js = json.dumps([
        {
            "model_id": mid,
            "phase_cutoff": art.config.get("phase_cutoff", "?"),
            "n_predictors": len(art.predictor_cols),
            "numeric_fields": art.numeric_cols,
            "categorical_fields": art.categorical_cols,
        }
        for mid, art in _models.items()
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trauma Mortality Prediction — Demo</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
         sans-serif; background: #f5f7fa; color: #333; padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; color: #1a1a2e; }}
  .subtitle {{ color: #666; margin-bottom: 2rem; }}
  .model-selector {{ display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .model-card {{ flex: 1; min-width: 240px; padding: 1.2rem; border: 2px solid #ddd;
                 border-radius: 12px; cursor: pointer; background: white;
                 transition: all 0.2s; }}
  .model-card:hover {{ border-color: #1f4e79; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
  .model-card.selected {{ border-color: #1f4e79; background: #eef4fb; }}
  .model-card h3 {{ font-size: 0.95rem; color: #1f4e79; }}
  .model-card p {{ font-size: 0.82rem; color: #666; margin-top: 0.4rem; }}
  .form-section {{ background: white; padding: 1.5rem; border-radius: 12px;
                   box-shadow: 0 1px 6px rgba(0,0,0,0.06); margin-bottom: 1.5rem; }}
  .form-section h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: #1a1a2e; }}
  .fields-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                  gap: 0.8rem; }}
  .field label {{ display: block; font-size: 0.78rem; font-weight: 600;
                  color: #555; margin-bottom: 0.2rem; }}
  .field input, .field select {{ width: 100%; padding: 0.5rem; border: 1px solid #ccc;
                                 border-radius: 6px; font-size: 0.9rem; }}
  .field input:focus {{ outline: none; border-color: #1f4e79; }}
  button.predict {{ background: #1f4e79; color: white; border: none; padding: 0.8rem 2.5rem;
                    border-radius: 8px; font-size: 1rem; cursor: pointer;
                    font-weight: 600; margin-top: 1rem; }}
  button.predict:hover {{ background: #163d5e; }}
  button.predict:disabled {{ background: #999; cursor: not-allowed; }}
  .result {{ background: white; padding: 1.5rem; border-radius: 12px;
             box-shadow: 0 1px 6px rgba(0,0,0,0.06); margin-top: 1.5rem; display: none; }}
  .result.survived {{ border-left: 5px solid #2ecc71; }}
  .result.died {{ border-left: 5px solid #e74c3c; }}
  .result h2 {{ font-size: 1.3rem; }}
  .result .proba {{ font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }}
  .result .proba.low {{ color: #2ecc71; }}
  .result .proba.high {{ color: #e74c3c; }}
  .result .details {{ font-size: 0.85rem; color: #666; margin-top: 0.5rem; }}
  .note {{ font-size: 0.8rem; color: #888; margin-top: 0.5rem; font-style: italic; }}
  .api-info {{ background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 2rem;
               font-size: 0.82rem; color: #555; }}
  .api-info code {{ background: #e9ecef; padding: 0.15rem 0.4rem; border-radius: 3px;
                    font-size: 0.8rem; }}
</style>
</head>
<body>
<div class="container">
  <h1>Trauma Mortality Prediction</h1>
  <p class="subtitle">XGBoost models trained on NTDB 2019-2022, validated on 2024 holdout.
     Select a model, enter patient data, and get a mortality probability.</p>

  <div class="model-selector" id="modelSelector"></div>

  <div class="form-section">
    <h2>Patient Data</h2>
    <p class="note">Leave fields blank if unknown — XGBoost handles missing values natively.
       More fields = better prediction.</p>
    <div class="fields-grid" id="fieldsGrid"></div>
    <button class="predict" id="predictBtn" onclick="doPrediction()" disabled>
      Predict Mortality
    </button>
  </div>

  <div class="result" id="result">
    <h2 id="resultTitle"></h2>
    <div class="proba" id="resultProba"></div>
    <div class="details" id="resultDetails"></div>
  </div>

  <div class="api-info">
    <strong>API endpoints:</strong>
    <code>GET /models</code> — list models |
    <code>GET /models/{{id}}/fields</code> — expected fields |
    <code>POST /predict/{{id}}</code> — single prediction |
    <code>POST /predict/{{id}}/batch</code> — batch (up to 10k rows) |
    <code>GET /docs</code> — full Swagger UI
  </div>
</div>

<script>
const MODELS = {models_js};
let selectedModel = null;

// Common fields to show first (most impactful for mortality prediction)
const PRIORITY_FIELDS = [
  'AGEYEARS', 'SEX', 'TRAUMATYPE', 'GCSTOTAL', 'SBPFIRST', 'RRFIRST',
  'MECHANISM', 'INTENT',
  'EDSBP', 'EDGCSTOTAL', 'PULSERATE', 'EDPULSERATE', 'PULSEOXIMETRY',
  'EDOXYGENSATURATION', 'EDRESPIRATORYRATE', 'TEMPERATURE', 'EDTEMPERATURE',
  'ISS', 'NISS', 'GCSHIGHEST', 'GCSLOWEST', 'SBPHIGHEST', 'SBPLOWEST',
  'RRHIGHEST', 'RRLOWEST',
  'BARELL_TBI', 'BARELL_THORAX', 'BARELL_ABDOMEN_PELVIS',
  'INJ_SUBDURAL_HEMORRHAGE', 'INJ_PNEUMOTHORAX', 'INJ_SPLENIC_LACERATION',
  'INJ_LIVER_LACERATION', 'INJ_RIB_FRACTURE_MULTIPLE',
  'INJ_FEMUR_FRACTURE', 'INJ_PELVIC_FRACTURE', 'INJ_CONCUSSION',
  'CHF', 'HYPERTENSION', 'DIABETESMELLITUS', 'COPD', 'CIRRHOSIS', 'ESRD',
  'DEMENTIA', 'BLEEDINGDISORDER', 'DISSEMINATEDCANCER',
];

// Placeholder hints: realistic ranges, categorical options, or "0 or 1"
const FIELD_HINTS = {{
  // Demographics
  AGEYEARS: "0-110 (e.g. 45)",
  SEX: "Male or Female",
  RACE: "White, Black or African American, Asian, Other Race",
  ETHNICITY: "Not Hispanic or Latino, Hispanic or Latino",
  PRIMARYINSURANCE: "Private, Medicare, Medicaid, Self-Pay",
  ADVANCEDDIRECTIVELIMITINGCARE: "0=No, 1=Yes (DNR)",
  FUNCTIONALLYDEPENDENTHEALTHSTATUS: "0=No, 1=Yes",
  // Injury type
  TRAUMATYPE: "Blunt, Penetrating, Burn",
  MECHANISM: "Fall, MVT - Occupant, Firearm, Cut/Pierce, Struck by/against",
  INTENT: "Unintentional, Self-inflicted, Assault",
  // On-scene vitals
  GCSTOTAL: "3-15 (15=alert, 3=deep coma)",
  SBPFIRST: "mmHg (normal 110-140, shock <90)",
  RRFIRST: "breaths/min (normal 12-20)",
  // ED vitals
  EDSBP: "mmHg (normal 110-140)",
  EDGCSTOTAL: "3-15 (GCS at ED arrival)",
  PULSERATE: "bpm (normal 60-100, tachy >120)",
  EDPULSERATE: "bpm (normal 60-100)",
  PULSEOXIMETRY: "% (normal 95-100, hypoxia <90)",
  EDOXYGENSATURATION: "% (normal 95-100)",
  EDRESPIRATORYRATE: "breaths/min (normal 12-20)",
  TEMPERATURE: "°C (normal 36-37.5, hypothermia <35)",
  EDTEMPERATURE: "°C (normal 36-37.5)",
  // In-hospital
  ISS: "1-75 (minor 1-8, major >15, critical >25)",
  NISS: "1-75 (similar to ISS)",
  GCSHIGHEST: "3-15 (best GCS during stay)",
  GCSLOWEST: "3-15 (worst GCS during stay)",
  SBPHIGHEST: "mmHg (highest SBP during stay)",
  SBPLOWEST: "mmHg (lowest SBP, <70=severe shock)",
  RRHIGHEST: "breaths/min (highest RR during stay)",
  RRLOWEST: "breaths/min (lowest RR)",
  // Comorbidities
  HYPERTENSION: "0=No, 1=Yes",
  DIABETESMELLITUS: "0=No, 1=Yes",
  CHF: "0=No, 1=Yes (heart failure)",
  COPD: "0=No, 1=Yes",
  MI: "0=No, 1=Yes (prior heart attack)",
  PERIPHERALVASCULARDISEASE: "0=No, 1=Yes",
  DEMENTIA: "0=No, 1=Yes",
  CIRRHOSIS: "0=No, 1=Yes (liver cirrhosis)",
  ESRD: "0=No, 1=Yes (dialysis)",
  BLEEDINGDISORDER: "0=No, 1=Yes (anticoag etc.)",
  DISSEMINATEDCANCER: "0=No, 1=Yes (metastatic)",
  ALCOHOLUSEDISORDER: "0=No, 1=Yes",
  SUBSTANCEABUSEDISORDERDRUG: "0=No, 1=Yes",
  MENTALPERSONALITYDISORDER: "0=No, 1=Yes",
  ATTENTIONDEFICITDISORDER: "0=No, 1=Yes",
  SMOKINGSTATUS: "0=No, 1=Yes",
  // Barell body regions
  BARELL_TBI: "0=No, 1=Yes (brain injury)",
  BARELL_OTHER_HEAD: "0=No, 1=Yes",
  BARELL_FACE: "0=No, 1=Yes",
  BARELL_NECK: "0=No, 1=Yes",
  BARELL_THORAX: "0=No, 1=Yes (chest)",
  BARELL_ABDOMEN_PELVIS: "0=No, 1=Yes",
  BARELL_VERTEBRAL_NO_SCI: "0=No, 1=Yes (spine, no cord)",
  BARELL_SCI: "0=No, 1=Yes (spinal cord injury)",
  BARELL_UPPER_EXTREMITY: "0=No, 1=Yes (arm/hand)",
  BARELL_LOWER_EXTREMITY: "0=No, 1=Yes (leg/foot)",
  BARELL_BURNS: "0=No, 1=Yes",
  BARELL_SYSTEM_OR_OTHER: "0=No, 1=Yes",
  // Specific injuries
  INJ_SUBDURAL_HEMORRHAGE: "0=No, 1=Yes (brain bleed)",
  INJ_CONCUSSION: "0=No, 1=Yes",
  INJ_PNEUMOTHORAX: "0=No, 1=Yes (collapsed lung)",
  INJ_SPLENIC_LACERATION: "0=No, 1=Yes (spleen tear)",
  INJ_LIVER_LACERATION: "0=No, 1=Yes (liver tear)",
  INJ_RIB_FRACTURE_MULTIPLE: "0=No, 1=Yes",
  INJ_FEMUR_FRACTURE: "0=No, 1=Yes (thigh bone)",
  INJ_PELVIC_FRACTURE: "0=No, 1=Yes",
  INJ_DISTAL_RADIUS_FRACTURE: "0=No, 1=Yes (wrist)",
  INJ_FOOT_FRACTURE: "0=No, 1=Yes",
}};

function renderModelCards() {{
  const container = document.getElementById('modelSelector');
  container.innerHTML = MODELS.map(m => `
    <div class="model-card" id="card-${{m.model_id}}" onclick="selectModel('${{m.model_id}}')">
      <h3>${{m.model_id}}</h3>
      <p><strong>${{m.phase_cutoff}}</strong></p>
      <p>${{m.n_predictors}} predictors</p>
    </div>
  `).join('');
}}

function selectModel(mid) {{
  selectedModel = MODELS.find(m => m.model_id === mid);
  document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
  document.getElementById('card-' + mid).classList.add('selected');
  document.getElementById('predictBtn').disabled = false;
  renderFields();
}}

function renderFields() {{
  const grid = document.getElementById('fieldsGrid');
  if (!selectedModel) return;
  // Merge numeric + categorical, priority fields first
  const allFields = [...selectedModel.numeric_fields, ...selectedModel.categorical_fields];
  const priority = PRIORITY_FIELDS.filter(f => allFields.includes(f));
  const rest = allFields.filter(f => !priority.includes(f));
  const ordered = [...priority, ...rest];

  grid.innerHTML = ordered.map(f => {{
    const isCat = selectedModel.categorical_fields.includes(f);
    const hint = FIELD_HINTS[f] || (isCat ? "text value" : "number");
    if (isCat) {{
      return `<div class="field">
        <label>${{f}}</label>
        <input type="text" id="f-${{f}}" placeholder="${{hint}}">
      </div>`;
    }}
    return `<div class="field">
      <label>${{f}}</label>
      <input type="number" step="any" id="f-${{f}}" placeholder="${{hint}}">
    </div>`;
  }}).join('');
}}

async function doPrediction() {{
  if (!selectedModel) return;
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  btn.textContent = 'Predicting...';

  const features = {{}};
  const allFields = [...selectedModel.numeric_fields, ...selectedModel.categorical_fields];
  for (const f of allFields) {{
    const el = document.getElementById('f-' + f);
    if (!el || el.value === '') continue;
    const isCat = selectedModel.categorical_fields.includes(f);
    features[f] = isCat ? el.value : parseFloat(el.value);
  }}

  try {{
    const resp = await fetch('/predict/' + selectedModel.model_id, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ features }})
    }});
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Prediction failed');

    const resultDiv = document.getElementById('result');
    const mortalityProba = data.probability ? data.probability[0][1] : null;
    const died = data.prediction[0] === 1;

    resultDiv.style.display = 'block';
    resultDiv.className = 'result ' + (died ? 'died' : 'survived');
    document.getElementById('resultTitle').textContent =
      died ? 'Predicted: Mortality' : 'Predicted: Survival';
    if (mortalityProba !== null) {{
      const pct = (mortalityProba * 100).toFixed(1);
      document.getElementById('resultProba').textContent = pct + '% mortality risk';
      document.getElementById('resultProba').className =
        'proba ' + (mortalityProba > 0.5 ? 'high' : 'low');
    }}
    document.getElementById('resultDetails').textContent =
      `Model: ${{data.model_id}} | Phase: ${{data.phase_cutoff}} | `
      + `Fields provided: ${{Object.keys(features).length}}/${{allFields.length}}`;
  }} catch (err) {{
    alert('Error: ' + err.message);
  }} finally {{
    btn.disabled = false;
    btn.textContent = 'Predict Mortality';
  }}
}}

renderModelCards();
if (MODELS.length > 0) selectModel(MODELS[0].model_id);
</script>
</body>
</html>"""
