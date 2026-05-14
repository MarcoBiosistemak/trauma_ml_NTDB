"""Model persistence — save/load the full pipeline state.

Each trained model is saved as a pair of files in ``outputs/models/<model_id>/``:

  * ``artifact.pkl``  — a pickle of the ModelArtifact dataclass
  * ``config.json``   — human-readable summary of the run
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    """Everything needed to reproduce a prediction pipeline from raw input."""
    model_id: str
    family: str
    task: str                                          # 'binary' | 'multiclass' | 'survival'
    target_spec: dict                                  # from TargetSpec
    predictor_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    transformers: dict                                 # {'encoders': {...}, 'scalers': {...}}
    imputer: Any
    model: Any
    calibrated_models: dict                            # {'platt': ..., 'isotonic': ...}
    target_inverse: dict[int, Any]                     # 0 -> class label
    config: dict                                       # free-form run config
    extras: dict = field(default_factory=dict)         # registry preset, phase, inclusion, ...

    # -------------------------------------------------------------- #
    def save(self, output_root: Path) -> Path:
        output_dir = Path(output_root) / self.model_id
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "artifact.pkl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        config_serialisable = {
            "model_id": self.model_id,
            "family": self.family,
            "actual_model_type": self.extras.get("actual_model_type", self.family),
            "task": self.task,
            "target_spec": self.target_spec,
            "predictor_cols": self.predictor_cols,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "target_inverse": {str(k): str(v) for k, v in self.target_inverse.items()},
            "config": self.config,
            "extras": self.extras,
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_serialisable, f, indent=2, default=str)

        log.info("Saved model %s to %s", self.model_id, output_dir)
        return output_dir

    # -------------------------------------------------------------- #
    @classmethod
    def load(cls, path: Path | str) -> "ModelArtifact":
        path = Path(path)
        if path.is_dir():
            path = path / "artifact.pkl"
        with open(path, "rb") as f:
            artifact: ModelArtifact = pickle.load(f)
        log.info("Loaded model %s from %s", artifact.model_id, path)
        return artifact

    # -------------------------------------------------------------- #
    def predict(
        self,
        new_data: pd.DataFrame,
        calibrated: str | None = "isotonic",
        return_proba: bool = True,
    ) -> pd.DataFrame:
        """Apply the full pipeline (transformers → imputer → model → calibration)
        to fresh data and return predictions.
        """
        df = new_data.copy()

        # 1) Encoders — label-encode categoricals with the training classes.
        for col, le in self.transformers.get("encoders", {}).items():
            if col in df.columns:
                non_null = df[col].notna()
                if non_null.any():
                    vals = df.loc[non_null, col].astype(str).values
                    encoded = np.array([
                        le.transform([v])[0] if v in le.classes_ else -1
                        for v in vals
                    ])
                    new_col = np.full(len(df), np.nan)
                    new_col[non_null.to_numpy()] = encoded
                    df[col] = new_col

        # 2) Scalers
        for col, scaler in self.transformers.get("scalers", {}).items():
            if col in df.columns:
                non_null = df[col].notna()
                if non_null.any():
                    # Promote to float first so scaled values fit
                    if not pd.api.types.is_float_dtype(df[col]):
                        df[col] = df[col].astype(float)
                    vals = np.asarray(df.loc[non_null, col].to_numpy(),
                                       dtype=float).reshape(-1, 1)
                    df.loc[non_null, col] = scaler.transform(vals).flatten()

        # 3) Imputer — only over predictor columns
        missing = [c for c in self.predictor_cols if c not in df.columns]
        if missing:
            raise KeyError(f"New data missing predictors: {missing}")
        X = df[self.predictor_cols]
        if self.imputer is not None:
            X = self.imputer.transform(X)
        else:
            X = X.copy()

        # 4) Model / calibration
        model_to_use = self.model
        if calibrated and self.calibrated_models.get(calibrated) is not None:
            model_to_use = self.calibrated_models[calibrated]

        y_proba = None
        if return_proba and hasattr(model_to_use, "predict_proba"):
            try:
                y_proba = model_to_use.predict_proba(X)
            except Exception:
                y_proba = None
        y_pred = model_to_use.predict(X)

        labels = [self.target_inverse.get(int(p), p) for p in y_pred]
        out = df.copy()
        out["predicted_class"] = labels
        if y_proba is not None:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                out["predicted_proba"] = y_proba[:, 1]
            else:
                for k in range(y_proba.shape[1]):
                    out[f"proba__class_{k}"] = y_proba[:, k]
        return out
