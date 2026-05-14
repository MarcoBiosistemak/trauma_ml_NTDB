"""`trauma-predict` CLI — apply a saved ModelArtifact to a CSV.

Example
-------
    trauma-predict --model-id model_0042 --input new_patients.csv \\
                    --output predictions.csv --calibrated isotonic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..persistence import ModelArtifact
from ._common import load_yaml, resolve_repo_root, setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict with a saved trauma_ml model.")
    parser.add_argument("--model-id", required=True,
                         help="Model identifier (folder name under outputs/models)")
    parser.add_argument("--input", required=True, help="Input CSV with the required predictors")
    parser.add_argument("--output", required=True, help="Output CSV with predictions")
    parser.add_argument("--models-root", default=None,
                         help="Override models root (default: config.models_dir)")
    parser.add_argument("--calibrated", default="isotonic",
                         choices=["platt", "isotonic", "none"],
                         help="Which calibration to apply (default: isotonic)")
    parser.add_argument("--paths-config", default="config/paths.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    repo_root = resolve_repo_root()
    paths_cfg = load_yaml(repo_root / args.paths_config)
    models_root = Path(args.models_root) if args.models_root else (
        repo_root / paths_cfg["models_dir"]
    )
    artifact_dir = models_root / args.model_id
    if not artifact_dir.exists():
        raise FileNotFoundError(f"No such model directory: {artifact_dir}")

    artifact = ModelArtifact.load(artifact_dir)
    df = pd.read_csv(args.input, low_memory=False)
    calibrated = None if args.calibrated == "none" else args.calibrated
    preds = artifact.predict(df, calibrated=calibrated)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
