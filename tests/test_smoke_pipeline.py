"""Smoke test — the full pipeline on a synthetic NTDB-shaped dataframe.

This test doesn't require real PUF CSVs.  It builds a realistic dataframe with
the NTDB column names, runs inclusion → target → split → impute → fit → eval,
and asserts the model was saved with a non-empty config.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trauma_ml.catalogue import Catalogue
from trauma_ml.persistence import ModelArtifact
from trauma_ml.targets import TargetSpec
from trauma_ml.trainer import Trainer, TrainerConfig


XLSX = Path(__file__).resolve().parents[1] / "NTDB_variable_mapping_for_trauma_ML.xlsx"
pytestmark = pytest.mark.skipif(not XLSX.exists(),
                                reason="Catalogue xlsx not available in this environment")


def _make_synthetic(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Fabricate an NTDB-like dataframe large enough to train a small model."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "INC_KEY":        np.arange(n),
        "__admission_year": rng.choice([2019, 2020, 2021, 2022, 2024], size=n),
        "AGEYEARS":       rng.integers(1, 95, size=n),
        "SEX":            rng.choice([1, 2], size=n),
        "ETHNICITY":      rng.choice([1, 2, 99], size=n),
        "WHITE":          rng.choice([0, 1], size=n, p=[0.4, 0.6]),
        "BLACK":          rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "TOTALGCS":       rng.integers(3, 16, size=n),
        "SBP":            rng.normal(130, 25, size=n).clip(40, 250),
        "RESPIRATORYRATE": rng.integers(0, 40, size=n),
        "PULSERATE":      rng.integers(40, 160, size=n),
        "ISS":            rng.integers(0, 75, size=n),
        "NISS":           rng.integers(0, 75, size=n),
        "PRIMARYECODEICD10": rng.choice(["V00.0XXA", "W19.XXXA", "X95.0XXA", "Y04.0XXA"], size=n),
        "TRAUMATYPE":     rng.choice([1, 2, 3], size=n, p=[0.88, 0.1, 0.02]),
        "MECHANISM":      rng.choice([1, 2, 3, 4, 5], size=n),
        "INTENT":         rng.choice([1, 2, 3], size=n),
        "INTERFACILITYTRANSFER": rng.choice([0, 1, np.nan], size=n, p=[0.7, 0.2, 0.1]),
        "TOTALICULOS":    rng.integers(0, 20, size=n),
        # Target variable - risk increases with age, low GCS, low SBP
    })
    logit = (
        -3.0
        + 0.035 * (df["AGEYEARS"] - 50)
        - 0.25 * (df["TOTALGCS"] - 8)
        - 0.03 * (df["SBP"] - 120)
        + 0.04 * (df["ISS"] - 9)
    )
    proba = 1.0 / (1.0 + np.exp(-logit.values))
    died = rng.binomial(1, proba)
    df["HOSPDISCHARGEDISPOSITION"] = np.where(died == 1, 5, 1).astype(float)
    df["EDDISCHARGEDISPOSITION"] = np.where(rng.random(n) < 0.05, 5, 1).astype(float)

    # Inject random NaN into a handful of variables to exercise the imputer
    for col, frac in [("RESPIRATORYRATE", 0.2), ("PULSERATE", 0.15),
                       ("TOTALGCS", 0.1), ("SBP", 0.1)]:
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    return df


def test_full_pipeline_smoke(tmp_path):
    """Build synthetic dataset, run trainer, assert artifact saved and predictions work."""
    df = _make_synthetic(n=2000)
    dataset_path = tmp_path / "unified.parquet"
    df.to_parquet(dataset_path, index=False)

    catalogue = Catalogue(XLSX)

    cfg = TrainerConfig(
        model_id="smoke_test_001",
        dataset_path=dataset_path,
        catalogue=catalogue,
        target=TargetSpec(
            name="in_hospital_mortality",
            kind="binary",
            spec={"positive_code": 5, "include_ed_death": True},
        ),
        predictor_type="Tran NTDB",
        phase_cutoff="On-scene + ED arrival",
        inclusion_strategy="Tran NTDB",
        imputer_method="median_mode",
        model_family="random_forest",
        missingness_threshold=0.50,
        n_jobs=1,
    )

    artifact = Trainer(cfg).run(tmp_path / "outputs")

    # Artifact is saved as a directory of files
    saved_dir = tmp_path / "outputs" / "models" / "smoke_test_001"
    assert (saved_dir / "artifact.pkl").exists()
    assert (saved_dir / "config.json").exists()

    # Metrics were written
    assert (tmp_path / "outputs" / "metrics" / "smoke_test_001" / "overall__test.json").exists()

    # Round-trip: load and predict
    reloaded = ModelArtifact.load(saved_dir)
    n_check = 20
    sample = df.head(n_check).copy()
    preds = reloaded.predict(sample, calibrated="isotonic")
    assert len(preds) == n_check
    assert "predicted_class" in preds.columns
    assert "predicted_proba" in preds.columns
