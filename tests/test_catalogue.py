"""Catalogue sanity tests — exercise queries against the real xlsx."""
from pathlib import Path

import pytest

from trauma_ml.catalogue import Catalogue, PHASE_ORDER, REGISTRY_COLUMNS

XLSX = Path(__file__).resolve().parents[1] / "NTDB_variable_mapping_for_trauma_ML.xlsx"
pytestmark = pytest.mark.skipif(not XLSX.exists(),
                                reason="Catalogue xlsx not available in this environment")


def test_loads_expected_variable_count():
    cat = Catalogue(XLSX)
    assert len(cat) >= 200, f"Expected ≥200 variables, got {len(cat)}"


def test_registry_presets_are_non_empty():
    cat = Catalogue(XLSX)
    for registry in REGISTRY_COLUMNS:
        names = cat.variables_by_registry(registry)
        assert len(names) > 0, f"Registry {registry!r} yielded no variables"


def test_phase_filter_is_monotonic():
    """A wider phase_cutoff must include a superset of a narrower one."""
    cat = Catalogue(XLSX)
    onscene = set(cat.variables_for(year=2024, phase_cutoff="On-scene"))
    ed      = set(cat.variables_for(year=2024, phase_cutoff="At ED arrival"))
    inhosp  = set(cat.variables_for(year=2024, phase_cutoff="In-hospital (a posteriori)"))
    assert onscene.issubset(ed)
    assert ed.issubset(inhosp)


def test_ageyears_is_cross_registry():
    cat = Catalogue(XLSX)
    entry = cat.get("AGEYEARS")
    assert entry is not None
    # Confirm it's present in all 5 years
    for yr in (2019, 2020, 2021, 2022, 2024):
        assert entry.is_available_in(yr), f"AGEYEARS missing in AY {yr}"
    # And in all 5 registries
    for registry in REGISTRY_COLUMNS:
        assert entry.used_in(registry), f"AGEYEARS not flagged for {registry}"
