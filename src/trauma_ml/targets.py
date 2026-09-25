"""Target definition — build the y vector from the unified dataframe.

Supported target kinds
----------------------
binary            : single positive code on an outcome variable
                    (e.g. in-hospital mortality = HOSPDISCHARGEDISPOSITION==5).
ordinal_bands     : ISS / NISS discretised into ordered categorical bands.
survival          : (time, event) tuple for survival models (stub).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TargetSpec:
    """Compact description of a prediction target.  Saved with each model."""
    name: str
    kind: str                        # 'binary' | 'ordinal_bands' | 'survival'
    spec: dict[str, Any]             # kind-specific spec (see YAML config)


def _binary_mortality(df: pd.DataFrame, spec: dict) -> pd.Series:
    pos = spec.get("positive_code", 5)
    include_ed = spec.get("include_ed_death", True)

    hosp = df.get("HOSPDISCHARGEDISPOSITION")
    if hosp is None:
        raise KeyError("HOSPDISCHARGEDISPOSITION missing — cannot build mortality target")

    y = (hosp == pos).astype("Int64")
    if include_ed and "EDDISCHARGEDISPOSITION" in df.columns:
        y = y | (df["EDDISCHARGEDISPOSITION"] == pos).astype("Int64")

    # Propagate NaN from HOSPDISCHARGEDISPOSITION (needed for the Tran-style exclusion)
    y = y.where(df["HOSPDISCHARGEDISPOSITION"].notna(), other=pd.NA)
    return y.rename("target")


def _ordinal_bands(df: pd.DataFrame, spec: dict, target_name: str = "") -> pd.Series:
    """Turn a numeric column (ISS/NISS or similar) into band labels.

    spec.bands is a list of (lower_inclusive, upper_exclusive, label).

    Round 32: defensive alias resolution.  ``AISSEVERITY`` is the per-injury
    column on the AIS DIAGNOSIS table — it never exists on the wide patient
    frame.  If a YAML config references it, we transparently route to
    ``NISS`` (for *NISS_band*-named targets) or ``ISS`` (otherwise) since
    that's almost certainly what the user intended, and emit a warning so
    the YAML can be corrected.
    """
    source_col = spec.get("variable") or spec.get("derive_from")

    # Round 32: alias resolution
    if source_col and source_col not in df.columns:
        original = source_col
        upper = source_col.upper()
        if upper == "AISSEVERITY" or upper.startswith("AIS"):
            tn_upper = target_name.upper()
            if "NISS" in tn_upper:
                source_col = "NISS"
            else:
                source_col = "ISS"
            import logging
            logging.getLogger(__name__).warning(
                "Target %r references %r which is a per-injury AIS column, "
                "not a patient-level column.  Auto-routing to %r based on "
                "target name.  To silence this warning, update your YAML "
                "config: spec.variable: %r",
                target_name, original, source_col, source_col,
            )

    if source_col not in df.columns:
        # Tighter error message — include the target name and what's available
        candidate_cols = [c for c in df.columns
                          if c.upper() in ("ISS", "NISS", "ISS_05")]
        raise KeyError(
            f"Target {target_name!r}: source column {source_col!r} not in "
            f"dataframe — cannot build banded target.  Available severity "
            f"columns: {candidate_cols}.  Update spec.variable in your "
            f"experiment_grids.yaml."
        )

    raw = df[source_col].astype(float)
    bands = spec["bands"]
    labels = [b[2] for b in bands]
    # Build a monotonic edges array from (lower, upper) pairs
    edges = [bands[0][0]] + [b[1] for b in bands]
    y = pd.cut(raw, bins=edges, labels=labels, right=False, include_lowest=True)
    # Keep NaN through; cast to object so downstream can convert cleanly
    return y.astype("object").where(raw.notna(), other=pd.NA).rename("target")


def _binary_threshold(df: pd.DataFrame, spec: dict, target_name: str = "") -> pd.Series:
    """Binarise a numeric column at a threshold.

    Used for the ISS_band / NISS_band "severe vs not" problem: the classic
    major-trauma cut is ISS >= 16, i.e. ISS > 15.  Spec keys:

        variable   : source column ("ISS" / "NISS").  ``derive_from`` is also
                     accepted as an alias.
        threshold  : numeric cut (default 15).
        above_is_positive : if True (default) y=1 when value > threshold
                     (strictly greater).  Set ``inclusive=True`` to use >=.

    The same AIS-alias auto-routing as ``_ordinal_bands`` applies so a YAML
    that points at ``AISSEVERITY`` is transparently routed to ISS/NISS.
    Rows with a NaN source value get a NaN target (dropped downstream).
    """
    source_col = spec.get("variable") or spec.get("derive_from")

    if source_col and source_col not in df.columns:
        original = source_col
        upper = source_col.upper()
        if upper == "AISSEVERITY" or upper.startswith("AIS"):
            source_col = "NISS" if "NISS" in target_name.upper() else "ISS"
            import logging
            logging.getLogger(__name__).warning(
                "Target %r references %r (a per-injury AIS column); auto-routing "
                "to %r.  Update spec.variable in experiment_grids.yaml to silence.",
                target_name, original, source_col,
            )

    if source_col not in df.columns:
        candidate_cols = [c for c in df.columns if c.upper() in ("ISS", "NISS", "ISS_05")]
        raise KeyError(
            f"Target {target_name!r}: source column {source_col!r} not in dataframe "
            f"— cannot build binary-threshold target.  Available severity columns: "
            f"{candidate_cols}.  Update spec.variable in experiment_grids.yaml."
        )

    threshold = spec.get("threshold", 15)
    inclusive = spec.get("inclusive", False)
    raw = pd.to_numeric(df[source_col], errors="coerce")
    if inclusive:
        y = (raw >= threshold)
    else:
        y = (raw > threshold)
    y = y.astype("Int64").where(raw.notna(), other=pd.NA)
    return y.rename("target")


def build_target(df: pd.DataFrame, spec: TargetSpec) -> pd.Series:
    """Return a y Series aligned with df.index, named 'target'."""
    if spec.kind == "binary":
        return _binary_mortality(df, spec.spec)
    if spec.kind == "ordinal_bands":
        return _ordinal_bands(df, spec.spec, target_name=spec.name)
    if spec.kind == "binary_threshold":
        return _binary_threshold(df, spec.spec, target_name=spec.name)
    if spec.kind == "survival":
        raise NotImplementedError("Survival target builder: see models.cox_ph scaffold")
    raise ValueError(f"Unknown target kind: {spec.kind!r}")


def encode_classes(y: pd.Series) -> tuple[np.ndarray, dict[int, Any]]:
    """Encode categorical/bool y into 0..K-1 integers; return (y_int, inverse_map)."""
    classes = pd.Series(y.dropna().unique()).sort_values().tolist()
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    y_int = y.map(mapping)
    return y_int.to_numpy(), {v: k for k, v in mapping.items()}
