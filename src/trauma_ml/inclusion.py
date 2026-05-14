"""Cohort inclusion / exclusion strategies.

Each class in this module implements ``apply(df) -> (df_filtered, log_dict)``.

* ``TranNTDB``    — exact reproduction of Tran 2022's inclusion recipe
                    (sheet 7 of the mapping xlsx).  Every filter is applied
                    in the order that yields Tran's 1.38M cohort on 2015-2017
                    data; applied to 2019-2024 it should shrink by roughly the
                    same fraction.
* ``ECTrauma``    — best-effort replica of the ECTrauma exclusion chain
                    (adults + ISS ≥ 9 severe-trauma gate).
* ``KarolinskaSweTrau`` — NISS > 15 severe-trauma cohort, adults, blunt-dominant.
* ``RETRAUCI``    — ICU-admission proxy (TOTALICULOS > 0 OR
                    EDDISCHARGEDISPOSITION == 8).

All four return an ``exclusion_log`` dict recording how many patients were
dropped at each step, for the methods section of papers and for QA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class InclusionStrategy(Protocol):
    name: str
    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        ...


# ---------------------------------------------------------------------------
# Tran 2022 — exact recipe from sheet 7
# ---------------------------------------------------------------------------
@dataclass
class TranNTDB:
    """Tran 2022 inclusion: ICD-10 trauma, non-burn, non-transferred, with survival info.

    Note: the ICD-10 initial-encounter filter (7th-char == 'A') is applied at
    diagnosis-code level in the full-feature version (see models.tran_full);
    at the TRAUMA-table level we only apply the recipe that can be evaluated
    without the long PUF_ICDDIAGNOSIS side-table.
    """
    name: str = "Tran NTDB"

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        log: dict[str, int] = {"n_in": len(df)}
        out = df.copy()

        # (1) INCLUDE traumatic mechanism: PRIMARYECODEICD10 starts with V-Y
        if "PRIMARYECODEICD10" in out.columns:
            mask = out["PRIMARYECODEICD10"].astype(str).str[:1].isin(list("VWXY"))
            log["after_trauma_mechanism"] = int(mask.sum())
            out = out[mask]

        # (2) EXCLUDE burns (X00-X19) and environmental causes (W65-W99, X20-X50)
        if "PRIMARYECODEICD10" in out.columns:
            code = out["PRIMARYECODEICD10"].astype(str).str.upper()
            burn_mask   = code.str[:3].between("X00", "X19")
            env_mask1   = code.str[:3].between("W65", "W99")
            env_mask2   = code.str[:3].between("X20", "X50")
            drop = burn_mask | env_mask1 | env_mask2
            out = out[~drop]
            log["after_burn_env_exclusion"] = len(out)

        # (3) EXCLUDE interfacility transfers out
        if "INTERFACILITYTRANSFER" in out.columns:
            out = out[out["INTERFACILITYTRANSFER"].fillna(0) != 1]
            log["after_exclude_transfers_out"] = len(out)

        # (4) EXCLUDE rows with missing survival info
        if "HOSPDISCHARGEDISPOSITION" in out.columns:
            out = out[out["HOSPDISCHARGEDISPOSITION"].notna()]
            log["after_valid_survival"] = len(out)

        log["n_out"] = len(out)
        return out.reset_index(drop=True), log


# ---------------------------------------------------------------------------
# ECTrauma — best-effort from the ECTrauma paper
# ---------------------------------------------------------------------------
@dataclass
class ECTrauma:
    """Adult severe-trauma cohort (ISS ≥ 9).  Mirrors ECTrauma's main cohort."""
    name: str = "ECTrauma"

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        log: dict[str, int] = {"n_in": len(df)}
        out = df.copy()

        # (1) Adults only
        if "AGEYEARS" in out.columns:
            out = out[out["AGEYEARS"].fillna(-1) >= 18]
            log["after_adults_only"] = len(out)

        # (2) Severe trauma: ISS >= 9  (major trauma threshold)
        if "ISS" in out.columns:
            out = out[out["ISS"].fillna(-1) >= 9]
            log["after_severe_ISS_ge_9"] = len(out)

        # (3) Exclude burns (same as Tran)
        if "PRIMARYECODEICD10" in out.columns:
            code = out["PRIMARYECODEICD10"].astype(str).str.upper()
            burn_mask = code.str[:3].between("X00", "X19")
            out = out[~burn_mask]
            log["after_burn_exclusion"] = len(out)

        # (4) Valid survival info
        if "HOSPDISCHARGEDISPOSITION" in out.columns:
            out = out[out["HOSPDISCHARGEDISPOSITION"].notna()]
            log["after_valid_survival"] = len(out)

        log["n_out"] = len(out)
        return out.reset_index(drop=True), log


# ---------------------------------------------------------------------------
# Karolinska / SweTrau — Holtenius 2024
# ---------------------------------------------------------------------------
@dataclass
class KarolinskaSweTrau:
    """Adults, NISS > 15, valid survival.  Best-effort NTDB analogue of SweTrau."""
    name: str = "Karolinska SweTrau"

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        log: dict[str, int] = {"n_in": len(df)}
        out = df.copy()

        if "AGEYEARS" in out.columns:
            out = out[out["AGEYEARS"].fillna(-1) >= 18]
            log["after_adults_only"] = len(out)

        # Severe polytrauma: NISS > 15
        if "NISS" in out.columns:
            out = out[out["NISS"].fillna(-1) > 15]
            log["after_severe_NISS_gt_15"] = len(out)

        if "HOSPDISCHARGEDISPOSITION" in out.columns:
            out = out[out["HOSPDISCHARGEDISPOSITION"].notna()]
            log["after_valid_survival"] = len(out)

        log["n_out"] = len(out)
        return out.reset_index(drop=True), log


# ---------------------------------------------------------------------------
# RETRAUCI — ICU-admission proxy
# ---------------------------------------------------------------------------
@dataclass
class RETRAUCI:
    """RETRAUCI collects only ICU-admitted trauma patients.  Proxy in NTDB:
    TOTALICULOS > 0  OR  EDDISCHARGEDISPOSITION == 8 (direct ICU admit).
    """
    name: str = "RETRAUCI"

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        log: dict[str, int] = {"n_in": len(df)}
        out = df.copy()

        has_icu_los = out.get("TOTALICULOS", pd.Series(0, index=out.index)).fillna(0) > 0
        direct_icu  = out.get("EDDISCHARGEDISPOSITION", pd.Series(0, index=out.index)).fillna(0) == 8
        out = out[has_icu_los | direct_icu]
        log["after_icu_admission"] = len(out)

        if "AGEYEARS" in out.columns:
            out = out[out["AGEYEARS"].fillna(-1) >= 15]      # RETRAUCI minimum age
            log["after_age_ge_15"] = len(out)

        if "HOSPDISCHARGEDISPOSITION" in out.columns:
            out = out[out["HOSPDISCHARGEDISPOSITION"].notna()]
            log["after_valid_survival"] = len(out)

        log["n_out"] = len(out)
        return out.reset_index(drop=True), log


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
STRATEGIES: dict[str, InclusionStrategy] = {
    "Tran NTDB":          TranNTDB(),
    "ECTrauma":           ECTrauma(),
    "Karolinska SweTrau": KarolinskaSweTrau(),
    "RETRAUCI":           RETRAUCI(),
}


def get_strategy(name: str) -> InclusionStrategy:
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown inclusion strategy {name!r}. "
            f"Available: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[name]
