"""Variable catalogue — read sheet 6 of the NTDB mapping xlsx and expose queries.

The catalogue is the single source of truth for:
    * which NTDB variables exist in which admission year
    * each variable's data-timing phase (on-scene / at ED arrival / in-hospital / admin)
    * each variable's category (Demographics, Mechanism, Treatment, …)
    * which registry studies use each variable (Tran 2022, Karolinska SweTrau,
      RETRAUCI, ECTrauma, EIPD)

Nothing else in the package should hard-code variable names — they all flow
through this catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


# Phase order (earlier → later in the patient journey).
# Any variable whose phase index exceeds the user-selected cutoff is excluded
# from the predictor pool (unless a registry flag re-includes it).
PHASE_ORDER = [
    "On-scene",
    "At ED arrival",
    "In-hospital (a posteriori)",
    "Demographics / admin",   # age/sex/race — always knowable, treated separately
]


# ---------------------------------------------------------------------------
# Round 18: hard blacklist of variables that must NEVER be predictors,
# regardless of how the catalogue xlsx tags them.
# ---------------------------------------------------------------------------
# Two kinds of leakage we explicitly block:
#
# 1. Row identifiers — INC_KEY is the unique patient record ID assigned by
#    the trauma registry.  It's needed during loading to JOIN auxiliary
#    tables (AIS diagnoses, ECODE lookups) but has zero clinical meaning.
#    If it ends up as a predictor, the model can learn date-encoded
#    patterns (since IDs are roughly date-ordered within centers).
#
# 2. Hospital-level attributes — TEACHINGSTATUS, HOSPITALTYPE, BEDSIZE,
#    STATEDESIGNATION, FACILITYID, etc. are properties of the trauma
#    center, NOT the patient.  They correlate strongly with mortality
#    because sicker patients go to academic level-1 centers, but the
#    correlation is mediated by case-mix, not causal at the patient level.
#    Tran 2022 (PLoS One) explicitly excluded hospital-level variables.
#    Including them lets the model learn "which hospital admits you"
#    rather than "what's wrong with you" — leakage with respect to the
#    research question.
#
# These are excluded EVEN IF the catalogue xlsx flags them as
# "Demographics / admin" (which the variables_for() phase filter normally
# always keeps).
NON_PREDICTOR_COLUMNS = {
    # Patient/incident identifiers (loader-internal join keys)
    "INC_KEY", "INCKEY", "INCIDENT_KEY", "INCIDENTKEY",

    # Hospital-level attributes (correlate with case-mix, not patient state)
    "TEACHINGSTATUS",
    "HOSPITALTYPE", "BEDSIZE",
    "STATEDESIGNATION", "ACSVERIFICATIONLEVEL", "VERIFICATIONLEVEL",
    "FACILITYID", "TRAUMAFACILITYID", "FACILITY_ID",
    "TRAUMACENTERLEVEL",

    # Raw ICD/E-code strings — high-cardinality identifiers used by the
    # loader to derive TRAUMATYPE / MECHANISM / INTENT via lookup, but
    # not features themselves once derived.
    "PRIMARYECODEICD10", "ICDDIAGNOSISCODE",
    "AISPREDOT",  # raw AIS code; ISS / NISS are the derived features

    # Discharge / outcome columns (would be target leakage)
    "HOSPDISCHARGEDISPOSITION", "EDDISCHARGEDISPOSITION", "DEATHINED",
    "DISCHARGE_DATE", "DISCHARGEDATE",
    "EDDISCHARGE", "HOSPDISCHARGE",

    # Inter-facility transfer flag (post-hoc admin)
    "INTERFACILITYTRANSFER",

    # Round 56: ED length-of-stay — LEAKAGE for mortality (an ED death's
    # "ED discharge" is the death event; ED LOS is fixed by the disposition).
    "EDDISCHARGEHRS",
}

# Case-insensitive lookup set — ALL comparisons against the blacklist must
# go through this set so that 'inc_key', 'Inc_Key', and 'INC_KEY' are all
# blocked equivalently.  The catalogue xlsx is inconsistent about casing
# (some columns are upper, some lower, some camel) and the loader's
# alias-resolver normalises to canonical names AFTER the filter runs.
NON_PREDICTOR_COLUMNS_LOWER = {c.lower() for c in NON_PREDICTOR_COLUMNS}

# Columns carried in the dataset but NOT run through the standard
# encoders/scalers/imputer/pre-filter — a model that knows how to consume them
# handles them itself.  ICD_DIAG_CODES is the raw per-patient ICD-10 code-list
# string, vectorised to a multi-hot inside the (faithful) Doshi ICD FFNN.
PASSTHROUGH_COLUMNS = {"ICD_DIAG_CODES"}


def is_passthrough(column_name: str) -> bool:
    return column_name in PASSTHROUGH_COLUMNS


def is_non_predictor(column_name: str) -> bool:
    """Case-insensitive blacklist check.  Use this everywhere instead of
    `c in NON_PREDICTOR_COLUMNS` to avoid casing mismatches."""
    return column_name.lower() in NON_PREDICTOR_COLUMNS_LOWER




# User-facing phase-cutoff labels (from the YAML config) mapped to the last
# phase to keep.  Each cutoff is inclusive (variables AT that phase are kept).
PHASE_CUTOFF_ALIASES = {
    "On-scene":                                 "On-scene",
    "On-scene + ED arrival":                    "At ED arrival",
    "On-scene + ED arrival + In-hospital":      "In-hospital (a posteriori)",
    # Pass-through for direct PHASE_ORDER values
    "At ED arrival":                            "At ED arrival",
    "In-hospital (a posteriori)":               "In-hospital (a posteriori)",
}

# Round 11 item 3: special "baseline-feature" phase cutoffs that pin the
# predictor set to the exact inputs of one clinical baseline score.  They
# are NOT phase-ordered — they bypass the phase filter and return only the
# named variable list (subject to other filters like registries / year).
# Useful when the user wants to train an ML model "with the same inputs
# TRISS uses" and compare directly.
BASELINE_FEATURE_CUTOFFS: dict[str, list[str]] = {
    # ISS uses ISS itself.  Demographics (age/sex) are always included via
    # the Demographics / admin pass-through, but listing them explicitly
    # makes intent clear.
    "iss_only": [
        "ISS",
        "AGEYEARS", "SEX",
    ],
    # NISS uses NISS itself.
    "niss_only": [
        "NISS",
        "AGEYEARS", "SEX",
    ],
    # TRISS combines anatomy (ISS), physiology (GCS, SBP, RR), age, and
    # mechanism (TRAUMATYPE -> blunt vs penetrating coefficients).
    "triss_inputs": [
        "ISS", "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "AGEYEARS", "SEX", "TRAUMATYPE",
    ],
    # All baseline inputs combined — useful for an ML model fit on the
    # full set of clinical baseline features (ISS + NISS + TRISS inputs).
    "all_baseline_inputs": [
        "ISS", "NISS",
        "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "AGEYEARS", "SEX", "TRAUMATYPE",
    ],
}


# ---------------------------------------------------------------------------
# Round 21: HARD-CODED phase-cutoff predictor lists.
# ---------------------------------------------------------------------------
# These are the SOLE source of truth for which variables get used at each
# phase cutoff.  We do NOT rely on the catalogue xlsx's "Phase" column or
# "Tran 2022" registry flag for filtering — the xlsx is too inconsistent
# across NTDB years (some years tag GCSTOTAL as "On-scene", others tag it
# as "At ED arrival", and the registry column has empty cells).
#
# Adding/removing a variable from a cutoff means editing the list below.
# This is intentional — explicit beats clever.
#
# Time-ordered cutoffs (each ADDS to the previous):
# ---------------------------------------------------------------------------
# Injury-description (Barell / Tran-specific) feature names.
# Populated by ntdb_loader.merge_barell_features at build time.  These are
# DISCHARGE-CODED (ICD-10-CM, finalised a posteriori), so for an OUTCOME
# target (mortality) they belong ONLY at the in-hospital cutoff.  For an
# INJURY-SEVERITY target (ISS_band / NISS_band) they may be used at every
# phase (the task is "predict the severity band from the injuries"), which
# the trainer enables via variables_for(..., include_injury_features=True).
INJURY_FEATURES: list[str] = [
    "BARELL_TBI", "BARELL_OTHER_HEAD", "BARELL_FACE", "BARELL_NECK",
    "BARELL_SCI", "BARELL_VERTEBRAL_NO_SCI",
    "BARELL_THORAX", "BARELL_ABDOMEN_PELVIS",
    "BARELL_UPPER_EXTREMITY", "BARELL_LOWER_EXTREMITY",
    "BARELL_BURNS", "BARELL_SYSTEM_OR_OTHER",
    "INJ_SUBDURAL_HEMORRHAGE", "INJ_CONCUSSION", "INJ_PNEUMOTHORAX",
    "INJ_RIB_FRACTURE_MULTIPLE", "INJ_SPLENIC_LACERATION",
    "INJ_LIVER_LACERATION", "INJ_PELVIC_FRACTURE",
    "INJ_FEMUR_FRACTURE", "INJ_DISTAL_RADIUS_FRACTURE", "INJ_FOOT_FRACTURE",
]


# ---------------------------------------------------------------------------
# Round 51: HARD-CODED phase-cutoff predictor lists (composed from blocks).
# ---------------------------------------------------------------------------
# SOLE source of truth for which variables are used at each phase cutoff.
# Built from shared sub-blocks so the three time-ordered cutoffs stay in
# sync and we cannot accidentally reference a column that does not exist.
#
# Temporal-correctness rules enforced here:
#   * Injury-description features (Barell/INJ_*) are NOT on-scene/ED-arrival
#     for outcome targets — they are discharge-coded, so they live only in
#     the in-hospital cutoff (re-enabled for band targets by the trainer).
#   * ISS / NISS (AIS-derived, a posteriori) live only in the in-hospital
#     cutoff.
#   * Only genuinely time-stamped physiology / events sit in on-scene/ED.
#
# Column names below are the REAL NTDB variables (or loader-produced
# canonicals): SBPFIRST<-SBP, RRFIRST<-RESPIRATORYRATE, GCSTOTAL<-TOTALGCS;
# PRIMARYMETHODPAYMENT (not "PRIMARYINSURANCE"); there is no native single
# "RACE" column (NTDB ships RACE_* category flags) so RACE is omitted; the
# "ED*"-prefixed and "*HIGHEST/*LOWEST" columns do NOT exist natively and
# have been removed (the base first-recorded vitals already represent the
# ED-arrival measurement).

# Demographics knowable at first contact (Tran 2022 S1 Table).
# PRIMARYMETHODPAYMENT (insurance/payer) is kept as an admission-time
# economic/administrative feature; it is also used as a sociodemographic
# SUBGROUP axis (fairness/equity reporting) for every target.
_DEMOGRAPHICS = ["AGEYEARS", "SEX", "ETHNICITY", "PRIMARYMETHODPAYMENT"]
# Anthropometry (numeric), known at admission -> on-scene/first-contact layer.
_ANTHRO = ["HEIGHT", "WEIGHT"]
# Mechanism / intent — describe WHAT happened (ECODE lookup).
_MECHANISM = ["TRAUMATYPE", "MECHANISM", "INTENT"]
# On-scene / first-contact physiology (first-recorded values).
_ONSCENE_PHYS = ["GCSTOTAL", "SBPFIRST", "RRFIRST"]
# Comorbidities — pre-existing, knowable from history (Tran 2022 S1 Table).
_COMORBID = [
    "SMOKINGSTATUS", "COPD", "CHF", "MI", "HYPERTENSION",
    "PERIPHERALVASCULARDISEASE", "ESRD", "CIRRHOSIS",
    "DIABETESMELLITUS", "BLEEDINGDISORDER", "DISSEMINATEDCANCER",
    "ALCOHOLUSEDISORDER", "MENTALPERSONALITYDISORDER",
    "SUBSTANCEABUSEDISORDERDRUG", "ATTENTIONDEFICITDISORDER",
    "DEMENTIA", "ADVANCEDDIRECTIVELIMITINGCARE",
    "FUNCTIONALLYDEPENDENTHEALTHSTATUS",
]
# On-scene events / transport (all NTDB years).
_L1_EXTRA = ["TRANSPORTMODE", "PREHOSPITALCARDIACARREST"]
# ED-arrival physiology (first-recorded temp / SpO2 / pulse).
_ED_VITALS = ["TEMPERATURE", "PULSEOXIMETRY", "PULSERATE"]
# ED-arrival extras (all NTDB years).  HOSPITALARRIVALHRS = time from
# incident to ED/hospital arrival (the prehospital "trip" interval, known
# at arrival); pupils + ED alcohol screen.  (NTDB has NO drug-screen field;
# only ALCOHOLSCREEN[/RESULT] exist.)
_L2_EXTRA = [
    "HOSPITALARRIVALHRS", "TBIPUPILLARYRESPONSE",
    "ALCOHOLSCREEN", "ALCOHOLSCREENRESULT",
]
# Anatomy scores (AIS-derived, a posteriori).
_ANATOMY = ["ISS", "NISS"]
# In-hospital extras.  (EDDISCHARGEHRS = ED length-of-stay was REMOVED in
# round 56: for the mortality target it leaks — an ED death's "ED discharge"
# IS the death event, so EDDISCHARGEHRS is time-to-death for those cases, and
# in general ED LOS is fixed by the disposition (died/admitted/home) and so is
# only known once the outcome is.  Dropped for all targets to be safe.)
_L3_EXTRA: list[str] = []

_ONSCENE_LIST = (_DEMOGRAPHICS + _ANTHRO + _MECHANISM + _ONSCENE_PHYS
                 + _COMORBID + _L1_EXTRA)
_ED_LIST = _ONSCENE_LIST + _ED_VITALS + _L2_EXTRA
_INHOSP_LIST = _ED_LIST + list(INJURY_FEATURES) + _ANATOMY + _L3_EXTRA

PHASE_CUTOFF_PREDICTORS: dict[str, list[str]] = {
    "On-scene": list(_ONSCENE_LIST),
    "On-scene + ED arrival": list(_ED_LIST),
    "On-scene + ED arrival + In-hospital": list(_INHOSP_LIST),

    # Baseline-feature cutoffs (pinned to one clinical score's exact inputs).
    "iss_only": [
        "ISS",
        "AGEYEARS", "SEX",
    ],
    "niss_only": [
        "NISS",
        "AGEYEARS", "SEX",
    ],
    "triss_inputs": [
        "ISS", "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "AGEYEARS", "SEX", "TRAUMATYPE",
    ],
    "all_baseline_inputs": [
        "ISS", "NISS",
        "GCSTOTAL", "SBPFIRST", "RRFIRST",
        "AGEYEARS", "SEX", "TRAUMATYPE",
    ],
}


# ---------------------------------------------------------------------------
# Round 26: SEMANTIC categorical override list.
# ---------------------------------------------------------------------------
# Some NTDB columns are STORED as integers (1=Male, 2=Female; 1=Blunt,
# 2=Penetrating, ...) but represent CATEGORICAL concepts.  Median-imputing
# them is meaningless ("median sex = 1.5"?).  These columns must be treated
# as categoricals and mode-imputed regardless of pandas dtype.
#
# The trainer's _select_predictors() flips any column listed here from
# numeric_cols to categorical_cols before the imputer fit step.
SEMANTIC_CATEGORICALS = {
    # Sex / mechanism / intent are coded integers but discrete categories
    "SEX", "TRAUMATYPE", "MECHANISM", "INTENT",
    # Demographics (there is no single native RACE column).  PRIMARYMETHODPAYMENT
    # (insurance/payer) is a coded categorical and also a subgroup axis.
    "ETHNICITY", "PRIMARYMETHODPAYMENT",
    # Round 51: integer-coded categoricals newly added as predictors
    "TRANSPORTMODE", "TBIPUPILLARYRESPONSE", "PREHOSPITALCARDIACARREST",
    "ALCOHOLSCREEN",
    # Comorbidity flags (Y/N or 1/0)
    "SMOKINGSTATUS", "COPD", "CHF", "MI", "HYPERTENSION",
    "PERIPHERALVASCULARDISEASE", "ESRD", "CIRRHOSIS",
    "DIABETESMELLITUS", "BLEEDINGDISORDER", "DISSEMINATEDCANCER",
    "ALCOHOLUSEDISORDER", "MENTALPERSONALITYDISORDER",
    "SUBSTANCEABUSEDISORDERDRUG", "ATTENTIONDEFICITDISORDER",
    "DEMENTIA", "ADVANCEDDIRECTIVELIMITINGCARE",
    "FUNCTIONALLYDEPENDENTHEALTHSTATUS",
    # Round 27: Barell injury features — always 0/1, no NaN.  Listed here
    # so the trainer treats them as categorical (mode-imputes to 0 if
    # somehow a NaN slips in; the loader normally fills with 0 directly).
    "BARELL_TBI", "BARELL_OTHER_HEAD", "BARELL_FACE", "BARELL_NECK",
    "BARELL_SCI", "BARELL_VERTEBRAL_NO_SCI",
    "BARELL_THORAX", "BARELL_ABDOMEN_PELVIS",
    "BARELL_UPPER_EXTREMITY", "BARELL_LOWER_EXTREMITY",
    "BARELL_BURNS", "BARELL_SYSTEM_OR_OTHER",
    "INJ_SUBDURAL_HEMORRHAGE", "INJ_CONCUSSION", "INJ_PNEUMOTHORAX",
    "INJ_RIB_FRACTURE_MULTIPLE", "INJ_SPLENIC_LACERATION",
    "INJ_LIVER_LACERATION", "INJ_PELVIC_FRACTURE",
    "INJ_FEMUR_FRACTURE", "INJ_DISTAL_RADIUS_FRACTURE",
    "INJ_FOOT_FRACTURE",
}
SEMANTIC_CATEGORICALS_LOWER = {c.lower() for c in SEMANTIC_CATEGORICALS}


def is_semantic_categorical(column_name: str) -> bool:
    """Case-insensitive check.  True → mode-impute, not median-impute."""
    return column_name.lower() in SEMANTIC_CATEGORICALS_LOWER





REGISTRY_COLUMNS = {
    "Tran NTDB":           "Tran 2022",
    "Karolinska SweTrau":  "Karolinska\nSweTrau\n(Holtenius)",
    "RETRAUCI":            "RETRAUCI\n(Servia)",
    "ECTrauma":            "ECTrauma",
    "EIPD":                "EIPD",
}


@dataclass
class VariableEntry:
    """One row of sheet 6 — metadata for a single NTDB variable."""
    name: str
    friendly_name: str | None
    category: str | None
    phase: str | None
    years_available: list[int] = field(default_factory=list)
    registry_flags: dict[str, bool] = field(default_factory=dict)
    observations: str | None = None

    def is_available_in(self, year: int) -> bool:
        return year in self.years_available

    def used_in(self, registry: str) -> bool:
        return self.registry_flags.get(registry, False)


class Catalogue:
    """Queryable wrapper around sheet 6 of the NTDB mapping xlsx.

    Parameters
    ----------
    xlsx_path : path to the NTDB_variable_mapping*.xlsx
    sheet_name : sheet name (defaults to the authoritative sheet 6)
    header_row : 1-indexed row holding column headers (4 in the supplied file)

    Examples
    --------
    >>> cat = Catalogue("NTDB_variable_mapping_for_trauma_ML.xlsx")
    >>> cat.variables_for(year=2024, phase_cutoff="At ED arrival",
    ...                    registries=["Tran NTDB"])
    ['AGEYEARS', 'SEX', 'TOTALGCS', ...]
    """

    def __init__(
        self,
        xlsx_path: str | Path,
        sheet_name: str = "6. NTDB variable catalogue",
        header_row: int = 4,
    ) -> None:
        self.xlsx_path = Path(xlsx_path)
        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"Variable catalogue not found: {self.xlsx_path}")

        # header_row is 1-indexed for user friendliness; pandas wants 0-indexed
        df = pd.read_excel(
            self.xlsx_path,
            sheet_name=sheet_name,
            header=header_row - 1,
            engine="openpyxl",
        )

        # Drop fully empty rows and section-header rows (which have only col 0 filled)
        df = df.dropna(subset=["NTDB variable"])
        df = df[df.iloc[:, 1:].notna().any(axis=1)].reset_index(drop=True)

        self._raw = df
        self._entries: dict[str, VariableEntry] = {}
        self._build_entries()

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #
    def _build_entries(self) -> None:
        year_cols = {2019: "AY 2019", 2020: "AY 2020", 2021: "AY 2021",
                     2022: "AY 2022", 2024: "AY 2024"}

        for _, row in self._raw.iterrows():
            name = str(row["NTDB variable"]).strip()
            years_available = [
                yr for yr, col in year_cols.items()
                if pd.notna(row.get(col)) and str(row[col]).strip() not in ("", "—", "-")
            ]
            registry_flags = {
                registry_name: str(row.get(col, "")).strip().lower() == "yes"
                for registry_name, col in REGISTRY_COLUMNS.items()
            }
            entry = VariableEntry(
                name=name,
                friendly_name=_clean(row.get("Suggested friendly name")),
                category=_clean(row.get("Category")),
                phase=_clean(row.get("Phase")),
                years_available=years_available,
                registry_flags=registry_flags,
                observations=_clean(row.get("Observations / caveats")),
            )
            self._entries[name] = entry

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def get(self, name: str) -> VariableEntry | None:
        return self._entries.get(name)

    def all_variables(self) -> list[str]:
        return list(self._entries.keys())

    def incremental_phase_blocks(
        self,
        phase_cutoff: str,
        registries: Iterable[str] | None = None,
        include_target_derivers: bool = False,
    ) -> dict[str, set]:
        """Return the INCREMENTAL variable blocks up to ``phase_cutoff``.

        Keys are 'on_scene', 'ed_arrival', 'in_hospital' (only those at/under
        the cutoff and non-empty).  Each value is the set of NEW variables that
        phase contributes, bounded by the requested cutoff.  Used to build the
        'phase-complete' evaluation slice (rows with >=1 real value per block).
        """
        target = set(self.variables_for(
            phase_cutoff=phase_cutoff, registries=registries,
            include_target_derivers=include_target_derivers))
        order = [("on_scene", "On-scene"),
                 ("ed_arrival", "On-scene + ED arrival"),
                 ("in_hospital", "On-scene + ED arrival + In-hospital")]
        blocks: dict[str, set] = {}
        prev: set = set()
        for name, cut in order:
            try:
                cur = set(self.variables_for(
                    phase_cutoff=cut, registries=registries,
                    include_target_derivers=include_target_derivers)) & target
            except Exception:
                cur = prev
            block = cur - prev
            if block:
                blocks[name] = block
            prev = cur
        return blocks

    def variables_for(
        self,
        year: int | None = None,
        phase_cutoff: str | None = None,
        registries: Iterable[str] | None = None,
        categories: Iterable[str] | None = None,
        include_target_derivers: bool = True,
        include_injury_features: bool = False,
    ) -> list[str]:
        """Return NTDB variable names matching all filters.

        Parameters
        ----------
        year : restrict to variables present in this admission year (optional).
        phase_cutoff : keep variables whose phase is <= this one in PHASE_ORDER.
            Use ``None`` for no phase filter.  "Demographics / admin" is always
            kept because age / sex / race are knowable at any point.
        registries : OR across registries — restrict to variables used by ANY
            of these registries.  Phase filtering is STILL applied; the registry
            flag is a *subset* filter, not a bypass of the phase filter.
        categories : OR across categories — keep only variables in these categories.
        include_target_derivers : if True, also keep variables that can be used
            to derive the target (mortality, ISS, NISS) even if they violate the
            phase or category filters.
        include_injury_features : if True, add the injury-description features
            (Barell / INJ_*) regardless of phase cutoff.  Used for ISS_band /
            NISS_band targets, where predicting the severity band from the
            injuries themselves is the task and those features are therefore
            legitimate at every phase.  ISS / NISS themselves remain excluded
            by the caller's target-column filter.  For outcome targets
            (mortality) this stays False, so injury features appear only at
            the in-hospital cutoff (where PHASE_CUTOFF_PREDICTORS lists them).
        """
        # ── Round 21: HARD-CODED phase cutoff predictor lists ─────────────
        # The catalogue xlsx's Phase column and Tran 2022 registry flag are
        # too inconsistent across NTDB years to drive predictor selection
        # reliably (some years tag GCSTOTAL as "On-scene", others as "At ED
        # arrival"; some Tran-flagged rows are blank; etc.).  We bypass the
        # xlsx entirely for predictor selection and return the explicit
        # PHASE_CUTOFF_PREDICTORS list instead.
        #
        # The catalogue object is still used elsewhere for friendly_name /
        # category / observations lookups, but it no longer GATES which
        # variables are predictors.
        if phase_cutoff is not None and phase_cutoff in PHASE_CUTOFF_PREDICTORS:
            out = list(PHASE_CUTOFF_PREDICTORS[phase_cutoff])
            if include_target_derivers:
                target_deriver_names = [
                    "HOSPDISCHARGEDISPOSITION", "EDDISCHARGEDISPOSITION", "DEATHINED",
                    "ISS", "ISS_05", "ISSVERSION",
                    "AISSEVERITY", "AISPREDOT", "AISVERSION", "ISSREGION",
                    "PRIMARYECODEICD10", "TRAUMATYPE", "INTERFACILITYTRANSFER",
                    "ICDDIAGNOSISCODE",
                ]
                out.extend(target_deriver_names)
            # Round 51: for band targets, allow injury features at every phase.
            if include_injury_features:
                out.extend(INJURY_FEATURES)
            # Always strip the non-predictor blacklist (case-insensitive).
            return sorted({c for c in set(out) if not is_non_predictor(c)})

        # If phase_cutoff is None, fall through to the legacy xlsx-driven
        # path (used by audit / reporting tools, not by the trainer).

        # Resolve the maximum allowable phase index
        if phase_cutoff is not None:
            resolved_phase = PHASE_CUTOFF_ALIASES.get(phase_cutoff, phase_cutoff)
            if resolved_phase not in PHASE_ORDER:
                raise ValueError(
                    f"Unknown phase_cutoff {phase_cutoff!r}. "
                    f"Known: {list(PHASE_CUTOFF_ALIASES)} or {PHASE_ORDER}"
                )
            phase_idx_cutoff = PHASE_ORDER.index(resolved_phase)
        else:
            phase_idx_cutoff = len(PHASE_ORDER) - 1   # no filter

        target_deriver_names = {
            # Mortality outcome variables
            "HOSPDISCHARGEDISPOSITION", "EDDISCHARGEDISPOSITION", "DEATHINED",
            # ISS family
            "ISS", "ISS_05", "ISSVERSION",
            # NISS derivers (AIS rows)
            "AISSEVERITY", "AISPREDOT", "AISVERSION", "ISSREGION",
            # Inclusion helpers referenced by Tran's recipe
            "PRIMARYECODEICD10", "TRAUMATYPE", "INTERFACILITYTRANSFER",
            "ICDDIAGNOSISCODE",
        }

        registries = set(registries) if registries else None
        categories = set(categories) if categories else None

        out: list[str] = []
        for name, e in self._entries.items():
            # ── Year filter ───────────────────────────────────────────────
            if year is not None and not e.is_available_in(year):
                continue

            # ── Target-deriver fast path ──────────────────────────────────
            if include_target_derivers and name in target_deriver_names:
                out.append(name)
                continue

            # ── Phase filter ──────────────────────────────────────────────
            # "Demographics / admin" is always knowable (age, sex, race, …)
            # regardless of the selected phase cutoff.
            is_demographics = e.phase == "Demographics / admin"
            if not is_demographics and e.phase is not None and e.phase in PHASE_ORDER:
                if PHASE_ORDER.index(e.phase) > phase_idx_cutoff:
                    continue   # variable comes too late in the patient journey

            # ── Registry filter ───────────────────────────────────────────
            # When a registry whitelist is given, only include variables that
            # appear in at least one of those registries.  This is applied
            # AFTER the phase filter, so it further restricts — not relaxes —
            # the set.
            if registries is not None:
                if not any(e.used_in(r) for r in registries):
                    continue

            # ── Category filter ───────────────────────────────────────────
            if categories is not None and e.category not in categories:
                continue

            out.append(name)

        # Round 51: band-target injury features (parity with hard-coded path).
        if include_injury_features:
            out.extend(INJURY_FEATURES)

        # Round 18: hard-blacklist IDs and hospital-level admin columns
        # before returning.  Round 19: case-insensitive — see is_non_predictor.
        return sorted({c for c in set(out) if not is_non_predictor(c)})

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def variables_by_phase(self, phase: str) -> list[str]:
        return sorted(n for n, e in self._entries.items() if e.phase == phase)

    def variables_by_category(self, category: str) -> list[str]:
        return sorted(n for n, e in self._entries.items() if e.category == category)

    def variables_by_registry(self, registry: str) -> list[str]:
        if registry not in REGISTRY_COLUMNS:
            raise ValueError(
                f"Unknown registry {registry!r}. "
                f"Known: {sorted(REGISTRY_COLUMNS)}"
            )
        return sorted(n for n, e in self._entries.items() if e.used_in(registry))

    def available_years(self, name: str) -> list[int]:
        e = self._entries.get(name)
        return list(e.years_available) if e else []

    def summary(self) -> pd.DataFrame:
        """Return a tidy summary dataframe (one row per variable)."""
        rows = []
        for e in self._entries.values():
            row = {
                "variable": e.name,
                "friendly_name": e.friendly_name,
                "category": e.category,
                "phase": e.phase,
                "years": ",".join(map(str, sorted(e.years_available))),
            }
            for registry in REGISTRY_COLUMNS:
                row[registry] = e.registry_flags.get(registry, False)
            rows.append(row)
        return pd.DataFrame(rows)


def _clean(value) -> str | None:
    """Convert NaN/empty cells to None, strip whitespace otherwise."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text and text not in ("—", "-") else None
