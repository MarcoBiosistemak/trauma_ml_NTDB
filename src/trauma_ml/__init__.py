"""trauma_ml — explainable ML pipeline for trauma mortality prediction.

Modules
-------
catalogue    : read the NTDB variable mapping xlsx (sheet 6) and expose queries
ntdb_loader  : load the 5 PUF admission years, apply whitelist, build unified df
targets      : define the prediction target (mortality, ISS, NISS, …)
inclusion    : cohort inclusion/exclusion strategies per registry
splitting    : stratified train/calibration/test split
imputation   : imputer bank + holdout imputation evaluation
models       : model family wrappers (linear, boosting, AutoML, DL, foundation)
trainer      : orchestrator — fit / save / predict
evaluation   : overall and subgroup metrics
persistence  : ModelArtifact (save/load of the whole pipeline state)
cli          : command-line entry points
"""
__version__ = "0.2.0"
