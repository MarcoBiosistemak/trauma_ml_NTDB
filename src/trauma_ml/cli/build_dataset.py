"""`trauma-build` CLI — build the unified multi-year NTDB dataset.

Example
-------
  # Build a single unified parquet (AY 2019-2024):
  trauma-build --anonymize

  # Split off 2024 as an external holdout; train pool is 2019-2022:
  trauma-build --anonymize --holdout-years 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..catalogue import Catalogue
from ..ntdb_loader import build_unified_dataset
from ._common import load_yaml, resolve_repo_root, setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build unified NTDB dataset.")
    parser.add_argument("--config", default="config/paths.yaml")
    parser.add_argument("--output", default=None,
                         help="TRAIN-pool parquet path (auto-named if omitted)")
    parser.add_argument("--holdout-output", default=None,
                         help="HOLDOUT parquet path (defaults to unified_holdout.parquet)")
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--holdout-years", nargs="+", type=int, default=None,
                         help="Admission years set aside as external holdout (e.g. 2024)")
    parser.add_argument("--phase-cutoff", default="In-hospital (a posteriori)")
    parser.add_argument("--registries", nargs="*", default=None,
                         choices=["Tran NTDB", "Karolinska SweTrau", "RETRAUCI",
                                  "ECTrauma", "EIPD"])
    parser.add_argument("--anonymize", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)
    repo_root = resolve_repo_root()
    cfg = load_yaml(repo_root / args.config)

    catalogue = Catalogue(
        repo_root / cfg["variable_catalogue"],
        sheet_name=cfg["catalogue_sheet_name"],
        header_row=cfg["catalogue_header_row"],
    )
    all_year_subdirs = {int(k): v for k, v in cfg["ntdb_year_subdirs"].items()}
    requested = args.years if args.years is not None else sorted(all_year_subdirs.keys())
    holdout = set(args.holdout_years) if args.holdout_years else set()
    train_years = [y for y in requested if y not in holdout]
    holdout_years = [y for y in requested if y in holdout]

    default_train_name = "unified_train.parquet" if holdout else "unified_ntdb.parquet"
    train_output = Path(args.output) if args.output else (
        repo_root / cfg["datasets_dir"] / default_train_name
    )
    print(f"[INFO] Training-pool years: {train_years}  ->  {train_output}")
    build_unified_dataset(
        catalogue=catalogue,
        ntdb_root=repo_root / cfg["ntdb_root"],
        ntdb_year_subdirs=all_year_subdirs,
        ntdb_tables=cfg["ntdb_tables"],
        years=train_years,
        phase_cutoff=args.phase_cutoff,
        registries=args.registries,
        anonymize=args.anonymize,
        output_path=train_output,
        inc_key_col=cfg.get("record_id_column", "INC_KEY"),
    )
    print(f"[OK] Training pool: {train_output}")

    if holdout_years:
        holdout_output = Path(args.holdout_output) if args.holdout_output else (
            repo_root / cfg["datasets_dir"] / "unified_holdout.parquet"
        )
        print(f"[INFO] Holdout years:        {holdout_years}  ->  {holdout_output}")
        build_unified_dataset(
            catalogue=catalogue,
            ntdb_root=repo_root / cfg["ntdb_root"],
            ntdb_year_subdirs=all_year_subdirs,
            ntdb_tables=cfg["ntdb_tables"],
            years=holdout_years,
            phase_cutoff=args.phase_cutoff,
            registries=args.registries,
            anonymize=args.anonymize,
            output_path=holdout_output,
            inc_key_col=cfg.get("record_id_column", "INC_KEY"),
        )
        print(f"[OK] External holdout: {holdout_output}")


if __name__ == "__main__":
    main()
