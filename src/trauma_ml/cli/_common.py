"""Shared helpers for the CLI entry points."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_repo_root(marker: str = "pyproject.toml",
                      start: Path | None = None) -> Path:
    """Walk up from ``start`` (or cwd) until we find ``marker``; return that dir."""
    here = Path(start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / marker).exists():
            return p
    # Fall back to cwd — handy when running from within PyCharm
    return Path.cwd().resolve()
