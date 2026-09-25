#!/usr/bin/env python3
"""Find models with corrupt/empty/missing metrics and invalidate them so the
next slurm run re-trains them **under the same model_id**.

WHY THIS EXISTS
---------------
Long sweeps get interrupted: walltime kills, cancelled jobs, a filesystem that
hit quota mid-write. That can leave a metrics JSON zero-length or truncated.
Two distinct failure modes follow, and only one is self-healing:

  * `overall__test.json` is the RESUME SENTINEL, and run_experiments.py checks
    `exists() AND st_size > 0`. A zero-length sentinel is therefore re-run
    automatically -- no action needed.

  * ANY OTHER json (overall__holdout.json, overall__test__platt.json,
    config.json, cohort_counts.json, ...) is NOT size-checked. If one of those
    is corrupt, the model still looks "complete" to the resume tracker, so it
    is SKIPPED FOREVER -- while the aggregator dies on it with
    `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.

The fix for the second case is to delete the model's metrics directory (and its
artifact), which makes the sentinel disappear. Because `model_id` is derived
from the combo's *position in the grid* (`{prefix}_{index:04d}`) and the grid is
enumerated deterministically, re-running the same slurm regenerates exactly the
same model_id with exactly the same configuration. Nothing is renumbered and no
other model is disturbed.

WHAT IT CHECKS
--------------
For every `outputs/<target>/metrics/<model_id>/`:
  1. every `*.json` under it parses as JSON and is non-empty
  2. `models/<model_id>/config.json` parses (if present)
  3. `overall__test.json` exists and is non-empty (the sentinel itself)
Subgroup CSVs are checked for zero length only (they are re-generated with the
model, so a corrupt one is repaired by the same re-run).

USAGE
-----
    # 1. DRY RUN -- report only, change nothing
    python3 scripts/repair_corrupt_metrics.py --outputs-root outputs

    # 2. Invalidate the affected models so they get re-trained
    python3 scripts/repair_corrupt_metrics.py --outputs-root outputs --apply

    # 3. Re-submit the slurms (finished combos are still skipped; only the
    #    invalidated ones are recomputed, with their original model_ids)
    # 4. Re-run the aggregator once the queue drains.

Add `--list-ids` to print the affected model_ids (useful for a targeted
--start-id/--end-id re-run instead of a whole-slurm resubmit).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time as _time
from pathlib import Path

DEFAULT_TARGETS = [
    "mortality", "iss_band", "niss_band", "iss_band_binary", "niss_band_binary",
]

SENTINEL = "overall__test.json"

# "General" metrics = the whole-cohort results. A model is only worth
# re-training if one of these is missing or corrupt. Calibrated variants
# (__platt/__isotonic) are NOT required: they exist only when that model's
# calibration axis was set, so a missing one is normal, not damage.
REQUIRED_OVERALL = [
    "overall__train.json",
    "overall__test.json",
    "overall__holdout.json",
]


def check_json(path: Path) -> str | None:
    """Return a reason string if the file is bad, else None."""
    try:
        if path.stat().st_size == 0:
            return "zero-length"
    except OSError as exc:
        return f"unstatable ({exc})"
    try:
        with open(path) as f:
            json.load(f)
    except ValueError as exc:            # covers JSONDecodeError
        return f"unparseable ({exc})"
    except OSError as exc:
        return f"unreadable ({exc})"
    return None


def human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PiB"


def scan_target(target: str, outputs_root: Path, deep: bool = False,
                progress_every: int = 500) -> dict:
    metrics_root = outputs_root / target / "metrics"
    models_root = outputs_root / target / "models"
    if not metrics_root.is_dir():
        return {"target": target, "skipped": "no metrics dir"}

    bad: dict[str, list[str]] = {}       # model_id -> CRITICAL problems
    minor: dict[str, list[str]] = {}     # model_id -> non-critical (subgroups)
    n_models = 0
    n_json = 0

    model_dirs = sorted(p for p in metrics_root.iterdir() if p.is_dir())
    total = len(model_dirs)
    t0 = _time.time()
    print(f"[{target}] scanning {total} model dirs "
          f"({'deep' if deep else 'fast'} mode)...", flush=True)

    for _i, mdir in enumerate(model_dirs, 1):
        if _i % progress_every == 0 or _i == total:
            el = _time.time() - t0
            rate = _i / el if el > 0 else 0
            eta = (total - _i) / rate if rate > 0 else 0
            print(f"    {_i}/{total} ({100*_i/total:.0f}%) "
                  f"| {el:.0f}s elapsed, ~{eta:.0f}s left "
                  f"| {len(bad)} critical so far", flush=True)
        model_id = mdir.name
        n_models += 1
        problems: list[str] = []       # critical
        soft: list[str] = []           # minor / subgroup-only

        # 1. JSONs the aggregator actually reads.
        #    DEFAULT (fast): a single iterdir() per model dir -- we never
        #    recurse into subgroups/ or subgroups_holdout/, which on BeeGFS
        #    means hundreds of thousands fewer stat() calls. --deep restores
        #    the exhaustive rglob walk.
        if deep:
            json_files = sorted(mdir.rglob("*.json"))
        else:
            json_files = sorted(
                f for f in mdir.iterdir()
                if f.is_file() and f.suffix == ".json"
            )
        for jf in json_files:
            n_json += 1
            why = check_json(jf)
            if not why:
                continue
            rel = str(jf.relative_to(mdir))
            # Only whole-cohort ("general") results justify a re-train.
            # Subgroup files live in subgroups*/ and are recorded as minor:
            # the pruner simply sees no value for that (axis, level, metric)
            # tag, i.e. it behaves as NaN and ranking falls back to the
            # general metrics.
            if "/" in rel or rel.startswith("subgroups"):
                soft.append(f"{rel}: {why}")
            else:
                problems.append(f"{rel}: {why}")

        # 2. the model's own config.json
        cfg = models_root / model_id / "config.json"
        if cfg.exists():
            why = check_json(cfg)
            if why:
                problems.append(f"../models/{model_id}/config.json: {why}")

        # 3. every REQUIRED general-metrics file must exist and be non-empty
        for req in REQUIRED_OVERALL:
            rp = mdir / req
            if not rp.exists():
                problems.append(f"{req}: MISSING")
            elif rp.stat().st_size == 0:
                note = ("  (sentinel: resume tracker would redo this anyway)"
                        if req == SENTINEL else "")
                problems.append(f"{req}: zero-length{note}")

        # 4. zero-length subgroup CSVs (deep mode only -- expensive walk)
        for cf in (sorted(mdir.rglob("*.csv")) if deep else []):
            try:
                if cf.stat().st_size == 0:
                    soft.append(f"{cf.relative_to(mdir)}: zero-length")
            except OSError:
                pass

        if problems:
            # a zero-length required file trips both the parse loop and the
            # presence check -- collapse duplicates, preserving order
            seen = set(); uniq = []
            for pr in problems:
                key = pr.split(":")[0]
                if key in seen:
                    continue
                seen.add(key); uniq.append(pr)
            bad[model_id] = uniq
        if soft:
            minor[model_id] = soft

    return {
        "target": target, "n_models": n_models, "n_json": n_json,
        "bad": bad, "minor": minor,
        "metrics_root": metrics_root, "models_root": models_root,
    }


def invalidate(model_id: str, metrics_root: Path, models_root: Path,
               apply: bool) -> int:
    """Delete a model's metrics dir + artifact so it is re-trained.

    Returns bytes reclaimed. The model_id itself is *not* reused by anything
    else -- it is regenerated identically on the next run.
    """
    freed = 0
    for d in (metrics_root / model_id, models_root / model_id):
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                try:
                    freed += p.stat().st_size
                except OSError:
                    pass
        if apply:
            shutil.rmtree(d, ignore_errors=True)
    return freed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--apply", action="store_true",
                    help="Delete the affected models so they get re-trained. "
                         "Without this flag, DRY RUN.")
    ap.add_argument("--list-ids", action="store_true",
                    help="Print affected model_ids (for targeted re-runs).")
    ap.add_argument("--max-detail", type=int, default=10,
                    help="Max models to show problem detail for, per target.")
    ap.add_argument("--progress-every", type=int, default=500,
                    help="Print a progress line every N model dirs (default 500).")
    ap.add_argument("--deep", action="store_true",
                    help="Also walk subgroups/ dirs and CSVs. MUCH slower on a "
                         "network filesystem; the default fast scan already "
                         "covers every file the aggregator reads.")
    ap.add_argument("--out", metavar="FILE",
                    help="Write affected ids as '<target> <model_id>' lines. "
                         "Scan on a fast local copy, then feed this file to "
                         "--ids-from on the cluster.")
    ap.add_argument("--ids-from", metavar="FILE",
                    help="Skip scanning; invalidate exactly the "
                         "'<target> <model_id>' pairs listed in FILE.")
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    mode = "APPLYING (bad models WILL be deleted)" if args.apply \
        else "DRY RUN (nothing deleted)"
    print(f"{'='*72}\nMODE: {mode}\n{'='*72}\n")

    # ---- --ids-from: invalidate a precomputed list, no scanning ----------
    if args.ids_from:
        pairs = []
        for line in Path(args.ids_from).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"  ignoring malformed line: {line!r}")
                continue
            pairs.append((parts[0], parts[1]))
        print(f"Loaded {len(pairs)} (target, model_id) pairs from {args.ids_from}\n")
        freed = 0
        missing = 0
        for target, mid in pairs:
            mroot = outputs_root / target / "metrics"
            droot = outputs_root / target / "models"
            if not (mroot / mid).is_dir() and not (droot / mid).is_dir():
                missing += 1
                continue
            freed += invalidate(mid, mroot, droot, args.apply)
        print(f"{'Deleted' if args.apply else 'Would delete'} "
              f"{len(pairs)-missing} model dir(s), {human(freed)}"
              + (f"  ({missing} not present -- already gone)" if missing else ""))
        if not args.apply:
            print("\nDRY RUN. Re-run with --apply.")
        print(f"{'='*72}")
        return

    grand_bad = 0
    grand_models = 0
    grand_freed = 0
    all_ids: dict[str, list[str]] = {}

    for target in args.targets:
        res = scan_target(target, outputs_root, deep=args.deep,
                          progress_every=args.progress_every)
        if res.get("skipped"):
            print(f"[{target}] skipped: {res['skipped']}\n")
            continue

        bad = res["bad"]
        grand_models += res["n_models"]
        grand_bad += len(bad)
        n_minor = len(res.get("minor", {}))
        print(f"[{target}] models: {res['n_models']}  "
              f"json checked: {res['n_json']}  "
              f"CRITICAL (general metrics): {len(bad)}  "
              f"minor (subgroup only, NOT re-trained): {n_minor}")

        for i, (mid, problems) in enumerate(sorted(bad.items())):
            if i < args.max_detail:
                print(f"    {mid}:")
                for pr in problems[:4]:
                    print(f"        - {pr}")
                if len(problems) > 4:
                    print(f"        - ... and {len(problems)-4} more")
            elif i == args.max_detail:
                print(f"    ... and {len(bad)-args.max_detail} more models")

        freed = 0
        for mid in bad:
            freed += invalidate(mid, res["metrics_root"], res["models_root"],
                                args.apply)
        grand_freed += freed
        if bad:
            print(f"    -> {'deleted' if args.apply else 'would delete'} "
                  f"{len(bad)} model dir(s), {human(freed)}")
            all_ids[target] = sorted(bad)
        print()

    print(f"{'='*72}")
    print(f"TOTAL: {grand_bad} affected of {grand_models} models; "
          f"{'freed' if args.apply else 'would free'} {human(grand_freed)}")

    if args.out and all_ids:
        with open(args.out, "w") as f:
            f.write("# <target> <model_id> -- feed to --ids-from on the cluster\n")
            for tgt, ids in all_ids.items():
                for mid in ids:
                    f.write(f"{tgt} {mid}\n")
        n = sum(len(v) for v in all_ids.values())
        print(f"\nWrote {n} affected ids to {args.out}")

    if args.list_ids and all_ids:
        print("\nAffected model_ids:")
        for t, ids in all_ids.items():
            print(f"  {t}: {' '.join(ids)}")

    if not args.apply and grand_bad:
        print("\nDRY RUN. Re-run with --apply, then re-submit the slurms: the "
              "deleted models are regenerated with the SAME model_id, and every "
              "other finished combo is still skipped.")
    elif args.apply and grand_bad:
        print("\nDone. Now re-submit the training slurms; the invalidated "
              "combos will be recomputed under their original model_ids. "
              "Re-run the aggregator afterwards.")
    elif not grand_bad:
        print("\nNo corrupt or missing metrics found.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
