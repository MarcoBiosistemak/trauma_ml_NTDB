#!/usr/bin/env python3
"""Report training progress per model-family prefix and list the MISSING ids.

`model_id` is `<prefix>_<grid_index>`, and the grid is enumerated
deterministically, so the set of completed indices should be a contiguous run
0..N-1. Any hole in that run is a combo that has not finished yet (or whose
metrics were lost). This script finds those holes.

A model counts as COMPLETE when
`outputs/<target>/metrics/<model_id>/overall__test.json` exists and is
non-empty -- exactly the sentinel `run_experiments.py` uses to decide whether
to skip a combo, so this mirrors what the cluster will actually re-run.

Pure standard library: runs on Windows, macOS or Linux with any Python 3.7+.

USAGE
-----
    python3 scripts/check_progress.py --outputs-root outputs
    python3 scripts/check_progress.py --outputs-root outputs --show-missing 40
    python3 scripts/check_progress.py --outputs-root outputs --expected-file exp.txt

`--expected-file` takes lines of `<prefix> <count>` if you want progress
measured against the full grid size rather than against the highest index
seen so far (the default). Without it, a prefix whose *tail* is missing looks
complete -- so pass expected counts when you need a true percentage.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

SENTINEL = "overall__test.json"
DEFAULT_TARGETS = [
    "mortality", "iss_band", "niss_band", "iss_band_binary", "niss_band_binary",
]

# Known full grid sizes (see DIPC_RUNBOOK "combos per slurm"). Used only when
# --expected-file is not supplied; unknown prefixes fall back to max-index+1.
KNOWN_EXPECTED = {
    # mortality
    "xgb": 360, "xgb_none": 72, "lgb": 360, "lgb_none": 72,
    "cb": 360, "cb_none": 72, "flm": 360, "flm_none": 72,
    "lgr": 720, "rf": 360, "tpt": 360, "tnt": 360, "tpf": 360,
    "doshi": 360,
    # slurm 35 trains 3 targets in ONE grid -> ids are global and the
    # outputs land in three different target folders.
    "doshi_icd": 9, "doshi_icdplus": 54,
    # ISS / NISS 4-class
    "iss_xgb": 90, "iss_xgb_none": 6, "iss_lgb": 90, "iss_lgb_none": 6,
    "iss_flm": 90, "iss_flm_none": 6, "iss_rf": 90, "iss_tpt": 90,
    "iss_doshi": 360,   # spans iss_band + iss_band_binary
    "niss_xgb": 90, "niss_xgb_none": 6, "niss_lgb": 90, "niss_lgb_none": 6,
    "niss_flm": 90, "niss_flm_none": 6, "niss_rf": 90, "niss_tpt": 90,
    "niss_doshi": 360,  # spans niss_band + niss_band_binary
    # band-binary
    "iss_bin_boost": 1296, "iss_bin_other": 1350,
    "niss_bin_boost": 1296, "niss_bin_other": 1350,
}

ID_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d{4,})$")


def collapse(nums: list[int], limit: int) -> str:
    """Render a sorted int list as compact ranges: 1-4, 7, 10-12."""
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    s = ", ".join(out[:limit])
    if len(out) > limit:
        s += f", ... (+{len(out)-limit} more ranges)"
    return s


def scan(target: str, outputs_root: Path) -> dict:
    mroot = outputs_root / target / "metrics"
    if not mroot.is_dir():
        return {}
    done: dict[str, set] = defaultdict(set)
    started: dict[str, set] = defaultdict(set)

    for d in mroot.iterdir():
        if not d.is_dir():
            continue
        m = ID_RE.match(d.name)
        if not m:
            continue
        prefix, idx = m.group("prefix"), int(m.group("idx"))
        started[prefix].add(idx)
        s = d / SENTINEL
        try:
            if s.is_file() and s.stat().st_size > 0:
                done[prefix].add(idx)
        except OSError:
            pass
    return {"done": done, "started": started}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--show-missing", type=int, default=12,
                    help="How many missing-index ranges to print per prefix.")
    ap.add_argument("--expected-file",
                    help="File of '<prefix> <count>' lines overriding the "
                         "built-in expected grid sizes.")
    ap.add_argument("--ids", action="store_true",
                    help="Also print missing ids as full model_ids.")
    args = ap.parse_args()

    expected = dict(KNOWN_EXPECTED)
    if args.expected_file:
        for line in Path(args.expected_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                expected[parts[0]] = int(parts[1])

    root = Path(args.outputs_root)

    # A slurm may train several targets from ONE grid (e.g. 33/34/35), so a
    # prefix's indices are spread across target folders. Union them per prefix
    # before looking for holes, otherwise every such prefix looks half-missing.
    agg_done: dict[str, set] = defaultdict(set)
    agg_started: dict[str, set] = defaultdict(set)
    where: dict[str, set] = defaultdict(set)
    for target in args.targets:
        res = scan(target, root)
        if not res:
            continue
        for pfx, idxs in res["done"].items():
            agg_done[pfx] |= idxs; where[pfx].add(target)
        for pfx, idxs in res["started"].items():
            agg_started[pfx] |= idxs; where[pfx].add(target)

    g_done = g_exp = 0
    print(f"{'='*92}")
    print(f"{'prefix':<24}{'done':>7}{'expected':>10}{'%':>7}"
          f"{'missing':>9}{'started':>9}   [targets]")
    print(f"{'='*92}")

    # A prefix with ZERO models on disk would otherwise be invisible (and its
    # expected count silently excluded from the total), which hides a family
    # that never produced anything at all. Surface those explicitly.
    absent = sorted(k for k, v in expected.items()
                    if v and k not in agg_done and k not in agg_started)

    if True:
        for prefix in sorted(set(agg_done) | set(agg_started)):
            d = sorted(agg_done.get(prefix, ()))
            st = agg_started.get(prefix, set())
            exp = expected.get(prefix)
            if exp is None:
                exp = (max(st) + 1) if st else 0
                flag = "~"            # inferred, not authoritative
            else:
                flag = " "
            pct = (100.0 * len(d) / exp) if exp else 0.0
            inprog = len(st) - len(d)
            tgts = ",".join(sorted(where.get(prefix, ())))
            miss_n = max(exp - len(d), 0)
            print(f"  {prefix:<22}{len(d):>7}{exp:>9}{flag}{pct:>6.1f}%"
                  f"{miss_n:>9}{inprog:>9}   [{tgts}]")
            missing = sorted(set(range(exp)) - set(d)) if exp else []
            if missing:
                print(f"      missing idx: {collapse(missing, args.show_missing)}")
                if args.ids:
                    ids = [f"{prefix}_{i:04d}" for i in missing]
                    print(f"      ids: {' '.join(ids[:40])}"
                          + (f" ... (+{len(ids)-40})" if len(ids) > 40 else ""))
            g_done += len(d)
            g_exp += exp

    if absent:
        print("\n  *** NOT STARTED (no models on disk at all) ***")
        for k in absent:
            print(f"  {k:<22}{0:>7}{expected[k]:>9} {0.0:>6.1f}%"
                  f"{expected[k]:>9}{0:>9}   [--]")
            g_exp += expected[k]

    print(f"\n{'='*92}")
    pct = (100.0 * g_done / g_exp) if g_exp else 0.0
    print(f"TOTAL: {g_done} / {g_exp} complete ({pct:.1f}%)   "
          f"remaining: {g_exp - g_done}")
    print("missing = expected - done (everything not finished).")
    print("started = dirs that EXIST but have no valid overall__test.json, i.e."
          "\n          combos interrupted mid-write. A combo that never ran has"
          "\n          no directory, so it counts in 'missing' but not 'started'.")
    print("'~' = expected inferred from highest index seen (not authoritative);"
          "\n     supply --expected-file for exact grid sizes.")
    print(f"{'='*92}")


if __name__ == "__main__":
    main()
