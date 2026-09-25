#!/usr/bin/env python3
"""Verify the re-run best-per-phase models reproduce their reported metrics."""
import json, re, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
TXT  = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT/"analysis/best_models_by_phase.txt")
BAND = {"iss_band", "niss_band"}
TOL  = 1e-4

def load(p):
    try:    return json.loads(p.read_text())
    except Exception: return {}

def parse_txt(path):
    tgt, sel = None, None
    for ln in path.read_text().splitlines():
        m = re.match(r"^###\s+.*\[(\w+)\]", ln)
        if m: tgt = m.group(1); continue
        m = re.match(r"^\s*selection metric:\s*(\S+)", ln)
        if m: sel = m.group(1); continue
        m = re.match(r"^\s{4}(L[123])\s+(\S+)\s+([\d.]+|n/a)\s", ln)
        if m and tgt: yield tgt, sel, m.group(1), m.group(2), m.group(3)

def metrics_for(target, mid):
    d = ROOT/target/"metrics"/mid
    keys = (["balanced_accuracy","f1_macro"] if target in BAND else ["AUROC","AUPRC"])
    out = {}
    for part, fn in (("test","overall__test.json"), ("hold","overall__holdout.json")):
        blob = load(d/fn)
        for k in keys:
            out[f"{k}_{part}"] = blob.get(k)
    return keys, out

rows, bad, miss = [], 0, 0
for target, sel, phase, mid, expected in parse_txt(TXT):
    art = ROOT/target/"models"/mid/"artifact.pkl"
    keys, m = metrics_for(target, mid)
    key_test = f"{sel}_test" if f"{sel}_test" in m else f"{keys[1]}_test"
    got = m.get(key_test)
    try:    exp = float(expected)
    except ValueError: exp = None
    if got is None:
        status, miss = "NO METRICS", miss + 1
    elif exp is None:
        status = "no expected"
    elif abs(got - exp) <= TOL:
        status = "MATCH"
    else:
        status, bad = f"DIFF {got-exp:+.4f}", bad + 1
    rows.append((target, phase, mid, art.is_file(), exp, got, m, keys, status))

w = max((len(r[0]) for r in rows), default=10)
print("="*118)
print(f"{'target':<{w}} {'ph':<3} {'model_id':<22} {'art':<4} "
      f"{'expected':>9} {'got':>9}  {'status':<14} other metrics (test / holdout)")
print("="*118)
for target, phase, mid, has_art, exp, got, m, keys, status in rows:
    extra = "  ".join(
        f"{k}={m.get(k+'_test') if m.get(k+'_test') is None else format(m[k+'_test'],'.4f')}"
        f"/{m.get(k+'_hold') if m.get(k+'_hold') is None else format(m[k+'_hold'],'.4f')}"
        for k in keys)
    e = f"{exp:.4f}" if exp is not None else "-"
    g = f"{got:.4f}" if got is not None else "-"
    print(f"{target:<{w}} {phase:<3} {mid:<22} {'yes' if has_art else 'NO ':<4} "
          f"{e:>9} {g:>9}  {status:<14} {extra}")
print("="*118)
print(f"{len(rows)} models: {sum(1 for r in rows if r[8]=='MATCH')} match, "
      f"{bad} differ, {miss} missing metrics, "
      f"{sum(1 for r in rows if not r[3])} without artifact.pkl")
print(f"(tolerance {TOL}; 'other metrics' are test/holdout for the target's metric pair)")
