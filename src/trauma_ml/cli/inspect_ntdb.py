"""Diagnostic: inspect a raw NTDB year folder and report what's there.

Run on a single year subdir to see:
  * What CSV files exist
  * What columns PUF_TRAUMA has that look comorbidity-related
  * Whether a long-format comorbidity table exists, and if so what condition
    strings appear in it

Usage:
    python -m trauma_ml.cli.inspect_ntdb data/NTDB/PUF_AY_2021/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Keywords we search for in column names AND condition strings to surface
# anything comorbidity-related, regardless of how NTDB names it that year.
COMORBIDITY_KEYWORDS = [
    "copd", "chronic", "obstructive", "pulmonary",
    "chf", "congestive", "heart", "failure",
    "myocardial", "infarction",
    "hypertension", "htn",
    "peripheral", "vascular", "arterial",
    "esrd", "renal", "dialysis", "kidney",
    "cirrhosis", "liver", "hepatic",
    "diabetes", "dm",
    "bleeding", "coagul", "anticoag",
    "cancer", "metastatic", "disseminated",
    "alcohol", "etoh",
    "smoking", "smoker", "tobacco",
    "psychiatric", "mental", "personality",
    "substance", "drug",
    "attention", "deficit", "adhd", "add",
    "dementia", "alzheimer",
    "advance", "directive", "dnr",
    "functional", "dependent",
    "comorbid", "preexisting", "pre-existing", "preexist",
    "condition", "history",
]


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("year_dir", help="Path to one NTDB year subdir, e.g. data/NTDB/PUF_AY_2021/")
    p.add_argument("--max-conditions", type=int, default=200,
                   help="Max distinct condition strings to print (default 200)")
    args = p.parse_args(argv)

    year_dir = Path(args.year_dir)
    if not year_dir.is_dir():
        raise SystemExit(f"Not a directory: {year_dir}")

    print(f"\n{'='*72}\nInspecting {year_dir}\n{'='*72}\n")

    # ── 1. List all CSV files ────────────────────────────────────────
    csvs = sorted(year_dir.glob("*.csv")) + sorted(year_dir.glob("*.CSV"))
    print(f"CSV files in this directory ({len(csvs)} total):")
    for c in csvs:
        size_mb = c.stat().st_size / 1e6
        print(f"  {c.name:50s}  {size_mb:>8.1f} MB")
    print()

    # ── 2. Identify likely main trauma + comorbidity tables ──────────
    trauma_path = None
    comorbidity_path = None
    for c in csvs:
        name_lower = c.name.lower()
        if "trauma" in name_lower and trauma_path is None:
            trauma_path = c
        if any(kw in name_lower for kw in
                ("comorbid", "preexisting", "pre_existing", "pre-existing")):
            comorbidity_path = c

    if trauma_path is None:
        print("[WARN] No file matching '*trauma*.csv' found — cannot inspect main table.")
    else:
        print(f"[INFO] Main trauma table: {trauma_path.name}")
        # Read just the header + a few rows to see columns
        df_head = pd.read_csv(trauma_path, nrows=5, low_memory=False, encoding="latin-1")
        cols = list(df_head.columns)
        print(f"       {len(cols)} columns total")

        # Find any column whose name contains comorbidity keywords
        suspicious = []
        for col in cols:
            col_lower = col.lower()
            for kw in COMORBIDITY_KEYWORDS:
                if kw in col_lower:
                    suspicious.append((col, kw))
                    break
        print(f"\n  Comorbidity-suspicious columns on PUF_TRAUMA ({len(suspicious)} found):")
        for col, kw in suspicious:
            sample_vals = df_head[col].dropna().head(3).tolist()
            print(f"    [{kw:>15s}]  {col}   sample: {sample_vals}")
        if not suspicious:
            print("    (none — comorbidities are likely in a separate table)")
        print()

    # ── 3. Inspect the comorbidity / preexisting condition table ────
    if comorbidity_path is None:
        print("[INFO] No '*comorbid*' or '*preexisting*' file found.")
    else:
        print(f"[INFO] Long-format comorbidity table: {comorbidity_path.name}")
        df = pd.read_csv(comorbidity_path, low_memory=False, encoding="latin-1")
        print(f"       Shape: {df.shape}")
        print(f"       Columns: {list(df.columns)}")
        print(f"       First 3 rows:")
        print(df.head(3).to_string(index=False))
        print()

        # Find the condition-description column
        condition_col = None
        for candidate in ("Cdescr", "CDESCR", "COMORBIDITY", "COMORBIDITY_DESCR",
                          "PreExistingCondition", "CONDITION", "COMORBID_NAME",
                          "DESCRIPTION", "CONDITIONNAME", "PRECONDITION_NAME"):
            for c in df.columns:
                if c.lower() == candidate.lower():
                    condition_col = c
                    break
            if condition_col:
                break
        if condition_col is None:
            obj_cols = [c for c in df.columns
                         if df[c].dtype == object
                         and not any(s in c.lower() for s in ("inc_key", "key"))]
            if len(obj_cols) == 1:
                condition_col = obj_cols[0]

        if condition_col:
            unique_conds = df[condition_col].astype(str).str.strip().unique()
            print(f"       Distinct values in '{condition_col}' "
                  f"({len(unique_conds)} unique):")
            for v in sorted(unique_conds)[:args.max_conditions]:
                count = (df[condition_col] == v).sum()
                print(f"         {count:>10d}  {v}")
            if len(unique_conds) > args.max_conditions:
                print(f"         ... and {len(unique_conds) - args.max_conditions} more")
        else:
            print(f"  [WARN] Could not auto-detect condition column.")

    print(f"\n{'='*72}\n")
    print("Send the output above (especially the suspicious columns and the")
    print("distinct condition strings) so I can map them to canonical names.")
    print()


if __name__ == "__main__":
    main()
