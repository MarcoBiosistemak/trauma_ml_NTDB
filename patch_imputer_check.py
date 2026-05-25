"""
patch_imputer_check.py — backfill imputer_check=false into existing configs.

Round 37 added the `imputer_check` field to config.json.  Models trained
BEFORE round 37 don't have it.  Anything reading config.json should treat a
missing imputer_check as False — but to make the field explicit (and to keep
downstream analysis code simple), this script walks every
outputs/<target>/models/<model_id>/config.json and, if `imputer_check` is
absent from the nested `config` block, writes it as `false` in place.

Idempotent: configs that already have the field are left untouched.

Usage (on the cluster or locally):
    python patch_imputer_check.py /scratch/mcapo/trauma_ml/outputs
    python patch_imputer_check.py ~/Escritorio/DIPC/results/trauma_ml_v1
"""
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python patch_imputer_check.py <outputs_dir>")
        print("  <outputs_dir> contains <target>/models/<model_id>/config.json")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        sys.exit(1)

    # Find every config.json under any */models/*/ path
    config_paths = list(root.glob("*/models/*/config.json"))
    # Also handle the case where root IS a single target dir
    config_paths += list(root.glob("models/*/config.json"))

    if not config_paths:
        print(f"No config.json files found under {root}")
        print("Looked for: <target>/models/<id>/config.json and "
              "models/<id>/config.json")
        sys.exit(0)

    n_patched = 0
    n_already = 0
    n_error = 0

    for cfg_path in sorted(config_paths):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as exc:
            print(f"  [error] {cfg_path}: {exc}")
            n_error += 1
            continue

        # The flag lives inside the nested 'config' block (matching how the
        # trainer writes it).  Some very old configs may be flat.
        inner = cfg.get("config")
        if isinstance(inner, dict):
            if "imputer_check" in inner:
                n_already += 1
                continue
            inner["imputer_check"] = False
        else:
            # Flat config — add at top level
            if "imputer_check" in cfg:
                n_already += 1
                continue
            cfg["imputer_check"] = False

        try:
            cfg_path.write_text(json.dumps(cfg, indent=2))
            n_patched += 1
        except Exception as exc:
            print(f"  [error writing] {cfg_path}: {exc}")
            n_error += 1

    print(f"\nScanned {len(config_paths)} config.json files under {root}")
    print(f"  patched (added imputer_check=false): {n_patched}")
    print(f"  already had the field:               {n_already}")
    if n_error:
        print(f"  errors:                              {n_error}")
    print("\nDone. Existing models now explicitly carry imputer_check=false, "
          "matching how round-37+ models are recorded.")


if __name__ == "__main__":
    main()
