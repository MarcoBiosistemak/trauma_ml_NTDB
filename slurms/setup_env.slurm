#!/bin/bash
#SBATCH --job-name=trauma_setup
#SBATCH --output=slurms/logs/setup_%j.out
#SBATCH --error=slurms/logs/setup_%j.err
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-user=marcovinicio.caporangel@bio-sistemak.eus
#SBATCH --mail-type=ALL

# trauma_ml — one-off venv creation on DIPC.
# Run from the repository root:
#     sbatch slurms/setup_env.slurm
#
# In practice it's faster and more reliable to do this interactively on
# a login node (so you can fix mid-install errors immediately).  See the
# DIPC_RUNBOOK.md "Step 2 — Build the venv" block for the interactive
# version.  This slurm is kept here as a reproducible record of the same
# steps + as a fallback if you don't want to babysit an interactive shell.
#
# IMPORTANT: this slurm uses module + venv (NOT conda).  On atlas-edr,
# /dipc/${USER} is mounted on login but inconsistently visible from
# compute nodes.  /scratch/${USER} IS visible from both → put venv there.

set -euo pipefail
mkdir -p slurms/logs

VENV=/scratch/${USER}/trauma_ml_venv

# ── 1. Load Python + CUDA modules ───────────────────────────────────────
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.1.1

# ── 2. Create or reuse venv ─────────────────────────────────────────────
if [[ -f "${VENV}/bin/activate" ]]; then
  echo "[INFO] venv ${VENV} already exists — updating in place."
else
  echo "[INFO] Creating venv ${VENV}"
  python -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
which python

# ── 3. Install build tools ──────────────────────────────────────────────
python -m pip install -U pip wheel
# setuptools<81 is required by tpot/stopit (pkg_resources removed in 81+)
python -m pip install "setuptools<81"

# ── 4. Install torch with CUDA ──────────────────────────────────────────
echo "[INFO] Installing torch (CUDA 12.1 wheels)"
python -m pip install torch \
  --index-url https://download.pytorch.org/whl/cu121 \
  || {
    echo "[WARN] CUDA torch install failed, falling back to CPU torch"
    python -m pip install torch
  }

# ── 5. Install everything else ──────────────────────────────────────────
echo "[INFO] Installing trauma_ml dependencies (--prefer-binary)"
python -m pip install --prefer-binary -r slurms/requirements.txt

# ── 6. Install trauma_ml in editable mode ───────────────────────────────
echo "[INFO] Installing trauma_ml in editable mode"
python -m pip install -e .

# ── 7. Sanity check ─────────────────────────────────────────────────────
echo "[INFO] Sanity check — what's installed?"
python -c "
import sklearn, xgboost, lightgbm
import pandas, numpy, pyarrow
print('sklearn      :', sklearn.__version__)
print('xgboost      :', xgboost.__version__)
print('lightgbm     :', lightgbm.__version__)
print('pandas       :', pandas.__version__)
print('numpy        :', numpy.__version__)
print('pyarrow      :', pyarrow.__version__)
for pkg in ('catboost', 'torch', 'flaml', 'tpot', 'tabpfn',
            'shap', 'imblearn', 'miceforest', 'sksurv'):
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', '(no __version__)')
        if pkg == 'torch':
            v = f'{v} cuda:{m.cuda.is_available()}'
        print(f'{pkg:12s} :', v)
    except ImportError as e:
        print(f'{pkg:12s} : MISSING')

# Make sure the trauma-* console scripts are on PATH
import shutil
for cmd in ('trauma-build', 'trauma-train', 'trauma-aggregate-metrics'):
    p = shutil.which(cmd)
    print(f'{cmd:30s} -> {p}')
"

echo "[OK] Setup complete."
