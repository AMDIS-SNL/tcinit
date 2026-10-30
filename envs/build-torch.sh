#!/bin/bash
# Build the PyTorch virtual environment: Aurora, SFNO (earth2studio), HICCUP
# backends on CUDA 12. See envs/README.md for the host and modules this was
# built against.
#
# Usage (from anywhere, with the modules in envs/README.md loaded, so
# python3 is 3.12; or set PYTHON=/path/to/python3.12):
#   envs/build-torch.sh
#   VENV=/other/place/torch-venv TORCH_CUDA=cu129 envs/build-torch.sh
set -euo pipefail

# Home quota is small on cee-compute; keep the venv and pip's wheel cache
# (torch alone is several GB) on machine scratch.
ENV_ROOT="${ENV_ROOT:-/fgs/$USER/causaltc}"
VENV="${VENV:-$ENV_ROOT/torch-venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ENV_ROOT/pip-cache}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
MAKANI_REV="b38fcb2799d7dbc146fa60459f3f9823394a8bf1"  # pin used by earth2studio

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "error: $PYTHON is $("$PYTHON" --version 2>&1) at $(command -v "$PYTHON");" \
         "need Python >= 3.11. Load the modules in envs/README.md" \
         "(aue/python/3.12.4) or set PYTHON=/path/to/python3.12" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -e "$VENV" ]]; then
    echo "error: $VENV already exists; remove it or set VENV" >&2
    exit 1
fi

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install --upgrade pip

# CUDA torch first, so later installs don't pull a CPU or other-CUDA build.
pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# SFNO. makani isn't on PyPI; install it from git before earth2studio.
pip install "makani @ git+https://github.com/NVIDIA/makani.git@${MAKANI_REV}"
pip install "earth2studio[sfno]"

# tcinit with every torch-side extra (microsoft-aurora, netCDF4, h5netcdf,
# cartopy, dev tools).
pip install -e "${REPO_ROOT}[aurora,hiccup,plotting,dev]"

# GRIB input (HRES .grb files).
pip install cfgrib eccodes

pip check
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version(),
      "gpu available", torch.cuda.is_available())
EOF
echo "built $VENV; then run pytest -q and envs/lock.sh torch"
