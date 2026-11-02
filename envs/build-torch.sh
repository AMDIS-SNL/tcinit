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
# torchvision comes from the same index: Aurora needs timm, which imports
# torchvision, and a PyPI torchvision built for another torch fails at import
# with "operator torchvision::nms does not exist".
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Hold both at these builds for every later install. Otherwise a dependency
# that asks for torchvision lets pip "upgrade" to a newer PyPI release (e.g.
# 0.29.2 over 0.29.1+cu126), which is built for another torch.
pip freeze | grep -E '^(torch|torchvision)==' > "$VENV/torch-constraints.txt"
export PIP_CONSTRAINT="$VENV/torch-constraints.txt"

# SFNO. makani isn't on PyPI; install it from git before earth2studio.
pip install "makani @ git+https://github.com/NVIDIA/makani.git@${MAKANI_REV}"
pip install "earth2studio[sfno]"

# tcinit with every torch-side extra (microsoft-aurora, netCDF4, h5netcdf,
# cartopy, dev tools).
pip install -e "${REPO_ROOT}[aurora,hiccup,plotting,dev]"

# GRIB input (HRES .grb files).
pip install cfgrib eccodes

# Jupyter, plus a named kernel so the demo notebooks can pick this env from
# any Jupyter session (the kernel spec is a small file under ~/.local).
pip install jupyter
python -m ipykernel install --user --name "$(basename "$VENV")" \
    --display-name "Python ($(basename "$VENV"))"

pip check
python - <<'EOF'
import torch
import torchvision
from aurora import AuroraHighRes  # noqa: F401  (imports timm -> torchvision ops)
from earth2studio.models.px import SFNO  # noqa: F401
print("torch", torch.__version__, "torchvision", torchvision.__version__,
      "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "gpu available", torch.cuda.is_available())
EOF
echo "built $VENV; then run pytest -q and envs/lock.sh torch"
