#!/bin/bash
# Build the JAX virtual environment: NeuralGCM, WeatherNext (GraphCast /
# GenCast / WN2), HICCUP backends on CUDA 12. See envs/README.md for the host
# and modules this was built against.
#
# Usage (from anywhere, with the modules in envs/README.md loaded):
#   envs/build-jax.sh
#   VENV=/other/place/jax-venv envs/build-jax.sh
set -euo pipefail

# Home quota is small on cee-compute; keep the venv and pip's wheel cache on
# machine scratch.
ENV_ROOT="${ENV_ROOT:-/fgs/$USER/causaltc}"
VENV="${VENV:-$ENV_ROOT/jax-venv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ENV_ROOT/pip-cache}"
# Commit of the external/weathernext clone the backends were written against.
WEATHERNEXT_REV="f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -e "$VENV" ]]; then
    echo "error: $VENV already exists; remove it or set VENV" >&2
    exit 1
fi

python3 -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
pip install --upgrade pip

# One pip call, so the resolver sees every constraint at once. Installed
# separately, a later step can move jax to a version that no longer matches
# jax-cuda12-plugin. neuralgcm comes from PyPI (tcinit's neuralgcm extra):
# its git version pulls `dinosaur` from git, which collides with
# weathernext's `dinosaur-dycore` (same import name).
pip install \
    "jax[cuda12]" \
    "weathernext @ git+https://github.com/google-deepmind/weathernext.git@${WEATHERNEXT_REV}" \
    -e "${REPO_ROOT}[neuralgcm,hiccup,plotting,dev]" \
    gcsfs \
    cfgrib eccodes

pip check
python - <<'EOF'
import jax, jaxlib
print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
print("devices", jax.devices())
EOF
if [[ ":${LD_LIBRARY_PATH:-}:" == *cuda* ]]; then
    echo "note: LD_LIBRARY_PATH contains CUDA module paths; if JAX reports" \
         "cuDNN/cuBLAS version errors, see envs/README.md" >&2
fi
echo "built $VENV; generate a lock file as described in envs/README.md"
