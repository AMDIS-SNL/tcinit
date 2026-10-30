#!/bin/bash
# Write a lock file for the active virtual environment.
#
# Usage, with the environment activated, on a GPU node, after pytest passes:
#   envs/lock.sh torch    # -> envs/torch-cu<XY>.lock.txt
#   envs/lock.sh jax      # -> envs/jax-cuda12.lock.txt
set -euo pipefail

KIND="${1:-}"
if [[ "$KIND" != torch && "$KIND" != jax ]]; then
    echo "usage: $0 torch|jax" >&2
    exit 1
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "error: no virtual environment is active" >&2
    exit 1
fi

ENVS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
ngpu="$(nvidia-smi -L | wc -l)"

if [[ "$KIND" == torch ]]; then
    # e.g. "12.6" -> "cu126", matching the wheel index the torch build came from
    cuda_tag="cu$(python -c 'import torch; print(torch.version.cuda.replace(".", ""))' 2>/dev/null)"
    versions="$(python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())' 2>/dev/null)"
    out="$ENVS_DIR/torch-${cuda_tag}.lock.txt"
    git_pkgs="makani"
else
    versions="$(python -c 'import jax, jaxlib; print("jax", jax.__version__, "jaxlib", jaxlib.__version__)' 2>/dev/null)"
    out="$ENVS_DIR/jax-cuda12.lock.txt"
    git_pkgs="weathernext gdm-xarray-jax"
fi

{
    echo "# $(basename "$VIRTUAL_ENV") lock, $(date +%F), $(hostname)"
    echo "# $(python --version)"
    echo "# GPU: $gpu (x$ngpu)"
    echo "# $versions"
    if [[ "$KIND" == torch ]]; then
        echo "--extra-index-url https://download.pytorch.org/whl/${cuda_tag}"
    fi
    pip freeze --exclude-editable
} > "$out"

echo "wrote $out"
head -5 "$out"

# Git-installed packages must keep their git source, or rebuilding from the
# lock fails (they aren't on PyPI at these versions).
status=0
for pkg in $git_pkgs; do
    line="$(grep -i "^${pkg} @ git+" "$out" || true)"
    if [[ -n "$line" ]]; then
        echo "ok: $line"
    else
        echo "warning: $pkg has no git source in $out; replace its line with the" \
             "git URL from the build script" >&2
        status=1
    fi
done
exit $status
