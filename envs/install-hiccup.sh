#!/bin/bash
# Clone HICCUP at a pinned commit and install it, editable, into the active
# virtual environment. build-torch.sh and build-jax.sh call this; run it by
# hand, with an environment active, to add HICCUP to an existing one.
#
# HICCUP is installed from a clone, not from git+https, because the EAMxx demo
# notebook needs the repo's data files (files_vert/, test_data/), and HICCUP's
# setup.py installs only the hiccup/ package. Both environments share the one
# clone. The notebook finds it from the installed package (hiccup.__file__).
#
# Usage:
#   envs/install-hiccup.sh
#   HICCUP_DIR=/other/place/hiccup envs/install-hiccup.sh
set -euo pipefail

ENV_ROOT="${ENV_ROOT:-/fgs/$USER/causaltc}"
HICCUP_DIR="${HICCUP_DIR:-$ENV_ROOT/hiccup}"
# Commit of the external/hiccup clone the EAMxx notebook was written against.
HICCUP_REV="820f1aec742d020d2d4748db1a8a603287647740"
HICCUP_URL="https://github.com/E3SM-Project/HICCUP.git"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "error: no virtual environment is active" >&2
    exit 1
fi

if [[ ! -d "$HICCUP_DIR/.git" ]]; then
    git clone "$HICCUP_URL" "$HICCUP_DIR"
fi
if [[ "$(git -C "$HICCUP_DIR" rev-parse HEAD)" != "$HICCUP_REV" ]]; then
    if [[ -n "$(git -C "$HICCUP_DIR" status --porcelain --untracked-files=no)" ]]; then
        echo "error: $HICCUP_DIR has local changes; commit or discard them" \
             "before it can move to $HICCUP_REV" >&2
        exit 1
    fi
    git -C "$HICCUP_DIR" fetch --quiet origin
    git -C "$HICCUP_DIR" checkout --quiet --detach "$HICCUP_REV"
fi

# HICCUP's setup.py declares no dependencies. Beyond numpy, pandas and xarray
# (already in both environments) it imports psutil.
pip install psutil
pip install -e "$HICCUP_DIR"
python -c "from hiccup import hiccup; print('hiccup', hiccup.__file__)"
