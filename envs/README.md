# HPC virtual environments

The model backends need two separate environments, because PyTorch and
JAX each bundle their own CUDA 12 libraries and the pinned versions clash
when installed together.

| Environment | Build script | Backends |
|---|---|---|
| `torch-venv` | `build-torch.sh` | Aurora, SFNO (earth2studio), EAMxx (via HICCUP), xarray |
| `jax-venv` | `build-jax.sh` | NeuralGCM, WeatherNext, EAMxx (via HICCUP), xarray |

Home-directory quota on cee-compute is small, so environments, pip's
wheel cache, and model checkpoints live on machine scratch under
`/fgs/$USER/causaltc/`:

```
/fgs/$USER/causaltc/
├── torch-venv/
├── jax-venv/
├── pip-cache/      # PIP_CACHE_DIR, set by the build scripts
└── model-cache/    # checkpoints; see "Model checkpoint caches" below
```

Override the location with `ENV_ROOT=...` (or `VENV=...` for a single
environment) when running a build script.

Each environment has two records:

- **`build-<name>.sh`** — what we meant to install: the commands, their
  order, and the index URLs. Use it to rebuild on purpose.
- **`<name>-<cuda>.lock.txt`** — what we actually got: exact versions
  from `pip freeze` of an environment whose tests passed. Use it to
  reproduce a working environment.

The old CausalTC package is deliberately not installed in either
environment; tcinit is meant to stand on its own.

## torch-venv

Built 2026-10-30 and verified with:

| | |
|---|---|
| Host | cee-compute044, 4× NVIDIA A100 80GB PCIe |
| NVIDIA driver | 580.126.20 |
| Python | 3.12.4 (`aue/python/3.12.4`) |
| torch | 2.14.1+cu126 (CUDA 12.6, cuDNN 9.10.02) |

These modules were loaded:

```sh
module load aue/python/3.12.4 \
    aue/cuda/12.9.0-gcc-12.3.0 \
    aue/binutils/2.45 \
    aue/gcc/12.3.0 \
    aue/hdf5/1.14.6-gcc-12.3.0-openmpi-4.1.6-cuda-12.9.0 \
    aue/openmpi/4.1.6-gcc-12.3.0-cuda-12.9.0 \
    aue/netcdf-c/4.9.2-gcc-12.3.0-openmpi-4.1.6-cuda-12.9.0
```

Build:

```sh
envs/build-torch.sh                                  # creates /fgs/$USER/causaltc/torch-venv
VENV=/other/place/torch-venv envs/build-torch.sh     # different location
```

The torch wheel is built for CUDA 12.6 (`TORCH_CUDA=cu126`) and ships
its own CUDA runtime, so it only needs a driver new enough for 12.6.
The `cuda/12.9.0` module matters only when a package compiles CUDA code
during install. If such a build complains about a CUDA version
mismatch, rebuild with `TORCH_CUDA=cu129` to match the module.

### torchvision must match torch

Aurora needs `timm`, which imports `torchvision`. If pip takes
`torchvision` from PyPI instead of the CUDA index torch came from,
`from aurora import ...` fails with
`RuntimeError: operator torchvision::nms does not exist`. `build-torch.sh`
installs both from the same index, then pins them with `PIP_CONSTRAINT`
so later installs (earth2studio, tcinit extras) can't "upgrade"
torchvision to a newer PyPI release. That is what happened in the
2026-10-30 build: PyPI's `0.29.2` replaced `0.29.1+cu126`.

To repair an existing env, install the torchvision the CUDA index pairs
with your torch (for torch 2.14.1 that is `0.29.1+cu126`; see
`pip index versions torchvision --index-url https://download.pytorch.org/whl/cu126`):

```sh
pip install --force-reinstall --no-deps "torchvision==0.29.1+cu126" \
    --index-url https://download.pytorch.org/whl/cu126
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"  # both +cu126
envs/lock.sh torch
```

### Known warning

Importing torch prints

```
FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.
```

`earth2studio[sfno]` requires `pynvml`, which is now a thin wrapper
around `nvidia-ml-py`, and torch warns when it finds it. It is harmless;
don't uninstall `pynvml`, or `pip check` will report earth2studio's
requirement as missing.

### Lock file

After the build, on a GPU node with `pytest -q` passing:

```sh
. /fgs/$USER/causaltc/torch-venv/bin/activate
envs/lock.sh torch          # writes envs/torch-cu126.lock.txt
```

The script records host, GPU, driver, Python and torch/CUDA/cuDNN
versions as header comments, adds the torch wheel index line, and
checks that makani kept its git source. Commit the file.

Reproduce from the lock:

```sh
python3 -m venv /fgs/$USER/causaltc/torch-venv
. /fgs/$USER/causaltc/torch-venv/bin/activate
export PIP_CACHE_DIR=/fgs/$USER/causaltc/pip-cache
pip install -r envs/torch-cu126.lock.txt
pip install -e . --no-deps
envs/install-hiccup.sh      # editable installs aren't in the lock; see "HICCUP" below
```

If scratch is ever purged, the venv is lost but the lock file in the
repo is not; that is the main reason to keep the lock file current.

## HICCUP (EAMxx initial conditions)

The EAMxx demo notebook uses [HICCUP](https://github.com/E3SM-Project/HICCUP)
to turn ERA5 into an EAMxx initial condition. tcinit itself doesn't import
HICCUP, so it's an environment dependency, like makani and weathernext, not
part of the package.

Both build scripts call `envs/install-hiccup.sh`, which:

- clones HICCUP into `/fgs/$USER/causaltc/hiccup` (one clone shared by both
  environments; override with `HICCUP_DIR=...`);
- checks out the pinned commit `HICCUP_REV` (the one the notebook was written
  against), refusing to move a clone with local changes;
- installs `psutil` (HICCUP imports it but its `setup.py` declares no
  dependencies) and then HICCUP itself with `pip install -e`.

It's an editable install from a clone, not `pip install git+...`, because the
notebook needs the repo's data files (`files_vert/`, `test_data/`), which
HICCUP's `setup.py` doesn't install. The notebook finds the clone from the
installed package (`hiccup.__file__`), so it doesn't depend on `external/`.

To add HICCUP to an environment built before this, without rebuilding:

```sh
. /fgs/$USER/causaltc/torch-venv/bin/activate      # or jax-venv
envs/install-hiccup.sh
envs/lock.sh torch                                  # or jax
```

`lock.sh` uses `pip freeze --exclude-editable`, so HICCUP doesn't appear as a
package line in the lock files. It writes a `# hiccup <commit>` header comment
instead. To move to a newer HICCUP, change `HICCUP_REV` in
`install-hiccup.sh` and rerun it in each environment.

HICCUP also needs command-line tools that these scripts don't install:
NCO ≥ 5.3.1 (`ncremap`, `ncks`, ...) and TempestRemap (`GenerateCSMesh`,
`GenerateOverlapMesh`, `GenerateOfflineMap`). They must be on `PATH` in the
shell that starts Jupyter.

## Model checkpoint caches

Model weights download to `~/.cache` by default and are large (Aurora
and SFNO checkpoints are several GB each). Point the caches at scratch
before running a model, e.g. in the job script or `.bashrc`:

```sh
export HF_HOME=/fgs/$USER/causaltc/model-cache/huggingface      # Aurora (Hugging Face Hub)
export EARTH2STUDIO_CACHE=/fgs/$USER/causaltc/model-cache/earth2studio  # SFNO
export PIP_CACHE_DIR=/fgs/$USER/causaltc/pip-cache              # any manual pip installs
```

If anything already downloaded to `~/.cache/pip`, `~/.cache/huggingface`
or `~/.cache/earth2studio`, it can be deleted to free quota.

## jax-venv

Same host and modules as `torch-venv`. Built 2026-10-30 and verified
with jax 0.12.0.1 seeing all 4 A100s as `CudaDevice`s, `pip check`
clean, and `pytest -q` at 77 passed / 2 skipped (the torch-only Aurora
and SFNO tests). Build:

```sh
envs/build-jax.sh                                    # creates /fgs/$USER/causaltc/jax-venv
```

What it installs, in a single `pip install` so the resolver sees every
constraint together:

- `jax[cuda12]` — JAX plus its CUDA 12 runtime as pip wheels
  (`jax-cuda12-plugin`, `nvidia-*`).
- `weathernext`, from git at the commit of the `external/weathernext`
  clone the backends were written against (`external/` is gitignored, so
  a local path would not be reproducible). It pins `xarray<=2026.2.0`.
- tcinit with `[neuralgcm,eamxx,plotting,dev]`. NeuralGCM comes from
  PyPI; its git version pulls `dinosaur` from git, which collides with
  WeatherNext's `dinosaur-dycore` (same `dinosaur` import name).
- `gcsfs` (ERA5 and NeuralGCM checkpoints on `gs://`), `cfgrib`, `eccodes`.

### CUDA libraries and the cuda module

JAX uses the CUDA libraries from its pip wheels. If the `aue/cuda`
module's library paths on `LD_LIBRARY_PATH` get picked up first, JAX can
fail with cuDNN/cuBLAS/cuSPARSE version errors. If that happens, run
JAX with the CUDA module paths removed:

```sh
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v cuda | paste -sd:)
```

### GPU memory

JAX preallocates 75% of GPU memory on first use. On a shared node, or
when running a torch model in the same job, set
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

### Lock file

```sh
. /fgs/$USER/causaltc/jax-venv/bin/activate
envs/lock.sh jax            # writes envs/jax-cuda12.lock.txt
```

The script checks that `weathernext` and `gdm-xarray-jax` kept their git
sources. As with torch-venv, run `envs/install-hiccup.sh` after
`pip install -r` when rebuilding from the lock.

## Jupyter

Both build scripts install `jupyter` (JupyterLab, the classic notebook,
`ipykernel`) and register the env as a named kernel, `Python (torch-venv)`
or `Python (jax-venv)`. Open a demo notebook and pick the kernel for its
env (see the table at the top). The kernel spec is a small file under
`~/.local/share/jupyter/kernels/`.

To add Jupyter to an env built before this change, without rebuilding:

```sh
. /fgs/$USER/causaltc/torch-venv/bin/activate      # or jax-venv
pip install jupyter
python -m ipykernel install --user --name torch-venv --display-name "Python (torch-venv)"
```

Then regenerate that env's lock file with `envs/lock.sh`.

Run Jupyter on the GPU node and reach it through an SSH tunnel:

```sh
# on cee-compute044, env active
jupyter lab --no-browser --port 8888
# on your workstation
ssh -N -L 8888:localhost:8888 cee-compute044
```

Then open the `http://localhost:8888/...` URL that `jupyter lab` prints.
Run `proxy_on` before starting Jupyter if the notebooks need network
access (ARCO ERA5, checkpoint downloads).

## Checking an environment

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # torch-venv
python -c "import jax; print(jax.devices())"                                    # jax-venv: expect CudaDevice
pip check
pytest -q
```

`torch.cuda.is_available()` is `False` on login nodes without a GPU;
run the check on a compute node.
