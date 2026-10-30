# HPC virtual environments

The model backends need two separate environments, because PyTorch and
JAX each bundle their own CUDA 12 libraries and the pinned versions clash
when installed together.

| Environment | Build script | Backends |
|---|---|---|
| `torch-venv` | `build-torch.sh` | Aurora, SFNO (earth2studio), HICCUP, xarray |
| `jax-venv` | `build-jax.sh` | NeuralGCM, WeatherNext, HICCUP, xarray |

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
```

If scratch is ever purged, the venv is lost but the lock file in the
repo is not; that is the main reason to keep the lock file current.

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

Same host and modules as `torch-venv`. Build:

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
- tcinit with `[neuralgcm,hiccup,plotting,dev]`. NeuralGCM comes from
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
sources.

## Checking an environment

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # torch-venv
python -c "import jax; print(jax.devices())"                                    # jax-venv: expect CudaDevice
pip check
pytest -q
```

`torch.cuda.is_available()` is `False` on login nodes without a GPU;
run the check on a compute node.
