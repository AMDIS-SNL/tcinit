"""Shims to unpickle CausalTC `SingleAuroraDataHistory` rollouts.

CausalTC pickles reference `aurora.*` and `torch` at the module level
through import side effects, even though the actual pickled payload is
pure numpy. In HPC environments where those packages are installed,
nothing extra is needed. In the container this project also runs in,
neither is present — importing `causaltc.data_collection.aurora_snapshot`
to materialise the `AuroraSnapshot` class fails before `pickle.load`
can even start.

`install_causaltc_shim` registers lightweight stand-ins for `aurora`
and `torch` in `sys.modules` (only when the real packages are not
available) and adds the vendored CausalTC source to `sys.path`.
`load_causaltc_pickle` is a convenience that installs the shim and
calls `pickle.load`.
"""

from __future__ import annotations

import importlib
import pickle
import sys
import types
from pathlib import Path
from typing import Any

_DEFAULT_CAUSALTC_SRC = (
    Path(__file__).resolve().parent.parent
    / "external"
    / "causalcyclogenesis-pb-physics-bogus"
)

_AURORA_NAMES = ("Aurora", "AuroraHighRes", "Batch", "Metadata", "rollout")


def _ensure_stub(name: str, attrs: tuple[str, ...] = ()) -> None:
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    stub = types.ModuleType(name)
    for attr in attrs:
        setattr(stub, attr, type(attr, (), {}))
    sys.modules[name] = stub


def install_causaltc_shim(causaltc_src: Path | str | None = None) -> None:
    """Make `causaltc.data_collection.aurora_snapshot` importable here.

    Idempotent and harmless: if `aurora` / `torch` are already importable
    (e.g. running on HPC), they are left alone. If `causaltc` is already
    importable, `sys.path` is not touched.

    Args:
        causaltc_src: Path to the CausalTC source tree (the directory
            that contains the top-level `causaltc/` package). Defaults
            to `<repo>/external/causalcyclogenesis-pb-physics-bogus`.
    """
    _ensure_stub("aurora", _AURORA_NAMES)
    _ensure_stub("torch")

    try:
        importlib.import_module("causaltc")
        return
    except ImportError:
        pass
    src = Path(causaltc_src) if causaltc_src is not None else _DEFAULT_CAUSALTC_SRC
    src = src.resolve()
    if not (src / "causaltc").is_dir():
        raise FileNotFoundError(
            f"CausalTC source not found at {src}; pass causaltc_src= explicitly"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_causaltc_pickle(
    path: Path | str, *, causaltc_src: Path | str | None = None
) -> Any:
    """Install the shim and `pickle.load` the file at `path`."""
    install_causaltc_shim(causaltc_src=causaltc_src)
    with open(path, "rb") as fh:
        return pickle.load(fh)
