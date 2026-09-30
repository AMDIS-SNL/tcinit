"""Generic canonical-xarray adapter — thin passthrough over BogusVortex.apply.

For an already-canonical Dataset (variables ``msl``, ``2t``, ``10u``, ``10v``,
``t``, ``u``, ``v``, ``q``, ``z``), this is equivalent to calling
``bv.apply(ds)`` directly. It exists for symmetry with the model-specific
backends and to give users a stable one-liner entry point that survives
future refactors of the vortex apply signature.
"""

from __future__ import annotations

from typing import Mapping, Optional

import xarray as xr

from tcinit.ideal_tc_vortex import BogusVortex


def apply(
    ds: xr.Dataset,
    bv: BogusVortex,
    *,
    var_map: Optional[Mapping[str, str]] = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
    additive_z: bool = True,
) -> xr.Dataset:
    """Apply the built vortex to ``ds`` and return a new Dataset."""
    return bv.apply(
        ds,
        var_map=var_map,
        lat_name=lat_name,
        lon_name=lon_name,
        level_name=level_name,
        additive_z=additive_z,
    )
