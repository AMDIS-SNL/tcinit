"""HICCUP / ERA5 initial-condition Dataset adapter.

HICCUP consumes ERA5 atmospheric + surface netCDF files (from CDS) and
produces E3SM-ready initial conditions by horizontal + vertical remap plus
a surface T/P adjustment. The natural injection point for a bogus TC
vortex is the ERA5 input file, *before* HICCUP reads it — modify it, then
hand the modified file off to HICCUP unchanged.

ERA5's GRIB shortNames match tcinit's canonical variable names for the
core atmospheric fields (``t``, ``u``, ``v``, ``q``, ``z``) and MSLP
(``msl``). HICCUP's own ERA5 subclass only reads surface pressure
(``sp``), skin temperature (``skt``), and surface geopotential (``z``,
distinct from the atmospheric ``z``) from the surface file — it does NOT
read ``msl``, ``2t``, ``10u``, or ``10v``. HICCUP performs its own
surface rebalance in :mod:`hiccup.hiccup_state_adjustment`, so leaving
those surface fields alone is safe.

For an ERA5 atmospheric-file Dataset, this backend blends the vortex into
``t``, ``u``, ``v``, ``q``, ``z`` (and ``msl`` if it's present).
"""

from __future__ import annotations

from typing import Mapping, Optional

import xarray as xr

from tcinit.naming import ERA5_VAR_MAP
from tcinit.vortex import BogusVortex


def apply(
    ds: xr.Dataset,
    bv: BogusVortex,
    *,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
    additive_z: bool = True,
    var_map: Optional[Mapping[str, str]] = None,
) -> xr.Dataset:
    """Apply the built vortex to an ERA5-shape Dataset for HICCUP consumption."""
    active_map = dict(ERA5_VAR_MAP)
    if var_map is not None:
        active_map.update(var_map)
    return bv.apply(
        ds,
        var_map=active_map,
        lat_name=lat_name,
        lon_name=lon_name,
        level_name=level_name,
        additive_z=additive_z,
    )
