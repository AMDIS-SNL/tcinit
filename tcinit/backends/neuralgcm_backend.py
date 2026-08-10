"""NeuralGCM initial-condition Dataset adapter.

NeuralGCM inputs are regridded ERA5 stored on a Gaussian grid with long-form
CF-ish variable names (``temperature``, ``u_component_of_wind``,
``v_component_of_wind``, ``specific_humidity``, ``geopotential``, plus
``specific_cloud_ice_water_content`` / ``specific_cloud_liquid_water_content``
which the K&C vortex does not touch). No MSLP / 2 m T / 10 m wind — the
surface state is carried only as forcing (SST, sea-ice), not as a level.

The regridded Dataset uses ``latitude`` / ``longitude`` / ``level`` coord
names, and its data variables trail as ``(level, longitude, latitude)`` —
dim order is intentional (dinosaur's spherical harmonic layout). Because
``BogusVortex.apply`` blends by xarray broadcasting (by dim NAME), the swap
is handled transparently.

Because NeuralGCM has no surface fields to inject and its ``geopotential``
is diagnostic (hydrostatically integrated from ``temperature``), we default
``additive_z=False`` — see the design-doc risks section.
"""

from __future__ import annotations

from typing import Optional

import xarray as xr

from tcinit.naming import NEURALGCM_VAR_MAP
from tcinit.vortex import BogusVortex


def apply(
    ds: xr.Dataset,
    bv: BogusVortex,
    *,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
    additive_z: bool = False,
    var_map: Optional[dict] = None,
) -> xr.Dataset:
    """Apply the built vortex to a NeuralGCM-shape Dataset.

    Uses :data:`tcinit.naming.NEURALGCM_VAR_MAP` by default, translating
    canonical ``t``/``u``/``v``/``q``/``z`` to NeuralGCM's long-form names.
    Surface variables are silently skipped (they're not present in ``ds``).
    Pass ``var_map`` to override / extend the default map (e.g. if you keep
    NeuralGCM's cloud-water variables around and want them left untouched).
    """
    active_map = dict(NEURALGCM_VAR_MAP)
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
