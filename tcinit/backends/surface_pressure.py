"""Carry the vortex's MSLP change into a model's surface-pressure field.

``BogusVortex.apply`` blends MSLP toward the K&C surface pressure but has no
notion of surface pressure ``sp``. Backends whose model reads ``sp`` (HICCUP
builds E3SM ``PS`` from it; SFNO takes it as an input channel) use
:func:`follow_msl` so ``sp`` and ``msl`` stay consistent in the storm core.
"""

from __future__ import annotations

import xarray as xr

from tcinit.ideal_tc_vortex import BogusVortex


def follow_msl(
    before: xr.Dataset,
    after: xr.Dataset,
    bv: BogusVortex,
    *,
    sp_name: str = "sp",
    msl_name: str = "msl",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Set ``after[sp_name]`` from the MSLP change between ``before`` and ``after``.

    ``sp`` is multiplied by ``msl_after / msl_before``, which keeps each
    cell's sea-level reduction factor and, over the ocean (``sp`` ≈ ``msl``),
    gives the same pressure drop. Without ``msl``, ``sp`` gets K&C's
    replacement blend toward the vortex surface pressure directly. Returns
    ``after`` unchanged when ``before`` has no ``sp_name``.
    """
    if sp_name not in before.data_vars:
        return after
    sp = before[sp_name]
    if msl_name in before.data_vars and msl_name in after.data_vars:
        new_sp = sp * (after[msl_name] / before[msl_name])
    else:
        coords = {lat_name: bv.snapshot.box_lats, lon_name: bv.snapshot.box_lons}
        blend = xr.DataArray(bv.blend, dims=(lat_name, lon_name), coords=coords)
        p_bogus = xr.DataArray(
            bv.p_sfc_bogus_pa, dims=(lat_name, lon_name), coords=coords
        )
        new_sp = (1.0 - blend) * sp + blend * p_bogus
    after[sp_name] = new_sp.transpose(*sp.dims)
    return after
