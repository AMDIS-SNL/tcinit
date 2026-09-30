"""Adapter for WeatherNext 2 / WeatherNext Cyclones (FGN) xarray inputs.

WN2 shares WN-Graph's xarray input signature (dims ``(batch, time, [level,]
lat, lon)``, short coord names, long-form variable names) and therefore
reuses nearly all of :mod:`tcinit.backends.weathernext_graph_backend`. The
one difference K&C cares about: WN2 adds ``100m_u_component_of_wind`` and
``100m_v_component_of_wind`` as surface fields. Kwon & Cheong's analytic
vortex only produces 10 m wind (via the PBL modifier); we leave 100 m wind
untouched by default.

Levels: WN2's operational checkpoint uses the HRES-25 subset; the Cyclones
and Mini checkpoints use WeatherBench-13. Both are proper subsets of ERA5-37
so nothing level-specific is required here — the vortex builds on whatever
level axis the input carries.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from tcinit.backends import weathernext_graph_backend as _wng
from tcinit.ideal_tc_vortex import BogusVortex

# WN2-only surface fields the K&C vortex does not synthesise. write_back
# leaves these untouched even if the user's caller accidentally includes
# them in the modified Dataset.
WN2_SKIP_VARS: tuple[str, ...] = (
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
)


def extract_snapshot_dataset(
    ds: xr.Dataset,
    *,
    box_center: Tuple[float, float],
    box_size: float,
    time_idx: int = -1,
    batch_idx: int = 0,
    lat_name: str = "lat",
    lon_name: str = "lon",
    level_name: str = "level",
) -> xr.Dataset:
    """Slice a lat/lon box from a WN2 inputs Dataset. See WN-Graph for details.

    Identical to :func:`tcinit.backends.weathernext_graph_backend.extract_snapshot_dataset`
    — WN2's extract set is a superset of what K&C reads, and the extra 100 m
    wind fields are already ignored by the WN long-form var map used inside
    :func:`extract_snapshot_dataset` (they're not listed in
    :data:`tcinit.naming.WEATHERNEXT_VAR_MAP`).
    """
    return _wng.extract_snapshot_dataset(
        ds,
        box_center=box_center,
        box_size=box_size,
        time_idx=time_idx,
        batch_idx=batch_idx,
        lat_name=lat_name,
        lon_name=lon_name,
        level_name=level_name,
    )


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
    """Blend the built vortex into a WN2-extracted canonical Dataset.

    Passthrough over :func:`tcinit.backends.weathernext_graph_backend.apply` —
    no WN2-specific behaviour is needed at the apply step since the extracted
    box shares WN-Graph's canonical form.
    """
    return _wng.apply(
        ds,
        bv,
        lat_name=lat_name,
        lon_name=lon_name,
        level_name=level_name,
        additive_z=additive_z,
        var_map=var_map,
    )


def write_back(
    ds: xr.Dataset,
    modified_ds: xr.Dataset,
    *,
    box_lats: np.ndarray,
    box_lons: np.ndarray,
    time_slots: Optional[Sequence[int]] = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
    level_name: str = "level",
    skip_vars: Sequence[str] = WN2_SKIP_VARS,
) -> xr.Dataset:
    """Paste a modified canonical box back into a WN2 Dataset.

    Extends :func:`tcinit.backends.weathernext_graph_backend.write_back` with
    a default ``skip_vars`` = :data:`WN2_SKIP_VARS` (100 m wind), so those
    fields never get overwritten even if the user's pipeline injects
    something into them upstream.
    """
    # Union caller-provided skips with the WN2 defaults so passing skip_vars
    # extends rather than replaces the 100m-wind guard.
    combined_skip = tuple(set(WN2_SKIP_VARS) | set(skip_vars))
    return _wng.write_back(
        ds,
        modified_ds,
        box_lats=box_lats,
        box_lons=box_lons,
        time_slots=time_slots,
        lat_name=lat_name,
        lon_name=lon_name,
        level_name=level_name,
        skip_vars=combined_skip,
    )
