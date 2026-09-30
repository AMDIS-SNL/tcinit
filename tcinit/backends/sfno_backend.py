"""Adapter between earth2studio SFNO's (tensor, CoordSystem) and tcinit's
canonical xarray schema.

SFNO's native input is a torch.Tensor with shape
``(batch, time, lead_time, variable, lat, lon)`` alongside an
``earth2studio.utils.type.CoordSystem`` (an OrderedDict keyed by those six
names). All 73 fields are stacked on the single ``variable`` axis with
per-level names — e.g. ``t500`` for temperature at 500 hPa, ``u10m`` for the
10 m zonal wind. See ``earth2studio/models/px/sfno.py`` for the full list.

This module provides:

- :func:`extract_snapshot_dataset` — pull a lat/lon box out of ``(x, coords)``
  into a canonical Dataset (dims ``(level, latitude, longitude)``, canonical
  short vars ``msl, 2t, 10u, 10v, t, u, v, q, z``).
- :func:`apply` — thin passthrough over ``BogusVortex.apply`` (identity var_map).
- :func:`write_back` — paste a modified canonical Dataset back into the SFNO
  tensor across every ``(batch, time, lead_time)`` slot.

Untouched SFNO channels (``u100m``, ``v100m``, ``sp``, ``tcwv``) are neither
extracted nor written back — leaving them stale in the storm core is a mild
inconsistency that matches the Aurora prototype's convention of leaving
``tcwv`` alone. Add a bespoke reconciliation step later if downstream tests
show it matters.

Torch is imported lazily inside the tensor-handling helpers so this module
can be imported (and tested) without earth2studio/torch installed, as long
as the entry points are called with plain numpy arrays.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from tcinit.naming import (
    ATMOS_VARS,
    SFNO_LEVELS_HPA,
    SURF_VARS,
    sfno_channel_index,
)
from tcinit.vortex import BogusVortex


def _tensor_to_numpy(t) -> np.ndarray:
    """Convert a torch tensor (or numpy array) to a CPU numpy array.

    Torch is imported lazily. bfloat16 tensors are upcast to float32 first
    since numpy has no bfloat16 dtype.
    """
    if hasattr(t, "detach"):
        import torch

        t = t.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    return np.asarray(t)


def _find_box_indices(global_coord: np.ndarray, target: np.ndarray, name: str) -> slice:
    """Locate ``target`` (a 1-D contiguous slice) inside ``global_coord``.

    Works for ascending or descending axes. SFNO uses descending lat and
    ascending lon in ``[0, 360)`` — the same layout as Aurora, so this
    helper mirrors ``aurora_backend._find_box_indices``.
    """
    if global_coord.ndim != 1:
        raise ValueError(f"{name}: global coordinate must be 1-D")
    start = int(np.argmin(np.abs(global_coord - target[0])))
    stop = start + target.size
    if stop > global_coord.size:
        raise ValueError(
            f"{name}: target extends past global grid "
            f"(start={start}, target_size={target.size}, "
            f"global_size={global_coord.size})"
        )
    if not np.allclose(global_coord[start:stop], target, atol=1e-6):
        raise ValueError(
            f"{name}: tensor coordinate does not match snapshot box "
            f"(start={start}). Antimeridian-crossing boxes are not yet supported."
        )
    return slice(start, stop)


def extract_snapshot_dataset(
    x,
    coords: Mapping,
    *,
    box_center: Tuple[float, float],
    box_size: float,
    batch_idx: int = 0,
    time_idx: int = 0,
    lead_time_idx: int = 0,
) -> xr.Dataset:
    """Extract a lat/lon box from an SFNO ``(tensor, CoordSystem)`` pair.

    Args:
        x: SFNO input tensor (torch or numpy) with shape
            ``(batch, time, lead_time, variable, lat, lon)`` matching ``coords``.
        coords: earth2studio CoordSystem-like mapping with ``"variable"``,
            ``"lat"``, ``"lon"`` keys at minimum (and typically ``"batch"``,
            ``"time"``, ``"lead_time"`` as well).
        box_center: ``(lat_deg, lon_deg)`` with lon in the same convention as
            ``coords["lon"]`` (typically ``[0, 360)`` for SFNO).
        box_size: half-width of the extracted box (deg), inclusive both sides.
        batch_idx, time_idx, lead_time_idx: indices to sample along those dims
            (default 0 for a single-sample input).

    Returns:
        Dataset with dims ``(level, latitude, longitude)``, canonical short
        variable names, and levels equal to the subset of
        :data:`tcinit.naming.SFNO_LEVELS_HPA` present in the tensor across
        every atmospheric canonical. Untouched SFNO channels (100m wind,
        ``sp``, ``tcwv``) are skipped.
    """
    lats_global = np.asarray(coords["lat"], dtype=float)
    lons_global = np.asarray(coords["lon"], dtype=float)
    variables = np.asarray(coords["variable"])

    lat_c, lon_c = box_center
    lat_lo, lat_hi = lat_c - box_size, lat_c + box_size
    lon_lo, lon_hi = lon_c - box_size, lon_c + box_size

    lat_mask = (lats_global >= lat_lo) & (lats_global <= lat_hi)
    lon_mask = (lons_global >= lon_lo) & (lons_global <= lon_hi)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(
            f"extract_snapshot_dataset: box centred at {box_center} with "
            f"half-width {box_size} deg does not intersect SFNO grid."
        )
    lat_sl = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_sl = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)

    box_lats = lats_global[lat_sl]
    box_lons = lons_global[lon_sl]

    x_np = _tensor_to_numpy(x)
    if x_np.ndim != 6:
        raise ValueError(
            "extract_snapshot_dataset: expected 6-D SFNO tensor "
            "(batch, time, lead_time, variable, lat, lon); "
            f"got shape {x_np.shape}"
        )

    data_vars: dict = {}

    # Surface: single slab per canonical, when the channel is present.
    for canon in SURF_VARS:
        ch = sfno_channel_index(variables, canon, None)
        if ch is None:
            continue
        slab = x_np[batch_idx, time_idx, lead_time_idx, ch, lat_sl, lon_sl]
        data_vars[canon] = (("latitude", "longitude"), slab)

    # Atmospheric: per-canonical level list = every SFNO level for which the
    # channel is present. Common level axis = intersection across canonicals,
    # preserving SFNO_LEVELS_HPA order. For the standard 73-channel model,
    # this is exactly SFNO_LEVELS_HPA.
    per_var_levels: dict = {}
    for canon in ATMOS_VARS:
        levs = [
            lev
            for lev in SFNO_LEVELS_HPA
            if sfno_channel_index(variables, canon, lev) is not None
        ]
        if levs:
            per_var_levels[canon] = levs

    if per_var_levels:
        common_levels = [
            lev
            for lev in SFNO_LEVELS_HPA
            if all(lev in lvs for lvs in per_var_levels.values())
        ]
        for canon in per_var_levels:
            indices = np.array(
                [sfno_channel_index(variables, canon, lev) for lev in common_levels],
                dtype=int,
            )
            # Fancy indexing on the variable axis returns
            # (n_lev, n_lat_box, n_lon_box) after collapsing the leading dims.
            slab = x_np[batch_idx, time_idx, lead_time_idx, indices, lat_sl, lon_sl]
            data_vars[canon] = (("level", "latitude", "longitude"), slab)
        levels_coord = np.asarray(common_levels, dtype=float)
    else:
        levels_coord = np.asarray([], dtype=float)

    coords_out: dict = {
        "latitude": box_lats,
        "longitude": box_lons,
    }
    if levels_coord.size > 0:
        coords_out["level"] = levels_coord

    return xr.Dataset(data_vars=data_vars, coords=coords_out)


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
    """Apply the built vortex to an SFNO-extracted canonical Dataset.

    The extracted Dataset already uses canonical short names, so the default
    ``var_map`` is identity. Kept for API symmetry with the other backends.
    """
    return bv.apply(
        ds,
        var_map=var_map,
        lat_name=lat_name,
        lon_name=lon_name,
        level_name=level_name,
        additive_z=additive_z,
    )


def write_back(
    x,
    coords: Mapping,
    modified_ds: xr.Dataset,
    *,
    box_lats: np.ndarray,
    box_lons: np.ndarray,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
):
    """Blend a modified canonical Dataset back into the SFNO tensor in-place.

    Broadcasts the box slice across every ``(batch, time, lead_time)`` slot,
    matching :mod:`tcinit.backends.aurora_backend`'s convention of injecting
    the same vortex at every input time. Because
    :meth:`tcinit.vortex.BogusVortex.apply` already applied a cosine taper
    (blend=0 outside the storm mask), this is a straight overwrite of the
    box slice — no per-cell blending here.

    Accepts either a torch tensor or a numpy array as ``x``; the writer
    matches ``x.dtype`` / ``x.device`` for torch inputs.

    Returns:
        ``(x, coords)``. ``x`` is the same object that was passed in,
        modified in-place; the return is for chaining ergonomics.
    """
    is_torch = hasattr(x, "detach")
    if is_torch:
        import torch

    lats_global = np.asarray(coords["lat"], dtype=float)
    lons_global = np.asarray(coords["lon"], dtype=float)
    variables = np.asarray(coords["variable"])

    lat_sl = _find_box_indices(
        lats_global, np.asarray(box_lats, dtype=float), name="coords[lat]"
    )
    lon_sl = _find_box_indices(
        lons_global, np.asarray(box_lons, dtype=float), name="coords[lon]"
    )

    def _cast(vals: np.ndarray):
        if is_torch:
            return torch.as_tensor(vals, dtype=x.dtype, device=x.device)
        return vals.astype(x.dtype, copy=False)

    # Surface: one 2-D slab per canonical.
    for canon in SURF_VARS:
        if canon not in modified_ds.data_vars:
            continue
        ch = sfno_channel_index(variables, canon, None)
        if ch is None:
            continue
        vals = np.asarray(modified_ds[canon].values, dtype=np.float64)
        # ``x[..., ch, lat_sl, lon_sl]`` broadcasts across batch/time/lead_time.
        x[..., ch, lat_sl, lon_sl] = _cast(vals)

    # Atmospheric: per-level write, only for levels present in both the
    # tensor's variable set and the modified Dataset.
    if level_name in modified_ds.dims:
        ds_levels = np.asarray(modified_ds[level_name].values, dtype=float)
    else:
        ds_levels = np.array([], dtype=float)

    for canon in ATMOS_VARS:
        if canon not in modified_ds.data_vars:
            continue
        # Normalize dim order so positional writes are unambiguous, regardless
        # of whatever ordering xarray broadcasts produced.
        da = modified_ds[canon].transpose(level_name, lat_name, lon_name)
        arr = np.asarray(da.values, dtype=np.float64)
        for k_ds, lev in enumerate(ds_levels):
            ch = sfno_channel_index(variables, canon, int(lev))
            if ch is None:
                continue
            x[..., ch, lat_sl, lon_sl] = _cast(arr[k_ds])

    return x, coords
