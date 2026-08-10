"""Adapter between Microsoft Aurora ``Batch`` and tcinit's canonical xarray schema.

Two entry points:

- :func:`extract_snapshot_dataset` slices a lat/lon box out of a ``Batch`` at a
  chosen time index and returns a canonical ``xr.Dataset`` (dims:
  ``level``, ``latitude``, ``longitude``; canonical variable names).
- :func:`write_back` writes a modified canonical Dataset back into a
  ``Batch``'s ``surf_vars``/``atmos_vars`` tensors across *all* timesteps
  in the batch. The prototype's per-batch pattern used the same value at
  both timesteps, so we do too.

``torch`` and the ``aurora`` package are imported lazily inside the
functions so that ``tcinit.backends.aurora_backend`` can be imported in
environments where only the extras haven't been installed (as long as no
one calls these functions).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr

from tcinit.naming import ATMOS_VARS, SURF_VARS


def _to_numpy(tensor) -> np.ndarray:
    """Detach a torch tensor to a numpy array. Torch is imported lazily."""
    import torch  # noqa: F401

    return tensor.detach().cpu().numpy()


def _find_box_indices(global_coord: np.ndarray, target: np.ndarray, name: str) -> slice:
    """Locate ``target`` (a 1-D coordinate slice) inside ``global_coord``.

    Works for either ascending or descending global axes (Aurora's lat is
    descending 90 -> -90). Uses argmin(|diff|) instead of searchsorted, which
    would break on descending axes.
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
            f"{name}: batch coordinate does not match snapshot box "
            f"(start={start}). Antimeridian-crossing boxes are not yet supported."
        )
    return slice(start, stop)


def extract_snapshot_dataset(
    batch,
    *,
    box_center: Tuple[float, float],
    box_size: float,
    time_idx: int = 0,
) -> xr.Dataset:
    """Extract a lat/lon box out of an Aurora Batch as a canonical xr.Dataset.

    Args:
        batch: Aurora ``Batch`` with ``surf_vars``, ``atmos_vars``, and
            ``metadata.lat``/``.lon``/``.atmos_levels`` populated.
        box_center: ``(lat_center_deg, lon_center_deg)``. Longitude convention
            must match ``batch.metadata.lon`` (typically ``[0, 360)`` for Aurora).
        box_size: Half-width of the extracted box (deg). Extraction is
            inclusive on both sides.
        time_idx: Which time slot in the batch to sample (default 0). Aurora
            batches typically carry two timesteps; the storm state at either
            is a valid snapshot input.

    Returns:
        Dataset with dims ``(level, latitude, longitude)``, variables
        ``msl, 2t, 10u, 10v, t, u, v, q, z``, and coords matching Aurora's
        native ordering (lat descending; lon in whatever convention ``batch``
        uses).
    """
    lats_global = _to_numpy(batch.metadata.lat)
    lons_global = _to_numpy(batch.metadata.lon)
    atmos_levels = np.asarray(list(batch.metadata.atmos_levels), dtype=float)

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
            f"half-width {box_size} deg does not intersect batch grid."
        )
    lat_sl = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_sl = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)

    box_lats = lats_global[lat_sl]
    box_lons = lons_global[lon_sl]

    data_vars = {}

    def _read_surf(name: str) -> np.ndarray:
        return _to_numpy(batch.surf_vars[name][0, time_idx, lat_sl, lon_sl])

    def _read_atmos(name: str) -> np.ndarray:
        return _to_numpy(batch.atmos_vars[name][0, time_idx, :, lat_sl, lon_sl])

    for name in SURF_VARS:
        if name in batch.surf_vars:
            data_vars[name] = (("latitude", "longitude"), _read_surf(name))
    for name in ATMOS_VARS:
        if name in batch.atmos_vars:
            data_vars[name] = (
                ("level", "latitude", "longitude"),
                _read_atmos(name),
            )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": box_lats,
            "longitude": box_lons,
            "level": atmos_levels,
        },
    )


def write_back(
    batch,
    modified_ds: xr.Dataset,
    *,
    box_lats: np.ndarray,
    box_lons: np.ndarray,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
):
    """Blend a modified canonical Dataset back into an Aurora Batch in-place.

    Because :meth:`tcinit.vortex.BogusVortex.apply` already applied the cosine
    taper (blend=0 outside the storm mask), this is a straight overwrite of
    the box slice — no per-cell blending is needed here. The overwrite is
    replicated across every time slot in the batch, matching the prototype's
    behaviour of injecting the same vortex at both timesteps of a 2-step
    input.
    """
    import torch

    lats_global = _to_numpy(batch.metadata.lat)
    lons_global = _to_numpy(batch.metadata.lon)
    batch_levels = list(batch.metadata.atmos_levels)

    lat_sl = _find_box_indices(
        lats_global, np.asarray(box_lats, dtype=float), name="batch.metadata.lat"
    )
    lon_sl = _find_box_indices(
        lons_global, np.asarray(box_lons, dtype=float), name="batch.metadata.lon"
    )

    # ---- Surface fields ---------------------------------------------------
    for canon in SURF_VARS:
        if canon not in modified_ds.data_vars or canon not in batch.surf_vars:
            continue
        tensor = batch.surf_vars[canon]
        dtype = tensor.dtype
        device = tensor.device
        values = np.asarray(modified_ds[canon].values, dtype=np.float64)
        val_t = torch.as_tensor(values, dtype=dtype, device=device)
        n_times = tensor.shape[1]
        for t in range(n_times):
            tensor[0, t, lat_sl, lon_sl] = val_t

    # ---- Atmospheric fields ----------------------------------------------
    if level_name in modified_ds.dims:
        ds_levels = np.asarray(modified_ds[level_name].values, dtype=float)
        level_lookup = {int(lev): idx for idx, lev in enumerate(ds_levels)}
    else:
        level_lookup = {}

    for canon in ATMOS_VARS:
        if canon not in modified_ds.data_vars or canon not in batch.atmos_vars:
            continue
        tensor = batch.atmos_vars[canon]
        dtype = tensor.dtype
        device = tensor.device
        da_values = np.asarray(modified_ds[canon].values, dtype=np.float64)
        n_times = tensor.shape[1]
        for batch_k, batch_lev in enumerate(batch_levels):
            src_k = level_lookup.get(int(batch_lev))
            if src_k is None:
                continue
            slab = torch.as_tensor(da_values[src_k], dtype=dtype, device=device)
            for t in range(n_times):
                tensor[0, t, batch_k, lat_sl, lon_sl] = slab

    return batch
