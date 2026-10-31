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

NeuralGCM has no surface-pressure input: its encoder diagnoses surface
pressure from ``geopotential``, as the pressure where ``geopotential``
equals the surface geopotential (``orography · g``), linearly interpolated
in pressure (dinosaur's ``vertical_interpolation.get_surface_pressure``).
So the vortex's surface low has to be written into ``geopotential``.
:func:`apply` does this by default (``match_surface_pressure=True``): after
K&C's warm-core T anomaly is added, each column's geopotential is updated
hydrostatically for the T change and then shifted so the diagnosed surface
pressure equals K&C's replacement blend toward the vortex surface pressure,
the same target the surface-field backends apply to MSLP. K&C's own
geopotential anomaly (``additive_z=True``) is not used: it is measured from
the environment, so adding it to an analysis that already holds a weaker
low double-counts the pressure drop.

Two container shapes are supported:

- **xarray Dataset**: use :func:`apply`. This is the natural target for a
  Dataset that came out of ``model.data_to_xarray(...)`` or was assembled by
  hand (e.g. as the pre-``inputs_from_xarray`` input, per the NeuralGCM
  data-preparation tutorial).
- **inputs dict** (``dict[str, np.ndarray]`` with values shaped
  ``(level, longitude, latitude)``, as returned by
  ``PressureLevelModel.inputs_from_xarray``): use
  :func:`extract_snapshot_dataset` to pull a box out as a canonical Dataset,
  build/apply the vortex on that box, then :func:`write_back` to paste the
  modified box back into a fresh copy of the dict. This mirrors the
  extract/write_back pattern of :mod:`tcinit.backends.aurora_backend` and
  lets the vortex be injected between ``inputs_from_xarray`` and
  ``model.encode``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from tcinit.constants import R_D
from tcinit.naming import NEURALGCM_VAR_MAP
from tcinit.ideal_tc_vortex import BogusVortex

# ---------------------------------------------------------------------------
# xarray-native path (unchanged public API)
# ---------------------------------------------------------------------------


def apply(
    ds: xr.Dataset,
    bv: BogusVortex,
    *,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
    additive_z: bool = False,
    match_surface_pressure: bool = True,
    surface_geopotential: Optional[np.ndarray] = None,
    var_map: Optional[dict] = None,
) -> xr.Dataset:
    """Apply the built vortex to a NeuralGCM-shape Dataset.

    Uses :data:`tcinit.naming.NEURALGCM_VAR_MAP` by default, translating
    canonical ``t``/``u``/``v``/``q``/``z`` to NeuralGCM's long-form names.
    Surface variables are silently skipped (they're not present in ``ds``).
    Pass ``var_map`` to override / extend the default map (e.g. if you keep
    NeuralGCM's cloud-water variables around and want them left untouched).

    Args:
        match_surface_pressure: rewrite ``geopotential`` so the surface
            pressure NeuralGCM's encoder diagnoses from it equals the K&C
            target (see the module docstring). ``False`` leaves
            ``geopotential`` as K&C's ``additive_z`` setting leaves it, which
            for the default ``additive_z=False`` means no surface low.
        surface_geopotential: ``orography · g`` (m²/s²) on the box,
            shaped ``(latitude, longitude)`` in the box's coordinate order.
            Default zero, i.e. sea level, which is right for an ocean storm.
            NeuralGCM's encoder uses its own spectrally filtered orography,
            which differs from the raw ``geopotential_at_surface`` near
            coasts; a 10 m difference moves the diagnosed surface pressure
            by about 1 hPa.
    """
    if match_surface_pressure and additive_z:
        raise ValueError(
            "neuralgcm_backend.apply: match_surface_pressure=True replaces "
            "K&C's geopotential anomaly; pass additive_z=False."
        )
    active_map = dict(NEURALGCM_VAR_MAP)
    if var_map is not None:
        active_map.update(var_map)
    out = bv.apply(
        ds,
        var_map=active_map,
        lat_name=lat_name,
        lon_name=lon_name,
        level_name=level_name,
        additive_z=additive_z,
    )
    z_name = active_map.get("z")
    t_name = active_map.get("t")
    if match_surface_pressure and z_name in ds and t_name in ds:
        dims = (level_name, lat_name, lon_name)
        z_new = _match_surface_pressure(
            z=ds[z_name].transpose(*dims).values,
            t_before=ds[t_name].transpose(*dims).values,
            t_after=out[t_name].transpose(*dims).values,
            levels_pa=np.asarray(ds[level_name].values, dtype=float) * 100.0,
            blend=bv.blend,
            p_sfc_target_pa=bv.p_sfc_bogus_pa,
            surface_geopotential=surface_geopotential,
        )
        out[z_name] = xr.DataArray(
            z_new, dims=dims, coords={d: ds[d] for d in dims}
        ).transpose(*ds[z_name].dims)
    return out


def _crossing_pressure(rel: np.ndarray, p: np.ndarray) -> float:
    """Pressure where ``rel`` (increasing with ``p``) crosses zero.

    Matches dinosaur's ``get_surface_pressure``
    (``linear_interp_with_linear_extrap``): linear in pressure inside the
    level range, and extrapolated from the bottom two levels for columns
    whose surface lies below the deepest level.
    """
    if rel[-1] >= 0.0:
        return float(np.interp(0.0, rel, p))
    slope = (rel[-1] - rel[-2]) / (p[-1] - p[-2])
    if not slope > 0.0:
        return float("nan")
    return float(p[-1] - rel[-1] / slope)


def _value_at(f: np.ndarray, p: np.ndarray, p_target: float) -> float:
    """Linear interpolation of ``f(p)``, extrapolated past the deepest level."""
    if p_target <= p[-1]:
        return float(np.interp(p_target, p, f))
    slope = (f[-1] - f[-2]) / (p[-1] - p[-2])
    return float(f[-1] + slope * (p_target - p[-1]))


def _match_surface_pressure(
    *,
    z: np.ndarray,
    t_before: np.ndarray,
    t_after: np.ndarray,
    levels_pa: np.ndarray,
    blend: np.ndarray,
    p_sfc_target_pa: np.ndarray,
    surface_geopotential: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Geopotential whose diagnosed surface pressure hits the K&C blend.

    All 3-D arrays are ``(level, lat, lon)``; 2-D arrays ``(lat, lon)``.
    Per column:

    1. ``p_ctrl``: the surface pressure the unmodified ``z`` implies.
    2. ``p_target = (1 - blend) · p_ctrl + blend · p_sfc_target``.
    3. ``w(p) = z(p) + R_d ∫_p^{p_bottom} ΔT d(ln p')``: ``z`` updated
       hydrostatically for the temperature change, so the warm core raises
       the upper levels relative to the surface.
    4. ``z_new = w + (Φ_s − w(p_target))``, so ``z_new(p_target) = Φ_s``.

    Where ``blend`` is zero, ``ΔT`` is zero and ``p_target = p_ctrl``, so the
    column comes back unchanged. Columns where no surface pressure can be
    diagnosed (``z`` not decreasing with pressure) are left unchanged.
    """
    order = np.argsort(levels_pa)  # top of atmosphere first
    p = levels_pa[order]
    z_o = np.asarray(z, dtype=float)[order]
    dT = (np.asarray(t_after, dtype=float) - np.asarray(t_before, dtype=float))[order]
    phi_s = (
        np.zeros(z_o.shape[1:])
        if surface_geopotential is None
        else np.asarray(surface_geopotential, dtype=float)
    )

    # Thickness change from each level down to the deepest one (trapezoid
    # in ln p), accumulated upward from the bottom.
    layer = 0.5 * (dT[1:] + dT[:-1]) * np.diff(np.log(p))[:, None, None]
    below = np.concatenate(
        [np.cumsum(layer[::-1], axis=0)[::-1], np.zeros((1,) + layer.shape[1:])]
    )
    w = z_o + R_D * below

    z_new = z_o.copy()
    n_lat, n_lon = z_o.shape[1:]
    for i in range(n_lat):
        for j in range(n_lon):
            b = float(blend[i, j])
            if b == 0.0 and not dT[:, i, j].any():
                continue  # untouched column: keep it bit-identical
            p_ctrl = _crossing_pressure(phi_s[i, j] - z_o[:, i, j], p)
            if not np.isfinite(p_ctrl):
                continue
            p_target = (1.0 - b) * p_ctrl + b * float(p_sfc_target_pa[i, j])
            shift = phi_s[i, j] - _value_at(w[:, i, j], p, p_target)
            z_new[:, i, j] = w[:, i, j] + shift

    out = np.empty_like(z_new)
    out[order] = z_new
    return out


# ---------------------------------------------------------------------------
# Inputs-dict path — extract / write_back mirroring aurora_backend
# ---------------------------------------------------------------------------


def _find_box_indices(global_coord: np.ndarray, target: np.ndarray, name: str) -> slice:
    """Locate ``target`` (a contiguous coordinate slice) inside ``global_coord``.

    Works for ascending or descending global axes. NeuralGCM's Gaussian-grid
    ``latitude`` is ascending; ``longitude`` starts at 0 and is ascending.
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
    if not np.allclose(global_coord[start:stop], target, atol=1e-4):
        raise ValueError(
            f"{name}: inputs coordinate does not match snapshot box "
            f"(start={start}). Antimeridian-crossing boxes are not yet supported."
        )
    return slice(start, stop)


def axes_from_model(model, inputs: Mapping[str, np.ndarray]):
    """Return ``(lats, lons, levels)`` for a NeuralGCM ``PressureLevelModel``.

    Convenience helper — round-trips ``inputs`` through
    ``model.data_to_xarray(inputs, times=None)`` and reads the coord arrays.
    If you already have these axes (e.g. from your own model config), pass
    them directly to :func:`extract_snapshot_dataset` and skip this.
    """
    ds = model.data_to_xarray(inputs, times=None)
    return (
        np.asarray(ds["latitude"].values, dtype=float),
        np.asarray(ds["longitude"].values, dtype=float),
        np.asarray(ds["level"].values, dtype=float),
    )


def extract_snapshot_dataset(
    inputs: Mapping[str, np.ndarray],
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    levels: np.ndarray,
    box_center: Tuple[float, float],
    box_size: float,
) -> xr.Dataset:
    """Slice a lat/lon box out of a NeuralGCM inputs dict into a canonical Dataset.

    Args:
        inputs: dict from ``PressureLevelModel.inputs_from_xarray`` (or
            hand-built with the same shape). Variable values must be arrays
            with shape ``(level, longitude, latitude)`` keyed by long-form
            names (``temperature``, ``u_component_of_wind``, etc.). Extra
            entries (e.g. ``sim_time`` scalars, ``specific_cloud_*`` fields)
            are silently skipped.
        lats: 1-D global latitude coordinate (ascending or descending). For
            NeuralGCM's Gaussian grid, ascending from south to north.
        lons: 1-D global longitude coordinate, matching ``inputs`` layout
            (typically ``[0, 360)``).
        levels: 1-D pressure levels (hPa) matching the ``level`` axis of the
            ``inputs`` arrays. For NeuralGCM this is the ERA5-37 set.
        box_center: ``(lat_center_deg, lon_center_deg)``. Longitude convention
            must match ``lons``.
        box_size: Half-width of the extracted box (deg). Inclusive both sides.

    Returns:
        Dataset with dims ``(level, longitude, latitude)``, long-form
        variable names for the K&C-relevant subset present in ``inputs``,
        and coords matching the input arrays' native ordering. Feed directly
        to ``Snapshot.from_xarray(ds, var_map=NEURALGCM_VAR_MAP)`` or to
        :func:`apply`.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    levels = np.asarray(levels, dtype=float)

    lat_c, lon_c = box_center
    lat_lo, lat_hi = lat_c - box_size, lat_c + box_size
    lon_lo, lon_hi = lon_c - box_size, lon_c + box_size

    lat_mask = (lats >= lat_lo) & (lats <= lat_hi)
    lon_mask = (lons >= lon_lo) & (lons <= lon_hi)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(
            f"extract_snapshot_dataset: box centred at {box_center} with "
            f"half-width {box_size} deg does not intersect NeuralGCM grid."
        )
    lat_sl = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_sl = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)

    box_lats = lats[lat_sl]
    box_lons = lons[lon_sl]

    data_vars = {}
    for canon, native in NEURALGCM_VAR_MAP.items():
        del canon
        if native not in inputs:
            continue
        arr = np.asarray(inputs[native])
        if arr.ndim != 3:
            # Skip 0-D sim_time, 4-D unexpected shapes, or missing-level
            # forcing arrays (those go through model.forcings_from_xarray).
            continue
        if arr.shape[0] != levels.size:
            raise ValueError(
                f"extract_snapshot_dataset: variable {native!r} has "
                f"level dim of size {arr.shape[0]}, does not match "
                f"levels arg of size {levels.size}."
            )
        if arr.shape[1] != lons.size or arr.shape[2] != lats.size:
            raise ValueError(
                f"extract_snapshot_dataset: variable {native!r} has "
                f"shape {arr.shape}, expected "
                f"({levels.size}, {lons.size}, {lats.size})."
            )
        data_vars[native] = (
            ("level", "longitude", "latitude"),
            arr[:, lon_sl, lat_sl],
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": box_lats,
            "longitude": box_lons,
            "level": levels,
        },
    )


def write_back(
    inputs: Mapping[str, np.ndarray],
    modified_ds: xr.Dataset,
    *,
    lats: np.ndarray,
    lons: np.ndarray,
    levels: np.ndarray,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
) -> dict:
    """Paste a modified canonical box back into a copy of a NeuralGCM inputs dict.

    Returns a new dict (does not mutate ``inputs``). Non-array values
    (e.g. ``sim_time`` scalars) and variables absent from ``modified_ds``
    are carried through untouched.

    Because :meth:`tcinit.ideal_tc_vortex.BogusVortex.apply` already applied the
    cosine taper (blend=0 outside the storm mask), this is a straight
    overwrite of the box slice — no per-cell blending is done here.

    Args:
        inputs: original dict from ``PressureLevelModel.inputs_from_xarray``.
        modified_ds: canonical box Dataset returned by :func:`apply` (or by
            ``bv.apply(...)`` directly), with long-form variable names.
        lats, lons, levels: full-globe coordinate arrays for ``inputs``
            (same values passed to :func:`extract_snapshot_dataset`).
        lat_name, lon_name, level_name: coord names in ``modified_ds``.
    """
    box_lats = np.asarray(modified_ds[lat_name].values, dtype=float)
    box_lons = np.asarray(modified_ds[lon_name].values, dtype=float)
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    levels_arr = np.asarray(levels, dtype=float)

    lat_sl = _find_box_indices(lats_arr, box_lats, "lats")
    lon_sl = _find_box_indices(lons_arr, box_lons, "lons")

    # Level-to-index map for the arrays in `inputs`.
    level_lookup = {float(lev): idx for idx, lev in enumerate(levels_arr)}
    if level_name in modified_ds.dims:
        ds_levels = np.asarray(modified_ds[level_name].values, dtype=float)
    else:
        ds_levels = np.array([], dtype=float)

    out: dict = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v

    for canon, native in NEURALGCM_VAR_MAP.items():
        del canon
        if native not in modified_ds.data_vars or native not in out:
            continue
        arr = out[native]
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            continue
        # Normalize modified_ds var to (level, longitude, latitude) layout so
        # positional writes match the inputs array's convention regardless of
        # whatever dim order xarray broadcast may have produced.
        da_vals = np.asarray(
            modified_ds[native].transpose(level_name, lon_name, lat_name).values,
            dtype=arr.dtype,
        )
        for src_k, lev in enumerate(ds_levels):
            dst_k = level_lookup.get(float(lev))
            if dst_k is None:
                continue
            arr[dst_k, lon_sl, lat_sl] = da_vals[src_k]

    return out
