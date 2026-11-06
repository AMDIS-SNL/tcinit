"""Adapter between WeatherNext-Graph (GraphCast) xarray inputs and tcinit's
canonical xarray schema.

The WeatherNext-Graph predictor
(``weathernext/weathernext1_graph/graphcast.py``) takes ``inputs``,
``targets_template``, ``forcings`` all as ``xarray.Dataset``. Dims are
``(batch, time, [level,] lat, lon)`` with short-form coord names (``lat``,
``lon``) — different from Aurora/HICCUP's ``latitude``/``longitude`` — and
long-form CF-ish variable names (``temperature``, ``mean_sea_level_pressure``,
``10m_u_component_of_wind``, etc.). The input carries two time slots
(``input_duration="12h"`` at 6h step): typically ``t=-6h`` and ``t=0h``.

This module provides:

- :func:`extract_snapshot_dataset` — pick one ``(batch, time)`` slot, slice a
  lat/lon box, and rename ``lat``/``lon`` to canonical ``latitude``/
  ``longitude``. Long-form variable names are preserved so callers can build
  a :class:`~tcinit.snapshot.Snapshot` with
  ``var_map=tcinit.naming.WEATHERNEXT_VAR_MAP``.
- :func:`apply` — thin passthrough over ``BogusVortex.apply`` with the WN
  long-form var map applied.
- :func:`write_back` — paste the modified box back into a fresh copy of the
  full Dataset. Default writes into every time slot across every batch,
  matching Aurora's convention of injecting the same vortex at both input
  timesteps. Pass ``time_slots=(1,)`` to inject only into the current-time
  slot.

Variables the K&C vortex does not touch (``total_precipitation_6hr``,
``geopotential_at_surface``, ``land_sea_mask``,
``toa_incident_solar_radiation``, time-progress) are neither extracted nor
written back — they pass through untouched. ``vertical_velocity`` does the
same by default.

Omega blend (opt-in)
--------------------
K&C has no vertical velocity in the paper, but its inflow/outflow is
divergent, and GraphCast reads omega as an input. Left alone, the analysis
omega sits next to bogus convergence it does not match. With
``extract_snapshot_dataset(..., include_omega=True)`` and
``apply(..., blend_omega=True)``, ``vertical_velocity`` is replacement-blended
(same weight as ``u``/``v``) toward ``bv.omega_bogus_pa_s``: the omega that
K&C's axisymmetric radial wind implies through sigma-coordinate continuity
with a steady surface pressure. It is integrated analytically in radius and
sigma, not from the gridded winds: at 0.25 deg the centred-difference
divergence of the (analytically non-divergent) tangential wind near the RMW
is ~3e-4 1/s, which integrates to O(10) Pa/s of spurious omega.
:func:`write_back` writes ``vertical_velocity`` back whenever the modified
box carries it. :func:`omega_diagnostics` reports the gridded continuity
residual and the hydrostatic balance of the result.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from tcinit.diagnostics.continuity import gradient_latlon, omega_balance_report
from tcinit.naming import ATMOS_VARS, SURF_VARS, WEATHERNEXT_OMEGA, WEATHERNEXT_VAR_MAP
from tcinit.ideal_tc_vortex import BogusVortex


def _find_box_indices(global_coord: np.ndarray, target: np.ndarray, name: str) -> slice:
    """Locate ``target`` (a contiguous 1-D slice) inside ``global_coord``.

    Works for ascending or descending axes. WeatherNext uses ascending
    ``lat`` (-90 → 90) and ascending ``lon`` (0 → 360).
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
            f"{name}: dataset coordinate does not match snapshot box "
            f"(start={start}). Antimeridian-crossing boxes are not yet supported."
        )
    return slice(start, stop)


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
    include_omega: bool = False,
) -> xr.Dataset:
    """Slice a lat/lon box out of a WeatherNext-Graph Dataset into canonical form.

    Args:
        ds: WN-Graph inputs Dataset. Expected dims include some subset of
            ``batch``, ``time``, ``level``, ``lat``, ``lon``.
        box_center: ``(lat_deg, lon_deg)`` matching ``ds``'s longitude
            convention (WN uses ``[0, 360)``).
        box_size: half-width of the extracted box (deg), inclusive both sides.
        time_idx: which time slot to sample. Default ``-1`` (most-recent,
            typically ``t=0h`` for the standard 12h input duration).
        batch_idx: which batch slot to sample. Default 0.
        lat_name, lon_name, level_name: source-side coord names.
        include_omega: also keep ``vertical_velocity``, which
            :func:`apply` needs for ``blend_omega=True``.

    Returns:
        Dataset with dims ``(level, latitude, longitude)`` (or just
        ``(latitude, longitude)`` for surface-only var subsets), long-form
        variable names preserved, ``time`` and ``batch`` collapsed away.
        Feed to ``Snapshot.from_xarray(ds, var_map=WEATHERNEXT_VAR_MAP)``.
    """
    work = ds
    if "batch" in work.dims:
        work = work.isel(batch=batch_idx)
    if "time" in work.dims:
        work = work.isel(time=time_idx)

    lats_global = np.asarray(work[lat_name].values, dtype=float)
    lons_global = np.asarray(work[lon_name].values, dtype=float)

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
            f"half-width {box_size} deg does not intersect WN-Graph grid."
        )
    lat_sl = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_sl = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)

    # Slice the K&C-relevant subset only — leaves precip/vertical
    # velocity/statics/forcings behind (they don't need to be in the extracted
    # box and would just clutter the Snapshot).
    keep = [name for name in WEATHERNEXT_VAR_MAP.values() if name in work.data_vars]
    if include_omega and WEATHERNEXT_OMEGA in work.data_vars:
        keep.append(WEATHERNEXT_OMEGA)
    work = work[keep].isel({lat_name: lat_sl, lon_name: lon_sl})

    # Canonicalise coord names so downstream Snapshot/BogusVortex use their
    # defaults (``latitude``/``longitude``).
    renames = {}
    if lat_name != "latitude":
        renames[lat_name] = "latitude"
    if lon_name != "longitude":
        renames[lon_name] = "longitude"
    if renames:
        work = work.rename(renames)

    return work


def apply(
    ds: xr.Dataset,
    bv: BogusVortex,
    *,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
    blend_z: bool = True,
    var_map: Optional[Mapping[str, str]] = None,
    blend_omega: bool = False,
) -> xr.Dataset:
    """Apply the built vortex to a WN-Graph-extracted canonical Dataset.

    The extracted Dataset uses long-form variable names, so the default
    ``var_map`` is :data:`tcinit.naming.WEATHERNEXT_VAR_MAP`. Coord names
    default to ``latitude``/``longitude`` since :func:`extract_snapshot_dataset`
    renames them.

    ``blend_omega=True`` also blends ``vertical_velocity`` toward the K&C
    omega (see the module docstring); ``ds`` must then come from
    ``extract_snapshot_dataset(..., include_omega=True)``.
    """
    active_map = dict(WEATHERNEXT_VAR_MAP)
    if var_map is not None:
        active_map.update(var_map)
    if blend_omega:
        if WEATHERNEXT_OMEGA not in ds.data_vars:
            raise ValueError(
                f"apply(blend_omega=True): {WEATHERNEXT_OMEGA!r} is not in the box; "
                "extract it with extract_snapshot_dataset(..., include_omega=True)"
            )
        active_map["w"] = WEATHERNEXT_OMEGA
    return bv.apply(
        ds,
        var_map=active_map,
        lat_name=lat_name,
        lon_name=lon_name,
        level_name=level_name,
        blend_z=blend_z,
        blend_omega=blend_omega,
    )


def _box_arrays(ds, names, *, lat_name, lon_name, level_name):
    """Numpy ``(level, lat, lon)`` / ``(lat, lon)`` arrays for ``names`` in ``ds``."""
    out = []
    for name in names:
        da = ds[name]
        dims = (
            (level_name, lat_name, lon_name)
            if level_name in da.dims
            else (lat_name, lon_name)
        )
        out.append(np.asarray(da.transpose(*dims).values, dtype=float))
    return out


def _surface_omega(u10, v10, p_sfc, lats, lons):
    """Kinematic surface omega V_s . grad(p_s) [Pa/s] (steady p_s)."""
    dp_dx, dp_dy = gradient_latlon(p_sfc, lats, lons)
    return u10 * dp_dx + v10 * dp_dy


def omega_diagnostics(
    before: xr.Dataset,
    after: xr.Dataset,
    bv: Optional[BogusVortex] = None,
    *,
    var_map: Optional[Mapping[str, str]] = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
) -> dict:
    """Gridded continuity residual and hydrostatic balance of an omega-blended box.

    ``before`` is the extracted box (``include_omega=True``), ``after`` the
    output of ``apply(..., blend_omega=True)``. With ``bv``, statistics are
    restricted to ``bv.blend > 0``. See
    :func:`tcinit.diagnostics.continuity.omega_balance_report` for the keys.
    """
    m = dict(WEATHERNEXT_VAR_MAP)
    if var_map is not None:
        m.update(var_map)
    kw = dict(lat_name=lat_name, lon_name=lon_name, level_name=level_name)
    names = [m["u"], m["v"], m["t"], m["z"], m["q"], WEATHERNEXT_OMEGA]
    u_b, v_b, t_b, z_b, q_b, w_b = _box_arrays(before, names, **kw)
    u_a, v_a, t_a, z_a, q_a, w_a = _box_arrays(after, names, **kw)
    lats = np.asarray(before[lat_name].values, dtype=float)
    lons = np.asarray(before[lon_name].values, dtype=float)
    ps_b, u10_b, v10_b = _box_arrays(before, [m["msl"], m["10u"], m["10v"]], **kw)
    ps_a, u10_a, v10_a = _box_arrays(after, [m["msl"], m["10u"], m["10v"]], **kw)
    target = _surface_omega(u10_a, v10_a, ps_a, lats, lons) - _surface_omega(
        u10_b, v10_b, ps_b, lats, lons
    )
    return omega_balance_report(
        u_before=u_b,
        v_before=v_b,
        u_after=u_a,
        v_after=v_a,
        t_before=t_b,
        t_after=t_a,
        z_before=z_b,
        z_after=z_a,
        q_before=q_b,
        q_after=q_a,
        omega_before=w_b,
        omega_after=w_a,
        lats_deg=lats,
        lons_deg=lons,
        levels_hpa=np.asarray(before[level_name].values, dtype=float),
        p_sfc_after_pa=ps_a,
        omega_sfc_target=target,
        mask=None if bv is None else np.asarray(bv.blend) > 0,
    )


def _write_var(
    target: xr.DataArray,
    mod_da: xr.DataArray,
    *,
    lat_name: str,
    lon_name: str,
    level_name: str,
    lat_sl: slice,
    lon_sl: slice,
    time_slots: Sequence[int] | None,
    level_index_map: dict,
) -> xr.DataArray:
    """Overwrite a lat/lon box in ``target`` from ``mod_da``.

    Returns a new DataArray with the box overwritten across all batches and
    the specified time slots (all slots if ``time_slots`` is None). Broadcasts
    the modified box implicitly across leading dims.
    """
    dims = target.dims
    target_vals = np.asarray(target.values).copy()

    lat_axis = dims.index(lat_name)
    lon_axis = dims.index(lon_name)
    level_axis = dims.index(level_name) if level_name in dims else None
    time_axis = dims.index("time") if "time" in dims else None

    if time_axis is not None:
        if time_slots is None:
            ts: Sequence[int | None] = list(range(target.sizes["time"]))
        else:
            ts = [int(t) for t in time_slots]
    else:
        ts = [None]

    if level_axis is not None and level_name in mod_da.dims:
        mod_ordered = np.asarray(
            mod_da.transpose(level_name, lat_name, lon_name).values
        )
        mod_levels = np.asarray(mod_da[level_name].values, dtype=float)
        for t in ts:
            for k_mod, lev in enumerate(mod_levels):
                dst_k = level_index_map.get(float(lev))
                if dst_k is None:
                    continue
                idx = [slice(None)] * target_vals.ndim
                idx[lat_axis] = lat_sl
                idx[lon_axis] = lon_sl
                idx[level_axis] = dst_k
                if t is not None and time_axis is not None:
                    idx[time_axis] = t
                target_vals[tuple(idx)] = mod_ordered[k_mod].astype(
                    target_vals.dtype, copy=False
                )
    else:
        mod_ordered = np.asarray(mod_da.transpose(lat_name, lon_name).values)
        for t in ts:
            idx = [slice(None)] * target_vals.ndim
            idx[lat_axis] = lat_sl
            idx[lon_axis] = lon_sl
            if t is not None and time_axis is not None:
                idx[time_axis] = t
            target_vals[tuple(idx)] = mod_ordered.astype(target_vals.dtype, copy=False)

    return xr.DataArray(
        target_vals,
        dims=dims,
        coords=target.coords,
        attrs=target.attrs,
        name=target.name,
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
    skip_vars: Sequence[str] = (),
) -> xr.Dataset:
    """Paste a modified canonical box back into a fresh copy of ``ds``.

    Args:
        ds: original WN-Graph inputs Dataset (returned untouched).
        modified_ds: canonical box Dataset from :func:`apply`, with long-form
            variable names and canonical ``latitude``/``longitude`` coords.
        box_lats, box_lons: the box coordinate arrays returned by
            :func:`extract_snapshot_dataset` (used to locate the destination
            slice in the source ``ds``).
        time_slots: which ``time`` indices to overwrite. Default ``None``
            (= all time slots, matching the Aurora convention of injecting
            the same vortex at every input step).
        lat_name, lon_name, level_name: source-side coord names.
        skip_vars: long-form variable names to leave untouched even if
            present in ``modified_ds``. ``vertical_velocity`` is written
            when ``modified_ds`` has it (``include_omega=True``). Used by the WN2 backend to preserve
            100m wind fields the K&C vortex does not synthesise.

    Returns:
        New Dataset with the box overwritten. ``ds`` is not mutated.
    """
    out = ds.copy(deep=True)

    # modified_ds carries canonical latitude/longitude — rename back to WN
    # short forms so var-level writes align by dim name.
    renames = {}
    if lat_name != "latitude" and "latitude" in modified_ds.dims:
        renames["latitude"] = lat_name
    if lon_name != "longitude" and "longitude" in modified_ds.dims:
        renames["longitude"] = lon_name
    mod = modified_ds.rename(renames) if renames else modified_ds

    lats_global = np.asarray(ds[lat_name].values, dtype=float)
    lons_global = np.asarray(ds[lon_name].values, dtype=float)
    lat_sl = _find_box_indices(lats_global, np.asarray(box_lats, dtype=float), lat_name)
    lon_sl = _find_box_indices(lons_global, np.asarray(box_lons, dtype=float), lon_name)

    if level_name in ds.dims:
        ds_levels = np.asarray(ds[level_name].values, dtype=float)
        level_index_map = {float(lev): int(idx) for idx, lev in enumerate(ds_levels)}
    else:
        level_index_map = {}

    skip = set(skip_vars)

    natives = [WEATHERNEXT_VAR_MAP.get(c) for c in list(SURF_VARS) + list(ATMOS_VARS)]
    for native in natives + [WEATHERNEXT_OMEGA]:
        if native is None or native in skip:
            continue
        if native not in mod.data_vars or native not in out.data_vars:
            continue
        out[native] = _write_var(
            out[native],
            mod[native],
            lat_name=lat_name,
            lon_name=lon_name,
            level_name=level_name,
            lat_sl=lat_sl,
            lon_sl=lon_sl,
            time_slots=time_slots,
            level_index_map=level_index_map,
        )

    return out
