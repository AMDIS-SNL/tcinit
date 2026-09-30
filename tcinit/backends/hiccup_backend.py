"""HICCUP / ERA5 initial-condition adapter.

HICCUP (``external/hiccup``) consumes ERA5 netCDF files from the Copernicus
CDS and produces E3SM-ready initial conditions via horizontal/vertical remap
plus a surface T/P adjustment. The natural TC-bogus injection point is the
ERA5 input file, *before* HICCUP reads it — modify the netCDF, then hand it
to HICCUP unchanged.

Two entry points are provided:

- **File-based** — :func:`extract_snapshot_dataset` opens one or more ERA5
  netCDF files (atmospheric + surface), auto-detects the CDS-vs-legacy coord
  naming (``pressure_level`` vs ``level``; ``valid_time`` vs ``time``) and
  variable naming (``t2m`` vs ``2t``, ``u10`` vs ``10u``, ``v10`` vs ``10v``),
  and returns a canonical box Dataset for the Snapshot/BogusVortex pipeline.
  :func:`write_back` writes a modified box back into a copy of the source
  netCDF, preserving every field HICCUP later reads.

- **Xarray passthrough** — :func:`apply` blends a built vortex into an
  already-canonical ``xr.Dataset``. Useful when the user has assembled a
  Dataset in memory instead of loading from disk.

HICCUP-relevant variable subsets:

- Atmospheric (K&C touches): ``t, u, v, q, z``. HICCUP-only vars that pass
  through untouched: ``ciwc, clwc, o3`` (plus ancillaries).
- Surface (K&C touches, when present): ``msl, 2t, 10u, 10v``. HICCUP's ERA5
  subclass only reads ``sp, skt, z`` from the surface file — leaving those
  and the other surface fields (``sst, siconc, sd, stl1..4, tsn``) alone is
  safe. Surface ``z`` (surface geopotential) is a 2-D field, distinct from
  atmospheric ``z`` (geopotential on pressure levels) which is 3-D; the
  file-based writer distinguishes by dim structure so surface ``z`` is
  never accidentally overwritten with a 3-D block.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from tcinit.naming import (
    ATMOS_VARS,
    ERA5_NETCDF_NAMES,
    ERA5_VAR_MAP,
    SURF_VARS,
)
from tcinit.ideal_tc_vortex import BogusVortex

# Coord names that ERA5 has used across CDS versions. First match wins on
# auto-detect. Canonical (returned by extract) is always ``level``/``time``.
_LEVEL_ALIASES: tuple[str, ...] = ("level", "pressure_level")
_TIME_ALIASES: tuple[str, ...] = ("time", "valid_time")


def _detect_dim_name(ds: xr.Dataset, aliases: Sequence[str]) -> str | None:
    """Return the first alias present as a dim in ``ds``, else None."""
    for name in aliases:
        if name in ds.dims:
            return name
    return None


def _detect_var_name(
    ds: xr.Dataset, aliases: Sequence[str], *, expect_level: bool | None = None
) -> str | None:
    """Return the first alias present as a data var in ``ds``, else None.

    ``expect_level`` disambiguates cases like ERA5 ``z`` (which lives in both
    the atm and sfc files with different rank). Pass ``True`` when looking
    for the atmospheric field (must have a level dim), ``False`` for the
    surface field (must not), or ``None`` to accept either.
    """
    level_dim = _detect_dim_name(ds, _LEVEL_ALIASES)
    for name in aliases:
        if name not in ds.data_vars:
            continue
        if expect_level is None:
            return name
        has_level = level_dim is not None and level_dim in ds[name].dims
        if has_level == expect_level:
            return name
    return None


def _canonicalise(ds: xr.Dataset) -> tuple[xr.Dataset, dict[str, str]]:
    """Rename ERA5 dim / var aliases to tcinit's canonical short names.

    Returns ``(ds_renamed, name_map)`` where ``name_map`` records the
    original names so :func:`write_back` can invert the rename when
    saving to the source file schema.
    """
    renames: dict[str, str] = {}
    level_name = _detect_dim_name(ds, _LEVEL_ALIASES)
    if level_name is not None and level_name != "level":
        renames[level_name] = "level"
    time_name = _detect_dim_name(ds, _TIME_ALIASES)
    if time_name is not None and time_name != "time":
        renames[time_name] = "time"

    # Variable-name canonicalisation for the K&C-touched subset. Atmospheric
    # ``z`` and ``t/u/v/q`` already share their canonical names in every ERA5
    # variant we've seen; surface ``2t``/``10u``/``10v`` are the ones that
    # differ across CDS versions.
    var_source_names: dict[str, str] = {}
    for canon in SURF_VARS + ATMOS_VARS:
        aliases = ERA5_NETCDF_NAMES.get(canon, (canon,))
        # For canonical ``z`` and ``t`` (which appear as both surface and
        # atmospheric fields), disambiguate by dim rank.
        expect_level: bool | None
        if canon in ATMOS_VARS:
            expect_level = True
        elif canon == "msl":
            expect_level = False
        else:
            expect_level = False
        native = _detect_var_name(ds, aliases, expect_level=expect_level)
        if native is None:
            continue
        var_source_names[canon] = native
        if native != canon:
            renames[native] = canon

    return ds.rename(renames) if renames else ds, var_source_names


def _find_box_indices(global_coord: np.ndarray, target: np.ndarray, name: str) -> slice:
    """Locate ``target`` (a contiguous 1-D slice) inside ``global_coord``.

    Works for ascending or descending axes. ERA5's ``latitude`` is descending
    (90 → -90); ``longitude`` is ascending in ``[0, 360)``.
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
            f"{name}: file coordinate does not match snapshot box "
            f"(start={start}). Antimeridian-crossing boxes are not yet supported."
        )
    return slice(start, stop)


# ---------------------------------------------------------------------------
# apply() — xarray passthrough (unchanged public API)
# ---------------------------------------------------------------------------


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
    """Apply the built vortex to an already-canonical ERA5-shape Dataset.

    Uses :data:`tcinit.naming.ERA5_VAR_MAP` (identity) by default. If the
    input Dataset uses CDS-netCDF variable names (``t2m``/``u10``/``v10``)
    rather than GRIB shortnames, either pass ``var_map`` explicitly or
    prefer :func:`extract_snapshot_dataset` / :func:`write_back`, which
    handle the naming disambiguation automatically.
    """
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


# ---------------------------------------------------------------------------
# File-based extract / write_back
# ---------------------------------------------------------------------------


def extract_snapshot_dataset(
    input_files: str | Sequence[str],
    *,
    box_center: Tuple[float, float],
    box_size: float,
    time: Optional[object] = None,
    time_idx: int = 0,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Open one or more ERA5 netCDF files and extract a canonical box Dataset.

    Args:
        input_files: single path or list of paths. Typical usage is one
            atmospheric file (with ``t, u, v, q, z, pressure_level, ...``)
            plus one surface file (with ``msl, t2m, u10, v10, sp, skt, z,
            ...``). Variables from all files are merged; on conflict the
            first file wins.
        box_center: ``(lat_deg, lon_deg)`` matching the file's longitude
            convention (ERA5 uses ``[0, 360)``).
        box_size: half-width of the extracted box (deg), inclusive both sides.
        time: if given, select via ``ds.sel({time_name: time})``.
            Otherwise ``time_idx`` selects positionally.
        time_idx: which time slot to sample when ``time`` is None. ERA5
            initial-condition files typically carry a single timestamp
            (``valid_time=1``), so 0 is the natural default.
        lat_name, lon_name: source-side coord names (ERA5 uses the full
            spellings by default).

    Returns:
        Dataset with dims ``(level, latitude, longitude)``, canonical short
        variable names (``msl, 2t, 10u, 10v, t, u, v, q, z``) filtered to
        the subset present across all input files, ``time`` collapsed away.
        HICCUP-only vars (``sp, skt, sst, siconc, sd, stl1..4, tsn, ciwc,
        clwc, o3, sfc-z``) are omitted from the extract but preserved on
        :func:`write_back`.
    """
    if isinstance(input_files, (str, os.PathLike)):
        input_files = [input_files]
    if len(input_files) == 0:
        raise ValueError("extract_snapshot_dataset: input_files is empty")

    per_file: list[xr.Dataset] = []
    for path in input_files:
        raw = xr.open_dataset(path)
        try:
            canon, _ = _canonicalise(raw)
            # Only keep the K&C-touched canonicals; drop everything else.
            keep = [
                name
                for name in (list(SURF_VARS) + list(ATMOS_VARS))
                if name in canon.data_vars
            ]
            per_file.append(canon[keep].load())
        finally:
            raw.close()

    if not any(len(ds.data_vars) for ds in per_file):
        raise ValueError(
            "extract_snapshot_dataset: no K&C-relevant variables found in "
            f"input files {list(input_files)}"
        )

    # Merge across files — atmospheric file typically supplies t/u/v/q/z,
    # surface file supplies msl/2t/10u/10v (each disjoint). ``compat=override``
    # lets us tolerate benign coord-attr differences between files.
    merged = xr.merge(per_file, compat="override")

    if time is not None and "time" in merged.dims:
        merged = merged.sel(time=time, method="nearest")
    elif "time" in merged.dims:
        merged = merged.isel(time=time_idx)

    lats_global = np.asarray(merged[lat_name].values, dtype=float)
    lons_global = np.asarray(merged[lon_name].values, dtype=float)

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
            f"half-width {box_size} deg does not intersect ERA5 grid."
        )
    lat_sl = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_sl = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)

    return merged.isel({lat_name: lat_sl, lon_name: lon_sl})


def _default_output_file(input_file: str) -> str:
    """Insert ``.tc_bogus`` before the ``.nc`` suffix of ``input_file``."""
    root, ext = os.path.splitext(input_file)
    return f"{root}.tc_bogus{ext or '.nc'}"


def write_back(
    input_file: str,
    modified_ds: xr.Dataset,
    *,
    box_lats: np.ndarray,
    box_lons: np.ndarray,
    output_file: Optional[str] = None,
    time_slots: Optional[Sequence[int]] = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    overwrite: bool = False,
) -> str:
    """Write a modified canonical box back into a copy of an ERA5 netCDF.

    For each K&C-touched canonical variable both present in ``modified_ds``
    and in the source ``input_file``, the box slice is overwritten across
    every time slot (or the specified ``time_slots``). Every other variable
    in the source file — ``sp``, ``skt``, surface ``z``, ``ciwc``, ``clwc``,
    ``o3``, sea-ice/SST/soil moisture, ancillary bounds — passes through
    unchanged so HICCUP's downstream reads still find what they expect.

    Args:
        input_file: source ERA5 netCDF path.
        modified_ds: canonical box Dataset from :func:`extract_snapshot_dataset`
            after the vortex has been applied.
        box_lats, box_lons: coord arrays defining where the box lives in
            the source file's global grid.
        output_file: destination path. ``None`` (default) writes to
            ``<input>.tc_bogus.nc`` next to the source.
        time_slots: which time indices to overwrite. ``None`` (default)
            writes all slots, matching the Aurora convention of injecting
            the same vortex at every input step.
        lat_name, lon_name: source-side coord names.
        overwrite: allow writing over an existing output file. Default False
            to prevent accidental clobber of a previously-generated bogus
            file.

    Returns:
        The output file path.
    """
    if output_file is None:
        output_file = _default_output_file(input_file)
    if os.path.abspath(output_file) == os.path.abspath(input_file):
        raise ValueError(
            "write_back: output_file must differ from input_file. "
            "Pass an explicit output_file if you want to overwrite in place."
        )
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"write_back: {output_file} already exists. "
            "Pass overwrite=True or choose a different output_file."
        )

    raw = xr.open_dataset(input_file)
    try:
        out = raw.load().copy(deep=True)
    finally:
        raw.close()

    # Detect source-side dim names (must be done on the RAW dataset since
    # we deliberately did NOT canonicalise out).
    src_level_name = _detect_dim_name(out, _LEVEL_ALIASES)
    src_time_name = _detect_dim_name(out, _TIME_ALIASES)

    lats_global = np.asarray(out[lat_name].values, dtype=float)
    lons_global = np.asarray(out[lon_name].values, dtype=float)
    lat_sl = _find_box_indices(lats_global, np.asarray(box_lats, dtype=float), lat_name)
    lon_sl = _find_box_indices(lons_global, np.asarray(box_lons, dtype=float), lon_name)

    # Level lookup: for atmospheric vars, match modified_ds's ``level``
    # values against the source's level values. Level axis in modified_ds
    # is canonical ``level``; in ``out`` it's whatever ``src_level_name`` is.
    if src_level_name is not None:
        src_levels = np.asarray(out[src_level_name].values, dtype=float)
        level_index_map = {float(lev): int(idx) for idx, lev in enumerate(src_levels)}
    else:
        level_index_map = {}

    if "level" in modified_ds.dims:
        mod_levels = np.asarray(modified_ds["level"].values, dtype=float)
    else:
        mod_levels = np.array([], dtype=float)

    for canon in list(SURF_VARS) + list(ATMOS_VARS):
        if canon not in modified_ds.data_vars:
            continue
        # Find the source-side variable name for this canonical.
        aliases = ERA5_NETCDF_NAMES.get(canon, (canon,))
        expect_level = canon in ATMOS_VARS
        native = _detect_var_name(out, aliases, expect_level=expect_level)
        if native is None:
            continue

        target = out[native]
        dims = target.dims
        target_vals = np.asarray(target.values).copy()

        lat_axis = dims.index(lat_name)
        lon_axis = dims.index(lon_name)
        level_axis = dims.index(src_level_name) if src_level_name in dims else None
        time_axis = dims.index(src_time_name) if src_time_name in dims else None

        if time_axis is not None:
            n_time = target.sizes[src_time_name]
            ts: Sequence[int | None] = (
                list(range(n_time))
                if time_slots is None
                else [int(t) for t in time_slots]
            )
        else:
            ts = [None]

        if (
            level_axis is not None
            and canon in ATMOS_VARS
            and "level" in modified_ds.dims
        ):
            mod_da = modified_ds[canon].transpose("level", lat_name, lon_name)
            mod_ordered = np.asarray(mod_da.values)
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
        elif canon in SURF_VARS:
            mod_da = modified_ds[canon].transpose(lat_name, lon_name)
            mod_ordered = np.asarray(mod_da.values)
            for t in ts:
                idx = [slice(None)] * target_vals.ndim
                idx[lat_axis] = lat_sl
                idx[lon_axis] = lon_sl
                if t is not None and time_axis is not None:
                    idx[time_axis] = t
                target_vals[tuple(idx)] = mod_ordered.astype(
                    target_vals.dtype, copy=False
                )
        else:
            # Atmos var in modified_ds but source is 2D (or vice versa) —
            # dim ranks don't match. Skip so we don't accidentally clobber
            # the surface ``z`` field with atmospheric ``z`` data.
            continue

        out[native] = xr.DataArray(
            target_vals,
            dims=dims,
            coords=target.coords,
            attrs=target.attrs,
            name=target.name,
        )

    out.to_netcdf(output_file)
    out.close()
    return output_file
