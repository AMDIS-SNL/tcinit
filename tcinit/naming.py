"""Canonical variable names and per-backend translation maps.

tcinit's core (Snapshot, BogusVortex) operates on an ``xarray.Dataset`` whose
variables are keyed by canonical short names. Backends map to/from their
native names via the dicts here.

Canonical short names (matches Aurora ``Batch`` and ERA5 GRIB shortName):

    Surface     msl, 2t, 10u, 10v
    Atmospheric t, u, v, q, z

Note: ``2t``, ``10u``, ``10v`` are valid dict keys and xarray Dataset variable
names, but not attribute names. Always use ``ds["2t"]``, never ``ds.2t``.
"""

from __future__ import annotations

from typing import Mapping

# Canonical surface variables. May be a subset of a Dataset (NeuralGCM has none).
SURF_VARS: tuple[str, ...] = ("msl", "2t", "10u", "10v")

# Canonical atmospheric (isobaric-level) variables.
ATMOS_VARS: tuple[str, ...] = ("t", "u", "v", "q", "z")

CANONICAL_VARS: tuple[str, ...] = SURF_VARS + ATMOS_VARS

# Aurora Batch: dict keys are already canonical. Kept as an explicit identity
# map to make backend adapters symmetric.
AURORA_VAR_MAP: Mapping[str, str] = {v: v for v in CANONICAL_VARS}

# ERA5 (as HICCUP consumes it), identity map for the GRIB shortName convention.
# Kept for backwards compatibility with the xarray-passthrough apply() path
# in hiccup_backend: when the input Dataset already uses GRIB shortnames
# (2t, 10u, 10v, msl, t, u, v, q, z), no rename is needed.
ERA5_VAR_MAP: Mapping[str, str] = {
    "msl": "msl",
    "2t": "2t",
    "10u": "10u",
    "10v": "10v",
    "t": "t",
    "u": "u",
    "v": "v",
    "q": "q",
    "z": "z",
}

# ERA5 as it actually lands on disk from the CDS API's netCDF export.
# The newer CDS renames some surface fields (2t → t2m, 10u → u10, 10v → v10)
# while keeping msl and the atmospheric names unchanged. Each canonical short
# name maps to an ordered list of aliases; the HICCUP backend picks the first
# one that's present in the source file so both old-GRIB-shortname files and
# modern CDS-netCDF files work. Surface geopotential ``z`` in the sfc file
# is intentionally NOT the same variable as atmospheric ``z`` — the backend
# distinguishes by dim structure at write time.
ERA5_NETCDF_NAMES: Mapping[str, tuple[str, ...]] = {
    "msl": ("msl",),
    "2t": ("t2m", "2t"),
    "10u": ("u10", "10u"),
    "10v": ("v10", "10v"),
    "t": ("t",),
    "u": ("u",),
    "v": ("v",),
    "q": ("q",),
    "z": ("z",),
}

# Long-form CF-ish variable names used by NeuralGCM, GraphCast, and WeatherNext.
# Split into atmos vs. surface so downstream backends can compose whichever
# subset their model actually carries. NeuralGCM has no surface fields;
# WeatherNext-Graph and WeatherNext-2 both have surface + atmos.
LONGFORM_ATMOS_VAR_MAP: Mapping[str, str] = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
LONGFORM_SURFACE_VAR_MAP: Mapping[str, str] = {
    "msl": "mean_sea_level_pressure",
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
}

# NeuralGCM: long-form names, atmospheric only (no surface). Missing surface
# entries signal to the vortex-apply layer that surface writes must be skipped.
NEURALGCM_VAR_MAP: Mapping[str, str] = dict(LONGFORM_ATMOS_VAR_MAP)

# WeatherNext (Graph, WN2, Cyclones): long-form names for both surface and
# atmospheric. Precipitation, vertical velocity, 100m wind, static fields,
# and forcings are intentionally absent so backends skip them.
WEATHERNEXT_VAR_MAP: Mapping[str, str] = {
    **LONGFORM_ATMOS_VAR_MAP,
    **LONGFORM_SURFACE_VAR_MAP,
}

# ---------------------------------------------------------------------------
# SFNO (earth2studio) — flat variable axis with per-level names.
# ---------------------------------------------------------------------------
#
# The 73-channel SFNO checkpoint stacks all fields on a single ``variable``
# axis of its (batch, time, lead_time, variable, lat, lon) input tensor. Names
# are ``u10m``, ``t500``, ``q850`` etc. — surface get an "m" or short suffix,
# atmospheric use "{prefix}{level_hPa}".

# Pressure levels carried by the standard 73-channel SFNO.
SFNO_LEVELS_HPA: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)

# SFNO surface names → canonical short names. Only the K&C-touched subset is
# listed; the untouched SFNO surface channels (``u100m``, ``v100m``, ``sp``,
# ``tcwv``) are intentionally absent so they get skipped by both extract and
# write_back.
SFNO_SURFACE_TO_CANON: Mapping[str, str] = {
    "u10m": "10u",
    "v10m": "10v",
    "t2m": "2t",
    "msl": "msl",
}

# Inverse map for name lookup.
_SFNO_CANON_TO_SURFACE: Mapping[str, str] = {
    v: k for k, v in SFNO_SURFACE_TO_CANON.items()
}

# Canonical atmospheric prefixes that SFNO uses in its per-level naming.
# Matches ``ATMOS_VARS`` but re-declared here to make the SFNO helpers
# usable without importing the full canonical schema.
SFNO_ATMOS_PREFIXES: tuple[str, ...] = ("u", "v", "z", "t", "q")


def sfno_channel_name(canon: str, level_hpa: int | None = None) -> str:
    """Build the SFNO channel name for a (canonical, level) pair.

    Surface canonicals (``10u``, ``10v``, ``2t``, ``msl``) require
    ``level_hpa=None``; atmospheric canonicals (``u``, ``v``, ``z``, ``t``,
    ``q``) require an integer hPa level. Raises ``ValueError`` on bad
    combinations so callers can guard extraction of unsupported vars.
    """
    if canon in _SFNO_CANON_TO_SURFACE:
        if level_hpa is not None:
            raise ValueError(
                f"sfno_channel_name({canon!r}): surface variable takes no level"
            )
        return _SFNO_CANON_TO_SURFACE[canon]
    if canon in SFNO_ATMOS_PREFIXES:
        if level_hpa is None:
            raise ValueError(
                f"sfno_channel_name({canon!r}): atmospheric variable requires level_hpa"
            )
        return f"{canon}{int(level_hpa)}"
    raise ValueError(f"sfno_channel_name: unknown canonical name {canon!r}")


def sfno_channel_index(
    variables,
    canon: str,
    level_hpa: int | None = None,
) -> int | None:
    """Locate the SFNO variable-axis index for a canonical (name, level) pair.

    ``variables`` is the array from ``coords["variable"]`` (an earth2studio
    CoordSystem entry). Returns ``None`` when the target channel is absent
    (truncated variable set, or an unhandled canonical). Callers use ``None``
    as a "skip this field" signal.
    """
    import numpy as np

    try:
        target = sfno_channel_name(canon, level_hpa)
    except ValueError:
        return None
    matches = np.where(np.asarray(variables) == target)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def sfno_variables_73() -> list[str]:
    """Return the 73-variable SFNO channel list in the checkpoint's canonical order.

    Matches ``earth2studio/models/px/sfno.py``'s ``VARIABLES`` constant, but
    defined here so tests and tools can use it without importing makani.
    """
    out = ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv"]
    for prefix in SFNO_ATMOS_PREFIXES:
        for lev in SFNO_LEVELS_HPA:
            out.append(f"{prefix}{lev}")
    return out


def resolve(name: str, var_map: Mapping[str, str] | None) -> str | None:
    """Translate a canonical name to the backend-native name.

    Returns None if the canonical name is not covered by ``var_map`` (which
    the vortex applier interprets as "backend doesn't carry this field —
    skip"). If ``var_map`` is None the canonical name passes through.
    """
    if var_map is None:
        return name
    return var_map.get(name)
