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

# ERA5 (as HICCUP consumes it). Short GRIB names for the fields we produce.
# HICCUP-relevant subset: t, u, v, q on isobaric levels. msl/2t/10u/10v/z pass
# through but HICCUP's ERA5 subclass only reads sp/skt/z for surface, so those
# writes are effectively cosmetic for HICCUP downstream.
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

# NeuralGCM: long-form names, atmospheric only (no surface). Missing surface
# entries signal to the vortex-apply layer that surface writes must be skipped.
NEURALGCM_VAR_MAP: Mapping[str, str] = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}


def resolve(name: str, var_map: Mapping[str, str] | None) -> str | None:
    """Translate a canonical name to the backend-native name.

    Returns None if the canonical name is not covered by ``var_map`` (which
    the vortex applier interprets as "backend doesn't carry this field —
    skip"). If ``var_map`` is None the canonical name passes through.
    """
    if var_map is None:
        return name
    return var_map.get(name)
