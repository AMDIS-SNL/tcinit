"""Model-agnostic storm-centric analysis window on a canonical xarray Dataset.

Snapshot bundles the geometry and environment statistics that
:class:`tcinit.ideal_tc_vortex.BogusVortex` needs: box lat/lon coordinates, isobaric
levels, storm centre ``loc``, reference disk radius ``rdr`` (km), storm and
environment boolean masks, environment-mean MSLP, and storm-mean 2 m T.

The class deliberately does *not* replicate every diagnostic in the prototype
:class:`causaltc.data_collection.aurora_snapshot.AuroraSnapshot` (vorticity,
warm-core, wind maxima, etc.). K&C 2009's bogus vortex only reads the fields
listed above; extra diagnostics belong in a separate diagnostics module.

Storm detection can be run automatically via :meth:`detect_storm`, or the
caller can populate ``loc`` / ``rdr`` / env-means directly via
:meth:`set_storm_center` (useful for NeuralGCM, which carries no surface
fields).
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from tcinit.constants import HPA_TO_PA
from tcinit.earth_util import haversine_dist
from tcinit.naming import resolve

# ---------------------------------------------------------------------------
# Small helpers (lifted from AuroraSnapshot, simplified to numpy-only)
# ---------------------------------------------------------------------------


def _distance_mask(
    lats: np.ndarray,
    lons: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> np.ndarray:
    """Boolean (lat, lon) mask of cells within ``radius_km`` of the centre."""
    if not (np.isfinite(center_lat) and np.isfinite(center_lon)):
        raise ValueError(
            f"_distance_mask: non-finite centre ({center_lat}, {center_lon})"
        )
    if not np.isfinite(radius_km) or radius_km < 0:
        raise ValueError(f"_distance_mask: invalid radius {radius_km}")
    mask = np.zeros((lats.size, lons.size), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            mask[i, j] = haversine_dist(lat, lon, center_lat, center_lon) <= radius_km
    return mask


def _find_extremum(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    find: str = "min",
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, Tuple[float, float]]:
    """Locate the (min or max) of ``field`` on the box grid, optionally masked."""
    work = np.asarray(field, dtype=float)
    if work.ndim != 2:
        raise ValueError(f"_find_extremum expects 2D field, got {work.shape}")
    if mask is not None:
        work = np.where(mask, work, np.nan)
    if not np.isfinite(work).any():
        return np.nan, (np.nan, np.nan)
    if find == "min":
        i, j = np.unravel_index(np.nanargmin(work), work.shape)
    elif find == "max":
        i, j = np.unravel_index(np.nanargmax(work), work.shape)
    else:
        raise ValueError(f"find must be 'min' or 'max', got {find!r}")
    return float(field[i, j]), (float(lats[i]), float(lons[j]))


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class Snapshot:
    """Storm-centric analysis window over a canonical xarray Dataset.

    Attributes populated by :meth:`from_xarray`:
        box_lats, box_lons : 1-D coordinate arrays (deg).
        levels             : 1-D isobaric levels (hPa), or ``None`` if the
                             Dataset has only surface variables.
        has_surface        : True if the source Dataset carried surface fields
                             (msl/2t/10u/10v).

    Populated by :meth:`detect_storm` or :meth:`set_storm_center`:
        loc                : (lat, lon) storm centre (deg).
        rdr                : reference disk radius (km).
        storm_mask         : (n_lat, n_lon) True inside ``rdr * multiplier``.
        env_mask           : (n_lat, n_lon) complement of ``storm_mask``.
        mslp_env_mean      : environment-mean MSLP (Pa) — feeds K&C ``p_n``.
        t2m_storm_mean     : storm-mean 2 m temperature (K) — feeds K&C ``T_0``.

    ``storm_computed`` becomes True once the detection/override step succeeds.
    """

    def __init__(
        self,
        *,
        storm_radius_multiplier: float = 3.0,
        fixed_rdr_km: Optional[float] = None,
    ) -> None:
        self.storm_radius_multiplier = float(storm_radius_multiplier)
        self.fixed_rdr_km = fixed_rdr_km

        # Populated by from_xarray:
        self.box_lats: Optional[np.ndarray] = None
        self.box_lons: Optional[np.ndarray] = None
        self.levels: Optional[np.ndarray] = None
        self.has_surface: bool = False
        # Cache the extracted per-cell 2-D surface fields (canonical names)
        # so detect_storm doesn't need to re-open the dataset. None-valued
        # when the source Dataset didn't carry that field.
        self._msl: Optional[np.ndarray] = None
        self._t2m: Optional[np.ndarray] = None
        self._u10: Optional[np.ndarray] = None
        self._v10: Optional[np.ndarray] = None

        # Populated by detect_storm / set_storm_center:
        self.loc: Optional[Tuple[float, float]] = None
        self.rdr: Optional[float] = None
        self.storm_mask: Optional[np.ndarray] = None
        self.env_mask: Optional[np.ndarray] = None
        self.mslp_env_mean: Optional[float] = None
        self.t2m_storm_mean: Optional[float] = None

        # Diagnostic sub-estimates from detect_storm (nan when unavailable):
        self.rdr_from_wind: float = float("nan")
        self.rdr_from_grad: float = float("nan")

        self.storm_computed: bool = False

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_xarray(
        cls,
        ds: xr.Dataset,
        *,
        var_map: Optional[Mapping[str, str]] = None,
        time=None,
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        level_name: str = "level",
        box: Optional[Tuple[float, float, float, float]] = None,
        storm_radius_multiplier: float = 3.0,
        fixed_rdr_km: Optional[float] = None,
    ) -> "Snapshot":
        """Ingest a canonical xarray Dataset into a Snapshot.

        Args:
            ds: Dataset carrying (at minimum) an isobaric level dim and one or
                more of the canonical variables (msl, 2t, 10u, 10v, t, u, v,
                q, z). Variable names may be backend-native if ``var_map`` is
                supplied.
            var_map: canonical -> native name map. If ``None``, treat variable
                names as already canonical.
            time: if the Dataset has a time dim, this is passed to
                ``ds.sel(time=time)``. Ignored for time-less Datasets.
            lat_name, lon_name, level_name: coordinate names in ``ds``.
            box: optional (lat_min, lat_max, lon_min, lon_max) window.
                Longitudes are assumed to share ``ds``'s convention.
            storm_radius_multiplier: passed through to the Snapshot instance.
            fixed_rdr_km: bypass ``detect_storm``'s wind/gradient estimation
                and use this value directly.
        """
        snap = cls(
            storm_radius_multiplier=storm_radius_multiplier,
            fixed_rdr_km=fixed_rdr_km,
        )

        if time is not None and "time" in ds.dims:
            ds = ds.sel(time=time)

        if box is not None:
            lat_min, lat_max, lon_min, lon_max = box
            # Slice along the lat/lon dims regardless of ascending/descending.
            lat_coord = ds[lat_name]
            lon_coord = ds[lon_name]
            lat_sel = (lat_coord >= lat_min) & (lat_coord <= lat_max)
            lon_sel = (lon_coord >= lon_min) & (lon_coord <= lon_max)
            ds = ds.where(lat_sel & lon_sel, drop=True)

        snap.box_lats = np.asarray(ds[lat_name].values, dtype=float)
        snap.box_lons = np.asarray(ds[lon_name].values, dtype=float)
        if level_name in ds.dims:
            snap.levels = np.asarray(ds[level_name].values, dtype=float)

        def _read(canon: str) -> Optional[np.ndarray]:
            native = resolve(canon, var_map)
            if native is None or native not in ds.data_vars:
                return None
            arr = ds[native].values
            return np.asarray(arr, dtype=float)

        snap._msl = _read("msl")
        snap._t2m = _read("2t")
        snap._u10 = _read("10u")
        snap._v10 = _read("10v")
        snap.has_surface = snap._msl is not None

        return snap

    # ------------------------------------------------------------------
    # Storm detection / override
    # ------------------------------------------------------------------

    def detect_storm(
        self,
        *,
        box_center: Optional[Tuple[float, float]] = None,
        max_loc_error_km: float = 450.0,
        verbose: bool = False,
    ) -> "Snapshot":
        """Populate ``loc``, ``rdr``, masks, and env-means from surface fields.

        Pipeline (simplified from AuroraSnapshot.compute_storm_data):
          1. Constrain the MSLP search to a disk of radius ``max_loc_error_km``
             centred on ``box_center`` (defaults to the box centroid).
          2. ``self.loc`` <- location of the minimum MSLP inside that disk.
          3. ``rdr_from_wind`` = distance from ``self.loc`` to the 10 m wind
             maximum (in the same disk).
          4. ``rdr_from_grad`` = distance from ``self.loc`` to the maximum of
             |grad MSLP|.
          5. ``self.rdr`` = min(rdr_from_wind, rdr_from_grad, max_loc_error_km);
             overridden by ``fixed_rdr_km`` if supplied at construction.
          6. Storm/env masks anchored at ``self.loc`` with radius
             ``storm_radius_multiplier * rdr``.
          7. ``mslp_env_mean`` = mean(msl[env_mask]);
             ``t2m_storm_mean`` = mean(2t[storm_mask]).
        """
        if not self.has_surface or self._msl is None:
            raise ValueError(
                "Snapshot.detect_storm: no surface fields loaded. "
                "For a surface-free backend (e.g. NeuralGCM), call "
                "set_storm_center() with explicit overrides."
            )

        if box_center is None:
            box_center = (
                float(0.5 * (self.box_lats[0] + self.box_lats[-1])),
                float(0.5 * (self.box_lons[0] + self.box_lons[-1])),
            )

        loc_guess_mask = _distance_mask(
            self.box_lats, self.box_lons, box_center[0], box_center[1], max_loc_error_km
        )

        _, mslp_loc = _find_extremum(
            self._msl, self.box_lats, self.box_lons, find="min", mask=loc_guess_mask
        )
        if not (np.isfinite(mslp_loc[0]) and np.isfinite(mslp_loc[1])):
            raise ValueError(
                "Snapshot.detect_storm: no finite MSLP minimum inside "
                f"{max_loc_error_km} km disk at {box_center}"
            )
        self.loc = mslp_loc

        # rdr_from_wind: distance to |10m wind| max in the same disk.
        if self._u10 is not None and self._v10 is not None:
            wspd10 = np.hypot(self._u10, self._v10)
            wind_mask = _distance_mask(
                self.box_lats,
                self.box_lons,
                self.loc[0],
                self.loc[1],
                max_loc_error_km,
            )
            _, wspd_loc = _find_extremum(
                wspd10, self.box_lats, self.box_lons, find="max", mask=wind_mask
            )
            if np.isfinite(wspd_loc[0]):
                self.rdr_from_wind = float(
                    haversine_dist(self.loc[0], self.loc[1], wspd_loc[0], wspd_loc[1])
                )

        # rdr_from_grad: distance to max |grad MSLP|.
        d_msl_dy, d_msl_dx = np.gradient(self._msl)
        grad_mag = np.hypot(d_msl_dx, d_msl_dy)
        _, grad_loc = _find_extremum(grad_mag, self.box_lats, self.box_lons, find="max")
        if np.isfinite(grad_loc[0]):
            self.rdr_from_grad = float(
                haversine_dist(self.loc[0], self.loc[1], grad_loc[0], grad_loc[1])
            )

        if self.fixed_rdr_km is not None and np.isfinite(self.fixed_rdr_km):
            self.rdr = float(self.fixed_rdr_km)
        else:
            candidates = [
                r
                for r in (self.rdr_from_wind, self.rdr_from_grad, max_loc_error_km)
                if np.isfinite(r) and r > 0
            ]
            if not candidates:
                raise ValueError(
                    "Snapshot.detect_storm: no finite rdr candidate; "
                    "pass fixed_rdr_km at construction to override."
                )
            self.rdr = float(min(candidates))

        self._build_masks_and_means()

        if verbose:
            print(
                f"Snapshot: loc={self.loc}, rdr={self.rdr:.1f} km "
                f"(wind={self.rdr_from_wind:.1f}, grad={self.rdr_from_grad:.1f}, "
                f"guess={max_loc_error_km}); mslp_env_mean="
                f"{self.mslp_env_mean/HPA_TO_PA:.1f} hPa, "
                f"t2m_storm_mean={self.t2m_storm_mean:.2f} K"
            )
        return self

    def set_storm_center(
        self,
        *,
        loc: Tuple[float, float],
        rdr_km: float,
        mslp_env_mean_pa: float,
        t2m_storm_mean_k: float,
    ) -> "Snapshot":
        """Populate the storm-detection outputs directly.

        Use this for backends without surface fields (NeuralGCM) or when the
        caller has already computed these from another source.
        """
        if not (np.isfinite(loc[0]) and np.isfinite(loc[1])):
            raise ValueError(f"set_storm_center: non-finite loc={loc}")
        if not (np.isfinite(rdr_km) and rdr_km > 0):
            raise ValueError(f"set_storm_center: non-positive rdr_km={rdr_km}")
        self.loc = (float(loc[0]), float(loc[1]))
        self.rdr = float(rdr_km)
        self.mslp_env_mean = float(mslp_env_mean_pa)
        self.t2m_storm_mean = float(t2m_storm_mean_k)
        # Build masks from loc/rdr; env_mean overrides come from arguments.
        storm_radius_km = self.storm_radius_multiplier * self.rdr
        self.storm_mask = _distance_mask(
            self.box_lats, self.box_lons, self.loc[0], self.loc[1], storm_radius_km
        )
        self.env_mask = ~self.storm_mask
        self.storm_computed = True
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_masks_and_means(self) -> None:
        """Build storm/env masks around ``self.loc`` and compute env-means."""
        storm_radius_km = self.storm_radius_multiplier * self.rdr
        self.storm_mask = _distance_mask(
            self.box_lats, self.box_lons, self.loc[0], self.loc[1], storm_radius_km
        )
        self.env_mask = ~self.storm_mask

        env_msl = self._msl[self.env_mask]
        if not np.isfinite(env_msl).any():
            raise ValueError(
                "Snapshot._build_masks_and_means: env_mask is empty or "
                "MSLP field has no finite values in the environment"
            )
        self.mslp_env_mean = float(np.nanmean(env_msl))

        if self._t2m is not None:
            storm_t2m = self._t2m[self.storm_mask]
            if np.isfinite(storm_t2m).any():
                self.t2m_storm_mean = float(np.nanmean(storm_t2m))
            else:
                self.t2m_storm_mean = float("nan")
        else:
            self.t2m_storm_mean = float("nan")

        self.storm_computed = True
