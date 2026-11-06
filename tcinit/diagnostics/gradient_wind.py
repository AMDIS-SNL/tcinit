"""Gradient-wind balance of a storm-centred Snapshot.

Compares the azimuthal-mean cyclonic tangential wind ``V_t(r, p)`` around the
storm centre with the gradient wind ``V_gr(r, p)`` implied by the snapshot's
own azimuthal-mean geopotential:

    V_gr^2 / r + |f| V_gr = dPhi/dr          (pressure levels)
    V_gr^2 / r + |f| V_gr = (1/rho) dp/dr     (surface, from MSLP)

so ``V_gr = -|f| r / 2 + sqrt(f^2 r^2 / 4 + r dPhi/dr)``, with ``f`` at the
storm centre. Using ``|f|`` and the cyclonic sign for ``V_t`` makes the same
expression hold in both hemispheres (as in
:mod:`tcinit.ideal_tc_vortex`).

Reading the ratio ``V_t / V_gr``:

- ~1 above the boundary layer: the vortex is in gradient balance.
- < 1 at 10 m: expected; surface friction makes the near-surface wind
  sub-gradient (K&C use 0.8, ``K0_FRICTION``).
- > 1 near the radius of maximum wind in the boundary layer: supergradient
  flow, which real intense TCs show; ML models may or may not reproduce it.
- Far from 1 aloft: the wind and mass fields disagree, e.g. right after a
  bogus insertion, or a model that has drifted out of balance.

Caveats
-------
Axisymmetric only
    Azimuthal means drop the asymmetric (storm-motion, shear) part, which
    is in a different balance. Annuli are kept only out to the nearest box
    edge, so every annulus is complete.
Coarse grids
    The innermost annulus holds a handful of points and the radius of
    maximum wind is resolved to about one grid spacing. Averaging a sharply
    peaked wind over an annulus also biases the ratio low: an exactly
    balanced Holland vortex with a 50 km RMW reads 0.97 at 0.25 deg and
    1.00 at 0.1 deg. At 1 deg such a core is not resolved and the ratio
    (0.74 for the same vortex) says little about balance.
Below ground
    Levels with ``p`` above the annulus-mean MSLP hold extrapolated values
    in pressure-level products; they are masked (NaN).
Density
    The surface density is dry, ``rho = p / (R_d T_2m)``; the virtual
    correction is ~0.5 %.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from tcinit.constants import HPA_TO_PA, OMEGA_EARTH, R_D
from tcinit.earth_util import great_circle_bearing, haversine_dist

KM_PER_DEG = 111.195


@dataclass
class GradientWindBalance:
    """Azimuthal-mean tangential and gradient wind around a storm centre.

    Arrays are ``(n_levels, n_r)`` on pressure levels and ``(n_r,)`` at the
    surface; NaN where unavailable (missing input, below ground, empty
    annulus, or no real gradient-wind root).
    """

    center: Tuple[float, float]
    f_coriolis: float
    r_km: np.ndarray
    levels_hpa: Optional[np.ndarray] = None
    v_t: Optional[np.ndarray] = None
    v_grad: Optional[np.ndarray] = None
    v_t_10m: Optional[np.ndarray] = None
    v_grad_sfc: Optional[np.ndarray] = None
    p_sfc_pa: Optional[np.ndarray] = None

    @property
    def imbalance(self) -> Optional[np.ndarray]:
        """``V_t - V_gr`` on pressure levels (m/s)."""
        if self.v_t is None or self.v_grad is None:
            return None
        return self.v_t - self.v_grad

    def level_summary(self, level_hpa: float) -> dict:
        """Radius of maximum ``V_t`` at ``level_hpa`` and the balance there.

        Returns ``rmw_km``, ``v_t_max``, ``v_grad_at_rmw`` (m/s) and
        ``ratio = v_t_max / v_grad_at_rmw``; NaN when the level is absent.
        """
        nan = {
            "rmw_km": np.nan,
            "v_t_max": np.nan,
            "v_grad_at_rmw": np.nan,
            "ratio": np.nan,
        }
        if self.levels_hpa is None or self.v_t is None:
            return nan
        hits = np.where(np.asarray(self.levels_hpa) == level_hpa)[0]
        if hits.size == 0:
            return nan
        k = int(hits[0])
        return _summary_at_max(self.r_km, self.v_t[k], self.v_grad[k])

    def surface_summary(self) -> dict:
        """Like :meth:`level_summary` for the 10 m wind vs. the MSLP gradient wind."""
        if self.v_t_10m is None or self.v_grad_sfc is None:
            return {
                "rmw_km": np.nan,
                "v_t_max": np.nan,
                "v_grad_at_rmw": np.nan,
                "ratio": np.nan,
            }
        return _summary_at_max(self.r_km, self.v_t_10m, self.v_grad_sfc)

    def imbalance_rms(
        self,
        *,
        r_max_km: Optional[float] = None,
        p_range_hpa: Tuple[float, float] = (200.0, 850.0),
    ) -> float:
        """RMS of ``V_t - V_gr`` (m/s) over ``r <= r_max_km`` and levels in ``p_range_hpa``.

        The default levels skip the boundary layer, where friction makes the
        wind legitimately unbalanced.
        """
        imb = self.imbalance
        if imb is None:
            return float("nan")
        lev = np.asarray(self.levels_hpa, dtype=float)
        rows = (lev >= p_range_hpa[0]) & (lev <= p_range_hpa[1])
        cols = (
            np.ones(self.r_km.shape, bool)
            if r_max_km is None
            else self.r_km <= r_max_km
        )
        vals = imb[np.ix_(rows, cols)]
        vals = vals[np.isfinite(vals)]
        return float(np.sqrt(np.mean(vals**2))) if vals.size else float("nan")


def _summary_at_max(r_km, v_t, v_grad) -> dict:
    if not np.isfinite(v_t).any():
        return {
            "rmw_km": np.nan,
            "v_t_max": np.nan,
            "v_grad_at_rmw": np.nan,
            "ratio": np.nan,
        }
    j = int(np.nanargmax(v_t))
    vg = float(v_grad[j])
    return {
        "rmw_km": float(r_km[j]),
        "v_t_max": float(v_t[j]),
        "v_grad_at_rmw": vg,
        "ratio": float(v_t[j] / vg) if np.isfinite(vg) and vg > 0 else np.nan,
    }


def _azimuthal_mean(field, bin_idx, n_bins):
    """Mean of ``field`` (``(..., n_lat, n_lon)``) per radial bin -> ``(..., n_bins)``."""
    field = np.asarray(field, dtype=float)
    lead = field.shape[:-2]
    flat = field.reshape((-1, bin_idx.size))
    b = bin_idx.ravel()
    out = np.full((flat.shape[0], n_bins), np.nan)
    for i, row in enumerate(flat):
        ok = np.isfinite(row) & (b >= 0)
        counts = np.bincount(b[ok], minlength=n_bins)
        sums = np.bincount(b[ok], weights=row[ok], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(counts > 0, sums / counts, np.nan)
    return out.reshape(lead + (n_bins,))


def _gradient_wind(r_m, dphi_dr, f_abs):
    under = (0.5 * f_abs * r_m) ** 2 + r_m * dphi_dr
    with np.errstate(invalid="ignore"):
        return np.where(under >= 0.0, -0.5 * f_abs * r_m + np.sqrt(under), np.nan)


def gradient_wind_balance(
    lats_deg,
    lons_deg,
    center: Tuple[float, float],
    *,
    levels_hpa=None,
    u=None,
    v=None,
    z=None,
    msl=None,
    u10=None,
    v10=None,
    t2m=None,
    dr_km: Optional[float] = None,
    r_max_km: Optional[float] = None,
) -> GradientWindBalance:
    """Gradient-wind balance around ``center`` from box fields.

    Args:
        lats_deg, lons_deg: 1-D box coordinates.
        center: storm centre ``(lat, lon)`` (deg), same longitude convention.
        levels_hpa: 1-D pressure levels for ``u``, ``v``, ``z``.
        u, v, z: ``(level, lat, lon)`` winds (m/s) and geopotential (m^2/s^2).
        msl, u10, v10, t2m: ``(lat, lon)`` surface fields (Pa, m/s, K). The
            surface balance needs ``msl`` and ``t2m``; ``u10``/``v10`` give
            the 10 m tangential wind. ``msl`` also masks levels below ground.
        dr_km: annulus width. Default: the box's latitude spacing.
        r_max_km: outermost annulus edge. Default (and upper limit): the
            distance from ``center`` to the nearest box edge.

    Pressure-level and surface parts are each skipped (left None) when their
    inputs are missing.
    """
    lats = np.asarray(lats_deg, dtype=float)
    lons = np.asarray(lons_deg, dtype=float)
    lat_c, lon_c = float(center[0]), float(center[1])
    lat2, lon2 = np.meshgrid(lats, lons, indexing="ij")

    dist = haversine_dist(lat_c, lon_c, lat2, lon2)
    edge = min(
        haversine_dist(lat_c, lon_c, lats.min(), lon_c),
        haversine_dist(lat_c, lon_c, lats.max(), lon_c),
        haversine_dist(lat_c, lon_c, lat_c, lons.min()),
        haversine_dist(lat_c, lon_c, lat_c, lons.max()),
    )
    r_out = edge if r_max_km is None else min(float(r_max_km), edge)
    if dr_km is None:
        dr_km = abs(float(lats[1] - lats[0])) * KM_PER_DEG if lats.size > 1 else 25.0
    n_bins = max(int(np.floor(r_out / dr_km)), 1)
    bin_idx = np.floor(dist / dr_km).astype(int)
    bin_idx[bin_idx >= n_bins] = -1
    # Mean radius of the points in each annulus, not its midpoint: the inner
    # annuli hold few points, and the midpoint biases dPhi/dr and r there.
    r_km = _azimuthal_mean(dist, bin_idx, n_bins)
    empty = ~np.isfinite(r_km)
    r_km[empty] = ((np.arange(n_bins) + 0.5) * dr_km)[empty]

    # Local outward radial direction: the bearing from each point back to
    # the centre, reversed. Cyclonic tangent is the radial rotated 90 deg
    # counter-clockwise (NH) or clockwise (SH).
    beta = great_circle_bearing(lat2, lon2, lat_c, lon_c) + np.pi
    sign = -1.0 if lat_c < 0 else 1.0
    t_east, t_north = -np.cos(beta) * sign, np.sin(beta) * sign

    f = 2.0 * OMEGA_EARTH * np.sin(np.deg2rad(lat_c))
    f_abs = abs(f)
    r_m = r_km * 1000.0
    out = GradientWindBalance(center=(lat_c, lon_c), f_coriolis=f, r_km=r_km)

    p_sfc = _azimuthal_mean(msl, bin_idx, n_bins) if msl is not None else None
    out.p_sfc_pa = p_sfc

    if u is not None and v is not None and z is not None and levels_hpa is not None:
        levels = np.asarray(levels_hpa, dtype=float)
        v_t_grid = np.asarray(u) * t_east + np.asarray(v) * t_north
        v_t = _azimuthal_mean(v_t_grid, bin_idx, n_bins)
        phi = _azimuthal_mean(z, bin_idx, n_bins)
        dphi_dr = (
            np.gradient(phi, r_m, axis=-1) if n_bins > 1 else np.full_like(phi, np.nan)
        )
        v_grad = _gradient_wind(r_m[None], dphi_dr, f_abs)
        if p_sfc is not None:
            below = levels[:, None] * HPA_TO_PA > p_sfc[None]
            v_t = np.where(below, np.nan, v_t)
            v_grad = np.where(below, np.nan, v_grad)
        out.levels_hpa = levels
        out.v_t = v_t
        out.v_grad = v_grad

    if u10 is not None and v10 is not None:
        out.v_t_10m = _azimuthal_mean(
            np.asarray(u10) * t_east + np.asarray(v10) * t_north, bin_idx, n_bins
        )
    if p_sfc is not None and t2m is not None and n_bins > 1:
        rho = p_sfc / (R_D * _azimuthal_mean(t2m, bin_idx, n_bins))
        out.v_grad_sfc = _gradient_wind(r_m, np.gradient(p_sfc, r_m) / rho, f_abs)
    return out


def snapshot_gradient_wind(snap, **kwargs) -> Optional[GradientWindBalance]:
    """:func:`gradient_wind_balance` for a :class:`tcinit.snapshot.Snapshot`.

    Uses ``snap.loc`` as the centre, the bundled ``src_u``/``src_v``/``src_z``
    for the pressure levels, and the cached surface fields. Returns None when
    the snapshot has no storm centre. ``kwargs`` go to
    :func:`gradient_wind_balance` (``dr_km``, ``r_max_km``).
    """
    loc = getattr(snap, "loc", None)
    if loc is None or not (np.isfinite(loc[0]) and np.isfinite(loc[1])):
        return None
    return gradient_wind_balance(
        snap.box_lats,
        snap.box_lons,
        loc,
        levels_hpa=snap.levels,
        u=getattr(snap, "src_u", None),
        v=getattr(snap, "src_v", None),
        z=getattr(snap, "src_z", None),
        msl=snap._msl,
        u10=snap._u10,
        v10=snap._v10,
        t2m=snap._t2m,
        **kwargs,
    )
