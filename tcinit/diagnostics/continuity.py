"""Pressure-coordinate continuity and hydrostatic-balance diagnostics.

The K&C bogus vortex (:mod:`tcinit.ideal_tc_vortex`) writes horizontal winds
with a divergent (inflow/outflow) part but no vertical velocity. Models that
read omega as an input (GraphCast / WeatherNext) then see the bogus
convergence next to the analysis omega. These helpers diagnose the omega
implied by a horizontal wind field through the pressure-coordinate
continuity equation

    d(omega)/dp = -div_p(V)    =>    omega(p) = omega(p_top) - int_{p_top}^{p} div_p(V) dp'

and check how well the state it is computed from satisfies the hydrostatic
relation that equation assumes:

    d(Phi)/d(ln p) = -R_d * T_v,    T_v = T * (1 + 0.608 q).

Conventions
-----------
Arrays are ``(level, lat, lon)``; ``levels_hpa`` can be in any order (they are
sorted internally and results are returned in the input order). Coordinates
are in degrees and may be ascending or descending. Omega is in Pa/s, positive
downward (ascent is negative).

Caveats
-------
Pressure-level data
    Levels below the surface (``p > p_sfc``) hold extrapolated values in
    ERA5/IFS pressure-level products. :func:`omega_from_continuity` stops the
    integral at ``p_sfc`` when one is given and returns NaN below it.
Gridded divergence of a bogus vortex is noisy
    The tangential wind is non-divergent analytically, but at 0.25 deg its
    centred-difference divergence near the radius of maximum wind is
    ~3e-4 1/s for a Mawar-strength vortex, and integrating it gives O(10)
    Pa/s of spurious omega. The K&C omega that the backends blend in
    (``BogusVortex.omega_bogus_pa_s``) is therefore integrated from the
    analytic radial wind, and :func:`omega_from_continuity` is a diagnostic
    here, not the source of the blended field.
Analysis omega is not continuity-consistent either
    ERA5 omega comes from the IFS model-level integral, then is interpolated
    to pressure levels. Recomputing it from pressure-level winds does not
    reproduce it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tcinit.constants import EARTH_RADIUS_A, EPSILON_RATIO, G_0, HPA_TO_PA, R_D

# T_v = T (1 + VIRTUAL_COEF q), VIRTUAL_COEF = (1 - eps) / eps ~ 0.608.
VIRTUAL_COEF = (1.0 - EPSILON_RATIO) / EPSILON_RATIO


def virtual_temperature(t, q):
    """T_v [K] from temperature [K] and specific humidity [kg/kg]."""
    return np.asarray(t) * (1.0 + VIRTUAL_COEF * np.asarray(q))


def gradient_latlon(f, lats_deg, lons_deg):
    """Horizontal gradient ``(df/dx, df/dy)`` [units/m] on the sphere.

    ``f`` has trailing dims ``(lat, lon)``. Centred differences in the
    interior, one-sided at the box edges (``np.gradient``).
    """
    f = np.asarray(f, dtype=float)
    lat = np.deg2rad(np.asarray(lats_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lons_deg, dtype=float))
    cos_lat = np.cos(lat)[:, None]
    df_dlat = np.gradient(f, lat, axis=-2)
    df_dlon = np.gradient(f, lon, axis=-1)
    return df_dlon / (EARTH_RADIUS_A * cos_lat), df_dlat / EARTH_RADIUS_A


def divergence_latlon(u, v, lats_deg, lons_deg):
    """Horizontal divergence [1/s] on the sphere.

    div = 1/(a cos(phi)) * [du/d(lambda) + d(v cos(phi))/d(phi)]
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    lat = np.deg2rad(np.asarray(lats_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lons_deg, dtype=float))
    cos_lat = np.cos(lat)[:, None]
    du_dlon = np.gradient(u, lon, axis=-1)
    dvcos_dlat = np.gradient(v * cos_lat, lat, axis=-2)
    return (du_dlon + dvcos_dlat) / (EARTH_RADIUS_A * cos_lat)


def omega_from_continuity(
    u,
    v,
    lats_deg,
    lons_deg,
    levels_hpa,
    *,
    p_sfc_pa=None,
    omega_sfc_target=None,
    correct_surface: bool = True,
):
    """Omega [Pa/s] from the horizontal wind by vertical integration of continuity.

    Integrates ``-div`` downward from the top level with ``omega(p_top) = 0``
    (trapezoidal rule in p). With ``p_sfc_pa`` the integral is carried to the
    surface, using the divergence at the lowest level above ground over the
    last partial layer, and levels below ground are returned as NaN.

    The integrated surface value generally misses the kinematic boundary
    condition ``omega_s = dp_s/dt + V_s . grad(p_s)``. Its error
    ``residual = omega(p_sfc) - omega_sfc_target`` measures how far the wind
    field is from column mass balance. With ``correct_surface=True`` the
    residual is removed linearly in pressure (O'Brien 1970), so the corrected
    profile is zero at the top and matches the target at the surface.

    Args:
        u, v: ``(level, lat, lon)`` winds [m/s].
        lats_deg, lons_deg: 1-D box coordinates.
        levels_hpa: 1-D pressure levels, any order.
        p_sfc_pa: ``(lat, lon)`` surface pressure [Pa]. None treats the
            highest-pressure level as the surface.
        omega_sfc_target: ``(lat, lon)`` surface boundary value [Pa/s].
            Default zero.
        correct_surface: apply the O'Brien correction.

    Returns:
        ``(omega, residual)``: ``omega`` is ``(level, lat, lon)`` in the
        input level order; ``residual`` is the ``(lat, lon)`` surface error
        before correction [Pa/s].
    """
    levels_hpa = np.asarray(levels_hpa, dtype=float)
    order = np.argsort(levels_hpa)  # top (low p) first
    p = levels_hpa[order] * HPA_TO_PA
    div = divergence_latlon(u, v, lats_deg, lons_deg)[order]

    # Cumulative trapezoid from the top.
    dp = np.diff(p)[:, None, None]
    layer = 0.5 * (div[1:] + div[:-1]) * dp
    omega = -np.concatenate([np.zeros_like(div[:1]), np.cumsum(layer, axis=0)], axis=0)

    horiz_shape = div.shape[1:]
    if p_sfc_pa is None:
        p_sfc = np.full(horiz_shape, p[-1])
    else:
        p_sfc = np.broadcast_to(np.asarray(p_sfc_pa, dtype=float), horiz_shape)
    target = (
        np.zeros(horiz_shape)
        if omega_sfc_target is None
        else np.broadcast_to(np.asarray(omega_sfc_target, dtype=float), horiz_shape)
    )

    p3 = p[:, None, None]
    above = p3 <= p_sfc[None]
    # Lowest level at or above the surface (index into the sorted levels).
    k_bot = np.clip(above.sum(axis=0) - 1, 0, p.size - 1)
    omega_bot = np.take_along_axis(omega, k_bot[None], axis=0)[0]
    div_bot = np.take_along_axis(div, k_bot[None], axis=0)[0]
    omega_s = omega_bot - div_bot * (p_sfc - p[k_bot])
    residual = omega_s - target

    if correct_surface:
        weight = (p3 - p[0]) / np.maximum(p_sfc - p[0], 1.0)[None]
        omega = omega - weight * residual[None]
    omega = np.where(above, omega, np.nan)

    out = np.empty_like(omega)
    out[order] = omega
    return out, residual


def hydrostatic_residual(t, z, q, levels_hpa, *, p_sfc_pa=None):
    """Layer hydrostatic error, expressed as a temperature [K].

    For each pair of adjacent levels, the thickness temperature
    ``T_hyd = (Phi_top - Phi_bot) / (R_d ln(p_bot / p_top))`` is compared
    with the layer-mean virtual temperature (mean of the two levels). A state
    in exact discrete balance gives zero; ``T_hyd - T_v`` is the virtual
    temperature error that the geopotential implies. 1 K of error over a
    100 hPa layer near 850 hPa is a ~35 m thickness error.

    Args:
        t, z, q: ``(level, lat, lon)`` temperature [K], geopotential
            [m^2/s^2] and specific humidity [kg/kg].
        levels_hpa: 1-D pressure levels, any order.
        p_sfc_pa: optional ``(lat, lon)`` surface pressure; layers whose
            bottom is below ground are NaN.

    Returns:
        ``(layer_p_hpa, residual_K)``: geometric-mean layer pressures
        (ascending) and ``(n_levels - 1, lat, lon)`` residuals.
    """
    levels_hpa = np.asarray(levels_hpa, dtype=float)
    order = np.argsort(levels_hpa)
    p = levels_hpa[order] * HPA_TO_PA
    tv = virtual_temperature(np.asarray(t)[order], np.asarray(q)[order])
    phi = np.asarray(z, dtype=float)[order]

    log_ratio = np.log(p[1:] / p[:-1])[:, None, None]
    t_hyd = (phi[:-1] - phi[1:]) / (R_D * log_ratio)
    residual = t_hyd - 0.5 * (tv[1:] + tv[:-1])
    if p_sfc_pa is not None:
        below = p[1:, None, None] > np.asarray(p_sfc_pa, dtype=float)[None]
        residual = np.where(below, np.nan, residual)
    layer_p_hpa = np.sqrt(p[1:] * p[:-1]) / HPA_TO_PA
    return layer_p_hpa, residual


def nonhydrostatic_ratio(omega, u, v, t, q, lats_deg, lons_deg, levels_hpa):
    """Hydrostatic w [m/s] and the ratio |Dw/Dt| / g.

    ``w = -omega / (rho g)`` with ``rho = p / (R_d T_v)``. The vertical
    acceleration is estimated from steady advection,
    ``Dw/Dt ~ u dw/dx + v dw/dy + omega dw/dp``. A ratio << 1 means the
    hydrostatic approximation behind pressure-coordinate continuity holds for
    this omega.

    Returns:
        ``(w, ratio)``, both ``(level, lat, lon)`` in the input level order
        (NaN where ``omega`` is NaN).
    """
    levels_hpa = np.asarray(levels_hpa, dtype=float)
    p3 = (levels_hpa * HPA_TO_PA)[:, None, None]
    rho = p3 / (R_D * virtual_temperature(t, q))
    w = -np.asarray(omega, dtype=float) / (rho * G_0)

    dw_dx, dw_dy = gradient_latlon(w, lats_deg, lons_deg)
    order = np.argsort(levels_hpa)
    dw_dp = np.empty_like(w)
    dw_dp[order] = np.gradient(w[order], p3[order, 0, 0], axis=0)
    accel = np.asarray(u) * dw_dx + np.asarray(v) * dw_dy + omega * dw_dp
    return w, np.abs(accel) / G_0


def _masked_stats(a, mask):
    vals = np.asarray(a)[..., mask] if mask is not None else np.asarray(a)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"max_abs": np.nan, "rms": np.nan}
    return {
        "max_abs": float(np.max(np.abs(vals))),
        "rms": float(np.sqrt(np.mean(vals**2))),
    }


def omega_balance_report(
    *,
    u_before,
    v_before,
    u_after,
    v_after,
    t_before,
    t_after,
    z_before,
    z_after,
    q_before,
    q_after,
    omega_before,
    omega_after,
    lats_deg,
    lons_deg,
    levels_hpa,
    p_sfc_after_pa=None,
    omega_sfc_target=None,
    mask=None,
) -> dict:
    """Summary numbers for an omega blend: continuity closure and hydrostatic balance.

    ``mask`` is a ``(lat, lon)`` bool array (e.g. ``bv.blend > 0``)
    restricting the statistics to the region the bogus touched.

    Returns a dict of ``{"max_abs", "rms"}`` entries:

    - ``omega_increment``: the change to omega [Pa/s].
    - ``surface_residual``: column mass-balance error of the gridded wind
      increment, ``omega(p_s)`` from :func:`omega_from_continuity` minus the
      surface target, before any correction [Pa/s]. Mostly truncation error
      of the tangential wind on the grid (see the module caveats).
    - ``continuity_after``: gridded ``div(dV) + d(d_omega)/dp`` of the
      increment [1/s], i.e. how far the blended omega is from discrete
      continuity with the blended winds. Compare with the TC boundary-layer
      convergence, ~1e-4 to 1e-3 1/s.
    - ``hydrostatic_before`` / ``hydrostatic_after``: layer T_v error [K]
      of the analysis and the blended state (see :func:`hydrostatic_residual`).
    - ``nonhydrostatic_ratio``: |Dw/Dt|/g of the blended omega.
    """
    levels = np.asarray(levels_hpa, dtype=float)
    du = np.asarray(u_after) - np.asarray(u_before)
    dv = np.asarray(v_after) - np.asarray(v_before)
    _, residual = omega_from_continuity(
        du,
        dv,
        lats_deg,
        lons_deg,
        levels,
        p_sfc_pa=p_sfc_after_pa,
        omega_sfc_target=omega_sfc_target,
    )
    d_omega = np.asarray(omega_after) - np.asarray(omega_before)

    order = np.argsort(levels)
    p_sorted = levels[order] * HPA_TO_PA
    ddomega_dp = np.empty_like(d_omega)
    ddomega_dp[order] = np.gradient(d_omega[order], p_sorted, axis=0)
    cont = divergence_latlon(du, dv, lats_deg, lons_deg) + ddomega_dp
    if p_sfc_after_pa is not None:
        cont = np.where(
            levels[:, None, None] * HPA_TO_PA <= p_sfc_after_pa, cont, np.nan
        )

    _, hyd_b = hydrostatic_residual(
        t_before, z_before, q_before, levels, p_sfc_pa=p_sfc_after_pa
    )
    _, hyd_a = hydrostatic_residual(
        t_after, z_after, q_after, levels, p_sfc_pa=p_sfc_after_pa
    )
    _, nh = nonhydrostatic_ratio(
        np.where(np.isfinite(omega_after), omega_after, 0.0),
        u_after,
        v_after,
        t_after,
        q_after,
        lats_deg,
        lons_deg,
        levels,
    )
    return {
        "omega_increment": _masked_stats(d_omega, mask),
        "surface_residual": _masked_stats(residual, mask),
        "continuity_after": _masked_stats(cont, mask),
        "hydrostatic_before": _masked_stats(hyd_b, mask),
        "hydrostatic_after": _masked_stats(hyd_a, mask),
        "nonhydrostatic_ratio": _masked_stats(nh, mask),
    }
