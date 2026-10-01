"""Moist Static Energy (MSE) diagnostics for tropical-cyclone rollouts.

MSE per unit mass:

    h = c_p * T + Phi + L_v * q

where Phi is geopotential in m^2/s^2 (ERA5/Aurora convention: 'z' is already
geopotential, not geopotential height; do NOT multiply by g).

Caveats
-------
Conservation limits
    MSE is conserved per parcel only under adiabatic, reversible-moist
    processes. Domain-integrated MSE drift over a multi-day forecast is
    expected due to radiation, precipitation, and surface-flux
    parameterisations (or their absence in a pure forecast model). Observed
    drift alone does not imply a model bug; it must be compared against
    physically expected drift rates before drawing conclusions.

Frozen MSE not available
    The thermodynamically exact conserved quantity for ice-bearing systems is
    frozen MSE: h_f = c_p T + Phi + L_v q - L_f q_i, where q_i is cloud-ice
    mixing ratio. We don't carry q_i, so h_f can't be computed here. For the
    lower / middle troposphere of a mature TC the ice correction is typically
    < 5 kJ/kg; above the freezing level in deep convection it becomes
    non-negligible.
"""

from __future__ import annotations

import numpy as np

from tcinit.constants import CP_DRY, G_0, LV


def mse_3d(T, geopotential, q):
    """MSE [J/kg] on whatever grid T, geopotential, q share.

    Args:
        T: temperature [K], any shape.
        geopotential: Phi [m^2/s^2] -- the 'z' field, NOT geopotential height.
            Typical tropospheric range: ~1000 m^2/s^2 at 1000 hPa to
            ~200000 m^2/s^2 at 50 hPa.
        q: specific humidity [kg/kg], same shape as T.

    Returns:
        h [J/kg], same shape as the inputs.
    """
    return CP_DRY * np.asarray(T) + np.asarray(geopotential) + LV * np.asarray(q)


def column_integrate_mse(mse, pressure_levels_hpa, level_axis=-3, p_surface_pa=None):
    """Column-integrated MSE [J/m^2] = (1/g) integral_p_top^p_sfc h dp.

    Uses the trapezoidal rule in pressure coordinates. Levels may be supplied
    in any order; the function sorts them internally.

    Args:
        mse: array whose ``level_axis`` dimension has length
            ``len(pressure_levels_hpa)``. Typical shape ``(n_lev, n_lat,
            n_lon)``.
        pressure_levels_hpa: 1-D array of pressure levels [hPa], any order.
        level_axis: axis of ``mse`` that corresponds to vertical levels.
            Default ``-3`` matches ``(n_lev, n_lat, n_lon)``. For a 1-D
            column use ``level_axis=0``.
        p_surface_pa: optional surface-pressure field [Pa], broadcastable to
            ``mse.shape`` with ``level_axis`` removed. When provided, a
            constant-extrapolation bottom segment from the lowest available
            level to the actual surface is added. If None, integration
            extends only between the highest- and lowest-pressure supplied
            levels. Pass MSLP as a proxy for surface pressure if desired,
            but pass it explicitly -- this function never reads MSLP on its
            own.

    Returns:
        Column MSE [J/m^2], shape equal to ``mse.shape`` with ``level_axis``
        removed. Typical TC-environment value: 2.5-3.0 x 10^9 J/m^2.
    """
    mse = np.asarray(mse, dtype=float)
    p_hpa = np.asarray(pressure_levels_hpa, dtype=float)

    ndim = mse.ndim
    ax = level_axis if level_axis >= 0 else ndim + level_axis

    # Sort ascending pressure (low p = upper troposphere first) so that
    # np.trapezoid integrates from p_top to p_bottom, giving a positive result.
    sort_idx = np.argsort(p_hpa)
    p_pa = p_hpa[sort_idx] * 100.0  # Pa

    mse_sorted = np.take(mse, sort_idx, axis=ax)
    integral = np.trapezoid(mse_sorted, x=p_pa, axis=ax) / G_0

    if p_surface_pa is not None:
        p_sfc = np.asarray(p_surface_pa, dtype=float)
        dp_extra = np.maximum(p_sfc - p_pa[-1], 0.0)
        idx = [slice(None)] * ndim
        idx[ax] = -1
        bottom_mse = mse_sorted[tuple(idx)]
        integral = integral + bottom_mse * dp_extra / G_0

    return integral


def mse_breakdown(T, geopotential, q):
    """Sensible-heat, potential-energy, and latent-heat components of MSE.

    Useful for diagnosing which term drives drift. The three components sum
    to :func:`mse_3d` for every grid cell.

    Args:
        T: temperature [K].
        geopotential: Phi [m^2/s^2] -- 'z', NOT geopotential height.
        q: specific humidity [kg/kg].

    Returns:
        ``(sensible, potential, latent)`` -- three arrays [J/kg], each the
        same shape as the inputs:
            ``sensible = c_p * T``,
            ``potential = Phi``,
            ``latent = L_v * q``.
    """
    T = np.asarray(T)
    phi = np.asarray(geopotential)
    q = np.asarray(q)
    return CP_DRY * T, phi, LV * q
