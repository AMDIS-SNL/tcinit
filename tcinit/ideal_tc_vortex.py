"""Analytic Kwon & Cheong (2009) bogus tropical-cyclone vortex.

Reference:
    Kwon, I.-H. and Cheong, H.-B., 2009: Tropical cyclone initialization with a
    spherical high-order filter and an idealized three-dimensional bogus vortex.
    *Monthly Weather Review* 138, 1344-1367. doi:10.1175/2009MWR2943.1

This is a model-agnostic port of the prototype's
``causaltc/bogus_model/kwon_cheong_bogus.py``. Two intentional deviations
from K&C, carried over from the prototype:

1. The maximum tangential wind is placed at ``snapshot.rdr`` (the reference
   disk radius from the Snapshot classifier) rather than at the radius of
   maximum surface-pressure gradient.
2. The PBL gradient-wind modifier (eq. 17,
   ``K_m(sigma) = 1 - 20.4*(sigma - 0.9)^2``) is only applied at sigma > 0.9;
   above that pressure surface the gradient wind is used unmodified.

The spherical-filter (section 2b) and asymmetric-domain (section 2c) steps
are skipped. Environment/storm separation comes from the Snapshot's masks
and the storm is treated as circular for this first pass.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Optional

import numpy as np
import xarray as xr

from tcinit.best_track import BestTrack
from tcinit.constants import (
    OMEGA_EARTH,
    G_0,
    R_D,
    EPSILON_RATIO,
    KNOTS_TO_MPS,
    HPA_TO_PA,
)
from tcinit.earth_util import haversine_dist, great_circle_bearing
from tcinit.naming import resolve
from tcinit.snapshot import Snapshot

# ---------------------------------------------------------------------------
# K&C 2009 parameters
# ---------------------------------------------------------------------------

R_0_KM = 100.0  # eq. 3 reference radius
K0_FRICTION = 0.8  # eq. 6 friction correction
BETA_0_DEG = 20.0  # eq. 18 inflow angle at r_m
BETA_1_DEG = 25.0  # inflow-angle correction at r_30
TAU = 5.5  # eq. 11c vertical-blend stiffness
GAMMA_LAPSE = 0.008  # K/m, eq. 15 lapse rate
RH_B = 0.99  # eq. 27 saturation target
R_H_KM = 900.0  # eq. 27 horizontal scale
P_0_HPA = 1000.0  # eq. 27 reference pressure
THETA_0_DEG = 30.0  # eq. 9b reference latitude
R_A_OFFSET_KM = 700.0  # eq. 9a: r_a = r_30 + 700 km


def _table_2_p_a_pa(p_c_hpa: float) -> float:
    """K&C 2009 Table 2: ambient pressure ``p_a`` (Pa) by central-pressure bin.

    Step lookup; smooth interpolation is deferred.
    """
    if p_c_hpa < 945.0:
        p_a_hpa = 120.0
    elif p_c_hpa < 960.0:
        p_a_hpa = 130.0
    elif p_c_hpa < 980.0:
        p_a_hpa = 140.0
    else:
        p_a_hpa = 150.0
    return p_a_hpa * HPA_TO_PA


def _q_saturation(t_kelvin: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    """Saturation specific humidity (kg/kg) via Tetens over liquid water."""
    t_c = t_kelvin - 273.15
    e_sat_pa = 611.2 * np.exp(17.67 * t_c / (t_kelvin - 29.65))
    return (
        EPSILON_RATIO
        * e_sat_pa
        / np.maximum(p_pa - (1.0 - EPSILON_RATIO) * e_sat_pa, 1.0)
    )


def _arctan_block(
    d_factor: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """``[arctan(exp(d*(b - ref))) - arctan(exp(d*(a - ref)))] / d``.

    Up to a factor of 2 that cancels in the q_1 num/denom ratio (K&C eq.
    23-24), this is the antiderivative of ``sech(d*(sigma - ref))`` evaluated
    from ``a`` to ``b``.
    """
    return (
        np.arctan(np.exp(d_factor * (b - ref)))
        - np.arctan(np.exp(d_factor * (a - ref)))
    ) / d_factor


def _solve_holland_A(
    *,
    B: float,
    r_30_km: float,
    V_g30: float,
    dp_pa: float,
    rho: float,
    f_coriolis: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> float:
    """Newton-Raphson solve for Holland's A parameter, K&C eqs. 7-8."""
    r_30_m = r_30_km * 1000.0
    # K&C eq. 7 is written with signed f; taking the magnitude of the cyclonic
    # gradient wind requires |f| so SH storms converge to the correct V_g.
    fr30_half = 0.5 * abs(f_coriolis) * r_30_m
    xi_30 = (R_0_KM / r_30_km) ** B
    coeff = B * dp_pa / rho * xi_30

    A = 1e-3  # K&C: "initial guess for A (set close to zero)"
    for _ in range(max_iter):
        exp_term = np.exp(-A * xi_30)
        under = fr30_half**2 + A * coeff * exp_term
        sqrt_under = np.sqrt(under)
        W = sqrt_under - fr30_half - V_g30
        dunder = coeff * exp_term * (1.0 - A * xi_30)
        W_prime = dunder / (2.0 * sqrt_under)
        if abs(W_prime) < 1e-15:
            break
        A_new = A - W / W_prime
        if A_new <= 0:
            A_new = 0.5 * A
        if abs(A_new - A) < tol * max(abs(A), 1.0):
            return float(A_new)
        A = A_new
    raise ValueError(
        f"_solve_holland_A: Newton-Raphson failed to converge in {max_iter} "
        f"iterations (B={B:.3g}, r_30={r_30_km:.1f} km, V_g30={V_g30:.2f} m/s). "
        f"K&C suggests perturbing r_30 to recover."
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class BogusVortex:
    """K&C 2009 analytic bogus tropical-cyclone vortex.

    Usage:
        bv = BogusVortex(snapshot, best_track, target_time).build()
        ds_modified = bv.apply(ds)
    """

    # ---- Surface variable names (canonical) -----------------------------
    # Names of the surface fields BogusVortex writes on apply(). These are
    # translated through the caller-supplied var_map at apply time.
    _SURF_VARS = ("msl", "2t", "10u", "10v")
    _ATMOS_VARS = ("t", "u", "v", "q", "z")

    def __init__(
        self,
        snapshot: Snapshot,
        best_track: BestTrack,
        target_time: datetime,
        *,
        n_radial_samples: int = 256,
    ) -> None:
        self.snapshot = snapshot
        self.best_track = best_track
        self.target_time = target_time
        self.n_radial_samples = int(n_radial_samples)

        self._built = False

        # Inputs (set by build()).
        self.p_c: Optional[float] = None
        self.p_n: Optional[float] = None
        self.p_a: Optional[float] = None
        self.V_m: Optional[float] = None
        self.V_g_max: Optional[float] = None
        self.T_0: Optional[float] = None
        self.R_v_km: Optional[float] = None
        self.r_30_km: Optional[float] = None
        self.rho: Optional[float] = None
        self.A: Optional[float] = None
        self.B: Optional[float] = None
        self.r_m_holland_km: Optional[float] = None
        self.f_coriolis: Optional[float] = None
        self.storm_lat_deg: Optional[float] = None
        self.storm_lon_deg: Optional[float] = None

        self.r_grid_km: Optional[np.ndarray] = None
        self.phi_grid_rad: Optional[np.ndarray] = None

        self.levels_hpa: Optional[np.ndarray] = None
        self.p_sfc_bogus_pa: Optional[np.ndarray] = None
        self.t2m_bogus_K: Optional[np.ndarray] = None
        self.u10_bogus_mps: Optional[np.ndarray] = None
        self.v10_bogus_mps: Optional[np.ndarray] = None
        self.T_anom_K: Optional[np.ndarray] = None
        self.Phi_anom_m2_s2: Optional[np.ndarray] = None
        self.u_bogus_mps: Optional[np.ndarray] = None
        self.v_bogus_mps: Optional[np.ndarray] = None
        self.rh_envelope: Optional[np.ndarray] = None
        self.blend: Optional[np.ndarray] = None
        self.storm_mask: Optional[np.ndarray] = None

        # Radial-vertical profiles, populated by build() and consumed by
        # plot_bogus. Shapes: r_radial_km is (n_radial_samples,); the rest
        # are (n_levels, n_radial_samples).
        self.r_radial_km: Optional[np.ndarray] = None
        self.Phi_anom_radial_m2_s2: Optional[np.ndarray] = None
        self.T_anom_radial_K: Optional[np.ndarray] = None
        self.V_g_radial_mps: Optional[np.ndarray] = None
        self.v_r_radial_mps: Optional[np.ndarray] = None
        self.p_sfc_radial_pa: Optional[np.ndarray] = None
        self.rh_envelope_radial: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # build()
    # ------------------------------------------------------------------

    def build(self, *, verbose: bool = False) -> "BogusVortex":
        """Precompute all intermediate fields for the bogus vortex.

        Raises ValueError if the snapshot lacks the masks / env-means
        required by the bogus injector, or if the best-track row is missing
        / invalid.
        """
        snap = self.snapshot
        if not snap.storm_computed:
            raise ValueError(
                "BogusVortex.build(): snapshot.storm_computed is False; "
                "call detect_storm() (or set_storm_center()) first"
            )
        if snap.storm_mask is None or snap.env_mask is None:
            raise ValueError("BogusVortex.build(): snapshot has no storm/env masks")
        if snap.mslp_env_mean is None or not np.isfinite(snap.mslp_env_mean):
            raise ValueError(
                "BogusVortex.build(): snapshot.mslp_env_mean is not finite"
            )
        if snap.t2m_storm_mean is None or not np.isfinite(snap.t2m_storm_mean):
            raise ValueError(
                "BogusVortex.build(): snapshot.t2m_storm_mean is not finite"
            )
        if snap.levels is None:
            raise ValueError(
                "BogusVortex.build(): snapshot.levels is None; "
                "provide a Dataset with an isobaric-level dim"
            )

        # ---- Pull from best track ----------------------------------------
        self.p_c = self.best_track.get_central_pressure(self.target_time)
        self.V_m = self.best_track.get_windspeed(self.target_time)
        r_30_km = self.best_track.get_30kt_radius(self.target_time)
        if r_30_km is None:
            # K&C climatology placeholder for ATCF-sourced tracks without r_30.
            r_30_km = 250.0
        self.r_30_km = r_30_km
        lat_storm, lon_storm = self.best_track.get_location(self.target_time)
        # Match the snapshot's longitude convention so antimeridian crossings
        # don't blow up the haversine distance.
        box_lon_centre = 0.5 * (float(snap.box_lons[0]) + float(snap.box_lons[-1]))
        while lon_storm - box_lon_centre > 180.0:
            lon_storm -= 360.0
        while box_lon_centre - lon_storm > 180.0:
            lon_storm += 360.0
        self.storm_lat_deg = lat_storm
        self.storm_lon_deg = lon_storm

        # ---- Pull from snapshot ------------------------------------------
        self.p_n = float(snap.mslp_env_mean)
        self.R_v_km = float(snap.rdr)
        self.T_0 = float(snap.t2m_storm_mean)
        self.levels_hpa = np.asarray(snap.levels, dtype=float)

        # ---- Table 2 -----------------------------------------------------
        self.p_a = _table_2_p_a_pa(self.p_c / HPA_TO_PA)

        # ---- Coriolis at storm centre ------------------------------------
        self.f_coriolis = 2.0 * OMEGA_EARTH * np.sin(np.deg2rad(lat_storm))
        cyclonic_sign = -1.0 if lat_storm < 0 else 1.0

        # ---- Geometry: r and bearing per cell ----------------------------
        n_lat = snap.box_lats.size
        n_lon = snap.box_lons.size
        r_grid = np.zeros((n_lat, n_lon))
        phi_grid = np.zeros((n_lat, n_lon))
        for i, lat in enumerate(snap.box_lats):
            for j, lon in enumerate(snap.box_lons):
                r_grid[i, j] = haversine_dist(lat_storm, lon_storm, lat, lon)
                phi_grid[i, j] = great_circle_bearing(lat_storm, lon_storm, lat, lon)
        # Avoid the r=0 singularity in derivatives; 100 m floor is well below
        # any storm-scale length.
        r_grid = np.maximum(r_grid, 1e-1)
        self.r_grid_km = r_grid
        self.phi_grid_rad = phi_grid

        # ---- Holland-style A, B (eqs. 5-8 with r_m := R_v) ---------------
        # TODO: use moist R_star (R_D·(1 + 0.608·q)) instead of dry R_D so rho
        # accounts for ambient humidity.
        self.rho = self.p_n / (R_D * self.T_0)
        self.V_g_max = (self.V_m / K0_FRICTION) * np.cos(np.deg2rad(BETA_0_DEG))
        dp = self.p_n - self.p_c
        if dp <= 0:
            raise ValueError(
                f"BogusVortex.build(): non-positive (p_n - p_c) = {dp:.3g} Pa "
                f"(p_n={self.p_n:.0f}, p_c={self.p_c:.0f}); cannot build a vortex"
            )
        self.B = self.rho * np.e * self.V_g_max**2 / dp
        V_30_mps = 30.0 * KNOTS_TO_MPS
        V_g30 = (V_30_mps / K0_FRICTION) * np.cos(np.deg2rad(BETA_1_DEG))
        self.A = _solve_holland_A(
            B=self.B,
            r_30_km=self.r_30_km,
            V_g30=V_g30,
            dp_pa=dp,
            rho=self.rho,
            f_coriolis=self.f_coriolis,
        )
        # K&C recovery clamp: if r_m falls outside [40, 100] km, recompute A.
        r_m_holland_km = R_0_KM * self.A ** (1.0 / self.B)
        if r_m_holland_km > 100.0:
            r_m_holland_km = 100.0
            self.A = (r_m_holland_km / R_0_KM) ** self.B
        elif r_m_holland_km < 40.0:
            r_m_holland_km = 40.0
            self.A = (r_m_holland_km / R_0_KM) ** self.B
        self.r_m_holland_km = r_m_holland_km

        if verbose:
            print(
                f"BogusVortex: p_c={self.p_c/HPA_TO_PA:.1f} hPa, "
                f"p_n={self.p_n/HPA_TO_PA:.1f} hPa, "
                f"p_a={self.p_a/HPA_TO_PA:.0f} hPa, "
                f"V_m={self.V_m:.1f} m/s, V_g_max={self.V_g_max:.1f} m/s, "
                f"R_v={self.R_v_km:.1f} km, r_30={self.r_30_km:.1f} km, "
                f"r_m_holland={self.r_m_holland_km:.1f} km, "
                f"A={self.A:.3g}, B={self.B:.3g}"
            )

        # ---- Surface pressure (eq. 3) ------------------------------------
        r_ratio_inv_B = (R_0_KM / r_grid) ** self.B
        self.p_sfc_bogus_pa = self.p_c + dp * np.exp(-self.A * r_ratio_inv_B)

        # ---- Geopotential anomaly fields (eqs. 9, 10) --------------------
        chi = 7.0e3 * np.tanh(np.deg2rad(lat_storm) / np.deg2rad(THETA_0_DEG)) ** 2
        r_a_km = self.r_30_km + R_A_OFFSET_KM
        bracket = 1.0 - np.tanh((r_grid - r_a_km) / (0.4 * r_a_km))
        phi_u_2d = (
            chi
            * (dp / (1000.0 * HPA_TO_PA))
            * bracket
            * np.exp(-self.A * r_ratio_inv_B)
        )
        phi_b_2d = -(R_D * self.T_0 / self.p_sfc_bogus_pa) * (
            self.p_n - self.p_sfc_bogus_pa
        )
        t_b_star_2d = -GAMMA_LAPSE / G_0 * phi_b_2d

        # ---- Per-level geopotential and temperature anomaly (eqs. 11-12) -
        n_lev = self.levels_hpa.size
        T_anom = np.zeros((n_lev, n_lat, n_lon))
        phi_field = np.zeros((n_lev, n_lat, n_lon))
        for k, p_hpa in enumerate(self.levels_hpa):
            p_pa = p_hpa * HPA_TO_PA
            if p_pa <= self.p_a:
                factor = self.p_a / p_pa - 1.0
                env = np.exp(-3.0 * factor**2)
                phi_field[k] = phi_u_2d * env
                T_anom[k] = -6.0 * self.p_a * phi_u_2d / (R_D * p_pa) * factor * env
            else:
                denom = self.p_sfc_bogus_pa - self.p_a
                xi = (p_pa - self.p_a) / denom
                xi = np.clip(xi, 0.0, 1.0)
                sech_tau = 1.0 / np.cosh(TAU)
                sech_txi = 1.0 / np.cosh(TAU * xi)
                tanh_txi = np.tanh(TAU * xi)
                E = (sech_txi - 1.0) / (sech_tau - 1.0)
                mu = xi**2 * (xi - 1.0) * (-3.0 * xi + 4.0)
                F = -TAU * sech_txi * tanh_txi / (sech_tau - 1.0)
                phi_diff = phi_b_2d - phi_u_2d
                safe_phi_diff = np.where(np.abs(phi_diff) > 1e-6, phi_diff, 1.0)
                sech_t_one = 1.0 / np.cosh(TAU)
                tanh_t_one = np.tanh(TAU)
                F_one = -TAU * sech_t_one * tanh_t_one / (sech_tau - 1.0)
                s = -(
                    F_one
                    + t_b_star_2d * R_D * denom / (self.p_sfc_bogus_pa * safe_phi_diff)
                )
                s = np.where(np.abs(phi_diff) > 1e-6, s, 0.0)
                phi_field[k] = phi_u_2d + phi_diff * (E + s * mu)
                T_anom[k] = -(p_pa * phi_diff / (R_D * denom)) * (
                    F - s * xi * (12.0 * xi**2 - 21.0 * xi + 8.0)
                )

        self.T_anom_K = T_anom
        self.Phi_anom_m2_s2 = phi_field

        # ---- t2m bogus field at surface (T_0 + T_b*(r)) -----------------
        self.t2m_bogus_K = self.T_0 + t_b_star_2d

        # ---- Tangential gradient wind V_g(r, p) (eq. 16) ----------------
        r_max_km = float(np.max(r_grid)) * 1.05
        r_max_km = max(r_max_km, 2.0 * self.R_v_km)
        r_radial = np.linspace(1e-1, r_max_km, self.n_radial_samples)
        r_ratio_inv_B_radial = (R_0_KM / r_radial) ** self.B
        p_sfc_radial = self.p_c + dp * np.exp(-self.A * r_ratio_inv_B_radial)
        bracket_radial = 1.0 - np.tanh((r_radial - r_a_km) / (0.4 * r_a_km))
        phi_u_radial = (
            chi
            * (dp / (1000.0 * HPA_TO_PA))
            * bracket_radial
            * np.exp(-self.A * r_ratio_inv_B_radial)
        )
        phi_b_radial = -(R_D * self.T_0 / p_sfc_radial) * (self.p_n - p_sfc_radial)
        t_b_star_radial = -GAMMA_LAPSE / G_0 * phi_b_radial

        V_g_radial = np.zeros((n_lev, r_radial.size))
        Phi_anom_radial = np.zeros((n_lev, r_radial.size))
        T_anom_radial = np.zeros((n_lev, r_radial.size))
        for k, p_hpa in enumerate(self.levels_hpa):
            p_pa = p_hpa * HPA_TO_PA
            if p_pa <= self.p_a:
                factor = self.p_a / p_pa - 1.0
                env = np.exp(-3.0 * factor**2)
                phi_radial = phi_u_radial * env
                T_anom_radial[k] = (
                    -6.0 * self.p_a * phi_u_radial / (R_D * p_pa) * factor * env
                )
            else:
                denom = p_sfc_radial - self.p_a
                xi = np.clip((p_pa - self.p_a) / denom, 0.0, 1.0)
                sech_tau = 1.0 / np.cosh(TAU)
                sech_txi = 1.0 / np.cosh(TAU * xi)
                tanh_txi = np.tanh(TAU * xi)
                E = (sech_txi - 1.0) / (sech_tau - 1.0)
                mu = xi**2 * (xi - 1.0) * (-3.0 * xi + 4.0)
                F = -TAU * sech_txi * tanh_txi / (sech_tau - 1.0)
                F_one = -TAU * (1.0 / np.cosh(TAU)) * np.tanh(TAU) / (sech_tau - 1.0)
                phi_diff = phi_b_radial - phi_u_radial
                safe_phi_diff = np.where(np.abs(phi_diff) > 1e-6, phi_diff, 1.0)
                s = -(
                    F_one
                    + t_b_star_radial * R_D * denom / (p_sfc_radial * safe_phi_diff)
                )
                s = np.where(np.abs(phi_diff) > 1e-6, s, 0.0)
                phi_radial = phi_u_radial + phi_diff * (E + s * mu)
                T_anom_radial[k] = -(p_pa * phi_diff / (R_D * denom)) * (
                    F - s * xi * (12.0 * xi**2 - 21.0 * xi + 8.0)
                )
            Phi_anom_radial[k] = phi_radial
            dphi_dr = np.gradient(phi_radial, r_radial * 1000.0)
            # Use |f|: K&C eq. 16 with signed f gives the wrong magnitude in
            # the SH; cyclonic-wind magnitude solves |V|^2/r + |f||V| = r dphi/dr.
            fr2 = 0.5 * abs(self.f_coriolis) * r_radial * 1000.0
            under = fr2**2 + (r_radial * 1000.0) * dphi_dr
            under = np.maximum(under, 0.0)
            V_g_radial[k] = -fr2 + np.sqrt(under)

        # Interpolate V_g(r, p) -> V_g at each (lat, lon) cell.
        V_g_field = np.zeros((n_lev, n_lat, n_lon))
        for k in range(n_lev):
            V_g_field[k] = np.interp(r_grid, r_radial, V_g_radial[k])

        # ---- PBL modifier (eq. 17), only at sigma > 0.9 ------------------
        for k, p_hpa in enumerate(self.levels_hpa):
            p_pa = p_hpa * HPA_TO_PA
            sigma_2d = p_pa / self.p_sfc_bogus_pa
            K_m = np.where(
                sigma_2d > 0.9,
                1.0 - 20.4 * (sigma_2d - 0.9) ** 2,
                1.0,
            )
            K_m = np.maximum(K_m, 0.0)
            V_g_field[k] *= K_m

        # ---- Radial wind (eqs. 18, 19, 20) -------------------------------
        r_m_km = self.R_v_km
        ratio = r_grid / r_m_km
        beta_deg = (50.0 * ratio**4 - 12.5) / (4.0 * ratio**4 + 1.0) + 12.5
        beta_rad = np.deg2rad(beta_deg)
        lo_idx = int(np.argmax(self.levels_hpa))
        V_g_pure_sfc = np.interp(r_grid, r_radial, V_g_radial[lo_idx])
        K_m_10m = 1.0 - 20.4 * (0.999 - 0.9) ** 2  # eq. 17 at sigma=0.999 (~0.8)
        V_g_10m = K_m_10m * V_g_pure_sfc
        v_r_surface = -V_g_10m * np.tan(beta_rad)

        sigma_a = self.p_a / np.maximum(self.p_sfc_bogus_pa, 1.0)
        c_r = 8.0 + 37.0 * np.tanh(r_grid / 167.0) * np.exp(
            -((r_grid / 350.0 - 1.0) ** 2)
        )
        d_r = 7.0 + self.p_a / (5.0 * HPA_TO_PA) - 0.5 * c_r
        q0 = v_r_surface / (1.0 / np.cosh(15.0 * (0.999 - 0.98)))
        sigma_T = 0.01

        # Note: K&C eq. 24 prints the first q_0 integral with bounds [0.98, 1],
        # but eq. 23 unambiguously puts it on [sigma_a, 0.98] (the middle region
        # where q_0 sech[d(sigma-0.98)] appears alongside q_1 sech[c(sigma-sigma_a)]).
        # Eq. 24 has a bounds typo; we implement eq. 23.
        num = -q0 * (
            _arctan_block(d_r, sigma_a, 0.98, ref=0.98)
            + _arctan_block(np.full_like(d_r, 15.0), 0.98, 1.0, ref=0.98)
        )
        denom = _arctan_block(c_r, sigma_a, 0.98, ref=sigma_a) + _arctan_block(
            np.full_like(c_r, 45.0), sigma_T, sigma_a, ref=sigma_a
        )
        safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        q1 = num / safe_denom

        v_r_field = np.zeros((n_lev, n_lat, n_lon))
        for k, p_hpa in enumerate(self.levels_hpa):
            sigma = p_hpa * HPA_TO_PA / self.p_sfc_bogus_pa
            below_a = sigma < sigma_a
            between = (sigma >= sigma_a) & (sigma < 0.98)
            above = sigma >= 0.98
            sech_45 = 1.0 / np.cosh(45.0 * (sigma - sigma_a))
            sech_c = 1.0 / np.cosh(c_r * (sigma - sigma_a))
            sech_d = 1.0 / np.cosh(d_r * (sigma - 0.98))
            sech_15 = 1.0 / np.cosh(15.0 * (sigma - 0.98))
            v_r_k = np.where(below_a, q1 * sech_45, 0.0)
            v_r_k = np.where(between, q1 * sech_c + q0 * sech_d, v_r_k)
            v_r_k = np.where(above, q0 * sech_15, v_r_k)
            v_r_field[k] = v_r_k

        # ---- Decompose (V_t, V_r) into (u, v) ----------------------------
        sin_phi = np.sin(self.phi_grid_rad)
        cos_phi = np.cos(self.phi_grid_rad)
        t_east = -cos_phi * cyclonic_sign
        t_north = sin_phi * cyclonic_sign
        r_east = sin_phi
        r_north = cos_phi

        u_field = np.zeros((n_lev, n_lat, n_lon))
        v_field = np.zeros((n_lev, n_lat, n_lon))
        for k in range(n_lev):
            u_field[k] = V_g_field[k] * t_east + v_r_field[k] * r_east
            v_field[k] = V_g_field[k] * t_north + v_r_field[k] * r_north
        self.u_bogus_mps = u_field
        self.v_bogus_mps = v_field
        self.u10_bogus_mps = V_g_10m * t_east + v_r_surface * r_east
        self.v10_bogus_mps = V_g_10m * t_north + v_r_surface * r_north

        # ---- Humidity envelope (eq. 27) ----------------------------------
        eta = (self.r_30_km / r_m_km) ** 2
        log_denom = np.log(1.0 + 0.556 * eta)
        if abs(log_denom) < 1e-12:
            log_denom = 1.0
        rh_env = np.zeros((n_lev, n_lat, n_lon))
        for k, p_hpa in enumerate(self.levels_hpa):
            vertical = np.exp(-8.0 * np.log(p_hpa / P_0_HPA) ** 2)
            num_r = np.log((R_H_KM + eta * r_grid) / (R_H_KM + eta * r_m_km))
            horizontal = np.exp(-10.0 * (num_r / log_denom) ** 2)
            rh_env[k] = vertical * horizontal
        self.rh_envelope = rh_env

        # ---- Cosine taper blend across the storm-mask edge ---------------
        storm_radius_km = float(snap.storm_radius_multiplier) * self.R_v_km
        self.storm_mask = r_grid <= storm_radius_km
        with np.errstate(invalid="ignore"):
            taper = 0.5 * (
                1.0
                + np.cos(
                    np.pi
                    * np.clip(
                        (r_grid - self.R_v_km) / (storm_radius_km - self.R_v_km),
                        0.0,
                        1.0,
                    )
                )
            )
        blend = np.where(r_grid <= self.R_v_km, 1.0, taper)
        blend = np.where(self.storm_mask, blend, 0.0)
        self.blend = blend

        # ---- Radial-vertical profiles for plotting -----------------------
        V_g_radial_pbl = np.zeros_like(V_g_radial)
        for k, p_hpa in enumerate(self.levels_hpa):
            sigma_r = (p_hpa * HPA_TO_PA) / p_sfc_radial
            K_m_r = np.where(sigma_r > 0.9, 1.0 - 20.4 * (sigma_r - 0.9) ** 2, 1.0)
            K_m_r = np.maximum(K_m_r, 0.0)
            V_g_radial_pbl[k] = V_g_radial[k] * K_m_r

        beta_rad_r = np.deg2rad(
            (50.0 * (r_radial / self.R_v_km) ** 4 - 12.5)
            / (4.0 * (r_radial / self.R_v_km) ** 4 + 1.0)
            + 12.5
        )
        V_g_10m_r = K_m_10m * V_g_radial[lo_idx]
        v_r_surface_r = -V_g_10m_r * np.tan(beta_rad_r)
        sigma_a_r = self.p_a / np.maximum(p_sfc_radial, 1.0)
        c_r_radial = 8.0 + 37.0 * np.tanh(r_radial / 167.0) * np.exp(
            -((r_radial / 350.0 - 1.0) ** 2)
        )
        d_r_radial = 7.0 + self.p_a / (5.0 * HPA_TO_PA) - 0.5 * c_r_radial
        q0_r = v_r_surface_r / (1.0 / np.cosh(15.0 * (0.999 - 0.98)))
        num_r = -q0_r * (
            _arctan_block(d_r_radial, sigma_a_r, 0.98, ref=0.98)
            + _arctan_block(np.full_like(d_r_radial, 15.0), 0.98, 1.0, ref=0.98)
        )
        denom_r = _arctan_block(
            c_r_radial, sigma_a_r, 0.98, ref=sigma_a_r
        ) + _arctan_block(
            np.full_like(c_r_radial, 45.0), sigma_T, sigma_a_r, ref=sigma_a_r
        )
        safe_denom_r = np.where(np.abs(denom_r) > 1e-12, denom_r, 1.0)
        q1_r = num_r / safe_denom_r

        v_r_radial = np.zeros((n_lev, r_radial.size))
        for k, p_hpa in enumerate(self.levels_hpa):
            sigma_r = (p_hpa * HPA_TO_PA) / p_sfc_radial
            below_a = sigma_r < sigma_a_r
            between = (sigma_r >= sigma_a_r) & (sigma_r < 0.98)
            above = sigma_r >= 0.98
            sech_45 = 1.0 / np.cosh(45.0 * (sigma_r - sigma_a_r))
            sech_c = 1.0 / np.cosh(c_r_radial * (sigma_r - sigma_a_r))
            sech_d = 1.0 / np.cosh(d_r_radial * (sigma_r - 0.98))
            sech_15 = 1.0 / np.cosh(15.0 * (sigma_r - 0.98))
            v_r_k = np.where(below_a, q1_r * sech_45, 0.0)
            v_r_k = np.where(between, q1_r * sech_c + q0_r * sech_d, v_r_k)
            v_r_k = np.where(above, q0_r * sech_15, v_r_k)
            v_r_radial[k] = v_r_k

        rh_env_radial = np.zeros((n_lev, r_radial.size))
        for k, p_hpa in enumerate(self.levels_hpa):
            vertical = np.exp(-8.0 * np.log(p_hpa / P_0_HPA) ** 2)
            num_r_1d = np.log((R_H_KM + eta * r_radial) / (R_H_KM + eta * r_m_km))
            horizontal = np.exp(-10.0 * (num_r_1d / log_denom) ** 2)
            rh_env_radial[k] = vertical * horizontal

        self.r_radial_km = r_radial
        self.Phi_anom_radial_m2_s2 = Phi_anom_radial
        self.T_anom_radial_K = T_anom_radial
        self.V_g_radial_mps = V_g_radial_pbl
        self.v_r_radial_mps = v_r_radial
        self.p_sfc_radial_pa = p_sfc_radial
        self.rh_envelope_radial = rh_env_radial

        self._built = True
        return self

    # ------------------------------------------------------------------
    # apply()
    # ------------------------------------------------------------------

    def apply(
        self,
        ds: xr.Dataset,
        *,
        var_map: Optional[Mapping[str, str]] = None,
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        level_name: str = "level",
        additive_z: bool = True,
    ) -> xr.Dataset:
        """Blend the built vortex into ``ds`` and return a new Dataset.

        ``ds`` is expected to share the snapshot's box: its ``lat_name`` and
        ``lon_name`` coordinates must exactly match ``snapshot.box_lats`` and
        ``snapshot.box_lons``. Levels present in ``ds[level_name]`` are
        matched against ``self.levels_hpa`` — extra levels in ``ds`` are left
        untouched.

        Variables absent from ``ds`` (or absent from ``var_map``) are silently
        skipped. This is what lets NeuralGCM Datasets (no surface vars) and
        HICCUP-shape Datasets (no ``2t``/``10u``/``10v``) share the same
        codepath as Aurora.
        """
        if not self._built:
            raise RuntimeError("BogusVortex.apply(): call build() first")

        snap = self.snapshot
        # Alignment check — cheap and prevents silent misregistration.
        ds_lats = np.asarray(ds[lat_name].values, dtype=float)
        ds_lons = np.asarray(ds[lon_name].values, dtype=float)
        if not (
            ds_lats.shape == snap.box_lats.shape
            and np.allclose(ds_lats, snap.box_lats, atol=1e-6)
        ):
            raise ValueError(
                "BogusVortex.apply(): ds latitudes do not match snapshot.box_lats"
            )
        if not (
            ds_lons.shape == snap.box_lons.shape
            and np.allclose(ds_lons, snap.box_lons, atol=1e-6)
        ):
            raise ValueError(
                "BogusVortex.apply(): ds longitudes do not match snapshot.box_lons"
            )

        out = ds.copy(deep=True)

        # Build the blend as a DataArray aligned by name — that way xarray
        # broadcasting handles arbitrary dim orderings (Aurora/ERA5 use
        # (level, latitude, longitude); NeuralGCM uses (level, longitude,
        # latitude)) without any transposes.
        blend_da = xr.DataArray(
            self.blend,
            dims=(lat_name, lon_name),
            coords={lat_name: snap.box_lats, lon_name: snap.box_lons},
        )

        # ---- Surface variables (replacement blend) ----------------------
        surf_map = {
            "msl": self.p_sfc_bogus_pa,
            "2t": self.t2m_bogus_K,
            "10u": self.u10_bogus_mps,
            "10v": self.v10_bogus_mps,
        }
        for canon, bogus_2d in surf_map.items():
            native = resolve(canon, var_map)
            if native is None or native not in out.data_vars:
                continue
            bogus_da = xr.DataArray(
                bogus_2d,
                dims=(lat_name, lon_name),
                coords={lat_name: snap.box_lats, lon_name: snap.box_lons},
            )
            out[native] = (1.0 - blend_da) * out[native] + blend_da * bogus_da

        # ---- Atmospheric variables (per-level, level-name-keyed) --------
        # Only levels that are present in both the built vortex and the
        # target Dataset are touched.
        if level_name in out.dims:
            ds_levels = np.asarray(out[level_name].values, dtype=float)
            common_levels = np.array(
                [lev for lev in self.levels_hpa if lev in ds_levels], dtype=float
            )
        else:
            ds_levels = np.array([], dtype=float)
            common_levels = np.array([], dtype=float)

        # Filter self.levels_hpa to those in the Dataset, preserving order.
        vortex_level_idx = [
            int(np.where(self.levels_hpa == lev)[0][0]) for lev in common_levels
        ]

        # u, v: replacement; T, z: additive perturbation.
        atmos_replace = {"u": self.u_bogus_mps, "v": self.v_bogus_mps}
        atmos_additive = {"t": self.T_anom_K}
        if additive_z:
            atmos_additive["z"] = self.Phi_anom_m2_s2

        for canon, bogus_3d in atmos_replace.items():
            native = resolve(canon, var_map)
            if native is None or native not in out.data_vars:
                continue
            bogus_da = xr.DataArray(
                bogus_3d[vortex_level_idx],
                dims=(level_name, lat_name, lon_name),
                coords={
                    level_name: common_levels,
                    lat_name: snap.box_lats,
                    lon_name: snap.box_lons,
                },
            )
            # Restrict the target var to the common levels for the blend,
            # then reindex back so untouched levels are preserved.
            var_da = out[native]
            var_sel = var_da.sel({level_name: common_levels})
            blended = (1.0 - blend_da) * var_sel + blend_da * bogus_da
            # Write blended levels back in place using xarray's coordinate
            # alignment; other levels stay untouched.
            out[native] = xr.where(
                out[native][level_name].isin(common_levels),
                blended.reindex({level_name: var_da[level_name].values}),
                out[native],
            )

        for canon, bogus_3d in atmos_additive.items():
            native = resolve(canon, var_map)
            if native is None or native not in out.data_vars:
                continue
            bogus_da = xr.DataArray(
                bogus_3d[vortex_level_idx],
                dims=(level_name, lat_name, lon_name),
                coords={
                    level_name: common_levels,
                    lat_name: snap.box_lats,
                    lon_name: snap.box_lons,
                },
            )
            var_da = out[native]
            var_sel = var_da.sel({level_name: common_levels})
            added = var_sel + blend_da * bogus_da
            out[native] = xr.where(
                out[native][level_name].isin(common_levels),
                added.reindex({level_name: var_da[level_name].values}),
                out[native],
            )

        # ---- Humidity via eq. 27 (per-level saturation blend) -----------
        q_native = resolve("q", var_map)
        t_native = resolve("t", var_map)
        if (
            q_native is not None
            and q_native in out.data_vars
            and t_native is not None
            and t_native in out.data_vars
            and level_name in out.dims
            and common_levels.size > 0
        ):
            rh_env_da = xr.DataArray(
                self.rh_envelope[vortex_level_idx],
                dims=(level_name, lat_name, lon_name),
                coords={
                    level_name: common_levels,
                    lat_name: snap.box_lats,
                    lon_name: snap.box_lons,
                },
            )
            p_pa_da = xr.DataArray(
                common_levels * HPA_TO_PA,
                dims=(level_name,),
                coords={level_name: common_levels},
            )
            t_sel = out[t_native].sel({level_name: common_levels})
            q_sat = _q_saturation(t_sel, p_pa_da)
            q_target = RH_B * q_sat
            q_sel = out[q_native].sel({level_name: common_levels})
            delta_q = (q_target - q_sel) * rh_env_da
            q_new = q_sel + blend_da * delta_q
            q_full = out[q_native]
            out[q_native] = xr.where(
                q_full[level_name].isin(common_levels),
                q_new.reindex({level_name: q_full[level_name].values}),
                q_full,
            )

        return out
