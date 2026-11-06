"""Continuity / hydrostatic diagnostics and the WeatherNext omega blend."""

from __future__ import annotations

import numpy as np
import pytest

from tcinit.backends import weathernext_graph_backend as wng
from tcinit.constants import EARTH_RADIUS_A, HPA_TO_PA, R_D
from tcinit.diagnostics.continuity import (
    divergence_latlon,
    hydrostatic_residual,
    omega_from_continuity,
    virtual_temperature,
)
from tests.conftest import DEFAULT_LEVELS_HPA
from tests.test_weathernext_graph_backend import _build_bv, _make_wn_ds

LATS = np.linspace(10.0, 20.0, 21)
LONS = np.linspace(125.0, 135.0, 21)


def _uniform_divergence(k_per_level):
    """u = k a cos(phi) (lambda - lambda_0), v = 0  =>  div = k exactly."""
    lat = np.deg2rad(LATS)[:, None]
    lam = np.deg2rad(LONS - LONS.mean())[None, :]
    u = np.asarray(k_per_level)[:, None, None] * EARTH_RADIUS_A * np.cos(lat) * lam
    return u, np.zeros_like(u)


def test_divergence_of_linear_field_is_exact():
    u, v = _uniform_divergence([2e-5])
    np.testing.assert_allclose(
        divergence_latlon(u[0], v[0], LATS, LONS), 2e-5, rtol=1e-10
    )


def test_omega_constant_divergence_matches_analytic():
    k = 1e-5
    levels = DEFAULT_LEVELS_HPA
    u, v = _uniform_divergence(np.full(levels.size, k))
    omega, residual = omega_from_continuity(
        u, v, LATS, LONS, levels, correct_surface=False
    )
    p = levels * HPA_TO_PA
    expected = -k * (p - p.min())
    np.testing.assert_allclose(omega[:, 10, 10], expected, rtol=1e-10)
    np.testing.assert_allclose(residual, expected[-1], rtol=1e-10)


def test_omega_level_order_does_not_matter():
    levels = DEFAULT_LEVELS_HPA
    u, v = _uniform_divergence(np.linspace(-1e-5, 2e-5, levels.size))
    om_asc, _ = omega_from_continuity(u, v, LATS, LONS, levels)
    rev = slice(None, None, -1)
    om_desc, _ = omega_from_continuity(u[rev], v[rev], LATS, LONS, levels[rev])
    np.testing.assert_allclose(om_desc[rev], om_asc)


def test_obrien_correction_hits_target_and_masks_underground():
    levels = DEFAULT_LEVELS_HPA
    u, v = _uniform_divergence(np.full(levels.size, -3e-5))
    p_sfc = np.full((LATS.size, LONS.size), 960.0 * HPA_TO_PA)
    target = np.full_like(p_sfc, -0.4)
    omega, residual = omega_from_continuity(
        u, v, LATS, LONS, levels, p_sfc_pa=p_sfc, omega_sfc_target=target
    )
    i_top = int(np.argmin(levels))
    np.testing.assert_allclose(omega[i_top], 0.0, atol=1e-12)
    assert np.all(np.isnan(omega[levels > 960.0]))
    # Uncorrected surface value is 3e-5 * (96000 - 5000); corrected profile
    # extrapolated to the surface equals the target.
    np.testing.assert_allclose(residual, 3e-5 * (96000.0 - 5000.0) + 0.4, rtol=1e-10)
    i_925 = int(np.where(levels == 925.0)[0][0])
    slope = (omega[i_925] - omega[i_top]) / ((925.0 - 50.0) * HPA_TO_PA)
    np.testing.assert_allclose(
        omega[i_925] + slope * (960.0 - 925.0) * HPA_TO_PA, target, rtol=1e-10
    )


def test_hydrostatic_residual_zero_for_balanced_column_and_detects_offset():
    levels = DEFAULT_LEVELS_HPA[::-1]  # surface first, to exercise sorting
    p = levels * HPA_TO_PA
    t = (300.0 - 60.0 * (1.0 - p / p.max()))[:, None, None] * np.ones((1, 3, 4))
    q = (0.015 * (p / p.max()) ** 3)[:, None, None] * np.ones((1, 3, 4))
    tv = virtual_temperature(t, q)
    z = np.zeros_like(t)
    for k in range(1, levels.size):  # integrate upward with layer-mean T_v
        z[k] = z[k - 1] + R_D * 0.5 * (tv[k] + tv[k - 1]) * np.log(p[k - 1] / p[k])
    _, res = hydrostatic_residual(t, z, q, levels)
    np.testing.assert_allclose(res, 0.0, atol=1e-9)
    _, res_warm = hydrostatic_residual(t + 1.0, z, q, levels)
    # A uniform +1 K with unchanged Phi is a ~-1 K (times 1 + 0.608 q) error.
    assert np.all((res_warm < -1.0) & (res_warm > -1.02))


# ---------------------------------------------------------------------------
# WeatherNext omega blend
# ---------------------------------------------------------------------------


def _wn_box(include_omega=True):
    ds = _make_wn_ds()
    box = wng.extract_snapshot_dataset(
        ds, box_center=(15.0, 130.0), box_size=8.0, include_omega=include_omega
    )
    return ds, box


def test_extract_omega_is_opt_in():
    _, box = _wn_box(include_omega=False)
    assert "vertical_velocity" not in box.data_vars
    _, box = _wn_box(include_omega=True)
    assert "vertical_velocity" in box.data_vars


def test_blend_omega_requires_omega_in_box():
    _, box = _wn_box(include_omega=False)
    bv = _build_bv(box)
    with pytest.raises(ValueError, match="include_omega"):
        wng.apply(box, bv, blend_omega=True)


def test_blend_omega_is_local_replacement_toward_kc_omega():
    _, box = _wn_box()
    bv = _build_bv(box)
    plain = wng.apply(box, bv)
    blended = wng.apply(box, bv, blend_omega=True)
    # Without blend_omega, omega passes through.
    np.testing.assert_array_equal(
        plain["vertical_velocity"].values, box["vertical_velocity"].values
    )
    dims = ("level", "latitude", "longitude")
    got = blended["vertical_velocity"].transpose(*dims).values
    src = box["vertical_velocity"].transpose(*dims).values
    b = np.asarray(bv.blend)
    np.testing.assert_allclose(got[:, b == 0], src[:, b == 0])
    expected = (1.0 - b) * src + b * bv.omega_bogus_pa_s
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    # Other fields are the same as without the omega blend.
    for name in ("temperature", "u_component_of_wind", "geopotential"):
        np.testing.assert_array_equal(blended[name].values, plain[name].values)


def test_write_back_carries_blended_omega():
    ds, box = _wn_box()
    bv = _build_bv(box)
    blended = wng.apply(box, bv, blend_omega=True)
    out = wng.write_back(
        ds,
        blended,
        box_lats=box.latitude.values,
        box_lons=box.longitude.values,
        time_slots=(1,),
    )
    got = (
        out["vertical_velocity"]
        .isel(batch=0, time=1)
        .sel(lat=box.latitude.values, lon=box.longitude.values)
    )
    np.testing.assert_allclose(
        got.transpose("level", "lat", "lon").values,
        blended["vertical_velocity"].transpose("level", "latitude", "longitude").values,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        out["vertical_velocity"].isel(time=0).values,
        ds["vertical_velocity"].isel(time=0).values,
    )


def test_omega_diagnostics_keys_and_finite():
    _, box = _wn_box()
    bv = _build_bv(box)
    blended = wng.apply(box, bv, blend_omega=True)
    rep = wng.omega_diagnostics(box, blended, bv)
    assert set(rep) == {
        "omega_increment",
        "surface_residual",
        "continuity_after",
        "hydrostatic_before",
        "hydrostatic_after",
        "nonhydrostatic_ratio",
    }
    for stats in rep.values():
        assert np.isfinite(stats["max_abs"]) and np.isfinite(stats["rms"])
