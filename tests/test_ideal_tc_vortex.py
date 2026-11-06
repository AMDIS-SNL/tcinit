"""Kwon & Cheong bogus vortex physics + xarray-apply tests."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from tcinit.constants import HPA_TO_PA, KNOTS_TO_MPS
from tcinit.ideal_tc_vortex import (
    BETA_1_DEG,
    BogusVortex,
    K0_FRICTION,
    R_0_KM,
    _solve_holland_A,
    _table_2_p_a_pa,
)

# ---------------------------------------------------------------------------
# Static physics helpers (no fixtures required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_c_hpa,expected_hpa",
    [
        (930.0, 120.0),
        (944.9, 120.0),
        (945.0, 130.0),
        (959.9, 130.0),
        (960.0, 140.0),
        (979.9, 140.0),
        (980.0, 150.0),
        (1000.0, 150.0),
    ],
)
def test_table_2_p_a_step_lookup(p_c_hpa, expected_hpa):
    assert _table_2_p_a_pa(p_c_hpa) == pytest.approx(expected_hpa * HPA_TO_PA)


def test_solve_holland_A_converges_at_realistic_inputs():
    # Ambient p_n = 1013 hPa; central p_c = 950 hPa.
    p_n = 1013.0 * HPA_TO_PA
    p_c = 950.0 * HPA_TO_PA
    dp = p_n - p_c
    rho = 1.15
    V_g_max = 60.0
    B = rho * np.e * V_g_max**2 / dp

    V_g30 = (30.0 * KNOTS_TO_MPS / K0_FRICTION) * np.cos(np.deg2rad(BETA_1_DEG))
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(15.0))
    A = _solve_holland_A(
        B=B, r_30_km=250.0, V_g30=V_g30, dp_pa=dp, rho=rho, f_coriolis=f
    )
    assert np.isfinite(A) and A > 0
    # r_m from Holland should live somewhere in the clamp band [40, 100] km.
    r_m = R_0_KM * A ** (1.0 / B)
    assert 10.0 < r_m < 500.0  # sanity band; final clamp happens in build()


# ---------------------------------------------------------------------------
# build() — smoke + finite-values assertions
# ---------------------------------------------------------------------------


def test_build_populates_all_fields(snapshot, best_track, target_time):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    n_lat = snapshot.box_lats.size
    n_lon = snapshot.box_lons.size
    n_lev = snapshot.levels.size

    assert bv.p_sfc_bogus_pa.shape == (n_lat, n_lon)
    assert bv.t2m_bogus_K.shape == (n_lat, n_lon)
    assert bv.u10_bogus_mps.shape == (n_lat, n_lon)
    assert bv.v10_bogus_mps.shape == (n_lat, n_lon)
    assert bv.T_anom_K.shape == (n_lev, n_lat, n_lon)
    assert bv.Phi_anom_m2_s2.shape == (n_lev, n_lat, n_lon)
    assert bv.u_bogus_mps.shape == (n_lev, n_lat, n_lon)
    assert bv.v_bogus_mps.shape == (n_lev, n_lat, n_lon)
    assert bv.rh_envelope.shape == (n_lev, n_lat, n_lon)
    assert bv.blend.shape == (n_lat, n_lon)

    for arr in (
        bv.p_sfc_bogus_pa,
        bv.t2m_bogus_K,
        bv.u10_bogus_mps,
        bv.v10_bogus_mps,
        bv.T_anom_K,
        bv.Phi_anom_m2_s2,
        bv.u_bogus_mps,
        bv.v_bogus_mps,
        bv.rh_envelope,
        bv.blend,
    ):
        assert np.isfinite(arr).all(), "bogus field has non-finite entries"


def test_build_blend_is_one_at_centre_zero_outside(snapshot, best_track, target_time):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    # Blend at storm centre cell should be 1 (r=0 < R_v).
    n_lat, n_lon = bv.blend.shape
    i, j = n_lat // 2, n_lon // 2
    assert bv.blend[i, j] == pytest.approx(1.0, abs=1e-6)
    # Outside storm_radius_multiplier * rdr the blend is 0.
    assert bv.blend.max() == pytest.approx(1.0, abs=1e-6)
    assert (bv.blend[~bv.storm_mask] == 0.0).all()


def test_build_mslp_drops_toward_central_pressure(snapshot, best_track, target_time):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    n_lat, n_lon = bv.p_sfc_bogus_pa.shape
    i, j = n_lat // 2, n_lon // 2
    # At the centre p_sfc should be within a couple of hPa of p_c.
    assert bv.p_sfc_bogus_pa[i, j] == pytest.approx(bv.p_c, rel=5e-3)
    # Far from the centre p_sfc should approach p_n.
    assert bv.p_sfc_bogus_pa[0, 0] == pytest.approx(bv.p_n, rel=1e-2)


# ---------------------------------------------------------------------------
# apply() — Dataset-level blending
# ---------------------------------------------------------------------------


def test_apply_leaves_far_field_untouched(
    canonical_ds, snapshot, best_track, target_time
):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    ds_out = bv.apply(canonical_ds)
    # Corner cell is outside storm_radius; should equal original for msl.
    orig_msl = canonical_ds["msl"].values
    new_msl = ds_out["msl"].values
    assert new_msl[0, 0] == pytest.approx(orig_msl[0, 0])


def test_apply_lowers_msl_at_centre(canonical_ds, snapshot, best_track, target_time):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    ds_out = bv.apply(canonical_ds)
    n_lat, n_lon = canonical_ds.sizes["latitude"], canonical_ds.sizes["longitude"]
    i, j = n_lat // 2, n_lon // 2
    orig = float(canonical_ds["msl"].values[i, j])
    new = float(ds_out["msl"].values[i, j])
    assert new < orig - 1000.0  # dropped by at least 10 hPa


@pytest.mark.parametrize("var,offset", [("t", 5.0), ("z", -500.0)])
def test_apply_replaces_analysed_storm(
    canonical_ds, snapshot, best_track, target_time, var, offset
):
    """Where blend == 1, t and z become env + K&C anomaly whatever was there."""
    bv = BogusVortex(snapshot, best_track, target_time).build()
    stormy = canonical_ds.copy(deep=True)
    storm = xr.DataArray(
        snapshot.storm_mask.astype(float), dims=("latitude", "longitude")
    )
    stormy[var] = stormy[var] + offset * storm  # an analysed storm, storm only

    out_plain = bv.apply(canonical_ds)
    out_stormy = bv.apply(stormy)

    core = bv.blend == 1.0
    assert core.any()
    v_plain = out_plain[var].transpose("level", "latitude", "longitude").values
    v_stormy = out_stormy[var].transpose("level", "latitude", "longitude").values
    np.testing.assert_allclose(v_stormy[:, core], v_plain[:, core])

    anom = {"t": bv.T_anom_K, "z": bv.Phi_anom_m2_s2}[var]
    env = canonical_ds[var].values[:, snapshot.env_mask].mean(axis=1)
    np.testing.assert_allclose(
        v_plain[:, core], env[:, None] + anom[:, core], rtol=1e-6
    )


def test_apply_blend_z_false_leaves_z_untouched(
    canonical_ds, snapshot, best_track, target_time
):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    out = bv.apply(canonical_ds, blend_z=False)
    np.testing.assert_array_equal(out["z"].values, canonical_ds["z"].values)


def test_apply_raises_without_environment(
    canonical_ds, snapshot, best_track, target_time
):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    snapshot.env_mask = np.zeros_like(snapshot.env_mask)
    with pytest.raises(ValueError, match="environment mask is empty"):
        bv.apply(canonical_ds)


def test_apply_skips_missing_variables(canonical_ds, snapshot, best_track, target_time):
    """Dropping a canonical var from the Dataset should not raise."""
    bv = BogusVortex(snapshot, best_track, target_time).build()
    ds_no_z = canonical_ds.drop_vars(["z"])
    ds_out = bv.apply(ds_no_z)
    assert "z" not in ds_out.data_vars


def test_apply_var_map_supports_native_names(
    canonical_ds, snapshot, best_track, target_time
):
    """Rename q -> specific_humidity via a var_map and ensure apply targets it."""
    ds_renamed = canonical_ds.rename({"q": "specific_humidity", "t": "temperature"})
    snap = snapshot  # snapshot was built from the canonical-named ds; that's fine
    bv = BogusVortex(snap, best_track, target_time).build()
    var_map = {
        "msl": "msl",
        "2t": "2t",
        "10u": "10u",
        "10v": "10v",
        "t": "temperature",
        "u": "u",
        "v": "v",
        "q": "specific_humidity",
        "z": "z",
    }
    ds_out = bv.apply(ds_renamed, var_map=var_map)
    # Both renamed vars should have been touched (values differ from originals).
    assert not np.allclose(
        ds_out["specific_humidity"].values, ds_renamed["specific_humidity"].values
    )
    assert not np.allclose(
        ds_out["temperature"].values, ds_renamed["temperature"].values
    )


# ---------------------------------------------------------------------------
# Point (unstructured-mesh) mode
# ---------------------------------------------------------------------------


def _box_as_points(ds):
    """The box's cells as an ``ncol`` list, with 1-D lat/lon variables."""
    lat2, lon2 = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing="ij")
    data = {}
    for name, da in ds.data_vars.items():
        da = da.transpose(..., "latitude", "longitude")
        flat = da.values.reshape(da.shape[:-2] + (-1,))
        data[name] = (da.dims[:-2] + ("ncol",), flat)
    data["lat"] = (("ncol",), lat2.ravel())
    data["lon"] = (("ncol",), lon2.ravel())
    return xr.Dataset(data, coords={"level": ds.level.values})


def test_points_mode_matches_box_mode(canonical_ds, snapshot, best_track, target_time):
    bv_box = BogusVortex(snapshot, best_track, target_time).build()
    pts = _box_as_points(canonical_ds)
    bv_pts = BogusVortex(snapshot, best_track, target_time).build(
        points=(pts["lat"].values, pts["lon"].values)
    )
    for name in ("p_sfc_bogus_pa", "T_anom_K", "u_bogus_mps", "rh_envelope", "blend"):
        a, b = getattr(bv_box, name), getattr(bv_pts, name)
        np.testing.assert_allclose(b, a.reshape(a.shape[:-2] + (-1,)), err_msg=name)
    np.testing.assert_array_equal(
        bv_pts.environment_mask(), np.asarray(snapshot.env_mask).ravel()
    )

    out_box = _box_as_points(bv_box.apply(canonical_ds))
    out_pts = bv_pts.apply(pts, lat_name="lat", lon_name="lon")
    for name in ("msl", "t", "u", "v", "q", "z"):
        np.testing.assert_allclose(
            out_pts[name].transpose(*out_box[name].dims).values,
            out_box[name].values,
            rtol=1e-12,
            err_msg=name,
        )


def test_points_mode_rejects_misaligned_dataset(
    canonical_ds, snapshot, best_track, target_time
):
    pts = _box_as_points(canonical_ds)
    bv = BogusVortex(snapshot, best_track, target_time).build(
        points=(pts["lat"].values, pts["lon"].values)
    )
    shuffled = pts.isel(ncol=np.roll(np.arange(pts.sizes["ncol"]), 1))
    with pytest.raises(ValueError, match="latitudes do not match"):
        bv.apply(shuffled, lat_name="lat", lon_name="lon")


# ---------------------------------------------------------------------------
# Omega implied by the K&C radial wind
# ---------------------------------------------------------------------------


def test_omega_fields_have_level_layout(snapshot, best_track, target_time):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    assert bv.omega_radial_pa_s.shape == bv.v_r_radial_mps.shape
    assert bv.omega_bogus_pa_s.shape == bv.u_bogus_mps.shape
    assert np.all(np.isfinite(bv.omega_bogus_pa_s))


def test_omega_from_radial_wind_boundary_values(snapshot, best_track, target_time):
    from tcinit.ideal_tc_vortex import N_SIGMA_OMEGA, _omega_from_radial_wind

    bv = BogusVortex(snapshot, best_track, target_time).build()
    r_m = bv.r_radial_km * 1000.0
    ps = bv.p_sfc_radial_pa
    sigma = np.linspace(0.0, 1.0, N_SIGMA_OMEGA)
    # Column-balanced synthetic profile: inflow below sigma 0.8, outflow above.
    shape = np.where(sigma < 0.8, 1.0, -4.0)[:, None]
    v_r = shape * (r_m / 1e5) * np.exp(-r_m / 2e5)[None]
    v_r = v_r - np.trapezoid(v_r, sigma, axis=0)[None]  # int v_r dsigma = 0
    omega = _omega_from_radial_wind(sigma, r_m, ps, v_r)
    np.testing.assert_allclose(omega[0], 0.0, atol=1e-12)
    # sigma_dot(1) = 0, so the surface value is the kinematic v_r dps/dr.
    np.testing.assert_allclose(
        omega[-1], v_r[-1] * np.gradient(ps, r_m), rtol=1e-3, atol=1e-6
    )


def test_kc_omega_is_ascent_in_core(snapshot, best_track, target_time):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    k500 = int(np.argmin(np.abs(np.asarray(bv.levels_hpa) - 500.0)))
    core = bv.r_radial_km < 2.0 * bv.R_v_km
    assert bv.omega_radial_pa_s[k500, core].min() < -0.1


# ---------------------------------------------------------------------------
# Weak-storm Holland-A fallback
# ---------------------------------------------------------------------------


def test_weak_storm_holland_failure_falls_back_to_rm_clamp(snapshot, target_time):
    from tests.conftest import make_best_track

    # Mawar's 2023-05-20 06Z TD fix: 30 kt, 994 hPa. The 30-kt radius is at
    # the wind maximum, so the r_30 Newton solve does not converge.
    bt = make_best_track(target_time, mslp_hpa=994.0, vmax_kt=30.0)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        bv = BogusVortex(snapshot, bt, target_time).build()
    assert bv.r_m_holland_km == pytest.approx(100.0)
    assert np.isfinite(bv.u_bogus_mps).all()


def test_strong_storm_holland_failure_still_raises(
    snapshot, best_track, target_time, monkeypatch
):
    import tcinit.ideal_tc_vortex as kc

    def _fail(**kwargs):
        raise ValueError("forced")

    monkeypatch.setattr(kc, "_solve_holland_A", _fail)
    with pytest.raises(ValueError, match="forced"):
        BogusVortex(snapshot, best_track, target_time).build()
