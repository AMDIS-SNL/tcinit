"""Kwon & Cheong bogus vortex physics + xarray-apply tests."""

from datetime import datetime

import numpy as np
import pytest

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
