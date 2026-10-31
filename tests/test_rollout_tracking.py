"""Tests for tcinit.rollout_tracking: storm following and bogus construction."""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from tcinit.constants import HPA_TO_PA
from tcinit.rollout_tracking import build_bogus, snapshot_from_box, track_rollout
from tests.conftest import DEFAULT_LEVELS_HPA, make_best_track

START = datetime(2023, 5, 26, 6)


def _model_state(center, *, depth_pa=4000.0, sigma_deg=1.0, spacing=0.25):
    """A 'global' (here 0-30N, 120-150E) canonical state with a Gaussian low."""
    lats = np.arange(30.0, -spacing / 2, -spacing)  # descending, like most models
    lons = np.arange(120.0, 150.0 + spacing / 2, spacing)
    lat2, lon2 = np.meshgrid(lats, lons, indexing="ij")
    r2 = (lat2 - center[0]) ** 2 + (lon2 - center[1]) ** 2
    low = np.exp(-r2 / (2.0 * sigma_deg**2))
    shape3d = (DEFAULT_LEVELS_HPA.size,) + lat2.shape
    rs = np.random.RandomState(0)
    return xr.Dataset(
        data_vars={
            "msl": (("latitude", "longitude"), 101300.0 - depth_pa * low),
            "2t": (("latitude", "longitude"), 300.0 + 0.1 * rs.randn(*lat2.shape)),
            # Wind peaks on a ring around the low, so rdr is finite.
            "10u": (("latitude", "longitude"), 30.0 * np.sqrt(r2) * low),
            "10v": (("latitude", "longitude"), 30.0 * np.sqrt(r2) * low),
            "t": (("level", "latitude", "longitude"), 280.0 + rs.randn(*shape3d)),
            "u": (("level", "latitude", "longitude"), rs.randn(*shape3d)),
            "v": (("level", "latitude", "longitude"), rs.randn(*shape3d)),
            "q": (("level", "latitude", "longitude"), 5e-3 * np.ones(shape3d)),
            "z": (("level", "latitude", "longitude"), 50000.0 * np.ones(shape3d)),
        },
        coords={"latitude": lats, "longitude": lons, "level": DEFAULT_LEVELS_HPA},
    )


def _extract_box(state, center, half=5.0):
    lat, lon = center
    return state.sel(
        latitude=slice(lat + half, lat - half), longitude=slice(lon - half, lon + half)
    )


def _track(n_steps=4):
    """Low starts at (15, 135) and moves 1 deg north-west per step."""
    return [(15.0 + k, 135.0 - k) for k in range(n_steps)]


def test_track_rollout_follows_moving_low():
    centers = _track()
    states = (_model_state(c) for c in centers)  # generator: consumed lazily
    rollout = track_rollout(
        states,
        _extract_box,
        cyclone_id="WP022023",
        start_time=START,
        initial_center=(14.0, 136.0),  # off by ~1.4 deg, as a best-track fix might be
        intervened=True,
        verbose=False,
    )
    assert rollout.cyclone_id == "WP022023"
    assert rollout.intervened is True
    assert rollout.valid_times == [START + timedelta(hours=6 * k) for k in range(4)]
    for snap, (lat, lon) in zip(rollout, centers):
        assert snap.storm_computed
        assert snap.loc[0] == pytest.approx(lat, abs=0.25)
        assert snap.loc[1] == pytest.approx(lon, abs=0.25)
        assert snap.mslp_min == pytest.approx(101300.0 - 4000.0, abs=50.0)
        # Atmospheric fields attached for the warm-core / MSE diagnostics.
        assert snap.src_t.shape == (DEFAULT_LEVELS_HPA.size,) + snap._msl.shape
        assert snap.src_q is not None and snap.src_z is not None


def test_track_rollout_diagnostics_have_atmospheric_fields():
    rollout = track_rollout(
        (_model_state(c) for c in _track(2)),
        _extract_box,
        cyclone_id="WP022023",
        start_time=START,
        initial_center=(15.0, 135.0),
        verbose=False,
    )
    diag = rollout.diagnostics(rollout[0])
    assert diag["vorticity850"] is not None
    assert np.isfinite(diag["T_avg_200_500_env_mean"])
    assert diag["mse"] is not None


def test_track_rollout_keeps_going_after_a_lost_step():
    centers = _track()
    states = [_model_state(c) for c in centers]
    states[1]["msl"][:] = np.nan  # detection fails at step 1
    rollout = track_rollout(
        states,
        _extract_box,
        cyclone_id="WP022023",
        start_time=START,
        initial_center=centers[0],
        verbose=False,
    )
    assert [s.storm_computed for s in rollout] == [True, False, True, True]
    # Step 2 searched around step 0's position (2 deg away) and still found it.
    assert rollout[2].loc[0] == pytest.approx(centers[2][0], abs=0.25)
    assert rollout[1].valid_time == START + timedelta(hours=6)


def test_snapshot_from_box_raises_when_no_storm_in_disk():
    box = _extract_box(_model_state((15.0, 135.0)), (15.0, 135.0))
    box["msl"][:] = np.nan
    with pytest.raises(ValueError):
        snapshot_from_box(box, valid_time=START, search_center=(15.0, 135.0))


def test_build_bogus_deepens_weak_low_toward_best_track():
    center = (15.0, 135.0)
    box = _extract_box(_model_state(center, depth_pa=800.0), center)  # 1005 hPa
    bt = make_best_track(START, mslp_hpa=950.0, lat_deg=center[0], lon_deg=center[1])
    snap, bv, modified = build_bogus(box, bt, START, search_center=center)
    assert snap.storm_computed
    assert bv.p_c == pytest.approx(950.0 * HPA_TO_PA)
    assert float(modified["msl"].min()) < float(box["msl"].min()) - 20.0 * HPA_TO_PA
    # Same grid in and out, so write_back can slot it into the native state.
    np.testing.assert_array_equal(modified.latitude, box.latitude)
    np.testing.assert_array_equal(modified.longitude, box.longitude)


# ---------------------------------------------------------------------------
# Backend-native variable names (var_map) and dim order
# ---------------------------------------------------------------------------


def _longform(ds):
    """Rename canonical variables to WeatherNext's long-form names."""
    from tcinit.naming import WEATHERNEXT_VAR_MAP

    return ds.rename({c: n for c, n in WEATHERNEXT_VAR_MAP.items() if c in ds})


def test_track_rollout_with_longform_names():
    from tcinit.naming import WEATHERNEXT_VAR_MAP

    centers = _track(3)
    rollout = track_rollout(
        (_longform(_model_state(c)) for c in centers),
        _extract_box,
        cyclone_id="WP022023",
        start_time=START,
        initial_center=centers[0],
        var_map=WEATHERNEXT_VAR_MAP,
        verbose=False,
    )
    for snap, (lat, lon) in zip(rollout, centers):
        assert snap.storm_computed
        assert snap.loc[0] == pytest.approx(lat, abs=0.25)
        assert snap.src_t is not None and snap.src_z is not None


def test_snapshot_from_box_needs_var_map_for_longform_names():
    box = _extract_box(_longform(_model_state((15.0, 135.0))), (15.0, 135.0))
    with pytest.raises(ValueError, match="no surface fields"):
        snapshot_from_box(box, valid_time=START, search_center=(15.0, 135.0))


@pytest.mark.parametrize("use_backend_apply", [True, False])
def test_build_bogus_with_longform_names(use_backend_apply):
    from tcinit.backends import weathernext_graph_backend
    from tcinit.naming import WEATHERNEXT_VAR_MAP

    center = (15.0, 135.0)
    box = _extract_box(_longform(_model_state(center, depth_pa=800.0)), center)
    bt = make_best_track(START, mslp_hpa=950.0, lat_deg=center[0], lon_deg=center[1])
    apply = weathernext_graph_backend.apply if use_backend_apply else None
    _, _, modified = build_bogus(
        box, bt, START, search_center=center, var_map=WEATHERNEXT_VAR_MAP, apply=apply
    )
    msl = WEATHERNEXT_VAR_MAP["msl"]
    assert set(modified.data_vars) == set(box.data_vars)  # names kept
    assert float(modified[msl].min()) < float(box[msl].min()) - 20.0 * HPA_TO_PA


def test_track_rollout_accepts_level_lon_lat_order():
    """NeuralGCM boxes come as (level, longitude, latitude).

    Off-centre start and a non-square box, so a lat/lon swap would show.
    """
    centers = _track(2)
    rollout = track_rollout(
        (_model_state(c).transpose("level", "longitude", "latitude") for c in centers),
        lambda state, center: _extract_box(state, center).isel(longitude=slice(0, -8)),
        cyclone_id="WP022023",
        start_time=START,
        initial_center=(14.0, 136.0),
        verbose=False,
    )
    for snap, (lat, lon) in zip(rollout, centers):
        assert snap.loc[0] == pytest.approx(lat, abs=0.25)
        assert snap.loc[1] == pytest.approx(lon, abs=0.25)
        assert snap.src_t.shape == (DEFAULT_LEVELS_HPA.size,) + snap._msl.shape
        assert snap._msl.shape == (snap.box_lats.size, snap.box_lons.size)
