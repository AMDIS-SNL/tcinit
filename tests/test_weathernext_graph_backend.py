"""WeatherNext-Graph adapter tests.

Synthetic Datasets mimic the DeepMind convention:
    dims (batch, time, [level,] lat, lon)
    ascending lat, lon in [0, 360), short coord names (lat/lon)
    long-form variable names (mean_sea_level_pressure, temperature, etc.)
    2 time slots (matches input_duration="12h" at 6h step)
"""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from tcinit.backends import weathernext_graph_backend as wng
from tcinit.snapshot import Snapshot
from tcinit.ideal_tc_vortex import BogusVortex
from tests.conftest import DEFAULT_LEVELS_HPA, make_best_track


def _make_wn_ds(
    *,
    n_lat: int = 41,
    n_lon: int = 81,
    lat_range=(-5.0, 35.0),  # ascending, WN convention
    lon_range=(100.0, 180.0),
    n_batch: int = 1,
    n_time: int = 2,
):
    """Build a small full-globe-ish WN-Graph inputs Dataset."""
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)
    levels = DEFAULT_LEVELS_HPA
    n_lev = levels.size

    surf_shape = (n_batch, n_time, n_lat, n_lon)
    atmos_shape = (n_batch, n_time, n_lev, n_lat, n_lon)

    rs = np.random.RandomState(3)

    def _surf(mean, std):
        return (mean + std * rs.randn(*surf_shape)).astype(np.float32)

    def _atmos(mean, std):
        return (mean + std * rs.randn(*atmos_shape)).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            # Surface (WN long-form)
            "mean_sea_level_pressure": (
                ("batch", "time", "lat", "lon"),
                _surf(101300.0, 20.0),
            ),
            "2m_temperature": (
                ("batch", "time", "lat", "lon"),
                _surf(298.0, 1.0),
            ),
            "10m_u_component_of_wind": (
                ("batch", "time", "lat", "lon"),
                _surf(0.0, 0.5),
            ),
            "10m_v_component_of_wind": (
                ("batch", "time", "lat", "lon"),
                _surf(0.0, 0.5),
            ),
            # K&C-irrelevant surface (must pass through untouched)
            "total_precipitation_6hr": (
                ("batch", "time", "lat", "lon"),
                _surf(0.001, 0.0005),
            ),
            # Atmospheric (WN long-form)
            "temperature": (
                ("batch", "time", "level", "lat", "lon"),
                _atmos(280.0, 1.0),
            ),
            "u_component_of_wind": (
                ("batch", "time", "level", "lat", "lon"),
                _atmos(0.0, 1.0),
            ),
            "v_component_of_wind": (
                ("batch", "time", "level", "lat", "lon"),
                _atmos(0.0, 1.0),
            ),
            "specific_humidity": (
                ("batch", "time", "level", "lat", "lon"),
                _atmos(5e-3, 0.0),
            ),
            "geopotential": (
                ("batch", "time", "level", "lat", "lon"),
                _atmos(50000.0, 0.0),
            ),
            # K&C-irrelevant atmos (must pass through untouched)
            "vertical_velocity": (
                ("batch", "time", "level", "lat", "lon"),
                _atmos(0.0, 0.01),
            ),
            # Statics (no time/batch dims)
            "geopotential_at_surface": (
                ("lat", "lon"),
                rs.randn(n_lat, n_lon).astype(np.float32),
            ),
            "land_sea_mask": (
                ("lat", "lon"),
                (rs.rand(n_lat, n_lon) > 0.5).astype(np.float32),
            ),
        },
        coords={
            "lat": lats,
            "lon": lons,
            "level": levels,
            "time": np.array(
                [np.datetime64("2020-09-01T00:00"), np.datetime64("2020-09-01T06:00")]
            )[:n_time],
            "batch": np.arange(n_batch),
        },
    )
    return ds


# ---------------------------------------------------------------------------
# extract_snapshot_dataset
# ---------------------------------------------------------------------------


def test_extract_produces_canonical_shape_and_names():
    ds = _make_wn_ds()
    box_center = (15.0, 130.0)
    box_size = 3.0
    out = wng.extract_snapshot_dataset(ds, box_center=box_center, box_size=box_size)
    # Coords renamed to canonical.
    assert "latitude" in out.coords
    assert "longitude" in out.coords
    assert "lat" not in out.coords
    assert "lon" not in out.coords

    # Batch and time collapsed away.
    assert "batch" not in out.dims
    assert "time" not in out.dims

    # K&C-relevant long-form vars present.
    for name in (
        "mean_sea_level_pressure",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "geopotential",
    ):
        assert name in out.data_vars

    # K&C-irrelevant vars dropped (they don't feed Snapshot).
    for excluded in (
        "total_precipitation_6hr",
        "vertical_velocity",
        "geopotential_at_surface",
        "land_sea_mask",
    ):
        assert excluded not in out.data_vars

    # Box coords match the intersection.
    lat_c, lon_c = box_center
    lats = ds["lat"].values
    lons = ds["lon"].values
    expected_lats = lats[(lats >= lat_c - box_size) & (lats <= lat_c + box_size)]
    expected_lons = lons[(lons >= lon_c - box_size) & (lons <= lon_c + box_size)]
    np.testing.assert_allclose(out["latitude"].values, expected_lats)
    np.testing.assert_allclose(out["longitude"].values, expected_lons)


def test_extract_time_idx_selects_correct_slot():
    ds = _make_wn_ds()
    box_center = (15.0, 130.0)
    box_size = 3.0
    out0 = wng.extract_snapshot_dataset(
        ds, box_center=box_center, box_size=box_size, time_idx=0
    )
    out1 = wng.extract_snapshot_dataset(
        ds, box_center=box_center, box_size=box_size, time_idx=1
    )
    # Distinct data at different time slots (values were sampled independently).
    assert not np.allclose(
        out0["mean_sea_level_pressure"].values,
        out1["mean_sea_level_pressure"].values,
    )
    # time_idx=-1 must match time_idx=1 (default = most recent).
    out_default = wng.extract_snapshot_dataset(
        ds, box_center=box_center, box_size=box_size
    )
    np.testing.assert_allclose(
        out_default["mean_sea_level_pressure"].values,
        out1["mean_sea_level_pressure"].values,
    )


def test_extract_raises_on_empty_box():
    ds = _make_wn_ds()
    with pytest.raises(ValueError, match="does not intersect"):
        wng.extract_snapshot_dataset(ds, box_center=(89.0, 0.0), box_size=0.1)


# ---------------------------------------------------------------------------
# apply + Snapshot integration
# ---------------------------------------------------------------------------


def _build_bv(box_ds):
    snap = Snapshot.from_xarray(box_ds, var_map=None)  # try defaults
    # Fall back to explicit var_map: box_ds has long-form names, Snapshot
    # needs to know how to read the surface fields.
    snap = Snapshot.from_xarray(
        box_ds,
        var_map={
            "msl": "mean_sea_level_pressure",
            "2t": "2m_temperature",
            "10u": "10m_u_component_of_wind",
            "10v": "10m_v_component_of_wind",
        },
    )
    lat_c = float(0.5 * (snap.box_lats[0] + snap.box_lats[-1]))
    lon_c = float(0.5 * (snap.box_lons[0] + snap.box_lons[-1]))
    snap.set_storm_center(
        loc=(lat_c, lon_c),
        rdr_km=100.0,
        mslp_env_mean_pa=101300.0,
        t2m_storm_mean_k=300.0,
    )
    bt = make_best_track(datetime(2020, 9, 1, 6), lat_deg=lat_c, lon_deg=lon_c)
    return BogusVortex(snap, bt, datetime(2020, 9, 1, 6)).build()


def test_apply_modifies_box_dataset():
    ds = _make_wn_ds()
    box_center = (15.0, 130.0)
    box_ds = wng.extract_snapshot_dataset(ds, box_center=box_center, box_size=3.0)
    bv = _build_bv(box_ds)
    modified = wng.apply(box_ds, bv)
    # MSLP inside storm should differ from source.
    assert not np.allclose(
        modified["mean_sea_level_pressure"].values,
        box_ds["mean_sea_level_pressure"].values,
    )


# ---------------------------------------------------------------------------
# write_back
# ---------------------------------------------------------------------------


def test_write_back_no_op_roundtrip_preserves_source():
    ds = _make_wn_ds()
    box_center = (15.0, 130.0)
    box_size = 3.0
    box_ds = wng.extract_snapshot_dataset(ds, box_center=box_center, box_size=box_size)
    out = wng.write_back(
        ds,
        box_ds,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
    )
    # Round-trip write of the (unmodified) most-recent time slot puts the same
    # values back into both time slots — so slot 1 is unchanged, but slot 0
    # now equals slot 1. Verify that all K&C-relevant variables match slot 1
    # (the extracted one) in both slots.
    for name in (
        "mean_sea_level_pressure",
        "2m_temperature",
        "temperature",
        "u_component_of_wind",
    ):
        lats = ds["lat"].values
        lons = ds["lon"].values
        lat_c, lon_c = box_center
        lat_mask = (lats >= lat_c - box_size) & (lats <= lat_c + box_size)
        lon_mask = (lons >= lon_c - box_size) & (lons <= lon_c + box_size)
        lat_sl = slice(
            int(np.where(lat_mask)[0].min()), int(np.where(lat_mask)[0].max()) + 1
        )
        lon_sl = slice(
            int(np.where(lon_mask)[0].min()), int(np.where(lon_mask)[0].max()) + 1
        )
        # Outside the box, both slots must be untouched.
        np.testing.assert_allclose(
            out[name].isel(lat=slice(0, lat_sl.start)).values,
            ds[name].isel(lat=slice(0, lat_sl.start)).values,
        )
        # Inside the box across all times: source slot-1 values propagated
        # into every time slot.
        src_slot1 = ds[name].isel(time=-1).isel(lat=lat_sl, lon=lon_sl).values
        for t in range(out.sizes["time"]):
            np.testing.assert_allclose(
                out[name].isel(time=t).isel(lat=lat_sl, lon=lon_sl).values, src_slot1
            )


def test_write_back_after_vortex_localises_and_broadcasts_time():
    ds = _make_wn_ds()
    box_center = (15.0, 130.0)
    box_size = 3.0
    box_ds = wng.extract_snapshot_dataset(ds, box_center=box_center, box_size=box_size)
    bv = _build_bv(box_ds)
    modified = wng.apply(box_ds, bv)
    out = wng.write_back(
        ds,
        modified,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
    )

    lats = ds["lat"].values
    lons = ds["lon"].values
    lat_c, lon_c = box_center
    lat_mask = (lats >= lat_c - box_size) & (lats <= lat_c + box_size)
    lon_mask = (lons >= lon_c - box_size) & (lons <= lon_c + box_size)
    lat_out = ~lat_mask
    lon_out = ~lon_mask

    # Cells strictly outside the box (lat or lon out) are untouched everywhere.
    for name in ("mean_sea_level_pressure", "temperature"):
        np.testing.assert_allclose(
            out[name].values[..., lat_out, :],
            ds[name].values[..., lat_out, :],
        )
        np.testing.assert_allclose(
            out[name].values[..., :, lon_out],
            ds[name].values[..., :, lon_out],
        )

    # Both time slots inside the box are identical (Aurora-style broadcast).
    lat_sl = slice(
        int(np.where(lat_mask)[0].min()), int(np.where(lat_mask)[0].max()) + 1
    )
    lon_sl = slice(
        int(np.where(lon_mask)[0].min()), int(np.where(lon_mask)[0].max()) + 1
    )
    for name in ("mean_sea_level_pressure", "temperature"):
        slot0 = out[name].isel(time=0).isel(lat=lat_sl, lon=lon_sl).values
        slot1 = out[name].isel(time=1).isel(lat=lat_sl, lon=lon_sl).values
        np.testing.assert_allclose(slot0, slot1)

    # K&C-irrelevant vars must be bit-identical to the source.
    for untouched in (
        "total_precipitation_6hr",
        "vertical_velocity",
        "geopotential_at_surface",
        "land_sea_mask",
    ):
        np.testing.assert_allclose(out[untouched].values, ds[untouched].values)


def test_write_back_time_slots_kwarg_writes_only_specified_slot():
    ds = _make_wn_ds()
    box_center = (15.0, 130.0)
    box_size = 3.0
    box_ds = wng.extract_snapshot_dataset(ds, box_center=box_center, box_size=box_size)
    bv = _build_bv(box_ds)
    modified = wng.apply(box_ds, bv)
    out = wng.write_back(
        ds,
        modified,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
        time_slots=(1,),
    )

    lats = ds["lat"].values
    lons = ds["lon"].values
    lat_c, lon_c = box_center
    lat_mask = (lats >= lat_c - box_size) & (lats <= lat_c + box_size)
    lon_mask = (lons >= lon_c - box_size) & (lons <= lon_c + box_size)
    lat_sl = slice(
        int(np.where(lat_mask)[0].min()), int(np.where(lat_mask)[0].max()) + 1
    )
    lon_sl = slice(
        int(np.where(lon_mask)[0].min()), int(np.where(lon_mask)[0].max()) + 1
    )

    # Slot 0 unchanged (both inside and outside the box).
    for name in ("mean_sea_level_pressure", "temperature"):
        np.testing.assert_allclose(
            out[name].isel(time=0).values, ds[name].isel(time=0).values
        )
        # Slot 1 changed inside the box.
        assert not np.allclose(
            out[name].isel(time=1).isel(lat=lat_sl, lon=lon_sl).values,
            ds[name].isel(time=1).isel(lat=lat_sl, lon=lon_sl).values,
        )


def test_write_back_skip_vars_leaves_those_untouched():
    ds = _make_wn_ds()
    box_center = (15.0, 130.0)
    box_size = 3.0
    box_ds = wng.extract_snapshot_dataset(ds, box_center=box_center, box_size=box_size)
    bv = _build_bv(box_ds)
    modified = wng.apply(box_ds, bv)
    out = wng.write_back(
        ds,
        modified,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
        skip_vars=("mean_sea_level_pressure",),
    )
    # MSLP untouched everywhere despite modified_ds containing it.
    np.testing.assert_allclose(
        out["mean_sea_level_pressure"].values,
        ds["mean_sea_level_pressure"].values,
    )
    # But 2m_temperature (still in the write set) is modified.
    assert not np.allclose(out["2m_temperature"].values, ds["2m_temperature"].values)
