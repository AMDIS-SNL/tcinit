"""WeatherNext-2 adapter tests.

WN2 shares WN-Graph's Dataset shape and reuses its extract/apply/write_back
via thin wrappers. These tests focus on the WN2-specific delta: the 100 m
wind fields must survive round-trip untouched.
"""

from datetime import datetime

import numpy as np
import xarray as xr

from tcinit.backends import weathernext2_backend as wn2
from tcinit.snapshot import Snapshot
from tcinit.vortex import BogusVortex
from tests.conftest import DEFAULT_LEVELS_HPA, make_best_track


def _make_wn2_ds(
    *,
    n_lat: int = 41,
    n_lon: int = 81,
    lat_range=(-5.0, 35.0),
    lon_range=(100.0, 180.0),
    n_batch: int = 1,
    n_time: int = 2,
):
    """WN-Graph-shape Dataset plus the WN2-only 100 m wind fields."""
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)
    levels = DEFAULT_LEVELS_HPA
    n_lev = levels.size

    surf_shape = (n_batch, n_time, n_lat, n_lon)
    atmos_shape = (n_batch, n_time, n_lev, n_lat, n_lon)

    rs = np.random.RandomState(4)

    def _surf(mean, std):
        return (mean + std * rs.randn(*surf_shape)).astype(np.float32)

    def _atmos(mean, std):
        return (mean + std * rs.randn(*atmos_shape)).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
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
            # WN2-only surface fields.
            "100m_u_component_of_wind": (
                ("batch", "time", "lat", "lon"),
                _surf(2.0, 1.0),
            ),
            "100m_v_component_of_wind": (
                ("batch", "time", "lat", "lon"),
                _surf(1.0, 1.0),
            ),
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


def _build_bv(box_ds):
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


def test_extract_excludes_100m_wind_fields():
    ds = _make_wn2_ds()
    box_ds = wn2.extract_snapshot_dataset(ds, box_center=(15.0, 130.0), box_size=3.0)
    # 100m wind is not in the WEATHERNEXT var map → excluded from extract.
    assert "100m_u_component_of_wind" not in box_ds.data_vars
    assert "100m_v_component_of_wind" not in box_ds.data_vars
    # 10m wind is in the map → included.
    assert "10m_u_component_of_wind" in box_ds.data_vars
    assert "10m_v_component_of_wind" in box_ds.data_vars


def test_write_back_preserves_100m_wind_and_modifies_10m():
    ds = _make_wn2_ds()
    box_center = (15.0, 130.0)
    box_size = 3.0
    box_ds = wn2.extract_snapshot_dataset(ds, box_center=box_center, box_size=box_size)
    bv = _build_bv(box_ds)
    modified = wn2.apply(box_ds, bv)
    out = wn2.write_back(
        ds,
        modified,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
    )
    # 100 m wind: bit-identical to source everywhere.
    np.testing.assert_allclose(
        out["100m_u_component_of_wind"].values,
        ds["100m_u_component_of_wind"].values,
    )
    np.testing.assert_allclose(
        out["100m_v_component_of_wind"].values,
        ds["100m_v_component_of_wind"].values,
    )
    # 10 m wind: modified inside the box.
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
    assert not np.allclose(
        out["10m_u_component_of_wind"].isel(lat=lat_sl, lon=lon_sl).values,
        ds["10m_u_component_of_wind"].isel(lat=lat_sl, lon=lon_sl).values,
    )


def test_write_back_accepts_additional_skip_vars():
    ds = _make_wn2_ds()
    box_ds = wn2.extract_snapshot_dataset(ds, box_center=(15.0, 130.0), box_size=3.0)
    bv = _build_bv(box_ds)
    modified = wn2.apply(box_ds, bv)
    # Ask WN2 to additionally skip MSLP; 100 m wind must still be skipped by default.
    out = wn2.write_back(
        ds,
        modified,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
        skip_vars=("mean_sea_level_pressure",),
    )
    np.testing.assert_allclose(
        out["mean_sea_level_pressure"].values,
        ds["mean_sea_level_pressure"].values,
    )
    np.testing.assert_allclose(
        out["100m_u_component_of_wind"].values,
        ds["100m_u_component_of_wind"].values,
    )
