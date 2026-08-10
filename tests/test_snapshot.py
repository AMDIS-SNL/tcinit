"""Snapshot construction and storm-detection tests."""

import numpy as np
import pytest

from tcinit.snapshot import Snapshot


def test_from_xarray_populates_grid_and_levels(canonical_ds):
    snap = Snapshot.from_xarray(canonical_ds)
    assert snap.box_lats.size == canonical_ds.sizes["latitude"]
    assert snap.box_lons.size == canonical_ds.sizes["longitude"]
    assert snap.levels.size == canonical_ds.sizes["level"]
    assert snap.has_surface  # canonical_ds includes msl/2t/10u/10v


def test_from_xarray_no_surface_when_absent():
    import xarray as xr

    ds = xr.Dataset(
        data_vars={
            "t": (("level", "latitude", "longitude"), np.zeros((3, 5, 5))),
        },
        coords={
            "level": [500.0, 850.0, 1000.0],
            "latitude": np.linspace(10, 20, 5),
            "longitude": np.linspace(125, 135, 5),
        },
    )
    snap = Snapshot.from_xarray(ds)
    assert not snap.has_surface


def test_set_storm_center_populates_masks(canonical_ds):
    snap = Snapshot.from_xarray(canonical_ds)
    lat_c = float(0.5 * (snap.box_lats[0] + snap.box_lats[-1]))
    lon_c = float(0.5 * (snap.box_lons[0] + snap.box_lons[-1]))
    snap.set_storm_center(
        loc=(lat_c, lon_c),
        rdr_km=100.0,
        mslp_env_mean_pa=101300.0,
        t2m_storm_mean_k=300.0,
    )
    assert snap.storm_computed
    assert snap.storm_mask.dtype == bool
    assert snap.storm_mask.sum() > 0
    assert (snap.storm_mask & snap.env_mask).sum() == 0
    assert (snap.storm_mask | snap.env_mask).all()


def test_detect_storm_finds_synthetic_minimum():
    """Plant an MSLP minimum off-centre and confirm detect_storm locates it."""
    import xarray as xr

    n_lat, n_lon = 41, 41
    lats = np.linspace(10.0, 20.0, n_lat)
    lons = np.linspace(125.0, 135.0, n_lon)
    lon_g, lat_g = np.meshgrid(lons, lats)

    # Storm at (16 N, 131 E), MSLP dip 3000 Pa deep, radius ~2 deg.
    storm_lat, storm_lon = 16.0, 131.0
    r2 = (lat_g - storm_lat) ** 2 + (lon_g - storm_lon) ** 2
    msl = 101300.0 - 3000.0 * np.exp(-r2 / (2 * 1.0**2))
    t2m = 300.0 - 2.0 * np.exp(-r2 / (2 * 1.0**2))
    u10 = 20.0 * np.exp(-r2 / (2 * 1.0**2)) * (lat_g - storm_lat)
    v10 = -20.0 * np.exp(-r2 / (2 * 1.0**2)) * (lon_g - storm_lon)

    ds = xr.Dataset(
        data_vars={
            "msl": (("latitude", "longitude"), msl),
            "2t": (("latitude", "longitude"), t2m),
            "10u": (("latitude", "longitude"), u10),
            "10v": (("latitude", "longitude"), v10),
        },
        coords={"latitude": lats, "longitude": lons},
    )
    snap = Snapshot.from_xarray(ds)
    snap.detect_storm(box_center=(15.0, 130.0), max_loc_error_km=600.0)
    assert snap.loc is not None
    assert snap.loc[0] == pytest.approx(storm_lat, abs=0.3)
    assert snap.loc[1] == pytest.approx(storm_lon, abs=0.3)
    assert snap.rdr > 0
