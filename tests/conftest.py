"""Shared test fixtures for tcinit.

The synthetic canonical Dataset avoids any dependency on aurora/torch/jax:
it's a pure-numpy xarray Dataset with the canonical short variable names
(msl, 2t, 10u, 10v, t, u, v, q, z) that the Snapshot / BogusVortex expect.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tcinit.best_track import BestTrack
from tcinit.snapshot import Snapshot

DEFAULT_LEVELS_HPA = np.array(
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=float
)


def make_canonical_ds(
    *,
    n_lat: int = 21,
    n_lon: int = 21,
    lat_centre: float = 15.0,
    lon_centre: float = 130.0,
    grid_spacing_deg: float = 0.25,
    levels_hpa: np.ndarray | None = None,
) -> xr.Dataset:
    """Small canonical (lat, lon, level) Dataset with plausible ambient values.

    Contains all nine canonical variables so both the surface and atmospheric
    injection paths of BogusVortex.apply() get exercised.
    """
    if levels_hpa is None:
        levels_hpa = DEFAULT_LEVELS_HPA

    half_lat = (n_lat - 1) / 2.0
    half_lon = (n_lon - 1) / 2.0
    lats = lat_centre + grid_spacing_deg * (np.arange(n_lat) - half_lat)
    lons = lon_centre + grid_spacing_deg * (np.arange(n_lon) - half_lon)

    shape2d = (n_lat, n_lon)
    shape3d = (levels_hpa.size, n_lat, n_lon)

    rs = np.random.RandomState(0)
    ds = xr.Dataset(
        data_vars={
            "msl": (("latitude", "longitude"), 101300.0 + 50.0 * rs.randn(*shape2d)),
            "2t": (("latitude", "longitude"), 298.0 + rs.randn(*shape2d)),
            "10u": (("latitude", "longitude"), 0.5 * rs.randn(*shape2d)),
            "10v": (("latitude", "longitude"), 0.5 * rs.randn(*shape2d)),
            "t": (("level", "latitude", "longitude"), 280.0 + rs.randn(*shape3d)),
            "u": (("level", "latitude", "longitude"), rs.randn(*shape3d)),
            "v": (("level", "latitude", "longitude"), rs.randn(*shape3d)),
            "q": (("level", "latitude", "longitude"), 5e-3 * np.ones(shape3d)),
            "z": (("level", "latitude", "longitude"), 50000.0 * np.ones(shape3d)),
        },
        coords={"latitude": lats, "longitude": lons, "level": levels_hpa},
    )
    return ds


def make_snapshot_from_ds(
    ds: xr.Dataset,
    *,
    rdr_km: float = 120.0,
    storm_radius_multiplier: float = 3.0,
    mslp_env_mean_pa: float = 101300.0,
    t2m_storm_mean_k: float = 300.0,
) -> Snapshot:
    """Populate a Snapshot from ds and force storm centre at the box centre."""
    snap = Snapshot.from_xarray(ds, storm_radius_multiplier=storm_radius_multiplier)
    lat_centre = float(0.5 * (snap.box_lats[0] + snap.box_lats[-1]))
    lon_centre = float(0.5 * (snap.box_lons[0] + snap.box_lons[-1]))
    snap.set_storm_center(
        loc=(lat_centre, lon_centre),
        rdr_km=rdr_km,
        mslp_env_mean_pa=mslp_env_mean_pa,
        t2m_storm_mean_k=t2m_storm_mean_k,
    )
    return snap


def make_best_track(
    target_time: datetime,
    *,
    mslp_hpa: float = 950.0,
    vmax_kt: float = 85.0,
    lat_deg: float = 15.0,
    lon_deg: float = 130.0,
    r_34kt_nm: float = 180.0,
) -> BestTrack:
    """Bypass BestTrack.__init__ (no CSV) but return a real BestTrack."""
    bt = BestTrack.__new__(BestTrack)
    bt.ID = "TEST01"
    bt.df_best_tracks_history = pd.DataFrame(
        {
            "datetimes": [target_time],
            "LAT_DEG": [lat_deg],
            "LON_DEG": [lon_deg],
            "MSLP": [mslp_hpa],
            "VMAX": [vmax_kt],
            "RAD34_NE": [r_34kt_nm],
            "RAD34_SE": [r_34kt_nm],
            "RAD34_SW": [r_34kt_nm],
            "RAD34_NW": [r_34kt_nm],
        }
    )
    return bt


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def target_time() -> datetime:
    return datetime(2020, 9, 1, 6)


@pytest.fixture
def canonical_ds() -> xr.Dataset:
    return make_canonical_ds()


@pytest.fixture
def snapshot(canonical_ds) -> Snapshot:
    return make_snapshot_from_ds(canonical_ds)


@pytest.fixture
def best_track(target_time) -> BestTrack:
    return make_best_track(target_time)
