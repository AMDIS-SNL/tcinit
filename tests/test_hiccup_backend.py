"""HICCUP / ERA5 adapter tests."""

from datetime import datetime

import numpy as np

from tcinit.backends import hiccup_backend
from tcinit.snapshot import Snapshot
from tcinit.vortex import BogusVortex
from tests.conftest import make_best_track


def test_hiccup_apply_uses_era5_shortnames(canonical_ds):
    """Canonical short names ARE the ERA5 short names, so this smoke-tests
    the passthrough path in hiccup_backend.apply."""
    snap = Snapshot.from_xarray(canonical_ds)
    lat_c = float(0.5 * (snap.box_lats[0] + snap.box_lats[-1]))
    lon_c = float(0.5 * (snap.box_lons[0] + snap.box_lons[-1]))
    snap.set_storm_center(
        loc=(lat_c, lon_c),
        rdr_km=100.0,
        mslp_env_mean_pa=101300.0,
        t2m_storm_mean_k=300.0,
    )
    bt = make_best_track(datetime(2020, 9, 1, 6), lat_deg=lat_c, lon_deg=lon_c)
    bv = BogusVortex(snap, bt, datetime(2020, 9, 1, 6)).build()
    ds_out = hiccup_backend.apply(canonical_ds, bv)

    # msl at centre should be lowered, t at 850 hPa should be modified.
    n_lat = canonical_ds.sizes["latitude"]
    n_lon = canonical_ds.sizes["longitude"]
    i, j = n_lat // 2, n_lon // 2
    msl_orig = float(canonical_ds["msl"].values[i, j])
    msl_new = float(ds_out["msl"].values[i, j])
    assert msl_new < msl_orig - 1000.0

    t850_orig = canonical_ds["t"].sel(level=850).values
    t850_new = ds_out["t"].sel(level=850).values
    assert not np.allclose(t850_new, t850_orig)


def test_hiccup_apply_leaves_missing_surface_vars_alone(canonical_ds):
    """Drop 2t/10u/10v and confirm apply doesn't complain (HICCUP typically
    has these in a separate surface file that isn't part of the atmospheric
    Dataset being modified)."""
    ds = canonical_ds.drop_vars(["2t", "10u", "10v"])
    snap = Snapshot.from_xarray(canonical_ds)  # detection uses full ds
    lat_c = float(0.5 * (snap.box_lats[0] + snap.box_lats[-1]))
    lon_c = float(0.5 * (snap.box_lons[0] + snap.box_lons[-1]))
    snap.set_storm_center(
        loc=(lat_c, lon_c),
        rdr_km=100.0,
        mslp_env_mean_pa=101300.0,
        t2m_storm_mean_k=300.0,
    )
    bt = make_best_track(datetime(2020, 9, 1, 6), lat_deg=lat_c, lon_deg=lon_c)
    bv = BogusVortex(snap, bt, datetime(2020, 9, 1, 6)).build()
    ds_out = hiccup_backend.apply(ds, bv)
    assert "2t" not in ds_out.data_vars
    assert "msl" in ds_out.data_vars
