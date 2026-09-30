"""Aurora backend round-trip tests. Gated on torch availability."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tcinit.backends import aurora_backend  # noqa: E402
from tcinit.snapshot import Snapshot  # noqa: E402
from tcinit.ideal_tc_vortex import BogusVortex  # noqa: E402
from tests.conftest import DEFAULT_LEVELS_HPA, make_best_track  # noqa: E402


def _make_batch(
    *,
    n_lat: int = 41,
    n_lon: int = 41,
    lat_range=(20.0, 10.0),  # descending, Aurora convention
    lon_range=(125.0, 135.0),
    n_times: int = 2,
    levels=None,
):
    if levels is None:
        levels = DEFAULT_LEVELS_HPA.astype(int).tolist()
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)

    def _surf(value):
        return torch.full((1, n_times, n_lat, n_lon), value, dtype=torch.float64)

    def _atmos(value):
        return torch.full(
            (1, n_times, len(levels), n_lat, n_lon), value, dtype=torch.float64
        )

    surf_vars = {
        "msl": _surf(101300.0),
        "2t": _surf(298.0),
        "10u": _surf(0.0),
        "10v": _surf(0.0),
    }
    atmos_vars = {
        "t": _atmos(280.0),
        "u": _atmos(0.0),
        "v": _atmos(0.0),
        "q": _atmos(5e-3),
        "z": _atmos(50000.0),
    }
    metadata = SimpleNamespace(
        lat=torch.as_tensor(lats),
        lon=torch.as_tensor(lons),
        atmos_levels=tuple(levels),
        time=(datetime(2020, 9, 1, 6),),
    )
    return SimpleNamespace(
        surf_vars=surf_vars, atmos_vars=atmos_vars, metadata=metadata
    )


def test_extract_snapshot_dataset_returns_canonical_dataset():
    batch = _make_batch()
    ds = aurora_backend.extract_snapshot_dataset(
        batch, box_center=(15.0, 130.0), box_size=3.0
    )
    for canon in ("msl", "2t", "10u", "10v", "t", "u", "v", "q", "z"):
        assert canon in ds.data_vars
    assert ds.sizes["latitude"] > 0
    assert ds.sizes["longitude"] > 0
    assert ds.sizes["level"] == DEFAULT_LEVELS_HPA.size


def test_write_back_matches_apply_on_dataset():
    batch = _make_batch()
    ds = aurora_backend.extract_snapshot_dataset(
        batch, box_center=(15.0, 130.0), box_size=3.0
    )
    snap = Snapshot.from_xarray(ds)
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
    ds_out = bv.apply(ds)
    aurora_backend.write_back(
        batch, ds_out, box_lats=snap.box_lats, box_lons=snap.box_lons
    )

    # Batch msl inside the box should now match ds_out.msl at every timestep.
    lats_global = batch.metadata.lat.numpy()
    lons_global = batch.metadata.lon.numpy()
    lat_start = int(np.argmin(np.abs(lats_global - snap.box_lats[0])))
    lon_start = int(np.argmin(np.abs(lons_global - snap.box_lons[0])))
    lat_sl = slice(lat_start, lat_start + snap.box_lats.size)
    lon_sl = slice(lon_start, lon_start + snap.box_lons.size)

    n_times = batch.surf_vars["msl"].shape[1]
    for t in range(n_times):
        got = batch.surf_vars["msl"][0, t, lat_sl, lon_sl].numpy()
        assert np.allclose(got, ds_out["msl"].values, atol=1e-6)

    # Batch cell far outside the box should be unchanged (still 101300 Pa).
    assert batch.surf_vars["msl"][0, 0, 0, 0].item() == pytest.approx(101300.0)
