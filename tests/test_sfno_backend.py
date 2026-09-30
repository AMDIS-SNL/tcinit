"""SFNO adapter tests.

Exercised entirely with numpy stand-ins for the SFNO tensor and CoordSystem,
so the backend's per-channel indexing, box-slice, and write-back logic are
covered without needing torch / earth2studio / makani. A torch smoke test at
the bottom is gated behind ``pytest.importorskip("torch")``.
"""

from collections import OrderedDict
from datetime import datetime

import numpy as np
import pytest

from tcinit.backends import sfno_backend
from tcinit.naming import (
    SFNO_LEVELS_HPA,
    SFNO_SURFACE_TO_CANON,
    sfno_channel_index,
    sfno_channel_name,
    sfno_variables_73,
)
from tcinit.snapshot import Snapshot
from tcinit.ideal_tc_vortex import BogusVortex
from tests.conftest import make_best_track


def _make_sfno_state(
    *,
    n_lat: int = 41,
    n_lon: int = 81,
    lat_range=(35.0, -5.0),  # descending, SFNO convention
    lon_range=(100.0, 180.0),
    n_batch: int = 1,
    n_time: int = 1,
    n_lead_time: int = 1,
):
    """Build a synthetic ``(x, coords)`` pair mimicking earth2studio's SFNO input.

    Grid is small enough for fast tests but keeps SFNO's dim conventions:
    ``lat`` descending north→south, ``lon`` ascending, ``variable`` axis
    carrying all 73 channels in checkpoint order.
    """
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)
    variables = np.array(sfno_variables_73())

    shape = (n_batch, n_time, n_lead_time, variables.size, n_lat, n_lon)
    rs = np.random.RandomState(2)
    x = rs.randn(*shape).astype(np.float32)

    # Inject plausible plausibility per channel type so the vortex build has
    # sensible ambient values (avoids NaN cascade in q_saturation etc.).
    def _ch(name):
        return int(np.where(variables == name)[0][0])

    x[..., _ch("msl"), :, :] = 101300.0 + 20.0 * rs.randn(n_lat, n_lon).astype(
        np.float32
    )
    x[..., _ch("t2m"), :, :] = 298.0 + rs.randn(n_lat, n_lon).astype(np.float32)
    x[..., _ch("u10m"), :, :] = 0.5 * rs.randn(n_lat, n_lon).astype(np.float32)
    x[..., _ch("v10m"), :, :] = 0.5 * rs.randn(n_lat, n_lon).astype(np.float32)
    for lev in SFNO_LEVELS_HPA:
        x[..., _ch(f"t{lev}"), :, :] = 280.0
        x[..., _ch(f"u{lev}"), :, :] = 0.0
        x[..., _ch(f"v{lev}"), :, :] = 0.0
        x[..., _ch(f"q{lev}"), :, :] = 5e-3
        x[..., _ch(f"z{lev}"), :, :] = 50000.0

    coords = OrderedDict(
        {
            "batch": np.arange(n_batch),
            "time": np.array([np.datetime64("2020-09-01T06:00:00")] * n_time),
            "lead_time": np.array([np.timedelta64(0, "h")] * n_lead_time),
            "variable": variables,
            "lat": lats,
            "lon": lons,
        }
    )
    return x, coords


# ---------------------------------------------------------------------------
# naming helpers
# ---------------------------------------------------------------------------


def test_sfno_channel_name_surface_and_atmos():
    assert sfno_channel_name("msl") == "msl"
    assert sfno_channel_name("10u") == "u10m"
    assert sfno_channel_name("10v") == "v10m"
    assert sfno_channel_name("2t") == "t2m"
    assert sfno_channel_name("t", 500) == "t500"
    assert sfno_channel_name("u", 1000) == "u1000"


def test_sfno_channel_name_rejects_bad_combos():
    with pytest.raises(ValueError, match="takes no level"):
        sfno_channel_name("msl", 500)
    with pytest.raises(ValueError, match="requires level_hpa"):
        sfno_channel_name("t")
    with pytest.raises(ValueError, match="unknown canonical"):
        sfno_channel_name("nonsense")


def test_sfno_channel_index_covers_all_kc_vars():
    variables = np.array(sfno_variables_73())
    # Every K&C-touched canonical must be locatable.
    for canon in SFNO_SURFACE_TO_CANON.values():
        idx = sfno_channel_index(variables, canon, None)
        assert idx is not None
    for canon in ("t", "u", "v", "q", "z"):
        for lev in SFNO_LEVELS_HPA:
            idx = sfno_channel_index(variables, canon, lev)
            assert idx is not None, f"missing {canon}{lev}"


def test_sfno_channel_index_returns_none_for_missing():
    variables = np.array(["msl", "2t"])  # tiny truncated set
    assert sfno_channel_index(variables, "10u") is None
    assert sfno_channel_index(variables, "t", 500) is None


# ---------------------------------------------------------------------------
# extract_snapshot_dataset
# ---------------------------------------------------------------------------


def test_extract_returns_canonical_dataset():
    x, coords = _make_sfno_state()
    box_center = (15.0, 130.0)
    box_size = 3.0
    ds = sfno_backend.extract_snapshot_dataset(
        x, coords, box_center=box_center, box_size=box_size
    )

    # Canonical short-name presence.
    for name in ("msl", "2t", "10u", "10v", "t", "u", "v", "q", "z"):
        assert name in ds.data_vars, f"missing {name}"

    # Untouched SFNO channels must NOT appear (they're not in the map).
    for excluded in ("u100m", "v100m", "sp", "tcwv"):
        assert excluded not in ds.data_vars

    # Dims and coord windowing.
    assert ds["msl"].dims == ("latitude", "longitude")
    assert ds["t"].dims == ("level", "latitude", "longitude")
    lats_global = coords["lat"]
    lons_global = coords["lon"]
    lat_c, lon_c = box_center
    expected_lats = lats_global[
        (lats_global >= lat_c - box_size) & (lats_global <= lat_c + box_size)
    ]
    expected_lons = lons_global[
        (lons_global >= lon_c - box_size) & (lons_global <= lon_c + box_size)
    ]
    np.testing.assert_allclose(ds["latitude"].values, expected_lats)
    np.testing.assert_allclose(ds["longitude"].values, expected_lons)
    np.testing.assert_array_equal(
        ds["level"].values.astype(int), np.array(SFNO_LEVELS_HPA)
    )


def test_extract_values_match_source_tensor():
    x, coords = _make_sfno_state()
    box_center = (15.0, 130.0)
    box_size = 3.0
    ds = sfno_backend.extract_snapshot_dataset(
        x, coords, box_center=box_center, box_size=box_size
    )

    lats_global = coords["lat"]
    lons_global = coords["lon"]
    lat_c, lon_c = box_center
    lat_idx = np.where(
        (lats_global >= lat_c - box_size) & (lats_global <= lat_c + box_size)
    )[0]
    lon_idx = np.where(
        (lons_global >= lon_c - box_size) & (lons_global <= lon_c + box_size)
    )[0]
    lat_sl = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_sl = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)

    msl_ch = sfno_channel_index(coords["variable"], "msl")
    np.testing.assert_allclose(ds["msl"].values, x[0, 0, 0, msl_ch, lat_sl, lon_sl])
    t500_ch = sfno_channel_index(coords["variable"], "t", 500)
    # Locate 500 hPa in the extracted level axis.
    k500 = int(np.where(ds["level"].values.astype(int) == 500)[0][0])
    np.testing.assert_allclose(
        ds["t"].values[k500], x[0, 0, 0, t500_ch, lat_sl, lon_sl]
    )


def test_extract_raises_on_wrong_ndim():
    x, coords = _make_sfno_state()
    with pytest.raises(ValueError, match="expected 6-D SFNO tensor"):
        sfno_backend.extract_snapshot_dataset(
            x[0],  # 5-D
            coords,
            box_center=(15.0, 130.0),
            box_size=3.0,
        )


def test_extract_raises_on_empty_box():
    x, coords = _make_sfno_state()
    with pytest.raises(ValueError, match="does not intersect"):
        sfno_backend.extract_snapshot_dataset(
            x,
            coords,
            box_center=(89.0, 0.0),  # past northernmost lat=35
            box_size=0.5,
        )


# ---------------------------------------------------------------------------
# write_back
# ---------------------------------------------------------------------------


def test_write_back_roundtrip_is_identity():
    x, coords = _make_sfno_state()
    box_center = (15.0, 130.0)
    box_size = 3.0
    ds = sfno_backend.extract_snapshot_dataset(
        x, coords, box_center=box_center, box_size=box_size
    )
    x_before = x.copy()
    sfno_backend.write_back(
        x,
        coords,
        ds,
        box_lats=ds["latitude"].values,
        box_lons=ds["longitude"].values,
    )
    # Round-trip write of the unmodified extract is an exact no-op.
    np.testing.assert_allclose(x, x_before)


def test_write_back_after_vortex_localises_changes():
    x, coords = _make_sfno_state()
    box_center = (15.0, 130.0)
    box_size = 3.0
    ds = sfno_backend.extract_snapshot_dataset(
        x, coords, box_center=box_center, box_size=box_size
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
    ds_mod = sfno_backend.apply(ds, bv)

    x_before = x.copy()
    sfno_backend.write_back(
        x,
        coords,
        ds_mod,
        box_lats=ds["latitude"].values,
        box_lons=ds["longitude"].values,
    )

    lats_global = coords["lat"]
    lons_global = coords["lon"]
    lat_idx = np.where(
        (lats_global >= lat_c - box_size) & (lats_global <= lat_c + box_size)
    )[0]
    lon_idx = np.where(
        (lons_global >= lon_c - box_size) & (lons_global <= lon_c + box_size)
    )[0]
    lat_sl = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_sl = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)
    lat_out = np.setdiff1d(
        np.arange(lats_global.size), np.arange(lat_sl.start, lat_sl.stop)
    )
    lon_out = np.setdiff1d(
        np.arange(lons_global.size), np.arange(lon_sl.start, lon_sl.stop)
    )

    # Cells strictly outside the box (either lat OR lon out of box) are
    # untouched across ALL channels.
    np.testing.assert_allclose(x[..., :, lat_out, :], x_before[..., :, lat_out, :])
    np.testing.assert_allclose(x[..., :, :, lon_out], x_before[..., :, :, lon_out])

    # Inside the box, msl and t500 differ from the source (vortex imprint).
    msl_ch = sfno_channel_index(coords["variable"], "msl")
    t500_ch = sfno_channel_index(coords["variable"], "t", 500)
    assert not np.allclose(
        x[..., msl_ch, lat_sl, lon_sl], x_before[..., msl_ch, lat_sl, lon_sl]
    )
    assert not np.allclose(
        x[..., t500_ch, lat_sl, lon_sl], x_before[..., t500_ch, lat_sl, lon_sl]
    )

    # Untouched channels (u100m, v100m, sp, tcwv) must be bit-identical
    # everywhere, including inside the box — write_back must skip them.
    for excluded in ("u100m", "v100m", "sp", "tcwv"):
        ch = int(np.where(coords["variable"] == excluded)[0][0])
        np.testing.assert_allclose(x[..., ch, :, :], x_before[..., ch, :, :])


def test_write_back_broadcasts_across_batch_and_time():
    # Multi-slot input: same vortex must land in every (batch, time, lead_time).
    x, coords = _make_sfno_state(n_batch=2, n_time=2, n_lead_time=1)
    box_center = (15.0, 130.0)
    box_size = 3.0
    ds = sfno_backend.extract_snapshot_dataset(
        x, coords, box_center=box_center, box_size=box_size
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
    ds_mod = sfno_backend.apply(ds, bv)
    sfno_backend.write_back(
        x,
        coords,
        ds_mod,
        box_lats=ds["latitude"].values,
        box_lons=ds["longitude"].values,
    )
    # All batch/time slots must be identical after the broadcasted write.
    msl_ch = sfno_channel_index(coords["variable"], "msl")
    ref = x[0, 0, 0, msl_ch]
    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            np.testing.assert_allclose(x[b, t, 0, msl_ch], ref)


def test_write_back_skips_vars_absent_from_modified_ds():
    x, coords = _make_sfno_state()
    box_center = (15.0, 130.0)
    box_size = 3.0
    ds = sfno_backend.extract_snapshot_dataset(
        x, coords, box_center=box_center, box_size=box_size
    )
    # Drop several vars so write_back has to skip them cleanly.
    partial = ds[["msl", "t"]]
    x_before = x.copy()
    sfno_backend.write_back(
        x,
        coords,
        partial,
        box_lats=ds["latitude"].values,
        box_lons=ds["longitude"].values,
    )
    # 10u/10v/2t/u/v/q/z channels must be untouched across the whole tensor.
    for canon in ("10u", "10v", "2t"):
        ch = sfno_channel_index(coords["variable"], canon)
        np.testing.assert_allclose(x[..., ch, :, :], x_before[..., ch, :, :])
    for canon in ("u", "v", "q", "z"):
        for lev in SFNO_LEVELS_HPA:
            ch = sfno_channel_index(coords["variable"], canon, lev)
            np.testing.assert_allclose(x[..., ch, :, :], x_before[..., ch, :, :])


# ---------------------------------------------------------------------------
# torch smoke test (skipped when torch is absent)
# ---------------------------------------------------------------------------


def test_write_back_with_torch_tensor_preserves_dtype_and_device():
    torch = pytest.importorskip("torch")

    x_np, coords = _make_sfno_state()
    x = torch.as_tensor(x_np, dtype=torch.float32)
    box_center = (15.0, 130.0)
    box_size = 3.0
    ds = sfno_backend.extract_snapshot_dataset(
        x, coords, box_center=box_center, box_size=box_size
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
    ds_mod = sfno_backend.apply(ds, bv)
    sfno_backend.write_back(
        x,
        coords,
        ds_mod,
        box_lats=ds["latitude"].values,
        box_lons=ds["longitude"].values,
    )
    assert x.dtype == torch.float32
    # In-place mutation: x is the same tensor we passed in, now vortex-imprinted.
    msl_ch = sfno_channel_index(coords["variable"], "msl")
    assert not torch.allclose(
        x[0, 0, 0, msl_ch], torch.as_tensor(x_np[0, 0, 0, msl_ch])
    )
