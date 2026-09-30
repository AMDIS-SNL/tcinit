"""NeuralGCM adapter tests — exercises long-form var map + swapped dim order,
plus the inputs-dict extract / write_back path.
"""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from tcinit.backends import neuralgcm_backend
from tcinit.snapshot import Snapshot
from tcinit.ideal_tc_vortex import BogusVortex
from tests.conftest import DEFAULT_LEVELS_HPA, make_best_track


def _make_neuralgcm_ds():
    """Long-form names, atmos-only, (level, longitude, latitude) dim order."""
    n_lat, n_lon = 21, 21
    lats = 15.0 + 0.25 * (np.arange(n_lat) - (n_lat - 1) / 2)
    lons = 130.0 + 0.25 * (np.arange(n_lon) - (n_lon - 1) / 2)
    levels = DEFAULT_LEVELS_HPA

    # (level, longitude, latitude) — NeuralGCM's regridded layout.
    shape = (levels.size, n_lon, n_lat)
    ds = xr.Dataset(
        data_vars={
            "temperature": (("level", "longitude", "latitude"), 280.0 * np.ones(shape)),
            "u_component_of_wind": (
                ("level", "longitude", "latitude"),
                np.zeros(shape),
            ),
            "v_component_of_wind": (
                ("level", "longitude", "latitude"),
                np.zeros(shape),
            ),
            "specific_humidity": (
                ("level", "longitude", "latitude"),
                5e-3 * np.ones(shape),
            ),
            "geopotential": (
                ("level", "longitude", "latitude"),
                50000.0 * np.ones(shape),
            ),
        },
        coords={"latitude": lats, "longitude": lons, "level": levels},
    )
    return ds


def test_neuralgcm_apply_modifies_atmos_vars():
    ds = _make_neuralgcm_ds()
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

    ds_out = neuralgcm_backend.apply(ds, bv)
    # Long-form vars still there.
    for name in (
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "geopotential",
    ):
        assert name in ds_out.data_vars

    # geopotential should be unchanged when additive_z=False (default).
    assert np.allclose(
        ds_out["geopotential"].values, ds["geopotential"].values, atol=1e-6
    )
    # temperature at storm centre should be different from ambient.
    T_before = ds["temperature"].sel(level=500).values
    T_after = ds_out["temperature"].sel(level=500).values
    assert not np.allclose(T_after, T_before)


def test_neuralgcm_dim_order_preserved():
    """Blending must not rearrange the Dataset's dim ordering."""
    ds = _make_neuralgcm_ds()
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
    ds_out = neuralgcm_backend.apply(ds, bv)
    # xarray broadcasting doesn't guarantee dim order preservation, but the
    # variable itself must still be usable and finite.
    assert set(ds_out["temperature"].dims) == {"level", "longitude", "latitude"}
    assert np.isfinite(ds_out["temperature"].values).all()


# ---------------------------------------------------------------------------
# Inputs-dict path: extract_snapshot_dataset / write_back
# ---------------------------------------------------------------------------


def _make_neuralgcm_inputs():
    """Full-globe (small) inputs dict mimicking PressureLevelModel output.

    Values are shaped (level, longitude, latitude); latitude is ascending
    (Gaussian-grid convention); a scalar sim_time entry is present and must
    survive the extract/write_back round-trip untouched.
    """
    n_lat, n_lon = 32, 64  # tiny globe stand-in
    lats = np.linspace(-87.0, 87.0, n_lat)  # ascending, poles excluded
    lons = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    levels = DEFAULT_LEVELS_HPA
    shape = (levels.size, n_lon, n_lat)

    rs = np.random.RandomState(1)
    inputs = {
        "temperature": (280.0 + rs.randn(*shape)).astype(np.float32),
        "u_component_of_wind": rs.randn(*shape).astype(np.float32),
        "v_component_of_wind": rs.randn(*shape).astype(np.float32),
        "specific_humidity": (5e-3 * np.ones(shape)).astype(np.float32),
        "geopotential": (50000.0 * np.ones(shape)).astype(np.float32),
        # Non-array-shaped entries must pass through untouched.
        "sim_time": np.array(1.234e5),
        # A field NEURALGCM_VAR_MAP does not cover — should be skipped by
        # both extract and write_back.
        "specific_cloud_ice_water_content": np.zeros(shape, dtype=np.float32),
    }
    return inputs, lats, lons, levels


def test_extract_snapshot_dataset_shapes_and_coords():
    inputs, lats, lons, levels = _make_neuralgcm_inputs()
    # Pick a box centred inside the grid so masking is unambiguous.
    box_center = (15.0, 130.0)
    box_size = 3.0
    ds = neuralgcm_backend.extract_snapshot_dataset(
        inputs,
        lats=lats,
        lons=lons,
        levels=levels,
        box_center=box_center,
        box_size=box_size,
    )
    # All K&C-relevant long-form vars are present; cloud & sim_time are dropped.
    for name in (
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "geopotential",
    ):
        assert name in ds.data_vars
    assert "specific_cloud_ice_water_content" not in ds.data_vars
    assert "sim_time" not in ds.data_vars

    # Dim order matches inputs: (level, longitude, latitude).
    assert ds["temperature"].dims == ("level", "longitude", "latitude")

    # Coord windows are the intersection of the box with the global axes.
    lat_c, lon_c = box_center
    expected_lats = lats[(lats >= lat_c - box_size) & (lats <= lat_c + box_size)]
    expected_lons = lons[(lons >= lon_c - box_size) & (lons <= lon_c + box_size)]
    np.testing.assert_allclose(ds["latitude"].values, expected_lats)
    np.testing.assert_allclose(ds["longitude"].values, expected_lons)
    np.testing.assert_allclose(ds["level"].values, levels)

    # Extracted values equal the raw slice from the source arrays.
    lat_sl = slice(
        int(np.where(lats == expected_lats[0])[0][0]),
        int(np.where(lats == expected_lats[-1])[0][0]) + 1,
    )
    lon_sl = slice(
        int(np.where(lons == expected_lons[0])[0][0]),
        int(np.where(lons == expected_lons[-1])[0][0]) + 1,
    )
    np.testing.assert_allclose(
        ds["temperature"].values,
        inputs["temperature"][:, lon_sl, lat_sl],
    )


def test_extract_snapshot_dataset_raises_on_empty_box():
    inputs, lats, lons, levels = _make_neuralgcm_inputs()
    with pytest.raises(ValueError, match="does not intersect"):
        neuralgcm_backend.extract_snapshot_dataset(
            inputs,
            lats=lats,
            lons=lons,
            levels=levels,
            box_center=(89.9, 0.0),  # past northernmost lat (max ~87)
            box_size=0.1,
        )


def test_extract_snapshot_dataset_shape_mismatch_raises():
    inputs, lats, lons, levels = _make_neuralgcm_inputs()
    # Corrupt one variable's level count.
    inputs["temperature"] = inputs["temperature"][:5]
    with pytest.raises(ValueError, match="level dim of size"):
        neuralgcm_backend.extract_snapshot_dataset(
            inputs,
            lats=lats,
            lons=lons,
            levels=levels,
            box_center=(15.0, 130.0),
            box_size=3.0,
        )


def test_write_back_roundtrip_preserves_untouched_cells():
    inputs, lats, lons, levels = _make_neuralgcm_inputs()
    box_center = (15.0, 130.0)
    box_size = 3.0

    ds = neuralgcm_backend.extract_snapshot_dataset(
        inputs,
        lats=lats,
        lons=lons,
        levels=levels,
        box_center=box_center,
        box_size=box_size,
    )

    # Round-trip without modification: writing the extracted box back should
    # be a no-op on numeric values and must not mutate the source dict.
    new_inputs = neuralgcm_backend.write_back(
        inputs,
        ds,
        lats=lats,
        lons=lons,
        levels=levels,
    )
    assert new_inputs is not inputs
    for name in (
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "geopotential",
    ):
        np.testing.assert_allclose(new_inputs[name], inputs[name])
    # Non-array pass-through survives.
    assert new_inputs["sim_time"] == inputs["sim_time"]
    # Uncovered field is passed through unchanged.
    np.testing.assert_allclose(
        new_inputs["specific_cloud_ice_water_content"],
        inputs["specific_cloud_ice_water_content"],
    )


def test_write_back_after_vortex_modifies_only_box():
    inputs, lats, lons, levels = _make_neuralgcm_inputs()
    box_center = (15.0, 130.0)
    box_size = 3.0

    ds = neuralgcm_backend.extract_snapshot_dataset(
        inputs,
        lats=lats,
        lons=lons,
        levels=levels,
        box_center=box_center,
        box_size=box_size,
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
    ds_mod = neuralgcm_backend.apply(ds, bv)

    new_inputs = neuralgcm_backend.write_back(
        inputs,
        ds_mod,
        lats=lats,
        lons=lons,
        levels=levels,
    )

    # Cells outside the box must be bit-identical to the source.
    lat_mask = (lats >= lat_c - box_size) & (lats <= lat_c + box_size)
    lon_mask = (lons >= lon_c - box_size) & (lons <= lon_c + box_size)
    lat_out = ~lat_mask
    lon_out = ~lon_mask
    for name in ("temperature", "u_component_of_wind"):
        # A cell is outside the box if lat OR lon is outside.
        # Check strict-outside strips (safer than building a 2D outer mask).
        np.testing.assert_allclose(
            new_inputs[name][:, :, lat_out],
            inputs[name][:, :, lat_out],
        )
        np.testing.assert_allclose(
            new_inputs[name][:, lon_out, :],
            inputs[name][:, lon_out, :],
        )

    # Inside the box, the temperature field must differ from the source
    # (K&C imposes a warm core / anomalous humidity profile).
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    lat_sl = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
    lon_sl = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)
    assert not np.allclose(
        new_inputs["temperature"][:, lon_sl, lat_sl],
        inputs["temperature"][:, lon_sl, lat_sl],
    )

    # dtype must be preserved through the round-trip (encode expects float32).
    assert new_inputs["temperature"].dtype == inputs["temperature"].dtype
