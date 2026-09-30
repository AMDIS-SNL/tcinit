"""HICCUP / ERA5 adapter tests.

Two paths under test:

- ``apply()`` on an already-canonical Dataset (existing behaviour).
- File-based ``extract_snapshot_dataset`` / ``write_back`` that operate on
  real netCDF files mirroring the CDS-ERA5 schema HICCUP consumes.
"""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from tcinit.backends import hiccup_backend
from tcinit.snapshot import Snapshot
from tcinit.vortex import BogusVortex
from tests.conftest import DEFAULT_LEVELS_HPA, make_best_track


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


# ---------------------------------------------------------------------------
# File-based path: extract_snapshot_dataset / write_back
# ---------------------------------------------------------------------------


def _write_era5_atm_file(
    path,
    *,
    n_lat=181,
    n_lon=360,
    lat_range=(90.0, -90.0),  # descending, ERA5 convention (1° global)
    lon_range=(0.0, 359.0),
    levels_hpa=None,
    time_name="valid_time",
    level_name="pressure_level",
):
    """Write a synthetic ERA5 atmospheric netCDF (CDS-style naming)."""
    if levels_hpa is None:
        # CDS ERA5 has levels descending 1000 → 50.
        levels_hpa = DEFAULT_LEVELS_HPA[::-1]
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)
    shape = (1, levels_hpa.size, n_lat, n_lon)  # (time, level, lat, lon)
    rs = np.random.RandomState(5)

    def _a(mean, std):
        return (mean + std * rs.randn(*shape)).astype(np.float32)

    dims = (time_name, level_name, "latitude", "longitude")
    ds = xr.Dataset(
        data_vars={
            "t": (dims, _a(280.0, 1.0)),
            "u": (dims, _a(0.0, 1.0)),
            "v": (dims, _a(0.0, 1.0)),
            "q": (dims, _a(5e-3, 0.0)),
            "z": (dims, _a(50000.0, 0.0)),
            # HICCUP-only atmospheric field — must round-trip untouched.
            "o3": (dims, _a(1e-6, 0.0)),
            "clwc": (dims, _a(0.0, 0.0)),
            "ciwc": (dims, _a(0.0, 0.0)),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            level_name: levels_hpa,
            time_name: np.array([np.datetime64("2020-09-01T06:00")]),
        },
    )
    ds.to_netcdf(path)
    ds.close()
    return lats, lons, levels_hpa


def _write_era5_sfc_file(
    path,
    *,
    n_lat=181,
    n_lon=360,
    lat_range=(90.0, -90.0),
    lon_range=(0.0, 359.0),
    time_name="valid_time",
    include_10m_wind_and_msl=True,
):
    """Write a synthetic ERA5 surface netCDF using CDS-style names.

    Uses ``t2m``/``u10``/``v10`` (not ``2t``/``10u``/``10v``) to exercise
    the backend's variable-name auto-detection.
    """
    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)
    shape = (1, n_lat, n_lon)
    rs = np.random.RandomState(6)

    def _s(mean, std):
        return (mean + std * rs.randn(*shape)).astype(np.float32)

    dims = (time_name, "latitude", "longitude")
    data_vars = {
        "t2m": (dims, _s(298.0, 1.0)),
        # HICCUP reads these from the sfc file; must survive round-trip
        # untouched.
        "sp": (dims, _s(101300.0, 100.0)),
        "skt": (dims, _s(295.0, 1.0)),
        "z": (dims, _s(1000.0, 10.0)),  # surface geopotential, 2-D
        "sst": (dims, _s(300.0, 0.5)),
        "siconc": (dims, _s(0.0, 0.0)),
    }
    if include_10m_wind_and_msl:
        data_vars["msl"] = (dims, _s(101300.0, 20.0))
        data_vars["u10"] = (dims, _s(0.0, 0.5))
        data_vars["v10"] = (dims, _s(0.0, 0.5))
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": lats,
            "longitude": lons,
            time_name: np.array([np.datetime64("2020-09-01T06:00")]),
        },
    )
    ds.to_netcdf(path)
    ds.close()
    return lats, lons


def _build_bv_from_box(box_ds):
    """Build a BogusVortex from a canonical box Dataset."""
    snap = Snapshot.from_xarray(box_ds)  # canonical short names + latitude/longitude
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


def test_extract_canonicalises_names_and_slices_box(tmp_path):
    atm = tmp_path / "era5_atm.nc"
    sfc = tmp_path / "era5_sfc.nc"
    _write_era5_atm_file(atm)
    _write_era5_sfc_file(sfc)

    box_ds = hiccup_backend.extract_snapshot_dataset(
        [str(atm), str(sfc)],
        box_center=(15.0, 130.0),
        box_size=5.0,
    )
    # Canonical short names present.
    for name in ("t", "u", "v", "q", "z", "msl", "2t", "10u", "10v"):
        assert name in box_ds.data_vars, f"missing {name}"

    # HICCUP-only fields excluded from extract (they stay in the source file
    # and pass through write_back).
    for excluded in ("o3", "clwc", "ciwc", "sp", "skt", "sst", "siconc"):
        assert excluded not in box_ds.data_vars

    # ``z`` in the extract is the ATMOSPHERIC z (3-D with level), not the
    # 2-D surface geopotential from the sfc file.
    assert "level" in box_ds["z"].dims
    assert "level" in box_ds["t"].dims

    # Time collapsed away, coord names canonicalised.
    assert "time" not in box_ds.dims
    assert "valid_time" not in box_ds.dims
    assert "level" in box_ds.dims
    assert "pressure_level" not in box_ds.dims


def test_extract_raises_on_empty_box(tmp_path):
    atm = tmp_path / "era5_atm.nc"
    _write_era5_atm_file(atm)
    with pytest.raises(ValueError, match="does not intersect"):
        hiccup_backend.extract_snapshot_dataset(
            [str(atm)],
            box_center=(0.0, 500.0),  # lon past 355
            box_size=1.0,
        )


def test_write_back_produces_new_file_and_preserves_everything_else(tmp_path):
    atm = tmp_path / "era5_atm.nc"
    sfc = tmp_path / "era5_sfc.nc"
    _write_era5_atm_file(atm)
    _write_era5_sfc_file(sfc)

    box_ds = hiccup_backend.extract_snapshot_dataset(
        [str(atm), str(sfc)],
        box_center=(15.0, 130.0),
        box_size=5.0,
    )
    bv = _build_bv_from_box(box_ds)
    mod = hiccup_backend.apply(box_ds, bv)

    atm_out = hiccup_backend.write_back(
        str(atm),
        mod,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
    )
    sfc_out = hiccup_backend.write_back(
        str(sfc),
        mod,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
    )
    assert atm_out.endswith(".tc_bogus.nc")
    assert sfc_out.endswith(".tc_bogus.nc")

    # Reopen and inspect.
    atm_src = xr.open_dataset(str(atm))
    atm_new = xr.open_dataset(atm_out)
    sfc_src = xr.open_dataset(str(sfc))
    sfc_new = xr.open_dataset(sfc_out)
    try:
        # HICCUP-only vars: bit-identical to source.
        for name in ("o3", "clwc", "ciwc"):
            np.testing.assert_allclose(atm_new[name].values, atm_src[name].values)
        for name in ("sp", "skt", "z", "sst", "siconc"):
            np.testing.assert_allclose(sfc_new[name].values, sfc_src[name].values)
        # K&C-touched atmos vars: differ from source (vortex was applied).
        for name in ("t", "u", "v"):
            assert not np.allclose(atm_new[name].values, atm_src[name].values)
        # K&C-touched surface vars: differ from source.
        for name in ("t2m", "msl", "u10", "v10"):
            assert not np.allclose(sfc_new[name].values, sfc_src[name].values)

        # Cells strictly outside the box are unchanged.
        lat_c, lon_c = 15.0, 130.0
        box_size = 5.0
        lats = atm_src["latitude"].values
        lons = atm_src["longitude"].values
        lat_mask = (lats >= lat_c - box_size) & (lats <= lat_c + box_size)
        lon_mask = (lons >= lon_c - box_size) & (lons <= lon_c + box_size)
        lat_out = ~lat_mask
        lon_out = ~lon_mask
        np.testing.assert_allclose(
            atm_new["t"].values[..., lat_out, :], atm_src["t"].values[..., lat_out, :]
        )
        np.testing.assert_allclose(
            atm_new["t"].values[..., :, lon_out], atm_src["t"].values[..., :, lon_out]
        )
        np.testing.assert_allclose(
            sfc_new["t2m"].values[..., lat_out, :],
            sfc_src["t2m"].values[..., lat_out, :],
        )

        # dtype preserved.
        assert atm_new["t"].dtype == atm_src["t"].dtype
        assert sfc_new["t2m"].dtype == sfc_src["t2m"].dtype

        # Original file coord names preserved (pressure_level/valid_time).
        assert "pressure_level" in atm_new.coords
        assert "valid_time" in atm_new.coords
    finally:
        atm_src.close()
        atm_new.close()
        sfc_src.close()
        sfc_new.close()


def test_write_back_surface_z_not_clobbered_by_atmospheric_z(tmp_path):
    """The 2-D surface geopotential ``z`` in the sfc file has the same variable
    name as the 3-D atmospheric geopotential ``z`` in the atm file. write_back
    must not overwrite the surface one with 3-D atmospheric data.
    """
    atm = tmp_path / "era5_atm.nc"
    sfc = tmp_path / "era5_sfc.nc"
    _write_era5_atm_file(atm)
    _write_era5_sfc_file(sfc)

    box_ds = hiccup_backend.extract_snapshot_dataset(
        [str(atm), str(sfc)],
        box_center=(15.0, 130.0),
        box_size=5.0,
    )
    bv = _build_bv_from_box(box_ds)
    mod = hiccup_backend.apply(box_ds, bv)

    sfc_out = hiccup_backend.write_back(
        str(sfc),
        mod,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
    )
    sfc_src = xr.open_dataset(str(sfc))
    sfc_new = xr.open_dataset(sfc_out)
    try:
        # 2-D surface z is bit-identical to source; the 3-D atmospheric z in
        # mod must NOT have been written into it.
        np.testing.assert_allclose(sfc_new["z"].values, sfc_src["z"].values)
        # And the dim shape confirms it stayed 2-D.
        assert "pressure_level" not in sfc_new["z"].dims
        assert "level" not in sfc_new["z"].dims
    finally:
        sfc_src.close()
        sfc_new.close()


def test_write_back_refuses_to_clobber_existing_output(tmp_path):
    atm = tmp_path / "era5_atm.nc"
    _write_era5_atm_file(atm)
    box_ds = hiccup_backend.extract_snapshot_dataset(
        [str(atm)],
        box_center=(15.0, 130.0),
        box_size=5.0,
    )
    bv = _build_bv_from_box(
        # Give the vortex a synthetic surface field to work with, since the
        # atm-only extract lacks msl/2t/10u/10v.
        box_ds.assign(
            msl=(
                ("latitude", "longitude"),
                101300.0
                * np.ones((box_ds.sizes["latitude"], box_ds.sizes["longitude"])),
            ),
            **{
                "2t": (
                    ("latitude", "longitude"),
                    300.0
                    * np.ones((box_ds.sizes["latitude"], box_ds.sizes["longitude"])),
                )
            },
        )
    )
    mod = hiccup_backend.apply(box_ds, bv)

    out1 = hiccup_backend.write_back(
        str(atm),
        mod,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
    )
    with pytest.raises(FileExistsError):
        hiccup_backend.write_back(
            str(atm),
            mod,
            box_lats=box_ds["latitude"].values,
            box_lons=box_ds["longitude"].values,
        )
    # But overwrite=True works.
    out2 = hiccup_backend.write_back(
        str(atm),
        mod,
        box_lats=box_ds["latitude"].values,
        box_lons=box_ds["longitude"].values,
        overwrite=True,
    )
    assert out1 == out2


def test_extract_accepts_legacy_coord_names(tmp_path):
    """Old CDS files use ``time`` / ``level`` instead of ``valid_time`` /
    ``pressure_level``. Both should just work."""
    atm = tmp_path / "era5_atm_legacy.nc"
    _write_era5_atm_file(atm, time_name="time", level_name="level")

    box_ds = hiccup_backend.extract_snapshot_dataset(
        [str(atm)],
        box_center=(15.0, 130.0),
        box_size=5.0,
    )
    assert "level" in box_ds.dims
    assert "t" in box_ds.data_vars
