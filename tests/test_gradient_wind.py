"""Gradient-wind balance diagnostic, snapshot plot, and rollout balance pages."""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from tcinit.constants import OMEGA_EARTH, R_D  # noqa: E402
from tcinit.diagnostics.gradient_wind import (  # noqa: E402
    _gradient_wind,
    gradient_wind_balance,
    snapshot_gradient_wind,
)
from tcinit.earth_util import great_circle_bearing, haversine_dist  # noqa: E402
from tcinit.plotting import plot_snapshot_diagnostics  # noqa: E402
from tcinit.rollout_snapshots import RolloutSnapshots  # noqa: E402
from tcinit.snapshot import Snapshot  # noqa: E402

LEVELS = np.array([200.0, 500.0, 850.0, 1000.0])
RHO = 1.15
P_ENV = 101000.0
DP = 8000.0
RM_KM = 50.0
B = 1.5


def _holland_dpdr(r_m):
    x = (RM_KM * 1e3 / r_m) ** B
    return DP * np.exp(-x) * B * x / r_m


def _vortex_ds(lat_c=16.0, lon_c=134.0, dlat=0.1, half=4.0, wind_scale=1.0):
    """Canonical box with an exactly gradient-balanced Holland vortex.

    Geopotential anomaly = amp(p) * p'(r) / rho on every level, and the wind
    is ``wind_scale`` times the gradient wind of that geopotential. 1000 hPa
    is below the central pressure, so it is underground near the centre.
    """
    lats = np.arange(lat_c - half, lat_c + half + 1e-9, dlat)
    lons = np.arange(lon_c - half, lon_c + half + 1e-9, dlat)
    lat2, lon2 = np.meshgrid(lats, lons, indexing="ij")
    r = np.maximum(haversine_dist(lat_c, lon_c, lat2, lon2), 1e-3) * 1e3
    p = P_ENV - DP * (1.0 - np.exp(-((RM_KM * 1e3 / r) ** B)))
    amp = np.array([0.3, 0.7, 1.0, 1.0])[:, None, None]
    f = abs(2.0 * OMEGA_EARTH * np.sin(np.deg2rad(lat_c)))
    v_lev = wind_scale * _gradient_wind(r[None], amp * _holland_dpdr(r)[None] / RHO, f)
    v_sfc = wind_scale * _gradient_wind(r, _holland_dpdr(r) / RHO, f)
    beta = great_circle_bearing(lat2, lon2, lat_c, lon_c) + np.pi
    sign = -1.0 if lat_c < 0 else 1.0
    te, tn = -np.cos(beta) * sign, np.sin(beta) * sign
    z = 1e4 * (1000.0 / LEVELS)[:, None, None] + amp * (p - P_ENV)[None] / RHO
    dims2, dims3 = ("latitude", "longitude"), ("level", "latitude", "longitude")
    return xr.Dataset(
        {
            "msl": (dims2, p),
            "2t": (dims2, p / (R_D * RHO)),
            "10u": (dims2, v_sfc * te),
            "10v": (dims2, v_sfc * tn),
            "u": (dims3, v_lev * te),
            "v": (dims3, v_lev * tn),
            "z": (dims3, z),
            "t": (dims3, np.full(z.shape, 280.0)),
            "q": (dims3, np.full(z.shape, 5e-3)),
        },
        coords={"latitude": lats, "longitude": lons, "level": LEVELS},
    )


def _snapshot(ds, valid_time=datetime(2023, 5, 26, 6)):
    snap = Snapshot.from_xarray(ds, valid_time=valid_time)
    lat_c = float(0.5 * (snap.box_lats[0] + snap.box_lats[-1]))
    lon_c = float(0.5 * (snap.box_lons[0] + snap.box_lons[-1]))
    snap.set_storm_center(
        loc=(lat_c, lon_c), rdr_km=60.0, mslp_env_mean_pa=P_ENV, t2m_storm_mean_k=300.0
    )
    for canon in ("t", "u", "v", "q", "z"):
        setattr(snap, f"src_{canon}", np.asarray(ds[canon].values, dtype=float))
    return snap


def _gwb(ds):
    return gradient_wind_balance(
        ds.latitude.values,
        ds.longitude.values,
        (16.0, 134.0) if ds.latitude.values.mean() > 0 else (-16.0, 134.0),
        levels_hpa=LEVELS,
        u=ds.u.values,
        v=ds.v.values,
        z=ds.z.values,
        msl=ds.msl.values,
        u10=ds["10u"].values,
        v10=ds["10v"].values,
        t2m=ds["2t"].values,
    )


def test_balanced_vortex_reads_balanced():
    g = _gwb(_vortex_ds())
    for summary in (
        g.level_summary(850.0),
        g.level_summary(500.0),
        g.surface_summary(),
    ):
        assert summary["ratio"] == pytest.approx(1.0, abs=0.02)
    s850 = g.level_summary(850.0)
    assert s850["rmw_km"] == pytest.approx(RM_KM, abs=10.0)
    assert g.imbalance_rms(r_max_km=300.0) < 1.5


def test_subgradient_wind_is_detected():
    g = _gwb(_vortex_ds(wind_scale=0.8))
    assert g.level_summary(850.0)["ratio"] == pytest.approx(0.8, abs=0.02)
    assert g.surface_summary()["ratio"] == pytest.approx(0.8, abs=0.02)


def test_southern_hemisphere_matches_northern():
    g_n = _gwb(_vortex_ds(lat_c=16.0))
    g_s = _gwb(_vortex_ds(lat_c=-16.0))
    np.testing.assert_allclose(g_s.v_t, g_n.v_t, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(g_s.v_grad, g_n.v_grad, rtol=1e-6, atol=1e-6)
    assert g_s.f_coriolis < 0 < g_n.f_coriolis


def test_levels_below_ground_are_masked():
    g = _gwb(_vortex_ds())
    k1000 = int(np.where(LEVELS == 1000.0)[0][0])
    core = g.r_km < RM_KM
    assert np.all(np.isnan(g.v_t[k1000, core]))
    assert np.isfinite(g.v_t[k1000, g.r_km > 300.0]).all()


def test_missing_inputs_leave_parts_empty():
    ds = _vortex_ds()
    g = gradient_wind_balance(
        ds.latitude.values, ds.longitude.values, (16.0, 134.0), msl=ds.msl.values
    )
    assert g.v_t is None and g.v_t_10m is None and g.v_grad_sfc is None
    assert np.isnan(g.level_summary(850.0)["ratio"])
    assert np.isnan(g.imbalance_rms())


def test_snapshot_diagnostics_and_dataframe_carry_balance():
    snap = _snapshot(_vortex_ds())
    assert snapshot_gradient_wind(snap) is not None
    hist = RolloutSnapshots("TEST01", snap.valid_time, [snap])
    diag = hist.diagnostics(snap)
    assert diag["gw_ratio_850"] == pytest.approx(1.0, abs=0.02)
    assert np.isfinite(diag["gw_imbalance_rms"])
    df = hist.to_dataframe()
    for col in ("gw_ratio_850", "gw_ratio_sfc", "gw_imbalance_rms", "gw_rmw_850_km"):
        assert col in df.columns
    assert "gw_ratio_850" not in hist.to_dataframe(include_balance=False).columns


def test_snapshot_without_center_has_no_balance():
    snap = Snapshot.from_xarray(_vortex_ds())
    assert snapshot_gradient_wind(snap) is None
    diag = RolloutSnapshots("TEST01", datetime(2023, 5, 26), [snap]).diagnostics(snap)
    assert diag["gradient_wind"] is None and np.isnan(diag["gw_ratio_850"])


def test_plot_snapshot_diagnostics_from_bundle_and_from_ds(tmp_path):
    ds = _vortex_ds()
    snap = _snapshot(ds)
    out = tmp_path / "balance.png"
    fig = plot_snapshot_diagnostics(snap, filename=out)
    assert out.exists() and out.stat().st_size > 0
    assert "850 hPa" in fig._suptitle.get_text()
    # No bundled fields: falls back to the Dataset.
    bare = Snapshot.from_xarray(ds, valid_time=snap.valid_time)
    bare.set_storm_center(
        loc=snap.loc, rdr_km=60.0, mslp_env_mean_pa=P_ENV, t2m_storm_mean_k=300.0
    )
    plot_snapshot_diagnostics(bare, ds, filename=tmp_path / "balance_ds.png")
    assert (tmp_path / "balance_ds.png").exists()


def test_rollout_balance_pages(tmp_path):
    from matplotlib.backends.backend_pdf import PdfPages

    from tcinit import rollout_snapshots as rs

    t0 = datetime(2023, 5, 26, 6)
    snaps = [
        _snapshot(_vortex_ds(wind_scale=s), valid_time=t0 + k * timedelta(hours=6))
        for k, s in enumerate((1.0, 0.9))
    ]
    hist = RolloutSnapshots("TEST01", t0, snaps)
    vt_lim, imb_lim = rs._balance_lims([hist.diagnostics(s) for s in snaps])
    assert np.isfinite(vt_lim) and imb_lim > 0
    out = tmp_path / "pages.pdf"
    with PdfPages(str(out)) as pdf:
        for s in snaps:
            rs._plot_balance_page(hist, s, pdf, vt_lim=vt_lim, imb_lim=imb_lim)
        rs._plot_balance_summary_page(hist, snaps, pdf)
        rs._plot_balance_comparison_page(
            hist, snaps, hist, snaps, pdf, labels=("a", "b")
        )
        n_pages = pdf.get_pagecount()
    assert n_pages == 4
    assert out.stat().st_size > 0
