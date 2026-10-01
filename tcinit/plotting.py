"""Plotting helpers for the K&C bogus vortex and state differences.

- :func:`plot_bogus`: radial-vertical (r, p) cross-section (K&C figure-7 style)
- :func:`plot_snapshot`: 3x3 map-view of a Snapshot (vorticity, MSLP, winds,
  T-anomalies, q, warm-core sum)
- :func:`plot_state_diff`: map-view diff of two xarray Datasets

Map-view helpers (:func:`_add_map_labels`, :func:`_plot_marker`,
:func:`_plot_radius_circle`) are model-agnostic; they consume cartopy only.
"""

from __future__ import annotations

from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tcinit.constants import EARTH_RADIUS_A, HPA_TO_PA
from tcinit.naming import resolve

# ---------------------------------------------------------------------------
# Radial-vertical plot (K&C figure 7)
# ---------------------------------------------------------------------------


def plot_bogus(
    bv,
    filename: Optional[str] = None,
    *,
    figsize: tuple = (18, 10),
    ncontours: int = 40,
) -> None:
    """Render a K&C figure-7-style radial-vertical view of the bogus vortex.

    The plot is a 2x3 grid of ``contourf``:

      [0, 0] geopotential anomaly Phi'   (m^2 s^-2)
      [0, 1] temperature deviation T'    (K)
      [0, 2] RH envelope                 (fraction)
      [1, 0] tangential wind V_g, PBL    (m s^-1)
      [1, 1] radial wind v_r             (m s^-1)
      [1, 2] unused

    X = radial distance (km); Y = pressure (hPa, inverted). The bogus vortex
    is axisymmetric so these profiles are exact.
    """
    if not bv._built:
        raise RuntimeError("plot_bogus(): call BogusVortex.build() first")

    fig, axs = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)

    r = bv.r_radial_km
    p = bv.levels_hpa

    panels = [
        (
            axs[0, 0],
            bv.Phi_anom_radial_m2_s2,
            "geopotential anomaly",
            r"m$^2$ s$^{-2}$",
            "RdYlBu",
            True,
        ),
        (axs[0, 1], bv.T_anom_radial_K, "temperature deviation", "K", "RdBu_r", True),
        (axs[0, 2], bv.rh_envelope_radial, "RH envelope", "fraction", "Blues", False),
        (axs[1, 0], bv.V_g_radial_mps, "tangential wind", r"m s$^{-1}$", "PuOr", True),
        (axs[1, 1], bv.v_r_radial_mps, "radial wind", r"m s$^{-1}$", "PuOr", True),
    ]
    axs[1, 2].set_visible(False)

    p_sfc_hpa = bv.p_sfc_radial_pa / HPA_TO_PA

    surface_line = None
    for ax, field, title, units, cmap, diverging in panels:
        if not np.isfinite(field).any():
            ax.set_title(f"{title} (unavailable)")
            continue
        if diverging:
            absmax = float(np.nanmax(np.abs(field)))
            absmax = absmax if absmax > 0 else 1.0
            vmin, vmax = -absmax, absmax
        else:
            vmin = float(np.nanmin(field))
            vmax = float(np.nanmax(field))
            if vmax <= vmin:
                vmax = vmin + 1.0
        c = ax.contourf(r, p, field, ncontours, cmap=cmap, vmin=vmin, vmax=vmax)
        c.set_clim(vmin, vmax)
        cb = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(units)
        line_levels = np.linspace(vmin, vmax, 9)
        cs = ax.contour(
            r, p, field, levels=line_levels, colors="k", linewidths=0.5, alpha=0.6
        )
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
        (surface_line,) = ax.plot(
            r, p_sfc_hpa, color="k", linewidth=1.5, label="surface pressure"
        )
        ax.set_title(title)

    y_top = float(np.min(p)) - 5.0
    y_bot = max(float(np.max(p)), float(np.max(p_sfc_hpa))) + 10.0
    axs[0, 0].set_ylim(y_bot, y_top)
    for ax in axs[1, :]:
        ax.set_xlabel("radial distance (km)")
    for ax in axs[:, 0]:
        ax.set_ylabel("pressure (hPa)")

    if surface_line is not None:
        fig.legend(
            handles=[surface_line],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Map-view helpers (used by plot_state_diff in Phase 2)
# ---------------------------------------------------------------------------


def _add_map_labels(ax, show_bottom: bool, show_left: bool):
    """Attach lightweight lat/lon gridline labels to a cartopy axes."""
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_bottom
    gl.left_labels = show_left
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    return gl


def _plot_radius_circle(
    ax,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    *,
    color: str = "k",
    linestyle: str = "-",
    linewidth: float = 1.5,
    label: Optional[str] = None,
    n_samples: int = 200,
):
    """Draw a great-circle radius ring at (center_lat, center_lon)."""
    if (
        center_lat is None
        or center_lon is None
        or radius_km is None
        or not np.isfinite(center_lat)
        or not np.isfinite(center_lon)
        or not np.isfinite(radius_km)
        or radius_km <= 0
    ):
        return
    import cartopy.crs as ccrs
    from cartopy import geodesic

    g = geodesic.Geodesic(radius=EARTH_RADIUS_A, flattening=0)
    pts = g.circle(
        lon=center_lon,
        lat=center_lat,
        radius=1000.0 * radius_km,
        n_samples=n_samples,
    )
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        transform=ccrs.PlateCarree(),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )


def _plot_marker(ax, loc, marker, label, *, color="k", markersize=12, transform):
    """Plot a (lat, lon) marker; silently skip non-finite locations."""
    if loc is None or not np.isfinite(loc[0]) or not np.isfinite(loc[1]):
        return
    ax.plot(
        loc[1],
        loc[0],
        marker,
        markersize=markersize,
        label=label,
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.5,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# State-diff map view (was plot_batch_diff in the prototype)
# ---------------------------------------------------------------------------


def plot_state_diff(
    ds_before: xr.Dataset,
    ds_after: xr.Dataset,
    *,
    snapshot=None,
    var_map: Optional[Mapping[str, str]] = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    level_name: str = "level",
    filename: Optional[str] = None,
    figsize: tuple = (16, 10),
    pad_deg: float = 5.0,
    title: Optional[str] = None,
) -> None:
    """Six-panel diff view of two canonical xarray Datasets.

    Panels (all show ``after - before``):

      [0, 0] Delta MSLP (Pa)             [0, 1] Delta 2 m T (K)
      [0, 2] Delta |10 m wind| (m/s)     [1, 0] Delta T at 850 hPa (K)
      [1, 1] Delta T at 500 hPa (K)      [1, 2] Delta T at 200 hPa (K)

    The plot uses cartopy; ``snapshot`` (when given) is used to overlay the
    storm centre and the reference-disk-radius + storm-radius rings, and to
    crop the view around the box with a ``pad_deg`` margin.
    """
    import cartopy.crs as ccrs

    lats = np.asarray(ds_before[lat_name].values, dtype=float)
    lons = np.asarray(ds_before[lon_name].values, dtype=float)

    if snapshot is not None:
        lat_lo = float(np.min(snapshot.box_lats)) - pad_deg
        lat_hi = float(np.max(snapshot.box_lats)) + pad_deg
        lon_lo = float(np.min(snapshot.box_lons)) - pad_deg
        lon_hi = float(np.max(snapshot.box_lons)) + pad_deg
    else:
        lat_lo, lat_hi = float(np.min(lats)), float(np.max(lats))
        lon_lo, lon_hi = float(np.min(lons)), float(np.max(lons))

    def _surf_diff(canon: str) -> Optional[np.ndarray]:
        native = resolve(canon, var_map)
        if native is None:
            return None
        if native not in ds_before.data_vars or native not in ds_after.data_vars:
            return None
        return (ds_after[native].values - ds_before[native].values).astype(float)

    def _atmos_diff(canon: str, level_hpa: float) -> Optional[np.ndarray]:
        native = resolve(canon, var_map)
        if native is None:
            return None
        if native not in ds_before.data_vars or native not in ds_after.data_vars:
            return None
        try:
            a = ds_before[native].sel({level_name: level_hpa}).values
            b = ds_after[native].sel({level_name: level_hpa}).values
        except KeyError:
            return None
        return (b - a).astype(float)

    d_msl = _surf_diff("msl")
    d_t2m = _surf_diff("2t")
    d_u10 = _surf_diff("10u")
    d_v10 = _surf_diff("10v")
    if d_u10 is not None and d_v10 is not None:
        wind_mag = np.hypot(d_u10, d_v10)
    else:
        wind_mag = None
    d_T850 = _atmos_diff("t", 850.0)
    d_T500 = _atmos_diff("t", 500.0)
    d_T200 = _atmos_diff("t", 200.0)

    pc = ccrs.PlateCarree()
    fig, axs = plt.subplots(
        2,
        3,
        figsize=figsize,
        sharex=True,
        sharey=True,
        subplot_kw={"projection": pc},
    )
    fig.subplots_adjust(wspace=0.24, right=0.84)

    panels = [
        (axs[0, 0], r"$\Delta$ MSLP (Pa)", d_msl, True),
        (axs[0, 1], r"$\Delta$ 2 m T (K)", d_t2m, True),
        (axs[0, 2], r"$\Delta$ |10 m wind| (m s$^{-1}$)", wind_mag, False),
        (axs[1, 0], r"$\Delta$ T at 850 hPa (K)", d_T850, True),
        (axs[1, 1], r"$\Delta$ T at 500 hPa (K)", d_T500, True),
        (axs[1, 2], r"$\Delta$ T at 200 hPa (K)", d_T200, True),
    ]

    for i, ax in enumerate(axs.flat):
        ax.set_extent([lon_lo, lon_hi, lat_lo, lat_hi], crs=ccrs.Geodetic())
        ax.coastlines(linewidth=0.5)
        row, col = divmod(i, 3)
        _add_map_labels(ax, show_bottom=(row == 1), show_left=(col == 0))

    for ax, panel_title, diff, diverging in panels:
        if diff is None or not np.isfinite(diff).any():
            ax.set_title(f"{panel_title} (unavailable)")
            continue
        if diverging:
            absmax = float(np.nanmax(np.abs(diff)))
            absmax = absmax if absmax > 0 else 1e-9
            vmin, vmax, cmap = -absmax, absmax, "RdBu_r"
        else:
            vmin = 0.0
            vmax = max(float(np.nanmax(diff)), 1e-9)
            cmap = "Purples"
        c = ax.pcolormesh(
            lons,
            lats,
            diff,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=pc,
            shading="auto",
        )
        cb = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(panel_title.split("$\\Delta$ ")[-1])
        ax.set_title(panel_title)

    if snapshot is not None:
        loc = getattr(snapshot, "loc", None)
        rdr = getattr(snapshot, "rdr", None)
        mult = getattr(snapshot, "storm_radius_multiplier", 3.0)
        if loc is not None and rdr is not None and np.isfinite(rdr):
            for ax in axs.flat:
                _plot_marker(ax, loc, "x", None, color="k", markersize=10, transform=pc)
                _plot_radius_circle(ax, loc[0], loc[1], rdr, color="k", linestyle="-")
                _plot_radius_circle(
                    ax, loc[0], loc[1], mult * rdr, color="k", linestyle="--"
                )

    if title is not None:
        fig.suptitle(title)

    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_snapshot (3x3 map view, mirrors the AuroraSnapshot prototype)
# ---------------------------------------------------------------------------


def _vorticity_from_uv(
    u: np.ndarray, v: np.ndarray, lats_deg: np.ndarray, lons_deg: np.ndarray
) -> np.ndarray:
    """Relative vorticity zeta = dv/dx - du/dy on a lat/lon grid.

    Spherical-earth finite differences in metres, with a cos(lat) factor on
    the longitudinal metric. Edges fall back to np.gradient's reduced stencil.
    """
    lat_rad = np.deg2rad(lats_deg)
    dlat = np.deg2rad(lats_deg[1] - lats_deg[0])
    dlon = np.deg2rad(lons_deg[1] - lons_deg[0])
    dy_m = EARTH_RADIUS_A * dlat
    dx_m = EARTH_RADIUS_A * np.cos(lat_rad) * dlon  # per-row (lat)
    du_dy = np.gradient(u, axis=0) / dy_m
    dv_dx = np.gradient(v, axis=1) / dx_m[:, None]
    return dv_dx - du_dy


def _grid_field(
    snapshot,
    canon: str,
    ds: Optional[xr.Dataset],
    var_map: Optional[Mapping[str, str]],
    level_hpa: Optional[float],
    level_name: str,
) -> Optional[np.ndarray]:
    """Fetch a 2D field by canonical name, preferring a bundled snapshot.src_*."""
    # Surface fields on Snapshot.
    if level_hpa is None:
        attr = {"msl": "_msl", "2t": "_t2m", "10u": "_u10", "10v": "_v10"}.get(canon)
        if attr is not None and getattr(snapshot, attr, None) is not None:
            return np.asarray(getattr(snapshot, attr))
    # Bundled atmospheric field (snapshot.src_*).
    src = getattr(snapshot, f"src_{canon}", None)
    if src is not None and level_hpa is not None and snapshot.levels is not None:
        levels = np.asarray(snapshot.levels)
        if level_hpa in levels:
            k = int(np.where(levels == level_hpa)[0][0])
            return np.asarray(src[k])
    # Fall back to a live Dataset.
    if ds is not None:
        native = resolve(canon, var_map)
        if native is None or native not in ds.data_vars:
            return None
        da = ds[native]
        if level_hpa is None:
            return np.asarray(da.values, dtype=float)
        try:
            return np.asarray(da.sel({level_name: level_hpa}).values, dtype=float)
        except KeyError:
            return None
    return None


def plot_snapshot(
    snapshot,
    ds: Optional[xr.Dataset] = None,
    *,
    var_map: Optional[Mapping[str, str]] = None,
    level_name: str = "level",
    filename: Optional[str] = None,
    figsize: tuple = (16, 16),
    ncontours: int = 40,
    title: Optional[str] = None,
) -> None:
    """3x3 map-view of a storm-centric Snapshot.

    Panels (in order):

      [0,0] 850 hPa vorticity              [0,1] MSLP
      [0,2] |10 m wind|                    [1,0] T anomaly at 300 hPa
      [1,1] T anomaly at 500 hPa           [1,2] T anomaly at 700 hPa
      [2,0] q at 700 hPa                   [2,1] T200-500 avg anomaly
      [2,2] sum of 300+500+700 anomalies

    Vorticity, T anomalies, and the warm-core sum are derived on the fly from
    the Dataset (or from fields bundled onto ``snapshot`` by ``to_netcdf``).
    Panels fall back to an "unavailable" title when their required variable is
    missing.

    Storm centre (``snapshot.loc``), the reference-disk-radius ring
    (``snapshot.rdr``), and the storm-radius ring (``storm_radius_multiplier *
    rdr``) are overlaid on every panel.

    Args:
        snapshot: a :class:`tcinit.snapshot.Snapshot` with ``storm_computed``.
            May carry bundled ``src_t``/``src_u``/``src_v``/``src_q`` from
            :meth:`Snapshot.from_netcdf`.
        ds: optional canonical Dataset for atmospheric fields when the
            snapshot has no bundled fields.
        var_map: canonical -> native name map for ``ds``.
        level_name: level coordinate name in ``ds``.
    """
    import cartopy.crs as ccrs

    if not snapshot.storm_computed:
        raise RuntimeError("plot_snapshot: snapshot.storm_computed is False")

    lats = np.asarray(snapshot.box_lats, dtype=float)
    lons = np.asarray(snapshot.box_lons, dtype=float)
    pc = ccrs.PlateCarree()
    fig, axs = plt.subplots(
        3,
        3,
        figsize=figsize,
        sharex=True,
        sharey=True,
        subplot_kw={"projection": pc},
    )
    fig.subplots_adjust(wspace=0.24, right=0.84)

    xmin, xmax = float(lons.min()), float(lons.max())
    ymin, ymax = float(lats.min()), float(lats.max())

    for i, ax in enumerate(axs.flat):
        ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.Geodetic())
        ax.coastlines(linewidth=0.5)
        row, col = divmod(i, 3)
        _add_map_labels(ax, show_bottom=(row == 2), show_left=(col == 0))

    def _overlay(ax):
        """Storm centre, R_v and storm-radius rings on every panel."""
        loc = snapshot.loc
        rdr = snapshot.rdr
        if loc is None or rdr is None or not np.isfinite(rdr):
            return
        _plot_marker(ax, loc, "x", None, color="k", markersize=10, transform=pc)
        _plot_radius_circle(
            ax, loc[0], loc[1], rdr, color="k", linestyle="-", label="rdr"
        )
        mult = snapshot.storm_radius_multiplier
        _plot_radius_circle(
            ax,
            loc[0],
            loc[1],
            mult * rdr,
            color="k",
            linestyle="--",
            label="storm radius",
        )

    def _fetch(canon, level_hpa=None):
        return _grid_field(snapshot, canon, ds, var_map, level_hpa, level_name)

    def _draw(ax, field, title_str, cmap, *, diverging, cb_label):
        _overlay(ax)
        if field is None or not np.isfinite(field).any():
            ax.set_title(f"{title_str} (unavailable)")
            return
        if diverging:
            absmax = float(np.nanmax(np.abs(field)))
            absmax = absmax if absmax > 0 else 1.0
            vmin, vmax = -absmax, absmax
        else:
            vmin = 0.0 if cmap == "Purples" else float(np.nanmin(field))
            vmax = float(np.nanmax(field))
            if vmax <= vmin:
                vmax = vmin + 1e-9
        c = ax.contourf(
            lons, lats, field, ncontours, cmap=cmap, vmin=vmin, vmax=vmax, transform=pc
        )
        c.set_clim(vmin, vmax)
        cb = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cb_label)
        ax.set_title(title_str)

    # ---- Row 0: 850 vorticity, MSLP, |10 m wind| ---------------------------
    u850 = _fetch("u", 850.0)
    v850 = _fetch("v", 850.0)
    if u850 is not None and v850 is not None:
        vort = _vorticity_from_uv(u850, v850, lats, lons)
    else:
        vort = None
    _draw(
        axs[0, 0],
        vort,
        "850 hPa relative vorticity",
        "PuOr_r",
        diverging=True,
        cb_label=r"s$^{-1}$",
    )

    msl = _fetch("msl")
    _draw(axs[0, 1], msl, "MSLP", "bone", diverging=False, cb_label="Pa")

    u10 = _fetch("10u")
    v10 = _fetch("10v")
    if u10 is not None and v10 is not None:
        wspd10 = np.hypot(u10, v10)
    else:
        wspd10 = None
    _draw(
        axs[0, 2],
        wspd10,
        "10 m wind speed",
        "Purples",
        diverging=False,
        cb_label=r"m s$^{-1}$",
    )

    # ---- Row 1: T300, T500, T700 anomalies ---------------------------------
    env_mask = snapshot.env_mask

    def _anom(level_hpa):
        T = _fetch("t", level_hpa)
        if T is None or env_mask is None:
            return None
        env_vals = T[env_mask]
        if not np.isfinite(env_vals).any():
            return None
        return T - float(np.nanmean(env_vals))

    T300_anom = _anom(300.0)
    T500_anom = _anom(500.0)
    T700_anom = _anom(700.0)
    _draw(axs[1, 0], T300_anom, "T 300 hPa anomaly", "RdBu_r", diverging=True, cb_label="K")
    _draw(axs[1, 1], T500_anom, "T 500 hPa anomaly", "RdBu_r", diverging=True, cb_label="K")
    _draw(axs[1, 2], T700_anom, "T 700 hPa anomaly", "RdBu_r", diverging=True, cb_label="K")

    # ---- Row 2: q700, T200-500 avg anomaly, warm-core sum -----------------
    q700 = _fetch("q", 700.0)
    _draw(
        axs[2, 0],
        q700,
        "specific humidity at 700 hPa",
        "YlGnBu",
        diverging=False,
        cb_label=r"kg kg$^{-1}$",
    )

    T200 = _fetch("t", 200.0)
    T500 = _fetch("t", 500.0)
    if T200 is not None and T500 is not None and env_mask is not None:
        T_avg = 0.5 * (T200 + T500)
        env_vals = T_avg[env_mask]
        T_avg_anom = (
            T_avg - float(np.nanmean(env_vals)) if np.isfinite(env_vals).any() else None
        )
    else:
        T_avg_anom = None
    _draw(
        axs[2, 1],
        T_avg_anom,
        "mean(T200, T500) anomaly",
        "RdBu_r",
        diverging=True,
        cb_label="K",
    )

    if T300_anom is not None and T500_anom is not None and T700_anom is not None:
        warm_sum = T300_anom + T500_anom + T700_anom
    else:
        warm_sum = None
    _draw(
        axs[2, 2],
        warm_sum,
        "sum of 300+500+700 anomalies",
        "RdBu_r",
        diverging=True,
        cb_label="K",
    )

    # Single legend for the overlay rings, taken from the first panel.
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.85, 0.5),
            borderaxespad=0,
            frameon=True,
            fontsize=9,
        )

    if title is None:
        parts = []
        if snapshot.loc is not None and all(np.isfinite(snapshot.loc)):
            parts.append(f"loc=({snapshot.loc[0]:.2f} N, {snapshot.loc[1]:.2f} E)")
        if snapshot.rdr is not None and np.isfinite(snapshot.rdr):
            parts.append(f"rdr={snapshot.rdr:.0f} km")
        if snapshot.mslp_env_mean is not None and np.isfinite(snapshot.mslp_env_mean):
            parts.append(f"p_n={snapshot.mslp_env_mean/HPA_TO_PA:.1f} hPa")
        title = "Snapshot — " + ", ".join(parts) if parts else "Snapshot"
    fig.suptitle(title)

    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
