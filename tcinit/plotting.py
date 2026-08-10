"""Plotting helpers for the K&C bogus vortex and state differences.

- :func:`plot_bogus`: radial-vertical (r, p) cross-section (K&C figure-7 style)
- :func:`plot_state_diff`: map-view diff of two xarray Datasets (Phase 2)

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
