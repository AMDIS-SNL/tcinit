"""Time-ordered sequence of :class:`tcinit.Snapshot` objects from one rollout.

Model-agnostic port of CausalTC's ``SingleAuroraDataHistory``. Any backend
that can produce a canonical xarray Dataset per rollout step can build
``tcinit.Snapshot`` instances for each step and bundle them into a
:class:`RolloutSnapshots` container; this module then provides the
dataframe, best-track comparison, and multi-page PDF plotting that used to
live in ``aurora_data_history.py``.

The container itself is backend-free: no Aurora, torch, or model imports.
``run_rollout`` is intentionally not provided here -- construct the list of
snapshots however your backend sees fit (loop over Aurora predictions, read
HRES time steps, etc.) and pass them in.

Save/load:
    ``save(path)`` / ``load(path)`` -- pickle (default).
    ``to_netcdf(dir)`` / ``from_netcdf(dir)`` -- opt-in NetCDF bundle;
    one file per snapshot plus a small JSON index.

Plot functions are module-level (not methods) to mirror the layout of
``aurora_data_history.py``.
"""

from __future__ import annotations

import datetime as _dt
import enum
import json
import pickle
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from tcinit.constants import HPA_TO_PA, KNOTS_TO_MPS
from tcinit.diagnostics.mse import column_integrate_mse, mse_3d
from tcinit.earth_util import EARTH_RADIUS_A, haversine_dist
from tcinit.snapshot import Snapshot


# ---------------------------------------------------------------------------
# Warm-core criterion enum
# ---------------------------------------------------------------------------


class WarmCoreCriterion(enum.Enum):
    """Which per-level T-anomaly combination defines the warm core.

    - ``ANOMALY_SUM``: pointwise sum of 300, 500, 700 hPa anomalies
      (Dulac et al. 2023).
    - ``T200_500_AVG``: anomaly of the vertical mean of T200 and T500.
    """

    ANOMALY_SUM = 0
    T200_500_AVG = 1


# ---------------------------------------------------------------------------
# Per-snapshot derived diagnostics
# ---------------------------------------------------------------------------


_MPS_TO_KT = 1.0 / KNOTS_TO_MPS  # 1.9438...


def _level_index(levels: Optional[np.ndarray], target_hpa: float) -> Optional[int]:
    if levels is None:
        return None
    arr = np.asarray(levels)
    hits = np.where(arr == target_hpa)[0]
    return int(hits[0]) if hits.size else None


def _env_level_mean(src: Optional[np.ndarray], k: Optional[int], env_mask) -> float:
    if src is None or k is None or env_mask is None:
        return float("nan")
    vals = src[k][env_mask]
    if not np.isfinite(vals).any():
        return float("nan")
    return float(np.nanmean(vals))


def _anomaly_max_in_storm(
    field: Optional[np.ndarray], storm_mask: Optional[np.ndarray]
) -> float:
    """Max of ``field`` inside ``storm_mask``; NaN when either is missing."""
    if field is None or storm_mask is None:
        return float("nan")
    vals = field[storm_mask]
    if not np.isfinite(vals).any():
        return float("nan")
    return float(np.nanmax(vals))


def _plate_carree_for_lons(lons: np.ndarray):
    """Return a PlateCarree projection whose central longitude accommodates ``lons``.

    Cartopy's default ``PlateCarree()`` (central_longitude=0) collapses to a
    global strip when ``set_extent`` is called with a right bound > 180.
    When the data box crosses or straddles the antimeridian, use a
    dateline-centred projection instead so the box is drawn naturally.
    """
    import cartopy.crs as ccrs

    arr = np.asarray(lons, dtype=float)
    if arr.size and (float(np.nanmax(arr)) > 180.0 or float(np.nanmax(arr) - np.nanmin(arr)) > 180.0):
        return ccrs.PlateCarree(central_longitude=180.0)
    return ccrs.PlateCarree()


def _vorticity_from_uv(
    u: np.ndarray, v: np.ndarray, lats_deg: np.ndarray, lons_deg: np.ndarray
) -> np.ndarray:
    """Relative vorticity zeta = dv/dx - du/dy on a lat/lon grid.

    Centered differences on a spherical Earth; cos(lat) metric on dlon.
    Duplicated from tcinit.plotting (keeps rollout_snapshots self-contained).
    """
    lat_rad = np.deg2rad(lats_deg)
    dlat = np.deg2rad(lats_deg[1] - lats_deg[0])
    dlon = np.deg2rad(lons_deg[1] - lons_deg[0])
    dy_m = EARTH_RADIUS_A * dlat
    dx_m = EARTH_RADIUS_A * np.cos(lat_rad) * dlon
    du_dy = np.gradient(u, axis=0) / dy_m
    dv_dx = np.gradient(v, axis=1) / dx_m[:, None]
    return dv_dx - du_dy


def snapshot_diagnostics(
    snap: Snapshot,
    *,
    warm_core_criterion: WarmCoreCriterion = WarmCoreCriterion.T200_500_AVG,
) -> dict:
    """Derived scalars + 2D fields a RolloutSnapshots consumer needs per snapshot.

    All fields derive from what ``Snapshot`` already carries plus the
    bundled ``src_t``/``src_u``/``src_v``/``src_q``/``src_z`` fields written
    by ``Snapshot.to_netcdf(source_ds=...)``. Missing inputs -> NaN or
    ``None`` outputs; no exceptions.

    Returns a dict with keys:
        mslp, mslp_min_val, mslp_min_loc, windspd10, windspd10_max_val,
        windspd10_max_loc, vorticity850, vorticity850_max_val,
        vorticity850_max_loc, T300_env_mean, T500_env_mean, T700_env_mean,
        T_avg_200_500_env_mean, T300_anomaly, T500_anomaly, T700_anomaly,
        T_avg_200_500_anomaly, warm_core, warm_core_max,
        T300_anomaly_max_val, T500_anomaly_max_val, T700_anomaly_max_val,
        T_avg_200_500_anomaly_max_val, mse, mse_column, warm_core_criterion.
    """
    out: dict = {"warm_core_criterion": warm_core_criterion}

    lats = snap.box_lats
    lons = snap.box_lons
    storm_mask = snap.storm_mask
    env_mask = snap.env_mask

    # --- surface scalars/fields ---
    msl = snap._msl
    out["mslp"] = msl
    if msl is not None and np.isfinite(msl).any() and snap.loc is not None:
        # Snapshot.detect_storm set self.loc to the MSLP minimum, so the min
        # value is the field at that location.
        if (
            snap.loc[0] is not None
            and snap.loc[1] is not None
            and np.isfinite(snap.loc[0])
            and np.isfinite(snap.loc[1])
            and lats is not None
            and lons is not None
        ):
            i = int(np.argmin(np.abs(lats - snap.loc[0])))
            j = int(np.argmin(np.abs(lons - snap.loc[1])))
            out["mslp_min_val"] = float(msl[i, j])
            out["mslp_min_loc"] = (float(lats[i]), float(lons[j]))
        else:
            k = np.unravel_index(np.nanargmin(msl), msl.shape)
            out["mslp_min_val"] = float(msl[k])
            out["mslp_min_loc"] = (float(lats[k[0]]), float(lons[k[1]]))
    else:
        out["mslp_min_val"] = float("nan")
        out["mslp_min_loc"] = (float("nan"), float("nan"))

    if snap._u10 is not None and snap._v10 is not None:
        wspd = np.hypot(snap._u10, snap._v10)
        out["windspd10"] = wspd
        search = wspd if storm_mask is None else np.where(storm_mask, wspd, np.nan)
        if np.isfinite(search).any():
            k = np.unravel_index(np.nanargmax(search), search.shape)
            out["windspd10_max_val"] = float(wspd[k])
            out["windspd10_max_loc"] = (float(lats[k[0]]), float(lons[k[1]]))
        else:
            out["windspd10_max_val"] = float("nan")
            out["windspd10_max_loc"] = (float("nan"), float("nan"))
    else:
        out["windspd10"] = None
        out["windspd10_max_val"] = float("nan")
        out["windspd10_max_loc"] = (float("nan"), float("nan"))

    # --- vorticity from bundled u/v at 850 hPa ---
    src_u = getattr(snap, "src_u", None)
    src_v = getattr(snap, "src_v", None)
    k850 = _level_index(snap.levels, 850.0)
    if src_u is not None and src_v is not None and k850 is not None and lats is not None:
        vort = _vorticity_from_uv(src_u[k850], src_v[k850], lats, lons)
        out["vorticity850"] = vort
        # Cyclonic sign: + in NH, - in SH. Finding max of (sign * vort) picks
        # the strongest cyclonic cell; the stored value is the raw signed
        # vorticity at that cell (negative for a SH cyclone).
        if snap.loc is not None and np.isfinite(snap.loc[0]):
            cyclonic_sign = -1.0 if snap.loc[0] < 0 else 1.0
        else:
            cyclonic_sign = 1.0 if float(np.nanmean(lats)) >= 0 else -1.0
        search = (
            cyclonic_sign * vort if storm_mask is None
            else np.where(storm_mask, cyclonic_sign * vort, np.nan)
        )
        if np.isfinite(search).any():
            k = np.unravel_index(np.nanargmax(search), search.shape)
            out["vorticity850_max_val"] = float(vort[k])
            out["vorticity850_max_loc"] = (float(lats[k[0]]), float(lons[k[1]]))
        else:
            out["vorticity850_max_val"] = float("nan")
            out["vorticity850_max_loc"] = (float("nan"), float("nan"))
    else:
        out["vorticity850"] = None
        out["vorticity850_max_val"] = float("nan")
        out["vorticity850_max_loc"] = (float("nan"), float("nan"))

    # --- temperature env means + anomalies ---
    src_t = getattr(snap, "src_t", None)
    k200 = _level_index(snap.levels, 200.0)
    k300 = _level_index(snap.levels, 300.0)
    k500 = _level_index(snap.levels, 500.0)
    k700 = _level_index(snap.levels, 700.0)

    out["T300_env_mean"] = _env_level_mean(src_t, k300, env_mask)
    out["T500_env_mean"] = _env_level_mean(src_t, k500, env_mask)
    out["T700_env_mean"] = _env_level_mean(src_t, k700, env_mask)

    def _anom_2d(k_idx, env_mean):
        if src_t is None or k_idx is None or not np.isfinite(env_mean):
            return None
        return src_t[k_idx] - env_mean

    T300_anom = _anom_2d(k300, out["T300_env_mean"])
    T500_anom = _anom_2d(k500, out["T500_env_mean"])
    T700_anom = _anom_2d(k700, out["T700_env_mean"])
    out["T300_anomaly"] = T300_anom
    out["T500_anomaly"] = T500_anom
    out["T700_anomaly"] = T700_anom
    out["T300_anomaly_max_val"] = _anomaly_max_in_storm(T300_anom, storm_mask)
    out["T500_anomaly_max_val"] = _anomaly_max_in_storm(T500_anom, storm_mask)
    out["T700_anomaly_max_val"] = _anomaly_max_in_storm(T700_anom, storm_mask)

    if src_t is not None and k200 is not None and k500 is not None:
        T_avg = 0.5 * (src_t[k200] + src_t[k500])
        env_mean = float("nan")
        if env_mask is not None:
            vals = T_avg[env_mask]
            if np.isfinite(vals).any():
                env_mean = float(np.nanmean(vals))
        out["T_avg_200_500_env_mean"] = env_mean
        if np.isfinite(env_mean):
            T_avg_anom = T_avg - env_mean
            out["T_avg_200_500_anomaly"] = T_avg_anom
            out["T_avg_200_500_anomaly_max_val"] = _anomaly_max_in_storm(
                T_avg_anom, storm_mask
            )
        else:
            out["T_avg_200_500_anomaly"] = None
            out["T_avg_200_500_anomaly_max_val"] = float("nan")
    else:
        out["T_avg_200_500_env_mean"] = float("nan")
        out["T_avg_200_500_anomaly"] = None
        out["T_avg_200_500_anomaly_max_val"] = float("nan")

    # --- warm core per criterion ---
    if warm_core_criterion == WarmCoreCriterion.ANOMALY_SUM:
        if T300_anom is not None and T500_anom is not None and T700_anom is not None:
            out["warm_core"] = T300_anom + T500_anom + T700_anom
        else:
            out["warm_core"] = None
    elif warm_core_criterion == WarmCoreCriterion.T200_500_AVG:
        out["warm_core"] = out["T_avg_200_500_anomaly"]
    else:
        raise ValueError(f"unknown WarmCoreCriterion: {warm_core_criterion!r}")

    if out["warm_core"] is not None and np.isfinite(out["warm_core"]).any():
        out["warm_core_max"] = float(np.nanmax(out["warm_core"]))
    else:
        out["warm_core_max"] = float("nan")

    # --- MSE from bundled t/z/q ---
    src_z = getattr(snap, "src_z", None)
    src_q = getattr(snap, "src_q", None)
    if (
        src_t is not None
        and src_z is not None
        and src_q is not None
        and snap.levels is not None
    ):
        mse = mse_3d(src_t, src_z, src_q)
        out["mse"] = mse
        out["mse_column"] = column_integrate_mse(
            mse, snap.levels, level_axis=0, p_surface_pa=msl
        )
    else:
        out["mse"] = None
        out["mse_column"] = None

    return out


# ---------------------------------------------------------------------------
# RolloutSnapshots container
# ---------------------------------------------------------------------------


class RolloutSnapshots:
    """Container for a time-ordered sequence of :class:`Snapshot` objects.

    Attributes:
        cyclone_id: Storm identifier (free text; typically the best-track ID).
        start_time: Analysis/initialization time of the rollout.
        snapshots: Ordered list of ``Snapshot`` objects (one per rollout step
            including the analysis time).
        intervened: Marker for whether a bogus/intervention was applied.
        warm_core_criterion: Which per-level T combination defines the warm
            core (used by plotting/dataframe).
    """

    def __init__(
        self,
        cyclone_id: str,
        start_time: _dt.datetime,
        snapshots: Optional[Iterable[Snapshot]] = None,
        *,
        intervened: bool = False,
        warm_core_criterion: WarmCoreCriterion = WarmCoreCriterion.T200_500_AVG,
    ) -> None:
        self.cyclone_id = str(cyclone_id)
        self.start_time = start_time
        self.snapshots: List[Snapshot] = list(snapshots) if snapshots else []
        self.intervened = bool(intervened)
        self.warm_core_criterion = warm_core_criterion
        self._diag_cache: dict = {}

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self):
        return iter(self.snapshots)

    def __getitem__(self, idx):
        return self.snapshots[idx]

    def append(self, snap: Snapshot) -> None:
        self.snapshots.append(snap)
        self._diag_cache.pop(id(snap), None)

    @property
    def rollout_length(self) -> int:
        """Number of snapshots in the rollout."""
        return len(self.snapshots)

    @property
    def valid_times(self) -> List[Optional[_dt.datetime]]:
        """Per-snapshot valid_times, in the order stored."""
        return [getattr(s, "valid_time", None) for s in self.snapshots]

    def diagnostics(self, snap: Snapshot) -> dict:
        """Return cached derived diagnostics for ``snap`` (compute on miss)."""
        key = id(snap)
        if key not in self._diag_cache:
            self._diag_cache[key] = snapshot_diagnostics(
                snap, warm_core_criterion=self.warm_core_criterion
            )
        return self._diag_cache[key]

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    @staticmethod
    def _pickle_filename(
        root: Path, cyclone_id: str, start_time: _dt.datetime,
        rollout_length: int, intervened: bool,
    ) -> Path:
        ts = start_time.strftime("%Y%m%d%H")
        suffix = "_intervened" if intervened else ""
        return Path(root) / cyclone_id / f"{ts}_steps{rollout_length:02d}{suffix}.pkl"

    def save(self, root: Union[str, Path], *, verbose: bool = False) -> Path:
        """Pickle this history under ``root/<cyclone_id>/<ts>_stepsNN[_intervened].pkl``.

        Returns the written path.
        """
        path = self._pickle_filename(
            Path(root), self.cyclone_id, self.start_time,
            self.rollout_length, self.intervened,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print(f"Wrote {path}")
        return path

    @classmethod
    def load(
        cls,
        cyclone_id: str,
        start_time: _dt.datetime,
        rollout_length: int,
        root: Union[str, Path],
        *,
        intervened: bool = False,
    ) -> "RolloutSnapshots":
        path = cls._pickle_filename(
            Path(root), cyclone_id, start_time, rollout_length, intervened,
        )
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as f:
            return pickle.load(f)

    def to_netcdf(
        self,
        root: Union[str, Path],
        *,
        bundle_source_fields: bool = True,
        verbose: bool = False,
    ) -> Path:
        """Write one NetCDF per snapshot plus an index.json under ``root``.

        Opt-in alternative to :meth:`save`. Each snapshot is written via its
        own :meth:`Snapshot.to_netcdf`; the index JSON records the order and
        metadata needed by :meth:`from_netcdf`.
        """
        root = Path(root) / self.cyclone_id
        ts = self.start_time.strftime("%Y%m%d%H")
        suffix = "_intervened" if self.intervened else ""
        rollout_dir = root / f"{ts}_steps{self.rollout_length:02d}{suffix}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        snap_files = []
        for i, snap in enumerate(self.snapshots):
            fname = f"snap_{i:03d}.nc"
            path = rollout_dir / fname
            # bundle_source_fields=True is only meaningful if snap carries
            # src_t etc. Snapshot.to_netcdf treats a missing source_ds as
            # "no bundling"; here we re-bundle any already-present src_* by
            # re-building a tiny xr.Dataset from them.
            source_ds = None
            if bundle_source_fields and _has_bundled_fields(snap):
                source_ds = _snap_bundled_to_dataset(snap)
            snap.to_netcdf(path, source_ds=source_ds)
            snap_files.append(fname)
        index = {
            "cyclone_id": self.cyclone_id,
            "start_time": self.start_time.isoformat(),
            "intervened": self.intervened,
            "warm_core_criterion": self.warm_core_criterion.name,
            "snapshots": snap_files,
            "valid_times": [
                (s.valid_time.isoformat() if s.valid_time is not None else None)
                for s in self.snapshots
            ],
        }
        with (rollout_dir / "index.json").open("w") as f:
            json.dump(index, f, indent=2)
        if verbose:
            print(f"Wrote {rollout_dir} ({len(snap_files)} snapshots + index.json)")
        return rollout_dir

    @classmethod
    def from_netcdf(cls, rollout_dir: Union[str, Path]) -> "RolloutSnapshots":
        """Inverse of :meth:`to_netcdf`."""
        rollout_dir = Path(rollout_dir)
        with (rollout_dir / "index.json").open() as f:
            index = json.load(f)
        snaps = [Snapshot.from_netcdf(rollout_dir / fname) for fname in index["snapshots"]]
        # Restore valid_times from index when the Snapshot NetCDF didn't carry one.
        for snap, iso in zip(snaps, index.get("valid_times", [])):
            if snap.valid_time is None and iso is not None:
                snap.valid_time = _dt.datetime.fromisoformat(iso)
        criterion = WarmCoreCriterion[index.get("warm_core_criterion", "T200_500_AVG")]
        return cls(
            cyclone_id=index["cyclone_id"],
            start_time=_dt.datetime.fromisoformat(index["start_time"]),
            snapshots=snaps,
            intervened=index.get("intervened", False),
            warm_core_criterion=criterion,
        )

    # ------------------------------------------------------------------
    # Dataframe exports
    # ------------------------------------------------------------------

    def to_dataframe(
        self,
        *,
        include_detection: bool = True,
        vorticity_threshold: float = 15e-5,
        warm_core_threshold: float = 1.0,
        windspeed_threshold: float = 10.0,
        datetime_col: str = "datetime",
        use_best_track_units: bool = True,
        include_raw_units: bool = True,
        include_location_details: bool = True,
        include_thermo: bool = True,
    ) -> pd.DataFrame:
        """One row per snapshot with scalar diagnostics (best-track-comparable)."""
        rows = []
        for snap in self.snapshots:
            diag = self.diagnostics(snap)
            loc = snap.loc if snap.loc is not None else (float("nan"), float("nan"))
            mslp_min_pa = diag["mslp_min_val"]
            vmax_mps = diag["windspd10_max_val"]
            mslp_min_hpa = (
                mslp_min_pa / 100.0 if np.isfinite(mslp_min_pa) else float("nan")
            )
            vmax_kt = vmax_mps * _MPS_TO_KT if np.isfinite(vmax_mps) else float("nan")
            if use_best_track_units:
                vmax_out, mslp_out = vmax_kt, mslp_min_hpa
            else:
                vmax_out, mslp_out = vmax_mps, mslp_min_pa

            lead_hours = None
            if snap.valid_time is not None and self.start_time is not None:
                lead_hours = int(
                    (snap.valid_time - self.start_time).total_seconds() / 3600.0
                )

            row = {
                "cyclone_id": self.cyclone_id,
                "start_time": self.start_time,
                datetime_col: snap.valid_time,
                "valid_time": snap.valid_time,
                "lead_hours": lead_hours,
                "intervened": self.intervened,
                "LAT_DEG": float(loc[0]),
                "LON_DEG": float(loc[1]),
                "VMAX": vmax_out,
                "MSLP": mslp_out,
                "vorticity850_max_val": diag["vorticity850_max_val"],
                "warm_core": diag["warm_core_max"],
                "rdr_km": snap.rdr if snap.rdr is not None else float("nan"),
            }

            if include_raw_units:
                row.update({
                    "VMAX_MPS": vmax_mps,
                    "VMAX_KT": vmax_kt,
                    "MSLP_PA": mslp_min_pa,
                    "MSLP_HPA": mslp_min_hpa,
                })

            if include_location_details:
                mslp_loc = diag["mslp_min_loc"]
                vort_loc = diag["vorticity850_max_loc"]
                wspd_loc = diag["windspd10_max_loc"]
                row.update({
                    "mslp_min_lat": mslp_loc[0],
                    "mslp_min_lon": mslp_loc[1],
                    "mslp_min_val_pa": mslp_min_pa,
                    "mslp_min_val_hpa": mslp_min_hpa,
                    "vorticity850_max_lat": vort_loc[0],
                    "vorticity850_max_lon": vort_loc[1],
                    "windspd10_max_lat": wspd_loc[0],
                    "windspd10_max_lon": wspd_loc[1],
                    "windspd10_max_mps": vmax_mps,
                    "windspd10_max_kt": vmax_kt,
                })

            if include_thermo:
                row.update({
                    "T300_env_mean": diag["T300_env_mean"],
                    "T500_env_mean": diag["T500_env_mean"],
                    "T700_env_mean": diag["T700_env_mean"],
                    "T_avg_200_500_env_mean": diag["T_avg_200_500_env_mean"],
                    "T300_anomaly": diag["T300_anomaly_max_val"],
                    "T500_anomaly": diag["T500_anomaly_max_val"],
                    "T700_anomaly": diag["T700_anomaly_max_val"],
                    "T_avg_200_500_anomaly": diag["T_avg_200_500_anomaly_max_val"],
                })

            if include_detection:
                hemi_sign = -1.0 if loc[0] is not None and loc[0] < 0 else 1.0
                cyc_vort = hemi_sign * diag["vorticity850_max_val"]
                row["meets_vorticity_threshold"] = (
                    bool(cyc_vort > vorticity_threshold)
                    if np.isfinite(cyc_vort) else pd.NA
                )
                row["meets_windspeed_threshold"] = (
                    bool(vmax_mps > windspeed_threshold)
                    if np.isfinite(vmax_mps) else pd.NA
                )
                row["meets_warm_core_threshold"] = (
                    bool(diag["warm_core_max"] > warm_core_threshold)
                    if np.isfinite(diag["warm_core_max"]) else pd.NA
                )
                vals = [
                    row["meets_vorticity_threshold"],
                    row["meets_windspeed_threshold"],
                    row["meets_warm_core_threshold"],
                ]
                row["storm_found"] = (
                    pd.NA if any(v is pd.NA for v in vals)
                    else bool(all(vals))
                )

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("valid_time").reset_index(drop=True)
        return df

    def to_comparison_dataframe(
        self,
        best_track,
        *,
        include_detection: bool = True,
        vorticity_threshold: float = 15e-5,
        warm_core_threshold: float = 1.0,
        windspeed_threshold: float = 10.0,
        use_best_track_units: bool = True,
        include_raw_units: bool = True,
        include_location_details: bool = True,
        include_thermo: bool = True,
        include_best_track_status: bool = True,
        compute_position_distance_km: bool = True,
        model_prefix: str = "model_",
        best_track_prefix: str = "bt_",
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Merge the model-side dataframe with best-track rows on ``datetime``.

        Columns from :meth:`to_dataframe` are renamed with ``model_prefix``
        (default ``model_``). Best-track columns are renamed with
        ``best_track_prefix``. Also computes ``position_error_km``,
        ``dVMAX``, and ``dMSLP`` when both sides are present.
        """
        model_df = self.to_dataframe(
            include_detection=include_detection,
            vorticity_threshold=vorticity_threshold,
            warm_core_threshold=warm_core_threshold,
            windspeed_threshold=windspeed_threshold,
            datetime_col="datetime",
            use_best_track_units=use_best_track_units,
            include_raw_units=include_raw_units,
            include_location_details=include_location_details,
            include_thermo=include_thermo,
        )
        if model_df.empty:
            return model_df.copy()

        model_df = model_df.rename(columns={
            col: f"{model_prefix}{col}"
            for col in model_df.columns if col != "datetime"
        })

        bt_df = best_track.df_best_tracks_history.copy()
        bt_cols = ["datetimes", "LAT_DEG", "LON_DEG", "VMAX", "MSLP"]
        if include_best_track_status and "STATUS" in bt_df.columns:
            bt_cols.append("STATUS")
        bt_cols = [c for c in bt_cols if c in bt_df.columns]
        bt_df = bt_df[bt_cols].rename(columns={
            "datetimes": "datetime",
            "LAT_DEG": f"{best_track_prefix}LAT_DEG",
            "LON_DEG": f"{best_track_prefix}LON_DEG",
            "VMAX": f"{best_track_prefix}VMAX",
            "MSLP": f"{best_track_prefix}MSLP",
            "STATUS": f"{best_track_prefix}STATUS",
        })

        merged = model_df.merge(bt_df, on="datetime", how="left")

        aur_lat = f"{model_prefix}LAT_DEG"
        aur_lon = f"{model_prefix}LON_DEG"
        bt_lat = f"{best_track_prefix}LAT_DEG"
        bt_lon = f"{best_track_prefix}LON_DEG"
        if compute_position_distance_km and all(
            c in merged.columns for c in (aur_lat, aur_lon, bt_lat, bt_lon)
        ):
            def _pos_dist(row):
                vals = (row[aur_lat], row[aur_lon], row[bt_lat], row[bt_lon])
                if all(pd.notna(v) for v in vals):
                    return haversine_dist(
                        float(row[aur_lat]), float(row[aur_lon]),
                        float(row[bt_lat]), float(row[bt_lon]),
                    )
                return float("nan")
            merged["position_error_km"] = merged.apply(_pos_dist, axis=1)

        for metric, label in (("VMAX", "dVMAX"), ("MSLP", "dMSLP")):
            m_col = f"{model_prefix}{metric}"
            b_col = f"{best_track_prefix}{metric}"
            if m_col in merged.columns and b_col in merged.columns:
                merged[label] = (
                    pd.to_numeric(merged[m_col], errors="coerce")
                    - pd.to_numeric(merged[b_col], errors="coerce")
                )

        merged = merged.sort_values("datetime").reset_index(drop=True)

        if verbose:
            self._print_comparison(
                merged,
                model_prefix=model_prefix,
                best_track_prefix=best_track_prefix,
            )

        return merged

    def _print_comparison(
        self, df: pd.DataFrame, *, model_prefix: str, best_track_prefix: str,
    ) -> None:
        aur_vmax = f"{model_prefix}VMAX"
        aur_mslp = f"{model_prefix}MSLP"
        bt_vmax = f"{best_track_prefix}VMAX"
        bt_mslp = f"{best_track_prefix}MSLP"
        wanted = [
            "datetime", aur_vmax, bt_vmax, "dVMAX",
            aur_mslp, bt_mslp, "dMSLP", "position_error_km",
        ]
        cols = [c for c in wanted if c in df.columns]
        view = df[cols] if cols else df
        print(
            f"Model vs best-track comparison: {self.cyclone_id}, "
            f"start {self.start_time}"
        )
        print(view.to_string(index=False))

        def _stats(series):
            s = pd.to_numeric(series, errors="coerce").abs().dropna()
            if s.empty:
                return float("nan"), float("nan"), float("nan")
            return float(s.mean()), float(s.max()), float(np.sqrt((s**2).mean()))

        rows = []
        if "dVMAX" in df.columns:
            rows.append(("|dVMAX|", "kt", *_stats(df["dVMAX"])))
        if "dMSLP" in df.columns:
            rows.append(("|dMSLP|", "hPa", *_stats(df["dMSLP"])))
        if "position_error_km" in df.columns:
            rows.append(("position_err", "km", *_stats(df["position_error_km"])))
        if rows:
            n = int(df[[c for c in ("dVMAX", "dMSLP", "position_error_km")
                        if c in df.columns]].dropna(how="all").shape[0])
            print(f"\nSummary (n={n}):")
            for label, unit, mean_v, max_v, rmse_v in rows:
                print(
                    f"  {label:<14} mean={mean_v:.2f} {unit}  "
                    f"max={max_v:.2f} {unit}  rmse={rmse_v:.2f} {unit}"
                )

    # ------------------------------------------------------------------
    # MSE time series
    # ------------------------------------------------------------------

    def mse_column_timeseries(
        self, mask_attr: Optional[str] = "storm_mask",
    ) -> tuple:
        """Mean column MSE inside a per-snapshot boolean mask, over the rollout.

        Args:
            mask_attr: ``'storm_mask'``, ``'env_mask'``, or ``None`` for the
                whole box.

        Returns:
            ``(times, values)`` -- ``times`` is a list of valid_times;
            ``values`` is a float array of mean column MSE [J/m^2], NaN
            where the mask is empty or MSE isn't available.
        """
        times, values = [], []
        for snap in self.snapshots:
            times.append(snap.valid_time)
            mse_col = self.diagnostics(snap).get("mse_column")
            if mse_col is None or not np.isfinite(mse_col).any():
                values.append(float("nan"))
                continue
            if mask_attr is None:
                values.append(float(np.nanmean(mse_col)))
            else:
                mask = getattr(snap, mask_attr, None)
                if mask is None or not mask.any():
                    values.append(float("nan"))
                else:
                    values.append(float(np.nanmean(mse_col[mask])))
        return times, np.array(values, dtype=float)

    def mse_anomaly_timeseries(self) -> tuple:
        """Storm-minus-environment mean column MSE over the rollout."""
        times, storm_vals = self.mse_column_timeseries("storm_mask")
        _, env_vals = self.mse_column_timeseries("env_mask")
        return times, storm_vals - env_vals

    def mse_vertical_profile_at_center(self) -> tuple:
        """MSE profile at storm centre at each snapshot.

        Returns ``(times, levels_hpa, profiles)``:
            ``levels_hpa`` is None when no snapshot has MSE;
            ``profiles`` has shape ``(n_times, n_levels)`` [J/kg].
        """
        times, profiles = [], []
        levels_hpa = None
        for snap in self.snapshots:
            mse = self.diagnostics(snap).get("mse")
            if mse is None or snap.levels is None:
                continue
            loc = snap.loc
            if loc is None or not (np.isfinite(loc[0]) and np.isfinite(loc[1])):
                continue
            lat_idx = int(np.argmin(np.abs(snap.box_lats - loc[0])))
            lon_idx = int(np.argmin(np.abs(snap.box_lons - loc[1])))
            profiles.append(mse[:, lat_idx, lon_idx])
            times.append(snap.valid_time)
            levels_hpa = snap.levels
        if not profiles:
            return [], None, np.empty((0, 0))
        return times, levels_hpa, np.stack(profiles, axis=0)


# ---------------------------------------------------------------------------
# Internals for NetCDF bundling
# ---------------------------------------------------------------------------


def _has_bundled_fields(snap: Snapshot) -> bool:
    return any(getattr(snap, f"src_{c}", None) is not None for c in ("t", "u", "v", "q", "z"))


def _snap_bundled_to_dataset(snap: Snapshot):
    """Repackage snap.src_* back into an xr.Dataset so Snapshot.to_netcdf can bundle it."""
    import xarray as xr
    data_vars = {}
    for canon in ("t", "u", "v", "q", "z"):
        arr = getattr(snap, f"src_{canon}", None)
        if arr is not None:
            data_vars[canon] = xr.DataArray(
                arr, dims=("level", "latitude", "longitude"),
            )
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": snap.box_lats,
            "longitude": snap.box_lons,
            "level": snap.levels,
        },
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
#
# Multi-page PDF layout (adapted from aurora_data_history.py):
#   * one 2x2 page per snapshot (vorticity, MSLP, 10 m wind, warm core)
#   * summary line plot (MSLP + VMAX; vorticity max + warm_core max)
#   * pressure-wind scatter
#   * storm-track map
#   * MSE 3-panel diagnostics


def _warm_core_panel_title(criterion: WarmCoreCriterion) -> str:
    if criterion == WarmCoreCriterion.ANOMALY_SUM:
        return "warm core: sum of 300, 500, 700 anomalies"
    if criterion == WarmCoreCriterion.T200_500_AVG:
        return "warm core: T200-500 avg anomaly"
    return "warm core"


def _symmetric_lim(arrays) -> float:
    vals = []
    for a in arrays:
        if a is None:
            continue
        finite = np.isfinite(a)
        if not finite.any():
            continue
        vals.append(float(np.nanmax(np.abs(a))))
    return max(vals) if vals else float("nan")


def _linear_lims(arrays) -> tuple:
    mins, maxs = [], []
    for a in arrays:
        if a is None:
            continue
        finite = np.isfinite(a)
        if not finite.any():
            continue
        mins.append(float(np.nanmin(a)))
        maxs.append(float(np.nanmax(a)))
    if not mins:
        return float("nan"), float("nan")
    return min(mins), max(maxs)


def _plot_snapshot_page(
    history: "RolloutSnapshots",
    snap: Snapshot,
    pdf,
    *,
    vort_lim: float,
    mslp_vmin_hpa: float,
    mslp_vmax_hpa: float,
    wspd_vmin: float,
    wspd_vmax: float,
    wc_lim: float,
) -> None:
    """Render one 2x2 snapshot page onto *pdf*."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    from tcinit.plotting import _add_map_labels, _plot_marker, _plot_radius_circle

    diag = history.diagnostics(snap)
    # Transform (how input coords are interpreted) always stays at central=0;
    # only the axes projection shifts to the dateline when needed.
    pc = ccrs.PlateCarree()
    axes_proj = _plate_carree_for_lons(snap.box_lons)
    fig, axs = plt.subplots(
        2, 2, figsize=(12, 12), sharex=True, sharey=True,
        subplot_kw={"projection": axes_proj},
    )
    fig.subplots_adjust(wspace=0.24, hspace=0.18, right=0.9)

    xmin = float(np.amin(snap.box_lons))
    xmax = float(np.amax(snap.box_lons))
    ymin = float(np.amin(snap.box_lats))
    ymax = float(np.amax(snap.box_lats))

    mslp_hpa = (
        diag["mslp"] / HPA_TO_PA if diag["mslp"] is not None else None
    )
    mslp_min_hpa = (
        diag["mslp_min_val"] / HPA_TO_PA
        if np.isfinite(diag["mslp_min_val"]) else float("nan")
    )

    def _fmt(v, spec):
        return format(v, spec) if np.isfinite(v) else "n/a"

    vort_title = (
        f"850 hPa vorticity (s$^{{-1}}$, "
        f"max = {_fmt(diag['vorticity850_max_val'], '.2e')})"
    )
    mslp_title = f"MSLP (hPa, min = {_fmt(mslp_min_hpa, '.2f')})"
    wspd_title = (
        f"10 m windspeed (m s$^{{-1}}$, "
        f"max = {_fmt(diag['windspd10_max_val'], '.2f')})"
    )
    wc_title = (
        f"{_warm_core_panel_title(history.warm_core_criterion)} "
        f"(K, max = {_fmt(diag['warm_core_max'], '.2f')})"
    )

    panels = [
        ((0, 0), diag["vorticity850"], vort_title, "PuOr_r", -vort_lim, vort_lim),
        ((0, 1), mslp_hpa, mslp_title, "viridis", mslp_vmin_hpa, mslp_vmax_hpa),
        ((1, 0), diag["windspd10"], wspd_title, "viridis", wspd_vmin, wspd_vmax),
        ((1, 1), diag["warm_core"], wc_title, "RdBu_r", -wc_lim, wc_lim),
    ]

    for (row, col), field, title, cmap, vmin, vmax in panels:
        ax = axs[row, col]
        ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
        ax.coastlines()
        _add_map_labels(ax, show_bottom=(row == 1), show_left=(col == 0))

        if (
            field is None
            or not np.isfinite(vmin)
            or not np.isfinite(vmax)
            or not np.isfinite(field).any()
        ):
            ax.set_title(f"{title} unavailable")
        else:
            cf = ax.contourf(
                snap.box_lons, snap.box_lats, field,
                levels=21, vmin=vmin, vmax=vmax, cmap=cmap, transform=pc,
            )
            cf.set_clim(vmin, vmax)
            plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)

        loc = snap.loc
        if loc is not None and np.isfinite(loc[0]) and np.isfinite(loc[1]):
            _plot_marker(ax, loc, "x", "storm loc", color="k", markersize=10, transform=pc)
            _plot_radius_circle(
                ax, loc[0], loc[1], snap.rdr if snap.rdr is not None else float("nan"),
                color="k", linestyle="--", linewidth=1.2, label="rdr",
            )

    fig.suptitle(
        f"{history.cyclone_id}  valid "
        f"{snap.valid_time if snap.valid_time is not None else '?'}"
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_page(
    history: "RolloutSnapshots",
    kept: List[Snapshot],
    pdf,
    *,
    best_track=None,
) -> None:
    """Time-series summary (MSLP + VMAX top; vorticity + warm core bottom)."""
    import matplotlib.pyplot as plt

    t = [s.valid_time for s in kept]
    diags = [history.diagnostics(s) for s in kept]
    mslp_hpa = [
        (d["mslp_min_val"] / HPA_TO_PA if np.isfinite(d["mslp_min_val"]) else float("nan"))
        for d in diags
    ]
    wspd_kt = [
        (d["windspd10_max_val"] * _MPS_TO_KT if np.isfinite(d["windspd10_max_val"]) else float("nan"))
        for d in diags
    ]
    vort = [d["vorticity850_max_val"] for d in diags]
    wc_max = [d["warm_core_max"] for d in diags]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_top.plot(t, mslp_hpa, "-o", color="C0", label="model MSLP")
    ax_top.set_ylabel("MSLP (hPa)", color="C0")
    ax_top.tick_params(axis="y", labelcolor="C0")

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(t, wspd_kt, "-s", color="C3", label="model VMAX")
    ax_top_r.set_ylabel("10 m windspeed (kt)", color="C3")
    ax_top_r.tick_params(axis="y", labelcolor="C3")

    bt_vmax_vals: list = []
    if best_track is not None:
        bt_df = best_track.df_best_tracks_history
        if "datetimes" in bt_df.columns:
            bt_t = bt_df["datetimes"]
            if "MSLP" in bt_df.columns:
                bt_mslp = pd.to_numeric(bt_df["MSLP"], errors="coerce")
                ax_top.plot(bt_t, bt_mslp, "--", color="C0", label="BT MSLP")
            if "VMAX" in bt_df.columns:
                bt_vmax = pd.to_numeric(bt_df["VMAX"], errors="coerce")
                ax_top_r.plot(bt_t, bt_vmax, "--", color="C3", label="BT VMAX")
                bt_vmax_vals = bt_vmax.dropna().tolist()

    def _fit_ylim(ax, values, *, pad_frac=0.05, pad_min=1.0):
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return
        lo, hi = min(finite), max(finite)
        if lo == hi:
            pad = max(abs(lo) * pad_frac, pad_min)
        else:
            pad = max((hi - lo) * pad_frac, pad_min)
        ax.set_ylim(lo - pad, hi + pad)

    # Physical envelope for MSLP: corrupt rows go off-plot, legitimate ones stay readable.
    ax_top.set_ylim(850, 1100)
    _fit_ylim(ax_top_r, list(wspd_kt) + bt_vmax_vals)

    lines_l, labels_l = ax_top.get_legend_handles_labels()
    lines_r, labels_r = ax_top_r.get_legend_handles_labels()
    ax_top.legend(lines_l + lines_r, labels_l + labels_r, loc="best", fontsize=8)
    ax_top.set_title("Surface intensity")

    ax_bot.plot(t, vort, "-o", color="C2", label="vort850 max")
    ax_bot.set_ylabel("vorticity (s$^{-1}$)", color="C2")
    ax_bot.tick_params(axis="y", labelcolor="C2")

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.plot(t, wc_max, "-^", color="C1", label="warm_core max")
    ax_bot_r.set_ylabel("warm_core max (K)", color="C1")
    ax_bot_r.tick_params(axis="y", labelcolor="C1")
    _fit_ylim(ax_bot, vort, pad_min=1e-6)
    _fit_ylim(ax_bot_r, wc_max, pad_min=0.05)

    lines_l, labels_l = ax_bot.get_legend_handles_labels()
    lines_r, labels_r = ax_bot_r.get_legend_handles_labels()
    ax_bot.legend(lines_l + lines_r, labels_l + labels_r, loc="best", fontsize=8)
    ax_bot.set_title("Upper-level dynamics / thermodynamics")
    ax_bot.set_xlabel("valid time")

    # If best-track extends past rollout, cap x-axis so the plot doesn't sprawl.
    if t:
        end_of_rollout = max(x for x in t if x is not None)
        x_cap = end_of_rollout + _dt.timedelta(days=1)
        if best_track is not None:
            bt_df = best_track.df_best_tracks_history
            if "datetimes" in bt_df.columns and not bt_df["datetimes"].empty:
                bt_max = pd.to_datetime(bt_df["datetimes"]).max()
                if pd.notna(bt_max) and bt_max > x_cap:
                    ax_top.set_xlim(right=x_cap)

    fig.autofmt_xdate()
    fig.suptitle(
        f"Rollout summary: {history.cyclone_id}, start {history.start_time}"
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_pressure_wind_page(
    history: "RolloutSnapshots",
    kept: List[Snapshot],
    pdf,
    *,
    best_track=None,
) -> None:
    """MSLP vs 10 m wind scatter (model + optional best-track overlay)."""
    import matplotlib.pyplot as plt

    diags = [history.diagnostics(s) for s in kept]
    aur_wspd_kt = [
        (d["windspd10_max_val"] * _MPS_TO_KT if np.isfinite(d["windspd10_max_val"]) else float("nan"))
        for d in diags
    ]
    aur_mslp_hpa = [
        (d["mslp_min_val"] / HPA_TO_PA if np.isfinite(d["mslp_min_val"]) else float("nan"))
        for d in diags
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(aur_wspd_kt, aur_mslp_hpa, "o", color="C0", label="model", alpha=0.85)

    if best_track is not None:
        bt_df = best_track.df_best_tracks_history
        if "VMAX" in bt_df.columns and "MSLP" in bt_df.columns:
            bt_vmax = pd.to_numeric(bt_df["VMAX"], errors="coerce")
            bt_mslp = pd.to_numeric(bt_df["MSLP"], errors="coerce")
            ax.plot(bt_vmax, bt_mslp, "s", color="C3", label="best track", alpha=0.85)

    ax.set_ylim(850, 1100)
    ax.set_xlabel("10 m windspeed (kt)")
    ax.set_ylabel("MSLP (hPa)")
    ax.set_title("Pressure-wind relationship")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Pressure-wind: {history.cyclone_id}, start {history.start_time}"
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_track_page(
    history: "RolloutSnapshots",
    kept: List[Snapshot],
    pdf,
    *,
    best_track=None,
) -> None:
    """Storm-track map (model + optional best-track overlay)."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    pc = ccrs.PlateCarree()
    # Pick axes projection based on the lons that will actually plot (model
    # locs + best-track longitudes).
    all_lons_for_proj = [s.loc[1] for s in kept if s.loc is not None and np.isfinite(s.loc[1])]
    if best_track is not None:
        bt_df = best_track.df_best_tracks_history
        if "LON_DEG" in bt_df.columns:
            bt_lon_vals = pd.to_numeric(bt_df["LON_DEG"], errors="coerce").dropna().tolist()
            all_lons_for_proj.extend(bt_lon_vals)
    axes_proj = _plate_carree_for_lons(np.asarray(all_lons_for_proj)) if all_lons_for_proj else pc
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": axes_proj})

    aur_lats, aur_lons = [], []
    for snap in kept:
        loc = snap.loc
        if loc is not None and np.isfinite(loc[0]) and np.isfinite(loc[1]):
            aur_lats.append(float(loc[0]))
            aur_lons.append(float(loc[1]))

    bt_lats, bt_lons = [], []
    if best_track is not None:
        bt_df = best_track.df_best_tracks_history
        if {"LAT_DEG", "LON_DEG", "datetimes"}.issubset(bt_df.columns):
            bt_clean = (
                bt_df[["datetimes", "LAT_DEG", "LON_DEG"]]
                .drop_duplicates(subset="datetimes")
                .sort_values("datetimes")
            )
            lat_vals = pd.to_numeric(bt_clean["LAT_DEG"], errors="coerce")
            lon_vals = pd.to_numeric(bt_clean["LON_DEG"], errors="coerce")
            valid = lat_vals.notna() & lon_vals.notna()
            bt_lats = lat_vals[valid].astype(float).tolist()
            bt_lons = lon_vals[valid].astype(float).tolist()

    all_lats = aur_lats + bt_lats
    all_lons = aur_lons + bt_lons
    if all_lats and all_lons:
        pad = 5.0
        ax.set_extent(
            [min(all_lons) - pad, max(all_lons) + pad,
             min(all_lats) - pad, max(all_lats) + pad],
            crs=ccrs.PlateCarree(),
        )
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle=":", color="gray")
    gl.top_labels = False
    gl.right_labels = False

    if bt_lats:
        ax.plot(bt_lons, bt_lats, "--s", color="C3", markersize=4, linewidth=1.0,
                label="best track (full)", transform=pc, zorder=3)
        ax.plot(bt_lons[0], bt_lats[0], "^", color="C3", markersize=9, transform=pc, zorder=4)
        ax.plot(bt_lons[-1], bt_lats[-1], "v", color="C3", markersize=9, transform=pc, zorder=4)

    if aur_lats:
        ax.plot(aur_lons, aur_lats, "-o", color="C0", markersize=5, linewidth=1.5,
                label="model track", transform=pc, zorder=5)
        ax.plot(aur_lons[0], aur_lats[0], "*", color="C0", markersize=12, transform=pc, zorder=6)

    ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Storm track: {history.cyclone_id}  (model start {history.start_time})\n"
        f"BT: triangle-up start  triangle-down end     Model: star start"
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_mse_figure(history: "RolloutSnapshots"):
    """Three-panel MSE figure (caller owns close/save)."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    t_storm, v_storm = history.mse_column_timeseries("storm_mask")
    t_env, v_env = history.mse_column_timeseries("env_mask")
    axs[0].plot(t_storm, v_storm / 1e9, "-o", color="C0", markersize=4, label="storm mask")
    axs[0].plot(t_env, v_env / 1e9, "-s", color="C3", markersize=4, label="env mask")
    axs[0].set_ylabel("column MSE (GJ m$^{-2}$)")
    axs[0].legend(fontsize=9)
    axs[0].set_title("Column-integrated MSE")

    t_anom, v_anom = history.mse_anomaly_timeseries()
    axs[1].plot(t_anom, v_anom / 1e9, "-o", color="C2", markersize=4)
    axs[1].axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    axs[1].set_ylabel(r"$\Delta$MSE storm-env (GJ m$^{-2}$)")
    axs[1].set_title("Storm-minus-environment MSE anomaly")

    times_p, levels_hpa, profiles = history.mse_vertical_profile_at_center()
    ax3 = axs[2]
    if levels_hpa is not None and len(times_p) > 0:
        n = len(times_p)
        cmap = plt.cm.tab20b
        for i, prof in enumerate(profiles):
            ax3.plot(prof / 1e3, levels_hpa, color=cmap(i / max(n - 1, 1)), linewidth=1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n - 1, 1)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04).set_label("rollout step")
    else:
        ax3.text(0.5, 0.5, "MSE profile unavailable", transform=ax3.transAxes, ha="center")
    ax3.invert_yaxis()
    ax3.set_xlabel("MSE (kJ kg$^{-1}$)")
    ax3.set_ylabel("pressure (hPa)")
    ax3.set_title("MSE vertical profile at storm centre")

    for ax in axs[:2]:
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("valid time")

    fig.suptitle(
        f"MSE diagnostics: {history.cyclone_id}, start {history.start_time}"
    )
    fig.tight_layout()
    return fig


def _plot_mse_page(history: "RolloutSnapshots", pdf) -> None:
    import matplotlib.pyplot as plt
    fig = _build_mse_figure(history)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_snapshots(
    history: "RolloutSnapshots",
    filename: Union[str, Path],
    *,
    best_track=None,
) -> None:
    """Render *history* as a multi-page PDF.

    One 2x2 page per snapshot with ``storm_computed`` set, then four
    summary pages (time-series, pressure-wind, track, MSE). Colour limits
    on the per-snapshot pages are shared across the rollout so frames are
    directly comparable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    kept = [s for s in history.snapshots if getattr(s, "storm_computed", False)]
    if not kept:
        print(
            f"plot_rollout_snapshots: nothing to plot for {history.cyclone_id} "
            f"(no storm_computed snapshots); skipping {filename}."
        )
        return

    diags = [history.diagnostics(s) for s in kept]
    vort_lim = _symmetric_lim([d["vorticity850"] for d in diags])
    mslp_vmin_pa, mslp_vmax_pa = _linear_lims([d["mslp"] for d in diags])
    mslp_vmin_hpa = mslp_vmin_pa / HPA_TO_PA if np.isfinite(mslp_vmin_pa) else float("nan")
    mslp_vmax_hpa = mslp_vmax_pa / HPA_TO_PA if np.isfinite(mslp_vmax_pa) else float("nan")
    wspd_vmin, wspd_vmax = _linear_lims([d["windspd10"] for d in diags])
    wspd_vmin = max(wspd_vmin, 0.0) if np.isfinite(wspd_vmin) else wspd_vmin
    wc_lim = _symmetric_lim([d["warm_core"] for d in diags])

    with PdfPages(str(filename)) as pdf:
        for snap in kept:
            _plot_snapshot_page(
                history, snap, pdf,
                vort_lim=vort_lim,
                mslp_vmin_hpa=mslp_vmin_hpa, mslp_vmax_hpa=mslp_vmax_hpa,
                wspd_vmin=wspd_vmin, wspd_vmax=wspd_vmax,
                wc_lim=wc_lim,
            )
        _plot_summary_page(history, kept, pdf, best_track=best_track)
        _plot_pressure_wind_page(history, kept, pdf, best_track=best_track)
        _plot_track_page(history, kept, pdf, best_track=best_track)
        _plot_mse_page(history, pdf)


def plot_mse_diagnostics(
    history: "RolloutSnapshots",
    filename: Optional[Union[str, Path]] = None,
) -> None:
    """Three-panel MSE diagnostics figure for a single rollout.

    Called automatically by :func:`plot_rollout_snapshots` as the final
    page of the summary PDF. Call directly to produce a standalone figure.
    """
    import matplotlib.pyplot as plt
    fig = _build_mse_figure(history)
    if filename is not None:
        fig.savefig(str(filename), bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison plotting (two rollouts, same storm, overlaid)
# ---------------------------------------------------------------------------


def _run_series(hist: "RolloutSnapshots", kept: List[Snapshot]) -> dict:
    """Pull the common time-series arrays used by the comparison plots."""
    diags = [hist.diagnostics(s) for s in kept]
    return {
        "t": [s.valid_time for s in kept],
        "mslp_hpa": [
            (d["mslp_min_val"] / HPA_TO_PA if np.isfinite(d["mslp_min_val"]) else float("nan"))
            for d in diags
        ],
        "wspd_kt": [
            (d["windspd10_max_val"] * _MPS_TO_KT if np.isfinite(d["windspd10_max_val"]) else float("nan"))
            for d in diags
        ],
        "vort": [d["vorticity850_max_val"] for d in diags],
        "wc_max": [d["warm_core_max"] for d in diags],
        "lats": [s.loc[0] for s in kept if s.loc is not None and np.isfinite(s.loc[0])],
        "lons": [s.loc[1] for s in kept if s.loc is not None and np.isfinite(s.loc[1])],
    }


def _fit_ylim(ax, values, *, pad_frac=0.05, pad_min=1.0) -> None:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    pad = max(abs(lo) * pad_frac, pad_min) if lo == hi else max((hi - lo) * pad_frac, pad_min)
    ax.set_ylim(lo - pad, hi + pad)


# Style convention: control = solid + circle marker; bogus = dashed + triangle.
_STYLES = {
    0: {"linestyle": "-", "marker": "o"},
    1: {"linestyle": "--", "marker": "^"},
}


def _plot_summary_comparison_page(
    ctrl: "RolloutSnapshots", kept_ctrl: List[Snapshot],
    bogus: "RolloutSnapshots", kept_bogus: List[Snapshot],
    pdf,
    *,
    best_track=None,
    labels: tuple,
) -> None:
    """Time-series summary with ctrl + bogus overlaid on the same axes."""
    import matplotlib.pyplot as plt

    sc = _run_series(ctrl, kept_ctrl)
    sb = _run_series(bogus, kept_bogus)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: MSLP (left, C0) + VMAX (right-twin, C3)
    for label, s, style in ((labels[0], sc, _STYLES[0]), (labels[1], sb, _STYLES[1])):
        ax_top.plot(
            s["t"], s["mslp_hpa"], color="C0",
            label=f"{label} MSLP", **style,
        )
    ax_top.set_ylabel("MSLP (hPa)", color="C0")
    ax_top.tick_params(axis="y", labelcolor="C0")

    ax_top_r = ax_top.twinx()
    for label, s, style in ((labels[0], sc, _STYLES[0]), (labels[1], sb, _STYLES[1])):
        ax_top_r.plot(
            s["t"], s["wspd_kt"], color="C3",
            label=f"{label} VMAX", **style,
        )
    ax_top_r.set_ylabel("10 m windspeed (kt)", color="C3")
    ax_top_r.tick_params(axis="y", labelcolor="C3")

    bt_vmax_vals: list = []
    if best_track is not None:
        bt_df = best_track.df_best_tracks_history
        if "datetimes" in bt_df.columns:
            if "MSLP" in bt_df.columns:
                bt_mslp = pd.to_numeric(bt_df["MSLP"], errors="coerce")
                ax_top.plot(bt_df["datetimes"], bt_mslp, ":", color="C0", label="BT MSLP")
            if "VMAX" in bt_df.columns:
                bt_vmax = pd.to_numeric(bt_df["VMAX"], errors="coerce")
                ax_top_r.plot(bt_df["datetimes"], bt_vmax, ":", color="C3", label="BT VMAX")
                bt_vmax_vals = bt_vmax.dropna().tolist()

    ax_top.set_ylim(850, 1100)
    _fit_ylim(ax_top_r, list(sc["wspd_kt"]) + list(sb["wspd_kt"]) + bt_vmax_vals)
    lines_l, labels_l = ax_top.get_legend_handles_labels()
    lines_r, labels_r = ax_top_r.get_legend_handles_labels()
    ax_top.legend(lines_l + lines_r, labels_l + labels_r, loc="best", fontsize=8)
    ax_top.set_title("Surface intensity")

    # Bottom: vort850 (left, C2) + warm_core (right-twin, C1)
    for label, s, style in ((labels[0], sc, _STYLES[0]), (labels[1], sb, _STYLES[1])):
        ax_bot.plot(s["t"], s["vort"], color="C2", label=f"{label} vort850 max", **style)
    ax_bot.set_ylabel("vorticity (s$^{-1}$)", color="C2")
    ax_bot.tick_params(axis="y", labelcolor="C2")

    ax_bot_r = ax_bot.twinx()
    for label, s, style in ((labels[0], sc, _STYLES[0]), (labels[1], sb, _STYLES[1])):
        ax_bot_r.plot(s["t"], s["wc_max"], color="C1", label=f"{label} warm_core max", **style)
    ax_bot_r.set_ylabel("warm_core max (K)", color="C1")
    ax_bot_r.tick_params(axis="y", labelcolor="C1")
    _fit_ylim(ax_bot, list(sc["vort"]) + list(sb["vort"]), pad_min=1e-6)
    _fit_ylim(ax_bot_r, list(sc["wc_max"]) + list(sb["wc_max"]), pad_min=0.05)

    lines_l, labels_l = ax_bot.get_legend_handles_labels()
    lines_r, labels_r = ax_bot_r.get_legend_handles_labels()
    ax_bot.legend(lines_l + lines_r, labels_l + labels_r, loc="best", fontsize=8)
    ax_bot.set_title("Upper-level dynamics / thermodynamics")
    ax_bot.set_xlabel("valid time")

    # X-cap identical to single-run version.
    all_t = [x for x in sc["t"] + sb["t"] if x is not None]
    if all_t:
        end_of_rollout = max(all_t)
        x_cap = end_of_rollout + _dt.timedelta(days=1)
        if best_track is not None:
            bt_df = best_track.df_best_tracks_history
            if "datetimes" in bt_df.columns and not bt_df["datetimes"].empty:
                bt_max = pd.to_datetime(bt_df["datetimes"]).max()
                if pd.notna(bt_max) and bt_max > x_cap:
                    ax_top.set_xlim(right=x_cap)

    fig.autofmt_xdate()
    fig.suptitle(
        f"Rollout summary: {ctrl.cyclone_id}, start {ctrl.start_time} "
        f"({labels[0]} vs {labels[1]})"
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_pressure_wind_comparison_page(
    ctrl: "RolloutSnapshots", kept_ctrl: List[Snapshot],
    bogus: "RolloutSnapshots", kept_bogus: List[Snapshot],
    pdf,
    *,
    best_track=None,
    labels: tuple,
) -> None:
    """MSLP vs VMAX scatter with ctrl + bogus overlaid."""
    import matplotlib.pyplot as plt

    sc = _run_series(ctrl, kept_ctrl)
    sb = _run_series(bogus, kept_bogus)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(sc["wspd_kt"], sc["mslp_hpa"], marker="o", linestyle="None",
            color="C0", label=labels[0], alpha=0.85)
    ax.plot(sb["wspd_kt"], sb["mslp_hpa"], marker="^", linestyle="None",
            color="C1", label=labels[1], alpha=0.85)
    if best_track is not None:
        bt_df = best_track.df_best_tracks_history
        if "VMAX" in bt_df.columns and "MSLP" in bt_df.columns:
            bt_vmax = pd.to_numeric(bt_df["VMAX"], errors="coerce")
            bt_mslp = pd.to_numeric(bt_df["MSLP"], errors="coerce")
            ax.plot(bt_vmax, bt_mslp, "s", color="C3", label="best track", alpha=0.75)
    ax.set_ylim(850, 1100)
    ax.set_xlabel("10 m windspeed (kt)")
    ax.set_ylabel("MSLP (hPa)")
    ax.set_title("Pressure-wind relationship")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Pressure-wind: {ctrl.cyclone_id}, start {ctrl.start_time} "
        f"({labels[0]} vs {labels[1]})"
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_track_comparison_page(
    ctrl: "RolloutSnapshots", kept_ctrl: List[Snapshot],
    bogus: "RolloutSnapshots", kept_bogus: List[Snapshot],
    pdf,
    *,
    best_track=None,
    labels: tuple,
) -> None:
    """Storm-track map with ctrl + bogus overlaid."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    pc = ccrs.PlateCarree()
    sc = _run_series(ctrl, kept_ctrl)
    sb = _run_series(bogus, kept_bogus)

    bt_lats: list = []
    bt_lons: list = []
    if best_track is not None:
        bt_df = best_track.df_best_tracks_history
        if {"LAT_DEG", "LON_DEG", "datetimes"}.issubset(bt_df.columns):
            bt_clean = (
                bt_df[["datetimes", "LAT_DEG", "LON_DEG"]]
                .drop_duplicates(subset="datetimes")
                .sort_values("datetimes")
            )
            lat_vals = pd.to_numeric(bt_clean["LAT_DEG"], errors="coerce")
            lon_vals = pd.to_numeric(bt_clean["LON_DEG"], errors="coerce")
            valid = lat_vals.notna() & lon_vals.notna()
            bt_lats = lat_vals[valid].astype(float).tolist()
            bt_lons = lon_vals[valid].astype(float).tolist()

    all_lons = sc["lons"] + sb["lons"] + bt_lons
    axes_proj = _plate_carree_for_lons(np.asarray(all_lons)) if all_lons else pc
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": axes_proj})

    all_lats = sc["lats"] + sb["lats"] + bt_lats
    if all_lats and all_lons:
        pad = 5.0
        ax.set_extent(
            [min(all_lons) - pad, max(all_lons) + pad,
             min(all_lats) - pad, max(all_lats) + pad],
            crs=ccrs.PlateCarree(),
        )
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle=":", color="gray")
    gl.top_labels = False
    gl.right_labels = False

    if bt_lats:
        ax.plot(bt_lons, bt_lats, ":s", color="C3", markersize=4, linewidth=1.0,
                label="best track (full)", transform=pc, zorder=3)
        ax.plot(bt_lons[0], bt_lats[0], "^", color="C3", markersize=9, transform=pc, zorder=4)
        ax.plot(bt_lons[-1], bt_lats[-1], "v", color="C3", markersize=9, transform=pc, zorder=4)
    if sc["lons"]:
        ax.plot(sc["lons"], sc["lats"], "-o", color="C0", markersize=5, linewidth=1.5,
                label=f"{labels[0]} track", transform=pc, zorder=5)
        ax.plot(sc["lons"][0], sc["lats"][0], "*", color="C0", markersize=12, transform=pc, zorder=6)
    if sb["lons"]:
        ax.plot(sb["lons"], sb["lats"], "--^", color="C1", markersize=5, linewidth=1.5,
                label=f"{labels[1]} track", transform=pc, zorder=5)
        ax.plot(sb["lons"][0], sb["lats"][0], "*", color="C1", markersize=12, transform=pc, zorder=6)
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Storm track: {ctrl.cyclone_id}  "
        f"(start {ctrl.start_time})  --  {labels[0]} vs {labels[1]}\n"
        f"BT: triangle-up start  triangle-down end     Model: star start"
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_mse_comparison_figure(
    ctrl: "RolloutSnapshots",
    bogus: "RolloutSnapshots",
    *,
    labels: tuple,
):
    """2x2 MSE panel with ctrl + bogus overlaid. Vertical profile split per run."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0): column MSE storm vs env, both runs overlaid.
    for i, (label, hist) in enumerate(((labels[0], ctrl), (labels[1], bogus))):
        t_s, v_s = hist.mse_column_timeseries("storm_mask")
        t_e, v_e = hist.mse_column_timeseries("env_mask")
        axs[0, 0].plot(t_s, v_s / 1e9, color="C0", label=f"{label} storm", **_STYLES[i])
        axs[0, 0].plot(t_e, v_e / 1e9, color="C3", label=f"{label} env", **_STYLES[i])
    axs[0, 0].set_ylabel("column MSE (GJ m$^{-2}$)")
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].set_title("Column-integrated MSE")
    axs[0, 0].tick_params(axis="x", rotation=30)
    axs[0, 0].set_xlabel("valid time")

    # (0,1): storm - env anomaly, both runs overlaid.
    for i, (label, hist) in enumerate(((labels[0], ctrl), (labels[1], bogus))):
        t_a, v_a = hist.mse_anomaly_timeseries()
        axs[0, 1].plot(
            t_a, v_a / 1e9,
            color=("C2" if i == 0 else "C4"),
            label=label, **_STYLES[i],
        )
    axs[0, 1].axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    axs[0, 1].set_ylabel(r"$\Delta$MSE storm-env (GJ m$^{-2}$)")
    axs[0, 1].set_title("Storm-minus-environment MSE anomaly")
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].tick_params(axis="x", rotation=30)
    axs[0, 1].set_xlabel("valid time")

    # (1,0) and (1,1): vertical MSE profiles at storm centre, one subplot per run.
    for ax, (label, hist) in zip(axs[1], ((labels[0], ctrl), (labels[1], bogus))):
        times_p, levels_hpa, profiles = hist.mse_vertical_profile_at_center()
        if levels_hpa is not None and len(times_p) > 0:
            n = len(times_p)
            cmap = plt.cm.tab20b
            for i, prof in enumerate(profiles):
                ax.plot(prof / 1e3, levels_hpa, color=cmap(i / max(n - 1, 1)), linewidth=1.0)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(n - 1, 1)))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label("rollout step")
        else:
            ax.text(0.5, 0.5, "MSE profile unavailable", transform=ax.transAxes, ha="center")
        ax.invert_yaxis()
        ax.set_xlabel("MSE (kJ kg$^{-1}$)")
        ax.set_ylabel("pressure (hPa)")
        ax.set_title(f"MSE vertical profile at storm centre -- {label}")

    fig.suptitle(
        f"MSE diagnostics: {ctrl.cyclone_id}, start {ctrl.start_time} "
        f"({labels[0]} vs {labels[1]})"
    )
    fig.tight_layout()
    return fig


def _plot_mse_comparison_page(
    ctrl: "RolloutSnapshots",
    bogus: "RolloutSnapshots",
    pdf,
    *,
    labels: tuple,
) -> None:
    import matplotlib.pyplot as plt
    fig = _build_mse_comparison_figure(ctrl, bogus, labels=labels)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_comparison(
    ctrl: "RolloutSnapshots",
    bogus: "RolloutSnapshots",
    filename: Union[str, Path],
    *,
    best_track=None,
    labels: tuple = ("control", "bogus"),
) -> None:
    """Four-page PDF comparing two rollouts of the same storm.

    Mirrors the final four pages of :func:`plot_rollout_snapshots`
    (summary line-plots, pressure-wind scatter, storm track, MSE
    diagnostics) with the two rollouts overlaid on the same axes.
    Variable identity is carried by colour (MSLP = C0, VMAX = C3,
    vorticity = C2, warm core = C1); run identity by linestyle/marker
    (``control`` -> solid+circle, ``bogus`` -> dashed+triangle).

    Args:
        ctrl: Control rollout.
        bogus: Bogus (intervened) rollout. Must share ``cyclone_id`` and
            ``start_time`` with ``ctrl`` for the labelling to make sense;
            no runtime assertion is enforced.
        filename: Output ``.pdf`` path.
        best_track: Optional best-track object for the same storm.
        labels: Legend labels for the two rollouts, in ``(ctrl, bogus)``
            order.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    kept_ctrl = [s for s in ctrl.snapshots if getattr(s, "storm_computed", False)]
    kept_bogus = [s for s in bogus.snapshots if getattr(s, "storm_computed", False)]
    if not kept_ctrl or not kept_bogus:
        print(
            f"plot_rollout_comparison: nothing to plot "
            f"(ctrl kept={len(kept_ctrl)}, bogus kept={len(kept_bogus)}); "
            f"skipping {filename}."
        )
        return

    with PdfPages(str(filename)) as pdf:
        _plot_summary_comparison_page(
            ctrl, kept_ctrl, bogus, kept_bogus, pdf,
            best_track=best_track, labels=labels,
        )
        _plot_pressure_wind_comparison_page(
            ctrl, kept_ctrl, bogus, kept_bogus, pdf,
            best_track=best_track, labels=labels,
        )
        _plot_track_comparison_page(
            ctrl, kept_ctrl, bogus, kept_bogus, pdf,
            best_track=best_track, labels=labels,
        )
        _plot_mse_comparison_page(ctrl, bogus, pdf, labels=labels)
