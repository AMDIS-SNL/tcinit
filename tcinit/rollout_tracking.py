"""Follow a storm through a model rollout; build the bogus for its initial state.

Model-agnostic glue for the per-backend control-vs-bogus demo notebooks.
Each notebook supplies an ``extract_box(state, center)`` callable that cuts a
box with ``level``/``latitude``/``longitude`` coordinates out of its native
model state -- usually a thin wrapper around that backend's
``extract_snapshot_dataset``. Variables may carry canonical names or the
backend's own (e.g. WeatherNext's ``mean_sea_level_pressure``), given a
``var_map`` (canonical -> native, as in :mod:`tcinit.naming`), and the dims
may come in any order. This module then:

- turns each box into a :class:`Snapshot` with the storm detected and the
  atmospheric fields attached as ``src_<name>`` (needed by the warm-core and
  MSE diagnostics in :mod:`tcinit.rollout_snapshots`);
- follows the storm from step to step, centring each box on the previous
  step's detected position, and collects the snapshots into a
  :class:`RolloutSnapshots`;
- builds and applies the K&C vortex on the initial-state box
  (:func:`build_bogus`), leaving the write-back into the native container
  to the notebook.

No model, torch, or JAX imports: the rollout loop itself stays in the
notebook, which passes its states in as any iterable (a generator is best,
so global fields for every step never need to be held in memory at once).
"""

from __future__ import annotations

import datetime as _dt
import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from tcinit.backends import xarray_backend
from tcinit.constants import HPA_TO_PA
from tcinit.ideal_tc_vortex import BogusVortex
from tcinit.naming import ATMOS_VARS, resolve
from tcinit.rollout_snapshots import RolloutSnapshots
from tcinit.snapshot import Snapshot

ExtractBox = Callable[[Any, Tuple[float, float]], xr.Dataset]
VarMap = Optional[Mapping[str, str]]

_BOX_DIMS = ("level", "latitude", "longitude")


def _box_order(box: xr.Dataset) -> xr.Dataset:
    """Put the box's dims in (level, latitude, longitude) order.

    Snapshot reads raw arrays and indexes 2-D fields as ``[lat, lon]``;
    NeuralGCM boxes arrive as ``(level, longitude, latitude)``.
    """
    present = [d for d in _BOX_DIMS if d in box.dims]
    return box.transpose(*present, ...)


def _snapshot(box, *, var_map, valid_time, storm_radius_multiplier, fixed_rdr_km):
    """Snapshot with atmospheric fields attached; no storm detection yet."""
    snap = Snapshot.from_xarray(
        box,
        var_map=var_map,
        storm_radius_multiplier=storm_radius_multiplier,
        fixed_rdr_km=fixed_rdr_km,
        valid_time=valid_time,
    )
    for canon in ATMOS_VARS:
        native = resolve(canon, var_map)
        if native is not None and native in box.data_vars:
            setattr(snap, f"src_{canon}", np.asarray(box[native].values, dtype=float))
    return snap


def snapshot_from_box(
    box: xr.Dataset,
    *,
    valid_time: _dt.datetime,
    search_center: Tuple[float, float],
    var_map: VarMap = None,
    max_loc_error_km: float = 450.0,
    storm_radius_multiplier: float = 3.0,
    fixed_rdr_km: Optional[float] = None,
) -> Snapshot:
    """Build a Snapshot from a box and detect the storm in it.

    The box's atmospheric fields (``t, u, v, q, z``) are attached as
    ``snap.src_<name>`` so :func:`tcinit.rollout_snapshots.snapshot_diagnostics`
    can compute vorticity, warm-core, and MSE diagnostics without the source
    Dataset.

    Args:
        var_map: canonical -> native variable names in ``box`` (e.g.
            :data:`tcinit.naming.WEATHERNEXT_VAR_MAP`). ``None`` means the box
            already uses canonical names. Canonical names missing from the
            map are treated as absent.

    Raises:
        ValueError: from :meth:`Snapshot.detect_storm` when no storm is found
            inside ``max_loc_error_km`` of ``search_center``.
    """
    snap = _snapshot(
        _box_order(box),
        var_map=var_map,
        valid_time=valid_time,
        storm_radius_multiplier=storm_radius_multiplier,
        fixed_rdr_km=fixed_rdr_km,
    )
    snap.detect_storm(box_center=search_center, max_loc_error_km=max_loc_error_km)
    return snap


def build_bogus(
    box: xr.Dataset,
    best_track,
    target_time: _dt.datetime,
    *,
    search_center: Tuple[float, float],
    var_map: VarMap = None,
    apply: Optional[Callable[[xr.Dataset, BogusVortex], xr.Dataset]] = None,
    max_loc_error_km: float = 450.0,
    storm_radius_multiplier: float = 3.0,
    fixed_rdr_km: Optional[float] = None,
) -> Tuple[Snapshot, BogusVortex, xr.Dataset]:
    """Detect the storm in ``box``, build the K&C vortex, and apply it.

    Args:
        box: initial-state box (see :func:`snapshot_from_box` for names and
            dim order).
        best_track: :class:`tcinit.BestTrack` for the storm; supplies the
            target intensity at ``target_time``.
        target_time: initial-condition valid time.
        search_center: ``(lat, lon)`` around which to search for the storm,
            usually the best-track position at ``target_time``.
        var_map: canonical -> native variable names in ``box``; used for
            storm detection, and passed to the default ``apply``.
        apply: the backend's ``apply(ds, bv)``, which already knows its own
            variable names. Defaults to the generic
            :func:`tcinit.backends.xarray_backend.apply` with ``var_map``.

    Returns:
        ``(snapshot, vortex, modified_box)``. ``modified_box`` keeps the
        input box's variable names and dim order; write it back into the
        native container with the backend's ``write_back``.
    """
    if apply is None:
        apply = functools.partial(xarray_backend.apply, var_map=var_map)
    snap = snapshot_from_box(
        box,
        valid_time=target_time,
        search_center=search_center,
        var_map=var_map,
        max_loc_error_km=max_loc_error_km,
        storm_radius_multiplier=storm_radius_multiplier,
        fixed_rdr_km=fixed_rdr_km,
    )
    bv = BogusVortex(snap, best_track, target_time).build()
    return snap, bv, apply(box, bv)


def track_rollout(
    states: Iterable[Any],
    extract_box: ExtractBox,
    *,
    cyclone_id: str,
    start_time: _dt.datetime,
    initial_center: Tuple[float, float],
    step: _dt.timedelta = _dt.timedelta(hours=6),
    intervened: bool = False,
    var_map: VarMap = None,
    max_loc_error_km: float = 450.0,
    storm_radius_multiplier: float = 3.0,
    fixed_rdr_km: Optional[float] = None,
    verbose: bool = True,
) -> RolloutSnapshots:
    """Follow the storm through a sequence of model states.

    Args:
        states: model states in time order; ``states[0]`` is the initial
            condition at ``start_time`` and each later one is ``step`` further
            on. Consumed lazily, one state at a time.
        extract_box: ``extract_box(state, (lat, lon))`` returns the box
            centred on ``(lat, lon)``.
        cyclone_id: storm identifier for the RolloutSnapshots.
        start_time: valid time of ``states[0]``.
        initial_center: where to look for the storm in ``states[0]``.
            Later steps centre on the previous step's detected position.
        intervened: ``True`` for the bogus run.
        var_map: canonical -> native variable names in the boxes (see
            :func:`snapshot_from_box`).

    A step where detection fails is kept with ``storm_computed=False``
    (plots and dataframes skip it) and the next step searches around the
    last successful position.
    """
    rollout = RolloutSnapshots(cyclone_id, start_time, intervened=intervened)
    center = (float(initial_center[0]), float(initial_center[1]))
    label = "bogus" if intervened else "control"

    for k, state in enumerate(states):
        valid_time = start_time + k * step
        box = extract_box(state, center)
        try:
            snap = snapshot_from_box(
                box,
                valid_time=valid_time,
                search_center=center,
                var_map=var_map,
                max_loc_error_km=max_loc_error_km,
                storm_radius_multiplier=storm_radius_multiplier,
                fixed_rdr_km=fixed_rdr_km,
            )
        except ValueError as err:
            snap = _snapshot(
                _box_order(box),
                var_map=var_map,
                valid_time=valid_time,
                storm_radius_multiplier=storm_radius_multiplier,
                fixed_rdr_km=fixed_rdr_km,
            )
            if verbose:
                print(
                    f"[{label}] step {k:2d} {valid_time:%Y-%m-%d %HZ}: no storm ({err})"
                )
        else:
            center = (float(snap.loc[0]), float(snap.loc[1]))
            if verbose:
                print(
                    f"[{label}] step {k:2d} {valid_time:%Y-%m-%d %HZ}: "
                    f"MSLP {snap.mslp_min / HPA_TO_PA:7.1f} hPa at "
                    f"({center[0]:6.2f}, {center[1]:7.2f})  rdr {snap.rdr:5.0f} km"
                )
        rollout.append(snap)

    return rollout
