# tcinit handoff — 2026-10-01

## Session summary

Two big arcs today:

1. **K&C audit on real data.** Walked the Kwon & Cheong (2009) paper
   equations against the implementation in `tcinit/ideal_tc_vortex.py`
   (renamed from `tcinit/vortex.py` early in the session). Found and
   fixed two real bugs (eq. 6 V_gmax operation order; SH-f sign in the
   gradient-wind quadratic). Verified §2d-e algebraically and
   dimensionally; no further bugs. The +28 K warm core we saw yesterday
   on Mawar is a legitimate K&C output for a 97-hPa dp storm; prior
   framing of it as "obviously wrong" was based on a mis-remembered
   observational floor (Frank 1977 actually gives 15 K+, and intense
   TCs like Haiyan can go higher).

2. **RolloutSnapshots port.** Built the model-agnostic analogue of
   CausalTC's `SingleAuroraDataHistory`. New `tcinit/rollout_snapshots.py`
   carries a list of `tcinit.Snapshot`s, provides `to_dataframe` /
   `to_comparison_dataframe` / MSE time-series methods, and the full
   plotting surface. Added `plot_rollout_comparison` for overlaying two
   rollouts on the same axes.

## Commits (3 ahead of origin/main; needs `git push` from outside the container)

- `a93d07d` — Add RolloutSnapshots, MSE diagnostics, and comparison plotting
- `efaab43` — Fix K&C V_gmax + SH gradient wind; add plot_snapshot and Snapshot persistence
- `977d413` — Rename `tcinit.vortex` module to `tcinit.ideal_tc_vortex`

## Code changes in detail

### `tcinit/ideal_tc_vortex.py` (renamed from `tcinit/vortex.py`)

Two bug fixes, two comments:

- **`:307` — K&C eq. 6 V_gmax.** Was `V_m / (K_0 * cos β_0)`; now
  `(V_m / K_0) * cos β_0`. Confirmed by the paper's companion
  V_g30 formula on `:316` which already used the correct order,
  so the two were internally inconsistent. V_gmax drops ~13% at
  β_0 = 20°; B drops ~28% (∝ V_g²).
- **`:120` and `:462` — SH gradient wind.** K&C eqs. 7 / 16 are
  written with signed f; taking the cyclonic-wind magnitude requires
  `|f|`. Both sites now use `abs(f_coriolis)` so the Newton-Raphson
  solve for A and the gradient-wind computation give the correct
  magnitude in both hemispheres.
- **`:306` — TODO** on ρ: dry R_D should strictly be moist R_m =
  R_d·(1 + 0.608·q). Measured empirically: ~1% change in Φ'_b, cancels
  to <0.1 K in the T anomaly derivation. Not urgent.
- **`:505` — comment** on eq. 24's bounds typo. The paper's eq. 24
  prints the first q_0 integral with bounds [0.98, 1], but eq. 23
  unambiguously puts it on [σ_a, 0.98]. Code already followed eq. 23;
  comment makes it explicit.

### `tcinit/snapshot.py`

- Added optional `valid_time` on `__init__`, auto-detected from
  `ds.time` in `from_xarray`, persisted through `to_netcdf` /
  `from_netcdf` as a scalar coord.
- Added `to_netcdf(path, *, source_ds=None)` and `from_netcdf(path)`.
  `source_ds=None` writes just the Snapshot state (~1 MB for Mawar's
  427×427 box). With `source_ds` supplied, atmospheric fields are
  bundled under `src_<canon>` so `plot_snapshot` and the diagnostics
  helper can run without reopening the original Dataset (~30 MB for
  the same box with 25 levels).

### `tcinit/plotting.py`

- Added `plot_snapshot(snapshot, ds=None, …)` — 3×3 map view
  (vorticity, MSLP, 10 m wind, T anomalies at 300/500/700,
  q at 700, mean(T200,T500) anomaly, warm-core sum). Takes fields
  from `snapshot.src_*` (bundled) or a live Dataset; graceful "unavailable"
  for missing panels.
- Added `_vorticity_from_uv(u, v, lats, lons)` — spherical-earth
  ζ = ∂v/∂x − ∂u/∂y with cos(lat) metric.
- Fixed `set_extent(..., crs=Geodetic)` → `crs=PlateCarree()` to avoid
  cartopy's global-wrap behaviour at the antimeridian.

### `tcinit/diagnostics/mse.py` (new)

Ported `causaltc.diagnostics.mse` verbatim with imports rewritten
against `tcinit.constants` (CP_DRY, LV, G_0). Functions: `mse_3d`,
`column_integrate_mse`, `mse_breakdown`.

### `tcinit/rollout_snapshots.py` (new, 1725 lines)

- `WarmCoreCriterion` enum: `ANOMALY_SUM` (sum of 300/500/700 anoms,
  Dulac 2023) or `T200_500_AVG` (anomaly of 0.5·(T200+T500)).
- `snapshot_diagnostics(snap, criterion)` — derives all the scalar
  and 2D-field diagnostics the plot code / dataframe code needs
  (mslp_min, windspd10_max, vorticity850 + max, T anomalies at
  300/500/700, warm_core per criterion, mse_3d, mse_column).
  Hemisphere-signed vorticity extremum: picks `argmax(sign·vort)`
  so SH storms report negative values and NH positive.
- `RolloutSnapshots`:
  - `__init__(cyclone_id, start_time, snapshots=…, intervened=…, warm_core_criterion=…)`
  - `save(root)` / `load(…)` — pickle (CausalTC-compatible layout
    `root/<cyclone_id>/<ts>_steps<N>[_intervened].pkl`).
  - `to_netcdf(root)` / `from_netcdf(dir)` — opt-in NetCDF bundle;
    one `.nc` per snapshot plus `index.json`.
  - `to_dataframe(…)` and `to_comparison_dataframe(best_track, …)`
    with the same column set as the CausalTC original (model_* /
    bt_* prefixes, position_error_km, dVMAX, dMSLP). Printed stats
    block from `_print_comparison`.
  - `mse_column_timeseries(mask_attr)`, `mse_anomaly_timeseries()`,
    `mse_vertical_profile_at_center()`.
- Plotting:
  - `plot_rollout_snapshots(history, filename, best_track=…)` —
    multipage PDF. One 2×2 per snapshot (vorticity, MSLP, 10 m wind,
    warm core), then summary + pressure-wind + track + MSE.
  - `plot_mse_diagnostics(history, filename=…)` — standalone 3-panel
    MSE figure.
  - `plot_rollout_comparison(ctrl, bogus, filename, best_track=…,
    labels=(…))` — 4-page PDF overlaying two rollouts. Variable
    identity by colour (MSLP=C0, VMAX=C3, vort=C2, warm_core=C1);
    run identity by linestyle+marker (ctrl=solid+circle,
    bogus=dashed+triangle). MSE page is 2×2: column-MSE time series,
    anomaly time series, side-by-side vertical profiles.
  - `_plate_carree_for_lons(lons)` — returns `PlateCarree(central_longitude=180)`
    when the box crosses the antimeridian (max lon > 180 or span > 180°),
    otherwise `PlateCarree()`. Prevents the global-strip rendering
    we hit on the Yasa test files.
  - All `set_extent` calls use `crs=PlateCarree()`.
  - `_run_series(hist, kept)` helper collates the common per-step
    arrays the comparison plots consume.

## Verified end-to-end on real data

- **Mawar 2023 (bwp022023) control + bogus Aurora rollouts** (13 steps
  each, 6 h cadence). The pickles were produced on your HPC with
  `SingleAuroraDataHistory.save(Path("aurora_rollout_control"))` after
  running Aurora with an IFS initial condition and separately with a
  K&C bogus. On this machine the pickles unpickle via stub shims for
  `aurora` and `torch` (see `/tmp/test_comparison.py`).
  - Control: starts at 957 / 70 kt (IFS surface analysis of Mawar);
    stays 955-970 hPa / 55-65 kt for the whole 72 h.
  - Bogus: starts at 897 / 174 kt (matches best-track to the Pa). By
    t+6 h: 932 / 71 kt. By t+12 h: 963 / 58 kt (indistinguishable
    from control). Aurora digests the K&C injection within ~12 h.
  - The MSE comparison figure shows this cleanly: the bogus's step-0
    vertical MSE profile has a +30 kJ/kg bulge at 300-400 hPa (from
    the +24 K warm core driving c_p·T up); by step 2+ the bogus
    profile family collapses onto the control family.
- **Yasa 2020 (SP02 2021)** — 25 pre-existing AuroraSnapshot-schema
  NetCDFs in `data/yasa_*.nc`. Loaded via an ad-hoc converter in
  `/tmp/test_yasa_rollout.py`, exercised the full plot pipeline.
  Found and fixed the dateline-crossing projection bug while on this
  one (box is 162-183°E, straddles 180). Vorticity correctly negative
  throughout (SH cyclone). MSE panels say "unavailable" since these
  files don't carry q/z stacks.

## Open items from today (physics, not code)

- **K&C warm core for intense storms.** Mawar / Haiyan analytical T'
  at r=0 climbs to +28-35 K at 300 hPa (dp = 97-113 hPa). We decided
  this is within K&C's design (Φ'_b scales linearly with dp, no cap)
  and may actually be within observational scatter per Frank 1977 and
  Hawkins-Imbembo 1976 for storms this intense. Not a bug to fix,
  but **worth confirming** against Haiyan microwave AMSU-A warm-core
  retrievals (CIMSS archive) before publishing.
- **Rollout rdr noise.** Mawar control vs bogus `rdr` ratios range
  0.65-2.55 across lead times, from the per-snapshot wind/gradient
  detection sitting on local extrema in different corners. Mask-area
  variation injects noise into mask-restricted diagnostics
  (warm_core_max, mse_anomaly). Flagged as a follow-up: force both
  rollouts onto a common `fixed_rdr_km` for comparison studies.
  `tcinit.Snapshot.__init__` already accepts the kwarg.
- **MSE page cosmetic.** When all MSE values are NaN (yasa case), the
  time-series panels default to a Jan 2000 x-axis instead of showing
  "unavailable". Not critical.

## Next session: other models + rollout tests

Session scope the user named: run other models (SFNO, WeatherNext,
NeuralGCM, HICCUP) through the same rollout + comparison pipeline that
Aurora went through today. All the backend extract/apply/write_back
plumbing is already in place from 2026-09-30; the task is to generate
rollouts on HPC and feed them through `RolloutSnapshots`.

Suggested order of attack:
1. **SFNO** — closest to Aurora semantically (torch tensor native
   container). The `sfno_backend` already has extract/apply/write_back.
   Need a notebook on HPC that mirrors the Mawar case-study pattern
   but calls earth2studio's SFNO rollout instead of Aurora.
2. **WeatherNext-2** — xarray-native, probably easiest.
3. **NeuralGCM** — surface-free; already uses `set_storm_center` path.
4. **HICCUP** — file-based; produces NetCDF outputs, so no in-memory
   rollout chaining — more of a one-shot comparison.

Pattern for each: generate control + bogus rollouts on HPC, pickle via
`SingleAuroraDataHistory.save(Path(...))` or equivalent, scp to local,
convert to `RolloutSnapshots` with the `aurora_to_tcinit` helper
(adaptations will be needed per backend), then run
`plot_rollout_comparison`.

## Infrastructure state

- **Container deps installed today**: no new system packages. `cartopy`,
  `matplotlib`, `pandas`, `netCDF4`, `scipy`, `xarray`, `numpy` all
  already in place. Still no `torch`, `aurora`, `earth2studio`,
  `neuralgcm`, `h5py`.
- **Pickle load path for CausalTC-produced rollouts**: stub `aurora`
  and `torch` modules before `pickle.load`; add
  `/workspace/external/causalcyclogenesis-pb-physics-bogus` to
  `sys.path`. See `/tmp/test_comparison.py` for the pattern (~15
  lines of stubs). We should probably bake this into a small helper
  module in tcinit so the user doesn't have to re-type the stubs
  every session — flag for next session.
- **Pickle-save gotcha**: `SingleAuroraDataHistory.save` takes a
  `Path`, not a string (uses `root / cyclone_id / fname`). User hit
  this on HPC today; `dh.save(Path("aurora_rollout_control"))` works.

## Where to start next session

1. Read this file.
2. Decide which model to port next (SFNO is the natural first step).
3. On HPC: generate a control + bogus rollout for Mawar with that
   backend, `dh.save(Path("sfno_rollout_control"))` / `..._bogus`.
4. scp both to `/workspace/data/`.
5. Write a minimal converter from the backend's native rollout output
   to `List[tcinit.Snapshot]`, following the Aurora template
   (`/tmp/test_comparison.py` lines 32-56).
6. Build two `RolloutSnapshots` and call
   `plot_rollout_comparison(ctrl, bogus, "comparison_sfno_mawar.pdf",
   best_track=bt, labels=("control", "bogus"))`.
7. Share the resulting PDF for comparison with Aurora's.
