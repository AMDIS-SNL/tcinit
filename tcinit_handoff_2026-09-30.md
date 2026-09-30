# tcinit handoff — 2026-09-30

## Session summary

Built out five model-specific backends around the model-agnostic
`Snapshot` + `BogusVortex` core, extended two existing backends (NeuralGCM,
HICCUP) with symmetric extract/write_back helpers, and shipped a
Typhoon Mawar (WP02 2023) end-to-end case-study notebook that exercises
the pipeline on real HRES analysis data.

## Commits (3 ahead of origin/main; needs `git push` from outside the container)

- `c11c330` — Add Typhoon Mawar (WP02 2023) end-to-end case-study notebook
- `f4a134e` — Add WP02 2023 best-track for first case study; ignore `data/*.grb`
- `5b799ae` — Add SFNO / WeatherNext / HICCUP backends; extract+write_back for NeuralGCM

## Backend coverage now complete

Every backend implements the three-function contract
(`extract_snapshot_dataset` / `apply` / `write_back`) except pure-xarray
passthroughs, which only need `apply`:

| Backend | Native container | Status |
|---|---|---|
| `aurora_backend` | `aurora.Batch` (torch tensors) | Existed before; unchanged |
| `sfno_backend` | earth2studio `(torch.Tensor, CoordSystem)` | **New** |
| `weathernext_graph_backend` | xarray `(batch, time, [level,] lat, lon)` | **New** |
| `weathernext2_backend` | WN-Graph shape + 100 m wind | **New** |
| `neuralgcm_backend` | `PressureLevelModel.inputs_from_xarray()` dict | **Extended** with extract/write_back |
| `hiccup_backend` | ERA5 netCDF files (atm + sfc) | **Extended** with file-based extract/write_back |
| `xarray_backend` | canonical `xr.Dataset` | Existed; unchanged |

Naming cleanup in `tcinit/naming.py`: `LONGFORM_ATMOS_VAR_MAP` /
`LONGFORM_SURFACE_VAR_MAP` shared between NeuralGCM and WeatherNext;
`SFNO_LEVELS_HPA`, `SFNO_SURFACE_TO_CANON`, `sfno_channel_name`,
`sfno_channel_index`, `sfno_variables_73`, `ERA5_NETCDF_NAMES` added.

77 tests pass, 2 torch-gated smoke tests skip in this container.

## Case-study state

`notebooks/case_study_wp02_2023.ipynb` — 28 cells, 606 KB, PNG plots
embedded as cell outputs so it renders on GitHub without re-execution.

Verified numeric results on real HRES data at 2023-05-26 12Z:

- Detected storm at (15.71 N, 136.55 E) — 0.05° from b-deck.
- `rdr` = 39 km, matching Mawar's compact eye.
- Post-bogus MSLP minimum = 914 hPa, matching best-track.
- HICCUP + WN-2 round-trip both verified to 914 hPa at storm centre.
- SFNO and Aurora cells skip cleanly (no torch / aurora in container).

Cached artefacts under `data/case_study_wp02_2023/` (gitignored):
- `atmos_box_20230526_12Z.nc`, `sfc_box_20230526_12Z.nc` — 15° box slices
  from the 13 GB / 400 MB gribs.
- `hiccup_atm_source.nc`, `hiccup_sfc_source.nc` — ERA5-shape input for
  the HICCUP backend cell.
- `hiccup_atm_source.tc_bogus.nc`, `hiccup_sfc_source.tc_bogus.nc` —
  HICCUP round-trip outputs.
- `plot_baseline.png`, `plot_post_bogus.png`,
  `plot_vortex_diagnostics.png` — same PNGs embedded in the notebook.

## TODOs for next session

- [ ] **Warm-core anomaly sanity check.** The K&C vortex we're
  producing has a T anomaly of ~28 K at 500 hPa for Mawar's
  914-hPa central pressure. That's roughly 2× what dropsondes
  observe in real intense TCs (~10–15 K peak). Cross-check
  against the Kwon & Cheong 2009 paper — is this what their
  Fig. 5/6 shows for the p_c=914 hPa case, or is our T-anom
  implementation over-strong? Suspects to look at first:
  `T_anom_radial_K` construction in `tcinit/vortex.py:~410-455`
  (the `T_anom_radial[k] = -(p_pa * phi_diff / (R_D * denom)) * (...)`
  block). Compare against K&C eq. 14 numerically at a level in the
  middle troposphere.
- [ ] **Side-by-side baseline vs. bogus plots.** Current
  `plot_snapshot` in the notebook produces separate 1×3 figures.
  Rework to a 2×3 (baseline on top, post-bogus on bottom) sharing
  colour scales per column so the vortex imprint is obvious.
- [ ] **Plot the fields the classifier uses.** `Snapshot.detect_storm`
  uses MSLP (for loc-min), 10-m wind speed (for `rdr_from_wind`),
  and |∇MSLP| (for `rdr_from_grad`). Add a diagnostic figure that
  shows all three with the detected storm centre and both `rdr`
  candidates marked, so the classifier decisions are legible.
- [ ] **Invert y-axis on pressure-coordinate vertical plots.**
  In `plot_vortex_diagnostics`, the V_g cross-section panel already
  uses `ax.set_ylim(1000, 100)`; verify other pressure-axis plots
  do the same, and apply the convention consistently (surface at
  bottom, TOA at top).
- [ ] **Choose xlim for radial profile plots.** The tangential-wind
  and T-anomaly panels currently span 0–2500 km (the full radial
  grid). Meaningful storm structure is inside ~500 km. Pick a
  default xlim (e.g. 0–500 km with an inset of the full range)
  and apply consistently across the four diagnostic panels.

## Environment notes for future sessions

- **No torch / aurora / earth2studio / neuralgcm** in this container.
  SFNO and Aurora backend tests use `pytest.importorskip`; the
  case-study notebook cells for those backends print a skip message.
  All work must land on an HPC (Flight, cee-compute, etc.) to
  actually exercise those two backends end-to-end.
- **cfgrib + eccodes** installed via `pip install --user cfgrib eccodes`
  (2026-09-30 session). No system `grib_ls` / `cdo` — Python is the
  only path in.
- **cfgrib scratch**: writes `.idx` files and a `.cd/` dir next to the
  grib. Both now gitignored under `data/`.
- **data/ contents** (all local, all gitignored):
  - `hres_atmos_20230526_fh00_06_12_18.grb` (13 GB)
  - `hres_surf_20230526.grb` (400 MB)
  - `case_study_wp02_2023/` (~500 MB of cached artefacts)
  - `bwp022023.dat` (JTWC b-deck) — the only file committed to the repo.

## Where to start next session

1. Read this file.
2. Open `notebooks/case_study_wp02_2023.ipynb` — the five TODOs above
   are the natural next iteration on it. The notebook already runs
   end-to-end and produces the plots; the TODOs are about presentation
   quality + a physical sanity check.
3. If tackling the warm-core anomaly TODO first, the Kwon & Cheong 2009
   paper (Mon. Wea. Rev. 138, 1344–1367, doi:10.1175/2009MWR2943.1) is
   the source of truth. The prototype's implementation is at
   `external/causalcyclogenesis-pb-physics-bogus/causaltc/bogus_model/kwon_cheong_bogus.py`
   in case a cross-check between our port and the prototype is useful.
