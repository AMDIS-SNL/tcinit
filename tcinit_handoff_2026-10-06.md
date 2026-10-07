# tcinit handoff — 2026-10-06

## Session summary

Short session, two scoped maintenance items on top of the 2026-10-01
state. No physics work; no new rollouts.

1. **Pickle shim extracted into tcinit.** The ad-hoc `aurora`/`torch`
   stubs that `/tmp/test_comparison.py` carried last session are now
   `tcinit/pickle_shim.py`, re-exported from `tcinit/__init__.py` as
   `install_causaltc_shim` and `load_causaltc_pickle`. One-liner usage:
   `history = tcinit.load_causaltc_pickle("…pkl")`. Idempotent and
   HPC-safe: only installs stubs when real `aurora` / `torch` are not
   already importable, only edits `sys.path` when `causaltc` is not
   already importable. Default `causaltc_src` resolves to
   `<repo>/external/causalcyclogenesis-pb-physics-bogus`. Smoke-tested
   end-to-end on `data/aurora_rollout_control/bwp022023/2023052606_steps12.pkl`:
   13 `AuroraSnapshot`s loaded, `mslp` shape `(200, 201)`.

2. **nbstripout wired into git.** `.gitattributes` (committed) holds
   the general `*.ipynb filter=nbstripout` / `diff=ipynb` rules plus a
   `-filter -diff` override for `notebooks/case_study_wp02_2023.ipynb`
   so its embedded PNGs survive the next commit (user elected to
   exempt it so GitHub keeps rendering it without re-execution). Added
   `nbstripout` to the `dev` extra in `pyproject.toml` and a
   **Developer setup** section to `README.md`. The local clone is
   wired via `nbstripout --install --attributes .gitattributes`.
   Verified by staging a dummy notebook (outputs stripped) and the
   case-study notebook (3 outputs preserved).

## .gitattributes / nbstripout gotcha worth repeating

`nbstripout --install` *without* `--attributes` writes its rule to
`.git/info/attributes`, which outranks the committed `.gitattributes`
and silently shadows the per-path override. Debugged this live today:
`git check-attr -a` kept reporting `filter: nbstripout` for the
exempt path until `.git/info/attributes` was cleared via
`nbstripout --uninstall` and re-installed with the `--attributes
.gitattributes` flag. Documented in the README.

## Files touched (not committed)

```
M pyproject.toml              # nbstripout in dev extra
M tcinit/__init__.py          # export install_causaltc_shim, load_causaltc_pickle
M README.md                   # Developer setup section
?? .gitattributes             # nbstripout rules + case-study override
?? tcinit/pickle_shim.py      # new helper
```

Still 3 commits ahead of `origin/main` from the 2026-10-01 session;
nothing new committed today.

## Open from previous sessions (unchanged)

Carried forward from `tcinit_handoff_2026-10-01.md`:

- **Model backend ports** through the rollout + comparison pipeline
  (named scope for "next" session, still pending):
  1. SFNO — closest to Aurora.
  2. WeatherNext-2 — xarray-native, probably easiest.
  3. NeuralGCM — surface-free.
  4. HICCUP — file-based, one-shot comparison.
- K&C warm-core sanity check against Haiyan AMSU-A microwave
  retrievals (CIMSS archive) before publishing.
- Rollout `rdr` noise: force both ctrl/bogus onto a common
  `fixed_rdr_km` for comparison studies.
- Cosmetic: MSE time-series panels fall back to Jan-2000 x-axis when
  all values are NaN (yasa case).

## New todo flagged today

- **CI testing workflows.** No `.github/workflows/` in the repo yet.
  Minimal first pass should run the existing pytest suite (77 tests
  pass locally per the 09-30 handoff, 2 torch-gated smoke tests skip)
  and `black --check`. Likely needs:
  - `.github/workflows/ci.yml` with a Python matrix (3.10, 3.11, 3.12
    probably; 3.14 is what the container has but may be aggressive).
  - `pip install -e ".[dev]"` as install step.
  - Separate job or step for `black --check tcinit tests` (keeps the
    style gate parallel to tests).
  - Skip the aurora/neuralgcm-gated tests gracefully — they already
    use `pytest.importorskip`, so default behaviour should be fine.
  - Consider caching the pip wheel directory to keep runs fast.
  - Decide whether nbstripout check runs in CI (would catch devs
    who skipped the `nbstripout --install` step).

## Where to start next session

1. Read this file, then `tcinit_handoff_2026-10-01.md` for the
   larger model-backend scope.
2. Decide between the two live threads:
   - CI workflows (fresh todo, scoped and self-contained).
   - SFNO rollout port on HPC (the named "next" scope from 10-01).
3. If staying local in the container, CI is a natural fit — no HPC
   needed, no new pickles needed.
