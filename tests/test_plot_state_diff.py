"""plot_state_diff smoke test (pure xarray path)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tcinit.plotting import plot_state_diff  # noqa: E402
from tcinit.ideal_tc_vortex import BogusVortex  # noqa: E402


def test_plot_state_diff_writes_png(
    canonical_ds, snapshot, best_track, target_time, tmp_path: Path
):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    ds_after = bv.apply(canonical_ds)
    out = tmp_path / "diff.png"
    plot_state_diff(canonical_ds, ds_after, snapshot=snapshot, filename=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
