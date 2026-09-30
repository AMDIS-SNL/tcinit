"""plot_bogus PNG smoke test."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tcinit.plotting import plot_bogus  # noqa: E402
from tcinit.ideal_tc_vortex import BogusVortex  # noqa: E402


def test_plot_bogus_writes_png(snapshot, best_track, target_time, tmp_path: Path):
    bv = BogusVortex(snapshot, best_track, target_time).build()
    out = tmp_path / "bogus.png"
    plot_bogus(bv, filename=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
