"""to_comparison_dataframe and plot_best_track_comparison against a real ATCF file."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from tcinit import BestTrack  # noqa: E402
from tcinit.rollout_snapshots import plot_best_track_comparison  # noqa: E402
from tcinit.rollout_tracking import track_rollout  # noqa: E402
from tests.test_rollout_tracking import (  # noqa: E402
    START,
    _extract_box,
    _model_state,
    _track,
)

BWP = Path(__file__).resolve().parents[1] / "data" / "bwp022023.dat"


def _rollout(intervened=False):
    centers = _track()
    return track_rollout(
        (_model_state(c) for c in centers),
        _extract_box,
        cyclone_id="WP022023",
        start_time=START,
        initial_center=centers[0],
        intervened=intervened,
        verbose=False,
    )


@pytest.fixture
def mawar():
    return BestTrack(BWP)


def test_comparison_dataframe_has_one_numeric_row_per_time(mawar):
    # bwp022023.dat repeats each time for the 34/50/64 kt radii, as strings.
    df = _rollout().to_comparison_dataframe(mawar)
    assert len(df) == 4
    assert df["datetime"].is_unique
    assert df["bt_MSLP"].dtype.kind in "if"
    assert df["bt_VMAX"].dtype.kind in "if"
    # Best track: 26 May 06Z 160 kt / 897 hPa, 12Z 145 kt / 914 hPa.
    assert df["bt_MSLP"].iloc[:2].tolist() == [897.0, 914.0]
    assert df["bt_VMAX"].iloc[:2].tolist() == [160.0, 145.0]


def test_plot_best_track_comparison_writes_png(mawar, tmp_path: Path):
    out = tmp_path / "vs_best_track.png"
    plot_best_track_comparison(
        _rollout(), _rollout(intervened=True), mawar, out, title="test"
    )
    assert out.exists()
    assert out.stat().st_size > 0
