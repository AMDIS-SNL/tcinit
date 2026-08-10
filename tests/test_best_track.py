"""MSLP plausibility filter and per-datetime accessor tests."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tcinit.best_track import BestTrack
from tcinit.constants import HPA_TO_PA


def _fake_bt(mslp_hpa: float):
    bt = BestTrack.__new__(BestTrack)
    bt.ID = "TEST"
    bt.df_best_tracks_history = pd.DataFrame(
        {
            "datetimes": [datetime(2020, 9, 1, 0)],
            "LAT_DEG": [15.0],
            "LON_DEG": [130.0],
            "MSLP": [mslp_hpa],
            "VMAX": [85.0],
        }
    )
    return bt


def test_get_central_pressure_accepts_plausible_mslp():
    bt = _fake_bt(950.0)
    p_pa = bt.get_central_pressure(datetime(2020, 9, 1, 0))
    assert p_pa == pytest.approx(950.0 * HPA_TO_PA)


@pytest.mark.parametrize(
    "bad_mslp", [-999.0, 0.0, 700.0, 849.9, 1100.0, 1234.0, 987654321.0]
)
def test_get_central_pressure_rejects_implausible_mslp(bad_mslp):
    bt = _fake_bt(bad_mslp)
    with pytest.raises(ValueError, match="implausible MSLP"):
        bt.get_central_pressure(datetime(2020, 9, 1, 0))


def test_get_windspeed_rejects_non_positive():
    bt = _fake_bt(950.0)
    bt.df_best_tracks_history.loc[0, "VMAX"] = 0.0
    with pytest.raises(ValueError, match="implausible VMAX"):
        bt.get_windspeed(datetime(2020, 9, 1, 0))


def test_get_30kt_radius_none_when_blank():
    bt = _fake_bt(950.0)
    # No RAD34_* columns present → None.
    assert bt.get_30kt_radius(datetime(2020, 9, 1, 0)) is None
