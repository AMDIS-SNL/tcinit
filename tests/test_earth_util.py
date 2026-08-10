"""Sanity checks for earth_util geometry helpers."""

import numpy as np
import pytest

from tcinit.earth_util import (
    great_circle_bearing,
    haversine_dist,
    lat_str_to_val,
    lon_str_to_val,
)


def test_haversine_distance_zero_for_same_point():
    assert haversine_dist(15.0, 130.0, 15.0, 130.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_distance_1_deg_north_at_equator():
    # 1 deg of latitude ~ 111 km on a mean-radius sphere.
    d = haversine_dist(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111.19, rel=1e-3)


@pytest.mark.parametrize(
    "lat0,lon0,lat1,lon1,expected_rad",
    [
        (0.0, 0.0, 1.0, 0.0, 0.0),  # north
        (0.0, 0.0, 0.0, 1.0, np.pi / 2),  # east
        (0.0, 0.0, -1.0, 0.0, np.pi),  # south
        (0.0, 0.0, 0.0, -1.0, 3 * np.pi / 2),  # west
    ],
)
def test_great_circle_bearing_cardinals(lat0, lon0, lat1, lon1, expected_rad):
    assert great_circle_bearing(lat0, lon0, lat1, lon1) == pytest.approx(
        expected_rad, abs=1e-2
    )


def test_lat_lon_str_parsers():
    assert lat_str_to_val("239N") == pytest.approx(23.9)
    assert lat_str_to_val("125S") == pytest.approx(-12.5)
    assert lon_str_to_val("1239W") == pytest.approx(-123.9)
    assert lon_str_to_val("1206E") == pytest.approx(120.6)
