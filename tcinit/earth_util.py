"""Great-circle geometry helpers and ATCF/HURDAT2 string parsers."""

from __future__ import annotations

import numpy as np

from tcinit.constants import EARTH_RADIUS_A


def km_to_arc_length_degrees(km_dist: float) -> float:
    """Convert a spherical-Earth surface distance from km to arc degrees."""
    return np.rad2deg(km_dist * 1000 / EARTH_RADIUS_A)


def haversine_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two (lat, lon) points."""
    lat1, lat2 = np.deg2rad(lat1), np.deg2rad(lat2)
    lon1, lon2 = np.deg2rad(lon1), np.deg2rad(lon2)
    rad_earth_km = EARTH_RADIUS_A / 1000
    inner = (
        1
        - np.cos(lat2 - lat1)
        + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))
    )
    return 2 * rad_earth_km * np.arcsin(np.sqrt(0.5 * inner))


def great_circle_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from (lat1, lon1) to (lat2, lon2).

    Returns radians on [0, 2*pi). 0 is true north; increasing clockwise
    (pi/2 = east, pi = south, 3*pi/2 = west). Coincident points return 0.
    """
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlambda = np.deg2rad(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    theta = np.arctan2(y, x)
    return np.mod(theta, 2 * np.pi)


def lat_str_to_val(latstr: str) -> float:
    """Convert a HURDAT2 latitude string (e.g., '239N') to decimal degrees."""
    assert len(latstr) == 4
    if latstr[-1] == "N":
        sign = 1
    elif latstr[-1] == "S":
        sign = -1
    else:
        raise ValueError(
            f"lat_str_to_val: received {latstr}, cannot determine sign of latitude"
        )
    return sign * float(latstr[:-1]) / 10


def lon_str_to_val(lonstr: str) -> float:
    """Convert a HURDAT2 longitude string (e.g., '1239W') to decimal degrees."""
    if lonstr[-1] == "E":
        sign = 1
    elif lonstr[-1] == "W":
        sign = -1
    else:
        raise ValueError(
            f"lon_str_to_val: received {lonstr}, cannot determine sign of longitude"
        )
    return sign * float(lonstr[:-1]) / 10
