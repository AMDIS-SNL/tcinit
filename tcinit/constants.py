"""Physical constants and unit conversions used across tcinit."""

# Mean Earth radius in meters (denoted 'a' in atmospheric science literature).
# This is the mean sea-level radius; Aurora uses the equatorial radius, which
# differs by ~0.1% (~7 km).
EARTH_RADIUS_A = 6371220.0

# Earth rotation rate (rad/s).
OMEGA_EARTH = 7.2921e-5

# Standard gravity (m/s^2).
G_0 = 9.80665
# Same value under the MSE-diagnostic name.
G_STD = G_0

# Gas constant for dry air (J/(kg*K)).
R_D = 287.05

# Specific heat of dry air at constant pressure (J/(kg*K)).
CP_DRY = 1004.6

# Ratio of molecular weights of water vapor to dry air.
EPSILON_RATIO = 0.622

# Latent heat of vaporization at 0 deg C (J/kg).
LV = 2.501e6

# Unit conversions.
KNOTS_TO_MPS = 0.5144444
NM_TO_KM = 1.852
HPA_TO_PA = 100.0
