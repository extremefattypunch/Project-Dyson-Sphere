"""Physical and astronomical constants (SI units), each with its source.

This module is the single home of every fixed physical constant used by ``helioswarm``; no
physics module may hard-code one of these values elsewhere (import it from here).

Quantities that a study may legitimately vary (e.g. the solar irradiance at 1 au) are
*configuration* parameters (``RunConfig.constants``); their sourced default values are defined
here and referenced by the configuration schema, so that the number appears exactly once.

Time-scale note: gravitational parameters are the TDB-compatible values used by the JPL
planetary ephemerides (and by SPICE); the epochs that accompany them are TDB seconds past
J2000 (SPICE ephemeris time).
"""

import math
from typing import Final

# ---------------------------------------------------------------------------------------------
# Defining / exact constants
# ---------------------------------------------------------------------------------------------

AU_M: Final[float] = 149_597_870_700.0
"""Astronomical unit [m]. Exact by definition: IAU 2012 Resolution B2."""

SPEED_OF_LIGHT_M_S: Final[float] = 299_792_458.0
"""Speed of light in vacuum c [m s^-1]. Exact by definition of the metre (SI, CGPM 1983;
CODATA 2018)."""

SECONDS_PER_DAY: Final[float] = 86_400.0
"""Seconds per day [s]. A 'day' in configuration files is exactly 86400 s (the Julian-day
length used by IAU/SPICE), never a solar day."""

M_PER_KM: Final[float] = 1000.0
"""Metres per kilometre [m km^-1] (exact). SPICE returns km and km/s; GM in km^3 s^-2 converts
with M_PER_KM**3 = 1e9 (exact)."""

ARCSEC_TO_RAD: Final[float] = math.pi / (180.0 * 3600.0)
"""Arcseconds to radians [rad arcsec^-1] (definition: 1 arcsec = pi / 648000 rad)."""

# ---------------------------------------------------------------------------------------------
# Gravitation
# ---------------------------------------------------------------------------------------------

GM_SUN_M3_S2: Final[float] = 1.3271244004127942e20
"""Heliocentric gravitational constant GM_sun [m^3 s^-2] (TDB-compatible).

Source: DE440 value, as distributed by NAIF in the generic text PCK ``gm_de440.tpc``
(``BODY10_GM = 1.3271244004127942E+11`` km^3 s^-2; Park et al. 2021, AJ 161, 105,
doi:10.3847/1538-3881/abd414). Converted km^3 -> m^3 by the exact decimal factor 1e9; the
literal here is the nearest float64 to that decimal. M2 loads ``gm_de440.tpc`` through SPICE and
must agree with this constant to float64 rounding (the kernel value times 1e9 may differ by
one ulp from the literal).
"""

# ---------------------------------------------------------------------------------------------
# Radiation
# ---------------------------------------------------------------------------------------------

STEFAN_BOLTZMANN_W_M2_K4: Final[float] = 5.670374419e-8
"""Stefan-Boltzmann constant sigma [W m^-2 K^-4]. CODATA 2018 (exact in the 2019 SI to the
quoted digits: sigma = 2 pi^5 k_B^4 / (15 h^3 c^2) with exact k_B, h, c)."""

SOLAR_IRRADIANCE_NOMINAL_1AU_W_M2: Final[float] = 1361.0
"""Nominal total solar irradiance at 1 au S_sun^N [W m^-2]. IAU 2015 Resolution B3 (nominal
conversion constant; consistent with Kopp & Lean 2011, GRL 38, L01706, 1360.8 +/- 0.5 W m^-2).
Used as the *default* of the configurable ``constants.solar_constant_1au_W_m2``."""

CMB_TEMPERATURE_K: Final[float] = 2.72548
"""Temperature of the cosmic microwave background T_CMB [K]. Fixsen (2009), ApJ 707, 916
(2.72548 +/- 0.00057 K). Used as the *default* of the configurable radiative sink temperature
``thermal.space_temperature_K``."""

# ---------------------------------------------------------------------------------------------
# Sun
# ---------------------------------------------------------------------------------------------

SOLAR_RADIUS_NOMINAL_M: Final[float] = 6.957e8
"""Nominal solar radius R_sun^N [m]. IAU 2015 Resolution B3 (exact nominal conversion constant,
695 700 km)."""

# ---------------------------------------------------------------------------------------------
# Moon
# ---------------------------------------------------------------------------------------------

MOON_RADIUS_M: Final[float] = 1_737_400.0
"""Mean radius of the Moon [m] (sphere). IAU WGCCRE 2015 report: Archinal et al. (2018),
Celest. Mech. Dyn. Astron. 130, 22, table 5 (1737.4 km); identical to ``BODY301_RADII`` of the
NAIF text PCK ``pck00011.tpc`` (a test reads the kernel and compares)."""

# ---------------------------------------------------------------------------------------------
# Earth
# ---------------------------------------------------------------------------------------------

WGS84_SEMIMAJOR_AXIS_M: Final[float] = 6_378_137.0
"""WGS 84 ellipsoid semi-major (equatorial) axis a [m]. Defining parameter, NGA.STND.0036
(WGS 84, 2014)."""

WGS84_INVERSE_FLATTENING: Final[float] = 298.257223563
"""WGS 84 ellipsoid inverse flattening 1/f [1]. Defining parameter, NGA.STND.0036."""

WGS84_FLATTENING: Final[float] = 1.0 / WGS84_INVERSE_FLATTENING
"""WGS 84 ellipsoid flattening f [1] (derived: 1 / WGS84_INVERSE_FLATTENING)."""

WGS84_SEMIMINOR_AXIS_M: Final[float] = WGS84_SEMIMAJOR_AXIS_M * (1.0 - WGS84_FLATTENING)
"""WGS 84 ellipsoid semi-minor (polar) axis b = a (1 - f) [m] (derived; 6 356 752.314 2 m,
NGA.STND.0036 table 3.1)."""

WGS84_FIRST_ECCENTRICITY_SQUARED: Final[float] = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
"""WGS 84 first eccentricity squared e^2 = f (2 - f) [1] (derived; 6.694 379 990 14e-3,
NGA.STND.0036 table 3.1)."""

EARTH_ROTATION_ANGLE_RATE_REV_PER_UT1_DAY: Final[float] = 1.00273781191135448
"""Rate of the Earth Rotation Angle [revolutions per UT1 day of 86400 s]. IERS Conventions
(2010), IERS Technical Note 36, eq. (5.15): ERA(T_u) = 2 pi (0.7790572732640 +
1.00273781191135448 T_u)."""

STELLAR_DAY_S: Final[float] = SECONDS_PER_DAY / EARTH_ROTATION_ANGLE_RATE_REV_PER_UT1_DAY
"""Earth's rotation period with respect to the celestial reference (ICRF) [UT1 s], i.e. the
*stellar* day, 86164.0989... s, derived from the IERS 2010 Earth Rotation Angle rate.

This is the inertial rotation period (the "sidereal day" item of the M1 brief). It is named
*stellar* on purpose: it is NOT the equinox-referenced *mean sidereal day* (86164.0905 s,
GMST rate), which is about 8.4 ms shorter because the equinox precesses, and which this module
deliberately does not provide. It also differs from the period implied by the simplified
IAU_EARTH prime-meridian rate of the text PCK (360.9856235 deg/day), which is 3.1e-8 relative
*shorter*. M3's "one rotation per sidereal day" test in the IAU_EARTH frame must use that
frame's own rate and say which day it means.
"""

# ---------------------------------------------------------------------------------------------
# Reference frames
# ---------------------------------------------------------------------------------------------

OBLIQUITY_J2000_ARCSEC: Final[float] = 84_381.448
"""Mean obliquity of the ecliptic at J2000 [arcsec] (IAU 1976 value, Lieske et al. 1977,
A&A 58, 1). This is the angle SPICE uses to define the ECLIPJ2000 frame from J2000
(NAIF Frames Required Reading, built-in inertial frame "ECLIPJ2000")."""

OBLIQUITY_J2000_RAD: Final[float] = OBLIQUITY_J2000_ARCSEC * ARCSEC_TO_RAD
"""Mean obliquity of the ecliptic at J2000 [rad] (OBLIQUITY_J2000_ARCSEC converted)."""

__all__ = [
    "ARCSEC_TO_RAD",
    "AU_M",
    "CMB_TEMPERATURE_K",
    "EARTH_ROTATION_ANGLE_RATE_REV_PER_UT1_DAY",
    "GM_SUN_M3_S2",
    "MOON_RADIUS_M",
    "M_PER_KM",
    "OBLIQUITY_J2000_ARCSEC",
    "OBLIQUITY_J2000_RAD",
    "SECONDS_PER_DAY",
    "SOLAR_IRRADIANCE_NOMINAL_1AU_W_M2",
    "SOLAR_RADIUS_NOMINAL_M",
    "SPEED_OF_LIGHT_M_S",
    "STEFAN_BOLTZMANN_W_M2_K4",
    "STELLAR_DAY_S",
    "WGS84_FIRST_ECCENTRICITY_SQUARED",
    "WGS84_FLATTENING",
    "WGS84_INVERSE_FLATTENING",
    "WGS84_SEMIMAJOR_AXIS_M",
    "WGS84_SEMIMINOR_AXIS_M",
]
