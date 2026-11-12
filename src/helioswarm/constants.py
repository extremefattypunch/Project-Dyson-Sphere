"""Physical and astronomical constants (SI units).

Only constants that are exact by definition or fixed by an IAU resolution live here as module
constants. Quantities that a study may legitimately vary (e.g. the solar irradiance at 1 au)
are *configuration* parameters (``RunConfig.constants``) with a sourced default, so that
every run records the value it used.

M0 defines the subset needed for configuration validation; M1 extends this module.
"""

from typing import Final

AU_M: Final[float] = 149_597_870_700.0
"""Astronomical unit [m]. Exact by definition: IAU 2012 Resolution B2."""

SOLAR_RADIUS_NOMINAL_M: Final[float] = 6.957e8
"""Nominal solar radius R_sun^N [m]. IAU 2015 Resolution B3 (exact nominal conversion constant)."""

SECONDS_PER_DAY: Final[float] = 86_400.0
"""Seconds per day [s]. A 'day' in configuration files is exactly 86400 s (the Julian-day
length used by IAU/SPICE), never a solar day."""

__all__ = ["AU_M", "SECONDS_PER_DAY", "SOLAR_RADIUS_NOMINAL_M"]
