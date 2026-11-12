"""Tests for helioswarm.constants (PDF §13.1: "constants / units").

Reference values are either exact SI definitions or are recomputed here from independent,
cited relations; they are not copied from the module under test.
"""

from __future__ import annotations

import math
import re
import token
import tokenize
from pathlib import Path

import pytest

from helioswarm import constants as hc
from helioswarm.config import ConstantsConfig, ThermalConfig

SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "helioswarm"


def test_exact_defining_constants() -> None:
    # IAU 2012 B2 (au) and the SI definition of the metre (c): exact integers.
    assert hc.AU_M == 149_597_870_700
    assert hc.SPEED_OF_LIGHT_M_S == 299_792_458
    assert hc.SECONDS_PER_DAY == 86_400
    # IAU 2015 B3 nominal solar radius 695 700 km and nominal TSI 1361 W m^-2.
    assert hc.SOLAR_RADIUS_NOMINAL_M == 695_700_000
    assert hc.SOLAR_IRRADIANCE_NOMINAL_1AU_W_M2 == 1361
    # WGS 84 defining parameters (NGA.STND.0036).
    assert hc.WGS84_SEMIMAJOR_AXIS_M == 6_378_137
    assert hc.WGS84_INVERSE_FLATTENING == 298.257223563
    assert pytest.approx(1.0, rel=1e-16, abs=0) == hc.WGS84_FLATTENING * 298.257223563


def test_gm_sun_matches_de440_kernel_literal() -> None:
    # gm_de440.tpc: BODY10_GM = 1.3271244004127942E+11 km^3/s^2 (NAIF generic PCK, retrieved
    # 2026-11-12, SHA-256 924ddf4f...f140). km^3 -> m^3 is the exact factor 1e9.
    kernel_km3_s2 = 1.3271244004127942e11
    assert kernel_km3_s2 * 1e9 == pytest.approx(hc.GM_SUN_M3_S2, rel=2.3e-16, abs=0)  # <= 1 ulp


def test_gm_sun_consistent_with_gaussian_gravitational_constant() -> None:
    # Independent check: the IAU 1976 Gaussian constant k = 0.01720209895 au^(3/2) d^-1
    # defines GM_sun = k^2 au^3 / d^2. With the IAU 2012 fixed au (whose 2009 best estimate
    # had a 3 m, i.e. 2e-11 relative, uncertainty -> 6e-11 in au^3), the DE440 GM must agree
    # to ~1e-10. This rejects the IAU 1976 GM (1.32712438e20, 1.5e-8 off) and the older
    # 1.32712440018e20 (1.7e-10 off).
    k = 0.01720209895
    gm_gauss = k * k * hc.AU_M**3 / 86_400.0**2
    assert pytest.approx(gm_gauss, rel=1e-10, abs=0) == hc.GM_SUN_M3_S2


def test_stefan_boltzmann_from_exact_si_constants() -> None:
    # sigma = 2 pi^5 k_B^4 / (15 h^3 c^2) with the exact 2019-SI k_B, h, c. CODATA 2018
    # quotes sigma to 10 significant digits, so the rounding is <= 0.5e-9 / 5.67 = 8.8e-11.
    k_b = 1.380649e-23
    h = 6.62607015e-34
    c = 299_792_458.0
    sigma = 2.0 * math.pi**5 * k_b**4 / (15.0 * h**3 * c**2)
    assert sigma == pytest.approx(hc.STEFAN_BOLTZMANN_W_M2_K4, rel=1e-10, abs=0)


def test_stellar_day_against_iers_era_and_iau_earth_rotation_model() -> None:
    # (1) Defining relation, recomputed here: IERS Conventions (2010), TN36 eq. (5.15),
    # ERA rate 1.00273781191135448 rev per UT1 day of 86400 s -> period 86400 / rate.
    # Float64 division: agreement to 1 ulp (~1.5e-11 s) is expected; 1e-9 s is ample.
    period_era_s = 86_400.0 / 1.00273781191135448
    assert pytest.approx(period_era_s, rel=0, abs=1e-9) == hc.STELLAR_DAY_S
    # (2) Plausibility against an independent convention: the IAU WGCCRE Earth prime-meridian
    # rate W' = 360.9856235 deg/day (pck00011.tpc, BODY399_PM) gives 86164.096228 s, 3.1e-8
    # relative *shorter* than the ERA period (measured -3.106e-8). rel=5e-8 brackets that
    # and, by itself, rejects the mean sidereal (equinox) day 86164.0905 s, which is 6.65e-8
    # from the PCK period, and the solar day (2.7e-3 away).
    period_pck_s = 86_400.0 * 360.0 / 360.9856235
    assert period_pck_s == pytest.approx(hc.STELLAR_DAY_S, rel=5e-8, abs=0)
    assert period_pck_s < hc.STELLAR_DAY_S
    # (3) Explicitly not the mean sidereal day (86164.0905 s, 8.4 ms shorter).
    assert abs(hc.STELLAR_DAY_S - 86_164.0905) > 5e-3


def test_obliquity_conversion() -> None:
    # 84381.448 arcsec = 23 deg 26' 21.448" (IAU 1976).
    expected = math.radians(23.0 + 26.0 / 60.0 + 21.448 / 3600.0)
    assert expected == pytest.approx(hc.OBLIQUITY_J2000_RAD, rel=1e-15, abs=0)
    assert math.radians(1.0 / 3600.0) == pytest.approx(hc.ARCSEC_TO_RAD, rel=1e-15, abs=0)


def test_dimensional_consistency_circular_speed_and_year() -> None:
    # v = sqrt(GM/r) at 1 au must be Earth's mean orbital speed 29.78 km/s (NASA Earth fact
    # sheet), and 2 pi sqrt(au^3/GM) the Gaussian year 2 pi / k = 365.2568983 d. A unit slip
    # (km vs m, AU vs m) is off by orders of magnitude.
    v = math.sqrt(hc.GM_SUN_M3_S2 / hc.AU_M)
    assert v == pytest.approx(29_780.0, rel=1e-3, abs=0)
    year_d = 2.0 * math.pi * math.sqrt(hc.AU_M**3 / hc.GM_SUN_M3_S2) / hc.SECONDS_PER_DAY
    assert year_d == pytest.approx(2.0 * math.pi / 0.01720209895, rel=1e-10, abs=0)
    # Light time over 1 au: 499.004784 s (AU/c, both exact).
    assert pytest.approx(499.004_783_836, rel=1e-11, abs=0) == hc.AU_M / hc.SPEED_OF_LIGHT_M_S


def test_config_default_solar_constant_is_the_module_constant() -> None:
    assert ConstantsConfig().solar_constant_1au_W_m2 == hc.SOLAR_IRRADIANCE_NOMINAL_1AU_W_M2


def test_cmb_temperature_and_config_default() -> None:
    # Fixsen (2009), ApJ 707, 916: T_CMB = 2.72548 +/- 0.00057 K.
    assert hc.CMB_TEMPERATURE_K == 2.72548
    assert ThermalConfig.model_fields["space_temperature_K"].default == hc.CMB_TEMPERATURE_K


# Distinctive digit strings of the constants; no numeric literal in another source module may
# match them (docstrings and comments may mention the values; only NUMBER tokens are checked).
_FORBIDDEN_LITERALS = {
    "astronomical unit": r"149_?597_?870_?700",
    "speed of light": r"299_?792_?458",
    "GM_sun": r"1\.32712440",
    "Stefan-Boltzmann": r"5\.670374",
    "obliquity": r"84_?381\.448",
    "WGS84 a": r"6_?378_?137",
    "WGS84 1/f": r"298\.257223563",
    "solar radius": r"6\.957e|695_?700",
    "solar constant": r"^1_?361(\.0*)?$",
    "CMB temperature": r"^2\.725",
}


def _number_tokens(path: Path) -> list[str]:
    with path.open("rb") as fh:
        return [tok.string for tok in tokenize.tokenize(fh.readline) if tok.type == token.NUMBER]


@pytest.mark.parametrize("name", sorted(_FORBIDDEN_LITERALS))
def test_constants_are_not_hard_coded_elsewhere(name: str) -> None:
    pattern = re.compile(_FORBIDDEN_LITERALS[name])
    offenders = [
        p.name
        for p in sorted(SRC_DIR.glob("*.py"))
        if p.name != "constants.py" and any(pattern.search(n) for n in _number_tokens(p))
    ]
    assert offenders == [], f"{name} hard-coded in {offenders}; import it from constants.py"


def test_literal_scan_detects_a_hard_coded_constant(tmp_path: Path) -> None:
    # The scan itself must be able to fail.
    f = tmp_path / "bad.py"
    f.write_text(
        '"""1.32712440 in a docstring is fine."""\nGM = 1.32712440018e20\nS = 1361.0\nT = 2.725\n'
    )
    nums = _number_tokens(f)
    assert any(re.search(_FORBIDDEN_LITERALS["GM_sun"], n) for n in nums)
    assert any(re.search(_FORBIDDEN_LITERALS["solar constant"], n) for n in nums)
    assert any(re.search(_FORBIDDEN_LITERALS["CMB temperature"], n) for n in nums)
    assert nums == ["1.32712440018e20", "1361.0", "2.725"]
