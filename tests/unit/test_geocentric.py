"""Closed-form perturbation estimates of the GEO baseline (helioswarm.geocentric; M7b review).

References are independent closed forms:

* J2: synchronism ``omega^2 r = (GM/r^2)(1 + (3/2) J2 (R/r)^2)`` with ``r = r0 (1 + x)``,
  ``r0^3 = GM/omega^2``, gives ``(1 + x)^3 = 1 + (3/2) J2 (R/r0)^2 (1 + x)^-2``, i.e.
  ``x = (1/2) J2 (R/r0)^2 + O(J2^2)``;
* SRP: the eccentricity-circle radius ``e_r = (3/2) a / (n r n_sun)`` of the orbit-averaged
  Gauss equations, evaluated by hand for the documented cases.
"""

from __future__ import annotations

import math

import pytest

from helioswarm.geocentric import (
    geostationary_radius_m,
    j2_synchronous_radius_offset_m,
    srp_eccentricity_circle_radius,
)

GM_EARTH = 3.9860043551e14  # m^3 s^-2, BODY399_GM of gm_de440.tpc (as documented)
OMEGA = 2.0 * math.pi / 86164.1004  # rad/s, IAU_EARTH stellar day of pck00011 (M7b)
J2 = 1.0826e-3  # EGM2008 (Pavlis et al. 2012), docs/model_assumptions.md M7b
R_E = 6378137.0  # m, WGS 84 equatorial radius


def test_j2_synchronous_radius_offset_matches_the_first_order_closed_form() -> None:
    r0 = geostationary_radius_m(GM_EARTH, OMEGA)
    assert r0 == pytest.approx(42_164_172.6, abs=1.0)
    dr = j2_synchronous_radius_offset_m(GM_EARTH, OMEGA, J2, R_E)
    first_order = 0.5 * J2 * (R_E / r0) ** 2 * r0
    # Second-order term: x^2 relative to x, x = 1.2e-5, i.e. ~1e-5 of 522 m = 5 mm:
    # tolerance 0.05 m; a missing factor 1/2 or (R/r) instead of (R/r)^2 fails by > 260 m.
    assert dr == pytest.approx(first_order, abs=0.05)
    assert dr == pytest.approx(522.2, abs=0.1)  # documented value
    # The fixed point satisfies the synchronism equation itself (relative 1e-12).
    r = r0 + dr
    lhs = OMEGA**2 * r
    rhs = GM_EARTH / r**2 * (1.0 + 1.5 * J2 * (R_E / r) ** 2)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert j2_synchronous_radius_offset_m(GM_EARTH, OMEGA, 0.0, R_E) == 0.0


@pytest.mark.parametrize(
    ("area_to_mass_m2_kg", "e_expected", "radial_km", "libration_deg"),
    [
        # A/m 0.02 m^2/kg (the heliocentric configs' SRP dynamics): documented 2.9e-4.
        (0.02, 2.892e-4, 12.19, 0.03314),
        # The comparison satellite: 10 000 m^2 collector / 16.69 t mass proxy = 0.599 m^2/kg.
        (10000.0 / 16690.0, 8.665e-3, 365.3, 0.9929),
    ],
)
def test_srp_eccentricity_circle_matches_the_closed_form(
    area_to_mass_m2_kg: float, e_expected: float, radial_km: float, libration_deg: float
) -> None:
    # a_srp = (S/c) C_r A/m with S = 1361 W/m^2 at 1 au, C_r = 1.3.
    a_srp = 1361.0 / 299792458.0 * 1.3 * area_to_mass_m2_kg
    r = 42_164_172.6
    n_sun = 2.0 * math.pi / (365.25636 * 86400.0)  # sidereal year
    e = srp_eccentricity_circle_radius(a_srp, OMEGA, r, n_sun)
    assert e == pytest.approx(1.5 * a_srp / (OMEGA * r * n_sun), rel=1e-15)
    # Hand-evaluated values (4 significant figures, the documented ones): 1e-3 relative.
    assert e == pytest.approx(e_expected, rel=1e-3)
    assert e * r / 1e3 == pytest.approx(radial_km, rel=1e-3)
    assert math.degrees(2.0 * e) == pytest.approx(libration_deg, rel=1e-3)
    # Linear in the acceleration: 30x A/m -> 30x eccentricity.
    e2 = srp_eccentricity_circle_radius(2.0 * a_srp, OMEGA, r, n_sun)
    assert e2 == pytest.approx(2.0 * e, rel=1e-14)
    with pytest.raises(ValueError, match="> 0"):
        srp_eccentricity_circle_radius(0.0, OMEGA, r, n_sun)
