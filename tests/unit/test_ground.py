"""Ground sites: WGS 84 geodetic conversion, local frame, Earth orientation and inertial states.

Reference values are closed forms (ellipsoid geometry), independent implementations (SPICE
``georec``/``surfnm``, astropy ``EarthLocation``/ERFA) or the IAU rotation constants read
directly from the loaded PCK and evaluated with an independent closed form.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import spiceypy

from helioswarm.constants import (
    SECONDS_PER_DAY,
    STELLAR_DAY_S,
    WGS84_FIRST_ECCENTRICITY_SQUARED,
    WGS84_FLATTENING,
    WGS84_SEMIMAJOR_AXIS_M,
    WGS84_SEMIMINOR_AXIS_M,
)
from helioswarm.ephemeris import AnalyticEphemeris, SpiceEphemeris
from helioswarm.frames import rotation_x
from helioswarm.ground import (
    EarthOrientation,
    GroundSite,
    SpiceEarthOrientation,
    UniformEarthRotation,
    geodetic_to_earth_fixed_m,
    local_enu_basis,
    site_trajectory,
)
from helioswarm.kernels import LoadedKernels
from helioswarm.timescale import utc_to_tdb_s

A = WGS84_SEMIMAJOR_AXIS_M
B = WGS84_SEMIMINOR_AXIS_M

# (latitude deg, longitude deg, height m): equator, poles, mid-latitudes, both hemispheres,
# negative longitude, high altitude, below the ellipsoid.
SITES = [
    (0.0, 0.0, 0.0),
    (90.0, 0.0, 0.0),
    (-90.0, 45.0, 0.0),
    (42.3601, -71.0942, 30.0),
    (-33.8688, 151.2093, 58.0),
    (19.8207, -155.4681, 4205.0),
    (31.5, 35.5, -430.0),
    (67.0, -120.0, 1000.0),
]


# ---------------------------------------------------------------------------------------------
# Geodetic -> Earth-fixed
# ---------------------------------------------------------------------------------------------


def test_wgs84_derived_constants_match_nga_table() -> None:
    # NGA.STND.0036 (WGS 84, 2014) table 3.1 (derived geometric constants): b = 6356752.3142 m,
    # e^2 = 6.69437999014e-3. The table rounds b to 0.1 mm and e^2 to 12 significant digits.
    assert pytest.approx(6_356_752.3142, abs=1e-4) == B
    assert pytest.approx(6.69437999014e-3, rel=1e-11) == WGS84_FIRST_ECCENTRICITY_SQUARED
    assert pytest.approx(A * (1.0 - WGS84_FLATTENING), rel=1e-16) == B


def test_equator_prime_meridian_is_semimajor_axis_on_x() -> None:
    r = geodetic_to_earth_fixed_m(0.0, 0.0, 0.0)
    np.testing.assert_array_equal(r, [A, 0.0, 0.0])  # exact: cos 0 = 1, sin 0 = 0


def test_equator_90_east_is_semimajor_axis_on_y() -> None:
    r = geodetic_to_earth_fixed_m(0.0, math.pi / 2, 0.0)
    # cos(pi/2) = 6.1e-17 in float64, so x = a * 6.1e-17 ~ 4e-10 m.
    np.testing.assert_allclose(r, [0.0, A, 0.0], rtol=0, atol=1e-9)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_poles_are_semiminor_axis_on_z(sign: float) -> None:
    r = geodetic_to_earth_fixed_m(sign * math.pi / 2, 0.3, 0.0)
    # N(90 deg) (1 - e^2) = a sqrt(1 - e^2) = b; x, y carry cos(pi/2) ~ 6e-17 times N.
    np.testing.assert_allclose(r, [0.0, 0.0, sign * B], rtol=0, atol=1e-9)


@pytest.mark.parametrize(("lat", "lon", "h"), SITES)
def test_height_is_along_the_ellipsoid_normal(lat: float, lon: float, h: float) -> None:
    phi, lam = math.radians(lat), math.radians(lon)
    _e, _n, up = local_enu_basis(phi, lam)
    d = geodetic_to_earth_fixed_m(phi, lam, h + 1234.5) - geodetic_to_earth_fixed_m(phi, lam, h)
    # Exact property of the formula: r(h + dh) - r(h) = dh * up; rounding at |r| ~ 6.4e6 m is
    # ~1e-9 m, so 1e-8 m.
    np.testing.assert_allclose(d, 1234.5 * up, rtol=0, atol=1e-8)


@pytest.mark.parametrize(("lat", "lon", "h"), SITES)
def test_surface_point_lies_on_the_ellipsoid(lat: float, lon: float, h: float) -> None:
    x, y, z = geodetic_to_earth_fixed_m(math.radians(lat), math.radians(lon), 0.0)
    # Implicit equation of the WGS 84 ellipsoid; 1e-15 is float64 rounding of O(1) terms.
    assert (x * x + y * y) / A**2 + z * z / B**2 == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(("lat", "lon", "h"), SITES)
def test_geodetic_matches_spice_georec_and_astropy_below_1_mm(
    lat: float, lon: float, h: float
) -> None:
    astropy_units = pytest.importorskip("astropy.units")
    from astropy.coordinates import EarthLocation

    phi, lam = math.radians(lat), math.radians(lon)
    r = geodetic_to_earth_fixed_m(phi, lam, h)
    # SPICE georec: independent implementation (kernel-free), km.
    r_spice = np.asarray(spiceypy.georec(lam, phi, h / 1000.0, A / 1000.0, WGS84_FLATTENING)) * 1e3
    # astropy EarthLocation.from_geodetic(..., ellipsoid="WGS84") -> ERFA gd2gc.
    loc = EarthLocation.from_geodetic(
        lon * astropy_units.deg, lat * astropy_units.deg, h * astropy_units.m, ellipsoid="WGS84"
    )
    r_astropy = np.array([q.to_value(astropy_units.m) for q in (loc.x, loc.y, loc.z)])
    # The brief's acceptance tolerance is 1 mm; the implementations agree to ~1e-8 m.
    np.testing.assert_allclose(r, r_spice, rtol=0, atol=1e-3)
    np.testing.assert_allclose(r, r_astropy, rtol=0, atol=1e-3)
    # Round trip through SPICE's inverse (recgeo) recovers the geodetic coordinates.
    lon_s, lat_s, alt_s = spiceypy.recgeo(r / 1000.0, A / 1000.0, WGS84_FLATTENING)
    assert lat_s == pytest.approx(phi, abs=1e-12)
    assert alt_s * 1000.0 == pytest.approx(h, abs=1e-6)
    if abs(lat) < 90.0:
        assert math.remainder(lon_s - lam, 2 * math.pi) == pytest.approx(0.0, abs=1e-12)


def test_published_station_meades_ranch_reset() -> None:
    """Published geodetic and Cartesian coordinates of one control station (PDF §13.1 "known
    station coordinates").

    Source: U.S. National Geodetic Survey datasheet for PID KG0640, designation MEADES RANCH
    RESET (Osborne County, Kansas; a Federal Base Network station), datasheet95 v8.12.5.20,
    retrieved 2026-11-12 from https://www.ngs.noaa.gov/cgi-bin/ds_mark.prl?PidBox=KG0640:
    NAD 83(2011) position 39 13 26.71220 N, 098 32 31.74540 W, ellipsoid height 573.961 m
    (ADJUSTED); NAD 83(2011) X = -734,972.563 m, Y = -4,893,188.492 m, Z = 4,011,982.811 m.

    NAD 83 coordinates refer to the GRS 80 ellipsoid (a = 6 378 137 m, 1/f = 298.257222101;
    Moritz, "Geodetic Reference System 1980", J. Geodesy 74, 2000). That is a conversion of
    coordinates on one ellipsoid, so the NAD 83 vs WGS 84 *frame* offset (~1 m) plays no
    role; GRS 80 and WGS 84 differ only in b (0.105 mm), which moves this point by 0.11 mm.
    Tolerance per component 1.5 mm: the published X, Y, Z are rounded to 1 mm (+-0.5 mm), the
    height to 1 mm (+-0.5 mm along the normal), the angles to 1e-5 arcsec (+-0.15 mm), plus the
    0.11 mm ellipsoid difference for the WGS 84 evaluation. Measured: <= 0.46 mm.
    """
    lat = math.radians(39.0 + 13.0 / 60.0 + 26.71220 / 3600.0)
    lon = -math.radians(98.0 + 32.0 / 60.0 + 31.74540 / 3600.0)
    h = 573.961
    published = np.array([-734_972.563, -4_893_188.492, 4_011_982.811])
    f_grs80 = 1.0 / 298.257222101
    grs80 = geodetic_to_earth_fixed_m(lat, lon, h, eccentricity_squared=f_grs80 * (2 - f_grs80))
    wgs84 = geodetic_to_earth_fixed_m(lat, lon, h)
    np.testing.assert_allclose(grs80, published, rtol=0, atol=1.5e-3)
    np.testing.assert_allclose(wgs84, published, rtol=0, atol=1.5e-3)
    # The two ellipsoids place this point 0.11 mm apart (the b difference, projected).
    assert np.linalg.norm(grs80 - wgs84) < 2e-4


# ---------------------------------------------------------------------------------------------
# Local frame
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("lat", "lon", "h"), SITES)
def test_up_vector_is_the_ellipsoid_normal(lat: float, lon: float, h: float) -> None:
    site = GroundSite.from_geodetic("s", math.radians(lat), math.radians(lon), 0.0)
    x, y, z = site.position_ef_m
    grad = np.array([2 * x / A**2, 2 * y / A**2, 2 * z / B**2])  # gradient of the implicit eq.
    np.testing.assert_allclose(site.up_ef, grad / np.linalg.norm(grad), rtol=0, atol=1e-14)
    normal_spice = np.asarray(spiceypy.surfnm(A, A, B, np.array(site.position_ef_m)))
    np.testing.assert_allclose(site.up_ef, normal_spice, rtol=0, atol=1e-14)


@pytest.mark.parametrize(("lat", "lon", "h"), SITES)
def test_local_basis_is_orthonormal_right_handed_and_oriented(
    lat: float, lon: float, h: float
) -> None:
    phi, lam = math.radians(lat), math.radians(lon)
    east, north, up = local_enu_basis(phi, lam)
    m = np.stack([east, north, up])
    np.testing.assert_allclose(m @ m.T, np.eye(3), rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.cross(east, north), up, rtol=0, atol=1e-15)
    # Orientation: north = direction of increasing latitude, east = of increasing longitude
    # (finite differences of the position; skipped where the direction is undefined).
    if abs(lat) < 89.0:
        eps = 1e-7
        dn = geodetic_to_earth_fixed_m(phi + eps, lam, h) - geodetic_to_earth_fixed_m(
            phi - eps, lam, h
        )
        de = geodetic_to_earth_fixed_m(phi, lam + eps, h) - geodetic_to_earth_fixed_m(
            phi, lam - eps, h
        )
        assert np.dot(dn / np.linalg.norm(dn), north) == pytest.approx(1.0, abs=1e-9)
        assert np.dot(de / np.linalg.norm(de), east) == pytest.approx(1.0, abs=1e-9)


def test_site_from_config_converts_degrees() -> None:
    from helioswarm.config import SiteConfig

    cfg = SiteConfig(name="x", latitude_deg=45.0, longitude_deg=-90.0, altitude_m=100.0)
    site = GroundSite.from_config(cfg)
    assert site.latitude_rad == pytest.approx(math.pi / 4, rel=1e-15)
    assert site.longitude_rad == pytest.approx(-math.pi / 2, rel=1e-15)
    np.testing.assert_array_equal(
        site.position_ef_m, geodetic_to_earth_fixed_m(math.pi / 4, -math.pi / 2, 100.0)
    )
    assert site.frame == "IAU_EARTH"
    assert not site.position_ef_m.flags.writeable


def test_invalid_site_is_rejected() -> None:
    with pytest.raises(ValueError, match="latitude"):
        GroundSite.from_geodetic("bad", 2.0, 0.0)


# ---------------------------------------------------------------------------------------------
# Earth orientation: kernel-free test double
# ---------------------------------------------------------------------------------------------


def test_uniform_rotation_matrix_and_rate() -> None:
    rot = UniformEarthRotation(angle_at_j2000_rad=0.3, rate_rad_s=7.0e-5)
    t = np.array([0.0, 1000.0, -5.0e4])
    r = rot.rotation(t)
    for k in range(t.size):
        m = r.matrix[k]
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-15)
        assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-15)
        th = 0.3 + 7.0e-5 * t[k]
        # Active rotation: the Earth-fixed x-axis points at angle theta in the J2000 x-y plane.
        np.testing.assert_allclose(m[:, 0], [math.cos(th), math.sin(th), 0.0], atol=1e-15)
        h = 1e-2
        fd = (rot.rotation(t[k] + h).matrix[0] - rot.rotation(t[k] - h).matrix[0]) / (2 * h)
        # Central difference error ~ rate^3 h^2/6 ~ 6e-18; rounding ~ eps/h ~ 1e-14.
        np.testing.assert_allclose(r.matrix_rate_per_s[k], fd, rtol=0, atol=1e-13)
    assert isinstance(rot, EarthOrientation)


def test_site_trajectory_with_test_doubles_is_closed_form() -> None:
    eph = AnalyticEphemeris()
    rate = 7.2921e-5
    rot = UniformEarthRotation(angle_at_j2000_rad=0.0, rate_rad_s=rate)
    site = GroundSite.from_geodetic("eq", 0.0, 0.0, 0.0)
    t = np.array([0.0, 3600.0, 86400.0])
    st = site_trajectory(site, t, ephemeris=eph, earth_orientation=rot)
    earth = eph.states("EARTH", t)
    th = rate * t
    geo = A * np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
    # Positions ~1.5e11 m carry float64 rounding ~3e-5 m, so the difference is good to 1e-4 m.
    np.testing.assert_allclose(st.position_m - earth.position_m, geo, rtol=0, atol=1e-4)
    # v_site - v_earth = omega x r = rate * a * (-sin, cos, 0): 465 m/s at the equator.
    vgeo = rate * A * np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], axis=1)
    np.testing.assert_allclose(st.velocity_m_s - earth.velocity_m_s, vgeo, rtol=0, atol=1e-10)
    np.testing.assert_allclose(st.up_unit, geo / A, atol=1e-15)
    assert st.frame == "J2000"
    assert st.origin == "SOLAR SYSTEM BARYCENTER"
    # Earth-fixed coordinates are kept separately and unchanged.
    np.testing.assert_array_equal(st.site.position_ef_m, [A, 0.0, 0.0])
    np.testing.assert_allclose(st.as_cartesian().position_m, st.position_m)


# ---------------------------------------------------------------------------------------------
# Earth orientation: SPICE IAU_EARTH
# ---------------------------------------------------------------------------------------------


def _rot_z_passive(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _pool(name: str) -> np.ndarray:
    return np.asarray(spiceypy.gdpool(name, 0, 3), dtype=np.float64)


def _iau_earth_to_j2000_closed_form(et: float) -> np.ndarray:
    """IAU rotation model from the PCK constants (NAIF PCK Required Reading):
    TIPM = [W]_3 [pi/2 - delta]_1 [pi/2 + alpha]_3 (J2000 -> body-fixed, passive rotations),
    alpha = a0 + a1 T, delta = d0 + d1 T (T in Julian centuries), W = W0 + W1 d (d in days),
    all TDB past J2000. Returns TIPM^T (body-fixed -> J2000)."""
    d = et / SECONDS_PER_DAY
    tc = d / 36525.0
    ra, dec, pm = _pool("BODY399_POLE_RA"), _pool("BODY399_POLE_DEC"), _pool("BODY399_PM")
    alpha = math.radians(ra[0] + ra[1] * tc + ra[2] * tc * tc)
    delta = math.radians(dec[0] + dec[1] * tc + dec[2] * tc * tc)
    w = math.radians(pm[0] + pm[1] * d + pm[2] * d * d)
    tipm = _rot_z_passive(w) @ rotation_x(math.pi / 2 - delta) @ _rot_z_passive(math.pi / 2 + alpha)
    return tipm.T


@pytest.mark.spice
def test_iau_earth_rotation_equals_the_pck_closed_form(spice_kernels: LoadedKernels) -> None:
    eo = SpiceEarthOrientation(spice_kernels)
    epochs = np.array([0.0, utc_to_tdb_s("2020-06-15T12:00:00", spice_kernels), 9.5e8, -3.0e8])
    rot = eo.rotation(epochs)
    for k, et in enumerate(epochs):
        ref = _iau_earth_to_j2000_closed_form(float(et))
        # sxform evaluates the same formula; differences are float64 rounding of W, which is
        # ~7e4 rad at 2030 (float64 spacing 1.5e-11 rad): tolerance 3e-11.
        np.testing.assert_allclose(rot.matrix[k], ref, rtol=0, atol=3e-11)
        # dR/dt: central difference of the closed form with h = 10 s: truncation
        # omega^3 h^2 / 6 = 6.5e-12, rounding 1.5e-11 rad / (2 h) ~ 1e-12; tolerance 2e-11
        # (3e-7 of |dR/dt| ~ 7.3e-5 s^-1).
        h = 10.0
        fd = (_iau_earth_to_j2000_closed_form(et + h) - _iau_earth_to_j2000_closed_form(et - h)) / (
            2 * h
        )
        np.testing.assert_allclose(rot.matrix_rate_per_s[k], fd, rtol=0, atol=2e-11)
    assert eo.source_ids == ("pck00011.tpc", "gm_de440.tpc")


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


@pytest.mark.spice
def test_site_returns_after_one_rotation_period_and_turns_per_solar_day(
    spice_kernels: LoadedKernels,
) -> None:
    eo = SpiceEarthOrientation(spice_kernels)
    site = GroundSite.from_geodetic("eq", 0.0, math.radians(37.0), 0.0)
    w_rate_deg_day = float(_pool("BODY399_PM")[1])  # 360.9856235 deg/day in pck00011
    assert w_rate_deg_day == 360.9856235
    period_s = SECONDS_PER_DAY * 360.0 / w_rate_deg_day  # IAU_EARTH rotation period
    t0 = utc_to_tdb_s("2030-03-01T00:00:00", spice_kernels)
    epochs = np.array(
        [t0, t0 + period_s, t0 + STELLAR_DAY_S, t0 + period_s / 2, t0 + SECONDS_PER_DAY]
    )
    geo = eo.rotation(epochs).matrix @ site.position_ef_m
    # (1) Same J2000 direction after one rotation period. The only change is the IAU pole's
    # linear precession, 0.557 deg/century in declination = 2.7e-7 rad/day; tolerance 1e-6 rad.
    assert _angle(geo[0], geo[1]) < 1e-6
    # The IERS 2010 stellar day differs from the IAU_EARTH period by 3.1e-8 relative
    # (~1.9e-7 rad per day), so it also returns within 1e-6 rad.
    assert _angle(geo[0], geo[2]) < 1e-6
    # (2) Half a period: the equatorial site points the opposite way.
    assert _angle(geo[0], geo[3]) == pytest.approx(math.pi, abs=1e-6)
    # (3) Over one 86400 s day it turns 360.9856235 deg, i.e. 0.9856235 deg beyond a full turn,
    # measured about the pole (R[:, 2]); tolerance 1e-4 deg covers the pole precession
    # (2.7e-7 rad = 1.6e-5 deg). A solar-day (360 deg/day) model would give 0 deg.
    pole = eo.rotation(t0).matrix[0][:, 2]
    a0 = geo[0] - np.dot(geo[0], pole) * pole
    a1 = geo[4] - np.dot(geo[4], pole) * pole
    turn = math.degrees(math.atan2(np.dot(np.cross(a0, a1), pole), np.dot(a0, a1)))
    assert turn == pytest.approx(w_rate_deg_day - 360.0, abs=1e-4)


@pytest.mark.spice
def test_site_inertial_velocity_is_omega_cross_r(spice_kernels: LoadedKernels) -> None:
    eph = SpiceEphemeris(spice_kernels)
    eo = SpiceEarthOrientation(spice_kernels)
    site = GroundSite.from_geodetic("s", math.radians(-23.0), math.radians(120.0), 2000.0)
    t0 = utc_to_tdb_s("2030-07-01T06:00:00", spice_kernels)
    h = 1.0
    st = site_trajectory(site, [t0 - h, t0, t0 + h], ephemeris=eph, earth_orientation=eo)
    fd = (st.position_m[2] - st.position_m[0]) / (2 * h)
    # Central difference of the J2000/SSB position: truncation (third derivative) h^2/6 <=
    # omega^3 r h^2/6 ~ 4e-7 m/s; rounding of positions ~1.5e11 m (ulp 3e-5 m) gives ~1.5e-5
    # m/s. Tolerance 1e-4 m/s, 2e-7 of the 465 m/s rotation speed: a missing omega x r fails.
    np.testing.assert_allclose(st.velocity_m_s[1], fd, rtol=0, atol=1e-4)
    v_rot = st.velocity_m_s[1] - eph.states("EARTH", t0).velocity_m_s[0]
    omega = math.radians(float(_pool("BODY399_PM")[1])) / SECONDS_PER_DAY
    rho = math.hypot(*site.position_ef_m[:2])  # distance from the rotation axis
    # |omega x r| = omega * rho up to the pole-precession part of the angular velocity,
    # d(alpha0)/dt = -0.641 deg/century = -3.5e-12 rad/s = 4.8e-8 of omega: rel 1e-7.
    assert np.linalg.norm(v_rot) == pytest.approx(omega * rho, rel=1e-7)
    assert np.dot(v_rot, st.site_geocentric_m[1]) == pytest.approx(0.0, abs=1e-6 * rho)
    # Earth-fixed coordinates stay separate from the inertial state.
    np.testing.assert_array_equal(st.site.position_ef_m, site.position_ef_m)


@pytest.mark.spice
def test_iau_earth_versus_astropy_itrs_to_gcrs_at_past_epochs(
    spice_kernels: LoadedKernels,
) -> None:
    """Cross-check of the documented IAU_EARTH approximation against astropy's IERS-based
    ITRS -> GCRS (IAU 2006/2000A precession-nutation, UT1 and polar motion from the bundled
    IERS-B table; no network). Measured (holy7c18406, astropy 8.0.2, astropy-iers-data
    0.2026.11.9): equatorial site 6.14 / 7.89 / 8.65 km at 2005-03-01 / 2020-06-15 / 2023-01-01;
    pole direction 4.4e-5 / 3.7e-5 / 3.9e-5 rad; rotation angle about the pole -9.6e-4 /
    -1.24e-3 / -1.36e-3 rad.

    Cause of the rotation-angle offset (pinned by
    test_iau_earth_rotation_angle_offset_is_the_w_definition_against_era): the IAU WGCCRE
    angle ``90 deg + alpha0 + W`` is a linear function of TDB, whereas the Earth's actual
    rotation is the IERS Earth Rotation Angle of UT1. The IAU constants (W0 = 190.147 deg) match
    ERA only for TT - UT1 = 75.06 s at J2000 (the actual 63.8 s gives -0.047 deg = -8.2e-4 rad
    already at J2000), and the IAU rate (W rate plus the -0.641 deg/century node term) is
    6.3e-6 deg/day slower than the ERA rate (-4.04e-5 rad/yr, 0.55 s of rotation per year at
    constant TT - UT1); the growth of TT - UT1 since 2000 (+5.5 s) partly offsets that drift.
    Nutation does not enter the angle about the pole (it moves the pole: the 4e-5 rad above).

    Tolerances. They are empirical: set from the measured values above with margin, and still
    catch gross errors (transposed matrix, wrong frame, time-scale slip: O(1) rad or > 100 km).
    * pole < 1e-4 rad: nutation, omitted by IAU_EARTH, moves the pole by up to
      sqrt((17.2" sin 23.4 deg)^2 + 9.2"^2) = 11.5" = 5.6e-5 rad, plus the IAU linear-precession
      approximation; measured <= 4.4e-5 rad;
    * rotation angle < 2e-3 rad and site < 13 km (= 2e-3 a): measured <= 1.36e-3 rad and
      8.65 km at these past epochs. (The offset keeps growing: ~1.64e-3 rad / 10.5 km in 2030 and
      ~2.04e-3 rad / 13.0 km in 2040 for constant TT - UT1, so these bounds hold only for the
      past epochs tested here.)
    * site velocity: |dv| ~ omega |dr| (a rigid rotation offset), checked to 10 %.
    """
    units = pytest.importorskip("astropy.units")
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils import iers

    eo = SpiceEarthOrientation(spice_kernels)
    site = GroundSite.from_geodetic("eq", 0.0, 0.0, 0.0)
    with iers.conf.set_temp("auto_download", False):
        for utc in ("2005-03-01T00:00:00", "2020-06-15T12:00:00", "2023-01-01T00:00:00"):
            t = Time(utc, scale="utc")
            rot = eo.rotation(utc_to_tdb_s(utc, spice_kernels))
            m, mdot = rot.matrix[0], rot.matrix_rate_per_s[0]
            pos, vel = EarthLocation.from_geocentric(
                *(site.position_ef_m * units.m)
            ).get_gcrs_posvel(t)
            r_ast = pos.xyz.to_value(units.m)
            v_ast = vel.xyz.to_value(units.m / units.s)
            dr = np.linalg.norm(m @ site.position_ef_m - r_ast)
            dv = np.linalg.norm(mdot @ site.position_ef_m - v_ast)
            pole_ast = (
                EarthLocation.from_geocentric(0 * units.m, 0 * units.m, 1e7 * units.m)
                .get_gcrs(t)
                .cartesian.xyz.to_value(units.m)
            )
            pole_err = _angle(m[:, 2], pole_ast)
            x_ast = (
                EarthLocation.from_geocentric(1e7 * units.m, 0 * units.m, 0 * units.m)
                .get_gcrs(t)
                .cartesian.xyz.to_value(units.m)
            )
            p = pole_ast / np.linalg.norm(pole_ast)
            xa = x_ast - np.dot(x_ast, p) * p
            xs = m[:, 0] - np.dot(m[:, 0], p) * p
            rot_err = math.atan2(np.dot(np.cross(xa, xs), p), np.dot(xa, xs))
            assert pole_err < 1e-4, (utc, pole_err)
            assert abs(rot_err) < 2e-3, (utc, rot_err)
            assert 1e3 < dr < 13e3, (utc, dr)  # > 1 km: the approximation is real, not zero
            omega = 2.0 * math.pi / STELLAR_DAY_S  # only to scale the expected offset
            assert dv == pytest.approx(omega * dr, rel=0.1), (utc, dv, dr)


@pytest.mark.spice
def test_iau_earth_rotation_angle_offset_is_the_w_definition_against_era(
    spice_kernels: LoadedKernels,
) -> None:
    """Pins the cause and the future size of the IAU_EARTH rotation-angle offset.

    Reference: ERFA ``c2t06a`` (IAU 2006/2000A celestial-to-terrestrial matrix, SOFA/ERFA),
    polar motion 0 and UT1 = UTC (unpredictable for 2030+; |UT1 - UTC| < 0.9 s by IERS rule,
    i.e. <= 6.6e-5 rad on the "true" offset). TT is taken equal to TDB (< 2 ms, 1.5e-7 rad).

    Closed form of the offset (no fit): the IAU angle of the prime meridian from the J2000
    x-axis, ``90 deg + alpha0 + W = 280.147 deg - 0.641 deg T + 360.9856235 deg d`` (pck00011;
    valid to first order in the pole tilts involved, < 2e-4 rad), minus the IERS Earth Rotation
    Angle
    ``ERA = 2 pi (0.7790572732640 + 1.00273781191135448 D_UT1)`` (IERS Conventions 2010,
    eq. 5.15). Measured agreement with c2t06a: <= 2.4e-8 rad (nutation does not enter the angle
    about the pole); tolerance 2e-7 rad, still 400x below the 8.4e-5 rad equation of the
    equinoxes that would show up if the offset were nutation.

    The offset is about -8e-4 rad at J2000 (W0 corresponds to TT - UT1 = 75.06 s, the actual
    value was 63.8 s) and drifts by the rate deficit of the IAU angle against ERA,
    ``360.9856235 - 0.641/36525 - 360 x 1.00273781191135448 = -6.3e-6 deg/day`` = -4.04e-5
    rad/yr, while TT - UT1 is constant (it is held at 69.18 s here). The 2030 / 2040 values
    (-1.64e-3 / -2.04e-3 rad; equatorial site 10.5 / 13.0 km) are the ones quoted in
    docs/model_assumptions.md (M3), asserted to 3e-5 rad.
    """
    erfa = pytest.importorskip("erfa")
    eo = SpiceEarthOrientation(spice_kernels)
    measured: dict[str, float] = {}
    for utc in (
        "2000-01-01T12:00:00",
        "2020-06-15T12:00:00",
        "2030-01-01T00:00:00",
        "2040-01-01T00:00:00",
    ):
        et = utc_to_tdb_s(utc, spice_kernels)
        et_minus_utc = float(spiceypy.deltet(et, "ET"))
        m = eo.rotation(et).matrix[0]
        rc2t = np.asarray(
            erfa.c2t06a(
                2451545.0,
                et / SECONDS_PER_DAY,
                2451545.0,
                (et - et_minus_utc) / SECONDS_PER_DAY,
                0.0,
                0.0,
            )
        )
        ref = rc2t.T  # terrestrial -> celestial
        p = ref[:, 2]
        xa = ref[:, 0] - np.dot(ref[:, 0], p) * p
        xs = m[:, 0] - np.dot(m[:, 0], p) * p
        offset = math.atan2(float(np.dot(np.cross(xa, xs), p)), float(np.dot(xa, xs)))
        d = et / SECONDS_PER_DAY
        iau = math.radians(280.147 - 0.641 * d / 36525.0 + 360.9856235 * d)
        d_ut1 = (et - et_minus_utc) / SECONDS_PER_DAY
        era = 2.0 * math.pi * (0.7790572732640 + 1.00273781191135448 * d_ut1)
        closed_form = math.remainder(iau - era, 2.0 * math.pi)
        assert offset == pytest.approx(closed_form, abs=2e-7), utc
        measured[utc] = offset
        dr = float(np.linalg.norm((m - ref) @ np.array([A, 0.0, 0.0])))
        if utc.startswith("2030"):
            assert offset == pytest.approx(-1.64e-3, abs=3e-5)
            assert dr == pytest.approx(10.5e3, abs=0.2e3)
        if utc.startswith("2040"):
            assert offset == pytest.approx(-2.04e-3, abs=3e-5)
            assert dr == pytest.approx(13.0e3, abs=0.2e3)
    # Constant offset at J2000: the IAU W0 corresponds to TT - UT1 = 75.06 s, not 63.8 s.
    assert measured["2000-01-01T12:00:00"] == pytest.approx(-8.0e-4, abs=3e-5)
    # Secular drift with TT - UT1 held at 69.18 s (2030-2040): rate deficit of the IAU angle,
    # (360.9856235 - 0.641/36525 - 360 * 1.00273781191135448) deg/day = -6.3e-6 deg/day.
    rate_deg_day = 360.9856235 - 0.641 / 36525.0 - 360.0 * 1.00273781191135448
    expected_drift = (
        math.radians(rate_deg_day)
        * (
            utc_to_tdb_s("2040-01-01T00:00:00", spice_kernels)
            - utc_to_tdb_s("2030-01-01T00:00:00", spice_kernels)
        )
        / SECONDS_PER_DAY
    )
    assert measured["2040-01-01T00:00:00"] - measured["2030-01-01T00:00:00"] == pytest.approx(
        expected_drift, abs=1e-6
    )
    assert expected_drift == pytest.approx(-4.04e-4, abs=0.05e-4)


@pytest.mark.spice
def test_spice_earth_orientation_vectorised_timing(spice_kernels: LoadedKernels) -> None:
    eo = SpiceEarthOrientation(spice_kernels)
    t = np.linspace(9.0e8, 9.0e8 + 3.156e7, 8767)
    t0 = time.perf_counter()
    rot = eo.rotation(t)
    elapsed = time.perf_counter() - t0
    assert rot.matrix.shape == (8767, 3, 3)
    # One sxform per epoch: ~10 us each here; 5 s is a generous ceiling for the hourly year.
    assert elapsed < 5.0
