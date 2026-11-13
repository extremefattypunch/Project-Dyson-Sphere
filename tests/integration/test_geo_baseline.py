"""M7b GEO baseline: geostationary preset, Earth-shadow eclipses and access geometry (SPICE).

Reference values are computed independently of the code under test:

* the IAU_EARTH spin rate from the pck00011 constants read directly with ``spiceypy.bodvrd``
  (``omega = dW/dt + d(alpha_0)/dt sin(delta_0)``, the Euler-angle kinematics of the IAU
  rotation ``[R_z(W) R_x(90 - delta_0) R_z(90 + alpha_0)]^T``) and ``GM_earth`` from
  ``BODY399_GM``;
* the sub-satellite point from ``spiceypy.pxform('J2000', 'IAU_EARTH', t)``;
* the eclipse durations from the closed-form geometry of a circular orbit and the Sun and
  Earth disks (apparent radii ``asin(R/d)``), solved with ``scipy.optimize.brentq``;
* the zenith range from the first-order light-time relation ``c tau = L + (u . v_site) tau``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import spiceypy
from scipy.optimize import brentq

from helioswarm.config import parse_config
from helioswarm.constants import (
    AU_M,
    SECONDS_PER_DAY,
    SPEED_OF_LIGHT_M_S,
    WGS84_SEMIMAJOR_AXIS_M,
)
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.geocentric import (
    geocentric_elements,
    geostationary_orbit,
    geostationary_radius_m,
    propagate_geocentric_point_masses,
    third_body_inclination_rate_vector,
    tilt_from_pole_rad,
)
from helioswarm.ground import GroundSite, SpiceEarthOrientation
from helioswarm.kernels import LoadedKernels
from helioswarm.propagate import (
    initial_state_from_config,
    l0_reference_trajectory,
    satellite_initial_state,
)
from helioswarm.timescale import utc_to_tdb_s
from helioswarm.visibility import (
    OcculterRadii,
    access_settings_from_config,
    compute_access,
    satellite_illumination,
)

from config_helpers import load_mvp_data

EPOCH_UTC = "2030-01-01T00:00:00"


def configured_geo_config(a_m: float, inclination_deg: float) -> Any:
    """The MVP configuration with configured (non-preset) circular Earth-centred elements:
    J2000 equatorial frame, node on +x (RAAN 0), satellite at the node at the epoch."""
    d = load_mvp_data()
    d["orbit"] = {
        "model": "two_body_kepler",
        "initial_epoch_utc": EPOCH_UTC,
        "elements_center": "EARTH",
        "elements_frame": "J2000",
        "semimajor_axis_au": a_m / AU_M,
        "eccentricity": 0.0,
        "inclination_deg": inclination_deg,
        "raan_deg": 0.0,
        "arg_periapsis_deg": 0.0,
        "true_anomaly_deg": 0.0,
    }
    del d["time"]["integrator_step_s"]
    return parse_config(d)


def geo_config(longitude_deg: float = 0.0, **models: Any) -> Any:
    """The MVP configuration with the geostationary preset (L0) and a given longitude."""
    d = load_mvp_data()
    d["orbit"] = {
        "model": "two_body_kepler",
        "initial_epoch_utc": EPOCH_UTC,
        "elements_center": "EARTH",
        "elements_frame": "J2000",
        "geostationary_longitude_deg": longitude_deg,
    }
    del d["time"]["integrator_step_s"]
    d["models"].update(models)
    return parse_config(d)


@pytest.fixture
def spice(spice_kernels: LoadedKernels) -> LoadedKernels:
    """The verified kernel set, furnished for one test."""
    return spice_kernels


def _iau_spin_rate_rad_s(epoch_tdb_s: float) -> float:
    """dW/dt + d(alpha_0)/dt sin(delta_0) from the PCK constants (rad/s, TDB)."""
    _, pm = spiceypy.bodvrd("EARTH", "PM", 3)
    _, ra = spiceypy.bodvrd("EARTH", "POLE_RA", 3)
    _, dec = spiceypy.bodvrd("EARTH", "POLE_DEC", 3)
    t_cy = epoch_tdb_s / (SECONDS_PER_DAY * 36525.0)
    delta0 = math.radians(dec[0] + dec[1] * t_cy)
    deg_per_day = pm[1] + (ra[1] / 36525.0) * math.sin(delta0)
    return math.radians(deg_per_day) / SECONDS_PER_DAY


def _sub_satellite(geo_pos_j2000: np.ndarray, epochs: np.ndarray) -> tuple[np.ndarray, ...]:
    lon, lat, rad = [], [], []
    for p, t in zip(geo_pos_j2000, epochs, strict=True):
        v = np.asarray(spiceypy.pxform("J2000", "IAU_EARTH", float(t))) @ p
        rad.append(float(np.linalg.norm(v)))
        lon.append(math.degrees(math.atan2(v[1], v[0])))
        lat.append(math.degrees(math.asin(v[2] / rad[-1])))
    return np.array(lon), np.array(lat), np.array(rad)


# ---------------------------------------------------------------------------------------------
# Geostationary preset
# ---------------------------------------------------------------------------------------------


def test_geostationary_period_equals_the_iau_earth_spin_period(spice: LoadedKernels) -> None:
    eph = SpiceEphemeris(spice)
    t0 = utc_to_tdb_s(EPOCH_UTC, spice)
    geo = geostationary_orbit(
        0.0, epoch_tdb_s=t0, ephemeris=eph, earth_orientation=SpiceEarthOrientation(spice)
    )
    omega = _iau_spin_rate_rad_s(t0)
    # Tolerance 1e-12 relative: sxform's matrices are exact to ~1e-16; the expected rate uses
    # the same PCK constants. Using dW/dt alone (the W rate without the pole precession term)
    # differs by 4.9e-8 relative, the IERS ERA rate by 6.6e-8: both fail.
    assert geo.spin_rate_rad_s == pytest.approx(omega, rel=1e-12)
    assert geo.period_s == pytest.approx(2.0 * math.pi / omega, rel=1e-12)
    # 86164.1004 s: the IAU stellar day of pck00011 (360.98560595 deg/day).
    assert geo.period_s == pytest.approx(86164.1004, abs=1e-3)
    _, gm_km3 = spiceypy.bodvrd("EARTH", "GM", 1)
    gm = gm_km3[0] * 1e9
    r_expected = (gm / omega**2) ** (1.0 / 3.0)
    assert geo.radius_m == pytest.approx(r_expected, rel=1e-13)
    assert geo.radius_m == pytest.approx(42_164_172.6, abs=1.0)  # documented GEO radius
    # The L0 elements of the preset: circular, period = 2 pi / omega (Kepler's third law).
    cfg = geo_config()
    el = geocentric_elements(
        cfg.orbit,
        initial_epoch_tdb_s=t0,
        ephemeris=eph,
        earth_orientation=SpiceEarthOrientation(spice),
    )
    assert el.center == "EARTH"
    assert el.frame == "J2000"
    assert el.e == 0.0
    assert el.period_s == pytest.approx(2.0 * math.pi / omega, rel=1e-10)
    # Equatorial plane of the epoch: inclination to the J2000 equator = colatitude of the IAU
    # pole, 0.557 deg/century * T (plus the RA term's tilt): ~0.167 deg in 2030.
    pole = geo.pole_unit_j2000
    assert math.degrees(el.i_rad) == pytest.approx(math.degrees(math.acos(pole[2])), abs=1e-9)


def test_sub_satellite_point_stays_fixed_over_ten_days(spice: LoadedKernels) -> None:
    eph = SpiceEphemeris(spice)
    ori = SpiceEarthOrientation(spice)
    t0 = utc_to_tdb_s(EPOCH_UTC, spice)
    for lon0 in (0.0, -75.5, 137.0):
        cfg = geo_config(lon0)
        t = t0 + np.linspace(0.0, 10.0 * SECONDS_PER_DAY, 401)
        traj = l0_reference_trajectory(
            cfg.orbit, t, initial_epoch_tdb_s=t0, ephemeris=eph, earth_orientation=ori
        )
        assert traj.origin == "SOLAR SYSTEM BARYCENTER"
        assert traj.frame == "J2000"
        geo = np.asarray(traj.position_m) - eph.positions("EARTH", t)
        lon, lat, rad = _sub_satellite(geo, t)
        # Longitude: 1e-6 deg (0.7 m at GEO). The IAU frame spins uniformly, so only round-off
        # and the slow pole motion remain (measured < 2e-9 deg). A rate error of 1e-9 relative
        # drifts by 3.6e-6 deg in 10 days; the W rate without the pole term by 1.8e-4 deg.
        dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
        assert np.max(np.abs(dlon)) < 1e-6
        # Latitude: the orbit is fixed in inertial space while the IAU pole precesses
        # (0.641 deg/cy in RA, 0.557 deg/cy in Dec, i.e. <= 0.0056 deg/yr); after 10 days the
        # sub-satellite latitude oscillates by <= 1.5e-4 deg: tolerance 2e-4 deg.
        assert np.max(np.abs(lat)) < 2e-4
        # Radius constant (circular Kepler orbit): round-off only.
        assert np.ptp(rad) < 1e-4


def test_geo_elevation_and_range_at_the_sub_satellite_site(spice: LoadedKernels) -> None:
    cfg = geo_config(0.0)
    eph = SpiceEphemeris(spice)
    ori = SpiceEarthOrientation(spice)
    t0 = utc_to_tdb_s(EPOCH_UTC, spice)
    t = t0 + np.linspace(0.0, SECONDS_PER_DAY, 97)
    traj = l0_reference_trajectory(
        cfg.orbit, t, initial_epoch_tdb_s=t0, ephemeris=eph, earth_orientation=ori
    )
    site = GroundSite.from_config(cfg.receiver.sites[0])  # 0 N, 0 E, h = 0
    acc = compute_access(
        traj,
        site,
        ephemeris=eph,
        earth_orientation=ori,
        settings=access_settings_from_config(cfg, OcculterRadii.from_kernels(spice)),
    )
    assert np.all(acc.visible)
    # Apparent elevation: 90 deg minus omega * tau. In the non-rotating geocentric frame the
    # photon leaves the satellite at t_e and reaches the site at t_a = t_e + tau, by which time
    # the Earth (and the site's local vertical) has turned by omega tau; the geometric
    # direction from the site to the satellite's emission point is therefore tilted west by
    # omega tau r / L, and the aberration of the site's rotational velocity (omega R_E / c =
    # omega tau R_E / L) tilts it back east, leaving omega tau (r - R_E) / L = omega tau
    # (planetary aberration of the relative velocity (v_sat - v_site) tau / L) ~ 8.7e-6 rad.
    # Remaining terms: the pole motion over one day (< 2e-7 rad, perpendicular, so second order
    # here) and (v/c)^2 ~ 1e-8: tolerance 5e-8 rad. A test of "exactly 90 deg" would miss the
    # retarded-time physics; an instantaneous-geometry model gives 0 and fails.
    omega = _iau_spin_rate_rad_s(t0)
    deficit = 0.5 * math.pi - acc.elevation_rad
    assert np.max(np.abs(deficit - omega * acc.light_time_s)) < 5e-8
    # ... and the tilt is towards the west (azimuth 270 deg; +-2 deg for the N-S component).
    assert np.max(np.abs(np.degrees(acc.azimuth_rad) - 270.0)) < 2.0
    # Range: the geometric zenith distance at emission is L = r_GEO - a_WGS84 (site and
    # satellite on one radius). The barycentric light time solves c tau = |L_vec + v_site tau|
    # = L + (u . v_site) tau to first order (u = unit vector satellite -> site), so
    # range = c tau = L / (1 - u . v_site / c): the Earth's orbital velocity changes it by up
    # to +-3.6 km over the day. Second-order terms (v tau)^2 / (2 L) <= 0.2 m: tolerance 1 m.
    r_geo = geostationary_radius_m(
        eph.gm_m3_s2("EARTH"),
        _iau_spin_rate_rad_s(t0),
    )
    big_l = r_geo - WGS84_SEMIMAJOR_AXIS_M
    st = acc.light_time.site_at_arrival
    u = -np.asarray(st.up_unit)
    v_site = np.asarray(st.velocity_m_s)
    expected = big_l / (1.0 - np.einsum("ij,ij->i", u, v_site) / SPEED_OF_LIGHT_M_S)
    assert np.max(np.abs(acc.range_m - expected)) < 1.0
    assert np.ptp(acc.range_m) > 5.0e3  # the first-order term is resolved, not negligible
    assert np.mean(acc.range_m) == pytest.approx(35_786_035.0, abs=4.0e3)


def test_configured_geocentric_elements_follow_kepler_about_the_earth(
    spice: LoadedKernels,
) -> None:
    """A configured (non-preset) Earth-centred orbit: J2000 equatorial elements, mu = GM_earth.

    Circular orbit, a = 42 164 172.6 m, i = 30 deg to the J2000 equator, node on +x, satellite
    at the node at the epoch. Closed forms: position (a, 0, 0) at the epoch, (0, a cos i, a sin i)
    a quarter period later, back at (a, 0, 0) after one period T = 2 pi sqrt(a^3 / GM_earth)
    with GM_earth = BODY399_GM read directly with bodvrd. Tolerance 1e-3 m (Kepler solver and
    origin shift in double precision: ~1e-8 m relative round-off at 4e7 m); using GM_sun (the
    heliocentric constant) would put the satellite ~1.8e6 m/s off and fail by > 1e10 m.
    """
    a = 42_164_172.6
    inc = math.radians(30.0)
    cfg = configured_geo_config(a, 30.0)
    eph = SpiceEphemeris(spice)
    t0 = utc_to_tdb_s(EPOCH_UTC, spice)
    _, gm_km3 = spiceypy.bodvrd("EARTH", "GM", 1)
    period = 2.0 * math.pi * math.sqrt(a**3 / (gm_km3[0] * 1e9))
    t = t0 + np.array([0.0, 0.25 * period, 0.5 * period, period])
    traj = l0_reference_trajectory(cfg.orbit, t, initial_epoch_tdb_s=t0, ephemeris=eph)
    assert traj.frame == "J2000"
    assert traj.origin == "SOLAR SYSTEM BARYCENTER"
    geo = np.asarray(traj.position_m) - eph.positions("EARTH", t)
    expected = a * np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(inc), math.sin(inc)], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(geo, expected, rtol=0.0, atol=1e-3)
    # Circular speed sqrt(GM/a) along the orbit normal x position (prograde, inclined plane).
    v_geo = np.asarray(traj.velocity_m_s) - np.asarray(eph.states("EARTH", t).velocity_m_s)
    speed = math.sqrt(gm_km3[0] * 1e9 / a)
    np.testing.assert_allclose(
        v_geo[0], speed * np.array([0.0, math.cos(inc), math.sin(inc)]), rtol=0.0, atol=1e-9
    )
    # The heliocentric initial-state builder refuses Earth-centred orbits (it would use GM_sun
    # and a Sun-relative collector normal: M7b review).
    with pytest.raises(ValueError, match="heliocentric states only"):
        satellite_initial_state(cfg.orbit, cfg.constants, epoch_tdb_s=t0, ephemeris=eph)
    with pytest.raises(ValueError, match="heliocentric states only"):
        initial_state_from_config(cfg, eph, spice)


def test_geo_elevation_and_range_from_an_off_nadir_site(spice: LoadedKernels) -> None:
    """GEO satellite at 0 deg E seen from an equatorial site at 40 deg E (central angle 40 deg).

    Spherical geometry in the plane of the equator (equatorial site: the ellipsoid normal is
    radial and the site radius is a_WGS84 = R): elevation ``atan2(cos g - R/r, sin g)`` and
    slant range ``L = sqrt(r^2 + R^2 - 2 r R cos g)``, azimuth due west. The run adds the
    retarded-time effects: the elevation is lowered by the Earth's rotation during the light
    time (omega tau r/L <= 1.3e-5 rad) and changed by aberration (<= omega R/c = 1.6e-6 rad):
    tolerance 3e-5 rad (a satellite 0.01 deg off its longitude changes the elevation by
    1.6e-4 rad and fails). The range is ``L/(1 - u . v_site/c)`` to first order in v/c (u from
    the satellite to the site), as at the sub-satellite site: tolerance 1 m (second-order
    terms <= 0.2 m); the bare geometric L would err by up to 3.6 km.
    """
    cfg = geo_config(0.0)
    eph = SpiceEphemeris(spice)
    ori = SpiceEarthOrientation(spice)
    t0 = utc_to_tdb_s(EPOCH_UTC, spice)
    t = t0 + np.linspace(0.0, SECONDS_PER_DAY, 49)
    traj = l0_reference_trajectory(
        cfg.orbit, t, initial_epoch_tdb_s=t0, ephemeris=eph, earth_orientation=ori
    )
    gamma = math.radians(40.0)
    site = GroundSite.from_geodetic("off_nadir", 0.0, gamma, 0.0)
    acc = compute_access(
        traj,
        site,
        ephemeris=eph,
        earth_orientation=ori,
        settings=access_settings_from_config(cfg, OcculterRadii.from_kernels(spice)),
    )
    assert np.all(acc.visible)
    r = geostationary_radius_m(eph.gm_m3_s2("EARTH"), _iau_spin_rate_rad_s(t0))
    big_r = WGS84_SEMIMAJOR_AXIS_M
    el_expected = math.atan2(math.cos(gamma) - big_r / r, math.sin(gamma))
    assert math.degrees(el_expected) == pytest.approx(43.73, abs=0.01)  # sanity of the formula
    assert np.max(np.abs(acc.elevation_rad - el_expected)) < 3e-5
    assert np.max(np.abs(np.degrees(acc.azimuth_rad) - 270.0)) < 0.01
    big_l = math.sqrt(r * r + big_r * big_r - 2.0 * r * big_r * math.cos(gamma))
    # u: unit vector satellite -> site, from the Earth-fixed geometry (pxform), in J2000 at the
    # arrival epoch; v_site: the site's barycentric velocity (Earth orbit + rotation).
    sat_fixed = np.array([r, 0.0, 0.0])
    site_fixed = np.array([big_r * math.cos(gamma), big_r * math.sin(gamma), 0.0])
    u_fixed = (site_fixed - sat_fixed) / big_l
    st = acc.light_time.site_at_arrival
    t_arr = np.asarray(acc.light_time.t_arrival_tdb_s)
    u = np.array([np.asarray(spiceypy.pxform("IAU_EARTH", "J2000", float(ta))) @ u_fixed
                  for ta in t_arr])  # fmt: skip
    v_site = np.asarray(st.velocity_m_s)
    expected = big_l / (1.0 - np.einsum("ij,ij->i", u, v_site) / SPEED_OF_LIGHT_M_S)
    assert np.max(np.abs(acc.range_m - expected)) < 1.0
    assert np.max(np.abs(acc.range_m - big_l)) > 1.0e3  # the light-time term is resolved


# ---------------------------------------------------------------------------------------------
# Earth-shadow eclipses (conical shadow with penumbra)
# ---------------------------------------------------------------------------------------------


def _analytic_shadow_durations_s(
    sun_geo_m: np.ndarray,
    pole: np.ndarray,
    r_sat_m: float,
    earth_radius_m: float,
    sun_radius_m: float,
    relative_rate_rad_s: float,
) -> tuple[float, float]:
    """(umbra, umbra + penumbra) durations of a circular orbit's passage through the shadow.

    Satellite at ``r (cos phi e1 + sin phi e2)`` with ``e1`` the anti-Sun direction projected on
    the orbit plane; ``beta`` = angle of the Sun above the plane, ``D`` = Sun distance. Seen
    from the satellite the angle between the Earth's and the Sun's centres satisfies
    ``cos theta = (r + D cos beta cos phi) / sqrt(D^2 + r^2 + 2 D r cos beta cos phi)``; the
    Sun is fully hidden for ``theta <= rho_E - rho_S`` and partly for ``theta < rho_E + rho_S``
    (``rho = asin(R/d)``). The half-widths ``phi*`` are found with brentq; durations are
    ``2 phi* / (relative angular rate)``.
    """
    d = float(np.linalg.norm(sun_geo_m))
    s = sun_geo_m / d
    beta = math.asin(float(np.dot(s, pole)))
    rho_e = math.asin(earth_radius_m / r_sat_m)

    def theta(phi: float) -> float:
        k = d * math.cos(beta) * math.cos(phi)
        return math.acos((r_sat_m + k) / math.sqrt(d * d + r_sat_m * r_sat_m + 2.0 * r_sat_m * k))

    def rho_s(phi: float) -> float:
        k = d * math.cos(beta) * math.cos(phi)
        return math.asin(sun_radius_m / math.sqrt(d * d + r_sat_m * r_sat_m + 2.0 * r_sat_m * k))

    out = []
    for sign in (-1.0, 1.0):

        def f(phi: float, sign: float = sign) -> float:
            return theta(phi) - (rho_e + sign * rho_s(phi))

        if f(0.0) >= 0.0:
            out.append(0.0)
            continue
        phi_star = brentq(f, 0.0, 0.5 * math.pi, xtol=1e-14)
        out.append(2.0 * phi_star / relative_rate_rad_s)
    return out[0], out[1]


@pytest.mark.parametrize(
    ("day_utc", "eclipse_expected"),
    [
        ("2030-03-20", True),  # March equinox (2030-03-20 13:52 UT): longest eclipses
        ("2030-04-04", True),  # ~15 days later, Sun ~6 deg above the equator: shorter
        ("2030-04-25", False),  # Sun > 9 deg above the equator: outside the season
        ("2030-09-23", True),  # September equinox
    ],
)
def test_geo_eclipse_durations_match_the_analytic_shadow_geometry(
    spice: LoadedKernels, day_utc: str, eclipse_expected: bool
) -> None:
    cfg = geo_config(0.0)
    eph = SpiceEphemeris(spice)
    ori = SpiceEarthOrientation(spice)
    t0 = utc_to_tdb_s(EPOCH_UTC, spice)
    radii = OcculterRadii.from_kernels(spice)
    # Local midnight at 0 deg E is ~00:00 UT (+- equation of time); sample +-80 min at 5 s.
    t_mid = utc_to_tdb_s(f"{day_utc}T00:00:00", spice)
    step = 5.0
    t = t_mid + np.arange(-80 * 60, 80 * 60 + step, step)
    traj = l0_reference_trajectory(
        cfg.orbit, t, initial_epoch_tdb_s=t0, ephemeris=eph, earth_orientation=ori
    )
    ill = satellite_illumination(traj, ephemeris=eph, occulters=("EARTH",), radii=radii)
    frac = ill.illumination_fraction
    umbra_s = float(np.count_nonzero(frac <= 0.0)) * step
    shadow_s = float(np.count_nonzero(frac < 1.0)) * step

    geo = geostationary_orbit(0.0, epoch_tdb_s=t0, ephemeris=eph, earth_orientation=ori)
    pole = geo.pole_unit_j2000

    # Sun (geocentric) at the shadow centre; the anti-Sun azimuth in the orbit plane turns at
    # the Sun's apparent rate (~0.9 deg/day), measured from SPICE over +-1 h.
    def sun_geo(tt: float) -> np.ndarray:
        return eph.positions("SUN", np.array([tt]))[0] - eph.positions("EARTH", np.array([tt]))[0]

    s0 = sun_geo(t_mid)
    s_m, s_p = sun_geo(t_mid - 3600.0), sun_geo(t_mid + 3600.0)

    def azimuth(v: np.ndarray) -> float:
        e1 = np.cross(pole, [0.0, 0.0, 1.0]) if abs(pole[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = e1 - np.dot(e1, pole) * pole
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(pole, e1)
        return math.atan2(float(np.dot(v, e2)), float(np.dot(v, e1)))

    sun_rate = (azimuth(s_p) - azimuth(s_m)) / 7200.0
    rel_rate = geo.spin_rate_rad_s - sun_rate
    umbra_ref, shadow_ref = _analytic_shadow_durations_s(
        s0,
        pole,
        geo.radius_m,
        radii.earth_equatorial_radius_m,
        radii.sun_radius_m,
        rel_rate,
    )
    if not eclipse_expected:
        assert shadow_ref == 0.0
        assert shadow_s == 0.0
        return
    # The window must contain the whole shadow passage.
    assert frac[0] == 1.0
    assert frac[-1] == 1.0
    # Tolerance 15 s on durations of 3000-4300 s: each edge is resolved to one 5 s step
    # (<= 10 s per duration); the light-time retardation of the Sun (499 s) and of the Earth
    # (0.14 s) shift both edges together (< 2 s on the duration), the Sun's direction changes by
    # < 0.06 deg during the passage. A cylindrical shadow (no cone) would give an umbra of
    # 2 asin(R_E/r)/omega = 4168 s at equinox instead of ~4040 s, and a model without the
    # penumbra equal umbra and total durations (difference ~255 s): both fail.
    assert umbra_s == pytest.approx(umbra_ref, abs=15.0)
    assert shadow_s == pytest.approx(shadow_ref, abs=15.0)
    assert shadow_s - umbra_s > 200.0  # the penumbra is resolved (~2 x 2.2 min)
    if day_utc.endswith("03-20") or day_utc.endswith("09-23"):
        # Equinox: up to ~72 min including the penumbra, ~67 min of umbra.
        assert 70.0 * 60 < shadow_s < 72.5 * 60
        assert 66.0 * 60 < umbra_s < 68.5 * 60


@pytest.mark.slow
def test_geo_eclipse_seasons_over_2030_match_the_analytic_geometry(spice: LoadedKernels) -> None:
    """Season boundaries and yearly shadow fractions of the GEO preset over 2030 (Earth shadow).

    Reference: for every day the closed-form shadow durations of
    :func:`_analytic_shadow_durations_s` with the geocentric Sun at 00:00 UT (local midnight at
    0 deg E +- the equation of time) and the relative rate omega - n_sun (mean solar motion);
    a day has shadow iff ``theta(0) < rho_E + rho_S`` (the condition beta < ~rho_E + rho_S).
    Simulation: the run's conical-shadow model on a 60 s grid over the year (Earth only: the
    analytic geometry has no Moon; the runs' lunar penumbra is reported separately in M7b.md).

    * Season boundaries: the first and last shadow day of each season must be the analytic ones
      (2030-02-26 .. 04-12 and 08-31 .. 10-16; the boundary days have >= 800 s of penumbra, i.e.
      >= 13 samples, and the day before/after none).
    * Shadow days 93 in both.
    * Umbra and penumbra time fractions: the 60 s count of an interval of length T has a
      sampling variance <= 1/4 sample^2 per edge pair, so over 93 umbra passages (186 penumbra
      intervals) the standard deviation is <= 4.8 (6.8) samples = 9e-6 (1.3e-5) of the year;
      the analytic reference's simplifications (Sun direction fixed at 00:00 UT, mean solar
      rate, unretarded Sun) shift each passage by <= ~2 s (< 1e-5 of the year in total).
      Tolerance 4e-5 absolute (umbra fraction 0.88 %; a cylindrical shadow adds ~128 s per
      passage = 3.8e-4 and fails, as does a model without penumbra).
    """
    cfg = geo_config(0.0)
    eph = SpiceEphemeris(spice)
    ori = SpiceEarthOrientation(spice)
    radii = OcculterRadii.from_kernels(spice)
    t0 = utc_to_tdb_s(EPOCH_UTC, spice)
    year = 365.25 * SECONDS_PER_DAY
    step = 60.0
    t = t0 + np.arange(0.0, year + 0.5 * step, step)
    traj = l0_reference_trajectory(
        cfg.orbit, t, initial_epoch_tdb_s=t0, ephemeris=eph, earth_orientation=ori
    )
    frac = satellite_illumination(
        traj, ephemeris=eph, occulters=("EARTH",), radii=radii
    ).illumination_fraction
    # Simulated shadow days: shadow epochs grouped by the nearest UT midnight.
    day_of = np.floor((t - t0) / SECONDS_PER_DAY + 0.5).astype(int)
    sim_days = sorted({int(d) for d in day_of[frac < 1.0]})
    umbra_sim = float(np.count_nonzero(frac <= 0.0)) / t.size
    pen_sim = float(np.count_nonzero((frac > 0.0) & (frac < 1.0))) / t.size

    geo = geostationary_orbit(0.0, epoch_tdb_s=t0, ephemeris=eph, earth_orientation=ori)
    rel_rate = geo.spin_rate_rad_s - 2.0 * math.pi / (365.2422 * SECONDS_PER_DAY)
    umbra_ref = shadow_ref = 0.0
    ref_days: list[int] = []
    for d in range(366):
        tm = np.array([t0 + d * SECONDS_PER_DAY])
        sun = eph.positions("SUN", tm)[0] - eph.positions("EARTH", tm)[0]
        u, s = _analytic_shadow_durations_s(
            sun,
            geo.pole_unit_j2000,
            geo.radius_m,
            radii.earth_equatorial_radius_m,
            radii.sun_radius_m,
            rel_rate,
        )
        umbra_ref += u
        shadow_ref += s
        if s > 0.0:
            ref_days.append(d)

    def seasons(days: list[int]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for d in days:
            if out and d == out[-1][1] + 1:
                out[-1] = (out[-1][0], d)
            else:
                out.append((d, d))
        return out

    # Day numbers from 2030-01-01: 56 = Feb 26, 101 = Apr 12, 242 = Aug 31, 288 = Oct 16.
    assert seasons(ref_days) == [(56, 101), (242, 288)]
    assert seasons(sim_days) == seasons(ref_days)
    assert len(sim_days) == len(ref_days) == 93
    assert umbra_sim == pytest.approx(umbra_ref / year, abs=4e-5)
    assert pen_sim == pytest.approx((shadow_ref - umbra_ref) / year, abs=4e-5)
    assert 0.0087 < umbra_sim < 0.0089  # documented: 0.881 % of 2030 (M7b.md)


# ---------------------------------------------------------------------------------------------
# Perturbation budget (comparison path of the omitted lunisolar perturbation)
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow  # one-year Earth+Sun+Moon DOP853 integration (~7 s)
def test_lunisolar_inclination_drift_matches_the_averaged_torque(spice: LoadedKernels) -> None:
    """Numerical Earth+Sun+Moon integration vs the orbit-averaged secular estimate, 2030.

    The secular rate is the time average over the year of ``-(3 GM_k / (2 n |s_k|^5))
    (h . s_k)(h x s_k)`` for the Sun and the Moon (SPICE positions). After one year the
    numerically integrated tilt of the orbit normal from the initial pole must equal that rate
    times one year to 5 %: the year is not a whole number of lunar months (13.37) and the
    secular theory neglects the change of h during the year (tilt <= 1 deg). The rate must lie
    in 0.75-0.95 deg/yr: the extremes of the sum of the solar 0.27 deg/yr and the lunar
    0.47-0.67 deg/yr ((3/8)(GM/(n a^3)) sin 2i with the Moon's 18.3-28.6 deg inclination to the
    equator) - the size of the omitted perturbation that station-keeping removes.
    """
    eph = SpiceEphemeris(spice)
    ori = SpiceEarthOrientation(spice)
    t0 = utc_to_tdb_s(EPOCH_UTC, spice)
    geo = geostationary_orbit(0.0, epoch_tdb_s=t0, ephemeris=eph, earth_orientation=ori)
    year = 365.25 * SECONDS_PER_DAY
    ts = t0 + np.linspace(0.0, year, 20000)
    earth = eph.positions("EARTH", ts)
    rate = np.zeros(3)
    for body in ("SUN", "MOON"):
        s = eph.positions(body, ts) - earth
        rate += np.mean(
            third_body_inclination_rate_vector(
                geo.pole_unit_j2000, geo.spin_rate_rad_s, s, eph.gm_m3_s2(body)
            ),
            axis=0,
        )
    averaged = float(np.linalg.norm(rate)) * year
    assert 0.75 < math.degrees(averaged) < 0.95
    t = t0 + np.linspace(0.0, year, 120)
    prop = propagate_geocentric_point_masses(geo.initial_state, t, ephemeris=eph)
    tilt = tilt_from_pole_rad(prop.trajectory, geo.pole_unit_j2000)
    assert tilt[-1] == pytest.approx(averaged, rel=0.05)
    assert np.all(np.diff(tilt[::10]) > 0.0)  # secular growth, not an oscillation
    # Without third bodies the tilt stays at round-off level (the integrator conserves h).
    two_body = propagate_geocentric_point_masses(geo.initial_state, t[:5], ephemeris=eph, bodies=())
    assert np.max(tilt_from_pole_rad(two_body.trajectory, geo.pole_unit_j2000)) < 1e-12
