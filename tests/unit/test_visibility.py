"""Access geometry on synthetic (hand-built) geometries with closed-form answers.

Bodies move uniformly (``geometry_helpers.LinearMotionEphemeris``) and the Earth either does not
rotate or rotates uniformly about the J2000 z-axis (``UniformEarthRotation``), so light times,
directions, elevations and occultation distances have exact closed forms. No kernels needed.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import spiceypy
from scipy.integrate import quad

from helioswarm.config import parse_config
from helioswarm.constants import (
    AU_M,
    MOON_RADIUS_M,
    SOLAR_RADIUS_NOMINAL_M,
    SPEED_OF_LIGHT_M_S,
    WGS84_SEMIMAJOR_AXIS_M,
    WGS84_SEMIMINOR_AXIS_M,
)
from helioswarm.ephemeris import SSB, AnalyticEphemeris
from helioswarm.frames import CartesianTrajectory, FrameMismatchError, transform_trajectory
from helioswarm.ground import GroundSite, UniformEarthRotation
from helioswarm.orbit import OrbitalElements, propagate_kepler
from helioswarm.visibility import (
    AccessResult,
    AccessSettings,
    LightTimeError,
    LightTimeSettings,
    OcculterRadii,
    aberrated_direction,
    access_settings_from_config,
    angle_between_rad,
    circle_overlap_area,
    compute_access,
    max_circular_gap_rad,
    orbit_drift_m,
    satellite_hour_angle_rad,
    satellite_illumination,
    segment_intersects_ellipsoid,
    segment_min_distance,
    solar_disk_occulted_fraction,
    solve_light_time,
    summarize_access,
)

from config_helpers import set_path
from geometry_helpers import LinearMotionEphemeris, linear_trajectory

C = SPEED_OF_LIGHT_M_S
A = WGS84_SEMIMAJOR_AXIS_M
B = WGS84_SEMIMINOR_AXIS_M
R_SUN = SOLAR_RADIUS_NOMINAL_M
NO_ROTATION = UniformEarthRotation()
FAR = (0.0, 0.0, 9.0e13)  # a body parked far from every test geometry
ZERO = (0.0, 0.0, 0.0)


def _settings(**kw: object) -> AccessSettings:
    return dataclasses.replace(AccessSettings(), **kw)  # type: ignore[arg-type]


def _ephemeris(
    earth: tuple[float, float, float] = ZERO,
    earth_v: tuple[float, float, float] = ZERO,
    sun: tuple[float, float, float] = (-AU_M, 0.0, 0.0),
    moon: tuple[float, float, float] = FAR,
    moon_v: tuple[float, float, float] = ZERO,
) -> LinearMotionEphemeris:
    return LinearMotionEphemeris(
        {"EARTH": (earth, earth_v), "SUN": (sun, ZERO), "MOON": (moon, moon_v)}
    )


def _access(
    sat: CartesianTrajectory,
    site: GroundSite,
    eph: LinearMotionEphemeris,
    settings: AccessSettings | None = None,
    rotation: UniformEarthRotation = NO_ROTATION,
) -> AccessResult:
    return compute_access(
        sat,
        site,
        ephemeris=eph,
        earth_orientation=rotation,
        settings=settings if settings is not None else AccessSettings(),
    )


EQ_SITE = GroundSite.from_geodetic("eq", 0.0, 0.0, 0.0)  # at (a, 0, 0), up = +x


# ---------------------------------------------------------------------------------------------
# Light time
# ---------------------------------------------------------------------------------------------


def test_static_geometry_light_time_is_distance_over_c() -> None:
    eph = _ephemeris()
    d = 1.234e11
    direction = np.array([0.6, 0.0, 0.8])
    sat = linear_trajectory([0.0, 1.0e8, 9.5e8], EQ_SITE.position_ef_m + d * direction)
    lt = solve_light_time(
        sat, EQ_SITE, ephemeris=eph, earth_orientation=NO_ROTATION, settings=LightTimeSettings()
    )
    # Exact up to the rounding of the positions (~1e-5 m at 1e11 m): rel 1e-14.
    np.testing.assert_allclose(lt.light_time_s, d / C, rtol=1e-14)
    np.testing.assert_allclose(lt.t_arrival_tdb_s, sat.epochs_tdb_s + d / C, rtol=0, atol=2e-7)
    # tau_0 is already exact; the first iteration confirms it.
    assert np.all(lt.iterations == 1)
    assert np.all(lt.method == "fixed_point")
    assert np.all(lt.converged)
    assert np.max(lt.residual_m) < 1e-4


def _uniform_motion_tau(d: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Closed form of c tau = |d + v tau| (site at r_site(t_e) + v tau, d = r_site(t_e) - r_sat):
    (c^2 - v^2) tau^2 - 2 (d.v) tau - d^2 = 0, positive root."""
    dv = np.einsum("ij,j->i", d, v)
    dd = np.einsum("ij,ij->i", d, d)
    vv = float(v @ v)
    return (dv + np.sqrt(dv * dv + (C * C - vv) * dd)) / (C * C - vv)


@pytest.mark.parametrize("tolerance_s", [1e-9, 1e-12])
def test_moving_site_light_time_matches_the_quadratic(tolerance_s: float) -> None:
    v = np.array([2.9e4, -1.1e4, 4.0e3])  # ~Earth orbital speed, arbitrary direction
    eph = _ephemeris(earth=(1.0e9, -2.0e9, 3.0e8), earth_v=tuple(v))  # type: ignore[arg-type]
    t = np.array([0.0, 3.6e3, 4.0e8, 9.5e8])
    sat = linear_trajectory(t, (-8.0e10, 5.0e10, 2.0e10), (1.0e3, 2.0e4, -3.0e3))
    lt = solve_light_time(
        sat,
        EQ_SITE,
        ephemeris=eph,
        earth_orientation=NO_ROTATION,
        settings=LightTimeSettings(tolerance_s=tolerance_s),
    )
    d = (np.asarray(eph.positions("EARTH", t)) + EQ_SITE.position_ef_m) - sat.position_m
    exact = _uniform_motion_tau(d, v)
    # The iteration stops when the step is < tolerance; the remaining error is the step times
    # v/c / (1 - v/c) ~ 1e-4 * tolerance, plus the representation of t_a ~ 1e9 s (6e-8 s ->
    # 2 mm of site motion -> 7e-12 s of light time). Tolerance: tolerance_s + 2e-11 s.
    np.testing.assert_allclose(lt.light_time_s, exact, rtol=0, atol=tolerance_s + 2e-11)
    # Contraction factor ~ |v|/c = 1e-4: 3-5 iterations.
    assert np.all(lt.iterations <= 5)
    # The instantaneous distance is wrong by ~ (d.v/|d|) tau / c ~ 0.01 s here, so a solver
    # that skipped the iteration would fail.
    assert np.max(np.abs(np.linalg.norm(d, axis=1) / C - exact)) > 1e-3


def test_fast_receding_site_uses_the_bracketed_fallback() -> None:
    v = np.array([0.9 * C, 0.0, 0.0])  # contraction factor 0.9
    eph = _ephemeris(earth_v=tuple(v))  # type: ignore[arg-type]
    t = np.array([0.0, 10.0])
    sat = linear_trajectory(t, (-1.0e9, 0.0, 0.0))
    d = (np.asarray(eph.positions("EARTH", t)) + EQ_SITE.position_ef_m) - sat.position_m
    exact = _uniform_motion_tau(d, v)
    capped = solve_light_time(
        sat,
        EQ_SITE,
        ephemeris=eph,
        earth_orientation=NO_ROTATION,
        settings=LightTimeSettings(tolerance_s=1e-9, max_iterations=5),
    )
    assert np.all(capped.method == "brentq")
    assert np.all(capped.iterations == 5)
    # brentq xtol = 1e-9 s; tau = d/(c - v) ~ 33 s, so add 4 eps * tau ~ 3e-14 s.
    np.testing.assert_allclose(capped.light_time_s, exact, rtol=0, atol=2e-9)
    free = solve_light_time(
        sat,
        EQ_SITE,
        ephemeris=eph,
        earth_orientation=NO_ROTATION,
        settings=LightTimeSettings(tolerance_s=1e-9, max_iterations=1000),
    )
    assert np.all(free.method == "fixed_point")
    assert np.all(free.iterations > 100)  # 0.9^k convergence
    # Error after stopping: step * 0.9/(1 - 0.9) = 9 * tolerance.
    np.testing.assert_allclose(free.light_time_s, exact, rtol=0, atol=1e-8)


def test_superluminal_site_raises() -> None:
    eph = _ephemeris(earth_v=(1.1 * C, 0.0, 0.0))
    sat = linear_trajectory([0.0], (-1.0e9, 0.0, 0.0))
    with pytest.raises(LightTimeError, match="bracket"):
        solve_light_time(
            sat,
            EQ_SITE,
            ephemeris=eph,
            earth_orientation=NO_ROTATION,
            settings=LightTimeSettings(max_iterations=3),
        )


def test_instantaneous_mode_and_target_motion_point_ahead() -> None:
    # Site moving perpendicular to the line of sight: the retarded direction leads the
    # instantaneous one by exactly asin(v/c) (c tau = |d + v tau| with d . v = 0).
    v = 3.0e4
    eph = _ephemeris(earth_v=(0.0, v, 0.0))
    sat = linear_trajectory([0.0, 100.0], (A + 1.5e11, 0.0, 0.0))
    retarded = _access(sat, EQ_SITE, eph, _settings(moon_occultation=False))
    assert retarded.target_motion_point_ahead_rad == pytest.approx(math.asin(v / C), rel=1e-9)
    inst = _access(
        sat,
        EQ_SITE,
        eph,
        _settings(moon_occultation=False, light_time=LightTimeSettings(enabled=False)),
    )
    np.testing.assert_array_equal(inst.t_arrival_tdb_s, inst.t_emit_tdb_s)
    np.testing.assert_allclose(inst.light_time_s, inst.range_m / C, rtol=1e-15)
    assert np.all(inst.light_time.method == "instantaneous")
    np.testing.assert_array_equal(inst.light_time_rate, 0.0)
    np.testing.assert_allclose(inst.target_motion_point_ahead_rad, 0.0, atol=1e-15)
    # The instantaneous approximation points asin(v/c) = 1e-4 rad away from the retarded aim.
    off = angle_between_rad(inst.transmit_direction_unit, retarded.transmit_direction_unit)
    np.testing.assert_allclose(off, math.asin(v / C), rtol=1e-9)


# ---------------------------------------------------------------------------------------------
# Receiver geometry
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "el_deg", "az_deg"),
    [("up", 90.0, None), ("east", 0.0, 90.0), ("north", 0.0, 0.0), ("ne30", 30.0, 45.0)],
)
def test_elevation_and_azimuth_of_placed_satellites(
    direction: str, el_deg: float, az_deg: float | None
) -> None:
    site = GroundSite.from_geodetic("s", math.radians(40.0), math.radians(20.0), 150.0)
    e, n, u = site.east_ef, site.north_ef, site.up_ef
    vec = {
        "up": u,
        "east": e,
        "north": n,
        "ne30": math.cos(math.radians(30)) * (e + n) / math.sqrt(2)
        + math.sin(math.radians(30)) * u,
    }[direction]
    sat = linear_trajectory([0.0], site.position_ef_m + 1.0e11 * vec)
    acc = _access(sat, site, _ephemeris(), _settings(moon_occultation=False))
    # Static geometry: exact up to rounding of 1e11 m positions (1e-16 relative): 1e-12 rad.
    assert math.degrees(acc.elevation_rad[0]) == pytest.approx(el_deg, abs=1e-10)
    if az_deg is not None:
        assert math.degrees(acc.azimuth_rad[0]) == pytest.approx(az_deg, abs=1e-10)
    if el_deg > 0.0:
        assert not bool(acc.below_min_elevation[0])


def test_min_elevation_flag() -> None:
    site = EQ_SITE
    vec = math.cos(math.radians(9.0)) * site.east_ef + math.sin(math.radians(9.0)) * site.up_ef
    sat = linear_trajectory([0.0], site.position_ef_m + 1.0e11 * vec)
    acc = _access(sat, site, _ephemeris(), _settings(moon_occultation=False))
    assert bool(acc.visible[0])
    acc10 = _access(
        sat,
        site,
        _ephemeris(),
        _settings(moon_occultation=False, min_elevation_rad=math.radians(10)),
    )
    assert bool(acc10.below_min_elevation[0])
    assert not bool(acc10.visible[0])
    assert acc10.occulted_by[0] == ""


def test_sun_separation_and_avoidance() -> None:
    # Sun at the origin; site on the sunward side (lon 180 deg faces -x); Earth at +1 au.
    site = GroundSite.from_geodetic("s", 0.0, math.pi, 0.0)
    eph = _ephemeris(earth=(AU_M, 0.0, 0.0), sun=ZERO)
    r_site = np.array([AU_M, 0.0, 0.0]) + site.position_ef_m
    theta = math.radians(12.5)
    sat = linear_trajectory(
        [0.0], r_site + 3.0e10 * np.array([-math.cos(theta), math.sin(theta), 0])
    )
    acc = _access(sat, site, eph, _settings(moon_occultation=False))
    assert acc.sun_separation_rad[0] == pytest.approx(theta, abs=1e-12)
    assert not bool(acc.sun_avoidance_violation[0])
    acc2 = _access(
        sat, site, eph, _settings(moon_occultation=False, sun_avoidance_rad=math.radians(13))
    )
    assert bool(acc2.sun_avoidance_violation[0])
    assert not bool(acc2.visible[0])
    # The satellite is in front of the Sun (0.2 au from it): not a solar occultation.
    assert not bool(acc2.sun_blocked[0])


def _first_order_apparent(direction_to_source: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apparent direction of a source for an observer moving with ``beta`` (first-order stellar
    aberration, e.g. Seidelmann (ed.), *Explanatory Supplement*, 1992, §3.252):
    ``s' = s + beta - (s . beta) s``, normalised."""
    out = direction_to_source + beta - float(direction_to_source @ beta) * direction_to_source
    return out / np.linalg.norm(out)


def test_sun_separation_with_a_moving_receiver_and_a_moving_sun() -> None:
    """Earth moving at 3e4 m/s perpendicular to the Sun line, Sun moving at 2e4 m/s along the
    same direction (in the plane of the Sun and the satellite, so both effects change the
    separation at first order; no rotation). Expected value built independently: closed-form light
    time to the satellite (quadratic), closed-form retarded epoch of the Sun (quadratic) and the
    first-order aberration of both directions into the receiver frame. Tolerance 4 beta^2
    (beta = 1e-4): the first-order formula is good to <= 2 beta^2 per direction (checked in
    test_aberration_formula_properties). Dropping the receiver aberration of the Sun direction
    changes the angle by ~1e-4 rad, using the geometric angle by ~2.3e-5 rad and an unretarded
    Sun by ~6.7e-5 rad (2e4 m/s x 500 s / 1 au)."""
    v_e = np.array([0.0, 3.0e4, 0.0])
    v_sun = np.array([0.0, 2.0e4, 0.0])
    eph = LinearMotionEphemeris(
        {
            "EARTH": ((AU_M, 0.0, 0.0), tuple(v_e)),  # type: ignore[dict-item]
            "SUN": (ZERO, tuple(v_sun)),  # type: ignore[dict-item]
            "MOON": (FAR, ZERO),
        }
    )
    site = GroundSite.from_geodetic("s", 0.0, math.pi, 0.0)  # faces -x, towards the Sun
    r_site0 = np.array([AU_M, 0.0, 0.0]) + site.position_ef_m
    theta = math.radians(40.0)
    sat_pos = r_site0 + 3.0e10 * np.array([-math.cos(theta), math.sin(theta), 0.0])
    acc = _access(linear_trajectory([0.0], sat_pos), site, eph, _settings(moon_occultation=False))

    tau = float(_uniform_motion_tau((r_site0 - sat_pos)[None, :], v_e)[0])
    r_site_a = r_site0 + v_e * tau
    u = (r_site_a - sat_pos) / np.linalg.norm(r_site_a - sat_pos)
    # Sun at t_a + t_r (t_r < 0): |v_sun (tau + t_r) - r_site_a| = -c t_r.
    p = v_sun * tau - r_site_a
    vv, pv, pp = float(v_sun @ v_sun), float(p @ v_sun), float(p @ p)
    t_r = (pv - math.sqrt(pv * pv + (C * C - vv) * pp)) / (C * C - vv)
    sun_ret = v_sun * (tau + t_r)
    to_sun = (sun_ret - r_site_a) / np.linalg.norm(sun_ret - r_site_a)
    beta = v_e / C
    expected = float(
        angle_between_rad(_first_order_apparent(-u, beta), _first_order_apparent(to_sun, beta))
    )
    b2 = float(beta @ beta)
    assert acc.sun_separation_rad[0] == pytest.approx(expected, abs=4.0 * b2)
    assert abs(acc.sun_separation_rad[0] - float(angle_between_rad(-u, to_sun))) > 1e-5
    sun_now = (v_sun * tau - r_site_a) / np.linalg.norm(v_sun * tau - r_site_a)
    unretarded = float(
        angle_between_rad(_first_order_apparent(-u, beta), _first_order_apparent(sun_now, beta))
    )
    assert abs(acc.sun_separation_rad[0] - unretarded) > 1e-5
    sun_unaberrated = float(angle_between_rad(_first_order_apparent(-u, beta), to_sun))
    assert abs(acc.sun_separation_rad[0] - sun_unaberrated) > 5e-5


# ---------------------------------------------------------------------------------------------
# Earth occultation
# ---------------------------------------------------------------------------------------------


def test_beam_through_the_earth_is_flagged() -> None:
    sat_behind = linear_trajectory([0.0], (-1.0e11, 0.0, 0.0))  # site faces +x
    sat_overhead = linear_trajectory([0.0], (1.0e11, 0.0, 0.0))
    eph = _ephemeris(sun=(0.0, -AU_M, 0.0))
    behind = _access(sat_behind, EQ_SITE, eph, _settings(moon_occultation=False))
    over = _access(sat_overhead, EQ_SITE, eph, _settings(moon_occultation=False))
    assert bool(behind.earth_blocked[0])
    assert behind.occulted_by[0] == "EARTH"
    assert not bool(behind.visible[0])
    assert not bool(over.earth_blocked[0])
    assert bool(over.visible[0])


@pytest.mark.parametrize(
    ("lat_deg", "elev_rad", "blocked"),
    [(0.0, -1e-5, True), (0.0, 1e-5, False), (60.0, -1e-5, True), (60.0, 1e-5, False)],
)
def test_earth_blocking_flips_at_the_geodetic_horizon(
    lat_deg: float, elev_rad: float, blocked: bool
) -> None:
    # On the ellipsoid the geodetic horizon plane is tangent to the surface, so a beam arriving
    # from below it crosses the Earth over a chord ~ 2 R |el| (~130 m at 1e-5 rad, well above
    # the 1 m standoff) and one from above does not touch it.
    site = GroundSite.from_geodetic("s", math.radians(lat_deg), 0.7, 0.0)
    vec = math.cos(elev_rad) * site.north_ef + math.sin(elev_rad) * site.up_ef
    sat = linear_trajectory([0.0], site.position_ef_m + 2.0e11 * vec)
    acc = _access(sat, site, _ephemeris(), _settings(moon_occultation=False))
    assert bool(acc.earth_blocked[0]) is blocked
    # With a negative minimum elevation only the Earth blockage can decide visibility.
    low = _access(
        sat, site, _ephemeris(), _settings(moon_occultation=False, min_elevation_rad=-0.1)
    )
    assert not bool(low.below_min_elevation[0])
    assert not bool(low.sun_avoidance_violation[0])
    assert bool(low.earth_blocked[0]) is blocked
    assert bool(low.visible[0]) is (not blocked)
    assert (low.occulted_by[0] == "EARTH") is blocked


@pytest.mark.parametrize("h", [-300.0, -1000.0])
@pytest.mark.parametrize(("elev_rad", "blocked"), [(-1e-5, True), (1e-5, False), (1e-4, False)])
def test_sites_below_the_ellipsoid_are_consistent_near_the_horizon(
    h: float, elev_rad: float, blocked: bool
) -> None:
    """A site below the ellipsoid is tested against the similar ellipsoid through the site, so
    a ray just above its geodetic horizon is clear and one just below is blocked. (Lowering
    both axes by |h| left the site ~1.4 mm inside the surface at h = -1000 m and blocked every
    ray up to ~1e-3 rad of elevation.) The scaled surface's normal differs from the geodetic up
    by <= 5.3e-7 rad at h = -1000 m, well inside +-1e-5 rad."""
    site = GroundSite.from_geodetic("s", math.radians(45.0), 0.3, h)
    vec = math.cos(elev_rad) * site.north_ef + math.sin(elev_rad) * site.up_ef
    sat = linear_trajectory([0.0], site.position_ef_m + 1.7e11 * vec)
    acc = _access(sat, site, _ephemeris(), _settings(moon_occultation=False))
    assert acc.elevation_rad[0] == pytest.approx(elev_rad, abs=1e-12)
    assert bool(acc.earth_blocked[0]) is blocked


@pytest.mark.parametrize("elev_rad", [5e-5, -5e-5])
def test_moving_earth_blocking_and_apparent_elevation_follow_the_earth_frame_path(
    elev_rad: float,
) -> None:
    """Earth moving at 3e4 m/s along the site's up direction (+x). The satellite is placed so
    that the photon path *relative to the Earth* (from r_sat - r_E(t_e) to the site) arrives at
    elevation +-5e-5 rad. Blockage must follow that path, and the apparent elevation in the
    receiver frame must equal it to O(beta^2) = 1e-8 rad. In the barycentric frame the line
    r_sat(t_e) -> r_site(t_a) instead looks ~1e-4 rad lower (v tau / D = v/c), so testing with
    the Earth at t_a, or using the unaberrated direction, flips the +5e-5 case."""
    v = np.array([3.0e4, 0.0, 0.0])
    eph = _ephemeris(earth_v=tuple(v), sun=(0.0, -AU_M, 0.0))  # type: ignore[arg-type]
    site = EQ_SITE
    vec = math.cos(elev_rad) * site.north_ef + math.sin(elev_rad) * site.up_ef
    sat = linear_trajectory([0.0], site.position_ef_m + 1.0e11 * vec)  # r_E(0) = 0
    acc = _access(sat, site, eph, _settings(moon_occultation=False))
    assert acc.elevation_rad[0] == pytest.approx(elev_rad, abs=1e-7)
    assert bool(acc.earth_blocked[0]) is (elev_rad < 0)
    geometric = -acc.transmit_direction_unit[0]
    assert math.asin(float(geometric @ site.up_ef)) == pytest.approx(elev_rad - 1.0e-4, abs=2e-6)


def test_standoff_and_sites_below_the_ellipsoid() -> None:
    for h in (0.0, 2500.0, -430.0):
        site = GroundSite.from_geodetic("s", math.radians(31.5), math.radians(35.5), h)
        sat = linear_trajectory([0.0], site.position_ef_m + 1.0e11 * site.up_ef)
        acc = _access(sat, site, _ephemeris(), _settings(moon_occultation=False))
        assert not bool(acc.earth_blocked[0]), h
    # Without lowering the ellipsoid, the h = -430 m site would be "inside the Earth": the
    # primitive itself reports the intersection for the full WGS 84 ellipsoid.
    site = GroundSite.from_geodetic("s", math.radians(31.5), math.radians(35.5), -430.0)
    p0 = site.position_ef_m + 1.0e11 * site.up_ef
    assert bool(segment_intersects_ellipsoid(p0, site.position_ef_m, A, B, 1.0 - 1.0 / 1.0e11))


def test_segment_ellipsoid_test_agrees_with_spice_surfpt() -> None:
    rng = np.random.default_rng(20261112)
    n_checked = n_hit = 0
    for _ in range(3000):
        p0 = rng.normal(size=3) * rng.uniform(1.0, 4.0) * A
        p1 = rng.normal(size=3) * rng.uniform(0.5, 4.0) * A
        scale = np.array([1 / A, 1 / A, 1 / B])
        if np.linalg.norm(p0 * scale) <= 1.0:
            continue  # SPICE's ray test assumes an observer outside
        dist, _ = segment_min_distance(p0 * scale, p1 * scale)
        if abs(float(dist) - 1.0) < 1e-9:
            continue  # tangent within rounding: either answer is acceptable
        ours = bool(segment_intersects_ellipsoid(p0, p1, A, B))
        with spiceypy.no_found_check():
            point, found = spiceypy.surfpt(p0, p1 - p0, A, A, B)
        ref = bool(found) and np.linalg.norm(np.asarray(point) - p0) <= np.linalg.norm(p1 - p0)
        assert ours == ref
        n_checked += 1
        n_hit += ref
    assert n_checked > 2000
    assert 0.2 * n_checked < n_hit < 0.8 * n_checked  # both outcomes well represented


def test_segment_min_distance_closed_form() -> None:
    d, s = segment_min_distance(
        [[-3.0, 1.0, 0.0], [2.0, 2.0, 0.0]], [[3.0, 1.0, 0.0], [5.0, 6.0, 0.0]]
    )
    np.testing.assert_allclose(d, [1.0, math.hypot(2.0, 2.0)])  # interior point; end point
    np.testing.assert_allclose(s, [0.5, 0.0])
    d2, s2 = segment_min_distance([-3.0, 1.0, 0.0], [3.0, 1.0, 0.0], s_max=0.25)
    assert float(s2) == 0.25
    assert float(d2) == pytest.approx(math.hypot(1.5, 1.0))
    with pytest.raises(ValueError, match="degenerate"):
        segment_min_distance([1.0, 0, 0], [1.0, 0, 0])


# ---------------------------------------------------------------------------------------------
# Sun and Moon occultation
# ---------------------------------------------------------------------------------------------


def _sun_geometry(
    target_distance_m: float,
) -> tuple[CartesianTrajectory, GroundSite, LinearMotionEphemeris]:
    """Sun at the origin, Earth at +1 au, sunward site (lon 180 deg), satellite at x = -1 au
    offset in y so that the beam segment passes ``target_distance_m`` from the Sun's centre.
    For the segment (-L1, y, 0) -> (L2, 0, 0): dist = |y| L2 / sqrt((L1 + L2)^2 + y^2), so
    y = D (L1 + L2) / sqrt(L2^2 - D^2)."""
    site = GroundSite.from_geodetic("s", 0.0, math.pi, 0.0)
    l2 = AU_M + float(site.position_ef_m[0])  # site x (= 1 au - a)
    l1 = AU_M
    y = target_distance_m * (l1 + l2) / math.sqrt(l2 * l2 - target_distance_m**2)
    sat = linear_trajectory([0.0], (-l1, y, 0.0))
    return sat, site, _ephemeris(earth=(AU_M, 0.0, 0.0), sun=ZERO)


@pytest.mark.parametrize("margin_deg", [0.0, 0.5])
@pytest.mark.parametrize(("factor", "blocked"), [(1.0 - 1e-6, True), (1.0 + 1e-6, False)])
def test_sun_blocking_flips_at_the_limb_plus_margin(
    margin_deg: float, factor: float, blocked: bool
) -> None:
    site = GroundSite.from_geodetic("s", 0.0, math.pi, 0.0)
    d_rs = AU_M + float(site.position_ef_m[0])  # receiver - Sun distance
    margin = math.radians(margin_deg)
    r_block = d_rs * math.sin(math.asin(R_SUN / d_rs) + margin)
    if margin_deg == 0.0:
        assert r_block == pytest.approx(R_SUN, rel=1e-15)  # margin 0 = the physical Sun
    sat, site, eph = _sun_geometry(factor * r_block)
    acc = _access(sat, site, eph, _settings(moon_occultation=False, sun_limb_margin_rad=margin))
    assert acc.sun_block_radius_m[0] == pytest.approx(r_block, rel=1e-12)
    # Positions ~1.5e11 m: the closest-approach distance is good to ~1e-5 m (1e-14 rel).
    assert acc.sun_beam_min_distance_m[0] == pytest.approx(factor * r_block, rel=1e-10)
    assert bool(acc.sun_blocked[0]) is blocked
    assert not bool(acc.earth_blocked[0])
    assert (acc.occulted_by[0] == "SUN") is blocked
    # Elevation ~90 deg and no Sun-avoidance limit: only the occultation decides visibility.
    assert not bool(acc.below_min_elevation[0])
    assert not bool(acc.sun_avoidance_violation[0])
    assert bool(acc.visible[0]) is (not blocked)


def test_superior_conjunction_is_sun_blocked() -> None:
    sat, site, eph = _sun_geometry(0.0)  # satellite exactly behind the Sun's centre
    acc = _access(sat, site, eph, _settings(moon_occultation=False))
    assert bool(acc.sun_blocked[0])
    assert acc.occulted_by[0] == "SUN"
    assert acc.sun_separation_rad[0] == pytest.approx(0.0, abs=1e-12)
    assert not bool(acc.below_min_elevation[0])
    assert not bool(acc.earth_blocked[0])
    assert not bool(acc.visible[0])


@pytest.mark.parametrize(("factor", "blocked"), [(1.0 - 1e-6, True), (1.0 + 1e-6, False)])
def test_moon_blocking_edge(factor: float, blocked: bool) -> None:
    # Static Moon 3.8e8 m from the site on the beam line, offset sideways by factor * R_moon.
    site = EQ_SITE
    r_site = site.position_ef_m
    sat_pos = r_site + 2.0e11 * site.up_ef
    moon = r_site + 3.8e8 * site.up_ef + factor * MOON_RADIUS_M * site.north_ef
    eph = _ephemeris(moon=tuple(moon))  # type: ignore[arg-type]
    acc = _access(linear_trajectory([0.0], sat_pos), site, eph)
    assert acc.moon_beam_min_distance_m[0] == pytest.approx(factor * MOON_RADIUS_M, rel=1e-8)
    assert bool(acc.moon_blocked[0]) is blocked
    assert (acc.occulted_by[0] == "MOON") is blocked
    assert not bool(acc.earth_blocked[0])
    assert not bool(acc.sun_blocked[0])
    assert not bool(acc.below_min_elevation[0])
    assert bool(acc.visible[0]) is (not blocked)


def test_moving_occulter_is_tested_along_the_photon_path() -> None:
    """A fast Moon (1e5 m/s, transverse): the closest approach of the *photon* to the moving
    Moon, found by brute-force sampling of both along the path, equals the comoving-segment
    result (exact for uniform motion), while a static-Moon calculation is wrong by ~4e7 m."""
    site = EQ_SITE
    v_moon = np.array([0.0, 1.0e5, 0.0])
    sat_pos = site.position_ef_m + 1.2e11 * site.up_ef
    moon0 = site.position_ef_m + 5.0e10 * site.up_ef + np.array([0.0, -2.0e7, 3.0e6])
    eph = _ephemeris(moon=tuple(moon0), moon_v=tuple(v_moon))  # type: ignore[arg-type]
    acc = _access(linear_trajectory([0.0], sat_pos), site, eph)
    tau = float(acc.light_time_s[0])
    s = np.linspace(0.0, 1.0, 400_001)
    photon = sat_pos + s[:, None] * (site.position_ef_m - sat_pos)
    moon_t = moon0 + (s * tau)[:, None] * v_moon
    brute = float(np.min(np.linalg.norm(photon - moon_t, axis=1)))
    # Sampling step 1.2e11/4e5 = 3e5 m along a path crossing the Moon's track at a steep angle;
    # the sampled minimum is within ~(step)^2 / (2 * min distance) of the true one: 1e-3 rel.
    assert acc.moon_beam_min_distance_m[0] == pytest.approx(brute, rel=1e-3)
    static_dist, _ = segment_min_distance(sat_pos - moon0, site.position_ef_m - moon0)
    assert abs(float(static_dist) - brute) > 1e7


def test_disabled_checks_are_not_evaluated() -> None:
    eph = LinearMotionEphemeris({"EARTH": (ZERO, ZERO), "SUN": ((-AU_M, 0.0, 0.0), ZERO)})
    sat = linear_trajectory([0.0], (-1.0e11, 0.0, 0.0))  # behind the Earth
    acc = _access(sat, EQ_SITE, eph, _settings(earth_occultation=False, moon_occultation=False))
    assert not bool(acc.earth_blocked[0])
    assert math.isnan(acc.moon_beam_min_distance_m[0])
    assert bool(acc.below_min_elevation[0])  # elevation is always evaluated
    with pytest.raises(Exception, match="MOON"):
        _access(sat, EQ_SITE, eph, _settings())  # the Moon is required when enabled


# ---------------------------------------------------------------------------------------------
# Pointing, rates and aberration
# ---------------------------------------------------------------------------------------------


def test_aberration_formula_properties() -> None:
    rng = np.random.default_rng(7)
    u = rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = rng.normal(size=(200, 3)) * 4.0e4
    up = aberrated_direction(u, v)
    np.testing.assert_allclose(np.linalg.norm(up, axis=1), 1.0, atol=1e-15)
    # Inverse Lorentz transformation (velocity -v) recovers the barycentric direction.
    np.testing.assert_allclose(aberrated_direction(up, -v), u, atol=1e-15)
    # First order: u' = u - beta + (u.beta) u; the remainder is O(beta^2) per component.
    beta = v / C
    first = u - beta + np.einsum("ij,ij->i", u, beta)[:, None] * u
    beta2 = np.einsum("ij,ij->i", beta, beta)
    assert np.all(np.linalg.norm(up - first, axis=1) <= 2.0 * beta2)
    assert np.max(np.linalg.norm(up - u, axis=1)) > 1e-4  # the correction itself is first order
    np.testing.assert_allclose(aberrated_direction(u, np.zeros_like(v)), u, atol=1e-15)
    # Exact case: u = y, v = beta c x -> u' = (-beta, 1/gamma, 0).
    b = 0.6
    out = aberrated_direction([0.0, 1.0, 0.0], [b * C, 0.0, 0.0])
    np.testing.assert_allclose(out, [-b, math.sqrt(1 - b * b), 0.0], atol=1e-15)
    # Velocity along the beam does not change its direction.
    np.testing.assert_allclose(aberrated_direction([1.0, 0, 0], [3e4, 0, 0]), [1, 0, 0])


def test_comoving_transmitter_and_receiver_need_no_net_point_ahead() -> None:
    """Satellite and site share one velocity V (no rotation): in their common rest frame the
    geometry is static, so the aberration-corrected transmit direction must equal the direction
    to the site at the emission epoch, up to O(beta^2). The two corrections (retarded target,
    transmitter aberration) are each ~beta sin(theta) ~ 1e-4 rad and cancel; a sign error in
    either would leave ~2e-4 rad."""
    vel = np.array([2.0e4, 2.2e4, -5.0e3])
    beta = float(np.linalg.norm(vel)) / C
    eph = _ephemeris(earth=(3.0e9, 0.0, 0.0), earth_v=tuple(vel))  # type: ignore[arg-type]
    t = np.array([0.0, 5.0e3])
    sat = linear_trajectory(t, (3.0e9 + A + 4.0e10, 9.0e10, 1.0e10), vel)
    acc = _access(sat, EQ_SITE, eph, _settings(moon_occultation=False, earth_occultation=False))
    inst = acc.light_time.site_at_emission.position_m - sat.position_m
    assert np.all(acc.transmitter_aberration_rad > 0.3 * beta)
    assert np.all(acc.target_motion_point_ahead_rad > 0.3 * beta)
    assert np.all(angle_between_rad(acc.transmit_direction_tx_frame_unit, inst) < 3 * beta**2)


def _kepler_case(t: np.ndarray, rotation_rate: float) -> tuple[AccessResult, AnalyticEphemeris]:
    eph = AnalyticEphemeris(earth_longitude_at_j2000_rad=0.4)  # Sun at rest at the SSB origin
    el = OrbitalElements(
        a_m=0.95 * AU_M,
        e=0.05,
        i_rad=0.1,
        raan_rad=0.3,
        argp_rad=0.2,
        nu_rad=0.9,
        mu_m3_s2=eph.gm_m3_s2("SUN"),
        epoch_tdb_s=0.0,
        frame="ECLIPJ2000",
    )
    helio = transform_trajectory(propagate_kepler(el, t).trajectory, "J2000")
    sat = CartesianTrajectory(  # the Sun sits at the SSB origin in this test double
        epochs_tdb_s=t,
        position_m=helio.position_m,
        velocity_m_s=helio.velocity_m_s,
        frame="J2000",
        origin=SSB,
    )
    site = GroundSite.from_geodetic("s", math.radians(35.0), math.radians(-100.0), 500.0)
    acc = compute_access(
        sat,
        site,
        ephemeris=eph,
        earth_orientation=UniformEarthRotation(0.2, rotation_rate),
        settings=_settings(moon_occultation=False),
    )
    return acc, eph


def test_pointing_rate_range_rate_and_light_time_rate_match_finite_differences() -> None:
    h = 5.0
    centre = np.array([1.0e6, 2.0e7])
    t = np.sort(np.concatenate([centre - h, centre, centre + h]))
    acc, _ = _kepler_case(t, 7.2921e-5)
    for k in (1, 4):
        fd_angle = angle_between_rad(
            acc.transmit_direction_unit[k + 1], acc.transmit_direction_unit[k - 1]
        )
        # Central difference: relative truncation ~ (omega_rot h)^2 / 6 ~ 2e-8 on the site-
        # rotation part and ~1e-15 on the orbital part; angle rounding ~1e-16 / 2e-6 rad.
        assert acc.pointing_rate_rad_s[k] == pytest.approx(fd_angle / (2 * h), rel=1e-6)
        fd_range = (acc.range_m[k + 1] - acc.range_m[k - 1]) / (2 * h)
        fd_tau = (acc.light_time_s[k + 1] - acc.light_time_s[k - 1]) / (2 * h)
        # d|rho|/dt_e = c dtau/dt_e exactly. Central-difference truncation omega^3 a h^2/6 =
        # 1e-5 m/s (site rotation), range rounding ~3e-5 m / 10 s = 3e-6 m/s: abs 5e-5 m/s.
        assert C * acc.light_time_rate[k] == pytest.approx(fd_range, abs=5e-5)
        assert acc.light_time_rate[k] == pytest.approx(fd_tau, abs=5e-5 / C)
    # range_rate is the relative radial velocity u.(v_site - v_sat); it differs from c tau' by
    # the factor (1 - u.v_site/c) (Doppler-like ~1e-4 correction): exact identity.
    st = acc.light_time.site_at_arrival
    u = acc.transmit_direction_unit
    u_vsite = np.einsum("ij,ij->i", u, st.velocity_m_s)
    np.testing.assert_allclose(
        acc.range_rate_m_s, C * acc.light_time_rate * (1.0 - u_vsite / C), rtol=1e-12, atol=1e-9
    )


def test_pointing_rate_of_a_static_line_of_sight_is_zero() -> None:
    sat = linear_trajectory([0.0, 1.0], (2.0e11, 1.0e10, 0.0))
    acc = _access(sat, EQ_SITE, _ephemeris(), _settings(moon_occultation=False))
    np.testing.assert_array_equal(acc.pointing_rate_rad_s, 0.0)
    np.testing.assert_array_equal(acc.range_rate_m_s, 0.0)


# ---------------------------------------------------------------------------------------------
# Satellite illumination
# ---------------------------------------------------------------------------------------------


def test_circle_overlap_area_against_quadrature_and_closed_forms() -> None:
    r = 2.0
    # Two equal circles with d = r: lens area r^2 (2 pi / 3 - sqrt(3)/2) (closed form).
    assert circle_overlap_area(r, r, r) == pytest.approx(
        r * r * (2 * math.pi / 3 - math.sqrt(3) / 2), rel=1e-14
    )
    assert circle_overlap_area(1.0, 0.3, 0.5) == pytest.approx(math.pi * 0.09, rel=1e-15)  # inside
    assert circle_overlap_area(1.0, 0.3, 1.3) == 0.0  # tangent outside
    for r1, r2, d in [(1.0, 0.7, 1.2), (0.3, 1.0, 1.1), (1.0, 1.0, 0.4), (4.6e-3, 8.5e-3, 1.0e-2)]:

        def chord(x: float, r1: float = r1, r2: float = r2, d: float = d) -> float:
            h1 = math.sqrt(max(r1 * r1 - x * x, 0.0))
            h2 = math.sqrt(max(r2 * r2 - (x - d) ** 2, 0.0))
            return 2.0 * min(h1, h2)

        lo, hi = max(-r1, d - r2), min(r1, d + r2)
        ref, _err = quad(
            chord,
            lo,
            hi,
            epsabs=0.0,
            epsrel=1e-12,
            limit=200,
            points=[(d * d + r1 * r1 - r2 * r2) / (2 * d)],
        )
        assert circle_overlap_area(r1, r2, d) == pytest.approx(ref, rel=1e-9)


def test_occulted_fraction_limits() -> None:
    rs = 4.65e-3
    assert solar_disk_occulted_fraction(rs, 1e-3, 1.0) == 0.0
    assert solar_disk_occulted_fraction(rs, 2 * rs, 0.5 * rs) == 1.0  # umbra
    assert solar_disk_occulted_fraction(rs, 0.4 * rs, 0.1 * rs) == pytest.approx(0.16, rel=1e-14)
    seps = np.linspace(0.0, 3 * rs, 301)
    f = solar_disk_occulted_fraction(rs, 0.8 * rs, seps)
    assert np.all(np.diff(f) <= 1e-15)  # monotone in separation
    assert f[0] == pytest.approx(0.64, rel=1e-14)


def _illum_ephemeris(
    earth: tuple[float, float, float],
    earth_v: tuple[float, float, float] = ZERO,
    moon: tuple[float, float, float] = FAR,
) -> LinearMotionEphemeris:
    return LinearMotionEphemeris(
        {"SUN": (ZERO, ZERO), "EARTH": (earth, earth_v), "MOON": (moon, ZERO)}
    )


def test_illumination_full_sun_umbra_annular_and_behind_the_sun() -> None:
    t = [0.0]
    full = satellite_illumination(
        linear_trajectory(t, (AU_M, 0.0, 0.0)), ephemeris=_illum_ephemeris((0.0, AU_M, 0.0))
    )
    assert full.illumination_fraction[0] == 1.0
    assert full.occulted_by[0] == ""
    assert not bool(full.multiple_occulters[0])
    assert full.time_scale == "TDB"
    assert full.source_ids == _illum_ephemeris((0.0, AU_M, 0.0)).source_kernel_ids
    # 5e8 m behind the Earth on the Sun line: Earth's apparent radius 1.3e-2 rad > Sun's 4.6e-3.
    umbra = satellite_illumination(
        linear_trajectory(t, (AU_M + 5.0e8, 0.0, 0.0)), ephemeris=_illum_ephemeris((AU_M, 0.0, 0.0))
    )
    assert umbra.illumination_fraction[0] == 0.0
    assert umbra.occulted_by[0] == "EARTH"
    assert not bool(umbra.multiple_occulters[0])  # the Moon is parked far away
    # Moon centred on the Sun, far enough to be smaller: annular, 1 - (a_moon/a_sun)^2.
    d_moon = 6.0e8
    sat = linear_trajectory(t, (AU_M + d_moon, 0.0, 0.0))
    ann = satellite_illumination(
        sat, ephemeris=_illum_ephemeris(FAR, moon=(AU_M, 0.0, 0.0)), occulters=("MOON",)
    )
    a_s = math.asin(R_SUN / (AU_M + d_moon))
    a_m = math.asin(MOON_RADIUS_M / d_moon)
    assert a_m < a_s
    assert ann.illumination_fraction[0] == pytest.approx(1.0 - (a_m / a_s) ** 2, rel=1e-12)
    # An occulter beyond the Sun casts no shadow on the satellite.
    behind = satellite_illumination(
        linear_trajectory(t, (AU_M, 0.0, 0.0)), ephemeris=_illum_ephemeris((-AU_M, 1.0e7, 0.0))
    )
    assert behind.illumination_fraction[0] == 1.0


def test_illumination_partial_and_light_time_of_the_occulter() -> None:
    """Penumbra with a moving Earth: the shadow uses the Earth's retarded position
    (t - d/c). Expected value from the independent quadrature of the overlap with the Earth
    displaced by -v d/c; the instantaneous position would give a clearly different value."""
    d = 1.0e9  # satellite-Earth distance (3.3 s of light time; Earth 6.4e-3 rad, Sun 4.6e-3)
    v = np.array([0.0, 3.0e4, 0.0])
    earth0 = np.array([AU_M, 0.0, 0.0])
    sat_pos = np.array([AU_M + d, 6.4e6, 0.0])  # Sun's centre near the Earth's limb
    eph = _illum_ephemeris(tuple(earth0), tuple(v))  # type: ignore[arg-type]
    res = satellite_illumination(
        linear_trajectory([0.0], sat_pos), ephemeris=eph, occulters=("EARTH",)
    )

    def fraction(earth_pos: np.ndarray) -> float:
        to_sun, to_earth = -sat_pos, earth_pos - sat_pos
        a_s = math.asin(R_SUN / np.linalg.norm(to_sun))
        a_e = math.asin(A / np.linalg.norm(to_earth))
        sep = float(angle_between_rad(to_sun, to_earth))
        lo, hi = max(-a_s, sep - a_e), min(a_s, sep + a_e)

        def chord(x: float) -> float:
            return 2.0 * min(
                math.sqrt(max(a_s * a_s - x * x, 0.0)),
                math.sqrt(max(a_e * a_e - (x - sep) ** 2, 0.0)),
            )

        area, _ = quad(chord, lo, hi, epsabs=0.0, epsrel=1e-12, limit=200)
        return 1.0 - area / (math.pi * a_s * a_s)

    # Uniform motion: retarded epoch t_r solves |earth0 + v t_r - sat| = -c t_r (t = 0).
    dvec = earth0 - sat_pos
    a2, b2, c2 = C * C - float(v @ v), -2.0 * float(dvec @ v), -float(dvec @ dvec)
    t_r = (-b2 - math.sqrt(b2 * b2 - 4 * a2 * c2)) / (2 * a2)
    expected = fraction(earth0 + v * t_r)
    instantaneous = fraction(earth0)
    assert 0.05 < expected < 0.95
    # Two fixed-point steps leave ~(v/c)^2 relative epoch error: 1e-9 is ample.
    assert res.illumination_fraction[0] == pytest.approx(expected, abs=1e-9)
    assert abs(expected - instantaneous) > 1e-3


def test_two_occulters_are_flagged() -> None:
    sat = np.array([AU_M + 6.0e8, 0.0, 0.0])
    eph = _illum_ephemeris((AU_M - 1.0e10, 6.0e6, 0.0), moon=(AU_M, 2.0e6, 0.0))
    res = satellite_illumination(linear_trajectory([0.0], sat), ephemeris=eph)
    assert bool(res.multiple_occulters[0])
    assert res.illumination_fraction[0] == pytest.approx(1.0 - res.occulted_fraction[0].sum())


# ---------------------------------------------------------------------------------------------
# Diagnostics (PDF §7.3) and configuration
# ---------------------------------------------------------------------------------------------


def test_summary_of_a_rotating_equatorial_site() -> None:
    """Satellite far away in the equatorial plane, Earth rotating uniformly: over one turn the
    satellite is above elevation e for a fraction (pi/2 - e)/pi of the time (sin el = cos H at
    the equator) and below the geometric horizon half the time."""
    period = 86_400.0
    n = 7200
    t = np.arange(n) * (period / n)
    rot = UniformEarthRotation(0.0, 2 * math.pi / period)
    eph = _ephemeris(sun=(0.0, 0.0, -AU_M))
    sat = linear_trajectory(t, (1.0e13, 0.0, 0.0))  # parallax a/D = 6e-7 rad
    e_min = math.radians(10.0)
    acc = compute_access(
        sat,
        EQ_SITE,
        ephemeris=eph,
        earth_orientation=rot,
        settings=_settings(min_elevation_rad=e_min, moon_occultation=False),
    )
    s = summarize_access(acc)
    # Sampling quantisation 1/n per crossing (2 crossings) plus the light-time phase shift
    # (tau = 33 000 s shifts the pattern but not its duty cycle): tolerance 3/n.
    assert s.access_fraction == pytest.approx(0.5 - e_min / math.pi, abs=3 / n)
    assert s.earth_occultation_fraction == pytest.approx(0.5, abs=3 / n)
    assert s.occultation_fraction == s.earth_occultation_fraction
    assert s.sun_occultation_fraction == 0.0
    assert s.elevation_max_rad == pytest.approx(math.pi / 2, abs=2 * math.pi / n)
    assert s.light_time_min_s == pytest.approx(float(np.min(acc.range_m)) / C, rel=1e-12)
    assert s.light_time_max_s == pytest.approx(float(np.max(acc.range_m)) / C, rel=1e-12)
    assert s.range_mean_m == pytest.approx(1.0e13, rel=1e-6)
    assert s.n_epochs == n
    assert s.span_s == pytest.approx(t[-1])
    assert s.light_time_fallback_count == 0
    assert s.orbit_drift_max_m is None
    drift = np.linspace(0.0, 5.0, n)
    s2 = summarize_access(acc, orbit_drift=drift)
    assert s2.orbit_drift_max_m == 5.0
    assert s2.orbit_drift_final_m == 5.0
    with pytest.raises(ValueError, match="one value per"):
        summarize_access(acc, orbit_drift=drift[:-1])


def test_summary_aggregates_every_occulter_and_averages_light_time() -> None:
    """Hand-set flag patterns on a real result (dataclasses.replace re-validates it): the
    occultation fraction is the fraction of epochs blocked by *any* body (union, not sum, not
    one body), a Moon-only blockage counts, and the light-time mean is the arithmetic mean."""
    n = 8
    sat = linear_trajectory(np.arange(n) * 60.0, (2.0e11, 1.0e10, 0.0))
    acc = _access(sat, EQ_SITE, _ephemeris(), _settings(moon_occultation=False))

    def flags(*idx: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.bool_)
        out[list(idx)] = True
        return out

    mixed = dataclasses.replace(
        acc,
        earth_blocked=flags(0, 4),
        sun_blocked=flags(0, 1),
        moon_blocked=flags(2),
        light_time_s=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0]),
    )
    s = summarize_access(mixed)
    assert s.occultation_fraction == 4 / n  # {0, 1, 2, 4}; the sum of the parts would be 5/8
    assert (s.earth_occultation_fraction, s.sun_occultation_fraction) == (2 / n, 2 / n)
    assert s.moon_occultation_fraction == 1 / n
    assert s.light_time_mean_s == 2.0  # median 1.0
    moon_only = summarize_access(
        dataclasses.replace(acc, moon_blocked=flags(3, 5), visible=~flags(3, 5))
    )
    assert moon_only.occultation_fraction == 2 / n
    assert moon_only.access_fraction == 6 / n
    # On the unmodified result the mean light time is the mean range / c.
    assert summarize_access(acc).light_time_mean_s == pytest.approx(
        float(np.mean(acc.range_m)) / C, rel=1e-14
    )


def test_hour_angle_and_sampling_gap() -> None:
    """Uniformly rotating Earth, satellite far away on +x: the sampled hour angle advances by
    2 pi / n per sample (gap 2 pi / n); 24 samples per day repeated for 3 days revisit the same
    24 hour angles (gap 2 pi / 24, the hourly-aliasing case); a non-rotating Earth samples a
    single hour angle (gap 2 pi). Hour angle 0 when the satellite is on the site's meridian."""
    period = 86_400.0
    rot = UniformEarthRotation(0.0, 2 * math.pi / period)
    far = 1.0e13  # parallax a/D = 6e-7 rad; light time 33 356 s shifts the phase uniformly

    def summary(t: np.ndarray, rotation: UniformEarthRotation) -> tuple[AccessResult, float]:
        sat = linear_trajectory(t, (far, 0.0, 0.0))
        acc = compute_access(
            sat,
            EQ_SITE,
            ephemeris=_ephemeris(sun=(0.0, 0.0, -AU_M)),
            earth_orientation=rotation,
            settings=_settings(moon_occultation=False),
        )
        return acc, summarize_access(acc).hour_angle_max_gap_rad

    acc, gap = summary(np.arange(360) * (period / 360), rot)
    # The site's longitude at t_a is 2 pi t_a / P; the satellite (at +x) has Earth-fixed
    # longitude -2 pi t_a / P, so H = 2 pi t_a / P mod 2 pi (up to the 6e-7 rad parallax).
    h_expected = np.mod(2 * math.pi * acc.t_arrival_tdb_s / period, 2 * math.pi)
    diff = np.angle(np.exp(1j * (satellite_hour_angle_rad(acc) - h_expected)))
    np.testing.assert_allclose(diff, 0.0, atol=1e-6)
    # The light time varies by +-a/c = 0.021 s around the rotation (1.6e-6 rad of phase) and the
    # parallax by 6e-7 rad: gaps equal 2 pi / n to 5e-6 rad.
    assert gap == pytest.approx(2 * math.pi / 360, abs=5e-6)
    _, gap_hourly = summary(np.arange(72) * 3600.0, rot)
    assert gap_hourly == pytest.approx(2 * math.pi / 24, abs=5e-6)
    _, gap_static = summary(np.arange(5) * 3600.0, NO_ROTATION)
    assert gap_static == pytest.approx(2 * math.pi, abs=1e-12)
    assert max_circular_gap_rad([0.1, 0.1 + math.pi]) == pytest.approx(math.pi)
    with pytest.raises(ValueError, match="no angles"):
        max_circular_gap_rad([])


def test_orbit_drift_requires_matching_frames_and_epochs() -> None:
    a = linear_trajectory([0.0, 1.0], (1.0, 2.0, 3.0))
    b = linear_trajectory([0.0, 1.0], (1.0, 2.0, 7.0))
    np.testing.assert_array_equal(orbit_drift_m(a, b), [4.0, 4.0])
    with pytest.raises(FrameMismatchError):
        orbit_drift_m(a, transform_trajectory(b, "ECLIPJ2000"))
    with pytest.raises(ValueError, match="epochs"):
        orbit_drift_m(a, linear_trajectory([0.0, 2.0], (0.0, 0.0, 0.0)))


def test_access_requires_j2000_ssb_satellite_states() -> None:
    sat = transform_trajectory(linear_trajectory([0.0], (1e11, 0.0, 0.0)), "ECLIPJ2000")
    with pytest.raises(FrameMismatchError):
        _access(sat, EQ_SITE, _ephemeris())


def test_access_settings_from_config(mvp_data: dict[str, object]) -> None:
    set_path(mvp_data, "receiver.sun_avoidance_deg", 5.0)  # type: ignore[arg-type]
    set_path(mvp_data, "models.sun_occultation_limb_margin_deg", 0.25)  # type: ignore[arg-type]
    set_path(mvp_data, "models.light_time", False)  # type: ignore[arg-type]
    set_path(mvp_data, "models.moon_occultation", False)  # type: ignore[arg-type]
    cfg = parse_config(mvp_data)
    s = access_settings_from_config(cfg)
    assert s.min_elevation_rad == pytest.approx(math.radians(10.0), rel=1e-15)
    assert s.sun_avoidance_rad == pytest.approx(math.radians(5.0), rel=1e-15)
    assert s.sun_limb_margin_rad == pytest.approx(math.radians(0.25), rel=1e-15)
    assert s.light_time.enabled is False
    assert s.light_time.tolerance_s == 1e-9
    assert s.light_time.max_iterations == 50
    assert s.moon_occultation is False
    assert s.earth_standoff_m == 1.0
    assert s.radii == OcculterRadii()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("models.light_time_tolerance_s", 0.0),
        ("models.light_time_tolerance_s", 1.0),
        ("models.light_time_max_iterations", 1),
        ("models.light_time_max_iterations", 2.5),
        ("models.earth_occultation_standoff_m", -1.0),
    ],
)
def test_new_config_fields_are_validated(
    mvp_data: dict[str, object], path: str, value: object
) -> None:
    from helioswarm.config import ConfigError

    set_path(mvp_data, path, value)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match=path.split(".")[1]):
        parse_config(mvp_data)


def test_settings_validation() -> None:
    with pytest.raises(ValueError, match="tolerance_s"):
        LightTimeSettings(tolerance_s=0.0)
    with pytest.raises(ValueError, match="sun_avoidance_rad"):
        AccessSettings(sun_avoidance_rad=-0.1)
    with pytest.raises(ValueError, match="earth_polar_radius_m"):
        OcculterRadii(earth_polar_radius_m=2 * A)
