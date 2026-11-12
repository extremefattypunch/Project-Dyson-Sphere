"""M3 exit criterion: MVP-configuration access diagnostics over a full year with SPICE.

The MVP satellite (configs/mvp_single_satellite.yaml) is on a 1 au circular orbit at ecliptic
longitude 30 deg on 2030-01-01. The Earth's heliocentric longitude on that date is ~100 deg, so
the satellite trails the Earth by ~70 deg (it is *not* "30 deg ahead of the Earth"): the Earth
range is ~2 sin(35 deg) au ~ 1.15 au, not 0.52 au. The tests below check the access geometry
against this actual configuration.

The L1 trajectory is integrated with a 1-day step cap instead of the configured 300 s; M2
measured the two to agree to 4.6 cm over the year (docs/milestones/M2.md), far below anything
the access geometry resolves, and it keeps the module under ~10 s.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

import numpy as np
import pytest
import spiceypy

from helioswarm.config import load_config, parse_config
from helioswarm.constants import (
    AU_M,
    MOON_RADIUS_M,
    SECONDS_PER_DAY,
    SOLAR_RADIUS_NOMINAL_M,
    SPEED_OF_LIGHT_M_S,
    WGS84_SEMIMAJOR_AXIS_M,
)
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.frames import CartesianTrajectory
from helioswarm.ground import GroundSite, SpiceEarthOrientation, site_trajectory
from helioswarm.kernels import KernelSet
from helioswarm.propagate import (
    ForceModel,
    KernelSetMismatchError,
    SatelliteState,
    initial_state_from_config,
    integrator_settings_from_config,
    l0_reference_trajectory,
    propagate_l1,
)
from helioswarm.timescale import time_grid_from_config, utc_to_tdb_s
from helioswarm.visibility import (
    AccessSettings,
    OcculterRadii,
    access_settings_from_config,
    angle_between_rad,
    compute_access,
    compute_access_from_config,
    orbit_drift_m,
    satellite_illumination,
    summarize_access,
)

from config_helpers import MVP_CONFIG, load_mvp_data, set_path

pytestmark = pytest.mark.spice

C = SPEED_OF_LIGHT_M_S


@dataclasses.dataclass
class YearRun:
    access_s: float
    illumination_s: float
    results: dict[str, Any]


def _equatorial_declination_and_hour_angle_fraction(
    sat_pos: np.ndarray, earth_pos: np.ndarray, pole: np.ndarray, e_min: float
) -> np.ndarray:
    """Analytic diurnal access fraction H0/pi for a site on the equator (geodetic = geocentric
    there): sin el = cos(delta) cos(H) (spherical astronomy; parallax a/range = 4e-5 rad
    neglected), so el >= e_min for |H| <= H0 = acos(sin e_min / cos delta), with delta the
    satellite's declination with respect to the Earth's rotation pole."""
    rel = sat_pos - earth_pos
    sin_dec = np.einsum("ij,ij->i", rel, pole) / np.linalg.norm(rel, axis=1)
    cos_dec = np.sqrt(1.0 - sin_dec**2)
    return np.arccos(np.clip(math.sin(e_min) / cos_dec, -1.0, 1.0)) / math.pi


@pytest.fixture(scope="module")
def year(kernel_set: KernelSet) -> YearRun:
    """Propagate the MVP year and compute access + illumination (kernels unloaded on return)."""
    cfg = load_config(MVP_CONFIG)
    out: dict[str, Any] = {"config": cfg}
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        grid = time_grid_from_config(cfg.time, k)
        init = initial_state_from_config(cfg, eph, k)
        settings = dataclasses.replace(
            integrator_settings_from_config(cfg.time), max_step_s=SECONDS_PER_DAY
        )
        prop = propagate_l1(init, grid.epochs_tdb_s, ephemeris=eph, settings=settings)
        t0 = time.perf_counter()
        (acc,) = compute_access_from_config(cfg, prop.trajectory, ephemeris=eph, kernels=k)
        access_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        ill = satellite_illumination(
            prop.trajectory, ephemeris=eph, radii=OcculterRadii.from_kernels(k)
        )
        illumination_s = time.perf_counter() - t0
        t_init = utc_to_tdb_s(cfg.orbit.initial_epoch_utc, k)
        ref = l0_reference_trajectory(
            cfg.orbit, grid.epochs_tdb_s, initial_epoch_tdb_s=t_init, ephemeris=eph
        )
        # Sun-only, no-SRP L1 from the same initial state: must reproduce the L0 reference
        # (M2: 7.7e-4 m over the year), which validates l0_reference_trajectory.
        sun_only = SatelliteState(state=init.state, force_model=ForceModel(("SUN",)))
        prop_sun = propagate_l1(sun_only, grid.epochs_tdb_s, ephemeris=eph, settings=settings)
        earth = eph.states("EARTH", grid.epochs_tdb_s)
        sun = eph.positions("SUN", grid.epochs_tdb_s)
        pole = SpiceEarthOrientation(k).rotation(grid.epochs_tdb_s).matrix[:, :, 2]
        radii = OcculterRadii.from_kernels(k)
        out.update(
            access=acc,
            ill=ill,
            traj=prop.trajectory,
            ref=ref,
            sun_only=prop_sun.trajectory,
            earth_pos=earth.position_m,
            earth_vel=earth.velocity_m_s,
            sun_pos=sun,
            pole=pole,
            radii=radii,
        )
    return YearRun(access_s=access_s, illumination_s=illumination_s, results=out)


def test_year_access_is_vectorised_and_light_time_converges(year: YearRun) -> None:
    acc = year.results["access"]
    assert len(acc) == 8767  # 365.25 days of hourly samples, both ends included
    # Measured ~1.2 s for 8767 epochs on holy7c18406 (SPICE calls dominate); 30 s ceiling.
    assert year.access_s < 30.0
    assert year.illumination_s < 30.0
    lt = acc.light_time
    assert np.all(lt.converged)
    assert np.all(lt.method == "fixed_point")
    # Contraction factor |v_site|/c ~ 1e-4 and tau_0 off by ~ v tau / c ~ 0.06 s: 3 iterations.
    assert int(np.max(lt.iterations)) <= 4
    # Residual |c tau - |rho||: the tolerance 1e-9 s times the ~1e-4 contraction, plus the
    # float64 spacing of 1.7e11 m (3e-5 m) and of t_a (6e-8 s -> 2 mm): < 1 cm.
    assert float(np.max(lt.residual_m)) < 1e-2
    np.testing.assert_allclose(acc.light_time_s, acc.range_m / C, rtol=1e-12)
    assert acc.source_ids[-2:] == ("pck00011.tpc", "gm_de440.tpc")


def test_year_range_and_pointing_are_physically_sensible(year: YearRun) -> None:
    r = year.results
    acc, traj = r["access"], r["traj"]
    # Heliocentric angle satellite-Sun-Earth at the first epoch, from the raw ephemeris.
    hs = traj.position_m[0] - r["sun_pos"][0]
    he = r["earth_pos"][0] - r["sun_pos"][0]
    sep0 = float(angle_between_rad(hs, he))
    assert math.degrees(sep0) == pytest.approx(70.0, abs=1.5)  # Earth at ~100 deg longitude
    # Range vs the geocentric distance at the emission epoch: they differ by the site offset
    # (<= a) and the Earth's motion during the light time (<= |v_E| tau): bound a + v tau.
    geo = np.linalg.norm(traj.position_m - r["earth_pos"], axis=1)
    bound = WGS84_SEMIMAJOR_AXIS_M + np.linalg.norm(r["earth_vel"], axis=1) * acc.light_time_s
    assert np.all(np.abs(acc.range_m - geo) <= bound)
    s = summarize_access(acc, orbit_drift=orbit_drift_m(traj, r["ref"]))
    # Chord between two ~1 au orbits separated by 70 +/- 3 deg: 1.10-1.20 au.
    assert 1.10 * AU_M < s.range_min_m < s.range_max_m < 1.20 * AU_M
    assert s.light_time_min_s == pytest.approx(s.range_min_m / C, rel=1e-12)
    # The line between two bodies on near-circular 1 au orbits turns at their mean motion
    # n = 2 pi / year (1.99e-7 rad/s); the site's rotation adds ~465 m/s / 1.7e11 m = 3e-9
    # rad/s and the eccentricities a few per cent: 10 %.
    n = 2 * math.pi / (365.25 * SECONDS_PER_DAY)
    assert s.pointing_rate_mean_rad_s == pytest.approx(n, rel=0.1)
    # Transmitter aberration: |beta_perp| to first order (remainder <= beta^2 ~ 1e-8 rad).
    beta = traj.velocity_m_s / C
    u = acc.transmit_direction_unit
    beta_perp = np.linalg.norm(beta - np.einsum("ij,ij->i", u, beta)[:, None] * u, axis=1)
    np.testing.assert_allclose(acc.transmitter_aberration_rad, beta_perp, rtol=0, atol=2e-8)
    assert 4e-5 < s.transmitter_aberration_max_rad < 1.1e-4  # ~ 1000 optical beam widths
    # Range rate is bounded by the relative orbital speed of two ~co-orbiting bodies
    # (eccentricity 0.0167 -> ~0.5 km/s radial) plus the site rotation (0.47 km/s).
    assert 0.0 < s.range_rate_max_m_s < 3.0e3
    assert -3.0e3 < s.range_rate_min_m_s < 0.0


def test_year_access_fraction_matches_the_diurnal_elevation_fraction(year: YearRun) -> None:
    """With no Sun/Moon blocking (Sun separation > 50 deg all year), access is the time above
    10 deg elevation. For the equatorial site the analytic fraction is mean(H0/pi). The hourly
    grid aliases with the diurnal cycle (the satellite's synodic day is within minutes of the
    solar day, so each day is sampled at nearly the same 24 hour angles): the hourly fraction
    is then quantised to ~1/24 and may differ from the analytic value by up to that."""
    r = year.results
    acc = r["access"]
    s = summarize_access(acc)
    e_min = r["config"].receiver.min_elevation_rad
    expected = float(
        np.mean(
            _equatorial_declination_and_hour_angle_fraction(
                r["traj"].position_m, r["earth_pos"], r["pole"], e_min
            )
        )
    )
    assert 0.43 < expected < 0.45
    assert s.access_fraction == pytest.approx(expected, abs=1.0 / 24.0)
    # The sampling diagnostic exposes the aliasing: the hourly grid leaves an 8.6 deg gap in
    # the sampled hour angles (24 clusters 15 deg apart, smeared by the equation of time), and
    # the access-fraction error (0.020) is of the size gap / (2 pi) = 0.024 per window edge.
    assert math.radians(5.0) < s.hour_angle_max_gap_rad < math.radians(15.0)
    assert abs(s.access_fraction - expected) <= 2.0 * s.hour_angle_max_gap_rad / (2.0 * math.pi)
    assert s.sun_occultation_fraction == 0.0
    assert s.moon_occultation_fraction == 0.0
    assert float(np.min(acc.sun_separation_rad)) > math.radians(50.0)
    assert s.access_fraction == pytest.approx(1.0 - s.below_min_elevation_fraction, abs=1e-15)
    # Equatorial site: the geodetic horizon is tangent to the ellipsoid, so "Earth blocks the
    # beam" (photon path in the Earth's translating frame) and "apparent elevation < 0"
    # (receiver frame, which also moves with the 465 m/s rotation: 465/c = 1.6e-6 rad) coincide
    # except within 3e-6 rad of the horizon.
    near_horizon = np.abs(acc.elevation_rad) < 3e-6
    assert np.array_equal(acc.earth_blocked[~near_horizon], (acc.elevation_rad < 0)[~near_horizon])
    assert s.earth_occultation_fraction == pytest.approx(0.5, abs=0.01)


def test_dense_month_access_fraction_matches_the_analytic_value(kernel_set: KernelSet) -> None:
    """30 days at 5-minute steps (no aliasing): the access fraction equals the analytic
    diurnal fraction to within the sampling quantisation (half a step per crossing, i.e.
    <= 2 * 2.5 min / 1440 min = 0.0035 per day if every crossing erred the same way): 0.005."""
    cfg = load_config(MVP_CONFIG)
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        init = initial_state_from_config(cfg, eph, k)
        t = utc_to_tdb_s("2030-02-01T00:00:00", k) + np.arange(0.0, 30 * SECONDS_PER_DAY, 300.0)
        settings = dataclasses.replace(
            integrator_settings_from_config(cfg.time), max_step_s=SECONDS_PER_DAY
        )
        traj = propagate_l1(init, t, ephemeris=eph, settings=settings).trajectory
        (acc,) = compute_access_from_config(cfg, traj, ephemeris=eph, kernels=k)
        pole = SpiceEarthOrientation(k).rotation(t).matrix[:, :, 2]
        expected = float(
            np.mean(
                _equatorial_declination_and_hour_angle_fraction(
                    traj.position_m, eph.positions("EARTH", t), pole, cfg.receiver.min_elevation_rad
                )
            )
        )
    dense = summarize_access(acc)
    assert dense.access_fraction == pytest.approx(expected, abs=0.005)
    # 5-minute steps (1.25 deg of hour angle) drifting over 30 days: gap < 0.2 deg.
    assert dense.hour_angle_max_gap_rad < math.radians(0.2)


def test_year_illumination_and_orbit_drift(year: YearRun) -> None:
    r = year.results
    ill = r["ill"]
    # The MVP satellite never passes behind the Earth or the Moon (it trails the Earth by
    # ~70 deg): fully lit all year.
    np.testing.assert_array_equal(ill.illumination_fraction, 1.0)
    assert np.all(ill.occulted_by == "")
    radii = r["radii"]
    # pck00011 BODY10_RADII = 695700 km, BODY301_RADII = 1737.4 km (km -> m and the mean of
    # three equal axes round at the last bit): the PCK and constants.py agree to 1e-15.
    assert radii.sun_radius_m == pytest.approx(SOLAR_RADIUS_NOMINAL_M, rel=1e-15)
    assert radii.moon_radius_m == pytest.approx(MOON_RADIUS_M, rel=1e-15)
    np.testing.assert_allclose(ill.sun_distance_m, AU_M, rtol=0.01)
    drift = orbit_drift_m(r["traj"], r["ref"])
    assert drift[0] < 1e-3  # same initial state
    # Planets + SRP move the orbit by ~1e7 m within a year (M2: SRP alone 3.7e7 m).
    assert 1e6 < float(np.max(drift)) < 1e8
    # The L0 reference equals Sun-only L1 (M2: 7.7e-4 m over the year): tolerance 1e-2 m.
    assert float(np.max(orbit_drift_m(r["sun_only"], r["ref"]))) < 1e-2


def _conjunction_sat(eph: SpiceEphemeris, t: np.ndarray, offset_m: float) -> CartesianTrajectory:
    """Satellite on the far side of the Sun from the Earth, offset perpendicular to the line."""
    sun = eph.positions("SUN", t)
    earth = eph.positions("EARTH", t)
    away = sun - earth
    perp = np.cross(away, [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp, axis=1)[:, None]
    pos = sun + away + offset_m * perp
    return CartesianTrajectory(
        epochs_tdb_s=t,
        position_m=pos,
        velocity_m_s=np.zeros_like(pos),
        frame="J2000",
        origin="SOLAR SYSTEM BARYCENTER",
    )


def test_superior_conjunction_and_moon_blocking_with_spice(kernel_set: KernelSet) -> None:
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        eo = SpiceEarthOrientation(k)
        site = GroundSite.from_geodetic("eq", 0.0, 0.0, 0.0)
        t = utc_to_tdb_s("2030-03-01T00:00:00", k) + np.array([0.0, 6 * 3600.0])
        settings = AccessSettings(radii=OcculterRadii.from_kernels(k))
        behind = compute_access(
            _conjunction_sat(eph, t, 0.0),
            site,
            ephemeris=eph,
            earth_orientation=eo,
            settings=settings,
        )
        assert np.all(behind.sun_blocked)
        # 3 R_sun sideways (the beam passes ~1.5 R_sun from the centre): clear of the Sun.
        clear = compute_access(
            _conjunction_sat(eph, t, 3 * SOLAR_RADIUS_NOMINAL_M),
            site,
            ephemeris=eph,
            earth_orientation=eo,
            settings=settings,
        )
        assert not np.any(clear.sun_blocked)
        assert np.all(clear.sun_beam_min_distance_m > 1.2 * SOLAR_RADIUS_NOMINAL_M)
        # A satellite 1e10 m behind the Moon as seen from the site: the Moon blocks the beam
        # (relative Moon-site motion ~1.5 km/s x 33 s light time = 50 km << R_moon).
        st = site_trajectory(site, t, ephemeris=eph, earth_orientation=eo)
        to_moon = eph.positions("MOON", t) - st.position_m
        sat_pos = st.position_m + 1.0e10 * to_moon / np.linalg.norm(to_moon, axis=1)[:, None]
        sat = CartesianTrajectory(
            epochs_tdb_s=t,
            position_m=sat_pos,
            velocity_m_s=np.zeros_like(sat_pos),
            frame="J2000",
            origin="SOLAR SYSTEM BARYCENTER",
        )
        moon = compute_access(sat, site, ephemeris=eph, earth_orientation=eo, settings=settings)
        assert np.all(moon.moon_blocked)
        assert np.all(moon.moon_beam_min_distance_m < 0.1 * MOON_RADIUS_M)


def _mars_as_satellite(eph: SpiceEphemeris, t: np.ndarray) -> CartesianTrajectory:
    """The Mars barycentre (de440s) as a stand-in "satellite": a real J2000/SSB trajectory
    with an independently known ephemeris, so SPICE can compute the reference geometry."""
    st = eph.states("MARS BARYCENTER", t)
    return CartesianTrajectory(
        epochs_tdb_s=t,
        position_m=st.position_m,
        velocity_m_s=st.velocity_m_s,
        frame="J2000",
        origin="SOLAR SYSTEM BARYCENTER",
    )


def test_retarded_targeting_and_point_ahead_match_spice_spkcpt(kernel_set: KernelSet) -> None:
    """Independent end-to-end check of the J2000 / SSB / TDB / IAU_EARTH chain (PDF §13.3).

    Reference: SPICE ``spkcpt`` with the site as a constant-position target in IAU_EARTH and
    the Mars barycentre as the observer, aberration correction ``XCN`` (transmission case,
    converged Newtonian light time: the target at ``et + lt`` seen from the observer at
    ``et``) and ``XCN+S`` (plus the stellar aberration of transmission for the observer's SSB
    velocity). These are exactly ``light_time_s`` / ``transmit_direction_unit`` and
    ``transmit_direction_tx_frame_unit``. SPICE evaluates the site with its own Earth state
    and IAU_EARTH rotation, independently of ``site_trajectory`` and ``solve_light_time``.

    Tolerances: light time 1e-9 s (the solver tolerance; measured 4.5e-13 s); geometric
    direction 1e-12 rad (measured 2e-16 rad; float64 of positions ~1e11 m); aberration-corrected
    direction 1e-8 rad (SPICE's stellar-aberration correction is first order in beta, ours is
    the exact Lorentz formula; they differ by O(beta^2) = 1e-8 for beta = 1e-4; measured
    1.4e-9 rad), against an aberration of ~8e-5 rad.
    """
    site = GroundSite.from_geodetic(
        "probe", math.radians(35.0), math.radians(-110.0), 1500.0
    )  # arbitrary mid-latitude site
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        t = utc_to_tdb_s("2030-01-01T00:00:00", k) + np.linspace(0.0, 30 * SECONDS_PER_DAY, 240)
        sat = _mars_as_satellite(eph, t)
        acc = compute_access(
            sat,
            site,
            ephemeris=eph,
            earth_orientation=SpiceEarthOrientation(k),
            settings=AccessSettings(moon_occultation=False),
        )
        site_km = site.position_ef_m / 1000.0
        lt_ref = np.empty(t.size)
        u_ref = np.empty((t.size, 3))
        u_tx_ref = np.empty((t.size, 3))
        for j, et in enumerate(t):
            state, lt = spiceypy.spkcpt(
                site_km,
                "EARTH",
                "IAU_EARTH",
                float(et),
                "J2000",
                "OBSERVER",
                "XCN",
                "MARS BARYCENTER",
            )
            state_s, _ = spiceypy.spkcpt(
                site_km,
                "EARTH",
                "IAU_EARTH",
                float(et),
                "J2000",
                "OBSERVER",
                "XCN+S",
                "MARS BARYCENTER",
            )
            lt_ref[j] = lt
            u_ref[j] = np.asarray(state[:3]) / np.linalg.norm(state[:3])
            u_tx_ref[j] = np.asarray(state_s[:3]) / np.linalg.norm(state_s[:3])
    np.testing.assert_allclose(acc.light_time_s, lt_ref, rtol=0, atol=1e-9)
    assert float(np.max(angle_between_rad(acc.transmit_direction_unit, u_ref))) < 1e-12
    assert float(np.max(angle_between_rad(acc.transmit_direction_tx_frame_unit, u_tx_ref))) < 1e-8
    # The comparison is discriminating: the aberration itself is ~1e-4 rad and the retarded
    # vs instantaneous site differ by tens of km (~1e-7 rad at 1-2 au).
    assert float(np.min(acc.transmitter_aberration_rad)) > 1e-5
    assert float(np.min(acc.target_motion_point_ahead_rad)) > 1e-5


def test_access_from_config_handles_every_site_and_checks_the_kernel_set(
    kernel_set: KernelSet,
) -> None:
    """PDF §6.4: each configured site is computed independently, in configuration order, and
    equals compute_access for that site alone; a configuration naming another kernel set is
    refused (KernelSetMismatchError)."""
    data = load_mvp_data()
    sites = [
        {"name": "north", "latitude_deg": 40.0, "longitude_deg": -105.0, "altitude_m": 1650.0},
        {"name": "south", "latitude_deg": -30.0, "longitude_deg": 20.0, "altitude_m": 0.0},
    ]
    set_path(data, "receiver.sites", sites)
    cfg = parse_config(data)
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        t = utc_to_tdb_s("2030-03-01T00:00:00", k) + np.arange(0.0, 2 * SECONDS_PER_DAY, 1800.0)
        sat = _mars_as_satellite(eph, t)
        results = compute_access_from_config(cfg, sat, ephemeris=eph, kernels=k)
        settings = access_settings_from_config(cfg, OcculterRadii.from_kernels(k))
        alone = [
            compute_access(
                sat,
                GroundSite.from_config(site_cfg),
                ephemeris=eph,
                earth_orientation=SpiceEarthOrientation(k),
                settings=settings,
            )
            for site_cfg in cfg.receiver.sites
        ]
        # frames.ephemeris_kernel_set is a schema literal, so the mismatch is produced the
        # other way round (as in the M2 test): a handle whose manifest has another id.
        other = KernelSet(
            manifest=dataclasses.replace(kernel_set.manifest, kernel_set_id="some-other-set"),
            directory=kernel_set.directory,
        )
        foreign = dataclasses.replace(k, kernel_set=other)
        with pytest.raises(KernelSetMismatchError, match="some-other-set"):
            compute_access_from_config(cfg, sat, ephemeris=eph, kernels=foreign)
    assert [r.site_name for r in results] == ["north", "south"]
    for got, ref in zip(results, alone, strict=True):
        assert got.site_name == ref.site_name
        for name in ("light_time_s", "elevation_rad", "azimuth_rad", "range_m", "visible"):
            np.testing.assert_array_equal(getattr(got, name), getattr(ref, name))
    # The two sites really differ (a version reusing the first site would fail here).
    assert float(np.max(np.abs(results[0].elevation_rad - results[1].elevation_rad))) > 0.5
