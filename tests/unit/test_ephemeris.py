"""Tests for helioswarm.ephemeris (providers, origin changes, Hermite cache).

Kernel-free tests use :class:`AnalyticEphemeris` and closed forms. SPICE tests (marked
``spice``) compare de440s against independent references:

* JPL Horizons (DE441) state vectors, retrieved 2026-11-12 with the query in
  ``HORIZONS_QUERY`` (reproducible: re-running it returns the same numbers unless JPL changes
  its default ephemeris);
* ERFA ``epv00`` (via astropy ``solar_system_ephemeris = "builtin"``), an analytic VSOP-based
  series whose documented maximum heliocentric error versus DE405 over 1900-2100 is 11.2 km
  in position and 5.0 mm/s in velocity (``erfa.epv00`` docstring, note 4);
* Standish, "Keplerian Elements for Approximate Positions of the Major Planets" (JPL SSD,
  Table 1, EM barycentre a = 1.00000261 au, e = 0.01671123) for the annual perihelion and
  aphelion distances;
* IERS Conventions (2010), Table 1.1, for GM_Earth;
* NAIF ``aa_summaries.txt`` for the de440s coverage (1849 DEC 26 to 2150 JAN 22, TDB).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from helioswarm.constants import AU_M, GM_SUN_M3_S2, OBLIQUITY_J2000_RAD, SECONDS_PER_DAY
from helioswarm.ephemeris import (
    NAIF_IDS,
    SSB,
    SUPPORTED_BODIES,
    AnalyticEphemeris,
    BodyName,
    EphemerisCoverageError,
    EphemerisProvider,
    EphemerisState,
    HermiteEphemerisCache,
    SpiceEphemeris,
    shift_state_origin,
    shift_trajectory_origin,
)
from helioswarm.frames import (
    CartesianState,
    CartesianTrajectory,
    FrameMismatchError,
    FrameName,
    OriginName,
)
from helioswarm.kernels import KernelPoolError, KernelSet, LoadedKernels

YEAR_S = 365.25 * SECONDS_PER_DAY
ECLIPTIC_POLE_J2000 = np.array([0.0, -math.sin(OBLIQUITY_J2000_RAD), math.cos(OBLIQUITY_J2000_RAD)])

# ---------------------------------------------------------------------------------------------
# Analytic provider (no kernels)
# ---------------------------------------------------------------------------------------------


def test_analytic_sun_moves_uniformly() -> None:
    eph = AnalyticEphemeris(sun_position_at_j2000_m=(1e9, -2e9, 3e8), sun_velocity_m_s=(10, -5, 2))
    s = eph.state("SUN", 1000.0)
    np.testing.assert_array_equal(s.position_m, [1e9 + 1e4, -2e9 - 5e3, 3e8 + 2e3])
    np.testing.assert_array_equal(s.velocity_m_s, [10.0, -5.0, 2.0])
    assert (s.frame, s.origin, s.source_kernel_ids) == ("J2000", SSB, ("analytic",))


def test_analytic_earth_is_circular_ecliptic_orbit_about_the_sun() -> None:
    lam0 = 0.7
    eph = AnalyticEphemeris(
        sun_position_at_j2000_m=(5e8, 1e8, -2e8),
        sun_velocity_m_s=(12.0, -3.0, 1.0),
        earth_longitude_at_j2000_rad=lam0,
    )
    t = np.linspace(-2 * YEAR_S, 2 * YEAR_S, 201)
    earth = eph.states("EARTH", t)
    sun = eph.states("SUN", t)
    rel = earth.position_m - sun.position_m
    vrel = earth.velocity_m_s - sun.velocity_m_s
    # Radius and speed of the circular orbit (rel 1e-14: a few roundings of a closed form).
    np.testing.assert_allclose(np.linalg.norm(rel, axis=1), AU_M, rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        np.linalg.norm(vrel, axis=1), math.sqrt(GM_SUN_M3_S2 / AU_M), rtol=1e-14, atol=0
    )
    # In the ecliptic plane (normal = ecliptic pole expressed in J2000), prograde.
    assert np.max(np.abs(rel @ ECLIPTIC_POLE_J2000)) < 1e-15 * AU_M * 10
    assert np.all(np.cross(rel, vrel) @ ECLIPTIC_POLE_J2000 > 0)
    # Longitude at t = 0: rel = AU (cos lam0, sin lam0, 0) in ECLIPJ2000 -> J2000.
    e = OBLIQUITY_J2000_RAD
    expected = AU_M * np.array(
        [math.cos(lam0), math.sin(lam0) * math.cos(e), math.sin(lam0) * math.sin(e)]
    )
    r0 = eph.state("EARTH", 0.0).position_m - eph.state("SUN", 0.0).position_m
    np.testing.assert_allclose(r0, expected, rtol=0, atol=1e-15 * AU_M * 4)
    # Period 2 pi sqrt(a^3/GM): back to the same relative position (1e-12 rel: Kepler solver).
    period = 2 * math.pi * math.sqrt(AU_M**3 / GM_SUN_M3_S2)
    r_t = eph.state("EARTH", period).position_m - eph.state("SUN", period).position_m
    np.testing.assert_allclose(r_t, r0, rtol=0, atol=1e-12 * AU_M)


def test_analytic_gm_table_and_coverage() -> None:
    eph = AnalyticEphemeris()
    assert eph.gm_m3_s2("SUN") == GM_SUN_M3_S2
    with pytest.raises(KeyError, match="EARTH"):
        eph.gm_m3_s2("EARTH")
    with pytest.raises(EphemerisCoverageError, match="MOON"):
        eph.state("MOON", 0.0)
    assert isinstance(eph, EphemerisProvider)


def test_ephemeris_state_contract() -> None:
    s = EphemerisState(
        body="SUN",
        epoch_tdb_s=1.0,
        position_m=np.array([1.0, 2.0, 3.0]),
        velocity_m_s=np.array([4.0, 5.0, 6.0]),
        source_kernel_ids=("x",),
    )
    c = s.as_cartesian()
    assert (c.frame, c.origin, c.time_scale) == ("J2000", SSB, "TDB")
    with pytest.raises(ValueError, match="read-only"):
        s.position_m[0] = 0.0
    with pytest.raises(ValueError, match="J2000 / SOLAR SYSTEM BARYCENTER"):
        EphemerisState(
            body="SUN",
            epoch_tdb_s=1.0,
            position_m=np.zeros(3),
            velocity_m_s=np.zeros(3),
            source_kernel_ids=("x",),
            origin="SUN",
        )
    with pytest.raises(ValueError, match="source_kernel_ids"):
        EphemerisState(
            body="SUN",
            epoch_tdb_s=1.0,
            position_m=np.zeros(3),
            velocity_m_s=np.zeros(3),
            source_kernel_ids=(),
        )
    with pytest.raises(ValueError, match="unsupported body"):
        EphemerisState(
            body="PLUTO",  # type: ignore[arg-type]
            epoch_tdb_s=1.0,
            position_m=np.zeros(3),
            velocity_m_s=np.zeros(3),
            source_kernel_ids=("x",),
        )


def test_naif_ids_match_naif_ids_required_reading() -> None:
    assert dict(NAIF_IDS) == {
        "SUN": 10,
        "MERCURY BARYCENTER": 1,
        "VENUS BARYCENTER": 2,
        "EARTH BARYCENTER": 3,
        "MARS BARYCENTER": 4,
        "JUPITER BARYCENTER": 5,
        "SATURN BARYCENTER": 6,
        "URANUS BARYCENTER": 7,
        "NEPTUNE BARYCENTER": 8,
        "EARTH": 399,
        "MOON": 301,
    }


def test_spk_segment_chain_rule() -> None:
    # DE4xx SPKs store a planet or satellite (id >= 100) relative to its system barycentre
    # (id // 100) and barycentres / the Sun relative to the SSB, so the coverage of EARTH needs
    # segments 399 and 3. With de440s both cover the same span, so only this direct test can
    # catch a chain rule that forgets the barycentre segment.
    from helioswarm.ephemeris import _segment_chain

    assert _segment_chain(399) == (399, 3)
    assert _segment_chain(301) == (301, 3)
    assert _segment_chain(5) == (5,)
    assert _segment_chain(10) == (10,)


def test_origin_shift_round_trip_and_values() -> None:
    eph = AnalyticEphemeris(sun_position_at_j2000_m=(7e8, -1e8, 2e7), sun_velocity_m_s=(11, 2, -1))
    helio = CartesianState(
        epoch_tdb_s=500.0,
        position_m=np.array([AU_M, 0.0, 0.0]),
        velocity_m_s=np.array([0.0, 30e3, 0.0]),
        frame="J2000",
        origin="SUN",
    )
    bary = shift_state_origin(helio, SSB, eph)
    assert bary.origin == SSB
    np.testing.assert_array_equal(bary.position_m, [AU_M + 7e8 + 5500.0, -1e8 + 1000.0, 2e7 - 500])
    np.testing.assert_array_equal(bary.velocity_m_s, [11.0, 30e3 + 2.0, -1.0])
    back = shift_state_origin(bary, "SUN", eph)
    np.testing.assert_allclose(back.position_m, helio.position_m, rtol=0, atol=1e-4)
    np.testing.assert_allclose(back.velocity_m_s, helio.velocity_m_s, rtol=0, atol=1e-12)
    assert shift_state_origin(helio, "SUN", eph) is helio
    ecl = CartesianState(
        epoch_tdb_s=0.0,
        position_m=np.ones(3),
        velocity_m_s=np.ones(3),
        frame="ECLIPJ2000",
        origin="SUN",
    )
    with pytest.raises(FrameMismatchError, match="rotate"):
        shift_state_origin(ecl, SSB, eph)


def test_trajectory_origin_shift_uses_the_sun_state_at_each_epoch() -> None:
    eph = AnalyticEphemeris(sun_velocity_m_s=(100.0, 0.0, 0.0))
    t = np.array([0.0, 10.0, 20.0])
    traj = CartesianTrajectory(
        epochs_tdb_s=t,
        position_m=np.zeros((3, 3)),
        velocity_m_s=np.zeros((3, 3)),
        frame="J2000",
        origin=SSB,
    )
    helio = shift_trajectory_origin(traj, "SUN", eph)
    np.testing.assert_array_equal(helio.position_m[:, 0], [0.0, -1000.0, -2000.0])
    np.testing.assert_array_equal(helio.velocity_m_s[:, 0], [-100.0] * 3)
    assert helio.origin == "SUN"


# ---------------------------------------------------------------------------------------------
# Hermite cache (no kernels)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _CubicProvider:
    """Bodies on exact cubic polynomials r(t) = c0 + c1 t + c2 t^2 + c3 t^3 (test double)."""

    coeffs: tuple[tuple[tuple[float, float, float], ...], ...]  # per body: 4 x 3
    frame: FrameName = field(default="J2000", init=False)
    origin: OriginName = field(default=SSB, init=False)

    @property
    def source_kernel_ids(self) -> tuple[str, ...]:
        return ("cubic",)

    def _idx(self, body: BodyName) -> int:
        return ["SUN", "EARTH"].index(body)

    def states(self, body: BodyName, epochs_tdb_s: ArrayLike) -> CartesianTrajectory:
        t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=float))
        c = np.array(self.coeffs[self._idx(body)])
        pos = c[0] + np.outer(t, c[1]) + np.outer(t**2, c[2]) + np.outer(t**3, c[3])
        vel = c[1] + np.outer(2 * t, c[2]) + np.outer(3 * t**2, c[3])
        return CartesianTrajectory(
            epochs_tdb_s=t, position_m=pos, velocity_m_s=vel, frame="J2000", origin=SSB
        )

    def state(self, body: BodyName, epoch_tdb_s: float) -> EphemerisState:
        tr = self.states(body, epoch_tdb_s)
        return EphemerisState(
            body=body,
            epoch_tdb_s=epoch_tdb_s,
            position_m=tr.position_m[0],
            velocity_m_s=tr.velocity_m_s[0],
            source_kernel_ids=("cubic",),
        )

    def positions(self, body: BodyName, epochs_tdb_s: ArrayLike) -> NDArray[np.float64]:
        return np.array(self.states(body, epochs_tdb_s).position_m)

    def gm_m3_s2(self, body: BodyName) -> float:
        raise KeyError(body)

    def check_coverage(self, bodies: Iterable[BodyName], start: float, end: float) -> None:
        return None


def test_hermite_cache_is_exact_for_cubic_motion() -> None:
    # Cubic Hermite interpolation reproduces cubics exactly; only rounding remains.
    prov = _CubicProvider(
        coeffs=(
            ((1e9, -2e9, 3e8), (10.0, -4.0, 2.0), (1e-4, 2e-5, -3e-5), (1e-9, -2e-9, 5e-10)),
            ((1.5e11, 0.0, 0.0), (0.0, 3e4, 1e3), (-6e-3, 1e-3, 0.0), (2e-10, 1e-10, 3e-11)),
        )
    )
    assert isinstance(prov, EphemerisProvider)
    cache = HermiteEphemerisCache.build(prov, ("SUN", "EARTH"), 0.0, 86400.0 * 10, 7200.0)
    t = np.random.default_rng(3).uniform(0.0, 86400.0 * 10, 300)
    got = cache.positions_many(t)
    for j, body in enumerate(("SUN", "EARTH")):
        exact = prov.positions(body, t)
        scale = np.max(np.abs(exact))
        tol = 8 * np.finfo(float).eps * scale
        np.testing.assert_allclose(got[:, j, :], exact, rtol=0, atol=tol)


def test_hermite_cache_error_scales_as_h4_and_matches_the_bound() -> None:
    # Earth on a circular orbit: |x''''| = a n^4, bound h^4/384 a n^4 per component.
    eph = AnalyticEphemeris()
    n = math.sqrt(GM_SUN_M3_S2 / AU_M**3)
    t = np.random.default_rng(5).uniform(0.0, 30 * 86400.0, 400)
    exact = eph.positions("EARTH", t)
    errs = []
    for h in (4 * 86400.0, 2 * 86400.0):
        cache = HermiteEphemerisCache.build(eph, ("EARTH",), 0.0, 30 * 86400.0, h)
        err = np.max(np.abs(cache.positions_many(t)[:, 0, :] - exact))
        assert err <= h**4 / 384 * AU_M * n**4
        errs.append(err)
    # Halving h divides the error by 2^4 = 16 (allow 12-20 for the random sample maxima).
    assert 12.0 < errs[0] / errs[1] < 20.0


def test_hermite_cache_reproduces_nodes_and_rejects_outside_span() -> None:
    eph = AnalyticEphemeris()
    cache = HermiteEphemerisCache.build(eph, ("SUN", "EARTH"), 100.0, 100.0 + 5 * 3600.0, 3600.0)
    assert cache.end_tdb_s == 100.0 + 5 * 3600.0
    nodes = 100.0 + 3600.0 * np.arange(6)
    np.testing.assert_allclose(
        cache.positions_many(nodes)[:, 1, :], eph.positions("EARTH", nodes), rtol=0, atol=1e-4
    )
    with pytest.raises(ValueError, match="outside the cache span"):
        cache.positions_at(99.0)
    with pytest.raises(ValueError, match="outside the cache span"):
        cache.positions_at(100.0 + 5 * 3600.0 + 1.0)
    # The end is rounded up to whole steps and covers the requested span.
    c2 = HermiteEphemerisCache.build(eph, ("SUN",), 0.0, 3601.0, 3600.0)
    assert c2.end_tdb_s == 7200.0


# ---------------------------------------------------------------------------------------------
# SPICE (de440s)
# ---------------------------------------------------------------------------------------------

HORIZONS_QUERY = (
    "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='399'&OBJ_DATA='NO'"
    "&MAKE_EPHEM='YES'&EPHEM_TYPE='VECTORS'&CENTER='500@10'"
    "&TLIST='2451545.0','2455000.5','2460313.5','2460497.5','2462502.5','2469807.5'"
    "&TLIST_TYPE='JD'&TIME_TYPE='TDB'&REF_PLANE='FRAME'&REF_SYSTEM='ICRF'&VEC_TABLE='2'"
    "&VEC_CORR='NONE'&OUT_UNITS='KM-S'&CSV_FORMAT='YES'&VEC_LABELS='NO'"
)
"""Horizons API query (retrieved 2026-11-12): geometric Earth (399) state relative to the Sun
centre (500@10), ICRF axes, TDB Julian dates; Horizons reported 'source: DE441'."""

# JD_TDB, x, y, z [km], vx, vy, vz [km/s] copied verbatim from the Horizons output.
HORIZONS_EARTH_WRT_SUN = (
    (2451545.0, -2.649903367743050e07, 1.327574173383451e08, 5.755671847054072e07,
     -2.979426007043741e01, -5.018052308799903e00, -2.175393802830554e00),
    (2455000.5, -8.549745292681532e06, -1.392338519857522e08, -6.036206935622022e07,
     2.926251490769752e01, -1.648676845731140e00, -7.149347726599851e-01),
    (2460313.5, -3.250851613065001e07, 1.316287155512157e08, 5.705947751136197e07,
     -2.953925062110152e01, -6.133467136288075e00, -2.657462346593541e00),
    (2460497.5, 3.703680700587676e07, -1.353513670973292e08, -5.867363874248910e07,
     2.841984608807874e01, 6.554476907936581e00, 2.841138255068112e00),
    (2462502.5, -2.600847757810495e07, 1.328460643106323e08, 5.758542822344756e07,
     -2.981505192741513e01, -4.927989008272961e00, -2.137152176720780e00),
    (2469807.5, -2.567281493326828e07, 1.329033229120269e08, 5.760271197711208e07,
     -2.981226758915503e01, -4.884237994138209e00, -2.115692431521131e00),
)  # fmt: skip


def _helio(eph: SpiceEphemeris, body: BodyName, et: float) -> tuple[np.ndarray, np.ndarray]:
    b = eph.state(body, et)
    s = eph.state("SUN", et)
    return b.position_m - s.position_m, b.velocity_m_s - s.velocity_m_s


@pytest.mark.spice
@pytest.mark.parametrize("row", HORIZONS_EARTH_WRT_SUN, ids=lambda r: f"JD{r[0]}")
def test_earth_sun_state_matches_jpl_horizons(
    row: tuple[float, ...], spice_kernels: LoadedKernels
) -> None:
    # Epochs: J2000, 2009, near the 2024 perihelion (Jan 3) and aphelion (Jul 5), 2030, 2050.
    # Tolerances 1 m and 1e-6 m/s: DE440 and DE441 are fits to the same data and differ by
    # < 6 cm here (measured); Horizons prints 16 significant digits (1e-8 km). An error in
    # units (km vs m), body id, origin (SSB vs Sun) or time scale (1 s = 30 km) fails by orders
    # of magnitude.
    eph = SpiceEphemeris(spice_kernels)
    et = (row[0] - 2451545.0) * SECONDS_PER_DAY
    r, v = _helio(eph, "EARTH", et)
    ref_r = np.array(row[1:4]) * 1e3
    ref_v = np.array(row[4:7]) * 1e3
    assert np.linalg.norm(r - ref_r) < 1.0
    assert np.linalg.norm(v - ref_v) < 1e-6
    assert abs(np.linalg.norm(r) - np.linalg.norm(ref_r)) < 1.0


@pytest.mark.spice
def test_earth_sun_state_agrees_with_erfa_epv00(spice_kernels: LoadedKernels) -> None:
    # Independent analytic theory over 1900-2100 (41 epochs). Tolerances are epv00's documented
    # maximum heliocentric errors vs DE405 (11.2 km, 5.0 mm/s); measured 7.1 km and 3.1 mm/s.
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
    from astropy.time import Time

    eph = SpiceEphemeris(spice_kernels)
    worst_r = worst_v = 0.0
    for jd in np.linspace(2451545.0 - 36525.0, 2451545.0 + 36500.0, 41):
        t = Time(jd, format="jd", scale="tdb")
        with solar_system_ephemeris.set("builtin"):
            pe, ve = get_body_barycentric_posvel("earth", t)
            ps, vs = get_body_barycentric_posvel("sun", t)
        ref_r = (pe.xyz - ps.xyz).to(u.m).value
        ref_v = (ve.xyz - vs.xyz).to(u.m / u.s).value
        r, v = _helio(eph, "EARTH", (jd - 2451545.0) * SECONDS_PER_DAY)
        worst_r = max(worst_r, float(np.linalg.norm(r - ref_r)))
        worst_v = max(worst_v, float(np.linalg.norm(v - ref_v)))
    assert worst_r < 11.2e3
    assert worst_v < 5.0e-3


def _yearly_extreme(
    eph: SpiceEphemeris, body: BodyName, year: int, month: int, kind: str
) -> tuple[float, float]:
    """(epoch, distance [au]) of the min/max heliocentric distance in a +-25 d window around
    the 4th of ``month``, hourly sampling refined by a parabola through the 3 best samples."""
    centre = (datetime(year, month, 4) - datetime(2000, 1, 1, 12)).total_seconds()
    t = centre + np.arange(-25 * 24, 25 * 24 + 1) * 3600.0
    d = np.linalg.norm(eph.positions(body, t) - eph.positions("SUN", t), axis=1) / AU_M
    k = int(np.argmin(d) if kind == "min" else np.argmax(d))
    assert 0 < k < len(t) - 1, "extremum at the window edge"
    y0, y1, y2 = d[k - 1], d[k], d[k + 1]
    denom = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / denom
    return t[k] + off * 3600.0, y1 - 0.25 * (y0 - y2) * off


@pytest.mark.spice
def test_annual_perihelion_and_aphelion_distances(spice_kernels: LoadedKernels) -> None:
    # Mean EMB elements (Standish, JPL SSD Table 1): q = a(1-e) = 0.98329 au, Q = a(1+e) =
    # 1.01671 au. The Earth's yearly extremes scatter around these by the lunar offset of the
    # Earth from the EMB (4.7e6 m = 3.1e-5 au) plus planetary perturbations (EMB alone
    # 0.98325-0.98333 au); tolerance 1.2e-4 au. Dates must fall early January / early July.
    eph = SpiceEphemeris(spice_kernels)
    a, e = 1.00000261, 0.01671123
    for year in range(2000, 2040):
        tp, q = _yearly_extreme(eph, "EARTH", year, 1, "min")
        ta, big_q = _yearly_extreme(eph, "EARTH", year, 7, "max")
        assert q == pytest.approx(a * (1 - e), abs=1.2e-4)
        assert big_q == pytest.approx(a * (1 + e), abs=1.2e-4)
        jan = (datetime(year, 1, 1) - datetime(2000, 1, 1, 12)).total_seconds()
        jul = (datetime(year, 7, 1) - datetime(2000, 1, 1, 12)).total_seconds()
        assert 0.0 < (tp - jan) / SECONDS_PER_DAY < 6.5
        assert 0.0 < (ta - jul) / SECONDS_PER_DAY < 7.5


@pytest.mark.spice
def test_emb_perihelion_period_is_the_anomalistic_year(spice_kernels: LoadedKernels) -> None:
    # Mean perihelion-to-perihelion interval of the Earth-Moon barycentre over 2000-2039 vs the
    # anomalistic year 365.259636 d (Explanatory Supplement to the Astronomical Almanac). The
    # perihelion time jitters by ~0.09 d (rms) from planetary perturbations, so the mean of 39
    # intervals is uncertain by ~0.09*sqrt(2)/39 = 0.003 d; tolerance 0.01 d. A seconds/days
    # slip or a wrong body (Earth's lunar wobble moves perihelion by +-1.6 d) is caught.
    eph = SpiceEphemeris(spice_kernels)
    times = [_yearly_extreme(eph, "EARTH BARYCENTER", y, 1, "min")[0] for y in range(2000, 2040)]
    mean_interval_d = (times[-1] - times[0]) / (len(times) - 1) / SECONDS_PER_DAY
    assert mean_interval_d == pytest.approx(365.259636, abs=0.01)


@pytest.mark.spice
def test_gm_values_from_gm_de440(spice_kernels: LoadedKernels) -> None:
    eph = SpiceEphemeris(spice_kernels)
    gm_sun = eph.gm_m3_s2("SUN")
    # Same DE440 number as constants.GM_SUN_M3_S2 (BODY10_GM = 1.3271244004127942E+11 km^3/s^2).
    # SPICE's text-kernel parser is not correctly rounded: it returns the float 2 ulp below the
    # nearest double of the decimal (measured), so agreement is to <= 2 ulp (2.5e-16 rel). The
    # kernel value is the one used whenever SPICE is used (docs/model_assumptions.md, M2).
    assert abs(gm_sun - GM_SUN_M3_S2) <= 2 * np.spacing(GM_SUN_M3_S2)
    # GM_Earth: IERS Conventions 2010 Table 1.1, 3.986004418e14 m^3/s^2 (TT-compatible,
    # +-8e5); TDB-compatible value = GM_TT (1 - L_B), L_B = 1.550519768e-8 (IAU 2006 B3).
    gm_earth_tdb = 3.986004418e14 * (1.0 - 1.550519768e-8)
    assert eph.gm_m3_s2("EARTH") == pytest.approx(gm_earth_tdb, rel=3e-9, abs=0)
    # Internal consistency: BODY3_GM (Earth-Moon barycentre) = BODY399_GM + BODY301_GM.
    total = eph.gm_m3_s2("EARTH") + eph.gm_m3_s2("MOON")
    assert total == pytest.approx(eph.gm_m3_s2("EARTH BARYCENTER"), rel=1e-15, abs=0)


@pytest.mark.spice
def test_earth_moon_barycentre_identity(spice_kernels: LoadedKernels) -> None:
    # r_EMB = (GM_E r_E + GM_M r_M)/(GM_E + GM_M): ties NAIF ids 3, 399 and 301 and the GM
    # values together (a swapped id or km/m slip fails by >= 1e6 m). Measured 3e-5 m over
    # 1905-2095; tolerance 1e-3 m.
    eph = SpiceEphemeris(spice_kernels)
    ge, gm = eph.gm_m3_s2("EARTH"), eph.gm_m3_s2("MOON")
    t = np.linspace(-3e9, 3e9, 51)
    rb = (ge * eph.positions("EARTH", t) + gm * eph.positions("MOON", t)) / (ge + gm)
    assert np.max(np.linalg.norm(rb - eph.positions("EARTH BARYCENTER", t), axis=1)) < 1e-3


@pytest.mark.spice
def test_spice_states_carry_provenance_and_contract(spice_kernels: LoadedKernels) -> None:
    eph = SpiceEphemeris(spice_kernels)
    assert isinstance(eph, EphemerisProvider)
    s = eph.state("JUPITER BARYCENTER", 1e8)
    assert s.source_kernel_ids == ("de440s.bsp",)
    assert (s.frame, s.origin, s.time_scale) == ("J2000", SSB, "TDB")
    traj = eph.states("MOON", np.array([0.0, 1e6]))
    np.testing.assert_array_equal(traj.position_m[1], eph.state("MOON", 1e6).position_m)
    # Jupiter's heliocentric distance 4.95-5.46 au (orbital elements), in metres.
    r = np.linalg.norm(s.position_m - eph.state("SUN", 1e8).position_m) / AU_M
    assert 4.9 < r < 5.5


@pytest.mark.spice
def test_coverage_matches_naif_summary_and_out_of_range_is_refused(
    spice_kernels: LoadedKernels,
) -> None:
    eph = SpiceEphemeris(spice_kernels)
    j2000 = datetime(2000, 1, 1, 12)
    start = (datetime(1849, 12, 26) - j2000).total_seconds()
    end = (datetime(2150, 1, 22) - j2000).total_seconds()
    for body in SUPPORTED_BODIES:
        assert eph.coverage(body) == ((start, end),)
    with pytest.raises(EphemerisCoverageError, match="EARTH: requested TDB span"):
        eph.state("EARTH", start - 1.0)
    with pytest.raises(EphemerisCoverageError, match="MOON"):
        eph.states("MOON", np.array([0.0, end + 86400.0]))
    with pytest.raises(EphemerisCoverageError):
        HermiteEphemerisCache.build(eph, ("SUN", "EARTH"), end - 3600.0, end + 3600.0, 3600.0)
    eph.check_coverage(SUPPORTED_BODIES, start, end)  # the closed span itself is accepted


@pytest.mark.spice
def test_hermite_cache_vs_direct_spice_below_one_metre(spice_kernels: LoadedKernels) -> None:
    # Default cache step 3600 s over the MVP year for every supported body. Measured maxima:
    # Mercury barycentre 0.06 m, Moon 0.015 m, others <= 2e-3 m (outer planets: float64
    # rounding of 4.5e12 m). Requirement: < 1 m.
    eph = SpiceEphemeris(spice_kernels)
    t0 = 946728069.183919  # 2030-01-01T00:00:00 UTC in TDB s (see test_timescale)
    cache = HermiteEphemerisCache.build(eph, SUPPORTED_BODIES, t0, t0 + YEAR_S, 3600.0)
    t = t0 + np.random.default_rng(11).uniform(0.0, YEAR_S, 400)
    interp = cache.positions_many(t)
    for j, body in enumerate(SUPPORTED_BODIES):
        err = np.max(np.linalg.norm(interp[:, j, :] - eph.positions(body, t), axis=1))
        assert err < 1.0, (body, err)


@pytest.mark.spice
def test_spice_provider_refuses_use_outside_the_loaded_scope(kernel_set: KernelSet) -> None:
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        eph.state("SUN", 0.0)
    with pytest.raises(KernelPoolError):
        eph.state("SUN", 0.0)
    with pytest.raises(KernelPoolError):
        eph.gm_m3_s2("SUN")
    with pytest.raises(KernelPoolError):
        SpiceEphemeris(k)
