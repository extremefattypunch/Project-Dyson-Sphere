"""Independent cross-check of the L1 propagator against REBOUND IAS15 (PDF §6.2 level L2,
§13.3 "compare selected orbital trajectories against REBOUND").

REBOUND API used (verified in the installed rebound 5.2.2 source/docstrings): ``Simulation()``,
``sim.G``, ``sim.integrator = "ias15"`` (adaptive 15th-order, default epsilon 1e-9),
``sim.dt`` (initial step), ``sim.add(m=, x=, y=, z=, vx=, vy=, vz=)``, ``sim.N_active`` and
``sim.testparticle_type = 0`` (massless particles do not act on massive ones),
``sim.integrate(t, exact_finish_time=1)``, ``sim.particles[i].x``. Units are SI with
``G = 1`` and masses set to GM [m^3 s^-2].

Why the two differ in the full model: REBOUND evolves the Sun and the nine bodies
self-consistently under Newtonian mutual gravity from the SPICE initial states, whereas L1
prescribes them from DE440, which also contains general relativity, ~340 asteroids, Pluto
and lunar/terrestrial figure effects. REBOUND's bodies therefore drift from DE440 (measured
over one year: Mercury barycentre 5.8e5 m, Earth 6.1e4 m, Sun 2.6e2 m). The satellite's
heliocentric motion is only weakly sensitive to those planet-position errors (tidal terms), so
the residual is ~1 m. L1's heliocentric formulation and REBOUND share the same physics for the
satellite relative to the Sun (direct and indirect terms of the same nine bodies); the L1
barycentric comparison path, which also applies the Sun's DE440 acceleration from bodies it
does not model, differs from REBOUND by 5.7e2 m (measured).
"""

from __future__ import annotations

import numpy as np
import pytest

from helioswarm.config import OrbitConfig, load_config
from helioswarm.constants import GM_SUN_M3_S2, SECONDS_PER_DAY
from helioswarm.ephemeris import AnalyticEphemeris, EphemerisProvider, SpiceEphemeris
from helioswarm.frames import transform_trajectory
from helioswarm.kernels import LoadedKernels
from helioswarm.orbit import elements_from_config, propagate_kepler
from helioswarm.propagate import ForceModel, IntegratorSettings, SatelliteState, propagate_l1
from helioswarm.propagate import satellite_initial_state as build_initial
from helioswarm.timescale import utc_to_tdb_s

from config_helpers import MVP_CONFIG

rebound = pytest.importorskip("rebound")

pytestmark = pytest.mark.xcheck

YEAR_S = 365.25 * SECONDS_PER_DAY
DAY_CAP = IntegratorSettings(max_step_s=SECONDS_PER_DAY)


def _rebound_run(
    ephemeris: EphemerisProvider, init: SatelliteState, epochs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Satellite positions (N, 3) and body positions (N, n_bodies, 3), SSB-origin J2000 [m]."""
    bodies = init.force_model.bodies
    t0 = init.epoch_tdb_s
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    sim.dt = 3600.0
    for body in bodies:
        s = ephemeris.state(body, t0)
        sim.add(
            m=ephemeris.gm_m3_s2(body),
            x=s.position_m[0], y=s.position_m[1], z=s.position_m[2],
            vx=s.velocity_m_s[0], vy=s.velocity_m_s[1], vz=s.velocity_m_s[2],
        )  # fmt: skip
    r, v = init.position_m, init.velocity_m_s
    sim.add(m=0.0, x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])
    sim.N_active = len(bodies)
    sim.testparticle_type = 0
    sat, body_pos = [], []
    for t in epochs:
        sim.integrate(float(t - t0), exact_finish_time=1)
        p = sim.particles
        sat.append([p[len(bodies)].x, p[len(bodies)].y, p[len(bodies)].z])
        body_pos.append([[p[i].x, p[i].y, p[i].z] for i in range(len(bodies))])
    return np.array(sat), np.array(body_pos)


def _no_srp(orbit_cfg: OrbitConfig) -> OrbitConfig:
    return orbit_cfg.model_copy(
        update={"include_solar_radiation_pressure": False, "cr": None, "area_to_mass_m2_kg": None}
    )


def test_two_body_rebound_matches_l1() -> None:
    # Sun + test particle only (analytic Sun at rest, MVP circular 1 au orbit), one year. Both
    # integrators are accurate to << 1 m here; measured 7.7e-4 m (L1's own error with the
    # 1-day cap). Tolerance 1e-2 m.
    cfg = load_config(MVP_CONFIG)
    orbit = _no_srp(cfg.orbit).model_copy(update={"include_planetary_perturbations": False})
    eph = AnalyticEphemeris()
    init = build_initial(orbit, cfg.constants, epoch_tdb_s=9.467e8, ephemeris=eph)
    t = init.epoch_tdb_s + np.linspace(0.0, YEAR_S, 37)
    l1 = propagate_l1(init, t, ephemeris=eph, settings=DAY_CAP).trajectory.position_m
    sat, bodies = _rebound_run(eph, init, t)
    helio_reb = sat - bodies[:, 0, :]
    helio_l1 = l1 - eph.positions("SUN", t)
    assert np.max(np.linalg.norm(helio_reb - helio_l1, axis=1)) < 1e-2
    assert eph.gm_m3_s2("SUN") == GM_SUN_M3_S2


@pytest.mark.spice
def test_full_nbody_rebound_matches_l1_within_ephemeris_model_differences(
    spice_kernels: LoadedKernels,
) -> None:
    # Sun + 9 bodies + satellite, initialised from the same de440s states and gm_de440 GMs,
    # one year from the MVP epoch. Measured: satellite heliocentric difference 1.03 m (max);
    # REBOUND's Jupiter stays within 491 m of DE440. Tolerances: 10 m for the satellite (~10x
    # measured; explained in the module docstring) and 2 km for Jupiter (a wrong GM or initial
    # state would fail by orders of magnitude).
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    t0 = utc_to_tdb_s(cfg.orbit.initial_epoch_utc, spice_kernels)
    orbit = _no_srp(cfg.orbit)
    init = build_initial(orbit, cfg.constants, epoch_tdb_s=t0, ephemeris=eph)
    assert len(init.force_model.bodies) == 10
    t = t0 + np.linspace(0.0, YEAR_S, 37)
    sun = eph.positions("SUN", t)
    l1 = propagate_l1(init, t, ephemeris=eph, settings=DAY_CAP).trajectory.position_m
    sat, bodies = _rebound_run(eph, init, t)
    helio_reb = sat - bodies[:, 0, :]
    assert np.max(np.linalg.norm(helio_reb - (l1 - sun), axis=1)) < 10.0
    j = init.force_model.bodies.index("JUPITER BARYCENTER")
    jup = np.linalg.norm(bodies[:, j, :] - eph.positions("JUPITER BARYCENTER", t), axis=1)
    assert np.max(jup) < 2.0e3
    # Scale: the planetary part of the force model moves the satellite by 3.25e7 m from L0
    # Kepler over this year (measured; asserted > 1e7 m so the figure cannot go stale), so the
    # cross-check resolves it to ~3e-8.
    el = elements_from_config(orbit, epoch_tdb_s=t0, mu_m3_s2=eph.gm_m3_s2("SUN"))
    l0 = transform_trajectory(propagate_kepler(el, t).trajectory, "J2000").position_m
    assert np.max(np.linalg.norm(l1 - sun - l0, axis=1)) > 1e7
    # The barycentric comparison path carries the Sun's unmodelled DE440 acceleration (566 m
    # measured, see test_l1_propagation): REBOUND must NOT agree with it at the 10 m level.
    bary = propagate_l1(
        init, t, ephemeris=eph, settings=DAY_CAP, formulation="barycentric"
    ).trajectory.position_m
    assert np.max(np.linalg.norm(helio_reb - (bary - sun), axis=1)) > 1e2


@pytest.mark.spice
def test_rebound_resolves_the_effect_of_dropping_neptune(spice_kernels: LoadedKernels) -> None:
    # Discrimination with a physically meaningful body subset: Sun + 8 bodies (no Neptune) in
    # both codes. Dropping Neptune changes L1 by 2.9e3 m over the year (its tidal effect at
    # 1 au, measured); REBOUND without Neptune agrees with L1 without Neptune to 1.4 m
    # (measured). Tolerance 10 m, i.e. the cross-check resolves Neptune's effect ~300x over.
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    t0 = utc_to_tdb_s(cfg.orbit.initial_epoch_utc, spice_kernels)
    full = build_initial(_no_srp(cfg.orbit), cfg.constants, epoch_tdb_s=t0, ephemeris=eph)
    no_nep = tuple(b for b in full.force_model.bodies if b != "NEPTUNE BARYCENTER")
    init = SatelliteState(state=full.state, force_model=ForceModel(no_nep))
    t = t0 + np.linspace(0.0, YEAR_S, 37)
    sun = eph.positions("SUN", t)
    l1_full = propagate_l1(full, t, ephemeris=eph, settings=DAY_CAP).trajectory.position_m
    l1 = propagate_l1(init, t, ephemeris=eph, settings=DAY_CAP).trajectory.position_m
    sat, bodies = _rebound_run(eph, init, t)
    assert np.max(np.linalg.norm((sat - bodies[:, 0, :]) - (l1 - sun), axis=1)) < 10.0
    assert np.max(np.linalg.norm(l1 - l1_full, axis=1)) > 1e3
