"""Tests for helioswarm.propagate (L1 test-particle propagator, PDF §6.2, §7.4, §13.1, §13.3).

Closed-form references:

* L0 Kepler propagation (M1, validated separately) for the Sun-only model, both with a
  uniformly moving analytic Sun and with the real (accelerated) DE440 Sun: the heliocentric
  formulation has consistent direct and indirect terms, so the Sun-only model is exactly
  two-body motion whatever the Sun's SSB trajectory;
* the same with GM_sun -> GM_sun - k for radial inverse-square SRP (exact equivalence);
* the algebraic identity heliocentric acceleration = barycentric acceleration minus the Sun's
  Newtonian acceleration under the same bodies;
* the circular-orbit SRP drift estimate ``Delta s ~ 2 beta n t a``
  (:func:`helioswarm.srp.srp_along_track_drift_estimate_m`, derived there).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from helioswarm.config import load_config
from helioswarm.constants import AU_M, GM_SUN_M3_S2, OBLIQUITY_J2000_RAD, SECONDS_PER_DAY
from helioswarm.ephemeris import (
    SSB,
    AnalyticEphemeris,
    EphemerisCoverageError,
    HermiteEphemerisCache,
    SpiceEphemeris,
    shift_state_origin,
    shift_trajectory_origin,
)
from helioswarm.frames import CartesianState, transform_state, transform_trajectory
from helioswarm.kernels import KernelSet, LoadedKernels
from helioswarm.orbit import (
    OrbitalElements,
    elements_from_config,
    elements_to_state,
    propagate_kepler,
)
from helioswarm.propagate import (
    ForceModel,
    IntegratorSettings,
    KernelSetMismatchError,
    L1Propagation,
    SatelliteState,
    assess_l1_run,
    check_kernel_set,
    effective_gm_m3_s2,
    force_model_from_config,
    heliocentric_l1_acceleration_m_s2,
    initial_state_from_config,
    integrator_settings_from_config,
    l1_acceleration_m_s2,
    point_mass_acceleration_m_s2,
    propagate_l1,
    propagate_l1_from_config,
    satellite_initial_state,
)
from helioswarm.srp import SrpModel, srp_along_track_drift_estimate_m
from helioswarm.timescale import time_grid_from_config, utc_to_tdb_s

from config_helpers import MVP_CONFIG

YEAR_S = 365.25 * SECONDS_PER_DAY
DAY_CAP = IntegratorSettings(max_step_s=SECONDS_PER_DAY)
MOVING_SUN = AnalyticEphemeris(
    sun_position_at_j2000_m=(7.0e8, -3.0e8, 1.0e8), sun_velocity_m_s=(12.0, -8.0, 3.0)
)
ELEMENTS = OrbitalElements(
    a_m=AU_M,
    e=0.3,
    i_rad=math.radians(20.0),
    raan_rad=1.0,
    argp_rad=2.0,
    nu_rad=0.5,
    mu_m3_s2=GM_SUN_M3_S2,
    epoch_tdb_s=9.0e8,
)
MVP_SRP = SrpModel.from_solar_constant(
    cr=1.3, area_to_mass_m2_kg=0.02, solar_constant_1au_W_m2=1361.0
)


def _sun_only_initial(eph: AnalyticEphemeris, srp: SrpModel | None = None) -> SatelliteState:
    helio = transform_state(elements_to_state(ELEMENTS), "J2000")
    return SatelliteState(
        state=shift_state_origin(helio, SSB, eph), force_model=ForceModel(("SUN",), srp)
    )


def _helio_l1(prop: L1Propagation, eph: AnalyticEphemeris | SpiceEphemeris) -> np.ndarray:
    return np.asarray(shift_trajectory_origin(prop.trajectory, "SUN", eph).position_m)


# ---------------------------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------------------------


def test_point_mass_acceleration_closed_form() -> None:
    r = np.array([2.0e11, 0.0, 0.0])
    a = point_mass_acceleration_m_s2(r, np.zeros((1, 3)), np.array([GM_SUN_M3_S2]))
    np.testing.assert_allclose(a, [-GM_SUN_M3_S2 / 4.0e22, 0.0, 0.0], rtol=1e-15, atol=0)
    # Two equal masses placed symmetrically about the particle cancel exactly.
    bodies = np.array([[1e11, 5e10, 0.0], [-1e11, -5e10, 0.0]])
    np.testing.assert_allclose(
        point_mass_acceleration_m_s2(np.zeros(3), bodies, np.array([1e20, 1e20])),
        np.zeros(3),
        rtol=0,
        atol=1e-18,
    )


def test_integrator_rhs_folding_of_srp_equals_the_reference_form() -> None:
    # The integrator folds radial SRP into the Sun's GM; verify the identity on random inputs.
    rng = np.random.default_rng(4)
    gms = np.array([GM_SUN_M3_S2, 1.27e17, 4.0e14])
    for _ in range(50):
        r = rng.normal(size=3) * AU_M
        bodies = rng.normal(size=(3, 3)) * 5 * AU_M
        ref = l1_acceleration_m_s2(r, bodies, gms, sun_index=0, srp=MVP_SRP)
        folded = l1_acceleration_m_s2(
            r, bodies, effective_gm_m3_s2(gms, sun_index=0, srp=MVP_SRP), sun_index=0, srp=None
        )
        np.testing.assert_allclose(folded, ref, rtol=1e-13, atol=0)
    # Without SRP the GMs are unchanged (copy, not alias).
    out = effective_gm_m3_s2(gms, sun_index=0, srp=None)
    np.testing.assert_array_equal(out, gms)
    assert out is not gms


def test_heliocentric_acceleration_is_barycentric_minus_the_suns_modelled_acceleration() -> None:
    # PDF §6.2: in a heliocentric frame the correct indirect term is the Sun's acceleration
    # under the *same* bodies. Identity check on random geometry (Sun first): a_helio(r - r_sun)
    # = a_bary(r) - sum_j mu_j (r_j - r_sun)/|r_j - r_sun|^3. Tolerance 1e-12 relative to the
    # largest term (float64 cancellation of the ~6e-3 m/s^2 solar term in the barycentric form).
    rng = np.random.default_rng(11)
    gms = np.array([GM_SUN_M3_S2, 1.27e17, 3.8e16, 4.0e14])
    for _ in range(50):
        bodies = rng.normal(size=(4, 3)) * 5 * AU_M
        r = bodies[0] + rng.normal(size=3) * AU_M
        bary = l1_acceleration_m_s2(r, bodies, gms, sun_index=0, srp=MVP_SRP)
        a_sun = point_mass_acceleration_m_s2(bodies[0], bodies[1:], gms[1:])
        helio = heliocentric_l1_acceleration_m_s2(
            r - bodies[0], bodies[1:] - bodies[0], gms[1:], gm_sun_m3_s2=gms[0], srp=MVP_SRP
        )
        np.testing.assert_allclose(helio, bary - a_sun, rtol=0, atol=1e-12 * np.linalg.norm(bary))
    # Sun only: exactly the two-body term with mu - k.
    r = np.array([AU_M, 0.0, 0.0])
    sun_only = heliocentric_l1_acceleration_m_s2(
        r, np.empty((0, 3)), np.empty(0), gm_sun_m3_s2=GM_SUN_M3_S2, srp=None
    )
    np.testing.assert_allclose(sun_only, [-GM_SUN_M3_S2 / AU_M**2, 0, 0], rtol=1e-15, atol=0)


# ---------------------------------------------------------------------------------------------
# Sun-only L1 vs L0 (kernel-free)
# ---------------------------------------------------------------------------------------------


def test_sun_only_l1_reproduces_l0_kepler_over_one_year() -> None:
    # PDF §7.4 "circular two-body orbit preserves radius and energy within the integrator
    # tolerance" and plan item "L1 with planets off == L0". e = 0.3, inclined, Sun offset from
    # the SSB and moving at 15 m/s: the comparison is done heliocentrically (origin shift with
    # the Sun's state). Measured 6.7e-4 m and energy drift 3.8e-15 with rtol 1e-12 and a 1-day
    # step cap; tolerances 1e-2 m and 1e-13. A frame/origin slip or a GM error of 1e-12
    # relative (~1 m along-track) fails.
    init = _sun_only_initial(MOVING_SUN)
    t = ELEMENTS.epoch_tdb_s + np.linspace(0.0, YEAR_S, 366)
    prop = propagate_l1(init, t, ephemeris=MOVING_SUN, settings=DAY_CAP)
    ref = transform_trajectory(propagate_kepler(ELEMENTS, t).trajectory, "J2000")
    err = np.linalg.norm(_helio_l1(prop, MOVING_SUN) - ref.position_m, axis=1)
    assert err.max() < 1e-2
    assert prop.diagnostics.sun_two_body_energy_rel_drift < 1e-13
    assert prop.diagnostics.success
    assert prop.trajectory.frame == "J2000"
    assert prop.trajectory.origin == SSB


def test_circular_orbit_radius_is_preserved_within_integrator_tolerance() -> None:
    # PDF §7.4 L1 acceptance item (the open row of docs/validation_report.md): a circular
    # 1 au orbit integrated by L1 (Sun-only, Sun at rest) keeps r = a over 3 years. Measured
    # max |r/a - 1| 4.7e-15 and energy drift 5.6e-15 with the 1-day cap; tolerance 1e-12
    # (= rtol).
    eph = AnalyticEphemeris()
    el = OrbitalElements(
        a_m=AU_M, e=0.0, i_rad=0.3, raan_rad=0.2, argp_rad=0.0, nu_rad=0.0,
        mu_m3_s2=GM_SUN_M3_S2, epoch_tdb_s=0.0,
    )  # fmt: skip
    st = transform_state(elements_to_state(el), "J2000")
    init = SatelliteState(state=shift_state_origin(st, SSB, eph), force_model=ForceModel(("SUN",)))
    prop = propagate_l1(init, np.linspace(0.0, 3 * YEAR_S, 500), ephemeris=eph, settings=DAY_CAP)
    r = np.linalg.norm(prop.trajectory.position_m, axis=1)
    assert np.max(np.abs(r / AU_M - 1.0)) < 1e-12
    assert prop.diagnostics.sun_two_body_energy_rel_drift < 1e-12


def test_error_converges_with_rtol_when_steps_are_uncapped() -> None:
    # Without a step cap the error is set by rtol: measured 130 m, 13 m, 1.5 m, 0.19 m for
    # rtol 1e-10 ... 1e-13 (one year, e = 0.3). Require a >= 5x decrease per decade and
    # < 10 m at rtol 1e-12.
    init = _sun_only_initial(MOVING_SUN)
    t = ELEMENTS.epoch_tdb_s + np.linspace(0.0, YEAR_S, 50)
    ref = transform_trajectory(propagate_kepler(ELEMENTS, t).trajectory, "J2000").position_m
    errs = []
    for rtol in (1e-11, 1e-12, 1e-13):
        prop = propagate_l1(init, t, ephemeris=MOVING_SUN, settings=IntegratorSettings(rtol=rtol))
        errs.append(float(np.max(np.linalg.norm(_helio_l1(prop, MOVING_SUN) - ref, axis=1))))
    assert errs[0] > 5 * errs[1] > 25 * errs[2]
    assert errs[1] < 10.0


def test_radial_srp_equals_kepler_with_reduced_solar_gm() -> None:
    # Inverse-square radial SRP is equivalent to GM_sun -> GM_sun - k (k = P AU^2 Cr A/m), so
    # L1 Sun-only + SRP must equal L0 with mu_eff: measured 5.8e-3 m with the 1-day cap at
    # these 100 output epochs (7th-order dense output between 1-day steps); tolerance 3e-2 m.
    # The SRP effect itself (vs mu = GM_sun) is 8.1e7 m for this orbit, so a sign error or a
    # missing SRP term fails by > 1e9x.
    init = _sun_only_initial(MOVING_SUN, srp=MVP_SRP)
    t = ELEMENTS.epoch_tdb_s + np.linspace(0.0, YEAR_S, 100)
    prop = propagate_l1(init, t, ephemeris=MOVING_SUN, settings=DAY_CAP)
    helio0 = transform_state(elements_to_state(ELEMENTS), "J2000")
    mu_eff = GM_SUN_M3_S2 - MVP_SRP.strength_m3_s2
    ref = propagate_kepler(helio0, t, mu_m3_s2=mu_eff).trajectory.position_m
    no_srp = propagate_kepler(helio0, t, mu_m3_s2=GM_SUN_M3_S2).trajectory.position_m
    got = _helio_l1(prop, MOVING_SUN)
    assert np.max(np.linalg.norm(got - ref, axis=1)) < 3e-2
    assert np.max(np.linalg.norm(got - no_srp, axis=1)) > 1e7
    # With SRP the conserved energy uses mu_eff, and the diagnostic accounts for it.
    assert prop.diagnostics.sun_two_body_energy_rel_drift < 1e-13


def test_backward_forward_integration_and_initial_epoch_output() -> None:
    init = _sun_only_initial(MOVING_SUN)
    t0 = init.epoch_tdb_s
    t = t0 + np.array([-180.0, -1.0, 0.0, 90.0, 180.0]) * SECONDS_PER_DAY
    prop = propagate_l1(init, t, ephemeris=MOVING_SUN, settings=DAY_CAP)
    assert [s.direction for s in prop.diagnostics.segments] == ["forward", "backward"]
    assert [s.n_output for s in prop.diagnostics.segments] == [2, 2]
    assert prop.diagnostics.nfev == sum(s.nfev for s in prop.diagnostics.segments)
    np.testing.assert_array_equal(prop.trajectory.position_m[2], init.position_m)
    np.testing.assert_array_equal(prop.trajectory.velocity_m_s[2], init.velocity_m_s)
    ref = transform_trajectory(propagate_kepler(ELEMENTS, t).trajectory, "J2000").position_m
    assert np.max(np.linalg.norm(_helio_l1(prop, MOVING_SUN) - ref, axis=1)) < 1e-2
    # Reversibility: integrate the +180 d state back to t0 (measured 7.3e-4 m, 2.0e-10 m/s;
    # Kepler agreement above measured 1.8e-4 m).
    end = SatelliteState(state=prop.trajectory.state_at(4), force_model=init.force_model)
    back = propagate_l1(end, np.array([t0]), ephemeris=MOVING_SUN, settings=DAY_CAP)
    assert np.linalg.norm(back.trajectory.position_m[0] - init.position_m) < 1e-2
    assert np.linalg.norm(back.trajectory.velocity_m_s[0] - init.velocity_m_s) < 1e-8


def test_input_validation() -> None:
    init = _sun_only_initial(MOVING_SUN)
    with pytest.raises(ValueError, match="strictly increasing"):
        propagate_l1(init, np.array([1.0, 1.0]), ephemeris=MOVING_SUN, settings=DAY_CAP)
    with pytest.raises(ValueError, match="must contain 'SUN'"):
        ForceModel(("EARTH",))
    with pytest.raises(ValueError, match="duplicates"):
        ForceModel(("SUN", "EARTH", "EARTH"))
    with pytest.raises(ValueError, match="double-counts"):
        ForceModel(("SUN", "EARTH", "EARTH BARYCENTER"))
    with pytest.raises(ValueError, match="rtol"):
        IntegratorSettings(rtol=1e-14)
    with pytest.raises(ValueError, match="rtol"):
        IntegratorSettings(rtol=1e-5)  # same upper bound as the config schema (1e-6)
    with pytest.raises(ValueError, match="validated range"):
        IntegratorSettings(ephemeris_cache_step_s=7201.0)
    # The barycentric comparison path refuses incomplete planetary sets (review finding M2).
    with pytest.raises(ValueError, match="incomplete body set"):
        propagate_l1(
            init,
            np.array([init.epoch_tdb_s + 10.0]),
            ephemeris=MOVING_SUN,
            settings=DAY_CAP,
            formulation="barycentric",
        )
    assert not ForceModel(("SUN", "JUPITER BARYCENTER")).has_complete_planetary_set
    full = ("SUN", "MERCURY BARYCENTER", "VENUS BARYCENTER", "EARTH", "MOON", "MARS BARYCENTER",
            "JUPITER BARYCENTER", "SATURN BARYCENTER", "URANUS BARYCENTER",
            "NEPTUNE BARYCENTER")  # fmt: skip
    assert ForceModel(full).has_complete_planetary_set
    no_moon = tuple(b for b in full if b != "MOON")
    assert not ForceModel(no_moon).has_complete_planetary_set
    emb = (*no_moon[:3], "EARTH BARYCENTER", *no_moon[4:])
    assert ForceModel(emb).has_complete_planetary_set
    with pytest.raises(ValueError, match="max_step_s"):
        IntegratorSettings(max_step_s=0.0)
    helio = transform_state(elements_to_state(ELEMENTS), "J2000")
    with pytest.raises(ValueError, match="J2000 with origin SSB"):
        SatelliteState(state=helio, force_model=ForceModel(("SUN",)))
    # A Sun-only model needs no other GM; asking for Earth with the analytic provider raises.
    bad = SatelliteState(state=init.state, force_model=ForceModel(("SUN", "EARTH")))
    with pytest.raises(KeyError, match="EARTH"):
        propagate_l1(
            bad, np.array([init.epoch_tdb_s + 10.0]), ephemeris=MOVING_SUN, settings=DAY_CAP
        )


# ---------------------------------------------------------------------------------------------
# Configuration hooks (kernel-free)
# ---------------------------------------------------------------------------------------------


def test_force_model_and_settings_from_mvp_config() -> None:
    cfg = load_config(MVP_CONFIG)
    fm = force_model_from_config(cfg.orbit, cfg.constants)
    assert fm.bodies == (
        "SUN",
        "MERCURY BARYCENTER",
        "VENUS BARYCENTER",
        "EARTH",
        "MOON",
        "MARS BARYCENTER",
        "JUPITER BARYCENTER",
        "SATURN BARYCENTER",
        "URANUS BARYCENTER",
        "NEPTUNE BARYCENTER",
    )
    assert fm.include_planetary_perturbations
    assert fm.include_solar_radiation_pressure
    off = cfg.orbit.model_copy(update={"include_planetary_perturbations": False})
    assert force_model_from_config(off, cfg.constants).bodies == ("SUN",)
    sub = cfg.orbit.model_copy(update={"perturbing_bodies": ("JUPITER BARYCENTER",)})
    assert force_model_from_config(sub, cfg.constants).bodies == ("SUN", "JUPITER BARYCENTER")
    s = integrator_settings_from_config(cfg.time)
    assert (s.rtol, s.atol_position_m, s.atol_velocity_m_s) == (1e-12, 1e-3, 1e-9)
    assert (s.max_step_s, s.ephemeris_cache_step_s, s.method) == (300.0, 3600.0, "DOP853")
    with pytest.raises(ValueError, match="integrator_step_s"):
        integrator_settings_from_config(cfg.time.model_copy(update={"integrator_step_s": None}))


def test_satellite_initial_state_closed_form_with_analytic_sun() -> None:
    # MVP elements: a = 1 au, e = 0, i = 0, nu = 30 deg in ECLIPJ2000 -> heliocentric position
    # AU (cos 30, sin 30, 0)_ECL rotated to J2000, velocity sqrt(GM/AU) (-sin 30, cos 30, 0);
    # then the Sun's SSB state is added.
    cfg = load_config(MVP_CONFIG)
    eph = MOVING_SUN
    st = satellite_initial_state(cfg.orbit, cfg.constants, epoch_tdb_s=1000.0, ephemeris=eph)
    c, s, e = math.cos(math.radians(30)), math.sin(math.radians(30)), OBLIQUITY_J2000_RAD
    rot = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    v = math.sqrt(GM_SUN_M3_S2 / AU_M)
    sun = eph.state("SUN", 1000.0)
    np.testing.assert_allclose(
        st.position_m, rot @ [AU_M * c, AU_M * s, 0.0] + sun.position_m, rtol=0, atol=1e-4
    )
    np.testing.assert_allclose(
        st.velocity_m_s, rot @ [-v * s, v * c, 0.0] + sun.velocity_m_s, rtol=0, atol=1e-10
    )
    assert (st.state.frame, st.state.origin, st.epoch_tdb_s) == ("J2000", SSB, 1000.0)
    assert st.mass_kg is None
    assert st.attitude == "sun_facing"


# ---------------------------------------------------------------------------------------------
# SPICE
# ---------------------------------------------------------------------------------------------


@pytest.mark.spice
def test_initial_state_from_mvp_config_in_j2000_ssb(spice_kernels: LoadedKernels) -> None:
    # Exit criterion of M2: the satellite initial state is built in J2000/SSB from the MVP
    # config. Checks: epoch = SPICE UTC->TDB of the config epoch; heliocentric radius 1 au and
    # circular speed; the SSB offset is exactly the Sun's de440s state at that epoch.
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    st = initial_state_from_config(cfg, eph, spice_kernels)
    assert st.epoch_tdb_s == utc_to_tdb_s("2030-01-01T00:00:00", spice_kernels)
    assert (st.state.frame, st.state.origin) == ("J2000", SSB)
    sun = eph.state("SUN", st.epoch_tdb_s)
    r = st.position_m - sun.position_m
    v = st.velocity_m_s - sun.velocity_m_s
    assert np.linalg.norm(r) == pytest.approx(AU_M, rel=1e-15, abs=0)
    gm = eph.gm_m3_s2("SUN")
    assert np.linalg.norm(v) == pytest.approx(math.sqrt(gm / AU_M), rel=1e-15, abs=0)
    # Ecliptic longitude 30 deg: in ECLIPJ2000 the position is AU (cos 30, sin 30, 0).
    r_ecl = transform_state(
        CartesianState(
            epoch_tdb_s=st.epoch_tdb_s, position_m=r, velocity_m_s=v, frame="J2000", origin="SUN"
        ),
        "ECLIPJ2000",
    ).position_m
    np.testing.assert_allclose(
        r_ecl / AU_M, [math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0], rtol=0, atol=1e-15
    )
    # The Sun is 9.5e7 m from the SSB at this epoch (measured), so a forgotten origin shift
    # would break the 1e-15 radius check above by ~6e-4 relative.
    assert np.linalg.norm(sun.position_m) > 5e7
    assert st.force_model.srp is not None
    assert len(st.force_model.bodies) == 10


@pytest.mark.spice
def test_convergence_study_full_model_one_year(spice_kernels: LoadedKernels) -> None:
    # PDF §13.3 "time-step convergence". Full MVP force model (Sun + 9 bodies + SRP), one
    # year, daily outputs. Base: rtol 1e-12, max_step 1 day. Measured changes: max_step 12 h
    # 1.4e-3 m; rtol 1e-13 1.2e-3 m; cache step 1800 s 0.0 m (the planets' sub-mm
    # interpolation errors are below float64 resolution of the satellite's acceleration at
    # 1 au). Tolerance 1e-2 m for each.
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    init = initial_state_from_config(cfg, eph, spice_kernels)
    t = init.epoch_tdb_s + np.arange(0.0, YEAR_S + 1.0, SECONDS_PER_DAY)
    base = propagate_l1(init, t, ephemeris=eph, settings=DAY_CAP).trajectory.position_m
    variants = (
        IntegratorSettings(max_step_s=SECONDS_PER_DAY / 2),
        IntegratorSettings(max_step_s=SECONDS_PER_DAY, rtol=1e-13),
        IntegratorSettings(max_step_s=SECONDS_PER_DAY, ephemeris_cache_step_s=1800.0),
    )
    for settings in variants:
        pos = propagate_l1(init, t, ephemeris=eph, settings=settings).trajectory.position_m
        assert np.max(np.linalg.norm(pos - base, axis=1)) < 1e-2, settings


@pytest.mark.spice
def test_cache_error_is_reported_and_below_1_m_over_the_validated_range(
    spice_kernels: LoadedKernels,
) -> None:
    # Review finding (M2): the run reports the Hermite cache error measured at every interval
    # midpoint. Measured over the MVP year (de440s): Mercury barycentre 0.060 m at 3600 s and
    # 0.968 m at 7200 s (the schema maximum), Moon 0.015 / 0.24 m. The midpoint value must
    # bound direct-SPICE errors at random epochs (the cubic-Hermite error kernel s^2 (1-s)^2
    # peaks at s = 1/2; 1 % excess allowed for the variation of the fourth derivative within
    # an interval, plus 1 mm for float64 resolution of SSB positions up to 4.5e12 m).
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    init = initial_state_from_config(cfg, eph, spice_kernels)
    t = init.epoch_tdb_s + np.array([0.0, YEAR_S])
    for step, limit in ((3600.0, 0.1), (7200.0, 1.0)):
        settings = IntegratorSettings(max_step_s=SECONDS_PER_DAY, ephemeris_cache_step_s=step)
        diag = propagate_l1(init, t, ephemeris=eph, settings=settings).diagnostics
        errs = dict(diag.cache_midpoint_error_m)
        assert set(errs) == set(init.force_model.bodies)
        assert diag.cache_max_midpoint_error_m == errs["MERCURY BARYCENTER"] < limit
    bodies = init.force_model.bodies
    cache = HermiteEphemerisCache.build(eph, bodies, t[0], t[-1], 7200.0)
    mid = cache.max_midpoint_errors_m(eph)
    tr = np.random.default_rng(3).uniform(t[0], t[-1], 400)
    ref = np.stack([eph.positions(b, tr) for b in bodies], axis=1)
    rand = np.max(np.linalg.norm(cache.positions_many(tr) - ref, axis=2), axis=0)
    assert np.all(rand <= 1.01 * mid + 1e-3)
    assert mid[bodies.index("MERCURY BARYCENTER")] > 0.1  # the diagnostic is not vacuous


@pytest.mark.spice
def test_run_assessment_reports_step_convergence_and_srp_effect(
    spice_kernels: LoadedKernels,
) -> None:
    # Plan M2 "step-convergence diagnostic and SRP-effect report", PDF §6.3. Full MVP model,
    # one year, 1-day cap. Measured: max_step halving 8.0e-4 m, rtol/10 2.4e-3 m (tolerance
    # 1e-2 m each); SRP on/off 3.7419e7 m at the end vs the circular-orbit estimate
    # 2 beta n t a = 3.7392e7 m (0.07 %; tolerance 1 %: planetary perturbations and O(beta^2)
    # terms are smaller). MVP: Cr 1.3, A/m 0.02 -> beta = 1.99e-5.
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    init = initial_state_from_config(cfg, eph, spice_kernels)
    assert init.force_model.srp is not None
    t = init.epoch_tdb_s + np.append(np.arange(0.0, YEAR_S, 10 * SECONDS_PER_DAY), YEAR_S)
    run = propagate_l1(init, t, ephemeris=eph, settings=DAY_CAP)
    report = assess_l1_run(run, ephemeris=eph)
    assert report.step_halving_max_delta_m is not None
    assert report.step_halving_max_delta_m < 1e-2
    assert report.rtol_tightening_max_delta_m is not None
    assert report.rtol_tightening_max_delta_m < 1e-2
    gm = eph.gm_m3_s2("SUN")
    beta = init.force_model.srp.lightness_number(gm)
    assert report.srp_lightness_number == beta
    estimate = srp_along_track_drift_estimate_m(
        beta, semimajor_axis_m=AU_M, duration_s=YEAR_S, gm_sun_m3_s2=gm
    )
    assert report.srp_effect_final_m == pytest.approx(estimate, rel=1e-2)
    assert report.srp_effect_max_m is not None
    assert report.srp_effect_max_m >= report.srp_effect_final_m
    # Without SRP and without a step cap the corresponding entries are None (not 0).
    plain = SatelliteState(state=init.state, force_model=ForceModel(("SUN",)))
    short = init.epoch_tdb_s + np.array([0.0, 10 * SECONDS_PER_DAY])
    rep = assess_l1_run(
        propagate_l1(plain, short, ephemeris=eph, settings=IntegratorSettings(rtol=1e-13)),
        ephemeris=eph,
    )
    assert rep.srp_effect_max_m is None
    assert rep.srp_lightness_number is None
    assert rep.step_halving_max_delta_m is None
    assert rep.rtol_tightening_max_delta_m is None


@pytest.mark.spice
@pytest.mark.parametrize("with_srp", [False, True])
def test_sun_only_l1_with_the_spice_sun_equals_l0(
    spice_kernels: LoadedKernels, with_srp: bool
) -> None:
    # Plan/PDF acceptance "L1 with planets off == L0" on the real path: de440s Sun (accelerated
    # by the planets), MVP orbit, one year, compared heliocentrically. The heliocentric
    # formulation includes the indirect term of exactly the modelled bodies (none here), so the
    # motion is two-body (with mu - k under SRP). Measured 7.7e-4 m (no SRP) and 1.5e-3 m
    # (SRP), energy drift 1.3e-15 / 2.2e-15. Tolerances 1e-2 m and 1e-13. The former
    # barycentric Sun-only model missed L0 by 2.2e8 m (review finding, M2).
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    full = initial_state_from_config(cfg, eph, spice_kernels)
    srp = full.force_model.srp if with_srp else None
    init = SatelliteState(state=full.state, force_model=ForceModel(("SUN",), srp))
    t = init.epoch_tdb_s + np.arange(0.0, YEAR_S + 1.0, SECONDS_PER_DAY)
    prop = propagate_l1(init, t, ephemeris=eph, settings=DAY_CAP)
    helio = _helio_l1(prop, eph)
    mu = eph.gm_m3_s2("SUN") - (srp.strength_m3_s2 if srp is not None else 0.0)
    sun0 = eph.state("SUN", init.epoch_tdb_s)
    h0 = CartesianState(
        epoch_tdb_s=init.epoch_tdb_s, position_m=init.position_m - sun0.position_m,
        velocity_m_s=init.velocity_m_s - sun0.velocity_m_s, frame="J2000", origin="SUN",
    )  # fmt: skip
    l0 = propagate_kepler(h0, t, mu_m3_s2=mu).trajectory.position_m
    assert np.max(np.linalg.norm(l0 - helio, axis=1)) < 1e-2
    assert prop.diagnostics.sun_two_body_energy_rel_drift < 1e-13
    assert prop.diagnostics.formulation == "heliocentric"
    # The Sun really is accelerated over this year (else the test would not discriminate):
    # its SSB velocity changes by more than 1 m/s.
    sun1 = eph.state("SUN", float(t[-1]))
    assert np.linalg.norm(sun1.velocity_m_s - sun0.velocity_m_s) > 1.0


@pytest.mark.spice
def test_full_model_heliocentric_vs_barycentric_and_body_subsets(
    spice_kernels: LoadedKernels,
) -> None:
    # MVP orbit without SRP, one year, daily outputs, positions compared heliocentrically.
    # (a) Full set: the barycentric comparison path differs from the heliocentric model by the
    #     Sun's DE440 acceleration that the nine point masses do not explain (asteroids, Pluto,
    #     relativity), which the barycentric form applies as an indirect term only. Measured
    #     566 m; bounds [100 m, 2 km]. REBOUND (tests/integration) agrees with the heliocentric
    #     form to ~1 m.
    # (b) Dropping a body removes only its physical (tidal) effect. Measured: Neptune 2.9e3 m
    #     (the former barycentric subset gave 3.2e5 m, the uncancelled Sun reflex), Jupiter
    #     7.4e6 m; full model vs L0 3.25e7 m. Bounds bracket each by ~3x.
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    orbit = cfg.orbit.model_copy(
        update={"include_solar_radiation_pressure": False, "cr": None, "area_to_mass_m2_kg": None}
    )
    epoch = utc_to_tdb_s(cfg.orbit.initial_epoch_utc, spice_kernels)
    init = satellite_initial_state(orbit, cfg.constants, epoch_tdb_s=epoch, ephemeris=eph)
    t = init.epoch_tdb_s + np.arange(0.0, YEAR_S + 1.0, SECONDS_PER_DAY)
    helio = propagate_l1(init, t, ephemeris=eph, settings=DAY_CAP).trajectory.position_m
    bary = propagate_l1(
        init, t, ephemeris=eph, settings=DAY_CAP, formulation="barycentric"
    ).trajectory.position_m
    assert 1e2 < np.max(np.linalg.norm(bary - helio, axis=1)) < 2e3

    def without(body: str) -> float:
        bodies = tuple(b for b in init.force_model.bodies if b != body)
        st = SatelliteState(state=init.state, force_model=ForceModel(bodies))
        pos = propagate_l1(st, t, ephemeris=eph, settings=DAY_CAP).trajectory.position_m
        return float(np.max(np.linalg.norm(pos - helio, axis=1)))

    assert 1e3 < without("NEPTUNE BARYCENTER") < 1e4
    assert 2e6 < without("JUPITER BARYCENTER") < 2e7
    el = elements_from_config(orbit, epoch_tdb_s=epoch, mu_m3_s2=eph.gm_m3_s2("SUN"))
    l0 = transform_trajectory(propagate_kepler(el, t).trajectory, "J2000").position_m
    sun = eph.positions("SUN", t)
    assert 1e7 < np.max(np.linalg.norm(helio - sun - l0, axis=1)) < 1e8


@pytest.mark.spice
def test_propagate_l1_from_config_on_a_short_window(spice_kernels: LoadedKernels) -> None:
    # The config path end to end (grid from time.start_utc, initial state, settings with the
    # MVP max_step 300 s) on a 2-day window; compared with the 1-day-cap run (< 1e-2 m).
    cfg = load_config(MVP_CONFIG)
    short = cfg.model_copy(update={"time": cfg.time.model_copy(update={"duration_days": 2.0})})
    eph = SpiceEphemeris(spice_kernels)
    run = propagate_l1_from_config(short, eph, spice_kernels)
    assert len(run.grid) == 49
    np.testing.assert_array_equal(run.propagation.trajectory.epochs_tdb_s, run.grid.epochs_tdb_s)
    assert run.propagation.diagnostics.success
    assert run.propagation.settings.max_step_s == 300.0
    assert run.propagation.source_kernel_ids == ("de440s.bsp",)
    assert dict(run.propagation.diagnostics.gm_m3_s2)["SUN"] == eph.gm_m3_s2("SUN")
    capped = propagate_l1(
        run.propagation.initial, run.grid.epochs_tdb_s, ephemeris=eph, settings=DAY_CAP
    )
    diff = run.propagation.trajectory.position_m - capped.trajectory.position_m
    assert np.max(np.linalg.norm(diff, axis=1)) < 1e-2
    two_body = cfg.model_copy(
        update={"orbit": cfg.orbit.model_copy(update={"model": "two_body_kepler"})}
    )
    with pytest.raises(ValueError, match=r"orbit\.model must be"):
        propagate_l1_from_config(two_body, eph, spice_kernels)


@pytest.mark.spice
@pytest.mark.parametrize("orbit_epoch_utc", ["2030-01-11T00:00:00", "2029-12-22T00:00:00"])
def test_orbit_epoch_and_grid_start_are_kept_apart(
    spice_kernels: LoadedKernels, orbit_epoch_utc: str
) -> None:
    # Review finding (M2): orbit.initial_epoch_utc (where the elements osculate) and
    # time.start_utc (first output epoch, 2030-01-01) are different inputs. With the orbit
    # epoch 10 days after (before) the grid start, the initial state must sit at the orbit
    # epoch, the grid must start at start_utc, and the run must equal a direct propagate_l1
    # call (identical calls: tolerance 1e-6 m). Swapping the epochs moves the satellite by
    # ~2.6e10 m (10 days at ~30 km/s) and fails every check.
    cfg = load_config(MVP_CONFIG)
    short = cfg.model_copy(
        update={
            "time": cfg.time.model_copy(
                update={
                    "duration_days": 20.0,
                    "output_step_s": 86400.0,
                    "integrator_step_s": 3600.0,
                }
            ),
            "orbit": cfg.orbit.model_copy(update={"initial_epoch_utc": orbit_epoch_utc}),
        }
    )
    eph = SpiceEphemeris(spice_kernels)
    run = propagate_l1_from_config(short, eph, spice_kernels)
    t_orbit = utc_to_tdb_s(orbit_epoch_utc, spice_kernels)
    assert run.propagation.initial.epoch_tdb_s == t_orbit
    assert run.grid.epochs_tdb_s[0] == utc_to_tdb_s("2030-01-01T00:00:00", spice_kernels)
    np.testing.assert_array_equal(
        run.grid.epochs_tdb_s, time_grid_from_config(short.time, spice_kernels).epochs_tdb_s
    )
    directions = {s.direction for s in run.propagation.diagnostics.segments}
    after_start = t_orbit > run.grid.epochs_tdb_s[0]
    assert directions == ({"forward", "backward"} if after_start else {"forward"})
    init = satellite_initial_state(short.orbit, short.constants, epoch_tdb_s=t_orbit, ephemeris=eph)
    direct = propagate_l1(
        init,
        run.grid.epochs_tdb_s,
        ephemeris=eph,
        settings=integrator_settings_from_config(short.time),
    )
    np.testing.assert_allclose(
        run.propagation.trajectory.position_m, direct.trajectory.position_m, rtol=0, atol=1e-6
    )
    # The elements osculate at the orbit epoch: heliocentric radius = 1 au there (it is a grid
    # epoch for the +10 d case; 1e-9 relative = 150 m, far above the integration error).
    if after_start:
        k = int(np.argmin(np.abs(run.grid.epochs_tdb_s - t_orbit)))
        assert run.grid.epochs_tdb_s[k] == pytest.approx(t_orbit, abs=1e-3)
        sun = eph.state("SUN", float(run.grid.epochs_tdb_s[k]))
        r = run.propagation.trajectory.position_m[k] - sun.position_m
        assert np.linalg.norm(r) == pytest.approx(AU_M, rel=1e-9)


@pytest.mark.spice
def test_a_foreign_kernel_set_is_refused(
    spice_kernels: LoadedKernels, kernel_set: KernelSet
) -> None:
    # Review finding (M2): frames.ephemeris_kernel_set pins the ephemeris/GM data of a run;
    # a handle whose manifest has another kernel_set_id is refused by every config entry point.
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    assert cfg.frames.ephemeris_kernel_set == spice_kernels.kernel_set.manifest.kernel_set_id
    check_kernel_set(cfg, spice_kernels)
    other = KernelSet(
        manifest=replace(kernel_set.manifest, kernel_set_id="some-other-set"),
        directory=kernel_set.directory,
    )
    foreign = replace(spice_kernels, kernel_set=other)
    with pytest.raises(KernelSetMismatchError, match="some-other-set"):
        check_kernel_set(cfg, foreign)
    with pytest.raises(KernelSetMismatchError, match=r"frames\.ephemeris_kernel_set"):
        initial_state_from_config(cfg, eph, foreign)
    with pytest.raises(KernelSetMismatchError):
        propagate_l1_from_config(cfg, eph, foreign)


@pytest.mark.spice
def test_epochs_outside_ephemeris_coverage_are_refused(spice_kernels: LoadedKernels) -> None:
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    init = initial_state_from_config(cfg, eph, spice_kernels)
    with pytest.raises(EphemerisCoverageError, match="not covered"):
        propagate_l1(init, np.array([4.8e9]), ephemeris=eph, settings=DAY_CAP)


@pytest.mark.spice
@pytest.mark.slow
def test_mvp_config_full_year_at_configured_step(spice_kernels: LoadedKernels) -> None:
    # The MVP run as configured (max_step 300 s, ~1.3e6 evaluations, ~50 s on one core of
    # holy7c18406 with the heliocentric right-hand side). It must succeed and agree with the
    # 1-day-cap run (measured 4.6e-2 m: the extra 1.0e5 steps accumulate round-off
    # ~eps |r| sqrt(N) ~ 1e-2 m); tolerance 0.1 m.
    cfg = load_config(MVP_CONFIG)
    eph = SpiceEphemeris(spice_kernels)
    run = propagate_l1_from_config(cfg, eph, spice_kernels)
    diag = run.propagation.diagnostics
    assert diag.success
    assert len(run.grid) == 8767
    assert diag.nfev > 1_000_000
    t = run.grid.epochs_tdb_s[::24]
    ref = propagate_l1(run.propagation.initial, t, ephemeris=eph, settings=DAY_CAP)
    diff = run.propagation.trajectory.position_m[::24] - ref.trajectory.position_m
    assert np.max(np.linalg.norm(diff, axis=1)) < 0.1
