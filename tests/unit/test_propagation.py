"""Tests for L0 two-body propagation (helioswarm.orbit.propagate_kepler), PDF §7.4 / §13.1.

Independent references:
* a direct numerical integration of r'' = -mu r / |r|^3 with scipy ``solve_ivp`` (DOP853),
  which shares no code with the closed-form propagator;
* the period measured from that integration with an event function (Kepler's third law);
* closed-form time of flight between true anomalies via the half-angle relation;
* exact symmetry facts (half a period from periapsis is apoapsis; a whole period returns).

Integrations use canonical units (mu = 1, a ~ 1) so that ``atol`` is meaningful; the
propagator is unit-agnostic (SI throughout), so this tests the same code path.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from helioswarm.constants import AU_M, GM_SUN_M3_S2, SECONDS_PER_DAY
from helioswarm.frames import transform_state, transform_trajectory
from helioswarm.orbit import (
    OrbitalElements,
    elements_to_state,
    orbital_period_s,
    propagate_kepler,
    specific_angular_momentum_m2_s,
    specific_energy_J_kg,
)

MU = GM_SUN_M3_S2


def _el(**kw: Any) -> OrbitalElements:
    base: dict[str, Any] = {
        "a_m": AU_M,
        "e": 0.0,
        "i_rad": 0.0,
        "raan_rad": 0.0,
        "argp_rad": 0.0,
        "nu_rad": 0.0,
        "mu_m3_s2": MU,
        "epoch_tdb_s": 0.0,
    }
    base.update(kw)
    return OrbitalElements(**base)


def _two_body_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    r = y[:3]
    return np.concatenate([y[3:], -r / np.linalg.norm(r) ** 3])  # mu = 1


def _integrate(el: OrbitalElements, t_eval: np.ndarray, rtol: float) -> Any:
    assert el.mu_m3_s2 == 1.0
    s0 = elements_to_state(el)
    return solve_ivp(
        _two_body_rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        np.concatenate([s0.position_m, s0.velocity_m_s]),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=rtol * 1e-2,
    )


# ---------------------------------------------------------------------------------------------
# Independent numerical cross-check
# ---------------------------------------------------------------------------------------------


def test_matches_direct_numerical_integration_e03_one_orbit() -> None:
    # e = 0.3, inclined, one full orbit. DOP853 error measured at rtol 1e-10/1e-12/1e-13:
    # 1.2e-9 / 1.5e-11 / 2.3e-12 (position, units of a). The agreement must (i) improve as
    # the integrator is tightened, i.e. the residual difference is integrator error, not a
    # propagator bias, and (ii) reach < 1e-10 a at rtol 1e-13. A wrong mu at the 1e-9 level
    # already gives a ~3e-9 a phase error after one orbit and fails (ii).
    el = _el(a_m=1.0, e=0.3, i_rad=0.4, raan_rad=1.0, argp_rad=2.0, nu_rad=0.5, mu_m3_s2=1.0)
    t = np.linspace(0.0, el.period_s, 401)
    kep = propagate_kepler(el, t).trajectory
    errs = []
    for rtol in (1e-10, 1e-13):
        sol = _integrate(el, t, rtol)
        assert sol.success
        dr = np.max(np.linalg.norm(sol.y[:3].T - kep.position_m, axis=1))
        dv = np.max(np.linalg.norm(sol.y[3:].T - kep.velocity_m_s, axis=1))
        errs.append((dr, dv))
    assert errs[1][0] < errs[0][0] / 50
    assert errs[1][1] < errs[0][1] / 50
    assert errs[1][0] < 1e-10
    assert errs[1][1] < 1e-10


@pytest.mark.parametrize("a", [0.4, 1.0, 5.2])
@pytest.mark.parametrize("e", [0.0, 0.5, 0.9])
def test_kepler_third_law_period_from_integration(a: float, e: float) -> None:
    # Measure the time for the integrated orbit to return to its initial direction (event:
    # component of r along h x r0 crosses zero upwards after a full turn) and compare with
    # T = 2 pi sqrt(a^3/mu). With rtol 1e-12 the event time agrees to <= 2.5e-12 relative
    # (measured over these 9 cases); 1e-10 bound. Wrong exponents (T ~ a or a^2) fail by
    # > 10 % for a != 1, and the e-independence of T is checked by e = 0.9.
    el = _el(a_m=a, e=e, i_rad=0.3, raan_rad=0.2, argp_rad=0.7, nu_rad=1.0, mu_m3_s2=1.0)
    s0 = elements_to_state(el)
    q_hat = np.cross(np.cross(s0.position_m, s0.velocity_m_s), s0.position_m)
    q_hat /= np.linalg.norm(q_hat)

    def crossing(_t: float, y: np.ndarray) -> float:
        return float(np.dot(y[:3], q_hat))

    crossing.direction = 1.0  # type: ignore[attr-defined]
    t_kepler = 2 * math.pi * math.sqrt(a**3)
    sol = solve_ivp(
        _two_body_rhs,
        (0.0, 1.5 * t_kepler),
        np.concatenate([s0.position_m, s0.velocity_m_s]),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        events=crossing,
    )
    crossings = sol.t_events[0]
    crossings = crossings[crossings > 0.25 * t_kepler]  # skip the start point itself
    assert crossings.size == 1
    assert crossings[0] == pytest.approx(t_kepler, rel=1e-10, abs=0)
    assert crossings[0] == pytest.approx(orbital_period_s(a, 1.0), rel=1e-10, abs=0)


def test_gaussian_year_at_one_au() -> None:
    # Kepler III with GM_sun (DE440) at a = 1 au equals the Gaussian year 2 pi / k days,
    # k = 0.01720209895 (IAU 1976), to ~2.5e-12 (half the GM difference; see test_constants).
    t_days = orbital_period_s(AU_M, MU) / SECONDS_PER_DAY
    assert t_days == pytest.approx(2 * math.pi / 0.01720209895, rel=1e-10, abs=0)


# ---------------------------------------------------------------------------------------------
# Invariants and closed-form checks
# ---------------------------------------------------------------------------------------------


def test_circular_orbit_radius_and_speed_constant_over_many_periods() -> None:
    el = _el(i_rad=0.2, raan_rad=0.4, nu_rad=0.1)
    t = np.linspace(-3.0, 100.0, 50_001) * el.period_s
    traj = propagate_kepler(el, t).trajectory
    r = np.linalg.norm(traj.position_m, axis=1)
    v = np.linalg.norm(traj.velocity_m_s, axis=1)
    # Closed form: only round-off; required ~1e-12 relative (measured 4.4e-16).
    assert np.max(np.abs(r / AU_M - 1.0)) <= 1e-12
    assert np.max(np.abs(v / math.sqrt(MU / AU_M) - 1.0)) <= 1e-12
    # And the motion is uniform: true longitude advances at n (closed form for e = 0).
    u = 0.1 + el.mean_motion_rad_s * t
    p_hat = np.array([math.cos(0.4), math.sin(0.4), 0.0])
    q_hat = np.array([-math.sin(0.4) * math.cos(0.2), math.cos(0.4) * math.cos(0.2), math.sin(0.2)])
    expected = AU_M * (np.cos(u)[:, None] * p_hat + np.sin(u)[:, None] * q_hat)
    # Phase error ~ eps * n t ~ 1e-13 rad after 100 revolutions -> 1e-12 relative bound.
    assert np.max(np.linalg.norm(traj.position_m - expected, axis=1)) <= 1e-12 * AU_M


@pytest.mark.parametrize("e", [0.0, 0.3, 0.9, 0.99])
def test_energy_and_angular_momentum_conserved(e: float) -> None:
    el = _el(a_m=2.0 * AU_M, e=e, i_rad=1.0, raan_rad=2.0, argp_rad=3.0, nu_rad=0.4)
    t = np.linspace(0.0, 10.0 * el.period_s, 20_001)
    traj = propagate_kepler(el, t).trajectory
    energy = specific_energy_J_kg(traj.position_m, traj.velocity_m_s, MU)
    h = specific_angular_momentum_m2_s(traj.position_m, traj.velocity_m_s)
    # Near periapsis v^2/2 and mu/r are each ~mu/(a(1-e)), i.e. 1/(1-e) times |epsilon|, so the
    # cancellation costs a few eps/(1-e): measured max 2.5e-13 at e = 0.99 (5.7e-15 at
    # e = 0.3); angular momentum ~10 eps. Bounds 1e-12 relative.
    e_ref = -MU / (2.0 * el.a_m)
    assert np.max(np.abs(energy / e_ref - 1.0)) <= 1e-12
    h_ref = h[0]
    assert np.max(np.linalg.norm(h - h_ref, axis=1)) <= 1e-12 * np.linalg.norm(h_ref)
    assert np.linalg.norm(h_ref) == pytest.approx(
        el.specific_angular_momentum_m2_s, rel=1e-13, abs=0
    )


@pytest.mark.parametrize("e", [0.0, 0.3, 0.9])
def test_returns_to_start_after_whole_periods_and_not_before(e: float) -> None:
    el = _el(a_m=1.7 * AU_M, e=e, i_rad=0.5, raan_rad=1.5, argp_rad=0.3, nu_rad=2.0)
    k = np.arange(0, 11)
    traj = propagate_kepler(el, k * el.period_s).trajectory
    r0, v0 = traj.position_m[0], traj.velocity_m_s[0]
    # Round-off in M after k turns ~ eps * 2 pi k / (1 - e)^2 amplification near periapsis:
    # 1e-11 relative bound (measured max over k = 1..10: 9.9e-15 at e = 0.3, 2.8e-14 at e = 0.9).
    for j in range(1, 11):
        assert np.linalg.norm(traj.position_m[j] - r0) <= 1e-11 * np.linalg.norm(r0)
        assert np.linalg.norm(traj.velocity_m_s[j] - v0) <= 1e-11 * np.linalg.norm(v0)
    # Discrimination: at half a period it is far from the start.
    half = propagate_kepler(el, 0.5 * el.period_s).trajectory
    assert np.linalg.norm(half.position_m[0] - r0) > 0.1 * el.a_m


def test_half_period_from_periapsis_is_apoapsis() -> None:
    a, e = 1.2 * AU_M, 0.6
    el = _el(a_m=a, e=e, argp_rad=0.9)
    traj = propagate_kepler(el, [0.5 * el.period_s]).trajectory
    assert np.linalg.norm(traj.position_m[0]) == pytest.approx(a * (1 + e), rel=1e-14, abs=0)
    # Apoapsis direction is opposite to periapsis (argp = 0.9 in the ecliptic plane).
    assert traj.position_m[0] == pytest.approx(
        -a * (1 + e) * np.array([math.cos(0.9), math.sin(0.9), 0.0]), rel=1e-12, abs=1e-3
    )


def test_time_of_flight_closed_form() -> None:
    # From periapsis to nu = 90 deg: tan(E/2) = sqrt((1-e)/(1+e)) tan(45 deg),
    # t = (E - e sin E)/n. After t the satellite is at r = p on the latus rectum.
    a, e = 0.8 * AU_M, 0.45
    el = _el(a_m=a, e=e)
    big_e = 2.0 * math.atan(math.sqrt((1 - e) / (1 + e)))
    t = (big_e - e * math.sin(big_e)) / el.mean_motion_rad_s
    traj = propagate_kepler(el, [t, -t]).trajectory
    p = a * (1 - e * e)
    assert traj.position_m[0] == pytest.approx([0.0, p, 0.0], abs=1e-4)
    assert traj.position_m[1] == pytest.approx([0.0, -p, 0.0], abs=1e-4)  # backward in time


def test_backward_then_forward_propagation_is_consistent() -> None:
    el = _el(a_m=3.0 * AU_M, e=0.4, i_rad=0.8, raan_rad=0.1, argp_rad=4.0, nu_rad=5.0)
    t_back = -2.3 * el.period_s
    past = propagate_kepler(el, [t_back]).trajectory.state_at(0)
    assert past.epoch_tdb_s == t_back
    forward = propagate_kepler(past, [0.0], mu_m3_s2=MU).trajectory.state_at(0)
    s0 = elements_to_state(el)
    assert np.linalg.norm(forward.position_m - s0.position_m) <= 1e-12 * np.linalg.norm(
        s0.position_m
    )
    assert np.linalg.norm(forward.velocity_m_s - s0.velocity_m_s) <= 1e-12 * np.linalg.norm(
        s0.velocity_m_s
    )


def test_initial_epoch_reproduces_initial_state_and_metadata() -> None:
    el = _el(e=0.2, i_rad=0.1, nu_rad=1.0, epoch_tdb_s=8.0e8, frame="ECLIPJ2000")
    res = propagate_kepler(el, np.array([8.0e8, 8.0e8 + 86_400.0]))
    s0 = elements_to_state(el)
    traj = res.trajectory
    assert traj.position_m.shape == (2, 3)
    assert traj.velocity_m_s.shape == (2, 3)
    assert traj.frame == "ECLIPJ2000"
    assert traj.origin == "SUN"
    assert traj.time_scale == "TDB"
    assert np.array_equal(traj.epochs_tdb_s, [8.0e8, 8.0e8 + 86_400.0])
    assert traj.position_m[0] == pytest.approx(s0.position_m, rel=1e-15, abs=0)
    assert traj.velocity_m_s[0] == pytest.approx(s0.velocity_m_s, rel=1e-15, abs=0)
    assert res.elements is el
    assert 1 <= res.kepler_iterations <= 12
    assert res.max_kepler_residual_rad <= 1e-15


def test_state_input_equals_elements_input() -> None:
    el = _el(a_m=1.4 * AU_M, e=0.25, i_rad=0.6, raan_rad=2.2, argp_rad=1.1, nu_rad=0.7)
    t = np.linspace(-1e8, 1e8, 101)
    from_el = propagate_kepler(el, t).trajectory
    from_state = propagate_kepler(elements_to_state(el), t, mu_m3_s2=MU).trajectory
    scale = np.linalg.norm(from_el.position_m, axis=1).max()
    assert np.max(np.abs(from_state.position_m - from_el.position_m)) <= 1e-12 * scale


def test_propagation_commutes_with_frame_rotation() -> None:
    # Central-force motion is rotation invariant: propagating in ECLIPJ2000 and rotating to
    # J2000 must equal rotating the initial state and propagating in J2000.
    el = _el(a_m=0.9 * AU_M, e=0.35, i_rad=0.05, raan_rad=1.0, argp_rad=2.0, nu_rad=3.0)
    t = np.linspace(0.0, 2.0 * el.period_s, 257)
    a = transform_trajectory(propagate_kepler(el, t).trajectory, "J2000")
    s0_j2000 = transform_state(elements_to_state(el), "J2000")
    b = propagate_kepler(s0_j2000, t, mu_m3_s2=MU).trajectory
    assert b.frame == "J2000"
    assert (
        np.max(np.linalg.norm(a.position_m - b.position_m, axis=1)) <= 1e-12 * el.apoapsis_radius_m
    )


def test_propagate_argument_validation() -> None:
    el = _el()
    s = elements_to_state(el)
    with pytest.raises(ValueError, match="mu_m3_s2 is required"):
        propagate_kepler(s, [0.0])
    with pytest.raises(ValueError, match="contradicts"):
        propagate_kepler(el, [0.0], mu_m3_s2=MU * (1 + 1e-12))
    with pytest.raises(ValueError, match="1-D"):
        propagate_kepler(el, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        propagate_kepler(el, [0.0, np.nan])
    assert propagate_kepler(el, 5.0).trajectory.position_m.shape == (1, 3)
