"""Tests for orbital elements <-> Cartesian state (helioswarm.orbit).

The independent reference for elements -> state is the textbook construction written out
here: perifocal position/velocity rotated by R3(-raan) R1(-i) R3(-argp) built from elementary
rotation matrices (Vallado 4th ed. Alg. 10), which is a different code path from the
implementation's plane-basis formulation. Closed-form checks use r_p = a(1-e),
v_p = sqrt(mu (1+e) / (a (1-e))) and the invariants epsilon = -mu/(2a), |h| = sqrt(mu p).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from helioswarm.config import parse_config
from helioswarm.constants import AU_M, GM_SUN_M3_S2
from helioswarm.frames import CartesianState
from helioswarm.orbit import (
    DEFAULT_CIRCULAR_TOL,
    OrbitalElements,
    eccentricity_vector,
    elements_from_config,
    elements_to_state,
    orbital_period_s,
    specific_angular_momentum_m2_s,
    specific_energy_J_kg,
    state_to_elements,
)

from config_helpers import load_mvp_data, set_path

MU = GM_SUN_M3_S2
EPS = float(np.finfo(np.float64).eps)


def _wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def _r1(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=float)


def _r3(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)


def _reference_state(el: OrbitalElements) -> tuple[np.ndarray, np.ndarray]:
    """Textbook COE2RV via perifocal frame and elementary rotations."""
    p = el.a_m * (1 - el.e**2)
    nu = el.nu_rad
    r = p / (1 + el.e * math.cos(nu))
    r_pf = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    v_pf = math.sqrt(el.mu_m3_s2 / p) * np.array([-math.sin(nu), el.e + math.cos(nu), 0.0])
    q = _r3(-el.raan_rad) @ _r1(-el.i_rad) @ _r3(-el.argp_rad)
    return q @ r_pf, q @ v_pf


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


def _random_elements(rng: np.random.Generator, n: int) -> Iterator[OrbitalElements]:
    for _ in range(n):
        yield _el(
            a_m=rng.uniform(0.1, 50.0) * AU_M,
            e=rng.uniform(1e-3, 0.95),
            i_rad=rng.uniform(1e-3, math.pi - 1e-3),
            raan_rad=rng.uniform(0, 2 * math.pi),
            argp_rad=rng.uniform(0, 2 * math.pi),
            nu_rad=rng.uniform(0, 2 * math.pi),
            epoch_tdb_s=rng.uniform(-1e9, 1e9),
        )


# ---------------------------------------------------------------------------------------------
# elements -> state
# ---------------------------------------------------------------------------------------------


def test_periapsis_and_apoapsis_closed_form() -> None:
    a, e = 1.3 * AU_M, 0.3
    s = elements_to_state(_el(a_m=a, e=e, nu_rad=0.0))
    assert s.position_m == pytest.approx([a * (1 - e), 0.0, 0.0], rel=1e-15, abs=1e-6)
    v_p = math.sqrt(MU * (1 + e) / (a * (1 - e)))
    assert s.velocity_m_s == pytest.approx([0.0, v_p, 0.0], rel=1e-15, abs=1e-12)
    s = elements_to_state(_el(a_m=a, e=e, nu_rad=math.pi))
    v_a = math.sqrt(MU * (1 - e) / (a * (1 + e)))
    assert s.position_m == pytest.approx([-a * (1 + e), 0.0, 0.0], rel=1e-15, abs=1e-4)
    assert s.velocity_m_s == pytest.approx([0.0, -v_a, 0.0], rel=1e-15, abs=1e-11)
    # Semi-latus-rectum point (nu = 90 deg): r = p along +y, radial speed sqrt(mu/p) e.
    p = a * (1 - e * e)
    s = elements_to_state(_el(a_m=a, e=e, nu_rad=math.pi / 2))
    assert s.position_m == pytest.approx([0.0, p, 0.0], rel=1e-15, abs=1e-4)
    # Equatorial (i = 0): the z component is exactly 0, so abs=0 is attainable.
    assert s.velocity_m_s == pytest.approx(
        [-math.sqrt(MU / p), e * math.sqrt(MU / p), 0.0], rel=1e-15, abs=0
    )


def test_orientation_closed_form_polar_orbit() -> None:
    # i = 90 deg, raan = 90 deg, argp = 0, nu = 0, circular: at the ascending node, which
    # points along +y; moving north (+z) at circular speed.
    a = 2.0 * AU_M
    s = elements_to_state(_el(a_m=a, i_rad=math.pi / 2, raan_rad=math.pi / 2))
    assert s.position_m == pytest.approx([0.0, a, 0.0], abs=1e-4)
    assert s.velocity_m_s == pytest.approx([0.0, 0.0, math.sqrt(MU / a)], abs=1e-11)
    # Same orbit a quarter-turn later (nu = 90 deg): at the northernmost point.
    s = elements_to_state(_el(a_m=a, i_rad=math.pi / 2, raan_rad=math.pi / 2, nu_rad=math.pi / 2))
    assert s.position_m == pytest.approx([0.0, 0.0, a], abs=1e-4)
    # Velocity heads back along -y (towards the descending node at -y).
    assert s.velocity_m_s == pytest.approx([0.0, -math.sqrt(MU / a), 0.0], abs=1e-11)


def test_elements_to_state_matches_textbook_rotation() -> None:
    rng = np.random.default_rng(101)
    for el in _random_elements(rng, 500):
        s = elements_to_state(el)
        r_ref, v_ref = _reference_state(el)
        # Two float64 evaluations of the same closed form: agreement to ~10 eps.
        assert np.linalg.norm(s.position_m - r_ref) <= 1e-14 * np.linalg.norm(r_ref)
        assert np.linalg.norm(s.velocity_m_s - v_ref) <= 1e-14 * np.linalg.norm(v_ref)
        assert s.frame == "ECLIPJ2000"
        assert s.origin == "SUN"
        assert s.epoch_tdb_s == el.epoch_tdb_s


def test_state_invariants_match_element_invariants() -> None:
    rng = np.random.default_rng(5)
    for el in _random_elements(rng, 200):
        s = elements_to_state(el)
        energy = float(specific_energy_J_kg(s.position_m, s.velocity_m_s, MU))
        assert energy == pytest.approx(-MU / (2 * el.a_m), rel=1e-12, abs=0)
        assert energy == pytest.approx(el.specific_energy_J_kg, rel=1e-12, abs=0)
        h = specific_angular_momentum_m2_s(s.position_m, s.velocity_m_s)
        assert np.linalg.norm(h) == pytest.approx(
            math.sqrt(MU * el.a_m * (1 - el.e**2)), rel=1e-13, abs=0
        )
        # h points along the orbit normal (sin i sin raan, -sin i cos raan, cos i).
        n_ref = [
            math.sin(el.i_rad) * math.sin(el.raan_rad),
            -math.sin(el.i_rad) * math.cos(el.raan_rad),
            math.cos(el.i_rad),
        ]
        assert h / np.linalg.norm(h) == pytest.approx(n_ref, abs=1e-14)
        ev = eccentricity_vector(s.position_m, s.velocity_m_s, MU)
        assert np.linalg.norm(ev) == pytest.approx(el.e, abs=1e-13)


def test_element_properties() -> None:
    el = _el(a_m=2 * AU_M, e=0.25)
    assert el.periapsis_radius_m == pytest.approx(1.5 * AU_M, rel=1e-15, abs=0)
    assert el.apoapsis_radius_m == pytest.approx(2.5 * AU_M, rel=1e-15, abs=0)
    assert el.semi_latus_rectum_m == pytest.approx(2 * AU_M * (1 - 0.0625), rel=1e-15, abs=0)
    # Kepler III: T(2 au) = 2^(3/2) T(1 au).
    assert el.period_s == pytest.approx(2**1.5 * orbital_period_s(AU_M, MU), rel=1e-15, abs=0)
    assert el.mean_motion_rad_s * el.period_s == pytest.approx(2 * math.pi, rel=1e-15, abs=0)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"a_m": -1.0}, "a_m"),
        ({"a_m": 0.0}, "a_m"),
        ({"e": 1.0}, "e must"),
        ({"e": -1e-9}, "e must"),
        ({"i_rad": math.pi + 1e-9}, "i_rad"),
        ({"i_rad": -1e-9}, "i_rad"),
        ({"mu_m3_s2": 0.0}, "mu"),
        ({"nu_rad": math.nan}, "finite"),
        ({"epoch_tdb_s": math.inf}, "finite"),
        ({"frame": "IAU_EARTH"}, "frame"),
        ({"center": "SOLAR SYSTEM BARYCENTER"}, "center"),
    ],
)
def test_elements_validation(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _el(**kw)


# ---------------------------------------------------------------------------------------------
# state -> elements
# ---------------------------------------------------------------------------------------------


def test_round_trip_elements_state_elements_random_grid() -> None:
    # Seeded grid away from the singular cases (e >= 1e-3, 1e-3 <= i <= pi - 1e-3).
    # Tolerances follow the conditioning of each element:
    # * a from the energy: relative error ~ eps * 2 a v^2 / mu <= eps * 2 (1+e)/(1-e)
    #   (<= 8.7e-15 at e = 0.95), so 1e-12 leaves > 100x margin (measured 2.1e-14).
    # * e, i: absolute ~ few eps (measured <= 1.4e-15); bound 1e-13.
    # * raan is ill-conditioned as 1/sin(i) and argp, nu as 1/e: we bound sin(i)*d(raan) and
    #   e*d(argp), e*d(nu) by 1e-13 (measured <= 2.5e-15).
    rng = np.random.default_rng(20261112)
    for el in _random_elements(rng, 2000):
        back = state_to_elements(elements_to_state(el), MU)
        assert back.a_m == pytest.approx(el.a_m, rel=1e-12, abs=0)
        assert back.e == pytest.approx(el.e, abs=1e-13)
        assert back.i_rad == pytest.approx(el.i_rad, abs=1e-13)
        assert abs(_wrap(back.raan_rad - el.raan_rad)) * math.sin(el.i_rad) <= 1e-13
        assert abs(_wrap(back.argp_rad - el.argp_rad)) * el.e <= 1e-13
        assert abs(_wrap(back.nu_rad - el.nu_rad)) * el.e <= 1e-13
        assert back.epoch_tdb_s == el.epoch_tdb_s
        assert back.mu_m3_s2 == MU
        assert back.frame == el.frame
        for ang in (back.raan_rad, back.argp_rad, back.nu_rad):
            assert 0.0 <= ang < 2 * math.pi


def test_round_trip_state_elements_state_random_grid() -> None:
    # state -> elements -> state is well conditioned everywhere: relative error ~10 eps
    # (measured <= 8.8e-15); bound 1e-13.
    rng = np.random.default_rng(99)
    for el in _random_elements(rng, 1000):
        s = elements_to_state(el)
        s2 = elements_to_state(state_to_elements(s, MU))
        assert np.linalg.norm(s2.position_m - s.position_m) <= 1e-13 * np.linalg.norm(s.position_m)
        assert np.linalg.norm(s2.velocity_m_s - s.velocity_m_s) <= 1e-13 * np.linalg.norm(
            s.velocity_m_s
        )


NEAR_E = [0.0, 1e-16, 1e-14, 1e-12, 1e-9, 1e-6]
# 0.7 rad is a clearly inclined value: with it the argument-of-latitude check (u = argp + nu,
# the circular convention) runs for near-circular inclined orbits, not only at e = 0 exactly.
NEAR_I = [0.0, 1e-16, 1e-14, 1e-12, 1e-9, 0.7, math.pi - 1e-9, math.pi - 1e-14, math.pi]


@pytest.mark.parametrize("e", NEAR_E)
@pytest.mark.parametrize("i", NEAR_I)
def test_near_singular_state_round_trip(e: float, i: float) -> None:
    # Near-circular / near-equatorial orbits: individual angles are ill-conditioned (or set by
    # convention), but the state they encode must be reproduced. Below the circular/equatorial
    # thresholds (1e-13) the conventions set e = 0 / i in {0, pi}; the state then changes by
    # at most ~2 e a or r sin(i) (<= 2e-13 relative); above them only round-off remains.
    # Bound: 1e-12 relative.
    # Distinct, reproducible draws per (e, i) case: seed from the parameter indices.
    rng = np.random.default_rng([NEAR_E.index(e), NEAR_I.index(i)])
    n_inclined_checks = 0
    for _ in range(20):
        el = _el(
            a_m=rng.uniform(0.2, 30) * AU_M,
            e=e,
            i_rad=i,
            raan_rad=rng.uniform(0, 2 * math.pi),
            argp_rad=rng.uniform(0, 2 * math.pi),
            nu_rad=rng.uniform(0, 2 * math.pi),
        )
        s = elements_to_state(el)
        back = state_to_elements(s, MU)
        s2 = elements_to_state(back)
        assert np.linalg.norm(s2.position_m - s.position_m) <= 1e-12 * np.linalg.norm(s.position_m)
        assert np.linalg.norm(s2.velocity_m_s - s.velocity_m_s) <= 1e-12 * np.linalg.norm(
            s.velocity_m_s
        )
        assert back.a_m == pytest.approx(el.a_m, rel=1e-12, abs=0)
        # Well-defined angle combinations: the argument of latitude u = argp + nu for clearly
        # inclined orbits; for (near-)equatorial ones the node is ill-conditioned (error
        # ~eps/sin i) and only the true longitude raan + u (prograde) or raan - u
        # (retrograde, clockwise) is meaningful.
        u_in = el.argp_rad + el.nu_rad
        u_out = back.argp_rad + back.nu_rad
        if 1e-6 < i < math.pi - 1e-6:
            n_inclined_checks += 1
            assert abs(_wrap(u_out - u_in)) <= 1e-12
            if e < DEFAULT_CIRCULAR_TOL:
                # Circular convention: e snapped to 0, argp = 0, nu = argument of latitude.
                assert back.e == 0.0
                assert back.argp_rad == 0.0
                assert abs(_wrap(back.nu_rad - u_in)) <= 1e-12
        elif i < math.pi / 2:
            assert abs(_wrap(back.raan_rad + u_out - el.raan_rad - u_in)) <= 1e-12
        else:
            assert abs(_wrap(back.raan_rad - u_out - el.raan_rad + u_in)) <= 1e-12
    # The inclined branch must actually have run for the inclined parameter value.
    assert n_inclined_checks == (20 if 1e-6 < i < math.pi - 1e-6 else 0)


def test_returned_angles_are_in_half_open_range_at_the_wrap_point() -> None:
    # Python's float modulo maps a tiny negative angle to exactly 2 pi
    # ((-1e-17) % (2 pi) == 2 pi), which violates the documented [0, 2 pi). Two inputs that
    # land exactly there before the fix (reviewer evidence, re-derived here):
    assert (-1e-17) % (2 * math.pi) == 2 * math.pi  # premise of the test
    # (1) raan = 2 pi - (tiny): the node atan2 comes out as a tiny negative angle.
    el = _el(e=0.1, i_rad=0.5, raan_rad=(-1e-17) % (2 * math.pi), argp_rad=1.0, nu_rad=2.0)
    back = state_to_elements(elements_to_state(el), MU)
    # (2) circular equatorial state a hair below the +x axis: true longitude ~ -6.7e-18 rad.
    v_c = math.sqrt(MU / AU_M)
    circ = CartesianState(
        epoch_tdb_s=0.0,
        position_m=np.array([AU_M, -1e-6, 0.0]),
        velocity_m_s=np.array([0.0, v_c, 0.0]),
        frame="ECLIPJ2000",
        origin="SUN",
    )
    back_c = state_to_elements(circ, MU)
    for b in (back, back_c):
        for ang in (b.raan_rad, b.argp_rad, b.nu_rad):
            assert 0.0 <= ang < 2 * math.pi
    # The value is the wrapped one (0, not some other number): raan ~ 0 and the true
    # longitude ~ 0 within the size of the perturbation.
    assert abs(_wrap(back.raan_rad)) <= 1e-15
    assert back_c.nu_rad < 1e-15


def test_circular_convention_argp_zero_nu_is_argument_of_latitude() -> None:
    el = _el(e=0.0, i_rad=0.7, raan_rad=1.1, argp_rad=2.0, nu_rad=0.5)
    back = state_to_elements(elements_to_state(el), MU)
    assert back.e == 0.0
    assert back.argp_rad == 0.0
    # nu = u = argp + nu of the input (the input split is not recoverable).
    assert back.nu_rad == pytest.approx(2.5, abs=1e-14)
    assert back.raan_rad == pytest.approx(1.1, abs=1e-14)


def test_equatorial_prograde_convention_raan_zero_argp_is_longitude_of_periapsis() -> None:
    el = _el(e=0.2, i_rad=0.0, raan_rad=1.0, argp_rad=2.0, nu_rad=0.3)
    back = state_to_elements(elements_to_state(el), MU)
    assert back.i_rad == 0.0
    assert back.raan_rad == 0.0
    assert back.argp_rad == pytest.approx(3.0, abs=1e-13)  # longitude of periapsis raan+argp
    assert back.nu_rad == pytest.approx(0.3, abs=1e-13)


def test_equatorial_retrograde_convention() -> None:
    # i = pi: angles are measured clockwise seen from +z. With the node fixed at +x, the
    # periapsis direction (cos(raan - argp), sin(raan - argp)) gives argp' = argp - raan.
    el = _el(e=0.2, i_rad=math.pi, raan_rad=1.0, argp_rad=2.5, nu_rad=0.3)
    s = elements_to_state(el)
    back = state_to_elements(s, MU)
    assert back.i_rad == math.pi
    assert back.raan_rad == 0.0
    assert back.argp_rad == pytest.approx(1.5, abs=1e-13)
    assert back.nu_rad == pytest.approx(0.3, abs=1e-13)
    s2 = elements_to_state(back)
    # Norm-relative: the z components are pure round-off (sin(pi) * r ~ 1e-5 m).
    assert np.linalg.norm(s2.position_m - s.position_m) <= 1e-14 * np.linalg.norm(s.position_m)


def test_circular_equatorial_true_longitude() -> None:
    # Prograde circular equatorial at true longitude 200 deg.
    lam = math.radians(200.0)
    s = CartesianState(
        0.0,
        AU_M * np.array([math.cos(lam), math.sin(lam), 0.0]),
        math.sqrt(MU / AU_M) * np.array([-math.sin(lam), math.cos(lam), 0.0]),
        "ECLIPJ2000",
        "SUN",
    )
    back = state_to_elements(s, MU)
    assert (back.e, back.i_rad, back.raan_rad, back.argp_rad) == (0.0, 0.0, 0.0, 0.0)
    assert back.nu_rad == pytest.approx(lam, abs=1e-15)
    assert back.a_m == pytest.approx(AU_M, rel=1e-15, abs=0)
    # Retrograde: velocity reversed -> i = pi, longitude measured clockwise: 360 - 200 deg.
    s_retro = CartesianState(0.0, s.position_m, -s.velocity_m_s, "ECLIPJ2000", "SUN")
    back = state_to_elements(s_retro, MU)
    assert back.i_rad == math.pi
    assert back.nu_rad == pytest.approx(2 * math.pi - lam, abs=1e-14)


def test_circular_threshold_is_respected() -> None:
    # Just above the threshold the periapsis is kept; just below it is dropped.
    for e, circular in ((10 * DEFAULT_CIRCULAR_TOL, False), (0.1 * DEFAULT_CIRCULAR_TOL, True)):
        el = _el(e=e, i_rad=0.5, argp_rad=1.0, nu_rad=0.2)
        back = state_to_elements(elements_to_state(el), MU)
        assert (back.e == 0.0) is circular


def test_state_to_elements_rejects_unbound_and_wrong_origin() -> None:
    r = np.array([AU_M, 0.0, 0.0])
    v_esc = math.sqrt(2 * MU / AU_M)
    with pytest.raises(ValueError, match="not elliptic"):
        state_to_elements(CartesianState(0.0, r, [0.0, v_esc, 0.0], "J2000", "SUN"), MU)
    with pytest.raises(ValueError, match="not elliptic"):
        state_to_elements(CartesianState(0.0, r, [0.0, 1.5 * v_esc, 0.0], "J2000", "SUN"), MU)
    with pytest.raises(ValueError, match="rectilinear"):
        state_to_elements(CartesianState(0.0, r, [1.0, 0.0, 0.0], "J2000", "SUN"), MU)
    with pytest.raises(ValueError, match="origin"):
        state_to_elements(
            CartesianState(0.0, r, [0.0, 3e4, 0.0], "J2000", "SOLAR SYSTEM BARYCENTER"), MU
        )
    with pytest.raises(ValueError, match="mu"):
        state_to_elements(CartesianState(0.0, r, [0.0, 3e4, 0.0], "J2000", "SUN"), -1.0)


def test_state_to_elements_keeps_frame() -> None:
    el = _el(e=0.1, i_rad=0.3, frame="J2000")
    back = state_to_elements(elements_to_state(el), MU)
    assert back.frame == "J2000"


# ---------------------------------------------------------------------------------------------
# Configuration hook
# ---------------------------------------------------------------------------------------------


def test_elements_from_mvp_config() -> None:
    cfg = parse_config(load_mvp_data())
    el = elements_from_config(cfg.orbit, epoch_tdb_s=946_728_069.18)
    assert el.a_m == AU_M  # 1.0 au, exact
    assert el.e == 0.0
    assert el.nu_rad == pytest.approx(math.pi / 6, rel=1e-15, abs=0)  # 30 deg
    assert el.mu_m3_s2 == GM_SUN_M3_S2
    assert el.frame == "ECLIPJ2000"
    assert el.center == "SUN"
    assert el.epoch_tdb_s == 946_728_069.18
    s = elements_to_state(el)
    assert np.linalg.norm(s.position_m) == pytest.approx(AU_M, rel=1e-15, abs=0)
    assert s.position_m == pytest.approx(
        AU_M * np.array([math.cos(math.pi / 6), 0.5, 0.0]), abs=1e-4
    )


def test_elements_from_config_converts_every_angle_and_au() -> None:
    data = load_mvp_data()
    set_path(data, "orbit.semimajor_axis_au", 2.5)
    set_path(data, "orbit.eccentricity", 0.2)
    set_path(data, "orbit.inclination_deg", 10.0)
    set_path(data, "orbit.raan_deg", 20.0)
    set_path(data, "orbit.arg_periapsis_deg", 30.0)
    set_path(data, "orbit.true_anomaly_deg", 40.0)
    el = elements_from_config(parse_config(data).orbit, epoch_tdb_s=0.0, mu_m3_s2=1.0e20)
    assert el.a_m == pytest.approx(2.5 * 149_597_870_700.0, rel=1e-16, abs=0)
    assert el.e == 0.2
    expected = [math.radians(v) for v in (10.0, 20.0, 30.0, 40.0)]
    got = [el.i_rad, el.raan_rad, el.argp_rad, el.nu_rad]
    assert got == pytest.approx(expected, rel=1e-15, abs=0)
    assert el.mu_m3_s2 == 1.0e20
