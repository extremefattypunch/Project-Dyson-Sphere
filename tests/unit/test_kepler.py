"""Tests for the anomaly conversions and the Kepler-equation solver (helioswarm.orbit).

Independent references: the half-angle relation tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2)
(a different formula from the implementation's Broucke-Cefola form), the orbit-equation
identity a (1 - e cos E) = p / (1 + e cos nu), and scipy's ``brentq`` bracketing root finder.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from helioswarm.orbit import (
    KeplerConvergenceError,
    eccentric_from_mean,
    eccentric_from_true,
    mean_from_eccentric,
    mean_from_true,
    solve_kepler,
    true_from_eccentric,
    true_from_mean,
)

ECCS = [0.0, 1e-12, 1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.999999]
EPS = float(np.finfo(np.float64).eps)


def _wrap(x: np.ndarray) -> np.ndarray:
    return np.asarray((x + np.pi) % (2 * np.pi) - np.pi)


def _mean_anomaly_grid() -> np.ndarray:
    # Dense grid over several revolutions in both directions, plus the delicate points
    # M = 0, +-pi and multiples of 2 pi and values a few ulps from them.
    special = np.array([0.0, math.pi, -math.pi, 2 * math.pi, -2 * math.pi, 1e-300, 1e-12])
    near = np.concatenate([np.nextafter(special, np.inf), np.nextafter(special, -np.inf)])
    return np.concatenate([np.linspace(-20.0, 20.0, 40_001), special, near])


@pytest.mark.parametrize("e", ECCS)
def test_kepler_residual_at_round_off(e: float) -> None:
    m = _mean_anomaly_grid()
    sol = solve_kepler(m, e)
    big_e = sol.eccentric_anomaly_rad
    # Backward error on the reduced anomaly (|x| <= pi): a few ulps of pi. On the unreduced
    # M (|M| <= 20) the representation error is eps * |M| itself, hence the second bound.
    assert sol.max_abs_residual_rad <= 4 * EPS * math.pi
    assert np.max(np.abs(big_e - e * np.sin(big_e) - m)) <= 4 * EPS * 20.0
    # Same revolution as M: |E - M| = e |sin E| <= e.
    assert np.all(np.abs(big_e - m) <= e + 8 * EPS * 20.0)
    # Newton (not bisection): bracketed Newton needs few iterations; plain bisection from a
    # bracket of width e would need ~log2(e/1e-12) ~ 40.
    assert sol.iterations <= (12 if e <= 0.999 else 25)


@pytest.mark.parametrize("e", [0.05, 0.3, 0.6, 0.9])
def test_kepler_matches_independent_brentq(e: float) -> None:
    rng = np.random.default_rng(42)
    m = rng.uniform(-3 * math.pi, 3 * math.pi, size=300)
    big_e = solve_kepler(m, e).eccentric_anomaly_rad
    ref = np.array(
        [
            brentq(lambda x, mm=mm: x - e * math.sin(x) - mm, mm - 1.0, mm + 1.0, xtol=1e-15)
            for mm in m
        ]
    )
    # Forward error <= residual / (1 - e) <= ~1e-14 for e <= 0.9; brentq xtol 1e-15 plus
    # its rtol 4 eps |E| (|E| <= 10) -> 1e-13 bound.
    assert np.max(np.abs(big_e - ref)) <= 1e-13


def test_kepler_circular_is_identity_and_scalar_shape() -> None:
    m = np.linspace(-10, 10, 101)
    assert np.array_equal(eccentric_from_mean(m, 0.0), m)
    out = solve_kepler(1.0, 0.5)
    assert out.eccentric_anomaly_rad.shape == ()
    # Known value: E for M = 1 rad, e = 0.5 verified by substitution.
    big_e = float(out.eccentric_anomaly_rad)
    assert big_e - 0.5 * math.sin(big_e) == pytest.approx(1.0, abs=1e-15)
    grid = solve_kepler(np.zeros((2, 3)), 0.3)
    assert grid.eccentric_anomaly_rad.shape == (2, 3)


def test_kepler_known_points() -> None:
    # E = M at M = 0 and M = pi for every e (apsides).
    for e in (0.2, 0.9, 0.999):
        big_e = solve_kepler(np.array([0.0, math.pi, 4 * math.pi]), e).eccentric_anomaly_rad
        # abs 1e-14: restoring the removed 2 pi multiples at |M| = 4 pi costs ~eps * 4 pi.
        assert big_e == pytest.approx([0.0, math.pi, 4 * math.pi], abs=1e-14)


def _e_minus_sin_e_series(x: float) -> float:
    """x - sin x by its Taylor series (|x| <= 5e-2: truncation x^11/11! < 1e-23 x^3/6)."""
    x2 = x * x
    return x * x2 * (1 / 6 - x2 * (1 / 120 - x2 * (1 / 5040 - x2 / 362_880)))


@pytest.mark.parametrize("one_minus_e", [1e-8, 1e-9, 1e-10])
def test_kepler_near_parabolic_small_mean_anomaly_converges_accurately(
    one_minus_e: float,
) -> None:
    # Regression (M1 review): for 1e-10 <= 1 - e <= 1e-8 and small M, f' = 1 - e cos E ~ 1e-8
    # at the root while f is evaluated with an absolute error ~ eps E, so the root is only
    # determined to ~eps E / f' (> 1e-12 rad) and a pure |dE| <= 1e-12 stop 2-cycles and
    # raised. The solver must converge (backward criterion) and be as accurate as float64
    # allows.
    e = 1.0 - one_minus_e
    ome = 1.0 - e  # exact (Sterbenz), the 1 - e the solver sees
    m = np.concatenate(
        [
            [1.5301179982083837e-12],  # reviewer's failing input at 1 - e = 1e-8
            np.logspace(-320, math.log10(3.1), 4001),  # reviewer's scan range, thinned
        ]
    )
    sol = solve_kepler(m, e)  # raised KeplerConvergenceError before the fix
    big_e = sol.eccentric_anomaly_rad
    # Backward error at round-off on the reduced anomaly (same bound as
    # test_kepler_residual_at_round_off; |M| < pi here so no reduction error enters).
    assert sol.max_abs_residual_rad <= 4 * EPS * math.pi
    assert np.max(np.abs(big_e - e * np.sin(big_e) - m)) <= 4 * EPS * math.pi
    # Forward error against an independent, cancellation-free reference: brentq on
    # (1 - e) E + e (E - sin E) - M with E - sin E from its series (no subtraction of nearly
    # equal terms), for M where the root is < 5e-2. Bound: f is evaluated with error
    # <= ~3 eps E and accepted at |f| <= 4 eps E, so |dE| <= ~7 eps E / f'; 8 for margin
    # (measured <= 0.51 of this bound).
    for mm, ee in zip(m, big_e, strict=True):
        if not (1e-14 <= mm <= 1e-6):
            continue
        ref = brentq(
            lambda x, mm=mm: ome * x + e * _e_minus_sin_e_series(x) - mm,
            0.0,
            5e-2,
            xtol=1e-300,
            rtol=4 * EPS,
            maxiter=500,
        )
        fp = 1.0 - e * math.cos(ref)
        assert abs(ee - ref) <= 8 * EPS * ref / fp


def test_kepler_non_convergence_raises_with_diagnostic() -> None:
    with pytest.raises(KeplerConvergenceError, match=r"did not converge .* after 1 iterations"):
        solve_kepler(np.array([1.0, 2.0]), 0.5, max_iter=1)
    with pytest.raises(KeplerConvergenceError, match=r"e=0\.9"):
        solve_kepler(0.1, 0.9, max_iter=2, tol_rad=1e-300)


@pytest.mark.parametrize("bad_e", [1.0, 1.5, -0.1, math.nan])
def test_kepler_rejects_non_elliptic(bad_e: float) -> None:
    with pytest.raises(ValueError, match="eccentricity"):
        solve_kepler(0.3, bad_e)


def test_kepler_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        solve_kepler(np.array([0.0, np.inf]), 0.1)
    with pytest.raises(ValueError, match="tol_rad"):
        solve_kepler(0.0, 0.1, tol_rad=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        solve_kepler(0.0, 0.1, max_iter=0)


@pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9, 0.99])
def test_true_from_eccentric_against_half_angle_formula(e: float) -> None:
    big_e = np.linspace(-math.pi + 1e-3, math.pi - 1e-3, 2001)
    nu = true_from_eccentric(big_e, e)
    nu_ref = 2.0 * np.arctan(math.sqrt((1 + e) / (1 - e)) * np.tan(big_e / 2.0))
    # Half-angle formula conditioning near E = +-pi: tan(E/2) ~ 2/1e-3; 1e-12 covers it.
    assert np.max(np.abs(_wrap(nu - nu_ref))) <= 1e-12
    # Orbit-equation identity r/a: (1 - e cos E) = (1 - e^2) / (1 + e cos nu).
    assert 1 - e * np.cos(big_e) == pytest.approx(
        (1 - e * e) / (1 + e * np.cos(nu)), rel=1e-14, abs=0
    )


@pytest.mark.parametrize("e", [0.0, 0.3, 0.9, 0.999])
def test_anomaly_conversions_are_continuous_inverses(e: float) -> None:
    big_e = np.linspace(-20.0, 20.0, 20_001)
    nu = true_from_eccentric(big_e, e)
    assert np.all(np.diff(nu) > 0)  # continuous and monotonic across revolutions
    assert np.all(np.abs(nu - big_e) < math.pi)  # same revolution
    assert eccentric_from_true(nu, e) == pytest.approx(big_e, abs=1e-12)
    m = mean_from_eccentric(big_e, e)
    assert mean_from_true(nu, e) == pytest.approx(m, abs=1e-12)
    # Forward error of nu(M) = (backward error of the Kepler solve in M, <= 4 eps |M| with
    # |M| <= 20) x (max dnu/dM = sqrt((1 + e) / (1 - e)^3), at periapsis), plus the round-off
    # of nu itself (~ a few eps |nu|, |nu| <~ 20). Measured: 0, 7.1e-15, 1.1e-13, 8.3e-11 for
    # e = 0, 0.3, 0.9, 0.999; bounds: 2.1e-14, 5.2e-14, 8.1e-13, 8.0e-10.
    tol_nu = 4 * EPS * 20.0 * math.sqrt((1 + e) / (1 - e) ** 3) + 8 * EPS * 20.0
    assert true_from_mean(m, e) == pytest.approx(nu, rel=0, abs=tol_nu)


def test_true_from_mean_apsides_and_quadrant() -> None:
    e = 0.6
    # Mean anomaly of the semi-latus rectum point nu = pi/2: E = acos(e) (cos E = e),
    # M = E - e sin E (independent closed form).
    big_e = math.acos(e)
    m = big_e - e * math.sin(big_e)
    assert float(true_from_mean(m, e)) == pytest.approx(math.pi / 2, abs=1e-14)
    assert float(true_from_mean(-m, e)) == pytest.approx(-math.pi / 2, abs=1e-14)
