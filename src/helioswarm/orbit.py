"""Analytic two-body (Keplerian) orbits: the L0 propagation level (PDF §6.2).

Contents:

* :class:`OrbitalElements`: classical osculating elements of an elliptic orbit about a
  central body, with explicit gravitational parameter, frame, centre and TDB epoch.
* :func:`elements_to_state` / :func:`state_to_elements`: conversions to and from a
  frame-tagged :class:`~helioswarm.frames.CartesianState`.
* Anomaly conversions (true <-> eccentric <-> mean) and a vectorised, bracketed Newton solver
  for Kepler's equation :func:`solve_kepler` that raises :class:`KeplerConvergenceError`
  on failure.
* :func:`propagate_kepler`: closed-form two-body propagation to an array of epochs.
* Invariants: specific orbital energy, specific angular momentum, eccentricity vector, period.
* :func:`elements_from_config`: builds elements from the validated ``orbit`` config section
  (au -> m and deg -> rad conversions happen in the config accessors, i.e. at the boundary).

Governing equations (Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
ch. 2; Curtis, *Orbital Mechanics for Engineering Students*, ch. 2-4):

* Kepler's equation ``M = E - e sin E``; mean motion ``n = sqrt(mu / a^3)``;
  period ``T = 2 pi sqrt(a^3 / mu)`` (Kepler's third law).
* Orbit equation ``r = p / (1 + e cos nu)``, ``p = a (1 - e^2)``.
* Vis-viva / energy ``epsilon = v^2/2 - mu/r = -mu / (2a)``;
  angular momentum ``h = r x v``, ``|h| = sqrt(mu p)``.
* Eccentricity vector ``e_vec = (v x h)/mu - r/|r|``.

Scope and validity: elliptic orbits only (0 <= e < 1, a > 0). A massless test particle orbits
a point mass; ``mu`` is the central body's GM (GM_sun for heliocentric elements, so the
satellite's own mass is neglected). No perturbations; those enter at L1 (M2).

Singular-case conventions (shared by both directions so that round trips are consistent):

* **Equatorial** (``sin i < equatorial_tol``): ``i`` is set to exactly 0 (prograde) or pi
  (retrograde) and the node is undefined, so ``raan = 0`` and the line of nodes is the
  frame's +x axis.
* **Circular** (``e < circular_tol``): ``e`` is set to exactly 0, the periapsis is undefined,
  so ``argp = 0`` and ``nu`` is the *argument of latitude* ``u`` (angle from the node line to
  the position, measured in the direction of motion).
* **Circular and equatorial**: both of the above, so ``nu`` is the *true longitude* measured
  from +x in the direction of motion.

All angles are computed with ``atan2`` of components in the orbital-plane basis
``P = (cos raan, sin raan, 0)`` (node direction) and ``Q = h_hat x P``, which is valid for every
inclination including the singular ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.constants import GM_SUN_M3_S2
from helioswarm.frames import (
    SUPPORTED_FRAMES,
    CartesianState,
    CartesianTrajectory,
    FrameName,
    OriginName,
)

if TYPE_CHECKING:
    from helioswarm.config import OrbitConfig

Float64Array = NDArray[np.float64]

TWO_PI: Final[float] = 2.0 * math.pi


def _wrap_2pi(angle_rad: float) -> float:
    """Reduce an angle to [0, 2 pi) [rad].

    Python's ``x % TWO_PI`` returns exactly ``TWO_PI`` for tiny negative x (e.g. -1e-17,
    because x + 2 pi rounds to 2 pi); that value is mapped to 0.0 so the half-open range holds.
    """
    r = angle_rad % TWO_PI
    return 0.0 if r >= TWO_PI else r


DEFAULT_KEPLER_TOL_RAD: Final[float] = 1e-12
"""Default convergence threshold on the magnitude of the last Newton correction |dE| [rad].
Newton converges quadratically, so the error remaining after a correction of size dE is of
order e dE^2 / (2 (1 - e)), i.e. at float64 round-off for any e < 1 - 1e-6. The solver also
stops on a round-off-level residual (see :data:`KEPLER_RESIDUAL_ULP_FACTOR`), because near
e -> 1 the root is not determined to 1e-12 rad in float64 at all."""

KEPLER_RESIDUAL_ULP_FACTOR: Final[float] = 4.0
"""Backward-error stopping factor c of the Kepler solver [1]. The iteration stops (after that
iteration's Newton step) when ``|E - e sin E - x| <= c * eps * max(x, E)`` (eps = float64
machine epsilon): the residual is then at the level of the rounding error of evaluating f
itself (about 2 eps E + eps x), so further Newton steps cannot reduce it systematically.
Without this test Newton can fall into a round-off 2-cycle when f' = 1 - e cos E ~ 1e-8
(1 - e <~ 1e-8, small M) and |dE| never drops below 1e-12."""

DEFAULT_KEPLER_MAX_ITER: Final[int] = 50
"""Default iteration cap of the Kepler solver. The bracketed Newton iteration normally needs
3-6 iterations; the cap only bounds pathological input."""

DEFAULT_CIRCULAR_TOL: Final[float] = 1e-13
"""Eccentricity below which :func:`state_to_elements` treats the orbit as circular. Chosen well
above the float64 noise of |e_vec| for an exactly circular state (measured <= ~1e-15) and small
enough that the induced state change (<= ~2 e a) is < 0.1 m at 1 au."""

DEFAULT_EQUATORIAL_TOL: Final[float] = 1e-13
"""sin(i) below which :func:`state_to_elements` treats the orbit as equatorial. The induced
position change is <= ~r sin(i), i.e. < 0.02 m at 1 au."""


class KeplerConvergenceError(RuntimeError):
    """Raised when the Kepler-equation solver does not converge within its iteration cap."""


# ---------------------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitalElements:
    """Classical osculating elements of an elliptic orbit.

    Attributes:
        a_m: Semi-major axis a > 0 [m].
        e: Eccentricity 0 <= e < 1 [1].
        i_rad: Inclination 0 <= i <= pi [rad], relative to the x-y plane of ``frame``.
        raan_rad: Right ascension / longitude of the ascending node [rad] (0 by convention
            for equatorial orbits).
        argp_rad: Argument of periapsis [rad] (0 by convention for circular orbits).
        nu_rad: True anomaly at ``epoch_tdb_s`` [rad] (argument of latitude for circular,
            true longitude for circular equatorial orbits).
        mu_m3_s2: Gravitational parameter of the central body [m^3 s^-2].
        epoch_tdb_s: Osculation epoch [s], TDB seconds past J2000 (SPICE ET).
        frame: Inertial reference frame of the angles (SPICE name); default ``ECLIPJ2000``.
        center: Central body (SPICE name): ``SUN`` (default) or ``EARTH`` (geocentric
            elements, M7b); ``mu_m3_s2`` must be that body's GM.

    Angles other than ``i_rad`` may be any finite value (they are used modulo 2 pi);
    :func:`state_to_elements` returns them in [0, 2 pi).
    """

    a_m: float
    e: float
    i_rad: float
    raan_rad: float
    argp_rad: float
    nu_rad: float
    mu_m3_s2: float
    epoch_tdb_s: float
    frame: FrameName = field(default="ECLIPJ2000")
    center: OriginName = field(default="SUN")

    def __post_init__(self) -> None:
        values = (
            self.a_m,
            self.e,
            self.i_rad,
            self.raan_rad,
            self.argp_rad,
            self.nu_rad,
            self.mu_m3_s2,
            self.epoch_tdb_s,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"orbital elements must be finite, got {values}")
        if self.a_m <= 0.0:
            raise ValueError(f"a_m must be > 0 (elliptic orbits only), got {self.a_m}")
        if not 0.0 <= self.e < 1.0:
            raise ValueError(f"e must satisfy 0 <= e < 1 (elliptic orbits only), got {self.e}")
        if not 0.0 <= self.i_rad <= math.pi:
            raise ValueError(f"i_rad must be in [0, pi], got {self.i_rad}")
        if self.mu_m3_s2 <= 0.0:
            raise ValueError(f"mu_m3_s2 must be > 0, got {self.mu_m3_s2}")
        if self.frame not in SUPPORTED_FRAMES:
            raise ValueError(f"unsupported frame {self.frame!r}")
        if self.center not in ("SUN", "EARTH"):
            raise ValueError(f"center must be 'SUN' or 'EARTH', got {self.center!r}")

    @property
    def semi_latus_rectum_m(self) -> float:
        """Semi-latus rectum p = a (1 - e^2) [m]."""
        return self.a_m * (1.0 - self.e * self.e)

    @property
    def periapsis_radius_m(self) -> float:
        """Periapsis distance a (1 - e) [m]."""
        return self.a_m * (1.0 - self.e)

    @property
    def apoapsis_radius_m(self) -> float:
        """Apoapsis distance a (1 + e) [m]."""
        return self.a_m * (1.0 + self.e)

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion n = sqrt(mu / a^3) [rad s^-1]."""
        return mean_motion_rad_s(self.a_m, self.mu_m3_s2)

    @property
    def period_s(self) -> float:
        """Orbital period T = 2 pi sqrt(a^3 / mu) [s] (Kepler's third law)."""
        return orbital_period_s(self.a_m, self.mu_m3_s2)

    @property
    def specific_energy_J_kg(self) -> float:
        """Specific orbital energy -mu / (2a) [J kg^-1 = m^2 s^-2]."""
        return -self.mu_m3_s2 / (2.0 * self.a_m)

    @property
    def specific_angular_momentum_m2_s(self) -> float:
        """Magnitude of the specific angular momentum sqrt(mu p) [m^2 s^-1]."""
        return math.sqrt(self.mu_m3_s2 * self.semi_latus_rectum_m)

    @property
    def mean_anomaly_rad(self) -> float:
        """Mean anomaly at the epoch [rad], on the same revolution as ``nu_rad``."""
        return float(mean_from_true(self.nu_rad, self.e))


@dataclass(frozen=True)
class KeplerSolution:
    """Result of :func:`solve_kepler`.

    Attributes:
        eccentric_anomaly_rad: E with ``E - e sin E = M``, same shape as M, on the same
            revolution as M (``|E - M| <= e``) [rad].
        iterations: Number of iterations used (maximum over all elements).
        max_abs_residual_rad: max |E - e sin E - M| evaluated on the reduced anomaly in
            [-pi, pi] [rad] (backward error of the solution).
    """

    eccentric_anomaly_rad: Float64Array
    iterations: int
    max_abs_residual_rad: float


@dataclass(frozen=True)
class KeplerPropagation:
    """Result of :func:`propagate_kepler`.

    Attributes:
        trajectory: States at the requested epochs, in ``elements.frame`` with origin
            ``elements.center``.
        elements: The osculating elements at the initial epoch that were propagated.
        kepler_iterations: Maximum Kepler-solver iteration count over all epochs.
        max_kepler_residual_rad: Maximum backward error of Kepler's equation [rad].
    """

    trajectory: CartesianTrajectory
    elements: OrbitalElements
    kepler_iterations: int
    max_kepler_residual_rad: float


# ---------------------------------------------------------------------------------------------
# Scalar relations and invariants
# ---------------------------------------------------------------------------------------------


def _check_mu(mu_m3_s2: float) -> None:
    if not (math.isfinite(mu_m3_s2) and mu_m3_s2 > 0.0):
        raise ValueError(f"mu_m3_s2 must be finite and > 0, got {mu_m3_s2}")


def _check_e(e: float) -> None:
    if not (math.isfinite(e) and 0.0 <= e < 1.0):
        raise ValueError(f"eccentricity must satisfy 0 <= e < 1, got {e}")


def mean_motion_rad_s(a_m: float, mu_m3_s2: float) -> float:
    """Mean motion n = sqrt(mu / a^3) [rad s^-1] of an elliptic orbit (Kepler's third law)."""
    _check_mu(mu_m3_s2)
    if not a_m > 0.0:
        raise ValueError(f"a_m must be > 0, got {a_m}")
    return math.sqrt(mu_m3_s2 / a_m**3)


def orbital_period_s(a_m: float, mu_m3_s2: float) -> float:
    """Orbital period T = 2 pi sqrt(a^3 / mu) [s] (Kepler's third law)."""
    return TWO_PI / mean_motion_rad_s(a_m, mu_m3_s2)


def specific_energy_J_kg(
    position_m: ArrayLike, velocity_m_s: ArrayLike, mu_m3_s2: float
) -> Float64Array:
    """Specific orbital energy epsilon = |v|^2/2 - mu/|r| [J kg^-1] (vis-viva integral).

    Args:
        position_m: Shape (..., 3) [m].
        velocity_m_s: Shape (..., 3) [m s^-1].
        mu_m3_s2: Gravitational parameter [m^3 s^-2].

    Returns:
        Array of shape (...).
    """
    _check_mu(mu_m3_s2)
    r = np.asarray(position_m, dtype=np.float64)
    v = np.asarray(velocity_m_s, dtype=np.float64)
    out: Float64Array = 0.5 * np.einsum("...i,...i->...", v, v) - mu_m3_s2 / np.linalg.norm(
        r, axis=-1
    )
    return out


def specific_angular_momentum_m2_s(position_m: ArrayLike, velocity_m_s: ArrayLike) -> Float64Array:
    """Specific angular momentum vector h = r x v [m^2 s^-1], shape (..., 3)."""
    r = np.asarray(position_m, dtype=np.float64)
    v = np.asarray(velocity_m_s, dtype=np.float64)
    return np.asarray(np.cross(r, v), dtype=np.float64)


def eccentricity_vector(
    position_m: ArrayLike, velocity_m_s: ArrayLike, mu_m3_s2: float
) -> Float64Array:
    """Eccentricity (Laplace-Runge-Lenz / mu) vector e = (v x h)/mu - r/|r| [1], shape (..., 3).

    Points to periapsis with magnitude e (Vallado 4th ed., ch. 2).
    """
    _check_mu(mu_m3_s2)
    r = np.asarray(position_m, dtype=np.float64)
    v = np.asarray(velocity_m_s, dtype=np.float64)
    h = np.cross(r, v)
    return np.asarray(
        np.cross(v, h) / mu_m3_s2 - r / np.linalg.norm(r, axis=-1, keepdims=True),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------------------------


def _beta(e: float) -> float:
    """beta = e / (1 + sqrt(1 - e^2)), the half-angle parameter of Broucke & Cefola (1973)."""
    return e / (1.0 + math.sqrt(1.0 - e * e))


def true_from_eccentric(eccentric_anomaly_rad: ArrayLike, e: float) -> Float64Array:
    """True anomaly from eccentric anomaly [rad].

    Uses the continuous form ``nu = E + 2 atan2(beta sin E, 1 - beta cos E)`` with
    ``beta = e / (1 + sqrt(1 - e^2))`` (Broucke & Cefola 1973, Celest. Mech. 7, 388),
    equivalent to ``tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2)`` but free of the quadrant
    ambiguity and continuous across revolutions (``nu`` stays on the revolution of ``E``).
    """
    _check_e(e)
    big_e = np.asarray(eccentric_anomaly_rad, dtype=np.float64)
    b = _beta(e)
    return np.asarray(
        big_e + 2.0 * np.arctan2(b * np.sin(big_e), 1.0 - b * np.cos(big_e)), dtype=np.float64
    )


def eccentric_from_true(true_anomaly_rad: ArrayLike, e: float) -> Float64Array:
    """Eccentric anomaly from true anomaly [rad].

    Inverse of :func:`true_from_eccentric`: ``E = nu - 2 atan2(beta sin nu, 1 + beta cos nu)``
    (Broucke & Cefola 1973), continuous across revolutions.
    """
    _check_e(e)
    nu = np.asarray(true_anomaly_rad, dtype=np.float64)
    b = _beta(e)
    return np.asarray(nu - 2.0 * np.arctan2(b * np.sin(nu), 1.0 + b * np.cos(nu)), dtype=np.float64)


def mean_from_eccentric(eccentric_anomaly_rad: ArrayLike, e: float) -> Float64Array:
    """Mean anomaly from eccentric anomaly, Kepler's equation M = E - e sin E [rad]."""
    _check_e(e)
    big_e = np.asarray(eccentric_anomaly_rad, dtype=np.float64)
    return np.asarray(big_e - e * np.sin(big_e), dtype=np.float64)


def mean_from_true(true_anomaly_rad: ArrayLike, e: float) -> Float64Array:
    """Mean anomaly from true anomaly [rad] (via the eccentric anomaly)."""
    return mean_from_eccentric(eccentric_from_true(true_anomaly_rad, e), e)


def solve_kepler(
    mean_anomaly_rad: ArrayLike,
    e: float,
    *,
    tol_rad: float = DEFAULT_KEPLER_TOL_RAD,
    max_iter: int = DEFAULT_KEPLER_MAX_ITER,
) -> KeplerSolution:
    """Solve Kepler's equation ``M = E - e sin E`` for E (elliptic, 0 <= e < 1), vectorised.

    Algorithm: M is reduced to x = |M mod [-pi, pi)| in [0, pi] (exactly, without arithmetic,
    when M is already in [-pi, pi)); there the root is unique and
    bracketed by [x, min(x + e, pi)] because f(E) = E - e sin E - x is strictly increasing
    (f' = 1 - e cos E >= 1 - e > 0) with f(x) <= 0 <= f(min(x + e, pi)). Newton's method is
    started from Danby's (1987, Celest. Mech. 40, 303) guess E0 = x + 0.85 e, and every
    iterate is clipped into the (shrinking) bracket. On [0, pi] f is also convex
    (f'' = e sin E >= 0), so a Newton step from either side lands at or to the right of the
    root, and from the right Newton decreases monotonically to the root; the clipping only
    removes an overshoot past the bracket end (relevant when the root is at x + e, i.e.
    sin E = 1). In exact arithmetic the iteration therefore converges for every M and every
    e < 1. The odd symmetry E(-x) = -E(x) and the removed multiple of 2 pi are restored at
    the end, so E is on the same revolution as M.

    Stopping rule (float64): an element is converged when either the last Newton correction
    satisfies ``|dE| <= tol_rad`` (forward criterion) or the residual is at round-off level,
    ``|f(E)| <= KEPLER_RESIDUAL_ULP_FACTOR * eps * max(x, E)`` (backward criterion; that
    iteration's clipped Newton step is still applied, which moves E only within the round-off
    uncertainty of the root and keeps the residual at round-off). The backward criterion is
    needed because f is evaluated with an absolute error of a few eps*E, so the root is only
    determined to about eps*E / (1 - e cos E); for 1 - e <~ 1e-8 and small M that exceeds
    1e-12 rad and Newton otherwise 2-cycles. When the backward criterion fires, the forward
    error is bounded by about
    ``|f| / (1 - e cos E) <= 4 eps E / ((1 - e) + e E^2 / 2) <= 2 sqrt(2) eps / sqrt(1 - e)``,
    which is below 1e-12 rad for every M when 1 - e > 4e-7; closer to e = 1 (small |M|) the
    float64 root itself is only that well determined. ``max_abs_residual_rad`` reports the
    backward error. ``tol_rad`` is an absolute tolerance: for |E| << tol_rad the relative
    error of E can be large (e.g. M = 1e-31, 1 - e = 1e-8), but its absolute error is
    <= tol_rad.

    Args:
        mean_anomaly_rad: Mean anomaly M, any shape [rad].
        e: Eccentricity, 0 <= e < 1.
        tol_rad: Convergence threshold on |last Newton correction| [rad].
        max_iter: Iteration cap.

    Returns:
        :class:`KeplerSolution` with E of the same shape as M.

    Raises:
        KeplerConvergenceError: if any element has not converged after ``max_iter`` iterations.
        ValueError: on invalid e, non-finite M or invalid tolerances.
    """
    _check_e(e)
    if not (tol_rad > 0.0 and math.isfinite(tol_rad)):
        raise ValueError(f"tol_rad must be finite and > 0, got {tol_rad}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    m_in = np.asarray(mean_anomaly_rad, dtype=np.float64)
    if not np.all(np.isfinite(m_in)):
        raise ValueError("mean_anomaly_rad must be finite")
    shape = m_in.shape
    m = m_in.ravel()

    # Reduce to [-pi, pi). M already in range is used as is: mod(M + pi) - pi would add an
    # absolute error of up to eps*pi to small |M|, which the near-parabolic conditioning
    # 1/(1 - e) amplifies (e.g. 2e-8 rad in E at M = 1e-14, e = 1 - 1e-8).
    in_range = (m >= -math.pi) & (m < math.pi)
    m_red = np.where(in_range, m, np.mod(m + math.pi, TWO_PI) - math.pi)
    sign = np.where(m_red < 0.0, -1.0, 1.0)
    x = np.abs(m_red)  # in [0, pi]
    lo = x.copy()
    hi = np.minimum(x + e, math.pi)
    big_e = np.clip(x + 0.85 * e, lo, hi)

    eps = float(np.finfo(np.float64).eps)
    converged = np.zeros(m.shape, dtype=bool)
    delta = np.full(m.shape, np.inf)
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        f = big_e - e * np.sin(big_e) - x
        fp = 1.0 - e * np.cos(big_e)
        # Backward criterion: residual at the rounding level of evaluating f (see docstring).
        at_roundoff = np.abs(f) <= KEPLER_RESIDUAL_ULP_FACTOR * eps * np.maximum(x, big_e)
        lo = np.where(f < 0.0, big_e, lo)
        hi = np.where(f > 0.0, big_e, hi)
        e_new = np.clip(big_e - f / fp, lo, hi)
        delta = np.abs(e_new - big_e)
        big_e = np.where(converged, big_e, e_new)
        converged |= at_roundoff | (delta <= tol_rad)
        if bool(np.all(converged)):
            break
    if not bool(np.all(converged)):
        bad = np.flatnonzero(~converged)
        worst = int(bad[np.argmax(delta[bad])])
        raise KeplerConvergenceError(
            f"Kepler solver did not converge for {bad.size} of {m.size} mean anomalies after "
            f"{max_iter} iterations (e={e}, tol_rad={tol_rad}); worst M={m[worst]!r} rad with "
            f"last correction {delta[worst]:.3e} rad"
        )
    residual = big_e - e * np.sin(big_e) - x
    max_res = float(np.max(np.abs(residual))) if residual.size else 0.0
    e_full = sign * big_e + (m - m_red)
    return KeplerSolution(
        eccentric_anomaly_rad=e_full.reshape(shape),
        iterations=iterations,
        max_abs_residual_rad=max_res,
    )


def eccentric_from_mean(
    mean_anomaly_rad: ArrayLike,
    e: float,
    *,
    tol_rad: float = DEFAULT_KEPLER_TOL_RAD,
    max_iter: int = DEFAULT_KEPLER_MAX_ITER,
) -> Float64Array:
    """Eccentric anomaly from mean anomaly [rad] (see :func:`solve_kepler`)."""
    return solve_kepler(
        mean_anomaly_rad, e, tol_rad=tol_rad, max_iter=max_iter
    ).eccentric_anomaly_rad


def true_from_mean(
    mean_anomaly_rad: ArrayLike,
    e: float,
    *,
    tol_rad: float = DEFAULT_KEPLER_TOL_RAD,
    max_iter: int = DEFAULT_KEPLER_MAX_ITER,
) -> Float64Array:
    """True anomaly from mean anomaly [rad], on the same revolution as M."""
    return true_from_eccentric(
        eccentric_from_mean(mean_anomaly_rad, e, tol_rad=tol_rad, max_iter=max_iter), e
    )


# ---------------------------------------------------------------------------------------------
# Elements <-> state
# ---------------------------------------------------------------------------------------------


def _plane_basis(i_rad: float, raan_rad: float) -> tuple[Float64Array, Float64Array]:
    """Orbital-plane basis: P (ascending-node direction) and Q = h_hat x P."""
    ci, si = math.cos(i_rad), math.sin(i_rad)
    co, so = math.cos(raan_rad), math.sin(raan_rad)
    p_vec = np.array([co, so, 0.0])
    q_vec = np.array([-so * ci, co * ci, si])
    return p_vec, q_vec


def _rv_from_elements(
    a_m: float,
    e: float,
    i_rad: float,
    raan_rad: float,
    argp_rad: float,
    nu_rad: Float64Array,
    mu_m3_s2: float,
) -> tuple[Float64Array, Float64Array]:
    """Position (N,3) [m] and velocity (N,3) [m/s] for an array of true anomalies (N,).

    r = p/(1 + e cos nu) (cos u P + sin u Q),
    v = sqrt(mu/p) (-(sin u + e sin w) P + (cos u + e cos w) Q),  u = w + nu
    (perifocal formulas rotated by R3(-raan) R1(-i) R3(-w); Vallado 4th ed. Alg. 10).
    """
    p = a_m * (1.0 - e * e)
    p_vec, q_vec = _plane_basis(i_rad, raan_rad)
    u = argp_rad + nu_rad
    r = p / (1.0 + e * np.cos(nu_rad))
    cu, su = np.cos(u), np.sin(u)
    pos = (r * cu)[:, None] * p_vec + (r * su)[:, None] * q_vec
    k = math.sqrt(mu_m3_s2 / p)
    vp = -k * (su + e * math.sin(argp_rad))
    vq = k * (cu + e * math.cos(argp_rad))
    vel = vp[:, None] * p_vec + vq[:, None] * q_vec
    return np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64)


def elements_to_state(elements: OrbitalElements) -> CartesianState:
    """Cartesian state at the elements' epoch, in ``elements.frame`` about ``elements.center``.

    Equations: see :func:`_rv_from_elements` (orbit equation and perifocal velocity,
    Vallado 4th ed. Alg. 10 "COE2RV"). Valid for 0 <= e < 1 including the singular
    conventions documented in the module docstring.
    """
    pos, vel = _rv_from_elements(
        elements.a_m,
        elements.e,
        elements.i_rad,
        elements.raan_rad,
        elements.argp_rad,
        np.array([elements.nu_rad]),
        elements.mu_m3_s2,
    )
    return CartesianState(
        epoch_tdb_s=elements.epoch_tdb_s,
        position_m=pos[0],
        velocity_m_s=vel[0],
        frame=elements.frame,
        origin=elements.center,
    )


def state_to_elements(
    state: CartesianState,
    mu_m3_s2: float,
    *,
    circular_tol: float = DEFAULT_CIRCULAR_TOL,
    equatorial_tol: float = DEFAULT_EQUATORIAL_TOL,
) -> OrbitalElements:
    """Classical elements of the osculating elliptic orbit of ``state`` about its origin.

    Equations (Vallado 4th ed. Alg. 9 "RV2COE", with atan2 forms): ``h = r x v``;
    ``epsilon = v^2/2 - mu/r``, ``a = -mu/(2 epsilon)``; ``e_vec = (v x h)/mu - r/|r|``;
    ``i = atan2(sqrt(h_x^2 + h_y^2), h_z)``; ``raan = atan2(h_x, -h_y)``; with the plane basis
    ``P = (cos raan, sin raan, 0)``, ``Q = h_hat x P``: ``u = atan2(r.Q, r.P)``,
    ``argp = atan2(e.Q, e.P)``, ``nu = u - argp``. Singular cases follow the module conventions.

    Args:
        state: State whose origin is the central body (``SUN`` or ``EARTH``).
        mu_m3_s2: Gravitational parameter of the central body [m^3 s^-2].
        circular_tol: Eccentricity below which the orbit is treated as circular.
        equatorial_tol: sin(i) below which the orbit is treated as equatorial.

    Raises:
        ValueError: if the origin is not a central body (the SSB), the motion is rectilinear
            (h = 0), or the orbit is not elliptic (epsilon >= 0).
    """
    _check_mu(mu_m3_s2)
    if state.origin not in ("SUN", "EARTH"):
        raise ValueError(
            f"two-body elements need a state relative to the central body; got origin "
            f"{state.origin!r} (expected 'SUN' or 'EARTH'). Change the origin with an "
            "ephemeris first."
        )
    r_vec = np.asarray(state.position_m, dtype=np.float64)
    v_vec = np.asarray(state.velocity_m_s, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("position is at the central body (r = 0)")
    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    if h == 0.0:
        raise ValueError("rectilinear motion (h = r x v = 0) has no orbital elements")
    energy = 0.5 * float(np.dot(v_vec, v_vec)) - mu_m3_s2 / r
    if energy >= 0.0:
        raise ValueError(
            f"orbit is not elliptic (specific energy {energy:.6e} J/kg >= 0); only 0 <= e < 1 "
            "is supported"
        )
    a = -mu_m3_s2 / (2.0 * energy)
    e_vec = np.cross(v_vec, h_vec) / mu_m3_s2 - r_vec / r
    e = float(np.linalg.norm(e_vec))

    h_xy = math.hypot(float(h_vec[0]), float(h_vec[1]))
    if h_xy / h < equatorial_tol:
        prograde = float(h_vec[2]) > 0.0
        i = 0.0 if prograde else math.pi
        raan = 0.0
        p_vec = np.array([1.0, 0.0, 0.0])
        q_vec = np.array([0.0, 1.0, 0.0]) if prograde else np.array([0.0, -1.0, 0.0])
    else:
        i = math.atan2(h_xy, float(h_vec[2]))
        raan = _wrap_2pi(math.atan2(float(h_vec[0]), -float(h_vec[1])))
        p_vec = np.array([math.cos(raan), math.sin(raan), 0.0])
        q_vec = np.cross(h_vec / h, p_vec)

    u = math.atan2(float(np.dot(r_vec, q_vec)), float(np.dot(r_vec, p_vec)))
    if e < circular_tol:
        e = 0.0
        argp = 0.0
        nu = _wrap_2pi(u)
    else:
        argp = _wrap_2pi(math.atan2(float(np.dot(e_vec, q_vec)), float(np.dot(e_vec, p_vec))))
        nu = _wrap_2pi(u - argp)
    if e >= 1.0:  # guarded by energy < 0; kept for clarity against round-off at e ~ 1
        raise ValueError(f"computed eccentricity {e} >= 1 for a bound orbit (round-off)")
    return OrbitalElements(
        a_m=a,
        e=e,
        i_rad=i,
        raan_rad=raan,
        argp_rad=argp,
        nu_rad=nu,
        mu_m3_s2=mu_m3_s2,
        epoch_tdb_s=state.epoch_tdb_s,
        frame=state.frame,
        center=state.origin,
    )


# ---------------------------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------------------------


def propagate_kepler(
    initial: OrbitalElements | CartesianState,
    epochs_tdb_s: ArrayLike,
    *,
    mu_m3_s2: float | None = None,
    kepler_tol_rad: float = DEFAULT_KEPLER_TOL_RAD,
    kepler_max_iter: int = DEFAULT_KEPLER_MAX_ITER,
) -> KeplerPropagation:
    """Closed-form two-body (L0) propagation to an array of TDB epochs.

    ``M(t) = M0 + n (t - t0)`` with ``n = sqrt(mu/a^3)``; Kepler's equation is solved for E,
    converted to the true anomaly and to position/velocity with the fixed elements
    (a, e, i, raan, argp). Forward and backward propagation are both supported.

    Accuracy: limited by float64 round-off only (no truncation error). Over a span dt the
    mean anomaly carries an absolute round-off of ~eps |n dt| [rad], i.e. ~1e-13 rad after
    100 revolutions.

    Args:
        initial: Osculating elements, or a state relative to the Sun (converted with
            :func:`state_to_elements`, which then requires ``mu_m3_s2``).
        epochs_tdb_s: 1-D array (or scalar) of TDB epochs [s past J2000].
        mu_m3_s2: Gravitational parameter; required for a state input, and if given with
            elements it must equal ``elements.mu_m3_s2``.
        kepler_tol_rad: Kepler-solver tolerance (see :func:`solve_kepler`).
        kepler_max_iter: Kepler-solver iteration cap.

    Returns:
        :class:`KeplerPropagation` whose trajectory is in the frame of the elements (the
        state's frame for a state input) with origin ``SUN``.
    """
    if isinstance(initial, CartesianState):
        if mu_m3_s2 is None:
            raise ValueError("mu_m3_s2 is required when propagating a CartesianState")
        elements = state_to_elements(initial, mu_m3_s2)
    else:
        elements = initial
        if mu_m3_s2 is not None and mu_m3_s2 != elements.mu_m3_s2:
            raise ValueError(
                f"mu_m3_s2={mu_m3_s2} contradicts elements.mu_m3_s2={elements.mu_m3_s2}"
            )
    t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
    if t.ndim != 1:
        raise ValueError(f"epochs_tdb_s must be a scalar or 1-D, got shape {t.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("epochs_tdb_s must be finite")

    m0 = elements.mean_anomaly_rad
    m = m0 + elements.mean_motion_rad_s * (t - elements.epoch_tdb_s)
    sol = solve_kepler(m, elements.e, tol_rad=kepler_tol_rad, max_iter=kepler_max_iter)
    nu = true_from_eccentric(sol.eccentric_anomaly_rad, elements.e)
    pos, vel = _rv_from_elements(
        elements.a_m,
        elements.e,
        elements.i_rad,
        elements.raan_rad,
        elements.argp_rad,
        nu,
        elements.mu_m3_s2,
    )
    traj = CartesianTrajectory(
        epochs_tdb_s=t,
        position_m=pos,
        velocity_m_s=vel,
        frame=elements.frame,
        origin=elements.center,
    )
    return KeplerPropagation(
        trajectory=traj,
        elements=elements,
        kepler_iterations=sol.iterations,
        max_kepler_residual_rad=sol.max_abs_residual_rad,
    )


# ---------------------------------------------------------------------------------------------
# Configuration hook
# ---------------------------------------------------------------------------------------------


def elements_from_config(
    orbit: OrbitConfig,
    *,
    epoch_tdb_s: float,
    mu_m3_s2: float = GM_SUN_M3_S2,
    true_anomaly_rad: float | None = None,
) -> OrbitalElements:
    """Build :class:`OrbitalElements` from the validated ``orbit`` configuration section.

    The unit conversions (au -> m with the exact IAU 2012 au, deg -> rad) are done by the
    config accessors (``semimajor_axis_m``, ``*_rad``), i.e. at the IO boundary.

    Args:
        orbit: Validated :class:`helioswarm.config.OrbitConfig`.
        epoch_tdb_s: The osculation epoch ``orbit.initial_epoch_utc`` converted to TDB
            seconds past J2000. It is a required argument because the UTC -> TDB conversion
            needs the leap-second kernel and is done by SPICE (``str2et``, M2); this module
            never converts time scales itself.
        mu_m3_s2: Central-body GM [m^3 s^-2]; defaults to GM_sun (DE440), the correct value
            for Sun-centred elements of a massless satellite. Earth-centred elements
            (``orbit.elements_center = 'EARTH'``) must pass GM_earth (the caller,
            :func:`helioswarm.propagate.l0_reference_trajectory`, takes it from the ephemeris
            provider, i.e. ``gm_de440.tpc``).
        true_anomaly_rad: The resolved true anomaly [rad]. Required when
            ``orbit.phase_reference = 'earth_relative'`` (the configured angle is then an
            Earth-relative ecliptic longitude offset that only the ephemeris can turn into a
            true anomaly: :func:`helioswarm.propagate.resolve_orbit_phase`); must be ``None``
            for ``absolute`` phases (``orbit.true_anomaly_deg`` is the true anomaly).

    Raises:
        ValueError: ``earth_relative`` without ``true_anomaly_rad``, or ``absolute`` with it.
    """
    if orbit.phase_reference == "absolute":
        if true_anomaly_rad is not None:
            raise ValueError(
                "orbit.phase_reference='absolute': true_anomaly_deg is the true anomaly; do not "
                "pass a resolved true_anomaly_rad"
            )
        nu = orbit.true_anomaly_rad
    else:
        if true_anomaly_rad is None:
            raise ValueError(
                "orbit.phase_reference='earth_relative' needs the true anomaly resolved from the "
                "ephemeris (helioswarm.propagate.resolve_orbit_phase)"
            )
        nu = float(true_anomaly_rad)
    return OrbitalElements(
        a_m=orbit.semimajor_axis_m,
        e=orbit.eccentricity,
        i_rad=orbit.inclination_rad,
        raan_rad=orbit.raan_rad,
        argp_rad=orbit.arg_periapsis_rad,
        nu_rad=nu,
        mu_m3_s2=mu_m3_s2,
        epoch_tdb_s=epoch_tdb_s,
        frame=orbit.elements_frame,
        center=orbit.elements_center,
    )


__all__ = [
    "DEFAULT_CIRCULAR_TOL",
    "DEFAULT_EQUATORIAL_TOL",
    "DEFAULT_KEPLER_MAX_ITER",
    "DEFAULT_KEPLER_TOL_RAD",
    "KEPLER_RESIDUAL_ULP_FACTOR",
    "KeplerConvergenceError",
    "KeplerPropagation",
    "KeplerSolution",
    "OrbitalElements",
    "eccentric_from_mean",
    "eccentric_from_true",
    "eccentricity_vector",
    "elements_from_config",
    "elements_to_state",
    "mean_from_eccentric",
    "mean_from_true",
    "mean_motion_rad_s",
    "orbital_period_s",
    "propagate_kepler",
    "solve_kepler",
    "specific_angular_momentum_m2_s",
    "specific_energy_J_kg",
    "state_to_elements",
    "true_from_eccentric",
    "true_from_mean",
]
