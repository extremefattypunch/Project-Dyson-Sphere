"""L1 orbit propagation: a test particle in prescribed-ephemeris gravity (+ optional SRP).

Level L1 of PDF §6.2: the satellite is a massless test particle. The Sun and the selected
bodies move on their prescribed ephemeris trajectories (they do not respond to the satellite;
they remain tied to the authoritative ephemeris). Inputs and outputs are J2000 states with
origin SSB.

**Integrated equations (heliocentric formulation, the default).** PDF §6.2 allows a
heliocentric non-inertial frame provided the correct indirect acceleration is included. With
``r = r_sat - r_sun`` and ``R_j = r_j - r_sun`` (J2000 axes)::

    d^2r/dt^2 = -(mu_sun - k_srp) r/|r|^3
                + sum_{j != Sun, j in model} mu_j [ (R_j - r)/|R_j - r|^3 - R_j/|R_j|^3 ]

The second bracket term is the indirect term: the Sun's Newtonian acceleration due to the
*same* bodies that act on the satellite, so the direct and indirect terms are always consistent
for any body subset (the direct and indirect parts of the disturbing function; Murray &
Dermott, *Solar System Dynamics*, 1999, ch. 6). It is the barycentric equation of PDF §6.2
minus the Sun's acceleration under the modelled bodies. The
J2000/SSB output is ``r + r_sun(t)`` with the Sun's state taken directly from the provider.
Consequences: the Sun-only model is exactly heliocentric two-body motion (equal to L0 with the
real, accelerated DE440 Sun; tested), and dropping a body removes only that body's physical
(tidal) effect.

**Barycentric formulation (comparison path).** ``formulation="barycentric"`` integrates the
PDF §6.2 equation literally in J2000/SSB with the Sun on its DE440 trajectory::

    d^2r_sat/dt^2 = sum_j mu_j (r_j(t) - r_sat)/|r_j(t) - r_sat|^3 + a_srp

It is consistent only when the modelled bodies explain the Sun's prescribed acceleration;
the DE440 Sun is accelerated by every planet, so the barycentric form is accepted only with a
complete planetary set (every planetary-system barycentre, Earth-Moon split or not). Even then
it differs from the heliocentric form by the Sun's unmodelled acceleration (asteroids, Pluto,
relativity in DE440), which acts as an indirect term without its direct counterpart (measured
for the MVP orbit in tests/unit/test_l1_propagation.py). Mixing the two formulations is never
done.

Integrator: ``scipy.integrate.solve_ivp`` with ``method="DOP853"`` (explicit Runge-Kutta 8(5,3)
of Dormand & Prince; Hairer, Norsett & Wanner, *Solving ODEs I*, §II.10), with relative
tolerance ``rtol``, per-component absolute tolerances (position [m], velocity [m/s]) and
``max_step``. Output epochs are evaluated with the integrator's dense output (``t_eval``), so
the step sequence does not depend on the output grid. Epochs before the initial epoch are
reached by a separate backward integration from the initial state.

Body positions inside the right-hand side are taken from a cubic-Hermite cache of the ephemeris
(:class:`helioswarm.ephemeris.HermiteEphemerisCache`, spacing ``ephemeris_cache_step_s``); its
interpolation error is measured at every interval midpoint at build time and reported in
:class:`L1Diagnostics` (< 1 m for every DE440 body at the default 3600 s, tested). Direct SPICE
calls inside the right-hand side were measured at ~5.4 us per body per call; with ~1.3e6
right-hand-side evaluations per simulated year at ``max_step = 300 s`` and 10 bodies, that
alone would add ~70 s per year, hence the cache.

Not modelled: relativistic (1PN) corrections (for a circular orbit at 1 au they act like
``mu -> mu (1 - 3 GM/(r c^2))``, i.e. ~5.6e4 m/yr along-track; docs/model_assumptions.md),
asteroids, Pluto, planetary oblateness, the satellite's influence on anything.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from helioswarm.config import (
    EPHEMERIS_CACHE_STEP_MAX_S,
    INTEGRATOR_RTOL_MAX,
    INTEGRATOR_RTOL_MIN,
    ConstantsConfig,
    OrbitConfig,
    RunConfig,
    TimeConfig,
)
from helioswarm.ephemeris import (
    SSB,
    BodyName,
    EphemerisProvider,
    HermiteEphemerisCache,
    shift_state_origin,
    shift_trajectory_origin,
)
from helioswarm.frames import (
    CartesianState,
    CartesianTrajectory,
    rotate_vectors,
    transform_state,
    transform_trajectory,
)
from helioswarm.ground import EarthOrientation
from helioswarm.kernels import LoadedKernels
from helioswarm.orbit import (
    elements_from_config,
    elements_to_state,
    propagate_kepler,
)
from helioswarm.srp import SrpModel, srp_acceleration_m_s2
from helioswarm.timescale import TimeGrid, time_grid_from_config, utc_to_tdb_s

Float64Array = NDArray[np.float64]

L1_MODEL: Final[str] = "prescribed_ephemeris_perturbed_test_particle"
"""Configuration name of the L1 model (``orbit.model``)."""

Formulation = Literal["heliocentric", "barycentric"]
"""Coordinates in which the L1 equations are integrated (see the module docstring)."""

PLANETARY_SYSTEMS: Final[tuple[tuple[BodyName, ...], ...]] = (
    ("MERCURY BARYCENTER",),
    ("VENUS BARYCENTER",),
    ("EARTH BARYCENTER", "EARTH", "MOON"),
    ("MARS BARYCENTER",),
    ("JUPITER BARYCENTER",),
    ("SATURN BARYCENTER",),
    ("URANUS BARYCENTER",),
    ("NEPTUNE BARYCENTER",),
)
"""The eight planetary systems of DE440 and the bodies that represent each (the Earth-Moon
system by its barycentre, or by EARTH and MOON together)."""


Attitude = Literal["sun_tracking", "fixed_normal"]
"""Collector attitude law of a :class:`SatelliteState` (``collector.incidence_mode``)."""

NORMAL_UNIT_TOLERANCE: Final[float] = 1.0e-6
"""Accepted ``| |n| - 1 |`` of a collector normal (the configuration schema's tolerance)."""


class PropagationError(RuntimeError):
    """The integrator failed (``solve_ivp`` reported no success)."""


# ---------------------------------------------------------------------------------------------
# Model and settings containers
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForceModel:
    """Forces acting on the satellite (the PDF p.4 "force-model flags").

    Attributes:
        bodies: Gravitating bodies (SPICE names); must contain ``SUN``. ``("SUN",)`` is the
            Sun-only model (``include_planetary_perturbations = false``).
        srp: Radial SRP model, or ``None`` for no SRP.
    """

    bodies: tuple[BodyName, ...]
    srp: SrpModel | None = None

    def __post_init__(self) -> None:
        if "SUN" not in self.bodies:
            raise ValueError("ForceModel.bodies must contain 'SUN'")
        if len(set(self.bodies)) != len(self.bodies):
            raise ValueError(f"ForceModel.bodies has duplicates: {self.bodies}")
        if "EARTH BARYCENTER" in self.bodies and ("EARTH" in self.bodies or "MOON" in self.bodies):
            raise ValueError("EARTH BARYCENTER double-counts EARTH/MOON")

    @property
    def include_planetary_perturbations(self) -> bool:
        """True if any body other than the Sun gravitates."""
        return len(self.bodies) > 1

    @property
    def has_complete_planetary_set(self) -> bool:
        """True if every planetary system of DE440 is represented (:data:`PLANETARY_SYSTEMS`;
        the Earth-Moon system needs EARTH BARYCENTER, or EARTH and MOON together)."""
        names = set(self.bodies)
        for system in PLANETARY_SYSTEMS:
            if len(system) == 1:
                ok = system[0] in names
            else:
                ok = system[0] in names or {system[1], system[2]} <= names
            if not ok:
                return False
        return True

    @property
    def include_solar_radiation_pressure(self) -> bool:
        """True if SRP is on."""
        return self.srp is not None


@dataclass(frozen=True)
class IntegratorSettings:
    """Numerical settings of the L1 integrator.

    Attributes:
        rtol: Relative tolerance (``solve_ivp`` rtol) in
            [:data:`~helioswarm.config.INTEGRATOR_RTOL_MIN`,
            :data:`~helioswarm.config.INTEGRATOR_RTOL_MAX`] (the same bounds as the
            configuration; the floor stays above scipy's 100 eps).
        atol_position_m: Absolute tolerance of each position component [m].
        atol_velocity_m_s: Absolute tolerance of each velocity component [m s^-1].
        max_step_s: Maximum step [s] (``solve_ivp`` max_step; ``inf`` = unlimited).
        ephemeris_cache_step_s: Node spacing of the Hermite ephemeris cache [s], at most
            :data:`~helioswarm.config.EPHEMERIS_CACHE_STEP_MAX_S` (the validated range).
        method: Always ``DOP853``.
    """

    rtol: float = 1.0e-12
    atol_position_m: float = 1.0e-3
    atol_velocity_m_s: float = 1.0e-9
    max_step_s: float = math.inf
    ephemeris_cache_step_s: float = 3600.0
    method: Literal["DOP853"] = "DOP853"

    def __post_init__(self) -> None:
        if not INTEGRATOR_RTOL_MIN <= self.rtol <= INTEGRATOR_RTOL_MAX:
            raise ValueError(
                f"rtol must be in [{INTEGRATOR_RTOL_MIN}, {INTEGRATOR_RTOL_MAX}], got {self.rtol}"
            )
        for name in ("atol_position_m", "atol_velocity_m_s", "ephemeris_cache_step_s"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {v}")
        if self.ephemeris_cache_step_s > EPHEMERIS_CACHE_STEP_MAX_S:
            raise ValueError(
                f"ephemeris_cache_step_s must be <= {EPHEMERIS_CACHE_STEP_MAX_S} s (validated "
                f"range), got {self.ephemeris_cache_step_s}"
            )
        if not self.max_step_s > 0.0:
            raise ValueError(f"max_step_s must be > 0, got {self.max_step_s}")
        if self.method != "DOP853":
            raise ValueError("only method='DOP853' is supported")


@dataclass(frozen=True)
class SatelliteState:
    """Satellite state with its force-model flags (PDF p.4 ``SatelliteState``).

    Attributes:
        state: Position/velocity/epoch in J2000 with origin SSB (enforced).
        force_model: Forces acting on the satellite.
        mass_kg: Satellite mass [kg], or ``None``: the configuration does not specify a mass
            and the dynamics need only ``A/m`` (SRP).
        attitude: Collector attitude law (``collector.incidence_mode``): ``sun_tracking`` (the
            collector front-face normal points at the Sun) or ``fixed_normal`` (inertially fixed
            normal). The SRP force model is radial for both (PDF §6.3 "simple radial SRP";
            attitude-dependent SRP is the later L3 model).
        collector_normal_unit: Unit normal of the collector front face at ``epoch`` in J2000
            (PDF p.4 "attitude/normal vector"), or ``None`` when not supplied (direct
            construction in tests). :func:`satellite_initial_state` always sets it.
    """

    state: CartesianState
    force_model: ForceModel
    mass_kg: float | None = None
    attitude: Attitude = "sun_tracking"
    collector_normal_unit: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.state.frame != "J2000" or self.state.origin != SSB:
            raise ValueError("SatelliteState.state must be in J2000 with origin SSB")
        if self.mass_kg is not None and not (math.isfinite(self.mass_kg) and self.mass_kg > 0):
            raise ValueError("mass_kg must be finite and > 0")
        if self.attitude not in ("sun_tracking", "fixed_normal"):
            raise ValueError(f"unknown attitude {self.attitude!r}")
        n = self.collector_normal_unit
        if n is not None:
            norm = math.sqrt(sum(c * c for c in n))
            if len(n) != 3 or not math.isfinite(norm) or abs(norm - 1.0) > NORMAL_UNIT_TOLERANCE:
                raise ValueError(f"collector_normal_unit must be a unit 3-vector, got {n}")

    @property
    def epoch_tdb_s(self) -> float:
        """Epoch [TDB s past J2000]."""
        return self.state.epoch_tdb_s

    @property
    def position_m(self) -> Float64Array:
        """J2000/SSB position [m]."""
        return self.state.position_m

    @property
    def velocity_m_s(self) -> Float64Array:
        """J2000/SSB velocity [m s^-1]."""
        return self.state.velocity_m_s


@dataclass(frozen=True)
class IntegrationSegment:
    """Diagnostics of one ``solve_ivp`` call.

    Attributes:
        direction: ``forward`` or ``backward`` from the initial epoch.
        t_start_tdb_s: Start (the initial epoch) [TDB s].
        t_end_tdb_s: End of the integration interval [TDB s].
        n_output: Number of output epochs produced by this segment.
        nfev: Right-hand-side evaluations.
        success: ``solve_ivp`` success flag.
        message: ``solve_ivp`` termination message.
    """

    direction: Literal["forward", "backward"]
    t_start_tdb_s: float
    t_end_tdb_s: float
    n_output: int
    nfev: int
    success: bool
    message: str


@dataclass(frozen=True)
class L1Diagnostics:
    """Diagnostics of an L1 propagation.

    Attributes:
        formulation: Coordinates the equations were integrated in (module docstring).
        segments: One entry per integration direction actually run.
        nfev: Total right-hand-side evaluations.
        success: All segments succeeded.
        gm_m3_s2: ``((body, GM), ...)`` used [m^3 s^-2].
        sun_two_body_energy_rel_drift: ``max |E(t) - E(t0)| / |E(t0)|`` over the outputs of the
            heliocentric two-body energy ``E = |v - v_sun|^2/2 - (GM_sun - k_srp)/|r - r_sun|``.
            In the heliocentric formulation it is an integral of motion of the Sun-only model
            (whatever the Sun's SSB motion), so it then measures the integration error; with
            planets it measures the departure from two-body motion.
        cache_step_s: Ephemeris-cache node spacing [s].
        cache_span_tdb_s: (first node, last node) of the cache [TDB s].
        cache_midpoint_error_m: ``((body, err), ...)``: max over all cache intervals of
            ``|cache(t_mid) - provider(t_mid)|`` at the interval midpoints [m], where the cubic
            Hermite error kernel ``s^2 (1 - s)^2`` peaks. It is the cache's contribution to the
            force-model error (computed at build time, one extra provider call per node).
    """

    formulation: Formulation
    segments: tuple[IntegrationSegment, ...]
    nfev: int
    success: bool
    gm_m3_s2: tuple[tuple[str, float], ...]
    sun_two_body_energy_rel_drift: float
    cache_step_s: float
    cache_span_tdb_s: tuple[float, float]
    cache_midpoint_error_m: tuple[tuple[str, float], ...]

    @property
    def cache_max_midpoint_error_m(self) -> float:
        """Largest cache midpoint error over all bodies [m]."""
        return max(err for _, err in self.cache_midpoint_error_m)


@dataclass(frozen=True)
class L1Propagation:
    """Result of :func:`propagate_l1`.

    Attributes:
        trajectory: Satellite states at the requested epochs (J2000, SSB).
        initial: The initial satellite state.
        settings: Integrator settings used.
        diagnostics: Integrator and model diagnostics.
        source_kernel_ids: Ephemeris data the bodies came from.
    """

    trajectory: CartesianTrajectory
    initial: SatelliteState
    settings: IntegratorSettings
    diagnostics: L1Diagnostics
    source_kernel_ids: tuple[str, ...]


# ---------------------------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------------------------


def point_mass_acceleration_m_s2(
    position_m: ArrayLike, body_positions_m: ArrayLike, gm_m3_s2: ArrayLike
) -> Float64Array:
    """Sum of point-mass accelerations ``sum_j mu_j (r_j - r)/|r_j - r|^3`` [m s^-2] (PDF §6.2).

    Args:
        position_m: Test-particle position, shape (3,) [m].
        body_positions_m: Body positions in the same frame/origin, shape (n, 3) [m].
        gm_m3_s2: Body GMs, shape (n,) [m^3 s^-2].
    """
    d = np.asarray(body_positions_m, dtype=np.float64) - np.asarray(position_m, dtype=np.float64)
    r2 = np.einsum("ij,ij->i", d, d)
    return np.asarray((np.asarray(gm_m3_s2, dtype=np.float64) / (r2 * np.sqrt(r2))) @ d)


def l1_acceleration_m_s2(
    position_m: ArrayLike,
    body_positions_m: ArrayLike,
    gm_m3_s2: ArrayLike,
    *,
    sun_index: int,
    srp: SrpModel | None,
) -> Float64Array:
    """Barycentric L1 acceleration exactly as written in PDF §6.2: point-mass gravity of every
    body plus radial SRP from the Sun at ``body_positions_m[sun_index]`` [m s^-2].

    This is the reference form of the ``formulation="barycentric"`` right-hand side (which uses
    the algebraically identical folded form of :func:`effective_gm_m3_s2`).
    """
    bodies = np.asarray(body_positions_m, dtype=np.float64)
    acc = point_mass_acceleration_m_s2(position_m, bodies, gm_m3_s2)
    if srp is not None:
        acc = acc + srp_acceleration_m_s2(position_m, bodies[sun_index], srp.strength_m3_s2)
    return acc


def heliocentric_l1_acceleration_m_s2(
    rel_position_m: ArrayLike,
    helio_body_positions_m: ArrayLike,
    gm_bodies_m3_s2: ArrayLike,
    *,
    gm_sun_m3_s2: float,
    srp: SrpModel | None,
) -> Float64Array:
    """Heliocentric L1 acceleration with consistent direct and indirect terms [m s^-2].

    ``a = -(mu_sun - k) r/|r|^3 + sum_j mu_j [(R_j - r)/|R_j - r|^3 - R_j/|R_j|^3]`` (PDF §6.2
    heliocentric frame "with the correct indirect acceleration"; Murray & Dermott 1999, ch. 6).
    Equals :func:`l1_acceleration_m_s2` minus the Sun's Newtonian acceleration under the same
    bodies (tested).

    Args:
        rel_position_m: Satellite position relative to the Sun, shape (3,) [m].
        helio_body_positions_m: Non-Sun body positions relative to the Sun, shape (n, 3) [m]
            (n may be 0).
        gm_bodies_m3_s2: Their GMs, shape (n,) [m^3 s^-2].
        gm_sun_m3_s2: GM of the Sun [m^3 s^-2].
        srp: Radial SRP model (folded as ``mu_sun -> mu_sun - k``), or ``None``.
    """
    r = np.asarray(rel_position_m, dtype=np.float64)
    big_r = np.asarray(helio_body_positions_m, dtype=np.float64).reshape(-1, 3)
    gm = np.asarray(gm_bodies_m3_s2, dtype=np.float64).reshape(-1)
    mu_sun = gm_sun_m3_s2 - (srp.strength_m3_s2 if srp is not None else 0.0)
    acc = -mu_sun * r / float(np.dot(r, r)) ** 1.5
    if gm.size:
        rr2 = np.einsum("ij,ij->i", big_r, big_r)
        acc = acc + point_mass_acceleration_m_s2(r, big_r, gm) - (gm / (rr2 * np.sqrt(rr2))) @ big_r
    return np.asarray(acc, dtype=np.float64)


def effective_gm_m3_s2(
    gm_m3_s2: ArrayLike, *, sun_index: int, srp: SrpModel | None
) -> Float64Array:
    """GM vector with radial SRP folded into the Sun's entry: ``mu_sun -> mu_sun - k_srp``.

    ``k (r - r_sun)/|r - r_sun|^3`` (SRP) equals ``-k (r_sun - r)/|r_sun - r|^3``, so gravity + SRP
    is the point-mass sum with ``mu_sun - k``; :func:`l1_acceleration_m_s2` with these GMs and
    no SRP is identical to it with the true GMs and SRP (tested).
    """
    gm_eff = np.array(gm_m3_s2, dtype=np.float64)
    if srp is not None:
        gm_eff[sun_index] -= srp.strength_m3_s2
    return gm_eff


def _make_barycentric_rhs(
    cache: HermiteEphemerisCache, gm_eff: Float64Array
) -> Callable[[float, Float64Array], Float64Array]:
    """``(v, sum_j mu_eff_j (r_j(t) - r)/|r_j(t) - r|^3)`` in J2000/SSB (inlined for speed)."""

    def rhs(t: float, y: Float64Array) -> Float64Array:
        d = cache.positions_at(t) - y[:3]
        r2 = np.einsum("ij,ij->i", d, d)
        out = np.empty(6, dtype=np.float64)
        out[:3] = y[3:]
        out[3:] = (gm_eff / (r2 * np.sqrt(r2))) @ d
        return out

    return rhs


def _make_heliocentric_rhs(
    cache: HermiteEphemerisCache, mu_sun_eff: float, gm_others: Float64Array
) -> Callable[[float, Float64Array], Float64Array]:
    """Right-hand side of :func:`heliocentric_l1_acceleration_m_s2` (inlined for speed).

    The cache holds the Sun first, then the other bodies (SSB positions); the Hermite
    interpolant is linear in its node data, so cached differences equal interpolated
    heliocentric positions.
    """
    has_others = gm_others.size > 0

    def rhs(t: float, y: Float64Array) -> Float64Array:
        r = y[:3]
        out = np.empty(6, dtype=np.float64)
        out[:3] = y[3:]
        acc = (-mu_sun_eff / (r @ r) ** 1.5) * r
        if has_others:
            p = cache.positions_at(t)
            big_r = p[1:] - p[0]
            d = big_r - r
            d2 = np.einsum("ij,ij->i", d, d)
            rr2 = np.einsum("ij,ij->i", big_r, big_r)
            acc = acc + (gm_others / (d2 * np.sqrt(d2))) @ d
            acc = acc - (gm_others / (rr2 * np.sqrt(rr2))) @ big_r
        out[3:] = acc
        return out

    return rhs


def propagate_l1(
    initial: SatelliteState,
    epochs_tdb_s: ArrayLike,
    *,
    ephemeris: EphemerisProvider,
    settings: IntegratorSettings,
    formulation: Formulation = "heliocentric",
) -> L1Propagation:
    """Integrate the L1 test-particle equations from ``initial`` to each requested epoch.

    Args:
        initial: Initial state (J2000/SSB) with force model.
        epochs_tdb_s: Strictly increasing 1-D array of output epochs [TDB s]; may lie on both
            sides of the initial epoch (integrated forward and backward separately).
        ephemeris: Provider of body states and GMs (the same provider that built ``initial``).
        settings: Integrator settings.
        formulation: ``heliocentric`` (default; consistent for any body subset) or
            ``barycentric`` (comparison path; requires a complete planetary set, see the module
            docstring).

    Returns:
        :class:`L1Propagation` with the trajectory in J2000/SSB.

    Raises:
        PropagationError: ``solve_ivp`` failed.
        EphemerisCoverageError: the span is outside the ephemeris coverage.
        ValueError: invalid epochs, or ``barycentric`` with an incomplete planetary set.
    """
    t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("epochs_tdb_s must be a non-empty finite 1-D array")
    if t.size > 1 and not np.all(np.diff(t) > 0.0):
        raise ValueError("epochs_tdb_s must be strictly increasing")
    fm = initial.force_model
    if formulation == "barycentric" and not fm.has_complete_planetary_set:
        raise ValueError(
            "formulation='barycentric' prescribes the DE440 Sun, whose acceleration comes from "
            "every planet; with the incomplete body set "
            f"{fm.bodies} the satellite would feel the indirect (Sun-reflex) term of the "
            "omitted bodies without their direct term (PDF §6.2 forbids mixing). Use "
            "formulation='heliocentric'."
        )
    if formulation not in ("heliocentric", "barycentric"):
        raise ValueError(f"unknown formulation {formulation!r}")
    t0 = initial.epoch_tdb_s
    others = tuple(b for b in fm.bodies if b != "SUN")
    order: tuple[BodyName, ...] = ("SUN", *others)
    gm_sun = ephemeris.gm_m3_s2("SUN")
    gm_others = np.array([ephemeris.gm_m3_s2(b) for b in others], dtype=np.float64)
    srp_k = fm.srp.strength_m3_s2 if fm.srp is not None else 0.0
    lo = min(float(t[0]), t0)
    hi = max(float(t[-1]), t0)
    cache = HermiteEphemerisCache.build(ephemeris, order, lo, hi, settings.ephemeris_cache_step_s)
    midpoint_err = cache.max_midpoint_errors_m(ephemeris)
    sun0 = ephemeris.state("SUN", t0)
    if formulation == "heliocentric":
        rhs = _make_heliocentric_rhs(cache, gm_sun - srp_k, gm_others)
        y0 = np.concatenate(
            [initial.position_m - sun0.position_m, initial.velocity_m_s - sun0.velocity_m_s]
        )
    else:
        gms = np.concatenate([[gm_sun], gm_others])
        rhs = _make_barycentric_rhs(cache, effective_gm_m3_s2(gms, sun_index=0, srp=fm.srp))
        y0 = np.concatenate([initial.position_m, initial.velocity_m_s])
    atol = np.array(
        [settings.atol_position_m] * 3 + [settings.atol_velocity_m_s] * 3, dtype=np.float64
    )
    out = np.empty((t.size, 6), dtype=np.float64)
    segments: list[IntegrationSegment] = []

    at_start = t == t0
    out[at_start] = y0
    for direction in ("forward", "backward"):
        mask = t > t0 if direction == "forward" else t < t0
        if not np.any(mask):
            continue
        t_eval = t[mask] if direction == "forward" else t[mask][::-1]
        t_end = float(t_eval[-1])
        sol = solve_ivp(
            rhs,
            (t0, t_end),
            y0,
            method=settings.method,
            t_eval=t_eval,
            rtol=settings.rtol,
            atol=atol,
            max_step=settings.max_step_s,
        )
        segments.append(
            IntegrationSegment(
                direction=direction,
                t_start_tdb_s=t0,
                t_end_tdb_s=t_end,
                n_output=int(sol.t.size),
                nfev=int(sol.nfev),
                success=bool(sol.success),
                message=str(sol.message),
            )
        )
        if not sol.success or sol.t.size != t_eval.size:
            raise PropagationError(
                f"L1 {direction} integration from {t0} to {t_end} TDB s failed after "
                f"{sol.nfev} evaluations: {sol.message}"
            )
        y = sol.y.T if direction == "forward" else sol.y.T[::-1]
        out[mask] = y

    sun = ephemeris.states("SUN", t)
    if formulation == "heliocentric":
        helio = out
        ssb_pos = out[:, :3] + sun.position_m
        ssb_vel = out[:, 3:] + sun.velocity_m_s
        # The initial epoch is returned bit-exactly as given (no round trip through r - r_sun).
        ssb_pos[at_start] = initial.position_m
        ssb_vel[at_start] = initial.velocity_m_s
    else:
        helio = np.concatenate([out[:, :3] - sun.position_m, out[:, 3:] - sun.velocity_m_s], axis=1)
        ssb_pos, ssb_vel = out[:, :3], out[:, 3:]
    traj = CartesianTrajectory(
        epochs_tdb_s=t, position_m=ssb_pos, velocity_m_s=ssb_vel, frame="J2000", origin=SSB
    )
    mu_eff = gm_sun - srp_k
    e0 = float(
        _two_body_energy(
            initial.position_m - sun0.position_m, initial.velocity_m_s - sun0.velocity_m_s, mu_eff
        )
    )
    e = _two_body_energy(helio[:, :3], helio[:, 3:], mu_eff)
    diagnostics = L1Diagnostics(
        formulation=formulation,
        segments=tuple(segments),
        nfev=sum(s.nfev for s in segments),
        success=all(s.success for s in segments),
        gm_m3_s2=(
            ("SUN", gm_sun),
            *((b, float(g)) for b, g in zip(others, gm_others, strict=True)),
        ),
        sun_two_body_energy_rel_drift=float(np.max(np.abs(e - e0)) / abs(e0)),
        cache_step_s=cache.step_s,
        cache_span_tdb_s=(cache.t0_tdb_s, cache.end_tdb_s),
        cache_midpoint_error_m=tuple(
            (b, float(err)) for b, err in zip(order, midpoint_err, strict=True)
        ),
    )
    return L1Propagation(
        trajectory=traj,
        initial=initial,
        settings=settings,
        diagnostics=diagnostics,
        source_kernel_ids=ephemeris.source_kernel_ids,
    )


# ---------------------------------------------------------------------------------------------
# Several independent test particles in one solve (M7 constellations)
# ---------------------------------------------------------------------------------------------


def max_stack_size(rtol: float) -> int:
    """Largest number of satellites one stacked solve may hold at ``rtol`` (>= 1).

    :func:`propagate_l1_stacked` divides every tolerance by ``sqrt(n)`` and the result must stay
    >= :data:`~helioswarm.config.INTEGRATOR_RTOL_MIN`: ``n <= (rtol / rtol_min)^2``.
    """
    return max(1, math.floor((rtol / INTEGRATOR_RTOL_MIN) ** 2 * (1.0 + 1e-12)))


@dataclass(frozen=True)
class StackedL1Propagation:
    """Result of :func:`propagate_l1_stacked`.

    Attributes:
        propagations: One :class:`L1Propagation` per input satellite, in input order. Each
            carries its own initial state, trajectory and two-body energy drift; its
            ``segments`` and ``nfev`` are those of the stack it was integrated in.
        stack_index: Stack (solve) of each satellite.
        stack_sizes: Number of satellites per stack.
        tolerance_scale: ``1/sqrt(stack size)`` applied to rtol and atol of each stack.
    """

    propagations: tuple[L1Propagation, ...]
    stack_index: tuple[int, ...]
    stack_sizes: tuple[int, ...]
    tolerance_scale: tuple[float, ...]


def _make_stacked_heliocentric_rhs(
    cache: HermiteEphemerisCache, mu_sun_eff: float, gm_others: Float64Array, n_sat: int
) -> Callable[[float, Float64Array], Float64Array]:
    """Right-hand side of :func:`heliocentric_l1_acceleration_m_s2` for ``n_sat`` independent
    test particles stacked as ``y = (r_1, v_1, ..., r_n, v_n)`` (shape (6 n,)).

    The particles do not interact (massless); every one sees the same bodies, so the cached
    body positions are evaluated once per call."""
    has_others = gm_others.size > 0

    def rhs(t: float, y: Float64Array) -> Float64Array:
        yy = y.reshape(n_sat, 6)
        r = yy[:, :3]
        out = np.empty((n_sat, 6), dtype=np.float64)
        out[:, :3] = yy[:, 3:]
        r2 = np.einsum("ij,ij->i", r, r)
        acc = (-mu_sun_eff / (r2 * np.sqrt(r2)))[:, None] * r
        if has_others:
            p = cache.positions_at(t)
            big_r = p[1:] - p[0]
            d = big_r[None, :, :] - r[:, None, :]
            d2 = np.einsum("nmk,nmk->nm", d, d)
            acc = acc + np.einsum("nm,nmk->nk", gm_others / (d2 * np.sqrt(d2)), d)
            rr2 = np.einsum("ij,ij->i", big_r, big_r)
            acc = acc - ((gm_others / (rr2 * np.sqrt(rr2))) @ big_r)[None, :]
        out[:, 3:] = acc
        return out.reshape(-1)

    return rhs


def propagate_l1_stacked(
    initials: tuple[SatelliteState, ...] | list[SatelliteState],
    epochs_tdb_s: ArrayLike,
    *,
    ephemeris: EphemerisProvider,
    settings: IntegratorSettings,
    max_satellites_per_stack: int | None = None,
) -> StackedL1Propagation:
    """L1 propagation of several independent satellites with stacked state vectors (M7).

    The satellites are massless test particles in the same prescribed-ephemeris field
    (identical force model and initial epoch required), so their equations decouple and can
    be integrated as one ``6 n`` system (heliocentric formulation, as :func:`propagate_l1`)
    sharing one ephemeris cache and one step sequence (PDF §14.3 item 2).

    Error control (a heuristic, validated empirically, not a guarantee): ``solve_ivp`` controls
    one error norm over *all* ``6 n`` components. Every tolerance is divided by ``sqrt(n)`` so
    that a norm that sums squared scaled errors over the components does not loosen as
    satellites are added. For an RMS norm (e.g. RK45) this makes each accepted stacked step meet
    every satellite's solo criterion. The integrator used here, DOP853, does **not** have that
    property: its scipy error estimate is ``|h| E5 / sqrt((E5 + 0.01 E3) * 6 n)`` with
    ``E5 = ||err5/scale||^2`` and ``E3 = ||err3/scale||^2`` summed over all components, which
    is not separable by satellite, so a stacked norm <= 1 does not imply each satellite's own
    norm <= 1 (a satellite with a large 3rd-order error estimate can lower the norm seen by
    another). What holds is empirical agreement: stacked and solo trajectories agree to well
    below the solo integration error, measured for heterogeneous rings over one year
    (``scripts/validate_stacked_l1.py``, docs/milestones/M7.md) and tested over shorter spans.
    Stacks hold at most :func:`max_stack_size` satellites so that ``rtol/sqrt(n)`` stays above
    the validated floor; larger sets are split into several stacks. A stack of one satellite is
    integrated by :func:`propagate_l1` itself (identical to a solo run).

    Args:
        initials: Initial states (same epoch, same force model).
        epochs_tdb_s: Output epochs (as :func:`propagate_l1`).
        ephemeris: Provider of body states and GMs.
        settings: Solo integrator settings (scaled per stack as above).
        max_satellites_per_stack: Optional smaller cap on the stack size.

    Returns:
        :class:`StackedL1Propagation`.

    Raises:
        ValueError: empty input, different epochs or force models, invalid epochs.
        PropagationError: ``solve_ivp`` failed.
    """
    inits = tuple(initials)
    if not inits:
        raise ValueError("need at least one satellite")
    t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("epochs_tdb_s must be a non-empty finite 1-D array")
    if t.size > 1 and not np.all(np.diff(t) > 0.0):
        raise ValueError("epochs_tdb_s must be strictly increasing")
    t0 = inits[0].epoch_tdb_s
    fm = inits[0].force_model
    for s in inits[1:]:
        if s.epoch_tdb_s != t0:
            raise ValueError("stacked satellites must share the initial epoch")
        if s.force_model != fm:
            raise ValueError("stacked satellites must share the force model")
    cap = max_stack_size(settings.rtol)
    if max_satellites_per_stack is not None:
        if max_satellites_per_stack < 1:
            raise ValueError("max_satellites_per_stack must be >= 1")
        cap = min(cap, max_satellites_per_stack)
    others = tuple(b for b in fm.bodies if b != "SUN")
    order: tuple[BodyName, ...] = ("SUN", *others)
    gm_sun = ephemeris.gm_m3_s2("SUN")
    gm_others = np.array([ephemeris.gm_m3_s2(b) for b in others], dtype=np.float64)
    srp_k = fm.srp.strength_m3_s2 if fm.srp is not None else 0.0
    mu_eff = gm_sun - srp_k
    lo = min(float(t[0]), t0)
    hi = max(float(t[-1]), t0)
    cache = HermiteEphemerisCache.build(ephemeris, order, lo, hi, settings.ephemeris_cache_step_s)
    midpoint_err = cache.max_midpoint_errors_m(ephemeris)
    sun0 = ephemeris.state("SUN", t0)
    sun = ephemeris.states("SUN", t)
    at_start = t == t0

    props: list[L1Propagation] = []
    stack_of: list[int] = []
    sizes: list[int] = []
    scales: list[float] = []
    for k, start in enumerate(range(0, len(inits), cap)):
        group = inits[start : start + cap]
        n = len(group)
        scale = 1.0 / math.sqrt(n)
        sizes.append(n)
        scales.append(scale)
        if n == 1:
            # A stack of one is the solo problem: integrate it with propagate_l1 itself (same
            # equations, bit-identical to a single-satellite run).
            props.append(propagate_l1(group[0], t, ephemeris=ephemeris, settings=settings))
            stack_of.append(k)
            continue
        # max_stack_size keeps rtol * scale >= the floor up to round-off (1e-12 * 0.1 is
        # 9.999...e-14 in binary); clamp that round-off only.
        rtol_eff = max(settings.rtol * scale, INTEGRATOR_RTOL_MIN) if n > 1 else settings.rtol
        stack_settings = replace(
            settings,
            rtol=rtol_eff,
            atol_position_m=settings.atol_position_m * scale,
            atol_velocity_m_s=settings.atol_velocity_m_s * scale,
        )
        rhs = _make_stacked_heliocentric_rhs(cache, mu_eff, gm_others, n)
        y0 = np.concatenate(
            [
                np.concatenate([s.position_m - sun0.position_m, s.velocity_m_s - sun0.velocity_m_s])
                for s in group
            ]
        )
        atol = np.tile(
            np.array(
                [stack_settings.atol_position_m] * 3 + [stack_settings.atol_velocity_m_s] * 3,
                dtype=np.float64,
            ),
            n,
        )
        out = np.empty((t.size, 6 * n), dtype=np.float64)
        out[at_start] = y0
        segments: list[IntegrationSegment] = []
        for direction in ("forward", "backward"):
            mask = t > t0 if direction == "forward" else t < t0
            if not np.any(mask):
                continue
            t_eval = t[mask] if direction == "forward" else t[mask][::-1]
            t_end = float(t_eval[-1])
            sol = solve_ivp(
                rhs,
                (t0, t_end),
                y0,
                method=settings.method,
                t_eval=t_eval,
                rtol=stack_settings.rtol,
                atol=atol,
                max_step=settings.max_step_s,
            )
            segments.append(
                IntegrationSegment(
                    direction=direction,
                    t_start_tdb_s=t0,
                    t_end_tdb_s=t_end,
                    n_output=int(sol.t.size),
                    nfev=int(sol.nfev),
                    success=bool(sol.success),
                    message=str(sol.message),
                )
            )
            if not sol.success or sol.t.size != t_eval.size:
                raise PropagationError(
                    f"stacked L1 {direction} integration ({n} satellites) from {t0} to {t_end} "
                    f"TDB s failed after {sol.nfev} evaluations: {sol.message}"
                )
            out[mask] = sol.y.T if direction == "forward" else sol.y.T[::-1]
        for j, init in enumerate(group):
            helio = out[:, 6 * j : 6 * j + 6]
            ssb_pos = helio[:, :3] + sun.position_m
            ssb_vel = helio[:, 3:] + sun.velocity_m_s
            ssb_pos[at_start] = init.position_m
            ssb_vel[at_start] = init.velocity_m_s
            traj = CartesianTrajectory(
                epochs_tdb_s=t, position_m=ssb_pos, velocity_m_s=ssb_vel, frame="J2000", origin=SSB
            )
            e0 = float(
                _two_body_energy(
                    init.position_m - sun0.position_m,
                    init.velocity_m_s - sun0.velocity_m_s,
                    mu_eff,
                )
            )
            e = _two_body_energy(helio[:, :3], helio[:, 3:], mu_eff)
            diag = L1Diagnostics(
                formulation="heliocentric",
                segments=tuple(segments),
                nfev=sum(sg.nfev for sg in segments),
                success=all(sg.success for sg in segments),
                gm_m3_s2=(
                    ("SUN", gm_sun),
                    *((b, float(g)) for b, g in zip(others, gm_others, strict=True)),
                ),
                sun_two_body_energy_rel_drift=float(np.max(np.abs(e - e0)) / abs(e0)),
                cache_step_s=cache.step_s,
                cache_span_tdb_s=(cache.t0_tdb_s, cache.end_tdb_s),
                cache_midpoint_error_m=tuple(
                    (b, float(err)) for b, err in zip(order, midpoint_err, strict=True)
                ),
            )
            props.append(
                L1Propagation(
                    trajectory=traj,
                    initial=init,
                    settings=stack_settings,
                    diagnostics=diag,
                    source_kernel_ids=ephemeris.source_kernel_ids,
                )
            )
            stack_of.append(k)
    return StackedL1Propagation(
        propagations=tuple(props),
        stack_index=tuple(stack_of),
        stack_sizes=tuple(sizes),
        tolerance_scale=tuple(scales),
    )


def _two_body_energy(rel_pos: Float64Array, rel_vel: Float64Array, mu_eff: float) -> Float64Array:
    r = np.sqrt(np.sum(rel_pos * rel_pos, axis=-1))
    return np.asarray(0.5 * np.sum(rel_vel * rel_vel, axis=-1) - mu_eff / r)


# ---------------------------------------------------------------------------------------------
# Run-level assessment (step convergence and SRP effect)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class L1RunAssessment:
    """Step-convergence diagnostic and SRP-effect report of one L1 run (PDF §13.3 "time-step
    convergence", §6.3 "report ... for which SRP materially changes the orbit").

    Every number is a max over the run's output epochs of the satellite position difference
    between the run and a variant re-integrated from the same initial state [m].

    Attributes:
        step_halving_max_delta_m: Variant with ``max_step_s / 2`` (``None`` if ``max_step_s``
            is unlimited).
        rtol_tightening_max_delta_m: Variant with ``rtol / 10`` (``None`` if that is below
            :data:`~helioswarm.config.INTEGRATOR_RTOL_MIN`).
        srp_effect_max_m: Variant with SRP switched off (``None`` if the run has no SRP).
        srp_effect_final_m: The same at the last output epoch (``None`` without SRP).
        srp_lightness_number: ``beta = k / GM_sun`` (``None`` without SRP).
    """

    step_halving_max_delta_m: float | None
    rtol_tightening_max_delta_m: float | None
    srp_effect_max_m: float | None
    srp_effect_final_m: float | None
    srp_lightness_number: float | None


def assess_l1_run(run: L1Propagation, *, ephemeris: EphemerisProvider) -> L1RunAssessment:
    """Re-integrate ``run`` with a halved step cap, a tenfold tighter rtol and without SRP.

    Costs up to three further propagations (the halved-step one about twice the original), so
    it is a separate, explicit call, not part of :func:`propagate_l1`.

    Args:
        run: A completed propagation (its initial state, epochs, settings and formulation are
            reused).
        ephemeris: The provider used for ``run``.
    """
    t = run.trajectory.epochs_tdb_s
    base = run.trajectory.position_m
    formulation = run.diagnostics.formulation

    def max_delta(initial: SatelliteState, settings: IntegratorSettings) -> Float64Array:
        other = propagate_l1(
            initial, t, ephemeris=ephemeris, settings=settings, formulation=formulation
        ).trajectory.position_m
        return np.asarray(np.linalg.norm(other - base, axis=1), dtype=np.float64)

    s = run.settings
    step = None
    if math.isfinite(s.max_step_s):
        step = float(np.max(max_delta(run.initial, replace(s, max_step_s=s.max_step_s / 2))))
    tight = None
    if s.rtol / 10.0 >= INTEGRATOR_RTOL_MIN:
        tight = float(np.max(max_delta(run.initial, replace(s, rtol=s.rtol / 10.0))))
    srp_max = srp_final = beta = None
    fm = run.initial.force_model
    if fm.srp is not None:
        off = SatelliteState(
            state=run.initial.state,
            force_model=ForceModel(fm.bodies),
            mass_kg=run.initial.mass_kg,
            attitude=run.initial.attitude,
            collector_normal_unit=run.initial.collector_normal_unit,
        )
        d = max_delta(off, s)
        srp_max, srp_final = float(np.max(d)), float(d[-1])
        beta = fm.srp.lightness_number(ephemeris.gm_m3_s2("SUN"))
    return L1RunAssessment(
        step_halving_max_delta_m=step,
        rtol_tightening_max_delta_m=tight,
        srp_effect_max_m=srp_max,
        srp_effect_final_m=srp_final,
        srp_lightness_number=beta,
    )


# ---------------------------------------------------------------------------------------------
# Configuration hooks
# ---------------------------------------------------------------------------------------------


def force_model_from_config(orbit: OrbitConfig, constants: ConstantsConfig) -> ForceModel:
    """Force model of the configured orbit.

    ``bodies = ("SUN", *orbit.perturbing_bodies)`` if ``include_planetary_perturbations`` else
    ``("SUN",)``; SRP from ``orbit.cr``, ``orbit.area_to_mass_m2_kg`` and
    ``constants.solar_constant_1au_W_m2`` (``P_srp_1AU = S_1AU / c``) if enabled.
    """
    bodies: tuple[BodyName, ...] = ("SUN",)
    if orbit.include_planetary_perturbations:
        bodies = ("SUN", *orbit.perturbing_bodies)
    srp = None
    if orbit.include_solar_radiation_pressure:
        # Both are schema-enforced when SRP is enabled.
        assert orbit.cr is not None
        assert orbit.area_to_mass_m2_kg is not None
        srp = SrpModel.from_solar_constant(
            cr=orbit.cr,
            area_to_mass_m2_kg=orbit.area_to_mass_m2_kg,
            solar_constant_1au_W_m2=constants.solar_constant_1au_W_m2,
        )
    return ForceModel(bodies=bodies, srp=srp)


def integrator_settings_from_config(time: TimeConfig) -> IntegratorSettings:
    """Integrator settings from the ``time`` section (``integrator_step_s`` is ``max_step``)."""
    if time.integrator_step_s is None:
        raise ValueError("time.integrator_step_s is required by the L1 integrator")
    return IntegratorSettings(
        rtol=time.integrator_rtol,
        atol_position_m=time.integrator_atol_position_m,
        atol_velocity_m_s=time.integrator_atol_velocity_m_s,
        max_step_s=time.integrator_step_s,
        ephemeris_cache_step_s=time.ephemeris_cache_step_s,
    )


def satellite_initial_state(
    orbit: OrbitConfig,
    constants: ConstantsConfig,
    *,
    epoch_tdb_s: float,
    ephemeris: EphemerisProvider,
    attitude: Attitude = "sun_tracking",
    fixed_normal_unit_j2000: tuple[float, float, float] | None = None,
) -> SatelliteState:
    """Initial satellite state in J2000/SSB from the configured heliocentric elements.

    Steps (PDF §6.1 item 3): (1) osculating Sun-centred ECLIPJ2000 elements with
    ``mu = GM_sun`` of the provider -> Cartesian state (:func:`helioswarm.orbit.elements_to_state`);
    (2) rotate ECLIPJ2000 -> J2000 (``R_x(-eps_0)``); (3) add the Sun's J2000/SSB state at the
    same epoch. With SRP enabled the elements remain *gravitational* osculating elements
    (mu = GM_sun, not GM_sun - k_srp). An ``earth_relative`` phase is first resolved to a true
    anomaly with :func:`resolve_orbit_phase` (the Earth's geometric heliocentric ecliptic
    longitude at the element epoch).

    The collector normal (PDF p.4 ``SatelliteState`` "attitude/normal vector") is
    ``n = -r_helio/|r_helio|`` (towards the Sun's centre, geometric direction at the same epoch)
    for ``sun_tracking`` and the configured J2000 unit vector for ``fixed_normal``.

    Raises:
        ValueError: ``orbit.elements_center`` is not ``SUN``. Earth-centred orbits (M7b) have
            no satellite initial state here: ``mu = GM_sun`` and the Sun-relative collector
            normal above would be wrong for them; their trajectory is
            :func:`helioswarm.geocentric.geocentric_l0_trajectory` (via
            :func:`l0_reference_trajectory`).
    """
    if orbit.elements_center != "SUN":
        raise ValueError(
            f"satellite_initial_state builds heliocentric states only (orbit.elements_center = "
            f"'{orbit.elements_center}'); Earth-centred orbits use "
            "helioswarm.geocentric.geocentric_l0_trajectory (L0 + station-keeping, M7b)"
        )
    elements = elements_from_config(
        orbit,
        epoch_tdb_s=epoch_tdb_s,
        mu_m3_s2=ephemeris.gm_m3_s2("SUN"),
        true_anomaly_rad=_resolved_nu(orbit, epoch_tdb_s, ephemeris),
    )
    helio_ecl = elements_to_state(elements)
    helio_j2000 = transform_state(helio_ecl, "J2000")
    ssb = shift_state_origin(helio_j2000, SSB, ephemeris)
    if attitude == "sun_tracking":
        if fixed_normal_unit_j2000 is not None:
            raise ValueError("sun_tracking takes no fixed_normal_unit_j2000")
        r = helio_j2000.position_m
        n = -r / float(np.linalg.norm(r))
        normal = (float(n[0]), float(n[1]), float(n[2]))
    elif attitude == "fixed_normal":
        if fixed_normal_unit_j2000 is None:
            raise ValueError("fixed_normal requires fixed_normal_unit_j2000")
        normal = fixed_normal_unit_j2000
    else:
        raise ValueError(f"unknown attitude {attitude!r}")
    return SatelliteState(
        state=ssb,
        force_model=force_model_from_config(orbit, constants),
        attitude=attitude,
        collector_normal_unit=normal,
    )


# ---------------------------------------------------------------------------------------------
# Orbit phase reference (absolute true anomaly or Earth-relative ecliptic longitude; M7)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitPhase:
    """Resolved phase of a configured orbit at its element epoch.

    Attributes:
        phase_reference: ``absolute`` or ``earth_relative`` (``orbit.phase_reference``).
        configured_angle_rad: ``orbit.true_anomaly_deg`` in radians (the true anomaly, or the
            Earth-relative ecliptic longitude offset) [rad].
        true_anomaly_rad: True anomaly of the osculating elements, in [0, 2 pi) [rad].
        ecliptic_longitude_rad: Heliocentric ECLIPJ2000 longitude of the satellite at the
            epoch, in [0, 2 pi) [rad].
        earth_ecliptic_longitude_rad: Geometric heliocentric ECLIPJ2000 longitude of the Earth
            (NAIF 399) at the epoch, in [0, 2 pi) [rad]; ``None`` for ``absolute`` phases (not
            needed, not computed).
        epoch_tdb_s: Element epoch [TDB s].
    """

    phase_reference: str
    configured_angle_rad: float
    true_anomaly_rad: float
    ecliptic_longitude_rad: float
    earth_ecliptic_longitude_rad: float | None
    epoch_tdb_s: float


def _wrap(angle_rad: float) -> float:
    out = math.fmod(angle_rad, 2.0 * math.pi)
    if out < 0.0:
        out += 2.0 * math.pi
    return 0.0 if out >= 2.0 * math.pi else out


def earth_ecliptic_longitude_rad(epoch_tdb_s: float, ephemeris: EphemerisProvider) -> float:
    """Geometric heliocentric ecliptic longitude of the Earth [rad], in [0, 2 pi).

    ``r = r_earth(t) - r_sun(t)`` (both J2000 / SSB at the same TDB epoch, no light time),
    rotated to ECLIPJ2000 (``R_x(eps_0)``, :func:`helioswarm.frames.rotate_vectors`),
    ``lambda = atan2(y, x)``. The Earth is NAIF 399 (the planet, not the Earth-Moon barycentre).
    """
    earth = ephemeris.state("EARTH", epoch_tdb_s)
    sun = ephemeris.state("SUN", epoch_tdb_s)
    rel = np.asarray(earth.position_m, dtype=np.float64) - np.asarray(
        sun.position_m, dtype=np.float64
    )
    ecl = rotate_vectors(rel.reshape(1, 3), "J2000", "ECLIPJ2000")[0]
    return _wrap(math.atan2(float(ecl[1]), float(ecl[0])))


def argument_of_latitude_for_longitude_rad(
    ecliptic_longitude_rad: float, *, inclination_rad: float, raan_rad: float
) -> float:
    """Argument of latitude ``u`` at which an orbit reaches a given ecliptic longitude [rad].

    The position direction is ``cos u P + sin u Q`` with ``P = (cos Omega, sin Omega, 0)`` and
    ``Q = (-sin Omega cos i, cos Omega cos i, sin i)`` (:mod:`helioswarm.orbit`), so in the
    frame rotated by ``Omega`` about the ecliptic pole its ecliptic components are
    ``(cos u, sin u cos i)`` and ``tan(lambda - Omega) = cos i tan u``. With
    ``Delta = lambda - Omega``: ``u = atan2(sin Delta sgn(cos i), cos Delta |cos i|)`` (the
    quadrant of ``u`` is fixed because ``cos u`` and ``cos Delta`` share their sign). Undefined
    for polar orbits (``cos i = 0``; rejected by the configuration schema).

    Raises:
        ValueError: ``|cos i|`` below
            :data:`~helioswarm.config.EARTH_RELATIVE_MIN_ABS_COS_INCLINATION`.
    """
    from helioswarm.config import EARTH_RELATIVE_MIN_ABS_COS_INCLINATION

    ci = math.cos(inclination_rad)
    if abs(ci) < EARTH_RELATIVE_MIN_ABS_COS_INCLINATION:
        raise ValueError("an ecliptic longitude does not determine the phase of a polar orbit")
    delta = ecliptic_longitude_rad - raan_rad
    return _wrap(math.atan2(math.sin(delta) * math.copysign(1.0, ci), math.cos(delta) * abs(ci)))


def resolve_orbit_phase(
    orbit: OrbitConfig, *, epoch_tdb_s: float, ephemeris: EphemerisProvider
) -> OrbitPhase:
    """Resolve ``orbit.true_anomaly_deg`` under ``orbit.phase_reference`` (M7).

    * ``absolute``: the configured angle *is* the true anomaly ``nu``.
    * ``earth_relative``: the configured angle ``phi`` is the offset of the satellite's
      heliocentric ECLIPJ2000 longitude from the Earth's (:func:`earth_ecliptic_longitude_rad`)
      at the element epoch, positive ahead of the Earth in its (prograde) orbital motion:
      ``lambda_sat = lambda_earth + phi``; then ``u`` from
      :func:`argument_of_latitude_for_longitude_rad` and ``nu = u - omega`` (mod 2 pi).

    The ecliptic longitude of the satellite is returned for both references (from
    ``u = omega + nu``: ``lambda = Omega + atan2(sin u cos i, cos u)``).
    """
    i, raan, argp = orbit.inclination_rad, orbit.raan_rad, orbit.arg_periapsis_rad
    angle = orbit.true_anomaly_rad
    lam_earth: float | None = None
    if orbit.phase_reference == "absolute":
        nu = _wrap(angle)
    elif orbit.phase_reference == "earth_relative":
        lam_earth = earth_ecliptic_longitude_rad(epoch_tdb_s, ephemeris)
        u = argument_of_latitude_for_longitude_rad(
            lam_earth + angle, inclination_rad=i, raan_rad=raan
        )
        nu = _wrap(u - argp)
    else:  # pragma: no cover - schema Literal
        raise ValueError(f"unknown phase_reference {orbit.phase_reference!r}")
    u_sat = argp + nu
    lam = _wrap(raan + math.atan2(math.sin(u_sat) * math.cos(i), math.cos(u_sat)))
    return OrbitPhase(
        phase_reference=orbit.phase_reference,
        configured_angle_rad=angle,
        true_anomaly_rad=nu,
        ecliptic_longitude_rad=lam,
        earth_ecliptic_longitude_rad=lam_earth,
        epoch_tdb_s=float(epoch_tdb_s),
    )


def _resolved_nu(
    orbit: OrbitConfig, epoch_tdb_s: float, ephemeris: EphemerisProvider
) -> float | None:
    """True anomaly to pass to :func:`~helioswarm.orbit.elements_from_config` (``None`` for
    ``absolute`` phases: the configured value is used unchanged, bit for bit)."""
    if orbit.phase_reference == "absolute":
        return None
    return resolve_orbit_phase(orbit, epoch_tdb_s=epoch_tdb_s, ephemeris=ephemeris).true_anomaly_rad


def l0_reference_trajectory(
    orbit: OrbitConfig,
    epochs_tdb_s: ArrayLike,
    *,
    initial_epoch_tdb_s: float,
    ephemeris: EphemerisProvider,
    earth_orientation: EarthOrientation | None = None,
) -> CartesianTrajectory:
    """L0 (two-body Kepler) reference trajectory of the configured orbit, in J2000 / SSB.

    The *orbit-drift* reference of PDF §7.3: the configured osculating Sun-centred ECLIPJ2000
    elements at ``initial_epoch_tdb_s`` with ``mu = GM_sun`` of the provider (no planets, no
    SRP), propagated in closed form (:func:`helioswarm.orbit.propagate_kepler`), rotated to
    J2000 (``R_x(-eps_0)``) and shifted to the SSB with the provider's Sun state at each epoch
    (``r_SSB = r_helio + r_sun(t)``). Its initial state equals :func:`satellite_initial_state`,
    so ``|r_L1(t) - r_L0(t)|`` is the drift caused by the planets and SRP.

    Earth-centred orbits (``orbit.elements_center = 'EARTH'``, M7b) are delegated to
    :func:`helioswarm.geocentric.geocentric_l0_trajectory` (Earth point mass, ``mu =
    GM_earth``, J2000 equatorial elements or the geostationary preset; shifted to the SSB with
    the Earth's state).

    Args:
        orbit: Orbit section of the configuration.
        epochs_tdb_s: Output epochs [TDB s past J2000].
        initial_epoch_tdb_s: Osculation epoch of the elements [TDB s past J2000].
        ephemeris: Provider of the Sun's (Earth's) state and GM.
        earth_orientation: Earth-fixed -> J2000 model; required by the geostationary preset
            only.

    Returns:
        J2000 / SSB trajectory [m, m s^-1].
    """
    if orbit.elements_center == "EARTH":
        from helioswarm.geocentric import geocentric_l0_trajectory

        return geocentric_l0_trajectory(
            orbit,
            epochs_tdb_s,
            initial_epoch_tdb_s=initial_epoch_tdb_s,
            ephemeris=ephemeris,
            earth_orientation=earth_orientation,
        )
    elements = elements_from_config(
        orbit,
        epoch_tdb_s=initial_epoch_tdb_s,
        mu_m3_s2=ephemeris.gm_m3_s2("SUN"),
        true_anomaly_rad=_resolved_nu(orbit, initial_epoch_tdb_s, ephemeris),
    )
    helio = propagate_kepler(elements, epochs_tdb_s).trajectory
    return shift_trajectory_origin(transform_trajectory(helio, "J2000"), SSB, ephemeris)


class KernelSetMismatchError(ValueError):
    """The loaded SPICE kernel set is not the one the configuration names."""


def check_kernel_set(config: RunConfig, kernels: LoadedKernels) -> None:
    """Raise :class:`KernelSetMismatchError` unless ``kernels`` is the configured kernel set.

    ``frames.ephemeris_kernel_set`` names the ``kernel_set_id`` of the manifest the run must
    use, so two runs with the same configuration hash cannot silently use different
    ephemerides or GM values (the manifest's SHA-256 values pin the files themselves).
    """
    loaded = kernels.kernel_set.manifest.kernel_set_id
    wanted = config.frames.ephemeris_kernel_set
    if loaded != wanted:
        raise KernelSetMismatchError(
            f"the loaded kernel set is '{loaded}' ({kernels.kernel_set.manifest.path}) but the "
            f"configuration requires frames.ephemeris_kernel_set='{wanted}'"
        )


def initial_state_from_config(
    config: RunConfig, ephemeris: EphemerisProvider, kernels: LoadedKernels
) -> SatelliteState:
    """:func:`satellite_initial_state` at ``orbit.initial_epoch_utc`` (UTC -> TDB by SPICE).

    Raises:
        KernelSetMismatchError: ``kernels`` is not ``frames.ephemeris_kernel_set``.
    """
    check_kernel_set(config, kernels)
    epoch = utc_to_tdb_s(config.orbit.initial_epoch_utc, kernels)
    return satellite_initial_state(
        config.orbit,
        config.constants,
        epoch_tdb_s=epoch,
        ephemeris=ephemeris,
        attitude=config.collector.incidence_mode,
        fixed_normal_unit_j2000=config.collector.fixed_normal_unit_j2000,
    )


@dataclass(frozen=True)
class ConfiguredL1Run:
    """An L1 propagation of a configured run on its output grid.

    Attributes:
        grid: Output time grid (TDB-uniform, from ``time.start_utc``).
        propagation: The propagation result on that grid.
    """

    grid: TimeGrid
    propagation: L1Propagation


def propagate_l1_from_config(
    config: RunConfig, ephemeris: EphemerisProvider, kernels: LoadedKernels
) -> ConfiguredL1Run:
    """Propagate a configuration with ``orbit.model`` = L1 on its output grid.

    The initial state is built at ``orbit.initial_epoch_utc``; the output grid starts at
    ``time.start_utc`` (the two may differ; epochs before the initial epoch are integrated
    backward).

    Raises:
        KernelSetMismatchError: ``kernels`` is not ``frames.ephemeris_kernel_set``.
    """
    if config.orbit.model != L1_MODEL:
        raise ValueError(f"orbit.model must be '{L1_MODEL}', got '{config.orbit.model}'")
    check_kernel_set(config, kernels)
    grid = time_grid_from_config(config.time, kernels)
    initial = initial_state_from_config(config, ephemeris, kernels)
    prop = propagate_l1(
        initial,
        grid.epochs_tdb_s,
        ephemeris=ephemeris,
        settings=integrator_settings_from_config(config.time),
    )
    return ConfiguredL1Run(grid=grid, propagation=prop)


__all__ = [
    "L1_MODEL",
    "NORMAL_UNIT_TOLERANCE",
    "PLANETARY_SYSTEMS",
    "Attitude",
    "ConfiguredL1Run",
    "ForceModel",
    "Formulation",
    "IntegrationSegment",
    "IntegratorSettings",
    "KernelSetMismatchError",
    "L1Diagnostics",
    "L1Propagation",
    "L1RunAssessment",
    "OrbitPhase",
    "PropagationError",
    "SatelliteState",
    "StackedL1Propagation",
    "argument_of_latitude_for_longitude_rad",
    "assess_l1_run",
    "check_kernel_set",
    "earth_ecliptic_longitude_rad",
    "effective_gm_m3_s2",
    "force_model_from_config",
    "heliocentric_l1_acceleration_m_s2",
    "initial_state_from_config",
    "integrator_settings_from_config",
    "l0_reference_trajectory",
    "l1_acceleration_m_s2",
    "max_stack_size",
    "point_mass_acceleration_m_s2",
    "propagate_l1",
    "propagate_l1_from_config",
    "propagate_l1_stacked",
    "resolve_orbit_phase",
    "satellite_initial_state",
]
