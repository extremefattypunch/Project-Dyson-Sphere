"""Planetary ephemerides: an injectable provider protocol, SPICE and analytic implementations,
origin changes Sun <-> SSB, and a cubic-Hermite cache for the L1 integrator.

Every state returned here is in the inertial frame ``J2000`` with origin
``SOLAR SYSTEM BARYCENTER`` (SSB), in SI units (m, m/s), at an epoch in TDB seconds past J2000
(PDF §3, §6.1). Callers pass a provider explicitly; nothing here holds global state apart from
SPICE's own kernel pool, which is only touched inside a :meth:`helioswarm.kernels.KernelSet
.loaded` scope (the provider checks the scope on every call).

* :class:`SpiceEphemeris`: geometric states from the loaded SPK (``spkgeo``; no light-time or
  aberration correction) converted km -> m, and gravitational parameters from the loaded
  ``gm_de440.tpc`` (``bodvrd``; km^3 s^-2 -> m^3 s^-2). It refuses epochs outside the SPK
  coverage with a clear error (de440s: 1849-12-26 to 2150-01-22 TDB).
* :class:`AnalyticEphemeris`: a kernel-free test double. The Sun moves uniformly (default: at
  rest at the origin) and the Earth moves on a circular Kepler orbit about it in the ecliptic.
  It exists so that unit tests of the propagator and of frame/origin logic run without kernels
  and against closed forms; it is not a physical ephemeris.
* :class:`HermiteEphemerisCache`: positions and velocities sampled on a uniform TDB grid and
  interpolated with piecewise cubic Hermite polynomials. Error bound per component
  ``|e| <= h^4 / 384 * max|x''''|`` (e.g. de Boor, *A Practical Guide to Splines*, ch. 2); the
  default ``h = 3600 s`` is validated against direct SPICE calls (< 1 m) in the tests.

NAIF IDs (NAIF "NAIF IDs Required Reading"): SUN 10, MERCURY BARYCENTER 1, VENUS BARYCENTER 2,
EARTH BARYCENTER 3, MARS BARYCENTER 4, JUPITER BARYCENTER 5, SATURN BARYCENTER 6, URANUS
BARYCENTER 7, NEPTUNE BARYCENTER 8, EARTH 399, MOON 301.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, Protocol, get_args, runtime_checkable

import numpy as np
import spiceypy
from numpy.typing import ArrayLike, NDArray

from helioswarm.constants import AU_M, GM_SUN_M3_S2, M_PER_KM
from helioswarm.frames import (
    CartesianState,
    CartesianTrajectory,
    FrameMismatchError,
    FrameName,
    OriginName,
    transform_trajectory,
)
from helioswarm.kernels import LoadedKernels
from helioswarm.orbit import OrbitalElements, propagate_kepler

BodyName = Literal[
    "SUN",
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "EARTH BARYCENTER",
    "EARTH",
    "MOON",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
]
"""Bodies an ephemeris provider can be asked for (SPICE body names)."""

NAIF_IDS: Final[Mapping[str, int]] = MappingProxyType(
    {
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
)
"""NAIF integer codes of :data:`BodyName` (NAIF IDs Required Reading)."""

SUPPORTED_BODIES: Final[tuple[str, ...]] = get_args(BodyName)

SSB: Final[OriginName] = "SOLAR SYSTEM BARYCENTER"

Float64Array = NDArray[np.float64]


class EphemerisCoverageError(ValueError):
    """Requested epochs lie outside the ephemeris coverage."""


def _check_body(body: str) -> int:
    try:
        return NAIF_IDS[body]
    except KeyError:
        raise ValueError(
            f"unsupported body {body!r}; supported: {list(SUPPORTED_BODIES)}"
        ) from None


def _epochs_1d(epochs_tdb_s: ArrayLike) -> Float64Array:
    t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("epochs_tdb_s must be a finite scalar or 1-D array")
    return t


# ---------------------------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EphemerisState:
    """State of a body from an ephemeris (PDF p.4 ``EphemerisState``).

    Attributes:
        body: SPICE body name.
        epoch_tdb_s: Epoch, TDB seconds past J2000 [s].
        position_m: Position, shape (3,) [m].
        velocity_m_s: Velocity, shape (3,) [m s^-1].
        source_kernel_ids: Kernels (file names) the state was computed from;
            ``("analytic",)`` for :class:`AnalyticEphemeris`.
        frame: Always ``J2000``.
        origin: Always ``SOLAR SYSTEM BARYCENTER``.
        time_scale: Always ``TDB``.
    """

    body: BodyName
    epoch_tdb_s: float
    position_m: Float64Array
    velocity_m_s: Float64Array
    source_kernel_ids: tuple[str, ...]
    frame: FrameName = field(default="J2000")
    origin: OriginName = field(default=SSB)
    time_scale: Literal["TDB"] = field(default="TDB")

    def __post_init__(self) -> None:
        _check_body(self.body)
        if self.frame != "J2000" or self.origin != SSB:
            raise ValueError("ephemeris states are J2000 / SOLAR SYSTEM BARYCENTER by contract")
        if not self.source_kernel_ids:
            raise ValueError("source_kernel_ids must not be empty")
        cart = self.as_cartesian()  # validates shapes, finiteness, time scale
        object.__setattr__(self, "epoch_tdb_s", cart.epoch_tdb_s)
        object.__setattr__(self, "position_m", cart.position_m)
        object.__setattr__(self, "velocity_m_s", cart.velocity_m_s)

    def as_cartesian(self) -> CartesianState:
        """The same state as a :class:`~helioswarm.frames.CartesianState` (J2000/SSB)."""
        return CartesianState(
            epoch_tdb_s=self.epoch_tdb_s,
            position_m=self.position_m,
            velocity_m_s=self.velocity_m_s,
            frame=self.frame,
            origin=self.origin,
        )


# ---------------------------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------------------------


@runtime_checkable
class EphemerisProvider(Protocol):
    """Source of body states in J2000 / SSB, SI units, TDB epochs.

    Implementations: :class:`SpiceEphemeris` (authoritative, PDF §3) and
    :class:`AnalyticEphemeris` (kernel-free test double).
    """

    @property
    def frame(self) -> FrameName:
        """Frame of every returned state (``J2000``)."""
        ...

    @property
    def origin(self) -> OriginName:
        """Origin of every returned state (``SOLAR SYSTEM BARYCENTER``)."""
        ...

    @property
    def source_kernel_ids(self) -> tuple[str, ...]:
        """Ids of the data the states come from (kernel file names, or ``("analytic",)``)."""
        ...

    def state(self, body: BodyName, epoch_tdb_s: float) -> EphemerisState:
        """State of ``body`` at one TDB epoch."""
        ...

    def states(self, body: BodyName, epochs_tdb_s: ArrayLike) -> CartesianTrajectory:
        """States of ``body`` at an array of TDB epochs (J2000 / SSB)."""
        ...

    def positions(self, body: BodyName, epochs_tdb_s: ArrayLike) -> Float64Array:
        """Positions of ``body`` [m], shape (N, 3), at an array of TDB epochs."""
        ...

    def gm_m3_s2(self, body: BodyName) -> float:
        """Gravitational parameter GM of ``body`` [m^3 s^-2]."""
        ...

    def check_coverage(
        self, bodies: Iterable[BodyName], start_tdb_s: float, end_tdb_s: float
    ) -> None:
        """Raise :class:`EphemerisCoverageError` unless all bodies are covered on the span."""
        ...


# ---------------------------------------------------------------------------------------------
# SPICE implementation
# ---------------------------------------------------------------------------------------------


def _segment_chain(naif_id: int) -> tuple[int, ...]:
    """NAIF ids whose SPK segments are needed to reach the SSB from ``naif_id``.

    In the DE4xx SPKs a planet or satellite (id >= 100) is stored relative to its system
    barycentre (id // 100), and barycentres and the Sun relative to the SSB (0).
    """
    return (naif_id, naif_id // 100) if naif_id >= 100 else (naif_id,)


@dataclass(frozen=True)
class SpiceEphemeris:
    """Ephemeris backed by the loaded SPICE kernel set.

    States: ``spkgeo(id, et, "J2000", 0)`` (geometric; km, km/s) times 1000.
    GM: ``bodvrd(name, "GM", 1)`` (km^3 s^-2) times 1e9.

    The kernels' GM values are the ones used whenever this provider is used (the L1 propagator
    asks the provider). ``helioswarm.constants.GM_SUN_M3_S2`` is the same DE440 number written
    as a literal for kernel-free code; the two agree to 2 ulp (SPICE's text-kernel parser is
    not correctly rounded), which a test enforces.

    Attributes:
        kernels: Active kernel-set handle (must contain an SPK; GM queries need the GM PCK).
    """

    kernels: LoadedKernels
    frame: FrameName = field(default="J2000", init=False)
    origin: OriginName = field(default=SSB, init=False)
    _coverage: Mapping[str, tuple[tuple[float, float], ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.kernels.require_active(kind="SPK")
        cov = {body: self._compute_coverage(NAIF_IDS[body]) for body in SUPPORTED_BODIES}
        object.__setattr__(self, "_coverage", MappingProxyType(cov))

    @property
    def source_kernel_ids(self) -> tuple[str, ...]:
        """SPK kernel ids of the loaded set."""
        return self.kernels.ids_of_kind("SPK")

    def _compute_coverage(self, naif_id: int) -> tuple[tuple[float, float], ...]:
        spk_paths = tuple(str(self.kernels.kernel_set.path(n)) for n in self.source_kernel_ids)
        window = None
        for code in _segment_chain(naif_id):
            cover = spiceypy.cell_double(2000)
            for path in spk_paths:
                spiceypy.spkcov(path, code, cover)
            window = cover if window is None else spiceypy.wnintd(window, cover)
        return tuple(
            (float(a), float(b))
            for a, b in (spiceypy.wnfetd(window, i) for i in range(spiceypy.wncard(window)))
        )

    def coverage(self, body: BodyName) -> tuple[tuple[float, float], ...]:
        """Coverage intervals [TDB s past J2000] over which ``body`` can be referred to the SSB.

        The intersection of the SPK coverage windows (``spkcov``, union over the loaded SPKs) of
        every segment in the body's chain to the SSB; computed once at construction.
        """
        _check_body(body)
        return self._coverage[body]

    def check_coverage(
        self, bodies: Iterable[BodyName], start_tdb_s: float, end_tdb_s: float
    ) -> None:
        """Raise :class:`EphemerisCoverageError` unless each body's coverage contains the whole
        closed span [start, end] in a single interval."""
        lo, hi = (start_tdb_s, end_tdb_s) if start_tdb_s <= end_tdb_s else (end_tdb_s, start_tdb_s)
        for body in bodies:
            intervals = self.coverage(body)
            if not any(a <= lo and hi <= b for a, b in intervals):
                cov = ", ".join(f"[{a:.6g}, {b:.6g}]" for a, b in intervals) or "none"
                raise EphemerisCoverageError(
                    f"{body}: requested TDB span [{lo:.6g}, {hi:.6g}] s past J2000 is not covered "
                    f"by the loaded SPK(s) {list(self.source_kernel_ids)} (coverage {cov} s)"
                )

    def _state_km(self, naif_id: int, et: float) -> NDArray[np.float64]:
        st, _lt = spiceypy.spkgeo(naif_id, et, "J2000", 0)
        return np.asarray(st, dtype=np.float64)

    def state(self, body: BodyName, epoch_tdb_s: float) -> EphemerisState:
        """Geometric J2000/SSB state of ``body`` at ``epoch_tdb_s`` [m, m/s]."""
        self.kernels.require_active(kind="SPK")
        self.check_coverage((body,), epoch_tdb_s, epoch_tdb_s)
        st = self._state_km(_check_body(body), float(epoch_tdb_s)) * M_PER_KM
        return EphemerisState(
            body=body,
            epoch_tdb_s=float(epoch_tdb_s),
            position_m=st[:3],
            velocity_m_s=st[3:],
            source_kernel_ids=self.source_kernel_ids,
        )

    def states(self, body: BodyName, epochs_tdb_s: ArrayLike) -> CartesianTrajectory:
        """Geometric J2000/SSB states of ``body`` at each epoch [m, m/s]."""
        self.kernels.require_active(kind="SPK")
        t = _epochs_1d(epochs_tdb_s)
        self.check_coverage((body,), float(np.min(t)), float(np.max(t)))
        naif_id = _check_body(body)
        out = np.empty((t.size, 6), dtype=np.float64)
        for k, et in enumerate(t):
            out[k] = self._state_km(naif_id, float(et))
        out *= M_PER_KM
        return CartesianTrajectory(
            epochs_tdb_s=t,
            position_m=out[:, :3],
            velocity_m_s=out[:, 3:],
            frame="J2000",
            origin=SSB,
        )

    def positions(self, body: BodyName, epochs_tdb_s: ArrayLike) -> Float64Array:
        """J2000/SSB positions of ``body`` [m], shape (N, 3)."""
        return np.array(self.states(body, epochs_tdb_s).position_m)

    def gm_m3_s2(self, body: BodyName) -> float:
        """GM of ``body`` from the loaded text PCK (``BODYnnn_GM``), converted to m^3 s^-2."""
        self.kernels.require_active(kind="PCK")
        _check_body(body)
        _n, values = spiceypy.bodvrd(body, "GM", 1)
        gm = float(values[0]) * M_PER_KM**3
        if not (math.isfinite(gm) and gm > 0.0):
            raise ValueError(f"invalid GM for {body}: {gm}")
        return gm


# ---------------------------------------------------------------------------------------------
# Analytic test double
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticEphemeris:
    """Kernel-free ephemeris for unit tests (NOT a physical ephemeris).

    * Sun: uniform motion ``r(t) = sun_position_at_j2000_m + sun_velocity_m_s * t``.
    * Earth: circular Kepler orbit about the Sun of radius ``earth_orbit_radius_m`` in the
      ECLIPJ2000 x-y plane, prograde, true longitude ``earth_longitude_at_j2000_rad`` at
      t = 0, mean motion ``sqrt(GM_sun / a^3)`` (the M1 L0 model), rotated to J2000.
    * GM values from ``gm_table`` (default: the Sun only, DE440 ``GM_SUN_M3_S2``). Asking for a
      GM that is not in the table raises, so a test cannot silently use an invented value.

    Attributes:
        sun_position_at_j2000_m: Sun position at t = 0 (J2000, SSB) [m].
        sun_velocity_m_s: Constant Sun velocity (J2000) [m s^-1].
        earth_orbit_radius_m: Earth's circular-orbit radius about the Sun [m].
        earth_longitude_at_j2000_rad: Earth's ecliptic true longitude at t = 0 [rad].
        gm_table: ``((body, GM [m^3 s^-2]), ...)``.
    """

    sun_position_at_j2000_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sun_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    earth_orbit_radius_m: float = AU_M
    earth_longitude_at_j2000_rad: float = 0.0
    gm_table: tuple[tuple[BodyName, float], ...] = (("SUN", GM_SUN_M3_S2),)
    frame: FrameName = field(default="J2000", init=False)
    origin: OriginName = field(default=SSB, init=False)

    @property
    def source_kernel_ids(self) -> tuple[str, ...]:
        """``("analytic",)``."""
        return ("analytic",)

    def check_coverage(
        self, bodies: Iterable[BodyName], start_tdb_s: float, end_tdb_s: float
    ) -> None:
        """Unlimited coverage; only the body names are checked."""
        for body in bodies:
            if body not in ("SUN", "EARTH"):
                raise EphemerisCoverageError(f"AnalyticEphemeris has no body {body!r}")

    def _sun(self, t: Float64Array) -> tuple[Float64Array, Float64Array]:
        r0 = np.asarray(self.sun_position_at_j2000_m, dtype=np.float64)
        v = np.asarray(self.sun_velocity_m_s, dtype=np.float64)
        pos = r0[None, :] + t[:, None] * v[None, :]
        vel = np.broadcast_to(v, pos.shape).copy()
        return pos, vel

    def states(self, body: BodyName, epochs_tdb_s: ArrayLike) -> CartesianTrajectory:
        """Analytic J2000/SSB states (see class docstring)."""
        t = _epochs_1d(epochs_tdb_s)
        self.check_coverage((body,), float(np.min(t)), float(np.max(t)))
        sun_pos, sun_vel = self._sun(t)
        if body == "SUN":
            pos, vel = sun_pos, sun_vel
        else:
            elements = OrbitalElements(
                a_m=self.earth_orbit_radius_m,
                e=0.0,
                i_rad=0.0,
                raan_rad=0.0,
                argp_rad=0.0,
                nu_rad=self.earth_longitude_at_j2000_rad,
                mu_m3_s2=GM_SUN_M3_S2,
                epoch_tdb_s=0.0,
                frame="ECLIPJ2000",
            )
            helio = transform_trajectory(propagate_kepler(elements, t).trajectory, "J2000")
            pos = sun_pos + helio.position_m
            vel = sun_vel + helio.velocity_m_s
        return CartesianTrajectory(
            epochs_tdb_s=t, position_m=pos, velocity_m_s=vel, frame="J2000", origin=SSB
        )

    def state(self, body: BodyName, epoch_tdb_s: float) -> EphemerisState:
        """Analytic J2000/SSB state at one epoch."""
        traj = self.states(body, epoch_tdb_s)
        return EphemerisState(
            body=body,
            epoch_tdb_s=float(epoch_tdb_s),
            position_m=traj.position_m[0],
            velocity_m_s=traj.velocity_m_s[0],
            source_kernel_ids=self.source_kernel_ids,
        )

    def positions(self, body: BodyName, epochs_tdb_s: ArrayLike) -> Float64Array:
        """Analytic J2000/SSB positions [m], shape (N, 3)."""
        return np.array(self.states(body, epochs_tdb_s).position_m)

    def gm_m3_s2(self, body: BodyName) -> float:
        """GM from ``gm_table`` [m^3 s^-2]; raises ``KeyError`` for bodies not listed."""
        for name, gm in self.gm_table:
            if name == body:
                return float(gm)
        raise KeyError(f"AnalyticEphemeris.gm_table has no GM for {body!r}")


# ---------------------------------------------------------------------------------------------
# Origin changes (Sun <-> SSB)
# ---------------------------------------------------------------------------------------------


def shift_state_origin(
    state: CartesianState, to_origin: OriginName, ephemeris: EphemerisProvider
) -> CartesianState:
    """Re-express a J2000 state relative to another origin (``SUN`` <-> SSB).

    ``r_SSB = r_SUN + r_sun/SSB(t)``, ``v_SSB = v_SUN + v_sun/SSB(t)``, with the Sun's state from
    ``ephemeris`` at the state's own epoch (pure translation between inertial origins; Galilean
    velocity addition, valid for the Newtonian model used here).

    Raises:
        FrameMismatchError: the state is not in the ephemeris frame (rotate it first).
    """
    if state.frame != ephemeris.frame:
        raise FrameMismatchError(
            f"state is in {state.frame}, ephemeris in {ephemeris.frame}; rotate it explicitly first"
        )
    if state.origin == to_origin:
        return state
    sun = ephemeris.state("SUN", state.epoch_tdb_s)
    sign = 1.0 if to_origin == SSB else -1.0
    return CartesianState(
        epoch_tdb_s=state.epoch_tdb_s,
        position_m=state.position_m + sign * sun.position_m,
        velocity_m_s=state.velocity_m_s + sign * sun.velocity_m_s,
        frame=state.frame,
        origin=to_origin,
    )


def shift_trajectory_origin(
    traj: CartesianTrajectory, to_origin: OriginName, ephemeris: EphemerisProvider
) -> CartesianTrajectory:
    """Trajectory version of :func:`shift_state_origin` (Sun state at every epoch)."""
    if traj.frame != ephemeris.frame:
        raise FrameMismatchError(
            f"trajectory is in {traj.frame}, ephemeris in {ephemeris.frame}; rotate it first"
        )
    if traj.origin == to_origin:
        return traj
    sun = ephemeris.states("SUN", traj.epochs_tdb_s)
    sign = 1.0 if to_origin == SSB else -1.0
    return CartesianTrajectory(
        epochs_tdb_s=traj.epochs_tdb_s,
        position_m=traj.position_m + sign * sun.position_m,
        velocity_m_s=traj.velocity_m_s + sign * sun.velocity_m_s,
        frame=traj.frame,
        origin=to_origin,
    )


# ---------------------------------------------------------------------------------------------
# Cubic-Hermite cache
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class HermiteEphemerisCache:
    """Piecewise cubic Hermite interpolant of several bodies' J2000/SSB positions.

    Samples ``t_k = t0 + k h`` (k = 0..N) carry positions ``p_k`` and velocities ``v_k`` from
    the provider. On ``[t_k, t_k + h]`` with ``s = (t - t_k)/h``::

        p(t) = (2s^3 - 3s^2 + 1) p_k + (s^3 - 2s^2 + s) h v_k
             + (-2s^3 + 3s^2) p_{k+1} + (s^3 - s^2) h v_{k+1}

    The interpolant matches position and velocity at every node (C^1). Error per component
    ``<= h^4/384 max|p''''|``. For speed it is stored in power form per interval,
    ``p(s) = c0 + c1 s + c2 s^2 + c3 s^3`` with ``c0 = p_k``, ``c1 = h v_k``,
    ``c2 = 3 (p_{k+1} - p_k) - 2 h v_k - h v_{k+1}``,
    ``c3 = 2 (p_k - p_{k+1}) + h v_k + h v_{k+1}`` (algebraically identical), so one evaluation
    is a (4,) x (4, 3 n_bodies) product.

    Attributes:
        bodies: Body names, in the order of the second axis of the arrays.
        t0_tdb_s: First node [TDB s].
        step_s: Node spacing h [s].
        positions_m: Node positions, shape (N+1, n_bodies, 3) [m].
        velocities_m_s: Node velocities, shape (N+1, n_bodies, 3) [m s^-1].
        source_kernel_ids: Provider's data ids.
    """

    bodies: tuple[BodyName, ...]
    t0_tdb_s: float
    step_s: float
    positions_m: Float64Array
    velocities_m_s: Float64Array
    source_kernel_ids: tuple[str, ...]
    _coef: Float64Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pos = np.array(self.positions_m, dtype=np.float64)
        vel = np.array(self.velocities_m_s, dtype=np.float64)
        nb = len(self.bodies)
        if pos.ndim != 3 or pos.shape[0] < 2 or pos.shape[1:] != (nb, 3) or vel.shape != pos.shape:
            raise ValueError("positions_m/velocities_m_s must have shape (N+1 >= 2, n_bodies, 3)")
        if not (math.isfinite(self.step_s) and self.step_s > 0.0):
            raise ValueError("step_s must be finite and > 0")
        for a in (pos, vel):
            a.setflags(write=False)
        object.__setattr__(self, "positions_m", pos)
        object.__setattr__(self, "velocities_m_s", vel)
        nb3 = 3 * nb
        p0 = pos[:-1].reshape(-1, nb3)
        p1 = pos[1:].reshape(-1, nb3)
        hv0 = vel[:-1].reshape(-1, nb3) * self.step_s
        hv1 = vel[1:].reshape(-1, nb3) * self.step_s
        coef = np.stack(
            [p0, hv0, 3.0 * (p1 - p0) - 2.0 * hv0 - hv1, 2.0 * (p0 - p1) + hv0 + hv1], axis=1
        )
        coef.setflags(write=False)
        object.__setattr__(self, "_coef", coef)

    @classmethod
    def build(
        cls,
        ephemeris: EphemerisProvider,
        bodies: Iterable[BodyName],
        start_tdb_s: float,
        end_tdb_s: float,
        step_s: float,
    ) -> HermiteEphemerisCache:
        """Sample ``bodies`` on ``start + k*step`` covering ``[start, end]`` (end rounded up to
        a whole step; at least one interval) and build the interpolant."""
        names = tuple(bodies)
        if not names:
            raise ValueError("at least one body is required")
        if not end_tdb_s >= start_tdb_s:
            raise ValueError("end_tdb_s must be >= start_tdb_s")
        n = max(1, math.ceil((end_tdb_s - start_tdb_s) / step_s))
        t = start_tdb_s + step_s * np.arange(n + 1, dtype=np.float64)
        ephemeris.check_coverage(names, float(t[0]), float(t[-1]))
        pos = np.empty((n + 1, len(names), 3), dtype=np.float64)
        vel = np.empty_like(pos)
        for j, body in enumerate(names):
            traj = ephemeris.states(body, t)
            pos[:, j, :] = traj.position_m
            vel[:, j, :] = traj.velocity_m_s
        return cls(
            bodies=names,
            t0_tdb_s=float(start_tdb_s),
            step_s=float(step_s),
            positions_m=pos,
            velocities_m_s=vel,
            source_kernel_ids=ephemeris.source_kernel_ids,
        )

    @property
    def end_tdb_s(self) -> float:
        """Last node [TDB s]."""
        return float(self.t0_tdb_s + self.step_s * (self.positions_m.shape[0] - 1))

    def _locate(self, t: float) -> tuple[int, float]:
        x = (t - self.t0_tdb_s) / self.step_s
        n = self.positions_m.shape[0] - 1
        if not -1e-12 <= x <= n + 1e-12:
            raise ValueError(
                f"epoch {t} TDB s outside the cache span [{self.t0_tdb_s}, {self.end_tdb_s}]"
            )
        i = min(max(math.floor(x), 0), n - 1)
        return i, x - i

    def positions_at(self, epoch_tdb_s: float) -> Float64Array:
        """Interpolated positions of all cached bodies at one epoch [m], shape (n_bodies, 3)."""
        i, s = self._locate(epoch_tdb_s)
        basis = np.array([1.0, s, s * s, s * s * s])
        return np.asarray((basis @ self._coef[i]).reshape(-1, 3), dtype=np.float64)

    def positions_many(self, epochs_tdb_s: ArrayLike) -> Float64Array:
        """Interpolated positions at many epochs [m], shape (M, n_bodies, 3)."""
        t = _epochs_1d(epochs_tdb_s)
        return np.stack([self.positions_at(float(x)) for x in t])

    def max_midpoint_errors_m(self, ephemeris: EphemerisProvider) -> Float64Array:
        """Per-body ``max_k |p(t_k + h/2) - provider(t_k + h/2)|`` over all intervals [m].

        At the midpoint (``s = 1/2``) the interpolant is ``(p_k + p_{k+1})/2 + h (v_k -
        v_{k+1})/8`` and the cubic-Hermite error kernel ``s^2 (1 - s)^2`` is at its maximum, so
        this measures the cache error where it is largest (a measured, not a bound, value).

        Args:
            ephemeris: The provider the cache was built from.

        Returns:
            Shape (n_bodies,), in the order of :attr:`bodies`.
        """
        p0, p1 = self.positions_m[:-1], self.positions_m[1:]
        v0, v1 = self.velocities_m_s[:-1], self.velocities_m_s[1:]
        mid_interp = 0.5 * (p0 + p1) + (self.step_s / 8.0) * (v0 - v1)
        n = p0.shape[0]
        t_mid = self.t0_tdb_s + self.step_s * (np.arange(n, dtype=np.float64) + 0.5)
        out = np.empty(len(self.bodies), dtype=np.float64)
        for j, body in enumerate(self.bodies):
            ref = ephemeris.positions(body, t_mid)
            out[j] = float(np.max(np.linalg.norm(mid_interp[:, j, :] - ref, axis=1)))
        return out


__all__ = [
    "NAIF_IDS",
    "SSB",
    "SUPPORTED_BODIES",
    "AnalyticEphemeris",
    "BodyName",
    "EphemerisCoverageError",
    "EphemerisProvider",
    "EphemerisState",
    "HermiteEphemerisCache",
    "SpiceEphemeris",
    "shift_state_origin",
    "shift_trajectory_origin",
]
