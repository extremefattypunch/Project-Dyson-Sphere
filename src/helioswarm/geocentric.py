"""Geocentric orbits for the GEO comparison baseline (M7b; PDF objective O5, §16).

The GEO solar-power-satellite baseline is simulated through the same chain as the
heliocentric swarm; only the orbit differs. This module supplies that orbit and the evidence
for the omissions of its force model.

Model used in runs (L0, Earth point mass)
    Earth-centred (NAIF 399) states in J2000, propagated in closed form
    (:func:`helioswarm.orbit.propagate_kepler`) with ``mu = GM_earth`` from the loaded
    ``gm_de440.tpc`` (``BODY399_GM``), then shifted to the SSB with the Earth's ephemeris
    state at every epoch (``r_SSB = r_geo + r_earth(t)``,
    :func:`helioswarm.ephemeris.shift_trajectory_origin`). The satellite therefore moves on a
    fixed Kepler ellipse about the accelerated Earth centre: the tidal (differential)
    accelerations of the Sun and Moon, the Earth's oblateness (J2, J22, ...) and solar
    radiation pressure are **omitted**. For a geostationary satellite this is the modelled
    assumption "L0 + station-keeping": an operational GEO satellite is held near its nominal
    longitude and in the equatorial plane by north-south and east-west manoeuvres that remove
    exactly these perturbations (their propellant and mass are not modelled); their sizes over
    the run are estimated below and in docs/model_assumptions.md ("M7b").

Geostationary preset (:func:`geostationary_orbit`)
    For a sub-satellite east longitude ``lambda`` (IAU_EARTH) at epoch ``t0``: with the
    IAU_EARTH -> J2000 rotation ``R(t0)`` and its rate ``dR/dt`` (SPICE ``sxform``), the
    frame's angular-velocity vector is ``omega_vec`` from the skew matrix
    ``[omega]_x = (dR/dt) R^T`` (Rotations Required Reading: ``d/dt (R x) = [omega]_x R x`` for a
    point fixed in the body). Its component about the frame's own pole ``p = R e_z`` is the
    spin rate ``omega = omega_vec . p``; for the IAU linear model
    ``omega = dW/dt + (d alpha_0/dt) sin(delta_0)`` (Euler-angle kinematics of
    ``R = [R_z(W) R_x(90 deg - delta_0) R_z(90 deg + alpha_0)]^T``), i.e. pck00011's
    360.9856235 deg/day minus 1.755e-5 deg/day from the pole's precession in right ascension
    (-0.641 deg/century), a stellar day of 86164.10 s. The orbit radius is the two-body
    synchronous radius ``r = (GM_earth / omega^2)^(1/3)`` (Kepler's third law with period
    ``2 pi / omega``), the position ``r R(t0) (cos lambda, sin lambda, 0)`` and the velocity
    ``omega p x r``: a circular orbit in the IAU_EARTH equatorial plane of ``t0`` whose mean
    motion equals the frame's spin, so the sub-satellite point stays at ``lambda`` (drift only
    from the slow change of the IAU pole, measured in the tests: < 1e-5 deg over 10 days).

Perturbation budget (comparison path; not used in runs)
    * :func:`third_body_inclination_rate_vector`: secular rate of the orbit normal of a
      circular orbit under the quadrupole tidal torque of the Sun and the Moon,
      ``dh/dt = -(3 GM_k / (2 n |s_k|^5)) (h . s_k)(h x s_k)`` (orbit-averaged; time-averaged
      over the ephemeris samples), e.g. Murray & Dermott, *Solar System Dynamics*, 1999,
      ch. 6-7 (secular quadrupole theory). For an equatorial orbit and a body on a circular
      orbit of inclination ``i_k`` to the equator the magnitude is
      ``(3/8) (GM_k / (n a_k^3)) sin(2 i_k)``: 0.27 deg/yr for the Sun and 0.47-0.67 deg/yr
      for the Moon (18.6-yr nodal cycle).
    * :func:`propagate_geocentric_point_masses`: numerical integration (DOP853) of the
      geocentric equation with the Earth point mass plus the Sun and Moon third-body terms
      ``mu_k [(s_k - r)/|s_k - r|^3 - s_k/|s_k|^3]`` (J2000; body positions from a cubic-Hermite
      cache of the ephemeris) - the independent check of the averaged rate.
    * :func:`j2_synchronous_radius_offset_m`, :func:`srp_eccentricity_circle_radius`: closed-form
      estimates of the omitted J2 and SRP effects.

Frames, origins and time scales are explicit on every state (``J2000``; ``EARTH`` or SSB; TDB).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from helioswarm.ephemeris import (
    SSB,
    BodyName,
    EphemerisProvider,
    HermiteEphemerisCache,
    shift_trajectory_origin,
)
from helioswarm.frames import CartesianState, CartesianTrajectory
from helioswarm.ground import EarthOrientation
from helioswarm.orbit import (
    OrbitalElements,
    elements_from_config,
    propagate_kepler,
    state_to_elements,
)

if TYPE_CHECKING:
    from helioswarm.config import OrbitConfig

Float64Array = NDArray[np.float64]

EARTH_CENTER: Final = "EARTH"
"""Central body of geocentric states (NAIF 399, the Earth's centre, not the Earth-Moon
barycentre)."""

THIRD_BODIES: Final[tuple[BodyName, ...]] = ("SUN", "MOON")
"""Third bodies of the perturbation-budget integration."""


def angular_velocity_from_rotation_rad_s(
    matrix: ArrayLike, matrix_rate_per_s: ArrayLike
) -> Float64Array:
    """Angular-velocity vector of a rotating frame in inertial components [rad s^-1].

    For ``v_inertial = R v_body`` and a point fixed in the body frame,
    ``d/dt (R x) = (dR/dt) x = [omega]_x R x`` with ``[omega]_x = (dR/dt) R^T`` (skew-symmetric
    for an orthogonal ``R``); ``omega = (W_32, W_13, W_21)``. The skew part
    ``(W - W^T)/2`` is used so that round-off in ``R`` does not leak a symmetric component.

    Args:
        matrix: Rotation matrix body -> inertial, shape (3, 3).
        matrix_rate_per_s: Its time derivative, shape (3, 3) [s^-1].

    Returns:
        Shape (3,) [rad s^-1].
    """
    r = np.asarray(matrix, dtype=np.float64)
    dr = np.asarray(matrix_rate_per_s, dtype=np.float64)
    if r.shape != (3, 3) or dr.shape != (3, 3):
        raise ValueError("matrix and matrix_rate_per_s must have shape (3, 3)")
    w = dr @ r.T
    w = 0.5 * (w - w.T)
    return np.array([w[2, 1], w[0, 2], w[1, 0]], dtype=np.float64)


def geostationary_radius_m(gm_earth_m3_s2: float, spin_rate_rad_s: float) -> float:
    """Two-body synchronous radius ``r = (GM / omega^2)^(1/3)`` [m] (Kepler's third law with
    period ``2 pi / omega``; Earth point mass, no J2)."""
    if not (gm_earth_m3_s2 > 0.0 and spin_rate_rad_s > 0.0):
        raise ValueError("gm_earth_m3_s2 and spin_rate_rad_s must be > 0")
    return float((gm_earth_m3_s2 / (spin_rate_rad_s * spin_rate_rad_s)) ** (1.0 / 3.0))


@dataclass(frozen=True, eq=False)
class GeostationaryOrbit:
    """The resolved geostationary preset (module docstring).

    Attributes:
        longitude_rad: Sub-satellite east longitude in IAU_EARTH at the epoch [rad].
        epoch_tdb_s: Epoch of the state [TDB s past J2000].
        gm_earth_m3_s2: GM of the Earth used [m^3 s^-2].
        angular_velocity_j2000_rad_s: Angular velocity of the IAU_EARTH frame, J2000 components,
            shape (3,) [rad s^-1].
        pole_unit_j2000: IAU_EARTH +z axis in J2000 at the epoch, shape (3,).
        spin_rate_rad_s: ``omega . pole`` (mean motion of the orbit) [rad s^-1].
        radius_m: Orbit radius ``(GM/omega^2)^(1/3)`` [m].
        period_s: ``2 pi / omega`` [s].
        initial_state: J2000 / EARTH state at the epoch.
        source_ids: Earth-orientation data ids (PCK).
    """

    longitude_rad: float
    epoch_tdb_s: float
    gm_earth_m3_s2: float
    angular_velocity_j2000_rad_s: Float64Array
    pole_unit_j2000: Float64Array
    spin_rate_rad_s: float
    radius_m: float
    period_s: float
    initial_state: CartesianState
    source_ids: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        for name in ("angular_velocity_j2000_rad_s", "pole_unit_j2000"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if arr.shape != (3,):
                raise ValueError(f"{name} must have shape (3,)")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if self.initial_state.origin != EARTH_CENTER or self.initial_state.frame != "J2000":
            raise ValueError("initial_state must be J2000 / EARTH")


def geostationary_orbit(
    longitude_rad: float,
    *,
    epoch_tdb_s: float,
    ephemeris: EphemerisProvider,
    earth_orientation: EarthOrientation,
) -> GeostationaryOrbit:
    """Geostationary preset state co-rotating with the Earth-fixed frame (module docstring).

    Args:
        longitude_rad: Sub-satellite east longitude in the Earth-fixed frame [rad].
        epoch_tdb_s: Epoch [TDB s past J2000].
        ephemeris: Provider of ``GM_earth`` (``gm_m3_s2("EARTH")``).
        earth_orientation: Earth-fixed -> J2000 model (SPICE ``IAU_EARTH`` in runs).

    Returns:
        :class:`GeostationaryOrbit`.
    """
    rot = earth_orientation.rotation(np.array([float(epoch_tdb_s)]))
    r_mat = np.asarray(rot.matrix[0], dtype=np.float64)
    omega_vec = angular_velocity_from_rotation_rad_s(r_mat, rot.matrix_rate_per_s[0])
    pole = r_mat[:, 2] / float(np.linalg.norm(r_mat[:, 2]))
    spin = float(np.dot(omega_vec, pole))
    gm = float(ephemeris.gm_m3_s2(EARTH_CENTER))
    radius = geostationary_radius_m(gm, spin)
    lon = float(longitude_rad)
    pos = radius * (r_mat @ np.array([math.cos(lon), math.sin(lon), 0.0]))
    vel = spin * np.cross(pole, pos)
    state = CartesianState(
        epoch_tdb_s=float(epoch_tdb_s),
        position_m=pos,
        velocity_m_s=vel,
        frame="J2000",
        origin=EARTH_CENTER,
    )
    return GeostationaryOrbit(
        longitude_rad=lon,
        epoch_tdb_s=float(epoch_tdb_s),
        gm_earth_m3_s2=gm,
        angular_velocity_j2000_rad_s=omega_vec,
        pole_unit_j2000=pole,
        spin_rate_rad_s=spin,
        radius_m=radius,
        period_s=2.0 * math.pi / spin,
        initial_state=state,
        source_ids=tuple(earth_orientation.source_ids),
    )


def geocentric_elements(
    orbit: OrbitConfig,
    *,
    initial_epoch_tdb_s: float,
    ephemeris: EphemerisProvider,
    earth_orientation: EarthOrientation | None,
) -> OrbitalElements:
    """Osculating Earth-centred J2000 elements of a configured geocentric orbit.

    Configured elements (``elements_center = EARTH``, ``elements_frame = J2000``) are used with
    ``mu = GM_earth`` of the provider; the geostationary preset is resolved with
    :func:`geostationary_orbit` and converted with :func:`helioswarm.orbit.state_to_elements`.

    Raises:
        ValueError: not an Earth-centred orbit, or the preset without an Earth-orientation
            model.
    """
    if orbit.elements_center != EARTH_CENTER:
        raise ValueError("geocentric_elements needs orbit.elements_center = 'EARTH'")
    gm = float(ephemeris.gm_m3_s2(EARTH_CENTER))
    if orbit.geostationary_longitude_deg is None:
        return elements_from_config(orbit, epoch_tdb_s=initial_epoch_tdb_s, mu_m3_s2=gm)
    if earth_orientation is None:
        raise ValueError("the geostationary preset needs an Earth-orientation model")
    geo = geostationary_orbit(
        orbit.geostationary_longitude_rad,
        epoch_tdb_s=initial_epoch_tdb_s,
        ephemeris=ephemeris,
        earth_orientation=earth_orientation,
    )
    return state_to_elements(geo.initial_state, gm)


def geocentric_l0_trajectory(
    orbit: OrbitConfig,
    epochs_tdb_s: ArrayLike,
    *,
    initial_epoch_tdb_s: float,
    ephemeris: EphemerisProvider,
    earth_orientation: EarthOrientation | None,
) -> CartesianTrajectory:
    """L0 trajectory of a geocentric orbit in J2000 / SSB (module docstring).

    Closed-form Kepler propagation of :func:`geocentric_elements` about the Earth's centre
    (J2000 / EARTH), then ``r_SSB = r + r_earth(t)``, ``v_SSB = v + v_earth(t)`` with the
    Earth's (NAIF 399) ephemeris state at each epoch.

    Args:
        orbit: Orbit section (``elements_center = EARTH``).
        epochs_tdb_s: Output epochs [TDB s].
        initial_epoch_tdb_s: Osculation epoch [TDB s].
        ephemeris: Provider of GM_earth and the Earth's state.
        earth_orientation: Earth-fixed -> J2000 model (needed by the geostationary preset).

    Returns:
        J2000 / SSB trajectory [m, m s^-1].
    """
    elements = geocentric_elements(
        orbit,
        initial_epoch_tdb_s=initial_epoch_tdb_s,
        ephemeris=ephemeris,
        earth_orientation=earth_orientation,
    )
    geo = propagate_kepler(elements, epochs_tdb_s).trajectory
    return shift_trajectory_origin(geo, SSB, ephemeris)


# ---------------------------------------------------------------------------------------------
# Perturbation budget (comparison path; not used by runs)
# ---------------------------------------------------------------------------------------------


def third_body_inclination_rate_vector(
    orbit_normal_unit: ArrayLike,
    mean_motion_rad_s: float,
    body_positions_m: ArrayLike,
    gm_m3_s2: float,
) -> Float64Array:
    """Orbit-averaged rate of the unit orbit normal of a circular orbit under a third body.

    ``dh/dt = -(3 GM / (2 n |s|^5)) (h . s)(h x s)`` per sample (quadrupole tidal torque on
    the orbit-averaged ring divided by the orbital angular momentum ``n a^2``; averaging over a
    circular third-body orbit gives ``-(3/4)(GM/(n a_k^3)) cos I (h_k x h)``, the secular
    precession of h about the third body's orbit normal; Murray & Dermott 1999, ch. 6-7).

    Args:
        orbit_normal_unit: Unit normal ``h`` of the satellite orbit, shape (3,).
        mean_motion_rad_s: Satellite mean motion ``n`` [rad s^-1].
        body_positions_m: Third-body positions relative to the central body, shape (N, 3) [m].
        gm_m3_s2: Third-body GM [m^3 s^-2].

    Returns:
        Shape (N, 3) [rad s^-1]; average over a full period of the third body for the secular
        rate.
    """
    h = np.asarray(orbit_normal_unit, dtype=np.float64)
    s = np.atleast_2d(np.asarray(body_positions_m, dtype=np.float64))
    d = np.linalg.norm(s, axis=1)
    hs = s @ h
    coef = -3.0 * gm_m3_s2 / (2.0 * mean_motion_rad_s * d**5) * hs
    return np.asarray(coef[:, None] * np.cross(h, s), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class GeocentricPropagation:
    """Numerical geocentric propagation with third bodies (perturbation budget).

    Attributes:
        trajectory: J2000 / EARTH states on the requested epochs.
        bodies: Third bodies included.
        nfev: Right-hand-side evaluations.
        rtol: Relative tolerance used.
        cache_step_s: Hermite cache node spacing [s].
    """

    trajectory: CartesianTrajectory
    bodies: tuple[BodyName, ...]
    nfev: int
    rtol: float
    cache_step_s: float


def propagate_geocentric_point_masses(
    initial_state: CartesianState,
    epochs_tdb_s: ArrayLike,
    *,
    ephemeris: EphemerisProvider,
    bodies: Sequence[BodyName] = THIRD_BODIES,
    rtol: float = 1.0e-11,
    atol_position_m: float = 1.0e-4,
    atol_velocity_m_s: float = 1.0e-8,
    cache_step_s: float = 3600.0,
) -> GeocentricPropagation:
    """Integrate a geocentric orbit: Earth point mass + third-body point masses (DOP853).

    ``d^2r/dt^2 = -GM_E r/|r|^3 + sum_k GM_k [(s_k - r)/|s_k - r|^3 - s_k/|s_k|^3]`` with
    ``r`` and ``s_k = r_k - r_E`` in J2000 (the indirect term ``-s_k/|s_k|^3`` is the Earth's own
    acceleration towards body k; Murray & Dermott 1999, ch. 6). Forward in time from the
    initial epoch only. Comparison path for the omitted lunisolar perturbation of the L0 GEO
    orbit; it is not a run model.

    Args:
        initial_state: J2000 / EARTH state.
        epochs_tdb_s: Output epochs, all >= the initial epoch [TDB s].
        ephemeris: Provider of the body states and GM values.
        bodies: Third bodies (default Sun and Moon).
        rtol: Relative tolerance.
        atol_position_m: Absolute tolerance on position components [m].
        atol_velocity_m_s: Absolute tolerance on velocity components [m s^-1].
        cache_step_s: Hermite cache spacing of the body positions [s].

    Returns:
        :class:`GeocentricPropagation`.
    """
    if initial_state.origin != EARTH_CENTER or initial_state.frame != "J2000":
        raise ValueError("initial_state must be J2000 / EARTH")
    t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
    t0 = initial_state.epoch_tdb_s
    if np.any(t < t0) or np.any(np.diff(t) <= 0.0):
        raise ValueError("epochs must be increasing and not before the initial epoch")
    names: tuple[BodyName, ...] = (EARTH_CENTER, *tuple(bodies))
    cache = HermiteEphemerisCache.build(ephemeris, names, t0, float(t[-1]), cache_step_s)
    gm_e = float(ephemeris.gm_m3_s2(EARTH_CENTER))
    gms = np.array([float(ephemeris.gm_m3_s2(b)) for b in bodies], dtype=np.float64)

    def rhs(time_s: float, y: Float64Array) -> Float64Array:
        p = cache.positions_at(time_s)
        s = p[1:] - p[0]
        r = y[:3]
        acc = -gm_e * r / float(np.linalg.norm(r)) ** 3
        d = s - r
        acc = acc + np.sum(
            gms[:, None]
            * (
                d / np.linalg.norm(d, axis=1)[:, None] ** 3
                - s / np.linalg.norm(s, axis=1)[:, None] ** 3
            ),
            axis=0,
        )
        return np.concatenate([y[3:], acc])

    y0 = np.concatenate([initial_state.position_m, initial_state.velocity_m_s])
    atol = np.array([atol_position_m] * 3 + [atol_velocity_m_s] * 3)
    sol = solve_ivp(
        rhs,
        (t0, float(t[-1])),
        y0,
        method="DOP853",
        t_eval=t,
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"geocentric integration failed: {sol.message}")
    traj = CartesianTrajectory(
        epochs_tdb_s=t,
        position_m=sol.y[:3].T,
        velocity_m_s=sol.y[3:].T,
        frame="J2000",
        origin=EARTH_CENTER,
    )
    return GeocentricPropagation(
        trajectory=traj,
        bodies=tuple(bodies),
        nfev=int(sol.nfev),
        rtol=float(rtol),
        cache_step_s=float(cache_step_s),
    )


def tilt_from_pole_rad(trajectory: CartesianTrajectory, pole_unit: ArrayLike) -> Float64Array:
    """Angle between the osculating orbit normal ``r x v`` and a reference pole [rad].

    With the IAU_EARTH pole of the element epoch this is the inclination of the osculating
    orbit to the initial equatorial plane (0 for the unperturbed geostationary orbit).
    """
    if trajectory.origin != EARTH_CENTER:
        raise ValueError("needs a geocentric (EARTH-origin) trajectory")
    h = np.cross(trajectory.position_m, trajectory.velocity_m_s)
    h /= np.linalg.norm(h, axis=1)[:, None]
    p = np.asarray(pole_unit, dtype=np.float64)
    p = p / float(np.linalg.norm(p))
    c = np.clip(h @ p, -1.0, 1.0)
    s = np.linalg.norm(np.cross(h, p), axis=1)
    return np.asarray(np.arctan2(s, c), dtype=np.float64)


def j2_synchronous_radius_offset_m(
    gm_earth_m3_s2: float, spin_rate_rad_s: float, j2: float, equatorial_radius_m: float
) -> float:
    """Increase of the synchronous radius due to J2 [m] (equatorial circular orbit).

    Synchronism with the J2 radial acceleration: ``omega^2 r = (GM/r^2)(1 + (3/2) J2 (R/r)^2)``,
    solved by fixed-point iteration from the point-mass radius ``r0``; returns ``r - r0``
    (``~ (1/2) J2 (R/r0)^2 r0`` to first order).
    """
    r0 = geostationary_radius_m(gm_earth_m3_s2, spin_rate_rad_s)
    r = r0
    for _ in range(50):
        r_new = r0 * (1.0 + 1.5 * j2 * (equatorial_radius_m / r) ** 2) ** (1.0 / 3.0)
        if abs(r_new - r) < 1e-9:
            r = r_new
            break
        r = r_new
    return float(r - r0)


def srp_eccentricity_circle_radius(
    srp_acceleration_m_s2: float, mean_motion_rad_s: float, radius_m: float, sun_rate_rad_s: float
) -> float:
    """Radius of the natural eccentricity circle of a GEO orbit under solar radiation pressure.

    Orbit-averaged Gauss equations for a constant-direction (Sun-pointing) acceleration ``a``
    on a near-circular orbit give ``|de/dt| = (3/2) a / (n r)``, perpendicular to the Sun
    direction; as the Sun's direction turns at ``n_sun`` the eccentricity vector describes a
    circle of radius ``e_r = (3/2) a / (n r n_sun)`` (no J2/lunisolar coupling). The
    corresponding radial excursion is ``+- e_r r`` and the longitude libration ``+- 2 e_r``.
    """
    for v in (srp_acceleration_m_s2, mean_motion_rad_s, radius_m, sun_rate_rad_s):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError("arguments must be finite and > 0")
    return 1.5 * srp_acceleration_m_s2 / (mean_motion_rad_s * radius_m * sun_rate_rad_s)


__all__ = [
    "EARTH_CENTER",
    "THIRD_BODIES",
    "GeocentricPropagation",
    "GeostationaryOrbit",
    "angular_velocity_from_rotation_rad_s",
    "geocentric_elements",
    "geocentric_l0_trajectory",
    "geostationary_orbit",
    "geostationary_radius_m",
    "j2_synchronous_radius_offset_m",
    "propagate_geocentric_point_masses",
    "srp_eccentricity_circle_radius",
    "third_body_inclination_rate_vector",
    "tilt_from_pole_rad",
]
