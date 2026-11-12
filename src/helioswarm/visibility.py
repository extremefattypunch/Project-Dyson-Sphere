"""Access geometry: retarded-time targeting, pointing, elevation, occultation and illumination.

Everything here works on J2000 / SSB states at TDB epochs (PDF §3) and is vectorised over the
output time grid. The grid epochs are **emission** epochs ``t_e`` of the satellite.

Retarded-time targeting (PDF §7.1)
    The light time ``tau`` solves ``c tau = |r_site(t_e + tau) - r_sat(t_e)|``. It is found by
    the fixed-point iteration of the PDF (``tau_{k+1} = |r_site(t_e + tau_k) - r_sat|/c``, a
    contraction with factor ``<= |v_site|/c ~ 1e-4``) with a tolerance and an iteration cap;
    epochs that do not converge are solved with a bracketed root finder (``scipy.optimize.
    brentq``) and flagged. Both the emission epoch and the arrival epoch ``t_a = t_e + tau`` are
    stored. Light time is Newtonian (straight line at speed c in the barycentric frame); the
    relativistic (Shapiro) delay and the gravitational bending of the path are neglected (their
    sizes for the MVP geometry are given in docs/model_assumptions.md, M3).
    With ``light_time`` disabled the *instantaneous geometry approximation* is used:
    ``t_a = t_e`` (the site at the emission epoch), and ``light_time_s = range/c`` is reported
    but not applied.

Pointing (beam direction)
    ``u = (r_site(t_a) - r_sat(t_e)) / |...|`` is the geometric (barycentric-frame) direction
    of the photon path; it already contains the point-ahead due to the *target's* motion. The
    transmitter must also correct for its *own* velocity (aberration): in the transmitter's
    instantaneous rest frame (axes parallel to J2000) the emission direction is the special-
    relativistic aberration of ``u`` with ``beta = v_sat/c`` (e.g. Rindler, *Relativity*, 2006,
    §4.4)::

        u' = [u + (gamma - 1)(u . b) b - gamma beta] / [gamma (1 - beta . u)],  b = beta/|beta|

    ``|u' - u| ~ |beta| sin(angle(u, v_sat)) <= |v_sat|/c ~ 1e-4 rad`` at 1 au, i.e. ~1000
    optical beam widths (lambda/D = 1e-7 rad), so it is implemented and reported
    (``transmit_direction_tx_frame_unit``, ``transmitter_aberration_rad``), not neglected.
    The angular rate of ``u`` is analytic: with ``tau' = d tau/d t_e = u.(v_site - v_sat) /
    (c - u.v_site)`` (differentiate ``c tau = |rho|``) and ``rho' = v_site (1 + tau') - v_sat``,
    ``du/dt_e = (I - u u^T) rho' / |rho|``; ``pointing_rate_rad_s = |du/dt_e|``.

Receiver geometry
    Directions at the receiver are *apparent* directions in the receiver's rest frame: the
    incoming photon direction ``u`` is aberrated with the site velocity ``v_site(t_a)`` (orbital
    + rotational, |v|/c ~ 1e-4 rad = 20"), ``s = -aberrated_direction(u, v_site)``. This keeps
    the elevation consistent with the Earth-blockage test, which follows the photon path in the
    Earth's frame (the two then differ only by the rotational speed, 465 m/s / c = 1.6e-6 rad).
    Atmospheric refraction (~0.1 deg at 10 deg elevation) is not applied. Elevation
    ``atan2(s . up, |s - (s . up) up|)`` with the geodetic up vector at ``t_a``; azimuth
    ``atan2(s . east, s . north)`` from north towards east in [0, 2 pi). The Sun separation is
    the angle between ``s`` and the apparent direction of the Sun's centre (light from the Sun
    at its retarded epoch ``t_a - tau_sun``, two fixed-point steps, aberrated the same way).

Occultation of the beam segment
    The beam is the straight segment from ``r_sat(t_e)`` to ``r_site(t_a)``. For an occulter B
    moving uniformly, the photon path relative to B is exactly the straight segment from
    ``p0 = r_sat(t_e) - r_B(t_e)`` to ``p1 = r_site(t_a) - r_B(t_a)`` (Galilean change to B's
    rest frame), so each body is tested in its own comoving frame with its positions at the two
    end epochs; the error from B's acceleration ``a_B`` is <= ``|a_B| tau^2 / 8`` (Sun:
    ~0.03 m; Moon: <= 9e-3 m s^-2 * (500 s)^2/8 ~ 0.3 km << R_moon).

    * Earth: segment-ellipsoid intersection in the Earth-fixed frame at ``t_a``:
      ``p0_ef = R(t_a)^T (r_sat(t_e) - r_E(t_e))``, ``p1_ef = r_site,ef``; scaling x, y by 1/a
      and z by 1/b maps the ellipsoid to the unit sphere (an affine map preserves segments and
      the interior), so the test is ``min_{s in [0, s_max]} |p0' + s (p1' - p0')| < 1``. The
      last ``earth_standoff_m`` of the segment (next to the receiver, which lies on the
      surface) are excluded: ``s_max = 1 - standoff/L``. The Earth's rotation during the last
      ``2a/c ~ 43 ms`` of the path (<= 20 m at the surface) is neglected. Sites below the
      ellipsoid (h < 0) are tested against the WGS 84 ellipsoid scaled by
      ``k = sqrt((x^2 + y^2)/a^2 + z^2/b^2) < 1`` (``r_site,ef = (x, y, z)``), the similar
      ellipsoid that passes exactly through the site, so the site lies on the tested surface as
      for h = 0. (Lowering both axes by |h| instead leaves the site ~1 mm inside the surface at
      h = -1000 m, which blocks every ray within ~1 mm / standoff ~ 1e-3 rad above the
      horizon.) Its normal at the site differs from the geodetic up vector by <= ~e^2 |h| / a
      (5e-7 rad at h = -1000 m), so blockage and apparent elevation < 0 agree to that level.
    * Sun: blocked if the segment comes closer to the Sun's centre than
      ``R_block = d sin(min(asin(R_sun/d) + margin, pi/2))``, ``d = |p1|`` the receiver-Sun
      distance, ``margin = models.sun_occultation_limb_margin_deg``. For a satellite beyond the
      Sun this is exactly "the satellite is within (apparent solar radius + margin) of the
      Sun's centre as seen from the receiver" (the configured meaning of the margin); with
      ``margin = 0`` it is the physical solar sphere (IAU 2015 nominal radius).
    * Moon: blocked if the segment comes closer to the Moon's centre than ``R_moon``.

Satellite illumination (conical shadow with penumbra)
    Seen from the satellite at ``t_e``, the Sun and each occulter (Earth, Moon) are uniform
    disks of apparent radii ``asin(R/d)`` at their light-time-retarded positions, separated by
    the angle between their directions. The occulted fraction of the solar disk is the
    circle-circle overlap area (planar formula on the sky; e.g. Montenbruck & Gill, *Satellite
    Orbits*, 2000, §3.4.2) divided by ``pi r_sun^2``: 0 when the disks are apart, total in the
    umbra, ``(r_B/r_sun)^2`` when annular. Only occulters nearer than the Sun count. No limb
    darkening; small-angle (planar) error O(r^2) ~ 2e-5 relative at 1 au. When two occulters
    overlap the solar disk at the same epoch their fractions are summed (an upper bound on the
    occultation if their disks also overlap each other) and the epoch is flagged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import spiceypy
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from helioswarm.config import RunConfig
from helioswarm.constants import (
    M_PER_KM,
    MOON_RADIUS_M,
    SOLAR_RADIUS_NOMINAL_M,
    SPEED_OF_LIGHT_M_S,
    WGS84_SEMIMAJOR_AXIS_M,
    WGS84_SEMIMINOR_AXIS_M,
)
from helioswarm.ephemeris import SSB, BodyName, EphemerisProvider
from helioswarm.frames import (
    CartesianTrajectory,
    FrameMismatchError,
    FrameName,
    OriginName,
    TimeScale,
)
from helioswarm.ground import (
    EarthOrientation,
    GroundSite,
    SiteTrajectory,
    SpiceEarthOrientation,
    site_trajectory,
)
from helioswarm.kernels import LoadedKernels
from helioswarm.propagate import check_kernel_set

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

OccultingBody = Literal["EARTH", "SUN", "MOON"]
"""Bodies that can block the beam segment."""

LightTimeMethod = Literal["fixed_point", "brentq", "instantaneous"]
"""How the light time of an epoch was obtained."""

ILLUMINATION_OCCULTERS: Final[tuple[BodyName, ...]] = ("EARTH", "MOON")
"""Default bodies that can shadow the satellite."""

_C: Final[float] = SPEED_OF_LIGHT_M_S


def _frozen(
    values: ArrayLike, dtype: type | np.dtype[np.generic] = np.float64
) -> NDArray[np.generic]:
    arr = np.array(values, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def _norm(v: Float64Array) -> Float64Array:
    return np.asarray(np.sqrt(np.einsum("...i,...i->...", v, v)), dtype=np.float64)


def _dot(a: Float64Array, b: Float64Array) -> Float64Array:
    return np.asarray(np.einsum("...i,...i->...", a, b), dtype=np.float64)


def angle_between_rad(a: ArrayLike, b: ArrayLike) -> Float64Array:
    """Angle between vectors [rad], ``atan2(|a x b|, a . b)`` (accurate at small angles).

    Args:
        a: Shape (..., 3).
        b: Shape (..., 3).
    """
    a_ = np.asarray(a, dtype=np.float64)
    b_ = np.asarray(b, dtype=np.float64)
    return np.asarray(np.arctan2(_norm(np.cross(a_, b_)), _dot(a_, b_)), dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OcculterRadii:
    """Sizes of the occulting bodies [m].

    Attributes:
        earth_equatorial_radius_m: Earth ellipsoid semi-major axis a (default WGS 84).
        earth_polar_radius_m: Earth ellipsoid semi-minor axis b (default WGS 84).
        sun_radius_m: Solar radius (default IAU 2015 nominal, 695 700 km).
        moon_radius_m: Lunar mean radius (default IAU WGCCRE 2015, 1737.4 km).
        source: Where the Sun/Moon radii come from.
    """

    earth_equatorial_radius_m: float = WGS84_SEMIMAJOR_AXIS_M
    earth_polar_radius_m: float = WGS84_SEMIMINOR_AXIS_M
    sun_radius_m: float = SOLAR_RADIUS_NOMINAL_M
    moon_radius_m: float = MOON_RADIUS_M
    source: str = "helioswarm.constants"

    def __post_init__(self) -> None:
        for name in (
            "earth_equatorial_radius_m",
            "earth_polar_radius_m",
            "sun_radius_m",
            "moon_radius_m",
        ):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be finite and > 0")
        if self.earth_polar_radius_m > self.earth_equatorial_radius_m:
            raise ValueError("earth_polar_radius_m must be <= earth_equatorial_radius_m")

    @classmethod
    def from_kernels(cls, kernels: LoadedKernels) -> OcculterRadii:
        """Sun and Moon radii from the loaded text PCK (``BODY10_RADII``, ``BODY301_RADII``;
        km -> m, mean of the three axes); the Earth stays the configured WGS 84 ellipsoid."""
        kernels.require_active(kind="PCK")

        def mean_radius_m(body: str) -> float:
            _n, radii = spiceypy.bodvrd(body, "RADII", 3)
            return float(np.mean(np.asarray(radii, dtype=np.float64))) * M_PER_KM

        return cls(
            sun_radius_m=mean_radius_m("SUN"),
            moon_radius_m=mean_radius_m("MOON"),
            source=f"PCK {list(kernels.ids_of_kind('PCK'))} (BODY10_RADII, BODY301_RADII)",
        )


@dataclass(frozen=True)
class LightTimeSettings:
    """Retarded-time (light-time) solution settings (PDF §7.1).

    Attributes:
        enabled: ``False`` selects the instantaneous geometry approximation (``t_a = t_e``).
        tolerance_s: Stop when ``|tau_new - tau| < tolerance_s`` [s].
        max_iterations: Fixed-point iteration cap before the bracketed fallback.
    """

    enabled: bool = True
    tolerance_s: float = 1.0e-9
    max_iterations: int = 50

    def __post_init__(self) -> None:
        if not (math.isfinite(self.tolerance_s) and self.tolerance_s > 0.0):
            raise ValueError("tolerance_s must be finite and > 0")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")


@dataclass(frozen=True)
class AccessSettings:
    """Access constraints and occultation switches (PDF §6.4).

    Attributes:
        min_elevation_rad: Minimum elevation above the local (geodetic) horizon [rad].
        sun_avoidance_rad: Minimum Sun-receiver-satellite angle; 0 = no constraint [rad].
        sun_limb_margin_rad: Angular margin around the solar limb seen from the receiver [rad].
        earth_occultation: Test the beam segment against the Earth ellipsoid.
        sun_occultation: Test the beam segment against the Sun (+ margin).
        moon_occultation: Test the beam segment against the Moon.
        earth_standoff_m: Segment length next to the receiver excluded from the Earth test [m].
        light_time: Light-time settings.
        radii: Occulter sizes.
    """

    min_elevation_rad: float = 0.0
    sun_avoidance_rad: float = 0.0
    sun_limb_margin_rad: float = 0.0
    earth_occultation: bool = True
    sun_occultation: bool = True
    moon_occultation: bool = True
    earth_standoff_m: float = 1.0
    light_time: LightTimeSettings = field(default_factory=LightTimeSettings)
    radii: OcculterRadii = field(default_factory=OcculterRadii)

    def __post_init__(self) -> None:
        if not -0.5 * math.pi <= self.min_elevation_rad < 0.5 * math.pi:
            raise ValueError("min_elevation_rad must lie in [-pi/2, pi/2)")
        if not 0.0 <= self.sun_avoidance_rad < math.pi:
            raise ValueError("sun_avoidance_rad must lie in [0, pi)")
        if not 0.0 <= self.sun_limb_margin_rad <= 0.5 * math.pi:
            raise ValueError("sun_limb_margin_rad must lie in [0, pi/2]")
        if not (math.isfinite(self.earth_standoff_m) and self.earth_standoff_m > 0.0):
            raise ValueError("earth_standoff_m must be finite and > 0")


def access_settings_from_config(
    config: RunConfig, radii: OcculterRadii | None = None
) -> AccessSettings:
    """Access settings from ``receiver`` and ``models`` (degrees converted here)."""
    m = config.models
    return AccessSettings(
        min_elevation_rad=config.receiver.min_elevation_rad,
        sun_avoidance_rad=config.receiver.sun_avoidance_rad,
        sun_limb_margin_rad=math.radians(m.sun_occultation_limb_margin_deg),
        earth_occultation=m.earth_occultation,
        sun_occultation=m.sun_occultation,
        moon_occultation=m.moon_occultation,
        earth_standoff_m=m.earth_occultation_standoff_m,
        light_time=LightTimeSettings(
            enabled=m.light_time,
            tolerance_s=m.light_time_tolerance_s,
            max_iterations=m.light_time_max_iterations,
        ),
        radii=radii if radii is not None else OcculterRadii(),
    )


# ---------------------------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------------------------


def segment_min_distance(
    p0: ArrayLike, p1: ArrayLike, s_max: ArrayLike = 1.0
) -> tuple[Float64Array, Float64Array]:
    """Minimum distance from the origin to the segments ``p0 + s (p1 - p0)``, ``s in [0, s_max]``.

    ``s* = clip(-p0 . d / |d|^2, 0, s_max)`` with ``d = p1 - p0`` (closest point of a line,
    clipped to the segment); distance ``|p0 + s* d|``.

    Args:
        p0: Segment starts, shape (..., 3) (any length unit).
        p1: Segment ends, shape (..., 3).
        s_max: Upper end of the parameter range, broadcastable to (...), in (0, 1].

    Returns:
        ``(distance, s*)``, each shape (...).
    """
    a = np.asarray(p0, dtype=np.float64)
    d = np.asarray(p1, dtype=np.float64) - a
    dd = _dot(d, d)
    if np.any(dd <= 0.0):
        raise ValueError("degenerate segment (p0 == p1)")
    s = np.clip(-_dot(a, d) / dd, 0.0, np.asarray(s_max, dtype=np.float64))
    return _norm(a + s[..., None] * d), np.asarray(s, dtype=np.float64)


def segment_intersects_ellipsoid(
    p0: ArrayLike,
    p1: ArrayLike,
    equatorial_radius_m: float,
    polar_radius_m: float,
    s_max: ArrayLike = 1.0,
) -> BoolArray:
    """Whether segments ``p0 + s (p1 - p0)``, ``s in [0, s_max]``, enter an oblate spheroid.

    The spheroid ``(x^2 + y^2)/a^2 + z^2/b^2 = 1`` is centred at the origin with its symmetry
    axis along z (Earth-fixed coordinates). The affine scaling ``(x/a, y/a, z/b)`` maps it to the
    unit sphere and segments to segments, so the test is
    :func:`segment_min_distance` < 1 in scaled coordinates.

    Args:
        p0: Segment starts (Earth-fixed) [m], shape (..., 3).
        p1: Segment ends (Earth-fixed) [m], shape (..., 3).
        equatorial_radius_m: a [m].
        polar_radius_m: b [m].
        s_max: Upper end of the tested parameter range.

    Returns:
        Boolean array of shape (...).
    """
    scale = np.array([1.0 / equatorial_radius_m, 1.0 / equatorial_radius_m, 1.0 / polar_radius_m])
    dist, _s = segment_min_distance(
        np.asarray(p0, dtype=np.float64) * scale, np.asarray(p1, dtype=np.float64) * scale, s_max
    )
    return np.asarray(dist < 1.0, dtype=np.bool_)


def _surface_scale_through(
    point_m: Float64Array, equatorial_radius_m: float, polar_radius_m: float
) -> float:
    """``k = sqrt((x^2 + y^2)/a^2 + z^2/b^2)``: the spheroid with axes ``(k a, k b)`` passes
    through ``point_m`` (Earth-fixed) [1]."""
    x, y, z = (float(c) for c in point_m)
    return math.sqrt((x * x + y * y) / equatorial_radius_m**2 + z * z / polar_radius_m**2)


def aberrated_direction(direction_unit: ArrayLike, velocity_m_s: ArrayLike) -> Float64Array:
    """Photon propagation direction in a frame moving with ``v`` (special-relativistic aberration).

    ``u' = [u + (gamma - 1)(u . b) b - gamma beta] / [gamma (1 - beta . u)]`` with
    ``beta = v/c``, ``b = beta/|beta|`` (Lorentz transformation of the photon direction; e.g.
    Rindler, *Relativity*, 2006, §4.4). ``u`` is the photon direction in the barycentric frame.
    For a transmitter moving with ``v``, ``u'`` is where it must point (axes parallel to J2000);
    for a receiver moving with ``v``, ``-u'`` is the apparent direction of the source.
    ``|u'| = 1`` exactly; to first order ``u' = u - beta + (u . beta) u``.

    Args:
        direction_unit: Barycentric photon directions, shape (..., 3).
        velocity_m_s: Velocity of the moving frame relative to the barycentric frame, shape
            (..., 3) [m s^-1].

    Returns:
        Unit vectors, shape (..., 3).
    """
    u = np.asarray(direction_unit, dtype=np.float64)
    beta = np.asarray(velocity_m_s, dtype=np.float64) / _C
    b2 = _dot(beta, beta)
    if np.any(b2 >= 1.0):
        raise ValueError("transmitter speed must be < c")
    gamma = 1.0 / np.sqrt(1.0 - b2)
    # (gamma - 1)/beta^2 = gamma^2/(gamma + 1): no division by |beta| (beta = 0 allowed).
    k = gamma * gamma / (gamma + 1.0)
    ub = _dot(u, beta)
    num = u + (k * ub)[..., None] * beta - gamma[..., None] * beta
    den = gamma * (1.0 - ub)
    out = num / den[..., None]
    return np.asarray(out / _norm(out)[..., None], dtype=np.float64)


def circle_overlap_area(r1: ArrayLike, r2: ArrayLike, d: ArrayLike) -> Float64Array:
    """Overlap area of two circles of radii ``r1``, ``r2`` with centre distance ``d`` (plane).

    0 for ``d >= r1 + r2``; ``pi min(r1, r2)^2`` for ``d <= |r1 - r2|``; otherwise the lens
    ``r1^2 acos((d^2 + r1^2 - r2^2)/(2 d r1)) + r2^2 acos((d^2 + r2^2 - r1^2)/(2 d r2))
    - sqrt(k)/2``, ``k = (-d + r1 + r2)(d + r1 - r2)(d - r1 + r2)(d + r1 + r2)`` (e.g.
    Weisstein, "Circle-Circle Intersection", MathWorld; Montenbruck & Gill 2000, §3.4.2). The
    angles are evaluated as ``atan2(sqrt(k), d^2 + r1^2 - r2^2)`` (same value, well conditioned
    near tangency, where acos loses ~sqrt(eps)).

    Args:
        r1: Radius of the first circle (any unit, e.g. rad on the sky).
        r2: Radius of the second circle (same unit).
        d: Centre distance (same unit).

    Returns:
        Area (unit squared), broadcast shape.
    """
    r1_, r2_, d_ = np.broadcast_arrays(
        np.asarray(r1, dtype=np.float64),
        np.asarray(r2, dtype=np.float64),
        np.asarray(d, dtype=np.float64),
    )
    if np.any(r1_ < 0.0) or np.any(r2_ < 0.0) or np.any(d_ < 0.0):
        raise ValueError("radii and distance must be >= 0")
    out = np.zeros(r1_.shape, dtype=np.float64)
    inside = d_ <= np.abs(r1_ - r2_)
    out[inside] = math.pi * np.minimum(r1_, r2_)[inside] ** 2
    lens = (~inside) & (d_ < r1_ + r2_)
    if np.any(lens):
        a, b, c = r1_[lens], r2_[lens], d_[lens]
        # Half-angles of the chord seen from each centre, in the well-conditioned atan2 form
        # acos((d^2 + r1^2 - r2^2)/(2 d r1)) = atan2(sqrt(k), d^2 + r1^2 - r2^2) (sin = h/r1
        # with the half chord h = sqrt(k)/(2d)); acos is ill-conditioned near tangency.
        sk = np.sqrt(np.maximum((-c + a + b) * (c + a - b) * (c - a + b) * (c + a + b), 0.0))
        alpha = np.arctan2(sk, c * c + a * a - b * b)
        beta = np.arctan2(sk, c * c + b * b - a * a)
        out[lens] = a * a * alpha + b * b * beta - 0.5 * sk
    return out


def solar_disk_occulted_fraction(
    sun_angular_radius_rad: ArrayLike,
    occulter_angular_radius_rad: ArrayLike,
    separation_rad: ArrayLike,
) -> Float64Array:
    """Fraction of a uniform solar disk hidden by an occulting disk in front of it.

    ``overlap_area(r_sun, r_B, theta) / (pi r_sun^2)`` (:func:`circle_overlap_area`, applied
    to apparent angular radii; small-angle approximation). 0 in full sunlight, 1 in the umbra,
    ``(r_B/r_sun)^2`` in an annular eclipse.

    Args:
        sun_angular_radius_rad: Apparent solar radius [rad].
        occulter_angular_radius_rad: Apparent occulter radius [rad].
        separation_rad: Angle between the two centres [rad].
    """
    rs = np.asarray(sun_angular_radius_rad, dtype=np.float64)
    if np.any(rs <= 0.0):
        raise ValueError("sun_angular_radius_rad must be > 0")
    frac = circle_overlap_area(rs, occulter_angular_radius_rad, separation_rad) / (math.pi * rs**2)
    return np.asarray(np.clip(frac, 0.0, 1.0), dtype=np.float64)


def _retarded_positions(
    ephemeris: EphemerisProvider,
    body: BodyName,
    observer_m: Float64Array,
    epochs_tdb_s: Float64Array,
    *,
    iterations: int = 2,
) -> Float64Array:
    """Positions of ``body`` at ``t - |r_B - r_obs|/c`` (received-light geometry), J2000/SSB.

    Two fixed-point steps leave an epoch error of order ``(v_B/c)^2 tau`` (< 1e-5 s for the
    planets), i.e. sub-metre position errors.
    """
    t_b = epochs_tdb_s.copy()
    pos = ephemeris.positions(body, t_b)
    for _ in range(iterations):
        t_b = epochs_tdb_s - _norm(pos - observer_m) / _C
        pos = ephemeris.positions(body, t_b)
    return pos


# ---------------------------------------------------------------------------------------------
# Light time
# ---------------------------------------------------------------------------------------------


class LightTimeError(RuntimeError):
    """The light-time equation could not be solved at some epoch."""


@dataclass(frozen=True, eq=False)
class LightTimeSolution:
    """Retarded-time solution for one satellite and one site (PDF §7.1).

    Attributes:
        t_emit_tdb_s: Emission epochs (the satellite grid) [TDB s].
        t_arrival_tdb_s: Arrival epochs ``t_e + tau`` (``= t_e`` if instantaneous) [TDB s].
        light_time_s: ``tau`` [s] (``range/c``, not applied, if instantaneous).
        iterations: Fixed-point iterations used per epoch.
        method: ``fixed_point``, ``brentq`` (fallback) or ``instantaneous`` per epoch.
        residual_m: ``|c tau - |r_site(t_a) - r_sat(t_e)||`` [m] (0 if instantaneous).
        converged: Whether the solution met the tolerance (always true if instantaneous).
        site_at_arrival: Site states at ``t_arrival_tdb_s``.
        site_at_emission: Site states at ``t_emit_tdb_s``.
        settings: Settings used.
    """

    t_emit_tdb_s: Float64Array
    t_arrival_tdb_s: Float64Array
    light_time_s: Float64Array
    iterations: NDArray[np.int64]
    method: NDArray[np.str_]
    residual_m: Float64Array
    converged: BoolArray
    site_at_arrival: SiteTrajectory
    site_at_emission: SiteTrajectory
    settings: LightTimeSettings

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_emit_tdb_s", _frozen(self.t_emit_tdb_s))
        object.__setattr__(self, "t_arrival_tdb_s", _frozen(self.t_arrival_tdb_s))
        object.__setattr__(self, "light_time_s", _frozen(self.light_time_s))
        object.__setattr__(self, "iterations", _frozen(self.iterations, np.int64))
        object.__setattr__(self, "method", _frozen(self.method, np.str_))
        object.__setattr__(self, "residual_m", _frozen(self.residual_m))
        object.__setattr__(self, "converged", _frozen(self.converged, np.bool_))


def _check_satellite(satellite: CartesianTrajectory) -> None:
    if satellite.frame != "J2000" or satellite.origin != SSB:
        raise FrameMismatchError(
            f"satellite trajectory is in ({satellite.frame}, {satellite.origin}); access "
            "geometry needs J2000 / SOLAR SYSTEM BARYCENTER"
        )


def solve_light_time(
    satellite: CartesianTrajectory,
    site: GroundSite,
    *,
    ephemeris: EphemerisProvider,
    earth_orientation: EarthOrientation,
    settings: LightTimeSettings,
) -> LightTimeSolution:
    """Solve ``c tau = |r_site(t_e + tau) - r_sat(t_e)|`` at every satellite epoch.

    Fixed point (PDF §7.1): ``tau_0 = |r_site(t_e) - r_sat(t_e)|/c``,
    ``tau_{k+1} = |r_site(t_e + tau_k) - r_sat(t_e)|/c`` until ``|tau_{k+1} - tau_k| <
    tolerance_s`` (vectorised; converged epochs drop out). Epochs still unconverged after
    ``max_iterations`` are solved with ``brentq`` on ``f(tau) = |r_site(t_e + tau) - r_sat| -
    c tau`` (``f(0) >= 0``; the upper bracket doubles from ``2 tau_0`` until ``f < 0``, which
    exists whenever the site moves slower than light) and are flagged ``method = brentq``.

    Precision note: epochs ~1e9 s have a float64 spacing of ~1.2e-7 s, so ``t_a`` is
    represented to ~6e-8 s (~2 mm of site motion at 30 km/s); this does not limit the
    convergence test, which compares light times.

    Args:
        satellite: Satellite states at the emission epochs (J2000 / SSB).
        site: Receiver site.
        ephemeris: Provider of the Earth state.
        earth_orientation: Earth-fixed -> J2000 model.
        settings: Light-time settings.

    Returns:
        The solution, with diagnostics per epoch.

    Raises:
        LightTimeError: the bracketed fallback failed (e.g. a site moving at >= c).
    """
    _check_satellite(satellite)
    t_e = np.asarray(satellite.epochs_tdb_s, dtype=np.float64)
    r_sat = np.asarray(satellite.position_m, dtype=np.float64)
    n = t_e.size

    def site_at(t: Float64Array) -> SiteTrajectory:
        return site_trajectory(site, t, ephemeris=ephemeris, earth_orientation=earth_orientation)

    st_emit = site_at(t_e)
    tau = _norm(st_emit.position_m - r_sat) / _C
    if not settings.enabled:
        return LightTimeSolution(
            t_emit_tdb_s=t_e,
            t_arrival_tdb_s=t_e,
            light_time_s=tau,
            iterations=np.zeros(n, dtype=np.int64),
            method=np.full(n, "instantaneous"),
            residual_m=np.zeros(n),
            converged=np.ones(n, dtype=np.bool_),
            site_at_arrival=st_emit,
            site_at_emission=st_emit,
            settings=settings,
        )

    iterations = np.zeros(n, dtype=np.int64)
    converged = np.zeros(n, dtype=np.bool_)
    for _ in range(settings.max_iterations):
        idx = np.flatnonzero(~converged)
        if idx.size == 0:
            break
        st = site_at(t_e[idx] + tau[idx])
        tau_new = _norm(st.position_m - r_sat[idx]) / _C
        iterations[idx] += 1
        done = np.abs(tau_new - tau[idx]) < settings.tolerance_s
        tau[idx] = tau_new
        converged[idx[done]] = True

    method = np.where(converged, "fixed_point", "brentq")
    for k in np.flatnonzero(~converged):
        tau[k] = _brentq_light_time(
            float(t_e[k]),
            r_sat[k],
            tau0=float(_norm(st_emit.position_m[k] - r_sat[k]) / _C),
            site_at=site_at,
            tolerance_s=settings.tolerance_s,
        )
        converged[k] = True

    st_arr = site_at(t_e + tau)
    residual = np.abs(_C * tau - _norm(st_arr.position_m - r_sat))
    return LightTimeSolution(
        t_emit_tdb_s=t_e,
        t_arrival_tdb_s=t_e + tau,
        light_time_s=tau,
        iterations=iterations,
        method=method,
        residual_m=residual,
        converged=converged,
        site_at_arrival=st_arr,
        site_at_emission=st_emit,
        settings=settings,
    )


def _brentq_light_time(
    t_emit: float,
    r_sat: Float64Array,
    *,
    tau0: float,
    site_at: Callable[[Float64Array], SiteTrajectory],
    tolerance_s: float,
) -> float:
    """Bracketed light-time root (fallback of :func:`solve_light_time`)."""

    def f(tau: float) -> float:
        st = site_at(np.array([t_emit + tau]))
        return float(_norm(st.position_m[0] - r_sat)) - _C * tau

    if tau0 == 0.0:
        return 0.0
    hi = 2.0 * tau0
    for _ in range(64):
        if f(hi) < 0.0:
            break
        hi *= 2.0
    else:
        raise LightTimeError(
            f"no light-time bracket at t_emit={t_emit} TDB s (site receding at >= c?)"
        )
    root, info = brentq(f, 0.0, hi, xtol=tolerance_s, full_output=True)
    if not info.converged:
        raise LightTimeError(f"brentq did not converge at t_emit={t_emit} TDB s: {info.flag}")
    return float(root)


# ---------------------------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AccessResult:
    """Access geometry of one satellite to one receiver site over a time grid (PDF p.4, §7).

    Per-epoch arrays have shape (N,) (vectors (N, 3)); epoch k is the emission epoch of grid
    sample k. Directions are J2000 components. Checks that are disabled in the settings are not
    evaluated: their flags are ``False`` and their distances ``NaN``.

    Attributes:
        site_name: Receiver site.
        t_emit_tdb_s: Emission epochs [TDB s].
        t_arrival_tdb_s: Arrival epochs [TDB s].
        light_time_s: Emission-to-arrival interval [s].
        range_m: ``|r_site(t_a) - r_sat(t_e)|`` [m].
        range_rate_m_s: Relative radial velocity ``u . (v_site(t_a) - v_sat(t_e))``, positive
            when separating [m s^-1].
        light_time_rate: ``d tau / d t_e`` [1] (0 if instantaneous).
        transmit_direction_unit: Geometric beam direction ``u`` (barycentric frame), (N, 3).
        transmit_direction_tx_frame_unit: Aberration-corrected pointing direction in the
            transmitter's rest frame, (N, 3).
        transmitter_aberration_rad: Angle between the two directions above [rad].
        target_motion_point_ahead_rad: Angle between ``u`` and the direction to the site at
            the emission epoch (the retarded-time correction) [rad].
        pointing_rate_rad_s: ``|du/dt_e|`` (inertial) [rad s^-1].
        elevation_rad: Apparent satellite elevation above the geodetic horizon at ``t_a``
            (aberrated, no refraction) [rad].
        azimuth_rad: Apparent azimuth from north towards east, [0, 2 pi) [rad].
        sun_separation_rad: Apparent angle Sun-receiver-satellite at ``t_a`` [rad].
        earth_blocked: Beam segment enters the Earth ellipsoid.
        sun_blocked: Beam segment passes within the solar limb (+ margin).
        moon_blocked: Beam segment enters the Moon.
        below_min_elevation: ``elevation < min_elevation``.
        sun_avoidance_violation: ``sun_separation < sun_avoidance`` (never with 0).
        visible: No flag set.
        occulted_by: ``EARTH``, ``SUN``, ``MOON`` (first blocking body in that order) or ``""``.
        sun_beam_min_distance_m: Closest approach of the beam to the Sun's centre [m].
        sun_block_radius_m: ``R_block`` of the Sun test [m].
        moon_beam_min_distance_m: Closest approach of the beam to the Moon's centre [m].
        light_time: Light-time solution and its diagnostics.
        settings: Settings used.
        source_ids: Ephemeris and Earth-orientation data ids.
        frame: ``J2000``.
        origin: ``SOLAR SYSTEM BARYCENTER``.
        time_scale: ``TDB``.
    """

    site_name: str
    t_emit_tdb_s: Float64Array
    t_arrival_tdb_s: Float64Array
    light_time_s: Float64Array
    range_m: Float64Array
    range_rate_m_s: Float64Array
    light_time_rate: Float64Array
    transmit_direction_unit: Float64Array
    transmit_direction_tx_frame_unit: Float64Array
    transmitter_aberration_rad: Float64Array
    target_motion_point_ahead_rad: Float64Array
    pointing_rate_rad_s: Float64Array
    elevation_rad: Float64Array
    azimuth_rad: Float64Array
    sun_separation_rad: Float64Array
    earth_blocked: BoolArray
    sun_blocked: BoolArray
    moon_blocked: BoolArray
    below_min_elevation: BoolArray
    sun_avoidance_violation: BoolArray
    visible: BoolArray
    occulted_by: NDArray[np.str_]
    sun_beam_min_distance_m: Float64Array
    sun_block_radius_m: Float64Array
    moon_beam_min_distance_m: Float64Array
    light_time: LightTimeSolution
    settings: AccessSettings
    source_ids: tuple[str, ...]
    frame: FrameName = field(default="J2000")
    origin: OriginName = field(default=SSB)
    time_scale: TimeScale = field(default="TDB")

    def __post_init__(self) -> None:
        n = np.asarray(self.t_emit_tdb_s).shape[0]
        for name, val in list(vars(self).items()):
            if isinstance(val, np.ndarray):
                arr = _frozen(val, val.dtype.type)
                if arr.shape[0] != n:
                    raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
                object.__setattr__(self, name, arr)
        if self.frame != "J2000" or self.origin != SSB or self.time_scale != "TDB":
            raise ValueError("AccessResult is J2000 / SSB / TDB by contract")

    def __len__(self) -> int:
        return int(self.t_emit_tdb_s.shape[0])


def compute_access(
    satellite: CartesianTrajectory,
    site: GroundSite,
    *,
    ephemeris: EphemerisProvider,
    earth_orientation: EarthOrientation,
    settings: AccessSettings,
) -> AccessResult:
    """Access geometry of one satellite to one site at every satellite epoch (module docstring).

    Args:
        satellite: Satellite states at the emission epochs (J2000 / SSB).
        site: Receiver site.
        ephemeris: Provider of Earth, Sun (and Moon, if ``moon_occultation``) states.
        earth_orientation: Earth-fixed -> J2000 model.
        settings: Constraints and switches.

    Returns:
        :class:`AccessResult`.
    """
    _check_satellite(satellite)
    lt = solve_light_time(
        satellite,
        site,
        ephemeris=ephemeris,
        earth_orientation=earth_orientation,
        settings=settings.light_time,
    )
    t_e = lt.t_emit_tdb_s
    t_a = lt.t_arrival_tdb_s
    n = t_e.size
    r_sat = np.asarray(satellite.position_m)
    v_sat = np.asarray(satellite.velocity_m_s)
    st = lt.site_at_arrival
    r_site, v_site = st.position_m, st.velocity_m_s

    rho = r_site - r_sat
    rng = _norm(rho)
    u = rho / rng[:, None]
    if settings.light_time.enabled:
        tau_rate = _dot(u, v_site - v_sat) / (_C - _dot(u, v_site))
    else:
        tau_rate = np.zeros(n)
    drho = v_site * (1.0 + tau_rate)[:, None] - v_sat
    du = (drho - u * _dot(u, drho)[:, None]) / rng[:, None]
    pointing_rate = _norm(du)
    range_rate = _dot(u, v_site - v_sat)
    u_tx = aberrated_direction(u, v_sat)
    aberration = angle_between_rad(u, u_tx)
    point_ahead = angle_between_rad(u, lt.site_at_emission.position_m - r_sat)

    s = -aberrated_direction(u, v_site)  # apparent direction to the satellite at the receiver
    s_up = _dot(s, st.up_unit)
    horiz = s - s_up[:, None] * st.up_unit
    elevation = np.arctan2(s_up, _norm(horiz))
    azimuth = np.mod(np.arctan2(_dot(s, st.east_unit), _dot(s, st.north_unit)), 2.0 * math.pi)

    sun_photon = r_site - _retarded_positions(ephemeris, "SUN", r_site, t_a)
    sun_photon /= _norm(sun_photon)[:, None]
    sun_sep = angle_between_rad(s, -aberrated_direction(sun_photon, v_site))

    radii = settings.radii
    false = np.zeros(n, dtype=np.bool_)
    nan = np.full(n, np.nan)

    earth_blocked = false.copy()
    if settings.earth_occultation:
        earth_e = ephemeris.positions("EARTH", t_e)
        rel = r_sat - earth_e  # Earth-centred J2000 (Earth at its emission-epoch position)
        p0_ef = np.einsum("nji,nj->ni", st.rotation_matrix, rel)  # R^T rel
        p1_ef = np.broadcast_to(site.position_ef_m, p0_ef.shape)
        seg_len = _norm(p1_ef - p0_ef)
        s_max = np.clip(1.0 - settings.earth_standoff_m / seg_len, 0.0, 1.0)
        scale = (
            _surface_scale_through(
                site.position_ef_m, radii.earth_equatorial_radius_m, radii.earth_polar_radius_m
            )
            if site.altitude_m < 0.0
            else 1.0
        )
        earth_blocked = segment_intersects_ellipsoid(
            p0_ef,
            p1_ef,
            scale * radii.earth_equatorial_radius_m,
            scale * radii.earth_polar_radius_m,
            s_max,
        )

    sun_blocked = false.copy()
    sun_dmin, sun_rblock = nan.copy(), nan.copy()
    if settings.sun_occultation:
        p0 = r_sat - ephemeris.positions("SUN", t_e)
        p1 = r_site - ephemeris.positions("SUN", t_a)
        sun_dmin, _ = segment_min_distance(p0, p1)
        d_rs = _norm(p1)
        half = np.minimum(
            np.arcsin(radii.sun_radius_m / d_rs) + settings.sun_limb_margin_rad, 0.5 * math.pi
        )
        sun_rblock = d_rs * np.sin(half)
        sun_blocked = sun_dmin < sun_rblock

    moon_blocked = false.copy()
    moon_dmin = nan.copy()
    if settings.moon_occultation:
        p0 = r_sat - ephemeris.positions("MOON", t_e)
        p1 = r_site - ephemeris.positions("MOON", t_a)
        moon_dmin, _ = segment_min_distance(p0, p1)
        moon_blocked = moon_dmin < radii.moon_radius_m

    below = elevation < settings.min_elevation_rad
    avoid = sun_sep < settings.sun_avoidance_rad
    visible = ~(earth_blocked | sun_blocked | moon_blocked | below | avoid)
    occulted_by = np.where(
        earth_blocked, "EARTH", np.where(sun_blocked, "SUN", np.where(moon_blocked, "MOON", ""))
    )
    return AccessResult(
        site_name=site.name,
        t_emit_tdb_s=t_e,
        t_arrival_tdb_s=t_a,
        light_time_s=lt.light_time_s,
        range_m=rng,
        range_rate_m_s=range_rate,
        light_time_rate=tau_rate,
        transmit_direction_unit=u,
        transmit_direction_tx_frame_unit=u_tx,
        transmitter_aberration_rad=aberration,
        target_motion_point_ahead_rad=point_ahead,
        pointing_rate_rad_s=pointing_rate,
        elevation_rad=elevation,
        azimuth_rad=azimuth,
        sun_separation_rad=sun_sep,
        earth_blocked=earth_blocked,
        sun_blocked=sun_blocked,
        moon_blocked=moon_blocked,
        below_min_elevation=below,
        sun_avoidance_violation=avoid,
        visible=visible,
        occulted_by=occulted_by,
        sun_beam_min_distance_m=sun_dmin,
        sun_block_radius_m=sun_rblock,
        moon_beam_min_distance_m=moon_dmin,
        light_time=lt,
        settings=settings,
        source_ids=tuple(ephemeris.source_kernel_ids) + tuple(earth_orientation.source_ids),
    )


def compute_access_from_config(
    config: RunConfig,
    satellite: CartesianTrajectory,
    *,
    ephemeris: EphemerisProvider,
    kernels: LoadedKernels,
) -> tuple[AccessResult, ...]:
    """:func:`compute_access` for every configured receiver site (SPICE IAU_EARTH, PCK radii).

    Each site is computed independently (PDF §6.4). Sun and Moon radii are read from the loaded
    PCK (:meth:`OcculterRadii.from_kernels`); the Earth is the configured WGS 84 ellipsoid.

    Raises:
        KernelSetMismatchError: ``kernels`` is not ``frames.ephemeris_kernel_set``.
    """
    check_kernel_set(config, kernels)
    settings = access_settings_from_config(config, OcculterRadii.from_kernels(kernels))
    orientation = SpiceEarthOrientation(kernels)
    return tuple(
        compute_access(
            satellite,
            GroundSite.from_config(s),
            ephemeris=ephemeris,
            earth_orientation=orientation,
            settings=settings,
        )
        for s in config.receiver.sites
    )


# ---------------------------------------------------------------------------------------------
# Satellite illumination
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class IlluminationResult:
    """Visible fraction of the solar disk seen from the satellite (conical shadow, penumbra).

    Attributes:
        epochs_tdb_s: Satellite epochs [TDB s].
        illumination_fraction: Visible fraction of the solar disk, in [0, 1].
        sun_distance_m: Distance satellite - Sun centre (retarded Sun) [m].
        sun_direction_unit: Unit vector satellite -> Sun centre (retarded Sun, geometric, no
            aberration), J2000, shape (N, 3). Used by ``helioswarm.solar`` for the incidence
            angle of a fixed-attitude collector.
        sun_angular_radius_rad: Apparent solar radius [rad].
        occulters: Bodies tested, in the order of ``occulted_fraction``'s columns.
        occulted_fraction: Fraction of the solar disk hidden by each occulter, (N, n_occ).
        occulted_by: Body hiding the largest fraction, or ``""``.
        multiple_occulters: More than one occulter overlaps the solar disk (fractions summed).
        radii: Body sizes used.
        source_ids: Ephemeris data ids (kernels) the Sun/occulter positions come from.
        time_scale: Time scale of ``epochs_tdb_s`` (``TDB``).
        frame: Inertial frame of ``sun_direction_unit`` (``J2000``).
    """

    epochs_tdb_s: Float64Array
    illumination_fraction: Float64Array
    sun_distance_m: Float64Array
    sun_direction_unit: Float64Array
    sun_angular_radius_rad: Float64Array
    occulters: tuple[BodyName, ...]
    occulted_fraction: Float64Array
    occulted_by: NDArray[np.str_]
    multiple_occulters: BoolArray
    radii: OcculterRadii
    source_ids: tuple[str, ...]
    time_scale: TimeScale = field(default="TDB")
    frame: FrameName = field(default="J2000")

    def __post_init__(self) -> None:
        if self.frame != "J2000":
            raise FrameMismatchError("IlluminationResult directions are J2000 by contract")
        if np.asarray(self.sun_direction_unit).shape != (np.asarray(self.epochs_tdb_s).size, 3):
            raise ValueError("sun_direction_unit must have shape (N, 3)")
        n = np.asarray(self.epochs_tdb_s).shape[0]
        for name, val in list(vars(self).items()):
            if isinstance(val, np.ndarray):
                arr = _frozen(val, val.dtype.type)
                if arr.shape[0] != n:
                    raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
                object.__setattr__(self, name, arr)
        object.__setattr__(self, "source_ids", tuple(str(x) for x in self.source_ids))
        if not self.source_ids:
            raise ValueError("IlluminationResult.source_ids must not be empty")
        if self.time_scale != "TDB":
            raise ValueError("IlluminationResult epochs are TDB by contract")


def satellite_illumination(
    satellite: CartesianTrajectory,
    *,
    ephemeris: EphemerisProvider,
    occulters: Sequence[BodyName] = ILLUMINATION_OCCULTERS,
    radii: OcculterRadii | None = None,
) -> IlluminationResult:
    """Fraction of the solar disk visible from the satellite at each epoch (module docstring).

    Sun and occulter positions are taken at their light-time-retarded epochs as seen from the
    satellite (an occulter's 30 km/s motion over ~250 s of light time is comparable to the
    Earth's radius, so the correction is not optional). Earth: sphere of the equatorial radius
    (conservative: the oblateness changes the shadow edge by <= 21 km); Moon: mean radius.

    Args:
        satellite: Satellite states (J2000 / SSB).
        ephemeris: Provider of the Sun and occulter states.
        occulters: Bodies that may shadow the satellite (``EARTH``, ``MOON``).
        radii: Body sizes (default :class:`OcculterRadii`).

    Returns:
        :class:`IlluminationResult`.
    """
    _check_satellite(satellite)
    radii = radii if radii is not None else OcculterRadii()
    body_radius = {"EARTH": radii.earth_equatorial_radius_m, "MOON": radii.moon_radius_m}
    for body in occulters:
        if body not in body_radius:
            raise ValueError(f"unsupported occulter {body!r}; supported: {sorted(body_radius)}")
    t = np.asarray(satellite.epochs_tdb_s, dtype=np.float64)
    r = np.asarray(satellite.position_m, dtype=np.float64)
    sun_vec = _retarded_positions(ephemeris, "SUN", r, t) - r
    d_sun = _norm(sun_vec)
    if np.any(d_sun <= radii.sun_radius_m):
        raise ValueError("satellite inside the Sun")
    a_sun = np.arcsin(radii.sun_radius_m / d_sun)
    frac = np.zeros((t.size, len(occulters)))
    for j, body in enumerate(occulters):
        vec = _retarded_positions(ephemeris, body, r, t) - r
        d = _norm(vec)
        if np.any(d <= body_radius[body]):
            raise ValueError(f"satellite inside {body}")
        a_b = np.arcsin(body_radius[body] / d)
        sep = angle_between_rad(sun_vec, vec)
        f = solar_disk_occulted_fraction(a_sun, a_b, sep)
        frac[:, j] = np.where(d < d_sun, f, 0.0)
    total = frac.sum(axis=1)
    names = np.array([*occulters, ""])
    idx = (
        np.where(total > 0.0, np.argmax(frac, axis=1), len(occulters))
        if occulters
        else np.zeros(t.size, dtype=np.int64)
    )
    return IlluminationResult(
        epochs_tdb_s=t,
        illumination_fraction=np.clip(1.0 - total, 0.0, 1.0),
        sun_distance_m=d_sun,
        sun_direction_unit=sun_vec / d_sun[:, None],
        sun_angular_radius_rad=a_sun,
        occulters=tuple(occulters),
        occulted_fraction=frac,
        occulted_by=names[idx],
        multiple_occulters=np.count_nonzero(frac > 0.0, axis=1) > 1,
        radii=radii,
        source_ids=tuple(ephemeris.source_kernel_ids),
    )


# ---------------------------------------------------------------------------------------------
# Orbital outputs and diagnostics (PDF §7.3)
# ---------------------------------------------------------------------------------------------


def orbit_drift_m(trajectory: CartesianTrajectory, reference: CartesianTrajectory) -> Float64Array:
    """``|r(t) - r_ref(t)|`` [m] at common epochs (PDF §7.3 "orbit drift").

    Raises:
        FrameMismatchError: different frames or origins.
        ValueError: different epochs.
    """
    if trajectory.frame != reference.frame or trajectory.origin != reference.origin:
        raise FrameMismatchError(
            f"cannot compare ({trajectory.frame}, {trajectory.origin}) with "
            f"({reference.frame}, {reference.origin})"
        )
    if trajectory.epochs_tdb_s.shape != reference.epochs_tdb_s.shape or not np.array_equal(
        trajectory.epochs_tdb_s, reference.epochs_tdb_s
    ):
        raise ValueError("trajectories must share their epochs")
    return _norm(np.asarray(trajectory.position_m) - np.asarray(reference.position_m))


def satellite_hour_angle_rad(access: AccessResult) -> Float64Array:
    """Local hour angle of the satellite at the receiver at each arrival epoch [rad, [0, 2 pi)].

    ``H = lambda_site - lambda_sat``, with ``lambda_sat = atan2(d_y, d_x)`` the Earth-fixed
    longitude of the geometric site-to-satellite direction ``d = R(t_a)^T (-u)`` (spherical
    astronomy: hour angle = local sidereal angle - right ascension, here measured in the
    Earth-fixed frame of the Earth-orientation model used). The geometric direction is used;
    aberration (<= 1e-4 rad) is irrelevant for a sampling diagnostic.

    Args:
        access: Access result (its light-time solution holds the rotation matrices at ``t_a``).

    Returns:
        Shape (N,) [rad].
    """
    st = access.light_time.site_at_arrival
    d_ef = np.einsum("nji,nj->ni", st.rotation_matrix, -access.transmit_direction_unit)
    lam = np.arctan2(d_ef[:, 1], d_ef[:, 0])
    return np.asarray(np.mod(st.site.longitude_rad - lam, 2.0 * math.pi), dtype=np.float64)


def max_circular_gap_rad(angles_rad: ArrayLike) -> float:
    """Largest gap between angles sorted on the circle (including the wrap-around) [rad].

    ``2 pi`` for a single sample (or all samples equal); ``2 pi / n`` for ``n`` equally spaced
    angles.
    """
    a = np.sort(np.mod(np.asarray(angles_rad, dtype=np.float64).ravel(), 2.0 * math.pi))
    if a.size == 0:
        raise ValueError("no angles")
    gaps = np.diff(np.concatenate([a, [a[0] + 2.0 * math.pi]]))
    return float(np.max(gaps))


@dataclass(frozen=True)
class AccessSummary:
    """Summary diagnostics of an :class:`AccessResult` time series (PDF §7.3).

    Fractions are fractions of grid samples; on the TDB-uniform output grid they estimate
    fractions of simulated time (each sample representing one output step). That estimate is
    only as good as the sampling of the *diurnal phase*: for a heliocentric satellite the
    satellite's hour angle at the site repeats with a period within minutes of the solar day,
    so a step that divides the day (e.g. 3600 s) samples nearly the same hour angles every day
    and the access, Earth-occultation and below-minimum-elevation fractions are quantised
    (MVP year at 3600 s: access fraction 0.4220 against the analytic 0.4420). The diagnostic
    ``hour_angle_max_gap_rad`` reports the largest unsampled interval of hour angle: a diurnal
    window's fraction is then resolved only to roughly ``gap / (2 pi)`` per window edge (an
    order-of-magnitude guide, not a bound, because samples need not be uniformly weighted).
    Use a step that is incommensurate with the day, or finer, when ``gap`` is not small.

    Attributes:
        site_name: Receiver site.
        n_epochs: Number of grid samples.
        span_s: Last minus first emission epoch [s].
        access_fraction: Fraction with ``visible``.
        occultation_fraction: Fraction blocked by the Earth, Sun or Moon.
        earth_occultation_fraction: Fraction with ``earth_blocked``.
        sun_occultation_fraction: Fraction with ``sun_blocked``.
        moon_occultation_fraction: Fraction with ``moon_blocked``.
        below_min_elevation_fraction: Fraction with ``below_min_elevation``.
        sun_avoidance_fraction: Fraction with ``sun_avoidance_violation``.
        range_min_m: Minimum Earth range [m].
        range_mean_m: Mean Earth range [m].
        range_max_m: Maximum Earth range [m].
        light_time_min_s: Minimum light time [s].
        light_time_mean_s: Mean light time [s].
        light_time_max_s: Maximum light time [s].
        range_rate_min_m_s: Minimum range rate [m s^-1].
        range_rate_max_m_s: Maximum range rate [m s^-1].
        pointing_rate_mean_rad_s: Mean inertial pointing rate [rad s^-1].
        pointing_rate_max_rad_s: Maximum inertial pointing rate [rad s^-1].
        transmitter_aberration_max_rad: Largest transmitter-velocity aberration [rad].
        target_motion_point_ahead_max_rad: Largest retarded-time point-ahead angle [rad].
        elevation_max_rad: Highest elevation [rad].
        light_time_max_iterations: Most fixed-point iterations at any epoch.
        light_time_fallback_count: Epochs solved by the brentq fallback.
        light_time_max_residual_m: Largest light-time residual [m].
        hour_angle_max_gap_rad: Largest gap between the sampled satellite hour angles at the
            site (:func:`satellite_hour_angle_rad`, :func:`max_circular_gap_rad`) [rad]; the
            diurnal-sampling diagnostic described above.
        orbit_drift_max_m: Max ``|r - r_L0|`` [m], or ``None`` if not supplied.
        orbit_drift_final_m: ``|r - r_L0|`` at the last epoch [m], or ``None``.
    """

    site_name: str
    n_epochs: int
    span_s: float
    access_fraction: float
    occultation_fraction: float
    earth_occultation_fraction: float
    sun_occultation_fraction: float
    moon_occultation_fraction: float
    below_min_elevation_fraction: float
    sun_avoidance_fraction: float
    range_min_m: float
    range_mean_m: float
    range_max_m: float
    light_time_min_s: float
    light_time_mean_s: float
    light_time_max_s: float
    range_rate_min_m_s: float
    range_rate_max_m_s: float
    pointing_rate_mean_rad_s: float
    pointing_rate_max_rad_s: float
    transmitter_aberration_max_rad: float
    target_motion_point_ahead_max_rad: float
    elevation_max_rad: float
    light_time_max_iterations: int
    light_time_fallback_count: int
    light_time_max_residual_m: float
    hour_angle_max_gap_rad: float
    orbit_drift_max_m: float | None = None
    orbit_drift_final_m: float | None = None


def summarize_access(
    access: AccessResult, *, orbit_drift: ArrayLike | None = None
) -> AccessSummary:
    """PDF §7.3 orbital outputs of one access time series.

    Args:
        access: The access result.
        orbit_drift: Optional ``|r - r_ref|`` per epoch [m] (:func:`orbit_drift_m`), same length.

    Returns:
        :class:`AccessSummary`.
    """
    a = access
    blocked = a.earth_blocked | a.sun_blocked | a.moon_blocked
    drift_max: float | None = None
    drift_final: float | None = None
    if orbit_drift is not None:
        drift = np.asarray(orbit_drift, dtype=np.float64)
        if drift.shape != a.t_emit_tdb_s.shape:
            raise ValueError("orbit_drift must have one value per access epoch")
        drift_max, drift_final = float(np.max(drift)), float(drift[-1])

    def frac(x: BoolArray) -> float:
        return float(np.mean(x))

    return AccessSummary(
        site_name=a.site_name,
        n_epochs=len(a),
        span_s=float(a.t_emit_tdb_s[-1] - a.t_emit_tdb_s[0]),
        access_fraction=frac(a.visible),
        occultation_fraction=frac(blocked),
        earth_occultation_fraction=frac(a.earth_blocked),
        sun_occultation_fraction=frac(a.sun_blocked),
        moon_occultation_fraction=frac(a.moon_blocked),
        below_min_elevation_fraction=frac(a.below_min_elevation),
        sun_avoidance_fraction=frac(a.sun_avoidance_violation),
        range_min_m=float(np.min(a.range_m)),
        range_mean_m=float(np.mean(a.range_m)),
        range_max_m=float(np.max(a.range_m)),
        light_time_min_s=float(np.min(a.light_time_s)),
        light_time_mean_s=float(np.mean(a.light_time_s)),
        light_time_max_s=float(np.max(a.light_time_s)),
        range_rate_min_m_s=float(np.min(a.range_rate_m_s)),
        range_rate_max_m_s=float(np.max(a.range_rate_m_s)),
        pointing_rate_mean_rad_s=float(np.mean(a.pointing_rate_rad_s)),
        pointing_rate_max_rad_s=float(np.max(a.pointing_rate_rad_s)),
        transmitter_aberration_max_rad=float(np.max(a.transmitter_aberration_rad)),
        target_motion_point_ahead_max_rad=float(np.max(a.target_motion_point_ahead_rad)),
        elevation_max_rad=float(np.max(a.elevation_rad)),
        light_time_max_iterations=int(np.max(a.light_time.iterations)),
        light_time_fallback_count=int(np.count_nonzero(a.light_time.method == "brentq")),
        light_time_max_residual_m=float(np.max(a.light_time.residual_m)),
        hour_angle_max_gap_rad=max_circular_gap_rad(satellite_hour_angle_rad(a)),
        orbit_drift_max_m=drift_max,
        orbit_drift_final_m=drift_final,
    )


__all__ = [
    "ILLUMINATION_OCCULTERS",
    "AccessResult",
    "AccessSettings",
    "AccessSummary",
    "IlluminationResult",
    "LightTimeError",
    "LightTimeMethod",
    "LightTimeSettings",
    "LightTimeSolution",
    "OcculterRadii",
    "OccultingBody",
    "aberrated_direction",
    "access_settings_from_config",
    "angle_between_rad",
    "circle_overlap_area",
    "compute_access",
    "compute_access_from_config",
    "max_circular_gap_rad",
    "orbit_drift_m",
    "satellite_hour_angle_rad",
    "satellite_illumination",
    "segment_intersects_ellipsoid",
    "segment_min_distance",
    "solar_disk_occulted_fraction",
    "solve_light_time",
    "summarize_access",
]
