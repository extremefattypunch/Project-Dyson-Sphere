"""Ground receiver sites: WGS 84 geodetic coordinates, Earth orientation and inertial site states.

Two kinds of coordinates are kept strictly apart (PDF §3, §7.2):

* **Earth-fixed** site coordinates (:class:`GroundSite`): geodetic latitude, longitude and
  height on the WGS 84 ellipsoid, the Earth-fixed Cartesian position and the local
  east/north/up unit vectors, all in the Earth-fixed frame ``IAU_EARTH``. They are constant.
* **Inertial** site states (:class:`SiteTrajectory`): position and velocity in ``J2000`` with
  origin SSB at explicit TDB epochs, obtained with an explicit Earth-orientation model.

Geodetic -> Earth-fixed Cartesian (WGS 84, NGA.STND.0036 §4; e.g. Hofmann-Wellenhof,
Lichtenegger & Wasle, *GNSS*, 2008, eq. 2.31)::

    N(phi) = a / sqrt(1 - e^2 sin^2 phi)          (prime-vertical radius of curvature)
    x = (N + h) cos phi cos lambda
    y = (N + h) cos phi sin lambda
    z = (N (1 - e^2) + h) sin phi

with ``e^2 = f (2 - f)``. The local *up* vector is the ellipsoid normal (geodetic vertical)
``u = (cos phi cos lambda, cos phi sin lambda, sin phi)``, *east* ``(-sin lambda, cos lambda, 0)``
and *north* ``(-sin phi cos lambda, -sin phi sin lambda, cos phi)``; ``east x north = up``.

Earth orientation (:class:`SpiceEarthOrientation`): the SPICE ``IAU_EARTH`` frame of the text
PCK ``pck00011.tpc``, i.e. the IAU WGCCRE 2009 model (Archinal et al. 2011; the 2015 report
gives no new Earth values): pole ``alpha0 = 0 - 0.641 T``, ``delta0 = 90 - 0.557 T`` [deg],
prime meridian ``W = 190.147 + 360.9856235 d`` [deg], with ``T`` (``d``) TDB Julian centuries
(days) past J2000. ``v_J2000 = R(t) v_IAU_EARTH`` with
``R = [R_z(W) R_x(90 deg - delta0) R_z(90 deg + alpha0)]^T`` (NAIF PCK Required Reading),
evaluated by ``sxform`` which also returns ``dR/dt``. It is a *documented approximation* of the
real Earth orientation:

* **Rotation angle (dominant).** The IAU angle of the prime meridian, ``90 deg + alpha0 + W``,
  is linear in TDB, while the Earth turns by the IERS Earth Rotation Angle of UT1
  (``ERA = 2 pi (0.7790572732640 + 1.00273781191135448 D_UT1)``, IERS Conventions 2010
  eq. 5.15). The IAU constants agree with ERA only for TT - UT1 = 75.06 s at J2000 (the actual
  value was 63.8 s: an offset of -0.047 deg = -8.2e-4 rad = -11 s of rotation already at J2000),
  and the IAU rate (W rate plus the -0.641 deg/century node term) is 6.3e-6 deg/day slower
  than ERA (-4.04e-5 rad/yr = -0.55 s of rotation per year while TT - UT1 is constant).
  Offset = ``omega [(TT - UT1) - 75.06 s - 0.554 s/yr (t - J2000)]``: -1.25e-3 rad (-17 s,
  8.0 km at the equator) in 2020, about -1.64e-3 rad (-22.5 s, 10.5 km) in 2030 and
  -2.04e-3 rad (-28 s, 13.0 km) in 2040 if TT - UT1 stays at its 2017-2026 value (~69.2 s).
  Nutation does not enter this angle (measured: the closed form matches ERFA ``c2t06a`` to
  2.4e-8 rad).
* **Pole (secondary).** Nutation is omitted (pole offset up to sqrt((17.2" sin 23.4 deg)^2 +
  (9.2")^2) ~ 11.5" ~ 5.6e-5 rad, ~0.36 km at the surface; measured 3.3-4.7e-5 rad), as is
  polar motion (~0.3" ~ 10 m).

Measured against astropy's IERS-based ITRS -> GCRS transformation (bundled IERS-B, no network)
an equatorial site is displaced by 6.1 km (2005), 7.9 km (2020) and 8.7 km (2023); see
``tests/unit/test_ground.py`` (astropy and ERFA checks) and docs/model_assumptions.md (M3).
For the epochs simulated (2030+) UT1 cannot be predicted to better than a few seconds anyway,
so this offset is a shift of the simulated site along its parallel (diurnal timing, ~23 s in
2030), not an error of the simulated pointing: the simulated transmitter targets the simulated
site.

Inertial site state (J2000/SSB, at TDB epoch t)::

    r_site(t) = r_Earth(t) + R(t) r_ef
    v_site(t) = v_Earth(t) + dR/dt(t) r_ef          (= v_Earth + omega x (R r_ef))

with the Earth's (NAIF 399) geometric J2000/SSB state from the ephemeris provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import spiceypy
from numpy.typing import ArrayLike, NDArray

from helioswarm.config import SiteConfig
from helioswarm.constants import (
    WGS84_FIRST_ECCENTRICITY_SQUARED,
    WGS84_SEMIMAJOR_AXIS_M,
)
from helioswarm.ephemeris import SSB, EphemerisProvider
from helioswarm.frames import CartesianTrajectory, FrameName, OriginName, TimeScale
from helioswarm.kernels import LoadedKernels

Float64Array = NDArray[np.float64]

EarthFixedFrame = Literal["IAU_EARTH"]
"""Earth-fixed frames supported (SPICE frame names)."""

EllipsoidName = Literal["WGS84"]
"""Reference ellipsoids supported."""


def _readonly(values: ArrayLike, shape: tuple[int, ...] | None = None) -> Float64Array:
    arr = np.array(values, dtype=np.float64, copy=True)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("array must be finite")
    arr.setflags(write=False)
    return arr


def _epochs_1d(epochs_tdb_s: ArrayLike) -> Float64Array:
    t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("epochs_tdb_s must be a non-empty finite scalar or 1-D array")
    return t


# ---------------------------------------------------------------------------------------------
# Geodetic coordinates on the WGS 84 ellipsoid
# ---------------------------------------------------------------------------------------------


def geodetic_to_earth_fixed_m(
    latitude_rad: ArrayLike,
    longitude_rad: ArrayLike,
    altitude_m: ArrayLike,
    *,
    semimajor_axis_m: float = WGS84_SEMIMAJOR_AXIS_M,
    eccentricity_squared: float = WGS84_FIRST_ECCENTRICITY_SQUARED,
) -> Float64Array:
    """Geodetic (phi, lambda, h) -> Earth-fixed Cartesian position [m].

    ``x = (N + h) cos phi cos lambda``, ``y = (N + h) cos phi sin lambda``,
    ``z = (N (1 - e^2) + h) sin phi``, ``N = a / sqrt(1 - e^2 sin^2 phi)`` (WGS 84,
    NGA.STND.0036; Hofmann-Wellenhof et al. 2008, eq. 2.31).

    Args:
        latitude_rad: Geodetic latitude phi [rad], in [-pi/2, pi/2].
        longitude_rad: Longitude lambda, east positive [rad].
        altitude_m: Height h above the ellipsoid along the normal [m].
        semimajor_axis_m: Ellipsoid semi-major axis a [m] (default WGS 84).
        eccentricity_squared: First eccentricity squared e^2 [1] (default WGS 84).

    Returns:
        Shape ``broadcast(...) + (3,)`` [m], in the Earth-fixed frame of the ellipsoid.
    """
    phi = np.asarray(latitude_rad, dtype=np.float64)
    lam = np.asarray(longitude_rad, dtype=np.float64)
    h = np.asarray(altitude_m, dtype=np.float64)
    if np.any(np.abs(phi) > 0.5 * math.pi + 1e-15):
        raise ValueError("geodetic latitude must lie in [-pi/2, pi/2]")
    sphi, cphi = np.sin(phi), np.cos(phi)
    n = semimajor_axis_m / np.sqrt(1.0 - eccentricity_squared * sphi * sphi)
    x = (n + h) * cphi * np.cos(lam)
    y = (n + h) * cphi * np.sin(lam)
    z = (n * (1.0 - eccentricity_squared) + h) * sphi
    return np.asarray(np.stack(np.broadcast_arrays(x, y, z), axis=-1), dtype=np.float64)


def local_enu_basis(
    latitude_rad: float, longitude_rad: float
) -> tuple[Float64Array, Float64Array, Float64Array]:
    """Local east, north, up unit vectors (Earth-fixed components) at a geodetic position.

    ``up = (cos phi cos lambda, cos phi sin lambda, sin phi)`` is the ellipsoid normal (the
    geodetic vertical; it does not pass through the Earth's centre unless phi = 0 or +-90 deg),
    ``east = (-sin lambda, cos lambda, 0)``,
    ``north = (-sin phi cos lambda, -sin phi sin lambda, cos phi)``; right-handed
    (east x north = up). At the poles east is the direction of increasing longitude lambda.

    Args:
        latitude_rad: Geodetic latitude [rad].
        longitude_rad: Longitude, east positive [rad].

    Returns:
        ``(east, north, up)``, each shape (3,) [1].
    """
    sphi, cphi = math.sin(latitude_rad), math.cos(latitude_rad)
    slam, clam = math.sin(longitude_rad), math.cos(longitude_rad)
    east = np.array([-slam, clam, 0.0])
    north = np.array([-sphi * clam, -sphi * slam, cphi])
    up = np.array([cphi * clam, cphi * slam, sphi])
    return east, north, up


@dataclass(frozen=True, eq=False)
class GroundSite:
    """A receiver site fixed to the Earth (Earth-fixed coordinates only; never inertial).

    Attributes:
        name: Site name.
        latitude_rad: Geodetic latitude (WGS 84) [rad].
        longitude_rad: Longitude, east positive [rad].
        altitude_m: Height above the ellipsoid [m].
        position_ef_m: Earth-fixed Cartesian position, shape (3,) [m].
        east_ef: Local east unit vector (Earth-fixed), shape (3,).
        north_ef: Local north unit vector (Earth-fixed), shape (3,).
        up_ef: Local up unit vector = ellipsoid normal (Earth-fixed), shape (3,).
        frame: Earth-fixed frame of the vectors (``IAU_EARTH``).
        ellipsoid: Reference ellipsoid (``WGS84``).
    """

    name: str
    latitude_rad: float
    longitude_rad: float
    altitude_m: float
    position_ef_m: Float64Array
    east_ef: Float64Array
    north_ef: Float64Array
    up_ef: Float64Array
    frame: EarthFixedFrame = field(default="IAU_EARTH")
    ellipsoid: EllipsoidName = field(default="WGS84")

    def __post_init__(self) -> None:
        for name in ("latitude_rad", "longitude_rad", "altitude_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if abs(self.latitude_rad) > 0.5 * math.pi + 1e-15:
            raise ValueError("latitude_rad must lie in [-pi/2, pi/2]")
        for name in ("position_ef_m", "east_ef", "north_ef", "up_ef"):
            object.__setattr__(self, name, _readonly(getattr(self, name), (3,)))
        if self.frame != "IAU_EARTH" or self.ellipsoid != "WGS84":
            raise ValueError("only IAU_EARTH / WGS84 sites are supported")

    @classmethod
    def from_geodetic(
        cls, name: str, latitude_rad: float, longitude_rad: float, altitude_m: float = 0.0
    ) -> GroundSite:
        """Build a site from geodetic coordinates on the WGS 84 ellipsoid (radians, metres)."""
        east, north, up = local_enu_basis(latitude_rad, longitude_rad)
        return cls(
            name=name,
            latitude_rad=latitude_rad,
            longitude_rad=longitude_rad,
            altitude_m=altitude_m,
            position_ef_m=geodetic_to_earth_fixed_m(latitude_rad, longitude_rad, altitude_m),
            east_ef=east,
            north_ef=north,
            up_ef=up,
        )

    @classmethod
    def from_config(cls, site: SiteConfig) -> GroundSite:
        """Build a site from a configuration entry (degrees are converted here, at the IO
        boundary)."""
        return cls.from_geodetic(site.name, site.latitude_rad, site.longitude_rad, site.altitude_m)


# ---------------------------------------------------------------------------------------------
# Earth orientation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EarthRotation:
    """Earth-fixed -> inertial rotation matrices and their time derivatives.

    ``v_inertial = matrix[k] @ v_earth_fixed`` and, for a point fixed to the Earth,
    ``d/dt (matrix[k] @ r_ef) = matrix_rate_per_s[k] @ r_ef``.

    Attributes:
        epochs_tdb_s: Epochs, shape (N,) [TDB s past J2000].
        matrix: Rotation matrices, shape (N, 3, 3) [1].
        matrix_rate_per_s: Their time derivatives, shape (N, 3, 3) [s^-1].
        from_frame: Earth-fixed frame name.
        to_frame: Inertial frame name (``J2000``).
    """

    epochs_tdb_s: Float64Array
    matrix: Float64Array
    matrix_rate_per_s: Float64Array
    from_frame: EarthFixedFrame
    to_frame: FrameName

    def __post_init__(self) -> None:
        t = _readonly(self.epochs_tdb_s)
        n = t.shape[0]
        object.__setattr__(self, "epochs_tdb_s", t)
        object.__setattr__(self, "matrix", _readonly(self.matrix, (n, 3, 3)))
        object.__setattr__(self, "matrix_rate_per_s", _readonly(self.matrix_rate_per_s, (n, 3, 3)))
        if self.to_frame != "J2000":
            raise ValueError("EarthRotation.to_frame must be J2000")


@runtime_checkable
class EarthOrientation(Protocol):
    """Model of the Earth-fixed -> J2000 rotation (injectable, like the ephemeris provider)."""

    @property
    def earth_fixed_frame(self) -> EarthFixedFrame:
        """Earth-fixed frame the model rotates from."""
        ...

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Data the rotation comes from (kernel names, or a model label)."""
        ...

    def rotation(self, epochs_tdb_s: ArrayLike) -> EarthRotation:
        """Rotation matrices and rates at TDB epochs."""
        ...


@dataclass(frozen=True)
class SpiceEarthOrientation:
    """``IAU_EARTH`` -> ``J2000`` from the loaded text PCK (``sxform``; see module docstring).

    ``sxform`` returns the 6x6 state transformation ``[[R, 0], [dR/dt, R]]``; its upper-left
    block is ``pxform``'s rotation and its lower-left block the exact derivative of the IAU
    model (NAIF Rotations Required Reading).

    Attributes:
        kernels: Active kernel-set handle (must contain a PCK with the IAU Earth constants).
    """

    kernels: LoadedKernels
    earth_fixed_frame: EarthFixedFrame = field(default="IAU_EARTH", init=False)

    @property
    def source_ids(self) -> tuple[str, ...]:
        """PCK kernel ids of the loaded set."""
        return self.kernels.ids_of_kind("PCK")

    def rotation(self, epochs_tdb_s: ArrayLike) -> EarthRotation:
        """IAU_EARTH -> J2000 rotation and rate at each epoch (one ``sxform`` per epoch)."""
        self.kernels.require_active(kind="PCK")
        t = _epochs_1d(epochs_tdb_s)
        mat = np.empty((t.size, 3, 3))
        rate = np.empty((t.size, 3, 3))
        for k, et in enumerate(t):
            x = np.asarray(spiceypy.sxform("IAU_EARTH", "J2000", float(et)), dtype=np.float64)
            mat[k] = x[:3, :3]
            rate[k] = x[3:, :3]
        return EarthRotation(
            epochs_tdb_s=t,
            matrix=mat,
            matrix_rate_per_s=rate,
            from_frame="IAU_EARTH",
            to_frame="J2000",
        )


@dataclass(frozen=True)
class UniformEarthRotation:
    """Kernel-free test double: rotation about the J2000 z-axis at a constant rate.

    ``R(t) = R_z(theta(t))`` (active rotation, ``[[c, -s, 0], [s, c, 0], [0, 0, 1]]``) with
    ``theta(t) = angle_at_j2000_rad + rate_rad_s * t``; ``dR/dt = rate * dR_z/dtheta``. With
    ``rate_rad_s = 0`` the Earth does not rotate, which isolates translation in tests. It is not
    a physical Earth-orientation model (no precession; pole fixed on the J2000 z-axis).

    Attributes:
        angle_at_j2000_rad: Rotation angle at t = 0 [rad].
        rate_rad_s: Rotation rate [rad s^-1].
    """

    angle_at_j2000_rad: float = 0.0
    rate_rad_s: float = 0.0
    earth_fixed_frame: EarthFixedFrame = field(default="IAU_EARTH", init=False)

    @property
    def source_ids(self) -> tuple[str, ...]:
        """``("uniform-rotation-test-double",)``."""
        return ("uniform-rotation-test-double",)

    def rotation(self, epochs_tdb_s: ArrayLike) -> EarthRotation:
        """``R_z(theta(t))`` and its derivative at each epoch."""
        t = _epochs_1d(epochs_tdb_s)
        th = self.angle_at_j2000_rad + self.rate_rad_s * t
        c, s = np.cos(th), np.sin(th)
        z, o = np.zeros_like(th), np.ones_like(th)
        mat = np.stack(
            [np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], 1
        )
        dmat = self.rate_rad_s * np.stack(
            [np.stack([-s, -c, z], -1), np.stack([c, -s, z], -1), np.stack([z, z, z], -1)], 1
        )
        return EarthRotation(
            epochs_tdb_s=t,
            matrix=mat,
            matrix_rate_per_s=dmat,
            from_frame="IAU_EARTH",
            to_frame="J2000",
        )


# ---------------------------------------------------------------------------------------------
# Inertial site states
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SiteTrajectory:
    """Inertial states of a ground site (J2000 / SSB / TDB) with its local frame.

    Attributes:
        site: The Earth-fixed site definition.
        epochs_tdb_s: Epochs, shape (N,) [TDB s past J2000].
        position_m: Site positions, shape (N, 3), J2000 / SSB [m].
        velocity_m_s: Site velocities, shape (N, 3), J2000 / SSB [m s^-1].
        earth_position_m: Earth-centre (NAIF 399) positions, shape (N, 3), J2000 / SSB [m].
        site_geocentric_m: ``R(t) r_ef``, shape (N, 3), J2000 axes, origin Earth centre [m].
        east_unit: Local east, J2000 components, shape (N, 3).
        north_unit: Local north, J2000 components, shape (N, 3).
        up_unit: Local up (ellipsoid normal), J2000 components, shape (N, 3).
        rotation_matrix: Earth-fixed -> J2000 matrices used, shape (N, 3, 3).
        frame: ``J2000``.
        origin: ``SOLAR SYSTEM BARYCENTER`` (of ``position_m``/``velocity_m_s``).
        time_scale: ``TDB``.
    """

    site: GroundSite
    epochs_tdb_s: Float64Array
    position_m: Float64Array
    velocity_m_s: Float64Array
    earth_position_m: Float64Array
    site_geocentric_m: Float64Array
    east_unit: Float64Array
    north_unit: Float64Array
    up_unit: Float64Array
    rotation_matrix: Float64Array
    frame: FrameName = field(default="J2000")
    origin: OriginName = field(default=SSB)
    time_scale: TimeScale = field(default="TDB")

    def __post_init__(self) -> None:
        t = _readonly(self.epochs_tdb_s)
        if t.ndim != 1:
            raise ValueError("epochs_tdb_s must be 1-D")
        n = t.shape[0]
        object.__setattr__(self, "epochs_tdb_s", t)
        for name in (
            "position_m",
            "velocity_m_s",
            "earth_position_m",
            "site_geocentric_m",
            "east_unit",
            "north_unit",
            "up_unit",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name), (n, 3)))
        object.__setattr__(self, "rotation_matrix", _readonly(self.rotation_matrix, (n, 3, 3)))
        if self.frame != "J2000" or self.origin != SSB or self.time_scale != "TDB":
            raise ValueError("SiteTrajectory is J2000 / SSB / TDB by contract")

    def __len__(self) -> int:
        return int(self.epochs_tdb_s.shape[0])

    def as_cartesian(self) -> CartesianTrajectory:
        """Site positions/velocities as a :class:`~helioswarm.frames.CartesianTrajectory`."""
        return CartesianTrajectory(
            epochs_tdb_s=self.epochs_tdb_s,
            position_m=self.position_m,
            velocity_m_s=self.velocity_m_s,
            frame=self.frame,
            origin=self.origin,
        )


def site_trajectory(
    site: GroundSite,
    epochs_tdb_s: ArrayLike,
    *,
    ephemeris: EphemerisProvider,
    earth_orientation: EarthOrientation,
) -> SiteTrajectory:
    """Inertial (J2000 / SSB) states of an Earth-fixed site at TDB epochs.

    ``r = r_Earth(t) + R(t) r_ef``, ``v = v_Earth(t) + dR/dt(t) r_ef``; local unit vectors
    ``e_J2000 = R(t) e_ef`` (module docstring; PDF §7.2: an explicit Earth-orientation model,
    never an inertially fixed site).

    Args:
        site: Earth-fixed site.
        epochs_tdb_s: TDB epochs [s past J2000].
        ephemeris: Provider of the Earth's (``EARTH``, NAIF 399) J2000/SSB state.
        earth_orientation: Earth-fixed -> J2000 rotation model (its Earth-fixed frame must be
            the site's).

    Returns:
        The site trajectory.
    """
    if earth_orientation.earth_fixed_frame != site.frame:
        raise ValueError(
            f"site is in {site.frame}, Earth-orientation model rotates from "
            f"{earth_orientation.earth_fixed_frame}"
        )
    if ephemeris.frame != "J2000" or ephemeris.origin != SSB:
        raise ValueError("ephemeris must provide J2000 / SSB states")
    t = _epochs_1d(epochs_tdb_s)
    rot = earth_orientation.rotation(t)
    earth = ephemeris.states("EARTH", t)
    geo = rot.matrix @ site.position_ef_m
    geo_vel = rot.matrix_rate_per_s @ site.position_ef_m
    return SiteTrajectory(
        site=site,
        epochs_tdb_s=t,
        position_m=earth.position_m + geo,
        velocity_m_s=earth.velocity_m_s + geo_vel,
        earth_position_m=earth.position_m,
        site_geocentric_m=geo,
        east_unit=rot.matrix @ site.east_ef,
        north_unit=rot.matrix @ site.north_ef,
        up_unit=rot.matrix @ site.up_ef,
        rotation_matrix=rot.matrix,
    )


__all__ = [
    "EarthFixedFrame",
    "EarthOrientation",
    "EarthRotation",
    "EllipsoidName",
    "GroundSite",
    "SiteTrajectory",
    "SpiceEarthOrientation",
    "UniformEarthRotation",
    "geodetic_to_earth_fixed_m",
    "local_enu_basis",
    "site_trajectory",
]
