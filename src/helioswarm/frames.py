"""Frame-tagged Cartesian states and fixed inertial frame rotations.

Every Cartesian state in ``helioswarm`` carries its **frame** (SPICE frame name), its
**origin** (SPICE body name) and its **epoch** as TDB seconds past J2000 (SPICE ephemeris time,
``epoch_tdb_s``). Positions are in m and velocities in m/s.

Frames implemented here (both inertial, non-rotating):

* ``J2000``: the SPICE realisation of the ICRF-aligned mean equator and equinox of J2000.
* ``ECLIPJ2000``: the mean ecliptic and equinox of J2000, defined by SPICE as a fixed rotation
  of ``J2000`` about the common x-axis (the J2000 equinox) by the IAU 1976 obliquity
  epsilon_0 = 84381.448 arcsec (NAIF Frames Required Reading).

The transformation is a pure rotation and does not change the origin. Changing the origin
(e.g. Sun -> solar-system barycentre) needs an ephemeris; this module never does it, and it
refuses to combine states whose frames or origins differ.

Rotation convention (passive, i.e. same vector expressed in another basis)::

    v_J2000 = R_x(-epsilon_0) v_ECLIPJ2000,
    R_x(-epsilon_0) = [[1, 0, 0], [0, cos e, -sin e], [0, sin e, cos e]]

so the ecliptic north pole (0, 0, 1)_ECL maps to (0, -sin e, cos e)_J2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal, get_args

import numpy as np
import spiceypy
from numpy.typing import ArrayLike, NDArray

from helioswarm.constants import OBLIQUITY_J2000_RAD
from helioswarm.kernels import LoadedKernels

FrameName = Literal["J2000", "ECLIPJ2000"]
"""Inertial frames supported by the frame utilities (SPICE frame names)."""

OriginName = Literal["SUN", "SOLAR SYSTEM BARYCENTER", "EARTH"]
"""State origins supported (SPICE body names). ``EARTH`` (NAIF 399, the planet's centre) is
the origin of geocentric orbital elements (M7b GEO baseline); origin changes need an ephemeris
(:func:`helioswarm.ephemeris.shift_state_origin`)."""

TimeScale = Literal["TDB"]
"""Time scale of every state epoch. Epochs are TDB seconds past J2000 (SPICE ET)."""

SUPPORTED_FRAMES: Final[tuple[str, ...]] = get_args(FrameName)
SUPPORTED_ORIGINS: Final[tuple[str, ...]] = get_args(OriginName)

Float64Array = NDArray[np.float64]


class FrameMismatchError(ValueError):
    """Raised when quantities expressed in different frames or origins would be combined."""


def _check_frame(frame: str) -> None:
    if frame not in SUPPORTED_FRAMES:
        raise ValueError(f"unsupported frame {frame!r}; supported: {list(SUPPORTED_FRAMES)}")


def _check_origin(origin: str) -> None:
    if origin not in SUPPORTED_ORIGINS:
        raise ValueError(f"unsupported origin {origin!r}; supported: {list(SUPPORTED_ORIGINS)}")


def _readonly_float_array(values: ArrayLike, shape_desc: str, name: str) -> Float64Array:
    arr = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (shape {shape_desc})")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class CartesianState:
    """Position and velocity of one body at one epoch, with explicit frame and origin.

    Attributes:
        epoch_tdb_s: Epoch [s], TDB seconds past J2000 (SPICE ephemeris time).
        position_m: Position vector, shape (3,) [m].
        velocity_m_s: Velocity vector, shape (3,) [m s^-1].
        frame: SPICE inertial frame name the vectors are expressed in.
        origin: SPICE body name of the origin.
        time_scale: Always ``"TDB"``; stored so that serialised states carry it.

    The arrays are copied and made read-only on construction.
    """

    epoch_tdb_s: float
    position_m: Float64Array
    velocity_m_s: Float64Array
    frame: FrameName
    origin: OriginName
    time_scale: TimeScale = field(default="TDB")

    def __post_init__(self) -> None:
        if not math.isfinite(self.epoch_tdb_s):
            raise ValueError("epoch_tdb_s must be finite")
        object.__setattr__(self, "epoch_tdb_s", float(self.epoch_tdb_s))
        pos = _readonly_float_array(self.position_m, "(3,)", "position_m")
        vel = _readonly_float_array(self.velocity_m_s, "(3,)", "velocity_m_s")
        if pos.shape != (3,) or vel.shape != (3,):
            raise ValueError(
                f"position_m and velocity_m_s must have shape (3,), got {pos.shape} and {vel.shape}"
            )
        object.__setattr__(self, "position_m", pos)
        object.__setattr__(self, "velocity_m_s", vel)
        _check_frame(self.frame)
        _check_origin(self.origin)
        if self.time_scale != "TDB":
            raise ValueError(f"time_scale must be 'TDB', got {self.time_scale!r}")


@dataclass(frozen=True, eq=False)
class CartesianTrajectory:
    """Time series of Cartesian states of one body, with explicit frame and origin.

    Attributes:
        epochs_tdb_s: Epochs, shape (N,) [s], TDB seconds past J2000.
        position_m: Positions, shape (N, 3) [m].
        velocity_m_s: Velocities, shape (N, 3) [m s^-1].
        frame: SPICE inertial frame name.
        origin: SPICE body name of the origin.
        time_scale: Always ``"TDB"``.
    """

    epochs_tdb_s: Float64Array
    position_m: Float64Array
    velocity_m_s: Float64Array
    frame: FrameName
    origin: OriginName
    time_scale: TimeScale = field(default="TDB")

    def __post_init__(self) -> None:
        t = _readonly_float_array(self.epochs_tdb_s, "(N,)", "epochs_tdb_s")
        pos = _readonly_float_array(self.position_m, "(N, 3)", "position_m")
        vel = _readonly_float_array(self.velocity_m_s, "(N, 3)", "velocity_m_s")
        if t.ndim != 1:
            raise ValueError(f"epochs_tdb_s must be 1-D, got shape {t.shape}")
        n = t.shape[0]
        if pos.shape != (n, 3) or vel.shape != (n, 3):
            raise ValueError(
                f"position_m and velocity_m_s must have shape ({n}, 3), got {pos.shape} and "
                f"{vel.shape}"
            )
        object.__setattr__(self, "epochs_tdb_s", t)
        object.__setattr__(self, "position_m", pos)
        object.__setattr__(self, "velocity_m_s", vel)
        _check_frame(self.frame)
        _check_origin(self.origin)
        if self.time_scale != "TDB":
            raise ValueError(f"time_scale must be 'TDB', got {self.time_scale!r}")

    def __len__(self) -> int:
        return int(self.epochs_tdb_s.shape[0])

    def state_at(self, index: int) -> CartesianState:
        """Return the state at sample ``index`` as a :class:`CartesianState`."""
        return CartesianState(
            epoch_tdb_s=float(self.epochs_tdb_s[index]),
            position_m=self.position_m[index],
            velocity_m_s=self.velocity_m_s[index],
            frame=self.frame,
            origin=self.origin,
        )


def rotation_x(angle_rad: float) -> Float64Array:
    """Passive rotation matrix about the x-axis by ``angle_rad`` [rad].

    ``R_x(t) = [[1, 0, 0], [0, cos t, sin t], [0, -sin t, cos t]]`` (frame rotation: the
    components of a fixed vector in a basis rotated by +t about x). Reference: Vallado,
    *Fundamentals of Astrodynamics and Applications*, 4th ed., ch. 3 (ROT1).
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=np.float64)


def ecliptic_j2000_to_j2000_matrix(obliquity_rad: float = OBLIQUITY_J2000_RAD) -> Float64Array:
    """Rotation matrix taking ECLIPJ2000 components to J2000 components [1].

    ``v_J2000 = M @ v_ECLIPJ2000`` with ``M = R_x(-epsilon_0)``, epsilon_0 the IAU 1976 mean
    obliquity of J2000 (84381.448 arcsec), as in the SPICE ECLIPJ2000 frame definition. The
    matrix is orthonormal with determinant +1; its transpose is the inverse transformation.
    """
    return rotation_x(-obliquity_rad)


def rotation_matrix(from_frame: FrameName, to_frame: FrameName) -> Float64Array:
    """Rotation matrix ``M`` with ``v_to = M @ v_from`` between two supported inertial frames.

    Both supported frames are inertial and share their origin-independent orientation, so
    the matrix is constant in time (no epoch argument).
    """
    _check_frame(from_frame)
    _check_frame(to_frame)
    if from_frame == to_frame:
        return np.eye(3, dtype=np.float64)
    m = ecliptic_j2000_to_j2000_matrix()
    if from_frame == "ECLIPJ2000":
        return m
    return np.ascontiguousarray(m.T)


def rotate_vectors(vectors: ArrayLike, from_frame: FrameName, to_frame: FrameName) -> Float64Array:
    """Express vectors given in ``from_frame`` in ``to_frame``.

    Args:
        vectors: Array of shape (3,) or (N, 3); any unit (rotation preserves units).
        from_frame: Frame the input components refer to.
        to_frame: Frame of the output components.

    Returns:
        Array of the same shape as ``vectors``.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.shape[-1:] != (3,) or arr.ndim not in (1, 2):
        raise ValueError(f"vectors must have shape (3,) or (N, 3), got {arr.shape}")
    m = rotation_matrix(from_frame, to_frame)
    return np.asarray(arr @ m.T, dtype=np.float64)


def transform_state(state: CartesianState, to_frame: FrameName) -> CartesianState:
    """Rotate a state into ``to_frame``. Epoch, origin and time scale are unchanged.

    Valid only between inertial frames (no transport term ``omega x r`` is needed).
    """
    return CartesianState(
        epoch_tdb_s=state.epoch_tdb_s,
        position_m=rotate_vectors(state.position_m, state.frame, to_frame),
        velocity_m_s=rotate_vectors(state.velocity_m_s, state.frame, to_frame),
        frame=to_frame,
        origin=state.origin,
    )


def transform_trajectory(traj: CartesianTrajectory, to_frame: FrameName) -> CartesianTrajectory:
    """Rotate every state of a trajectory into ``to_frame`` (inertial frames only)."""
    return CartesianTrajectory(
        epochs_tdb_s=traj.epochs_tdb_s,
        position_m=rotate_vectors(traj.position_m, traj.frame, to_frame),
        velocity_m_s=rotate_vectors(traj.velocity_m_s, traj.frame, to_frame),
        frame=to_frame,
        origin=traj.origin,
    )


def spice_rotation_matrix(
    from_frame: FrameName, to_frame: FrameName, epoch_tdb_s: float, kernels: LoadedKernels
) -> Float64Array:
    """Rotation matrix ``v_to = M @ v_from`` computed by SPICE ``pxform`` (cross-check path).

    SPICE builds ``ECLIPJ2000`` from ``J2000`` with the IAU 1976 obliquity (NAIF Frames
    Required Reading); this function exists so that :func:`rotation_matrix` can be validated
    against the toolkit. The epoch is irrelevant for these inertial frames but is passed to
    ``pxform`` as required by its signature.

    Args:
        from_frame: Source frame.
        to_frame: Target frame.
        epoch_tdb_s: Epoch [TDB s past J2000].
        kernels: Active kernel-set handle (SPICE is only queried inside a loaded scope).
    """
    _check_frame(from_frame)
    _check_frame(to_frame)
    kernels.require_active()
    return np.asarray(spiceypy.pxform(from_frame, to_frame, float(epoch_tdb_s)), dtype=np.float64)


def require_same_frame_and_origin(a: CartesianState, b: CartesianState) -> None:
    """Raise :class:`FrameMismatchError` unless ``a`` and ``b`` share frame and origin.

    Call this before subtracting or otherwise combining two states (PDF §3: never subtract
    vectors expressed in different frames or origins).
    """
    if a.frame != b.frame or a.origin != b.origin:
        raise FrameMismatchError(
            f"cannot combine a state in ({a.frame}, {a.origin}) with one in "
            f"({b.frame}, {b.origin}); transform explicitly first"
        )


__all__ = [
    "SUPPORTED_FRAMES",
    "SUPPORTED_ORIGINS",
    "CartesianState",
    "CartesianTrajectory",
    "FrameMismatchError",
    "FrameName",
    "OriginName",
    "TimeScale",
    "ecliptic_j2000_to_j2000_matrix",
    "require_same_frame_and_origin",
    "rotate_vectors",
    "rotation_matrix",
    "rotation_x",
    "spice_rotation_matrix",
    "transform_state",
    "transform_trajectory",
]
