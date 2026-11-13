"""Tests for helioswarm.frames: frame-tagged states and the ECLIPJ2000 <-> J2000 rotation.

The expected rotation is derived here independently from the definition of the ECLIPJ2000
frame (rotation of J2000 about the common x-axis / equinox by the IAU 1976 obliquity
epsilon_0 = 23 deg 26' 21.448"), and cross-checked against SPICE ``pxform`` (M2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helioswarm.frames import (
    CartesianState,
    CartesianTrajectory,
    FrameMismatchError,
    ecliptic_j2000_to_j2000_matrix,
    require_same_frame_and_origin,
    rotate_vectors,
    rotation_matrix,
    spice_rotation_matrix,
    transform_state,
    transform_trajectory,
)
from helioswarm.kernels import KernelPoolError, KernelSet, LoadedKernels

EPS0 = math.radians(23.0 + 26.0 / 60.0 + 21.448 / 3600.0)  # computed here, not imported


def test_matrix_is_proper_rotation() -> None:
    m = ecliptic_j2000_to_j2000_matrix()
    assert np.allclose(m @ m.T, np.eye(3), rtol=0, atol=1e-15)
    assert np.linalg.det(m) == pytest.approx(1.0, abs=1e-15)


def test_ecliptic_north_pole_maps_to_expected_j2000_vector() -> None:
    # The ecliptic pole lies at RA = 270 deg, Dec = 90 deg - eps in J2000, i.e. the vector
    # (0, -sin eps, cos eps). A sign error in the rotation gives (0, +sin eps, cos eps).
    pole = rotate_vectors([0.0, 0.0, 1.0], "ECLIPJ2000", "J2000")
    assert pole == pytest.approx([0.0, -math.sin(EPS0), math.cos(EPS0)], abs=1e-15)


def test_equinox_axis_invariant_and_ecliptic_y_axis() -> None:
    x = rotate_vectors([1.0, 0.0, 0.0], "ECLIPJ2000", "J2000")
    assert x == pytest.approx([1.0, 0.0, 0.0], abs=0.0)
    # Ecliptic longitude 90 deg lies above the equator by eps: (0, cos eps, sin eps).
    y = rotate_vectors([0.0, 1.0, 0.0], "ECLIPJ2000", "J2000")
    assert y == pytest.approx([0.0, math.cos(EPS0), math.sin(EPS0)], abs=1e-15)


def test_celestial_pole_in_ecliptic_frame() -> None:
    # The J2000 celestial north pole has ecliptic latitude 90 - eps at longitude 90 deg.
    z = rotate_vectors([0.0, 0.0, 1.0], "J2000", "ECLIPJ2000")
    assert z == pytest.approx([0.0, math.sin(EPS0), math.cos(EPS0)], abs=1e-15)


def test_rotation_round_trip_and_identity() -> None:
    rng = np.random.default_rng(7)
    v = rng.normal(size=(50, 3)) * 1e11
    back = rotate_vectors(rotate_vectors(v, "J2000", "ECLIPJ2000"), "ECLIPJ2000", "J2000")
    assert np.max(np.abs(back - v)) <= 1e-15 * np.max(np.abs(v)) * 4
    assert np.array_equal(rotation_matrix("J2000", "J2000"), np.eye(3))
    assert np.allclose(
        rotation_matrix("J2000", "ECLIPJ2000"),
        rotation_matrix("ECLIPJ2000", "J2000").T,
        rtol=0,
        atol=0,
    )
    # Norms preserved (pure rotation).
    rot = rotate_vectors(v, "ECLIPJ2000", "J2000")
    assert np.linalg.norm(rot, axis=1) == pytest.approx(np.linalg.norm(v, axis=1), rel=1e-15, abs=0)


def test_rotate_vectors_rejects_bad_shapes_and_frames() -> None:
    with pytest.raises(ValueError, match="shape"):
        rotate_vectors([1.0, 2.0], "J2000", "ECLIPJ2000")
    with pytest.raises(ValueError, match="unsupported frame"):
        rotate_vectors([1.0, 2.0, 3.0], "J2000", "IAU_EARTH")  # type: ignore[arg-type]


def _state(frame: str = "ECLIPJ2000", origin: str = "SUN") -> CartesianState:
    return CartesianState(
        epoch_tdb_s=1234.5,
        position_m=np.array([1.0e11, 2.0e10, 3.0e9]),
        velocity_m_s=np.array([-1.0e3, 2.9e4, 5.0e2]),
        frame=frame,  # type: ignore[arg-type]
        origin=origin,  # type: ignore[arg-type]
    )


def test_transform_state_keeps_epoch_origin_and_tags_frame() -> None:
    s = _state()
    t = transform_state(s, "J2000")
    assert t.frame == "J2000"
    assert t.origin == "SUN"
    assert t.epoch_tdb_s == s.epoch_tdb_s
    assert t.time_scale == "TDB"
    m = ecliptic_j2000_to_j2000_matrix()
    assert t.position_m == pytest.approx(m @ s.position_m, rel=1e-15, abs=0)
    assert t.velocity_m_s == pytest.approx(m @ s.velocity_m_s, rel=1e-15, abs=0)
    assert transform_state(t, "ECLIPJ2000").position_m == pytest.approx(
        s.position_m, rel=1e-15, abs=0
    )


def test_transform_trajectory_matches_per_state_transform() -> None:
    rng = np.random.default_rng(3)
    traj = CartesianTrajectory(
        epochs_tdb_s=np.arange(4.0),
        position_m=rng.normal(size=(4, 3)),
        velocity_m_s=rng.normal(size=(4, 3)),
        frame="ECLIPJ2000",
        origin="SUN",
    )
    out = transform_trajectory(traj, "J2000")
    assert out.frame == "J2000"
    for k in range(4):
        s = transform_state(traj.state_at(k), "J2000")
        assert out.position_m[k] == pytest.approx(s.position_m, rel=1e-15, abs=0)
        assert out.velocity_m_s[k] == pytest.approx(s.velocity_m_s, rel=1e-15, abs=0)


def test_state_validation_and_immutability() -> None:
    s = _state()
    with pytest.raises(ValueError, match="read-only"):
        s.position_m[0] = 0.0
    with pytest.raises(ValueError, match="shape"):
        CartesianState(0.0, np.zeros(2), np.zeros(3), "J2000", "SUN")
    with pytest.raises(ValueError, match="finite"):
        CartesianState(0.0, np.array([np.nan, 0, 0]), np.zeros(3), "J2000", "SUN")
    with pytest.raises(ValueError, match="unsupported origin"):
        _state(origin="MOON")  # EARTH is a supported origin since M7b (geocentric elements)
    assert _state(origin="EARTH").origin == "EARTH"
    with pytest.raises(ValueError, match="unsupported frame"):
        _state(frame="IAU_EARTH")
    with pytest.raises(ValueError, match="epoch"):
        CartesianState(math.inf, np.zeros(3), np.zeros(3), "J2000", "SUN")
    with pytest.raises(ValueError, match="time_scale"):
        CartesianState(0.0, np.zeros(3), np.zeros(3), "J2000", "SUN", "UTC")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        CartesianTrajectory(np.zeros(3), np.zeros((2, 3)), np.zeros((3, 3)), "J2000", "SUN")


def test_input_arrays_are_copied() -> None:
    pos = np.array([1.0, 2.0, 3.0])
    s = CartesianState(0.0, pos, np.zeros(3), "J2000", "SUN")
    pos[0] = 99.0
    assert s.position_m[0] == 1.0


def test_require_same_frame_and_origin() -> None:
    a = _state("J2000", "SUN")
    require_same_frame_and_origin(a, _state("J2000", "SUN"))
    with pytest.raises(FrameMismatchError, match="ECLIPJ2000"):
        require_same_frame_and_origin(a, _state("ECLIPJ2000", "SUN"))
    with pytest.raises(FrameMismatchError, match="SOLAR SYSTEM BARYCENTER"):
        require_same_frame_and_origin(a, _state("J2000", "SOLAR SYSTEM BARYCENTER"))


@pytest.mark.spice
@pytest.mark.parametrize("epoch_tdb_s", [-3.0e9, 0.0, 9.467e8])
def test_ecliptic_rotation_matches_spice_pxform(
    epoch_tdb_s: float, spice_kernels: LoadedKernels
) -> None:
    # SPICE defines ECLIPJ2000 from J2000 with the same IAU 1976 obliquity; both sides compute
    # cos/sin of the same float, so they agree to rounding (measured: exactly). Tolerance
    # 1e-15 per element; a 1 mas change of the obliquity (4.8e-9 rad) would fail by 1e6x.
    m_spice = spice_rotation_matrix("ECLIPJ2000", "J2000", epoch_tdb_s, spice_kernels)
    np.testing.assert_allclose(m_spice, ecliptic_j2000_to_j2000_matrix(), rtol=0, atol=1e-15)
    inv = spice_rotation_matrix("J2000", "ECLIPJ2000", epoch_tdb_s, spice_kernels)
    np.testing.assert_allclose(inv, rotation_matrix("J2000", "ECLIPJ2000"), rtol=0, atol=1e-15)
    eye = spice_rotation_matrix("J2000", "J2000", epoch_tdb_s, spice_kernels)
    np.testing.assert_array_equal(eye, np.eye(3))


@pytest.mark.spice
def test_spice_rotation_requires_loaded_scope(kernel_set: KernelSet) -> None:
    with kernel_set.loaded(verify=False) as k:
        pass
    with pytest.raises(KernelPoolError):
        spice_rotation_matrix("ECLIPJ2000", "J2000", 0.0, k)
