"""Synthetic ephemeris provider for geometry tests (importable as ``geometry_helpers``).

:class:`LinearMotionEphemeris` implements :class:`helioswarm.ephemeris.EphemerisProvider` for
bodies in uniform rectilinear motion, ``r(t) = r0 + v t`` (J2000 / SSB, TDB s past J2000).
Such hand-built geometries have closed-form light times and occultation distances, so the
access code can be tested against exact values without kernels. It is not a physical
ephemeris.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.ephemeris import (
    SSB,
    BodyName,
    EphemerisCoverageError,
    EphemerisState,
)
from helioswarm.frames import CartesianTrajectory, FrameName, OriginName

Float64Array = NDArray[np.float64]


@dataclass(frozen=True)
class LinearMotionEphemeris:
    """Bodies moving uniformly: ``bodies[name] = ((x0, y0, z0), (vx, vy, vz))`` [m, m/s]."""

    bodies: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float]]]
    frame: FrameName = field(default="J2000", init=False)
    origin: OriginName = field(default=SSB, init=False)

    @property
    def source_kernel_ids(self) -> tuple[str, ...]:
        return ("synthetic-linear-motion",)

    def check_coverage(
        self, bodies: Iterable[BodyName], start_tdb_s: float, end_tdb_s: float
    ) -> None:
        for body in bodies:
            if body not in self.bodies:
                raise EphemerisCoverageError(f"LinearMotionEphemeris has no body {body!r}")

    def states(self, body: BodyName, epochs_tdb_s: ArrayLike) -> CartesianTrajectory:
        t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
        self.check_coverage((body,), 0.0, 0.0)
        r0, v = (np.asarray(x, dtype=np.float64) for x in self.bodies[body])
        pos = r0[None, :] + t[:, None] * v[None, :]
        vel = np.broadcast_to(v, pos.shape)
        return CartesianTrajectory(
            epochs_tdb_s=t, position_m=pos, velocity_m_s=vel, frame="J2000", origin=SSB
        )

    def state(self, body: BodyName, epoch_tdb_s: float) -> EphemerisState:
        tr = self.states(body, epoch_tdb_s)
        return EphemerisState(
            body=body,
            epoch_tdb_s=float(epoch_tdb_s),
            position_m=tr.position_m[0],
            velocity_m_s=tr.velocity_m_s[0],
            source_kernel_ids=self.source_kernel_ids,
        )

    def positions(self, body: BodyName, epochs_tdb_s: ArrayLike) -> Float64Array:
        return np.array(self.states(body, epochs_tdb_s).position_m)

    def gm_m3_s2(self, body: BodyName) -> float:
        raise KeyError("LinearMotionEphemeris has no GM values")


def linear_trajectory(
    epochs_tdb_s: ArrayLike, r0_m: ArrayLike, v_m_s: ArrayLike = (0.0, 0.0, 0.0)
) -> CartesianTrajectory:
    """Uniformly moving satellite ``r(t) = r0 + v t`` (J2000 / SSB)."""
    t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
    r0 = np.asarray(r0_m, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    return CartesianTrajectory(
        epochs_tdb_s=t,
        position_m=r0[None, :] + t[:, None] * v[None, :],
        velocity_m_s=np.broadcast_to(v, (t.size, 3)),
        frame="J2000",
        origin=SSB,
    )
