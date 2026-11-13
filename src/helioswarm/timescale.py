"""Time scales: UTC strings at the IO boundary, TDB seconds past J2000 everywhere inside.

Rules (PDF §3 "Time and coordinate conventions"):

* Internal epochs are **TDB seconds past J2000** (J2000 = 2000-01-01 12:00:00 TDB), i.e.
  SPICE ephemeris time (ET). Every container carries ``scale = "TDB"`` explicitly.
* UTC appears only as ISO-8601 strings in configuration files and reports. UTC <-> TDB
  conversion is done **only** by SPICE (``str2et`` / ``et2utc``) with the leap-seconds kernel
  of the loaded kernel set; there is no implicit conversion anywhere else.

Accuracy of the SPICE conversion: UTC -> TAI uses the leap-second table of the LSK
(``naif0012.tls``: leap seconds through 2017-01-01, TAI - UTC = 37 s; epochs after a future
leap second would be off by 1 s until the LSK is updated). TT = TAI + 32.184 s exactly. TDB - TT
is SPICE's one-term periodic approximation ``K sin(E)``, ``E = M + EB sin M`` (amplitude
K = 1.657e-3 s; NAIF Time Required Reading), which differs from the full Fairhead & Bretagnon
(1990) series by a few times 1e-5 s. That is far below anything the simulator resolves
(1e-4 s of TDB error moves Earth by 3 m along its orbit).

Output time grids are uniform in **TDB** (start converted from UTC once, then
``t_k = t_0 + k * output_step_s``); they are not uniform in UTC (the two differ by the
millisecond-level periodic TDB - TT term and by any leap second inside the window).

References: NAIF "Time Required Reading" (STR2ET, ET2UTC, DELTET); IERS Conventions (2010)
ch. 10; IERS Bulletin C (leap seconds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import spiceypy
from numpy.typing import NDArray

from helioswarm.config import TimeConfig, canonical_utc
from helioswarm.constants import SECONDS_PER_DAY
from helioswarm.kernels import LoadedKernels

TimeScaleName = Literal["TDB"]
"""The only internal time scale."""

J2000_JD_TDB: Final[float] = 2451545.0
"""Julian date of J2000 in TDB (epoch of ``tdb_s = 0``)."""

Float64Array = NDArray[np.float64]


@dataclass(frozen=True)
class Epoch:
    """One instant, as TDB seconds past J2000.

    Attributes:
        tdb_s: TDB seconds past J2000 (SPICE ET) [s].
        scale: Always ``"TDB"``.
    """

    tdb_s: float
    scale: TimeScaleName = field(default="TDB")

    def __post_init__(self) -> None:
        if not math.isfinite(self.tdb_s):
            raise ValueError("tdb_s must be finite")
        if self.scale != "TDB":
            raise ValueError(f"scale must be 'TDB', got {self.scale!r}")
        object.__setattr__(self, "tdb_s", float(self.tdb_s))


@dataclass(frozen=True, eq=False)
class TimeGrid:
    """A strictly increasing grid of TDB epochs.

    Attributes:
        epochs_tdb_s: Epochs, shape (N,), TDB seconds past J2000 [s] (read-only copy).
        step_s: Nominal grid spacing [s] (``t_k = t_0 + k * step_s``).
        start_utc: The UTC string the first epoch was converted from (provenance label only;
            never used for arithmetic).
        scale: Always ``"TDB"``.
    """

    epochs_tdb_s: Float64Array
    step_s: float
    start_utc: str
    scale: TimeScaleName = field(default="TDB")

    def __post_init__(self) -> None:
        t = np.array(self.epochs_tdb_s, dtype=np.float64, copy=True)
        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
            raise ValueError("epochs_tdb_s must be a non-empty finite 1-D array")
        if t.size > 1 and not np.all(np.diff(t) > 0.0):
            raise ValueError("epochs_tdb_s must be strictly increasing")
        if not (math.isfinite(self.step_s) and self.step_s > 0.0):
            raise ValueError("step_s must be finite and > 0")
        if self.scale != "TDB":
            raise ValueError(f"scale must be 'TDB', got {self.scale!r}")
        t.setflags(write=False)
        object.__setattr__(self, "epochs_tdb_s", t)

    def __len__(self) -> int:
        return int(self.epochs_tdb_s.shape[0])


def _spice_utc_string(utc_iso: str) -> str:
    """Canonical ISO UTC -> an explicitly UTC-labelled SPICE time string.

    SPICE's ISO 'T' form cannot carry a time-system label and is read in the *default* system
    (UTC unless changed by TIMDEF). The space-separated form with a trailing ``UTC`` label is
    unambiguous regardless of TIMDEF.
    """
    canonical = canonical_utc(utc_iso)
    date, time_of_day = canonical.split("T")
    return f"{date} {time_of_day} UTC"


def utc_to_tdb_s(utc_iso: str, kernels: LoadedKernels) -> float:
    """Convert an ISO-8601 UTC epoch to TDB seconds past J2000 with SPICE ``str2et``.

    ``TDB = UTC + (TAI - UTC)_LSK + 32.184 s + (TDB - TT)``, evaluated by SPICE (NAIF Time
    Required Reading). Requires an LSK in the loaded kernel set.

    Args:
        utc_iso: ``YYYY-MM-DDTHH:MM:SS[.ffffff][Z]`` (UTC by definition; no offsets, no ``:60``).
        kernels: Active kernel-set handle (must contain an LSK).

    Returns:
        TDB seconds past J2000 [s].
    """
    kernels.require_active(kind="LSK")
    return float(spiceypy.str2et(_spice_utc_string(utc_iso)))


def utc_to_epoch(utc_iso: str, kernels: LoadedKernels) -> Epoch:
    """:func:`utc_to_tdb_s` wrapped in an :class:`Epoch`."""
    return Epoch(tdb_s=utc_to_tdb_s(utc_iso, kernels))


def tdb_s_to_utc(tdb_s: float, kernels: LoadedKernels, *, decimals: int = 6) -> str:
    """Convert TDB seconds past J2000 to an ISO-8601 UTC string with SPICE ``et2utc``.

    Args:
        tdb_s: TDB seconds past J2000 [s].
        kernels: Active kernel-set handle (must contain an LSK).
        decimals: Digits of the seconds fraction (0-9; SPICE rounds).

    Returns:
        ``YYYY-MM-DDTHH:MM:SS[.f...]`` (UTC; a positive leap second prints as ``:60``).
    """
    kernels.require_active(kind="LSK")
    if not 0 <= decimals <= 9:
        raise ValueError(f"decimals must be in [0, 9], got {decimals}")
    if not math.isfinite(tdb_s):
        raise ValueError("tdb_s must be finite")
    return str(spiceypy.et2utc(float(tdb_s), "ISOC", decimals))


def tdb_s_to_jd_tdb(tdb_s: float) -> float:
    """Julian date (TDB) of a TDB epoch: ``JD = 2451545.0 + tdb_s / 86400`` (same scale)."""
    return J2000_JD_TDB + tdb_s / SECONDS_PER_DAY


def uniform_time_grid(
    start_utc: str, step_s: float, n_intervals: int, kernels: LoadedKernels
) -> TimeGrid:
    """TDB-uniform grid ``t_k = utc_to_tdb_s(start_utc) + k * step_s``, k = 0..n_intervals."""
    if n_intervals < 0:
        raise ValueError("n_intervals must be >= 0")
    t0 = utc_to_tdb_s(start_utc, kernels)
    k = np.arange(n_intervals + 1, dtype=np.float64)
    return TimeGrid(epochs_tdb_s=t0 + k * step_s, step_s=step_s, start_utc=canonical_utc(start_utc))


def time_grid_from_config(time: TimeConfig, kernels: LoadedKernels) -> TimeGrid:
    """Output time grid of a run: ``time.start_utc`` + k ``time.output_step_s``, k = 0..n,
    with ``n = time.n_output_intervals`` (the grid ends exactly at the window end)."""
    return uniform_time_grid(time.start_utc, time.output_step_s, time.n_output_intervals, kernels)


__all__ = [
    "J2000_JD_TDB",
    "Epoch",
    "TimeGrid",
    "TimeScaleName",
    "tdb_s_to_jd_tdb",
    "tdb_s_to_utc",
    "time_grid_from_config",
    "uniform_time_grid",
    "utc_to_epoch",
    "utc_to_tdb_s",
]
