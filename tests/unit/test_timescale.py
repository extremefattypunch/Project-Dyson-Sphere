"""Tests for helioswarm.timescale (UTC <-> TDB via SPICE; explicit TDB containers).

Independent references used:

* Leap seconds TAI - UTC from IERS Bulletin C (history table, e.g.
  https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat): 32 s from 1999-01-01, 33 s from
  2006-01-01, 34 s from 2009-01-01, 35 s from 2012-07-01, 36 s from 2015-07-01, 37 s from
  2017-01-01 (no leap second announced after it as of 2026-11).
* TT = TAI + 32.184 s exactly (IAU 1991 / IERS Conventions 2010, ch. 10).
* TDB - TT two-term approximation, USNO Circular 179 (Kaplan 2005), eq. 2.6:
  ``0.001657 sin g + 0.000014 sin 2g`` s, ``g = 357.53 deg + 0.98560028 deg (JD - 2451545.0)``.
* astropy ``Time(..., scale="utc").tdb`` (ERFA ``dtdb``: the full Fairhead & Bretagnon 1990
  series), an implementation independent of SPICE.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
import spiceypy

from helioswarm.config import load_config
from helioswarm.kernels import KernelPoolError, KernelSet, LoadedKernels
from helioswarm.timescale import (
    Epoch,
    TimeGrid,
    tdb_s_to_jd_tdb,
    tdb_s_to_utc,
    time_grid_from_config,
    utc_to_epoch,
    utc_to_tdb_s,
)

from config_helpers import MVP_CONFIG

pytestmark = pytest.mark.spice

# IERS Bulletin C leap-second history (TAI - UTC [s]) from the dates below.
_LEAP_TABLE = (
    (datetime(1999, 1, 1), 32),
    (datetime(2006, 1, 1), 33),
    (datetime(2009, 1, 1), 34),
    (datetime(2012, 7, 1), 35),
    (datetime(2015, 7, 1), 36),
    (datetime(2017, 1, 1), 37),
)
_J2000_UTC_CALENDAR = datetime(2000, 1, 1, 12, 0, 0)


def _tai_minus_utc(dt: datetime) -> int:
    value = None
    for start, delta_at in _LEAP_TABLE:
        if dt >= start:
            value = delta_at
    assert value is not None
    return value


def _tdb_minus_tt_usno(jd: float) -> float:
    g = math.radians(357.53 + 0.98560028 * (jd - 2451545.0))
    return 0.001657 * math.sin(g) + 0.000014 * math.sin(2.0 * g)


def _independent_et(utc: str) -> float:
    """ET = (calendar seconds since 2000-01-01T12:00 in TT's leap-free count) + TAI-UTC + 32.184
    + (TDB - TT); Python's datetime arithmetic has no leap seconds, exactly like TT."""
    dt = datetime.fromisoformat(utc)
    cal = (dt - _J2000_UTC_CALENDAR).total_seconds()
    tt = cal + _tai_minus_utc(dt) + 32.184
    return tt + _tdb_minus_tt_usno(2451545.0 + tt / 86400.0)


_EPOCHS = [
    "2000-01-01T12:00:00",
    "2005-06-15T00:00:00",
    "2008-03-01T06:30:00",
    "2010-06-15T00:00:00",
    "2013-01-01T00:00:00",
    "2016-01-01T00:00:00",
    "2017-01-01T00:00:00",
    "2025-06-01T00:00:00",
    "2030-01-01T00:00:00",
]


@pytest.mark.parametrize("utc", _EPOCHS)
def test_utc_to_tdb_against_leap_second_table_and_usno_tdb_formula(
    utc: str, spice_kernels: LoadedKernels
) -> None:
    # Tolerance 1e-5 s. SPICE's K sin(M + EB sin M) and the USNO two-term series expand the
    # same leading periodic terms (the EB sin M inside the sine supplies the 2g harmonic), so
    # they agree closely: measured max 3.6e-7 s over these epochs. A wrong leap-second count
    # is off by >= 1 s, a missing 32.184 s offset by 32 s, a dropped TDB - TT term by <= 1.7 ms.
    assert utc_to_tdb_s(utc, spice_kernels) == pytest.approx(_independent_et(utc), abs=1e-5)


def test_j2000_epoch_in_utc(spice_kernels: LoadedKernels) -> None:
    # 2000-01-01 12:00:00 UTC is 64.184 s (32 s leap + 32.184 s) + TDB - TT (-7.3e-5 s) after
    # J2000 TDB.
    et = utc_to_tdb_s("2000-01-01T12:00:00", spice_kernels)
    assert et == pytest.approx(64.184 + _tdb_minus_tt_usno(2451545.0), abs=1e-5)
    assert 64.1838 < et < 64.1840


def test_leap_second_at_end_of_2016(spice_kernels: LoadedKernels) -> None:
    # 2016-12-31T23:59:60 was a leap second, so one UTC "second" on the calendar spans 2 SI s.
    before = utc_to_tdb_s("2016-12-31T23:59:59", spice_kernels)
    after = utc_to_tdb_s("2017-01-01T00:00:00", spice_kernels)
    assert after - before == pytest.approx(2.0, abs=1e-6)
    # Control: no leap second at the end of June 2016.
    b2 = utc_to_tdb_s("2016-06-30T23:59:59", spice_kernels)
    a2 = utc_to_tdb_s("2016-07-01T00:00:00", spice_kernels)
    assert a2 - b2 == pytest.approx(1.0, abs=1e-6)
    # et2utc prints the leap second itself.
    assert tdb_s_to_utc(before + 1.5, spice_kernels, decimals=1) == "2016-12-31T23:59:60.5"


def test_utc_to_tdb_matches_astropy_erfa(spice_kernels: LoadedKernels) -> None:
    # Cross-check against astropy (ERFA dtdb, full Fairhead-Bretagnon series, geocentre).
    # Tolerance 1e-4 s: SPICE's TDB - TT keeps only the main periodic term (amplitude
    # 1.657 ms); the omitted terms are a few 1e-5 s (measured max 2.7e-5 s). Epochs end in
    # 2025 because ERFA flags later UTC dates as "dubious year" (leap seconds unknown);
    # astropy's IERS auto-download is disabled so the check is offline and reproducible.
    from astropy.time import Time
    from astropy.utils import iers

    epochs = _EPOCHS[:-1]
    diffs = []
    with iers.conf.set_temp("auto_download", False):
        for utc in epochs:
            tdb = Time(utc, scale="utc").tdb
            et_astropy = (tdb.jd1 - 2451545.0) * 86400.0 + tdb.jd2 * 86400.0
            diffs.append(utc_to_tdb_s(utc, spice_kernels) - et_astropy)
    assert max(abs(d) for d in diffs) < 1e-4, diffs


@pytest.mark.parametrize(
    "utc", ["2030-01-01T00:00:00", "2017-01-01T00:00:00.123456", "1999-12-31T23:59:59.5"]
)
def test_round_trip_utc_tdb_utc(utc: str, spice_kernels: LoadedKernels) -> None:
    et = utc_to_tdb_s(utc, spice_kernels)
    expected = utc if "." in utc else utc + ".000000"
    assert tdb_s_to_utc(et, spice_kernels, decimals=6) == expected.ljust(26, "0")


def test_conversion_is_labelled_utc_independent_of_spice_default_system(
    spice_kernels: LoadedKernels,
) -> None:
    # SPICE reads an unlabelled ISO string in the TIMDEF default system. The converter labels
    # its strings "UTC", so changing the default (global SPICE state) must not change results;
    # an unlabelled string would shift by TDB - UTC = 69.18 s here.
    ref = utc_to_tdb_s("2030-01-01T00:00:00", spice_kernels)
    spiceypy.timdef("SET", "SYSTEM", 10, "TDB")
    try:
        assert utc_to_tdb_s("2030-01-01T00:00:00", spice_kernels) == ref
        assert spiceypy.str2et("2030-01-01T00:00:00") == pytest.approx(ref - 69.184, abs=2e-3)
    finally:
        spiceypy.timdef("SET", "SYSTEM", 10, "UTC")


@pytest.mark.parametrize(
    ("bad", "fragment"),
    [
        ("2030-01-01T00:00:00+01:00", "ISO-8601 UTC"),
        ("2016-12-31T23:59:60", "leap-second"),
        ("2030-02-30T00:00:00", "not a valid calendar epoch"),
        ("2030-01-01 00:00:00 TDB", "ISO-8601 UTC"),
    ],
)
def test_invalid_utc_strings_are_rejected(
    bad: str, fragment: str, spice_kernels: LoadedKernels
) -> None:
    with pytest.raises(ValueError, match=fragment):
        utc_to_tdb_s(bad, spice_kernels)


def test_mvp_output_grid_is_uniform_in_tdb(spice_kernels: LoadedKernels) -> None:
    cfg = load_config(MVP_CONFIG)
    grid = time_grid_from_config(cfg.time, spice_kernels)
    assert grid.scale == "TDB"
    assert len(grid) == 8767  # 365.25 d / 3600 s = 8766 intervals
    t = grid.epochs_tdb_s
    assert t[0] == utc_to_tdb_s("2030-01-01T00:00:00", spice_kernels)
    assert np.all(np.diff(t) == 3600.0)
    assert t[-1] - t[0] == 365.25 * 86400.0
    assert grid.start_utc == "2030-01-01T00:00:00"
    with pytest.raises(ValueError, match="read-only"):
        grid.epochs_tdb_s[0] = 0.0


def test_epoch_helpers(spice_kernels: LoadedKernels) -> None:
    e = utc_to_epoch("2000-01-01T11:58:55.816", spice_kernels)
    assert e.scale == "TDB"
    assert abs(e.tdb_s) < 1e-3  # J2000 TDB is 11:58:55.816 UTC (NAIF Time Required Reading)
    assert tdb_s_to_jd_tdb(0.0) == 2451545.0
    assert tdb_s_to_jd_tdb(86400.0) == 2451546.0


def test_conversions_outside_the_loaded_scope_raise(kernel_set: KernelSet) -> None:
    with kernel_set.loaded(verify=False) as k:
        pass
    with pytest.raises(KernelPoolError):
        utc_to_tdb_s("2030-01-01T00:00:00", k)
    with pytest.raises(KernelPoolError):
        tdb_s_to_utc(0.0, k)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tdb_s": math.nan},
        {"tdb_s": 0.0, "scale": "UTC"},
    ],
)
def test_epoch_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        Epoch(**kwargs)  # type: ignore[arg-type]


def test_time_grid_validation() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TimeGrid(epochs_tdb_s=np.array([0.0, 0.0]), step_s=1.0, start_utc="x")
    with pytest.raises(ValueError, match="non-empty"):
        TimeGrid(epochs_tdb_s=np.array([]), step_s=1.0, start_utc="x")
    with pytest.raises(ValueError, match="TDB"):
        TimeGrid(epochs_tdb_s=np.array([0.0]), step_s=1.0, start_utc="x", scale="UTC")  # type: ignore[arg-type]
