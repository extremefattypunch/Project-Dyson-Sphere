"""Stage-B bookkeeping of :mod:`helioswarm.constellation` on hand-built arrays (no SPICE).

* the per-satellite primary reason behind the PDF §11.4 "why power is unavailable" breakdown
  (:func:`satellite_reason_index`), against expected values written out by hand from the
  documented rule (first reason of each link in precedence order, largest index over sites);
* the reception-time reconstruction of the emission-epoch aggregation approximation
  (:func:`reception_time_aggregates`), against interval arithmetic done by hand.

All comparisons are exact (integers and sums of a few binary-exact numbers) except where noted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helioswarm.constellation import (
    DELIVERING,
    LINK_REASONS,
    reception_time_aggregates,
    satellite_reason_index,
)
from helioswarm.scheduler import SiteCapacity
from helioswarm.system import Unavailability

U = Unavailability

IDX = {r: k for k, r in enumerate(LINK_REASONS)}


def test_link_reasons_precedence_order_is_the_documented_one() -> None:
    assert [r.name for r in LINK_REASONS] == [
        "BELOW_MIN_ELEVATION",
        "EARTH_OCCULTATION",
        "SUN_OCCULTATION",
        "MOON_OCCULTATION",
        "SUN_AVOIDANCE",
        "ECLIPSE",
        "THERMAL",
        "WEATHER",
        "POINTING",
        "POWER_DENSITY_ABORT",
        "DUTY_CYCLE",
        "RECEIVER_CAPACITY",
        "SCHEDULER_NOT_SELECTED",
    ]


def test_satellite_primary_reason_is_the_furthest_link_by_first_reason() -> None:
    """One epoch, 8 satellites, 2 sites (A, B). Expected primary reasons by hand:

    sat 0: A = elevation|Earth occultation (-> elevation), B = eclipse        -> ECLIPSE
    sat 1: A = scheduler not selected, B = elevation      -> SCHEDULER_NOT_SELECTED
    sat 2: A = receiver capacity, B = weather|pointing (-> weather)          -> RECEIVER_CAPACITY
    sat 3: assigned to B (saturated flag on its link)                         -> DELIVERING
    sat 4: A = eclipse|thermal (-> eclipse), B = eclipse|elevation (-> elev.) -> ECLIPSE
    sat 5: A = duty cycle, B = duty cycle                                     -> DUTY_CYCLE
    sat 6: A = thermal|weather (-> thermal), B = PD abort|pointing (-> pointing) -> POINTING
    sat 7: A = receiver capacity, B = scheduler not selected -> SCHEDULER_NOT_SELECTED

    A "last reason wins" precedence would give THERMAL for sat 4 and POWER_DENSITY_ABORT for
    sat 6; a minimum over sites would give BELOW_MIN_ELEVATION for sats 0, 1, 4."""
    code_a = [
        U.BELOW_MIN_ELEVATION | U.EARTH_OCCULTATION,
        U.SCHEDULER_NOT_SELECTED,
        U.RECEIVER_CAPACITY,
        U.BELOW_MIN_ELEVATION,
        U.ECLIPSE | U.THERMAL,
        U.DUTY_CYCLE,
        U.THERMAL | U.WEATHER,
        U.RECEIVER_CAPACITY,
    ]
    code_b = [
        U.ECLIPSE,
        U.BELOW_MIN_ELEVATION,
        U.WEATHER | U.POINTING,
        U.RECEIVER_SATURATION,
        U.ECLIPSE | U.BELOW_MIN_ELEVATION,
        U.DUTY_CYCLE,
        U.POWER_DENSITY_ABORT | U.POINTING,
        U.SCHEDULER_NOT_SELECTED,
    ]
    codes = np.array([[[int(a), int(b)] for a, b in zip(code_a, code_b, strict=True)]], np.uint32)
    assigned = np.array([[-1, -1, -1, 1, -1, -1, -1, -1]])
    got = satellite_reason_index(codes, assigned)
    expected = [
        IDX[U.ECLIPSE],
        IDX[U.SCHEDULER_NOT_SELECTED],
        IDX[U.RECEIVER_CAPACITY],
        DELIVERING,
        IDX[U.ECLIPSE],
        IDX[U.DUTY_CYCLE],
        IDX[U.POINTING],
        IDX[U.SCHEDULER_NOT_SELECTED],
    ]
    np.testing.assert_array_equal(got[0], expected)
    with pytest.raises(RuntimeError, match="unattributed"):
        satellite_reason_index(np.zeros((1, 1, 2), np.uint32), np.array([[-1]]))


def _single_site(k: int | None, limit: float, dmax: float | None) -> list[SiteCapacity]:
    return [SiteCapacity(max_beams=k, max_accepted_W=limit, max_power_density_W_m2=dmax)]


def test_reception_time_two_beams_with_different_light_times() -> None:
    """Epochs 0, 100, 200 s (step 100 s); satellites 0 and 1 are both scheduled to the site at
    emission epoch 0 only, with constant light times 10 s and 50 s, P = 6 and 8 W, densities
    0.6 and 0.7; site: 1 slot, L = 10 W, D = 1.0. Arrivals: [10, 110) and [50, 150).

    slot 0 [0, 100): [10, 50) one beam (6 W), [50, 100) two beams (14 W, density 1.3)
        incident 6*40 + 14*50 = 940 J, accepted 6*40 + 10*50 = 740 J, overruns 50 s
    slot 1 [100, 200): [100, 110) two beams, [110, 150) beam 1 (8 W)
        incident 14*10 + 8*40 = 460 J, accepted 10*10 + 8*40 = 420 J, overruns 10 s
    Total incident 1400 J = (6 + 8) * 100 (energy is shifted, not created); emission-grid
    accepted (hold) min(14, 10) * 100 = 1000 J vs 1160 J in reception time."""
    t = np.array([0.0, 100.0, 200.0])
    assigned = np.array([[0, 0], [-1, -1], [-1, -1]])
    tau = np.zeros((3, 2, 1))
    tau[:, 0, 0] = 10.0
    tau[:, 1, 0] = 50.0
    p = np.zeros((3, 2, 1))
    p[0, :, 0] = [6.0, 8.0]
    d = np.zeros((3, 2, 1))
    d[0, :, 0] = [0.6, 0.7]
    r = reception_time_aggregates(t, 100.0, assigned, tau, p, d, _single_site(1, 10.0, 1.0))
    np.testing.assert_array_equal(r.light_time_spread_s[:, 0], [40.0, 0.0, 0.0])
    np.testing.assert_array_equal(r.max_beams[:, 0], [2, 2, 0])
    np.testing.assert_allclose(r.max_density_W_m2[:, 0], [1.3, 1.3, 0.0], rtol=1e-15)
    np.testing.assert_array_equal(r.incident_energy_J[:, 0], [940.0, 460.0, 0.0])
    np.testing.assert_array_equal(r.accepted_energy_J[:, 0], [740.0, 420.0, 0.0])
    np.testing.assert_array_equal(r.beam_overrun_s[:, 0], [50.0, 10.0, 0.0])
    np.testing.assert_array_equal(r.density_overrun_s[:, 0], [50.0, 10.0, 0.0])


def test_reception_time_consecutive_beams_of_one_satellite_abut_without_overlap() -> None:
    """One satellite scheduled at all three epochs with light times 10, 12, 12 s: arrivals
    [10, 112), [112, 212), [212, 312) (the last held for one step). Never two beams at once;
    slot energies 90 P, 100 P and 112 P (the open-ended last slot), total 302 P = the
    emission energy 300 P plus the 2 s light-time stretch."""
    t = np.array([0.0, 100.0, 200.0])
    tau = np.array([10.0, 12.0, 12.0]).reshape(3, 1, 1)
    p = np.full((3, 1, 1), 3.0)
    r = reception_time_aggregates(
        t,
        100.0,
        np.zeros((3, 1), int),
        tau,
        p,
        np.zeros((3, 1, 1)),
        _single_site(1, math.inf, None),
    )
    np.testing.assert_array_equal(r.max_beams[:, 0], [1, 1, 1])
    np.testing.assert_array_equal(r.incident_energy_J[:, 0], [270.0, 300.0, 336.0])
    np.testing.assert_array_equal(r.beam_overrun_s[:, 0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(r.light_time_spread_s[:, 0], [0.0, 0.0, 0.0])


def test_reception_time_without_light_time_equals_the_emission_grid() -> None:
    """tau = 0 (models.light_time = false): reception slots coincide with emission steps, so
    beams, densities and energies equal the emission-grid hold values exactly."""
    rng = np.random.default_rng(5)
    t = np.arange(6) * 50.0
    assigned = rng.integers(-1, 2, (6, 4))
    p = rng.uniform(0, 5, (6, 4, 2))
    d = rng.uniform(0, 1, (6, 4, 2))
    caps = [SiteCapacity(None, 7.0), SiteCapacity(3, math.inf)]
    r = reception_time_aggregates(t, 50.0, assigned, np.zeros(p.shape), p, d, caps)
    for j, cap in enumerate(caps):
        on = assigned == j
        inc = np.where(on, p[:, :, j], 0.0).sum(axis=1)
        np.testing.assert_array_equal(r.max_beams[:, j], on.sum(axis=1))
        np.testing.assert_allclose(r.incident_energy_J[:, j], inc * 50.0, rtol=1e-13)
        np.testing.assert_allclose(
            r.accepted_energy_J[:, j], np.minimum(inc, cap.max_accepted_W) * 50.0, rtol=1e-13
        )
