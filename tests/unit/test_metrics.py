"""Run metrics (helioswarm.metrics) on synthetic time series with hand-computed references."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from helioswarm.metrics import (
    SummaryInputs,
    outage_statistics,
    sample_weights_s,
    summarize_timeseries,
)
from helioswarm.system import (
    BLOCKING_MASK,
    BLOCKING_REASONS,
    Unavailability,
    primary_reason,
    reason_names,
)


def test_sample_weights_are_trapezoid() -> None:
    w = sample_weights_s(5, 10.0)
    np.testing.assert_array_equal(w, [5.0, 10.0, 10.0, 10.0, 5.0])
    assert w.sum() == 40.0  # window = (n - 1) dt
    assert sample_weights_s(1, 10.0).tolist() == [0.0]


def test_outage_statistics_hand_computed() -> None:
    #           0  1  2  3  4  5  6  7  8  9
    av = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 1], dtype=bool)
    t = 100.0 * np.arange(10)
    st = outage_statistics(av, t, 100.0)
    # Runs: [0,1] (censored at start; weights 50 + 100), [4,5,6] (300), [8] (100).
    assert st.count == 3
    assert st.censored_count == 1
    assert st.total_s == 550.0
    assert st.longest_s == 300.0
    assert st.longest_start_tdb_s == 400.0
    assert st.median_s == 150.0
    assert st.histogram_counts[0] == 3  # all < 1 h
    assert st.resolution_s == 100.0
    none = outage_statistics(np.ones(4, dtype=bool), np.arange(4.0), 1.0)
    assert none.count == 0
    assert none.longest_s == 0.0
    assert none.longest_start_tdb_s is None


def _table(n: int, step: float, **cols: Any) -> pa.Table:
    t = 1.0e9 + step * np.arange(n)
    base: dict[str, Any] = {
        "t_emit_tdb_s": t,
        "grid_W": np.zeros(n),
        "available": np.zeros(n, dtype=bool),
        "reason_code": np.zeros(n, dtype=np.uint32),
        "incident_solar_W": np.full(n, 1.0e7),
        "transmitting": np.zeros(n, dtype=bool),
        "peak_power_density_W_m2": np.zeros(n),
        "requested_peak_power_density_W_m2": np.zeros(n),
        "eta_capture": np.zeros(n),
        "hour_angle_rad": np.linspace(0.0, 2 * math.pi, n, endpoint=False),
        "visible": np.zeros(n, dtype=bool),
        "pv_temperature_K": np.full(n, 300.0),
        "thermal_margin_K": np.full(n, 100.0),
        "radiator_area_m2": np.full(n, 400.0),
        "thermal_feasible": np.ones(n, dtype=bool),
        "range_m": np.full(n, 1.7e11),
        "sun_distance_m": np.full(n, 1.5e11),
    }
    for _, attr in __import__("helioswarm.link_budget", fromlist=["x"]).STAGE_COLUMNS:
        base.setdefault(attr, np.zeros(n))
    base.update(cols)
    return pa.table({k: pa.array(np.asarray(v)) for k, v in base.items()})


INPUTS = SummaryInputs(
    project_name="t", config_hash="0" * 64, transmitter_mode="optical", output_step_s=10.0,
    rated_power_W=100.0, rated_power_definition="def", reference_range_m=1.0,
    reference_sun_distance_m=1.0, reference_eta_capture=0.1, reference_eta_atmosphere=1.0,
    reference_thermal_feasible=True, reference_power_density_abort=False,
    max_power_density_W_m2=None,
    max_input_power_W=1e12, collector_area_m2=10.0,
)  # fmt: skip


def test_summary_energy_availability_capacity_factor() -> None:
    n = 5
    grid = np.array([10.0, 20.0, 0.0, 40.0, 50.0])
    av = grid > 0
    code = np.where(av, 0, int(Unavailability.EARTH_OCCULTATION)).astype(np.uint32)
    s = summarize_timeseries(
        _table(n, 10.0, grid_W=grid, available=av, reason_code=code, transmitting=av,
               visible=av, eta_capture=np.where(av, 0.01, 0.0)),
        INPUTS,
    )  # fmt: skip
    # Trapezoid: 5*10 + 10*20 + 10*0 + 10*40 + 5*50 = 900 J over a 40 s window.
    assert s["delivered_energy_J"] == 900.0
    assert s["p_grid_W"]["time_mean"] == 900.0 / 40.0
    assert s["capacity_factor"] == pytest.approx(22.5 / 100.0, rel=1e-15)
    # Available weight 5 + 10 + 10 + 5 = 30 of 40.
    assert s["availability"] == 30.0 / 40.0
    blk = s["unavailability"]["blocking"]
    assert blk["EARTH_OCCULTATION"]["time_fraction_primary"] == 10.0 / 40.0
    total_primary = sum(v["time_fraction_primary"] for v in blk.values())
    assert total_primary == pytest.approx(1.0 - s["availability"], abs=1e-15)
    assert s["p_grid_W"]["peak"] == 50.0
    assert s["p_grid_W"]["median"] == 20.0
    # End-to-end: 900 J / (1e7 W * 40 s).
    assert s["end_to_end_efficiency"]["energy_ratio"] == pytest.approx(900.0 / 4.0e8, rel=1e-15)
    assert s["outages"]["count"] == 1
    assert s["outages"]["longest_s"] == 10.0


def test_summary_capacity_factor_undefined_without_rated_power() -> None:
    import dataclasses

    s = summarize_timeseries(_table(3, 10.0), dataclasses.replace(INPUTS, rated_power_W=0.0))
    assert s["capacity_factor"] is None


def test_aliasing_diagnostic_from_hour_angles() -> None:
    n = 48
    # Hourly samples of a 24 h cycle: only 24 distinct phases -> gap 2 pi / 24, warning.
    ha = np.mod(2 * math.pi * np.arange(n) / 24.0, 2 * math.pi)
    s = summarize_timeseries(_table(n, 3600.0, hour_angle_rad=ha), INPUTS)
    al = s["diurnal_aliasing"]
    assert al["hour_angle_max_gap_rad"] == pytest.approx(2 * math.pi / 24, rel=1e-12)
    assert al["effective_distinct_diurnal_phases"] == pytest.approx(24.0, rel=1e-12)
    assert al["warning"] is True
    dense = np.linspace(0.0, 2 * math.pi, 2000, endpoint=False)
    s2 = summarize_timeseries(_table(2000, 300.0, hour_angle_rad=dense), INPUTS)
    assert s2["diurnal_aliasing"]["warning"] is False


def test_reason_codes_and_precedence() -> None:
    c = np.array(
        [
            0,
            int(Unavailability.THERMAL | Unavailability.EARTH_OCCULTATION),
            int(Unavailability.RECEIVER_SATURATION),
            int(Unavailability.WEATHER | Unavailability.POWER_DENSITY_ABORT),
        ],
        dtype=np.uint32,
    )
    assert primary_reason(c).tolist() == ["", "EARTH_OCCULTATION", "", "WEATHER"]
    assert reason_names(int(c[1])) == ("EARTH_OCCULTATION", "THERMAL")
    # Limiting flags are not blocking.
    assert int(Unavailability.RECEIVER_SATURATION) & BLOCKING_MASK == 0
    assert int(Unavailability.PARTIAL_ECLIPSE) & BLOCKING_MASK == 0
    assert all(int(r) & BLOCKING_MASK for r in BLOCKING_REASONS)
