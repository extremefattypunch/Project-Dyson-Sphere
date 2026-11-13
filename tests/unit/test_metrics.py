"""Run metrics (helioswarm.metrics) on synthetic time series with hand-computed references."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from helioswarm.metrics import (
    LEGACY_SUMMARY_SCHEMA,
    SUMMARY_SCHEMA,
    RunSummary,
    outage_statistics,
    read_summary,
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
from summary_helpers import INPUTS
from summary_helpers import LAYOUT1_SUMMARY as LAYOUT1_FIXTURE
from summary_helpers import synthetic_table as _table


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
    assert s.delivered_energy_J == 900.0
    assert s.p_grid_W.time_mean == 900.0 / 40.0
    assert s.capacity_factor == pytest.approx(22.5 / 100.0, rel=1e-15)
    # Available weight 5 + 10 + 10 + 5 = 30 of 40.
    assert s.availability == 30.0 / 40.0
    blk = s.unavailability.blocking
    assert blk["EARTH_OCCULTATION"].time_fraction_primary == 10.0 / 40.0
    total_primary = sum(v.time_fraction_primary for v in blk.values())
    assert total_primary == pytest.approx(1.0 - s.availability, abs=1e-15)
    assert s.p_grid_W.peak == 50.0
    assert s.p_grid_W.median == 20.0
    # End-to-end: 900 J / (1e7 W * 40 s).
    assert s.end_to_end_efficiency.energy_ratio == pytest.approx(900.0 / 4.0e8, rel=1e-15)
    assert s.outages.count == 1
    assert s.outages.longest_s == 10.0


def test_summary_capacity_factor_undefined_without_rated_power() -> None:
    import dataclasses

    s = summarize_timeseries(_table(3, 10.0), dataclasses.replace(INPUTS, rated_power_W=0.0))
    assert s.capacity_factor is None


def test_aliasing_diagnostic_from_hour_angles() -> None:
    n = 48
    # Hourly samples of a 24 h cycle: only 24 distinct phases -> gap 2 pi / 24, warning.
    ha = np.mod(2 * math.pi * np.arange(n) / 24.0, 2 * math.pi)
    s = summarize_timeseries(_table(n, 3600.0, hour_angle_rad=ha), INPUTS)
    al = s.diurnal_aliasing
    assert al.hour_angle_max_gap_rad == pytest.approx(2 * math.pi / 24, rel=1e-12)
    assert al.effective_distinct_diurnal_phases == pytest.approx(24.0, rel=1e-12)
    assert al.warning is True
    dense = np.linspace(0.0, 2 * math.pi, 2000, endpoint=False)
    s2 = summarize_timeseries(_table(2000, 300.0, hour_angle_rad=dense), INPUTS)
    assert s2.diurnal_aliasing.warning is False


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


def test_orbital_diagnostics_hand_computed() -> None:
    # PDF §7.3 outputs from the saved columns. Trapezoid weights 5, 10, 10, 10, 5 (window 40 s).
    n = 5
    earth = np.array([1, 0, 0, 0, 1], dtype=bool)
    sun = np.array([0, 1, 0, 0, 0], dtype=bool)
    moon = np.array([0, 1, 0, 0, 1], dtype=bool)
    light = np.array([500.0, 510.0, 520.0, 530.0, 545.0])
    rr = np.array([-3.0, -1.0, 0.0, 2.0, 7.0])
    rate = np.array([1e-7, 2e-7, 3e-7, 4e-7, 5e-7])
    drift = np.array([0.0, 1.0, 4.0, 9.0, 8.0])
    u_last = np.array([0.0, 0.6, 0.8])
    s = summarize_timeseries(
        _table(n, 10.0, earth_blocked=earth, sun_blocked=sun, moon_blocked=moon,
               visible=~(earth | sun | moon), light_time_s=light, range_rate_m_s=rr,
               pointing_rate_rad_s=rate, orbit_drift_vs_l0_m=drift,
               range_m=np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e11,
               transmit_dir_x=np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
               transmit_dir_y=np.array([0.0, 0.0, 0.0, 0.0, 0.6]),
               transmit_dir_z=np.array([0.0, 0.0, 0.0, 0.0, 0.8])),
        INPUTS,
    )  # fmt: skip
    od = s.orbital_diagnostics
    assert od is not None
    assert od.earth_occultation_fraction == (5.0 + 5.0) / 40.0
    assert od.sun_occultation_fraction == 10.0 / 40.0
    assert od.moon_occultation_fraction == (10.0 + 5.0) / 40.0
    assert od.occultation_fraction == (5.0 + 10.0 + 5.0) / 40.0  # union, not the sum (35/40)
    assert od.access_fraction == 20.0 / 40.0
    assert (od.light_time.min_s, od.light_time.max_s) == (500.0, 545.0)
    assert od.light_time.mean_s == pytest.approx(521.0, rel=1e-15)  # unweighted sample mean
    assert (od.range_rate_m_s.min, od.range_rate_m_s.max) == (-3.0, 7.0)
    assert od.pointing_rate_rad_s.max == 5e-7
    assert (od.orbit_drift_vs_l0_m_max, od.orbit_drift_vs_l0_m_final) == (9.0, 8.0)
    assert od.earth_range_m.max == 5e11
    assert od.beam_direction_j2000_unit_first == (1.0, 0.0, 0.0)
    np.testing.assert_array_equal(od.beam_direction_j2000_unit_last, u_last)
    # No L0 reference (NaN column) -> drift reported as undefined, not as NaN or 0.
    s2 = summarize_timeseries(_table(n, 10.0, orbit_drift_vs_l0_m=np.full(n, np.nan)), INPUTS)
    assert s2.orbital_diagnostics is not None
    assert s2.orbital_diagnostics.orbit_drift_vs_l0_m_max is None


def test_disabled_occultation_check_is_reported_as_not_evaluated() -> None:
    # AccessResult: a disabled check is not evaluated and its flag is False. The summary must
    # say "not evaluated" (None), not "never blocked" (0.0); the union covers checked bodies.
    n = 5
    earth = np.array([1, 0, 0, 0, 1], dtype=bool)
    moon = np.array([0, 1, 0, 0, 1], dtype=bool)
    table = _table(n, 10.0, earth_blocked=earth, moon_blocked=moon)
    no_sun = dataclasses.replace(INPUTS, sun_occultation_checked=False)
    od = summarize_timeseries(table, no_sun).orbital_diagnostics
    assert od is not None
    assert od.sun_occultation_fraction is None
    assert od.earth_occultation_fraction == 10.0 / 40.0
    assert od.occultation_fraction == 20.0 / 40.0  # earth | moon: weights 5 + 10 + 5
    no_moon = dataclasses.replace(INPUTS, moon_occultation_checked=False)
    od = summarize_timeseries(table, no_moon).orbital_diagnostics
    assert od is not None
    assert od.moon_occultation_fraction is None
    assert od.occultation_fraction == 10.0 / 40.0  # Moon-only epoch 1 is not counted
    none = dataclasses.replace(
        INPUTS,
        earth_occultation_checked=False,
        sun_occultation_checked=False,
        moon_occultation_checked=False,
    )
    od = summarize_timeseries(table, none).orbital_diagnostics
    assert od is not None
    assert (od.occultation_fraction, od.earth_occultation_fraction) == (None, None)
    assert json.loads(json.dumps(od.model_dump(mode="json")))["occultation_fraction"] is None


def test_layout1_summaries_written_before_m6b_are_still_read() -> None:
    # M6b.md: summaries of layout helioswarm-summary/1 (M6 runs and sweeps) must stay readable
    # for collation; an unmodified M6 file is the fixture.
    doc = json.loads(LAYOUT1_FIXTURE.read_text(encoding="utf-8"))
    assert doc["summary_schema"] == LEGACY_SUMMARY_SCHEMA
    assert "orbital_diagnostics" not in doc
    s = read_summary(LAYOUT1_FIXTURE)
    assert s.summary_schema == LEGACY_SUMMARY_SCHEMA
    assert s.orbital_diagnostics is None
    assert s.availability == doc["availability"]
    # The current layout requires the diagnostics; unknown layouts are rejected.
    with pytest.raises(ValidationError, match="orbital_diagnostics is required"):
        RunSummary.model_validate({**doc, "summary_schema": SUMMARY_SCHEMA})
    with pytest.raises(ValidationError, match="unknown summary_schema"):
        RunSummary.model_validate({**doc, "summary_schema": "helioswarm-summary/3"})


def test_summary_record_is_typed_and_round_trips(tmp_path: Path) -> None:
    s = summarize_timeseries(_table(4, 10.0, grid_W=np.array([1.0, 2.0, 3.0, 4.0])), INPUTS)
    assert isinstance(s, RunSummary)
    assert s.summary_schema == SUMMARY_SCHEMA == "helioswarm-summary/2"
    doc = s.to_json_dict()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert read_summary(path) == s
    with pytest.raises(ValidationError):  # unknown key
        RunSummary.model_validate({**doc, "p_grid_mean": 1.0})
    with pytest.raises(ValidationError):  # missing required section
        RunSummary.model_validate({k: v for k, v in doc.items() if k != "p_grid_W"})
    with pytest.raises(ValidationError):  # frozen record
        s.availability = 0.5  # type: ignore[misc]
    # A one-sample window has no time mean: None, not NaN (summary.json stays valid JSON).
    one = summarize_timeseries(_table(1, 10.0), INPUTS)
    assert one.p_grid_W.time_mean is None
    assert one.capacity_factor is None
