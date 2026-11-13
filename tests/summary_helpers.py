"""Synthetic saved time series for summary tests (importable as ``summary_helpers``)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from helioswarm.link_budget import STAGE_COLUMNS
from helioswarm.metrics import SummaryInputs

INPUTS = SummaryInputs(
    project_name="t", config_hash="0" * 64, transmitter_mode="optical", output_step_s=10.0,
    rated_power_W=100.0, rated_power_definition="def", reference_range_m=1.0,
    reference_sun_distance_m=1.0, reference_eta_capture=0.1, reference_eta_atmosphere=1.0,
    reference_thermal_feasible=True, reference_power_density_abort=False,
    max_power_density_W_m2=None,
    max_input_power_W=1e12, collector_area_m2=10.0,
    earth_occultation_checked=True, sun_occultation_checked=True, moon_occultation_checked=True,
)  # fmt: skip
"""Run scalars of the synthetic summaries."""

LAYOUT1_SUMMARY = (
    Path(__file__).resolve().parent / "data" / "summary_layout1_m6_optical_sweep_p0000.json"
)
"""Unmodified ``summary.json`` of an M6 run in layout ``helioswarm-summary/1`` (before M6b):
point p0000 of ``$HELIOSWARM_RUNS_DIR/optical_1au_sweep`` (Slurm array 60127591, task 0,
job 60127633)."""


def synthetic_table(n: int, step: float, **cols: Any) -> pa.Table:
    """A ``timeseries.parquet``-like table with ``n`` samples at ``step`` s; ``cols`` override
    the neutral defaults (no access, zero power)."""
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
        "earth_blocked": np.zeros(n, dtype=bool),
        "sun_blocked": np.zeros(n, dtype=bool),
        "moon_blocked": np.zeros(n, dtype=bool),
        "light_time_s": np.full(n, 567.0),
        "range_rate_m_s": np.zeros(n),
        "pointing_rate_rad_s": np.zeros(n),
        "orbit_drift_vs_l0_m": np.zeros(n),
        "transmit_dir_x": np.ones(n),
        "transmit_dir_y": np.zeros(n),
        "transmit_dir_z": np.zeros(n),
    }
    for _, attr in STAGE_COLUMNS:
        base.setdefault(attr, np.zeros(n))
    base.update(cols)
    return pa.table({k: pa.array(np.asarray(v)) for k, v in base.items()})
