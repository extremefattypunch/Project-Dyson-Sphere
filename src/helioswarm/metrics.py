"""Run metrics (PDF §10.5, §11.4) computed from the saved time series.

Every metric is computed from the columns written to ``timeseries.parquet`` (via
:func:`summarize_timeseries`), so ``summary.json`` can always be regenerated from saved data
(PDF §13.4).

Sampling convention (one definition used by every time integral and fraction)
------------------------------------------------------------------------------
The output grid is TDB-uniform with step ``dt`` (emission epochs ``t_0 .. t_N``, window
``T = N dt``). Sample ``k`` represents the interval ``[t_k - dt/2, t_k + dt/2]`` clipped to the
window, i.e. weight ``w_k = dt`` for interior samples and ``dt/2`` for the two end samples
(``sum w_k = T``). Then

* delivered energy ``E = sum w_k P_grid,k`` (composite trapezoidal rule) [J];
* time-mean power ``<P_grid> = E / T`` [W];
* availability ``= sum_{available} w_k / T`` (fraction of time all conditions are met);
* reason fractions use the same weights;
* an outage is a maximal run of consecutive unavailable samples; its duration is the sum of
  their weights [s] (resolution +- one step at each end: the true transition lies somewhere
  between two samples); outages touching the window start or end are *censored* and flagged.

Percentiles (median, p05, p95) are unweighted sample percentiles (numpy ``linear``
interpolation); with ``N + 1`` samples the end weights change them by < 1/N.

Times are emission epochs at the satellite. Power is delivered at the arrival epochs
``t_a = t_e + tau``; ``dt_a/dt_e = 1 + d tau/d t_e`` differs from 1 by ``|range rate|/c``
(<= 1e-5 for the MVP), and the Doppler factor on the photon energy is of the same order, so the
received-power time axis and its energy are equal to the emission-axis values to ~1e-5
(neglected; named approximation).

Capacity factor (PDF §10.5: "define the denominator clearly")
    ``capacity_factor = <P_grid> / rated_power_W`` with ``rated_power_W`` the reference link of
    :class:`helioswarm.system.ReferenceLink` (definition string copied into summary.json).

End-to-end efficiency (PDF §10.5)
    ``E_grid / E_incident`` with ``E_incident`` the time integral of the sunlight incident on
    the collector (``incident_solar_W``), and, separately, the mean over *available* epochs of
    ``P_grid / P_incident``.

Diurnal aliasing diagnostic (M3 finding)
    The satellite's hour angle at the site repeats with a period within minutes of the solar
    day, so a step that divides the day samples nearly the same hour angles every day. Reported:
    the largest unsampled hour-angle gap ``g`` (``visibility.max_circular_gap_rad`` of the saved
    ``hour_angle_rad`` column), the effective number of distinct diurnal phases ``2 pi / g``,
    the per-edge resolution ``g / (2 pi)`` of diurnal fractions (an order-of-magnitude guide; M3
    measured a 0.020 bias at ``g/2pi = 0.024``) and a warning flag when ``g / 2pi`` exceeds
    ``ALIASING_WARNING_FRACTION``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
import pyarrow as pa
from numpy.typing import ArrayLike, NDArray

from helioswarm.link_budget import STAGE_COLUMNS
from helioswarm.system import (
    ALIASING_WARNING_FRACTION,
    BLOCKING_REASONS,
    LIMITING_FLAGS,
    primary_reason,
)
from helioswarm.visibility import max_circular_gap_rad

Float64Array = NDArray[np.float64]

SUMMARY_SCHEMA: Final[str] = "helioswarm-summary/1"
"""Version tag of the summary.json layout."""

OUTAGE_HISTOGRAM_EDGES_S: Final[tuple[float, ...]] = (
    0.0,
    3600.0,
    3.0 * 3600.0,
    6.0 * 3600.0,
    12.0 * 3600.0,
    24.0 * 3600.0,
    3.0 * 86400.0,
    10.0 * 86400.0,
    math.inf,
)
"""Bin edges of the outage-duration histogram [s] (1 h, 3 h, 6 h, 12 h, 1 d, 3 d, 10 d)."""


@dataclass(frozen=True)
class SummaryInputs:
    """Scalars the summary needs besides the time series (all from the run, written once).

    Attributes:
        project_name: ``project.name``.
        config_hash: Configuration hash.
        transmitter_mode: ``optical`` or ``microwave``.
        output_step_s: Grid step ``dt`` [s].
        rated_power_W: Capacity-factor denominator [W].
        rated_power_definition: Its definition (:data:`helioswarm.system.REFERENCE_DEFINITION`).
        reference_range_m: Reference range of the rated link [m].
        reference_sun_distance_m: Reference heliocentric distance [m].
        reference_eta_capture: Expected capture at reference [1].
        reference_eta_atmosphere: Atmospheric transmission at reference [1].
        reference_thermal_feasible: Thermal verdict at the reference operating point.
        reference_power_density_abort: The reference beam is aborted (power density).
        max_power_density_W_m2: Configured beam-abort limit or ``None`` [W m^-2].
        max_input_power_W: Receiver input limit [W].
        collector_area_m2: Collector area [m^2].
    """

    project_name: str
    config_hash: str
    transmitter_mode: str
    output_step_s: float
    rated_power_W: float
    rated_power_definition: str
    reference_range_m: float
    reference_sun_distance_m: float
    reference_eta_capture: float
    reference_eta_atmosphere: float
    reference_thermal_feasible: bool
    reference_power_density_abort: bool
    max_power_density_W_m2: float | None
    max_input_power_W: float
    collector_area_m2: float


@dataclass(frozen=True)
class OutageStatistics:
    """Outage count and duration distribution (PDF §10.5).

    Attributes:
        count: Number of outages (maximal unavailable runs).
        total_s: Total unavailable time [s].
        mean_s, median_s, p95_s, longest_s: Duration statistics [s] (0 if no outage).
        longest_start_tdb_s: Emission epoch of the first sample of the longest outage [TDB s]
            (``None`` if no outage).
        censored_count: Outages touching the window start or end (true duration unknown).
        histogram_edges_s: Bin edges [s] (last edge ``inf`` written as ``null``).
        histogram_counts: Outages per bin.
        resolution_s: Duration uncertainty at each end of an outage (one step) [s].
    """

    count: int
    total_s: float
    mean_s: float
    median_s: float
    p95_s: float
    longest_s: float
    longest_start_tdb_s: float | None
    censored_count: int
    histogram_edges_s: tuple[float | None, ...]
    histogram_counts: tuple[int, ...]
    resolution_s: float


def sample_weights_s(n_samples: int, step_s: float) -> Float64Array:
    """Trapezoid sample weights ``w_k`` [s] (module docstring); ``sum = (n - 1) step``."""
    if n_samples < 1:
        raise ValueError("need at least one sample")
    w = np.full(n_samples, float(step_s))
    if n_samples == 1:
        return np.zeros(1)
    w[0] = w[-1] = 0.5 * step_s
    return w


def outage_statistics(
    available: ArrayLike, epochs_tdb_s: ArrayLike, step_s: float
) -> OutageStatistics:
    """Outages = maximal runs of unavailable samples, durations = sums of sample weights.

    Args:
        available: Availability per sample (bool).
        epochs_tdb_s: Sample epochs [TDB s].
        step_s: Grid step [s].

    Returns:
        :class:`OutageStatistics`.
    """
    av = np.asarray(available, dtype=bool)
    t = np.asarray(epochs_tdb_s, dtype=np.float64)
    w = sample_weights_s(av.size, step_s)
    down = ~av
    edges = np.diff(np.concatenate([[0], down.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    cs = np.concatenate([[0.0], np.cumsum(w)])
    durations = cs[ends] - cs[starts]
    censored = int(np.count_nonzero((starts == 0) | (ends == av.size)))
    hist_edges = np.asarray(OUTAGE_HISTOGRAM_EDGES_S)
    counts, _ = (
        np.histogram(durations, bins=hist_edges)
        if durations.size
        else (
            np.zeros(hist_edges.size - 1, dtype=int),
            None,
        )
    )
    if durations.size:
        i_long = int(np.argmax(durations))
        stats = (
            float(np.mean(durations)),
            float(np.median(durations)),
            float(np.percentile(durations, 95.0)),
            float(durations[i_long]),
            float(t[starts[i_long]]),
        )
    else:
        stats = (0.0, 0.0, 0.0, 0.0, None)  # type: ignore[assignment]
    return OutageStatistics(
        count=int(durations.size),
        total_s=float(np.sum(durations)),
        mean_s=stats[0],
        median_s=stats[1],
        p95_s=stats[2],
        longest_s=stats[3],
        longest_start_tdb_s=stats[4],
        censored_count=censored,
        histogram_edges_s=tuple(None if math.isinf(e) else float(e) for e in hist_edges),
        histogram_counts=tuple(int(c) for c in counts),
        resolution_s=float(step_s),
    )


def _stats(values: Float64Array) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "mean": None, "median": None, "p05": None, "p95": None, "max": None}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _col(table: pa.Table, name: str) -> NDArray[Any]:
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False))


def summarize_timeseries(table: pa.Table, inputs: SummaryInputs) -> dict[str, Any]:
    """The ``summary.json`` document from a saved time series (module docstring).

    Args:
        table: The ``timeseries.parquet`` table (``pyarrow.Table``; columns of
            :func:`helioswarm.io.timeseries_table`).
        inputs: Run scalars.

    Returns:
        JSON-ready nested mapping (the on-disk summary format; numbers in SI, unit-suffixed
        keys). This is an I/O document, not passed between models.
    """
    t = _col(table, "t_emit_tdb_s").astype(np.float64)
    n = t.size
    dt = inputs.output_step_s
    w = sample_weights_s(n, dt)
    window = float(np.sum(w))
    p_grid = _col(table, "grid_W").astype(np.float64)
    available = _col(table, "available").astype(bool)
    code = _col(table, "reason_code").astype(np.uint32)
    incident = _col(table, "incident_solar_W").astype(np.float64)
    energy = float(np.sum(w * p_grid))
    mean_p = energy / window if window > 0 else float("nan")
    rated = inputs.rated_power_W
    cf = mean_p / rated if rated > 0.0 else None

    stage_energy = {
        attr.replace("_W", "_J"): float(np.sum(w * _col(table, attr).astype(np.float64)))
        for _, attr in STAGE_COLUMNS
    }
    e_incident = float(np.sum(w * incident))
    with np.errstate(invalid="ignore", divide="ignore"):
        inst_eff = np.where(incident > 0.0, p_grid / np.where(incident > 0, incident, 1.0), 0.0)

    prim = primary_reason(code)
    blocking: dict[str, Any] = {}
    for r in BLOCKING_REASONS:
        bit = (code & np.uint32(int(r))) != 0
        blocking[str(r.name)] = {
            "time_fraction_flagged": float(np.sum(w[bit]) / window) if window else 0.0,
            "time_fraction_primary": float(np.sum(w[prim == r.name]) / window) if window else 0.0,
        }
    limiting = {
        str(r.name): float(np.sum(w[(code & np.uint32(int(r))) != 0]) / window) if window else 0.0
        for r in LIMITING_FLAGS
    }
    outages = outage_statistics(available, t, dt)

    transmitting = _col(table, "transmitting").astype(bool)
    density = _col(table, "peak_power_density_W_m2").astype(np.float64)
    requested = _col(table, "requested_peak_power_density_W_m2").astype(np.float64)
    capture = _col(table, "eta_capture").astype(np.float64)
    gap = max_circular_gap_rad(_col(table, "hour_angle_rad").astype(np.float64))
    visible = _col(table, "visible").astype(bool)

    pv_temp = _col(table, "pv_temperature_K").astype(np.float64)
    margin = _col(table, "thermal_margin_K").astype(np.float64)
    rad_area = _col(table, "radiator_area_m2").astype(np.float64)
    feasible = _col(table, "thermal_feasible").astype(bool)
    rng_m = _col(table, "range_m").astype(np.float64)
    sun_d = _col(table, "sun_distance_m").astype(np.float64)

    return {
        "summary_schema": SUMMARY_SCHEMA,
        "project_name": inputs.project_name,
        "config_hash": inputs.config_hash,
        "transmitter_mode": inputs.transmitter_mode,
        "time": {
            "n_samples": int(n),
            "output_step_s": dt,
            "window_s": window,
            "start_tdb_s": float(t[0]),
            "end_tdb_s": float(t[-1]),
            "sampling": "trapezoid weights (end samples dt/2); emission epochs (TDB)",
        },
        "p_grid_W": {
            "time_mean": mean_p,
            "median": float(np.median(p_grid)),
            "p05": float(np.percentile(p_grid, 5.0)),
            "p95": float(np.percentile(p_grid, 95.0)),
            "peak": float(np.max(p_grid)),
            "mean_when_available": float(np.mean(p_grid[available])) if available.any() else None,
            "percentile_method": "unweighted sample percentiles, numpy linear interpolation",
        },
        "delivered_energy_J": energy,
        "delivered_energy_definition": "sum_k w_k P_grid,k (trapezoid weights; emission epochs)",
        "availability": float(np.sum(w[available]) / window) if window else 0.0,
        "availability_definition": "time fraction with no blocking unavailability reason "
        "(all geometry, occultation, Sun-avoidance, eclipse, thermal, weather, pointing and "
        "power-density conditions met)",
        "capacity_factor": cf,
        "capacity_factor_denominator": {
            "rated_power_W": rated,
            "definition": inputs.rated_power_definition,
            "reference_range_m": inputs.reference_range_m,
            "reference_sun_distance_m": inputs.reference_sun_distance_m,
            "reference_eta_capture": inputs.reference_eta_capture,
            "reference_eta_atmosphere": inputs.reference_eta_atmosphere,
            "reference_thermal_feasible": inputs.reference_thermal_feasible,
            "reference_power_density_abort": inputs.reference_power_density_abort,
        },
        "end_to_end_efficiency": {
            "energy_ratio": energy / e_incident if e_incident > 0 else None,
            "energy_ratio_definition": "E_grid / E_incident_on_collector over the window",
            "mean_when_available": float(np.mean(inst_eff[available])) if available.any() else None,
            "incident_solar_energy_J": e_incident,
        },
        "stage_energy_J": stage_energy,
        "beam_capture": {
            "eta_capture_when_available": _stats(capture[available]),
            "eta_capture_when_visible": _stats(capture[visible]),
            "definition": "expected fraction of the beam power at the receiver plane intercepted "
            "by the receiver (diffraction, pointing, wavefront and lattice factors)",
        },
        "receiver_power_density_W_m2": {
            "incident_peak_when_transmitting": _stats(density[transmitting]),
            "requested_peak_max": float(np.max(requested)) if requested.size else None,
            "limit": inputs.max_power_density_W_m2,
            "abort_time_fraction": blocking["POWER_DENSITY_ABORT"]["time_fraction_flagged"],
            "definition": "peak irradiance of the aberration-free pattern on the beam-normal "
            "plane at the receiver (upper bound)",
        },
        "receiver_saturation_time_fraction": limiting["RECEIVER_SATURATION"],
        "max_input_power_W": inputs.max_input_power_W,
        "unavailability": {
            "blocking": blocking,
            "limiting_flags_time_fraction": limiting,
            "note": "time_fraction_flagged counts every epoch with the bit set (reasons "
            "overlap); time_fraction_primary assigns each unavailable epoch to its first "
            "reason in precedence order and sums to 1 - availability",
        },
        "outages": asdict(outages),
        "diurnal_aliasing": {
            "hour_angle_max_gap_rad": gap,
            "effective_distinct_diurnal_phases": 2.0 * math.pi / gap if gap > 0 else None,
            "per_edge_resolution_day_fraction": gap / (2.0 * math.pi),
            "warning": bool(gap / (2.0 * math.pi) > ALIASING_WARNING_FRACTION),
            "warning_threshold_day_fraction": ALIASING_WARNING_FRACTION,
            "access_fraction": float(np.sum(w[visible]) / window) if window else 0.0,
        },
        "thermal": {
            "feasible_time_fraction": float(np.sum(w[feasible]) / window) if window else 0.0,
            "pv_temperature_K": _stats(pv_temp),
            "binding_margin_K_min": float(np.min(margin)),
            "radiator_area_m2_max": float(np.max(rad_area)),
        },
        "geometry": {
            "range_m": _stats(rng_m),
            "sun_distance_m": _stats(sun_d),
        },
        "power_per_collector_area_W_m2": mean_p / inputs.collector_area_m2,
    }


__all__ = [
    "OUTAGE_HISTOGRAM_EDGES_S",
    "SUMMARY_SCHEMA",
    "OutageStatistics",
    "SummaryInputs",
    "outage_statistics",
    "sample_weights_s",
    "summarize_timeseries",
]
