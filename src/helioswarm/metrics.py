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

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Self

import numpy as np
import pyarrow as pa
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from helioswarm.link_budget import STAGE_COLUMNS
from helioswarm.system import (
    ALIASING_WARNING_FRACTION,
    BLOCKING_REASONS,
    LIMITING_FLAGS,
    primary_reason,
)
from helioswarm.visibility import max_circular_gap_rad

Float64Array = NDArray[np.float64]

SUMMARY_SCHEMA: Final[str] = "helioswarm-summary/2"
"""Version tag of the summary.json layout."""

LEGACY_SUMMARY_SCHEMA: Final[str] = "helioswarm-summary/1"
"""Layout written before M6b (no ``orbital_diagnostics``); still readable."""

READABLE_SUMMARY_SCHEMAS: Final[tuple[str, ...]] = (LEGACY_SUMMARY_SCHEMA, SUMMARY_SCHEMA)
"""Layouts accepted by :func:`read_summary`."""

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
        earth_occultation_checked: ``models.earth_occultation`` (the check ran).
        sun_occultation_checked: ``models.sun_occultation``.
        moon_occultation_checked: ``models.moon_occultation``.
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
    earth_occultation_checked: bool
    sun_occultation_checked: bool
    moon_occultation_checked: bool


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


class _SummaryRecord(BaseModel):
    """Base of the typed ``summary.json`` records: unknown keys rejected, attributes cannot be
    reassigned (pydantic ``frozen``). The ``dict``-valued fields (``stage_energy_J``,
    ``unavailability.blocking``/``limiting``, ``orbital_diagnostics.definitions``) are plain
    dictionaries: the freeze is shallow, so treat them as read-only."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=False)


class SampleStatistics(_SummaryRecord):
    """Unweighted sample statistics of one quantity (``None`` for an empty sample); unit of the
    quantity named by the field that holds it."""

    min: float | None
    mean: float | None
    median: float | None
    p05: float | None
    p95: float | None
    max: float | None


class TimeSummary(_SummaryRecord):
    """Sampling of the run (emission epochs, TDB seconds past J2000)."""

    n_samples: int
    output_step_s: float
    window_s: float
    start_tdb_s: float
    end_tdb_s: float
    sampling: str


class GridPowerSummary(_SummaryRecord):
    """``P_grid`` statistics [W] (PDF §5 summary.json: mean/median/p05/p95)."""

    time_mean: float | None
    median: float
    p05: float
    p95: float
    peak: float
    mean_when_available: float | None
    percentile_method: str


class CapacityFactorDenominator(_SummaryRecord):
    """Definition and value of the capacity-factor denominator (PDF §10.5)."""

    rated_power_W: float
    definition: str
    reference_range_m: float
    reference_sun_distance_m: float
    reference_eta_capture: float
    reference_eta_atmosphere: float
    reference_thermal_feasible: bool
    reference_power_density_abort: bool


class EndToEndEfficiency(_SummaryRecord):
    """``E_grid / E_incident`` and the mean instantaneous ratio when available (PDF §10.5)."""

    energy_ratio: float | None
    energy_ratio_definition: str
    mean_when_available: float | None
    incident_solar_energy_J: float


class BeamCaptureSummary(_SummaryRecord):
    """Expected capture fraction statistics [1] (PDF §10.5 "beam capture")."""

    eta_capture_when_available: SampleStatistics
    eta_capture_when_visible: SampleStatistics
    definition: str


class PowerDensitySummary(_SummaryRecord):
    """Receiver incident power density [W m^-2] and the beam-abort constraint (PDF §10.5)."""

    incident_peak_when_transmitting: SampleStatistics
    requested_peak_max: float | None
    limit: float | None
    abort_time_fraction: float
    definition: str


class ReasonFraction(_SummaryRecord):
    """Time fractions of one blocking unavailability reason [1]."""

    time_fraction_flagged: float
    time_fraction_primary: float


class UnavailabilitySummary(_SummaryRecord):
    """Why power is unavailable (PDF §11.4 breakdown; reason names of
    :class:`helioswarm.system.Unavailability`)."""

    blocking: dict[str, ReasonFraction]
    limiting_flags_time_fraction: dict[str, float]
    note: str


class OutageSummary(_SummaryRecord):
    """:class:`OutageStatistics` as written to summary.json (durations [s])."""

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


class DiurnalAliasingSummary(_SummaryRecord):
    """Diurnal-sampling diagnostic (module docstring)."""

    hour_angle_max_gap_rad: float
    effective_distinct_diurnal_phases: float | None
    per_edge_resolution_day_fraction: float
    warning: bool
    warning_threshold_day_fraction: float
    access_fraction: float


class ThermalSummary(_SummaryRecord):
    """Thermal feasibility over the window (PDF §8.4)."""

    feasible_time_fraction: float
    pv_temperature_K: SampleStatistics
    binding_margin_K_min: float
    radiator_area_m2_max: float


class GeometrySummary(_SummaryRecord):
    """Earth-receiver range and heliocentric distance statistics [m]."""

    range_m: SampleStatistics
    sun_distance_m: SampleStatistics


class LightTimeSummary(_SummaryRecord):
    """Emission-to-arrival interval [s] (PDF §7.3: "report min/mean/max")."""

    min_s: float
    mean_s: float
    max_s: float


class OrbitalDiagnostics(_SummaryRecord):
    """PDF §7.3 orbital outputs and diagnostics, from the saved time series.

    Time fractions use the trapezoid weights of the module docstring (fractions of simulated
    time); ``earth_range_m``, ``range_rate_m_s`` and ``pointing_rate_rad_s`` are unweighted
    sample statistics. ``beam_direction_j2000_unit_*`` are the components of the inertial
    (J2000) beam direction at the first and last epoch; the direction at every epoch is the
    ``transmit_dir_{x,y,z}`` columns of the time series. ``orbit_drift_vs_l0_m_*`` is ``None``
    when no L0 reference trajectory was computed. An occultation fraction is ``None`` when its
    check is disabled in the configuration (``models.<body>_occultation = false``: not
    evaluated, which is not the same as never blocked); ``occultation_fraction`` is the union
    over the checked bodies and ``None`` when no body is checked.
    """

    earth_range_m: SampleStatistics
    access_fraction: float
    occultation_fraction: float | None
    earth_occultation_fraction: float | None
    sun_occultation_fraction: float | None
    moon_occultation_fraction: float | None
    pointing_rate_rad_s: SampleStatistics
    beam_direction_j2000_unit_first: tuple[float, float, float]
    beam_direction_j2000_unit_last: tuple[float, float, float]
    light_time: LightTimeSummary
    range_rate_m_s: SampleStatistics
    orbit_drift_vs_l0_m_max: float | None
    orbit_drift_vs_l0_m_final: float | None
    definitions: dict[str, str]


class RunSummary(_SummaryRecord):
    """The typed ``summary.json`` document of a run (PDF §5 "summary.json", §10.5, §7.3).

    Every metric is computed by :func:`summarize_timeseries` from the saved time series. The
    record is validated on construction (unknown keys rejected) and on reading
    (:func:`read_summary`). ``summary_schema`` must be one of :data:`READABLE_SUMMARY_SCHEMAS`;
    ``orbital_diagnostics`` is required in layout ``helioswarm-summary/2`` and absent
    (``None``) in summaries written with layout ``helioswarm-summary/1`` (before M6b), which
    are still read so that existing run and sweep directories can be collated.
    """

    summary_schema: str
    project_name: str
    config_hash: str
    transmitter_mode: str
    time: TimeSummary
    p_grid_W: GridPowerSummary
    delivered_energy_J: float
    delivered_energy_definition: str
    availability: float
    availability_definition: str
    capacity_factor: float | None
    capacity_factor_denominator: CapacityFactorDenominator
    end_to_end_efficiency: EndToEndEfficiency
    stage_energy_J: dict[str, float]
    beam_capture: BeamCaptureSummary
    receiver_power_density_W_m2: PowerDensitySummary
    receiver_saturation_time_fraction: float
    max_input_power_W: float
    unavailability: UnavailabilitySummary
    outages: OutageSummary
    diurnal_aliasing: DiurnalAliasingSummary
    thermal: ThermalSummary
    geometry: GeometrySummary
    orbital_diagnostics: OrbitalDiagnostics | None = None
    power_per_collector_area_W_m2: float | None
    timeseries_content_sha256: str | None = None
    timeseries_saved: bool = True

    @model_validator(mode="after")
    def _check_layout(self) -> Self:
        if self.summary_schema not in READABLE_SUMMARY_SCHEMAS:
            raise ValueError(
                f"unknown summary_schema {self.summary_schema!r}; readable layouts: "
                + ", ".join(READABLE_SUMMARY_SCHEMAS)
            )
        if self.summary_schema != LEGACY_SUMMARY_SCHEMA and self.orbital_diagnostics is None:
            raise ValueError(
                f"orbital_diagnostics is required in layout {self.summary_schema!r} "
                f"(optional only in {LEGACY_SUMMARY_SCHEMA!r})"
            )
        return self

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-ready mapping (the on-disk layout)."""
        return self.model_dump(mode="json")


def read_summary(path: Path | str) -> RunSummary:
    """Read and validate a ``summary.json`` file.

    Raises:
        pydantic.ValidationError: the file does not have the :class:`RunSummary` layout.
    """
    return RunSummary.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def _stats(values: Float64Array) -> SampleStatistics:
    if values.size == 0:
        return SampleStatistics(min=None, mean=None, median=None, p05=None, p95=None, max=None)
    return SampleStatistics(
        min=float(np.min(values)),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p05=float(np.percentile(values, 5.0)),
        p95=float(np.percentile(values, 95.0)),
        max=float(np.max(values)),
    )


def _col(table: pa.Table, name: str) -> NDArray[Any]:
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False))


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


ORBITAL_DIAGNOSTIC_DEFINITIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "earth_range_m": "|r_site(t_arrival) - r_sat(t_emit)| at the converged light-time "
        "geometry (PDF §7.3 Earth range)",
        "access_fraction": "time fraction with geometric and operational access (no Earth/Sun/Moon "
        "occultation, elevation >= minimum, Sun-avoidance met)",
        "occultation_fraction": "time fraction with the beam segment blocked by the Earth, Sun or "
        "Moon (per-body fractions overlap); union over the checked bodies; null for a check "
        "disabled in the configuration (not evaluated)",
        "pointing_rate_rad_s": "|d u_tx / d t_emit| of the inertial (J2000) beam direction u_tx",
        "light_time": "emission-to-arrival interval t_arrival - t_emit",
        "range_rate_m_s": "u_tx . (v_site(t_arrival) - v_sat(t_emit)), positive when separating",
        "orbit_drift_vs_l0_m": "|r_sat - r_L0| against the two-body (L0) reference trajectory "
        "from the same initial state",
    }
)
"""Definitions written with :class:`OrbitalDiagnostics` (PDF §7.3)."""


def _orbital_diagnostics(
    table: pa.Table, w: Float64Array, window: float, inputs: SummaryInputs
) -> OrbitalDiagnostics:
    def frac(mask: NDArray[np.bool_]) -> float:
        return float(np.sum(w[mask]) / window) if window else 0.0

    checked = {
        "earth": inputs.earth_occultation_checked,
        "sun": inputs.sun_occultation_checked,
        "moon": inputs.moon_occultation_checked,
    }
    # A disabled check is not evaluated (AccessResult: flag False): report None, not 0.
    blocked = {b: _col(table, f"{b}_blocked").astype(bool) for b, on in checked.items() if on}
    union = np.zeros(table.num_rows, dtype=bool)
    for mask in blocked.values():
        union |= mask
    light = _col(table, "light_time_s").astype(np.float64)
    drift = _col(table, "orbit_drift_vs_l0_m").astype(np.float64)
    direction = np.column_stack(
        [_col(table, f"transmit_dir_{c}").astype(np.float64) for c in ("x", "y", "z")]
    )
    has_drift = bool(np.all(np.isfinite(drift)))
    return OrbitalDiagnostics(
        earth_range_m=_stats(_col(table, "range_m").astype(np.float64)),
        access_fraction=frac(_col(table, "visible").astype(bool)),
        occultation_fraction=frac(union) if blocked else None,
        earth_occultation_fraction=frac(blocked["earth"]) if "earth" in blocked else None,
        sun_occultation_fraction=frac(blocked["sun"]) if "sun" in blocked else None,
        moon_occultation_fraction=frac(blocked["moon"]) if "moon" in blocked else None,
        pointing_rate_rad_s=_stats(_col(table, "pointing_rate_rad_s").astype(np.float64)),
        beam_direction_j2000_unit_first=(
            float(direction[0, 0]),
            float(direction[0, 1]),
            float(direction[0, 2]),
        ),
        beam_direction_j2000_unit_last=(
            float(direction[-1, 0]),
            float(direction[-1, 1]),
            float(direction[-1, 2]),
        ),
        light_time=LightTimeSummary(
            min_s=float(np.min(light)), mean_s=float(np.mean(light)), max_s=float(np.max(light))
        ),
        range_rate_m_s=_stats(_col(table, "range_rate_m_s").astype(np.float64)),
        orbit_drift_vs_l0_m_max=float(np.max(drift)) if has_drift else None,
        orbit_drift_vs_l0_m_final=float(drift[-1]) if has_drift else None,
        definitions=dict(ORBITAL_DIAGNOSTIC_DEFINITIONS),
    )


def summarize_timeseries(table: pa.Table, inputs: SummaryInputs) -> RunSummary:
    """The ``summary.json`` record from a saved time series (module docstring).

    Args:
        table: The ``timeseries.parquet`` table (``pyarrow.Table``; columns of
            :func:`helioswarm.io.timeseries_table`).
        inputs: Run scalars.

    Returns:
        :class:`RunSummary` (numbers in SI, unit-suffixed keys). A degenerate one-sample window
        gives ``None`` for the time-mean quantities.
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
    mean_p = energy / window if window > 0 else None
    rated = inputs.rated_power_W
    cf = mean_p / rated if (rated > 0.0 and mean_p is not None) else None

    stage_energy = {
        attr.replace("_W", "_J"): float(np.sum(w * _col(table, attr).astype(np.float64)))
        for _, attr in STAGE_COLUMNS
    }
    e_incident = float(np.sum(w * incident))
    with np.errstate(invalid="ignore", divide="ignore"):
        inst_eff = np.where(incident > 0.0, p_grid / np.where(incident > 0, incident, 1.0), 0.0)

    prim = primary_reason(code)
    blocking: dict[str, ReasonFraction] = {}
    for r in BLOCKING_REASONS:
        bit = (code & np.uint32(int(r))) != 0
        blocking[str(r.name)] = ReasonFraction(
            time_fraction_flagged=float(np.sum(w[bit]) / window) if window else 0.0,
            time_fraction_primary=float(np.sum(w[prim == r.name]) / window) if window else 0.0,
        )
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

    return RunSummary(
        summary_schema=SUMMARY_SCHEMA,
        project_name=inputs.project_name,
        config_hash=inputs.config_hash,
        transmitter_mode=inputs.transmitter_mode,
        time=TimeSummary(
            n_samples=int(n),
            output_step_s=dt,
            window_s=window,
            start_tdb_s=float(t[0]),
            end_tdb_s=float(t[-1]),
            sampling="trapezoid weights (end samples dt/2); emission epochs (TDB)",
        ),
        p_grid_W=GridPowerSummary(
            time_mean=mean_p,
            median=float(np.median(p_grid)),
            p05=float(np.percentile(p_grid, 5.0)),
            p95=float(np.percentile(p_grid, 95.0)),
            peak=float(np.max(p_grid)),
            mean_when_available=float(np.mean(p_grid[available])) if available.any() else None,
            percentile_method="unweighted sample percentiles, numpy linear interpolation",
        ),
        delivered_energy_J=energy,
        delivered_energy_definition="sum_k w_k P_grid,k (trapezoid weights; emission epochs)",
        availability=float(np.sum(w[available]) / window) if window else 0.0,
        availability_definition="time fraction with no blocking unavailability reason "
        "(all geometry, occultation, Sun-avoidance, eclipse, thermal, weather, pointing and "
        "power-density conditions met)",
        capacity_factor=cf,
        capacity_factor_denominator=CapacityFactorDenominator(
            rated_power_W=rated,
            definition=inputs.rated_power_definition,
            reference_range_m=inputs.reference_range_m,
            reference_sun_distance_m=inputs.reference_sun_distance_m,
            reference_eta_capture=inputs.reference_eta_capture,
            reference_eta_atmosphere=inputs.reference_eta_atmosphere,
            reference_thermal_feasible=inputs.reference_thermal_feasible,
            reference_power_density_abort=inputs.reference_power_density_abort,
        ),
        end_to_end_efficiency=EndToEndEfficiency(
            energy_ratio=energy / e_incident if e_incident > 0 else None,
            energy_ratio_definition="E_grid / E_incident_on_collector over the window",
            mean_when_available=float(np.mean(inst_eff[available])) if available.any() else None,
            incident_solar_energy_J=e_incident,
        ),
        stage_energy_J=stage_energy,
        beam_capture=BeamCaptureSummary(
            eta_capture_when_available=_stats(capture[available]),
            eta_capture_when_visible=_stats(capture[visible]),
            definition="expected fraction of the beam power at the receiver plane intercepted "
            "by the receiver (diffraction, pointing, wavefront and lattice factors)",
        ),
        receiver_power_density_W_m2=PowerDensitySummary(
            incident_peak_when_transmitting=_stats(density[transmitting]),
            requested_peak_max=float(np.max(requested)) if requested.size else None,
            limit=inputs.max_power_density_W_m2,
            abort_time_fraction=blocking["POWER_DENSITY_ABORT"].time_fraction_flagged,
            definition="peak irradiance of the aberration-free pattern on the beam-normal "
            "plane at the receiver (upper bound)",
        ),
        receiver_saturation_time_fraction=limiting["RECEIVER_SATURATION"],
        max_input_power_W=inputs.max_input_power_W,
        unavailability=UnavailabilitySummary(
            blocking=blocking,
            limiting_flags_time_fraction=limiting,
            note="time_fraction_flagged counts every epoch with the bit set (reasons "
            "overlap); time_fraction_primary assigns each unavailable epoch to its first "
            "reason in precedence order and sums to 1 - availability",
        ),
        outages=OutageSummary(**asdict(outages)),
        diurnal_aliasing=DiurnalAliasingSummary(
            hour_angle_max_gap_rad=gap,
            effective_distinct_diurnal_phases=2.0 * math.pi / gap if gap > 0 else None,
            per_edge_resolution_day_fraction=gap / (2.0 * math.pi),
            warning=bool(gap / (2.0 * math.pi) > ALIASING_WARNING_FRACTION),
            warning_threshold_day_fraction=ALIASING_WARNING_FRACTION,
            access_fraction=float(np.sum(w[visible]) / window) if window else 0.0,
        ),
        thermal=ThermalSummary(
            feasible_time_fraction=float(np.sum(w[feasible]) / window) if window else 0.0,
            pv_temperature_K=_stats(pv_temp),
            binding_margin_K_min=float(np.min(margin)),
            radiator_area_m2_max=float(np.max(rad_area)),
        ),
        geometry=GeometrySummary(range_m=_stats(rng_m), sun_distance_m=_stats(sun_d)),
        orbital_diagnostics=_orbital_diagnostics(table, w, window, inputs),
        power_per_collector_area_W_m2=(
            mean_p / inputs.collector_area_m2 if mean_p is not None else None
        ),
    )


__all__ = [
    "LEGACY_SUMMARY_SCHEMA",
    "ORBITAL_DIAGNOSTIC_DEFINITIONS",
    "OUTAGE_HISTOGRAM_EDGES_S",
    "READABLE_SUMMARY_SCHEMAS",
    "SUMMARY_SCHEMA",
    "BeamCaptureSummary",
    "CapacityFactorDenominator",
    "DiurnalAliasingSummary",
    "EndToEndEfficiency",
    "GeometrySummary",
    "GridPowerSummary",
    "LightTimeSummary",
    "OrbitalDiagnostics",
    "OutageStatistics",
    "OutageSummary",
    "PowerDensitySummary",
    "ReasonFraction",
    "RunSummary",
    "SampleStatistics",
    "SummaryInputs",
    "ThermalSummary",
    "TimeSummary",
    "UnavailabilitySummary",
    "outage_statistics",
    "read_summary",
    "sample_weights_s",
    "summarize_timeseries",
]
