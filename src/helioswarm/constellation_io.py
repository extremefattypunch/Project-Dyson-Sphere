"""Constellation run outputs, network metrics and array-task link files (PDF §11.4, §14.1-14.2).

A constellation run directory contains::

    config_resolved.yaml   resolved configuration (constellation section included)
    manifest.json          provenance (helioswarm.manifest.RunManifest)
    network.parquet        one row per epoch: network P_grid and stage powers of the scheduled
                           links, active-satellite / valid-link / scheduled-link counts, scheduler
                           method, satellite counts per unavailability reason
    receivers.parquet      one row per (epoch, site): beams, summed incident/accepted/receiver/grid
                           power, clipping, summed peak irradiance, link counts per reason
    satellites.parquet     one row per satellite: ring, phase, resolved elements, areas, mass
                           proxy, delivered energy, time fractions, orbit diagnostics
    links/                 optional per-link detail (one Parquet file per block of satellites,
                           row groups per satellite), written only on request and after a size
                           estimate is checked against a cap
    summary.json           ConstellationSummary computed from the three tables above
    figures/               figures drawn only from the saved tables
    logs/run.log           diagnostics, warnings and stage wall times

Time weighting, percentiles and outages use exactly the conventions of
:mod:`helioswarm.metrics` (trapezoid sample weights, unweighted sample percentiles, outages as
maximal runs of zero-power samples).

Array tasks (``helioswarm run CONFIG --chunk-index i --chunk-count n --candidates-dir D``) write
the stage-A :class:`~helioswarm.constellation.LinkSet` of their satellites to
``D/linkset_<i>_of_<n>.npz`` (uncompressed NumPy archive: an intermediate file, not a result);
``helioswarm run CONFIG --candidates-dir D`` merges all of them (configuration hash, epochs and
satellite coverage checked) and runs stage B.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from helioswarm.config import RunConfig, config_hash, write_resolved_yaml
from helioswarm.constellation import (
    DELIVERING,
    LINK_REASONS,
    ConstellationRunResult,
    LinkSet,
    chunk_indices,
    config_log,
    constellation_members,
    evaluate_links,
    finish_constellation,
    link_reason_index,
    merge_link_sets,
)
from helioswarm.kernels import KernelSet
from helioswarm.metrics import (
    GridPowerSummary,
    OutageSummary,
    SampleStatistics,
    TimeSummary,
    outage_statistics,
    sample_weights_s,
)
from helioswarm.scheduler import SOLVE_METHODS, LinkDecision
from helioswarm.system import Backend, RunLogEntry, StageTiming

Float64Array = NDArray[np.float64]

CONSTELLATION_SUMMARY_SCHEMA: Final[str] = "helioswarm-constellation-summary/1"
NETWORK_SCHEMA: Final[str] = "helioswarm-constellation-network/1"
NETWORK_FILE: Final[str] = "network.parquet"
RECEIVERS_FILE: Final[str] = "receivers.parquet"
SATELLITES_FILE: Final[str] = "satellites.parquet"
LINKS_DIR: Final[str] = "links"
LINKSET_PREFIX: Final[str] = "linkset_"

LINK_DETAIL_BYTES_PER_ROW: Final[int] = 4 + 4 + 2 + 4 + 1 + 1 + 1 + 5 * 8
"""Uncompressed size of one per-link detail row [bytes] (the size estimate's basis)."""

BYTES_PER_GB: Final[int] = 1_000_000_000
"""Decimal gigabyte (SI, 1e9 bytes): the unit of ``--link-detail-max-gb`` and of the messages."""

DEFAULT_LINK_DETAIL_MAX_BYTES: Final[int] = 4 * BYTES_PER_GB
"""Default cap on the uncompressed per-link detail [bytes] (4 GB = 4e9 bytes)."""

REASON_NAMES: Final[tuple[str, ...]] = tuple(str(r.name) for r in LINK_REASONS)


class LinkDetailTooLargeError(RuntimeError):
    """The per-link detail would exceed the configured size cap."""


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------


def _f64(a: Any) -> pa.Array:
    return pa.array(np.asarray(a, dtype=np.float64))


def _seq_sum(arr: Float64Array, axis_last_mask: NDArray[np.bool_] | None = None) -> Float64Array:
    """Sum over the last axis in index order (deterministic floating-point order)."""
    a = np.asarray(arr, dtype=np.float64)
    if axis_last_mask is not None:
        a = np.where(axis_last_mask, a, 0.0)
    out = np.zeros(a.shape[:-1])
    for k in range(a.shape[-1]):
        out = out + a[..., k]
    return out


def network_table(result: ConstellationRunResult) -> pa.Table:
    """``network.parquet``: one row per epoch (columns in SI units, unit-suffixed)."""
    ls = result.links
    sched = result.schedule
    t = result.grid.epochs_tdb_s
    assigned = sched.assigned_site.astype(np.int64)
    on = assigned >= 0
    n_t, n_sat = assigned.shape
    rows = np.arange(n_t)[:, None]
    sats = np.arange(n_sat)[None, :]
    site_safe = np.where(on, assigned, 0)
    sel_atm = np.where(on, ls.atm_W[rows, sats, site_safe], 0.0)
    sel_cap = np.where(on, ls.captured_W[rows, sats, site_safe], 0.0)
    feasible = ls.feasible
    # Independent recount of the network P_grid from the assignments and the link powers
    # (not from the scheduler's site totals): sum_j min(sum_{i -> j} P_captured, L_j) eta eta.
    recount = np.zeros(n_t)
    for j, cap in enumerate(result.capacities):
        inc_j = _seq_sum(ls.captured_W[:, :, j], assigned == j)
        recount = recount + (
            np.minimum(inc_j, cap.max_accepted_W)
            * result.receiver_conversion_efficiency
            * result.receiver_grid_efficiency
        )
    cols: dict[str, pa.Array] = {
        "epoch_index": pa.array(np.arange(n_t, dtype=np.int32)),
        "t_emit_tdb_s": _f64(t),
        "elapsed_s": _f64(t - t[0]),
        "grid_W": _f64(result.network_grid_W),
        "grid_recount_W": _f64(recount),
        "unconstrained_grid_W": _f64(result.unconstrained_grid_W),
        "receiver_W": _f64(_seq_sum(result.site_receiver_W)),
        "accepted_W": _f64(_seq_sum(result.site_accepted_W)),
        "captured_W": _f64(_seq_sum(result.site_incident_W)),
        "captured_check_W": _f64(_seq_sum(sel_cap)),
        "atm_W": _f64(_seq_sum(sel_atm)),
        "aperture_W": _f64(_seq_sum(ls.aperture_W, on)),
        "beam_W": _f64(_seq_sum(ls.beam_W, on)),
        "dc_to_tx_W": _f64(_seq_sum(ls.dc_to_tx_W, on)),
        "pv_to_transmitter_W": _f64(_seq_sum(ls.pv_electric_W, on)),
        "pv_electric_W": _f64(_seq_sum(ls.pv_electric_W)),
        "incident_solar_W": _f64(_seq_sum(ls.incident_solar_W)),
        "active_satellite_count": pa.array(np.count_nonzero(on, axis=1).astype(np.int32)),
        "valid_link_count": pa.array(np.count_nonzero(feasible, axis=(1, 2)).astype(np.int32)),
        "scheduled_link_count": pa.array(
            np.count_nonzero(sched.decision == int(LinkDecision.SELECTED), axis=(1, 2)).astype(
                np.int32
            )
        ),
        "satellites_with_valid_link": pa.array(
            np.count_nonzero(np.any(feasible, axis=2), axis=1).astype(np.int32)
        ),
        "illuminated_satellite_count": pa.array(
            np.count_nonzero(ls.illumination_fraction > 0.0, axis=1).astype(np.int32)
        ),
        "thermally_feasible_satellite_count": pa.array(
            np.count_nonzero(ls.thermal_feasible, axis=1).astype(np.int32)
        ),
        "retarget_events": pa.array(sched.retarget_events.astype(np.int32)),
        "saturated_site_count": pa.array(
            np.count_nonzero(
                result.site_incident_W > np.array([c.max_accepted_W for c in result.capacities]),
                axis=1,
            ).astype(np.int32)
        ),
        "scheduler_method": pa.array(
            [SOLVE_METHODS[m] for m in sched.method.tolist()], pa.string()
        ),
        "n_sat_delivering": pa.array(
            np.count_nonzero(result.satellite_reason_index == DELIVERING, axis=1).astype(np.int32)
        ),
    }
    for k, name in enumerate(REASON_NAMES):
        cols[f"n_sat_reason_{name}"] = pa.array(
            np.count_nonzero(result.satellite_reason_index == k, axis=1).astype(np.int32)
        )
    meta = {
        b"helioswarm_network_schema": NETWORK_SCHEMA.encode(),
        b"config_hash": result.config_hash.encode(),
    }
    return pa.table(cols).replace_schema_metadata(meta)


def receivers_table(result: ConstellationRunResult) -> pa.Table:
    """``receivers.parquet``: one row per (epoch, site), epoch-major.

    The ``rx_slot_*`` columns refer to reception slot ``[t_k, t_{k+1})`` of the site in
    *reception* time (:class:`~helioswarm.constellation.ReceptionTimeAggregates`), the others
    to emission epoch ``t_k``."""
    n_t, n_s = result.site_grid_W.shape
    sched = result.schedule
    t = result.grid.epochs_tdb_s
    names = [s.name for s in result.config.receiver.sites]
    limit = np.array([c.max_accepted_W for c in result.capacities])
    feasible = result.links.feasible
    primary = link_reason_index(result.link_code)
    rx = result.reception
    cols: dict[str, pa.Array] = {
        "epoch_index": pa.array(np.repeat(np.arange(n_t, dtype=np.int32), n_s)),
        "t_emit_tdb_s": _f64(np.repeat(t, n_s)),
        "site_index": pa.array(np.tile(np.arange(n_s, dtype=np.int16), n_t)),
        "site_name": pa.array(names * n_t, pa.string()),
        "beams": pa.array(sched.site_beams.reshape(-1).astype(np.int32)),
        "incident_W": _f64(result.site_incident_W.reshape(-1)),
        "accepted_W": _f64(result.site_accepted_W.reshape(-1)),
        "clipped_W": _f64((result.site_incident_W - result.site_accepted_W).reshape(-1)),
        "receiver_W": _f64(result.site_receiver_W.reshape(-1)),
        "grid_W": _f64(result.site_grid_W.reshape(-1)),
        "density_sum_W_m2": _f64(sched.site_density_W_m2.reshape(-1)),
        "saturated": pa.array((result.site_incident_W > limit[None, :]).reshape(-1)),
        "valid_link_count": pa.array(
            np.count_nonzero(feasible, axis=1).reshape(-1).astype(np.int32)
        ),
        "light_time_spread_s": _f64(rx.light_time_spread_s.reshape(-1)),
        "rx_slot_max_beams": pa.array(rx.max_beams.reshape(-1).astype(np.int32)),
        "rx_slot_max_density_W_m2": _f64(rx.max_density_W_m2.reshape(-1)),
        "rx_slot_incident_energy_J": _f64(rx.incident_energy_J.reshape(-1)),
        "rx_slot_accepted_energy_J": _f64(rx.accepted_energy_J.reshape(-1)),
        "rx_slot_beam_overrun_s": _f64(rx.beam_overrun_s.reshape(-1)),
        "rx_slot_density_overrun_s": _f64(rx.density_overrun_s.reshape(-1)),
    }
    for k, name in enumerate(REASON_NAMES):
        cols[f"n_links_reason_{name}"] = pa.array(
            np.count_nonzero(primary == k, axis=1).reshape(-1).astype(np.int32)
        )
    meta = {b"config_hash": result.config_hash.encode()}
    return pa.table(cols).replace_schema_metadata(meta)


def proxy_radiator_area_m2(config: RunConfig, required_max_m2: float) -> float:
    """Radiator area of the mass proxy: ``thermal.radiator_area_m2`` (physical plate) if
    configured, else the largest required area over the run [m^2]."""
    a = config.thermal.radiator_area_m2
    return float(required_max_m2) if a is None else float(a)


def mass_proxy_kg(
    config: RunConfig, collector_area_m2: float, radiator_area_m2: float, aperture_area_m2: float
) -> float | None:
    """``m = m_bus + s_col A_col + s_rad A_rad + s_tx A_tx`` [kg] (``constellation.mass_proxy``
    with ``s_rad = thermal.radiator_specific_mass_kg_m2``), ``None`` without a mass proxy."""
    assert config.constellation is not None
    mp = config.constellation.mass_proxy
    if mp is None:
        return None
    s_rad = config.thermal.radiator_specific_mass_kg_m2
    assert s_rad is not None  # schema rule when a mass proxy is configured
    return float(
        mp.bus_mass_kg
        + mp.collector_areal_mass_kg_m2 * collector_area_m2
        + s_rad * radiator_area_m2
        + mp.transmitter_areal_mass_kg_m2 * aperture_area_m2
    )


def satellites_table(result: ConstellationRunResult) -> pa.Table:
    """``satellites.parquet``: one row per satellite."""
    ls = result.links
    t = result.grid.epochs_tdb_s
    w = sample_weights_s(t.size, result.config.time.output_step_s)
    window = float(np.sum(w))
    on = result.schedule.assigned_site >= 0
    feasible_any = np.any(ls.feasible, axis=2)
    rows: dict[str, list[Any]] = {
        k: []
        for k in (
            "satellite",
            "ring_index",
            "ring_name",
            "index_in_ring",
            "phase_angle_deg",
            "phase_reference",
            "true_anomaly_deg",
            "ecliptic_longitude_deg",
            "earth_relative_longitude_deg",
            "semimajor_axis_m",
            "eccentricity",
            "inclination_deg",
            "collector_area_m2",
            "transmitter_aperture_diameter_m",
            "transmitter_aperture_area_m2",
            "radiator_area_required_m2",
            "radiator_area_proxy_m2",
            "mass_proxy_kg",
            "delivered_energy_J",
            "unconstrained_energy_J",
            "transmitting_time_fraction",
            "valid_link_time_fraction",
            "thermal_feasible_time_fraction",
            "pv_electric_mean_W",
            "incident_solar_energy_J",
            "orbit_drift_max_m",
            "orbit_energy_rel_drift",
            "orbit_nfev",
            "light_time_fallbacks",
        )
    }
    for i, m in enumerate(result.members):
        cfg = m.config
        a_col = cfg.collector.area_m2
        d_tx = cfg.transmitter.aperture_diameter_m
        a_tx = math.pi * d_tx**2 / 4.0
        a_rad = float(np.max(ls.radiator_area_m2[:, i]))
        a_rad_proxy = proxy_radiator_area_m2(cfg, a_rad)
        mass = mass_proxy_kg(result.config, a_col, a_rad_proxy, a_tx)
        values = {
            "satellite": m.index,
            "ring_index": m.ring_index,
            "ring_name": m.ring_name,
            "index_in_ring": m.index_in_ring,
            "phase_angle_deg": m.phase_angle_deg,
            "phase_reference": cfg.orbit.phase_reference,
            "true_anomaly_deg": math.degrees(float(ls.true_anomaly_rad[i])),
            "ecliptic_longitude_deg": math.degrees(float(ls.ecliptic_longitude_rad[i])),
            "earth_relative_longitude_deg": math.degrees(float(ls.earth_relative_longitude_rad[i])),
            "semimajor_axis_m": cfg.orbit.semimajor_axis_m,
            "eccentricity": cfg.orbit.eccentricity,
            "inclination_deg": cfg.orbit.inclination_deg,
            "collector_area_m2": a_col,
            "transmitter_aperture_diameter_m": d_tx,
            "transmitter_aperture_area_m2": a_tx,
            "radiator_area_required_m2": a_rad,
            "radiator_area_proxy_m2": a_rad_proxy,
            "mass_proxy_kg": math.nan if mass is None else mass,
            "delivered_energy_J": float(np.sum(w * result.satellite_delivered_W[:, i])),
            "unconstrained_energy_J": float(np.sum(w * result.satellite_unconstrained_W[:, i])),
            "transmitting_time_fraction": float(np.sum(w[on[:, i]]) / window) if window else 0.0,
            "valid_link_time_fraction": (
                float(np.sum(w[feasible_any[:, i]]) / window) if window else 0.0
            ),
            "thermal_feasible_time_fraction": (
                float(np.sum(w[ls.thermal_feasible[:, i]]) / window) if window else 0.0
            ),
            "pv_electric_mean_W": float(np.sum(w * ls.pv_electric_W[:, i]) / window)
            if window
            else math.nan,
            "incident_solar_energy_J": float(np.sum(w * ls.incident_solar_W[:, i])),
            "orbit_drift_max_m": float(ls.orbit_drift_max_m[i]),
            "orbit_energy_rel_drift": float(ls.orbit_energy_rel_drift[i]),
            "orbit_nfev": int(ls.orbit_nfev[i]),
            "light_time_fallbacks": int(ls.light_time_fallbacks[i]),
        }
        for k, v in values.items():
            rows[k].append(v)
    meta = {b"config_hash": result.config_hash.encode()}
    return pa.table(rows).replace_schema_metadata(meta)


def link_detail_rows(result: ConstellationRunResult) -> int:
    """Rows of the per-link detail: ``T * N * S``."""
    return int(np.prod(result.link_code.shape))


def write_link_detail(
    result: ConstellationRunResult,
    directory: Path,
    *,
    max_bytes: int = DEFAULT_LINK_DETAIL_MAX_BYTES,
    satellites_per_file: int = 100,
) -> tuple[Path, ...]:
    """Write the per-link detail into ``directory`` (``links/part-XXXXX.parquet``).

    Each file holds a block of satellites, one row group per satellite (rows ordered by site,
    then epoch). Columns: ``epoch_index``, ``satellite``, ``site_index``, ``reason_code``
    (final link code), ``primary_reason`` (index into :data:`REASON_NAMES`, -1 = scheduled),
    ``decision`` (:class:`~helioswarm.scheduler.LinkDecision`, 255 = not feasible),
    ``scheduled``, ``range_m``, ``elevation_rad``, ``captured_W``, ``atm_W``, ``density_W_m2``.

    Raises:
        LinkDetailTooLargeError: the uncompressed estimate
            ``rows * LINK_DETAIL_BYTES_PER_ROW`` exceeds ``max_bytes`` (nothing is written).
    """
    n_rows = link_detail_rows(result)
    estimate = n_rows * LINK_DETAIL_BYTES_PER_ROW
    if estimate > max_bytes:
        raise LinkDetailTooLargeError(
            f"per-link detail would be ~{estimate / 1e9:.2f} GB uncompressed ({n_rows} rows) > "
            f"cap {max_bytes / 1e9:.2f} GB; raise the cap or omit the link detail"
        )
    directory.mkdir(parents=True, exist_ok=True)
    ls = result.links
    n_t, n_sat, n_s = result.link_code.shape
    primary = link_reason_index(result.link_code)
    dec = result.schedule.decision
    paths = []
    for f0 in range(0, n_sat, satellites_per_file):
        path = directory / f"part-{f0 // satellites_per_file:05d}.parquet"
        writer = None
        try:
            for i in range(f0, min(n_sat, f0 + satellites_per_file)):
                sel = dec[:, i, :] == int(LinkDecision.SELECTED)
                prim = np.where(sel, -1, primary[:, i, :]).astype(np.int8)

                def flat(a: NDArray[Any]) -> NDArray[Any]:
                    return np.asarray(a).T.reshape(-1)  # (S, T) order: site, then epoch

                table = pa.table(
                    {
                        "epoch_index": pa.array(np.tile(np.arange(n_t, dtype=np.int32), n_s)),
                        "satellite": pa.array(np.full(n_t * n_s, i, dtype=np.int32)),
                        "site_index": pa.array(np.repeat(np.arange(n_s, dtype=np.int16), n_t)),
                        "reason_code": pa.array(flat(result.link_code[:, i, :]).astype(np.uint32)),
                        "primary_reason": pa.array(flat(prim)),
                        "decision": pa.array(flat(dec[:, i, :]).astype(np.uint8)),
                        "scheduled": pa.array(flat(sel)),
                        "range_m": _f64(flat(ls.range_m[:, i, :])),
                        "elevation_rad": _f64(flat(ls.elevation_rad[:, i, :])),
                        "captured_W": _f64(flat(ls.captured_W[:, i, :])),
                        "atm_W": _f64(flat(ls.atm_W[:, i, :])),
                        "density_W_m2": _f64(flat(ls.density_W_m2[:, i, :])),
                    }
                )
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=table.num_rows)
        finally:
            if writer is not None:
                writer.close()
        paths.append(path)
    return tuple(paths)


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=False)


class CountSummary(_Record):
    """Time mean (trapezoid weights) and unweighted sample statistics of a count series."""

    time_mean: float | None
    min: int
    median: float
    p05: float
    p95: float
    max: int


class PowerSeriesSummary(_Record):
    """Grid-power time series of the network or one receiver (PDF §10.5, §11.4)."""

    p_grid_W: GridPowerSummary
    delivered_energy_J: float
    availability: float
    availability_definition: str
    outages: OutageSummary


class ReceiverSummary(_Record):
    """One receiver site."""

    site_index: int
    site_name: str
    max_beams: int | None
    max_accepted_W: float | None
    max_power_density_W_m2: float | None
    power: PowerSeriesSummary
    saturation_time_fraction: float
    clipped_energy_J: float
    beams: CountSummary
    density_sum_max_W_m2: float
    valid_links: CountSummary


class SchedulerSummary(_Record):
    """Scheduler configuration and activity."""

    policy: str
    duty_cycle_max: float | None
    duty_cycle_window_s: float | None
    retarget_model: str
    retarget_events: int
    epochs_by_method: dict[str, int]


class TotalsSummary(_Record):
    """Constellation totals (PDF §11.4)."""

    collector_area_m2: float
    radiator_area_required_m2: float
    radiator_area_definition: str
    transmitter_aperture_area_m2: float
    mass_proxy_kg: float | None
    mass_proxy_definition: str | None


class RingSummary(_Record):
    """One ring."""

    name: str
    n_satellites: int
    phase_distribution: str
    phase_reference: str
    delivered_energy_J: float


class UnavailabilityBreakdown(_Record):
    """Why power is unavailable (PDF §11.4)."""

    satellite_time_fraction_delivering: float
    satellite_time_fraction_primary: dict[str, float]
    link_time_fraction_primary: dict[str, float]
    definition: str


class DoubleCountingCheck(_Record):
    """No double counting (PDF §10.4).

    ``relative_difference`` (network energy vs the sum of the receiver energies) is a
    *consistency* check only: the network column is the sum of the receiver columns by
    construction. The independent evidence is ``recount_relative_difference`` (network energy
    vs a recount from the satellite assignments and the link powers, not from the scheduler's
    site totals), ``satellites_relative_difference`` (vs the sum of the per-satellite shares)
    and ``beams_equal_active_satellites`` (scheduled beams over all sites equal the satellites
    transmitting: no satellite counted twice)."""

    network_energy_J: float
    receivers_energy_sum_J: float
    relative_difference: float
    recount_energy_J: float
    recount_relative_difference: float
    satellites_energy_sum_J: float
    satellites_relative_difference: float
    beams_equal_active_satellites: bool
    passed: bool
    definition: str


class UnconstrainedBaseline(_Record):
    """Unlimited-receiver baseline (every satellite on its best feasible link, no site
    capacity): separates placement (geometry) from receiver capacity."""

    energy_J: float
    mean_W: float | None
    capacity_loss_fraction: float | None
    power_per_collector_area_W_m2: float | None
    satellite_mean_energy_J: float
    definition: str


class ReceptionTimeCheck(_Record):
    """One site's scheduled beams reconstructed in reception time (emission-epoch aggregation
    approximation; :class:`~helioswarm.constellation.ReceptionTimeAggregates`)."""

    site_index: int
    site_name: str
    multi_beam_epochs: int
    light_time_spread_max_s: float
    light_time_spread_median_s: float | None
    max_beams: int
    beam_slot_overrun_time_s: float
    max_density_sum_W_m2: float
    density_overrun_time_s: float
    incident_energy_emission_J: float
    incident_energy_reception_J: float
    accepted_energy_emission_J: float
    accepted_energy_reception_J: float
    accepted_energy_relative_difference: float | None


class ReceptionTimeSummary(_Record):
    """Bound on the emission-epoch aggregation approximation (all sites)."""

    receivers: list[ReceptionTimeCheck]
    accepted_energy_emission_J: float
    accepted_energy_reception_J: float
    accepted_energy_relative_difference: float | None
    beam_slot_overrun_time_s: float
    density_overrun_time_s: float
    definition: str


class ConstellationSummary(_Record):
    """``summary.json`` of a constellation run (computed from the saved tables)."""

    summary_schema: str
    project_name: str
    config_hash: str
    transmitter_mode: str
    n_satellites: int
    n_sites: int
    rings: list[RingSummary]
    scheduler: SchedulerSummary
    time: TimeSummary
    network: PowerSeriesSummary
    receivers: list[ReceiverSummary]
    active_satellites: CountSummary
    valid_links: CountSummary
    scheduled_links: CountSummary
    totals: TotalsSummary
    power_per_collector_area_W_m2: float | None
    power_per_mass_W_kg: float | None
    end_to_end_efficiency: float | None
    end_to_end_efficiency_definition: str
    stage_energy_J: dict[str, float]
    unavailability: UnavailabilityBreakdown
    double_counting: DoubleCountingCheck
    unconstrained: UnconstrainedBaseline
    reception_time: ReceptionTimeSummary
    satellite_delivered_energy_J: SampleStatistics
    incoherent_summation: str
    link_detail_saved: bool = False
    link_detail_rows: int = 0

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.summary_schema != CONSTELLATION_SUMMARY_SCHEMA:
            raise ValueError(f"unknown constellation summary_schema {self.summary_schema!r}")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return self.model_dump(mode="json")


def read_constellation_summary(path: Path | str) -> ConstellationSummary:
    """Read and validate a constellation ``summary.json``."""
    return ConstellationSummary.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def is_constellation_summary(document: Any) -> bool:
    """True if a parsed summary.json is a constellation summary."""
    return isinstance(document, dict) and document.get("summary_schema") == (
        CONSTELLATION_SUMMARY_SCHEMA
    )


def _rel(a: float, b: float) -> float:
    """``|a - b| / |b|`` (0 when both are 0)."""
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(b), 1e-300)


def _col(table: pa.Table, name: str) -> NDArray[Any]:
    return np.asarray(table.column(name).to_numpy(zero_copy_only=False))


def _count_summary(x: NDArray[Any], w: Float64Array, window: float) -> CountSummary:
    v = np.asarray(x, dtype=np.float64)
    return CountSummary(
        time_mean=float(np.sum(w * v) / window) if window > 0 else None,
        min=int(np.min(v)),
        median=float(np.median(v)),
        p05=float(np.percentile(v, 5.0)),
        p95=float(np.percentile(v, 95.0)),
        max=int(np.max(v)),
    )


def _power_series(
    p: Float64Array, t: Float64Array, w: Float64Array, window: float, step: float, what: str
) -> PowerSeriesSummary:
    energy = float(np.sum(w * p))
    up = p > 0.0
    return PowerSeriesSummary(
        p_grid_W=GridPowerSummary(
            time_mean=energy / window if window > 0 else None,
            median=float(np.median(p)),
            p05=float(np.percentile(p, 5.0)),
            p95=float(np.percentile(p, 95.0)),
            peak=float(np.max(p)),
            mean_when_available=float(np.mean(p[up])) if up.any() else None,
            percentile_method="unweighted sample percentiles, numpy linear interpolation",
        ),
        delivered_energy_J=energy,
        availability=float(np.sum(w[up]) / window) if window else 0.0,
        availability_definition=f"time fraction with {what} P_grid > 0 (at least one scheduled "
        "link delivering power)",
        outages=OutageSummary(**asdict(outage_statistics(up, t, step))),
    )


@dataclass(frozen=True)
class ConstellationSummaryInputs:
    """Scalars of the summary that are not in the tables.

    Attributes:
        project_name: ``project.name``.
        config_hash: Configuration hash.
        transmitter_mode: ``optical`` or ``microwave``.
        output_step_s: Grid step [s].
        site_names: Site names in index order.
        site_max_beams: Beam slots per site (``None`` = unlimited).
        site_max_accepted_W: ``L_j`` per site [W] (``None`` if infinite).
        max_power_density_W_m2: Density interlock [W m^-2] or ``None``.
        rings: ``(name, n_satellites, phase_distribution, phase_reference)`` per ring.
        policy: Scheduler policy.
        duty_cycle_max: Duty-cycle limit or ``None``.
        duty_cycle_window_s: Its window [s] or ``None``.
        retarget_model: Retarget model.
        mass_proxy_definition: Formula with the configured values, or ``None``.
        link_detail_saved: Whether the per-link detail was written.
        link_detail_rows: Its rows.
    """

    project_name: str
    config_hash: str
    transmitter_mode: str
    output_step_s: float
    site_names: tuple[str, ...]
    site_max_beams: tuple[int | None, ...]
    site_max_accepted_W: tuple[float | None, ...]
    max_power_density_W_m2: float | None
    rings: tuple[tuple[str, int, str, str], ...]
    policy: str
    duty_cycle_max: float | None
    duty_cycle_window_s: float | None
    retarget_model: str
    mass_proxy_definition: str | None
    link_detail_saved: bool = False
    link_detail_rows: int = 0


def summary_inputs(
    result: ConstellationRunResult, *, link_detail_saved: bool = False
) -> ConstellationSummaryInputs:
    """:class:`ConstellationSummaryInputs` of a run."""
    cfg = result.config
    c = cfg.constellation
    assert c is not None
    mp = c.mass_proxy
    mass_def = None
    if mp is not None:
        a_rad = (
            "A_rad,required,max"
            if cfg.thermal.radiator_area_m2 is None
            else f"A_rad = {cfg.thermal.radiator_area_m2} m^2 (configured plate)"
        )
        mass_def = (
            f"m_sat = {mp.bus_mass_kg} kg + {mp.collector_areal_mass_kg_m2} kg/m^2 * A_col + "
            f"{cfg.thermal.radiator_specific_mass_kg_m2} kg/m^2 (thermal."
            f"radiator_specific_mass_kg_m2) * {a_rad} + {mp.transmitter_areal_mass_kg_m2} "
            "kg/m^2 * pi D^2/4 (assumed specific masses; a proxy, not a mass budget)"
        )
    return ConstellationSummaryInputs(
        project_name=cfg.project.name,
        config_hash=result.config_hash,
        transmitter_mode=cfg.transmitter.mode,
        output_step_s=cfg.time.output_step_s,
        site_names=tuple(s.name for s in cfg.receiver.sites),
        site_max_beams=tuple(cap.max_beams for cap in result.capacities),
        site_max_accepted_W=tuple(
            cap.max_accepted_W if math.isfinite(cap.max_accepted_W) else None
            for cap in result.capacities
        ),
        max_power_density_W_m2=cfg.receiver.max_power_density_W_m2,
        rings=tuple(
            (
                r.name,
                r.n_satellites,
                r.phase_distribution,
                r.phase_reference or cfg.orbit.phase_reference,
            )
            for r in c.rings
        ),
        policy=c.scheduler.policy,
        duty_cycle_max=c.scheduler.duty_cycle_max,
        duty_cycle_window_s=c.scheduler.duty_cycle_window_s,
        retarget_model=c.scheduler.retarget_model,
        mass_proxy_definition=mass_def,
        link_detail_saved=link_detail_saved,
        link_detail_rows=link_detail_rows(result) if link_detail_saved else 0,
    )


STAGE_ENERGY_COLUMNS: Final[tuple[str, ...]] = (
    "pv_electric_W",
    "pv_to_transmitter_W",
    "dc_to_tx_W",
    "beam_W",
    "aperture_W",
    "atm_W",
    "captured_W",
    "accepted_W",
    "receiver_W",
    "grid_W",
)
"""Stage columns of ``network.parquet`` integrated into ``stage_energy_J`` (``pv_electric_W``
is generated by all satellites; the others are sums over the scheduled links)."""


def summarize_constellation(
    network: pa.Table,
    receivers: pa.Table,
    satellites: pa.Table,
    inputs: ConstellationSummaryInputs,
) -> ConstellationSummary:
    """The constellation ``summary.json`` record from the saved tables (module docstring)."""
    t = _col(network, "t_emit_tdb_s").astype(np.float64)
    n_t = t.size
    dt = inputs.output_step_s
    w = sample_weights_s(n_t, dt)
    window = float(np.sum(w))
    p_net = _col(network, "grid_W").astype(np.float64)
    n_s = len(inputs.site_names)
    site_idx = _col(receivers, "site_index").astype(np.int64).reshape(n_t, n_s)
    if not np.array_equal(site_idx, np.tile(np.arange(n_s), (n_t, 1))):
        raise ValueError("receivers.parquet is not epoch-major with sites in index order")

    def rcol(name: str) -> NDArray[Any]:
        return _col(receivers, name).reshape(n_t, n_s)

    p_site = rcol("grid_W").astype(np.float64)
    beams = rcol("beams").astype(np.int64)
    rec_summaries = []
    for j in range(n_s):
        rec_summaries.append(
            ReceiverSummary(
                site_index=j,
                site_name=inputs.site_names[j],
                max_beams=inputs.site_max_beams[j],
                max_accepted_W=inputs.site_max_accepted_W[j],
                max_power_density_W_m2=inputs.max_power_density_W_m2,
                power=_power_series(p_site[:, j], t, w, window, dt, "this receiver's"),
                saturation_time_fraction=float(np.sum(w[rcol("saturated")[:, j].astype(bool)]))
                / window
                if window
                else 0.0,
                clipped_energy_J=float(np.sum(w * rcol("clipped_W")[:, j].astype(np.float64))),
                beams=_count_summary(beams[:, j], w, window),
                density_sum_max_W_m2=float(np.max(rcol("density_sum_W_m2")[:, j])),
                valid_links=_count_summary(rcol("valid_link_count")[:, j], w, window),
            )
        )
    active = _col(network, "active_satellite_count").astype(np.int64)
    n_sat = satellites.num_rows
    # Unavailability breakdowns.
    deliver = _col(network, "n_sat_delivering").astype(np.float64)
    denom = window * n_sat
    sat_frac = {
        name: float(np.sum(w * _col(network, f"n_sat_reason_{name}")) / denom) if denom else 0.0
        for name in REASON_NAMES
    }
    link_counts = {name: rcol(f"n_links_reason_{name}").astype(np.float64) for name in REASON_NAMES}
    link_denom = window * n_sat * n_s
    link_frac = {
        name: float(np.sum(w[:, None] * c) / link_denom) if link_denom else 0.0
        for name, c in link_counts.items()
    }
    energy_net = float(np.sum(w * p_net))
    energy_sites = float(sum(np.sum(w * p_site[:, j]) for j in range(n_s)))
    rel = _rel(energy_net, energy_sites)
    energy_recount = float(np.sum(w * _col(network, "grid_recount_W").astype(np.float64)))
    rel_recount = _rel(energy_net, energy_recount)
    sat_energy_sum = float(np.sum(_col(satellites, "delivered_energy_J").astype(np.float64)))
    rel_sat = _rel(energy_net, sat_energy_sum)
    beams_ok = bool(np.array_equal(beams.sum(axis=1), active))
    # Unconstrained baseline.
    energy_unc = float(np.sum(w * _col(network, "unconstrained_grid_W").astype(np.float64)))
    sat_unc = _col(satellites, "unconstrained_energy_J").astype(np.float64)
    # Reception-time reconstruction vs the emission grid (sample-and-hold for both).
    hold = np.append(np.diff(t), dt) if n_t else np.zeros(0)
    rt_checks = []
    for j in range(n_s):
        spread = rcol("light_time_spread_s")[:, j].astype(np.float64)
        multi = beams[:, j] >= 2
        acc_em = float(np.sum(hold * rcol("accepted_W")[:, j].astype(np.float64)))
        acc_rx = float(np.sum(rcol("rx_slot_accepted_energy_J")[:, j].astype(np.float64)))
        rt_checks.append(
            ReceptionTimeCheck(
                site_index=j,
                site_name=inputs.site_names[j],
                multi_beam_epochs=int(np.count_nonzero(multi)),
                light_time_spread_max_s=float(np.max(spread)) if n_t else 0.0,
                light_time_spread_median_s=float(np.median(spread[multi])) if multi.any() else None,
                max_beams=int(np.max(rcol("rx_slot_max_beams")[:, j])),
                beam_slot_overrun_time_s=float(np.sum(rcol("rx_slot_beam_overrun_s")[:, j])),
                max_density_sum_W_m2=float(np.max(rcol("rx_slot_max_density_W_m2")[:, j])),
                density_overrun_time_s=float(np.sum(rcol("rx_slot_density_overrun_s")[:, j])),
                incident_energy_emission_J=float(
                    np.sum(hold * rcol("incident_W")[:, j].astype(np.float64))
                ),
                incident_energy_reception_J=float(
                    np.sum(rcol("rx_slot_incident_energy_J")[:, j].astype(np.float64))
                ),
                accepted_energy_emission_J=acc_em,
                accepted_energy_reception_J=acc_rx,
                accepted_energy_relative_difference=(acc_rx - acc_em) / acc_em if acc_em else None,
            )
        )
    acc_em_all = float(sum(c.accepted_energy_emission_J for c in rt_checks))
    acc_rx_all = float(sum(c.accepted_energy_reception_J for c in rt_checks))
    col_area = float(np.sum(_col(satellites, "collector_area_m2")))
    rad_area = float(np.sum(_col(satellites, "radiator_area_required_m2")))
    tx_area = float(np.sum(_col(satellites, "transmitter_aperture_area_m2")))
    masses = _col(satellites, "mass_proxy_kg").astype(np.float64)
    mass = float(np.sum(masses)) if np.all(np.isfinite(masses)) and masses.size else None
    mean_p = energy_net / window if window > 0 else None
    incident = float(np.sum(w * _col(network, "incident_solar_W")))
    sat_energy = _col(satellites, "delivered_energy_J").astype(np.float64)
    ring_names = _col(satellites, "ring_name")
    methods_col = _col(network, "scheduler_method")
    return ConstellationSummary(
        summary_schema=CONSTELLATION_SUMMARY_SCHEMA,
        project_name=inputs.project_name,
        config_hash=inputs.config_hash,
        transmitter_mode=inputs.transmitter_mode,
        n_satellites=int(n_sat),
        n_sites=n_s,
        rings=[
            RingSummary(
                name=name,
                n_satellites=n,
                phase_distribution=dist,
                phase_reference=ref,
                delivered_energy_J=float(np.sum(sat_energy[ring_names == name])),
            )
            for name, n, dist, ref in inputs.rings
        ],
        scheduler=SchedulerSummary(
            policy=inputs.policy,
            duty_cycle_max=inputs.duty_cycle_max,
            duty_cycle_window_s=inputs.duty_cycle_window_s,
            retarget_model=inputs.retarget_model,
            retarget_events=int(np.sum(_col(network, "retarget_events"))),
            epochs_by_method={m: int(np.count_nonzero(methods_col == m)) for m in SOLVE_METHODS},
        ),
        time=TimeSummary(
            n_samples=int(n_t),
            output_step_s=dt,
            window_s=window,
            start_tdb_s=float(t[0]),
            end_tdb_s=float(t[-1]),
            sampling="trapezoid weights (end samples dt/2); emission epochs (TDB)",
        ),
        network=_power_series(p_net, t, w, window, dt, "the network"),
        receivers=rec_summaries,
        active_satellites=_count_summary(active, w, window),
        valid_links=_count_summary(_col(network, "valid_link_count"), w, window),
        scheduled_links=_count_summary(_col(network, "scheduled_link_count"), w, window),
        totals=TotalsSummary(
            collector_area_m2=col_area,
            radiator_area_required_m2=rad_area,
            radiator_area_definition="sum over satellites of the largest required radiator area "
            "of the satellite over the window (M4 sizing at the transmitter operating point)",
            transmitter_aperture_area_m2=tx_area,
            mass_proxy_kg=mass,
            mass_proxy_definition=inputs.mass_proxy_definition,
        ),
        power_per_collector_area_W_m2=mean_p / col_area if mean_p is not None else None,
        power_per_mass_W_kg=mean_p / mass if (mean_p is not None and mass) else None,
        end_to_end_efficiency=energy_net / incident if incident > 0 else None,
        end_to_end_efficiency_definition="E_grid,network / E_incident on all collectors",
        stage_energy_J={
            c.replace("_W", "_J"): float(np.sum(w * _col(network, c).astype(np.float64)))
            for c in STAGE_ENERGY_COLUMNS
        },
        unavailability=UnavailabilityBreakdown(
            satellite_time_fraction_delivering=float(np.sum(w * deliver) / denom) if denom else 0.0,
            satellite_time_fraction_primary=sat_frac,
            link_time_fraction_primary=link_frac,
            definition="satellite-time: a satellite delivering nothing is attributed the reason of "
            "its link that got furthest through the chain (largest precedence index over its "
            "sites; precedence: geometry, occultation, Sun avoidance, eclipse, thermal, weather, "
            "pointing, power-density abort, duty cycle, receiver capacity, scheduler); fractions "
            "of satellite-time sum with the delivering fraction to 1. link-time: first reason of "
            "each (satellite, site) link in the same precedence; scheduled links count as none",
        ),
        double_counting=DoubleCountingCheck(
            network_energy_J=energy_net,
            receivers_energy_sum_J=energy_sites,
            relative_difference=rel,
            recount_energy_J=energy_recount,
            recount_relative_difference=rel_recount,
            satellites_energy_sum_J=sat_energy_sum,
            satellites_relative_difference=rel_sat,
            beams_equal_active_satellites=beams_ok,
            passed=bool(rel <= 1e-12 and rel_recount <= 1e-12 and rel_sat <= 1e-12 and beams_ok),
            definition="relative_difference: network vs sum of receivers (consistency only: the "
            "network column is that sum by construction); recount_relative_difference: network "
            "vs sum_j min(sum of P_captured of the satellites assigned to j, L_j) eta_rx "
            "eta_grid recounted from the assignments and link powers; "
            "satellites_relative_difference: network vs the sum of the per-satellite pro-rata "
            "shares; beams_equal_active_satellites: scheduled beams = transmitting satellites "
            "at every epoch. Pass: all three relative differences <= 1e-12 and beams equal",
        ),
        unconstrained=UnconstrainedBaseline(
            energy_J=energy_unc,
            mean_W=energy_unc / window if window > 0 else None,
            capacity_loss_fraction=1.0 - energy_net / energy_unc if energy_unc > 0 else None,
            power_per_collector_area_W_m2=(
                energy_unc / window / col_area if window > 0 and col_area > 0 else None
            ),
            satellite_mean_energy_J=float(np.mean(sat_unc)) if sat_unc.size else 0.0,
            definition="every satellite on its largest-P_captured feasible link with no site "
            "capacity (no beam slots, input/heat or density limit, no duty cycle), times "
            "eta_rx eta_grid (= the sum of the satellites' runs with unlimited receivers); "
            "capacity_loss_fraction = 1 - E_grid / E_unconstrained; satellite_mean_energy_J is "
            "the mean over satellites (placement-neutral per-satellite baseline for an evenly "
            "phased ring)",
        ),
        reception_time=ReceptionTimeSummary(
            receivers=rt_checks,
            accepted_energy_emission_J=acc_em_all,
            accepted_energy_reception_J=acc_rx_all,
            accepted_energy_relative_difference=(
                (acc_rx_all - acc_em_all) / acc_em_all if acc_em_all else None
            ),
            beam_slot_overrun_time_s=float(sum(c.beam_slot_overrun_time_s for c in rt_checks)),
            density_overrun_time_s=float(sum(c.density_overrun_time_s for c in rt_checks)),
            definition="named approximation 'emission-epoch aggregation': the scheduler sums and "
            "limits the beams of one emission epoch, which reach a site up to the light-time "
            "spread apart. Check: each scheduled beam is held over its emission step and "
            "shifted by its own light time (range/c); per site, the arriving beams are counted "
            "and summed in continuous reception time. max_beams / beam_slot_overrun_time_s and "
            "max_density_sum_W_m2 / density_overrun_time_s show the slot and density limits "
            "seen at the receiver; accepted energies compare min(sum, L_j) integrated in "
            "reception time with the emission-grid min(sum, L_j) under the same hold "
            "convention (incident energies differ only by the light-time rate, ~|rdot|/c)",
        ),
        satellite_delivered_energy_J=SampleStatistics(
            min=float(np.min(sat_energy)),
            mean=float(np.mean(sat_energy)),
            median=float(np.median(sat_energy)),
            p05=float(np.percentile(sat_energy, 5.0)),
            p95=float(np.percentile(sat_energy, 95.0)),
            max=float(np.max(sat_energy)),
        ),
        incoherent_summation="received powers of independent satellites add (incoherent "
        "intensities); no field amplitudes are summed and no coherent combining is modelled",
        link_detail_saved=inputs.link_detail_saved,
        link_detail_rows=inputs.link_detail_rows,
    )


# ---------------------------------------------------------------------------------------------
# Writing a run
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstellationOutputs:
    """Files written for one constellation run.

    Attributes:
        run_dir: Run directory.
        summary: The summary record.
        files: Relative file name -> SHA-256.
        figures: Figure paths.
    """

    run_dir: Path
    summary: ConstellationSummary
    files: dict[str, str]
    figures: tuple[Path, ...]


def write_constellation_log(path: Path, result: ConstellationRunResult) -> Path:
    """``logs/run.log`` of a constellation run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{e.level:<7} [{e.stage}] {e.message}" for e in result.log]
    lines += [f"TIMING  [{t.stage}] {t.wall_s:.3f} s" for t in result.timings]
    n_warn = sum(1 for e in result.log if e.level == "WARNING")
    lines.append(
        f"SUMMARY {n_warn} warning(s); {len(result)} output epochs; {result.n_satellites} "
        f"satellites; {result.n_sites} sites"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_constellation_run(
    result: ConstellationRunResult,
    run_dir: Path,
    *,
    wall_clock_start_utc: str,
    link_detail: bool = False,
    link_detail_max_bytes: int = DEFAULT_LINK_DETAIL_MAX_BYTES,
    make_figures: bool = True,
    command: Sequence[str] | None = None,
) -> ConstellationOutputs:
    """Write every output of a constellation run into the prepared ``run_dir``.

    The summary is computed from the tables read back from disk (PDF §13.4).
    """
    from helioswarm.io import (
        CONFIG_FILE,
        LOG_FILE,
        MANIFEST_FILE,
        SUMMARY_FILE,
        file_sha256,
        table_content_sha256,
        write_json,
    )
    from helioswarm.manifest import build_manifest, utc_now_iso

    files: dict[str, str] = {}
    files[CONFIG_FILE] = file_sha256(write_resolved_yaml(result.config, run_dir / CONFIG_FILE))
    tables = {
        NETWORK_FILE: network_table(result),
        RECEIVERS_FILE: receivers_table(result),
        SATELLITES_FILE: satellites_table(result),
    }
    content: dict[str, str] = {}
    for name, table in tables.items():
        path = run_dir / name
        pq.write_table(table, path, compression="zstd")
        files[name] = file_sha256(path)
        content[name.replace(".parquet", "_content_sha256")] = table_content_sha256(table)
    if link_detail:
        for p in write_link_detail(result, run_dir / LINKS_DIR, max_bytes=link_detail_max_bytes):
            files[f"{LINKS_DIR}/{p.name}"] = file_sha256(p)
    saved = {name: pq.read_table(run_dir / name) for name in tables}
    summary = summarize_constellation(
        saved[NETWORK_FILE],
        saved[RECEIVERS_FILE],
        saved[SATELLITES_FILE],
        summary_inputs(result, link_detail_saved=link_detail),
    )
    files[SUMMARY_FILE] = file_sha256(write_json(run_dir / SUMMARY_FILE, summary.to_json_dict()))
    files[LOG_FILE] = file_sha256(write_constellation_log(run_dir / LOG_FILE, result))
    figures: tuple[Path, ...] = ()
    if make_figures:
        from helioswarm.plotting import make_constellation_figures

        figures = make_constellation_figures(run_dir)
    # The manifest is written last: its presence marks a complete run directory.
    manifest = build_manifest(
        result,
        wall_clock_start_utc=wall_clock_start_utc,
        wall_clock_end_utc=utc_now_iso(),
        outputs={**files, **content},
        command=list(command) if command is not None else None,
    )
    write_json(run_dir / MANIFEST_FILE, manifest.to_json_dict())
    return ConstellationOutputs(run_dir=run_dir, summary=summary, files=files, figures=figures)


# ---------------------------------------------------------------------------------------------
# Array-task link files
# ---------------------------------------------------------------------------------------------


def linkset_path(directory: Path, chunk_index: int, chunk_count: int) -> Path:
    """``<directory>/linkset_<i>_of_<n>.npz``."""
    return directory / f"{LINKSET_PREFIX}{chunk_index:05d}_of_{chunk_count:05d}.npz"


def save_link_set(path: Path, links: LinkSet) -> Path:
    """Write a :class:`~helioswarm.constellation.LinkSet` (atomic: temporary file + rename)."""
    meta = {
        "config_hash": links.config_hash,
        "itu_r_versions": list(links.itu_r_versions),
        "itu_r_polarization_tilt_deg": links.itu_r_polarization_tilt_deg,
        "source_ids": list(links.source_ids),
        "backend": links.backend,
        "timings": [[t.stage, t.wall_s] for t in links.timings],
        "log": [[e.level, e.stage, e.message] for e in links.log],
        "code_provenance": links.code_provenance,
    }
    arrays: dict[str, Any] = {
        "__meta__": np.array(json.dumps(meta)),
        "satellite_index": links.satellite_index,
        "epochs_tdb_s": links.epochs_tdb_s,
    }
    for name in links.LINK_FIELDS + links.SATELLITE_FIELDS + links.SCALAR_FIELDS:
        arrays[name] = getattr(links, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_link_set(path: Path) -> LinkSet:
    """Read a file written by :func:`save_link_set`."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        arrays = {k: np.array(z[k]) for k in z.files if k != "__meta__"}
    return LinkSet(
        config_hash=meta["config_hash"],
        itu_r_versions=tuple(meta["itu_r_versions"]),
        itu_r_polarization_tilt_deg=meta["itu_r_polarization_tilt_deg"],
        source_ids=tuple(meta["source_ids"]),
        backend=meta["backend"],
        timings=tuple(StageTiming(stage=s, wall_s=float(v)) for s, v in meta["timings"]),
        log=tuple(RunLogEntry(level, stage, msg) for level, stage, msg in meta["log"]),
        code_provenance=meta.get("code_provenance"),
        **arrays,
    )


def execute_link_chunk(
    config: RunConfig,
    candidates_dir: Path,
    *,
    chunk_index: int,
    chunk_count: int,
    kernel_set: KernelSet | None = None,
    backend: Backend = "numpy",
) -> Path:
    """Stage A for one array task's contiguous block of satellites
    (:func:`~helioswarm.constellation.chunk_indices`); writes its link file with the task's
    code provenance (:func:`helioswarm.manifest.code_provenance`)."""
    from dataclasses import replace

    from helioswarm.ephemeris import SpiceEphemeris
    from helioswarm.manifest import code_provenance

    if config.constellation is None:
        raise ValueError("configuration has no constellation section")
    members = constellation_members(config, np.random.default_rng(config.project.seed))
    block = chunk_indices(len(members), chunk_index, chunk_count)
    ks = kernel_set if kernel_set is not None else KernelSet.from_manifest()
    with ks.loaded(verify=True) as kernels:
        links = evaluate_links(
            config,
            [members[i] for i in block],
            ephemeris=SpiceEphemeris(kernels),
            kernels=kernels,
            backend=backend,
        )
    links = replace(links, code_provenance=code_provenance())
    return save_link_set(linkset_path(candidates_dir, chunk_index, chunk_count), links)


def load_link_chunks(candidates_dir: Path, config: RunConfig) -> LinkSet:
    """Merge every link file of ``candidates_dir`` (all from one chunk count) for ``config``.

    The chunks must agree on the configuration, the epochs, the beam backend and the code
    provenance (git commit, dirty flag, source-tree hash, package versions;
    :func:`~helioswarm.constellation.merge_link_sets`). The merging process does not have to
    run the same code (stage B is recorded in its own manifest), but a difference is logged
    as a WARNING and the chunks' provenance goes into the merged manifest.

    Raises:
        ValueError: no files, mixed chunk counts, a missing chunk, a different configuration,
            different backends or code provenance, or incomplete satellite coverage.
    """
    files = sorted(candidates_dir.glob(f"{LINKSET_PREFIX}*_of_*.npz"))
    if not files:
        raise ValueError(f"no link files in {candidates_dir}")
    counts = {f.stem.split("_of_")[1] for f in files}
    if len(counts) != 1:
        raise ValueError(f"link files from different chunk counts in {candidates_dir}: {counts}")
    n_chunks = int(next(iter(counts)))
    have = sorted(int(f.stem[len(LINKSET_PREFIX) :].split("_of_")[0]) for f in files)
    if have != list(range(n_chunks)):
        raise ValueError(f"missing link chunks in {candidates_dir}: have {len(have)} of {n_chunks}")
    sets = [load_link_set(f) for f in files]
    h = config_hash(config)
    for f, s in zip(files, sets, strict=True):
        if s.config_hash != h:
            raise ValueError(f"{f.name} was computed for configuration {s.config_hash[:12]}")
    assert config.constellation is not None
    return merge_link_sets(sets, config.constellation.n_satellites)


def _stage_a_provenance_entry(provenance: dict[str, Any] | None) -> RunLogEntry:
    """Log entry comparing the link files' code provenance with this process's."""
    from helioswarm.manifest import code_provenance, provenance_digest

    if provenance is None:
        return RunLogEntry(
            "WARNING",
            "stage_a",
            "the link files record no code provenance (written before it was recorded); the "
            "stage-A code cannot be verified",
        )
    here = code_provenance()
    if provenance_digest(provenance) == provenance_digest(here):
        return RunLogEntry(
            "INFO", "stage_a", "link files were computed by the same code as this merge"
        )
    return RunLogEntry(
        "WARNING",
        "stage_a",
        "link files were computed by different code than this merge (stage A: commit "
        f"{provenance.get('git_commit')}, dirty {provenance.get('git_dirty')}, source "
        f"{str(provenance.get('source_sha256'))[:16]}; this process: commit "
        f"{here.get('git_commit')}, dirty {here.get('git_dirty')}, source "
        f"{str(here.get('source_sha256'))[:16]}); recorded in manifest.json",
    )


def execute_constellation_run(
    config: RunConfig,
    run_dir: Path,
    *,
    kernel_set: KernelSet | None = None,
    backend: Backend = "numpy",
    candidates_dir: Path | None = None,
    link_detail: bool = False,
    link_detail_max_bytes: int = DEFAULT_LINK_DETAIL_MAX_BYTES,
    make_figures: bool = True,
    command: Sequence[str] | None = None,
) -> ConstellationOutputs:
    """Constellation run into a prepared ``run_dir``: stage A in this process (or merged from
    the link files of ``candidates_dir``), stage B, outputs."""
    import time as _time

    from helioswarm.ephemeris import SpiceEphemeris
    from helioswarm.manifest import utc_now_iso
    from helioswarm.timescale import tdb_s_to_utc, time_grid_from_config

    start = utc_now_iso()
    rng = np.random.default_rng(config.project.seed)
    state = rng.bit_generator.state
    members = constellation_members(config, rng)
    unchanged = rng.bit_generator.state == state
    ks = kernel_set if kernel_set is not None else KernelSet.from_manifest()
    with ks.loaded(verify=True) as kernels:
        t0 = _time.perf_counter()
        if candidates_dir is None:
            links = evaluate_links(
                config,
                members,
                ephemeris=SpiceEphemeris(kernels),
                kernels=kernels,
                backend=backend,
            )
            merged_log: tuple[RunLogEntry, ...] = ()
        else:
            links = load_link_chunks(candidates_dir, config)
            merged_log = (
                RunLogEntry(
                    "INFO",
                    "stage_a",
                    f"link sets merged from {candidates_dir} (array tasks; stage timings are "
                    "sums over tasks)",
                ),
                _stage_a_provenance_entry(links.code_provenance),
            )
        t_links = _time.perf_counter() - t0
        grid = time_grid_from_config(config.time, kernels)
        end_utc = tdb_s_to_utc(float(grid.epochs_tdb_s[-1]), kernels, decimals=3)
        provenance = tuple(kernels.provenance)
    result = finish_constellation(
        config,
        members,
        links,
        grid=grid,
        kernel_provenance=provenance,
        rng_state_unchanged=bool(unchanged),
        window_end_utc=end_utc,
        extra_log=config_log(config) + merged_log,
        extra_timings=(StageTiming(stage="stage_a_total", wall_s=t_links),),
    )
    return write_constellation_run(
        result,
        run_dir,
        wall_clock_start_utc=start,
        link_detail=link_detail,
        link_detail_max_bytes=link_detail_max_bytes,
        make_figures=make_figures,
        command=command,
    )


__all__ = [
    "BYTES_PER_GB",
    "CONSTELLATION_SUMMARY_SCHEMA",
    "DEFAULT_LINK_DETAIL_MAX_BYTES",
    "LINKS_DIR",
    "LINK_DETAIL_BYTES_PER_ROW",
    "NETWORK_FILE",
    "NETWORK_SCHEMA",
    "REASON_NAMES",
    "RECEIVERS_FILE",
    "SATELLITES_FILE",
    "STAGE_ENERGY_COLUMNS",
    "ConstellationOutputs",
    "ConstellationSummary",
    "ConstellationSummaryInputs",
    "CountSummary",
    "DoubleCountingCheck",
    "LinkDetailTooLargeError",
    "PowerSeriesSummary",
    "ReceiverSummary",
    "ReceptionTimeCheck",
    "ReceptionTimeSummary",
    "RingSummary",
    "SchedulerSummary",
    "TotalsSummary",
    "UnavailabilityBreakdown",
    "UnconstrainedBaseline",
    "execute_constellation_run",
    "execute_link_chunk",
    "is_constellation_summary",
    "link_detail_rows",
    "linkset_path",
    "load_link_chunks",
    "load_link_set",
    "mass_proxy_kg",
    "network_table",
    "proxy_radiator_area_m2",
    "read_constellation_summary",
    "receivers_table",
    "satellites_table",
    "save_link_set",
    "summarize_constellation",
    "summary_inputs",
    "write_constellation_log",
    "write_constellation_run",
    "write_link_detail",
]
