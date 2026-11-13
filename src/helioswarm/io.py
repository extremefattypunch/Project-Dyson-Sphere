"""Run directories and run outputs (PDF §5 "Required run outputs", §14.1-14.2).

A run directory contains::

    config_resolved.yaml   fully resolved configuration (all defaults explicit)
    manifest.json          provenance (helioswarm.manifest.RunManifest)
    timeseries.parquet     time-indexed states, access flags, reason codes, every power stage,
                           thermal outputs and beam diagnostics (units in the column names)
    summary.json           metrics (helioswarm.metrics.summarize_timeseries of the saved table)
    figures/               plots made only from the saved files (helioswarm.plotting)
    report.md              one-page run report (helioswarm.report)
    logs/run.log           warnings, integrator/solver diagnostics, stage wall times

Location: the ``--out`` directory, else ``$HELIOSWARM_RUNS_DIR/<project.name>``, else
``runs/<project.name>`` relative to the working directory (``project.output_dir`` is
informational and not read). An existing non-empty directory is
never overwritten unless ``overwrite`` is set, and even then only if it is recognisably a
helioswarm run directory (it holds a ``manifest.json``), so a mistyped path cannot delete
unrelated data.
"""

from __future__ import annotations

import hashlib
import io as _io
import json
import math
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from helioswarm.config import RunConfig, write_resolved_yaml
from helioswarm.kernels import KernelSet
from helioswarm.metrics import RunSummary, SummaryInputs, summarize_timeseries
from helioswarm.system import Backend, SystemRunResult, primary_reason, run_single_satellite
from helioswarm.visibility import satellite_hour_angle_rad

TIMESERIES_SCHEMA: Final[str] = "helioswarm-timeseries/2"
"""Version tag stored in the Parquet schema metadata (2: M6b added ``transmit_dir_{x,y,z}``)."""

RUNS_DIR_ENV: Final[str] = "HELIOSWARM_RUNS_DIR"
"""Environment variable naming the default parent of run directories."""

CONFIG_FILE: Final[str] = "config_resolved.yaml"
MANIFEST_FILE: Final[str] = "manifest.json"
TIMESERIES_FILE: Final[str] = "timeseries.parquet"
SUMMARY_FILE: Final[str] = "summary.json"
LOG_FILE: Final[str] = "logs/run.log"
FIGURES_DIR: Final[str] = "figures"


class RunDirectoryError(RuntimeError):
    """The run directory cannot be used (exists without --overwrite, or is not a run dir)."""


def default_runs_dir() -> Path:
    """``$HELIOSWARM_RUNS_DIR`` if set, else ``runs`` (relative to the working directory)."""
    env = os.environ.get(RUNS_DIR_ENV)
    return Path(env) if env else Path("runs")


def resolve_run_dir(config: RunConfig, out: str | Path | None) -> Path:
    """Run directory: ``out`` or ``<default_runs_dir()>/<project.name>`` (absolute)."""
    path = Path(out) if out is not None else default_runs_dir() / config.project.name
    return path.expanduser().resolve()


ERROR_FILE: Final[str] = "error.json"
"""Written instead of the outputs when a sweep point fails (helioswarm.sweep)."""


def prepare_run_dir(
    path: Path, *, overwrite: bool, markers: tuple[str, ...] = (MANIFEST_FILE, ERROR_FILE)
) -> Path:
    """Create ``path`` (empty). An existing non-empty directory raises
    :class:`RunDirectoryError` unless ``overwrite`` is set and it contains one of ``markers``
    (a previous run's ``manifest.json`` or a failed sweep point's ``error.json``)."""
    if path.exists():
        if not path.is_dir():
            raise RunDirectoryError(f"{path} exists and is not a directory")
        if any(path.iterdir()):
            if not overwrite:
                raise RunDirectoryError(
                    f"run directory {path} already exists and is not empty; use --overwrite "
                    "to replace it"
                )
            if not any((path / m).is_file() for m in markers):
                raise RunDirectoryError(
                    f"refusing to overwrite {path}: it contains none of {list(markers)}, so it "
                    "is not recognisably a helioswarm output directory"
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _bool(a: Any) -> pa.Array:
    return pa.array(np.asarray(a, dtype=bool))


def _f64(a: Any) -> pa.Array:
    return pa.array(np.asarray(a, dtype=np.float64))


def timeseries_table(result: SystemRunResult) -> pa.Table:
    """The ``timeseries.parquet`` table of a run (one row per output epoch; SI units in the
    column names; J2000 / SSB states; TDB epochs)."""
    acc = result.access
    pt = result.power_thermal
    th = pt.thermal
    sol = pt.solar
    lk = result.link
    t = result.grid.epochs_tdb_s
    beam = result.beam
    nan_full = np.full(len(result), np.nan)

    def beam_col(name: str) -> Any:
        if beam is None:
            return nan_full
        return result.expand(np.asarray(getattr(beam, name), dtype=np.float64))

    def beam_diag(name: str) -> Any:
        if beam is None:
            return nan_full
        return result.expand(np.asarray(getattr(beam.diagnostics, name), dtype=np.float64))

    pos = result.trajectory.position_m
    vel = result.trajectory.velocity_m_s
    cols: dict[str, pa.Array] = {
        "t_emit_tdb_s": _f64(t),
        "t_arrival_tdb_s": _f64(acc.t_arrival_tdb_s),
        "elapsed_s": _f64(t - t[0]),
        "sat_x_m": _f64(pos[:, 0]),
        "sat_y_m": _f64(pos[:, 1]),
        "sat_z_m": _f64(pos[:, 2]),
        "sat_vx_m_s": _f64(vel[:, 0]),
        "sat_vy_m_s": _f64(vel[:, 1]),
        "sat_vz_m_s": _f64(vel[:, 2]),
        "orbit_drift_vs_l0_m": _f64(
            result.orbit_drift_m if result.orbit_drift_m is not None else nan_full
        ),
        "sun_distance_m": _f64(result.illumination.sun_distance_m),
        "illumination_fraction": _f64(result.illumination.illumination_fraction),
        "range_m": _f64(acc.range_m),
        "range_rate_m_s": _f64(acc.range_rate_m_s),
        "light_time_s": _f64(acc.light_time_s),
        "elevation_rad": _f64(acc.elevation_rad),
        "azimuth_rad": _f64(acc.azimuth_rad),
        "sun_separation_rad": _f64(acc.sun_separation_rad),
        "transmit_dir_x": _f64(acc.transmit_direction_unit[:, 0]),
        "transmit_dir_y": _f64(acc.transmit_direction_unit[:, 1]),
        "transmit_dir_z": _f64(acc.transmit_direction_unit[:, 2]),
        "pointing_rate_rad_s": _f64(acc.pointing_rate_rad_s),
        "transmitter_aberration_rad": _f64(acc.transmitter_aberration_rad),
        "hour_angle_rad": _f64(satellite_hour_angle_rad(acc)),
        "earth_blocked": _bool(acc.earth_blocked),
        "sun_blocked": _bool(acc.sun_blocked),
        "moon_blocked": _bool(acc.moon_blocked),
        "below_min_elevation": _bool(acc.below_min_elevation),
        "sun_avoidance_violation": _bool(acc.sun_avoidance_violation),
        "visible": _bool(acc.visible),
        "incident_solar_W": _f64(sol.incident_W),
        "absorbed_solar_W": _f64(sol.absorbed_W),
        "pv_temperature_K": _f64(sol.pv_temperature_K),
        "pv_efficiency": _f64(sol.pv_efficiency),
        "pv_heat_W": _f64(sol.pv_heat_W),
        "waste_heat_W": _f64(th.waste_heat_W),
        "radiator_area_m2": _f64(th.radiator_area_m2),
        "radiator_temperature_K": _f64(th.equilibrium_temperature_K),
        "thermal_margin_K": _f64(th.margin_K),
        "thermal_feasible": _bool(th.feasible),
        "fresnel_number": beam_col("fresnel_number"),
        "projected_receiver_radius_m": beam_col("projected_receiver_radius_m"),
        "capture_nominal": beam_col("capture_fraction_nominal"),
        "capture_bias_only": beam_col("capture_fraction_bias_only"),
        "capture_p05": beam_col("capture_fraction_p05"),
        "capture_p50": beam_col("capture_fraction_p50"),
        "capture_p95": beam_col("capture_fraction_p95"),
        "pointing_efficiency": beam_col("pointing_loss"),
        "peak_irradiance_per_W_m2": beam_col("peak_irradiance_per_W_m2"),
        "first_null_radius_m": beam_diag("first_null_radius_m"),
        "receiver_projection_error": beam_diag("receiver_projection_error"),
        "eta_atmosphere": _f64(lk.eta_atmosphere),
        "eta_capture": _f64(lk.eta_capture),
        "eta_receiver_acceptance": _f64(lk.eta_receiver_acceptance),
    }
    for attr in (
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
        "receiver_clipped_W",
        "receiver_heat_W",
        "requested_peak_power_density_W_m2",
        "peak_power_density_W_m2",
    ):
        cols[attr] = _f64(getattr(lk, attr))
    for attr in (
        "link_enabled",
        "transmitting",
        "power_density_abort",
        "receiver_saturated",
        "receiver_heat_limited",
    ):
        cols[attr] = _bool(getattr(lk, attr))
    cols["reason_code"] = pa.array(np.asarray(result.reason_code, dtype=np.uint32))
    cols["primary_reason"] = pa.array(primary_reason(result.reason_code).tolist(), pa.string())
    cols["available"] = _bool(result.available)
    meta = {
        b"helioswarm_timeseries_schema": TIMESERIES_SCHEMA.encode(),
        b"config_hash": result.config_hash.encode(),
        b"time_scale": b"TDB seconds past J2000 (emission epochs; t_arrival_tdb_s alongside)",
        b"frame": b"J2000",
        b"origin": b"SOLAR SYSTEM BARYCENTER",
        b"units": b"SI; unit suffix in every column name; dimensionless columns are ratios",
        b"eta_constants": json.dumps(
            {
                "eta_power_conditioning": lk.eta_power_conditioning,
                "eta_electrical_to_beam": lk.eta_electrical_to_beam,
                "eta_optics_or_rf": lk.eta_optics_or_rf,
                "eta_receiver_conversion": lk.eta_receiver_conversion,
                "eta_grid_conditioning": lk.eta_grid_conditioning,
            },
            sort_keys=True,
        ).encode(),
    }
    return pa.table(cols).replace_schema_metadata(meta)


def table_content_sha256(table: pa.Table) -> str:
    """SHA-256 of the table content (Arrow IPC stream of schema + columns, deterministic)."""
    sink = _io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_clean(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_clean(obj.item())
    return obj


def write_json(path: Path, document: Any) -> Path:
    """Write ``document`` as sorted, indented JSON (non-finite floats -> ``null``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_clean(document), sort_keys=True, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def summary_inputs(result: SystemRunResult) -> SummaryInputs:
    """:class:`~helioswarm.metrics.SummaryInputs` of a run."""
    cfg = result.config
    ref = result.reference
    return SummaryInputs(
        project_name=cfg.project.name,
        config_hash=result.config_hash,
        transmitter_mode=cfg.transmitter.mode,
        output_step_s=cfg.time.output_step_s,
        rated_power_W=ref.rated_power_W,
        rated_power_definition=ref.definition,
        reference_range_m=ref.range_m,
        reference_sun_distance_m=ref.sun_distance_m,
        reference_eta_capture=ref.eta_capture,
        reference_eta_atmosphere=ref.eta_atmosphere,
        reference_thermal_feasible=ref.thermal_feasible,
        reference_power_density_abort=ref.power_density_abort,
        max_power_density_W_m2=cfg.receiver.max_power_density_W_m2,
        max_input_power_W=cfg.receiver.site_max_input_power_W(cfg.receiver.sites[0]),
        collector_area_m2=cfg.collector.area_m2,
        earth_occultation_checked=cfg.models.earth_occultation,
        sun_occultation_checked=cfg.models.sun_occultation,
        moon_occultation_checked=cfg.models.moon_occultation,
    )


def write_run_log(path: Path, result: SystemRunResult) -> Path:
    """``logs/run.log``: diagnostics and warnings of the run, then stage wall times."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{e.level:<7} [{e.stage}] {e.message}" for e in result.log]
    lines += [f"TIMING  [{t.stage}] {t.wall_s:.3f} s" for t in result.timings]
    n_warn = sum(1 for e in result.log if e.level == "WARNING")
    lines.append(f"SUMMARY {n_warn} warning(s); {len(result)} output epochs")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class RunOutputs:
    """Files written for one run.

    Attributes:
        run_dir: Run directory.
        summary: The validated summary record (as written to summary.json).
        timeseries_content_sha256: Content hash of the time-series table.
        files: Relative file name -> SHA-256 of the written bytes.
        figures: Figure paths (empty without the time series).
    """

    run_dir: Path
    summary: RunSummary
    timeseries_content_sha256: str
    files: dict[str, str]
    figures: tuple[Path, ...]


def write_run(
    result: SystemRunResult,
    run_dir: Path,
    *,
    wall_clock_start_utc: str,
    write_timeseries: bool = True,
    make_figures: bool = True,
    command: Sequence[str] | None = None,
) -> RunOutputs:
    """Write every run output into an already prepared ``run_dir`` (module docstring).

    The summary is computed from the table that is written to ``timeseries.parquet`` (also when
    ``write_timeseries`` is false, e.g. ``sweep --summary-only``; then no figures are made,
    because figures are only drawn from saved data).
    """
    from helioswarm.manifest import build_manifest, utc_now_iso
    from helioswarm.plotting import make_figures as _make_figures
    from helioswarm.report import write_report

    files: dict[str, str] = {}
    cfg_path = write_resolved_yaml(result.config, run_dir / CONFIG_FILE)
    files[CONFIG_FILE] = file_sha256(cfg_path)
    table = timeseries_table(result)
    content_hash = table_content_sha256(table)
    if write_timeseries:
        ts_path = run_dir / TIMESERIES_FILE
        pq.write_table(table, ts_path, compression="zstd")
        files[TIMESERIES_FILE] = file_sha256(ts_path)
        table = pq.read_table(ts_path)  # metrics from the saved data
    summary = summarize_timeseries(table, summary_inputs(result)).model_copy(
        update={
            "timeseries_content_sha256": content_hash,
            "timeseries_saved": bool(write_timeseries),
        }
    )
    sum_path = write_json(run_dir / SUMMARY_FILE, summary.to_json_dict())
    files[SUMMARY_FILE] = file_sha256(sum_path)
    log_path = write_run_log(run_dir / LOG_FILE, result)
    files[LOG_FILE] = file_sha256(log_path)
    figures: tuple[Path, ...] = ()
    manifest = build_manifest(
        result,
        wall_clock_start_utc=wall_clock_start_utc,
        wall_clock_end_utc=utc_now_iso(),
        outputs={**files, "timeseries_content_sha256": content_hash},
        command=list(command) if command is not None else None,
    )
    write_json(run_dir / MANIFEST_FILE, manifest.to_json_dict())
    if write_timeseries and make_figures:
        figures = _make_figures(run_dir)
        write_report(run_dir)
    return RunOutputs(
        run_dir=run_dir,
        summary=summary,
        timeseries_content_sha256=content_hash,
        files=files,
        figures=figures,
    )


def read_timeseries(run_dir: Path) -> pa.Table:
    """The saved ``timeseries.parquet`` of a run directory."""
    return pq.read_table(Path(run_dir) / TIMESERIES_FILE)


def execute_run(
    config: RunConfig,
    run_dir: Path,
    *,
    kernel_set: KernelSet | None = None,
    backend: Backend = "numpy",
    write_timeseries: bool = True,
    make_figures: bool = True,
    command: Sequence[str] | None = None,
) -> RunOutputs:
    """Load and verify the kernels, run :func:`~helioswarm.system.run_single_satellite` with a
    Generator seeded from ``project.seed``, and write the outputs into ``run_dir`` (prepared).
    """
    from helioswarm.ephemeris import SpiceEphemeris
    from helioswarm.manifest import utc_now_iso

    start = utc_now_iso()
    ks = kernel_set if kernel_set is not None else KernelSet.from_manifest()
    with ks.loaded(verify=True) as kernels:
        result = run_single_satellite(
            config,
            ephemeris=SpiceEphemeris(kernels),
            kernels=kernels,
            rng=np.random.default_rng(config.project.seed),
            backend=backend,
        )
    return write_run(
        result,
        run_dir,
        wall_clock_start_utc=start,
        write_timeseries=write_timeseries,
        make_figures=make_figures,
        command=command,
    )


__all__ = [
    "CONFIG_FILE",
    "ERROR_FILE",
    "FIGURES_DIR",
    "LOG_FILE",
    "MANIFEST_FILE",
    "RUNS_DIR_ENV",
    "SUMMARY_FILE",
    "TIMESERIES_FILE",
    "TIMESERIES_SCHEMA",
    "RunDirectoryError",
    "RunOutputs",
    "default_runs_dir",
    "execute_run",
    "file_sha256",
    "prepare_run_dir",
    "read_json",
    "read_timeseries",
    "resolve_run_dir",
    "summary_inputs",
    "table_content_sha256",
    "timeseries_table",
    "write_json",
    "write_run",
    "write_run_log",
]
