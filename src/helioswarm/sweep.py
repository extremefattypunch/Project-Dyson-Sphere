"""Deterministic parameter sweeps over dotted configuration keys (``helioswarm sweep``).

A sweep file names a base run configuration and a Cartesian grid of values for dotted
configuration keys::

    sweep_schema_version: 1
    name: optical_1au_sweep
    base_config: mvp_single_satellite.yaml      # relative to the sweep file
    overrides:                                  # optional, applied to every point
      time.duration_days: 365.25
    parameters:                                 # grid; the last key varies fastest
      orbit.semimajor_axis_au: [0.3, 0.5, 0.7, 1.0, 1.2]
      receiver.radius_m: [1000.0, 5000.0, 10000.0]

Point ``i`` of the grid (``itertools.product`` order) is the base mapping with the overrides
and its parameter values applied, ``project.name = <name>_p<iiii>`` and
``provenance_overrides[key] = "swept"`` for every swept key; it is validated as a complete
:class:`~helioswarm.config.RunConfig` before anything runs. Each point is a full run in
``<out>/points/p<iiii>/`` (with ``--summary-only``: everything except the time series and
figures). A point whose run raises (e.g. a beam model outside its validity range) is recorded
as **failed** with its error in ``<out>/points/p<iiii>/error.json`` and in the collated table;
failed and negative results are never dropped (PDF §13.4).

Slurm job arrays: ``--array-index i --array-count n`` runs the points ``p`` with
``p mod n == i`` (round robin; deterministic). Re-running a task resumes it: complete points
(manifest, summary and, with time series, report present) and failed points (``error.json``)
are skipped unless ``--overwrite``; an interrupted point (e.g. a task killed at its time limit)
is discarded and recomputed. Every task re-collates the finished points into
``sweep_summary.parquet`` when it ends; after the whole array, ``--collate-only`` rebuilds it
from every point directory (use a dependent job; see ``slurm/sweep_array.sbatch``).
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import shutil
import tempfile
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from helioswarm.config import (
    ConfigError,
    ConfigIssue,
    RunConfig,
    config_hash,
    leaf_paths,
    parse_config,
    read_yaml,
)
from helioswarm.system import Backend

SWEEP_SCHEMA_VERSION: Final[int] = 1
SWEEP_DEFINITION_FILE: Final[str] = "sweep_definition.yaml"
SWEEP_SUMMARY_FILE: Final[str] = "sweep_summary.parquet"
POINTS_DIR: Final[str] = "points"


class SweepFile(BaseModel):
    """Schema of a sweep YAML file (module docstring)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)

    sweep_schema_version: int
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=96)
    description: str = ""
    base_config: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, list[Any]]

    @field_validator("sweep_schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SWEEP_SCHEMA_VERSION:
            raise ValueError(f"unsupported sweep_schema_version {v} (expected 1)")
        return v

    @field_validator("parameters")
    @classmethod
    def _params(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not v:
            raise ValueError("parameters must name at least one key")
        for key, values in v.items():
            if not values:
                raise ValueError(f"parameter {key!r} has no values")
            if len({json.dumps(x, sort_keys=True) for x in values}) != len(values):
                raise ValueError(f"parameter {key!r} has duplicate values")
        return v


@dataclass(frozen=True)
class SweepPoint:
    """One validated grid point.

    Attributes:
        index: Position in the grid (product order).
        values: ``(key, value)`` pairs of the swept parameters.
        config: The validated run configuration.
    """

    index: int
    values: tuple[tuple[str, Any], ...]
    config: RunConfig

    @property
    def label(self) -> str:
        """Directory name ``p<iiii>``."""
        return f"p{self.index:04d}"


@dataclass(frozen=True)
class SweepPlan:
    """A loaded sweep: its definition, base mapping and every validated point.

    Attributes:
        path: Sweep file.
        definition: Parsed sweep file.
        base_path: Resolved base configuration path.
        points: Grid points in product order.
        definition_sha256: SHA-256 of the canonical JSON of the definition and base mapping.
    """

    path: Path
    definition: SweepFile
    base_path: Path
    points: tuple[SweepPoint, ...]
    definition_sha256: str


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    node: Any = data
    parts: list[str | int] = []
    for token in key.split("."):
        if "[" in token:
            name, idx = token[:-1].split("[")
            parts += [name, int(idx)]
        else:
            parts.append(token)
    for part in parts[:-1]:
        if isinstance(part, str):
            if not isinstance(node, dict) or part not in node:
                if isinstance(node, dict):
                    node[part] = {}
                else:
                    raise KeyError(key)
            node = node[part]
        else:
            node = node[part]
    last = parts[-1]
    if isinstance(last, int):
        node[last] = value
    else:
        node[last] = value


def load_sweep(path: str | Path) -> SweepPlan:
    """Read, validate and expand a sweep file (every point validated as a full config).

    Raises:
        ConfigError: invalid sweep file, unknown swept key, or an invalid point.
    """
    p = Path(path).resolve()
    raw = read_yaml(p)
    try:
        definition = SweepFile.model_validate(raw)
    except Exception as exc:
        raise ConfigError(str(p), [ConfigIssue("<sweep>", str(exc))]) from exc
    base_path = (p.parent / definition.base_config).resolve()
    base = read_yaml(base_path)
    if not isinstance(base, dict):
        raise ConfigError(str(base_path), [ConfigIssue("<root>", "base config is not a mapping")])
    base = copy.deepcopy(base)
    for key, value in definition.overrides.items():
        _set_dotted(base, key, value)
    base_cfg = parse_config(base, source=f"{base_path} + overrides")
    known = set(leaf_paths(base_cfg))
    for key in definition.parameters:
        if key not in known:
            raise ConfigError(
                str(p), [ConfigIssue(key, "swept key is not a configuration leaf path")]
            )
    keys = list(definition.parameters)
    points: list[SweepPoint] = []
    for index, combo in enumerate(itertools.product(*definition.parameters.values())):
        data = copy.deepcopy(base)
        for key, value in zip(keys, combo, strict=True):
            _set_dotted(data, key, value)
        data["project"]["name"] = f"{definition.name}_p{index:04d}"
        prov = dict(data.get("provenance_overrides") or {})
        prov.update(dict.fromkeys(keys, "swept"))
        data["provenance_overrides"] = prov
        cfg = parse_config(data, source=f"{p} point {index} {dict(zip(keys, combo, strict=True))}")
        points.append(
            SweepPoint(index=index, values=tuple(zip(keys, combo, strict=True)), config=cfg)
        )
    canon = json.dumps(
        {"definition": definition.model_dump(mode="json"), "base": base},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return SweepPlan(
        path=p,
        definition=definition,
        base_path=base_path,
        points=tuple(points),
        definition_sha256=hashlib.sha256(canon.encode()).hexdigest(),
    )


def points_for_task(n_points: int, array_index: int, array_count: int) -> tuple[int, ...]:
    """Point indices of one array task: ``p mod array_count == array_index`` (round robin)."""
    if array_count < 1 or not 0 <= array_index < array_count:
        raise ValueError("need array_count >= 1 and 0 <= array_index < array_count")
    return tuple(i for i in range(n_points) if i % array_count == array_index)


def prepare_sweep_dir(plan: SweepPlan, out: Path) -> Path:
    """Create the sweep directory and its ``sweep_definition.yaml`` (shared by array tasks).

    An existing directory is accepted only if it is empty or holds the *same* sweep
    definition (same hash), so array tasks can share it but two different sweeps cannot.
    Concurrent array tasks are safe: the marker is written to a private temporary file and
    published with ``os.link`` (atomic create-if-absent); a task that loses the race verifies
    the winner's definition hash instead.
    """
    out = out.expanduser().resolve()
    marker = out / SWEEP_DEFINITION_FILE
    header = f"# definition_sha256: {plan.definition_sha256}\n"

    def check_existing() -> Path:
        first = marker.read_text(encoding="utf-8").splitlines()[0] + "\n"
        if first != header:
            raise FileExistsError(
                f"{out} holds a different sweep definition ({first.strip()}); choose another --out"
            )
        return out

    if marker.is_file():
        return check_existing()
    if out.exists() and any(not p.name.startswith(".") for p in out.iterdir()):
        if marker.is_file():  # created by a concurrent task since the first check
            return check_existing()
        raise FileExistsError(f"{out} exists, is not empty and is not a helioswarm sweep dir")
    out.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        {
            "sweep_file": str(plan.path),
            "base_config": str(plan.base_path),
            "definition": plan.definition.model_dump(mode="json"),
            "n_points": len(plan.points),
            "points": [
                {
                    "index": pt.index,
                    "label": pt.label,
                    "values": dict(pt.values),
                    "config_hash": config_hash(pt.config),
                }
                for pt in plan.points
            ],
        },
        sort_keys=False,
    )
    tmp = out / f".{SWEEP_DEFINITION_FILE}.{os.getpid()}.{threading.get_ident()}"
    tmp.write_text(header + body, encoding="utf-8")
    try:
        os.link(tmp, marker)
    except FileExistsError:
        return check_existing()
    finally:
        tmp.unlink(missing_ok=True)
    return out


PointStatus = Literal["ok", "failed"]


def _point_complete(d: Path, manifest: str, summary: str, read_json: Callable[[Path], Any]) -> bool:
    """A point directory is complete when its manifest and summary exist and, if the time
    series was saved, its report (written last) exists too. A constellation run writes its
    manifest last, so manifest + summary mark it complete."""
    from helioswarm.constellation_io import is_constellation_summary

    if not ((d / manifest).is_file() and (d / summary).is_file()):
        return False
    doc = read_json(d / summary)
    if is_constellation_summary(doc):
        return True
    saved = bool(doc.get("timeseries_saved", True))
    return (not saved) or (d / "report.md").is_file()


def run_sweep_points(
    plan: SweepPlan,
    out: Path,
    indices: tuple[int, ...],
    *,
    summary_only: bool = False,
    overwrite: bool = False,
    backend: Backend = "numpy",
    log: Callable[[str], None] | None = None,
) -> dict[int, PointStatus]:
    """Run the given points (each a full validated run) into ``<out>/points/p<iiii>/``.

    Returns:
        Point index -> ``ok`` / ``failed``.
    """
    from helioswarm.io import (
        ERROR_FILE,
        MANIFEST_FILE,
        SUMMARY_FILE,
        execute_run,
        prepare_run_dir,
        read_json,
        write_json,
    )

    status: dict[int, PointStatus] = {}
    for i in indices:
        pt = plan.points[i]
        point_dir = out / POINTS_DIR / pt.label
        if point_dir.is_dir() and not overwrite:
            if (point_dir / ERROR_FILE).is_file():
                status[i] = "failed"
                if log is not None:
                    log(f"point {pt.label}: failed earlier (error.json kept; --overwrite reruns)")
                continue
            if _point_complete(point_dir, MANIFEST_FILE, SUMMARY_FILE, read_json):
                status[i] = "ok"
                if log is not None:
                    log(f"point {pt.label}: already complete, skipped (--overwrite recomputes)")
                continue
            # An interrupted attempt (e.g. a task killed at its time limit) inside this verified
            # sweep directory: its partial files are discarded and the point is recomputed.
            shutil.rmtree(point_dir)
        run_dir = prepare_run_dir(point_dir, overwrite=overwrite)
        try:
            command = ["helioswarm", "sweep", str(plan.path), f"point={i}"]
            if pt.config.constellation is not None:
                from helioswarm.constellation_io import execute_constellation_run

                # Constellation points always save their network/receiver/satellite tables
                # (the summary is computed from them); --summary-only skips the figures.
                execute_constellation_run(
                    pt.config,
                    run_dir,
                    backend=backend,
                    make_figures=not summary_only,
                    command=command,
                )
            else:
                execute_run(
                    pt.config,
                    run_dir,
                    backend=backend,
                    write_timeseries=not summary_only,
                    make_figures=not summary_only,
                    command=command,
                )
            status[i] = "ok"
        except Exception as exc:  # record every failure; never drop a point silently
            write_json(
                run_dir / ERROR_FILE,
                {
                    "point": i,
                    "values": dict(pt.values),
                    "config_hash": config_hash(pt.config),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            status[i] = "failed"
        if log is not None:
            log(f"point {pt.label} {dict(pt.values)}: {status[i]}")
    return status


SWEEP_SUMMARY_COLUMNS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("p_grid_mean_W", ("p_grid_W", "time_mean")),
    ("p_grid_median_W", ("p_grid_W", "median")),
    ("p_grid_p05_W", ("p_grid_W", "p05")),
    ("p_grid_p95_W", ("p_grid_W", "p95")),
    ("p_grid_peak_W", ("p_grid_W", "peak")),
    ("availability", ("availability",)),
    ("capacity_factor", ("capacity_factor",)),
    ("rated_power_W", ("capacity_factor_denominator", "rated_power_W")),
    ("reference_eta_atmosphere", ("capacity_factor_denominator", "reference_eta_atmosphere")),
    ("delivered_energy_J", ("delivered_energy_J",)),
    ("end_to_end_efficiency", ("end_to_end_efficiency", "energy_ratio")),
    ("pv_energy_J", ("stage_energy_J", "pv_electric_J")),
    ("aperture_energy_J", ("stage_energy_J", "aperture_J")),
    ("captured_energy_J", ("stage_energy_J", "captured_J")),
    ("eta_capture_median_available", ("beam_capture", "eta_capture_when_available", "median")),
    ("eta_capture_median_visible", ("beam_capture", "eta_capture_when_visible", "median")),
    ("thermal_feasible_fraction", ("thermal", "feasible_time_fraction")),
    ("pv_temperature_max_K", ("thermal", "pv_temperature_K", "max")),
    ("radiator_area_max_m2", ("thermal", "radiator_area_m2_max")),
    ("access_fraction", ("diurnal_aliasing", "access_fraction")),
    ("hour_angle_max_gap_rad", ("diurnal_aliasing", "hour_angle_max_gap_rad")),
    ("outage_count", ("outages", "count")),
    ("outage_longest_s", ("outages", "longest_s")),
    ("range_mean_m", ("geometry", "range_m", "mean")),
    ("sun_distance_mean_m", ("geometry", "sun_distance_m", "mean")),
    ("light_time_min_s", ("orbital_diagnostics", "light_time", "min_s")),
    ("light_time_max_s", ("orbital_diagnostics", "light_time", "max_s")),
    ("occultation_fraction", ("orbital_diagnostics", "occultation_fraction")),
)
"""Columns of ``sweep_summary.parquet`` taken from each point's summary.json (float64)."""

SWEEP_SUMMARY_FLAG_COLUMNS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("reference_thermal_feasible", ("capacity_factor_denominator", "reference_thermal_feasible")),
    (
        "reference_power_density_abort",
        ("capacity_factor_denominator", "reference_power_density_abort"),
    ),
    ("diurnal_aliasing_warning", ("diurnal_aliasing", "warning")),
)
"""Boolean columns of ``sweep_summary.parquet`` (null when the point has no summary)."""

CONSTELLATION_SWEEP_COLUMNS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("p_grid_mean_W", ("network", "p_grid_W", "time_mean")),
    ("p_grid_median_W", ("network", "p_grid_W", "median")),
    ("p_grid_p05_W", ("network", "p_grid_W", "p05")),
    ("p_grid_p95_W", ("network", "p_grid_W", "p95")),
    ("p_grid_peak_W", ("network", "p_grid_W", "peak")),
    ("availability", ("network", "availability")),
    ("delivered_energy_J", ("network", "delivered_energy_J")),
    ("end_to_end_efficiency", ("end_to_end_efficiency",)),
    ("pv_energy_J", ("stage_energy_J", "pv_electric_J")),
    ("aperture_energy_J", ("stage_energy_J", "aperture_J")),
    ("captured_energy_J", ("stage_energy_J", "captured_J")),
    ("outage_count", ("network", "outages", "count")),
    ("outage_longest_s", ("network", "outages", "longest_s")),
    ("n_satellites", ("n_satellites",)),
    ("active_satellites_mean", ("active_satellites", "time_mean")),
    ("valid_links_mean", ("valid_links", "time_mean")),
    ("collector_area_m2", ("totals", "collector_area_m2")),
    ("radiator_area_required_m2", ("totals", "radiator_area_required_m2")),
    ("transmitter_aperture_area_m2", ("totals", "transmitter_aperture_area_m2")),
    ("mass_proxy_kg", ("totals", "mass_proxy_kg")),
    ("power_per_collector_area_W_m2", ("power_per_collector_area_W_m2",)),
    ("power_per_mass_W_kg", ("power_per_mass_W_kg",)),
    ("unconstrained_energy_J", ("unconstrained", "energy_J")),
    ("capacity_loss_fraction", ("unconstrained", "capacity_loss_fraction")),
    (
        "unconstrained_power_per_collector_area_W_m2",
        ("unconstrained", "power_per_collector_area_W_m2"),
    ),
    (
        "reception_accepted_energy_rel_diff",
        ("reception_time", "accepted_energy_relative_difference"),
    ),
    ("reception_beam_slot_overrun_s", ("reception_time", "beam_slot_overrun_time_s")),
)
"""Columns of ``sweep_summary.parquet`` taken from a *constellation* point's summary.json
(the single-satellite-only columns are null for such points, and vice versa)."""


def _dig(doc: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def collate_sweep(plan: SweepPlan, out: Path) -> Path:
    """Collate every point directory into ``sweep_summary.parquet`` (atomic replace).

    One row per grid point, in index order: the swept values, the configuration hash, the
    status, the error message of failed points and the metrics of
    :data:`SWEEP_SUMMARY_COLUMNS` and :data:`SWEEP_SUMMARY_FLAG_COLUMNS`. Status uses the same
    completeness rule as the resume logic (``_point_complete``): ``ok`` (complete), ``failed``
    (``error.json``), ``incomplete`` (a point directory with partial files, e.g. a task killed
    between writing summary.json and report.md; no metrics are collated) or ``missing``.
    The temporary file is unique per writer (``tempfile.mkstemp`` in ``out``), so array tasks on
    different nodes can collate concurrently; the last ``os.replace`` wins and every version is
    a complete table.
    """
    from helioswarm.constellation_io import is_constellation_summary, read_constellation_summary
    from helioswarm.io import ERROR_FILE, MANIFEST_FILE, SUMMARY_FILE, read_json
    from helioswarm.metrics import read_summary

    rows: dict[str, list[Any]] = {"point": [], "status": [], "config_hash": [], "error": []}
    for key in plan.definition.parameters:
        rows[key] = []
    numeric = dict(SWEEP_SUMMARY_COLUMNS)
    constellation_cols = dict(CONSTELLATION_SWEEP_COLUMNS)
    all_numeric = list(numeric) + [c for c in constellation_cols if c not in numeric]
    for col in (*all_numeric, *(c for c, _ in SWEEP_SUMMARY_FLAG_COLUMNS)):
        rows[col] = []
    for pt in plan.points:
        d = out / POINTS_DIR / pt.label
        summary = None
        error = None
        if (d / ERROR_FILE).is_file():
            err = read_json(d / ERROR_FILE)
            error = f"{err['error_type']}: {err['error']}"
            st = "failed"
        elif d.is_dir() and _point_complete(d, MANIFEST_FILE, SUMMARY_FILE, read_json):
            # Validated against the typed summary layout before any value is collated.
            if is_constellation_summary(read_json(d / SUMMARY_FILE)):
                summary = read_constellation_summary(d / SUMMARY_FILE).to_json_dict()
            else:
                summary = read_summary(d / SUMMARY_FILE).to_json_dict()
            st = "ok"
        elif d.is_dir() and any(d.iterdir()):
            st = "incomplete"
        else:
            st = "missing"
        rows["point"].append(pt.index)
        rows["status"].append(st)
        rows["config_hash"].append(config_hash(pt.config))
        rows["error"].append(error)
        for key, value in pt.values:
            rows[key].append(value)
        paths = constellation_cols if is_constellation_summary(summary) else numeric
        for col in all_numeric:
            path = paths.get(col)
            v = _dig(summary, path) if (summary is not None and path is not None) else None
            rows[col].append(None if v is None else float(v))
        for col, path in SWEEP_SUMMARY_FLAG_COLUMNS:
            v = _dig(summary, path) if summary is not None else None
            rows[col].append(None if v is None else bool(v))
    table = pa.table(rows)
    dest = out / SWEEP_SUMMARY_FILE
    fd, tmp_name = tempfile.mkstemp(prefix=f".{SWEEP_SUMMARY_FILE}.", suffix=".tmp", dir=out)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


__all__ = [
    "CONSTELLATION_SWEEP_COLUMNS",
    "POINTS_DIR",
    "SWEEP_DEFINITION_FILE",
    "SWEEP_SCHEMA_VERSION",
    "SWEEP_SUMMARY_COLUMNS",
    "SWEEP_SUMMARY_FILE",
    "SWEEP_SUMMARY_FLAG_COLUMNS",
    "SweepFile",
    "SweepPlan",
    "SweepPoint",
    "collate_sweep",
    "load_sweep",
    "points_for_task",
    "prepare_sweep_dir",
    "run_sweep_points",
]
