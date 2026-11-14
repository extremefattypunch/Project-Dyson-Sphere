"""Design evaluation for optimisation and uncertainty studies (M8; PDF §12, §14.3).

One *design* is a set of dotted configuration overrides applied to a base run configuration.
:class:`ForwardModel` turns a design into a validated :class:`~helioswarm.config.RunConfig` at a
stated **fidelity level**, runs the forward simulator of record (the single-satellite chain of
:func:`helioswarm.system.run_single_satellite`, or the constellation chain of
:func:`helioswarm.constellation_io.execute_constellation_run` when the configuration has a
``constellation`` section) and reduces the run to named scalar **metrics**
(:data:`METRIC_DEFINITIONS`). No physics is re-implemented here: every metric is read from the
run's own summary or time series, or computed from configuration values with the existing
formulae (mass proxy: :func:`helioswarm.compare.mass_proxy_value_kg`; LCOE:
:func:`helioswarm.economics.evaluate_economics`).

Fidelity levels (PDF §14.3.3 coarse-to-fine; ``docs/model_assumptions.md`` M8)
-------------------------------------------------------------------------------
A fidelity level is a named set of dotted overrides (plus keys to remove) applied to the base
configuration *before* the design overrides, and the beam-capture backend. The shipped studies
use ``screening`` (L0 Kepler orbit, the day-incommensurate 3896 s grid of M7, the batched Airy
capture, scenario clear-sky weather) and ``high`` (L1 with planets and SRP, the per-epoch
reference capture, ERA5 cloud outages; 974 s for the single-satellite study, the unchanged
3896 s grid for the constellation study); both use the base file's beam model (Airy far field
for the optical studies, so near-field designs fail at both levels). Levels are stated in the
study configuration, never implied, and the differences are measured by re-evaluating selected
designs at ``high``. Design overrides are applied after the level's, so a design value always
wins: an optimisation study refuses a decision variable whose key a level also sets
(:func:`helioswarm.optimization.load_optimization`), whereas an uncertainty study deliberately
lets a sampled value override a level's base value of the same key (e.g. the weather year;
:mod:`helioswarm.uncertainty`).

Failures are results: a design whose configuration is invalid or whose run raises (e.g. a beam
model outside its validity range) is returned with ``status = "failed"`` and the error, never
dropped. A metric that is undefined for a run (NaN) is returned as NaN; the callers decide (an
optimisation treats a non-finite objective or constraint as an infeasible design).

Synthetic evaluators (:class:`SyntheticModel`) evaluate analytic test functions with known
optima (sphere, ZDT1, Binh-Korn, Ishigami) and one physics-derived case with a closed-form
optimum (Airy capture minus an areal aperture-mass penalty) through the same interface, so the
optimisation and sensitivity machinery is verified on problems whose answers are known
(PDF §12.3 step 2, §13.1 "optimization: synthetic known optimum").

Parallelism: :func:`evaluate_parallel` splits the tasks into chunks evaluated by a process pool
(``spawn``; each chunk loads and verifies the SPICE kernels once, so no state is shared between
processes); results are returned in task order and do not depend on the number of workers
(each evaluation is deterministic).
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from helioswarm.config import MassProxyConfig, RunConfig, config_hash, leaf_paths, parse_config
from helioswarm.system import Backend

Float64Array = NDArray[np.float64]
EvaluationStatus = Literal["ok", "failed"]

SPAWN_CONTEXT: Final[str] = "spawn"
"""Process start method of :func:`evaluate_parallel` (no inherited SPICE state)."""


# ---------------------------------------------------------------------------------------------
# Configuration pieces
# ---------------------------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


class FidelitySpec(_Model):
    """One fidelity level: dotted overrides applied to the base configuration (module docstring).

    Attributes:
        description: What the level stands for (written to every output).
        overrides: Dotted key -> value applied before the design overrides.
        unset: Dotted keys removed before the overrides (e.g. L1-only orbit keys for L0).
        backend: Beam-capture backend of the single-satellite chain.
        capture_percentiles: Compute the capture percentiles (diagnostics only; ``False``
            needs a batched backend and leaves every power/energy unchanged).
    """

    description: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    unset: tuple[str, ...] = ()
    backend: Backend = "numpy"
    capture_percentiles: bool = True


class EvaluationSettings(_Model):
    """Inputs of the metric layer other than the run itself.

    Attributes:
        power_threshold_W: Threshold of ``availability_above_threshold`` [W] (PDF §12.2
            "availability above a specified minimum power threshold").
        mass_proxy: Assumed specific masses of the M7 mass proxy (``mass_proxy_kg``); for a
            constellation the run's own ``constellation.mass_proxy`` is used.
        economics_config: Economics scenario file for ``lcoe_usd_per_kWh`` (absolute path), or
            ``None`` (the metric is then unavailable). Results depend on assumed economics.
    """

    power_threshold_W: float = Field(default=0.0, ge=0.0)
    mass_proxy: MassProxyConfig | None = None
    economics_config: str | None = None


@dataclass(frozen=True)
class MetricDefinition:
    """A named scalar result of one evaluated design.

    Attributes:
        unit: SI unit.
        description: Definition (with the governing summary field or formula).
        constellation: Available for constellation runs.
    """

    unit: str
    description: str
    constellation: bool = True


METRIC_DEFINITIONS: Final[Mapping[str, MetricDefinition]] = MappingProxyType(
    {
        "p_grid_mean_W": MetricDefinition(
            "W", "time-averaged grid power (trapezoid time mean; summary p_grid_W.time_mean)"
        ),
        "delivered_energy_J": MetricDefinition("J", "grid energy over the window"),
        "availability": MetricDefinition(
            "1", "time fraction with no blocking unavailability reason (network for constellations)"
        ),
        "availability_above_threshold": MetricDefinition(
            "1",
            "time fraction with P_grid >= power_threshold_W (trapezoid weights as availability)",
        ),
        "capacity_factor": MetricDefinition(
            "1", "mean P_grid / rated power (single satellite)", constellation=False
        ),
        "end_to_end_efficiency": MetricDefinition("1", "E_grid / E_sun over the window"),
        "mass_proxy_kg": MetricDefinition(
            "kg",
            "M7 mass proxy m_bus + s_col A_col + s_rad A_rad + s_tx pi D^2/4 (assumed specific "
            "masses; not a mass budget)",
        ),
        "collector_area_m2": MetricDefinition("m^2", "total collector area"),
        "transmitter_aperture_area_m2": MetricDefinition("m^2", "total aperture area pi D^2/4"),
        "receiver_area_m2": MetricDefinition("m^2", "receiver infrastructure n_sites pi R^2"),
        "pv_temperature_max_K": MetricDefinition(
            "K", "largest PV cell temperature of the run", constellation=False
        ),
        "thermal_feasible_fraction": MetricDefinition(
            "1", "time fraction of thermal feasibility", constellation=False
        ),
        "peak_incident_irradiance_W_m2": MetricDefinition(
            "W m^-2",
            "largest incident peak irradiance at the receiver while transmitting (0 if never)",
            constellation=False,
        ),
        "captured_power_peak_W": MetricDefinition(
            "W", "largest captured power at the receiver (receiver capacity)"
        ),
        "receiver_saturation_time_fraction": MetricDefinition(
            "1", "time fraction with the receiver input limit clipping", constellation=False
        ),
        "pointing_sigma_rad": MetricDefinition(
            "rad", "configured per-axis pointing jitter", constellation=False
        ),
        "aperture_diameter_m": MetricDefinition(
            "m", "configured transmitter aperture diameter", constellation=False
        ),
        "capacity_loss_fraction": MetricDefinition(
            "1",
            "energy lost to receiver capacity vs every satellite on its best link (constellation)",
        ),
        "lcoe_usd_per_kWh": MetricDefinition(
            "USD kWh^-1",
            "LCOE of the economics scenario (helioswarm.economics; depends on ASSUMED economics; "
            "NaN when undefined)",
        ),
    }
)
"""Metrics of the forward model (single satellite and, where marked, constellation)."""


# ---------------------------------------------------------------------------------------------
# Dotted mappings
# ---------------------------------------------------------------------------------------------


def _tokens(key: str) -> list[str | int]:
    parts: list[str | int] = []
    for token in key.split("."):
        if "[" in token:
            name, idx = token[:-1].split("[")
            parts += [name, int(idx)]
        else:
            parts.append(token)
    return parts


def set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    """Set ``data[a][b][i]... = value`` for ``key = "a.b[i]..."`` (missing mappings created)."""
    node: Any = data
    parts = _tokens(key)
    for part in parts[:-1]:
        if isinstance(part, str):
            if not isinstance(node, dict):
                raise KeyError(key)
            node = node.setdefault(part, {})
        else:
            node = node[part]
    node[parts[-1]] = copy.deepcopy(value)


def delete_dotted(data: dict[str, Any], key: str) -> None:
    """Remove ``key`` from ``data`` if present (no error when absent)."""
    node: Any = data
    parts = _tokens(key)
    for part in parts[:-1]:
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError):
            return
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def _leaf_keys(key: str, value: Any) -> list[str]:
    """Dotted leaf keys under ``key`` for a (possibly nested) override value."""
    if isinstance(value, Mapping):
        return [k for sub, v in value.items() for k in _leaf_keys(f"{key}.{sub}", v)]
    if isinstance(value, list | tuple) and value and isinstance(value[0], Mapping):
        return [k for i, v in enumerate(value) for k in _leaf_keys(f"{key}[{i}]", v)]
    return [key]


def get_dotted(data: Mapping[str, Any], key: str) -> Any:
    """``data[a][b][i]...`` for ``key = "a.b[i]..."``; ``KeyError`` when absent."""
    node: Any = data
    for part in _tokens(key):
        try:
            node = node[part]
        except (IndexError, TypeError) as exc:
            raise KeyError(key) from exc
    return node


# ---------------------------------------------------------------------------------------------
# Tasks and results
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignTask:
    """One design to evaluate.

    Attributes:
        index: Task index (results are returned in this order).
        label: Short label (becomes part of ``project.name``).
        overrides: ``(dotted key, value)`` pairs of the design.
    """

    index: int
    label: str
    overrides: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class DesignResult:
    """Outcome of one evaluated design.

    Attributes:
        index: Task index.
        label: Task label.
        status: ``ok`` or ``failed``.
        metrics: Metric name -> value (empty when failed; NaN when undefined).
        config_hash: Hash of the evaluated configuration (``None`` if it did not validate).
        error_type: Exception class name of a failure.
        error: Message of a failure.
        wall_s: Wall time of the evaluation [s].
    """

    index: int
    label: str
    status: EvaluationStatus
    metrics: Mapping[str, float]
    config_hash: str | None = None
    error_type: str | None = None
    error: str | None = None
    wall_s: float = 0.0


class Evaluator(Protocol):
    """Anything that evaluates a sequence of designs (picklable; see :func:`evaluate_parallel`)."""

    def evaluate(self, tasks: Sequence[DesignTask]) -> list[DesignResult]:
        """Evaluate ``tasks`` sequentially, in order."""
        ...

    def known_keys(self) -> frozenset[str]:
        """Dotted keys a design may set."""
        ...

    def metric_names(self) -> frozenset[str]:
        """Metrics this evaluator can return."""
        ...


# ---------------------------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------------------------


def _finite_or_nan(v: Any) -> float:
    if v is None:
        return math.nan
    return float(v)


def _threshold_fraction(grid_W: Float64Array, step_s: float, threshold_W: float) -> float:
    from helioswarm.metrics import sample_weights_s

    if grid_W.size < 2:
        return math.nan
    w = sample_weights_s(grid_W.size, step_s)
    return float(np.sum(w[grid_W >= threshold_W]) / np.sum(w))


def single_satellite_metrics(
    config: RunConfig,
    summary: Any,
    table: pa.Table,
    settings: EvaluationSettings,
) -> dict[str, float]:
    """Metrics of a single-satellite run from its summary (:class:`~helioswarm.metrics.
    RunSummary`) and time-series table (:data:`METRIC_DEFINITIONS`; LCOE excluded)."""
    from helioswarm.compare import mass_proxy_value_kg

    grid = np.asarray(table.column("grid_W").to_numpy(zero_copy_only=False), dtype=np.float64)
    captured = np.asarray(
        table.column("captured_W").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    d = config.transmitter.aperture_diameter_m
    peak = summary.receiver_power_density_W_m2.incident_peak_when_transmitting.max
    rad_area = float(summary.thermal.radiator_area_m2_max)
    m: dict[str, float] = {
        "p_grid_mean_W": _finite_or_nan(summary.p_grid_W.time_mean),
        "delivered_energy_J": float(summary.delivered_energy_J),
        "availability": float(summary.availability),
        "availability_above_threshold": _threshold_fraction(
            grid, config.time.output_step_s, settings.power_threshold_W
        ),
        "capacity_factor": _finite_or_nan(summary.capacity_factor),
        "end_to_end_efficiency": _finite_or_nan(summary.end_to_end_efficiency.energy_ratio),
        "collector_area_m2": float(config.collector.area_m2),
        "transmitter_aperture_area_m2": math.pi * d * d / 4.0,
        "receiver_area_m2": len(config.receiver.sites) * math.pi * config.receiver.radius_m**2,
        "pv_temperature_max_K": _finite_or_nan(summary.thermal.pv_temperature_K.max),
        "thermal_feasible_fraction": float(summary.thermal.feasible_time_fraction),
        "peak_incident_irradiance_W_m2": 0.0 if peak is None else float(peak),
        "captured_power_peak_W": float(np.max(captured)) if captured.size else 0.0,
        "receiver_saturation_time_fraction": float(summary.receiver_saturation_time_fraction),
        "pointing_sigma_rad": float(config.transmitter.pointing_sigma_rad),
        "aperture_diameter_m": float(d),
        "capacity_loss_fraction": 0.0,
    }
    if settings.mass_proxy is not None and config.thermal.radiator_specific_mass_kg_m2 is not None:
        m["mass_proxy_kg"] = mass_proxy_value_kg(settings.mass_proxy, config, rad_area)
    else:
        m["mass_proxy_kg"] = math.nan
    return m


def constellation_metrics(
    config: RunConfig, run_dir: Path, settings: EvaluationSettings
) -> dict[str, float]:
    """Metrics of a saved constellation run (``summary.json`` + ``network.parquet``)."""
    import pyarrow.parquet as pq

    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    net = pq.read_table(run_dir / "network.parquet")
    grid = np.asarray(net.column("grid_W").to_numpy(zero_copy_only=False), dtype=np.float64)
    captured = np.asarray(net.column("captured_W").to_numpy(zero_copy_only=False), dtype=np.float64)
    tot = s["totals"]
    mass = tot.get("mass_proxy_kg")
    return {
        "p_grid_mean_W": _finite_or_nan(s["network"]["p_grid_W"]["time_mean"]),
        "delivered_energy_J": float(s["network"]["delivered_energy_J"]),
        "availability": float(s["network"]["availability"]),
        "availability_above_threshold": _threshold_fraction(
            grid, config.time.output_step_s, settings.power_threshold_W
        ),
        "end_to_end_efficiency": _finite_or_nan(s.get("end_to_end_efficiency")),
        "mass_proxy_kg": _finite_or_nan(mass),
        "collector_area_m2": float(tot["collector_area_m2"]),
        "transmitter_aperture_area_m2": float(tot["transmitter_aperture_area_m2"]),
        "receiver_area_m2": len(config.receiver.sites) * math.pi * config.receiver.radius_m**2,
        "captured_power_peak_W": float(np.max(captured)) if captured.size else 0.0,
        "capacity_loss_fraction": _finite_or_nan(
            s.get("unconstrained", {}).get("capacity_loss_fraction")
        ),
    }


ECONOMICS_PREFIX: Final[str] = "economics."
"""Design keys ``economics.<name>`` vary an input of the economics scenario
(:data:`helioswarm.economics.UNCERTAIN_PARAMETERS`) instead of the run configuration."""

MASS_PROXY_PREFIX: Final[str] = "mass_proxy."
"""Design keys ``mass_proxy.<field>`` vary an assumed specific mass of the mass proxy."""


def lcoe_from_run_dir(
    run_dir: Path,
    economics_config: str,
    ephemeris: Any | None,
    values: Mapping[str, float] | None = None,
) -> float:
    """LCOE [USD/kWh] of a run directory with ``summary.json`` and ``config_resolved.yaml``
    under one economics scenario, with some inputs replaced (``values``, names of
    :data:`helioswarm.economics.UNCERTAIN_PARAMETERS`); NaN when undefined (zero or negligible
    energy)."""
    from helioswarm.economics import (
        evaluate_economics,
        load_economics_config,
        run_inputs,
        with_values,
    )

    cfg = load_economics_config(economics_config)
    if values:
        cfg = with_values(cfg, values)
    res = evaluate_economics(run_inputs(run_dir, ephemeris=ephemeris), cfg)
    return math.nan if res.lcoe_usd_per_kWh is None else float(res.lcoe_usd_per_kWh)


@dataclass(frozen=True)
class ForwardModel:
    """The forward simulator at one fidelity level (module docstring).

    Attributes:
        base_mapping: Resolved base configuration mapping (``extends`` already applied).
        base_source: Where the base came from (for error messages).
        fidelity_name: Name of the fidelity level.
        fidelity: The fidelity level.
        settings: Metric-layer settings.
        name_prefix: Prefix of ``project.name`` of every evaluated configuration.
        kernel_dir: Kernel directory (default: ``$HELIOSWARM_KERNEL_DIR``).
        swept_provenance: Mark every design key ``swept`` in ``provenance_overrides``.
        run_root: When set, every design is written as a complete run directory
            ``<run_root>/<label>/`` (time series, summary, manifest, figures, report) and its
            metrics are computed from the saved files (used for the high-fidelity
            re-evaluation, so that each re-evaluated design is a traceable run).
    """

    base_mapping: Mapping[str, Any]
    base_source: str
    fidelity_name: str
    fidelity: FidelitySpec
    settings: EvaluationSettings
    name_prefix: str = "design"
    kernel_dir: str | None = None
    swept_provenance: bool = True
    run_root: str | None = None

    def fidelity_mapping(self) -> dict[str, Any]:
        """The base mapping with the fidelity level applied."""
        data = copy.deepcopy(dict(self.base_mapping))
        for key in self.fidelity.unset:
            delete_dotted(data, key)
        for key, value in self.fidelity.overrides.items():
            set_dotted(data, key, value)
        return data

    def known_keys(self) -> frozenset[str]:
        """Leaf paths of the fidelity-level base configuration, plus ``economics.<name>``
        (with an economics scenario) and ``mass_proxy.<field>`` (with a mass proxy)."""
        from helioswarm.economics import UNCERTAIN_PARAMETERS

        keys = set(leaf_paths(parse_config(self.fidelity_mapping(), self.base_source)))
        if self.settings.economics_config is not None:
            keys |= {ECONOMICS_PREFIX + k for k in UNCERTAIN_PARAMETERS}
        if self.settings.mass_proxy is not None:
            keys |= {MASS_PROXY_PREFIX + k for k in MassProxyConfig.model_fields}
        return frozenset(keys)

    def split_overrides(
        self, task: DesignTask
    ) -> tuple[tuple[tuple[str, Any], ...], EvaluationSettings, dict[str, float]]:
        """``(configuration overrides, settings with the varied mass proxy, economics
        values)`` of a task."""
        cfg_o: list[tuple[str, Any]] = []
        econ: dict[str, float] = {}
        mass: dict[str, float] = {}
        for k, v in task.overrides:
            if k.startswith(ECONOMICS_PREFIX):
                econ[k[len(ECONOMICS_PREFIX) :]] = float(v)
            elif k.startswith(MASS_PROXY_PREFIX):
                mass[k[len(MASS_PROXY_PREFIX) :]] = float(v)
            else:
                cfg_o.append((k, v))
        settings = self.settings
        if mass:
            if settings.mass_proxy is None:
                raise ValueError("mass_proxy.* keys need a configured mass proxy")
            mp = MassProxyConfig.model_validate({**settings.mass_proxy.model_dump(), **mass})
            settings = settings.model_copy(update={"mass_proxy": mp})
        if econ and settings.economics_config is None:
            raise ValueError("economics.* keys need an economics scenario")
        return tuple(cfg_o), settings, econ

    def metric_names(self) -> frozenset[str]:
        """Every metric of :data:`METRIC_DEFINITIONS`."""
        return frozenset(METRIC_DEFINITIONS)

    def config_for(self, task: DesignTask) -> RunConfig:
        """Validated configuration of ``task`` (raises :class:`~helioswarm.config.ConfigError`)."""
        data = self.fidelity_mapping()
        overrides, _, _ = self.split_overrides(task)
        for key, value in overrides:
            set_dotted(data, key, value)
        project = dict(data.get("project") or {})
        project["name"] = f"{self.name_prefix}_{task.label}"[:96]
        data["project"] = project
        if self.swept_provenance and overrides:
            prov = dict(data.get("provenance_overrides") or {})
            prov.update(dict.fromkeys((k for o in overrides for k in _leaf_keys(*o)), "swept"))
            data["provenance_overrides"] = prov
        return parse_config(data, source=f"{self.base_source} [{self.fidelity_name}] {task.label}")

    def _kernel_set(self) -> Any:
        from helioswarm.kernels import KernelSet

        return KernelSet.from_manifest(kernel_dir=self.kernel_dir)

    def _single(
        self,
        config: RunConfig,
        kernels: Any,
        label: str,
        settings: EvaluationSettings,
        econ: Mapping[str, float],
    ) -> dict[str, float]:
        from helioswarm.config import write_resolved_yaml
        from helioswarm.ephemeris import SpiceEphemeris
        from helioswarm.io import (
            prepare_run_dir,
            read_timeseries,
            summary_inputs,
            timeseries_table,
            write_json,
            write_run,
        )
        from helioswarm.manifest import utc_now_iso
        from helioswarm.metrics import summarize_timeseries
        from helioswarm.system import run_single_satellite

        eph = SpiceEphemeris(kernels)
        start = utc_now_iso()
        result = run_single_satellite(
            config,
            ephemeris=eph,
            kernels=kernels,
            rng=np.random.default_rng(config.project.seed),
            backend=self.fidelity.backend,
            capture_percentiles=self.fidelity.capture_percentiles,
        )
        if self.run_root is not None:
            run_dir = prepare_run_dir(Path(self.run_root) / label, overwrite=True)
            outputs = write_run(
                result,
                run_dir,
                wall_clock_start_utc=start,
                command=["helioswarm", "evaluation", self.fidelity_name, label],
            )
            metrics = single_satellite_metrics(
                config, outputs.summary, read_timeseries(run_dir), settings
            )
            metrics["lcoe_usd_per_kWh"] = (
                lcoe_from_run_dir(run_dir, settings.economics_config, eph, econ)
                if settings.economics_config is not None
                else math.nan
            )
            return metrics
        table = timeseries_table(result)
        summary = summarize_timeseries(table, summary_inputs(result))
        metrics = single_satellite_metrics(config, summary, table, settings)
        if settings.economics_config is not None:
            with tempfile.TemporaryDirectory(prefix="helioswarm-lcoe-") as tmp:
                d = Path(tmp)
                write_resolved_yaml(config, d / "config_resolved.yaml")
                write_json(d / "summary.json", summary.to_json_dict())
                metrics["lcoe_usd_per_kWh"] = lcoe_from_run_dir(
                    d, settings.economics_config, eph, econ
                )
        else:
            metrics["lcoe_usd_per_kWh"] = math.nan
        return metrics

    def _constellation_into(
        self,
        config: RunConfig,
        run_dir: Path,
        settings: EvaluationSettings,
        econ: Mapping[str, float],
    ) -> dict[str, float]:
        from helioswarm.constellation_io import execute_constellation_run

        execute_constellation_run(
            config,
            run_dir,
            kernel_set=self._kernel_set(),
            backend=self.fidelity.backend,
            make_figures=self.run_root is not None,
            command=["helioswarm", "evaluation", self.fidelity_name, config.project.name],
        )
        metrics = constellation_metrics(config, run_dir, settings)
        metrics["lcoe_usd_per_kWh"] = (
            lcoe_from_run_dir(run_dir, settings.economics_config, None, econ)
            if settings.economics_config is not None
            else math.nan
        )
        return metrics

    def _constellation(
        self,
        config: RunConfig,
        label: str,
        settings: EvaluationSettings,
        econ: Mapping[str, float],
    ) -> dict[str, float]:
        from helioswarm.io import prepare_run_dir

        if self.run_root is not None:
            run_dir = prepare_run_dir(Path(self.run_root) / label, overwrite=True)
            return self._constellation_into(config, run_dir, settings, econ)
        with tempfile.TemporaryDirectory(prefix="helioswarm-eval-") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            return self._constellation_into(config, run_dir, settings, econ)

    def evaluate(self, tasks: Sequence[DesignTask]) -> list[DesignResult]:
        """Evaluate ``tasks`` in order (kernels loaded once for the single-satellite designs)."""
        results: list[DesignResult] = []
        configs: list[RunConfig | BaseException] = []
        for task in tasks:
            try:
                configs.append(self.config_for(task))
            except Exception as exc:  # invalid design: a result, not a crash
                configs.append(exc)
        need_kernels = any(isinstance(c, RunConfig) and c.constellation is None for c in configs)
        ks = self._kernel_set() if need_kernels else None
        ctx: Any = ks.loaded(verify=True) if ks is not None else _NullContext()
        with ctx as kernels:
            for task, cfg in zip(tasks, configs, strict=True):
                t0 = time.perf_counter()
                if isinstance(cfg, BaseException):
                    results.append(_failure(task, cfg, None, t0))
                    continue
                h = config_hash(cfg)
                try:
                    _, settings, econ = self.split_overrides(task)
                    metrics = (
                        self._single(cfg, kernels, task.label, settings, econ)
                        if cfg.constellation is None
                        else self._constellation(cfg, task.label, settings, econ)
                    )
                except Exception as exc:  # recorded, never dropped (PDF §13.4)
                    results.append(_failure(task, exc, h, t0))
                    continue
                results.append(
                    DesignResult(
                        index=task.index,
                        label=task.label,
                        status="ok",
                        metrics=metrics,
                        config_hash=h,
                        wall_s=time.perf_counter() - t0,
                    )
                )
        return results


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _failure(
    task: DesignTask, exc: BaseException, h: str | None, t0: float, *, tb: bool = False
) -> DesignResult:
    msg = str(exc)
    if tb:
        msg += "\n" + "".join(traceback.format_exception(exc))
    return DesignResult(
        index=task.index,
        label=task.label,
        status="failed",
        metrics={},
        config_hash=h,
        error_type=type(exc).__name__,
        error=msg[:4000],
        wall_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------------------------
# Synthetic evaluators (verification; PDF §12.3 step 2)
# ---------------------------------------------------------------------------------------------


def _x(args: Mapping[str, float], n: int) -> Float64Array:
    return np.array([float(args[f"x{i}"]) for i in range(n)], dtype=np.float64)


def _sphere(args: Mapping[str, float], p: Mapping[str, float]) -> dict[str, float]:
    """``f = sum_i (x_i - c_i)^2``, minimum 0 at ``x_i = c_i = p['c<i>']`` (default 0)."""
    n = len([k for k in args if k.startswith("x")])
    x = _x(args, n)
    c = np.array([float(p.get(f"c{i}", 0.0)) for i in range(n)])
    return {"f": float(np.sum((x - c) ** 2))}


def _zdt1(args: Mapping[str, float], p: Mapping[str, float]) -> dict[str, float]:
    """ZDT1 (Zitzler, Deb & Thiele 2000, Evol. Comput. 8(2) 173): ``f1 = x0``,
    ``g = 1 + 9 sum_{i>=1} x_i/(n-1)``, ``f2 = g (1 - sqrt(f1/g))``; Pareto front
    ``f2 = 1 - sqrt(f1)`` at ``x_i = 0`` (i >= 1)."""
    n = len([k for k in args if k.startswith("x")])
    x = _x(args, n)
    g = 1.0 + 9.0 * float(np.sum(x[1:])) / (n - 1)
    f1 = float(x[0])
    return {"f1": f1, "f2": g * (1.0 - math.sqrt(f1 / g))}


def _bnh(args: Mapping[str, float], p: Mapping[str, float]) -> dict[str, float]:
    """Binh & Korn (1997) constrained problem: ``f1 = 4x0^2 + 4x1^2``, ``f2 = (x0-5)^2 +
    (x1-5)^2``, constraints ``c1 = (x0-5)^2 + x1^2 <= 25``, ``c2 = (x0-8)^2 + (x1+3)^2 >= 7.7``."""
    x0, x1 = float(args["x0"]), float(args["x1"])
    return {
        "f1": 4 * x0**2 + 4 * x1**2,
        "f2": (x0 - 5) ** 2 + (x1 - 5) ** 2,
        "c1": (x0 - 5) ** 2 + x1**2,
        "c2": (x0 - 8) ** 2 + (x1 + 3) ** 2,
    }


def _ishigami(args: Mapping[str, float], p: Mapping[str, float]) -> dict[str, float]:
    """Ishigami & Homma (1990): ``y = sin x0 + a sin^2 x1 + b x2^4 sin x0`` (a = 7, b = 0.1
    unless set), inputs uniform on ``[-pi, pi]``."""
    a = float(p.get("a", 7.0))
    b = float(p.get("b", 0.1))
    x0, x1, x2 = float(args["x0"]), float(args["x1"]), float(args["x2"])
    return {"y": math.sin(x0) + a * math.sin(x1) ** 2 + b * x2**4 * math.sin(x0)}


def _linear(args: Mapping[str, float], p: Mapping[str, float]) -> dict[str, float]:
    """``y = sum_i w_i x_i`` (``w_i = p['w<i>']``, default 1): variance bookkeeping checks
    (``Var y = sum w_i^2 Var x_i`` for independent inputs)."""
    n = len([k for k in args if k.startswith("x")])
    x = _x(args, n)
    w = np.array([float(p.get(f"w{i}", 1.0)) for i in range(n)])
    return {"y": float(np.dot(w, x))}


def _airy_aperture_net_power(args: Mapping[str, float], p: Mapping[str, float]) -> dict[str, float]:
    """Physics-derived case with a closed-form optimum: centred Airy beam of power ``P0`` on a
    receiver of radius ``a`` at range ``R`` (helioswarm.optical.AiryPattern encircled energy),
    minus an areal aperture-mass penalty ``c pi D^2/4``:
    ``J(D) = P0 EE(pi D a/(lambda R)) - c pi D^2/4``. With ``x = beta D``,
    ``beta = pi a/(lambda R)`` and ``dEE/dx = 2 J1(x)^2/x``, the maximum satisfies
    ``J1(x*)/x* = sqrt(c pi/(4 P0))/beta`` (unique on ``(0, j_{1,1})`` when the right side is
    in ``(0, 1/2)``)."""
    from helioswarm.optical import AiryPattern

    d = float(args["aperture_diameter_m"])
    pat = AiryPattern(
        aperture_diameter_m=d,
        wavelength_m=float(p["wavelength_m"]),
        range_m=float(p["range_m"]),
        max_fresnel_number=float(p.get("max_fresnel_number", 0.5)),
    )
    ee = float(pat.encircled_energy(np.array([float(p["receiver_radius_m"])]))[0])
    captured = float(p["beam_power_W"]) * ee
    penalty = float(p["penalty_W_per_m2"]) * math.pi * d * d / 4.0
    return {"net_power_W": captured - penalty, "captured_power_W": captured, "penalty_W": penalty}


SYNTHETIC_FUNCTIONS: Final[
    Mapping[str, Callable[[Mapping[str, float], Mapping[str, float]], dict[str, float]]]
] = MappingProxyType(
    {
        "sphere": _sphere,
        "zdt1": _zdt1,
        "bnh": _bnh,
        "ishigami": _ishigami,
        "linear": _linear,
        "airy_aperture_net_power": _airy_aperture_net_power,
    }
)
"""Analytic evaluators with known optima / sensitivity indices (verification only)."""

SYNTHETIC_ARGUMENTS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "bnh": frozenset({"x0", "x1"}),
        "ishigami": frozenset({"x0", "x1", "x2"}),
        "airy_aperture_net_power": frozenset({"aperture_diameter_m"}),
    }
)
"""Fixed argument names (the other synthetic functions take ``x0 .. x{n-1}``)."""

SYNTHETIC_METRICS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "sphere": frozenset({"f"}),
        "zdt1": frozenset({"f1", "f2"}),
        "bnh": frozenset({"f1", "f2", "c1", "c2"}),
        "ishigami": frozenset({"y"}),
        "linear": frozenset({"y"}),
        "airy_aperture_net_power": frozenset({"net_power_W", "captured_power_W", "penalty_W"}),
    }
)


@dataclass(frozen=True)
class SyntheticModel:
    """Analytic evaluator: a design's overrides are the function arguments (module docstring).

    Attributes:
        name: Key of :data:`SYNTHETIC_FUNCTIONS`.
        arguments: Argument names a design may set.
        parameters: Fixed function parameters.
    """

    name: str
    arguments: tuple[str, ...]
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in SYNTHETIC_FUNCTIONS:
            raise ValueError(
                f"unknown synthetic evaluator {self.name!r}; known: {sorted(SYNTHETIC_FUNCTIONS)}"
            )

    def known_keys(self) -> frozenset[str]:
        """Arguments of the function: the fixed names of ``bnh``, ``ishigami`` and
        ``airy_aperture_net_power``; ``x0 .. x{n-1}`` (n = number of given arguments) for the
        vector functions ``sphere``, ``zdt1`` and ``linear``."""
        fixed = SYNTHETIC_ARGUMENTS.get(self.name)
        if fixed is not None:
            return fixed
        return frozenset(f"x{i}" for i in range(len(self.arguments)))

    def metric_names(self) -> frozenset[str]:
        """Outputs of the function."""
        return SYNTHETIC_METRICS[self.name]

    def evaluate(self, tasks: Sequence[DesignTask]) -> list[DesignResult]:
        """Evaluate every task (failures recorded as for the forward model)."""
        fn = SYNTHETIC_FUNCTIONS[self.name]
        out: list[DesignResult] = []
        for task in tasks:
            t0 = time.perf_counter()
            try:
                metrics = fn(dict(task.overrides), self.parameters)
            except Exception as exc:
                out.append(_failure(task, exc, None, t0))
                continue
            out.append(
                DesignResult(
                    index=task.index,
                    label=task.label,
                    status="ok",
                    metrics=metrics,
                    config_hash=None,
                    wall_s=time.perf_counter() - t0,
                )
            )
        return out


# ---------------------------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------------------------


def _evaluate_chunk(evaluator: Evaluator, tasks: Sequence[DesignTask]) -> list[DesignResult]:
    return evaluator.evaluate(tasks)


def chunk_tasks(tasks: Sequence[DesignTask], workers: int) -> list[list[DesignTask]]:
    """Split ``tasks`` into contiguous chunks, about two per worker (load balance vs the
    per-chunk kernel load); one chunk per task when there are fewer tasks than chunks."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    n_chunks = min(len(tasks), 2 * workers) if tasks else 0
    if n_chunks == 0:
        return []
    bounds = np.linspace(0, len(tasks), n_chunks + 1).round().astype(int)
    return [list(tasks[bounds[i] : bounds[i + 1]]) for i in range(n_chunks)]


def evaluate_parallel(
    evaluator: Evaluator,
    tasks: Sequence[DesignTask],
    *,
    workers: int = 1,
    executor: ProcessPoolExecutor | None = None,
) -> list[DesignResult]:
    """Evaluate ``tasks`` with ``workers`` processes; results in task order (module docstring).

    ``workers = 1`` evaluates in this process. An ``executor`` may be passed to reuse a pool
    across calls (e.g. one optimisation generation after another).
    """
    if workers <= 1 and executor is None:
        return sorted(evaluator.evaluate(tasks), key=lambda r: r.index)
    chunks = chunk_tasks(tasks, workers)
    own = executor is None
    pool = executor or ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context(SPAWN_CONTEXT)
    )
    try:
        futures = [pool.submit(_evaluate_chunk, evaluator, c) for c in chunks]
        results = [r for f in futures for r in f.result()]
    finally:
        if own:
            pool.shutdown()
    return sorted(results, key=lambda r: r.index)


def make_pool(workers: int) -> ProcessPoolExecutor | None:
    """A ``spawn`` process pool for ``workers > 1``, else ``None`` (evaluate in-process)."""
    if workers <= 1:
        return None
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context(SPAWN_CONTEXT))


def build_evaluator(
    evaluator: str,
    *,
    study_path: Path,
    base_config: str | None,
    fidelity_name: str,
    fidelity: FidelitySpec | None,
    settings: EvaluationSettings,
    arguments: Sequence[str] = (),
    synthetic_parameters: Mapping[str, float] | None = None,
    name_prefix: str = "design",
    run_root: str | None = None,
) -> ForwardModel | SyntheticModel:
    """The evaluator of a study file: the forward model at ``fidelity`` (base configuration
    relative to the study file, ``extends`` resolved) or a synthetic function.

    Raises:
        ValueError: missing base configuration or fidelity level for the forward model.
    """
    from helioswarm.config import read_config_mapping

    if evaluator != "forward_model":
        return SyntheticModel(
            name=evaluator, arguments=tuple(arguments), parameters=dict(synthetic_parameters or {})
        )
    if base_config is None or fidelity is None:
        raise ValueError("the forward_model evaluator needs base_config and a fidelity level")
    base_path = (study_path.parent / base_config).resolve()
    base = read_config_mapping(base_path)
    econ = settings.economics_config
    if econ is not None and not Path(econ).is_absolute():
        settings = settings.model_copy(
            update={"economics_config": str((study_path.parent / econ).resolve())}
        )
    return ForwardModel(
        base_mapping=base,
        base_source=str(base_path),
        fidelity_name=fidelity_name,
        fidelity=fidelity,
        settings=settings,
        name_prefix=name_prefix,
        run_root=run_root,
    )


STUDY_PACKAGES: Final[tuple[str, ...]] = ("pymoo", "SALib", "numpy", "scipy", "pyarrow")
"""Distributions whose versions every optimisation/uncertainty manifest records."""


def study_provenance() -> dict[str, Any]:
    """Provenance block of an optimisation or uncertainty output (PDF §14.1): code
    (:func:`helioswarm.manifest.code_provenance`), study package versions, CPU, Slurm ids.
    Returned as the JSON-ready mapping that is written into ``run.json`` / ``summary.json``
    (IO boundary only; no model code consumes it)."""
    import platform

    from helioswarm.manifest import code_provenance, cpu_model, package_versions, slurm_info

    return {
        "code": code_provenance(),
        "study_packages": package_versions(STUDY_PACKAGES),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "slurm": slurm_info(),
    }


def results_table(results: Sequence[DesignResult], metric_names: Sequence[str]) -> pa.Table:
    """Status, error, hash, wall time and one float64 column ``m_<metric>`` per metric."""
    cols: dict[str, Any] = {
        "index": pa.array([r.index for r in results], pa.int64()),
        "label": pa.array([r.label for r in results], pa.string()),
        "status": pa.array([r.status for r in results], pa.string()),
        "error_type": pa.array([r.error_type for r in results], pa.string()),
        "error": pa.array([r.error for r in results], pa.string()),
        "config_hash": pa.array([r.config_hash for r in results], pa.string()),
        "wall_s": pa.array([r.wall_s for r in results], pa.float64()),
    }
    for name in metric_names:
        cols[f"m_{name}"] = pa.array(
            [float(r.metrics.get(name, math.nan)) for r in results], pa.float64()
        )
    return pa.table(cols)


__all__ = [
    "ECONOMICS_PREFIX",
    "MASS_PROXY_PREFIX",
    "METRIC_DEFINITIONS",
    "STUDY_PACKAGES",
    "SYNTHETIC_FUNCTIONS",
    "SYNTHETIC_METRICS",
    "DesignResult",
    "DesignTask",
    "EvaluationSettings",
    "Evaluator",
    "FidelitySpec",
    "ForwardModel",
    "MetricDefinition",
    "SyntheticModel",
    "build_evaluator",
    "chunk_tasks",
    "constellation_metrics",
    "delete_dotted",
    "evaluate_parallel",
    "get_dotted",
    "lcoe_from_run_dir",
    "make_pool",
    "results_table",
    "set_dotted",
    "single_satellite_metrics",
    "study_provenance",
]
