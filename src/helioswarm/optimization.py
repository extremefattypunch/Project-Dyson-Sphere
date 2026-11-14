"""Design optimisation: problem definition, NSGA-II, convergence, re-evaluation (M8; PDF §12).

Workflow (PDF §12.3, §14.3.3 coarse-to-fine, §17 "optimization of a flawed model")
----------------------------------------------------------------------------------
1. **Problem** from ``configs/optimization*.yaml`` (:class:`OptimizationFile`): decision
   variables (§12.1) with bounds and types, objectives (§12.2; reported as a Pareto front,
   never collapsed into one weighted score), inequality constraints, the evaluator (the forward
   model at a stated fidelity level, :mod:`helioswarm.evaluation`, or a synthetic test
   function), the NSGA-II settings, seeds, convergence criteria and the re-evaluation.
2. **Sanity checks** (step 1): one real variable varied with the others at their nominal
   values, a dense sweep and :func:`scipy.optimize.minimize_scalar` (bounded) on the same
   scalar objective; the two optima must agree (:func:`run_sanity_check`).
3. **Synthetic known optima** (step 2): the same machinery on analytic problems with known
   solutions (``tests/unit/test_optimization.py``).
4. **NSGA-II** (step 3; pymoo 0.6.2 ``NSGA2`` driven by ask/tell) with constraints as
   inequality constraints ``g <= 0`` (pymoo's constraint domination: feasible before
   infeasible; infeasible ordered by total violation).
5. **Several seeds + convergence** (step 4): hypervolume of the feasible non-dominated archive
   per generation; every evaluated design, the final population and the front are saved
   (Parquet) with a manifest (:func:`run_seed`, :func:`collate`).
6. **High-fidelity re-evaluation** (step 5) of selected Pareto designs, with the screening vs
   high differences and the feasibility at high fidelity reported (:func:`reevaluate`).

Encoding: the genome is the unit hypercube ``u in [0, 1]^n`` (pymoo real-coded SBX + PM);
:meth:`VariableSpec.decode` maps it to the variable: real ``lo + u (hi - lo)`` (or the same in
``log`` space), integer ``min(floor(lo + u (hi - lo + 1)), hi)`` (equal-width bins), categorical
``choices[min(floor(u n), n - 1)]``. A categorical choice is a *bundle* of dotted overrides
(e.g. optical vs microwave technology sets mode, wavelength, receiver technology and beam
model together), so every decoded design is a complete, validated configuration. Designs with
the same decoded values are evaluated once per seed (cache by design key).

Objectives are stored in minimisation form ``F = metric`` (min) or ``F = -metric`` (max).
Constraint ``k`` of metric ``v``: ``g = (v - limit)/scale`` (``<=``) or ``(limit - v)/scale``
(``>=``); one extra constraint ``g_evaluation`` is ``-1`` for a successful evaluation with
finite objective and constraint metrics and ``+1`` otherwise (a failed or undefined design is
infeasible, never reported as a Pareto member). Failed designs get ``F = FAILED_OBJECTIVE``.
A design is *feasible* iff every ``g <= 0``.

Hypervolume: of the feasible non-dominated archive in minimisation space against the
configured reference point (natural units, converted to minimisation form), computed with
``pymoo.indicators.hv.HV`` (points not strictly better than the reference contribute nothing).
Convergence per seed: relative hypervolume change over the last ``window`` generations
``<= hv_relative_tolerance``; across seeds: ``(max - min)/max`` of the final hypervolumes
``<= seed_hv_relative_tolerance``. Both are reported, not enforced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from helioswarm.config import ConfigError, ConfigIssue, MassProxyConfig, read_yaml
from helioswarm.evaluation import (
    METRIC_DEFINITIONS,
    DesignResult,
    DesignTask,
    EvaluationSettings,
    Evaluator,
    FidelitySpec,
    ForwardModel,
    build_evaluator,
    evaluate_parallel,
    make_pool,
    results_table,
    study_provenance,
)

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

OPTIMIZATION_SCHEMA_VERSION: Final[int] = 1
FAILED_OBJECTIVE: Final[float] = 1.0e30
"""Minimisation-form objective of a failed or undefined design (it is infeasible anyway)."""

EVALUATIONS_FILE: Final[str] = "evaluations.parquet"
POPULATION_FILE: Final[str] = "population.parquet"
PARETO_FILE: Final[str] = "pareto.parquet"
CONVERGENCE_FILE: Final[str] = "convergence.parquet"
RUN_FILE: Final[str] = "run.json"
DEFINITION_FILE: Final[str] = "optimization_definition.yaml"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


# ---------------------------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------------------------


class VariableSpec(_Model):
    """One decision variable (PDF §12.1); see the module docstring for the decoding.

    Attributes:
        name: Variable name (column ``v_<name>`` of the outputs).
        kind: ``real``, ``integer`` or ``categorical``.
        key: Dotted configuration key (real/integer; synthetic evaluators: argument name).
        lower, upper: Bounds (real/integer, inclusive).
        log: Decode in log space (real, ``lower > 0``).
        choices: Categorical bundles: choice name -> dotted overrides.
        nominal: Value used by the sanity checks for the variables not varied (default: the
            base configuration's value; for a categorical the first choice).
        unit: Unit of the value.
        description: What the variable stands for.
    """

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=64)
    kind: Literal["real", "integer", "categorical"]
    key: str | None = None
    lower: float | None = None
    upper: float | None = None
    log: bool = False
    choices: dict[str, dict[str, Any]] | None = None
    nominal: Any = None
    unit: str = "-"
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> VariableSpec:
        if self.kind == "categorical":
            if not self.choices or len(self.choices) < 2:
                raise ValueError(f"categorical variable {self.name!r} needs >= 2 choices")
            if self.key is not None or self.lower is not None or self.upper is not None:
                raise ValueError(
                    f"categorical variable {self.name!r} takes choices only (no key/bounds)"
                )
            if self.nominal is not None and self.nominal not in self.choices:
                raise ValueError(f"nominal {self.nominal!r} is not a choice of {self.name!r}")
            return self
        if self.key is None or self.lower is None or self.upper is None:
            raise ValueError(f"variable {self.name!r} needs key, lower and upper")
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError(f"variable {self.name!r}: bounds must be finite")
        if not self.lower < self.upper:
            raise ValueError(f"variable {self.name!r}: need lower < upper")
        if self.log and self.lower <= 0.0:
            raise ValueError(f"log variable {self.name!r} needs lower > 0")
        if self.kind == "integer":
            if self.log:
                raise ValueError(f"integer variable {self.name!r} cannot be log-scaled")
            if self.lower != int(self.lower) or self.upper != int(self.upper):
                raise ValueError(f"integer variable {self.name!r} needs integral bounds")
        if self.choices is not None:
            raise ValueError(f"variable {self.name!r}: choices only for categorical variables")
        return self

    def decode(self, u: float) -> Any:
        """Variable value of the genome coordinate ``u in [0, 1]`` (clipped)."""
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "categorical":
            assert self.choices is not None
            names = list(self.choices)
            return names[min(math.floor(u * len(names)), len(names) - 1)]
        assert self.lower is not None
        assert self.upper is not None
        if self.kind == "integer":
            lo, hi = int(self.lower), int(self.upper)
            return min(math.floor(lo + u * (hi - lo + 1)), hi)
        if self.log:
            a, b = math.log(self.lower), math.log(self.upper)
            return float(min(max(math.exp(a + u * (b - a)), self.lower), self.upper))
        return float(self.lower + u * (self.upper - self.lower))

    def encode(self, value: Any) -> float:
        """Genome coordinate at the centre of ``value``'s decoding cell (inverse of decode)."""
        if self.kind == "categorical":
            assert self.choices is not None
            names = list(self.choices)
            return (names.index(value) + 0.5) / len(names)
        assert self.lower is not None
        assert self.upper is not None
        if self.kind == "integer":
            lo, hi = int(self.lower), int(self.upper)
            return (int(value) - lo + 0.5) / (hi - lo + 1)
        if self.log:
            a, b = math.log(self.lower), math.log(self.upper)
            return (math.log(float(value)) - a) / (b - a)
        return (float(value) - self.lower) / (self.upper - self.lower)

    def overrides(self, value: Any) -> list[tuple[str, Any]]:
        """Dotted overrides of ``value``."""
        if self.kind == "categorical":
            assert self.choices is not None
            return list(self.choices[value].items())
        assert self.key is not None
        return [(self.key, value)]

    def dotted_keys(self) -> set[str]:
        """Every dotted key this variable may set."""
        if self.kind == "categorical":
            assert self.choices is not None
            return {k for bundle in self.choices.values() for k in bundle}
        assert self.key is not None
        return {self.key}


Sense = Literal["min", "max"]


class ObjectiveSpec(_Model):
    """One objective (PDF §12.2).

    Attributes:
        metric: Evaluator metric.
        sense: ``min`` or ``max``.
        reference: Hypervolume reference value in natural units (a value worse than every
            design of interest: below for ``max``, above for ``min``).
        note: Caveat written next to the objective (e.g. "depends on assumed economics").
    """

    metric: str
    sense: Sense
    reference: float
    note: str = ""

    def to_min(self, value: float) -> float:
        """Minimisation form."""
        return value if self.sense == "min" else -value


class ConstraintSpec(_Model):
    """One inequality constraint ``metric op limit`` (PDF §12.2 constraints)."""

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=64)
    metric: str
    op: Literal["<=", ">="]
    limit: float
    scale: float | None = Field(default=None, gt=0.0)
    description: str = ""

    def g(self, value: float) -> float:
        """Normalised constraint value (``<= 0`` satisfied)."""
        s = self.scale if self.scale is not None else max(abs(self.limit), 1e-300)
        return (value - self.limit) / s if self.op == "<=" else (self.limit - value) / s


class AlgorithmSpec(_Model):
    """NSGA-II settings (pymoo ``NSGA2``; SBX crossover, polynomial mutation)."""

    pop_size: int = Field(ge=4)
    n_generations: int = Field(ge=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    crossover_prob: float = Field(default=0.9, ge=0.0, le=1.0)
    crossover_eta: float = Field(default=15.0, gt=0.0)
    mutation_eta: float = Field(default=20.0, gt=0.0)

    @field_validator("seeds")
    @classmethod
    def _unique(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(v)) != len(v):
            raise ValueError("seeds must be unique")
        return v


class ConvergenceSpec(_Model):
    """Convergence criteria (reported; module docstring)."""

    window: int = Field(default=5, ge=1)
    hv_relative_tolerance: float = Field(default=0.01, gt=0.0)
    seed_hv_relative_tolerance: float = Field(default=0.05, gt=0.0)


class ReevaluationSpec(_Model):
    """High-fidelity re-evaluation of selected Pareto designs (PDF §12.3 step 5)."""

    fidelity: str = "high"
    n_candidates: int = Field(default=5, ge=1)


class SanityCheckSpec(_Model):
    """Single-objective sanity check: sweep + scipy bounded scalar optimisation (§12.3 step 1).

    Objective (maximised): ``J = s * metric - penalty_weight * penalty_metric`` with ``s = +1``
    for ``sense = max`` and ``-1`` for ``min``. A penalised score is a sanity-check device
    only (never a reported design objective).
    """

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=64)
    variable: str
    metric: str
    sense: Sense = "max"
    penalty_metric: str | None = None
    penalty_weight: float = 0.0
    sweep_points: int = Field(default=21, ge=3)
    xatol: float = Field(default=1e-4, gt=0.0, description="tolerance in the genome coordinate")


class EvaluationSection(_Model):
    """Metric-layer settings (:class:`helioswarm.evaluation.EvaluationSettings`); the
    economics path is relative to the study file."""

    power_threshold_W: float = Field(default=0.0, ge=0.0)
    mass_proxy: MassProxyConfig | None = None
    economics_config: str | None = None


class OptimizationFile(_Model):
    """Schema of ``configs/optimization*.yaml`` (module docstring)."""

    optimization_schema_version: int
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=96)
    description: str = ""
    evaluator: str = "forward_model"
    synthetic_parameters: dict[str, float] = Field(default_factory=dict)
    base_config: str | None = None
    search_fidelity: str = "screening"
    fidelity: dict[str, FidelitySpec] = Field(default_factory=dict)
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    variables: tuple[VariableSpec, ...] = Field(min_length=1)
    objectives: tuple[ObjectiveSpec, ...] = Field(min_length=1)
    constraints: tuple[ConstraintSpec, ...] = ()
    algorithm: AlgorithmSpec
    convergence: ConvergenceSpec = Field(default_factory=ConvergenceSpec)
    reevaluation: ReevaluationSpec | None = None
    sanity_checks: tuple[SanityCheckSpec, ...] = ()
    workers: int = Field(default=1, ge=1)

    @field_validator("optimization_schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != OPTIMIZATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported optimization_schema_version {v} (expected 1)")
        return v

    @model_validator(mode="after")
    def _check(self) -> OptimizationFile:
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("variable names must be unique")
        metrics = [o.metric for o in self.objectives]
        if len(set(metrics)) != len(metrics):
            raise ValueError("objective metrics must be unique")
        cnames = [c.name for c in self.constraints]
        if len(set(cnames)) != len(cnames):
            raise ValueError("constraint names must be unique")
        if self.evaluator == "forward_model":
            if self.base_config is None:
                raise ValueError("the forward_model evaluator needs base_config")
            if self.search_fidelity not in self.fidelity:
                raise ValueError(f"search_fidelity {self.search_fidelity!r} is not defined")
            if self.reevaluation is not None and self.reevaluation.fidelity not in self.fidelity:
                raise ValueError(
                    f"reevaluation fidelity {self.reevaluation.fidelity!r} is not defined"
                )
        for chk in self.sanity_checks:
            if chk.variable not in names:
                raise ValueError(f"sanity check {chk.name!r}: unknown variable {chk.variable!r}")
            var = next(v for v in self.variables if v.name == chk.variable)
            if var.kind != "real":
                raise ValueError(f"sanity check {chk.name!r}: variable must be real")
        return self


@dataclass(frozen=True)
class OptimizationProblem:
    """A loaded, cross-checked optimisation study.

    Attributes:
        path: Study file.
        definition: Parsed study file.
        evaluator: Evaluator at the search fidelity.
        definition_sha256: SHA-256 of the canonical JSON of the definition and the resolved
            base configuration mapping.
    """

    path: Path
    definition: OptimizationFile
    evaluator: Evaluator
    definition_sha256: str

    @property
    def variables(self) -> tuple[VariableSpec, ...]:
        """Decision variables."""
        return self.definition.variables

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        """Objectives."""
        return self.definition.objectives

    @property
    def constraints(self) -> tuple[ConstraintSpec, ...]:
        """Constraints."""
        return self.definition.constraints

    def evaluator_at(self, fidelity: str, *, run_root: str | None = None) -> Evaluator:
        """Evaluator at another fidelity level (forward model), or the synthetic evaluator."""
        d = self.definition
        return build_evaluator(
            d.evaluator,
            study_path=self.path,
            base_config=d.base_config,
            fidelity_name=fidelity,
            fidelity=d.fidelity.get(fidelity),
            settings=_settings(d),
            arguments=_arguments(d),
            synthetic_parameters=d.synthetic_parameters,
            name_prefix=d.name,
            run_root=run_root,
        )


def _settings(d: OptimizationFile) -> EvaluationSettings:
    e = d.evaluation
    return EvaluationSettings(
        power_threshold_W=e.power_threshold_W,
        mass_proxy=e.mass_proxy,
        economics_config=e.economics_config,
    )


def _arguments(d: OptimizationFile) -> tuple[str, ...]:
    return tuple(sorted({k for v in d.variables for k in v.dotted_keys()}))


def _issue(path: Path, where: str, msg: str) -> ConfigError:
    return ConfigError(str(path), [ConfigIssue(where, msg)])


def load_optimization(path: str | Path) -> OptimizationProblem:
    """Read and cross-check an optimisation study (keys, metrics, fidelity conflicts).

    Raises:
        ConfigError: invalid file, unknown key or metric, a design key also set by a fidelity
            level, or a metric that needs an input that is not configured.
    """
    p = Path(path).resolve()
    raw = read_yaml(p)
    try:
        d = OptimizationFile.model_validate(raw)
    except Exception as exc:
        raise _issue(p, "<optimization>", str(exc)) from exc
    try:
        ev = build_evaluator(
            d.evaluator,
            study_path=p,
            base_config=d.base_config,
            fidelity_name=d.search_fidelity,
            fidelity=d.fidelity.get(d.search_fidelity),
            settings=_settings(d),
            arguments=_arguments(d),
            synthetic_parameters=d.synthetic_parameters,
            name_prefix=d.name,
        )
    except (ValueError, ConfigError) as exc:
        raise _issue(p, "evaluator", str(exc)) from exc
    known = ev.known_keys()
    for v in d.variables:
        for key in v.dotted_keys():
            if isinstance(ev, ForwardModel):
                # A bundle may add keys of an optional section; validated when decoded.
                top = key.split(".")[0].split("[")[0]
                if key not in known and not any(k.startswith(top + ".") for k in known):
                    raise _issue(p, f"variables.{v.name}", f"unknown configuration key {key!r}")
            elif key not in known:
                raise _issue(p, f"variables.{v.name}", f"unknown argument {key!r}")
        for name, fid in d.fidelity.items():
            clash = v.dotted_keys() & (set(fid.overrides) | set(fid.unset))
            if clash:
                raise _issue(
                    p,
                    f"variables.{v.name}",
                    f"keys {sorted(clash)} are also set by fidelity level {name!r}",
                )
    available = ev.metric_names()
    used = [o.metric for o in d.objectives] + [c.metric for c in d.constraints]
    used += [c.metric for c in d.sanity_checks]
    used += [c.penalty_metric for c in d.sanity_checks if c.penalty_metric is not None]
    for m in used:
        if m not in available:
            raise _issue(p, "metrics", f"unknown metric {m!r}; available: {sorted(available)}")
    if isinstance(ev, ForwardModel):
        if "lcoe_usd_per_kWh" in used and d.evaluation.economics_config is None:
            raise _issue(p, "evaluation", "lcoe_usd_per_kWh needs evaluation.economics_config")
        base = ev.fidelity_mapping()
        if "constellation" in base:
            for m in used:
                if not METRIC_DEFINITIONS[m].constellation:
                    raise _issue(p, "metrics", f"metric {m!r} is not defined for constellations")
        elif "mass_proxy_kg" in used and d.evaluation.mass_proxy is None:
            raise _issue(p, "evaluation", "mass_proxy_kg needs evaluation.mass_proxy")
        for name in d.fidelity:
            fev = build_evaluator(
                d.evaluator,
                study_path=p,
                base_config=d.base_config,
                fidelity_name=name,
                fidelity=d.fidelity[name],
                settings=_settings(d),
            )
            assert isinstance(fev, ForwardModel)
            _check_designs_validate(p, d, fev)
        base_for_hash: Any = base
    else:
        base_for_hash = None
    canon = json.dumps(
        {"definition": d.model_dump(mode="json"), "base": base_for_hash},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return OptimizationProblem(
        path=p,
        definition=d,
        evaluator=ev,
        definition_sha256=hashlib.sha256(canon.encode()).hexdigest(),
    )


def _check_designs_validate(p: Path, d: OptimizationFile, ev: ForwardModel) -> None:
    """The nominal design, every categorical choice and both bound corners must give valid
    configurations at this fidelity (catches unknown keys and inconsistent bundles early)."""
    nominal = _nominal_from_mapping(d.variables, ev.fidelity_mapping())
    trials: list[dict[str, Any]] = [dict(nominal)]
    for v in d.variables:
        if v.kind == "categorical":
            assert v.choices is not None
            trials += [{**nominal, v.name: c} for c in v.choices]
    trials.append({v.name: v.decode(0.0) for v in d.variables})
    trials.append({v.name: v.decode(1.0) for v in d.variables})
    for k, values in enumerate(trials):
        overrides = tuple(o for v in d.variables for o in v.overrides(values[v.name]))
        try:
            ev.config_for(DesignTask(index=k, label=f"check{k}", overrides=overrides))
        except ConfigError as exc:
            raise _issue(
                p, "variables", f"design {values} invalid at fidelity {ev.fidelity_name!r}: {exc}"
            ) from exc


def _nominal_from_mapping(
    variables: Sequence[VariableSpec], base: Mapping[str, Any] | None
) -> dict[str, Any]:
    from helioswarm.evaluation import get_dotted

    out: dict[str, Any] = {}
    for v in variables:
        if v.nominal is not None:
            out[v.name] = v.nominal
        elif v.kind == "categorical":
            assert v.choices is not None
            out[v.name] = next(iter(v.choices))
        else:
            assert v.lower is not None
            assert v.upper is not None
            assert v.key is not None
            value: Any = None
            if base is not None:
                try:
                    value = get_dotted(base, v.key)
                except KeyError:
                    value = None
            out[v.name] = value if value is not None else 0.5 * (v.lower + v.upper)
    return out


# ---------------------------------------------------------------------------------------------
# Designs, objectives, constraints
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """A decoded design.

    Attributes:
        values: Variable name -> value.
        overrides: Dotted overrides in variable order.
        key: Canonical JSON of ``values`` (cache key).
    """

    values: Mapping[str, Any]
    overrides: tuple[tuple[str, Any], ...]
    key: str


def decode(variables: Sequence[VariableSpec], u: Sequence[float]) -> Design:
    """Decode a genome (module docstring)."""
    if len(u) != len(variables):
        raise ValueError("genome length differs from the number of variables")
    values: dict[str, Any] = {}
    overrides: list[tuple[str, Any]] = []
    for var, x in zip(variables, u, strict=True):
        value = var.decode(float(x))
        values[var.name] = value
        overrides += var.overrides(value)
    return Design(
        values=values,
        overrides=tuple(overrides),
        key=json.dumps(values, sort_keys=True, default=str),
    )


def design_task(design: Design, index: int, label: str) -> DesignTask:
    """The evaluation task of a design."""
    return DesignTask(index=index, label=label, overrides=design.overrides)


@dataclass(frozen=True)
class Scored:
    """Objective/constraint values of one evaluated design.

    Attributes:
        F: Minimisation-form objectives.
        G: Constraint values incl. ``g_evaluation`` (last).
        feasible: Every ``g <= 0``.
        cv: Total violation ``sum max(g, 0)``.
        undefined: Names of non-finite objective/constraint metrics of an ok evaluation.
    """

    F: Float64Array
    G: Float64Array
    feasible: bool
    cv: float
    undefined: tuple[str, ...]


def score(
    result: DesignResult,
    objectives: Sequence[ObjectiveSpec],
    constraints: Sequence[ConstraintSpec],
) -> Scored:
    """Objective and constraint vectors of an evaluated design (module docstring)."""
    n_obj, n_con = len(objectives), len(constraints)
    if result.status != "ok":
        F = np.full(n_obj, FAILED_OBJECTIVE)
        G = np.zeros(n_con + 1)
        G[-1] = 1.0
        return Scored(F, G, False, 1.0, ())
    undefined = tuple(
        name
        for name in {o.metric for o in objectives} | {c.metric for c in constraints}
        if not math.isfinite(float(result.metrics.get(name, math.nan)))
    )
    if undefined:
        F = np.full(n_obj, FAILED_OBJECTIVE)
        G = np.zeros(n_con + 1)
        G[-1] = 1.0
        return Scored(F, G, False, 1.0, tuple(sorted(undefined)))
    F = np.array([o.to_min(float(result.metrics[o.metric])) for o in objectives])
    G = np.array([c.g(float(result.metrics[c.metric])) for c in constraints] + [-1.0])
    cv = float(np.sum(np.maximum(G, 0.0)))
    return Scored(F, G, bool(np.all(G <= 0.0)), cv, ())


def pareto_mask(F: Float64Array) -> BoolArray:
    """Non-dominated rows of ``F`` (minimisation; duplicates are all kept). ``O(n^2 m)``."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        le = np.all(F[i] >= F, axis=1)
        lt = np.any(F[i] > F, axis=1)
        if np.any(le & lt):
            keep[i] = False
    return keep


def hypervolume(F: Float64Array, reference: Float64Array) -> float:
    """Hypervolume of the points of ``F`` (minimisation) dominated region bounded by
    ``reference`` (pymoo ``HV``; points not strictly better than ``reference`` in every
    objective are ignored; 0 for none)."""
    from pymoo.indicators.hv import HV

    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    ref = np.asarray(reference, dtype=np.float64)
    if F.size == 0:
        return 0.0
    inside = np.all(ref > F, axis=1)
    if not np.any(inside):
        return 0.0
    return float(HV(ref_point=ref)(F[inside]))


def reference_point(objectives: Sequence[ObjectiveSpec]) -> Float64Array:
    """Hypervolume reference point in minimisation form."""
    return np.array([o.to_min(o.reference) for o in objectives], dtype=np.float64)


# ---------------------------------------------------------------------------------------------
# NSGA-II run of one seed
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationRecord:
    """One evaluated individual of one seed.

    Attributes:
        generation: Generation (0 = initial population).
        member: Position in that generation's offspring.
        eval_id: Running index over the seed's evaluations.
        u: Genome.
        design: Decoded design.
        result: Evaluation result.
        scored: Objective/constraint values.
        cached: Same decoded design evaluated earlier in this seed (result reused).
    """

    generation: int
    member: int
    eval_id: int
    u: Float64Array
    design: Design
    result: DesignResult
    scored: Scored
    cached: bool


@dataclass(frozen=True)
class SeedRun:
    """Everything one seed produced (written by :func:`write_seed`).

    Attributes:
        seed: Seed.
        records: Every evaluated individual in evaluation order.
        final_population: ``eval_id`` of each member of the final population.
        hv_by_generation: Archive hypervolume after each generation.
        feasible_by_generation: Feasible evaluations so far, per generation.
        wall_s: Total wall time [s].
        workers: Worker processes used.
    """

    seed: int
    records: tuple[EvaluationRecord, ...]
    final_population: tuple[int, ...]
    hv_by_generation: tuple[float, ...]
    feasible_by_generation: tuple[int, ...]
    wall_s: float
    workers: int


def archive_front(records: Sequence[EvaluationRecord]) -> list[EvaluationRecord]:
    """Feasible non-dominated designs of ``records`` (one record per distinct design)."""
    seen: dict[str, EvaluationRecord] = {}
    for r in records:
        if r.scored.feasible and r.design.key not in seen:
            seen[r.design.key] = r
    pool = list(seen.values())
    if not pool:
        return []
    F = np.array([r.scored.F for r in pool])
    mask = pareto_mask(F)
    return [r for r, m in zip(pool, mask, strict=True) if m]


def run_seed(
    problem: OptimizationProblem,
    seed: int,
    *,
    workers: int | None = None,
    evaluator: Evaluator | None = None,
    log: Callable[[str], None] | None = None,
) -> SeedRun:
    """NSGA-II for one seed (module docstring); pymoo's Generator is seeded with ``seed``."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.evaluator import Evaluator as PymooEvaluator
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.problems.static import StaticProblem

    d = problem.definition
    alg_spec = d.algorithm
    ev = evaluator if evaluator is not None else problem.evaluator
    n_workers = workers if workers is not None else d.workers
    n_var, n_obj, n_con = len(d.variables), len(d.objectives), len(d.constraints)
    pproblem = Problem(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_con + 1, xl=0.0, xu=1.0)
    algorithm = NSGA2(
        pop_size=alg_spec.pop_size,
        crossover=SBX(prob=alg_spec.crossover_prob, eta=alg_spec.crossover_eta),
        mutation=PM(eta=alg_spec.mutation_eta),
        eliminate_duplicates=True,
    )
    algorithm.setup(pproblem, termination=("n_gen", alg_spec.n_generations), seed=seed)
    ref = reference_point(d.objectives)
    cache: dict[str, DesignResult] = {}
    records: list[EvaluationRecord] = []
    hv: list[float] = []
    n_feasible: list[int] = []
    by_genome: dict[bytes, int] = {}
    t_start = time.perf_counter()
    pool = make_pool(n_workers)
    generation = 0
    try:
        while algorithm.has_next():
            pop = algorithm.ask()
            X = np.asarray(pop.get("X"), dtype=np.float64)
            designs = [decode(d.variables, x) for x in X]
            todo: dict[str, DesignTask] = {}
            for i, des in enumerate(designs):
                if des.key not in cache and des.key not in todo:
                    todo[des.key] = design_task(
                        des, len(todo), f"s{seed}_g{generation:03d}_m{i:03d}"
                    )
            new = evaluate_parallel(ev, list(todo.values()), workers=n_workers, executor=pool)
            fresh = dict(zip(todo, new, strict=True))
            F = np.empty((len(designs), n_obj))
            G = np.empty((len(designs), n_con + 1))
            for i, (x, des) in enumerate(zip(X, designs, strict=True)):
                cached = des.key in cache
                res = cache[des.key] if cached else fresh[des.key]
                sc = score(res, d.objectives, d.constraints)
                F[i], G[i] = sc.F, sc.G
                rec = EvaluationRecord(
                    generation=generation,
                    member=i,
                    eval_id=len(records),
                    u=x.copy(),
                    design=des,
                    result=res,
                    scored=sc,
                    cached=cached,
                )
                by_genome[x.tobytes()] = rec.eval_id
                records.append(rec)
            cache.update(fresh)
            PymooEvaluator().eval(StaticProblem(pproblem, F=F, G=G), pop)
            algorithm.tell(infills=pop)
            front = archive_front(records)
            hv.append(hypervolume(np.array([r.scored.F for r in front]), ref) if front else 0.0)
            n_feasible.append(sum(1 for r in records if r.scored.feasible))
            if log is not None:
                n_fail = sum(1 for r in fresh.values() if r.status != "ok")
                log(
                    f"seed {seed} generation {generation}: {len(designs)} individuals, "
                    f"{len(fresh)} new evaluations ({n_fail} failed), feasible so far "
                    f"{n_feasible[-1]}, front {len(front)}, HV {hv[-1]:.6g}"
                )
            generation += 1
    finally:
        if pool is not None:
            pool.shutdown()
    final_X = np.asarray(algorithm.pop.get("X"), dtype=np.float64)
    final = tuple(by_genome[x.tobytes()] for x in final_X)
    return SeedRun(
        seed=seed,
        records=tuple(records),
        final_population=final,
        hv_by_generation=tuple(hv),
        feasible_by_generation=tuple(n_feasible),
        wall_s=time.perf_counter() - t_start,
        workers=n_workers,
    )


def converged(hv_by_generation: Sequence[float], window: int, tolerance: float) -> bool:
    """Relative hypervolume change over the last ``window`` generations ``<= tolerance``
    (``False`` with fewer than ``window + 1`` generations or a zero final hypervolume)."""
    if len(hv_by_generation) < window + 1:
        return False
    last = hv_by_generation[-1]
    before = hv_by_generation[-1 - window]
    if last <= 0.0:
        return False
    return abs(last - before) / last <= tolerance


# ---------------------------------------------------------------------------------------------
# Tables and files
# ---------------------------------------------------------------------------------------------


def _metric_columns(problem: OptimizationProblem) -> list[str]:
    return sorted(problem.evaluator.metric_names())


def records_table(
    problem: OptimizationProblem, records: Sequence[EvaluationRecord], seed: int
) -> pa.Table:
    """One row per record: genome ``u_*``, values ``v_*``, metrics ``m_*``, ``F_*``, ``G_*``,
    feasibility, status and error (module docstring)."""
    d = problem.definition
    base = results_table([r.result for r in records], _metric_columns(problem))
    cols: dict[str, Any] = {
        "seed": pa.array([seed] * len(records), pa.int64()),
        "generation": pa.array([r.generation for r in records], pa.int64()),
        "member": pa.array([r.member for r in records], pa.int64()),
        "eval_id": pa.array([r.eval_id for r in records], pa.int64()),
        "cached": pa.array([r.cached for r in records], pa.bool_()),
        "design_key": pa.array([r.design.key for r in records], pa.string()),
    }
    for j, var in enumerate(d.variables):
        cols[f"u_{var.name}"] = pa.array([float(r.u[j]) for r in records], pa.float64())
        vals = [r.design.values[var.name] for r in records]
        cols[f"v_{var.name}"] = (
            pa.array([str(v) for v in vals], pa.string())
            if var.kind == "categorical"
            else pa.array([float(v) for v in vals], pa.float64())
        )
    for j, obj in enumerate(d.objectives):
        cols[f"F_{obj.metric}"] = pa.array([float(r.scored.F[j]) for r in records], pa.float64())
    for j, con in enumerate(d.constraints):
        cols[f"G_{con.name}"] = pa.array([float(r.scored.G[j]) for r in records], pa.float64())
    cols["G_evaluation"] = pa.array([float(r.scored.G[-1]) for r in records], pa.float64())
    cols["feasible"] = pa.array([r.scored.feasible for r in records], pa.bool_())
    cols["constraint_violation"] = pa.array([r.scored.cv for r in records], pa.float64())
    cols["undefined_metrics"] = pa.array(
        [",".join(r.scored.undefined) for r in records], pa.string()
    )
    for name in base.column_names:
        if name in ("index", "label"):
            continue
        cols[name] = base.column(name)
    cols["label"] = base.column("label")
    return pa.table(cols)


def _write_parquet(table: pa.Table, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path


def _write_json(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    return path


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    return x


def seed_dir(out: Path, seed: int) -> Path:
    """Output directory of one seed."""
    return out / f"seed_{seed}"


def prepare_out(problem: OptimizationProblem, out: Path) -> Path:
    """Create ``out`` with the study definition (shared by array tasks; a different study in
    the same directory is refused)."""
    out = out.expanduser().resolve()
    marker = out / DEFINITION_FILE
    header = f"# definition_sha256: {problem.definition_sha256}\n"
    if marker.is_file():
        first = marker.read_text(encoding="utf-8").splitlines()[0] + "\n"
        if first != header:
            raise FileExistsError(f"{out} holds a different optimisation ({first.strip()})")
        return out
    out.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        {
            "study_file": str(problem.path),
            "definition": problem.definition.model_dump(mode="json"),
        },
        sort_keys=False,
    )
    tmp = out / f".{DEFINITION_FILE}.{os.getpid()}.{time.time_ns()}"
    tmp.write_text(header + body, encoding="utf-8")
    try:
        os.link(tmp, marker)  # atomic create-if-absent (concurrent array tasks)
    except FileExistsError:
        first = marker.read_text(encoding="utf-8").splitlines()[0] + "\n"
        if first != header:
            raise FileExistsError(
                f"{out} holds a different optimisation ({first.strip()})"
            ) from None
    finally:
        tmp.unlink(missing_ok=True)
    return out


def write_seed(problem: OptimizationProblem, run: SeedRun, out: Path, started_utc: str) -> Path:
    """Write one seed's outputs (``evaluations``, ``population``, ``pareto``, ``convergence``
    Parquet files and ``run.json`` with the manifest)."""
    from helioswarm.manifest import utc_now_iso

    d = seed_dir(out, run.seed)
    d.mkdir(parents=True, exist_ok=True)
    table = records_table(problem, run.records, run.seed)
    _write_parquet(table, d / EVALUATIONS_FILE)
    _write_parquet(
        table.take(pa.array(list(run.final_population), pa.int64())), d / POPULATION_FILE
    )
    front = archive_front(run.records)
    _write_parquet(
        records_table(problem, front, run.seed) if front else table.slice(0, 0), d / PARETO_FILE
    )
    conv = problem.definition.convergence
    _write_parquet(
        pa.table(
            {
                "generation": pa.array(range(len(run.hv_by_generation)), pa.int64()),
                "hypervolume": pa.array(run.hv_by_generation, pa.float64()),
                "feasible_evaluations": pa.array(run.feasible_by_generation, pa.int64()),
            }
        ),
        d / CONVERGENCE_FILE,
    )
    n_eval = sum(1 for r in run.records if not r.cached)
    doc = {
        "layout": "helioswarm-optimization-seed/1",
        "study": problem.definition.name,
        "study_file": str(problem.path),
        "definition_sha256": problem.definition_sha256,
        "seed": run.seed,
        "search_fidelity": problem.definition.search_fidelity,
        "individuals": len(run.records),
        "evaluations": n_eval,
        "cached": len(run.records) - n_eval,
        "failed": sum(1 for r in run.records if not r.cached and r.result.status != "ok"),
        "undefined": sum(1 for r in run.records if not r.cached and r.scored.undefined),
        "feasible": sum(1 for r in run.records if not r.cached and r.scored.feasible),
        "front_size": len(front),
        "hypervolume_final": run.hv_by_generation[-1] if run.hv_by_generation else 0.0,
        "converged": converged(run.hv_by_generation, conv.window, conv.hv_relative_tolerance),
        "convergence_rule": f"relative HV change over the last {conv.window} generations <= "
        f"{conv.hv_relative_tolerance}",
        "reference_point_min_form": reference_point(problem.objectives).tolist(),
        "wall_s": run.wall_s,
        "evaluation_wall_s_sum": float(sum(r.result.wall_s for r in run.records if not r.cached)),
        "workers": run.workers,
        "started_utc": started_utc,
        "finished_utc": utc_now_iso(),
        "provenance": study_provenance(),
    }
    _write_json(d / RUN_FILE, _clean(doc))
    return d


# ---------------------------------------------------------------------------------------------
# Collation across seeds
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Collation:
    """Collated multi-seed result (``collated/``).

    Attributes:
        path: ``collated`` directory.
        seeds: Seeds collated.
        missing_seeds: Configured seeds without a finished ``run.json``.
        union_front: Feasible non-dominated union (Arrow table with a ``seed`` column).
        hv_final: Final hypervolume per seed.
        seed_spread: ``(max - min)/max`` of ``hv_final``.
        seeds_agree: ``seed_spread <= seed_hv_relative_tolerance``.
    """

    path: Path
    seeds: tuple[int, ...]
    missing_seeds: tuple[int, ...]
    union_front: pa.Table
    hv_final: Mapping[int, float]
    seed_spread: float
    seeds_agree: bool


def _f_matrix(table: pa.Table, problem: OptimizationProblem) -> Float64Array:
    return np.column_stack(
        [
            np.asarray(table.column(f"F_{o.metric}").to_numpy(), dtype=np.float64)
            for o in problem.objectives
        ]
    ).reshape(table.num_rows, len(problem.objectives))


def collate(problem: OptimizationProblem, out: Path) -> Collation:
    """Union front, per-seed convergence and seed agreement from the saved seed outputs;
    writes ``collated/{pareto_union,convergence}.parquet``, ``summary.json``, ``summary.md``
    and figures. Missing seeds are listed, never silently skipped."""
    out = out.expanduser().resolve()
    seeds, missing, fronts, convs, runs = [], [], [], [], {}
    for s in problem.definition.algorithm.seeds:
        d = seed_dir(out, s)
        if not (d / RUN_FILE).is_file():
            missing.append(s)
            continue
        seeds.append(s)
        runs[s] = json.loads((d / RUN_FILE).read_text(encoding="utf-8"))
        fronts.append(pq.read_table(d / PARETO_FILE))
        c = pq.read_table(d / CONVERGENCE_FILE)
        convs.append(c.append_column("seed", pa.array([s] * c.num_rows, pa.int64())))
    if not seeds:
        raise FileNotFoundError(f"{out}: no finished seed (run.json) to collate")
    allf = pa.concat_tables(fronts, promote_options="default")
    if allf.num_rows:
        F = _f_matrix(allf, problem)
        keys = allf.column("design_key").to_pylist()
        first: dict[str, int] = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        idx = np.array(sorted(first.values()), dtype=np.int64)
        mask = pareto_mask(F[idx])
        union = allf.take(pa.array(idx[mask]))
    else:
        union = allf
    hv_final = {s: float(runs[s]["hypervolume_final"]) for s in seeds}
    hv_max = max(hv_final.values())
    spread = (hv_max - min(hv_final.values())) / hv_max if hv_max > 0 else math.inf
    tol = problem.definition.convergence.seed_hv_relative_tolerance
    cdir = out / "collated"
    _write_parquet(union, cdir / "pareto_union.parquet")
    _write_parquet(pa.concat_tables(convs), cdir / "convergence.parquet")
    hv_union = (
        hypervolume(_f_matrix(union, problem), reference_point(problem.objectives))
        if union.num_rows
        else 0.0
    )
    seed_defs = sorted({str(runs[s]["definition_sha256"]) for s in seeds})
    summary = {
        "layout": "helioswarm-optimization-collation/1",
        "study": problem.definition.name,
        "definition_sha256": problem.definition_sha256,
        "seed_definition_sha256": seed_defs,
        "definition_note": (
            "the seeds were run with the definition(s) listed in seed_definition_sha256, which "
            "differ from the current study file (e.g. an added diagnostic fidelity level)"
            if seed_defs != [problem.definition_sha256]
            else ""
        ),
        "seeds": seeds,
        "missing_seeds": missing,
        "per_seed": {str(s): {k: runs[s][k] for k in _RUN_KEYS} for s in seeds},
        "hypervolume_union_front": hv_union,
        "seed_hv_spread": spread,
        "seeds_agree": bool(spread <= tol),
        "seed_rule": f"(max - min)/max of the final hypervolumes <= {tol}",
        "union_front_size": union.num_rows,
        "objectives": [o.model_dump() for o in problem.objectives],
        "provenance": study_provenance(),
    }
    _write_json(cdir / "summary.json", _clean(summary))
    (cdir / "summary.md").write_text(_summary_md(problem, summary, union), encoding="utf-8")
    _figures(problem, union, pa.concat_tables(convs), cdir)
    return Collation(
        path=cdir,
        seeds=tuple(seeds),
        missing_seeds=tuple(missing),
        union_front=union,
        hv_final=hv_final,
        seed_spread=spread,
        seeds_agree=bool(spread <= tol),
    )


_RUN_KEYS: Final[tuple[str, ...]] = (
    "individuals",
    "evaluations",
    "cached",
    "failed",
    "undefined",
    "feasible",
    "front_size",
    "hypervolume_final",
    "converged",
    "wall_s",
    "workers",
)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "—" if not math.isfinite(v) else f"{v:.4g}"
    return str(v)


def _summary_md(problem: OptimizationProblem, s: Mapping[str, Any], union: pa.Table) -> str:
    d = problem.definition
    lines = [
        f"# Optimisation `{d.name}`",
        "",
        d.description,
        "",
        f"Search fidelity: `{d.search_fidelity}`"
        + (
            f" — {d.fidelity[d.search_fidelity].description}"
            if d.search_fidelity in d.fidelity
            else ""
        ),
        "",
        "Objectives: "
        + "; ".join(
            f"{o.sense} `{o.metric}`" + (f" ({o.note})" if o.note else "") for o in d.objectives
        ),
        "",
        "Constraints: "
        + ("; ".join(f"`{c.metric}` {c.op} {c.limit:g}" for c in d.constraints) or "none"),
        "",
        "## Seeds",
        "",
        "| seed | " + " | ".join(_RUN_KEYS) + " |",
        "|---" * (len(_RUN_KEYS) + 1) + "|",
    ]
    for seed, r in s["per_seed"].items():
        lines.append(f"| {seed} | " + " | ".join(_fmt(r[k]) for k in _RUN_KEYS) + " |")
    lines += [
        "",
        f"Missing seeds: {s['missing_seeds'] or 'none'}. Seed HV spread {_fmt(s['seed_hv_spread'])}"
        f" ({s['seed_rule']}: {'yes' if s['seeds_agree'] else 'NO'}).",
        "",
        f"## Union Pareto front ({union.num_rows} designs, feasible only)",
        "",
    ]
    vcols = [f"v_{v.name}" for v in d.variables]
    mcols = [f"m_{o.metric}" for o in d.objectives] + [
        f"m_{c.metric}" for c in d.constraints if f"m_{c.metric}" not in vcols
    ]
    mcols = list(dict.fromkeys(mcols))
    head = ["seed", *vcols, *mcols]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|---" * len(head) + "|")
    if union.num_rows:
        order = np.argsort(_f_matrix(union, problem)[:, 0], kind="stable")
        for i in order:
            row = [union.column(c)[int(i)].as_py() for c in head]
            lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _figures(problem: OptimizationProblem, union: pa.Table, conv: pa.Table, cdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objs = problem.objectives
    if union.num_rows and len(objs) >= 2:
        pairs = [(i, j) for i in range(len(objs)) for j in range(i + 1, len(objs))]
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.2), squeeze=False)
        seeds = np.asarray(union.column("seed").to_numpy())
        for ax, (i, j) in zip(axes[0], pairs, strict=True):
            x = np.asarray(union.column(f"m_{objs[i].metric}").to_numpy())
            y = np.asarray(union.column(f"m_{objs[j].metric}").to_numpy())
            for s in np.unique(seeds):
                m = seeds == s
                ax.scatter(x[m], y[m], s=12, label=f"seed {s}", alpha=0.8)
            ax.set_xlabel(f"{objs[i].metric} ({objs[i].sense})")
            ax.set_ylabel(f"{objs[j].metric} ({objs[j].sense})")
            if objs[i].metric.endswith(("_W", "_kg", "_m2")) and np.all(x > 0):
                ax.set_xscale("log")
            if objs[j].metric.endswith(("_W", "_kg", "_m2")) and np.all(y > 0):
                ax.set_yscale("log")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"{problem.definition.name}: union Pareto front (feasible designs only)")
        fig.tight_layout()
        fig.savefig(cdir / "pareto.png", dpi=110)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    seeds_c = np.asarray(conv.column("seed").to_numpy())
    for s in np.unique(seeds_c):
        m = seeds_c == s
        ax.plot(
            np.asarray(conv.column("generation").to_numpy())[m],
            np.asarray(conv.column("hypervolume").to_numpy())[m],
            label=f"seed {s}",
        )
    ax.set_xlabel("generation")
    ax.set_ylabel("hypervolume (min-form, ref. point)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cdir / "convergence.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# High-fidelity re-evaluation (PDF §12.3 step 5)
# ---------------------------------------------------------------------------------------------


def select_candidates(F: Float64Array, n: int) -> list[int]:
    """Indices of up to ``n`` front members: the best design of each objective first, then
    designs evenly spaced along the first objective (deterministic)."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    chosen: list[int] = []
    for j in range(F.shape[1]):
        i = int(np.lexsort((np.arange(F.shape[0]), F[:, j]))[0])
        if i not in chosen:
            chosen.append(i)
    order = np.lexsort((np.arange(F.shape[0]), F[:, 0]))
    for q in np.linspace(0, F.shape[0] - 1, max(n, 2)):
        if len(chosen) >= n:
            break
        i = int(order[round(float(q))])
        if i not in chosen:
            chosen.append(i)
    return chosen[:n]


@dataclass(frozen=True)
class Reevaluation:
    """High-fidelity re-evaluation (``reevaluation/``).

    Attributes:
        path: Output directory.
        table: One row per candidate: values, screening and high metrics, relative
            differences, feasibility at both levels.
    """

    path: Path
    table: pa.Table


def reevaluate(
    problem: OptimizationProblem,
    out: Path,
    *,
    workers: int | None = None,
    n_candidates: int | None = None,
    save_runs: bool = True,
    fidelity: str | None = None,
    log: Callable[[str], None] | None = None,
) -> Reevaluation:
    """Re-evaluate selected union-front designs at the re-evaluation fidelity (module
    docstring), or at another defined level (``fidelity``, e.g. a diagnostic level that
    separates model effects from weather; output ``reevaluation_<fidelity>/``). Each candidate
    is written as a complete run directory under ``<dir>/runs/`` (``save_runs``); the
    comparison table goes to ``<dir>/reevaluation.parquet`` + ``.md`` + ``.json``. The
    candidates are those of the configured re-evaluation (same selection), so tables of
    different levels compare the same designs."""
    d = problem.definition
    if d.reevaluation is None:
        raise ValueError("the study has no reevaluation section")
    level = fidelity if fidelity is not None else d.reevaluation.fidelity
    if d.evaluator == "forward_model" and level not in d.fidelity:
        raise ValueError(f"fidelity level {level!r} is not defined in the study")
    out = out.expanduser().resolve()
    union_path = out / "collated" / "pareto_union.parquet"
    if not union_path.is_file():
        raise FileNotFoundError(f"{union_path} missing: collate first")
    union = pq.read_table(union_path)
    if union.num_rows == 0:
        raise ValueError("the union front is empty: nothing to re-evaluate")
    collated = json.loads((out / "collated" / "summary.json").read_text(encoding="utf-8"))
    n = n_candidates if n_candidates is not None else d.reevaluation.n_candidates
    picks = select_candidates(_f_matrix(union, problem), n)
    rdir = out / ("reevaluation" if fidelity is None else f"reevaluation_{level}")
    ev = problem.evaluator_at(level, run_root=str(rdir / "runs") if save_runs else None)
    tasks = []
    for k, i in enumerate(picks):
        u = [float(union.column(f"u_{v.name}")[i].as_py()) for v in d.variables]
        tasks.append(design_task(decode(d.variables, u), k, f"c{k:02d}"))
    if log is not None:
        log(f"re-evaluating {len(tasks)} candidates at fidelity {level!r}")
    results = evaluate_parallel(ev, tasks, workers=workers or d.workers)
    metrics = sorted({o.metric for o in d.objectives} | {c.metric for c in d.constraints})
    cols: dict[str, Any] = {
        "candidate": pa.array([t.label for t in tasks], pa.string()),
        "seed": pa.array([union.column("seed")[i].as_py() for i in picks], pa.int64()),
        "screening_eval_id": pa.array([union.column("eval_id")[i].as_py() for i in picks]),
    }
    for v in d.variables:
        cols[f"v_{v.name}"] = pa.array([union.column(f"v_{v.name}")[i].as_py() for i in picks])
    cols["high_status"] = pa.array([r.status for r in results], pa.string())
    cols["high_error"] = pa.array([r.error for r in results], pa.string())
    for m in metrics:
        s_vals = np.array([float(union.column(f"m_{m}")[i].as_py()) for i in picks])
        h_vals = np.array([float(r.metrics.get(m, math.nan)) for r in results])
        cols[f"screening_{m}"] = pa.array(s_vals, pa.float64())
        cols[f"high_{m}"] = pa.array(h_vals, pa.float64())
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(s_vals != 0.0, (h_vals - s_vals) / np.abs(s_vals), np.nan)
        cols[f"reldiff_{m}"] = pa.array(rel, pa.float64())
    high_scores = [score(r, d.objectives, d.constraints) for r in results]
    cols["feasible_screening"] = pa.array([True] * len(picks), pa.bool_())
    cols["feasible_high"] = pa.array([s.feasible for s in high_scores], pa.bool_())
    cols["high_wall_s"] = pa.array([r.wall_s for r in results], pa.float64())
    table = pa.table(cols)
    _write_parquet(table, rdir / "reevaluation.parquet")
    doc = {
        "layout": "helioswarm-optimization-reevaluation/1",
        "study": d.name,
        "definition_sha256": problem.definition_sha256,
        "front_definition_sha256": collated.get("definition_sha256"),
        "screening_fidelity": d.search_fidelity,
        "high_fidelity": level,
        "screening_description": d.fidelity[d.search_fidelity].description
        if d.search_fidelity in d.fidelity
        else d.evaluator,
        "high_description": d.fidelity[level].description if level in d.fidelity else d.evaluator,
        "candidates": table.to_pylist(),
        "provenance": study_provenance(),
    }
    _write_json(rdir / "reevaluation.json", _clean(doc))
    (rdir / "reevaluation.md").write_text(
        _reeval_md(problem, table, metrics, level), encoding="utf-8"
    )
    return Reevaluation(path=rdir, table=table)


def _reeval_md(
    problem: OptimizationProblem, table: pa.Table, metrics: Sequence[str], level: str
) -> str:
    d = problem.definition
    lines = [
        f"# Re-evaluation of selected front designs: `{d.name}`",
        "",
        f"Screening (`{d.search_fidelity}`) vs `{level}`; relative difference "
        "= (high - screening)/|screening|.",
        "",
    ]
    head = ["candidate", *[f"v_{v.name}" for v in d.variables], "high_status", "feasible_high"]
    for m in metrics:
        head += [f"screening_{m}", f"high_{m}", f"reldiff_{m}"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|---" * len(head) + "|")
    for row in table.to_pylist():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in head) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Sanity checks (PDF §12.3 step 1)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SanityResult:
    """Outcome of one sanity check.

    Attributes:
        name: Check name.
        sweep_u: Genome coordinates of the sweep.
        sweep_values: Variable values of the sweep.
        sweep_objective: ``J`` at the sweep points (NaN where the evaluation failed).
        scipy_u: Optimum genome coordinate found by ``minimize_scalar``.
        scipy_value: Its variable value.
        scipy_objective: ``J`` there.
        scipy_evaluations: Function evaluations used by scipy.
        best_sweep_index: Index of the best sweep point.
        agree: scipy's ``J >=`` the best sweep ``J`` minus ``3 xatol`` times the largest sweep
            slope (Brent stops within ~``xatol`` of its optimum and never evaluates a bound
            exactly) and its location within one sweep spacing of the best sweep point.
    """

    name: str
    sweep_u: Float64Array
    sweep_values: Float64Array
    sweep_objective: Float64Array
    scipy_u: float
    scipy_value: float
    scipy_objective: float
    scipy_evaluations: int
    best_sweep_index: int
    agree: bool


def nominal_values(problem: OptimizationProblem) -> dict[str, Any]:
    """Nominal value of every variable: configured ``nominal``, else the base configuration's
    value (forward model), else the bounds' midpoint (synthetic); categorical: first choice."""
    ev = problem.evaluator
    base = ev.fidelity_mapping() if isinstance(ev, ForwardModel) else None
    return _nominal_from_mapping(problem.variables, base)


def run_sanity_check(
    problem: OptimizationProblem,
    check: SanityCheckSpec,
    *,
    workers: int = 1,
) -> SanityResult:
    """Sweep + :func:`scipy.optimize.minimize_scalar` (``method="bounded"``) on one variable
    with the others at their nominal values (:class:`SanityCheckSpec`)."""
    from scipy.optimize import minimize_scalar

    variables = problem.variables
    nominal = nominal_values(problem)
    j = next(i for i, v in enumerate(variables) if v.name == check.variable)
    u0 = [variables[i].encode(nominal[variables[i].name]) for i in range(len(variables))]
    s = 1.0 if check.sense == "max" else -1.0

    def objective(res: DesignResult) -> float:
        if res.status != "ok":
            return math.nan
        val = s * float(res.metrics[check.metric])
        if check.penalty_metric is not None:
            val -= check.penalty_weight * float(res.metrics[check.penalty_metric])
        return val

    def task(u: float, k: int) -> DesignTask:
        g = list(u0)
        g[j] = u
        return design_task(decode(variables, g), k, f"sanity_{check.name}_{k:03d}")

    grid = np.linspace(0.0, 1.0, check.sweep_points)
    sweep = evaluate_parallel(
        problem.evaluator, [task(float(u), k) for k, u in enumerate(grid)], workers=workers
    )
    sweep_J = np.array([objective(r) for r in sweep])
    counter = [0]

    def neg(u: float) -> float:
        counter[0] += 1
        val = objective(problem.evaluator.evaluate([task(float(u), 1000 + counter[0])])[0])
        return -val if math.isfinite(val) else math.inf

    res = minimize_scalar(neg, bounds=(0.0, 1.0), method="bounded", options={"xatol": check.xatol})
    var = variables[j]
    ok = np.isfinite(sweep_J)
    best = int(np.nanargmax(np.where(ok, sweep_J, -np.inf)))
    j_scipy = -float(res.fun)
    spacing = 1.0 / (check.sweep_points - 1)
    # Brent's bounded search stops within ~xatol of its optimum (it never evaluates a bound
    # exactly), so J is compared to the sweep's best within slope * 3 xatol, the slope bounded
    # by the largest finite difference of the sweep.
    finite = sweep_J[ok]
    slope = float(np.max(np.abs(np.diff(finite)))) / spacing if finite.size > 1 else 0.0
    tol_j = 3.0 * check.xatol * slope + 1e-12 * max(abs(float(sweep_J[best])), 1.0)
    agree = bool(j_scipy >= sweep_J[best] - tol_j and abs(float(res.x) - grid[best]) <= spacing)
    return SanityResult(
        name=check.name,
        sweep_u=grid,
        sweep_values=np.array([float(var.decode(float(u))) for u in grid]),
        sweep_objective=sweep_J,
        scipy_u=float(res.x),
        scipy_value=float(var.decode(float(res.x))),
        scipy_objective=j_scipy,
        scipy_evaluations=int(res.nfev),
        best_sweep_index=best,
        agree=agree,
    )


def write_sanity(problem: OptimizationProblem, result: SanityResult, out: Path) -> Path:
    """``sanity/<name>.parquet`` (sweep) and ``sanity/<name>.json`` (scipy optimum, agreement)."""
    d = out.expanduser().resolve() / "sanity"
    _write_parquet(
        pa.table(
            {
                "u": pa.array(result.sweep_u, pa.float64()),
                "value": pa.array(result.sweep_values, pa.float64()),
                "objective": pa.array(result.sweep_objective, pa.float64()),
            }
        ),
        d / f"{result.name}.parquet",
    )
    doc = {
        "check": result.name,
        "spec": next(
            c for c in problem.definition.sanity_checks if c.name == result.name
        ).model_dump(),
        "scipy": {
            "method": "scipy.optimize.minimize_scalar(method='bounded')",
            "u": result.scipy_u,
            "value": result.scipy_value,
            "objective": result.scipy_objective,
            "evaluations": result.scipy_evaluations,
        },
        "sweep_best": {
            "index": result.best_sweep_index,
            "value": float(result.sweep_values[result.best_sweep_index]),
            "objective": float(result.sweep_objective[result.best_sweep_index]),
        },
        "sweep_failed_points": int(np.sum(~np.isfinite(result.sweep_objective))),
        "agree": result.agree,
        "provenance": study_provenance(),
    }
    return _write_json(d / f"{result.name}.json", _clean(doc))


# ---------------------------------------------------------------------------------------------
# One-call drivers (CLI)
# ---------------------------------------------------------------------------------------------


def seed_for_task(problem: OptimizationProblem, array_index: int) -> int:
    """Seed of Slurm array task ``array_index`` (the configured seeds in order)."""
    seeds = problem.definition.algorithm.seeds
    if not 0 <= array_index < len(seeds):
        raise ValueError(f"--array-index {array_index} outside 0..{len(seeds) - 1}")
    return seeds[array_index]


def optimize_seed(
    problem: OptimizationProblem,
    out: Path,
    seed: int,
    *,
    workers: int | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Run and write one seed into ``out/seed_<seed>`` (refuses an existing finished seed
    unless ``overwrite``)."""
    from helioswarm.manifest import utc_now_iso

    out = prepare_out(problem, out)
    d = seed_dir(out, seed)
    if (d / RUN_FILE).is_file() and not overwrite:
        raise FileExistsError(f"{d} is finished; use --overwrite to recompute")
    started = utc_now_iso()
    run = run_seed(problem, seed, workers=workers, log=log)
    return write_seed(problem, run, out, started)


__all__ = [
    "FAILED_OBJECTIVE",
    "OPTIMIZATION_SCHEMA_VERSION",
    "AlgorithmSpec",
    "Collation",
    "ConstraintSpec",
    "ConvergenceSpec",
    "Design",
    "EvaluationRecord",
    "ObjectiveSpec",
    "OptimizationFile",
    "OptimizationProblem",
    "Reevaluation",
    "ReevaluationSpec",
    "SanityCheckSpec",
    "SanityResult",
    "Scored",
    "SeedRun",
    "VariableSpec",
    "archive_front",
    "collate",
    "converged",
    "decode",
    "design_task",
    "hypervolume",
    "load_optimization",
    "nominal_values",
    "optimize_seed",
    "pareto_mask",
    "prepare_out",
    "records_table",
    "reevaluate",
    "reference_point",
    "run_sanity_check",
    "run_seed",
    "score",
    "seed_dir",
    "seed_for_task",
    "select_candidates",
    "write_sanity",
    "write_seed",
]
