"""Uncertainty and sensitivity: seeded Monte Carlo, Sobol indices, scenarios (M8; PDF §12.4).

Model (``docs/model_assumptions.md`` M8 "Uncertainty model")
------------------------------------------------------------
Uncertain inputs are split, in the study file, into

* **epistemic** parameters: fixed but imperfectly known technology/economic performance (PV
  efficiency and temperature coefficient, transmitter efficiency, receiver conversion,
  radiator specific mass, economic inputs ...), and
* **aleatory** parameters: variability that remains for a *known* design (ERA5 weather year,
  a pointing-bias realisation, wavefront (phase) error realisation). Operational randomness
  (random outages, scheduling realisations) is not sampled by any shipped study; pointing
  jitter enters through its epistemic sigma, its realisations being integrated analytically in
  the expected capture (``docs/model_assumptions.md`` M8).

Each parameter has a marginal distribution with provenance (``sourced`` with a source, or
``assumed``) and is mapped to the evaluator through a dotted key (a configuration key,
``economics.<name>`` or ``mass_proxy.<field>``; :mod:`helioswarm.evaluation`). Inputs are
independent. Every value is drawn as ``x = F^-1(u)`` with ``u`` uniform on ``[0, 1)``
(:meth:`Distribution.ppf`), so Monte Carlo and Sobol use the same marginals.

**Nested Monte Carlo** (``monte_carlo``): ``N_e`` epistemic samples ``theta_i``; for each,
``N_a`` aleatory samples ``xi_ij``; the evaluation ``(i, j)`` is task ``i N_a + j``. One
``numpy.random.SeedSequence(seed)`` is spawned into two independent streams (epistemic,
aleatory); within a stream, parameters are drawn in their listed order, one array each
(epistemic ``(N_e,)``, aleatory ``(N_e, N_a)``). Reported per output metric ``Y``:

* the conditional (aleatory) mean ``m_i = mean_j Y_ij`` and spread (p05/p95 over ``j``) per
  epistemic sample;
* the **epistemic interval of the expected value**: p05/p50/p95 of ``m_i`` over ``i``;
* the pooled distribution of ``Y_ij``;
* the law-of-total-variance split ``Var Y = E_theta[Var(Y|theta)] + Var_theta[E(Y|theta)]``
  with the unbiased one-way random-effects estimators (:class:`VarianceSplit`): aleatory
  ``E_i[s_i^2]`` (within-row sample variances, ddof = 1) and epistemic
  ``max(Var_i(m_i; ddof=1) - E_i[s_i^2] / N_a, 0)``; the epistemic fraction is the latter over
  their sum. The row means average only ``N_a`` aleatory samples, so ``Var_i(m_i)`` alone also
  contains the aleatory noise ``E[Var(Y|theta)] / N_a``; the uncorrected ratio
  ``Var_i[m_i] / Var Y`` is reported only as ``epistemic_fraction_upper_bound``, and the noise
  sd ``sqrt(E_i[s_i^2] / N_a)`` that it adds to every ``m_i`` (hence to the epistemic interval
  of the expected value) is reported with that interval.

**Sobol indices** (``sobol``; Sobol' 2001; Saltelli 2002/2010 design): SALib 1.6
``SALib.sample.sobol.sample`` (scrambled Sobol' sequence, ``calc_second_order=False``,
``N`` a power of two, rows grouped per base sample with ``D + 2`` rows each) and
``SALib.analyze.sobol.analyze`` (first-order index of Saltelli et al. 2010, total index of
Jansen 1999; bootstrap confidence intervals with ``num_resamples`` resamples at
``conf_level``). The parameters may be epistemic and/or aleatory (named in the study);
the others stay at their nominal values. The fidelity level may set a base value of a
parameter's key (used by the scenarios and by parts that do not vary it); a sample's value
always overrides it. Convergence: the indices are recomputed on the
first ``n'`` base samples (a prefix of the same design) for every ``n'`` in
``convergence_n``. Failed or undefined evaluations make the indices of that output
undefined (reported, never imputed).

**Scenarios** (``scenarios``): explicit run configurations (``configs/scenarios/*.yaml``:
optimistic / nominal / conservative, PV technology cases) evaluated deterministically at the
same fidelity; reported as a range. With ``scenario_consistency: true`` every triangular
epistemic parameter must have ``mode`` = the nominal scenario's value and ``{low, high}`` =
the optimistic and conservative values (checked on load).

Execution: the tasks of each part (``mc``, ``sobol``, ``scenarios``) are split round-robin
over Slurm array tasks (``--array-index/--array-count``), each writing
``<part>/part_<i>_of_<n>.parquet``; :func:`collate` merges the parts (every index must be
present exactly once) and computes the statistics. Results do not depend on the split.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
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
    DesignResult,
    DesignTask,
    EvaluationSettings,
    Evaluator,
    FidelitySpec,
    ForwardModel,
    build_evaluator,
    evaluate_parallel,
    get_dotted,
    results_table,
    study_provenance,
)

Float64Array = NDArray[np.float64]
Part = Literal["mc", "sobol", "scenarios"]
PARTS: Final[tuple[Part, ...]] = ("mc", "sobol", "scenarios")
UNCERTAINTY_SCHEMA_VERSION: Final[int] = 1
DEFINITION_FILE: Final[str] = "uncertainty_definition.yaml"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


# ---------------------------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------------------------


DistributionType = Literal[
    "uniform", "triangular", "normal", "lognormal", "rayleigh", "discrete_uniform"
]


class Distribution(_Model):
    """Marginal distribution of one uncertain input (inverse CDF :meth:`ppf`).

    ``uniform(low, high)``; ``triangular(low, mode, high)``; ``normal(mean, sd)``, truncated
    to ``[low, high]`` when both are given; ``lognormal(median, sigma_log)`` (``ln x`` normal
    with mean ``ln median`` and sd ``sigma_log``); ``rayleigh(scale)`` (magnitude of a 2-D
    isotropic Gaussian with per-axis sd ``scale``, e.g. a pointing-bias realisation);
    ``discrete_uniform(values)`` (equal probability per listed value, e.g. weather years).
    """

    type: DistributionType
    low: float | None = None
    high: float | None = None
    mode: float | None = None
    mean: float | None = None
    sd: float | None = Field(default=None, gt=0.0)
    median: float | None = Field(default=None, gt=0.0)
    sigma_log: float | None = Field(default=None, gt=0.0)
    scale: float | None = Field(default=None, gt=0.0)
    values: tuple[Any, ...] | None = None

    @model_validator(mode="after")
    def _check(self) -> Distribution:
        t = self.type
        need: dict[str, tuple[str, ...]] = {
            "uniform": ("low", "high"),
            "triangular": ("low", "mode", "high"),
            "normal": ("mean", "sd"),
            "lognormal": ("median", "sigma_log"),
            "rayleigh": ("scale",),
            "discrete_uniform": ("values",),
        }
        missing = [k for k in need[t] if getattr(self, k) is None]
        if missing:
            raise ValueError(f"{t} distribution needs {missing}")
        if t in ("uniform", "triangular"):
            assert self.low is not None
            assert self.high is not None
            if not self.low < self.high:
                raise ValueError(f"{t}: need low < high")
            if t == "triangular":
                assert self.mode is not None
                if not self.low <= self.mode <= self.high:
                    raise ValueError("triangular: need low <= mode <= high")
        if t == "normal" and (self.low is None) != (self.high is None):
            raise ValueError("truncated normal needs both low and high (or neither)")
        if t == "discrete_uniform":
            assert self.values is not None
            if len(self.values) < 1:
                raise ValueError("discrete_uniform needs at least one value")
        return self

    def ppf(self, u: Float64Array) -> NDArray[Any]:
        """Inverse CDF at ``u in [0, 1)`` (vectorised)."""
        from scipy import stats

        u = np.asarray(u, dtype=np.float64)
        t = self.type
        if t == "uniform":
            assert self.low is not None
            assert self.high is not None
            return self.low + u * (self.high - self.low)
        if t == "triangular":
            assert self.low is not None
            assert self.high is not None
            assert self.mode is not None
            w = self.high - self.low
            c = (self.mode - self.low) / w
            return np.asarray(stats.triang.ppf(u, c, loc=self.low, scale=w), dtype=np.float64)
        if t == "normal":
            assert self.mean is not None
            assert self.sd is not None
            if self.low is not None and self.high is not None:
                a = (self.low - self.mean) / self.sd
                b = (self.high - self.mean) / self.sd
                return np.asarray(
                    stats.truncnorm.ppf(u, a, b, loc=self.mean, scale=self.sd), dtype=np.float64
                )
            return np.asarray(stats.norm.ppf(u, loc=self.mean, scale=self.sd), dtype=np.float64)
        if t == "lognormal":
            assert self.median is not None
            assert self.sigma_log is not None
            return np.asarray(
                stats.lognorm.ppf(u, self.sigma_log, scale=self.median), dtype=np.float64
            )
        if t == "rayleigh":
            assert self.scale is not None
            return np.asarray(stats.rayleigh.ppf(u, scale=self.scale), dtype=np.float64)
        assert self.values is not None
        idx = np.minimum(np.floor(u * len(self.values)).astype(np.int64), len(self.values) - 1)
        return np.array([self.values[int(i)] for i in idx.ravel()], dtype=object).reshape(u.shape)

    def nominal(self) -> Any:
        """Nominal value: mode (triangular), mean (normal), median (lognormal, uniform:
        midpoint), ``scale sqrt(pi/2)`` mean (rayleigh), first value (discrete)."""
        if self.type == "triangular":
            return self.mode
        if self.type == "normal":
            return self.mean
        if self.type == "lognormal":
            return self.median
        if self.type == "uniform":
            assert self.low is not None
            assert self.high is not None
            return 0.5 * (self.low + self.high)
        if self.type == "rayleigh":
            assert self.scale is not None
            return self.scale * math.sqrt(math.pi / 2.0)
        assert self.values is not None
        return self.values[0]


Provenance = Literal["sourced", "assumed", "derived"]


class UncertainParameter(_Model):
    """One uncertain input.

    Attributes:
        name: Parameter name (column ``x_<name>``).
        key: Dotted key (configuration, ``economics.<name>`` or ``mass_proxy.<field>``).
        distribution: Marginal distribution.
        provenance: ``sourced`` (needs ``source``), ``derived`` or ``assumed``.
        source: Citation / derivation.
        nominal: Value used when the parameter is not varied (Sobol of other inputs);
            default :meth:`Distribution.nominal`.
        unit: Unit.
        description: What it represents.
    """

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=64)
    key: str
    distribution: Distribution
    provenance: Provenance
    source: str | None = None
    nominal: Any = None
    unit: str = "-"
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> UncertainParameter:
        if self.provenance in ("sourced", "derived") and not self.source:
            raise ValueError(f"parameter {self.name!r}: {self.provenance} needs a source")
        return self

    def nominal_value(self) -> Any:
        """Configured nominal value, else the distribution's."""
        return self.nominal if self.nominal is not None else self.distribution.nominal()


class MonteCarloSpec(_Model):
    """Nested Monte Carlo sizes and seed (module docstring)."""

    seed: int = Field(ge=1)
    n_epistemic: int = Field(ge=2)
    n_aleatory: int = Field(ge=1)


class SobolSpec(_Model):
    """Sobol analysis (module docstring). ``seed >= 1``: SALib 1.6 treats a bootstrap seed of
    0 as 'unseeded' (global ``np.random``), which would make the confidence intervals
    non-reproducible."""

    seed: int = Field(ge=1)
    n_base: int = Field(ge=2)
    parameters: tuple[str, ...] = Field(min_length=1)
    num_resamples: int = Field(default=200, ge=10)
    conf_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    convergence_n: tuple[int, ...] = ()

    @field_validator("n_base")
    @classmethod
    def _pow2(cls, v: int) -> int:
        if v & (v - 1):
            raise ValueError("n_base must be a power of two (Sobol' sequence balance)")
        return v

    @model_validator(mode="after")
    def _conv(self) -> SobolSpec:
        for n in self.convergence_n:
            if n & (n - 1) or not 2 <= n <= self.n_base:
                raise ValueError("convergence_n must be powers of two <= n_base")
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError("Sobol parameters must be unique")
        return self


class ScenarioSpec(_Model):
    """One explicit scenario: a run configuration (relative to the study file)."""

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", max_length=64)
    config: str
    description: str = ""


class EvaluationSection(_Model):
    """Metric-layer settings (paths relative to the study file)."""

    power_threshold_W: float = Field(default=0.0, ge=0.0)
    mass_proxy: MassProxyConfig | None = None
    economics_config: str | None = None


class UncertaintyFile(_Model):
    """Schema of ``configs/uncertainty*.yaml`` (module docstring)."""

    uncertainty_schema_version: int
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=96)
    description: str = ""
    evaluator: str = "forward_model"
    synthetic_parameters: dict[str, float] = Field(default_factory=dict)
    base_config: str | None = None
    fidelity_name: str = "screening"
    fidelity: FidelitySpec | None = None
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
    outputs: tuple[str, ...] = Field(min_length=1)
    epistemic: tuple[UncertainParameter, ...] = ()
    aleatory: tuple[UncertainParameter, ...] = ()
    monte_carlo: MonteCarloSpec | None = None
    sobol: SobolSpec | None = None
    scenarios: tuple[ScenarioSpec, ...] = ()
    scenario_consistency: bool = False
    workers: int = Field(default=1, ge=1)

    @field_validator("uncertainty_schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != UNCERTAINTY_SCHEMA_VERSION:
            raise ValueError(f"unsupported uncertainty_schema_version {v} (expected 1)")
        return v

    @model_validator(mode="after")
    def _check(self) -> UncertaintyFile:
        params = [*self.epistemic, *self.aleatory]
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique across epistemic and aleatory")
        keys = [p.key for p in params]
        if len(set(keys)) != len(keys):
            raise ValueError("each key may be varied by one parameter only")
        if self.monte_carlo is not None and not params:
            raise ValueError("monte_carlo needs at least one parameter")
        if self.sobol is not None:
            unknown = set(self.sobol.parameters) - set(names)
            if unknown:
                raise ValueError(f"Sobol parameters {sorted(unknown)} are not defined")
        if self.evaluator == "forward_model" and (
            self.base_config is None or self.fidelity is None
        ):
            raise ValueError("the forward_model evaluator needs base_config and fidelity")
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario names must be unique")
        if self.scenario_consistency and not {"optimistic", "nominal", "conservative"} <= {
            s.name for s in self.scenarios
        }:
            raise ValueError("scenario_consistency needs optimistic/nominal/conservative")
        return self


@dataclass(frozen=True)
class UncertaintyStudy:
    """A loaded, cross-checked uncertainty study.

    Attributes:
        path: Study file.
        definition: Parsed file.
        evaluator: Evaluator at the study's fidelity.
        definition_sha256: SHA-256 of the canonical definition and resolved base mapping.
    """

    path: Path
    definition: UncertaintyFile
    evaluator: Evaluator
    definition_sha256: str

    @property
    def parameters(self) -> tuple[UncertainParameter, ...]:
        """Epistemic then aleatory parameters."""
        return (*self.definition.epistemic, *self.definition.aleatory)

    def kind_of(self, name: str) -> Literal["epistemic", "aleatory"]:
        """Category of a parameter."""
        return "epistemic" if name in {p.name for p in self.definition.epistemic} else "aleatory"

    def scenario_evaluator(self, scenario: ScenarioSpec) -> Evaluator:
        """The study's evaluator with the scenario's run configuration as base."""
        d = self.definition
        ev = build_evaluator(
            d.evaluator,
            study_path=self.path,
            base_config=scenario.config,
            fidelity_name=d.fidelity_name,
            fidelity=d.fidelity,
            settings=_settings(d),
            name_prefix=f"{d.name}_{scenario.name}",
        )
        return ev


def _settings(d: UncertaintyFile) -> EvaluationSettings:
    e = d.evaluation
    return EvaluationSettings(
        power_threshold_W=e.power_threshold_W,
        mass_proxy=e.mass_proxy,
        economics_config=e.economics_config,
    )


def _issue(path: Path, where: str, msg: str) -> ConfigError:
    return ConfigError(str(path), [ConfigIssue(where, msg)])


def load_uncertainty(path: str | Path) -> UncertaintyStudy:
    """Read and cross-check an uncertainty study (keys, outputs, scenario consistency).

    Raises:
        ConfigError: invalid file, unknown key/output, or scenario inconsistency.
    """
    p = Path(path).resolve()
    raw = read_yaml(p)
    try:
        d = UncertaintyFile.model_validate(raw)
    except Exception as exc:
        raise _issue(p, "<uncertainty>", str(exc)) from exc
    args = tuple(par.key for par in (*d.epistemic, *d.aleatory))
    try:
        ev = build_evaluator(
            d.evaluator,
            study_path=p,
            base_config=d.base_config,
            fidelity_name=d.fidelity_name,
            fidelity=d.fidelity,
            settings=_settings(d),
            arguments=args,
            synthetic_parameters=d.synthetic_parameters,
            name_prefix=d.name,
        )
    except (ValueError, ConfigError) as exc:
        raise _issue(p, "evaluator", str(exc)) from exc
    known = ev.known_keys()
    for par in (*d.epistemic, *d.aleatory):
        if par.key not in known:
            raise _issue(p, f"parameters.{par.name}", f"unknown key {par.key!r}")
        if d.fidelity is not None and par.key in d.fidelity.unset:
            raise _issue(p, f"parameters.{par.name}", "key is removed by the fidelity level")
    for m in d.outputs:
        if m not in ev.metric_names():
            raise _issue(p, "outputs", f"unknown metric {m!r}")
    if isinstance(ev, ForwardModel):
        if "lcoe_usd_per_kWh" in d.outputs and d.evaluation.economics_config is None:
            raise _issue(p, "evaluation", "lcoe_usd_per_kWh needs evaluation.economics_config")
        from helioswarm.evaluation import METRIC_DEFINITIONS

        if "constellation" in ev.fidelity_mapping():
            for m in d.outputs:
                if not METRIC_DEFINITIONS[m].constellation:
                    raise _issue(p, "outputs", f"metric {m!r} is not defined for constellations")
        elif "mass_proxy_kg" in d.outputs and d.evaluation.mass_proxy is None:
            raise _issue(p, "evaluation", "mass_proxy_kg needs evaluation.mass_proxy")
        # every parameter's extremes must give valid configurations
        for par in (*d.epistemic, *d.aleatory):
            for u in (0.0, 0.5, 0.999999):
                val = par.distribution.ppf(np.array([u]))[0]
                task = DesignTask(0, "check", ((par.key, _py(val)),))
                try:
                    ev.config_for(task)
                except ConfigError as exc:
                    raise _issue(
                        p, f"parameters.{par.name}", f"value {val!r} gives an invalid config: {exc}"
                    ) from exc
        base_for_hash: Any = ev.fidelity_mapping()
    else:
        base_for_hash = None
    scen_maps: dict[str, Any] = {}
    for s in d.scenarios:
        sev = build_evaluator(
            d.evaluator,
            study_path=p,
            base_config=s.config,
            fidelity_name=d.fidelity_name,
            fidelity=d.fidelity,
            settings=_settings(d),
        )
        if isinstance(sev, ForwardModel):
            try:
                sev.known_keys()
            except ConfigError as exc:
                raise _issue(p, f"scenarios.{s.name}", str(exc)) from exc
            scen_maps[s.name] = sev.fidelity_mapping()
    if d.scenario_consistency:
        _check_scenario_consistency(p, d, scen_maps)
    canon = json.dumps(
        {"definition": d.model_dump(mode="json"), "base": base_for_hash, "scenarios": scen_maps},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return UncertaintyStudy(
        path=p,
        definition=d,
        evaluator=ev,
        definition_sha256=hashlib.sha256(canon.encode()).hexdigest(),
    )


def _check_scenario_consistency(p: Path, d: UncertaintyFile, scen: Mapping[str, Any]) -> None:
    for par in d.epistemic:
        dist = par.distribution
        if (
            dist.type != "triangular"
            or "." not in par.key
            or par.key.startswith(("economics.", "mass_proxy."))
        ):
            continue
        try:
            vals = {k: float(get_dotted(scen[k], par.key)) for k in scen}
        except KeyError as exc:
            raise _issue(p, f"parameters.{par.name}", f"scenario lacks {par.key!r}") from exc
        ok_mode = math.isclose(vals["nominal"], float(dist.mode or 0.0), rel_tol=1e-12)
        ends = sorted([vals["optimistic"], vals["conservative"]])
        ok_ends = math.isclose(ends[0], float(dist.low or 0.0), rel_tol=1e-12) and math.isclose(
            ends[1], float(dist.high or 0.0), rel_tol=1e-12
        )
        if not (ok_mode and ok_ends):
            raise _issue(
                p,
                f"parameters.{par.name}",
                f"triangular ({dist.low}, {dist.mode}, {dist.high}) does not match the scenario "
                f"values {vals} (mode = nominal; low/high = optimistic/conservative)",
            )


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One planned evaluation.

    Attributes:
        index: Task index within its part.
        epistemic_index: ``i`` (MC), else -1.
        aleatory_index: ``j`` (MC), else -1.
        values: Parameter name -> value (every parameter of the study).
        varied: Names varied in this part.
    """

    index: int
    epistemic_index: int
    aleatory_index: int
    values: Mapping[str, Any]
    varied: tuple[str, ...]


def monte_carlo_samples(study: UncertaintyStudy) -> list[Sample]:
    """The nested Monte Carlo design (module docstring)."""
    d = study.definition
    mc = d.monte_carlo
    if mc is None:
        raise ValueError("the study has no monte_carlo section")
    ss_e, ss_a = np.random.SeedSequence(mc.seed).spawn(2)
    rng_e, rng_a = np.random.default_rng(ss_e), np.random.default_rng(ss_a)
    ne, na = mc.n_epistemic, mc.n_aleatory
    epi = {p.name: p.distribution.ppf(rng_e.random(ne)) for p in d.epistemic}
    ale = {p.name: p.distribution.ppf(rng_a.random((ne, na))) for p in d.aleatory}
    varied = tuple(p.name for p in study.parameters)
    out: list[Sample] = []
    for i in range(ne):
        for j in range(na):
            values: dict[str, Any] = {k: _py(v[i]) for k, v in epi.items()}
            values.update({k: _py(v[i, j]) for k, v in ale.items()})
            out.append(Sample(i * na + j, i, j, values, varied))
    return out


def sobol_unit_samples(study: UncertaintyStudy) -> Float64Array:
    """SALib Saltelli/Sobol' design in the unit hypercube, shape ``(N (D + 2), D)``."""
    from SALib.sample import sobol as sobol_sample

    sb = study.definition.sobol
    if sb is None:
        raise ValueError("the study has no sobol section")
    problem = _salib_problem(sb.parameters)
    return np.asarray(
        sobol_sample.sample(problem, sb.n_base, calc_second_order=False, seed=sb.seed),
        dtype=np.float64,
    )


def _salib_problem(names: Sequence[str]) -> dict[str, Any]:
    return {"num_vars": len(names), "names": list(names), "bounds": [[0.0, 1.0]] * len(names)}


def sobol_samples(study: UncertaintyStudy) -> list[Sample]:
    """Sobol design mapped through the marginals; the other parameters at nominal values."""
    sb = study.definition.sobol
    assert sb is not None
    U = sobol_unit_samples(study)
    params = {p.name: p for p in study.parameters}
    cols = {n: params[n].distribution.ppf(U[:, k]) for k, n in enumerate(sb.parameters)}
    nominal = {p.name: p.nominal_value() for p in study.parameters}
    out: list[Sample] = []
    for r in range(U.shape[0]):
        values = dict(nominal)
        values.update({n: _py(cols[n][r]) for n in sb.parameters})
        out.append(Sample(r, -1, -1, values, tuple(sb.parameters)))
    return out


def scenario_samples(study: UncertaintyStudy) -> list[Sample]:
    """One sample per scenario (the parameters stay as in the scenario configuration)."""
    return [Sample(k, -1, -1, {}, ()) for k in range(len(study.definition.scenarios))]


def part_samples(study: UncertaintyStudy, part: Part) -> list[Sample]:
    """Samples of ``part``."""
    if part == "mc":
        return monte_carlo_samples(study)
    if part == "sobol":
        return sobol_samples(study)
    return scenario_samples(study)


def task_of(study: UncertaintyStudy, sample: Sample, part: Part) -> DesignTask:
    """Evaluation task of a sample (only the varied parameters are overridden: the others
    keep the base configuration's values)."""
    keys = {p.name: p.key for p in study.parameters}
    if part == "sobol":
        # Not-varied parameters are pinned to their nominal values in the Sobol design.
        names: Sequence[str] = [p.name for p in study.parameters]
    else:
        names = sample.varied
    return DesignTask(
        index=sample.index,
        label=f"{part}_{sample.index:06d}",
        overrides=tuple((keys[n], sample.values[n]) for n in names),
    )


def indices_for_task(n: int, array_index: int, array_count: int) -> tuple[int, ...]:
    """Round-robin split ``i mod array_count == array_index``."""
    if array_count < 1 or not 0 <= array_index < array_count:
        raise ValueError("need array_count >= 1 and 0 <= array_index < array_count")
    return tuple(i for i in range(n) if i % array_count == array_index)


# ---------------------------------------------------------------------------------------------
# Running parts
# ---------------------------------------------------------------------------------------------


def _write_json(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(doc), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, np.generic):
        return _clean(x.item())
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    return x


def prepare_out(study: UncertaintyStudy, out: Path) -> Path:
    """Create ``out`` with the definition (a different study in the same directory refused)."""
    import os

    out = out.expanduser().resolve()
    marker = out / DEFINITION_FILE
    header = f"# definition_sha256: {study.definition_sha256}\n"

    def check() -> Path:
        first = marker.read_text(encoding="utf-8").splitlines()[0] + "\n"
        if first != header:
            raise FileExistsError(f"{out} holds a different uncertainty study ({first.strip()})")
        return out

    if marker.is_file():
        return check()
    out.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        {"study_file": str(study.path), "definition": study.definition.model_dump(mode="json")},
        sort_keys=False,
    )
    tmp = out / f".{DEFINITION_FILE}.{os.getpid()}.{time.time_ns()}"
    tmp.write_text(header + body, encoding="utf-8")
    try:
        os.link(tmp, marker)
    except FileExistsError:
        return check()
    finally:
        tmp.unlink(missing_ok=True)
    return out


def part_file(out: Path, part: Part, array_index: int, array_count: int) -> Path:
    """``<out>/<part>/part_<i>_of_<n>.parquet``."""
    return out / part / f"part_{array_index:04d}_of_{array_count:04d}.parquet"


def samples_table(
    study: UncertaintyStudy, part: Part, samples: Sequence[Sample], results: Sequence[DesignResult]
) -> pa.Table:
    """Sample bookkeeping (indices, category of each varied input), inputs ``x_<name>``
    (string columns for non-numeric values) and results (``m_<metric>``)."""
    base = results_table(results, sorted(study.evaluator.metric_names()))
    cols: dict[str, Any] = {
        "part": pa.array([part] * len(samples), pa.string()),
        "index": pa.array([s.index for s in samples], pa.int64()),
        "epistemic_index": pa.array([s.epistemic_index for s in samples], pa.int64()),
        "aleatory_index": pa.array([s.aleatory_index for s in samples], pa.int64()),
    }
    if part == "scenarios":
        cols["scenario"] = pa.array(
            [study.definition.scenarios[s.index].name for s in samples], pa.string()
        )
    for p in study.parameters:
        vals = [s.values.get(p.name) for s in samples]
        if all(v is None or isinstance(v, int | float) for v in vals):
            cols[f"x_{p.name}"] = pa.array(
                [math.nan if v is None else float(v) for v in vals], pa.float64()
            )
        else:
            cols[f"x_{p.name}"] = pa.array([None if v is None else str(v) for v in vals])
    for name in base.column_names:
        if name != "index":
            cols[name] = base.column(name)
    return pa.table(cols)


def run_part(
    study: UncertaintyStudy,
    out: Path,
    part: Part,
    *,
    array_index: int = 0,
    array_count: int = 1,
    workers: int | None = None,
    overwrite: bool = False,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Evaluate this array task's share of ``part`` and write its part file."""
    out = prepare_out(study, out)
    target = part_file(out, part, array_index, array_count)
    if target.is_file() and not overwrite:
        raise FileExistsError(f"{target} exists; use --overwrite to recompute")
    samples = part_samples(study, part)
    mine = [samples[i] for i in indices_for_task(len(samples), array_index, array_count)]
    t0 = time.perf_counter()
    if part == "scenarios":
        results: list[DesignResult] = []
        for s in mine:
            spec = study.definition.scenarios[s.index]
            ev = study.scenario_evaluator(spec)
            r = ev.evaluate([DesignTask(s.index, f"scenario_{spec.name}", ())])[0]
            results.append(r)
    else:
        tasks = [task_of(study, s, part) for s in mine]
        results = evaluate_parallel(
            study.evaluator, tasks, workers=workers or study.definition.workers
        )
    table = samples_table(study, part, mine, results)
    meta = {
        b"helioswarm_uncertainty": json.dumps(
            {
                "study": study.definition.name,
                "definition_sha256": study.definition_sha256,
                "part": part,
                "array_index": array_index,
                "array_count": array_count,
                "n_samples_part": len(samples),
                "wall_s": time.perf_counter() - t0,
            }
        ).encode()
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    pq.write_table(table.replace_schema_metadata(meta), tmp, compression="zstd")
    tmp.replace(target)
    if log is not None:
        n_fail = sum(1 for r in results if r.status != "ok")
        log(f"{part}: {len(mine)} evaluations ({n_fail} failed) -> {target}")
    return target


def read_part(study: UncertaintyStudy, out: Path, part: Part) -> pa.Table:
    """Merge every part file of ``part``; every sample index must appear exactly once.

    Raises:
        FileNotFoundError: no part file, or indices missing.
        ValueError: a part file of another study, or duplicate indices.
    """
    files = sorted((out / part).glob("part_*_of_*.parquet"))
    if not files:
        raise FileNotFoundError(f"{out / part}: no part files")
    tables = []
    for f in files:
        t = pq.read_table(f)
        meta = json.loads((t.schema.metadata or {})[b"helioswarm_uncertainty"])
        if meta["definition_sha256"] != study.definition_sha256:
            raise ValueError(f"{f} belongs to another study definition")
        tables.append(t.replace_schema_metadata(None))
    table = pa.concat_tables(tables, promote_options="default")
    idx = np.asarray(table.column("index").to_numpy())
    n = len(part_samples(study, part))
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{part}: duplicate sample indices (mixed array splits?)")
    missing = sorted(set(range(n)) - set(idx.tolist()))
    if missing:
        raise FileNotFoundError(f"{part}: {len(missing)} of {n} samples missing ({missing[:5]}...)")
    return table.take(pa.array(np.argsort(idx, kind="stable")))


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PercentileSummary:
    """Sample summary of a set of finite values (NaN entries when ``n == 0``).

    Attributes:
        n: Number of values.
        mean: Arithmetic mean.
        p05: 5th percentile (``numpy.percentile``, linear interpolation).
        p50: Median.
        p95: 95th percentile.
        min: Minimum.
        max: Maximum.
    """

    n: int
    mean: float
    p05: float
    p50: float
    p95: float
    min: float
    max: float

    @classmethod
    def of(cls, x: Float64Array) -> PercentileSummary:
        """Summary of ``x`` (1-D, finite values)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            nan = math.nan
            return cls(n=0, mean=nan, p05=nan, p50=nan, p95=nan, min=nan, max=nan)
        p = np.percentile(x, [5.0, 50.0, 95.0])
        return cls(
            n=int(x.size),
            mean=float(np.mean(x)),
            p05=float(p[0]),
            p50=float(p[1]),
            p95=float(p[2]),
            min=float(np.min(x)),
            max=float(np.max(x)),
        )


@dataclass(frozen=True)
class VarianceSplit:
    """Epistemic/aleatory variance split of a nested ``(N_e, N_a)`` sample (module docstring).

    With row means ``m_i`` and within-row sample variances ``s_i^2`` (ddof = 1), the unbiased
    one-way random-effects (ANOVA) estimators are (e.g. Searle, Casella & McCulloch 1992,
    "Variance Components" §3.5):

    * ``aleatory = mean_i s_i^2`` estimates ``E_theta[Var(Y | theta)]``;
    * ``epistemic = max(Var_i(m_i; ddof=1) - aleatory / N_a, 0)`` estimates
      ``Var_theta(E[Y | theta])``. ``Var_i(m_i)`` alone also contains the sampling noise of
      each row mean, ``E[Var(Y | theta)] / N_a``, which is aleatory and is subtracted
      here (the uncorrected ratio is kept as ``epistemic_fraction_upper_bound``);
    * ``epistemic_fraction = epistemic / (epistemic + aleatory)``.

    Attributes:
        total_population: ``Var Y`` of the used rows (population variance).
        within_row_population: ``E_i[Var_j Y]`` (population variances).
        between_row_population: ``Var_i[m_i]`` (population): ``total_population =
            within_row_population + between_row_population`` exactly for equal-size rows. It
            includes the aleatory noise ``E[Var(Y | theta)] / N_a``.
        aleatory: Unbiased estimate of ``E_theta[Var(Y | theta)]`` (NaN if ``N_a < 2``).
        epistemic_unclipped: ``Var_i(m_i; ddof=1) - aleatory / N_a`` (may be < 0 by noise;
            NaN if ``N_a < 2`` or fewer than two rows).
        epistemic: ``max(epistemic_unclipped, 0)``.
        epistemic_fraction: ``epistemic / (epistemic + aleatory)`` (NaN if undefined or both 0).
        epistemic_fraction_upper_bound: ``between_row_population / total_population``: the
            biased ratio (includes ``sigma_a^2 / N_a``); an upper bound in expectation.
        aleatory_noise_sd_of_row_mean: ``sqrt(aleatory / N_a)``: the scatter that averaging only
            ``N_a`` aleatory samples adds to every ``m_i`` (the noise floor of the epistemic
            interval of the expected value).
    """

    total_population: float
    within_row_population: float
    between_row_population: float
    aleatory: float
    epistemic_unclipped: float
    epistemic: float
    epistemic_fraction: float
    epistemic_fraction_upper_bound: float
    aleatory_noise_sd_of_row_mean: float

    @classmethod
    def of(cls, Yr: Float64Array) -> VarianceSplit:
        """Split of a finite ``(n_rows, N_a)`` matrix (``n_rows >= 1``)."""
        Yr = np.asarray(Yr, dtype=np.float64)
        n_rows, n_a = Yr.shape
        m = Yr.mean(axis=1)
        total = float(Yr.var())
        within_pop = float(np.mean(Yr.var(axis=1)))
        between_pop = float(m.var())
        nan = math.nan
        aleatory = float(np.mean(Yr.var(axis=1, ddof=1))) if n_a >= 2 else nan
        identifiable = n_a >= 2 and n_rows >= 2
        epi_raw = float(m.var(ddof=1)) - aleatory / n_a if identifiable else nan
        epi = max(epi_raw, 0.0) if math.isfinite(epi_raw) else nan
        denom = epi + aleatory
        frac = epi / denom if math.isfinite(denom) and denom > 0 else nan
        return cls(
            total_population=total,
            within_row_population=within_pop,
            between_row_population=between_pop,
            aleatory=aleatory,
            epistemic_unclipped=epi_raw,
            epistemic=epi,
            epistemic_fraction=frac,
            epistemic_fraction_upper_bound=between_pop / total if total > 0 else nan,
            aleatory_noise_sd_of_row_mean=math.sqrt(aleatory / n_a) if n_a >= 2 else nan,
        )


@dataclass(frozen=True)
class NestedStatistics:
    """Statistics of one output of the nested Monte Carlo (module docstring).

    Attributes:
        n_epistemic: ``N_e``.
        n_aleatory: ``N_a``.
        n_nonfinite: Non-finite (failed/undefined) evaluations.
        epistemic_rows_used: Rows without a non-finite entry (used below).
        pooled: Pooled distribution of every finite ``Y_ij``.
        epistemic_interval_of_expected_value: Percentiles of the row means ``m_i``; each
            ``m_i`` carries aleatory sampling noise of sd
            ``variance.aleatory_noise_sd_of_row_mean`` (None without usable rows).
        aleatory_p05_p95_width: Distribution over rows of the within-row p05–p95 width (None
            if ``N_a < 2``).
        variance: Variance split (None without usable rows).
    """

    n_epistemic: int
    n_aleatory: int
    n_nonfinite: int
    epistemic_rows_used: int
    pooled: PercentileSummary
    epistemic_interval_of_expected_value: PercentileSummary | None
    aleatory_p05_p95_width: PercentileSummary | None
    variance: VarianceSplit | None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (IO boundary)."""
        return asdict(self)


def nested_statistics(Y: Float64Array) -> NestedStatistics:
    """Statistics of an ``(N_e, N_a)`` matrix of one output (module docstring;
    :class:`VarianceSplit` for the estimators). Epistemic rows with any non-finite entry are
    excluded from the conditional statistics (counted)."""
    Y = np.asarray(Y, dtype=np.float64)
    ok_rows = np.all(np.isfinite(Y), axis=1)
    Yr = Y[ok_rows]
    pooled = PercentileSummary.of(Y[np.isfinite(Y)])
    has_rows = Yr.shape[0] > 0
    width = None
    if has_rows and Yr.shape[1] > 1:
        width = PercentileSummary.of(
            np.percentile(Yr, 95.0, axis=1) - np.percentile(Yr, 5.0, axis=1)
        )
    return NestedStatistics(
        n_epistemic=int(Y.shape[0]),
        n_aleatory=int(Y.shape[1]),
        n_nonfinite=int(np.sum(~np.isfinite(Y))),
        epistemic_rows_used=int(np.sum(ok_rows)),
        pooled=pooled,
        epistemic_interval_of_expected_value=(
            PercentileSummary.of(Yr.mean(axis=1)) if has_rows else None
        ),
        aleatory_p05_p95_width=width,
        variance=VarianceSplit.of(Yr) if has_rows else None,
    )


@dataclass(frozen=True)
class SobolIndices:
    """First-order and total Sobol indices with bootstrap confidence half-widths, per
    parameter name (SALib ``analyze.sobol``; module docstring)."""

    S1: Mapping[str, float]
    S1_conf: Mapping[str, float]
    ST: Mapping[str, float]
    ST_conf: Mapping[str, float]

    def as_dict(self) -> dict[str, dict[str, float]]:
        """JSON-serialisable form (IO boundary)."""
        return {k: dict(getattr(self, k)) for k in ("S1", "S1_conf", "ST", "ST_conf")}


def sobol_indices(
    names: Sequence[str],
    Y: Float64Array,
    *,
    num_resamples: int,
    conf_level: float,
    seed: int,
) -> SobolIndices:
    """SALib first-order and total Sobol indices with bootstrap confidence half-widths.

    ``seed`` must be >= 1: SALib 1.6 ``analyze.sobol.analyze`` tests ``if seed:`` and falls
    back to the global, unseeded ``np.random.randint`` for 0 (non-reproducible intervals)."""
    if seed < 1:
        raise ValueError("Sobol bootstrap seed must be >= 1 (SALib treats 0 as 'unseeded')")
    from SALib.analyze import sobol as sobol_analyze

    Si = sobol_analyze.analyze(
        _salib_problem(names),
        np.asarray(Y, dtype=np.float64),
        calc_second_order=False,
        num_resamples=num_resamples,
        conf_level=conf_level,
        seed=seed,
    )
    return SobolIndices(
        S1=dict(zip(names, map(float, Si["S1"]), strict=True)),
        S1_conf=dict(zip(names, map(float, Si["S1_conf"]), strict=True)),
        ST=dict(zip(names, map(float, Si["ST"]), strict=True)),
        ST_conf=dict(zip(names, map(float, Si["ST_conf"]), strict=True)),
    )


@dataclass(frozen=True)
class UncertaintyCollation:
    """Collated statistics.

    Attributes:
        path: ``collated/`` directory.
        summary: The JSON document written to ``collated/summary.json`` (IO boundary; built
            from the typed results below).
        monte_carlo: Nested Monte Carlo statistics per output (empty without an MC part).
        sobol: Sobol indices at the full base size per output whose indices are defined.
    """

    path: Path
    summary: Mapping[str, Any]
    monte_carlo: Mapping[str, NestedStatistics]
    sobol: Mapping[str, SobolIndices]


def collate(study: UncertaintyStudy, out: Path) -> UncertaintyCollation:
    """Merge the finished parts and compute the statistics (module docstring); writes
    ``collated/{mc,sobol,scenarios}.parquet``, ``summary.json``, ``summary.md`` and figures.
    A part that is configured but missing is reported as missing, never skipped silently."""
    out = out.expanduser().resolve()
    d = study.definition
    cdir = out / "collated"
    cdir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "layout": "helioswarm-uncertainty-collation/1",
        "study": d.name,
        "definition_sha256": study.definition_sha256,
        "fidelity": d.fidelity_name,
        "fidelity_description": d.fidelity.description if d.fidelity is not None else d.evaluator,
        "outputs": list(d.outputs),
        "parameters": [
            {
                "name": p.name,
                "category": study.kind_of(p.name),
                "key": p.key,
                "distribution": p.distribution.model_dump(exclude_none=True),
                "provenance": p.provenance,
                "source": p.source,
                "unit": p.unit,
            }
            for p in study.parameters
        ],
        "parts": {},
        "provenance": study_provenance(),
    }
    tables: dict[str, pa.Table] = {}
    mc_stats: dict[str, NestedStatistics] = {}
    sobol_stats: dict[str, SobolIndices] = {}
    configured = {
        "mc": d.monte_carlo is not None,
        "sobol": d.sobol is not None,
        "scenarios": bool(d.scenarios),
    }
    for part in PARTS:
        if not configured[part]:
            continue
        try:
            t = read_part(study, out, part)
        except FileNotFoundError as exc:
            summary["parts"][part] = {"status": "missing", "error": str(exc)}
            continue
        tables[part] = t
        pq.write_table(t, cdir / f"{part}.parquet", compression="zstd")
        status = t.column("status").to_pylist()
        summary["parts"][part] = {
            "status": "complete",
            "n": t.num_rows,
            "failed": sum(1 for s in status if s != "ok"),
            "failures": sorted(
                {
                    f"{a}: {b}"
                    for a, b, s in zip(
                        t.column("error_type").to_pylist(),
                        t.column("error").to_pylist(),
                        status,
                        strict=True,
                    )
                    if s != "ok"
                }
            )[:10],
        }
    if "mc" in tables:
        assert d.monte_carlo is not None
        t = tables["mc"]
        ne, na = d.monte_carlo.n_epistemic, d.monte_carlo.n_aleatory
        mc_stats = {
            m: nested_statistics(
                np.asarray(t.column(f"m_{m}").to_numpy(), dtype=np.float64).reshape(ne, na)
            )
            for m in d.outputs
        }
        summary["monte_carlo"] = {m: st.as_dict() for m, st in mc_stats.items()}
        summary["monte_carlo_settings"] = d.monte_carlo.model_dump()
    if "sobol" in tables:
        assert d.sobol is not None
        sb = d.sobol
        t = tables["sobol"]
        res: dict[str, Any] = {}
        D = len(sb.parameters)
        for m in d.outputs:
            Y = np.asarray(t.column(f"m_{m}").to_numpy(), dtype=np.float64)
            if not np.all(np.isfinite(Y)):
                res[m] = {"status": "undefined", "n_nonfinite": int(np.sum(~np.isfinite(Y)))}
                continue
            if float(np.var(Y)) == 0.0:
                res[m] = {"status": "constant output (no variance to apportion)"}
                continue
            conv = {
                n: sobol_indices(
                    sb.parameters,
                    Y[: n * (D + 2)],
                    num_resamples=sb.num_resamples,
                    conf_level=sb.conf_level,
                    seed=sb.seed,
                )
                for n in sorted({*sb.convergence_n, sb.n_base})
            }
            sobol_stats[m] = conv[sb.n_base]
            res[m] = {
                "status": "ok",
                "variance": float(np.var(Y)),
                "indices": conv[sb.n_base].as_dict(),
                "convergence": {str(n): c.as_dict() for n, c in conv.items()},
            }
        summary["sobol"] = res
        summary["sobol_settings"] = sb.model_dump()
    if "scenarios" in tables:
        t = tables["scenarios"]
        summary["scenarios"] = {
            name: {
                "status": t.column("status")[k].as_py(),
                "error": t.column("error")[k].as_py(),
                **{m: t.column(f"m_{m}")[k].as_py() for m in d.outputs},
            }
            for k, name in enumerate(t.column("scenario").to_pylist())
        }
        summary["scenario_range"] = {
            m: _range([t.column(f"m_{m}")[k].as_py() for k in range(t.num_rows)]) for m in d.outputs
        }
    _write_json(cdir / "summary.json", summary)
    (cdir / "summary.md").write_text(_summary_md(study, summary), encoding="utf-8")
    _figures(study, summary, tables, cdir)
    return UncertaintyCollation(path=cdir, summary=summary, monte_carlo=mc_stats, sobol=sobol_stats)


def _range(vals: Sequence[Any]) -> dict[str, float | None]:
    x = np.array([math.nan if v is None else float(v) for v in vals])
    x = x[np.isfinite(x)]
    return {"min": float(x.min()), "max": float(x.max())} if x.size else {"min": None, "max": None}


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}" if math.isfinite(v) else "—"
    return str(v)


def _summary_md(study: UncertaintyStudy, s: Mapping[str, Any]) -> str:
    d = study.definition
    lines = [
        f"# Uncertainty study `{d.name}`",
        "",
        d.description,
        "",
        f"Fidelity `{d.fidelity_name}`: {s['fidelity_description']}",
        "",
        "## Inputs",
        "",
        "| name | category | key | distribution | provenance | source |",
        "|---|---|---|---|---|---|",
    ]
    for p in s["parameters"]:
        dist = ", ".join(f"{k}={v}" for k, v in p["distribution"].items())
        lines.append(
            f"| {p['name']} | {p['category']} | `{p['key']}` | {dist} | {p['provenance']} | "
            f"{p['source'] or ''} |"
        )
    lines += ["", "## Parts", ""]
    for part, info in s["parts"].items():
        lines.append(f"- `{part}`: {info}")
    if "monte_carlo" in s:
        mc = s["monte_carlo_settings"]
        lines += [
            "",
            f"## Nested Monte Carlo (N_e = {mc['n_epistemic']}, N_a = {mc['n_aleatory']}, "
            f"seed {mc['seed']})",
            "",
            "Epistemic variance fraction: unbiased between-row component "
            "`max(Var_i m_i - E_i s_i^2 / N_a, 0)` over itself plus `E_i s_i^2` (ddof = 1); "
            "'upper bound' is the uncorrected `Var_i m_i / Var Y`, which also counts the "
            "aleatory noise `sigma_a^2 / N_a` of each row mean. The epistemic interval of "
            "E_aleatory (percentiles of the row means `m_i`) carries that noise too: its sd is "
            "the 'noise sd' column.",
            "",
            "| output | pooled p05 | p50 | p95 | epistemic interval of E_aleatory: p05 | p50 | "
            "p95 | noise sd of m_i | epistemic variance fraction | upper bound | non-finite |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m, st in s["monte_carlo"].items():
            pooled = st["pooled"]
            e = st.get("epistemic_interval_of_expected_value") or {}
            v = st.get("variance") or {}
            lines.append(
                f"| {m} | {_fmt(pooled.get('p05'))} | {_fmt(pooled.get('p50'))} | "
                f"{_fmt(pooled.get('p95'))} | {_fmt(e.get('p05'))} | {_fmt(e.get('p50'))} | "
                f"{_fmt(e.get('p95'))} | {_fmt(v.get('aleatory_noise_sd_of_row_mean'))} | "
                f"{_fmt(v.get('epistemic_fraction'))} | "
                f"{_fmt(v.get('epistemic_fraction_upper_bound'))} | {st['n_nonfinite']} |"
            )
    if "sobol" in s:
        sb = s["sobol_settings"]
        lines += [
            "",
            f"## Sobol indices (N = {sb['n_base']}, {sb['num_resamples']} bootstrap resamples, "
            f"{sb['conf_level']} confidence)",
            "",
        ]
        for m, r in s["sobol"].items():
            if r.get("status") != "ok":
                lines.append(f"- `{m}`: {r}")
                continue
            ind = r["indices"]
            lines += [
                f"### `{m}`",
                "",
                "| parameter | S1 | ± | ST | ± |",
                "|---|---|---|---|---|",
            ]
            order = sorted(ind["ST"], key=lambda k: -ind["ST"][k])
            for k in order:
                lines.append(
                    f"| {k} | {_fmt(ind['S1'][k])} | {_fmt(ind['S1_conf'][k])} | "
                    f"{_fmt(ind['ST'][k])} | {_fmt(ind['ST_conf'][k])} |"
                )
            lines += ["", "Convergence (ST by N):", ""]
            for n, c in r["convergence"].items():
                lines.append(
                    f"- N = {n}: "
                    + ", ".join(f"{k} {c['ST'][k]:.3g}±{c['ST_conf'][k]:.2g}" for k in order)
                )
            lines.append("")
    if "scenarios" in s:
        lines += ["", "## Scenarios", "", "| scenario | status | " + " | ".join(d.outputs) + " |"]
        lines.append("|---" * (len(d.outputs) + 2) + "|")
        for name, r in s["scenarios"].items():
            lines.append(
                f"| {name} | {r['status']} | " + " | ".join(_fmt(r[m]) for m in d.outputs) + " |"
            )
    return "\n".join(lines) + "\n"


def _figures(
    study: UncertaintyStudy, s: Mapping[str, Any], tables: Mapping[str, pa.Table], cdir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = study.definition
    if "mc" in tables:
        n = len(d.outputs)
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), squeeze=False)
        for ax, m in zip(axes[0], d.outputs, strict=True):
            y = np.asarray(tables["mc"].column(f"m_{m}").to_numpy(), dtype=np.float64)
            y = y[np.isfinite(y)]
            if y.size:
                ax.hist(y, bins=30, color="tab:blue", alpha=0.8)
            ax.set_xlabel(m)
        axes[0][0].set_ylabel("samples (pooled)")
        fig.suptitle(f"{d.name}: nested Monte Carlo")
        fig.tight_layout()
        fig.savefig(cdir / "monte_carlo.png", dpi=110)
        plt.close(fig)
    if "sobol" in s:
        ok = {m: r for m, r in s["sobol"].items() if r.get("status") == "ok"}
        if ok:
            fig, axes = plt.subplots(1, len(ok), figsize=(4.5 * len(ok), 3.6), squeeze=False)
            for ax, (m, r) in zip(axes[0], ok.items(), strict=True):
                ind = r["indices"]
                names = list(ind["ST"])
                x = np.arange(len(names))
                ax.bar(
                    x - 0.2,
                    [ind["S1"][k] for k in names],
                    0.4,
                    yerr=[ind["S1_conf"][k] for k in names],
                    label="S1",
                )
                ax.bar(
                    x + 0.2,
                    [ind["ST"][k] for k in names],
                    0.4,
                    yerr=[ind["ST_conf"][k] for k in names],
                    label="ST",
                )
                ax.set_xticks(x, names, rotation=45, ha="right", fontsize=7)
                ax.set_title(m, fontsize=9)
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(cdir / "sobol.png", dpi=110)
            plt.close(fig)


__all__ = [
    "PARTS",
    "UNCERTAINTY_SCHEMA_VERSION",
    "Distribution",
    "MonteCarloSpec",
    "NestedStatistics",
    "PercentileSummary",
    "Sample",
    "ScenarioSpec",
    "SobolIndices",
    "SobolSpec",
    "UncertainParameter",
    "UncertaintyCollation",
    "UncertaintyFile",
    "UncertaintyStudy",
    "VarianceSplit",
    "collate",
    "indices_for_task",
    "load_uncertainty",
    "monte_carlo_samples",
    "nested_statistics",
    "part_file",
    "part_samples",
    "prepare_out",
    "read_part",
    "run_part",
    "samples_table",
    "scenario_samples",
    "sobol_indices",
    "sobol_samples",
    "sobol_unit_samples",
    "task_of",
]
