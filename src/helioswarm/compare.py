"""Harmonised heliocentric-vs-GEO comparison from saved runs (M7b; PDF O5, §15, §16).

A comparison file names sweeps (one per case) whose base configurations share their
technology blocks through ``extends`` (``configs/comparison/``)::

    comparison_schema_version: 1
    name: geo_vs_helio
    mass_proxy:                       # assumed; applied identically to every case
      collector_areal_mass_kg_m2: 1.0
      transmitter_areal_mass_kg_m2: 50.0
      bus_mass_kg: 500.0
    cases:
      - label: helio_1au_optical
        sweep: helio_1au_optical_sweep.yaml
      - ...

:func:`load_comparison` loads every sweep, identifies its **base point** (the grid point whose
swept values equal the base configuration's) and checks the **harmonisation**: within each
technology group (``transmitter.mode``) the resolved base configurations may differ only in
orbit/geometry keys (:data:`ORBIT_GEOMETRY_PREFIXES`) and in the beam-model selection of the
propagation regime (:data:`PROPAGATION_REGIME_PREFIXES`); anything else raises
:class:`HarmonisationError`. :func:`write_comparison` then reads the saved runs
(``<runs_dir>/<sweep name>/points/p<iiii>/``: ``summary.json``, ``config_resolved.yaml``;
failed points' ``error.json``) - never in-memory results - and writes:

* ``comparison_points.csv`` / ``.md``: every point of every sweep, with its status;
* ``comparison_baseline.csv`` / ``.md``: the base points, side by side per technology;
* ``harmonised_assumptions.csv`` / ``.md``: every configuration leaf with its unit,
  provenance label and value per case, classified as shared by all cases, shared within the
  technology group, or orbit/geometry; plus the mass-proxy assumptions;
* ``comparison.png``: delivered mean power and end-to-end efficiency versus receiver radius
  for every aperture, both architectures, per technology;
* ``comparison_manifest.json``: inputs (file hashes of every summary read), code provenance.

Every metric is on the same system boundary: ``P_grid`` at the receiver's grid output for one
satellite and one receiver site (PDF §10.1 chain), over the same window and time grid:

* delivered mean power ``E_grid / window``; availability; capacity factor (with the run's
  rated power); end-to-end efficiency ``E_grid / E_incident_on_collector``;
* eclipse fraction: time fraction with the ``ECLIPSE`` (umbra) bit, and with
  ``PARTIAL_ECLIPSE`` (penumbra) separately;
* capture: energy-weighted ``E_captured / E_atm`` (beam energy at the receiver plane that the
  receiver intercepts) and the mean expected capture over available epochs;
* range: mean, min and max satellite-receiver range;
* power density at the receiver: the largest peak incident irradiance while transmitting
  (aberration-free upper bound) and the mean intercepted flux ``E_captured / (t_available
  pi R_rx^2)``;
* mass proxy ``m = m_bus + s_col A_col + s_rad A_rad + s_tx pi D^2/4`` (the M7 formula, with
  ``A_rad`` the largest required radiator area of the run and ``s_rad =
  thermal.radiator_specific_mass_kg_m2``) and ``m / (P_mean / 1 kW)``. Not included in the
  mass, for either architecture: launch/transfer propellant, GEO station-keeping propellant,
  structure, the ground receiver.

Context values from external studies (e.g. the NASA 2024 SBSP report) are deliberately not
part of these tables (PDF §16, §19: comparator/context only).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from helioswarm.config import (
    ConfigError,
    ConfigIssue,
    MassProxyConfig,
    RunConfig,
    config_hash,
    field_meta,
    load_config,
    provenance_table,
    read_yaml,
)
from helioswarm.sweep import POINTS_DIR, SweepPlan, SweepPoint, load_sweep

COMPARISON_SCHEMA_VERSION: Final[int] = 1

ORBIT_GEOMETRY_PREFIXES: Final[tuple[str, ...]] = (
    "orbit.",
    "time.integrator_step_s",
    "project.",
)
"""Configuration leaves that may differ between the cases of one technology group: the orbit
(and the L1 integrator step that only the L1 orbit model reads) and run metadata."""

PROPAGATION_REGIME_PREFIXES: Final[tuple[str, ...]] = ("models.beam",)
"""Leaves that may also differ: the beam-model selection and its evaluation method
(``models.beam``, ``models.beam_evaluation``, ``models.beam_table_*``). One scalar-diffraction
physics, chosen by the propagation regime of each geometry: Fraunhofer models where
``N_F = D^2/(lambda R) << 0.5`` (heliocentric ranges), Fresnel-Hankel in the near field (the
optical GEO link, N_F = 2.8); the two agree where both are valid (tested)."""

POINTS_CSV: Final[str] = "comparison_points.csv"
POINTS_MD: Final[str] = "comparison_points.md"
BASELINE_CSV: Final[str] = "comparison_baseline.csv"
BASELINE_MD: Final[str] = "comparison_baseline.md"
ASSUMPTIONS_CSV: Final[str] = "harmonised_assumptions.csv"
ASSUMPTIONS_MD: Final[str] = "harmonised_assumptions.md"
FIGURE_FILE: Final[str] = "comparison.png"
MANIFEST_FILE: Final[str] = "comparison_manifest.json"

Architecture = Literal["heliocentric", "geostationary", "geocentric"]
PointStatus = Literal["ok", "failed", "missing", "incomplete"]


class HarmonisationError(ValueError):
    """The base configurations of one technology group differ outside orbit/geometry."""


class ComparisonCaseSpec(BaseModel):
    """One case of a comparison file: a label and a sweep file (relative to the file)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=96)
    sweep: str


class ComparisonFile(BaseModel):
    """Schema of a comparison YAML file (module docstring)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_schema_version: int
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=96)
    description: str = ""
    mass_proxy: MassProxyConfig
    cases: tuple[ComparisonCaseSpec, ...] = Field(min_length=2)

    @field_validator("comparison_schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != COMPARISON_SCHEMA_VERSION:
            raise ValueError(f"unsupported comparison_schema_version {v} (expected 1)")
        return v

    @field_validator("cases")
    @classmethod
    def _unique(cls, v: tuple[ComparisonCaseSpec, ...]) -> tuple[ComparisonCaseSpec, ...]:
        labels = [c.label for c in v]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            raise ValueError(f"case labels must be unique; duplicated: {dupes}")
        return v


def architecture_of(config: RunConfig) -> Architecture:
    """``heliocentric`` (Sun-centred orbit), ``geostationary`` (the preset) or ``geocentric``."""
    if config.orbit.elements_center == "SUN":
        return "heliocentric"
    return "geostationary" if config.orbit.is_geostationary_preset else "geocentric"


@dataclass(frozen=True)
class ComparisonCase:
    """A loaded case.

    Attributes:
        label: Case label.
        sweep: The loaded sweep.
        base_config: The base configuration (the sweep's ``base_config``, resolved).
        base_index: Index of the grid point equal to the base configuration.
        mode: ``transmitter.mode`` (technology group).
        architecture: :func:`architecture_of` the base configuration.
    """

    label: str
    sweep: SweepPlan
    base_config: RunConfig
    base_index: int
    mode: str
    architecture: Architecture


@dataclass(frozen=True)
class ComparisonPlan:
    """A loaded, harmonisation-checked comparison.

    Attributes:
        path: Comparison file.
        definition: Parsed file.
        cases: Loaded cases, in file order.
        file_sha256: SHA-256 of the comparison file's bytes.
    """

    path: Path
    definition: ComparisonFile
    cases: tuple[ComparisonCase, ...]
    file_sha256: str


def _leaf(config: RunConfig, path: str) -> Any:
    for row in provenance_table(config):
        if row.path == path:
            return row.value
    raise KeyError(path)


def _base_index(sweep: SweepPlan, base: RunConfig) -> int:
    keys = list(sweep.definition.parameters)
    base_values = {k: _leaf(base, k) for k in keys}
    hits = [p.index for p in sweep.points if all(dict(p.values)[k] == base_values[k] for k in keys)]
    if len(hits) != 1:
        raise ConfigError(
            str(sweep.path),
            [
                ConfigIssue(
                    "parameters",
                    f"the grid must contain the base configuration's values {base_values} "
                    f"exactly once (found {len(hits)})",
                )
            ],
        )
    return hits[0]


def is_orbit_geometry_path(path: str) -> bool:
    """Whether a leaf path is an orbit/geometry key (:data:`ORBIT_GEOMETRY_PREFIXES`)."""
    return any(path == p or path.startswith(p) for p in ORBIT_GEOMETRY_PREFIXES)


def is_propagation_regime_path(path: str) -> bool:
    """Whether a leaf path selects the beam model (:data:`PROPAGATION_REGIME_PREFIXES`)."""
    return any(path == p or path.startswith(p) for p in PROPAGATION_REGIME_PREFIXES)


def may_differ(path: str) -> bool:
    """Whether a leaf path may differ between the cases of one technology group."""
    return is_orbit_geometry_path(path) or is_propagation_regime_path(path)


def harmonisation_violations(configs: Sequence[tuple[str, RunConfig]]) -> list[str]:
    """Leaves that differ between configurations although they must not (:func:`may_differ`).

    Args:
        configs: ``(label, config)`` pairs of one technology group.

    Returns:
        One message per differing leaf (empty if harmonised).
    """
    if len(configs) < 2:
        return []
    tables = [{r.path: r.value for r in provenance_table(c)} for _, c in configs]
    paths = sorted(set().union(*tables))
    out: list[str] = []
    for path in paths:
        if may_differ(path):
            continue
        values = [t.get(path, "<absent>") for t in tables]
        if any(v != values[0] for v in values[1:]):
            listing = ", ".join(f"{lbl}={v!r}" for (lbl, _), v in zip(configs, values, strict=True))
            out.append(f"{path}: {listing}")
    return out


def load_comparison(path: str | Path) -> ComparisonPlan:
    """Read a comparison file, load its sweeps and check the harmonisation.

    Raises:
        ConfigError: invalid file or sweep, or a grid without its base point.
        HarmonisationError: a technology group differs outside orbit/geometry keys.
    """
    p = Path(path).resolve()
    raw = read_yaml(p)
    try:
        definition = ComparisonFile.model_validate(raw)
    except Exception as exc:
        raise ConfigError(str(p), [ConfigIssue("<comparison>", str(exc))]) from exc
    cases: list[ComparisonCase] = []
    for spec in definition.cases:
        sweep = load_sweep(p.parent / spec.sweep)
        base = load_config(sweep.base_path)
        if sweep.definition.overrides:
            raise ConfigError(
                str(sweep.path),
                [ConfigIssue("overrides", "comparison sweeps must not override the base")],
            )
        cases.append(
            ComparisonCase(
                label=spec.label,
                sweep=sweep,
                base_config=base,
                base_index=_base_index(sweep, base),
                mode=base.transmitter.mode,
                architecture=architecture_of(base),
            )
        )
    for mode in sorted({c.mode for c in cases}):
        group = [(c.label, c.base_config) for c in cases if c.mode == mode]
        bad = harmonisation_violations(group)
        if bad:
            raise HarmonisationError(
                f"comparison '{definition.name}', {mode} cases are not harmonised (they may "
                "differ only in orbit/geometry keys "
                + str(ORBIT_GEOMETRY_PREFIXES + PROPAGATION_REGIME_PREFIXES)
                + "): "
                + "; ".join(bad)
            )
        grids = {
            json.dumps(c.sweep.definition.parameters, sort_keys=True)
            for c in cases
            if c.mode == mode
        }
        if len(grids) != 1:
            raise HarmonisationError(
                f"comparison '{definition.name}': the {mode} sweeps use different grids"
            )
    return ComparisonPlan(
        path=p,
        definition=definition,
        cases=tuple(cases),
        file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    )


# ---------------------------------------------------------------------------------------------
# Metrics of one saved point
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonRow:
    """Metrics of one sweep point (``None`` where undefined or the point did not run).

    Attributes are SI with unit suffixes; see the module docstring for the definitions.
    """

    case: str
    mode: str
    architecture: str
    point: str
    is_base: bool
    aperture_diameter_m: float
    receiver_radius_m: float
    status: str
    error: str | None
    config_hash: str
    delivered_mean_power_W: float | None = None
    delivered_energy_J: float | None = None
    availability: float | None = None
    capacity_factor: float | None = None
    rated_power_W: float | None = None
    end_to_end_efficiency: float | None = None
    eclipse_umbra_fraction: float | None = None
    eclipse_penumbra_fraction: float | None = None
    capture_energy_weighted: float | None = None
    capture_mean_when_available: float | None = None
    range_mean_m: float | None = None
    range_min_m: float | None = None
    range_max_m: float | None = None
    peak_irradiance_max_W_m2: float | None = None
    mean_intercepted_flux_W_m2: float | None = None
    radiator_area_m2: float | None = None
    mass_proxy_kg: float | None = None
    mass_per_delivered_kW_kg: float | None = None
    summary_sha256: str | None = None


def mass_proxy_value_kg(
    mass_proxy: MassProxyConfig, config: RunConfig, radiator_area_m2: float
) -> float:
    """``m_bus + s_col A_col + s_rad A_rad + s_tx pi D^2/4`` [kg] (M7 mass proxy formula).

    ``A_rad`` is ``thermal.radiator_area_m2`` if configured, else ``radiator_area_m2`` (the
    largest required area of the run); ``s_rad = thermal.radiator_specific_mass_kg_m2``.

    Raises:
        ValueError: ``thermal.radiator_specific_mass_kg_m2`` is not configured.
    """
    s_rad = config.thermal.radiator_specific_mass_kg_m2
    if s_rad is None:
        raise ValueError("the mass proxy needs thermal.radiator_specific_mass_kg_m2")
    a_rad = config.thermal.radiator_area_m2
    a_rad = float(radiator_area_m2) if a_rad is None else float(a_rad)
    d = config.transmitter.aperture_diameter_m
    return float(
        mass_proxy.bus_mass_kg
        + mass_proxy.collector_areal_mass_kg_m2 * config.collector.area_m2
        + s_rad * a_rad
        + mass_proxy.transmitter_areal_mass_kg_m2 * math.pi * d * d / 4.0
    )


def point_row(
    case: ComparisonCase, point: SweepPoint, point_dir: Path, mass_proxy: MassProxyConfig
) -> ComparisonRow:
    """Metrics of one saved sweep point (module docstring), from its files only.

    Raises:
        ValueError: the saved run belongs to another configuration (stale directory).
    """
    from helioswarm.io import CONFIG_FILE, ERROR_FILE, SUMMARY_FILE, read_json
    from helioswarm.metrics import read_summary

    cfg_hash = config_hash(point.config)
    base: dict[str, Any] = {
        "case": case.label,
        "mode": case.mode,
        "architecture": case.architecture,
        "point": point.label,
        "is_base": point.index == case.base_index,
        "aperture_diameter_m": float(point.config.transmitter.aperture_diameter_m),
        "receiver_radius_m": float(point.config.receiver.radius_m),
        "config_hash": cfg_hash,
    }
    if (point_dir / ERROR_FILE).is_file():
        err = read_json(point_dir / ERROR_FILE)
        if err.get("config_hash") not in (None, cfg_hash):
            raise ValueError(f"{point_dir}: error.json is for another configuration")
        return ComparisonRow(status="failed", error=f"{err['error_type']}: {err['error']}", **base)
    summary_path = point_dir / SUMMARY_FILE
    if not summary_path.is_file() or not (point_dir / CONFIG_FILE).is_file():
        status = "incomplete" if point_dir.is_dir() and any(point_dir.iterdir()) else "missing"
        return ComparisonRow(status=status, error=None, **base)
    saved_cfg = load_config(point_dir / CONFIG_FILE)
    s = read_summary(summary_path)
    if s.config_hash != cfg_hash or config_hash(saved_cfg) != cfg_hash:
        raise ValueError(
            f"{point_dir}: saved run has config hash {s.config_hash}, the sweep point "
            f"{cfg_hash}; re-run the point (stale or foreign run directory)"
        )
    window = s.time.window_s
    e = s.stage_energy_J
    avail_time = s.availability * window
    area_rx = math.pi * saved_cfg.receiver.radius_m**2
    p_mean = s.p_grid_W.time_mean
    rad_area = s.thermal.radiator_area_m2_max
    mass = mass_proxy_value_kg(mass_proxy, saved_cfg, rad_area)
    blocking = s.unavailability.blocking
    return ComparisonRow(
        status="ok",
        error=None,
        delivered_mean_power_W=p_mean,
        delivered_energy_J=s.delivered_energy_J,
        availability=s.availability,
        capacity_factor=s.capacity_factor,
        rated_power_W=s.capacity_factor_denominator.rated_power_W,
        end_to_end_efficiency=s.end_to_end_efficiency.energy_ratio,
        eclipse_umbra_fraction=blocking["ECLIPSE"].time_fraction_flagged,
        eclipse_penumbra_fraction=s.unavailability.limiting_flags_time_fraction["PARTIAL_ECLIPSE"],
        capture_energy_weighted=(e["captured_J"] / e["atm_J"]) if e["atm_J"] > 0.0 else None,
        capture_mean_when_available=s.beam_capture.eta_capture_when_available.mean,
        range_mean_m=s.geometry.range_m.mean,
        range_min_m=s.geometry.range_m.min,
        range_max_m=s.geometry.range_m.max,
        peak_irradiance_max_W_m2=s.receiver_power_density_W_m2.incident_peak_when_transmitting.max,
        mean_intercepted_flux_W_m2=(
            e["captured_J"] / (avail_time * area_rx) if avail_time > 0.0 else None
        ),
        radiator_area_m2=rad_area,
        mass_proxy_kg=mass,
        mass_per_delivered_kW_kg=(
            mass / (p_mean / 1000.0) if p_mean is not None and p_mean > 0.0 else None
        ),
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        **base,
    )


def collect_rows(plan: ComparisonPlan, runs_dir: Path) -> tuple[ComparisonRow, ...]:
    """Rows of every point of every case, in case and grid order."""
    rows: list[ComparisonRow] = []
    for case in plan.cases:
        sweep_dir = runs_dir / case.sweep.definition.name / POINTS_DIR
        for pt in case.sweep.points:
            rows.append(point_row(case, pt, sweep_dir / pt.label, plan.definition.mass_proxy))
    return tuple(rows)


# ---------------------------------------------------------------------------------------------
# Harmonised-assumption table
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AssumptionRow:
    """One assumption with its value in every case.

    Attributes:
        path: Configuration leaf (or ``mass_proxy.*`` of the comparison file).
        unit: Unit.
        provenance: Effective provenance label(s) (``assumed``, ``sourced``, ...; several
            joined with ``/`` if the cases label it differently).
        scope: ``shared by all cases``, ``shared within technology``, ``orbit/geometry``,
            ``propagation regime (beam model by geometry)`` or ``comparison
            (post-processing)``.
        values: Case label -> value.
    """

    path: str
    unit: str
    provenance: str
    scope: str
    values: dict[str, Any]


def harmonised_assumptions(plan: ComparisonPlan) -> tuple[AssumptionRow, ...]:
    """Every leaf of the base configurations, classified (module docstring)."""
    tables = {c.label: {r.path: r for r in provenance_table(c.base_config)} for c in plan.cases}
    paths: list[str] = []
    for c in plan.cases:
        for p in tables[c.label]:
            if p not in paths:
                paths.append(p)
    out: list[AssumptionRow] = []
    for path in paths:
        rows = {lbl: t.get(path) for lbl, t in tables.items()}
        present = [r for r in rows.values() if r is not None]
        values = {lbl: (None if r is None else r.value) for lbl, r in rows.items()}
        all_same = len(present) == len(rows) and all(r.value == present[0].value for r in present)
        if all_same:
            scope = "shared by all cases"
        elif is_orbit_geometry_path(path):
            scope = "orbit/geometry"
        elif is_propagation_regime_path(path):
            scope = "propagation regime (beam model by geometry)"
        else:
            scope = "shared within technology"
        provs = sorted({r.provenance for r in present})
        out.append(
            AssumptionRow(
                path=path,
                unit=present[0].unit if present else "-",
                provenance="/".join(provs),
                scope=scope,
                values=values,
            )
        )
    mp = plan.definition.mass_proxy
    for name in type(mp).model_fields:
        meta = field_meta(type(mp), name)
        out.append(
            AssumptionRow(
                path=f"mass_proxy.{name}",
                unit=meta.unit,
                provenance=meta.provenance,
                scope="comparison (post-processing)",
                values={c.label: getattr(mp, name) for c in plan.cases},
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------

_ROW_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in fields(ComparisonRow))

_MD_COLUMNS: Final[tuple[tuple[str, str, float], ...]] = (
    ("delivered_mean_power_W", "P_grid mean [kW]", 1e-3),
    ("availability", "availability", 1.0),
    ("capacity_factor", "capacity factor", 1.0),
    ("end_to_end_efficiency", "end-to-end E_grid/E_sun", 1.0),
    ("eclipse_umbra_fraction", "eclipse (umbra)", 1.0),
    ("eclipse_penumbra_fraction", "penumbra", 1.0),
    ("capture_energy_weighted", "capture E_capt/E_atm", 1.0),
    ("range_mean_m", "range mean [km]", 1e-3),
    ("peak_irradiance_max_W_m2", "peak irradiance [W/m2]", 1.0),
    ("mean_intercepted_flux_W_m2", "mean flux on Rx [W/m2]", 1.0),
    ("mass_proxy_kg", "mass proxy [t]", 1e-3),
    ("mass_per_delivered_kW_kg", "mass / delivered kW [t/kW]", 1e-3),
)


def _fmt(v: Any, scale: float = 1.0) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        x = v * scale
        if x == 0.0 or not math.isfinite(x):
            return str(x)
        return f"{x:.4g}"
    return str(v)


def _fmt_exact(v: Any) -> str:
    """A configuration value as configured: floats by ``repr`` (shortest round-trip form, so
    365.25 stays 365.25 and c/2.45 GHz keeps all 17 digits); the harmonised-assumption table
    must show literally the values that were run, not a 4-significant-figure rounding."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _points_md(plan: ComparisonPlan, rows: Sequence[ComparisonRow]) -> str:
    header = ["case", "point", "D [m]", "R_rx [m]", "status", *(h for _, h, _ in _MD_COLUMNS)]
    body = []
    for r in rows:
        status = r.status if r.error is None else f"{r.status}: {r.error[:120]}"
        body.append(
            [
                r.case,
                r.point + (" (base)" if r.is_base else ""),
                _fmt(r.aperture_diameter_m),
                _fmt(r.receiver_radius_m),
                status,
                *(_fmt(getattr(r, k), sc) for k, _, sc in _MD_COLUMNS),
            ]
        )
    return (
        f"# Comparison '{plan.definition.name}': every sweep point\n\n"
        "Generated by `helioswarm compare` from the saved runs (summary.json of each point). "
        "One satellite, one receiver site, same window and time grid; P_grid at the "
        "receiver's grid output (PDF §10.1). Failed points are listed with their error.\n\n"
        + _md_table(header, body)
    )


def _baseline_md(plan: ComparisonPlan, rows: Sequence[ComparisonRow]) -> str:
    base = [r for r in rows if r.is_base]
    out = [
        f"# Comparison '{plan.definition.name}': base configurations\n",
        "Generated by `helioswarm compare` from the saved runs. Same system boundary for every "
        "case: one satellite, one receiver site, P_grid at the receiver's grid output; the "
        "shared technology blocks are identical (harmonised_assumptions.md). Mass proxy: "
        "assumed specific masses (not a mass budget; no propellant, structure or ground "
        "segment, for either architecture).\n",
    ]
    for mode in sorted({r.mode for r in base}):
        rs = [r for r in base if r.mode == mode]
        header = ["metric", *(f"{r.case} ({r.architecture})" for r in rs)]
        body = [
            ["aperture D [m]", *(_fmt(r.aperture_diameter_m) for r in rs)],
            ["receiver radius [m]", *(_fmt(r.receiver_radius_m) for r in rs)],
            ["status", *(r.status for r in rs)],
            *([h, *(_fmt(getattr(r, k), sc) for r in rs)] for k, h, sc in _MD_COLUMNS),
        ]
        out.append(f"\n## {mode}\n\n" + _md_table(header, body))
    return "\n".join(out)


def _assumptions_md(plan: ComparisonPlan, rows: Sequence[AssumptionRow]) -> str:
    labels = [c.label for c in plan.cases]
    header = ["path", "unit", "provenance", "scope", *labels]
    body = [
        [r.path, r.unit, r.provenance, r.scope, *(_fmt_exact(r.values.get(lbl)) for lbl in labels)]
        for r in rows
    ]
    return (
        f"# Comparison '{plan.definition.name}': harmonised assumptions\n\n"
        "Every resolved configuration leaf of the base configurations with its effective "
        "provenance label (schema metadata; `assumed` = scenario input, `sourced` = cited "
        "default) and its scope: `shared by all cases` (literally the same value), `shared "
        "within technology` (optical vs microwave technology cases, PDF §10.4), "
        "`orbit/geometry` and `propagation regime` (the only keys allowed to differ between "
        "the heliocentric and the GEO case of one technology: the orbit, and the beam model "
        "that its Fresnel number requires; checked when the comparison is loaded).\n\n"
        + _md_table(header, body)
    )


def _figure(plan: ComparisonPlan, rows: Sequence[ComparisonRow], path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    modes = sorted({r.mode for r in rows})
    fig = Figure(figsize=(6.0 * len(modes), 8.0))
    FigureCanvasAgg(fig)
    styles = {"heliocentric": "--", "geostationary": "-", "geocentric": ":"}
    for j, mode in enumerate(modes):
        ax_p = fig.add_subplot(2, len(modes), j + 1)
        ax_e = fig.add_subplot(2, len(modes), len(modes) + j + 1)
        rs = [r for r in rows if r.mode == mode]
        apertures = sorted({r.aperture_diameter_m for r in rs})
        colors = {d: f"C{k}" for k, d in enumerate(apertures)}
        for case in [c for c in plan.cases if c.mode == mode]:
            for d in apertures:
                pts = sorted(
                    (r for r in rs if r.case == case.label and r.aperture_diameter_m == d),
                    key=lambda r: r.receiver_radius_m,
                )
                ok = [r for r in pts if r.status == "ok"]
                lbl = f"{case.architecture}, D = {d:g} m"
                if ok:
                    ax_p.plot(
                        [r.receiver_radius_m for r in ok],
                        [(r.delivered_mean_power_W or math.nan) / 1e3 for r in ok],
                        styles[case.architecture],
                        marker="o",
                        color=colors[d],
                        label=lbl,
                    )
                    ax_e.plot(
                        [r.receiver_radius_m for r in ok],
                        [r.end_to_end_efficiency or math.nan for r in ok],
                        styles[case.architecture],
                        marker="o",
                        color=colors[d],
                    )
                bad = [r for r in pts if r.status != "ok"]
                if bad:
                    # Not-ok points (failed validity check, missing) as crosses on the top edge.
                    ax_p.plot(
                        [r.receiver_radius_m for r in bad],
                        [0.97 - 0.04 * apertures.index(d)] * len(bad),
                        "x",
                        color=colors[d],
                        transform=ax_p.get_xaxis_transform(),
                        clip_on=False,
                    )
        for ax in (ax_p, ax_e):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlabel("receiver radius [m]")
        ax_p.set_ylabel("delivered mean power P_grid [kW]")
        ax_e.set_ylabel("end-to-end efficiency E_grid / E_sun")
        ax_p.set_title(
            f"{mode}: solid = GEO, dashed = heliocentric; x (top) = point not computed",
            fontsize=8,
        )
        ax_p.legend(fontsize=6, ncol=2, loc="lower right")
    fig.suptitle(
        f"helioswarm comparison '{plan.definition.name}' (one satellite, one receiver site, "
        "same technology; from saved runs)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)


@dataclass(frozen=True)
class ComparisonOutputs:
    """Files written by :func:`write_comparison` and the rows behind them."""

    out_dir: Path
    files: tuple[Path, ...]
    rows: tuple[ComparisonRow, ...]
    assumptions: tuple[AssumptionRow, ...]


def write_comparison(plan: ComparisonPlan, runs_dir: Path, out_dir: Path) -> ComparisonOutputs:
    """Read every saved point and write the comparison tables, figure and manifest."""
    from helioswarm.manifest import code_provenance, utc_now_iso

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(plan, runs_dir)
    assumptions = harmonised_assumptions(plan)
    files: list[Path] = []

    p = out_dir / POINTS_CSV
    _write_csv(p, _ROW_FIELDS, [[getattr(r, k) for k in _ROW_FIELDS] for r in rows])
    files.append(p)
    p = out_dir / POINTS_MD
    p.write_text(_points_md(plan, rows), encoding="utf-8")
    files.append(p)
    base = [r for r in rows if r.is_base]
    p = out_dir / BASELINE_CSV
    _write_csv(p, _ROW_FIELDS, [[getattr(r, k) for k in _ROW_FIELDS] for r in base])
    files.append(p)
    p = out_dir / BASELINE_MD
    p.write_text(_baseline_md(plan, rows), encoding="utf-8")
    files.append(p)
    labels = [c.label for c in plan.cases]
    p = out_dir / ASSUMPTIONS_CSV
    _write_csv(
        p,
        ["path", "unit", "provenance", "scope", *labels],
        [
            [a.path, a.unit, a.provenance, a.scope, *(a.values.get(lbl) for lbl in labels)]
            for a in assumptions
        ],
    )
    files.append(p)
    p = out_dir / ASSUMPTIONS_MD
    p.write_text(_assumptions_md(plan, assumptions), encoding="utf-8")
    files.append(p)
    p = out_dir / FIGURE_FILE
    _figure(plan, rows, p)
    files.append(p)
    manifest = {
        "comparison_file": str(plan.path),
        "comparison_file_sha256": plan.file_sha256,
        "runs_dir": str(runs_dir),
        "generated_utc": utc_now_iso(),
        "code": code_provenance(),
        "cases": [
            {
                "label": c.label,
                "sweep": str(c.sweep.path),
                "sweep_definition_sha256": c.sweep.definition_sha256,
                "base_config": str(c.sweep.base_path),
                "base_point": c.sweep.points[c.base_index].label,
                "mode": c.mode,
                "architecture": c.architecture,
            }
            for c in plan.cases
        ],
        "points": [
            {
                "case": r.case,
                "point": r.point,
                "status": r.status,
                "summary_sha256": r.summary_sha256,
            }
            for r in rows
        ],
        "outputs": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    }
    p = out_dir / MANIFEST_FILE
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files.append(p)
    return ComparisonOutputs(
        out_dir=out_dir, files=tuple(files), rows=rows, assumptions=assumptions
    )


def rows_as_dicts(rows: Sequence[ComparisonRow]) -> list[dict[str, Any]]:
    """Plain dictionaries of rows (for tests and reports)."""
    return [asdict(r) for r in rows]


__all__ = [
    "ASSUMPTIONS_CSV",
    "ASSUMPTIONS_MD",
    "BASELINE_CSV",
    "BASELINE_MD",
    "COMPARISON_SCHEMA_VERSION",
    "FIGURE_FILE",
    "MANIFEST_FILE",
    "ORBIT_GEOMETRY_PREFIXES",
    "POINTS_CSV",
    "POINTS_MD",
    "PROPAGATION_REGIME_PREFIXES",
    "AssumptionRow",
    "ComparisonCase",
    "ComparisonCaseSpec",
    "ComparisonFile",
    "ComparisonOutputs",
    "ComparisonPlan",
    "ComparisonRow",
    "HarmonisationError",
    "architecture_of",
    "collect_rows",
    "harmonisation_violations",
    "harmonised_assumptions",
    "is_orbit_geometry_path",
    "is_propagation_regime_path",
    "load_comparison",
    "mass_proxy_value_kg",
    "may_differ",
    "point_row",
    "rows_as_dicts",
    "write_comparison",
]
