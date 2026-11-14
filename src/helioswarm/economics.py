"""Parameterised lifecycle economics: CAPEX, LCOE and NPV scenario ranges (M8a; PDF §12.5).

**Not a core physical result.** PDF §12.5: "A cost-per-kWh model is useful only after mass,
deployment and lifetime assumptions are transparent. Add it as a separate parameterized
module, not as a core physical truth. ... Compare scenarios and uncertainty; do not present a
single speculative LCOE as a robust prediction." (PDF §17 risk "Economic overclaim".) Every
input is a :class:`CostParameter` carrying its provenance (``sourced`` with a citation,
``derived`` from cited numbers with the arithmetic stated, or ``assumed``); results are
reported per scenario (``configs/economics/{optimistic,nominal,conservative}.yaml``) and as
seeded Monte Carlo intervals, never as one number.

Inputs come from a **saved run directory** (``summary.json`` + ``config_resolved.yaml``; no
re-simulation): delivered energy over the window, satellite count, collector/radiator/aperture
areas or the M7 mass proxy, receiver sites and radius, and the orbit (transport Δv).

Model (all money in one currency unit, "USD" without inflation adjustment; annual periods;
equations in ``docs/model_assumptions.md`` M8a):

* Mass ``m_i`` per satellite: the run's M7 mass proxy (``mass.source = run``) or
  ``m = m_bus + s_col A_col + s_rad A_rad + s_tx pi D^2/4`` with the configured specific
  masses (``specific_masses``; single-satellite runs have no proxy).
* Transport: ``C_tr,i = c_LEO m_i exp(Δv_i / (g0 Isp))`` (Tsiolkovsky mass ratio; the
  propellant for the in-space Δv is launched to LEO at the same price per kg; stage/tank dry
  mass ignored -- optimistic). ``Δv_i`` is explicit (e.g. LEO->GEO) or the impulsive coplanar
  patched-conic estimate :func:`heliocentric_transfer_delta_v` for satellite ``i``'s own
  orbit (ring overrides applied) and Earth-relative phase.
* CAPEX ``C0 = c_mfg sum m_i + sum C_tr,i + c_tx A_tx + c_rx A_rx`` (``A_tx`` total aperture
  area, ``A_rx = n_sites pi r_rx^2``; ``c_rx`` is the optical or the microwave receiver cost
  per m^2 by the run's transmitter mode), spent in ``T_c`` equal instalments at
  ``t = 0 .. T_c - 1``.
* Operation years ``k = 1 .. n`` at ``t = T_c - 1 + k``: energy ``E_k = E_1 P_k`` with
  ``E_1`` the run's annualised delivered energy, performance ``P_1 = 1``,
  ``P_{k+1} = P_k (1 - d)``, or, after a replacement at the end of year ``k`` (every hardware
  lifetime ``L``), ``P_{k+1} = f + (1 - f) P_k (1 - d)``; replacement cost ``f C_space`` at that
  ``t`` (``C_space`` = manufacturing + transport + transmitter); opex
  ``O = O_fix + O_sat N + o C0`` per year. Named approximation: delivered energy scales with
  the space-segment performance ``P_k``; for receiver-capacity-limited (saturated) networks the
  real loss is smaller, so the LCOE is biased high there.
* Discount factor ``D_t = (1 + r)^-t``. ``LCOE = sum_t cost_t D_t / sum_t E_t D_t`` [$/kWh];
  ``NPV(p) = sum_t (p E_t - cost_t) D_t``; equivalent annual cost
  ``CRF(r, n) PV(costs) (1 + r)^(T_c - 1)`` (PV moved to ``t = T_c - 1``, then annualised over
  the operation years) with ``CRF = r (1 + r)^n / ((1 + r)^n - 1)``. For ``T_c = 1``,
  ``d = 0`` and no replacement: ``LCOE = CRF C0 / E + O / E`` (the NREL simple LCOE used by
  NASA OTPS 2024, PDF p. 73).
* Zero discounted energy: LCOE undefined (``lcoe_usd_per_kWh = None``, flag
  ``zero_energy``). Annual energy below ``min_meaningful_energy_kWh_per_year`` (near-zero
  delivery, e.g. the MVP microwave case): LCOE also reported as undefined, flag
  ``negligible_energy``; the arithmetic ratio is kept as ``lcoe_formal_usd_per_kWh`` for
  diagnosis only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from helioswarm.constants import AU_M, GM_SUN_M3_S2

Float64Array = NDArray[np.float64]

STANDARD_GRAVITY_M_S2: Final[float] = 9.80665
"""Standard acceleration of gravity g0 [m s^-2] (exact by definition; 3rd CGPM 1901)."""

GM_EARTH_M3_S2: Final[float] = 3.9860043550702266e14
"""GM of the Earth [m^3 s^-2] (DE440 ``gm_de440.tpc`` ``BODY399_GM = 3.9860043550702266D+05``
km^3 s^-2, the kernel of the helioswarm kernel set)."""

EARTH_EQUATORIAL_RADIUS_M: Final[float] = 6_378_136.6
"""Earth equatorial radius [m] (``pck00011.tpc`` ``BODY399_RADII``)."""

JULIAN_YEAR_S: Final[float] = 365.25 * 86_400.0
J_PER_KWH: Final[float] = 3.6e6

ECONOMICS_FILE: Final[str] = "economics.json"
ECONOMICS_TABLE: Final[str] = "economics.md"
SCHEMA: Final[str] = "helioswarm-economics/1"


class EconomicsError(RuntimeError):
    """Economics configuration or input error."""


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class CostParameter(_Model):
    """A value with its provenance (``sourced``/``derived`` need ``source``)."""

    value: float
    provenance: Literal["sourced", "derived", "assumed"]
    source: str | None = None

    @model_validator(mode="after")
    def _check(self) -> CostParameter:
        if self.provenance in ("sourced", "derived") and not self.source:
            raise ValueError(f"provenance '{self.provenance}' needs a source")
        return self


class IntParameter(_Model):
    """An integer model input (years, revolutions) with its provenance, like
    :class:`CostParameter`."""

    value: int
    provenance: Literal["sourced", "derived", "assumed"]
    source: str | None = None

    @model_validator(mode="after")
    def _check(self) -> IntParameter:
        if self.provenance in ("sourced", "derived") and not self.source:
            raise ValueError(f"provenance '{self.provenance}' needs a source")
        return self


def _p(value: float, provenance: str = "assumed", source: str | None = None) -> CostParameter:
    return CostParameter.model_validate(
        {"value": value, "provenance": provenance, "source": source}, strict=False
    )


class SpecificMasses(_Model):
    """Specific masses of the economics mass model (when the run has no mass proxy)."""

    collector_areal_mass_kg_m2: CostParameter
    radiator_areal_mass_kg_m2: CostParameter
    transmitter_areal_mass_kg_m2: CostParameter
    bus_mass_kg: CostParameter


class MassConfig(_Model):
    """Where the deployed mass comes from: the run's M7 mass proxy (``run``), the specific
    masses below (``specific_masses``), or the proxy when the run has one and the specific
    masses otherwise (``auto``; single-satellite runs have no proxy)."""

    source: Literal["run", "specific_masses", "auto"]
    specific_masses: SpecificMasses | None = None

    @model_validator(mode="after")
    def _check(self) -> MassConfig:
        if self.source != "run" and self.specific_masses is None:
            raise ValueError(f"mass.source='{self.source}' needs mass.specific_masses")
        return self


class TransportConfig(_Model):
    """Transport to the operational orbit (module docstring)."""

    launch_cost_per_kg_leo_usd: CostParameter
    isp_s: CostParameter
    delta_v_model: Literal["patched_conic_heliocentric", "explicit"]
    delta_v_m_s: CostParameter | None = None
    parking_orbit_altitude_m: CostParameter = Field(default_factory=lambda: _p(400e3))
    phasing_years: IntParameter | None = None
    """Deployment time budget ``N`` [Earth orbital periods]: free phase drift after the
    transfer and the phasing revolutions (:func:`heliocentric_transfer_delta_v`)."""

    @model_validator(mode="after")
    def _check(self) -> TransportConfig:
        if (self.delta_v_model == "explicit") != (self.delta_v_m_s is not None):
            raise ValueError("transport.delta_v_m_s is required with, and only with, 'explicit'")
        if (self.delta_v_model == "patched_conic_heliocentric") != (self.phasing_years is not None):
            raise ValueError(
                "transport.phasing_years is required with, and only with, "
                "'patched_conic_heliocentric'"
            )
        if self.phasing_years is not None and not 1 <= self.phasing_years.value <= 20:
            raise ValueError("transport.phasing_years must lie in [1, 20]")
        return self


class TriangularRange(_Model):
    """Triangular distribution ``(low, mode, high)`` of one input (``low < high``)."""

    low: float
    mode: float
    high: float

    @model_validator(mode="after")
    def _check(self) -> TriangularRange:
        if not (self.low <= self.mode <= self.high and self.low < self.high):
            raise ValueError("need low <= mode <= high and low < high")
        return self


UNCERTAIN_PARAMETERS: Final[tuple[str, ...]] = (
    "discount_rate",
    "annual_degradation",
    "manufacturing_cost_per_kg_usd",
    "microwave_receiver_capex_per_m2_usd",
    "optical_receiver_capex_per_m2_usd",
    "transmitter_capex_per_m2_usd",
    "opex_fixed_per_year_usd",
    "opex_per_satellite_per_year_usd",
    "opex_fraction_of_capex_per_year",
    "replacement_fraction",
    "hardware_lifetime_years",
    "transport.launch_cost_per_kg_leo_usd",
    "transport.isp_s",
)
"""Inputs a Monte Carlo may vary (dotted paths of :class:`EconomicsConfig`)."""


class UncertaintyConfig(_Model):
    """Seeded Monte Carlo over cost inputs (independent triangular distributions).

    Each range's ``mode`` must equal the scenario value and both bounds must be admissible
    values of the parameter (checked by :class:`EconomicsConfig`)."""

    n_samples: int = Field(ge=10, le=1_000_000)
    seed: int = Field(ge=0)
    parameters: dict[str, TriangularRange] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> UncertaintyConfig:
        bad = sorted(set(self.parameters) - set(UNCERTAIN_PARAMETERS))
        if bad:
            raise ValueError(f"unknown uncertain parameters {bad}; allowed {UNCERTAIN_PARAMETERS}")
        return self


class EconomicsConfig(_Model):
    """One economic scenario (``configs/economics/*.yaml``)."""

    economics_schema_version: Literal[1]
    scenario: str = Field(min_length=1, max_length=64)
    description: str = ""
    currency: str = "USD (nominal values of the cited sources; no inflation adjustment)"
    discount_rate: CostParameter
    system_lifetime_years: IntParameter
    construction_years: IntParameter
    hardware_lifetime_years: CostParameter
    replacement_fraction: CostParameter
    annual_degradation: CostParameter
    manufacturing_cost_per_kg_usd: CostParameter
    microwave_receiver_capex_per_m2_usd: CostParameter
    """Receiver (rectenna + ground segment) CAPEX per m^2 of a ``microwave`` run."""
    optical_receiver_capex_per_m2_usd: CostParameter
    """Receiver (laser-PV array + ground segment) CAPEX per m^2 of an ``optical`` run."""
    transmitter_capex_per_m2_usd: CostParameter = Field(default_factory=lambda: _p(0.0))
    """Transmitter CAPEX per m^2 of aperture beyond its mass x manufacturing cost (default 0:
    folded into the flat manufacturing cost per kg)."""
    opex_fixed_per_year_usd: CostParameter
    opex_per_satellite_per_year_usd: CostParameter
    opex_fraction_of_capex_per_year: CostParameter
    electricity_price_usd_per_kWh: CostParameter
    min_meaningful_energy_kWh_per_year: CostParameter
    mass: MassConfig
    transport: TransportConfig
    uncertainty: UncertaintyConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> EconomicsConfig:
        issues: list[str] = []
        if not 1 <= self.system_lifetime_years.value <= 200:
            issues.append("system_lifetime_years must lie in [1, 200]")
        if not 1 <= self.construction_years.value <= 50:
            issues.append("construction_years must lie in [1, 50]")
        if self.min_meaningful_energy_kWh_per_year.value < 0.0:
            issues.append("min_meaningful_energy_kWh_per_year must be >= 0")
        if self.electricity_price_usd_per_kWh.value < 0.0:
            issues.append("electricity_price_usd_per_kWh must be >= 0")
        for name in UNCERTAIN_PARAMETERS:
            issues += _value_issues(name, _get(self, name).value)
        if self.uncertainty is not None:
            for name, rng in self.uncertainty.parameters.items():
                value = _get(self, name).value
                if not math.isclose(rng.mode, value, rel_tol=1e-12, abs_tol=1e-15):
                    issues.append(
                        f"uncertainty.{name}: mode {rng.mode} differs from the scenario value "
                        f"{value}"
                    )
                for bound in (rng.low, rng.high):
                    issues += [f"uncertainty.{name}: {m}" for m in _value_issues(name, bound)]
        if issues:
            raise ValueError("; ".join(issues))
        return self


def load_economics_config(path: str | Path) -> EconomicsConfig:
    """Read and validate an economics configuration (YAML)."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    try:
        return EconomicsConfig.model_validate(raw, strict=False)
    except ValueError as exc:
        raise EconomicsError(f"{path}: invalid economics configuration: {exc}") from exc


def _get(config: EconomicsConfig, path: str) -> CostParameter:
    """The :class:`CostParameter` at a dotted path of :data:`UNCERTAIN_PARAMETERS`."""
    node: Any = config
    for part in path.split("."):
        node = getattr(node, part)
    assert isinstance(node, CostParameter)
    return node


def _value_issues(name: str, value: float) -> list[str]:
    """Admissibility of a value of one of :data:`UNCERTAIN_PARAMETERS` (empty if valid)."""
    if name == "discount_rate":
        ok, rule = -0.5 < value < 1.0, "lie in (-0.5, 1)"
    elif name == "annual_degradation":
        ok, rule = 0.0 <= value < 1.0, "lie in [0, 1)"
    elif name == "replacement_fraction":
        ok, rule = 0.0 <= value <= 1.0, "lie in [0, 1]"
    elif name == "hardware_lifetime_years":
        ok, rule = value >= 1.0, "be >= 1"
    elif name == "transport.isp_s":
        ok, rule = value > 0.0, "be > 0"
    else:  # costs, cost fractions
        ok, rule = value >= 0.0, "be >= 0"
    return [] if ok else [f"{name} must {rule} (got {value})"]


def with_values(config: EconomicsConfig, values: Mapping[str, float]) -> EconomicsConfig:
    """Copy of ``config`` with the values of some :data:`UNCERTAIN_PARAMETERS` replaced
    (provenance kept). The copy is re-validated (so a sampled value outside the admissible
    range raises) and carries no ``uncertainty`` section (its modes refer to the original
    scenario values)."""
    top: dict[str, Any] = {}
    tr: dict[str, Any] = {}
    for path, v in values.items():
        if path not in UNCERTAIN_PARAMETERS:
            raise EconomicsError(f"cannot vary {path!r}")
        if path.startswith("transport."):
            name = path.split(".", 1)[1]
            tr[name] = getattr(config.transport, name).model_copy(update={"value": float(v)})
        else:
            top[path] = getattr(config, path).model_copy(update={"value": float(v)})
    if tr:
        top["transport"] = config.transport.model_copy(update=tr)
    top["uncertainty"] = None
    data = config.model_copy(update=top).model_dump(mode="python")
    try:
        return EconomicsConfig.model_validate(data, strict=False)
    except ValueError as exc:
        raise EconomicsError(f"invalid varied economics inputs {dict(values)}: {exc}") from exc


# ---------------------------------------------------------------------------------------------
# Closed-form building blocks
# ---------------------------------------------------------------------------------------------


def capital_recovery_factor(rate: float, years: int) -> float:
    """``CRF = r (1+r)^n / ((1+r)^n - 1)`` [1/yr]; ``1/n`` for ``r = 0`` (limit).

    NREL simple LCOE / NASA OTPS 2024 (PDF p. 74, footnote): ``i(1+i)^n / ((1+i)^n - 1)``.
    """
    if years < 1:
        raise ValueError("years must be >= 1")
    if rate == 0.0:
        return 1.0 / years
    # r / (1 - (1+r)^-n), evaluated with expm1/log1p: no cancellation for small r.
    return rate / -math.expm1(-years * math.log1p(rate))


def tsiolkovsky_mass_ratio(delta_v_m_s: float, isp_s: float) -> float:
    """Initial/final mass ``exp(Δv / (g0 Isp))`` (Tsiolkovsky rocket equation) [1]."""
    if isp_s <= 0.0 or delta_v_m_s < 0.0:
        raise ValueError("need isp_s > 0 and delta_v_m_s >= 0")
    return math.exp(delta_v_m_s / (STANDARD_GRAVITY_M_S2 * isp_s))


def escape_delta_v_m_s(v_inf_m_s: float, parking_radius_m: float, mu_m3_s2: float) -> float:
    """Impulsive Δv from a circular parking orbit to a hyperbola with excess speed ``v_inf``:
    ``sqrt(v_inf^2 + 2 mu / r) - sqrt(mu / r)`` (vis-viva at periapsis)."""
    return math.sqrt(v_inf_m_s**2 + 2.0 * mu_m3_s2 / parking_radius_m) - math.sqrt(
        mu_m3_s2 / parking_radius_m
    )


def hohmann_delta_v_m_s(r1_m: float, r2_m: float, mu_m3_s2: float) -> tuple[float, float]:
    """The two impulses of a coplanar Hohmann transfer between circular orbits [m s^-1]:
    ``dv1 = sqrt(mu/r1) |sqrt(2 r2/(r1+r2)) - 1|``, ``dv2 = sqrt(mu/r2) |1 - sqrt(2 r1/(r1+r2))|``.
    """
    s = r1_m + r2_m
    dv1 = math.sqrt(mu_m3_s2 / r1_m) * abs(math.sqrt(2.0 * r2_m / s) - 1.0)
    dv2 = math.sqrt(mu_m3_s2 / r2_m) * abs(1.0 - math.sqrt(2.0 * r1_m / s))
    return dv1, dv2


def phasing_delta_v_m_s(
    phase_rad: float, revolutions: int, radius_m: float, mu_m3_s2: float
) -> float:
    """Impulse that puts a body on a circular orbit of radius ``a`` onto the phasing ellipse
    that gains ``phase`` (positive = ahead) in ``N`` revolutions of the reference orbit, and
    (same magnitude) back onto the circle: period ``P' = P (1 - phase/(2 pi N))``,
    ``a' = a (P'/P)^(2/3)``, ``dv = |sqrt(mu (2/a - 1/a')) - sqrt(mu/a)|`` [m s^-1].
    ``phase`` is wrapped to (-pi, pi]."""
    phi = math.remainder(phase_rad, 2.0 * math.pi)
    if phi == -math.pi:
        phi = math.pi
    ratio = 1.0 - phi / (2.0 * math.pi * revolutions)
    a2 = radius_m * ratio ** (2.0 / 3.0)
    return abs(math.sqrt(mu_m3_s2 * (2.0 / radius_m - 1.0 / a2)) - math.sqrt(mu_m3_s2 / radius_m))


@dataclass(frozen=True)
class TransferDeltaV:
    """Impulsive patched-conic Δv from LEO to a slot on a heliocentric circular orbit.

    Attributes:
        method: ``phasing`` (co-orbital, ``a = 1 au``), ``hohmann`` (the slot is reached by
            free phase drift within the time budget) or ``hohmann+phasing`` (a residual phase
            is made up by a phasing ellipse).
        v_inf_m_s: Hyperbolic excess speed at Earth departure [m s^-1].
        departure_m_s: LEO -> escape with ``v_inf`` [m s^-1].
        arrival_m_s: Heliocentric insertion (Hohmann dv2 + phasing return) [m s^-1].
        total_m_s: Sum [m s^-1].
        hohmann_m_s: The two Hohmann impulses ``(dv1, dv2)`` (zero at ``a = 1 au``) [m s^-1].
        phasing_m_s: Phasing impulse ``dv_ph`` (applied twice) [m s^-1].
        drift_arc_rad: Signed Earth-relative phase reachable by free drift within the time
            budget, ``(n_a - n_E) T_w`` [rad] (``|.| >= 2 pi``: every phase).
        residual_phase_rad: Phase made up by phasing (0 when drift suffices) [rad].
        synodic_period_years: ``1 / |1 - (a / 1 au)^-1.5|`` [Earth orbital periods] (``inf``
            at ``a = 1 au``).
    """

    method: Literal["hohmann", "phasing", "hohmann+phasing"]
    v_inf_m_s: float
    departure_m_s: float
    arrival_m_s: float
    total_m_s: float
    hohmann_m_s: tuple[float, float] = (0.0, 0.0)
    phasing_m_s: float = 0.0
    drift_arc_rad: float = 0.0
    residual_phase_rad: float = 0.0
    synodic_period_years: float = math.inf


def heliocentric_transfer_delta_v(
    semimajor_axis_m: float,
    earth_relative_phase_rad: float,
    *,
    phasing_years: int,
    parking_altitude_m: float,
) -> TransferDeltaV:
    """Patched-conic impulsive Δv LEO -> a slot with Earth-relative phase ``phi`` (at the
    start of operation) on a circular ecliptic heliocentric orbit of radius ``a``.

    Validity: coplanar, Earth on a circular 1 au orbit, impulsive burns, no gravity assists.
    The slot's phase is part of every transfer (time budget ``T_w = N`` Earth orbital periods,
    ``N = phasing_years``):

    1. Hohmann transfer 1 au -> ``a`` (:func:`hohmann_delta_v_m_s`; zero at ``a = 1 au``),
       time ``t_H = pi sqrt(a_t^3 / mu)``, ``a_t = (1 au + a) / 2``. At arrival the
       Earth-relative phase is ``theta_0 = pi - n_E t_H`` (Earth moved ``n_E t_H`` while the
       craft moved ``pi``); afterwards it drifts at ``n_a - n_E``.
    2. Waiting up to ``T_w`` (before launch or after arrival; equivalent) reaches every phase
       of the arc ``theta_0 + (n_a - n_E) [0, T_w]``. If ``phi`` lies on it: pure Hohmann
       (always the case when the synodic period ``1 / |1 - (a/1 au)^-1.5|`` <= ``N``).
    3. Otherwise the residual ``psi = phi - e`` from the cheaper arc end ``e`` is made up by
       a phasing ellipse of ``N`` revolutions at radius ``a`` (:func:`phasing_delta_v_m_s`),
       whose impulse ``dv_ph`` is added to both Hohmann impulses, the first through ``v_inf``
       (the departure from Earth): ``v_inf = dv1 + dv_ph``, arrival ``dv2 + dv_ph``.

    At ``a = 1 au`` this is the co-orbital phasing transfer (``v_inf = dv_ph``, arrival
    ``dv_ph``) exactly; Δv is continuous in ``a`` and ``phi``. Step 3 for ``a != 1 au`` is a
    named approximation (an upper bound on a combined phasing transfer: impulses added in
    magnitude); it only arises where ``|1 - (a/1 au)^-1.5| < 1/N``, i.e. near 1 au, where it
    tends to the exact co-orbital case. The phasing revolutions extend the deployment time;
    revenue is not delayed (module docstring).

    Earth departure from a circular LEO of altitude ``h``: :func:`escape_delta_v_m_s` with
    ``r = R_E + h``.
    """
    a_e = AU_M
    a = semimajor_axis_m
    if a <= 0.0:
        raise ValueError("semimajor_axis_m must be > 0")
    dv1, dv2 = hohmann_delta_v_m_s(a_e, a, GM_SUN_M3_S2)
    n_e = math.sqrt(GM_SUN_M3_S2 / a_e**3)
    n_a = math.sqrt(GM_SUN_M3_S2 / a**3)
    # n_E t_H = pi (a_t / 1 au)^1.5 (exactly pi at a = 1 au, so theta_0 = 0 exactly there).
    theta0 = math.pi * (1.0 - (0.5 * (a_e + a) / a_e) ** 1.5)
    t_w = phasing_years * 2.0 * math.pi / n_e
    arc = (n_a - n_e) * t_w
    two_pi = 2.0 * math.pi
    rel = abs(1.0 - (a / a_e) ** -1.5)
    synodic = math.inf if rel == 0.0 else 1.0 / rel
    phi = earth_relative_phase_rad
    if abs(arc) >= two_pi:
        reachable = True
    else:
        x = (phi - theta0) % two_pi if arc >= 0.0 else (theta0 - phi) % two_pi
        reachable = x <= abs(arc) or x == two_pi  # (x == 2 pi: round-off of a 0 offset)
    if reachable:
        dv_ph, psi = 0.0, 0.0
    else:
        cands = [
            (phasing_delta_v_m_s(phi - e, phasing_years, a, GM_SUN_M3_S2), phi - e)
            for e in (theta0, theta0 + arc)
        ]
        dv_ph, psi = min(cands)
        psi = math.remainder(psi, two_pi)
    method: Literal["hohmann", "phasing", "hohmann+phasing"]
    if a == a_e:
        method = "phasing"
    elif dv_ph == 0.0:
        method = "hohmann"
    else:
        method = "hohmann+phasing"
    v_inf = dv1 + dv_ph
    arr = dv2 + dv_ph
    dep = escape_delta_v_m_s(v_inf, EARTH_EQUATORIAL_RADIUS_M + parking_altitude_m, GM_EARTH_M3_S2)
    return TransferDeltaV(
        method=method,
        v_inf_m_s=v_inf,
        departure_m_s=dep,
        arrival_m_s=arr,
        total_m_s=dep + arr,
        hohmann_m_s=(dv1, dv2),
        phasing_m_s=dv_ph,
        drift_arc_rad=arc,
        residual_phase_rad=psi,
        synodic_period_years=synodic,
    )


# ---------------------------------------------------------------------------------------------
# Inputs from a saved run
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunEconomicInputs:
    """Physical quantities of a saved run used by the economics (SI units).

    Attributes:
        run_dir: Run directory.
        kind: ``single`` or ``constellation``.
        project_name: Run project name.
        config_hash: Run configuration hash.
        transmitter_mode: ``optical`` or ``microwave``.
        n_satellites: Satellites.
        n_sites: Receiver sites.
        delivered_energy_J: Grid energy over the window [J].
        window_s: Window length [s].
        collector_area_m2: Total collector area [m^2].
        radiator_area_m2: Total (largest required) radiator area [m^2].
        aperture_area_m2: Total transmitter aperture area [m^2].
        receiver_area_m2: Total receiver area ``n_sites pi r^2`` [m^2].
        run_mass_proxy_kg: The run's M7 mass proxy (total), or ``None``.
        semimajor_axis_m: Heliocentric orbit semimajor axis per satellite (ring overrides
            applied) [m]; NaN for geocentric orbits (M7b GEO baseline), whose transfer is not
            the heliocentric patched conic.
        earth_relative_phase_rad: Earth-relative phase per satellite [rad], or ``None`` when it
            cannot be resolved or the orbits are geocentric (then only explicit Δv works).
        circular_ecliptic: Every satellite's orbit is heliocentric, circular and in the
            ecliptic (patched-conic validity).
        geocentric: The orbits are Earth-centred (``orbit.elements_center = EARTH``, the M7b
            GEO baseline): only an explicit Δv applies (``configs/economics/geo_nominal.yaml``).
    """

    run_dir: str
    kind: Literal["single", "constellation"]
    project_name: str
    config_hash: str
    transmitter_mode: str
    n_satellites: int
    n_sites: int
    delivered_energy_J: float
    window_s: float
    collector_area_m2: float
    radiator_area_m2: float
    aperture_area_m2: float
    receiver_area_m2: float
    run_mass_proxy_kg: float | None
    semimajor_axis_m: tuple[float, ...]
    earth_relative_phase_rad: tuple[float, ...] | None
    circular_ecliptic: bool
    geocentric: bool = False

    def __post_init__(self) -> None:
        if len(self.semimajor_axis_m) != self.n_satellites:
            raise ValueError("semimajor_axis_m needs one value per satellite")
        if (
            self.earth_relative_phase_rad is not None
            and len(self.earth_relative_phase_rad) != self.n_satellites
        ):
            raise ValueError("earth_relative_phase_rad needs one value per satellite")

    @property
    def annual_energy_kWh(self) -> float:
        """Annualised delivered energy ``E_win (Julian year / window) / 3.6e6`` [kWh/yr]."""
        return self.delivered_energy_J * (JULIAN_YEAR_S / self.window_s) / J_PER_KWH


def member_orbits(config: Any) -> list[Any]:
    """The resolved :class:`~helioswarm.config.OrbitConfig` of every satellite (ring element
    and phase overrides applied, :func:`helioswarm.constellation.constellation_members`)."""
    from helioswarm.config import RunConfig

    assert isinstance(config, RunConfig)
    if config.constellation is None:
        return [config.orbit]
    from helioswarm.constellation import constellation_members

    return [
        m.config.orbit
        for m in constellation_members(config, np.random.default_rng(config.project.seed))
    ]


def _earth_relative_phases(
    orbits: Sequence[Any], ephemeris: Any | None
) -> tuple[float, ...] | None:
    """Earth-relative ecliptic phase ``lambda_sat - lambda_Earth`` of every satellite at its
    initial epoch [rad] (None if an absolute phase cannot be resolved without ephemeris)."""
    out: list[float] = []
    for o in orbits:
        if o.phase_reference == "earth_relative":
            out.append(o.true_anomaly_rad)
            continue
        if ephemeris is None:
            return None
        from helioswarm.propagate import earth_ecliptic_longitude_rad, resolve_orbit_phase
        from helioswarm.timescale import utc_to_tdb_s

        t0 = utc_to_tdb_s(o.initial_epoch_utc, ephemeris.kernels)
        ph = resolve_orbit_phase(o, epoch_tdb_s=t0, ephemeris=ephemeris)
        out.append(ph.ecliptic_longitude_rad - earth_ecliptic_longitude_rad(t0, ephemeris))
    return tuple(out)


def run_inputs(run_dir: Path, *, ephemeris: Any | None = None) -> RunEconomicInputs:
    """Read :class:`RunEconomicInputs` from a saved run directory.

    Args:
        run_dir: Directory with ``summary.json`` and ``config_resolved.yaml``.
        ephemeris: A :class:`~helioswarm.ephemeris.SpiceEphemeris` (with ``.kernels``) to
            resolve the Earth-relative phase of ``absolute``-phase orbits; without it such runs
            get ``earth_relative_phase_rad = None``.
    """
    from helioswarm.config import load_config

    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config_resolved.yaml"
    if not summary_path.is_file() or not config_path.is_file():
        raise EconomicsError(f"{run_dir}: summary.json or config_resolved.yaml missing")
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    cfg = load_config(config_path)
    schema = str(s.get("summary_schema", ""))
    orbits = member_orbits(cfg)
    # Geocentric orbits (M7b GEO baseline; constellations are heliocentric by validation) have
    # no heliocentric semimajor axis or Earth-relative phase (the geostationary preset does not
    # even configure elements): their transport needs an explicit Δv (transport_delta_v).
    geocentric = any(o.is_geocentric for o in orbits)
    if geocentric:
        circular = False
        sma: tuple[float, ...] = (math.nan,) * len(orbits)
        phases: tuple[float, ...] | None = None
    else:
        circular = all(o.eccentricity == 0.0 and o.inclination_deg == 0.0 for o in orbits)
        sma = tuple(o.semimajor_axis_m for o in orbits)
        phases = _earth_relative_phases(orbits, ephemeris)
    sites = len(cfg.receiver.sites)
    rx_area = sites * math.pi * cfg.receiver.radius_m**2
    if schema.startswith("helioswarm-constellation-summary"):
        tot = s["totals"]
        return RunEconomicInputs(
            run_dir=str(run_dir),
            kind="constellation",
            project_name=str(s["project_name"]),
            config_hash=str(s["config_hash"]),
            transmitter_mode=str(s["transmitter_mode"]),
            n_satellites=int(s["n_satellites"]),  # == len(orbits) (checked on construction)
            n_sites=sites,
            delivered_energy_J=float(s["network"]["delivered_energy_J"]),
            window_s=float(s["time"]["window_s"]),
            collector_area_m2=float(tot["collector_area_m2"]),
            radiator_area_m2=float(tot["radiator_area_required_m2"]),
            aperture_area_m2=float(tot["transmitter_aperture_area_m2"]),
            receiver_area_m2=rx_area,
            run_mass_proxy_kg=(
                None if tot.get("mass_proxy_kg") is None else float(tot["mass_proxy_kg"])
            ),
            semimajor_axis_m=sma,
            earth_relative_phase_rad=phases,
            circular_ecliptic=circular,
            geocentric=geocentric,
        )
    if schema.startswith("helioswarm-summary"):
        d = cfg.transmitter.aperture_diameter_m
        return RunEconomicInputs(
            run_dir=str(run_dir),
            kind="single",
            project_name=str(s["project_name"]),
            config_hash=str(s["config_hash"]),
            transmitter_mode=str(s["transmitter_mode"]),
            n_satellites=1,
            n_sites=sites,
            delivered_energy_J=float(s["delivered_energy_J"]),
            window_s=float(s["time"]["window_s"]),
            collector_area_m2=cfg.collector.area_m2,
            radiator_area_m2=float(s["thermal"]["radiator_area_m2_max"]),
            aperture_area_m2=math.pi * d * d / 4.0,
            receiver_area_m2=rx_area,
            run_mass_proxy_kg=None,
            semimajor_axis_m=sma,
            earth_relative_phase_rad=phases,
            circular_ecliptic=circular,
            geocentric=geocentric,
        )
    raise EconomicsError(f"{summary_path}: unknown summary schema {schema!r}")


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CashFlows:
    """Annual cash flows (index = year ``t``, cash at the end of year ``t``; ``t = 0`` is the
    start of construction).

    Attributes:
        t_years: Years ``0 .. T_c - 1 + n``.
        capex_usd, replacement_usd, opex_usd: Costs per year.
        energy_kWh: Delivered energy per year.
        discount_factor: ``(1 + r)^-t``.
    """

    t_years: Float64Array
    capex_usd: Float64Array
    replacement_usd: Float64Array
    opex_usd: Float64Array
    energy_kWh: Float64Array
    discount_factor: Float64Array

    @property
    def cost_usd(self) -> Float64Array:
        """Total cost per year."""
        out: Float64Array = self.capex_usd + self.replacement_usd + self.opex_usd
        return out


def cash_flows(
    *,
    capex_usd: float,
    space_capex_usd: float,
    opex_per_year_usd: float,
    first_year_energy_kWh: float,
    discount_rate: float,
    lifetime_years: int,
    construction_years: int,
    hardware_lifetime_years: float,
    replacement_fraction: float,
    annual_degradation: float,
) -> CashFlows:
    """Cash flows of the module docstring (pure)."""
    tc, n = construction_years, lifetime_years
    t = np.arange(tc + n, dtype=np.float64)
    capex = np.zeros(t.size)
    capex[:tc] = capex_usd / tc
    repl = np.zeros(t.size)
    opex = np.zeros(t.size)
    energy = np.zeros(t.size)
    perf = 1.0
    life = hardware_lifetime_years
    next_replacement = life
    for k in range(1, n + 1):
        idx = tc - 1 + k
        energy[idx] = first_year_energy_kWh * perf
        opex[idx] = opex_per_year_usd
        degraded = perf * (1.0 - annual_degradation)
        if k < n and k >= next_replacement - 1e-9:
            repl[idx] = replacement_fraction * space_capex_usd
            perf = replacement_fraction + (1.0 - replacement_fraction) * degraded
            next_replacement += life
        else:
            perf = degraded
    disc = (1.0 + discount_rate) ** (-t)
    return CashFlows(
        t_years=t,
        capex_usd=capex,
        replacement_usd=repl,
        opex_usd=opex,
        energy_kWh=energy,
        discount_factor=disc,
    )


@dataclass(frozen=True)
class EconomicsResult:
    """Economics of one run under one scenario (module docstring).

    Attributes:
        scenario: Scenario name.
        mass_total_kg: Deployed mass ``sum m_i`` [kg].
        mass_source: ``run`` (M7 proxy) or ``specific_masses``.
        delta_v_m_s: Per-satellite in-space Δv [m s^-1].
        delta_v_method: ``explicit``, ``hohmann`` or ``phasing``.
        mass_ratio_mean: Mean Tsiolkovsky mass ratio [1].
        annual_energy_kWh: First-year delivered energy [kWh/yr].
        capex_breakdown_usd: ``manufacturing``, ``transport``, ``transmitter``, ``receiver``
            [USD].
        capex_usd: Total CAPEX [USD].
        opex_per_year_usd: Annual opex [USD/yr].
        pv_breakdown_usd: Present values of ``capex``, ``replacement``, ``opex`` [USD].
        pv_cost_usd: Present value of all costs [USD].
        pv_energy_kWh: Discounted energy [kWh].
        crf: Capital recovery factor over the system lifetime [1/yr].
        equivalent_annual_cost_usd: ``CRF PV(costs) (1 + r)^(T_c - 1)``: the constant payment
            over the ``n`` operation years with the same present value as all costs [USD/yr].
        lcoe_usd_per_kWh: LCOE, or ``None`` (zero or negligible energy).
        lcoe_usd_per_MWh: LCOE x 1000, or ``None``.
        lcoe_formal_usd_per_kWh: ``PV(costs) / PV(energy)`` whenever the energy is > 0, even
            when negligible (diagnostic; not a reportable LCOE), else ``None``.
        npv_usd: NPV at the configured electricity price [USD].
        electricity_price_usd_per_kWh: Price used for the NPV.
        flags: ``zero_energy``, ``negligible_energy``.
    """

    scenario: str
    mass_total_kg: float
    mass_source: str
    delta_v_m_s: tuple[float, ...]
    delta_v_method: str
    mass_ratio_mean: float
    annual_energy_kWh: float
    capex_breakdown_usd: Mapping[str, float]
    capex_usd: float
    opex_per_year_usd: float
    pv_breakdown_usd: Mapping[str, float]
    pv_cost_usd: float
    pv_energy_kWh: float
    crf: float
    equivalent_annual_cost_usd: float
    lcoe_usd_per_kWh: float | None
    lcoe_usd_per_MWh: float | None
    npv_usd: float
    electricity_price_usd_per_kWh: float
    flags: tuple[str, ...] = field(default=())
    lcoe_formal_usd_per_kWh: float | None = None


def satellite_masses_kg(inputs: RunEconomicInputs, cfg: EconomicsConfig) -> tuple[float, str]:
    """Total deployed mass [kg] and its source."""
    if cfg.mass.source == "auto" and inputs.run_mass_proxy_kg is not None:
        return inputs.run_mass_proxy_kg, "run"
    if cfg.mass.source == "run":
        if inputs.run_mass_proxy_kg is None:
            raise EconomicsError(
                f"{inputs.run_dir}: the run has no mass proxy; use mass.source='specific_masses'"
            )
        return inputs.run_mass_proxy_kg, "run"
    sm = cfg.mass.specific_masses
    assert sm is not None  # schema rule
    m = (
        inputs.n_satellites * sm.bus_mass_kg.value
        + sm.collector_areal_mass_kg_m2.value * inputs.collector_area_m2
        + sm.radiator_areal_mass_kg_m2.value * inputs.radiator_area_m2
        + sm.transmitter_areal_mass_kg_m2.value * inputs.aperture_area_m2
    )
    return float(m), "specific_masses"


def receiver_capex_per_m2(inputs: RunEconomicInputs, cfg: EconomicsConfig) -> CostParameter:
    """The receiver cost per m^2 of the run's technology (PDF §10.4: optical and microwave
    receivers are different technology cases)."""
    if inputs.transmitter_mode == "microwave":
        return cfg.microwave_receiver_capex_per_m2_usd
    if inputs.transmitter_mode == "optical":
        return cfg.optical_receiver_capex_per_m2_usd
    raise EconomicsError(f"unknown transmitter_mode {inputs.transmitter_mode!r}")


def transport_delta_v(
    inputs: RunEconomicInputs, cfg: EconomicsConfig
) -> tuple[tuple[float, ...], str]:
    """Per-satellite in-space Δv [m s^-1] and the method."""
    tr = cfg.transport
    if tr.delta_v_model == "explicit":
        assert tr.delta_v_m_s is not None
        return (tr.delta_v_m_s.value,) * inputs.n_satellites, "explicit"
    if inputs.geocentric:
        raise EconomicsError(
            "patched_conic_heliocentric is a heliocentric transfer; this run's orbit is "
            "geocentric (orbit.elements_center = EARTH, GEO baseline): use an explicit Δv "
            "(e.g. configs/economics/geo_nominal.yaml)"
        )
    if not inputs.circular_ecliptic:
        raise EconomicsError(
            "patched_conic_heliocentric needs a circular ecliptic orbit; use an explicit Δv"
        )
    if inputs.earth_relative_phase_rad is None:
        raise EconomicsError(
            "the Earth-relative phase of this run's absolute-phase orbit needs the ephemeris "
            "(kernels); none was given"
        )
    assert tr.phasing_years is not None  # schema rule
    dvs: list[float] = []
    methods: set[str] = set()
    for a_m, phi in zip(inputs.semimajor_axis_m, inputs.earth_relative_phase_rad, strict=True):
        d = heliocentric_transfer_delta_v(
            a_m,
            phi,
            phasing_years=tr.phasing_years.value,
            parking_altitude_m=tr.parking_orbit_altitude_m.value,
        )
        dvs.append(d.total_m_s)
        methods.add(d.method)
    return tuple(dvs), "+".join(sorted(methods))


def evaluate_economics(inputs: RunEconomicInputs, cfg: EconomicsConfig) -> EconomicsResult:
    """Economics of one run under one scenario (module docstring)."""
    mass, mass_source = satellite_masses_kg(inputs, cfg)
    dvs, method = transport_delta_v(inputs, cfg)
    isp = cfg.transport.isp_s.value
    ratios = np.array([tsiolkovsky_mass_ratio(dv, isp) for dv in dvs])
    m_sat = mass / inputs.n_satellites  # equal share per satellite (documented)
    transport = float(cfg.transport.launch_cost_per_kg_leo_usd.value * m_sat * np.sum(ratios))
    manufacturing = cfg.manufacturing_cost_per_kg_usd.value * mass
    transmitter = cfg.transmitter_capex_per_m2_usd.value * inputs.aperture_area_m2
    receiver = receiver_capex_per_m2(inputs, cfg).value * inputs.receiver_area_m2
    capex = manufacturing + transport + transmitter + receiver
    opex = (
        cfg.opex_fixed_per_year_usd.value
        + cfg.opex_per_satellite_per_year_usd.value * inputs.n_satellites
        + cfg.opex_fraction_of_capex_per_year.value * capex
    )
    e1 = inputs.annual_energy_kWh
    r = cfg.discount_rate.value
    n = cfg.system_lifetime_years.value
    tc = cfg.construction_years.value
    cf = cash_flows(
        capex_usd=capex,
        space_capex_usd=manufacturing + transport + transmitter,
        opex_per_year_usd=opex,
        first_year_energy_kWh=e1,
        discount_rate=r,
        lifetime_years=n,
        construction_years=tc,
        hardware_lifetime_years=cfg.hardware_lifetime_years.value,
        replacement_fraction=cfg.replacement_fraction.value,
        annual_degradation=cfg.annual_degradation.value,
    )
    d = cf.discount_factor
    pv = {
        "capex": float(np.sum(cf.capex_usd * d)),
        "replacement": float(np.sum(cf.replacement_usd * d)),
        "opex": float(np.sum(cf.opex_usd * d)),
    }
    pv_cost = sum(pv.values())
    pv_energy = float(np.sum(cf.energy_kWh * d))
    flags: list[str] = []
    formal = pv_cost / pv_energy if pv_energy > 0.0 else None
    lcoe: float | None = formal
    if pv_energy <= 0.0:
        flags.append("zero_energy")
    if e1 < cfg.min_meaningful_energy_kWh_per_year.value:
        flags.append("negligible_energy")
        lcoe = None
    price = cfg.electricity_price_usd_per_kWh.value
    npv = float(np.sum((price * cf.energy_kWh - cf.cost_usd) * d))
    crf = capital_recovery_factor(r, n)
    return EconomicsResult(
        scenario=cfg.scenario,
        mass_total_kg=mass,
        mass_source=mass_source,
        delta_v_m_s=dvs,
        delta_v_method=method,
        mass_ratio_mean=float(np.mean(ratios)),
        annual_energy_kWh=e1,
        capex_breakdown_usd={
            "manufacturing": manufacturing,
            "transport": transport,
            "transmitter": transmitter,
            "receiver": receiver,
        },
        capex_usd=capex,
        opex_per_year_usd=opex,
        pv_breakdown_usd=pv,
        pv_cost_usd=pv_cost,
        pv_energy_kWh=pv_energy,
        crf=crf,
        # PV referred to t = T_c - 1 (one year before the first operation year), then
        # annualised over the n operation years: equals LCOE x E for constant energy.
        equivalent_annual_cost_usd=crf * pv_cost * (1.0 + r) ** (tc - 1),
        lcoe_usd_per_kWh=lcoe,
        lcoe_usd_per_MWh=None if lcoe is None else 1000.0 * lcoe,
        npv_usd=npv,
        electricity_price_usd_per_kWh=price,
        flags=tuple(flags),
        lcoe_formal_usd_per_kWh=formal,
    )


@dataclass(frozen=True)
class MonteCarloResult:
    """Seeded Monte Carlo of one scenario (independent triangular inputs).

    Attributes:
        n_samples: Samples.
        seed: Seed of the Generator.
        parameters: Varied inputs and their ranges.
        lcoe_percentiles_usd_per_kWh: ``{"p05","p50","p95","mean"}`` over samples with a
            defined LCOE, or empty.
        npv_percentiles_usd: Same for the NPV.
        n_undefined_lcoe: Samples with zero energy.
        fraction_lcoe_below_price: Fraction of samples with LCOE < the electricity price.
    """

    n_samples: int
    seed: int
    parameters: Mapping[str, Mapping[str, float]]
    lcoe_percentiles_usd_per_kWh: Mapping[str, float]
    npv_percentiles_usd: Mapping[str, float]
    n_undefined_lcoe: int
    fraction_lcoe_below_price: float


def _percentiles(x: Float64Array) -> dict[str, float]:
    if x.size == 0:
        return {}
    p = np.percentile(x, [5.0, 50.0, 95.0])
    return {"p05": float(p[0]), "p50": float(p[1]), "p95": float(p[2]), "mean": float(x.mean())}


def monte_carlo(
    inputs: RunEconomicInputs, cfg: EconomicsConfig, rng: np.random.Generator
) -> MonteCarloResult:
    """Monte Carlo over ``cfg.uncertainty`` (parameters drawn in sorted-name order, one array
    per parameter). ``hardware_lifetime_years`` samples are used as real numbers (the
    replacement happens at the end of the first operation year >= the sampled lifetime)."""
    unc = cfg.uncertainty
    if unc is None:
        raise EconomicsError(f"scenario {cfg.scenario!r} has no uncertainty section")
    names = sorted(unc.parameters)
    draws = {
        k: rng.triangular(
            unc.parameters[k].low, unc.parameters[k].mode, unc.parameters[k].high, unc.n_samples
        )
        for k in names
    }
    lcoe = np.full(unc.n_samples, np.nan)
    npv = np.empty(unc.n_samples)
    for i in range(unc.n_samples):
        res = evaluate_economics(inputs, with_values(cfg, {k: draws[k][i] for k in names}))
        if res.lcoe_usd_per_kWh is not None:
            lcoe[i] = res.lcoe_usd_per_kWh
        npv[i] = res.npv_usd
    ok = np.isfinite(lcoe)
    price = cfg.electricity_price_usd_per_kWh.value
    return MonteCarloResult(
        n_samples=unc.n_samples,
        seed=unc.seed,
        parameters={k: unc.parameters[k].model_dump() for k in names},
        lcoe_percentiles_usd_per_kWh=_percentiles(lcoe[ok]),
        npv_percentiles_usd=_percentiles(npv),
        n_undefined_lcoe=int(np.sum(~ok)),
        fraction_lcoe_below_price=float(np.mean(lcoe[ok] < price)) if ok.any() else 0.0,
    )


# ---------------------------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------------------------


UncostedStatus = Literal["failed", "incomplete", "missing"]


@dataclass(frozen=True)
class UncostedPoint:
    """A sweep point without a saved run to cost. Reported in ``economics.json`` /
    ``economics.md`` (and by the CLI), never dropped silently (PDF §13.4).

    Attributes:
        point: Point label (``p<iiii>``).
        status: ``failed`` (``error.json``: the run was refused, e.g. a beam-model validity
            check), ``incomplete`` (a point directory without ``summary.json`` /
            ``config_resolved.yaml``) or ``missing`` (listed in ``sweep_definition.yaml`` but no
            directory; not run yet).
        error: ``"<error_type>: <error>"`` of a failed point, else ``None``.
    """

    point: str
    status: UncostedStatus
    error: str | None


def _sweep_point_labels(path: Path) -> list[str]:
    """Point labels of a sweep directory: its ``points/*`` directories plus every point listed
    in ``sweep_definition.yaml`` (so points that never ran are reported as missing)."""
    import yaml

    labels = {p.name for p in (path / "points").iterdir() if p.is_dir()}
    definition = path / "sweep_definition.yaml"
    if definition.is_file():
        raw = yaml.safe_load(definition.read_text(encoding="utf-8")) or {}
        for pt in raw.get("points") or []:
            if isinstance(pt, Mapping) and isinstance(pt.get("label"), str):
                labels.add(pt["label"])
    return sorted(labels)


def sweep_points(path: Path) -> tuple[list[Path], list[UncostedPoint]]:
    """The runs to cost and the sweep points that cannot be costed.

    A run directory (``summary.json``) is one run. For a sweep directory every point is
    classified with the precedence of :func:`helioswarm.sweep.collate_sweep` and
    ``helioswarm compare``: ``error.json`` -> failed; ``summary.json`` and
    ``config_resolved.yaml`` -> costed; other files -> incomplete; nothing -> missing.

    Raises:
        EconomicsError: neither a run directory nor a sweep directory with points.
    """
    if (path / "summary.json").is_file():
        return [path], []
    pts = path / "points"
    if not pts.is_dir():
        raise EconomicsError(f"{path} is neither a run directory nor a sweep directory with points")
    runs: list[Path] = []
    uncosted: list[UncostedPoint] = []
    for label in _sweep_point_labels(path):
        d = pts / label
        if (d / "error.json").is_file():
            err = json.loads((d / "error.json").read_text(encoding="utf-8"))
            msg = f"{err.get('error_type', 'error')}: {err.get('error', '')}"
            uncosted.append(UncostedPoint(label, "failed", msg))
        elif (d / "summary.json").is_file() and (d / "config_resolved.yaml").is_file():
            runs.append(d)
        elif d.is_dir() and any(d.iterdir()):
            uncosted.append(UncostedPoint(label, "incomplete", None))
        else:
            uncosted.append(UncostedPoint(label, "missing", None))
    return runs, uncosted


def run_directories(path: Path) -> list[Path]:
    """The run directory itself, or every costable ``points/*`` run of a sweep directory
    (:func:`sweep_points`; the points it cannot cost are returned there).

    Raises:
        EconomicsError: not a run/sweep directory, or a sweep without a costable point.
    """
    runs, uncosted = sweep_points(path)
    if not runs:
        counts = _status_counts(uncosted)
        raise EconomicsError(
            f"{path}: no sweep point to cost ({counts or 'no points'}; failed points are "
            "listed in their error.json)"
        )
    return runs


def _status_counts(uncosted: Sequence[UncostedPoint]) -> str:
    counts: dict[str, int] = {}
    for u in uncosted:
        counts[u.status] = counts.get(u.status, 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def _needs_ephemeris(run_dirs: Sequence[Path], configs: Sequence[EconomicsConfig]) -> bool:
    import yaml

    if all(c.transport.delta_v_model == "explicit" for c in configs):
        return False
    for d in run_dirs:
        raw = yaml.safe_load((d / "config_resolved.yaml").read_text(encoding="utf-8"))
        if (raw.get("orbit") or {}).get("elements_center", "SUN") == "EARTH":
            continue  # geocentric (M7b): no Earth-relative phase is resolved (run_inputs)
        base = (raw.get("orbit") or {}).get("phase_reference", "absolute")
        refs = [base]
        for ring in (raw.get("constellation") or {}).get("rings") or []:
            ref = ring.get("phase_reference")
            refs.append(base if ref is None else ref)  # None = the orbit's reference
        if "absolute" in refs:
            return True
    return False


def _json_value(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Mapping):
        return {str(k): _json_value(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_value(v) for v in x]
    return x


@dataclass(frozen=True)
class EconomicsOutput:
    """Result of :func:`economics_command`.

    Attributes:
        path: The written ``economics.json``.
        n_costed: Number of runs costed.
        uncosted: Sweep points that could not be costed (also in the written files).
    """

    path: Path
    n_costed: int
    uncosted: tuple[UncostedPoint, ...]


def economics_command(
    run_dir: Path,
    config_paths: Sequence[Path],
    *,
    out_dir: Path | None,
    overwrite: bool,
) -> EconomicsOutput:
    """``helioswarm economics``: evaluate every run of ``run_dir`` under every scenario and
    write ``economics.json`` and ``economics.md``. Sweep points without a saved run (failed,
    incomplete, missing) are listed in both files (``uncosted_points``), never dropped."""
    configs = [load_economics_config(p) for p in config_paths]
    runs = run_directories(run_dir)
    _, uncosted = sweep_points(run_dir)
    out = out_dir if out_dir is not None else run_dir / "economics"
    if (out / ECONOMICS_FILE).exists() and not overwrite:
        raise EconomicsError(f"{out / ECONOMICS_FILE} exists (use --overwrite)")
    out.mkdir(parents=True, exist_ok=True)

    def evaluate_all(ephemeris: Any | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d in runs:
            inp = run_inputs(d, ephemeris=ephemeris)
            for cfg in configs:
                res = evaluate_economics(inp, cfg)
                mc = (
                    monte_carlo(inp, cfg, np.random.default_rng(cfg.uncertainty.seed))
                    if cfg.uncertainty is not None
                    else None
                )
                rows.append(
                    {
                        "run_dir": str(d),
                        "inputs": {
                            **{k: v for k, v in inp.__dict__.items() if k != "run_dir"},
                            "annual_energy_kWh": inp.annual_energy_kWh,
                        },
                        "scenario": cfg.scenario,
                        "result": res.__dict__,
                        "monte_carlo": None if mc is None else mc.__dict__,
                    }
                )
        return rows

    if _needs_ephemeris(runs, configs):
        from helioswarm.ephemeris import SpiceEphemeris
        from helioswarm.kernels import KernelSet

        with KernelSet.from_manifest().loaded(verify=True) as kernels:
            rows = evaluate_all(SpiceEphemeris(kernels))
    else:
        rows = evaluate_all(None)
    doc = {
        "schema": SCHEMA,
        "disclaimer": "Parameterised scenario economics (PDF §12.5): not a core physical "
        "result and not a cost prediction. Every input carries its provenance; compare the "
        "scenario range and the Monte Carlo interval, never a single LCOE.",
        "scenarios": {c.scenario: c.model_dump(mode="json") for c in configs},
        "scenario_files": [str(p) for p in config_paths],
        "runs": rows,
        "uncosted_points": [u.__dict__ for u in uncosted],
    }
    path = out / ECONOMICS_FILE
    path.write_text(
        json.dumps(_json_value(doc), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / ECONOMICS_TABLE).write_text(economics_table(rows, uncosted), encoding="utf-8")
    return EconomicsOutput(path=path, n_costed=len(runs), uncosted=tuple(uncosted))


def _fmt(x: float | None, unit: str = "") -> str:
    if x is None:
        return "undefined"
    if x == 0.0:
        return f"0{unit}"
    return f"{x:.3g}{unit}"


def economics_table(
    rows: Sequence[Mapping[str, Any]], uncosted: Sequence[UncostedPoint] = ()
) -> str:
    """Markdown table of :func:`economics_command` rows, then the sweep points not costed."""
    lines = [
        "# Economics (parameterised scenarios; not a prediction)",
        "",
    ]
    if uncosted:
        n_runs = len({r["run_dir"] for r in rows})
        lines += [
            f"**Incomplete sweep: {len(uncosted)} of {n_runs + len(uncosted)} points not "
            f"costed** ({_status_counts(uncosted)}); listed under *Points not costed* below.",
            "",
        ]
    lines += [
        "| run | scenario | E [kWh/yr] | mass [kg] | Δv mean [km/s] | CAPEX [USD] "
        "(mfg/transport/transmitter/receiver) | opex [USD/yr] | LCOE [USD/kWh] | LCOE MC p05-p95 "
        "[USD/kWh] | NPV [USD] | flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        res = r["result"]
        cb = res["capex_breakdown_usd"]
        mc = r["monte_carlo"]
        interval = "-"
        if mc is not None and mc["lcoe_percentiles_usd_per_kWh"]:
            p = mc["lcoe_percentiles_usd_per_kWh"]
            interval = f"{_fmt(p['p05'])}-{_fmt(p['p95'])}"
        lines.append(
            f"| {Path(r['run_dir']).name} | {r['scenario']} | "
            f"{_fmt(res['annual_energy_kWh'])} | {_fmt(res['mass_total_kg'])} | "
            f"{_fmt(float(np.mean(res['delta_v_m_s'])) / 1000.0)} | {_fmt(res['capex_usd'])} "
            f"({_fmt(cb['manufacturing'])}/{_fmt(cb['transport'])}/"
            f"{_fmt(cb.get('transmitter', 0.0))}/{_fmt(cb['receiver'])}) | "
            f"{_fmt(res['opex_per_year_usd'])} | {_fmt(res['lcoe_usd_per_kWh'])} | {interval} | "
            f"{_fmt(res['npv_usd'])} | {', '.join(res['flags']) or '-'} |"
        )
    lines += [
        "",
        "LCOE = sum of discounted costs / sum of discounted energy; undefined at zero energy. "
        "Inputs and their provenance: economics.json (scenarios).",
        "",
    ]
    if uncosted:
        lines += [
            "## Points not costed",
            "",
            "No saved run to cost (economics.json `uncosted_points`); a failed point's run was "
            "refused (its error.json), it is not a zero-energy result.",
            "",
            "| point | status | error |",
            "|---|---|---|",
        ]
        for u in uncosted:
            err = "-" if u.error is None else u.error.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {u.point} | {u.status} | {err} |")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "EARTH_EQUATORIAL_RADIUS_M",
    "ECONOMICS_FILE",
    "GM_EARTH_M3_S2",
    "JULIAN_YEAR_S",
    "J_PER_KWH",
    "STANDARD_GRAVITY_M_S2",
    "UNCERTAIN_PARAMETERS",
    "CashFlows",
    "CostParameter",
    "EconomicsConfig",
    "EconomicsError",
    "EconomicsOutput",
    "EconomicsResult",
    "IntParameter",
    "MassConfig",
    "MonteCarloResult",
    "RunEconomicInputs",
    "SpecificMasses",
    "TransferDeltaV",
    "TransportConfig",
    "TriangularRange",
    "UncertaintyConfig",
    "UncostedPoint",
    "capital_recovery_factor",
    "cash_flows",
    "economics_command",
    "economics_table",
    "escape_delta_v_m_s",
    "evaluate_economics",
    "heliocentric_transfer_delta_v",
    "hohmann_delta_v_m_s",
    "load_economics_config",
    "member_orbits",
    "monte_carlo",
    "phasing_delta_v_m_s",
    "receiver_capex_per_m2",
    "run_directories",
    "run_inputs",
    "satellite_masses_kg",
    "sweep_points",
    "transport_delta_v",
    "tsiolkovsky_mass_ratio",
    "with_values",
]
