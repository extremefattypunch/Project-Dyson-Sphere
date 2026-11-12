"""Thermal feasibility: PV-array node, power-electronics/transmitter radiator, energy ledger.

PDF §8.3 requires every radiating surface to have a defined geometry, absorbed solar power,
exported electrical power, transmitter output and waste heat to be kept in one energy balance,
reflected sunlight not to be counted as waste heat, and a bracketed root solver that reports
whether a solution exists below the allowed temperature. Two explicit steady-state nodes are
modelled; they exchange no heat with each other (conduction and mutual radiation between them
are neglected; docs/model_assumptions.md, M4).

(a) PV array node (:func:`solve_pv_temperature`)
    Geometry: a thin, isothermal flat plate of area ``A`` (``collector.area_m2``) whose front
    face (cells) faces the Sun. Both faces radiate to deep space, with IR emissivities
    ``eps_f`` (``thermal.pv_front_emissivity``) and ``eps_b`` (``thermal.pv_back_emissivity``)
    and view factor ``F`` (``thermal.pv_view_factor``) from each face to deep space at
    ``T_space`` (``thermal.space_temperature_K``); the rest ``1 - F`` of each face's
    hemisphere sees surfaces assumed to be at the plate temperature (no net exchange). Steady
    state::

        P_abs - P_PV(T) = (eps_f + eps_b) sigma A F (T^4 - T_space^4),   P_PV = P_abs eta_PV(T)

    solved for ``T`` with ``scipy.optimize.brentq`` on a bracket derived in closed form. With
    ``G = (eps_f + eps_b) sigma A F``, ``f(T) = P_abs (1 - eta_PV(T)) - G (T^4 - T_space^4)``
    and ``T_eq(eta) = ((1 - eta) P_abs / G + T_space^4)^(1/4)`` (radiative equilibrium at a
    constant efficiency; ``T_eq(0)`` is the non-converting plate):

    1. ``T_0 = T_eq(eta_min)``. No root lies outside ``[T_space, T_0]``: ``f > 0`` below
       ``T_space`` and ``f(T) <= P_abs (1 - eta_min) - G (T^4 - T_space^4) < 0`` above ``T_0``.
    2. The clipped linear law is monotone, so on ``[T_space, T_0]`` it lies between
       ``eta_lo = min(eta(T_space), eta(T_0))`` and ``eta_hi = max(eta(T_space), eta(T_0))``.
    3. Bracket ``T_lo = T_eq(eta_hi)``, ``T_hi = T_eq(eta_lo)``: ``f(T_lo) = P_abs (eta_hi -
       eta(T_lo)) >= 0 >= P_abs (eta_lo - eta(T_hi)) = f(T_hi)``, so a root always exists in
       it, and by the same bounds every root lies in it.

    These statements hold in exact arithmetic. In float64, the closed-form end points carry a
    few ulp of rounding, and for tiny ``P_abs`` (e.g. a sliver of Sun at penumbra entry) the
    bracket collapses to one or two floats next to ``T_space``. Therefore: ``T^4 - T_space^4``
    is evaluated in the factored form ``(T - T_s)(T + T_s)(T^2 + T_s^2)`` (no cancellation,
    :func:`fourth_power_difference_K4`); the residual criterion includes the float64
    representation floor ``(P_abs |eta_ref gamma_T| + 4 G T^3) spacing(T)``
    (:func:`pv_balance_floor_W`), below which no float64 ``T`` can go; an end point whose
    sign rounding flipped is used as the start of a polish that walks float64 neighbours to the
    sign change (at most ``PV_POLISH_MAX_ULPS``); and a brentq result above the tolerance is
    polished the same way. Only if all of this fails is the sample left NaN / not converged.

    The root is unique when ``f`` is strictly decreasing on ``[T_lo, T_hi]``: always for
    ``gamma_T >= 0``, and for ``gamma_T < 0`` whenever ``P_abs eta_ref |gamma_T| < 4 G T_lo^3``
    (``f' = -P_abs eta' - 4 G T^3`` with ``eta' = eta_ref gamma_T`` or 0 where clipped); this
    sufficient condition is reported per sample (``unique_root``). A sample is ``converged``
    only when the balance residual ``|f(T)|`` is at most ``PV_BALANCE_RTOL * P_abs`` plus the
    representation floor. The PV feasibility flag compares ``T`` with
    ``collector.max_operating_temperature_K``.

(b) Power-electronics / transmitter radiator node (:func:`radiator_thermal`)
    Waste heat, exactly the on-board losses of the PDF §10.1 chain (each counted once, owned by
    :mod:`helioswarm.power_chain`) plus a configured extra load::

        Q_waste = P_PV (1 - eta_pc) + P_dc_to_tx (1 - eta_e2b) + P_beam (1 - eta_optics_or_rf)
                  + Q_other

    Departure from the PDF §8.3 formula, stated explicitly: PDF §8.3 lumps the PV heat into
    one radiator load, ``Q_waste = Q_PV + Q_transmitter + Q_power_electronics + Q_other``. Here
    ``Q_PV = P_abs - P_PV`` is radiated by the PV plate itself (node (a), which sets the PV
    temperature), and this radiator carries only the on-board chain losses (``Q_transmitter +
    Q_power_electronics``, split into the three §10.1 stages) plus ``Q_other``. Nothing is
    dropped: the whole-spacecraft ledger identity checks ``incident + Q_other = reflected +
    Q_PV,radiated + Q_rad,rejected + P_aperture``.

    Geometry: a thin flat plate of physical area ``A_rad`` radiating from ``n_faces =
    thermal.radiating_sides`` faces (1: back face insulated; 2: both faces exposed), each with
    view factor ``F_rad = thermal.radiator_view_factor`` to deep space and no view of the Sun
    or of the PV array (it absorbs no sunlight); emissivity ``eps = thermal.emissivity``::

        Q_rad = eps sigma A_rad n_faces F_rad (T^4 - T_space^4)            (PDF §8.3)
        A_req = Q_waste / (eps sigma n_faces F_rad (T_lim^4 - T_space^4))  (sized at the limit)
        T_eq  = (Q_waste / (eps sigma A_rad n_faces F_rad) + T_space^4)^(1/4)  (given A_rad)

    With ``thermal.radiator_area_m2`` unset the radiator is sized at the temperature limit
    (``equilibrium_temperature_K = T_lim``, radiator margin 0); with an area the equilibrium
    temperature is solved analytically and the radiator feasibility flag is ``T_eq <= T_lim``
    with radiator margin ``T_lim - T_eq`` and area margin ``A_rad - A_req``. An optional areal
    mass gives the radiator mass proxy. The PDF-level ``ThermalResult.margin_K`` is the binding
    margin ``min(pv_margin_K, radiator_margin_K)``, so ``feasible == pv_converged & (margin_K
    >= 0)``.

Energy ledger (:func:`energy_ledger`)
    ``incident = reflected + absorbed``; ``absorbed = P_PV + Q_PV,radiated``;
    ``P_PV = P_dc_to_tx + loss_pc``; ``P_dc_to_tx = P_beam + loss_e2b``;
    ``P_beam = P_aperture + loss_optics``; ``Q_waste = loss_pc + loss_e2b + loss_optics +
    Q_other``; ``Q_waste = Q_rad,rejected``; and for the whole spacecraft ``incident + Q_other =
    reflected + Q_PV,radiated + Q_rad,rejected + P_aperture``. Each residual is reported and
    checked against ``LEDGER_RTOL`` relative to ``incident + Q_other``, plus the float64
    representation floor of the solved temperatures in that identity (:class:`EnergyLedger`).
    Only the PV-node identity and the collector/inventory sums are independent checks; the
    chain identities hold by construction and the radiator balance is a round trip.

Neglected (documented in docs/model_assumptions.md, M4): Earth/planet albedo and IR on the
collector (heliocentric distances), conduction and radiation between the nodes, transient
(thermal-mass) response (eclipse epochs are evaluated in steady state), temperature gradients
through and across the plate, sunlight absorbed by the back face, and temperature dependence of
the emissivities.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from helioswarm.config import RunConfig
from helioswarm.constants import STEFAN_BOLTZMANN_W_M2_K4
from helioswarm.power_chain import (
    OnboardEfficiencies,
    OnboardPowerChain,
    onboard_efficiencies_from_config,
    onboard_power_chain,
)
from helioswarm.solar import (
    CollectorModel,
    PvEfficiencyLaw,
    SolarPowerResult,
    absorbed_solar_power_W,
    collector_model_from_config,
    incidence_cosine,
    solar_power,
)
from helioswarm.visibility import IlluminationResult

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SIGMA: Final[float] = STEFAN_BOLTZMANN_W_M2_K4

PV_BALANCE_RTOL: Final[float] = 1.0e-12
"""Accepted PV-node balance residual ``|f(T)|`` relative to ``P_abs``, on top of the float64
representation floor (:func:`pv_balance_floor_W`). Reached with margin: the root is located to a
few ulp of ``T`` (brentq, then a float-neighbour polish), i.e. a relative power error of ~1e-14
or less whenever ``P_abs`` dominates ``G T_space^4``."""

PV_SOLVER_XTOL_ULPS: Final[float] = 4.0
"""Absolute ``brentq`` temperature tolerance, in units of the float64 spacing at the lower
bracket end (``xtol = 4 spacing(T_lo)``): a fixed kelvin tolerance would be hundreds of ulp near
``T_space`` (e.g. 1e-13 K is ~225 ulp of 2.7 K) and too coarse for tiny ``P_abs``."""

PV_SOLVER_RTOL: Final[float] = 4.0 * sys.float_info.epsilon
"""Relative temperature tolerance passed to ``brentq`` (scipy's minimum, ``4 eps``)."""

PV_SOLVER_MAXITER: Final[int] = 200
"""Iteration cap of ``brentq`` (it needs ~10-60 on these smooth functions)."""

PV_POLISH_MAX_ULPS: Final[int] = 64
"""Largest number of float64 steps the post-solve polish walks from the brentq/end-point
estimate towards the sign change of the balance (brentq leaves at most ~12 ulp)."""

LEDGER_RTOL: Final[float] = 1.0e-12
"""Energy-ledger closure tolerance, relative to ``incident + Q_other`` per sample."""

RadiatorMode = Literal["sized_at_limit", "fixed_area"]
"""``sized_at_limit``: no radiator area configured, required area at ``T_lim`` reported;
``fixed_area``: the configured area's equilibrium temperature is solved."""


def _frozen(values: ArrayLike, dtype: type = np.float64) -> NDArray[np.generic]:
    arr: NDArray[np.generic] = np.array(values, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def _freeze_1d_fields(obj: object, shape: tuple[int, ...]) -> None:
    for name, value in list(vars(obj).items()):
        if isinstance(value, np.ndarray):
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            object.__setattr__(obj, name, _frozen(value, value.dtype.type))


def fourth_power_difference_K4(temperature_K: ArrayLike, sink_K: float) -> Float64Array:
    """``T^4 - T_s^4`` evaluated as ``(T - T_s)(T + T_s)(T^2 + T_s^2)`` [K^4].

    Algebraically identical to the Stefan-Boltzmann difference; the factored form has no
    cancellation when ``T`` is close to ``T_s`` (``T - T_s`` is exact there by Sterbenz's lemma),
    so the result keeps full relative accuracy down to ``T = T_s`` (where it is exactly 0).
    """
    t = np.asarray(temperature_K, dtype=np.float64)
    return np.asarray((t - sink_K) * (t + sink_K) * (t * t + sink_K * sink_K), dtype=np.float64)


def _fourth_power_difference_scalar(t: float, ts: float) -> float:
    return (t - ts) * (t + ts) * (t * t + ts * ts)


def _check_fraction(name: str, value: float, *, allow_zero: bool = False) -> None:
    ok = (0.0 <= value <= 1.0) if allow_zero else (0.0 < value <= 1.0)
    if not (math.isfinite(value) and ok):
        raise ValueError(f"{name}={value} must lie in {'[0, 1]' if allow_zero else '(0, 1]'}")


# ---------------------------------------------------------------------------------------------
# (a) PV array node
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PvThermalNode:
    """Radiative properties of the flat PV array plate (module docstring, node (a)).

    Attributes:
        front_emissivity: IR emissivity of the Sun-facing face ``eps_f`` [1].
        back_emissivity: IR emissivity of the back face ``eps_b`` [1].
        view_factor: View factor ``F`` from each face to deep space [1].
        space_temperature_K: Radiative sink temperature ``T_space`` [K].
    """

    front_emissivity: float
    back_emissivity: float
    view_factor: float
    space_temperature_K: float

    def __post_init__(self) -> None:
        _check_fraction("front_emissivity", self.front_emissivity)
        _check_fraction("back_emissivity", self.back_emissivity)
        _check_fraction("view_factor", self.view_factor)
        if not (math.isfinite(self.space_temperature_K) and self.space_temperature_K > 0.0):
            raise ValueError("space_temperature_K must be > 0")

    def conductance_W_K4(self, area_m2: float) -> float:
        """Radiative conductance ``G = (eps_f + eps_b) sigma A F`` [W K^-4]."""
        return (self.front_emissivity + self.back_emissivity) * SIGMA * area_m2 * self.view_factor

    def radiated_W(self, temperature_K: ArrayLike, area_m2: float) -> Float64Array:
        """Net power radiated to deep space ``G (T^4 - T_space^4)`` [W] (Stefan-Boltzmann).

        ``T^4 - T_space^4`` is evaluated without cancellation
        (:func:`fourth_power_difference_K4`).
        """
        return np.asarray(
            self.conductance_W_K4(area_m2)
            * fourth_power_difference_K4(temperature_K, self.space_temperature_K),
            dtype=np.float64,
        )


def pv_thermal_node_from_config(config: RunConfig) -> PvThermalNode:
    """:class:`PvThermalNode` from the ``thermal`` section."""
    th = config.thermal
    return PvThermalNode(
        front_emissivity=th.pv_front_emissivity,
        back_emissivity=th.pv_back_emissivity,
        view_factor=th.pv_view_factor,
        space_temperature_K=th.space_temperature_K,
    )


def pv_radiative_equilibrium_temperature_K(
    absorbed_W: ArrayLike,
    area_m2: float,
    node: PvThermalNode,
    *,
    efficiency: ArrayLike = 0.0,
) -> Float64Array:
    """Closed-form plate temperature at a *constant* efficiency ``eta`` [K].

    ``T = ((1 - eta) P_abs / ((eps_f + eps_b) sigma A F) + T_space^4)^(1/4)`` (Stefan-Boltzmann
    radiative equilibrium of a two-faced flat plate). With ``eta = 0`` this is the
    non-converting plate; it also gives the solver bracket (module docstring).

    Args:
        absorbed_W: ``P_abs`` [W], >= 0.
        area_m2: Plate area ``A`` [m^2].
        node: PV radiative node.
        efficiency: Constant ``eta`` in [0, 1], broadcastable to ``absorbed_W``.
    """
    eta = np.asarray(efficiency, dtype=np.float64)
    if not np.all(np.isfinite(eta)) or np.any((eta < 0.0) | (eta > 1.0)):
        raise ValueError("efficiency must lie in [0, 1]")
    p = np.asarray(absorbed_W, dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("absorbed_W must be finite and >= 0")
    g = node.conductance_W_K4(area_m2)
    return np.asarray(((1.0 - eta) * p / g + node.space_temperature_K**4) ** 0.25, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class PvTemperatureSolution:
    """Self-consistent PV plate temperature (node (a)), vectorised.

    Attributes:
        absorbed_W: ``P_abs`` [W].
        temperature_K: Solved plate temperature ``T`` [K] (NaN where no root was found).
        efficiency: ``eta_PV(T)`` [1].
        electric_W: ``P_PV = P_abs eta_PV(T)`` [W].
        radiated_W: ``G (T^4 - T_space^4)`` [W].
        residual_W: Balance residual ``P_abs - P_PV - radiated`` [W].
        balance_floor_W: float64 representation floor of the residual at ``T``
            (:func:`pv_balance_floor_W`) [W].
        converged: ``|residual| <= PV_BALANCE_RTOL * P_abs + balance_floor_W`` (and a root was
            found).
        unique_root: Sufficient condition for a unique root holds (module docstring).
        bracket_low_K: ``T_lo`` [K].
        bracket_high_K: ``T_hi`` [K].
        iterations: brentq iterations (0 when an end point or ``P_abs = 0`` was exact).
    """

    absorbed_W: Float64Array
    temperature_K: Float64Array
    efficiency: Float64Array
    electric_W: Float64Array
    radiated_W: Float64Array
    residual_W: Float64Array
    balance_floor_W: Float64Array
    converged: BoolArray
    unique_root: BoolArray
    bracket_low_K: Float64Array
    bracket_high_K: Float64Array
    iterations: NDArray[np.int64]

    def __post_init__(self) -> None:
        _freeze_1d_fields(self, np.shape(self.absorbed_W))


def _eta_scalar(t: float, law: PvEfficiencyLaw) -> float:
    dt = t - law.reference_temperature_K
    raw = law.reference * (1.0 + law.temperature_coefficient_per_K * dt)
    return min(max(raw, law.minimum), law.maximum)


def pv_balance_floor_W(
    temperature_K: ArrayLike,
    absorbed_W: ArrayLike,
    law: PvEfficiencyLaw,
    conductance_W_K4: float,
) -> Float64Array:
    """float64 representation floor of the PV-node balance residual at ``T`` [W].

    ``floor = (P_abs |eta_ref gamma_T| + 4 G T^3) * spacing(T)``: an upper bound of ``|f'(T)|``
    times the gap between ``T`` and the next float64. No float64 temperature can make
    ``|f(T)|`` smaller than about half this value in general, so a residual criterion purely
    relative to ``P_abs`` is unreachable once ``P_abs`` is comparable to ``G T_space^4 eps`` (MVP
    plate: floor ~4e-17 W near ``T_space``, i.e. ``P_abs`` below ~1e-4 W, which the M3
    penumbra produces at shadow entry and exit). Elsewhere it is ~1e-15 of ``P_abs`` or less.

    Args:
        temperature_K: Plate temperature ``T`` [K].
        absorbed_W: ``P_abs`` [W], broadcastable to ``temperature_K``.
        law: PV efficiency law (its unclipped slope bounds ``|d eta / dT|``).
        conductance_W_K4: ``G = (eps_f + eps_b) sigma A F`` [W K^-4].
    """
    t = np.asarray(temperature_K, dtype=np.float64)
    p = np.asarray(absorbed_W, dtype=np.float64)
    slope = abs(law.reference * law.temperature_coefficient_per_K)
    return np.asarray(
        (p * slope + 4.0 * conductance_W_K4 * t**3) * np.spacing(np.abs(t)), dtype=np.float64
    )


def _polish_root(balance: Callable[[float], float], t: float) -> float:
    """Best float64 near ``t``: walk towards the sign change of ``balance``.

    Steps one ulp at a time (at most ``PV_POLISH_MAX_ULPS``) in the direction in which the
    balance changes sign for a decreasing ``f`` (up when ``f > 0``), stops at the first sign
    change, and returns the visited temperature with the smallest ``|f|``.
    """
    f = balance(t)
    best_t, best_abs = t, abs(f)
    if f == 0.0:
        return t
    direction = math.inf if f > 0.0 else -math.inf
    cur = t
    for _ in range(PV_POLISH_MAX_ULPS):
        cur = math.nextafter(cur, direction)
        f_cur = balance(cur)
        if abs(f_cur) < best_abs:
            best_t, best_abs = cur, abs(f_cur)
        if f_cur == 0.0 or (f_cur > 0.0) != (f > 0.0):
            break
    return best_t


def solve_pv_temperature(
    absorbed_W: ArrayLike, collector: CollectorModel, node: PvThermalNode
) -> PvTemperatureSolution:
    """Solve ``P_abs - P_abs eta_PV(T) = (eps_f + eps_b) sigma A F (T^4 - T_space^4)`` for ``T``.

    Bracketed ``scipy.optimize.brentq`` on ``[T_lo, T_hi]`` (module docstring), one solve per
    distinct ``P_abs`` value. ``P_abs = 0`` gives ``T = T_space`` exactly.

    Args:
        absorbed_W: Absorbed solar power ``P_abs`` [W], shape (N,), finite, >= 0.
        collector: Collector (area, efficiency law).
        node: PV radiative node.

    Returns:
        :class:`PvTemperatureSolution`.
    """
    p_all = np.atleast_1d(np.asarray(absorbed_W, dtype=np.float64))
    if p_all.ndim != 1:
        raise ValueError("absorbed_W must be 1-D")
    if not np.all(np.isfinite(p_all)) or np.any(p_all < 0.0):
        raise ValueError("absorbed_W must be finite and >= 0")
    law = collector.efficiency_law
    g = node.conductance_W_K4(collector.area_m2)
    uniq, inverse = np.unique(p_all, return_inverse=True)
    n = uniq.size
    temp = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    iters = np.zeros(n, dtype=np.int64)
    unique = np.ones(n, dtype=np.bool_)
    t_0 = pv_radiative_equilibrium_temperature_K(
        uniq, collector.area_m2, node, efficiency=law.minimum
    )
    eta_space = law.efficiency(node.space_temperature_K)
    eta_t0 = law.efficiency(t_0)
    lo_all = pv_radiative_equilibrium_temperature_K(
        uniq, collector.area_m2, node, efficiency=np.maximum(eta_space, eta_t0)
    )
    hi_all = pv_radiative_equilibrium_temperature_K(
        uniq, collector.area_m2, node, efficiency=np.minimum(eta_space, eta_t0)
    )
    slope = law.reference * law.temperature_coefficient_per_K  # d eta / dT where unclipped
    ts = node.space_temperature_K

    abs_slope = abs(slope)

    def tolerance(t: float, p: float) -> float:
        # Scalar form of pv_balance_floor_W (math.ulp == np.spacing for T > 0).
        floor = (p * abs_slope + 4.0 * g * t**3) * math.ulp(t)
        return PV_BALANCE_RTOL * p + floor

    for k in range(n):
        p = float(uniq[k])
        lo, hi = float(lo_all[k]), float(hi_all[k])

        def balance(t: float, p: float = p) -> float:
            return p * (1.0 - _eta_scalar(t, law)) - g * _fourth_power_difference_scalar(t, ts)

        if slope < 0.0:
            unique[k] = p * (-slope) < 4.0 * g * lo**3
        if p == 0.0:
            temp[k] = ts
            resid[k] = 0.0
            continue
        f_lo, f_hi = balance(lo), balance(hi)
        root: float
        if abs(f_lo) <= tolerance(lo, p):
            root = lo
        elif abs(f_hi) <= tolerance(hi, p):
            root = hi
        elif f_lo > 0.0 > f_hi:
            root, info = brentq(
                balance,
                lo,
                hi,
                xtol=PV_SOLVER_XTOL_ULPS * math.ulp(lo),
                rtol=PV_SOLVER_RTOL,
                maxiter=PV_SOLVER_MAXITER,
                full_output=True,
                disp=False,
            )
            iters[k] = int(info.iterations)
            if not info.converged:
                continue  # reported as NaN / not converged, never a midpoint
        else:
            # Exact arithmetic gives f(T_lo) >= 0 >= f(T_hi); this branch is reached only when
            # rounding of the closed-form end points (a few ulp; the bracket can collapse to a
            # single float for tiny P_abs) flips a sign. Start from the better end point.
            root = lo if abs(f_lo) <= abs(f_hi) else hi
        if abs(balance(root)) > tolerance(root, p):
            root = _polish_root(balance, root)
        f_root = balance(root)
        if abs(f_root) > tolerance(root, p):
            continue  # genuinely no root found: NaN, reported (evaluate_power_thermal raises)
        temp[k] = root
        resid[k] = f_root
    temperature = temp[inverse]
    eta = np.where(np.isfinite(temperature), law.efficiency(np.nan_to_num(temperature)), np.nan)
    electric = p_all * eta
    radiated = node.radiated_W(temperature, collector.area_m2)
    residual = resid[inverse]
    floor = pv_balance_floor_W(temperature, p_all, law, g)
    converged = np.isfinite(residual) & (np.abs(residual) <= PV_BALANCE_RTOL * p_all + floor)
    return PvTemperatureSolution(
        absorbed_W=p_all,
        temperature_K=temperature,
        efficiency=eta,
        electric_W=electric,
        radiated_W=radiated,
        residual_W=residual,
        balance_floor_W=floor,
        converged=converged,
        unique_root=unique[inverse],
        bracket_low_K=lo_all[inverse],
        bracket_high_K=hi_all[inverse],
        iterations=iters[inverse],
    )


# ---------------------------------------------------------------------------------------------
# (b) Power-electronics / transmitter radiator node
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RadiatorModel:
    """Flat-plate radiator of the power electronics and transmitter (module docstring, (b)).

    Attributes:
        emissivity: IR emissivity ``eps`` of the radiating faces [1].
        temperature_limit_K: Maximum allowed radiator temperature ``T_lim`` [K].
        radiating_sides: Faces of the plate that see space, ``n_faces`` in {1, 2}.
        view_factor: View factor ``F_rad`` from each radiating face to deep space [1].
        space_temperature_K: Sink temperature ``T_space`` [K].
        physical_area_m2: Physical plate area ``A_rad`` [m^2], or ``None`` (size at the limit).
        specific_mass_kg_m2: Areal mass for the mass proxy [kg m^-2], or ``None``.
        other_heat_load_W: Extra heat ``Q_other`` rejected by this radiator [W].
    """

    emissivity: float
    temperature_limit_K: float
    radiating_sides: int
    view_factor: float
    space_temperature_K: float
    physical_area_m2: float | None = None
    specific_mass_kg_m2: float | None = None
    other_heat_load_W: float = 0.0

    def __post_init__(self) -> None:
        _check_fraction("emissivity", self.emissivity)
        _check_fraction("view_factor", self.view_factor)
        if self.radiating_sides not in (1, 2):
            raise ValueError("radiating_sides must be 1 or 2 (faces of a thin flat plate)")
        if not self.temperature_limit_K > self.space_temperature_K > 0.0:
            raise ValueError("need temperature_limit_K > space_temperature_K > 0")
        for name in ("physical_area_m2", "specific_mass_kg_m2"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be > 0 or None")
        if not (math.isfinite(self.other_heat_load_W) and self.other_heat_load_W >= 0.0):
            raise ValueError("other_heat_load_W must be >= 0")

    @property
    def mode(self) -> RadiatorMode:
        """``fixed_area`` when a physical area is configured, else ``sized_at_limit``."""
        return "sized_at_limit" if self.physical_area_m2 is None else "fixed_area"

    @property
    def effective_area_factor(self) -> float:
        """``A_eff / A_physical = n_faces F_rad`` (PDF §8.3 emitting_side_factor * view_factor)."""
        return self.radiating_sides * self.view_factor


def radiator_model_from_config(config: RunConfig) -> RadiatorModel:
    """:class:`RadiatorModel` from the ``thermal`` section."""
    th = config.thermal
    return RadiatorModel(
        emissivity=th.emissivity,
        temperature_limit_K=th.radiator_temperature_limit_K,
        radiating_sides=th.radiating_sides,
        view_factor=th.radiator_view_factor,
        space_temperature_K=th.space_temperature_K,
        physical_area_m2=th.radiator_area_m2,
        specific_mass_kg_m2=th.radiator_specific_mass_kg_m2,
        other_heat_load_W=th.other_heat_load_W,
    )


def radiator_rejected_heat_W(
    temperature_K: ArrayLike, area_m2: ArrayLike, radiator: RadiatorModel
) -> Float64Array:
    """``Q_rad = eps sigma A_rad n_faces F_rad (T^4 - T_space^4)`` [W] (PDF §8.3).

    ``T^4 - T_space^4`` is evaluated without cancellation (:func:`fourth_power_difference_K4`).
    """
    a = np.asarray(area_m2, dtype=np.float64)
    return np.asarray(
        radiator.emissivity
        * SIGMA
        * a
        * radiator.effective_area_factor
        * fourth_power_difference_K4(temperature_K, radiator.space_temperature_K),
        dtype=np.float64,
    )


def radiator_balance_floor_W(
    temperature_K: ArrayLike, area_m2: ArrayLike, radiator: RadiatorModel
) -> Float64Array:
    """float64 representation floor of ``Q_rad`` at a solved radiator temperature [W].

    ``|dQ_rad/dT| spacing(T) = 4 eps sigma A_rad n_faces F_rad T^3 spacing(T)``: the change of
    the rejected heat over one float64 step of ``T``. The analytic equilibrium temperature can
    only be represented to this level, so ``Q_waste - Q_rad(T_eq)`` cannot be checked more
    tightly (relevant only when ``Q_waste`` is comparable to ``eps sigma A_eff T_space^4 eps``).
    """
    t = np.asarray(temperature_K, dtype=np.float64)
    a = np.asarray(area_m2, dtype=np.float64)
    g = radiator.emissivity * SIGMA * a * radiator.effective_area_factor
    return np.asarray(4.0 * g * t**3 * np.spacing(np.abs(t)), dtype=np.float64)


def radiator_required_area_m2(waste_heat_W: ArrayLike, radiator: RadiatorModel) -> Float64Array:
    """Physical plate area that rejects ``Q_waste`` at ``T_lim`` [m^2].

    ``A_req = Q_waste / (eps sigma n_faces F_rad (T_lim^4 - T_space^4))`` (PDF §8.3 solved for
    the area): inversely proportional to ``eps`` and to ``T_lim^4 - T_space^4``.
    """
    q = np.asarray(waste_heat_W, dtype=np.float64)
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("waste_heat_W must be finite and >= 0")
    denom = (
        radiator.emissivity
        * SIGMA
        * radiator.effective_area_factor
        * float(
            fourth_power_difference_K4(radiator.temperature_limit_K, radiator.space_temperature_K)
        )
    )
    return np.asarray(q / denom, dtype=np.float64)


def radiator_equilibrium_temperature_K(
    waste_heat_W: ArrayLike, area_m2: float, radiator: RadiatorModel
) -> Float64Array:
    """Steady-state temperature of a radiator of physical area ``A_rad`` rejecting ``Q`` [K].

    ``T = (Q / (eps sigma A_rad n_faces F_rad) + T_space^4)^(1/4)`` (PDF §8.3 solved for T).
    """
    q = np.asarray(waste_heat_W, dtype=np.float64)
    if not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("waste_heat_W must be finite and >= 0")
    if not (math.isfinite(area_m2) and area_m2 > 0.0):
        raise ValueError("area_m2 must be > 0")
    a_eff = area_m2 * radiator.effective_area_factor
    return np.asarray(
        (q / (radiator.emissivity * SIGMA * a_eff) + radiator.space_temperature_K**4) ** 0.25,
        dtype=np.float64,
    )


@dataclass(frozen=True, eq=False)
class ThermalResult:
    """Thermal state of both nodes and the feasibility verdicts (PDF p.4 ThermalResult).

    Attributes:
        waste_heat_W: Radiator-node waste heat ``Q_waste`` [W].
        power_conditioning_loss_W: ``P_PV (1 - eta_pc)`` [W].
        electrical_to_beam_loss_W: ``P_dc_to_tx (1 - eta_e2b)`` [W].
        optics_or_rf_loss_W: ``P_beam (1 - eta_optics_or_rf)`` [W].
        other_heat_W: ``Q_other`` [W].
        radiator_area_m2: Required physical radiator area at ``T_lim`` [m^2].
        radiator_physical_area_m2: Configured physical area [m^2] or ``None``.
        radiator_mode: ``sized_at_limit`` or ``fixed_area``.
        equilibrium_temperature_K: Radiator temperature [K]: ``T_lim`` when sized at the limit,
            the analytic equilibrium temperature of the configured area otherwise.
        radiator_rejected_W: ``Q_rad`` at that temperature and area [W].
        radiator_feasible: ``equilibrium_temperature_K <= T_lim``.
        radiator_margin_K: Radiator temperature margin ``T_lim - equilibrium_temperature_K``
            [K] (0 by construction when sized at the limit).
        radiator_balance_floor_W: float64 representation floor of ``Q_rad`` at the solved
            radiator temperature (:func:`radiator_balance_floor_W`; 0 when sized at the limit,
            where no temperature is solved) [W].
        margin_K: Binding thermal margin to the temperature limits (PDF §8.4 "margin to
            temperature limit"): ``min(pv_margin_K, radiator_margin_K)`` [K]. ``feasible`` is
            exactly ``pv_converged & (margin_K >= 0)``, so the PDF pair (``feasible``,
            ``margin_K``) is always consistent.
        area_margin_m2: ``A_rad - A_req`` [m^2] (0 when sized at the limit).
        radiator_mass_kg: ``A_rad`` (or ``A_req``) times the areal mass [kg], or ``None``.
        pv_temperature_K: Solved PV plate temperature [K].
        pv_radiated_W: Power radiated by the PV node ``G (T^4 - T_space^4)`` [W].
        pv_balance_residual_W: PV-node balance residual [W].
        pv_balance_floor_W: float64 representation floor of that residual
            (:func:`pv_balance_floor_W`) [W].
        pv_converged: PV temperature solve converged.
        pv_unique_root: Uniqueness condition of the PV root holds.
        pv_feasible: ``pv_temperature_K <= collector.max_operating_temperature_K``.
        pv_margin_K: ``max_operating_temperature_K - pv_temperature_K`` [K].
        feasible: ``pv_converged & pv_feasible & radiator_feasible``.
        radiator: Radiator model used.
        pv_node: PV node used.
    """

    waste_heat_W: Float64Array
    power_conditioning_loss_W: Float64Array
    electrical_to_beam_loss_W: Float64Array
    optics_or_rf_loss_W: Float64Array
    other_heat_W: Float64Array
    radiator_area_m2: Float64Array
    radiator_physical_area_m2: float | None
    radiator_mode: RadiatorMode
    equilibrium_temperature_K: Float64Array
    radiator_rejected_W: Float64Array
    radiator_feasible: BoolArray
    radiator_margin_K: Float64Array
    radiator_balance_floor_W: Float64Array
    margin_K: Float64Array
    area_margin_m2: Float64Array
    radiator_mass_kg: Float64Array | None
    pv_temperature_K: Float64Array
    pv_radiated_W: Float64Array
    pv_balance_residual_W: Float64Array
    pv_balance_floor_W: Float64Array
    pv_converged: BoolArray
    pv_unique_root: BoolArray
    pv_feasible: BoolArray
    pv_margin_K: Float64Array
    feasible: BoolArray
    radiator: RadiatorModel = field(repr=False)
    pv_node: PvThermalNode = field(repr=False)

    def __post_init__(self) -> None:
        _freeze_1d_fields(self, np.shape(self.waste_heat_W))


def radiator_thermal(
    chain: OnboardPowerChain,
    radiator: RadiatorModel,
    pv: PvTemperatureSolution,
    pv_node: PvThermalNode,
    collector: CollectorModel,
) -> ThermalResult:
    """Radiator waste heat, sizing/temperature and both feasibility verdicts (node (b)).

    ``Q_waste`` is the sum of the three on-board stage losses of ``chain`` (single owner:
    :func:`helioswarm.power_chain.onboard_power_chain`) plus ``Q_other``.

    Args:
        chain: On-board power chain driven by the PV output of ``pv``.
        radiator: Radiator model.
        pv: PV-node solution.
        pv_node: PV node (reported in the result).
        collector: Collector (PV temperature limit).

    Returns:
        :class:`ThermalResult`.
    """
    shape = np.shape(chain.pv_electric_W)
    other = np.full(shape, radiator.other_heat_load_W)
    q_waste = (
        chain.power_conditioning_loss_W
        + chain.electrical_to_beam_loss_W
        + chain.optics_or_rf_loss_W
        + other
    )
    a_req = radiator_required_area_m2(q_waste, radiator)
    t_lim = radiator.temperature_limit_K
    if radiator.physical_area_m2 is None:
        t_rad = np.full(shape, t_lim)
        rejected = radiator_rejected_heat_W(t_rad, a_req, radiator)
        area_margin = np.zeros(shape)
        mass_area = a_req
        rad_floor = np.zeros(shape)
    else:
        t_rad = radiator_equilibrium_temperature_K(q_waste, radiator.physical_area_m2, radiator)
        rejected = radiator_rejected_heat_W(t_rad, radiator.physical_area_m2, radiator)
        area_margin = radiator.physical_area_m2 - a_req
        mass_area = np.full(shape, radiator.physical_area_m2)
        rad_floor = radiator_balance_floor_W(t_rad, radiator.physical_area_m2, radiator)
    rad_ok = t_rad <= t_lim
    t_pv = pv.temperature_K
    t_pv_max = collector.max_operating_temperature_K
    pv_ok = np.asarray(np.isfinite(t_pv) & (t_pv <= t_pv_max))
    areal_mass = radiator.specific_mass_kg_m2
    mass = None if areal_mass is None else mass_area * areal_mass
    rad_margin = t_lim - t_rad
    pv_margin = t_pv_max - t_pv
    return ThermalResult(
        waste_heat_W=q_waste,
        power_conditioning_loss_W=np.array(chain.power_conditioning_loss_W),
        electrical_to_beam_loss_W=np.array(chain.electrical_to_beam_loss_W),
        optics_or_rf_loss_W=np.array(chain.optics_or_rf_loss_W),
        other_heat_W=other,
        radiator_area_m2=a_req,
        radiator_physical_area_m2=radiator.physical_area_m2,
        radiator_mode=radiator.mode,
        equilibrium_temperature_K=t_rad,
        radiator_rejected_W=rejected,
        radiator_feasible=np.asarray(rad_ok),
        radiator_margin_K=rad_margin,
        radiator_balance_floor_W=np.asarray(rad_floor, dtype=np.float64),
        margin_K=np.minimum(pv_margin, rad_margin),
        area_margin_m2=np.asarray(area_margin, dtype=np.float64),
        radiator_mass_kg=None if mass is None else np.asarray(mass, dtype=np.float64),
        pv_temperature_K=np.array(t_pv),
        pv_radiated_W=np.array(pv.radiated_W),
        pv_balance_residual_W=np.array(pv.residual_W),
        pv_balance_floor_W=np.array(pv.balance_floor_W),
        pv_converged=np.array(pv.converged),
        pv_unique_root=np.array(pv.unique_root),
        pv_feasible=pv_ok,
        pv_margin_K=pv_margin,
        feasible=np.asarray(pv.converged & pv_ok & rad_ok),
        radiator=radiator,
        pv_node=pv_node,
    )


# ---------------------------------------------------------------------------------------------
# Energy ledger
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EnergyLedger:
    """One energy balance for collector, PV node, on-board chain and radiator (PDF §8.3).

    Residual attributes (each [W], per sample; module docstring for the identities):
        residual_collector_W: ``incident - reflected - absorbed``.
        residual_pv_node_W: ``absorbed - P_PV - Q_PV,radiated``.
        residual_power_conditioning_W: ``P_PV - P_dc_to_tx - loss_pc``.
        residual_transmitter_W: ``P_dc_to_tx - P_beam - loss_e2b``.
        residual_optics_W: ``P_beam - P_aperture - loss_optics``.
        residual_radiator_inventory_W: ``Q_waste - (loss_pc + loss_e2b + loss_optics +
            Q_other)``.
        residual_radiator_balance_W: ``Q_waste - Q_rad,rejected``.
        residual_spacecraft_W: ``incident + Q_other - (reflected + Q_PV,radiated +
            Q_rad,rejected + P_aperture)``.

    Other attributes:
        scale_W: ``incident + Q_other`` (residual normalisation) [W].
        pv_floor_W: float64 representation floor of the PV-node balance at the solved PV
            temperature (``ThermalResult.pv_balance_floor_W``) [W].
        radiator_floor_W: Same for the solved radiator temperature
            (``ThermalResult.radiator_balance_floor_W``; 0 when sized at the limit) [W].
        max_relative_residual: Largest ``|residual| / (scale + floor / rtol)`` over all
            identities and samples, where ``floor`` is the representation floor of the solved
            temperatures entering that identity (``pv_floor_W`` for the PV node,
            ``radiator_floor_W`` for the radiator balance, their sum for the spacecraft
            identity, 0 otherwise). ``<= rtol`` is thus ``|residual| <= rtol * scale + floor``;
            the floor matters only when ``scale`` is below ~``floor / rtol`` (e.g. a sliver of
            Sun at penumbra entry: MVP ``floor / rtol`` ~ 1e-4 W). Every identity's floor also
            contains the smallest normal float64 (2.2e-308 W), the absolute precision left once
            powers underflow to subnormals. NaN residuals count as infinite.
        rtol: Closure tolerance.
        closed: ``max_relative_residual <= rtol``.

    Independence of the identities: ``residual_pv_node_W`` (the solved PV temperature against
    Stefan-Boltzmann) and ``residual_collector_W`` / ``residual_radiator_inventory_W`` (sums
    formed independently of the code that produced their terms) are genuine checks. The three
    chain identities hold by construction (each loss is stage input minus output in
    :mod:`helioswarm.power_chain`), ``residual_radiator_balance_W`` is a round trip (``A_req``
    or ``T_eq`` is derived from the same ``Q_waste``), and ``residual_spacecraft_W`` is the
    algebraic sum of the others; they guard against regressions in the bookkeeping, not
    against wrong physics.
    """

    residual_collector_W: Float64Array
    residual_pv_node_W: Float64Array
    residual_power_conditioning_W: Float64Array
    residual_transmitter_W: Float64Array
    residual_optics_W: Float64Array
    residual_radiator_inventory_W: Float64Array
    residual_radiator_balance_W: Float64Array
    residual_spacecraft_W: Float64Array
    scale_W: Float64Array
    pv_floor_W: Float64Array
    radiator_floor_W: Float64Array
    max_relative_residual: float
    rtol: float
    closed: bool

    def __post_init__(self) -> None:
        _freeze_1d_fields(self, np.shape(self.scale_W))

    def residuals(self) -> dict[str, Float64Array]:
        """Residual name -> array [W] (for reporting)."""
        return {k: v for k, v in vars(self).items() if k.startswith("residual_")}


def energy_ledger(
    solar: SolarPowerResult,
    chain: OnboardPowerChain,
    thermal: ThermalResult,
    *,
    rtol: float = LEDGER_RTOL,
) -> EnergyLedger:
    """Check the energy balance of one evaluation (module docstring, "Energy ledger").

    Args:
        solar: Solar collection / PV conversion result.
        chain: On-board power chain driven by ``solar.pv_electric_W``.
        thermal: Thermal result of the same evaluation.
        rtol: Closure tolerance relative to ``incident + Q_other`` (default ``1e-12``), on
            top of the float64 representation floors of the solved temperatures
            (:class:`EnergyLedger`).

    Returns:
        :class:`EnergyLedger` (a diagnostic; it does not raise when the ledger is open).
    """
    if not (math.isfinite(rtol) and rtol > 0.0):
        raise ValueError("rtol must be finite and > 0")
    q_other = thermal.other_heat_W
    res = {
        "residual_collector_W": solar.incident_W - solar.reflected_W - solar.absorbed_W,
        "residual_pv_node_W": solar.absorbed_W - solar.pv_electric_W - thermal.pv_radiated_W,
        "residual_power_conditioning_W": chain.pv_electric_W
        - chain.dc_to_tx_W
        - chain.power_conditioning_loss_W,
        "residual_transmitter_W": chain.dc_to_tx_W - chain.beam_W - chain.electrical_to_beam_loss_W,
        "residual_optics_W": chain.beam_W - chain.aperture_W - chain.optics_or_rf_loss_W,
        "residual_radiator_inventory_W": thermal.waste_heat_W
        - (
            chain.power_conditioning_loss_W
            + chain.electrical_to_beam_loss_W
            + chain.optics_or_rf_loss_W
            + q_other
        ),
        "residual_radiator_balance_W": thermal.waste_heat_W - thermal.radiator_rejected_W,
        "residual_spacecraft_W": solar.incident_W
        + q_other
        - (
            solar.reflected_W
            + thermal.pv_radiated_W
            + thermal.radiator_rejected_W
            + chain.aperture_W
        ),
    }
    # The PV electric power entering the chain must be the PV node's output (one quantity).
    if not np.array_equal(chain.pv_electric_W, solar.pv_electric_W):
        raise ValueError("chain.pv_electric_W differs from solar.pv_electric_W")
    scale = solar.incident_W + q_other
    pv_floor = np.asarray(thermal.pv_balance_floor_W, dtype=np.float64)
    rad_floor = np.asarray(thermal.radiator_balance_floor_W, dtype=np.float64)
    floors = {
        "residual_pv_node_W": pv_floor,
        "residual_radiator_balance_W": rad_floor,
        "residual_spacecraft_W": pv_floor + rad_floor,
    }
    tiny = np.finfo(np.float64).tiny  # smallest normal float64: subnormal-underflow floor
    worst = 0.0
    for name, value in res.items():
        floor = floors.get(name, 0.0) + tiny
        rel = np.abs(value) / (scale + floor / rtol)
        rel = np.where(np.isfinite(rel), rel, np.inf)
        if rel.size:
            worst = max(worst, float(np.max(rel)))
    return EnergyLedger(
        **{k: np.asarray(v, dtype=np.float64) for k, v in res.items()},
        scale_W=np.asarray(scale, dtype=np.float64),
        pv_floor_W=pv_floor,
        radiator_floor_W=rad_floor,
        max_relative_residual=worst,
        rtol=rtol,
        closed=worst <= rtol,
    )


# ---------------------------------------------------------------------------------------------
# Orchestration: time series and heliocentric-distance curves (PDF §8.4)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerThermalModels:
    """All model parameters of the M4 evaluation (one place, from one configuration).

    The transmitter *operating point* is ``efficiencies`` (sweep it with
    ``dataclasses.replace``) together with the radiator model.
    """

    collector: CollectorModel
    pv_node: PvThermalNode
    efficiencies: OnboardEfficiencies
    radiator: RadiatorModel

    def __post_init__(self) -> None:
        if self.pv_node.space_temperature_K != self.radiator.space_temperature_K:
            raise ValueError("PV node and radiator must share the sink temperature T_space")


def power_thermal_models_from_config(config: RunConfig) -> PowerThermalModels:
    """:class:`PowerThermalModels` from a validated :class:`~helioswarm.config.RunConfig`."""
    return PowerThermalModels(
        collector=collector_model_from_config(config),
        pv_node=pv_thermal_node_from_config(config),
        efficiencies=onboard_efficiencies_from_config(config.transmitter),
        radiator=radiator_model_from_config(config),
    )


@dataclass(frozen=True, eq=False)
class PowerThermalResult:
    """Solar, on-board chain, thermal and ledger results of one evaluation.

    ``chain.dc_to_tx_W`` is the net electrical power available to the transmitter after
    power conditioning (PDF §8.4).
    """

    solar: SolarPowerResult
    chain: OnboardPowerChain
    thermal: ThermalResult
    ledger: EnergyLedger
    pv_solution: PvTemperatureSolution


def evaluate_power_thermal(
    sun_distance_m: ArrayLike,
    incidence_cosine: ArrayLike,
    illumination_fraction: ArrayLike,
    models: PowerThermalModels,
    *,
    epochs_tdb_s: ArrayLike | None = None,
) -> PowerThermalResult:
    """Solar collection, self-consistent PV temperature, on-board chain and radiator (M4).

    Steps: ``P_abs = S(r) A alpha cos(inc) f`` -> PV node ``T`` (brentq) -> ``P_PV`` ->
    on-board chain (:func:`~helioswarm.power_chain.onboard_power_chain`) -> radiator ``Q_waste``,
    ``A_req`` / ``T_eq`` -> energy ledger.

    Args:
        sun_distance_m: ``r_sun`` [m], shape (N,).
        incidence_cosine: ``cos(inc)`` [1], broadcastable to (N,).
        illumination_fraction: ``f`` [1], broadcastable to (N,).
        models: Model parameters.
        epochs_tdb_s: Optional TDB epochs (N,).

    Returns:
        :class:`PowerThermalResult`.
    """
    r = np.atleast_1d(np.asarray(sun_distance_m, dtype=np.float64))
    if r.ndim != 1:
        raise ValueError("sun_distance_m must be 1-D")
    cos_i = np.broadcast_to(np.asarray(incidence_cosine, dtype=np.float64), r.shape)
    frac = np.broadcast_to(np.asarray(illumination_fraction, dtype=np.float64), r.shape)
    absorbed = absorbed_solar_power_W(r, cos_i, frac, models.collector)
    pv = solve_pv_temperature(absorbed, models.collector, models.pv_node)
    if not np.all(np.isfinite(pv.temperature_K)):
        # Not expected for a valid efficiency law (the bracket contains a root in exact
        # arithmetic, and float64 end-point rounding is handled by the polish); a failure here
        # is a solver defect and must not be silently propagated.
        bad = int(np.count_nonzero(~np.isfinite(pv.temperature_K)))
        raise RuntimeError(f"PV temperature solve found no root at {bad} sample(s)")
    solar = solar_power(
        r, cos_i, frac, pv.temperature_K, models.collector, epochs_tdb_s=epochs_tdb_s
    )
    chain = onboard_power_chain(solar.pv_electric_W, models.efficiencies)
    thermal = radiator_thermal(chain, models.radiator, pv, models.pv_node, models.collector)
    ledger = energy_ledger(solar, chain, thermal)
    return PowerThermalResult(
        solar=solar, chain=chain, thermal=thermal, ledger=ledger, pv_solution=pv
    )


def power_thermal_vs_distance(
    sun_distance_m: ArrayLike,
    models: PowerThermalModels,
    *,
    incidence_cosine: float = 1.0,
    illumination_fraction: float = 1.0,
) -> PowerThermalResult:
    """PDF §8.4 curves over heliocentric distance (e.g. 0.3-1.2 au, given in metres).

    Returns, per distance: flux and collector power, PV efficiency and temperature, waste heat
    and required radiator area (or radiator temperature) at the transmitter operating point
    ``models.efficiencies``, net electrical power to the transmitter (``chain.dc_to_tx_W``),
    and the feasibility flags and margins. The default incidence (normal) and illumination
    (full Sun) are the Sun-tracking, unshadowed case.

    Args:
        sun_distance_m: Heliocentric distances [m], shape (M,).
        models: Model parameters.
        incidence_cosine: ``cos(inc)`` applied at every distance [1].
        illumination_fraction: ``f`` applied at every distance [1].

    Returns:
        :class:`PowerThermalResult` with ``solar.epochs_tdb_s = None``.
    """
    return evaluate_power_thermal(sun_distance_m, incidence_cosine, illumination_fraction, models)


def power_thermal_from_illumination(
    illumination: IlluminationResult, models: PowerThermalModels
) -> PowerThermalResult:
    """M4 evaluation along a trajectory, from the M3 illumination result.

    ``r_sun = illumination.sun_distance_m`` (satellite - retarded Sun centre, J2000 / SSB states
    from the ephemeris provider), ``cos(inc)`` from ``illumination.sun_direction_unit`` and the
    collector attitude, and ``f = illumination.illumination_fraction``.

    Args:
        illumination: Output of :func:`helioswarm.visibility.satellite_illumination`.
        models: Model parameters.

    Returns:
        :class:`PowerThermalResult` with the illumination epochs (TDB).
    """
    if illumination.frame != "J2000" or illumination.time_scale != "TDB":
        raise ValueError("illumination must be J2000 / TDB")
    cos_i = incidence_cosine(models.collector, illumination.sun_direction_unit)
    return evaluate_power_thermal(
        illumination.sun_distance_m,
        cos_i,
        illumination.illumination_fraction,
        models,
        epochs_tdb_s=illumination.epochs_tdb_s,
    )


__all__ = [
    "LEDGER_RTOL",
    "PV_BALANCE_RTOL",
    "EnergyLedger",
    "PowerThermalModels",
    "PowerThermalResult",
    "PvTemperatureSolution",
    "PvThermalNode",
    "RadiatorMode",
    "RadiatorModel",
    "ThermalResult",
    "energy_ledger",
    "evaluate_power_thermal",
    "fourth_power_difference_K4",
    "power_thermal_from_illumination",
    "power_thermal_models_from_config",
    "power_thermal_vs_distance",
    "pv_balance_floor_W",
    "pv_radiative_equilibrium_temperature_K",
    "pv_thermal_node_from_config",
    "radiator_balance_floor_W",
    "radiator_equilibrium_temperature_K",
    "radiator_model_from_config",
    "radiator_rejected_heat_W",
    "radiator_required_area_m2",
    "radiator_thermal",
    "solve_pv_temperature",
]
