"""Tests for helioswarm.thermal (PV node, radiator node, energy ledger; PDF §8.3-8.5, §13.1).

PDF §13.1 "thermal: energy closure and analytic radiative-equilibrium limiting cases". Every
reference value is an independent closed form written out here (Stefan-Boltzmann constant and
the au typed in from CODATA 2018 / IAU 2012), or an independent method (polynomial roots for
the coupled PV temperature).
"""

from __future__ import annotations

import dataclasses
import math
from decimal import Decimal, localcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from helioswarm.config import load_config, parse_config
from helioswarm.power_chain import OnboardEfficiencies, onboard_power_chain
from helioswarm.solar import CollectorModel, PvEfficiencyLaw
from helioswarm.thermal import (
    LEDGER_RTOL,
    PowerThermalModels,
    PvThermalNode,
    RadiatorModel,
    energy_ledger,
    evaluate_power_thermal,
    power_thermal_from_illumination,
    power_thermal_models_from_config,
    power_thermal_vs_distance,
    pv_radiative_equilibrium_temperature_K,
    radiator_equilibrium_temperature_K,
    radiator_rejected_heat_W,
    radiator_required_area_m2,
    solve_pv_temperature,
)
from helioswarm.visibility import satellite_illumination, solar_disk_occulted_fraction

from config_helpers import MVP_CONFIG, set_path
from geometry_helpers import LinearMotionEphemeris, linear_trajectory

AU = 149_597_870_700.0  # m, IAU 2012 B2 (exact)
SIGMA = 5.670374419e-8  # W m^-2 K^-4, CODATA 2018 (exact to these digits in the 2019 SI)
T_CMB = 2.72548  # K, Fixsen 2009
S1 = 1361.0  # W m^-2, IAU 2015 B3 nominal TSI
R_AU = np.linspace(0.3, 1.2, 46)


def _law(**kw: float) -> PvEfficiencyLaw:
    base: dict[str, float] = {
        "reference": 0.30,
        "reference_temperature_K": 300.0,
        "temperature_coefficient_per_K": -0.0004,
        "minimum": 0.0,
        "maximum": 1.0,
    }
    base.update(kw)
    return PvEfficiencyLaw(**base)


ZERO_LAW = PvEfficiencyLaw(0.0, 300.0, 0.0, 0.0, 1.0)
"""eta(T) = 0 identically: a non-converting plate (radiative-equilibrium limiting case)."""


def _collector(**kw: Any) -> CollectorModel:
    base: dict[str, Any] = {
        "area_m2": 10_000.0,
        "absorptivity": 0.9,
        "efficiency_law": _law(),
        "incidence_mode": "sun_tracking",
        "fixed_normal_unit_j2000": None,
        "max_operating_temperature_K": 400.0,
        "solar_constant_1au_W_m2": S1,
    }
    base.update(kw)
    return CollectorModel(**base)


def _node(**kw: float) -> PvThermalNode:
    base = {
        "front_emissivity": 0.9,
        "back_emissivity": 0.9,
        "view_factor": 1.0,
        "space_temperature_K": T_CMB,
    }
    base.update(kw)
    return PvThermalNode(**base)


def _radiator(**kw: Any) -> RadiatorModel:
    base: dict[str, Any] = {
        "emissivity": 0.9,
        "temperature_limit_K": 450.0,
        "radiating_sides": 2,
        "view_factor": 1.0,
        "space_temperature_K": T_CMB,
    }
    base.update(kw)
    return RadiatorModel(**base)


def _models(**kw: Any) -> PowerThermalModels:
    base: dict[str, Any] = {
        "collector": _collector(),
        "pv_node": _node(),
        "efficiencies": OnboardEfficiencies(0.9, 0.5, 0.8),
        "radiator": _radiator(),
    }
    base.update(kw)
    return PowerThermalModels(**base)


# ---------------------------------------------------------------------------------------------
# (a) PV node: radiative-equilibrium limiting cases and the coupled solve
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alpha", "eps_f", "eps_b", "view"),
    [(0.9, 0.9, 0.9, 1.0), (0.5, 0.85, 0.3, 1.0), (0.95, 0.8, 0.8, 0.6)],
)
def test_non_converting_plate_matches_radiative_equilibrium(
    alpha: float, eps_f: float, eps_b: float, view: float
) -> None:
    # eta = 0: alpha S(r) A = (eps_f + eps_b) sigma A F (T^4 - T_space^4), so
    # T = (alpha S / ((eps_f + eps_b) sigma F) + T_space^4)^(1/4) independent of A.
    col = _collector(absorptivity=alpha, efficiency_law=ZERO_LAW)
    node = _node(front_emissivity=eps_f, back_emissivity=eps_b, view_factor=view)
    res = evaluate_power_thermal(R_AU * AU, 1.0, 1.0, _models(collector=col, pv_node=node))
    s = S1 / R_AU**2
    expected = (alpha * s / ((eps_f + eps_b) * SIGMA * view) + T_CMB**4) ** 0.25
    # brentq locates T to a few ulp (xtol 4 ulp + rtol 4 eps): 1e-13 relative is ample.
    np.testing.assert_allclose(res.solar.pv_temperature_K, expected, rtol=1e-13, atol=0)
    assert np.all(res.solar.pv_electric_W == 0.0)
    # Closed-form helper agrees too (same physics, written independently above).
    t_helper = pv_radiative_equilibrium_temperature_K(res.solar.absorbed_W, col.area_m2, node)
    np.testing.assert_allclose(t_helper, expected, rtol=1e-14, atol=0)


def test_documented_1au_equilibrium_value() -> None:
    # alpha = 0.9, eps_f + eps_b = 1.8, F = 1 at 1 au: (0.9 * 1361 / (1.8 sigma))^(1/4).
    # By hand: 0.9 * 1361 = 1224.9; 1.8 * 5.670374419e-8 = 1.0206674e-7;
    # 1224.9 / 1.0206674e-7 = 1.200097e10 K^4; fourth root = 330.98 K (T_space^4 = 55.18 K^4
    # adds 55.18 / (4 * 330.98^3) = 3.8e-7 K).
    col = _collector(efficiency_law=ZERO_LAW)
    res = evaluate_power_thermal([AU], 1.0, 1.0, _models(collector=col))
    assert res.solar.pv_temperature_K[0] == pytest.approx(330.98, abs=0.01)


def _quartic_temperature(p_abs: float, g: float, law: PvEfficiencyLaw, ts: float) -> float:
    """Independent solution of the unclipped balance by polynomial roots.

    P (1 - eta_ref (1 + gamma (T - T_ref))) = G (T^4 - T_s^4)  <=>
    T^4 + (P eta_ref gamma / G) T - [P (1 - eta_ref + eta_ref gamma T_ref) / G + T_s^4] = 0.
    """
    a1 = p_abs * law.reference * law.temperature_coefficient_per_K / g
    a0 = -(
        p_abs
        * (
            1.0
            - law.reference
            + law.reference * law.temperature_coefficient_per_K * law.reference_temperature_K
        )
        / g
        + ts**4
    )
    roots = np.roots([1.0, 0.0, 0.0, a1, a0])
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9 * abs(r) and r.real > 0.0)
    assert len(real) == 1, roots
    return float(real[0])


@pytest.mark.parametrize("gamma", [-0.0004, -0.0015, 0.002])
def test_coupled_pv_temperature_matches_quartic_roots(gamma: float) -> None:
    law = _law(temperature_coefficient_per_K=gamma)
    col = _collector(efficiency_law=law)
    node = _node()
    res = evaluate_power_thermal(R_AU * AU, 1.0, 1.0, _models(collector=col, pv_node=node))
    assert not np.any(res.solar.pv_efficiency_clipped)  # the quartic is the unclipped law
    g = 1.8 * SIGMA * 10_000.0
    expected = [_quartic_temperature(p, g, law, T_CMB) for p in 0.9 * S1 / R_AU**2 * 10_000.0]
    # np.roots (companion eigenvalues) is accurate to ~1e-14 relative here; 1e-12 allows it.
    np.testing.assert_allclose(res.solar.pv_temperature_K, expected, rtol=1e-12, atol=0)
    # A solver that ignored the efficiency (eta = 0 equilibrium) is clearly different.
    t0 = (0.9 * S1 / R_AU**2 / (1.8 * SIGMA) + T_CMB**4) ** 0.25
    assert np.all(np.abs(res.solar.pv_temperature_K - t0) > 1.0)


def test_pv_balance_satisfied_to_tolerance_and_inside_bracket() -> None:
    col = _collector(efficiency_law=_law(temperature_coefficient_per_K=-0.004, minimum=0.02))
    p = 0.9 * S1 / R_AU**2 * col.area_m2
    sol = solve_pv_temperature(p, col, _node())
    eta = col.efficiency_law.efficiency(sol.temperature_K)
    # Independent recomputation of the balance with the constants typed in here.
    resid = p - p * eta - 1.8 * SIGMA * col.area_m2 * (sol.temperature_K**4 - T_CMB**4)
    assert np.all(np.abs(resid) <= 1e-12 * p)
    assert np.all(sol.converged)
    assert np.all(sol.unique_root)
    assert np.all(sol.bracket_low_K <= sol.temperature_K)
    assert np.all(sol.temperature_K <= sol.bracket_high_K)
    # The clip is active at the hot end for this steep coefficient (eta_min = 0.02).
    assert np.any(col.efficiency_law.is_clipped(sol.temperature_K))


def test_zero_absorbed_power_gives_sink_temperature() -> None:
    sol = solve_pv_temperature([0.0, 1.0e6], _collector(), _node())
    assert sol.temperature_K[0] == T_CMB
    assert sol.radiated_W[0] == 0.0
    assert sol.converged[0]


def test_hotter_closer_to_the_sun() -> None:
    res = power_thermal_vs_distance(R_AU * AU, _models())
    t = res.solar.pv_temperature_K
    assert np.all(np.diff(t) < 0.0)  # R_AU increases -> strictly cooler
    assert np.all(np.diff(res.solar.pv_efficiency) > 0.0)  # gamma < 0: cooler -> better
    assert np.all(np.diff(res.solar.flux_W_m2) < 0.0)


def test_pv_temperature_scales_with_view_factor() -> None:
    # eta = 0 and T >> T_space: T ∝ F^(-1/4). With T_space included exactly:
    col = _collector(efficiency_law=ZERO_LAW)
    t1 = evaluate_power_thermal([AU], 1.0, 1.0, _models(collector=col)).solar.pv_temperature_K
    t2 = evaluate_power_thermal(
        [AU], 1.0, 1.0, _models(collector=col, pv_node=_node(view_factor=0.5))
    ).solar.pv_temperature_K
    expected = (2.0 * (t1[0] ** 4 - T_CMB**4) + T_CMB**4) ** 0.25
    assert t2[0] == pytest.approx(expected, rel=1e-13)


def test_non_unique_root_is_flagged_not_hidden() -> None:
    # eta = 1 below 288.9 K (clipped) and 0 above 400 K: f(T_space) = 0 and f changes sign twice
    # more, so the uniqueness condition must fail; the returned root still balances.
    law = PvEfficiencyLaw(0.9, 300.0, -0.01, 0.0, 1.0)
    col = _collector(efficiency_law=law)
    g = 1.8 * SIGMA * col.area_m2
    p = 1.5 * g * 400.0**4
    sol = solve_pv_temperature([p], col, _node())
    assert not sol.unique_root[0]
    assert sol.converged[0]
    eta = law.efficiency(sol.temperature_K)
    assert abs(p * (1 - eta[0]) - g * (sol.temperature_K[0] ** 4 - T_CMB**4)) <= 1e-12 * p


# ---------------------------------------------------------------------------------------------
# (b) Radiator node
# ---------------------------------------------------------------------------------------------


def test_required_area_closed_form_and_inverse_scalings() -> None:
    q = 1.0e6
    rad = _radiator()
    a = radiator_required_area_m2(q, rad)
    expected = q / (0.9 * SIGMA * 2 * 1.0 * (450.0**4 - T_CMB**4))
    assert float(a) == pytest.approx(expected, rel=1e-14)
    # ∝ 1/eps
    a_half_eps = radiator_required_area_m2(q, _radiator(emissivity=0.45))
    assert float(a_half_eps / a) == pytest.approx(2.0, rel=1e-14)
    # ∝ 1/(T^4 - T_space^4)
    a_300 = radiator_required_area_m2(q, _radiator(temperature_limit_K=300.0))
    ratio = (450.0**4 - T_CMB**4) / (300.0**4 - T_CMB**4)
    assert float(a_300 / a) == pytest.approx(ratio, rel=1e-14)
    # Simple limiting case (T_space -> 0): ∝ 1/T^4 exactly.
    cold = {"space_temperature_K": 1e-6}
    ratio_t4 = radiator_required_area_m2(q, _radiator(temperature_limit_K=300.0, **cold))
    ratio_t4 = ratio_t4 / radiator_required_area_m2(q, _radiator(**cold))
    assert float(ratio_t4) == pytest.approx((450.0 / 300.0) ** 4, rel=1e-14)
    # Geometry: one radiating face needs twice the plate; F = 0.5 likewise.
    assert float(radiator_required_area_m2(q, _radiator(radiating_sides=1)) / a) == pytest.approx(
        2.0, rel=1e-14
    )
    assert float(radiator_required_area_m2(q, _radiator(view_factor=0.5)) / a) == pytest.approx(
        2.0, rel=1e-14
    )
    # Linear in Q.
    assert float(radiator_required_area_m2(3 * q, rad) / a) == pytest.approx(3.0, rel=1e-14)


def test_equilibrium_temperature_closed_form() -> None:
    rad = _radiator()
    q = np.array([0.0, 2.0e5, 1.0e6])
    t = radiator_equilibrium_temperature_K(q, 100.0, rad)
    expected = (q / (0.9 * SIGMA * 100.0 * 2.0) + T_CMB**4) ** 0.25
    np.testing.assert_allclose(t, expected, rtol=1e-14)
    assert t[0] == pytest.approx(T_CMB, rel=1e-15)
    np.testing.assert_allclose(radiator_rejected_heat_W(t, 100.0, rad), q, rtol=1e-13, atol=1e-9)
    # A radiator of exactly the required area runs at the limit.
    a_req = float(radiator_required_area_m2(1.0e6, rad))
    assert float(radiator_equilibrium_temperature_K(1.0e6, a_req, rad)) == pytest.approx(
        450.0, rel=1e-14
    )


def _waste_heat_by_hand(p_pv: np.ndarray, eff: OnboardEfficiencies, q_other: float) -> np.ndarray:
    p_dc = p_pv * eff.power_conditioning
    p_beam = p_dc * eff.electrical_to_beam
    return (
        p_pv * (1 - eff.power_conditioning)
        + p_dc * (1 - eff.electrical_to_beam)
        + p_beam * (1 - eff.optics_or_rf)
        + q_other
    )


def test_waste_heat_is_exactly_the_onboard_losses_plus_other() -> None:
    eff = OnboardEfficiencies(0.9, 0.5, 0.8)
    m = _models(efficiencies=eff, radiator=_radiator(other_heat_load_W=1.0e4))
    res = power_thermal_vs_distance(R_AU * AU, m)
    p_pv = res.solar.pv_electric_W
    expected = _waste_heat_by_hand(p_pv, eff, 1.0e4)
    np.testing.assert_allclose(res.thermal.waste_heat_W, expected, rtol=1e-14)
    # Net DC power to the transmitter after power conditioning (PDF §8.4).
    np.testing.assert_allclose(res.chain.dc_to_tx_W, 0.9 * p_pv, rtol=1e-15)
    # Beam leaving the aperture + radiator waste heat = P_PV + Q_other (nothing double-counted).
    np.testing.assert_allclose(
        res.chain.aperture_W + res.thermal.waste_heat_W, p_pv + 1.0e4, rtol=1e-14
    )
    expected_area = expected / (0.9 * SIGMA * 2.0 * (450.0**4 - T_CMB**4))
    np.testing.assert_allclose(res.thermal.radiator_area_m2, expected_area, rtol=1e-13)


def test_sized_radiator_runs_at_the_limit_and_reports_mass() -> None:
    m = _models(radiator=_radiator(specific_mass_kg_m2=5.0))
    th = power_thermal_vs_distance([AU], m).thermal
    assert th.radiator_mode == "sized_at_limit"
    assert th.equilibrium_temperature_K[0] == 450.0
    assert th.radiator_margin_K[0] == 0.0
    # The PDF-level margin is the binding one: here the radiator (0 K) binds, PV has ~97 K.
    assert th.pv_margin_K[0] > 90.0
    assert th.margin_K[0] == 0.0
    assert th.radiator_feasible[0]
    assert th.radiator_mass_kg is not None
    assert th.radiator_mass_kg[0] == pytest.approx(5.0 * th.radiator_area_m2[0], rel=1e-15)


def test_fixed_area_radiator_feasible_and_infeasible() -> None:
    base = power_thermal_vs_distance([AU], _models())
    q = float(base.thermal.waste_heat_W[0])
    a_req = float(base.thermal.radiator_area_m2[0])
    big = power_thermal_vs_distance([AU], _models(radiator=_radiator(physical_area_m2=2 * a_req)))
    small = power_thermal_vs_distance(
        [AU], _models(radiator=_radiator(physical_area_m2=0.5 * a_req))
    )
    for res, area in ((big, 2 * a_req), (small, 0.5 * a_req)):
        th = res.thermal
        t_expected = (q / (0.9 * SIGMA * area * 2.0) + T_CMB**4) ** 0.25
        assert th.radiator_mode == "fixed_area"
        assert th.equilibrium_temperature_K[0] == pytest.approx(t_expected, rel=1e-14)
        assert th.radiator_margin_K[0] == pytest.approx(450.0 - t_expected, rel=1e-12)
        # Binding margin: min of the PV (~97 K at 1 au) and radiator margins.
        assert th.margin_K[0] == min(th.pv_margin_K[0], th.radiator_margin_K[0])
        assert th.area_margin_m2[0] == pytest.approx(area - a_req, rel=1e-12)
        assert res.ledger.closed  # an overheated radiator still balances (it just runs hot)
    assert big.thermal.radiator_feasible[0]
    assert big.thermal.feasible[0]
    assert big.thermal.radiator_margin_K[0] > 0.0
    assert big.thermal.margin_K[0] > 0.0
    # Half the required area: Q cannot be rejected below the limit -> infeasible, flagged.
    assert not small.thermal.radiator_feasible[0]
    assert not small.thermal.feasible[0]
    assert small.thermal.radiator_margin_K[0] < 0.0
    assert small.thermal.margin_K[0] == small.thermal.radiator_margin_K[0]
    assert small.thermal.area_margin_m2[0] < 0.0
    # T scales as (A_req / A)^(1/4) T_lim when T_space^4 is negligible (1e-9 relative here).
    assert small.thermal.equilibrium_temperature_K[0] == pytest.approx(450.0 * 2.0**0.25, rel=1e-8)


def test_pv_over_temperature_is_infeasible() -> None:
    res = power_thermal_vs_distance(np.array([0.3, 1.0]) * AU, _models())
    th = res.thermal
    # 0.3 au: ~559 K > 400 K limit; 1 au: ~303 K.
    assert th.pv_temperature_K[0] > 400.0 > th.pv_temperature_K[1]
    np.testing.assert_array_equal(th.pv_feasible, [False, True])
    np.testing.assert_array_equal(th.feasible, [False, True])
    np.testing.assert_allclose(th.pv_margin_K, 400.0 - th.pv_temperature_K, rtol=1e-15)
    # Sized radiator: radiator margin 0 by construction; the PDF-level margin is the binding
    # one, so an over-temperature PV array gives a negative margin_K (not the radiator's 0).
    np.testing.assert_array_equal(th.radiator_margin_K, [0.0, 0.0])
    assert th.margin_K[0] == th.pv_margin_K[0] < -150.0
    assert th.margin_K[1] == 0.0


@pytest.mark.parametrize("radiator_area", [None, 400.0, 2000.0])
def test_feasible_flag_and_binding_margin_agree(radiator_area: float | None) -> None:
    # feasible == pv_converged & (margin_K >= 0) at every distance, in both radiator modes, and
    # margin_K is the smaller of the two node margins.
    m = _models(radiator=_radiator(physical_area_m2=radiator_area))
    th = power_thermal_vs_distance(R_AU * AU, m).thermal
    np.testing.assert_array_equal(th.margin_K, np.minimum(th.pv_margin_K, th.radiator_margin_K))
    np.testing.assert_array_equal(th.feasible, th.pv_converged & (th.margin_K >= 0.0))
    assert th.feasible.any()
    assert not th.feasible.all()


# ---------------------------------------------------------------------------------------------
# Energy ledger
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["sized", "fixed_area_small", "other_heat", "penumbra", "eclipse", "zero_law", "steep_clip"],
)
def test_energy_ledger_closes(case: str) -> None:
    cos_i, frac = 1.0, 1.0
    m = _models()
    if case == "fixed_area_small":
        m = _models(radiator=_radiator(physical_area_m2=50.0))
    elif case == "other_heat":
        m = _models(radiator=_radiator(other_heat_load_W=3.0e5))
    elif case == "penumbra":
        cos_i, frac = 0.7, 0.3
    elif case == "eclipse":
        frac = 0.0
        m = _models(radiator=_radiator(other_heat_load_W=1.0e3))
    elif case == "zero_law":
        m = _models(collector=_collector(efficiency_law=ZERO_LAW))
    elif case == "steep_clip":
        law = _law(temperature_coefficient_per_K=-0.004, minimum=0.02)
        m = _models(collector=_collector(efficiency_law=law))
    res = evaluate_power_thermal(R_AU * AU, cos_i, frac, m)
    led = res.ledger
    assert led.closed, led.max_relative_residual
    assert led.max_relative_residual <= LEDGER_RTOL
    for name, value in led.residuals().items():
        assert np.all(np.abs(value) <= 1e-12 * led.scale_W + 1e-300), name


def test_eclipse_steady_state() -> None:
    m = _models(radiator=_radiator(other_heat_load_W=2.0e3))
    res = evaluate_power_thermal([AU], 1.0, 0.0, m)
    assert res.solar.pv_temperature_K[0] == T_CMB
    assert res.solar.pv_electric_W[0] == 0.0
    assert res.thermal.waste_heat_W[0] == 2.0e3  # only Q_other is left
    assert res.ledger.closed


def test_reflected_sunlight_is_not_waste_heat() -> None:
    # alpha = 0.5 and eta = 0: the PV node must radiate the absorbed half only, so its
    # temperature is the equilibrium with alpha S (counting reflected light as heat would give
    # (S / (1.8 sigma))^(1/4), 19 % hotter).
    col = _collector(absorptivity=0.5, efficiency_law=ZERO_LAW)
    res = evaluate_power_thermal([AU], 1.0, 1.0, _models(collector=col))
    expected = (0.5 * S1 / (1.8 * SIGMA) + T_CMB**4) ** 0.25
    assert res.solar.pv_temperature_K[0] == pytest.approx(expected, rel=1e-13)
    assert res.thermal.pv_radiated_W[0] == pytest.approx(res.solar.absorbed_W[0], rel=1e-12)
    assert res.solar.reflected_W[0] == pytest.approx(0.5 * S1 * 1e4, rel=1e-15)
    # The radiator sees only the on-board losses (zero here: no electrical power).
    assert res.thermal.waste_heat_W[0] == 0.0
    # With conversion: radiator heat is independent of the reflected power. Two collectors
    # with the same absorbed power but different reflected power give the same radiator load.
    col_a = _collector(absorptivity=0.9)
    res_a = evaluate_power_thermal([AU], 1.0, 1.0, _models(collector=col_a))
    res_b = evaluate_power_thermal(
        [AU / math.sqrt(0.9 / 0.45)], 1.0, 1.0, _models(collector=_collector(absorptivity=0.45))
    )
    np.testing.assert_allclose(res_b.solar.absorbed_W, res_a.solar.absorbed_W, rtol=1e-14)
    assert res_b.solar.reflected_W[0] > 5 * res_a.solar.reflected_W[0]
    np.testing.assert_allclose(res_b.thermal.waste_heat_W, res_a.thermal.waste_heat_W, rtol=1e-12)
    np.testing.assert_allclose(
        res_b.solar.pv_temperature_K, res_a.solar.pv_temperature_K, rtol=1e-12
    )


def test_ledger_detects_a_broken_balance() -> None:
    res = power_thermal_vs_distance(R_AU * AU, _models())
    assert res.ledger.closed
    broken = dataclasses.replace(res.thermal, waste_heat_W=res.thermal.waste_heat_W * (1 + 1e-9))
    led = energy_ledger(res.solar, res.chain, broken)
    assert not led.closed
    assert led.max_relative_residual > 1e-11
    # A chain driven by a different PV power is a different quantity: refused.
    other_chain = onboard_power_chain(res.solar.pv_electric_W * 2, res.chain.efficiencies)
    with pytest.raises(ValueError, match="differs"):
        energy_ledger(res.solar, other_chain, res.thermal)


# ---------------------------------------------------------------------------------------------
# Curves vs distance (PDF §8.4), configuration and trajectory path
# ---------------------------------------------------------------------------------------------


def test_mvp_curves_vs_distance() -> None:
    m = power_thermal_models_from_config(load_config(MVP_CONFIG))
    r = np.linspace(0.3, 1.2, 91) * AU
    res = power_thermal_vs_distance(r, m)
    assert res.solar.epochs_tdb_s is None
    for arr in (
        res.solar.flux_W_m2,
        res.solar.pv_electric_W,
        res.solar.pv_efficiency,
        res.solar.pv_temperature_K,
        res.thermal.waste_heat_W,
        res.thermal.radiator_area_m2,
        res.chain.dc_to_tx_W,
        res.thermal.feasible,
        res.thermal.pv_margin_K,
    ):
        assert arr.shape == (91,)
    assert res.ledger.closed
    assert np.all(res.thermal.pv_converged)
    # MVP defaults: eta_pc = eta_optics = 1, eta_e2b = 0.5 -> Q_waste = P_PV / 2.
    np.testing.assert_allclose(res.thermal.waste_heat_W, 0.5 * res.solar.pv_electric_W, rtol=1e-15)
    # Hot inner orbits exceed the (assumed) 400 K PV limit; 1 au is feasible.
    assert not res.thermal.feasible[0]
    assert res.thermal.feasible[np.argmin(np.abs(r - AU))]


def test_models_from_config(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "thermal.pv_view_factor", 0.8)
    set_path(mvp_data, "thermal.pv_front_emissivity", 0.85)
    set_path(mvp_data, "thermal.pv_back_emissivity", 0.7)
    set_path(mvp_data, "thermal.radiator_area_m2", 500.0)
    set_path(mvp_data, "thermal.radiator_specific_mass_kg_m2", 3.0)
    set_path(mvp_data, "thermal.other_heat_load_W", 42.0)
    m = power_thermal_models_from_config(parse_config(mvp_data))
    assert m.pv_node == PvThermalNode(0.85, 0.7, 0.8, T_CMB)
    assert m.radiator == RadiatorModel(0.9, 450.0, 2, 1.0, T_CMB, 500.0, 3.0, 42.0)
    assert m.efficiencies == OnboardEfficiencies(1.0, 0.5, 1.0)


def test_config_rejects_pv_limit_below_sink(mvp_data: dict[str, Any]) -> None:
    from helioswarm.config import ConfigError

    set_path(mvp_data, "collector.max_operating_temperature_K", 2.0)
    with pytest.raises(ConfigError, match="max_operating_temperature_K"):
        parse_config(mvp_data)


def test_config_defaults_for_m4_fields() -> None:
    cfg = load_config(MVP_CONFIG)
    assert cfg.collector.max_operating_temperature_K == 400.0
    assert cfg.thermal.pv_view_factor == 1.0
    assert cfg.thermal.radiator_specific_mass_kg_m2 is None


def test_trajectory_path_fixed_normal_and_umbra() -> None:
    """Through the M3 illumination result: Sun at the origin, satellite 0.5 au away on +x.

    Epoch 0: full Sun, fixed normal at 60 deg from the Sun line -> S = 4 S_1AU, cos = 1/2.
    Epoch 1 (1e4 s later, the satellite has moved 3e9 m along +y): the static Earth lies 5e8 m
    sunward of it on the satellite-Sun line (Earth 1.3e-2 rad > Sun 9.3e-3 rad apparent
    radius: umbra) -> zero power, T = T_space.
    """
    zero = (0.0, 0.0, 0.0)
    far = (0.0, 0.0, 50 * AU)
    sat1 = np.array([0.5 * AU, 3.0e9, 0.0])
    earth = sat1 - 5.0e8 * sat1 / np.linalg.norm(sat1)  # on the satellite-Sun line at epoch 1
    eph = LinearMotionEphemeris(
        {"SUN": (zero, zero), "EARTH": (tuple(earth), zero), "MOON": (far, zero)}
    )
    sat = linear_trajectory([0.0, 1.0e4], (0.5 * AU, 0.0, 0.0), (0.0, 3.0e5, 0.0))
    ill = satellite_illumination(sat, ephemeris=eph)
    assert ill.illumination_fraction[0] == 1.0
    assert ill.illumination_fraction[1] == 0.0
    # Satellite -> Sun unit vector (J2000) carried by the M3 result for the incidence angle.
    np.testing.assert_allclose(ill.sun_direction_unit[0], [-1.0, 0.0, 0.0], rtol=0, atol=1e-15)
    assert ill.frame == "J2000"
    normal = (-0.5, math.sqrt(3.0) / 2.0, 0.0)  # Sun direction is -x
    col = _collector(incidence_mode="fixed_normal", fixed_normal_unit_j2000=normal)
    res = power_thermal_from_illumination(ill, _models(collector=col))
    assert res.solar.epochs_tdb_s is not None
    np.testing.assert_array_equal(res.solar.epochs_tdb_s, [0.0, 1.0e4])
    assert res.solar.flux_W_m2[0] == pytest.approx(4.0 * S1, rel=1e-15)
    assert res.solar.incidence_cosine[0] == pytest.approx(0.5, rel=1e-15)
    assert res.solar.absorbed_W[0] == pytest.approx(4.0 * S1 * 1e4 * 0.9 * 0.5, rel=1e-14)
    assert res.solar.pv_electric_W[1] == 0.0
    assert res.solar.pv_temperature_K[1] == T_CMB
    assert res.ledger.closed


# ---------------------------------------------------------------------------------------------
# Review additions: tiny absorbed power (penumbra edge), solver failure reporting, area scaling,
# hand-computed ledger
# ---------------------------------------------------------------------------------------------


def _decimal_pv_temperature(p_abs: float, law: PvEfficiencyLaw, g: float, ts: float) -> float:
    """Independent 50-digit fixed-point solution of P (1 - eta(T)) = G (T^4 - T_s^4).

    T_{n+1} = (T_s^4 + P (1 - eta(T_n)) / G)^(1/4) with the unclipped linear law (asserted
    unclipped by the caller); the contraction factor P |eta_ref gamma| / (4 G T^3) is < 1e-9
    for the powers used, so 6 steps converge far below float64 resolution.
    """
    with localcontext() as ctx:
        ctx.prec = 50
        p, gg, tsd = Decimal(p_abs), Decimal(g), Decimal(ts)
        eta_ref = Decimal(law.reference)
        gamma = Decimal(law.temperature_coefficient_per_K)
        t_ref = Decimal(law.reference_temperature_K)
        t = tsd
        for _ in range(6):
            eta = eta_ref * (1 + gamma * (t - t_ref))
            t = (tsd**4 + p * (1 - eta) / gg).sqrt().sqrt()
        return float(t)


@pytest.mark.parametrize("radiator_area", [None, 50.0])
def test_tiny_illumination_fraction_converges_and_closes(radiator_area: float | None) -> None:
    # Review finding (major): penumbra entry/exit gives tiny positive P_abs (a sliver of the
    # solar disk). The solve must converge to the right temperature, the sample must be
    # feasible, and the ledger must close; previously frac <= 1e-15 raised RuntimeError and
    # 1e-13 was reported unconverged/infeasible. 5e-324 is the smallest positive subnormal.
    fracs = np.array([1e-11, 1e-13, 1e-16, 1e-20, 1e-300, 5e-324])
    m = _models(radiator=_radiator(physical_area_m2=radiator_area))
    res = evaluate_power_thermal(np.full(fracs.size, AU), 1.0, fracs, m)
    th = res.thermal
    assert np.all(th.pv_converged)
    assert np.all(th.feasible)
    assert res.ledger.closed, res.ledger.max_relative_residual
    t = res.solar.pv_temperature_K
    assert np.all(t >= T_CMB)
    assert not np.any(res.solar.pv_efficiency_clipped)
    g = 1.8 * SIGMA * 10_000.0
    law = m.collector.efficiency_law
    expected = np.array([_decimal_pv_temperature(p, law, g, T_CMB) for p in res.solar.absorbed_W])
    # Within 2 float64 steps of the 50-digit root: the best representable T is <= 1 ulp away,
    # and the converged criterion admits a residual of one ulp's worth of |f'| (pv floor).
    assert np.all(np.abs(t - expected) <= 2 * np.spacing(expected)), (t - expected) / np.spacing(t)
    # The residual is at the float64 floor, not at 1e-12 P_abs (unreachable for tiny P_abs).
    assert np.all(
        np.abs(th.pv_balance_residual_W) <= 1e-12 * res.solar.absorbed_W + th.pv_balance_floor_W
    )
    assert np.all(th.pv_balance_floor_W < 1e-16)  # ~4e-17 W for this plate near T_space


def test_penumbra_edge_sweep_never_aborts() -> None:
    # Visible fractions from the real M3 disk-overlap function within +-1e-6 rad of the umbra
    # edge (Sun 4.65e-3 rad, Earth 1.2e-2 rad apparent radius): fractions down to ~1e-12.
    rs, rb = 4.65e-3, 1.2e-2
    theta = (rb - rs) + np.linspace(-1e-6, 1e-6, 2001)
    frac = 1.0 - solar_disk_occulted_fraction(rs, rb, theta)
    positive = frac[frac > 0.0]
    assert positive.size > 100
    assert positive.min() < 1e-10
    for r_au in (0.3, 1.0, 1.2):
        res = evaluate_power_thermal(np.full(theta.size, r_au * AU), 1.0, frac, _models())
        assert np.all(res.thermal.pv_converged)
        assert res.ledger.closed, (r_au, res.ledger.max_relative_residual)


def _fake_brentq(root_choice: str) -> Any:
    """Stand-in for scipy's brentq returning a chosen (root, info).

    ``midpoint``: bracket midpoint, converged False. ``low_end``: lower end, converged True.
    ``true_root_unconverged``: scipy's real root, but converged False. ``shifted_10ulp``: the
    real root moved 10 float64 steps up, converged True.
    """
    from scipy.optimize import brentq as real_brentq

    def fake(f: Any, a: float, b: float, **kw: Any) -> tuple[float, SimpleNamespace]:
        if root_choice == "midpoint":
            return 0.5 * (a + b), SimpleNamespace(converged=False, iterations=7)
        if root_choice == "low_end":
            return a, SimpleNamespace(converged=True, iterations=7)
        root = float(real_brentq(f, a, b, xtol=kw["xtol"], rtol=kw["rtol"]))
        if root_choice == "true_root_unconverged":
            return root, SimpleNamespace(converged=False, iterations=7)
        for _ in range(10):
            root = math.nextafter(root, math.inf)
        return root, SimpleNamespace(converged=True, iterations=7)

    return fake


def test_brentq_failure_is_reported_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    # brentq reports non-convergence: the midpoint must not be stored (T NaN, converged False)
    # and evaluate_power_thermal must refuse to propagate it.
    import helioswarm.thermal as thermal_module

    col = _collector()
    p = np.array([0.9 * S1 * col.area_m2])  # 1 au: bracket is ~K wide, brentq branch taken
    sol_ok = solve_pv_temperature(p, col, _node())
    assert sol_ok.converged[0]
    assert sol_ok.iterations[0] > 0
    monkeypatch.setattr(thermal_module, "brentq", _fake_brentq("midpoint"))
    sol = solve_pv_temperature(p, col, _node())
    assert np.isnan(sol.temperature_K[0])
    assert np.isnan(sol.residual_W[0])
    assert not sol.converged[0]
    assert sol.iterations[0] == 7
    with pytest.raises(RuntimeError, match="no root at 1 sample"):
        evaluate_power_thermal([AU], 1.0, 1.0, _models())


def test_unconverged_brentq_is_discarded_even_if_its_root_is_good(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The solver's own non-convergence verdict is respected (conservative): NaN, not converged.
    import helioswarm.thermal as thermal_module

    monkeypatch.setattr(thermal_module, "brentq", _fake_brentq("true_root_unconverged"))
    col = _collector()
    sol = solve_pv_temperature([0.9 * S1 * col.area_m2], col, _node())
    assert not sol.converged[0]
    assert np.isnan(sol.temperature_K[0])


def test_root_a_few_ulp_off_is_polished(monkeypatch: pytest.MonkeyPatch) -> None:
    # Small P_abs (1e-5 W): T sits next to T_space and one float64 step changes the radiated
    # power by ~4 G T_s^3 spacing(T_s) ~ 3.7e-17 W, more than 1e-12 P_abs. The bracket is still
    # open (brentq runs; asserted). A root 10 ulp off (residual measured 3.5e-16 W, tolerance
    # 4.7e-17 W) must be walked back to the best float64 by the polish.
    import helioswarm.thermal as thermal_module

    col = _collector()
    p = np.array([1e-5])
    ref = solve_pv_temperature(p, col, _node())
    assert ref.iterations[0] > 0  # brentq branch, not an accepted bracket end
    monkeypatch.setattr(thermal_module, "brentq", _fake_brentq("shifted_10ulp"))
    sol = solve_pv_temperature(p, col, _node())
    assert sol.converged[0]
    assert abs(sol.temperature_K[0] - ref.temperature_K[0]) <= np.spacing(ref.temperature_K[0])
    assert abs(sol.residual_W[0]) <= sol.balance_floor_W[0]


def test_fourth_power_difference_has_no_cancellation() -> None:
    # T^4 - T_s^4 at T = T_s + k ulp, against 50-digit Decimal arithmetic. The naive float form
    # T**4 - T_s**4 has absolute error ~ eps T_s^4 ~ 1e-14 K^4, i.e. O(1) relative error for
    # k = 1 (true value 3.6e-14 K^4); the factored form must be good to a few eps relative.
    from helioswarm.thermal import _fourth_power_difference_scalar, fourth_power_difference_K4

    ts = T_CMB
    temps = [ts]
    for _ in range(1000):
        temps.append(math.nextafter(temps[-1], math.inf))
    temps = [temps[k] for k in (0, 1, 2, 5, 1000)] + [ts * (1 + 1e-6), 300.0]
    with localcontext() as ctx:
        ctx.prec = 50
        exact = [float(Decimal(t) ** 4 - Decimal(ts) ** 4) for t in temps]
    vec = fourth_power_difference_K4(np.array(temps), ts)
    sca = np.array([_fourth_power_difference_scalar(t, ts) for t in temps])
    assert vec[0] == 0.0
    assert sca[0] == 0.0
    # Three roundings of exactly-representable-difference products: <= 4 eps relative.
    np.testing.assert_allclose(vec[1:], exact[1:], rtol=4 * np.finfo(float).eps, atol=0)
    np.testing.assert_allclose(sca[1:], exact[1:], rtol=4 * np.finfo(float).eps, atol=0)
    # The naive form is really worse at one ulp (this is what the factoring buys).
    naive = temps[1] ** 4 - ts**4
    assert abs(naive / exact[1] - 1.0) > 1e-3


def test_wrong_root_with_large_residual_is_not_converged(monkeypatch: pytest.MonkeyPatch) -> None:
    # A solver claiming convergence at the lower bracket end (far from the root: the residual
    # is ~1e-3 P_abs, beyond the 64-ulp polish) must not be accepted.
    import helioswarm.thermal as thermal_module

    monkeypatch.setattr(thermal_module, "brentq", _fake_brentq("low_end"))
    col = _collector()
    sol = solve_pv_temperature([0.9 * S1 * col.area_m2], col, _node())
    assert not sol.converged[0]
    assert np.isnan(sol.temperature_K[0])


def test_area_scaling_at_fixed_everything_else() -> None:
    # PDF §13.2: P_PV ∝ A; G ∝ A too, so T_pv is independent of A, and every downstream power
    # and the required radiator area scale by k. Tolerances: float rounding of A in P_abs and
    # in G (measured <= 4.4e-16 relative), 1e-14 allows it.
    k = 3.7
    a = evaluate_power_thermal(R_AU * AU, 1.0, 1.0, _models())
    b = evaluate_power_thermal(
        R_AU * AU, 1.0, 1.0, _models(collector=_collector(area_m2=k * 10_000.0))
    )
    np.testing.assert_allclose(b.solar.pv_temperature_K, a.solar.pv_temperature_K, rtol=1e-14)
    np.testing.assert_allclose(b.solar.pv_electric_W, k * a.solar.pv_electric_W, rtol=1e-14)
    np.testing.assert_allclose(b.thermal.waste_heat_W, k * a.thermal.waste_heat_W, rtol=1e-14)
    np.testing.assert_allclose(
        b.thermal.radiator_area_m2, k * a.thermal.radiator_area_m2, rtol=1e-14
    )
    # A conductance built with a different area than the absorbed power would shift T by
    # (1/k)^(1/4) = 28 %: far outside the tolerance.
    assert not np.allclose(a.solar.pv_temperature_K * k**-0.25, a.solar.pv_temperature_K)


def test_energy_ledger_from_hand_computed_terms() -> None:
    # Constant efficiency eta = 0.3 (gamma = 0): T = ((1 - 0.3) alpha S / (1.8 sigma) +
    # T_s^4)^(1/4) in closed form, so every ledger term is computable here without the module.
    eta = 0.3
    law = PvEfficiencyLaw(eta, 300.0, 0.0, 0.0, 1.0)
    eff = OnboardEfficiencies(0.9, 0.5, 0.8)
    m = _models(collector=_collector(efficiency_law=law), efficiencies=eff)
    r = np.array([0.4, 1.0]) * AU
    res = evaluate_power_thermal(r, 1.0, 1.0, m)
    area, alpha = 10_000.0, 0.9
    incident = S1 * (AU / r) ** 2 * area
    reflected = (1 - alpha) * incident
    absorbed = alpha * incident
    t_pv = ((1 - eta) * absorbed / (1.8 * SIGMA * area) + T_CMB**4) ** 0.25
    q_pv_rad = 1.8 * SIGMA * area * (t_pv**4 - T_CMB**4)
    p_pv = eta * absorbed
    aperture = p_pv * 0.9 * 0.5 * 0.8
    q_waste = p_pv - aperture
    # Closure of the hand terms (sanity of the hand formulas themselves).
    np.testing.assert_allclose(reflected + q_pv_rad + q_waste + aperture, incident, rtol=1e-14)
    # The module's terms equal the hand terms (rtol: brentq + float rounding, 1e-13).
    np.testing.assert_allclose(res.solar.pv_temperature_K, t_pv, rtol=1e-13)
    np.testing.assert_allclose(res.solar.reflected_W, reflected, rtol=1e-14)
    np.testing.assert_allclose(res.thermal.pv_radiated_W, q_pv_rad, rtol=1e-12)
    np.testing.assert_allclose(res.solar.pv_electric_W, p_pv, rtol=1e-14)
    np.testing.assert_allclose(res.chain.aperture_W, aperture, rtol=1e-14)
    np.testing.assert_allclose(res.thermal.waste_heat_W, q_waste, rtol=1e-14)
    np.testing.assert_allclose(res.thermal.radiator_rejected_W, q_waste, rtol=1e-14)
    assert res.ledger.closed
