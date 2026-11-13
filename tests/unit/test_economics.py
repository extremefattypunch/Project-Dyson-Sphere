"""Parameterised economics (helioswarm.economics; M8a, PDF §12.5).

Every reference is an independent closed form (annuity sums, geometric series, vis-viva,
Kepler's third law) or a hand computation stated in the test; tolerances are float64 round-off
unless stated.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helioswarm.cli import main
from helioswarm.config import load_config, parse_config, read_yaml, write_resolved_yaml
from helioswarm.constants import AU_M, GM_SUN_M3_S2
from helioswarm.economics import (
    EARTH_EQUATORIAL_RADIUS_M,
    GM_EARTH_M3_S2,
    STANDARD_GRAVITY_M_S2,
    EconomicsConfig,
    EconomicsError,
    RunEconomicInputs,
    _needs_ephemeris,
    capital_recovery_factor,
    cash_flows,
    escape_delta_v_m_s,
    evaluate_economics,
    heliocentric_transfer_delta_v,
    hohmann_delta_v_m_s,
    load_economics_config,
    monte_carlo,
    phasing_delta_v_m_s,
    receiver_capex_per_m2,
    run_inputs,
    transport_delta_v,
    tsiolkovsky_mass_ratio,
    with_values,
)

from config_helpers import CONFIG_DIR

ECON = CONFIG_DIR / "economics"


def _p(v: float) -> dict[str, Any]:
    return {"value": v, "provenance": "assumed"}


def _cfg(**over: Any) -> EconomicsConfig:
    base: dict[str, Any] = {
        "economics_schema_version": 1,
        "scenario": "test",
        "discount_rate": _p(0.05),
        "system_lifetime_years": {"value": 20, "provenance": "assumed"},
        "construction_years": {"value": 1, "provenance": "assumed"},
        "hardware_lifetime_years": _p(100.0),  # > lifetime: no replacement
        "replacement_fraction": _p(1.0),
        "annual_degradation": _p(0.0),
        "manufacturing_cost_per_kg_usd": _p(1000.0),
        "microwave_receiver_capex_per_m2_usd": _p(7.0),
        "optical_receiver_capex_per_m2_usd": _p(10.0),
        "opex_fixed_per_year_usd": _p(1.0e5),
        "opex_per_satellite_per_year_usd": _p(0.0),
        "opex_fraction_of_capex_per_year": _p(0.0),
        "electricity_price_usd_per_kWh": _p(0.1),
        "min_meaningful_energy_kWh_per_year": _p(1000.0),
        "mass": {
            "source": "specific_masses",
            "specific_masses": {
                "collector_areal_mass_kg_m2": _p(1.0),
                "radiator_areal_mass_kg_m2": _p(5.0),
                "transmitter_areal_mass_kg_m2": _p(50.0),
                "bus_mass_kg": _p(500.0),
            },
        },
        "transport": {
            "launch_cost_per_kg_leo_usd": _p(500.0),
            "isp_s": _p(450.0),
            "delta_v_model": "explicit",
            "delta_v_m_s": _p(0.0),
        },
    }
    for k, v in over.items():
        node = base
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return EconomicsConfig.model_validate(base, strict=False)


def _inputs(**over: Any) -> RunEconomicInputs:
    d: dict[str, Any] = {
        "run_dir": "synthetic",
        "kind": "single",
        "project_name": "p",
        "config_hash": "h",
        "transmitter_mode": "optical",
        "n_satellites": 1,
        "n_sites": 1,
        "delivered_energy_J": 3.6e6 * 1.0e6,  # 1e6 kWh (3.6e6 J each) over one Julian year
        "window_s": 365.25 * 86400.0,
        "collector_area_m2": 1.0e4,
        "radiator_area_m2": 100.0,
        "aperture_area_m2": 10.0,
        "receiver_area_m2": 1.0e4,
        "run_mass_proxy_kg": None,
        "semimajor_axis_m": (AU_M,),
        "earth_relative_phase_rad": (math.radians(60.0),),
        "circular_ecliptic": True,
    }
    d.update(over)
    if "semimajor_axis_m" not in over:  # 1 au for every satellite
        d["semimajor_axis_m"] = (AU_M,) * int(d["n_satellites"])
    return RunEconomicInputs(**d)


# ---- closed-form building blocks ---------------------------------------------------------------


@pytest.mark.parametrize(("r", "n"), [(0.03, 30), (0.07, 10), (0.2, 1), (1e-9, 25)])
def test_crf_is_the_inverse_annuity_factor(r: float, n: int) -> None:
    annuity = sum((1.0 + r) ** -k for k in range(1, n + 1))
    assert capital_recovery_factor(r, n) == pytest.approx(1.0 / annuity, rel=1e-12)


def test_crf_zero_rate_limit_and_known_value() -> None:
    assert capital_recovery_factor(0.0, 30) == 1.0 / 30.0
    # NASA OTPS 2024 inputs (r = 3 %, n = 30): 1.03^30 = 2.4272624712, so
    # CRF = 0.03 * 2.4272624712 / 1.4272624712 = 0.0510192593.
    assert capital_recovery_factor(0.03, 30) == pytest.approx(0.0510192593, abs=1e-10)


def test_tsiolkovsky_mass_ratio() -> None:
    isp = 450.0
    assert tsiolkovsky_mass_ratio(STANDARD_GRAVITY_M_S2 * isp * math.log(2.0), isp) == (
        pytest.approx(2.0, rel=1e-14)
    )
    assert tsiolkovsky_mass_ratio(0.0, isp) == 1.0
    # GEO case of configs/economics/geo_nominal.yaml: exp(4700 / (9.80665 * 450)) = 2.9010...
    assert tsiolkovsky_mass_ratio(4700.0, 450.0) == pytest.approx(
        math.exp(4700.0 / 4412.9925), rel=1e-12
    )
    with pytest.raises(ValueError, match="isp_s > 0"):
        tsiolkovsky_mass_ratio(100.0, 0.0)


def test_escape_and_hohmann_against_vis_viva() -> None:
    r = EARTH_EQUATORIAL_RADIUS_M + 400e3
    vc = math.sqrt(GM_EARTH_M3_S2 / r)
    assert escape_delta_v_m_s(0.0, r, GM_EARTH_M3_S2) == pytest.approx(
        (math.sqrt(2.0) - 1.0) * vc, rel=1e-14
    )
    # Hyperbola energy: v_p^2/2 - mu/r = v_inf^2/2 at periapsis.
    v_inf = 3000.0
    vp = vc + escape_delta_v_m_s(v_inf, r, GM_EARTH_M3_S2)
    assert vp**2 / 2.0 - GM_EARTH_M3_S2 / r == pytest.approx(v_inf**2 / 2.0, rel=1e-12)
    # Hohmann 1 au -> 1.5 au: transfer ellipse a_t = 1.25 au, vis-viva at both apsides.
    r1, r2 = AU_M, 1.5 * AU_M
    a_t = 0.5 * (r1 + r2)
    vp_t = math.sqrt(GM_SUN_M3_S2 * (2.0 / r1 - 1.0 / a_t))
    va_t = math.sqrt(GM_SUN_M3_S2 * (2.0 / r2 - 1.0 / a_t))
    dv1, dv2 = hohmann_delta_v_m_s(r1, r2, GM_SUN_M3_S2)
    assert dv1 == pytest.approx(vp_t - math.sqrt(GM_SUN_M3_S2 / r1), rel=1e-12)
    assert dv2 == pytest.approx(math.sqrt(GM_SUN_M3_S2 / r2) - va_t, rel=1e-12)
    assert hohmann_delta_v_m_s(r2, r1, GM_SUN_M3_S2) == pytest.approx((dv2, dv1), rel=1e-12)


@pytest.mark.parametrize(("phase_deg", "revs"), [(60.0, 1), (60.0, 3), (-120.0, 2), (330.0, 2)])
def test_phasing_orbit_gains_the_phase(phase_deg: float, revs: int) -> None:
    mu, a = GM_SUN_M3_S2, AU_M
    dv = phasing_delta_v_m_s(math.radians(phase_deg), revs, a, mu)
    phi = math.remainder(math.radians(phase_deg), 2 * math.pi)
    v = math.sqrt(mu / a) + (-dv if phi > 0 else dv)  # ahead: slow down (smaller, faster orbit)
    a2 = 1.0 / (2.0 / a - v * v / mu)
    period = 2 * math.pi * math.sqrt(a**3 / mu)
    period2 = 2 * math.pi * math.sqrt(a2**3 / mu)
    # After N revolutions of the phasing orbit the reference body has moved N P'/P turns.
    assert revs * (period - period2) / period * 2 * math.pi == pytest.approx(phi, rel=1e-9)
    assert phasing_delta_v_m_s(0.0, revs, a, mu) == 0.0


def test_more_phasing_time_needs_less_delta_v() -> None:
    dvs = [phasing_delta_v_m_s(math.radians(90.0), n, AU_M, GM_SUN_M3_S2) for n in (1, 2, 4)]
    assert dvs[0] > dvs[1] > dvs[2] > 0.0


def test_heliocentric_transfer_methods() -> None:
    co = heliocentric_transfer_delta_v(AU_M, 0.0, phasing_years=2, parking_altitude_m=400e3)
    r = EARTH_EQUATORIAL_RADIUS_M + 400e3
    assert co.method == "phasing"
    assert co.v_inf_m_s == 0.0
    assert co.arrival_m_s == 0.0
    assert co.total_m_s == pytest.approx((math.sqrt(2) - 1) * math.sqrt(GM_EARTH_M3_S2 / r))
    h = heliocentric_transfer_delta_v(0.7 * AU_M, 1.0, phasing_years=2, parking_altitude_m=400e3)
    dv1, dv2 = hohmann_delta_v_m_s(AU_M, 0.7 * AU_M, GM_SUN_M3_S2)
    assert h.method == "hohmann"
    assert h.v_inf_m_s == dv1
    assert h.arrival_m_s == dv2
    assert h.total_m_s == pytest.approx(h.departure_m_s + dv2)


# ---- cash flows, LCOE, NPV ----------------------------------------------------------------------


def test_lcoe_of_the_analytic_case_is_crf_capex_plus_opex_over_energy() -> None:
    cfg = _cfg()
    inp = _inputs()
    res = evaluate_economics(inp, cfg)
    # mass = 500 + 1 * 1e4 + 5 * 100 + 50 * 10 = 11500 kg; CAPEX = 11500 (1000 + 500) + 10 * 1e4.
    assert res.mass_total_kg == 11500.0
    assert res.capex_usd == pytest.approx(11500 * 1500.0 + 1.0e5)
    e = 1.0e6
    assert res.annual_energy_kWh == pytest.approx(e, rel=1e-12)
    crf = capital_recovery_factor(0.05, 20)
    expected = crf * res.capex_usd / e + 1.0e5 / e
    assert res.lcoe_usd_per_kWh == pytest.approx(expected, rel=1e-12)
    assert res.lcoe_usd_per_MWh == pytest.approx(1000.0 * expected, rel=1e-12)
    # NPV = (p E - O) / CRF - C0 for constant annual flows.
    assert res.npv_usd == pytest.approx((0.1 * e - 1.0e5) / crf - res.capex_usd, rel=1e-10)
    # Equivalent annual cost = LCOE x E for constant energy (T_c = 1).
    assert res.equivalent_annual_cost_usd == pytest.approx(expected * e, rel=1e-12)
    # NPV is zero exactly at a price equal to the LCOE.
    price = cfg.electricity_price_usd_per_kWh.model_copy(update={"value": expected})
    at_lcoe = evaluate_economics(
        inp, cfg.model_copy(update={"electricity_price_usd_per_kWh": price})
    )
    assert at_lcoe.npv_usd == pytest.approx(0.0, abs=1e-6 * res.capex_usd)
    assert res.flags == ()


def test_degradation_follows_the_geometric_series_and_raises_lcoe() -> None:
    d, r, n, e = 0.02, 0.05, 20, 1.0e6
    res0 = evaluate_economics(_inputs(), _cfg())
    res = evaluate_economics(_inputs(), _cfg(**{"annual_degradation": _p(d)}))
    q = (1 - d) / (1 + r)
    assert res.pv_energy_kWh == pytest.approx(e / (1 + r) * (1 - q**n) / (1 - q), rel=1e-12)
    assert res.lcoe_usd_per_kWh is not None
    assert res0.lcoe_usd_per_kWh is not None
    assert res.lcoe_usd_per_kWh > res0.lcoe_usd_per_kWh
    lcoes = [
        evaluate_economics(_inputs(), _cfg(**{"annual_degradation": _p(x)})).lcoe_usd_per_kWh
        for x in (0.0, 0.005, 0.01, 0.05)
    ]
    assert all(a < b for a, b in itertools.pairwise(lcoes))  # type: ignore[operator]


def test_discounting_raises_lcoe_of_front_loaded_capex() -> None:
    lcoes = [
        evaluate_economics(_inputs(), _cfg(**{"discount_rate": _p(x)})).lcoe_usd_per_kWh
        for x in (0.0, 0.03, 0.07, 0.12)
    ]
    assert all(a < b for a, b in itertools.pairwise(lcoes))  # type: ignore[operator]
    # r = 0: LCOE = (C0 + n O) / (n E).
    res = evaluate_economics(_inputs(), _cfg(**{"discount_rate": _p(0.0)}))
    assert res.lcoe_usd_per_kWh == pytest.approx((res.capex_usd + 20 * 1e5) / (20 * 1e6))


def test_replacements_at_each_hardware_end_of_life_with_performance_reset() -> None:
    cf = cash_flows(
        capex_usd=100.0,
        space_capex_usd=80.0,
        opex_per_year_usd=1.0,
        first_year_energy_kWh=10.0,
        discount_rate=0.0,
        lifetime_years=30,
        construction_years=1,
        hardware_lifetime_years=10.0,
        replacement_fraction=1.0,
        annual_degradation=0.1,
    )
    assert np.flatnonzero(cf.replacement_usd).tolist() == [10, 20]  # 2 cycles in 30 years
    assert cf.replacement_usd.sum() == 160.0
    assert cf.energy_kWh[1] == 10.0
    assert cf.energy_kWh[11] == 10.0
    assert cf.energy_kWh[10] == pytest.approx(10.0 * 0.9**9)
    # Partial replacement: P = f + (1 - f) P (1 - d).
    cf2 = cash_flows(
        capex_usd=100.0, space_capex_usd=80.0, opex_per_year_usd=0.0, first_year_energy_kWh=10.0,
        discount_rate=0.0, lifetime_years=12, construction_years=2, hardware_lifetime_years=5.0,
        replacement_fraction=0.5, annual_degradation=0.1,
    )  # fmt: skip
    assert cf2.capex_usd[:2].tolist() == [50.0, 50.0]  # two construction instalments
    assert cf2.energy_kWh[1] == 0.0
    assert cf2.energy_kWh[2] == 10.0
    p6 = 0.5 + 0.5 * 0.9**5
    assert cf2.energy_kWh[1 + 6] == pytest.approx(10.0 * p6)


def test_zero_energy_gives_undefined_lcoe_with_flag_and_negative_npv() -> None:
    res = evaluate_economics(_inputs(delivered_energy_J=0.0), _cfg())
    assert res.lcoe_usd_per_kWh is None
    assert res.lcoe_usd_per_MWh is None
    assert "zero_energy" in res.flags
    assert "negligible_energy" in res.flags
    assert res.npv_usd == pytest.approx(-res.pv_cost_usd)
    # Near-zero delivery (0.1 kWh/yr, below min_meaningful_energy_kWh_per_year = 1000):
    # LCOE undefined with a flag; the formal ratio is kept for diagnosis.
    tiny = evaluate_economics(_inputs(delivered_energy_J=3.6e5), _cfg())
    assert tiny.lcoe_usd_per_kWh is None
    assert tiny.flags == ("negligible_energy",)
    assert tiny.lcoe_formal_usd_per_kWh == pytest.approx(tiny.pv_cost_usd / tiny.pv_energy_kWh)
    ok = evaluate_economics(_inputs(delivered_energy_J=3.6e6 * 1000.0), _cfg())  # 1000 kWh/yr
    assert ok.lcoe_usd_per_kWh is not None
    assert ok.flags == ()


def test_transport_uses_per_satellite_delta_v_and_mass_ratio() -> None:
    cfg = _cfg(
        **{
            "transport.delta_v_model": "patched_conic_heliocentric",
            "transport.delta_v_m_s": None,
            "transport.phasing_years": {"value": 2, "provenance": "assumed"},
        }
    )
    inp = _inputs(n_satellites=2, earth_relative_phase_rad=(math.radians(60), math.radians(-90)))
    res = evaluate_economics(inp, cfg)
    expect = 0.0
    for phi in inp.earth_relative_phase_rad or ():
        d = heliocentric_transfer_delta_v(AU_M, phi, phasing_years=2, parking_altitude_m=400e3)
        expect += 500.0 * (res.mass_total_kg / 2) * tsiolkovsky_mass_ratio(d.total_m_s, 450.0)
    assert res.capex_breakdown_usd["transport"] == pytest.approx(expect, rel=1e-12)
    assert res.delta_v_method == "phasing"
    with pytest.raises(EconomicsError, match="ephemeris"):
        evaluate_economics(_inputs(earth_relative_phase_rad=None), cfg)
    with pytest.raises(EconomicsError, match="circular ecliptic"):
        evaluate_economics(_inputs(circular_ecliptic=False), cfg)


def test_run_mass_proxy_is_used_when_requested() -> None:
    cfg = _cfg(**{"mass.source": "run", "mass.specific_masses": None})
    res = evaluate_economics(_inputs(run_mass_proxy_kg=1234.0), cfg)
    assert res.mass_total_kg == 1234.0
    assert res.mass_source == "run"
    with pytest.raises(EconomicsError, match="no mass proxy"):
        evaluate_economics(_inputs(), cfg)
    auto = _cfg(**{"mass.source": "auto"})
    assert evaluate_economics(_inputs(run_mass_proxy_kg=1234.0), auto).mass_total_kg == 1234.0
    assert evaluate_economics(_inputs(), auto).mass_total_kg == 11500.0


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="needs a source"):
        _cfg(**{"discount_rate": {"value": 0.03, "provenance": "sourced"}})
    with pytest.raises(ValueError, match="delta_v_m_s is required"):
        _cfg(**{"transport.delta_v_m_s": None})
    with pytest.raises(ValueError, match="annual_degradation"):
        _cfg(**{"annual_degradation": _p(1.5)})
    with pytest.raises(ValueError, match="unknown uncertain"):
        _cfg(
            uncertainty={
                "n_samples": 10,
                "seed": 1,
                "parameters": {"bogus": {"low": 0, "mode": 1, "high": 2}},
            }
        )
    with pytest.raises(EconomicsError, match="cannot vary"):
        with_values(_cfg(), {"system_lifetime_years": 3})
    with pytest.raises(ValueError, match="needs a source"):
        _cfg(**{"construction_years": {"value": 2, "provenance": "derived"}})
    # Monte Carlo ranges: the mode must be the scenario value, the bounds admissible values.
    mc = {"n_samples": 10, "seed": 1}
    with pytest.raises(ValueError, match=r"mode 0\.04 differs"):
        _cfg(
            uncertainty={
                **mc,
                "parameters": {"discount_rate": {"low": 0.03, "mode": 0.04, "high": 0.07}},
            }
        )
    with pytest.raises(ValueError, match=r"uncertainty\.annual_degradation"):
        _cfg(
            uncertainty={
                **mc,
                "parameters": {"annual_degradation": {"low": 0.0, "mode": 0.0, "high": 1.2}},
            }
        )
    # A sampled configuration is re-validated (with_values bypasses nothing).
    with pytest.raises(EconomicsError, match="replacement_fraction"):
        with_values(_cfg(), {"replacement_fraction": 1.5})
    assert (
        with_values(
            _cfg(
                uncertainty={
                    **mc,
                    "parameters": {"discount_rate": {"low": 0.03, "mode": 0.05, "high": 0.07}},
                }
            ),
            {"discount_rate": 0.06},
        ).uncertainty
        is None
    )


# ---- shipped scenarios ---------------------------------------------------------------------------


def _demo_inputs() -> RunEconomicInputs:
    phases = tuple(math.radians(30 + 30 * k) for k in range(10))
    return _inputs(
        kind="constellation",
        n_satellites=10,
        n_sites=3,
        run_mass_proxy_kg=166196.27,
        receiver_area_m2=3 * math.pi * 1e6,
        earth_relative_phase_rad=phases,
        delivered_energy_J=1.3953e12,
    )


def test_shipped_scenarios_are_ordered_and_provenance_complete() -> None:
    cfgs = {
        n: load_economics_config(ECON / f"{n}.yaml")
        for n in ("optimistic", "nominal", "conservative")
    }
    inp = _demo_inputs()
    lcoe = {n: evaluate_economics(inp, c).lcoe_usd_per_kWh for n, c in cfgs.items()}
    assert lcoe["optimistic"] < lcoe["nominal"] < lcoe["conservative"]  # type: ignore[operator]
    npv = {n: evaluate_economics(inp, c).npv_usd for n, c in cfgs.items()}
    assert npv["optimistic"] > npv["nominal"] > npv["conservative"]
    for c in cfgs.values():
        for name, field in c:
            if hasattr(field, "provenance"):
                assert field.provenance in ("sourced", "derived", "assumed"), name
                if field.provenance != "assumed":
                    assert "PDF p." in (field.source or ""), name  # page-level citation
    geo = load_economics_config(ECON / "geo_nominal.yaml")
    assert geo.transport.delta_v_model == "explicit"
    assert geo.transport.delta_v_m_s is not None
    res = evaluate_economics(_inputs(), geo)
    assert res.delta_v_m_s == (4700.0,)


def test_monte_carlo_is_seeded_and_brackets_the_median() -> None:
    nominal = load_economics_config(ECON / "nominal.yaml")
    small = nominal.model_copy(
        update={"uncertainty": nominal.uncertainty.model_copy(update={"n_samples": 200})}  # type: ignore[union-attr]
    )
    inp = _demo_inputs()
    a = monte_carlo(inp, small, np.random.default_rng(1))
    b = monte_carlo(inp, small, np.random.default_rng(1))
    c = monte_carlo(inp, small, np.random.default_rng(2))
    assert a.lcoe_percentiles_usd_per_kWh == b.lcoe_percentiles_usd_per_kWh
    assert a.lcoe_percentiles_usd_per_kWh != c.lcoe_percentiles_usd_per_kWh
    p = a.lcoe_percentiles_usd_per_kWh
    assert p["p05"] < p["p50"] < p["p95"]
    opt = evaluate_economics(inp, load_economics_config(ECON / "optimistic.yaml"))
    con = evaluate_economics(inp, load_economics_config(ECON / "conservative.yaml"))
    # The MC varies a subset of the scenario inputs; each range's bounds are exactly the
    # optimistic and conservative scenario values (checked here; the mode = nominal is checked
    # on load). That the 5-95 % interval then lies inside the scenario LCOE range is an observed
    # property for this seed (the LCOE is not proven monotone in r), asserted below.
    o_cfg = load_economics_config(ECON / "optimistic.yaml")
    c_cfg = load_economics_config(ECON / "conservative.yaml")
    assert nominal.uncertainty is not None
    for name, rng in nominal.uncertainty.parameters.items():
        node_o: Any = o_cfg
        node_c: Any = c_cfg
        for part in name.split("."):
            node_o, node_c = getattr(node_o, part), getattr(node_c, part)
        assert {rng.low, rng.high} == {node_o.value, node_c.value}, name
    assert opt.lcoe_usd_per_kWh < p["p05"]
    assert p["p95"] < con.lcoe_usd_per_kWh
    assert a.n_undefined_lcoe == 0


# ---- saved run directory and CLI -----------------------------------------------------------------


def _fake_constellation_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    cfg = load_config(CONFIG_DIR / "constellation_demo.yaml")
    write_resolved_yaml(cfg, run / "config_resolved.yaml")
    summary = {
        "summary_schema": "helioswarm-constellation-summary/1",
        "project_name": "constellation_demo",
        "config_hash": "x",
        "transmitter_mode": "optical",
        "n_satellites": 10,
        "network": {"delivered_energy_J": 1.3953e12},
        "time": {"window_s": 365.25 * 86400.0},
        "totals": {
            "collector_area_m2": 1e5,
            "radiator_area_required_m2": 4385.27,
            "transmitter_aperture_area_m2": 785.398,
            "mass_proxy_kg": 166196.27,
        },
    }
    (run / "summary.json").write_text(json.dumps(summary))
    return run


def test_run_inputs_and_cli_from_a_saved_run(tmp_path: Path) -> None:
    run = _fake_constellation_run(tmp_path)
    inp = run_inputs(run)
    assert inp.kind == "constellation"
    assert inp.n_satellites == 10
    assert inp.n_sites == 3
    assert inp.receiver_area_m2 == pytest.approx(3 * math.pi * 1000.0**2)
    assert inp.run_mass_proxy_kg == 166196.27
    # Earth-relative even phases over [30, 330): 30 + 30 k deg.
    np.testing.assert_allclose(
        np.degrees(inp.earth_relative_phase_rad), 30.0 + 30.0 * np.arange(10), atol=1e-9
    )
    assert inp.annual_energy_kWh == pytest.approx(1.3953e12 / 3.6e6)
    out = tmp_path / "econ"
    args = ["economics", str(run), "--out", str(out)]
    for n in ("optimistic", "nominal", "conservative"):
        args += ["--config", str(ECON / f"{n}.yaml")]
    assert main(args) == 0
    doc = json.loads((out / "economics.json").read_text())
    assert [r["scenario"] for r in doc["runs"]] == ["optimistic", "nominal", "conservative"]
    assert doc["runs"][1]["monte_carlo"]["lcoe_percentiles_usd_per_kWh"]["p50"] > 0.0
    assert "not a core physical result" in doc["disclaimer"]
    assert "| run |" in (out / "economics.md").read_text()
    assert main(args) == 1  # refuses to overwrite
    assert main([*args, "--overwrite"]) == 0


# ---- review follow-ups (M8a review) --------------------------------------------------------------


def test_opex_terms_and_window_annualisation_in_closed_form() -> None:
    cfg = _cfg(
        **{
            "opex_per_satellite_per_year_usd": _p(2.0e4),
            "opex_fraction_of_capex_per_year": _p(0.01),
        }
    )
    window = 30.0 * 86400.0
    e_win_kWh = 5.0e4
    inp = _inputs(
        n_satellites=3,
        semimajor_axis_m=(AU_M,) * 3,
        earth_relative_phase_rad=None,
        delivered_energy_J=e_win_kWh * 3.6e6,
        window_s=window,
    )
    res = evaluate_economics(inp, cfg)
    e = e_win_kWh * 365.25 / 30.0  # E_win (Julian year / window)
    assert res.annual_energy_kWh == pytest.approx(e, rel=1e-12)
    # mass = 3 x 500 + 1e4 + 5 x 100 + 50 x 10 = 12500 kg; C0 = 12500 x 1500 + 10 x 1e4.
    c0 = 12500.0 * 1500.0 + 1.0e5
    assert res.capex_usd == pytest.approx(c0, rel=1e-12)
    o = 1.0e5 + 3 * 2.0e4 + 0.01 * c0  # O_fix + N O_sat + o C0
    assert res.opex_per_year_usd == pytest.approx(o, rel=1e-12)
    crf = capital_recovery_factor(0.05, 20)
    assert res.lcoe_usd_per_kWh == pytest.approx(crf * c0 / e + o / e, rel=1e-12)


def test_equivalent_annual_cost_with_a_construction_period() -> None:
    # Reviewer's case: r = 5 %, n = 20, E = 1e6 kWh/yr, C0 = 1e8, no opex: EAC = LCOE x E for
    # every construction period (it was low by 1.05^-(T_c - 1) before the fix).
    for tc in (1, 3, 5):
        cfg = _cfg(
            **{
                "construction_years": {"value": tc, "provenance": "assumed"},
                "opex_fixed_per_year_usd": _p(0.0),
                "manufacturing_cost_per_kg_usd": _p(1.0e8 / 11500.0),
                "transport.launch_cost_per_kg_leo_usd": _p(0.0),
                "optical_receiver_capex_per_m2_usd": _p(0.0),
            }
        )
        res = evaluate_economics(_inputs(), cfg)
        assert res.capex_usd == pytest.approx(1.0e8, rel=1e-12)
        assert res.lcoe_usd_per_kWh is not None
        assert res.equivalent_annual_cost_usd == pytest.approx(
            res.lcoe_usd_per_kWh * 1.0e6, rel=1e-12
        )
        # Closed form: PV = C0/T_c sum_{t<T_c} 1.05^-t; EAC = CRF PV 1.05^(T_c - 1).
        pv = 1.0e8 / tc * sum(1.05**-t for t in range(tc))
        assert res.equivalent_annual_cost_usd == pytest.approx(
            capital_recovery_factor(0.05, 20) * pv * 1.05 ** (tc - 1), rel=1e-12
        )


def test_receiver_cost_follows_the_technology_and_transmitter_capex_is_space_capex() -> None:
    cfg = _cfg(
        **{
            "transmitter_capex_per_m2_usd": _p(1000.0),
            "hardware_lifetime_years": _p(10.0),
            "discount_rate": _p(0.0),
            "opex_fixed_per_year_usd": _p(0.0),
        }
    )
    opt = evaluate_economics(_inputs(), cfg)
    mw = evaluate_economics(_inputs(transmitter_mode="microwave"), cfg)
    assert receiver_capex_per_m2(_inputs(), cfg).value == 10.0
    assert opt.capex_breakdown_usd["receiver"] == 10.0 * 1.0e4
    assert mw.capex_breakdown_usd["receiver"] == 7.0 * 1.0e4
    assert opt.capex_breakdown_usd["transmitter"] == 1000.0 * 10.0  # x aperture area 10 m^2
    # Replacements (years 10 and 20 of 20 operation years -> only year 10; the last year has
    # none) renew manufacturing + transport + transmitter, not the receiver.
    space = 11500.0 * 1500.0 + 1.0e4
    assert opt.pv_breakdown_usd["replacement"] == pytest.approx(space, rel=1e-12)
    with pytest.raises(EconomicsError, match="transmitter_mode"):
        evaluate_economics(_inputs(transmitter_mode="laser"), cfg)


# -- transport: the slot phase is part of every transfer -------------------------------------------


def _theta0(a_m: float) -> float:
    """Earth-relative phase at Hohmann arrival, from Kepler's third law (independent)."""
    a_t = 0.5 * (AU_M + a_m)
    t_h = math.pi * math.sqrt(a_t**3 / GM_SUN_M3_S2)
    return math.pi - math.sqrt(GM_SUN_M3_S2 / AU_M**3) * t_h


def _hohmann_only(a_m: float) -> float:
    dv1, dv2 = hohmann_delta_v_m_s(AU_M, a_m, GM_SUN_M3_S2)
    return escape_delta_v_m_s(dv1, EARTH_EQUATORIAL_RADIUS_M + 400e3, GM_EARTH_M3_S2) + dv2


def test_transfer_delta_v_is_continuous_across_1_au() -> None:
    phi = math.radians(150.0)
    ref = heliocentric_transfer_delta_v(AU_M, phi, phasing_years=2, parking_altitude_m=400e3)
    assert ref.method == "phasing"
    # Reviewer's counter-example: 1.1e-6 au (165 m) away the Hohmann branch used to drop the
    # phasing requirement (6115 -> 3176 m/s). Now the change is O(eps): the drift arc is
    # ~ 3 pi eps N, ~1e-5 rad here, and the Hohmann impulses ~ v_E eps / 2 ~ 0.02 m/s.
    prev = 0.0
    for eps in (1e-4, 1.1e-6, 1e-9):
        for sgn in (1.0, -1.0):
            d = heliocentric_transfer_delta_v(
                AU_M * (1.0 + sgn * eps), phi, phasing_years=2, parking_altitude_m=400e3
            )
            assert d.method == "hohmann+phasing"
            diff = abs(d.total_m_s - ref.total_m_s)
            assert diff < 1.0e5 * eps * 30.0, (
                eps,
                diff,
            )  # Lipschitz-type bound, ~3 km/s per 0.001 au
        if prev:
            assert diff < prev
        prev = diff
    near = heliocentric_transfer_delta_v(
        AU_M * 1.0000011, phi, phasing_years=2, parking_altitude_m=400e3
    )
    assert near.total_m_s == pytest.approx(ref.total_m_s, abs=1.0)


def test_far_orbits_reach_any_slot_by_free_drift() -> None:
    # 0.7 au: synodic period 1 / (0.7^-1.5 - 1) = 1.41 Earth years <= 2: pure Hohmann, any phase.
    a = 0.7 * AU_M
    for deg in (0.0, 45.0, 150.0, -100.0):
        d = heliocentric_transfer_delta_v(a, math.radians(deg), phasing_years=2,
                                          parking_altitude_m=400e3)  # fmt: skip
        assert d.method == "hohmann"
        assert d.synodic_period_years == pytest.approx(1.0 / (0.7**-1.5 - 1.0), rel=1e-12)
        assert d.total_m_s == pytest.approx(_hohmann_only(a), rel=1e-12)
    # 1.05 au: synodic period 14.2 yr > 2: the drift arc covers only part of the circle.
    a = 1.05 * AU_M
    n_e = math.sqrt(GM_SUN_M3_S2 / AU_M**3)
    arc = (math.sqrt(GM_SUN_M3_S2 / a**3) - n_e) * 2 * (2 * math.pi / n_e)  # negative: outer
    assert -2 * math.pi < arc < 0.0
    inside = heliocentric_transfer_delta_v(
        a, _theta0(a) + 0.5 * arc, phasing_years=2, parking_altitude_m=400e3
    )
    assert inside.method == "hohmann"
    assert inside.drift_arc_rad == pytest.approx(arc, rel=1e-12)
    assert inside.total_m_s == pytest.approx(_hohmann_only(a), rel=1e-12)
    # Just outside either end of the arc: tiny residual phase, Δv just above the Hohmann value.
    for e, sgn in ((_theta0(a), 1.0), (_theta0(a) + arc, -1.0)):
        d = heliocentric_transfer_delta_v(
            a, e + sgn * 1e-3, phasing_years=2, parking_altitude_m=400e3
        )
        assert d.method == "hohmann+phasing"
        assert abs(d.residual_phase_rad) == pytest.approx(1e-3, rel=1e-6)
        assert 0.0 < d.total_m_s - _hohmann_only(a) < 1.0
    # Opposite the arc: a large residual; dv_ph is the cheaper of the two arc ends.
    far = heliocentric_transfer_delta_v(
        a, _theta0(a) + 0.5 * arc + math.pi, phasing_years=2, parking_altitude_m=400e3
    )
    ends = [
        phasing_delta_v_m_s(_theta0(a) + 0.5 * arc + math.pi - e, 2, a, GM_SUN_M3_S2)
        for e in (_theta0(a), _theta0(a) + arc)
    ]
    assert far.phasing_m_s == pytest.approx(min(ends), rel=1e-12)
    assert far.total_m_s > _hohmann_only(a) + 1000.0


def _ring_run(tmp_path: Path, **ring: Any) -> Path:
    tmp_path.mkdir(parents=True)
    run = _fake_constellation_run(tmp_path)
    raw = read_yaml(CONFIG_DIR / "constellation_demo.yaml")
    raw["constellation"]["rings"][0].update(ring)
    write_resolved_yaml(parse_config(raw), run / "config_resolved.yaml")
    return run


def test_ring_element_overrides_set_each_satellites_transfer(tmp_path: Path) -> None:
    cfg = load_economics_config(ECON / "nominal.yaml")
    inner = run_inputs(_ring_run(tmp_path / "a", semimajor_axis_au=0.7))
    assert inner.semimajor_axis_m == (0.7 * AU_M,) * 10
    assert inner.circular_ecliptic
    dvs, method = transport_delta_v(inner, cfg)
    assert method == "hohmann"
    assert dvs == pytest.approx((_hohmann_only(0.7 * AU_M),) * 10, rel=1e-12)
    # An inclined ring violates the coplanar patched-conic validity: refused, not mis-costed.
    tilted = run_inputs(_ring_run(tmp_path / "b", semimajor_axis_au=0.7, inclination_deg=10.0))
    assert not tilted.circular_ecliptic
    with pytest.raises(EconomicsError, match="circular ecliptic"):
        transport_delta_v(tilted, cfg)


def test_ring_with_absolute_phase_loads_the_ephemeris(tmp_path: Path) -> None:
    cfgs = [load_economics_config(ECON / "nominal.yaml")]
    (tmp_path / "a").mkdir()
    base = _fake_constellation_run(tmp_path / "a")  # earth_relative everywhere
    assert not _needs_ephemeris([base], cfgs)
    ring_abs = _ring_run(tmp_path / "b", phase_reference="absolute")
    assert _needs_ephemeris([ring_abs], cfgs)
    assert run_inputs(ring_abs).earth_relative_phase_rad is None  # needs kernels
    assert not _needs_ephemeris([ring_abs], [load_economics_config(ECON / "geo_nominal.yaml")])


@pytest.mark.spice
def test_single_run_with_absolute_phase(tmp_path: Path, kernel_set: Any) -> None:
    """Single-run summary branch + absolute-phase branch (the MVP economics path), against an
    independent SPICE evaluation of the Earth's heliocentric ecliptic longitude."""
    import spiceypy

    from helioswarm.ephemeris import SpiceEphemeris

    from config_helpers import MVP_CONFIG

    run = tmp_path / "mvp"
    run.mkdir()
    cfg = load_config(MVP_CONFIG)
    assert cfg.orbit.phase_reference == "absolute"
    write_resolved_yaml(cfg, run / "config_resolved.yaml")
    summary = {
        "summary_schema": "helioswarm-summary/2",
        "project_name": "mvp",
        "config_hash": "x",
        "transmitter_mode": "optical",
        "delivered_energy_J": 5.5e10,
        "time": {"window_s": 365.25 * 86400.0},
        "thermal": {"radiator_area_m2_max": 123.0},
    }
    (run / "summary.json").write_text(json.dumps(summary))
    with kernel_set.loaded(verify=False) as k:  # the CLI below loads its own pool
        inp = run_inputs(run, ephemeris=SpiceEphemeris(k))
        et = spiceypy.str2et(cfg.orbit.initial_epoch_utc.replace("T", " ") + " UTC")
        pos, _lt = spiceypy.spkpos("EARTH", et, "ECLIPJ2000", "NONE", "SUN")
    assert inp.kind == "single"
    assert inp.n_satellites == 1
    d = cfg.transmitter.aperture_diameter_m
    assert inp.aperture_area_m2 == pytest.approx(math.pi * d * d / 4.0, rel=1e-15)
    assert inp.radiator_area_m2 == 123.0
    assert inp.collector_area_m2 == cfg.collector.area_m2
    # Independent: lambda_sat = Omega + omega + nu (circular, i = 0) = 30 deg; Earth from
    # spkpos(EARTH wrt SUN, ECLIPJ2000, geometric) at the initial epoch; phase = sat - Earth.
    lam_sat = math.radians(
        cfg.orbit.raan_deg + cfg.orbit.arg_periapsis_deg + cfg.orbit.true_anomaly_deg
    )
    expected = math.remainder(lam_sat - math.atan2(pos[1], pos[0]), 2 * math.pi)
    assert inp.earth_relative_phase_rad is not None
    got = math.remainder(inp.earth_relative_phase_rad[0], 2 * math.pi)
    # 1e-9 rad: float64 round-off of the two longitude evaluations (same kernels, same epoch).
    assert got == pytest.approx(expected, abs=1e-9)
    assert math.degrees(got) == pytest.approx(-70.18, abs=0.01)  # the value in M8a.md
    nominal = load_economics_config(ECON / "nominal.yaml")
    dvs, method = transport_delta_v(inp, nominal)
    ref = heliocentric_transfer_delta_v(AU_M, expected, phasing_years=2, parking_altitude_m=400e3)
    assert method == "phasing"
    assert dvs == pytest.approx((ref.total_m_s,), rel=1e-9)
    # The sign matters (phasing is asymmetric): +70.18 deg would need ~208 m/s more.
    flip = heliocentric_transfer_delta_v(AU_M, -expected, phasing_years=2, parking_altitude_m=400e3)
    assert flip.total_m_s - ref.total_m_s > 100.0
    out = tmp_path / "econ"
    assert (
        main(["economics", str(run), "--out", str(out), "--config", str(ECON / "nominal.yaml")])
        == 0
    )
    doc = json.loads((out / "economics.json").read_text())
    assert doc["runs"][0]["inputs"]["kind"] == "single"
    assert doc["runs"][0]["result"]["delta_v_m_s"] == pytest.approx([ref.total_m_s], rel=1e-9)
