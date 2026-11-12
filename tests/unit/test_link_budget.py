"""PDF §10.1 / §10.4 link budget (helioswarm.link_budget).

References are hand computations of the stage chain written out here; tolerances are exact or
1 ulp-level where products are re-associated (stated per test).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

import helioswarm.link_budget as lb_mod
from helioswarm.config import parse_config
from helioswarm.link_budget import (
    STAGE_COLUMNS,
    LinkBudgetInvariantError,
    ReceiverModel,
    evaluate_link_budget,
    link_budget_violations,
    receiver_model_from_config,
)
from helioswarm.power_chain import OnboardEfficiencies

from config_helpers import set_path

EFF = OnboardEfficiencies(power_conditioning=0.95, electrical_to_beam=0.5, optics_or_rf=0.9)
RX = ReceiverModel("laser_photovoltaic", conversion_efficiency=0.4, grid_efficiency=0.97,
                   max_input_power_W=1.0e12)  # fmt: skip


def _budget(p_pv: Any, rx: ReceiverModel = RX, **kw: Any) -> lb_mod.LinkBudgetResult:
    n = np.size(p_pv)
    args: dict[str, Any] = {
        "link_enabled": np.ones(n, dtype=bool),
        "eta_atmosphere": np.full(n, 0.8),
        "eta_capture": np.full(n, 0.01),
        "peak_irradiance_per_W_m2": np.full(n, 1.0e-9),
    }
    args.update(kw)
    return evaluate_link_budget(np.atleast_1d(np.asarray(p_pv, float)), EFF, rx, **args)


def test_stage_chain_matches_hand_computation() -> None:
    p_pv = 3.0e6
    r = _budget([p_pv])
    # Hand-computed chain (PDF §10.1), each product written in the chain's own order.
    dc = p_pv * 0.95
    beam = dc * 0.5
    ap = beam * 0.9
    atm = ap * 0.8
    cap = atm * 0.01
    rec = cap * 0.4
    grid = rec * 0.97
    expected = (p_pv, p_pv, dc, beam, ap, atm, cap, cap, rec, grid)
    for (_, attr), ref in zip(STAGE_COLUMNS, expected, strict=True):
        # Exact: the module multiplies in the same order (one owner per factor).
        assert float(getattr(r, attr)[0]) == ref, attr
    assert r.receiver_heat_W[0] == pytest.approx(cap * 0.6, rel=1e-15)
    assert r.eta_receiver_acceptance[0] == 1.0
    # PDF p.4 field names.
    assert r.power_at_transmitter_W[0] == dc
    assert r.power_at_receiver_W[0] == cap
    assert r.atmospheric_factor[0] == 0.8
    assert r.rx_electric_W[0] == rec
    assert r.grid_W[0] == grid


def test_capture_and_atmosphere_are_applied_once_without_aperture_factor() -> None:
    # Discriminating for double counting: P_captured / P_atm must equal eta_capture exactly
    # and P_atm / P_aperture eta_atmosphere exactly (no extra aperture/taper efficiency).
    rng = np.random.default_rng(3)
    n = 200
    eta_a = rng.uniform(0.1, 1.0, n)
    eta_c = rng.uniform(1e-8, 1.0, n)
    r = _budget(rng.uniform(1e5, 4e6, n), eta_atmosphere=eta_a, eta_capture=eta_c)
    np.testing.assert_array_equal(r.atm_W, r.aperture_W * eta_a)
    np.testing.assert_array_equal(r.captured_W, r.atm_W * eta_c)
    np.testing.assert_array_equal(r.eta_capture, eta_c)


def test_onboard_stages_have_a_single_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []
    real = lb_mod.onboard_power_chain

    def spy(p: Any, eff: OnboardEfficiencies) -> Any:
        calls.append(float(np.sum(p)))
        return real(p, eff)

    monkeypatch.setattr(lb_mod, "onboard_power_chain", spy)
    r = _budget([1.0e6, 2.0e6])
    assert calls, "link budget must obtain the on-board stages from power_chain"
    # The returned stages are exactly those of power_chain.onboard_power_chain.
    ref = real(np.array([1.0e6, 2.0e6]), EFF)
    np.testing.assert_array_equal(r.dc_to_tx_W, ref.dc_to_tx_W)
    np.testing.assert_array_equal(r.beam_W, ref.beam_W)
    np.testing.assert_array_equal(r.aperture_W, ref.aperture_W)


def test_disabled_epochs_carry_zero_power_and_zero_factors() -> None:
    en = np.array([True, False, True])
    r = _budget(
        [1e6, 1e6, 1e6],
        link_enabled=en,
        eta_atmosphere=np.array([0.5, np.nan, 0.5]),
        eta_capture=np.array([0.1, np.nan, 0.1]),
        peak_irradiance_per_W_m2=np.array([1e-9, np.nan, 1e-9]),
    )
    assert r.pv_electric_W[1] == 1e6  # still generated
    for _, attr in STAGE_COLUMNS[1:]:
        assert getattr(r, attr)[1] == 0.0
    assert r.eta_atmosphere[1] == 0.0
    assert r.eta_capture[1] == 0.0
    assert list(r.transmitting) == [True, False, True]
    assert not link_budget_violations(r)


def test_nan_factor_at_enabled_epoch_is_rejected() -> None:
    with pytest.raises(ValueError, match="eta_capture"):
        _budget([1e6], eta_capture=np.array([np.nan]))
    with pytest.raises(ValueError, match="eta_atmosphere"):
        _budget([1e6], eta_atmosphere=np.array([1.2]))


def test_receiver_input_limit_clips_captured_power() -> None:
    rx = dataclasses.replace(RX, max_input_power_W=2.0e3)
    p = np.array([1e5, 1e6, 4e6])
    r = _budget(p, rx=rx)
    cap = p * 0.95 * 0.5 * 0.9 * 0.8 * 0.01
    clipped = cap > 2.0e3
    assert clipped.tolist() == [False, True, True]
    np.testing.assert_array_equal(r.receiver_saturated, clipped)
    np.testing.assert_array_equal(r.accepted_W, np.minimum(cap, 2.0e3))
    np.testing.assert_allclose(r.receiver_clipped_W, cap - np.minimum(cap, 2.0e3), rtol=1e-15)
    # Clipped epochs deliver exactly limit * eta_rx * eta_grid.
    assert np.all(r.grid_W[clipped] == 2.0e3 * 0.4 * 0.97)
    assert r.eta_receiver_acceptance[2] == pytest.approx(2.0e3 / cap[2], rel=1e-15)


def test_receiver_heat_rejection_limit() -> None:
    # Q_rx = P_accepted (1 - 0.4) <= 600 W  =>  P_accepted <= 1000 W.
    rx = dataclasses.replace(RX, max_heat_rejection_W=600.0)
    r = _budget([1e6], rx=rx)
    cap = 1e6 * 0.95 * 0.5 * 0.9 * 0.8 * 0.01
    assert cap > 1000.0
    assert r.accepted_W[0] == pytest.approx(1000.0, rel=1e-15)
    assert r.receiver_heat_W[0] == pytest.approx(600.0, rel=1e-13)
    assert r.receiver_heat_limited[0]
    assert not r.receiver_saturated[0]


def test_power_density_beam_abort() -> None:
    p = 1.0e6
    peak_per_w = 1.0e-9
    density = p * 0.95 * 0.5 * 0.9 * 0.8 * peak_per_w  # requested peak irradiance [W m^-2]
    below = dataclasses.replace(RX, max_power_density_W_m2=density * 1.000001)
    above = dataclasses.replace(RX, max_power_density_W_m2=density * 0.999999)
    ok = _budget([p], rx=below)
    assert ok.transmitting[0]
    assert not ok.power_density_abort[0]
    assert ok.peak_power_density_W_m2[0] == pytest.approx(density, rel=1e-15)
    ab = _budget([p], rx=above)
    assert ab.power_density_abort[0]
    assert not ab.transmitting[0]
    assert ab.requested_peak_power_density_W_m2[0] == pytest.approx(density, rel=1e-15)
    for _, attr in STAGE_COLUMNS[1:]:
        assert getattr(ab, attr)[0] == 0.0
    assert ab.peak_power_density_W_m2[0] == 0.0


def test_invariant_violation_is_detected_on_construction() -> None:
    r = _budget([1e6])
    # A grid power above the receiver output (a passive stage increasing power) must raise.
    with pytest.raises(LinkBudgetInvariantError, match="grid_W exceeds receiver_W"):
        dataclasses.replace(r, grid_W=r.receiver_W * 1.5)
    with pytest.raises(LinkBudgetInvariantError, match="eta_capture"):
        dataclasses.replace(r, eta_capture=np.array([1.5]))


def test_random_budgets_satisfy_invariants() -> None:
    rng = np.random.default_rng(11)
    n = 5000
    rx = dataclasses.replace(RX, max_input_power_W=1e4, max_heat_rejection_W=4e3)
    r = _budget(
        rng.uniform(0, 4e6, n),
        rx=rx,
        link_enabled=rng.random(n) > 0.4,
        eta_atmosphere=rng.random(n),
        eta_capture=rng.random(n),
        peak_irradiance_per_W_m2=rng.uniform(0, 1e-8, n),
    )
    assert link_budget_violations(r) == ()
    for _, attr in STAGE_COLUMNS[1:]:
        assert np.all(r.grid_W <= getattr(r, attr))


def test_receiver_model_from_config_is_technology_specific(mvp_data: dict[str, Any]) -> None:
    rx = receiver_model_from_config(parse_config(mvp_data).receiver)
    assert rx.technology == "laser_photovoltaic"
    assert (rx.conversion_efficiency, rx.grid_efficiency, rx.max_input_power_W) == (0.4, 0.97, 1e12)
    set_path(mvp_data, "receiver.max_heat_rejection_W", 2.0e3)
    rx2 = receiver_model_from_config(parse_config(mvp_data).receiver)
    assert rx2.heat_limited_input_W == pytest.approx(2.0e3 / 0.6, rel=1e-15)
    assert RX.heat_limited_input_W == math.inf
    with pytest.raises(ValueError, match="conversion_efficiency"):
        ReceiverModel("rectenna", 1.2, 0.9, 1.0)
