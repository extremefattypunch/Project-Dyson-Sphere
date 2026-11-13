"""Tests for helioswarm.power_chain (on-board part of the PDF §10.1 stage chain).

PDF §13.1 "link budget: power never increases through passive stages; all efficiencies lie in
[0, 1]" -- for the on-board stages that M4 owns. Expected values are written out by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from helioswarm.config import parse_config
from helioswarm.power_chain import (
    OnboardEfficiencies,
    onboard_efficiencies_from_config,
    onboard_power_chain,
)

from config_helpers import set_path


def test_stage_powers_and_losses_by_hand() -> None:
    eff = OnboardEfficiencies(power_conditioning=0.9, electrical_to_beam=0.5, optics_or_rf=0.8)
    chain = onboard_power_chain([1000.0, 0.0], eff)
    # 1000 W -> 900 W conditioned -> 450 W beam -> 360 W out of the aperture.
    np.testing.assert_allclose(chain.dc_to_tx_W, [900.0, 0.0], rtol=1e-15)
    np.testing.assert_allclose(chain.beam_W, [450.0, 0.0], rtol=1e-15)
    np.testing.assert_allclose(chain.aperture_W, [360.0, 0.0], rtol=1e-15)
    # Losses: 100 W conditioning, 450 W conversion, 90 W optics; each counted exactly once.
    np.testing.assert_allclose(chain.power_conditioning_loss_W, [100.0, 0.0], rtol=1e-13)
    np.testing.assert_allclose(chain.electrical_to_beam_loss_W, [450.0, 0.0], rtol=1e-13)
    np.testing.assert_allclose(chain.optics_or_rf_loss_W, [90.0, 0.0], rtol=1e-13)
    np.testing.assert_allclose(chain.total_loss_W, [640.0, 0.0], rtol=1e-13)
    np.testing.assert_allclose(chain.total_loss_W + chain.aperture_W, chain.pv_electric_W)
    assert eff.dc_to_aperture == pytest.approx(0.36, rel=1e-15)


def test_passive_stages_never_increase_power() -> None:
    rng = np.random.default_rng(7)
    p = rng.uniform(0.0, 1e9, 500)
    for _ in range(50):
        eff = OnboardEfficiencies(*rng.uniform(1e-3, 1.0, 3))
        c = onboard_power_chain(p, eff)
        assert np.all(c.dc_to_tx_W <= c.pv_electric_W)
        assert np.all(c.beam_W <= c.dc_to_tx_W)
        assert np.all(c.aperture_W <= c.beam_W)
        for loss in (c.power_conditioning_loss_W, c.electrical_to_beam_loss_W):
            assert np.all(loss >= 0.0)
        assert np.all(c.optics_or_rf_loss_W >= 0.0)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0 + 1e-12, float("nan")])
def test_efficiencies_outside_unit_interval_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        OnboardEfficiencies(power_conditioning=bad, electrical_to_beam=0.5, optics_or_rf=1.0)


def test_negative_input_power_rejected() -> None:
    eff = OnboardEfficiencies(1.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        onboard_power_chain([-1.0], eff)


def test_efficiencies_from_config_have_one_owner_each(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "transmitter.power_conditioning_efficiency", 0.93)
    set_path(mvp_data, "transmitter.optics_or_rf_efficiency", 0.81)
    eff = onboard_efficiencies_from_config(parse_config(mvp_data).transmitter)
    assert eff == OnboardEfficiencies(0.93, 0.50, 0.81)


def test_chain_arrays_are_read_only() -> None:
    c = onboard_power_chain([1.0], OnboardEfficiencies(1.0, 0.5, 1.0))
    with pytest.raises(ValueError, match="read-only"):
        c.beam_W[0] = 3.0
