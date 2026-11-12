"""Tests for helioswarm.solar (solar irradiance, incidence, PV efficiency law; PDF §8.1-8.2, §8.5).

PDF §13.1 "solar: 1/r^2 flux scaling, incidence and eclipse limits, efficiency bounds".
Reference values are closed forms written out here (the au is the exact IAU 2012 value, typed
here rather than imported, so a wrong module constant is caught).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from helioswarm.config import parse_config
from helioswarm.solar import (
    CollectorModel,
    PvEfficiencyLaw,
    absorbed_solar_power_W,
    collector_model_from_config,
    incidence_cosine,
    incident_solar_power_W,
    pv_efficiency,
    solar_flux_W_m2,
    solar_power,
)

from config_helpers import set_path

AU = 149_597_870_700.0  # m, IAU 2012 Resolution B2 (exact)


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


def _collector(**kw: Any) -> CollectorModel:
    base: dict[str, Any] = {
        "area_m2": 10_000.0,
        "absorptivity": 0.9,
        "efficiency_law": _law(),
        "incidence_mode": "sun_tracking",
        "fixed_normal_unit_j2000": None,
        "max_operating_temperature_K": 400.0,
        "solar_constant_1au_W_m2": 1361.0,
    }
    base.update(kw)
    return CollectorModel(**base)


# ---------------------------------------------------------------------------------------------
# Irradiance
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("s1", [1361.0, 1360.8, 1500.0])
def test_flux_at_1au_equals_configured_constant_exactly(s1: float) -> None:
    # (AU/AU)^2 = 1 exactly in floating point, so the configured value must come back bit for
    # bit; a hard-coded constant (1361) fails for the other two values.
    assert solar_flux_W_m2(AU, s1) == s1
    res = solar_power([AU], 1.0, 1.0, 300.0, _collector(solar_constant_1au_W_m2=s1))
    assert res.flux_W_m2[0] == s1
    # Normal incidence, full Sun: intercepted power is S_1AU * A exactly.
    assert res.incident_W[0] == s1 * 10_000.0


def test_flux_inverse_square_scaling() -> None:
    r_au = np.linspace(0.3, 1.2, 37)
    flux = solar_flux_W_m2(r_au * AU, 1361.0)
    # S r^2 is constant; 4e-15 relative = a few float64 roundings of (AU/r)^2.
    np.testing.assert_allclose(flux * r_au**2, 1361.0, rtol=4e-15, atol=0)
    assert solar_flux_W_m2(0.5 * AU, 1361.0) == pytest.approx(4.0 * 1361.0, rel=1e-15, abs=0)
    assert solar_flux_W_m2(0.3 * AU, 1361.0) == pytest.approx(1361.0 / 0.09, rel=1e-15, abs=0)


def test_flux_rejects_non_positive_distance() -> None:
    with pytest.raises(ValueError, match="> 0"):
        solar_flux_W_m2([AU, 0.0], 1361.0)
    with pytest.raises(ValueError, match="finite"):
        solar_flux_W_m2([math.nan], 1361.0)


def test_collected_power_scales_as_inverse_square_at_fixed_area_and_efficiency() -> None:
    # PDF §8.5: at fixed area and efficiency (fixed cell temperature -> fixed eta) the
    # electrical power scales as 1/r^2.
    r_au = np.array([0.3, 0.5, 0.8, 1.0, 1.2])
    res = solar_power(r_au * AU, 1.0, 1.0, 320.0, _collector())
    eta = 0.30 * (1.0 - 0.0004 * 20.0)  # eta_ref (1 + gamma (T - T_ref)) at T = 320 K
    expected = 1361.0 * (1.0 / r_au**2) * 10_000.0 * 0.9 * eta
    np.testing.assert_allclose(res.pv_electric_W, expected, rtol=1e-14, atol=0)
    np.testing.assert_allclose(res.pv_electric_W * r_au**2, expected[3], rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------------------------
# Incidence and illumination
# ---------------------------------------------------------------------------------------------


def test_sun_tracking_cosine_is_one_for_any_sun_direction() -> None:
    s = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.6, 0.8]])
    np.testing.assert_array_equal(incidence_cosine(_collector(), s), [1.0, 1.0, 1.0])


def test_fixed_normal_cosine_and_back_side_gives_zero() -> None:
    n = (0.0, 0.0, 1.0)
    col = _collector(incidence_mode="fixed_normal", fixed_normal_unit_j2000=n)
    theta = np.radians([0.0, 30.0, 60.0, 89.0, 90.0, 120.0, 180.0])
    s = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
    cos_i = incidence_cosine(col, s)
    expected = np.array([1.0, math.cos(math.radians(30)), 0.5, math.cos(math.radians(89))])
    np.testing.assert_allclose(cos_i[:4], expected, rtol=0, atol=1e-15)
    # 90 deg (grazing) and behind the panel: zero, never negative.
    np.testing.assert_allclose(cos_i[4:], 0.0, rtol=0, atol=1e-16)
    assert np.all(cos_i >= 0.0)
    res = solar_power(np.full(theta.size, AU), cos_i, 1.0, 300.0, col)
    np.testing.assert_allclose(res.pv_electric_W[4:], 0.0, atol=1e-9)
    assert not res.illuminated[-1]
    assert res.illuminated[0]


def test_fixed_normal_cosine_uses_the_configured_normal_not_an_axis() -> None:
    n = np.array([1.0, 2.0, -2.0]) / 3.0
    col = _collector(incidence_mode="fixed_normal", fixed_normal_unit_j2000=tuple(n))
    s = np.array([[0.0, 0.6, -0.8]])
    # n . s = (0 + 1.2 + 1.6) / 3
    assert incidence_cosine(col, s)[0] == pytest.approx(2.8 / 3.0, rel=1e-15)


def test_incidence_rejects_non_unit_sun_vectors() -> None:
    with pytest.raises(ValueError, match="unit"):
        incidence_cosine(_collector(), np.array([[2.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        incidence_cosine(_collector(), np.array([1.0, 0.0, 0.0]))


def test_zero_incidence_and_eclipse_give_zero_collected_power() -> None:
    col = _collector()
    cases = {"zero incidence": (0.0, 1.0), "umbra": (1.0, 0.0), "both": (0.0, 0.0)}
    for name, (cos_i, frac) in cases.items():
        res = solar_power([0.7 * AU], cos_i, frac, 300.0, col)
        for attr in ("incident_W", "absorbed_W", "reflected_W", "pv_electric_W", "pv_heat_W"):
            assert getattr(res, attr)[0] == 0.0, (name, attr)
        assert not res.illuminated[0], name
        # The flux at the satellite's distance is still reported (it is not a shadow model).
        assert res.flux_W_m2[0] == pytest.approx(1361.0 / 0.49, rel=1e-15)


def test_penumbra_fraction_and_incidence_scale_linearly() -> None:
    col = _collector()
    full = solar_power([AU], 1.0, 1.0, 300.0, col)
    part = solar_power([AU], 0.5, 0.37, 300.0, col)
    assert part.absorbed_W[0] == pytest.approx(0.5 * 0.37 * full.absorbed_W[0], rel=1e-15)
    assert part.illuminated[0]


@pytest.mark.parametrize(("cos_i", "frac"), [(1.1, 1.0), (-0.1, 1.0), (1.0, 1.2), (1.0, -0.2)])
def test_out_of_range_incidence_or_illumination_rejected(cos_i: float, frac: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        incident_solar_power_W(AU, cos_i, frac, _collector())


# ---------------------------------------------------------------------------------------------
# Absorbed / reflected split
# ---------------------------------------------------------------------------------------------


def test_absorbed_and_reflected_split_of_incident_light() -> None:
    col = _collector(absorptivity=0.75)
    res = solar_power([0.5 * AU], 1.0, 1.0, 300.0, col)
    incident = 4.0 * 1361.0 * 10_000.0
    assert res.incident_W[0] == pytest.approx(incident, rel=1e-15)
    assert res.absorbed_W[0] == pytest.approx(0.75 * incident, rel=1e-15)
    assert res.reflected_W[0] == pytest.approx(0.25 * incident, rel=1e-15)
    # Heat in the PV array is absorbed minus electric -- the reflected part is not in it.
    eta = 0.30
    assert res.pv_heat_W[0] == pytest.approx(0.75 * incident * (1.0 - eta), rel=1e-14)
    assert absorbed_solar_power_W(0.5 * AU, 1.0, 1.0, col) == res.absorbed_W[0]


# ---------------------------------------------------------------------------------------------
# PV efficiency law (PDF §8.2)
# ---------------------------------------------------------------------------------------------


def test_efficiency_linear_inside_and_clipped_outside() -> None:
    law = _law(temperature_coefficient_per_K=-0.004, minimum=0.05, maximum=0.45)
    t = np.array([50.0, 200.0, 300.0, 350.0, 600.0])
    eta = pv_efficiency(t, law)
    # Linear law 0.3 (1 - 0.004 (T - 300)): 0.6 at 50 K -> clipped to 0.45; 0.42 at 200 K;
    # 0.3 at T_ref; 0.24 at 350 K; -0.06 at 600 K -> clipped to 0.05.
    np.testing.assert_allclose(eta, [0.45, 0.42, 0.30, 0.24, 0.05], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(law.is_clipped(t), [True, False, False, False, True])
    lo, hi = law.linear_range_K()
    # eta = 0.45 at T = 300 - (0.45/0.3 - 1)/0.004 = 175 K; eta = 0.05 at 300 + (1 - 1/6)/0.004.
    assert lo == pytest.approx(175.0, rel=1e-14)
    assert hi == pytest.approx(300.0 + (1.0 - 0.05 / 0.3) / 0.004, rel=1e-14)


def test_efficiency_never_outside_bounds_or_above_one() -> None:
    rng = np.random.default_rng(20261112)
    t = rng.uniform(1.0, 3000.0, 2000)
    for _ in range(200):
        lo = rng.uniform(0.0, 0.5)
        hi = rng.uniform(lo + 1e-3, 1.0)
        law = PvEfficiencyLaw(
            reference=rng.uniform(lo, hi),
            reference_temperature_K=rng.uniform(200.0, 400.0),
            temperature_coefficient_per_K=rng.uniform(-0.01, 0.01),
            minimum=lo,
            maximum=hi,
        )
        eta = law.efficiency(t)
        assert np.all(eta >= lo)
        assert np.all(eta <= hi)
        assert np.all(eta <= 1.0)
    # Defaults (0, 1): a strongly negative coefficient at high T cannot go negative.
    assert pv_efficiency(5000.0, _law(temperature_coefficient_per_K=-0.01)) == 0.0


def test_constant_law_has_unbounded_linear_range() -> None:
    assert _law(temperature_coefficient_per_K=0.0).linear_range_K() == (-math.inf, math.inf)


@pytest.mark.parametrize(
    ("kw", "fragment"),
    [
        ({"minimum": 0.5, "maximum": 0.4}, "minimum < maximum"),
        ({"maximum": 1.1}, "maximum <= 1"),
        ({"minimum": -0.1}, "0 <= minimum"),
        ({"reference": 0.5, "maximum": 0.4}, "outside"),
        ({"reference_temperature_K": 0.0}, "reference_temperature_K"),
        ({"temperature_coefficient_per_K": math.nan}, "finite"),
    ],
)
def test_invalid_efficiency_law_rejected(kw: dict[str, float], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        _law(**kw)


def test_result_reports_efficiency_and_clip_flag() -> None:
    col = _collector(efficiency_law=_law(temperature_coefficient_per_K=-0.004, minimum=0.05))
    res = solar_power([AU, AU], 1.0, 1.0, [300.0, 700.0], col)
    np.testing.assert_allclose(res.pv_efficiency, [0.30, 0.05], rtol=1e-14)
    np.testing.assert_array_equal(res.pv_efficiency_clipped, [False, True])
    assert res.pv_electric_W[1] == pytest.approx(0.05 * res.absorbed_W[1], rel=1e-15)


# ---------------------------------------------------------------------------------------------
# Configuration mapping and result immutability
# ---------------------------------------------------------------------------------------------


def test_collector_model_from_config(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "constants.solar_constant_1au_W_m2", 1360.8)
    set_path(mvp_data, "collector.max_operating_temperature_K", 380.0)
    set_path(mvp_data, "collector.pv_efficiency_min", 0.02)
    set_path(mvp_data, "collector.pv_efficiency_max", 0.4)
    col = collector_model_from_config(parse_config(mvp_data))
    assert col.solar_constant_1au_W_m2 == 1360.8
    assert col.area_m2 == 10_000.0
    assert col.absorptivity == 0.9
    assert col.max_operating_temperature_K == 380.0
    assert col.incidence_mode == "sun_tracking"
    assert col.efficiency_law == PvEfficiencyLaw(0.30, 300.0, -0.0004, 0.02, 0.4)


def test_fixed_normal_from_config(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "collector.incidence_mode", "fixed_normal")
    set_path(mvp_data, "collector.fixed_normal_unit_j2000", [0.0, 0.6, 0.8])
    col = collector_model_from_config(parse_config(mvp_data))
    assert col.fixed_normal_unit_j2000 == (0.0, 0.6, 0.8)


def test_collector_model_validation() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        _collector(incidence_mode="fixed_normal", fixed_normal_unit_j2000=(1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="requires"):
        _collector(incidence_mode="fixed_normal")
    with pytest.raises(ValueError, match="absorptivity"):
        _collector(absorptivity=0.0)


def test_solar_result_is_immutable() -> None:
    res = solar_power([AU], 1.0, 1.0, 300.0, _collector())
    with pytest.raises(ValueError, match="read-only"):
        res.pv_electric_W[0] = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.flux_W_m2 = np.zeros(1)  # type: ignore[misc]
