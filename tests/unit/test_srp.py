"""Tests for helioswarm.srp (radial cannonball / Sun-facing flat-plate SRP, PDF §6.2-6.3).

PDF §13.1 "SRP: correct inverse-square scaling and direction away from Sun for the simple
model". Reference values are closed forms computed here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helioswarm.config import load_config
from helioswarm.constants import AU_M, GM_SUN_M3_S2
from helioswarm.propagate import force_model_from_config
from helioswarm.srp import (
    SrpModel,
    srp_acceleration_m_s2,
    srp_along_track_drift_estimate_m,
    srp_material_area_to_mass_m2_kg,
    srp_pressure_1au_N_m2,
)

from config_helpers import MVP_CONFIG

C = 299_792_458.0  # m/s, exact (SI); written here so the test does not import the module value


def test_pressure_at_1au_is_irradiance_over_c() -> None:
    assert srp_pressure_1au_N_m2(1361.0) == pytest.approx(1361.0 / C, rel=1e-15, abs=0)
    assert srp_pressure_1au_N_m2(1361.0) == pytest.approx(4.5398e-6, rel=1e-4, abs=0)
    with pytest.raises(ValueError, match="> 0"):
        srp_pressure_1au_N_m2(0.0)


def _model(cr: float = 1.3, am: float = 0.02, s: float = 1361.0) -> SrpModel:
    return SrpModel.from_solar_constant(cr=cr, area_to_mass_m2_kg=am, solar_constant_1au_W_m2=s)


def test_magnitude_at_1au_and_direction_away_from_an_offset_sun() -> None:
    # The Sun is NOT at the origin, so a model that used |r| from the origin (or the SSB)
    # instead of |r - r_sun| fails both checks.
    sun = np.array([7.0e8, -3.0e8, 1.0e8])
    u = np.array([0.6, -0.8, 0.0])
    sat = sun + AU_M * u
    a = _model().acceleration_m_s2(sat, sun)
    expected = (1361.0 / C) * 1.3 * 0.02  # P * Cr * A/m at exactly 1 au
    assert np.linalg.norm(a) == pytest.approx(expected, rel=1e-14, abs=0)
    np.testing.assert_allclose(a / np.linalg.norm(a), u, rtol=0, atol=1e-15)  # anti-sunward
    assert a @ (sat - sun) > 0


@pytest.mark.parametrize("r_au", [0.3, 0.5, 2.0, 5.2])
def test_inverse_square_scaling(r_au: float) -> None:
    m = _model()
    u = np.array([0.0, 0.0, 1.0])
    a1 = np.linalg.norm(m.acceleration_m_s2(AU_M * u, np.zeros(3)))
    ar = np.linalg.norm(m.acceleration_m_s2(r_au * AU_M * u, np.zeros(3)))
    assert ar / a1 == pytest.approx(1.0 / r_au**2, rel=1e-14, abs=0)


def test_linear_in_cr_area_to_mass_and_irradiance() -> None:
    r = np.array([AU_M, 0.0, 0.0])
    base = np.linalg.norm(_model(1.0, 0.01, 1361.0).acceleration_m_s2(r, np.zeros(3)))
    assert np.linalg.norm(_model(2.0, 0.01).acceleration_m_s2(r, np.zeros(3))) == pytest.approx(
        2 * base, rel=1e-15, abs=0
    )
    assert np.linalg.norm(_model(1.0, 0.05).acceleration_m_s2(r, np.zeros(3))) == pytest.approx(
        5 * base, rel=1e-14, abs=0
    )
    assert np.linalg.norm(
        _model(1.0, 0.01, 1500.0).acceleration_m_s2(r, np.zeros(3))
    ) == pytest.approx(base * 1500.0 / 1361.0, rel=1e-14, abs=0)


def test_vectorised_matches_scalar_and_strength_definition() -> None:
    m = _model()
    rng = np.random.default_rng(2)
    sats = rng.normal(size=(50, 3)) * AU_M
    sun = np.array([1e9, 2e8, -3e8])
    vec = m.acceleration_m_s2(sats, sun)
    for i in range(50):
        np.testing.assert_array_equal(vec[i], m.acceleration_m_s2(sats[i], sun))
    # k = P AU^2 Cr A/m, and the lightness number beta = k / GM_sun (r-independent).
    k = (1361.0 / C) * AU_M**2 * 1.3 * 0.02
    assert m.strength_m3_s2 == pytest.approx(k, rel=1e-15, abs=0)
    assert m.lightness_number(GM_SUN_M3_S2) == pytest.approx(k / GM_SUN_M3_S2, rel=1e-15, abs=0)
    # MVP (Cr 1.3, A/m 0.02): beta = 2.0e-5, i.e. SRP is 0.002 % of solar gravity.
    assert m.lightness_number(GM_SUN_M3_S2) == pytest.approx(1.99e-5, rel=1e-2, abs=0)
    d = sats[0] - sun
    np.testing.assert_allclose(
        srp_acceleration_m_s2(sats[0], sun, k), k * d / np.linalg.norm(d) ** 3, rtol=1e-15, atol=0
    )


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"cr": 0.9, "area_to_mass_m2_kg": 0.01, "pressure_1au_N_m2": 4.5e-6}, "cr"),
        ({"cr": 2.1, "area_to_mass_m2_kg": 0.01, "pressure_1au_N_m2": 4.5e-6}, "cr"),
        ({"cr": 1.0, "area_to_mass_m2_kg": 0.0, "pressure_1au_N_m2": 4.5e-6}, "area_to_mass"),
        ({"cr": 1.0, "area_to_mass_m2_kg": 0.01, "pressure_1au_N_m2": -1.0}, "pressure"),
        ({"cr": math.nan, "area_to_mass_m2_kg": 0.01, "pressure_1au_N_m2": 1.0}, "cr"),
    ],
)
def test_model_validation(kwargs: dict[str, float], fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        SrpModel(**kwargs)


def test_srp_at_the_sun_centre_is_rejected() -> None:
    with pytest.raises(ValueError, match="centre"):
        _model().acceleration_m_s2(np.zeros(3), np.zeros(3))


def test_srp_model_from_mvp_config_uses_configured_constants() -> None:
    cfg = load_config(MVP_CONFIG)
    srp = force_model_from_config(cfg.orbit, cfg.constants).srp
    assert srp is not None
    assert srp.cr == 1.3
    assert srp.area_to_mass_m2_kg == 0.02
    assert srp.pressure_1au_N_m2 == pytest.approx(
        cfg.constants.solar_constant_1au_W_m2 / C, rel=1e-15, abs=0
    )
    off = cfg.orbit.model_copy(
        update={"include_solar_radiation_pressure": False, "cr": None, "area_to_mass_m2_kg": None}
    )
    assert force_model_from_config(off, cfg.constants).srp is None


def test_along_track_drift_estimate_closed_form_and_materiality_inverse() -> None:
    # Delta s = 2 beta n t a (derivation in the docstring; checked against the L1 integration
    # of the MVP orbit in test_l1_propagation). Independent arithmetic: one sidereal period of a
    # 1 au orbit (n t = 2 pi) gives 4 pi beta AU.
    n = math.sqrt(GM_SUN_M3_S2 / AU_M**3)
    period = 2.0 * math.pi / n
    got = srp_along_track_drift_estimate_m(
        1e-5, semimajor_axis_m=AU_M, duration_s=period, gm_sun_m3_s2=GM_SUN_M3_S2
    )
    assert got == pytest.approx(4.0 * math.pi * 1e-5 * AU_M, rel=1e-14)
    # Inverse: beta = (S/c) AU^2 Cr (A/m) / GM. With S = 1361 W/m^2, Cr = 1.3 and a 1 km
    # threshold over one period: A/m = 1e3 / (4 pi AU (S/c) AU^2 Cr / GM).
    am = srp_material_area_to_mass_m2_kg(
        1.0e3, cr=1.3, solar_constant_1au_W_m2=1361.0, semimajor_axis_m=AU_M,
        duration_s=period, gm_sun_m3_s2=GM_SUN_M3_S2,
    )  # fmt: skip
    beta_per_am = (1361.0 / C) * AU_M**2 * 1.3 / GM_SUN_M3_S2
    assert am == pytest.approx(1.0e3 / (4.0 * math.pi * AU_M * beta_per_am), rel=1e-13)
    # Round trip, and linearity in the threshold.
    beta = SrpModel.from_solar_constant(
        cr=1.3, area_to_mass_m2_kg=am, solar_constant_1au_W_m2=1361.0
    ).lightness_number(GM_SUN_M3_S2)
    back = srp_along_track_drift_estimate_m(
        beta, semimajor_axis_m=AU_M, duration_s=period, gm_sun_m3_s2=GM_SUN_M3_S2
    )
    assert back == pytest.approx(1.0e3, rel=1e-13)
    with pytest.raises(ValueError, match="threshold_m"):
        srp_material_area_to_mass_m2_kg(
            0.0, cr=1.3, solar_constant_1au_W_m2=1361.0, semimajor_axis_m=AU_M,
            duration_s=period, gm_sun_m3_s2=GM_SUN_M3_S2,
        )  # fmt: skip
    with pytest.raises(ValueError, match="lightness_number"):
        srp_along_track_drift_estimate_m(
            -1.0, semimajor_axis_m=AU_M, duration_s=1.0, gm_sun_m3_s2=GM_SUN_M3_S2
        )
