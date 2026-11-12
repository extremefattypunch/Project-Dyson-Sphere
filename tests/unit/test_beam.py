"""Tests of beam evaluation along a trajectory and the BeamResult contract (M5)."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from scipy import special, stats

from helioswarm.aperture import ApertureField, BeamModelValidityError
from helioswarm.beam import (
    BeamSettings,
    beam_settings_from_config,
    evaluate_beam,
    make_beam_pattern,
)
from helioswarm.config import parse_config

from config_helpers import microwave_data, set_path

AU_M = 149_597_870_700.0


def _airy_ee(x: np.ndarray) -> np.ndarray:
    return 1 - special.j0(x) ** 2 - special.j1(x) ** 2


def test_mvp_settings_and_capture_closed_form(mvp_data: dict[str, Any]) -> None:
    s = beam_settings_from_config(parse_config(mvp_data))
    assert (s.mode, s.beam_model, s.receiver_orientation) == (
        "optical",
        "airy_far_field",
        "horizontal",
    )
    assert s.wavefront_efficiency == 1.0
    assert s.lattice_main_lobe_fraction == 1.0
    # Perfect pointing, receiver tracking: capture = 1 - J0^2 - J1^2 at x = pi D a/(lambda R).
    s0 = dataclasses.replace(s, pointing_sigma_rad=0.0, receiver_orientation="tracking")
    r = np.array([0.3, 0.5, 1.0, 1.7]) * AU_M
    res = evaluate_beam(r, np.radians([90.0] * 4), s0)
    x = math.pi * 10.0 * 1000.0 / (1e-6 * r)
    np.testing.assert_allclose(res.capture_fraction, _airy_ee(x), rtol=1e-13)
    np.testing.assert_allclose(res.capture_fraction_nominal, _airy_ee(x), rtol=1e-13)
    np.testing.assert_allclose(res.pointing_loss, 1.0, rtol=1e-15)
    np.testing.assert_allclose(res.fresnel_number, 100.0 / (1e-6 * r), rtol=1e-15)
    assert res.fresnel_number_limit == 0.5
    np.testing.assert_allclose(
        res.diagnostics.first_null_radius_m, 1.2196698912665045e-7 * r, rtol=1e-12
    )
    assert res.scalar_diffraction is True
    assert res.coherent_combining is False


def test_horizontal_receiver_projection_and_pointing(mvp_data: dict[str, Any]) -> None:
    s = beam_settings_from_config(parse_config(mvp_data))  # sigma = 1e-9 rad, horizontal
    r = np.full(3, 0.5 * AU_M)
    el = np.radians([90.0, 30.0, 10.0])
    res = evaluate_beam(r, el, s)
    a_eff = 1000.0 * np.sqrt(np.sin(el))
    np.testing.assert_allclose(res.projected_receiver_radius_m, a_eff, rtol=1e-15)
    x = math.pi * 10.0 * a_eff / (1e-6 * r)
    np.testing.assert_allclose(res.capture_fraction_nominal, _airy_ee(x), rtol=1e-13)
    # Small receiver and small jitter (s = R sigma = 75 m << lambda R/D = 7.5 km): to leading
    # order E[eta] = eta(0) * (1 - <d^2> kappa'), with <d^2> = 2 s^2 and the Airy curvature
    # kappa' = (pi D/(lambda R))^2 / 4 -> loss 2 s^2 kappa'.
    s_m = 1e-9 * r
    kappa = (math.pi * 10.0 / (1e-6 * r)) ** 2 / 4
    np.testing.assert_allclose(1 - res.pointing_loss, 2 * s_m**2 * kappa, rtol=0.02)
    assert np.all(res.capture_fraction < res.capture_fraction_nominal)
    assert np.all(res.capture_fraction_p05 <= res.capture_fraction_p50)
    assert np.all(res.capture_fraction_p50 <= res.capture_fraction_p95)
    assert set(res.diagnostics.percentile_method) == {"exact_monotone"}
    # Larger jitter cannot help in this centred baseline.
    worse = evaluate_beam(r, el, dataclasses.replace(s, pointing_sigma_rad=5e-8))
    assert np.all(worse.capture_fraction < res.capture_fraction)


def test_wavefront_factor_applied_exactly_once(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "transmitter.rms_wavefront_error_m", 0.05e-6)
    s = beam_settings_from_config(parse_config(mvp_data))
    eta_wf = math.exp(-((2 * math.pi * 0.05e-6 / 1e-6) ** 2))
    perfect = dataclasses.replace(s, rms_wavefront_error_m=0.0)
    r, el = np.array([AU_M]), np.radians([60.0])
    a = evaluate_beam(r, el, s)
    b = evaluate_beam(r, el, perfect)
    assert a.wavefront_efficiency == pytest.approx(eta_wf, rel=1e-15)
    for name in ("capture_fraction", "capture_fraction_nominal", "capture_fraction_p50"):
        np.testing.assert_allclose(getattr(a, name), eta_wf * getattr(b, name), rtol=1e-14)
    # Pointing loss is a ratio and does not contain the wavefront factor.
    np.testing.assert_allclose(a.pointing_loss, b.pointing_loss, rtol=1e-14)
    # Peak irradiance is the aberration-free upper bound (not reduced).
    np.testing.assert_array_equal(a.peak_irradiance_per_W_m2, b.peak_irradiance_per_W_m2)


def test_peak_irradiance_and_power_density_constraint(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "receiver.max_power_density_W_m2", 1000.0)
    s = beam_settings_from_config(parse_config(mvp_data))
    r = np.array([0.2, 1.0]) * AU_M
    p_in = np.array([1.0e6, 1.0e6])
    res = evaluate_beam(r, np.radians([45.0, 45.0]), s, incident_power_W=p_in)
    peak = math.pi * 10.0**2 / (4 * (1e-6 * r) ** 2)  # pi D^2 / (4 lambda^2 R^2)
    np.testing.assert_allclose(res.peak_irradiance_per_W_m2, peak, rtol=1e-14)
    assert res.peak_irradiance_W_m2 is not None
    np.testing.assert_allclose(res.peak_irradiance_W_m2, p_in * peak, rtol=1e-14)
    # 1 MW at 0.2 au: 1e6 * 8.8e-8 = 0.088 W/m^2 -> below 1000 W/m^2.
    assert res.exceeds_max_power_density is not None
    assert not np.any(res.exceeds_max_power_density)
    hot = evaluate_beam(r, np.radians([45.0, 45.0]), s, incident_power_W=np.array([2e10, 1e6]))
    assert hot.exceeds_max_power_density is not None
    assert hot.exceeds_max_power_density.tolist() == [True, False]
    plain = evaluate_beam(r, np.radians([45.0, 45.0]), s)
    assert plain.peak_irradiance_W_m2 is None
    assert plain.exceeds_max_power_density is None


def test_far_field_model_outside_validity_raises(mvp_data: dict[str, Any]) -> None:
    s = beam_settings_from_config(parse_config(mvp_data))
    # D = 10 m, lambda = 1 um: far-field distance 2 D^2/lambda = 2e8 m; 1e6 m is N_F = 100.
    with pytest.raises(BeamModelValidityError, match="fresnel_hankel"):
        evaluate_beam(np.array([AU_M, 1.0e6]), np.radians([45.0, 45.0]), s)
    # The same epochs with the Fresnel-Hankel model are accepted (a 20 m tracking receiver,
    # so that the near-field quadrature stays small).
    fh = dataclasses.replace(
        s,
        beam_model="fresnel_hankel",
        pointing_sigma_rad=0.0,
        receiver_orientation="tracking",
        receiver_radius_m=20.0,
    )
    res = evaluate_beam(np.array([AU_M, 1.0e6]), np.radians([45.0, 45.0]), fh)
    assert res.fresnel_number[1] == pytest.approx(100.0)
    assert res.fresnel_number_limit is None
    # Near field (N_F = 100): the ~5 m beam (geometric shadow of the 10 m aperture plus edge
    # fringes of width ~sqrt(lambda z) = 1 m) falls inside the 20 m receiver.
    assert res.capture_fraction[1] > 0.99


def test_fresnel_model_matches_airy_in_far_field(mvp_data: dict[str, Any]) -> None:
    s = beam_settings_from_config(parse_config(mvp_data))
    fh = dataclasses.replace(s, beam_model="fresnel_hankel")
    r = np.array([0.5, 1.0]) * AU_M
    el = np.radians([40.0, 70.0])
    a = evaluate_beam(r, el, s)
    b = evaluate_beam(r, el, fh)
    # N_F ~ 1e-3: Fresnel corrections (pi N_F)^2/192 ~ 1e-7 relative.
    np.testing.assert_allclose(b.capture_fraction, a.capture_fraction, rtol=1e-6)


def test_gaussian_beam_settings(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "transmitter.aperture_illumination", "truncated_gaussian")
    set_path(mvp_data, "transmitter.gaussian_waist_radius_m", 2.0)  # a/w0 = 2.5
    set_path(mvp_data, "models.beam", "gaussian_beam")
    s = beam_settings_from_config(parse_config(mvp_data))
    s0 = dataclasses.replace(s, pointing_sigma_rad=0.0, receiver_orientation="tracking")
    r = np.array([AU_M])
    res = evaluate_beam(r, np.radians([90.0]), s0)
    w = 2.0 * math.sqrt(1 + (AU_M / (math.pi * 4.0 / 1e-6)) ** 2)
    assert res.diagnostics.gaussian_beam_radius_m[0] == pytest.approx(w, rel=1e-14)
    assert res.capture_fraction[0] == pytest.approx(1 - math.exp(-2 * 1000.0**2 / w**2), rel=1e-12)
    assert res.diagnostics.source_spillover_fraction == pytest.approx(
        math.exp(-2 * 2.5**2), rel=1e-14
    )
    # Jitter: exact Gaussian-beam expectation 1 - exp(-2 a^2 / (w^2 + 4 s^2)).
    sj = dataclasses.replace(s0, pointing_sigma_rad=1e-8)
    rj = evaluate_beam(r, np.radians([90.0]), sj)
    sm = 1e-8 * AU_M
    assert rj.capture_fraction[0] == pytest.approx(
        float(stats.ncx2.cdf(1000.0**2 / (w * w / 4 + sm * sm), 2, 0.0)), rel=1e-9
    )
    with pytest.raises(BeamModelValidityError, match="a/w0"):
        dataclasses.replace(
            s, aperture=ApertureField("truncated_gaussian", 10.0, gaussian_waist_radius_m=4.0)
        )


def test_microwave_lattice_and_impossible_beam(mvp_data: dict[str, Any]) -> None:
    data = microwave_data(mvp_data)  # 2.45 GHz, D = 1 km, 10 dB parabolic taper, d = 6 cm < lambda
    s = beam_settings_from_config(parse_config(data))
    assert s.lattice is not None
    assert s.lattice_main_lobe_fraction == 1.0
    r = np.array([AU_M])
    res = evaluate_beam(r, np.radians([60.0]), s)
    assert res.diagnostics.lattice_element_count > 2e8
    assert res.lattice_main_lobe_fraction == 1.0
    # Capture at 1 au: x = pi D a_eff / (lambda R) ~ 1.6e-4 -> taper_eff * x^2/4 ~ 6e-9.
    lam = 299_792_458.0 / 2.45e9
    a_eff = 1000.0 * math.sqrt(math.sin(math.radians(60.0)))
    x = math.pi * 1000.0 * a_eff / (lam * AU_M)
    tap = res.diagnostics.taper_efficiency
    assert res.capture_fraction_nominal[0] == pytest.approx(tap * x * x / 4, rel=1e-6)
    # Deliberately impossible beam (PDF §13.2): 1 m aperture -> capture ~ 1e-14 (not tuned away).
    set_path(data, "transmitter.aperture_diameter_m", 1.0)
    set_path(data, "transmitter.array_element_spacing_m", None)
    set_path(data, "transmitter.array_element_pattern_exponent", None)
    s1 = beam_settings_from_config(parse_config(data))
    res1 = evaluate_beam(r, np.radians([60.0]), s1)
    assert res1.capture_fraction[0] < 1e-13
    assert res1.capture_fraction[0] * 1e9 < 1e-4  # 1 GW at the aperture -> < 0.1 mW captured


def test_sparse_lattice_main_lobe_factor(mvp_data: dict[str, Any]) -> None:
    data = microwave_data(mvp_data)
    lam = 299_792_458.0 / 2.45e9
    set_path(data, "transmitter.array_element_spacing_m", 1.6 * lam)
    set_path(data, "transmitter.array_element_pattern_exponent", 1.0)
    s = beam_settings_from_config(parse_config(data))
    assert s.lattice_main_lobe_fraction == pytest.approx(1 / 9, rel=1e-12)
    dense = dataclasses.replace(s, lattice=None)
    r, el = np.array([AU_M]), np.radians([60.0])
    np.testing.assert_allclose(
        evaluate_beam(r, el, s).capture_fraction,
        evaluate_beam(r, el, dense).capture_fraction / 9,
        rtol=1e-13,
    )


def test_beam_result_contract(mvp_data: dict[str, Any]) -> None:
    s = beam_settings_from_config(parse_config(mvp_data))
    res = evaluate_beam(np.array([AU_M]), np.radians([45.0]), s, epochs_tdb_s=np.array([0.0]))
    with pytest.raises(ValueError, match="coherent"):
        dataclasses.replace(res, coherent_combining=True)
    with pytest.raises(ValueError, match="scalar"):
        dataclasses.replace(res, scalar_diffraction=False)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        dataclasses.replace(res, capture_fraction=np.array([1.1]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.beam_model = "gaussian_beam"  # type: ignore[misc]
    assert not res.capture_fraction.flags.writeable
    assert any("scalar diffraction" in n for n in res.validity_notes)
    with pytest.raises(ValueError, match="equal length"):
        evaluate_beam(np.array([AU_M, AU_M]), np.radians([45.0]), s)


def test_settings_reject_inconsistent_models() -> None:
    ap = ApertureField("uniform_circular", 10.0)
    common: dict[str, Any] = {
        "wavelength_m": 1e-6,
        "aperture": ap,
        "receiver_radius_m": 1000.0,
        "receiver_orientation": "horizontal",
        "pointing_bias_rad": 0.0,
        "pointing_sigma_rad": 0.0,
    }
    with pytest.raises(ValueError, match="not allowed"):
        BeamSettings(mode="optical", beam_model="aperture_far_field", **common)
    with pytest.raises(ValueError, match="uniform"):
        BeamSettings(
            mode="optical",
            beam_model="airy_far_field",
            **(
                common
                | {"aperture": ApertureField("parabolic_on_pedestal", 10.0, edge_taper_dB=10.0)}
            ),
        )
    s = BeamSettings(mode="optical", beam_model="fresnel_hankel", **common)
    pat = make_beam_pattern(s, AU_M, 1e4)
    assert pat.model == "fresnel_hankel"


def test_beam_evaluation_is_fast_enough(mvp_data: dict[str, Any]) -> None:
    # Vectorised over time for M6: 500 epochs of the MVP (Airy + jitter) well under 5 s.
    import time

    s = beam_settings_from_config(parse_config(mvp_data))
    rng = np.random.default_rng(5)
    r = rng.uniform(0.2, 1.9, 500) * AU_M
    el = np.radians(rng.uniform(10.0, 90.0, 500))
    t0 = time.perf_counter()
    res = evaluate_beam(r, el, s)
    assert time.perf_counter() - t0 < 5.0
    assert res.capture_fraction.shape == (500,)


def test_receiver_projection_error_diagnostic(mvp_data: dict[str, Any]) -> None:
    # Exact centred ellipse a x a sin(el) by an independent polar trapezoid of the closed-form
    # Airy EE, vs the equal-area circle used by the model; the diagnostic must report the
    # relative difference (small in the MVP regime, large when the receiver ~ beam).
    s = beam_settings_from_config(parse_config(mvp_data))
    s0 = dataclasses.replace(s, pointing_sigma_rad=0.0)
    for r_m, el_deg, small in ((AU_M, 10.0, True), (1.0e10, 10.0, False)):  # a/ell = 0.07, 1
        res = evaluate_beam(np.array([r_m]), np.radians([el_deg]), s0)
        a, b = 1000.0, 1000.0 * math.sin(math.radians(el_deg))
        psi = 2 * math.pi * np.arange(4096) / 4096
        rr = a * b / np.sqrt((b * np.cos(psi)) ** 2 + (a * np.sin(psi)) ** 2)
        ellipse = float(np.mean(_airy_ee(math.pi * 10.0 * rr / (1e-6 * r_m))))
        expected = (res.capture_fraction_nominal[0] - ellipse) / ellipse
        assert res.diagnostics.receiver_projection_error[0] == pytest.approx(expected, rel=1e-6)
        if small:
            assert 0 < expected < 3e-3
        else:
            assert expected > 0.05
    track = evaluate_beam(
        np.array([AU_M]),
        np.radians([10.0]),
        dataclasses.replace(s0, receiver_orientation="tracking"),
    )
    assert track.diagnostics.receiver_projection_error[0] == 0.0


# ---------------------------------------------------------------------------------------------
# Review M5: pointing bias, configured validity thresholds, tapered peak irradiance
# ---------------------------------------------------------------------------------------------


def _gaussian_beam_settings(mvp_data: dict[str, Any], bias: float, sigma: float) -> BeamSettings:
    set_path(mvp_data, "transmitter.aperture_illumination", "truncated_gaussian")
    set_path(mvp_data, "transmitter.gaussian_waist_radius_m", 2.0)  # a/w0 = 2.5
    set_path(mvp_data, "models.beam", "gaussian_beam")
    set_path(mvp_data, "transmitter.pointing_bias_rad", bias)
    set_path(mvp_data, "transmitter.pointing_sigma_rad", sigma)
    s = beam_settings_from_config(parse_config(mvp_data))
    assert (s.pointing_bias_rad, s.pointing_sigma_rad) == (bias, sigma)
    return dataclasses.replace(s, receiver_orientation="tracking")


def test_pointing_bias_wiring_gaussian_beam_closed_form(mvp_data: dict[str, Any]) -> None:
    # The beam position at the receiver is a 2-D normal vector of mean nu = R * bias and per-axis
    # std s = R * sigma; the Gaussian beam is a 2-D normal of per-axis variance w^2/4, so
    # (independent closed forms, non-central chi-square with 2 dof):
    #   bias only:    eta = ncx2.cdf(a^2/v, 2, nu^2/v),           v = w^2/4
    #   bias+jitter:  E[eta] = ncx2.cdf(a^2/v', 2, nu^2/v'),      v' = w^2/4 + s^2.
    # bias and sigma are given different values, so dropping the bias, omitting the factor R,
    # doubling it or swapping it with sigma changes the result far beyond the 1e-9 tolerance
    # (quadrature accuracy of the Rice expectation).
    bias, sigma = 5.0e-8, 3.0e-8
    r_m, a = AU_M, 1000.0
    w = 2.0 * math.sqrt(1 + (r_m / (math.pi * 4.0 / 1e-6)) ** 2)
    v = w * w / 4
    nu, sm = r_m * bias, r_m * sigma
    s_bias = _gaussian_beam_settings(mvp_data, bias, 0.0)
    res = evaluate_beam(np.array([r_m]), np.radians([90.0]), s_bias)
    ref_bias = float(stats.ncx2.cdf(a * a / v, 2, nu * nu / v))
    ref_nominal = -math.expm1(-2 * a * a / (w * w))
    assert res.capture_fraction_bias_only[0] == pytest.approx(ref_bias, rel=1e-11)
    assert res.capture_fraction[0] == pytest.approx(ref_bias, rel=1e-11)  # sigma = 0
    assert res.capture_fraction_nominal[0] == pytest.approx(ref_nominal, rel=1e-12)
    assert res.capture_fraction_bias_only[0] < 0.9 * res.capture_fraction_nominal[0]
    assert res.pointing_loss[0] == pytest.approx(ref_bias / ref_nominal, rel=1e-11)
    assert res.pointing_bias_rad == bias
    both = evaluate_beam(
        np.array([r_m]),
        np.radians([90.0]),
        dataclasses.replace(s_bias, pointing_sigma_rad=sigma),
    )
    vj = v + sm * sm
    ref_both = float(stats.ncx2.cdf(a * a / vj, 2, nu * nu / vj))
    assert both.capture_fraction[0] == pytest.approx(ref_both, rel=1e-9)
    assert both.capture_fraction_bias_only[0] == pytest.approx(ref_bias, rel=1e-11)
    swapped = float(stats.ncx2.cdf(a * a / (v + nu * nu), 2, sm * sm / (v + nu * nu)))
    assert abs(swapped - ref_both) > 1e-3 * ref_both  # the test can tell bias from sigma


def test_configured_fresnel_number_limit_is_used(mvp_data: dict[str, Any]) -> None:
    # Review M5: a stricter models.fraunhofer_max_fresnel_number must reach the hard check.
    # D = 10 m, lambda = 1 um: N_F(1 au) = 100/(1e-6 * 1.496e11) = 6.7e-4 > 1e-4.
    set_path(mvp_data, "models.fraunhofer_max_fresnel_number", 1.0e-4)
    s = beam_settings_from_config(parse_config(mvp_data))
    assert s.max_fresnel_number == 1.0e-4
    with pytest.raises(BeamModelValidityError, match=r"fraunhofer_max_fresnel_number = 0\.0001"):
        evaluate_beam(np.array([AU_M]), np.radians([60.0]), s)
    ok = evaluate_beam(np.array([10.0 * AU_M]), np.radians([60.0]), s)  # N_F = 6.7e-5
    assert ok.fresnel_number_limit == 1.0e-4


def test_configured_gaussian_truncation_limit_is_used(mvp_data: dict[str, Any]) -> None:
    # Review M5: models.gaussian_beam_min_truncation_ratio = 3 (non-default) must be carried;
    # a/w0 = 2.7 passes the default 2.5 but must be refused under 3.
    set_path(mvp_data, "transmitter.aperture_illumination", "truncated_gaussian")
    set_path(mvp_data, "transmitter.gaussian_waist_radius_m", 5.0 / 3.5)  # a/w0 = 3.5
    set_path(mvp_data, "models.beam", "gaussian_beam")
    set_path(mvp_data, "models.gaussian_beam_min_truncation_ratio", 3.0)
    s = beam_settings_from_config(parse_config(mvp_data))
    assert s.gaussian_beam_min_truncation_ratio == 3.0
    with pytest.raises(BeamModelValidityError, match="a/w0"):
        dataclasses.replace(
            s, aperture=ApertureField("truncated_gaussian", 10.0, gaussian_waist_radius_m=5 / 2.7)
        )


def test_microwave_peak_irradiance_includes_taper(mvp_data: dict[str, Any]) -> None:
    # aperture_far_field (default microwave model, 10 dB parabolic-on-pedestal taper): the peak
    # irradiance for the power-density check is P * eta_taper * pi D^2/(4 lambda^2 R^2), with
    # eta_taper written out from the closed form of the pedestal illumination (Balanis §15.4):
    # eta_taper = (C + (1-C)/2)^2 / (C + (1-C)^2/3), C = 10^(-10/20).
    data = microwave_data(mvp_data)
    set_path(data, "receiver.max_power_density_W_m2", 1000.0)
    s = beam_settings_from_config(parse_config(data))
    assert s.beam_model == "aperture_far_field"
    c_edge = 10 ** (-10.0 / 20)
    eta_t = (c_edge + (1 - c_edge) / 2) ** 2 / (c_edge + (1 - c_edge) ** 2 / 3)
    assert s.aperture.taper_efficiency == pytest.approx(eta_t, rel=1e-14)
    lam = 299_792_458.0 / 2.45e9
    r = np.array([0.5, 1.0]) * AU_M
    p_in = np.array([1.0e9, 1.0e9])
    res = evaluate_beam(r, np.radians([60.0, 60.0]), s, incident_power_W=p_in)
    peak = eta_t * math.pi * 1000.0**2 / (4 * lam**2 * r**2)
    np.testing.assert_allclose(res.peak_irradiance_per_W_m2, peak, rtol=1e-12)
    assert res.peak_irradiance_W_m2 is not None
    np.testing.assert_allclose(res.peak_irradiance_W_m2, p_in * peak, rtol=1e-12)
    assert res.exceeds_max_power_density is not None
    assert not np.any(res.exceeds_max_power_density)  # ~1e-5 W/m^2 at 0.5 au
