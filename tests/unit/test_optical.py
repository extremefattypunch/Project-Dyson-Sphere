"""Tests of the Airy far-field and Gaussian-beam optical models (M5, PDF §9.1-9.2, §13.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize, special

from helioswarm.aperture import (
    ApertureField,
    BeamModelValidityError,
    FarFieldAperturePattern,
    FresnelHankelPattern,
    numerical_encircled_energy,
)
from helioswarm.optical import (
    AIRY_FIRST_NULL_FACTOR,
    J1_FIRST_ZERO,
    AiryPattern,
    GaussianBeamPattern,
    airy_encircled_energy_x,
    airy_first_null_radius_m,
    gaussian_beam_radius_m,
    gaussian_rayleigh_range_m,
)

AU_M = 149_597_870_700.0


def test_first_zero_of_j1() -> None:
    assert pytest.approx(float(special.jn_zeros(1, 1)[0]), rel=1e-15) == J1_FIRST_ZERO
    # A&S Table 9.5: j_{1,1} = 3.83170 59702; 1.21967 = j_{1,1}/pi (to 5 decimals).
    assert pytest.approx(3.8317059702, abs=1e-10) == J1_FIRST_ZERO
    assert round(AIRY_FIRST_NULL_FACTOR, 5) == 1.21967


@pytest.mark.parametrize(
    ("diameter_m", "wavelength_m", "range_m"),
    [
        (10.0, 1e-6, AU_M),
        (20.0, 1e-6, AU_M),
        (10.0, 0.5e-6, AU_M),
        (10.0, 1e-6, 0.3 * AU_M),
        (2.0, 1.55e-6, 1e9),
    ],
)
def test_first_null_radius_scaling(diameter_m: float, wavelength_m: float, range_m: float) -> None:
    pat = AiryPattern(diameter_m, wavelength_m, range_m, max_fresnel_number=0.5)
    expected = 1.2196698912665045 * wavelength_m * range_m / diameter_m
    assert pat.first_null_radius_m == pytest.approx(expected, rel=1e-14)
    # The irradiance really vanishes there and the first minimum found numerically agrees.
    i0 = pat.peak_intensity_m2_inv()
    assert float(pat.intensity_m2_inv(np.array([expected]))[0]) < 1e-25 * i0
    res = optimize.minimize_scalar(
        lambda r: float(pat.intensity_m2_inv(np.array([r]))[0]) / i0,
        bounds=(0.8 * expected, 1.2 * expected),
        method="bounded",
        options={"xatol": 1e-9 * expected},
    )
    assert res.x == pytest.approx(expected, rel=1e-6)
    np.testing.assert_allclose(
        airy_first_null_radius_m(wavelength_m, np.array([range_m]), diameter_m),
        expected,
        rtol=1e-14,
    )


def test_airy_peak_irradiance_closed_form() -> None:
    d, lam, r = 10.0, 1.0e-6, AU_M
    pat = AiryPattern(d, lam, r, 0.5)
    assert pat.peak_intensity_m2_inv() == pytest.approx(
        math.pi * d**2 / (4 * lam**2 * r**2), rel=1e-15
    )
    assert float(pat.intensity_m2_inv(np.array([0.0]))[0]) == pytest.approx(
        pat.peak_intensity_m2_inv(), rel=1e-15
    )


def test_airy_pattern_integrates_to_one() -> None:
    # Numerical integral of I(rho) 2 pi rho out to x = pi D rho/(lambda R) = 1e6 (2.5e7 nodes is
    # avoided: 8 Gauss-Legendre nodes per half ring period => ~5e6 nodes). The power outside
    # x = 1e6 is J0^2 + J1^2 ~ 2/(pi x) = 6.4e-7, so the integral equals 1 within 1e-6 and
    # equals the closed-form EE(1e6) to 1e-9 (quadrature accuracy).
    pat = AiryPattern(10.0, 1.0e-6, 1.0e9, 0.5)
    x_max = 1.0e6
    rho_max = x_max * pat.feature_scale_m / math.pi
    total = float(numerical_encircled_energy(pat, np.array([rho_max]))[0])
    assert total == pytest.approx(1.0, abs=1e-6)
    assert total == pytest.approx(float(airy_encircled_energy_x(np.array([x_max]))[0]), abs=1e-9)


def test_encircled_energy_at_first_dark_ring() -> None:
    # Rayleigh: EE(j11) = 1 - J0(j11)^2 = 0.83777 (Born & Wolf §8.5.2: 83.8 % in the Airy disk).
    pat = AiryPattern(10.0, 1.0e-6, AU_M, 0.5)
    ee = float(pat.encircled_energy(np.array([pat.first_null_radius_m]))[0])
    assert ee == pytest.approx(1.0 - special.j0(3.8317059702075125) ** 2, rel=1e-14)
    assert ee == pytest.approx(0.8378, abs=5e-5)
    # Independent: numerical quadrature of the intensity, not the closed form.
    num = float(numerical_encircled_energy(pat, np.array([pat.first_null_radius_m]))[0])
    assert num == pytest.approx(ee, abs=1e-12)


def test_airy_encircled_energy_closed_form_vs_quadrature() -> None:
    pat = AiryPattern(10.0, 1.0e-6, 1.0e9, 0.5)
    rho = np.array([1e-6, 0.01, 0.3, 1.0, 1.7, 2.5, 7.3, 30.0]) * pat.feature_scale_m
    np.testing.assert_allclose(
        pat.encircled_energy(rho), numerical_encircled_energy(pat, rho), atol=1e-12
    )
    # Small-x series branch: continuous at x = 1e-4 and equal to x^2/4 - x^4/32 below it.
    x = np.array([1e-9, 1e-6, 0.99e-4, 1.01e-4])
    ee = airy_encircled_energy_x(x)
    np.testing.assert_allclose(ee[:3], x[:3] ** 2 / 4 - x[:3] ** 4 / 32, rtol=1e-15)
    assert ee[3] == pytest.approx(x[3] ** 2 / 4 - x[3] ** 4 / 32, rel=1e-6)


def test_uniform_far_field_pattern_equals_airy() -> None:
    d, lam, r = 7.0, 0.8e-6, 2.0e9
    airy = AiryPattern(d, lam, r, 0.5)
    ff = FarFieldAperturePattern(ApertureField("uniform_circular", d), lam, r, 0.5)
    rho = np.linspace(0.0, 10.0, 501) * airy.feature_scale_m
    np.testing.assert_allclose(
        ff.intensity_m2_inv(rho),
        airy.intensity_m2_inv(rho),
        rtol=1e-12,
        atol=1e-15 * airy.peak_intensity_m2_inv(),
    )
    assert ff.peak_intensity_m2_inv() == pytest.approx(airy.peak_intensity_m2_inv(), rel=1e-14)


def test_airy_outside_far_field_raises() -> None:
    # Near field: D = 10 m, lambda = 1 um, R = 1000 km -> N_F = 100 >> 0.5.
    with pytest.raises(BeamModelValidityError, match="N_F"):
        AiryPattern(10.0, 1.0e-6, 1.0e6, max_fresnel_number=0.5)
    # Exactly at the limit R = 2 D^2/lambda is accepted; 0.1 % closer is not.
    r_ff = 2 * 10.0**2 / 1.0e-6
    AiryPattern(10.0, 1.0e-6, r_ff, 0.5)
    with pytest.raises(BeamModelValidityError):
        AiryPattern(10.0, 1.0e-6, 0.999 * r_ff, 0.5)
    # The Fresnel-Hankel model is the valid alternative in the near field.
    FresnelHankelPattern(
        ApertureField("uniform_circular", 10.0), 1.0e-6, 1.0e6, paraxial_rho_max_m=20.0
    )


# ---------------------------------------------------------------------------------------------
# Gaussian beam (separate model)
# ---------------------------------------------------------------------------------------------


def test_gaussian_beam_radius() -> None:
    w0, lam = 2.0, 1.0e-6
    z_r = math.pi * w0**2 / lam
    assert gaussian_rayleigh_range_m(w0, lam) == pytest.approx(z_r, rel=1e-15)
    np.testing.assert_allclose(
        gaussian_beam_radius_m(w0, lam, np.array([0.0, z_r, 3 * z_r, 1e3 * z_r])),
        w0 * np.sqrt(1 + np.array([0.0, 1.0, 9.0, 1e6])),
        rtol=1e-15,
    )
    # Far field: w ~ lambda z / (pi w0), i.e. divergence half-angle lambda/(pi w0).
    z = 1e6 * z_r
    assert float(gaussian_beam_radius_m(w0, lam, z)) == pytest.approx(
        lam * z / (math.pi * w0), rel=1e-11
    )


def test_gaussian_beam_power_and_encircled_energy() -> None:
    pat = GaussianBeamPattern(waist_radius_m=1.0, wavelength_m=1.0e-6, range_m=1.0e7)
    w = pat.beam_radius_m
    assert pat.peak_intensity_m2_inv() == pytest.approx(2 / (math.pi * w * w), rel=1e-15)
    for a in (0.1 * w, w, 2.0 * w):
        assert float(pat.encircled_energy(np.array([a]))[0]) == pytest.approx(
            1 - math.exp(-2 * a * a / w / w), rel=1e-14
        )
    # 1 - e^-2 = 0.8647 of the power within w (Siegman 1986, §17.1).
    assert float(pat.encircled_energy(np.array([w]))[0]) == pytest.approx(0.864664716763, rel=1e-11)
    total = float(numerical_encircled_energy(pat, np.array([8.0 * w]))[0])
    assert total == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("n_f", [20.0, 1.0, 1e-3])
def test_gaussian_beam_agrees_with_fresnel_hankel_for_weak_truncation(n_f: float) -> None:
    # Comparison path of the Gaussian-beam approximation: the same truncated_gaussian field
    # propagated by Fresnel-Hankel. The truncation perturbs the field linearly in the edge
    # amplitude e^{-r^2} (r = a/w0; e.g. the far-field peak ratio (1-e^{-r^2})^2/(1-e^{-2r^2})
    # ~ 1 - 2 e^{-r^2}), so the encircled energies must agree to 2 e^{-r^2} (2.5e-4 at r = 3)
    # at all ranges (near and far field); with strong truncation (r = 1) they differ by > 1e-2.
    d = 6.0
    lam = 1.0e-6
    z = d**2 / (lam * n_f)
    for ratio, tol, should_agree in ((3.0, 2 * math.exp(-9.0), True), (1.0, 1e-2, False)):
        w0 = d / 2 / ratio
        gb = GaussianBeamPattern(w0, lam, z)
        fh = FresnelHankelPattern(
            ApertureField("truncated_gaussian", d, gaussian_waist_radius_m=w0),
            lam,
            z,
            paraxial_rho_max_m=4 * gb.beam_radius_m,
        )
        rho = np.array([0.5, 1.0, 1.5]) * gb.beam_radius_m
        diff = float(np.max(np.abs(gb.encircled_energy(rho) - fh.encircled_energy(rho))))
        if should_agree:
            assert diff < tol
        else:
            assert diff > tol
