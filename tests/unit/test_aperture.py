"""Tests of the aperture fields and the scalar Fresnel/Fraunhofer propagation core (M5).

Reference values are closed forms written out in the tests (or independent ``scipy.integrate``
quadrature), never the module's own helpers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, special

from helioswarm.aperture import (
    FRESNEL_PARAXIAL_MAX_PHASE_RAD,
    ApertureField,
    BeamModelValidityError,
    FarFieldAperturePattern,
    FresnelHankelPattern,
    aperture_from_config,
    check_far_field_validity,
    fresnel_number,
    fresnel_paraxial_parameter,
    ruze_efficiency,
    wavefront_efficiency,
)
from helioswarm.config import parse_config

from config_helpers import microwave_data

D_M = 10.0
LAM_M = 1.0e-6

APERTURES = [
    ApertureField("uniform_circular", D_M),
    ApertureField("truncated_gaussian", D_M, gaussian_waist_radius_m=D_M / 2 / 1.0),
    ApertureField("truncated_gaussian", D_M, gaussian_waist_radius_m=D_M / 2 / 2.5),
    ApertureField("parabolic_on_pedestal", D_M, edge_taper_dB=10.0),
    ApertureField("parabolic_on_pedestal", D_M, edge_taper_dB=30.0),
]
IDS = ["uniform", "gauss_r1", "gauss_r2.5", "parabolic10dB", "parabolic30dB"]


# ---------------------------------------------------------------------------------------------
# Aperture field normalisation and diagnostics
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ap", APERTURES, ids=IDS)
def test_aperture_power_is_one_watt(ap: ApertureField) -> None:
    # Independent adaptive quadrature of |A|^2 2 pi r over [0, a]; tolerance 1e-12 = quad's
    # requested accuracy (the closed-form normalisation is exact).
    val, _ = integrate.quad(
        lambda r: float(ap.amplitude_m_inv(r) ** 2) * 2 * math.pi * r,
        0.0,
        ap.radius_m,
        epsabs=0.0,
        epsrel=1e-13,
        limit=200,
    )
    assert val == pytest.approx(1.0, rel=1e-12)
    # Zero outside the edge: power outside is not emitted (spill-over owned upstream).
    assert ap.amplitude_m_inv(ap.radius_m * 1.0000001) == 0.0


def test_normalisation_constants_closed_form() -> None:
    a = D_M / 2
    assert ApertureField("uniform_circular", D_M).normalisation_m_inv == pytest.approx(
        1 / math.sqrt(math.pi * a * a), rel=1e-15
    )
    w0 = 2.0
    c2 = 1.0 / ((math.pi * w0**2 / 2) * (1 - math.exp(-2 * a**2 / w0**2)))
    g = ApertureField("truncated_gaussian", D_M, gaussian_waist_radius_m=w0)
    assert g.normalisation_m_inv**2 == pytest.approx(c2, rel=1e-14)
    taper_db = 12.0
    c_edge = 10 ** (-taper_db / 20)
    p = ApertureField("parabolic_on_pedestal", D_M, edge_taper_dB=taper_db)
    assert p.pedestal == pytest.approx(c_edge, rel=1e-15)
    # Edge *intensity* taper of 12 dB: |A(a)|^2/|A(0)|^2 = 10^-1.2.
    assert p.edge_intensity_ratio == pytest.approx(10 ** (-1.2), rel=1e-14)
    edge_over_centre = float(p.amplitude_m_inv(a) / p.amplitude_m_inv(0.0)) ** 2
    assert edge_over_centre == pytest.approx(10 ** (-1.2), rel=1e-13)


@pytest.mark.parametrize("ap", APERTURES, ids=IDS)
def test_taper_efficiency_matches_quadrature(ap: ApertureField) -> None:
    # eta_t = |int A dA|^2 / (pi a^2) for a unit-power field (Balanis §15.4); quad reference.
    mean, _ = integrate.quad(
        lambda r: float(ap.amplitude_m_inv(r)) * 2 * math.pi * r, 0.0, ap.radius_m, epsrel=1e-13
    )
    assert ap.amplitude_integral_m == pytest.approx(mean, rel=1e-12)
    assert ap.taper_efficiency == pytest.approx(mean**2 / ap.area_m2, rel=1e-12)
    assert 0.0 < ap.taper_efficiency <= 1.0 + 1e-15
    if ap.illumination == "uniform_circular":
        assert ap.taper_efficiency == pytest.approx(1.0, rel=1e-15)
    else:
        assert ap.taper_efficiency < 0.99  # any taper lowers the on-axis gain


def test_gaussian_spillover_fraction_is_untruncated_power_outside_aperture() -> None:
    a, w0 = D_M / 2, D_M / 2 / 1.3
    ap = ApertureField("truncated_gaussian", D_M, gaussian_waist_radius_m=w0)
    # Untruncated TEM00 normalised to 1 W: I = 2/(pi w0^2) exp(-2 r^2/w0^2).
    inside, _ = integrate.quad(
        lambda r: 2 / (math.pi * w0**2) * math.exp(-2 * r * r / w0**2) * 2 * math.pi * r,
        0.0,
        a,
        epsrel=1e-13,
    )
    assert ap.source_spillover_fraction == pytest.approx(1 - inside, rel=1e-10)
    assert ApertureField("uniform_circular", D_M).source_spillover_fraction == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"illumination": "truncated_gaussian", "diameter_m": 2.0}, "waist"),
        (
            {
                "illumination": "truncated_gaussian",
                "diameter_m": 2.0,
                "gaussian_waist_radius_m": 1.01,
            },
            "exceeds",
        ),
        ({"illumination": "parabolic_on_pedestal", "diameter_m": 2.0}, "edge_taper"),
        ({"illumination": "uniform_circular", "diameter_m": 2.0, "edge_taper_dB": 3.0}, "only"),
        (
            {"illumination": "uniform_circular", "diameter_m": 2.0, "gaussian_waist_radius_m": 0.5},
            "only",
        ),
        ({"illumination": "uniform_circular", "diameter_m": -1.0}, "diameter"),
    ],
)
def test_aperture_field_rejects_inconsistent_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ApertureField(**kwargs)  # type: ignore[arg-type]


def test_aperture_from_config(mvp_data: dict[str, object]) -> None:
    ap = aperture_from_config(parse_config(mvp_data).transmitter)
    assert (ap.illumination, ap.diameter_m) == ("uniform_circular", 10.0)
    mw = aperture_from_config(parse_config(microwave_data(mvp_data)).transmitter)  # type: ignore[arg-type]
    assert (mw.illumination, mw.diameter_m, mw.edge_taper_dB) == (
        "parabolic_on_pedestal",
        1000.0,
        10.0,
    )


# ---------------------------------------------------------------------------------------------
# Far field (Fraunhofer) of the aperture fields
# ---------------------------------------------------------------------------------------------


def _closed_form_far_field(ap: ApertureField, theta: np.ndarray) -> np.ndarray:
    """Independent closed forms (module-free) of J(theta) [sr^-1]."""
    a = ap.radius_m
    u = 2 * math.pi * a * theta / LAM_M
    if ap.illumination == "uniform_circular":
        return (math.pi * a * a / LAM_M**2) * (2 * special.j1(u) / u) ** 2
    c_edge = 10 ** (-(ap.edge_taper_dB or 0) / 20)
    c2 = 1.0 / (math.pi * a * a * (c_edge + (1 - c_edge) ** 2 / 3))
    # Lommel: int_0^1 (1 - t^2) J0(u t) t dt = 2 J2(u)/u^2.
    amp = (
        2
        * math.pi
        * a
        * a
        * (c_edge * special.j1(u) / u + (1 - c_edge) * 2 * special.jv(2, u) / u**2)
    )
    return c2 * amp**2 / LAM_M**2


@pytest.mark.parametrize("ap", [APERTURES[0], APERTURES[3], APERTURES[4]], ids=IDS[0:1] + IDS[3:5])
def test_hankel_quadrature_reproduces_closed_form_far_fields(ap: ApertureField) -> None:
    theta = np.linspace(1e-3, 20.0, 400) * LAM_M / D_M
    ref = _closed_form_far_field(ap, theta)
    numeric = np.abs(ap.hankel_amplitude(theta, LAM_M, None)) ** 2
    closed = ap.radiant_intensity_sr_inv(theta, LAM_M)
    peak = ap.on_axis_radiant_intensity_factor / LAM_M**2
    # 16 Gauss-Legendre nodes per oscillation cycle: agreement at the float64 level.
    assert np.max(np.abs(numeric - ref)) < 1e-12 * peak
    assert np.max(np.abs(closed - ref)) < 1e-12 * peak
    # On axis: J(0) = |int A dA|^2 / lambda^2.
    assert float(ap.radiant_intensity_sr_inv(np.array([0.0]), LAM_M)[0]) == pytest.approx(
        peak, rel=1e-14
    )


def test_truncated_gaussian_far_field_against_scipy_quad() -> None:
    ap = APERTURES[1]  # a/w0 = 1: strongly truncated, no closed form
    a = ap.radius_m
    for theta in (0.0, 0.7 * LAM_M / D_M, 2.3 * LAM_M / D_M, 7.9 * LAM_M / D_M):
        k = 2 * math.pi * theta / LAM_M
        val, _ = integrate.quad(
            lambda r, k=k: float(ap.amplitude_m_inv(r)) * special.j0(k * r) * r,
            0.0,
            a,
            epsabs=1e-13 * ap.amplitude_integral_m / (2 * math.pi),
            epsrel=1e-12,
            limit=200,
        )
        ref = (2 * math.pi * val / LAM_M) ** 2
        got = float(ap.radiant_intensity_sr_inv(np.array([theta]), LAM_M)[0])
        # quad accuracy ~1e-12 of the amplitude integral -> ~1e-11 of the peak intensity.
        peak = ap.on_axis_radiant_intensity_factor / LAM_M**2
        assert got == pytest.approx(ref, rel=1e-10, abs=1e-11 * peak)


@pytest.mark.parametrize("range_m", [None, 2.0e6, 2.0e7])
def test_hankel_quadrature_converges_under_refinement(range_m: float | None) -> None:
    # Strongly truncated Gaussian, Fresnel (N_F = 50 and 5) and far field. Refinement x4 changes
    # the result by < 1e-12 of the peak; a deliberately under-resolved rule (refinement 0.02,
    # 2 panels) is visibly wrong, so the test can fail.
    ap = APERTURES[1]
    z = 1.0 if range_m is None else range_m
    rho = np.linspace(0.0, 3.0 * D_M if range_m else 6 * LAM_M / D_M, 301)
    base = np.abs(ap.hankel_amplitude(rho, LAM_M, range_m)) ** 2
    fine = np.abs(ap.hankel_amplitude(rho, LAM_M, range_m, refinement=4.0)) ** 2
    peak = float(np.max(fine))
    assert np.max(np.abs(base - fine)) < 1e-12 * peak
    if range_m == 2.0e6:  # N_F = 50: many Fresnel zones, so 2 panels cannot resolve them
        coarse = np.abs(ap.hankel_amplitude(rho, LAM_M, range_m, refinement=0.02)) ** 2
        assert np.max(np.abs(coarse - fine)) > 1e-3 * peak
    assert z > 0


# ---------------------------------------------------------------------------------------------
# Fresnel propagation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_f", [0.3, 1.0, 3.9, 4.0, 8.0, 21.7])
def test_fresnel_on_axis_intensity_closed_form(n_f: float) -> None:
    # Uniform aperture, on axis (Goodman 2005, §4.5.1 / problem 4-17): |U(0)|^2 = (4/(pi a^2))
    # sin^2(phi/2) with phi = pi a^2/(lambda z) = pi N_F / 4. Zero at N_F = 8 (two Fresnel
    # zones), four times the geometric irradiance at N_F = 4 (one zone): exact near-field test.
    ap = APERTURES[0]
    z = D_M**2 / (LAM_M * n_f)
    pat = FresnelHankelPattern(ap, LAM_M, z, paraxial_rho_max_m=D_M)
    phi = math.pi * n_f / 4
    ref = 4 / (math.pi * ap.radius_m**2) * math.sin(phi / 2) ** 2
    got = float(pat.intensity_m2_inv(np.array([0.0]))[0])
    assert got == pytest.approx(ref, rel=1e-11, abs=1e-12 * 4 / (math.pi * ap.radius_m**2))


@pytest.mark.parametrize("n_f", [1e-2, 1e-3, 1e-4])
def test_fresnel_reduces_to_airy_in_far_field(n_f: float) -> None:
    # Fraunhofer limit of the Fresnel-Hankel integral = Airy. The deviation is second order in
    # the aperture phase pi N_F/4 (on axis exactly 1 - sinc^2(pi N_F/8) ~ (pi N_F)^2/192).
    ap = APERTURES[0]
    z = D_M**2 / (LAM_M * n_f)
    fres = FresnelHankelPattern(ap, LAM_M, z, paraxial_rho_max_m=10 * LAM_M * z / D_M)
    ff = FarFieldAperturePattern(ap, LAM_M, z, max_fresnel_number=0.5)
    rho = np.linspace(0.0, 6.0, 241) * LAM_M * z / D_M
    i0 = ff.peak_intensity_m2_inv()
    dev = np.max(np.abs(fres.intensity_m2_inv(rho) - ff.intensity_m2_inv(rho))) / i0
    bound = (math.pi * n_f) ** 2 / 192
    assert dev <= 1.5 * bound
    assert dev >= 0.5 * bound  # the deviation is really there (second-order, not zero)


def test_fresnel_pattern_conserves_power_near_field() -> None:
    # Weakly truncated Gaussian (a/w0 = 4, edge intensity e^-32 = 1.3e-14: edge-diffraction
    # power scattered beyond 2 D is below 1e-12) at N_F = 20: integral of |U|^2 over the plane
    # equals 1 (Parseval) to 1e-11. (With a/w0 = 3 the edge-diffracted tail beyond 2 D is
    # ~1.5e-9, consistent with the e^-18 edge intensity.)
    ap = ApertureField("truncated_gaussian", D_M, gaussian_waist_radius_m=D_M / 8)
    z = D_M**2 / (LAM_M * 20.0)
    pat = FresnelHankelPattern(ap, LAM_M, z, paraxial_rho_max_m=2 * D_M)
    total = float(pat.encircled_energy(np.array([2.0 * D_M]))[0])
    assert total == pytest.approx(1.0, abs=1e-11)


def test_fresnel_peak_finds_near_field_maximum() -> None:
    # N_F = 4 (one Fresnel zone): the on-axis irradiance 4/(pi a^2) is the global maximum.
    ap = APERTURES[0]
    z = D_M**2 / (LAM_M * 4.0)
    pat = FresnelHankelPattern(ap, LAM_M, z, paraxial_rho_max_m=D_M)
    assert pat.peak_intensity_m2_inv() == pytest.approx(4 / (math.pi * ap.radius_m**2), rel=1e-6)
    # N_F = 8: on-axis zero, the maximum is off axis and above the geometric irradiance.
    pat8 = FresnelHankelPattern(ap, LAM_M, z / 2, paraxial_rho_max_m=D_M)
    assert float(pat8.intensity_m2_inv(np.array([0.0]))[0]) < 1e-12 / ap.area_m2
    assert pat8.peak_intensity_m2_inv() > 1.0 / ap.area_m2


# ---------------------------------------------------------------------------------------------
# Validity of the propagation regime
# ---------------------------------------------------------------------------------------------


def test_fresnel_number_definition() -> None:
    assert float(fresnel_number(10.0, 1e-6, 1e8)) == pytest.approx(1.0, rel=1e-15)
    r = np.array([1e6, 2e6])
    np.testing.assert_allclose(fresnel_number(2.0, 0.5e-6, r), 4.0 / (0.5e-6 * r), rtol=1e-15)
    with pytest.raises(ValueError, match="range_m"):
        fresnel_number(1.0, 1e-6, 0.0)


def test_far_field_validity_check_is_a_hard_error() -> None:
    check_far_field_validity("airy_far_field", 0.5, 0.5)  # boundary accepted
    with pytest.raises(BeamModelValidityError, match="fresnel_hankel"):
        check_far_field_validity("airy_far_field", 0.5 * (1 + 1e-12), 0.5)
    ap = APERTURES[3]
    z_ff = D_M**2 / (LAM_M * 0.5)
    FarFieldAperturePattern(ap, LAM_M, z_ff, max_fresnel_number=0.5)
    with pytest.raises(BeamModelValidityError, match="Fresnel number"):
        FarFieldAperturePattern(ap, LAM_M, 0.99 * z_ff, max_fresnel_number=0.5)


def test_fresnel_paraxial_check() -> None:
    # pi (a + rho)^4 / (4 lambda z^3) with a = 5, rho = 5, lambda = 1e-6: equals 1 rad at
    # z^3 = pi 1e4 / 4e-6. It is the maximum neglected quartic phase k L^4/(8 z^3), L = a + rho.
    z1 = (math.pi * 10.0**4 / (4 * 1e-6)) ** (1 / 3)
    assert fresnel_paraxial_parameter(5.0, 5.0, 1e-6, z1) == pytest.approx(1.0, rel=1e-12)
    k = 2 * math.pi / 1e-6
    assert fresnel_paraxial_parameter(5.0, 5.0, 1e-6, z1) == pytest.approx(
        k * 10.0**4 / (8 * z1**3), rel=1e-12
    )
    # Review M5: Goodman's ">>" is given the assumed limit 0.1 rad (not 1 rad): the parameter
    # scales as z^-3, so the accepted boundary is z1 * 10^(1/3).
    assert FRESNEL_PARAXIAL_MAX_PHASE_RAD == 0.1
    z_lim = z1 * 10 ** (1 / 3)
    FresnelHankelPattern(APERTURES[0], 1e-6, z_lim * 1.001, paraxial_rho_max_m=5.0)
    for z in (z_lim * 0.999, z1 * 1.001):  # 1 rad (the former limit) is now rejected
        with pytest.raises(BeamModelValidityError, match="paraxial"):
            FresnelHankelPattern(APERTURES[0], 1e-6, z, paraxial_rho_max_m=5.0)


def test_fresnel_paraxial_check_applies_to_every_evaluation() -> None:
    # Review M5: radii beyond the declared paraxial_rho_max_m are re-checked when evaluated.
    z_lim = (math.pi * 10.0**4 / (4 * 1e-6 * FRESNEL_PARAXIAL_MAX_PHASE_RAD)) ** (1 / 3)
    pat = FresnelHankelPattern(APERTURES[0], 1e-6, z_lim * 1.001, paraxial_rho_max_m=5.0)
    pat.intensity_m2_inv(np.array([0.0, 4.0]))  # inside the checked region
    with pytest.raises(BeamModelValidityError, match="paraxial"):
        pat.intensity_m2_inv(np.array([0.0, 6.0]))  # (5 + 6)^4 / 10^4 = 1.46 x the limit
    # The peak search covers rho <= a + 4 lambda z / D, beyond a pattern declared with
    # paraxial_rho_max_m = 0 that sits just inside the limit for L = a: it must refuse rather
    # than evaluate outside the paraxial region.
    z_a = (math.pi * 5.0**4 / (4 * 1e-6 * FRESNEL_PARAXIAL_MAX_PHASE_RAD)) ** (1 / 3)
    edge = FresnelHankelPattern(APERTURES[0], 1e-6, z_a * 1.0001, paraxial_rho_max_m=0.0)
    with pytest.raises(BeamModelValidityError, match="paraxial"):
        edge.peak_intensity_m2_inv()


def test_far_field_encircled_energy_honours_refinement(monkeypatch: pytest.MonkeyPatch) -> None:
    # Review M5: quadrature_refinement must reach the radial EE quadrature of
    # aperture_far_field (it feeds the inner-disk term of every capture evaluation).
    import helioswarm.aperture as aperture_module

    seen: list[float] = []
    real = aperture_module.numerical_encircled_energy

    def spy(pattern: object, rho_m: object, *, refinement: float = 1.0) -> np.ndarray:
        seen.append(refinement)
        return real(pattern, rho_m, refinement=refinement)  # type: ignore[arg-type]

    ap = APERTURES[3]
    rho = np.array([0.3, 1.0, 2.5]) * LAM_M * 1e9 / D_M
    base = FarFieldAperturePattern(ap, LAM_M, 1e9, 0.5).encircled_energy(rho)
    monkeypatch.setattr(aperture_module, "numerical_encircled_energy", spy)
    fine = FarFieldAperturePattern(ap, LAM_M, 1e9, 0.5, refinement=3.0).encircled_energy(rho)
    assert seen == [3.0]
    # Converged: 8-node panels of <= lambda R/(2 D); refinement x3 changes EE by < 1e-13.
    np.testing.assert_allclose(fine, base, rtol=0, atol=1e-13)


def _taper_efficiency_quad(ap: ApertureField) -> float:
    """|int A dA|^2 / (pi a^2 int |A|^2 dA) by independent adaptive quadrature."""
    a = ap.radius_m
    first, _ = integrate.quad(
        lambda r: float(ap.amplitude_m_inv(r)) * 2 * math.pi * r, 0.0, a, epsrel=1e-13
    )
    second, _ = integrate.quad(
        lambda r: float(ap.amplitude_m_inv(r)) ** 2 * 2 * math.pi * r, 0.0, a, epsrel=1e-13
    )
    return first**2 / (math.pi * a * a * second)


@pytest.mark.parametrize("ap", [APERTURES[1], APERTURES[3], APERTURES[4]], ids=[IDS[1], *IDS[3:5]])
def test_tapered_far_field_peak_irradiance(ap: ApertureField) -> None:
    # Review M5: the peak irradiance feeding the power-density check for aperture_far_field
    # (default microwave model) must carry the taper: I_peak = eta_taper pi D^2/(4 lambda^2 R^2)
    # (Balanis, Antenna Theory, §15.4: directivity = eta_taper (pi D/lambda)^2), with eta_taper
    # from independent quadrature (1e-10: quad accuracy). It must also be the pattern maximum.
    r_m = 1.0e9
    pat = FarFieldAperturePattern(ap, LAM_M, r_m, 0.5)
    eta_t = _taper_efficiency_quad(ap)
    assert eta_t < 0.99  # genuinely tapered: a missing taper factor would be detected
    ref = eta_t * math.pi * D_M**2 / (4 * LAM_M**2 * r_m**2)
    assert pat.peak_intensity_m2_inv() == pytest.approx(ref, rel=1e-10)
    rho = np.linspace(0.0, 8.0, 801) * LAM_M * r_m / D_M
    vals = pat.intensity_m2_inv(rho)
    assert float(np.max(vals)) == pytest.approx(pat.peak_intensity_m2_inv(), rel=1e-12)
    assert int(np.argmax(vals)) == 0


def _edge_intensity_closed_form(ap: ApertureField) -> float:
    """|A(a)|^2 [m^-2] from the closed-form normalisations (module docstring of aperture.py)."""
    a = ap.radius_m
    if ap.illumination == "uniform_circular":
        return 1.0 / (math.pi * a * a)
    if ap.illumination == "parabolic_on_pedestal":
        c_edge = 10 ** (-(ap.edge_taper_dB or 0.0) / 20)
        return c_edge**2 / (math.pi * a * a * (c_edge + (1 - c_edge) ** 2 / 3))
    w0 = ap.gaussian_waist_radius_m or 0.0
    c2 = 1.0 / ((math.pi * w0 * w0 / 2) * -math.expm1(-2 * a * a / (w0 * w0)))
    return c2 * math.exp(-2 * a * a / (w0 * w0))


@pytest.mark.parametrize("ap", [APERTURES[0], APERTURES[1], APERTURES[3]], ids=[*IDS[0:2], IDS[3]])
def test_far_field_total_power_is_one(ap: ApertureField) -> None:
    # PDF §13.1 "total power normalisation of each pattern" (review M5). A field with an edge
    # discontinuity A(a) has, for large u, int_0^a A J0(k r) r dr ~ A(a) a J1(k a)/k, so the
    # power beyond x = pi D rho/(lambda R) is 2 a^2 |A(a)|^2 / x + O(x^-2) (uniform: 2/(pi x),
    # the known tail of 1 - J0^2 - J1^2). EE(x = 1e3) + tail must be 1 within 1e-6 (measured
    # residuals <= 1.2e-7). The tail itself is 1e-4 to 6e-4, so dropping it, or a pattern
    # normalisation error above 1e-6, fails.
    r_m = 1.0e9
    pat = FarFieldAperturePattern(ap, LAM_M, r_m, 0.5)
    x = 1.0e3
    rho = x * LAM_M * r_m / (math.pi * D_M)
    ee = float(pat.encircled_energy(np.array([rho]))[0])
    tail = 2 * ap.radius_m**2 * _edge_intensity_closed_form(ap) / x
    assert tail > 1e-4
    assert ee + tail == pytest.approx(1.0, abs=1e-6)


def test_uniform_fresnel_total_power_is_one() -> None:
    # Uniform aperture at N_F = 20 (near field): beyond the geometric shadow the field is the
    # edge-diffraction tail, whose power beyond rho is 2/(pi x) (1 + O((a/rho)^2)) with
    # x = 2 pi a rho/(lambda z) (same edge term as the far field, the chirp only shifting the
    # local frequency by a/rho). At rho = 300 m = 60 a: tail 3.4e-4, correction ~1e-7.
    ap = APERTURES[0]
    z = D_M**2 / (LAM_M * 20.0)
    rho = 300.0
    pat = FresnelHankelPattern(ap, LAM_M, z, paraxial_rho_max_m=rho)
    ee = float(pat.encircled_energy(np.array([rho]))[0])
    x = 2 * math.pi * ap.radius_m * rho / (LAM_M * z)
    assert ee + 2 / (math.pi * x) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------------------------
# Wavefront error (single owner)
# ---------------------------------------------------------------------------------------------


def test_ruze_and_wavefront_efficiency() -> None:
    assert ruze_efficiency(0.0) == 1.0
    assert ruze_efficiency(0.3) == pytest.approx(math.exp(-0.09), rel=1e-15)
    # sigma_w = lambda/20 -> sigma_phi = pi/10 -> exp(-pi^2/100) = 0.906...
    assert wavefront_efficiency(0.05e-6, 1e-6) == pytest.approx(
        math.exp(-(math.pi**2) / 100), rel=1e-14
    )
    # Wavefront (not surface) error: no factor 2 (M0 definition).
    assert wavefront_efficiency(0.05e-6, 1e-6) != pytest.approx(math.exp(-4 * math.pi**2 / 100))
    # Microwave: 1 mm RMS at 2.45 GHz (lambda = 0.1224 m).
    lam = 299_792_458.0 / 2.45e9
    assert wavefront_efficiency(1e-3, lam) == pytest.approx(
        math.exp(-((2 * math.pi * 1e-3 / lam) ** 2)), rel=1e-15
    )
    with pytest.raises(ValueError, match="rms_wavefront_error_m"):
        wavefront_efficiency(-1e-9, 1e-6)
    with pytest.raises(ValueError, match="phase_rms_rad"):
        ruze_efficiency(float("nan"))
