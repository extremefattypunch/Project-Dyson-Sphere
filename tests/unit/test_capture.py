"""Tests of finite-receiver capture, receiver projection and pointing statistics (M5, PDF §9.4-9.5).

Strongest references: for a Gaussian beam the offset capture and the jitter expectation have
exact closed forms in the non-central chi-square distribution (a 2-D Gaussian integrated over a
disk), independent of the module's 1-D quadrature:

* beam ``I ∝ exp(-2 rho^2/w^2)`` is a 2-D normal density with per-axis variance ``w^2/4``;
  ``eta(d) = P(|X + d| < a) = ncx2.cdf(a^2/v, 2, d^2/v)`` with ``v = w^2/4``;
* with isotropic jitter ``s`` and bias ``nu``, the beam position adds an independent normal
  vector, so ``E[eta] = ncx2.cdf(a^2/v', 2, nu^2/v')`` with ``v' = w^2/4 + s^2``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize, special, stats

from helioswarm.aperture import ApertureField, FarFieldAperturePattern, FresnelHankelPattern
from helioswarm.capture import (
    arc_half_angle_rad,
    capture_fraction,
    capture_fraction_cartesian,
    capture_fraction_centred_ellipse,
    capture_fraction_ellipse_cartesian,
    capture_statistics,
    projected_receiver_radius_m,
    rice_offset_cdf,
    rice_offset_pdf_m_inv,
    rice_offset_quantile_m,
)
from helioswarm.optical import AiryPattern, GaussianBeamPattern

AU_M = 149_597_870_700.0
AIRY = AiryPattern(10.0, 1.0e-6, 1.0e9, 0.5)  # feature scale lambda R / D = 100 m
ELL = AIRY.feature_scale_m
GAUSS = GaussianBeamPattern(waist_radius_m=1.0, wavelength_m=1.0e-6, range_m=1.0e7)
W = GAUSS.beam_radius_m


def _gauss_capture_exact(a: float, d: float, w: float, s: float = 0.0) -> float:
    v = w * w / 4 + s * s
    return float(stats.ncx2.cdf(a * a / v, 2, d * d / v))


# ---------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------


def test_arc_half_angle_cases() -> None:
    a = 2.0
    assert float(arc_half_angle_rad(0.5, 1.0, a)) == pytest.approx(math.pi)  # rho < a - d
    assert float(arc_half_angle_rad(3.5, 1.0, a)) == 0.0  # rho > a + d
    assert float(arc_half_angle_rad(1.0, 0.0, a)) == pytest.approx(math.pi)  # centred, inside
    assert float(arc_half_angle_rad(2.5, 0.0, a)) == 0.0  # centred, outside
    # rho = d = a: cos(phi) = (a^2 + a^2 - a^2)/(2 a^2) = 1/2 -> phi = pi/3.
    assert float(arc_half_angle_rad(a, a, a)) == pytest.approx(math.pi / 3, rel=1e-15)
    # Continuity at the band ends.
    assert float(arc_half_angle_rad(1.0 + 1e-12, 1.0, a)) == pytest.approx(math.pi, abs=1e-5)
    assert float(arc_half_angle_rad(3.0 - 1e-12, 1.0, a)) == pytest.approx(0.0, abs=1e-5)
    # Monte-Carlo check of the arc fraction phi/pi for an off-centre point set.
    rng = np.random.default_rng(12345)
    psi = rng.uniform(0, 2 * math.pi, 400_000)
    rho, d = 1.7, 1.2
    inside = (rho * np.cos(psi) - d) ** 2 + (rho * np.sin(psi)) ** 2 <= a * a
    assert np.mean(inside) == pytest.approx(
        float(arc_half_angle_rad(rho, d, a)) / math.pi, abs=5 * 0.5 / math.sqrt(psi.size)
    )


@pytest.mark.parametrize("pattern", [AIRY, GAUSS], ids=["airy", "gauss"])
def test_centred_capture_equals_closed_form_encircled_energy(
    pattern: AiryPattern | GaussianBeamPattern,
) -> None:
    # Independent closed forms written out here (not the pattern's own encircled_energy):
    # Airy (Rayleigh) 1 - J0(x)^2 - J1(x)^2, x = pi D rho/(lambda R) = pi rho/ell;
    # Gaussian beam 1 - exp(-2 rho^2/w^2) (Siegman 1986, ch. 17). Tolerance 1e-13: both are
    # evaluated to float64 accuracy, the difference is rounding only.
    for a in (0.01, 0.4, 1.0, 1.21967, 3.0, 10.0):
        r = a * pattern.feature_scale_m
        got = float(capture_fraction(pattern, r, np.array([0.0]))[0])
        if isinstance(pattern, AiryPattern):
            x = math.pi * r / pattern.feature_scale_m
            ref = float(1 - special.j0(x) ** 2 - special.j1(x) ** 2)
        else:
            ref = -math.expm1(-2 * r * r / W**2)
        assert got == pytest.approx(ref, rel=1e-13, abs=1e-16)


def test_offset_capture_gaussian_beam_exact() -> None:
    # 1-D arc-length integral against the closed-form non-central chi-square (module docstring).
    for a in (0.05 * W, 0.7 * W, 2.0 * W):
        d = np.array([0.0, 0.1, 0.5, 0.99, 1.0, 1.01, 2.0, 4.0]) * max(a, W)
        got = capture_fraction(GAUSS, a, d)
        ref = np.array([_gauss_capture_exact(a, float(di), W) for di in d])
        np.testing.assert_allclose(got, ref, rtol=1e-11, atol=1e-14)


def test_offset_capture_converges_to_cartesian_grid() -> None:
    # PDF §9.4/§13.3: compare with a 2-D grid and show convergence under refinement. The grid
    # error is O(cell size) at the disk edge; it must shrink with n and end below 2e-5, and the
    # 1-D result itself must be converged (refinement x4 changes it by < 1e-13).
    a = 1.5 * ELL
    for d in (0.0, 0.3 * ELL, 1.5 * ELL, 3.0 * ELL):
        eta = float(capture_fraction(AIRY, a, np.array([d]))[0])
        eta_fine = float(capture_fraction(AIRY, a, np.array([d]), refinement=4.0)[0])
        assert eta == pytest.approx(eta_fine, abs=1e-13)
        errs = [abs(capture_fraction_cartesian(AIRY, a, d, n) - eta) for n in (25, 100, 400)]
        assert errs[2] < 2e-5
        assert errs[2] < errs[0]


def test_capture_quadrature_is_discriminating() -> None:
    # A wrong arc (full circle for every rho in the band, i.e. ignoring phi) overestimates.
    a, d = 0.8 * ELL, 0.6 * ELL
    right = float(capture_fraction(AIRY, a, np.array([d]))[0])
    wrong = float(AIRY.encircled_energy(np.array([a + d]))[0])
    assert right < wrong - 0.05
    assert right == pytest.approx(capture_fraction_cartesian(AIRY, a, d, 200), abs=2e-5)


def test_capture_in_unit_interval_and_monotone_in_receiver_size() -> None:
    radii = np.linspace(0.01, 6.0, 60) * ELL
    eta = np.array([float(capture_fraction(AIRY, r, np.array([0.5 * ELL]))[0]) for r in radii])
    assert np.all((eta >= 0) & (eta <= 1))
    assert np.all(np.diff(eta) >= -1e-14)  # a larger disk containing the smaller captures more


# ---------------------------------------------------------------------------------------------
# Diffraction scaling (PDF §13.2) and the impossible beam
# ---------------------------------------------------------------------------------------------


def test_smaller_wavelength_or_larger_aperture_increases_capture() -> None:
    a = 1000.0
    base = AiryPattern(10.0, 1.0e-6, AU_M, 0.5)
    eta0 = float(capture_fraction(base, a, np.array([0.0]))[0])
    for d, lam in ((20.0, 1.0e-6), (10.0, 0.5e-6), (40.0, 0.25e-6)):
        eta = float(capture_fraction(AiryPattern(d, lam, AU_M, 0.5), a, np.array([0.0]))[0])
        assert eta > eta0
    # Capture = EE(x) with x = pi D a/(lambda R) (closed form recomputed here): doubling D
    # gives EE(2x)/EE(x), close to 4 in the small-x regime (x = 0.21 here).
    eta2 = float(capture_fraction(AiryPattern(20.0, 1.0e-6, AU_M, 0.5), a, np.array([0.0]))[0])
    x = math.pi * 10.0 * a / (1.0e-6 * AU_M)

    def ee(v: float) -> float:
        return float(1 - special.j0(v) ** 2 - special.j1(v) ** 2)

    assert eta2 / eta0 == pytest.approx(ee(2 * x) / ee(x), rel=1e-10)
    assert 3.9 < eta2 / eta0 < 4.0


def test_impossible_beam_gives_negligible_capture() -> None:
    # PDF §13.2: 1 m microwave aperture at 2.45 GHz, 1 au, 1 km receiver. x = pi D a/(lambda R)
    # = 1.7e-7, so EE = x^2/4 - x^4/32 = 7.4e-15 (small-x limit of 1 - J0^2 - J1^2).
    lam = 299_792_458.0 / 2.45e9
    x = math.pi * 1.0 * 1000.0 / (lam * AU_M)
    for pat in (
        AiryPattern(1.0, lam, AU_M, 0.5),
        FarFieldAperturePattern(ApertureField("uniform_circular", 1.0), lam, AU_M, 0.5),
    ):
        eta = float(capture_fraction(pat, 1000.0, np.array([0.0]))[0])
        assert eta == pytest.approx(x * x / 4, rel=1e-9)
        assert eta < 1e-14
    # Tapered aperture: x^2/4 times the taper efficiency (on-axis gain reduction).
    tap = ApertureField("parabolic_on_pedestal", 1.0, edge_taper_dB=10.0)
    eta_t = float(
        capture_fraction(FarFieldAperturePattern(tap, lam, AU_M, 0.5), 1000.0, np.array([0.0]))[0]
    )
    assert eta_t == pytest.approx(tap.taper_efficiency * x * x / 4, rel=1e-6)


# ---------------------------------------------------------------------------------------------
# Receiver projection
# ---------------------------------------------------------------------------------------------


def test_projected_receiver_radius() -> None:
    el = np.radians([90.0, 30.0, 10.0])
    np.testing.assert_allclose(
        projected_receiver_radius_m(1000.0, el, "horizontal"),
        1000.0 * np.sqrt(np.sin(el)),
        rtol=1e-15,
    )
    assert float(
        projected_receiver_radius_m(1000.0, np.radians([30.0]), "horizontal")[0]
    ) == pytest.approx(1000.0 * math.sqrt(0.5))
    np.testing.assert_array_equal(projected_receiver_radius_m(1000.0, el, "tracking"), 1000.0)
    with pytest.raises(ValueError, match="elevation"):
        projected_receiver_radius_m(1000.0, np.array([0.0]), "horizontal")


@pytest.mark.parametrize(
    ("a_over_ell", "el_deg"), [(0.05, 10.0), (0.05, 45.0), (0.2, 10.0), (1.0, 10.0)]
)
def test_equal_area_circle_vs_exact_ellipse(a_over_ell: float, el_deg: float) -> None:
    # Horizontal receiver of radius a seen at elevation el: ellipse a x b, b = a sin(el).
    # Exact centred-ellipse capture by the polar form (spectrally accurate), cross-checked by a
    # 2-D grid. In the receiver << beam regime the equal-area-circle error follows the
    # second-order prediction kappa (a - b)^2 / 4, kappa = pi^2/(4 ell^2) for the Airy core
    # I0 (1 - x^2/4) (capture.py docstring), within 10 % (higher orders ~ (a/ell)^2).
    a = a_over_ell * ELL
    b = a * math.sin(math.radians(el_deg))
    circle = float(capture_fraction(AIRY, math.sqrt(a * b), np.array([0.0]))[0])
    ellipse = capture_fraction_centred_ellipse(AIRY, a, b)
    assert ellipse == pytest.approx(capture_fraction_centred_ellipse(AIRY, a, b, 2048), rel=1e-12)
    assert ellipse == pytest.approx(
        capture_fraction_ellipse_cartesian(AIRY, a, b, 0.0, 0.0, 800), rel=2e-4
    )
    rel = (circle - ellipse) / ellipse
    assert rel > 0  # the circle sits closer to the peak than the elongated ellipse
    if a_over_ell <= 0.2:
        pred = (math.pi**2 / (4 * ELL**2)) * (a - b) ** 2 / 4
        assert rel == pytest.approx(pred, rel=0.1)
    if a_over_ell <= 0.05:
        assert rel < 1.2e-3  # heliocentric regime (MVP a/ell ~ 0.07 at 1 au)
    else:
        assert rel < 0.5  # receiver ~ beam: documented, the approximation degrades


# ---------------------------------------------------------------------------------------------
# Pointing statistics
# ---------------------------------------------------------------------------------------------


def test_rice_parametrisation_matches_explicit_density() -> None:
    nu, s = 30.0, 12.0
    d = np.linspace(0.0, 120.0, 49)
    ref = d / s**2 * np.exp(-(d * d + nu * nu) / (2 * s * s)) * special.i0(d * nu / s**2)
    np.testing.assert_allclose(rice_offset_pdf_m_inv(d, nu, s), ref, rtol=1e-12, atol=1e-300)
    # Rayleigh special case (nu = 0): median s sqrt(2 ln 2).
    assert float(rice_offset_quantile_m(np.array([0.5]), 0.0, s)[0]) == pytest.approx(
        s * math.sqrt(2 * math.log(2)), rel=1e-12
    )
    # Monte Carlo of |nu e_x + N(0, s^2 I)|.
    rng = np.random.default_rng(2026)
    xy = rng.normal(0.0, s, size=(200_000, 2)) + np.array([nu, 0.0])
    r = np.hypot(xy[:, 0], xy[:, 1])
    q = rice_offset_quantile_m(np.array([0.05, 0.5, 0.95]), nu, s)
    emp = np.quantile(r, [0.05, 0.5, 0.95])
    np.testing.assert_allclose(emp, q, rtol=5e-3)


@pytest.mark.parametrize(
    ("a", "nu", "s"),
    [
        (0.3 * W, 0.0, 0.2 * W),
        (1.0 * W, 0.0, 0.5 * W),
        (0.5 * W, 0.4 * W, 0.1 * W),
        (2.0 * W, 1.5 * W, 0.8 * W),
        (0.1 * W, 2.0 * W, 3.0 * W),
    ],
)
def test_expected_capture_gaussian_beam_exact(a: float, nu: float, s: float) -> None:
    st = capture_statistics(GAUSS, a, nu, s)
    assert st.expected == pytest.approx(_gauss_capture_exact(a, nu, W, s), rel=1e-9, abs=1e-14)
    assert st.nominal == pytest.approx(_gauss_capture_exact(a, 0.0, W), rel=1e-12)
    assert st.bias_only == pytest.approx(_gauss_capture_exact(a, nu, W), rel=1e-11)
    assert st.offset_probability_mass == pytest.approx(1.0, abs=1e-10)
    assert st.percentile_method == "exact_monotone"
    # Percentiles: eta(d) decreasing -> q-quantile of eta is eta at the (1-q)-quantile of d.
    for q, val in ((0.05, st.p05), (0.5, st.p50), (0.95, st.p95)):
        d_q = float(stats.rice.ppf(1 - q, nu / s, scale=s))
        assert val == pytest.approx(_gauss_capture_exact(a, d_q, W), rel=1e-10, abs=1e-14)
    assert st.p05 <= st.p50 <= st.p95


def test_expected_capture_matches_monte_carlo_airy() -> None:
    # Independent stochastic check with an explicit Generator: sample 2-D offsets, evaluate the
    # (separately validated) 1-D capture at |offset|, average. 5 standard errors.
    a, nu, s = 0.4 * ELL, 0.2 * ELL, 0.3 * ELL
    st = capture_statistics(AIRY, a, nu, s)
    rng = np.random.default_rng(777)
    xy = rng.normal(0.0, s, size=(40_000, 2)) + np.array([nu, 0.0])
    eta = capture_fraction(AIRY, a, np.hypot(xy[:, 0], xy[:, 1]))
    se = float(np.std(eta)) / math.sqrt(eta.size)
    assert st.expected == pytest.approx(float(np.mean(eta)), abs=5 * se)


def _reference_capture_quantile(a: float, nu: float, s: float, q: float) -> float:
    """Exact q-quantile of eta(d), d ~ Rice(nu, s), by root finding (independent of capture.py's
    spline): P(eta <= y) = Rice mass of {d: eta(d) <= y}, whose boundary points are the roots of
    eta(d) = y (bracketed on a 4001-point grid, refined by brentq on capture_fraction itself);
    then brentq on y. Root and level tolerances 1e-13."""
    lo, hi = max(0.0, nu - 9 * s), nu + 9 * s
    grid = np.linspace(lo, hi, 4001)
    eta_grid = capture_fraction(AIRY, a, grid)

    def eta_at(d: float) -> float:
        return float(capture_fraction(AIRY, a, np.array([d]))[0])

    def prob_at_most(y: float) -> float:
        g = eta_grid - y
        pts = [lo]
        for i in np.nonzero(g[:-1] * g[1:] < 0)[0]:
            pts.append(
                optimize.brentq(
                    lambda d: eta_at(d) - y, grid[i], grid[i + 1], xtol=1e-13 * ELL, rtol=1e-15
                )
            )
        pts.append(hi)
        p = np.array(pts)
        below = capture_fraction(AIRY, a, 0.5 * (p[1:] + p[:-1])) <= y
        return float(np.sum(np.diff(stats.rice.cdf(p, nu / s, scale=s))[below]))

    return float(
        optimize.brentq(
            lambda y: prob_at_most(y) - q,
            float(eta_grid.min()),
            float(eta_grid.max()),
            xtol=1e-13,
            rtol=1e-15,
        )
    )


@pytest.mark.parametrize(
    ("a", "nu", "s"),
    [(0.4, 0.2, 0.3), (0.5, 0.2, 0.4), (1.5, 0.0, 0.2), (0.3, 1.0, 0.8)],
)
def test_level_set_percentiles_match_exact_quantiles(a: float, nu: float, s: float) -> None:
    # Review M5: when eta(d) is not monotone over the offset window (Airy rings), percentiles
    # come from the cubic-spline level set. Measured agreement with the root-finding reference
    # is <= 1.2e-9; the tolerance 1e-8 leaves a margin for platform rounding. The former
    # node-step quantile was off by up to 2e-3 here (e.g. p05 0.04487 vs 0.04289 for
    # (0.5, 0.2, 0.4)), so this test discriminates.
    st = capture_statistics(AIRY, a * ELL, nu * ELL, s * ELL)
    assert st.percentile_method == "level_set"
    ref = [_reference_capture_quantile(a * ELL, nu * ELL, s * ELL, q) for q in (0.05, 0.5, 0.95)]
    np.testing.assert_allclose([st.p05, st.p50, st.p95], ref, rtol=0, atol=1e-8)
    # Converged: a factor-3 refinement of the spline grid moves them by far less than 1e-8.
    fine = capture_statistics(AIRY, a * ELL, nu * ELL, s * ELL, refinement=3.0)
    np.testing.assert_allclose([fine.p05, fine.p50, fine.p95], ref, rtol=0, atol=1e-8)


def test_offset_law_gaussian_limit_matches_rice() -> None:
    # Large bias-to-jitter ratio (review M5): for nu/s >= 1e4 the Rice law is replaced by its
    # Gaussian limit N(nu + s^2/(2 nu), s^2 (1 + s^2/(2 nu^2))). Where scipy is still finite
    # (b = 1e4, 1e5) quantiles must agree to 1e-8 s and the CDF to 2e-9 (measured asymptotic
    # differences 0.83 s/b^2 and 0.12/b^2).
    s = 3.0
    q = np.array([0.05, 0.5, 0.95])  # the quantiles used; scipy ppf is ~1e-12 relative
    for b in (1.0e4, 1.0e5):
        nu = b * s
        np.testing.assert_allclose(
            rice_offset_quantile_m(q, nu, s),
            stats.rice.ppf(q, b, scale=s),
            rtol=0,
            atol=1e-8 * s,
        )
        x = nu + s * np.linspace(-5, 5, 11)
        np.testing.assert_allclose(
            rice_offset_cdf(x, nu, s), stats.rice.cdf(x, b, scale=s), rtol=0, atol=2e-9
        )
    # Below the switch-over the exact Rice law is used (scipy); continuity across it:
    below = capture_statistics(AIRY, 0.1 * ELL, 0.5 * ELL, 0.5 * ELL / 9_999.0)
    above = capture_statistics(AIRY, 0.1 * ELL, 0.5 * ELL, 0.5 * ELL / 10_001.0)
    assert below.expected == pytest.approx(above.expected, abs=1e-12)
    # scipy itself fails here (NaN), which is why the limit exists:
    assert np.isnan(stats.rice.ppf(0.5, 1.0e6))


@pytest.mark.parametrize("ratio", [1.0e6, 1.0e17])
def test_tiny_jitter_with_bias_is_finite_and_consistent(ratio: float) -> None:
    # nu/s = 1e6 (scipy.stats.rice returns NaN) and 1e17 (offsets nu +- 9 s not representable):
    # the expectation must equal eta(nu) to second order, and the 5-95 % spread must equal
    # 2 * 1.6449 s |eta'(nu)| (linearisation; eta' by an independent central difference).
    nu = 0.5 * ELL
    s = nu / ratio
    a = 0.1 * ELL
    st = capture_statistics(AIRY, a, nu, s)
    assert all(math.isfinite(v) for v in (st.expected, st.p05, st.p50, st.p95))
    assert st.offset_probability_mass == pytest.approx(1.0, abs=1e-12)
    assert st.expected == pytest.approx(st.bias_only, abs=1e-12)
    h = 1e-3 * ELL
    slope = float(np.diff(capture_fraction(AIRY, a, np.array([nu - h, nu + h])))[0]) / (2 * h)
    spread = st.p95 - st.p05
    # Relative 1e-3 on the spread where it is resolvable; at 1e17 it is below float resolution.
    assert spread == pytest.approx(2 * 1.6448536269514722 * s * abs(slope), rel=1e-3, abs=1e-15)
    assert st.p05 <= st.p50 <= st.p95


def test_offset_quadrature_converges() -> None:
    a, nu, s = 0.4 * ELL, 0.2 * ELL, 0.3 * ELL
    base = capture_statistics(AIRY, a, nu, s)
    fine = capture_statistics(AIRY, a, nu, s, refinement=3.0)
    # Both evaluations are accurate to ~1e-11 (8-node panels of <= min(s, ell)/2 in d and of
    # <= ell/2 in rho); a factor-3 refinement must not move the result by more than that.
    assert base.expected == pytest.approx(fine.expected, abs=1e-10)
    assert base.n_offset_nodes < fine.n_offset_nodes


def test_no_jitter_is_deterministic() -> None:
    st = capture_statistics(AIRY, 0.5 * ELL, 0.3 * ELL, 0.0)
    eta = float(capture_fraction(AIRY, 0.5 * ELL, np.array([0.3 * ELL]))[0])
    assert (st.expected, st.p05, st.p50, st.p95, st.bias_only) == (eta, eta, eta, eta, eta)
    assert st.percentile_method == "deterministic"


@pytest.mark.parametrize("a_over_w", [0.1, 1.0, 3.0])
def test_expected_capture_non_increasing_with_sigma_gaussian(a_over_w: float) -> None:
    # PDF §13.1/§13.2: centred baseline. For a unimodal symmetric beam eta(d) is non-increasing
    # in |d| (Anderson 1955, Proc. AMS 6, 170) and |d| grows stochastically with sigma, so
    # E[eta] cannot increase with sigma; strictly decreasing here.
    sig = np.linspace(0.0, 2.0, 21) * W
    e = [capture_statistics(GAUSS, a_over_w * W, 0.0, float(s)).expected for s in sig]
    assert np.all(np.diff(e) < 0)


def test_expected_capture_non_increasing_with_sigma_airy_baseline() -> None:
    # Airy beam with the receiver inside the central lobe (a = 0.07 ell, the MVP regime
    # a/ell ~ 0.07; jitter up to 0.6 ell keeps essentially all offset mass in the main lobe).
    a = 0.07 * ELL
    sig = np.linspace(0.0, 0.6, 25) * ELL
    e = np.array([capture_statistics(AIRY, a, 0.0, float(s)).expected for s in sig])
    assert np.all(np.diff(e) < 0)
    assert e[-1] < 0.5 * e[0]


def test_jitter_can_raise_capture_when_receiver_cuts_the_first_ring() -> None:
    # Not a theorem for non-unimodal patterns: a receiver of radius 1.5 ell (between the first
    # null 1.22 ell and the first bright ring at 1.64 ell) gains ring power when the beam
    # wanders. This is real physics (confirmed by an independent 2-D Monte Carlo), so the
    # monotonicity guarantee is stated only for the centred baseline above.
    a, s = 1.5 * ELL, 0.2 * ELL
    st = capture_statistics(AIRY, a, 0.0, s)
    assert st.expected > st.nominal + 1e-3
    rng = np.random.default_rng(99)
    xy = rng.normal(0.0, s, size=(20_000, 2))
    eta = capture_fraction(AIRY, a, np.hypot(xy[:, 0], xy[:, 1]))
    assert float(np.mean(eta)) > st.nominal + 1e-3
    assert st.percentile_method == "level_set"


def test_bias_reduces_capture() -> None:
    a, s = 0.2 * ELL, 0.05 * ELL
    exp = [
        capture_statistics(AIRY, a, float(b), s).expected for b in np.linspace(0.0, 0.8, 9) * ELL
    ]
    assert np.all(np.diff(exp) < 0)
    st = capture_statistics(AIRY, a, 0.3 * ELL, 0.0)
    assert st.bias_only < st.nominal


def test_capture_statistics_input_validation() -> None:
    with pytest.raises(ValueError, match="bias_offset_m"):
        capture_statistics(AIRY, 1.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="offset_m"):
        capture_fraction(AIRY, 1.0, np.array([-0.1]))
    with pytest.raises(ValueError, match="jitter"):
        capture_statistics(AIRY, 0.1 * ELL, 0.0, 1e3 * ELL)  # jitter >> beam: budget exceeded


def test_capture_with_fresnel_pattern_matches_cartesian() -> None:
    # Near-field (N_F = 8) pattern with an on-axis null: the 1-D form and a 2-D grid agree, and
    # the grid error shrinks under refinement (the grid needs one Hankel quadrature per cell,
    # so it is kept modest; the 1-D form is converged to ~1e-12).
    ap = ApertureField("uniform_circular", 10.0)
    z = 10.0**2 / (1e-6 * 8.0)
    pat = FresnelHankelPattern(ap, 1e-6, z, paraxial_rho_max_m=20.0)
    for a, d in ((3.0, 0.0), (3.0, 2.0)):
        eta = float(capture_fraction(pat, a, np.array([d]))[0])
        assert eta == pytest.approx(
            float(capture_fraction(pat, a, np.array([d]), refinement=3.0)[0]), abs=1e-11
        )
        errs = [abs(capture_fraction_cartesian(pat, a, d, n) - eta) for n in (15, 60)]
        assert errs[1] < 1e-3
        assert errs[1] < errs[0]
