"""Batched Airy capture kernel (helioswarm.capture_batch): numpy and JAX backends against the
M5 per-point reference (helioswarm.capture.capture_statistics) and scipy Bessel functions.

Tolerances: the batch uses at least the M5 panel resolution on a common grid, so it agrees
with the adaptive reference to quadrature accuracy (1e-9 asserted, ~5e-12 measured); the JAX
backend runs the same algorithm with trapezoid Bessel functions (accurate to ~3e-15), so it must
agree with the numpy batch to 1e-12.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from scipy import special

from helioswarm.beam import beam_settings_from_config, evaluate_beam
from helioswarm.capture import capture_statistics
from helioswarm.capture_batch import (
    airy_capture_batch,
    bessel_j01_trapezoid,
    evaluate_beam_batched,
    trapezoid_bessel_nodes,
)
from helioswarm.config import load_config, parse_config
from helioswarm.optical import AiryPattern

from config_helpers import MVP_CONFIG, load_mvp_data, microwave_data

D, LAM = 10.0, 1.0e-6


def _cases(n: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.5, 1.5, n) * 1.5e11
    a = rng.uniform(50.0, 5000.0, n)
    nu = rng.choice([0.0, 2e-9, 5e-8], n) * r
    s = rng.choice([0.0, 1e-9, 2e-8], n) * r
    return r, a, nu, s


@pytest.mark.parametrize("x_max", [1.0, 30.0, 300.0])
def test_trapezoid_bessel_matches_scipy(x_max: float) -> None:
    x = np.linspace(0.0, x_max, 4001)
    j0, j1 = bessel_j01_trapezoid(np, x, trapezoid_bessel_nodes(x_max))
    # 3e-15 measured; the aliasing terms J_{N-1}(x) are < 1e-16 by the node rule.
    np.testing.assert_allclose(j0, special.j0(x), rtol=0, atol=1e-13)
    np.testing.assert_allclose(j1, special.j1(x), rtol=0, atol=1e-13)
    # Discriminating: too few nodes (N - 1 not >> x) fails the 1e-13 tolerance by >= 1e3x.
    j0_bad, _ = bessel_j01_trapezoid(np, x, max(2, int(x_max // 8)))
    assert np.max(np.abs(j0_bad - special.j0(x))) > 1e-10


def test_numpy_batch_matches_m5_reference() -> None:
    r, a, nu, s = _cases(30, 5)
    k = math.pi * D / (LAM * r)
    res = airy_capture_batch(k, a, nu, s, backend="numpy")
    for i in range(r.size):
        p = AiryPattern(aperture_diameter_m=D, wavelength_m=LAM, range_m=float(r[i]),
                        max_fresnel_number=0.5)  # fmt: skip
        st = capture_statistics(p, float(a[i]), float(nu[i]), float(s[i]))
        assert res.nominal[i] == pytest.approx(st.nominal, abs=1e-12)
        assert res.bias_only[i] == pytest.approx(st.bias_only, abs=1e-9)
        assert res.expected[i] == pytest.approx(st.expected, abs=1e-9)
        for q in ("p05", "p50", "p95"):
            assert getattr(res, q)[i] == pytest.approx(getattr(st, q), abs=1e-9)
        assert res.offset_probability_mass[i] == pytest.approx(st.offset_probability_mass, abs=1e-9)
    assert res.n_fallback == int(np.sum(res.percentile_method == "level_set"))


def test_batch_detects_wrong_physics() -> None:
    # Discriminating: dropping the band integral (eta = EE(a - d) only) is far off for an
    # offset receiver, and the jitter expectation is strictly below the nominal for a small one.
    r = np.array([1.5e11])
    k = math.pi * D / (LAM * r)
    ell = LAM * r[0] / D
    res = airy_capture_batch(k, np.array([0.3 * ell]), np.array([0.2 * ell]), np.array([0.1 * ell]))
    inner_only = 1 - special.j0(k[0] * 0.1 * ell) ** 2 - special.j1(k[0] * 0.1 * ell) ** 2
    assert res.bias_only[0] > 2 * inner_only
    assert res.expected[0] < res.nominal[0]


def test_jax_backend_equals_numpy_batch() -> None:
    pytest.importorskip("jax")
    r, a, nu, s = _cases(40, 9)
    k = math.pi * D / (LAM * r)
    rn = airy_capture_batch(k, a, nu, s, backend="numpy")
    rj = airy_capture_batch(k, a, nu, s, backend="jax")
    for name in ("nominal", "bias_only", "expected", "p05", "p50", "p95"):
        np.testing.assert_allclose(getattr(rj, name), getattr(rn, name), rtol=0, atol=1e-12)
    assert rj.backend == "jax"
    assert rj.bessel_quarter_nodes == rn.bessel_quarter_nodes


def test_evaluate_beam_batched_matches_evaluate_beam() -> None:
    cfg = load_config(MVP_CONFIG)
    st = beam_settings_from_config(cfg)
    rng = np.random.default_rng(2)
    r = rng.uniform(1.6e11, 1.8e11, 25)
    el = rng.uniform(0.2, 1.5, 25)
    ref = evaluate_beam(r, el, st)
    got = evaluate_beam_batched(r, el, st, backend="numpy")
    for name in (
        "capture_fraction",
        "capture_fraction_nominal",
        "capture_fraction_bias_only",
        "capture_fraction_p05",
        "capture_fraction_p50",
        "capture_fraction_p95",
        "pointing_loss",
        "peak_irradiance_per_W_m2",
        "projected_receiver_radius_m",
        "fresnel_number",
    ):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(
        got.diagnostics.receiver_projection_error,
        ref.diagnostics.receiver_projection_error,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        got.diagnostics.first_null_radius_m, ref.diagnostics.first_null_radius_m, rtol=1e-15
    )


def test_batched_beam_refuses_other_models() -> None:
    cfg = parse_config(microwave_data(load_mvp_data()))
    with pytest.raises(NotImplementedError, match="airy_far_field"):
        evaluate_beam_batched(np.array([1.5e11]), np.array([1.0]), beam_settings_from_config(cfg))


@pytest.mark.gpu
def test_jax_gpu_backend_equals_numpy_on_a_real_device() -> None:
    jax = pytest.importorskip("jax")
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        pytest.skip("JAX_PLATFORMS=cpu: no GPU requested (run by slurm/gpu_bench.sbatch)")
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("no CUDA device visible to JAX")
    r, a, nu, s = _cases(200, 13)
    k = math.pi * D / (LAM * r)
    rn = airy_capture_batch(k, a, nu, s, backend="numpy")
    rj = airy_capture_batch(k, a, nu, s, backend="jax")
    np.testing.assert_allclose(rj.expected, rn.expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rj.p95, rn.p95, rtol=0, atol=1e-12)
