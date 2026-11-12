"""Batched Airy far-field capture under pointing errors, with numpy and JAX backends.

Why this module exists (PDF §14.3)
----------------------------------
The M5 reference (:func:`helioswarm.capture.capture_statistics`, called per epoch by
:func:`helioswarm.beam.evaluate_beam`) adapts its quadrature to each epoch and is the CPU
reference of record. Sweeps and (later) Monte-Carlo studies evaluate the same physics at very
many independent points ``(range, receiver radius, bias, jitter, aperture, wavelength)``, an
embarrassingly parallel workload. This module evaluates it as one array program with *fixed*
quadrature shapes per batch, so it can run vectorised on the CPU (``backend="numpy"``) or be
compiled by JAX for a GPU (``backend="jax"``). Selection: ``helioswarm run/sweep --backend``;
the default is ``numpy`` (the per-epoch M5 reference in :mod:`helioswarm.system`). The JAX
result must equal the numpy batch result and the M5 reference to the tolerances stated in
``tests/unit/test_capture_batch.py``.

Physics (identical to :mod:`helioswarm.capture`, Airy pattern only)
---------------------------------------------------------------------
With ``k = pi D / (lambda R)`` [m^-1] and ``x = k rho`` the Airy irradiance per watt is
``I(rho) = (k^2 / pi) J1(x)^2 / x^2`` and the encircled energy ``EE = 1 - J0(x)^2 - J1(x)^2``
(Born & Wolf §8.5.2). An offset disk captures (PDF §9.4)::

    eta(d) = EE(k max(a - d, 0)) + int_{|a-d|}^{a+d} I(rho) 2 phi(rho; d, a) rho drho,

``phi = arccos((rho^2 + d^2 - a^2)/(2 rho d))``, and with the Rice offset law ``d ~ Rice(nu, s)``
(bias ``nu = R b``, jitter ``s = R sigma``) the expected capture is ``E[eta] = int eta f``
(PDF §9.5). For ``nu/s >= 1e4`` the offset law is the same named Gaussian limit as in M5.

Quadrature (fixed per batch): the band uses ``rho = m - h cos t``, ``t in [0, pi]``, with
``ceil(4 k h_max) + 2`` Gauss-Legendre panels of 8 nodes (the M5 rule: panels of at most a
quarter of the feature scale ``lambda R / D`` in ``rho``), ``h_max`` the largest band half
width in the batch; the offset window ``[max(0, nu - 9 s), nu + 9 s]`` (Gaussian limit: ``t``
in ``[-9, 9]``) uses the largest M5 panel count of the batch, so every point gets at least the
M5 resolution. Percentiles: exact when ``eta(d)`` is non-increasing on the offset nodes
(``eta(d_{1-q})``, quantiles from ``scipy.stats.rice`` on the host); otherwise the point falls
back to the M5 per-point reference (``level_set``), and the fallback count is reported.

Bessel functions: the numpy backend uses ``scipy.special.j0/j1`` (Cephes). JAX 0.12 provides
no accurate ``j0/j1`` (``jax.scipy.special.bessel_jn`` returned NaN for x < 1 and O(1) errors
for x > 60 when probed, see docs/milestones/M6.md), so the JAX backend evaluates
``J0(x) = (1/2 pi) int cos(x sin tau) dtau`` and ``J1(x) = (1/2 pi) int sin(tau) sin(x sin
tau) dtau`` by the N-point trapezoidal rule, which is exact up to the aliasing terms
``J_{N -+ n}(x)`` (generating function ``e^{i x sin tau} = sum_m J_m(x) e^{i m tau}``);
``N - 1 >= x_max + 12 x_max^{1/3} + 30`` makes them < 1e-16 (checked against scipy in the
tests). Using the quarter-period symmetry only ``N/4 + 1`` evaluations are needed.

Validity: Airy far field (``models.beam = airy_far_field``) only; any other model raises
``NotImplementedError`` rather than silently substituting a different pattern.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

from helioswarm.aperture import BeamModelValidityError, fresnel_number
from helioswarm.beam import BeamFieldDiagnostics, BeamResult, BeamSettings
from helioswarm.capture import (
    MONOTONE_TOLERANCE,
    RICE_GAUSSIAN_LIMIT_SHAPE,
    RICE_SPAN_SIGMAS,
    capture_statistics,
    projected_receiver_radius_m,
)
from helioswarm.optical import AiryPattern, airy_first_null_radius_m

Float64Array = NDArray[np.float64]
BatchBackend = Literal["numpy", "jax"]

GL_ORDER: Final[int] = 8
"""Gauss-Legendre nodes per panel (as ``aperture.GL_ORDER_RADIAL``)."""

SMALL_X: Final[float] = 1.0e-4
"""Below this ``x`` the Airy EE and profile use their series (as ``optical._SMALL_X``)."""

MAX_ELEMENTS_PER_CHUNK: Final[int] = 1 << 23
"""Largest ``points x offsets x band nodes (x Bessel nodes)`` array evaluated at once."""

MAX_BAND_PANELS: Final[int] = 4096
"""Budget of band panels per batch (receiver >> beam feature scale raises)."""

MAX_OFFSET_PANELS: Final[int] = 4096
"""Budget of offset panels per batch (jitter >> beam feature scale raises)."""

PercentileMethod = Literal["deterministic", "exact_monotone", "level_set", "not_computed"]


def _gl(order: int) -> tuple[Float64Array, Float64Array]:
    x, w = np.polynomial.legendre.leggauss(order)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def _composite(lo: float, hi: float, panels: int) -> tuple[Float64Array, Float64Array]:
    xi, wi = _gl(GL_ORDER)
    edges = np.linspace(lo, hi, panels + 1)
    half = 0.5 * np.diff(edges)[:, None]
    nodes = edges[:-1, None] + half * (xi + 1.0)
    return nodes.ravel(), (half * wi).ravel()


def trapezoid_bessel_nodes(x_max: float) -> int:
    """Quarter-period node count ``M`` (``N = 4 M`` trapezoid points) for ``x <= x_max``."""
    xm = max(float(x_max), 0.0)
    m: int = math.ceil((xm + 12.0 * xm ** (1.0 / 3.0) + 31.0) / 4.0)
    return max(8, m)


def bessel_j01_trapezoid(xp: Any, x: Any, m_quarter: int) -> tuple[Any, Any]:
    """``J0(x), J1(x)`` by the periodic trapezoidal rule (module docstring), any array module.

    Args:
        xp: Array namespace (``numpy`` or ``jax.numpy``).
        x: Arguments (any shape, >= 0).
        m_quarter: ``M``; ``N = 4 M`` points per period (:func:`trapezoid_bessel_nodes`).

    Returns:
        ``(J0, J1)`` with the shape of ``x``.
    """
    k = np.arange(m_quarter + 1, dtype=np.float64)
    s = np.sin(0.5 * math.pi * k / m_quarter)
    w = np.full(m_quarter + 1, 4.0)
    w[0] = 2.0
    w[-1] = 2.0
    w /= 4.0 * m_quarter
    sx = x[..., None] * xp.asarray(s)
    j0 = xp.sum(xp.asarray(w) * xp.cos(sx), axis=-1)
    j1 = xp.sum(xp.asarray(w * s) * xp.sin(sx), axis=-1)
    return j0, j1


def _scipy_j01(xp: Any, x: Any) -> tuple[Any, Any]:
    return special.j0(x), special.j1(x)


def _eta_offsets(
    xp: Any,
    j01: Callable[[Any, Any], tuple[Any, Any]],
    k: Any,
    a: Any,
    d: Any,
    t: Any,
    wt: Any,
) -> Any:
    """``eta(d)`` for shapes ``k, a: (B, 1)``, ``d: (B, K)``; band nodes ``t, wt: (T,)``."""
    x_in = k * xp.maximum(a - d, 0.0)
    j0, j1 = j01(xp, x_in)
    ee = xp.where(x_in < SMALL_X, x_in**2 / 4.0 - x_in**4 / 32.0, 1.0 - j0**2 - j1**2)
    h = xp.minimum(a, d)[..., None]
    m = xp.maximum(a, d)[..., None]
    rho = m - h * xp.cos(t)
    x = k[..., None] * rho
    xs = xp.where(x < SMALL_X, 1.0, x)
    _, j1x = j01(xp, xs)
    two_j1 = xp.where(x < SMALL_X, 1.0 - x**2 / 8.0, 2.0 * j1x / xs)
    intensity = (k[..., None] ** 2 / (4.0 * math.pi)) * two_j1**2
    dd = d[..., None]
    denom = xp.where(dd > 0.0, 2.0 * rho * dd, 1.0)
    c = xp.clip((rho**2 + dd**2 - a[..., None] ** 2) / denom, -1.0, 1.0)
    phi = xp.arccos(c)
    band = xp.sum(intensity * 2.0 * phi * rho * h * xp.sin(t) * wt, axis=-1)
    band = xp.where(d > 0.0, band, 0.0)
    return xp.clip(ee + band, 0.0, 1.0)


@dataclass(frozen=True, eq=False)
class BatchCaptureResult:
    """Capture statistics of a batch of independent points (arrays shape (B,), diffraction
    only: the wavefront and lattice factors are applied by :func:`evaluate_beam_batched`).

    Attributes:
        nominal: ``eta(0)`` (perfect pointing).
        bias_only: ``eta(nu)``.
        expected: ``E[eta]`` over the offset law.
        p05, p50, p95: Percentiles of the capture distribution.
        offset_probability_mass: Quadrature mass of the offset density (1 for no jitter).
        percentile_method: ``deterministic``, ``exact_monotone`` or ``level_set`` (fallback).
        n_fallback: Points computed by the M5 per-point reference (non-monotone ``eta(d)``).
        backend: Backend used.
        band_panels, offset_panels, bessel_quarter_nodes: Static quadrature shapes used.
    """

    nominal: Float64Array
    bias_only: Float64Array
    expected: Float64Array
    p05: Float64Array
    p50: Float64Array
    p95: Float64Array
    offset_probability_mass: Float64Array
    percentile_method: NDArray[np.str_]
    n_fallback: int
    backend: BatchBackend
    band_panels: int
    offset_panels: int
    bessel_quarter_nodes: int


class _Engine:
    """Evaluates ``eta(d)`` and offset expectations in chunks on one backend."""

    def __init__(self, backend: BatchBackend, x_max: float) -> None:
        self.backend = backend
        self.m_quarter = trapezoid_bessel_nodes(x_max)
        if backend == "numpy":
            self.xp: Any = np
            self._eta: Callable[..., Any] = lambda k, a, d, t, wt: _eta_offsets(
                np, _scipy_j01, k, a, d, t, wt
            )
            self.bessel_factor = 1
        elif backend == "jax":
            try:
                import jax
                import jax.numpy as jnp
            except ImportError as exc:  # pragma: no cover - depends on the installed extras
                raise ImportError(
                    "backend='jax' needs the optional 'jax' (CPU) or 'gpu' extra: "
                    "uv sync --extra jax"
                ) from exc
            self._jax = jax
            self.xp = jnp
            mq = self.m_quarter

            def j01(xp: Any, x: Any) -> tuple[Any, Any]:
                return bessel_j01_trapezoid(xp, x, mq)

            def eta(k: Any, a: Any, d: Any, t: Any, wt: Any) -> Any:
                return _eta_offsets(jnp, j01, k, a, d, t, wt)

            self._eta = jax.jit(eta)
            self.bessel_factor = mq + 1
        else:
            raise ValueError(f"unknown backend {backend!r}; use 'numpy' or 'jax'")

    def eta(
        self, k: Float64Array, a: Float64Array, d: Float64Array, t: Float64Array, wt: Float64Array
    ) -> Float64Array:
        """``eta`` for ``k, a: (B,)``, ``d: (B, K)`` in chunks; returns a numpy (B, K) array."""
        b, kk = d.shape
        per_point = max(1, kk * t.size * self.bessel_factor)
        chunk = max(1, MAX_ELEMENTS_PER_CHUNK // per_point)
        out = np.empty((b, kk))
        if self.backend == "numpy":
            for i in range(0, b, chunk):
                sl = slice(i, i + chunk)
                out[sl] = self._eta(k[sl, None], a[sl, None], d[sl], t, wt)
            return out
        jax = self._jax
        jnp = self.xp
        with jax.enable_x64(True):
            tj, wj = jnp.asarray(t), jnp.asarray(wt)
            for i in range(0, b, chunk):
                n = min(chunk, b - i)
                # Pad to a fixed chunk so the jitted kernel is compiled once per batch shape.
                pad = (chunk - n) if b > chunk else 0
                kc = np.pad(k[i : i + n], (0, pad), constant_values=1.0)
                ac = np.pad(a[i : i + n], (0, pad), constant_values=1.0)
                dc = np.pad(d[i : i + n], ((0, pad), (0, 0)), constant_values=0.0)
                res = self._eta(
                    jnp.asarray(kc)[:, None], jnp.asarray(ac)[:, None], jnp.asarray(dc), tj, wj
                )
                out[i : i + n] = np.asarray(res, dtype=np.float64)[:n]
        return out


def _rice_ppf(q: Float64Array, nu: Float64Array, s: Float64Array) -> Float64Array:
    """Offset quantiles ``d_q`` per point (shape (B, len(q))), Rice or its Gaussian limit."""
    gauss = nu >= RICE_GAUSSIAN_LIMIT_SHAPE * s
    out = np.empty((nu.size, q.size))
    if np.any(~gauss):
        g = ~gauss
        out[g] = stats.rice.ppf(q[None, :], (nu[g] / s[g])[:, None], scale=s[g][:, None])
    if np.any(gauss):
        mean = nu[gauss] + 0.5 * s[gauss] ** 2 / nu[gauss]
        sd = s[gauss] * np.sqrt(1.0 + 0.5 * (s[gauss] / nu[gauss]) ** 2)
        out[gauss] = mean[:, None] + sd[:, None] * special.ndtri(q)[None, :]
    if np.any(~np.isfinite(out)):
        raise ValueError("pointing-offset quantile is not finite (scipy.stats.rice failure)")
    return np.maximum(out, 0.0)


def airy_capture_batch(
    k_m_inv: ArrayLike,
    receiver_radius_m: ArrayLike,
    bias_offset_m: ArrayLike,
    sigma_offset_m: ArrayLike,
    *,
    backend: BatchBackend = "numpy",
    percentiles: bool = True,
) -> BatchCaptureResult:
    """Capture statistics of the Airy pattern for a batch of independent points (PDF §9.4-9.5).

    Args:
        k_m_inv: ``k = pi D / (lambda R)`` per point [m^-1].
        receiver_radius_m: (Projected) receiver radius ``a`` per point [m].
        bias_offset_m: ``nu = R b`` per point [m].
        sigma_offset_m: ``s = R sigma`` per point [m].
        backend: ``numpy`` or ``jax``.
        percentiles: Compute the p05/p50/p95 capture percentiles (default). With ``False`` they
            are NaN and ``percentile_method`` is ``not_computed`` (expected/nominal/bias-only
            capture only, e.g. for sweeps that need the expectation alone; no CPU fallback).

    Returns:
        :class:`BatchCaptureResult` (deterministic quadrature; no sampling).
    """
    k = np.atleast_1d(np.asarray(k_m_inv, dtype=np.float64))
    shape = k.shape
    a = np.broadcast_to(np.asarray(receiver_radius_m, dtype=np.float64), shape).copy()
    nu = np.broadcast_to(np.asarray(bias_offset_m, dtype=np.float64), shape).copy()
    s = np.broadcast_to(np.asarray(sigma_offset_m, dtype=np.float64), shape).copy()
    if k.ndim != 1:
        raise ValueError("inputs must be 1-D")
    for name, v, strict in (("k", k, True), ("a", a, True), ("nu", nu, False), ("s", s, False)):
        if np.any(~np.isfinite(v)) or (np.any(v <= 0.0) if strict else np.any(v < 0.0)):
            raise ValueError(f"{name} must be finite and {'>' if strict else '>='} 0")
    feature = math.pi / k  # lambda R / D
    jit = s > 0.0
    gauss = jit & (nu >= RICE_GAUSSIAN_LIMIT_SHAPE * s)
    rice = jit & ~gauss
    sd_g = np.where(gauss, s * np.sqrt(1.0 + 0.5 * (s / np.where(nu > 0, nu, 1.0)) ** 2), 1.0)
    mean_g = np.where(gauss, nu + 0.5 * s**2 / np.where(nu > 0, nu, 1.0), 0.0)
    d_lo = np.where(
        gauss, mean_g - RICE_SPAN_SIGMAS * sd_g, np.maximum(nu - RICE_SPAN_SIGMAS * s, 0)
    )
    d_hi = np.where(gauss, mean_g + RICE_SPAN_SIGMAS * sd_g, nu + RICE_SPAN_SIGMAS * s)
    d_hi = np.where(jit, d_hi, nu)

    # Static quadrature shapes from the most demanding point (M5 panel rules).
    h_max = np.minimum(a, np.maximum(d_hi, nu))
    band_panels = int(np.max(np.ceil(4.0 * k * h_max))) + 2
    if band_panels > MAX_BAND_PANELS:
        raise ValueError(
            f"batched capture band needs {band_panels} panels (> {MAX_BAND_PANELS}); receiver "
            "much larger than the beam feature scale"
        )
    off_panels = 1
    if np.any(rice):
        width = 0.5 * np.minimum(s[rice], feature[rice])
        off_panels = max(off_panels, int(np.max(np.ceil((d_hi[rice] - d_lo[rice]) / width))) + 1)
    if np.any(gauss):
        width_t = 0.5 * np.minimum(1.0, feature[gauss] / sd_g[gauss])
        off_panels = max(off_panels, int(np.max(np.ceil(2.0 * RICE_SPAN_SIGMAS / width_t))) + 1)
    if off_panels > MAX_OFFSET_PANELS:
        raise ValueError(
            f"batched offset quadrature needs {off_panels} panels (> {MAX_OFFSET_PANELS}); "
            "jitter much larger than the beam feature scale"
        )
    t, wt = _composite(0.0, math.pi, band_panels)
    x_max = float(np.max(k * (a + d_hi)))
    eng = _Engine(backend, x_max)

    # nominal and bias-only: two offsets per point.
    eta2 = eng.eta(k, a, np.stack([np.zeros_like(nu), nu], axis=1), t, wt)
    nominal, bias_only = eta2[:, 0], eta2[:, 1]
    expected = bias_only.copy()
    mass = np.ones(shape)
    p05, p50, p95 = bias_only.copy(), bias_only.copy(), bias_only.copy()
    method = np.full(shape, "deterministic", dtype="<U16")
    n_fallback = 0
    if np.any(jit):
        idx = np.flatnonzero(jit)
        u, wu = _composite(-1.0, 1.0, off_panels)  # reference nodes, mapped per point
        lo, hi = d_lo[idx], d_hi[idx]
        half = 0.5 * (hi - lo)
        d_nodes = lo[:, None] + half[:, None] * (u[None, :] + 1.0)
        g = gauss[idx]
        # Rice density f(d) = (d/s^2) exp(-(d - nu)^2/(2 s^2)) i0e(d nu / s^2).
        ss = s[idx][:, None]
        nn = nu[idx][:, None]
        f_rice = (
            (d_nodes / ss**2)
            * np.exp(-0.5 * ((d_nodes - nn) / ss) ** 2)
            * special.i0e(d_nodes * nn / ss**2)
        )
        w_rice = half[:, None] * wu[None, :] * f_rice
        tg = RICE_SPAN_SIGMAS * u  # standardised variable on [-9, 9]
        w_gauss = (RICE_SPAN_SIGMAS * wu * np.exp(-0.5 * tg * tg) / math.sqrt(2.0 * math.pi))[
            None, :
        ]
        weights = np.where(g[:, None], w_gauss, w_rice)
        eta_n = eng.eta(k[idx], a[idx], d_nodes, t, wt)
        expected[idx] = np.clip(np.sum(weights * eta_n, axis=1), 0.0, 1.0)
        mass[idx] = np.sum(weights, axis=1)
        tol = MONOTONE_TOLERANCE * np.maximum(np.max(eta_n, axis=1), 1e-300)
        mono = np.all(np.diff(eta_n, axis=1) <= tol[:, None], axis=1)
        q = np.array([0.95, 0.5, 0.05])
        if not percentiles:
            p05[idx] = p50[idx] = p95[idx] = np.nan
            method[idx] = "not_computed"
            mono = np.ones(idx.size, dtype=bool)
        elif np.any(mono):
            im = idx[mono]
            d_q = _rice_ppf(q, nu[im], s[im])
            eq = eng.eta(k[im], a[im], d_q, t, wt)
            p05[im], p50[im], p95[im] = eq[:, 0], eq[:, 1], eq[:, 2]
            method[im] = "exact_monotone"
        for i in idx[~mono] if percentiles else ():
            n_fallback += 1
            # Synthetic Airy pattern with the same k = pi D/(lambda R) (only k enters the
            # pattern): D = 1e-6 pi / k gives N_F = D k / pi = 1e-6, well inside far field.
            d_syn = 1.0e-6 * math.pi / float(k[i])
            pattern = AiryPattern(
                aperture_diameter_m=d_syn,
                wavelength_m=math.pi * d_syn / float(k[i]),
                range_m=1.0,
                max_fresnel_number=0.5,
            )
            st = capture_statistics(pattern, float(a[i]), float(nu[i]), float(s[i]))
            p05[i], p50[i], p95[i] = st.p05, st.p50, st.p95
            method[i] = st.percentile_method
    return BatchCaptureResult(
        nominal=nominal,
        bias_only=bias_only,
        expected=expected,
        p05=p05,
        p50=p50,
        p95=p95,
        offset_probability_mass=mass,
        percentile_method=method,
        n_fallback=n_fallback,
        backend=backend,
        band_panels=band_panels,
        offset_panels=off_panels,
        bessel_quarter_nodes=eng.m_quarter,
    )


def evaluate_beam_batched(
    range_m: ArrayLike,
    elevation_rad: ArrayLike,
    settings: BeamSettings,
    *,
    backend: BatchBackend = "numpy",
    epochs_tdb_s: ArrayLike | None = None,
) -> BeamResult:
    """:func:`helioswarm.beam.evaluate_beam` for ``airy_far_field`` through the batched kernel.

    Same inputs, outputs and validity checks as the M5 function; only the capture quadrature
    is batched (module docstring). The peak irradiance per watt ``pi D^2/(4 lambda^2 R^2)``,
    the first-null radius and the centred-ellipse projection error use the same closed forms.

    Raises:
        NotImplementedError: any beam model other than ``airy_far_field``.
        BeamModelValidityError: ``N_F`` above the far-field limit at some epoch.
    """
    if settings.beam_model != "airy_far_field":
        raise NotImplementedError(
            f"the batched capture kernel implements airy_far_field only, not "
            f"'{settings.beam_model}'; use backend='numpy'"
        )
    r = np.atleast_1d(np.asarray(range_m, dtype=np.float64))
    el = np.atleast_1d(np.asarray(elevation_rad, dtype=np.float64))
    if r.shape != el.shape or r.ndim != 1:
        raise ValueError("range_m and elevation_rad must be 1-D arrays of equal length")
    d_ap = settings.aperture.diameter_m
    lam = settings.wavelength_m
    nf = fresnel_number(d_ap, lam, r)
    if np.any(nf > settings.max_fresnel_number):
        raise BeamModelValidityError(
            f"airy_far_field outside its validity range: max N_F = {float(np.max(nf)):.6g} > "
            f"{settings.max_fresnel_number}"
        )
    a_eff = projected_receiver_radius_m(
        settings.receiver_radius_m, el, settings.receiver_orientation
    )
    k = math.pi * d_ap / (lam * r)
    stats_b = airy_capture_batch(
        k,
        a_eff,
        r * settings.pointing_bias_rad,
        r * settings.pointing_sigma_rad,
        backend=backend,
    )
    factor = settings.wavefront_efficiency * settings.lattice_main_lobe_fraction
    proj_err = np.zeros(r.size)
    if settings.receiver_orientation == "horizontal":
        n_ang = 256
        psi = 2.0 * math.pi * np.arange(n_ang) / n_ang
        a1 = settings.receiver_radius_m
        a2 = a1 * np.sin(el)
        rr = (
            a1
            * a2[:, None]
            / np.sqrt((a2[:, None] * np.cos(psi)[None, :]) ** 2 + (a1 * np.sin(psi)[None, :]) ** 2)
        )
        x = k[:, None] * rr
        ee = np.where(
            x < SMALL_X, x**2 / 4.0 - x**4 / 32.0, 1.0 - special.j0(x) ** 2 - special.j1(x) ** 2
        )
        ellipse = np.mean(ee, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            proj_err = np.where(ellipse > 0.0, (stats_b.nominal - ellipse) / ellipse, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pointing_loss = np.where(stats_b.nominal > 0.0, stats_b.expected / stats_b.nominal, np.nan)
    feature = lam * r / d_ap
    peak = k**2 / (4.0 * math.pi)
    diag = BeamFieldDiagnostics(
        taper_efficiency=settings.aperture.taper_efficiency,
        source_spillover_fraction=settings.aperture.source_spillover_fraction,
        edge_intensity_ratio=settings.aperture.edge_intensity_ratio,
        feature_scale_m=feature,
        first_null_radius_m=airy_first_null_radius_m(lam, r, d_ap),
        gaussian_beam_radius_m=np.full(r.size, np.nan),
        receiver_angular_radius_rad=a_eff / r,
        receiver_projection_error=proj_err,
        offset_probability_mass=stats_b.offset_probability_mass,
        percentile_method=stats_b.percentile_method,
        lattice_element_count=0 if settings.lattice is None else settings.lattice.element_count,
    )
    notes = (
        "scalar diffraction (no polarisation/vector effects, no thermal distortion)",
        "single transmitter; no coherent combining across satellites",
        f"receiver orientation '{settings.receiver_orientation}': equal-projected-area circle",
        "pointing: bias + isotropic 2-D Gaussian jitter, Rice offset, deterministic quadrature",
        "wavefront loss exp(-(2 pi sigma_w/lambda)^2); scattered power assumed not captured",
        "peak irradiance of the aberration-free pattern (upper bound)",
        f"far-field model valid: max N_F = {float(np.max(nf)):.6g} <= "
        f"{settings.max_fresnel_number}",
        f"batched capture kernel (helioswarm.capture_batch), backend '{backend}': band panels "
        f"{stats_b.band_panels}, offset panels {stats_b.offset_panels}, per-point fallbacks "
        f"{stats_b.n_fallback}",
    )
    return BeamResult(
        wavelength_m=lam,
        aperture_m=d_ap,
        range_m=r,
        elevation_rad=el,
        beam_model=settings.beam_model,
        mode=settings.mode,
        fresnel_number=nf,
        fresnel_number_limit=settings.max_fresnel_number,
        capture_fraction=factor * stats_b.expected,
        capture_fraction_nominal=factor * stats_b.nominal,
        capture_fraction_bias_only=factor * stats_b.bias_only,
        capture_fraction_p05=factor * stats_b.p05,
        capture_fraction_p50=factor * stats_b.p50,
        capture_fraction_p95=factor * stats_b.p95,
        pointing_loss=pointing_loss,
        wavefront_efficiency=settings.wavefront_efficiency,
        lattice_main_lobe_fraction=settings.lattice_main_lobe_fraction,
        projected_receiver_radius_m=a_eff,
        pointing_bias_rad=settings.pointing_bias_rad,
        pointing_sigma_rad=settings.pointing_sigma_rad,
        peak_irradiance_per_W_m2=peak,
        peak_irradiance_W_m2=None,
        exceeds_max_power_density=None,
        receiver_orientation=settings.receiver_orientation,
        diagnostics=diag,
        validity_notes=notes,
        epochs_tdb_s=(None if epochs_tdb_s is None else np.asarray(epochs_tdb_s, dtype=np.float64)),
    )


__all__ = [
    "GL_ORDER",
    "BatchBackend",
    "BatchCaptureResult",
    "airy_capture_batch",
    "bessel_j01_trapezoid",
    "evaluate_beam_batched",
    "trapezoid_bessel_nodes",
]
