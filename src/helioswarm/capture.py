"""Finite-receiver capture, receiver projection and pointing statistics (PDF §9.4-9.5).

All functions take a circularly symmetric beam pattern ``I(rho)`` per watt
(:class:`helioswarm.aperture.RadialBeamPattern`) in the plane normal to the beam at the
receiver, and work in that plane (SI, metres).

Capture of an offset circular receiver (PDF §9.4)
-------------------------------------------------
A receiver disk of radius ``a`` whose centre is offset by ``d`` from the beam axis captures::

    eta(d) = int_0^inf I(rho) * 2 phi(rho; d, a) * rho drho

where ``2 phi rho`` is the length of the circle of radius ``rho`` (centred on the beam axis)
that lies inside the disk. By the law of cosines, a point at radius ``rho`` and azimuth ``psi``
(measured from the receiver centre direction) is inside iff
``rho^2 + d^2 - 2 rho d cos psi <= a^2``, so::

    phi(rho; d, a) = arccos( clip( (rho^2 + d^2 - a^2) / (2 rho d), -1, 1 ) )

which is ``pi`` (full circle) for ``rho <= a - d`` and 0 for ``rho >= a + d``. The integral is
split into the full-circle part ``EE(max(a - d, 0))`` (closed form for Airy and Gaussian
patterns, quadrature otherwise) and the band ``|a - d| <= rho <= a + d``. In the band the
substitution ``rho = m - h cos t`` (``m, h`` = band centre and half width, ``t in [0, pi]``)
removes the square-root behaviour of ``phi`` at both ends, and the band is integrated with
composite Gauss-Legendre panels of at most ``feature_scale / 2`` in ``rho``. Exact (no
approximation beyond quadrature); ``d = 0`` reduces to the encircled energy ``EE(a)``.
Cross-checks: a converged 2-D Cartesian grid (:func:`capture_fraction_cartesian`) and the
closed-form ``EE`` (tests).

Receiver projection
-------------------
``receiver.orientation = horizontal`` (a flat field on the ground, normal = local up): seen
along a beam arriving at elevation ``el`` the disk projects to an ellipse with semi-axes
``a`` and ``a sin(el)``, area ``pi a^2 sin(el)``. The model replaces it by the **circle of equal
projected area**, radius ``a_eff = a sqrt(sin el)`` (named approximation). The intensity
variation over the receiver depth (``2 a cos el`` along the beam, a relative range change
``<= 2a/R ~ 1e-8``) is neglected. The exact ellipse is available for comparison by 2-D
integration (:func:`capture_fraction_ellipse_cartesian`) and, centred, by the exact polar form
(:func:`capture_fraction_centred_ellipse`). For a centred receiver the leading
error of the equal-area circle is second order in the beam curvature: with
``I ~ I0 (1 - kappa rho^2)``, ``int_ellipse rho^2 = pi a b (a^2 + b^2)/4`` vs
``pi (ab)^2 / 2`` for the circle, i.e. a relative error ``kappa (a - b)^2 / 4 / (1 - ...)``,
negligible whenever the receiver is small compared with the beam (the heliocentric regime) and
also when it is much larger (both capture ~1). Measured worst cases are in
``docs/model_assumptions.md`` (M5). ``receiver.orientation = tracking`` keeps the receiver
normal on the beam (``a_eff = a``).

Pointing (PDF §9.5)
-------------------
A deterministic bias of magnitude ``b`` [rad] plus isotropic 2-D Gaussian jitter of per-axis
standard deviation ``sigma`` [rad] (small-angle model) displace the beam axis at the receiver
by a 2-D Gaussian vector of mean ``R b`` and per-axis std ``R sigma``. Its length is
Rice-distributed, ``d ~ Rice(nu = R b, s = R sigma)``, with density
``f(d) = (d/s^2) exp(-(d^2 + nu^2)/(2 s^2)) I0(d nu / s^2)``. ``scipy.stats.rice`` uses the
shape ``b_scipy = nu/s`` and ``scale = s`` (verified in its source: ``_pdf(x, b) = x
exp(-(x-b)^2/2) i0e(x b)``). The beam is circularly symmetric, so capture depends only on
``|d|`` and::

    E[eta] = int_0^inf eta(d) f(d) dd

evaluated by deterministic composite Gauss-Legendre quadrature over
``[max(0, nu - 9 s), nu + 9 s]`` (tail mass < 1e-17) with panels of at most
``min(s, feature_scale) / 2``. No random sampling is involved (deterministic physics); a
Monte-Carlo cross-check with an explicit ``numpy.random.Generator`` lives in the tests.

Large bias-to-jitter ratio (named approximation). ``scipy.stats.rice`` returns NaN from ``cdf``
and ``ppf`` for shape ``b = nu/s >~ 2.5e5`` (scipy 1.18.1), and for ``s < eps * nu`` the offsets
``nu +- k s`` are not even representable in float64. For ``b >= RICE_GAUSSIAN_LIMIT_SHAPE``
(1e4) the offset law is therefore replaced by its large-``b`` limit
``d ~ N(nu + s^2/(2 nu), s^2 (1 + 1/(2 b^2)))`` (from ``d = |nu e_x + (x, y)| = nu + x +
y^2/(2 nu) + O(s^3/nu^2)``). Measured against scipy for ``b = 1e2 ... 1e5`` its quantiles
differ by ``~0.83 s / b^2`` and its CDF by ``~0.12 / b^2``, i.e. <= 1e-8 s and 1.3e-9 at the
switch-over. The expectation is then a Gauss-Legendre quadrature in the standardised variable
``t = (d - mean)/sd`` over ``[-9, 9]`` with standard-normal weights, so the probability mass
stays exact even when the offsets collapse onto ``nu`` in floating point (the result is then
``eta(nu)``, the correct limit). Non-finite offset-law values always raise.

Percentiles of the capture distribution: if ``eta(d)`` is non-increasing in ``d`` on the
quadrature nodes (the usual case: receiver within the central lobe), the ``q``-quantile of
``eta`` is exactly ``eta(d_{1-q})`` with ``d_{1-q}`` the offset ``(1-q)``-quantile
(``method = "exact_monotone"``). Otherwise (rings of the pattern inside the offset window)
``method = "level_set"``: ``eta`` is sampled on a uniform grid of spacing
``min(s, feature_scale) / 32`` across the window and interpolated by a cubic spline
``eta~(d)``; ``P(eta <= y)`` is the offset-law probability of the level set
``{d : eta~(d) <= y}``, computed from the spline roots of ``eta~(d) = y`` and exact CDF
differences, and each quantile is located by bisection on ``y`` to 1e-13. The only
approximation is the spline (interpolation error fourth order in the grid spacing); against an
independent root-finding reference (brentq on ``eta(d) = y`` with exact Rice CDF differences)
the percentiles agree to <= 1.2e-9 absolute (test tolerance 1e-8,
``tests/unit/test_capture.py``). With zero jitter the distribution is a point mass
(``"deterministic"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats
from scipy.interpolate import CubicSpline

from helioswarm.aperture import GL_ORDER_RADIAL, RadialBeamPattern, composite_gauss_legendre

Float64Array = NDArray[np.float64]

ReceiverOrientation = Literal["horizontal", "tracking"]
PercentileMethod = Literal["deterministic", "exact_monotone", "level_set"]

RICE_GAUSSIAN_LIMIT_SHAPE: Final[float] = 1.0e4
"""Bias-to-jitter ratio ``nu/s`` from which the offset law is its Gaussian limit (docstring)."""

PERCENTILE_GRID_PER_SCALE: Final[int] = 32
"""Samples of ``eta(d)`` per ``min(s, feature_scale)`` for the level-set percentiles."""

MAX_PERCENTILE_POINTS: Final[int] = 200_000
"""Budget of ``eta(d)`` samples for the level-set percentiles; exceeding it raises."""

PERCENTILE_Y_TOLERANCE: Final[float] = 1.0e-13
"""Bisection tolerance on the capture level ``y`` of the level-set percentiles."""

RICE_SPAN_SIGMAS: Final[float] = 9.0
"""Half-width of the offset quadrature window in units of the jitter ``s`` (the Rice tail mass
beyond ``nu + 9 s`` is below 3e-18 for any ``nu``)."""

MAX_OFFSET_PANELS: Final[int] = 4096
"""Budget of offset-quadrature panels; exceeding it raises (jitter >> beam feature scale)."""

MAX_BAND_PANELS: Final[int] = 20_000
"""Budget of band-quadrature panels; exceeding it raises (receiver >> beam feature scale)."""

MONOTONE_TOLERANCE: Final[float] = 1.0e-12
"""Absolute tolerance (times the largest node capture) of the non-increasing check."""


def arc_half_angle_rad(
    rho_m: ArrayLike, offset_m: ArrayLike, receiver_radius_m: float
) -> Float64Array:
    """Half-angle ``phi`` of the arc of the circle ``|x| = rho`` inside the offset receiver disk.

    ``phi = arccos(clip((rho^2 + d^2 - a^2) / (2 rho d), -1, 1))`` (law of cosines); ``pi`` when
    the whole circle is inside (``rho <= a - d``), 0 when none is (``rho >= a + d``). ``d = 0`` or
    ``rho = 0`` give ``pi`` inside the disk and 0 outside. Broadcasts ``rho_m`` and ``offset_m``.
    """
    rho = np.asarray(rho_m, dtype=np.float64)
    d = np.asarray(offset_m, dtype=np.float64)
    a = float(receiver_radius_m)
    rho_b, d_b = np.broadcast_arrays(rho, d)
    denom = 2.0 * rho_b * d_b
    degenerate = denom <= 0.0
    safe = np.where(degenerate, 1.0, denom)
    cos_phi = np.clip((rho_b**2 + d_b**2 - a * a) / safe, -1.0, 1.0)
    phi = np.arccos(cos_phi)
    inside = np.where(rho_b + d_b < a, math.pi, 0.0)
    out: Float64Array = np.where(degenerate, inside, phi)
    return out


def _band_nodes(
    offset_m: Float64Array, receiver_radius_m: float, feature_scale_m: float, refinement: float
) -> tuple[Float64Array, Float64Array]:
    """``rho`` nodes and weights (incl. the Jacobian) on each band ``[|a-d|, a+d]``."""
    a = receiver_radius_m
    lo = np.abs(a - offset_m)
    hi = a + offset_m
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    h_max = float(np.max(half)) if half.size else 0.0
    panels = math.ceil(refinement * 2.0 * math.pi * h_max / (0.5 * feature_scale_m)) + 2
    if panels > MAX_BAND_PANELS:
        raise ValueError(
            f"offset-capture band needs {panels} panels (> {MAX_BAND_PANELS}): receiver/offset "
            f"half-width {h_max:.6g} m is too large relative to the beam feature scale "
            f"{feature_scale_m:.6g} m for direct quadrature"
        )
    t, wt = composite_gauss_legendre(np.linspace(0.0, math.pi, panels + 1), GL_ORDER_RADIAL)
    rho = mid[:, None] - half[:, None] * np.cos(t)[None, :]
    w = half[:, None] * np.sin(t)[None, :] * wt[None, :]
    return rho, w


def capture_fraction(
    pattern: RadialBeamPattern,
    receiver_radius_m: float,
    offset_m: ArrayLike,
    *,
    refinement: float = 1.0,
) -> Float64Array:
    """Fraction of the beam power captured by a disk of radius ``a`` offset by ``d`` (exact 1-D).

    ``eta(d) = EE(max(a-d, 0)) + int_{|a-d|}^{a+d} I(rho) 2 phi(rho; d, a) rho drho`` (module
    docstring), PDF §9.4.

    Args:
        pattern: Beam pattern per watt in the plane normal to the beam.
        receiver_radius_m: Disk radius ``a`` [m] (> 0).
        offset_m: Offset(s) ``d`` [m] of the disk centre from the beam axis (>= 0).
        refinement: Multiplies the number of band quadrature panels (convergence studies).

    Returns:
        ``eta(d)`` in [0, 1], shape of ``offset_m``.
    """
    a = float(receiver_radius_m)
    if not (math.isfinite(a) and a > 0.0):
        raise ValueError("receiver_radius_m must be finite and > 0")
    d = np.asarray(offset_m, dtype=np.float64)
    flat = d.ravel()
    if np.any(~np.isfinite(flat)) or np.any(flat < 0.0):
        raise ValueError("offset_m must be finite and >= 0")
    inner = pattern.encircled_energy(np.maximum(a - flat, 0.0))
    rho, w = _band_nodes(flat, a, pattern.feature_scale_m, refinement)
    integrand = (
        pattern.intensity_m2_inv(rho) * 2.0 * arc_half_angle_rad(rho, flat[:, None], a) * rho
    )
    band = np.sum(integrand * w, axis=1)
    out: Float64Array = np.clip(inner + band, 0.0, 1.0).reshape(d.shape)
    return out


def capture_fraction_cartesian(
    pattern: RadialBeamPattern, receiver_radius_m: float, offset_m: float, n_per_radius: int
) -> float:
    """Capture by direct 2-D Cartesian midpoint integration over the receiver disk (validation).

    The disk ``(x - d)^2 + y^2 <= a^2`` is covered by a square grid of ``2 n`` x ``2 n`` cells
    (cell size ``a / n``); each cell is split into 4 x 4 sub-cells whose centres are tested
    against the disk boundary (anti-aliasing of the edge), and the irradiance is evaluated at the
    sub-cell centres. Error O(cell size) at the edge; used to demonstrate convergence of the 1-D
    form under grid refinement (PDF §13.3).
    """
    a = float(receiver_radius_m)
    sub = 4
    m = 2 * n_per_radius * sub
    h = 2.0 * a / m
    c = -a + h * (np.arange(m) + 0.5)
    total = 0.0
    for row in np.array_split(np.arange(m), max(1, m // 512)):
        y = c[row][:, None]
        x = c[None, :]
        inside = x * x + y * y <= a * a
        rho = np.hypot(x + offset_m, y)
        total += float(np.sum(np.where(inside, pattern.intensity_m2_inv(rho), 0.0)))
    return total * h * h


def capture_fraction_ellipse_cartesian(
    pattern: RadialBeamPattern,
    semi_axis_x_m: float,
    semi_axis_y_m: float,
    offset_x_m: float,
    offset_y_m: float,
    n_per_axis: int,
) -> float:
    """Capture of an elliptical receiver (projected horizontal disk) by 2-D midpoint integration.

    Ellipse ``((x - dx)/A)^2 + ((y - dy)/B)^2 <= 1`` in the beam-normal plane, sampled on a
    ``n_per_axis`` x ``n_per_axis`` grid of the ellipse's bounding box (affine map of a square,
    so the cell count does not depend on the ellipticity). Used to quantify the equal-area-circle
    approximation of :func:`projected_receiver_radius_m`.
    """
    n = int(n_per_axis)
    u = -1.0 + (2.0 * np.arange(n) + 1.0) / n
    total = 0.0
    for row in np.array_split(np.arange(n), max(1, n // 512)):
        uy = u[row][:, None]
        ux = u[None, :]
        inside = ux * ux + uy * uy <= 1.0
        x = offset_x_m + semi_axis_x_m * ux
        y = offset_y_m + semi_axis_y_m * uy
        rho = np.hypot(x, y)
        total += float(np.sum(np.where(inside, pattern.intensity_m2_inv(rho), 0.0)))
    cell = (2.0 * semi_axis_x_m / n) * (2.0 * semi_axis_y_m / n)
    return total * cell


def capture_fraction_centred_ellipse(
    pattern: RadialBeamPattern, semi_axis_1_m: float, semi_axis_2_m: float, n_angles: int = 512
) -> float:
    """Capture of a centred elliptical receiver (exact reference for the projection error).

    In polar coordinates about the beam axis, ``eta = (1/2 pi) int_0^{2 pi} EE(r(psi)) dpsi``
    with the ellipse boundary ``r(psi) = A B / sqrt((B cos psi)^2 + (A sin psi)^2)``. The
    integrand is smooth and periodic, so the trapezoidal rule in ``psi`` converges
    exponentially (Trefethen & Weideman 2014, SIAM Rev. 56, 385).
    """
    psi = 2.0 * math.pi * np.arange(n_angles) / n_angles
    a1, a2 = float(semi_axis_1_m), float(semi_axis_2_m)
    r = a1 * a2 / np.sqrt((a2 * np.cos(psi)) ** 2 + (a1 * np.sin(psi)) ** 2)
    return float(np.mean(pattern.encircled_energy(r)))


def projected_receiver_radius_m(
    receiver_radius_m: float, elevation_rad: ArrayLike, orientation: ReceiverOrientation
) -> Float64Array:
    """Radius of the equal-projected-area circle of the receiver seen along the beam [m].

    ``horizontal``: ``a sqrt(sin el)`` (projected ellipse ``a`` x ``a sin el``, area
    ``pi a^2 sin el``); ``tracking``: ``a``. Elevations must lie in (0, pi/2].
    """
    el = np.asarray(elevation_rad, dtype=np.float64)
    if orientation == "tracking":
        return np.full(el.shape, float(receiver_radius_m))
    if orientation != "horizontal":
        raise ValueError(f"unknown receiver orientation '{orientation}'")
    if np.any(~np.isfinite(el)) or np.any(el <= 0.0) or np.any(el > 0.5 * math.pi + 1e-12):
        raise ValueError("elevation_rad must lie in (0, pi/2] for a horizontal receiver")
    out: Float64Array = receiver_radius_m * np.sqrt(np.sin(el))
    return out


@dataclass(frozen=True)
class _OffsetLaw:
    """Law of the pointing-offset length ``d = |nu e_x + N(0, s^2 I)|`` (Rice; module docstring).

    ``nu`` [m] >= 0 and ``s`` [m] > 0. For ``nu/s >= RICE_GAUSSIAN_LIMIT_SHAPE`` the named
    Gaussian limit ``N(nu + s^2/(2 nu), s^2 (1 + s^2/(2 nu^2)))`` is used.
    """

    nu: float
    s: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.nu) and self.nu >= 0.0):
            raise ValueError("bias offset must be finite and >= 0")
        if not (math.isfinite(self.s) and self.s > 0.0):
            raise ValueError("jitter offset must be finite and > 0")

    @property
    def gaussian(self) -> bool:
        """True in the large bias-to-jitter regime (Gaussian limit)."""
        return self.nu >= RICE_GAUSSIAN_LIMIT_SHAPE * self.s

    @property
    def mean_m(self) -> float:
        """Mean of the Gaussian limit ``nu + s^2/(2 nu)`` [m] (Gaussian regime only)."""
        return self.nu + 0.5 * self.s * self.s / self.nu

    @property
    def sd_m(self) -> float:
        """Standard deviation of the Gaussian limit ``s sqrt(1 + s^2/(2 nu^2))`` [m]."""
        return self.s * math.sqrt(1.0 + 0.5 * (self.s / self.nu) ** 2)

    def window(self) -> tuple[float, float]:
        """Offset quadrature window (tail mass outside < 3e-18) [m]."""
        if self.gaussian:
            return (
                self.mean_m - RICE_SPAN_SIGMAS * self.sd_m,
                self.mean_m + RICE_SPAN_SIGMAS * self.sd_m,
            )
        return (max(0.0, self.nu - RICE_SPAN_SIGMAS * self.s), self.nu + RICE_SPAN_SIGMAS * self.s)

    @staticmethod
    def _finite(values: Float64Array, what: str) -> Float64Array:
        if np.any(~np.isfinite(values)):
            raise ValueError(f"pointing-offset {what} is not finite (scipy.stats.rice failure)")
        return values

    def pdf(self, d_m: ArrayLike) -> Float64Array:
        """Density of ``d`` [m^-1]."""
        d = np.asarray(d_m, dtype=np.float64)
        if self.gaussian:
            out: Float64Array = np.exp(-0.5 * ((d - self.mean_m) / self.sd_m) ** 2) / (
                math.sqrt(2.0 * math.pi) * self.sd_m
            )
        else:
            out = stats.rice.pdf(d, self.nu / self.s, scale=self.s)
        return self._finite(out, "density")

    def cdf(self, d_m: ArrayLike) -> Float64Array:
        """Distribution function of ``d`` [1]."""
        d = np.asarray(d_m, dtype=np.float64)
        if self.gaussian:
            out: Float64Array = special.ndtr((d - self.mean_m) / self.sd_m)
        else:
            out = stats.rice.cdf(d, self.nu / self.s, scale=self.s)
        return self._finite(out, "CDF")

    def ppf(self, q: ArrayLike) -> Float64Array:
        """Quantile function of ``d`` [m]."""
        qq = np.asarray(q, dtype=np.float64)
        if self.gaussian:
            out: Float64Array = self.mean_m + self.sd_m * special.ndtri(qq)
        else:
            out = stats.rice.ppf(qq, self.nu / self.s, scale=self.s)
        return self._finite(out, "quantile")

    def quadrature(
        self, feature_scale_m: float, refinement: float
    ) -> tuple[Float64Array, Float64Array]:
        """Offset nodes [m] and probability masses (composite Gauss-Legendre).

        Rice regime: panels of at most ``min(s, feature_scale)/2`` in ``d``, masses
        ``w_i f(d_i)``. Gaussian regime: the same panel size in ``t = (d - mean)/sd`` over
        ``[-9, 9]``, masses ``w_i phi(t_i)`` (exact mass even when ``d_i`` collapse onto ``nu``).
        """
        if self.gaussian:
            sd = self.sd_m
            width_t = 0.5 * min(1.0, feature_scale_m / sd) / refinement
            panels = math.ceil(2.0 * RICE_SPAN_SIGMAS / width_t) + 1
            _check_offset_panels(panels, self.s, feature_scale_m)
            t, wt = composite_gauss_legendre(
                np.linspace(-RICE_SPAN_SIGMAS, RICE_SPAN_SIGMAS, panels + 1), GL_ORDER_RADIAL
            )
            mass_w: Float64Array = wt * np.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
            d: Float64Array = self.mean_m + sd * t
            return d, mass_w
        d_lo, d_hi = self.window()
        width = 0.5 * min(self.s, feature_scale_m) / refinement
        panels = math.ceil((d_hi - d_lo) / width) + 1
        _check_offset_panels(panels, self.s, feature_scale_m)
        d, wd = composite_gauss_legendre(np.linspace(d_lo, d_hi, panels + 1), GL_ORDER_RADIAL)
        return d, wd * self.pdf(d)


def _check_offset_panels(panels: int, s: float, feature_scale_m: float) -> None:
    if panels > MAX_OFFSET_PANELS:
        raise ValueError(
            f"pointing-offset quadrature needs {panels} panels (> {MAX_OFFSET_PANELS}): the "
            f"jitter s = {s:.6g} m is much larger than the beam feature scale "
            f"{feature_scale_m:.6g} m"
        )


def rice_offset_pdf_m_inv(
    offset_m: ArrayLike, bias_offset_m: float, sigma_offset_m: float
) -> Float64Array:
    """Density of the pointing-offset length ``f(d; nu, s)`` [m^-1].

    Rice (``scipy.stats.rice`` with shape ``nu/s``, scale ``s``); its Gaussian limit for
    ``nu/s >= RICE_GAUSSIAN_LIMIT_SHAPE`` (module docstring). ``sigma_offset_m`` must be > 0.
    """
    return _OffsetLaw(float(bias_offset_m), float(sigma_offset_m)).pdf(offset_m)


def rice_offset_cdf(
    offset_m: ArrayLike, bias_offset_m: float, sigma_offset_m: float
) -> Float64Array:
    """Distribution function of the pointing-offset length ``P(d <= offset)`` [1] (as above)."""
    return _OffsetLaw(float(bias_offset_m), float(sigma_offset_m)).cdf(offset_m)


def rice_offset_quantile_m(
    q: ArrayLike, bias_offset_m: float, sigma_offset_m: float
) -> Float64Array:
    """Quantile of the pointing-offset length [m] (Rice, or its Gaussian limit as above)."""
    return _OffsetLaw(float(bias_offset_m), float(sigma_offset_m)).ppf(q)


def _level_set_quantiles(
    pattern: RadialBeamPattern,
    receiver_radius_m: float,
    law: _OffsetLaw,
    probabilities: tuple[float, ...],
    refinement: float,
) -> tuple[float, ...]:
    """Quantiles of ``eta(d)`` for a non-monotone ``eta`` (``"level_set"``, module docstring)."""
    lo, hi = law.window()
    scale = law.sd_m if law.gaussian else law.s
    h = min(scale, pattern.feature_scale_m) / (PERCENTILE_GRID_PER_SCALE * refinement)
    n = math.ceil((hi - lo) / h) + 1
    if n > MAX_PERCENTILE_POINTS:
        raise ValueError(
            f"level-set percentiles need {n} capture samples (> {MAX_PERCENTILE_POINTS}): the "
            "pointing jitter is much larger than the beam feature scale"
        )
    d = np.linspace(lo, hi, max(n, 4))
    if not np.all(np.diff(d) > 0.0):  # offsets collapsed in floating point: a point mass
        eta0 = float(capture_fraction(pattern, receiver_radius_m, d[:1], refinement=refinement)[0])
        return tuple(eta0 for _ in probabilities)
    eta = capture_fraction(pattern, receiver_radius_m, d, refinement=refinement)
    spline = CubicSpline(d, eta)
    cdf_lo, cdf_hi = (float(v) for v in law.cdf(np.array([lo, hi])))
    total = cdf_hi - cdf_lo

    def prob_at_most(y: float) -> float:
        roots = np.asarray(spline.solve(y, discontinuity=False, extrapolate=False))
        roots = roots[np.isfinite(roots)]
        pts = np.unique(np.concatenate(([lo], np.clip(roots, lo, hi), [hi])))
        below = spline(0.5 * (pts[:-1] + pts[1:])) <= y
        return float(np.sum(np.diff(law.cdf(pts))[below]) / total)

    span = float(np.max(eta) - np.min(eta))
    out = []
    for q in probabilities:
        y_lo = float(np.min(eta)) - 0.05 * span - 1e-12
        y_hi = float(np.max(eta)) + 0.05 * span + 1e-12
        while prob_at_most(y_lo) >= q:  # spline undershoot below the samples (defensive)
            y_lo -= span + 1e-12
        while y_hi - y_lo > PERCENTILE_Y_TOLERANCE:
            mid = 0.5 * (y_lo + y_hi)
            if prob_at_most(mid) >= q:
                y_hi = mid
            else:
                y_lo = mid
        out.append(min(max(y_hi, 0.0), 1.0))
    return tuple(out)


@dataclass(frozen=True)
class CaptureStatistics:
    """Capture of one receiver at one epoch under pointing bias and jitter (diffraction only).

    Attributes:
        nominal: ``eta(0)`` = encircled energy within the receiver radius (perfect pointing).
        bias_only: ``eta(nu)`` with the deterministic bias and no jitter.
        expected: ``E[eta(d)]``, ``d ~ Rice(nu, s)``.
        p05, p50, p95: 5th, 50th and 95th percentiles of the capture distribution.
        percentile_method: How the percentiles were obtained (module docstring):
            deterministic, exact_monotone or level_set.
        offset_probability_mass: Quadrature integral of the Rice density over the window
            (1 up to quadrature and the < 3e-18 tail; reported as a check).
        n_offset_nodes: Number of offset quadrature nodes (0 without jitter).
    """

    nominal: float
    bias_only: float
    expected: float
    p05: float
    p50: float
    p95: float
    percentile_method: PercentileMethod
    offset_probability_mass: float
    n_offset_nodes: int


def capture_statistics(
    pattern: RadialBeamPattern,
    receiver_radius_m: float,
    bias_offset_m: float,
    sigma_offset_m: float,
    *,
    refinement: float = 1.0,
) -> CaptureStatistics:
    """Nominal, bias-only and expected capture plus percentiles under pointing errors (PDF §9.5).

    Args:
        pattern: Beam pattern per watt at the receiver range.
        receiver_radius_m: (Projected) receiver radius ``a`` [m].
        bias_offset_m: Bias offset at the receiver ``nu = R * bias_rad`` [m] (>= 0).
        sigma_offset_m: Per-axis jitter at the receiver ``s = R * sigma_rad`` [m] (>= 0).
        refinement: Multiplies all quadrature panel counts (convergence studies).

    Returns:
        :class:`CaptureStatistics` (deterministic quadrature; no sampling).
    """
    nu = float(bias_offset_m)
    s = float(sigma_offset_m)
    if not (math.isfinite(nu) and nu >= 0.0 and math.isfinite(s) and s >= 0.0):
        raise ValueError("bias_offset_m and sigma_offset_m must be finite and >= 0")
    a = float(receiver_radius_m)
    nominal = float(capture_fraction(pattern, a, np.array([0.0]), refinement=refinement)[0])
    bias_only = (
        nominal
        if nu == 0.0
        else float(capture_fraction(pattern, a, np.array([nu]), refinement=refinement)[0])
    )
    if s == 0.0:
        return CaptureStatistics(
            nominal=nominal,
            bias_only=bias_only,
            expected=bias_only,
            p05=bias_only,
            p50=bias_only,
            p95=bias_only,
            percentile_method="deterministic",
            offset_probability_mass=1.0,
            n_offset_nodes=0,
        )
    law = _OffsetLaw(nu, s)
    d, mass_w = law.quadrature(pattern.feature_scale_m, refinement)
    eta = capture_fraction(pattern, a, d, refinement=refinement)
    expected = float(np.clip(np.sum(mass_w * eta), 0.0, 1.0))
    mass = float(np.sum(mass_w))
    tol = MONOTONE_TOLERANCE * max(float(np.max(eta)), 1e-300)
    method: PercentileMethod
    if np.all(np.diff(eta) <= tol):
        method = "exact_monotone"
        d_q = law.ppf(np.array([0.95, 0.5, 0.05]))
        p05, p50, p95 = (
            float(v)
            for v in capture_fraction(pattern, a, np.maximum(d_q, 0.0), refinement=refinement)
        )
    else:
        method = "level_set"
        p05, p50, p95 = _level_set_quantiles(pattern, a, law, (0.05, 0.5, 0.95), refinement)
    return CaptureStatistics(
        nominal=nominal,
        bias_only=bias_only,
        expected=expected,
        p05=p05,
        p50=p50,
        p95=p95,
        percentile_method=method,
        offset_probability_mass=mass,
        n_offset_nodes=int(d.size),
    )


__all__ = [
    "MAX_BAND_PANELS",
    "MAX_OFFSET_PANELS",
    "MAX_PERCENTILE_POINTS",
    "PERCENTILE_GRID_PER_SCALE",
    "RICE_GAUSSIAN_LIMIT_SHAPE",
    "RICE_SPAN_SIGMAS",
    "CaptureStatistics",
    "PercentileMethod",
    "ReceiverOrientation",
    "arc_half_angle_rad",
    "capture_fraction",
    "capture_fraction_cartesian",
    "capture_fraction_centred_ellipse",
    "capture_fraction_ellipse_cartesian",
    "capture_statistics",
    "projected_receiver_radius_m",
    "rice_offset_cdf",
    "rice_offset_pdf_m_inv",
    "rice_offset_quantile_m",
]
