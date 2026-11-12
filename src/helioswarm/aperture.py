"""Aperture fields and the scalar-diffraction propagation core (PDF §9.1-9.3).

This module owns three things that the optical (``optical.py``) and microwave
(``microwave.py``) beam models share, so that neither re-implements them:

1. **Aperture illumination** :class:`ApertureField` -- the radial field amplitude ``A(r)``
   across a circular aperture of radius ``a = D/2``, zero outside the edge, normalised so that
   the power leaving the aperture is one watt::

       integral_0^a |A(r)|^2 2 pi r dr = 1          (A in m^-1, |A|^2 in W m^-2 per W)

   * ``uniform_circular``: ``A = 1 / sqrt(pi a^2)``.
   * ``truncated_gaussian``: ``A = c exp(-r^2 / w0^2)`` (``w0`` = 1/e^2 *intensity* radius),
     ``c^2 = 1 / [(pi w0^2 / 2)(1 - exp(-2 a^2 / w0^2))]``.
   * ``parabolic_on_pedestal``: ``A = c [C + (1 - C)(1 - r^2/a^2)]`` with the edge field
     ``C = 10^(-T/20)`` for an edge *intensity* taper ``T`` dB, ``c^2 = 1 / (pi a^2
     (C + (1 - C)^2 / 3))`` (Balanis, *Antenna Theory*, 4th ed., §15.4 / Table 15.1 aperture
     distributions; Silver, *Microwave Antenna Theory and Design*, 1949, §6.6).

   **Spill-over owner (one owner, M0 definition).** The illumination is the field *inside* the
   aperture and the beam models are normalised to the power in that field (``P_aperture``).
   Source power that falls outside the aperture edge before emission -- e.g. the fraction
   ``exp(-2 a^2 / w0^2)`` of a Gaussian laser beam clipped by the aperture -- is **not emitted**
   and is a transmitter-optics loss owned by ``transmitter.optics_or_rf_efficiency``
   (``power_chain``). :attr:`ApertureField.source_spillover_fraction` reports it as a
   diagnostic only; no beam model applies it. The truncated field is therefore *renormalised*
   inside the aperture, not attenuated.

2. **Scalar Fresnel / Fraunhofer propagation of a circularly symmetric field** (Goodman,
   *Introduction to Fourier Optics*, 3rd ed. 2005, eq. 4-17 and §4.4; the Fourier-Bessel
   (Hankel) transform §2.2.5). At distance ``z`` from the aperture plane, transverse radius
   ``rho``::

       U(rho; z) = (2 pi / (lambda z)) exp(i pi rho^2 / (lambda z))
                   * integral_0^a A(r) exp(i pi r^2 / (lambda z)) J0(2 pi r rho / (lambda z)) r dr

   (the constant phase ``exp(i k z) / i`` is dropped). ``I(rho) = |U|^2`` is the irradiance
   per watt of aperture power [m^-2]. Power is conserved exactly: with
   ``g(r) = A(r) exp(i pi r^2/(lambda z))`` and the Hankel-transform Parseval identity,
   ``integral |U|^2 2 pi rho drho = integral |g|^2 2 pi r dr = 1``. Dropping the aperture chirp
   gives the Fraunhofer (far-field) limit, which depends on ``rho`` and ``z`` only through the
   angle ``theta = rho / z`` (paraxial)::

       I(rho; z) = J(theta) / z^2,   J(theta) = (1/lambda^2) |2 pi integral_0^a A J0(2 pi r theta
                                                 / lambda) r dr|^2   [sr^-1 per W]

   Closed forms (``u = 2 pi a theta / lambda = pi D theta / lambda``):

   * uniform: ``J = (pi a^2 / lambda^2) [2 J1(u)/u]^2`` (Airy);
   * parabolic on pedestal (Lommel integral ``int_0^1 (1 - t^2) J0(ut) t dt = 2 J2(u)/u^2``):
     ``J = (c^2 (2 pi a^2)^2 / lambda^2) [C J1(u)/u + 2 (1-C) J2(u)/u^2]^2``;
   * truncated Gaussian: no elementary closed form; evaluated by quadrature.

   **Scalar diffraction only.** Polarisation, vector-field effects near grazing angles, the
   obliquity factor (``cos theta ~ 1`` at the micro-radian angles involved), non-linear optics
   and high-power thermal distortion are not modelled (PDF §9.2). Every ``BeamResult`` records
   ``scalar_diffraction = True``.

3. **Validity of the propagation regime.** The Fresnel number ``N_F = D^2 / (lambda R)`` is
   computed for every range. Far-field (Fraunhofer) models raise
   :class:`BeamModelValidityError` when ``N_F > models.fraunhofer_max_fresnel_number``
   (default 0.5, i.e. ``R >= 2 D^2 / lambda``; IEEE Std 145-2013, Balanis ch. 2). The Fresnel
   (paraxial) propagator itself is checked against Goodman's condition (eq. 4-18)
   ``z^3 >> pi (a + rho_max)^4 / (4 lambda)``. The ratio :func:`fresnel_paraxial_parameter`
   ``pi (a + rho)^4 / (4 lambda z^3)`` is the largest phase [rad] of the first neglected
   (quartic) term of the distance expansion over the evaluated region, so it bounds the phase
   error of the Fresnel integrand pointwise. It must not exceed
   :data:`FRESNEL_PARAXIAL_MAX_PHASE_RAD` = 0.1 rad: an **assumed engineering limit** that
   gives "``>>``" a numerical meaning (a factor 10). It bounds the integrand phase error, not
   the field error directly; Goodman §4.2.2 notes that, by a stationary-phase argument, the
   pointwise condition is overly stringent for the field, so the limit errs on the safe side.
   It is enforced at construction (for the declared ``paraxial_rho_max_m``) and again on every
   field evaluation (for the radii actually requested), so no evaluation -- including the peak
   search -- can leave the checked region.

Numerics: the Hankel integrals are evaluated by composite Gauss-Legendre quadrature in ``r``
with panels sized to the fastest local oscillation of the integrand (chirp ``r/(lambda z)`` plus
Bessel ``rho/(lambda z)`` cycles per metre) and, for a Gaussian, to the waist; each panel
holds :data:`GL_ORDER_HANKEL` nodes, i.e. at least that many nodes per oscillation cycle.
Convergence under refinement is tested (``tests/unit/test_aperture.py``).

The wavefront-error (Ruze) loss is owned here too (:func:`wavefront_efficiency`), so that the
optical and microwave models apply the same single factor exactly once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from helioswarm.config import ApertureIllumination, TransmitterConfig

Float64Array = NDArray[np.float64]
Complex128Array = NDArray[np.complex128]

GL_ORDER_HANKEL: Final[int] = 16
"""Gauss-Legendre nodes per panel of the Hankel (aperture) integrals."""

GL_ORDER_RADIAL: Final[int] = 8
"""Gauss-Legendre nodes per panel of transverse-plane (rho) integrals."""

_MAX_HANKEL_NODES: Final[int] = 2_000_000
"""Upper bound on the number of aperture quadrature nodes per evaluation (memory guard)."""

_MAX_HANKEL_WORK: Final[int] = 2_000_000_000
"""Upper bound on (output points x aperture nodes) Bessel evaluations per call (cost guard)."""

_MATRIX_CHUNK: Final[int] = 4_000_000
"""Maximum number of Bessel-matrix elements evaluated at once (memory guard)."""

_SMALL_U: Final[float] = 1.0e-4
"""Below this ``u`` the far-field closed forms use their Taylor series (avoids 0/0)."""

FRESNEL_PARAXIAL_MAX_PHASE_RAD: Final[float] = 0.1
"""Largest accepted neglected quartic phase ``pi (a + rho)^4 / (4 lambda z^3)`` [rad] of the
Fresnel-Hankel propagator (assumed engineering limit for Goodman's ``>>``; module docstring)."""


def _gauss_legendre(order: int) -> tuple[Float64Array, Float64Array]:
    x, w = np.polynomial.legendre.leggauss(order)
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


_GL_HANKEL: Final = _gauss_legendre(GL_ORDER_HANKEL)
_GL_RADIAL: Final = _gauss_legendre(GL_ORDER_RADIAL)


def composite_gauss_legendre(
    edges: ArrayLike, order: int = GL_ORDER_RADIAL
) -> tuple[Float64Array, Float64Array]:
    """Nodes and weights of composite Gauss-Legendre quadrature over consecutive panels.

    Args:
        edges: Panel boundaries, shape ``(..., n_panels + 1)``, non-decreasing along the last
            axis (leading axes are independent integrals).
        order: Nodes per panel (exact for polynomials of degree ``2 order - 1`` per panel).

    Returns:
        ``(nodes, weights)`` of shape ``(..., n_panels * order)``.
    """
    e = np.asarray(edges, dtype=np.float64)
    if order == GL_ORDER_HANKEL:
        xi, wi = _GL_HANKEL
    elif order == GL_ORDER_RADIAL:
        xi, wi = _GL_RADIAL
    else:
        xi, wi = _gauss_legendre(order)
    lo = e[..., :-1, None]
    half = 0.5 * (e[..., 1:, None] - lo)
    nodes = lo + half * (xi + 1.0)
    weights = half * wi
    shape = (*e.shape[:-1], -1)
    return nodes.reshape(shape), weights.reshape(shape)


# ---------------------------------------------------------------------------------------------
# Validity of the propagation regime
# ---------------------------------------------------------------------------------------------


class BeamModelValidityError(ValueError):
    """A beam model was asked for a configuration outside its documented validity range."""


def fresnel_number(
    aperture_diameter_m: float, wavelength_m: float, range_m: ArrayLike
) -> Float64Array:
    """Fresnel number ``N_F = D^2 / (lambda R)`` of the transmitting aperture (PDF §9.1).

    Args:
        aperture_diameter_m: Aperture diameter ``D`` [m].
        wavelength_m: Wavelength ``lambda`` [m].
        range_m: Propagation distance(s) ``R`` [m], > 0.

    Returns:
        ``N_F`` [1], same shape as ``range_m``. ``N_F << 1`` is the Fraunhofer (far-field)
        regime; the conventional far-field distance ``R = 2 D^2 / lambda`` is ``N_F = 0.5``.
    """
    r = np.asarray(range_m, dtype=np.float64)
    if np.any(~np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("range_m must be finite and > 0")
    out: Float64Array = aperture_diameter_m**2 / (wavelength_m * r)
    return out


def check_far_field_validity(
    model: str, fresnel_number_value: float, max_fresnel_number: float
) -> None:
    """Raise :class:`BeamModelValidityError` if a far-field model is used at ``N_F`` > limit.

    PDF §9.1: "The agent must not silently use a Fraunhofer Airy formula outside its validity
    regime." The limit is ``models.fraunhofer_max_fresnel_number``.
    """
    if not fresnel_number_value <= max_fresnel_number:
        raise BeamModelValidityError(
            f"beam model '{model}' is a Fraunhofer (far-field) model but the Fresnel number "
            f"N_F = D^2/(lambda R) = {fresnel_number_value:.6g} exceeds "
            f"models.fraunhofer_max_fresnel_number = {max_fresnel_number:.6g} (range too short "
            "for the far-field approximation); use models.beam='fresnel_hankel'"
        )


def fresnel_paraxial_parameter(
    aperture_radius_m: float, rho_max_m: float, wavelength_m: float, range_m: float
) -> float:
    """Largest neglected (quartic) phase of the Fresnel approximation [rad].

    ``pi (a + rho_max)^4 / (4 lambda z^3) = k L^4 / (8 z^3)`` with ``L = a + rho_max``: the
    maximum over the aperture and the evaluated region of the first term dropped from the
    binomial expansion of the source-to-point distance (Goodman 2005, eq. 4-18, which asks for
    ``z^3 >> pi L^4 / (4 lambda)``). The Fresnel-Hankel propagator accepts values up to
    :data:`FRESNEL_PARAXIAL_MAX_PHASE_RAD` (0.1 rad, an assumed limit; module docstring).
    """
    return math.pi * (aperture_radius_m + rho_max_m) ** 4 / (4.0 * wavelength_m * range_m**3)


# ---------------------------------------------------------------------------------------------
# Wavefront (phase) error: single owner of the Ruze / Strehl factor
# ---------------------------------------------------------------------------------------------


def ruze_efficiency(phase_rms_rad: float) -> float:
    """Ruze random phase-error efficiency ``exp(-sigma_phi^2)`` (Ruze 1966, Proc. IEEE 54, 633).

    Fraction of the aperture power that stays in the diffraction-limited (coherent) pattern
    when the aperture phase has a Gaussian random error of RMS ``sigma_phi`` [rad] with a
    correlation length much smaller than the aperture; the remainder is scattered into a wide
    halo. Equal to the extended Marechal approximation of the Strehl ratio
    (Mahajan 1983, JOSA 73, 860).
    """
    if not (math.isfinite(phase_rms_rad) and phase_rms_rad >= 0.0):
        raise ValueError("phase_rms_rad must be finite and >= 0")
    return math.exp(-(phase_rms_rad**2))


def wavefront_efficiency(rms_wavefront_error_m: float, wavelength_m: float) -> float:
    """Wavefront-error loss ``exp(-(2 pi sigma_w / lambda)^2)`` (M0 definition, one owner).

    ``sigma_w`` is the RMS *wavefront* (optical path) error, ``transmitter.rms_wavefront_error_m``
    (not a reflector surface error, which would carry an extra factor 2). This is the only place
    the factor is computed; ``beam.evaluate_beam`` applies it once to the captured fraction.
    """
    if not (math.isfinite(rms_wavefront_error_m) and rms_wavefront_error_m >= 0.0):
        raise ValueError("rms_wavefront_error_m must be finite and >= 0")
    if not (math.isfinite(wavelength_m) and wavelength_m > 0.0):
        raise ValueError("wavelength_m must be finite and > 0")
    return ruze_efficiency(2.0 * math.pi * rms_wavefront_error_m / wavelength_m)


# ---------------------------------------------------------------------------------------------
# Aperture field
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ApertureField:
    """Radial field amplitude across a circular aperture, normalised to 1 W (module docstring).

    Attributes:
        illumination: ``uniform_circular``, ``truncated_gaussian`` or ``parabolic_on_pedestal``.
        diameter_m: Aperture diameter ``D`` [m].
        gaussian_waist_radius_m: ``w0`` (1/e^2 intensity radius) [m], ``truncated_gaussian``
            only; must satisfy ``w0 <= D/2``.
        edge_taper_dB: Edge intensity taper ``T`` [dB] (centre/edge), ``parabolic_on_pedestal``
            only.
    """

    illumination: ApertureIllumination
    diameter_m: float
    gaussian_waist_radius_m: float | None = None
    edge_taper_dB: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.diameter_m) and self.diameter_m > 0.0):
            raise ValueError("diameter_m must be finite and > 0")
        if self.illumination == "truncated_gaussian":
            w0 = self.gaussian_waist_radius_m
            if w0 is None or not (math.isfinite(w0) and w0 > 0.0):
                raise ValueError("truncated_gaussian needs a finite gaussian_waist_radius_m > 0")
            if w0 > 0.5 * self.diameter_m:
                raise ValueError(
                    f"gaussian_waist_radius_m={w0} exceeds the aperture radius "
                    f"{0.5 * self.diameter_m}"
                )
        elif self.gaussian_waist_radius_m is not None:
            raise ValueError("gaussian_waist_radius_m is only meaningful for truncated_gaussian")
        if self.illumination == "parabolic_on_pedestal":
            t = self.edge_taper_dB
            if t is None or not (math.isfinite(t) and t > 0.0):
                raise ValueError("parabolic_on_pedestal needs a finite edge_taper_dB > 0")
        elif self.edge_taper_dB is not None:
            raise ValueError("edge_taper_dB is only meaningful for parabolic_on_pedestal")
        if self.illumination not in (
            "uniform_circular",
            "truncated_gaussian",
            "parabolic_on_pedestal",
        ):
            raise ValueError(f"unknown illumination '{self.illumination}'")

    # -- geometry and normalisation -------------------------------------------------------------

    @property
    def radius_m(self) -> float:
        """Aperture radius ``a = D/2`` [m]."""
        return 0.5 * self.diameter_m

    @property
    def area_m2(self) -> float:
        """Geometric aperture area ``pi a^2`` [m^2]."""
        return math.pi * self.radius_m**2

    @property
    def pedestal(self) -> float:
        """Edge field ``C = 10^(-T/20)`` of ``parabolic_on_pedestal`` (1 for uniform) [1]."""
        if self.illumination == "parabolic_on_pedestal":
            assert self.edge_taper_dB is not None
            return float(10.0 ** (-self.edge_taper_dB / 20.0))
        return 1.0

    @property
    def _x(self) -> float:
        """``a^2 / w0^2`` of the truncated Gaussian."""
        assert self.gaussian_waist_radius_m is not None
        return (self.radius_m / self.gaussian_waist_radius_m) ** 2

    @property
    def normalisation_m_inv(self) -> float:
        """Amplitude constant ``c`` [m^-1] such that ``int |A|^2 dA = 1`` (closed forms above)."""
        a2 = self.radius_m**2
        if self.illumination == "uniform_circular":
            return 1.0 / math.sqrt(math.pi * a2)
        if self.illumination == "truncated_gaussian":
            assert self.gaussian_waist_radius_m is not None
            w2 = self.gaussian_waist_radius_m**2
            return 1.0 / math.sqrt(0.5 * math.pi * w2 * -math.expm1(-2.0 * self._x))
        c_edge = self.pedestal
        return 1.0 / math.sqrt(math.pi * a2 * (c_edge + (1.0 - c_edge) ** 2 / 3.0))

    def amplitude_m_inv(self, r_m: ArrayLike) -> Float64Array:
        """Field amplitude ``A(r)`` [m^-1] (``|A|^2`` in W m^-2 per W); zero for ``r > a``."""
        r = np.abs(np.asarray(r_m, dtype=np.float64))
        a = self.radius_m
        c = self.normalisation_m_inv
        if self.illumination == "uniform_circular":
            prof = np.ones_like(r)
        elif self.illumination == "truncated_gaussian":
            assert self.gaussian_waist_radius_m is not None
            prof = np.exp(-((r / self.gaussian_waist_radius_m) ** 2))
        else:
            c_edge = self.pedestal
            prof = c_edge + (1.0 - c_edge) * (1.0 - (r / a) ** 2)
        out: Float64Array = np.where(r <= a, c * prof, 0.0)
        return out

    @property
    def amplitude_integral_m(self) -> float:
        """``int A dA = 2 pi int_0^a A r dr`` [m] (sets the on-axis far field)."""
        c = self.normalisation_m_inv
        if self.illumination == "uniform_circular":
            return c * self.area_m2
        if self.illumination == "truncated_gaussian":
            assert self.gaussian_waist_radius_m is not None
            w2 = self.gaussian_waist_radius_m**2
            return c * math.pi * w2 * -math.expm1(-self._x)
        return c * self.area_m2 * 0.5 * (1.0 + self.pedestal)

    @property
    def taper_efficiency(self) -> float:
        """Aperture illumination (taper) efficiency ``|int A dA|^2 / (pi a^2 int |A|^2 dA)``.

        Ratio of the on-axis far-field intensity to that of a uniformly illuminated aperture of
        the same size and power (Balanis §15.4). It is a *diagnostic*: the propagated patterns
        already contain it, so it is never applied as a separate factor (PDF §10.1).
        """
        return self.amplitude_integral_m**2 / self.area_m2

    @property
    def source_spillover_fraction(self) -> float:
        """Fraction of an untruncated Gaussian source beam that misses the aperture.

        ``exp(-2 a^2 / w0^2)`` for ``truncated_gaussian``, 0 otherwise. Diagnostic only: this
        power is never emitted and its loss belongs to ``transmitter.optics_or_rf_efficiency``
        (module docstring, "spill-over owner").
        """
        if self.illumination == "truncated_gaussian":
            return math.exp(-2.0 * self._x)
        return 0.0

    @property
    def edge_intensity_ratio(self) -> float:
        """``|A(a)|^2 / |A(0)|^2`` [1] (edge taper as a ratio)."""
        if self.illumination == "truncated_gaussian":
            return math.exp(-2.0 * self._x)
        return self.pedestal**2

    @property
    def has_closed_form_far_field(self) -> bool:
        """True for uniform (Airy) and parabolic-on-pedestal (Lommel) illumination."""
        return self.illumination != "truncated_gaussian"

    # -- aperture quadrature ---------------------------------------------------------------------

    def _aperture_nodes(
        self, cycles_per_m: float, refinement: float
    ) -> tuple[Float64Array, Float64Array]:
        """Composite Gauss-Legendre nodes on ``[0, a]`` resolving ``cycles_per_m`` and the waist."""
        a = self.radius_m
        n = a * cycles_per_m
        if self.illumination == "truncated_gaussian":
            assert self.gaussian_waist_radius_m is not None
            n += a / self.gaussian_waist_radius_m
        panels = math.ceil(refinement * n) + 2
        if panels * GL_ORDER_HANKEL > _MAX_HANKEL_NODES:
            raise ValueError(
                f"aperture quadrature would need {panels * GL_ORDER_HANKEL} nodes "
                f"(> {_MAX_HANKEL_NODES}); the requested field is too oscillatory"
            )
        edges = np.linspace(0.0, a, panels + 1)
        return composite_gauss_legendre(edges, GL_ORDER_HANKEL)

    def hankel_amplitude(
        self,
        rho_m: ArrayLike,
        wavelength_m: float,
        range_m: float | None,
        *,
        refinement: float = 1.0,
    ) -> Complex128Array:
        """Scalar field ``U(rho)`` by Gauss-Legendre quadrature (module docstring, item 2).

        Args:
            rho_m: Transverse radii [m] at distance ``range_m``; or, with ``range_m=None``,
                far-field angles ``theta`` [rad].
            wavelength_m: Wavelength [m].
            range_m: Propagation distance ``z`` [m] (Fresnel, with the aperture chirp), or
                ``None`` for the Fraunhofer angular spectrum (no chirp).
            refinement: Multiplies the number of quadrature panels (convergence studies).

        Returns:
            Fresnel: ``U(rho; z)`` [m^-1] with ``|U|^2`` the irradiance per watt [m^-2]
            (including the output phase ``exp(i pi rho^2/(lambda z))``).
            Fraunhofer: the angular amplitude ``(2 pi / lambda) int A J0(2 pi r theta/lambda) r dr``
            [sr^-1/2], ``|.|^2`` = radiant intensity per watt [sr^-1].
        """
        q = np.asarray(rho_m, dtype=np.float64)
        flat = np.abs(q.ravel())
        if range_m is None:
            scale = wavelength_m  # J0 argument 2 pi r theta / lambda
            chirp_cycles_per_m = 0.0
        else:
            scale = wavelength_m * range_m
            chirp_cycles_per_m = self.radius_m / scale
        q_max = float(np.max(flat)) if flat.size else 0.0
        cycles_per_m = chirp_cycles_per_m + q_max / scale
        r, w = self._aperture_nodes(cycles_per_m, refinement)
        if flat.size * r.size > _MAX_HANKEL_WORK:
            raise ValueError(
                f"Hankel evaluation of {flat.size} points x {r.size} aperture nodes exceeds the "
                f"work budget {_MAX_HANKEL_WORK}: the receiver region is far larger than the "
                "beam feature scale (use a far-field or closed-form model, or a smaller region)"
            )
        g = self.amplitude_m_inv(r) * r * w
        if range_m is not None:
            g = g * np.exp(1j * math.pi * r**2 / scale)
        out = np.empty(flat.shape, dtype=np.complex128)
        chunk = max(1, _MATRIX_CHUNK // r.size)
        for i in range(0, flat.size, chunk):
            arg = (2.0 * math.pi / scale) * np.outer(flat[i : i + chunk], r)
            out[i : i + chunk] = special.j0(arg) @ g
        out *= 2.0 * math.pi / scale
        if range_m is not None:
            out *= np.exp(1j * math.pi * flat**2 / scale)
        return out.reshape(q.shape)

    def radiant_intensity_sr_inv(
        self, theta_rad: ArrayLike, wavelength_m: float, *, refinement: float = 1.0
    ) -> Float64Array:
        """Fraunhofer radiant intensity per watt ``J(theta)`` [sr^-1] (paraxial).

        Closed forms for ``uniform_circular`` (Airy) and ``parabolic_on_pedestal`` (Lommel);
        Gauss-Legendre Hankel quadrature for ``truncated_gaussian`` (module docstring).
        """
        th = np.abs(np.asarray(theta_rad, dtype=np.float64))
        lam2 = wavelength_m**2
        if self.illumination == "truncated_gaussian":
            amp = self.hankel_amplitude(th, wavelength_m, None, refinement=refinement)
            out: Float64Array = np.abs(amp) ** 2
            return out
        u = (2.0 * math.pi * self.radius_m / wavelength_m) * th
        small = u < _SMALL_U
        us = np.where(small, 1.0, u)
        j1_over_u = np.where(small, 0.5 - u**2 / 16.0, special.j1(us) / us)
        if self.illumination == "uniform_circular":
            out = (self.area_m2 / lam2) * (2.0 * j1_over_u) ** 2
            return out
        c_edge = self.pedestal
        j2_over_u2 = np.where(small, 0.125 - u**2 / 96.0, special.jv(2, us) / us**2)
        amp_m = (
            self.normalisation_m_inv
            * 2.0
            * self.area_m2
            * (c_edge * j1_over_u + 2.0 * (1.0 - c_edge) * j2_over_u2)
        )
        out = amp_m**2 / lam2
        return out

    @property
    def on_axis_radiant_intensity_factor(self) -> float:
        """``J(0) lambda^2 = |int A dA|^2 = taper_efficiency * pi a^2`` [m^2]."""
        return self.amplitude_integral_m**2


def aperture_from_config(transmitter: TransmitterConfig) -> ApertureField:
    """:class:`ApertureField` from the ``transmitter`` configuration section."""
    return ApertureField(
        illumination=transmitter.aperture_illumination,
        diameter_m=transmitter.aperture_diameter_m,
        gaussian_waist_radius_m=transmitter.gaussian_waist_radius_m,
        edge_taper_dB=transmitter.edge_taper_dB,
    )


# ---------------------------------------------------------------------------------------------
# Radial beam patterns (shared interface of every beam model)
# ---------------------------------------------------------------------------------------------


@runtime_checkable
class RadialBeamPattern(Protocol):
    """Circularly symmetric irradiance pattern in the plane normal to the beam at one range.

    ``intensity_m2_inv(rho)`` is the irradiance per watt of beam power [W m^-2 per W = m^-2] at
    transverse radius ``rho`` [m]; it integrates to 1 over the plane. ``encircled_energy(rho)``
    is the fraction of the power inside radius ``rho``. ``feature_scale_m`` is the smallest
    transverse length on which the pattern varies (quadrature panels are a fraction of it).
    """

    @property
    def model(self) -> str: ...

    @property
    def range_m(self) -> float: ...

    @property
    def feature_scale_m(self) -> float: ...

    def intensity_m2_inv(self, rho_m: ArrayLike) -> Float64Array: ...

    def encircled_energy(self, rho_m: ArrayLike) -> Float64Array: ...

    def peak_intensity_m2_inv(self) -> float: ...


def numerical_encircled_energy(
    pattern: RadialBeamPattern, rho_m: ArrayLike, *, refinement: float = 1.0
) -> Float64Array:
    """``EE(rho) = int_0^rho I(rho') 2 pi rho' drho'`` by composite Gauss-Legendre quadrature.

    One common panel grid (width ``<= feature_scale / 2 / refinement``) with every requested
    radius inserted as a panel boundary; cumulative sums give all radii at once.
    """
    rho = np.asarray(rho_m, dtype=np.float64)
    flat = rho.ravel()
    if np.any(~np.isfinite(flat)) or np.any(flat < 0.0):
        raise ValueError("rho_m must be finite and >= 0")
    if flat.size == 0:
        return np.zeros(rho.shape)
    r_max = float(np.max(flat))
    if r_max == 0.0:
        return np.zeros(rho.shape)
    h = 0.5 * pattern.feature_scale_m / refinement
    n_uniform = math.ceil(r_max / h)
    if n_uniform > 5_000_000:
        raise ValueError(
            f"encircled energy to rho={r_max:.6g} m needs {n_uniform} panels of "
            f"{h:.6g} m; the receiver is too large relative to the beam feature scale"
        )
    edges = np.unique(np.concatenate((np.linspace(0.0, r_max, n_uniform + 1), flat)))
    nodes, weights = composite_gauss_legendre(edges, GL_ORDER_RADIAL)
    vals = pattern.intensity_m2_inv(nodes) * 2.0 * math.pi * nodes * weights
    per_panel = vals.reshape(-1, GL_ORDER_RADIAL).sum(axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(per_panel)))
    idx = np.searchsorted(edges, flat)
    out: Float64Array = cumulative[idx].reshape(rho.shape)
    return out


@dataclass(frozen=True)
class FarFieldAperturePattern:
    """Fraunhofer pattern of a circular aperture field at range ``R`` (any illumination).

    ``I(rho) = J(rho / R) / R^2`` with ``J`` from :meth:`ApertureField.radiant_intensity_sr_inv`.
    Construction raises :class:`BeamModelValidityError` if ``N_F > max_fresnel_number``.

    Attributes:
        aperture: Aperture field.
        wavelength_m: Wavelength [m].
        range_m: Range ``R`` [m].
        max_fresnel_number: Far-field validity limit (``models.fraunhofer_max_fresnel_number``).
        model: Model label recorded in results.
        refinement: Quadrature refinement: multiplies the panel counts of the encircled-energy
            quadrature and (``truncated_gaussian``) of the Hankel integral.
    """

    aperture: ApertureField
    wavelength_m: float
    range_m: float
    max_fresnel_number: float
    model: str = "aperture_far_field"
    refinement: float = 1.0

    def __post_init__(self) -> None:
        nf = float(fresnel_number(self.aperture.diameter_m, self.wavelength_m, self.range_m))
        check_far_field_validity(self.model, nf, self.max_fresnel_number)

    @property
    def fresnel_number(self) -> float:
        """``D^2 / (lambda R)`` [1]."""
        return float(fresnel_number(self.aperture.diameter_m, self.wavelength_m, self.range_m))

    @property
    def feature_scale_m(self) -> float:
        """``lambda R / D`` [m] (sidelobe period of the far field)."""
        return self.wavelength_m * self.range_m / self.aperture.diameter_m

    def intensity_m2_inv(self, rho_m: ArrayLike) -> Float64Array:
        """Irradiance per watt [m^-2] at transverse radius ``rho`` [m]."""
        rho = np.asarray(rho_m, dtype=np.float64)
        j = self.aperture.radiant_intensity_sr_inv(
            rho / self.range_m, self.wavelength_m, refinement=self.refinement
        )
        out: Float64Array = j / self.range_m**2
        return out

    def encircled_energy(self, rho_m: ArrayLike) -> Float64Array:
        """Fraction of the beam power within radius ``rho`` (numerical quadrature)."""
        return numerical_encircled_energy(self, rho_m, refinement=self.refinement)

    def peak_intensity_m2_inv(self) -> float:
        """On-axis irradiance per watt ``|int A dA|^2 / (lambda R)^2`` [m^-2].

        The far-field pattern of a real, non-negative, radially non-increasing aperture field
        peaks on axis (``|int A J0 dA| <= int |A| dA``).
        """
        return (
            self.aperture.on_axis_radiant_intensity_factor / (self.wavelength_m * self.range_m) ** 2
        )


@dataclass(frozen=True)
class FresnelHankelPattern:
    """Fresnel (paraxial, finite-distance) pattern of a circular aperture field at range ``z``.

    ``I(rho) = |U(rho; z)|^2`` with ``U`` the Fresnel-Hankel integral (module docstring). Valid
    at any Fresnel number provided Goodman's paraxial condition holds; construction raises
    :class:`BeamModelValidityError` if :func:`fresnel_paraxial_parameter` at
    ``rho_max = paraxial_rho_max_m`` exceeds :data:`FRESNEL_PARAXIAL_MAX_PHASE_RAD`, and every
    field evaluation re-checks it at the largest requested radius (so the peak search and any
    other caller cannot evaluate outside the paraxial region).

    Attributes:
        aperture: Aperture field.
        wavelength_m: Wavelength [m].
        range_m: Range ``z`` [m].
        paraxial_rho_max_m: Largest transverse radius at which the pattern will be used [m].
        refinement: Hankel-quadrature refinement factor.
        model: Model label.
    """

    aperture: ApertureField
    wavelength_m: float
    range_m: float
    paraxial_rho_max_m: float
    refinement: float = 1.0
    model: str = "fresnel_hankel"

    def __post_init__(self) -> None:
        self._check_paraxial(self.paraxial_rho_max_m)

    def _check_paraxial(self, rho_max_m: float) -> None:
        p = fresnel_paraxial_parameter(
            self.aperture.radius_m, rho_max_m, self.wavelength_m, self.range_m
        )
        if not p <= FRESNEL_PARAXIAL_MAX_PHASE_RAD:
            raise BeamModelValidityError(
                f"Fresnel (paraxial) approximation not justified: neglected quartic phase "
                f"pi (a + rho)^4 / (4 lambda z^3) = {p:.6g} rad > "
                f"{FRESNEL_PARAXIAL_MAX_PHASE_RAD} rad at rho = {rho_max_m:.6g} m, "
                f"z = {self.range_m:.6g} m (Goodman eq. 4-18); a non-paraxial "
                "(Rayleigh-Sommerfeld) propagator would be needed"
            )

    @property
    def fresnel_number(self) -> float:
        """``D^2 / (lambda z)`` [1]."""
        return float(fresnel_number(self.aperture.diameter_m, self.wavelength_m, self.range_m))

    @property
    def feature_scale_m(self) -> float:
        """``lambda z / D`` [m]: period of the finest (edge-to-edge) interference fringes."""
        return self.wavelength_m * self.range_m / self.aperture.diameter_m

    def field(self, rho_m: ArrayLike) -> Complex128Array:
        """Complex field ``U(rho; z)`` [m^-1] (paraxial validity re-checked at ``max(rho)``)."""
        rho = np.asarray(rho_m, dtype=np.float64)
        if rho.size and np.all(np.isfinite(rho)):
            r_max = float(np.max(np.abs(rho)))
            if r_max > self.paraxial_rho_max_m:
                self._check_paraxial(r_max)
        return self.aperture.hankel_amplitude(
            rho_m, self.wavelength_m, self.range_m, refinement=self.refinement
        )

    def intensity_m2_inv(self, rho_m: ArrayLike) -> Float64Array:
        """Irradiance per watt [m^-2] at transverse radius ``rho`` [m]."""
        out: Float64Array = np.abs(self.field(rho_m)) ** 2
        return out

    def encircled_energy(self, rho_m: ArrayLike) -> Float64Array:
        """Fraction of the beam power within radius ``rho`` (numerical quadrature)."""
        return numerical_encircled_energy(self, rho_m, refinement=self.refinement)

    def peak_intensity_m2_inv(self) -> float:
        """Maximum irradiance per watt over the transverse plane [m^-2].

        The near-field pattern can peak off axis (Fresnel-zone structure), so the maximum is
        taken over a grid of spacing ``feature_scale / 16`` covering ``rho <= a + 4 lambda z / D``
        (beyond which the pattern is a decaying diffraction tail), followed by a parabolic
        refinement of the largest sample. The result is accurate to O(grid spacing^2). The
        paraxial check is re-applied at ``a + 4 lambda z / D`` (:meth:`field`).
        """
        ell = self.feature_scale_m
        rho_max = self.aperture.radius_m + 4.0 * ell
        n = int(min(200_000, math.ceil(16.0 * rho_max / ell))) + 1
        rho = np.linspace(0.0, rho_max, n)
        vals = self.intensity_m2_inv(rho)
        k = int(np.argmax(vals))
        best = float(vals[k])
        if 0 < k < n - 1:
            y0, y1, y2 = float(vals[k - 1]), best, float(vals[k + 1])
            denom = y0 - 2.0 * y1 + y2
            if denom < 0.0:
                shift = 0.5 * (y0 - y2) / denom
                rho_star = rho[k] + shift * (rho[1] - rho[0])
                best = max(best, float(self.intensity_m2_inv(np.array([rho_star]))[0]))
        return best


__all__ = [
    "FRESNEL_PARAXIAL_MAX_PHASE_RAD",
    "GL_ORDER_HANKEL",
    "GL_ORDER_RADIAL",
    "ApertureField",
    "BeamModelValidityError",
    "FarFieldAperturePattern",
    "FresnelHankelPattern",
    "RadialBeamPattern",
    "aperture_from_config",
    "check_far_field_validity",
    "composite_gauss_legendre",
    "fresnel_number",
    "fresnel_paraxial_parameter",
    "numerical_encircled_energy",
    "ruze_efficiency",
    "wavefront_efficiency",
]
