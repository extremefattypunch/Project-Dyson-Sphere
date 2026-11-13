"""Optical beam models: Airy far field and the separate Gaussian-beam model (PDF §9.1-9.2).

Two closed-form optical models, each a :class:`helioswarm.aperture.RadialBeamPattern`; the
general Fresnel-Hankel propagator for arbitrary illumination lives in ``aperture.py``
(:class:`~helioswarm.aperture.FresnelHankelPattern`) and is shared with the microwave model.

1. **Airy far field** (``models.beam = airy_far_field``): uniformly illuminated circular
   aperture of diameter ``D`` in the Fraunhofer regime (Born & Wolf, *Principles of Optics*,
   7th ed., §8.5.2). With ``x = pi D rho / (lambda R)`` and the pattern normalised to one watt::

       I(rho) = I0 [2 J1(x) / x]^2,       I0 = pi D^2 / (4 lambda^2 R^2)        [m^-2]
       EE(rho) = 1 - J0(x)^2 - J1(x)^2                                   (Rayleigh 1881)

   The first dark ring is at ``x = j_{1,1} = 3.8317059702...`` (first zero of J1), i.e.
   ``rho_null = (j_{1,1}/pi) lambda R / D = 1.21967 lambda R / D``, and encloses
   ``EE = 1 - J0(j_{1,1})^2 = 0.83777`` of the power. The pattern is only valid for
   ``N_F = D^2/(lambda R) <= models.fraunhofer_max_fresnel_number``; construction raises
   :class:`~helioswarm.aperture.BeamModelValidityError` otherwise.

2. **Gaussian beam** (``models.beam = gaussian_beam``): an *untruncated* TEM00 beam with its
   waist ``w0`` (1/e^2 intensity radius) in the aperture plane (collimated output; no focusing
   or defocus), Siegman, *Lasers* (1986), ch. 17::

       z_R = pi w0^2 / lambda,   w(z) = w0 sqrt(1 + (z / z_R)^2)
       I(rho) = (2 / (pi w^2)) exp(-2 rho^2 / w^2),     EE(rho) = 1 - exp(-2 rho^2 / w^2)

   It is a **separate model, not an interchangeable label** (PDF §9.2): it approximates the
   ``truncated_gaussian`` aperture field only when the truncation is weak, which the schema
   enforces (``a / w0 >= models.gaussian_beam_min_truncation_ratio``, default 2.5; the far-field
   on-axis error is then 0.4 %, see ``docs/model_assumptions.md``). It is valid at any range
   (no Fresnel-number limit) because a Gaussian beam is an exact solution of the paraxial wave
   equation. The comparison path is ``fresnel_hankel`` with the same ``truncated_gaussian``
   illumination (tested).

All quantities are scalar diffraction (``aperture.py`` module docstring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

from helioswarm.aperture import check_far_field_validity, fresnel_number

Float64Array = NDArray[np.float64]

J1_FIRST_ZERO: Final[float] = 3.8317059702075125
"""First positive zero ``j_{1,1}`` of the Bessel function J1 (Abramowitz & Stegun Table 9.5;
equal to ``scipy.special.jn_zeros(1, 1)[0]``, checked in the tests)."""

AIRY_FIRST_NULL_FACTOR: Final[float] = J1_FIRST_ZERO / math.pi
"""``j_{1,1} / pi = 1.21967...``: first-null radius in units of ``lambda R / D``."""

_SMALL_X: Final[float] = 1.0e-4


def airy_first_null_radius_m(
    wavelength_m: float, range_m: ArrayLike, aperture_diameter_m: float
) -> Float64Array:
    """Radius of the first dark ring ``1.21967 lambda R / D`` [m] (PDF §9.1)."""
    out: Float64Array = (
        AIRY_FIRST_NULL_FACTOR
        * wavelength_m
        * np.asarray(range_m, dtype=np.float64)
        / aperture_diameter_m
    )
    return out


def airy_encircled_energy_x(x: ArrayLike) -> Float64Array:
    """Rayleigh's encircled energy ``1 - J0(x)^2 - J1(x)^2`` of the Airy pattern.

    ``x = pi D rho / (lambda R)``. For ``x < 1e-4`` the series ``x^2/4 - x^4/32`` is used
    (the closed form loses digits to cancellation there).
    """
    xa = np.abs(np.asarray(x, dtype=np.float64))
    small = xa < _SMALL_X
    direct = 1.0 - special.j0(xa) ** 2 - special.j1(xa) ** 2
    out: Float64Array = np.where(small, xa**2 / 4.0 - xa**4 / 32.0, direct)
    return out


def airy_profile_x(x: ArrayLike) -> Float64Array:
    """Normalised Airy profile ``[2 J1(x)/x]^2`` (1 on axis)."""
    xa = np.abs(np.asarray(x, dtype=np.float64))
    small = xa < _SMALL_X
    xs = np.where(small, 1.0, xa)
    two_j1_over_x = np.where(small, 1.0 - xa**2 / 8.0, 2.0 * special.j1(xs) / xs)
    out: Float64Array = two_j1_over_x**2
    return out


@dataclass(frozen=True)
class AiryPattern:
    """Fraunhofer pattern of a uniformly illuminated circular aperture at range ``R``.

    Construction raises :class:`~helioswarm.aperture.BeamModelValidityError` when
    ``N_F = D^2/(lambda R) > max_fresnel_number``.

    Attributes:
        aperture_diameter_m: ``D`` [m].
        wavelength_m: ``lambda`` [m].
        range_m: ``R`` [m].
        max_fresnel_number: Far-field validity limit.
    """

    aperture_diameter_m: float
    wavelength_m: float
    range_m: float
    max_fresnel_number: float
    model: str = "airy_far_field"

    def __post_init__(self) -> None:
        for name in ("aperture_diameter_m", "wavelength_m", "range_m", "max_fresnel_number"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"AiryPattern.{name} must be finite and > 0 (got {v})")
        check_far_field_validity(self.model, self.fresnel_number, self.max_fresnel_number)

    @property
    def fresnel_number(self) -> float:
        """``D^2 / (lambda R)`` [1]."""
        return float(fresnel_number(self.aperture_diameter_m, self.wavelength_m, self.range_m))

    @property
    def feature_scale_m(self) -> float:
        """``lambda R / D`` [m] (ring period ~ pi in ``x``)."""
        return self.wavelength_m * self.range_m / self.aperture_diameter_m

    @property
    def first_null_radius_m(self) -> float:
        """``1.21967 lambda R / D`` [m]."""
        return float(
            airy_first_null_radius_m(self.wavelength_m, self.range_m, self.aperture_diameter_m)
        )

    def x_of_rho(self, rho_m: ArrayLike) -> Float64Array:
        """Dimensionless radius ``x = pi D rho / (lambda R)``."""
        out: Float64Array = (
            math.pi * self.aperture_diameter_m / (self.wavelength_m * self.range_m)
        ) * np.asarray(rho_m, dtype=np.float64)
        return out

    def peak_intensity_m2_inv(self) -> float:
        """On-axis irradiance per watt ``pi D^2 / (4 lambda^2 R^2)`` [m^-2]."""
        return (
            math.pi * self.aperture_diameter_m**2 / (4.0 * (self.wavelength_m * self.range_m) ** 2)
        )

    def intensity_m2_inv(self, rho_m: ArrayLike) -> Float64Array:
        """Irradiance per watt ``I0 [2 J1(x)/x]^2`` [m^-2]."""
        out: Float64Array = self.peak_intensity_m2_inv() * airy_profile_x(self.x_of_rho(rho_m))
        return out

    def encircled_energy(self, rho_m: ArrayLike) -> Float64Array:
        """``1 - J0(x)^2 - J1(x)^2`` (closed form)."""
        return airy_encircled_energy_x(self.x_of_rho(rho_m))


def gaussian_rayleigh_range_m(waist_radius_m: float, wavelength_m: float) -> float:
    """Rayleigh range ``z_R = pi w0^2 / lambda`` [m] (Siegman 1986, eq. 17.11)."""
    return math.pi * waist_radius_m**2 / wavelength_m


def gaussian_beam_radius_m(
    waist_radius_m: float, wavelength_m: float, range_m: ArrayLike
) -> Float64Array:
    """1/e^2 intensity radius ``w(z) = w0 sqrt(1 + (z/z_R)^2)`` [m] (Siegman 1986, eq. 17.12)."""
    z_r = gaussian_rayleigh_range_m(waist_radius_m, wavelength_m)
    out: Float64Array = waist_radius_m * np.sqrt(
        1.0 + (np.asarray(range_m, dtype=np.float64) / z_r) ** 2
    )
    return out


@dataclass(frozen=True)
class GaussianBeamPattern:
    """Untruncated TEM00 Gaussian beam (waist ``w0`` at the aperture) at range ``z``.

    Attributes:
        waist_radius_m: ``w0`` [m].
        wavelength_m: ``lambda`` [m].
        range_m: ``z`` [m].
    """

    waist_radius_m: float
    wavelength_m: float
    range_m: float
    model: str = "gaussian_beam"

    def __post_init__(self) -> None:
        for name in ("waist_radius_m", "wavelength_m", "range_m"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"GaussianBeamPattern.{name} must be finite and > 0 (got {v})")

    @property
    def beam_radius_m(self) -> float:
        """``w(z)`` [m]."""
        return float(gaussian_beam_radius_m(self.waist_radius_m, self.wavelength_m, self.range_m))

    @property
    def feature_scale_m(self) -> float:
        """``w(z) / 2`` [m]."""
        return 0.5 * self.beam_radius_m

    def peak_intensity_m2_inv(self) -> float:
        """On-axis irradiance per watt ``2 / (pi w^2)`` [m^-2]."""
        return 2.0 / (math.pi * self.beam_radius_m**2)

    def intensity_m2_inv(self, rho_m: ArrayLike) -> Float64Array:
        """Irradiance per watt ``(2/(pi w^2)) exp(-2 rho^2/w^2)`` [m^-2]."""
        w = self.beam_radius_m
        rho = np.asarray(rho_m, dtype=np.float64)
        out: Float64Array = self.peak_intensity_m2_inv() * np.exp(-2.0 * (rho / w) ** 2)
        return out

    def encircled_energy(self, rho_m: ArrayLike) -> Float64Array:
        """``1 - exp(-2 rho^2 / w^2)`` (closed form)."""
        w = self.beam_radius_m
        rho = np.asarray(rho_m, dtype=np.float64)
        out: Float64Array = -np.expm1(-2.0 * (rho / w) ** 2)
        return out


__all__ = [
    "AIRY_FIRST_NULL_FACTOR",
    "J1_FIRST_ZERO",
    "AiryPattern",
    "GaussianBeamPattern",
    "airy_encircled_energy_x",
    "airy_first_null_radius_m",
    "airy_profile_x",
    "gaussian_beam_radius_m",
    "gaussian_rayleigh_range_m",
]
