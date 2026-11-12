"""Microwave geometry: filled tapered aperture and sparse phased-array lattice (PDF §9.3).

Two distinct geometries (PDF §9.3, "treat a continuous filled aperture and a sparse phased
array as different geometries"):

1. **Continuous filled aperture** with a radial illumination taper. Its far field
   (``models.beam = aperture_far_field``) and Fresnel field (``fresnel_hankel``) come from the
   *same* aperture/Hankel machinery as the optical models (``aperture.py``:
   :class:`~helioswarm.aperture.FarFieldAperturePattern`,
   :class:`~helioswarm.aperture.FresnelHankelPattern`); the taper (``uniform_circular``,
   ``truncated_gaussian``, ``parabolic_on_pedestal``) is the stated aperture-illumination
   assumption, and the far field of the tapered aperture is its Hankel transform (closed form
   for the parabolic-on-pedestal taper via the Lommel integral, quadrature otherwise).

2. **Uniform square lattice of identical elements** (``transmitter.array_element_spacing_m``
   set) filling the same circular aperture, phased for broadside emission towards the
   receiver, with element *power* pattern ``cos^q(theta)``
   (``transmitter.array_element_pattern_exponent``). Array factor of an ``n x n`` lattice
   (Balanis, *Antenna Theory*, 4th ed., §6.10, eq. 6-89; centred, so it is real)::

       AF(u, v) = [sin(n psi_x / 2) / sin(psi_x / 2)] [sin(n psi_y / 2) / sin(psi_y / 2)],
       psi_x = 2 pi d u / lambda,  psi_y = 2 pi d v / lambda,  u = sin th cos ph, v = sin th sin ph

   ``|AF|`` peaks at ``N = n^2`` on axis and again at every **grating lobe**
   ``(u, v) = (m lambda/d, n lambda/d)``; lobes with ``u^2 + v^2 < 1`` are visible, so a
   broadside lattice has none when ``d < lambda``.

   *Main-lobe (grating-lobe) efficiency.* For a large array each lobe is a replica of the main
   lobe in ``(u, v)``; ``|AF|^2`` integrates to ``N (lambda/d)^2`` over each ``(u, v)`` period
   cell (Parseval), and ``dOmega = du dv / cos(theta)``. The power radiated into lobe ``(m, n)``
   is therefore proportional to ``cos^q(theta_mn) / cos(theta_mn)`` and the fraction of the
   radiated power in the main lobe is::

       eta_lattice = 1 / sum_{visible (m,n)} cos^{q-1}(theta_mn)

   (``= 1`` without visible grating lobes; ``1 / N_visible`` for the ideal unit-cell element
   ``q = 1``). Validity: many elements (the main lobe of the sampled aperture then equals the
   continuous-aperture pattern with the same taper), ``q >= 1``, no lobe at grazing incidence,
   and the receiver plus pointing offsets well inside the main lobe's period cell (angular
   separation ``lambda/d`` to the nearest grating lobe). :func:`lattice_main_lobe_fraction` is
   checked against direct numerical integration of ``cos^q |AF|^2`` over the hemisphere.

   The lattice beam model is therefore ``eta_lattice`` times the continuous-aperture pattern;
   ``eta_lattice`` belongs to the capture fraction (power radiated into grating lobes never
   reaches the receiver) and is recorded separately in ``BeamResult``.

**Phase errors.** Random aperture phase errors reduce the main beam by the Ruze factor
``exp(-sigma_phi^2)``, ``sigma_phi = 2 pi sigma_w / lambda``; its single owner is
:func:`helioswarm.aperture.wavefront_efficiency` (applied once by ``beam.evaluate_beam``), so it
is not re-implemented here.

**No coherent combining across satellites.** A constellation of independent RF transmitters is
*not* one coherent array (PDF §9.3, §17): every beam model here describes one transmitter, and
received powers from different satellites may only be added incoherently. ``BeamResult``
asserts ``coherent_combining is False``; coherent synthesis would need phase, frequency, timing
and relative-position control and is out of scope (a later, separate module).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.aperture import BeamModelValidityError

Float64Array = NDArray[np.float64]

GRAZING_TOLERANCE: Final[float] = 1.0e-6
"""Grating lobes with ``|u^2 + v^2 - 1|`` below this are 'grazing': the large-array lobe model
fails there and :func:`lattice_main_lobe_fraction` raises."""

MIN_LATTICE_ELEMENTS: Final[int] = 100
"""Fewest elements for which the lattice main lobe is treated as the continuous-aperture lobe."""

GRATING_LOBE_CLEARANCE: Final[float] = 0.25
"""Receiver angular radius plus pointing offsets must stay below this fraction of the
grating-lobe separation ``lambda / d`` (else the receiver would sample the lobe structure)."""


def dirichlet_kernel(psi: ArrayLike, n: int) -> Float64Array:
    """Centred 1-D array factor ``sin(n psi/2) / sin(psi/2)`` of ``n`` unit elements (real).

    At ``psi = 2 pi k`` its limit ``n (-1)^{k (n-1)}`` is returned.
    """
    p = np.asarray(psi, dtype=np.float64)
    half = 0.5 * p
    s = np.sin(half)
    k = np.round(p / (2.0 * math.pi))
    near = np.abs(p - 2.0 * math.pi * k) < 1.0e-9
    limit = n * np.where((k * (n - 1)) % 2 == 0, 1.0, -1.0)
    safe = np.where(near, 1.0, s)
    out: Float64Array = np.where(near, limit, np.sin(n * half) / safe)
    return out


def square_lattice_array_factor(
    u: ArrayLike, v: ArrayLike, n_side: int, spacing_m: float, wavelength_m: float
) -> Float64Array:
    """Array factor of an ``n_side x n_side`` broadside square lattice (Balanis eq. 6-89).

    Args:
        u, v: Direction cosines ``sin th cos ph``, ``sin th sin ph``.
        n_side: Elements per side (``N = n_side^2``).
        spacing_m: Element spacing ``d`` [m].
        wavelength_m: Wavelength [m].

    Returns:
        Real ``AF(u, v)``; ``|AF| <= N`` with equality on axis and at grating lobes.
    """
    k_d = 2.0 * math.pi * spacing_m / wavelength_m
    return dirichlet_kernel(k_d * np.asarray(u, dtype=np.float64), n_side) * dirichlet_kernel(
        k_d * np.asarray(v, dtype=np.float64), n_side
    )


def visible_grating_lobes(spacing_m: float, wavelength_m: float) -> Float64Array:
    """Direction cosines ``(u, v) = (m, n) lambda/d`` of all visible lobes (incl. the main lobe).

    Returns an array of shape (K, 2) with ``u^2 + v^2 < 1``; the main lobe ``(0, 0)`` is first.
    """
    step = wavelength_m / spacing_m
    m_max = math.floor(1.0 / step)
    idx = np.arange(-m_max, m_max + 1)
    mm, nn = np.meshgrid(idx, idx, indexing="ij")
    uv = np.stack([mm.ravel() * step, nn.ravel() * step], axis=1)
    r2 = np.sum(uv**2, axis=1)
    keep = r2 < 1.0
    out = uv[keep]
    order = np.argsort(np.sum(out**2, axis=1), kind="stable")
    result: Float64Array = out[order]
    return result


def element_power_pattern(theta_rad: ArrayLike, exponent_q: float) -> Float64Array:
    """Element power pattern ``cos^q(theta)`` for ``theta < pi/2``, 0 behind the array."""
    th = np.asarray(theta_rad, dtype=np.float64)
    c = np.cos(th)
    out: Float64Array = np.where(c > 0.0, np.abs(c) ** exponent_q, 0.0)
    return out


def lattice_main_lobe_fraction(spacing_m: float, wavelength_m: float, exponent_q: float) -> float:
    """Fraction of the radiated power in the main lobe of a large broadside square lattice.

    ``1 / sum_{visible lobes} cos^{q-1}(theta_mn)`` (module docstring). 1 when ``d < lambda``.

    Raises:
        ValueError: ``q < 1`` or non-positive inputs.
        BeamModelValidityError: a grating lobe at grazing incidence (``|u^2+v^2-1| < 1e-6``).
    """
    if not (spacing_m > 0.0 and wavelength_m > 0.0):
        raise ValueError("spacing_m and wavelength_m must be > 0")
    if not exponent_q >= 1.0:
        raise ValueError("element pattern exponent q must be >= 1 (large-array lobe model)")
    step = wavelength_m / spacing_m
    m_max = math.floor(1.0 / step) + 1
    idx = np.arange(-m_max, m_max + 1)
    r2 = ((idx[:, None] ** 2 + idx[None, :] ** 2) * step**2).ravel()
    if np.any(np.abs(r2 - 1.0) < GRAZING_TOLERANCE):
        raise BeamModelValidityError(
            f"a grating lobe of the lattice (d = {spacing_m:.6g} m, lambda = {wavelength_m:.6g} m) "
            "lies at grazing incidence; the large-array lobe model is not valid there"
        )
    cos_theta = np.sqrt(1.0 - r2[r2 < 1.0])
    return float(1.0 / np.sum(cos_theta ** (exponent_q - 1.0)))


@dataclass(frozen=True)
class LatticeArray:
    """Uniform square phased-array lattice filling a circular aperture (broadside).

    Attributes:
        spacing_m: Element spacing ``d`` [m].
        aperture_diameter_m: Diameter of the filled circle [m].
        element_exponent_q: Element power-pattern exponent ``q`` (>= 1).
    """

    spacing_m: float
    aperture_diameter_m: float
    element_exponent_q: float = 1.0

    def __post_init__(self) -> None:
        if not (self.spacing_m > 0.0 and self.aperture_diameter_m > self.spacing_m):
            raise ValueError("need 0 < spacing_m < aperture_diameter_m")
        if not self.element_exponent_q >= 1.0:
            raise ValueError("element_exponent_q must be >= 1")

    @property
    def element_count(self) -> int:
        """Number of lattice sites ``(i d, j d)`` (centred lattice) inside the aperture circle.

        Counted row by row: row ``i`` holds ``2 floor(sqrt(m^2 - i^2)) + 1`` sites with
        ``m = r/d`` (integer arithmetic on ``floor``, O(sqrt N) work).
        """
        m = 0.5 * self.aperture_diameter_m / self.spacing_m
        i = np.arange(-math.floor(m), math.floor(m) + 1, dtype=np.float64)
        half = np.floor(np.sqrt(np.maximum(m * m - i * i, 0.0)))
        return int(np.sum(2 * half + 1))

    def grating_lobe_separation_rad(self, wavelength_m: float) -> float:
        """Angular distance to the nearest grating lobe, ``asin(lambda/d)`` (pi/2 if none) [rad]."""
        ratio = wavelength_m / self.spacing_m
        return math.asin(ratio) if ratio < 1.0 else 0.5 * math.pi

    def main_lobe_fraction(self, wavelength_m: float) -> float:
        """:func:`lattice_main_lobe_fraction` for this lattice."""
        return lattice_main_lobe_fraction(self.spacing_m, wavelength_m, self.element_exponent_q)

    def check_validity(self, wavelength_m: float, max_offset_angle_rad: float) -> None:
        """Raise :class:`BeamModelValidityError` outside the lattice model's validity range.

        Args:
            wavelength_m: Wavelength [m].
            max_offset_angle_rad: Largest angle from the beam axis at which the receiver is
                sampled (receiver angular radius + pointing bias + 9 sigma jitter) [rad].
        """
        n = self.element_count
        if n < MIN_LATTICE_ELEMENTS:
            raise BeamModelValidityError(
                f"lattice has only {n} elements (< {MIN_LATTICE_ELEMENTS}); the main lobe is "
                "not the continuous-aperture pattern"
            )
        sep = self.grating_lobe_separation_rad(wavelength_m)
        if wavelength_m < self.spacing_m and max_offset_angle_rad > GRATING_LOBE_CLEARANCE * sep:
            raise BeamModelValidityError(
                f"receiver/pointing angular extent {max_offset_angle_rad:.6g} rad exceeds "
                f"{GRATING_LOBE_CLEARANCE} of the grating-lobe separation {sep:.6g} rad"
            )


__all__ = [
    "GRATING_LOBE_CLEARANCE",
    "MIN_LATTICE_ELEMENTS",
    "LatticeArray",
    "dirichlet_kernel",
    "element_power_pattern",
    "lattice_main_lobe_fraction",
    "square_lattice_array_factor",
    "visible_grating_lobes",
]
