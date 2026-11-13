"""Independent cross-check of the optical far field against HCIPy (PDF §9.2, §13.3).

HCIPy 0.7.1 (API verified from the installed source): ``make_pupil_grid(dims, diameter)``,
``make_circular_aperture(diameter)``, ``evaluate_supersampled(field_generator, grid,
oversampling)``, ``FraunhoferPropagator(input_grid, output_grid, focal_length)`` (a perfect lens,
Goodman 2005; the output plane coordinate is ``f * theta``), ``Wavefront(electric_field,
wavelength)``, ``Wavefront.total_power``, ``CartesianGrid(UnstructuredCoords([x, y]),
weights)``. ``FraunhoferPropagator.forward`` returns ``FT[E] / (i f lambda)`` with the
Fourier kernel ``exp(-2 pi i x.xi / (f lambda))`` (read from the installed source), so with
``f = 1`` the focal-plane coordinate is ``theta`` [rad] and ``|E|^2`` per unit input power is
the radiant intensity ``J(theta)`` [sr^-1 per W] that ``aperture.py`` computes.

The comparison is made at a physical wavelength and diameter (lambda = 1 um, D = 10 m; review
M5): with unit values a wrong power of lambda or D in ``J(0) = pi a^2 / lambda^2`` (or in the
angle scale ``lambda / D``) would be invisible. The absolute on-axis value ``pi a^2/lambda^2 =
7.85e13 sr^-1`` is asserted against HCIPy, and a deliberately wrong ``lambda^1`` scaling is
shown to fail.

HCIPy samples the pupil on a pixel grid (1024 px across the aperture, 8x supersampled edge),
so its effective aperture area differs from ``pi a^2`` by ~1e-3; the tolerances below (2e-3
relative on the profile, 3e-3 absolute on the encircled energy summed over 1/32-resolution
focal pixels) are set by that discretisation, not by our model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from helioswarm.aperture import ApertureField
from helioswarm.optical import airy_encircled_energy_x

hcipy = pytest.importorskip("hcipy")

pytestmark = pytest.mark.xcheck

D = 10.0  # m
LAM = 1.0e-6  # m
N_PUPIL = 1024


def _pupil_field(ap: ApertureField) -> object:
    pupil = hcipy.make_pupil_grid(N_PUPIL, D)
    mask = np.asarray(hcipy.evaluate_supersampled(hcipy.make_circular_aperture(D), pupil, 8))
    r = np.hypot(np.asarray(pupil.x), np.asarray(pupil.y))
    profile = np.asarray(ap.amplitude_m_inv(np.minimum(r, ap.radius_m)))
    return pupil, hcipy.Field((mask * profile).astype(complex), pupil)


@pytest.mark.parametrize(
    "ap",
    [
        ApertureField("uniform_circular", D),
        ApertureField("truncated_gaussian", D, gaussian_waist_radius_m=0.5 * D),
        ApertureField("parabolic_on_pedestal", D, edge_taper_dB=10.0),
    ],
    ids=["uniform", "truncated_gaussian_r1", "parabolic_10dB"],
)
def test_far_field_profile_matches_hcipy(ap: ApertureField) -> None:
    pupil, field = _pupil_field(ap)
    theta = np.array([0.0, 0.3, 0.7, 1.0, 1.5, 2.2, 3.1]) * LAM / D
    out = hcipy.CartesianGrid(
        hcipy.UnstructuredCoords([theta, np.zeros_like(theta)]), weights=np.ones_like(theta)
    )
    prop = hcipy.FraunhoferPropagator(pupil, out, focal_length=1.0)
    wf = hcipy.Wavefront(field, wavelength=LAM)
    img = prop(wf)
    j_hcipy = np.abs(np.asarray(img.electric_field)) ** 2 / float(wf.total_power)
    j_ours = ap.radiant_intensity_sr_inv(theta, LAM)
    peak = float(j_ours[0])
    # Relative 2e-3 on the profile (pupil pixelisation), absolute 2e-3 of the peak near nulls.
    np.testing.assert_allclose(j_hcipy, j_ours, rtol=2e-3, atol=2e-3 * peak)
    # Discriminating: a 1 % change of the diameter (or the wrong taper) is detected.
    wrong = ApertureField("uniform_circular", 1.01 * D).radiant_intensity_sr_inv(theta, LAM)
    if ap.illumination == "uniform_circular":
        assert np.max(np.abs(wrong - j_hcipy) / peak) > 5e-3
        # Absolute dimensional check: J(0) = pi a^2 / lambda^2 (Airy) in sr^-1 per W.
        j0_closed = math.pi * (D / 2) ** 2 / LAM**2
        assert j_hcipy[0] == pytest.approx(j0_closed, rel=2e-3)
        # A wrong power of lambda (pi a^2 / lambda) is off by a factor 1e6.
        assert abs(math.pi * (D / 2) ** 2 / LAM - j_hcipy[0]) > 0.5 * j_hcipy[0]


def test_airy_encircled_energy_matches_hcipy() -> None:
    pupil, field = _pupil_field(ApertureField("uniform_circular", D))
    # Focal grid in theta (f = 1): resolution element lambda f / D = 1e-7 rad.
    focal = hcipy.make_focal_grid(q=32, num_airy=6, spatial_resolution=LAM / D)
    prop = hcipy.FraunhoferPropagator(pupil, focal, focal_length=1.0)
    wf = hcipy.Wavefront(field, wavelength=LAM)
    power = np.asarray(prop(wf).power)
    r = np.hypot(np.asarray(focal.x), np.asarray(focal.y))
    radii = np.array([0.5, 1.0, 1.2196698912665045, 2.0, 3.0, 5.0])  # in lambda/D
    ee_hcipy = np.array([power[r <= rr * LAM / D].sum() for rr in radii]) / float(wf.total_power)
    ee_closed = airy_encircled_energy_x(math.pi * radii)
    np.testing.assert_allclose(ee_hcipy, ee_closed, atol=3e-3)
    # At the first dark ring both give 0.838 (Rayleigh).
    assert ee_hcipy[2] == pytest.approx(0.8378, abs=3e-3)
