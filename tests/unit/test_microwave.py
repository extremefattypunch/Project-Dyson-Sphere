"""Tests of the microwave lattice geometry: array factor, grating lobes, main-lobe fraction (M5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helioswarm.aperture import BeamModelValidityError
from helioswarm.microwave import (
    LatticeArray,
    dirichlet_kernel,
    element_power_pattern,
    lattice_main_lobe_fraction,
    square_lattice_array_factor,
    visible_grating_lobes,
)

LAM = 0.1


def test_dirichlet_kernel_equals_explicit_sum() -> None:
    # Centred sum sum_m exp(i (m - (n-1)/2) psi) computed directly.
    for n in (1, 2, 7, 16):
        psi = np.linspace(-13.0, 13.0, 1001)
        m = np.arange(n) - (n - 1) / 2
        explicit = np.exp(1j * np.outer(psi, m)).sum(axis=1)
        np.testing.assert_allclose(dirichlet_kernel(psi, n), explicit.real, atol=1e-10)
        assert np.max(np.abs(explicit.imag)) < 1e-9  # centred array factor is real
        for k in (-2, -1, 0, 1, 3):
            assert float(dirichlet_kernel(np.array([2 * math.pi * k]), n)[0]) == pytest.approx(
                n * (-1.0) ** (k * (n - 1))
            )


@pytest.mark.parametrize(("n_side", "spacing"), [(8, 0.6 * LAM), (16, 1.6 * LAM), (5, 2.5 * LAM)])
def test_array_factor_peak_nulls_and_grating_lobes(n_side: int, spacing: float) -> None:
    n_tot = n_side**2
    assert abs(float(square_lattice_array_factor(0.0, 0.0, n_side, spacing, LAM))) == pytest.approx(
        n_tot
    )
    # Peak N is the maximum over all directions.
    u = np.linspace(-1, 1, 801)
    af = square_lattice_array_factor(u[:, None], u[None, :], n_side, spacing, LAM)
    assert np.max(np.abs(af)) <= n_tot * (1 + 1e-12)
    # First null at u = lambda / (n d).
    assert (
        abs(float(square_lattice_array_factor(LAM / (n_side * spacing), 0.0, n_side, spacing, LAM)))
        < 1e-9 * n_tot
    )
    # Grating lobes at u = m lambda / d, with |AF| = N there.
    lobes = visible_grating_lobes(spacing, LAM)
    step = LAM / spacing
    expected = sorted(
        (m * step, k * step)
        for m in range(-3, 4)
        for k in range(-3, 4)
        if (m * step) ** 2 + (k * step) ** 2 < 1
    )
    assert sorted(map(tuple, np.round(lobes, 12))) == sorted(
        (round(x, 12), round(y, 12)) for x, y in expected
    )
    assert tuple(lobes[0]) == (0.0, 0.0)
    for uu, vv in lobes:
        assert abs(
            float(square_lattice_array_factor(uu, vv, n_side, spacing, LAM))
        ) == pytest.approx(n_tot, rel=1e-9)
    if spacing < LAM:
        assert len(lobes) == 1


def test_element_power_pattern() -> None:
    th = np.radians([0.0, 60.0, 90.0, 120.0])
    np.testing.assert_allclose(element_power_pattern(th, 2.0), [1.0, 0.25, 0.0, 0.0], atol=1e-16)


def _numeric_main_lobe_fraction(n_side: int, spacing: float, q: float, n_grid: int) -> float:
    """Direct hemisphere integral of cos^q |AF|^2 dOmega, dOmega = du dv / cos(theta)."""
    u = -1 + (2 * np.arange(n_grid) + 1) / n_grid
    total = 0.0
    main = 0.0
    half_cell = 0.5 * LAM / spacing
    for rows in np.array_split(np.arange(n_grid), 16):
        uu = u[rows][:, None]
        vv = u[None, :]
        r2 = uu * uu + vv * vv
        vis = r2 < 1
        cos_t = np.sqrt(np.where(vis, 1 - r2, 1.0))
        af2 = square_lattice_array_factor(uu, vv, n_side, spacing, LAM) ** 2
        integrand = np.where(vis, cos_t ** (q - 1) * af2, 0.0)
        total += float(integrand.sum())
        main += float(
            np.where((np.abs(uu) < half_cell) & (np.abs(vv) < half_cell), integrand, 0.0).sum()
        )
    return main / total


@pytest.mark.parametrize(
    ("spacing", "q", "expected"),
    [(1.6 * LAM, 1.0, 1 / 9), (1.6 * LAM, 2.0, None), (0.7 * LAM, 1.5, 1.0)],
)
def test_lattice_main_lobe_fraction_against_hemisphere_integral(
    spacing: float, q: float, expected: float | None
) -> None:
    frac = lattice_main_lobe_fraction(spacing, LAM, q)
    if expected is not None:
        assert frac == pytest.approx(expected, rel=1e-12)
    else:
        # q = 2: 1 / (1 + 4 cos th1 + 4 cos th2), sin th1 = 0.625, sin^2 th2 = 2 * 0.625^2.
        s1 = 0.625
        assert frac == pytest.approx(
            1 / (1 + 4 * math.sqrt(1 - s1**2) + 4 * math.sqrt(1 - 2 * s1**2)), rel=1e-12
        )
    # Independent numerical integration for a finite 24 x 24 lattice; the large-array lobe model
    # neglects the finite-N sidelobe exchange between period cells, O(1/n) -> 3 % tolerance.
    numeric = _numeric_main_lobe_fraction(24, spacing, q, 1600)
    assert numeric == pytest.approx(frac, rel=0.03)


def test_lattice_validity_checks() -> None:
    with pytest.raises(BeamModelValidityError, match="grazing"):
        lattice_main_lobe_fraction(LAM, LAM, 1.0)  # lobe exactly at u = 1
    with pytest.raises(ValueError, match="q"):
        lattice_main_lobe_fraction(2 * LAM, LAM, 0.5)
    arr = LatticeArray(spacing_m=0.06, aperture_diameter_m=1000.0, element_exponent_q=1.0)
    # Centred lattice in a circle: ~ pi (D/2)^2 / d^2 sites (Gauss circle problem, rel 1e-4).
    assert arr.element_count == pytest.approx(math.pi * 500.0**2 / 0.06**2, rel=1e-4)
    arr.check_validity(299_792_458.0 / 2.45e9, 1e-3)  # d < lambda: no grating lobes
    sparse = LatticeArray(spacing_m=0.3, aperture_diameter_m=30.0, element_exponent_q=1.0)
    assert sparse.grating_lobe_separation_rad(0.1) == pytest.approx(math.asin(1 / 3))
    sparse.check_validity(0.1, 0.01)
    with pytest.raises(BeamModelValidityError, match="grating-lobe"):
        sparse.check_validity(0.1, 0.1)
    with pytest.raises(BeamModelValidityError, match="elements"):
        LatticeArray(spacing_m=0.3, aperture_diameter_m=2.0).check_validity(0.1, 1e-6)
