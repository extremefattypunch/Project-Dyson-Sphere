"""Chebyshev beam table (M7b): interpolation exactness, refinement, validation and refusal."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from helioswarm.beam import BeamResult, beam_settings_from_config, evaluate_beam
from helioswarm.beam_table import (
    BeamTableError,
    BeamTableSettings,
    chebyshev_lobatto_nodes,
    chebyshev_tail,
    evaluate_beam_table,
    lagrange_basis_matrix,
)
from helioswarm.config import parse_config


def test_lobatto_nodes_are_nested_and_include_the_end_points() -> None:
    lo, hi = 3.57e7, 3.58e7
    for n in (3, 5, 9, 17):
        a = chebyshev_lobatto_nodes(lo, hi, n)
        b = chebyshev_lobatto_nodes(lo, hi, 2 * n - 1)
        assert a[0] == lo
        assert a[-1] == hi
        assert np.all(np.diff(a) > 0)
        # bit-for-bit nesting (reuse of the evaluations when refining)
        assert np.array_equal(a, b[::2])
    assert np.array_equal(chebyshev_lobatto_nodes(2.0, 2.0, 9), [2.0])


def test_lagrange_basis_reproduces_polynomials_exactly() -> None:
    nodes = chebyshev_lobatto_nodes(-1.0, 3.0, 5)
    x = np.linspace(-1.0, 3.0, 41)
    poly = np.polynomial.Polynomial([0.3, -1.2, 0.5, 2.0, -0.7])  # degree 4 = n - 1
    lb = lagrange_basis_matrix(nodes, x)
    assert np.max(np.abs(lb @ poly(nodes) - poly(x))) < 1e-12
    np.testing.assert_allclose(lb.sum(axis=1), 1.0, rtol=0, atol=1e-14)
    # At a node the basis row is the unit vector (no 0/0).
    assert np.array_equal(lagrange_basis_matrix(nodes, nodes[2:3])[0], np.eye(5)[2])
    # Degree 5 on 5 nodes is not exact: the interpolant differs.
    p5 = np.polynomial.Polynomial([0, 0, 0, 0, 0, 1.0])
    assert np.max(np.abs(lb @ p5(nodes) - p5(x))) > 1e-3


def test_chebyshev_tail_is_the_last_coefficient() -> None:
    # f = T_4 on [-1, 1] sampled at 5 Lobatto points: last coefficient = 1; T_2 gives 0.
    x = chebyshev_lobatto_nodes(-1.0, 1.0, 5)
    t4 = np.cos(4.0 * np.arccos(np.clip(x, -1, 1)))
    t2 = np.cos(2.0 * np.arccos(np.clip(x, -1, 1)))
    assert chebyshev_tail(t4[:, None], 0) == pytest.approx(1.0, abs=1e-14)
    assert chebyshev_tail(t2[:, None], 0) == pytest.approx(0.0, abs=1e-14)
    assert chebyshev_tail(np.ones((1, 3)), 0) == 0.0


def _settings(**overrides: Any) -> Any:
    from config_helpers import load_mvp_data

    d = load_mvp_data()
    d["models"]["beam"] = "fresnel_hankel"
    for k, v in overrides.items():
        section, key = k.split(".")
        d[section][key] = v
    return beam_settings_from_config(parse_config(d))


def _reference(settings: Any) -> Any:
    calls: list[int] = []

    def ev(r: np.ndarray, e: np.ndarray, t: np.ndarray) -> BeamResult:
        calls.append(r.size)
        return evaluate_beam(r, e, settings, epochs_tdb_s=t)

    return ev, calls


@pytest.mark.slow  # Fresnel-Hankel reference evaluations (M7b review: suite budget)
def test_geo_like_geometry_needs_few_nodes_and_matches_the_reference() -> None:
    s = _settings(**{"receiver.radius_m": 100.0})
    x = np.linspace(0.0, 30.0, 500)
    r = 3.5786e7 + 3.6e3 * np.sin(x)
    el = 0.5 * math.pi - 9e-6 * (1.0 + 0.05 * np.cos(x))
    ev, calls = _reference(s)
    res, diag = evaluate_beam_table(r, el, s, table=BeamTableSettings(), evaluate=ev)
    assert diag.range_nodes_m.size <= 5
    assert diag.elevation_nodes_rad.size <= 5
    assert diag.node_evaluations == sum(calls) - diag.check_indices.size
    # Independent check at epochs that were neither nodes nor check epochs (tolerance: the
    # configured 1e-9 on captures, 1e-6 relative on the peak).
    idx = np.setdiff1d(np.arange(7, 500, 37), diag.check_indices)
    direct = evaluate_beam(r[idx], el[idx], s)
    assert np.max(np.abs(res.capture_fraction[idx] - direct.capture_fraction)) < 1e-9
    assert (
        np.max(np.abs(res.capture_fraction_nominal[idx] - direct.capture_fraction_nominal)) < 1e-9
    )
    np.testing.assert_allclose(
        res.peak_irradiance_per_W_m2[idx], direct.peak_irradiance_per_W_m2, rtol=1e-6
    )
    # Closed-form quantities are exact, not interpolated.
    np.testing.assert_array_equal(res.fresnel_number[idx], direct.fresnel_number)
    np.testing.assert_array_equal(
        res.projected_receiver_radius_m[idx], direct.projected_receiver_radius_m
    )
    assert any("Chebyshev" in n for n in res.validity_notes)


@pytest.mark.slow  # Fresnel-Hankel reference evaluations (M7b review: suite budget)
def test_wide_elevation_span_is_refined_until_the_tolerance_holds() -> None:
    s = _settings()  # 10 m optical, 1 km receiver: far field at 0.5 au
    x = np.linspace(0.0, 6.0, 600)
    r = 7.7e10 * (1.0 + 0.03 * np.sin(x / 3.0))
    el = math.radians(10.0) + math.radians(80.0) * np.abs(np.sin(5.0 * x))
    ev, _ = _reference(s)
    res, diag = evaluate_beam_table(r, el, s, table=BeamTableSettings(), evaluate=ev)
    assert diag.elevation_nodes_rad.size >= 9  # sin(el) over 80 deg needs refinement
    idx = np.arange(5, 600, 53)
    direct = evaluate_beam(r[idx], el[idx], s)
    assert np.max(np.abs(res.capture_fraction[idx] - direct.capture_fraction)) < 1e-9


@pytest.mark.slow  # Fresnel-Hankel reference evaluations (M7b review: suite budget)
def test_a_table_that_cannot_reach_the_tolerance_is_refused() -> None:
    """A reference with a jump (no polynomial interpolates it) must raise, not interpolate."""
    s = _settings(**{"receiver.radius_m": 100.0})
    r = np.linspace(3.578e7, 3.579e7, 200)
    el = np.full(200, 0.5 * math.pi - 1e-5)

    def jumpy(rr: np.ndarray, ee: np.ndarray, tt: np.ndarray) -> BeamResult:
        res = evaluate_beam(rr, ee, s, epochs_tdb_s=tt)
        cap = np.where(rr > 3.5785e7, res.capture_fraction, 0.5 * res.capture_fraction)
        return dataclasses.replace(res, capture_fraction=cap)  # captures halved below a range

    with pytest.raises(BeamTableError, match="misses its tolerance"):
        evaluate_beam_table(r, el, s, table=BeamTableSettings(max_nodes=5), evaluate=jumpy)


def test_settings_validation() -> None:
    with pytest.raises(ValueError, match="2\\^k \\+ 1"):
        BeamTableSettings(max_nodes=10)
    with pytest.raises(ValueError, match="> 0"):
        BeamTableSettings(capture_abs_tolerance=0.0)
