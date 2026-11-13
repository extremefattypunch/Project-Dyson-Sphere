"""Chebyshev-table evaluation of the beam over (range, elevation) (M7b numerical method).

The beam quantities of :class:`~helioswarm.beam.BeamResult` depend on an epoch only through the
transmitter-receiver range ``R`` and the elevation ``el`` (the settings are fixed for a run).
The reference evaluation (:func:`helioswarm.beam.evaluate_beam`) costs up to minutes per epoch
for a near-field optical beam on a large receiver (Fresnel-Hankel quadrature, work ~ (D R_rx /
(lambda R))^2), e.g. 4.4 s for D = 10 m, R_rx = 1 km and 268 s for D = 100 m at the GEO range
(measured, docs/milestones/M7b.md). A geostationary link has an almost constant geometry, so
a year can be evaluated on a small table:

1. **Nodes.** Chebyshev-Lobatto points ``x_j = cos(pi j/(n-1))`` mapped to
   ``[min R, max R]`` and ``[min el, max el]`` of the epochs (a single node for a zero span);
   the reference evaluator is called once per tensor node. ``n`` starts at 3 per dimension and
   is refined to ``2n - 1`` (the Lobatto sets are nested, so earlier evaluations are reused)
   while the last Chebyshev coefficient of any interpolated quantity along that dimension
   exceeds its tolerance (the standard tail estimate of the truncation error; e.g. Trefethen,
   *Approximation Theory and Approximation Practice*, SIAM 2013, ch. 3-4), up to ``max_nodes``.
2. **Interpolation.** Every per-epoch float quantity is interpolated with the tensor-product
   barycentric Lagrange formula (Berrut & Trefethen, "Barycentric Lagrange interpolation",
   SIAM Review 46 (2004) 501-517, eq. 4.2 with the Chebyshev-Lobatto weights
   ``(-1)^j delta_j``, ``delta_0 = delta_{n-1} = 1/2``). Closed-form quantities
   (``N_F = D^2/(lambda R)``, the projected receiver radius and its angular size) are
   recomputed exactly; ``pointing_loss`` is the ratio of the interpolated expected and nominal
   captures, as in the reference.
3. **Validation in every run.** The reference evaluator is called at two disjoint sets of
   check epochs - the epochs of extreme range and elevation plus ``check_epochs`` evenly
   spaced ones, and ``check_epochs`` epochs half-way between those - and every interpolated
   quantity is compared with it: capture fractions (and the receiver-projection diagnostic)
   to an absolute tolerance, the peak irradiance and the other quantities to a relative
   tolerance. If a check fails, both dimensions are refined and the checks repeated; at
   ``max_nodes`` a failure raises :class:`BeamTableError` (the run stops: the table never
   silently replaces the reference).

Validity of the reference model (far-field Fresnel-number limit, Fresnel paraxial condition,
lattice angle) is enforced at the nodes, which include the range and elevation extremes (the
Lobatto end points); these conditions are monotone in range, so the extremes cover the epochs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.aperture import fresnel_number
from helioswarm.beam import BeamResult, BeamSettings
from helioswarm.capture import projected_receiver_radius_m
from helioswarm.config import RunConfig

Float64Array = NDArray[np.float64]

BeamEvaluator = Callable[[Float64Array, Float64Array, Float64Array], BeamResult]
"""``(range_m, elevation_rad, epochs_tdb_s) -> BeamResult`` (the reference evaluator)."""

CAPTURE_FIELDS: Final[tuple[str, ...]] = (
    "capture_fraction",
    "capture_fraction_nominal",
    "capture_fraction_bias_only",
    "capture_fraction_p05",
    "capture_fraction_p50",
    "capture_fraction_p95",
)
"""Capture fractions (interpolated; checked to the absolute tolerance)."""

RELATIVE_FIELDS: Final[tuple[str, ...]] = ("peak_irradiance_per_W_m2",)
"""Interpolated :class:`~helioswarm.beam.BeamResult` arrays checked to the relative tolerance."""

DIAGNOSTIC_FIELDS: Final[tuple[str, ...]] = (
    "feature_scale_m",
    "first_null_radius_m",
    "gaussian_beam_radius_m",
    "receiver_projection_error",
    "offset_probability_mass",
)
"""Interpolated :class:`~helioswarm.beam.BeamFieldDiagnostics` arrays (all-NaN columns stay
NaN). ``receiver_projection_error`` (a difference of captures) is checked to the absolute
tolerance, the others relatively."""

ABSOLUTE_DIAGNOSTICS: Final[frozenset[str]] = frozenset({"receiver_projection_error"})

PERCENTILE_FIELDS: Final[frozenset[str]] = frozenset(
    {"capture_fraction_p05", "capture_fraction_p50", "capture_fraction_p95"}
)
"""Jitter percentiles of the capture (diagnostics; not used by the power chain). They are
checked to their own absolute tolerance because the reference switches between its
``exact_monotone`` and ``level_set`` methods (:mod:`helioswarm.capture`) with the geometry,
which makes them non-smooth at the 1e-7 level (measured: 3.4e-7 for a 10 km optical receiver
at 0.5 au)."""

INITIAL_NODES: Final[int] = 3
"""Nodes per dimension of the first table (refined as 3 -> 5 -> 9 -> 17 -> 33 -> ...)."""


class BeamTableError(RuntimeError):
    """The interpolated beam table does not reach its tolerance against the reference."""


@dataclass(frozen=True)
class BeamTableSettings:
    """Table limits and acceptance tolerances (``models.beam_table_*``).

    Attributes:
        max_nodes: Largest number of nodes per dimension (``2^k + 1``, >= 3).
        check_epochs: Evenly spaced check epochs per check set (two disjoint sets).
        capture_abs_tolerance: Largest accepted |interpolated - reference| of a capture
            fraction (expected, nominal, bias-only) and of the receiver-projection
            diagnostic [1].
        percentile_abs_tolerance: Same for the jitter percentiles p05/p50/p95 [1].
        relative_tolerance: Largest accepted relative difference of the peak irradiance and
            the other interpolated quantities [1].
    """

    max_nodes: int = 33
    check_epochs: int = 8
    capture_abs_tolerance: float = 1.0e-9
    percentile_abs_tolerance: float = 1.0e-6
    relative_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        k = math.log2(self.max_nodes - 1) if self.max_nodes > 1 else -1.0
        if self.max_nodes < INITIAL_NODES or k != int(k):
            raise ValueError("max_nodes must be 2^k + 1 >= 3 (3, 5, 9, 17, 33, 65, ...)")
        if self.check_epochs < 1:
            raise ValueError("check_epochs must be >= 1")
        for name in ("capture_abs_tolerance", "percentile_abs_tolerance", "relative_tolerance"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be finite and > 0")


@dataclass(frozen=True, eq=False)
class BeamTableDiagnostics:
    """What the table did and how well it agreed with the reference.

    Attributes:
        range_nodes_m: Range nodes of the final table [m].
        elevation_nodes_rad: Elevation nodes of the final table [rad].
        node_evaluations: Reference evaluations at nodes (all refinement levels).
        check_indices: Epoch indices (into the evaluated arrays) of both check sets.
        max_capture_abs_error: Largest |interp - ref| of the absolutely checked quantities.
        max_relative_error: Largest relative difference of the other quantities.
        worst_quantity: Quantity with the largest error relative to its tolerance.
        worst_score: That error divided by its tolerance (<= 1 for an accepted table).
        settings: Settings used.
    """

    range_nodes_m: Float64Array
    elevation_nodes_rad: Float64Array
    node_evaluations: int
    check_indices: NDArray[np.int64]
    max_capture_abs_error: float
    max_relative_error: float
    worst_quantity: str
    worst_score: float
    settings: BeamTableSettings = field(repr=False)

    def describe(self) -> str:
        """One-line run-log text."""
        return (
            f"beam Chebyshev table: {self.range_nodes_m.size} range x "
            f"{self.elevation_nodes_rad.size} elevation nodes (range "
            f"{float(np.min(self.range_nodes_m)):.9e}..{float(np.max(self.range_nodes_m)):.9e} m, "
            f"elevation {math.degrees(float(np.min(self.elevation_nodes_rad))):.9f}.."
            f"{math.degrees(float(np.max(self.elevation_nodes_rad))):.9f} deg), "
            f"{self.node_evaluations} node evaluations; check against the reference at "
            f"{self.check_indices.size} epochs: max absolute error "
            f"{self.max_capture_abs_error:.3e} (tolerance "
            f"{self.settings.capture_abs_tolerance:.1e}), max relative error "
            f"{self.max_relative_error:.3e} (tolerance {self.settings.relative_tolerance:.1e});"
            f" worst: {self.worst_quantity} at {self.worst_score:.3g} of its tolerance "
            f"(percentile tolerance {self.settings.percentile_abs_tolerance:.1e})"
        )


def beam_table_settings_from_config(config: RunConfig) -> BeamTableSettings:
    """:class:`BeamTableSettings` from ``models.beam_table_*`` of a validated configuration."""
    m = config.models
    return BeamTableSettings(
        max_nodes=m.beam_table_max_nodes,
        check_epochs=m.beam_table_check_epochs,
        capture_abs_tolerance=m.beam_table_capture_tolerance,
        percentile_abs_tolerance=m.beam_table_percentile_tolerance,
        relative_tolerance=m.beam_table_relative_tolerance,
    )


def chebyshev_lobatto_nodes(lo: float, hi: float, n: int) -> Float64Array:
    """``n`` Chebyshev-Lobatto points on ``[lo, hi]``, increasing, end points included exactly
    (``n = 1`` or ``lo == hi`` gives the single point ``lo``). Sets with ``n - 1 = 2^k`` are
    nested bit for bit (node ``j`` of ``n`` equals node ``2j`` of ``2n - 1``)."""
    if not (math.isfinite(lo) and math.isfinite(hi) and hi >= lo):
        raise ValueError("need finite lo <= hi")
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1 or hi == lo:
        return np.array([lo], dtype=np.float64)
    j = np.arange(n, dtype=np.float64)
    x = -np.cos(math.pi * (j / (n - 1)))
    nodes = 0.5 * (lo + hi) + 0.5 * (hi - lo) * x
    nodes[0], nodes[-1] = lo, hi
    return np.asarray(nodes, dtype=np.float64)


def lagrange_basis_matrix(nodes: Float64Array, x: ArrayLike) -> Float64Array:
    """Barycentric Lagrange basis on Chebyshev-Lobatto ``nodes`` at points ``x``.

    Returns ``L`` of shape (M, n) with ``p(x_m) = sum_j L[m, j] f(nodes_j)``; weights
    ``w_j = (-1)^j delta_j`` (Berrut & Trefethen 2004, eq. 5.4); an ``x`` equal to a node gives
    the unit row. One node: ``L = 1`` (used only for a zero span, where it is exact).
    """
    xq = np.atleast_1d(np.asarray(x, dtype=np.float64))
    n = nodes.size
    if n == 1:
        return np.ones((xq.size, 1), dtype=np.float64)
    w = (-1.0) ** np.arange(n)
    w[0] *= 0.5
    w[-1] *= 0.5
    diff = xq[:, None] - nodes[None, :]
    exact = diff == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = w[None, :] / diff
        mat = terms / np.sum(terms, axis=1, keepdims=True)
    rows = np.any(exact, axis=1)
    if np.any(rows):
        mat[rows] = exact[rows].astype(np.float64)
    return np.asarray(mat, dtype=np.float64)


def chebyshev_tail(values: Float64Array, axis: int) -> float:
    """Largest |last Chebyshev coefficient| of the Lobatto samples along ``axis``.

    ``c_k = (2/(n-1)) sum''_j f_j cos(pi j k/(n-1))`` (end terms halved; the last coefficient
    also halved), i.e. the discrete cosine transform of the samples on the Lobatto grid; 0 for a
    single node. NaN samples (quantities undefined for the model) are ignored.
    """
    v = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    n = v.shape[0]
    if n == 1:
        return 0.0
    j = np.arange(n, dtype=np.float64)
    weights = np.cos(math.pi * j * (n - 1) / (n - 1))  # (-1)^j, k = n - 1
    weights[0] *= 0.5
    weights[-1] *= 0.5
    c_last = (2.0 / (n - 1)) * 0.5 * np.tensordot(weights, v, axes=(0, 0))
    c_last = c_last[np.isfinite(c_last)]
    return float(np.max(np.abs(c_last))) if c_last.size else 0.0


def _check_sets(
    range_m: Float64Array, elevation_rad: Float64Array, n_check: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Two disjoint check sets: extremes + evenly spaced epochs, and the half-way epochs."""
    m = range_m.size
    k = min(n_check, m)
    first = {int(np.argmin(range_m)), int(np.argmax(range_m))}
    first |= {int(np.argmin(elevation_rad)), int(np.argmax(elevation_rad))}
    grid = np.linspace(0.0, m - 1.0, k)
    first |= {int(i) for i in np.round(grid)}
    half = (
        {int(i) for i in np.round(0.5 * (grid[:-1] + grid[1:]))} if k > 1 else {int(m // 2)}
    ) - first
    return (
        np.array(sorted(first), dtype=np.int64),
        np.array(sorted(half), dtype=np.int64),
    )


def _quantities(res: BeamResult) -> dict[str, Float64Array]:
    out = {name: np.asarray(getattr(res, name), dtype=np.float64) for name in CAPTURE_FIELDS}
    out.update({name: np.asarray(getattr(res, name), dtype=np.float64) for name in RELATIVE_FIELDS})
    out.update(
        {
            name: np.asarray(getattr(res.diagnostics, name), dtype=np.float64)
            for name in DIAGNOSTIC_FIELDS
        }
    )
    return out


def _is_absolute(name: str) -> bool:
    return name in CAPTURE_FIELDS or name in ABSOLUTE_DIAGNOSTICS


def _tolerance(name: str, table: BeamTableSettings) -> float:
    """Absolute tolerance of an absolutely checked quantity, else the relative tolerance."""
    if name in PERCENTILE_FIELDS:
        return table.percentile_abs_tolerance
    if _is_absolute(name):
        return table.capture_abs_tolerance
    return table.relative_tolerance


@dataclass
class _NodeCache:
    """Reference results at tensor nodes, keyed by the exact node values."""

    evaluate: BeamEvaluator
    values: dict[tuple[float, float], dict[str, float]] = field(default_factory=dict)
    methods: set[str] = field(default_factory=set)
    template: BeamResult | None = None
    evaluations: int = 0

    def grid(self, rn: Float64Array, en: Float64Array) -> dict[str, Float64Array]:
        missing = [
            (float(a), float(b)) for a in rn for b in en if (float(a), float(b)) not in self.values
        ]
        if missing:
            arr = np.array(missing, dtype=np.float64)
            res = self.evaluate(arr[:, 0], arr[:, 1], np.zeros(len(missing)))
            self.evaluations += len(missing)
            q = _quantities(res)
            for i, key in enumerate(missing):
                self.values[key] = {name: float(v[i]) for name, v in q.items()}
            self.methods.update(str(m) for m in res.diagnostics.percentile_method)
            if self.template is None:
                self.template = res
        names = next(iter(self.values.values())).keys()
        return {
            name: np.array(
                [[self.values[(float(a), float(b))][name] for b in en] for a in rn],
                dtype=np.float64,
            )
            for name in names
        }


def _interp_all(
    grids: dict[str, Float64Array],
    rn: Float64Array,
    en: Float64Array,
    r: Float64Array,
    el: Float64Array,
) -> dict[str, Float64Array]:
    lr = lagrange_basis_matrix(rn, r)
    le = lagrange_basis_matrix(en, el)
    out: dict[str, Float64Array] = {}
    for name, g in grids.items():
        if np.all(np.isnan(g)):
            out[name] = np.full(r.size, np.nan)
        else:
            out[name] = np.asarray(np.einsum("mi,ij,mj->m", lr, g, le), dtype=np.float64)
    for name in CAPTURE_FIELDS:
        out[name] = np.clip(out[name], 0.0, 1.0)
    return out


def _errors(
    interp: dict[str, Float64Array], ref: dict[str, Float64Array], table: BeamTableSettings
) -> tuple[float, float, str, float]:
    """Check errors: (max abs error of the capture-tolerance quantities, max relative error,
    worst quantity, worst error / its tolerance); the percentiles enter only the score."""
    max_abs, max_rel = 0.0, 0.0
    worst, worst_score = "", -1.0
    for name, want in ref.items():
        got = interp[name]
        nan_got, nan_want = np.isnan(got), np.isnan(want)
        if np.any(nan_got != nan_want):
            return math.inf, math.inf, name, math.inf
        g, w = got[~nan_want], want[~nan_want]
        if not g.size:
            continue
        if _is_absolute(name):
            err = float(np.max(np.abs(g - w)))
            if name not in PERCENTILE_FIELDS:
                max_abs = max(max_abs, err)
        else:
            err = float(np.max(np.abs(g - w) / np.maximum(np.abs(w), np.finfo(float).tiny)))
            max_rel = max(max_rel, err)
        score = err / _tolerance(name, table)
        if score > worst_score:
            worst, worst_score = name, score
    return max_abs, max_rel, worst, worst_score


def evaluate_beam_table(
    range_m: ArrayLike,
    elevation_rad: ArrayLike,
    settings: BeamSettings,
    *,
    table: BeamTableSettings,
    evaluate: BeamEvaluator,
    epochs_tdb_s: ArrayLike | None = None,
) -> tuple[BeamResult, BeamTableDiagnostics]:
    """Beam result per epoch from an adaptive, validated Chebyshev table (module docstring).

    Args:
        range_m: Range per epoch [m], shape (M,).
        elevation_rad: Elevation per epoch [rad], shape (M,).
        settings: Beam settings of the run (closed-form quantities).
        table: Table settings.
        evaluate: Reference evaluator ``(range, elevation, epochs) -> BeamResult`` (the run's
            beam backend).
        epochs_tdb_s: Epoch labels [TDB s] (optional).

    Returns:
        ``(BeamResult, BeamTableDiagnostics)``.

    Raises:
        BeamTableError: the tolerance is not reached at ``max_nodes`` nodes.
        BeamModelValidityError: from the reference evaluator at a node.
    """
    r = np.atleast_1d(np.asarray(range_m, dtype=np.float64))
    el = np.atleast_1d(np.asarray(elevation_rad, dtype=np.float64))
    if r.shape != el.shape or r.ndim != 1 or r.size == 0:
        raise ValueError("range_m and elevation_rad must be non-empty 1-D arrays of equal length")
    t = (
        np.zeros_like(r)
        if epochs_tdb_s is None
        else np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
    )
    set_a, set_b = _check_sets(r, el, table.check_epochs)
    idx = np.concatenate([set_a, set_b])
    ref = _quantities(evaluate(r[idx], el[idx], t[idx]))
    cache = _NodeCache(evaluate)
    lo_r, hi_r = float(np.min(r)), float(np.max(r))
    lo_e, hi_e = float(np.min(el)), float(np.max(el))
    n_r = n_e = INITIAL_NODES
    tol_rel = table.relative_tolerance
    while True:
        rn = chebyshev_lobatto_nodes(lo_r, hi_r, n_r)
        en = chebyshev_lobatto_nodes(lo_e, hi_e, n_e)
        grids = cache.grid(rn, en)
        # Tail estimates along each dimension, scaled by each quantity's tolerance.
        tail = [0.0, 0.0]
        for name, g in grids.items():
            scale = (
                _tolerance(name, table)
                if _is_absolute(name)
                else tol_rel
                * max(
                    float(np.nanmax(np.abs(g))) if np.any(np.isfinite(g)) else 0.0,
                    np.finfo(float).tiny,
                )
            )
            for axis in (0, 1):
                tail[axis] = max(tail[axis], chebyshev_tail(g, axis) / scale)
        refine_r = rn.size > 1 and tail[0] > 1.0 and n_r < table.max_nodes
        refine_e = en.size > 1 and tail[1] > 1.0 and n_e < table.max_nodes
        if not (refine_r or refine_e):
            interp = _interp_all(grids, rn, en, r[idx], el[idx])
            max_abs, max_rel, worst, score = _errors(interp, ref, table)
            ok = score <= 1.0
            if ok or (n_r >= table.max_nodes and n_e >= table.max_nodes):
                break
            refine_r = rn.size > 1 and n_r < table.max_nodes
            refine_e = en.size > 1 and n_e < table.max_nodes
            if not (refine_r or refine_e):
                break
        if refine_r:
            n_r = 2 * n_r - 1
        if refine_e:
            n_e = 2 * n_e - 1

    diagnostics = BeamTableDiagnostics(
        range_nodes_m=rn,
        elevation_nodes_rad=en,
        node_evaluations=cache.evaluations,
        check_indices=idx,
        max_capture_abs_error=max_abs,
        max_relative_error=max_rel,
        worst_quantity=worst,
        worst_score=score,
        settings=table,
    )
    if not score <= 1.0:
        raise BeamTableError(
            "the beam Chebyshev table misses its tolerance against the reference: "
            + diagnostics.describe()
            + "; raise models.beam_table_max_nodes or use models.beam_evaluation = 'per_epoch'"
        )

    # ---- assemble the result on every epoch ---------------------------------------------
    full = _interp_all(grids, rn, en, r, el)
    assert cache.template is not None
    tmpl = cache.template
    a_eff = projected_receiver_radius_m(
        settings.receiver_radius_m, el, settings.receiver_orientation
    )
    method = next(iter(cache.methods)) if len(cache.methods) == 1 else "table:mixed"
    with np.errstate(invalid="ignore", divide="ignore"):
        pointing_loss = np.where(
            full["capture_fraction_nominal"] > 0.0,
            full["capture_fraction"] / full["capture_fraction_nominal"],
            np.nan,
        )
    diag = replace(
        tmpl.diagnostics,
        feature_scale_m=full["feature_scale_m"],
        first_null_radius_m=full["first_null_radius_m"],
        gaussian_beam_radius_m=full["gaussian_beam_radius_m"],
        receiver_angular_radius_rad=a_eff / r,
        receiver_projection_error=full["receiver_projection_error"],
        offset_probability_mass=full["offset_probability_mass"],
        percentile_method=np.full(r.size, method, dtype="<U16"),
    )
    notes = (
        *tmpl.validity_notes,
        "beam quantities interpolated on an adaptive Chebyshev-Lobatto table over (range, "
        "elevation), validated against the reference at check epochs (helioswarm.beam_table)",
    )
    result = BeamResult(
        wavelength_m=tmpl.wavelength_m,
        aperture_m=tmpl.aperture_m,
        range_m=r,
        elevation_rad=el,
        beam_model=tmpl.beam_model,
        mode=tmpl.mode,
        fresnel_number=np.asarray(
            fresnel_number(settings.aperture.diameter_m, settings.wavelength_m, r),
            dtype=np.float64,
        ),
        fresnel_number_limit=tmpl.fresnel_number_limit,
        capture_fraction=full["capture_fraction"],
        capture_fraction_nominal=full["capture_fraction_nominal"],
        capture_fraction_bias_only=full["capture_fraction_bias_only"],
        capture_fraction_p05=full["capture_fraction_p05"],
        capture_fraction_p50=full["capture_fraction_p50"],
        capture_fraction_p95=full["capture_fraction_p95"],
        pointing_loss=pointing_loss,
        wavefront_efficiency=tmpl.wavefront_efficiency,
        lattice_main_lobe_fraction=tmpl.lattice_main_lobe_fraction,
        projected_receiver_radius_m=a_eff,
        pointing_bias_rad=tmpl.pointing_bias_rad,
        pointing_sigma_rad=tmpl.pointing_sigma_rad,
        peak_irradiance_per_W_m2=full["peak_irradiance_per_W_m2"],
        peak_irradiance_W_m2=None,
        exceeds_max_power_density=None,
        receiver_orientation=tmpl.receiver_orientation,
        diagnostics=diag,
        validity_notes=notes,
        epochs_tdb_s=None if epochs_tdb_s is None else t,
    )
    return result, diagnostics


__all__ = [
    "CAPTURE_FIELDS",
    "DIAGNOSTIC_FIELDS",
    "INITIAL_NODES",
    "PERCENTILE_FIELDS",
    "RELATIVE_FIELDS",
    "BeamEvaluator",
    "BeamTableDiagnostics",
    "BeamTableError",
    "BeamTableSettings",
    "beam_table_settings_from_config",
    "chebyshev_lobatto_nodes",
    "chebyshev_tail",
    "evaluate_beam_table",
    "lagrange_basis_matrix",
]
