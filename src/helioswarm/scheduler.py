"""Deterministic, capacity-constrained link scheduler of a constellation (PDF §11.2, §10.4).

At every epoch a set of *feasible* satellite-to-site links is given (every availability
condition of the single link met: geometry, occultation, eclipse, thermal, weather, pointing
and the single-beam power-density abort; :func:`helioswarm.system.site_link_budget`). Each
link ``l = (i, j)`` carries the beam power it would put into the receiver aperture
``P_l = P_captured`` [W] and the peak incident irradiance ``d_l`` [W m^-2] of its beam. The
scheduler chooses a subset of links subject to:

* **beam exclusivity**: a satellite serves at most one site per epoch (one beam per satellite;
  no beam splitting);
* **beam slots**: site ``j`` accepts at most ``k_j`` beams (``None`` = unlimited);
* **power-density interlock on the sum**: ``sum_{l in j} d_l <= D_j``. The beams of different
  satellites are mutually incoherent (independent transmitters, PDF §11.3): intensities add,
  fields are never added. Each ``d_l`` is the aberration-free peak of its own pattern (an upper
  bound) and the peaks are assumed to coincide (all beams aimed at the receiver centre), so the
  sum is an upper bound of the incident peak irradiance;
* **receiver input and heat limits on the sum**: the accepted power of a site is
  ``A_j = min(sum_{l in j} P_l, L_j)`` with ``L_j = min(P_max_input, Q_max / (1 - eta_rx))``
  (M6 single-link definitions applied to the summed incident power). Power above ``L_j`` is
  clipped (receiver saturation), never counted;
* **transmitter duty cycle** (optional, enforced by :func:`schedule_timeseries`): a satellite
  that has transmitted in ``m = floor(duty * N_w)`` of the previous ``N_w - 1`` epochs may not
  transmit now (at most ``m`` transmitting epochs in any ``N_w`` consecutive ones). ``N_w`` and
  ``m`` are evaluated in exact rational arithmetic on the configured decimal values
  (:func:`helioswarm.config.duty_cycle_epochs`), so ``0.29 x 100`` gives 29, not the 28 of
  binary floating point.

The objective is the total accepted power ``sum_j A_j`` (every site has the same receiver
conversion and grid efficiencies, so this is also the total ``P_grid``). Links with
``P_l = 0`` deliver nothing and are never scheduled.

Policies
--------
``greedy_max_power``
    Links in the order (``P_l`` descending, satellite index, site index) - a total order, so
    the result does not depend on the order in which the links are supplied - each taken iff
    its satellite is still free, its site has a free beam slot, the site is not yet saturated
    (``sum P < L_j``) and adding its ``d_l`` keeps the density sum within ``D_j``.
``optimal_assignment``
    The per-epoch maximum of the objective (exact in steps 1-2; certified to the HiGHS
    tolerances in step 3, see below):

    1. if every satellite's best link (largest ``P``; ties to the lowest site index) together
       form a feasible, unsaturated set, that set is optimal (it attains the upper bound
       ``sum_i max_j P_ij``);
    2. else, if no site's power or density limit can bind (the ``k_j`` largest candidate powers
       sum to ``<= L_j`` and the ``k_j`` largest densities to ``<= D_j``), the problem is a
       bipartite assignment of satellites to beam slots solved with
       ``scipy.optimize.linear_sum_assignment`` (site ``j`` replicated ``k_j`` times);
    3. otherwise a mixed-integer program solved with ``scipy.optimize.milp`` (HiGHS) with
       ``mip_rel_gap = 0`` and ``mip_abs_gap = 0`` (the latter is not a documented ``milp``
       option; scipy passes it to HiGHS verbatim, whose default 1e-6 would otherwise certify
       the optimum only to 1e-6 of the largest link power): binary ``x_l``, continuous
       ``0 <= y_j <= L_j``, ``y_j <= sum_l P_l x_l``, ``sum_{l of i} x_l <= 1``,
       ``sum_{l at j} x_l <= k_j``, ``sum_{l at j} d_l x_l <= D_j``, maximise ``sum_j y_j``.
       Powers and densities are scaled to O(1) (by ``max P`` and ``D_j``) before the solve.
       HiGHS still works to its primal feasibility and integrality tolerances (default 1e-6
       on the scaled problem), so the optimum is certified to that tolerance, not exactly. The
       rounded solution is checked against every constraint exactly and, if the tolerance let
       a violation through, repaired by dropping the weakest beam of the violating site
       (counted in ``EpochSchedule.repaired``): a repaired epoch is feasible but **not
       certified optimal**.

The candidate links are first put in canonical (satellite, site) order, so both policies are
deterministic and invariant under permutations of their input.

Decision codes per link (:class:`LinkDecision`): ``SELECTED``; ``SCHEDULER_NOT_SELECTED``
(the satellite serves another site, or the scheduler preferred other links); ``RECEIVER_CAPACITY``
(the site's beam slots are full, it is saturated, or the beam would break its density
interlock); ``DUTY_CYCLE`` (the satellite must rest).
"""

from __future__ import annotations

import enum
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp

from helioswarm.config import SchedulerPolicy, duty_cycle_epochs

Float64Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

SolveMethod = Literal["empty", "best_link", "greedy", "assignment", "milp"]
SOLVE_METHODS: Final[tuple[SolveMethod, ...]] = (
    "empty",
    "best_link",
    "greedy",
    "assignment",
    "milp",
)


MILP_OPTIONS: Final[tuple[tuple[str, float | bool], ...]] = (
    ("mip_rel_gap", 0.0),
    ("mip_abs_gap", 0.0),
    ("presolve", True),
)
"""Options of the ``optimal_assignment`` MILP (``mip_abs_gap`` is passed to HiGHS verbatim)."""


class LinkDecision(enum.IntEnum):
    """Scheduler decision for one feasible link at one epoch."""

    SELECTED = 0
    SCHEDULER_NOT_SELECTED = 1
    RECEIVER_CAPACITY = 2
    DUTY_CYCLE = 3


@dataclass(frozen=True)
class SiteCapacity:
    """Capacity of one receiver site (PDF §10.4).

    Attributes:
        max_beams: Beam slots ``k_j`` (``None`` = unlimited).
        max_accepted_W: ``L_j = min(P_max_input, Q_max/(1 - eta_rx))`` [W] (``inf`` allowed).
        max_power_density_W_m2: Density interlock ``D_j`` on the summed peak irradiance
            [W m^-2] (``None`` = no interlock).
    """

    max_beams: int | None
    max_accepted_W: float
    max_power_density_W_m2: float | None = None

    def __post_init__(self) -> None:
        if self.max_beams is not None and self.max_beams < 1:
            raise ValueError("max_beams must be >= 1 or None")
        if not (self.max_accepted_W > 0.0):
            raise ValueError("max_accepted_W must be > 0 (inf allowed)")
        d = self.max_power_density_W_m2
        if d is not None and not (math.isfinite(d) and d > 0.0):
            raise ValueError("max_power_density_W_m2 must be finite and > 0 or None")


@dataclass(frozen=True)
class EpochSchedule:
    """Schedule of one epoch.

    Attributes:
        decision: :class:`LinkDecision` per input link (input order), ``uint8``.
        site_incident_W: ``sum P_l`` of the selected links per site [W].
        site_accepted_W: ``min(site_incident_W, L_j)`` per site [W].
        site_beams: Selected beams per site.
        site_density_W_m2: Summed peak irradiance of the selected beams per site [W m^-2].
        objective_W: ``sum_j site_accepted_W`` [W].
        method: How the epoch was solved (``empty``, ``best_link``, ``greedy``,
            ``assignment``, ``milp``).
        repaired: Beams dropped after a MILP solution violated a constraint by round-off.
    """

    decision: NDArray[np.uint8]
    site_incident_W: Float64Array
    site_accepted_W: Float64Array
    site_beams: IntArray
    site_density_W_m2: Float64Array
    objective_W: float
    method: SolveMethod
    repaired: int = 0


def _site_totals(
    sel: BoolArray,
    site: IntArray,
    power: Float64Array,
    density: Float64Array,
    capacities: Sequence[SiteCapacity],
) -> tuple[Float64Array, Float64Array, IntArray, Float64Array]:
    n_site = len(capacities)
    incident = np.zeros(n_site)
    beams = np.zeros(n_site, dtype=np.int64)
    dens = np.zeros(n_site)
    # Sequential sums in canonical (satellite) order: deterministic floating-point results.
    for idx in np.flatnonzero(sel):
        j = int(site[idx])
        incident[j] += power[idx]
        beams[j] += 1
        dens[j] += density[idx]
    limit = np.array([c.max_accepted_W for c in capacities], dtype=np.float64)
    accepted = np.minimum(incident, limit)
    return incident, accepted, beams, dens


def _attribute(
    sel: BoolArray,
    sat: IntArray,
    site: IntArray,
    power: Float64Array,
    density: Float64Array,
    capacities: Sequence[SiteCapacity],
    incident: Float64Array,
    beams: IntArray,
    dens: Float64Array,
) -> NDArray[np.uint8]:
    """Decision code of every link from the final state (module docstring)."""
    n_busy = int(np.max(sat)) + 1 if sat.size else 0
    busy = np.zeros(n_busy, dtype=np.bool_)
    busy[sat[sel]] = True
    out = np.full(sat.size, int(LinkDecision.SCHEDULER_NOT_SELECTED), dtype=np.uint8)
    out[sel] = int(LinkDecision.SELECTED)
    for idx in np.flatnonzero(~sel):
        if busy[sat[idx]] or not power[idx] > 0.0:
            continue
        cap = capacities[int(site[idx])]
        j = int(site[idx])
        full = cap.max_beams is not None and beams[j] >= cap.max_beams
        saturated = incident[j] >= cap.max_accepted_W
        over = (
            cap.max_power_density_W_m2 is not None
            and dens[j] + density[idx] > cap.max_power_density_W_m2
        )
        if full or saturated or over:
            out[idx] = int(LinkDecision.RECEIVER_CAPACITY)
    return out


def _check_inputs(
    satellite: ArrayLike,
    site: ArrayLike,
    power_W: ArrayLike,
    density_W_m2: ArrayLike,
    capacities: Sequence[SiteCapacity],
) -> tuple[IntArray, IntArray, Float64Array, Float64Array, IntArray]:
    sat = np.asarray(satellite, dtype=np.int64).reshape(-1)
    st = np.asarray(site, dtype=np.int64).reshape(-1)
    p = np.asarray(power_W, dtype=np.float64).reshape(-1)
    d = np.asarray(density_W_m2, dtype=np.float64).reshape(-1)
    if not (sat.size == st.size == p.size == d.size):
        raise ValueError("satellite, site, power_W and density_W_m2 must have equal length")
    if np.any(sat < 0) or np.any(st < 0) or np.any(st >= len(capacities)):
        raise ValueError("satellite indices must be >= 0 and site indices < len(capacities)")
    if np.any(~np.isfinite(p)) or np.any(p < 0.0) or np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("power_W and density_W_m2 must be finite and >= 0")
    order = np.lexsort((st, sat))  # canonical (satellite, site) order
    if sat.size > 1:
        s2, t2 = sat[order], st[order]
        dup = (s2[1:] == s2[:-1]) & (t2[1:] == t2[:-1])
        if np.any(dup):
            raise ValueError("duplicate (satellite, site) link")
    return sat[order], st[order], p[order], d[order], order


def _best_links(sat: IntArray, site: IntArray, power: Float64Array) -> BoolArray:
    """Each satellite's largest-power link (ties: lowest site index); powers > 0 only."""
    sel = np.zeros(sat.size, dtype=bool)
    if not sat.size:
        return sel
    # Canonical order is (sat, site); sort by (sat, -power, site) and take the first per sat.
    order = np.lexsort((site, -power, sat))
    first = np.ones(order.size, dtype=bool)
    first[1:] = sat[order][1:] != sat[order][:-1]
    pick = order[first]
    sel[pick[power[pick] > 0.0]] = True
    return sel


def _feasible_unsaturated(
    sel: BoolArray,
    site: IntArray,
    power: Float64Array,
    density: Float64Array,
    capacities: Sequence[SiteCapacity],
) -> bool:
    incident, _, beams, dens = _site_totals(sel, site, power, density, capacities)
    for j, cap in enumerate(capacities):
        if cap.max_beams is not None and beams[j] > cap.max_beams:
            return False
        if incident[j] > cap.max_accepted_W:
            return False
        if cap.max_power_density_W_m2 is not None and dens[j] > cap.max_power_density_W_m2:
            return False
    return True


def _greedy(
    sat: IntArray,
    site: IntArray,
    power: Float64Array,
    density: Float64Array,
    capacities: Sequence[SiteCapacity],
) -> BoolArray:
    sel = np.zeros(sat.size, dtype=bool)
    order = np.lexsort((site, sat, -power))
    n_site = len(capacities)
    busy: set[int] = set()
    incident = [0.0] * n_site
    beams = [0] * n_site
    dens = [0.0] * n_site
    k = [c.max_beams for c in capacities]
    lim = [c.max_accepted_W for c in capacities]
    dmax = [c.max_power_density_W_m2 for c in capacities]
    for idx in order.tolist():
        p = float(power[idx])
        if not p > 0.0:
            break  # sorted descending: the rest deliver nothing
        i = int(sat[idx])
        j = int(site[idx])
        if i in busy:
            continue
        kj = k[j]
        if kj is not None and beams[j] >= kj:
            continue
        if incident[j] >= lim[j]:
            continue
        dj = dmax[j]
        dl = float(density[idx])
        if dj is not None and dens[j] + dl > dj:
            continue
        sel[idx] = True
        busy.add(i)
        incident[j] += p
        beams[j] += 1
        dens[j] += dl
    return sel


def _limits_cannot_bind(
    site: IntArray, power: Float64Array, density: Float64Array, capacities: Sequence[SiteCapacity]
) -> bool:
    for j, cap in enumerate(capacities):
        at = site == j
        n = int(np.count_nonzero(at))
        if n == 0:
            continue
        kj = n if cap.max_beams is None else min(n, cap.max_beams)
        top_p = np.sort(power[at])[::-1][:kj]
        if float(np.sum(top_p)) > cap.max_accepted_W:
            return False
        if cap.max_power_density_W_m2 is not None:
            top_d = np.sort(density[at])[::-1][:kj]
            if float(np.sum(top_d)) > cap.max_power_density_W_m2:
                return False
    return True


def _assignment(
    sat: IntArray, site: IntArray, power: Float64Array, capacities: Sequence[SiteCapacity]
) -> BoolArray:
    sats = np.unique(sat)
    row_of = {int(s): r for r, s in enumerate(sats.tolist())}
    cols: list[int] = []  # column -> site
    for j, cap in enumerate(capacities):
        n = int(np.count_nonzero(site == j))
        if n:
            cols.extend([j] * (n if cap.max_beams is None else min(n, cap.max_beams)))
    col_site = np.array(cols, dtype=np.int64)
    weight = np.zeros((sats.size, col_site.size))
    link_at = np.full((sats.size, len(capacities)), -1, dtype=np.int64)
    for idx in range(sat.size):
        link_at[row_of[int(sat[idx])], int(site[idx])] = idx
    for c, j in enumerate(col_site.tolist()):
        rows = np.flatnonzero(link_at[:, j] >= 0)
        weight[rows, c] = power[link_at[rows, j]]
    r_ind, c_ind = linear_sum_assignment(weight, maximize=True)
    sel = np.zeros(sat.size, dtype=bool)
    for r, c in zip(r_ind.tolist(), c_ind.tolist(), strict=True):
        idx = int(link_at[r, col_site[c]])
        if idx >= 0 and power[idx] > 0.0:
            sel[idx] = True
    return sel


def _milp(
    sat: IntArray,
    site: IntArray,
    power: Float64Array,
    density: Float64Array,
    capacities: Sequence[SiteCapacity],
) -> tuple[BoolArray, int]:
    n_l = sat.size
    n_s = len(capacities)
    p_scale = float(np.max(power))
    pw = power / p_scale
    c = np.concatenate([np.zeros(n_l), -np.ones(n_s)])
    lim = np.array([cap.max_accepted_W for cap in capacities]) / p_scale
    lb = np.zeros(n_l + n_s)
    ub = np.concatenate([np.ones(n_l), lim])
    rows: list[sparse.csr_matrix] = []
    lo: list[float] = []
    hi: list[float] = []
    link_idx = np.arange(n_l)
    # y_j - sum P x <= 0
    a = sparse.lil_matrix((n_s, n_l + n_s))
    for j in range(n_s):
        at = link_idx[site == j]
        a[j, at] = -pw[at]
        a[j, n_l + j] = 1.0
    rows.append(a.tocsr())
    lo += [-np.inf] * n_s
    hi += [0.0] * n_s
    # one site per satellite
    sats = np.unique(sat)
    a = sparse.lil_matrix((sats.size, n_l + n_s))
    for r, s in enumerate(sats.tolist()):
        a[r, link_idx[sat == s]] = 1.0
    rows.append(a.tocsr())
    lo += [-np.inf] * sats.size
    hi += [1.0] * sats.size
    for j, cap in enumerate(capacities):
        at = link_idx[site == j]
        if not at.size:
            continue
        if cap.max_beams is not None and at.size > cap.max_beams:
            a = sparse.lil_matrix((1, n_l + n_s))
            a[0, at] = 1.0
            rows.append(a.tocsr())
            lo.append(-np.inf)
            hi.append(float(cap.max_beams))
        dmax = cap.max_power_density_W_m2
        if dmax is not None and float(np.sum(density[at])) > dmax:
            a = sparse.lil_matrix((1, n_l + n_s))
            a[0, at] = density[at] / dmax
            rows.append(a.tocsr())
            lo.append(-np.inf)
            hi.append(1.0)
    constraint = LinearConstraint(sparse.vstack(rows).tocsr(), np.array(lo), np.array(hi))
    integrality = np.concatenate([np.ones(n_l), np.zeros(n_s)])
    with warnings.catch_warnings():
        # mip_abs_gap is a HiGHS option that scipy forwards verbatim with a RuntimeWarning.
        warnings.filterwarnings(
            "ignore", message="Unrecognized options detected", category=RuntimeWarning
        )
        res = milp(
            c,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=constraint,
            options=dict(MILP_OPTIONS),
        )
    if not res.success or res.x is None:
        raise RuntimeError(f"scheduler MILP failed: status {res.status}, {res.message}")
    sel = (np.asarray(res.x[:n_l]) > 0.5) & (power > 0.0)
    # Exact feasibility check of the rounded solution (HiGHS works to a feasibility tolerance).
    repaired = 0
    while True:
        _, _, beams, dens = _site_totals(sel, site, power, density, capacities)
        bad = [
            j
            for j, cap in enumerate(capacities)
            if (cap.max_beams is not None and beams[j] > cap.max_beams)
            or (cap.max_power_density_W_m2 is not None and dens[j] > cap.max_power_density_W_m2)
        ]
        counts = np.bincount(sat[sel], minlength=int(np.max(sat)) + 1)
        if counts.size and int(np.max(counts)) > 1:
            raise RuntimeError("scheduler MILP assigned a satellite twice after rounding")
        if not bad:
            break
        j = bad[0]
        at = np.flatnonzero(sel & (site == j))
        weakest = at[np.lexsort((-sat[at], power[at]))[0]]
        sel[weakest] = False
        repaired += 1
    return sel, repaired


def schedule_epoch(
    satellite: ArrayLike,
    site: ArrayLike,
    power_W: ArrayLike,
    density_W_m2: ArrayLike,
    capacities: Sequence[SiteCapacity],
    *,
    policy: SchedulerPolicy = "greedy_max_power",
) -> EpochSchedule:
    """Schedule the feasible links of one epoch (module docstring).

    Args:
        satellite: Satellite index per link.
        site: Site index per link (``0 .. len(capacities) - 1``).
        power_W: ``P_captured`` per link [W] (>= 0).
        density_W_m2: Peak incident irradiance of each link's beam [W m^-2] (>= 0).
        capacities: Capacity per site.
        policy: ``greedy_max_power`` or ``optimal_assignment``.

    Returns:
        :class:`EpochSchedule` (decisions in the input order).

    Raises:
        ValueError: inconsistent inputs, duplicate links or an unknown policy.
    """
    if policy not in ("greedy_max_power", "optimal_assignment"):
        raise ValueError(f"unknown scheduler policy {policy!r}")
    sat, st, p, d, order = _check_inputs(satellite, site, power_W, density_W_m2, capacities)
    n_site = len(capacities)
    method: SolveMethod
    repaired = 0
    if sat.size == 0 or not np.any(p > 0.0):
        sel = np.zeros(sat.size, dtype=bool)
        method = "empty"
    else:
        best = _best_links(sat, st, p)
        if _feasible_unsaturated(best, st, p, d, capacities):
            sel, method = best, "best_link"
        elif policy == "greedy_max_power":
            sel, method = _greedy(sat, st, p, d, capacities), "greedy"
        else:
            pos = p > 0.0
            idx = np.flatnonzero(pos)
            if _limits_cannot_bind(st[pos], p[pos], d[pos], capacities):
                sub = _assignment(sat[pos], st[pos], p[pos], capacities)
                method = "assignment"
            else:
                sub, repaired = _milp(sat[pos], st[pos], p[pos], d[pos], capacities)
                method = "milp"
            sel = np.zeros(sat.size, dtype=bool)
            sel[idx[sub]] = True
    incident, accepted, beams, dens = _site_totals(sel, st, p, d, capacities)
    canon = _attribute(sel, sat, st, p, d, capacities, incident, beams, dens)
    decision = np.empty(sat.size, dtype=np.uint8)
    decision[order] = canon
    return EpochSchedule(
        decision=decision,
        site_incident_W=incident,
        site_accepted_W=accepted,
        site_beams=beams if n_site else np.zeros(0, dtype=np.int64),
        site_density_W_m2=dens,
        objective_W=float(np.sum(accepted)),
        method=method,
        repaired=repaired,
    )


@dataclass(frozen=True)
class DutyCycle:
    """Transmitter duty-cycle limit: at most ``max_on`` transmitting epochs in any
    ``window_epochs`` consecutive output epochs (causal, per satellite).

    Attributes:
        window_epochs: ``N_w = floor(duty_cycle_window_s / output_step_s)`` (>= 1).
        max_on: ``m = floor(duty_cycle_max * N_w)`` (>= 1).
    """

    window_epochs: int
    max_on: int

    def __post_init__(self) -> None:
        if self.window_epochs < 1 or not 1 <= self.max_on <= self.window_epochs:
            raise ValueError("need window_epochs >= 1 and 1 <= max_on <= window_epochs")

    @classmethod
    def from_limits(cls, duty_cycle_max: float, window_s: float, step_s: float) -> DutyCycle:
        """``N_w = floor(window_s / step_s)``, ``m = floor(duty_cycle_max N_w)`` in exact
        rational arithmetic on the decimal values (:func:`helioswarm.config.duty_cycle_epochs`)."""
        n_w, m = duty_cycle_epochs(duty_cycle_max, window_s, step_s)
        return cls(window_epochs=n_w, max_on=m)


@dataclass(frozen=True, eq=False)
class ScheduleTimeseries:
    """Schedule of every epoch (``T`` epochs, ``N`` satellites, ``S`` sites).

    Attributes:
        decision: :class:`LinkDecision` per (epoch, satellite, site), ``uint8``; 255 where the
            link was not feasible (not a scheduler input).
        assigned_site: Site served by each satellite, ``-1`` if none, shape (T, N), ``int16``.
        site_incident_W: Summed ``P_captured`` of the scheduled beams, (T, S) [W].
        site_accepted_W: Accepted power ``min(incident, L_j)``, (T, S) [W].
        site_beams: Scheduled beams per site, (T, S).
        site_density_W_m2: Summed peak irradiance of the scheduled beams, (T, S) [W m^-2].
        method: Solve method per epoch (:data:`SOLVE_METHODS` index), ``uint8``.
        repaired: MILP round-off repairs per epoch.
        retarget_events: Satellites whose (non-idle) target differs from their previous
            non-idle target, per epoch.
    """

    decision: NDArray[np.uint8]
    assigned_site: NDArray[np.int16]
    site_incident_W: Float64Array
    site_accepted_W: Float64Array
    site_beams: IntArray
    site_density_W_m2: Float64Array
    method: NDArray[np.uint8]
    repaired: IntArray
    retarget_events: IntArray


NOT_A_CANDIDATE: Final[int] = 255
"""``ScheduleTimeseries.decision`` value of a link that was not feasible at that epoch."""


def schedule_timeseries(
    feasible: ArrayLike,
    power_W: ArrayLike,
    density_W_m2: ArrayLike,
    capacities: Sequence[SiteCapacity],
    *,
    policy: SchedulerPolicy = "greedy_max_power",
    duty_cycle: DutyCycle | None = None,
) -> ScheduleTimeseries:
    """Schedule every epoch of a constellation (arrays shaped (T, N, S)).

    Epochs are independent except through the causal duty-cycle state; with
    ``duty_cycle=None`` every epoch is scheduled by :func:`schedule_epoch` alone.

    Args:
        feasible: Link feasible (all single-link conditions met), bool (T, N, S).
        power_W: ``P_captured`` per link [W] (ignored where not feasible).
        density_W_m2: Peak incident irradiance per link [W m^-2] (ignored where not feasible).
        capacities: One :class:`SiteCapacity` per site (``S``).
        policy: Scheduler policy.
        duty_cycle: Optional :class:`DutyCycle`.
    """
    f = np.asarray(feasible, dtype=bool)
    if f.ndim != 3 or f.shape[2] != len(capacities):
        raise ValueError("feasible must have shape (T, N, S) with S = len(capacities)")
    p_all = np.asarray(power_W, dtype=np.float64)
    d_all = np.asarray(density_W_m2, dtype=np.float64)
    if p_all.shape != f.shape or d_all.shape != f.shape:
        raise ValueError("power_W and density_W_m2 must have the shape of feasible")
    n_t, n_sat, n_site = f.shape
    decision = np.full(f.shape, NOT_A_CANDIDATE, dtype=np.uint8)
    assigned = np.full((n_t, n_sat), -1, dtype=np.int16)
    incident = np.zeros((n_t, n_site))
    accepted = np.zeros((n_t, n_site))
    beams = np.zeros((n_t, n_site), dtype=np.int64)
    dens = np.zeros((n_t, n_site))
    method = np.zeros(n_t, dtype=np.uint8)
    repaired = np.zeros(n_t, dtype=np.int64)
    retarget = np.zeros(n_t, dtype=np.int64)
    on_count = np.zeros(n_sat, dtype=np.int64)
    last_target = np.full(n_sat, -1, dtype=np.int64)
    for k in range(n_t):
        if duty_cycle is not None:
            if k >= 1:
                on_count += assigned[k - 1] >= 0
            if k - duty_cycle.window_epochs >= 0:
                on_count -= assigned[k - duty_cycle.window_epochs] >= 0
            resting = on_count >= duty_cycle.max_on
        sat, st = np.nonzero(f[k])
        p = p_all[k, sat, st]
        d = d_all[k, sat, st]
        if duty_cycle is not None and np.any(resting[sat]):
            rest = resting[sat]
            decision[k, sat[rest], st[rest]] = int(LinkDecision.DUTY_CYCLE)
            sat, st, p, d = sat[~rest], st[~rest], p[~rest], d[~rest]
        es = schedule_epoch(sat, st, p, d, capacities, policy=policy)
        decision[k, sat, st] = es.decision
        chosen = es.decision == int(LinkDecision.SELECTED)
        assigned[k, sat[chosen]] = st[chosen]
        incident[k], accepted[k], beams[k], dens[k] = (
            es.site_incident_W,
            es.site_accepted_W,
            es.site_beams,
            es.site_density_W_m2,
        )
        method[k] = SOLVE_METHODS.index(es.method)
        repaired[k] = es.repaired
        tgt = assigned[k].astype(np.int64)
        active = tgt >= 0
        retarget[k] = int(np.count_nonzero(active & (last_target >= 0) & (tgt != last_target)))
        last_target[active] = tgt[active]
    return ScheduleTimeseries(
        decision=decision,
        assigned_site=assigned,
        site_incident_W=incident,
        site_accepted_W=accepted,
        site_beams=beams,
        site_density_W_m2=dens,
        method=method,
        repaired=repaired,
        retarget_events=retarget,
    )


__all__ = [
    "MILP_OPTIONS",
    "NOT_A_CANDIDATE",
    "SOLVE_METHODS",
    "DutyCycle",
    "EpochSchedule",
    "LinkDecision",
    "ScheduleTimeseries",
    "SchedulerPolicy",
    "SiteCapacity",
    "SolveMethod",
    "schedule_epoch",
    "schedule_timeseries",
]
