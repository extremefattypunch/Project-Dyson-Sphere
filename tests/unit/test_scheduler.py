"""PDF §13.1 scheduler row ("no receiver over-capacity, no double allocation and deterministic
scheduling") and §13.2 ("a receiver capacity limit clips or schedules incoming power
correctly") for :mod:`helioswarm.scheduler`.

References are independent: a brute-force enumeration of every assignment (each satellite to
none or one of its feasible sites) written here, closed-form sums, and hand-built instances
whose optimum is evident. Tolerances: exact comparisons for counts and assignments; ``1e-9``
relative on objective comparisons between the MILP and the enumeration (HiGHS solves to a
feasibility tolerance of 1e-7 on scaled O(1) data; the rounded solution is re-evaluated
exactly, so only ties at that level could differ).
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest

from helioswarm.config import duty_cycle_epochs
from helioswarm.scheduler import (
    MILP_OPTIONS,
    NOT_A_CANDIDATE,
    DutyCycle,
    LinkDecision,
    SiteCapacity,
    schedule_epoch,
    schedule_timeseries,
)

INF = math.inf


def _value(
    sel_sites: dict[int, int],
    links: dict[tuple[int, int], tuple[float, float]],
    caps: list[SiteCapacity],
) -> float | None:
    """Objective of an assignment {sat: site}, or None if it breaks a hard constraint."""
    inc = [0.0] * len(caps)
    beams = [0] * len(caps)
    dens = [0.0] * len(caps)
    for i, j in sel_sites.items():
        p, d = links[(i, j)]
        inc[j] += p
        beams[j] += 1
        dens[j] += d
    for j, c in enumerate(caps):
        if c.max_beams is not None and beams[j] > c.max_beams:
            return None
        if c.max_power_density_W_m2 is not None and dens[j] > c.max_power_density_W_m2:
            return None
    return float(sum(min(inc[j], caps[j].max_accepted_W) for j in range(len(caps))))


def brute_force(
    links: dict[tuple[int, int], tuple[float, float]], caps: list[SiteCapacity]
) -> float:
    """Maximum of sum_j min(sum P, L_j) over every assignment (independent reference)."""
    sats = sorted({i for i, _ in links})
    options = [[None, *sorted(j for (s, j) in links if s == i)] for i in sats]
    best = 0.0
    for combo in itertools.product(*options):
        sel = {i: j for i, j in zip(sats, combo, strict=True) if j is not None}
        v = _value(sel, links, caps)
        if v is not None and v > best:
            best = v
    return best


def _random_instance(rng: np.random.Generator, n_sat: int, n_site: int, density: bool, limit: bool):
    links: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(n_sat):
        for j in range(n_site):
            if rng.random() < 0.7:
                links[(i, j)] = (float(rng.uniform(0.1, 10.0)), float(rng.uniform(0.1, 1.0)))
    caps = []
    for _ in range(n_site):
        k = int(rng.integers(1, 3)) if rng.random() < 0.7 else None
        lim = float(rng.uniform(3.0, 15.0)) if limit and rng.random() < 0.7 else INF
        dmax = float(rng.uniform(0.5, 1.5)) if density and rng.random() < 0.5 else None
        caps.append(SiteCapacity(max_beams=k, max_accepted_W=lim, max_power_density_W_m2=dmax))
    return links, caps


def _arrays(links: dict[tuple[int, int], tuple[float, float]]):
    keys = list(links)
    sat = np.array([k[0] for k in keys], dtype=np.int64)
    site = np.array([k[1] for k in keys], dtype=np.int64)
    p = np.array([links[k][0] for k in keys])
    d = np.array([links[k][1] for k in keys])
    return keys, sat, site, p, d


def _check_hard_constraints(res, sat, site, p, d, caps) -> None:
    sel = res.decision == LinkDecision.SELECTED
    # no satellite twice (beam exclusivity)
    if sat.size:
        assert np.bincount(sat[sel], minlength=sat.max() + 1).max(initial=0) <= 1
    for j, c in enumerate(caps):
        at = sel & (site == j)
        assert res.site_beams[j] == int(np.count_nonzero(at))
        if c.max_beams is not None:
            assert res.site_beams[j] <= c.max_beams
        if c.max_power_density_W_m2 is not None:
            assert float(np.sum(d[at])) <= c.max_power_density_W_m2
        assert res.site_incident_W[j] == pytest.approx(float(np.sum(p[at])), rel=1e-15)
        assert res.site_accepted_W[j] == min(res.site_incident_W[j], c.max_accepted_W)
        assert res.site_accepted_W[j] <= c.max_accepted_W


@pytest.mark.parametrize("seed", range(40))
def test_optimal_equals_brute_force_and_dominates_greedy(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n_sat = int(rng.integers(1, 6))
    n_site = int(rng.integers(1, 4))
    links, caps = _random_instance(rng, n_sat, n_site, density=seed % 2 == 0, limit=seed % 3 != 0)
    _, sat, site, p, d = _arrays(links)
    ref = brute_force(links, caps)
    opt = schedule_epoch(sat, site, p, d, caps, policy="optimal_assignment")
    gre = schedule_epoch(sat, site, p, d, caps, policy="greedy_max_power")
    for res in (opt, gre):
        _check_hard_constraints(res, sat, site, p, d, caps)
    assert opt.objective_W == pytest.approx(ref, rel=1e-9, abs=1e-12)
    assert opt.objective_W >= gre.objective_W * (1.0 - 1e-12)


def test_greedy_is_strictly_suboptimal_on_a_known_instance() -> None:
    """Classic greedy trap: sat 0 has the largest link (10 at site 0) but sat 1 can only use
    site 0 (9), while sat 0 also reaches site 1 (8). Greedy: 10 (sat 1 blocked) = 10.
    Optimal: sat 0 -> site 1 (8) and sat 1 -> site 0 (9) = 17. One slot per site."""
    caps = [SiteCapacity(1, INF), SiteCapacity(1, INF)]
    sat = np.array([0, 0, 1])
    site = np.array([0, 1, 0])
    p = np.array([10.0, 8.0, 9.0])
    d = np.zeros(3)
    gre = schedule_epoch(sat, site, p, d, caps, policy="greedy_max_power")
    opt = schedule_epoch(sat, site, p, d, caps, policy="optimal_assignment")
    assert gre.objective_W == 10.0
    assert opt.objective_W == 17.0
    assert opt.method == "assignment"
    np.testing.assert_array_equal(
        gre.decision,
        [
            LinkDecision.SELECTED,
            LinkDecision.SCHEDULER_NOT_SELECTED,
            LinkDecision.RECEIVER_CAPACITY,
        ],
    )


def test_no_contention_gives_equal_greedy_and_optimal_and_takes_every_best_link() -> None:
    caps = [SiteCapacity(None, INF), SiteCapacity(None, INF), SiteCapacity(None, INF)]
    rng = np.random.default_rng(7)
    sat = np.repeat(np.arange(6), 3)
    site = np.tile(np.arange(3), 6)
    p = rng.uniform(1.0, 2.0, sat.size)
    d = rng.uniform(0.0, 1.0, sat.size)
    gre = schedule_epoch(sat, site, p, d, caps, policy="greedy_max_power")
    opt = schedule_epoch(sat, site, p, d, caps, policy="optimal_assignment")
    np.testing.assert_array_equal(gre.decision, opt.decision)
    assert gre.method == opt.method == "best_link"
    expected = float(sum(p.reshape(6, 3).max(axis=1)))
    assert gre.objective_W == pytest.approx(expected, rel=1e-15)


def test_permutation_of_input_order_gives_identical_assignment() -> None:
    rng = np.random.default_rng(3)
    for policy in ("greedy_max_power", "optimal_assignment"):
        links, caps = _random_instance(rng, 5, 3, density=True, limit=True)
        keys, sat, site, p, d = _arrays(links)
        base = schedule_epoch(sat, site, p, d, caps, policy=policy)
        chosen = {keys[k] for k in np.flatnonzero(base.decision == LinkDecision.SELECTED)}
        for _ in range(5):
            perm = rng.permutation(len(keys))
            res = schedule_epoch(sat[perm], site[perm], p[perm], d[perm], caps, policy=policy)
            got = {keys[perm[k]] for k in np.flatnonzero(res.decision == LinkDecision.SELECTED)}
            assert got == chosen
            np.testing.assert_array_equal(res.site_accepted_W, base.site_accepted_W)
            np.testing.assert_array_equal(res.decision, base.decision[perm])


def test_ties_are_broken_by_satellite_then_site_index() -> None:
    """Equal powers: greedy takes the lowest satellite index first, and a satellite with
    equal links takes the lowest site index."""
    caps = [SiteCapacity(1, INF), SiteCapacity(1, INF)]
    res = schedule_epoch([2, 1, 0], [0, 0, 0], [5.0, 5.0, 5.0], [0, 0, 0], caps)
    np.testing.assert_array_equal(
        res.decision,
        [LinkDecision.RECEIVER_CAPACITY, LinkDecision.RECEIVER_CAPACITY, LinkDecision.SELECTED],
    )
    res = schedule_epoch([0, 0], [1, 0], [5.0, 5.0], [0, 0], [SiteCapacity(None, INF)] * 2)
    np.testing.assert_array_equal(
        res.decision, [LinkDecision.SCHEDULER_NOT_SELECTED, LinkDecision.SELECTED]
    )


def test_receiver_power_limit_clips_the_sum_and_stops_further_beams() -> None:
    """§13.2: three beams of 4, 3, 2 W at one site with L = 6 W and unlimited slots. Greedy
    takes 4 (sum 4 < 6), then 3 (sum 7 >= 6, accepted = 6 = L), then rejects 2 (saturated:
    RECEIVER_CAPACITY). Accepted power is exactly L; nothing above L is counted."""
    caps = [SiteCapacity(None, 6.0)]
    res = schedule_epoch([0, 1, 2], [0, 0, 0], [4.0, 3.0, 2.0], [0, 0, 0], caps)
    np.testing.assert_array_equal(
        res.decision,
        [LinkDecision.SELECTED, LinkDecision.SELECTED, LinkDecision.RECEIVER_CAPACITY],
    )
    assert res.site_incident_W[0] == 7.0
    assert res.site_accepted_W[0] == 6.0
    assert res.objective_W == 6.0


def test_summed_power_density_interlock_is_enforced() -> None:
    """Incoherent sum of peak irradiances: two beams of density 0.6 at a site with
    D = 1.0 cannot both be scheduled (0.6 + 0.6 > 1.0), although each alone is allowed."""
    caps = [SiteCapacity(None, INF, max_power_density_W_m2=1.0)]
    for policy in ("greedy_max_power", "optimal_assignment"):
        res = schedule_epoch([0, 1], [0, 0], [5.0, 4.0], [0.6, 0.6], caps, policy=policy)
        assert res.site_beams[0] == 1
        assert res.site_density_W_m2[0] == 0.6
        assert res.objective_W == 5.0


def test_incoherent_sum_adds_powers_not_amplitudes() -> None:
    """Two beams of P1 = 1 W and P2 = 4 W on one receiver: the scheduled incident power is
    P1 + P2 = 5 W. A coherent (in-phase field) sum would give (sqrt(P1) + sqrt(P2))^2 = 9 W."""
    res = schedule_epoch([0, 1], [0, 0], [1.0, 4.0], [0.0, 0.0], [SiteCapacity(None, INF)])
    assert res.site_incident_W[0] == 5.0
    assert res.site_incident_W[0] != (math.sqrt(1.0) + math.sqrt(4.0)) ** 2


def test_milp_path_handles_binding_power_limits() -> None:
    """Instance where only the MILP is exact: site 0 has L = 5, site 1 L = inf, 1 slot each
    is not binding (3 slots). Sats 0, 1 deliver 5 to site 0 each and 4, 3 to site 1. Optimum:
    one of them to site 0 (5), the other to site 1 -> sat 0 -> site 0 (5) + sat 1 -> site 1
    (3) = 8 or sat 1 -> site 0 (5) + sat 0 -> site 1 (4) = 9. Optimal = 9."""
    caps = [SiteCapacity(3, 5.0), SiteCapacity(3, INF)]
    res = schedule_epoch(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [5.0, 4.0, 5.0, 3.0],
        [0.0] * 4,
        caps,
        policy="optimal_assignment",
    )
    assert res.method == "milp"
    assert res.objective_W == pytest.approx(9.0, rel=1e-12)
    np.testing.assert_array_equal(
        res.decision,
        [
            LinkDecision.SCHEDULER_NOT_SELECTED,
            LinkDecision.SELECTED,
            LinkDecision.SELECTED,
            LinkDecision.SCHEDULER_NOT_SELECTED,
        ],
    )


def test_zero_power_links_are_never_scheduled_and_bad_inputs_raise() -> None:
    res = schedule_epoch([0], [0], [0.0], [0.0], [SiteCapacity(None, INF)])
    assert res.method == "empty"
    assert res.decision[0] == LinkDecision.SCHEDULER_NOT_SELECTED
    with pytest.raises(ValueError, match="duplicate"):
        schedule_epoch([0, 0], [0, 0], [1.0, 2.0], [0, 0], [SiteCapacity(None, INF)])
    with pytest.raises(ValueError, match="site indices"):
        schedule_epoch([0], [1], [1.0], [0.0], [SiteCapacity(None, INF)])
    with pytest.raises(ValueError, match="policy"):
        schedule_epoch([0], [0], [1.0], [0.0], [SiteCapacity(None, INF)], policy="best")  # type: ignore[arg-type]


def test_schedule_timeseries_duty_cycle_limits_every_window() -> None:
    """One always-feasible satellite, duty 0.5 over a 4-epoch window (m = 2): at most two
    transmitting epochs in every 4 consecutive ones; with these rules the pattern is
    on, on, off, off, on, on, ..."""
    t = 12
    f = np.ones((t, 1, 1), dtype=bool)
    p = np.ones((t, 1, 1))
    d = np.zeros((t, 1, 1))
    duty = DutyCycle.from_limits(0.5, 4.0, 1.0)
    assert (duty.window_epochs, duty.max_on) == (4, 2)
    out = schedule_timeseries(f, p, d, [SiteCapacity(None, INF)], duty_cycle=duty)
    on = out.assigned_site[:, 0] >= 0
    np.testing.assert_array_equal(on, [1, 1, 0, 0] * 3)
    for k in range(t - 3):
        assert on[k : k + 4].sum() <= 2
    assert np.all(out.decision[~on, 0, 0] == LinkDecision.DUTY_CYCLE)
    free = schedule_timeseries(f, p, d, [SiteCapacity(None, INF)])
    assert np.all(free.assigned_site >= 0)


def test_schedule_timeseries_marks_infeasible_links_and_counts_retargets() -> None:
    # sat 0 alternates between site 0 and site 1 being its only feasible site
    f = np.zeros((4, 1, 2), dtype=bool)
    f[[0, 2], 0, 0] = True
    f[[1, 3], 0, 1] = True
    p = np.ones(f.shape)
    out = schedule_timeseries(f, p, np.zeros(f.shape), [SiteCapacity(None, INF)] * 2)
    np.testing.assert_array_equal(out.assigned_site[:, 0], [0, 1, 0, 1])
    np.testing.assert_array_equal(out.retarget_events, [0, 1, 1, 1])
    assert np.all(out.decision[~f] == NOT_A_CANDIDATE)


def test_schedule_timeseries_is_deterministic_bitwise() -> None:
    rng = np.random.default_rng(11)
    f = rng.random((30, 8, 3)) < 0.6
    p = rng.uniform(0, 10, f.shape)
    d = rng.uniform(0, 1, f.shape)
    caps = [SiteCapacity(2, 12.0, 1.5), SiteCapacity(3, INF), SiteCapacity(None, 9.0)]
    for policy in ("greedy_max_power", "optimal_assignment"):
        a = schedule_timeseries(f, p, d, caps, policy=policy)
        b = schedule_timeseries(f, p, d, caps, policy=policy)
        np.testing.assert_array_equal(a.decision, b.decision)
        assert a.site_accepted_W.tobytes() == b.site_accepted_W.tobytes()
        # every epoch: optimal >= greedy
    g = schedule_timeseries(f, p, d, caps, policy="greedy_max_power")
    o = schedule_timeseries(f, p, d, caps, policy="optimal_assignment")
    assert np.all(o.site_accepted_W.sum(axis=1) >= g.site_accepted_W.sum(axis=1) * (1 - 1e-12))
    assert np.any(o.site_accepted_W.sum(axis=1) > g.site_accepted_W.sum(axis=1) * (1 + 1e-9))


@pytest.mark.parametrize(
    ("duty", "window_s", "step_s", "expected"),
    [
        # 0.29 * 100 = 28.999999999999996 in binary floating point: must give 29
        (0.29, 100 * 3896.0, 3896.0, (100, 29)),
        (0.57, 100.0, 1.0, (100, 57)),
        (0.5, 4.0, 1.0, (4, 2)),
        # a genuinely fractional allowance is still floored
        (0.295, 100.0, 1.0, (100, 29)),
        # window not a multiple of the step: N_w = floor(10.5) = 10, m = floor(3.3) = 3
        (0.33, 10.5, 1.0, (10, 3)),
        # 0.7 / 0.1 = 6.999999999999999 in binary floating point: N_w must be 7
        (1.0, 0.7, 0.1, (7, 7)),
    ],
)
def test_duty_cycle_epochs_are_exact_on_decimal_inputs(
    duty: float, window_s: float, step_s: float, expected: tuple[int, int]
) -> None:
    """The duty-cycle floors are taken on the configured decimal values (exact rationals):
    expected values are the decimal arithmetic by hand. The scheduler and the configuration
    validator use the same function."""
    assert duty_cycle_epochs(duty, window_s, step_s) == expected
    dc = DutyCycle.from_limits(duty, window_s, step_s)
    assert (dc.window_epochs, dc.max_on) == expected


def test_milp_is_solved_with_zero_absolute_and_relative_gap_and_no_warning() -> None:
    """HiGHS's default absolute gap (1e-6 of the scaled objective) is overridden: the options
    passed to scipy.optimize.milp carry mip_abs_gap = 0, and the verbatim-option warning of
    scipy is suppressed (no warning escapes)."""
    opts = dict(MILP_OPTIONS)
    assert opts["mip_rel_gap"] == 0.0
    assert opts["mip_abs_gap"] == 0.0
    caps = [SiteCapacity(3, 5.0), SiteCapacity(3, INF)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = schedule_epoch(
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [5.0, 4.0, 5.0, 3.0],
            [0.0] * 4,
            caps,
            policy="optimal_assignment",
        )
    assert res.method == "milp"
    assert res.objective_W == pytest.approx(9.0, rel=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_greedy_is_optimal_without_power_or_density_limits_and_shared_satellites(
    seed: int,
) -> None:
    """Contention without a greedy trap: each satellite reaches exactly one site, sites have
    k beam slots and no power or density limit. The best-link set is infeasible (more
    satellites than slots), so both algorithms really run (greedy and the assignment), and
    greedy (the k largest per site) must equal the brute-force optimum."""
    rng = np.random.default_rng(500 + seed)
    n_site = int(rng.integers(1, 4))
    n_sat = int(rng.integers(4, 8))
    home = rng.integers(0, n_site, n_sat)
    home[: n_site + 1] = 0  # site 0 has more satellites than its slots
    links = {(i, int(home[i])): (float(rng.uniform(0.1, 10.0)), 0.0) for i in range(n_sat)}
    caps = [SiteCapacity(int(rng.integers(1, 3)), INF) for _ in range(n_site)]
    caps[0] = SiteCapacity(1, INF)
    _, sat, site, p, d = _arrays(links)
    ref = brute_force(links, caps)
    gre = schedule_epoch(sat, site, p, d, caps, policy="greedy_max_power")
    opt = schedule_epoch(sat, site, p, d, caps, policy="optimal_assignment")
    assert gre.method == "greedy"
    assert opt.method == "assignment"
    assert gre.objective_W == pytest.approx(ref, rel=1e-12)
    assert opt.objective_W == pytest.approx(ref, rel=1e-12)
    np.testing.assert_array_equal(gre.decision, opt.decision)
