"""Optimisation machinery (M8; PDF §12.3, §13.1 "optimization: synthetic known optimum, bounds
and repeatability").

Every reference value is independent of :mod:`helioswarm.optimization`:

* ZDT1 (Zitzler, Deb & Thiele 2000): Pareto front ``f2 = 1 - sqrt(f1)``, ``f1 in [0, 1]``; with
  the reference point (1.1, 1.1) its hypervolume is ``1.21 - int_0^1 (1 - sqrt x) dx = 1.21 -
  1/3`` (closed form).
* Binh & Korn (1997): the feasible front is computed by brute force on a dense grid here.
* Sphere: minimum 0 at the configured centre.
* Airy aperture/mass case: the optimum solves ``J1(x)/x = sqrt(c pi/(4 P0))/beta`` (derived in
  :func:`helioswarm.evaluation._airy_aperture_net_power`), found here with ``scipy.optimize.
  brentq`` on ``scipy.special.j1`` -- not with the optimiser or the project's Airy code.
* Hypervolume and non-domination are recomputed with plain loops (2-D rectangle sum).
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from scipy import optimize, special

from helioswarm.cli import main
from helioswarm.config import ConfigError
from helioswarm.evaluation import DesignResult, DesignTask, SyntheticModel, evaluate_parallel
from helioswarm.optimization import (
    FAILED_OBJECTIVE,
    ConstraintSpec,
    ObjectiveSpec,
    VariableSpec,
    archive_front,
    collate,
    converged,
    hypervolume,
    load_optimization,
    optimize_seed,
    pareto_mask,
    records_table,
    reference_point,
    run_sanity_check,
    run_seed,
    score,
    select_candidates,
)

pytest.importorskip("pymoo")

ZDT1_HV_EXACT = 1.21 - 1.0 / 3.0


def _write(tmp_path: Path, doc: dict[str, Any], name: str = "study.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def _zdt1(n: int = 5, pop: int = 40, gens: int = 40, seeds: list[int] | None = None) -> dict:
    return {
        "optimization_schema_version": 1,
        "name": "zdt1",
        "evaluator": "zdt1",
        "variables": [
            {"name": f"x{i}", "kind": "real", "key": f"x{i}", "lower": 0.0, "upper": 1.0}
            for i in range(n)
        ],
        "objectives": [
            {"metric": "f1", "sense": "min", "reference": 1.1},
            {"metric": "f2", "sense": "min", "reference": 1.1},
        ],
        "algorithm": {"pop_size": pop, "n_generations": gens, "seeds": seeds or [1, 2]},
        "convergence": {
            "window": 5,
            "hv_relative_tolerance": 0.01,
            "seed_hv_relative_tolerance": 0.03,
        },
    }


def _bnh() -> dict:
    return {
        "optimization_schema_version": 1,
        "name": "bnh",
        "evaluator": "bnh",
        "variables": [
            {"name": "x0", "kind": "real", "key": "x0", "lower": 0.0, "upper": 5.0},
            {"name": "x1", "kind": "real", "key": "x1", "lower": 0.0, "upper": 3.0},
        ],
        "objectives": [
            {"metric": "f1", "sense": "min", "reference": 140.0},
            {"metric": "f2", "sense": "min", "reference": 50.0},
        ],
        "constraints": [
            {"name": "c1", "metric": "c1", "op": "<=", "limit": 25.0},
            {"name": "c2", "metric": "c2", "op": ">=", "limit": 7.7},
        ],
        "algorithm": {"pop_size": 40, "n_generations": 40, "seeds": [3]},
    }


# ---------------------------------------------------------------------------------------------
# Variables, decoding, bounds
# ---------------------------------------------------------------------------------------------


def test_real_log_integer_categorical_decoding_respects_bounds() -> None:
    real = VariableSpec(name="a", kind="real", key="k", lower=2.0, upper=6.0)
    logv = VariableSpec(name="b", kind="real", key="k", lower=1e-9, upper=1e-6, log=True)
    integer = VariableSpec(name="c", kind="integer", key="k", lower=2, upper=5)
    cat = VariableSpec(
        name="d", kind="categorical", choices={"x": {"k": 1}, "y": {"k": 2}, "z": {}}
    )
    u = np.linspace(-0.2, 1.2, 1401)  # beyond [0, 1]: clipped
    assert real.decode(0.25) == pytest.approx(3.0, rel=0, abs=1e-15)
    assert logv.decode(0.5) == pytest.approx(math.sqrt(1e-9 * 1e-6), rel=1e-12)
    for var in (real, logv):
        vals = np.array([var.decode(x) for x in u])
        assert var.lower is not None
        assert var.upper is not None
        assert np.all(vals >= var.lower)
        assert np.all(vals <= var.upper)
    # Integer: equal-width bins on [0, 1]: each of 2..5 covers a quarter of the genome.
    grid = np.linspace(0.0, 1.0, 40001)
    ints = np.array([integer.decode(x) for x in grid])
    assert set(ints.tolist()) == {2, 3, 4, 5}
    counts = np.array([np.sum(ints == k) for k in (2, 3, 4, 5)]) / grid.size
    assert np.allclose(counts, 0.25, atol=1e-3)  # bin width 1/4 sampled on a 1/40000 grid
    cats = [cat.decode(x) for x in grid]
    assert {c: cats.count(c) / grid.size for c in "xyz"} == pytest.approx(
        {"x": 1 / 3, "y": 1 / 3, "z": 1 / 3}, abs=1e-3
    )
    # encode() lands in the centre of the decoding cell (round trip)
    for var, val in ((real, 4.5), (logv, 3e-8), (integer, 4), (cat, "y")):
        back = var.decode(var.encode(val))
        assert back == (val if var.kind == "categorical" else pytest.approx(val, rel=1e-12))
    assert cat.overrides("x") == [("k", 1)]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"name": "a", "kind": "real", "key": "k", "lower": 1.0, "upper": 1.0}, "lower < upper"),
        (
            {"name": "a", "kind": "real", "key": "k", "lower": 0.0, "upper": 1.0, "log": True},
            "lower > 0",
        ),
        ({"name": "a", "kind": "integer", "key": "k", "lower": 0.5, "upper": 3}, "integral"),
        ({"name": "a", "kind": "categorical", "choices": {"x": {}}}, ">= 2 choices"),
        ({"name": "a", "kind": "real", "lower": 0.0, "upper": 1.0}, "needs key"),
    ],
)
def test_invalid_variables_are_rejected(spec: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VariableSpec.model_validate(spec)


# ---------------------------------------------------------------------------------------------
# Scoring, Pareto, hypervolume
# ---------------------------------------------------------------------------------------------


def test_score_conventions_failed_and_undefined_designs_are_infeasible() -> None:
    objs = (
        ObjectiveSpec(metric="p", sense="max", reference=0.0),
        ObjectiveSpec(metric="m", sense="min", reference=10.0),
    )
    cons = (
        ConstraintSpec(name="hi", metric="t", op="<=", limit=400.0),
        ConstraintSpec(name="lo", metric="a", op=">=", limit=0.5, scale=0.1),
    )
    ok = DesignResult(0, "a", "ok", {"p": 3.0, "m": 2.0, "t": 410.0, "a": 0.6})
    sc = score(ok, objs, cons)
    assert sc.F.tolist() == [-3.0, 2.0]  # maximise -> negated
    assert sc.G[0] == pytest.approx(10.0 / 400.0)  # (v - limit)/|limit|
    assert sc.G[1] == pytest.approx(-1.0)  # (limit - v)/scale
    assert sc.G[2] == -1.0  # evaluation ok
    assert not sc.feasible
    assert sc.cv == pytest.approx(0.025)
    fail = DesignResult(1, "b", "failed", {}, error_type="BeamModelValidityError", error="x")
    sf = score(fail, objs, cons)
    assert not sf.feasible
    assert sf.G[-1] == 1.0
    assert np.all(sf.F == FAILED_OBJECTIVE)
    undef = DesignResult(2, "c", "ok", {"p": math.nan, "m": 2.0, "t": 1.0, "a": 1.0})
    su = score(undef, objs, cons)
    assert not su.feasible
    assert su.undefined == ("p",)


def _pareto_bruteforce(F: np.ndarray) -> np.ndarray:
    keep = []
    for i in range(len(F)):
        dominated = False
        for j in range(len(F)):
            if all(F[j] <= F[i]) and any(F[j] < F[i]):
                dominated = True
                break
        keep.append(not dominated)
    return np.array(keep)


def _hv2d(F: np.ndarray, ref: np.ndarray) -> float:
    pts = sorted(p for p in map(tuple, F) if p[0] < ref[0] and p[1] < ref[1])
    hv, best_f2 = 0.0, ref[1]
    for f1, f2 in pts:  # ascending f1; each non-dominated point adds a rectangle
        if f2 < best_f2:
            hv += (ref[0] - f1) * (best_f2 - f2)
            best_f2 = f2
    return hv


def test_pareto_mask_and_hypervolume_match_independent_loops() -> None:
    rng = np.random.default_rng(5)
    F = rng.random((300, 2))
    F[:10] = F[10:20]  # duplicates are kept together
    assert np.array_equal(pareto_mask(F), _pareto_bruteforce(F))
    F3 = rng.random((150, 3))
    assert np.array_equal(pareto_mask(F3), _pareto_bruteforce(F3))
    ref = np.array([1.05, 1.1])
    assert hypervolume(F, ref) == pytest.approx(_hv2d(F, ref), rel=1e-12)
    assert hypervolume(np.array([[2.0, 0.0]]), ref) == 0.0  # outside the reference box


def test_hypervolume_reference_of_a_max_objective_with_nonzero_reference() -> None:
    # Objective a is maximised with reference 10, b minimised with reference 5. In natural
    # units two designs (a, b) = (15, 4) and (12, 3) dominate the rectangles [10, 15] x [4, 5]
    # (area 5) and [10, 12] x [3, 5] (area 4), overlapping on [10, 12] x [4, 5] (area 2):
    # union 5 + 4 - 2 = 7 (inclusion-exclusion by hand).
    objectives = [
        ObjectiveSpec(metric="a", sense="max", reference=10.0),
        ObjectiveSpec(metric="b", sense="min", reference=5.0),
    ]
    ref = reference_point(objectives)
    assert ref.tolist() == [-10.0, 5.0]
    natural = [(15.0, 4.0), (12.0, 3.0)]
    F = np.array([[o.to_min(v) for o, v in zip(objectives, row, strict=True)] for row in natural])
    assert hypervolume(F, ref) == pytest.approx(7.0, rel=1e-12)
    # a design worse than the reference in the maximised objective contributes nothing
    worse = np.vstack([F, [[-9.0, 0.0]]])  # a = 9 < 10
    assert hypervolume(worse, ref) == pytest.approx(7.0, rel=1e-12)


def test_converged_rule_and_candidate_selection() -> None:
    assert converged([1.0, 2.0, 2.0, 2.0], window=2, tolerance=1e-9)
    assert not converged([1.0, 2.0, 3.0], window=2, tolerance=0.1)
    assert not converged([1.0], window=2, tolerance=1.0)
    # The rule compares generation G with G - window, not G - window + 1: over [1, 1, 2, 2]
    # the last 2 generations changed HV by 50 % (2 vs 1), the last 1 by 0 %.
    assert not converged([1.0, 1.0, 2.0, 2.0], window=2, tolerance=0.01)
    assert converged([1.0, 1.0, 2.0, 2.0], window=1, tolerance=0.01)
    assert not converged([1.0, 2.0, 2.0], window=3, tolerance=1.0)  # needs window + 1 values
    F = np.array([[0.0, 1.0], [0.5, 0.4], [1.0, 0.0], [0.2, 0.8], [0.8, 0.1]])
    picks = select_candidates(F, 3)
    assert picks[:2] == [0, 2]  # best of each objective first
    assert len(set(picks)) == 3
    assert picks == select_candidates(F, 3)


# ---------------------------------------------------------------------------------------------
# NSGA-II on synthetic problems with known solutions
# ---------------------------------------------------------------------------------------------


def test_nsga2_recovers_the_zdt1_front_and_seeds_agree(tmp_path: Path) -> None:
    problem = load_optimization(_write(tmp_path, _zdt1()))
    out = tmp_path / "out"
    for seed in (1, 2):
        optimize_seed(problem, out, seed)
    c = collate(problem, out)
    f1 = np.array(c.union_front.column("m_f1").to_pylist())
    f2 = np.array(c.union_front.column("m_f2").to_pylist())
    # Front within 0.02 of 1 - sqrt(f1) (40 x 40 evaluations of a 5-variable ZDT1; the measured
    # worst gap is ~0.01; a wrong objective sign or decoding is off by O(0.1-1)).
    assert np.max(f2 - (1.0 - np.sqrt(f1))) < 0.02
    assert np.min(f2 - (1.0 - np.sqrt(f1))) > -1e-12  # nothing better than the true front
    assert f1.min() < 0.02
    assert f1.max() > 0.95
    for s in (1, 2):  # each seed's HV within 2 % of the exact value (and never above it)
        assert c.hv_final[s] == pytest.approx(ZDT1_HV_EXACT, rel=0.02)
        assert c.hv_final[s] <= ZDT1_HV_EXACT + 1e-12
    assert c.seeds_agree
    assert c.seed_spread < 0.03
    summary = json.loads((c.path / "summary.json").read_text())
    assert summary["missing_seeds"] == []
    assert (c.path / "pareto.png").is_file()


def test_same_seed_is_bit_identical_and_independent_of_worker_count(tmp_path: Path) -> None:
    problem = load_optimization(_write(tmp_path, _zdt1(pop=12, gens=4, seeds=[7, 8])))
    a = run_seed(problem, 7, workers=1)
    b = run_seed(problem, 7, workers=1)
    c = run_seed(problem, 7, workers=2)  # spawn pool; chunked evaluation
    other = run_seed(problem, 8, workers=1)
    cols = [n for n in records_table(problem, a.records, 7).column_names if n != "wall_s"]
    ta = records_table(problem, a.records, 7).select(cols)
    assert ta.equals(records_table(problem, b.records, 7).select(cols))
    assert ta.equals(records_table(problem, c.records, 7).select(cols))
    assert a.final_population == c.final_population
    assert not ta.equals(records_table(problem, other.records, 7).select(cols))


def test_constrained_bnh_front_is_feasible_and_matches_brute_force(tmp_path: Path) -> None:
    problem = load_optimization(_write(tmp_path, _bnh()))
    run = run_seed(problem, 3)
    front = archive_front(run.records)
    assert front
    infeasible = [r for r in run.records if not r.scored.feasible]
    assert infeasible, "the search must have met the constraint boundary"
    keys = {r.design.key for r in front}
    assert not keys & {r.design.key for r in infeasible}
    for r in front:  # every reported member satisfies both constraints
        assert r.result.metrics["c1"] <= 25.0 + 1e-12
        assert r.result.metrics["c2"] >= 7.7 - 1e-12
    # Brute-force feasible front on a 501 x 301 grid (independent of the optimiser).
    x0, x1 = np.meshgrid(np.linspace(0, 5, 501), np.linspace(0, 3, 301))
    x0, x1 = x0.ravel(), x1.ravel()
    ok = ((x0 - 5) ** 2 + x1**2 <= 25) & ((x0 - 8) ** 2 + (x1 + 3) ** 2 >= 7.7)
    Fb = np.column_stack([4 * x0**2 + 4 * x1**2, (x0 - 5) ** 2 + (x1 - 5) ** 2])[ok]
    ref = np.array([140.0, 50.0])
    hv_ref = _hv2d(Fb, ref)  # the rectangle sweep ignores dominated points
    hv_opt = _hv2d(np.array([r.scored.F for r in front]), ref)
    # 1600 evaluations: HV within 2 % of the dense-grid front (grid step 0.01 limits the
    # reference itself to ~0.1 %).
    assert hv_opt == pytest.approx(hv_ref, rel=0.02)


def test_single_objective_sphere_optimum_and_bounds(tmp_path: Path) -> None:
    doc = {
        "optimization_schema_version": 1,
        "name": "sphere",
        "evaluator": "sphere",
        "synthetic_parameters": {"c0": 0.3, "c1": -1.2},
        "variables": [
            {"name": "x0", "kind": "real", "key": "x0", "lower": -2.0, "upper": 2.0},
            {"name": "x1", "kind": "real", "key": "x1", "lower": -2.0, "upper": 2.0},
        ],
        "objectives": [{"metric": "f", "sense": "min", "reference": 10.0}],
        "algorithm": {"pop_size": 20, "n_generations": 40, "seeds": [11]},
    }
    problem = load_optimization(_write(tmp_path, doc))
    run = run_seed(problem, 11)
    best = min(run.records, key=lambda r: r.result.metrics["f"])
    # 800 evaluations of a 2-D quadratic: the best design is within 0.02 of (0.3, -1.2).
    assert best.design.values["x0"] == pytest.approx(0.3, abs=0.02)
    assert best.design.values["x1"] == pytest.approx(-1.2, abs=0.02)
    for r in run.records:  # bounds respected by every evaluated design
        assert all(-2.0 <= v <= 2.0 for v in r.design.values.values())


def _airy_closed_form(p: dict[str, float]) -> float:
    beta = math.pi * p["receiver_radius_m"] / (p["wavelength_m"] * p["range_m"])
    rhs = math.sqrt(p["penalty_W_per_m2"] * math.pi / (4.0 * p["beam_power_W"])) / beta
    x = optimize.brentq(lambda x: special.j1(x) / x - rhs, 1e-6, 3.8317, xtol=1e-14)
    return x / beta


AIRY = {
    "beam_power_W": 1.0e5,
    "wavelength_m": 1.0e-6,
    "range_m": 1.0e11,
    "receiver_radius_m": 1000.0,
    "penalty_W_per_m2": 11.3,
}


def test_physics_case_scipy_and_nsga2_recover_the_closed_form_aperture(tmp_path: Path) -> None:
    d_star = _airy_closed_form(AIRY)
    assert 40.0 < d_star < 120.0  # interior optimum inside the bounds (precondition)
    doc = {
        "optimization_schema_version": 1,
        "name": "airy",
        "evaluator": "airy_aperture_net_power",
        "synthetic_parameters": AIRY,
        "variables": [
            {
                "name": "aperture",
                "kind": "real",
                "key": "aperture_diameter_m",
                "lower": 1.0,
                "upper": 200.0,
                "log": True,
            }
        ],
        "objectives": [{"metric": "net_power_W", "sense": "max", "reference": 0.0}],
        "algorithm": {"pop_size": 12, "n_generations": 25, "seeds": [4]},
        "sanity_checks": [
            {
                "name": "aperture",
                "variable": "aperture",
                "metric": "net_power_W",
                "sense": "max",
                "sweep_points": 41,
                "xatol": 1e-7,
            }
        ],
    }
    problem = load_optimization(_write(tmp_path, doc))
    res = run_sanity_check(problem, problem.definition.sanity_checks[0])
    # scipy bounded Brent on the log-coordinate, xatol 1e-7 in u (log(200) * 1e-7 relative in D)
    assert res.scipy_value == pytest.approx(d_star, rel=1e-5)
    assert res.agree
    run = run_seed(problem, 4)
    best = max(run.records, key=lambda r: r.result.metrics["net_power_W"])
    # 300 GA evaluations of a smooth 1-D function: within 1 % of the optimum location.
    assert best.design.values["aperture"] == pytest.approx(d_star, rel=0.01)


def test_sanity_check_with_penalty_metric_recovers_the_closed_form(tmp_path: Path) -> None:
    # J = captured_power_W - w * penalty_W with w = 1 is the net power of the closed-form case;
    # with w = 2 the areal penalty doubles. Without the penalty the captured power is
    # non-decreasing in D (dEE/dx = 2 J1^2/x >= 0), so ignoring penalty_metric would put the
    # optimum at the 200 m bound.
    for weight in (1.0, 2.0):
        d_star = _airy_closed_form(AIRY | {"penalty_W_per_m2": weight * AIRY["penalty_W_per_m2"]})
        assert 20.0 < d_star < 120.0  # interior optimum (precondition)
        doc = {
            "optimization_schema_version": 1,
            "name": "airy_penalty",
            "evaluator": "airy_aperture_net_power",
            "synthetic_parameters": AIRY,
            "variables": [
                {
                    "name": "aperture",
                    "kind": "real",
                    "key": "aperture_diameter_m",
                    "lower": 1.0,
                    "upper": 200.0,
                    "log": True,
                }
            ],
            "objectives": [{"metric": "net_power_W", "sense": "max", "reference": 0.0}],
            "algorithm": {"pop_size": 4, "n_generations": 1, "seeds": [1]},
            "sanity_checks": [
                {
                    "name": "aperture",
                    "variable": "aperture",
                    "metric": "captured_power_W",
                    "sense": "max",
                    "penalty_metric": "penalty_W",
                    "penalty_weight": weight,
                    "sweep_points": 41,
                    "xatol": 1e-7,
                }
            ],
        }
        problem = load_optimization(_write(tmp_path, doc, f"pen_{weight}.yaml"))
        res = run_sanity_check(problem, problem.definition.sanity_checks[0])
        # same Brent tolerance as the net-power case above
        assert res.scipy_value == pytest.approx(d_star, rel=1e-5)
        assert res.agree


def test_parallel_evaluation_preserves_order_and_results() -> None:
    model = SyntheticModel("sphere", ("x0", "x1"), {"c0": 1.0})
    tasks = [DesignTask(i, f"t{i}", (("x0", i * 0.1), ("x1", -i * 0.2))) for i in range(9)]
    seq = evaluate_parallel(model, tasks, workers=1)
    par = evaluate_parallel(model, tasks, workers=3)
    assert [r.index for r in par] == list(range(9))
    assert [r.metrics for r in par] == [r.metrics for r in seq]


# ---------------------------------------------------------------------------------------------
# Study file validation and CLI
# ---------------------------------------------------------------------------------------------


def test_study_file_errors(tmp_path: Path) -> None:
    bad = _zdt1()
    bad["objectives"][0]["metric"] = "nope"
    with pytest.raises(ConfigError, match="unknown metric"):
        load_optimization(_write(tmp_path, bad, "a.yaml"))
    bad = _zdt1()
    bad["variables"][1]["name"] = "x0"
    with pytest.raises(ConfigError, match="unique"):
        load_optimization(_write(tmp_path, bad, "b.yaml"))
    bad = _zdt1()
    bad["variables"][0]["key"] = "x99"
    with pytest.raises(ConfigError, match="unknown argument"):
        load_optimization(_write(tmp_path, bad, "c.yaml"))
    bad = _zdt1()
    bad["algorithm"]["seeds"] = [1, 1]
    with pytest.raises(ConfigError, match="unique"):
        load_optimization(_write(tmp_path, bad, "d.yaml"))
    # a design variable may not also be set (or removed) by a fidelity level
    bad = _zdt1()
    bad["fidelity"] = {"high": {"description": "clash", "overrides": {"x1": 0.5}}}
    with pytest.raises(ConfigError, match="also set by fidelity level 'high'"):
        load_optimization(_write(tmp_path, bad, "e.yaml"))
    bad = _zdt1()
    bad["fidelity"] = {"high": {"description": "clash", "unset": ["x2"]}}
    with pytest.raises(ConfigError, match="also set by fidelity level 'high'"):
        load_optimization(_write(tmp_path, bad, "f.yaml"))


def test_cli_seed_collate_and_sanity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = _zdt1(n=3, pop=8, gens=3, seeds=[5, 6])
    doc["sanity_checks"] = [
        {"name": "f1_check", "variable": "x0", "metric": "f1", "sense": "min", "sweep_points": 5}
    ]
    study = _write(tmp_path, doc)
    out = tmp_path / "opt"
    assert main(["optimize", str(study), "--out", str(out), "--array-index", "1"]) == 0
    assert (out / "seed_6" / "run.json").is_file()
    assert not (out / "seed_5").exists()
    # collation with a missing seed: listed, exit 1
    assert main(["optimize", str(study), "--out", str(out), "--collate"]) == 1
    assert main(["optimize", str(study), "--out", str(out), "--seed", "5"]) == 0
    assert main(["optimize", str(study), "--out", str(out), "--collate"]) == 0
    s = json.loads((out / "collated" / "summary.json").read_text())
    assert s["seeds"] == [5, 6]
    assert s["missing_seeds"] == []
    # a finished seed is not recomputed silently
    assert main(["optimize", str(study), "--out", str(out), "--seed", "5"]) == 1
    # f1 = x0 minimised over x0 in [0, 1]: optimum at the bound 0
    assert main(["optimize", str(study), "--out", str(out), "--sanity"]) == 0
    chk = json.loads((out / "sanity" / "f1_check.json").read_text())
    assert chk["agree"]
    assert chk["scipy"]["value"] == pytest.approx(0.0, abs=1e-3)
    ev = pq.read_table(out / "seed_5" / "evaluations.parquet")
    assert ev.num_rows >= 8
    assert set(ev.column("status").to_pylist()) == {"ok"}
    # a different study in the same directory is refused
    other = _write(tmp_path, _zdt1(n=4, pop=8, gens=2, seeds=[5]), "other.yaml")
    err = io.StringIO()
    assert main(["optimize", str(other), "--out", str(out), "--seed", "5"]) == 1
    del err
