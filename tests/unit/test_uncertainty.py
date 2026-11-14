"""Uncertainty machinery (M8; PDF §12.4): distributions, seeded nested Monte Carlo, epistemic /
aleatory bookkeeping, Sobol indices, array-split invariance, scenario consistency.

Reference values are closed forms stated in each test (Ishigami indices: Ishigami & Homma 1990
with the variance decomposition of Sobol' 2001, e.g. Saltelli et al. 2008 "Global Sensitivity
Analysis: The Primer" §4.4; linear-model variances: Var(sum w_i x_i) = sum w_i^2 Var x_i).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from scipy import stats

from helioswarm.cli import main
from helioswarm.config import ConfigError
from helioswarm.uncertainty import (
    Distribution,
    collate,
    load_uncertainty,
    monte_carlo_samples,
    nested_statistics,
    read_part,
    run_part,
    sobol_indices,
    sobol_samples,
)

pytest.importorskip("SALib")

PI = math.pi


def _write(tmp_path: Path, doc: dict[str, Any], name: str = "unc.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def _ishigami(n_base: int = 4096) -> dict[str, Any]:
    conv = [n for n in (256, 1024) if n < n_base]
    u = {"type": "uniform", "low": -PI, "high": PI}
    return {
        "uncertainty_schema_version": 1,
        "name": "ishigami",
        "evaluator": "ishigami",
        "outputs": ["y"],
        "epistemic": [
            {"name": "x0", "key": "x0", "provenance": "assumed", "distribution": u},
            {"name": "x1", "key": "x1", "provenance": "assumed", "distribution": u},
        ],
        "aleatory": [{"name": "x2", "key": "x2", "provenance": "assumed", "distribution": u}],
        "monte_carlo": {"seed": 3, "n_epistemic": 20, "n_aleatory": 5},
        "sobol": {
            "seed": 9,
            "n_base": n_base,
            "parameters": ["x0", "x1", "x2"],
            "num_resamples": 100,
            "convergence_n": conv,
        },
    }


def _linear(n_e: int, n_a: int, seed: int = 1) -> dict[str, Any]:
    return {
        "uncertainty_schema_version": 1,
        "name": "linear",
        "evaluator": "linear",
        "outputs": ["y"],
        "epistemic": [
            {
                "name": "x0",
                "key": "x0",
                "provenance": "assumed",
                "distribution": {"type": "uniform", "low": 0.0, "high": 1.0},
            }
        ],
        "aleatory": [
            {
                "name": "x1",
                "key": "x1",
                "provenance": "assumed",
                "distribution": {"type": "uniform", "low": 0.0, "high": 2.0},
            }
        ],
        "monte_carlo": {"seed": seed, "n_epistemic": n_e, "n_aleatory": n_a},
    }


# ---------------------------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------------------------


def test_distribution_ppf_against_closed_forms() -> None:
    u = np.array([0.0, 0.1, 0.5, 0.9, 0.999999])
    tri = Distribution(type="triangular", low=1.0, mode=2.0, high=4.0)
    # Triangular inverse CDF: F(m) = (m-a)/(b-a) = 1/3; below it x = a + sqrt(u (b-a)(m-a)),
    # above it x = b - sqrt((1-u)(b-a)(b-m)).
    exp = [1.0 + math.sqrt(q * 3 * 1) if q <= 1 / 3 else 4 - math.sqrt((1 - q) * 3 * 2) for q in u]
    assert tri.ppf(u) == pytest.approx(exp, rel=1e-12)
    ray = Distribution(type="rayleigh", scale=2.0)
    assert ray.ppf(np.array([0.5]))[0] == pytest.approx(2.0 * math.sqrt(2 * math.log(2)), rel=1e-12)
    uni = Distribution(type="uniform", low=-1.0, high=3.0)
    assert uni.ppf(u) == pytest.approx(-1.0 + 4.0 * u)
    tn = Distribution(type="normal", mean=0.0, sd=1.0, low=-0.5, high=2.0)
    v = tn.ppf(np.linspace(0, 1, 101)[:-1])
    assert v.min() >= -0.5
    assert v.max() <= 2.0
    # Truncated N(0,1) on [a, b]: F^-1(u) = Phi^-1(Phi(a) + u (Phi(b) - Phi(a))) (closed form
    # with the untruncated normal; a clipped normal would give 0 at u = 0.5, not 0.366).
    pa_, pb_ = stats.norm.cdf(-0.5), stats.norm.cdf(2.0)
    q = np.array([0.1, 0.5, 0.9])
    assert tn.ppf(q) == pytest.approx(stats.norm.ppf(pa_ + q * (pb_ - pa_)), rel=1e-9)
    assert tn.ppf(np.array([0.5]))[0] == pytest.approx(0.3664, abs=1e-3)
    ln = Distribution(type="lognormal", median=3.0, sigma_log=0.2)
    assert ln.ppf(np.array([0.5]))[0] == pytest.approx(3.0)
    assert ln.ppf(np.array([stats.norm.cdf(1.0)]))[0] == pytest.approx(3.0 * math.exp(0.2))
    du = Distribution(type="discrete_uniform", values=(2014, 2015, 2016, 2017))
    grid = np.linspace(0, 1, 4000, endpoint=False)
    vals = du.ppf(grid).tolist()
    assert {y: vals.count(y) for y in (2014, 2015, 2016, 2017)} == dict.fromkeys(
        (2014, 2015, 2016, 2017), 1000
    )
    with pytest.raises(ValueError, match="low <= mode <= high"):
        Distribution(type="triangular", low=0.0, mode=2.0, high=1.0)
    with pytest.raises(ValueError, match="needs"):
        Distribution(type="rayleigh")


# ---------------------------------------------------------------------------------------------
# Nested Monte Carlo: seeding and bookkeeping
# ---------------------------------------------------------------------------------------------


def test_monte_carlo_is_seeded_and_streams_are_separate(tmp_path: Path) -> None:
    a = monte_carlo_samples(load_uncertainty(_write(tmp_path, _linear(5, 4), "a.yaml")))
    b = monte_carlo_samples(load_uncertainty(_write(tmp_path, _linear(5, 4), "b.yaml")))
    c = monte_carlo_samples(load_uncertainty(_write(tmp_path, _linear(5, 4, seed=2), "c.yaml")))
    d = monte_carlo_samples(load_uncertainty(_write(tmp_path, _linear(5, 9), "d.yaml")))
    assert [s.values for s in a] == [s.values for s in b]  # same seed -> identical design
    assert [s.values for s in a] != [s.values for s in c]
    # Bookkeeping: task i N_a + j; the epistemic value is constant within an epistemic index,
    # the aleatory value varies within it.
    assert [(s.index, s.epistemic_index, s.aleatory_index) for s in a[:5]] == [
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 2),
        (3, 0, 3),
        (4, 1, 0),
    ]
    for i in range(5):
        rows = [s for s in a if s.epistemic_index == i]
        assert len({s.values["x0"] for s in rows}) == 1
        assert len({s.values["x1"] for s in rows}) == 4
    # Separate streams: adding an epistemic parameter leaves the aleatory draws unchanged, and
    # changing the number of aleatory samples leaves the epistemic draws unchanged.
    doc = _linear(5, 4)
    doc["epistemic"].append(
        {
            "name": "x2",
            "key": "x2",
            "provenance": "assumed",
            "distribution": {"type": "uniform", "low": 0.0, "high": 1.0},
        }
    )
    e = monte_carlo_samples(load_uncertainty(_write(tmp_path, doc, "e.yaml")))
    assert [s.values["x1"] for s in e] == [s.values["x1"] for s in a]
    assert [s.values["x0"] for s in a if s.aleatory_index == 0] == [
        s.values["x0"] for s in d if s.aleatory_index == 0
    ]


def test_law_of_total_variance_split_epistemic_vs_aleatory(tmp_path: Path) -> None:
    study = load_uncertainty(_write(tmp_path, _linear(400, 50)))
    run_part(study, tmp_path / "out", "mc")
    c = collate(study, tmp_path / "out")
    v = c.monte_carlo["y"].variance
    assert v is not None
    assert c.summary["monte_carlo"]["y"]["variance"] == c.monte_carlo["y"].as_dict()["variance"]
    # y = x0 + x1, x0 ~ U(0,1) epistemic, x1 ~ U(0,2) aleatory: Var x0 = 1/12, Var x1 = 4/12.
    assert v.total_population == pytest.approx(
        v.within_row_population + v.between_row_population, rel=1e-12
    )  # exact identity for equal-size rows
    assert v.total_population == pytest.approx(5 / 12, rel=0.05)  # 20 000 samples
    # The uncorrected between-row variance estimates Var x0 + Var(mean of 50 x1)
    # = 1/12 + 4/12/50; 400 rows: relative SE of a sample variance ~ sqrt(2/399) = 7 %,
    # 3 sigma = 21 %.
    assert v.between_row_population == pytest.approx(1 / 12 + (4 / 12) / 50, rel=0.21)
    # The unbiased components: epistemic Var x0 = 1/12 (same 21 % plus the subtracted noise
    # term's own scatter, < 1 %), aleatory Var x1 = 4/12 (20 000 samples, 5 %).
    assert v.epistemic == pytest.approx(1 / 12, rel=0.22)
    assert v.aleatory == pytest.approx(4 / 12, rel=0.05)
    assert v.epistemic_fraction == pytest.approx(0.2, rel=0.22)  # (1/12) / (5/12)
    assert v.aleatory_noise_sd_of_row_mean == pytest.approx(math.sqrt(4 / 12 / 50), rel=0.03)
    e = c.monte_carlo["y"].epistemic_interval_of_expected_value
    assert e is not None
    # E[y | x0] = x0 + 1: its p05/p95 are 1.05 / 1.95 up to the 50-sample aleatory noise (sd 0.08)
    assert e.p05 == pytest.approx(1.05, abs=0.06)
    assert e.p95 == pytest.approx(1.95, abs=0.06)


def test_epistemic_share_is_unbiased_at_small_aleatory_sample_size() -> None:
    """The row means average only N_a aleatory samples, so Var_i(m_i) contains sigma_a^2/N_a
    of aleatory noise; the reported epistemic share must not (reviewer finding M8)."""
    rng = np.random.default_rng(20261114)
    # (1) Y does not depend on the epistemic index: true epistemic variance 0. Averaged over
    # 200 replicate (64, 8) studies the uncorrected ratio is 1/N_a = 0.125 (what a biased
    # estimator reports); the unclipped corrected estimate averages to 0: its replicate sd is
    # ~ sqrt(2/63) sigma^2/8 = 0.0019 for sigma^2 = 1/12, so the mean of 200 is within
    # 3 * 0.0019/sqrt(200) = 4e-4 of 0.
    raw, upper, frac = [], [], []
    for _ in range(200):
        v = nested_statistics(rng.random((64, 8))).variance
        assert v is not None
        raw.append(v.epistemic_unclipped)
        upper.append(v.epistemic_fraction_upper_bound)
        frac.append(v.epistemic_fraction)
    assert float(np.mean(upper)) == pytest.approx(1 / 8, abs=0.01)
    assert abs(float(np.mean(raw))) < 4e-4
    assert float(np.median(frac)) < 0.02  # clipped at 0 about half of the time
    # One large study at N_a = 4: Var_i(m_i) has relative SE sqrt(2/2047) = 3 % of
    # sigma^2/4, so the corrected share is < 4 sigma = 0.031 while the uncorrected ratio
    # is 1/4.
    v = nested_statistics(rng.random((2048, 4))).variance
    assert v is not None
    assert v.epistemic_fraction < 0.031
    assert v.epistemic_fraction_upper_bound == pytest.approx(0.25, abs=0.02)
    assert v.aleatory == pytest.approx(1 / 12, rel=0.03)
    assert v.aleatory_noise_sd_of_row_mean == pytest.approx(math.sqrt(1 / 12 / 4), rel=0.02)
    # (2) Y depends only on the epistemic input: no within-row variance, share exactly 1.
    theta = rng.random(64)
    v = nested_statistics(np.repeat(theta[:, None], 4, axis=1)).variance
    assert v is not None
    assert v.aleatory == 0.0
    assert v.epistemic == pytest.approx(float(np.var(theta, ddof=1)), rel=1e-12)
    assert v.epistemic_fraction == 1.0
    # (3) Mixed at N_a = 4: Y = theta + xi, Var theta = 1/12, Var xi = 4/12. The unbiased
    # epistemic estimate is 1/12 (relative SE ~ sqrt(2/2047) (1/12 + 1/12)/(1/12) = 6 %,
    # 3 sigma = 19 %); the uncorrected one would be 1/12 + (4/12)/4 = 2/12.
    Y = rng.random((2048, 1)) + 2.0 * rng.random((2048, 4))
    v = nested_statistics(Y).variance
    assert v is not None
    assert v.epistemic == pytest.approx(1 / 12, rel=0.19)
    assert v.epistemic_fraction == pytest.approx(0.2, rel=0.19)
    # (4) N_a = 1: the split is not identifiable (no within-row variance estimate).
    v = nested_statistics(rng.random((10, 1))).variance
    assert v is not None
    assert math.isnan(v.epistemic_fraction)


def test_nested_statistics_excludes_rows_with_failures() -> None:
    Y = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    s = nested_statistics(Y)
    assert s.n_nonfinite == 1
    assert s.epistemic_rows_used == 2
    assert s.epistemic_interval_of_expected_value is not None
    assert s.epistemic_interval_of_expected_value.mean == pytest.approx(3.5)
    assert s.pooled.n == 5
    # rows (1, 2) and (5, 6): s_i^2 = 0.5 each; Var(m = 1.5, 5.5; ddof=1) = 8;
    # epistemic = 8 - 0.5/2 = 7.75, fraction 7.75 / 8.25.
    assert s.variance is not None
    assert s.variance.aleatory == pytest.approx(0.5)
    assert s.variance.epistemic == pytest.approx(7.75)
    assert s.variance.epistemic_fraction == pytest.approx(7.75 / 8.25)
    assert nested_statistics(np.full((2, 2), np.nan)).variance is None


# ---------------------------------------------------------------------------------------------
# Sobol indices on Ishigami
# ---------------------------------------------------------------------------------------------


def _ishigami_indices(a: float = 7.0, b: float = 0.1) -> tuple[list[float], list[float]]:
    v1 = 0.5 * (1 + b * PI**4 / 5) ** 2
    v2 = a**2 / 8
    v13 = b**2 * PI**8 * (1 / 18 - 1 / 50)
    v = v1 + v2 + v13
    return [v1 / v, v2 / v, 0.0], [(v1 + v13) / v, v2 / v, v13 / v]


def test_sobol_indices_of_ishigami_and_convergence(tmp_path: Path) -> None:
    s1_true, st_true = _ishigami_indices()
    assert s1_true == pytest.approx([0.3139, 0.4424, 0.0], abs=1e-4)  # literature values
    study = load_uncertainty(_write(tmp_path, _ishigami()))
    run_part(study, tmp_path / "out", "sobol")
    c = collate(study, tmp_path / "out")
    r = c.summary["sobol"]["y"]
    assert r["status"] == "ok"
    ind = r["indices"]
    names = ["x0", "x1", "x2"]
    # N = 4096 (20 480 evaluations): the Saltelli/Jansen estimators' standard errors are
    # ~0.01-0.02 here (the bootstrap half-widths below); 0.04 is > 2 of them.
    for k, n in enumerate(names):
        assert ind["S1"][n] == pytest.approx(s1_true[k], abs=0.04)
        assert ind["ST"][n] == pytest.approx(st_true[k], abs=0.04)
        assert 0.0 < ind["ST_conf"][n] < 0.05
    assert set(r["convergence"]) == {"256", "1024", "4096"}
    # The 4096 entry is the full design (identical to the reported indices).
    assert r["convergence"]["4096"] == ind
    # Coarser prefixes are farther from the truth on the interaction term ST(x2) in this seed's
    # design (not a law, but the half-widths shrink with N):
    assert r["convergence"]["256"]["ST_conf"]["x2"] > ind["ST_conf"]["x2"]


def test_sobol_design_pins_other_parameters_to_nominal(tmp_path: Path) -> None:
    doc = _ishigami(n_base=8)
    doc["sobol"]["parameters"] = ["x0", "x2"]
    doc["sobol"]["convergence_n"] = []
    doc["epistemic"][1]["nominal"] = 0.25
    study = load_uncertainty(_write(tmp_path, doc))
    samples = sobol_samples(study)
    assert len(samples) == 8 * (2 + 2)
    assert {s.values["x1"] for s in samples} == {0.25}
    assert len({s.values["x0"] for s in samples}) > 4


# ---------------------------------------------------------------------------------------------
# Array split invariance, missing parts, CLI
# ---------------------------------------------------------------------------------------------


def test_array_split_gives_the_same_tables_and_missing_parts_are_reported(tmp_path: Path) -> None:
    study = load_uncertainty(_write(tmp_path, _ishigami(n_base=64)))
    one, three = tmp_path / "one", tmp_path / "three"
    for part in ("mc", "sobol"):
        run_part(study, one, part)
        for i in range(3):
            run_part(study, three, part, array_index=i, array_count=3)
    for part in ("mc", "sobol"):
        a, b = read_part(study, one, part), read_part(study, three, part)
        assert a.drop(["wall_s"]).equals(b.drop(["wall_s"]))
    (three / "sobol" / "part_0001_of_0003.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        read_part(study, three, "sobol")
    c = collate(study, three)
    assert c.summary["parts"]["sobol"]["status"] == "missing"
    assert c.summary["parts"]["mc"]["status"] == "complete"
    with pytest.raises(FileExistsError):
        run_part(study, one, "mc")  # no silent overwrite


def test_cli_uncertainty_end_to_end(tmp_path: Path) -> None:
    study = _write(tmp_path, _ishigami(n_base=32) | {"workers": 1})
    out = tmp_path / "u"
    assert main(["uncertainty", str(study), "--out", str(out)]) == 0
    s = json.loads((out / "collated" / "summary.json").read_text())
    assert s["parts"]["mc"]["n"] == 100
    assert s["parts"]["sobol"]["n"] == 32 * 5
    assert {p["category"] for p in s["parameters"]} == {"epistemic", "aleatory"}
    t = pq.read_table(out / "collated" / "mc.parquet")
    assert t.column("epistemic_index").to_pylist()[:6] == [0, 0, 0, 0, 0, 1]
    assert (out / "collated" / "sobol.png").is_file()


def test_study_validation_errors(tmp_path: Path) -> None:
    doc = _ishigami()
    doc["outputs"] = ["nope"]
    with pytest.raises(ConfigError, match="unknown metric"):
        load_uncertainty(_write(tmp_path, doc, "a.yaml"))
    doc = _ishigami()
    doc["sobol"]["n_base"] = 100
    with pytest.raises(ConfigError, match="power of two"):
        load_uncertainty(_write(tmp_path, doc, "b.yaml"))
    doc = _ishigami()
    doc["aleatory"][0]["name"] = "x0"
    with pytest.raises(ConfigError, match="unique"):
        load_uncertainty(_write(tmp_path, doc, "c.yaml"))
    doc = _ishigami()
    doc["epistemic"][0]["provenance"] = "sourced"
    with pytest.raises(ConfigError, match="needs a source"):
        load_uncertainty(_write(tmp_path, doc, "d.yaml"))
    # SALib 1.6 treats a bootstrap seed of 0 as 'unseeded' (global np.random): refused.
    for part in ("sobol", "monte_carlo"):
        doc = _ishigami()
        doc[part]["seed"] = 0
        with pytest.raises(ConfigError, match="greater than or equal to 1"):
            load_uncertainty(_write(tmp_path, doc, f"e_{part}.yaml"))
    with pytest.raises(ValueError, match=">= 1"):
        sobol_indices(["a", "b"], np.zeros(8), num_resamples=10, conf_level=0.95, seed=0)
