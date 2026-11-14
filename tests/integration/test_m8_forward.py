"""M8 forward-model evaluation, fidelity re-evaluation and the shipped study files (SPICE).

Short windows (1-2 days) keep these tests fast; the physics is the validated M6/M7b chain. The
independent paths are: a saved run directory written by :func:`helioswarm.io.execute_run`
(summary, time series, economics read back from files) compared with the in-memory metrics of
:class:`helioswarm.evaluation.ForwardModel`; the trapezoid threshold fraction recomputed with
numpy from the saved time series; the mass proxy recomputed from its formula.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from helioswarm.compare import harmonisation_violations
from helioswarm.config import (
    ConfigError,
    MassProxyConfig,
    load_config,
    parse_config,
    provenance_table,
    read_config_mapping,
    read_yaml,
)
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.evaluation import (
    DesignTask,
    EvaluationSettings,
    FidelitySpec,
    ForwardModel,
    lcoe_from_run_dir,
)
from helioswarm.io import execute_run, prepare_run_dir, read_json
from helioswarm.kernels import KernelSet, LoadedKernels
from helioswarm.optimization import collate, load_optimization, optimize_seed, reevaluate
from helioswarm.system import run_single_satellite
from helioswarm.uncertainty import load_uncertainty

from config_helpers import CONFIG_DIR

pytestmark = pytest.mark.spice

MP = MassProxyConfig(
    collector_areal_mass_kg_m2=1.0, transmitter_areal_mass_kg_m2=50.0, bus_mass_kg=500.0
)
L1_KEYS = (
    "orbit.include_planetary_perturbations",
    "orbit.include_solar_radiation_pressure",
    "orbit.cr",
    "orbit.area_to_mass_m2_kg",
    "time.integrator_step_s",
)


def _short(days: float = 1.0) -> dict[str, Any]:
    return {"time.duration_days": days, "time.output_step_s": 3600}


def _helio_base() -> dict[str, Any]:
    return copy.deepcopy(read_config_mapping(CONFIG_DIR / "comparison" / "helio_1au_optical.yaml"))


def _screening(days: float = 1.0, **kw: Any) -> FidelitySpec:
    return FidelitySpec(
        description="test screening",
        overrides={"orbit.model": "two_body_kepler", **_short(days)},
        unset=L1_KEYS,
        backend=kw.get("backend", "numpy_batch"),
        capture_percentiles=kw.get("capture_percentiles", True),
    )


def test_capture_percentiles_off_leaves_every_power_unchanged(spice_kernels: LoadedKernels) -> None:
    data = _helio_base()
    for k in L1_KEYS:
        section, key = k.split(".")
        data[section].pop(key, None)
    data["orbit"]["model"] = "two_body_kepler"
    data["time"].update(duration_days=2.0, output_step_s=3600)
    data["transmitter"]["pointing_sigma_rad"] = 3e-8  # non-monotone eta(d): level-set percentiles
    cfg = parse_config(data)
    eph = SpiceEphemeris(spice_kernels)
    runs = {
        p: run_single_satellite(
            cfg,
            ephemeris=eph,
            kernels=spice_kernels,
            rng=np.random.default_rng(1),
            backend="numpy_batch",
            capture_percentiles=p,
        )
        for p in (True, False)
    }
    on, off = runs[True], runs[False]
    assert on.beam is not None
    assert off.beam is not None
    assert on.beam.range_m.size > 0
    # Identical quadrature for the expected capture: bit-identical powers.
    assert np.array_equal(on.link.grid_W, off.link.grid_W)
    assert np.array_equal(on.beam.capture_fraction, off.beam.capture_fraction)
    assert np.all(np.isnan(off.beam.capture_fraction_p05))
    assert set(off.beam.diagnostics.percentile_method.tolist()) == {"not_computed"}
    assert np.all(np.isfinite(on.beam.capture_fraction_p05))
    with pytest.raises(ValueError, match="batched backends"):
        run_single_satellite(
            cfg,
            ephemeris=eph,
            kernels=spice_kernels,
            rng=np.random.default_rng(1),
            backend="numpy",
            capture_percentiles=False,
        )


def test_forward_model_metrics_equal_a_saved_run(kernel_set: KernelSet, tmp_path: Path) -> None:
    econ = str(CONFIG_DIR / "economics" / "nominal.yaml")
    fm = ForwardModel(
        base_mapping=_helio_base(),
        base_source="helio",
        fidelity_name="screening",
        fidelity=_screening(2.0, backend="numpy"),
        settings=EvaluationSettings(power_threshold_W=1.0e5, mass_proxy=MP, economics_config=econ),
    )
    task = DesignTask(0, "d0", (("transmitter.aperture_diameter_m", 20.0),))
    res = fm.evaluate([task])[0]
    assert res.status == "ok", res.error
    cfg = fm.config_for(task)
    assert cfg.provenance_overrides == {"transmitter.aperture_diameter_m": "swept"}
    run_dir = prepare_run_dir(tmp_path / "run", overwrite=False)
    out = execute_run(cfg, run_dir, kernel_set=kernel_set, make_figures=False)
    s = out.summary
    m = res.metrics
    assert m["p_grid_mean_W"] == s.p_grid_W.time_mean
    assert m["delivered_energy_J"] == s.delivered_energy_J
    assert m["availability"] == s.availability
    ts = pq.read_table(run_dir / "timeseries.parquet")
    grid = np.asarray(ts.column("grid_W").to_numpy(), dtype=float)
    w = np.full(grid.size, 3600.0)
    w[0] = w[-1] = 1800.0
    assert m["availability_above_threshold"] == pytest.approx(
        np.sum(w[grid >= 1.0e5]) / np.sum(w), rel=1e-15
    )
    assert 0.0 < m["availability_above_threshold"] < m["availability"]  # threshold binds
    a_rad = s.thermal.radiator_area_m2_max
    assert m["mass_proxy_kg"] == pytest.approx(
        500.0 + 1.0 * 10000.0 + 5.0 * a_rad + 50.0 * math.pi * 20.0**2 / 4.0, rel=1e-14
    )
    with kernel_set.loaded(verify=False) as k:
        lcoe = lcoe_from_run_dir(run_dir, econ, SpiceEphemeris(k))
    assert m["lcoe_usd_per_kWh"] == pytest.approx(lcoe, rel=1e-12)
    # varied economics/mass-proxy inputs act on the metrics only (same label -> same
    # configuration and hash)
    t2 = DesignTask(
        1,
        "d0",
        (
            ("transmitter.aperture_diameter_m", 20.0),
            ("economics.manufacturing_cost_per_kg_usd", 9000.0),
            ("mass_proxy.bus_mass_kg", 1500.0),
        ),
    )
    r2 = fm.evaluate([t2])[0]
    assert r2.config_hash == res.config_hash
    assert r2.metrics["mass_proxy_kg"] == pytest.approx(m["mass_proxy_kg"] + 1000.0, rel=1e-14)
    assert r2.metrics["lcoe_usd_per_kWh"] > m["lcoe_usd_per_kWh"]
    assert r2.metrics["p_grid_mean_W"] == m["p_grid_mean_W"]


def test_constellation_forward_model_metrics_from_its_saved_run(tmp_path: Path) -> None:
    """The constellation branch (M7 chain + :func:`constellation_metrics`), checked against
    quantities recomputed here from the saved per-epoch network table and per-satellite table
    (trapezoid integrals with numpy; the mass proxy from its formula)."""
    base = copy.deepcopy(read_config_mapping(CONFIG_DIR / "constellation_demo.yaml"))
    fm = ForwardModel(
        base_mapping=base,
        base_source="constellation_demo",
        fidelity_name="screening",
        fidelity=_screening(1.0),
        settings=EvaluationSettings(power_threshold_W=1.0e4),
        run_root=str(tmp_path / "runs"),
    )
    n_sat, a_col = 3, 5000.0
    task = DesignTask(
        0,
        "c0",
        (("constellation.rings[0].n_satellites", n_sat), ("collector.area_m2", a_col)),
    )
    res = fm.evaluate([task])[0]
    assert res.status == "ok", res.error
    m = res.metrics
    run = tmp_path / "runs" / "c0"
    sats = pq.read_table(run / "satellites.parquet")
    assert sats.num_rows == n_sat  # the design reached the run
    net = pq.read_table(run / "network.parquet")
    t = np.asarray(net.column("elapsed_s").to_numpy(), dtype=float)
    grid = np.asarray(net.column("grid_W").to_numpy(), dtype=float)
    assert t.size == 25  # 1 day at 3600 s, both ends
    energy = float(np.trapezoid(grid, t))
    # trapezoid sums in a different order: 1e-12 relative
    assert m["delivered_energy_J"] == pytest.approx(energy, rel=1e-12)
    assert m["p_grid_mean_W"] == pytest.approx(energy / (t[-1] - t[0]), rel=1e-12)
    w = np.full(grid.size, 3600.0)
    w[0] = w[-1] = 1800.0
    frac = float(np.sum(w[grid >= 1.0e4]) / np.sum(w))
    assert m["availability_above_threshold"] == pytest.approx(frac, rel=1e-15)
    assert 0.0 < frac < 1.0  # the threshold binds in this window (discriminating)
    a_rad = np.asarray(sats.column("radiator_area_required_m2").to_numpy(), dtype=float)
    # constellation_demo.yaml mass proxy: 500 kg + 1 kg/m^2 A_col + 5 kg/m^2 A_rad
    # (thermal.radiator_specific_mass_kg_m2) + 50 kg/m^2 pi D^2/4, D = 10 m, per satellite
    mass = n_sat * (500.0 + 1.0 * a_col + 50.0 * math.pi * 10.0**2 / 4.0) + 5.0 * a_rad.sum()
    assert m["mass_proxy_kg"] == pytest.approx(mass, rel=1e-12)
    assert m["collector_area_m2"] == n_sat * a_col
    cfg = fm.config_for(task)
    assert m["receiver_area_m2"] == pytest.approx(
        len(cfg.receiver.sites) * math.pi * cfg.receiver.radius_m**2, rel=1e-15
    )
    assert m["captured_power_peak_W"] == float(
        np.max(np.asarray(net.column("captured_W").to_numpy(), dtype=float))
    )
    assert 0.0 <= m["capacity_loss_fraction"] < 1.0
    assert math.isnan(m["lcoe_usd_per_kWh"])  # no economics configured


def test_invalid_beam_regime_is_a_failed_design_not_a_crash(kernel_set: KernelSet) -> None:
    fm = ForwardModel(
        base_mapping=_helio_base(),
        base_source="helio",
        fidelity_name="screening",
        fidelity=_screening(1.0),
        settings=EvaluationSettings(mass_proxy=MP),
    )
    # 400 m optical aperture at ~0.5 au: N_F = D^2/(lambda R) ~ 2 > 0.5 -> Airy refuses.
    bad = DesignTask(0, "bad", (("transmitter.aperture_diameter_m", 400.0),))
    invalid = DesignTask(1, "invalid", (("receiver.min_elevation_deg", 1.0),))  # config error
    r_bad, r_invalid = fm.evaluate([bad, invalid])
    assert r_bad.status == "failed"
    assert r_bad.error_type == "BeamModelValidityError"
    assert r_bad.config_hash is not None
    assert r_invalid.status == "failed"
    assert r_invalid.error_type == "ConfigError"
    assert r_invalid.config_hash is None


def _tiny_study(tmp_path: Path) -> Path:
    base = str(CONFIG_DIR / "comparison" / "helio_1au_optical.yaml")
    doc = {
        "optimization_schema_version": 1,
        "name": "tiny",
        "base_config": base,
        "search_fidelity": "screening",
        "fidelity": {
            "screening": {
                "description": "L0, 1 day",
                "unset": list(L1_KEYS),
                "overrides": {"orbit.model": "two_body_kepler", **_short(1.0)},
                "backend": "numpy_batch",
                "capture_percentiles": False,
            },
            "high": {"description": "L1, 1 day", "overrides": _short(1.0), "backend": "numpy"},
        },
        "evaluation": {"mass_proxy": MP.model_dump()},
        "variables": [
            {
                "name": "aperture",
                "kind": "real",
                "key": "transmitter.aperture_diameter_m",
                "lower": 5.0,
                "upper": 60.0,
                "log": True,
            },
            {
                "name": "radius",
                "kind": "real",
                "key": "receiver.radius_m",
                "lower": 300.0,
                "upper": 3000.0,
                "log": True,
            },
        ],
        "objectives": [
            {"metric": "p_grid_mean_W", "sense": "max", "reference": 0.0},
            {"metric": "mass_proxy_kg", "sense": "min", "reference": 3.0e5},
        ],
        "constraints": [
            {"name": "irr", "metric": "peak_incident_irradiance_W_m2", "op": "<=", "limit": 1000.0}
        ],
        "algorithm": {"pop_size": 4, "n_generations": 2, "seeds": [1]},
        "reevaluation": {"fidelity": "high", "n_candidates": 2},
    }
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


@pytest.mark.slow
def test_screening_search_then_high_fidelity_reevaluation(
    kernel_set: KernelSet, tmp_path: Path
) -> None:
    problem = load_optimization(_tiny_study(tmp_path))
    out = tmp_path / "opt"
    optimize_seed(problem, out, 1)
    c = collate(problem, out)
    assert c.union_front.num_rows >= 1
    r = reevaluate(problem, out)
    t = r.table
    assert t.num_rows == min(2, c.union_front.num_rows)
    assert set(t.column("high_status").to_pylist()) == {"ok"}
    for k, label in enumerate(t.column("candidate").to_pylist()):
        run_dir = r.path / "runs" / label
        manifest = read_json(run_dir / "manifest.json")
        cfg = load_config(run_dir / "config_resolved.yaml")
        assert cfg.orbit.model == "prescribed_ephemeris_perturbed_test_particle"  # high = L1
        assert manifest["config_hash"]
        summary = read_json(run_dir / "summary.json")
        assert t.column("high_p_grid_mean_W")[k].as_py() == summary["p_grid_W"]["time_mean"]
        # the screening value is the one recorded by the search for the same design
        ev = pq.read_table(out / "seed_1" / "evaluations.parquet").to_pylist()
        sid = t.column("screening_eval_id")[k].as_py()
        rec = next(e for e in ev if e["eval_id"] == sid)
        assert t.column("screening_p_grid_mean_W")[k].as_py() == rec["m_p_grid_mean_W"]
        # L0 vs L1 over one day at 1 au: P_grid differs by < 1e-3 relative (orbit drift ~km)
        assert abs(t.column("reldiff_p_grid_mean_W")[k].as_py()) < 1e-3
    assert (r.path / "reevaluation.md").is_file()
    # Re-evaluating the same designs at the SEARCH fidelity (written as complete runs and read
    # back from the saved files) reproduces the recorded screening metrics exactly: the
    # evaluation is deterministic and the in-memory and saved-file paths agree.
    again = reevaluate(problem, out, fidelity="screening")
    assert again.path.name == "reevaluation_screening"
    for m in ("p_grid_mean_W", "mass_proxy_kg", "peak_incident_irradiance_W_m2"):
        assert again.table.column(f"reldiff_{m}").to_pylist() == [0.0] * again.table.num_rows
    with pytest.raises(ValueError, match="not defined"):
        reevaluate(problem, out, fidelity="nope")


# ---------------------------------------------------------------------------------------------
# Shipped study files
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["optimization.yaml", "optimization_technology.yaml", "optimization_constellation.yaml"]
)
def test_shipped_optimisation_studies_load(name: str) -> None:
    p = load_optimization(CONFIG_DIR / name)
    assert p.definition.reevaluation is not None
    assert len(p.definition.algorithm.seeds) >= 2  # repeatability across seeds is part of M8


@pytest.mark.parametrize(
    "name", ["uncertainty.yaml", "uncertainty_geo.yaml", "uncertainty_constellation.yaml"]
)
def test_shipped_uncertainty_studies_load(name: str) -> None:
    s = load_uncertainty(CONFIG_DIR / name)
    cats = {s.kind_of(p.name) for p in s.parameters}
    assert cats == {"epistemic", "aleatory"}
    for p in s.parameters:
        assert p.provenance in ("sourced", "derived", "assumed")


def test_scenario_twins_differ_only_in_orbit_geometry() -> None:
    for name in ("optimistic", "nominal", "conservative"):
        helio = load_config(CONFIG_DIR / "scenarios" / f"{name}.yaml")
        geo = load_config(CONFIG_DIR / "scenarios" / f"geo_{name}.yaml")
        bad = [
            v
            for v in harmonisation_violations([("h", helio), ("g", geo)])
            if not v.startswith("project.")
        ]
        assert bad == []


def test_uncertainty_ranges_are_the_scenario_values(tmp_path: Path) -> None:
    # A changed scenario value must be caught on load (scenario_consistency).
    doc = read_yaml(CONFIG_DIR / "uncertainty.yaml")
    doc["base_config"] = str(CONFIG_DIR / doc["base_config"])
    doc["evaluation"]["economics_config"] = str(CONFIG_DIR / doc["evaluation"]["economics_config"])
    for sc in doc["scenarios"]:
        sc["config"] = str(CONFIG_DIR / sc["config"])
    ok = tmp_path / "ok.yaml"
    ok.write_text(yaml.safe_dump(doc), encoding="utf-8")
    load_uncertainty(ok)  # the shipped values are consistent
    doc["epistemic"][0]["distribution"]["mode"] = 0.31
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ConfigError, match="does not match the scenario"):
        load_uncertainty(bad)


def test_pv_technology_values_follow_from_the_sourced_numbers() -> None:
    iii_v = load_config(CONFIG_DIR / "scenarios" / "pv_iii_v_3g30a.yaml").collector
    si = load_config(CONFIG_DIR / "scenarios" / "pv_silicon_k7_1979.yaml").collector
    # AZUR SPACE DB 00010892-01: CIC 29.0 % (28 C), dVmp/dT -6.7 mV/C at 2400 mV,
    # dImp/dT 0.48 mA/C at 998 mA, absorptivity <= 0.91.
    gamma = -6.7 / 2400 + 0.48 / 998
    assert iii_v.temperature_coefficient_per_K == pytest.approx(gamma, abs=5e-6)  # 3 digits
    assert iii_v.absorptivity == 0.91
    eta = 0.290 / 0.91 * (1 + gamma * (300.0 - 301.15))
    assert iii_v.pv_efficiency_reference == pytest.approx(eta, abs=5e-5)  # 4 digits
    # JPL Pub. 78-15 Vol. VIII Table 7 (135.3 mW/cm^2): 15.24 % at 20 C, 13.90 % at 40 C.
    g_si = (13.90 - 15.24) / (20.0 * 15.24)
    assert si.temperature_coefficient_per_K == pytest.approx(g_si, abs=5e-5)
    assert si.pv_efficiency_reference == pytest.approx(
        0.1524 / 0.90 * (1 + g_si * (300 - 293.15)), abs=5e-5
    )
    # Machine-readable provenance (PDF §13.4): sourced/derived cell data labelled 'sourced',
    # the K7 absorptivity (not in the source) and the PDF §5 example values 'assumed'.
    expected = {
        "pv_iii_v_3g30a": ("sourced", "sourced", "sourced"),
        "pv_silicon_k7_1979": ("sourced", "sourced", "assumed"),
        "nominal": ("assumed", "sourced", "assumed"),
        "optimistic": ("sourced", "assumed", "assumed"),
        "conservative": ("sourced", "sourced", "assumed"),
        "geo_nominal": ("assumed", "sourced", "assumed"),
        "geo_optimistic": ("sourced", "assumed", "assumed"),
        "geo_conservative": ("sourced", "sourced", "assumed"),
    }
    keys = (
        "collector.pv_efficiency_reference",
        "collector.temperature_coefficient_per_K",
        "collector.absorptivity",
    )
    for name, labels in expected.items():
        rows = {
            r.path: r.provenance
            for r in provenance_table(load_config(CONFIG_DIR / "scenarios" / f"{name}.yaml"))
        }
        assert tuple(rows[k] for k in keys) == labels, name


def test_geo_economics_scenarios_differ_only_in_transport() -> None:
    for sc in ("optimistic", "nominal", "conservative"):
        a = read_yaml(CONFIG_DIR / "economics" / f"{sc}.yaml")
        b = read_yaml(CONFIG_DIR / "economics" / f"geo_{sc}.yaml")
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert diff <= {"scenario", "description", "transport"}
        assert b["transport"]["delta_v_model"] == "explicit"
        assert b["transport"]["delta_v_m_s"]["value"] == 4700.0
        assert (
            a["transport"]["launch_cost_per_kg_leo_usd"]
            == b["transport"]["launch_cost_per_kg_leo_usd"]
        )
