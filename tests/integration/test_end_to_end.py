"""PDF §13.2 integration tests of the single-satellite vertical slice (SPICE, short windows).

Every run uses the MVP configuration shortened to 2 days (49 hourly epochs, ~20 with access):
long enough to contain access, Earth occultation and below-horizon epochs, short enough to keep
each run below a second. References are either closed forms of the stage chain or the
comparison of two runs that differ in exactly one input.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

import helioswarm.system as system_mod
from helioswarm.cli import main
from helioswarm.config import RunConfig, parse_config, read_yaml
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.io import execute_run, prepare_run_dir, read_json
from helioswarm.kernels import KernelSet
from helioswarm.link_budget import STAGE_COLUMNS, link_budget_violations
from helioswarm.system import SystemRunResult, Unavailability, run_single_satellite

from config_helpers import CONFIG_DIR, MVP_CONFIG, set_path

pytestmark = pytest.mark.spice

DAYS = 2.0


def _cfg(**paths: Any) -> RunConfig:
    data = copy.deepcopy(read_yaml(MVP_CONFIG))
    data["time"]["duration_days"] = DAYS
    for k, v in paths.items():
        set_path(data, k, v)
    return parse_config(data)


def _run(ks: KernelSet, cfg: RunConfig, **kw: Any) -> SystemRunResult:
    with ks.loaded(verify=False) as k:
        return run_single_satellite(
            cfg, ephemeris=SpiceEphemeris(k), kernels=k, rng=np.random.default_rng(1), **kw
        )


@pytest.fixture(scope="module")
def base(kernel_set: KernelSet) -> SystemRunResult:
    return _run(kernel_set, _cfg())


def _bit(res: SystemRunResult, r: Unavailability) -> np.ndarray:
    return (res.reason_code & np.uint32(int(r))) != 0


def test_baseline_has_access_occultation_and_consistent_reasons(base: SystemRunResult) -> None:
    acc = base.access
    assert 10 < int(acc.visible.sum()) < len(base)
    assert acc.earth_blocked.any()
    assert acc.below_min_elevation.any()
    np.testing.assert_array_equal(_bit(base, Unavailability.EARTH_OCCULTATION), acc.earth_blocked)
    np.testing.assert_array_equal(
        _bit(base, Unavailability.BELOW_MIN_ELEVATION), acc.below_min_elevation
    )
    # Every availability condition holds at the MVP epochs with access (clear sky, feasible).
    np.testing.assert_array_equal(base.available, acc.visible)
    assert base.rng_state_unchanged


# Exact SI defining constants (2019 SI) and the IAU 2012 au, written here so the reference does
# not import the package values.
_K_B, _H_PL, _C = 1.380649e-23, 6.62607015e-34, 299_792_458.0
_SIGMA_SB = 2.0 * np.pi**5 * _K_B**4 / (15.0 * _H_PL**3 * _C**2)
_AU = 149_597_870_700.0


def _independent_pv_electric_W(cfg: RunConfig, sun_distance_m: float) -> float:
    """P_PV of the sun-facing, fully lit plate: brentq on the M4 plate balance written here,
    P_abs (1 - eta(T)) = (eps_f + eps_b) sigma A F (T^4 - T_s^4), eta(T) the clipped linear law
    (PDF §8.2), P_abs = alpha S_1au (au/r)^2 A, P_PV = P_abs eta(T)."""
    from scipy.optimize import brentq

    col, th = cfg.collector, cfg.thermal
    p_abs = col.absorptivity * cfg.constants.solar_constant_1au_W_m2 * (_AU / sun_distance_m) ** 2
    p_abs *= col.area_m2
    g = (th.pv_front_emissivity + th.pv_back_emissivity) * _SIGMA_SB * col.area_m2
    g *= th.pv_view_factor

    def eta(t: float) -> float:
        lin = col.pv_efficiency_reference * (
            1.0 + col.temperature_coefficient_per_K * (t - col.reference_temperature_K)
        )
        return min(max(lin, col.pv_efficiency_min), col.pv_efficiency_max)

    ts = th.space_temperature_K
    t = brentq(lambda t: p_abs * (1 - eta(t)) - g * (t**4 - ts**4), ts, 2000.0, xtol=1e-12)
    return p_abs * eta(t)


def _independent_expected_airy_capture(
    d_m: float, lam_m: float, range_m: float, a_m: float, sigma_rad: float
) -> float:
    """E[eta] for a centred disk of radius a in the Airy pattern with an isotropic Gaussian
    pointing error of sigma_rad per axis (offset Rayleigh with scale sigma R).

    By symmetry E[eta] = int I(rho) Pr(|u + delta| <= a) dA with |u| = rho, delta ~ N(0, s^2 I),
    s = sigma R; |u + delta|^2 / s^2 is noncentral chi-square (2 dof, nc = rho^2/s^2), so
    E[eta] = int_0^inf I(rho) F_ncx2(a^2/s^2; 2, rho^2/s^2) 2 pi rho drho with
    I(rho) = pi D^2/(4 lambda^2 R^2) (2 J1(k rho)/(k rho))^2, k = pi D/(lambda R) (unit power).
    One 1-D quadrature, independent of the M5 offset-capture and Rice quadratures."""
    from scipy import integrate, special, stats

    k = np.pi * d_m / (lam_m * range_m)
    i0 = np.pi * d_m**2 / (4.0 * lam_m**2 * range_m**2)
    s = sigma_rad * range_m

    def f(rho: float) -> float:
        x = k * rho
        airy = 1.0 if x == 0.0 else (2.0 * special.j1(x) / x) ** 2
        prob = stats.ncx2.cdf(a_m**2 / s**2, 2, rho**2 / s**2)
        return float(i0 * airy * prob * 2.0 * np.pi * rho)

    val, _ = integrate.quad(f, 0.0, a_m + 40.0 * s, epsabs=0.0, epsrel=1e-12, limit=400,
                            points=[max(a_m - 10 * s, 0.0), a_m, a_m + 10 * s])  # fmt: skip
    return float(val)


def test_reference_link_conditions_and_rated_power_independently(base: SystemRunResult) -> None:
    # The capacity-factor denominator (PDF §10.5) checked on its own terms, not from the factors
    # the ReferenceLink reports about itself: (1) the reference conditions are the defined ones,
    # (2) P_PV and the expected capture there match independent references, (3) the rated
    # power is the PDF §10.1 chain of those independent values.
    cfg = base.config
    ref = base.reference
    assert ref.range_m == float(np.mean(base.access.range_m))
    assert ref.sun_distance_m == float(np.mean(base.illumination.sun_distance_m))
    assert ref.elevation_rad == 0.5 * np.pi
    assert ref.thermal_feasible
    assert not ref.power_density_abort
    assert cfg.transmitter.pointing_bias_rad == 0.0  # centred jitter assumed by the reference
    # P_PV: brentq on the plate balance; tolerance 1e-9 (the package sigma is CODATA's
    # 10-digit rounding, 3e-11 relative; brentq xtol 1e-12 K).
    pv = _independent_pv_electric_W(cfg, ref.sun_distance_m)
    assert ref.pv_electric_W == pytest.approx(pv, rel=1e-9)
    # Expected capture at zenith (full receiver projection): independent 1-D noncentral-chi2
    # quadrature. Tolerance 1e-8 relative (both quadratures converge to < 1e-10; the M5 Rice
    # quadrature truncates the offset law at a probability mass 1 - 1e-12).
    tx = cfg.transmitter
    cap = _independent_expected_airy_capture(
        tx.aperture_diameter_m, tx.wavelength_m, ref.range_m, cfg.receiver.radius_m,
        tx.pointing_sigma_rad,
    )  # fmt: skip
    assert ref.eta_capture == pytest.approx(cap, rel=1e-8)
    # Discrimination: a reference evaluated at 3x the range changes capture by ~9x, and the
    # jitter loss itself (~(pi D sigma/lambda)^2/2 ~ 5e-4) is 5e4x the tolerance.
    from scipy import special

    x = np.pi * tx.aperture_diameter_m * cfg.receiver.radius_m / (tx.wavelength_m * ref.range_m)
    nojit = 1.0 - special.j0(x) ** 2 - special.j1(x) ** 2  # Rayleigh encircled energy, no jitter
    assert 1e-4 < 1.0 - cap / nojit < 1e-3
    # Clear-sky optical scenario at zenith: tau_z^1 = 1.0 for the MVP.
    assert ref.eta_atmosphere == cfg.atmosphere.optical_zenith_transmittance_clear == 1.0
    eff = (
        cfg.transmitter.power_conditioning_efficiency
        * cfg.transmitter.electrical_to_beam_efficiency
        * cfg.transmitter.optics_or_rf_efficiency
    )
    rated = pv * eff * 1.0 * cap * cfg.receiver.conversion_efficiency
    rated *= cfg.receiver.grid_efficiency
    assert ref.rated_power_W == pytest.approx(rated, rel=1e-8)
    assert 5.0e3 < ref.rated_power_W < 7.0e3  # 5.9 kW for the 2-day window (scale check)


def test_atmosphere_stage_follows_the_plane_parallel_law(
    kernel_set: KernelSet, base: SystemRunResult
) -> None:
    # A non-unit clear-sky transmittance makes the atmosphere stage visible end to end: the
    # scenario law eta = tau_z^(1/sin el) (plane-parallel air mass, M5) at the epoch's own
    # elevation, applied once between P_aperture and P_captured.
    tau = 0.8
    res = _run(kernel_set, _cfg(**{"atmosphere.optical_zenith_transmittance_clear": tau}))
    m = res.available
    np.testing.assert_array_equal(m, base.available)
    el = res.access.elevation_rad[m]
    assert float(np.ptp(el)) > 0.5  # elevations differ: a zenith-only (air mass 1) model fails
    expect = tau ** (1.0 / np.sin(el))
    np.testing.assert_allclose(res.link.eta_atmosphere[m], expect, rtol=1e-12)
    np.testing.assert_allclose(res.link.atm_W[m], res.link.aperture_W[m] * expect, rtol=1e-12)
    np.testing.assert_allclose(res.link.grid_W[m], base.link.grid_W[m] * expect, rtol=1e-12)
    np.testing.assert_array_equal(res.link.eta_capture[m], base.link.eta_capture[m])
    # Reference link: zenith, air mass 1.
    assert res.reference.eta_atmosphere == pytest.approx(tau, rel=1e-15)
    assert res.reference.rated_power_W == pytest.approx(tau * base.reference.rated_power_W,
                                                        rel=1e-12)  # fmt: skip


def test_unavailable_epochs_deliver_zero_and_energy_accounting(base: SystemRunResult) -> None:
    lk = base.link
    assert np.all(lk.grid_W[~base.available] == 0.0)
    assert np.all(lk.grid_W[base.available] > 0.0)
    assert link_budget_violations(lk) == ()
    prev = base.power_thermal.solar.incident_W
    for _, attr in STAGE_COLUMNS:
        cur = getattr(lk, attr)
        assert np.all(cur <= prev), attr
        prev = cur
    # P_grid = P_aperture * eta_atm * eta_capture * 0.4 * 0.97 exactly at available epochs.
    m = base.available
    ref = lk.aperture_W[m] * lk.eta_atmosphere[m] * lk.eta_capture[m] * 0.4 * 0.97
    np.testing.assert_allclose(lk.grid_W[m], ref, rtol=1e-15)


def test_impossible_beam_gives_near_zero_capture(kernel_set: KernelSet) -> None:
    res = _run(
        kernel_set,
        _cfg(**{"transmitter.aperture_diameter_m": 0.01, "receiver.radius_m": 1.0}),
    )
    m = res.available
    assert m.any()
    # Closed form: x = pi D a_eff / (lambda R) ~ 1.8e-7, capture ~ x^2/4 ~ 8e-15 (Airy small-x
    # limit; the 170 m jitter is negligible against the 1.7e7 m beam feature scale).
    a_eff = np.sqrt(np.sin(res.access.elevation_rad[m]))
    x = np.pi * 0.01 * a_eff / (1e-6 * res.access.range_m[m])
    np.testing.assert_allclose(res.link.eta_capture[m], x**2 / 4, rtol=1e-6)
    assert float(np.max(res.link.eta_capture[m])) < 1e-13
    assert float(np.max(res.link.grid_W / np.maximum(res.link.aperture_W, 1e-300))) < 1e-14


def test_receiver_capacity_limit_clips(kernel_set: KernelSet, base: SystemRunResult) -> None:
    limit = 0.5 * float(np.max(base.link.captured_W))
    res = _run(kernel_set, _cfg(**{"receiver.max_input_power_W": limit}))
    sat = res.link.receiver_saturated
    assert sat.any()
    assert not sat.all()
    assert np.all(res.link.accepted_W[sat] == limit)
    assert np.all(res.link.grid_W[sat] == limit * 0.4 * 0.97)
    np.testing.assert_array_equal(res.link.grid_W[~sat], base.link.grid_W[~sat])
    np.testing.assert_array_equal(_bit(res, Unavailability.RECEIVER_SATURATION), sat)
    np.testing.assert_array_equal(res.available, base.available)  # clipping is not an outage


def test_collector_area_scales_power_linearly(kernel_set: KernelSet, base: SystemRunResult) -> None:
    res = _run(kernel_set, _cfg(**{"collector.area_m2": 20000.0}))
    np.testing.assert_allclose(
        res.power_thermal.solar.pv_electric_W, 2 * base.power_thermal.solar.pv_electric_W,
        rtol=1e-12,
    )  # fmt: skip
    np.testing.assert_allclose(
        res.power_thermal.solar.pv_temperature_K, base.power_thermal.solar.pv_temperature_K,
        rtol=1e-12,
    )  # fmt: skip
    m = base.available
    np.testing.assert_allclose(res.link.grid_W[m], 2 * base.link.grid_W[m], rtol=1e-12)


def test_smaller_wavelength_or_larger_aperture_improves_capture(
    kernel_set: KernelSet, base: SystemRunResult
) -> None:
    m = base.available
    for change in ({"transmitter.wavelength_m": 5e-7}, {"transmitter.aperture_diameter_m": 20.0}):
        res = _run(kernel_set, _cfg(**change))
        assert np.all(res.link.eta_capture[m] > base.link.eta_capture[m]), change
        assert np.all(res.link.grid_W[m] > base.link.grid_W[m]), change


def test_larger_pointing_sigma_cannot_improve_expected_capture(
    kernel_set: KernelSet, base: SystemRunResult
) -> None:
    res = _run(kernel_set, _cfg(**{"transmitter.pointing_sigma_rad": 2.0e-8}))
    m = base.available
    assert np.all(res.link.eta_capture[m] < base.link.eta_capture[m])


def test_thermal_infeasibility_stops_the_transmitter(kernel_set: KernelSet) -> None:
    # T_PV ~ 302.8 K at 1 au: a 300 K limit makes every epoch infeasible.
    res = _run(kernel_set, _cfg(**{"collector.max_operating_temperature_K": 300.0}))
    assert np.all(_bit(res, Unavailability.THERMAL))
    assert not res.available.any()
    assert np.all(res.link.grid_W == 0.0)
    assert np.all(res.link.pv_to_transmitter_W == 0.0)
    assert np.all(res.link.pv_electric_W > 0.0)  # still generated, not transmitted
    # The same policy applies to the capacity-factor reference: not operated -> rated 0.
    assert not res.reference.thermal_feasible
    assert res.reference.rated_power_W == 0.0
    assert res.reference.pv_electric_W > 0.0


def test_weather_pointing_and_power_density_conditions(
    kernel_set: KernelSet, base: SystemRunResult
) -> None:
    cloudy = _run(kernel_set, _cfg(**{"models.weather_scenario": "cloudy"}))
    assert np.all(_bit(cloudy, Unavailability.WEATHER))
    assert np.all(cloudy.link.grid_W == 0.0)
    # Pointing condition: jitter makes E[eta]/eta(0) < 1 at every epoch, so a threshold of 1
    # blocks every epoch with access, and a threshold below the minimum blocks none.
    pe = base.beam.pointing_loss if base.beam is not None else np.array([])
    strict = _run(kernel_set, _cfg(**{"transmitter.min_pointing_efficiency": 1.0}))
    vis = base.access.visible
    assert np.all(_bit(strict, Unavailability.POINTING)[vis])
    assert not strict.available.any()
    loose = _run(
        kernel_set, _cfg(**{"transmitter.min_pointing_efficiency": float(np.min(pe)) * 0.999})
    )
    np.testing.assert_array_equal(loose.available, base.available)
    # Power density beam abort between the smallest and largest peak irradiance.
    dens = base.link.peak_power_density_W_m2[base.available]
    lim = float(np.median(dens))
    ab = _run(kernel_set, _cfg(**{"receiver.max_power_density_W_m2": lim}))
    abort = _bit(ab, Unavailability.POWER_DENSITY_ABORT)
    assert abort.any()
    assert (base.available & ~abort).any()
    np.testing.assert_array_equal(abort, base.available & (base.link.peak_power_density_W_m2 > lim))
    assert np.all(ab.link.grid_W[abort] == 0.0)
    assert np.all(ab.link.peak_power_density_W_m2 <= lim)


def test_eclipse_reason_is_wired(kernel_set: KernelSet, monkeypatch: pytest.MonkeyPatch) -> None:
    # The 2-day MVP window has no eclipse, so the illumination is replaced by a synthetic one
    # (umbra at epochs 20-22, penumbra at 23) to check the reason wiring and the zero power.
    real = system_mod.satellite_illumination

    def shadowed(*args: Any, **kw: Any) -> Any:
        ill = real(*args, **kw)
        f = np.array(ill.illumination_fraction)
        f[20:23] = 0.0
        f[23] = 0.5
        import dataclasses

        return dataclasses.replace(ill, illumination_fraction=f)

    monkeypatch.setattr(system_mod, "satellite_illumination", shadowed)
    res = _run(kernel_set, _cfg())
    ecl = _bit(res, Unavailability.ECLIPSE)
    assert ecl.tolist() == [20 <= i < 23 for i in range(len(res))]
    assert np.all(res.link.grid_W[ecl] == 0.0)
    assert np.all(res.link.pv_electric_W[ecl] == 0.0)
    assert _bit(res, Unavailability.PARTIAL_ECLIPSE).tolist() == [i == 23 for i in range(len(res))]


def test_jax_backend_run_matches_numpy(kernel_set: KernelSet, base: SystemRunResult) -> None:
    pytest.importorskip("jax")
    res = _run(kernel_set, _cfg(), backend="jax")
    np.testing.assert_allclose(res.link.grid_W, base.link.grid_W, rtol=1e-9, atol=0.0)
    np.testing.assert_array_equal(res.available, base.available)
    assert res.backend == "jax"


def test_more_than_one_site_is_rejected(kernel_set: KernelSet) -> None:
    data = copy.deepcopy(read_yaml(MVP_CONFIG))
    data["time"]["duration_days"] = DAYS
    data["receiver"]["sites"].append({"name": "b", "latitude_deg": 10.0, "longitude_deg": 0.0})
    with pytest.raises(ValueError, match="exactly one receiver site"):
        _run(kernel_set, parse_config(data))


def _write_cfg(tmp_path: Path, cfg: RunConfig) -> Path:
    from helioswarm.config import write_resolved_yaml

    return write_resolved_yaml(cfg, tmp_path / "cfg.yaml")


def test_cli_run_outputs_plot_and_reproducibility(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_cfg(tmp_path, _cfg())
    out = tmp_path / "run"
    assert main(["run", str(cfg_path), "--out", str(out)]) == 0
    assert "P_grid mean" in capsys.readouterr().out
    for name in (
        "config_resolved.yaml",
        "manifest.json",
        "timeseries.parquet",
        "summary.json",
        "report.md",
        "logs/run.log",
        "figures/composite.png",
        "figures/power_stages.png",
    ):
        assert (out / name).is_file(), name  # fmt: skip
    man = read_json(out / "manifest.json")
    assert man["config_hash"] == read_json(out / "summary.json")["config_hash"]
    assert len(man["git"]["commit"]) == 40
    assert isinstance(man["git"]["dirty"], bool)
    assert [k["name"] for k in man["kernels"]] == [
        "naif0012.tls", "pck00011.tpc", "gm_de440.tpc", "de440s.bsp"
    ]  # fmt: skip
    assert all(len(k["sha256"]) == 64 for k in man["kernels"])
    assert man["packages"]["numpy"]
    assert man["seed"] == 20261111
    assert man["rng_state_unchanged"]
    assert "TDB" in man["time_scale"]
    assert "J2000" in man["frame"]
    tab = pq.read_table(out / "timeseries.parquet")
    assert tab.num_rows == 49
    assert "grid_W" in tab.column_names
    assert "reason_code" in tab.column_names
    assert tab.schema.metadata[b"frame"] == b"J2000"
    log = (out / "logs" / "run.log").read_text()
    assert "diurnal aliasing" in log
    assert "TIMING" in log

    # Refuse to overwrite without --overwrite; succeed with it (same content again).
    assert main(["run", str(cfg_path), "--out", str(out)]) == 1
    assert "--overwrite" in capsys.readouterr().err
    first = (out / "summary.json").read_bytes()
    assert main(["run", str(cfg_path), "--out", str(out), "--overwrite"]) == 0
    assert (out / "summary.json").read_bytes() == first

    # Reproducible: a second run has identical summary.json and time-series content.
    out2 = tmp_path / "run2"
    assert main(["run", str(cfg_path), "--out", str(out2)]) == 0
    assert (out / "summary.json").read_bytes() == (out2 / "summary.json").read_bytes()
    # Content hash of the saved tables (NaN-aware, unlike Table.equals): bitwise identical.
    from helioswarm.io import table_content_sha256

    h1 = table_content_sha256(pq.read_table(out / "timeseries.parquet"))
    h2 = table_content_sha256(pq.read_table(out2 / "timeseries.parquet"))
    assert h1 == h2 == read_json(out / "summary.json")["timeseries_content_sha256"]

    # Figures regenerate from the saved files alone (no model is run by `plot`).
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("plot must not run the model")

    import helioswarm.io as io_mod

    monkeypatch.setattr(system_mod, "run_single_satellite", boom)
    monkeypatch.setattr(io_mod, "run_single_satellite", boom)
    before = hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()
    for f in (out / "figures").iterdir():
        f.unlink()
    (out / "report.md").unlink()
    assert main(["plot", str(out)]) == 0
    assert len(list((out / "figures").glob("*.png"))) == 6
    assert (out / "report.md").is_file()
    assert hashlib.sha256((out / "summary.json").read_bytes()).hexdigest() == before
    assert "## Metrics" in (out / "report.md").read_text()


def test_summary_is_recomputable_from_saved_timeseries(
    tmp_path: Path, kernel_set: KernelSet
) -> None:
    from helioswarm.io import summary_inputs
    from helioswarm.metrics import summarize_timeseries

    cfg = _cfg()
    out = execute_run(cfg, prepare_run_dir(tmp_path / "r", overwrite=False), kernel_set=kernel_set,
                      make_figures=False)  # fmt: skip
    with kernel_set.loaded(verify=False) as k:
        res = run_single_satellite(cfg, ephemeris=SpiceEphemeris(k), kernels=k,
                                   rng=np.random.default_rng(0))  # fmt: skip
    again = summarize_timeseries(pq.read_table(tmp_path / "r" / "timeseries.parquet"),
                                 summary_inputs(res))  # fmt: skip
    saved = json.loads((tmp_path / "r" / "summary.json").read_text())
    assert again["delivered_energy_J"] == saved["delivered_energy_J"]
    assert again["availability"] == saved["availability"]
    assert out.summary["timeseries_content_sha256"] == saved["timeseries_content_sha256"]


def test_cli_sweep_with_array_split_and_collation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    sw = tmp_path / "sw.yaml"
    sw.write_text(
        yaml.safe_dump(
            {
                "sweep_schema_version": 1,
                "name": "tiny",
                "base_config": str(MVP_CONFIG),
                "overrides": {"time.duration_days": 1.0},
                "parameters": {
                    "receiver.radius_m": [1000.0, 2000.0],
                    # D = 1 km at 1 um: N_F = D^2/(lambda R) ~ 6 > 0.5, so the Airy far field is
                    # invalid; those points must fail and be recorded, never dropped.
                    "transmitter.aperture_diameter_m": [10.0, 1000.0],
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert main(["sweep", str(sw), "--out", str(out), "--summary-only",
                 "--array-index", "0", "--array-count", "2"]) == 0  # fmt: skip
    rows = pq.read_table(out / "sweep_summary.parquet").to_pylist()
    assert [r["status"] for r in rows] == ["ok", "missing", "ok", "missing"]
    assert main(["sweep", str(sw), "--out", str(out), "--summary-only",
                 "--array-index", "1", "--array-count", "2"]) == 1  # fmt: skip
    rows = pq.read_table(out / "sweep_summary.parquet").to_pylist()
    assert [r["status"] for r in rows] == ["ok", "failed", "ok", "failed"]
    assert "BeamModelValidityError" in rows[1]["error"]
    assert rows[2]["eta_capture_median_visible"] > rows[0]["eta_capture_median_visible"]
    assert not (out / "points" / "p0000" / "timeseries.parquet").exists()  # --summary-only
    assert (out / "points" / "p0000" / "summary.json").is_file()
    assert main(["sweep", str(sw), "--out", str(out), "--collate-only"]) == 0
    capsys.readouterr()
    # Resume: complete points are skipped, failed ones stay failed, interrupted ones rerun.
    summary_p0 = (out / "points" / "p0000" / "summary.json").read_bytes()
    (out / "points" / "p0002" / "manifest.json").unlink()  # simulate a task killed mid-write
    assert main(["sweep", str(sw), "--out", str(out), "--summary-only",
                 "--array-index", "0", "--array-count", "2"]) == 0  # fmt: skip
    log = capsys.readouterr().out
    assert "p0000: already complete, skipped" in log
    assert "point p0002" in log
    assert ": ok" in log
    assert (out / "points" / "p0000" / "summary.json").read_bytes() == summary_p0
    assert (out / "points" / "p0002" / "manifest.json").is_file()
    assert main(["sweep", str(sw), "--out", str(out), "--summary-only",
                 "--array-index", "1", "--array-count", "2"]) == 1  # fmt: skip
    assert "failed earlier" in capsys.readouterr().out
    capsys.readouterr()


def test_microwave_config_runs_with_itu_r_provenance(kernel_set: KernelSet) -> None:
    pytest.importorskip("itur")
    data = copy.deepcopy(read_yaml(CONFIG_DIR / "mvp_single_satellite_microwave.yaml"))
    data["time"]["duration_days"] = 1.0
    res = _run(kernel_set, parse_config(data))
    assert "ITU-R P.618-13" in res.itu_r_versions
    assert res.itu_r_polarization_tilt_deg == 45.0
    m = res.available
    assert m.any()
    # Negative result: ~1e-7 of the beam reaches a 10 km rectenna from ~1.15 au.
    assert float(np.max(res.link.eta_capture[m])) < 1e-5
    # Closed form of the small-receiver limit: eta = eta_taper x^2/4 * pointing efficiency,
    # x = pi D a sqrt(sin el)/(lambda R) (horizontal receiver: projected area pi a^2 sin el).
    # eta_taper of the parabolic-on-pedestal field C + (1 - C)(1 - t^2), C = 10^(-T/20):
    # |int A dA|^2 / (area int A^2 dA) = ((1 + C)/2)^2 / (C^2 + C(1 - C) + (1 - C)^2/3).
    tx = res.config.transmitter
    c_edge = 10.0 ** (-tx.edge_taper_dB / 20.0)
    eta_taper = ((1 + c_edge) / 2) ** 2 / (
        c_edge**2 + c_edge * (1 - c_edge) + (1 - c_edge) ** 2 / 3
    )
    assert eta_taper == pytest.approx(0.917467, abs=1e-6)
    el = res.access.elevation_rad[m]
    x = np.pi * tx.aperture_diameter_m * res.config.receiver.radius_m * np.sqrt(np.sin(el))
    x /= tx.wavelength_m * res.access.range_m[m]
    assert res.beam is not None
    pe = res.expand(res.beam.pointing_loss)[m]
    # Tolerance 1e-4 relative: the O(x^2) term of the small-x expansion is ~x^2/8 ~ 5e-6 here
    # (x ~ 6e-3); the element pattern is 1 at boresight and the half-wavelength lattice has no
    # grating lobe. A missing taper (x 1.09), unprojected receiver (x 1/sin el) or wrong
    # wavelength fails by > 9 %.
    np.testing.assert_allclose(res.link.eta_capture[m], eta_taper * x**2 / 4 * pe, rtol=1e-4)
    assert np.all((pe > 0.999) & (pe < 1.0))
    # ITU-R attenuation: 0.9 < eta_atm < 1, and non-decreasing with elevation (shorter slant
    # path; same site, frequency and exceedance).
    eta_atm = res.link.eta_atmosphere[m]
    assert np.all(eta_atm < 1.0)
    assert np.all(eta_atm > 0.9)
    order = np.argsort(el)
    assert np.all(np.diff(eta_atm[order]) >= -1e-12)
    assert float(eta_atm[order][-1] - eta_atm[order][0]) > 1e-3  # elevation dependence present
