"""M7b: the GEO baseline through the identical chain, and `helioswarm compare` on saved runs.

Short windows and a 100 m receiver keep the Fresnel-Hankel evaluations cheap (0.14 s each at
the GEO range); the physics is that of the shipped comparison configurations (same files,
time window overridden).
"""

from __future__ import annotations

import copy
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from helioswarm.aperture import BeamModelValidityError
from helioswarm.beam import beam_settings_from_config, evaluate_beam
from helioswarm.cli import main
from helioswarm.compare import load_comparison, write_comparison
from helioswarm.config import parse_config, read_config_mapping
from helioswarm.constants import AU_M
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.kernels import LoadedKernels
from helioswarm.metrics import read_summary
from helioswarm.sweep import load_sweep, prepare_sweep_dir, run_sweep_points
from helioswarm.system import Unavailability, run_single_satellite

from config_helpers import CONFIG_DIR

CMP = CONFIG_DIR / "comparison"


def _case(name: str, **time: Any) -> dict[str, Any]:
    d = copy.deepcopy(read_config_mapping(CMP / f"{name}.yaml"))
    d["time"].update(time)
    return d


def _run(data: dict[str, Any], kernels: LoadedKernels) -> Any:
    cfg = parse_config(data)
    return run_single_satellite(
        cfg,
        ephemeris=SpiceEphemeris(kernels),
        kernels=kernels,
        rng=np.random.default_rng(cfg.project.seed),
    )


def test_fresnel_regime_selection_at_the_geo_range(spice_kernels: LoadedKernels) -> None:
    """The far-field validity check triggers for the optical GEO link; Fresnel-Hankel runs."""
    d = _case("geo_optical", start_utc="2030-03-21T00:00:00", duration_days=0.25, output_step_s=900)
    d["receiver"]["radius_m"] = 100.0
    far = copy.deepcopy(d)
    far["models"]["beam"] = "airy_far_field"
    far["models"]["beam_evaluation"] = "per_epoch"
    with pytest.raises(BeamModelValidityError, match="Fresnel number"):
        _run(far, spice_kernels)
    res = _run(d, spice_kernels)
    assert res.beam is not None
    assert res.beam_table is not None
    # N_F = D^2 / (lambda R) = 100 / (1e-6 * 3.579e7) = 2.79 > 0.5: near field (closed form).
    r = res.access.range_m[res.beam_epoch_index]
    np.testing.assert_allclose(res.beam.fresnel_number, 10.0**2 / (1.0e-6 * r), rtol=1e-15)
    assert np.all(res.beam.fresnel_number > 2.7)
    # Microwave GEO link: N_F = 1000^2 / (0.1224 * 3.579e7) = 0.228 <= 0.5, so the far-field
    # model is valid there and the two models must agree. Measured difference (M7b review):
    # 0.97862 (FH) vs 0.97873 (far field), |d| = 1.1e-4. The far-field model drops the
    # quadratic aperture phase pi rho'^2/(lambda R) (at most pi N_F / 4 = 0.18 rad at the edge,
    # weighted down by the 10 dB edge taper), whose effect on the encircled power of a receiver
    # that already captures ~98 % is a small fraction of the 2 % that misses it. Tolerance
    # 1e-3 = 9x the measured difference (a regime mix-up, e.g. the optical aperture, changes
    # the capture by > 1e-2); the lower bound shows the near-field term is actually resolved.
    m = _case("geo_microwave")
    s_fh = beam_settings_from_config(parse_config(m))
    m["models"]["beam"] = "aperture_far_field"
    m["models"]["beam_evaluation"] = "per_epoch"
    s_ff = beam_settings_from_config(parse_config(m))
    rr, el = np.array([3.5786e7]), np.array([0.5 * math.pi])
    fh, ff = evaluate_beam(rr, el, s_fh), evaluate_beam(rr, el, s_ff)
    assert float(fh.fresnel_number[0]) == pytest.approx(0.2284, abs=1e-4)
    diff = abs(float(fh.capture_fraction[0]) - float(ff.capture_fraction[0]))
    assert 1e-6 < diff < 1e-3


@pytest.mark.parametrize(
    ("case", "far_model"),
    [("helio_1au_optical", "airy_far_field"), ("helio_1au_microwave", "aperture_far_field")],
)
def test_far_field_and_fresnel_hankel_agree_at_the_heliocentric_range(
    case: str, far_model: str
) -> None:
    """The evidence for the one non-orbit harmonisation exception (configs/comparison/
    _shared.yaml): at the heliocentric ranges of the comparison (0.48-0.55 au) the far-field
    model of the heliocentric case and the Fresnel-Hankel model of the GEO case give the same
    capture for the base technology (zenith and 30 deg elevation, ranges 7.2e10 and 8.2e10 m).

    N_F = D^2/(lambda R) <= 1.4e-3 (optical, 10 m) and <= 1.1e-4 (microwave, 1 km): the
    neglected aperture phase is <= pi N_F/4, so the far-field capture is exact to ~N_F^2
    relative (< 2e-6). Measured: -9.9e-8 / -7.6e-8 (optical), -6.0e-10 / -4.6e-10 (microwave).
    Tolerance 1e-5 relative (captures 0.018-0.046 optical, 1.1e-6-2.9e-6 microwave); a wrong
    range, aperture or receiver radius changes the capture by orders of magnitude more.
    """
    d = _case(case)
    d["models"]["beam"] = far_model
    s_ff = beam_settings_from_config(parse_config(d))
    d["models"]["beam"] = "fresnel_hankel"
    s_fh = beam_settings_from_config(parse_config(d))
    rr = np.array([7.2e10, 8.2e10])
    el = np.array([0.5 * math.pi, math.radians(30.0)])
    for r, e in zip(rr, el, strict=True):
        fh = evaluate_beam(np.array([r]), np.array([e]), s_fh)
        ff = evaluate_beam(np.array([r]), np.array([e]), s_ff)
        assert float(fh.fresnel_number[0]) < 2e-3
        assert float(fh.capture_fraction[0]) > 0.0
        assert float(fh.capture_fraction[0]) == pytest.approx(
            float(ff.capture_fraction[0]), rel=1e-5
        )


def test_geo_link_through_the_chain_at_equinox(spice_kernels: LoadedKernels) -> None:
    """Two days at the March equinox, 120 s step: eclipses block the link; the rest is the
    same chain as the heliocentric runs (solar/thermal at ~1 au, zenith beam, clear sky)."""
    d = _case("geo_optical", start_utc="2030-03-19T12:00:00", duration_days=2.0, output_step_s=120)
    d["receiver"]["radius_m"] = 100.0
    res = _run(d, spice_kernels)
    code = res.reason_code
    ecl = (code & np.uint32(int(Unavailability.ECLIPSE))) != 0
    pen = (code & np.uint32(int(Unavailability.PARTIAL_ECLIPSE))) != 0
    # One umbra passage per day of ~67 min (test_geo_baseline checks it against the analytic
    # geometry): 2 passages x 67 min / 2 min ~ 67 epochs, +-2 per edge.
    assert 60 <= int(np.count_nonzero(ecl)) <= 72
    assert int(np.count_nonzero(pen)) >= 2
    assert np.all(res.link.grid_W[ecl] == 0.0)
    assert np.all(~res.available[ecl])
    # The only blocking reason is the eclipse: access is continuous (no elevation, Earth or
    # Sun-occultation outage of the zenith link in this window).
    others = int(
        sum(
            int(r)
            for r in (
                Unavailability.BELOW_MIN_ELEVATION,
                Unavailability.EARTH_OCCULTATION,
                Unavailability.THERMAL,
            )
        )
    )
    assert not np.any(code & np.uint32(others))
    assert np.array_equal(res.available, ~ecl)
    # Penumbra reduces the incident sunlight in proportion to the visible solar disk (the PV
    # efficiency then changes with the lower PV temperature, so P_PV is not proportional).
    # Incident = S_1au (1 au / d)^2 A f for the Sun-tracking collector (inverse-square law; d
    # changes by ~3e-4 per day in March, so it is taken per epoch).
    sol = res.power_thermal.solar
    full = res.illumination.illumination_fraction == 1.0
    i = np.flatnonzero(pen)
    cfg = res.config
    d = res.illumination.sun_distance_m[i]
    expected = (
        cfg.constants.solar_constant_1au_W_m2
        * (AU_M / d) ** 2
        * cfg.collector.area_m2
        * res.illumination.illumination_fraction[i]
    )
    np.testing.assert_allclose(sol.incident_W[i], expected, rtol=1e-12)
    assert np.all(sol.pv_electric_W[i] < float(np.min(sol.pv_electric_W[full])))
    # Zenith, ~35 786 km, nearly everything captured by a 100 m receiver.
    assert float(np.min(res.access.elevation_rad)) > math.radians(89.99)
    assert float(np.mean(res.access.range_m)) == pytest.approx(3.5786e7, rel=2e-4)
    assert float(np.min(res.beam.capture_fraction)) > 0.99
    # Link budget: P_grid = P_PV x eta chain x eta_atm x capture (PDF §10.1) at available epochs.
    avail = np.flatnonzero(res.available)
    eff = res.link.grid_W[avail] / res.power_thermal.solar.pv_electric_W[avail]
    tx, rx = res.config.transmitter, res.config.receiver
    eta_chain = (
        tx.power_conditioning_efficiency
        * tx.electrical_to_beam_efficiency
        * tx.optics_or_rf_efficiency
        * rx.conversion_efficiency
        * rx.grid_efficiency
    )
    assert eta_chain == pytest.approx(0.50 * 0.40 * 0.97)  # the shared optical block
    cap = res.expand(res.beam.capture_fraction)[avail]
    atm = res.expand(res.atmosphere.transmittance)[avail]
    np.testing.assert_allclose(eff, eta_chain * cap * atm, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# helioswarm compare on saved sweeps (tiny, real runs)
# ---------------------------------------------------------------------------------------------


TINY_STEP_S = 120.0


def _tiny_comparison(tmp: Path) -> Path:
    """Optical helio + GEO cases, 3 h across the GEO eclipse at the 2030 March equinox
    (2030-03-20T23:00 + 3 h at 120 s: the umbra and both penumbra transits of the 03-21 00:00
    UT passage are inside), receiver radius {100 m, 100 km}: at the GEO range the 100 km
    receiver violates the Fresnel paraxial condition (a recorded failure)."""
    cfg_dir = tmp / "configs"
    cfg_dir.mkdir()
    for f in ("_shared.yaml", "_optical.yaml", "helio_1au_optical.yaml", "geo_optical.yaml"):
        shutil.copy(CMP / f, cfg_dir / f)
    shared = yaml.safe_load((cfg_dir / "_shared.yaml").read_text(encoding="utf-8"))
    shared["time"].update(
        start_utc="2030-03-20T23:00:00", duration_days=0.125, output_step_s=TINY_STEP_S
    )
    (cfg_dir / "_shared.yaml").write_text(yaml.safe_dump(shared), encoding="utf-8")
    opt = yaml.safe_load((cfg_dir / "_optical.yaml").read_text(encoding="utf-8"))
    opt["receiver"]["radius_m"] = 100.0
    (cfg_dir / "_optical.yaml").write_text(yaml.safe_dump(opt), encoding="utf-8")
    for case in ("helio_1au_optical", "geo_optical"):
        (cfg_dir / f"{case}_sweep.yaml").write_text(
            yaml.safe_dump(
                {
                    "sweep_schema_version": 1,
                    "name": f"tiny_{case}",
                    "base_config": f"{case}.yaml",
                    "parameters": {"receiver.radius_m": [100.0, 100000.0]},
                }
            ),
            encoding="utf-8",
        )
    cmp_file = cfg_dir / "comparison.yaml"
    cmp_file.write_text(
        yaml.safe_dump(
            {
                "comparison_schema_version": 1,
                "name": "tiny",
                "mass_proxy": {
                    "collector_areal_mass_kg_m2": 1.0,
                    "transmitter_areal_mass_kg_m2": 50.0,
                    "bus_mass_kg": 500.0,
                },
                "cases": [
                    {"label": "helio", "sweep": "helio_1au_optical_sweep.yaml"},
                    {"label": "geo", "sweep": "geo_optical_sweep.yaml"},
                ],
            }
        ),
        encoding="utf-8",
    )
    runs = tmp / "runs"
    for case in ("helio_1au_optical", "geo_optical"):
        plan = load_sweep(cfg_dir / f"{case}_sweep.yaml")
        out = prepare_sweep_dir(plan, runs / plan.definition.name)
        run_sweep_points(plan, out, (0, 1), summary_only=False)
    return cmp_file


@pytest.fixture(scope="module")
def tiny_comparison(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One tiny real comparison (configs + saved sweep runs) shared by the compare tests."""
    return _tiny_comparison(tmp_path_factory.mktemp("tiny_comparison"))


def _trapezoid_weights_s(n: int, dt: float) -> np.ndarray:
    """Sample weights of the metrics module docstring: dt inside, dt/2 at both ends."""
    w = np.full(n, dt)
    w[0] = w[-1] = 0.5 * dt
    return w


@pytest.mark.slow
def test_compare_writes_tables_from_the_saved_runs(tiny_comparison: Path, tmp_path: Path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(tiny_comparison.parent.parent, root)
    tmp_path = root
    cmp_file = root / "configs" / "comparison.yaml"
    plan = load_comparison(cmp_file)
    out = write_comparison(plan, tmp_path / "runs", tmp_path / "report")
    rows = {(r.case, r.point): r for r in out.rows}
    assert {k: r.status for k, r in rows.items()} == {
        ("helio", "p0000"): "ok",
        ("helio", "p0001"): "ok",
        ("geo", "p0000"): "ok",
        ("geo", "p0001"): "failed",
    }
    # 100 km receiver at the GEO range: the paraxial condition of the Fresnel-Hankel model fails.
    assert "paraxial" in (rows[("geo", "p0001")].error or "")
    assert [r.is_base for r in out.rows] == [True, False, True, False]
    # Values come from the saved summaries (independent read) and closed forms.
    for case, sweep in (("helio", "tiny_helio_1au_optical"), ("geo", "tiny_geo_optical")):
        r = rows[(case, "p0000")]
        s = read_summary(tmp_path / "runs" / sweep / "points" / "p0000" / "summary.json")
        assert r.delivered_mean_power_W == s.p_grid_W.time_mean
        assert r.availability == s.availability
        assert r.capture_energy_weighted == pytest.approx(
            s.stage_energy_J["captured_J"] / s.stage_energy_J["atm_J"], rel=1e-15
        )
        mass = 500.0 + 1.0 * 10000.0 + 5.0 * s.thermal.radiator_area_m2_max + 50.0 * math.pi * 25.0
        assert r.mass_proxy_kg == pytest.approx(mass, rel=1e-14)
        assert r.mass_per_delivered_kW_kg == pytest.approx(mass / (s.p_grid_W.time_mean / 1e3))
        assert r.range_mean_m == s.geometry.range_m.mean
        # The remaining table metrics, recomputed from the saved timeseries.parquet with the
        # trapezoid weights of the metrics module (independent of compare.py and of the
        # summary): umbra = ECLIPSE bit, penumbra = PARTIAL_ECLIPSE bit (time fractions);
        # capacity factor = mean P_grid / rated power; end-to-end = E_grid / E_incident;
        # peak irradiance = max peak density while transmitting; mean intercepted flux =
        # E_captured / (t_available pi R_rx^2). Tolerance 1e-12 relative (summation order).
        ts = pq.read_table(
            tmp_path / "runs" / sweep / "points" / "p0000" / "timeseries.parquet"
        ).to_pydict()
        code = np.asarray(ts["reason_code"], dtype=np.uint32)
        w = _trapezoid_weights_s(code.size, TINY_STEP_S)
        window = float(np.sum(w))
        frac = np.asarray(ts["illumination_fraction"])
        umbra = float(np.sum(w[(code & np.uint32(int(Unavailability.ECLIPSE))) != 0])) / window
        pen_bit = np.uint32(int(Unavailability.PARTIAL_ECLIPSE))
        pen = float(np.sum(w[(code & pen_bit) != 0])) / window
        assert r.eclipse_umbra_fraction == pytest.approx(umbra, rel=1e-12, abs=0.0)
        assert r.eclipse_penumbra_fraction == pytest.approx(pen, rel=1e-12, abs=0.0)
        # ... and the flags are the illumination geometry: umbra f = 0, penumbra 0 < f < 1.
        assert umbra == pytest.approx(float(np.sum(w[frac <= 0.0])) / window, rel=1e-15)
        assert pen == pytest.approx(float(np.sum(w[(frac > 0) & (frac < 1)])) / window, rel=1e-15)
        grid = np.asarray(ts["grid_W"])
        e_grid = float(np.sum(w * grid))
        assert r.delivered_mean_power_W == pytest.approx(e_grid / window, rel=1e-12)
        assert r.capacity_factor == pytest.approx(
            (e_grid / window) / s.capacity_factor_denominator.rated_power_W, rel=1e-12
        )
        assert r.rated_power_W == s.capacity_factor_denominator.rated_power_W
        e_inc = float(np.sum(w * np.asarray(ts["incident_solar_W"])))
        assert r.end_to_end_efficiency == pytest.approx(e_grid / e_inc, rel=1e-12)
        tx = np.asarray(ts["transmitting"], dtype=bool)
        dens = np.asarray(ts["peak_power_density_W_m2"])
        assert r.peak_irradiance_max_W_m2 == pytest.approx(float(np.max(dens[tx])), rel=1e-12)
        avail = np.asarray(ts["available"], dtype=bool)
        e_capt = float(np.sum(w * np.asarray(ts["captured_W"])))
        flux = e_capt / (float(np.sum(w[avail])) * math.pi * 100.0**2)
        assert r.mean_intercepted_flux_W_m2 == pytest.approx(flux, rel=1e-12)
        if case == "geo":
            # The window crosses the 03-21 equinox eclipse: both fractions are non-trivial
            # (umbra ~67 of 180 min, two ~2 min penumbra transits; one 120 s sample each at
            # least), so a swapped or zero mapping fails.
            assert 0.30 < umbra < 0.45
            assert pen > 0.0
            assert r.capacity_factor < 0.7
        else:
            assert umbra == 0.0
            assert pen == 0.0
    geo, helio = rows[("geo", "p0000")], rows[("helio", "p0000")]
    assert geo.architecture == "geostationary"
    assert helio.architecture == "heliocentric"
    assert geo.range_mean_m < 4.0e7 < 1.0e10 < (helio.range_mean_m or 0.0)
    # Files: tables, assumptions, figure, manifest with the summary hashes.
    for name in (
        "comparison_points.csv",
        "comparison_points.md",
        "comparison_baseline.csv",
        "comparison_baseline.md",
        "harmonised_assumptions.csv",
        "harmonised_assumptions.md",
        "comparison.png",
        "comparison_manifest.json",
    ):
        assert (tmp_path / "report" / name).is_file()
    with (tmp_path / "report" / "comparison_baseline.csv").open(encoding="utf-8") as fh:
        base_rows = list(csv.DictReader(fh))
    assert [b["case"] for b in base_rows] == ["helio", "geo"]
    man = json.loads((tmp_path / "report" / "comparison_manifest.json").read_text())
    assert sum(1 for p in man["points"] if p["summary_sha256"]) == 3
    md = (tmp_path / "report" / "harmonised_assumptions.md").read_text(encoding="utf-8")
    assert "| collector.area_m2 | m^2 | assumed | shared by all cases |" in md
    # The Markdown table shows the configured values exactly (M7b review: a 4-significant-
    # figure rounding printed 365.25 days as 365.2): every value cell equals the CSV's.
    with (tmp_path / "report" / "harmonised_assumptions.csv").open(encoding="utf-8") as fh:
        csv_rows = {row["path"]: row for row in csv.DictReader(fh)}
    md_rows = {
        cells[0]: cells
        for line in md.splitlines()
        if line.startswith("| ") and not line.startswith("| path")
        for cells in [[c.strip() for c in line.strip("|").split("|")]]
    }
    checked = 0
    for path, row in csv_rows.items():
        if path not in md_rows:
            continue
        for j, label in enumerate(("helio", "geo")):
            value = row.get(label, "")
            try:
                float(value)
            except ValueError:
                continue
            assert md_rows[path][4 + j] == value, (path, label)
            checked += 1
    assert checked > 20
    assert "| 0.125 | 0.125 |" in md  # time.duration_days of the tiny window, verbatim
    # CLI: exit 0 with every point present; 1 once a point directory is missing.
    assert (
        main(
            [
                "compare",
                str(cmp_file),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--out",
                str(tmp_path / "report2"),
            ]
        )
        == 0
    )
    shutil.rmtree(tmp_path / "runs" / "tiny_geo_optical" / "points" / "p0000")
    assert (
        main(
            [
                "compare",
                str(cmp_file),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--out",
                str(tmp_path / "report3"),
            ]
        )
        == 1
    )


@pytest.mark.slow
def test_compare_refuses_a_stale_run_directory(tiny_comparison: Path, tmp_path: Path) -> None:
    shutil.copytree(tiny_comparison.parent.parent, tmp_path / "copy")
    tmp_path = tmp_path / "copy"
    cmp_file = tmp_path / "configs" / "comparison.yaml"
    # Change the shared technology after the runs: the saved summaries no longer match.
    shared = yaml.safe_load((cmp_file.parent / "_shared.yaml").read_text(encoding="utf-8"))
    shared["collector"]["area_m2"] = 20000
    (cmp_file.parent / "_shared.yaml").write_text(yaml.safe_dump(shared), encoding="utf-8")
    with pytest.raises(ValueError, match="stale or foreign run directory"):
        write_comparison(load_comparison(cmp_file), tmp_path / "runs", tmp_path / "report")
