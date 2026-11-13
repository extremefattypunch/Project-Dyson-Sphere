"""ERA5 weather in end-to-end runs (M8a; SPICE, short windows, synthetic ERA5 files).

The weather directory is populated through the real downloader with a fake CDS client
(:mod:`era5_helpers`) whose total cloud cover is a known function of the UTC hour. The expected
outage of every accessible epoch is recomputed independently: arrival TDB -> UTC with SPICE
``et2utc`` (string), parsed with ``datetime`` and rounded to the nearest hour, then the known
cloud function -- a path that shares no code with :mod:`helioswarm.weather`'s ``deltet`` table.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import spiceypy

from helioswarm.config import RunConfig, parse_config, read_yaml
from helioswarm.constellation import run_constellation
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.io import execute_run, prepare_run_dir
from helioswarm.kernels import KernelSet
from helioswarm.system import SystemRunResult, Unavailability, run_single_satellite
from helioswarm.weather import WEATHER_FILE, build_weather, weather_generator

from config_helpers import CONFIG_DIR, MVP_CONFIG
from era5_helpers import make_weather_dir

pytestmark = pytest.mark.spice
pytest.importorskip("xarray")

DAYS = 2.0


def _odd_hours_cloudy(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """tcc = 0.9 at odd UTC hours, 0.1 at even hours (every node)."""
    return np.where(((t // 3600) % 24) % 2 == 1, 0.9, 0.1) + 0.0 * lat + 0.0 * lon


def _all(value: float):  # type: ignore[no-untyped-def]
    def f(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.full(t.shape, value) + 0.0 * lat + 0.0 * lon

    return f


def _mvp(era5: bool = True, start_utc: str | None = None, **weather: Any) -> RunConfig:
    data = copy.deepcopy(read_yaml(MVP_CONFIG))
    data["time"]["duration_days"] = DAYS
    if start_utc is not None:
        data["time"]["start_utc"] = start_utc
    if era5:
        data["models"]["weather_source"] = "era5"
        w: dict[str, Any] = {
            "era5_first_year": 2014,
            "era5_last_year": 2015,
            "cloud_cover_threshold": 0.5,
            "historical_year": 2015,
        }
        w.update(weather)
        data["weather"] = w
    return parse_config(data)


@pytest.fixture(scope="module")
def odd_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("weather_odd")
    make_weather_dir(d, [("example_site", 0.0, 0.0)], 2014, 2015, _odd_hours_cloudy)
    return d


def _run(ks: KernelSet, cfg: RunConfig, directory: Path | None) -> SystemRunResult:
    with ks.loaded(verify=False) as k:
        weather = None
        if directory is not None:
            from helioswarm.timescale import time_grid_from_config

            g = time_grid_from_config(cfg.time, k)
            weather = build_weather(
                cfg,
                k,
                tdb_window_s=(float(g.epochs_tdb_s[0]), float(g.epochs_tdb_s[-1])),
                rng=weather_generator(cfg.project.seed),
                directory=directory,
            )
        return run_single_satellite(
            cfg,
            ephemeris=SpiceEphemeris(k),
            kernels=k,
            rng=np.random.default_rng(1),
            weather=weather,
        )


@pytest.fixture(scope="module")
def clear_run(kernel_set: KernelSet) -> SystemRunResult:
    return _run(kernel_set, _mvp(era5=False), None)


def _utc_seconds_of_hour(ks: KernelSet, tdb_s: np.ndarray) -> np.ndarray:
    """UTC seconds past the hour of TDB epochs (SPICE ``et2utc`` string path)."""
    with ks.loaded(verify=False):
        out = []
        for et in tdb_s:
            t = dt.datetime.fromisoformat(spiceypy.et2utc(float(et), "ISOC", 3))
            out.append(60.0 * t.minute + t.second + t.microsecond * 1e-6)
    return np.array(out)


def _expected_odd_outage(ks: KernelSet, arrival_tdb_s: np.ndarray) -> np.ndarray:
    with ks.loaded(verify=False):
        out = []
        for et in arrival_tdb_s:
            s = spiceypy.et2utc(float(et), "ISOC", 3)
            t = dt.datetime.fromisoformat(s).replace(tzinfo=dt.UTC)
            hour = (t + dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0).hour
            out.append(hour % 2 == 1)
    return np.array(out)


def test_default_scenario_path_reproduces_the_pre_m8a_content_hash(
    clear_run: SystemRunResult,
) -> None:
    """Regression pin of the default (``weather_source = scenario``) path: the 2-day MVP
    time-series content hash. Obtained with ``scripts/m8a_default_path_pin.py`` (same config,
    ``default_rng(1)``) both with the M7 code (``git archive e86f8e7``) and with the M8a code
    (docs/milestones/M8a.md D); any change of the default physics or table layout changes it."""
    from helioswarm.io import table_content_sha256, timeseries_table

    assert clear_run.config.models.weather_source == "scenario"
    assert table_content_sha256(timeseries_table(clear_run)) == (
        "34c97c9fa1f94a5161fd2fe188a4f82cff25375c3a4d164ee4f77bba04147e2b"
    )


@pytest.mark.parametrize(
    ("start_utc", "case"),
    [
        # Arrival ~hh:29:32 UTC (light time ~572 s): less than TDB - UTC (69.184 s in 2030)
        # before the half hour, so TDB used as UTC, or DeltaET with the wrong sign, rounds to
        # hh + 1 instead of hh.
        ("2030-01-01T00:20:00", "tdb_vs_utc"),
        # Emission hh:25 (rounds to hh), arrival ~hh:34:32 (rounds to hh + 1): the weather
        # hour must come from the arrival epoch, not the emission (grid) epoch.
        ("2030-01-01T00:25:00", "arrival_vs_emission"),
    ],
)
def test_weather_hour_uses_the_arrival_utc_epoch(
    kernel_set: KernelSet, odd_dir: Path, start_utc: str, case: str
) -> None:
    res = _run(kernel_set, _mvp(start_utc=start_utc), odd_dir)
    vis = res.access.visible
    assert vis.sum() > 10
    arrival = _utc_seconds_of_hour(kernel_set, res.access.t_arrival_tdb_s[vis])
    emission = _utc_seconds_of_hour(kernel_set, res.access.t_emit_tdb_s[vis])
    # Preconditions that make the case discriminating (checked, not assumed).
    if case == "tdb_vs_utc":
        assert np.all((arrival > 1800.0 - 66.0) & (arrival < 1800.0)), arrival
    else:
        assert np.all(emission < 1800.0), emission
        assert np.all(arrival > 1800.0), arrival
    weather_bit = (res.reason_code & np.uint32(int(Unavailability.WEATHER))) != 0
    expected = np.zeros(vis.size, dtype=bool)
    expected[vis] = _expected_odd_outage(kernel_set, res.access.t_arrival_tdb_s[vis])
    np.testing.assert_array_equal(weather_bit, expected)
    assert 0 < expected.sum() < vis.sum()


def test_era5_outage_hits_exactly_the_cloudy_arrival_hours(
    kernel_set: KernelSet, clear_run: SystemRunResult, odd_dir: Path
) -> None:
    res = _run(kernel_set, _mvp(), odd_dir)
    vis = clear_run.access.visible
    assert vis.sum() > 10  # the 2-day MVP window has ~20 accessible epochs
    weather_bit = (res.reason_code & np.uint32(int(Unavailability.WEATHER))) != 0
    expected = np.zeros(vis.size, dtype=bool)
    expected[vis] = _expected_odd_outage(kernel_set, res.access.t_arrival_tdb_s[vis])
    np.testing.assert_array_equal(weather_bit, expected)
    assert 0 < expected.sum() < vis.sum()  # both clear and cloudy hours occur
    # Cloudy epochs deliver nothing; clear epochs deliver exactly the clear-sky run's power.
    np.testing.assert_array_equal(res.link.grid_W[expected], 0.0)
    np.testing.assert_array_equal(res.link.grid_W[~expected], clear_run.link.grid_W[~expected])
    assert res.weather is not None
    assert res.weather.year_mapping == {2030: 2015}  # window + 1 day stays in 2030


def test_era5_all_clear_reproduces_the_scenario_run_and_all_cloudy_blocks(
    kernel_set: KernelSet, clear_run: SystemRunResult, tmp_path: Path
) -> None:
    clear_dir = tmp_path / "clear"
    make_weather_dir(clear_dir, [("example_site", 0.0, 0.0)], 2014, 2015, _all(0.0))
    res = _run(kernel_set, _mvp(), clear_dir)
    np.testing.assert_array_equal(res.reason_code, clear_run.reason_code)
    np.testing.assert_array_equal(res.link.grid_W, clear_run.link.grid_W)
    cloudy_dir = tmp_path / "cloudy"
    make_weather_dir(cloudy_dir, [("example_site", 0.0, 0.0)], 2014, 2015, _all(1.0))
    res = _run(kernel_set, _mvp(), cloudy_dir)
    assert not np.any(res.available)
    vis = clear_run.access.visible
    np.testing.assert_array_equal(
        (res.reason_code & np.uint32(int(Unavailability.WEATHER))) != 0, vis
    )
    # The threshold is strict: tcc exactly at the threshold is clear.
    res = _run(kernel_set, _mvp(cloud_cover_threshold=1.0), cloudy_dir)
    np.testing.assert_array_equal(res.link.grid_W, clear_run.link.grid_W)


def test_bernoulli_mode_is_seeded_and_between_the_extremes(
    kernel_set: KernelSet, clear_run: SystemRunResult, odd_dir: Path
) -> None:
    cfg = _mvp(outage_model="bernoulli", cloud_cover_threshold=None)
    a = _run(kernel_set, cfg, odd_dir)
    b = _run(kernel_set, cfg, odd_dir)
    np.testing.assert_array_equal(a.reason_code, b.reason_code)  # same seed -> same weather
    vis = clear_run.access.visible
    blocked = (a.reason_code & np.uint32(int(Unavailability.WEATHER))) != 0
    assert 0 < blocked.sum() < vis.sum()


def test_cli_run_writes_weather_provenance(
    kernel_set: KernelSet, odd_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HELIOSWARM_WEATHER_DIR", str(odd_dir))
    cfg = _mvp()
    run_dir = prepare_run_dir(tmp_path / "run", overwrite=False)
    out = execute_run(cfg, run_dir, kernel_set=kernel_set, make_figures=False)
    w = json.loads((run_dir / WEATHER_FILE).read_text())
    assert w["dataset_doi"] == "10.24381/1cf1ad76"  # the time-series product (default)
    assert w["licence_spdx"] == "CC-BY-4.0"
    assert w["simulated_to_historical_year"] == {"2030": 2015}
    assert w["sites"]["example_site"]["node_key"] == "N00.00_E000.00"
    assert len(w["sites"]["example_site"]["file_sha256"]) == 1  # one 2014-2015 time-series file
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert WEATHER_FILE in manifest["outputs"]
    assert any("models.weather_source" in a for a in manifest["approximations"])
    assert out.summary.unavailability.blocking["WEATHER"].time_fraction_flagged > 0.0


def test_constellation_stage_a_applies_site_weather(
    kernel_set: KernelSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = copy.deepcopy(read_yaml(CONFIG_DIR / "constellation_demo.yaml"))
    d["time"]["duration_days"] = 40 * 3896.0 / 86400.0
    d["constellation"]["rings"][0]["n_satellites"] = 3
    d["models"]["weather_source"] = "era5"
    d["weather"] = {
        "era5_first_year": 2014,
        "era5_last_year": 2015,
        "cloud_cover_threshold": 0.5,
        "historical_year": 2014,
    }
    cfg = parse_config(d)
    sites = [(s.name, s.latitude_deg, s.longitude_deg) for s in cfg.receiver.sites]

    def site_lon120w_cloudy(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.where(np.abs(lon + 120.0) < 1.0, 1.0, 0.0) + 0.0 * t

    make_weather_dir(tmp_path, sites, 2014, 2015, site_lon120w_cloudy)
    monkeypatch.setenv("HELIOSWARM_WEATHER_DIR", str(tmp_path))
    with kernel_set.loaded(verify=False) as k:
        res = run_constellation(
            cfg, ephemeris=SpiceEphemeris(k), kernels=k, rng=np.random.default_rng(cfg.project.seed)
        )
    code = res.links.reason_code  # (T, N, S)
    w = (code & np.uint32(int(Unavailability.WEATHER))) != 0
    geometry = np.uint32(
        int(Unavailability.BELOW_MIN_ELEVATION)
        | int(Unavailability.EARTH_OCCULTATION)
        | int(Unavailability.SUN_OCCULTATION)
        | int(Unavailability.MOON_OCCULTATION)
        | int(Unavailability.SUN_AVOIDANCE)
    )
    access = (code & geometry) == 0
    j = [s.name for s in cfg.receiver.sites].index("site_lon120w")
    others = [i for i in range(len(sites)) if i != j]
    assert not w[:, :, others].any()  # clear sites are never blocked by weather
    assert access[:, :, j].any()
    # At the overcast site exactly the links with geometric access carry the WEATHER bit.
    np.testing.assert_array_equal(w[:, :, j], access[:, :, j])
    assert np.all(res.links.captured_W[:, :, j] == 0.0)
    assert np.all(res.site_grid_W[:, j] == 0.0)
    assert res.site_grid_W[:, others].sum() > 0.0
