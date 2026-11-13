"""``helioswarm weather download|climatology`` end to end on synthetic ERA5 files (M8a review).

No network: the CDS client is :class:`era5_helpers.FakeCdsClient`. The climatology numbers are
checked against hand counts of the known synthetic cloud pattern (stated in each assertion).
"""

from __future__ import annotations

import datetime as dt
import functools
import json
from pathlib import Path

import numpy as np
import pytest

import helioswarm.era5 as era5
from helioswarm.cli import main
from helioswarm.era5 import build_requests, download_requests, load_download_config
from helioswarm.weather import climatology_command

from era5_helpers import FakeCdsClient

pytest.importorskip("xarray")

HOURS = 2 * 8760  # 2014 + 2015 (neither is a leap year)


def _tcc(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Site A (lat > 0): overcast in January and at 00-05 UTC; site B (lat < 0, lon > 0):
    overcast at 03-08 UTC; site C (lon < 0): always clear. 0.1 elsewhere (clear at c = 0.5)."""
    hour = (t // 3600) % 24
    month = np.array(
        [dt.datetime.fromtimestamp(int(x), tz=dt.UTC).month for x in t.ravel()]
    ).reshape(t.shape)
    a = (month == 1) | (hour <= 5)
    b = (hour >= 3) & (hour <= 8)
    cloudy = np.where(lat > 0, a, np.where(lon > 0, b, False))
    return np.where(cloudy, 0.9, 0.1)


def _download_config(tmp_path: Path) -> Path:
    path = tmp_path / "era5.yaml"
    path.write_text(
        "era5_schema_version: 1\n"
        "dataset: reanalysis-era5-single-levels-timeseries\n"
        "variables: [total_cloud_cover]\n"
        "first_year: 2014\n"
        "last_year: 2015\n"
        "sites:\n"
        "  - {name: site_a, latitude_deg: 10.0, longitude_deg: 20.0, origin: test}\n"
        "  - {name: site_b, latitude_deg: -10.0, longitude_deg: 20.0, origin: test}\n"
        "  - {name: site_c, latitude_deg: -10.0, longitude_deg: -20.0, origin: test}\n"
    )
    return path


def test_weather_download_cli_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = _download_config(tmp_path)
    wdir = tmp_path / "weather"
    fake = FakeCdsClient(_tcc)
    monkeypatch.setattr(
        era5,
        "download_requests",
        functools.partial(download_requests, client_factory=lambda: fake),
    )
    base = ["weather", "download", str(cfg), "--weather-dir", str(wdir)]
    assert main([*base, "--dry-run"]) == 0
    assert fake.requests == []  # dry run: nothing submitted
    assert "3 request(s)" in capsys.readouterr().out
    assert main([*base, "--site-index", "7"]) == 1  # out of range
    assert main([*base, "--site", "nowhere"]) == 1
    assert main([*base, "--site", "site_b"]) == 0
    assert len(fake.requests) == 1
    assert main(base) == 0
    assert len(fake.requests) == 3  # site_b was already cached and verified
    out = capsys.readouterr().out
    assert "(2 downloaded)" in out
    assert main(base) == 0
    assert len(fake.requests) == 3  # idempotent
    assert "(0 downloaded)" in capsys.readouterr().out


def test_climatology_command_against_hand_counts(tmp_path: Path) -> None:
    cfg = _download_config(tmp_path)
    wdir = tmp_path / "weather"
    retrieved = dt.datetime(2019, 6, 1, 12, tzinfo=dt.UTC)
    download_requests(
        build_requests(load_download_config(cfg)),
        wdir,
        client_factory=lambda: FakeCdsClient(_tcc),
        now_utc=lambda: retrieved,
    )
    out = tmp_path / "clim"
    args = ["weather", "climatology", str(cfg), "--out", str(out), "--threshold", "0.5"]
    assert main([*args, "--weather-dir", str(wdir)]) == 0
    assert main([*args, "--weather-dir", str(wdir)]) == 1  # refuses to overwrite
    text = (out / "climatology.json").read_text()

    def _no_nan(token: str) -> float:
        raise AssertionError(f"non-standard JSON token {token}")

    doc = json.loads(text, parse_constant=_no_nan)  # strict JSON: NaN written as null
    sites = {d["site_name"]: d for d in doc["sites"]}
    # Site A cloudy hours: all of January (2 x 744) + 6 h on each of the other 2 x 334 days.
    a_cloudy = 2 * 744 + 6 * 2 * 334
    assert sites["site_a"]["clear_fraction"] == pytest.approx(1 - a_cloudy / HOURS, abs=1e-12)
    assert sites["site_b"]["clear_fraction"] == pytest.approx(18 / 24, abs=1e-12)
    assert sites["site_c"]["clear_fraction"] == 1.0
    assert sites["site_a"]["clear_fraction_by_month"][0] == 0.0  # January
    assert sites["site_a"]["clear_fraction_by_month"][1] == pytest.approx(18 / 24, abs=1e-12)
    # Bernoulli expectation 1 - mean(tcc) (float32 storage of 0.1 / 0.9: 1e-7 tolerance).
    assert sites["site_b"]["bernoulli_clear_fraction"] == pytest.approx(
        1 - (0.9 * 6 + 0.1 * 18) / 24, abs=1e-6
    )
    (joint,) = doc["joint"]
    # All three cloudy never happens (site C is always clear); A and B both cloudy: 03-08 UTC
    # in January (6 h x 62 d) and 03-05 UTC on the other days (3 h x 668 d).
    assert joint["all_cloudy_fraction"] == 0.0
    # All clear: outside January, 09-23 UTC (A clear 06-23, B clear outside 03-08): 15 h x 668 d.
    assert joint["all_clear_fraction"] == pytest.approx(15 * 668 / HOURS, abs=1e-12)
    corr = joint["pairwise_outage_correlation"]
    assert corr["site_a|site_c"] is None  # constant outage flag of C: undefined -> null
    assert corr["site_b|site_c"] is None
    assert isinstance(corr["site_a|site_b"], float)
    # Attribution year = the data's retrieval year (provenance), not the wall clock.
    assert "2019" in doc["attribution"]
    assert doc["retrieved_utc"] == ["2019-06-01T12:00:00Z"]
    monthly = __import__("pandas").read_parquet(out / "climatology_monthly.parquet")
    row = monthly[(monthly.site == "site_a") & (monthly.month == 1)]
    assert row.clear_fraction.tolist() == [0.0]
    # The function and the CLI agree (re-run into a new directory).
    path = climatology_command(
        cfg,
        tmp_path / "clim2",
        thresholds=[0.5],
        first_year=None,
        last_year=None,
        interpolation="nearest",
        directory=wdir,
        overwrite=False,
    )
    assert json.loads(path.read_text())["sites"] == doc["sites"]
