"""ERA5 weather availability model (helioswarm.weather; M8a): time mapping, outage models,
climatology and joint statistics, configuration rules and the hash-stability rule.

References are closed forms or hand counts on synthetic series; the UTC conversion is checked
against SPICE ``et2utc`` (the LSK's leap-second table).
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helioswarm.atmosphere import AtmosphereResult
from helioswarm.config import (
    ConfigError,
    WeatherConfig,
    approximations_in_force,
    config_hash,
    load_config,
    parse_config,
)
from helioswarm.kernels import LoadedKernels
from helioswarm.timescale import tdb_s_to_utc, utc_to_tdb_s
from helioswarm.weather import (
    SiteWeather,
    UtcOffsetTable,
    WeatherError,
    WeatherProvider,
    calendar_fields,
    historical_hour_posix,
    joint_statistics,
    nearest_hour_posix,
    outage_from_cloud_cover,
    site_climatology,
    tdb_to_utc_posix_exact,
    weather_generator,
    year_map,
)

from config_helpers import MVP_CONFIG, set_path

pytest.importorskip("xarray")


def _posix(*args: int) -> int:
    return int(dt.datetime(*args, tzinfo=dt.UTC).timestamp())


def _ws(**kw: Any) -> WeatherConfig:
    base: dict[str, Any] = {
        "era5_first_year": 2014,
        "era5_last_year": 2023,
        "outage_model": "threshold",
        "cloud_cover_threshold": 0.5,
        "year_mapping": "fixed",
        "historical_year": 2015,
    }
    base.update(kw)
    return WeatherConfig.model_validate(base)


# ---- time ----------------------------------------------------------------------------------------


def test_nearest_hour_rounding_edges() -> None:
    h0 = _posix(2030, 1, 1, 0)
    t = np.array([h0 + 0.0, h0 + 1799.999, h0 + 1800.0, h0 - 1800.0, h0 - 1800.001])
    np.testing.assert_array_equal(
        nearest_hour_posix(t), [h0, h0, h0 + 3600, h0, h0 - 3600]
    )  # half hour rounds up (to the later analysis)


def test_calendar_fields_against_datetime() -> None:
    rng = np.random.default_rng(3)
    hours = _posix(1999, 1, 1) + 3600 * rng.integers(0, 40 * 8766, size=500)
    y, m, d, h = calendar_fields(hours)
    for k in range(hours.size):
        ref = dt.datetime.fromtimestamp(int(hours[k]), tz=dt.UTC)
        assert (y[k], m[k], d[k], h[k]) == (ref.year, ref.month, ref.day, ref.hour)


def test_fixed_year_mapping_is_cyclic_and_keeps_calendar_position() -> None:
    s = _ws(historical_year=2022)
    mp = year_map(s, [2030, 2031, 2032], np.random.default_rng(0))
    assert mp == {2030: 2022, 2031: 2023, 2032: 2014}  # wraps inside 2014-2023
    t = np.array([_posix(2030, 7, 4, 13, 10), _posix(2031, 1, 1, 0, 40)], dtype=np.float64)
    out = historical_hour_posix(t, mp)
    assert out[0] == _posix(2022, 7, 4, 13)  # 13:10 -> 13:00
    assert out[1] == _posix(2023, 1, 1, 1)  # 00:40 -> 01:00 in the next mapped year


def test_leap_day_maps_to_feb_28_of_a_non_leap_year() -> None:
    mp = {2032: 2015, 2033: 2016}  # 2032 leap -> 2015 not; 2033 not -> 2016 leap
    t = np.array(
        [_posix(2032, 2, 29, 5), _posix(2032, 3, 1, 5), _posix(2033, 2, 28, 5)], dtype=float
    )
    out = historical_hour_posix(t, mp)
    assert out.tolist() == [_posix(2015, 2, 28, 5), _posix(2015, 3, 1, 5), _posix(2016, 2, 28, 5)]
    with pytest.raises(WeatherError, match="no historical year"):
        historical_hour_posix(np.array([float(_posix(2040, 1, 1))]), mp)
    assert calendar.isleap(2032)
    assert not calendar.isleap(2015)


def test_sampled_years_are_deterministic_per_seed_and_cover_the_record() -> None:
    s = _ws(year_mapping="sampled", historical_year=None)
    years = list(range(2030, 2130))
    a = year_map(s, years, weather_generator(7))
    b = year_map(s, years, weather_generator(7))
    c = year_map(s, years, weather_generator(8))
    assert a == b
    assert a != c
    assert set(a.values()) == set(range(2014, 2024))  # 100 draws hit all 10 years (p ~ 3e-4 miss)
    # The weather stream is independent of the run Generator seeded with the same seed.
    assert weather_generator(7).random() != np.random.default_rng(7).random()


def test_tdb_to_utc_matches_spice_including_a_leap_second(spice_kernels: LoadedKernels) -> None:
    labels = ["2030-01-01T00:00:00", "2016-12-31T23:59:59", "2017-01-01T00:00:00"]
    tdb = np.array([utc_to_tdb_s(s, spice_kernels) for s in labels])
    exact = tdb_to_utc_posix_exact(tdb, spice_kernels)
    for k, s in enumerate(labels):
        ref = dt.datetime.fromisoformat(s).replace(tzinfo=dt.UTC).timestamp()
        assert exact[k] == pytest.approx(ref, abs=1e-6)  # float64 TDB seconds: ~1e-7 s
    # Hourly table + linear interpolation: < 1 ms away from leap seconds (TDB-TT periodic term
    # varies ~1e-9 s/s), and the smeared 1 s step within one hour of the leap second.
    window = (utc_to_tdb_s("2016-12-30T00:00:00", spice_kernels), tdb.max() + 86400.0)
    table = UtcOffsetTable.build(*window, spice_kernels)
    leap = utc_to_tdb_s("2017-01-01T00:00:00", spice_kernels)
    probe = np.concatenate(
        [np.linspace(window[0], window[1] - 3600.0, 2001), leap + np.linspace(-3500, 3500, 71)]
    )
    err = table.utc_posix_s(probe) - tdb_to_utc_posix_exact(probe, spice_kernels)
    near = np.abs(probe - leap) < 3600.0
    assert np.max(np.abs(err[~near])) < 1e-3
    assert np.max(np.abs(err[near])) <= 1.0 + 1e-6
    assert tdb_s_to_utc(float(tdb[0]), spice_kernels, decimals=0) == "2030-01-01T00:00:00"


# ---- outage models -------------------------------------------------------------------------------


def test_threshold_outage_edges() -> None:
    tcc = np.array([0.0, 0.3, 0.5, 0.5000001, 1.0])
    out = outage_from_cloud_cover(tcc, _ws(cloud_cover_threshold=0.5), None)
    assert out.tolist() == [False, False, False, True, True]  # equal to threshold is clear
    assert outage_from_cloud_cover(tcc, _ws(cloud_cover_threshold=0.0), None).tolist() == [
        False,
        True,
        True,
        True,
        True,
    ]


def test_bernoulli_outage_frequency_is_tcc() -> None:
    s = _ws(outage_model="bernoulli", cloud_cover_threshold=None)
    tcc = np.repeat([0.0, 0.2, 0.7, 1.0], 20000)
    out = outage_from_cloud_cover(tcc, s, weather_generator(1))
    freq = out.reshape(4, -1).mean(axis=1)
    # Binomial standard error sqrt(p(1-p)/n) <= 0.0036 at n = 20000; 5 sigma tolerance.
    np.testing.assert_allclose(freq, [0.0, 0.2, 0.7, 1.0], atol=0.018)
    assert freq[0] == 0.0
    assert freq[3] == 1.0
    same = outage_from_cloud_cover(tcc, s, weather_generator(1))
    np.testing.assert_array_equal(out, same)
    with pytest.raises(WeatherError, match="Generator"):
        outage_from_cloud_cover(tcc, s, None)


# ---- provider ------------------------------------------------------------------------------------


def _provider(
    cloud: np.ndarray, start: int, table: UtcOffsetTable | None = None
) -> WeatherProvider:
    s = _ws(historical_year=2015)
    sw = SiteWeather(
        site_name="x",
        latitude_deg=0.0,
        longitude_deg=0.0,
        node_key="N00.00_E000.00",
        interpolation="nearest",
        start_posix_s=start,
        cloud_cover=cloud,
        outage=outage_from_cloud_cover(cloud, s, None),
        files=(),
        file_sha256=(),
        cds_request_ids=(),
        retrieved_utc=("2026-11-13T00:00:00Z",),
    )
    # Identity UTC table: TDB offset 0 so that "TDB" = UTC - J2000 exactly.
    j2000 = 946_728_000
    t = np.array([start - j2000 - 7200.0, start - j2000 + 3600.0 * cloud.size + 7200.0])
    if table is None:
        table = UtcOffsetTable(tdb_s=t, delta_et_s=np.zeros(2))
    return WeatherProvider(
        settings=s, sites={"x": sw}, year_mapping={2015: 2015}, utc_table=table, weather_dir="-"
    )


def test_provider_applies_outage_only_at_cloudy_hours() -> None:
    start = _posix(2015, 1, 1)
    cloud = np.where(np.arange(48) % 2 == 1, 0.9, 0.1)  # odd hours cloudy
    prov = _provider(cloud, start)
    hours = np.arange(10, 30)
    tdb = (start - 946_728_000) + 3600.0 * hours + 600.0  # hh:10 -> hour hh
    el = np.full(hours.size, 0.8)
    atm = AtmosphereResult(
        transmittance=np.full(hours.size, 0.9),
        attenuation_dB=np.full(hours.size, -10 * math.log10(0.9)),
        airmass=np.full(hours.size, 1.4),
        weather_available=np.ones(hours.size, dtype=bool),
        in_validity=np.ones(hours.size, dtype=bool),
        elevation_rad=el,
        model="scenario",
        mode="optical",
        weather_scenario="clear",
    )
    out = prov.apply(atm, "x", tdb)
    cloudy = hours % 2 == 1
    np.testing.assert_array_equal(out.weather_available, ~cloudy)
    np.testing.assert_array_equal(out.transmittance, np.where(cloudy, 0.0, 0.9))
    assert np.all(np.isinf(out.attenuation_dB[cloudy]))
    _outage, tcc, _hist = prov.lookup("x", tdb)
    np.testing.assert_array_equal(tcc, np.where(cloudy, 0.9, 0.1))
    with pytest.raises(WeatherError):
        prov.lookup("x", np.array([tdb[0] + 400 * 86400.0]))


def test_provider_converts_arrival_tdb_to_utc_before_rounding(
    spice_kernels: LoadedKernels,
) -> None:
    """With the real DeltaET (67.184 s in January 2015), arrival epochs 40 s either side of
    hh:30 UTC round to hh and hh + 1. Treating TDB seconds as UTC (+67 s), flipping the sign
    of DeltaET (+134 s) or shifting the epoch by minutes (e.g. emission instead of arrival)
    moves at least one probe across the half hour, which flips the odd/even cloud pattern."""
    start = _posix(2015, 1, 1)
    cloud = np.where(np.arange(48) % 2 == 1, 0.9, 0.1)  # odd UTC hours cloudy
    labels = ["2015-01-01T05:29:20", "2015-01-01T05:30:40", "2015-01-01T06:29:20",
              "2015-01-01T06:30:40", "2015-01-01T06:31:30"]  # fmt: skip
    tdb = np.array([utc_to_tdb_s(x, spice_kernels) for x in labels])
    table = UtcOffsetTable.build(float(tdb.min()), float(tdb.max()), spice_kernels)
    # TT - UTC = 32.184 s + (TAI - UTC = 35 s until 2015-06-30); TDB - TT periodic < 2 ms.
    np.testing.assert_allclose(table.delta_et_s, 67.184, atol=2e-3)
    outage, tcc, hist = _provider(cloud, start, table).lookup("x", tdb)
    assert hist.tolist() == [
        _posix(2015, 1, 1, 5), _posix(2015, 1, 1, 6), _posix(2015, 1, 1, 6),
        _posix(2015, 1, 1, 7), _posix(2015, 1, 1, 7),
    ]  # fmt: skip
    assert outage.tolist() == [True, False, False, True, True]
    np.testing.assert_array_equal(tcc, [0.9, 0.1, 0.1, 0.9, 0.9])


# ---- climatology ---------------------------------------------------------------------------------


def test_site_climatology_counts() -> None:
    t = _posix(2015, 1, 1) + 3600 * np.arange(8760 + 8784, dtype=np.int64)
    y, m, _d, h = calendar_fields(t)
    # Cloudy (tcc = 1) in January and at 06 UTC every day; clear (tcc = 0) otherwise; 2016 also
    # cloudy on its first 100 hours of February.
    tcc = ((m == 1) | (h == 6)).astype(float)
    feb16 = np.flatnonzero((y == 2016) & (m == 2))[24:124]  # 2 Feb 00 UTC .. 6 Feb 03 UTC
    tcc[feb16] = 1.0
    c = site_climatology("s", 10.0, 90.0, "k", t, tcc, 0.5)
    assert c.n_hours == t.size
    assert c.clear_fraction_by_month[0] == 0.0
    # March: every hour except 06 UTC -> 23/24.
    assert c.clear_fraction_by_month[2] == pytest.approx(23 / 24, abs=1e-12)
    assert c.clear_fraction_by_utc_hour[6] == 0.0
    # Local mean solar time = UTC + 90/15 = UTC + 6 h: the 06 UTC clouds appear at 12 local.
    assert c.clear_fraction_by_local_solar_hour[12] == 0.0
    assert c.clear_fraction_by_local_solar_hour[6] > 0.9
    n_cloudy = int(np.sum(tcc > 0.5))
    assert c.clear_fraction == pytest.approx(1 - n_cloudy / t.size, abs=1e-12)
    assert c.bernoulli_clear_fraction == pytest.approx(1 - tcc.mean(), abs=1e-12)
    assert c.clear_fraction_by_year[2016] < c.clear_fraction_by_year[2015]
    # Longest cloudy run: January (744 h); 31 Dec 23 UTC and 1 Feb 00 UTC are clear and the
    # February block is separated from 1 Feb 06 UTC by clear hours.
    assert c.longest_outage_h == 744


def test_joint_statistics_against_hand_count() -> None:
    a = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=float)
    b = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=float)
    c = np.array([1, 1, 1, 0, 0, 0, 0, 1], dtype=float)
    js = joint_statistics(["a", "b", "c"], np.vstack([a, b, c]), 0.5)
    assert js.all_cloudy_fraction == pytest.approx(2 / 8)  # hours 0 and 7
    assert js.all_cloudy_if_independent == pytest.approx((5 / 8) * (4 / 8) * (4 / 8))
    assert js.all_clear_fraction == pytest.approx(2 / 8)  # hours 3 and 5
    assert js.at_least_one_clear_fraction == pytest.approx(6 / 8)
    assert js.pairwise_outage_correlation["a|b"] == pytest.approx(np.corrcoef(a, b)[0, 1])


# ---- configuration -------------------------------------------------------------------------------


def _era5_data(**weather: Any) -> dict[str, Any]:
    from config_helpers import load_mvp_data

    data = load_mvp_data()
    set_path(data, "models.weather_source", "era5")
    w: dict[str, Any] = {"cloud_cover_threshold": 0.5, "historical_year": 2015}
    w.update(weather)
    data["weather"] = w
    return data


def test_weather_config_rules() -> None:
    cfg = parse_config(_era5_data())
    assert cfg.weather is not None
    assert cfg.weather.outage_model == "threshold"
    for data, msg in [
        (_era5_data(cloud_cover_threshold=None), "needs weather.cloud_cover_threshold"),
        (_era5_data(outage_model="bernoulli"), "bernoulli' ignores it"),
        (_era5_data(historical_year=None), "needs weather.historical_year"),
        (_era5_data(historical_year=2030), "outside"),
        (_era5_data(year_mapping="sampled"), "sampled' ignores it"),
    ]:
        with pytest.raises(ConfigError, match=msg):
            parse_config(data)
    data = _era5_data()
    del data["weather"]
    with pytest.raises(ConfigError, match="needs the weather section"):
        parse_config(data)
    data = _era5_data()
    set_path(data, "models.weather_source", "scenario")
    with pytest.raises(ConfigError, match="ignores it"):
        parse_config(data)
    data = _era5_data()
    set_path(data, "models.weather_scenario", "cloudy")
    with pytest.raises(ConfigError, match="weather_scenario='clear'"):
        parse_config(data)
    data = _era5_data()
    set_path(data, "models.atmosphere", "none")
    with pytest.raises(ConfigError, match="atmosphere='scenario'"):
        parse_config(data)


def test_era5_is_rejected_for_microwave() -> None:
    from config_helpers import microwave_data

    data = microwave_data(_era5_data())
    set_path(data, "models.atmosphere", "scenario")
    with pytest.raises(ConfigError, match=r"requires transmitter\.mode='optical'"):
        parse_config(data)


def test_hash_unchanged_while_off_and_changed_when_on() -> None:
    mvp = load_config(MVP_CONFIG)
    # Pinned in tests/unit/test_config_validity.py; the M8a keys at their 'off' values do not
    # enter the hash (config._hash_input).
    assert config_hash(mvp) == "a3fafc4decf6ca483342010b4206bd8affa3dfe65e8369d0e94c7075f00823af"
    on = parse_config(_era5_data())
    assert config_hash(on) != config_hash(mvp)
    other = parse_config(_era5_data(cloud_cover_threshold=0.3))
    assert config_hash(other) != config_hash(on)
    names = [a.path for a in approximations_in_force(on)]
    assert "models.weather_source" in names
    assert "models.weather_source" not in [a.path for a in approximations_in_force(mvp)]


def test_era5_run_is_not_labelled_permanently_clear() -> None:
    # Merge review (M7b x M8a): with ERA5 the clear scenario only sets the clear-hour factors;
    # the cloudy hours are hard outages, so 'permanently clear sky' must not be listed. The
    # clear-hour transmittance label stays (it is still lossless at the clear hours), and the
    # same configuration with the scenario source keeps the upper-bound label.
    from helioswarm.config import idealised_settings

    on = {s.path: s for s in idealised_settings(parse_config(_era5_data()))}
    assert "models.weather_scenario" not in on
    assert all("permanently clear" not in s.effect for s in on.values())
    assert "atmosphere.optical_zenith_transmittance_clear" in on
    off = {s.path: s for s in idealised_settings(load_config(MVP_CONFIG))}
    assert "permanently clear sky" in off["models.weather_scenario"].effect
    assert set(off) - set(on) == {"models.weather_scenario"}
    for name in ("mvp_single_satellite_era5.yaml", "constellation_demo_era5.yaml"):
        cfg = load_config(MVP_CONFIG.parent / name)
        assert cfg.models.weather_source == "era5"
        assert "models.weather_scenario" not in {s.path for s in idealised_settings(cfg)}


def test_report_key_parameters_include_the_weather_source() -> None:
    from helioswarm.report import KEY_PARAMETERS

    for key in ("models.weather_source", "weather.outage_model", "weather.cloud_cover_threshold",
                "weather.year_mapping", "weather.historical_year"):  # fmt: skip
        assert key in KEY_PARAMETERS


def test_resolved_dump_of_an_era5_config_round_trips(tmp_path: Path) -> None:
    from helioswarm.config import write_resolved_yaml

    cfg = parse_config(_era5_data())
    p = write_resolved_yaml(cfg, tmp_path / "c.yaml")
    assert config_hash(load_config(p)) == config_hash(cfg)
