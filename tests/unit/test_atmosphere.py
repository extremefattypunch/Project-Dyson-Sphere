"""Tests of the terminal-segment atmosphere models (M5, PDF §10.2-10.3, §13.3)."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from helioswarm.atmosphere import (
    ITU_R_VERSIONS,
    AtmosphereSettings,
    atmosphere_settings_from_config,
    atmospheric_transmission,
    itu_r_slant_path_attenuation_dB,
    kasten_young_airmass,
    plane_parallel_airmass,
)
from helioswarm.config import ConfigError, parse_config

from config_helpers import microwave_data, set_path


def _settings(**kw: Any) -> AtmosphereSettings:
    base: dict[str, Any] = {
        "model": "scenario",
        "mode": "optical",
        "weather_scenario": "clear",
        "optical_zenith_transmittance": 0.8,
        "microwave_zenith_attenuation_dB": 0.0,
        "min_valid_elevation_rad": math.radians(10.0),
        "wavelength_m": 1.0e-6,
    }
    base.update(kw)
    return AtmosphereSettings(**base)


def test_plane_parallel_and_kasten_young_airmass() -> None:
    el = np.radians([90.0, 30.0, 10.0, 5.0])
    np.testing.assert_allclose(
        plane_parallel_airmass(el),
        [1.0, 2.0, 1 / math.sin(math.radians(10)), 1 / math.sin(math.radians(5))],
        rtol=1e-15,
    )
    ky = kasten_young_airmass(el)
    # Kasten & Young (1989): X(90 deg) = 0.9997 (their formula is not exactly 1 at zenith);
    # documented plane-parallel excesses (M0 schema text): +3.1 % at 10 deg, +11 % at 5 deg.
    assert ky[0] == pytest.approx(1.0 / (1.0 + 0.50572 * 96.07995**-1.6364), rel=1e-15)
    excess = plane_parallel_airmass(el) / ky - 1
    assert excess[2] == pytest.approx(0.031, abs=0.001)
    assert excess[3] == pytest.approx(0.11, abs=0.005)


def test_optical_scenario_transmittance_scales_with_airmass() -> None:
    el = np.radians([90.0, 30.0, 14.4775])
    res = atmospheric_transmission(el, _settings())
    np.testing.assert_allclose(res.transmittance, 0.8 ** (1 / np.sin(el)), rtol=1e-14)
    np.testing.assert_allclose(res.attenuation_dB, -10 * np.log10(res.transmittance), rtol=1e-13)
    assert res.transmittance[1] == pytest.approx(0.64, rel=1e-14)  # X = 2 at 30 deg
    assert np.all(res.weather_available)
    assert np.all(res.in_validity)


def test_cloudy_optical_is_a_hard_outage() -> None:
    res = atmospheric_transmission(
        np.radians([45.0, 80.0]),
        _settings(weather_scenario="cloudy", optical_zenith_transmittance=0.0),
    )
    np.testing.assert_array_equal(res.transmittance, [0.0, 0.0])
    assert not np.any(res.weather_available)
    assert np.all(np.isinf(res.attenuation_dB))
    # A partially transparent cloud scenario is not an outage.
    res2 = atmospheric_transmission(
        np.radians([45.0]), _settings(weather_scenario="cloudy", optical_zenith_transmittance=0.1)
    )
    assert res2.weather_available[0]
    assert res2.transmittance[0] == pytest.approx(0.1 ** math.sqrt(2), rel=1e-14)


def test_microwave_scenario_attenuation_in_db() -> None:
    s = _settings(mode="microwave", microwave_zenith_attenuation_dB=0.5, wavelength_m=0.1224)
    el = np.radians([90.0, 30.0])
    res = atmospheric_transmission(el, s)
    np.testing.assert_allclose(res.attenuation_dB, [0.5, 1.0], rtol=1e-14)
    np.testing.assert_allclose(res.transmittance, [10 ** (-0.05), 10 ** (-0.1)], rtol=1e-14)
    assert np.all(res.weather_available)


def test_none_model_and_validity_floor() -> None:
    el = np.radians([45.0, 5.0, -1.0])
    res = atmospheric_transmission(el, _settings(model="none"))
    np.testing.assert_array_equal(res.transmittance[:2], [1.0, 1.0])
    assert np.isnan(res.transmittance[2])
    assert not res.in_validity[2]
    sc = atmospheric_transmission(el, _settings())
    # Below the plane-parallel floor (10 deg): NaN + flag, never a silent extrapolation.
    assert sc.in_validity.tolist() == [True, False, False]
    assert np.isnan(sc.transmittance[1])
    assert np.isnan(sc.airmass[1])


def test_transmission_depends_on_elevation_only() -> None:
    # PDF §10.3: no attenuation of the interplanetary segment -- the API takes no range, and the
    # same elevation gives the same factor whatever the link geometry.
    el = np.radians([20.0, 20.0, 60.0])
    res = atmospheric_transmission(el, _settings())
    assert res.transmittance[0] == res.transmittance[1]
    import inspect

    assert "range" not in " ".join(inspect.signature(atmospheric_transmission).parameters)


def test_settings_from_config(mvp_data: dict[str, Any]) -> None:
    s = atmosphere_settings_from_config(parse_config(mvp_data))
    assert (s.model, s.mode, s.weather_scenario, s.optical_zenith_transmittance) == (
        "scenario",
        "optical",
        "clear",
        1.0,
    )
    assert s.min_valid_elevation_rad == pytest.approx(math.radians(10.0))
    cloudy = set_path(
        set_path(mvp_data, "models.weather_scenario", "cloudy"),
        "atmosphere.optical_zenith_transmittance_clear",
        0.9,
    )
    sc = atmosphere_settings_from_config(parse_config(cloudy))
    assert sc.optical_zenith_transmittance == 0.0
    assert sc.hard_outage


# ---------------------------------------------------------------------------------------------
# ITU-R (optional itur extra)
# ---------------------------------------------------------------------------------------------


def test_itu_r_reproduces_itu_validation_example() -> None:
    pytest.importorskip("itur")
    # Reference: ITU-R SG3 validation examples (CG-3M3J-13-ValEx, P.618-13 total attenuation),
    # as transcribed in itur 0.4.0's bundled test/ITU_validation_test.py
    # (ITUR618_13TestCase.test_total_attenuation, first and 13th rows): London 51.5 N 0.14 W,
    # el = 31.07694309 deg, D = 1 m, eta = 0.65, tau = 0, R001 from P.837-7 at 0.01 %.
    # Expected (A_g, A_c, A_r, A_s, A) at 14.25 GHz, p = 1 %:
    #   0.223693782, 0.45517046, 0.48914539, 0.26193234, 1.203663661 dB
    # and at 29 GHz, p = 0.01 %: 0.799999368, 1.77247154, 23.19077435, 0.93186567, 25.78063225 dB.
    # The ITU file used P.676-11 and P.1511-1; we pin those revisions for this comparison and
    # compare the gas, cloud and rain components (scintillation is excluded by design).
    import itur.models as itu_models

    versions = dict(ITU_R_VERSIONS) | {"itu676": 11, "itu1511": 1}
    r001 = float(itu_models.itu837.rainfall_rate(51.5, -0.14, 0.01000000001).value)
    cases = [
        (14.25e9, 1.0, (0.223693782, 0.45517046, 0.48914539)),
        (29.0e9, 0.01, (0.799999368, 1.77247154, 23.19077435)),
    ]
    # hs: P.1511 topographic height used by the ITU example (itur estimates it when hs=None);
    # our wrapper passes the site height, so feed the same P.1511-1 value. The direct itur call
    # changes itur's global revision; it is restored so later tests see the default state.
    prev_1511 = itu_models.itu1511.get_version()
    try:
        itu_models.itu1511.change_version(1)
        hs_km = float(itu_models.itu1511.topographic_altitude(51.5, -0.14).value)
    finally:
        itu_models.itu1511.change_version(prev_1511)
    for f_hz, p, (ag, ac, ar) in cases:
        res = itu_r_slant_path_attenuation_dB(
            51.5,
            -0.14,
            hs_km * 1000.0,
            f_hz,
            np.radians([31.07694309]),
            p,
            versions=versions,
            rainfall_rate_001_mm_h=r001,
            polarization_tilt_deg=0.0,
        )
        # Validation-file precision: 1e-5 dB (the file's assertAlmostEqual places=5).
        assert float(res.gas_dB[0]) == pytest.approx(ag, abs=1e-5)
        assert float(res.cloud_dB[0]) == pytest.approx(ac, abs=1e-5)
        assert float(res.rain_dB[0]) == pytest.approx(ar, abs=1e-5)
        assert float(res.total_dB[0]) == pytest.approx(ag + ac + ar, abs=1e-5)
        assert "ITU-R P.618-13" in res.recommendation_versions
        assert "ITU-R P.676-11" in res.recommendation_versions


def test_itu_r_model_through_settings(mvp_data: dict[str, Any]) -> None:
    pytest.importorskip("itur")
    data = microwave_data(mvp_data)
    set_path(data, "models.atmosphere", "itu_r")
    cfg = parse_config(data)
    s = atmosphere_settings_from_config(cfg)
    assert s.model == "itu_r"
    assert s.min_valid_elevation_rad == pytest.approx(math.radians(5.0))
    el = np.radians([10.0, 30.0, 60.0, 3.0])
    res = atmospheric_transmission(el, s, site=cfg.receiver.sites[0])
    assert res.in_validity.tolist() == [True, True, True, False]
    att = res.attenuation_dB[:3]
    assert np.all(att > 0)
    assert np.all(np.diff(att) < 0)
    np.testing.assert_allclose(res.transmittance[:3], 10 ** (-att / 10), rtol=1e-14)
    assert res.gas_attenuation_dB is not None
    assert res.rain_attenuation_dB is not None
    assert "ITU-R P.618-13" in res.recommendation_versions
    assert res.polarization_tilt_deg == 45.0
    # Wrapper pins the versions on every call: a caller changing itur's global state does not
    # leak into the next evaluation, and the caller's state is restored afterwards.
    import itur.models as itu_models

    prev = itu_models.itu676.get_version()
    try:
        itu_models.itu676.change_version(11)
        res2 = atmospheric_transmission(el, s, site=cfg.receiver.sites[0])
        assert itu_models.itu676.get_version() == 11
    finally:
        itu_models.itu676.change_version(prev)
    np.testing.assert_array_equal(res2.attenuation_dB[:3], att)
    with pytest.raises(ValueError, match="site"):
        atmospheric_transmission(el, s)


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"models.atmosphere": "itu_r"}, "requires transmitter.mode='microwave'"),
        (
            {"__mw__": True, "models.atmosphere": "itu_r", "transmitter.wavelength_m": 0.5},
            "carrier frequency",
        ),
        (
            {"__mw__": True, "models.atmosphere": "itu_r", "receiver.min_elevation_deg": 4.0},
            "itu_r_min_elevation_deg",
        ),
        (
            {"__mw__": True, "models.atmosphere": "itu_r", "models.weather_scenario": "cloudy"},
            "weather_scenario",
        ),
    ],
)
def test_itu_r_config_contradictions(
    mvp_data: dict[str, Any], changes: dict[str, Any], fragment: str
) -> None:
    data = microwave_data(mvp_data) if changes.pop("__mw__", False) else mvp_data
    for path, value in changes.items():
        set_path(data, path, value)
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    assert fragment in exc.value.format()


def test_itu_r_versions_are_read_only() -> None:
    # Review M5: no mutable module-level state (a dict here would let one caller re-pin every
    # later run silently).
    with pytest.raises(TypeError):
        ITU_R_VERSIONS["itu676"] = 11  # type: ignore[index]
    assert ITU_R_VERSIONS["itu676"] == 12


def test_itu_r_production_revisions_regression_pin() -> None:
    pytest.importorskip("itur")
    # REGRESSION PIN, NOT A VALIDATION (review M5). The production revisions (P.676-12,
    # P.1511-2, tau from the config) have no ITU-R validation set in itur 0.4.0; these values
    # were produced by this wrapper with itur 0.4.0 at the London validation site (51.5 N,
    # 0.14 W, site height 31.5 m, el = 31.07694309 deg, P.837 rain rate from the map) and pin it
    # against unintended changes (itur upgrade, revision table, wrapper wiring). The gas term
    # differs from the P.676-11 validation value 0.22369 dB by +1.4 % (revision change).
    # Tolerance 1e-9 dB: deterministic float evaluation of the same code and data.
    cases = {
        # (f, p, tau): (A_gas, A_cloud, A_rain) [dB]
        (14.25e9, 1.0, 45.0): (0.2268658304615068, 0.4551704590725894, 0.4735058439566246),
        (29.0e9, 0.01, 45.0): (0.8376158204113131, 1.7724715442334655, 22.46957040199824),
        (29.0e9, 0.01, 0.0): (0.8376158204113131, 1.7724715442334655, 23.443678674206332),
        (29.0e9, 0.01, 90.0): (0.8376158204113131, 1.7724715442334655, 21.502561645175707),
    }
    for (f_hz, p, tau), (ag, ac, ar) in cases.items():
        res = itu_r_slant_path_attenuation_dB(
            51.5, -0.14, 31.5, f_hz, np.radians([31.07694309]), p, polarization_tilt_deg=tau
        )
        np.testing.assert_allclose(
            [res.gas_dB[0], res.cloud_dB[0], res.rain_dB[0], res.total_dB[0]],
            [ag, ac, ar, ag + ac + ar],
            rtol=0,
            atol=1e-9,
        )
        assert "ITU-R P.676-12" in res.recommendation_versions
        assert "ITU-R P.1511-2" in res.recommendation_versions
    # Physics sanity of the tilt (P.838: k_H > k_V at these frequencies): horizontal linear
    # polarisation is attenuated most by rain, vertical least, circular in between.
    a_h, a_c, a_v = (cases[(29.0e9, 0.01, t)][2] for t in (0.0, 45.0, 90.0))
    assert a_h > a_c > a_v
    with pytest.raises(ValueError, match="polarization_tilt_deg"):
        itu_r_slant_path_attenuation_dB(
            51.5, -0.14, 31.5, 14.25e9, np.radians([30.0]), 1.0, polarization_tilt_deg=91.0
        )


def test_itu_r_polarization_tilt_from_config(mvp_data: dict[str, Any]) -> None:
    pytest.importorskip("itur")
    # Review M5: the tilt is a configured physical input, carried to the result (manifest).
    data = microwave_data(mvp_data)
    set_path(data, "models.atmosphere", "itu_r")
    set_path(data, "atmosphere.itu_r_polarization_tilt_deg", 0.0)
    set_path(data, "atmosphere.itu_r_exceedance_percent", 0.01)
    cfg = parse_config(data)
    s = atmosphere_settings_from_config(cfg)
    assert s.itu_r_polarization_tilt_deg == 0.0
    el = np.radians([30.0])
    site = cfg.receiver.sites[0]
    horiz = atmospheric_transmission(el, s, site=site)
    vert = atmospheric_transmission(
        el, dataclasses.replace(s, itu_r_polarization_tilt_deg=90.0), site=site
    )
    assert horiz.polarization_tilt_deg == 0.0
    assert vert.polarization_tilt_deg == 90.0
    assert horiz.rain_attenuation_dB is not None
    assert vert.rain_attenuation_dB is not None
    assert float(horiz.rain_attenuation_dB[0]) > float(vert.rain_attenuation_dB[0])
    scen = atmospheric_transmission(el, _settings())
    assert scen.polarization_tilt_deg is None
