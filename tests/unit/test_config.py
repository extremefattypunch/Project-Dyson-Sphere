"""Unit tests for the run-configuration schema (helioswarm.config).

Every invalid-config test asserts (a) that validation fails, (b) that a reported issue names the
offending field path, and (c) a fragment of the explanation, so a validator that rejects for the
wrong reason, or names the wrong field, fails the test.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from helioswarm import config as cfgmod
from helioswarm.config import (
    SCHEMA_VERSION,
    ConfigError,
    RunConfig,
    canonical_json,
    config_hash,
    dump_resolved_yaml,
    field_meta,
    leaf_paths,
    load_config,
    parse_config,
    provenance_table,
    read_yaml,
)
from helioswarm.constants import AU_M, SECONDS_PER_DAY, SOLAR_RADIUS_NOMINAL_M

from config_helpers import MVP_CONFIG, del_path, microwave_data, set_path


def _issues(data: Any) -> list[tuple[str, str]]:
    """Validate ``data`` expecting failure; return (path, formatted message) of every issue."""
    with pytest.raises(ConfigError) as info:
        parse_config(data)
    return [(i.path, i.format()) for i in info.value.issues]


def _assert_issue(data: Any, field: str, fragment: str) -> None:
    """Assert that validation fails with an issue that names ``field`` and contains ``fragment``.

    The field may be reported either as the issue path (single-field validators) or inside the
    message (cross-field validators report the dotted paths of every field involved).
    """
    issues = _issues(data)
    matching = [msg for path, msg in issues if (path == field or field in msg)]
    assert matching, f"no issue names {field!r}; got {issues}"
    assert any(fragment in msg for msg in matching), f"{fragment!r} not in {matching}"


# ---------------------------------------------------------------------------------------------
# Valid configurations
# ---------------------------------------------------------------------------------------------


def test_mvp_config_loads_with_expected_values() -> None:
    cfg = load_config(MVP_CONFIG)
    assert cfg.schema_version == SCHEMA_VERSION == 1
    assert cfg.transmitter.mode == "optical"
    assert cfg.models.beam == "airy_far_field"
    assert cfg.receiver.sites[0].name == "example_site"
    # PDF example writes "1.0e12" which PyYAML's YAML-1.1 resolver reads as a *string*; the
    # config loader must read it as the float 1e12 (strict schema would otherwise reject it).
    assert cfg.receiver.max_input_power_W == 1.0e12
    assert isinstance(cfg.receiver.max_input_power_W, float)
    # Unquoted-timestamp safety: epochs stay canonical strings.
    assert cfg.time.start_utc == "2030-01-01T00:00:00"


def test_si_accessors_are_exact_conversions() -> None:
    cfg = load_config(MVP_CONFIG)
    # 365.25 d * 86400 s/d = 31 557 600 s exactly; / 3600 s = 8766 intervals.
    assert cfg.time.duration_s == 31_557_600.0
    assert cfg.time.n_output_intervals == 8766
    # 1 au is exact by definition (IAU 2012 B2).
    assert cfg.orbit.semimajor_axis_m == 149_597_870_700.0
    assert cfg.orbit.true_anomaly_rad == pytest.approx(math.pi / 6, rel=0, abs=1e-15)
    assert cfg.receiver.min_elevation_rad == pytest.approx(math.radians(10.0), rel=1e-15)
    assert cfg.receiver.area_m2 == pytest.approx(math.pi * 1000.0**2, rel=1e-15)


def test_constants_are_the_defined_values() -> None:
    # Values fixed by IAU resolutions (exact, no tolerance).
    assert AU_M == 149_597_870_700.0  # IAU 2012 Resolution B2
    assert SOLAR_RADIUS_NOMINAL_M == 695_700_000.0  # IAU 2015 Resolution B3
    assert SECONDS_PER_DAY == 86_400.0


def test_defaults_are_filled_and_documented(mvp_data: dict[str, Any]) -> None:
    cfg = parse_config(mvp_data)
    # Sections absent from the YAML are created with documented defaults.
    assert cfg.constants.solar_constant_1au_W_m2 == 1361.0
    assert cfg.thermal.space_temperature_K == 2.72548  # T_CMB, Fixsen (2009) ApJ 707, 916
    assert cfg.atmosphere.optical_zenith_transmittance_cloudy == 0.0
    assert cfg.models.fraunhofer_max_fresnel_number == 0.5
    meta = field_meta(type(cfg.constants), "solar_constant_1au_W_m2")
    assert meta.provenance == "sourced"
    assert meta.source is not None
    assert "IAU 2015" in meta.source


def test_microwave_configuration_is_accepted(mvp_data: dict[str, Any]) -> None:
    cfg = parse_config(microwave_data(mvp_data))
    assert cfg.transmitter.mode == "microwave"
    assert cfg.receiver.technology == "rectenna"
    assert cfg.transmitter.wavelength_m == pytest.approx(0.12236, rel=1e-4)


def test_two_body_config_without_integrator_step_is_accepted(mvp_data: dict[str, Any]) -> None:
    data = set_path(mvp_data, "orbit.model", "two_body_kepler")
    data["orbit"]["include_planetary_perturbations"] = False
    data["orbit"]["include_solar_radiation_pressure"] = False
    del data["orbit"]["cr"]
    del data["orbit"]["area_to_mass_m2_kg"]
    del data["time"]["integrator_step_s"]
    cfg = parse_config(data)
    assert cfg.time.integrator_step_s is None
    assert cfg.orbit.cr is None


def test_eccentric_orbit_with_perihelion_outside_sun_is_accepted(
    mvp_data: dict[str, Any],
) -> None:
    # a(1-e) = 0.01 au = 1.496e9 m > R_sun = 6.957e8 m: accepted. (Using a or a(1+e) instead of
    # a(1-e) would also accept this; test_perihelion_inside_sun_is_rejected discriminates.)
    set_path(mvp_data, "orbit.eccentricity", 0.99)
    cfg = parse_config(mvp_data)
    assert cfg.orbit.perihelion_m == pytest.approx(0.01 * AU_M, rel=1e-12)


def test_fixed_normal_mode_accepts_unit_vector(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "collector.incidence_mode", "fixed_normal")
    set_path(mvp_data, "collector.fixed_normal_unit_j2000", [0.6, 0.8, 0.0])
    cfg = parse_config(mvp_data)
    assert cfg.collector.fixed_normal_unit_j2000 == (0.6, 0.8, 0.0)


# ---------------------------------------------------------------------------------------------
# Invalid configurations, one class at a time
# ---------------------------------------------------------------------------------------------

RANGE_CASES = [
    # (path, value, fragment)  bad unit-range / efficiency > 1 / sign errors
    ("receiver.min_elevation_deg", 95.0, "less than 90"),
    ("receiver.sites[0].latitude_deg", 91.0, "less than or equal to 90"),
    ("receiver.sites[0].longitude_deg", -181.0, "greater than or equal to -180"),
    ("transmitter.electrical_to_beam_efficiency", 1.2, "less than or equal to 1"),
    ("receiver.conversion_efficiency", 1.0000001, "less than or equal to 1"),
    ("receiver.grid_efficiency", 0.0, "greater than 0"),
    ("collector.pv_efficiency_reference", 1.5, "less than or equal to 1"),
    ("collector.absorptivity", -0.1, "greater than 0"),
    ("thermal.emissivity", 1.01, "less than or equal to 1"),
    ("collector.area_m2", -5.0, "greater than 0"),
    ("transmitter.aperture_diameter_m", 0.0, "greater than 0"),
    ("receiver.radius_m", -1.0, "greater than 0"),
    ("thermal.radiator_temperature_limit_K", -10.0, "greater than 0"),
    ("orbit.inclination_deg", 181.0, "less than or equal to 180"),
    ("orbit.raan_deg", 360.0, "less than 360"),
    ("orbit.semimajor_axis_au", 0.0, "greater than 0"),
    ("time.duration_days", 0.0, "greater than 0"),
    ("time.output_step_s", -3600.0, "greater than 0"),
    ("transmitter.pointing_sigma_rad", -1e-9, "greater than or equal to 0"),
    ("thermal.radiating_sides", 3, "Input should be 1 or 2"),
    ("orbit.cr", 0.5, "greater than or equal to 1"),
    ("project.seed", -1, "greater than or equal to 0"),
    ("receiver.max_heat_rejection_W", 0.0, "greater than 0"),
    ("atmosphere.plane_parallel_min_elevation_deg", 90.0, "less than 90"),
    ("models.gaussian_beam_min_truncation_ratio", 0.99, "greater than or equal to 1"),
]


@pytest.mark.parametrize(("path", "value", "fragment"), RANGE_CASES)
def test_out_of_range_values_are_rejected(
    mvp_data: dict[str, Any], path: str, value: Any, fragment: str
) -> None:
    _assert_issue(set_path(mvp_data, path, value), path, fragment)


@pytest.mark.parametrize("ecc", [1.0, 1.5])
def test_non_elliptic_eccentricity_is_rejected(mvp_data: dict[str, Any], ecc: float) -> None:
    _assert_issue(
        set_path(mvp_data, "orbit.eccentricity", ecc), "orbit.eccentricity", "less than 1"
    )


def test_perihelion_inside_sun_is_rejected(mvp_data: dict[str, Any]) -> None:
    # a = 0.5 au, e = 0.995: a(1-e) = 0.0025 au = 3.74e8 m < R_sun = 6.957e8 m, whereas
    # a = 0.5 au and a(1+e) are far outside the Sun -> only the a(1-e) formula rejects this.
    set_path(mvp_data, "orbit.semimajor_axis_au", 0.5)
    set_path(mvp_data, "orbit.eccentricity", 0.995)
    _assert_issue(mvp_data, "orbit.eccentricity", "inside the Sun")


@pytest.mark.parametrize(
    "path",
    [
        "collector.area_m2",
        "transmitter.wavelength_m",
        "receiver.radius_m",
        "orbit.true_anomaly_deg",
    ],
)
def test_missing_required_field_is_rejected(mvp_data: dict[str, Any], path: str) -> None:
    _assert_issue(del_path(mvp_data, path), path, "missing required field")


def test_missing_required_section_is_rejected(mvp_data: dict[str, Any]) -> None:
    _assert_issue(del_path(mvp_data, "thermal"), "thermal", "missing required field")


@pytest.mark.parametrize(
    ("path", "suggestion"),
    [
        ("collector.area", "area_m2"),
        ("transmitter.wavelength", "wavelength_m"),
        ("orbitt", "orbit"),
        ("receiver.sites[0].latitude", "latitude_deg"),
    ],
)
def test_unknown_key_is_rejected_with_suggestion(
    mvp_data: dict[str, Any], path: str, suggestion: str
) -> None:
    _assert_issue(set_path(mvp_data, path, 1.0), path, f"did you mean '{suggestion}'")


def test_wrong_types_are_not_coerced(mvp_data: dict[str, Any]) -> None:
    # Strict schema: a quoted number or a non-integer seed must not be silently converted.
    set_path(mvp_data, "collector.area_m2", "10000")
    set_path(mvp_data, "project.seed", 1.5)
    set_path(mvp_data, "models.light_time", "yes")
    paths = {p for p, _ in _issues(mvp_data)}
    assert {"collector.area_m2", "project.seed", "models.light_time"} <= paths


def test_non_finite_values_are_rejected(mvp_data: dict[str, Any]) -> None:
    _assert_issue(set_path(mvp_data, "collector.area_m2", math.inf), "collector.area_m2", "finite")
    data = set_path(copy.deepcopy(mvp_data), "receiver.radius_m", math.nan)
    _assert_issue(data, "receiver.radius_m", "finite")


@pytest.mark.parametrize(
    ("mutations", "field", "fragment"),
    [
        # optical mode with a microwave-only beam model
        ({"models.beam": "aperture_far_field"}, "models.beam", "not allowed with"),
        # optical mode with a microwave wavelength
        ({"transmitter.wavelength_m": 0.1224}, "transmitter.wavelength_m", "optical range"),
        # optical link with a rectenna receiver
        ({"receiver.technology": "rectenna"}, "receiver.technology", "does not match"),
        # Airy pattern requires uniform illumination
        (
            {
                "transmitter.aperture_illumination": "truncated_gaussian",
                "transmitter.gaussian_waist_radius_m": 4.0,
            },
            "models.beam",
            "uniformly illuminated",
        ),
        # Gaussian beam model requires Gaussian illumination
        ({"models.beam": "gaussian_beam"}, "models.beam", "truncated_gaussian"),
        # phased-array lattice is microwave-only
        (
            {"transmitter.array_element_spacing_m": 0.5},
            "transmitter.array_element_spacing_m",
            "requires transmitter.mode='microwave'",
        ),
        # taper parameter without a tapered illumination
        ({"transmitter.edge_taper_dB": 10.0}, "transmitter.edge_taper_dB", "only meaningful"),
        # analytic two-body model cannot include perturbations
        (
            {"orbit.model": "two_body_kepler", "orbit.include_solar_radiation_pressure": False},
            "orbit.include_planetary_perturbations",
            "contradicts",
        ),
        # SRP without its parameters
        ({"orbit.cr": None}, "orbit.cr", "required when include_solar_radiation_pressure"),
        # SRP parameters set while SRP is off would be silently ignored
        (
            {"orbit.include_solar_radiation_pressure": False},
            "orbit.cr",
            "is set but include_solar_radiation_pressure=false ignores it",
        ),
        (
            {"orbit.include_solar_radiation_pressure": False},
            "orbit.area_to_mass_m2_kg",
            "is set but include_solar_radiation_pressure=false ignores it",
        ),
        # cloudy weather with no atmosphere model would silently become the lossless bound
        (
            {"models.atmosphere": "none", "models.weather_scenario": "cloudy"},
            "models.weather_scenario",
            "has no effect with models.atmosphere='none'",
        ),
        # plane-parallel air mass 1/sin(el) diverges at the horizon
        (
            {"receiver.min_elevation_deg": 0.0},
            "receiver.min_elevation_deg",
            "below atmosphere.plane_parallel_min_elevation_deg",
        ),
        # numerical propagator without a step setting
        ({"time.integrator_step_s": None}, "time.integrator_step_s", "required when orbit.model"),
        # radiator cannot reject heat at or below the sink temperature
        (
            {"thermal.radiator_temperature_limit_K": 2.725},
            "thermal.radiator_temperature_limit_K",
            "must exceed",
        ),
        # output grid must tile the window: 365.25 d / 7 s is not an integer
        ({"time.output_step_s": 7.0}, "time.output_step_s", "not an integer multiple"),
        ({"time.duration_days": 0.01}, "time.output_step_s", "exceeds the window length"),
        # efficiency law clip bounds inconsistent with the reference efficiency
        ({"collector.pv_efficiency_max": 0.2}, "collector.pv_efficiency_reference", "outside"),
        (
            {"collector.incidence_mode": "fixed_normal"},
            "collector.fixed_normal_unit_j2000",
            "required",
        ),
        (
            {
                "collector.incidence_mode": "fixed_normal",
                "collector.fixed_normal_unit_j2000": [1.0, 1.0, 0.0],
            },
            "collector.fixed_normal_unit_j2000",
            "unit vector",
        ),
        (
            {
                "atmosphere.optical_zenith_transmittance_clear": 0.5,
                "atmosphere.optical_zenith_transmittance_cloudy": 0.6,
            },
            "atmosphere.optical_zenith_transmittance_cloudy",
            "exceeds the clear-sky value",
        ),
    ],
)
def test_contradictory_settings_are_rejected(
    mvp_data: dict[str, Any], mutations: dict[str, Any], field: str, fragment: str
) -> None:
    for path, value in mutations.items():
        set_path(mvp_data, path, value)
    _assert_issue(mvp_data, field, fragment)


def test_microwave_mode_rejects_optical_beam_and_receiver(mvp_data: dict[str, Any]) -> None:
    data = microwave_data(mvp_data)
    data["models"]["beam"] = "airy_far_field"
    data["receiver"]["technology"] = "laser_photovoltaic"
    # uniform illumination so that Airy's own illumination requirement is satisfied
    data["transmitter"]["aperture_illumination"] = "uniform_circular"
    del data["transmitter"]["edge_taper_dB"]
    issues = [msg for _, msg in _issues(data)]
    # Two independent cross-field problems -> two separate issue lines, each naming its fields.
    assert len(issues) == 2, issues
    assert sum("models.beam='airy_far_field' is not allowed" in m for m in issues) == 1
    assert sum("receiver.technology='laser_photovoltaic'" in m for m in issues) == 1


def test_microwave_mode_rejects_optical_wavelength(mvp_data: dict[str, Any]) -> None:
    data = microwave_data(mvp_data)
    data["transmitter"]["wavelength_m"] = 1.0e-6
    _assert_issue(data, "transmitter.wavelength_m", "microwave range")


def test_all_problems_in_a_section_are_reported_together(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "orbit.eccentricity", 1.2)
    set_path(mvp_data, "collector.area_m2", -1.0)
    set_path(mvp_data, "receiver.conversion_efficiency", 2.0)
    set_path(mvp_data, "thermal.colour", "red")
    paths = {p for p, _ in _issues(mvp_data)}
    assert paths == {
        "orbit.eccentricity",
        "collector.area_m2",
        "receiver.conversion_efficiency",
        "thermal.colour",
    }


def test_offending_value_is_reported(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "transmitter.electrical_to_beam_efficiency", 1.25)
    (issue,) = _issues(mvp_data)
    assert issue == (
        "transmitter.electrical_to_beam_efficiency",
        "transmitter.electrical_to_beam_efficiency: Input should be less than or equal to 1 "
        "(got: 1.25)",
    )


def test_unsupported_schema_version_is_rejected(mvp_data: dict[str, Any]) -> None:
    _assert_issue(set_path(mvp_data, "schema_version", 2), "schema_version", "unsupported")


def test_missing_schema_version_is_rejected(mvp_data: dict[str, Any]) -> None:
    # No default: an unversioned file must never be silently read under a newer schema.
    _assert_issue(del_path(mvp_data, "schema_version"), "schema_version", "missing required field")


@pytest.mark.parametrize(
    "epoch",
    [
        "2030-13-01T00:00:00",  # month 13
        "2030-02-30T00:00:00",  # no such day
        "2030-01-01T00:00:00+01:00",  # offset: not UTC
        "2030-01-01",  # no time of day
        "2016-12-31T23:59:60",  # leap second epochs not accepted
    ],
)
def test_bad_utc_epochs_are_rejected(mvp_data: dict[str, Any], epoch: str) -> None:
    _assert_issue(set_path(mvp_data, "time.start_utc", epoch), "time.start_utc", epoch)


def test_equivalent_utc_spellings_are_canonicalised(mvp_data: dict[str, Any]) -> None:
    a = parse_config(set_path(copy.deepcopy(mvp_data), "time.start_utc", "2030-01-01 00:00:00Z"))
    b = parse_config(set_path(copy.deepcopy(mvp_data), "time.start_utc", "2030-01-01T00:00:00.000"))
    assert a.time.start_utc == b.time.start_utc == "2030-01-01T00:00:00"
    c = parse_config(set_path(copy.deepcopy(mvp_data), "time.start_utc", "2030-01-01T00:00:00.25"))
    assert c.time.start_utc == "2030-01-01T00:00:00.25"


def test_duplicate_site_names_are_rejected(mvp_data: dict[str, Any]) -> None:
    mvp_data["receiver"]["sites"].append(dict(mvp_data["receiver"]["sites"][0]))
    _assert_issue(mvp_data, "receiver.sites", "unique")


def test_empty_site_list_is_rejected(mvp_data: dict[str, Any]) -> None:
    _assert_issue(set_path(mvp_data, "receiver.sites", []), "receiver.sites", "at least 1")


def test_top_level_must_be_mapping() -> None:
    issues = _issues([1, 2, 3])
    assert issues[0][0] == "<root>"
    assert "mapping" in issues[0][1]


def test_models_are_frozen() -> None:
    cfg = load_config(MVP_CONFIG)
    with pytest.raises(ValidationError, match="frozen"):
        cfg.collector.area_m2 = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------------------------


def test_yaml_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text("collector:\n  area_m2: 1\n  area_m2: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert "duplicate key 'area_m2'" in info.value.format()


def test_yaml_syntax_and_missing_file_errors_are_readable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML syntax error"):
        load_config(bad)
    with pytest.raises(ConfigError, match="cannot read file"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_yaml_number_and_timestamp_resolution(tmp_path: Path) -> None:
    path = tmp_path / "n.yaml"
    path.write_text(
        "a: 1e-6\nb: 1.0e12\nc: -2E+3\nd: 1.5\ne: 7\nf: 2030-01-01T00:00:00\ng: '1e3'\n",
        encoding="utf-8",
    )
    data = read_yaml(path)
    assert data == {
        "a": 1e-6,
        "b": 1e12,
        "c": -2000.0,
        "d": 1.5,
        "e": 7,
        "f": "2030-01-01T00:00:00",
        "g": "1e3",  # explicitly quoted strings stay strings
    }
    assert isinstance(data["e"], int)
    assert isinstance(data["b"], float)


# ---------------------------------------------------------------------------------------------
# Resolved dump and hash
# ---------------------------------------------------------------------------------------------


def test_resolved_dump_round_trips(tmp_path: Path) -> None:
    cfg = load_config(MVP_CONFIG)
    text = dump_resolved_yaml(cfg)
    # Defaults absent from the input are explicit in the dump.
    resolved = yaml.safe_load(text)
    assert resolved["constants"]["solar_constant_1au_W_m2"] == 1361.0
    assert resolved["thermal"]["space_temperature_K"] == 2.72548
    assert resolved["frames"]["earth_ellipsoid"] == "WGS84"
    assert f"config_hash_sha256: {config_hash(cfg)}" in text
    path = tmp_path / "resolved.yaml"
    path.write_text(text, encoding="utf-8")
    again = load_config(path)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)
    # The dump is a fixed point.
    assert dump_resolved_yaml(again) == text


def test_resolved_dump_round_trips_microwave(mvp_data: dict[str, Any], tmp_path: Path) -> None:
    cfg = parse_config(microwave_data(mvp_data))
    path = tmp_path / "mw.yaml"
    path.write_text(dump_resolved_yaml(cfg), encoding="utf-8")
    assert load_config(path) == cfg


def test_hash_is_sha256_of_canonical_json() -> None:
    cfg = load_config(MVP_CONFIG)
    expected = hashlib.sha256(canonical_json(cfg).encode("ascii")).hexdigest()
    assert config_hash(cfg) == expected
    assert len(expected) == 64


def test_hash_invariant_under_equivalent_spellings(mvp_data: dict[str, Any]) -> None:
    base = config_hash(parse_config(mvp_data))
    # (1) key order reversed everywhere
    reordered = {k: mvp_data[k] for k in reversed(list(mvp_data))}
    reordered["collector"] = dict(reversed(list(mvp_data["collector"].items())))
    assert config_hash(parse_config(reordered)) == base
    # (2) defaults written explicitly
    explicit = set_path(copy.deepcopy(mvp_data), "constants.solar_constant_1au_W_m2", 1361.0)
    set_path(explicit, "thermal.space_temperature_K", 2.72548)
    assert config_hash(parse_config(explicit)) == base
    # (3) user-ordered mapping (provenance_overrides) in a different key order; pydantic keeps
    # schema order for fields, so this is the case that needs sort_keys in canonical_json.
    a = copy.deepcopy(mvp_data)
    a["provenance_overrides"] = {"receiver.radius_m": "swept", "collector.area_m2": "swept"}
    b = copy.deepcopy(mvp_data)
    b["provenance_overrides"] = {"collector.area_m2": "swept", "receiver.radius_m": "swept"}
    assert config_hash(parse_config(a)) == config_hash(parse_config(b))
    # (4) integer vs float spelling and 'Z' suffix on epochs
    spelled = set_path(copy.deepcopy(mvp_data), "collector.area_m2", 1.0e4)
    set_path(spelled, "time.start_utc", "2030-01-01T00:00:00Z")
    assert config_hash(parse_config(spelled)) == base


def test_hash_changes_with_any_physical_value(mvp_data: dict[str, Any]) -> None:
    base = config_hash(parse_config(mvp_data))
    for path, value in [
        ("collector.area_m2", math.nextafter(10000.0, math.inf)),  # 1 ulp change
        ("constants.solar_constant_1au_W_m2", 1360.8),  # a default being overridden
        ("receiver.sites[0].altitude_m", 1.0),
        ("project.seed", 20261112),
    ]:
        changed = set_path(copy.deepcopy(mvp_data), path, value)
        assert config_hash(parse_config(changed)) != base, path


def test_hash_is_stable_across_processes() -> None:
    """Hash must not depend on interpreter state such as string-hash randomisation."""
    here = config_hash(load_config(MVP_CONFIG))
    code = textwrap.dedent(
        f"""
        from helioswarm.config import config_hash, load_config
        print(config_hash(load_config({str(MVP_CONFIG)!r})))
        """
    )
    for seed in ("0", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert out.stdout.strip() == here


# ---------------------------------------------------------------------------------------------
# Schema metadata conventions
# ---------------------------------------------------------------------------------------------


def _section_classes() -> list[type[BaseModel]]:
    return [
        obj
        for obj in vars(cfgmod).values()
        if isinstance(obj, type) and issubclass(obj, cfgmod._Section) and obj is not cfgmod._Section
    ]


SECTION_FIELDS = {"project", "time", "frames", "orbit", "collector", "transmitter", "receiver"}
SECTION_FIELDS |= {"thermal", "atmosphere", "models", "constants", "sites"}

UNIT_SUFFIX = {
    "m": "_m",
    "s": "_s",
    "K": "_K",
    "W": "_W",
    "W m^-2": "_W_m2",
    "m^2": "_m2",
    "au": "_au",
    "day": "_days",
    "deg": "_deg",
    "rad": "_rad",
    "dB": "_dB",
    "K^-1": "_per_K",
    "m^2 kg^-1": "_m2_kg",
    "kg m^-2": "_kg_m2",
    "m s^-1": "_m_s",
    "UTC ISO-8601": "_utc",
    "%": "_percent",
}


def test_every_leaf_field_has_provenance_unit_and_description() -> None:
    classes = _section_classes()
    # PDF p.8 sections plus the extra sections required by later milestones.
    names = {c.__name__ for c in classes}
    assert {
        "ProjectConfig",
        "TimeConfig",
        "FramesConfig",
        "OrbitConfig",
        "CollectorConfig",
        "TransmitterConfig",
        "ReceiverConfig",
        "SiteConfig",
        "ThermalConfig",
        "ModelsConfig",
        "AtmosphereConfig",
        "ConstantsConfig",
        "RunConfig",
    } <= names
    n_checked = 0
    for cls in classes:
        assert cls.model_config.get("extra") == "forbid", cls
        assert cls.model_config.get("frozen") is True, cls
        for name in cls.model_fields:
            if name in SECTION_FIELDS:
                continue
            meta = field_meta(cls, name)
            assert meta.provenance in {"measured", "sourced", "assumed", "swept"}, (cls, name)
            assert meta.kind in {"physical", "numerical", "model_choice", "metadata"}, (cls, name)
            assert meta.description.strip(), (cls, name)
            assert meta.unit.strip(), (cls, name)
            if meta.provenance == "sourced":
                assert meta.source, (cls, name)
            n_checked += 1
    assert n_checked >= 70


def test_field_names_carry_their_unit_suffix() -> None:
    for cls in _section_classes():
        for name in cls.model_fields:
            if name in SECTION_FIELDS:
                continue
            unit = field_meta(cls, name).unit
            if unit in {"-", "1"}:
                continue
            assert unit in UNIT_SUFFIX, f"{cls.__name__}.{name}: unmapped unit {unit!r}"
            assert name.endswith(UNIT_SUFFIX[unit]), f"{cls.__name__}.{name} [{unit}]"


def test_json_schema_exposes_provenance() -> None:
    schema = RunConfig.model_json_schema()
    collector = schema["$defs"]["CollectorConfig"]["properties"]["area_m2"]
    assert collector["provenance"] == "assumed"
    assert collector["unit"] == "m^2"
    assert collector["exclusiveMinimum"] == 0.0


def test_provenance_table_and_overrides(mvp_data: dict[str, Any]) -> None:
    mvp_data["provenance_overrides"] = {
        "receiver.radius_m": "swept",
        "receiver.sites[0].latitude_deg": "measured",
    }
    cfg = parse_config(mvp_data)
    rows = {r.path: r for r in provenance_table(cfg)}
    assert set(rows) == set(leaf_paths(cfg))
    assert rows["receiver.radius_m"].provenance == "swept"
    assert rows["receiver.radius_m"].schema_provenance == "assumed"
    assert rows["receiver.sites[0].latitude_deg"].provenance == "measured"
    assert rows["constants.solar_constant_1au_W_m2"].provenance == "sourced"
    assert rows["collector.area_m2"].value == 10000.0
    assert rows["collector.area_m2"].unit == "m^2"


def test_provenance_override_of_unknown_path_is_rejected(mvp_data: dict[str, Any]) -> None:
    mvp_data["provenance_overrides"] = {"receiver.radius": "swept"}
    _assert_issue(mvp_data, "provenance_overrides", "did you mean 'receiver.radius_m'")
    mvp_data["provenance_overrides"] = {"receiver.radius_m": "guessed"}
    _assert_issue(mvp_data, "provenance_overrides", "'measured', 'sourced', 'assumed' or 'swept'")


def test_idealised_settings_of_the_mvp_and_after_overrides() -> None:
    # PDF §13.4 "label idealised upper bounds": the MVP config keeps the optimistic defaults
    # listed in docs/model_assumptions.md ("Optimistic defaults"), so each must be reported.
    from helioswarm.config import approximations_in_force, idealised_settings, parse_config

    from config_helpers import load_mvp_data

    paths = {s.path for s in idealised_settings(cfgmod.load_config(MVP_CONFIG))}
    assert paths == {
        "transmitter.power_conditioning_efficiency",
        "transmitter.optics_or_rf_efficiency",
        "transmitter.rms_wavefront_error_m",
        "transmitter.min_pointing_efficiency",
        "collector.incidence_mode",
        "collector.pv_efficiency_max",
        "thermal.pv_view_factor",
        "models.weather_scenario",  # one clear-sky state for the whole year: no cloud outage
        "atmosphere.optical_zenith_transmittance_clear",
        "receiver.max_power_density_W_m2",
        "receiver.max_heat_rejection_W",
        "receiver.sun_avoidance_deg",
    }
    # Configuring a real loss or constraint removes exactly that label (not a fixed list).
    data = load_mvp_data()
    set_path(data, "transmitter.power_conditioning_efficiency", 0.95)
    set_path(data, "receiver.sun_avoidance_deg", 5.0)
    set_path(data, "transmitter.min_pointing_efficiency", 0.5)
    set_path(data, "transmitter.pointing_sigma_rad", 0.0)  # perfect pointing is now idealised
    after = {s.path for s in idealised_settings(parse_config(data))}
    assert after == (paths - {
        "transmitter.power_conditioning_efficiency", "receiver.sun_avoidance_deg",
        "transmitter.min_pointing_efficiency",
    }) | {"transmitter.pointing_sigma_rad"}  # fmt: skip
    # Every other lossless stage and every disabled occultation check is labelled; a cloudy
    # scenario is not an upper bound (and an optical cloudy transmittance < 1 is not labelled).
    data = load_mvp_data()
    for path in ("transmitter.electrical_to_beam_efficiency", "receiver.conversion_efficiency",
                 "receiver.grid_efficiency"):  # fmt: skip
        set_path(data, path, 1.0)
    for body in ("earth", "sun", "moon"):
        set_path(data, f"models.{body}_occultation", False)
    set_path(data, "models.weather_scenario", "cloudy")
    set_path(data, "models.light_time", False)
    lossless = {s.path for s in idealised_settings(parse_config(data))}
    assert lossless == (paths - {
        "models.weather_scenario", "atmosphere.optical_zenith_transmittance_clear",
    }) | {
        "transmitter.electrical_to_beam_efficiency", "receiver.conversion_efficiency",
        "receiver.grid_efficiency", "models.earth_occultation", "models.sun_occultation",
        "models.moon_occultation",
    }  # fmt: skip
    # Light-time off is a named approximation of either sign, not an upper bound.
    assert [a.path for a in approximations_in_force(parse_config(data))] == ["models.light_time"]
    assert approximations_in_force(cfgmod.load_config(MVP_CONFIG)) == ()
    # The microwave MVP uses ITU-R (no clear-sky scenario label) and a rectenna.
    mw_cfg = cfgmod.load_config(MVP_CONFIG.parent / "mvp_single_satellite_microwave.yaml")
    assert mw_cfg.models.atmosphere == "itu_r"
    mw = {s.path for s in idealised_settings(mw_cfg)}
    assert "atmosphere.optical_zenith_transmittance_clear" not in mw
    assert "atmosphere.microwave_zenith_attenuation_clear_dB" not in mw
    assert "models.weather_scenario" not in mw
    # Scenario-mode microwave: the configured scenario's 0 dB zenith attenuation is labelled.
    mw_data = read_yaml(MVP_CONFIG.parent / "mvp_single_satellite_microwave.yaml")
    set_path(mw_data, "models.atmosphere", "scenario")
    set_path(mw_data, "models.weather_scenario", "cloudy")
    mw_cloudy = {s.path for s in idealised_settings(parse_config(mw_data))}
    assert "atmosphere.microwave_zenith_attenuation_cloudy_dB" in mw_cloudy
    assert "models.weather_scenario" not in mw_cloudy
    set_path(mw_data, "atmosphere.microwave_zenith_attenuation_cloudy_dB", 3.0)
    mw_cloudy = {s.path for s in idealised_settings(parse_config(mw_data))}
    assert "atmosphere.microwave_zenith_attenuation_cloudy_dB" not in mw_cloudy


def test_fixed_normal_with_srp_is_a_named_approximation() -> None:
    # PDF §17: the radial SRP model on a fixed-normal collector is named, with a comparison path.
    from helioswarm.config import approximations_in_force, parse_config

    from config_helpers import load_mvp_data

    data = load_mvp_data()
    set_path(data, "collector.incidence_mode", "fixed_normal")
    set_path(data, "collector.fixed_normal_unit_j2000", [0.0, 0.6, 0.8])
    (a,) = approximations_in_force(parse_config(data))
    assert (a.path, a.value) == ("collector.incidence_mode", "fixed_normal")
    assert "include_solar_radiation_pressure = false" in a.comparison_path
    assert "(1+eps)/(cos(theta)[(1-eps)+2 eps cos^2(theta)])" in a.approximation
    # Without SRP there is nothing to approximate.
    set_path(data, "orbit.include_solar_radiation_pressure", False)
    del_path(data, "orbit.cr")
    del_path(data, "orbit.area_to_mass_m2_kg")
    assert approximations_in_force(parse_config(data)) == ()
