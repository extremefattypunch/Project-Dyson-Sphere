"""Model-validity gates, compatibility matrices and regression pins of the config schema.

These tests complement ``test_config.py``. Each one is built so that a plausible wrong
implementation (a mutated allowed-set, a missing deg->rad conversion, a dropped bound, a looser
tolerance, a changed serialisation) fails it.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
from typing import Any

import pytest
from scipy.integrate import quad

from helioswarm.config import (
    ConfigError,
    canonical_json,
    config_hash,
    load_config,
    parse_config,
    provenance_table,
    resolved_dict,
)

from config_helpers import MVP_CONFIG, del_path, microwave_data, set_path


def _issue_messages(data: Any) -> list[str]:
    with pytest.raises(ConfigError) as info:
        parse_config(data)
    return [i.format() for i in info.value.issues]


# ---------------------------------------------------------------------------------------------
# Transmitter mode x beam model x aperture illumination
# ---------------------------------------------------------------------------------------------

# Written out independently of helioswarm.config.OPTICAL_BEAM_MODELS / MICROWAVE_BEAM_MODELS so
# that a change to those sets is caught here (documented in models.beam and M0.md).
DOCUMENTED_BEAMS_FOR_MODE = {
    "optical": {"airy_far_field", "gaussian_beam", "fresnel_hankel"},
    "microwave": {"aperture_far_field", "fresnel_hankel"},
}
# Beam models that fix the aperture illumination (others accept any illumination).
REQUIRED_ILLUMINATION = {
    "airy_far_field": "uniform_circular",
    "gaussian_beam": "truncated_gaussian",
}
MODES = ("optical", "microwave")
BEAMS = ("airy_far_field", "gaussian_beam", "fresnel_hankel", "aperture_far_field")
ILLUMINATIONS = ("uniform_circular", "truncated_gaussian", "parabolic_on_pedestal")


def _link(mvp: dict[str, Any], mode: str, beam: str, illum: str) -> dict[str, Any]:
    """Self-consistent config for (mode, beam, illumination); only the beam choice can clash."""
    data = microwave_data(mvp) if mode == "microwave" else copy.deepcopy(mvp)
    tx = data["transmitter"]
    tx.pop("gaussian_waist_radius_m", None)
    tx.pop("edge_taper_dB", None)
    tx["aperture_illumination"] = illum
    if illum == "truncated_gaussian":
        # a/w0 = 3 > default validity threshold 2.5 of the Gaussian-beam model.
        tx["gaussian_waist_radius_m"] = tx["aperture_diameter_m"] / 2.0 / 3.0
    if illum == "parabolic_on_pedestal":
        tx["edge_taper_dB"] = 10.0
    data["models"]["beam"] = beam
    return data


@pytest.mark.parametrize(
    ("mode", "beam", "illum"), list(itertools.product(MODES, BEAMS, ILLUMINATIONS))
)
def test_mode_beam_illumination_matrix(
    mvp_data: dict[str, Any], mode: str, beam: str, illum: str
) -> None:
    data = _link(mvp_data, mode, beam, illum)
    expected_ok = beam in DOCUMENTED_BEAMS_FOR_MODE[mode] and (
        REQUIRED_ILLUMINATION.get(beam, illum) == illum
    )
    if expected_ok:
        cfg = parse_config(data)
        assert (cfg.transmitter.mode, cfg.models.beam) == (mode, beam)
        return
    messages = _issue_messages(data)
    # Every reported problem is about the beam choice (nothing else in the config is wrong).
    assert messages, "expected rejection"
    assert all("models.beam=" in m for m in messages), messages
    if beam not in DOCUMENTED_BEAMS_FOR_MODE[mode]:
        assert any(f"not allowed with transmitter.mode='{mode}'" in m for m in messages)


# ---------------------------------------------------------------------------------------------
# Gaussian-beam validity (truncation ratio a / w0)
# ---------------------------------------------------------------------------------------------


def _gaussian(mvp: dict[str, Any], waist_m: float, beam: str) -> dict[str, Any]:
    # MVP transmitter: D = 10 m, so the aperture radius is a = 5 m.
    set_path(mvp, "transmitter.aperture_illumination", "truncated_gaussian")
    set_path(mvp, "transmitter.gaussian_waist_radius_m", waist_m)
    return set_path(mvp, "models.beam", beam)


def test_strongly_truncated_gaussian_rejected_for_gaussian_beam(mvp_data: dict[str, Any]) -> None:
    # w0 = a: 13.5 % of the Gaussian is clipped; untruncated model overstates peak ~2.2x.
    messages = _issue_messages(_gaussian(mvp_data, 5.0, "gaussian_beam"))
    assert len(messages) == 1, messages
    assert "models.gaussian_beam_min_truncation_ratio=2.5" in messages[0]
    assert "aperture radius / waist = 1 " in messages[0]
    assert "fresnel_hankel" in messages[0]


def test_strongly_truncated_gaussian_accepted_for_fresnel_hankel(
    mvp_data: dict[str, Any],
) -> None:
    cfg = parse_config(_gaussian(mvp_data, 5.0, "fresnel_hankel"))
    assert cfg.transmitter.gaussian_waist_radius_m == 5.0


def test_gaussian_truncation_threshold_boundary(mvp_data: dict[str, Any]) -> None:
    # a / w0 = 5 / 2 = 2.5 exactly equals the default threshold -> accepted (>=).
    parse_config(_gaussian(copy.deepcopy(mvp_data), 2.0, "gaussian_beam"))
    # 1e-9 relative above the boundary waist -> ratio just below 2.5 -> rejected.
    with pytest.raises(ConfigError, match="gaussian_beam_min_truncation_ratio"):
        parse_config(_gaussian(copy.deepcopy(mvp_data), 2.0 * (1 + 1e-9), "gaussian_beam"))
    # The threshold is configurable: a/w0 = 2 is accepted when the user relaxes it to 2.
    relaxed = _gaussian(copy.deepcopy(mvp_data), 2.5, "gaussian_beam")
    set_path(relaxed, "models.gaussian_beam_min_truncation_ratio", 2.0)
    parse_config(relaxed)


def test_waist_larger_than_aperture_radius_is_the_only_issue(mvp_data: dict[str, Any]) -> None:
    messages = _issue_messages(_gaussian(mvp_data, 5.5, "fresnel_hankel"))
    assert len(messages) == 1, messages
    assert messages[0].startswith("transmitter: transmitter.gaussian_waist_radius_m=5.5 exceeds")


@pytest.mark.parametrize(("r", "documented"), [(2.5, 0.9961), (2.0, 0.964), (1.0, 0.46)])
def test_documented_truncation_errors_match_quadrature(r: float, documented: float) -> None:
    """Check the numbers quoted in models.gaussian_beam_min_truncation_ratio.

    Fraunhofer on-axis amplitude of a radial aperture field E(rho) is proportional to
    integral E 2 pi rho drho, the power to integral |E|^2 2 pi rho drho. The ratio of
    (amplitude^2 / power) for the field truncated at a to that of the untruncated Gaussian is
    computed here by numerical quadrature (independent of the closed form in the docs).
    Tolerance: half a unit in the last quoted digit.
    """
    w0, a = 1.0, r

    def amp(upper: float) -> float:
        return quad(lambda p: math.exp(-(p**2) / w0**2) * 2 * math.pi * p, 0, upper)[0]

    def power(upper: float) -> float:
        return quad(lambda p: math.exp(-2 * p**2 / w0**2) * 2 * math.pi * p, 0, upper)[0]

    ratio = (amp(a) ** 2 / power(a)) / (amp(math.inf) ** 2 / power(math.inf))
    digits = len(str(documented).split(".")[1])
    assert ratio == pytest.approx(documented, abs=0.5 * 10.0**-digits)


# ---------------------------------------------------------------------------------------------
# Plane-parallel air-mass validity floor
# ---------------------------------------------------------------------------------------------


def test_min_elevation_floor_applies_only_with_atmosphere_scenario(
    mvp_data: dict[str, Any],
) -> None:
    # MVP: models.atmosphere='scenario', default floor 10 deg.
    parse_config(set_path(copy.deepcopy(mvp_data), "receiver.min_elevation_deg", 10.0))
    below = set_path(copy.deepcopy(mvp_data), "receiver.min_elevation_deg", 9.999)
    messages = _issue_messages(below)
    assert len(messages) == 1
    assert "plane_parallel_min_elevation_deg=10.0" in messages[0]
    # No atmosphere model -> no air mass -> horizon access is allowed.
    no_atm = set_path(copy.deepcopy(below), "models.atmosphere", "none")
    set_path(no_atm, "receiver.min_elevation_deg", 0.0)
    parse_config(no_atm)
    # A user-chosen floor is honoured.
    relaxed = set_path(copy.deepcopy(below), "atmosphere.plane_parallel_min_elevation_deg", 5.0)
    parse_config(relaxed)


def _kasten_young_airmass(elevation_deg: float) -> float:
    """Kasten & Young (1989), Appl. Opt. 28, 4735: X = 1/(cos z + 0.50572 (96.07995 - z)^-1.6364),
    z = zenith angle in degrees."""
    z = 90.0 - elevation_deg
    return 1.0 / (math.cos(math.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


@pytest.mark.parametrize(("el_deg", "documented"), [(10.0, 0.031), (5.0, 0.11), (3.0, 0.26)])
def test_documented_plane_parallel_airmass_error(el_deg: float, documented: float) -> None:
    """The percentages quoted in AtmosphereConfig must match Kasten & Young (half-digit tol.)."""
    excess = (1.0 / math.sin(math.radians(el_deg))) / _kasten_young_airmass(el_deg) - 1.0
    digits = len(str(documented).split(".")[1])
    assert excess == pytest.approx(documented, abs=0.5 * 10.0**-digits)


# ---------------------------------------------------------------------------------------------
# Settings that would be silently ignored
# ---------------------------------------------------------------------------------------------


def test_cloudy_scenario_accepted_with_atmosphere_scenario(mvp_data: dict[str, Any]) -> None:
    cfg = parse_config(set_path(mvp_data, "models.weather_scenario", "cloudy"))
    assert cfg.models.weather_scenario == "cloudy"


def test_integrator_step_rejected_with_two_body_model(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "orbit.model", "two_body_kepler")
    set_path(mvp_data, "orbit.include_planetary_perturbations", False)
    set_path(mvp_data, "orbit.include_solar_radiation_pressure", False)
    del_path(mvp_data, "orbit.cr")
    del_path(mvp_data, "orbit.area_to_mass_m2_kg")
    messages = _issue_messages(mvp_data)  # MVP keeps time.integrator_step_s = 300
    assert len(messages) == 1, messages
    assert "time.integrator_step_s is set but orbit.model='two_body_kepler'" in messages[0]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("integrator_rtol", 1e-12),  # even when equal to the default: it is explicitly set
        ("integrator_atol_position_m", 1e-2),
        ("integrator_atol_velocity_m_s", 1e-8),
        ("ephemeris_cache_step_s", 1800.0),
    ],
)
def test_l1_only_time_fields_rejected_with_two_body_model(
    mvp_data: dict[str, Any], name: str, value: float
) -> None:
    # Review finding (M2): treated like time.integrator_step_s (ignored by L0 -> rejected).
    set_path(mvp_data, "orbit.model", "two_body_kepler")
    set_path(mvp_data, "orbit.include_planetary_perturbations", False)
    set_path(mvp_data, "orbit.include_solar_radiation_pressure", False)
    del_path(mvp_data, "orbit.cr")
    del_path(mvp_data, "orbit.area_to_mass_m2_kg")
    del_path(mvp_data, "time.integrator_step_s")
    assert parse_config(mvp_data).orbit.model == "two_body_kepler"  # valid without the field
    set_path(mvp_data, f"time.{name}", value)
    messages = _issue_messages(mvp_data)
    assert len(messages) == 1, messages
    assert f"time.{name} is set but orbit.model='two_body_kepler' is analytic" in messages[0]


def test_perturbing_bodies_rejected_when_perturbations_are_off(mvp_data: dict[str, Any]) -> None:
    # Review finding (M2): mirrors the cr / area_to_mass_m2_kg rule for SRP.
    set_path(mvp_data, "orbit.include_planetary_perturbations", False)
    assert parse_config(mvp_data).orbit.include_planetary_perturbations is False
    set_path(mvp_data, "orbit.perturbing_bodies", ["JUPITER BARYCENTER"])
    messages = _issue_messages(mvp_data)
    expected = (
        "orbit.perturbing_bodies is set but include_planetary_perturbations=false ignores it; "
        "remove it or set include_planetary_perturbations=true"
    )
    assert any(expected in m for m in messages), messages


# ---------------------------------------------------------------------------------------------
# Output-grid tolerance
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "step_s"),
    [
        (36525.0, 7.0),  # 100 yr / 7 s: ratio 4.5e8 + 1/7; grid would end 1 s early
        (3652.5, 0.7),  # ratio 4.5e8 + 1/7; grid would end 0.1 s early
        (365.25, 3600.000003),  # 8766 steps end 0.026 s off
        (1.0, 86399.9999),  # single step 0.1 ms short of the window
    ],
)
def test_non_multiple_grid_rejected_even_for_large_ratios(
    mvp_data: dict[str, Any], days: float, step_s: float
) -> None:
    set_path(mvp_data, "time.duration_days", days)
    set_path(mvp_data, "time.output_step_s", step_s)
    messages = _issue_messages(mvp_data)
    assert len(messages) == 1, messages
    assert "not an integer multiple" in messages[0]


@pytest.mark.parametrize(
    ("days", "step_s", "n"),
    [
        (36525.0, 3600.0, 876_600),  # exact multiple at the longest window
        (36525.0, 1.0, 3_155_760_000),  # ratio > 5e8, exact
        (1.0, 0.1, 864_000),  # 0.1 is not representable in binary; must still be accepted
        (0.1, 864.0, 10),  # 0.1 d * 86400 = 8640.000000000002 in float64
    ],
)
def test_exact_multiple_grids_accepted(
    mvp_data: dict[str, Any], days: float, step_s: float, n: int
) -> None:
    set_path(mvp_data, "time.duration_days", days)
    set_path(mvp_data, "time.output_step_s", step_s)
    assert parse_config(mvp_data).time.n_output_intervals == n


# ---------------------------------------------------------------------------------------------
# SI angle accessors with distinct non-zero values
# ---------------------------------------------------------------------------------------------


def test_angle_accessors_convert_their_own_field(mvp_data: dict[str, Any]) -> None:
    values = {
        "orbit.inclination_deg": 10.0,
        "orbit.raan_deg": 20.0,
        "orbit.arg_periapsis_deg": 30.0,
        "orbit.true_anomaly_deg": 40.0,
        "receiver.sites[0].latitude_deg": -33.0,
        "receiver.sites[0].longitude_deg": 151.0,
        "receiver.min_elevation_deg": 15.0,
        "receiver.sun_avoidance_deg": 5.0,
    }
    for path, value in values.items():
        set_path(mvp_data, path, value)
    cfg = parse_config(mvp_data)
    site = cfg.receiver.sites[0]
    got = {
        "orbit.inclination_deg": cfg.orbit.inclination_rad,
        "orbit.raan_deg": cfg.orbit.raan_rad,
        "orbit.arg_periapsis_deg": cfg.orbit.arg_periapsis_rad,
        "orbit.true_anomaly_deg": cfg.orbit.true_anomaly_rad,
        "receiver.sites[0].latitude_deg": site.latitude_rad,
        "receiver.sites[0].longitude_deg": site.longitude_rad,
        "receiver.min_elevation_deg": cfg.receiver.min_elevation_rad,
        "receiver.sun_avoidance_deg": cfg.receiver.sun_avoidance_rad,
    }
    for path, value in values.items():
        # deg -> rad by pi/180 (independent of math.radians); 1e-15 relative = float rounding.
        assert got[path] == pytest.approx(value * math.pi / 180.0, rel=1e-15), path


# ---------------------------------------------------------------------------------------------
# Generic [0, 1] bounds of every efficiency-like dimensionless field
# ---------------------------------------------------------------------------------------------

_FRACTION_KEYWORDS = ("efficiency", "emissivity", "absorptivity", "transmittance", "view_factor")


def _fraction_paths() -> list[str]:
    cfg = load_config(MVP_CONFIG)
    return [
        row.path
        for row in provenance_table(cfg)
        if row.unit == "1" and any(k in row.path.rsplit(".", 1)[-1] for k in _FRACTION_KEYWORDS)
    ]


FRACTION_PATHS = _fraction_paths()


def test_fraction_field_discovery_is_complete() -> None:
    # Guards the discovery filter itself: these PDF §10.1 / §8 owners must all be covered.
    assert {
        "collector.pv_efficiency_reference",
        "collector.pv_efficiency_min",
        "collector.pv_efficiency_max",
        "collector.absorptivity",
        "transmitter.power_conditioning_efficiency",
        "transmitter.electrical_to_beam_efficiency",
        "transmitter.optics_or_rf_efficiency",
        "receiver.conversion_efficiency",
        "receiver.grid_efficiency",
        "thermal.emissivity",
        "thermal.radiator_view_factor",
        "thermal.pv_front_emissivity",
        "thermal.pv_back_emissivity",
        "atmosphere.optical_zenith_transmittance_clear",
        "atmosphere.optical_zenith_transmittance_cloudy",
    } <= set(FRACTION_PATHS)


@pytest.mark.parametrize("path", FRACTION_PATHS)
@pytest.mark.parametrize(
    ("value", "fragment"),
    [(1.0 + 1e-9, "less than or equal to 1"), (-1e-9, "greater than")],
)
def test_fraction_fields_are_bounded_to_unit_interval(
    path: str, value: float, fragment: str
) -> None:
    # Start from the resolved MVP config so that every default-valued field is present.
    data = resolved_dict(load_config(MVP_CONFIG))
    messages = _issue_messages(set_path(data, path, value))
    assert any(m.startswith(f"{path}: ") and fragment in m for m in messages), messages


# ---------------------------------------------------------------------------------------------
# Receiver thermal rejection capability (PDF §10.4)
# ---------------------------------------------------------------------------------------------


def test_receiver_heat_rejection_is_optional_and_settable(mvp_data: dict[str, Any]) -> None:
    assert parse_config(copy.deepcopy(mvp_data)).receiver.max_heat_rejection_W is None
    cfg = parse_config(set_path(mvp_data, "receiver.max_heat_rejection_W", 5.0e8))
    assert cfg.receiver.max_heat_rejection_W == 5.0e8


# ---------------------------------------------------------------------------------------------
# Config-hash regression pin
# ---------------------------------------------------------------------------------------------

# SHA-256 of the canonical JSON of configs/mvp_single_satellite.yaml. This value changes only
# when the MVP file, the schema (fields/defaults) or the canonical serialisation changes. Update
# it deliberately together with such a change (and bump schema_version if the change is
# backward-incompatible); an unexplained change re-keys every run manifest.
# History: M0 381090d7...7274; M1 review changed the thermal.space_temperature_K default from
# 2.725 to the cited Fixsen (2009) value 2.72548 K (constants.CMB_TEMPERATURE_K). With
# 2.725 set explicitly the M0 digest is reproduced, so that default is the only change.
# M2 (37999a09...5fc2 -> f2d46ab0...7561) added defaulted fields time.integrator_rtol,
# time.integrator_atol_position_m, time.integrator_atol_velocity_m_s,
# time.ephemeris_cache_step_s and orbit.perturbing_bodies; removing exactly those five keys
# from the canonical JSON reproduces the M1 digest (checked when re-pinning). schema_version
# is unchanged because every previously valid file still parses with the same meaning.
# M2 review (f2d46ab0...7561 -> b12c5a85...9c18) added the defaulted field
# frames.ephemeris_kernel_set (the kernel set is a physical input of the run); removing that
# one key from the canonical JSON reproduces the previous digest (checked when re-pinning).
# M3 (b12c5a85...9c18 -> d0087098...962d) added the defaulted numerical fields
# models.light_time_tolerance_s, models.light_time_max_iterations and
# models.earth_occultation_standoff_m; removing exactly those three keys from the canonical
# JSON reproduces b12c5a85...9c18 (checked when re-pinning). schema_version is unchanged.
MVP_CONFIG_HASH = "d0087098881aa08201665032373a01a3d9ee02b270784aa3c85eb6233695962d"


def test_mvp_config_hash_is_pinned() -> None:
    assert config_hash(load_config(MVP_CONFIG)) == MVP_CONFIG_HASH


def test_canonical_json_format_is_pinned() -> None:
    cfg = load_config(MVP_CONFIG)
    text = canonical_json(cfg)
    # Independent reconstruction of the documented format from the plain resolved data.
    expected = json.dumps(
        json.loads(text), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    assert text == expected
    # Float/None/str spellings that a serializer change would alter.
    for fragment in (
        '"area_m2":10000.0',
        '"wavelength_m":1e-06',
        '"max_input_power_W":1000000000000.0',
        '"gaussian_waist_radius_m":null',
        '"start_utc":"2030-01-01T00:00:00"',
        '"schema_version":1',
    ):
        assert fragment in text, fragment
    assert text.startswith('{"atmosphere":{')


# ---------------------------------------------------------------------------------------------
# M2 fields: L1 integrator tolerances, ephemeris cache, perturbing bodies
# ---------------------------------------------------------------------------------------------


def test_m2_integrator_defaults(mvp_data: dict[str, Any]) -> None:
    cfg = parse_config(mvp_data)
    assert cfg.time.integrator_rtol == 1e-12
    assert cfg.time.integrator_atol_position_m == 1e-3
    assert cfg.time.integrator_atol_velocity_m_s == 1e-9
    assert cfg.time.ephemeris_cache_step_s == 3600.0
    assert cfg.orbit.perturbing_bodies == (
        "MERCURY BARYCENTER",
        "VENUS BARYCENTER",
        "EARTH",
        "MOON",
        "MARS BARYCENTER",
        "JUPITER BARYCENTER",
        "SATURN BARYCENTER",
        "URANUS BARYCENTER",
        "NEPTUNE BARYCENTER",
    )


@pytest.mark.parametrize(
    ("path", "value", "fragment"),
    [
        # scipy raises rtol below 100 eps (2.2e-14) silently; the schema floor is 1e-13.
        ("time.integrator_rtol", 1e-14, "greater than or equal to 0.0000000000001"),
        ("time.integrator_rtol", 1e-5, "less than or equal to 0.000001"),
        ("time.integrator_atol_position_m", 0.0, "greater than 0"),
        ("time.integrator_atol_velocity_m_s", -1e-9, "greater than 0"),
        ("time.ephemeris_cache_step_s", 0.0, "greater than 0"),
        # Validated range of the Hermite cache (< 1 m for every DE440 body; review M2).
        ("time.ephemeris_cache_step_s", 7201.0, "less than or equal to 7200"),
        ("frames.ephemeris_kernel_set", "my-kernels", "Input should be"),
        ("orbit.perturbing_bodies", ["PLUTO BARYCENTER"], "Input should be"),
        ("orbit.perturbing_bodies", [], "at least 1 item"),
    ],
)
def test_m2_fields_are_range_checked(
    mvp_data: dict[str, Any], path: str, value: Any, fragment: str
) -> None:
    messages = _issue_messages(set_path(mvp_data, path, value))
    assert any(path in m and fragment in m for m in messages), messages


def test_duplicate_perturbing_bodies_are_rejected(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "orbit.perturbing_bodies", ["EARTH", "MOON", "EARTH"])
    messages = _issue_messages(mvp_data)
    expected = "orbit.perturbing_bodies lists bodies more than once: ['EARTH']"
    assert any(expected in m for m in messages), messages


def test_perturbing_bodies_subset_is_accepted(mvp_data: dict[str, Any]) -> None:
    set_path(mvp_data, "orbit.perturbing_bodies", ["JUPITER BARYCENTER", "SATURN BARYCENTER"])
    cfg = parse_config(mvp_data)
    assert cfg.orbit.perturbing_bodies == ("JUPITER BARYCENTER", "SATURN BARYCENTER")
