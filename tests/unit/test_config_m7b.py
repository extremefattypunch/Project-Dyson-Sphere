"""M7b schema additions: geocentric orbits, the geostationary preset, the beam-table keys, the
hash-stable ``omit_when_default`` fields and the ``extends`` mechanism."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from helioswarm.config import (
    BEAM_TABLE_FIELDS,
    ConfigError,
    ModelsConfig,
    OrbitConfig,
    approximations_in_force,
    canonical_json,
    config_hash,
    config_sources,
    dump_resolved_yaml,
    load_config,
    merge_config_mappings,
    omits_when_default,
    parse_config,
    read_config_mapping,
    resolved_dict,
)

from config_helpers import MVP_CONFIG, load_mvp_data

GEO_ORBIT: dict[str, Any] = {
    "model": "two_body_kepler",
    "initial_epoch_utc": "2030-01-01T00:00:00",
    "elements_center": "EARTH",
    "elements_frame": "J2000",
    "geostationary_longitude_deg": 0.0,
}


def _geo(mvp: dict[str, Any], **orbit: Any) -> dict[str, Any]:
    d = copy.deepcopy(mvp)
    d["orbit"] = {**GEO_ORBIT, **orbit}
    d["time"].pop("integrator_step_s", None)
    return d


def _messages(exc: pytest.ExceptionInfo[ConfigError]) -> str:
    return "\n".join(i.format() for i in exc.value.issues)


# ---------------------------------------------------------------------------------------------
# Hash stability of the new fields
# ---------------------------------------------------------------------------------------------


NEW_FIELDS: tuple[tuple[type, str], ...] = (
    (OrbitConfig, "geostationary_longitude_deg"),
    (ModelsConfig, "beam_evaluation"),
    *((ModelsConfig, name) for name in BEAM_TABLE_FIELDS),
)


@pytest.mark.parametrize(("cls", "name"), NEW_FIELDS, ids=lambda x: getattr(x, "__name__", x))
def test_new_fields_are_omitted_from_the_canonical_form_at_their_default(
    cls: type, name: str
) -> None:
    assert omits_when_default(cls, name)
    cfg = load_config(MVP_CONFIG)
    section = "orbit" if cls is OrbitConfig else "models"
    assert name not in resolved_dict(cfg)[section]
    assert name not in canonical_json(cfg)


def test_pre_m7b_fields_are_never_omitted() -> None:
    # Only the M7b additions carry the flag; an older field omitted at its default would
    # change every existing hash (e.g. orbit.eccentricity = 0 in the MVP).
    for name in OrbitConfig.model_fields:
        assert omits_when_default(OrbitConfig, name) == (name == "geostationary_longitude_deg")
    assert "eccentricity" in resolved_dict(load_config(MVP_CONFIG))["orbit"]


def test_new_fields_enter_the_hash_when_used(mvp_data: dict[str, Any]) -> None:
    base = parse_config(copy.deepcopy(mvp_data))
    d = copy.deepcopy(mvp_data)
    d["models"]["beam"] = "airy_far_field"
    d["models"]["beam_evaluation"] = "chebyshev_table"
    table = parse_config(d)
    assert resolved_dict(table)["models"]["beam_evaluation"] == "chebyshev_table"
    assert config_hash(table) != config_hash(base)
    # Explicitly writing the default is the default: same canonical form and hash.
    d["models"]["beam_evaluation"] = "per_epoch"
    assert config_hash(parse_config(d)) == config_hash(base)


def test_mvp_canonical_json_is_the_m7_text(mvp_data: dict[str, Any]) -> None:
    """Independent recomputation: the M7 digest from the plain model dump minus nothing."""
    cfg = load_config(MVP_CONFIG)
    dumped = cfg.model_dump(mode="json")
    for key in ("geostationary_longitude_deg",):
        assert dumped["orbit"].pop(key) is None
    for key in ("beam_evaluation", *BEAM_TABLE_FIELDS):
        dumped["models"].pop(key)
    # M8a keys (merged after M7b) are not hashed while off (config._hash_input): drop them too.
    assert dumped["models"].pop("weather_source") == "scenario"
    assert dumped.pop("weather") is None
    text = json.dumps(
        dumped, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    assert hashlib.sha256(text.encode("ascii")).hexdigest() == config_hash(cfg)


def test_geo_config_round_trips_through_its_resolved_dump(
    mvp_data: dict[str, Any], tmp_path: Path
) -> None:
    cfg = parse_config(_geo(mvp_data))
    p = tmp_path / "geo.yaml"
    p.write_text(dump_resolved_yaml(cfg), encoding="utf-8")
    again = load_config(p)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)
    assert resolved_dict(cfg)["orbit"]["geostationary_longitude_deg"] == 0.0


# ---------------------------------------------------------------------------------------------
# Geocentric orbit rules
# ---------------------------------------------------------------------------------------------


def test_geostationary_preset_is_accepted_and_exposes_its_accessors(
    mvp_data: dict[str, Any],
) -> None:
    o = parse_config(_geo(mvp_data, geostationary_longitude_deg=-75.0)).orbit
    assert o.is_geocentric
    assert o.is_geostationary_preset
    assert o.geostationary_longitude_rad == pytest.approx(-75.0 * 3.141592653589793 / 180.0)
    with pytest.raises(ValueError, match="derived from the ephemeris"):
        _ = o.semimajor_axis_m
    with pytest.raises(ValueError, match="derived from the ephemeris"):
        _ = o.true_anomaly_rad


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"elements_frame": "ECLIPJ2000"}, "requires orbit.elements_frame='J2000'"),
        ({"semimajor_axis_au": 2.8e-4}, "semimajor_axis_au is set but the geostationary"),
        ({"true_anomaly_deg": 10.0}, "true_anomaly_deg is set but the geostationary"),
        ({"inclination_deg": 1.0}, "inclination_deg=1.0 contradicts the geostationary"),
        ({"eccentricity": 0.01}, "eccentricity=0.01 contradicts the geostationary"),
        ({"raan_deg": 5.0}, "raan_deg=5.0 contradicts"),
        ({"phase_reference": "earth_relative"}, "undefined for orbit.elements_center='EARTH'"),
        (
            {"model": "prescribed_ephemeris_perturbed_test_particle"},
            "supports only orbit.model='two_body_kepler'",
        ),
        ({"elements_center": "SUN", "elements_frame": "ECLIPJ2000"}, "needs orbit.elements_center"),
    ],
)
def test_geocentric_rules(mvp_data: dict[str, Any], changes: dict[str, Any], message: str) -> None:
    d = _geo(mvp_data, **changes)
    if changes.get("model") == "prescribed_ephemeris_perturbed_test_particle":
        d["time"]["integrator_step_s"] = 300.0
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    assert message in _messages(exc)


def test_geocentric_orbits_name_the_station_keeping_approximation(
    mvp_data: dict[str, Any],
) -> None:
    """GEO runs record "L0 + station-keeping" in the manifest/report (M7b review, PDF §17)."""
    approx = approximations_in_force(parse_config(_geo(mvp_data)))
    geo = [a for a in approx if a.path == "orbit.elements_center"]
    assert len(geo) == 1
    assert geo[0].value == "EARTH"
    for word in ("lunisolar", "J2", "SRP", "station-keeping", "mass proxy"):
        assert word in geo[0].approximation
    assert "propagate_geocentric_point_masses" in geo[0].comparison_path
    # Heliocentric configurations are unaffected (the MVP names no approximation).
    assert approximations_in_force(parse_config(mvp_data)) == ()


def test_geocentric_elements_need_their_values_and_stay_outside_the_earth(
    mvp_data: dict[str, Any],
) -> None:
    d = _geo(mvp_data)
    del d["orbit"]["geostationary_longitude_deg"]
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    msg = _messages(exc)
    assert "orbit.semimajor_axis_au is required" in msg
    assert "orbit.true_anomaly_deg is required" in msg
    # 6000 km < WGS 84 a: inside the Earth.
    d["orbit"].update(semimajor_axis_au=6.0e6 / 1.495978707e11, true_anomaly_deg=0.0)
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    assert "is inside the Earth" in _messages(exc)
    d["orbit"].update(semimajor_axis_au=4.2164e7 / 1.495978707e11, inclination_deg=0.1)
    cfg = parse_config(d)
    assert cfg.orbit.is_geocentric
    assert not cfg.orbit.is_geostationary_preset


def test_heliocentric_orbits_still_need_their_elements(mvp_data: dict[str, Any]) -> None:
    d = copy.deepcopy(mvp_data)
    del d["orbit"]["semimajor_axis_au"]
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    assert "orbit.semimajor_axis_au is required" in _messages(exc)
    d = copy.deepcopy(mvp_data)
    d["orbit"]["elements_frame"] = "J2000"
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    assert "orbit.elements_center='SUN' requires orbit.elements_frame='ECLIPJ2000'" in _messages(
        exc
    )


def test_constellations_reject_geocentric_orbits_and_the_beam_table() -> None:
    from helioswarm.config import read_yaml

    from config_helpers import CONFIG_DIR

    demo = read_yaml(CONFIG_DIR / "constellation_demo.yaml")
    d = copy.deepcopy(demo)
    d["orbit"] = dict(GEO_ORBIT)
    d["time"].pop("integrator_step_s")
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    assert "geocentric orbits (orbit.elements_center='EARTH') are a single-satellite" in (
        _messages(exc)
    )
    d = copy.deepcopy(demo)
    d["models"]["beam_evaluation"] = "chebyshev_table"
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    assert "implemented for the single-satellite run only" in _messages(exc)


def test_beam_table_keys_are_rejected_without_the_table(mvp_data: dict[str, Any]) -> None:
    d = copy.deepcopy(mvp_data)
    d["models"]["beam_table_max_nodes"] = 17
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    assert "models.beam_table_max_nodes is set but models.beam_evaluation='per_epoch'" in (
        _messages(exc)
    )
    d["models"]["beam_evaluation"] = "chebyshev_table"
    assert parse_config(d).models.beam_table_max_nodes == 17
    d["models"]["beam_table_max_nodes"] = 20
    with pytest.raises(ConfigError) as exc:
        parse_config(d)
    assert "must be 2^k + 1" in _messages(exc)


# ---------------------------------------------------------------------------------------------
# extends
# ---------------------------------------------------------------------------------------------


def test_merge_config_mappings_merges_mappings_and_replaces_everything_else() -> None:
    base = {"a": {"x": 1, "y": {"p": 1, "q": 2}}, "lst": [1, 2], "s": "base", "n": 3}
    over = {"a": {"y": {"q": 20}, "z": 3}, "lst": [9], "n": None}
    out = merge_config_mappings(base, over)
    assert out == {
        "a": {"x": 1, "y": {"p": 1, "q": 20}, "z": 3},
        "lst": [9],
        "s": "base",
        "n": None,
    }
    # Inputs are not modified.
    assert base["a"]["y"] == {"p": 1, "q": 2}
    assert over == {
        "a": {"y": {"q": 20}, "z": 3},
        "lst": [9],
        "n": None,
    }


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_extends_reproduces_the_flat_configuration(tmp_path: Path) -> None:
    mvp = load_mvp_data()
    shared = {k: v for k, v in mvp.items() if k not in ("project", "orbit")}
    (tmp_path / "parts").mkdir()
    _write(tmp_path / "parts" / "shared.yaml", shared)
    _write(
        tmp_path / "parts" / "tech.yaml",
        {"extends": "shared.yaml", "receiver": {"radius_m": 1000}},
    )
    child = _write(
        tmp_path / "run.yaml",
        {"extends": "parts/tech.yaml", "project": mvp["project"], "orbit": mvp["orbit"]},
    )
    cfg = load_config(child)
    assert config_hash(cfg) == config_hash(load_config(MVP_CONFIG))
    assert config_sources(child) == (
        child.resolve(),
        (tmp_path / "parts" / "tech.yaml").resolve(),
        (tmp_path / "parts" / "shared.yaml").resolve(),
    )
    assert "extends" not in read_config_mapping(child)
    # The extending file wins.
    _write(
        tmp_path / "run2.yaml",
        {
            "extends": "parts/tech.yaml",
            "project": mvp["project"],
            "orbit": mvp["orbit"],
            "receiver": {"radius_m": 2500.0},
        },
    )
    assert load_config(tmp_path / "run2.yaml").receiver.radius_m == 2500.0


def test_extends_errors(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    _write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="circular 'extends' chain"):
        read_config_mapping(tmp_path / "a.yaml")
    _write(tmp_path / "c.yaml", {"extends": ["a.yaml"]})
    with pytest.raises(ConfigError, match="must be the path of one configuration file"):
        read_config_mapping(tmp_path / "c.yaml")
    _write(tmp_path / "d.yaml", {"extends": "missing.yaml"})
    with pytest.raises(ConfigError, match="cannot read file"):
        read_config_mapping(tmp_path / "d.yaml")
    (tmp_path / "e.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    _write(tmp_path / "f.yaml", {"extends": "e.yaml"})
    with pytest.raises(ConfigError, match="is not a mapping of sections"):
        read_config_mapping(tmp_path / "f.yaml")
