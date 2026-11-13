"""M7b harmonised comparison configurations (configs/comparison/).

The shared technology blocks are literally the same values (one ``extends`` chain); within a
technology the resolved heliocentric and GEO configurations differ only in orbit/geometry keys
(``helioswarm.compare.ORBIT_GEOMETRY_PREFIXES``). Checked here independently by comparing the
plain resolved dictionaries leaf by leaf.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from helioswarm.compare import (
    HarmonisationError,
    harmonisation_violations,
    harmonised_assumptions,
    load_comparison,
)
from helioswarm.config import (
    ConfigError,
    config_hash,
    config_sources,
    dump_resolved_yaml,
    load_config,
    parse_config,
    read_config_mapping,
    resolved_dict,
)
from helioswarm.sweep import load_sweep

from config_helpers import CONFIG_DIR

CMP = CONFIG_DIR / "comparison"
RUN_FILES = sorted(
    p
    for p in CMP.glob("*.yaml")
    if p.stem in {"helio_1au_optical", "geo_optical", "helio_1au_microwave", "geo_microwave"}
)
FRAGMENTS = sorted(CMP.glob("_*.yaml"))
SWEEPS = sorted(CMP.glob("*_sweep.yaml"))
PAIRS = (("helio_1au_optical", "geo_optical"), ("helio_1au_microwave", "geo_microwave"))


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix[:-1]}[{i}]."))
    else:
        out[prefix[:-1]] = d
    return out


def test_the_four_cases_and_their_sweeps_exist() -> None:
    assert [p.stem for p in RUN_FILES] == sorted(
        ["helio_1au_optical", "geo_optical", "helio_1au_microwave", "geo_microwave"]
    )
    assert len(SWEEPS) == 4
    assert len(FRAGMENTS) == 3


@pytest.mark.parametrize("path", RUN_FILES, ids=lambda p: p.name)
def test_case_configs_validate_round_trip_and_use_the_shared_blocks(
    path: Path, tmp_path: Path
) -> None:
    cfg = load_config(path)
    again = tmp_path / "resolved.yaml"
    again.write_text(dump_resolved_yaml(cfg), encoding="utf-8")
    assert load_config(again) == cfg
    names = [p.name for p in config_sources(path)]
    assert names[0] == path.name
    assert names[-1] == "_shared.yaml"
    assert names[1] == f"_{cfg.transmitter.mode}.yaml"
    # The GEO optical beam is near-field: the GEO cases use Fresnel-Hankel on the table, the
    # heliocentric ones the far-field model at every epoch.
    geo = cfg.orbit.elements_center == "EARTH"
    assert (cfg.models.beam == "fresnel_hankel") == geo
    assert (cfg.models.beam_evaluation == "chebyshev_table") == geo


@pytest.mark.parametrize("path", FRAGMENTS, ids=lambda p: p.name)
def test_fragments_are_not_run_configurations(path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize(("helio", "geo"), PAIRS)
def test_resolved_configs_differ_only_in_orbit_and_geometry(helio: str, geo: str) -> None:
    a = _flatten(resolved_dict(load_config(CMP / f"{helio}.yaml")))
    b = _flatten(resolved_dict(load_config(CMP / f"{geo}.yaml")))
    differing = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    allowed = {k for k in differing if k.startswith(("orbit.", "project."))}
    allowed |= {"time.integrator_step_s"} & differing
    # The beam model of the propagation regime: far field (heliocentric), Fresnel-Hankel on
    # the validated table (GEO).
    regime = {"models.beam", "models.beam_evaluation"}
    assert differing - allowed == regime, sorted(differing - allowed)
    assert a["models.beam"] in ("airy_far_field", "aperture_far_field")
    assert b["models.beam"] == "fresnel_hankel"
    assert b["models.beam_evaluation"] == "chebyshev_table"
    # ... and they do differ in the orbit (the comparison is not vacuous).
    assert any(k.startswith("orbit.") for k in differing)
    # Shared technology values, spelled out for the record (the PDF §16 harmonisation list).
    for key in (
        "collector.area_m2",
        "collector.pv_efficiency_reference",
        "transmitter.electrical_to_beam_efficiency",
        "transmitter.aperture_diameter_m",
        "receiver.conversion_efficiency",
        "receiver.radius_m",
        "thermal.radiator_temperature_limit_K",
        "models.atmosphere",
        "time.output_step_s",
        "receiver.sites[0].longitude_deg",
    ):
        assert a[key] == b[key]


def test_harmonisation_check_catches_a_technology_difference() -> None:
    helio = load_config(CMP / "helio_1au_optical.yaml")
    geo = load_config(CMP / "geo_optical.yaml")
    assert harmonisation_violations([("h", helio), ("g", geo)]) == []
    d = read_config_mapping(CMP / "geo_optical.yaml")
    d = copy.deepcopy(d)
    d["receiver"]["conversion_efficiency"] = 0.5
    geo_better = parse_config(d)
    bad = harmonisation_violations([("h", helio), ("g", geo_better)])
    assert len(bad) == 1
    assert bad[0].startswith("receiver.conversion_efficiency")


def test_comparison_file_loads_with_base_points_and_assumption_scopes() -> None:
    plan = load_comparison(CMP / "comparison.yaml")
    assert [c.label for c in plan.cases] == [
        "helio_1au_optical",
        "geo_optical",
        "helio_1au_microwave",
        "geo_microwave",
    ]
    for c in plan.cases:
        base_pt = c.sweep.points[c.base_index]
        assert config_hash(base_pt.config) != ""  # validated
        assert (
            base_pt.config.transmitter.aperture_diameter_m
            == c.base_config.transmitter.aperture_diameter_m
        )
        assert base_pt.config.receiver.radius_m == c.base_config.receiver.radius_m
    arch = {c.label: c.architecture for c in plan.cases}
    assert arch["geo_optical"] == "geostationary"
    assert arch["helio_1au_optical"] == "heliocentric"
    scopes = {a.path: a.scope for a in harmonised_assumptions(plan)}
    assert scopes["collector.area_m2"] == "shared by all cases"
    assert scopes["transmitter.mode"] == "shared within technology"
    assert scopes["orbit.elements_center"] == "orbit/geometry"
    assert scopes["models.beam"] == "propagation regime (beam model by geometry)"
    assert scopes["mass_proxy.bus_mass_kg"] == "comparison (post-processing)"
    provs = {a.path: a.provenance for a in harmonised_assumptions(plan)}
    assert provs["collector.area_m2"] == "assumed"
    assert "sourced" in provs["constants.solar_constant_1au_W_m2"]


def test_load_comparison_refuses_an_unharmonised_case(tmp_path: Path) -> None:
    for f in CMP.glob("*.yaml"):
        (tmp_path / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    geo = yaml.safe_load((tmp_path / "geo_microwave.yaml").read_text(encoding="utf-8"))
    geo["receiver"] = {"conversion_efficiency": 0.95}  # a better rectenna for GEO only
    (tmp_path / "geo_microwave.yaml").write_text(yaml.safe_dump(geo), encoding="utf-8")
    with pytest.raises(HarmonisationError, match=r"receiver\.conversion_efficiency"):
        load_comparison(tmp_path / "comparison.yaml")


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_comparison_sweeps_expand(path: Path) -> None:
    plan = load_sweep(path)
    assert len(plan.points) == 9
    assert len({config_hash(p.config) for p in plan.points}) == 9


def test_heliocentric_mvp_configs_carry_none_of_the_m7b_keys() -> None:
    for name in ("mvp_single_satellite.yaml", "mvp_single_satellite_microwave.yaml"):
        text = yaml.safe_dump(resolved_dict(load_config(CONFIG_DIR / name)))
        for key in ("geostationary_longitude_deg", "beam_evaluation", "beam_table_"):
            assert key not in text


def test_flux_limit_sensitivity_differs_from_geo_optical_only_in_the_bounded_keys() -> None:
    """The bounded sensitivity (M7b review) is geo_optical plus an ASSUMED receiver flux limit
    and a 100 m receiver; everything else is literally the comparison case."""
    sens = CMP / "sensitivity" / "geo_optical_flux_limited.yaml"
    a = _flatten(resolved_dict(load_config(CMP / "geo_optical.yaml")))
    b = _flatten(resolved_dict(load_config(sens)))
    differing = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    differing -= {k for k in differing if k.startswith("project.")}
    assert differing == {"receiver.max_power_density_W_m2", "receiver.radius_m"}
    assert b["receiver.max_power_density_W_m2"] == 1.0e4
    assert a.get("receiver.max_power_density_W_m2") is None
    plan = load_sweep(CMP / "sensitivity" / "geo_optical_flux_limit_sweep.yaml")
    assert [p.config.transmitter.aperture_diameter_m for p in plan.points] == [10.0, 30.0, 100.0]
    # Not part of the harmonised comparison: the four case files and their sweeps only.
    assert sens not in RUN_FILES
