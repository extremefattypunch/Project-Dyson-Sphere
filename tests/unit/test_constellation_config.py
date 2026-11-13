"""M7 configuration schema: orbit phase reference, constellation rings, receiver capacity,
scheduler and mass proxy (no kernels needed)."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest

from helioswarm.config import (
    ConfigError,
    leaf_paths,
    parse_config,
    read_yaml,
    satellite_config,
)
from helioswarm.constellation import chunk_indices, constellation_members, site_capacities
from helioswarm.link_budget import receiver_model_for_site, receiver_model_from_config
from helioswarm.propagate import (
    argument_of_latitude_for_longitude_rad,
    max_stack_size,
)

from config_helpers import CONFIG_DIR, MVP_CONFIG

DEMO = CONFIG_DIR / "constellation_demo.yaml"


def _demo() -> dict[str, Any]:
    return copy.deepcopy(read_yaml(DEMO))


def _messages(data: dict[str, Any]) -> list[str]:
    with pytest.raises(ConfigError) as exc:
        parse_config(data)
    return [i.message for i in exc.value.issues]


def test_shipped_constellation_configs_are_valid_and_n1_matches_the_mvp() -> None:
    demo = parse_config(_demo())
    assert demo.constellation is not None
    assert demo.constellation.n_satellites == 10
    assert len(demo.receiver.sites) == 3
    n1 = parse_config(read_yaml(CONFIG_DIR / "constellation_n1_mvp.yaml"))
    mvp = parse_config(read_yaml(MVP_CONFIG))
    sat = satellite_config(n1, 0, n1.orbit.true_anomaly_deg)
    # The one-satellite constellation's satellite is the MVP except for the project name.
    a = sat.model_dump(mode="json")
    b = mvp.model_dump(mode="json")
    a["project"].pop("name"), a["project"].pop("output_dir")
    b["project"].pop("name"), b["project"].pop("output_dir")
    assert a == b


def test_ring_elements_phase_and_technology_overrides_reach_the_satellite() -> None:
    d = _demo()
    d["constellation"]["rings"].append(
        {
            "name": "inner",
            "n_satellites": 2,
            "semimajor_axis_au": 0.8,
            "inclination_deg": 3.0,
            "phase_reference": "absolute",
            "technology": {"collector_area_m2": 5000.0, "transmitter_aperture_diameter_m": 20.0},
        }
    )
    cfg = parse_config(d)
    members = constellation_members(cfg, np.random.default_rng(0))
    assert len(members) == 12
    inner = [m for m in members if m.ring_name == "inner"]
    assert [m.index for m in inner] == [10, 11]
    c = inner[1].config
    assert c.orbit.semimajor_axis_au == 0.8
    assert c.orbit.inclination_deg == 3.0
    assert c.orbit.phase_reference == "absolute"
    assert c.collector.area_m2 == 5000.0
    assert c.transmitter.aperture_diameter_m == 20.0
    # phase offset defaults to orbit.true_anomaly_deg (30); 2 even phases over 360 deg
    assert [m.phase_angle_deg for m in inner] == [30.0, 210.0]
    outer = members[0].config
    assert outer.collector.area_m2 == 10000.0
    assert outer.orbit.phase_reference == "earth_relative"
    assert outer.constellation is None


def test_ring_overrides_are_validated_with_the_satellite_rules() -> None:
    d = _demo()
    # perihelion inside the Sun: a(1 - e) = 0.001 au
    d["constellation"]["rings"][0]["semimajor_axis_au"] = 0.01
    d["constellation"]["rings"][0]["eccentricity"] = 0.9
    msgs = _messages(d)
    assert any("constellation.rings[0]" in m and "inside the Sun" in m for m in msgs), msgs


@pytest.mark.parametrize(
    ("edit", "fragment"),
    [
        (
            lambda d: d["constellation"]["rings"].append(dict(d["constellation"]["rings"][0])),
            "unique",
        ),
        (lambda d: d["constellation"].update(rings=[]), "at least 1"),
        (lambda d: d["constellation"]["rings"][0].update(n_satellites=0), "greater than or equal"),
        (lambda d: d["constellation"]["rings"][0].update(phase_span_deg=0.0), "greater than"),
        (
            lambda d: d["constellation"]["scheduler"].update(duty_cycle_max=0.5),
            "must be set together",
        ),
        (
            lambda d: d["constellation"]["scheduler"].update(
                duty_cycle_max=0.1, duty_cycle_window_s=3896.0 * 5
            ),
            "allows no transmitting epoch",
        ),
        (lambda d: d["constellation"]["scheduler"].update(policy="best"), "Input should be"),
        (lambda d: d["receiver"].update(max_simultaneous_beams=0), "greater than or equal"),
        (lambda d: d["constellation"]["mass_proxy"].pop("bus_mass_kg"), "missing required field"),
        (
            lambda d: d["thermal"].pop("radiator_specific_mass_kg_m2"),
            "mass_proxy needs thermal.radiator_specific_mass_kg_m2",
        ),
    ],
)
def test_invalid_constellation_settings_are_rejected(edit: Any, fragment: str) -> None:
    d = _demo()
    edit(d)
    msgs = _messages(d)
    assert any(fragment in m for m in msgs), msgs


def test_earth_relative_phase_is_rejected_for_polar_orbits() -> None:
    d = _demo()
    d["orbit"]["inclination_deg"] = 90.0
    assert any("polar orbit" in m for m in _messages(d))
    d["orbit"]["phase_reference"] = "absolute"
    parse_config(d)  # a polar orbit with an absolute phase is fine


def test_argument_of_latitude_inverts_the_longitude_relation() -> None:
    """Independent forward map: for u on a grid and several (i, Omega), the ecliptic longitude
    of cos u P + sin u Q is atan2(y, x); the inverse must return u (1e-12 rad: round-off)."""
    for i_deg in (0.0, 7.0, 60.0, 120.0, 179.0):
        for om_deg in (0.0, 75.0, 300.0):
            i, om = math.radians(i_deg), math.radians(om_deg)
            for u in np.linspace(0.05, 2 * math.pi - 0.05, 37):
                x = math.cos(u) * math.cos(om) - math.sin(u) * math.sin(om) * math.cos(i)
                y = math.cos(u) * math.sin(om) + math.sin(u) * math.cos(om) * math.cos(i)
                got = argument_of_latitude_for_longitude_rad(
                    math.atan2(y, x), inclination_rad=i, raan_rad=om
                )
                assert math.remainder(got - u, 2 * math.pi) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="polar"):
        argument_of_latitude_for_longitude_rad(0.3, inclination_rad=math.pi / 2, raan_rad=0.0)


def test_site_capacities_and_per_site_receiver_models() -> None:
    cfg = parse_config(_demo())
    caps = site_capacities(cfg)
    assert [c.max_beams for c in caps] == [8, 8, 4]
    assert [c.max_accepted_W for c in caps] == [1.0e12, 1.0e12, 2.0e5]
    mvp = parse_config(read_yaml(MVP_CONFIG))
    assert receiver_model_for_site(mvp.receiver, mvp.receiver.sites[0]) == (
        receiver_model_from_config(mvp.receiver)
    )
    d = _demo()
    d["receiver"]["max_heat_rejection_W"] = 6.0e3  # L = Q / (1 - eta_rx) = 1e4 W
    caps = site_capacities(parse_config(d))
    assert caps[0].max_accepted_W == pytest.approx(6.0e3 / (1 - 0.40), rel=1e-15)
    assert caps[2].max_accepted_W == pytest.approx(6.0e3 / (1 - 0.40), rel=1e-15)


def test_constellation_keys_are_sweepable_leaf_paths() -> None:
    paths = set(leaf_paths(parse_config(_demo())))
    for p in (
        "constellation.rings[0].n_satellites",
        "constellation.rings[0].phase_distribution",
        "constellation.scheduler.policy",
        "constellation.mass_proxy.bus_mass_kg",
        "receiver.sites[2].max_input_power_W",
        "orbit.phase_reference",
    ):
        assert p in paths, p


def test_chunks_cover_every_satellite_once_and_stack_cap() -> None:
    for n, c in ((1000, 5), (7, 3), (3, 5)):
        got = [i for k in range(c) for i in chunk_indices(n, k, c)]
        assert got == list(range(n))
    with pytest.raises(ValueError, match="chunk"):
        chunk_indices(10, 3, 3)
    # (1e-12 / 1e-13)^2 = 100 satellites per stack keep rtol / sqrt(n) >= 1e-13
    assert max_stack_size(1e-12) == 100
    assert max_stack_size(1e-13) == 1
    assert max_stack_size(4e-13) == 16


def test_earth_relative_angle_is_never_read_as_a_true_anomaly() -> None:
    """orbit.elements_from_config refuses an Earth-relative orbit without the anomaly resolved
    from the ephemeris, and refuses a resolved anomaly for an absolute one."""
    from helioswarm.orbit import elements_from_config

    rel = parse_config(_demo()).orbit
    with pytest.raises(ValueError, match="earth_relative"):
        elements_from_config(rel, epoch_tdb_s=0.0)
    el = elements_from_config(rel, epoch_tdb_s=0.0, true_anomaly_rad=1.25)
    assert el.nu_rad == 1.25
    absolute = parse_config(read_yaml(MVP_CONFIG)).orbit
    with pytest.raises(ValueError, match="absolute"):
        elements_from_config(absolute, epoch_tdb_s=0.0, true_anomaly_rad=1.0)
    assert elements_from_config(absolute, epoch_tdb_s=0.0).nu_rad == math.radians(30.0)
