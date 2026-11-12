"""Every config shipped in configs/ must validate and round-trip through its resolved dump.

Run configurations are every ``configs/*.yaml`` except sweep definitions (``*_sweep.yaml``),
which must load as sweeps whose every grid point is a valid run configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helioswarm.config import config_hash, dump_resolved_yaml, load_config
from helioswarm.sweep import load_sweep

from config_helpers import CONFIG_DIR

SHIPPED = sorted(p for p in CONFIG_DIR.glob("*.yaml") if not p.stem.endswith("_sweep"))
SWEEPS = sorted(CONFIG_DIR.glob("*_sweep.yaml"))


def test_configs_directory_is_not_empty() -> None:
    assert SHIPPED, "no configs/*.yaml found"
    assert SWEEPS, "no configs/*_sweep.yaml found"


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_config_validates_and_round_trips(path: Path, tmp_path: Path) -> None:
    cfg = load_config(path)
    resolved = tmp_path / f"{path.stem}.resolved.yaml"
    resolved.write_text(dump_resolved_yaml(cfg), encoding="utf-8")
    again = load_config(resolved)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)


@pytest.mark.parametrize("path", SWEEPS, ids=lambda p: p.name)
def test_shipped_sweep_expands_to_valid_points(path: Path) -> None:
    plan = load_sweep(path)
    n = 1
    for values in plan.definition.parameters.values():
        n *= len(values)
    assert len(plan.points) == n
    assert len({config_hash(p.config) for p in plan.points}) == n
