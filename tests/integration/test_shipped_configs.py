"""Every config shipped in configs/ must validate and round-trip through its resolved dump."""

from __future__ import annotations

from pathlib import Path

import pytest

from helioswarm.config import config_hash, dump_resolved_yaml, load_config

from config_helpers import CONFIG_DIR

SHIPPED = sorted(CONFIG_DIR.glob("*.yaml"))


def test_configs_directory_is_not_empty() -> None:
    assert SHIPPED, "no configs/*.yaml found"


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.name)
def test_shipped_config_validates_and_round_trips(path: Path, tmp_path: Path) -> None:
    cfg = load_config(path)
    resolved = tmp_path / f"{path.stem}.resolved.yaml"
    resolved.write_text(dump_resolved_yaml(cfg), encoding="utf-8")
    again = load_config(resolved)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)
