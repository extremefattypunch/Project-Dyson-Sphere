"""Test helpers for configuration mappings (importable as ``config_helpers``)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from helioswarm.config import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
MVP_CONFIG = CONFIG_DIR / "mvp_single_satellite.yaml"


def load_mvp_data() -> dict[str, Any]:
    """Fresh, mutable copy of the parsed MVP optical configuration mapping."""
    data = read_yaml(MVP_CONFIG)
    assert isinstance(data, dict)
    return copy.deepcopy(data)


def _split(path: str) -> list[str | int]:
    parts: list[str | int] = []
    for token in path.split("."):
        if "[" in token:
            name, idx = token[:-1].split("[")
            parts.extend([name, int(idx)])
        else:
            parts.append(token)
    return parts


def set_path(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Set ``data[a][b]... = value`` for dotted ``path`` (``name[i]`` indexes lists)."""
    parts = _split(path)
    node: Any = data
    for part in parts[:-1]:
        if isinstance(part, str) and part not in node:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return data


def del_path(data: dict[str, Any], path: str) -> dict[str, Any]:
    """Delete the entry at dotted ``path``."""
    parts = _split(path)
    node: Any = data
    for part in parts[:-1]:
        node = node[part]
    del node[parts[-1]]
    return data


def microwave_data(base: dict[str, Any]) -> dict[str, Any]:
    """Turn the MVP optical mapping into a self-consistent microwave configuration."""
    data = copy.deepcopy(base)
    data["project"]["name"] = "single_satellite_microwave_test"
    data["transmitter"].update(
        {
            "mode": "microwave",
            # 2.45 GHz: lambda = c / f = 299792458 / 2.45e9 m
            "wavelength_m": 299_792_458.0 / 2.45e9,
            "aperture_diameter_m": 1000.0,
            "pointing_sigma_rad": 1.0e-6,
            "aperture_illumination": "parabolic_on_pedestal",
            "edge_taper_dB": 10.0,
            "array_element_spacing_m": 0.06,
        }
    )
    data["receiver"]["technology"] = "rectenna"
    data["models"]["beam"] = "aperture_far_field"
    return data
