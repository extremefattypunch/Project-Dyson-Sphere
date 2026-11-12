"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from config_helpers import load_mvp_data


@pytest.fixture
def mvp_data() -> dict[str, Any]:
    """Fresh, mutable copy of the parsed MVP optical configuration mapping."""
    return load_mvp_data()
