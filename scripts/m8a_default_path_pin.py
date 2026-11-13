"""Content hash of the 2-day MVP time series on the default (``weather_source = scenario``) path.

M8a reproduction check (docs/milestones/M8a.md C/D): run with the M7 code and with the M8a code
and compare; both gave ``34c97c9fa1f94a5161fd2fe188a4f82cff25375c3a4d164ee4f77bba04147e2b``,
which ``tests/integration/test_weather_run.py`` pins. Usage (from the repository root, after
``source scripts/cluster_env.sh`` and the worktree overlay)::

    uv run python scripts/m8a_default_path_pin.py
    # M7 code: git archive e86f8e7 src | tar -x -C <dir>; PYTHONPATH=<dir>/src uv run python ...
"""

from __future__ import annotations

import numpy as np

import helioswarm
from helioswarm.config import parse_config, read_yaml
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.io import table_content_sha256, timeseries_table
from helioswarm.kernels import KernelSet
from helioswarm.system import run_single_satellite


def main() -> None:
    """Print the package location and the content hash."""
    print("helioswarm from", helioswarm.__file__)
    data = read_yaml("configs/mvp_single_satellite.yaml")
    data["time"]["duration_days"] = 2.0
    cfg = parse_config(data)
    with KernelSet.from_manifest().loaded() as k:
        res = run_single_satellite(
            cfg, ephemeris=SpiceEphemeris(k), kernels=k, rng=np.random.default_rng(1)
        )
    print("content", table_content_sha256(timeseries_table(res)))


if __name__ == "__main__":
    main()
