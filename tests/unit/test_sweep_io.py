"""Sweep expansion/array splitting/collation (helioswarm.sweep) and run-directory handling
(helioswarm.io), without running the physics."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from helioswarm.config import ConfigError
from helioswarm.io import (
    RunDirectoryError,
    prepare_run_dir,
    resolve_run_dir,
    write_json,
)
from helioswarm.metrics import summarize_timeseries
from helioswarm.sweep import (
    SWEEP_SUMMARY_FILE,
    collate_sweep,
    load_sweep,
    points_for_task,
    prepare_sweep_dir,
)
from summary_helpers import INPUTS, LAYOUT1_SUMMARY, synthetic_table

from config_helpers import MVP_CONFIG


def _sweep(tmp_path: Path, params: dict[str, Any], **extra: Any) -> Path:
    doc = {
        "sweep_schema_version": 1,
        "name": "t",
        "base_config": str(MVP_CONFIG),
        "parameters": params,
        **extra,
    }
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_grid_is_cartesian_in_key_order_with_swept_provenance(tmp_path: Path) -> None:
    plan = load_sweep(
        _sweep(
            tmp_path,
            {"orbit.semimajor_axis_au": [0.7, 1.0], "receiver.radius_m": [1000.0, 5000.0, 9e3]},
            overrides={"time.duration_days": 2.0},
        )
    )
    assert len(plan.points) == 6
    # Last key varies fastest (itertools.product order).
    assert [dict(p.values)["receiver.radius_m"] for p in plan.points[:3]] == [1000.0, 5000.0, 9e3]
    assert plan.points[3].config.orbit.semimajor_axis_au == 1.0
    p4 = plan.points[4].config
    assert p4.receiver.radius_m == 5000.0
    assert p4.time.duration_days == 2.0
    assert p4.project.name == "t_p0004"
    assert p4.provenance_overrides == {
        "orbit.semimajor_axis_au": "swept",
        "receiver.radius_m": "swept",
    }


def test_invalid_sweeps_fail_early(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not a configuration leaf"):
        load_sweep(_sweep(tmp_path, {"receiver.radius": [1.0]}))
    with pytest.raises(ConfigError, match="duplicate"):
        load_sweep(_sweep(tmp_path, {"receiver.radius_m": [1.0, 1.0]}))
    # A point that violates the schema (e > 1) is rejected before anything runs.
    with pytest.raises(ConfigError, match="eccentricity"):
        load_sweep(_sweep(tmp_path, {"orbit.eccentricity": [0.0, 1.5]}))


def test_array_tasks_partition_the_grid() -> None:
    n, count = 45, 9
    parts = [points_for_task(n, i, count) for i in range(count)]
    flat = sorted(i for p in parts for i in p)
    assert flat == list(range(n))
    assert all(len(p) == 5 for p in parts)
    assert points_for_task(5, 0, 1) == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="array_index"):
        points_for_task(5, 3, 3)


def test_sweep_dir_and_collation_record_failures(tmp_path: Path) -> None:
    plan = load_sweep(_sweep(tmp_path, {"receiver.radius_m": [1000.0, 2000.0, 3000.0, 4000.0]}))
    out = prepare_sweep_dir(plan, tmp_path / "out")
    assert prepare_sweep_dir(plan, out) == out  # same definition: shared by array tasks
    other = load_sweep(_sweep(tmp_path, {"receiver.radius_m": [7000.0]}))
    with pytest.raises(FileExistsError, match="different sweep"):
        prepare_sweep_dir(other, out)
    # A complete, valid summary record (collation validates the layout): 12.5 W constant, rated
    # power 0 (capacity factor undefined), reference thermally infeasible.
    inputs = dataclasses.replace(INPUTS, rated_power_W=0.0, reference_thermal_feasible=False)
    summary = summarize_timeseries(
        synthetic_table(5, 10.0, grid_W=np.full(5, 12.5)), inputs
    ).to_json_dict()
    write_json(out / "points" / "p0000" / "summary.json", summary | {"timeseries_saved": False})
    write_json(out / "points" / "p0000" / "manifest.json", {})
    write_json(
        out / "points" / "p0001" / "error.json",
        {"error_type": "BeamModelValidityError", "error": "N_F too large"},
    )
    # p0003: killed between summary.json and report.md (time series saved, no report): the
    # resume logic recomputes it, so collation must not call it "ok".
    write_json(out / "points" / "p0003" / "summary.json", summary | {"timeseries_saved": True})
    write_json(out / "points" / "p0003" / "manifest.json", {})
    t = pq.read_table(collate_sweep(plan, out)).to_pylist()
    assert [r["status"] for r in t] == ["ok", "failed", "missing", "incomplete"]
    assert t[0]["p_grid_mean_W"] == 12.5
    assert t[0]["capacity_factor"] is None
    assert t[0]["reference_thermal_feasible"] is False
    assert t[0]["reference_power_density_abort"] is False
    assert t[3]["p_grid_mean_W"] is None
    assert not list(out.glob(f".{SWEEP_SUMMARY_FILE}.*"))  # no temporary left behind
    assert t[1]["error"] == "BeamModelValidityError: N_F too large"
    assert [r["receiver.radius_m"] for r in t] == [1000.0, 2000.0, 3000.0, 4000.0]
    assert (out / SWEEP_SUMMARY_FILE).is_file()

    # A point summary of layout helioswarm-summary/1 (written by M6, before the orbital
    # diagnostics existed) is still collated; its three diagnostics columns are null.
    legacy = json.loads(LAYOUT1_SUMMARY.read_text(encoding="utf-8"))
    write_json(out / "points" / "p0002" / "summary.json", legacy)
    write_json(out / "points" / "p0002" / "manifest.json", {})
    (out / "points" / "p0002" / "report.md").write_text("x", encoding="utf-8")
    t = pq.read_table(collate_sweep(plan, out)).to_pylist()
    assert t[2]["status"] == "ok"
    assert t[2]["availability"] == 0.0
    for col in ("light_time_min_s", "light_time_max_s", "occultation_fraction"):
        assert t[2][col] is None, col
        assert t[0][col] is not None, col  # layout-2 point: present


def test_run_dir_refuses_overwrite_and_only_replaces_run_dirs(tmp_path: Path) -> None:
    d = tmp_path / "run"
    prepare_run_dir(d, overwrite=False)
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RunDirectoryError, match="--overwrite"):
        prepare_run_dir(d, overwrite=False)
    prepare_run_dir(d, overwrite=True)
    assert not any(d.iterdir())
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RunDirectoryError, match="not recognisably"):
        prepare_run_dir(foreign, overwrite=True)
    assert (foreign / "precious.txt").is_file()


def test_default_run_dir_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mvp_data: dict[str, Any]
) -> None:
    from helioswarm.config import parse_config

    cfg = parse_config(mvp_data)
    monkeypatch.setenv("HELIOSWARM_RUNS_DIR", str(tmp_path / "runs"))
    assert resolve_run_dir(cfg, None) == (tmp_path / "runs" / cfg.project.name).resolve()
    assert resolve_run_dir(cfg, tmp_path / "x") == (tmp_path / "x").resolve()


def test_json_writer_maps_non_finite_to_null(tmp_path: Path) -> None:
    p = write_json(tmp_path / "a.json", {"b": math.nan, "a": [1.0, math.inf]})
    assert json.loads(p.read_text()) == {"a": [1.0, None], "b": None}
    assert p.read_text().index('"a"') < p.read_text().index('"b"')  # sorted keys


def test_sweep_dir_creation_is_safe_for_concurrent_array_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from concurrent.futures import ThreadPoolExecutor

    import helioswarm.sweep as sweep_mod

    plan = load_sweep(_sweep(tmp_path, {"receiver.radius_m": [1000.0, 2000.0]}))
    out = tmp_path / "race"
    # Many tasks starting at once (the failure seen with a 5-task array): all must succeed.
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda _: prepare_sweep_dir(plan, out), range(16)))
    assert all(r == out.resolve() for r in results)
    assert [p.name for p in out.iterdir()] == ["sweep_definition.yaml"]  # no temp files left
    # A task that loses the link race verifies the winner's definition instead of failing.
    out2 = tmp_path / "race2"
    real_link = os.link

    def lose(src: Any, dst: Any) -> None:
        real_link(src, dst)
        raise FileExistsError(dst)

    monkeypatch.setattr(sweep_mod.os, "link", lose)
    assert prepare_sweep_dir(plan, out2) == out2.resolve()
