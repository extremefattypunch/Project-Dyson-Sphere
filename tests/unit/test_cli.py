"""Tests of the ``helioswarm validate-config`` command (in-process and as a subprocess)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from helioswarm import __version__
from helioswarm.cli import main
from helioswarm.config import config_hash, load_config

from config_helpers import MVP_CONFIG, set_path


def _write(tmp_path: Path, data: dict[str, Any], name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_valid_config_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-config", str(MVP_CONFIG)]) == 0
    out = capsys.readouterr()
    assert out.out.startswith("OK: ")
    assert config_hash(load_config(MVP_CONFIG)) in out.out
    assert out.err == ""


def test_invalid_config_exits_one_with_field_named(
    mvp_data: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_path(mvp_data, "receiver.conversion_efficiency", 1.4)
    set_path(mvp_data, "orbit.eccentricity", 1.0)
    path = _write(tmp_path, mvp_data)
    assert main(["validate-config", str(path)]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "2 problems" in out.err
    assert "receiver.conversion_efficiency: Input should be less than or equal to 1" in out.err
    assert "(got: 1.4)" in out.err
    assert "orbit.eccentricity: Input should be less than 1 (got: 1.0)" in out.err


def test_contradictory_config_exits_one(
    mvp_data: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_path(mvp_data, "models.beam", "aperture_far_field")
    path = _write(tmp_path, mvp_data)
    assert main(["validate-config", str(path)]) == 1
    err = capsys.readouterr().err
    assert "models.beam='aperture_far_field' is not allowed with transmitter.mode='optical'" in err


def test_missing_file_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-config", str(tmp_path / "nope.yaml")]) == 1
    assert "cannot read file" in capsys.readouterr().err


def test_show_provenance_lists_every_leaf(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-config", str(MVP_CONFIG), "--show-provenance"]) == 0
    lines = capsys.readouterr().out.splitlines()
    solar = [ln for ln in lines if ln.startswith("constants.solar_constant_1au_W_m2 ")]
    assert len(solar) == 1
    assert "1361.0" in solar[0]
    assert "sourced" in solar[0]
    assert "IAU 2015" in solar[0]
    area = [ln for ln in lines if ln.startswith("collector.area_m2 ")]
    assert len(area) == 1
    assert "m^2" in area[0]
    assert "assumed" in area[0]
    assert any(ln.startswith("receiver.sites[0].latitude_deg ") for ln in lines)


def test_show_provenance_marks_overrides(
    mvp_data: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mvp_data["provenance_overrides"] = {"receiver.radius_m": "swept"}
    path = _write(tmp_path, mvp_data)
    assert main(["validate-config", str(path), "--show-provenance"]) == 0
    out = capsys.readouterr().out
    (line,) = [ln for ln in out.splitlines() if ln.startswith("receiver.radius_m ")]
    assert "swept*" in line
    assert "relabelled by provenance_overrides" in out


def test_write_resolved_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "sub" / "resolved.yaml"
    assert main(["validate-config", str(MVP_CONFIG), "--write-resolved", str(target)]) == 0
    assert str(target) in capsys.readouterr().out
    assert load_config(target) == load_config(MVP_CONFIG)


def test_invalid_config_does_not_write_resolved(
    mvp_data: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_path(mvp_data, "collector.area_m2", -1.0)
    path = _write(tmp_path, mvp_data)
    target = tmp_path / "resolved.yaml"
    assert main(["validate-config", str(path), "--write-resolved", str(target)]) == 1
    capsys.readouterr()
    assert not target.exists()


def test_usage_errors_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in ([], ["no-such-command"], ["validate-config"]):
        with pytest.raises(SystemExit) as info:
            main(argv)
        assert info.value.code == 2, argv
    capsys.readouterr()


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"helioswarm {__version__}"


def test_console_script_exit_codes(mvp_data: dict[str, Any], tmp_path: Path) -> None:
    """The installed ``helioswarm`` entry point returns 0/1 as a real process."""
    script = Path(sys.executable).parent / "helioswarm"
    assert script.exists(), "console script not installed (run `uv sync`)"
    ok = subprocess.run(
        [str(script), "validate-config", str(MVP_CONFIG)], capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.startswith("OK: ")
    set_path(mvp_data, "transmitter.wavelength_m", 0.1224)
    bad = subprocess.run(
        [str(script), "validate-config", str(_write(tmp_path, mvp_data))],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert "transmitter.wavelength_m=0.1224 is outside the optical range" in bad.stderr


def test_module_invocation(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "helioswarm.cli", "validate-config", str(MVP_CONFIG)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_validate_config_accepts_sweep_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The shipped sweep validates as a sweep (45 points), not as a run configuration.
    sweep = MVP_CONFIG.parent / "optical_1au_sweep.yaml"
    assert main(["validate-config", str(sweep)]) == 0
    out = capsys.readouterr()
    assert "valid helioswarm sweep file" in out.out
    assert "45 points" in out.out
    assert out.err == ""
    # Provenance/resolved output is for run configurations only.
    assert main(["validate-config", str(sweep), "--show-provenance"]) == 1
    assert "not sweep files" in capsys.readouterr().err
    # A sweep with an invalid point (e >= 1) is rejected without running anything.
    bad = _write(
        tmp_path,
        {
            "sweep_schema_version": 1,
            "name": "bad",
            "base_config": str(MVP_CONFIG),
            "parameters": {"orbit.eccentricity": [0.0, 1.5]},
        },
        "bad_sweep.yaml",
    )
    assert main(["validate-config", str(bad)]) == 1
    assert "eccentricity" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_validate_config_accepts_several_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # README: `helioswarm validate-config configs/*.yaml` checks every shipped file (run
    # configurations and sweeps); found missing by the M6b fresh-clone reproduction.
    shipped = sorted(str(p) for p in MVP_CONFIG.parent.glob("*.yaml"))
    assert len(shipped) >= 5
    assert main(["validate-config", *shipped]) == 0
    out = capsys.readouterr().out
    assert out.count("OK: ") == len(shipped)
    # One invalid file among valid ones: every file is still reported, exit 1.
    bad = _write(tmp_path, {"schema_version": 1}, "bad.yaml")
    assert main(["validate-config", str(MVP_CONFIG), str(bad), shipped[-1]]) == 1
    cap = capsys.readouterr()
    assert cap.out.count("OK: ") == 2
    assert "bad.yaml" in cap.err
    # Single-file options with several files are a usage error (exit 2, module convention).
    assert main(["validate-config", str(MVP_CONFIG), shipped[-1], "--show-provenance"]) == 2
    assert "exactly one CONFIG" in capsys.readouterr().err
