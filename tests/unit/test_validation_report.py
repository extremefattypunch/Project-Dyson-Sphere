"""validate-models machinery (helioswarm.validation): report block/markers, failure reporting,
and the kernel-free checks themselves (each must pass on the real models)."""

from __future__ import annotations

from pathlib import Path

import pytest

from helioswarm import validation as v
from helioswarm.cli import main
from helioswarm.config import load_config
from helioswarm.kernels import KernelSet

from config_helpers import MVP_CONFIG


def _ctx() -> v.ValidationContext:
    return v.ValidationContext(
        base_config=load_config(MVP_CONFIG), kernel_set=KernelSet.from_manifest()
    )


KERNEL_FREE = (
    v._c_l0_circular,
    v._c_l1_circular,
    v._c_occultation,
    v._c_light_time,
    v._c_solar,
    v._c_zero_incidence,
    v._c_eclipse_and_ledger,
    v._c_radiator,
    v._c_first_null,
    v._c_ee_ring_and_power,
    v._c_capture_grid,
    v._c_pointing_monotone,
    v._c_lambda_aperture,
    v._c_impossible_beam,
    v._c_fresnel_to_airy,
    v._c_batch_kernel,
    v._c_link_budget,
    v._c_srp,
    v._c_capture_refinement,
)
SPICE = (
    v._c_horizons,
    v._c_sidereal,
    v._c_time_scale,
    v._c_reference_link,
    v._c_e2e_baseline,
    v._c_e2e_unavailable_zero,
    v._c_e2e_clip,
    v._c_e2e_area_linear,
    v._c_e2e_lambda_aperture,
    v._c_e2e_pointing,
    v._c_e2e_microwave_impossible,
    v._c_e2e_energy_accounting,
    v._c_e2e_reproducible,
)
XCHECK = ((v._c_rebound, "rebound"), (v._c_hcipy, "hcipy"), (v._c_itur, "itur"))
SLOW = (v._c_l1_step_convergence, v._c_diurnal_sampling)


def _assert_passes(fn: object) -> None:
    out = fn(_ctx())  # type: ignore[operator]
    assert out.passed is True or bool(out.passed) is True, out


def test_every_registered_check_runs_under_pytest() -> None:
    # A new validate-models check must be added to one of the lists below, so that a change
    # that breaks it (or makes it non-discriminating) fails the test gate, not only the job.
    covered = {*KERNEL_FREE, *SPICE, *(f for f, _ in XCHECK), *SLOW}
    registered = {c.fn for c in v.CHECKS}
    assert registered == covered
    assert {c.fn for c in v.CHECKS if c.slow} == set(SLOW)


@pytest.mark.parametrize("fn", KERNEL_FREE, ids=lambda f: f.__name__)
def test_kernel_free_checks_pass(fn: object) -> None:
    _assert_passes(fn)


@pytest.mark.spice
@pytest.mark.parametrize("fn", SPICE, ids=lambda f: f.__name__)
def test_kernel_checks_pass(fn: object, kernel_set: KernelSet) -> None:
    _assert_passes(fn)


@pytest.mark.spice
@pytest.mark.xcheck
@pytest.mark.parametrize(("fn", "module"), XCHECK, ids=lambda x: getattr(x, "__name__", x))
def test_crosscheck_checks_pass(fn: object, module: str, kernel_set: KernelSet) -> None:
    pytest.importorskip(module)
    _assert_passes(fn)


@pytest.mark.slow
@pytest.mark.spice
@pytest.mark.parametrize("fn", SLOW, ids=lambda f: f.__name__)
def test_slow_checks_pass(fn: object, kernel_set: KernelSet) -> None:
    _assert_passes(fn)


def test_crashing_check_is_reported_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_: v.ValidationContext) -> v._Outcome:
        raise RuntimeError("deliberate")

    monkeypatch.setattr(v, "CHECKS", (v.Check("§0", "broken", broken),
                                      v.Check("§0", "slow", broken, slow=True)))  # fmt: skip
    ctx = v.ValidationContext(
        base_config=load_config(MVP_CONFIG), kernel_set=KernelSet.from_manifest(),
        include_slow=False,
    )  # fmt: skip
    res = v.run_validation(ctx)
    assert res[0].passed is False
    assert "RuntimeError: deliberate" in res[0].note
    assert res[1].passed is None  # skipped slow check is "not run", never "pass"
    block = v.report_block(res, {"run": "test"})
    assert "0 passed, 1 failed, 1 not run" in block
    assert "### Unresolved (generated)" in block
    assert "broken" in block.split("### Unresolved (generated)")[1]


def test_report_markers_replace_only_the_generated_block(tmp_path: Path) -> None:
    p = tmp_path / "r.md"
    p.write_text("# Validation report\n\ncurated text\n", encoding="utf-8")
    r = (v.CheckResult("§9", "x", "1", "1", "0", True),)
    v.write_report(p, v.report_block(r, {"run": "one"}))
    v.write_report(p, v.report_block(r, {"run": "two"}))
    text = p.read_text()
    assert text.startswith("# Validation report\n\ncurated text\n")
    assert text.count(v.BEGIN_MARK) == 1
    assert "run: two" in text
    assert "run: one" not in text


def test_cli_error_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\n", encoding="utf-8")
    assert main(["run", str(bad), "--out", str(tmp_path / "o")]) == 1
    assert "invalid configuration" in capsys.readouterr().err
    assert main(["plot", str(tmp_path / "nothing")]) == 1
    two_sites = tmp_path / "two.yaml"
    two_sites.write_text(
        MVP_CONFIG.read_text().replace(
            "      altitude_m: 0.0\n",
            '      altitude_m: 0.0\n    - name: "b"\n      latitude_deg: 5.0\n'
            "      longitude_deg: 0.0\n",
        ),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert main(["run", str(two_sites), "--out", str(tmp_path / "o2")]) == 1
    assert "exactly one receiver site" in capsys.readouterr().err
    sw = tmp_path / "sw.yaml"
    sw.write_text(
        f"sweep_schema_version: 1\nname: s\nbase_config: {MVP_CONFIG}\n"
        "parameters:\n  receiver.radius_m: [1000.0]\n",
        encoding="utf-8",
    )
    assert main(["sweep", str(sw), "--out", str(tmp_path / "s"), "--array-index", "0"]) == 1
    assert "together" in capsys.readouterr().err
