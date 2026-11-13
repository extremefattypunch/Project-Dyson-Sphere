"""M4 mutation check: apply single-line source mutations and confirm the M4 tests catch each one.

Usage (from the repository root, after ``source scripts/cluster_env.sh``)::

    uv run python scripts/mutation_check_m4.py

Each mutation replaces exactly one occurrence of a source snippet, runs the four M4 test files
(``-x``), records CAUGHT (pytest exit 1: tests ran and failed), SURVIVED (exit 0) or ERROR (any
other exit code, e.g. a collection error, which is not counted as caught), and restores the
original file in a ``finally`` block, so the working tree is unchanged afterwards. Exit status 1
if any mutation was not caught or a snippet no longer matches the source exactly once.

This is a development aid for docs/milestones/M4.md and docs/validation_report.md; it is not
part of the test suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent

TESTS = (
    "tests/unit/test_solar.py",
    "tests/unit/test_thermal.py",
    "tests/unit/test_power_chain.py",
    "tests/integration/test_power_thermal_mvp.py",
)


class Mutation(NamedTuple):
    path: str
    old: str
    new: str
    name: str


MUTATIONS = (
    Mutation(
        "src/helioswarm/solar.py",
        "solar_constant_1au_W_m2 * (AU_M / r) ** 2",
        "solar_constant_1au_W_m2 * (AU_M / r)",
        "flux 1/r",
    ),
    Mutation(
        "src/helioswarm/solar.py",
        "np.clip(s @ n, 0.0, 1.0)",
        "np.abs(s @ n)",
        "no back-side clip",
    ),
    Mutation(
        "src/helioswarm/solar.py",
        "pv_heat_W=absorbed - electric",
        "pv_heat_W=incident - electric",
        "reflected as PV heat",
    ),
    Mutation(
        "src/helioswarm/solar.py",
        "np.clip(law.unclipped(temperature_K), law.minimum, law.maximum)",
        "law.unclipped(temperature_K)",
        "no efficiency clip",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "(self.front_emissivity + self.back_emissivity) * SIGMA * area_m2 * self.view_factor",
        "self.front_emissivity * SIGMA * area_m2 * self.view_factor",
        "PV front face only",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "        return self.radiating_sides * self.view_factor",
        "        return self.view_factor",
        "radiator sides ignored",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "        + chain.optics_or_rf_loss_W\n        + other\n",
        "        + chain.optics_or_rf_loss_W\n"
        "        + chain.optics_or_rf_loss_W\n"
        "        + other\n",
        "optics loss double counted",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "    rad_ok = t_rad <= t_lim",
        "    rad_ok = np.ones(shape, dtype=bool)",
        "radiator always feasible",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "- g * _fourth_power_difference_scalar(t, ts)",
        "- g * t**4",
        "PV T_space dropped",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "    pv_ok = np.asarray(np.isfinite(t_pv) & (t_pv <= t_pv_max))",
        "    pv_ok = np.isfinite(t_pv)",
        "PV limit ignored",
    ),
    Mutation(
        "src/helioswarm/power_chain.py",
        "    p_dc = p_pv * efficiencies.power_conditioning",
        "    p_dc = p_pv",
        "power conditioning dropped",
    ),
    # Review fixes (M4 review outcomes).
    Mutation(
        "src/helioswarm/thermal.py",
        "        return PV_BALANCE_RTOL * p + floor",
        "        return PV_BALANCE_RTOL * p",
        "PV representation floor dropped",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "            root = _polish_root(balance, root)",
        "            pass",
        "PV float-neighbour polish dropped",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "    return (t - ts) * (t + ts) * (t * t + ts * ts)",
        "    return t**4 - ts**4",
        "PV T^4 - Ts^4 unfactored",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "        floor = floors.get(name, 0.0) + tiny",
        "        floor = tiny",
        "ledger temperature floors dropped",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "        margin_K=np.minimum(pv_margin, rad_margin),",
        "        margin_K=rad_margin,",
        "margin_K = radiator margin only",
    ),
    Mutation(
        "src/helioswarm/thermal.py",
        "            if not info.converged:\n                continue",
        "            if False:\n                continue",
        "brentq non-convergence ignored",
    ),
    Mutation(
        "src/helioswarm/visibility.py",
        '    sun_vec = _retarded_positions(ephemeris, "SUN", r, t) - r',
        '    sun_vec = ephemeris.positions("SUN", t) - r',
        "instantaneous Sun for illumination",
    ),
)


def main() -> int:
    """Run every mutation; print one line each; return 1 if any survived or failed to apply."""
    bad = 0
    for mut in MUTATIONS:
        path = REPO / mut.path
        src = path.read_text()
        count = src.count(mut.old)
        if count != 1:
            print(f"{mut.name:38s} -> NOT APPLIED (snippet found {count} times)")
            bad += 1
            continue
        path.write_text(src.replace(mut.old, mut.new))
        try:
            run = subprocess.run(
                ["uv", "run", "pytest", "-q", "-x", *TESTS],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.write_text(src)
        lines = [line for line in run.stdout.splitlines() if line.strip()]
        # pytest exit 1 = tests ran and some failed; any other non-zero code (usage or
        # collection error, nothing collected) is not evidence that the mutation was caught.
        if run.returncode == 1:
            verdict = "CAUGHT"
        elif run.returncode == 0:
            verdict = "SURVIVED"
        else:
            verdict = f"ERROR (pytest exit {run.returncode})"
        bad += run.returncode != 1
        print(f"{mut.name:38s} -> {verdict}: {lines[-1] if lines else run.stderr.strip()[-200:]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
