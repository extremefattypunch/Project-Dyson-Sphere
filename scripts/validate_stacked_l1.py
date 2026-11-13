"""Validation of the stacked L1 integration (``propagate_l1_stacked``) on heterogeneous orbits.

Usage::

    uv run --locked python scripts/validate_stacked_l1.py OUT_JSON [--steps 8100]
        [--config configs/constellation_demo.yaml]

Why: scipy's DOP853 error norm ``|h| E5 / sqrt((E5 + 0.01 E3) * n)`` (``E5``, ``E3`` summed
over all components) is not separable by satellite, so dividing the tolerances by ``sqrt(N)``
does not guarantee that each accepted stacked step meets every satellite's solo criterion
(review of M7). This script measures what does hold, on purpose with dissimilar members whose
step-size needs differ: a satellite 5 deg ahead of the Earth (strong Earth perturbation), a
fast inner orbit at 0.4 au, eccentric and inclined orbits at 0.7 and 1.3 au, and two 1 au
satellites. All share the configuration's force model (planets + SRP) and integrator settings.

Two regimes: the configured settings (in the demo the step is capped by ``max_step`` = 300 s,
so the error norm rarely sets the step) and the same tolerances without ``max_step`` (the
DOP853 error norm alone sets the step: the regime in which the non-separability could
matter). For each regime and satellite it reports, over the whole span, the maximum position
deviations

* ``stacked_vs_solo``: all six stacked in one solve (tolerances / sqrt(6)) vs its solo run;
* ``solo_vs_reference`` and ``stacked_vs_reference``: against a solo run with tighter settings
  (rtol at the validated floor 1e-13, atol / 10, max_step / 2) - the solo integration error
  estimate.

Pass criterion (stated before running): ``stacked_vs_reference <= 2 * solo_vs_reference + 1 cm``
for every satellite, i.e. the stacked solve is not materially less accurate than the solo one.
Measured numbers only; nothing is extrapolated.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from helioswarm.config import INTEGRATOR_RTOL_MIN, parse_config, read_yaml
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.kernels import KernelSet
from helioswarm.propagate import (
    integrator_settings_from_config,
    propagate_l1,
    propagate_l1_stacked,
    satellite_initial_state,
)
from helioswarm.timescale import time_grid_from_config, utc_to_tdb_s

# name, a [au], e, i [deg], raan [deg], argp [deg], Earth-relative phase [deg]
MEMBERS: tuple[tuple[str, float, float, float, float, float, float], ...] = (
    ("near_earth_1au_+5deg", 1.0, 0.0, 0.0, 0.0, 0.0, 5.0),
    ("opposite_1au_+180deg", 1.0, 0.0, 0.0, 0.0, 0.0, 180.0),
    ("inner_0.4au_e0.05_i3", 0.4, 0.05, 3.0, 20.0, 60.0, 30.0),
    ("venus_like_0.7au_e0.1_i5", 0.7, 0.1, 5.0, 10.0, 30.0, 90.0),
    ("outer_1.3au_e0.2_i10", 1.3, 0.2, 10.0, 40.0, 100.0, 250.0),
    ("demo_like_1au_e0.017_+30deg", 1.0, 0.017, 0.5, 0.0, 0.0, 30.0),
)


def _member_config(base: dict[str, Any], m: tuple[str, float, float, float, float, float, float]):
    d = copy.deepcopy(base)
    d.pop("constellation", None)
    _, a, e, i, raan, argp, phase = m
    d["orbit"].update(
        semimajor_axis_au=a,
        eccentricity=e,
        inclination_deg=i,
        raan_deg=raan,
        arg_periapsis_deg=argp,
        true_anomaly_deg=phase,
        phase_reference="earth_relative",
    )
    return parse_config(d)


def _dev(a: np.ndarray, b: np.ndarray) -> float:
    """Largest position difference over the span [m]."""
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def _settings(st: Any) -> dict[str, float]:
    return {
        "rtol": st.rtol,
        "atol_position_m": st.atol_position_m,
        "atol_velocity_m_s": st.atol_velocity_m_s,
        "max_step_s": st.max_step_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_json")
    ap.add_argument(
        "--steps", type=int, default=8100, help="output steps (default 8100 = one year at 3896 s)"
    )
    ap.add_argument("--config", default="configs/constellation_demo.yaml")
    args = ap.parse_args()
    base = read_yaml(Path(args.config))
    base["time"]["duration_days"] = args.steps * float(base["time"]["output_step_s"]) / 86400.0
    cfgs = [_member_config(base, m) for m in MEMBERS]
    ks = KernelSet.from_manifest()
    with ks.loaded(verify=True) as k:
        eph = SpiceEphemeris(k)
        t = time_grid_from_config(cfgs[0].time, k).epochs_tdb_s
        t0 = utc_to_tdb_s(cfgs[0].orbit.initial_epoch_utc, k)
        inits = [
            satellite_initial_state(
                c.orbit,
                c.constants,
                epoch_tdb_s=t0,
                ephemeris=eph,
                attitude=c.collector.incidence_mode,
                fixed_normal_unit_j2000=c.collector.fixed_normal_unit_j2000,
            )
            for c in cfgs
        ]
        st_cfg = integrator_settings_from_config(cfgs[0].time)
        regimes = {
            # production settings: the step is capped by max_step (300 s in the demo)
            "configured_max_step": st_cfg,
            # error-controlled: no max_step, the DOP853 error norm alone sets the step
            "error_controlled_no_max_step": replace(st_cfg, max_step_s=math.inf),
        }
        results: dict[str, Any] = {}
        for regime, st in regimes.items():
            ref_st = replace(
                st,
                rtol=INTEGRATOR_RTOL_MIN,
                atol_position_m=st.atol_position_m / 10.0,
                atol_velocity_m_s=st.atol_velocity_m_s / 10.0,
                max_step_s=st.max_step_s / 2.0,
            )
            rows: list[dict[str, Any]] = []
            timing: dict[str, float] = {}
            c0 = time.perf_counter()
            stacked = propagate_l1_stacked(inits, t, ephemeris=eph, settings=st)
            timing["stacked_s"] = time.perf_counter() - c0
            timing["solo_s"] = 0.0
            timing["reference_s"] = 0.0
            for q, (m, init) in enumerate(zip(MEMBERS, inits, strict=True)):
                c0 = time.perf_counter()
                solo = propagate_l1(init, t, ephemeris=eph, settings=st)
                c1 = time.perf_counter()
                ref = propagate_l1(init, t, ephemeris=eph, settings=ref_st)
                c2 = time.perf_counter()
                timing["solo_s"] += c1 - c0
                timing["reference_s"] += c2 - c1
                r_st = stacked.propagations[q].trajectory.position_m
                r_so = solo.trajectory.position_m
                r_re = ref.trajectory.position_m
                row: dict[str, Any] = {
                    "name": m[0],
                    "elements": dict(
                        zip(
                            ("a_au", "e", "i_deg", "raan_deg", "argp_deg", "earth_rel_phase_deg"),
                            m[1:],
                            strict=True,
                        )
                    ),
                    "stacked_vs_solo_max_m": _dev(r_st, r_so),
                    "solo_vs_reference_max_m": _dev(r_so, r_re),
                    "stacked_vs_reference_max_m": _dev(r_st, r_re),
                    "solo_nfev": int(solo.diagnostics.nfev),
                    "reference_nfev": int(ref.diagnostics.nfev),
                    "stacked_nfev": int(stacked.propagations[q].diagnostics.nfev),
                }
                row["pass"] = bool(
                    row["stacked_vs_reference_max_m"] <= 2.0 * row["solo_vs_reference_max_m"] + 0.01
                )
                rows.append(row)
                print(regime, json.dumps(row), flush=True)
            results[regime] = {
                "settings": {
                    "solo": _settings(st),
                    "stacked_tolerance_scale": list(stacked.tolerance_scale),
                    "stack_sizes": list(stacked.stack_sizes),
                    "reference": _settings(ref_st),
                },
                "all_pass": all(r["pass"] for r in rows),
                "satellites": rows,
                "wall_s": timing,
            }
    out = {
        "script": "scripts/validate_stacked_l1.py",
        "config": args.config,
        "steps": args.steps,
        "days": base["time"]["duration_days"],
        "n_epochs": int(t.size),
        "criterion": "stacked_vs_reference <= 2 * solo_vs_reference + 0.01 m, every satellite",
        "all_pass": all(r["all_pass"] for r in results.values()),
        "regimes": results,
        "host": platform.node(),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": out["all_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
