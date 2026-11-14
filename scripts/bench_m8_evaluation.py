"""M8 GPU benchmark: one screening evaluation of the optimisation study on the CPU (numpy batched
kernel) and on the GPU (JAX batched kernel), per design, with the stage timings.

    python scripts/bench_m8_evaluation.py OUT.json [--study configs/optimization.yaml] [--n 8]

The designs are the study's nominal design plus ``n - 1`` designs drawn uniformly from the genome
(seeded, numpy Generator 20261114). Both backends run the same configuration; the JSON records the
wall time of every stage (helioswarm.system timings), the beam-stage fraction, the JAX devices and
the largest relative difference of the delivered energy between backends (the JAX kernel must
reproduce the numpy kernel, tests/unit/test_capture_batch.py). Run by slurm/m8_gpu_bench.sbatch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.evaluation import ForwardModel
from helioswarm.kernels import KernelSet
from helioswarm.manifest import gpu_devices
from helioswarm.optimization import decode, design_task, load_optimization, nominal_values
from helioswarm.system import run_single_satellite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--study", default="configs/optimization.yaml")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--backends", nargs="+", default=["numpy_batch", "jax"])
    args = ap.parse_args()
    problem = load_optimization(args.study)
    ev = problem.evaluator
    assert isinstance(ev, ForwardModel)
    nominal = nominal_values(problem)
    genomes = [[v.encode(nominal[v.name]) for v in problem.variables]]
    rng = np.random.default_rng(20261114)
    genomes += rng.random((args.n - 1, len(problem.variables))).tolist()
    rows = []
    with KernelSet.from_manifest().loaded(verify=True) as k:
        eph = SpiceEphemeris(k)
        for i, g in enumerate(genomes):
            cfg = ev.config_for(design_task(decode(problem.variables, g), i, f"b{i}"))
            row: dict = {"design": i}
            for backend in args.backends:
                t0 = time.perf_counter()
                try:
                    res = run_single_satellite(
                        cfg,
                        ephemeris=eph,
                        kernels=k,
                        rng=np.random.default_rng(1),
                        backend=backend,
                        capture_percentiles=False,
                    )
                except Exception as exc:  # recorded (e.g. a beam model outside its validity)
                    row[backend] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                    continue
                wall = time.perf_counter() - t0
                st = {t.stage: t.wall_s for t in res.timings}
                row[backend] = {
                    "wall_s": wall,
                    "stages_s": st,
                    "beam_fraction": st.get("beam", 0.0) / wall,
                    "beam_points": int(res.beam_epoch_index.size),
                    "energy_J": float(np.sum(res.link.grid_W) * cfg.time.output_step_s),
                }
            ok = [b for b in args.backends if "energy_J" in row.get(b, {})]
            if len(ok) == 2:
                e0, e1 = (row[b]["energy_J"] for b in ok)
                row["energy_rel_diff"] = abs(e1 - e0) / max(abs(e0), 1e-300)
            rows.append(row)
            print(json.dumps(row), flush=True)
    doc = {
        "study": args.study,
        "devices": gpu_devices("jax") if "jax" in args.backends else [],
        "designs": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
