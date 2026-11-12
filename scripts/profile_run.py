"""Profile one end-to-end run (PDF §14.3 item 1: profile the vertical slice before optimising).

Usage::

    uv run --locked python scripts/profile_run.py CONFIG OUT_DIR [--backend numpy|jax]

Runs :func:`helioswarm.system.run_single_satellite` under ``cProfile`` (no output files other
than the profile), and writes into OUT_DIR:

* ``profile.pstats``: the raw cProfile statistics;
* ``profile_top.txt``: the 40 functions with the largest cumulative time;
* ``profile_summary.json``: stage wall times (``SystemRunResult.timings``), total wall time,
  peak resident memory (``resource.getrusage``, kB on Linux) and the epoch counts.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import resource
import time
from pathlib import Path

import numpy as np

from helioswarm.config import load_config
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.kernels import KernelSet
from helioswarm.system import run_single_satellite


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    ks = KernelSet.from_manifest()
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    with ks.loaded(verify=True) as k:
        prof.enable()
        res = run_single_satellite(
            cfg,
            ephemeris=SpiceEphemeris(k),
            kernels=k,
            rng=np.random.default_rng(cfg.project.seed),
            backend=args.backend,
        )
        prof.disable()
    wall = time.perf_counter() - t0
    prof.dump_stats(out / "profile.pstats")
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
    (out / "profile_top.txt").write_text(buf.getvalue(), encoding="utf-8")
    summary = {
        "config": str(args.config),
        "backend": args.backend,
        "n_epochs": len(res),
        "n_beam_epochs": int(res.beam_epoch_index.size),
        "wall_s_total_incl_kernel_load": wall,
        "stage_wall_s": {t.stage: t.wall_s for t in res.timings},
        "peak_rss_kB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out / "profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
