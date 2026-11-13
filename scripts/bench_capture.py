"""CPU vs GPU benchmark of the batched Airy capture kernel (helioswarm.capture_batch).

Usage::

    uv run --locked [--extra gpu] python scripts/bench_capture.py OUT_JSON [--sizes 1000 10000 ...]

The batch is a realistic sweep-size workload: ranges, elevations and receiver radii drawn from
the optical sweep's design space (R in 0.3-2.2 au, D = 10/30/100 m at 1 um, receiver radius
1/5/10 km projected with sqrt(sin el), pointing jitter 1 nrad, bias 0 or 2 nrad), with a fixed
seed. For each size it times (wall clock, ``time.perf_counter``):

* ``numpy``: the batched kernel on the CPU (vectorised numpy, scipy Bessel functions),
  expectation only (``percentiles=False``: nominal, bias-only and expected capture, the
  quantities a sweep or Monte Carlo needs);
* ``jax``: the same kernel through JAX on the default JAX device (the GPU when
  ``JAX_PLATFORMS=cuda``), first call (includes compilation) and the best of 3 warm calls;
* ``numpy_with_percentiles`` (sizes <= 1000): the full statistics including the percentiles,
  whose non-monotone points fall back to the per-point CPU reference;
* ``m5_reference``: the per-point M5 reference (capture_statistics) on the first 300 points,
  as the per-point cost of the code path ``helioswarm run`` uses by default.

It records the maximum difference JAX - numpy of the expected capture (must be <= 1e-12) and the
devices. Results are written as JSON; nothing is extrapolated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from helioswarm.capture import capture_statistics
from helioswarm.capture_batch import airy_capture_batch
from helioswarm.constants import AU_M
from helioswarm.optical import AiryPattern


def batch(n: int, seed: int = 20261112) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    lam = 1.0e-6
    r = rng.uniform(0.3, 2.2, n) * AU_M
    d = rng.choice([10.0, 30.0, 100.0], n)
    a = rng.choice([1000.0, 5000.0, 10000.0], n) * np.sqrt(np.sin(rng.uniform(0.18, 1.57, n)))
    nu = rng.choice([0.0, 2.0e-9], n) * r
    s = 1.0e-9 * r
    k = math.pi * d / (lam * r)
    return k, a, nu, s, d, r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    args = ap.parse_args()
    import jax

    devices = [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    rows = []
    for n in args.sizes:
        k, a, nu, s, d, r = batch(n)
        t0 = time.perf_counter()
        rn = airy_capture_batch(k, a, nu, s, backend="numpy", percentiles=False)
        t_np = time.perf_counter() - t0
        t0 = time.perf_counter()
        rj = airy_capture_batch(k, a, nu, s, backend="jax", percentiles=False)
        t_first = time.perf_counter() - t0
        warm = []
        for _ in range(3):
            t0 = time.perf_counter()
            rj = airy_capture_batch(k, a, nu, s, backend="jax", percentiles=False)
            warm.append(time.perf_counter() - t0)
        t_full: float | None = None
        n_fallback: int | None = None
        if n <= 1000:
            t0 = time.perf_counter()
            full = airy_capture_batch(k, a, nu, s, backend="numpy")
            t_full = time.perf_counter() - t0
            n_fallback = full.n_fallback
        diff = float(np.max(np.abs(rj.expected - rn.expected)))
        m = min(n, 300)
        t0 = time.perf_counter()
        for i in range(m):
            p = AiryPattern(aperture_diameter_m=float(d[i]), wavelength_m=1e-6,
                            range_m=float(r[i]), max_fresnel_number=0.5)  # fmt: skip
            capture_statistics(p, float(a[i]), float(nu[i]), float(s[i]))
        t_ref = (time.perf_counter() - t0) / m
        row = {
            "n_points": n,
            "numpy_batch_s": t_np,
            "jax_first_call_s": t_first,
            "jax_warm_best_s": min(warm),
            "speedup_numpy_over_jax_warm": t_np / min(warm),
            "m5_reference_s_per_point_measured_on": m,
            "m5_reference_s_per_point": t_ref,
            "numpy_with_percentiles_s": t_full,
            "max_abs_diff_expected_jax_vs_numpy": diff,
            "n_fallback_points_with_percentiles": n_fallback,
            "band_panels": rn.band_panels,
            "offset_panels": rn.offset_panels,
            "bessel_quarter_nodes": rn.bessel_quarter_nodes,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    doc = {
        "devices": devices,
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
