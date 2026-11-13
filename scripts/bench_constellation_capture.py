"""CPU vs GPU benchmark of the batched capture kernel on the real N = 1000 constellation workload.

Usage::

    uv run --locked [--extra gpu] python scripts/bench_constellation_capture.py \\
        CONFIG CANDIDATES_DIR OUT_JSON [--sizes 10000 100000 1000000]

The workload is the set of (epoch, satellite, site) points with geometric access in the stage-A
link files of a constellation run (``helioswarm run CONFIG --chunk-index ... --candidates-dir``):
every point whose single-link reason code has none of the geometry bits (elevation, Earth/Sun/Moon
occultation, Sun avoidance) needs one beam-capture evaluation. Its range and elevation come from
the link files; the beam settings from the configuration (one ring, one technology). For each
size (a deterministic, seeded subset of those points) it times (wall clock):

* ``numpy``: :func:`helioswarm.capture_batch.airy_capture_batch` on the CPU (expectation only:
  nominal, bias-only and expected capture, ``percentiles=False``);
* ``jax``: the same kernel through JAX on the default device (GPU when ``JAX_PLATFORMS=cuda``),
  first call and best of 3 warm calls;
* ``m5_reference``: the per-point M5 reference (:func:`helioswarm.capture.capture_statistics`,
  the default ``numpy`` backend of the constellation run) on 300 points.

It records the maximum |JAX - numpy| and |numpy - reference| of the expected capture and the
total number of points with access. Measured numbers only; the JSON says which projections
(total time = per-point time x total points) are derived.
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

from helioswarm.beam import beam_settings_from_config
from helioswarm.capture import capture_statistics, projected_receiver_radius_m
from helioswarm.capture_batch import airy_capture_batch
from helioswarm.config import load_config
from helioswarm.constellation import constellation_members
from helioswarm.constellation_io import LINKSET_PREFIX, load_link_set
from helioswarm.optical import AiryPattern
from helioswarm.system import Unavailability

GEOMETRY_BITS = int(
    Unavailability.BELOW_MIN_ELEVATION
    | Unavailability.EARTH_OCCULTATION
    | Unavailability.SUN_OCCULTATION
    | Unavailability.MOON_OCCULTATION
    | Unavailability.SUN_AVOIDANCE
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("candidates_dir")
    ap.add_argument("out")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 100000, 1000000])
    args = ap.parse_args()
    import jax

    cfg = load_config(args.config)
    members = constellation_members(cfg, np.random.default_rng(cfg.project.seed))
    if len({m.ring_index for m in members}) != 1:
        raise SystemExit("benchmark expects a single ring (one beam technology)")
    settings = beam_settings_from_config(members[0].config)
    rng_parts, el_parts = [], []
    for f in sorted(Path(args.candidates_dir).glob(f"{LINKSET_PREFIX}*.npz")):
        ls = load_link_set(f)
        vis = (ls.reason_code & np.uint32(GEOMETRY_BITS)) == 0
        rng_parts.append(ls.range_m[vis])
        el_parts.append(ls.elevation_rad[vis])
    r_all = np.concatenate(rng_parts)
    el_all = np.concatenate(el_parts)
    n_total = int(r_all.size)
    devices = [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    sel_rng = np.random.default_rng(20261113)
    d_ap = settings.aperture.diameter_m
    lam = settings.wavelength_m
    rows = []
    for n in args.sizes:
        idx = sel_rng.choice(n_total, size=min(n, n_total), replace=False)
        r = r_all[idx]
        el = el_all[idx]
        a = projected_receiver_radius_m(
            settings.receiver_radius_m, el, settings.receiver_orientation
        )
        k = math.pi * d_ap / (lam * r)
        nu = r * settings.pointing_bias_rad
        s = r * settings.pointing_sigma_rad
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
        m = min(idx.size, 300)
        ref = np.empty(m)
        t0 = time.perf_counter()
        for i in range(m):
            p = AiryPattern(
                aperture_diameter_m=d_ap,
                wavelength_m=lam,
                range_m=float(r[i]),
                max_fresnel_number=settings.max_fresnel_number,
            )
            ref[i] = capture_statistics(p, float(a[i]), float(nu[i]), float(s[i])).expected
        t_ref = (time.perf_counter() - t0) / m
        row = {
            "n_points": int(idx.size),
            "numpy_batch_s": t_np,
            "jax_first_call_s": t_first,
            "jax_warm_best_s": min(warm),
            "speedup_numpy_over_jax_warm": t_np / min(warm),
            "m5_reference_s_per_point": t_ref,
            "m5_reference_points": m,
            "max_abs_diff_expected_jax_vs_numpy": float(np.max(np.abs(rj.expected - rn.expected))),
            "max_abs_diff_expected_numpy_vs_m5": float(np.max(np.abs(rn.expected[:m] - ref))),
            "band_panels": rn.band_panels,
            "offset_panels": rn.offset_panels,
            "bessel_quarter_nodes": rn.bessel_quarter_nodes,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
    doc = {
        "workload": "beam-capture points with geometric access of the constellation link files",
        "config": str(args.config),
        "candidates_dir": str(args.candidates_dir),
        "n_points_with_access_total": n_total,
        "projection_note": "total time for all points = per-point time of a row x "
        "n_points_with_access_total (derived, not measured)",
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
