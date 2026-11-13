"""Command-line interface (``helioswarm``).

Implemented subcommands (later milestones add more; none exist as placeholders). Exit codes
follow one convention: 0 success, 1 invalid input or failure, 2 command-line usage error
(argparse).

``helioswarm validate-config CONFIG [CONFIG ...] [--show-provenance] [--write-resolved PATH]``
    Validate run configurations and/or sweep files, each one in turn (the provenance and
    resolved-dump options take a single CONFIG). A sweep file is recognised by its
    ``sweep_schema_version`` key; every grid point is validated as a full configuration and
    nothing runs. Exit code 0 if every file is valid, 1 if any is invalid or unreadable.

``helioswarm run CONFIG [--out DIR] [--overwrite] [--backend numpy|numpy_batch|jax]
[--kernel-dir DIR] [--chunk-index I --chunk-count N --candidates-dir D] [--candidates-dir D]
[--link-detail] [--link-detail-max-gb G]``
    End-to-end run into a run directory. A configuration without a ``constellation`` section is
    a single-satellite run (``helioswarm.io``); with one, a constellation run
    (``helioswarm.constellation_io``): either in one process, or as Slurm array tasks that each
    write the link file of a contiguous block of satellites (``--chunk-index/--chunk-count``
    with ``--candidates-dir``) followed by one merge-and-schedule run (``--candidates-dir``
    alone). ``--link-detail`` also writes the per-link detail (size estimate checked against
    ``--link-detail-max-gb``). Exit code 0 on success, 1 on an invalid configuration, an
    existing run directory without ``--overwrite``, a kernel problem or a model-validity error.

``helioswarm plot RUN_DIR``
    Redraw ``figures/`` and ``report.md`` (single satellite) or the constellation figures from
    the saved files only. Exit code 0 / 1.

``helioswarm sweep SWEEP_CONFIG --out DIR [--summary-only] [--array-index I --array-count N]
[--collate-only] [--overwrite] [--backend numpy|numpy_batch|jax]``
    Deterministic Cartesian sweep (``helioswarm.sweep``). Exit code 0 if every point of this
    task is ok, 1 if any failed or on an error.

``helioswarm validate-models --report PATH [--quick] [--base-config CONFIG]``
    Programmatic validation checks (``helioswarm.validation``); writes the generated block of
    the report. Exit code 0 if every check passed, 1 if any failed or did not run.

``helioswarm characterize CONFIG --out DIR [--overwrite] [--range-au R] [--elevation-deg E]
[--radii-au ...] [--electrical-to-beam-efficiency ...] [--receiver-radii-m ...]
[--pointing-sigmas-rad ...]``
    Response curves of one configuration (``helioswarm.characterize``; PDF §8.4, §9.5): solar,
    PV, thermal and radiator quantities versus heliocentric distance and transmitter operating
    point; beam capture versus pointing sigma, receiver radius and wavefront error at one stated
    link geometry. No kernels needed. Exit code 0 on success, 1 on an error.

``helioswarm fetch-kernels [--manifest PATH] [--kernel-dir DIR]``
    Download the SPICE kernels listed in ``data/kernels/kernels.json`` into the kernel
    directory (``$HELIOSWARM_KERNEL_DIR`` or ``data/kernels/``) if missing, and verify size,
    SHA-256 and NAIF MD5 of every file. Idempotent. Exit code 0 when every kernel is present and
    verified, 1 on a checksum mismatch, a missing manifest or a download error.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from helioswarm import __version__
from helioswarm.config import (
    ConfigError,
    RunConfig,
    config_hash,
    load_config,
    provenance_table,
    read_yaml,
    write_resolved_yaml,
)
from helioswarm.kernels import KernelError, KernelSet, fetch_kernels

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helioswarm",
        description="Heliocentric solar-power swarm feasibility simulator.",
    )
    parser.add_argument("--version", action="version", version=f"helioswarm {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    validate = sub.add_parser(
        "validate-config",
        help="validate a run configuration (or sweep) file",
        description="Validate a YAML run configuration against the helioswarm schema, or a "
        "sweep file (sweep_schema_version; every point validated, nothing runs). "
        "Exit status: 0 valid, 1 invalid, 2 usage error.",
    )
    validate.add_argument(
        "config",
        metavar="CONFIG",
        nargs="+",
        help="path(s) to YAML configurations or sweep files (each validated; exit 1 if any fails)",
    )
    validate.add_argument(
        "--show-provenance",
        action="store_true",
        help="list every resolved value with unit and provenance (measured/sourced/assumed/swept)",
    )
    validate.add_argument(
        "--write-resolved",
        metavar="PATH",
        help="write the fully resolved configuration (all defaults explicit) as YAML to PATH",
    )

    fetch = sub.add_parser(
        "fetch-kernels",
        help="download and verify the SPICE kernel set",
        description="Download missing SPICE kernels listed in the manifest and verify the "
        "size, SHA-256 and (where NAIF publishes one) MD5 of every kernel. Existing files that "
        "do not match are never overwritten. Exit status: 0 ok, 1 error, 2 usage error.",
    )
    fetch.add_argument(
        "--manifest",
        metavar="PATH",
        default=None,
        help="kernel manifest (default: data/kernels/kernels.json of the source checkout)",
    )
    fetch.add_argument(
        "--kernel-dir",
        metavar="DIR",
        default=None,
        help="kernel directory (default: $HELIOSWARM_KERNEL_DIR, else data/kernels/)",
    )

    backend_help = (
        "beam-capture backend: numpy (M5 per-epoch reference, default), numpy_batch (batched "
        "kernel on the CPU) or jax (batched kernel compiled by JAX; needs the jax or gpu "
        "extra); the batched kernel is airy_far_field only"
    )
    backends = ("numpy", "numpy_batch", "jax")
    run = sub.add_parser(
        "run",
        help="run one configuration end to end",
        description="End-to-end run: single satellite, or a constellation when the "
        "configuration has a constellation section. Exit status: 0 ok, 1 error, 2 usage error.",
    )
    run.add_argument("config", metavar="CONFIG", help="path to the YAML configuration")
    run.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="run directory (default: $HELIOSWARM_RUNS_DIR/<project.name>, else runs/<name>)",
    )
    run.add_argument("--overwrite", action="store_true", help="replace an existing run directory")
    run.add_argument(
        "--backend", choices=backends, default=None, help=backend_help + " (default numpy)"
    )
    run.add_argument("--kernel-dir", metavar="DIR", default=None, help="SPICE kernel directory")
    run.add_argument(
        "--chunk-index",
        type=int,
        default=None,
        help="constellation stage A only: this array task's block of satellites",
    )
    run.add_argument("--chunk-count", type=int, default=None, help="number of array tasks")
    run.add_argument(
        "--candidates-dir",
        metavar="DIR",
        default=None,
        help="constellation link files: written by --chunk-index tasks, merged otherwise",
    )
    run.add_argument(
        "--link-detail",
        action="store_true",
        help="constellation: also write the per-link detail (links/*.parquet)",
    )
    run.add_argument(
        "--link-detail-max-gb",
        type=float,
        default=None,
        help="cap on the uncompressed per-link detail estimate [GB = 1e9 bytes] (default 4)",
    )

    plot = sub.add_parser(
        "plot",
        help="regenerate figures and report from a saved run",
        description="Redraw figures/ and report.md from the saved files of RUN_DIR.",
    )
    plot.add_argument("run_dir", metavar="RUN_DIR", help="run directory")

    sweep = sub.add_parser(
        "sweep",
        help="run a deterministic parameter sweep",
        description="Cartesian sweep over dotted configuration keys; every point is a full "
        "validated run. Exit status: 0 all points ok, 1 any point failed or error.",
    )
    sweep.add_argument("sweep_config", metavar="SWEEP_CONFIG", help="sweep YAML file")
    sweep.add_argument("--out", metavar="DIR", required=True, help="sweep output directory")
    sweep.add_argument(
        "--summary-only",
        action="store_true",
        help="do not save time series or figures (summary.json per point only)",
    )
    sweep.add_argument("--overwrite", action="store_true", help="replace existing point dirs")
    sweep.add_argument("--array-index", type=int, default=None, help="this task's index")
    sweep.add_argument("--array-count", type=int, default=None, help="number of array tasks")
    sweep.add_argument(
        "--collate-only",
        action="store_true",
        help="only rebuild sweep_summary.parquet from the point directories",
    )
    sweep.add_argument("--backend", choices=backends, default="numpy", help=backend_help)

    val = sub.add_parser(
        "validate-models",
        help="run the programmatic validation checks and write the report",
        description="Run the PDF §7.4/§8.5/§9/§10.5/§13.1-13.3 checks and write the generated "
        "block of the validation report. Exit status: 0 all passed, 1 otherwise.",
    )
    val.add_argument("--report", metavar="PATH", required=True, help="report Markdown file")
    val.add_argument(
        "--base-config",
        metavar="CONFIG",
        default="configs/mvp_single_satellite.yaml",
        help="configuration the end-to-end checks start from",
    )
    val.add_argument("--quick", action="store_true", help="skip the slow convergence checks")
    val.add_argument("--kernel-dir", metavar="DIR", default=None, help="SPICE kernel directory")

    char = sub.add_parser(
        "characterize",
        help="response curves of one configuration (PDF §8.4, §9.5)",
        description="Evaluate the solar/thermal model versus heliocentric distance and "
        "transmitter operating point, and the beam capture versus pointing sigma, receiver "
        "radius and wavefront error at one stated link geometry; write Parquet tables, "
        "metadata and figures drawn from them. No kernels needed. Exit status: 0 ok, 1 error.",
    )
    char.add_argument("config", metavar="CONFIG", help="path to the YAML configuration")
    char.add_argument("--out", metavar="DIR", required=True, help="output directory")
    char.add_argument("--overwrite", action="store_true", help="replace an existing output")
    char.add_argument(
        "--range-au",
        type=float,
        default=1.0,
        help="link range of the capture curves [au] (scenario input; default 1)",
    )
    char.add_argument(
        "--elevation-deg",
        type=float,
        default=90.0,
        help="beam elevation at the receiver of the capture curves [deg] (default 90)",
    )
    char.add_argument(
        "--radii-au",
        type=float,
        nargs="+",
        default=None,
        help="heliocentric distances [au] (default 0.3-1.2 in 0.05 steps)",
    )
    char.add_argument(
        "--electrical-to-beam-efficiency",
        type=float,
        nargs="+",
        default=None,
        help="transmitter operating points eta_electrical_to_beam (default: configured)",
    )
    char.add_argument(
        "--receiver-radii-m",
        type=float,
        nargs="+",
        default=None,
        help="receiver radii [m] (default 1, 5, 10 km plus the configured radius)",
    )
    char.add_argument(
        "--pointing-sigmas-rad",
        type=float,
        nargs="+",
        default=None,
        help="per-axis pointing sigmas [rad] (default 0-1 x lambda/D plus the configured one)",
    )
    cmp_ = sub.add_parser(
        "compare",
        help="harmonised comparison table and figure from saved sweep runs (M7b)",
        description="Read a comparison file (cases = sweeps whose base configurations share "
        "their technology blocks), check that each technology group differs only in "
        "orbit/geometry keys, read every saved point under RUNS_DIR/<sweep name>/points/ and "
        "write comparison tables (CSV + Markdown), the harmonised-assumption table, a figure "
        "and a manifest. Nothing is simulated. Exit status: 0 ok (every point present and "
        "complete or recorded as failed), 1 error or missing/incomplete points.",
    )
    cmp_.add_argument("comparison", metavar="COMPARISON", help="comparison YAML file")
    cmp_.add_argument(
        "--runs-dir",
        metavar="DIR",
        required=True,
        help="directory holding one sweep output directory per case (named after the sweep)",
    )
    cmp_.add_argument("--out", metavar="DIR", required=True, help="output directory")
    return parser


def _cmd_compare(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from helioswarm.compare import HarmonisationError, load_comparison, write_comparison

    try:
        plan = load_comparison(args.comparison)
    except ConfigError as exc:
        err.write(f"ERROR: {exc.format()}\n")
        return EXIT_INVALID
    except HarmonisationError as exc:
        err.write(f"ERROR: {exc}\n")
        return EXIT_INVALID
    try:
        res = write_comparison(plan, Path(args.runs_dir), Path(args.out))
    except (OSError, ValueError, ConfigError) as exc:
        err.write(f"ERROR: {type(exc).__name__}: {exc}\n")
        return EXIT_INVALID
    counts: dict[str, int] = {}
    for r in res.rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    out.write(
        f"OK: comparison '{plan.definition.name}' written to {res.out_dir} "
        f"({len(res.files)} files); points: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + "\n"
    )
    for r in res.rows:
        if r.is_base:
            p = r.delivered_mean_power_W
            out.write(
                f"  {r.case} (base {r.point}): {r.status}"
                + (
                    ""
                    if p is None
                    else f", P_grid mean {p:.6g} W, availability {r.availability:.6g}"
                )
                + "\n"
            )
    incomplete = counts.get("missing", 0) + counts.get("incomplete", 0)
    if incomplete:
        err.write(f"WARNING: {incomplete} point(s) missing or incomplete (not run yet?)\n")
        return EXIT_INVALID
    return EXIT_OK


def _cmd_run(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from helioswarm.aperture import BeamModelValidityError
    from helioswarm.io import RunDirectoryError, execute_run, prepare_run_dir, resolve_run_dir

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        err.write(f"ERROR: {exc.format()}\n")
        return EXIT_INVALID
    if config.constellation is not None:
        return _run_constellation(args, config, out, err)
    if (
        args.chunk_index is not None
        or args.chunk_count is not None
        or args.candidates_dir is not None
        or args.link_detail
        or args.link_detail_max_gb is not None
    ):
        err.write(
            "ERROR: --chunk-index/--chunk-count/--candidates-dir/--link-detail/"
            "--link-detail-max-gb apply to constellation configurations only\n"
        )
        return EXIT_INVALID
    try:
        run_dir = prepare_run_dir(resolve_run_dir(config, args.out), overwrite=args.overwrite)
        res = execute_run(
            config,
            run_dir,
            kernel_set=KernelSet.from_manifest(None, args.kernel_dir),
            backend=args.backend or "numpy",
            command=["helioswarm", *sys.argv[1:]],
        )
    except (
        RunDirectoryError,
        KernelError,
        BeamModelValidityError,
        OSError,
        ValueError,
        NotImplementedError,
        ImportError,
    ) as exc:
        err.write(f"ERROR: {type(exc).__name__}: {exc}\n")
        return EXIT_INVALID
    s = res.summary
    cf = s.capacity_factor
    mean = s.p_grid_W.time_mean
    out.write(
        f"OK: run written to {res.run_dir}\n"
        f"  P_grid mean {'n/a' if mean is None else f'{mean:.6g}'} W, median "
        f"{s.p_grid_W.median:.6g} W, peak {s.p_grid_W.peak:.6g} W\n"
        f"  availability {s.availability:.6g}, capacity factor "
        f"{'n/a' if cf is None else f'{cf:.6g}'}, delivered energy "
        f"{s.delivered_energy_J:.6g} J\n"
    )
    if s.diurnal_aliasing.warning:
        out.write(
            "  WARNING: diurnal aliasing (hour-angle gap "
            f"{s.diurnal_aliasing.hour_angle_max_gap_rad:.4g} rad); see logs/run.log\n"
        )
    return EXIT_OK


def _run_constellation(
    args: argparse.Namespace, config: RunConfig, out: TextIO, err: TextIO
) -> int:
    from helioswarm.aperture import BeamModelValidityError
    from helioswarm.constellation_io import (
        BYTES_PER_GB,
        LinkDetailTooLargeError,
        execute_constellation_run,
        execute_link_chunk,
    )
    from helioswarm.io import RunDirectoryError, prepare_run_dir, resolve_run_dir
    from helioswarm.system import Backend

    chunked = args.chunk_index is not None or args.chunk_count is not None
    if chunked and (args.chunk_index is None or args.chunk_count is None):
        err.write("ERROR: --chunk-index and --chunk-count must be given together\n")
        return EXIT_INVALID
    if chunked and args.candidates_dir is None:
        err.write("ERROR: --chunk-index needs --candidates-dir (where the link file goes)\n")
        return EXIT_INVALID
    # Options that the selected mode would otherwise ignore are refused.
    if chunked:
        ignored = [
            name
            for name, given in (
                ("--out", args.out is not None),
                ("--overwrite", args.overwrite),
                ("--link-detail", args.link_detail),
                ("--link-detail-max-gb", args.link_detail_max_gb is not None),
            )
            if given
        ]
        if ignored:
            err.write(
                f"ERROR: {', '.join(ignored)} do(es) not apply to a stage-A array task "
                "(--chunk-index/--chunk-count writes only its link file to --candidates-dir)\n"
            )
            return EXIT_INVALID
    elif args.candidates_dir is not None and args.backend is not None:
        err.write(
            "ERROR: --backend does not apply to a merge (--candidates-dir without --chunk-index):"
            " the beam backend is the one the array tasks used, recorded in their link files\n"
        )
        return EXIT_INVALID
    if args.link_detail_max_gb is not None and not args.link_detail:
        err.write("ERROR: --link-detail-max-gb needs --link-detail\n")
        return EXIT_INVALID
    max_gb = 4.0 if args.link_detail_max_gb is None else float(args.link_detail_max_gb)
    if not (math.isfinite(max_gb) and max_gb > 0.0):
        err.write("ERROR: --link-detail-max-gb must be a finite number > 0\n")
        return EXIT_INVALID
    backend: Backend = args.backend or "numpy"
    command = ["helioswarm", *sys.argv[1:]]
    try:
        ks = KernelSet.from_manifest(None, args.kernel_dir)
        if chunked:
            path = execute_link_chunk(
                config,
                Path(args.candidates_dir),
                chunk_index=args.chunk_index,
                chunk_count=args.chunk_count,
                kernel_set=ks,
                backend=backend,
            )
            out.write(f"OK: link file written to {path}\n")
            return EXIT_OK
        run_dir = prepare_run_dir(resolve_run_dir(config, args.out), overwrite=args.overwrite)
        res = execute_constellation_run(
            config,
            run_dir,
            kernel_set=ks,
            backend=backend,
            candidates_dir=None if args.candidates_dir is None else Path(args.candidates_dir),
            link_detail=args.link_detail,
            link_detail_max_bytes=int(max_gb * BYTES_PER_GB),
            command=command,
        )
    except (
        RunDirectoryError,
        KernelError,
        BeamModelValidityError,
        LinkDetailTooLargeError,
        OSError,
        ValueError,
        NotImplementedError,
        ImportError,
        RuntimeError,
    ) as exc:
        err.write(f"ERROR: {type(exc).__name__}: {exc}\n")
        return EXIT_INVALID
    s = res.summary
    mean = s.network.p_grid_W.time_mean
    act = s.active_satellites.time_mean
    out.write(
        f"OK: constellation run written to {res.run_dir}\n"
        f"  {s.n_satellites} satellites, {s.n_sites} sites, scheduler {s.scheduler.policy}\n"
        f"  network P_grid mean {'n/a' if mean is None else f'{mean:.6g}'} W, peak "
        f"{s.network.p_grid_W.peak:.6g} W, availability {s.network.availability:.6g}\n"
        f"  active satellites (time mean) "
        f"{'n/a' if act is None else f'{act:.4g}'}"
        f", delivered energy {s.network.delivered_energy_J:.6g} J\n"
    )
    return EXIT_OK


def _cmd_plot(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from helioswarm.constellation_io import NETWORK_FILE
    from helioswarm.plotting import make_constellation_figures, make_figures
    from helioswarm.report import write_report

    run_dir = Path(args.run_dir)
    if (run_dir / NETWORK_FILE).is_file():
        try:
            paths = make_constellation_figures(run_dir)
        except (OSError, KeyError, ValueError, ConfigError) as exc:
            err.write(f"ERROR: cannot plot {run_dir}: {exc}\n")
            return EXIT_INVALID
        out.write(f"OK: {len(paths)} constellation figures regenerated in {run_dir}\n")
        return EXIT_OK
    try:
        paths = make_figures(run_dir)
        report = write_report(run_dir)
    except (OSError, KeyError, ValueError, ConfigError) as exc:
        err.write(f"ERROR: cannot plot {run_dir}: {exc}\n")
        return EXIT_INVALID
    out.write(f"OK: {len(paths)} figures and {report.name} regenerated in {run_dir}\n")
    return EXIT_OK


def _cmd_sweep(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from helioswarm.io import RunDirectoryError
    from helioswarm.sweep import (
        collate_sweep,
        load_sweep,
        points_for_task,
        prepare_sweep_dir,
        run_sweep_points,
    )

    try:
        plan = load_sweep(args.sweep_config)
    except ConfigError as exc:
        err.write(f"ERROR: {exc.format()}\n")
        return EXIT_INVALID
    if (args.array_index is None) != (args.array_count is None):
        err.write("ERROR: --array-index and --array-count must be given together\n")
        return EXIT_INVALID
    try:
        sweep_dir = prepare_sweep_dir(plan, Path(args.out))
        if args.collate_only:
            dest = collate_sweep(plan, sweep_dir)
            out.write(f"OK: collated {len(plan.points)} points into {dest}\n")
            return EXIT_OK
        count = 1 if args.array_count is None else args.array_count
        index = 0 if args.array_index is None else args.array_index
        indices = points_for_task(len(plan.points), index, count)

        def log(msg: str) -> None:
            out.write(msg + "\n")
            out.flush()

        out.write(
            f"sweep {plan.definition.name}: {len(plan.points)} points, this task "
            f"{index}/{count} runs {len(indices)}\n"
        )
        status = run_sweep_points(
            plan,
            sweep_dir,
            indices,
            summary_only=args.summary_only,
            overwrite=args.overwrite,
            backend=args.backend,
            log=log,
        )
        dest = collate_sweep(plan, sweep_dir)
    except (FileExistsError, RunDirectoryError, ValueError, OSError) as exc:
        err.write(f"ERROR: {exc}\n")
        return EXIT_INVALID
    n_fail = sum(1 for v in status.values() if v != "ok")
    out.write(
        f"{'OK' if n_fail == 0 else 'DONE WITH FAILURES'}: {len(status) - n_fail} ok, "
        f"{n_fail} failed; collated into {dest}\n"
    )
    return EXIT_OK if n_fail == 0 else EXIT_INVALID


def _cmd_validate_models(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    import os
    import platform

    from helioswarm.manifest import git_info, utc_now_iso
    from helioswarm.validation import (
        ValidationContext,
        load_base_config,
        report_block,
        run_validation,
        write_report,
    )

    try:
        base = load_base_config(args.base_config)
        ks = KernelSet.from_manifest(None, args.kernel_dir)
        ks.verify()
    except (ConfigError, KernelError) as exc:
        err.write(f"ERROR: {exc}\n")
        return EXIT_INVALID

    def log(msg: str) -> None:
        out.write(msg + "\n")
        out.flush()

    start = utc_now_iso()
    results = run_validation(
        ValidationContext(base_config=base, kernel_set=ks, include_slow=not args.quick), log
    )
    gi = git_info()
    header = {
        "run": f"{start} UTC",
        "helioswarm": f"{__version__}, git {gi.commit} (dirty {gi.dirty})",
        "host": platform.node(),
        "Slurm job": os.environ.get("SLURM_JOB_ID", "none"),
        "base config": str(args.base_config),
    }
    path = write_report(Path(args.report), report_block(results, header))
    n_bad = sum(1 for r in results if r.passed is not True)
    out.write(f"report written to {path}; {len(results) - n_bad}/{len(results)} passed\n")
    return EXIT_OK if n_bad == 0 else EXIT_INVALID


def _cmd_fetch_kernels(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    try:
        kernel_set = KernelSet.from_manifest(args.manifest, args.kernel_dir)
        out.write(f"kernel set {kernel_set.manifest.kernel_set_id} -> {kernel_set.directory}\n")

        def log(msg: str) -> None:
            out.write(msg + "\n")

        fetch_kernels(kernel_set, log=log)
    except (KernelError, OSError) as exc:
        err.write(f"ERROR: {exc}\n")
        return EXIT_INVALID
    out.write(f"OK: {len(kernel_set.records)} kernels present and verified\n")
    return EXIT_OK


def _format_value(value: object) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(repr(v) for v in value) + "]"
    return repr(value)


def _print_provenance(config: RunConfig, out: TextIO) -> None:
    rows = provenance_table(config)
    width_path = max(len(r.path) for r in rows)
    width_value = min(40, max(len(_format_value(r.value)) for r in rows))
    width_unit = max(len(r.unit) for r in rows)
    header = (
        f"{'field':<{width_path}}  {'value':<{width_value}}  {'unit':<{width_unit}}  "
        f"{'provenance':<10}  {'kind':<12}  description"
    )
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")
    for r in rows:
        prov = r.provenance + ("*" if r.provenance != r.schema_provenance else "")
        desc = r.description + (f" [source: {r.source}]" if r.source else "")
        out.write(
            f"{r.path:<{width_path}}  {_format_value(r.value):<{width_value}}  "
            f"{r.unit:<{width_unit}}  {prov:<10}  {r.kind:<12}  {desc}\n"
        )
    if any(r.provenance != r.schema_provenance for r in rows):
        out.write("(* provenance relabelled by provenance_overrides)\n")


def _is_sweep_file(path: str) -> bool:
    try:
        raw = read_yaml(path)
    except (ConfigError, OSError):
        return False  # load_config reports the problem
    return isinstance(raw, dict) and "sweep_schema_version" in raw


def _cmd_validate_sweep(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from helioswarm.sweep import load_sweep

    if args.show_provenance or args.write_resolved:
        err.write(
            "ERROR: --show-provenance/--write-resolved apply to run configurations, not sweep "
            "files; validate the base configuration instead\n"
        )
        return EXIT_INVALID
    try:
        plan = load_sweep(args.config)
    except ConfigError as exc:
        err.write(f"ERROR: {exc.format()}\n")
        return EXIT_INVALID
    keys = ", ".join(plan.definition.parameters)
    out.write(
        f"OK: {args.config} is a valid helioswarm sweep file (sweep_schema_version "
        f"{plan.definition.sweep_schema_version}, {len(plan.points)} points over {keys}; base "
        f"{plan.base_path}; every point validated as a full configuration)\n"
    )
    return EXIT_OK


def _cmd_validate_config(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    paths: list[str] = list(args.config)
    if len(paths) > 1 and (args.show_provenance or args.write_resolved):
        err.write(
            "ERROR: --show-provenance/--write-resolved take exactly one CONFIG "
            f"({len(paths)} given)\n"
        )
        return EXIT_USAGE  # a command-line usage error (module convention), not invalid input
    worst = EXIT_OK
    for path in paths:
        one = argparse.Namespace(**{**vars(args), "config": path})
        worst = max(worst, _validate_one_config(one, out, err))
    return worst


def _validate_one_config(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if _is_sweep_file(args.config):
        return _cmd_validate_sweep(args, out, err)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        err.write(f"ERROR: {exc.format()}\n")
        return EXIT_INVALID
    out.write(
        f"OK: {args.config} is a valid helioswarm configuration "
        f"(schema_version {config.schema_version}, config_hash {config_hash(config)})\n"
    )
    if args.show_provenance:
        _print_provenance(config, out)
    if args.write_resolved:
        try:
            path = write_resolved_yaml(config, args.write_resolved)
        except OSError as exc:
            err.write(f"ERROR: cannot write resolved configuration: {exc}\n")
            return EXIT_INVALID
        out.write(f"resolved configuration written to {path}\n")
    return EXIT_OK


def _cmd_characterize(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    import math

    from helioswarm.aperture import BeamModelValidityError
    from helioswarm.characterize import METADATA_FILE, LinkGeometry, write_characterization
    from helioswarm.constants import AU_M
    from helioswarm.io import RunDirectoryError, prepare_run_dir

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        err.write(f"ERROR: {exc.format()}\n")
        return EXIT_INVALID
    try:
        geometry = LinkGeometry(
            range_m=args.range_au * AU_M, elevation_rad=math.radians(args.elevation_deg)
        )
        out_dir = prepare_run_dir(
            Path(args.out), overwrite=args.overwrite, markers=(METADATA_FILE,)
        )
        res = write_characterization(
            config,
            out_dir,
            geometry=geometry,
            sun_distance_m=None if args.radii_au is None else [r * AU_M for r in args.radii_au],
            electrical_to_beam_efficiencies=args.electrical_to_beam_efficiency,
            receiver_radius_m=args.receiver_radii_m,
            pointing_sigma_rad=args.pointing_sigmas_rad,
        )
    except (RunDirectoryError, BeamModelValidityError, OSError, ValueError) as exc:
        err.write(f"ERROR: {type(exc).__name__}: {exc}\n")
        return EXIT_INVALID
    out.write(
        f"OK: characterization written to {res.out_dir} ({len(res.files)} files, "
        f"{len(res.figures)} figures)\n"
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``helioswarm`` console script; returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    command: str = args.command
    if command == "validate-config":
        return _cmd_validate_config(args, sys.stdout, sys.stderr)
    if command == "fetch-kernels":
        return _cmd_fetch_kernels(args, sys.stdout, sys.stderr)
    if command == "run":
        return _cmd_run(args, sys.stdout, sys.stderr)
    if command == "plot":
        return _cmd_plot(args, sys.stdout, sys.stderr)
    if command == "sweep":
        return _cmd_sweep(args, sys.stdout, sys.stderr)
    if command == "validate-models":
        return _cmd_validate_models(args, sys.stdout, sys.stderr)
    if command == "characterize":
        return _cmd_characterize(args, sys.stdout, sys.stderr)
    if command == "compare":
        return _cmd_compare(args, sys.stdout, sys.stderr)
    parser.error(f"unknown command {command!r}")  # argparse already rejects unknown commands


if __name__ == "__main__":
    raise SystemExit(main())
