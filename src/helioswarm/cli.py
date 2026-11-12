"""Command-line interface (``helioswarm``).

Implemented subcommands (later milestones add more; none exist as placeholders). Exit codes
follow one convention: 0 success, 1 invalid input or failure, 2 command-line usage error
(argparse).

``helioswarm validate-config CONFIG [--show-provenance] [--write-resolved PATH]``
    Validate a run configuration, or a sweep file (recognised by its ``sweep_schema_version``
    key; every grid point is validated as a full configuration, nothing runs). Exit code 0 if
    valid, 1 if invalid or unreadable.

``helioswarm run CONFIG [--out DIR] [--overwrite] [--backend numpy|jax] [--kernel-dir DIR]``
    End-to-end single-satellite run into a run directory (``helioswarm.io``). Exit code 0 on
    success, 1 on an invalid configuration, an existing run directory without ``--overwrite``,
    a kernel problem or a model-validity error.

``helioswarm plot RUN_DIR``
    Redraw ``figures/`` and ``report.md`` from the saved files only. Exit code 0 / 1.

``helioswarm sweep SWEEP_CONFIG --out DIR [--summary-only] [--array-index I --array-count N]
[--collate-only] [--overwrite] [--backend numpy|jax]``
    Deterministic Cartesian sweep (``helioswarm.sweep``). Exit code 0 if every point of this
    task is ok, 1 if any failed or on an error.

``helioswarm validate-models --report PATH [--quick] [--base-config CONFIG]``
    Programmatic validation checks (``helioswarm.validation``); writes the generated block of
    the report. Exit code 0 if every check passed, 1 if any failed or did not run.

``helioswarm fetch-kernels [--manifest PATH] [--kernel-dir DIR]``
    Download the SPICE kernels listed in ``data/kernels/kernels.json`` into the kernel
    directory (``$HELIOSWARM_KERNEL_DIR`` or ``data/kernels/``) if missing, and verify size,
    SHA-256 and NAIF MD5 of every file. Idempotent. Exit code 0 when every kernel is present and
    verified, 1 on a checksum mismatch, a missing manifest or a download error.
"""

from __future__ import annotations

import argparse
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
    validate.add_argument("config", metavar="CONFIG", help="path to the YAML configuration")
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
        "beam-capture backend: numpy (M5 per-epoch reference, default) or jax (batched kernel, "
        "airy_far_field only; needs the jax or gpu extra)"
    )
    run = sub.add_parser(
        "run",
        help="run one configuration end to end",
        description="End-to-end single-satellite run. Exit status: 0 ok, 1 error, 2 usage error.",
    )
    run.add_argument("config", metavar="CONFIG", help="path to the YAML configuration")
    run.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help="run directory (default: $HELIOSWARM_RUNS_DIR/<project.name>, else runs/<name>)",
    )
    run.add_argument("--overwrite", action="store_true", help="replace an existing run directory")
    run.add_argument("--backend", choices=("numpy", "jax"), default="numpy", help=backend_help)
    run.add_argument("--kernel-dir", metavar="DIR", default=None, help="SPICE kernel directory")

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
    sweep.add_argument("--backend", choices=("numpy", "jax"), default="numpy", help=backend_help)

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
    return parser


def _cmd_run(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from helioswarm.aperture import BeamModelValidityError
    from helioswarm.io import RunDirectoryError, execute_run, prepare_run_dir, resolve_run_dir

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        err.write(f"ERROR: {exc.format()}\n")
        return EXIT_INVALID
    try:
        run_dir = prepare_run_dir(resolve_run_dir(config, args.out), overwrite=args.overwrite)
        res = execute_run(
            config,
            run_dir,
            kernel_set=KernelSet.from_manifest(None, args.kernel_dir),
            backend=args.backend,
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
    cf = s["capacity_factor"]
    out.write(
        f"OK: run written to {res.run_dir}\n"
        f"  P_grid mean {s['p_grid_W']['time_mean']:.6g} W, median {s['p_grid_W']['median']:.6g} W,"
        f" peak {s['p_grid_W']['peak']:.6g} W\n"
        f"  availability {s['availability']:.6g}, capacity factor "
        f"{'n/a' if cf is None else f'{cf:.6g}'}, delivered energy "
        f"{s['delivered_energy_J']:.6g} J\n"
    )
    if s["diurnal_aliasing"]["warning"]:
        out.write(
            "  WARNING: diurnal aliasing (hour-angle gap "
            f"{s['diurnal_aliasing']['hour_angle_max_gap_rad']:.4g} rad); see logs/run.log\n"
        )
    return EXIT_OK


def _cmd_plot(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from helioswarm.plotting import make_figures
    from helioswarm.report import write_report

    run_dir = Path(args.run_dir)
    try:
        paths = make_figures(run_dir)
        report = write_report(run_dir)
    except (OSError, KeyError, ConfigError) as exc:
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
    parser.error(f"unknown command {command!r}")  # argparse already rejects unknown commands


if __name__ == "__main__":
    raise SystemExit(main())
