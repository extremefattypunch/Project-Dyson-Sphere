"""Command-line interface (``helioswarm``).

Implemented subcommands (later milestones add more; none exist as placeholders):

``helioswarm validate-config CONFIG [--show-provenance] [--write-resolved PATH]``
    Validate a run configuration. Exit code 0 if valid, 1 if invalid or unreadable,
    2 for command-line usage errors (argparse convention).

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
from typing import TextIO

from helioswarm import __version__
from helioswarm.config import (
    ConfigError,
    RunConfig,
    config_hash,
    load_config,
    provenance_table,
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
        help="validate a run configuration file",
        description="Validate a YAML run configuration against the helioswarm schema. "
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
    return parser


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


def _cmd_validate_config(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
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
    parser.error(f"unknown command {command!r}")  # argparse already rejects unknown commands


if __name__ == "__main__":
    raise SystemExit(main())
