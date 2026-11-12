"""Run manifest: code, environment, data and time-convention provenance (PDF §5, §14.1, p.4).

``manifest.json`` records everything needed to reproduce or audit a run: configuration hash
and schema version, git commit and dirty flag, Python and dependency versions
(``importlib.metadata``), operating system, CPU model, host name, Slurm job/array ids, the GPU
device if the JAX backend ran on one, the seed and whether the Generator was used, the SPICE
kernel names/URLs/SHA-256 (from the verified :class:`~helioswarm.kernels.KernelProvenance`),
the ITU-R recommendation versions (``itu_r`` atmosphere), the time-scale and frame
conventions, wall-clock start/end and per-stage wall times. Wall-clock values live **only** in
the manifest, so ``summary.json`` and the time-series content are reproducible bit for bit.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from helioswarm import __version__
from helioswarm.config import SCHEMA_VERSION
from helioswarm.system import SystemRunResult

MANIFEST_SCHEMA: Final[str] = "helioswarm-run-manifest/1"
"""Version tag of the manifest.json layout."""

RECORDED_PACKAGES: Final[tuple[str, ...]] = (
    "helioswarm",
    "numpy",
    "scipy",
    "astropy",
    "spiceypy",
    "pydantic",
    "pyyaml",
    "pandas",
    "pyarrow",
    "matplotlib",
    "itur",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
)
"""Distributions whose installed versions are recorded (absent ones as ``null``)."""

TIME_SCALE_NOTE: Final[str] = (
    "Internal time scale TDB (seconds past J2000, SPICE ET). UTC strings of the configuration "
    "are converted once with SPICE str2et (naif0012.tls leap seconds); UTC and TDB are never "
    "mixed implicitly. Output epochs are satellite emission epochs t_emit; arrival epochs "
    "t_arrival = t_emit + light time are stored alongside."
)
FRAME_NOTE: Final[str] = (
    "Inertial frame J2000 (ICRF-aligned), origin SOLAR SYSTEM BARYCENTER for every state; "
    "orbital elements Sun-centred ECLIPJ2000 at the configuration boundary only; Earth-fixed "
    "frame IAU_EARTH (pck00011) for the receiver site; SI units throughout."
)


@dataclass(frozen=True)
class GitInfo:
    """Source-code revision.

    Attributes:
        commit: ``git rev-parse HEAD`` or ``None`` (not a git checkout / git unavailable).
        dirty: Uncommitted changes to tracked files (``git status --porcelain
            --untracked-files=no``), or ``None``.
        repository: Top-level directory, or ``None``.
    """

    commit: str | None
    dirty: bool | None
    repository: str | None


def git_info(start: Path | None = None) -> GitInfo:
    """Git revision of the source tree containing ``start`` (default: this package)."""
    here = start if start is not None else Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(here), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    top = git("rev-parse", "--show-toplevel")
    if top is None:
        return GitInfo(commit=None, dirty=None, repository=None)
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return GitInfo(commit=commit, dirty=None if status is None else bool(status), repository=top)


def package_versions(names: tuple[str, ...] = RECORDED_PACKAGES) -> dict[str, str | None]:
    """Installed versions (``importlib.metadata.version``) or ``None`` if not installed."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def cpu_model() -> str | None:
    """CPU model name from ``/proc/cpuinfo`` (Linux) or :func:`platform.processor`."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def gpu_devices(backend: str) -> list[str]:
    """JAX devices used by the ``jax`` backend (``[]`` for numpy)."""
    if backend != "jax":
        return []
    try:
        import jax
    except ImportError:  # pragma: no cover - depends on the installed extras
        return []
    return [f"{d.platform}:{d.device_kind}:{d.id}" for d in jax.devices()]


def slurm_info() -> dict[str, str | None]:
    """Slurm job identifiers from the environment (``None`` outside Slurm)."""
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_NODELIST",
    )
    return {k: os.environ.get(k) for k in keys}


def utc_now_iso() -> str:
    """Current wall-clock time, ISO-8601 UTC with ``Z`` (manifest timestamps only)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class RunManifest:
    """PDF p.4 ``RunManifest`` (written as ``manifest.json``).

    Attributes:
        manifest_schema: :data:`MANIFEST_SCHEMA`.
        helioswarm_version: Package version.
        config_hash: SHA-256 of the canonical resolved configuration.
        config_schema_version: ``schema_version`` of the configuration.
        project_name: ``project.name``.
        seed: ``project.seed``.
        rng_state_unchanged: The run's Generator was not drawn from (no stochastic step).
        backend: Beam-capture backend.
        git: Source revision.
        python_version: ``sys.version``.
        packages: Distribution versions.
        os: ``platform.platform()``.
        cpu_model: CPU model.
        hostname: Host name.
        slurm: Slurm identifiers.
        gpu_devices: Devices of the JAX backend.
        kernels: Kernel provenance (name, kind, url, size, SHA-256, MD5, retrieval time, path).
        ephemeris_source_ids: Ephemeris / Earth-orientation data ids.
        itu_r_versions: ITU-R recommendations used, or empty.
        itu_r_polarization_tilt_deg: Rain polarisation tilt used, or ``None``.
        time_scale: Time-scale handling.
        frame: Frame/origin/unit conventions.
        command: Command line.
        wall_clock_start_utc: Run start (UTC).
        wall_clock_end_utc: Run end (UTC).
        stage_wall_s: Wall time per stage [s].
        outputs: Output file -> SHA-256 (``timeseries_content_sha256`` is the content hash).
    """

    manifest_schema: str
    helioswarm_version: str
    config_hash: str
    config_schema_version: int
    project_name: str
    seed: int
    rng_state_unchanged: bool
    backend: str
    git: GitInfo
    python_version: str
    packages: dict[str, str | None]
    os: str
    cpu_model: str | None
    hostname: str
    slurm: dict[str, str | None]
    gpu_devices: list[str]
    kernels: list[dict[str, Any]]
    ephemeris_source_ids: list[str]
    itu_r_versions: list[str]
    itu_r_polarization_tilt_deg: float | None
    time_scale: str
    frame: str
    command: list[str]
    wall_clock_start_utc: str
    wall_clock_end_utc: str
    stage_wall_s: dict[str, float]
    outputs: dict[str, str]

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-ready mapping of the manifest."""
        return asdict(self)


def build_manifest(
    result: SystemRunResult,
    *,
    wall_clock_start_utc: str,
    wall_clock_end_utc: str,
    outputs: dict[str, str],
    command: list[str] | None = None,
) -> RunManifest:
    """Assemble the :class:`RunManifest` of a finished run."""
    cfg = result.config
    return RunManifest(
        manifest_schema=MANIFEST_SCHEMA,
        helioswarm_version=__version__,
        config_hash=result.config_hash,
        config_schema_version=SCHEMA_VERSION,
        project_name=cfg.project.name,
        seed=cfg.project.seed,
        rng_state_unchanged=result.rng_state_unchanged,
        backend=result.backend,
        git=git_info(),
        python_version=sys.version,
        packages=package_versions(),
        os=platform.platform(),
        cpu_model=cpu_model(),
        hostname=socket.gethostname(),
        slurm=slurm_info(),
        gpu_devices=gpu_devices(result.backend),
        kernels=[asdict(k) for k in result.kernel_provenance],
        ephemeris_source_ids=list(result.source_ids),
        itu_r_versions=list(result.itu_r_versions),
        itu_r_polarization_tilt_deg=result.itu_r_polarization_tilt_deg,
        time_scale=TIME_SCALE_NOTE,
        frame=FRAME_NOTE,
        command=list(command) if command is not None else list(sys.argv),
        wall_clock_start_utc=wall_clock_start_utc,
        wall_clock_end_utc=wall_clock_end_utc,
        stage_wall_s={t.stage: t.wall_s for t in result.timings},
        outputs=dict(outputs),
    )


__all__ = [
    "FRAME_NOTE",
    "MANIFEST_SCHEMA",
    "RECORDED_PACKAGES",
    "TIME_SCALE_NOTE",
    "GitInfo",
    "RunManifest",
    "build_manifest",
    "cpu_model",
    "git_info",
    "gpu_devices",
    "package_versions",
    "slurm_info",
    "utc_now_iso",
]
