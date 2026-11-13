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

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

from helioswarm import __version__
from helioswarm.config import SCHEMA_VERSION, RunConfig, approximations_in_force
from helioswarm.kernels import KernelProvenance
from helioswarm.system import StageTiming
from helioswarm.timescale import TimeGrid


class ManifestSource(Protocol):
    """What :func:`build_manifest` reads from a finished run (single-satellite
    :class:`~helioswarm.system.SystemRunResult` or constellation
    :class:`~helioswarm.constellation.ConstellationRunResult`)."""

    @property
    def config(self) -> RunConfig:
        """The validated configuration."""
        ...

    @property
    def config_hash(self) -> str:
        """Configuration hash."""
        ...

    @property
    def rng_state_unchanged(self) -> bool:
        """Whether the run's Generator was left untouched."""
        ...

    @property
    def backend(self) -> str:
        """Beam-capture backend."""
        ...

    @property
    def kernel_provenance(self) -> tuple[KernelProvenance, ...]:
        """Verified kernel files."""
        ...

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Ephemeris / Earth-orientation data ids."""
        ...

    @property
    def itu_r_versions(self) -> tuple[str, ...]:
        """ITU-R recommendations used, or empty."""
        ...

    @property
    def itu_r_polarization_tilt_deg(self) -> float | None:
        """Rain polarisation tilt [deg], or None."""
        ...

    @property
    def grid(self) -> TimeGrid:
        """Output time grid."""
        ...

    @property
    def window_end_utc(self) -> str | None:
        """UTC label of the last epoch, or None."""
        ...

    @property
    def timings(self) -> tuple[StageTiming, ...]:
        """Wall time per stage."""
        ...


MANIFEST_SCHEMA: Final[str] = "helioswarm-run-manifest/3"
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
UNITS_NOTE: Final[str] = (
    "SI internally and in every output file: m, s, kg, W, J, rad, K (dimensionless ratios "
    "unsuffixed); every quantity name carries its unit suffix. Astronomical units, degrees, "
    "days and dB appear only in configuration field names that say so and are converted at the "
    "configuration boundary."
)
FRAME_NOTE: Final[str] = (
    "Inertial frame J2000 (ICRF-aligned), origin SOLAR SYSTEM BARYCENTER for every state; "
    "orbital elements Sun-centred ECLIPJ2000 at the configuration boundary only; Earth-fixed "
    "frame IAU_EARTH (pck00011) for the receiver site; SI units throughout."
)
FRAME_NOTE_GEOCENTRIC: Final[str] = (
    "Inertial frame J2000 (ICRF-aligned), origin SOLAR SYSTEM BARYCENTER for every state; "
    "orbital elements Earth-centred (NAIF 399) J2000 equatorial (or the geostationary preset "
    "in the IAU_EARTH equator of the element epoch) at the configuration boundary only, "
    "shifted to the SSB with the Earth's ephemeris state; Earth-fixed frame IAU_EARTH "
    "(pck00011) for the receiver site and the preset; SI units throughout."
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


def source_tree_sha256(root: Path | None = None) -> str:
    """SHA-256 of the package source: every ``*.py`` under ``root`` (default: this package) in
    sorted relative-path order, each contributing its relative POSIX path, a NUL byte, its
    content and a NUL byte. Identifies the code that ran even in a dirty checkout."""
    base = root if root is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(base.rglob("*.py"), key=lambda q: q.relative_to(base).as_posix()):
        h.update(f.relative_to(base).as_posix().encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def code_provenance() -> dict[str, Any]:
    """Code provenance of this process: git commit and dirty flag, source-tree SHA-256
    (:func:`source_tree_sha256`), package version, Python version and the recorded
    distribution versions. Written into constellation stage-A link files (M7) so a merge can
    refuse chunks produced by different code."""
    g = git_info()
    return {
        "git_commit": g.commit,
        "git_dirty": g.dirty,
        "source_sha256": source_tree_sha256(),
        "helioswarm_version": __version__,
        "python_version": sys.version.split()[0],
        "packages": package_versions(),
    }


def provenance_digest(provenance: dict[str, Any] | None) -> str | None:
    """Canonical SHA-256 of a :func:`code_provenance` mapping (``None`` for ``None``)."""
    if provenance is None:
        return None
    blob = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


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
class SimulationWindow:
    """Simulated time window of a run (PDF §14.1 "simulation start/end time").

    Attributes:
        start_utc: Configured start (``time.start_utc``; converted once with SPICE ``str2et``).
        end_utc: UTC of the last output epoch (SPICE ``et2utc``, ms), or ``None``.
        start_tdb_s: First output (emission) epoch [TDB s past J2000].
        end_tdb_s: Last output epoch [TDB s past J2000].
        output_step_s: Output step [s].
        n_epochs: Number of output epochs.
        orbit_initial_epoch_utc: Epoch of the configured orbital elements (UTC).
    """

    start_utc: str
    end_utc: str | None
    start_tdb_s: float
    end_tdb_s: float
    output_step_s: float
    n_epochs: int
    orbit_initial_epoch_utc: str


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
        source_sha256: :func:`source_tree_sha256` of the package source on disk when the
            manifest is written (identifies the code of a run made from a dirty checkout;
            layout ``/3``, M7b review). Source files must not be edited while runs are in
            progress: modules imported earlier in the process are not re-read.
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
        frame: Frame/origin conventions.
        units: Unit conventions (:data:`UNITS_NOTE`).
        approximations: Named approximations selected by the configuration, each with its
            comparison path (:func:`helioswarm.config.approximations_in_force`, PDF §17);
            empty when none is in force.
        simulation_window: Simulated start/end epochs (:class:`SimulationWindow`).
        command: Command line.
        wall_clock_start_utc: Run start (UTC).
        wall_clock_end_utc: Run end (UTC).
        stage_wall_s: Wall time per stage [s].
        outputs: Output file -> SHA-256 (``timeseries_content_sha256`` is the content hash).
        stage_a_code_provenance: Constellation runs merged from array-task link files: the
            :func:`code_provenance` recorded by the tasks (identical in every file, checked at
            the merge; ``None`` for files written before it was recorded, or for runs that
            did stage A in this process, whose provenance is the manifest's own).
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
    source_sha256: str
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
    units: str
    approximations: list[str]
    simulation_window: SimulationWindow
    command: list[str]
    wall_clock_start_utc: str
    wall_clock_end_utc: str
    stage_wall_s: dict[str, float]
    outputs: dict[str, str]
    stage_a_code_provenance: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-ready mapping of the manifest."""
        return asdict(self)


def build_manifest(
    result: ManifestSource,
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
        source_sha256=source_tree_sha256(),
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
        frame=FRAME_NOTE_GEOCENTRIC if cfg.orbit.elements_center == "EARTH" else FRAME_NOTE,
        units=UNITS_NOTE,
        approximations=[a.describe() for a in approximations_in_force(cfg)],
        simulation_window=SimulationWindow(
            start_utc=cfg.time.start_utc,
            end_utc=result.window_end_utc,
            start_tdb_s=float(result.grid.epochs_tdb_s[0]),
            end_tdb_s=float(result.grid.epochs_tdb_s[-1]),
            output_step_s=float(result.grid.step_s),
            n_epochs=len(result.grid),
            orbit_initial_epoch_utc=cfg.orbit.initial_epoch_utc,
        ),
        command=list(command) if command is not None else list(sys.argv),
        wall_clock_start_utc=wall_clock_start_utc,
        wall_clock_end_utc=wall_clock_end_utc,
        stage_wall_s={t.stage: t.wall_s for t in result.timings},
        outputs=dict(outputs),
        stage_a_code_provenance=getattr(result, "stage_a_code_provenance", None),
    )


__all__ = [
    "FRAME_NOTE",
    "FRAME_NOTE_GEOCENTRIC",
    "MANIFEST_SCHEMA",
    "RECORDED_PACKAGES",
    "TIME_SCALE_NOTE",
    "UNITS_NOTE",
    "GitInfo",
    "ManifestSource",
    "RunManifest",
    "SimulationWindow",
    "build_manifest",
    "code_provenance",
    "cpu_model",
    "git_info",
    "gpu_devices",
    "package_versions",
    "provenance_digest",
    "slurm_info",
    "source_tree_sha256",
    "utc_now_iso",
]
