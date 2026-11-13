"""SPICE kernel set: manifest, checksum verification, idempotent fetching and scoped loading.

The kernel set is described by ``data/kernels/kernels.json`` (the *manifest*): file name, kind,
source URL on naif.jpl.nasa.gov, purpose, size, SHA-256 (trust-on-first-use, see the
manifest's ``hash_policy``), NAIF-published MD5 where one exists, and retrieval date. The
kernel files themselves live in a *kernel directory*:

* ``$HELIOSWARM_KERNEL_DIR`` if set (on the Cannon cluster ``scripts/cluster_env.sh`` points it
  at netscratch), otherwise
* ``data/kernels/`` of the repository (portable default; the files are git-ignored).

SPICE keeps one process-wide kernel pool. To keep that global state out of the rest of the
code, kernels are only ever furnished through :meth:`KernelSet.loaded`, a context manager that

1. verifies every file against the manifest (size, SHA-256, MD5 if recorded),
2. requires the SPICE pool to be empty on entry (so no other kernel can silently take
   priority, and nested loading, which would unload the outer set on exit, is impossible),
3. ``furnsh``-es the files in manifest order and yields a :class:`LoadedKernels` handle,
4. on exit ``unload``-s them and calls ``kclear`` so that no kernel or pool variable leaks.

Code that needs SPICE takes the :class:`LoadedKernels` handle as an explicit argument and
checks that it is still active; nothing reads the pool outside such a scope. SPICE is not
thread-safe: one loaded scope per process at a time.

References: NAIF "Kernel Required Reading" (FURNSH/UNLOAD/KCLEAR semantics and priority of
later-loaded kernels); SpiceyPy documentation (spiceypy.readthedocs.io).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, get_args

import spiceypy

KernelKind = Literal["LSK", "PCK", "SPK"]
"""SPICE kernel types used by helioswarm (text PCKs, including GM files, are ``PCK``)."""

KERNEL_DIR_ENV: Final[str] = "HELIOSWARM_KERNEL_DIR"
"""Environment variable naming the kernel directory."""

MANIFEST_VERSION: Final[int] = 1
"""Version of the ``kernels.json`` format understood by this module."""

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH: Final[Path] = _REPO_ROOT / "data" / "kernels" / "kernels.json"
"""Manifest location in a source checkout (``src`` layout: ``src/helioswarm`` -> repo root)."""

_CHUNK_BYTES: Final[int] = 1 << 20


class KernelError(RuntimeError):
    """Base class of kernel-set errors."""


class KernelMissingError(KernelError):
    """A kernel listed in the manifest is not present in the kernel directory."""


class KernelChecksumError(KernelError):
    """A kernel file's size or checksum differs from the manifest."""


class KernelPoolError(KernelError):
    """The SPICE kernel pool is not in the state a scoped load requires."""


# ---------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelRecord:
    """One kernel file as recorded in the manifest.

    Attributes:
        name: File name (also the kernel id used in provenance records).
        kind: ``LSK``, ``PCK`` or ``SPK``.
        url: Source URL (naif.jpl.nasa.gov generic_kernels).
        purpose: What the kernel is used for.
        size_bytes: Expected file size [byte].
        sha256: Expected SHA-256 hex digest (trust-on-first-use, see the manifest policy).
        md5: NAIF-published MD5 hex digest, or ``None`` if NAIF publishes none.
        md5_source: URL of the NAIF checksum list the MD5 comes from.
        server_last_modified: HTTP ``Last-Modified`` of the source at retrieval.
        retrieved_utc: UTC time of the first retrieval that fixed ``sha256``.
    """

    name: str
    kind: KernelKind
    url: str
    purpose: str
    size_bytes: int
    sha256: str
    md5: str | None
    md5_source: str | None
    server_last_modified: str | None
    retrieved_utc: str

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name.startswith("."):
            raise ValueError(f"invalid kernel file name {self.name!r}")
        if self.kind not in get_args(KernelKind):
            raise ValueError(f"{self.name}: unknown kernel kind {self.kind!r}")
        if self.size_bytes <= 0:
            raise ValueError(f"{self.name}: size_bytes must be > 0")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError(f"{self.name}: sha256 must be 64 lower-case hex digits")
        if self.md5 is not None and (
            len(self.md5) != 32 or any(c not in "0123456789abcdef" for c in self.md5)
        ):
            raise ValueError(f"{self.name}: md5 must be 32 lower-case hex digits")


@dataclass(frozen=True)
class KernelManifest:
    """Parsed ``kernels.json``.

    Attributes:
        path: Manifest file path.
        sha256: SHA-256 of the manifest file bytes (identifies the kernel-set definition; used
            e.g. as the CI cache key and in run manifests).
        kernel_set_id: Human-readable identifier of the set.
        hash_policy: Statement of where the checksums come from.
        records: Kernel records in load order.
    """

    path: Path
    sha256: str
    kernel_set_id: str
    hash_policy: str
    records: tuple[KernelRecord, ...]

    @classmethod
    def load(cls, path: str | Path | None = None) -> KernelManifest:
        """Read and validate a manifest (default: :data:`DEFAULT_MANIFEST_PATH`)."""
        p = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
        try:
            raw = p.read_bytes()
        except OSError as exc:
            raise KernelError(
                f"cannot read kernel manifest {p}: {exc} (pass the path of "
                "data/kernels/kernels.json explicitly when running outside a source checkout)"
            ) from exc
        try:
            data: Any = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KernelError(f"{p}: not a valid JSON kernel manifest: {exc}") from exc
        if not isinstance(data, dict) or data.get("manifest_version") != MANIFEST_VERSION:
            raise KernelError(f"{p}: expected manifest_version {MANIFEST_VERSION}")
        missing = [k for k in ("kernel_set_id", "hash_policy", "kernels") if k not in data]
        if missing:
            raise KernelError(f"{p}: missing top-level key(s) {missing}")
        try:
            records = tuple(KernelRecord(**rec) for rec in data["kernels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise KernelError(f"{p}: invalid kernel record: {exc}") from exc
        names = [r.name for r in records]
        if len(set(names)) != len(names):
            raise KernelError(f"{p}: duplicate kernel names {names}")
        if not records:
            raise KernelError(f"{p}: no kernels listed")
        return cls(
            path=p,
            sha256=hashlib.sha256(raw).hexdigest(),
            kernel_set_id=str(data["kernel_set_id"]),
            hash_policy=str(data["hash_policy"]),
            records=records,
        )


def default_kernel_dir() -> Path:
    """Kernel directory: ``$HELIOSWARM_KERNEL_DIR`` if set and non-empty, else repo
    ``data/kernels``."""
    env = os.environ.get(KERNEL_DIR_ENV, "").strip()
    return Path(env) if env else DEFAULT_MANIFEST_PATH.parent


def file_digests(path: Path) -> tuple[str, str, int]:
    """Return ``(sha256_hex, md5_hex, size_bytes)`` of a file, read in 1 MiB chunks."""
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def _check_digests(record: KernelRecord, path: Path, sha256: str, md5: str, size: int) -> None:
    problems: list[str] = []
    if size != record.size_bytes:
        problems.append(f"size {size} B != manifest {record.size_bytes} B")
    if sha256 != record.sha256:
        problems.append(f"SHA-256 {sha256} != manifest {record.sha256}")
    if record.md5 is not None and md5 != record.md5:
        problems.append(f"MD5 {md5} != NAIF-published {record.md5}")
    if problems:
        raise KernelChecksumError(f"kernel {path}: " + "; ".join(problems))


# ---------------------------------------------------------------------------------------------
# Provenance and the kernel set
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelProvenance:
    """Provenance of one verified kernel file, for run manifests (PDF §14.1).

    Attributes:
        name: Kernel file name (kernel id).
        kind: Kernel type.
        path: Absolute path the file was loaded from.
        url: Source URL.
        size_bytes: Verified size [byte].
        sha256: Verified SHA-256.
        md5: Verified NAIF MD5, or ``None`` if NAIF publishes none.
        retrieved_utc: First-retrieval time recorded in the manifest.
    """

    name: str
    kind: KernelKind
    path: str
    url: str
    size_bytes: int
    sha256: str
    md5: str | None
    retrieved_utc: str


@dataclass(frozen=True)
class KernelSet:
    """A manifest plus the directory holding its files.

    Attributes:
        manifest: The parsed manifest.
        directory: Directory containing the kernel files.
    """

    manifest: KernelManifest
    directory: Path

    @classmethod
    def from_manifest(
        cls, manifest_path: str | Path | None = None, kernel_dir: str | Path | None = None
    ) -> KernelSet:
        """Build a set from a manifest (default repo manifest) and directory (default
        :func:`default_kernel_dir`)."""
        manifest = KernelManifest.load(manifest_path)
        directory = Path(kernel_dir) if kernel_dir is not None else default_kernel_dir()
        return cls(manifest=manifest, directory=directory.resolve())

    @property
    def records(self) -> tuple[KernelRecord, ...]:
        """Kernel records in load order."""
        return self.manifest.records

    def path(self, name: str) -> Path:
        """Absolute path of kernel ``name`` in this set's directory."""
        for rec in self.records:
            if rec.name == name:
                return self.directory / rec.name
        raise KeyError(f"kernel {name!r} is not in manifest {self.manifest.path}")

    @property
    def paths(self) -> tuple[Path, ...]:
        """Absolute kernel paths in load order."""
        return tuple(self.directory / r.name for r in self.records)

    def names_of_kind(self, kind: KernelKind) -> tuple[str, ...]:
        """File names of the kernels of one kind, in load order."""
        return tuple(r.name for r in self.records if r.kind == kind)

    def missing(self) -> tuple[str, ...]:
        """Names of manifest kernels absent from the directory."""
        return tuple(r.name for r in self.records if not (self.directory / r.name).is_file())

    def verify(self) -> tuple[KernelProvenance, ...]:
        """Check presence, size, SHA-256 and (if recorded) MD5 of every kernel.

        Raises:
            KernelMissingError: a file is absent (run ``helioswarm fetch-kernels``).
            KernelChecksumError: a file differs from the manifest.
        """
        missing = self.missing()
        if missing:
            raise KernelMissingError(
                f"kernels {list(missing)} not found in {self.directory}; run "
                "`helioswarm fetch-kernels` (or scripts/fetch_kernels.py)"
            )
        out: list[KernelProvenance] = []
        for rec in self.records:
            p = self.directory / rec.name
            sha, md5, size = file_digests(p)
            _check_digests(rec, p, sha, md5, size)
            out.append(_provenance(rec, p))
        return tuple(out)

    @contextmanager
    def loaded(self, *, verify: bool = True) -> Iterator[LoadedKernels]:
        """Furnish the kernel set for the duration of a ``with`` block.

        Args:
            verify: Verify checksums first (default). ``False`` only skips hashing (for test
                fixtures that verified the same files once per session); presence is always
                checked.

        Raises:
            KernelPoolError: the SPICE pool is not empty on entry.
            KernelMissingError, KernelChecksumError: see :meth:`verify`.
        """
        if verify:
            provenance = self.verify()
        else:
            missing = self.missing()
            if missing:
                raise KernelMissingError(f"kernels {list(missing)} not found in {self.directory}")
            provenance = tuple(_provenance(r, self.directory / r.name) for r in self.records)
        n_loaded = spiceypy.ktotal("ALL")
        if n_loaded != 0:
            raise KernelPoolError(
                f"the SPICE kernel pool already holds {n_loaded} kernel(s); helioswarm loads "
                "its kernel set only into an empty pool (no nesting, no foreign kernels)"
            )
        handle = LoadedKernels(kernel_set=self, provenance=provenance)
        try:
            for p in self.paths:
                spiceypy.furnsh(str(p))
            handle._active = True
            yield handle
        finally:
            handle._active = False
            for p in reversed(self.paths):
                spiceypy.unload(str(p))
            spiceypy.kclear()


def _provenance(rec: KernelRecord, path: Path) -> KernelProvenance:
    return KernelProvenance(
        name=rec.name,
        kind=rec.kind,
        path=str(path),
        url=rec.url,
        size_bytes=rec.size_bytes,
        sha256=rec.sha256,
        md5=rec.md5,
        retrieved_utc=rec.retrieved_utc,
    )


@dataclass(eq=False)
class LoadedKernels:
    """Handle proving that a :class:`KernelSet` is furnished in the SPICE pool.

    Only :meth:`KernelSet.loaded` creates active handles; the handle becomes inactive when the
    ``with`` block exits. Every SPICE-backed function takes the handle explicitly and calls
    :meth:`require_active`, so SPICE is never queried with a stale or foreign pool.

    Attributes:
        kernel_set: The set that was furnished.
        provenance: Provenance records of the furnished (and, by default, verified) files.
    """

    kernel_set: KernelSet
    provenance: tuple[KernelProvenance, ...]
    _active: bool = field(default=False, repr=False)

    @property
    def active(self) -> bool:
        """True while the kernels are furnished."""
        return self._active

    @property
    def kernel_ids(self) -> tuple[str, ...]:
        """Kernel file names in load order (the ``source_kernel_ids`` of derived states)."""
        return tuple(p.name for p in self.provenance)

    def require_active(self, *, kind: KernelKind | None = None) -> None:
        """Raise :class:`KernelPoolError` unless the set is loaded (and contains ``kind``)."""
        if not self._active:
            raise KernelPoolError(
                "kernel set is not loaded; use it inside `with kernel_set.loaded() as k:`"
            )
        if kind is not None and not self.kernel_set.names_of_kind(kind):
            raise KernelPoolError(f"the loaded kernel set contains no {kind} kernel")

    def ids_of_kind(self, kind: KernelKind) -> tuple[str, ...]:
        """Kernel ids of one kind, in load order."""
        return self.kernel_set.names_of_kind(kind)


# ---------------------------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------------------------


FetchAction = Literal["present", "downloaded"]


@dataclass(frozen=True)
class FetchResult:
    """Outcome for one kernel of :func:`fetch_kernels`.

    Attributes:
        name: Kernel file name.
        path: Final path.
        action: ``present`` (already there and verified) or ``downloaded`` (fetched and
            verified).
    """

    name: str
    path: Path
    action: FetchAction


Downloader = Callable[[str, Path], None]
"""``download(url, destination)``: write the bytes at ``url`` to ``destination``."""


def urllib_download(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination`` with :func:`urllib.request.urlopen` (stdlib)."""
    with urllib.request.urlopen(url, timeout=120) as resp, destination.open("wb") as out:
        while chunk := resp.read(_CHUNK_BYTES):
            out.write(chunk)


def fetch_kernels(
    kernel_set: KernelSet,
    *,
    download: Downloader = urllib_download,
    log: Callable[[str], None] | None = None,
) -> tuple[FetchResult, ...]:
    """Make every kernel of ``kernel_set`` present and verified (idempotent).

    For each manifest record: an existing file is verified and left alone; a missing file is
    downloaded to a temporary file in the kernel directory, verified (size, SHA-256, MD5 if
    recorded) and only then renamed into place. Nothing unverified is ever left under the final
    name.

    Raises:
        KernelChecksumError: an existing file, or a freshly downloaded one, does not match the
            manifest. Existing mismatching files are never overwritten (delete them by hand
            after investigating).
    """
    emit = log if log is not None else (lambda _msg: None)
    directory = kernel_set.directory
    directory.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    for rec in kernel_set.records:
        final = directory / rec.name
        if final.is_file():
            sha, md5, size = file_digests(final)
            _check_digests(rec, final, sha, md5, size)
            emit(f"present  {rec.name}  sha256 {sha}  ({size} B)")
            results.append(FetchResult(name=rec.name, path=final, action="present"))
            continue
        fd, tmp_name = tempfile.mkstemp(prefix=f".{rec.name}.", suffix=".part", dir=directory)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            emit(f"download {rec.url}")
            download(rec.url, tmp)
            sha, md5, size = file_digests(tmp)
            _check_digests(rec, tmp, sha, md5, size)
            tmp.chmod(0o644)  # mkstemp creates 0600; kernels are public, read-only data
            tmp.replace(final)
        finally:
            tmp.unlink(missing_ok=True)
        emit(f"verified {rec.name}  sha256 {sha}  ({size} B)")
        results.append(FetchResult(name=rec.name, path=final, action="downloaded"))
    return tuple(results)


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "KERNEL_DIR_ENV",
    "MANIFEST_VERSION",
    "Downloader",
    "FetchResult",
    "KernelChecksumError",
    "KernelError",
    "KernelKind",
    "KernelManifest",
    "KernelMissingError",
    "KernelPoolError",
    "KernelProvenance",
    "KernelRecord",
    "KernelSet",
    "LoadedKernels",
    "default_kernel_dir",
    "fetch_kernels",
    "file_digests",
    "urllib_download",
]
