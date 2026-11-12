"""Tests for helioswarm.kernels: manifest, checksum verification, idempotent fetch, scoped
SPICE loading.

The fetch/verify/load mechanics are tested on small synthetic *text* kernels served from a
temporary directory through ``file://`` URLs (no network); the real kernel set is checked in
the ``spice``-marked tests at the end.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import spiceypy

from helioswarm.cli import main
from helioswarm.kernels import (
    DEFAULT_MANIFEST_PATH,
    KERNEL_DIR_ENV,
    KernelChecksumError,
    KernelError,
    KernelManifest,
    KernelMissingError,
    KernelPoolError,
    KernelSet,
    default_kernel_dir,
    fetch_kernels,
    file_digests,
)

NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"


def _text_kernel(name: str, value: int) -> bytes:
    # A minimal SPICE text kernel (NAIF Kernel Required Reading: \begindata / \begintext).
    return (
        f"KPL/PCK\n\nSynthetic test kernel {name}.\n\n\\begindata\n\n"
        f"HELIOSWARM_TEST_{value} = ( {value} )\n\n\\begintext\n"
    ).encode("ascii")


def _make_set(
    tmp_path: Path, contents: dict[str, bytes], *, mutate: dict[str, dict[str, Any]] | None = None
) -> tuple[KernelSet, Path]:
    """Serve ``contents`` from tmp/server via file:// URLs; manifest records their digests."""
    server = tmp_path / "server"
    server.mkdir(parents=True)
    records = []
    for name, data in contents.items():
        (server / name).write_bytes(data)
        rec: dict[str, Any] = {
            "name": name,
            "kind": "PCK",
            "url": (server / name).as_uri(),
            "purpose": "test",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "md5": hashlib.md5(data, usedforsecurity=False).hexdigest(),
            "md5_source": "test",
            "server_last_modified": None,
            "retrieved_utc": "2026-11-12T00:00:00Z",
        }
        rec.update((mutate or {}).get(name, {}))
        records.append(rec)
    manifest = tmp_path / "kernels.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "kernel_set_id": "test",
                "hash_policy": "test",
                "kernels": records,
            }
        )
    )
    kdir = tmp_path / "kernels"
    return KernelSet.from_manifest(manifest, kdir), server


# ---------------------------------------------------------------------------------------------
# Project manifest content
# ---------------------------------------------------------------------------------------------


def test_project_manifest_lists_the_planned_kernels_in_load_order() -> None:
    m = KernelManifest.load()
    assert m.path == DEFAULT_MANIFEST_PATH
    assert [r.name for r in m.records] == [
        "naif0012.tls",
        "pck00011.tpc",
        "gm_de440.tpc",
        "de440s.bsp",
    ]
    assert [r.kind for r in m.records] == ["LSK", "PCK", "PCK", "SPK"]
    subdirs = ["lsk/", "pck/", "pck/", "spk/planets/"]
    for rec, sub in zip(m.records, subdirs, strict=True):
        assert rec.url == NAIF_BASE + sub + rec.name
    # The manifest hash identifies the exact file bytes (used as the CI cache key).
    assert m.sha256 == hashlib.sha256(DEFAULT_MANIFEST_PATH.read_bytes()).hexdigest()


def test_project_manifest_records_naif_md5_for_de440s() -> None:
    # NAIF publishes MD5s for the planetary SPKs only:
    # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aa_checksums.txt
    # (retrieved 2026-11-12) lists "3917ee56769db332790c751e2168843d  de440s.bsp".
    recs = {r.name: r for r in KernelManifest.load().records}
    assert recs["de440s.bsp"].md5 == "3917ee56769db332790c751e2168843d"
    assert recs["de440s.bsp"].md5_source == NAIF_BASE + "spk/planets/aa_checksums.txt"
    # gm_de440.tpc: the SHA-256 recorded in M1 (docs/data_provenance.md) when GM_sun was sourced.
    assert recs["gm_de440.tpc"].sha256 == (
        "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140"
    )


def test_manifest_with_bad_sha_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KernelError, match="sha256 must be 64"):
        _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1)}, mutate={"a.tpc": {"sha256": "x"}})


def test_default_kernel_dir_honours_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KERNEL_DIR_ENV, str(tmp_path))
    assert default_kernel_dir() == tmp_path
    monkeypatch.setenv(KERNEL_DIR_ENV, "  ")
    assert default_kernel_dir() == DEFAULT_MANIFEST_PATH.parent
    monkeypatch.delenv(KERNEL_DIR_ENV)
    assert default_kernel_dir() == DEFAULT_MANIFEST_PATH.parent


# ---------------------------------------------------------------------------------------------
# Fetch and verify
# ---------------------------------------------------------------------------------------------


def test_fetch_downloads_verifies_and_is_idempotent(tmp_path: Path) -> None:
    ks, _ = _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1), "b.tpc": _text_kernel("b", 2)})
    assert ks.missing() == ("a.tpc", "b.tpc")
    first = fetch_kernels(ks)
    assert [r.action for r in first] == ["downloaded", "downloaded"]
    assert ks.missing() == ()
    assert (ks.directory / "a.tpc").read_bytes() == _text_kernel("a", 1)
    assert oct((ks.directory / "a.tpc").stat().st_mode & 0o777) == oct(0o644)
    mtime = (ks.directory / "a.tpc").stat().st_mtime_ns
    second = fetch_kernels(ks)
    assert [r.action for r in second] == ["present", "present"]
    assert (ks.directory / "a.tpc").stat().st_mtime_ns == mtime  # not re-downloaded
    assert sorted(p.name for p in ks.directory.iterdir()) == ["a.tpc", "b.tpc"]  # no .part left
    assert [p.name for p in ks.verify()] == ["a.tpc", "b.tpc"]


def test_fetch_refuses_corrupted_download_and_leaves_nothing(tmp_path: Path) -> None:
    ks, server = _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1)})
    (server / "a.tpc").write_bytes(_text_kernel("a", 9))  # same size, different content
    with pytest.raises(KernelChecksumError, match="SHA-256"):
        fetch_kernels(ks)
    assert list(ks.directory.iterdir()) == []  # neither final file nor temporary file


def test_fetch_refuses_existing_mismatching_file_without_overwriting(tmp_path: Path) -> None:
    ks, _ = _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1)})
    ks.directory.mkdir()
    bad = _text_kernel("a", 7)
    (ks.directory / "a.tpc").write_bytes(bad)
    with pytest.raises(KernelChecksumError, match=r"a\.tpc"):
        fetch_kernels(ks)
    assert (ks.directory / "a.tpc").read_bytes() == bad


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("md5", "0" * 32, "MD5"),
        ("size_bytes", 3, "size"),
        ("sha256", "0" * 64, "SHA-256"),
    ],
)
def test_verify_checks_size_sha256_and_md5(
    tmp_path: Path, field: str, value: Any, fragment: str
) -> None:
    good, _ = _make_set(tmp_path / "good", {"a.tpc": _text_kernel("a", 1)})
    fetch_kernels(good)
    bad, _ = _make_set(
        tmp_path / "bad", {"a.tpc": _text_kernel("a", 1)}, mutate={"a.tpc": {field: value}}
    )
    bad = KernelSet(manifest=bad.manifest, directory=good.directory)
    with pytest.raises(KernelChecksumError, match=fragment):
        bad.verify()


def test_verify_reports_missing_kernels_with_remedy(tmp_path: Path) -> None:
    ks, _ = _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1)})
    with pytest.raises(KernelMissingError, match="fetch-kernels"):
        ks.verify()
    with pytest.raises(KernelMissingError), ks.loaded():
        pass


def test_file_digests_match_hashlib(tmp_path: Path) -> None:
    data = bytes(range(256)) * 9000  # > 1 MiB chunk size
    p = tmp_path / "x.bin"
    p.write_bytes(data)
    sha, md5, size = file_digests(p)
    assert sha == hashlib.sha256(data).hexdigest()
    assert md5 == hashlib.md5(data, usedforsecurity=False).hexdigest()
    assert size == len(data)


def test_cli_fetch_kernels_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ks, server = _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1)})
    args = ["fetch-kernels", "--manifest", str(ks.manifest.path), "--kernel-dir", str(ks.directory)]
    assert main(args) == 0
    assert "OK: 1 kernels present and verified" in capsys.readouterr().out
    (ks.directory / "a.tpc").write_bytes(_text_kernel("a", 5))
    assert main(args) == 1
    assert "SHA-256" in capsys.readouterr().err
    assert main(["fetch-kernels", "--manifest", str(tmp_path / "nope.json")]) == 1
    assert "cannot read kernel manifest" in capsys.readouterr().err
    del server


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{bad", "not a valid JSON kernel manifest"),
        (b"\xff\xfe", "not a valid JSON kernel manifest"),
        (b'{"manifest_version": 1, "kernels": []}', "missing top-level key"),
    ],
)
def test_malformed_manifest_is_a_readable_kernel_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes, message: str
) -> None:
    # Review finding (M2): a corrupt manifest used to escape as a raw JSONDecodeError/KeyError
    # traceback from the CLI.
    bad = tmp_path / "bad.json"
    bad.write_bytes(content)
    with pytest.raises(KernelError, match=message):
        KernelManifest.load(bad)
    assert main(["fetch-kernels", "--manifest", str(bad)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ")
    assert message in err


# ---------------------------------------------------------------------------------------------
# Scoped loading (synthetic text kernels; real SPICE pool)
# ---------------------------------------------------------------------------------------------


def test_loaded_furnishes_and_always_unloads(tmp_path: Path) -> None:
    ks, _ = _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1), "b.tpc": _text_kernel("b", 2)})
    fetch_kernels(ks)
    with ks.loaded() as k:
        assert k.active
        assert spiceypy.ktotal("ALL") == 2
        assert k.kernel_ids == ("a.tpc", "b.tpc")
        assert spiceypy.gdpool("HELIOSWARM_TEST_2", 0, 1)[0] == 2.0
        k.require_active()
    assert not k.active
    assert spiceypy.ktotal("ALL") == 0
    with pytest.raises(KernelPoolError, match="not loaded"):
        k.require_active()
    # Pool variables are gone too (kclear), not only the file entries.
    assert not spiceypy.expool("HELIOSWARM_TEST_2")
    # An exception inside the block still unloads everything.
    with pytest.raises(RuntimeError, match="boom"), ks.loaded():
        raise RuntimeError("boom")
    assert spiceypy.ktotal("ALL") == 0


def test_loaded_refuses_nesting_and_foreign_kernels(tmp_path: Path) -> None:
    ks, _ = _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1)})
    fetch_kernels(ks)
    with ks.loaded():
        with pytest.raises(KernelPoolError, match="already holds 1"), ks.loaded():
            pass
        # The failed inner entry did not unload the outer set.
        assert spiceypy.ktotal("ALL") == 1
    spiceypy.furnsh(str(ks.directory / "a.tpc"))
    try:
        with pytest.raises(KernelPoolError), ks.loaded():
            pass
        assert spiceypy.ktotal("ALL") == 1  # a foreign load is left untouched
    finally:
        spiceypy.kclear()


def test_require_active_checks_kernel_kind(tmp_path: Path) -> None:
    ks, _ = _make_set(tmp_path, {"a.tpc": _text_kernel("a", 1)})
    fetch_kernels(ks)
    with ks.loaded() as k:
        k.require_active(kind="PCK")
        with pytest.raises(KernelPoolError, match="no LSK"):
            k.require_active(kind="LSK")


# ---------------------------------------------------------------------------------------------
# Real kernel set
# ---------------------------------------------------------------------------------------------


@pytest.mark.spice
def test_real_kernel_set_verifies_and_loads(kernel_set: KernelSet) -> None:
    prov = kernel_set.verify()
    assert [p.name for p in prov] == ["naif0012.tls", "pck00011.tpc", "gm_de440.tpc", "de440s.bsp"]
    for p, rec in zip(prov, kernel_set.records, strict=True):
        assert p.sha256 == rec.sha256
        assert Path(p.path).is_file()
    with kernel_set.loaded() as k:
        assert spiceypy.ktotal("ALL") == 4
        assert spiceypy.ktotal("SPK") == 1
        assert k.ids_of_kind("SPK") == ("de440s.bsp",)
        assert k.ids_of_kind("LSK") == ("naif0012.tls",)
