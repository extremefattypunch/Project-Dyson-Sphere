"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
import spiceypy

from helioswarm.kernels import KernelError, KernelSet, LoadedKernels, fetch_kernels

from config_helpers import load_mvp_data

# The optional JAX backend is tested on the CPU unless the caller selected a platform
# (slurm/gpu_bench.sbatch exports JAX_PLATFORMS=cuda for the @pytest.mark.gpu test). Without
# this, jaxlib's CUDA plugin (gpu extra) logs a cuInit failure on nodes without a GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture
def mvp_data() -> dict[str, Any]:
    """Fresh, mutable copy of the parsed MVP optical configuration mapping."""
    return load_mvp_data()


@pytest.fixture(autouse=True)
def _no_leaked_spice_state() -> Iterator[None]:
    """Every test starts and ends with an empty SPICE kernel pool.

    A test that leaks kernels fails here (and the pool is cleared so later tests are not
    affected); SPICE state therefore never crosses test boundaries.
    """
    assert spiceypy.ktotal("ALL") == 0, "SPICE kernel pool not empty at test start"
    yield
    leaked = spiceypy.ktotal("ALL")
    spiceypy.kclear()
    assert leaked == 0, f"test leaked {leaked} SPICE kernel(s)"


@pytest.fixture(scope="session")
def kernel_set() -> KernelSet:
    """The project kernel set, fetched if missing and verified once per session.

    Missing kernels are downloaded (network). If that is impossible the SPICE tests are
    skipped with the reason; a checksum mismatch is never skipped (it is an error).
    """
    ks = KernelSet.from_manifest()
    if ks.missing():
        try:
            fetch_kernels(ks)
        except KernelError:
            raise
        except OSError as exc:  # network unavailable, DNS failure, timeout
            pytest.skip(
                f"SPICE kernels {list(ks.missing())} are not in {ks.directory} and could not "
                f"be downloaded ({exc}); run `helioswarm fetch-kernels` with network access"
            )
    ks.verify()
    return ks


@pytest.fixture
def spice_kernels(kernel_set: KernelSet) -> Iterator[LoadedKernels]:
    """The verified kernel set furnished for one test (unloaded and kclear-ed afterwards)."""
    with kernel_set.loaded(verify=False) as loaded:
        yield loaded
