"""Download and verify the helioswarm SPICE kernel set (wraps ``helioswarm fetch-kernels``).

Usage (from the repository root, inside the project environment)::

    uv run python scripts/fetch_kernels.py [--manifest data/kernels/kernels.json] [--kernel-dir DIR]

Kernels go to ``--kernel-dir``, else ``$HELIOSWARM_KERNEL_DIR``, else ``data/kernels/``. Every
file is checked against the size, SHA-256 and NAIF MD5 recorded in the manifest; existing
mismatching files are never overwritten. Exit status 0 = all kernels present and verified.
"""

from __future__ import annotations

import sys

from helioswarm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["fetch-kernels", *sys.argv[1:]]))
