"""Rebuild the request of the committed ERA5 test fixture (M8a; ``tests/data/era5``).

Without arguments: builds the CDS request of the fixture (gridded ``reanalysis-era5-single-
levels``, ``tcc``, February 2014, 3 x 3 nodes around 23.5 N 0.0 E, site ``site_lon000``) and
checks its SHA-256 against the fixture's provenance JSON (offline). With ``--download DIR``:
downloads it with :func:`helioswarm.era5.download_requests` (needs the ``weather`` extra and a
CDS account, ~/.cdsapirc), as was done for the fixture on 2026-11-13.

    uv run python scripts/era5_fixture_request.py [--download DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from helioswarm.era5 import Era5Request, download_requests, grid_node, request_area

FIXTURE = Path("tests/data/era5/N23.50_E000.00_k1_tcc_2014_m02-02.nc.provenance.json")


def fixture_request() -> Era5Request:
    """The fixture's request."""
    node = grid_node(23.5, 0.0)
    return Era5Request(
        dataset="reanalysis-era5-single-levels",
        product_type="reanalysis",
        node=node,
        neighbourhood_nodes=1,
        area=request_area(node, 1),
        year=2014,
        months=(2,),
        variables=("total_cloud_cover",),
        site_names=("site_lon000",),
    )


def main() -> None:
    """Check (and optionally download) the fixture request."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", metavar="DIR", default=None)
    args = ap.parse_args()
    req = fixture_request()
    recorded = json.loads(FIXTURE.read_text())["request_sha256"]
    ok = req.request_sha256 == recorded
    print("request sha256", req.request_sha256, "matches fixture:", ok)
    if args.download is not None:
        for o in download_requests([req], Path(args.download), log=print):
            print(o)


if __name__ == "__main__":
    main()
