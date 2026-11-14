"""Cross-check the two ERA5 CDS products used by helioswarm.era5 (M8a).

For every grid node with gridded single-levels files (reanalysis-era5-single-levels, one file per
year) and a time-series file (reanalysis-era5-single-levels-timeseries) in the weather directory,
compare total cloud cover at all 3 x 3 nodes and all hours the two have in common and print the
maximum absolute difference. Usage:

    uv run python scripts/era5_product_crosscheck.py [WEATHER_DIR]
"""

from __future__ import annotations

import sys

import numpy as np

from helioswarm.era5 import (
    DATASET,
    DATASET_TIMESERIES,
    GridNode,
    load_node_series,
    read_provenance,
    weather_dir,
)


def main() -> int:
    root = weather_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    gridded = root / "era5" / DATASET
    n_bad = 0
    for node_dir in sorted(p for p in gridded.iterdir() if p.is_dir()) if gridded.is_dir() else []:
        years = sorted(
            int(
                read_provenance(f.with_name(f.name[: -len(".provenance.json")])).period_start_utc[
                    :4
                ]
            )
            for f in node_dir.glob("*.nc.provenance.json")
        )
        if not years:
            continue
        prov = read_provenance(next(node_dir.glob("*.nc")))
        node = GridNode(
            lat_index=round(prov.node_latitude_deg / 0.25),
            lon_index=round(prov.node_longitude_deg / 0.25),
        )
        for y in years:
            a = load_node_series(root, node, "total_cloud_cover", [y], dataset=DATASET)
            b = load_node_series(root, node, "total_cloud_cover", [y], dataset=DATASET_TIMESERIES)
            same_axes = np.array_equal(a.block.time_utc_posix_s, b.block.time_utc_posix_s) and (
                np.array_equal(a.block.latitude_deg, b.block.latitude_deg)
                and np.array_equal(a.block.longitude_deg, b.block.longitude_deg)
            )
            diff = float(np.max(np.abs(a.block.values["tcc"] - b.block.values["tcc"])))
            ok = same_axes and diff == 0.0
            n_bad += 0 if ok else 1
            print(
                f"{node.key} {y}: {a.block.values['tcc'].size} values, axes equal {same_axes}, "
                f"max |gridded - timeseries| = {diff:.3g} -> {'OK' if ok else 'DIFFERENT'}"
            )
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
