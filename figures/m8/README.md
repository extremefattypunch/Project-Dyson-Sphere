# figures/m8

Small tables and figures copied from the M8 Slurm outputs on netscratch
(`$HELIOSWARM_RUNS_DIR/m8_*`); every file is produced by a command listed in
`docs/milestones/M8.md` C from saved run data. Nothing here is edited by hand.

| File | Source |
|---|---|
| `optimization_*_summary.md`, `optimization_*_pareto.png`, `optimization_*_convergence.png`, `optimization_*_pareto_union.csv` | `helioswarm optimize ... --collate` (`<out>/collated/`) |
| `optimization_*_reevaluation.md` | `helioswarm optimize ... --reevaluate` (`<out>/reevaluation/`) |
| `sanity_aperture_net_power.json` | `helioswarm optimize configs/optimization.yaml --sanity` |
| `uncertainty_*_summary.{md,json}`, `uncertainty_*_sobol.png`, `uncertainty_*_monte_carlo.png` | `helioswarm uncertainty ... --collate` (`<out>/collated/`) |
| `helio_vs_geo_economics.md` | `scripts/m8_helio_vs_geo_economics.py` over `helioswarm economics` outputs |
| `comparison_era5_{example_site,aswan}_points.md` | `helioswarm compare configs/comparison/era5/comparison_era5_*.yaml` (point = weather year 2014 + index) |
