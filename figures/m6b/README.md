# M6b figures and tables (copies of Slurm outputs)

Small copies of outputs whose full directories lived on netscratch (`$HELIOSWARM_RUNS_DIR`,
90-day purge; regenerate with the commands in `docs/milestones/M6b.md`, section C). Each
figure states its configuration hash; `mvp_characterization.json` records the grids, the
link geometry, the git commit (`218de9b` + uncommitted M6b tree, dirty) and package versions.

| File | Source (Slurm job) |
|---|---|
| `microwave_1au_sweep_summary.csv` | `helioswarm sweep configs/microwave_1au_sweep.yaml` (array 60141210, collated by 60142439): `sweep_summary.parquet` as CSV, one row per grid point, SI units in the column names |
| `power_thermal_vs_distance.png`, `capture_vs_pointing.png`, `capture_vs_wavefront.png`, `mvp_characterization.json` | `helioswarm characterize configs/mvp_single_satellite.yaml --out $HELIOSWARM_RUNS_DIR/m6b_review_characterize` (interactive, 6 s, after the M6b review: the figures and the JSON now carry the configuration's idealised settings; first produced inside the fresh-clone reproduction, job 60144119; the regenerated tables give the same 1 au values as recorded in `docs/milestones/M6b.md` E: P_PV 3.6706 MW, T_PV 302.8 K, A_rad 438.5 m²) |
