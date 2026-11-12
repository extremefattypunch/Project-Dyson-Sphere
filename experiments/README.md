# Experiment log

Production logic lives in `src/helioswarm/`. This directory holds only the experiment log and
exploratory notebooks (PDF §4, §14.4).

Each scientific experiment gets one entry with the following fields:

| Field | Content |
|---|---|
| Date | ISO date |
| Question | The physical/engineering question being asked |
| Config | Path to the YAML config(s) and their `config_hash` (`helioswarm validate-config`) |
| Code commit | `git rev-parse HEAD` and the dirty flag |
| Result location | Run directory, e.g. under `$HELIOSWARM_RUNS_DIR` |
| Interpretation | What the result shows, including negative or failed results (PDF §13.4) |

## Entries

All M6 runs were executed as Slurm jobs on the uncommitted M6 working tree on top of commit
`1cc897e0186e74a16f72d9d7aaa67ebdc0912f81` (manifests record `git.dirty = true`; the commit that
follows contains exactly this tree). Run directories are under
`$HELIOSWARM_RUNS_DIR = /n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs` (90-day purge;
regenerate with the commands in `docs/milestones/M6.md`, C). The runs were repeated after the M6
review (jobs below) because the capacity-factor reference and summary keys changed; the
physics time series were unchanged (identical headline numbers).

### E1 (2026-11-12): optical MVP, one year

| Field | Content |
|---|---|
| Question | What does the PDF §5 example design (10 m, 1 µm laser, 1 km laser-PV receiver, 1 au orbit 30° ahead in true anomaly) deliver to the grid over one year? |
| Config | `configs/mvp_single_satellite.yaml`, config_hash `3db50d831d9546c79850b95e1857f8cab67d1a4385ade096e9e6cbe1a70eef7e` |
| Code commit | `1cc897e` + uncommitted M6 tree (dirty) |
| Slurm | 60110148 (first run), 60127166 (after the review) |
| Result location | `$HELIOSWARM_RUNS_DIR/mvp_single_satellite/` |
| Interpretation | **Negative result.** Mean P_grid 1751.1 W from 3.67 MW of PV (end-to-end 1.29e-4); availability 0.422 (geometry only); capacity factor 0.297 against a 5895.9 W rated link. Diffraction dominates: the 1 km receiver captures < 1 % of a beam whose first null is ~21 km wide at ~1.15 au. The hourly step aliases with the diurnal cycle (warning in the summary; see E3). |

### E2 (2026-11-12): microwave MVP, one year

| Field | Content |
|---|---|
| Question | Does a generous microwave design (2.45 GHz, 1 km phased array, 10 km rectenna, ITU-R atmosphere, all values assumed) do better from the same orbit? |
| Config | `configs/mvp_single_satellite_microwave.yaml`, config_hash `2343150a624d4894f22145706fe7442877fb1ec2f34890dc4bb364849c19d7a7` |
| Code commit | `1cc897e` + uncommitted M6 tree (dirty) |
| Slurm | 60110151, 60127172 |
| Result location | `$HELIOSWARM_RUNS_DIR/mvp_single_satellite_microwave/` |
| Interpretation | **Negative result.** Mean P_grid 0.31 W (capture ~4e-7); microwave beaming from ~1 au is negligible. The rated power uses the ITU-R attenuation exceeded 1 % of the time at zenith (η_atm,ref = 0.987), not clear sky. |

### E3 (2026-11-12): diurnal-aliasing validation run (3896 s step)

| Field | Content |
|---|---|
| Question | How much does the PDF's 3600 s output step bias the yearly metrics through aliasing with the ~24 h synodic day? |
| Config | `configs/mvp_single_satellite_step3896s.yaml`, config_hash `bcbb09fe67187e33021e587e7ea007508807e87e4679ca3faada2abe27016dd9` |
| Code commit | `1cc897e` + uncommitted M6 tree (dirty) |
| Slurm | 60110154, 60127174 |
| Result location | `$HELIOSWARM_RUNS_DIR/mvp_single_satellite_step3896s/` |
| Interpretation | Hourly sampling biases availability by −0.0195 absolute (0.4221 vs 0.4416) and delivered energy by −1.2 %; the hourly MVP keeps the PDF step and carries the warning. |

### E4 (2026-11-12): optical sweep over radius, aperture and receiver radius

| Field | Content |
|---|---|
| Question | How do heliocentric radius (0.3–1.2 au), transmit aperture (10/30/100 m) and receiver radius (1/5/10 km) change the delivered power of the optical design? |
| Config | `configs/optical_1au_sweep.yaml` (45 points; per-point config hashes in `sweep_summary.parquet` and `sweep_definition.yaml`) |
| Code commit | `1cc897e` + uncommitted M6 tree (dirty) |
| Slurm | array 60112261 + collation 60118113 (first run); array 60127591 + collation 60130847 (after the review) |
| Result location | `$HELIOSWARM_RUNS_DIR/optical_1au_sweep/` (copy of the table: `figures/m6/optical_1au_sweep_summary.csv`) |
| Interpretation | 0.3 and 0.5 au deliver nothing: the PV plate exceeds the assumed 400 K limit all year (thermal policy: transmitter off; rated power 0, capacity factor undefined, `reference_thermal_feasible = false`). Best point 0.7 au, D = 100 m, 10 km receiver: 604.5 kW mean. **No Sun-avoidance constraint was applied**: at 0.7 au 1.0–1.6 % of the delivered energy arrives within 1° of the Sun as seen from the receiver, 5.5–8.6 % within 5° and 10–16 % within 10°, and for 7 of the 9 points the peak P_grid occurs during a transit of the solar disk (0.005° separation; the other two peak at 1.6°); at 1.0 and 1.2 au the minimum separation is 53.7° and 13.1°. The 0.7 au results therefore depend on staring into the Sun. |

### E5 (2026-11-12): programmatic model validation

| Field | Content |
|---|---|
| Question | Do the models pass the PDF §7.4/§8.5/§9/§10.5/§13.1–13.3 checks against independent references? |
| Config | `configs/mvp_single_satellite.yaml` (base of the end-to-end checks) |
| Code commit | `1cc897e` + uncommitted M6 tree (dirty) |
| Slurm | 60118119 (31 checks), 60127176 (37 checks, after the review) |
| Result location | generated block of `docs/validation_report.md`; copy in `$HELIOSWARM_RUNS_DIR/validate_models/<jobid>.md` |
| Interpretation | 37/37 passed, none unresolved. |
