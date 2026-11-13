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
| Interpretation | **Negative result.** Mean P_grid 1751.1 W from 3.67 MW of PV (end-to-end 1.29e-4); availability 0.422 (geometry only: the `clear` weather scenario keeps the sky clear all year, so this is a weather upper bound); capacity factor 0.297 against a 5895.9 W rated link. Diffraction dominates: the 1 km receiver captures < 1 % of a beam whose first-null radius is ~21 km (dark ring ~42 km across) at ~1.15 au. The hourly step aliases with the diurnal cycle (warning in the summary; see E3). |

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

### E6 (2026-11-13): microwave sweep over radius and transmit aperture (M6b)

| Field | Content |
|---|---|
| Question | Does a larger microwave aperture (1, 3, 10 km phased array, 2.45 GHz, 10 km rectenna) or a different heliocentric radius (0.3–1.2 au) make microwave delivery from heliocentric orbit useful? |
| Config | `configs/microwave_1au_sweep.yaml` (15 points; base `configs/mvp_single_satellite_microwave.yaml`, config_hash `2343150a…19d7a7`; per-point hashes in the table) |
| Code commit | `218de9b` + uncommitted M6b tree (dirty) |
| Slurm | array 60141210 (5 tasks × 3 points, `--mem=768M`; 2:13–2:19 wall, peak RSS 632–704 MB, CPU 96 %), collation 60142439 |
| Result location | `$HELIOSWARM_RUNS_DIR/microwave_1au_sweep/` (`sweep_summary.parquet`); copy `figures/m6b/microwave_1au_sweep_summary.csv` |
| Interpretation | **Negative result.** 15/15 points ok. 0.3 and 0.5 au deliver nothing (PV 558.8 K / 430.5 K above the assumed 400 K limit; same collector as the optical sweep). Mean P_grid at 0.7 / 1.0 / 1.2 au: 2.169 / 0.3145 / 0.09747 W (D = 1 km), 19.48 / 2.824 / 0.8753 W (3 km), 211.1 / 30.62 / 9.49 W (10 km); median expected capture ≤ 4.5e-5 everywhere. Capture grows as D² in the small-x limit (3 km / 1 km: 8.98×). The 1.0 au, 1 km point reproduces E2 exactly (0.3145 W). Even a 10 km array delivers ~0.2 kW at best from ~3.7 MW of PV. |

### E7 (2026-11-13): fresh-clone reproduction (PDF §19 definition of done; M6b)

| Field | Content |
|---|---|
| Question | Can another researcher clone, install the locked environment, obtain the kernels, run the supplied configurations, regenerate the figures and reproduce the reported metrics by following only the README? |
| Config | `configs/mvp_single_satellite.yaml`, `configs/mvp_single_satellite_microwave.yaml`; every shipped config validated |
| Code commit | clone of `helioswarm-m0-m6` at `218de9b` + the uncommitted M6b working tree applied as a patch (M6b is committed afterwards) |
| Slurm | setup in a clean `env -i` shell (no Slurm: clone, `uv sync`, `fetch-kernels`, `validate-config`); compute job 60144119 (runs, plots, characterize, gates, full test suite) |
| Result location | throwaway tree `/n/netscratch/hekstra_lab/Lab/ian_poon/m6b_fresh_clone/` (deleted after the check); outcome in `docs/compliance_matrix.md` and `docs/milestones/M6b.md` |
| Interpretation | **Reproduced.** Locked install from an empty uv cache (63 s), kernels fetched and SHA-256/MD5-verified, all five shipped configs valid; the optical and microwave MVP runs reproduce E1/E2 to every printed digit (1751.11 W, availability 0.422085, capacity factor 0.297005, 5.52609e10 J; 0.314505 W); figures and reports regenerated from saved data; ruff, ruff format and the full suite (1052 passed, 1 skipped) passed. Two problems found and fixed: `validate-config configs/*.yaml` (documented in the README) took only one file; one-off mypy INTERNAL ERROR (not reproduced; per-checkout mypy cache as a precaution). The throwaway HOME received no files. Details: `docs/compliance_matrix.md`, "Fresh-clone reproduction". |
