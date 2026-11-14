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

### E8 (2026-11-13): constellation demo sweep — N, phase distribution, scheduler (M7)

| Field | Content |
|---|---|
| Question | How do network power, availability and capacity use change with the number of satellites (1, 10, 100) on one 1 au ring, with even vs randomized Earth-relative phases over [30°, 330°), and with the greedy vs the exact per-epoch optimal scheduler, for three receiver sites with beam-slot and input-power limits? |
| Config | `configs/constellation_demo_sweep.yaml` (base `configs/constellation_demo.yaml`; 12 points, hashes in `sweep_definition.yaml`) |
| Code commit | `80d0715` + uncommitted M7 tree (dirty; src SHA-256 prefix `ed373b76`) |
| Slurm | array 60272015 (tasks 0, 1, 3, 4 hit a NODE_FAIL on holy8a26601 after 52 min and were requeued automatically; the resumable sweep skipped the finished points) |
| Result location | `$HELIOSWARM_RUNS_DIR/constellation_demo_sweep/` (`sweep_summary.parquet`, `points/p0000`…`p0011`) |
| Interpretation | One satellite 30° ahead of the Earth delivers 20.7 kW mean (end-to-end 1.5e-3; with three sites 120° apart the network is never in outage, each site alone is available ~33 % of the time). Ten satellites deliver 44.2 kW (even) / 29.7 kW (random): adding farther satellites adds little because capture falls with range. At N = 100 the receivers are the bottleneck: 174 kW with ~15.4 of 20 beam slots busy on average and 84 % of satellite-time lost to `RECEIVER_CAPACITY`; even and random phases then give the same power (173.5 vs 173.2 kW). The exact optimum beats greedy by 0.4–0.9 % (never less at any epoch) (MILP needed in 126–360 epochs where the 200 kW site limit binds). Every summary passes the double-counting check. All scenario values are assumed (clear sky all year: a weather upper bound). |

### E9 (2026-11-13): constellation demo with N = 1000 (array + merge) and GPU benchmark (M7)

| Field | Content |
|---|---|
| Question | Does the constellation pipeline scale to 1000 satellites as a Slurm array, what does the capacity-limited network deliver, and does the GPU accelerate the satellite × site × time capture batch? |
| Config | `configs/constellation_demo_n1000.yaml` (config_hash `8318afa00e68…`) |
| Code commit | `80d0715` + uncommitted M7 tree (dirty; src SHA-256 prefix `ed373b76`) |
| Slurm | link array 60272021 (5 × 200 satellites, shared), merge 60272040 (shared), GPU benchmark 60286071 (gpu_test; 60284257 ran out of host memory at 8 GB and was resubmitted with 24 GB) |
| Result location | `$HELIOSWARM_RUNS_DIR/constellation_demo_n1000/`, link files `$HELIOSWARM_RUNS_DIR/constellation_demo_n1000_links/` (1.6 GB), benchmark `$HELIOSWARM_RUNS_DIR/gpu_bench/constellation_60286071.json` |
| Interpretation | 284.5 kW network mean (peak 423 kW) from 1000 satellites (10 km² of collectors): 0.028 W per m² of collector, 0.017 W/kg of the assumed mass proxy (16.6 kt), end-to-end 2.1e-5 — ~70 times less per collector area than N = 1 (2.07 W/m²) because only ~17 satellites can be served at once (20 beam slots) and 98 % of satellite-time is `RECEIVER_CAPACITY`. The pipeline scaled linearly (≈ 20 CPU-s per satellite-year on Cascade Lake nodes; 5.6 CPU-h stage A, 61 s merge at 3.5 GB). The beam stage was 72 % of stage A; on the 10.5 M capture points the JAX kernel on an A100 MIG slice is 16× faster than the numpy batch and 27× faster than the per-point reference at 1e6 points (agreement 4e-16 and 6e-13), but a stage-A task is CPU-bound for the rest, so production stays on the CPU reference (see `docs/milestones/M7.md` D). |

### E10 (2026-11-13): GEO baseline vs heliocentric 1 au under harmonised assumptions (M7b)

| Field | Content |
|---|---|
| Question | Under identical optical and microwave technology, one satellite and one receiver site, how does a geostationary satellite above the site compare with a heliocentric satellite 30° ahead of the Earth (delivered power, availability, capacity factor, end-to-end efficiency, eclipses, capture, range, flux, mass proxy per kW), and how does it depend on transmit aperture and receiver radius? |
| Config | `configs/comparison/comparison.yaml`; cases `helio_1au_optical.yaml`, `geo_optical.yaml`, `helio_1au_microwave.yaml`, `geo_microwave.yaml` (shared blocks `_shared.yaml`, `_optical.yaml`, `_microwave.yaml` via `extends`); sweeps `*_sweep.yaml` (9 points each; hashes in each `sweep_definition.yaml`) |
| Code commit | `e86f8e7` + uncommitted M7b tree (dirty; src SHA-256 prefix `51ab61d9` for the GEO arrays, `7e00564e` for the heliocentric arrays; the trees differ only in `compare.py`) |
| Slurm | GEO optical 60309248 (5 tasks), GEO microwave 60309192, heliocentric optical 60310851 (+ 60313986 and 60313993 for the two points that hit the 15 min limit), heliocentric microwave 60310868; all on `shared`, 1 core |
| Result location | `$HELIOSWARM_RUNS_DIR/comparison/cmp_*/points/p*/`; comparison report `$HELIOSWARM_RUNS_DIR/comparison/report/` (tables copied into `docs/milestones/M7b.md`) |
| Interpretation | Per satellite and with identical technology, GEO above the site delivers far more: optical base (10 m, 1 km receiver) 704.9 kW vs 8.76 kW, microwave base (1 km array, 10 km rectenna) 2.028 MW vs 1.59 W, because the heliocentric range (0.48-0.55 au) is ~2160x the GEO range and the far-field capture collapses (2.8 % optical, 1.7e-6 microwave vs 99.9 % / 97.9 % at GEO). GEO availability 0.9911 (umbra 0.889 %, penumbra 0.130 %, 93 eclipse days) vs 0.4421 for the heliocentric satellite seen from one site; the heliocentric case wins only on eclipses. With large optical apertures and 10 km receivers the heliocentric satellite reaches 252-308 kW, within the single-site availability factor (2.24) of GEO; microwave never comes close (best 6.4 mW). GEO optical peak flux is 7.65e4 W/m^2 at D = 10 m (no receiver flux limit configured: idealised). The three GEO-optical 10 km points fail the Fresnel paraxial check (recorded). Negative result for the heliocentric architecture as a per-satellite power source under these assumptions (clear sky, no cost/lifetime); details docs/milestones/M7b.md. |

### E11 (2026-11-13): ERA5 cloud-cover download and site climatology (M8a)

| Field | Content |
|---|---|
| Question | How often is the sky over each configured and candidate receiver site clear enough for an optical power link (ERA5 hourly total cloud cover, 2014–2023), how much does that vary between years, and how correlated is cloudiness between sites of a network? |
| Config | `configs/weather/era5_sites.yaml` (12 sites; time-series product, 3 × 3 nodes) |
| Code commit | `e86f8e7` + uncommitted M8a tree (dirty) |
| Slurm | download 60341911 (60337126 with the gridded product cancelled: CDS queue too slow); climatology in session (seconds) |
| Result location | `$HELIOSWARM_WEATHER_DIR/era5/`, `$HELIOSWARM_RUNS_DIR/era5_climatology*/` |
| Interpretation | Clear fraction at `tcc ≤ 0.5` from 0.38 (MVP site 0 N 0 E) to 0.89 (aswan); year-to-year range up to ±0.1. The clearest candidate per longitude sector (atacama, aswan, Pilbara) defines `configs/constellation_demo_era5_clear_sites.yaml`. All-sites-cloudy hours slightly exceed the independence product (0.0298 vs 0.0260 demo sites). `tcc` is a grid-box fraction, not a line-of-sight measurement (`docs/model_assumptions.md` M8a). |

### E12 (2026-11-13): MVP and constellation demo under ERA5 weather of 2014–2023 (M8a)

| Field | Content |
|---|---|
| Question | How much optical energy and availability does real cloud cover remove, how much does it vary between weather years, and how much do site diversity and site choice recover? |
| Config | `configs/mvp_era5_years_sweep.yaml`, `configs/constellation_demo_era5_years_sweep.yaml`, `configs/constellation_demo_era5_clear_sites_years_sweep.yaml` (10 historical years each), clear-sky references `configs/mvp_single_satellite.yaml` (job 60342911) and `configs/constellation_demo_clear_sites.yaml` (job 60344241) |
| Code commit | `e86f8e7` + uncommitted M8a tree (dirty) |
| Slurm | 60342908, 60343733, 60343742 (test partition, one job per sweep) |
| Result location | `$HELIOSWARM_RUNS_DIR/{mvp_era5_years_sweep,constellation_demo_era5_years_sweep,constellation_demo_era5_clear_sites_years_sweep,constellation_demo_clear_sites,mvp_single_satellite_m8a_check}/` |
| Interpretation | MVP: 37–48 % of the clear-sky energy, availability 0.16–0.20 instead of 0.42. Demo network (N = 10, three sites): 68–74 % of clear sky, availability 0.91–0.95. Same satellites with the three ERA5-chosen sites: 82–88 % of clear sky (+15.6 % energy vs the demo sites), availability 0.97–0.99. The default `scenario` path reproduces the M6 MVP time series bit for bit. |

### E13 (2026-11-13): parameterised economics of the saved M6/M7/M8a runs (M8a)

| Field | Content |
|---|---|
| Question | Under transparent, provenance-labelled cost scenarios (NASA OTPS 2024 values where available, the rest assumed), what LCOE/NPV ranges do the simulated systems imply, and which terms dominate? |
| Config | `configs/economics/{optimistic,nominal,conservative}.yaml` (nominal with a 2 000-sample seeded Monte Carlo) |
| Code commit | `e86f8e7` + uncommitted M8a tree (dirty) |
| Slurm | none (in session, seconds per run directory; reads saved summaries only) |
| Result location | `$HELIOSWARM_RUNS_DIR/economics_m8a/<run>/economics.{json,md}` |
| Interpretation | Every case is 3–6 orders of magnitude above 0.05 USD/kWh (demo N = 10 clear sky: 93 / 417 / 1 750 USD/kWh, MC 345–804; N = 1000: 436 / 3 700 / 21 600; MVP optical 1 250 / 2 670 / 6 650; MVP microwave undefined, 2.8 kWh/yr). Receiver area and fixed operations dominate the MVP; transport (mass ratio ~3.1 at ~4.9 km/s) and replacements dominate the constellation. Scenario ranges for an assumed technology, not predictions (PDF §12.5). |
