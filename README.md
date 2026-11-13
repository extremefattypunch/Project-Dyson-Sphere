# Project-Dyson-Sphere: `helioswarm`

`helioswarm` is a reproducible Python research simulator for a **heliocentric solar-power
swarm**. Solar collectors in heliocentric orbit convert sunlight to electricity and beam power to
Earth receivers over optical (laser) or microwave links. The central question is: under which
orbital, optical/RF, thermal and receiver conditions can such collectors deliver useful
electrical power to an Earth grid? Every conversion, loss, geometry and availability condition
stays visible.

This is a feasibility study. It is **not** a claim that a Dyson sphere, or any complete
swarm, is buildable. A physically infeasible architecture is a valid result, provided the
limiting assumptions are quantified. "Dyson swarm" is only the motivating concept. The modelled
architecture is independent collectors with independent (incoherent) power links.

The requirements document is
`gpt-insights/Heliocentric_Dyson_Swarm_Coding_Agent_Implementation_Plan.pdf`. Implementation
notes for each milestone are in `docs/milestones/`.

## Install

You need Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --branch helioswarm-m0-m6 git@github.com:extremefattypunch/Project-Dyson-Sphere.git
cd Project-Dyson-Sphere
uv sync --locked --extra dev --extra xcheck --extra itur --extra jax   # exact env from uv.lock
```

The code lives on the branch `helioswarm-m0-m6` (the default branch `main` holds only the
requirements document until the branch is merged). The GitHub repository is currently
**private**: cloning needs read access (an SSH key as above, or
`https://github.com/extremefattypunch/Project-Dyson-Sphere.git` with a personal access token).

Do **not** use `uv sync --all-extras`: it also installs the `gpu` extra (`jax[cuda12]`, ~4.4 GB
of NVIDIA wheels on Linux), which only the GPU benchmark needs. Add `--extra gpu` explicitly
for that (`slurm/gpu_bench.sbatch`).

Extras:

- `dev` adds pytest, ruff, mypy and pre-commit.
- `xcheck` adds REBOUND and HCIPy, which are used only for independent cross-check tests.
- `itur` adds the ITU-R P.618/P.676/P.840 wrapper (`models.atmosphere: itu_r`, microwave).
- `jax` adds CPU JAX for the optional batched capture backend (`--backend jax`; Python ≥ 3.12).
- `gpu` adds the CUDA 12 JAX build (~4.4 GB of NVIDIA wheels); only the GPU benchmark needs it.
  CI installs every extra except `gpu`.

### Harvard FASRC Cannon cluster (Hekstra lab)

On the cluster, source the environment script **before every uv command**:

```bash
source scripts/cluster_env.sh
uv sync --locked --extra dev --extra xcheck --extra itur --extra jax   # + --extra gpu for the GPU benchmark
```

The script keeps everything out of `/n/home05`:

| What | Where |
|---|---|
| Interpreter | `/n/holylabs/.../envs/helioswarm-base` |
| Project environment | `/n/holylabs/.../envs/helioswarm` |
| uv, pre-commit and matplotlib caches | `/n/netscratch/.../cache` |
| SPICE kernels | `$HELIOSWARM_KERNEL_DIR` (netscratch) |
| Run outputs | `$HELIOSWARM_RUNS_DIR` (netscratch) |

No `.venv` is created in the repository. Netscratch is purged after 90 days. Everything stored
there can be regenerated.

To build a second, throwaway environment (e.g. to check a fresh clone without touching the
shared one), override the locations **after** sourcing the script, all on netscratch:

```bash
source scripts/cluster_env.sh
SCR=/n/netscratch/hekstra_lab/Lab/ian_poon/scratch-env      # any netscratch directory
export UV_PROJECT_ENVIRONMENT=$SCR/venv UV_CACHE_DIR=$SCR/uv-cache \
       HELIOSWARM_KERNEL_DIR=$SCR/kernels HELIOSWARM_RUNS_DIR=$SCR/runs
uv sync --locked --extra dev --extra xcheck --extra itur --extra jax
# ... and `rm -rf $SCR` when done.
```

## Quick start

```bash
uv run helioswarm fetch-kernels          # download + verify the SPICE kernels (~33 MB, once)
uv run helioswarm validate-config configs/mvp_single_satellite.yaml --show-provenance
uv run helioswarm run configs/mvp_single_satellite.yaml --out runs/mvp            # ~1 min
uv run helioswarm plot runs/mvp                       # redraw figures/ + report.md from saved data
uv run helioswarm sweep configs/optical_1au_sweep.yaml --out runs/optical_sweep   # 45 runs
uv run helioswarm validate-models --report docs/validation_report.md              # ~1 min
uv run helioswarm characterize configs/mvp_single_satellite.yaml --out runs/mvp_curves  # ~10 s
uv run helioswarm run configs/constellation_demo.yaml --out runs/constellation_demo  # 10 satellites, ~3 min
```

`helioswarm validate-config configs/*.yaml` checks every shipped file (run configurations and
sweeps). Without `--out`, `run` writes to `$HELIOSWARM_RUNS_DIR/<project.name>`.

- `validate-config` checks a configuration (exit 1 with every problem listed if invalid);
  `--show-provenance` lists every parameter with unit and provenance (`measured`, `sourced`,
  `assumed`, `swept`); `--write-resolved` writes it with all defaults explicit. A sweep file
  (`sweep_schema_version`) is recognised and every grid point is validated without running, so
  `helioswarm validate-config configs/*.yaml` checks every shipped file.
- `run` executes the single-satellite vertical slice (orbit → illumination → solar/thermal →
  retarded-time access and occultation → beam capture with pointing errors → atmosphere →
  receiver limits → grid) and writes a run directory:

  | File | Content |
  |---|---|
  | `config_resolved.yaml` | the configuration with all defaults |
  | `manifest.json` | git commit + dirty flag, package versions, kernel SHA-256, ITU-R versions, seed, host, Slurm job, timings |
  | `timeseries.parquet` | per epoch: states (J2000/SSB), access flags, reason codes, every power stage of PDF §10.1, thermal outputs, beam diagnostics (units in column names) |
  | `summary.json` | P_grid mean/median/p05/p95/peak, availability, capacity factor (denominator defined), outages, delivered energy, efficiencies, unavailability breakdown, diurnal-aliasing diagnostic |
  | `figures/`, `report.md` | plots and a one-page report drawn from the saved files only |
  | `logs/run.log` | warnings, integrator/solver diagnostics, stage wall times |

  Without `--out` the directory is `$HELIOSWARM_RUNS_DIR/<project.name>` (else
  `runs/<project.name>`); an existing run directory is replaced only with `--overwrite`. The
  config field `project.output_dir` is informational only and is **not** used by `run`.
- `sweep` runs a deterministic Cartesian grid over dotted configuration keys (each point a full
  validated run; `--summary-only` skips time series and figures) and collates
  `sweep_summary.parquet` (status `ok`, `failed`, `incomplete` or `missing` per point); failed
  points are recorded with their error, never dropped. The shipped `optical_1au_sweep.yaml`
  applies **no Sun-avoidance constraint** (`receiver.sun_avoidance_deg` = 0): at 0.7 au part of
  the delivered energy arrives while the satellite is within a few degrees of the Sun as seen
  from the receiver (quantified in `docs/milestones/M6.md`).
- `validate-models` runs the programmatic checks of PDF §7.4 (circular orbits, Horizons,
  sidereal rotation, occultation, light time), §8.5 (solar flux, zero incidence/eclipse,
  energy closure, radiator), §9 (beam), §10.5 (capacity-factor reference link), §13.1 (link
  budget, SRP, time scales), §13.2 (end to end) and §13.3 (REBOUND, HCIPy, ITU-R, convergence)
  and writes value, reference, tolerance and pass/fail into the generated block of
  `docs/validation_report.md`; a check whose optional library is missing is reported as
  "not run" and listed as unresolved.
- `characterize` writes the PDF §8.4 and §9.5 response curves of one configuration (no kernels
  needed): solar flux, PV power/efficiency/temperature, waste heat, radiator area, net power to
  the transmitter and thermal feasibility versus heliocentric distance and transmitter
  operating point (`power_thermal_vs_distance.parquet`), and expected capture versus pointing
  sigma, receiver radius and RMS wavefront error at one stated link geometry (`--range-au`,
  `--elevation-deg`; `capture_vs_pointing.parquet`, `capture_vs_wavefront.parquet`), with
  `characterization.json` and figures drawn from the saved tables.
- `--backend jax` (with the `jax` or `gpu` extra) evaluates the Airy capture through the batched
  JAX kernel, `--backend numpy_batch` through the same batched kernel on the CPU; the default
  `numpy` is the reference.
- **Constellations (M7).** A configuration with a `constellation` section (rings of
  satellites with even or seeded-random phases, optionally Earth-relative
  `orbit.phase_reference`, per-ring technology; several receiver sites with beam-slot,
  input-power and heat limits; a deterministic `greedy_max_power` or exact
  `optimal_assignment` scheduler; optional duty cycle and mass proxy) makes `run` a
  constellation run: every satellite × site link is evaluated with the single-satellite chain,
  the scheduler assigns at most one site per satellite under the summed receiver limits, and
  received powers add incoherently. Outputs: `network.parquet` (network P_grid and stage
  powers, active-satellite / valid-link / scheduled-link counts, reasons), `receivers.parquet`,
  `satellites.parquet` (areas, mass proxy, delivered energy), optional `links/` per-link
  detail (`--link-detail`, size-capped), `summary.json` (`ConstellationSummary`), figures.
  Large constellations run as Slurm arrays of satellite chunks (`--chunk-index/--chunk-count
  --candidates-dir`) plus one merge-and-schedule run (`--candidates-dir`); `sweep` accepts
  constellation keys such as `constellation.rings[0].n_satellites`.
- **GEO baseline and harmonised comparison (M7b).** `orbit.elements_center: EARTH` with J2000
  equatorial elements, or the geostationary preset `orbit.geostationary_longitude_deg`, runs a
  geocentric satellite (L0 Earth point mass + the station-keeping assumption) through the same
  chain (Earth-shadow eclipses, access, beam, link budget). `configs/comparison/` holds the
  harmonised heliocentric and GEO cases: the shared technology is one file per technology
  joined with `extends:`, so only the orbit (and the beam model its Fresnel number requires)
  differs. `helioswarm compare configs/comparison/comparison.yaml --runs-dir DIR --out OUT`
  reads the saved sweep runs and writes the comparison tables (CSV + Markdown), the
  harmonised-assumption table with provenance labels, a figure and a manifest.
  `models.beam_evaluation: chebyshev_table` evaluates the beam on a validated (range,
  elevation) table (used for the near-constant GEO geometry).

Every `report.md` lists the run's *idealised* settings (lossless stages, inactive constraints,
Sun-tracking collector, clear sky for the whole window) so that a result that is an upper bound
is labelled as one (PDF §13.4), and any named approximation the configuration selects (e.g.
radial SRP on a fixed-normal collector) with its comparison path (PDF §17; also in
`manifest.json` and `logs/run.log`);
`summary.json` is a typed, versioned record (`helioswarm.metrics.RunSummary`) that includes the
PDF §7.3 orbital diagnostics (range, access and occultation fractions, pointing direction and
rate, light time min/mean/max, range rate, orbit drift).

### Slurm (Harvard Cannon; see `slurm/`)

Long runs go to Slurm jobs, never the interactive session. Create the log directory once
(`mkdir -p /n/netscratch/hekstra_lab/Lab/ian_poon/logs`), then from the repository:

```bash
sbatch slurm/run.sbatch configs/mvp_single_satellite.yaml           # -> $HELIOSWARM_RUNS_DIR/<stem>/
sbatch slurm/run.sbatch configs/mvp_single_satellite_microwave.yaml
sbatch --array=0-4 --mem=768M -t 0-00:12 slurm/sweep_array.sbatch \
       configs/microwave_1au_sweep.yaml $HELIOSWARM_RUNS_DIR/microwave_1au_sweep  # 15 points
JID=$(sbatch --parsable --array=0-4 slurm/sweep_array.sbatch configs/optical_1au_sweep.yaml \
      $HELIOSWARM_RUNS_DIR/optical_1au_sweep)
# test QOS: max 5 submitted jobs per user -> submit the collation after the array has finished
sbatch --export=ALL,COLLATE_ONLY=1 slurm/sweep_array.sbatch \
      configs/optical_1au_sweep.yaml $HELIOSWARM_RUNS_DIR/optical_1au_sweep
sbatch slurm/validate_models.sbatch                                  # rewrites docs/validation_report.md
sbatch slurm/profile.sbatch configs/mvp_single_satellite.yaml        # cProfile + stage timings
sbatch slurm/gpu_bench.sbatch                                        # gpu_test, 1 GPU: CPU vs GPU kernel
# constellation sweep (N = 1/10/100 x even/random x greedy/optimal) and the N = 1000 array + merge
sbatch --array=0-4 -t 0-02:00 --mem=4G slurm/sweep_array.sbatch \
       configs/constellation_demo_sweep.yaml $HELIOSWARM_RUNS_DIR/constellation_demo_sweep
JID=$(sbatch --parsable --array=0-4 -p shared slurm/constellation_links_array.sbatch \
      configs/constellation_demo_n1000.yaml $HELIOSWARM_RUNS_DIR/constellation_demo_n1000_links)
sbatch -p shared --dependency=afterok:$JID slurm/constellation_merge.sbatch \
      configs/constellation_demo_n1000.yaml $HELIOSWARM_RUNS_DIR/constellation_demo_n1000_links
sbatch slurm/gpu_bench_constellation.sbatch configs/constellation_demo_n1000.yaml \
      $HELIOSWARM_RUNS_DIR/constellation_demo_n1000_links             # gpu_test: N = 1000 capture batch
bash slurm/comparison_submit.sh                                      # M7b: the four comparison sweeps + flux-limit sensitivity
uv run helioswarm compare configs/comparison/comparison.yaml \
      --runs-dir $HELIOSWARM_RUNS_DIR/comparison --out $HELIOSWARM_RUNS_DIR/comparison/report
seff <jobid>; jobstats <jobid>                                       # right-sizing check after each job
```

All templates use `#!/bin/bash -l`, source `scripts/cluster_env.sh`, run `uv run --locked`,
use a node-local `TMPDIR=/scratch/${USER}_${SLURM_JOB_ID}` (removed on exit), log to
`/n/netscratch/hekstra_lab/Lab/ian_poon/logs/%x-%j.out` and write outputs under
`$HELIOSWARM_RUNS_DIR`. Resource requests are the measured values in
`docs/milestones/M6.md` (section D). The vertical slice is single-threaded and CPU-bound (the
orbit integration is ~86 % of a run), so only the GPU benchmark requests a GPU.

Development checks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

## Configuration

A single YAML file defines every physical parameter of a run. See
`configs/mvp_single_satellite.yaml` and the schema in `src/helioswarm/config.py`.

The schema follows these rules:

- **Units:** SI is used internally. Config-boundary units (au, degrees, days, dB) appear only in
  field names that carry the unit as a suffix, for example `semimajor_axis_au` and
  `pointing_sigma_rad`.
- **Provenance:** every field records its provenance, unit and description.
- **Strictness:** unknown keys, type coercion, NaN/inf and contradictory settings are rejected.
  For example, an Airy beam with a tapered aperture fails, and so does an optical link with a
  rectenna receiver.
- **Versioning:** `schema_version` is required (no default) and checked, so that old runs stay
  interpretable.

## Project status

Test counts are the size of the whole suite (`uv run pytest -q`) at the end of each milestone,
after its review, as recorded in `docs/milestones/`; every SPICE, REBOUND, HCIPy and ITU-R test
runs on the cluster environment (the one skip is the real-GPU test, run by
`slurm/gpu_bench.sbatch`).

| Milestone | Content | Status | Tests (suite) |
|---|---|---|---|
| M0 Scaffold | env + lock, package layout, config schema, `validate-config`, lint/type/test, CI | done | 192 passed |
| M1 Analytic orbit | constants, two-body elements ↔ state, Kepler propagation, invariants | done | 368 passed |
| M2 Ephemeris geometry | SPICE kernels, UTC→TDB, ephemeris provider, L1 propagator, SRP | done | 504 passed |
| M3 Access + light time | ground sites, Earth rotation, retarded-time targeting, occultation | done | 628 passed |
| M4 Solar + thermal | S(r), PV efficiency law, PV and radiator thermal balance | done | 717 passed |
| M5 Beam physics | Airy/Gaussian/Fresnel optics, microwave aperture, capture, pointing, atmosphere | done | 874 passed |
| M6 End-to-end single satellite | link budget, system run, metrics, run outputs, CLI run/plot/sweep/validate-models, Slurm templates, optional JAX/GPU capture kernel | done | 972 passed, 1 skipped |
| M6b Completeness audit | `docs/compliance_matrix.md`; typed summary, §7.3 diagnostics, manifest window/units, `characterize`, idealised-bound labels, microwave sweep, fresh-clone reproduction, review fixes | done | 1059 passed, 1 skipped |
| M7 Constellation | rings/phases (Earth-relative), stacked L1, link candidates per satellite × site, deterministic capacity-constrained scheduler (greedy + exact), incoherent network sums, network/receiver/satellite outputs, mass proxy, Slurm arrays, GPU benchmark | done | 1155 passed, 1 skipped |
| M7b GEO baseline | geocentric orbits + geostationary preset, Earth-shadow eclipses, Fresnel-Hankel at the GEO range on a validated Chebyshev beam table, `extends` configs, harmonised comparison (`helioswarm compare`), Slurm sweeps | done | 1257 passed, 2 skipped |
| M8 Optimisation/uncertainty | sweeps, constrained optimisation, sensitivity | not started | — |
| M9 Paper package | frozen configs, figures, methods/validation notes | not started | — |

## Headline results so far

All numbers below are copied from the committed validation report
(`docs/validation_report.md`, generated block, Slurm job 60127176) and from the M6 run
summaries (`summary.json`) recorded in `docs/milestones/M6.md` and `experiments/README.md`.
Runs: one year from 2030-01-01 UTC, hourly output, run directories under
`$HELIOSWARM_RUNS_DIR = /n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs` (netscratch,
90-day purge; regenerate with the commands above). Every design value is an **assumed**
scenario input (the optical design is the PDF §5 schema example, which the PDF calls "not a
recommended engineering design").

| Quantity | Optical MVP | Microwave MVP |
|---|---|---|
| Run (Slurm job) | `mvp_single_satellite/` (60110148; repeated as 60127166) | `mvp_single_satellite_microwave/` (60110151; repeated as 60127172) |
| Design | 10 m aperture, 1 µm, 1 km laser-PV receiver | 1 km phased array, 2.45 GHz, 10 km rectenna, ITU-R p = 1 % |
| PV power (mean) | 3.6706 MW | 3.6706 MW |
| P_grid time mean | **1751.1 W** | **0.3145 W** |
| P_grid median / p95 / peak | 0 / 5647 / 5937 W | 0 / 1.019 / 1.073 W |
| Availability | 0.42209 (sky clear all year: a weather upper bound) | 0.42209 (no weather outage; ITU-R attenuation exceeded 1 % of the time applied at every epoch) |
| Capacity factor (rated power) | 0.2970 (5895.9 W) | 0.2951 (1.0658 W) |
| Delivered energy | 5.526e10 J = 15.35 MWh | 9.925e6 J = 2.76 kWh |
| End-to-end efficiency E_grid/E_incident | 1.287e-4 | 2.31e-8 |
| Expected capture when available (median) | 6.31e-3 | 3.88e-7 |
| Outages: count / longest | 366 / 14.0 h | 366 / 14.0 h |

- **Negative result:** diffraction dominates. At ~1.15 au the 10 m, 1 µm beam's first-null
  radius is ~21 km (dark ring ~42 km across) and a 1 km receiver captures < 1 % of it; microwave power beaming from ~1 au with
  a 1 km array delivers ~0.3 W.
- **Sweep** (`optical_1au_sweep/`, array 60112261, repeated as 60127591, collated by 60130847;
  45 points, all ok): 0.3 and 0.5 au deliver nothing (PV temperature 558.8 K and 430.5 K above
  the assumed 400 K limit, so the transmitter never operates); the best point is 0.7 au,
  D = 100 m, 10 km receiver: **604.5 kW** mean, capacity factor 0.44, end-to-end efficiency
  0.0218. That point applies no Sun-avoidance constraint: 1.6 % of its energy arrives within 1°
  and 14.4 % within 10° of the Sun as seen from the receiver.
- **Sampling bias:** hourly steps alias with the diurnal cycle: availability 0.4221 at 3600 s
  vs 0.4416 at the day-incommensurate 3896 s step (`mvp_single_satellite_step3896s/`, job
  60110154, repeated as 60127174); delivered energy −1.2 %.
- **Validation** (`helioswarm validate-models`, job 60127176): 37 checks, 37 passed, 0 failed,
  0 not run; e.g. Earth–Sun state vs JPL Horizons 5.3 cm, encircled energy at the first Airy
  dark ring 0.837784869173, REBOUND full force model 1.03 m over one year, ITU-R P.618-13
  validation example 1.5e-7 dB (production P.676-12 gas term regression-pinned only; open item).

- **GEO baseline (M7b, `configs/comparison/`, Slurm arrays 60309248/60309192/60310851/60310868
  and 60313986/60313993):** with identical technology, one satellite and one receiver site,
  a GEO satellite above the site delivers 704.9 kW (optical) / 2.028 MW (microwave) against
  8.76 kW / 1.59 W from a heliocentric satellite 30° ahead of the Earth (availability 0.991 vs
  0.442; GEO loses 0.89 % of the year to eclipses). Only large optical apertures with 10 km
  receivers bring the heliocentric case within the single-site availability factor (~2.2) of
  GEO. Tables and findings: `docs/milestones/M7b.md`.

## Reproducing from a fresh clone

The definition of done (PDF §19) was checked in M6b by following only this README in a clean
shell with a throwaway environment on netscratch; the procedure and its outcome are in
`docs/compliance_matrix.md` ("Fresh-clone reproduction") and `docs/milestones/M6b.md`. In
short:

```bash
git clone --branch helioswarm-m0-m6 git@github.com:extremefattypunch/Project-Dyson-Sphere.git
cd Project-Dyson-Sphere
source scripts/cluster_env.sh            # cluster only; then optionally the throwaway overrides above
uv sync --locked --extra dev --extra xcheck --extra itur --extra jax
uv run helioswarm fetch-kernels
uv run helioswarm validate-config configs/*.yaml
uv run helioswarm run configs/mvp_single_satellite.yaml --out $HELIOSWARM_RUNS_DIR/mvp_single_satellite
uv run helioswarm run configs/mvp_single_satellite_microwave.yaml \
    --out $HELIOSWARM_RUNS_DIR/mvp_single_satellite_microwave
uv run helioswarm plot $HELIOSWARM_RUNS_DIR/mvp_single_satellite     # figures + report from saved data
uv run pytest -q                                                      # full suite
```

The two runs reproduce the headline table above (compare `summary.json` with
`docs/milestones/M6.md`); on the cluster run them as Slurm jobs (`slurm/run.sbatch`). The
sweeps reproduce with the `sweep` commands in the Slurm section.

## Documentation

- `docs/model_assumptions.md`: every equation, assumption and validity range.
- `docs/data_provenance.md`: kernels, external data and the provenance of parameter defaults.
- `docs/validation_report.md`: tests, independent comparisons, tolerances and discrepancies.
- `docs/milestones/`: per-milestone implementation notes (sections A–F of PDF §18).
- `docs/compliance_matrix.md`: every PDF requirement in M0–M7b scope with its evidence and status
  (M6b audit), including the deferred M7–M9 items and the fresh-clone reproduction.
- `experiments/README.md`: the experiment log convention.

## Notes vault

Created at: 2026-11-09T10:52:24.299Z
Active markdown file: Project Dyson Sphere.md
Obsidian vault: Obsidian Md Notes
