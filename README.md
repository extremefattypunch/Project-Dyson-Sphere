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
git clone git@github.com:extremefattypunch/Project-Dyson-Sphere.git
cd Project-Dyson-Sphere
uv sync --locked --extra dev --extra xcheck --extra itur --extra jax   # exact env from uv.lock
```

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

## Quick start

```bash
uv run helioswarm fetch-kernels          # download + verify the SPICE kernels (~33 MB, once)
uv run helioswarm validate-config configs/mvp_single_satellite.yaml --show-provenance
uv run helioswarm run configs/mvp_single_satellite.yaml --out runs/mvp            # ~1 min
uv run helioswarm plot runs/mvp                       # redraw figures/ + report.md from saved data
uv run helioswarm sweep configs/optical_1au_sweep.yaml --out runs/optical_sweep   # 45 runs
uv run helioswarm validate-models --report docs/validation_report.md              # ~1 min
```

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
- `--backend jax` (with the `jax` or `gpu` extra) evaluates the Airy capture through the batched
  JAX kernel; the default `numpy` is the reference.

### Slurm (Harvard Cannon; see `slurm/`)

Long runs go to Slurm jobs, never the interactive session. Create the log directory once
(`mkdir -p /n/netscratch/hekstra_lab/Lab/ian_poon/logs`), then from the repository:

```bash
sbatch slurm/run.sbatch configs/mvp_single_satellite.yaml           # -> $HELIOSWARM_RUNS_DIR/<stem>/
sbatch slurm/run.sbatch configs/mvp_single_satellite_microwave.yaml
JID=$(sbatch --parsable --array=0-4 slurm/sweep_array.sbatch configs/optical_1au_sweep.yaml \
      $HELIOSWARM_RUNS_DIR/optical_1au_sweep)
# test QOS: max 5 submitted jobs per user -> submit the collation after the array has finished
sbatch --export=ALL,COLLATE_ONLY=1 slurm/sweep_array.sbatch \
      configs/optical_1au_sweep.yaml $HELIOSWARM_RUNS_DIR/optical_1au_sweep
sbatch slurm/validate_models.sbatch                                  # rewrites docs/validation_report.md
sbatch slurm/profile.sbatch configs/mvp_single_satellite.yaml        # cProfile + stage timings
sbatch slurm/gpu_bench.sbatch                                        # gpu_test, 1 GPU: CPU vs GPU kernel
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

| Milestone | Content | Status |
|---|---|---|
| M0 Scaffold | env + lock, package layout, config schema, `validate-config`, lint/type/test, CI | implemented |
| M1 Analytic orbit | constants, two-body elements ↔ state, Kepler propagation, invariants | implemented |
| M2 Ephemeris geometry | SPICE kernels, UTC→TDB, ephemeris provider, L1 propagator, SRP | implemented |
| M3 Access + light time | ground sites, Earth rotation, retarded-time targeting, occultation | implemented |
| M4 Solar + thermal | S(r), PV efficiency law, PV and radiator thermal balance | implemented |
| M5 Beam physics | Airy/Gaussian/Fresnel optics, microwave aperture, capture, pointing, atmosphere | implemented |
| M6 End-to-end single satellite | link budget, system run, metrics, run outputs, CLI run/plot/sweep/validate-models, Slurm templates, optional JAX/GPU capture kernel | implemented |
| M7 Constellation | multiple satellites, scheduling, receiver capacity | not authorised |
| M8 Optimisation/uncertainty | sweeps, constrained optimisation, sensitivity | not authorised |
| M9 Paper package | frozen configs, figures, methods/validation notes | not authorised |

## Documentation

- `docs/model_assumptions.md`: every equation, assumption and validity range.
- `docs/data_provenance.md`: kernels, external data and the provenance of parameter defaults.
- `docs/validation_report.md`: tests, independent comparisons, tolerances and discrepancies.
- `docs/milestones/`: per-milestone implementation notes (sections A–F of PDF §18).
- `experiments/README.md`: the experiment log convention.

## Notes vault

Created at: 2026-11-09T10:52:24.299Z
Active markdown file: Project Dyson Sphere.md
Obsidian vault: Obsidian Md Notes
