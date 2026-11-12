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
uv sync --all-extras --locked        # exact environment from uv.lock (dev + xcheck extras)
```

Extras:

- `dev` adds pytest, ruff, mypy and pre-commit.
- `xcheck` adds REBOUND and HCIPy, which are used only for independent cross-check tests.

### Harvard FASRC Cannon cluster (Hekstra lab)

On the cluster, source the environment script **before every uv command**:

```bash
source scripts/cluster_env.sh
uv sync --all-extras --locked
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
uv run helioswarm validate-config configs/mvp_single_satellite.yaml
uv run helioswarm validate-config configs/mvp_single_satellite.yaml --show-provenance
uv run helioswarm validate-config configs/mvp_single_satellite.yaml --write-resolved /tmp/resolved.yaml
```

What these commands do:

- `validate-config` prints `OK: … (schema_version 1, config_hash <sha256>)` and exits with
  status 0 when the configuration is valid.
- If the configuration is invalid, it lists every problem with its field path and the offending
  value, then exits with status 1.
- `--show-provenance` lists every resolved parameter with its unit and its provenance
  (`measured`, `sourced`, `assumed` or `swept`).
- `--write-resolved` writes the configuration with all defaults made explicit.

Simulation commands such as `run`, `plot`, `sweep` and `validate-models` arrive in later
milestones. They do not exist yet.

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
| M1 Analytic orbit | constants, two-body elements ↔ state, Kepler propagation, invariants | planned |
| M2 Ephemeris geometry | SPICE kernels, UTC→TDB, ephemeris provider, L1 propagator, SRP | planned |
| M3 Access + light time | ground sites, Earth rotation, retarded-time targeting, occultation | planned |
| M4 Solar + thermal | S(r), PV efficiency law, PV and radiator thermal balance | planned |
| M5 Beam physics | Airy/Gaussian/Fresnel optics, microwave aperture, capture, pointing, atmosphere | planned |
| M6 End-to-end single satellite | link budget, run outputs, metrics, figures, validation report | planned |
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
