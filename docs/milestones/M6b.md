# M6b: M0–M6 completeness audit against the PDF acceptance lists, and fixes

Status: done; all gates pass (D, and after the review: "Review outcomes"). The audit result is
`docs/compliance_matrix.md` (215 items after the review: 172 satisfied before M6b, 24
satisfied after an M6b fix, 14 deferred to M7–M9 or to the PDF's "later" items with reasons,
5 N/A with reasons, 0 unexplained gaps). The fresh-clone
reproduction (PDF §19) succeeded; it is documented in the matrix and in
`experiments/README.md` (E7). Work was done on the uncommitted tree on top of `218de9b`; a
separate step commits.

## A. Summary

**Method.** Every page of the PDF (pp. 1–28) was read and a checklist built from §2 (fixed
decisions/non-goals that constrain M0–M6), §3 (objects, boundary, conventions), §4 (stack,
layout, engineering rules), §5 (config contract, CLI, run outputs), §6.1–§6.4, §7.1–§7.4,
§8.1–§8.5, §9.1–§9.5, §10.1–§10.5, §11.1 (all 8 steps) and the single-satellite parts of §11.4,
§13.1 (M0–M6 rows), §13.2, §13.3, §13.4, §14.1–§14.4, §15 (M0–M6 exits), §17 (risks and agent
rules) and §19 (references, definition of done). Each item was checked against the code and
the tests (file:line of the definition, test names; references resolved by a script that fails
on any unknown symbol, so no evidence string points at nothing). Out-of-scope items (scheduler,
optimisation, paper, constellation outputs, receiver double counting, coarse-to-fine finalist
re-evaluation, L3 SRP) are listed as deferred with the milestone and reason.

**Gaps found and fixed** (each with discriminating tests):

1. **§3 `SatelliteState` attitude/normal vector.** The state carried a fixed label
   `sun_facing` and no normal vector, and ignored `collector.incidence_mode`. Now
   `attitude ∈ {sun_tracking, fixed_normal}` from the configuration and
   `collector_normal_unit` (J2000 unit vector; `−r̂_helio` for Sun tracking, the configured
   vector for a fixed normal), validated as a unit 3-vector.
2. **§3/§14.1 `RunManifest` units and simulated window.** Added `units` (SI convention) and
   `simulation_window` (configured start UTC, last-epoch UTC via SPICE `et2utc`, first/last
   TDB epoch, step, epoch count, element epoch). Manifest layout `helioswarm-run-manifest/2`.
3. **§4 typed result records.** `summary.json` (~80 fields) was a nested `dict` passed to the
   report, figures and sweep collation (M6 review finding 16, deferred there). It is now the
   frozen pydantic record `metrics.RunSummary` (unknown keys rejected), written with
   `to_json_dict()` and read with `read_summary()`; the report, figures, CLI, validation and
   sweep collation use the typed record. A one-sample window gives `null` (not NaN) time means.
4. **§4 engineering rules made machine-checked.** New `tests/unit/test_engineering_rules.py`:
   every public module/class/function/method of `src/helioswarm` has a docstring (four
   `RadialBeamPattern` protocol members had none; added with units), no `global` statement and
   no module-level mutable container (two module dicts became `MappingProxyType`), no stubs for
   later milestones. Each check has a discriminating self-test.
5. **§4 layout `configs/microwave_1au_sweep.yaml`** (missing). 5 radii × 3 phased-array
   diameters (1, 3, 10 km), run on Slurm (E6: 15/15 ok; best 211.1 W mean at 0.7 au with a
   10 km array; negative result).
6. **§7.3 orbital outputs.** The diagnostics existed in memory (`AccessSummary`) but were not in
   any saved output; the inertial beam direction was not saved at all. `timeseries.parquet` now
   has `transmit_dir_{x,y,z}` (layout `helioswarm-timeseries/2`) and `summary.json` an
   `orbital_diagnostics` section (layout `helioswarm-summary/2`): Earth range, access and
   occultation fractions (any body and per body), pointing direction and rate, light time
   min/mean/max, range rate, orbit drift vs L0. `report.md` shows them.
7. **§8.4 required results and §9.5 "report capture versus pointing sigma and receiver
   radius"** (plus §9.3 phase-error sensitivity). The functions existed (`power_thermal_vs_distance`,
   `evaluate_beam`) but no shipped command produced or saved these curves. New module
   `helioswarm.characterize` and CLI `helioswarm characterize CONFIG --out DIR`: Parquet tables,
   `characterization.json` and figures drawn from the saved tables.
8. **§13.4 idealised upper bounds labelled.** The optimistic defaults were listed only in
   `docs/model_assumptions.md`. `config.idealised_settings(config)` evaluates them on the actual
   configuration; `report.md` lists them, the power-stage and composite figures name them and the
   solar figure marks the Sun-tracking collector as an optimistic bound.
9. **§14.1 external-data provenance** for the ITU-R digital maps bundled in `itur` (variables,
   measured grid spacings, temporal nature, wheel hash, attribution, licence status).
10. **§14.4 README**: status table with the suite size at each milestone (M0–M6b done, M7–M9
    not started), "Headline results so far" (numbers only from the committed validation report
    and the M6 run summaries, with run paths and job ids), correct clone instructions (private
    repository; code branch), throwaway-environment recipe, fresh-clone section.
11. **Found by the fresh clone (§19):** `helioswarm validate-config configs/*.yaml`, documented
    in the README since M6, was a usage error (the CLI took one file); it now validates every
    file given and exits 1 if any is invalid. A one-off mypy INTERNAL ERROR in the clone job
    (not reproduced) led to a per-checkout `MYPY_CACHE_DIR` in `scripts/cluster_env.sh`
    (precaution; required exports unchanged).

**Fresh-clone reproduction.** Clone of `helioswarm-m0-m6` into a throwaway netscratch tree +
the uncommitted M6b patch (the only non-README step), clean `env -i` shell with an empty HOME,
README throwaway overrides, locked install from an empty uv cache, kernels fetched and
verified, all configs validated, then Slurm job 60144119: both MVP runs reproduce the M6 runs to
every printed digit (optical 1751.11 W mean, availability 0.422085, capacity factor 0.297005,
5.52609e10 J; microwave 0.314505 W), figures/reports regenerated from saved data,
`characterize`, ruff/format clean, **1052 passed, 1 skipped**; mypy see item 11. No file was
written to HOME; the tree was deleted.

## B. Files added/changed

Added:

- `src/helioswarm/characterize.py` (response curves)
- `configs/microwave_1au_sweep.yaml`
- `tests/unit/test_characterize.py` (5), `tests/unit/test_engineering_rules.py` (70, mostly
  parametrised), `tests/summary_helpers.py` (synthetic time series, moved from
  `test_metrics.py`)
- `docs/compliance_matrix.md`, `docs/milestones/M6b.md`
- `figures/m6b/` (README, microwave sweep table, three characterisation figures and their
  metadata; 232 KB)

Changed:

- `src/helioswarm/metrics.py`: `RunSummary` and nested records, `read_summary`,
  `OrbitalDiagnostics`, `ORBITAL_DIAGNOSTIC_DEFINITIONS`; layout `helioswarm-summary/2`.
- `src/helioswarm/io.py`: `transmit_dir_{x,y,z}` columns, layout `helioswarm-timeseries/2`;
  typed summary written; `RunOutputs.summary: RunSummary`.
- `src/helioswarm/manifest.py`: `UNITS_NOTE`, `SimulationWindow`, `RunManifest.units`,
  `RunManifest.simulation_window`; layout `helioswarm-run-manifest/2`.
- `src/helioswarm/system.py`: `SystemRunResult.window_end_utc` (SPICE `et2utc`).
- `src/helioswarm/propagate.py`: `Attitude`, `NORMAL_UNIT_TOLERANCE`,
  `SatelliteState.attitude`/`collector_normal_unit`; `satellite_initial_state(...,
  attitude=..., fixed_normal_unit_j2000=...)`; `initial_state_from_config` passes the collector
  attitude.
- `src/helioswarm/config.py`: `IdealisedSetting`, `idealised_settings`;
  `RECEIVER_TECHNOLOGY_FOR_MODE` read-only.
- `src/helioswarm/report.py`, `plotting.py`: typed summary; idealised-settings section/labels;
  §7.3 table; `make_characterization_figures`, `CHARACTERIZATION_FIGURE_FILES`.
- `src/helioswarm/cli.py`: `characterize` subcommand; `validate-config` takes several files;
  `plot` reports invalid summaries; typed summary output.
- `src/helioswarm/sweep.py`: collation validates each point's summary; three new columns
  (`light_time_min_s`, `light_time_max_s`, `occultation_fraction`).
- `src/helioswarm/validation.py`: typed summary access. `src/helioswarm/aperture.py`: protocol
  docstrings.
- Tests: `test_metrics.py` (+2, typed access), `test_sweep_io.py` (valid summary record),
  `test_end_to_end.py` (§7.3 vs in-memory `AccessSummary`, manifest window/units, report
  sections), `test_l1_propagation.py` (normal vector), `test_config.py` (+1 idealised settings),
  `test_cli.py` (+1 several files).
- `scripts/cluster_env.sh` (per-checkout mypy cache), `slurm/sweep_array.sbatch` (microwave
  sizing note), `README.md`, `docs/model_assumptions.md`, `docs/data_provenance.md`,
  `experiments/README.md` (E6, E7).

No stubs for later milestones (machine-checked).

## C. Exact commands run

All from `/n/home05/ianpoon/Project-Dyson-Sphere` after `source scripts/cluster_env.sh`
(interactive node holy7c02106, 8 cores) unless stated.

```bash
uv run pytest -q -rs --durations=15                          # baseline: 972 passed, 1 skipped (216 s)
# API checks before use (installed sources): pydantic BaseModel/ConfigDict/model_copy/model_dump,
# pyarrow.parquet.write_table, spiceypy.et2utc (via helioswarm.timescale.tdb_s_to_utc)
uv run python -c "import itur, pathlib; ..."                  # ITU-R map files and grid spacings
uv run helioswarm characterize configs/mvp_single_satellite.yaml --out /n/netscratch/.../m6b/char
uv run helioswarm validate-config configs/microwave_1au_sweep.yaml
sbatch --array=0-4 --mem=768M -t 0-00:12 slurm/sweep_array.sbatch \
       configs/microwave_1au_sweep.yaml $HELIOSWARM_RUNS_DIR/microwave_1au_sweep     # 60141210
sbatch --export=ALL,COLLATE_ONLY=1 -t 0-00:02 --mem=512M slurm/sweep_array.sbatch \
       configs/microwave_1au_sweep.yaml $HELIOSWARM_RUNS_DIR/microwave_1au_sweep     # 60142439
seff 60141210_{0..4}
# Fresh clone (throwaway tree /n/netscratch/hekstra_lab/Lab/ian_poon/m6b_fresh_clone):
git ls-remote https://github.com/...   # (anonymous: 404 -> repository is private; README fixed)
git clone --branch helioswarm-m0-m6 git@github.com:extremefattypunch/Project-Dyson-Sphere.git repo
git diff HEAD --binary > m6b.patch; git ls-files --others --exclude-standard | tar -cf ... -T -
env -i HOME=$SCR/home PATH=/usr/bin:/bin bash --noprofile --norc fresh_setup.sh
#   git apply --binary m6b.patch; tar -xf untracked.tar; source scripts/cluster_env.sh
#   export UV_PROJECT_ENVIRONMENT=$SCR/venv UV_CACHE_DIR=$SCR/uv-cache \
#          HELIOSWARM_KERNEL_DIR=$SCR/kernels HELIOSWARM_RUNS_DIR=$SCR/runs
#   uv sync --locked --extra dev --extra xcheck --extra itur --extra jax
#   uv run helioswarm fetch-kernels; uv run helioswarm validate-config configs/*.yaml
#   (first attempt failed here: validate-config took one file -> CLI fixed, clone redone)
sbatch --export=NONE fresh_clone_compute.sbatch                                    # 60144119
#   (env -i inside) validate-config; run optical; run microwave; rm figures; plot x2;
#   characterize; ruff check; ruff format --check; mypy src; pytest -q -rs; print summaries
env -i ... bash mypy_check.sh      # mypy in the clone: shared cache and private cache -> pass
env -i ... bash mypy_race.sh       # 3 concurrent rounds clone + main checkout -> all pass
env -i ... bash mypy_final.sh      # per-checkout cache (new cluster_env.sh) -> ruff, format, mypy pass
ls -a /n/home05/ianpoon | wc -l    # 54 before and after; ls -A $SCR/home -> empty
rm -rf /n/netscratch/hekstra_lab/Lab/ian_poon/m6b_fresh_clone
# Final gates (main checkout)
uv run ruff check . ; uv run ruff format --check . ; uv run mypy src ; uv run pytest -q -rs
```

The compliance matrix was generated from a template by a resolver script (scratchpad, not
committed) that replaces each symbol/test placeholder by its `file:line` / `file::test` and
exits non-zero on any unresolved name.

## D. Tests and results

```text
$ uv run ruff check .          -> All checks passed!
$ uv run ruff format --check . -> 96 files already formatted
$ uv run mypy src              -> Success: no issues found in 33 source files
$ uv run pytest -q -rs         -> 1052 passed, 1 skipped in 228.74s (0:03:48)
SKIPPED [1] tests/unit/test_capture_batch.py:139: JAX_PLATFORMS=cpu: no GPU requested (run by slurm/gpu_bench.sbatch)
```

80 new collected tests (973 → 1053, checked with `pytest --collect-only`):
`test_engineering_rules.py` 70 (two checks parametrised over the 33 files of
`src/helioswarm`, plus 4), `test_characterize.py` 5, `test_metrics.py` +2, `test_config.py` +1,
`test_cli.py` +1, `test_shipped_configs.py` +1 parameter (microwave sweep). Existing tests in
`test_end_to_end.py`, `test_l1_propagation.py` and `test_sweep_io.py` gained assertions. The
whole suite takes 3 min 48 s on the shared interactive node (slowest: the M2 full-year L1 run,
48 s); the 3-minute target is exceeded by ~50 s, as at the end of M6 (3 min 37 s).

| Criterion | Test(s) | Reference / tolerance |
|---|---|---|
| SatelliteState normal = −r̂ (Sun tracking), configured vector (fixed), unit-vector validation, config wiring | `test_l1_propagation.py::test_satellite_initial_state_closed_form_with_analytic_sun`, `::test_initial_state_from_mvp_config_in_j2000_ssb` | closed form 1e-15 abs; SPICE path 1e-15 |
| §7.3 diagnostics: weighted occultation union vs per-body, light time, range rate, drift (NaN → None), direction | `test_metrics.py::test_orbital_diagnostics_hand_computed` | hand-computed, exact (sum of the per-body fractions 35/40 ≠ union 20/40 discriminates) |
| §7.3 from the saved table = in-memory `AccessSummary` | `test_end_to_end.py::test_summary_is_recomputable_from_saved_timeseries` | extremes exact; mean 1e-14; fractions ≤ 1/N (trapezoid vs sample weights, bound derived in the test) |
| Typed summary: round trip, unknown/missing keys rejected, frozen, one-sample window → None | `test_metrics.py::test_summary_record_is_typed_and_round_trips` | exact |
| Manifest units and simulation window; report sections | `test_end_to_end.py::test_cli_run_outputs_plot_and_reproducibility` | window = first/last saved epoch exactly; end UTC = start + 2 d to the ms |
| Idealised settings: exact MVP set; overriding a value removes exactly that label; perfect pointing added | `test_config.py::test_idealised_settings_of_the_mvp_and_after_overrides` | exact sets |
| §8.4 curves vs closed forms (S(r), P_abs, η_PV law, P_PV, P_dc, Q_waste, A_rad with σ from exact SI constants), operating-point difference | `test_characterize.py::test_power_thermal_curves_against_closed_forms` | 1e-15–1e-13 relative; A_rad 1e-9 (CODATA σ vs exact-SI σ differ by 3e-11) |
| §9.5 capture: centred Airy 1 − J0² − J1², strictly decreasing in σ (receivers inside the first ring), increasing in radius | `test_characterize.py::test_capture_curves_centred_airy_and_monotonicity` | 1e-9 relative; strict orderings |
| Ruze factor applied once | `test_characterize.py::test_capture_wavefront_curve_applies_the_ruze_factor_once` | 1e-12 |
| CLI `characterize`: files, metadata, requested + configured grid values, overwrite refusal, determinism, figures from saved data, invalid geometry | `test_characterize.py::test_cli_characterize_writes_tables_metadata_and_figures` | bitwise Parquet equality on rerun |
| Docstrings on every public definition; no global mutable state; no later-milestone stubs | `test_engineering_rules.py` (each check with a discriminating self-test) | exact |
| `validate-config` several files | `test_cli.py::test_validate_config_accepts_several_files` | exit codes, every file reported |
| Microwave sweep expands to 15 distinct valid points | `test_shipped_configs.py::test_shipped_sweep_expands_to_valid_points[microwave_1au_sweep.yaml]` | exact |

### Slurm jobs and `seff`

| Job | Partition / request | Result | Wall | Peak RSS | `seff` |
|---|---|---|---|---|---|
| 60141210_0…4 microwave sweep (3 points each) | test, 1 c, 768 MB, 12 min | COMPLETED, 15/15 ok | 2:15–2:19 | 632–704 MB (`/usr/bin/time`) | CPU 95.6–96.4 %, mem 45–51 % (sampled) |
| 60142439 collation | test, 1 c, 512 MB, 2 min | COMPLETED | 6 s | — | — |
| 60144119 fresh-clone compute | test, 2 c, 4 GB, 40 min | COMPLETED; every step exit 0 except a one-off mypy INTERNAL ERROR (item A.11) | 6 min | 1.07 GB (pytest) | — |

The template's 560 MB would have been exceeded by the microwave points (ITU-R maps); the
`sweep_array.sbatch` header and the README now give `--mem=768M -t 0-00:12` for this sweep.

## E. Assumptions and equations implemented

No new physics: every curve is an existing M4/M5 model evaluated with one parameter changed.
Full text in `docs/model_assumptions.md` ("Additions of the M6b completeness audit").

- Sun-tracking collector normal `n = −r_helio/|r_helio|` (geometric, same epoch); the SRP force
  remains radial for both attitudes (PDF §6.3 "simple radial SRP"); for a fixed normal this is a
  named approximation (L3, deferred to M8).
- Orbital diagnostics: time fractions `Σ_{k∈mask} w_k / T` with the trapezoid weights of
  `metrics` (M6); unweighted sample statistics for range, range rate and pointing rate; light time
  `t_a − t_e` min/mean/max; drift `|r − r_L0|` max and final.
- Characterisation (PDF §8.1–§8.4, §9.4–§9.5): `S(r) = S_1AU (au/r)²`, `P_abs = α A S`,
  `P_PV = P_abs η_PV(T)` (self-consistent T), `P_dc_to_tx = η_pc P_PV`,
  `Q_waste = P_PV − P_aperture + Q_other`, `A_rad = Q_waste/(ε σ n F (T_lim⁴ − T_space⁴))`;
  expected capture `E[η] = ∫ η(d) f_Rice(d) dd` with the configured beam, Ruze factor
  `exp(−(2π σ_w/λ)²)`. Results at 1 au reproduce the run: P_PV 3.6706 MW, T_PV 302.8 K,
  A_rad 438.5 m²; thermal feasibility (400 K PV limit) is lost between 0.55 and 0.60 au; with
  the MVP beam at 1 au and zenith, a 1 km receiver captures 0.0110 (perfect pointing), 0.0105
  at σ = 0.1 λ/D and 0.0017 at σ = λ/D.
- Idealised settings (PDF §13.4): a setting is listed when it is lossless/unconstrained in the
  configuration (the "Optimistic defaults" list of `model_assumptions.md`).
- Simulated window end: SPICE `et2utc` of the last TDB epoch (label only, never arithmetic).

## F. Known limitations and next milestone

### Deviations from the plan / task text, and why

1. **New CLI subcommand `characterize`** (not in the PDF's CLI list): the PDF requires the §8.4
   and §9.5 results but names no command; a separate, kernel-free command keeps `run` outputs
   unchanged. It is not a stub: it is tested against closed forms.
2. **Output layouts bumped** (`summary/2`, `timeseries/2`, `run-manifest/2`): new keys/columns.
   Summaries written before M6b (layout 1) still validate (`orbital_diagnostics` optional) so
   existing sweep directories can be collated; old run directories are not rewritten.
3. **Fresh clone needed one non-README step** (applying the uncommitted M6b patch) and was cloned
   over SSH because the repository is private; the README now says so.
4. **`scripts/cluster_env.sh` changed** (per-checkout mypy cache) although M0 fixed its required
   exports: only the documented "additions" block changed; the required exports are identical.
5. **`validate-models` not re-run**: M6b changes outputs, not physics; the generated block
   (job 60127176) stays valid and every check runs under pytest. The headline numbers in the
   README come only from that block and the M6 summaries, as the task required.

### Limitations / open items

- mypy INTERNAL ERROR in the fresh-clone job: cause unknown (not reproduced: re-runs with the
  shared and with a private cache, three concurrent rounds from both checkouts, and the
  per-checkout cache all passed).
- ITU-R: production P.676-12 gas term regression-pinned only (unchanged M5 open item); ITU's
  terms for the bundled digital maps not checked (maps are not redistributed).
- Not repeated in the fresh clone: the 45-point optical sweep and the 3896 s run (reproduced
  twice in M6); the microwave sweep ran from the main checkout.
- `SatelliteState.collector_normal_unit` is the initial-epoch normal; along the trajectory the
  solar model computes incidence per epoch from the attitude law (unchanged M4 behaviour).
- `characterize` uses one explicit link geometry for the capture curves; capture along a real
  trajectory comes from `run`.

### Next milestone

M7 (not started): constellation, receiver scheduling and capacity (PDF §11.2–§11.4), with the
deferred items of the compliance matrix (several sites passed to a scheduler, network totals,
mass proxy, receiver double counting).

## Review outcomes

Sixteen reviewer findings were each checked against the code before anything was changed.
14 are fixed, 1 is fixed with one sub-item refuted, and 1 is partly fixed, partly refuted and
partly deferred. None is refuted outright.

1. **(minor, physics) `idealised_settings` misses lossless stages, disabled occultation
   checks and 0 dB scenario attenuation; disabled checks are written as a measured 0.0. Fixed.**
   Confirmed: the six overrides passed `parse_config` without a label, and
   `_orbital_diagnostics` ignored the switches. Changes:
   - New labels: `transmitter.electrical_to_beam_efficiency`, `receiver.conversion_efficiency`
     and `receiver.grid_efficiency` equal to 1.0, and a disabled
     `models.{earth,sun,moon}_occultation` (each named an inactive constraint).
   - The scenario-atmosphere rule now checks the *configured* weather scenario:
     `optical_zenith_transmittance_{clear,cloudy}` = 1 or
     `microwave_zenith_attenuation_{clear,cloudy}_dB` = 0.
   - `SummaryInputs` has `{earth,sun,moon}_occultation_checked`. A disabled check gives
     `null` in `summary.json`, and the union covers only the checked bodies.
   - Tests: `test_config.py::test_idealised_settings_of_the_mvp_and_after_overrides`
     (extended) and `test_metrics.py::test_disabled_occultation_check_is_reported_as_not_evaluated`.
2. **(minor, physics) README "first null ~21 km wide" is the radius. Fixed.**
   1.22 × 1e-6 × 1.15 au / 10 m = 20.99 km is the radius. `README.md` and
   `experiments/README.md` E1 now say "first-null radius ~21 km (dark ring ~42 km across)".
3. **(minor, physics) SRP ratio `1/cos(inc)` holds only for a perfect absorber. Fixed.**
   Re-derived from the flat-plate law `F = P A cosθ [(1−ε) ŝ + 2ε cosθ n̂]` with
   `ε = C_r − 1`. The radial/true ratio is `(1+ε)/(cosθ[(1−ε)+2ε cos²θ])`, which is
   1.3/(0.5 × 0.85) = 3.06 at 60° for C_r = 1.3. The omitted part is the non-radial component
   `2ε P A cos²θ sinθ`. `docs/model_assumptions.md` and the `approximations_in_force`
   docstring now say this. The reviewer cited an equation number of Montenbruck & Gill that I
   could not check, so the text cites §3.4.
4. **(minor, tests) Comment "x <= 0.6" in the characterize capture test. Fixed.**
   π × 10 × 5000/(1e-6 × 1 au) = 1.050. The comment now reads "x <= 1.05, inside the first
   dark ring at x = 3.83".
5. **(minor, tests) No test that a configured `fixed_normal` reaches `SatelliteState`.
   Fixed.** New `test_l1_propagation.py::test_fixed_normal_config_reaches_the_satellite_state`
   (SPICE). It uses MVP data with `fixed_normal`, normal (0, 0.6, 0.8), and checks: attitude
   and normal exact, position and velocity bitwise equal to the Sun-tracking state. It fails
   if `initial_state_from_config` stops passing the attitude, because the default is
   `sun_tracking`.
6. **(minor, tests) Layout-1 compatibility not tested. Fixed.**
   - Fixture `tests/data/summary_layout1_m6_optical_sweep_p0000.json`: an unmodified M6 point
     summary (array 60127591, job 60127633), referenced by `summary_helpers.LAYOUT1_SUMMARY`.
   - `test_metrics.py::test_layout1_summaries_written_before_m6b_are_still_read` checks that
     it is read with `orbital_diagnostics = None`.
   - `test_sweep_io.py::test_sweep_dir_and_collation_record_failures` now collates it and
     checks that the three diagnostics columns are null.
7. **(minor, tests) Global-mutable-state guard misses NumPy arrays, set comprehensions,
   bytearray and memoisation caches. Fixed.** The check now also flags:
   - module-level set comprehensions, `bytearray`, and NumPy array constructors (`np.array`,
     `zeros`, `full`, `arange`, ...), unless the module calls `<name>.setflags(write=False)`;
   - `functools.lru_cache`/`cache` on any function or method.

   The self-test covers each case, plus two accepted cases: a read-only array and a scalar
   reduction. `src` is still clean.
8. **(minor, tests) Missing labels for `sun_occultation = false`, `light_time = false` and
   clear weather; no test that the figures carry the labels. Fixed, with one sub-item
   refuted.**
   - Occultation: as in 1. Clear weather: as in 9.
   - Figures: `test_end_to_end.py::test_cli_run_outputs_plot_and_reproducibility` captures
     the figure text before saving and asserts that every effect appears on `power_stages.png`
     and `composite.png`. `test_characterize.py::test_characterization_records_and_labels_the_idealised_settings`
     does the same for the characterize figures. This found a real defect: the new line
     wrapping split "Sun-avoidance" at the hyphen, so `break_on_hyphens=False` is now set.
   - **Refuted: `models.light_time = false` is not an upper bound.** It evaluates the geometry
     with the site at the emission epoch. That shifts the site by |v_site| τ (~1.5e4 km
     orbital per au of range), which moves elevation and occultation timing with either sign,
     and it does not raise the captured power (pointing is modelled separately). It is
     therefore reported as a named approximation instead (see 12), not as an idealised label.
9. **(major, spec) Clear sky for the whole year not labelled; also
   `min_pointing_efficiency = None`, `max_input_power_W = 1e12` and the characterize figures.
   Fixed, partly refuted.**
   Confirmed: weather is one state per run (`atmosphere.py`, `np.full(el.shape, not
   hard_outage)`), so the MVP availability 0.42209 has no cloud outage. Changes:
   - New label `models.weather_scenario = clear` (scenario atmosphere): "permanently clear
     sky: no cloud outage at any epoch (availability is a weather upper bound)". The
     `atmosphere = none` label now also says "no weather outage".
   - New label `transmitter.min_pointing_efficiency = None`: no pointing-availability
     threshold.
   - `characterization.json` has `idealised_settings`, and the power/thermal and capture
     figures name them (`plotting.idealised_note`).
   - The README headline row says "0.42209 (sky clear all year: a weather upper bound)", the
     microwave row "no weather outage; ITU-R attenuation exceeded 1 % of the time applied at
     every epoch", and E1 says the same.
   - The matrix §13.4 row is updated.

   **Refuted: `receiver.max_input_power_W = 1e12`.** It is a finite configured limit, not a
   switched-off one. Whether it binds is a run outcome, already reported as
   `receiver_saturation_time_fraction`. Any threshold above which a limit counts as
   "inactive" would be arbitrary. `docs/model_assumptions.md` lists it as not labelled, with
   this reason.
10. **(minor, spec) §8.4 "technology scenario" marked satisfied without a scenario. Partly
    fixed, partly deferred.** Confirmed. The row is now "partial → deferred (M8)": the curves
    versus distance exist, but no PV technology scenario is shipped. A second technology
    needs sourced cell data (reference efficiency, temperature coefficient, operating limit)
    that the project does not have, and a made-up one would be an unlabelled assumption. Its
    place is the M8 sensitivity studies. Matrix counts updated (172 / 24 / 14 / 5).
11. **(minor, spec) `read_summary` does not check `summary_schema`; the "frozen" wording.
    Fixed.**
    - `RunSummary` validator: `summary_schema` must be in `READABLE_SUMMARY_SCHEMAS`
      (`helioswarm-summary/1`, `/2`), and `/2` requires `orbital_diagnostics`. Tested by the
      layout-1 test in 6 (unknown layout and `/2` without diagnostics both rejected).
    - The record is now described as "attributes cannot be reassigned; dict-valued fields
      are a shallow freeze, treat as read-only" in the docstring and the matrix §4 row. They
      were not converted to `MappingProxyType` because pydantic's JSON dump of a mapping proxy
      would need a custom serializer, for no gain in a record that is only read.
12. **(minor, spec) Radial SRP on a fixed-normal collector gives no run-time notice and no
    comparison path. Fixed.**
    - New `config.approximations_in_force` (`NamedApproximation`: path, value, validity,
      comparison path). It covers fixed-normal + SRP in the L1 model and
      `models.light_time = false`.
    - It is written as a WARNING to `logs/run.log` (stage `config`), to `manifest.json`
      (new key `approximations`, empty for the MVP) and to `report.md` ("Named approximations
      in force").
    - Comparison path available now: rerun with `orbit.include_solar_radiation_pressure:
      false`, which bounds the SRP contribution. Attitude-dependent SRP (L3) stays in M8.
    - Tests: `test_config.py::test_fixed_normal_with_srp_is_a_named_approximation` and
      `test_end_to_end.py::test_fixed_normal_srp_approximation_is_logged_and_in_the_manifest`.
13. **(minor, spec) README first-null "width". Fixed** (same as 2).
14. **(minor, spec) Matrix "https clone" and the `cluster_env.sh` "--all-extras" advice
    contradict the README. Fixed.** The matrix §14.4 row now says "SSH clone (private
    repository) + branch". The comment in `scripts/cluster_env.sh` gives the README's
    `uv sync --locked --extra dev --extra xcheck --extra itur --extra jax` and the reason
    not to use `--all-extras`. The required exports are unchanged.
15. **(minor, spec) `validate-config A B --show-provenance` exits 1 instead of 2. Fixed.**
    New `EXIT_USAGE = 2`. `test_cli.py::test_validate_config_accepts_several_files` asserts 2.
16. **(minor, spec) The characterize thermal test takes the PV temperature from the module
    output. Fixed.**
    - `test_power_thermal_curves_against_closed_forms` now closes the M4 PV-plate balance at
      the returned T: `α A S − P_PV = (ε_f + ε_b) σ F A (T⁴ − T_space⁴)` to 1e-10 of `P_abs`
      (solver bound 1e-12 with CODATA σ; exact-SI σ differs by 3e-11).
    - New `test_power_thermal_curves_with_an_undersized_fixed_radiator`: a 100 m² fixed
      radiator. Checks the closed-form radiator temperature (> T_lim), PV within its limit,
      `thermal_feasible` False, and margin = min(PV margin, T_lim − T_rad) < 0.

Matrix line references were re-resolved after the edits (16 shifted, none unresolved; all test
references exist).

### Gates after the review

```text
$ uv run ruff check .          -> All checks passed!
$ uv run ruff format --check . -> 96 files already formatted
$ uv run mypy src              -> Success: no issues found in 33 source files
$ uv run pytest -q -rs         -> 1059 passed, 1 skipped in 224.69s (0:03:44)
SKIPPED [1] tests/unit/test_capture_batch.py:139: JAX_PLATFORMS=cpu: no GPU requested (run by slurm/gpu_bench.sbatch)
```

There are 7 new tests: `test_config.py` +1, `test_l1_propagation.py` +1, `test_metrics.py`
+2, `test_characterize.py` +2 and `test_end_to_end.py` +1. Existing tests gained assertions
(idealised sets, figure text, manifest `approximations`, CLI exit 2, layout-1 collation).

The committed characterisation copies in `figures/m6b/` were regenerated so that they carry the
labels (`helioswarm characterize configs/mvp_single_satellite.yaml --out
$HELIOSWARM_RUNS_DIR/m6b_review_characterize`, 6 s interactive). The 1 au values are unchanged:
P_PV 3.6706 MW, T_PV 302.8 K, A_rad 438.5 m². No other committed output changed. All 63
existing `summary.json` files under `$HELIOSWARM_RUNS_DIR` (48 layout 1, 15 layout 2) still
pass `read_summary`; the occultation fractions only became nullable. Their `report.md` files predate the new
labels, and `helioswarm plot RUN_DIR` regenerates them.
