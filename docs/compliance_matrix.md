# Compliance matrix: M0–M7b (+ M8a) against the requirements PDF

**M7b update (GEO baseline, PDF objective O5).** New rows: §1 O5, §2 "GEO is a comparison
baseline", §15 "minimum publishable result: comparison against a GEO baseline under
harmonised assumptions", §16 "Comparative baseline" and the §19 NASA SBSP reference,
**satisfied (M7b)** with the evidence in the section "M7b: GEO baseline and harmonised
comparison" below (results in `docs/milestones/M7b.md`), except the §16 harmonisation row,
which is **partial**: lifetime and cost are not modelled for either architecture (→ M8a; the
M8a economics module, merged with M7b, has not yet been run on the comparison outputs), and
delivered power is harmonised as a system boundary, not as an equal quantity (M7b review).

**M8a update (ERA5 site weather, parameterised economics).** §10.2 "site-specific cloud data
later", §12.5 economics, §14.1 external weather-data provenance, §14.2 NetCDF for gridded
weather and §17 "economic overclaim" / "atmosphere over-simplified" are updated below with
**satisfied (M8a)** and the evidence (`docs/milestones/M8a.md`).

**M7 update (constellation).** The rows deferred to M7 (§4 layout `constellation_demo.yaml` /
`scheduler.py`, §4 SciPy.optimize for the scheduler, §6.4 several sites to the scheduler, §10.4
several transmitters, §11.2–§11.4, §13.1 scheduler row, §15 M7 exit, §17 receiver double
counting) are now **satisfied (M7)** with the evidence given in each row; the remaining
deferrals are M8/M9 items. The counts in the summary paragraph below are those of the M6b audit.

Audit of the whole M0–M6 code base (milestone M6b) against every requirement and acceptance
item of `gpt-insights/Heliocentric_Dyson_Swarm_Coding_Agent_Implementation_Plan.pdf` that falls
in M0–M6 scope, plus the out-of-scope items with the milestone they belong to. Evidence is a
source location (`file:line` of the definition, resolved when this file was written) or a test
(`file::test`); "validate-models" refers to the generated block of
`docs/validation_report.md` (37 checks, 37 passed, Slurm job 60127176).

Status vocabulary:

- **satisfied**: implemented and tested before M6b.
- **satisfied (M6b)**: missing or partial at the start of the audit, fixed in M6b (code, test and
  documentation listed in `docs/milestones/M6b.md`).
- **deferred (Mx)**: outside M0–M6 by the PDF's own milestone table (§15) or explicitly a
  "later" item in the PDF; the reason is given. Nothing in M0–M6 scope is left deferred
  without a reason.
- **N/A**: the requirement is conditional and its condition does not occur (reason given).

Summary of the audit (counts are rows of the tables below): **215 items**; **172 satisfied**
before M6b; **24 satisfied after an M6b fix** (including the fixes of the M6b review);
**14 deferred** to M7–M9 or to the PDF's "later" items (10 outright, 4 partially implemented with the remainder deferred), each with
its reason; **5 N/A** with reason; **0 unexplained gaps**. The M6b fixes, by area: §3
`SatelliteState` attitude/normal vector; §3/§14.1 `RunManifest` units and simulated window;
§4 layout `configs/microwave_1au_sweep.yaml` (and its Slurm run); §4 typed summary record and
machine-checked docstring/global-state rules; §5 `validate-config` with several files (found by
the fresh clone); §7.3 orbital diagnostics in `summary.json`, the report and the time series;
§8.4 and §9.3/§9.5 response curves (`helioswarm characterize`); §13.4 idealised-bound labels
in figures and reports; §14.1 ITU-R map provenance; §14.4 README (status with test counts,
headline results, clone instructions); §17 named approximations reported at run time (M6b
review); §19 fresh-clone reproduction.

## §3 Required state/result objects (PDF p.4, minimum fields)

| Object | Minimum fields (PDF) | Implementation | Status |
|---|---|---|---|
| EphemerisState | epoch, frame, origin, position_m[3], velocity_m_s[3], source_kernel_ids | `src/helioswarm/ephemeris.py:117` `EphemerisState`: `epoch_tdb_s`, `frame`, `origin`, `position_m`, `velocity_m_s`, `source_kernel_ids`, `time_scale`; J2000/SSB enforced in `__post_init__`; `tests/unit/test_ephemeris.py::test_ephemeris_state_contract`, `tests/unit/test_ephemeris.py::test_spice_states_carry_provenance_and_contract` | satisfied |
| SatelliteState | epoch, position_m[3], velocity_m_s[3], mass_kg, attitude/normal vector, force-model flags | `src/helioswarm/propagate.py:226` `SatelliteState`: `state` (epoch/position/velocity, J2000/SSB enforced), `force_model` (bodies + SRP flags), `mass_kg` (`None` = not specified; only A/m enters the dynamics), `attitude` (`sun_tracking`/`fixed_normal`) and **`collector_normal_unit`** (J2000 unit normal, set by `src/helioswarm/propagate.py:774` `satellite_initial_state`); `tests/unit/test_l1_propagation.py::test_satellite_initial_state_closed_form_with_analytic_sun` (normal = −r̂ closed form, fixed normal, validation), `tests/unit/test_l1_propagation.py::test_initial_state_from_mvp_config_in_j2000_ssb` | satisfied (M6b): before the audit the attitude was a fixed label `sun_facing` with no normal vector and ignored `collector.incidence_mode` |
| AccessResult | visible, range_m, elevation_rad, occulted_by, light_time_s, transmit_direction_unit | `src/helioswarm/visibility.py:683` `AccessResult` (all six, plus t_emit/t_arrival, range rate, pointing rate, aberration, per-body flags); `tests/unit/test_visibility.py::test_elevation_and_azimuth_of_placed_satellites`, `tests/unit/test_visibility.py::test_beam_through_the_earth_is_flagged` | satisfied |
| SolarPowerResult | flux_W_m2, incidence_cosine, illuminated, absorbed_W, pv_electric_W, pv_heat_W | `src/helioswarm/solar.py:325` `SolarPowerResult` (all six, plus reflected, PV temperature/efficiency/clip flag); `tests/unit/test_solar.py::test_result_reports_efficiency_and_clip_flag`, `tests/unit/test_solar.py::test_solar_result_is_immutable` | satisfied |
| BeamResult | wavelength_m, aperture_m, range_m, beam_model, capture_fraction, pointing_loss, field diagnostics | `src/helioswarm/beam.py:294` `BeamResult` (all, `diagnostics: BeamFieldDiagnostics`, Fresnel number, percentiles, `scalar_diffraction=True`, `coherent_combining=False`); `tests/unit/test_beam.py::test_beam_result_contract` | satisfied |
| ThermalResult | waste_heat_W, radiator_area_m2, equilibrium_temperature_K, feasible, margin | `src/helioswarm/thermal.py:645` `ThermalResult` (`margin_K` = binding margin; `feasible = converged & margin ≥ 0`); `tests/unit/test_thermal.py::test_feasible_flag_and_binding_margin_agree` | satisfied |
| LinkBudgetResult | power_at_transmitter_W, power_at_receiver_W, atmospheric_factor, rx_electric_W, grid_W | `src/helioswarm/link_budget.py:150` `LinkBudgetResult` (stage arrays and the PDF names as properties `power_at_transmitter_W` = P_dc_to_tx, `power_at_receiver_W` = P_captured, `atmospheric_factor`, `rx_electric_W`); invariants on construction; `tests/unit/test_link_budget.py::test_stage_chain_matches_hand_computation` | satisfied |
| RunManifest | config_hash, git_commit, package_versions, kernel hashes, seed, frame, units, timestamps | `src/helioswarm/manifest.py:186` `RunManifest`: `config_hash`, `git` (commit + dirty), `packages`, `kernels` (SHA-256), `seed`, `frame`, **`units`** (`src/helioswarm/manifest.py:57` `UNITS_NOTE`), `wall_clock_start/end_utc`, **`simulation_window`** (`src/helioswarm/manifest.py:163` `SimulationWindow`); `tests/integration/test_end_to_end.py::test_cli_run_outputs_plot_and_reproducibility` | satisfied (M6b): `units` was only a phrase inside the frame note and the simulated start/end epochs were not in the manifest |

## §3 Architecture, boundary and conventions

| Requirement | Evidence | Status |
|---|---|---|
| Chain of independently testable models, typed inputs, structured results with units and diagnostics | one module per stage (orbit, propagate, visibility, solar, thermal, aperture/optical/microwave/capture, atmosphere, link_budget, system); frozen result dataclasses above; per-module unit tests | satisfied |
| Power boundary: incident sunlight on the collector → electrical power accepted by the grid; every loss exactly once | `src/helioswarm/thermal.py:873` `energy_ledger` (incident = reflected + PV heat + radiator + P_aperture), `src/helioswarm/link_budget.py:310` `evaluate_link_budget` (one owner per η, no aperture efficiency); `tests/unit/test_thermal.py::test_energy_ledger_closes`, `tests/unit/test_link_budget.py::test_capture_and_atmosphere_are_applied_once_without_aperture_factor`, `tests/unit/test_link_budget.py::test_onboard_stages_have_a_single_owner`, `tests/unit/test_beam.py::test_wavefront_factor_applied_exactly_once`; e2e `tests/integration/test_end_to_end.py::test_unavailable_epochs_deliver_zero_and_energy_accounting` | satisfied |
| SI internally; au/degrees only at I/O | config fields carry unit suffixes (`_au`, `_deg`) converted in properties, e.g. `src/helioswarm/config.py:448` `OrbitConfig`; `tests/unit/test_orbit.py::test_elements_from_config_converts_every_angle_and_au`, `tests/unit/test_ground.py::test_site_from_config_converts_degrees`; outputs SI (`src/helioswarm/io.py:112` `timeseries_table` metadata `units`) | satisfied |
| Documented inertial frame and origin; never subtract positions in different frames/origins | `src/helioswarm/frames.py:78` `CartesianState`, `src/helioswarm/frames.py:274` `require_same_frame_and_origin`, `src/helioswarm/ephemeris.py:454` `shift_state_origin`; `tests/unit/test_frames.py::test_require_same_frame_and_origin`, `tests/unit/test_visibility.py::test_access_requires_j2000_ssb_satellite_states`, `tests/unit/test_visibility.py::test_orbit_drift_requires_matching_frames_and_epochs` | satisfied |
| SPICE as the authoritative planetary source; record kernel names, versions, checksums | `src/helioswarm/ephemeris.py:229` `SpiceEphemeris`, `src/helioswarm/kernels.py:244` `KernelSet` (SHA-256 + NAIF MD5 verify), manifest `kernels`; `tests/unit/test_kernels.py::test_verify_checks_size_sha256_and_md5`, `tests/unit/test_kernels.py::test_real_kernel_set_verifies_and_loads` | satisfied |
| Time explicit with scale; ET and UTC never mixed implicitly | `src/helioswarm/timescale.py:51` `Epoch`, `src/helioswarm/timescale.py:71` `TimeGrid` (scale `TDB` enforced), `src/helioswarm/timescale.py:116` `utc_to_tdb_s` (only `str2et`, UTC label explicit); `tests/unit/test_timescale.py::test_conversion_is_labelled_utc_independent_of_spice_default_system`, `tests/unit/test_l1_propagation.py::test_orbit_epoch_and_grid_start_are_kept_apart` | satisfied |
| Earth-fixed receiver coordinates separate from the inertial state; explicit Earth-orientation transform | `src/helioswarm/ground.py:178` `GroundSite` (Earth-fixed only), `src/helioswarm/ground.py:386` `SiteTrajectory` (inertial), `src/helioswarm/ground.py:298` `SpiceEarthOrientation`; `tests/unit/test_ground.py::test_iau_earth_rotation_equals_the_pck_closed_form` | satisfied |

## §2 Fixed decisions and non-goals (constraints on M0–M6)

| Requirement | Evidence | Status |
|---|---|---|
| Python, physics as ordinary tested functions (no notebooks) | `src/helioswarm/`; no notebooks in the repository | satisfied |
| One spacecraft and one receiver before a constellation | `src/helioswarm/system.py:502` `run_single_satellite` refuses several sites (`tests/integration/test_end_to_end.py::test_more_than_one_site_is_rejected`) | satisfied |
| Optical and microwave both modelled as alternatives | `configs/mvp_single_satellite.yaml`, `configs/mvp_single_satellite_microwave.yaml`; `tests/integration/test_end_to_end.py::test_microwave_config_runs_with_itu_r_provenance` | satisfied |
| Earth delivery (P_grid) is the key metric | `summary.json` `p_grid_W`, delivered energy; report headline | satisfied |
| No coherent aperture across independent transmitters | `BeamResult.coherent_combining` is always `False` (`tests/unit/test_beam.py::test_beam_result_contract`) | satisfied |
| Optical feasibility not inferred from central intensity: finite capture and pointing losses | `src/helioswarm/capture.py:528` `capture_statistics`; `UPPER_BOUND_NOTE` on figures | satisfied |
| Sun not modelled as a point source for passive concentration | no passive concentration is modelled (active PV conversion only) | N/A |

## M7b: GEO baseline and harmonised comparison (PDF §1 O5, §2, §15, §16, §19)

| Requirement | Evidence | Status |
|---|---|---|
| §1 O5 Harmonised comparison against a GEO baseline; identify conditions under which heliocentric deployment helps or hurts | GEO simulated in-model through the identical chain (`src/helioswarm/geocentric.py`, `orbit.elements_center = EARTH`, geostationary preset; `system.run_single_satellite` unchanged apart from the orbit stage); `configs/comparison/` (shared technology blocks via `extends`), `helioswarm compare` (`src/helioswarm/compare.py`); Slurm sweeps over aperture × receiver radius for both architectures and both technologies (`docs/milestones/M7b.md` A, D) | satisfied (M7b) |
| §2 A GEO system is a comparison baseline, not the target architecture | GEO enters only `configs/comparison/` and the `compare` command; the heliocentric configurations, hashes and results are unchanged (`tests/unit/test_config_validity.py::test_mvp_config_hash_is_pinned`, `tests/unit/test_config_m7b.py::test_mvp_canonical_json_is_the_m7_text`, `tests/integration/test_comparison_configs.py::test_heliocentric_mvp_configs_carry_none_of_the_m7b_keys`) | satisfied (M7b) |
| GEO orbit physics: period = IAU_EARTH spin period; sub-satellite point fixed; documented GEO radius from GM_earth and the IAU rate | `tests/integration/test_geo_baseline.py::test_geostationary_period_equals_the_iau_earth_spin_period`, `::test_sub_satellite_point_stays_fixed_over_ten_days` (1e-6° over 10 days; three longitudes) | satisfied (M7b) |
| GEO eclipse seasons with penumbra (conical shadow) against the analytic geometry | `tests/integration/test_geo_baseline.py::test_geo_eclipse_durations_match_the_analytic_shadow_geometry` (both equinoxes, mid-season, out of season; 15 s); season boundaries and yearly umbra/penumbra fractions over 2030 `::test_geo_eclipse_seasons_over_2030_match_the_analytic_geometry` (slow); run-level `tests/integration/test_geo_comparison_run.py::test_geo_link_through_the_chain_at_equinox` | satisfied (M7b) |
| Elevation ~90° and range r_GEO − R_E at the sub-satellite site | `tests/integration/test_geo_baseline.py::test_geo_elevation_and_range_at_the_sub_satellite_site` (elevation 90° − ωτ to 5e-8 rad; range = L/(1 − u·v/c) to 1 m); off-nadir site `::test_geo_elevation_and_range_from_an_off_nadir_site` (el = atan2(cos γ − R/r, sin γ), slant range to 1 m); configured (non-preset) geocentric elements `::test_configured_geocentric_elements_follow_kepler_about_the_earth` | satisfied (M7b) |
| Omitted perturbations named with magnitudes; station-keeping as the modelled assumption | `docs/model_assumptions.md` "M7b" (lunisolar 0.84°/yr, J2 522 m, SRP ±365 km / ±0.99° at the comparison satellite's A/m ≈ 0.6 m²/kg); every GEO manifest/report names "L0 + station-keeping" (`config.approximations_in_force`, `tests/unit/test_config_m7b.py::test_geocentric_orbits_name_the_station_keeping_approximation`); closed-form J2/SRP tests `tests/unit/test_geocentric.py`; numerical comparison path `geocentric.propagate_geocentric_point_masses`, `tests/integration/test_geo_baseline.py::test_lunisolar_inclination_drift_matches_the_averaged_torque` | satisfied (M7b) |
| Fresnel regime at the GEO range: far-field validity check triggers; Fresnel-Hankel used explicitly | `tests/integration/test_geo_comparison_run.py::test_fresnel_regime_selection_at_the_geo_range` (N_F = 2.79 optical refused by `airy_far_field`; microwave N_F = 0.228, FH = far field to 1e-3, measured 1.1e-4); heliocentric range FH = far field `::test_far_field_and_fresnel_hankel_agree_at_the_heliocentric_range` (≤ 1e-7 relative measured); `configs/comparison/geo_*.yaml` `models.beam: fresnel_hankel` | satisfied (M7b) |
| §15 Minimum publishable result: transparent comparison against a GEO baseline under harmonised assumptions | comparison table + figure generated from the saved runs (`helioswarm compare`, `docs/milestones/M7b.md` A); harmonised-assumption table with provenance labels | satisfied (M7b) |
| §16 Harmonise delivered power, PV efficiency, transmitter efficiency, receiver conversion, lifetime, collector technology, thermal treatment, mass/cost boundary before comparing | shared blocks `configs/comparison/_shared.yaml`, `_optical.yaml`, `_microwave.yaml`; leaf-by-leaf check `tests/integration/test_comparison_configs.py::test_resolved_configs_differ_only_in_orbit_and_geometry`; `compare.load_comparison` refuses unharmonised cases (`::test_load_comparison_refuses_an_unharmonised_case`); delivered power harmonised as the same `P_grid` system boundary (one satellite, one site, same collector area), not as equal delivered power (704.9 kW vs 8.76 kW optical); mass per delivered kW is the normalised metric; mass proxy identical formula and values, exclusions stated. **Lifetime/degradation and cost are not modelled** for either (one-year windows; M8a economics) | partial (lifetime, cost → M8a: `src/helioswarm/economics.py` and `configs/economics/geo_nominal.yaml` are merged, the comparison runs have not yet been costed; M7b review) |
| §16 No claim of heliocentric superiority unless the same system boundary supports it | `docs/milestones/M7b.md` A states the result as computed (GEO delivers orders of magnitude more per satellite under these assumptions) and the conditions | satisfied (M7b) |
| §16/§19 NASA 2024 SBSP assessment as context only, not transplanted | citation verified from the PDF (`docs/data_provenance.md` M7b); quoted values kept in a separate context paragraph of `docs/milestones/M7b.md`, absent from every configuration and table | satisfied (M7b) |
| Shared assumptions literally identical (include/extends mechanism) | `config.read_config_mapping`/`config_sources`/`merge_config_mappings`; `tests/unit/test_config_m7b.py::test_extends_reproduces_the_flat_configuration`, `::test_extends_errors` | satisfied (M7b) |

## §4 Software stack and repository layout

| Requirement | Evidence | Status |
|---|---|---|
| Python ≥ 3.11, uv, lock file; record runtime versions in every manifest | `pyproject.toml` (`requires-python`), `uv.lock`; manifest `python_version`, `packages` (`src/helioswarm/manifest.py:111` `package_versions`); CI on 3.11 and 3.12 | satisfied |
| NumPy/SciPy; Astropy for checks; SpiceyPy; REBOUND cross-check; HCIPy; NumPy/SciPy microwave model; ITU-R; pandas/pyarrow/matplotlib; pytest/ruff/mypy/pre-commit/CI | `pyproject.toml` dependencies and extras (`xcheck`: rebound, hcipy; `itur`; `jax`/`gpu`); `.pre-commit-config.yaml`; `.github/workflows/ci.yml` | satisfied |
| REBOUNDx | no additional forces are integrated in REBOUND (it is a cross-check of the gravitational model only) | N/A |
| SciPy.optimize / pymoo | `scipy.optimize.linear_sum_assignment` and `milp` used by the M7 scheduler (`src/helioswarm/scheduler.py`); the `opt` extra (pymoo 0.6.2, SALib 1.6.0) is installed and locked for M8; optimisation itself is M8 | partial: scheduler satisfied (M7); optimisation deferred (M8) |
| JAX/Numba only after profiling; CPU reference kept | profiling in `docs/milestones/M6.md` D; `src/helioswarm/capture_batch.py:285` `airy_capture_batch` optional, numpy default; `tests/unit/test_capture_batch.py::test_jax_backend_equals_numpy_batch` | satisfied |
| Layout: README, pyproject, uv.lock, .gitignore, `src/helioswarm/{__init__,constants,config,ephemeris,frames,orbit,srp,ground,visibility,solar,thermal,aperture,optical,microwave,atmosphere,link_budget,system,metrics,plotting,cli}.py`, `tests/{unit,integration,regression}`, `experiments/README.md`, `data/kernels`, `runs/.gitkeep`, `figures/`, `docs/{model_assumptions,validation_report,data_provenance}.md` | all present (`tests/regression/` holds no files: regression pins live beside the unit tests, e.g. `tests/unit/test_atmosphere.py::test_itu_r_production_revisions_regression_pin`) | satisfied |
| `configs/mvp_single_satellite.yaml`, `optical_1au_sweep.yaml`, `microwave_1au_sweep.yaml` | present; **`microwave_1au_sweep.yaml` added in M6b** (5 radii × 3 apertures; validated by `test_shipped_sweep_expands_to_valid_points[microwave_1au_sweep.yaml]`) | satisfied (M6b) |
| `configs/constellation_demo.yaml`, `scheduler.py` | present (M7): `configs/constellation_demo.yaml` (+ `constellation_demo_n1000.yaml`, `constellation_demo_sweep.yaml`, `constellation_n1_mvp.yaml`), `src/helioswarm/scheduler.py`; `tests/unit/test_constellation_config.py::test_shipped_constellation_configs_are_valid_and_n1_matches_the_mvp` | satisfied (M7) |
| `optimization.yaml`, `optimization.py`, `paper/` | not created (no stubs for later milestones; `tests/unit/test_engineering_rules.py::test_no_stubs_for_later_milestones`) | deferred (M8, M9) |
| `experiments/notebooks/`, `data/external/` | no notebook and no cached external dataset exists (ITU-R maps live inside the `itur` wheel, see `docs/data_provenance.md`) | N/A |

## §4 Engineering rules

| Rule | Evidence | Status |
|---|---|---|
| Production logic in `src/` | all physics in `src/helioswarm`; `scripts/` only wrap the CLI/benchmarks | satisfied |
| Every public model function: type hints, docstrings, units, reference to the governing equation | `mypy --strict` on `src` (hints); **docstring presence machine-checked for every public module/class/function/method** (`tests/unit/test_engineering_rules.py::test_every_public_definition_has_a_docstring`, discriminating `tests/unit/test_engineering_rules.py::test_docstring_check_is_discriminating`); units and references reviewed by hand (unit-suffixed names; docstrings cite PDF sections, Goodman, Born & Wolf, IERS, IAU) | satisfied (M6b): four `RadialBeamPattern` protocol members had no docstring; the rule had no automated check |
| Dataclasses or validated pydantic models for configuration and results; no unstructured dicts through the code base | config: pydantic, frozen, `extra="forbid"` (`src/helioswarm/config.py:1544` `RunConfig`); results: frozen dataclasses; **summary.json is now the typed pydantic record `src/helioswarm/metrics.py:423` `RunSummary`** (unknown keys rejected, attributes not reassignable; dict-valued fields are a shallow freeze; layout checked: `helioswarm-summary/1` still read, `/2` requires the §7.3 diagnostics; `tests/unit/test_metrics.py::test_layout1_summaries_written_before_m6b_are_still_read`), produced by `src/helioswarm/metrics.py:580` `summarize_timeseries`, read and validated by `src/helioswarm/metrics.py:481` `read_summary` (report, plots, sweep collation); `tests/unit/test_metrics.py::test_summary_record_is_typed_and_round_trips` | satisfied (M6b): the ~80-field summary was a nested `dict` passed to the report, figures and sweep (M6 review finding 16, deferred there) |
| No global mutable state; providers, configs and Generators passed explicitly | `EphemerisProvider`/`EarthOrientation` protocols injected; `rng` argument of `src/helioswarm/system.py:502` `run_single_satellite`; **machine check** `tests/unit/test_engineering_rules.py::test_no_global_mutable_state` (+ `tests/unit/test_engineering_rules.py::test_mutable_state_check_is_discriminating`); two module-level dicts made read-only (`MappingProxyType`) in M6b | satisfied (M6b, check added) |
| Deterministic physics separate from stochastic sampling | pointing jitter enters as a deterministic Rice expectation/percentiles (`src/helioswarm/capture.py:528` `capture_statistics`); the run's Generator is recorded as unused (`rng_state_unchanged`) | satisfied |
| One configuration file defines every physical parameter | `src/helioswarm/config.py:1544` `RunConfig`; constants overridable in `constants:`; `tests/unit/test_constants.py::test_constants_are_not_hard_coded_elsewhere` | satisfied |

## §5 Configuration contract and CLI

| Requirement | Evidence | Status |
|---|---|---|
| Config validation before physics; invalid units, missing values, contradictions fail early and readably | `src/helioswarm/config.py:1544` `RunConfig` validators; `tests/unit/test_cli.py::test_invalid_config_exits_one_with_field_named`, `tests/unit/test_cli.py::test_contradictory_config_exits_one`; `tests/unit/test_config.py`, `test_config_validity.py` | satisfied |
| Every section of the PDF example schema | `configs/mvp_single_satellite.yaml` (PDF §5 values); `tests/integration/test_shipped_configs.py::test_shipped_config_validates_and_round_trips` | satisfied |
| Validate parameter ranges and document every default | range validators per field; `validate-config --show-provenance` lists every leaf with unit and provenance (`tests/unit/test_cli.py::test_show_provenance_lists_every_leaf`); `docs/model_assumptions.md` "Parameter definitions" | satisfied |
| Schema versioning | `schema_version` required (`src/helioswarm/config.py:53` `SCHEMA_VERSION`); output layouts versioned (`helioswarm-summary/2`, `-timeseries/2`, `-run-manifest/2` after M6b) | satisfied |
| `helioswarm validate-config CONFIG` | `tests/unit/test_cli.py::test_valid_config_exits_zero`, `tests/unit/test_cli.py::test_validate_config_accepts_sweep_files`; **several files per call** (`tests/unit/test_cli.py::test_validate_config_accepts_several_files`) | satisfied (M6b: the README's `helioswarm validate-config configs/*.yaml` failed with a usage error in the fresh clone; the CLI accepted one file) |
| `helioswarm run CONFIG --out DIR` | `tests/integration/test_end_to_end.py::test_cli_run_outputs_plot_and_reproducibility` | satisfied |
| `helioswarm plot RUN_DIR` | same test (figures/report deleted, regenerated with the model patched to raise) | satisfied |
| `helioswarm sweep SWEEP --out DIR` | `tests/integration/test_end_to_end.py::test_cli_sweep_with_array_split_and_collation` | satisfied |
| `helioswarm validate-models --report docs/validation_report.md` | `tests/unit/test_validation_report.py::test_cli_error_paths`, `tests/unit/test_validation_report.py::test_every_registered_check_runs_under_pytest` | satisfied |
| `helioswarm optimize configs/optimization.yaml --out ...` | not created (no stub) | deferred (M8) |
| Run output `config_resolved.yaml` (all defaults) | `src/helioswarm/io.py:341` `write_run`; CLI run test | satisfied |
| `manifest.json`: commit, dependency versions, kernel ids/checksums, time scale, frame, seed, timestamp | `src/helioswarm/manifest.py:260` `build_manifest`; CLI run test asserts commit, dirty flag, packages, kernels + SHA-256, seed, time scale, frame, units, simulation window | satisfied |
| `timeseries.parquet`: states, access flags, power stages, thermal | `src/helioswarm/io.py:112` `timeseries_table` (79 columns incl. J2000/SSB states, flags, every §10.1 stage, thermal, beam, reason codes; `transmit_dir_*` added in M6b) | satisfied |
| `summary.json`: mean/median/p05/p95, capacity factor, availability, outage durations, efficiencies | `src/helioswarm/metrics.py:423` `RunSummary`; `tests/unit/test_metrics.py::test_summary_energy_availability_capacity_factor`, `tests/unit/test_metrics.py::test_outage_statistics_hand_computed` | satisfied |
| `figures/` from saved results | `src/helioswarm/plotting.py:231` `make_figures` reads only the saved files; CLI run test | satisfied |
| `logs/`: warnings, integrator diagnostics, failed/invalid steps | `src/helioswarm/io.py:311` `write_run_log`; CLI run test asserts the aliasing warning and timings | satisfied |

## §6 Ephemerides, propagation, SRP, access (WP1)

| Requirement | Evidence | Status |
|---|---|---|
| §6.1.1 Load LSK, SPK, PCK; record files and checksums | `src/helioswarm/kernels.py:244` `KernelSet`, `data/kernels/kernels.json`; `tests/unit/test_kernels.py::test_project_manifest_lists_the_planned_kernels_in_load_order` | satisfied |
| §6.1.2 Sun, Earth, planets in one frame/origin, SI | `src/helioswarm/ephemeris.py:229` `SpiceEphemeris` (J2000/SSB, m, m/s); `tests/unit/test_ephemeris.py::test_spk_segment_chain_rule` | satisfied |
| §6.1.3 Initial state from Sun-relative classical elements, transformed to the inertial frame/origin | `src/helioswarm/propagate.py:774` `satellite_initial_state`; `tests/unit/test_l1_propagation.py::test_satellite_initial_state_closed_form_with_analytic_sun`, `tests/unit/test_l1_propagation.py::test_initial_state_from_mvp_config_in_j2000_ssb` | satisfied |
| §6.1.4 Round-trip state↔elements; SPICE Earth–Sun range vs trusted reference | `tests/unit/test_orbit.py::test_round_trip_elements_state_elements_random_grid`, `tests/unit/test_orbit.py::test_round_trip_state_elements_state_random_grid`; `tests/unit/test_ephemeris.py::test_earth_sun_state_matches_jpl_horizons` (5.3 cm vs JPL Horizons DE441), `tests/unit/test_ephemeris.py::test_earth_sun_state_agrees_with_erfa_epv00` | satisfied |
| §6.2 L0 analytic Kepler | `src/helioswarm/orbit.py:674` `propagate_kepler`, `src/helioswarm/orbit.py:380` `solve_kepler`; `tests/unit/test_propagation.py`, `test_kepler.py` | satisfied |
| §6.2 L1 test particle with SPICE ephemerides; barycentric sum of direct accelerations, or heliocentric with the indirect term, never mixed | `src/helioswarm/propagate.py:495` `propagate_l1` (heliocentric with consistent indirect terms, M2 review); `tests/unit/test_l1_propagation.py::test_heliocentric_acceleration_is_barycentric_minus_the_suns_modelled_acceleration`, `tests/unit/test_l1_propagation.py::test_full_model_heliocentric_vs_barycentric_and_body_subsets` | satisfied |
| §6.2 L2 REBOUND N-body from SPICE states | `tests/integration/test_rebound_crosscheck.py::test_full_nbody_rebound_matches_l1_within_ephemeris_model_differences`, `tests/integration/test_rebound_crosscheck.py::test_two_body_rebound_matches_l1` | satisfied |
| §6.2 L3 attitude-dependent area and other non-gravitational forces | radial SRP only (the PDF's first implementation); L3 is "sensitivity and robustness studies" | deferred (M8) |
| §6.3 Configurable Cr and A/m; simple radial SRP | `orbit.cr`, `orbit.area_to_mass_m2_kg`; `src/helioswarm/srp.py:103` `srp_acceleration_m_s2`; `tests/unit/test_srp.py::test_magnitude_at_1au_and_direction_away_from_an_offset_sun` | satisfied |
| §6.3 Do not silently omit SRP; report the A/m range where SRP materially changes the orbit | SRP on in the MVP; `src/helioswarm/srp.py:160` `srp_material_area_to_mass_m2_kg`, `docs/model_assumptions.md` "SRP materiality"; per-run SRP effect `src/helioswarm/propagate.py:683` `assess_l1_run`; `tests/unit/test_srp.py::test_along_track_drift_estimate_closed_form_and_materiality_inverse`, `tests/unit/test_l1_propagation.py::test_run_assessment_reports_step_convergence_and_srp_effect` | satisfied |
| §6.3 Poynting–Robertson drag optional | not modelled; size stated in `docs/model_assumptions.md` (SRP section) | N/A (optional; named) |
| §6.4 Range, line-of-sight unit vector, elevation | `src/helioswarm/visibility.py:775` `compute_access`; `tests/unit/test_visibility.py::test_elevation_and_azimuth_of_placed_satellites` | satisfied |
| §6.4 Earth blockage by ray/ellipsoid intersection | `src/helioswarm/visibility.py:335` `segment_intersects_ellipsoid`; `tests/unit/test_visibility.py::test_segment_ellipsoid_test_agrees_with_spice_surfpt`, `tests/unit/test_visibility.py::test_earth_blocking_flips_at_the_geodetic_horizon` | satisfied |
| §6.4 Sun blocking the beam with configurable limb margin | `models.sun_occultation_limb_margin_deg`; `tests/unit/test_visibility.py::test_sun_blocking_flips_at_the_limb_plus_margin` | satisfied |
| §6.4 Receiver minimum elevation and Sun-avoidance angle | `receiver.min_elevation_deg`, `receiver.sun_avoidance_deg`; `tests/unit/test_visibility.py::test_min_elevation_flag`, `tests/unit/test_visibility.py::test_sun_separation_and_avoidance` | satisfied |
| §6.4 Several sites: access independently, then valid links to the scheduler | access per satellite × site in stage A (`src/helioswarm/constellation.py` `evaluate_links`), feasible links to the scheduler (`finish_constellation`); `tests/integration/test_constellation.py::test_capacity_limits_clip_and_schedule_incoming_power`, `::test_optimal_policy_never_delivers_less_than_greedy_and_breakdown_is_complete` (3 sites) | satisfied (M7) |

## §7 Light-time targeting, ground geometry, orbital outputs

| Requirement | Evidence | Status |
|---|---|---|
| §7.1 Retarded-time fixed point, convergence limit and diagnostic, robust fallback | `src/helioswarm/visibility.py:548` `solve_light_time` (fixed point, `light_time_max_iterations`, brentq fallback, residuals); `tests/unit/test_visibility.py::test_moving_site_light_time_matches_the_quadratic`, `tests/unit/test_visibility.py::test_fast_receding_site_uses_the_bracketed_fallback` | satisfied |
| §7.1 Store emission and arrival epochs; aim along the converged direction; re-evaluate occultation and pointing there | `AccessResult.t_emit_tdb_s`/`t_arrival_tdb_s`; `tests/integration/test_access_mvp.py::test_retarded_targeting_and_point_ahead_match_spice_spkcpt` | satisfied |
| §7.1 State the omission of relativistic light-time corrections | `docs/model_assumptions.md` M3 "Neglected (stated, PDF §7.1)" | satisfied |
| §7.2 Geodetic config; documented ellipsoid; explicit Earth orientation at the arrival epoch; not inertially stationary | `src/helioswarm/ground.py:114` `geodetic_to_earth_fixed_m` (WGS 84), `IAU_EARTH` via SPICE; `tests/unit/test_ground.py::test_geodetic_matches_spice_georec_and_astropy_below_1_mm`, `tests/unit/test_ground.py::test_iau_earth_versus_astropy_itrs_to_gcrs_at_past_epochs` | satisfied |
| §7.2 Tests: site position, up vector, elevation, daily rotation | `tests/unit/test_ground.py::test_published_station_meades_ranch_reset`, `tests/unit/test_ground.py::test_up_vector_is_the_ellipsoid_normal`, `tests/unit/test_ground.py::test_site_returns_after_one_rotation_period_and_turns_per_solar_day` | satisfied |
| §7.3 Earth range | `timeseries.parquet` `range_m`; **summary `orbital_diagnostics.earth_range_m`** | satisfied (M6b: the statistics were not in summary.json) |
| §7.3 Access fraction | summary `orbital_diagnostics.access_fraction` (and `diurnal_aliasing.access_fraction`) | satisfied |
| §7.3 Occultation fraction (Earth or Sun) | **summary `orbital_diagnostics.occultation_fraction` and per-body fractions**; `tests/unit/test_metrics.py::test_orbital_diagnostics_hand_computed` | satisfied (M6b: only the overlapping reason flags were reported) |
| §7.3 Pointing angle: required inertial beam direction and its time derivative | **`timeseries.parquet` `transmit_dir_{x,y,z}`** (J2000 unit vector) and `pointing_rate_rad_s`; summary first/last direction and rate statistics | satisfied (M6b: the inertial direction was not saved) |
| §7.3 Light time min/mean/max | **summary `orbital_diagnostics.light_time`**; e2e cross-check against the in-memory `AccessSummary` (`tests/integration/test_end_to_end.py::test_summary_is_recomputable_from_saved_timeseries`) | satisfied (M6b) |
| §7.3 Range rate | `range_rate_m_s` column; summary statistics | satisfied (M6b for the summary) |
| §7.3 Orbit drift vs the two-body reference | `orbit_drift_vs_l0_m` column (`src/helioswarm/visibility.py:1085` `orbit_drift_m`); summary max/final | satisfied (M6b for the summary) |
| §7.4 Circular orbit keeps radius/energy within integrator tolerance | `tests/unit/test_l1_propagation.py::test_circular_orbit_radius_is_preserved_within_integrator_tolerance`; validate-models | satisfied |
| §7.4 Period matches Kepler III | `tests/unit/test_propagation.py::test_kepler_third_law_period_from_integration` | satisfied |
| §7.4 Earth–Sun distance agrees with SPICE/JPL | `tests/unit/test_ephemeris.py::test_earth_sun_state_matches_jpl_horizons` | satisfied |
| §7.4 Synthetic geometry gives analytic light time | `tests/unit/test_visibility.py::test_static_geometry_light_time_is_distance_over_c`, `tests/unit/test_visibility.py::test_moving_site_light_time_matches_the_quadratic` | satisfied |
| §7.4 One sidereal rotation per sidereal day | `tests/unit/test_ground.py::test_site_returns_after_one_rotation_period_and_turns_per_solar_day` | satisfied |
| §7.4 Beam segment through Earth or Sun flagged | `tests/unit/test_visibility.py::test_beam_through_the_earth_is_flagged`, `tests/unit/test_visibility.py::test_superior_conjunction_is_sun_blocked` | satisfied |

## §8 Solar collection and thermal feasibility (WP2)

| Requirement | Evidence | Status |
|---|---|---|
| §8.1 S(r) = S_1AU (AU/r)², configurable S_1AU (1361 W/m²) | `src/helioswarm/solar.py:235` `solar_flux_W_m2`, `constants.solar_constant_1au_W_m2`; `tests/unit/test_solar.py::test_flux_at_1au_equals_configured_constant_exactly` | satisfied |
| §8.1 P_abs, P_PV, Q_PV = P_abs − P_PV (reflected light leaves) | `src/helioswarm/solar.py:325` `SolarPowerResult`; `tests/unit/test_solar.py::test_absorbed_and_reflected_split_of_incident_light`, `tests/unit/test_thermal.py::test_reflected_sunlight_is_not_waste_heat` | satisfied |
| §8.1 Track incidence, occultation (eclipse) and attitude; Sun-tracking vs fixed attitude as separate scenarios | `collector.incidence_mode`; `src/helioswarm/visibility.py:1012` `satellite_illumination`; `tests/unit/test_solar.py::test_fixed_normal_cosine_and_back_side_gives_zero`, `tests/unit/test_thermal.py::test_trajectory_path_fixed_normal_and_umbra` | satisfied |
| §8.2 Explicit configurable η_PV(T) law, labelled local linear approximation, clipped to [η_min, η_max] | `src/helioswarm/solar.py:79` `PvEfficiencyLaw`; `tests/unit/test_solar.py::test_efficiency_linear_inside_and_clipped_outside`, `tests/unit/test_solar.py::test_efficiency_never_outside_bounds_or_above_one` | satisfied |
| §8.3 Q_waste = Q_PV + Q_tx + Q_pe + Q_other; Q_rad = εσA_eff(T⁴ − T_space⁴); A_eff with defined geometry | `src/helioswarm/thermal.py:718` `radiator_thermal`, `src/helioswarm/thermal.py:605` `radiator_required_area_m2`; `radiating_sides` defined as faces of a thin flat plate; `tests/unit/test_thermal.py::test_waste_heat_is_exactly_the_onboard_losses_plus_other`, `tests/unit/test_thermal.py::test_required_area_closed_form_and_inverse_scalings` | satisfied |
| §8.3 One energy balance; reflected sunlight not waste heat | `src/helioswarm/thermal.py:873` `energy_ledger`; `tests/unit/test_thermal.py::test_energy_ledger_from_hand_computed_terms`, `tests/unit/test_thermal.py::test_ledger_detects_a_broken_balance` | satisfied |
| §8.3 Solved temperature: bracketed root solver; report whether a solution exists below the limit | `src/helioswarm/thermal.py:384` `solve_pv_temperature` (brentq, convergence and uniqueness flags); `tests/unit/test_thermal.py::test_brentq_failure_is_reported_not_hidden`, `tests/unit/test_thermal.py::test_non_unique_root_is_flagged_not_hidden` | satisfied |
| §8.4 Solar flux and collector power vs heliocentric distance | `src/helioswarm/thermal.py:1047` `power_thermal_vs_distance` (M4); **saved curves `helioswarm characterize` → `power_thermal_vs_distance.parquet` + figure** (`src/helioswarm/characterize.py:102` `power_thermal_table`); `tests/unit/test_characterize.py::test_power_thermal_curves_against_closed_forms`, `tests/unit/test_characterize.py::test_cli_characterize_writes_tables_metadata_and_figures` | satisfied (M6b: the function existed, but no shipped command produced or saved the curves) |
| §8.4 PV efficiency and temperature vs distance and technology scenario | versus distance: same table (`pv_efficiency`, `pv_temperature_K`); `characterize` runs per configuration, so a technology scenario is one collector configuration | partial → deferred (M8): the distance curves exist, but no PV technology scenario is shipped. Both shipped configurations use the PDF §5 example collector, and a second technology (reference efficiency, temperature coefficient, operating limit) needs sourced cell data, which this project does not yet have; inventing one would be an unlabelled assumption. The scenario comparison belongs to the M8 sensitivity studies (M6b review) |
| §8.4 Waste heat and radiator area vs distance and transmitter operating point | same table, operating points `--electrical-to-beam-efficiency`; test checks Q_waste(η₁) − Q_waste(η₂) = P_dc (η₂ − η₁) η_o | satisfied (M6b) |
| §8.4 Net electrical power to the transmitter after power conditioning | `dc_to_tx_W` (table and time series) | satisfied |
| §8.4 Thermal feasibility flag and margin | `thermal_feasible`, `thermal_margin_K` (table, time series, summary) | satisfied |
| §8.5 1 au normal incidence → configured irradiance | `tests/unit/test_solar.py::test_flux_at_1au_equals_configured_constant_exactly`; validate-models | satisfied |
| §8.5 Power ∝ 1/r² at fixed area and efficiency | `tests/unit/test_solar.py::test_collected_power_scales_as_inverse_square_at_fixed_area_and_efficiency` | satisfied |
| §8.5 Zero incidence or eclipse → zero power | `tests/unit/test_solar.py::test_zero_incidence_and_eclipse_give_zero_collected_power` | satisfied |
| §8.5 Thermal solver closes the energy balance | `tests/unit/test_thermal.py::test_energy_ledger_closes`, `tests/unit/test_thermal.py::test_pv_balance_satisfied_to_tolerance_and_inside_bracket` | satisfied |
| §8.5 Radiator area ∝ 1/ε and 1/T⁴ | `tests/unit/test_thermal.py::test_required_area_closed_form_and_inverse_scalings` | satisfied |

## §9 Beam physics and diffraction (WP3)

| Requirement | Evidence | Status |
|---|---|---|
| §9.1 First null 1.22 λ/D, radius 1.22 λR/D; Fresnel number D²/(λR) | `src/helioswarm/optical.py:62` `airy_first_null_radius_m`, `src/helioswarm/aperture.py:172` `fresnel_number`; `tests/unit/test_optical.py::test_first_null_radius_scaling`, `tests/unit/test_aperture.py::test_fresnel_number_definition` | satisfied |
| §9.1 Choose the propagation model by Fresnel number; never silently use Fraunhofer outside validity | `src/helioswarm/aperture.py:193` `check_far_field_validity` (hard error), `src/helioswarm/aperture.py:670` `FresnelHankelPattern`; `tests/unit/test_aperture.py::test_far_field_validity_check_is_a_hard_error`, `tests/unit/test_optical.py::test_airy_outside_far_field_raises`, `tests/unit/test_beam.py::test_far_field_model_outside_validity_raises` | satisfied |
| §9.2 Normalised Airy pattern; Gaussian beam as a separate model | `src/helioswarm/optical.py:99` `AiryPattern`, `src/helioswarm/optical.py:182` `GaussianBeamPattern`; `tests/unit/test_optical.py::test_airy_pattern_integrates_to_one`, `tests/unit/test_optical.py::test_gaussian_beam_power_and_encircled_energy` | satisfied |
| §9.2 2-D capture for a finite receiver with off-axis pointing error | `src/helioswarm/capture.py:184` `capture_fraction`; `tests/unit/test_capture.py::test_offset_capture_converges_to_cartesian_grid` | satisfied |
| §9.2 Taper and wavefront error as configurable extensions | `transmitter.aperture_illumination`, `edge_taper_dB`, `rms_wavefront_error_m`; `tests/unit/test_aperture.py::test_taper_efficiency_matches_quadrature`, `tests/unit/test_aperture.py::test_ruze_and_wavefront_efficiency` | satisfied |
| §9.2 HCIPy/POPPY cross-check | `tests/integration/test_hcipy_crosscheck.py::test_far_field_profile_matches_hcipy`, `tests/integration/test_hcipy_crosscheck.py::test_airy_encircled_energy_matches_hcipy` | satisfied |
| §9.2 Record that the result is scalar diffraction | `BeamResult.scalar_diffraction`; `UPPER_BOUND_NOTE` on beam figures and report | satisfied |
| §9.3 Circular aperture / array-factor model with stated illumination and element pattern | `src/helioswarm/microwave.py:100` `square_lattice_array_factor`, `src/helioswarm/microwave.py:146` `lattice_main_lobe_fraction`; `tests/unit/test_microwave.py::test_array_factor_peak_nulls_and_grating_lobes`, `tests/unit/test_microwave.py::test_element_power_pattern` | satisfied |
| §9.3 Finite capture, sidelobes, pointing loss and phase-error sensitivity | capture and pointing as optical; grating-lobe factor; Ruze; **capture vs RMS phase error saved by `characterize`** (`src/helioswarm/characterize.py:216` `capture_wavefront_table`; `tests/unit/test_characterize.py::test_capture_wavefront_curve_applies_the_ruze_factor_once`) | satisfied (M6b for the saved sensitivity curve) |
| §9.3 Filled aperture vs sparse phased array as different geometries | lattice optional (`array_element_spacing_m`); `tests/unit/test_beam.py::test_sparse_lattice_main_lobe_factor` | satisfied |
| §9.3 Independent RF transmitters are not one coherent array | `coherent_combining=False` | satisfied |
| §9.4 η_capture = ∫_receiver I dA / ∫_plane I dA; centred Airy vs encircled energy at the first dark ring | `tests/unit/test_capture.py::test_centred_capture_equals_closed_form_encircled_energy`, `tests/unit/test_optical.py::test_encircled_energy_at_first_dark_ring` (0.83778) | satisfied |
| §9.4 Offset receivers by adaptive quadrature or converged 2-D grid | exact 1-D arc integral + grid cross-check; `tests/unit/test_capture.py::test_offset_capture_converges_to_cartesian_grid`, `tests/unit/test_capture.py::test_capture_quadrature_is_discriminating`; validate-models nested-quad check | satisfied |
| §9.5 Pointing as isotropic Gaussian with configurable sigma; bias distinct from jitter | `pointing_sigma_rad`, `pointing_bias_rad`; Rice law `src/helioswarm/capture.py:426` `rice_offset_pdf_m_inv`; `tests/unit/test_capture.py::test_bias_reduces_capture`, `tests/unit/test_beam.py::test_pointing_bias_wiring_gaussian_beam_closed_form` | satisfied |
| §9.5 Report capture versus pointing sigma and receiver radius | **`capture_vs_pointing.parquet` + figure** (`src/helioswarm/characterize.py:167` `capture_pointing_table`); `tests/unit/test_characterize.py::test_capture_curves_centred_airy_and_monotonicity` | satisfied (M6b: only per-run pointing efficiency and the receiver-radius sweep existed) |
| §9.5 Coherent arrays: phase errors degrade the array factor | Ruze factor on the main lobe of the (incoherently independent) array; capture vs phase error curve | satisfied (M6b, curve) |
| §9.5 Pointing specification is an input assumption | provenance `assumed` for both pointing fields | satisfied |

## §10 Link budget, atmosphere, receiver (WP3 continued)

| Requirement | Evidence | Status |
|---|---|---|
| §10.1 Single auditable stage chain; one owner per efficiency; no extra aperture efficiency | `src/helioswarm/link_budget.py:310` `evaluate_link_budget`, `src/helioswarm/power_chain.py:122` `onboard_power_chain`; `tests/unit/test_link_budget.py::test_stage_chain_matches_hand_computation`, `tests/unit/test_link_budget.py::test_onboard_stages_have_a_single_owner` | satisfied |
| §10.2 Optical clear/cloudy scenario factors and an availability mask | `src/helioswarm/atmosphere.py:239` `atmospheric_transmission`; cloudy optical = hard outage; `tests/unit/test_atmosphere.py::test_cloudy_optical_is_a_hard_outage`, `tests/unit/test_atmosphere.py::test_optical_scenario_transmittance_scales_with_airmass` | satisfied |
| §10.2 Site-specific cloud data later; cloud obstruction as a hard optical outage | `models.weather_source = era5` + `weather` section (`src/helioswarm/config.py` `WeatherConfig`); ERA5 hourly total cloud cover at each site (`src/helioswarm/era5.py`: request builder, downloader with provenance, cache reader; `src/helioswarm/weather.py`: TDB→UTC, year mapping, threshold/Bernoulli outage, climatology and joint statistics); wired into `system.atmosphere_at_access` for single-satellite and constellation runs; tests `tests/unit/test_era5.py`, `tests/unit/test_weather.py`, `tests/integration/test_weather_run.py::test_era5_outage_hits_exactly_the_cloudy_arrival_hours` (independent `et2utc` path), `::test_constellation_stage_a_applies_site_weather`; Slurm re-runs of the MVP and the constellation demo with ERA5 weather of 2014–2023 (`docs/milestones/M8a.md` D). Not a precise transmission from cloud fraction (PDF §10.2 first bullet): `tcc` only decides a hard outage, documented as a named approximation | satisfied (M8a) |
| §10.2 Turbulence only with a specified receiver/wavefront-correction architecture | not modelled; stated on every beam figure (`UPPER_BOUND_NOTE`) | satisfied (correctly omitted, labelled) |
| §10.2 Atmosphere separate from capture and receiver conversion | separate stages in `src/helioswarm/link_budget.py:150` `LinkBudgetResult`; `tests/unit/test_link_budget.py::test_capture_and_atmosphere_are_applied_once_without_aperture_factor`, e2e `tests/integration/test_end_to_end.py::test_atmosphere_stage_follows_the_plane_parallel_law` | satisfied |
| §10.3 ITU-R Earth-space gaseous/rain/cloud attenuation where valid | `src/helioswarm/atmosphere.py:343` `itu_r_slant_path_attenuation_dB`; `tests/unit/test_atmosphere.py::test_itu_r_reproduces_itu_validation_example` | satisfied |
| §10.3 Recommendation number and version in the manifest | manifest `itu_r_versions`; `tests/integration/test_end_to_end.py::test_microwave_config_runs_with_itu_r_provenance` | satisfied |
| §10.3 Polarisation, frequency and elevation assumptions | `atmosphere.itu_r_polarization_tilt_deg`, carrier from `wavelength_m`, `itu_r_min_elevation_deg`; `tests/unit/test_atmosphere.py::test_itu_r_polarization_tilt_from_config` | satisfied |
| §10.3 No terrestrial attenuation on the vacuum segment | attenuation only on the slant path at the receiver | satisfied |
| §10.4 Receiver area, conversion efficiency, max input power, thermal rejection | `src/helioswarm/link_budget.py:89` `ReceiverModel`; `tests/unit/test_link_budget.py::test_receiver_input_limit_clips_captured_power`, `tests/unit/test_link_budget.py::test_receiver_heat_rejection_limit` | satisfied |
| §10.4 Max power density and beam abort as explicit constraints | `receiver.max_power_density_W_m2`; `tests/unit/test_link_budget.py::test_power_density_beam_abort` | satisfied |
| §10.4 Link unavailable on pointing, elevation, weather, occultation or receiver state | `src/helioswarm/system.py:120` `Unavailability` bitmask; `tests/integration/test_end_to_end.py::test_weather_pointing_and_power_density_conditions`, `tests/integration/test_end_to_end.py::test_baseline_has_access_occultation_and_consistent_reasons` | satisfied |
| §10.4 Several transmitters: receiver limits, no double counting | limits on the site sum (slots, input, heat, summed density interlock) in `src/helioswarm/scheduler.py`; beam exclusivity; `summary.json` double-counting check (independent recount of P_grid from the assignments and link powers, sum of per-satellite shares, beams = active satellites; network = Σ receivers as a consistency check only); emission-epoch aggregation bounded by the reception-time reconstruction (`summary.json` `reception_time`); `tests/unit/test_scheduler.py::test_receiver_power_limit_clips_the_sum_and_stops_further_beams`, `tests/unit/test_constellation_stage_b.py`, `tests/integration/test_constellation.py::test_summed_density_interlock_is_wired_through_the_run`, `::test_two_rings_with_technology_overrides_sum_the_individual_runs`, `::test_summed_power_density_interlock_is_enforced`, `::test_optimal_equals_brute_force_and_dominates_greedy`, `tests/integration/test_constellation.py::test_cli_constellation_run_outputs_and_determinism` | satisfied (M7) |
| §10.4 Optical and microwave receivers as different technology cases | `receiver.technology` tied to mode (`src/helioswarm/config.py:120` `RECEIVER_TECHNOLOGY_FOR_MODE`); `tests/unit/test_link_budget.py::test_receiver_model_from_config_is_technology_specific`; configs state their separate assumptions | satisfied |
| §10.5 End-to-end efficiency, beam capture, availability, capacity factor (denominator defined), outage statistics, delivered energy, power density | `src/helioswarm/metrics.py:580` `summarize_timeseries`, `src/helioswarm/system.py:400` `reference_link`; `tests/unit/test_metrics.py::test_summary_energy_availability_capacity_factor`, `tests/integration/test_end_to_end.py::test_reference_link_conditions_and_rated_power_independently` | satisfied |

## §11 Integration

| Requirement | Evidence | Status |
|---|---|---|
| §11.1.1 Propagate one satellite on a documented heliocentric orbit | `src/helioswarm/system.py:502` `run_single_satellite` (L1 or L0) | satisfied |
| §11.1.2 Solar flux, PV output, waste heat | M4 stage of the run; time-series columns | satisfied |
| §11.1.3 Transmitter input and emitted beam power | `dc_to_tx_W`, `beam_W`, `aperture_W` | satisfied |
| §11.1.4 Retarded-time geometry and occultation | M3 stage; reason codes | satisfied |
| §11.1.5 Diffraction pattern, pointing offset, capture | M5 stage; beam columns | satisfied |
| §11.1.6 Atmosphere, receiver conversion, grid conditioning | link budget stages | satisfied |
| §11.1.7 Save P_grid(t), all intermediate stages, thermal feasibility, link diagnostics | `timeseries.parquet` | satisfied |
| §11.1.8 One-page run report with the configuration and plots | `src/helioswarm/report.py:80` `report_markdown` (key parameters with provenance, idealised settings, metrics, §7.3 diagnostics, provenance, figures) | satisfied |
| §11.1 Preserve units, energy accounting, time geometry and capture | e2e `tests/integration/test_end_to_end.py::test_unavailable_epochs_deliver_zero_and_energy_accounting`, `tests/integration/test_end_to_end.py::test_collector_area_scales_power_linearly` | satisfied |
| §11.2 Independent satellites with own technology; vectorised; one ring evenly phased then randomized; rings/inclinations/eccentricities; feasible link list; capacity/exclusivity/interlock/duty-cycle scheduling; deterministic tested scheduler | `src/helioswarm/config.py` `ConstellationConfig`/`RingConfig`; `src/helioswarm/constellation.py` `constellation_members`, `evaluate_links`; `src/helioswarm/propagate.py` `propagate_l1_stacked`; `src/helioswarm/scheduler.py`; tests `tests/integration/test_constellation.py::test_even_and_random_phases`, `::test_stacked_l1_equals_per_satellite_l1`, `::test_identical_even_satellites_with_unlimited_receiver_sum_the_individual_runs`, `::test_stacks_are_split_and_tolerances_scaled_by_one_over_sqrt_n`, `::test_duty_cycle_limit_is_wired_through_the_run_and_the_summary`, `::test_numpy_batch_beam_backend_equals_the_reference_on_two_rings`, `tests/unit/test_scheduler.py` (determinism, permutation, brute force, duty cycle incl. exact decimal floors); `scripts/validate_stacked_l1.py` (heterogeneous rings, one year); Slurm runs N = 1/10/100/1000, even vs random, greedy vs optimal (`docs/milestones/M7.md` D) | satisfied (M7) |
| §11.3 Independent vs coherent: incoherent power sum only | `src/helioswarm/constellation.py` stage B; `tests/unit/test_scheduler.py::test_incoherent_sum_adds_powers_not_amplitudes`; `BeamResult.coherent_combining` always False | satisfied (M7) |
| §11.4 P_grid per receiver and network, active satellite/link counts, collector/radiator/aperture/mass totals, power per mass, scheduler reason | `src/helioswarm/constellation_io.py` (`network.parquet`, `receivers.parquet`, `satellites.parquet`, `ConstellationSummary`: network and per-receiver statistics, availability, outages, counts, totals, mass proxy, power per area and mass, unavailability breakdown incl. `RECEIVER_CAPACITY`/`SCHEDULER_NOT_SELECTED`/`DUTY_CYCLE`, unconstrained baseline and capacity-loss fraction); `tests/integration/test_constellation.py::test_cli_constellation_run_outputs_and_determinism`, `::test_duty_cycle_limit_is_wired_through_the_run_and_the_summary` (breakdown vs independent recount), `tests/unit/test_constellation_stage_b.py::test_satellite_primary_reason_is_the_furthest_link_by_first_reason` | satisfied (M7) |

## §12 Optimisation and uncertainty (M8a part)

| Requirement | Evidence | Status |
|---|---|---|
| §12.5 Economics as a separate parameterised module (manufacturing cost/kg, transport, launch/deployment, lifetime, replacement, operations, discount rate); compare scenarios and uncertainty; no single LCOE as a robust prediction | `src/helioswarm/economics.py` (`EconomicsConfig` with `CostParameter` provenance; Tsiolkovsky transport with per-satellite, slot-phase-aware patched-conic or explicit Δv; CAPEX with technology-specific receiver cost (optical value an `assumed` proxy) and optional transmitter capex, replacement, degradation, opex; CRF, LCOE, NPV, EAC; `monte_carlo` with ranges checked against the scenarios); `helioswarm economics RUN_OR_SWEEP_DIR --config ...`; `configs/economics/{optimistic,nominal,conservative,geo_nominal}.yaml` (NASA OTPS 2024 values with PDF pages, others `assumed`); tests `tests/unit/test_economics.py` (CRF vs annuity sum, analytic LCOE, geometric-series degradation, replacement cycles, vis-viva/Kepler checks of the Δv model, zero-energy handling, USD/kWh vs USD/MWh, scenario ordering, seeded MC); results on the M6/M7 outputs in `docs/milestones/M8a.md` | satisfied (M8a) |
| §12.1–12.4 optimisation, Monte Carlo/Sobol of the physics, scenarios | — | deferred (M8) |

## §13 Verification and validation gates

### §13.1 Unit tests (M0–M6 rows)

| Module | Required | Evidence | Status |
|---|---|---|---|
| constants/units | dimensional consistency; AU, solar flux, c, σ, GM | `tests/unit/test_constants.py::test_dimensional_consistency_circular_speed_and_year`, `tests/unit/test_constants.py::test_exact_defining_constants`, `tests/unit/test_constants.py::test_stefan_boltzmann_from_exact_si_constants`, `tests/unit/test_constants.py::test_gm_sun_matches_de440_kernel_literal` | satisfied |
| frames/ephemeris | known frame transforms, consistent origin, kernel coverage, time-scale checks | `tests/unit/test_frames.py::test_ecliptic_rotation_matches_spice_pxform`, `tests/unit/test_ephemeris.py::test_origin_shift_round_trip_and_values`, `tests/unit/test_ephemeris.py::test_coverage_matches_naif_summary_and_out_of_range_is_refused`, `tests/unit/test_timescale.py::test_utc_to_tdb_against_leap_second_table_and_usno_tdb_formula` | satisfied |
| orbit | Kepler period, energy/angular momentum conservation, convergence with step size | `tests/unit/test_propagation.py::test_kepler_third_law_period_from_integration`, `tests/unit/test_propagation.py::test_energy_and_angular_momentum_conserved`, `tests/unit/test_l1_propagation.py::test_error_converges_with_rtol_when_steps_are_uncapped`, `tests/unit/test_l1_propagation.py::test_convergence_study_full_model_one_year` | satisfied |
| SRP | inverse-square, anti-sunward | `tests/unit/test_srp.py::test_inverse_square_scaling`, `tests/unit/test_srp.py::test_magnitude_at_1au_and_direction_away_from_an_offset_sun` | satisfied |
| ground/visibility | station coordinates, Earth rotation, elevation, occultation edges | `tests/unit/test_ground.py::test_published_station_meades_ranch_reset`, `tests/unit/test_ground.py::test_uniform_rotation_matrix_and_rate`, `tests/unit/test_visibility.py::test_elevation_and_azimuth_of_placed_satellites`, `tests/unit/test_visibility.py::test_moon_blocking_edge` | satisfied |
| solar | 1/r², incidence and eclipse limits, efficiency bounds | `tests/unit/test_solar.py::test_flux_inverse_square_scaling`, `tests/unit/test_solar.py::test_zero_incidence_and_eclipse_give_zero_collected_power`, `tests/unit/test_solar.py::test_efficiency_never_outside_bounds_or_above_one` | satisfied |
| thermal | energy closure, radiative-equilibrium limits | `tests/unit/test_thermal.py::test_energy_ledger_closes`, `tests/unit/test_thermal.py::test_non_converting_plate_matches_radiative_equilibrium`, `tests/unit/test_thermal.py::test_zero_absorbed_power_gives_sink_temperature` | satisfied |
| beam | Airy first null, total power, capture convergence, pointing monotonicity | `tests/unit/test_optical.py::test_first_null_radius_scaling`, `tests/unit/test_optical.py::test_airy_pattern_integrates_to_one`, `tests/unit/test_capture.py::test_offset_quadrature_converges`, `tests/unit/test_capture.py::test_expected_capture_non_increasing_with_sigma_airy_baseline` | satisfied |
| link budget | passive stages never increase power; η ∈ [0,1] | `tests/unit/test_link_budget.py::test_random_budgets_satisfy_invariants`, `tests/unit/test_link_budget.py::test_invariant_violation_is_detected_on_construction` | satisfied |
| scheduler | no receiver over-capacity, no double allocation, deterministic | `tests/unit/test_scheduler.py::test_optimal_equals_brute_force_and_dominates_greedy`, `::test_permutation_of_input_order_gives_identical_assignment`, `::test_schedule_timeseries_is_deterministic_bitwise`, `::test_receiver_power_limit_clips_the_sum_and_stops_further_beams` | satisfied (M7) |
| optimization | — | — | deferred (M8) |

### §13.2 Integration tests

| Requirement | Evidence | Status |
|---|---|---|
| 1-AU baseline from config to saved time series and summary, no manual steps | `tests/integration/test_end_to_end.py::test_cli_run_outputs_plot_and_reproducibility`; fresh-clone reproduction (below) | satisfied |
| Impossible beam → near-zero capture | `tests/integration/test_end_to_end.py::test_impossible_beam_gives_near_zero_capture` | satisfied |
| Occulted/unavailable link → zero power | `tests/integration/test_end_to_end.py::test_unavailable_epochs_deliver_zero_and_energy_accounting` | satisfied |
| Receiver capacity limit clips | single transmitter: `tests/integration/test_end_to_end.py::test_receiver_capacity_limit_clips`, `::test_site_capacity_overrides_apply_to_the_single_satellite_run`; several transmitters (M7): `tests/integration/test_constellation.py::test_capacity_limits_clip_and_schedule_incoming_power`, `::test_summed_density_interlock_is_wired_through_the_run`, `tests/unit/test_scheduler.py::test_receiver_power_limit_clips_the_sum_and_stops_further_beams`, `::test_summed_power_density_interlock_is_enforced` | satisfied |
| Larger collector area → linearly more PV power | `tests/integration/test_end_to_end.py::test_collector_area_scales_power_linearly` | satisfied |
| Smaller λ or larger D improves the diffraction scale | `tests/integration/test_end_to_end.py::test_smaller_wavelength_or_larger_aperture_improves_capture` | satisfied |
| Larger pointing uncertainty cannot improve centred expected capture | `tests/integration/test_end_to_end.py::test_larger_pointing_sigma_cannot_improve_expected_capture` | satisfied |

### §13.3 Independent checks

| Requirement | Evidence | Status |
|---|---|---|
| Orbits vs REBOUND (and state why any discrepancy exists) | `tests/integration/test_rebound_crosscheck.py::test_full_nbody_rebound_matches_l1_within_ephemeris_model_differences` (1.03 m/yr: REBOUND evolves the planets, L1 prescribes DE440; stated); validate-models | satisfied |
| Planetary positions and Sun–Earth range vs SPICE/JPL | `tests/unit/test_ephemeris.py::test_earth_sun_state_matches_jpl_horizons`, `tests/unit/test_ephemeris.py::test_earth_sun_state_agrees_with_erfa_epv00` | satisfied |
| Airy patterns and encircled energy vs analytic | `tests/unit/test_optical.py::test_airy_encircled_energy_closed_form_vs_quadrature` | satisfied |
| Optical propagation vs HCIPy/POPPY | `tests/integration/test_hcipy_crosscheck.py::test_far_field_profile_matches_hcipy` | satisfied |
| RF attenuation vs ITU-R or a validated implementation | `tests/unit/test_atmosphere.py::test_itu_r_reproduces_itu_validation_example` (P.618-13 with P.676-11 pinned). The production P.676-12 gas term has no validation set in `itur` 0.4.0 and is regression-pinned only (`tests/unit/test_atmosphere.py::test_itu_r_production_revisions_regression_pin`); recorded as open in `docs/validation_report.md` | satisfied for the wrapper; open item stated, not hidden |
| Time-step and spatial-grid convergence for all numerical integrations and capture calculations | L1: `tests/unit/test_l1_propagation.py::test_convergence_study_full_model_one_year`, validate-models `_c_l1_step_convergence`; light time: tolerance tests; capture: `tests/unit/test_capture.py::test_offset_quadrature_converges`, `tests/unit/test_aperture.py::test_hankel_quadrature_converges_under_refinement`, `tests/unit/test_aperture.py::test_far_field_encircled_energy_honours_refinement`; Bessel kernel: `tests/unit/test_capture_batch.py::test_trapezoid_bessel_matches_scipy`; sampling: validate-models diurnal check; lattice main-lobe fraction is a closed-form lobe sum (no grid) | satisfied |

### §13.4 Research integrity

| Rule | Evidence | Status |
|---|---|---|
| Never fabricate results, citations, validation outcomes or capabilities | every number in `docs/` traced to a test, a Slurm job or a cited source; generated validation block; this audit's numbers are from runs listed in `docs/milestones/M6b.md` | satisfied |
| Every figure generated from saved data and states its configuration | `src/helioswarm/plotting.py:231` `make_figures`, `src/helioswarm/plotting.py:311` `make_characterization_figures`: read saved files only, suptitle carries project and config hash; CLI tests delete figures and redraw with the model disabled | satisfied |
| Every parameter labelled measured/sourced/assumed/swept | `json_schema_extra["provenance"]` per field, overrides for swept values; `tests/unit/test_cli.py::test_show_provenance_marks_overrides`, `tests/unit/test_sweep_io.py::test_grid_is_cartesian_in_key_order_with_swept_provenance` | satisfied |
| Idealised upper bounds labelled in figure and manuscript | beam figures: `UPPER_BOUND_NOTE`; **active idealised settings (`src/helioswarm/config.py:1837` `idealised_settings`: lossless chain stages, perfect wavefront/pointing, inactive constraints incl. disabled occultation checks and no pointing-availability threshold, clear sky for the whole window, lossless scenario atmosphere) listed in `report.md` and `characterization.json`, and on the power-stage, composite and `characterize` power/thermal and capture figures; Sun-tracking labelled on the solar figure**; `tests/unit/test_config.py::test_idealised_settings_of_the_mvp_and_after_overrides`, `tests/unit/test_characterize.py::test_characterization_records_and_labels_the_idealised_settings`, `tests/integration/test_end_to_end.py::test_cli_run_outputs_plot_and_reproducibility` (figure text contains every label) | satisfied (M6b: lossless defaults, no Sun avoidance and the Sun-tracking bound were listed in the docs but not on figures/reports; M6b review added the clear-sky availability bound, the remaining lossless stages and inactive constraints) |
| Record negative and failed runs | sweep records failed points (`tests/unit/test_sweep_io.py::test_sweep_dir_and_collation_record_failures`); negative results in `experiments/README.md` E1–E4 and M6 | satisfied |
| Raw inputs, derived outputs and plotting code distinct | `data/kernels` (raw), `$HELIOSWARM_RUNS_DIR` (derived), `plotting.py` | satisfied |

## §14 Reproducibility, data formats, performance, documentation

| Requirement | Evidence | Status |
|---|---|---|
| §14.1 Unique run directory and resolved config snapshot | `src/helioswarm/io.py:79` `prepare_run_dir` (refuses to overwrite), `config_resolved.yaml` | satisfied |
| §14.1 Git hash, Python, dependency versions, OS, CPU/GPU, seed, simulation start/end time | manifest `git`, `python_version`, `packages`, `os`, `cpu_model`, `gpu_devices`, `seed`, **`simulation_window`** | satisfied (M6b for the simulated window) |
| §14.1 Kernel names, versions, URLs, SHA-256; large kernels not committed | `data/kernels/kernels.json`, `.gitignore`, manifest `kernels`; `docs/data_provenance.md` | satisfied |
| §14.1 External weather data: product, retrieval, variables, resolution, licence/attribution | ITU-R digital maps inside `itur` (M6b) and ERA5 (M8a): per-file `*.nc.provenance.json` (`src/helioswarm/era5.py` `Era5FileProvenance`: collection, DOI, citation, variables, area, period, request + SHA-256, CDS request id, retrieval UTC, cdsapi version, file/archive SHA-256, licence, attribution), per-run `weather.json` listed in `manifest.json`; `docs/data_provenance.md` M8a; `tests/unit/test_era5.py::test_real_fixture_provenance_schema_and_checksum`, `tests/integration/test_weather_run.py::test_cli_run_writes_weather_provenance` | satisfied (M6b, M8a) |
| §14.1 Every random operation receives an explicit seeded Generator | `rng = default_rng(project.seed)` in `src/helioswarm/io.py:406` `execute_run`; MC tests use explicit Generators | satisfied |
| §14.2 Parquet for time series (CSV optional), JSON for summaries/manifests, no giant JSON arrays | `timeseries.parquet` (zstd), `summary.json`/`manifest.json` small; sweep table copied as CSV in `figures/m6/` | satisfied |
| §14.2 NetCDF/Zarr and FITS only where natural | gridded ERA5 cloud cover is cached as the CDS NetCDF files (`$HELIOSWARM_WEATHER_DIR/era5/<dataset>/<node>/*.nc`, read with xarray/h5netcdf in `src/helioswarm/era5.py`, M8a); no other gridded data; no FITS | satisfied (M8a) |
| §14.3.1 Profile the vertical slice before optimising | `slurm/profile.sbatch`, `docs/milestones/M6.md` D (orbit 85.6 %) | satisfied |
| §14.3.2 Vectorise time; cache ephemeris samples when valid | vectorised access/beam/link; `src/helioswarm/ephemeris.py:510` `HermiteEphemerisCache` with measured error (`tests/unit/test_ephemeris.py::test_hermite_cache_vs_direct_spice_below_one_metre`) | satisfied |
| §14.3.3 Coarse-to-fine sweeps (analytic beam for screening) | sweeps use the analytic Airy model; finalist re-evaluation belongs to optimisation | partial → deferred (M8) |
| §14.3.4 Parallelise independent scenarios before GPU code | Slurm job arrays (`slurm/sweep_array.sbatch`) | satisfied |
| §14.3.5–6 JAX only after profiling, same reference tests; CPU reference kept | `capture_batch.py` tested against the M5 reference; default numpy | satisfied |
| §14.4 README: installation, quick start, expected outputs, project status | `README.md` (M6b: status table with test counts, headline results with run paths/job ids, SSH clone (private repository) + branch, fresh-clone procedure) | satisfied (M6b: status said "implemented"/"not authorised" without counts; clone only via SSH and default branch) |
| §14.4 model_assumptions.md, data_provenance.md, validation_report.md, experiment log | `docs/`, `experiments/README.md` | satisfied |

## §15 Milestone exit conditions (M0–M6)

| Milestone | Exit condition | Evidence | Status |
|---|---|---|---|
| M0 | clean install; lint/type/test pass; config validates | fresh-clone reproduction (below); gates in `docs/milestones/M6b.md` D | satisfied |
| M1 | Kepler period and invariants pass documented tolerances | `test_propagation.py`, `test_orbit.py`, `test_kepler.py` | satisfied |
| M2 | known ephemeris checks pass; state metadata recorded | Horizons/epv00 tests; `EphemerisState`/`SatelliteState` metadata | satisfied |
| M3 | synthetic geometry and Earth-rotation tests pass | `test_visibility.py`, `test_ground.py` | satisfied |
| M4 | energy closure and limiting cases pass | `test_thermal.py`, `test_solar.py` | satisfied |
| M5 | analytic diffraction and capture tests pass | `test_optical.py`, `test_capture.py`, `test_aperture.py` | satisfied |
| M6 | one config produces P_grid(t), manifest, summary, plots | CLI run test; Slurm runs 60127166/60127172; fresh clone | satisfied |
| M7 | no double counting; deterministic capacity-constrained results | `tests/unit/test_scheduler.py`, `tests/integration/test_constellation.py`; Slurm constellation runs with network metrics (`docs/milestones/M7.md` D) | satisfied (M7) |
| M7b | GEO baseline through the identical chain; harmonised comparison table + figure from saved runs | `docs/milestones/M7b.md`; section "M7b" above | satisfied (M7b) |
| M8a (plan) | ERA5 data downloaded with provenance; weather affects optical results through config; economics gives LCOE/NPV ranges from saved runs | `docs/milestones/M8a.md` (download job 60341911; ERA5 re-runs; economics on the M6/M7 outputs) | satisfied (M8a) |
| M8–M9 | — | not started | deferred (M8–M9) |

## §17 Risks and agent behaviour rules

| Item | Evidence | Status |
|---|---|---|
| Frame/time inconsistency: one contract, explicit conversions, SPICE tests | §3 rows above | satisfied |
| Diffraction outside validity: Fresnel number, Fresnel propagation | §9.1 rows | satisfied |
| Pointing treated as perfect: bias and jitter, capture distributions | percentiles p05/p50/p95 per epoch; characterisation curves | satisfied |
| Coherence assumed: independent powers | `coherent_combining=False` | satisfied |
| Thermal losses omitted: energy balance, radiator feasibility, mass proxy | ledger, feasibility; radiator mass when `thermal.radiator_specific_mass_kg_m2` is set (`tests/unit/test_thermal.py::test_sized_radiator_runs_at_the_limit_and_reports_mass`) | satisfied |
| Atmosphere over-simplified: scenario masks first, site-specific weather later | scenario + ITU-R; ERA5 site weather (M8a) | satisfied (M8a for the site-specific part) |
| Receiver double counting: explicit scheduler and receiver power/thermal limits | `src/helioswarm/scheduler.py`; double-counting check in the constellation `summary.json` | satisfied (M7) |
| Optimisation of a flawed model | — | deferred (M8) |
| Economic overclaim: economics optional and parameterised, sensitivity ranges | `src/helioswarm/economics.py` (separate module, reads saved summaries only; every input with provenance; scenario configs `configs/economics/*.yaml`; seeded Monte Carlo intervals; zero/near-zero energy → LCOE undefined with flag; disclaimer in `economics.json`); `tests/unit/test_economics.py` | satisfied (M8a) |
| Overengineering: one-satellite slice first | M6 slice before constellation | satisfied |
| Inspect existing interfaces; no duplicate implementations | single owners (e.g. Ruze in `aperture.py`, on-board chain in `power_chain.py`); M6b reuses M4/M5 functions in `characterize.py` | satisfied |
| Implementation note per milestone (files, equations, assumptions, tests, issues) | `docs/milestones/M0.md` … `M6b.md` (A–F) | satisfied |
| Never silently replace a model with an approximation (name, validity, comparison path) | e.g. Gaussian beam vs Fresnel–Hankel (`tests/unit/test_optical.py::test_gaussian_beam_agrees_with_fresnel_hankel_for_weak_truncation`), IAU_EARTH vs astropy, plane-parallel vs Kasten–Young, radial SRP on a fixed-normal collector and instantaneous light-time geometry: `src/helioswarm/config.py:1989` `approximations_in_force` writes them with a comparison path to `logs/run.log`, `manifest.json` and `report.md` (`tests/unit/test_config.py::test_fixed_normal_with_srp_is_a_named_approximation`, `tests/integration/test_end_to_end.py::test_fixed_normal_srp_approximation_is_logged_and_in_the_manifest`) | satisfied (M6b review: the radial-SRP approximation was named only in the docs) |
| Unavailable dependency: report and propose the smallest alternative; no invented APIs | JAX Bessel probe (M6 deviation); `itur` revision limits documented | satisfied |
| Large data: provenance, licensing, size documented | kernels and ITU-R maps in `docs/data_provenance.md` | satisfied |
| No manuscript conclusions before validated runs | no `paper/` | satisfied |

## §19 References and definition of done

| Item | Evidence | Status |
|---|---|---|
| Cite exact standard revision and software version | ITU-R revisions pinned and recorded per run; package versions per manifest | satisfied |
| Check licence/redistribution before bundling kernels or weather data | NAIF rules cited; ITU map terms recorded as an open item (maps not redistributed) | satisfied |
| Definition of done: clone, locked install, kernels, supplied configs, regenerate figures, inspect assumptions, reproduce metrics without undocumented steps | fresh-clone reproduction below | satisfied (M6b) |

## Fresh-clone reproduction (PDF §19 definition of done)

**Procedure** (2026-11-13, M6b). Everything ran in a throwaway tree on netscratch
(`/n/netscratch/hekstra_lab/Lab/ian_poon/m6b_fresh_clone/`, deleted afterwards) and followed
only the README:

1. `git clone --branch helioswarm-m0-m6 git@github.com:extremefattypunch/Project-Dyson-Sphere.git`
   (commit `218de9b`). The repository is private: an anonymous https clone returns 404, so the
   README's earlier "public, no account needed" wording (written in this audit) was corrected
   before the run. **One step outside the README:** the uncommitted M6b working tree was applied
   to the clone as a patch (`git diff HEAD --binary` plus the new files), because M6b is
   committed after this audit; the manifests therefore record `git.dirty = true`.
2. A clean shell: `env -i HOME=<empty scratch dir> PATH=/usr/bin:/bin bash --noprofile --norc`,
   then `source scripts/cluster_env.sh` and the README's throwaway overrides
   (`UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, `HELIOSWARM_KERNEL_DIR`, `HELIOSWARM_RUNS_DIR` all in
   the scratch tree, so the install came from PyPI into an empty uv cache).
3. `uv sync --locked --extra dev --extra xcheck --extra itur --extra jax` (63 s; environment
   1.2 GB, uv cache 1.2 GB), `uv run helioswarm fetch-kernels` (116 s; all four kernels
   downloaded from NAIF and verified: SHA-256 equal to `data/kernels/kernels.json`, de440s MD5),
   `uv run helioswarm validate-config configs/*.yaml` (all five files valid; MVP config hash
   `3db50d83…ef7e`, identical to the M6 runs).
4. Slurm job **60144119** (test partition, 2 cores, 4 GB; same clean `env -i` shell): both MVP
   runs, figures deleted and redrawn with `helioswarm plot`, `helioswarm characterize`, ruff,
   ruff format, mypy and the full test suite.

**Outcome.**

| Check | Result |
|---|---|
| Optical MVP (`helioswarm run configs/mvp_single_satellite.yaml`, 51 s, 407 MB) | P_grid mean 1751.11 W, median 0, p95 5647.49 W, peak 5936.99 W; availability 0.422085; capacity factor 0.297005 (rated 5895.91 W); delivered energy 5.52609e10 J; end-to-end 1.28663e-4; median capture 6.30584e-3; 366 outages, longest 50400 s: **identical to the M6 run 60127166 to every printed digit** |
| Microwave MVP (56 s, 469 MB) | P_grid mean 0.314505 W, peak 1.07303 W; availability 0.422085; capacity factor 0.295098; 9.92504e6 J; end-to-end 2.31082e-8; median capture 3.87698e-7: **identical to M6 run 60127172** |
| New summary content | `helioswarm-summary/2` with §7.3 diagnostics (light time 559.941 / 574.494 / 588.886 s, occultation fraction 0.500342, access fraction 0.422085, orbit drift vs L0 max 1.21484e7 m); manifest simulation window 2030-01-01T00:00:00 → 2031-01-01T06:00:00.000 UTC, 8767 epochs |
| Figures from saved data | `helioswarm plot` regenerated 6 figures + `report.md` for each run after the figures were deleted; `characterize` wrote 3 tables and 3 figures (copied to `figures/m6b/`) |
| `ruff check .`, `ruff format --check .` | pass |
| `pytest -q -rs` | **1052 passed, 1 skipped** in 200 s (skip = real-GPU test) |
| `mypy src` | **one-off mypy 2.4.1 INTERNAL ERROR** in the job (exit 2, 12 s); the identical command in the same clone afterwards passed with the shared cache and with a private cache, and three concurrent runs from the clone and the main checkout on the shared cache passed. Cause not determined (the job shared `MYPY_CACHE_DIR` with the main checkout). Precaution: `scripts/cluster_env.sh` now gives every checkout its own mypy cache directory (keyed by its absolute path; the required exports are unchanged); with it, `mypy src` in the clone passed |
| README defect found | `helioswarm validate-config configs/*.yaml` (documented since M6) was a usage error: the CLI took one file. Fixed (several files per call; `tests/unit/test_cli.py::test_validate_config_accepts_several_files`) and the reproduction restarted from a new clone |
| Storage hygiene | the throwaway HOME received **0 files**; the real home directory had 54 entries before and after; nothing was written under `/n/home05` outside the repository; the throwaway tree (2.4 GB) was deleted |

Not repeated in the fresh clone (already reproduced twice in M6 on the same physics code, and
long): the 45-point optical sweep and the 3896 s aliasing run; the new 15-point microwave sweep
ran from the main checkout (array 60141210; its 1.0 au / 1 km point reproduces the microwave
MVP exactly, 0.3145 W). `validate-models` was not re-run: M6b changes outputs, not physics, and
its checks run under pytest (`tests/unit/test_validation_report.py`), which passed.

