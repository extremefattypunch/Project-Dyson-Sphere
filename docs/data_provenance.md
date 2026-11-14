# Data provenance

This file records the provenance of every external dataset and of every **sourced** parameter
default (PDF §14.1, §14.4, §13.4). Every parameter is labelled `measured`, `sourced`, `assumed` or
`swept` in the configuration schema (`json_schema_extra["provenance"]`):

- The label in the schema describes the *default* value.
- A run can relabel a value with `provenance_overrides`, for example when a parameter is swept.
- `helioswarm validate-config CONFIG --show-provenance` prints the effective label of every
  resolved value.

## External data

### SPICE kernels (M2)

The kernel set is defined by `data/kernels/kernels.json` (manifest version 1, kernel set id
`helioswarm-de440s-v1`). The files are fetched and verified by `helioswarm fetch-kernels`
(`scripts/fetch_kernels.py`), are git-ignored, and are loaded only through
`helioswarm.kernels.KernelSet.loaded()`, which verifies them first. Retrieval date for all four
files: 2026-11-12T06:54:48Z.

| Kernel | Kind | Source URL | Size [B] | SHA-256 | NAIF MD5 | Server Last-Modified |
|---|---|---|---|---|---|---|
| `naif0012.tls` | LSK | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls | 5 257 | `678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b` | none published | Fri, 15 Jul 2016 00:00:37 GMT |
| `pck00011.tpc` | text PCK | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc | 131 226 | `3dff7b1dbeceaa01f25467767d3fa25816051c85d162d1edf04acb310ee28bb1` | none published | Tue, 27 Dec 2022 16:28:50 GMT |
| `gm_de440.tpc` | text PCK | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc | 12 406 | `924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140` | none published | Wed, 14 Dec 2022 20:00:00 GMT |
| `de440s.bsp` | SPK | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp | 32 726 016 | `c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2` | `3917ee56769db332790c751e2168843d` (matches `spk/planets/aa_checksums.txt`) | Mon, 21 Dec 2020 19:39:26 GMT |

**Checksum policy.** NAIF does not publish SHA-256 checksums.

- **SHA-256 (trust on first use).** The SHA-256 values were computed by helioswarm on first
  retrieval and are enforced from then on. A mismatching file is refused and never
  overwritten.
- **MD5.** NAIF publishes an MD5 only for the planetary SPKs (`aa_checksums.txt`). The de440s
  MD5 matched at retrieval and is enforced as well.
- **Consistency with M1.** The `gm_de440.tpc` SHA-256 equals the value recorded in M1, when
  GM☉ was sourced by hand.

**Contents used.**

- `naif0012.tls`: leap seconds through 2017-01-01 (TAI − UTC = 37 s).
- `gm_de440.tpc`: the `BODYnnn_GM` values of DE440, BVS/NAIF 2022-12-14 (Park et al. 2021,
  AJ 161, 105).
- `de440s.bsp`: covers 1849-12-26 to 2150-01-22 TDB (NAIF `aa_summaries.txt`; checked by a
  test against `spkcov`).
- `pck00011.tpc`: loaded now, used from M3 for the IAU_EARTH frame.

**License.** The NAIF generic kernels are public data produced by NASA/JPL NAIF and are freely
redistributable (NAIF rules, https://naif.jpl.nasa.gov/naif/rules.html). They are nevertheless
not committed, because they are large and regenerable (PDF §14.1).

**Run manifests.** `LoadedKernels.provenance` exposes, for each kernel, its name, kind, path,
URL, size, SHA-256, MD5 and retrieval date. `KernelManifest.sha256` (the hash of
`kernels.json`) identifies the kernel-set definition; it is also the CI cache key. M6 writes
these records into `manifest.json`.

**Configuration pin (M2 review).** The run configuration names its kernel set in
`frames.ephemeris_kernel_set` (`helioswarm-de440s-v1`, the `kernel_set_id` of
`kernels.json`). `propagate.initial_state_from_config` and `propagate_l1_from_config` raise
`KernelSetMismatchError` if the loaded set has another id, so two runs with the same config
hash cannot use different ephemerides or GM values; the SHA-256 values in the manifest pin
the individual files. A new kernel set needs a new `kernel_set_id` and a schema change.

### Reference data used only by tests

| Data | Source / retrieval | Used in |
|---|---|---|
| Earth state relative to the Sun at 6 TDB epochs (1 Jan 2000 to 1 Jan 2050) | JPL Horizons API, `source: DE441`, retrieved 2026-11-12. The exact query URL and the verbatim numbers are in `tests/unit/test_ephemeris.py` (`HORIZONS_QUERY`, `HORIZONS_EARTH_WRT_SUN`). | Earth–Sun state check (de440s vs DE441, < 1 m) |
| ERFA `epv00` (via astropy `solar_system_ephemeris="builtin"`) | astropy 8.0.2 / pyerfa, installed from `uv.lock` | Independent Earth–Sun check (documented maximum errors 11.2 km and 5.0 mm/s) |
| TAI − UTC history | IERS Bulletin C | UTC → TDB test |
| GM⊕ = 3.986004418 × 10¹⁴ m³ s⁻² (TT), L_B = 1.550519768 × 10⁻⁸ | IERS Conventions (2010), Table 1.1 | `gm_de440` Earth GM check |
| EM-barycentre mean elements a = 1.00000261 au, e = 0.01671123 | Standish, *Keplerian Elements for Approximate Positions of the Major Planets*, JPL SSD, Table 1 | Perihelion and aphelion distance check |
| Anomalistic year 365.259636 d | Explanatory Supplement to the Astronomical Almanac | EMB perihelion-interval check |
| IERS-B Earth orientation (UT1 − UTC, polar motion) | `astropy-iers-data` 0.2026.11.9 bundled with astropy 8.0.2 (from `uv.lock`), read offline (`iers.conf.auto_download = False`) | IAU_EARTH vs ITRS → GCRS cross-check (M3) |
| WGS 84 b = 6 356 752.3142 m, e² = 6.69437999014e-3 | NGA.STND.0036 (WGS 84, 2014), table 3.1 | Derived-constant check (M3) |
| Station KG0640 MEADES RANCH RESET: NAD 83(2011) 39°13′26.71220″ N, 98°32′31.74540″ W, h = 573.961 m; X = −734 972.563 m, Y = −4 893 188.492 m, Z = 4 011 982.811 m | U.S. National Geodetic Survey datasheet (datasheet95 v8.12.5.20), https://www.ngs.noaa.gov/cgi-bin/ds_mark.prl?PidBox=KG0640, retrieved 2026-11-12; GRS 80 1/f = 298.257222101 (Moritz, J. Geodesy 74, 2000) | Published-station geodetic → Cartesian check (M3 review) |
| ERA = 2π(0.7790572732640 + 1.00273781191135448 D_UT1) | IERS Conventions (2010), eq. 5.15 | IAU_EARTH rotation-angle decomposition (M3 review) |
| IAU 2006/2000A celestial-to-terrestrial matrix | ERFA `c2t06a` (pyerfa, installed with astropy from `uv.lock`) | IAU_EARTH offset at 2000–2040 with UT1 = UTC (M3 review) |
| Transmission light time and stellar aberration (`XCN`, `XCN+S`) | SPICE `spkcpt` with the project kernel set | End-to-end retarded targeting / point-ahead check (M3 review) |

### ITU-R propagation models via `itur` (M5, optional extra)

| Item | Value |
|---|---|
| Package | `itur` (ITU-Rpy) 0.4.0, I. del Portillo, MIT licence, https://github.com/inigodelportillo/ITU-Rpy; pinned in `uv.lock`, installed with `uv sync --extra itur` (or `--all-extras`) |
| Data | ITU-R digital maps bundled inside the wheel (~170 MB installed in the project environment on holylabs); nothing is downloaded at run time and nothing is written to `$HOME` (checked: no cache paths in the package source; a probe run with a fresh `HOME` created no files) |
| Recommendations used by `models.atmosphere = "itu_r"` | P.618-13, P.676-12, P.840-7, P.453-13, P.835-6, P.836-6, P.837-7, P.838-3, P.839-4, P.1510-1, P.1511-2 (`atmosphere.ITU_R_VERSIONS`; pinned on every call, returned in `AtmosphereResult.recommendation_versions` for the run manifest) |
| Currency | These are the newest revisions `itur` 0.4.0 implements, **not** the newest in force (e.g. P.618-14 and P.676-13 have been published). Re-validation against current revisions is an open item |
| Validation reference | ITU-R Study Group 3 validation examples (CG-3M3J-13-ValEx, cited by `itur` as rev. 5.1, https://www.itu.int/en/ITU-R/study-groups/rsg3/ionotropospheric/CG-3M3J-13-ValEx-Rev5_1.xlsx) **as transcribed** in `itur`'s bundled `test/ITU_validation_test.py` (`ITUR618_13TestCase.test_total_attenuation`); the spreadsheet itself was not downloaded. Reproduced to 1e-5 dB in `tests/unit/test_atmosphere.py` (with the file's P.676-11 / P.1511-1 revisions and τ = 0). The installed test file has validation sets for P.676-9 and P.676-11 only (`grep '^class'`), none for P.676-12: the **production gas component (P.676-12) and P.1511-2 are unvalidated** and only regression-pinned (`test_itu_r_production_revisions_regression_pin`); validation against P.676-12/-13 and P.618-14 is an open item |
| Polarisation tilt | `atmosphere.itu_r_polarization_tilt_deg` (assumed, default 45° = circular), passed as P.618/P.838 τ and returned in `AtmosphereResult.polarization_tilt_deg` for the manifest (review M5) |
| Thread safety | `itur` keeps the selected revisions as library-global state; the wrapper holds a module lock, pins the revisions and restores the previous ones on exit. Direct `itur` calls from other threads are not protected |
| Packaging caveats | `itur` installs a top-level package named `test` into site-packages (the Python standard library's `test` package shadows it; nothing in helioswarm imports it). `itur` emits a spurious P.676 "elevation" warning at exactly 90° (its check uses `mod(el, 90) < 5`); the wrapper records warnings in `AtmosphereResult.notes` instead of suppressing them |

**Gridded data inside `itur` (PDF §14.1 "external weather-data product", M6b audit).** With
`models.atmosphere = "itu_r"` the attenuation is computed from the ITU-R digital maps that the
`itur` 0.4.0 wheel bundles (`itur/data/`, 169 MB measured with `du -sh`; wheel
`itur-0.4.0-py2.py3-none-any.whl` from PyPI, SHA-256 pinned in `uv.lock`). No other weather
data (ERA5, site statistics) is used. Grid spacings below were **measured** from the bundled
latitude grids (`numpy.load`, unique-value spacing) for the map revisions this project pins:

| Recommendation (pinned revision) | Variable used | Bundled file(s) | Grid spacing (lat × lon) | Temporal nature |
|---|---|---|---|---|
| P.837-7 | rainfall rate exceeded for 0.01 % of an average year R₀.₀₁; monthly mean total rainfall | `837/v7_lat_r001.npz`, `837/v7_lat_mt.npz` (+ lon, values) | 0.125° (R₀.₀₁); 0.25° (monthly totals) | long-term climatological statistics (annual exceedance; 12 monthly means) |
| P.839-4 | mean annual 0 °C isotherm height (rain height) | `839/v4_esa0height.npz`, `v4_esalat/lon` | 1.5° | long-term annual mean |
| P.840-7 | reduced cloud liquid water content (exceedance statistics) | `840/v7_*` | 1.125° | long-term exceedance statistics |
| P.836-6 | surface water-vapour density / total columnar water vapour (exceedance statistics) | `836/v6_*` (topography grid `v6_topolat` 0.5°) | 1.125° | long-term exceedance statistics |
| P.453-13 | wet term of the surface refractivity (exceedance statistics) | `453/v13_*` | 0.75° | long-term exceedance statistics |
| P.1510-1 | mean annual surface temperature | `1510/v1_*` | 0.75° | long-term annual (and monthly) means |
| P.1511-2 | topographic height of the Earth's surface | `1511/v2_topo.npz`, `v2_egm2008.npz` | 0.0833° (1/12°) | static |

Retrieval: installed from PyPI by `uv sync --locked --extra itur` (the wheel's hash in
`uv.lock` fixes the exact bytes; nothing is downloaded at run time). Attribution: ITU-R
Recommendations P.453, P.618, P.676, P.836, P.837, P.839, P.840, P.1510, P.1511 (International
Telecommunication Union, Radiocommunication Sector), as packaged by ITU-Rpy (I. del Portillo,
MIT, https://github.com/inigodelportillo/ITU-Rpy). Licence: the wheel's `LICENSE.txt` is the
MIT licence of the ITU-Rpy *code* (copyright 2016 I. del Portillo, MIT); it says nothing about
the ITU digital maps, whose terms are ITU's. This project neither redistributes the maps nor
derived gridded products; checking ITU's terms is an open item before any data product
derived from them is published. The pinned revisions are recorded in every run manifest
(`itu_r_versions`).

M5 adds no other external data. Its reference values in tests are closed forms (Airy,
Lommel, Fresnel on-axis, non-central χ²), the zero j₁,₁ = 3.8317059702 (Abramowitz & Stegun
Table 9.5, equal to `scipy.special.jn_zeros`), Kasten & Young (1989, Appl. Opt. 28, 4735) for
the air-mass comparison, and HCIPy 0.7.1 (from `uv.lock`). The new M5 configuration fields
(`receiver.orientation`, `transmitter.array_element_pattern_exponent`,
`atmosphere.itu_r_exceedance_percent`, `atmosphere.itu_r_min_elevation_deg`, and from the M5
review `atmosphere.itu_r_polarization_tilt_deg`) are **assumed**.

### M6: run outputs, validation references and the optional JAX backend

M6 adds no external data set. Provenance of what it uses or produces:

| Item | Provenance |
|---|---|
| Run outputs (`timeseries.parquet`, `summary.json`, `manifest.json`, figures, reports) | Written to `$HELIOSWARM_RUNS_DIR` on netscratch (90-day purge; regenerable from the configs with `helioswarm run` / `slurm/run.sbatch`). Every run's `manifest.json` records git commit + dirty flag, package versions, kernel SHA-256, ITU-R versions, seed and host. Nothing large is committed. |
| `validate-models` references | Closed forms in `helioswarm.validation`; JPL Horizons DE441 Earth-wrt-Sun positions (6 epochs, retrieved 2026-11-12, query in `tests/unit/test_ephemeris.py::HORIZONS_QUERY`, values copied verbatim into `validation.HORIZONS_EARTH_WRT_SUN_KM`); ITU-R SG3 validation examples via `itur` 0.4.0's test file (as in M5); HCIPy 0.7.1 (xcheck extra); exact SI constants (2019 SI: k_B = 1.380649e-23 J/K, h = 6.62607015e-34 J s) for the independent Stefan–Boltzmann value.; after the M6 review also: IERS Bulletin C leap seconds (TAI−UTC 32 s from 1999, 37 s from 2017), TT−TAI = 32.184 s and the USNO Circular 179 (Kaplan 2005, eq. 2.6) TDB−TT series for the time-scale check; REBOUND IAS15 (xcheck extra) for the orbit cross-check. |
| JAX (optional `jax` / `gpu` extras) | `jax` / `jaxlib` 0.12.0.1 from PyPI (Apache-2.0), pinned in `uv.lock`, Python ≥ 3.12 only. The `gpu` extra adds `jax-cuda12-plugin`/`jax-cuda12-pjrt` and the NVIDIA CUDA 12 runtime wheels (`nvidia-*`, ~4.4 GB; project environment total 6.0 GB on holylabs, checked with `du`; nothing written to `$HOME`). Used only by `helioswarm.capture_batch` and the GPU benchmark (`slurm/gpu_bench.sbatch`). |
| Microwave MVP scenario values | `configs/mvp_single_satellite_microwave.yaml`: every transmitter/receiver value is **assumed** (documented in the file header); 2.45 GHz is the PDF §2 illustrative carrier. |
| Sweep values | `configs/optical_1au_sweep.yaml`: heliocentric radii and receiver radii from the PDF §2 scenario table; aperture diameters 10 m (PDF §5 example), 30 m and 100 m (assumed); relabelled `swept` per point. |
| Microwave sweep values (M6b) | `configs/microwave_1au_sweep.yaml`: heliocentric radii from the PDF §2 scenario table; phased-array diameters 1 km (the microwave MVP), 3 km and 10 km (assumed, deliberately generous); relabelled `swept` per point. |
| Characterisation grids (M6b) | `helioswarm characterize` defaults: 0.3–1.2 au in 0.05 au steps (contains the PDF §2 radii), receiver radii 1/5/10 km (PDF §2) plus the configured one, pointing sigmas 0–1 × λ/D and wavefront errors 0–0.2 λ plus the configured values (assumed grid choices); the capture curves' link range and elevation are explicit inputs written to `characterization.json`. |

### M7: constellation scenarios, link files and optional dependencies

M7 adds no external data set. Provenance of what it uses or produces:

| Item | Provenance |
|---|---|
| Constellation configurations | `configs/constellation_demo.yaml` (N = 10), `configs/constellation_demo_n1000.yaml`, `configs/constellation_demo_sweep.yaml` (N = 1/10/100 × even/random × greedy/optimal), `configs/constellation_n1_mvp.yaml` (the MVP as a one-satellite ring). Every value is **assumed**: the three receiver sites (scenario locations, not real installations), their capacities (8 beam slots; one site with 4 slots and a 200 kW input limit), the ring (1 au, Earth-relative phases over [30°, 330°)), and the mass-proxy specific masses (collector 1 kg/m², radiator 5 kg/m² via `thermal.radiator_specific_mass_kg_m2`, transmitter 50 kg/m² of aperture, bus 500 kg; **not sourced**, a proxy for M8 sensitivity studies). The technology is the MVP's (PDF §5 example). Random phases come from `numpy.random.default_rng(project.seed)` and are reproducible. |
| Earth position for Earth-relative phases | DE440 via the kernel set of M2 (`spkezr`-equivalent provider calls, NAIF 399 relative to the Sun, geometric); no new data. |
| Constellation run outputs | `network.parquet`, `receivers.parquet`, `satellites.parquet`, optional `links/*.parquet`, `summary.json`, `manifest.json`, figures in `$HELIOSWARM_RUNS_DIR` on netscratch (90-day purge; regenerable from the configs). Array-task link files `linkset_<i>_of_<n>.npz` (intermediate stage-A arrays, uncompressed NumPy archives, also on netscratch). Nothing large is committed. |
| `opt` extra (for M8) | `pymoo` 0.6.2 (Apache-2.0; cp312 manylinux wheel) and `SALib` 1.6.0 (MIT; pure Python) from PyPI, pinned `pymoo>=0.6.2,<0.7`, `SALib>=1.6,<1.7` and locked in `uv.lock`; installed from wheels on Python 3.12 (checked with `uv sync --locked --all-extras`). Not imported by M7 code. |
| Scheduler solvers | `scipy.optimize.linear_sum_assignment` and `scipy.optimize.milp` (HiGHS) of SciPy 1.18.1 (`uv.lock`); signatures checked from the installed package. |

### M7b: GEO baseline, harmonised comparison and the NASA context source

M7b adds no data set to the runs. Provenance of what it uses or produces:

| Item | Provenance |
|---|---|
| Comparison configurations | `configs/comparison/_shared.yaml`, `_optical.yaml`, `_microwave.yaml` (shared technology blocks: the PDF §5 example / MVP values and the MVP microwave design, radiator areal mass 5 kg/m² as the M7 demo), the four case files and their sweeps (apertures 10/30/100 m and receiver radii 0.1/1/10 km optical; 0.5/1/2 km and 1/5/10 km microwave). Every value is **assumed** (scenario inputs, not design recommendations) unless the schema marks the default **sourced**; the harmonised-assumption table of each `helioswarm compare` output lists every leaf with its provenance label. The mass-proxy specific masses of `configs/comparison/comparison.yaml` are the M7 demo's (**assumed**, not sourced). |
| Earth GM and rotation | `BODY399_GM` of `gm_de440.tpc` (GM_earth = 3.9860043551e14 m³/s²) and the IAU_EARTH constants of `pck00011.tpc` (`BODY399_PM` = 190.147°, 360.9856235°/day; `BODY399_POLE_RA` rate −0.641°/century; `BODY399_POLE_DEC` rate −0.557°/century), both from the M2 kernel set (SHA-256 pinned); no new kernel. |
| J2 of the perturbation budget | J2 = 1.0826e-3 (EGM2008: Pavlis, N. K., Holmes, S. A., Kenyon, S. C., Factor, J. K., "The development and evaluation of the Earth Gravitational Model 2008 (EGM2008)", J. Geophys. Res. 117, B04406, 2012, doi:10.1029/2011JB008916); used only in the documented estimate of the omitted J2 effect (≈ 522 m on the synchronous radius), never in a run. |
| Bounded flux-limit sensitivity (M7b review) | `configs/comparison/sensitivity/geo_optical_flux_limited.yaml` and its sweep: `receiver.max_power_density_W_m2 = 1.0e4 W/m²` is an **assumed, illustrative** value (about 7 solar constants, chosen between the GEO peaks of the 30 m and 10 m apertures), **not** a sourced laser-PV, eye- or skin-safety limit; the project still has no sourced laser-PV flux limit. Outputs `$HELIOSWARM_RUNS_DIR/comparison/cmp_geo_optical_flux_limited/`. |
| GEO Earth albedo/IR bound (M7b review) | Global-mean Bond albedo 0.3 and outgoing long-wave 240 W m⁻² (Trenberth, Fasullo & Kiehl 2009, BAMS 90, 311, as in M4), used only for the documented order-of-magnitude bound in `docs/model_assumptions.md` M7b; not used by any computation. |
| Comparison outputs | One sweep directory per case under `$HELIOSWARM_RUNS_DIR/comparison/` (netscratch, 90-day purge; regenerable with `slurm/comparison_submit.sh`, which encodes the measured sizing; the cited runs were regenerated after the review from one source tree, recorded as `source_sha256` in every run manifest, `docs/milestones/M7b.md` "Review outcomes") and the `helioswarm compare` report (`comparison_points.*`, `comparison_baseline.*`, `harmonised_assumptions.*`, `comparison.png`, `comparison_manifest.json` with the SHA-256 of every summary read). Copies of the report tables are in `docs/milestones/M7b.md`. |
| NASA 2024 SBSP report (**context only**) | E. Rodgers, E. Gertsen, J. Sotudeh, C. Mullins, A. Hernandez, H. N. Le, P. Smith, N. Joseph, *Space-Based Solar Power*, NASA Office of Technology, Policy, and Strategy, Report ID 20230018600, 11 January 2024; https://www.nasa.gov/wp-content/uploads/2024/01/otps-sbsp-report-final-tagged-approved-1-8-24-tagged-v2.pdf (linked from https://www.nasa.gov/organizations/otps/space-based-solar-power-report/; title, authors, date and report ID read from the PDF's title page on 2026-11-13, file SHA-256 60ecde46…45a4, not stored). Context values quoted in `docs/milestones/M7b.md` (Executive Summary, p. ii): two representative designs normalised to 2 GW delivered to the grid, RD1 "Innovative Heliostat Swarm" generating power 99 % of the year, RD2 "Mature Planar Array" 60 %; solar panel area 11.5 km² (RD1) and 19 km² (RD2); mass 5.9 Mkg (RD1) and 10 Mkg (RD2); systems assessed for operation in 2050. These are the report's design assumptions, **not** transplanted into any configuration, table or figure of this project (PDF §16, §19). Since the merge with M8a, individual cost inputs of the same report (same file SHA-256), not these design values, enter the economics scenarios `configs/economics/*.yaml` with page-level sources and their own provenance labels (section "M8a" below). |

### M8a: ERA5 site weather (external data set) and economics inputs

**ERA5 hourly total cloud cover** (PDF §14.1 "external weather-data product, retrieval date,
variables, spatial/temporal resolution and license/attribution"; §14.2 NetCDF for gridded data):

| Item | Value |
|---|---|
| Product (used) | *ERA5 hourly time-series data on single levels from 1940 to present*, CDS collection `reanalysis-era5-single-levels-timeseries`, DOI [10.24381/1cf1ad76](https://doi.org/10.24381/1cf1ad76) (DataCite: creator C3S, publisher ECMWF, 2025) |
| Product (cross-check, same analyses) | *ERA5 hourly data on single levels from 1940 to present*, `reanalysis-era5-single-levels`, DOI [10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47) (DataCite: creator C3S, 2018); two files (2014 at nodes N23.50 E000.00 and N00.00 E000.00) downloaded by the first bulk job before it was replaced; `scripts/era5_product_crosscheck.py`: identical `tcc` at all 3 × 3 × 8760 values of both |
| Reference paper | Hersbach, H. et al. (2020), "The ERA5 global reanalysis", Q. J. R. Meteorol. Soc. 146(730), 1999–2049, doi:[10.1002/qj.3803](https://doi.org/10.1002/qj.3803) (verified via Crossref, 2026-11-13) |
| Variable | `total_cloud_cover` (`tcc`, ECMWF parameter 164, fraction of the grid box covered by cloud, 0–1, instantaneous analysis) |
| Spatial resolution / area | 0.25° regular lat/lon grid; for each site the 3 × 3 nodes centred on the nearest node (`area = [N, W, S, E]` = node ± 0.25°) |
| Temporal resolution / period | hourly (hh:00 UTC), 2014-01-01 00 UTC – 2023-12-31 23 UTC (87 648 hours; ten complete years of final ERA5, no preliminary ERA5T) |
| Sites | `configs/weather/era5_sites.yaml`: the receiver sites of the shipped run configurations (`example_site` 0 N 0 E; `site_lon000` 23.4 N 0 E; `site_lon120e` 23.4 S 120 E; `site_lon120w` 33 N 120 W) and eight candidate sites (atacama 24.6 S 70.4 W; sonora_arizona 32.0 N 111.6 W; hawaii_summit 19.8 N 155.5 W; aswan 23.9 N 32.9 E; namibia_highland 23.3 S 16.5 E; rub_al_khali 21.0 N 50.0 E; ladakh 32.8 N 79.0 E; south_australia 29.0 S 134.75 E) |
| Retrieval | 2026-11-13 (UTC), Slurm job 60341911, `cdsapi` 0.7.7 / `ecmwf-datastores-client` 0.6.0; one CDS request per site (request ids, request JSON, SHA-256 of the delivered zip and of the extracted NetCDF in each `*.nc.provenance.json`) |
| Volume / location | 12 files × 3.88 MB = 45 MB in `$HELIOSWARM_WEATHER_DIR = /n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/weather/era5/<collection>/<node>/` (netscratch; regenerable with `helioswarm weather download`); not committed |
| Committed fixture | `tests/data/era5/N23.50_E000.00_k1_tcc_2014_m02-02.nc` (74 kB, gridded product, February 2014, 3 × 3 nodes; unmodified CDS output, README with attribution) |
| Licence | CC-BY-4.0 (CDS licence record `cc-by` revision 1, "This dataset is released for use under the CC-BY licence", retrieved 2026-11-13; both collections) |
| Required attribution | "Generated using Copernicus Climate Change Service information [year]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains." ("Contains modified …" for derived data: climatology and run outputs). Source: ECMWF Copernicus Knowledge Base, "How to acknowledge and cite a Climate Data Store (CDS) catalogue entry and the data published as part of it", step 3 (retrieved 2026-11-13). Recorded in every provenance file, `weather.json` and `climatology.json`. |
| Run provenance | `weather.json` in every run with `models.weather_source = era5` (collection, DOI, citation, licence, attribution, variable, resolution, simulated → historical year map, per site: node, files, SHA-256, CDS request ids, retrieval times), listed with its SHA-256 in `manifest.json` `outputs` |

**Economics inputs** (`configs/economics/*.yaml`; every value carries `provenance` and, unless
`assumed`, a page-level source). The cited report: NASA Office of Technology, Policy, and
Strategy, E. Rodgers et al., *Space-Based Solar Power*, Report ID 20230018600, 11 January 2024,
<https://www.nasa.gov/wp-content/uploads/2024/01/otps-sbsp-report-final-tagged-approved-1-8-24-tagged-v2.pdf>
(retrieved 2026-11-13, 108 pages, SHA-256 `60ecde4628d6f4057ace4e853d3f429165d5642ff9e3028a9fa4fad2a64745a4`).
Page numbers are PDF pages (printed page in brackets). Used as a source of individual input
values and of the LCOE formula only. Its microwave reference designs are not used as a design,
but two of its system-level numbers are **transferred** to these runs and labelled `assumed`
for that reason (M8a review): the rectenna / ground-segment cost per m² of the microwave
reference designs is also used as a proxy for the optical (laser-PV) receiver, for which no
sourced cost was found, and the operations cost of one GW-class SBSP system is applied unchanged
to runs of any size.

| Input | Values (optimistic / nominal / conservative) | Provenance |
|---|---|---|
| Launch cost to LEO [USD/kg] | 85 / 850 / 1500 | 850: sourced, PDF p. 38 [21] ("$100M or $1000/kg (with 15% block buy discount, $85M or 850/kg)"); 1500: sourced, PDF p. 90 [73] and p. 28 [11] (Falcon Heavy 2019, lowest recorded); 85: derived, PDF p. 90 [73] ($10M/launch announced by SpaceX) at the report's $100M ↔ $1000/kg conversion, × 0.85 block-buy discount (PDF p. 28) |
| Discount rate | 0.03 / 0.03 / 0.07 | 0.03: sourced, PDF p. 73 [56] (NREL simple LCOE, "default discount rate of 3%"); 0.07 assumed |
| System lifetime [yr] | 30 | sourced, PDF p. 28 [11], Table 2 |
| Hardware lifetime [yr] | 15 / 10 / 8 | 10: sourced, PDF p. 28 [11]; 15: sourced, PDF p. 91 [74] (GEO hardware state of practice); 8 assumed |
| Microwave receiver capex [USD/m²] (`microwave_receiver_capex_per_m2_usd`) | 22.06 / 109.64 / 131.62 | derived: PDF p. 84 [67] RD1 rectenna $623.8M / π(3 km)²; PDF p. 68 [51] RD1 ground system $3.1B / π(3 km)²; RD2 ground systems $8.27B / (5 π (2 km)²) |
| Optical receiver capex [USD/m²] (`optical_receiver_capex_per_m2_usd`) | 22.06 / 109.64 / 131.62 | **assumed**: the microwave values used as a stated proxy (technology transfer rectenna → laser-PV array; no sourced optical-receiver cost); drives the optical LCOEs (M8a.md F) |
| Transmitter capex [USD/m² of aperture] (`transmitter_capex_per_m2_usd`) | 0 | **assumed**: transmitter cost folded into the flat manufacturing cost per kg |
| Fixed operations [USD/yr] | 1.44 × 10⁷ | **assumed** transfer of a sourced number: PDF p. 68 [51] Table 11 ($1.2M/month for one GW-class SBSP system, autonomous operations), applied unchanged to every run |
| Operations per satellite [USD/yr] | 0 / 5 × 10⁴ / 5 × 10⁵ | 5 × 10⁵: sourced, PDF p. 28 [11] ("today ~500k/year per satellite"); others assumed |
| Electricity price for the NPV [USD/kWh] | 0.05 | sourced, PDF p. 8 [vii] (upper end of the 0.02–0.05 USD/kWh 2050 terrestrial projections) |
| LEO → GEO Δv [m/s] (`geo_nominal.yaml`) | 4 700 | sourced, PDF p. 90 [73] (">4,700m/s is needed to reach GEO from LEO", electric-propulsion sensitivity) |
| Manufacturing [USD/kg], Isp [s], degradation [1/yr], construction years, replacement fraction, opex fraction of capex, LEO altitude, phasing years (deployment time budget), minimum meaningful energy, specific masses | see the YAML files | **assumed** (no defensible single source; varied in the scenarios / Monte Carlo); every one, including the integer inputs, carries a `provenance` field |
| Monte Carlo ranges (`uncertainty`) | low / mode / high | bounds = the optimistic and conservative scenario values, mode = the nominal value (both checked on load), so their provenance is that of those values |
| Constants | g0 = 9.80665 m s⁻² (exact, 3rd CGPM 1901); GM_E = 3.9860043550702266 × 10¹⁴ m³ s⁻² (DE440 `gm_de440.tpc` `BODY399_GM`); R_E = 6 378 136.6 m (`pck00011.tpc` `BODY399_RADII`) | sourced |

The `weather` extra (`cdsapi>=0.7.7`, `xarray>=2024.1`, `h5netcdf[h5py]>=1.3`; locked: cdsapi
0.7.7, xarray 2026.9.0, h5netcdf 1.8.3, h5py 3.16.0; all manylinux/pure wheels on Python 3.12)
is optional; without it ERA5 configurations are rejected at validation and the ERA5 tests skip.

## Sourced constants and defaults

| Quantity | Value | Where | Source |
|---|---|---|---|
| Astronomical unit | 149 597 870 700 m (exact) | `helioswarm.constants.AU_M` | IAU 2012 Resolution B2 |
| Speed of light | 299 792 458 m s⁻¹ (exact) | `helioswarm.constants.SPEED_OF_LIGHT_M_S` | SI definition of the metre (CGPM 1983) |
| Heliocentric gravitational constant | 1.3271244004127942 × 10²⁰ m³ s⁻² (TDB-compatible) | `helioswarm.constants.GM_SUN_M3_S2` | DE440 (Park et al. 2021, AJ 161, 105) via NAIF `gm_de440.tpc` `BODY10_GM` (see above) |
| Stefan–Boltzmann constant | 5.670374419 × 10⁻⁸ W m⁻² K⁻⁴ | `helioswarm.constants.STEFAN_BOLTZMANN_W_M2_K4` | CODATA 2018 |
| Nominal solar irradiance (default) | 1361 W m⁻² | `helioswarm.constants.SOLAR_IRRADIANCE_NOMINAL_1AU_W_M2` | IAU 2015 Resolution B3; consistent with Kopp & Lean (2011), GRL 38, L01706 |
| WGS 84 a, 1/f | 6 378 137 m, 298.257223563 | `helioswarm.constants.WGS84_*` | NGA.STND.0036 (WGS 84) |
| Earth Rotation Angle rate | 1.00273781191135448 rev / UT1 day | `helioswarm.constants.EARTH_ROTATION_ANGLE_RATE_REV_PER_UT1_DAY` | IERS Conventions (2010), TN 36, eq. 5.15 |
| Stellar ("sidereal") day | 86 164.0989 s (derived) | `helioswarm.constants.STELLAR_DAY_S` | 86400 s / ERA rate |
| Mean obliquity of J2000 | 84 381.448″ | `helioswarm.constants.OBLIQUITY_J2000_ARCSEC` | IAU 1976 (Lieske et al. 1977, A&A 58, 1); ECLIPJ2000 definition in NAIF Frames Required Reading |
| Nominal solar radius | 6.957 × 10⁸ m | `helioswarm.constants.SOLAR_RADIUS_NOMINAL_M` | IAU 2015 Resolution B3 (equal to `pck00011` `BODY10_RADII`; M3 reads the PCK value at run time and a test checks the two agree) |
| Lunar mean radius | 1 737 400 m | `helioswarm.constants.MOON_RADIUS_M` | IAU WGCCRE 2015 (Archinal et al. 2018, Celest. Mech. Dyn. Astron. 130, 22); equal to `pck00011` `BODY301_RADII` |
| WGS 84 b, e² (derived) | a(1 − f), f(2 − f) | `helioswarm.constants.WGS84_SEMIMINOR_AXIS_M`, `WGS84_FIRST_ECCENTRICITY_SQUARED` | NGA.STND.0036 table 3.1 |
| IAU Earth rotation model | α₀ = −0.641 T, δ₀ = 90 − 0.557 T, W = 190.147 + 360.9856235 d (deg) | `pck00011.tpc` (read by SPICE, `IAU_EARTH` frame) | IAU WGCCRE 2009 (Archinal et al. 2011, Celest. Mech. Dyn. Astron. 109, 101); not updated in the 2015 report |
| Day | 86 400 s | `helioswarm.constants.SECONDS_PER_DAY` | Definition used by IAU/SPICE (Julian day) |
| Solar irradiance at 1 au | 1361 W m⁻² (default; configurable) | `constants.solar_constant_1au_W_m2` (default taken from `helioswarm.constants`) | IAU 2015 Resolution B3, nominal total solar irradiance |
| Deep-space sink temperature | 2.72548 K (default, `helioswarm.constants.CMB_TEMPERATURE_K`) | `thermal.space_temperature_K` | Fixsen (2009), ApJ 707, 916: T_CMB = 2.72548 K |
| Far-field threshold | N_F ≤ 0.5 (default) | `models.fraunhofer_max_fresnel_number` | Far-field distance R ≥ 2D²/λ: IEEE Std 145-2013 (Definitions of Terms for Antennas, "far-field region"); C. A. Balanis, *Antenna Theory: Analysis and Design* (Wiley), ch. 2 "Field Regions" |
| Frame/body names | `J2000`, `SOLAR SYSTEM BARYCENTER`, `IAU_EARTH` | `frames.*` | NAIF SPICE Frames / NAIF IDs / PCK Required Reading |
| NAIF body ids | SUN 10, MERCURY/VENUS/EARTH/MARS/JUPITER/SATURN/URANUS/NEPTUNE BARYCENTER 1–8, EARTH 399, MOON 301 | `helioswarm.ephemeris.NAIF_IDS` | NAIF IDs Required Reading |
| GM of the Sun and the perturbing bodies | DE440 values, read at run time with `bodvrd` | `gm_de440.tpc` | Park et al. 2021 (AJ 161, 105) via NAIF |
| Reference ellipsoid | WGS84 | `frames.earth_ellipsoid` | NGA.STND.0036 (WGS 84) |

M4 adds no external data and no new sourced constant (it uses σ, `S_1AU`, `T_CMB` and the au
from the table above). Its three new configuration fields
(`collector.max_operating_temperature_K` = 400 K, `thermal.pv_view_factor` = 1.0,
`thermal.radiator_specific_mass_kg_m2` = None) are **assumed**. The order-of-magnitude bound
on Earth albedo/IR at the collector in `docs/model_assumptions.md` (M4) uses global-mean Bond
albedo ≈ 0.3 and outgoing long-wave radiation ≈ 240 W m⁻² from Trenberth, Fasullo & Kiehl
(2009), BAMS 90, 311; these values are not used by any computation.

All other defaults and every value in `configs/mvp_single_satellite.yaml` are **assumed**
scenario inputs. The MVP config is adapted from the PDF §5 example, which the PDF itself calls
"a schema example, not a recommended engineering design".

The model-validity thresholds `models.gaussian_beam_min_truncation_ratio` (default 2.5) and
`atmosphere.plane_parallel_min_elevation_deg` (default 10°) are also **assumed**. Their
documented error bounds are computed, not cited: the Gaussian truncation factor by closed form
(checked by quadrature in `tests/unit/test_config_validity.py`), and the air-mass excess against
the Kasten & Young (1989), Appl. Opt. 28, 4735 interpolation formula.

## M8: PV technology data (sourced), study inputs

**Verification of the sources.** Both sources were fetched on 2026-11-14 into
`/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/sources/pv_datasheets/` (not committed; values
transcribed below with `pdftotext -layout`), and their identity was checked at the publisher:
the AZUR SPACE datasheets are served from the manufacturer's own site (azurspace.com product
download links); the JPL report's bibliographic record was read from the NASA Technical Reports
Server API (`https://ntrs.nasa.gov/api/citations/19790024459`: title, authors, report numbers
NASA-CR-162290 / JPL-PUB-78-15-VOL-8, date 1979-09-01). Neither has a DOI.

| Source | File / SHA-256 | Values used | Use |
|---|---|---|---|
| AZUR SPACE Solar Power GmbH, "30% Triple-Junction (TJ) III-V Bare Solar Cells & Assemblies, TJ 3G30-Advanced (8x8)", datasheet DB 00010892-01, issue 11-Apr-2025, https://www.azurspace.com/media/uploads/file_links/file/bdb_00010892-01-00_tj3g30-advanced_8x8.pdf | `07a7f5ff5cbaa358a79cd1430fbaa496a51a0c62e03b82221f249f5bd77821ff` | CIC (assembly) BOL: η = 29.0 % at AM0 1367 W/m², 28 °C; Vmp = 2400 mV, Imp = 998 mA; ΔVmp/ΔT = −6.7 mV/°C, ΔImp/ΔT = +0.48 mA/°C (BOL, range 28–140 °C); absorptivity ≤ 0.91 (threshold value, 100 µm AR coverglass) | `configs/scenarios/pv_iii_v_3g30a.yaml`; optimistic η end / nominal γ of `configs/scenarios/*` and `configs/uncertainty*.yaml` |
| AZUR SPACE, TJ 3G30-Advanced (HP), datasheet DB 00010894-02, issue 24-Mar-2026, https://www.azurspace.com/media/uploads/file_links/file/bdb_00010894-02-00_3g30-advanced_hp.pdf | `ce04aaf519a2df4b3ef31d4c79c36b2951d595b449f8df340ca17f1479b73998` | same Voc/Vmp coefficients; CIC η 28.7 % (1367 W/m²) | cross-check only (not used) |
| B. E. Anspaugh, D. M. Beckert, R. G. Downing, T. F. Miyahira, R. S. Weiss, "Characterization of Solar Cells for Space Applications, Volume VIII: Electrical Characteristics of Spectrolab BSF, BSR, Textured 290-Micron Solar Cells (K7) as a Function of Intensity, Temperature and Irradiation", JPL Publication 78-15 Vol. VIII (NASA-CR-162290), 1 Sep 1979, https://ntrs.nasa.gov/api/citations/19790024459/downloads/19790024459.pdf | `4d55c65a6f3b91ff9b52b1f89cf5fdec1d044b5a1bbbc2cffee8e0ea857e7f95` | Table 7 (average AM0 efficiency, %, 135.3 mW/cm² column; bare cells, no coverslide): 15.24 at 20 °C, 13.90 at 40 °C (scanned report, OCR text; the neighbouring entries 16.48 at 0 °C and 12.54 at 60 °C confirm the monotone trend) | `configs/scenarios/pv_silicon_k7_1979.yaml`; conservative ends |
| JPL Publication 78-15 Vol. V (OCLI MLAR wraparound silicon cells), NTRS 19790014459 | `c95743526dc35c1a45ee407926d793a04ee32a7f6736d0400864c11e4981c7e0` | — | fetched while searching; not used |

Derived values (arithmetic on the sourced numbers; provenance `derived` in the uncertainty study
files — an extension of the PDF §13.4 label set that requires a source — and **`sourced`** in the
machine-readable provenance of every run configuration that uses them: `provenance_overrides`
in `configs/scenarios/{nominal,optimistic,conservative,pv_iii_v_3g30a,pv_silicon_k7_1979}.yaml`
and the GEO twins, so run manifests and provenance tables report them as sourced; the citation is
in each file's header and `project.description`; the K7 absorptivity 0.90 stays `assumed`;
tested by `tests/integration/test_m8_forward.py::test_pv_technology_values_follow_from_the_sourced_numbers`):

| Quantity | Formula | Value |
|---|---|---|
| III-V γ | −6.7/2400 + 0.48/998 | −0.00231 K⁻¹ (−2.311e-3) |
| III-V η_ref at 300 K | (0.290/0.91)(1 − 0.00231 (300 − 301.15)) | 0.3195 |
| K7 silicon γ | (13.90 − 15.24)/(20 × 15.24) | −0.0044 K⁻¹ (−4.396e-3) |
| K7 silicon η_ref at 300 K | (0.1524/0.90)(1 − 0.0044 × 6.85); α = 0.90 **assumed** (not in the source) | 0.1642 |

The optimistic scenario and the triangular upper end of `pv_efficiency_reference` in
`configs/uncertainty*.yaml` use the III-V η_ref = 0.3195 (converted with α = 0.91) with the base
absorptivity 0.90 and the assumed γ = −0.0004 K⁻¹, i.e. 28.74 % incident-basis efficiency at
28 °C, 0.9 % below the datasheet's 29.0 % (only `pv_iii_v_3g30a.yaml` reproduces 29.0 %; M8
review, documented in the scenario header and `docs/model_assumptions.md` M8).

Not sourced (labelled `assumed` everywhere): laser electrical-to-beam efficiency, laser-PV
receiver conversion efficiency, pointing jitter, radiator specific mass, the optimisation bounds,
objective reference points and constraint limits (e.g. the 1000 W/m² irradiance cap, 10 MW
receiver capacity, 100 m aperture, 0.2 minimum availability), the Rayleigh pointing-bias scale and
the wavefront-error range of the uncertainty studies, and equal probabilities of the ERA5 years.
A sourced laser-PV receiver efficiency or flux limit was not found; none was invented.

ERA5 (M8a provenance) is the aleatory weather input of `configs/uncertainty*.yaml` and the high
fidelity level; economics inputs are those of `configs/economics/*.yaml` (M8a), plus
`geo_optimistic.yaml` / `geo_conservative.yaml` (M8): the optimistic/conservative cost files with
the explicit, sourced LEO→GEO Δv of `geo_nominal.yaml` (NASA OTPS 2024, PDF p. 90: ">4,700 m/s");
the geo/helio files differ only in the transport block (tested).
