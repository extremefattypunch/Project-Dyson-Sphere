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
