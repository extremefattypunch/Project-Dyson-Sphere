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
| Nominal solar radius | 6.957 × 10⁸ m | `helioswarm.constants.SOLAR_RADIUS_NOMINAL_M` | IAU 2015 Resolution B3 |
| Day | 86 400 s | `helioswarm.constants.SECONDS_PER_DAY` | Definition used by IAU/SPICE (Julian day) |
| Solar irradiance at 1 au | 1361 W m⁻² (default; configurable) | `constants.solar_constant_1au_W_m2` (default taken from `helioswarm.constants`) | IAU 2015 Resolution B3, nominal total solar irradiance |
| Deep-space sink temperature | 2.72548 K (default, `helioswarm.constants.CMB_TEMPERATURE_K`) | `thermal.space_temperature_K` | Fixsen (2009), ApJ 707, 916: T_CMB = 2.72548 K |
| Far-field threshold | N_F ≤ 0.5 (default) | `models.fraunhofer_max_fresnel_number` | Far-field distance R ≥ 2D²/λ: IEEE Std 145-2013 (Definitions of Terms for Antennas, "far-field region"); C. A. Balanis, *Antenna Theory: Analysis and Design* (Wiley), ch. 2 "Field Regions" |
| Frame/body names | `J2000`, `SOLAR SYSTEM BARYCENTER`, `IAU_EARTH` | `frames.*` | NAIF SPICE Frames / NAIF IDs / PCK Required Reading |
| NAIF body ids | SUN 10, MERCURY/VENUS/EARTH/MARS/JUPITER/SATURN/URANUS/NEPTUNE BARYCENTER 1–8, EARTH 399, MOON 301 | `helioswarm.ephemeris.NAIF_IDS` | NAIF IDs Required Reading |
| GM of the Sun and the perturbing bodies | DE440 values, read at run time with `bodvrd` | `gm_de440.tpc` | Park et al. 2021 (AJ 161, 105) via NAIF |
| Reference ellipsoid | WGS84 | `frames.earth_ellipsoid` | NGA.STND.0036 (WGS 84) |

All other defaults and every value in `configs/mvp_single_satellite.yaml` are **assumed**
scenario inputs. The MVP config is adapted from the PDF §5 example, which the PDF itself calls
"a schema example, not a recommended engineering design".

The model-validity thresholds `models.gaussian_beam_min_truncation_ratio` (default 2.5) and
`atmosphere.plane_parallel_min_elevation_deg` (default 10°) are also **assumed**. Their
documented error bounds are computed, not cited: the Gaussian truncation factor by closed form
(checked by quadrature in `tests/unit/test_config_validity.py`), and the air-mass excess against
the Kasten & Young (1989), Appl. Opt. 28, 4735 interpolation formula.
