# Data provenance

This file records the provenance of every external dataset and of every **sourced** parameter
default (PDF §14.1, §14.4, §13.4). Every parameter is labelled `measured`, `sourced`, `assumed` or
`swept` in the configuration schema (`json_schema_extra["provenance"]`):

- The label in the schema describes the *default* value.
- A run can relabel a value with `provenance_overrides`, for example when a parameter is swept.
- `helioswarm validate-config CONFIG --show-provenance` prints the effective label of every
  resolved value.

## External data

| Dataset | Version | Source URL | SHA-256 | Retrieved | Used by |
|---|---|---|---|---|---|
| SPICE kernels | M2 | (to be recorded in `data/kernels/kernels.json`) | | | M2+ |

No external data file is read by the code as of M1. To source the GM☉ constant, M1 *read*
NAIF's `gm_de440.tpc` once by hand. The file was retrieved on 2026-11-12 from
`https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc`, with SHA-256
`924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140`. Its header reads "GMs for
1–10, 199, 299, 301, 399 reset to their DE440 values", BVS/NAIF 2022-12-14. The line used was
`BODY10_GM = ( 1.3271244004127942E+11 )` km³ s⁻². The file was not kept; M2 fetches and
verifies it through `scripts/fetch_kernels.py`.

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
| Reference ellipsoid | WGS84 | `frames.earth_ellipsoid` | NGA.STND.0036 (WGS 84) |

All other defaults and every value in `configs/mvp_single_satellite.yaml` are **assumed**
scenario inputs. The MVP config is adapted from the PDF §5 example, which the PDF itself calls
"a schema example, not a recommended engineering design".

The model-validity thresholds `models.gaussian_beam_min_truncation_ratio` (default 2.5) and
`atmosphere.plane_parallel_min_elevation_deg` (default 10°) are also **assumed**. Their
documented error bounds are computed, not cited: the Gaussian truncation factor by closed form
(checked by quadrature in `tests/unit/test_config_validity.py`), and the air-mass excess against
the Kasten & Young (1989), Appl. Opt. 28, 4735 interpolation formula.
