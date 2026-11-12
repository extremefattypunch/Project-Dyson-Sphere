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

None are used yet as of M0.

## Sourced constants and defaults

| Quantity | Value | Where | Source |
|---|---|---|---|
| Astronomical unit | 149 597 870 700 m (exact) | `helioswarm.constants.AU_M` | IAU 2012 Resolution B2 |
| Nominal solar radius | 6.957 × 10⁸ m | `helioswarm.constants.SOLAR_RADIUS_NOMINAL_M` | IAU 2015 Resolution B3 |
| Day | 86 400 s | `helioswarm.constants.SECONDS_PER_DAY` | Definition used by IAU/SPICE (Julian day) |
| Solar irradiance at 1 au | 1361 W m⁻² (default; configurable) | `constants.solar_constant_1au_W_m2` | IAU 2015 Resolution B3, nominal total solar irradiance |
| Deep-space sink temperature | 2.725 K (default) | `thermal.space_temperature_K` | Fixsen (2009), ApJ 707, 916: T_CMB = 2.72548 K |
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
