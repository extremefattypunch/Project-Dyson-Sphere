# M7b: GEO solar-power-satellite baseline through the same chain (harmonised comparison)

Status: implemented; all gates pass (D, "Review outcomes"). **Provenance of the cited runs
(corrected in the review):** the comparison data first ran on the uncommitted working tree
while `report.py` (lazily imported when a run is written) and `compare.py` were still being
edited, so runs of one array were written by different code (affecting only `report.md`'s
key-parameter list, not any number), and the source-tree hashes quoted then were recorded in
no run output. After the review fixes, every comparison point was **regenerated from one
source tree** with `slurm/comparison_submit.sh`, and each run manifest now records it
(`manifest.source_sha256` = `c2be65c6…06a9`, layout `helioswarm-run-manifest/3`;
equivalently `find src -name '*.py' | sort | xargs sha256sum | sha256sum` = `40edb742…a5bd`;
`git.commit = e86f8e7`, `git.dirty = true`). The regenerated summaries are bit-identical to the
first runs for every point (diff of all `summary.json`), so the tables below are unchanged. A
separate step commits.

## A. Summary

- **Geocentric orbits** (`config.py`, `geocentric.py`, `frames.py`, `ephemeris.py`, `orbit.py`,
  `propagate.py`): `orbit.elements_center: EARTH` with J2000 equatorial elements and
  `mu = GM_earth` (`BODY399_GM`, gm_de440), L0 two-body Kepler about the Earth's centre, shifted
  to the SSB with the Earth's DE440 state (origin `EARTH` added to frame-tagged states;
  Sun↔SSB unchanged bit for bit). **Geostationary preset** `orbit.geostationary_longitude_deg`:
  circular orbit in the IAU_EARTH equator of the epoch, `ω = ω_vec · p` from `sxform`
  (360.98560595°/day; 86 164.1004 s), `r = (GM_earth/ω²)^(1/3) = 42 164 172.6 m`, velocity
  `ω p × r`. Over 2030 the sub-satellite longitude stays within 1.3e-7° (latitude ±0.0055° by
  year end from the IAU pole precession). L1 for geocentric orbits is rejected (documented):
  **L0 + station-keeping** is the modelled assumption; the omitted perturbations are quantified
  (lunisolar 0.84°/yr inclination growth, numerically confirmed 0.849°; J2 +522 m synchronous
  radius; SRP eccentricity 8.7e-3, ±365 km, ±0.99° at the comparison satellite's A/m ≈ 0.60
  m²/kg — corrected in the review from 2.9e-4/±12 km/±0.033°, which hold for A/m = 0.02).
  Every GEO run names the assumption in its manifest and report (review).
- **Eclipses**: the existing conical-shadow illumination (penumbra included) gives the GEO
  equinox seasons; checked against the closed-form shadow geometry at both equinoxes,
  mid-season and out of season (15 s on 50–72 min) and over the whole of 2030 (season
  boundaries 02-26 – 04-12 and 08-31 – 10-16, 93 shadow days, yearly fractions; slow test). In
  the year 2030: 93 days with Earth shadow, longest outage 81 min (4870 s at the 974 s step).
  Converged (60 s) fractions: umbra 0.881 %, penumbra 0.145 % (of which 0.028 % is **lunar**
  partial shadowing of the GEO satellite, ~148 min/yr with the Moon as the main occulter; the
  same model, Earth and Moon occulters). The 974 s comparison grid gives 0.889 % / 0.130 %:
  counting noise of ±1.7 % (umbra) and ±12 % (penumbra) relative, as expected
  (`docs/model_assumptions.md` M7b "Sampling of the reported fractions").
- **Access**: zenith link with apparent elevation 90° − ωτ (8.7e-6 rad west, planetary
  aberration; tested to 5e-8 rad), range `L/(1 − u·v/c)` around 35 786.0 km (±3.6 km
  daily; tested to 1 m). The diurnal-aliasing warning no longer fires when the access never
  changes (GEO), the gap is still reported.
- **Beam regime**: N_F = 2.79 for the 10 m optical aperture at GEO: `airy_far_field` refuses
  (tested), the GEO configurations use **`fresnel_hankel`**; microwave N_F = 0.228 (FH and far
  field differ by 1.1e-4 in capture there, tested < 1e-3). Heliocentric cases keep the
  far-field models (N_F ≤ 0.14); at heliocentric ranges (7.2e10 and 8.2e10 m, zenith and 30°)
  the base-technology captures of the far-field and Fresnel-Hankel models agree to 1e-7
  (optical) and 6e-10 (microwave) relative
  (`test_geo_comparison_run.py::test_far_field_and_fresnel_hankel_agree_at_the_heliocentric_range`;
  the year-integrated figures quoted before the review came from runs whose outputs were
  deleted and are withdrawn). **Chebyshev beam table**
  (`models.beam_evaluation: chebyshev_table`, `beam_table.py`): FH is evaluated at 9–15
  (range, elevation) nodes per GEO run instead of 32 401 epochs, interpolated and validated in
  every run against the reference (max error ≤ 3.6e-11 in all GEO runs); a failed check stops
  the run.
- **Harmonised comparison**: `configs/comparison/` with one shared technology block per
  technology joined by the new **`extends:`** mechanism (`config.read_config_mapping`); the
  resolved heliocentric and GEO configurations differ only in orbit keys, the L1 integrator
  step and the beam model of the regime (tested leaf by leaf; `compare` refuses anything else).
  **`helioswarm compare`** (`compare.py`) writes the comparison tables (CSV + Markdown), the
  harmonised-assumption table with provenance labels, a figure and a manifest from the saved
  sweep runs. New fields are hash-stable (`omit_when_default`): the MVP hash is unchanged
  (`a3fafc4d…23af`).

### Comparison results (Slurm sweeps; one year 2030 at 974 s; one satellite, one receiver site at 0° N 0° E; clear sky all year)

Base configurations (`helioswarm compare` → `comparison_baseline.md`):

#### microwave

| metric | helio_1au_microwave (heliocentric) | geo_microwave (geostationary) |
|---|---|---|
| aperture D [m] | 1000 | 1000 |
| receiver radius [m] | 1e+04 | 1e+04 |
| status | ok | ok |
| P_grid mean [kW] | 0.001592 | 2028 |
| availability | 0.4421 | 0.9911 |
| capacity factor | 0.3004 | 0.9909 |
| end-to-end E_grid/E_sun | 1.17e-07 | 0.1504 |
| eclipse (umbra) | 0.0 | 0.008889 |
| penumbra | 0.0 | 0.001296 |
| capture E_capt/E_atm | 1.74e-06 | 0.9786 |
| range mean [km] | 7.723e+07 | 3.579e+04 |
| peak irradiance [W/m2] | 2.332e-08 | 0.0983 |
| mean flux on Rx [W/m2] | 1.39e-08 | 0.007898 |
| mass proxy [t] | 3.928e+04 | 3.928e+04 |
| mass / delivered kW [t/kW] | 2.467e+07 | 19.37 |


#### optical

| metric | helio_1au_optical (heliocentric) | geo_optical (geostationary) |
|---|---|---|
| aperture D [m] | 10 | 10 |
| receiver radius [m] | 1000 | 1000 |
| status | ok | ok |
| P_grid mean [kW] | 8.758 | 704.9 |
| availability | 0.4421 | 0.9911 |
| capacity factor | 0.3036 | 0.9909 |
| end-to-end E_grid/E_sun | 0.0006435 | 0.05228 |
| eclipse (umbra) | 0.0 | 0.008889 |
| penumbra | 0.0 | 0.001296 |
| capture E_capt/E_atm | 0.02782 | 0.9993 |
| range mean [km] | 7.723e+07 | 3.579e+04 |
| peak irradiance [W/m2] | 0.02756 | 7.653e+04 |
| mean flux on Rx [W/m2] | 0.01625 | 0.5835 |
| mass proxy [t] | 16.62 | 16.69 |
| mass / delivered kW [t/kW] | 1.898 | 0.02368 |

Every sweep point (aperture × receiver radius; `comparison_points.md`; figure `$HELIOSWARM_RUNS_DIR/comparison/report/comparison.png`: delivered power and end-to-end efficiency vs receiver radius per aperture, both architectures, both technologies; `compare` exit 0, 33 points ok, 3 failed):

| case | point | D [m] | R_rx [m] | status | P_grid mean [kW] | availability | capacity factor | end-to-end E_grid/E_sun | eclipse (umbra) | penumbra | capture E_capt/E_atm | range mean [km] | peak irradiance [W/m2] | mean flux on Rx [W/m2] | mass proxy [t] | mass / delivered kW [t/kW] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| helio_1au_optical | p0000 | 10 | 100 | ok | 0.08898 | 0.4421 | 0.3022 | 6.537e-06 | 0.0 | 0.0 | 0.0002826 | 7.723e+07 | 0.02756 | 0.01651 | 16.62 | 186.8 |
| helio_1au_optical | p0001 (base) | 10 | 1000 | ok | 8.758 | 0.4421 | 0.3036 | 0.0006435 | 0.0 | 0.0 | 0.02782 | 7.723e+07 | 0.02756 | 0.01625 | 16.62 | 1.898 |
| helio_1au_optical | p0002 | 10 | 1e+04 | ok | 251.6 | 0.4421 | 0.4215 | 0.01849 | 0.0 | 0.0 | 0.7991 | 7.723e+07 | 0.02756 | 0.004669 | 16.62 | 0.06606 |
| helio_1au_optical | p0003 | 30 | 100 | ok | 0.7967 | 0.4421 | 0.3023 | 5.853e-05 | 0.0 | 0.0 | 0.00253 | 7.723e+07 | 0.2481 | 0.1478 | 48.04 | 60.3 |
| helio_1au_optical | p0004 | 30 | 1000 | ok | 69.23 | 0.4421 | 0.3151 | 0.005086 | 0.0 | 0.0 | 0.2199 | 7.723e+07 | 0.2481 | 0.1285 | 48.04 | 0.6938 |
| helio_1au_optical | p0005 | 30 | 1e+04 | ok | 293.5 | 0.4421 | 0.4341 | 0.02156 | 0.0 | 0.0 | 0.9321 | 7.723e+07 | 0.2481 | 0.005446 | 48.04 | 0.1637 |
| helio_1au_optical | p0006 | 100 | 100 | ok | 8.352 | 0.4421 | 0.3036 | 0.0006136 | 0.0 | 0.0 | 0.02652 | 7.723e+07 | 2.756 | 1.55 | 405.4 | 48.54 |
| helio_1au_optical | p0007 | 100 | 1000 | ok | 249.5 | 0.4421 | 0.4174 | 0.01833 | 0.0 | 0.0 | 0.7924 | 7.723e+07 | 2.756 | 0.4629 | 405.4 | 1.625 |
| helio_1au_optical | p0008 | 100 | 1e+04 | ok | 308.4 | 0.4421 | 0.4399 | 0.02266 | 0.0 | 0.0 | 0.9796 | 7.723e+07 | 2.756 | 0.005724 | 405.4 | 1.314 |
| geo_optical | p0000 | 10 | 100 | ok | 700.3 | 0.9911 | 0.9909 | 0.05194 | 0.008889 | 0.001296 | 0.9928 | 3.579e+04 | 7.653e+04 | 57.97 | 16.69 | 0.02384 |
| geo_optical | p0001 (base) | 10 | 1000 | ok | 704.9 | 0.9911 | 0.9909 | 0.05228 | 0.008889 | 0.001296 | 0.9993 | 3.579e+04 | 7.653e+04 | 0.5835 | 16.69 | 0.02368 |
| geo_optical | p0002 | 10 | 1e+04 | failed: BeamModelValidityError: Fresnel (paraxial) approximation not justified: neglected quartic phase pi (a + rho)^4 / (4 lamb | — | — | — | — | — | — | — | — | — | — | — | — |
| geo_optical | p0003 | 30 | 100 | ok | 703.7 | 0.9911 | 0.9909 | 0.05219 | 0.008889 | 0.001296 | 0.9975 | 3.579e+04 | 4269 | 58.25 | 48.11 | 0.06837 |
| geo_optical | p0004 | 30 | 1000 | ok | 705.3 | 0.9911 | 0.9909 | 0.05231 | 0.008889 | 0.001296 | 0.9998 | 3.579e+04 | 4269 | 0.5838 | 48.11 | 0.06822 |
| geo_optical | p0005 | 30 | 1e+04 | failed: BeamModelValidityError: Fresnel (paraxial) approximation not justified: neglected quartic phase pi (a + rho)^4 / (4 lamb | — | — | — | — | — | — | — | — | — | — | — | — |
| geo_optical | p0006 | 100 | 100 | ok | 704.7 | 0.9911 | 0.9909 | 0.05227 | 0.008889 | 0.001296 | 0.999 | 3.579e+04 | 459.6 | 58.33 | 405.5 | 0.5753 |
| geo_optical | p0007 | 100 | 1000 | ok | 705.4 | 0.9911 | 0.9909 | 0.05232 | 0.008889 | 0.001296 | 0.9999 | 3.579e+04 | 459.6 | 0.5839 | 405.5 | 0.5748 |
| geo_optical | p0008 | 100 | 1e+04 | failed: BeamModelValidityError: Fresnel (paraxial) approximation not justified: neglected quartic phase pi (a + rho)^4 / (4 lamb | — | — | — | — | — | — | — | — | — | — | — | — |
| helio_1au_microwave | p0000 | 500 | 1000 | ok | 3.982e-06 | 0.4421 | 0.3004 | 2.925e-10 | 0.0 | 0.0 | 4.35e-09 | 7.723e+07 | 5.83e-09 | 3.477e-09 | 9829 | 2.469e+09 |
| helio_1au_microwave | p0001 | 500 | 5000 | ok | 9.954e-05 | 0.4421 | 0.3004 | 7.313e-09 | 0.0 | 0.0 | 1.088e-07 | 7.723e+07 | 5.83e-09 | 3.477e-09 | 9829 | 9.875e+07 |
| helio_1au_microwave | p0002 | 500 | 1e+04 | ok | 0.0003982 | 0.4421 | 0.3004 | 2.925e-08 | 0.0 | 0.0 | 4.35e-07 | 7.723e+07 | 5.83e-09 | 3.477e-09 | 9829 | 2.469e+07 |
| helio_1au_microwave | p0003 | 1000 | 1000 | ok | 1.592e-05 | 0.4421 | 0.3004 | 1.17e-09 | 0.0 | 0.0 | 1.74e-08 | 7.723e+07 | 2.332e-08 | 1.39e-08 | 3.928e+04 | 2.467e+09 |
| helio_1au_microwave | p0004 | 1000 | 5000 | ok | 0.0003981 | 0.4421 | 0.3004 | 2.925e-08 | 0.0 | 0.0 | 4.349e-07 | 7.723e+07 | 2.332e-08 | 1.39e-08 | 3.928e+04 | 9.868e+07 |
| helio_1au_microwave | p0005 (base) | 1000 | 1e+04 | ok | 0.001592 | 0.4421 | 0.3004 | 1.17e-07 | 0.0 | 0.0 | 1.74e-06 | 7.723e+07 | 2.332e-08 | 1.39e-08 | 3.928e+04 | 2.467e+07 |
| helio_1au_microwave | p0006 | 2000 | 1000 | ok | 6.364e-05 | 0.4421 | 0.3004 | 4.675e-09 | 0.0 | 0.0 | 6.953e-08 | 7.723e+07 | 9.328e-08 | 5.557e-08 | 1.571e+05 | 2.468e+09 |
| helio_1au_microwave | p0007 | 2000 | 5000 | ok | 0.001591 | 0.4421 | 0.3004 | 1.169e-07 | 0.0 | 0.0 | 1.738e-06 | 7.723e+07 | 9.328e-08 | 5.557e-08 | 1.571e+05 | 9.874e+07 |
| helio_1au_microwave | p0008 | 2000 | 1e+04 | ok | 0.006364 | 0.4421 | 0.3004 | 4.675e-07 | 0.0 | 0.0 | 6.953e-06 | 7.723e+07 | 9.328e-08 | 5.557e-08 | 1.571e+05 | 2.468e+07 |
| geo_microwave | p0000 | 500 | 1000 | ok | 60.33 | 0.9911 | 0.9909 | 0.004475 | 0.008889 | 0.001296 | 0.02912 | 3.579e+04 | 0.02463 | 0.0235 | 9829 | 162.9 |
| geo_microwave | p0001 | 500 | 5000 | ok | 1106 | 0.9911 | 0.9909 | 0.08205 | 0.008889 | 0.001296 | 0.5339 | 3.579e+04 | 0.02463 | 0.01724 | 9829 | 8.885 |
| geo_microwave | p0002 | 500 | 1e+04 | ok | 1953 | 0.9911 | 0.9909 | 0.1448 | 0.008889 | 0.001296 | 0.9425 | 3.579e+04 | 0.02463 | 0.007606 | 9829 | 5.034 |
| geo_microwave | p0003 | 1000 | 1000 | ok | 231.4 | 0.9911 | 0.9909 | 0.01716 | 0.008889 | 0.001296 | 0.1117 | 3.579e+04 | 0.0983 | 0.09012 | 3.928e+04 | 169.8 |
| geo_microwave | p0004 | 1000 | 5000 | ok | 1950 | 0.9911 | 0.9909 | 0.1446 | 0.008889 | 0.001296 | 0.941 | 3.579e+04 | 0.0983 | 0.03038 | 3.928e+04 | 20.15 |
| geo_microwave | p0005 (base) | 1000 | 1e+04 | ok | 2028 | 0.9911 | 0.9909 | 0.1504 | 0.008889 | 0.001296 | 0.9786 | 3.579e+04 | 0.0983 | 0.007898 | 3.928e+04 | 19.37 |
| geo_microwave | p0006 | 2000 | 1000 | ok | 763.5 | 0.9911 | 0.9909 | 0.05663 | 0.008889 | 0.001296 | 0.3685 | 3.579e+04 | 0.379 | 0.2974 | 1.571e+05 | 205.8 |
| geo_microwave | p0007 | 2000 | 5000 | ok | 2024 | 0.9911 | 0.9909 | 0.1501 | 0.008889 | 0.001296 | 0.9769 | 3.579e+04 | 0.379 | 0.03154 | 1.571e+05 | 77.61 |
| geo_microwave | p0008 | 2000 | 1e+04 | ok | 2050 | 0.9911 | 0.9909 | 0.1521 | 0.008889 | 0.001296 | 0.9896 | 3.579e+04 | 0.379 | 0.007987 | 1.571e+05 | 76.61 |

Findings, stated plainly:

- **Per satellite, with identical technology and one receiver site, the GEO baseline delivers
  far more than the heliocentric satellite**, in every metric except eclipses. Optical base
  (10 m, 1 µm, 1 km laser-PV receiver): **704.9 kW vs 8.76 kW** mean P_grid (80×); microwave
  base (1 km array, 2.45 GHz, 10 km rectenna): **2.028 MW vs 1.59 W** (1.3e6×). The cause is
  the range: 35 786 km vs 0.48–0.55 au (7.7e7 km mean, 2160× farther), which collapses the
  far-field capture (∝ (D R_rx/(λR))²) of the heliocentric beam (optical 2.8 %, microwave
  1.7e-6) while the GEO beam is captured almost completely (optical 99.9 %, microwave 97.9 %).
  This is a negative result for the heliocentric architecture as a power-beaming source under
  these assumptions, not a tuning artefact: the same models, the same technology files and
  the same system boundary produce both numbers.
- **Availability and capacity factor**: GEO 0.9911 / 0.991 (continuous zenith access; only
  the umbra blocks the link, 0.889 % of the year, plus 0.130 % in penumbra at reduced power;
  93 eclipse days, ≤ 72 min each); heliocentric 0.4421 / 0.30 (the single equatorial site sees
  the satellite 44 % of the time; no eclipse). The heliocentric case is better only in the
  eclipse metric (~1 % of the time).
- **Conditions under which the heliocentric satellite approaches GEO (optical only)**: large
  apertures **and** km-scale receivers. Capture reaches 0.80 (10 m, 10 km), 0.93 (30 m, 10 km)
  and 0.98 (100 m, 10 km), delivering 252, 294 and 308 kW: within a factor 2.3 of GEO
  (705 kW), and that factor is the single-site availability (0.991/0.442 = 2.24). With
  continuous access (e.g. the three-site network of M7, never in outage) and full capture a
  heliocentric satellite could at most match GEO per satellite plus the ~1 % that GEO loses to
  eclipses; it needs a 10–100× larger transmit aperture or a 10–100× larger receiver radius
  than GEO for that (GEO optical captures 99.3 % with D = 10 m on a 100 m receiver). The
  microwave heliocentric link never approaches GEO in this grid (best 6.4 mW, capture 7e-6 at
  2 km / 10 km).
- **Mass proxy per delivered kW** (same assumed specific masses; no propellant, structure or
  ground segment for either): optical GEO 0.024 t/kW at the base point, heliocentric 1.9 t/kW
  (best heliocentric point 0.066 t/kW at 10 m / 10 km, where the receiver, which the proxy
  excludes, does the work); microwave GEO 19.4 t/kW, heliocentric 2.5e7 t/kW. The GEO
  station-keeping propellant is not in the proxy (L0 + station-keeping assumption).
- **Power density at the receiver**: the GEO optical beam is metres wide, so its peak
  incident irradiance (aberration-free upper bound) is 7.65e4 W/m² for D = 10 m (56× the solar
  constant), 4.27e3 W/m² for 30 m and 460 W/m² for 100 m; the heliocentric optical peak is
  0.028–2.8 W/m². No receiver flux or beam-safety limit is configured (an idealised setting,
  labelled in every report; the limits are now per technology, `_optical.yaml` /
  `_microwave.yaml`). **Bounded sensitivity (review)**: with an *assumed, illustrative* laser-PV
  limit of 1e4 W/m² (beam abort; not a sourced value) and a 100 m receiver
  (`configs/comparison/sensitivity/`, Slurm 60359285), the GEO D = 10 m beam is aborted 99.09 %
  of the year and delivers **13.8 W** (it transmits only in 8 penumbra samples, when the
  reduced PV power pulls its peak just under the limit) instead of 700 kW; D = 30 m and 100 m
  are unaffected (703.7 and 704.7 kW, peaks 4.27e3 and 460 W/m²). The best GEO-optical mass per
  delivered kW then becomes 0.068 t/kW (D = 30 m), about equal to the best heliocentric point
  (0.066 t/kW, 10 m / 10 km receiver), while GEO still delivers 2.4–2.8× more power per
  satellite. Throttling the transmitter to the limit instead of aborting is not modelled
  (estimate for D = 10 m: P scales with the transmitted power, ≈ 705 kW × 1e4/7.65e4 ≈ 92 kW).
  No heliocentric point comes within 3 orders of magnitude of such a limit. Microwave peaks:
  GEO 0.025–0.38 W/m², heliocentric ≤ 1e-7 W/m², below the ~10 W/m² general-public RF
  reference level cited in configs/mvp_single_satellite_microwave.yaml.
- **Same physics, different evaluation regime**: heliocentric cases use the far-field models
  (N_F ≤ 0.14), GEO cases Fresnel-Hankel (optical N_F = 2.79); at heliocentric ranges the two
  give the same base-technology capture to 1e-7 (optical) and 6e-10 (microwave) relative
  (tested). The three GEO-optical 10 km points fail the Fresnel paraxial validity
  check (0.17 rad > 0.1 rad) and are reported as failed; their capture is bounded below by
  the 1 km points (≥ 0.9993).
- **Geometry side-effects**: the GEO link sees the Sun within 0.28° of the beam direction at
  equinox noons (0.31 % of the delivered energy arrives with the Sun within 5° of the beam;
  no Sun-avoidance limit is set for either case); the heliocentric satellite never comes
  closer than 73° to the Sun as seen from the site. PV temperatures are the same (303–305 K);
  the GEO radiator is 3 % larger (453 vs 439 m²) because the Earth reaches 0.983 au at
  perihelion while the heliocentric orbit is a 1 au circle.
- **Not compared**: lifetime, degradation, cost (M8a), weather (clear sky for both: a
  weather upper bound), constellations of either (one satellite each; GEO slot and spectrum
  limits are outside the model).

**Context only (not simulated, not harmonised; never mixed into the tables above).** NASA's
Office of Technology, Policy, and Strategy report *Space-Based Solar Power* (E. Rodgers et al.,
11 January 2024, Report ID 20230018600; citation verified from the PDF, `docs/data_provenance.md`
M7b) assessed two representative GEO-class designs for operation in 2050, each normalised to
2 GW delivered to the grid: RD1 "Innovative Heliostat Swarm" generating power 99 % of the year,
11.5 km² of solar panels, 5.9 Mkg; RD2 "Mature Planar Array" 60 % of the year, 19 km²,
10 Mkg (Executive Summary, p. ii). Those are the report's design assumptions on its own
system boundary (they include structure and the full system mass); this project's mass proxy
and technology blocks are not theirs and no number here is calibrated to them.

## B. Files added/changed

Added:

- `src/helioswarm/geocentric.py` (geocentric L0 trajectory, geostationary preset, perturbation
  budget: secular lunisolar rate, Earth + Sun + Moon integration, J2 and SRP estimates)
- `src/helioswarm/beam_table.py` (validated Chebyshev beam table)
- `src/helioswarm/compare.py` (harmonised comparison from saved runs)
- `configs/comparison/`: `_shared.yaml`, `_optical.yaml`, `_microwave.yaml` (fragments),
  `helio_1au_optical.yaml`, `geo_optical.yaml`, `helio_1au_microwave.yaml`, `geo_microwave.yaml`,
  their `*_sweep.yaml`, `comparison.yaml`
- `slurm/comparison_submit.sh`
- Tests: `tests/integration/test_geo_baseline.py` (8), `tests/integration/test_geo_comparison_run.py`
  (4), `tests/integration/test_comparison_configs.py` (18), `tests/unit/test_config_m7b.py` (28),
  `tests/unit/test_beam_table.py` (7)
- `docs/milestones/M7b.md`

Changed:

- `src/helioswarm/config.py`: `elements_center` EARTH, `elements_frame` J2000,
  `geostationary_longitude_deg`, optional `semimajor_axis_au`/`true_anomaly_deg` with rules,
  geocentric rules, `models.beam_evaluation` + `beam_table_*`, `omit_when_default` field
  metadata and `resolved_dict` stripping, `extends` (`read_config_mapping`, `config_sources`,
  `merge_config_mappings`), constellation rules (no geocentric orbits, no beam table).
- `src/helioswarm/frames.py` (`OriginName` += EARTH), `ephemeris.py` (origin shifts for
  EARTH), `orbit.py` (`center` EARTH, `state_to_elements` about the Earth), `propagate.py`
  (`l0_reference_trajectory` dispatches geocentric orbits, takes the Earth orientation),
  `system.py` (orbit stage with the Earth orientation, GEO orbit log line, beam table stage,
  `SystemRunResult.beam_table`, aliasing note without access edges), `metrics.py` (aliasing
  warning only with access edges), `manifest.py` (geocentric frame note), `report.py` (key
  parameters), `sweep.py`/`validation.py` (`extends`-aware loading), `cli.py` (`compare`).
- `slurm/common.sh`: BLAS/OpenMP threads pinned to the allocation (measured: no speed-up from
  BLAS threads for the FH evaluation).
- Tests: `test_config.py` (required-field messages for the now-optional elements),
  `test_frames.py` (EARTH is a valid origin), `test_metrics.py` (aliasing warning with/without
  access edges).
- Docs: `model_assumptions.md` (M7b section), `data_provenance.md` (M7b), `compliance_matrix.md`
  (O5/§2/§15/§16/§19 rows), `README.md`, `experiments/README.md` (E10).

Added/changed in the review ("Review outcomes"): `tests/unit/test_geocentric.py` (3),
`configs/comparison/sensitivity/` (bounded flux-limit sensitivity), new tests in
`test_geo_baseline.py` (3), `test_geo_comparison_run.py` (2) and `test_config_m7b.py` (1);
`propagate.py` (geocentric guard), `config.py` (GEO named approximation, docstring),
`manifest.py` (`source_sha256`, layout `/3`), `compare.py` (exact assumption values),
`slurm/comparison_submit.sh` (measured sizing), receiver limits moved from `_shared.yaml` to
`_optical.yaml`/`_microwave.yaml` (resolved configurations unchanged).

No stubs for later milestones (no optimisation module, `optimize` command, economics or `paper/`).

## C. Exact commands run

All from `/n/home05/ianpoon/Project-Dyson-Sphere` after `source scripts/cluster_env.sh`.

```bash
# API checks against the installed packages
uv run python -c "import spiceypy, inspect; print(inspect.signature(spiceypy.sxform), inspect.signature(spiceypy.bodvrd))"
uv run python -c "import inspect; from scipy.integrate import solve_ivp; from scipy.optimize import brentq; print(inspect.signature(solve_ivp)); print(inspect.signature(brentq))"
# PCK constants used by the preset and the tests: bodvrd('EARTH','PM'|'POLE_RA'|'POLE_DEC'|'GM')
# benchmarks (in session; scratchpad scripts): FH cost per evaluation at the GEO range for
# D = 10/30/100 m and R_rx = 0.1/1/5 km; BLAS-thread scaling (1/4/8 threads: 26.0/24.8/25.7 s);
# Chebyshev-table accuracy on GEO-like and heliocentric geometries; GEO preset drift over 1-365 d;
# lunisolar secular rate and one-year Earth+Sun+Moon integration (0.839 vs 0.849 deg)
uv run helioswarm validate-config configs/comparison/{helio_1au,geo}_{optical,microwave}.yaml
uv run helioswarm validate-config configs/comparison/*_sweep.yaml
# sizing runs (in session, one year, 1 thread): helio_1au_optical 70 s / 493 MB,
# helio_1au_microwave 78 s / 493 MB, geo_microwave 32 s / 500 MB (FH table at that time for
# all four; the heliocentric cases were then switched to the far-field models, see D)
uv run helioswarm run configs/comparison/<case>.yaml --out <scratch>/full_<case> --overwrite
# Slurm (outputs $HELIOSWARM_RUNS_DIR/comparison/<sweep name>; slurm/comparison_submit.sh)
sbatch --parsable -J helioswarm-cmp-geoopt -p shared --array=0-4 -c 1 --mem=1500M -t 0-08:00 \
    slurm/sweep_array.sbatch configs/comparison/geo_optical_sweep.yaml $HELIOSWARM_RUNS_DIR/comparison/cmp_geo_optical
sbatch --parsable -J helioswarm-cmp-geomw -p shared --array=0-2 -c 1 --mem=800M -t 0-00:08 \
    slurm/sweep_array.sbatch configs/comparison/geo_microwave_sweep.yaml $HELIOSWARM_RUNS_DIR/comparison/cmp_geo_microwave
sbatch ... -J helioswarm-cmp-helioopt -p shared --array=0-2 -c 1 --mem=800M -t 0-00:15 ... helio_1au_optical_sweep.yaml
sbatch ... -J helioswarm-cmp-heliomw  -p shared --array=0-2 -c 1 --mem=800M -t 0-00:15 ... helio_1au_microwave_sweep.yaml
sbatch ... --array=5 -t 0-00:35 --export=ALL,SWEEP_ARRAY_COUNT=9 ... helio_1au_optical_sweep.yaml   # timed-out point p0005
sbatch ... --array=8 -t 0-01:40 --export=ALL,SWEEP_ARRAY_COUNT=9 ... helio_1au_optical_sweep.yaml   # timed-out point p0008
timeout 540 bash -c 'while squeue -h -j <ids> | grep -q .; do sleep 30; done'; sacct ...; seff <jobid>
uv run helioswarm compare configs/comparison/comparison.yaml \
    --runs-dir $HELIOSWARM_RUNS_DIR/comparison --out $HELIOSWARM_RUNS_DIR/comparison/report
uv run ruff check . ; uv run ruff format --check . ; uv run mypy src ; uv run pytest -q
```

The NASA report: `curl` of https://www.nasa.gov/organizations/otps/space-based-solar-power-report/
and of the PDF it links (title page and p. ii read; file not kept).

## D. Tests and results

Final gate (same node, after all source changes, including the review fixes):

```text
$ uv run ruff check .          -> All checks passed!
$ uv run ruff format --check . -> 116 files already formatted
$ uv run mypy src              -> Success: no issues found in 39 source files
$ uv run pytest -q             -> 1267 passed, 2 skipped in 348.92s (0:05:48)
$ uv run pytest -q -m "not slow" -> 1256 passed, 2 skipped, 10 deselected in 194.84s (0:03:14)
SKIPPED [1] tests/integration/test_constellation.py:544: JAX_PLATFORMS=cpu: no GPU requested (run by slurm/gpu_bench.sbatch)
SKIPPED [1] tests/unit/test_capture_batch.py:139: JAX_PLATFORMS=cpu: no GPU requested (run by slurm/gpu_bench.sbatch)
```

(Before the review: 1257 passed, 2 skipped in 320.70 s. The review added 10 tests: the 9 listed
in "Review outcomes" and `test_comparison_configs.py::test_flux_limit_sensitivity_differs_from_geo_optical_only_in_the_bounded_keys`;
the `not slow` timing above was taken before that last ~0.1 s test was added.)

The suite grew from 1186 (end of M7) to 1259 collected tests: 65 in the five new files, one
new metrics test, one new required-field case and the six engineering-rule checks
(docstrings, no global mutable state) parametrised over the three new modules. Three existing tests
were updated for intended behaviour changes (EARTH is now a valid state origin; the
required-field message of the now schema-optional orbit elements; the aliasing warning needs
an access edge). Every SPICE, ITU-R and cross-check test ran.

### New tests and what they discriminate

| Acceptance item | Test(s) | Tolerance / why it can fail |
|---|---|---|
| GEO period = IAU_EARTH spin period; documented radius | `test_geo_baseline.py::test_geostationary_period_equals_the_iau_earth_spin_period` | 1e-12 relative against `dW/dt + dα₀/dt sin δ₀` from `bodvrd`; `dW/dt` alone (4.9e-8) or the ERA rate (6.6e-8) fail; radius `(GM/ω²)^(1/3)` to 1e-13 and 42 164 172.6 m ± 1 m; elements circular with `2π/ω` period |
| Sub-satellite point fixed | `::test_sub_satellite_point_stays_fixed_over_ten_days` (0°, −75.5°, 137° E) | longitude 1e-6° over 10 days via independent `pxform` (a 1e-9 rate error drifts 3.6e-6°); latitude 2e-4° (IAU pole motion ≤ 1.5e-4°); radius constant 1e-4 m |
| Elevation ~90° / range r_GEO − R_E at the sub-satellite site | `::test_geo_elevation_and_range_at_the_sub_satellite_site` | elevation deficit = ωτ to 5e-8 rad, azimuth 270° ± 2° (an instantaneous geometry gives 0 and fails); range = `L/(1 − u·v_site/c)` to 1 m with `L = r_GEO − a_WGS84` (ignoring the first-order term errs by up to 3.6 km) |
| Eclipse seasons, durations, penumbra | `::test_geo_eclipse_durations_match_the_analytic_shadow_geometry` (03-20, 04-04, 04-25, 09-23) | umbra and total shadow durations vs brentq solution of the closed-form shadow geometry, 15 s (5 s sampling); no shadow outside the season; penumbra > 200 s; a cylindrical shadow (+128 s umbra) or no penumbra fail |
| Lunisolar perturbation size (omission documented) | `::test_lunisolar_inclination_drift_matches_the_averaged_torque` | numerical Earth+Sun+Moon tilt after one year vs secular rate × 1 yr, 5 % (measured 1.2 %); rate in 0.75–0.95°/yr; no third bodies → tilt < 1e-12 |
| Fresnel regime selection | `test_geo_comparison_run.py::test_fresnel_regime_selection_at_the_geo_range` | `airy_far_field` raises `BeamModelValidityError` at the GEO range; FH run N_F = D²/(λR) per epoch to 1e-15; microwave N_F = 0.2284 and FH = far field to 1e-3 |
| GEO through the identical chain | `::test_geo_link_through_the_chain_at_equinox` | ECLIPSE epochs 60–72 of 1441 (two ~67 min umbras at 120 s), P_grid = 0 there, availability = no eclipse, incident sunlight in penumbra = S₁ₐᵤ(au/d)²A·f to 1e-12, link budget P_grid/P_PV = η-chain × capture × atmosphere to 1e-12 |
| `compare` from saved runs | `::test_compare_writes_tables_from_the_saved_runs`, `::test_compare_refuses_a_stale_run_directory` | tiny real sweeps; statuses (ok/failed), values equal to independently read summaries, mass proxy closed form 1e-14, files and manifest hashes, CLI exit codes 0/1; a changed shared block → refused as stale |
| Shared assumptions literally equal | `test_comparison_configs.py::test_resolved_configs_differ_only_in_orbit_and_geometry` (both technologies), `::test_harmonisation_check_catches_a_technology_difference`, `::test_load_comparison_refuses_an_unharmonised_case` | leaf-by-leaf flat dictionaries: differing keys = orbit.*, project.*, time.integrator_step_s, models.beam(_evaluation) exactly; a changed conversion efficiency is caught |
| `extends` mechanism | `test_config_m7b.py::test_extends_reproduces_the_flat_configuration`, `::test_merge_config_mappings_merges_mappings_and_replaces_everything_else`, `::test_extends_errors` | a two-level chain reproduces the MVP hash exactly; cycles, bad values, missing files refused |
| Heliocentric configs unaffected | `test_config_validity.py::test_mvp_config_hash_is_pinned` (unchanged pin), `test_config_m7b.py::test_mvp_canonical_json_is_the_m7_text`, `::test_new_fields_are_omitted_from_the_canonical_form_at_their_default` (7 fields), `test_comparison_configs.py::test_heliocentric_mvp_configs_carry_none_of_the_m7b_keys`; the whole M6/M7 suite (end-to-end MVP numbers) passes unchanged | exact hashes |
| Geocentric schema rules | `test_config_m7b.py::test_geocentric_rules` (9 cases) and three more | exact messages |
| Chebyshev table | `test_beam_table.py` (7) | nested nodes bitwise, polynomial exactness 1e-12, tail = last coefficient, GEO-like table ≤ 5 × 5 nodes and 1e-9 at independent epochs, wide elevation refined (≥ 9 nodes) and 1e-9, a reference with a jump refused |

### Slurm jobs and `seff`

Wall time and peak RSS from `/usr/bin/time -v` in each task log; CPU and memory efficiency
from `seff`. All tasks 1 core on `shared` (the test partition's 5-job QOS was in use by other
work). A task exits 1 when one of its points failed (recorded as `error.json`), so `sacct`
shows FAILED for tasks with an expected validity failure.

| Job | Points (task) | Request | State | Wall | Peak RSS | `seff` CPU / memory |
|---|---|---|---|---|---|---|
| 60309248_0 GEO optical | p0000 ok, p0005 failed (paraxial) | 1500M, 8 h | FAILED (expected) | 0:45 | 536 MB | 52 % / 26 % |
| 60309248_1 GEO optical | p0001, p0006 ok | 1500M, 8 h | COMPLETED | 9:52 | 657 MB | 96.5 % / 37 % |
| 60309248_2 GEO optical | p0002 failed (paraxial), p0007 ok | 1500M, 8 h | FAILED (expected) | 2:31:41 | 529 MB | 99.1 % / 29 % |
| 60309248_3 GEO optical | p0003 ok, p0008 failed (paraxial) | 1500M, 8 h | FAILED (expected) | 0:55 | 535 MB | 66 % / 16 % |
| 60309248_4 GEO optical | p0004 ok | 1500M, 8 h | COMPLETED | 14:54 | 503 MB | 97.6 % / 21 % |
| 60309192_0…2 GEO microwave | 3 points each, all ok | 800M, 8 min | COMPLETED | 1:44–2:12 | 737–771 MB | 60–70 % / 75–89 % |
| 60309181 / 60309188 (first heliocentric submission) | — | — | cancelled / superseded | — | — | the Fresnel-Hankel model failed its paraxial check for the 500 m microwave aperture at 0.5 au and two optical tables missed their tolerance; the heliocentric cases were switched to the far-field models (A) and the output directories deleted before resubmission |
| 60310868_0…2 heliocentric microwave | 3 points each, all ok | 800M, 15 min | COMPLETED | 5:50–8:17 | 787–850 MB | 89–92 % / 71–98 % (task 0: 781 MB sampled of 800 MB: use 900M) |
| 60310851_0, _1 heliocentric optical | p0000/3/6, p0001/4/7 ok | 800M, 15 min | COMPLETED | 4:32, 11:28 | 612, 624 MB | 92 %, 93 % / 68 %, 70 % |
| 60310851_2 heliocentric optical | p0002 ok, p0005 interrupted | 800M, 15 min | TIMEOUT | 15:21 | 462 MB (seff) | 96 % / 58 % |
| 60313986_5, 60313993_8 heliocentric optical (one point per task) | p0005, p0008 ok | 800M, 35 min / 1 h 40 min | COMPLETED | 11:06, 20:13 | 488, 490 MB | 98.4 %, 97.9 % / 35 %, 25 % |

Per-point cost: GEO microwave ~35 s (ITU-R atmosphere at every epoch ~12 s); heliocentric
~75–85 s plus the per-epoch beam (Airy with a 10 km horizontal receiver: 5–20 min); GEO
optical 0.1–2.5 h, all in the ~33 Fresnel-Hankel evaluations of the table (one evaluation:
0.14 s at D = 10 m, R_rx = 100 m up to 268 s at 100 m, 1 km). The resumable sweep re-ran only
the interrupted point after the timeout.

### GPU decision (user rule; PDF §14.3)

Profiling showed one hot spot, the Fresnel-Hankel evaluation of the GEO optical link (0.14 s to
268 s per epoch on one core; BLAS threads give no speed-up because the cost is in the
single-threaded Bessel-function evaluations). The batched JAX kernel of
`capture_batch.py` covers only the Airy far field, so it cannot take this work, and the
Chebyshev table reduces it from 32 401 epochs to 9–15 node evaluations plus ~17 checks per run
(a factor ~1000), leaving at most ~2.5 h on one CPU core for the largest point. A GPU port of
the Fresnel-Hankel quadrature is therefore not justified for M7b; no GPU was requested.

## E. Assumptions and equations implemented

Full text: `docs/model_assumptions.md`, "M7b: GEO baseline and harmonised comparison".

- Geocentric L0: Kepler about the Earth's centre, `mu = GM_earth` (DE440), J2000 equatorial
  elements; `r_SSB = r_geo + r_earth(t)` (Galilean origin shift).
- Preset: `[ω]× = (dR/dt)Rᵀ`, `ω = ω_vec·p`, `r = (GM/ω²)^(1/3)`, `v = ω p × r`.
- Perturbation budget: secular `dĥ/dt = −(3GM_k/(2n|s_k|⁵))(ĥ·s_k)(ĥ×s_k)`; geocentric
  third-body equation `r̈ = −GM_E r/|r|³ + Σ GM_k[(s_k − r)/|s_k − r|³ − s_k/|s_k|³]`; J2
  synchronism `ω²r = (GM/r²)(1 + 1.5 J2 (R/r)²)`; SRP eccentricity circle
  `e_r = 1.5 a/(n r n_sun)`.
- Shadow geometry (test reference): `cos θ = (r + D cos β cos φ)/√(D² + r² + 2Dr cos β cos φ)`,
  umbra `θ ≤ ρ_E − ρ_S`, shadow `θ < ρ_E + ρ_S`.
- Apparent zenith tilt ωτ; range `L/(1 − u·v_site/c)`.
- Chebyshev-Lobatto table, barycentric interpolation (Berrut & Trefethen 2004), refinement by
  the last coefficient (Trefethen 2013), validation at two disjoint check sets.
- Comparison metrics and the M7 mass proxy (compare module docstring).
- Named choices: L0 + station-keeping (GEO), beam model by propagation regime, clear sky,
  one receiver site for both, the heliocentric satellite in the favourable +30° slot.

## F. Known limitations and next milestone

- **One satellite, one site, one year, clear sky.** The comparison is per satellite on the
  same receiver site; network effects (several sites for the heliocentric case, GEO slot and
  spectrum limits, constellations of either) are outside it. Clear sky all year is a weather
  upper bound for both (ERA5 availability comes with M8a).
- **Scenario choices that favour each side**: the GEO satellite is directly above an
  equatorial site (zenith link, the most favourable GEO geometry); the heliocentric satellite
  is in the most favourable slot of the M7 ring (+30°, 0.48–0.55 au). Other placements of
  either are worse; a mid-latitude site would lower the GEO elevation (not run).
- **L0 + station-keeping for GEO**: the perturbations are quantified, not simulated in the
  run; station-keeping propellant and mass are outside the mass proxy. The numerical
  Earth + Sun + Moon propagator is a comparison path only (no J2, no SRP in it).
- **Lifetime, degradation and cost are not modelled** for either architecture (equal by
  construction over the one-year window): the PDF §16 harmonisation row is therefore
  **partial** (→ M8a); delivered power is harmonised as the same `P_grid` system boundary, not
  as equal delivered power (mass per delivered kW is the normalised metric).
- **Unmodelled GEO environment terms**: Earth albedo and IR on the GEO collector/radiator
  (≤ ~15 W/m², ~1 % of S; ≤ ~1 K PV temperature, ≤ 0.1 % P_grid), SRP-driven station-keeping
  (±365 km / ±1° uncontrolled at the comparison A/m): `docs/model_assumptions.md` M7b.
- **Receiver flux and safety limits are not configured** (idealised; labelled): the GEO
  optical peak flux (up to 7.65e4 W/m²) would make such a limit bind for GEO only. The
  bounded sensitivity run (assumed 1e4 W/m², A) quantifies it; a *sourced* laser-PV / eye- or
  skin-safety limit is still missing, and the beam-abort policy (no throttling) is the
  model's. Receiver Sun-avoidance is not configured either.
- **Non-paraxial propagation** is not implemented: the three GEO-optical points with a 10 km
  receiver fail the Fresnel paraxial check and are reported as failed (capture bounded below
  by the 1 km points).
- **Chebyshev beam table**: validated per run at check epochs, not at every epoch; a function
  that is smooth at the checks but not between them would escape (the tail estimate and two
  disjoint check sets make this unlikely; independent spot checks agreed to ~1e-14).
- **Runtime**: the full suite (slow tests included, as the gates run it) takes 5.6–5.9 min on
  this node (target < 3 min; M7 was 4.9 min). After the review the M7b tests that need more
  than a few seconds are marked `slow` (one-year lunisolar integration, one-year eclipse
  seasons, the shared tiny comparison, the three Fresnel-Hankel beam-table tests), and the
  `-m "not slow"` subset runs in 3.2 min (1256 tests).
- **Slurm sizing**: the first heliocentric optical submission (3 round-robin tasks, 15 min)
  timed out on the 10 km points (11–20 min each with the per-epoch Airy reference) and the
  heliocentric microwave task peaked at 781 MB of 800 MB. `slurm/comparison_submit.sh` now
  encodes the measured sizing (one point per task for both optical sweeps, GEO p0007 alone
  with 3 h, memory = measured peak + ~15 %); the review rerun with it completed every point
  (D, "Review outcomes").

Next: M8a (ERA5 weather, parameterised economics: heliocentric vs GEO under one economic
boundary, which adds the lifetime and cost terms this comparison leaves out), then M8.

## Review outcomes

Seventeen findings from the independent M7b review (1 major, 16 minor). Each was verified
before acting; none was refuted outright (one suggested remedy was declined, finding 7).

| # | Finding (file) | Disposition | What was done / evidence |
|---|---|---|---|
| 1 | `satellite_initial_state`/`initial_state_from_config` silently build a heliocentric state (GM_sun, Sun-relative normal) for a geocentric config (`propagate.py`) | **fixed** | Reproduced (velocity off by ~1.8e6 m/s). Both now raise `ValueError` for `elements_center != SUN`, pointing to `geocentric.geocentric_l0_trajectory`; tested in `test_geo_baseline.py::test_configured_geocentric_elements_follow_kepler_about_the_earth`. |
| 2 | Earth albedo/IR on the GEO collector not named; the M4 bound is heliocentric only (`model_assumptions.md`) | **fixed** | Bound recomputed: (R⊕/d)² = 0.0229, ≤ 9.3 + 5.5 ≈ 15 W/m² (1.1 % of S); upper-bound effect ≈ +1.2 K PV, ≤ 0.1 % P_grid, radiator ≲ +2 %. Added to M7b "Neglected effects"; M4 text now says its bound is heliocentric only. |
| 3 | GEO eclipse fractions quoted to 4 significant figures without sampling error; lunar shadowing not mentioned (`M7b.md`) | **fixed** | Reproduced exactly (974 s: 0.0088886 / 0.0012963; 60 s: 0.0088086 / 0.0014450). Sampling uncertainty derived (±1.7 % umbra, ±12 % penumbra relative) and documented with the converged 60 s values; lunar partial shadowing (0.028 % of the year) documented; the 60 s yearly fractions are now a slow regression test against the closed-form geometry (finding 6). The tables keep the run values (974 s), labelled. |
| 4 | SRP budget used A/m = 0.02 although the comparison satellite has ≈ 0.60 m²/kg; heliocentric SRP setting inconsistent with the mass proxy; J2/SRP functions untested (`geocentric.py`, docs) | **fixed** | Confirmed: e_r = 8.7e-3, ±365 km, ±0.99° (not 2.9e-4/±12 km/±0.033°); docs, compliance row and the run approximation text corrected. Closed-form unit tests `tests/unit/test_geocentric.py` (J2 first order 522.2 m, SRP at both A/m). The heliocentric configs keep A/m = 0.02 (PDF §5 L1 dynamics), documented as not reconciled with the post-processing proxy; quantified by a one-year run at A/m = 0.6: P_grid +1.1 % (8.854 vs 8.758 kW), range −0.65 %. |
| 5 | Six comparison-table metrics unasserted; tiny comparison outside eclipse season (`test_geo_comparison_run.py`) | **fixed** | Tiny comparison moved to 2030-03-20T23:00 + 3 h at 120 s (crosses the equinox umbra); umbra/penumbra fractions, capacity factor, end-to-end efficiency, peak irradiance and mean intercepted flux are recomputed from the saved `timeseries.parquet` with independent trapezoid weights (1e-12), plus non-trivial-value checks for GEO. |
| 6 | No test of configured (non-preset) geocentric elements, of off-nadir GEO geometry, or of season boundaries / yearly figures (`test_geo_baseline.py`) | **fixed** | `test_configured_geocentric_elements_follow_kepler_about_the_earth` (a = 42 164 km, i = 30°: (a,0,0) → (0, a cos i, a sin i) → (−a,0,0) → (a,0,0) to 1 mm, circular speed); `test_geo_elevation_and_range_from_an_off_nadir_site` (γ = 40°: el = atan2(cos γ − R/r, sin γ) to 3e-5 rad, azimuth 270° ± 0.01°, range L/(1 − u·v/c) to 1 m); `test_geo_eclipse_seasons_over_2030_match_the_analytic_geometry` (slow: seasons 02-26..04-12 and 08-31..10-16, 93 days, umbra and penumbra fractions to 4e-5 against the per-day closed form). |
| 7 | Tolerance reason "O(N_F²) ~ 1e-3" wrong for the GEO microwave FH-vs-far-field check (N_F² = 0.052) (`test_geo_comparison_run.py`) | **fixed** (reason); suggested model switch **declined** | Measured difference 1.1e-4 confirmed; the comment now states it and the 9× margin, and the test also requires the difference to be resolved (> 1e-6). Switching `geo_microwave` to the far-field model was declined: the approved plan requires the Fresnel-Hankel model explicitly in the GEO configs, both microwave models are valid there, and the 1.1e-4 capture difference is documented and below every reported digit of the comparison. |
| 8 | Slow M7b tests not marked; suite > 3 min (`test_beam_table.py` etc.) | **fixed** | The three Fresnel-Hankel beam-table tests, the shared tiny comparison (now one module-scoped fixture instead of two builds) and the lunisolar integration are marked `slow`; the refusal test uses `max_nodes=5` (13 s → 2 s). |
| 9 | **Major**: `slurm/comparison_submit.sh` (the documented regeneration path) cannot finish: 15 min / 800M round-robin tasks, manual resubmissions not in the script | **fixed and validated** | Script rewritten with the measured sizing: one point per task for both optical sweeps (`SWEEP_ARRAY_COUNT=9`), GEO p0007 alone with 3 h, memory = measured peak + ~15 %, plus the sensitivity sweep. All comparison points were regenerated with it (job table below): every task completed within its limit (helio optical p0008 20:30 of 30 min, GEO p0004 14:38 of 30 min); no manual step besides partition moves made while the shared queue was congested. `helioswarm compare` exit 0, tables identical to the first runs. |
| 10 | §16 row marked satisfied although lifetime and cost are not modelled; "delivered power" not harmonised as a quantity (`compliance_matrix.md`) | **fixed** | Row now **partial (lifetime, cost → M8a)**, states that delivered power is harmonised as the same `P_grid` boundary (704.9 vs 8.76 kW) with mass per delivered kW as the normalised metric; header note and `model_assumptions.md` updated. |
| 11 | `approximations_in_force` does not name "L0 + station-keeping" for GEO; GEO manifests list no approximation (`config.py`) | **fixed** | New `NamedApproximation` for `orbit.elements_center = EARTH` (lunisolar, J2, SRP magnitudes; propellant outside the mass proxy; comparison path `geocentric.propagate_geocentric_point_masses`); the regenerated GEO manifests and reports carry it; test `test_config_m7b.py::test_geocentric_orbits_name_the_station_keeping_approximation` (the MVP still names none). |
| 12 | Code of the comparison runs not traceable (dirty tree; `report.py` changed during the arrays; quoted tree hashes recorded nowhere) (`M7b.md`) | **fixed** | Confirmed (`report.md` key-parameter lists differed between points of one array). Every run manifest now records `source_sha256` (layout `/3`; `test_end_to_end.py` checks it), the provenance statement at the top was corrected, and all comparison points were regenerated from one tree (`source_sha256 = c2be65c6…06a9` in every manifest). The regenerated `summary.json` files are bit-identical to the first runs. |
| 13 | The FH vs far-field agreement at heliocentric range (8.7e-8 / 5.3e-10) was unreproducible (`_shared.yaml`, `M7b.md`) | **fixed** | New test `test_far_field_and_fresnel_hankel_agree_at_the_heliocentric_range` (both technologies, 7.2e10 and 8.2e10 m, zenith and 30°): measured −9.9e-8/−7.6e-8 (optical), −6.0e-10/−4.6e-10 (microwave), tolerance 1e-5. The unreproducible year-integrated numbers were withdrawn and replaced by the tested ones; `_shared.yaml` cites the new test. |
| 14 | Harmonised-assumption Markdown table rounds configured values to 4 significant figures (365.25 → 365.2) (`compare.py`) | **fixed** | Configuration values are printed with `repr` (`_fmt_exact`); metric tables keep 4 figures. The compare test checks every numeric Markdown cell equals the CSV value; the regenerated report shows `365.25` and `0.12236426857142857`. |
| 15 | Docstring and error message cite the non-existent `propagate.geostationary_initial_state` (`config.py`) | **fixed** | Both now cite `helioswarm.geocentric.geostationary_orbit`. |
| 16 | Suite runtime past the 3 min budget; M7b tests not marked slow (`test_geo_comparison_run.py`) | **fixed** (partly; see F) | Slow marks as in 8; the `-m "not slow"` subset runs in 3.2 min. The full gate including slow tests takes 5.8 min (it was 4.9 min before M7b), as the gates run slow tests by rule. |
| 17 | No receiver flux / beam-abort / heat limit; one shared receiver-limit block for laser-PV and rectenna (`_shared.yaml`) | **fixed** (bounded); sourced limit **deferred** | Receiver limits moved into `_optical.yaml` / `_microwave.yaml` with technology-specific statements (same values: resolved configs and hashes unchanged). A bounded sensitivity sweep with an **assumed, illustrative** 1e4 W/m² laser-PV limit (`configs/comparison/sensitivity/`) quantifies the effect (A: GEO D = 10 m drops to 13.8 W, D = 30/100 m unchanged; best GEO mass/kW 0.068 t/kW). A *sourced* laser-PV or eye/skin-safety value is still not available in the project and is deferred (it must be cited, not invented). |

Review rerun (`bash slurm/comparison_submit.sh` after the fixes; source tree `c2be65c6…`; all
on 1 core; wall and peak RSS from `/usr/bin/time -v`, CPU/memory from `seff`). The shared
queue was congested (> 1 h pending), so four arrays were moved to the free `test` partition
with the same per-point sizing (GEO microwave, heliocentric microwave and the sensitivity as
one task each with `SWEEP_ARRAY_COUNT=1`, GEO p0007 as its own task):

| Job | Points | Request | State | Wall | Peak RSS | `seff` CPU / memory |
|---|---|---|---|---|---|---|
| 60340569_0,1,3,4,6 GEO optical (shared) | p0000, p0001, p0003, p0004, p0006 ok | 750M, 30 min | COMPLETED | 0:36, 2:52, 1:02, 14:38, 7:18 | 501–518 MB | 53–98 % / 34–58 % |
| 60340569_2,5,8 GEO optical (shared) | p0002, p0005, p0008: Fresnel paraxial check failed (expected) | 750M, 30 min | FAILED (expected) | 0:28–0:29 | 224–264 MB | — |
| 60348695_7 GEO optical (test) | p0007 ok | 1G, 4 h | COMPLETED | 1:46:39 | 529 MB | 99.1 % / 33 % |
| 60340573_0…8 heliocentric optical (shared, one point per task) | all 9 ok | 720M, 30 min | COMPLETED | 1:34–20:30 (p0008) | 466–516 MB | 75–98 % / 23–58 % |
| 60362334_0 heliocentric microwave (test, 9 points) | all 9 ok | 960M, 35 min | COMPLETED | 10:57 | 897 MB | 95.6 % / 76 % |
| 60359278_0 GEO microwave (test, 9 points) | all 9 ok | 880M, 12 min | COMPLETED | 3:23 | 915 MB | 81.3 % / 84 % |
| 60359285_0 GEO optical flux-limit sensitivity (test, 3 points) | all 3 ok | 750M, 20 min | COMPLETED | 5:56 | 689 MB | 94.4 % / 79 % |
| 60340579, 60340605, 60340606 (shared) | — | — | cancelled while pending (moved to test, above) | — | — | — |
