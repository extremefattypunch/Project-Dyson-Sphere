# Model assumptions

This file lists every equation, assumption, validity range and simplification used by
`helioswarm` (PDF §14.4). Each physics milestone appends its section. Nothing listed here is a
validation claim; validation outcomes live in `docs/validation_report.md`.

## Conventions (M0)

- **Units.** All computation is in SI: m, s, kg, W, rad, K. Fixed physical constants live only
  in `helioswarm.constants` (M1), each with its source. The astronomical unit, degrees, days
  and dB appear only at the configuration/IO boundary, in field names that carry the unit
  (`semimajor_axis_au`, `inclination_deg`, `duration_days`, `edge_taper_dB`). Conversions:
  - 1 au = 149 597 870 700 m exactly (IAU 2012 Resolution B2);
  - 1 day = 86 400 s exactly;
  - degrees to radians uses `math.radians`.
- **Frames and origin.** Internal inertial states use the SPICE frame `J2000` with origin
  `SOLAR SYSTEM BARYCENTER` (`frames` section). Orbital elements are Sun-centred and referred to
  `ECLIPJ2000` (`orbit.elements_frame`, `orbit.elements_center`). The Earth-fixed frame is
  `IAU_EARTH` (text PCK rotation model) and the reference ellipsoid is WGS84. These are the only
  accepted values in schema_version 1.
- **Time scales.** Configuration epochs are UTC strings (`*_utc`) in canonical ISO-8601 form.
  Conversion to TDB happens only through SPICE (`helioswarm.timescale`, M2). UTC and TDB are
  never mixed implicitly.
  Leap-second epochs (`23:59:60`) are rejected as configuration epochs.
- **Output time grid.** `duration_days × 86400` must be an integer multiple of `output_step_s`,
  so the grid ends exactly at the window end. The check is done in seconds:
  `|duration_s − n·output_step_s| ≤ 8 ε · duration_s` with `n = round(duration_s/output_step_s)`
  and ε the float64 machine epsilon. The tolerance absorbs only float rounding (≤ 0.7 µs for the
  longest allowed window of 100 years).
- **Integrator step.** `time.integrator_step_s` is the *maximum* step of the adaptive L1
  integrator (`scipy.integrate.solve_ivp(max_step=…)`). It is required for the L1 model and
  rejected with `two_body_kepler`. The integration tolerances `integrator_rtol`,
  `integrator_atol_position_m` and `integrator_atol_velocity_m_s`, and the ephemeris-cache
  spacing `ephemeris_cache_step_s`, were added in M2 (see the M2 section).

## Parameter definitions fixed by the configuration schema (M0)

These definitions are fixed now so that later milestones cannot double-count a loss. Each
efficiency has exactly one owner (PDF §10.1):

| Stage | Equation | Config field |
|---|---|---|
| Absorbed solar power | `P_abs = S(r) · A · α · cos(incidence)`, with `S(r) = S_1AU (au/r)²` | `collector.area_m2`, `collector.absorptivity`, `constants.solar_constant_1au_W_m2` |
| PV output | `P_PV = P_abs · η_PV(T)` (η is defined relative to **absorbed** power, as in PDF §8.1) | `collector.pv_efficiency_*` |
| PV efficiency law | `η_PV(T) = clip(η_ref (1 + γ_T (T − T_ref)), η_min, η_max)`: a labelled local linear approximation (PDF §8.2) | `collector.*` |
| Power conditioning | `P_dc_to_tx = P_PV · η_pc` | `transmitter.power_conditioning_efficiency` |
| DC to beam | `P_beam = P_dc_to_tx · η_e2b` | `transmitter.electrical_to_beam_efficiency` |
| Transmitter optics/feed | `P_aperture = P_beam · η_optics_or_rf`. This is a separate physical loss and not an "aperture efficiency", because beam models are normalised to aperture power. Spill-over of source power outside the aperture edge (e.g. `exp(−2a²/w0²)` for a Gaussian laser clipped at radius `a`) belongs here; no beam model applies it again. | `transmitter.optics_or_rf_efficiency` |
| Atmosphere | `P_atm = P_aperture · η_atm`; zenith scenario factors scaled with the plane-parallel air mass `X = 1/sin(el)`, or ITU-R P.618 slant-path attenuation (`itu_r`, M5) | `atmosphere.*`, `models.atmosphere` |
| Capture | `P_captured = P_atm · η_capture`; `η_capture = BeamResult.capture_fraction` (diffraction × pointing × wavefront × lattice factor, M5) | `models.beam`, `transmitter.*`, `receiver.radius_m`, `receiver.orientation` |
| Receiver input limit | `P_accepted = min(P_captured, P_max_in, Q_max/(1 − η_rx))`; the excess is reported as receiver saturation (M6: `receiver_clipped_W`, flag `RECEIVER_SATURATION`, link still available). The limit acts on captured beam (optical/RF) power, **before** conversion. | `receiver.max_input_power_W`, `receiver.max_heat_rejection_W` |
| Receiver conversion | `P_receiver = P_accepted · η_rx` (equal to `P_captured · η_rx` while the receiver is not saturated) | `receiver.conversion_efficiency` |
| Receiver waste heat | `Q_rx = P_accepted · (1 − η_rx)` must not exceed the receiver's thermal rejection capability (not enforced when unset). Not a power stage; M6 enforces it as the second bound on `P_accepted` above. | `receiver.max_heat_rejection_W` |
| Power-density beam abort (M6) | if `P_atm · I_peak/W > max_power_density_W_m2` the transmitter is not operated at that epoch (all stages from `P_PV,tx` on are 0; reason `POWER_DENSITY_ABORT`). The schema defines abort only, no throttling. | `receiver.max_power_density_W_m2` |
| Pointing availability (M6) | link unavailable where `E[η]/η(perfect pointing) < min_pointing_efficiency` | `transmitter.min_pointing_efficiency` |
| Grid conditioning | `P_grid = P_receiver · η_grid` | `receiver.grid_efficiency` |

The stage order is copied from PDF §10.1 exactly. It applies the atmosphere factor before the
capture factor. Both are multiplicative scalars, so their order does not change `P_grid`.

Further definitions:

- **Radiator.** `Q_rad = ε σ A_eff (T_rad⁴ − T_space⁴)` with
  `A_eff = A_physical · radiating_sides · radiator_view_factor`.
  - `radiating_sides` is defined geometrically: the number of faces of a thin flat plate that see
    space.
  - `T_space` defaults to the CMB temperature.
  - The schema rejects `radiator_temperature_limit_K ≤ space_temperature_K`.
- **Wavefront error.** The loss factor is `exp(−(2π σ_w / λ)²)`, where `σ_w` is the RMS
  **wavefront** (path-length) error. It is not the reflector surface error, which would carry an
  extra factor of 2.
- **Pointing.** `pointing_sigma_rad` is the per-axis 1σ of an isotropic 2-D Gaussian jitter.
  `pointing_bias_rad` is a separate deterministic offset. Both use a small-angle model and are
  bounded by 0.1 rad.
- **Far-field validity.** Far-field beam models must raise an error when the Fresnel number
  `N_F = D²/(λ R)` exceeds `models.fraunhofer_max_fresnel_number`. The default of 0.5 is
  equivalent to the conventional far-field distance `R ≥ 2D²/λ` (IEEE Std 145-2013, "far-field
  region"; Balanis, *Antenna Theory*, ch. 2 "Field Regions").
- **Aperture illumination and the Gaussian-beam model.** `transmitter.aperture_illumination` is
  the field across the aperture, zero outside its edge `a = D/2`; `P_aperture` is the power in
  that field. `models.beam = gaussian_beam` propagates an *untruncated* Gaussian of waist `w0`,
  normalised to `P_aperture`. It approximates the truncated field and is accepted only when
  `a/w0 ≥ models.gaussian_beam_min_truncation_ratio` (default 2.5). With `r = a/w0`, the
  Fraunhofer on-axis intensity per unit aperture power of the truncated field is
  `(1 − e^{−r²})² / (1 − e^{−2r²})` times the untruncated value. That factor is 0.9961 at
  `r = 2.5` (the untruncated model overstates peak irradiance by 0.4 %; the clipped fraction is
  `e^{−2r²}` = 3.7 × 10⁻⁶), 0.964 at `r = 2` and 0.46 at `r = 1`. More strongly truncated
  apertures must use `fresnel_hankel`. The waist must also satisfy `w0 ≤ a` for every beam model.
- **Atmospheric air mass.** The `scenario` atmosphere scales zenith factors with the
  plane-parallel air mass `X = 1/sin(el)`. X diverges at the horizon and exceeds the Kasten &
  Young (1989, Appl. Opt. 28, 4735) air mass by 3.1 % at 10°, 11 % at 5° and 26 % at 3°. With
  `models.atmosphere = scenario` the schema therefore requires
  `receiver.min_elevation_deg ≥ atmosphere.plane_parallel_min_elevation_deg` (default 10°).
- **Settings that a model choice would ignore are rejected.** Examples: `weather_scenario =
  cloudy` with `atmosphere = none`; `orbit.cr` or `orbit.area_to_mass_m2_kg` without SRP;
  `time.integrator_step_s` with `two_body_kepler`; `gaussian_waist_radius_m`, `edge_taper_dB` or
  `fixed_normal_unit_j2000` without the mode that uses them. Parameters that have a non-`None`
  default, such as the whole `atmosphere` section under `atmosphere = none`, cannot be told apart
  from user input and are tolerated.

## Optimistic defaults (explicitly labelled)

Several schema defaults make a stage lossless. They are upper bounds, tagged `assumed`, and are
listed by `helioswarm validate-config --show-provenance`:

- `transmitter.power_conditioning_efficiency = 1.0`
- `transmitter.optics_or_rf_efficiency = 1.0`
- `transmitter.rms_wavefront_error_m = 0.0`
- `atmosphere.optical_zenith_transmittance_clear = 1.0`
- `atmosphere.microwave_zenith_attenuation_*_dB = 0.0`
- `collector.pv_efficiency_max = 1.0`, which is only the energy-conservation bound
- `receiver.max_power_density_W_m2 = None`, `receiver.max_heat_rejection_W = None` and
  `receiver.sun_avoidance_deg = 0`, meaning these constraints are not enforced
- `thermal.pv_view_factor = 1.0` (isolated PV plate; M4) and `collector.incidence_mode =
  sun_tracking` in the MVP config (normal incidence at every epoch)
- `transmitter.min_pointing_efficiency = None` (no pointing-availability threshold)
- `models.atmosphere = scenario` with `models.weather_scenario = clear` (the schema default):
  the weather is one deterministic state for the whole window, so the sky is clear at every
  epoch and no cloud outage ever occurs; availability is then a weather upper bound
  (PDF §10.2/§17 "ground optical availability may be overestimated")
- also labelled when configured (not defaults): any chain efficiency
  (`transmitter.electrical_to_beam_efficiency`, `receiver.conversion_efficiency`,
  `receiver.grid_efficiency`) equal to 1, a disabled `models.earth_occultation`,
  `models.sun_occultation` or `models.moon_occultation` (inactive constraint), and
  `models.atmosphere = none` (no attenuation, no weather outage)
- `collector.max_operating_temperature_K = 400 K` (M4) is an assumed scenario limit, neither
  optimistic nor sourced; replace it with the qualification limit of the chosen cell

`config.idealised_settings` evaluates exactly this list on a configuration (M6b). Not listed,
with reason: `models.light_time = false` changes the geometry with either sign (a named
approximation, `config.approximations_in_force`), a finite `receiver.max_input_power_W`
(whether it binds is a run outcome, `receiver_saturation_time_fraction` in `summary.json`)
and the ITU-R attenuation exceeded p % of the time (a statistical design value, not a bound).

Studies must override these defaults with sourced values or sweep them.

## Per-milestone sections

Milestones M2–M6 add the ephemeris, access, solar/thermal, beam and link-budget models here,
each with equations, validity ranges and references. M1–M6 follow below.

## M1: Constants, frames and the analytic two-body orbit (L0)

### Constants (`helioswarm.constants`)

`constants.py` is the only place a fixed physical constant is written as a number. A test
(`test_constants_are_not_hard_coded_elsewhere`) scans the numeric literals of every other source
module for the constants' digit strings. Configurable quantities (S_1AU) take their default from
here.

| Symbol | Value | Source |
|---|---|---|
| au | 149 597 870 700 m (exact) | IAU 2012 Resolution B2 |
| c | 299 792 458 m s⁻¹ (exact) | SI definition of the metre |
| GM☉ | 1.3271244004127942 × 10²⁰ m³ s⁻² (TDB-compatible) | DE440 via NAIF `gm_de440.tpc` (`BODY10_GM`, km³ s⁻² × 10⁹); Park et al. 2021, AJ 161, 105 |
| σ | 5.670374419 × 10⁻⁸ W m⁻² K⁻⁴ | CODATA 2018 (= 2π⁵k_B⁴/(15h³c²) with exact SI constants) |
| S☉ᴺ | 1361 W m⁻² (default of `constants.solar_constant_1au_W_m2`) | IAU 2015 Resolution B3; Kopp & Lean 2011 |
| R☉ᴺ | 6.957 × 10⁸ m | IAU 2015 Resolution B3 |
| a_WGS84, 1/f | 6 378 137 m, 298.257223563 | NGA.STND.0036 (WGS 84) |
| ERA rate | 1.00273781191135448 rev per UT1 day | IERS Conventions 2010 (TN 36), eq. 5.15 |
| Stellar ("sidereal") day | 86400 s / ERA rate = 86 164.0989 s | derived. This is the inertial rotation period, not the equinox-referenced mean sidereal day (86 164.0905 s) |
| ε₀ | 84 381.448″ | IAU 1976 (Lieske et al. 1977); the ECLIPJ2000 definition in SPICE |
| day | 86 400 s | definition |

**GM☉ and time scale.** The DE440 GM is TDB-compatible, and it is consistent with the TDB epochs
used everywhere (`epoch_tdb_s`). As an independent check it agrees with the IAU 1976 Gaussian
value k²·au³/d² to 5.0 × 10⁻¹². M2 must check that the value SPICE loads from `gm_de440.tpc`
equals this constant to float64 round-off.

### Frames (`helioswarm.frames`)

**State containers.**

- Every Cartesian state is a `CartesianState` and every time series a `CartesianTrajectory`.
  Both are frozen with read-only arrays. Each carries its `frame` (SPICE name), its `origin`
  (SPICE body name) and its `epoch_tdb_s`, which is TDB seconds past J2000 (SPICE ET), with
  `time_scale = "TDB"`.
- Supported frames: `J2000`, `ECLIPJ2000`. Supported origins: `SUN`,
  `SOLAR SYSTEM BARYCENTER`. Others are rejected.
- `require_same_frame_and_origin` must guard every subtraction of two states.

**ECLIPJ2000 → J2000.** This is a fixed passive rotation about the common x-axis (the J2000
equinox): `v_J2000 = R_x(−ε₀) v_ECL`, with

```
R_x(−ε₀) = [[1, 0, 0], [0, cos ε₀, −sin ε₀], [0, sin ε₀, cos ε₀]]
```

The ecliptic pole maps to (0, −sin ε₀, cos ε₀). The rotation is time-independent because both
frames are inertial, and it never changes the origin. Moving from the Sun to the SSB needs the
ephemeris (M2).

**Validity.** ECLIPJ2000 is the *mean* ecliptic of J2000 as SPICE defines it, not an
ecliptic of date. Using it as a fixed frame is exact by definition, so there is no
approximation. M2 cross-checks the matrix against SPICE `pxform`.

### Orbit model L0 (`helioswarm.orbit`)

The model is the two-body problem of a massless test particle about a point-mass Sun, with
μ = GM☉. Nothing else is included: no planets, no SRP and no relativistic terms. That is the
L0 level of PDF §6.2. It is valid as a baseline and for unit tests; its deviation from the real
trajectory ("orbit drift") is assessed against L1 in M2.

The orbital elements `(a, e, i, Ω, ω, ν)` are osculating at `epoch_tdb_s`. They are
Sun-centred and refer to the x–y plane of `frame`, which is `ECLIPJ2000` for configured orbits.
Only elliptic orbits are supported (0 ≤ e < 1, a > 0), and the schema enforces perihelion
> R☉ᴺ.

**Equations** (Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., ch. 2,
Alg. 9/10; Curtis, *Orbital Mechanics for Engineering Students*):

- Kepler III: `n = √(μ/a³)`, `T = 2π/n`. The period does not depend on e.
- Orbit equation `r = p/(1 + e cos ν)` with `p = a(1 − e²)`.
- Elements to state, with `P = (cos Ω, sin Ω, 0)`, `Q = (−sin Ω cos i, cos Ω cos i, sin i)`
  and `u = ω + ν`:
  - `r = r (cos u P + sin u Q)`
  - `v = √(μ/p) [−(sin u + e sin ω) P + (cos u + e cos ω) Q]`

  This is algebraically identical to the perifocal vectors rotated by R₃(−Ω)R₁(−i)R₃(−ω).
  The tests compare against that textbook form.
- State to elements:
  - `h = r × v`
  - `ε = v²/2 − μ/r` and `a = −μ/(2ε)`; ε ≥ 0 is rejected
  - `e = (v × h)/μ − r/|r|`
  - `i = atan2(√(h_x² + h_y²), h_z)`
  - `Ω = atan2(h_x, −h_y)`
  - `u = atan2(r·Q, r·P)` with `Q = ĥ × P`
  - `ω = atan2(e·Q, e·P)` and `ν = u − ω`
- Anomalies use Broucke & Cefola (1973) forms, which are continuous across revolutions:
  - `ν = E + 2 atan2(β sin E, 1 − β cos E)`
  - `E = ν − 2 atan2(β sin ν, 1 + β cos ν)`, with `β = e/(1 + √(1 − e²))`
  - `M = E − e sin E`

**Singular-case conventions.** Both directions use the same conventions, so state round trips
are exact to round-off.

| Case | Test | Convention | Effect on the state |
|---|---|---|---|
| Equatorial | sin i < 10⁻¹³ | i := 0 or π; Ω := 0, so the node line is +x. Retrograde angles are measured clockwise seen from +z. | ≤ r·sin i (< 0.02 m at 1 au) |
| Circular | e < 10⁻¹³ | e := 0; ω := 0; ν := argument of latitude u | ≤ ~2ea (< 0.03 m at 1 au) |
| Circular and equatorial | both | ν := true longitude | as above |

Both thresholds lie above the measured float64 noise. That noise is |e| ≤ 8.5 × 10⁻¹⁶ for an
exactly circular state and sin i ≤ 8.3 × 10⁻¹⁷ after an ECLIPJ2000↔J2000 round trip.

**Kepler's equation.**

1. M is reduced to x = |M mod [−π, π)| ∈ [0, π] (M already in [−π, π) is used unchanged, so
   small |M| carries no reduction round-off). There the root is bracketed by
   [x, min(x + e, π)].
2. Newton's method starts from Danby's guess x + 0.85e. Each iterate is clipped into the
   shrinking bracket.
3. On [0, π] the function f(E) = E − e sin E − x is increasing and convex, so in exact
   arithmetic the clipped Newton iteration converges monotonically for every M and every
   e < 1.
4. In float64 it stops when |ΔE| ≤ 10⁻¹² rad (after that step the error is
   O(e·ΔE²/(1 − e))) **or** when the residual is at round-off, |f(E)| ≤ 4 ε max(x, E). The
   second (backward-error) test is needed because f is evaluated with an absolute error of a
   few ε·E, so near e → 1 the root is only determined to ~ε·E/(1 − e cos E). For
   10⁻¹⁰ ≤ 1 − e ≤ 10⁻⁸ and small M that exceeds 10⁻¹² rad; with only the |ΔE| test Newton
   fell into a round-off 2-cycle and raised (M1 review). The forward error after a
   round-off stop is ≲ 2√2 ε/√(1 − e), below 10⁻¹² rad for every M when 1 − e > 4 × 10⁻⁷.
   The tolerance is absolute: for |E| ≪ 10⁻¹² rad the relative error of E can be large, but
   its absolute error stays ≤ 10⁻¹² rad.
5. The cap is 50 iterations. Exceeding it raises `KeplerConvergenceError` with the worst M and
   the last correction.

Measured over M ∈ [−20, 20]: 4–7 iterations for e ≤ 0.9, 11 for e = 0.999 and 20 for
e = 0.999999. The backward error is ≤ 8.9 × 10⁻¹⁶ rad on the reduced anomaly. For
1 − e ∈ {3·10⁻⁸, 10⁻⁸, 3·10⁻⁹, 10⁻⁹, 10⁻¹⁰, 10⁻¹²} and 200 000 log-spaced M ∈ [10⁻³²⁰, 3.1]
every element converges (25–37 iterations), and against a cancellation-free reference the
forward error is ≤ 0.11 of the round-off bound 8 ε E/(1 − e cos E).

**Propagation.** `propagate_kepler` computes M(t) = M₀ + n(t − t₀) and then E, ν and r, v with
fixed (a, e, i, Ω, ω). It works forward and backward in time, and it is vectorised over an
array of TDB epochs. A state input is first converted with `state_to_elements` and needs an
explicit μ. The output is in the frame of the elements, with origin `SUN`. There is no
truncation error. Round-off in M grows as ε_mach·|n Δt|, about 10⁻¹³ rad after 100 revolutions.

**Configuration hook.** `elements_from_config(orbit, *, epoch_tdb_s, mu_m3_s2=GM☉)`. The
conversions au → m (exact) and deg → rad happen in the config accessors, which is the IO
boundary. The epoch must be passed in TDB because UTC → TDB conversion needs the leap-second
kernel and is done only through SPICE `str2et` (M2). This module never converts time scales.

**Not modelled at L0.** Planetary perturbations, SRP and Poynting–Robertson drag, relativistic
corrections, and the Sun's motion about the SSB. The heliocentric L0 state must be moved to SSB
with the ephemeris before it is combined with SSB-origin states (M2).


## M2: Kernels, time scales, ephemerides, SRP and the L1 propagator

### Kernels (`helioswarm.kernels`)

The kernel set is `naif0012.tls`, `pck00011.tpc`, `gm_de440.tpc` and `de440s.bsp`. URLs,
sizes and checksums are in `data/kernels/kernels.json` and `docs/data_provenance.md`.

- **Loading.** Kernels are loaded only by `KernelSet.loaded()`. It verifies the files, requires
  an empty SPICE pool (so there is no nesting and no foreign kernel that could take priority),
  furnishes the kernels in manifest order, and unloads and `kclear`s them on exit.
- **Handles.** SPICE-backed functions take the resulting `LoadedKernels` handle as an explicit
  argument and refuse to run when it is inactive.
- **Global state.** The SPICE pool is process-global and not thread-safe, so only one scope is
  active per process.
- **Load order.** `gm_de440.tpc` is loaded after `pck00011.tpc`. The latter defines no GM
  variables (checked), so there is no override.

### Time scales (`helioswarm.timescale`)

- **Internal time.** Epochs are TDB seconds past J2000 (SPICE ET). Containers (`Epoch`,
  `TimeGrid`, `CartesianState`, `EphemerisState`) carry `scale/time_scale = "TDB"`.
- **UTC → TDB.** The conversion uses `str2et` on an explicitly labelled string
  (`"YYYY-MM-DD HH:MM:SS[.f] UTC"`). An unlabelled ISO string would be read in SPICE's
  `TIMDEF` default system; a test sets that default to TDB and shows the 69.18 s error the
  label prevents.
- **TDB → UTC.** The conversion uses `et2utc(…, "ISOC", decimals)`.
- **Equation (SPICE):** `TDB = UTC + ΔAT + 32.184 s + K sin(M + EB sin M)`, with
  K = 1.657 × 10⁻³ s. ΔAT comes from the LSK (37 s since 2017-01-01). Source: NAIF Time
  Required Reading.
- **Accuracy.**
  - Against the full Fairhead–Bretagnon series (astropy/ERFA `dtdb`) the difference is
    ≤ 2.7 × 10⁻⁵ s (measured, 2000–2025).
  - Against an independent reconstruction (IERS Bulletin C ΔAT + 32.184 s + the USNO
    Circular 179 two-term TDB − TT) the difference is ≤ 3.6 × 10⁻⁷ s.
  - For scale, 10⁻⁴ s of time error moves the Earth 3 m along its orbit.
- **Future leap seconds.** `naif0012.tls` lists no leap second after 2017. An epoch after a
  future leap second is off by 1 s per missing leap second until the LSK is updated, which
  changes its SHA-256 and therefore the manifest.
- **Output grids.** Grids are uniform in **TDB**: `t_k = TDB(start_utc) + k·output_step_s`.
  They are not uniform in UTC; the two differ by the ms-level periodic TDB − TT term and by
  any leap second.

### Ephemerides (`helioswarm.ephemeris`)

- **Contract.** `EphemerisProvider` is a protocol (`state`, `states`, `positions`,
  `gm_m3_s2`, `check_coverage`). Every state is in J2000 with origin SSB, in SI units, with a
  TDB epoch, and carries `source_kernel_ids`.
- **`SpiceEphemeris`.**
  - States come from `spkgeo(id, et, "J2000", 0)`, which gives geometric states (no
    light-time or aberration correction), in km converted to m.
  - Bodies and NAIF ids: SUN 10; MERCURY/VENUS/EARTH/MARS/JUPITER/SATURN/URANUS/NEPTUNE
    BARYCENTER 1–8; EARTH 399; MOON 301.
  - "J2000" in SPICE is the ICRF-aligned frame of the DE ephemerides.
- **GM values.**
  - They come from `bodvrd(body, "GM", 1)`, converted from km³ s⁻² to m³ s⁻² by the exact
    factor 10⁹. They are TDB-compatible.
  - The kernel GM☉ and `constants.GM_SUN_M3_S2` represent the same DE440 number. They agree to
    2 ulp, because SPICE's text-kernel parser is not correctly rounded (measured; enforced by a
    test).
  - **Authoritative value:** whenever SPICE is used, the kernel value (via the provider) is
    used. The constant serves kernel-free code (L0, analytic tests).
  - The effect of the 2-ulp difference on any trajectory is far below other errors (nm over a
    year).
- **Coverage.**
  - For each body, the coverage is the intersection of the `spkcov` windows of every segment in
    its chain to the SSB: a planet or moon (id ≥ 100) needs both its own segment and its system
    barycentre (id // 100).
  - A request outside that coverage raises `EphemerisCoverageError` before SPICE is queried.
  - de440s covers 1849-12-26 to 2150-01-22 TDB for every supported body.
- **`AnalyticEphemeris` (test double only).** The Sun moves uniformly; the Earth follows a
  circular Kepler orbit about it in the ecliptic. GM values come from an explicit table, and
  asking for a GM that is not in it raises. It is not a physical ephemeris.
- **Origin changes.** `shift_state_origin` and `shift_trajectory_origin` translate
  `r_SSB = r_SUN + r_☉/SSB(t)` and `v_SSB = v_SUN + v_☉/SSB(t)` (Galilean), using the Sun's
  state at each epoch. Both refuse inputs whose frame differs from the ephemeris frame.

### Satellite initial state

The steps follow PDF §6.1:

1. The configured elements are osculating, Sun-centred and ECLIPJ2000, with μ = GM☉ of the
   provider. They are converted with `elements_to_state` (M1).
2. The state is rotated ECLIPJ2000 → J2000 by `R_x(−ε₀)`; this agrees exactly with SPICE
   `pxform`.
3. The Sun's J2000/SSB state at the same TDB epoch is added.

The result is a `SatelliteState` in J2000/SSB. Its `force_model` lists the bodies and SRP
model. `mass_kg` is `None` because the config gives no mass and only A/m enters the dynamics.
`attitude` follows `collector.incidence_mode` and `collector_normal_unit` is the J2000 unit
normal of the collector front face at the initial epoch (PDF p.4 "attitude/normal vector";
added in M6b): `n = −r_helio/|r_helio|` (towards the Sun's centre) for `sun_tracking`, the
configured `fixed_normal_unit_j2000` for `fixed_normal`. The SRP force stays radial for both
attitudes (PDF §6.3 "simple radial SRP"); for a fixed-normal collector this is a named
approximation. Model the collector as a flat plate of area A that absorbs a fraction 1 − ε and
reflects a fraction ε specularly. Let ŝ be the unit vector from the Sun to the satellite, n̂
the plate's unit normal on the side away from the Sun, and θ the incidence angle
(cosθ = n̂·ŝ). The SRP force is `F = P A cosθ [(1 − ε) ŝ + 2ε cosθ n̂]` (Montenbruck & Gill
2000, §3.4). At normal incidence this is `P A (1 + ε) ŝ`, so the configured `C_r`
corresponds to `ε = C_r − 1` (0.3 for the MVP's C_r = 1.3; diffuse reflection is not
included in this split). The radial model applies
`P A C_r ŝ` at every θ. Relative to the plate's radial component it therefore overestimates by
`(1 + ε) / (cosθ [(1 − ε) + 2ε cos²θ])`. This ratio reduces to `1/cosθ` only for a perfect
absorber (ε = 0); with ε = 0.3 it is 3.06 at θ = 60°, not 2. The radial model also omits the
non-radial component of magnitude `2ε P A cos²θ sinθ` (the part of the reflected-light term
`2ε P A cos²θ n̂` perpendicular to ŝ). The
approximation is reported at run time: `config.approximations_in_force` names it in
`logs/run.log` (WARNING), `manifest.json` (`approximations`) and `report.md`, with its
comparison path. The available comparison is a rerun with
`orbit.include_solar_radiation_pressure: false`, which bounds the whole SRP contribution.
Attitude-dependent SRP (L3) is deferred to the M8 sensitivity studies. `models.light_time:
false` (instantaneous geometry) is reported the same way.

With SRP enabled the elements remain *gravitational* osculating elements (μ = GM☉, not
GM☉ − k). A "circular" configured orbit is therefore not exactly circular under SRP.

### SRP (`helioswarm.srp`)

- **Equation** (PDF §6.2):
  - `P_srp(r) = (S_1AU/c)·(AU/r_☉)²`
  - `a_srp = P_srp·C_r·(A/m)·ŝ`

  Here `ŝ` is the unit vector from the Sun's centre to the satellite, and `r_☉` is the distance
  from the Sun's centre, not from the SSB. `S_1AU` is `constants.solar_constant_1au_W_m2`.
- **Equivalent form:** `a = k (r − r_☉)/|r − r_☉|³` with `k = (S_1AU/c)·AU²·C_r·A/m`.
  - For the MVP (C_r 1.3, A/m 0.02 m² kg⁻¹, S 1361 W m⁻²): k ≈ 2.64 × 10¹⁵ m³ s⁻², and the
    lightness number is β = k/GM☉ = 1.99 × 10⁻⁵.
- **Named approximation.** This is the cannonball model, equivalently a flat plate kept normal
  to the Sun line. C_r ∈ [1, 2]: 1 is a perfect absorber, 2 a perfect specular reflector at
  normal incidence (Montenbruck & Gill 2000, §3.4).
- **Not modelled:**
  - attitude-dependent area and off-normal reflection (the L3 item);
  - planetary shadowing;
  - finite solar disk, limb darkening and solar-cycle variation of S;
  - Poynting–Robertson drag and thermal re-emission (relative size ~v/c ≈ 10⁻⁴ of the radial
    force).
- **Exact equivalence.** Because the force is radial and ∝ 1/r², it is conservative and equals
  a reduced solar GM, `μ_eff = GM☉ − k`. The integrator uses that folded form, which is
  algebraically identical (tested). The closed form L0(μ_eff) is used as a validation
  reference.

**SRP materiality (PDF §6.3: "report the A/m range for which SRP materially changes the
orbit").** For a near-circular orbit started with the gravitational circular speed, SRP turns
μ into μ(1 − β) and the along-track separation from the SRP-free orbit grows as
`Δs ≈ 2β·n·t·a` with `β = (S_1AU/c)·AU²·C_r·(A/m)/GM☉ = 7.66 × 10⁻⁴·C_r·(A/m)` [A/m in
m² kg⁻¹] (`srp.srp_along_track_drift_estimate_m`; derivation in its docstring; matches the L1
integration of the MVP orbit to 0.07 %). Valid for β ≪ 1, e ≪ 1 and 2βnt ≪ 1. One Julian year
at 1 au, S = 1361 W m⁻²:

| A/m [m² kg⁻¹] | β (C_r = 1.3) | Δs after 1 yr, C_r = 1 / 1.3 / 2 [m] |
|---|---|---|
| 10⁻⁴ | 9.95 × 10⁻⁸ | 1.44 × 10⁵ / 1.87 × 10⁵ / 2.88 × 10⁵ |
| 10⁻³ | 9.95 × 10⁻⁷ | 1.44 × 10⁶ / 1.87 × 10⁶ / 2.88 × 10⁶ |
| 10⁻² | 9.95 × 10⁻⁶ | 1.44 × 10⁷ / 1.87 × 10⁷ / 2.88 × 10⁷ |
| 0.02 (MVP) | 1.99 × 10⁻⁵ | 2.88 × 10⁷ / 3.74 × 10⁷ / 5.76 × 10⁷ |
| 0.1 | 9.95 × 10⁻⁵ | 1.44 × 10⁸ / 1.87 × 10⁸ / 2.88 × 10⁸ |
| 1 | 9.95 × 10⁻⁴ | 1.44 × 10⁹ / 1.87 × 10⁹ / 2.88 × 10⁹ |
| 10 | 9.95 × 10⁻³ | 1.4 × 10¹⁰ / 1.9 × 10¹⁰ / 2.9 × 10¹⁰ (2βnt ≈ 0.1–0.2: linear estimate only indicative) |

Criterion: SRP is *material* when Δs exceeds the largest unmodelled force-model term, the 1PN
drift of 5.6 × 10⁴ m/yr (L1 section below). That holds for A/m ≳ 3.9 × 10⁻⁵ / 3.0 × 10⁻⁵ /
1.9 × 10⁻⁵ m² kg⁻¹ (C_r = 1 / 1.3 / 2; `srp.srp_material_area_to_mass_m2_kg`); with a 1 km/yr
criterion the thresholds are 7 × 10⁻⁷ / 5 × 10⁻⁷ / 3.5 × 10⁻⁷ m² kg⁻¹. The MVP value
(0.02 m² kg⁻¹) is ~700× above the 1PN-level threshold, so SRP is material for it and for any
lighter collector; `assess_l1_run` reports the SRP on/off difference for any configured run.
The schema default `include_solar_radiation_pressure = false` exists for force-model
comparisons; a collector study should enable SRP with its own C_r and A/m.

### L1 propagator (`helioswarm.propagate`)

**Model.** L1 of PDF §6.2 is a massless test particle. The Sun and the selected bodies follow
their prescribed DE440 positions; their μ_j come from `gm_de440.tpc`. Inputs and outputs are
J2000 states with origin SSB.

- The bodies are SUN plus `orbit.perturbing_bodies` when `include_planetary_perturbations` is
  true. The default perturbers are all eight planetary-system barycentres, with the
  Earth–Moon system split into EARTH and MOON. When perturbations are off, the Sun is the only
  body (an explicitly set `perturbing_bodies` is then rejected, like `cr` without SRP).

**Integrated equations: heliocentric formulation with consistent indirect term (M2 review).**
PDF §6.2 allows a heliocentric non-inertial frame if the correct indirect acceleration is
included. With `r = r_sat − r_☉` and `R_j = r_j − r_☉` (J2000 axes):

```
d²r/dt² = −(GM☉ − k) r/|r|³ + Σ_{j≠☉} μ_j [ (R_j − r)/|R_j − r|³ − R_j/|R_j|³ ]
```

- The bracket's second term is the indirect term: the Sun's Newtonian acceleration due to the
  *same* bodies that act on the satellite. It is the barycentric PDF equation minus that
  acceleration (identity tested). Direct and indirect terms are therefore consistent for any
  body subset (the direct and indirect parts of the disturbing function; Murray & Dermott
  1999, ch. 6).
- SRP is folded in as `GM☉ → GM☉ − k` (exact for the radial model, see SRP above).
- Output: `r_SSB(t) = r(t) + r_☉(t)` with the Sun's state taken directly from the provider
  (not from the cache).
- Consequences (tested with the real de440s Sun): Sun-only L1 equals L0 Kepler (7.7 × 10⁻⁴ m
  over a year; with SRP, L0 with GM☉ − k: 1.5 × 10⁻³ m). Dropping a body removes only its
  physical tidal effect (Neptune 2.9 × 10³ m, Jupiter 7.4 × 10⁶ m over the MVP year).

**Why not the barycentric form.** Integrating the PDF equation literally in J2000/SSB with
the Sun on its DE440 trajectory (the original M2 implementation) applies the Sun's *full*
DE440 acceleration as an implicit indirect term. That acceleration comes from every body in
DE440, so any body missing from the model contributes its Sun-reflex term without its direct
term — the mixing PDF §6.2 forbids. Measured over the MVP year: Sun-only barycentric L1 was
2.2 × 10⁸ m from L0 (6× further from the full model than L0 itself); dropping Neptune moved
the orbit by 3.2 × 10⁵ m instead of its physical 2.9 × 10³ m. Even with all nine bodies the
barycentric form carries the Sun's unmodelled acceleration (asteroids, Pluto, relativity in
DE440): it differs from the heliocentric form by 566 m over the year, and REBOUND agrees with
the heliocentric form to 1.0 m but with the barycentric form only to 565 m. The barycentric
form is kept as an explicit comparison path (`propagate_l1(..., formulation="barycentric")`,
not reachable from the configuration) and refuses any body set without all eight planetary
systems.

**Integrator.**

- `scipy.integrate.solve_ivp`, method `DOP853` (Dormand–Prince 8(5,3); Hairer, Nørsett &
  Wanner, *Solving ODEs I*, §II.10).
- `rtol = time.integrator_rtol` (default 10⁻¹²; bounds [10⁻¹³, 10⁻⁶] shared by the schema
  and `IntegratorSettings`).
- Per-component `atol`: `integrator_atol_position_m` for positions (default 10⁻³ m) and
  `integrator_atol_velocity_m_s` for velocities (default 10⁻⁹ m/s). Both are below
  rtol·|state| at 1 au, so rtol governs.
- `max_step = time.integrator_step_s`.
- Outputs are evaluated with the integrator's 7th-order dense output (`t_eval`).
- Epochs before the initial epoch (`orbit.initial_epoch_utc`) are integrated backwards,
  separately, from the initial state; the output grid starts at `time.start_utc` (the two
  epochs are independent inputs; tested with a ±10-day offset).
- Diagnostics (`L1Diagnostics`): formulation, per-direction `nfev`, success, message and
  number of outputs, GM values used, the cache error (below), and
  `sun_two_body_energy_rel_drift = max|E(t) − E₀|/|E₀|` with
  `E = |v − v_☉|²/2 − (GM☉ − k)/|r − r_☉|`. In the heliocentric formulation E is an integral
  of motion of the Sun-only model whatever the Sun's SSB motion (measured 1.3 × 10⁻¹⁵ over a
  year with the de440s Sun), so it then measures the integration error.
- Run-level report (`assess_l1_run`, an explicit call because it re-integrates up to three
  times): max position change when `max_step` is halved, when rtol is tightened tenfold, and
  the SRP on/off difference (max and final) with β. For the MVP year at a 1-day cap:
  8.0 × 10⁻⁴ m, 2.4 × 10⁻³ m and 3.742 × 10⁷ m (β = 1.99 × 10⁻⁵).

**Ephemeris cache.**

- Body positions inside the right-hand side come from a piecewise cubic Hermite interpolant of
  SPICE positions and velocities on a uniform TDB grid (`time.ephemeris_cache_step_s`, default
  3600 s). The error bound is `h⁴/384·max|x⁗|`. Heliocentric body positions are differences
  of cached SSB positions; the interpolant is linear in its node data, so this equals
  interpolating the differences.
- Every run measures the cache error at every interval midpoint (where the error kernel
  s²(1−s)² peaks) against the provider and reports it per body
  (`L1Diagnostics.cache_midpoint_error_m`). The midpoint maximum bounds direct-SPICE errors at
  400 random epochs (tested).
- Measured over the 2030 year (de440s), Mercury barycentre / Moon: 0.060 / 0.015 m at 3600 s,
  0.97 / 0.24 m at 7200 s, 78 / 20 m at 21 600 s; every other body ≤ 0.9 m at 21 600 s. The
  outer planets sit at the float64 floor of their 4.5 × 10¹² m positions.
- Validated range: the schema caps the step at 7200 s, where every body stays below 1 m (M2
  review; it previously allowed 1 day, where the Mercury barycentre error reaches ~2 × 10⁴ m
  and the Moon's ~5 × 10³ m).
- At 1 au the satellite's trajectory does not change measurably (< 10⁻² m over a year) when the
  cache step is halved; near Mercury or the Moon this would no longer hold, which is why the
  error is reported for every run.
- Why a cache: direct SPICE costs 5.4 µs per body per call (measured), which would add about
  70 s per simulated year at max_step = 300 s.

**Measured accuracy** (holy7c18406, one simulated year, heliocentric formulation):

| Case | Result |
|---|---|
| Sun-only L1 vs L0 Kepler, uniformly moving analytic Sun, e = 0.3, rtol 10⁻¹², 1-day step cap | 6.7 × 10⁻⁴ m; energy drift 3.8 × 10⁻¹⁵ |
| Sun-only L1 vs L0 Kepler, de440s Sun, MVP orbit, without / with SRP (L0 with GM☉ − k) | 7.7 × 10⁻⁴ m / 1.5 × 10⁻³ m |
| Same as row 1, no step cap | 130 m at rtol 10⁻¹⁰, 13 m at 10⁻¹¹, 1.5 m at 10⁻¹², 0.19 m at 10⁻¹³ (error ∝ rtol) |
| Full MVP model, 1-day cap vs 12-h cap / rtol 10⁻¹³ / cache 1800 s | each < 10⁻² m (test tolerance); `assess_l1_run`: 8.0 × 10⁻⁴ m (halved cap), 2.4 × 10⁻³ m (rtol/10) |
| Full MVP model, configured cap 300 s vs 1-day cap | 4.6 × 10⁻² m (test tolerance 0.1 m). Round-off accumulated over ~10⁵ steps (~ε·\|r\|·√N), not truncation. |
| Runtime, full MVP year | ~1.4 s at a 1-day cap (5.5 × 10³ RHS evaluations); ~50 s at the configured 300 s cap (1.29 × 10⁶ evaluations; the heliocentric right-hand side costs ~25 % more than the original barycentric one) |

The MVP's 300 s cap therefore buys no accuracy at 1 au. It is kept because it is the PDF's
example value, and the cost is reported rather than tuned away.

**Not modelled (force-model error budget).**

- Relativistic (1PN) acceleration, the largest omitted term at 1 au. For a test particle the
  Schwarzschild/PPN term is `a = GM/(c²r³)[(4GM/r − v²) r + 4(r·v) v]`; on a circular orbit
  (r·v = 0, v² = GM/r) it is radial and outward with relative size β₁PN = 3GM☉/(rc²) =
  2.96 × 10⁻⁸ at 1 au, i.e. it acts like `GM☉ → GM☉(1 − β₁PN)`. Since the configured elements
  are Newtonian osculating elements, the same argument as for SRP gives an along-track drift
  `Δs ≈ 2β₁PN·n·t·a` = **5.6 × 10⁴ m per year** at 1 au. Measured (M2 review): a 1 au circular
  orbit integrated with and without the term (DOP853, rtol 10⁻¹³) differs by 55 667 m after a
  Julian year, matching the closed form to 10⁻⁶. (The M2 implementation notes first gave an
  unmeasured "~10 km/yr"; that was wrong by 5.6×.) In general `2β₁PN·n·t·a = 6GM☉·n·t/c²`,
  so for a fixed elapsed time the drift scales as a^(−3/2). The term is not yet an optional
  named force; adding it is a later-fidelity item (deferred, M2 review).
- Pluto, asteroids, planetary oblateness, solar oblateness.
- Any back-reaction of the satellite.

**Independent comparison (L2, REBOUND 5.2.2 IAS15).**

- Two-body: REBOUND agrees with L1 to 7.7 × 10⁻⁴ m over a year.
- Full model:
  - REBOUND evolves the Sun and the nine bodies self-consistently and Newtonian from the same
    SPICE initial states and GM values.
  - The satellite's heliocentric difference from L1 is 1.03 m after a year (test tolerance
    10 m). L1's planetary force model moves the satellite 3.25 × 10⁷ m from L0 over that
    year, so the cross-check resolves it to ~3 × 10⁻⁸.
  - REBOUND's own bodies drift from DE440 by Mercury 5.8 × 10⁵ m, Earth 6.1 × 10⁴ m,
    Jupiter 491 m and Sun 261 m, because DE440 also contains general relativity, asteroids
    and Pluto. The satellite at 1 au is only tidally sensitive to those errors.
  - Discrimination: without Neptune in both codes they agree to 1.4 m, while dropping Neptune
    changes L1 by 2.9 × 10³ m.
  - REBOUND vs the barycentric comparison path: 565 m (the Sun's unmodelled DE440
    acceleration; see above).

### Configuration fields added in M2

All are `assumed` numerical or model-choice fields with defaults. They change the config hash
but not `schema_version`, because every previously valid file still parses with the same
meaning.

- `time.integrator_rtol`: 10⁻¹², allowed range [10⁻¹³, 10⁻⁶]. The floor stays above scipy's
  silent clamp at 100ε.
- `time.integrator_atol_position_m`: 10⁻³ m.
- `time.integrator_atol_velocity_m_s`: 10⁻⁹ m s⁻¹.
- `time.ephemeris_cache_step_s`: 3600 s, allowed up to 7200 s (validated range, M2 review;
  previously up to 1 day).
- `orbit.perturbing_bodies`: the nine bodies above. Duplicates are rejected; setting it
  explicitly with `include_planetary_perturbations = false` is rejected (M2 review).
- The L1-only `time` fields (`integrator_step_s`, `integrator_rtol`, the two atols,
  `ephemeris_cache_step_s`) are rejected when set with `orbit.model = two_body_kepler`.
- `frames.ephemeris_kernel_set` (M2 review, `sourced`): `helioswarm-de440s-v1`, the
  `kernel_set_id` of `data/kernels/kernels.json`. SPICE-backed run entry points
  (`initial_state_from_config`, `propagate_l1_from_config`) refuse a loaded kernel set with a
  different id, so the configuration pins the ephemeris and GM data (the manifest's SHA-256
  values pin the files).

## M3: Ground sites, Earth rotation, light-time targeting, occultation and access

Modules: `helioswarm.ground` (sites, Earth orientation, inertial site states) and
`helioswarm.visibility` (light time, pointing, receiver geometry, occultation, illumination,
PDF §7.3 diagnostics). Every state is J2000 / SSB at TDB epochs. The Earth-fixed site is kept
apart from its inertial state (`GroundSite` vs `SiteTrajectory`, PDF §3).

### Ground sites (WGS 84)

- **Geodetic → Earth-fixed** (NGA.STND.0036; Hofmann-Wellenhof et al., *GNSS*, 2008,
  eq. 2.31): `N = a/√(1 − e² sin²φ)`, `x = (N + h) cos φ cos λ`, `y = (N + h) cos φ sin λ`,
  `z = (N(1 − e²) + h) sin φ`, with `a = 6 378 137 m`, `e² = f(2 − f)` (derived constants
  `WGS84_SEMIMINOR_AXIS_M`, `WGS84_FIRST_ECCENTRICITY_SQUARED`, checked against NGA table 3.1).
  Agrees with SPICE `georec` and astropy/ERFA `gd2gc` to < 1 mm (measured ~1e-9 m).
- **Local frame**: up = ellipsoid normal (geodetic vertical)
  `(cos φ cos λ, cos φ sin λ, sin φ)`; east `(−sin λ, cos λ, 0)`; north
  `(−sin φ cos λ, −sin φ sin λ, cos φ)`; east × north = up (checked against the gradient of the
  ellipsoid equation and SPICE `surfnm`).
- Sites are fixed to the Earth-fixed frame `IAU_EARTH`. Terrain, plate motion, tides and the
  difference between WGS 84 and ITRF coordinates (centimetres) are not modelled.

### Earth orientation (`IAU_EARTH`, a documented approximation)

- `v_J2000 = R(t) v_IAU_EARTH` with `R = [W]₃[π/2 − δ₀]₁[π/2 + α₀]₃)ᵀ` (NAIF PCK Required
  Reading) and the `pck00011.tpc` constants (IAU WGCCRE 2009; the 2015 report gives no new Earth
  values): `α₀ = 0° − 0.641° T`, `δ₀ = 90° − 0.557° T`, `W = 190.147° + 360.9856235° d` (T
  centuries, d days of TDB past J2000). SPICE `sxform` returns `R` and the exact `dR/dt`; a test
  re-evaluates the closed form from the kernel-pool constants (agreement 3e-11, the float64
  spacing of W ~ 7e4 rad). The rotation period of the model is 86 400 × 360/360.9856235 s
  = 86 164.0989 s; it differs from the IERS 2010 stellar day (`constants.STELLAR_DAY_S`) by
  3.1e-8 relative. A site returns to its J2000 direction after one period to < 1e-6 rad (pole
  precession 2.7e-7 rad/day) and turns 0.9856235° beyond a full turn per 86 400 s.
- **Accuracy: rotation angle (the dominant term).** The IAU angle of the prime meridian
  measured from the J2000 x-axis, `90° + α₀ + W = 280.147° − 0.641° T + 360.9856235° d`, is a
  linear function of TDB. The Earth actually turns by the IERS Earth Rotation Angle of UT1,
  `ERA = 2π(0.7790572732640 + 1.00273781191135448 D_UT1)` (IERS Conventions 2010, eq. 5.15).
  The offset about the pole is therefore `ω[(TT − UT1) − 75.06 s − 0.554 s/yr · (t − J2000)]`:
  - **Constant part.** The IAU W₀ = 190.147° matches ERA only for TT − UT1 = 75.06 s at J2000.
    The actual value was 63.8 s, so the IAU frame is already −0.047° = −8.2e-4 rad (−11 s of
    rotation, 5.2 km at the equator) off at J2000.
  - **Secular part.** The IAU rate (W rate plus the −0.641°/century node term) is 6.3e-6
    deg/day slower than the ERA rate: −4.04e-5 rad/yr, i.e. −0.55 s of rotation per year
    while TT − UT1 is constant. The growth of TT − UT1 (+5.5 s from 2000 to 2020) partly
    offset this drift in the past.
  - **Nutation does not enter this angle.** The closed form above matches ERFA `c2t06a`
    (IAU 2006/2000A, UT1 = UTC, no polar motion) to ≤ 2.4e-8 rad at 2000, 2020, 2030 and
    2040; the equation of the equinoxes (≤ 1.2 s) and ΔT drift alone, which earlier versions
    of this section blamed, cannot produce the 11 s offset at J2000.
  - **Simulated epochs.** With TT − UT1 held at its 2017–2026 value (~69.2 s), the offset is
    −1.64e-3 rad (−22.5 s of rotation, 10.5 km at the equator) in 2030 and −2.04e-3 rad (−28 s,
    13.0 km) in 2040, growing ~0.26 km/yr at the equator; the real value depends on the
    unpredictable future TT − UT1 (a few seconds).
- **Accuracy: pole (secondary).** IAU_EARTH omits nutation (pole offset up to
  √((17.2″ sin 23.4°)² + 9.2″²) = 11.5″ = 5.6e-5 rad ≈ 0.36 km at the surface; measured
  3.3–4.7e-5 rad) and polar motion (~0.3″, ~10 m).
- **Measured against IERS (past epochs).** astropy's IERS-based ITRS → GCRS (IAU 2006/2000A,
  bundled IERS-B, no network; `tests/unit/test_ground.py`), equatorial site:

  | Epoch (UTC) | Site position | Pole direction | Rotation angle about the pole |
  |---|---|---|---|
  | 2005-03-01 | 6.14 km | 4.4e-5 rad | −9.6e-4 rad |
  | 2020-06-15 | 7.89 km | 3.7e-5 rad | −1.24e-3 rad |
  | 2023-01-01 | 8.65 km | 3.9e-5 rad | −1.36e-3 rad |

  The site velocity differs by ω·|Δr| (0.45–0.63 m/s). The ERFA test
  (`test_iau_earth_rotation_angle_offset_is_the_w_definition_against_era`) pins the
  decomposition and the 2030/2040 values above.
- **Consequence.** For the simulated epochs (2030+) UT1 cannot be predicted to better than a
  few seconds, so no model can place the site in inertial space to better than ~1 km years
  ahead. The IAU offset (~23 s in 2030) shifts the simulated site along its parallel, i.e. it
  shifts the diurnal access pattern by ~23 s (growing ~0.55 s/yr); access fractions and
  elevation statistics are unaffected. It is *not* a pointing error in the simulation, because
  the simulated transmitter aims at the simulated site. Comparisons with real passes would need
  an ITRS (IERS) Earth orientation; the astropy and ERFA comparison tests are the path for that.

### Inertial site state

`r_site(t) = r_⊕(t) + R(t) r_ef`, `v_site(t) = v_⊕(t) + Ṙ(t) r_ef` (= ω × r), with the Earth
(NAIF 399) geometric J2000/SSB state from the ephemeris provider. Checked: velocity vs central
differences (< 1e-4 m/s), |ω × r| = ω ρ (rel 1e-7; the pole-precession part of ω is 4.8e-8).

### Retarded-time (light-time) targeting (PDF §7.1)

- Grid epochs are **emission** epochs `t_e`. The light time τ solves
  `c τ = |r_site(t_e + τ) − r_sat(t_e)|` by the PDF's fixed-point iteration
  (`τ₀ = |r_site(t_e) − r_sat(t_e)|/c`; contraction factor ≤ |v_site|/c ≈ 1e-4), vectorised over
  the grid, with `models.light_time_tolerance_s` (1e-9 s = 0.3 m of path) and
  `models.light_time_max_iterations` (50). Unconverged epochs fall back to `scipy.optimize.brentq`
  on `f(τ) = |r_site(t_e + τ) − r_sat| − cτ` (bracket [0, 2ᵏ·2τ₀]) and are flagged
  (`LightTimeSolution.method`). Stored: `t_emit_tdb_s`, `t_arrival_tdb_s`, iterations, method,
  residual `|cτ − |ρ||`. MVP year: 3 iterations at every epoch, residual ≤ 3 mm, no fallback.
- Float64 epochs ~1e9 s are spaced 1.2e-7 s, so `t_arrival` is represented to ~6e-8 s
  (≤ 2 mm of site motion).
- `models.light_time = false` selects the *instantaneous geometry approximation*: the site at
  `t_e`, `t_arrival = t_e`, and `light_time_s = range/c` reported but not applied. The aim
  then differs from the retarded aim by ~|v_⊥|/c (asin(v/c) exactly for a site moving
  perpendicular to the line of sight; tested), i.e. ~1e-4 rad.
- **Neglected (stated, PDF §7.1):** relativistic light propagation. For the MVP geometry
  (two 1 au orbits 70° apart, path 1.15 au, impact parameter 0.82 au) the Shapiro delay
  `(2GM☉/c³) ln((r₁ + r₂ + R)/(r₁ + r₂ − R))` is 1.3e-5 s (3.9 km of path; it moves the
  arrival epoch, not the aim: 1.3e-5 s × 30 km/s = 0.4 m of site motion). The gravitational
  bending of the path displaces the beam transversely at the receiver by ~2.4 km (1.4e-8 rad
  as seen from the transmitter; numerical integration of the weak-field transverse acceleration
  `2GM b/(c² r³)`, coordinate-dependent at this level), about 0.14 λ/D for a 1 µm, 10 m
  aperture. Near superior conjunction both grow (1.1e-4 s and ~2e-6 rad for a path passing
  ~2 solar radii from the Sun, heliocentric separation 179°), where the link is blocked anyway. A real link would point closed-loop on a receiver beacon
  (reciprocal path), which removes the bending to first order; the simulator's flat-space
  aim is self-consistent.

### Pointing (beam direction) and aberration

- `u = (r_site(t_a) − r_sat(t_e))/|…|` (J2000): the geometric photon direction in the
  barycentric frame. It contains the point-ahead for the **target's** motion
  (`target_motion_point_ahead_rad`, the angle to the site's emission-epoch position:
  5.9e-5 rad max for the MVP year).
- **Transmitter aberration (implemented, not neglected).** The transmitter, moving with
  `v_sat` relative to the barycentric frame, must point along the Lorentz-aberrated direction
  `u′ = [u + (γ − 1)(u·b̂)b̂ − γβ]/[γ(1 − β·u)]`, `β = v_sat/c` (Rindler, *Relativity*, 2006,
  §4.4); to first order `u′ = u − β + (u·β)u`, magnitude `|β_⊥| ≤ |v|/c ≈ 1e-4 rad`.
  Stored as `transmit_direction_tx_frame_unit` and `transmitter_aberration_rad` (5.9e-5 rad
  max for the MVP year, ~600 optical beam widths λ/D = 1e-7 rad). Net angle between `u′` and
  the emission-epoch line of sight: 1.11–1.18e-4 rad (relative speed 34 km/s). Checks: inverse
  transformation round trip, exact case `u = ŷ, v = βc x̂ → (−β, 1/γ, 0)`, and a comoving
  transmitter/receiver pair whose two corrections cancel to < 3β² (a sign error leaves 2e-4).
- **Pointing rate** (inertial, analytic): `τ′ = u·(v_site − v_sat)/(c − u·v_site)`,
  `ρ′ = v_site(1 + τ′) − v_sat`, `u̇ = (I − uuᵀ)ρ′/|ρ|`, `pointing_rate_rad_s = |u̇|`;
  matches central differences to 1e-6 relative. MVP year: mean 1.99e-7 rad/s (the line
  between two near-co-orbiting bodies turns at the mean motion 2π/yr).
- **Range rate** `range_rate_m_s = u·(v_site(t_a) − v_sat(t_e))` (relative radial velocity,
  positive when separating); `light_time_rate = dτ/dt_e`, with `c·τ′ = range_rate/(1 − u·v_site/c)`
  (exact identity, tested).

### Receiver geometry

- Apparent direction to the satellite in the receiver's rest frame:
  `s = −u′(u, v_site(t_a))` (same aberration formula with the site's orbital + rotational
  velocity). This keeps elevation consistent with the Earth-blockage test (photon path in the
  Earth's translating frame); the two differ only by the rotation speed, 465 m/s / c = 1.6e-6
  rad. Without it, the barycentric direction would misplace the horizon by up to 1e-4 rad.
- Elevation `atan2(s·up, |s − (s·up)up|)` (geodetic up at `t_a`); azimuth
  `atan2(s·east, s·north)` from north through east. **No atmospheric refraction** (it raises
  the apparent elevation by ~0.1° at 10°; the min-elevation constraint applies to the
  unrefracted elevation).
- Sun separation: angle between `s` and the apparent direction of the Sun's centre (light
  leaving the Sun at `t_a − τ_☉`, two fixed-point steps, aberrated the same way). The Sun's
  SSB speed (~13 m/s) makes the remaining epoch error negligible.
- Flags: `below_min_elevation = elevation < receiver.min_elevation_deg`;
  `sun_avoidance_violation = sun_separation < receiver.sun_avoidance_deg` (0 → never).

### Occultation of the beam segment (PDF §6.4)

- The beam is the straight segment `r_sat(t_e) → r_site(t_a)`. Each occulter B is tested in
  its own (Galilean) comoving frame: the segment `p₀ = r_sat(t_e) − r_B(t_e)` →
  `p₁ = r_site(t_a) − r_B(t_a)` is the exact photon path relative to a uniformly moving body;
  B's acceleration bends it by ≤ |a_B|τ²/8 (Sun ~0.03 m; Moon ≤ 9e-3 m s⁻² × (600 s)²/8
  ≈ 0.4 km ≪ R_moon). A test compares with a brute-force photon/Moon sampling for a Moon moving
  at 100 km/s (static-Moon treatment off by 4e7 m).
- **Earth**: segment–ellipsoid intersection in `IAU_EARTH` at `t_a`
  (`p₀ = R(t_a)ᵀ(r_sat(t_e) − r_⊕(t_e))`, `p₁ = r_ef`). Scaling `(x/a, y/a, z/b)` maps the
  WGS 84 ellipsoid to the unit sphere, so the test is `min_{s∈[0, s_max]}|p₀′ + s(p₁′ − p₀′)| < 1`.
  The last `models.earth_occultation_standoff_m` (1 m) next to the receiver are excluded
  (`s_max = 1 − standoff/L`) because the receiver lies on the surface; this is blind only for
  |elevation| < standoff/(2a) ≈ 8e-8 rad. Sites with h < 0 are tested against the WGS 84
  ellipsoid scaled by `k = √((x² + y²)/a² + z²/b²) < 1`, the similar ellipsoid that passes
  exactly through the site, so the site lies on the tested surface as it does for h = 0. Its
  normal at the site differs from the geodetic up by ≤ ~e²|h|/a (5.3e-7 rad at h = −1000 m), so
  blockage and apparent elevation < 0 agree to that level. (The M3 version lowered both axes
  by |h|; that is not a parallel surface and left the site ~1.4 mm inside it at h = −1000 m,
  which flagged every ray below ~1e-3 rad of elevation as Earth-blocked. Fixed in the M3
  review.) The Earth's rotation during the last 2a/c ≈ 43 ms of the path
  (≤ 20 m at the surface) is neglected. Cross-checked against SPICE `surfpt` on ~2900 random
  segments (identical outcomes). No atmosphere/terrain margin: the min-elevation constraint
  plays that role.
- **Sun**: blocked if the segment comes closer to the Sun's centre than
  `R_block = d sin(min(asin(R☉/d) + m, π/2))`, `d` = receiver–Sun distance,
  `m = models.sun_occultation_limb_margin_deg`. For a satellite beyond the Sun this is exactly
  "within (apparent solar radius + margin) of the Sun's centre as seen from the receiver", the
  configured meaning of the margin (the M3 brief's `R☉(1 + margin)` is a fractional variant of
  the same idea; the existing angular field was kept). With m = 0 it is the IAU 2015 nominal
  sphere (695 700 km = `pck00011` `BODY10_RADII`). A satellite *in front of* the Sun near its
  direction is not occulted but is caught by the Sun-avoidance angle.
- **Moon**: blocked if the segment comes closer to the Moon's centre than R_moon = 1737.4 km
  (IAU WGCCRE 2015 = `pck00011` `BODY301_RADII`; `constants.MOON_RADIUS_M`).
- `visible = not (earth_blocked or sun_blocked or moon_blocked or below_min_elevation or
  sun_avoidance_violation)`; `occulted_by` names the first blocking body (EARTH, SUN, MOON).
  Checks disabled in `models.*_occultation` are not evaluated (flag false, distance NaN).

### Satellite illumination (conical shadow with penumbra)

- From the satellite at `t_e`, the Sun and each occulter (Earth, Moon) are uniform disks of
  apparent radius `asin(R/d)` at their light-time-retarded positions (two fixed-point steps;
  the occulter's ~30 km/s motion over a few hundred seconds is comparable to the Earth's
  radius, so the correction matters; tested). Hidden fraction = circle–circle overlap area of
  the two disks (planar formula on the sky, evaluated with `atan2` for conditioning near
  tangency; Montenbruck & Gill 2000, §3.4.2) / (π r☉²): 0 apart, 1 in the umbra,
  (r_B/r☉)² annular. Only occulters nearer than the Sun count. Earth: sphere of the equatorial
  radius (conservative; oblateness moves the shadow edge by ≤ 21 km). Moon: mean radius.
- Neglected: limb darkening (during partial phases the hidden *power* fraction differs from
  the hidden *area* fraction computed here; not quantified in M3), the
  small-angle error O(r²) ≈ 2e-5 relative at 1 au, atmospheric refraction/extinction of the
  Earth's limb, and planets as occulters. If two occulters overlap the solar disk at once their
  fractions are summed (an upper bound on the loss if they also overlap each other) and the
  epoch is flagged `multiple_occulters`.
- `IlluminationResult` records the ephemeris kernel ids it was computed from (`source_ids`)
  and its time scale (`TDB`), like `AccessResult`.
- MVP year: illumination 1.0 at every epoch (the satellite trails the Earth by ~70°; the
  Earth–Sun separation seen from the satellite is ≥ 53.7°).

### Orbital outputs and diagnostics (PDF §7.3)

`summarize_access(access, orbit_drift=…)` → `AccessSummary`: access, occultation (any, Earth,
Sun, Moon), below-min-elevation and Sun-avoidance fractions (of grid samples = of time on the
TDB-uniform grid); Earth range min/mean/max; light time min/mean/max; range rate min/max;
pointing rate mean/max; transmitter aberration and target point-ahead maxima; maximum
elevation; light-time solver diagnostics; the diurnal-sampling diagnostic
`hour_angle_max_gap_rad` (below); and the orbit drift `|r_L1 − r_L0|` (max, final)
against `propagate.l0_reference_trajectory` (the configured elements propagated in closed form
with μ = GM☉, no planets, no SRP, shifted to the SSB with the provider's Sun; equal to Sun-only
L1 to < 1e-2 m over a year, tested).

**MVP year (2030, hourly, site 0°N 0°E, min elevation 10°):** range 1.122–1.180 au (the
satellite trails the Earth by ~70°, so 2 sin 35° au ≈ 1.15 au), light time 559.9 / 574.5 /
588.9 s (min/mean/max), range rate −1.31 … +1.33 km/s, pointing rate 1.99e-7 rad/s mean,
access fraction 0.4220, Earth occultation 0.5004, Sun/Moon occultation 0, Sun separation
53.7–56.1°, orbit drift vs L0 1.21e7 m max (planets and SRP partly cancel; M2 measured
3.25e7 m for planets alone and 3.74e7 m for SRP alone). The access computation takes ~1.2–1.6
s for the 8767 epochs.

**Sampling caveat (hourly aliasing).** The satellite's synodic day is within minutes of the
solar day, so an hourly grid samples nearly the same 24 hour angles every day. The hourly
access fraction (0.4220) is therefore quantised and differs from the analytic diurnal fraction
for an equatorial site, mean of `acos(sin 10°/cos δ)/π` = 0.4420, by 0.020 (≤ 1/24). With
5-minute steps over 30 days the two agree to < 0.005 (tested). The summary now reports this
explicitly: `AccessSummary.hour_angle_max_gap_rad` is the largest gap between the sampled
satellite hour angles at the site (`satellite_hour_angle_rad`, `max_circular_gap_rad`). A
diurnal window's fraction is resolved only to roughly gap/(2π) per window edge (an
order-of-magnitude guide, not a bound). MVP year at 3600 s: gap 8.6° (gap/2π = 0.024, against
the observed 0.020 bias); 30 days at 300 s: 0.07°. M6 metrics must use an output step that
resolves the diurnal pattern (or is incommensurate with the day) when reporting availability,
and should check this diagnostic.

### Configuration fields added in M3

All `assumed`, `numerical`, with defaults; they change the config hash (re-pinned:
`b12c5a85…9c18` → `d0087098…962d`; removing exactly these three keys reproduces the old
digest) but not `schema_version`.

- `models.light_time_tolerance_s`: 1e-9 s, range (0, 1e-3].
- `models.light_time_max_iterations`: 50, range [2, 1000].
- `models.earth_occultation_standoff_m`: 1 m, range (0, 1000].

The existing fields `receiver.min_elevation_deg`, `receiver.sun_avoidance_deg`,
`models.light_time`, `models.{earth,sun,moon}_occultation` and
`models.sun_occultation_limb_margin_deg` are used as defined in M0.

## M4: Solar collection, PV conversion and thermal feasibility

Modules: `helioswarm.solar` (irradiance, incidence, absorbed/reflected split, PV efficiency
law, `SolarPowerResult`), `helioswarm.power_chain` (the on-board part of the PDF §10.1 chain,
single owner of `eta_pc`, `eta_e2b`, `eta_optics_or_rf`) and `helioswarm.thermal` (PV node,
radiator node, `ThermalResult`, energy ledger, time-series and distance-sweep drivers). All
quantities are SI; every result is vectorised over epochs (TDB) or over heliocentric distance.

### Solar irradiance (PDF §8.1)

- `S(r) = S_1AU (au / r_sun)²` with `S_1AU = constants.solar_constant_1au_W_m2` (default
  1361 W m⁻², IAU 2015 B3) and au = 149 597 870 700 m (IAU 2012 B2).
- `r_sun` is the satellite – Sun-centre distance from the M3 `IlluminationResult`
  (`sun_distance_m`): the Sun's position from the ephemeris provider (J2000 / SSB) at its
  light-time-retarded epoch as seen from the satellite. The retarded and instantaneous Sun
  differ by ≲ 8 km at 1 au (≲ 1e-7 in flux; checked in the MVP integration test).
- Validity: for a spherically symmetric Sun the flux through a plane normal to the Sun
  direction is exactly `L/(4πr²)` for any `r > R☉` (energy conservation), so the
  inverse-square law needs no finite-disk correction at normal incidence even at 0.3 au
  (64 R☉). Neglected: solar-cycle TSI variability (~0.1 %), spectral dependence, and the
  O(v/c) ≈ 1e-4 Doppler/aberration change of the flux and of the sunlight direction in the
  satellite frame.

### Incidence and illumination

- `collector.incidence_mode = sun_tracking`: `cos(inc) = 1` at every epoch (optimistic bound,
  PDF §8.1).
- `fixed_normal`: `cos(inc) = max(n · ŝ, 0)` with the J2000 unit normal `n`
  (`collector.fixed_normal_unit_j2000`) and the satellite → Sun unit vector `ŝ`
  (`IlluminationResult.sun_direction_unit`, added in M4; retarded Sun, geometric direction).
  Back-face illumination (`n · ŝ < 0`) yields **zero** absorbed and electrical power: the
  back face carries no cells, and sunlight absorbed by the back face is **not** modelled
  (limitation: a back-lit panel is colder in the model than in reality, but it produces no
  power either way). The Sun is a point source for the incidence angle (its 4.65e-3 rad
  radius at 1 au matters only within that angle of grazing incidence).
- The M3 visible solar-disk fraction `f` (conical shadow with penumbra) multiplies the flux.
- `incident_W = S A cos(inc) f` is the sunlight intercepted by the collector plane.

### Absorbed and reflected power

- `absorbed_W = α · incident_W` (PDF §8.1 `P_abs = S A α cos(inc)`, times `f`);
  `reflected_W = (1 − α) · incident_W`. The array is opaque: "reflected" covers everything
  not absorbed.
- Reflected sunlight leaves the spacecraft and is **not waste heat**: it appears in neither
  node's heat load and is a separate term of the energy ledger. Because `P_abs` already
  excludes it, the PDF §8.1 term "explicitly_modelled_reflected_or_transmitted_power" is not
  subtracted again: `Q_PV = P_abs − P_PV`.

### PV conversion (PDF §8.2, labelled local linear approximation)

- `η_PV` is defined relative to **absorbed** power (fixed by the M0 schema):
  `P_PV = P_abs · η_PV(T)`.
- `η_PV(T) = clip(η_ref (1 + γ_T (T − T_ref)), η_min, η_max)` with `0 ≤ η_min < η_max ≤ 1`.
  This is a local linearisation about `T_ref`, not a measured efficiency–temperature curve.
  Its unclipped validity interval is `PvEfficiencyLaw.linear_range_K()` (the temperatures at
  which the line reaches `η_max` and `η_min`); every `SolarPowerResult` flags samples where
  the clip is active (`pv_efficiency_clipped`). Comparison path: a technology-specific
  measured curve can replace `PvEfficiencyLaw` behind the same `efficiency(T)` interface.
- Not modelled: spectral response, irradiance dependence of η, cell mismatch, degradation and
  radiation damage, wiring losses (all would be part of η or of `η_pc`).

### PV array thermal node (node a)

- Geometry: thin, isothermal flat plate of area `A = collector.area_m2`, front face (cells)
  towards the Sun, both faces radiating to deep space with IR emissivities
  `ε_f = thermal.pv_front_emissivity`, `ε_b = thermal.pv_back_emissivity` and view factor
  `F = thermal.pv_view_factor` from each face (M4 field, default 1.0 = isolated plate). The
  rest `1 − F` of each face's hemisphere is assumed to see surfaces at the plate temperature
  (no net exchange).
- Steady state: `P_abs − P_PV(T) = (ε_f + ε_b) σ A F (T⁴ − T_space⁴)` (Stefan–Boltzmann;
  σ CODATA 2018), `T_space = thermal.space_temperature_K`.
- Solver: `scipy.optimize.brentq` (xtol 4 ulp of `T_lo`, rtol 4ε) on a closed-form bracket.
  With `G = (ε_f + ε_b) σ A F` and `T_eq(η) = ((1 − η) P_abs/G + T_space⁴)^¼`: every root lies
  in `[T_space, T_eq(η_min)]`; the monotone law lies between `η_lo` and `η_hi` (its values at
  the two ends) there; in exact arithmetic `[T_eq(η_hi), T_eq(η_lo)]` then contains every
  root, so a solution always exists. Uniqueness is guaranteed for `γ_T ≥ 0`, and for
  `γ_T < 0` when `P_abs η_ref |γ_T| < 4 G T_lo³`; this sufficient condition is reported per
  sample (`unique_root`; MVP: always true; a pathological law with η = 1 below 289 K is
  flagged in the tests). The deviation from the plan's fixed bracket
  `[T_space + ε, T_max_bracket]` is deliberate: the derived bracket needs no arbitrary upper
  temperature and proves existence.
- Floating-point handling (M4 review). For tiny `P_abs` (a sliver of Sun at penumbra entry
  or exit; the M3 disk-overlap function gives visible fractions down to ~1e-13 within 1 µrad
  of the umbra edge) the plate temperature is within a few ulp of `T_space`, the closed-form
  bracket collapses to one or two float64 values, and its end-point signs can be flipped by
  rounding. Three measures make the solve robust down to the smallest subnormal `P_abs`:
  1. `T⁴ − T_space⁴` is evaluated as `(T − T_s)(T + T_s)(T² + T_s²)` (no cancellation;
     ≤ 4ε relative, tested against 50-digit arithmetic), in the solver, in
     `PvThermalNode.radiated_W` and in the radiator formulas.
  2. The convergence criterion is `|f(T)| ≤ 1e-12 P_abs + floor`, with the float64
     representation floor `floor = (P_abs |η_ref γ_T| + 4 G T³) spacing(T)`: no float64 `T`
     can in general do better than about half of it. It is ~4e-17 W for the MVP plate near
     `T_space` and matters only for `P_abs ≲ 1e-4 W`; elsewhere it is ≲ 1e-15 `P_abs`.
  3. A brentq result above that tolerance, or a bracket end whose sign rounding flipped, is
     polished by walking float64 neighbours (≤ 64) to the sign change. Only if all of this
     fails is the sample NaN / not converged; `evaluate_power_thermal` then raises.

  Verified: every `P_abs` from 1e-30 W to 1e8 W converges; illumination fractions 1e-11 …
  5e-324 at 1 au match a 50-digit fixed-point solution to ≤ 2 ulp, are feasible and close the
  ledger (`test_tiny_illumination_fraction_converges_and_closes`).
- Limiting case: with `η ≡ 0` and `F = 1`, `T = (α S / ((ε_f + ε_b) σ) + T_space⁴)^¼`
  (330.98 K at 1 au for α = 0.9, ε_f + ε_b = 1.8; tested).
- Feasibility: `T ≤ collector.max_operating_temperature_K` (M4 field, default 400 K,
  **assumed**, not a sourced qualification limit). MVP collector: 302.8 K at 1 au (feasible),
  559 K at 0.3 au (infeasible), crossing 400 K between 0.5 and 0.7 au.

### Power-electronics / transmitter radiator node (node b)

- Waste heat = exactly the on-board losses of PDF §10.1, each counted once, computed by
  `power_chain.onboard_power_chain` (the function M6 will reuse):
  `Q_waste = P_PV (1 − η_pc) + P_dc_to_tx (1 − η_e2b) + P_beam (1 − η_optics_or_rf) + Q_other`
  (`thermal.other_heat_load_W`).
- **Departure from the PDF §8.3 formula.** PDF §8.3 lumps the PV heat into a single
  radiator load, `Q_waste = Q_PV + Q_transmitter + Q_power_electronics + Q_other`. Here
  `Q_PV = P_abs − P_PV` is radiated by the PV plate itself (node a, which therefore sets the
  PV operating temperature), and this radiator carries only the on-board chain losses
  (`Q_transmitter + Q_power_electronics`, split into the three §10.1 stages) plus `Q_other`.
  Nothing is dropped: the whole-spacecraft ledger identity
  `incident + Q_other = reflected + Q_PV,radiated + Q_rad,rejected + P_aperture` checks the
  sum. The two-node split is the physically consistent reading of a PV array whose cells
  dissipate their heat locally; a design that pipes PV heat to a common radiator would need
  a conductive link, which is neglected (see below). Assigning the whole optics/RF loss to this radiator is
  **conservative**: spill-over radiated to space would not need rejecting.
- Geometry: a thin flat plate of physical area `A_rad` radiating from
  `n_faces = thermal.radiating_sides` faces (1: back insulated, 2: both faces exposed), each
  with view factor `F_rad = thermal.radiator_view_factor` to deep space, no view of the Sun
  or of the PV array (absorbs no sunlight), emissivity `ε = thermal.emissivity`:
  `Q_rad = ε σ A_rad n_faces F_rad (T⁴ − T_space⁴)` (PDF §8.3,
  `A_eff = A_physical · emitting_side_factor · view_factor`).
- Required area at the limit: `A_req = Q_waste / (ε σ n_faces F_rad (T_lim⁴ − T_space⁴))`,
  ∝ 1/ε and ∝ 1/(T_lim⁴ − T_space⁴) (≈ 1/T_lim⁴; tested).
- `thermal.radiator_area_m2` unset: radiator sized at the limit (`equilibrium_temperature_K
  = T_lim`, `radiator_margin_K = 0`). Set: analytic equilibrium
  `T = (Q_waste/(ε σ A_rad n_faces F_rad) + T_space⁴)^¼`, radiator feasible iff `T ≤ T_lim`,
  margins `radiator_margin_K = T_lim − T` and `area_margin_m2 = A_rad − A_req`.
- PDF-level verdict (PDF p.4 `ThermalResult` "feasible, margin"; §8.4 "feasibility flag and
  margin to temperature limit"): `margin_K = min(pv_margin_K, radiator_margin_K)`, the binding
  margin over both nodes, and `feasible = pv_converged & (margin_K ≥ 0)`. In the default
  sized-at-limit mode the radiator margin is 0 by construction, so the PV node is what can
  make a case infeasible (MVP at 0.3 au: `margin_K = −158.8 K`).
- Mass proxy: `m = A · thermal.radiator_specific_mass_kg_m2` (M4 field, optional, assumed).

### Energy ledger

`thermal.energy_ledger` checks per sample, relative to `incident + Q_other`, to 1e-12:
`incident = reflected + absorbed`; `absorbed = P_PV + Q_PV,radiated` (the solved
temperature's radiation, so this is the solver check); `P_PV = P_dc_to_tx + loss_pc`;
`P_dc_to_tx = P_beam + loss_e2b`; `P_beam = P_aperture + loss_optics`;
`Q_waste = Σ losses + Q_other`; `Q_waste = Q_rad,rejected`; and the spacecraft boundary
`incident + Q_other = reflected + Q_PV,radiated + Q_rad,rejected + P_aperture`. Measured:
≤ 6.5e-16 (MVP year and 0.3–1.2 au sweeps).

Identities that contain a solved temperature (PV node, radiator balance in the
configured-area mode, spacecraft) are checked against `1e-12 · scale + floor`, where `floor`
is the float64 representation floor of that temperature (`|dQ/dT| spacing(T)`; PV node as
above, radiator `4 ε σ A_rad n F T³ spacing(T)`). Every identity also gets the smallest normal
float64, 2.2e-308 W, the absolute precision left once powers underflow to subnormals. The
floors matter only when `incident + Q_other` is below ~1e-4 W (MVP plate).

Which identities are independent checks. The PV-node identity tests the solved temperature
against Stefan–Boltzmann. The collector identity and the radiator inventory are sums formed
independently of the code that produced their terms. The three chain identities hold by
construction, since each loss is stage input minus output in `power_chain`. The radiator
balance is a round trip, because `A_req` or `T_eq` is derived from the same `Q_waste`. The
spacecraft identity is the algebraic sum of the others. Those guard the bookkeeping against
regressions; they do not validate physics. The physics is validated by the hand-computed
waste-heat test and by `test_energy_ledger_from_hand_computed_terms`, which computes every
ledger term in closed form for a constant-efficiency plate (η = 0.3).

### Neglected effects (M4)

- Earth/planet albedo and IR on the collector: negligible at heliocentric distances. Upper
  bound at d = 0.1 au from the Earth (closer than any MVP geometry): reflected
  `A_B S (R⊕/d)²` with Bond albedo 0.3 plus emitted `240 W m⁻² (R⊕/d)²` (full disk facing the
  collector; global-mean values, Trenberth, Fasullo & Kiehl 2009, BAMS 90, 311) is
  1.2e-4 W m⁻², i.e. < 1e-7 of S.
- Conduction and mutual radiation between the PV node and the radiator node; the PV
  electrical output leaves the PV node without further dissipation there.
- Transient (thermal-capacity) response: every epoch, including eclipse, is in steady state;
  an eclipse epoch therefore drops to `T_space` (PV) instantly, a known non-physical limit
  for short eclipses (none occur in the MVP year).
- Temperature gradients through and across the plates; temperature dependence of ε and α;
  back-face absorption of sunlight (fixed attitude, see above).

### Configuration fields added in M4

All `assumed`, with defaults; they change the config hash (re-pinned `d0087098…962d` →
`31de4816…7651`; removing exactly these three keys reproduces the old digest) but not
`schema_version`:

- `collector.max_operating_temperature_K`: 400 K (> 0; the schema also requires it to exceed
  `thermal.space_temperature_K`).
- `thermal.pv_view_factor`: 1.0, range (0, 1].
- `thermal.radiator_specific_mass_kg_m2`: None (no mass proxy), > 0 when set.

## M5: Beam physics (aperture, propagation, capture, pointing, microwave, atmosphere)

Modules: `helioswarm.aperture` (aperture fields, Fresnel/Fraunhofer core, validity, wavefront
loss), `helioswarm.optical` (Airy, Gaussian beam), `helioswarm.capture` (finite receiver,
projection, pointing statistics), `helioswarm.microwave` (lattice array factor),
`helioswarm.beam` (model selection, `BeamResult`), `helioswarm.atmosphere`. Everything is
**scalar diffraction** (no polarisation, vector or near-grazing effects, no obliquity factor,
no non-linear optics or thermal distortion; PDF §9.2) and describes **one** transmitter: there
is no coherent combining across satellites (PDF §9.3; `BeamResult` refuses
`coherent_combining=True`).

### Aperture fields (normalised to the power leaving the aperture)

`∫₀ᵃ |A(r)|² 2πr dr = 1` (a = D/2; A in m⁻¹), zero outside the edge:

| Illumination | A(r) | c² |
|---|---|---|
| `uniform_circular` | c | 1/(πa²) |
| `truncated_gaussian` (w0 = 1/e² intensity radius, w0 ≤ a) | c·exp(−r²/w0²) | 1/[(πw0²/2)(1 − e^{−2a²/w0²})] |
| `parabolic_on_pedestal` (edge intensity taper T dB, C = 10^{−T/20}) | c·[C + (1−C)(1 − r²/a²)] | 1/[πa²(C + (1−C)²/3)] |

- **Spill-over owner.** The truncated field is *renormalised* inside the aperture. Source
  power outside the edge (`exp(−2a²/w0²)` for a clipped Gaussian laser) is never emitted; it
  is a loss of `transmitter.optics_or_rf_efficiency` (M0 definition) and is only reported as
  `BeamFieldDiagnostics.source_spillover_fraction`.
- **Taper efficiency** `|∫A dA|²/(πa²)` (Balanis §15.4) is a diagnostic: the propagated pattern
  already contains it, so it is never applied as an extra "aperture efficiency" (PDF §10.1).

### Propagation (Goodman, *Introduction to Fourier Optics*, 3rd ed., eq. 4-17, §4.4)

- Fresnel–Hankel: `U(ρ;z) = (2π/(λz)) e^{iπρ²/(λz)} ∫₀ᵃ A(r) e^{iπr²/(λz)} J0(2πrρ/(λz)) r dr`,
  `I = |U|²` [m⁻² per W]. Power is conserved exactly (Hankel Parseval). Evaluated with composite
  16-node Gauss–Legendre panels sized to the fastest local oscillation (chirp `r/(λz)` + Bessel
  `ρ/(λz)` cycles per metre) and to the Gaussian waist. Validity: the largest neglected
  (quartic) phase of the Fresnel expansion, `π(a + ρ_max)⁴/(4λz³)` (Goodman eq. 4-18 asks for
  `z³ ≫ π(a + ρ_max)⁴/(4λ)`), must not exceed `FRESNEL_PARAXIAL_MAX_PHASE_RAD = 0.1` rad, else
  `BeamModelValidityError` (checked at construction for the declared `ρ_max` and again on every
  field evaluation at the largest requested radius; the former limit of 1 rad was not a
  sufficient condition and was tightened in the M5 review). This is an **assumed engineering threshold, not a sufficient
  accuracy guarantee**: by the triangle inequality the absolute field error is at most
  `φ_max ∫|A| dA/(λz)`, a pessimistic bound (Goodman §4.2.2: by stationary phase only the
  region near ρ contributes, and the actual error is much smaller). `ρ_max` covers the receiver
  plus the pointing window, the physical receiver radius (exact-ellipse diagnostic) and, for
  the near-field peak-irradiance search, `a + 4λz/D`. In the
  heliocentric regime the parameter is ~1e-20.
- Fraunhofer: `I(ρ) = J(ρ/R)/R²`, `J(θ) = (1/λ²)|2π∫A J0(2πrθ/λ) r dr|²` [sr⁻¹ per W]
  (paraxial, `sin θ ≈ θ`). Closed forms (u = πDθ/λ): uniform `(πa²/λ²)[2J1(u)/u]²`;
  parabolic on pedestal `(c²(2πa²)²/λ²)[C J1(u)/u + 2(1−C) J2(u)/u²]²` (Lommel integral
  `∫₀¹(1−t²)J0(ut)t dt = 2J2(u)/u²`); truncated Gaussian by quadrature.
- **Validity enforcement.** `N_F = D²/(λR)` is computed for every epoch. Far-field models
  (`airy_far_field`, `aperture_far_field`) raise `BeamModelValidityError` if
  `N_F > models.fraunhofer_max_fresnel_number` (0.5, i.e. R ≥ 2D²/λ). Fresnel corrections to
  the far field are second order in the edge phase πN_F/4 (on axis exactly
  `1 − sinc²(πN_F/8) ≈ (πN_F)²/192`; tested). MVP year: N_F = 5.7–6.0e-4.

### Optical models

- **Airy** (uniform, far field): `I = I0[2J1(x)/x]²`, `I0 = πD²/(4λ²R²)`, `x = πDρ/(λR)`;
  `EE(x) = 1 − J0(x)² − J1(x)²` (Rayleigh); first null `x = j₁,₁ = 3.8317`, i.e.
  `1.21967 λR/D`, enclosing 0.8378 of the power (Born & Wolf §8.5.2). Small-x: `EE ≈ x²/4`.
- **Gaussian beam** (separate model, not a label for the truncated Gaussian): untruncated TEM00
  with waist w0 in the aperture plane (collimated), `w(z) = w0√(1 + (z/z_R)²)`,
  `z_R = πw0²/λ`, `I = (2/(πw²))e^{−2ρ²/w²}`, `EE = 1 − e^{−2ρ²/w²}` (Siegman 1986, ch. 17).
  Valid at any range; accepted only for `a/w0 ≥ models.gaussian_beam_min_truncation_ratio`.
  Comparison path: `fresnel_hankel` with the same truncated field; their encircled energies
  agree to 2e^{−(a/w0)²} (first-order edge-field effect, tested at a/w0 = 3) and differ by
  > 1 % at a/w0 = 1.

### Finite-receiver capture (PDF §9.4)

`η(d) = ∫ I(ρ) 2φ(ρ;d,a) ρ dρ`, `φ = arccos(clip((ρ² + d² − a²)/(2ρd), −1, 1))`, split into
`EE(max(a−d,0))` plus the band `|a−d| ≤ ρ ≤ a+d` integrated with the substitution
`ρ = m − h cos t` (removes the square-root endpoints) and 8-node Gauss–Legendre panels
≤ ℓ/2 in ρ (ℓ = λR/D, or w/2 for the Gaussian beam). Exact up to quadrature (≤ 1e-12
measured). Checks: d = 0 equals EE; Gaussian beam equals the non-central χ² closed form
`ncx2.cdf(4a²/w², 2, 4d²/w²)` to 1e-11; a 2-D Cartesian grid converges to it (2e-5 at the
finest grid); near-field (N_F = 8) pattern agrees with the grid.

### Receiver projection (`receiver.orientation`, new field)

- `horizontal` (default): the disk projects to an ellipse `a × a sin(el)` normal to the beam.
  **Named approximation:** the circle of equal projected area, `a_eff = a√(sin el)`. The
  intensity variation over the receiver depth (`≤ 2a/R ~ 1e-8` relative) is neglected.
  The circle is slightly optimistic (closer to the peak). Error vs the exact centred ellipse
  (polar form `η = (1/2π)∮EE(r(ψ))dψ`, `capture_fraction_centred_ellipse`), Airy beam,
  ℓ = λR/D:

  | a/ℓ | el = 10° | el = 30° | el = 60° |
  |---|---|---|---|
  | 0.05 | +1.1e-3 | +3.9e-4 | +2.8e-5 |
  | 0.07 (MVP) | +2.1e-3 | +7.5e-4 | +5.4e-5 |
  | 0.2 | +1.7e-2 | +6.1e-3 | +4.3e-4 |
  | 0.5 | +0.10 | +3.5e-2 | +2.3e-3 |
  | 1.0 | +0.36 | +8.7e-2 | +3.3e-3 |

  Second-order prediction for a receiver ≪ beam: relative error `κ(a − b)²/4` with the
  pattern curvature κ (Airy: π²/(4ℓ²)); tested to 10 %. Every `BeamResult` reports the
  per-epoch error `diagnostics.receiver_projection_error`, so a run where the approximation
  is poor (receiver comparable to the beam at low elevation) is visible.
- `tracking`: the receiver normal follows the beam, `a_eff = a`.

### Pointing (PDF §9.5)

Bias magnitude b and isotropic 2-D Gaussian jitter σ per axis (small angle) give a beam-axis
offset at the receiver `d ~ Rice(ν = Rb, s = Rσ)` (scipy `rice(b = ν/s, scale = s)`, verified
from its source and against the explicit density). `E[η] = ∫η(d)f(d)dd` by deterministic
Gauss–Legendre quadrature over `[max(0, ν − 9s), ν + 9s]` (tail < 3e-18) with panels
≤ min(s, ℓ)/2. Percentiles p05/p50/p95: exact (`η(d_{1−q})`) when η is non-increasing on the
nodes (`exact_monotone`); otherwise (`level_set`) η(d) is sampled every min(s, ℓ)/32 across the
window, interpolated by a cubic spline, and `P(η ≤ y)` is the exact offset-law mass of the level
set `{d: η̃(d) ≤ y}` (spline roots + CDF differences), inverted by bisection in y to 1e-13.
Accuracy: ≤ 1.2e-9 absolute against an independent brentq root-finding reference (test
tolerance 1e-8); the earlier node-step quantile (review M5) was off by up to 2e-3.
**Large bias-to-jitter ratio (named approximation).** `scipy.stats.rice` returns NaN from
`cdf`/`ppf` for `b = ν/s ≳ 2.5e5`; for `b ≥ 1e4` the offset law is its Gaussian limit
`N(ν + s²/(2ν), s²(1 + s²/(2ν²)))` (quantile error ≈ 0.83 s/b², CDF error ≈ 0.12/b², measured
against scipy), with the expectation integrated in the standardised variable so the probability
mass is exact even when `ν ± 9s` collapse onto ν in floating point. Tests: Gaussian-beam closed form
`E[η] = ncx2.cdf(a²/v', 2, ν²/v')`, `v' = w²/4 + s²` (1e-9); Monte Carlo with an explicit
Generator (Airy); monotone non-increasing with σ for the centred baseline (Gaussian beam: any
receiver size, by Anderson's theorem; Airy: receiver inside the central lobe).
**Not a theorem for ring patterns:** with a receiver cutting the first Airy ring (a = 1.5ℓ)
jitter *raises* the expected capture (confirmed by Monte Carlo); this is physics, recorded in a
test, and `pointing_loss = E[η]/η(0)` may then exceed 1 (also possible in the Fresnel near field).

### Factors combined in `BeamResult` (one owner each)

`capture_fraction = η_wavefront · η_lattice · E[η_diffraction]`, a fraction of the beam power
reaching the receiver plane (`P_atm`):

- `η_wavefront = exp(−(2πσ_w/λ)²)` (Ruze 1966 / extended Maréchal), single owner
  `aperture.wavefront_efficiency`, σ_w = `transmitter.rms_wavefront_error_m` (wavefront, not
  surface, error). The scattered fraction is assumed **not** captured (conservative when the
  phase-error correlation length is ≪ D, so the halo is ≫ the diffraction pattern).
- `η_lattice` (microwave lattice only, else 1), see below.
- The atmosphere is **not** in `capture_fraction`; the link budget (M6) multiplies
  `η_atmosphere` from `atmosphere.py`.
- `peak_irradiance_per_W_m2` is the aberration-free pattern maximum (on axis for the far-field
  models; grid maximum + parabolic refinement for Fresnel, which can peak off axis) — an upper
  bound for the beam-abort constraint `receiver.max_power_density_W_m2`, which is checked
  against the beam-normal irradiance when the incident power is supplied.

### Microwave (PDF §9.3)

- Filled tapered aperture: the same Fraunhofer/Fresnel machinery with the configured taper
  (`aperture_far_field`, `fresnel_hankel`).
- Sparse uniform square lattice (`array_element_spacing_m` + new
  `array_element_pattern_exponent` q ≥ 1): array factor `AF = D_n(ψx)D_n(ψy)`,
  `D_n(ψ) = sin(nψ/2)/sin(ψ/2)`, ψ = 2πd·u/λ (Balanis eq. 6-89); grating lobes at
  `(u,v) = (m,n)λ/d`. Large-array main-lobe fraction
  `η_lattice = 1/Σ_visible cos^{q−1}θ_mn` (Parseval per period cell, `dΩ = du dv/cosθ`; 1 for
  d < λ, 1/N_visible for q = 1). Checked against direct hemisphere integration of
  `cos^q|AF|²` for a 24 × 24 lattice (3 %, finite-N). Validity enforced: ≥ 100 elements,
  no grazing lobe, receiver + 9σ + bias within 0.25 of the grating-lobe separation.
  Broadside lattice assumed (beam steered by pointing the array).

### Atmosphere (PDF §10.2–10.3)

Terminal segment only; depends on elevation, site and carrier, never on range.

- `none`: η = 1.
- `scenario`: optical `η = τ_z^X`, microwave `A = A_z X` dB, plane-parallel `X = 1/sin el`
  (valid above `atmosphere.plane_parallel_min_elevation_deg`; Kasten & Young 1989 provided as
  the comparison function: +3.1 % at 10°, +11 % at 5°). Optical `τ_z = 0` (default cloudy) is a
  hard outage (`weather_available = False`).
- `itu_r` (new option, microwave only, `itur` extra): ITU-R P.618-13 slant-path
  `A = A_gas + A_cloud + A_rain` exceeded `atmosphere.itu_r_exceedance_percent` (new field,
  0.001–5 %) of an average year at each site, gases P.676-12 Annex 2, clouds P.840-7 (versions
  pinned on every call and returned). Scintillation excluded (P.618 antenna-averaging model is
  for earth stations of a few metres, not km-scale rectennas). Rain polarisation tilt τ from
  `atmosphere.itu_r_polarization_tilt_deg` (default 45° = circular; returned in the result).
  Validation status: the wrapper is validated against the ITU SG3 examples with P.676-11 /
  P.1511-1 pinned; the production P.676-12 / P.1511-2 path is regression-pinned only (no
  validation set in `itur` 0.4.0). Thread safety: a module lock serialises the wrapper, which
  pins the revisions and restores `itur`'s previous global revisions on exit;
  `ITU_R_VERSIONS` is a read-only mapping.
  Frequency 1–55 GHz and `receiver.min_elevation_deg ≥ atmosphere.itu_r_min_elevation_deg`
  (new field, default 5°, P.676 Annex 2 range) are schema-enforced. Site ellipsoidal height
  used as height above mean sea level (geoid undulation neglected). A long-term statistic
  applied deterministically (a design level), not a weather time series.
- Elevations below the model floor give NaN with `in_validity = False` (never extrapolated).

### Configuration fields added in M5

All `assumed`; config hash re-pinned `31de4816…7651` → `874f79f0…cb4b` (removing exactly the
four keys reproduces the old digest), and in the M5 review `874f79f0…cb4b` → `ad186cc3…5b62`
for the tilt field (removing that one key reproduces `874f79f0…cb4b`); `schema_version`
unchanged.

- `receiver.orientation`: `horizontal` (default) | `tracking`.
- `transmitter.array_element_pattern_exponent`: q, required with and only with
  `array_element_spacing_m`, range [1, 20].
- `atmosphere.itu_r_exceedance_percent`: 1.0 %, range [0.001, 5].
- `atmosphere.itu_r_min_elevation_deg`: 5°, range [5, 90).
- `atmosphere.itu_r_polarization_tilt_deg` (review M5): 45° (circular), range [0, 90].
- `models.atmosphere = itu_r` is rejected at validation when the optional `itur` package is not
  installed (review M5).
- `models.atmosphere` gains `itu_r`; `weather_scenario = cloudy` is now rejected for every
  model except `scenario`.

### Neglected effects (M5)

Polarisation and vector diffraction; obliquity factor; atmospheric turbulence/seeing and beam
wander (PDF §10.2: only with a specified receiver and wavefront-correction architecture);
refraction-induced beam bending; thermal blooming; aperture obscuration (secondary mirror,
struts); partial coherence and finite laser linewidth; the scattered (halo) part of the
wavefront-error power; receiver depth along the beam; tropospheric scintillation (ITU-R
option); grating-lobe power that might reach other receivers.

## M6: Link budget, system orchestration, metrics and run outputs

Modules: `link_budget.py`, `system.py`, `metrics.py`, `io.py`, `manifest.py`, `plotting.py`,
`report.py`, `sweep.py`, `validation.py`, `capture_batch.py` (optional JAX backend). Implementation
notes and measured numbers: `docs/milestones/M6.md`.

### Link budget (PDF §10.1, §10.4; `helioswarm.link_budget`)

```
P_PV,tx    = P_PV · 1[transmitting]                      operating decision (system)
P_dc_to_tx = P_PV,tx · η_pc        P_beam = P_dc_to_tx · η_e2b        P_aperture = P_beam · η_o
                                                        (power_chain.onboard_power_chain, M4)
P_atm      = P_aperture · η_atm                         (atmosphere.atmospheric_transmission, M5)
P_captured = P_atm · η_capture                          (BeamResult.capture_fraction, M5)
P_accepted = min(P_captured, P_max_in, Q_max/(1 − η_rx))  receiver limits
P_receiver = P_accepted · η_rx      P_grid = P_receiver · η_grid
```

- **One owner per factor.** The on-board stages come only from `onboard_power_chain` (the same
  function the M4 radiator uses). `η_capture` already contains diffraction (pattern normalised
  to `P_aperture`, so there is no extra aperture efficiency; the taper is inside the pattern),
  pointing (Rice expectation), the Ruze wavefront factor and the lattice grating-lobe factor,
  each applied once in `beam.evaluate_beam`. Gaussian spill-over belongs to `η_optics_or_rf`.
  `η_atm` is applied once and depends only on elevation.
- **Invariants** (checked at construction of every `LinkBudgetResult`): every η in [0, 1], and
  every stage ≤ the previous stage. Multiplying by a factor in [0, 1] is monotone in IEEE
  arithmetic (`fl(x·η) ≤ x`), so the checks are exact.
- **Disabled epochs** (link not enabled by the availability conditions) carry 0 W in every stage
  from `P_PV,tx` on; their atmosphere and capture factors are recorded as 0 (transmission of a
  blocked / unevaluated path). `P_PV` is still reported (generated but not transmitted).
- **Receiver limits** are engineering constraints of the chain (PDF §10.4), not post-processing:
  the input limit acts on `P_captured` (M0 field definition); the heat-rejection limit is
  enforced as `P_accepted ≤ Q_max/(1 − η_rx)`; both are *clipping* (the link stays available,
  flag `RECEIVER_SATURATION`). The power-density limit is a *beam abort* evaluated on the peak
  irradiance of the aberration-free pattern on the beam-normal plane, `P_atm · I_peak/W` (an
  upper bound: wavefront and pointing errors only lower the peak).
- **Technology cases.** Optical (`laser_photovoltaic`) and microwave (`rectenna`) receivers are
  separate configurations with their own conversion efficiencies and limits; nothing is shared
  between them in code.

### Availability conditions and reason codes (`helioswarm.system`)

Per epoch a `uint32` bitmask `reason_code` records every condition that holds:

| Bit | Reason | Blocking | Source |
|---|---|---|---|
| 0 | `BELOW_MIN_ELEVATION` | yes | M3 apparent elevation < `receiver.min_elevation_deg` |
| 1 | `EARTH_OCCULTATION` | yes | M3 beam segment enters the WGS 84 ellipsoid |
| 2 | `SUN_OCCULTATION` | yes | M3 beam passes within the solar limb + margin |
| 3 | `MOON_OCCULTATION` | yes | M3 beam enters the Moon |
| 4 | `SUN_AVOIDANCE` | yes | M3 Sun–receiver–satellite angle < `receiver.sun_avoidance_deg` |
| 5 | `ECLIPSE` | yes | M3 illumination fraction 0 (umbra; no power to transmit) |
| 6 | `THERMAL` | yes | M4 `ThermalResult.feasible` false |
| 7 | `WEATHER` | yes | hard optical outage of the weather scenario (zenith transmittance 0) |
| 8 | `POINTING` | yes | `E[η]/η(0) < transmitter.min_pointing_efficiency` (if set) |
| 9 | `POWER_DENSITY_ABORT` | yes | beam abort (above) |
| 16 | `RECEIVER_SATURATION` | no (limiting) | receiver input/heat clip |
| 17 | `PARTIAL_ECLIPSE` | no (limiting) | 0 < illumination fraction < 1 |

Available = no blocking bit; then and only then `P_grid > 0` is possible. The *primary* reason
of an unavailable epoch is the first blocking bit in the table order (geometry before spacecraft
state before link conditions), so primary fractions sum to `1 − availability`.

Policies (the schema defines no alternatives, so none is offered):

- **Thermal infeasibility → transmitter not operated.** The thermal state is evaluated at the
  full operating point (transmitter on) at every epoch; the off-state PV temperature (all
  absorbed power dissipated in the cells, or shunted) is not modelled.
- **Eclipse.** Umbra blocks; penumbra only scales `P_PV` through the M4 illumination fraction.
- **Weather.** The MVP weather is a deterministic scenario over the whole window (PDF §10.2);
  cloud statistics are a later extension.
- **Single receiver site.** Several sites need receiver scheduling to avoid double counting
  (PDF §10.4/§11.2, M7), so `run_single_satellite` rejects them.
- **Determinism.** No stochastic step exists in M6 (pointing jitter enters through deterministic
  quadrature); the run takes an explicit `numpy.random.Generator` seeded from `project.seed`
  and records that it was not drawn from.

### Capacity factor denominator (rated power)

`rated_power_W` is the `P_grid` the configured system delivers at reference conditions:
satellite at the window-mean heliocentric distance, collector at normal incidence and fully
illuminated, receiver at the window-mean satellite range with the beam at zenith (elevation 90°:
full receiver projection, air mass 1), the configured pointing errors, receiver limits applied.
Atmosphere at the reference: for `models.atmosphere = scenario` the **clear-sky** scenario values
replace the configured weather (`none`: transmission 1); for `itu_r` the configured ITU-R
gas + cloud + rain attenuation exceeded `itu_r_exceedance_percent` of the time is used at zenith
(the ITU-R path has no separate clear-sky state, so this reference is **not** clear sky; for the
microwave MVP, p = 1 %, η_atm,ref = 0.987). `capacity_factor = ⟨P_grid⟩ / rated_power_W`; it can
exceed 1 only if the actual geometry is better than the window means (it is ≈ availability ×
(mean capture at actual elevations / zenith capture)).

The run's operating policies apply to the reference exactly as to any epoch (M6 review): a
thermally infeasible reference operating point is not operated and a reference beam above the
power-density limit is aborted; in both cases `rated_power_W = 0` and the capacity factor is
undefined (null). `summary.json` (`capacity_factor_denominator`) and `sweep_summary.parquet`
carry `reference_thermal_feasible`, `reference_power_density_abort` and
`reference_eta_atmosphere`, so a zero or null value always states its cause. (Before the M6
review the thermal verdict was not applied to the reference: thermally infeasible sweep points at
0.3/0.5 au reported a non-zero, undeliverable rated power with capacity factor 0.)

### Metrics (PDF §10.5, `helioswarm.metrics`)

- **Sampling convention.** Sample k represents `[t_k − Δt/2, t_k + Δt/2]` clipped to the window
  (trapezoid weights: `Δt` inside, `Δt/2` at the ends). Delivered energy `E = Σ w_k P_grid,k`,
  time-mean power `E/T`, availability and reason fractions are weighted the same way.
- **Percentiles** (median, p05, p95) are unweighted sample percentiles (numpy `linear`).
- **Outages** are maximal runs of unavailable samples; duration = sum of their weights,
  resolution ±Δt at each end; outages touching the window ends are reported as censored.
- **End-to-end efficiency** = `E_grid / E_incident` with `E_incident` the sunlight incident on
  the collector; also the mean of `P_grid/P_incident` over available epochs.
- **Diurnal aliasing diagnostic.** `hour_angle_max_gap_rad = g` (largest unsampled gap of the
  satellite's hour angle at the site, M3), effective distinct phases `2π/g`, per-edge resolution
  `g/2π`; a warning is raised (summary, run log, report) when `g/2π > 0.005`.
- **Time axis.** Metrics use the emission epochs. Power arrives at `t_a = t_e + τ`;
  `dt_a/dt_e − 1` and the Doppler energy factor are of order `|ṙ|/c ≤ 1e-5` for the MVP, so the
  received-power time axis and energy differ by ~1e-5 relative (neglected; named).

### Run outputs, provenance and reproducibility (PDF §5, §14.1)

`config_resolved.yaml`, `manifest.json` (config hash, schema version, git commit + dirty flag,
Python and package versions via `importlib.metadata`, OS, CPU model, host, Slurm ids, JAX
devices, seed, kernel names/URLs/SHA-256, ITU-R recommendation versions and polarisation tilt,
time-scale and frame conventions, wall-clock start/end, per-stage wall time),
`timeseries.parquet` (zstd; units in column names; frame/origin/time scale and the constant
efficiencies in the schema metadata), `summary.json` (computed from the saved table), `figures/`
and `report.md` (drawn from saved files only), `logs/run.log`. Wall-clock values appear only in
the manifest and the log, so `summary.json` and the time-series content hash
(`timeseries_content_sha256`, SHA-256 of the Arrow IPC stream) are bitwise reproducible.

### Batched capture kernel and optional JAX/GPU backend (`helioswarm.capture_batch`)

The same Airy capture physics as M5 with fixed quadrature shapes per batch (at least the M5
panel resolution for every point), evaluated as one array program on numpy or JAX. JAX 0.12 has
no accurate `j0/j1` (probed: `jax.scipy.special.bessel_jn` returns NaN for x < 1 and O(1) errors
for x > 60), so the JAX path computes `J0, J1` by the N-point periodic trapezoidal rule of their
integral representations, exact up to aliasing terms `J_{N∓n}(x)` (< 1e-16 with
`N − 1 ≥ x + 12 x^{1/3} + 30`; tested to 1e-13 against scipy). Percentiles are exact on monotone
`η(d)`; otherwise the point falls back to the M5 per-point reference. Selected with
`--backend jax` (airy_far_field only; any other model raises `NotImplementedError`). The CPU
reference remains the default.

### Additions of the M6b completeness audit

- **PDF §7.3 orbital diagnostics in `summary.json`** (`orbital_diagnostics`, summary layout
  `helioswarm-summary/2`), computed from the saved time series: Earth range statistics, access
  fraction, occultation fraction (any body and per body; trapezoid time weights), inertial
  beam direction (first/last epoch; every epoch in the new `transmit_dir_{x,y,z}` columns of
  `timeseries.parquet`, layout `helioswarm-timeseries/2`) and pointing-rate statistics, light
  time min/mean/max, range-rate statistics and the orbit drift against the L0 reference
  (max/final). Definitions are written with the numbers.
- **Typed summary record.** `summary.json` is produced and read as the frozen pydantic model
  `metrics.RunSummary` (unknown keys rejected); a one-sample window gives `null` (not NaN) for
  time means.
- **Manifest** (`helioswarm-run-manifest/2`): `units` (SI convention) and `simulation_window`
  (configured start UTC, last-epoch UTC via SPICE `et2utc`, first/last TDB epochs, step, epoch
  count, orbit element epoch) (PDF §14.1 "simulation start/end time").
- **Idealised-settings labels** (PDF §13.4): `config.idealised_settings` lists the active
  settings that make a stage lossless or a constraint inactive (the "Optimistic defaults"
  above, evaluated on the actual configuration); they are listed in `report.md`, in
  `characterization.json`, on the power-stage and composite figures and on the `characterize`
  power/thermal and capture figures, and the Sun-tracking collector is labelled an optimistic
  bound on the solar figure. A disabled occultation check is written to `summary.json` as
  `null` (not evaluated), not as a measured 0.
- **Named approximations** (PDF §17): `config.approximations_in_force` (radial SRP on a
  fixed-normal collector; instantaneous geometry with `models.light_time = false`) is written
  as a WARNING to `logs/run.log`, to `manifest.json` (`approximations`) and to `report.md`,
  each with its comparison path.
- **Response curves** (`helioswarm characterize`, `helioswarm.characterize`; PDF §8.4, §9.5):
  the M4 model versus heliocentric distance and transmitter operating point
  (`eta_electrical_to_beam`) at normal incidence in full Sun; the M5 beam capture versus pointing
  sigma, receiver radius and RMS wavefront error at one explicit link geometry (range and
  elevation recorded with the output). Each row changes exactly one parameter of the validated
  configuration; no new physics.

### Configuration field added in M6

- `transmitter.min_pointing_efficiency` (default `None` = not enforced, `assumed`): pointing
  availability condition (PDF §10.4).

### Neglected effects (M6)

- Receiver-side scheduling and capacity sharing (M7), cloud statistics and site diversity.
- Off-state thermal behaviour; thermal inertia (M4 steady state).
- Arrival-time/Doppler rescaling of received power (~1e-5).
- The capacity-factor reference uses window means, not a design point; it is a normalisation,
  not a nameplate rating.

