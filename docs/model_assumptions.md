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
| Atmosphere | `P_atm = P_aperture · η_atm`; zenith scenario factors scaled with the plane-parallel air mass `X = 1/sin(el)` | `atmosphere.*`, `models.atmosphere` |
| Capture | `P_captured = P_atm · η_capture` | (beam model, M5) |
| Receiver input limit | `P_accepted = min(P_captured, P_max_in)`; the excess is reported as receiver saturation. The limit acts on captured beam (optical/RF) power, **before** conversion. | `receiver.max_input_power_W` |
| Receiver conversion | `P_receiver = P_accepted · η_rx` (equal to `P_captured · η_rx` while the receiver is not saturated) | `receiver.conversion_efficiency` |
| Receiver waste heat | `Q_rx = P_accepted · (1 − η_rx)` must not exceed the receiver's thermal rejection capability (not enforced when unset). Not a power stage; it is a constraint. | `receiver.max_heat_rejection_W` |
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

Studies must override these defaults with sourced values or sweep them.

## Per-milestone sections

Milestones M2–M6 add the ephemeris, access, solar/thermal, beam and link-budget models here,
each with equations, validity ranges and references. M2 follows M1 below.

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
`attitude` is `"sun_facing"`, which is implied by the radial SRP model.

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
