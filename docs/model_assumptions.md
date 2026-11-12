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
  Conversion to TDB will happen only through SPICE (M2). UTC and TDB are never mixed implicitly.
  Leap-second epochs (`23:59:60`) are rejected as configuration epochs.
- **Output time grid.** `duration_days × 86400` must be an integer multiple of `output_step_s`,
  so the grid ends exactly at the window end. The check is done in seconds:
  `|duration_s − n·output_step_s| ≤ 8 ε · duration_s` with `n = round(duration_s/output_step_s)`
  and ε the float64 machine epsilon. The tolerance absorbs only float rounding (≤ 0.7 µs for the
  longest allowed window of 100 years).
- **Integrator step.** `time.integrator_step_s` is the *maximum* step of the adaptive L1
  integrator (`scipy.integrate.solve_ivp(max_step=…)`). It is required for the L1 model and
  rejected with `two_body_kepler`. Integration tolerances (rtol/atol) are added in M2.

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

Milestones M2–M6 will add the ephemeris, access, solar/thermal, beam and link-budget models
here, each with equations, validity ranges and references.

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
