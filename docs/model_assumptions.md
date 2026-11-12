# Model assumptions

This file lists every equation, assumption, validity range and simplification used by
`helioswarm` (PDF §14.4). Each physics milestone appends its section. Nothing listed here is a
validation claim; validation outcomes live in `docs/validation_report.md`.

## Conventions (M0)

- **Units.** All computation is in SI: m, s, kg, W, rad, K. The astronomical unit, degrees, days
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

Milestones M1–M6 will add the orbit, ephemeris, access, solar/thermal, beam and link-budget
models here, each with equations, validity ranges and references.
