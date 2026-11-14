# Validation report

This report lists every test, independent comparison, tolerance and unresolved discrepancy
(PDF §13, §14.4). From M6 on, the physics sections are generated from real runs by
`helioswarm validate-models --report docs/validation_report.md`. No entry is written here
before the corresponding check has been run.

## M0: configuration and tooling

M0 contains no physics models, so there is nothing to validate physically. The configuration
contract is verified by the automated test suite (`tests/unit/test_config.py`,
`tests/unit/test_cli.py`, `tests/integration/test_shipped_configs.py`). The results are recorded
in `docs/milestones/M0.md`, section D.

## Physics validation (M1–M6)

| PDF section | Check | Tolerance | Result |
|---|---|---|---|
| §7.4 | Circular two-body orbit preserves radius and energy — L0 closed-form baseline | 1e-12 relative | **pass, baseline only** (M1 unit tests). L0 is closed form and conserves radius and energy by construction, so this checks implementation round-off, not an integrator: max abs(r/a − 1) = 4.4e-16 over 103 periods; energy deviation ≤ 1.1e-15 (e = 0), ≤ 2.5e-13 (e = 0.99) over 10 periods |
| §7.4 | Circular two-body orbit preserves radius and energy *within the selected integrator tolerance* — L1 numerical propagator (planets off) | 1e-12 (= rtol) | **pass** (M2). L1 Sun-only (Sun at rest), circular 1 au, i = 0.3 rad, 3 years, rtol 1e-12, 1-day step cap: max abs(r/a − 1) = 4.7e-15, energy drift 5.6e-15 (`test_l1_propagation::test_circular_orbit_radius_is_preserved_within_integrator_tolerance`) |
| §7.4 | Orbital period matches Kepler's third law | 1e-10 relative | **pass** (M1). The period was measured independently by a DOP853 integration (rtol 1e-12) with a return-direction event for a ∈ {0.4, 1, 5.2}, e ∈ {0, 0.5, 0.9}; max deviation from 2π√(a³/μ) is 2.5e-12. T(1 au) with GM☉(DE440) equals the Gaussian year 2π/k to 1e-10 |
| §7.4 | Earth–Sun distance agrees with SPICE at selected epochs | 1 m (Horizons); 11.2 km (epv00) | **pass** (M2). de440s vs JPL Horizons (DE441) at 6 TDB epochs 2000–2050, including the 2024 perihelion and aphelion: position ≤ 5.4e-2 m, velocity ≤ 1.5e-7 m/s. vs ERFA epv00 at 41 epochs 1900–2100: 7.1 km, 3.1 mm/s (documented epv00 maxima 11.2 km, 5.0 mm/s) |
| §7.4 | Synthetic fixed geometry gives the analytic light time | 1e-14 relative (static); tolerance + 2e-11 s (moving site) | **pass** (M3). Static: τ = d/c to rounding; uniformly moving site: fixed point = closed-form quadratic root for tolerances 1e-9 and 1e-12 s; 0.9c receding site: brentq fallback = closed form to 2e-9 s (`test_visibility.py`) |
| §7.4 | Receiver site makes one sidereal rotation per sidereal day | 1e-6 rad; 1e-4 deg | **pass** (M3). IAU_EARTH (pck00011): the site's J2000 direction returns after one model period 86 164.0989 s (and after the IERS stellar day) to < 1e-6 rad; it turns 0.9856235° beyond a full turn per 86 400 s; `sxform` = closed form from the PCK constants to 3e-11 (`test_ground.py`) |
| §7.4 | Beam segment intersecting the Earth/Sun is flagged as occulted | flips at R_block(1 ± 1e-6); horizon ± 1e-5 rad | **pass** (M3). Earth: through-Earth beams flagged, flips at the geodetic horizon (lat 0°, 60°), moving Earth handled in its own frame, ~2900 random segments identical to SPICE `surfpt`; Sun: flips at R☉ and at the limb + 0.5° margin, superior conjunction flagged (synthetic and with de440s); Moon: flips at R_moon, moving Moon = brute-force photon path (`test_visibility.py`, `test_access_mvp.py`) |
| §8.5 | 1 au, normal incidence recovers S_1AU; 1/r² scaling; zero at eclipse; energy closure; A_rad ∝ 1/(εT⁴) | exact; 4e-15; exact; 1e-12; 1e-14 | **pass** (M4). S(1 au) = configured S_1AU bit for bit (also for 1360.8, 1500); S r² const. to 4e-15 over 0.3–1.2 au; zero incidence, back face and umbra give exactly 0 W; ledger residual ≤ 6.5e-16 (8 identities, of which the PV node, collector and inventory sums are independent checks and the rest bookkeeping; plus a fully hand-computed ledger); tiny penumbra illumination (down to 5e-324) converges and closes; A_req ratios for ε, T_lim, faces, F to 1e-14 (`test_solar.py`, `test_thermal.py`, `test_power_thermal_mvp.py`) |
| §9 / §13.1 | Airy first-null scaling, total power normalisation, encircled energy, capture convergence, pointing monotonicity | 1e-14 (null); 1e-6 (power); 1e-14 (EE); 2e-5 (grid) | **pass** (M5). First null 1.21967 λR/D (I < 1e-25 I0; numerical minimum 1e-6) for 5 (D, λ, R); Airy power integrated to x = 1e6 is 1 within 1e-6; EE(j₁,₁) = 0.83778; 1-D capture = 2-D grid (2e-5) and = non-central χ² closed form (Gaussian beam, 1e-11); E[η] non-increasing with σ in the centred baseline; documented counter-example for a receiver cutting the first Airy ring (`test_optical.py`, `test_capture.py`, `test_aperture.py`) |
| §13.2 | End-to-end integration tests | see generated block | **M6**: run programmatically by `helioswarm validate-models` (generated block at the end of this report) and by `tests/integration/test_end_to_end.py` |
| §13.1 (ground/visibility) | Known station coordinates | 1.5 mm per component | **pass** (M3 review). NGS datasheet KG0640 (MEADES RANCH RESET, NAD 83(2011)/GRS 80) published geodetic → published X, Y, Z: ≤ 0.46 mm (`test_ground.py::test_published_station_meades_ranch_reset`) |
| §13.3 | Independent checks (REBOUND, SPICE/JPL, HCIPy, ITU-R) | M2–M6 | REBOUND and SPICE/JPL **pass** (M2, see below); SPICE `spkcpt` XCN/XCN+S end-to-end light time and point-ahead **pass** (M3 review); HCIPy **pass** (M5: far-field profiles of uniform, truncated-Gaussian and parabolic apertures to 2e-3 at λ = 1 µm, D = 10 m, including the absolute J(0) = πa²/λ², Airy EE to 3e-3; `test_hcipy_crosscheck.py`); ITU-R **pass for the wrapper and revision wiring, validated with P.676-11 / P.1511-1 / τ = 0** (M5: `itur` 0.4.0 reproduces two ITU-R SG3 validation examples of P.618-13 gas/cloud/rain to 1e-5 dB with those revisions pinned; `test_atmosphere.py`). **Open:** the production revisions P.676-12 and P.1511-2 (and the newer in-force P.676-13, P.618-14) have no validation set in `itur` 0.4.0, so the production gas component is regression-pinned only, not validated (review M5) |

## M1: analytic orbit (L0): independent checks

These numbers come from `tests/unit/test_{constants,frames,kepler,orbit,propagation}.py`. The
details are in `docs/milestones/M1.md`.

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| L0 vs direct numerical integration, e = 0.3, one orbit | scipy `solve_ivp` DOP853 (no shared code) | < 1e-10 a, and improving ≥ 50× from rtol 1e-10 to 1e-13 | 2.3e-12 a at rtol 1e-13 (1.2e-9 a at rtol 1e-10) |
| Elements → state | textbook R₃(−Ω)R₁(−i)R₃(−ω) perifocal construction | 1e-14 relative | pass (500 seeded cases) |
| Elements → state → elements | seeded grid, 2000 cases, 1e-3 ≤ e ≤ 0.95 | a 1e-12 relative; e, i 1e-13; e·Δω, e·Δν, sin i·ΔΩ 1e-13 | a ≤ 2.1e-14; others ≤ 2.5e-15 |
| Near-singular state round trip | e ∈ {0 … 1e-6}, i ∈ {0 … π} | 1e-12 relative | pass |
| Return after k whole periods (k ≤ 10) | exact periodicity | 1e-11 relative | ≤ 2.8e-14 |
| Kepler solver | scipy `brentq`; residual | 1e-13 rad; residual ≤ 4 ulp(π) | pass |
| ECLIPJ2000 → J2000 | ecliptic pole → (0, −sin ε₀, cos ε₀) | 1e-15 | pass; M2 SPICE `pxform` cross-check: identical (difference 0) |
| GM☉ | Gaussian constant k²au³/d² | 1e-10 relative | 5.0e-12 |
| σ | 2π⁵k_B⁴/(15h³c²) with exact SI constants | 1e-10 relative | 3.3e-11 |

## M2: kernels, time, ephemerides, SRP, L1: independent checks

These numbers come from `tests/unit/test_{kernels,timescale,ephemeris,frames,srp,l1_propagation}.py`
and `tests/integration/test_rebound_crosscheck.py`. Every SPICE and REBOUND test ran; none was
skipped. Details are in `docs/milestones/M2.md`.

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| Kernel integrity | manifest SHA-256 (TOFU), NAIF MD5 for de440s | exact | all 4 verified; de440s MD5 = NAIF `aa_checksums.txt` |
| UTC → TDB, 9 epochs 2000–2030 | IERS Bulletin C ΔAT + 32.184 s + USNO Circ. 179 TDB − TT | 1e-5 s | 3.6e-7 s |
| UTC → TDB, 8 epochs 2000–2025 | astropy/ERFA (full FB series) | 1e-4 s | 2.7e-5 s |
| Leap second 2016-12-31 | definition | 1e-6 s | 2.000000 s across the leap second; 1 s at the June 2016 control |
| ECLIPJ2000 → J2000 | SPICE `pxform` | 1e-15 | 0 (identical) |
| Earth–Sun state | JPL Horizons DE441, 6 epochs | 1 m, 1e-6 m/s | 5.4e-2 m, 1.5e-7 m/s |
| Earth–Sun state | ERFA epv00, 41 epochs 1900–2100 | 11.2 km, 5.0 mm/s | 7.1 km, 3.1 mm/s |
| Annual perihelion / aphelion distance of the Earth, 2000–2039 | EMB mean elements a(1∓e), Standish | 1.2e-4 au | q −4.8e-5…+6.4e-5 au; Q −7.1e-5…+4.0e-5 au |
| EMB perihelion interval (mean, 39 years) | anomalistic year 365.259636 d | 0.01 d | 365.2631 d (Δ = 3.4e-3 d, ~1σ of the perihelion-time jitter) |
| GM☉ (kernel) vs `constants.GM_SUN_M3_S2` | same DE440 number | 2 ulp | 2 ulp (SPICE parser rounding) |
| GM⊕ (kernel) | IERS 2010 GM⊕(TT)·(1 − L_B) | 3e-9 relative | 2.8e-10 |
| EMB identity r_B = (μ_E r_E + μ_M r_M)/(μ_E + μ_M) | definition | 1e-3 m | 3e-5 m |
| de440s coverage | NAIF summary 1849-12-26 to 2150-01-22 TDB | exact | exact |
| Hermite cache (3600 s) vs direct SPICE, 11 bodies, 2030 | SPICE | 1 m | 0.060 m (Mercury barycentre) |
| Hermite cache at the schema maximum 7200 s; per-run midpoint diagnostic bounds random-epoch errors | SPICE | 1 m; random ≤ 1.01·midpoint + 1 mm | 0.97 m (Mercury barycentre); pass |
| SRP magnitude at 1 au, direction, 1/r², linearity in C_r, A/m, S | closed form | 1e-14 relative | pass |
| L1 Sun-only vs L0 Kepler, e = 0.3, moving Sun, 1 yr | M1 L0 | 1e-2 m | 6.7e-4 m |
| L1 Sun-only + SRP vs L0 with μ − k, 1 yr | closed form | 3e-2 m | 5.8e-3 m |
| L1 rtol convergence (no step cap) | error ∝ rtol | ≥ 5× per decade | 130 → 13 → 1.5 → 0.19 m |
| L1 heliocentric acceleration = barycentric − Sun's acceleration under the same bodies | algebraic identity | 1e-12 relative | pass |
| L1 Sun-only vs L0 Kepler with the de440s Sun (accelerated), MVP orbit, 1 yr, without / with SRP (μ − k) | M1 L0 | 1e-2 m | 7.7e-4 m / 1.5e-3 m (M2 review fix; previously 2.2e8 m) |
| L1 full model step/tolerance convergence, 1 yr | self (1-day cap) | 1e-2 m | each < 1e-2 m (12 h cap, rtol 1e-13, cache 1800 s); `assess_l1_run`: 8.0e-4 m (halved cap), 2.4e-3 m (rtol/10) |
| L1 full MVP at the configured 300 s cap vs 1-day cap | self | 0.1 m | 4.6e-2 m (heliocentric form; 2.5e-2 m with the original barycentric form) |
| SRP effect on the MVP orbit, 1 yr (`assess_l1_run`) | 2βnta (circular-orbit estimate) | 1 % | 3.7419e7 m vs 3.7392e7 m (0.07 %) |
| Body subsets, MVP orbit, 1 yr: dropping Neptune / Jupiter; full model vs L0 | bounds (physical tidal effect) | [1e3, 1e4] m / [2e6, 2e7] m / [1e7, 1e8] m | 2.9e3 m / 7.4e6 m / 3.25e7 m |
| L1 barycentric comparison path vs heliocentric, full set, 1 yr | self | [1e2, 2e3] m | 566 m (the Sun's DE440 acceleration not explained by the nine bodies) |
| REBOUND IAS15 two-body vs L1, 1 yr | REBOUND | 1e-2 m | 7.7e-4 m |
| REBOUND IAS15 full N-body vs L1, 1 yr | REBOUND | 10 m | 1.03 m (REBOUND's planets drift from DE440: Earth 61 km, Mercury 581 km; the satellite is only tidally sensitive). Barycentric path vs REBOUND: 565 m |
| REBOUND vs L1, both without Neptune, 1 yr | REBOUND | 10 m | 1.4 m (dropping Neptune changes L1 by 2.9e3 m) |
| 1PN along-track drift estimate 2β₁PN n t a (documented omission) | direct integration with/without the 1PN term | — | 55 667 m both ways (M2 review; the earlier "~10 km/yr" was an unmeasured underestimate) |

## M3: ground sites, Earth rotation, light time, occultation, access: independent checks

These numbers come from `tests/unit/test_ground.py`, `tests/unit/test_visibility.py` and
`tests/integration/test_access_mvp.py`; every SPICE test ran (none skipped). Details are in
`docs/milestones/M3.md`.

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| WGS 84 geodetic → Earth-fixed, 8 sites (poles, h = −430 … 4205 m) | SPICE `georec`; astropy/ERFA `gd2gc` | 1 mm | ~1e-9 m; `recgeo` round trip 1e-12 rad, 1e-6 m |
| Equator/prime meridian, poles, 90° E | (a, 0, 0), (0, 0, ±b), (0, a, 0) | exact / 1e-9 m | pass |
| Up vector = ellipsoid normal | gradient of the ellipsoid equation; SPICE `surfnm` | 1e-14 | pass |
| WGS 84 derived b, e² | NGA.STND.0036 table 3.1 | 1e-4 m; 1e-11 rel | pass |
| IAU_EARTH rotation matrix and rate (4 epochs) | closed form from the `pck00011` constants (`gdpool`) | 3e-11; 2e-11 s⁻¹ | pass |
| Site velocity | central difference; ω ρ | 1e-4 m/s; 1e-7 rel | pass |
| IAU_EARTH vs astropy ITRS → GCRS (IERS-B, offline), equatorial site | astropy 8.0.2 / astropy-iers-data 0.2026.11.9 | site < 13 km, pole < 1e-4 rad, angle < 2e-3 rad | 6.14 / 7.89 / 8.65 km; 4.4e-5 / 3.7e-5 / 3.9e-5 rad; −0.96 / −1.24 / −1.36e-3 rad (2005 / 2020 / 2023). **Documented approximation**, see model_assumptions.md M3 |
| IAU_EARTH rotation-angle offset = IAU angle(TDB) − ERA(UT1), 2000/2020/2030/2040, UT1 = UTC | ERFA `c2t06a`; ERA closed form (IERS 2010 eq. 5.15) | 2e-7 rad; 2030/2040 values 3e-5 rad; drift 1e-6 rad | agreement ≤ 2.4e-8 rad; −8.0e-4 (J2000), −1.64e-3 (2030, 10.5 km), −2.04e-3 rad (2040, 13.0 km); drift −4.04e-5 rad/yr at constant TT − UT1 |
| Published station KG0640 (NGS datasheet, NAD 83(2011)/GRS 80) geodetic → XYZ | NGS published X, Y, Z | 1.5 mm per component (published rounding) | ≤ 0.46 mm (GRS 80 and WGS 84) |
| Elevation/azimuth of placed satellites (90°, 0°, 30°) | closed form | 1e-10 deg | pass |
| Light time, geometric and aberration-corrected transmit direction (Mars barycentre as transmitter, site 35° N 110° W 1500 m, 240 epochs, 2030) | SPICE `spkcpt` `XCN` / `XCN+S` | 1e-9 s; 1e-12 rad; 1e-8 rad | 4.5e-13 s; 2.0e-16 rad; 1.4e-9 rad (aberration 8.1–8.4e-5 rad) |
| Sun separation with moving receiver (3e4 m/s) and moving Sun (2e4 m/s) | closed-form retarded epochs + first-order aberration of both directions | 4β² = 4e-8 rad | 2.5e-9 rad |
| Sites below the ellipsoid (h = −300, −1000 m), rays at ±1e-5, 1e-4 rad | geodetic horizon | flag flips at the horizon | pass |
| Multi-site `compute_access_from_config` (2 sites) | `compute_access` per site; foreign kernel set | identical arrays; `KernelSetMismatchError` | pass |
| Instantaneous vs retarded aim, site moving ⊥ | asin(v/c) | 1e-9 rel | pass |
| Transmitter aberration | Lorentz round trip; exact (−β, 1/γ); comoving pair cancels | 1e-15; 3β² | pass |
| Pointing rate, dτ/dt, range rate | central differences; identity c τ′(1 − u·v/c) | 1e-6 rel; 5e-5 m/s; 1e-12 | pass |
| Moving occulter (Moon at 100 km/s) | brute-force photon/Moon sampling | 1e-3 rel | pass (static-Moon value off by > 1e7 m) |
| Circle overlap / occulted fraction | quadrature; r²(2π/3 − √3/2); (r_B/r☉)² | 1e-9; 1e-14 | pass; monotone in separation |
| Penumbra with a moving Earth (light-time-retarded occulter) | quadrature with the retarded position | 1e-9 | pass (instantaneous position differs by > 1e-3) |
| Rotating equatorial site, far satellite | (π/2 − e)/π access, ½ Earth-blocked | 3/n | pass |
| MVP year, hourly, SPICE: light time | — | ≤ 4 iterations, residual < 1 cm | 3 iterations everywhere, residual ≤ 3 mm, 0 fallbacks |
| MVP year: geometry | chord of two 1 au orbits 70 ± 3° apart; mean motion | 1.10–1.20 au; 10 % | 1.122–1.180 au; pointing rate 1.99e-7 rad/s |
| MVP year: access fraction | analytic mean acos(sin 10°/cos δ)/π = 0.4420 | 1/24 (hourly aliasing); also ≤ 2·gap/(2π) with the hour-angle gap diagnostic | 0.4220; hour-angle gap 8.6° (30 days at 5 min: 0.07°) |
| 30 days at 5 min: access fraction | same analytic expression | 0.005 | pass |
| MVP year: Earth blockage ⇔ apparent elevation < 0 (equator) | geodetic horizon tangent to the ellipsoid | except within 3e-6 rad | pass |
| MVP year: illumination | no eclipse geometry | exact | 1.0 everywhere |
| L0 reference (orbit drift) | Sun-only L1 (M2) | 1e-2 m | pass |
| Sun/Moon radii from `pck00011` | `constants` | 1e-15 rel | pass |

Mutation check: 21 single-line source mutations of `ground.py`/`visibility.py` (aberration and
rate signs, geocentric up, spherical Earth, transposed rotation, missing ω × r, no limb margin,
no retardation, swapped azimuth, static Moon, Earth at the wrong epoch, unaberrated receiver
direction, …): all 21 caught. M3 review: 10 further mutations (Sun separation without the
receiver aberration of the Sun or without its retardation; `visible` without the Earth, Sun or
Moon flag; occultation fraction without the Moon or without Sun and Moon; median light time;
`multiple_occulters` with ≥ 1; no below-ellipsoid surface scaling): all 10 caught by the new
tests.

## M4: solar collection, PV conversion, thermal nodes: independent checks

These numbers come from `tests/unit/test_solar.py`, `tests/unit/test_thermal.py`,
`tests/unit/test_power_chain.py` and `tests/integration/test_power_thermal_mvp.py` (SPICE test
ran, not skipped). Details are in `docs/milestones/M4.md`.

| Check | Reference | Tolerance | Measured |
|---|---|---|---|
| S at 1 au, normal incidence | configured `S_1AU` (1361, 1360.8, 1500 W m⁻²) | exact | exact |
| 1/r² flux and P_PV at fixed T (0.3–1.2 au) | `S_1AU (au/r)²`, au typed in | 4e-15; 1e-14 | pass |
| Zero incidence, back-face light, umbra (M3 geometry), penumbra | 0 W; linear in `f cos(inc)` | exact; 1e-15 | pass |
| Fixed-normal incidence through the M3 illumination result | `cos 60° = 0.5`, `S = 4 S_1AU` at 0.5 au | 1e-15 | pass |
| η_PV clip: linear values, clipped values, `linear_range_K` | hand values | 1e-14 | pass; 200 random laws × 2000 T never outside `[η_min, η_max] ⊂ [0, 1]` |
| PV node, η ≡ 0 (3 α/ε/F sets, 46 distances) | `(α S/((ε_f+ε_b) σ F) + T_space⁴)^¼` | 1e-13 rel | pass; 330.98 K at 1 au (α 0.9, ε 1.8) |
| PV node, coupled linear law (γ = −4e-4, −1.5e-3, +2e-3) | independent quartic roots (`numpy.roots`) | 1e-12 rel | pass |
| PV balance residual (clip active) | recomputed in the test | 1e-12 P_abs | pass; root inside bracket, unique |
| PV node at tiny P_abs (illumination 1e-11 … 5e-324, penumbra-edge sweep with the M3 overlap function) | 50-digit `Decimal` fixed-point root | 2 ulp of T | pass; converged, feasible, ledger closed (review fix: previously RuntimeError for f ≤ 1e-15) |
| `T⁴ − T_space⁴` near `T_space` | 50-digit `Decimal` | 4ε relative | pass (naive form: O(1) relative error at 1 ulp) |
| PV solver failure reporting | monkeypatched brentq: unconverged midpoint / unconverged true root / wrong root / root 10 ulp off | — | NaN + not converged + `RuntimeError`; 10-ulp root polished to ≤ 1 ulp |
| Area scaling (PDF §13.2) | A and 3.7 A | 1e-14 | pass; T_pv unchanged, P_PV, Q_waste, A_req × 3.7 |
| PV view factor scaling | `T⁴ − T_space⁴ ∝ 1/F` | 1e-13 | pass |
| Hotter closer to the Sun | strict monotonicity of T(r), η(r), S(r) | — | pass |
| Non-unique PV root (η = 1 below 289 K) | uniqueness condition fails | flagged | flagged, residual ≤ 1e-12 P_abs |
| Radiator required area | `Q/(ε σ n F (T_lim⁴ − T_space⁴))`; ratios for ε, T_lim, n, F, Q | 1e-14 | pass; ∝ (450/300)⁴ exactly for T_space → 0 |
| Radiator equilibrium T for a given area; A = A_req → T_lim | closed form | 1e-14 | pass |
| Infeasible radiator (A = A_req/2) | `T = 2^¼ T_lim`, margin < 0 | 1e-8 (T_space⁴ term) | flagged infeasible |
| Waste heat = on-board losses + Q_other | hand computation from P_PV | 1e-14 | pass; `P_aperture + Q_waste = P_PV + Q_other` |
| Reflected light not waste heat | α = 0.5 equilibrium; equal-absorbed collectors with 5× reflected | 1e-13; 1e-12 | pass |
| Ledger can fail | `Q_waste × (1 + 1e-9)` | open | detected (relative residual > 1e-11) |
| Ledger from hand-computed terms (η ≡ 0.3, 0.4 and 1 au) | closed-form T, Q_PV,rad, P_PV, P_aperture, Q_waste | 1e-12 … 1e-14 | pass |
| Binding margin | `margin_K = min(pv_margin_K, radiator_margin_K)`; `feasible = pv_converged & (margin_K ≥ 0)` | exact | pass (3 radiator modes × 46 distances) |
| MVP year (SPICE, hourly, 8767 epochs) | independent `S_1AU (au/|r_sat − r_sun(t_e)|)²` from raw SPICE positions, Sun at its light-time-retarded epoch (separate fixed-point iteration in the test) | 1e-12 | pass, measured 5.6e-16; instantaneous Sun differs by 5.9e-8 (asserted > 1e-8, < 1e-7) and a 69.184 s epoch shift by 6.7e-8, so both would fail; T_pv 302.780–302.786 K, P_PV 3.6705–3.6708 MW, A_req 438.49–438.53 m², ledger 5.5e-16, evaluation 0.15 s |

Mutation check (reproducible: `uv run python scripts/mutation_check_m4.py`, which restores
every file and counts only pytest exit 1 as caught): 18 single-line source mutations of
`solar.py`/`thermal.py`/`power_chain.py`/`visibility.py`. The 11 original ones are 1/r flux, no
back-face clip, reflected light as PV heat, no efficiency clip, PV front face only, radiator
faces ignored, optics loss counted twice, radiator always feasible, PV `T_space` dropped, PV
temperature limit ignored and power conditioning dropped. The 7 added in the M4 review are PV
representation floor dropped, float-neighbour polish dropped, unfactored `T⁴ − T_s⁴`, ledger
floors dropped, `margin_K` = radiator margin only, brentq non-convergence ignored and
instantaneous Sun for illumination. All 18 were caught on the final source.

## Unresolved discrepancies

None recorded. (M2 review: the Sun-only / reduced-body-set inconsistency of the original
barycentric L1 formulation was fixed by integrating heliocentrically with consistent indirect
terms, not left as a documented discrepancy; see docs/milestones/M2.md "Review outcomes".)

## M5: beam physics: independent checks

References are closed forms written in the tests, independent quadrature (`scipy.integrate`),
Monte Carlo with explicit `numpy.random.Generator`s, HCIPy 0.7.1, or the ITU-R validation
examples; never the module's own helpers.

| Check | Reference | Tolerance | Measured / result |
|---|---|---|---|
| Aperture power = 1 W (5 illuminations) | `scipy.integrate.quad` | 1e-12 | pass |
| Taper efficiency, spill-over | quad of ∫A dA; untruncated Gaussian power outside a | 1e-12 / 1e-10 | pass |
| Hankel quadrature vs closed far fields (uniform, parabolic 10/30 dB) | Airy, Lommel `2J2(u)/u²` | 1e-12 of peak | pass |
| Truncated-Gaussian far field (a/w0 = 1) | quad | 1e-10 rel | pass |
| Hankel refinement ×4 (far field, N_F = 5, 50) | self | 1e-12 of peak | pass; 2-panel rule fails (discriminating) |
| Fresnel on-axis irradiance, N_F = 0.3–21.7 incl. on-axis zero (N_F = 8) and 4× maximum (N_F = 4) | `(4/(πa²))sin²(πN_F/8)` (Goodman) | 1e-11 rel | pass |
| Fresnel → Airy as N_F → 0 | `(πN_F)²/192` scaling | within [0.5, 1.5] × prediction | pass for N_F = 1e-2, 1e-3, 1e-4 |
| Fresnel power conservation, N_F = 20 | Parseval (= 1) | 1e-11 | pass |
| Airy first null for 5 (D, λ, R) | 1.21967 λR/D; numerical minimum | 1e-14; 1e-6 | pass |
| Airy power within x = 1e6 | 1 (tail 6.4e-7) | 1e-6 | pass (= EE(1e6) to 1e-9) |
| Far-field total power, uniform / truncated Gaussian a/w0 = 1 / parabolic 10 dB (review M5) | EE(x = 1e3) + analytic edge tail `2a²|A(a)|²/x` = 1 | 1e-6 | pass (residual ≤ 1.2e-7; tail 1e-4–6e-4) |
| Uniform Fresnel total power, N_F = 20 (review M5) | EE(300 m) + `2/(πx)` = 1 | 1e-6 | pass (residual 4e-9) |
| Tapered far-field peak irradiance (truncated Gaussian, parabolic 10/30 dB; review M5) | `η_taper πD²/(4λ²R²)`, η_taper by quad; pattern maximum on a grid | 1e-10 | pass; also through `evaluate_beam` (microwave, W m⁻²) |
| EE at first dark ring | 1 − J0(j₁,₁)² = 0.83778 | 1e-14 | pass (numerical quadrature 1e-12) |
| Gaussian beam w(z), EE(w) = 1 − e⁻² | Siegman | 1e-15 / 1e-11 | pass |
| Gaussian beam vs Fresnel–Hankel truncated Gaussian (a/w0 = 3) | comparison path | 2e^{−9} EE | pass (N_F = 20, 1, 1e-3); a/w0 = 1 differs > 1e-2 |
| Offset capture, Gaussian beam | `ncx2.cdf(4a²/w², 2, 4d²/w²)` | 1e-11 | pass |
| Offset capture, Airy, 2-D grid (n = 25, 100, 400) | Cartesian midpoint | 2e-5 at n = 400, decreasing | pass |
| Expected capture with bias + jitter, Gaussian beam (5 cases) | `ncx2.cdf(a²/v', 2, ν²/v')` | 1e-9 | pass; percentiles exact to 1e-10 |
| Bias wiring through `evaluate_beam` (Gaussian beam, b = 50 nrad, σ = 0 and 30 nrad; review M5) | `ncx2.cdf(a²/v, 2, (Rb)²/v)`; with jitter `v' = v + (Rσ)²` | 1e-11; 1e-9 | pass; bias dropped / doubled / swapped with σ each fail |
| Expected capture, Airy | Monte Carlo (40 000 samples) | 5 s.e. | pass |
| Capture percentiles, non-monotone η(d) (4 Airy cases; review M5) | brentq level sets with exact Rice CDF differences | 1e-8 abs | pass (≤ 1.2e-9) |
| Rice law for ν/s ≥ 1e4 (review M5) | scipy Rice at b = 1e4, 1e5; linearised spread at ν/s = 1e6, 1e17 | 1e-8 s / 2e-9; 1e-3 | pass (scipy returns NaN at 1e6) |
| Rice parametrisation | explicit density; MC quantiles | 1e-12; 5e-3 | pass |
| E[η] vs σ, centred | monotone (Anderson) | strict | pass (Gaussian beam, 3 sizes; Airy a = 0.07ℓ) |
| Jitter with receiver on first ring (a = 1.5ℓ) | Monte Carlo | — | E[η] > η(0) (real effect, documented) |
| Smaller λ / larger D, fixed receiver | EE(2x)/EE(x) | 1e-10 | pass |
| Impossible beam (1 m, 2.45 GHz, 1 au, 1 km) | x²/4 = 7.4e-15 | 1e-9 rel | pass (< 1e-13 delivered fraction) |
| Equal-area circle vs exact ellipse | κ(a−b)²/4 | 10 % of prediction | pass; table in `model_assumptions.md` |
| Lattice AF peak N, nulls λ/(nd), grating lobes mλ/d | explicit sum / closed form | 1e-9 | pass |
| Lattice main-lobe fraction (d = 1.6λ, q = 1, 2; d = 0.7λ) | hemisphere integral of cos^q\|AF\|², 24 × 24 | 3 % | pass (1/9, 0.167, 1) |
| HCIPy far-field profile (uniform, truncated Gaussian, parabolic), λ = 1 µm, D = 10 m | HCIPy 0.7.1 Fraunhofer (f = 1, output in rad) | 2e-3 rel | pass; absolute J(0) = πa²/λ² = 7.85e13 sr⁻¹ to 2e-3 |
| HCIPy Airy encircled energy at 6 radii | HCIPy | 3e-3 abs | pass |
| ITU-R P.618-13 gas/cloud/rain (London, 14.25 GHz p = 1 %; 29 GHz p = 0.01 %), **with P.676-11, P.1511-1, τ = 0 pinned** | ITU-R SG3 validation examples via `itur` test file (second-hand) | 1e-5 dB | pass (validates the wrapper and revision wiring, not the production P.676-12 gas term) |
| ITU-R production revisions (P.676-12, P.1511-2, τ = 0/45/90°) at the same site | **regression pin** of this wrapper with `itur` 0.4.0 (not a validation) | 1e-9 dB | pinned; gas 0.22687 dB vs 0.22369 dB with P.676-11 (+1.4 %); rain H > circular > V (P.838 k_H > k_V) |
| MVP year (SPICE, 3700 visible epochs) | closed-form EE(x_eff); 2s²κ pointing loss | 1e-12; 1 % | pass; capture 0.15–0.83 % (negative result) |

## M6b: completeness audit: added checks and reproduction

| Check | Reference | Tolerance | Result |
|---|---|---|---|
| §8.4 curves (`helioswarm characterize`) vs closed forms: S(r), P_abs, η_PV(T) law, P_PV, P_dc_to_tx, Q_waste, A_rad; Q_waste difference between operating points | closed forms written in the test; σ from the exact 2019 SI constants | 1e-15–1e-13; A_rad 1e-9 | **pass** (`test_characterize.py::test_power_thermal_curves_against_closed_forms`) |
| §9.5 capture vs pointing sigma and receiver radius: perfect pointing = 1 − J0(x)² − J1(x)², orderings | Born & Wolf encircled energy (scipy `j0`, `j1`) | 1e-9; strict | **pass** (`test_capture_curves_centred_airy_and_monotonicity`) |
| Ruze factor applied once to the expected capture | exp(−(2πσ_w/λ)²) | 1e-12 | **pass** |
| §7.3 diagnostics from the saved time series vs the in-memory `AccessSummary` of the same run | independent code path (`visibility.summarize_access`) | extremes exact; fractions ≤ 1/N (weighting) | **pass** (`test_end_to_end.py::test_summary_is_recomputable_from_saved_timeseries`) |
| Microwave sweep consistency: 1.0 au / 1 km point vs the microwave MVP run | M6 job 60127172 (0.3145 W) | printed digits | **pass** (array 60141210: 0.3145 W) |
| Fresh-clone reproduction of the optical and microwave MVP runs | M6 jobs 60127166 / 60127172 | printed digits | **pass** (job 60144119; `docs/compliance_matrix.md`) |

<!-- BEGIN validate-models -->

## Programmatic validation (`helioswarm validate-models`, generated)

Generated by `helioswarm validate-models`; do not edit by hand. run: 2026-11-12T23:19:13.935403Z UTC; helioswarm: 0.1.0.dev0, git 1cc897e0186e74a16f72d9d7aaa67ebdc0912f81 (dirty True); host: holy8a26601.rc.fas.harvard.edu; Slurm job: 60127176; base config: configs/mvp_single_satellite.yaml.

**37 passed, 0 failed, 0 not run** of 37 checks.

| PDF | Check | Value | Reference | Tolerance | Result |
|---|---|---|---|---|---|
| §7.4 | Circular two-body orbit (L0) preserves radius; Kepler III period | max\|r/a-1\| = 4.44e-16; \|r(10T)-r(0)\|/a = 2.44e-15 | r = a exactly; return after 10 Kepler periods 2 pi sqrt(a^3/mu) (Kepler III) | 1e-12 / 1e-11 (float64 round-off of the closed form) | **pass** |
| §7.4 | Circular orbit under L1 (Sun only) within integrator tolerance | max\|r/a-1\| = 4.00e-15; energy drift 3.76e-15 | circular two-body orbit (L1, Sun only, Sun at rest), 3 years | 1e-12 (= integrator rtol) | **pass** |
| §7.4 | Earth-Sun state vs JPL Horizons | max position error 5.336e-02 m over 6 epochs 2000-2050 | JPL Horizons DE441 Earth wrt Sun (retrieved 2026-11-12) | 1 m (DE440 vs DE441 differ by < 6 cm; a unit/origin/time-scale error fails by >1e3 m) | **pass** |
| §7.4 | Receiver site: one sidereal rotation per IAU_EARTH period | site direction change after one IAU_EARTH period = 3.46e-07 rad | 0 (one rotation per 86400*360/360.9856235 s; pck00011 BODY399_PM) | 1e-6 rad (float64 rounding of W ~ 7e4 rad) | **pass** |
| §7.4 | Beam segment through / past the Earth ellipsoid flagged correctly | through-Earth True, tangent +1e-4 a False, -1e-4 a True | True, False, True (geometry of a chord vs the WGS84 ellipsoid) | exact (boolean) | **pass** |
| §7.4 | Light time: static d/c and uniformly moving site closed form | static: max\|tau c/d - 1\| = 0.0e+00; moving site: max\|tau - tau_exact\| = 1.5e-11 s (instantaneous-geometry error 1.3e+01 s) | tau = d/c (static); positive root of (c^2 - v^2) tau^2 - 2 (D.v) tau - D^2 = 0 (uniformly moving site, PDF §7.1/§7.4) | 1e-14 relative (position rounding); tolerance_s + 2e-11 s (t_a representation at 1e9 s); the instantaneous distance must be wrong by > 1e-3 s (discrimination) | **pass** |
| §8.5 | S(1 au) = S_1AU and 1/r^2 scaling | S(1 au) = 1361.0 W/m^2; max\|S r^2/(S1 au^2) - 1\| = 2.2e-16 | S_1AU = 1361.0 W/m^2 (IAU 2015 B3), 1/r^2 | exact at 1 au; 1e-14 (rounding) | **pass** |
| §8.5 | Zero incidence cosine or umbra gives zero power | max \|power\| at cos(inc) = 0 or umbra = 0.0 W (lit: P_PV(1 au) = 3.6747e+06 W) | 0 W: incident = S A cos(inc) f (PDF §8.1) | exact | **pass** |
| §8.5 | Energy closure and zero power in eclipse (0.3-1.2 au) | ledger max relative residual 3.94e-16; P_PV in umbra 0.0 | energy closure incident = reflected + PV heat radiated + radiator + P_aperture; 0 W in eclipse | 1e-12 (closure tolerance); exact 0 | **pass** |
| §8.5 | Radiator area proportional to 1/(eps (T^4 - Ts^4)) | A(1 MW) = 238.927 m^2; A(eps/2)/A = 2; A(T)/A(2T) = 16.00000002 | Q/(eps sigma n F (T^4-Ts^4)) = 238.927 m^2 (sigma = 2 pi^5 k^4/(15 h^3 c^2), exact SI constants); 2; 16.00000002 | 1e-10 on the area (the package sigma is CODATA's 10-digit rounding of the exact value, 3e-11 apart); 1e-12 on the ratios | **pass** |
| §9 | Airy first null at 1.21967 lambda R/D | max I(1.21967 lambda R/D)/I0 = 2.4e-33 (3 cases) | 0: first zero of J1 at x = 3.8317059702 (Abramowitz & Stegun Table 9.5) | 1e-25 relative (float64 rounding of x) | **pass** |
| §9 | Encircled energy at first dark ring; total power normalisation | EE(first dark ring) = 0.837784869173; quad power to x=2000 + tail = 0.99999994 | 1 - J0(j11)^2 = 0.837784869173 (Rayleigh); total 1 (Parseval) | 1e-12; 1e-5 (quad tolerance and O(x^-2) tail) | **pass** |
| §9 | Offset capture vs independent nested quadrature (2-D grid as supporting evidence) | 1-D eta = 0.372038179092; nested-quad eta = 0.372038179092 (rel diff 0.0e+00); 2-D grid - 1-D at n=25/100/200/400: +1.1e-04/-1.4e-06/+3.0e-06/+1.4e-06 | independent nested scipy quad over the offset disk (receiver-centred polar coordinates); Cartesian anti-aliased midpoint grid as supporting evidence | 1e-9 relative vs nested quad (quadrature tolerances 1e-12/1e-13); grid \|error\| < 2e-5 for n >= 100 and below the n = 25 error (empirical ceiling, edge error O(cell), not monotone in n) | **pass** |
| §9 | Expected capture non-increasing with pointing sigma (centred) | E[eta] = 1.201742e-02, 1.187126e-02, 1.145019e-02, 9.994717e-03, 6.480543e-03 for sigma/(lambda R/D) = 0..0.4 | non-increasing in sigma for the centred baseline (Anderson 1955, unimodal beam) | strict ordering | **pass** |
| §9 | Smaller wavelength / larger aperture improve capture | EE: base 0.00850136, lambda/2 0.03357324, 2D 0.03357324 | both = EE(2x) = 0.03357324 > EE(x) (x = pi D a/(lambda R)) | 1e-12; strict improvement | **pass** |
| §9 | Impossible beam gives ~x^2/4 capture | capture = 7.363422e-15 | x^2/4 = 7.363422e-15 (small-x limit of 1 - J0^2 - J1^2), 1 m aperture at 2.45 GHz, 1 au, 1 km receiver | 1e-9 relative | **pass** |
| §9 | Fresnel-Hankel -> Airy as N_F -> 0 | N_F=0.01: dev 5.140e-06; N_F=0.001: dev 5.140e-08 | (pi N_F)^2/192 = 5.140e-06; (pi N_F)^2/192 = 5.140e-08 (on-axis sinc expansion of sin^2(pi N_F/8)) | [0.5, 1.5] x prediction | **pass** |
| §9/§14.3 | Batched capture kernel (numpy, JAX) vs M5 reference | batch(numpy) vs M5 per-point max diff 9.3e-13; jax vs numpy batch max diff 5.7e-16 | helioswarm.capture.capture_statistics (adaptive M5 reference) | 1e-9 (quadrature); jax vs numpy 1e-12 (same algorithm, trapezoid Bessel) | **pass** |
| §13.1 | Link budget: passive stages never increase power; eta in [0,1]; clip | 0 invariant violations over 10000 random epochs; 4816 clipped | P_PV >= P_PV,tx >= P_dc_to_tx >= P_beam >= P_aperture >= P_atm >= P_captured >= P_accepted >= P_receiver >= P_grid; every eta in [0, 1]; P_accepted = max_input when clipped | exact (monotone float multiplication) | **pass** |
| §13.1 | SRP: magnitude, anti-sunward direction, inverse square | \|a(1 au)\| rel err 2.2e-16; direction err 1.1e-16; max\|a(r) r^2/a(1 au) - 1\| = 1.1e-16 (0.3-5.2 au, Sun offset from the origin) | a = (S_1AU/c) Cr (A/m) (au/r)^2 along (r - r_sun)/\|r - r_sun\| (anti-sunward; PDF §6.3, §13.1) | 1e-14 relative / 1e-15 (float64 rounding) | **pass** |
| §13.1 | Time scales: UTC -> TDB at known epochs | max \|TDB_spice - TDB_ref\| = 1.2e-07 s at 2000-01-01T12:00:00, 2017-01-01T00:00:00, 2030-01-01T00:00:00 | TAI-UTC from IERS Bulletin C (32 s from 1999, 37 s from 2017; none announced since) + 32.184 s + USNO Circular 179 TDB-TT; MVP epoch 2030-01-01 UTC | 1e-5 s (two-term TDB-TT series; a wrong leap-second count fails by >= 1 s, a dropped TDB-TT term by up to 1.7 ms) | **pass** |
| §10.5 | Capacity-factor reference link: conditions and rated power | conditions ok: True; E[eta] 0.00843074044083 vs 0.00843074044083 (rel 1.8e-14); rated 6003.52 W (chain rel 1.8e-14) | window-mean range and heliocentric distance, zenith; independent noncentral-chi2 quadrature of the jittered Airy capture; PDF §10.1 chain (2-day MVP window) | 1e-8 relative (quadratures converge to < 1e-10) | **pass** |
| §13.2 | 1-AU baseline: config -> saved outputs without manual steps | outputs ['config_resolved.yaml', 'figures', 'logs', 'manifest.json', 'report.md', 'summary.json', 'timeseries.parquet']; 6 figures; mean P_grid 1829 W | config -> resolved config, manifest, timeseries, summary, figures, report, log | all present | **pass** |
| §13.2 | Occulted / unavailable link delivers zero power | P_grid at 29 unavailable epochs (incl. 25 Earth-occulted): max 0.0 W; cloudy optical outage max 0.0 W | 0 W | exact | **pass** |
| §13.2 | Receiver capacity limit clips correctly | 16 clipped epochs; \|P_grid - limit*eta_rx*eta_grid\| max 0.0e+00 W | limit 7615.88 W x 0.4 x 0.97 at clipped epochs; unchanged elsewhere | 1e-9 W absolute | **pass** |
| §13.2 | Doubling collector area doubles P_PV (and P_grid unclipped) | max\|P_PV(2A)/P_PV(A) - 2\| = 0.0e+00; max\|P_grid ratio - 2\| = 0.0e+00 | 2 (PV temperature independent of area: absorbed and radiated both scale with A) | 1e-12 relative | **pass** |
| §13.2 | Smaller lambda / larger D improve capture end to end | median capture: base 6.6094e-03, lambda/2 2.6137e-02, 2D 2.6137e-02 | smaller wavelength and larger aperture improve capture at every available epoch | strict | **pass** |
| §13.2 | Larger pointing sigma cannot improve expected capture | median expected capture sigma x1 6.609377e-03, x10 6.299956e-03 | larger pointing sigma cannot improve expected capture (centred baseline) | <= at every available epoch | **pass** |
| §13.2 | Impossible microwave beam: near-zero capture, not high power | 20 available epochs: max\|eta_capture/(x^2/4) - 1\| = 3.4e-10; max P_grid/P_aperture = 2.161e-15 | x^2/4, x = pi D a sqrt(sin el)/(lambda R) (small-x limit of 1 - J0^2 - J1^2, uniform 1 m aperture, 2.45 GHz, 1 km horizontal receiver, ~1.15 au); P_grid/P_aperture <= x^2/4 *0.4* 0.97 ~ 1e-14 | 1e-6 relative (O(x^2) ~ 1e-15 and jitter (sigma D/lambda)^2 ~ 7e-11 corrections; quadrature 1e-9); P_grid/P_aperture < 1e-12 | **pass** |
| §13.2 | Energy accounting: P_grid <= every upstream stage | 0 violations; max P_grid/P_incident 4.342e-04 | P_grid <= every upstream stage <= incident sunlight | exact | **pass** |
| §13.2 | Same config + seed -> identical summary and timeseries content | summary.json identical: True; content hash 4330a1d750856c84 vs 4330a1d750856c84 | identical outputs for the same config and seed (timestamps only in manifest.json) | bitwise | **pass** |
| §13.3 | REBOUND orbit cross-check (two-body and full force model) | two-body: max heliocentric diff 1.5e-03 m; Sun + 9 bodies: 1.03 m (1 year, 37 epochs) | REBOUND 5.2.2 IAS15 from the same initial states and GMs (PDF §13.3) | 1e-2 m two-body (both integrators << 1 m); 10 m N-body (REBOUND evolves the planets self-consistently, L1 prescribes DE440; measured ~1 m, module docstring of tests/integration/test_rebound_crosscheck.py) | **pass** |
| §13.3 | HCIPy far-field cross-check | max \|J_hcipy - J_ours\| / J(0) = 1.3e-03; J(0) = 7.863887e+13 sr^-1 | HCIPy 0.7.1 Fraunhofer propagator, uniform 10 m pupil, 1 um; pi a^2/lambda^2 = 7.853982e+13 | 4e-3 of the peak (512-px pupil pixelisation) | **pass** |
| §13.3 | ITU-R P.618 validation example (itur) | max \|A - A_ref\| (gas, cloud, rain; 2 cases) = 1.5e-07 dB | ITU-R SG3 validation examples of P.618-13 (London; via itur 0.4.0 test file), P.676-11/P.1511-1 pinned | 1e-5 dB (validation-file precision); production P.676-12 is regression-pinned only | **pass** |
| §13.3 | Capture quadrature refinement convergence | \|E[eta](refinement 2) - E[eta](1)\| = 1.1e-16 | quadrature converged | 1e-12 | **pass** |
| §13.3 | L1 integrator step / tolerance convergence (30 d MVP) | 30 d MVP orbit: max step 300 s vs 1 d: 5.15e-04 m; rtol 1e-12 vs 1e-13: 8.35e-04 m | converged trajectory (differences far below the 1e3-1e4 m beam scale) | 1 m | **pass** |
| §13.3 | Time-step (diurnal sampling) convergence of the access fraction | access fraction 3600 s 0.4161 (gap 13.17 deg), 300 s 0.4440 (0.07 deg), 3593 s 0.4480 (0.65 deg) | analytic diurnal fraction acos((sin el_min - sin lat sin dec)/(cos lat cos dec))/pi = 0.4442 (rotation and declination from the 300 s run; no refraction) | \|300 s - analytic\| < 0.005 and \|3593 s - analytic\| < 0.01; \|3600 s - analytic\| <= gap/2pi (aliasing quantified) | **pass** |

### Unresolved (generated)

None: every check passed.

Notes:

- Time-step (diurnal sampling) convergence of the access fraction: hourly steps alias with the ~24 h synodic day (M3); reported, not hidden

<!-- END validate-models -->
