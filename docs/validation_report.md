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

## Physics validation (to be filled by M1–M6)

| PDF section | Check | Tolerance | Result |
|---|---|---|---|
| §7.4 | Circular two-body orbit preserves radius and energy — L0 closed-form baseline | 1e-12 relative | **pass, baseline only** (M1 unit tests). L0 is closed form and conserves radius and energy by construction, so this checks implementation round-off, not an integrator: max abs(r/a − 1) = 4.4e-16 over 103 periods; energy deviation ≤ 1.1e-15 (e = 0), ≤ 2.5e-13 (e = 0.99) over 10 periods |
| §7.4 | Circular two-body orbit preserves radius and energy *within the selected integrator tolerance* — L1 numerical propagator (planets off) | 1e-12 (= rtol) | **pass** (M2). L1 Sun-only (Sun at rest), circular 1 au, i = 0.3 rad, 3 years, rtol 1e-12, 1-day step cap: max abs(r/a − 1) = 4.7e-15, energy drift 5.6e-15 (`test_l1_propagation::test_circular_orbit_radius_is_preserved_within_integrator_tolerance`) |
| §7.4 | Orbital period matches Kepler's third law | 1e-10 relative | **pass** (M1). The period was measured independently by a DOP853 integration (rtol 1e-12) with a return-direction event for a ∈ {0.4, 1, 5.2}, e ∈ {0, 0.5, 0.9}; max deviation from 2π√(a³/μ) is 2.5e-12. T(1 au) with GM☉(DE440) equals the Gaussian year 2π/k to 1e-10 |
| §7.4 | Earth–Sun distance agrees with SPICE at selected epochs | 1 m (Horizons); 11.2 km (epv00) | **pass** (M2). de440s vs JPL Horizons (DE441) at 6 TDB epochs 2000–2050, including the 2024 perihelion and aphelion: position ≤ 5.4e-2 m, velocity ≤ 1.5e-7 m/s. vs ERFA epv00 at 41 epochs 1900–2100: 7.1 km, 3.1 mm/s (documented epv00 maxima 11.2 km, 5.0 mm/s) |
| §7.4 | Synthetic fixed geometry gives the analytic light time | M3 | not yet run |
| §7.4 | Receiver site makes one sidereal rotation per sidereal day | M3 | not yet run |
| §7.4 | Beam segment intersecting the Earth/Sun is flagged as occulted | M3 | not yet run |
| §8.5 | 1 au, normal incidence recovers S_1AU; 1/r² scaling; zero at eclipse; energy closure; A_rad ∝ 1/(εT⁴) | M4 | not yet run |
| §9 / §13.1 | Airy first-null scaling, total power normalisation, encircled energy, capture convergence, pointing monotonicity | M5 | not yet run |
| §13.2 | End-to-end integration tests | M6 | not yet run |
| §13.3 | Independent checks (REBOUND, SPICE/JPL, HCIPy, ITU-R) | M2–M6 | REBOUND and SPICE/JPL **pass** (M2, see below); HCIPy and ITU-R not yet run |

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

## Unresolved discrepancies

None recorded. (M2 review: the Sun-only / reduced-body-set inconsistency of the original
barycentric L1 formulation was fixed by integrating heliocentrically with consistent indirect
terms, not left as a documented discrepancy; see docs/milestones/M2.md "Review outcomes".)
