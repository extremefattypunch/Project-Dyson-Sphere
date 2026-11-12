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
| §7.4 | Circular two-body orbit preserves radius and energy *within the selected integrator tolerance* — L1 numerical propagator (planets off) | integrator rtol/atol (M2) | **open**: not yet run; this is the PDF acceptance item and needs the M2 L1 propagator |
| §7.4 | Orbital period matches Kepler's third law | 1e-10 relative | **pass** (M1). The period was measured independently by a DOP853 integration (rtol 1e-12) with a return-direction event for a ∈ {0.4, 1, 5.2}, e ∈ {0, 0.5, 0.9}; max deviation from 2π√(a³/μ) is 2.5e-12. T(1 au) with GM☉(DE440) equals the Gaussian year 2π/k to 1e-10 |
| §7.4 | Earth–Sun distance agrees with SPICE at selected epochs | M2 | not yet run |
| §7.4 | Synthetic fixed geometry gives the analytic light time | M3 | not yet run |
| §7.4 | Receiver site makes one sidereal rotation per sidereal day | M3 | not yet run |
| §7.4 | Beam segment intersecting the Earth/Sun is flagged as occulted | M3 | not yet run |
| §8.5 | 1 au, normal incidence recovers S_1AU; 1/r² scaling; zero at eclipse; energy closure; A_rad ∝ 1/(εT⁴) | M4 | not yet run |
| §9 / §13.1 | Airy first-null scaling, total power normalisation, encircled energy, capture convergence, pointing monotonicity | M5 | not yet run |
| §13.2 | End-to-end integration tests | M6 | not yet run |
| §13.3 | Independent checks (REBOUND, SPICE/JPL, HCIPy, ITU-R) | M2–M6 | not yet run |

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
| ECLIPJ2000 → J2000 | ecliptic pole → (0, −sin ε₀, cos ε₀) | 1e-15 | pass (SPICE `pxform` cross-check deferred to M2) |
| GM☉ | Gaussian constant k²au³/d² | 1e-10 relative | 5.0e-12 |
| σ | 2π⁵k_B⁴/(15h³c²) with exact SI constants | 1e-10 relative | 3.3e-11 |

## Unresolved discrepancies

None recorded.
