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
| §7.4 | Circular two-body orbit preserves radius and energy | M1 | not yet run |
| §7.4 | Orbital period matches Kepler's third law | M1 | not yet run |
| §7.4 | Earth–Sun distance agrees with SPICE at selected epochs | M2 | not yet run |
| §7.4 | Synthetic fixed geometry gives the analytic light time | M3 | not yet run |
| §7.4 | Receiver site makes one sidereal rotation per sidereal day | M3 | not yet run |
| §7.4 | Beam segment intersecting the Earth/Sun is flagged as occulted | M3 | not yet run |
| §8.5 | 1 au, normal incidence recovers S_1AU; 1/r² scaling; zero at eclipse; energy closure; A_rad ∝ 1/(εT⁴) | M4 | not yet run |
| §9 / §13.1 | Airy first-null scaling, total power normalisation, encircled energy, capture convergence, pointing monotonicity | M5 | not yet run |
| §13.2 | End-to-end integration tests | M6 | not yet run |
| §13.3 | Independent checks (REBOUND, SPICE/JPL, HCIPy, ITU-R) | M2–M6 | not yet run |

## Unresolved discrepancies

None recorded.
