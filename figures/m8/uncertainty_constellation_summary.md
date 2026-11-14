# Uncertainty study `m8_uncertainty_constellation`

Nested Monte Carlo of the constellation demo (N = 10, three sites, scheduler) with ERA5 weather years as the aleatory input: network P_grid, availability, mass proxy.

Fidelity `screening_era5`: L0 Kepler orbits (the demo is at 1 au, planets/SRP omitted), 3896 s grid, batched Airy capture (numpy_batch), ERA5 cloud outages at every site (threshold 0.5; the weather year is the aleatory input), M7 scheduler.

## Inputs

| name | category | key | distribution | provenance | source |
|---|---|---|---|---|---|
| pv_efficiency | epistemic | `collector.pv_efficiency_reference` | type=triangular, low=0.1642, high=0.3195, mode=0.3 | derived | as configs/uncertainty.yaml (AZUR SPACE DB 00010892-01; JPL Pub. 78-15 Vol. VIII; mode PDF §5) |
| pv_temperature_coefficient | epistemic | `collector.temperature_coefficient_per_K` | type=triangular, low=-0.0044, high=-0.0004, mode=-0.00231 | derived | as configs/uncertainty.yaml |
| electrical_to_beam_efficiency | epistemic | `transmitter.electrical_to_beam_efficiency` | type=triangular, low=0.35, high=0.6, mode=0.5 | assumed |  |
| receiver_conversion | epistemic | `receiver.conversion_efficiency` | type=triangular, low=0.3, high=0.55, mode=0.4 | assumed |  |
| radiator_specific_mass | epistemic | `thermal.radiator_specific_mass_kg_m2` | type=triangular, low=2.0, high=10.0, mode=5.0 | assumed |  |
| weather_year | aleatory | `weather.historical_year` | type=discrete_uniform, values=(2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023) | sourced | ERA5 hourly total cloud cover 2014-2023 at the site nodes (Hersbach et al. 2020, doi:10.1002/qj.3803; CDS doi:10.24381/1cf1ad76); each year equally likely (assumed) |

## Parts

- `mc`: {'status': 'complete', 'n': 64, 'failed': 0, 'failures': []}

## Nested Monte Carlo (N_e = 16, N_a = 4, seed 20261116)

Epistemic variance fraction: unbiased between-row component `max(Var_i m_i - E_i s_i^2 / N_a, 0)` over itself plus `E_i s_i^2` (ddof = 1); 'upper bound' is the uncorrected `Var_i m_i / Var Y`, which also counts the aleatory noise `sigma_a^2 / N_a` of each row mean. The epistemic interval of E_aleatory (percentiles of the row means `m_i`) carries that noise too: its sd is the 'noise sd' column.

| output | pooled p05 | p50 | p95 | epistemic interval of E_aleatory: p05 | p50 | p95 | noise sd of m_i | epistemic variance fraction | upper bound | non-finite |
|---|---|---|---|---|---|---|---|---|---|---|
| p_grid_mean_W | 2.111e+04 | 2.732e+04 | 4.091e+04 | 2.242e+04 | 2.699e+04 | 4.003e+04 | 329.9 | 0.9884 | 0.9907 | 0 |
| availability | 0.9104 | 0.9381 | 0.9477 | 0.93 | 0.9377 | 0.9423 | 0.004856 | 0 | 0.1694 | 0 |
| availability_above_threshold | 0.8201 | 0.8767 | 0.9163 | 0.8374 | 0.8746 | 0.9127 | 0.006054 | 0.8335 | 0.8679 | 0 |
| mass_proxy_kg | 1.577e+05 | 1.678e+05 | 1.799e+05 | 1.578e+05 | 1.678e+05 | 1.782e+05 | 0 | 1 | 1 | 0 |
| capacity_loss_fraction | 0.02465 | 0.027 | 0.02972 | 0.02623 | 0.02715 | 0.02813 | 0.000835 | 0 | 0.1343 | 0 |
