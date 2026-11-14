# Uncertainty study `m8_uncertainty_geo_optical`

Nested Monte Carlo of the GEO optical baseline with ERA5 weather (3896 s, Fresnel-Hankel Chebyshev beam table, ERA5 cloud outages), plus the explicit scenarios.

Fidelity `screening_era5`: GEO L0 orbit (as configured), 3896 s grid, Fresnel-Hankel capture evaluated on the validated Chebyshev (range, elevation) table (M7b), ERA5 total cloud cover at the receiver (threshold 0.5, nearest node; the weather year is an aleatory input).

## Inputs

| name | category | key | distribution | provenance | source |
|---|---|---|---|---|---|
| pv_efficiency | epistemic | `collector.pv_efficiency_reference` | type=triangular, low=0.1642, high=0.3195, mode=0.3 | derived | ends derived from sourced cell data (AZUR SPACE 3G30A-Advanced datasheet DB 00010892-01; JPL Pub. 78-15 Vol. VIII Table 7), mode assumed (PDF §5 example); configs/scenarios/*.yaml |
| pv_temperature_coefficient | epistemic | `collector.temperature_coefficient_per_K` | type=triangular, low=-0.0044, high=-0.0004, mode=-0.00231 | derived | low derived from JPL Pub. 78-15 Vol. VIII (K7 silicon), mode derived from the 3G30A datasheet dVmp/dT, dImp/dT, high = PDF §5 example (assumed) |
| electrical_to_beam_efficiency | epistemic | `transmitter.electrical_to_beam_efficiency` | type=triangular, low=0.35, high=0.6, mode=0.5 | assumed |  |
| receiver_conversion | epistemic | `receiver.conversion_efficiency` | type=triangular, low=0.3, high=0.55, mode=0.4 | assumed |  |
| pointing_jitter | epistemic | `transmitter.pointing_sigma_rad` | type=triangular, low=5e-10, high=3e-09, mode=1e-09 | assumed |  |
| radiator_specific_mass | epistemic | `thermal.radiator_specific_mass_kg_m2` | type=triangular, low=2.0, high=10.0, mode=5.0 | assumed |  |
| launch_cost | epistemic | `economics.transport.launch_cost_per_kg_leo_usd` | type=triangular, low=85.0, high=1500.0, mode=850.0 | sourced | configs/economics/geo_{optimistic,nominal,conservative}.yaml (NASA OTPS 2024, PDF pp. 28, 38, 90; 85 USD/kg derived) |
| manufacturing_cost | epistemic | `economics.manufacturing_cost_per_kg_usd` | type=triangular, low=500.0, high=10000.0, mode=2000.0 | assumed | configs/economics/geo_nominal.yaml uncertainty range (M8a) |
| weather_year | aleatory | `weather.historical_year` | type=discrete_uniform, values=(2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023) | sourced | ERA5 hourly total cloud cover 2014-2023 at the site node (Hersbach et al. 2020, doi:10.1002/qj.3803; CDS doi:10.24381/1cf1ad76); each year equally likely (assumed) |
| pointing_bias | aleatory | `transmitter.pointing_bias_rad` | type=rayleigh, scale=1e-09 | assumed |  |
| wavefront_error | aleatory | `transmitter.rms_wavefront_error_m` | type=uniform, low=0.0, high=5e-08 | assumed |  |

## Parts

- `mc`: {'status': 'complete', 'n': 128, 'failed': 0, 'failures': []}
- `scenarios`: {'status': 'complete', 'n': 3, 'failed': 0, 'failures': []}

## Nested Monte Carlo (N_e = 32, N_a = 4, seed 20261114)

Epistemic variance fraction: unbiased between-row component `max(Var_i m_i - E_i s_i^2 / N_a, 0)` over itself plus `E_i s_i^2` (ddof = 1); 'upper bound' is the uncorrected `Var_i m_i / Var Y`, which also counts the aleatory noise `sigma_a^2 / N_a` of each row mean. The epistemic interval of E_aleatory (percentiles of the row means `m_i`) carries that noise too: its sd is the 'noise sd' column.

| output | pooled p05 | p50 | p95 | epistemic interval of E_aleatory: p05 | p50 | p95 | noise sd of m_i | epistemic variance fraction | upper bound | non-finite |
|---|---|---|---|---|---|---|---|---|---|---|
| p_grid_mean_W | 1.44e+05 | 2.32e+05 | 2.964e+05 | 1.45e+05 | 2.34e+05 | 2.848e+05 | 9885 | 0.8296 | 0.8686 | 0 |
| availability | 0.3212 | 0.3781 | 0.4275 | 0.3539 | 0.378 | 0.4021 | 0.01459 | 0.03644 | 0.271 | 0 |
| availability_above_threshold | 0.3212 | 0.3781 | 0.4275 | 0.3539 | 0.378 | 0.4021 | 0.01459 | 0.03644 | 0.271 | 0 |
| pv_temperature_max_K | 304.7 | 307.6 | 314.9 | 304.7 | 307.6 | 314.8 | 0 | 1 | 1 | 0 |
| mass_proxy_kg | 1.554e+04 | 1.68e+04 | 1.844e+04 | 1.558e+04 | 1.68e+04 | 1.837e+04 | 0 | 1 | 1 | 0 |
| lcoe_usd_per_kWh | 16.11 | 21.61 | 39.06 | 16.9 | 21.83 | 39.29 | 1.09 | 0.9145 | 0.9339 | 0 |

## Scenarios

| scenario | status | p_grid_mean_W | availability | availability_above_threshold | pv_temperature_max_K | mass_proxy_kg | lcoe_usd_per_kWh |
|---|---|---|---|---|---|---|---|
| optimistic | ok | 4.628e+05 | 0.3707 | 0.3707 | 303.2 | 1.52e+04 | 10.22 |
| nominal | ok | 2.617e+05 | 0.3707 | 0.3707 | 305.8 | 1.667e+04 | 18.15 |
| conservative | ok | 6.985e+04 | 0.3707 | 0.3707 | 320.6 | 1.737e+04 | 67.31 |
