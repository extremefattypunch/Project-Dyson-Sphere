# Uncertainty study `m8_uncertainty_helio_optical`

Nested Monte Carlo and Sobol indices of the heliocentric optical MVP-class link with ERA5 weather (fidelity 'screening_era5': L0, 3896 s, batched Airy capture, ERA5 cloud outages), plus the explicit technology and PV-technology scenarios.

Fidelity `screening_era5`: L0 two-body orbit (planets/SRP omitted: 0.02 % on P_grid at 1 au, M7b/M8 re-evaluation), day-incommensurate 3896 s grid, batched Airy capture (numpy_batch), ERA5 total cloud cover at the receiver (threshold 0.5, nearest node; the weather year is an aleatory input).

## Inputs

| name | category | key | distribution | provenance | source |
|---|---|---|---|---|---|
| pv_efficiency | epistemic | `collector.pv_efficiency_reference` | type=triangular, low=0.1642, high=0.3195, mode=0.3 | derived | ends derived from sourced cell data (AZUR SPACE 3G30A-Advanced datasheet DB 00010892-01; JPL Pub. 78-15 Vol. VIII Table 7), mode assumed (PDF §5 example); configs/scenarios/*.yaml |
| pv_temperature_coefficient | epistemic | `collector.temperature_coefficient_per_K` | type=triangular, low=-0.0044, high=-0.0004, mode=-0.00231 | derived | low derived from JPL Pub. 78-15 Vol. VIII (K7 silicon), mode derived from the 3G30A datasheet dVmp/dT, dImp/dT, high = PDF §5 example (assumed) |
| electrical_to_beam_efficiency | epistemic | `transmitter.electrical_to_beam_efficiency` | type=triangular, low=0.35, high=0.6, mode=0.5 | assumed |  |
| receiver_conversion | epistemic | `receiver.conversion_efficiency` | type=triangular, low=0.3, high=0.55, mode=0.4 | assumed |  |
| pointing_jitter | epistemic | `transmitter.pointing_sigma_rad` | type=triangular, low=5e-10, high=3e-09, mode=1e-09 | assumed |  |
| radiator_specific_mass | epistemic | `thermal.radiator_specific_mass_kg_m2` | type=triangular, low=2.0, high=10.0, mode=5.0 | assumed |  |
| launch_cost | epistemic | `economics.transport.launch_cost_per_kg_leo_usd` | type=triangular, low=85.0, high=1500.0, mode=850.0 | sourced | configs/economics/{optimistic,nominal,conservative}.yaml (NASA OTPS 2024, PDF pp. 28, 38, 90; 85 USD/kg derived) |
| manufacturing_cost | epistemic | `economics.manufacturing_cost_per_kg_usd` | type=triangular, low=500.0, high=10000.0, mode=2000.0 | assumed | configs/economics/nominal.yaml uncertainty range (M8a) |
| weather_year | aleatory | `weather.historical_year` | type=discrete_uniform, values=(2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023) | sourced | ERA5 hourly total cloud cover 2014-2023 at the site node (Hersbach et al. 2020, doi:10.1002/qj.3803; CDS doi:10.24381/1cf1ad76); each year equally likely (assumed) |
| pointing_bias | aleatory | `transmitter.pointing_bias_rad` | type=rayleigh, scale=1e-09 | assumed |  |
| wavefront_error | aleatory | `transmitter.rms_wavefront_error_m` | type=uniform, low=0.0, high=5e-08 | assumed |  |

## Parts

- `mc`: {'status': 'complete', 'n': 512, 'failed': 0, 'failures': []}
- `sobol`: {'status': 'complete', 'n': 3328, 'failed': 0, 'failures': []}
- `scenarios`: {'status': 'complete', 'n': 5, 'failed': 0, 'failures': []}

## Nested Monte Carlo (N_e = 64, N_a = 8, seed 20261114)

Epistemic variance fraction: unbiased between-row component `max(Var_i m_i - E_i s_i^2 / N_a, 0)` over itself plus `E_i s_i^2` (ddof = 1); 'upper bound' is the uncorrected `Var_i m_i / Var Y`, which also counts the aleatory noise `sigma_a^2 / N_a` of each row mean. The epistemic interval of E_aleatory (percentiles of the row means `m_i`) carries that noise too: its sd is the 'noise sd' column.

| output | pooled p05 | p50 | p95 | epistemic interval of E_aleatory: p05 | p50 | p95 | noise sd of m_i | epistemic variance fraction | upper bound | non-finite |
|---|---|---|---|---|---|---|---|---|---|---|
| p_grid_mean_W | 1630 | 2496 | 3564 | 1675 | 2505 | 3458 | 82.53 | 0.8387 | 0.857 | 0 |
| availability | 0.1245 | 0.1566 | 0.1667 | 0.1448 | 0.1513 | 0.1588 | 0.004333 | 0.0229 | 0.1431 | 0 |
| availability_above_threshold | 0.1245 | 0.1566 | 0.1667 | 0.1448 | 0.1513 | 0.1588 | 0.004333 | 0.0229 | 0.1431 | 0 |
| pv_temperature_max_K | 302 | 305.1 | 312.3 | 302 | 305.1 | 312.3 | 0 | 1 | 1 | 0 |
| mass_proxy_kg | 1.571e+04 | 1.667e+04 | 1.807e+04 | 1.571e+04 | 1.667e+04 | 1.807e+04 | 0 | 1 | 1 | 0 |
| lcoe_usd_per_kWh | 1355 | 2087 | 3302 | 1394 | 2085 | 3297 | 74.39 | 0.8755 | 0.8896 | 0 |

## Sobol indices (N = 256, 500 bootstrap resamples, 0.95 confidence)

### `p_grid_mean_W`

| parameter | S1 | ± | ST | ± |
|---|---|---|---|---|
| pv_efficiency | 0.3474 | 0.09619 | 0.359 | 0.06533 |
| receiver_conversion | 0.2828 | 0.09244 | 0.2878 | 0.05582 |
| electrical_to_beam_efficiency | 0.2163 | 0.07942 | 0.2162 | 0.04457 |
| weather_year | 0.1222 | 0.06205 | 0.1307 | 0.02951 |
| wavefront_error | 0.019 | 0.02148 | 0.01606 | 0.003466 |
| pv_temperature_coefficient | 0.0008103 | 0.005181 | 0.0007539 | 0.0001907 |
| pointing_jitter | -1.057e-05 | 0.0006931 | 1.488e-05 | 3.372e-06 |
| pointing_bias | -1.912e-05 | 0.0003974 | 5.758e-06 | 2.023e-06 |
| radiator_specific_mass | 0 | 0 | 0 | 0 |
| launch_cost | 0 | 0 | 0 | 0 |
| manufacturing_cost | 0 | 0 | 0 | 0 |

Convergence (ST by N):

- N = 32: pv_efficiency 0.107±0.068, receiver_conversion 0.188±0.077, electrical_to_beam_efficiency 0.165±0.1, weather_year 0.136±0.068, wavefront_error 0.0156±0.012, pv_temperature_coefficient 0.000858±0.00067, pointing_jitter 1.03e-05±5.7e-06, pointing_bias 5.46e-06±4.4e-06, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0
- N = 64: pv_efficiency 0.365±0.12, receiver_conversion 0.281±0.084, electrical_to_beam_efficiency 0.212±0.09, weather_year 0.143±0.057, wavefront_error 0.0142±0.0077, pv_temperature_coefficient 0.000768±0.00041, pointing_jitter 1.13e-05±4.6e-06, pointing_bias 5.22e-06±3.3e-06, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0
- N = 128: pv_efficiency 0.346±0.08, receiver_conversion 0.288±0.079, electrical_to_beam_efficiency 0.219±0.065, weather_year 0.129±0.036, wavefront_error 0.014±0.0044, pv_temperature_coefficient 0.000763±0.00028, pointing_jitter 1.43e-05±4.3e-06, pointing_bias 4.92e-06±2.1e-06, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0
- N = 256: pv_efficiency 0.359±0.065, receiver_conversion 0.288±0.056, electrical_to_beam_efficiency 0.216±0.045, weather_year 0.131±0.03, wavefront_error 0.0161±0.0035, pv_temperature_coefficient 0.000754±0.00019, pointing_jitter 1.49e-05±3.4e-06, pointing_bias 5.76e-06±2e-06, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0

### `availability`

| parameter | S1 | ± | ST | ± |
|---|---|---|---|---|
| weather_year | 1.032 | 0.1754 | 1.035 | 0.1187 |
| pv_efficiency | 0 | 0 | 0 | 0 |
| pv_temperature_coefficient | 0 | 0 | 0 | 0 |
| electrical_to_beam_efficiency | 0 | 0 | 0 | 0 |
| receiver_conversion | 0 | 0 | 0 | 0 |
| pointing_jitter | 0 | 0 | 0 | 0 |
| radiator_specific_mass | 0 | 0 | 0 | 0 |
| launch_cost | 0 | 0 | 0 | 0 |
| manufacturing_cost | 0 | 0 | 0 | 0 |
| pointing_bias | 0 | 0 | 0 | 0 |
| wavefront_error | 0 | 0 | 0 | 0 |

Convergence (ST by N):

- N = 32: weather_year 1.08±0.24, pv_efficiency 0±0, pv_temperature_coefficient 0±0, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 64: weather_year 1.12±0.17, pv_efficiency 0±0, pv_temperature_coefficient 0±0, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 128: weather_year 1.07±0.16, pv_efficiency 0±0, pv_temperature_coefficient 0±0, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 256: weather_year 1.03±0.12, pv_efficiency 0±0, pv_temperature_coefficient 0±0, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, pointing_bias 0±0, wavefront_error 0±0

### `availability_above_threshold`

| parameter | S1 | ± | ST | ± |
|---|---|---|---|---|
| weather_year | 1.032 | 0.1754 | 1.035 | 0.1187 |
| pv_efficiency | 0 | 0 | 0 | 0 |
| pv_temperature_coefficient | 0 | 0 | 0 | 0 |
| electrical_to_beam_efficiency | 0 | 0 | 0 | 0 |
| receiver_conversion | 0 | 0 | 0 | 0 |
| pointing_jitter | 0 | 0 | 0 | 0 |
| radiator_specific_mass | 0 | 0 | 0 | 0 |
| launch_cost | 0 | 0 | 0 | 0 |
| manufacturing_cost | 0 | 0 | 0 | 0 |
| pointing_bias | 0 | 0 | 0 | 0 |
| wavefront_error | 0 | 0 | 0 | 0 |

Convergence (ST by N):

- N = 32: weather_year 1.08±0.24, pv_efficiency 0±0, pv_temperature_coefficient 0±0, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 64: weather_year 1.12±0.17, pv_efficiency 0±0, pv_temperature_coefficient 0±0, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 128: weather_year 1.07±0.16, pv_efficiency 0±0, pv_temperature_coefficient 0±0, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 256: weather_year 1.03±0.12, pv_efficiency 0±0, pv_temperature_coefficient 0±0, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, pointing_bias 0±0, wavefront_error 0±0

### `pv_temperature_max_K`

| parameter | S1 | ± | ST | ± |
|---|---|---|---|---|
| pv_efficiency | 0.966 | 0.1597 | 0.9656 | 0.125 |
| pv_temperature_coefficient | 0.001437 | 0.007836 | 0.002041 | 0.0004401 |
| electrical_to_beam_efficiency | 0 | 0 | 0 | 0 |
| receiver_conversion | 0 | 0 | 0 | 0 |
| pointing_jitter | 0 | 0 | 0 | 0 |
| radiator_specific_mass | 0 | 0 | 0 | 0 |
| launch_cost | 0 | 0 | 0 | 0 |
| manufacturing_cost | 0 | 0 | 0 | 0 |
| weather_year | 0 | 0 | 0 | 0 |
| pointing_bias | 0 | 0 | 0 | 0 |
| wavefront_error | 0 | 0 | 0 | 0 |

Convergence (ST by N):

- N = 32: pv_efficiency 0.274±0.12, pv_temperature_coefficient 0.00212±0.0016, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, weather_year 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 64: pv_efficiency 0.952±0.24, pv_temperature_coefficient 0.00212±0.00095, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, weather_year 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 128: pv_efficiency 0.962±0.18, pv_temperature_coefficient 0.00212±0.0007, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, weather_year 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 256: pv_efficiency 0.966±0.13, pv_temperature_coefficient 0.00204±0.00044, electrical_to_beam_efficiency 0±0, receiver_conversion 0±0, pointing_jitter 0±0, radiator_specific_mass 0±0, launch_cost 0±0, manufacturing_cost 0±0, weather_year 0±0, pointing_bias 0±0, wavefront_error 0±0

### `mass_proxy_kg`

| parameter | S1 | ± | ST | ± |
|---|---|---|---|---|
| radiator_specific_mass | 0.7143 | 0.1349 | 0.7191 | 0.1209 |
| pv_efficiency | 0.1676 | 0.06898 | 0.1737 | 0.03869 |
| electrical_to_beam_efficiency | 0.08534 | 0.05312 | 0.09147 | 0.02008 |
| pv_temperature_coefficient | 0.0003547 | 0.003538 | 0.0003764 | 0.0001017 |
| receiver_conversion | 0 | 0 | 0 | 0 |
| pointing_jitter | 0 | 0 | 0 | 0 |
| launch_cost | 0 | 0 | 0 | 0 |
| manufacturing_cost | 0 | 0 | 0 | 0 |
| weather_year | 0 | 0 | 0 | 0 |
| pointing_bias | 0 | 0 | 0 | 0 |
| wavefront_error | 0 | 0 | 0 | 0 |

Convergence (ST by N):

- N = 32: radiator_specific_mass 0.804±0.38, pv_efficiency 0.0586±0.043, electrical_to_beam_efficiency 0.0766±0.058, pv_temperature_coefficient 0.000358±0.00036, receiver_conversion 0±0, pointing_jitter 0±0, launch_cost 0±0, manufacturing_cost 0±0, weather_year 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 64: radiator_specific_mass 0.74±0.24, pv_efficiency 0.173±0.075, electrical_to_beam_efficiency 0.0912±0.046, pv_temperature_coefficient 0.000359±0.00022, receiver_conversion 0±0, pointing_jitter 0±0, launch_cost 0±0, manufacturing_cost 0±0, weather_year 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 128: radiator_specific_mass 0.749±0.17, pv_efficiency 0.185±0.056, electrical_to_beam_efficiency 0.0951±0.033, pv_temperature_coefficient 0.000369±0.00014, receiver_conversion 0±0, pointing_jitter 0±0, launch_cost 0±0, manufacturing_cost 0±0, weather_year 0±0, pointing_bias 0±0, wavefront_error 0±0
- N = 256: radiator_specific_mass 0.719±0.12, pv_efficiency 0.174±0.039, electrical_to_beam_efficiency 0.0915±0.02, pv_temperature_coefficient 0.000376±0.0001, receiver_conversion 0±0, pointing_jitter 0±0, launch_cost 0±0, manufacturing_cost 0±0, weather_year 0±0, pointing_bias 0±0, wavefront_error 0±0

### `lcoe_usd_per_kWh`

| parameter | S1 | ± | ST | ± |
|---|---|---|---|---|
| pv_efficiency | 0.3194 | 0.1094 | 0.3356 | 0.06974 |
| receiver_conversion | 0.219 | 0.0809 | 0.2275 | 0.04618 |
| electrical_to_beam_efficiency | 0.1657 | 0.07218 | 0.1936 | 0.04554 |
| manufacturing_cost | 0.1166 | 0.05015 | 0.1231 | 0.02709 |
| weather_year | 0.1064 | 0.05901 | 0.1156 | 0.02621 |
| wavefront_error | 0.01039 | 0.01947 | 0.01323 | 0.003025 |
| launch_cost | 0.0104 | 0.01877 | 0.01206 | 0.002817 |
| pv_temperature_coefficient | 0.001816 | 0.006101 | 0.0009852 | 0.0002919 |
| pointing_jitter | -0.0002932 | 0.0006474 | 1.273e-05 | 3.13e-06 |
| pointing_bias | 6.022e-06 | 0.0002935 | 3.644e-06 | 1.185e-06 |
| radiator_specific_mass | 0 | 0 | 0 | 0 |

Convergence (ST by N):

- N = 32: pv_efficiency 0.119±0.083, receiver_conversion 0.137±0.053, electrical_to_beam_efficiency 0.161±0.068, manufacturing_cost 0.203±0.1, weather_year 0.104±0.049, wavefront_error 0.0118±0.0086, launch_cost 0.0126±0.0055, pv_temperature_coefficient 0.000905±0.00053, pointing_jitter 1.25e-05±7.3e-06, pointing_bias 3.83e-06±3.6e-06, radiator_specific_mass 0±0
- N = 64: pv_efficiency 0.25±0.096, receiver_conversion 0.212±0.097, electrical_to_beam_efficiency 0.189±0.12, manufacturing_cost 0.104±0.054, weather_year 0.103±0.048, wavefront_error 0.0104±0.0056, launch_cost 0.0104±0.0048, pv_temperature_coefficient 0.00105±0.00071, pointing_jitter 1.22e-05±6.6e-06, pointing_bias 3.06e-06±1.9e-06, radiator_specific_mass 0±0
- N = 128: pv_efficiency 0.313±0.096, receiver_conversion 0.222±0.064, electrical_to_beam_efficiency 0.182±0.073, manufacturing_cost 0.113±0.034, weather_year 0.108±0.036, wavefront_error 0.0119±0.004, launch_cost 0.0121±0.004, pv_temperature_coefficient 0.000996±0.00046, pointing_jitter 1.12e-05±3.7e-06, pointing_bias 3.24e-06±1.5e-06, radiator_specific_mass 0±0
- N = 256: pv_efficiency 0.336±0.07, receiver_conversion 0.227±0.046, electrical_to_beam_efficiency 0.194±0.046, manufacturing_cost 0.123±0.027, weather_year 0.116±0.026, wavefront_error 0.0132±0.003, launch_cost 0.0121±0.0028, pv_temperature_coefficient 0.000985±0.00029, pointing_jitter 1.27e-05±3.1e-06, pointing_bias 3.64e-06±1.2e-06, radiator_specific_mass 0±0


## Scenarios

| scenario | status | p_grid_mean_W | availability | availability_above_threshold | pv_temperature_max_K | mass_proxy_kg | lcoe_usd_per_kWh |
|---|---|---|---|---|---|---|---|
| optimistic | ok | 5078 | 0.1488 | 0.1488 | 300.6 | 1.517e+04 | 907.1 |
| nominal | ok | 2870 | 0.1488 | 0.1488 | 303 | 1.661e+04 | 1611 |
| conservative | ok | 762.7 | 0.1488 | 0.1488 | 317.7 | 1.731e+04 | 6009 |
| pv_iii_v_3g30a | ok | 3100 | 0.1488 | 0.1488 | 301.6 | 1.678e+04 | 1494 |
| pv_silicon_k7_1979 | ok | 1458 | 0.1488 | 0.1488 | 317.7 | 1.553e+04 | 3130 |
