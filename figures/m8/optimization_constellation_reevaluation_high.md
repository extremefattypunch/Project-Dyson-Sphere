# High-fidelity re-evaluation: `m8_constellation`

Screening (`screening`) vs `high`; relative difference = (high - screening)/|screening|.

| candidate | v_ring_count | v_n_satellites | v_phase_distribution | v_phase_span | v_collector_area | v_beam_slots | v_scheduling_policy | high_status | feasible_high | screening_availability | high_availability | reldiff_availability | screening_availability_above_threshold | high_availability_above_threshold | reldiff_availability_above_threshold | screening_capacity_loss_fraction | high_capacity_loss_fraction | reldiff_capacity_loss_fraction | screening_mass_proxy_kg | high_mass_proxy_kg | reldiff_mass_proxy_kg | screening_p_grid_mean_W | high_p_grid_mean_W | reldiff_p_grid_mean_W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c00 | two_rings | 16 | random | 155.5 | 2.945e+04 | 10 | greedy | ok | True | 1 | 0.9274 | -0.07259 | 1 | 0.8768 | -0.1232 | 0.2765 | 0.2345 | -0.1521 | 8.469e+05 | 8.469e+05 | 5.653e-06 | 1.886e+05 | 1.464e+05 | -0.2239 |
| c01 | one_ring | 2 | even | 92.05 | 1746 | 2 | greedy | ok | False | 1 | 0.8014 | -0.1986 | 0 | 0 | — | 0 | 0 | — | 1.311e+04 | 1.311e+04 | 7.51e-06 | 4267 | 3105 | -0.2723 |
| c02 | one_ring | 2 | even | 227.5 | 1.802e+04 | 2 | greedy | ok | False | 1 | 0.8627 | -0.1373 | 1 | 0.7559 | -0.2441 | 0 | 0 | — | 5.28e+04 | 5.28e+04 | 1.762e-05 | 4.017e+04 | 2.921e+04 | -0.2729 |
