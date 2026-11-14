# Optimisation `m8_constellation`

Constellation demo network: ring count, satellites in ring 1, phase distribution/span, collector area, receiver beam slots, scheduling policy; maximise network P_grid and availability above 10 kW, minimise the network mass proxy.

Search fidelity: `screening` — L0 Kepler orbits, 3896 s grid (M7 production grid), batched Airy capture (numpy_batch), scenario clear sky at every site.

Objectives: max `p_grid_mean_W`; min `mass_proxy_kg`; max `availability_above_threshold`

Constraints: `availability` >= 0.9; `capacity_loss_fraction` <= 0.3

## Seeds

| seed | individuals | evaluations | cached | failed | undefined | feasible | front_size | hypervolume_final | converged | wall_s | workers |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261131 | 160 | 148 | 12 | 0 | 0 | 134 | 27 | 3.135e+11 | False | 2565 | 8 |
| 20261132 | 160 | 139 | 21 | 0 | 0 | 120 | 26 | 3.052e+11 | False | 1972 | 8 |

Missing seeds: none. Seed HV spread 0.02642 ((max - min)/max of the final hypervolumes <= 0.05: yes).

## Union Pareto front (31 designs, feasible only)

| seed | v_ring_count | v_n_satellites | v_phase_distribution | v_phase_span | v_collector_area | v_beam_slots | v_scheduling_policy | m_p_grid_mean_W | m_mass_proxy_kg | m_availability_above_threshold | m_availability | m_capacity_loss_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261131 | two_rings | 16 | random | 155.5 | 2.945e+04 | 10 | greedy | 1.886e+05 | 8.469e+05 | 1 | 1 | 0.2765 |
| 20261131 | two_rings | 16 | random | 162.2 | 2.919e+04 | 10 | optimal | 1.863e+05 | 8.403e+05 | 1 | 1 | 0.2632 |
| 20261131 | one_ring | 17 | random | 162.8 | 2.915e+04 | 10 | optimal | 1.711e+05 | 6.795e+05 | 1 | 1 | 0.2196 |
| 20261132 | one_ring | 12 | even | 95.84 | 2.887e+04 | 9 | greedy | 1.711e+05 | 4.755e+05 | 1 | 1 | 0.2088 |
| 20261132 | one_ring | 12 | even | 87.7 | 2.49e+04 | 9 | greedy | 1.563e+05 | 4.175e+05 | 1 | 1 | 0.2021 |
| 20261131 | one_ring | 14 | even | 85.53 | 1.932e+04 | 10 | greedy | 1.407e+05 | 3.917e+05 | 1 | 1 | 0.2021 |
| 20261131 | one_ring | 14 | even | 95.28 | 1.918e+04 | 10 | greedy | 1.335e+05 | 3.893e+05 | 1 | 1 | 0.1891 |
| 20261132 | one_ring | 6 | even | 95.84 | 2.916e+04 | 7 | greedy | 1.147e+05 | 2.399e+05 | 1 | 1 | 0.09466 |
| 20261132 | one_ring | 6 | even | 88.94 | 2.52e+04 | 7 | greedy | 1.043e+05 | 2.109e+05 | 1 | 1 | 0.07796 |
| 20261132 | one_ring | 5 | even | 95.94 | 2.887e+04 | 7 | greedy | 1.027e+05 | 1.982e+05 | 1 | 1 | 0.07201 |
| 20261132 | one_ring | 5 | even | 87.81 | 2.491e+04 | 7 | greedy | 9.357e+04 | 1.74e+05 | 1 | 1 | 0.05497 |
| 20261132 | one_ring | 2 | even | 98.08 | 2.953e+04 | 3 | greedy | 7.086e+04 | 8.086e+04 | 1 | 1 | 0.008407 |
| 20261132 | one_ring | 2 | even | 95.59 | 2.925e+04 | 7 | greedy | 7.049e+04 | 8.018e+04 | 1 | 1 | 0.007882 |
| 20261132 | one_ring | 2 | even | 67.05 | 2.697e+04 | 7 | greedy | 6.913e+04 | 7.463e+04 | 1 | 1 | 0.006322 |
| 20261132 | one_ring | 2 | even | 116.3 | 2.645e+04 | 3 | greedy | 6.25e+04 | 7.334e+04 | 1 | 1 | 0.0001658 |
| 20261132 | one_ring | 3 | even | 61.96 | 1.532e+04 | 10 | optimal | 4.98e+04 | 6.933e+04 | 1 | 1 | -2.22e-16 |
| 20261132 | one_ring | 3 | even | 61.96 | 1.532e+04 | 10 | optimal | 4.977e+04 | 6.93e+04 | 1 | 1 | 0 |
| 20261132 | one_ring | 3 | even | 63.27 | 1.516e+04 | 2 | optimal | 4.561e+04 | 6.872e+04 | 1 | 1 | 0.06824 |
| 20261131 | one_ring | 2 | even | 227.5 | 1.802e+04 | 2 | greedy | 4.017e+04 | 5.28e+04 | 1 | 1 | 0 |
| 20261132 | one_ring | 2 | even | 64.39 | 1.516e+04 | 2 | optimal | 3.939e+04 | 4.582e+04 | 1 | 1 | -2.22e-16 |
| 20261132 | one_ring | 2 | even | 92.19 | 1.448e+04 | 2 | optimal | 3.537e+04 | 4.417e+04 | 1 | 1 | 0 |
| 20261132 | one_ring | 2 | even | 87.71 | 1.424e+04 | 3 | greedy | 3.505e+04 | 4.357e+04 | 1 | 1 | 0 |
| 20261132 | one_ring | 2 | even | 95.44 | 1.411e+04 | 8 | greedy | 3.428e+04 | 4.325e+04 | 1 | 1 | 0 |
| 20261132 | one_ring | 2 | even | 92.19 | 1.393e+04 | 2 | greedy | 3.402e+04 | 4.281e+04 | 1 | 1 | 0 |
| 20261132 | one_ring | 2 | even | 87.71 | 1.364e+04 | 3 | greedy | 3.359e+04 | 4.213e+04 | 1 | 1 | 0 |
| 20261132 | one_ring | 5 | even | 116.3 | 2137 | 8 | greedy | 7528 | 3.516e+04 | 0.06216 | 1 | 0.009237 |
| 20261132 | one_ring | 5 | even | 199.3 | 2137 | 6 | greedy | 6391 | 3.516e+04 | 0 | 1 | 0.003745 |
| 20261132 | one_ring | 2 | even | 66.06 | 2074 | 6 | greedy | 5364 | 1.391e+04 | 0 | 1 | 0 |
| 20261132 | one_ring | 2 | even | 77.9 | 1851 | 3 | greedy | 4647 | 1.337e+04 | 0 | 1 | 0 |
| 20261132 | one_ring | 2 | even | 66.06 | 1796 | 6 | greedy | 4645 | 1.323e+04 | 0 | 1 | 0 |
| 20261132 | one_ring | 2 | even | 92.05 | 1746 | 2 | greedy | 4267 | 1.311e+04 | 0 | 1 | 0 |
