# Heliocentric vs GEO economics (same economic boundary; M8)

## optical

helio: `/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs/m8_economics/cmp_helio_1au_optical`; GEO: `/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs/m8_economics/cmp_geo_optical`

| architecture | point | annual_energy_kWh | lcoe_optimistic | lcoe_nominal | lcoe_conservative | mc_nominal_lcoe_p05 | mc_nominal_lcoe_p50 | mc_nominal_lcoe_p95 | npv_nominal | mass_kg_nominal | delta_v_m_s_nominal | flags_nominal | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heliocentric | p0000 | 780 | undefined | undefined | undefined | undefined | undefined | undefined | -4.367e+08 | 1.662e+04 | 3617 | negligible_energy | costed |
| heliocentric | p0001 | 7.677e+04 | 250.1 | 527.9 | 1298 | 444.7 | 587.5 | 739.9 | -7.776e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0002 | 2.206e+06 | 170 | 824.1 | 1779 | 409.6 | 798.2 | 1149 | -3.488e+10 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0003 | 6984 | 2504 | 5376 | 2.116e+04 | 4806 | 7264 | 1.121e+04 | -7.203e+08 | 4.804e+04 | 3617 |  | costed |
| heliocentric | p0004 | 6.069e+05 | 34.68 | 91.14 | 306.5 | 80.14 | 112 | 159.1 | -1.061e+09 | 4.804e+04 | 3617 |  | costed |
| heliocentric | p0005 | 2.573e+06 | 146.5 | 712.3 | 1559 | 360.4 | 694.5 | 994.9 | -3.516e+10 | 4.804e+04 | 3617 |  | costed |
| heliocentric | p0006 | 7.321e+04 | 525.4 | 2809 | 1.543e+04 | 2332 | 4311 | 7478 | -3.947e+09 | 4.054e+05 | 3617 |  | costed |
| heliocentric | p0007 | 2.187e+06 | 19.21 | 102.2 | 534.1 | 86.27 | 152 | 258.7 | -4.286e+09 | 4.054e+05 | 3617 |  | costed |
| heliocentric | p0008 | 2.704e+06 | 147.1 | 739.9 | 1846 | 436.1 | 764.7 | 1065 | -3.839e+10 | 4.054e+05 | 3617 |  | costed |
| GEO | p0000 | 6.139e+06 | 2.561 | 3.888 | 11.03 | 3.663 | 4.711 | 6.333 | -4.521e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0001 | 6.179e+06 | 3.12 | 6.738 | 17.15 | 5.694 | 7.517 | 9.46 | -7.93e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0002 | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | not costed: failed |
| GEO | p0003 | 6.168e+06 | 2.87 | 6.593 | 26.83 | 5.851 | 8.911 | 13.56 | -7.744e+08 | 4.811e+04 | 4700 |  | costed |
| GEO | p0004 | 6.182e+06 | 3.439 | 9.452 | 32.96 | 8.233 | 11.69 | 16.4 | -1.115e+09 | 4.811e+04 | 4700 |  | costed |
| GEO | p0005 | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | not costed: failed |
| GEO | p0006 | 6.178e+06 | 6.514 | 37.52 | 206.8 | 30.99 | 56.73 | 95.97 | -4.441e+09 | 4.055e+05 | 4700 |  | costed |
| GEO | p0007 | 6.183e+06 | 7.084 | 40.36 | 212.8 | 33.93 | 59.41 | 98.48 | -4.782e+09 | 4.055e+05 | 4700 |  | costed |
| GEO | p0008 | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | undefined | not costed: failed |

Ratios heliocentric / GEO per criterion: (a) best point of each architecture selected separately under that criterion; (b) min-max of the ratio at the same point over the points costed in both.

| criterion | best helio point | helio LCOE | best GEO point | GEO LCOE | (a) ratio | (b) same-point ratio min | max | points |
|---|---|---|---|---|---|---|---|---|
| lcoe_optimistic | p0007 | 19.21 | p0000 | 2.561 | 7.502 | 2.712 | 872.6 | 5 |
| lcoe_nominal | p0004 | 91.14 | p0000 | 3.888 | 23.44 | 2.532 | 815.4 | 5 |
| lcoe_conservative | p0004 | 306.5 | p0000 | 11.03 | 27.78 | 2.509 | 788.6 | 5 |
| mc_nominal_lcoe_p50 | p0004 | 112 | p0000 | 4.711 | 23.77 | 2.558 | 815.1 | 5 |

## microwave

helio: `/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs/m8_economics/cmp_helio_1au_microwave`; GEO: `/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs/m8_economics/cmp_geo_microwave`

| architecture | point | annual_energy_kWh | lcoe_optimistic | lcoe_nominal | lcoe_conservative | mc_nominal_lcoe_p05 | mc_nominal_lcoe_p50 | mc_nominal_lcoe_p95 | npv_nominal | mass_kg_nominal | delta_v_m_s_nominal | flags_nominal | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heliocentric | p0000 | 0.0349 | undefined | undefined | undefined | undefined | undefined | undefined | -8.937e+10 | 9.829e+06 | 3617 | negligible_energy | costed |
| heliocentric | p0001 | 0.8725 | undefined | undefined | undefined | undefined | undefined | undefined | -9.763e+10 | 9.829e+06 | 3617 | negligible_energy | costed |
| heliocentric | p0002 | 3.49 | undefined | undefined | undefined | undefined | undefined | undefined | -1.235e+11 | 9.829e+06 | 3617 | negligible_energy | costed |
| heliocentric | p0003 | 0.1396 | undefined | undefined | undefined | undefined | undefined | undefined | -3.553e+11 | 3.928e+07 | 3617 | negligible_energy | costed |
| heliocentric | p0004 | 3.489 | undefined | undefined | undefined | undefined | undefined | undefined | -3.635e+11 | 3.928e+07 | 3617 | negligible_energy | costed |
| heliocentric | p0005 | 13.96 | undefined | undefined | undefined | undefined | undefined | undefined | -3.894e+11 | 3.928e+07 | 3617 | negligible_energy | costed |
| heliocentric | p0006 | 0.5579 | undefined | undefined | undefined | undefined | undefined | undefined | -1.419e+12 | 1.571e+08 | 3617 | negligible_energy | costed |
| heliocentric | p0007 | 13.95 | undefined | undefined | undefined | undefined | undefined | undefined | -1.427e+12 | 1.571e+08 | 3617 | negligible_energy | costed |
| heliocentric | p0008 | 55.79 | undefined | undefined | undefined | undefined | undefined | undefined | -1.453e+12 | 1.571e+08 | 3617 | negligible_energy | costed |
| GEO | p0000 | 5.289e+05 | 1207 | 1e+04 | 5.795e+04 | 8140 | 1.543e+04 | 2.653e+04 | -1.015e+11 | 9.829e+06 | 4700 |  | costed |
| GEO | p0001 | 9.697e+06 | 74.7 | 589.8 | 3256 | 484.4 | 883.6 | 1494 | -1.097e+11 | 9.829e+06 | 4700 |  | costed |
| GEO | p0002 | 1.712e+07 | 58.07 | 412.8 | 2014 | 342.6 | 574.7 | 922.9 | -1.356e+11 | 9.829e+06 | 4700 |  | costed |
| GEO | p0003 | 2.028e+06 | 1231 | 1.037e+04 | 6.031e+04 | 8427 | 1.603e+04 | 2.761e+04 | -4.037e+11 | 3.928e+07 | 4700 |  | costed |
| GEO | p0004 | 1.709e+07 | 151.1 | 1256 | 7211 | 1030 | 1924 | 3301 | -4.12e+11 | 3.928e+07 | 4700 |  | costed |
| GEO | p0005 | 1.777e+07 | 160.4 | 1284 | 7097 | 1054 | 1924 | 3254 | -4.378e+11 | 3.928e+07 | 4700 |  | costed |
| GEO | p0006 | 6.693e+06 | 1483 | 1.256e+04 | 7.307e+04 | 1.02e+04 | 1.942e+04 | 3.345e+04 | -1.613e+12 | 1.571e+08 | 4700 |  | costed |
| GEO | p0007 | 1.774e+07 | 564.3 | 4761 | 2.761e+04 | 3876 | 7343 | 1.263e+04 | -1.621e+12 | 1.571e+08 | 4700 |  | costed |
| GEO | p0008 | 1.797e+07 | 572.1 | 4774 | 2.742e+04 | 3916 | 7314 | 1.255e+04 | -1.647e+12 | 1.571e+08 | 4700 |  | costed |

Ratios heliocentric / GEO per criterion: (a) best point of each architecture selected separately under that criterion; (b) min-max of the ratio at the same point over the points costed in both.

| criterion | best helio point | helio LCOE | best GEO point | GEO LCOE | (a) ratio | (b) same-point ratio min | max | points |
|---|---|---|---|---|---|---|---|---|
| lcoe_optimistic | - | undefined | p0002 | 58.07 | undefined | undefined | undefined | 0 |
| lcoe_nominal | - | undefined | p0002 | 412.8 | undefined | undefined | undefined | 0 |
| lcoe_conservative | - | undefined | p0002 | 2014 | undefined | undefined | undefined | 0 |
| mc_nominal_lcoe_p50 | - | undefined | p0002 | 574.7 | undefined | undefined | undefined | 0 |

## optical_era5_example_site

helio: `/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs/m8_economics/cmp_helio_1au_optical_era5`; GEO: `/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs/m8_economics/cmp_geo_optical_era5`

| architecture | point | annual_energy_kWh | lcoe_optimistic | lcoe_nominal | lcoe_conservative | mc_nominal_lcoe_p05 | mc_nominal_lcoe_p50 | mc_nominal_lcoe_p95 | npv_nominal | mass_kg_nominal | delta_v_m_s_nominal | flags_nominal | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heliocentric | p0000 | 2.373e+04 | 809.2 | 1708 | 4200 | 1439 | 1901 | 2394 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0001 | 2.501e+04 | 767.7 | 1620 | 3985 | 1365 | 1803 | 2271 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0002 | 2.807e+04 | 684 | 1444 | 3550 | 1216 | 1607 | 2023 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0003 | 2.761e+04 | 695.5 | 1468 | 3610 | 1237 | 1634 | 2058 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0004 | 2.755e+04 | 697.1 | 1471 | 3618 | 1239 | 1637 | 2062 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0005 | 2.542e+04 | 755.6 | 1595 | 3921 | 1343 | 1775 | 2235 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0006 | 2.728e+04 | 703.9 | 1486 | 3653 | 1251 | 1653 | 2082 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0007 | 2.635e+04 | 728.9 | 1538 | 3783 | 1296 | 1712 | 2156 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0008 | 2.643e+04 | 726.4 | 1533 | 3770 | 1291 | 1706 | 2149 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0009 | 2.106e+04 | 911.7 | 1924 | 4732 | 1621 | 2142 | 2697 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| GEO | p0000 | 2.171e+06 | 8.879 | 19.18 | 48.8 | 16.2 | 21.39 | 26.92 | -7.969e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0001 | 2.348e+06 | 8.213 | 17.74 | 45.13 | 14.99 | 19.79 | 24.9 | -7.967e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0002 | 2.677e+06 | 7.203 | 15.56 | 39.58 | 13.14 | 17.35 | 21.84 | -7.964e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0003 | 2.514e+06 | 7.67 | 16.57 | 42.15 | 14 | 18.48 | 23.26 | -7.966e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0004 | 2.382e+06 | 8.094 | 17.48 | 44.48 | 14.77 | 19.5 | 24.54 | -7.967e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0005 | 2.319e+06 | 8.313 | 17.95 | 45.68 | 15.17 | 20.03 | 25.2 | -7.967e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0006 | 2.556e+06 | 7.543 | 16.29 | 41.45 | 13.77 | 18.17 | 22.87 | -7.965e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0007 | 2.314e+06 | 8.33 | 17.99 | 45.78 | 15.2 | 20.07 | 25.26 | -7.967e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0008 | 2.369e+06 | 8.138 | 17.57 | 44.72 | 14.85 | 19.61 | 24.67 | -7.967e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0009 | 2.013e+06 | 9.576 | 20.68 | 52.62 | 17.48 | 23.07 | 29.03 | -7.97e+08 | 1.669e+04 | 4700 |  | costed |

Ratios heliocentric / GEO per criterion: (a) best point of each architecture selected separately under that criterion; (b) min-max of the ratio at the same point over the points costed in both.

| criterion | best helio point | helio LCOE | best GEO point | GEO LCOE | (a) ratio | (b) same-point ratio min | max | points |
|---|---|---|---|---|---|---|---|---|
| lcoe_optimistic | p0002 | 684 | p0002 | 7.203 | 94.96 | 86.13 | 95.21 | 10 |
| lcoe_nominal | p0002 | 1444 | p0002 | 15.56 | 92.81 | 84.17 | 93.05 | 10 |
| lcoe_conservative | p0002 | 3550 | p0002 | 39.58 | 89.69 | 81.34 | 89.92 | 10 |
| mc_nominal_lcoe_p50 | p0002 | 1607 | p0002 | 17.35 | 92.58 | 83.97 | 92.82 | 10 |

## optical_era5_aswan

helio: `/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs/m8_economics/cmp_helio_1au_optical_era5_aswan`; GEO: `/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/runs/m8_economics/cmp_geo_optical_era5_aswan`

| architecture | point | annual_energy_kWh | lcoe_optimistic | lcoe_nominal | lcoe_conservative | mc_nominal_lcoe_p05 | mc_nominal_lcoe_p50 | mc_nominal_lcoe_p95 | npv_nominal | mass_kg_nominal | delta_v_m_s_nominal | flags_nominal | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heliocentric | p0000 | 6.344e+04 | 302.7 | 638.9 | 1571 | 538.1 | 711 | 895.5 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0001 | 6.369e+04 | 301.5 | 636.4 | 1565 | 536 | 708.2 | 892 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0002 | 6.222e+04 | 308.6 | 651.4 | 1602 | 548.7 | 724.9 | 913 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0003 | 5.942e+04 | 323.2 | 682.1 | 1677 | 574.5 | 759.1 | 956 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0004 | 6.498e+04 | 295.5 | 623.7 | 1534 | 525.4 | 694.1 | 874.2 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0005 | 6.136e+04 | 313 | 660.6 | 1624 | 556.4 | 735.1 | 925.9 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0006 | 6.389e+04 | 300.5 | 634.3 | 1560 | 534.3 | 705.9 | 889.1 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0007 | 6.42e+04 | 299.1 | 631.3 | 1552 | 531.8 | 702.6 | 884.9 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0008 | 6.313e+04 | 304.2 | 642 | 1579 | 540.8 | 714.5 | 899.8 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| heliocentric | p0009 | 6.244e+04 | 307.5 | 649.1 | 1596 | 546.7 | 722.4 | 909.8 | -7.777e+08 | 1.662e+04 | 3617 |  | costed |
| GEO | p0000 | 5.581e+06 | 3.454 | 7.46 | 18.98 | 6.304 | 8.323 | 10.47 | -7.936e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0001 | 5.539e+06 | 3.48 | 7.516 | 19.13 | 6.352 | 8.385 | 10.55 | -7.937e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0002 | 5.378e+06 | 3.585 | 7.743 | 19.7 | 6.543 | 8.638 | 10.87 | -7.938e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0003 | 5.233e+06 | 3.684 | 7.957 | 20.25 | 6.724 | 8.877 | 11.17 | -7.939e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0004 | 5.633e+06 | 3.423 | 7.391 | 18.81 | 6.246 | 8.246 | 10.38 | -7.936e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0005 | 5.276e+06 | 3.654 | 7.892 | 20.08 | 6.669 | 8.804 | 11.08 | -7.939e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0006 | 5.623e+06 | 3.429 | 7.405 | 18.84 | 6.257 | 8.261 | 10.4 | -7.936e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0007 | 5.581e+06 | 3.454 | 7.46 | 18.98 | 6.304 | 8.323 | 10.47 | -7.936e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0008 | 5.555e+06 | 3.471 | 7.496 | 19.07 | 6.334 | 8.363 | 10.52 | -7.936e+08 | 1.669e+04 | 4700 |  | costed |
| GEO | p0009 | 5.451e+06 | 3.537 | 7.638 | 19.44 | 6.454 | 8.521 | 10.72 | -7.937e+08 | 1.669e+04 | 4700 |  | costed |

Ratios heliocentric / GEO per criterion: (a) best point of each architecture selected separately under that criterion; (b) min-max of the ratio at the same point over the points costed in both.

| criterion | best helio point | helio LCOE | best GEO point | GEO LCOE | (a) ratio | (b) same-point ratio min | max | points |
|---|---|---|---|---|---|---|---|---|
| lcoe_optimistic | p0004 | 295.5 | p0004 | 3.423 | 86.34 | 85.65 | 87.71 | 10 |
| lcoe_nominal | p0004 | 623.7 | p0004 | 7.391 | 84.39 | 83.7 | 85.72 | 10 |
| lcoe_conservative | p0004 | 1534 | p0004 | 18.81 | 81.54 | 80.89 | 82.83 | 10 |
| mc_nominal_lcoe_p50 | p0004 | 694.1 | p0004 | 8.246 | 84.18 | 83.5 | 85.51 | 10 |

