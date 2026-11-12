"""M4 exit criterion on the MVP configuration: solar/thermal along the SPICE-based MVP year.

The MVP satellite (configs/mvp_single_satellite.yaml) is propagated with the L1 model (1-day
step cap instead of the configured 300 s, as in tests/integration/test_access_mvp.py; M2
measured 4.6 cm difference over the year), its illumination is computed with the M3 conical
shadow model, and the M4 chain runs on that result. The flux is checked against an independent
inverse-square evaluation from the raw ephemeris positions, with the Sun at its light-time
retarded epoch computed here by a separate fixed-point iteration.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np
import pytest

from helioswarm.config import load_config
from helioswarm.constants import SECONDS_PER_DAY
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.kernels import KernelSet
from helioswarm.propagate import (
    initial_state_from_config,
    integrator_settings_from_config,
    propagate_l1,
)
from helioswarm.thermal import power_thermal_from_illumination, power_thermal_models_from_config
from helioswarm.timescale import time_grid_from_config
from helioswarm.visibility import OcculterRadii, satellite_illumination

from config_helpers import MVP_CONFIG

pytestmark = pytest.mark.spice

AU = 149_597_870_700.0  # m, IAU 2012 B2
C_M_S = 299_792_458.0  # m/s, exact (SI)


@pytest.fixture(scope="module")
def mvp_year(kernel_set: KernelSet) -> dict[str, Any]:
    cfg = load_config(MVP_CONFIG)
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        grid = time_grid_from_config(cfg.time, k)
        init = initial_state_from_config(cfg, eph, k)
        settings = dataclasses.replace(
            integrator_settings_from_config(cfg.time), max_step_s=SECONDS_PER_DAY
        )
        traj = propagate_l1(init, grid.epochs_tdb_s, ephemeris=eph, settings=settings).trajectory
        ill = satellite_illumination(traj, ephemeris=eph, radii=OcculterRadii.from_kernels(k))
        sun = eph.positions("SUN", grid.epochs_tdb_s)
        # Independent retarded Sun: t_e = t - |r_sat(t) - r_sun(t_e)| / c, 3 fixed-point steps
        # (contraction factor v_sun / c ~ 5e-8: the epoch error after 3 steps is < 1e-18 s).
        t_e = grid.epochs_tdb_s.copy()
        sun_ret = sun
        for _ in range(3):
            t_e = grid.epochs_tdb_s - np.linalg.norm(traj.position_m - sun_ret, axis=1) / C_M_S
            sun_ret = eph.positions("SUN", t_e)
    t0 = time.perf_counter()
    res = power_thermal_from_illumination(ill, power_thermal_models_from_config(cfg))
    elapsed = time.perf_counter() - t0
    return {
        "res": res,
        "traj": traj,
        "sun": sun,
        "sun_retarded": sun_ret,
        "ill": ill,
        "elapsed": elapsed,
        "cfg": cfg,
    }


def test_mvp_year_solar_thermal(mvp_year: dict[str, Any]) -> None:
    res = mvp_year["res"]
    sol, th = res.solar, res.thermal
    assert sol.pv_temperature_K.shape == (8767,)
    np.testing.assert_array_equal(sol.epochs_tdb_s, mvp_year["ill"].epochs_tdb_s)
    # Independent flux from the raw SPICE positions with the Sun at its light-time-retarded
    # epoch (computed in the fixture). Same model, separate code; measured max deviation
    # 5.6e-16 (float rounding). Tolerance 1e-12: ~2000x above that, yet ~6e4x below the
    # instantaneous-Sun deviation (5.9e-8, asserted below) and a TDB/UTC-like 69.184 s Sun
    # epoch shift (measured 6.7e-8), so both of those errors would fail here.
    d_ret = np.linalg.norm(mvp_year["traj"].position_m - mvp_year["sun_retarded"], axis=1)
    np.testing.assert_allclose(sol.flux_W_m2, 1361.0 * (AU / d_ret) ** 2, rtol=1e-12, atol=0)
    # The retardation is really applied: the instantaneous Sun differs by up to ~6e-8 in flux
    # (the Sun moves <~ 15 m/s about the SSB, <~ 8 km in the ~500 s light time), and by at most
    # the documented 1e-7 bound.
    d = np.linalg.norm(mvp_year["traj"].position_m - mvp_year["sun"], axis=1)
    rel_inst = np.abs(sol.flux_W_m2 / (1361.0 * (AU / d) ** 2) - 1.0)
    assert rel_inst.max() > 1e-8
    assert rel_inst.max() < 1e-7
    # Sun-tracking, never shadowed in the MVP year (M3): full collection at every epoch.
    assert np.all(sol.incidence_cosine == 1.0)
    assert np.all(sol.illuminated)
    np.testing.assert_allclose(sol.absorbed_W, 0.9 * 1e4 * sol.flux_W_m2, rtol=1e-15)
    # Energy ledger closes at every epoch; PV node solved, unique, feasible.
    assert res.ledger.closed, res.ledger.max_relative_residual
    assert np.all(th.pv_converged)
    assert np.all(th.pv_unique_root)
    assert np.all(th.feasible)
    # Near 1 au the PV plate runs at ~303 K (eta ~ 0.2997): far below the 400 K limit.
    assert np.all((sol.pv_temperature_K > 295.0) & (sol.pv_temperature_K < 310.0))
    # Cooler where farther: temperature anti-correlated with distance over the year.
    assert np.corrcoef(d, sol.pv_temperature_K)[0, 1] < -0.99
    # MVP chain: eta_pc = eta_optics = 1, eta_e2b = 0.5 -> Q_waste = P_PV / 2 = P_beam.
    np.testing.assert_allclose(th.waste_heat_W, 0.5 * sol.pv_electric_W, rtol=1e-15)
    np.testing.assert_allclose(res.chain.aperture_W, 0.5 * sol.pv_electric_W, rtol=1e-15)
    # Vectorised and cheap (measured ~0.05 s); 10 s ceiling.
    assert mvp_year["elapsed"] < 10.0
