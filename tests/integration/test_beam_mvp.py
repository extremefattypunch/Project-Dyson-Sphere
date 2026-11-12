"""M5 on the MVP year: beam capture along the real (SPICE) access geometry.

The MVP trajectory is integrated with a 1-day step cap (as in ``test_access_mvp.py``; M2 measured
4.6 cm difference over the year to the configured 300 s). The beam is evaluated on the visible
epochs only (horizontal receiver needs elevation > 0). References are recomputed here from the
``AccessResult`` fields with closed forms, independently of ``capture.py``:

* nominal capture = Airy encircled energy ``1 - J0(x)^2 - J1(x)^2``, ``x = pi D a_eff/(lambda R)``,
  ``a_eff = a sqrt(sin el)`` (equal-projected-area horizontal receiver);
* pointing loss for jitter ``s = R sigma`` much smaller than the beam: to leading order
  ``1 - E[eta]/eta(0) = 2 s^2 kappa`` with the Airy curvature ``kappa = (pi D/(lambda R))^2 / 4``
  (``<d^2> = 2 s^2`` for an isotropic 2-D Gaussian; receiver << beam so eta(d) ~ eta(0)(1 -
  kappa d^2)).
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest
from scipy import special

from helioswarm.atmosphere import atmosphere_settings_from_config, atmospheric_transmission
from helioswarm.beam import beam_settings_from_config, evaluate_beam
from helioswarm.config import load_config
from helioswarm.constants import SECONDS_PER_DAY
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.kernels import KernelSet
from helioswarm.propagate import (
    initial_state_from_config,
    integrator_settings_from_config,
    propagate_l1,
)
from helioswarm.timescale import time_grid_from_config
from helioswarm.visibility import compute_access_from_config

from config_helpers import MVP_CONFIG

pytestmark = pytest.mark.spice


def test_mvp_year_beam_capture(kernel_set: KernelSet) -> None:
    cfg = load_config(MVP_CONFIG)
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        grid = time_grid_from_config(cfg.time, k)
        settings = dataclasses.replace(
            integrator_settings_from_config(cfg.time), max_step_s=SECONDS_PER_DAY
        )
        traj = propagate_l1(
            initial_state_from_config(cfg, eph, k),
            grid.epochs_tdb_s,
            ephemeris=eph,
            settings=settings,
        ).trajectory
        (acc,) = compute_access_from_config(cfg, traj, ephemeris=eph, kernels=k)
    vis = acc.visible
    assert 1000 < int(vis.sum()) < vis.size
    r = acc.range_m[vis]
    el = acc.elevation_rad[vis]
    t0 = time.perf_counter()
    res = evaluate_beam(r, el, beam_settings_from_config(cfg), epochs_tdb_s=acc.t_emit_tdb_s[vis])
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0  # measured ~4 s for 3700 epochs

    d, lam, a, sigma = 10.0, 1.0e-6, 1000.0, 1.0e-9
    # Far-field validity holds all year (N_F ~ 6e-4 << 0.5) and is recorded per epoch.
    np.testing.assert_allclose(res.fresnel_number, d * d / (lam * r), rtol=1e-15)
    assert float(np.max(res.fresnel_number)) < 1e-3
    a_eff = a * np.sqrt(np.sin(el))
    x = math.pi * d * a_eff / (lam * r)
    np.testing.assert_allclose(
        res.capture_fraction_nominal, 1 - special.j0(x) ** 2 - special.j1(x) ** 2, rtol=1e-12
    )
    kappa = (math.pi * d / (lam * r)) ** 2 / 4
    np.testing.assert_allclose(1 - res.pointing_loss, 2 * (sigma * r) ** 2 * kappa, rtol=0.01)
    # Negative result recorded, not tuned away: a 10 m, 1 um aperture at ~1.15 au delivers
    # < 1 % of the beam to a 1 km receiver (the beam's first null is ~21 km).
    assert float(np.max(res.capture_fraction)) < 0.01
    assert np.all(res.diagnostics.first_null_radius_m > 2.0e4)
    # Clear-sky scenario with the default zenith transmittance 1.0: no loss, all valid.
    atm = atmospheric_transmission(el, atmosphere_settings_from_config(cfg))
    assert np.all(atm.in_validity)
    np.testing.assert_array_equal(atm.transmittance, 1.0)
