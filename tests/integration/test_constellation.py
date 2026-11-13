"""M7 constellation runs (PDF §11.2-11.4, §10.4 several transmitters, §13.2 capacity).

Short windows (2 days hourly, or 40 steps of 3896 s) keep every run to seconds. References:

* the single-satellite run (:func:`helioswarm.system.run_single_satellite`), which is the M6
  vertical slice validated earlier, for the one-satellite constellation and for the sum over
  independent satellites;
* SPICE directly (``spkezr``/``pxform``) for the Earth-relative phase;
* closed-form receiver sums and limits for the capacity checks.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
import spiceypy

from helioswarm.cli import main
from helioswarm.config import RunConfig, parse_config, read_yaml, satellite_config
from helioswarm.constellation import (
    DELIVERING,
    LINK_REASONS,
    ConstellationRunResult,
    chunk_indices,
    constellation_members,
    duty_cycle_from_config,
    evaluate_links,
    finish_constellation,
    merge_link_sets,
    run_constellation,
)
from helioswarm.constellation_io import (
    LinkDetailTooLargeError,
    linkset_path,
    load_link_set,
    read_constellation_summary,
    satellites_table,
    save_link_set,
    write_constellation_run,
    write_link_detail,
)
from helioswarm.ephemeris import SpiceEphemeris
from helioswarm.kernels import KernelSet
from helioswarm.metrics import sample_weights_s
from helioswarm.propagate import (
    integrator_settings_from_config,
    propagate_l1,
    propagate_l1_stacked,
    satellite_initial_state,
)
from helioswarm.scheduler import SOLVE_METHODS, DutyCycle, LinkDecision
from helioswarm.system import SystemRunResult, Unavailability, run_single_satellite
from helioswarm.timescale import time_grid_from_config, utc_to_tdb_s

from config_helpers import CONFIG_DIR, MVP_CONFIG

pytestmark = pytest.mark.spice

DEMO = CONFIG_DIR / "constellation_demo.yaml"
N_STEPS_3896 = 40  # 40 x 3896 s = 1.8 days


def _demo(n: int = 4, **edits: Any) -> dict[str, Any]:
    d = copy.deepcopy(read_yaml(DEMO))
    d["time"]["duration_days"] = N_STEPS_3896 * 3896.0 / 86400.0
    d["constellation"]["rings"][0]["n_satellites"] = n
    for k, v in edits.items():
        node = d
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[int(p)] if p.isdigit() else node[p]
        node[parts[-1]] = v
    return d


def _l0(d: dict[str, Any]) -> dict[str, Any]:
    """Same configuration with the closed-form L0 orbit (no integrator: exact comparisons)."""
    o = d["orbit"]
    o["model"] = "two_body_kepler"
    o["include_planetary_perturbations"] = False
    o["include_solar_radiation_pressure"] = False
    o.pop("cr", None)
    o.pop("area_to_mass_m2_kg", None)
    d["time"].pop("integrator_step_s", None)
    return d


def _run_c(ks: KernelSet, cfg: RunConfig, **kw: Any) -> ConstellationRunResult:
    with ks.loaded(verify=False) as k:
        return run_constellation(
            cfg,
            ephemeris=SpiceEphemeris(k),
            kernels=k,
            rng=np.random.default_rng(cfg.project.seed),
            **kw,
        )


def _run_s(ks: KernelSet, cfg: RunConfig) -> SystemRunResult:
    with ks.loaded(verify=False) as k:
        return run_single_satellite(
            cfg, ephemeris=SpiceEphemeris(k), kernels=k, rng=np.random.default_rng(1)
        )


# ---------------------------------------------------------------------------------------------
# N = 1 reproduces the single-satellite (M6) run
# ---------------------------------------------------------------------------------------------


def test_one_satellite_constellation_reproduces_the_m6_run_exactly(kernel_set: KernelSet) -> None:
    """configs/constellation_n1_mvp.yaml (the MVP + a one-satellite ring) vs the MVP, 2 days,
    L1 with planets and SRP: every stage power, the reason codes and P_grid agree bit for bit
    (same trajectory integrator, same per-pair beam reference, same link-budget function and
    the receiver limit applied to a one-beam sum in the same floating-point order)."""
    single = read_yaml(MVP_CONFIG)
    const = read_yaml(CONFIG_DIR / "constellation_n1_mvp.yaml")
    for d in (single, const):
        d["time"]["duration_days"] = 2.0
    s = _run_s(kernel_set, parse_config(single))
    c = _run_c(kernel_set, parse_config(const))
    np.testing.assert_array_equal(c.network_grid_W, s.link.grid_W)
    np.testing.assert_array_equal(c.site_accepted_W[:, 0], s.link.accepted_W)
    np.testing.assert_array_equal(c.site_receiver_W[:, 0], s.link.receiver_W)
    np.testing.assert_array_equal(c.site_incident_W[:, 0], s.link.captured_W)
    ls = c.links
    np.testing.assert_array_equal(ls.pv_electric_W[:, 0], s.link.pv_electric_W)
    on = c.schedule.assigned_site[:, 0] >= 0
    np.testing.assert_array_equal(on, s.available)
    np.testing.assert_array_equal(np.where(on, ls.aperture_W[:, 0], 0.0), s.link.aperture_W)
    np.testing.assert_array_equal(np.where(on, ls.atm_W[:, 0, 0], 0.0), s.link.atm_W)
    np.testing.assert_array_equal(ls.reason_code[:, 0, 0], s.reason_code)
    np.testing.assert_array_equal(ls.range_m[:, 0, 0], s.access.range_m)
    assert 0 < on.sum() < on.size  # both available and unavailable epochs are exercised


# ---------------------------------------------------------------------------------------------
# Independent satellites add; capacity limits clip
# ---------------------------------------------------------------------------------------------


def _one_site(d: dict[str, Any]) -> dict[str, Any]:
    d["receiver"]["sites"] = [d["receiver"]["sites"][0]]
    d["receiver"].pop("max_simultaneous_beams", None)
    return d


def test_identical_even_satellites_with_unlimited_receiver_sum_the_individual_runs(
    kernel_set: KernelSet,
) -> None:
    """N = 4 identical satellites evenly phased, one site with unlimited slots and power: the
    network P_grid equals the sum of the four single-satellite runs (each satellite's own
    configuration through run_single_satellite) at every epoch. L0 orbits: identical
    trajectories by construction. Tolerance 1e-12 relative: the sum of the per-satellite P_grid
    and the site-level (sum of P_captured) * eta_rx * eta_grid differ only by the order of the
    floating-point operations."""
    d = _one_site(_l0(_demo(4)))
    cfg = parse_config(d)
    c = _run_c(kernel_set, cfg)
    members = constellation_members(cfg, np.random.default_rng(0))
    total = np.zeros(len(c))
    for m in members:
        total += _run_s(kernel_set, m.config).link.grid_W
    assert np.max(total) > 0.0
    np.testing.assert_allclose(c.network_grid_W, total, rtol=1e-12, atol=0.0)
    # More than one satellite delivers at some epoch: the sum is really exercised.
    assert np.max(np.count_nonzero(c.schedule.assigned_site >= 0, axis=1)) >= 2


def test_capacity_limits_clip_and_schedule_incoming_power(kernel_set: KernelSet) -> None:
    """PDF §13.2: one site, 2 beam slots, input limit L below the sum of the two best beams.
    At every epoch: beams <= 2, no satellite twice, accepted = min(sum P_captured of the
    scheduled beams, L) <= L exactly, and the excess is reported as clipping (saturation)."""
    base = _one_site(_l0(_demo(6)))
    free = _run_c(kernel_set, parse_config(copy.deepcopy(base)))
    best_two = np.sort(free.links.captured_W[:, :, 0], axis=1)[:, -2:].sum(axis=1)
    limit = float(np.percentile(best_two[best_two > 0], 50))
    d = copy.deepcopy(base)
    d["receiver"]["sites"][0]["max_simultaneous_beams"] = 2
    d["receiver"]["sites"][0]["max_input_power_W"] = limit
    c = _run_c(kernel_set, parse_config(d))
    beams = c.schedule.site_beams[:, 0]
    assert np.all(beams <= 2)
    assert np.all(np.count_nonzero(c.schedule.assigned_site >= 0, axis=1) == beams)
    np.testing.assert_array_equal(
        c.site_accepted_W[:, 0], np.minimum(c.site_incident_W[:, 0], limit)
    )
    assert np.all(c.site_accepted_W[:, 0] <= limit)
    sat = c.site_incident_W[:, 0] > limit
    assert sat.any()
    assert (~sat & (beams > 0)).any()
    eta = c.receiver_conversion_efficiency * c.receiver_grid_efficiency
    np.testing.assert_allclose(c.network_grid_W[sat], limit * eta, rtol=1e-15)
    # The scheduled links of saturated epochs carry RECEIVER_SATURATION; capacity rejections
    # appear when more than two satellites are feasible.
    sel = c.schedule.decision[:, :, 0] == int(LinkDecision.SELECTED)
    flag = (c.link_code[:, :, 0] & np.uint32(int(Unavailability.RECEIVER_SATURATION))) != 0
    np.testing.assert_array_equal(flag, sel & sat[:, None])
    cap = (c.link_code & np.uint32(int(Unavailability.RECEIVER_CAPACITY))) != 0
    assert cap.any()
    # Unlimited run delivers at least as much at every epoch (a limit never adds power).
    assert np.all(c.network_grid_W <= free.network_grid_W * (1 + 1e-12))


def test_optimal_policy_never_delivers_less_than_greedy_and_breakdown_is_complete(
    kernel_set: KernelSet,
) -> None:
    d = _l0(_demo(6))
    for s in d["receiver"]["sites"]:
        s["max_simultaneous_beams"] = 1
    g = _run_c(kernel_set, parse_config(copy.deepcopy(d)))
    d["constellation"]["scheduler"]["policy"] = "optimal_assignment"
    o = _run_c(kernel_set, parse_config(d))
    assert np.all(o.network_grid_W >= g.network_grid_W * (1 - 1e-12))
    # The comparison is not vacuous: there is contention (greedy and assignment/MILP epochs
    # are solved, not only the best-link shortcut) and the optimum is strictly better somewhere.
    assert np.count_nonzero(g.schedule.method == SOLVE_METHODS.index("greedy")) > 0
    solved = (o.schedule.method == SOLVE_METHODS.index("assignment")) | (
        o.schedule.method == SOLVE_METHODS.index("milp")
    )
    assert np.count_nonzero(solved) > 0
    assert np.any(o.network_grid_W > g.network_grid_W * (1 + 1e-9))
    # every non-delivering satellite-epoch has exactly one primary reason, equal to an
    # independent per-element recount of the documented rule
    for r in (g, o):
        idx = r.satellite_reason_index
        assert np.all((idx == DELIVERING) == (r.schedule.assigned_site >= 0))
        assert np.all((idx >= -1) & (idx < len(LINK_REASONS)))
        np.testing.assert_array_equal(idx, _recount_satellite_reasons(r))


def _recount_satellite_reasons(r: ConstellationRunResult) -> np.ndarray:
    """Primary reason per (epoch, satellite) by explicit loops (independent of the vectorised
    implementation): per link, the first reason set in LINK_REASONS order; per satellite, the
    largest such index over its sites; scheduled satellites deliver."""
    n_t, n_sat, n_s = r.link_code.shape
    out = np.empty((n_t, n_sat), dtype=np.int64)
    for k in range(n_t):
        for i in range(n_sat):
            if r.schedule.assigned_site[k, i] >= 0:
                out[k, i] = DELIVERING
                continue
            best = -1
            for j in range(n_s):
                code = int(r.link_code[k, i, j])
                first = next((q for q, b in enumerate(LINK_REASONS) if code & int(b)), -1)
                best = max(best, first)
            out[k, i] = best
    return out


# ---------------------------------------------------------------------------------------------
# Phases, propagation
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inc", "raan", "argp", "ecc"), [(0.0, 0.0, 0.0, 0.0), (7.0, 40.0, 100.0, 0.05)]
)
def test_earth_relative_phase_places_the_satellite_at_the_offset(
    kernel_set: KernelSet, inc: float, raan: float, argp: float, ecc: float
) -> None:
    """A satellite configured at +30 deg (earth_relative) has heliocentric ecliptic longitude
    lambda_Earth + 30 deg at the element epoch, with both longitudes computed here from SPICE
    (spkezr EARTH/SUN geometric, J2000 -> ECLIPJ2000 by pxform). Tolerance 1e-9 rad: double
    round-off of the element conversions (the rotation matrices agree to ~1e-16)."""
    d = _demo(1)
    d["orbit"].update(inclination_deg=inc, raan_deg=raan, arg_periapsis_deg=argp, eccentricity=ecc)
    d["constellation"]["rings"][0]["phase_offset_deg"] = 30.0
    cfg = parse_config(d)
    sat = satellite_config(cfg, 0, 30.0)
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        t0 = utc_to_tdb_s(cfg.orbit.initial_epoch_utc, k)
        init = satellite_initial_state(sat.orbit, sat.constants, epoch_tdb_s=t0, ephemeris=eph)
        rot = np.array(spiceypy.pxform("J2000", "ECLIPJ2000", t0))
        earth = np.array(spiceypy.spkezr("EARTH", t0, "J2000", "NONE", "SUN")[0][:3]) * 1e3
        sun = np.array(
            spiceypy.spkezr("SUN", t0, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")[0][:3]
        )
    r_sat = rot @ (init.position_m - sun * 1e3)
    r_earth = rot @ earth
    lam_s = math.atan2(r_sat[1], r_sat[0])
    lam_e = math.atan2(r_earth[1], r_earth[0])
    rel = (lam_s - lam_e + math.pi) % (2 * math.pi) - math.pi
    assert rel == pytest.approx(math.radians(30.0), abs=1e-9)
    # A wrong sign convention (behind the Earth) would give -30 deg; a raw true anomaly would
    # put the satellite at lambda = 30 deg + raan + argp, not relative to the Earth.
    assert abs(rel + math.radians(30.0)) > 0.5


def test_even_and_random_phases(kernel_set: KernelSet) -> None:
    """Even phases are offset + span k/n; random phases are offset + U[0, span) drawn from the
    Generator seeded with project.seed (independent draw here), reproducible, and they mark the
    run's Generator as used."""
    d = _demo(5)
    cfg = parse_config(d)
    even = [m.phase_angle_deg for m in constellation_members(cfg, np.random.default_rng(1))]
    np.testing.assert_allclose(even, (30.0 + 300.0 * np.arange(5) / 5) % 360.0, rtol=0, atol=1e-12)
    d["constellation"]["rings"][0]["phase_distribution"] = "random"
    cfg_r = parse_config(d)
    ref = (30.0 + np.random.default_rng(cfg_r.project.seed).uniform(0.0, 300.0, 5)) % 360.0
    got = [
        m.phase_angle_deg
        for m in constellation_members(cfg_r, np.random.default_rng(cfg_r.project.seed))
    ]
    np.testing.assert_array_equal(got, ref)
    d["time"]["duration_days"] = 3 * 3896.0 / 86400.0
    r = _run_c(kernel_set, _l0_cfg(d))
    assert not r.rng_state_unchanged
    np.testing.assert_allclose(
        np.degrees(r.links.earth_relative_longitude_rad) % 360.0, ref, rtol=0, atol=1e-9
    )


def _l0_cfg(d: dict[str, Any]) -> RunConfig:
    return parse_config(_l0(copy.deepcopy(d)))


def test_stacked_l1_equals_per_satellite_l1(kernel_set: KernelSet) -> None:
    """Three satellites stacked in one solve vs three solo solves, 20 days, planets + SRP.
    Agreement to 1 cm: the step sequences differ, so the trajectories differ by the
    integration error (measured 0.8-1.3 mm here; the DOP853 error norm is not separable by
    satellite, so this agreement is empirical, see propagate_l1_stacked). A stack of one is
    the solo integrator, bit for bit. Heterogeneous rings over one year:
    scripts/validate_stacked_l1.py (docs/milestones/M7.md)."""
    cfg = parse_config(_demo(3))
    members = constellation_members(cfg, np.random.default_rng(0))
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        t0 = utc_to_tdb_s(cfg.orbit.initial_epoch_utc, k)
        t = t0 + np.arange(0.0, 20 * 86400.0 + 1.0, 3896.0)
        inits = [
            satellite_initial_state(
                m.config.orbit, m.config.constants, epoch_tdb_s=t0, ephemeris=eph
            )
            for m in members
        ]
        st = integrator_settings_from_config(cfg.time)
        stacked = propagate_l1_stacked(inits, t, ephemeris=eph, settings=st)
        solo = [propagate_l1(s, t, ephemeris=eph, settings=st) for s in inits]
        one = propagate_l1_stacked(inits[:1], t, ephemeris=eph, settings=st)
    assert stacked.stack_sizes == (3,)
    for a, b in zip(stacked.propagations, solo, strict=True):
        dev = np.linalg.norm(a.trajectory.position_m - b.trajectory.position_m, axis=1)
        assert np.max(dev) < 1e-2
        assert np.max(dev) > 0.0  # really a different (stacked) solve
    np.testing.assert_array_equal(
        one.propagations[0].trajectory.position_m, solo[0].trajectory.position_m
    )


def test_stacks_are_split_and_tolerances_scaled_by_one_over_sqrt_n(
    kernel_set: KernelSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """max_satellites_per_stack = 2 for three satellites gives stacks (2, 1); the stack of two
    is solved with rtol and atol divided by sqrt(2) (captured from the solve_ivp call), the
    stack of one with the solo settings; every trajectory agrees with its solo run to 1 cm
    over 10 days (integration error, see test_stacked_l1_equals_per_satellite_l1)."""
    import helioswarm.propagate as prop

    cfg = parse_config(_demo(3))
    members = constellation_members(cfg, np.random.default_rng(0))
    calls: list[tuple[int, float, np.ndarray]] = []
    real = prop.solve_ivp

    def spy(fun: Any, t_span: Any, y0: Any, **kw: Any) -> Any:
        calls.append((len(y0), float(kw["rtol"]), np.atleast_1d(np.asarray(kw["atol"]))))
        return real(fun, t_span, y0, **kw)

    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        t0 = utc_to_tdb_s(cfg.orbit.initial_epoch_utc, k)
        t = t0 + np.arange(0.0, 10 * 86400.0 + 1.0, 3896.0)
        inits = [
            satellite_initial_state(
                m.config.orbit, m.config.constants, epoch_tdb_s=t0, ephemeris=eph
            )
            for m in members
        ]
        st = integrator_settings_from_config(cfg.time)
        monkeypatch.setattr(prop, "solve_ivp", spy)
        stacked = propagate_l1_stacked(
            inits, t, ephemeris=eph, settings=st, max_satellites_per_stack=2
        )
        monkeypatch.undo()
        solo = [propagate_l1(s, t, ephemeris=eph, settings=st) for s in inits]
    assert stacked.stack_sizes == (2, 1)
    assert stacked.stack_index == (0, 0, 1)
    twelve = [c for c in calls if c[0] == 12]
    assert twelve
    assert all(c[1] == pytest.approx(st.rtol / math.sqrt(2), rel=1e-15) for c in twelve)
    for _, _, atol in twelve:
        np.testing.assert_allclose(atol[:3], st.atol_position_m / math.sqrt(2), rtol=1e-15)
        np.testing.assert_allclose(atol[3:6], st.atol_velocity_m_s / math.sqrt(2), rtol=1e-15)
    six = [c for c in calls if c[0] == 6]
    assert six
    assert all(c[1] == st.rtol for c in six)
    for a, b in zip(stacked.propagations, solo, strict=True):
        dev = np.linalg.norm(a.trajectory.position_m - b.trajectory.position_m, axis=1)
        assert np.max(dev) < 1e-2
    np.testing.assert_array_equal(
        stacked.propagations[2].trajectory.position_m, solo[2].trajectory.position_m
    )


# ---------------------------------------------------------------------------------------------
# Interlocks, duty cycle, batched beams, several rings
# ---------------------------------------------------------------------------------------------


def _no_slot_or_power_limits(d: dict[str, Any]) -> dict[str, Any]:
    d["receiver"].pop("max_simultaneous_beams", None)
    for s in d["receiver"]["sites"]:
        s.pop("max_simultaneous_beams", None)
        s.pop("max_input_power_W", None)
    return d


def test_summed_density_interlock_is_wired_through_the_run(kernel_set: KernelSet) -> None:
    """receiver.max_power_density_W_m2 = D between the largest single-beam peak density and
    the sum of two beams at one site (no slot or power limit, so only the density can bind):
    every single beam passes the M6 abort, the scheduled densities at each site sum (incoherent
    peak irradiances, from the link arrays) to the reported site density <= D, and a site with
    two feasible beams whose densities sum above D receives only one (RECEIVER_CAPACITY)."""
    base = _no_slot_or_power_limits(_l0(_demo(6)))
    free = _run_c(kernel_set, parse_config(copy.deepcopy(base)))
    f = free.links.feasible
    dens = np.where(f, free.links.density_W_m2, 0.0)
    d_one = float(np.max(dens))
    top2 = np.sort(dens, axis=1)[:, -2:, :].sum(axis=1)  # (T, S) two largest per site
    d_lim = d_one * (1.0 + 1e-9)
    assert np.any(top2 > d_lim)  # some site-epoch has two beams that cannot coexist
    d = copy.deepcopy(base)
    d["receiver"]["max_power_density_W_m2"] = d_lim
    c = _run_c(kernel_set, parse_config(d))
    np.testing.assert_array_equal(c.links.feasible, f)  # no single beam aborts
    sel = c.schedule.decision == int(LinkDecision.SELECTED)
    summed = np.where(sel, c.links.density_W_m2, 0.0).sum(axis=1)
    np.testing.assert_allclose(c.schedule.site_density_W_m2, summed, rtol=1e-14, atol=0.0)
    assert np.all(c.schedule.site_density_W_m2 <= d_lim)
    two_feasible_one_scheduled = (f.sum(axis=1) >= 2) & (c.schedule.site_beams == 1)
    assert two_feasible_one_scheduled.any()
    cap = (c.link_code & np.uint32(int(Unavailability.RECEIVER_CAPACITY))) != 0
    assert cap.any()
    assert np.all(c.network_grid_W <= free.network_grid_W * (1 + 1e-12))


def test_duty_cycle_limit_is_wired_through_the_run_and_the_summary(
    kernel_set: KernelSet, tmp_path: Path
) -> None:
    """scheduler.duty_cycle_max = 0.5 over 4 output steps (15584 s at 3896 s): N_w = 4,
    m = 2. Every satellite transmits in at most 2 of any 4 consecutive epochs, resting
    satellites carry DUTY_CYCLE, and the summary's satellite-time breakdown (recounted here
    from the link codes with trapezoid weights) reports the duty-cycle share."""
    d = _no_slot_or_power_limits(_l0(_demo(4)))
    d["constellation"]["scheduler"]["duty_cycle_max"] = 0.5
    d["constellation"]["scheduler"]["duty_cycle_window_s"] = 4 * 3896.0
    cfg = parse_config(d)
    assert duty_cycle_from_config(cfg) == DutyCycle(window_epochs=4, max_on=2)
    c = _run_c(kernel_set, cfg)
    on = c.schedule.assigned_site >= 0
    for k in range(on.shape[0] - 3):
        assert np.all(on[k : k + 4].sum(axis=0) <= 2)
    duty = (c.link_code & np.uint32(int(Unavailability.DUTY_CYCLE))) != 0
    assert duty.any()
    assert not np.any(duty & (c.schedule.decision == int(LinkDecision.SELECTED)))
    out = write_constellation_run(
        c, tmp_path / "r", wall_clock_start_utc="2026-01-01T00:00:00Z", make_figures=False
    )
    s = out.summary
    assert s.scheduler.duty_cycle_max == 0.5
    w = sample_weights_s(len(c), cfg.time.output_step_s)
    recount = _recount_satellite_reasons(c)
    denom = float(np.sum(w)) * c.n_satellites
    for q, reason in enumerate(LINK_REASONS):
        expected = float(np.sum(w[:, None] * (recount == q))) / denom
        got = s.unavailability.satellite_time_fraction_primary[reason.name]
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-15), reason.name
    assert s.unavailability.satellite_time_fraction_primary["DUTY_CYCLE"] > 0.0


def _two_rings(d: dict[str, Any]) -> dict[str, Any]:
    d["constellation"]["rings"][0]["n_satellites"] = 2
    d["constellation"]["rings"].append(
        {
            "name": "ring_b",
            "n_satellites": 2,
            "phase_distribution": "even",
            "phase_offset_deg": 45.0,
            "phase_span_deg": 270.0,
            "technology": {"collector_area_m2": 5000.0, "transmitter_aperture_diameter_m": 20.0},
        }
    )
    return d


def _compare_backends(kernel_set: KernelSet, backend: str) -> None:
    """2 rings (the second with its own collector and aperture), 3 sites: the batched beam
    path of evaluate_links (visible epochs of every satellite-site pair concatenated per ring
    and sliced back) against the per-pair M5 reference. Feasibility identical; P_captured and
    the network P_grid to 1e-9 relative (the batched-kernel vs reference tolerance of
    tests/unit/test_capture_batch.py::test_evaluate_beam_batched_matches_evaluate_beam and of
    the M6 JAX run test)."""
    cfg = parse_config(_two_rings(_l0(_demo(2))))
    ref = _run_c(kernel_set, cfg)
    got = _run_c(kernel_set, cfg, backend=backend)
    assert got.backend == backend
    assert ref.n_satellites == 4
    assert ref.n_sites == 3
    np.testing.assert_array_equal(got.links.feasible, ref.links.feasible)
    assert ref.links.feasible.sum() > 0
    for name in ("captured_W", "density_W_m2", "atm_W"):
        np.testing.assert_allclose(
            getattr(got.links, name), getattr(ref.links, name), rtol=1e-9, atol=0.0
        )
    np.testing.assert_allclose(got.network_grid_W, ref.network_grid_W, rtol=1e-9, atol=0.0)
    np.testing.assert_array_equal(got.schedule.assigned_site, ref.schedule.assigned_site)


def test_numpy_batch_beam_backend_equals_the_reference_on_two_rings(
    kernel_set: KernelSet,
) -> None:
    _compare_backends(kernel_set, "numpy_batch")


def test_jax_beam_backend_equals_the_reference_on_two_rings(kernel_set: KernelSet) -> None:
    pytest.importorskip("jax")
    _compare_backends(kernel_set, "jax")


@pytest.mark.gpu
def test_jax_gpu_beam_backend_equals_the_reference_on_two_rings(kernel_set: KernelSet) -> None:
    jax = pytest.importorskip("jax")
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        pytest.skip("JAX_PLATFORMS=cpu: no GPU requested (run by slurm/gpu_bench.sbatch)")
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if not gpus:
        pytest.skip("no CUDA device visible to JAX")
    _compare_backends(kernel_set, "jax")


def test_two_rings_with_technology_overrides_sum_the_individual_runs(
    kernel_set: KernelSet,
) -> None:
    """Ring 0 (2 satellites, MVP technology) + ring 1 (2 satellites, 5000 m^2 collector, 20 m
    aperture), one site with no capacity limit: the network P_grid equals the sum of the four
    satellites' own single-satellite runs (each with its ring's technology) to 1e-12 relative
    (order of floating-point operations only), and the totals and mass proxy use each ring's
    own areas (closed form)."""
    d = _one_site(_two_rings(_l0(_demo(2))))
    d["receiver"]["sites"][0].pop("max_simultaneous_beams", None)
    d["receiver"]["sites"][0].pop("max_input_power_W", None)
    cfg = parse_config(d)
    c = _run_c(kernel_set, cfg)
    members = constellation_members(cfg, np.random.default_rng(0))
    assert [m.ring_index for m in members] == [0, 0, 1, 1]
    total = np.zeros(len(c))
    per_ring = [np.zeros(len(c)), np.zeros(len(c))]
    for m in members:
        g = _run_s(kernel_set, m.config).link.grid_W
        total += g
        per_ring[m.ring_index] += g
    assert per_ring[0].max() > 0.0
    assert per_ring[1].max() > 0.0
    np.testing.assert_allclose(c.network_grid_W, total, rtol=1e-12, atol=0.0)
    sats = satellites_table(c).to_pydict()
    a0 = d["collector"]["area_m2"]
    assert sats["collector_area_m2"] == [a0, a0, 5000.0, 5000.0]
    tx = [math.pi * 25.0, math.pi * 25.0, math.pi * 100.0, math.pi * 100.0]
    np.testing.assert_allclose(sats["transmitter_aperture_area_m2"], tx, rtol=1e-15)
    mp = d["constellation"]["mass_proxy"]
    s_rad = d["thermal"]["radiator_specific_mass_kg_m2"]
    for q in range(4):
        expected = (
            mp["bus_mass_kg"]
            + mp["collector_areal_mass_kg_m2"] * sats["collector_area_m2"][q]
            + s_rad * sats["radiator_area_proxy_m2"][q]
            + mp["transmitter_areal_mass_kg_m2"] * tx[q]
        )
        assert sats["mass_proxy_kg"][q] == pytest.approx(expected, rel=1e-12)
    # The radiator is sized per satellite: half the collector needs a smaller radiator.
    assert sats["radiator_area_required_m2"][2] < sats["radiator_area_required_m2"][0]


# ---------------------------------------------------------------------------------------------
# Array split, outputs, CLI, sweep
# ---------------------------------------------------------------------------------------------


def test_chunked_link_sets_merge_to_the_single_process_run(
    kernel_set: KernelSet, tmp_path: Path
) -> None:
    """Stage A in 3 chunks (saved and re-loaded) + merge + stage B equals the one-process run
    bit for bit (L1 stacks are per chunk, so L0 orbits are used for exact equality)."""
    cfg = _l0_cfg(_demo(5))
    full = _run_c(kernel_set, cfg)
    members = constellation_members(cfg, np.random.default_rng(cfg.project.seed))
    sets = []
    with kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        for i in range(3):
            block = chunk_indices(len(members), i, 3)
            ls = evaluate_links(cfg, [members[j] for j in block], ephemeris=eph, kernels=k)
            sets.append(load_link_set(save_link_set(tmp_path / f"c{i}.npz", ls)))
        grid = time_grid_from_config(cfg.time, k)
    with pytest.raises(ValueError, match="cover"):
        merge_link_sets(sets[:2], len(members))
    merged = merge_link_sets(sets[::-1], len(members))
    res = finish_constellation(
        cfg, members, merged, grid=grid, kernel_provenance=(), rng_state_unchanged=True
    )
    np.testing.assert_array_equal(res.network_grid_W, full.network_grid_W)
    np.testing.assert_array_equal(res.link_code, full.link_code)
    assert [chunk_indices(10, i, 3) for i in range(3)] == [range(0, 3), range(3, 6), range(6, 10)]


def test_cli_constellation_run_outputs_and_determinism(
    kernel_set: KernelSet, tmp_path: Path
) -> None:
    d = _l0(_demo(4))
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(json.dumps(d))
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    kd = str(kernel_set.directory)
    assert (
        main(["run", str(cfg_path), "--out", str(out1), "--kernel-dir", kd, "--link-detail"]) == 0
    )
    assert main(["run", str(cfg_path), "--out", str(out2), "--kernel-dir", kd]) == 0
    for name in ("network.parquet", "receivers.parquet", "satellites.parquet", "summary.json"):
        assert (out1 / name).is_file()
    s1 = json.loads((out1 / "summary.json").read_text())
    s2 = json.loads((out2 / "summary.json").read_text())
    assert s1.pop("link_detail_saved") is True
    assert s2.pop("link_detail_saved") is False
    s1.pop("link_detail_rows")
    s2.pop("link_detail_rows")
    assert s1 == s2  # deterministic: identical metrics from two independent runs
    summary = read_constellation_summary(out1 / "summary.json")
    net = pq.read_table(out1 / "network.parquet")
    rec = pq.read_table(out1 / "receivers.parquet")
    # no double counting: network = sum of receivers (consistency); independent recount from
    # the assignments and link powers; sum of the per-satellite shares; beams = active
    p_rec = rec.column("grid_W").to_numpy().reshape(-1, summary.n_sites)
    np.testing.assert_allclose(net.column("grid_W").to_numpy(), p_rec.sum(axis=1), rtol=1e-15)
    dc = summary.double_counting
    assert dc.passed
    assert dc.recount_relative_difference <= 1e-12
    assert dc.satellites_relative_difference <= 1e-12
    assert dc.network_energy_J > 0.0
    # unconstrained baseline >= delivered; reception-time bound is recorded
    assert summary.unconstrained.energy_J >= summary.network.delivered_energy_J
    assert summary.unconstrained.capacity_loss_fraction is not None
    rt = summary.reception_time
    assert len(rt.receivers) == summary.n_sites
    for r in rt.receivers:
        # light time 258-964 s on this ring: beams summed at one site are minutes apart
        assert r.multi_beam_epochs == 0 or r.light_time_spread_max_s > 0.0
        # incident energy is shifted, not created: equal up to the light-time rate (|rdot|/c)
        if r.incident_energy_emission_J > 0:
            assert r.incident_energy_reception_J == pytest.approx(
                r.incident_energy_emission_J, rel=1e-3
            )
    u = summary.unavailability
    assert u.satellite_time_fraction_delivering + sum(
        u.satellite_time_fraction_primary.values()
    ) == pytest.approx(1.0, abs=1e-12)
    assert summary.totals.collector_area_m2 == 4 * d["collector"]["area_m2"]
    assert summary.totals.transmitter_aperture_area_m2 == pytest.approx(
        4 * math.pi * 25.0, rel=1e-15
    )
    mp = d["constellation"]["mass_proxy"]
    sats = pq.read_table(out1 / "satellites.parquet")
    rad = sats.column("radiator_area_required_m2").to_numpy()
    mass = (
        4 * mp["bus_mass_kg"]
        + mp["collector_areal_mass_kg_m2"] * 4 * d["collector"]["area_m2"]
        + d["thermal"]["radiator_specific_mass_kg_m2"] * rad.sum()
        + mp["transmitter_areal_mass_kg_m2"] * 4 * math.pi * 25.0
    )
    assert summary.totals.mass_proxy_kg == pytest.approx(mass, rel=1e-12)
    assert summary.power_per_mass_W_kg == pytest.approx(
        summary.network.p_grid_W.time_mean / mass, rel=1e-12
    )
    links = pq.read_table(sorted((out1 / "links").glob("*.parquet"))[0])
    assert (
        links.num_rows == summary.link_detail_rows == 4 * summary.n_sites * summary.time.n_samples
    )
    assert main(["plot", str(out1)]) == 0
    assert len(list((out1 / "figures").glob("constellation_*.png"))) == 4


def test_link_detail_size_cap_refuses_before_writing(kernel_set: KernelSet, tmp_path: Path) -> None:
    r = _run_c(kernel_set, _l0_cfg(_demo(2)))
    with pytest.raises(LinkDetailTooLargeError):
        write_link_detail(r, tmp_path / "links", max_bytes=100)
    assert not (tmp_path / "links").exists()


def test_chunk_and_merge_through_the_cli(kernel_set: KernelSet, tmp_path: Path) -> None:
    d = _l0(_demo(3))
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(json.dumps(d))
    kd = str(kernel_set.directory)
    cand = tmp_path / "cand"
    for i in range(2):
        args = ["run", str(cfg_path), "--kernel-dir", kd, "--candidates-dir", str(cand)]
        assert main([*args, "--chunk-index", str(i), "--chunk-count", "2"]) == 0
    assert (
        main(
            [
                "run",
                str(cfg_path),
                "--out",
                str(tmp_path / "m"),
                "--kernel-dir",
                kd,
                "--candidates-dir",
                str(cand),
            ]
        )
        == 0
    )
    assert main(["run", str(cfg_path), "--out", str(tmp_path / "s"), "--kernel-dir", kd]) == 0
    a = json.loads((tmp_path / "m" / "summary.json").read_text())
    b = json.loads((tmp_path / "s" / "summary.json").read_text())
    assert a == b
    assert main(["run", str(cfg_path), "--kernel-dir", kd, "--chunk-index", "0"]) == 1
    # options that the selected mode would ignore are refused
    chunk = ["run", str(cfg_path), "--kernel-dir", kd, "--candidates-dir", str(cand)]
    chunk += ["--chunk-index", "0", "--chunk-count", "2"]
    for extra in (["--out", str(tmp_path / "x")], ["--link-detail"], ["--overwrite"]):
        assert main([*chunk, *extra]) == 1
    merge = ["run", str(cfg_path), "--kernel-dir", kd, "--candidates-dir", str(cand)]
    assert main([*merge, "--out", str(tmp_path / "y"), "--backend", "numpy"]) == 1
    assert main([*merge, "--out", str(tmp_path / "y"), "--link-detail-max-gb", "1"]) == 1
    # the link files carry the code provenance; the merged manifest records it
    ls0 = load_link_set(linkset_path(cand, 0, 2))
    assert ls0.code_provenance is not None
    assert len(ls0.code_provenance["source_sha256"]) == 64
    man = json.loads((tmp_path / "m" / "manifest.json").read_text())
    assert man["stage_a_code_provenance"] == ls0.code_provenance
    # a chunk from different code is refused at the merge
    other = dict(ls0.code_provenance, source_sha256="0" * 64)
    from dataclasses import replace

    save_link_set(linkset_path(cand, 0, 2), replace(ls0, code_provenance=other))
    assert main([*merge, "--out", str(tmp_path / "z")]) == 1


def test_sweep_over_constellation_keys(kernel_set: KernelSet, tmp_path: Path) -> None:
    base = _l0(_demo(1))
    (tmp_path / "base.yaml").write_text(json.dumps(base))
    (tmp_path / "sw.yaml").write_text(
        json.dumps(
            {
                "sweep_schema_version": 1,
                "name": "csw",
                "base_config": "base.yaml",
                "parameters": {"constellation.rings[0].n_satellites": [1, 2]},
            }
        )
    )
    assert (
        main(["sweep", str(tmp_path / "sw.yaml"), "--out", str(tmp_path / "o"), "--summary-only"])
        == 0
    )
    t = pq.read_table(tmp_path / "o" / "sweep_summary.parquet").to_pydict()
    assert t["status"] == ["ok", "ok"]
    assert t["n_satellites"] == [1.0, 2.0]
    assert t["p_grid_mean_W"][1] >= t["p_grid_mean_W"][0]
