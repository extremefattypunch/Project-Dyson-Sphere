"""Programmatic model validation (``helioswarm validate-models``; PDF §7.4, §8.5, §9, §13.1-13.3).

Each check calls the package's own model functions (not pytest) and compares the result with
an **independent** reference: a closed form written here, an external authoritative value with
its source stated (JPL Horizons, ITU-R validation examples), an independent library (HCIPy,
scipy quadrature) or an exact invariant. Every check reports the value, the reference, the
tolerance (with its reason) and pass/fail. A check that cannot run (missing optional library)
is reported as *not run*, and any failure or non-run is listed under "unresolved" in the
report; nothing is dropped or tuned.

The report is written into ``docs/validation_report.md`` between the markers
``<!-- BEGIN validate-models -->`` and ``<!-- END validate-models -->`` (the curated per-
milestone test record outside the markers is kept); a file without markers gets the generated
block appended.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import integrate, special

from helioswarm.aperture import ApertureField, FresnelHankelPattern
from helioswarm.capture import (
    capture_fraction,
    capture_fraction_cartesian,
    capture_statistics,
)
from helioswarm.config import RunConfig, parse_config, read_yaml
from helioswarm.constants import (
    AU_M,
    CMB_TEMPERATURE_K,
    GM_SUN_M3_S2,
    SECONDS_PER_DAY,
    SPEED_OF_LIGHT_M_S,
    WGS84_SEMIMAJOR_AXIS_M,
    WGS84_SEMIMINOR_AXIS_M,
)
from helioswarm.ephemeris import (
    SSB,
    AnalyticEphemeris,
    BodyName,
    SpiceEphemeris,
    shift_state_origin,
)
from helioswarm.frames import FrameName, OriginName, transform_state
from helioswarm.kernels import KernelSet, LoadedKernels
from helioswarm.link_budget import (
    STAGE_COLUMNS,
    ReceiverModel,
    evaluate_link_budget,
    link_budget_violations,
)
from helioswarm.optical import AIRY_FIRST_NULL_FACTOR, J1_FIRST_ZERO, AiryPattern
from helioswarm.orbit import OrbitalElements, elements_to_state, orbital_period_s, propagate_kepler
from helioswarm.power_chain import OnboardEfficiencies
from helioswarm.propagate import ForceModel, IntegratorSettings, SatelliteState, propagate_l1
from helioswarm.solar import solar_flux_W_m2
from helioswarm.system import SystemRunResult, run_single_satellite
from helioswarm.thermal import (
    RadiatorModel,
    power_thermal_models_from_config,
    power_thermal_vs_distance,
    radiator_required_area_m2,
)
from helioswarm.visibility import segment_intersects_ellipsoid

BEGIN_MARK: Final[str] = "<!-- BEGIN validate-models -->"
END_MARK: Final[str] = "<!-- END validate-models -->"

HORIZONS_EARTH_WRT_SUN_KM: Final[tuple[tuple[float, float, float, float], ...]] = (
    (2451545.0, -2.649903367743050e07, 1.327574173383451e08, 5.755671847054072e07),
    (2455000.5, -8.549745292681532e06, -1.392338519857522e08, -6.036206935622022e07),
    (2460313.5, -3.250851613065001e07, 1.316287155512157e08, 5.705947751136197e07),
    (2460497.5, 3.703680700587676e07, -1.353513670973292e08, -5.867363874248910e07),
    (2462502.5, -2.600847757810495e07, 1.328460643106323e08, 5.758542822344756e07),
    (2469807.5, -2.567281493326828e07, 1.329033229120269e08, 5.760271197711208e07),
)
"""JPL Horizons (DE441) geometric Earth (399) position relative to the Sun centre (500@10),
ICRF, TDB Julian dates, km; retrieved 2026-11-12 with the API query recorded in
``tests/unit/test_ephemeris.py`` (``HORIZONS_QUERY``); values copied verbatim."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one validation check.

    Attributes:
        section: PDF section (e.g. ``§9``).
        name: What is checked.
        value: Measured value (formatted).
        reference: Reference value and its source.
        tolerance: Tolerance and its reason.
        passed: ``True``/``False``, or ``None`` if the check could not run.
        note: Extra information (error message for a failed/unrun check).
        wall_s: Time taken [s].
    """

    section: str
    name: str
    value: str
    reference: str
    tolerance: str
    passed: bool | None
    note: str = ""
    wall_s: float = 0.0


@dataclass(frozen=True)
class _Outcome:
    value: str
    reference: str
    tolerance: str
    passed: bool | None
    note: str = ""


@dataclass(frozen=True)
class ValidationContext:
    """Inputs shared by the checks.

    Attributes:
        base_config: Validated MVP configuration (end-to-end checks shorten it to 2 days).
        kernel_set: Verified kernel set.
        include_slow: Run the slower convergence checks (L1 step and diurnal-sampling runs).
    """

    base_config: RunConfig
    kernel_set: KernelSet
    include_slow: bool = True


def _g(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}g}"


# ---------------------------------------------------------------------------------------------
# §7.4 orbit and geometry
# ---------------------------------------------------------------------------------------------


def _c_l0_circular(_: ValidationContext) -> _Outcome:
    el = OrbitalElements(
        a_m=AU_M, e=0.0, i_rad=0.3, raan_rad=0.2, argp_rad=0.0, nu_rad=0.0,
        mu_m3_s2=GM_SUN_M3_S2, epoch_tdb_s=0.0,
    )  # fmt: skip
    period = orbital_period_s(AU_M, GM_SUN_M3_S2)
    t = np.linspace(0.0, 10.0 * period, 2001)
    traj = propagate_kepler(el, t).trajectory
    r = np.linalg.norm(traj.position_m, axis=1)
    dev = float(np.max(np.abs(r / AU_M - 1.0)))
    ret = float(np.linalg.norm(traj.position_m[-1] - traj.position_m[0]) / AU_M)
    ok = dev < 1e-12 and ret < 1e-11
    return _Outcome(
        f"max|r/a-1| = {dev:.2e}; |r(10T)-r(0)|/a = {ret:.2e}",
        "r = a exactly; return after 10 Kepler periods 2 pi sqrt(a^3/mu) (Kepler III)",
        "1e-12 / 1e-11 (float64 round-off of the closed form)",
        ok,
    )


def _c_l1_circular(_: ValidationContext) -> _Outcome:
    eph = AnalyticEphemeris()
    el = OrbitalElements(
        a_m=AU_M, e=0.0, i_rad=0.3, raan_rad=0.2, argp_rad=0.0, nu_rad=0.0,
        mu_m3_s2=GM_SUN_M3_S2, epoch_tdb_s=0.0,
    )  # fmt: skip
    st = transform_state(elements_to_state(el), "J2000")
    init = SatelliteState(state=shift_state_origin(st, SSB, eph), force_model=ForceModel(("SUN",)))
    prop = propagate_l1(
        init,
        np.linspace(0.0, 3 * 365.25 * SECONDS_PER_DAY, 300),
        ephemeris=eph,
        settings=IntegratorSettings(max_step_s=SECONDS_PER_DAY),
    )
    r = np.linalg.norm(prop.trajectory.position_m, axis=1)
    dev = float(np.max(np.abs(r / AU_M - 1.0)))
    drift = prop.diagnostics.sun_two_body_energy_rel_drift
    return _Outcome(
        f"max|r/a-1| = {dev:.2e}; energy drift {drift:.2e}",
        "circular two-body orbit (L1, Sun only, Sun at rest), 3 years",
        "1e-12 (= integrator rtol)",
        dev < 1e-12 and drift < 1e-12,
    )


def _c_horizons(ctx: ValidationContext) -> _Outcome:
    with ctx.kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        errs = []
        for jd, x, y, z in HORIZONS_EARTH_WRT_SUN_KM:
            et = (jd - 2451545.0) * SECONDS_PER_DAY
            r = eph.state("EARTH", et).position_m - eph.state("SUN", et).position_m
            errs.append(float(np.linalg.norm(r - 1e3 * np.array([x, y, z]))))
    return _Outcome(
        f"max position error {max(errs):.3e} m over 6 epochs 2000-2050",
        "JPL Horizons DE441 Earth wrt Sun (retrieved 2026-11-12)",
        "1 m (DE440 vs DE441 differ by < 6 cm; a unit/origin/time-scale error fails by >1e3 m)",
        max(errs) < 1.0,
    )


def _c_sidereal(ctx: ValidationContext) -> _Outcome:
    from helioswarm.ground import GroundSite, SpiceEarthOrientation

    with ctx.kernel_set.loaded(verify=False) as k:
        eo = SpiceEarthOrientation(k)
        site = GroundSite.from_geodetic("eq", 0.0, math.radians(37.0), 0.0)
        period = SECONDS_PER_DAY * 360.0 / 360.9856235
        t0 = 9.5e8
        rot = eo.rotation(np.array([t0, t0 + period]))
        v0 = rot.matrix[0] @ site.position_ef_m
        v1 = rot.matrix[1] @ site.position_ef_m
    ang = float(np.arctan2(np.linalg.norm(np.cross(v0, v1)), np.dot(v0, v1)))
    return _Outcome(
        f"site direction change after one IAU_EARTH period = {ang:.2e} rad",
        "0 (one rotation per 86400*360/360.9856235 s; pck00011 BODY399_PM)",
        "1e-6 rad (float64 rounding of W ~ 7e4 rad)",
        ang < 1e-6,
    )


def _c_occultation(_: ValidationContext) -> _Outcome:
    a, b = WGS84_SEMIMAJOR_AXIS_M, WGS84_SEMIMINOR_AXIS_M
    far = np.array([[2.0e11, 0.0, 0.0]])
    through = segment_intersects_ellipsoid(far, np.array([[-a, 0.0, 0.0]]), a, b, 1.0 - 1e-6)
    above = segment_intersects_ellipsoid(
        np.array([[2.0e11, a * 1.0001, 0.0]]), np.array([[-2.0e11, a * 1.0001, 0.0]]), a, b
    )
    grazing_in = segment_intersects_ellipsoid(
        np.array([[2.0e11, a * 0.9999, 0.0]]), np.array([[-2.0e11, a * 0.9999, 0.0]]), a, b
    )
    ok = bool(through[0]) and not bool(above[0]) and bool(grazing_in[0])
    return _Outcome(
        f"through-Earth {bool(through[0])}, tangent +1e-4 a {bool(above[0])}, "
        f"-1e-4 a {bool(grazing_in[0])}",
        "True, False, True (geometry of a chord vs the WGS84 ellipsoid)",
        "exact (boolean)",
        ok,
    )


@dataclass(frozen=True)
class _UniformEarth:
    """Synthetic provider (EphemerisProvider protocol): only the Earth, in uniform rectilinear
    motion ``r(t) = r0 + v t`` (J2000 / SSB). Not a physical ephemeris; it gives the light-time
    equation a closed form."""

    r0_m: tuple[float, float, float]
    v_m_s: tuple[float, float, float]
    frame: FrameName = field(default="J2000", init=False)
    origin: OriginName = field(default=SSB, init=False)

    @property
    def source_kernel_ids(self) -> tuple[str, ...]:
        return ("validate-models-uniform-earth",)

    def check_coverage(self, bodies: Any, start_tdb_s: float, end_tdb_s: float) -> None:
        for b in bodies:
            if b != "EARTH":
                raise KeyError(f"synthetic provider has no body {b!r}")

    def states(self, body: BodyName, epochs_tdb_s: Any) -> Any:
        from helioswarm.frames import CartesianTrajectory

        self.check_coverage((body,), 0.0, 0.0)
        t = np.atleast_1d(np.asarray(epochs_tdb_s, dtype=np.float64))
        r0, v = np.asarray(self.r0_m), np.asarray(self.v_m_s)
        return CartesianTrajectory(
            epochs_tdb_s=t, position_m=r0[None, :] + t[:, None] * v[None, :],
            velocity_m_s=np.broadcast_to(v, (t.size, 3)), frame="J2000", origin=SSB,
        )  # fmt: skip

    def positions(self, body: BodyName, epochs_tdb_s: Any) -> Any:
        return np.array(self.states(body, epochs_tdb_s).position_m)

    def state(self, body: BodyName, epoch_tdb_s: float) -> Any:
        from helioswarm.ephemeris import EphemerisState

        tr = self.states(body, epoch_tdb_s)
        return EphemerisState(body=body, epoch_tdb_s=float(epoch_tdb_s),
                              position_m=tr.position_m[0], velocity_m_s=tr.velocity_m_s[0],
                              source_kernel_ids=self.source_kernel_ids)  # fmt: skip

    def gm_m3_s2(self, body: BodyName) -> float:
        raise KeyError("synthetic provider has no GM values")


def _c_light_time(_: ValidationContext) -> _Outcome:
    from helioswarm.frames import CartesianTrajectory
    from helioswarm.ground import GroundSite, UniformEarthRotation
    from helioswarm.visibility import LightTimeSettings, solve_light_time

    c = SPEED_OF_LIGHT_M_S  # exact by SI definition (constants.py; no independent value exists)
    site = GroundSite.from_geodetic("eq", 0.0, 0.0, 0.0)  # at (a, 0, 0) with no rotation
    still = UniformEarthRotation()

    def traj(t: Any, r0: Any, v: Any = (0.0, 0.0, 0.0)) -> CartesianTrajectory:
        t = np.asarray(t, dtype=np.float64)
        r0, v = np.asarray(r0, dtype=np.float64), np.asarray(v, dtype=np.float64)
        return CartesianTrajectory(
            epochs_tdb_s=t, position_m=r0[None, :] + t[:, None] * v[None, :],
            velocity_m_s=np.broadcast_to(v, (t.size, 3)), frame="J2000", origin=SSB,
        )  # fmt: skip

    # (1) static geometry: tau = d / c.
    d = 1.234e11
    t = np.array([0.0, 1.0e8, 9.5e8])
    sat = traj(t, site.position_ef_m + d * np.array([0.6, 0.0, 0.8]))
    lt = solve_light_time(sat, site, ephemeris=_UniformEarth((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                          earth_orientation=still, settings=LightTimeSettings())  # fmt: skip
    err_static = float(np.max(np.abs(lt.light_time_s / (d / c) - 1.0)))
    # (2) site moving uniformly at ~Earth orbital speed: c tau = |D + v tau| with
    # D = r_site(t_e) - r_sat(t_e) -> (c^2 - v^2) tau^2 - 2 (D.v) tau - D^2 = 0, positive root.
    v = np.array([2.9e4, -1.1e4, 4.0e3])
    eph = _UniformEarth((1.0e9, -2.0e9, 3.0e8), (float(v[0]), float(v[1]), float(v[2])))
    t2 = np.array([0.0, 3.6e3, 4.0e8, 9.5e8])
    sat2 = traj(t2, (-8.0e10, 5.0e10, 2.0e10), (1.0e3, 2.0e4, -3.0e3))
    tol = 1e-12
    lt2 = solve_light_time(sat2, site, ephemeris=eph, earth_orientation=still,
                           settings=LightTimeSettings(tolerance_s=tol))  # fmt: skip
    dd = (eph.positions("EARTH", t2) + site.position_ef_m) - sat2.position_m
    dv = dd @ v
    d2 = np.einsum("ij,ij->i", dd, dd)
    vv = float(v @ v)
    exact = (dv + np.sqrt(dv * dv + (c * c - vv) * d2)) / (c * c - vv)
    err_moving = float(np.max(np.abs(lt2.light_time_s - exact)))
    inst = float(np.max(np.abs(np.sqrt(d2) / c - exact)))
    ok = err_static < 1e-14 and err_moving < tol + 2e-11 and inst > 1e-3
    return _Outcome(
        f"static: max|tau c/d - 1| = {err_static:.1e}; moving site: max|tau - tau_exact| = "
        f"{err_moving:.1e} s (instantaneous-geometry error {inst:.1e} s)",
        "tau = d/c (static); positive root of (c^2 - v^2) tau^2 - 2 (D.v) tau - D^2 = 0 "
        "(uniformly moving site, PDF §7.1/§7.4)",
        "1e-14 relative (position rounding); tolerance_s + 2e-11 s (t_a representation at "
        "1e9 s); the instantaneous distance must be wrong by > 1e-3 s (discrimination)",
        ok,
    )


# ---------------------------------------------------------------------------------------------
# §8.5 solar and thermal
# ---------------------------------------------------------------------------------------------


def _c_solar(ctx: ValidationContext) -> _Outcome:
    s1 = ctx.base_config.constants.solar_constant_1au_W_m2
    r = np.array([0.3, 0.5, 0.7, 1.0, 1.2]) * AU_M
    f = solar_flux_W_m2(r, s1)
    at1 = float(solar_flux_W_m2(np.array([AU_M]), s1)[0])
    inv_sq = float(np.max(np.abs(f * (r / AU_M) ** 2 / s1 - 1.0)))
    return _Outcome(
        f"S(1 au) = {at1!r} W/m^2; max|S r^2/(S1 au^2) - 1| = {inv_sq:.1e}",
        f"S_1AU = {s1} W/m^2 (IAU 2015 B3), 1/r^2",
        "exact at 1 au; 1e-14 (rounding)",
        at1 == s1 and inv_sq < 1e-14,
    )


def _c_zero_incidence(ctx: ValidationContext) -> _Outcome:
    from helioswarm.solar import collector_model_from_config, solar_power

    col = collector_model_from_config(ctx.base_config)
    r = np.array([0.3, 0.7, 1.0, 1.2]) * AU_M
    worst = 0.0
    lit = solar_power(r, 1.0, 1.0, 300.0, col)
    for cos_i, frac in ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0)):
        res = solar_power(r, cos_i, frac, 300.0, col)
        for attr in ("incident_W", "absorbed_W", "pv_electric_W", "pv_heat_W", "reflected_W"):
            worst = max(worst, float(np.max(np.abs(getattr(res, attr)))))
    ok = worst == 0.0 and bool(np.all(lit.pv_electric_W > 0.0))
    return _Outcome(
        f"max |power| at cos(inc) = 0 or umbra = {worst} W (lit: P_PV(1 au) = "
        f"{float(lit.pv_electric_W[2]):.6g} W)",
        "0 W: incident = S A cos(inc) f (PDF §8.1)",
        "exact",
        ok,
    )


def _c_eclipse_and_ledger(ctx: ValidationContext) -> _Outcome:
    models = power_thermal_models_from_config(ctx.base_config)
    r = np.array([0.3, 0.5, 0.7, 1.0, 1.2]) * AU_M
    pt = power_thermal_vs_distance(r, models)
    dark = power_thermal_vs_distance(r, models, illumination_fraction=0.0)
    zero = float(np.max(dark.solar.pv_electric_W))
    led = pt.ledger
    return _Outcome(
        f"ledger max relative residual {led.max_relative_residual:.2e}; P_PV in umbra {zero}",
        "energy closure incident = reflected + PV heat radiated + radiator + P_aperture; 0 W "
        "in eclipse",
        f"{led.rtol:.0e} (closure tolerance); exact 0",
        bool(led.closed) and zero == 0.0,
    )


def _c_radiator(ctx: ValidationContext) -> _Outcome:
    base = RadiatorModel(
        emissivity=0.9, temperature_limit_K=450.0, radiating_sides=2, view_factor=1.0,
        space_temperature_K=CMB_TEMPERATURE_K,
    )  # fmt: skip
    q = np.array([1.0e6])
    a0 = float(radiator_required_area_m2(q, base)[0])
    a_eps = float(radiator_required_area_m2(q, dataclasses.replace(base, emissivity=0.45))[0])
    a_t = float(
        radiator_required_area_m2(q, dataclasses.replace(base, temperature_limit_K=900.0))[0]
    )
    # Independent sigma from the exact SI defining constants (2019 SI): k_B, h, c.
    k_b, h_pl = 1.380649e-23, 6.62607015e-34
    sigma = 2.0 * math.pi**5 * k_b**4 / (15.0 * h_pl**3 * SPEED_OF_LIGHT_M_S**2)
    closed = 1.0e6 / (0.9 * sigma * 2 * (450.0**4 - CMB_TEMPERATURE_K**4))
    ratio_t = a0 / a_t
    ratio_t_ref = (900.0**4 - CMB_TEMPERATURE_K**4) / (450.0**4 - CMB_TEMPERATURE_K**4)
    err_abs = abs(a0 / closed - 1)
    err_ratio = max(abs(a_eps / a0 / 2.0 - 1), abs(ratio_t / ratio_t_ref - 1))
    return _Outcome(
        f"A(1 MW) = {a0:.6g} m^2; A(eps/2)/A = {a_eps / a0:.12g}; A(T)/A(2T) = {ratio_t:.10g}",
        f"Q/(eps sigma n F (T^4-Ts^4)) = {closed:.6g} m^2 (sigma = 2 pi^5 k^4/(15 h^3 c^2), "
        f"exact SI constants); 2; {ratio_t_ref:.10g}",
        "1e-10 on the area (the package sigma is CODATA's 10-digit rounding of the exact value, "
        "3e-11 apart); 1e-12 on the ratios",
        err_abs < 1e-10 and err_ratio < 1e-12,
    )


# ---------------------------------------------------------------------------------------------
# §9 beam
# ---------------------------------------------------------------------------------------------

_D, _LAM, _R = 10.0, 1.0e-6, 1.7e11


def _airy(d: float = _D, lam: float = _LAM, r: float = _R) -> AiryPattern:
    return AiryPattern(aperture_diameter_m=d, wavelength_m=lam, range_m=r, max_fresnel_number=0.5)


def _c_first_null(_: ValidationContext) -> _Outcome:
    worst = 0.0
    for d, lam, r in ((10.0, 1e-6, 1.7e11), (3.0, 5e-7, 1e10), (1000.0, 0.1224, 1.5e11)):
        p = _airy(d, lam, r)
        rho0 = AIRY_FIRST_NULL_FACTOR * lam * r / d
        worst = max(
            worst, float(p.intensity_m2_inv(np.array([rho0]))[0] / p.peak_intensity_m2_inv())
        )
    return _Outcome(
        f"max I(1.21967 lambda R/D)/I0 = {worst:.1e} (3 cases)",
        "0: first zero of J1 at x = 3.8317059702 (Abramowitz & Stegun Table 9.5)",
        "1e-25 relative (float64 rounding of x)",
        worst < 1e-25,
    )


def _c_ee_ring_and_power(_: ValidationContext) -> _Outcome:
    p = _airy()
    ell = _LAM * _R / _D
    ee_ring = float(p.encircled_energy(np.array([J1_FIRST_ZERO / math.pi * ell]))[0])
    ref_ring = 1.0 - special.j0(J1_FIRST_ZERO) ** 2
    x_max = 2000.0
    val, _ = integrate.quad(
        lambda x: (2 * special.j1(x) / x) ** 2 * x / 2.0, 1e-12, x_max, limit=4000
    )
    total = val + 2.0 / (math.pi * x_max)
    ok = abs(ee_ring - ref_ring) < 1e-12 and abs(total - 1.0) < 1e-5
    return _Outcome(
        f"EE(first dark ring) = {ee_ring:.12f}; quad power to x=2000 + tail = {total:.8f}",
        f"1 - J0(j11)^2 = {ref_ring:.12f} (Rayleigh); total 1 (Parseval)",
        "1e-12; 1e-5 (quad tolerance and O(x^-2) tail)",
        ok,
    )


def _airy_offset_capture_dblquad(
    d_ap: float, lam: float, r: float, a: float, offset: float
) -> float:
    """Independent reference: capture of a disk of radius ``a`` whose centre is ``offset`` from
    the Airy axis, by nested ``scipy.integrate.quad`` in receiver-centred polar coordinates,
    ``eta = int_0^a int_0^{2 pi} I(sqrt(s^2 + d^2 + 2 s d cos phi)) s dphi ds`` with the unit-
    power Airy irradiance ``I(rho) = pi D^2/(4 lambda^2 R^2) (2 J1(k rho)/(k rho))^2``,
    ``k = pi D/(lambda R)`` (Born & Wolf §8.5.2). Shares no code with the package's 1-D
    band/arc formula or its Cartesian grid."""
    k = math.pi * d_ap / (lam * r)
    i0 = math.pi * d_ap * d_ap / (4.0 * lam * lam * r * r)

    def irr(rho: float) -> float:
        x = k * rho
        return i0 * (1.0 if x == 0.0 else (2.0 * float(special.j1(x)) / x) ** 2)

    def ring(sr: float) -> float:
        val, _ = integrate.quad(
            lambda ph: irr(math.sqrt(sr * sr + offset * offset + 2 * sr * offset * math.cos(ph))),
            0.0, math.pi, epsabs=0.0, epsrel=1e-13, limit=200,
        )  # fmt: skip
        return 2.0 * float(val) * sr

    total, _ = integrate.quad(ring, 0.0, a, epsabs=0.0, epsrel=1e-12, limit=200)
    return float(total)


def _c_capture_grid(_: ValidationContext) -> _Outcome:
    p = _airy()
    ell = _LAM * _R / _D
    a, d = 0.6 * ell, 0.5 * ell
    exact = float(capture_fraction(p, a, np.array([d]))[0])
    ref = _airy_offset_capture_dblquad(_D, _LAM, _R, a, d)
    ns = (25, 100, 200, 400)
    errs = [capture_fraction_cartesian(p, a, d, n) - exact for n in ns]
    rel = abs(exact / ref - 1.0)
    # The anti-aliased midpoint grid converges, but not monotonically: which edge sub-cells
    # are counted inside changes erratically with n (lattice-point error), so the sequence
    # changes sign and stalls (measured +1.1e-4, -1.4e-6, +3.0e-6, +1.4e-6). It is reported as
    # supporting evidence with a ceiling, not as the pass criterion.
    grid_ok = all(abs(e) < 2e-5 for e in errs[1:]) and abs(errs[0]) > max(abs(e) for e in errs[1:])
    return _Outcome(
        f"1-D eta = {exact:.12f}; nested-quad eta = {ref:.12f} (rel diff {rel:.1e}); "
        "2-D grid - 1-D at n="
        + "/".join(str(n) for n in ns)
        + ": "
        + "/".join(f"{e:+.1e}" for e in errs),
        "independent nested scipy quad over the offset disk (receiver-centred polar "
        "coordinates); Cartesian anti-aliased midpoint grid as supporting evidence",
        "1e-9 relative vs nested quad (quadrature tolerances 1e-12/1e-13); grid |error| < 2e-5 "
        "for n >= 100 and below the n = 25 error (empirical ceiling, edge error O(cell), not "
        "monotone in n)",
        rel < 1e-9 and grid_ok,
    )


def _c_pointing_monotone(_: ValidationContext) -> _Outcome:
    p = _airy()
    ell = _LAM * _R / _D
    a = 0.07 * ell
    sig = np.array([0.0, 0.05, 0.1, 0.2, 0.4]) * ell
    e = [capture_statistics(p, a, 0.0, float(s)).expected for s in sig]
    ok = all(e[i + 1] <= e[i] for i in range(len(e) - 1)) and e[-1] < e[0]
    return _Outcome(
        "E[eta] = " + ", ".join(f"{v:.6e}" for v in e) + " for sigma/(lambda R/D) = 0..0.4",
        "non-increasing in sigma for the centred baseline (Anderson 1955, unimodal beam)",
        "strict ordering",
        ok,
    )


def _c_lambda_aperture(_: ValidationContext) -> _Outcome:
    a = 1000.0
    base = float(_airy().encircled_energy(np.array([a]))[0])
    half_lam = float(_airy(lam=_LAM / 2).encircled_energy(np.array([a]))[0])
    two_d = float(_airy(d=2 * _D).encircled_energy(np.array([a]))[0])
    x = math.pi * _D * a / (_LAM * _R)
    ref = 1 - special.j0(2 * x) ** 2 - special.j1(2 * x) ** 2
    ok = (
        half_lam > base
        and two_d > base
        and abs(half_lam - ref) < 1e-12
        and abs(two_d - ref) < 1e-12
    )
    return _Outcome(
        f"EE: base {base:.8f}, lambda/2 {half_lam:.8f}, 2D {two_d:.8f}",
        f"both = EE(2x) = {ref:.8f} > EE(x) (x = pi D a/(lambda R))",
        "1e-12; strict improvement",
        ok,
    )


def _c_impossible_beam(_: ValidationContext) -> _Outcome:
    lam = SPEED_OF_LIGHT_M_S / 2.45e9
    p = _airy(1.0, lam, AU_M)
    a = 1000.0
    eta = float(p.encircled_energy(np.array([a]))[0])
    x = math.pi * 1.0 * a / (lam * AU_M)
    ref = x * x / 4.0
    return _Outcome(
        f"capture = {eta:.6e}",
        f"x^2/4 = {ref:.6e} (small-x limit of 1 - J0^2 - J1^2), 1 m aperture at 2.45 GHz, 1 au, "
        "1 km receiver",
        "1e-9 relative",
        abs(eta / ref - 1.0) < 1e-9 and eta < 1e-13,
    )


def _c_fresnel_to_airy(_: ValidationContext) -> _Outcome:
    ap = ApertureField("uniform_circular", _D)
    devs = []
    for nf in (1e-2, 1e-3):
        r = _D * _D / (_LAM * nf)
        fr = FresnelHankelPattern(
            aperture=ap, wavelength_m=_LAM, range_m=r, paraxial_rho_max_m=1.0, refinement=1.0
        )
        airy = AiryPattern(
            aperture_diameter_m=_D, wavelength_m=_LAM, range_m=r, max_fresnel_number=0.5
        )
        dev = abs(float(fr.intensity_m2_inv(np.array([0.0]))[0]) / airy.peak_intensity_m2_inv() - 1)
        pred = (math.pi * nf) ** 2 / 192.0
        devs.append((nf, dev, pred))
    ok = all(0.5 * pred <= dev <= 1.5 * pred for _, dev, pred in devs)
    return _Outcome(
        "; ".join(f"N_F={nf:g}: dev {dev:.3e}" for nf, dev, _ in devs),
        "; ".join(f"(pi N_F)^2/192 = {pred:.3e}" for _, _, pred in devs)
        + " (on-axis sinc expansion of sin^2(pi N_F/8))",
        "[0.5, 1.5] x prediction",
        ok,
    )


def _c_batch_kernel(_: ValidationContext) -> _Outcome:
    from helioswarm.capture_batch import airy_capture_batch

    rng = np.random.default_rng(20261112)
    n = 24
    r = rng.uniform(0.5, 1.5, n) * 1.5e11
    a = rng.uniform(100.0, 3000.0, n)
    nu = rng.choice([0.0, 2e-9, 5e-8], n) * r
    s = rng.choice([0.0, 1e-9, 2e-8], n) * r
    k = math.pi * _D / (_LAM * r)
    res = airy_capture_batch(k, a, nu, s, backend="numpy")
    err = 0.0
    for i in range(n):
        st = capture_statistics(_airy(r=float(r[i])), float(a[i]), float(nu[i]), float(s[i]))
        err = max(err, abs(st.expected - res.expected[i]), abs(st.p05 - res.p05[i]))
    jax_note = "jax not installed"
    jerr: float | None = None
    try:
        rj = airy_capture_batch(k, a, nu, s, backend="jax")
        jerr = float(np.max(np.abs(rj.expected - res.expected)))
        jax_note = f"jax vs numpy batch max diff {jerr:.1e}"
    except ImportError:
        pass
    ok = err < 1e-9 and (jerr is None or jerr < 1e-12)
    return _Outcome(
        f"batch(numpy) vs M5 per-point max diff {err:.1e}; {jax_note}",
        "helioswarm.capture.capture_statistics (adaptive M5 reference)",
        "1e-9 (quadrature); jax vs numpy 1e-12 (same algorithm, trapezoid Bessel)",
        ok,
    )


# ---------------------------------------------------------------------------------------------
# §13.1 link budget
# ---------------------------------------------------------------------------------------------


def _c_link_budget(_: ValidationContext) -> _Outcome:
    rng = np.random.default_rng(7)
    n = 10_000
    eff = OnboardEfficiencies(0.95, 0.5, 0.9)
    rx = ReceiverModel("laser_photovoltaic", 0.4, 0.97, max_input_power_W=5.0e4)
    lb = evaluate_link_budget(
        rng.uniform(0, 4e6, n),
        eff,
        rx,
        link_enabled=rng.random(n) > 0.3,
        eta_atmosphere=rng.random(n),
        eta_capture=rng.random(n),
        peak_irradiance_per_W_m2=rng.uniform(0, 1e-6, n),
    )
    v = link_budget_violations(lb)
    clipped = lb.captured_W > rx.max_input_power_W
    clip_ok = bool(np.all(lb.accepted_W[clipped] == rx.max_input_power_W)) and clipped.any()
    chain = np.asarray(lb.grid_W) / np.where(lb.captured_W > 0, lb.captured_W, 1.0)
    ok = bool(not v and clip_ok and float(np.max(chain)) <= 0.4 * 0.97 + 1e-15)
    stages = " >= ".join(s for s, _ in STAGE_COLUMNS)
    return _Outcome(
        f"{len(v)} invariant violations over {n} random epochs; {int(clipped.sum())} clipped",
        f"{stages}; every eta in [0, 1]; P_accepted = max_input when clipped",
        "exact (monotone float multiplication)",
        ok,
    )


def _c_srp(ctx: ValidationContext) -> _Outcome:
    from helioswarm.srp import SrpModel

    c = SPEED_OF_LIGHT_M_S  # exact by SI definition (constants.py; no independent value exists)
    s1 = ctx.base_config.constants.solar_constant_1au_W_m2
    cr, am = 1.3, 0.02
    m = SrpModel.from_solar_constant(cr=cr, area_to_mass_m2_kg=am, solar_constant_1au_W_m2=s1)
    sun = np.array([7.0e8, -3.0e8, 1.0e8])  # the Sun is NOT at the origin
    u = np.array([0.6, -0.8, 0.0])
    a1 = m.acceleration_m_s2(sun + AU_M * u, sun)
    mag_err = abs(float(np.linalg.norm(a1)) / (s1 / c * cr * am) - 1.0)
    dir_err = float(np.max(np.abs(a1 / np.linalg.norm(a1) - u)))
    inv = 0.0
    for r_au in (0.3, 0.5, 2.0, 5.2):
        ar = m.acceleration_m_s2(sun + r_au * AU_M * u, sun)
        inv = max(inv, abs(float(np.linalg.norm(ar) / np.linalg.norm(a1)) * r_au**2 - 1.0))
    ok = mag_err < 1e-14 and dir_err < 1e-15 and inv < 1e-14
    return _Outcome(
        f"|a(1 au)| rel err {mag_err:.1e}; direction err {dir_err:.1e}; max|a(r) r^2/a(1 au) - "
        f"1| = {inv:.1e} (0.3-5.2 au, Sun offset from the origin)",
        "a = (S_1AU/c) Cr (A/m) (au/r)^2 along (r - r_sun)/|r - r_sun| (anti-sunward; PDF §6.3, "
        "§13.1)",
        "1e-14 relative / 1e-15 (float64 rounding)",
        ok,
    )


def _tdb_minus_utc_independent_s(utc: tuple[int, int, int, int], tai_minus_utc_s: int) -> float:
    """TDB - UTC: TAI - UTC (IERS Bulletin C) + 32.184 s (TT - TAI, IAU 1991) + the USNO
    Circular 179 (Kaplan 2005, eq. 2.6) two-term TDB - TT series."""
    from datetime import datetime

    dt = datetime(*utc)
    cal = (dt - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
    tt = cal + tai_minus_utc_s + 32.184
    g = math.radians(357.53 + 0.98560028 * (tt / SECONDS_PER_DAY))
    return tt + 0.001657 * math.sin(g) + 0.000014 * math.sin(2.0 * g)


def _c_time_scale(ctx: ValidationContext) -> _Outcome:
    from helioswarm.timescale import utc_to_tdb_s

    cases = (
        ("2000-01-01T12:00:00", (2000, 1, 1, 12), 32),
        ("2017-01-01T00:00:00", (2017, 1, 1, 0), 37),
        ("2030-01-01T00:00:00", (2030, 1, 1, 0), 37),
    )
    with ctx.kernel_set.loaded(verify=False) as k:
        errs = [abs(utc_to_tdb_s(u, k) - _tdb_minus_utc_independent_s(t, n)) for u, t, n in cases]
    return _Outcome(
        f"max |TDB_spice - TDB_ref| = {max(errs):.1e} s at {', '.join(u for u, _, _ in cases)}",
        "TAI-UTC from IERS Bulletin C (32 s from 1999, 37 s from 2017; none announced since) + "
        "32.184 s + USNO Circular 179 TDB-TT; MVP epoch 2030-01-01 UTC",
        "1e-5 s (two-term TDB-TT series; a wrong leap-second count fails by >= 1 s, a dropped "
        "TDB-TT term by up to 1.7 ms)",
        max(errs) < 1e-5,
    )


# ---------------------------------------------------------------------------------------------
# §13.2 end-to-end integration (short SPICE runs)
# ---------------------------------------------------------------------------------------------


def _short(ctx: ValidationContext, days: float = 2.0, **paths: Any) -> RunConfig:
    data = copy.deepcopy(ctx.base_config.model_dump(mode="json"))
    data["time"]["duration_days"] = days
    for key, value in paths.items():
        node = data
        parts = key.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return parse_config(data, source="validate-models variant")


def _run(ctx: ValidationContext, cfg: RunConfig) -> SystemRunResult:
    with ctx.kernel_set.loaded(verify=False) as k:
        return run_single_satellite(
            cfg, ephemeris=SpiceEphemeris(k), kernels=k, rng=np.random.default_rng(cfg.project.seed)
        )


def _c_e2e_baseline(ctx: ValidationContext) -> _Outcome:
    from helioswarm.io import execute_run, prepare_run_dir

    cfg = _short(ctx)
    with tempfile.TemporaryDirectory(prefix="helioswarm-validate-") as tmp:
        run_dir = prepare_run_dir(Path(tmp) / "run", overwrite=False)
        out = execute_run(cfg, run_dir, kernel_set=ctx.kernel_set, command=["validate-models"])
        files = sorted(p.name for p in run_dir.iterdir())
        figs = len(out.figures)
    need = {"config_resolved.yaml", "manifest.json", "timeseries.parquet", "summary.json",
            "figures", "logs", "report.md"}  # fmt: skip
    ok = need.issubset(files) and figs == 6
    return _Outcome(
        f"outputs {files}; {figs} figures; mean P_grid "
        f"{out.summary['p_grid_W']['time_mean']:.4g} W",
        "config -> resolved config, manifest, timeseries, summary, figures, report, log",
        "all present",
        ok,
    )


def _c_e2e_unavailable_zero(ctx: ValidationContext) -> _Outcome:
    res = _run(ctx, _short(ctx))
    off = ~res.available
    zero = bool(np.all(res.link.grid_W[off] == 0.0))
    occ = res.access.earth_blocked
    cloudy = _run(ctx, _short(ctx, models__weather_scenario="cloudy"))
    ok = bool(zero and occ.any() and np.all(cloudy.link.grid_W == 0.0))
    return _Outcome(
        f"P_grid at {int(off.sum())} unavailable epochs (incl. {int(occ.sum())} Earth-occulted): "
        f"max {float(np.max(res.link.grid_W[off]))} W; cloudy optical outage max "
        f"{float(np.max(cloudy.link.grid_W))} W",
        "0 W",
        "exact",
        ok,
    )


def _c_e2e_clip(ctx: ValidationContext) -> _Outcome:
    base = _run(ctx, _short(ctx))
    cap_max = float(np.max(base.link.captured_W))
    limit = 0.5 * cap_max
    res = _run(ctx, _short(ctx, receiver__max_input_power_W=limit))
    sat = res.link.receiver_saturated
    expect = limit * res.receiver.conversion_efficiency * res.receiver.grid_efficiency
    err = float(np.max(np.abs(res.link.grid_W[sat] - expect))) if sat.any() else math.inf
    unclipped_same = bool(np.array_equal(res.link.grid_W[~sat], base.link.grid_W[~sat]))
    return _Outcome(
        f"{int(sat.sum())} clipped epochs; |P_grid - limit*eta_rx*eta_grid| max {err:.1e} W",
        f"limit {limit:.6g} W x 0.4 x 0.97 at clipped epochs; unchanged elsewhere",
        "1e-9 W absolute",
        err < 1e-9 and unclipped_same,
    )


def _c_e2e_area_linear(ctx: ValidationContext) -> _Outcome:
    a = _run(ctx, _short(ctx))
    b = _run(ctx, _short(ctx, collector__area_m2=2 * ctx.base_config.collector.area_m2))
    pv = float(
        np.max(
            np.abs(b.power_thermal.solar.pv_electric_W / a.power_thermal.solar.pv_electric_W - 2)
        )
    )
    on = a.available & (a.link.grid_W > 0)
    grid = float(np.max(np.abs(b.link.grid_W[on] / a.link.grid_W[on] - 2)))
    return _Outcome(
        f"max|P_PV(2A)/P_PV(A) - 2| = {pv:.1e}; max|P_grid ratio - 2| = {grid:.1e}",
        "2 (PV temperature independent of area: absorbed and radiated both scale with A)",
        "1e-12 relative",
        pv < 1e-12 and grid < 1e-12,
    )


def _c_e2e_lambda_aperture(ctx: ValidationContext) -> _Outcome:
    a = _run(ctx, _short(ctx))
    b = _run(
        ctx, _short(ctx, transmitter__wavelength_m=0.5 * ctx.base_config.transmitter.wavelength_m)
    )
    c = _run(
        ctx,
        _short(
            ctx,
            transmitter__aperture_diameter_m=2 * ctx.base_config.transmitter.aperture_diameter_m,
        ),
    )
    m = a.available
    ok = bool(np.all(b.link.eta_capture[m] > a.link.eta_capture[m])) and bool(
        np.all(c.link.eta_capture[m] > a.link.eta_capture[m])
    )
    return _Outcome(
        f"median capture: base {np.median(a.link.eta_capture[m]):.4e}, lambda/2 "
        f"{np.median(b.link.eta_capture[m]):.4e}, 2D {np.median(c.link.eta_capture[m]):.4e}",
        "smaller wavelength and larger aperture improve capture at every available epoch",
        "strict",
        ok,
    )


def _c_e2e_pointing(ctx: ValidationContext) -> _Outcome:
    a = _run(ctx, _short(ctx))
    b = _run(
        ctx,
        _short(
            ctx, transmitter__pointing_sigma_rad=10 * ctx.base_config.transmitter.pointing_sigma_rad
        ),
    )
    m = a.available
    ok = bool(np.all(b.link.eta_capture[m] <= a.link.eta_capture[m]))
    return _Outcome(
        f"median expected capture sigma x1 {np.median(a.link.eta_capture[m]):.6e}, x10 "
        f"{np.median(b.link.eta_capture[m]):.6e}",
        "larger pointing sigma cannot improve expected capture (centred baseline)",
        "<= at every available epoch",
        ok,
    )


def _c_e2e_reproducible(ctx: ValidationContext) -> _Outcome:
    from helioswarm.io import execute_run, prepare_run_dir

    cfg = _short(ctx)
    with tempfile.TemporaryDirectory(prefix="helioswarm-validate-") as tmp:
        o1 = execute_run(cfg, prepare_run_dir(Path(tmp) / "a", overwrite=False),
                         kernel_set=ctx.kernel_set, make_figures=False)  # fmt: skip
        o2 = execute_run(cfg, prepare_run_dir(Path(tmp) / "b", overwrite=False),
                         kernel_set=ctx.kernel_set, make_figures=False)  # fmt: skip
        same_summary = (Path(tmp) / "a" / "summary.json").read_bytes() == (
            Path(tmp) / "b" / "summary.json"
        ).read_bytes()
    ok = same_summary and o1.timeseries_content_sha256 == o2.timeseries_content_sha256
    return _Outcome(
        f"summary.json identical: {same_summary}; content hash {o1.timeseries_content_sha256[:16]} "
        f"vs {o2.timeseries_content_sha256[:16]}",
        "identical outputs for the same config and seed (timestamps only in manifest.json)",
        "bitwise",
        ok,
    )


def _c_e2e_energy_accounting(ctx: ValidationContext) -> _Outcome:
    res = _run(ctx, _short(ctx))
    v = link_budget_violations(res.link)
    inc = res.power_thermal.solar.incident_W
    ok = not v and bool(np.all(res.link.grid_W <= inc))
    return _Outcome(
        f"{len(v)} violations; max P_grid/P_incident {float(np.max(res.link.grid_W / inc)):.3e}",
        "P_grid <= every upstream stage <= incident sunlight",
        "exact",
        ok,
    )


def _c_e2e_microwave_impossible(ctx: ValidationContext) -> _Outcome:
    lam = SPEED_OF_LIGHT_M_S / 2.45e9
    cfg = _short(
        ctx,
        transmitter__mode="microwave",
        transmitter__wavelength_m=lam,
        transmitter__aperture_diameter_m=1.0,
        transmitter__pointing_sigma_rad=1e-6,
        transmitter__aperture_illumination="uniform_circular",
        receiver__technology="rectenna",
        models__beam="aperture_far_field",
    )
    res = _run(ctx, cfg)
    m = res.available
    a = cfg.receiver.radius_m
    x = math.pi * 1.0 * a * np.sqrt(np.sin(res.access.elevation_rad[m]))
    x = x / (lam * res.access.range_m[m])
    ref = x * x / 4.0
    ratio = res.link.eta_capture[m] / ref
    dev = float(np.max(np.abs(ratio - 1.0))) if m.any() else math.inf
    frac = float(np.max(res.link.grid_W / np.maximum(res.link.aperture_W, 1e-300)))
    return _Outcome(
        f"{int(m.sum())} available epochs: max|eta_capture/(x^2/4) - 1| = {dev:.1e}; "
        f"max P_grid/P_aperture = {frac:.3e}",
        "x^2/4, x = pi D a sqrt(sin el)/(lambda R) (small-x limit of 1 - J0^2 - J1^2, uniform "
        "1 m aperture, 2.45 GHz, 1 km horizontal receiver, ~1.15 au); P_grid/P_aperture "
        "<= x^2/4 * 0.4 * 0.97 ~ 1e-14",
        "1e-6 relative (O(x^2) ~ 1e-15 and jitter (sigma D/lambda)^2 ~ 7e-11 corrections; "
        "quadrature 1e-9); P_grid/P_aperture < 1e-12",
        bool(m.any()) and dev < 1e-6 and frac < 1e-12,
    )


# ---------------------------------------------------------------------------------------------
# §13.3 independent checks and convergence
# ---------------------------------------------------------------------------------------------


def _c_hcipy(_: ValidationContext) -> _Outcome:
    try:
        import hcipy
    except ImportError:
        return _Outcome("not run", "HCIPy", "-", None, "hcipy not installed (xcheck extra)")
    n_pupil = 512
    pupil = hcipy.make_pupil_grid(n_pupil, _D)
    mask = np.asarray(hcipy.evaluate_supersampled(hcipy.make_circular_aperture(_D), pupil, 8))
    field = hcipy.Field(mask.astype(complex), pupil)
    theta = np.array([0.0, 0.3, 0.7, 1.0, 1.5]) * _LAM / _D
    out = hcipy.CartesianGrid(
        hcipy.UnstructuredCoords([theta, np.zeros_like(theta)]), weights=np.ones_like(theta)
    )
    prop = hcipy.FraunhoferPropagator(pupil, out, focal_length=1.0)
    wf = hcipy.Wavefront(field, wavelength=_LAM)
    j_h = np.abs(np.asarray(prop(wf).electric_field)) ** 2 / float(wf.total_power)
    j_ours = ApertureField("uniform_circular", _D).radiant_intensity_sr_inv(theta, _LAM)
    err = float(np.max(np.abs(j_h - j_ours)) / j_ours[0])
    return _Outcome(
        f"max |J_hcipy - J_ours| / J(0) = {err:.1e}; J(0) = {j_h[0]:.6e} sr^-1",
        f"HCIPy {hcipy.__version__} Fraunhofer propagator, uniform 10 m pupil, 1 um; "
        f"pi a^2/lambda^2 = {math.pi * 25 / _LAM**2:.6e}",
        "4e-3 of the peak (512-px pupil pixelisation)",
        err < 4e-3,
    )


def _c_itur(_: ValidationContext) -> _Outcome:
    try:
        import itur.models as itu_models
    except ImportError:
        return _Outcome("not run", "ITU-R", "-", None, "itur not installed (itur extra)")
    from helioswarm.atmosphere import ITU_R_VERSIONS, itu_r_slant_path_attenuation_dB

    versions = dict(ITU_R_VERSIONS) | {"itu676": 11, "itu1511": 1}
    r001 = float(itu_models.itu837.rainfall_rate(51.5, -0.14, 0.01000000001).value)
    prev = itu_models.itu1511.get_version()
    try:
        itu_models.itu1511.change_version(1)
        hs_km = float(itu_models.itu1511.topographic_altitude(51.5, -0.14).value)
    finally:
        itu_models.itu1511.change_version(prev)
    worst = 0.0
    for f_hz, p, ref in (
        (14.25e9, 1.0, (0.223693782, 0.45517046, 0.48914539)),
        (29.0e9, 0.01, (0.799999368, 1.77247154, 23.19077435)),
    ):
        res = itu_r_slant_path_attenuation_dB(
            51.5, -0.14, hs_km * 1000.0, f_hz, np.radians([31.07694309]), p,
            versions=versions, rainfall_rate_001_mm_h=r001, polarization_tilt_deg=0.0,
        )  # fmt: skip
        got = (float(res.gas_dB[0]), float(res.cloud_dB[0]), float(res.rain_dB[0]))
        worst = max(worst, *(abs(g - r) for g, r in zip(got, ref, strict=True)))
    return _Outcome(
        f"max |A - A_ref| (gas, cloud, rain; 2 cases) = {worst:.1e} dB",
        "ITU-R SG3 validation examples of P.618-13 (London; via itur 0.4.0 test file), "
        "P.676-11/P.1511-1 pinned",
        "1e-5 dB (validation-file precision); production P.676-12 is regression-pinned only",
        worst < 1e-5,
    )


def _c_l1_step_convergence(ctx: ValidationContext) -> _Outcome:
    from helioswarm.propagate import initial_state_from_config, integrator_settings_from_config
    from helioswarm.timescale import time_grid_from_config

    cfg = _short(ctx, days=30.0)
    with ctx.kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        grid = time_grid_from_config(cfg.time, k)
        init = initial_state_from_config(cfg, eph, k)
        base = integrator_settings_from_config(cfg.time)
        p1 = propagate_l1(init, grid.epochs_tdb_s, ephemeris=eph, settings=base)
        p2 = propagate_l1(
            init,
            grid.epochs_tdb_s,
            ephemeris=eph,
            settings=dataclasses.replace(base, max_step_s=SECONDS_PER_DAY),
        )
        p3 = propagate_l1(
            init,
            grid.epochs_tdb_s,
            ephemeris=eph,
            settings=dataclasses.replace(base, rtol=1e-13),
        )
    d_step = float(
        np.max(np.linalg.norm(p1.trajectory.position_m - p2.trajectory.position_m, axis=1))
    )
    d_tol = float(
        np.max(np.linalg.norm(p1.trajectory.position_m - p3.trajectory.position_m, axis=1))
    )
    return _Outcome(
        f"30 d MVP orbit: max step 300 s vs 1 d: {d_step:.2e} m; rtol 1e-12 vs 1e-13: "
        f"{d_tol:.2e} m",
        "converged trajectory (differences far below the 1e3-1e4 m beam scale)",
        "1 m",
        d_step < 1.0 and d_tol < 1.0,
    )


def _c_capture_refinement(_: ValidationContext) -> _Outcome:
    p = _airy()
    ell = _LAM * _R / _D
    s1 = capture_statistics(p, 0.8 * ell, 0.1 * ell, 0.2 * ell)
    s2 = capture_statistics(p, 0.8 * ell, 0.1 * ell, 0.2 * ell, refinement=2.0)
    d = abs(s1.expected - s2.expected)
    return _Outcome(
        f"|E[eta](refinement 2) - E[eta](1)| = {d:.1e}",
        "quadrature converged",
        "1e-12",
        d < 1e-12,
    )


def _c_diurnal_sampling(ctx: ValidationContext) -> _Outcome:
    hourly = _run(ctx, _short(ctx, days=30.0))
    fine = _run(ctx, _short(ctx, days=30.0, time__output_step_s=300.0))
    # 3593 s (prime) is incommensurate with the day: 720 steps = 29.94 d, all phases distinct.
    odd = _run(ctx, _short(ctx, days=3593.0 * 720 / SECONDS_PER_DAY, time__output_step_s=3593.0))
    site = ctx.base_config.receiver.sites[0]
    lat = math.radians(site.latitude_deg)
    el_min = math.radians(ctx.base_config.receiver.min_elevation_deg)
    # Declination of the satellite seen from the site (J2000 z of the site->satellite direction).
    dec = np.arcsin(np.clip(-fine.access.transmit_direction_unit[:, 2], -1.0, 1.0))
    cos_h0 = (math.sin(el_min) - math.sin(lat) * np.sin(dec)) / (math.cos(lat) * np.cos(dec))
    analytic = float(np.mean(np.arccos(np.clip(cos_h0, -1.0, 1.0)) / math.pi))
    fa = [float(np.mean(r.access.visible)) for r in (hourly, fine, odd)]
    gaps = [r.access_summary.hour_angle_max_gap_rad for r in (hourly, fine, odd)]
    return _Outcome(
        f"access fraction 3600 s {fa[0]:.4f} (gap {math.degrees(gaps[0]):.2f} deg), 300 s "
        f"{fa[1]:.4f} ({math.degrees(gaps[1]):.2f} deg), 3593 s {fa[2]:.4f} "
        f"({math.degrees(gaps[2]):.2f} deg)",
        f"analytic diurnal fraction acos((sin el_min - sin lat sin dec)/(cos lat cos dec))/pi = "
        f"{analytic:.4f} (rotation and declination from the 300 s run; no refraction)",
        "|300 s - analytic| < 0.005 and |3593 s - analytic| < 0.01; |3600 s - analytic| <= "
        "gap/2pi (aliasing quantified)",
        abs(fa[1] - analytic) < 0.005
        and abs(fa[2] - analytic) < 0.01
        and abs(fa[0] - analytic) <= gaps[0] / (2 * math.pi),
        "hourly steps alias with the ~24 h synodic day (M3); reported, not hidden",
    )


def _c_reference_link(ctx: ValidationContext) -> _Outcome:
    from scipy import stats

    res = _run(ctx, _short(ctx))
    ref = res.reference
    cfg = res.config
    tx = cfg.transmitter
    cond_ok = (
        ref.range_m == float(np.mean(res.access.range_m))
        and ref.sun_distance_m == float(np.mean(res.illumination.sun_distance_m))
        and ref.elevation_rad == 0.5 * math.pi
        and tx.pointing_bias_rad == 0.0
    )
    # Expected capture of the centred disk with Gaussian jitter: E[eta] = int I(rho)
    # F_ncx2(a^2/s^2; 2, rho^2/s^2) 2 pi rho drho, s = sigma R (one 1-D quadrature,
    # independent of the M5 offset/Rice quadratures).
    d_ap, lam, rr, a = tx.aperture_diameter_m, tx.wavelength_m, ref.range_m, cfg.receiver.radius_m
    k = math.pi * d_ap / (lam * rr)
    i0 = math.pi * d_ap**2 / (4.0 * lam**2 * rr**2)
    sg = tx.pointing_sigma_rad * rr

    def f(rho: float) -> float:
        x = k * rho
        airy = 1.0 if x == 0.0 else (2.0 * float(special.j1(x)) / x) ** 2
        return (
            i0
            * airy
            * float(stats.ncx2.cdf(a * a / (sg * sg), 2, rho * rho / (sg * sg)))
            * (2.0 * math.pi * rho)
        )

    cap, _ = integrate.quad(f, 0.0, a + 40.0 * sg, epsabs=0.0, epsrel=1e-12, limit=400,
                            points=[max(a - 10 * sg, 0.0), a, a + 10 * sg])  # fmt: skip
    cap_err = abs(ref.eta_capture / cap - 1.0)
    chain = (
        ref.pv_electric_W
        * tx.power_conditioning_efficiency
        * tx.electrical_to_beam_efficiency
        * tx.optics_or_rf_efficiency
        * ref.eta_atmosphere
        * cap
        * cfg.receiver.conversion_efficiency
        * cfg.receiver.grid_efficiency
    )
    rated_err = abs(ref.rated_power_W / chain - 1.0)
    ok = cond_ok and cap_err < 1e-8 and rated_err < 1e-8
    return _Outcome(
        f"conditions ok: {cond_ok}; E[eta] {ref.eta_capture:.12g} vs {cap:.12g} (rel "
        f"{cap_err:.1e}); rated {ref.rated_power_W:.6g} W (chain rel {rated_err:.1e})",
        "window-mean range and heliocentric distance, zenith; independent noncentral-chi2 "
        "quadrature of the jittered Airy capture; PDF §10.1 chain (2-day MVP window)",
        "1e-8 relative (quadratures converge to < 1e-10)",
        ok,
    )


def _rebound_heliocentric(eph: Any, init: SatelliteState, epochs: Any) -> tuple[Any, Any]:
    """REBOUND IAS15 run of the force-model bodies and the satellite (test particle), G = 1,
    masses = GM [m^3 s^-2], SI units. Returns satellite and Sun positions (SSB) [m]. API as in
    tests/integration/test_rebound_crosscheck.py (verified against rebound 5.2.2)."""
    import rebound

    bodies = init.force_model.bodies
    t0 = init.epoch_tdb_s
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    sim.dt = 3600.0
    for body in bodies:
        st = eph.state(body, t0)
        sim.add(m=eph.gm_m3_s2(body), x=st.position_m[0], y=st.position_m[1],
                z=st.position_m[2], vx=st.velocity_m_s[0], vy=st.velocity_m_s[1],
                vz=st.velocity_m_s[2])  # fmt: skip
    r, v = init.position_m, init.velocity_m_s
    sim.add(m=0.0, x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])
    sim.N_active = len(bodies)
    sim.testparticle_type = 0
    i_sun = bodies.index("SUN")
    sat, sun = [], []
    for t in epochs:
        sim.integrate(float(t - t0), exact_finish_time=1)
        ps = sim.particles
        sat.append([ps[len(bodies)].x, ps[len(bodies)].y, ps[len(bodies)].z])
        sun.append([ps[i_sun].x, ps[i_sun].y, ps[i_sun].z])
    return np.array(sat), np.array(sun)


def _c_rebound(ctx: ValidationContext) -> _Outcome:
    try:
        import rebound
    except ImportError:
        return _Outcome("not run", "REBOUND", "-", None, "rebound not installed (xcheck extra)")
    from helioswarm.propagate import satellite_initial_state
    from helioswarm.timescale import utc_to_tdb_s

    year = 365.25 * SECONDS_PER_DAY
    cfg = ctx.base_config
    orbit = cfg.orbit.model_copy(
        update={"include_solar_radiation_pressure": False, "cr": None, "area_to_mass_m2_kg": None}
    )
    cap = IntegratorSettings(max_step_s=SECONDS_PER_DAY)
    # (1) two-body: Sun at rest + test particle, one year.
    eph0 = AnalyticEphemeris()
    o2 = orbit.model_copy(update={"include_planetary_perturbations": False})
    init0 = satellite_initial_state(o2, cfg.constants, epoch_tdb_s=9.467e8, ephemeris=eph0)
    t = init0.epoch_tdb_s + np.linspace(0.0, year, 37)
    l1 = propagate_l1(init0, t, ephemeris=eph0, settings=cap).trajectory.position_m
    sat, sun = _rebound_heliocentric(eph0, init0, t)
    d2 = float(np.max(np.linalg.norm((sat - sun) - (l1 - eph0.positions("SUN", t)), axis=1)))
    # (2) full force model (Sun + 9 bodies from de440s / gm_de440), MVP epoch, one year.
    with ctx.kernel_set.loaded(verify=False) as k:
        eph = SpiceEphemeris(k)
        t0 = utc_to_tdb_s(cfg.orbit.initial_epoch_utc, k)
        init = satellite_initial_state(orbit, cfg.constants, epoch_tdb_s=t0, ephemeris=eph)
        tt = t0 + np.linspace(0.0, year, 37)
        l1n = propagate_l1(init, tt, ephemeris=eph, settings=cap).trajectory.position_m
        sun_de = eph.positions("SUN", tt)
        satn, sunn = _rebound_heliocentric(eph, init, tt)
    dn = float(np.max(np.linalg.norm((satn - sunn) - (l1n - sun_de), axis=1)))
    return _Outcome(
        f"two-body: max heliocentric diff {d2:.1e} m; Sun + {len(init.force_model.bodies) - 1} "
        f"bodies: {dn:.2f} m (1 year, 37 epochs)",
        f"REBOUND {rebound.__version__} IAS15 from the same initial states and GMs (PDF §13.3)",
        "1e-2 m two-body (both integrators << 1 m); 10 m N-body (REBOUND evolves the planets "
        "self-consistently, L1 prescribes DE440; measured ~1 m, module docstring of "
        "tests/integration/test_rebound_crosscheck.py)",
        d2 < 1e-2 and dn < 10.0,
    )


@dataclass(frozen=True)
class Check:
    """One registered check: PDF section, name, function and whether it is slow."""

    section: str
    name: str
    fn: Callable[[ValidationContext], _Outcome]
    slow: bool = False


CHECKS: Final[tuple[Check, ...]] = (
    Check(
        "§7.4", "Circular two-body orbit (L0) preserves radius; Kepler III period", _c_l0_circular
    ),
    Check("§7.4", "Circular orbit under L1 (Sun only) within integrator tolerance", _c_l1_circular),
    Check("§7.4", "Earth-Sun state vs JPL Horizons", _c_horizons),
    Check("§7.4", "Receiver site: one sidereal rotation per IAU_EARTH period", _c_sidereal),
    Check(
        "§7.4", "Beam segment through / past the Earth ellipsoid flagged correctly", _c_occultation
    ),
    Check("§7.4", "Light time: static d/c and uniformly moving site closed form", _c_light_time),
    Check("§8.5", "S(1 au) = S_1AU and 1/r^2 scaling", _c_solar),
    Check("§8.5", "Zero incidence cosine or umbra gives zero power", _c_zero_incidence),
    Check("§8.5", "Energy closure and zero power in eclipse (0.3-1.2 au)", _c_eclipse_and_ledger),
    Check("§8.5", "Radiator area proportional to 1/(eps (T^4 - Ts^4))", _c_radiator),
    Check("§9", "Airy first null at 1.21967 lambda R/D", _c_first_null),
    Check(
        "§9", "Encircled energy at first dark ring; total power normalisation", _c_ee_ring_and_power
    ),
    Check(
        "§9",
        "Offset capture vs independent nested quadrature (2-D grid as supporting evidence)",
        _c_capture_grid,
    ),
    Check(
        "§9", "Expected capture non-increasing with pointing sigma (centred)", _c_pointing_monotone
    ),
    Check("§9", "Smaller wavelength / larger aperture improve capture", _c_lambda_aperture),
    Check("§9", "Impossible beam gives ~x^2/4 capture", _c_impossible_beam),
    Check("§9", "Fresnel-Hankel -> Airy as N_F -> 0", _c_fresnel_to_airy),
    Check("§9/§14.3", "Batched capture kernel (numpy, JAX) vs M5 reference", _c_batch_kernel),
    Check(
        "§13.1",
        "Link budget: passive stages never increase power; eta in [0,1]; clip",
        _c_link_budget,
    ),
    Check("§13.1", "SRP: magnitude, anti-sunward direction, inverse square", _c_srp),
    Check("§13.1", "Time scales: UTC -> TDB at known epochs", _c_time_scale),
    Check("§10.5", "Capacity-factor reference link: conditions and rated power", _c_reference_link),
    Check("§13.2", "1-AU baseline: config -> saved outputs without manual steps", _c_e2e_baseline),
    Check("§13.2", "Occulted / unavailable link delivers zero power", _c_e2e_unavailable_zero),
    Check("§13.2", "Receiver capacity limit clips correctly", _c_e2e_clip),
    Check(
        "§13.2", "Doubling collector area doubles P_PV (and P_grid unclipped)", _c_e2e_area_linear
    ),
    Check("§13.2", "Smaller lambda / larger D improve capture end to end", _c_e2e_lambda_aperture),
    Check("§13.2", "Larger pointing sigma cannot improve expected capture", _c_e2e_pointing),
    Check(
        "§13.2",
        "Impossible microwave beam: near-zero capture, not high power",
        _c_e2e_microwave_impossible,
    ),
    Check("§13.2", "Energy accounting: P_grid <= every upstream stage", _c_e2e_energy_accounting),
    Check(
        "§13.2",
        "Same config + seed -> identical summary and timeseries content",
        _c_e2e_reproducible,
    ),
    Check("§13.3", "REBOUND orbit cross-check (two-body and full force model)", _c_rebound),
    Check("§13.3", "HCIPy far-field cross-check", _c_hcipy),
    Check("§13.3", "ITU-R P.618 validation example (itur)", _c_itur),
    Check("§13.3", "Capture quadrature refinement convergence", _c_capture_refinement),
    Check(
        "§13.3",
        "L1 integrator step / tolerance convergence (30 d MVP)",
        _c_l1_step_convergence,
        slow=True,
    ),
    Check(
        "§13.3",
        "Time-step (diurnal sampling) convergence of the access fraction",
        _c_diurnal_sampling,
        slow=True,
    ),
)
"""Registered checks in report order."""


def run_validation(
    ctx: ValidationContext, log: Callable[[str], None] | None = None
) -> tuple[CheckResult, ...]:
    """Run every check (slow ones only with ``ctx.include_slow``); never raises for a check."""
    out: list[CheckResult] = []
    for check in CHECKS:
        section, name, fn = check.section, check.name, check.fn
        if check.slow and not ctx.include_slow:
            out.append(
                CheckResult(
                    section, name, "not run", "-", "-", None, "slow check skipped (--quick)"
                )
            )
            continue
        t0 = time.perf_counter()
        try:
            o = fn(ctx)
            passed = None if o.passed is None else bool(o.passed)
            res = CheckResult(section, name, o.value, o.reference, o.tolerance, passed, o.note,
                              time.perf_counter() - t0)  # fmt: skip
        except Exception as exc:  # a crashing check is a failed check, reported verbatim
            res = CheckResult(section, name, "error", "-", "-", False,
                              f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=2)}",
                              time.perf_counter() - t0)  # fmt: skip
        out.append(res)
        if log is not None:
            status = {True: "PASS", False: "FAIL", None: "NOT RUN"}[res.passed]
            log(f"{status:7} {section:8} {name} ({res.wall_s:.1f} s)")
    return tuple(out)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def report_block(results: tuple[CheckResult, ...], header: dict[str, str]) -> str:
    """Markdown block (between the markers) for ``docs/validation_report.md``."""
    n_pass = sum(1 for r in results if r.passed is True)
    n_fail = sum(1 for r in results if r.passed is False)
    n_skip = sum(1 for r in results if r.passed is None)
    lines = [
        BEGIN_MARK,
        "",
        "## Programmatic validation (`helioswarm validate-models`, generated)",
        "",
        "Generated by `helioswarm validate-models`; do not edit by hand. "
        + "; ".join(f"{k}: {v}" for k, v in header.items())
        + ".",
        "",
        f"**{n_pass} passed, {n_fail} failed, {n_skip} not run** of {len(results)} checks.",
        "",
        "| PDF | Check | Value | Reference | Tolerance | Result |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        status = {True: "**pass**", False: "**FAIL**", None: "not run"}[r.passed]
        lines.append(
            f"| {r.section} | {_cell(r.name)} | {_cell(r.value)} | {_cell(r.reference)} | "
            f"{_cell(r.tolerance)} | {status} |"
        )
    unresolved = [r for r in results if r.passed is not True]
    lines += ["", "### Unresolved (generated)", ""]
    if unresolved:
        lines += [f"- {r.section} {r.name}: {r.value}; {_cell(r.note)}" for r in unresolved]
    else:
        lines.append("None: every check passed.")
    notes = [r for r in results if r.passed is True and r.note]
    if notes:
        lines += ["", "Notes:", ""] + [f"- {r.name}: {r.note}" for r in notes]
    lines += ["", END_MARK]
    return "\n".join(lines) + "\n"


def write_report(path: Path, block: str) -> Path:
    """Insert ``block`` between the markers of ``path`` (append if absent; create if missing)."""
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if BEGIN_MARK in text and END_MARK in text:
            pre = text.split(BEGIN_MARK, 1)[0]
            post = text.split(END_MARK, 1)[1]
            text = pre + block.rstrip("\n") + post
        else:
            text = text.rstrip("\n") + "\n\n" + block
    else:
        text = "# Validation report\n\n" + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_base_config(path: str | Path) -> RunConfig:
    """The MVP configuration the end-to-end checks start from."""
    return parse_config(read_yaml(path), source=str(path))


__all__ = [
    "BEGIN_MARK",
    "CHECKS",
    "END_MARK",
    "HORIZONS_EARTH_WRT_SUN_KM",
    "Check",
    "CheckResult",
    "LoadedKernels",
    "ValidationContext",
    "load_base_config",
    "report_block",
    "run_validation",
    "write_report",
]
