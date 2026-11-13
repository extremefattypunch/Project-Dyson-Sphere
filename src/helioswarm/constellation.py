"""Constellation runs: many independent satellites, several receiver sites, a scheduler
(PDF §11.2-§11.4, §10.4 several transmitters).

Architecture (two stages, so large constellations can be split over Slurm array tasks)
--------------------------------------------------------------------------------------
**Stage A - link candidates** (:func:`evaluate_links`, per satellite; embarrassingly
parallel). For each satellite of the constellation (:func:`constellation_members`: rings,
phases, per-ring technology, each satellite a validated single-satellite configuration):

1. orbit: L1 test particles integrated together in stacked solves
   (:func:`helioswarm.propagate.propagate_l1_stacked`; a stack of one is the solo
   integrator) or the closed-form L0 orbit; the L0 reference of each satellite for the drift;
2. illumination and the M4 solar/PV/thermal chain at the transmitter operating point;
3. for every receiver site: retarded-time access and occultation (M3), atmosphere and beam
   capture at the epochs with access (M5), and the single-link availability conditions and
   link budget (:func:`helioswarm.system.site_link_budget`, the very function of the
   single-satellite run).

The result (:class:`LinkSet`) holds, per (epoch, satellite, site), the single-link reason code
(the M6 blocking bits; a link is *feasible* when none is set), the beam power entering the
receiver aperture ``P_captured``, ``P_atm`` and the peak incident irradiance; per (epoch,
satellite) the PV chain and thermal outputs. Receiver limits are **not** applied here (they
are applied once, to the sum over all beams at a site, in stage B).

**Stage B - scheduling and network power** (:func:`finish_constellation`). Per epoch the
feasible links go to the scheduler (:mod:`helioswarm.scheduler`: beam exclusivity, beam slots,
summed power-density interlock, summed receiver input/heat limits, optional duty cycle;
greedy or exact optimal policy). For site ``j``::

    P_incident,j = sum_{scheduled l at j} P_captured,l          (independent beams: powers add)
    P_accepted,j = min(P_incident,j, P_max_input,j, Q_max,j / (1 - eta_rx))
    P_receiver,j = P_accepted,j * eta_receiver_conversion
    P_grid,j     = P_receiver,j * eta_grid_conditioning
    P_grid,net   = sum_j P_grid,j

A satellite serves at most one site, so no beam and no receiver capacity is counted twice
(PDF §10.4 "prevent double-counting"). Beams of different satellites are incoherent: their
powers and intensities add; no field amplitudes are ever added (PDF §2 non-goal, §11.3;
``BeamResult.coherent_combining`` is always False). For one satellite and one site with no
binding site constraint, stage B reduces exactly to the single-satellite link budget
(``min(min(P_captured, P_max), Q_max/(1 - eta_rx))`` in the same floating-point order; tested
against :func:`helioswarm.system.run_single_satellite`).

Per-satellite delivered power (for per-satellite summaries only) is the site's ``P_grid``
shared in proportion to the captured powers of the scheduled beams at that site (pro-rata
clipping); the network totals never use it.

Emission-epoch aggregation (named approximation; :class:`ReceptionTimeAggregates`)
--------------------------------------------------------------------------------
The output grid is the satellites' *emission* epochs ``t_e``. A beam emitted at ``t_e`` reaches
site ``j`` at ``t_e + tau_ij`` and ``tau_ij = range_ij / c`` differs between satellites (258 s
at 30 deg from the Earth, 964 s at 150 deg on the 1 au ring). Stage B sums, clips and limits
(beam slots, summed input/heat limit, summed density interlock) the beams scheduled at the same
*emission* epoch, i.e. beams that are not simultaneous at the receiver, and ``P_grid,j`` is
reported on the emission grid. Beam exclusivity is a transmitter constraint and is exact on the
emission grid. The approximation is named and bounded on every run: stage B also reconstructs
the arriving beams in *reception* time (each scheduled beam held over its emission step and
shifted by its own light time, sample-and-hold) and records per site and reception slot the
largest number of simultaneously arriving beams, the largest summed density, the time over the
beam-slot and density limits, and the incident and accepted energies, together with the
light-time spread of the beams summed at each site and epoch. The summary compares the
accepted energy in reception time with the emission-grid value (same hold convention).

Unconstrained baseline (``unconstrained_grid_W``): every satellite on its best feasible link
with no site capacity (no beam slots, input/heat or density limit, no duty cycle), times
``eta_rx eta_grid``: the sum of what the satellites would deliver with unlimited receivers.
It separates the geometry (placement) effect from the capacity effect: ``1 - E_grid /
E_unconstrained`` is the capacity loss.

Unavailability (PDF §11.4 "why power is unavailable"): each link carries the M6 blocking bits
plus the scheduler bits ``DUTY_CYCLE``, ``RECEIVER_CAPACITY``, ``SCHEDULER_NOT_SELECTED``
(:class:`helioswarm.system.Unavailability`, precedence :data:`LINK_REASONS`). A satellite that
delivers nothing at an epoch is attributed the reason of its link that got *furthest* through
the chain (largest precedence index over its sites; e.g. below the horizon of site A but in
eclipse while visible from site B -> ``ECLIPSE``; feasible but not scheduled -> the scheduler
reason).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, TypeVar

import numpy as np
from numpy.typing import NDArray

from helioswarm.atmosphere import AtmosphereResult, atmosphere_settings_from_config
from helioswarm.beam import BeamResult, BeamSettings, beam_settings_from_config
from helioswarm.config import (
    ConstellationConfig,
    RunConfig,
    approximations_in_force,
    config_hash,
    ring_phase_offset_deg,
    satellite_config,
)
from helioswarm.constants import SPEED_OF_LIGHT_M_S
from helioswarm.ephemeris import EphemerisProvider
from helioswarm.frames import CartesianTrajectory
from helioswarm.ground import EarthOrientation, GroundSite, SpiceEarthOrientation
from helioswarm.kernels import KernelProvenance, LoadedKernels
from helioswarm.link_budget import ReceiverModel, receiver_model_for_site
from helioswarm.power_chain import onboard_power_chain
from helioswarm.propagate import (
    L1_MODEL,
    check_kernel_set,
    integrator_settings_from_config,
    l0_reference_trajectory,
    propagate_l1_stacked,
    resolve_orbit_phase,
    satellite_initial_state,
)
from helioswarm.scheduler import (
    NOT_A_CANDIDATE,
    DutyCycle,
    LinkDecision,
    ScheduleTimeseries,
    SiteCapacity,
    schedule_timeseries,
)
from helioswarm.system import (
    BLOCKING_MASK,
    BLOCKING_REASONS,
    Backend,
    BeamCapture,
    RunLogEntry,
    StageTiming,
    Unavailability,
    _evaluate_beam_backend,
    atmosphere_at_access,
    site_link_budget,
)
from helioswarm.thermal import (
    PowerThermalModels,
    PowerThermalResult,
    power_thermal_from_illumination,
    power_thermal_models_from_config,
)
from helioswarm.timescale import TimeGrid, tdb_s_to_utc, time_grid_from_config, utc_to_tdb_s
from helioswarm.visibility import (
    AccessResult,
    IlluminationResult,
    OcculterRadii,
    access_settings_from_config,
    compute_access,
    orbit_drift_m,
    satellite_illumination,
)

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_T = TypeVar("_T")

LINK_REASONS: Final[tuple[Unavailability, ...]] = (
    *BLOCKING_REASONS,
    Unavailability.DUTY_CYCLE,
    Unavailability.RECEIVER_CAPACITY,
    Unavailability.SCHEDULER_NOT_SELECTED,
)
"""Reasons a link or satellite delivers nothing, in precedence order (geometry, spacecraft
state, link conditions, then the scheduler's decisions)."""

SCHEDULER_BITS: Final[int] = int(
    Unavailability.DUTY_CYCLE
    | Unavailability.RECEIVER_CAPACITY
    | Unavailability.SCHEDULER_NOT_SELECTED
)

_DECISION_BIT: Final[tuple[tuple[int, int], ...]] = (
    (int(LinkDecision.SCHEDULER_NOT_SELECTED), int(Unavailability.SCHEDULER_NOT_SELECTED)),
    (int(LinkDecision.RECEIVER_CAPACITY), int(Unavailability.RECEIVER_CAPACITY)),
    (int(LinkDecision.DUTY_CYCLE), int(Unavailability.DUTY_CYCLE)),
)

DELIVERING: Final[int] = -1
"""``satellite_primary_reason`` value of a satellite that delivers power (scheduled)."""


# ---------------------------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstellationMember:
    """One satellite of a constellation.

    Attributes:
        index: Global satellite index (rings in order, then position on the ring).
        ring_index: Ring of the satellite.
        ring_name: Its name.
        index_in_ring: ``k`` on the ring.
        phase_angle_deg: Configured phase angle ``phi_k`` in [0, 360) [deg], interpreted with
            ``config.orbit.phase_reference``.
        config: The satellite's single-satellite configuration (ring elements, phase and
            technology applied; validated with the full schema).
    """

    index: int
    ring_index: int
    ring_name: str
    index_in_ring: int
    phase_angle_deg: float
    config: RunConfig


def _ring_phases(
    constellation: ConstellationConfig, ring_index: int, offset_deg: float, rng: np.random.Generator
) -> Float64Array:
    ring = constellation.rings[ring_index]
    n = ring.n_satellites
    span = ring.phase_span_deg
    if ring.phase_distribution == "even":
        phi = offset_deg + span * np.arange(n, dtype=np.float64) / n
    else:
        phi = offset_deg + rng.uniform(0.0, span, size=n)
    out = np.mod(phi, 360.0)
    out[out >= 360.0] = 0.0
    return np.asarray(out, dtype=np.float64)


def constellation_members(
    config: RunConfig, rng: np.random.Generator
) -> tuple[ConstellationMember, ...]:
    """Every satellite of ``config.constellation`` (PDF §11.2).

    Ring ``r`` with ``n`` satellites gives phases ``phi_k = offset + span k/n`` (``even``) or
    ``offset + U_k``, ``U_k ~ uniform[0, span)`` drawn from ``rng`` (``random``; rings drawn in
    configuration order, ``n`` draws each), reduced mod 360 (``span = ring.phase_span_deg``).
    ``offset`` is ``ring.phase_offset_deg`` or ``orbit.true_anomaly_deg``. The first satellite
    of each ring is validated as a full single-satellite configuration
    (:func:`helioswarm.config.satellite_config`); the others differ only in the phase angle,
    which is in range by construction.

    Args:
        config: Configuration with a ``constellation`` section.
        rng: The run's Generator, seeded from ``project.seed`` by the caller (used only by
            ``random`` rings).

    Raises:
        ValueError: no constellation section.
    """
    c = config.constellation
    if c is None:
        raise ValueError("configuration has no constellation section")
    out: list[ConstellationMember] = []
    for r, ring in enumerate(c.rings):
        phases = _ring_phases(c, r, ring_phase_offset_deg(config, r), rng)
        base = satellite_config(config, r, float(phases[0]))
        for k, phi in enumerate(phases.tolist()):
            cfg = (
                base
                if k == 0
                else base.model_copy(
                    update={"orbit": base.orbit.model_copy(update={"true_anomaly_deg": phi})}
                )
            )
            out.append(
                ConstellationMember(
                    index=len(out),
                    ring_index=r,
                    ring_name=ring.name,
                    index_in_ring=k,
                    phase_angle_deg=float(phi),
                    config=cfg,
                )
            )
    return tuple(out)


def chunk_indices(n_satellites: int, chunk_index: int, chunk_count: int) -> range:
    """Contiguous block of satellites of one array task: ``[floor(i N/n), floor((i+1) N/n))``."""
    if chunk_count < 1 or not 0 <= chunk_index < chunk_count:
        raise ValueError("need chunk_count >= 1 and 0 <= chunk_index < chunk_count")
    return range(
        (chunk_index * n_satellites) // chunk_count,
        ((chunk_index + 1) * n_satellites) // chunk_count,
    )


def site_capacities(config: RunConfig) -> tuple[SiteCapacity, ...]:
    """:class:`~helioswarm.scheduler.SiteCapacity` of every receiver site (site overrides of
    the beam slots, input and heat limits; the density interlock of the receiver section)."""
    out = []
    for site in config.receiver.sites:
        model = receiver_model_for_site(config.receiver, site)
        out.append(
            SiteCapacity(
                max_beams=config.receiver.site_max_beams(site),
                max_accepted_W=min(model.max_input_power_W, model.heat_limited_input_W),
                max_power_density_W_m2=model.max_power_density_W_m2,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# Stage A: link candidates
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LinkSet:
    """Stage-A result for a set of satellites (``T`` epochs, ``n`` satellites, ``S`` sites).

    Link arrays have shape (T, n, S); satellite arrays (T, n); per-satellite scalars (n,).

    Attributes:
        config_hash: Hash of the constellation configuration.
        satellite_index: Global indices of the satellites held, increasing.
        epochs_tdb_s: Output (emission) epochs [TDB s].
        reason_code: Single-link :class:`~helioswarm.system.Unavailability` bits (no scheduler
            bits; the single-beam receiver-saturation flag is cleared); feasible =
            no blocking bit.
        captured_W: ``P_captured`` of the link if operated [W] (0 where not feasible).
        atm_W: ``P_atm`` of the link [W] (0 where not feasible).
        density_W_m2: Peak incident irradiance of the beam [W m^-2] (0 where not feasible).
        range_m: Satellite-site range [m].
        elevation_rad: Apparent elevation at the site [rad].
        pv_electric_W: ``P_PV`` [W].
        incident_solar_W: Sunlight incident on the collector [W].
        aperture_W: ``P_aperture`` the satellite emits when transmitting [W].
        dc_to_tx_W: ``P_dc_to_tx`` when transmitting [W].
        beam_W: ``P_beam`` when transmitting [W].
        illumination_fraction: Visible solar-disk fraction [1].
        thermal_feasible: Thermal verdict at the operating point.
        radiator_area_m2: Required radiator area [m^2].
        pv_temperature_K: PV plate temperature [K].
        sun_distance_m: Heliocentric distance [m].
        true_anomaly_rad: Resolved true anomaly at the element epoch [rad].
        ecliptic_longitude_rad: Heliocentric ecliptic longitude at the element epoch [rad].
        earth_relative_longitude_rad: That longitude minus the Earth's, wrapped to
            [-pi, pi) [rad].
        orbit_drift_max_m: ``max |r - r_L0|`` (L1), NaN for L0 [m].
        orbit_energy_rel_drift: L1 two-body energy drift diagnostic (NaN for L0) [1].
        orbit_nfev: Right-hand-side evaluations of the satellite's stack (0 for L0).
        light_time_fallbacks: Brent fallbacks of the light-time solver (all sites).
        itu_r_versions: ITU-R recommendations used (``itu_r`` atmosphere), else empty.
        itu_r_polarization_tilt_deg: Rain polarisation tilt (``itu_r``), else ``None``.
        source_ids: Ephemeris / Earth-orientation data ids.
        backend: Beam backend used.
        timings: Wall time per stage [s].
        log: Diagnostics.
        code_provenance: Code that computed the set
            (:func:`helioswarm.manifest.code_provenance`; recorded by array tasks), or
            ``None``.
    """

    config_hash: str
    satellite_index: NDArray[np.int64]
    epochs_tdb_s: Float64Array
    reason_code: NDArray[np.uint32]
    captured_W: Float64Array
    atm_W: Float64Array
    density_W_m2: Float64Array
    range_m: Float64Array
    elevation_rad: Float64Array
    pv_electric_W: Float64Array
    incident_solar_W: Float64Array
    aperture_W: Float64Array
    dc_to_tx_W: Float64Array
    beam_W: Float64Array
    illumination_fraction: Float64Array
    thermal_feasible: BoolArray
    radiator_area_m2: Float64Array
    pv_temperature_K: Float64Array
    sun_distance_m: Float64Array
    true_anomaly_rad: Float64Array
    ecliptic_longitude_rad: Float64Array
    earth_relative_longitude_rad: Float64Array
    orbit_drift_max_m: Float64Array
    orbit_energy_rel_drift: Float64Array
    orbit_nfev: NDArray[np.int64]
    light_time_fallbacks: NDArray[np.int64]
    itu_r_versions: tuple[str, ...] = ()
    itu_r_polarization_tilt_deg: float | None = None
    source_ids: tuple[str, ...] = ()
    backend: str = "numpy"
    timings: tuple[StageTiming, ...] = field(default=(), repr=False)
    log: tuple[RunLogEntry, ...] = field(default=(), repr=False)
    code_provenance: dict[str, Any] | None = field(default=None, repr=False)

    LINK_FIELDS: ClassVar[tuple[str, ...]] = (
        "reason_code",
        "captured_W",
        "atm_W",
        "density_W_m2",
        "range_m",
        "elevation_rad",
    )
    SATELLITE_FIELDS: ClassVar[tuple[str, ...]] = (
        "pv_electric_W",
        "incident_solar_W",
        "aperture_W",
        "dc_to_tx_W",
        "beam_W",
        "illumination_fraction",
        "thermal_feasible",
        "radiator_area_m2",
        "pv_temperature_K",
        "sun_distance_m",
    )
    SCALAR_FIELDS: ClassVar[tuple[str, ...]] = (
        "true_anomaly_rad",
        "ecliptic_longitude_rad",
        "earth_relative_longitude_rad",
        "orbit_drift_max_m",
        "orbit_energy_rel_drift",
        "orbit_nfev",
        "light_time_fallbacks",
    )

    def __post_init__(self) -> None:
        n_t = self.epochs_tdb_s.shape[0]
        n = self.satellite_index.shape[0]
        if n and np.any(np.diff(self.satellite_index) <= 0):
            raise ValueError("satellite_index must be strictly increasing")
        n_s = self.reason_code.shape[2] if self.reason_code.ndim == 3 else -1
        for name in self.LINK_FIELDS:
            if getattr(self, name).shape != (n_t, n, n_s):
                raise ValueError(f"{name} must have shape (T, n, S) = {(n_t, n, n_s)}")
        for name in self.SATELLITE_FIELDS:
            if getattr(self, name).shape != (n_t, n):
                raise ValueError(f"{name} must have shape (T, n) = {(n_t, n)}")
        for name in self.SCALAR_FIELDS:
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape (n,) = {(n,)}")

    @property
    def n_satellites(self) -> int:
        """Satellites held."""
        return int(self.satellite_index.shape[0])

    @property
    def n_sites(self) -> int:
        """Receiver sites."""
        return int(self.reason_code.shape[2])

    @property
    def feasible(self) -> BoolArray:
        """Links with no single-link blocking reason, (T, n, S)."""
        return np.asarray((self.reason_code & np.uint32(BLOCKING_MASK)) == 0)


def _timed(stage: str, acc: dict[str, float], fn: Callable[[], _T]) -> _T:
    t0 = time.perf_counter()
    value = fn()
    acc[stage] = acc.get(stage, 0.0) + time.perf_counter() - t0
    return value


def _wrap_pi(x: Float64Array) -> Float64Array:
    return np.asarray(np.mod(x + math.pi, 2.0 * math.pi) - math.pi, dtype=np.float64)


@dataclass(frozen=True)
class _CaptureSlice:
    """:class:`~helioswarm.system.BeamCapture` view of part of a batched beam result."""

    capture_fraction: Float64Array
    peak_irradiance_per_W_m2: Float64Array
    pointing_loss: Float64Array


def evaluate_links(
    config: RunConfig,
    members: Sequence[ConstellationMember],
    *,
    ephemeris: EphemerisProvider,
    kernels: LoadedKernels,
    earth_orientation: EarthOrientation | None = None,
    backend: Backend = "numpy",
    block_size: int = 16,
) -> LinkSet:
    """Stage A for ``members`` (module docstring).

    Args:
        config: The constellation configuration (sites, receiver, atmosphere, time grid).
        members: Satellites to evaluate (a subset of :func:`constellation_members`, increasing
            index).
        ephemeris: Ephemeris provider.
        kernels: Active kernel set (the configured one).
        earth_orientation: Earth-fixed -> J2000 model (default SPICE ``IAU_EARTH``).
        backend: Beam backend. ``numpy`` evaluates every satellite-site pair with the M5
            per-epoch reference (exactly as the single-satellite run); ``numpy_batch`` and
            ``jax`` evaluate the visible epochs of all pairs of a block of satellites with the
            same beam settings in one batched call (:mod:`helioswarm.capture_batch`).
        block_size: Satellites per block (bounds the memory of the per-site access results).

    Returns:
        :class:`LinkSet`.
    """
    if config.constellation is None:
        raise ValueError("configuration has no constellation section")
    if backend not in ("numpy", "numpy_batch", "jax"):
        raise ValueError(f"unknown backend {backend!r}")
    check_kernel_set(config, kernels)
    mem = tuple(members)
    idx = np.array([m.index for m in mem], dtype=np.int64)
    if idx.size and np.any(np.diff(idx) <= 0):
        raise ValueError("members must be in increasing index order")
    timings: dict[str, float] = {}
    log: list[RunLogEntry] = []
    orientation: EarthOrientation = (
        earth_orientation if earth_orientation is not None else SpiceEarthOrientation(kernels)
    )
    radii = OcculterRadii.from_kernels(kernels)
    grid = time_grid_from_config(config.time, kernels)
    t = grid.epochs_tdb_s
    t0 = utc_to_tdb_s(config.orbit.initial_epoch_utc, kernels)
    sites = config.receiver.sites
    ground = [GroundSite.from_config(s) for s in sites]
    receivers: list[ReceiverModel] = [receiver_model_for_site(config.receiver, s) for s in sites]
    atm_settings = atmosphere_settings_from_config(config)
    access_settings = access_settings_from_config(config, radii)
    n_t, n, n_s = t.size, len(mem), len(sites)

    # ---- orbits -----------------------------------------------------------------------
    phases = [resolve_orbit_phase(m.config.orbit, epoch_tdb_s=t0, ephemeris=ephemeris) for m in mem]
    lam_earth = None
    if any(p.earth_ecliptic_longitude_rad is not None for p in phases):
        lam_earth = next(
            p.earth_ecliptic_longitude_rad
            for p in phases
            if p.earth_ecliptic_longitude_rad is not None
        )
    if lam_earth is None and mem:
        from helioswarm.propagate import earth_ecliptic_longitude_rad

        lam_earth = earth_ecliptic_longitude_rad(t0, ephemeris)
    l0 = _timed(
        "orbit_l0",
        timings,
        lambda: [
            l0_reference_trajectory(m.config.orbit, t, initial_epoch_tdb_s=t0, ephemeris=ephemeris)
            for m in mem
        ],
    )
    trajs: list[CartesianTrajectory]
    drift_max = np.full(n, np.nan)
    energy = np.full(n, np.nan)
    nfev = np.zeros(n, dtype=np.int64)
    if config.orbit.model == L1_MODEL and mem:
        inits = [
            satellite_initial_state(
                m.config.orbit,
                m.config.constants,
                epoch_tdb_s=t0,
                ephemeris=ephemeris,
                attitude=m.config.collector.incidence_mode,
                fixed_normal_unit_j2000=m.config.collector.fixed_normal_unit_j2000,
            )
            for m in mem
        ]
        stacked = _timed(
            "orbit",
            timings,
            lambda: propagate_l1_stacked(
                inits,
                t,
                ephemeris=ephemeris,
                settings=integrator_settings_from_config(config.time),
            ),
        )
        trajs = [p.trajectory for p in stacked.propagations]
        for i, p in enumerate(stacked.propagations):
            drift_max[i] = float(np.max(orbit_drift_m(p.trajectory, l0[i])))
            energy[i] = p.diagnostics.sun_two_body_energy_rel_drift
            nfev[i] = p.diagnostics.nfev
        log.append(
            RunLogEntry(
                "INFO",
                "orbit",
                f"L1 stacked integration: {len(stacked.stack_sizes)} stack(s) of sizes "
                f"{list(stacked.stack_sizes)}, tolerance scale "
                f"{[round(s, 6) for s in stacked.tolerance_scale]}; nfev "
                f"{sorted(set(nfev.tolist()))}",
            )
        )
        if not all(p.diagnostics.success for p in stacked.propagations):
            log.append(RunLogEntry("WARNING", "orbit", "an L1 stack reported no success"))
    else:
        trajs = list(l0)
        log.append(RunLogEntry("INFO", "orbit", "L0 two-body Kepler orbits (closed form)"))

    # ---- output arrays ----------------------------------------------------------------
    code = np.zeros((n_t, n, n_s), dtype=np.uint32)
    captured = np.zeros((n_t, n, n_s))
    atm_w = np.zeros((n_t, n, n_s))
    density = np.zeros((n_t, n, n_s))
    range_m = np.zeros((n_t, n, n_s))
    elev = np.zeros((n_t, n, n_s))
    sat_arrays = {
        name: np.zeros((n_t, n))
        for name in (
            "pv_electric_W",
            "incident_solar_W",
            "aperture_W",
            "dc_to_tx_W",
            "beam_W",
            "illumination_fraction",
            "radiator_area_m2",
            "pv_temperature_K",
            "sun_distance_m",
        )
    }
    thermal_ok = np.zeros((n_t, n), dtype=bool)
    lt_fallbacks = np.zeros(n, dtype=np.int64)
    itu_versions: tuple[str, ...] = ()
    itu_tilt: float | None = None
    source_ids: tuple[str, ...] = ()
    models_by_ring: dict[int, PowerThermalModels] = {}
    beam_by_ring: dict[int, BeamSettings] = {}
    n_pv_unconverged = 0
    n_ledger_open = 0
    step = max(1, block_size)

    for b0 in range(0, n, step):
        pending: list[
            tuple[int, int, AccessResult, NDArray[np.int64], AtmosphereResult | None]
        ] = []
        state: dict[int, tuple[IlluminationResult, PowerThermalResult, PowerThermalModels]] = {}
        for i in range(b0, min(n, b0 + step)):
            m = mem[i]
            if m.ring_index not in models_by_ring:
                models_by_ring[m.ring_index] = power_thermal_models_from_config(m.config)
                beam_by_ring[m.ring_index] = beam_settings_from_config(m.config)
            models = models_by_ring[m.ring_index]
            traj = trajs[i]
            t_a = time.perf_counter()
            illum = satellite_illumination(traj, ephemeris=ephemeris, radii=radii)
            t_b = time.perf_counter()
            pt = power_thermal_from_illumination(illum, models)
            t_c = time.perf_counter()
            timings["illumination"] = timings.get("illumination", 0.0) + t_b - t_a
            timings["solar_thermal"] = timings.get("solar_thermal", 0.0) + t_c - t_b
            th = pt.thermal
            n_pv_unconverged += int(np.count_nonzero(~th.pv_converged))
            n_ledger_open += 0 if pt.ledger.closed else 1
            chain = onboard_power_chain(pt.solar.pv_electric_W, models.efficiencies)
            sat_arrays["pv_electric_W"][:, i] = pt.solar.pv_electric_W
            sat_arrays["incident_solar_W"][:, i] = pt.solar.incident_W
            sat_arrays["aperture_W"][:, i] = chain.aperture_W
            sat_arrays["dc_to_tx_W"][:, i] = chain.dc_to_tx_W
            sat_arrays["beam_W"][:, i] = chain.beam_W
            sat_arrays["illumination_fraction"][:, i] = illum.illumination_fraction
            sat_arrays["radiator_area_m2"][:, i] = th.radiator_area_m2
            sat_arrays["pv_temperature_K"][:, i] = pt.solar.pv_temperature_K
            sat_arrays["sun_distance_m"][:, i] = illum.sun_distance_m
            thermal_ok[:, i] = th.feasible
            state[i] = (illum, pt, models)
            for j, site_cfg in enumerate(sites):
                t_a = time.perf_counter()
                acc = compute_access(
                    traj,
                    ground[j],
                    ephemeris=ephemeris,
                    earth_orientation=orientation,
                    settings=access_settings,
                )
                t_b = time.perf_counter()
                vis_idx, atm = atmosphere_at_access(acc, atm_settings, site_cfg)
                t_c = time.perf_counter()
                timings["access"] = timings.get("access", 0.0) + t_b - t_a
                timings["atmosphere"] = timings.get("atmosphere", 0.0) + t_c - t_b
                if not source_ids:
                    source_ids = tuple(acc.source_ids)
                lt_fallbacks[i] += int(np.count_nonzero(acc.light_time.method == "brentq"))
                if atm is not None and atm.recommendation_versions and not itu_versions:
                    itu_versions = tuple(atm.recommendation_versions)
                    itu_tilt = atm.polarization_tilt_deg
                range_m[:, i, j] = acc.range_m
                elev[:, i, j] = acc.elevation_rad
                pending.append((i, j, acc, vis_idx, atm))

        # ---- beams: per pair (numpy reference) or batched per ring (numpy_batch / jax) -----
        beams: dict[tuple[int, int], BeamCapture] = {}
        t_a = time.perf_counter()
        if backend == "numpy":
            for i, j, acc, vis_idx, atm in pending:
                if atm is None:
                    continue
                beams[(i, j)] = _evaluate_beam_backend(
                    acc.range_m[vis_idx],
                    acc.elevation_rad[vis_idx],
                    beam_by_ring[mem[i].ring_index],
                    "numpy",
                    acc.t_emit_tdb_s[vis_idx],
                )
        else:
            for r in sorted({mem[i].ring_index for i, *_ in pending}):
                parts = [
                    (i, j, acc, vis_idx)
                    for i, j, acc, vis_idx, atm in pending
                    if mem[i].ring_index == r and atm is not None
                ]
                if not parts:
                    continue
                res: BeamResult = _evaluate_beam_backend(
                    np.concatenate([a.range_m[v] for _, _, a, v in parts]),
                    np.concatenate([a.elevation_rad[v] for _, _, a, v in parts]),
                    beam_by_ring[r],
                    backend,
                    np.concatenate([a.t_emit_tdb_s[v] for _, _, a, v in parts]),
                )
                pos = 0
                for i, j, _acc, vis_idx in parts:
                    sl = slice(pos, pos + vis_idx.size)
                    beams[(i, j)] = _CaptureSlice(
                        capture_fraction=res.capture_fraction[sl],
                        peak_irradiance_per_W_m2=res.peak_irradiance_per_W_m2[sl],
                        pointing_loss=res.pointing_loss[sl],
                    )
                    pos += vis_idx.size
        timings["beam"] = timings.get("beam", 0.0) + time.perf_counter() - t_a

        # ---- single-link conditions and budgets ---------------------------------------
        t_a = time.perf_counter()
        for i, j, acc, vis_idx, atm in pending:
            illum, pt, models = state[i]
            c, link, _available = site_link_budget(
                mem[i].config,
                access=acc,
                illumination=illum,
                power_thermal=pt,
                models=models,
                receiver=receivers[j],
                beam_epoch_index=vis_idx,
                atmosphere=atm,
                beam=beams.get((i, j)),
                atmosphere_settings=atm_settings,
            )
            # The receiver limits belong to the site sum (stage B): drop the single-beam flag.
            code[:, i, j] = c & ~np.uint32(int(Unavailability.RECEIVER_SATURATION))
            ok = link.transmitting
            captured[:, i, j] = np.where(ok, link.captured_W, 0.0)
            atm_w[:, i, j] = np.where(ok, link.atm_W, 0.0)
            density[:, i, j] = np.where(ok, link.peak_power_density_W_m2, 0.0)
        timings["link_budget"] = timings.get("link_budget", 0.0) + time.perf_counter() - t_a

    if n_pv_unconverged:
        log.append(
            RunLogEntry(
                "WARNING",
                "solar_thermal",
                f"PV temperature solve not converged at {n_pv_unconverged} satellite-epochs",
            )
        )
    log.append(
        RunLogEntry(
            "INFO" if n_ledger_open == 0 else "WARNING",
            "solar_thermal",
            f"energy ledgers closed for {n - n_ledger_open} of {n} satellites",
        )
    )
    n_fb = int(lt_fallbacks.sum())
    log.append(
        RunLogEntry(
            "WARNING" if n_fb else "INFO",
            "access",
            f"light time: {n_fb} brentq fallbacks over {n} satellites x {n_s} sites",
        )
    )
    lam = np.array([p.ecliptic_longitude_rad for p in phases], dtype=np.float64)
    rel = _wrap_pi(lam - (lam_earth if lam_earth is not None else 0.0))
    return LinkSet(
        config_hash=config_hash(config),
        satellite_index=idx,
        epochs_tdb_s=np.asarray(t, dtype=np.float64),
        reason_code=code,
        captured_W=captured,
        atm_W=atm_w,
        density_W_m2=density,
        range_m=range_m,
        elevation_rad=elev,
        pv_electric_W=sat_arrays["pv_electric_W"],
        incident_solar_W=sat_arrays["incident_solar_W"],
        aperture_W=sat_arrays["aperture_W"],
        dc_to_tx_W=sat_arrays["dc_to_tx_W"],
        beam_W=sat_arrays["beam_W"],
        illumination_fraction=sat_arrays["illumination_fraction"],
        thermal_feasible=thermal_ok,
        radiator_area_m2=sat_arrays["radiator_area_m2"],
        pv_temperature_K=sat_arrays["pv_temperature_K"],
        sun_distance_m=sat_arrays["sun_distance_m"],
        true_anomaly_rad=np.array([p.true_anomaly_rad for p in phases], dtype=np.float64),
        ecliptic_longitude_rad=lam,
        earth_relative_longitude_rad=rel,
        orbit_drift_max_m=drift_max,
        orbit_energy_rel_drift=energy,
        orbit_nfev=nfev,
        light_time_fallbacks=lt_fallbacks,
        itu_r_versions=itu_versions,
        itu_r_polarization_tilt_deg=itu_tilt,
        source_ids=source_ids,
        backend=backend,
        timings=tuple(StageTiming(stage=k, wall_s=v) for k, v in timings.items()),
        log=tuple(log),
    )


def merge_link_sets(sets: Sequence[LinkSet], n_satellites: int) -> LinkSet:
    """Concatenate the link sets of array tasks (satellite axis) into one covering
    ``0 .. n_satellites - 1``.

    Raises:
        ValueError: different configuration hashes, epochs, beam backends or code provenance
            (:attr:`LinkSet.code_provenance`), overlapping or missing satellites.
    """
    if not sets:
        raise ValueError("no link sets to merge")
    ordered = sorted(sets, key=lambda s: int(s.satellite_index[0]) if s.n_satellites else -1)
    ordered = [s for s in ordered if s.n_satellites]
    h = {s.config_hash for s in ordered}
    if len(h) != 1:
        raise ValueError(f"link sets come from different configurations: {sorted(h)}")
    t = ordered[0].epochs_tdb_s
    for s in ordered[1:]:
        if not np.array_equal(s.epochs_tdb_s, t):
            raise ValueError("link sets have different epochs")
    backends = sorted({s.backend for s in ordered})
    if len(backends) != 1:
        raise ValueError(f"link sets were computed with different beam backends: {backends}")
    for s in ordered[1:]:
        if s.code_provenance != ordered[0].code_provenance:
            raise ValueError(
                "link sets were computed by different code (git commit, source-tree hash or "
                "package versions differ); recompute every chunk with one code version"
            )
    idx = np.concatenate([s.satellite_index for s in ordered])
    if not np.array_equal(idx, np.arange(n_satellites)):
        missing = sorted(set(range(n_satellites)) - set(idx.tolist()))
        raise ValueError(
            f"link sets do not cover satellites 0..{n_satellites - 1} exactly once "
            f"(missing {missing[:20]}{'...' if len(missing) > 20 else ''})"
        )
    first = ordered[0]
    kw: dict[str, object] = {}
    for name in first.LINK_FIELDS + first.SATELLITE_FIELDS:
        kw[name] = np.concatenate([getattr(s, name) for s in ordered], axis=1)
    for name in first.SCALAR_FIELDS:
        kw[name] = np.concatenate([getattr(s, name) for s in ordered])
    timing: dict[str, float] = {}
    for s in ordered:
        for st in s.timings:
            timing[st.stage] = timing.get(st.stage, 0.0) + st.wall_s
    itu = next((s.itu_r_versions for s in ordered if s.itu_r_versions), ())
    tilt = next(
        (s.itu_r_polarization_tilt_deg for s in ordered if s.itu_r_polarization_tilt_deg), None
    )
    return LinkSet(
        config_hash=first.config_hash,
        satellite_index=idx,
        epochs_tdb_s=t,
        itu_r_versions=itu,
        itu_r_polarization_tilt_deg=tilt,
        source_ids=first.source_ids,
        backend=first.backend,
        timings=tuple(StageTiming(stage=k, wall_s=v) for k, v in timing.items()),
        log=tuple(e for s in ordered for e in s.log),
        code_provenance=first.code_provenance,
        **kw,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------------------------
# Stage B: scheduling and network power
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ConstellationRunResult:
    """Everything a constellation run produced (``T`` epochs, ``N`` satellites, ``S`` sites).

    Attributes:
        config: The constellation configuration.
        config_hash: Its hash.
        grid: Output time grid.
        members: The satellites.
        links: Stage-A link set of all satellites.
        capacities: Capacity of every site.
        schedule: Scheduler output.
        link_code: Final link reason codes: single-link bits | scheduler bits (unscheduled
            feasible links) | ``RECEIVER_SATURATION`` (scheduled links at a saturated site),
            (T, N, S), ``uint32``.
        site_incident_W: ``sum P_captured`` of the scheduled beams per site, (T, S) [W].
        site_accepted_W: Accepted power per site, (T, S) [W].
        site_receiver_W: ``P_receiver`` per site, (T, S) [W].
        site_grid_W: ``P_grid`` per site, (T, S) [W].
        network_grid_W: ``sum_j P_grid,j``, (T,) [W].
        satellite_delivered_W: Pro-rata share of each satellite in its site's ``P_grid``,
            (T, N) [W] (per-satellite summaries only).
        satellite_reason_index: Index into :data:`LINK_REASONS` of the reason a satellite
            delivers nothing, or :data:`DELIVERING`, (T, N), ``int8``.
        unconstrained_grid_W: Unconstrained baseline (module docstring), (T,) [W].
        satellite_unconstrained_W: Each satellite's best feasible link times ``eta_rx
            eta_grid`` (no site capacity), (T, N) [W].
        reception: Reception-time reconstruction (:class:`ReceptionTimeAggregates`).
        receiver_conversion_efficiency: ``eta_receiver_conversion`` [1].
        receiver_grid_efficiency: ``eta_grid_conditioning`` [1].
        backend: Beam backend.
        kernel_provenance: Verified kernel files.
        source_ids: Ephemeris / Earth-orientation data ids.
        itu_r_versions: ITU-R recommendations used, or empty.
        itu_r_polarization_tilt_deg: Rain polarisation tilt, or ``None``.
        rng_state_unchanged: The Generator was not drawn from (no ``random`` ring).
        timings: Wall time per stage.
        log: Diagnostics.
        window_end_utc: UTC of the last epoch (label only).
        stage_a_code_provenance: Code provenance of merged array-task link files, or ``None``
            (stage A ran in this process).
    """

    config: RunConfig
    config_hash: str
    grid: TimeGrid
    members: tuple[ConstellationMember, ...]
    links: LinkSet
    capacities: tuple[SiteCapacity, ...]
    schedule: ScheduleTimeseries
    link_code: NDArray[np.uint32]
    site_incident_W: Float64Array
    site_accepted_W: Float64Array
    site_receiver_W: Float64Array
    site_grid_W: Float64Array
    network_grid_W: Float64Array
    satellite_delivered_W: Float64Array
    satellite_reason_index: NDArray[np.int8]
    unconstrained_grid_W: Float64Array
    satellite_unconstrained_W: Float64Array
    reception: ReceptionTimeAggregates
    receiver_conversion_efficiency: float
    receiver_grid_efficiency: float
    backend: str
    kernel_provenance: tuple[KernelProvenance, ...]
    source_ids: tuple[str, ...]
    itu_r_versions: tuple[str, ...]
    itu_r_polarization_tilt_deg: float | None
    rng_state_unchanged: bool
    timings: tuple[StageTiming, ...] = field(repr=False)
    log: tuple[RunLogEntry, ...] = field(repr=False)
    window_end_utc: str | None = None
    stage_a_code_provenance: dict[str, Any] | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.grid)

    @property
    def n_satellites(self) -> int:
        """Number of satellites."""
        return len(self.members)

    @property
    def n_sites(self) -> int:
        """Number of receiver sites."""
        return len(self.capacities)


def link_reason_index(codes: NDArray[np.uint32]) -> NDArray[np.int8]:
    """Index into :data:`LINK_REASONS` of the first reason set per element, ``-1`` if none."""
    c = np.asarray(codes, dtype=np.uint32)
    out = np.full(c.shape, -1, dtype=np.int8)
    for k in range(len(LINK_REASONS) - 1, -1, -1):
        out[(c & np.uint32(int(LINK_REASONS[k]))) != 0] = k
    return out


def satellite_reason_index(
    link_code: NDArray[np.uint32], assigned_site: NDArray[np.integer]
) -> NDArray[np.int8]:
    """Primary reason per (epoch, satellite) of a satellite that delivers nothing (PDF §11.4).

    Each link's reason is its first reason in :data:`LINK_REASONS` precedence
    (:func:`link_reason_index`); the satellite gets the reason of the link that got *furthest*
    through the chain, i.e. the **largest** link index over its sites (below the horizon of
    site A but in eclipse while visible from site B -> ``ECLIPSE``; one link refused for
    capacity and one geometrically blocked -> ``RECEIVER_CAPACITY``). A scheduled satellite is
    :data:`DELIVERING`.

    Args:
        link_code: Final link reason codes, (T, N, S).
        assigned_site: Site served per (epoch, satellite), ``-1`` if none, (T, N).

    Returns:
        Index into :data:`LINK_REASONS` or :data:`DELIVERING`, (T, N), ``int8``.

    Raises:
        RuntimeError: a non-delivering satellite-epoch without any reason.
    """
    link_idx = link_reason_index(link_code)
    out = np.max(link_idx, axis=2).astype(np.int8)
    assigned = np.asarray(assigned_site)
    out[assigned >= 0] = DELIVERING
    if np.any((out == -1) & (assigned < 0)):
        raise RuntimeError("unattributed non-delivering satellite epoch")
    return out


@dataclass(frozen=True, eq=False)
class ReceptionTimeAggregates:
    """Scheduled beams at each receiver in *reception* time (module docstring; a diagnostic of
    the emission-epoch aggregation approximation).

    The beam that satellite ``i`` emits towards site ``j`` at emission epoch ``t_k`` is held
    over the emission step and arrives over ``[t_k + tau_k, t_{k+1} + tau_{k+1})`` with
    ``tau = range / c`` of that satellite and site (last epoch: ``[t_k + tau_k, t_k + dt +
    tau_k)``; ``tau = 0`` with ``models.light_time = false``), carrying its ``P_captured`` and
    peak density of epoch ``k``. Reception slot ``k`` of a site is ``[t_k, t_{k+1})`` in
    reception time (the last slot is open-ended). Arrays have shape (T, S).

    Attributes:
        light_time_spread_s: ``max - min`` light time of the beams scheduled to the site at
            emission epoch ``k`` (0 for fewer than two beams) [s].
        max_beams: Largest number of simultaneously arriving beams during slot ``k``.
        max_density_W_m2: Largest summed peak irradiance during slot ``k`` [W m^-2].
        incident_energy_J: Integral of the summed arriving ``P_captured`` over slot ``k`` [J].
        accepted_energy_J: Integral of ``min(sum, L_j)`` over slot ``k`` [J].
        beam_overrun_s: Time of slot ``k`` with more arriving beams than the site's slots [s].
        density_overrun_s: Time of slot ``k`` with the summed density above ``D_j`` [s].
    """

    light_time_spread_s: Float64Array
    max_beams: NDArray[np.int64]
    max_density_W_m2: Float64Array
    incident_energy_J: Float64Array
    accepted_energy_J: Float64Array
    beam_overrun_s: Float64Array
    density_overrun_s: Float64Array


def reception_time_aggregates(
    epochs_tdb_s: Float64Array,
    step_s: float,
    assigned_site: NDArray[np.integer],
    light_time_s: Float64Array,
    captured_W: Float64Array,
    density_W_m2: Float64Array,
    capacities: Sequence[SiteCapacity],
) -> ReceptionTimeAggregates:
    """:class:`ReceptionTimeAggregates` of a schedule (piecewise-constant sweep per site).

    Args:
        epochs_tdb_s: Emission epochs, (T,) [TDB s].
        step_s: Output step (hold length of the last epoch) [s].
        assigned_site: Site of each satellite per epoch, ``-1`` if idle, (T, N).
        light_time_s: Light time per link, (T, N, S) [s].
        captured_W: ``P_captured`` per link, (T, N, S) [W].
        density_W_m2: Peak irradiance per link, (T, N, S) [W m^-2].
        capacities: Site capacities (``S``).
    """
    t = np.asarray(epochs_tdb_s, dtype=np.float64)
    assigned = np.asarray(assigned_site, dtype=np.int64)
    n_t = t.size
    n_s = len(capacities)
    t_next = np.append(t[1:], t[-1] + step_s) if n_t else t
    spread = np.zeros((n_t, n_s))
    max_beams = np.zeros((n_t, n_s), dtype=np.int64)
    max_dens = np.zeros((n_t, n_s))
    inc_e = np.zeros((n_t, n_s))
    acc_e = np.zeros((n_t, n_s))
    beam_over = np.zeros((n_t, n_s))
    dens_over = np.zeros((n_t, n_s))
    for j, cap in enumerate(capacities):
        kk, ii = np.nonzero(assigned == j)
        if kk.size == 0:
            continue
        tau = light_time_s[kk, ii, j]
        lo = np.full(n_t, np.inf)
        hi = np.full(n_t, -np.inf)
        np.minimum.at(lo, kk, tau)
        np.maximum.at(hi, kk, tau)
        has = np.isfinite(lo)
        spread[has, j] = hi[has] - lo[has]
        last = kk + 1 >= n_t
        k_next = np.where(last, kk, kk + 1)
        start = t[kk] + tau
        end = t_next[kk] + np.where(last, tau, light_time_s[k_next, ii, j])
        p = captured_W[kk, ii, j]
        d = density_W_m2[kk, ii, j]
        bounds = t[1:]
        times = np.concatenate([start, end, bounds])
        dn = np.concatenate([np.ones(kk.size), -np.ones(kk.size), np.zeros(bounds.size)]).astype(
            np.int64
        )
        dp = np.concatenate([p, -p, np.zeros(bounds.size)])
        dd = np.concatenate([d, -d, np.zeros(bounds.size)])
        order = np.lexsort((dn, times))  # at equal times: ends, slot bounds, then starts
        times, dn, dp, dd = times[order], dn[order], dp[order], dd[order]
        b = np.cumsum(dn)
        on = b > 0
        s_w = np.where(on, np.cumsum(dp), 0.0)
        s_d = np.where(on, np.cumsum(dd), 0.0)
        dur = np.append(np.diff(times), 0.0)
        slot = np.clip(np.searchsorted(t, times, side="right") - 1, 0, n_t - 1)
        seg = dur > 0.0
        sl, bb, sw, sd, du = slot[seg], b[seg], s_w[seg], s_d[seg], dur[seg]
        col_beams = np.zeros(n_t, dtype=np.int64)
        col_dens = np.zeros(n_t)
        np.maximum.at(col_beams, sl, bb)
        np.maximum.at(col_dens, sl, sd)
        max_beams[:, j] = col_beams
        max_dens[:, j] = col_dens
        inc_e[:, j] = np.bincount(sl, weights=sw * du, minlength=n_t)
        acc_e[:, j] = np.bincount(
            sl, weights=np.minimum(sw, cap.max_accepted_W) * du, minlength=n_t
        )
        if cap.max_beams is not None:
            beam_over[:, j] = np.bincount(sl, weights=du * (bb > cap.max_beams), minlength=n_t)
        if cap.max_power_density_W_m2 is not None:
            over = sd > cap.max_power_density_W_m2 * (1.0 + 1e-12)
            dens_over[:, j] = np.bincount(sl, weights=du * over, minlength=n_t)
    return ReceptionTimeAggregates(
        light_time_spread_s=spread,
        max_beams=max_beams,
        max_density_W_m2=max_dens,
        incident_energy_J=inc_e,
        accepted_energy_J=acc_e,
        beam_overrun_s=beam_over,
        density_overrun_s=dens_over,
    )


def duty_cycle_from_config(config: RunConfig) -> DutyCycle | None:
    """:class:`~helioswarm.scheduler.DutyCycle` of the scheduler section (``None`` if unset)."""
    assert config.constellation is not None
    sch = config.constellation.scheduler
    if sch.duty_cycle_max is None or sch.duty_cycle_window_s is None:
        return None
    return DutyCycle.from_limits(
        sch.duty_cycle_max, sch.duty_cycle_window_s, config.time.output_step_s
    )


def finish_constellation(
    config: RunConfig,
    members: Sequence[ConstellationMember],
    links: LinkSet,
    *,
    grid: TimeGrid,
    kernel_provenance: tuple[KernelProvenance, ...],
    rng_state_unchanged: bool,
    window_end_utc: str | None = None,
    extra_log: Sequence[RunLogEntry] = (),
    extra_timings: Sequence[StageTiming] = (),
) -> ConstellationRunResult:
    """Stage B (module docstring): schedule the feasible links and sum the network power.

    Args:
        config: The constellation configuration.
        members: All satellites (:func:`constellation_members`).
        links: Stage-A link set covering every member.
        grid: Output grid (must match ``links.epochs_tdb_s``).
        kernel_provenance: Kernel provenance for the manifest.
        rng_state_unchanged: Whether the run's Generator was drawn from.
        window_end_utc: UTC label of the last epoch.
        extra_log: Log entries to prepend (configuration warnings).
        extra_timings: Stage timings to prepend.
    """
    if config.constellation is None:
        raise ValueError("configuration has no constellation section")
    if links.config_hash != config_hash(config):
        raise ValueError("the link set was computed for a different configuration")
    if links.n_satellites != len(members) or not np.array_equal(
        links.satellite_index, np.arange(len(members))
    ):
        raise ValueError("the link set must cover every satellite exactly once")
    if not np.array_equal(links.epochs_tdb_s, grid.epochs_tdb_s):
        raise ValueError("the link set epochs differ from the output grid")
    caps = site_capacities(config)
    sch_cfg = config.constellation.scheduler
    t0 = time.perf_counter()
    feasible = links.feasible
    schedule = schedule_timeseries(
        feasible,
        links.captured_W,
        links.density_W_m2,
        caps,
        policy=sch_cfg.policy,
        duty_cycle=duty_cycle_from_config(config),
    )
    t_sched = time.perf_counter() - t0
    eta_rx = config.receiver.conversion_efficiency
    eta_grid = config.receiver.grid_efficiency
    accepted = schedule.site_accepted_W
    p_rx = accepted * eta_rx
    p_grid = p_rx * eta_grid
    net = np.zeros(p_grid.shape[0])
    for j in range(p_grid.shape[1]):
        net = net + p_grid[:, j]

    # Final link codes.
    code = links.reason_code.copy()
    dec = schedule.decision
    for d, bit in _DECISION_BIT:
        code[dec == d] |= np.uint32(bit)
    saturated_site = schedule.site_incident_W > np.array([c.max_accepted_W for c in caps])
    sel = dec == int(LinkDecision.SELECTED)
    code[sel & saturated_site[:, None, :]] |= np.uint32(int(Unavailability.RECEIVER_SATURATION))
    if np.any(sel & ~feasible) or np.any((dec != NOT_A_CANDIDATE) & ~feasible):
        raise RuntimeError("the scheduler selected an infeasible link")

    # Per-satellite share and reason.
    assigned = schedule.assigned_site.astype(np.int64)
    n_t, n_sat = assigned.shape
    rows = np.arange(n_t)[:, None]
    site_safe = np.where(assigned >= 0, assigned, 0)
    cap_i = np.where(
        assigned >= 0, links.captured_W[rows, np.arange(n_sat)[None, :], site_safe], 0.0
    )
    inc = schedule.site_incident_W[rows, site_safe]
    grid_j = p_grid[rows, site_safe]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(
            (assigned >= 0) & (inc > 0.0), grid_j * cap_i / np.where(inc > 0, inc, 1.0), 0.0
        )
    sat_reason = satellite_reason_index(code, assigned)

    # Unconstrained baseline: each satellite on its best feasible link, no site capacity.
    best_link_W = np.max(np.where(feasible, links.captured_W, 0.0), axis=2)
    unconstrained_sat = best_link_W * eta_rx * eta_grid
    unconstrained = np.zeros(n_t)
    for i in range(n_sat):
        unconstrained = unconstrained + unconstrained_sat[:, i]

    # Reception-time reconstruction (named approximation diagnostic).
    tau = (
        links.range_m / SPEED_OF_LIGHT_M_S
        if config.models.light_time
        else np.zeros_like(links.range_m)
    )
    reception = reception_time_aggregates(
        links.epochs_tdb_s,
        config.time.output_step_s,
        schedule.assigned_site,
        tau,
        links.captured_W,
        links.density_W_m2,
        caps,
    )

    n_retarget = int(schedule.retarget_events.sum())
    methods = np.bincount(schedule.method, minlength=5).tolist()
    log = list(extra_log) + list(links.log)
    log.append(
        RunLogEntry(
            "INFO",
            "scheduler",
            f"policy {sch_cfg.policy}; epochs by method (empty, best_link, greedy, assignment, "
            f"milp) = {methods}; MILP round-off repairs {int(schedule.repaired.sum())}; "
            f"retarget events {n_retarget} (instantaneous retargeting: named approximation)",
        )
    )
    log.append(
        RunLogEntry(
            "INFO",
            "scheduler",
            "incoherent summation: site powers are sums of independent beam powers; "
            "no coherent combining",
        )
    )
    multi = schedule.site_beams >= 2
    spread_max = float(np.max(reception.light_time_spread_s)) if multi.any() else 0.0
    log.append(
        RunLogEntry(
            "INFO",
            "scheduler",
            "emission-epoch aggregation (named approximation): beams summed at a site per "
            f"emission epoch; light-time spread over {int(np.count_nonzero(multi))} multi-beam "
            f"site-epochs max {spread_max:.1f} s (step {config.time.output_step_s} s); in "
            f"reception time max beams per site {reception.max_beams.max(axis=0).tolist()}, "
            f"beam-slot overrun {float(reception.beam_overrun_s.sum()):.1f} s, density "
            f"overrun {float(reception.density_overrun_s.sum()):.1f} s",
        )
    )
    timings = (
        *extra_timings,
        *links.timings,
        StageTiming(stage="scheduler", wall_s=t_sched),
    )
    return ConstellationRunResult(
        config=config,
        config_hash=links.config_hash,
        grid=grid,
        members=tuple(members),
        links=links,
        capacities=caps,
        schedule=schedule,
        link_code=code,
        site_incident_W=schedule.site_incident_W,
        site_accepted_W=accepted,
        site_receiver_W=p_rx,
        site_grid_W=p_grid,
        network_grid_W=net,
        satellite_delivered_W=share,
        satellite_reason_index=sat_reason,
        unconstrained_grid_W=unconstrained,
        satellite_unconstrained_W=unconstrained_sat,
        reception=reception,
        receiver_conversion_efficiency=eta_rx,
        receiver_grid_efficiency=eta_grid,
        backend=links.backend,
        kernel_provenance=kernel_provenance,
        source_ids=links.source_ids,
        itu_r_versions=links.itu_r_versions,
        itu_r_polarization_tilt_deg=links.itu_r_polarization_tilt_deg,
        rng_state_unchanged=rng_state_unchanged,
        timings=timings,
        log=tuple(log),
        window_end_utc=window_end_utc,
        stage_a_code_provenance=links.code_provenance,
    )


def config_log(config: RunConfig) -> tuple[RunLogEntry, ...]:
    """Warnings for the named approximations of the configuration."""
    return tuple(
        RunLogEntry("WARNING", "config", "named approximation in force: " + a.describe())
        for a in approximations_in_force(config)
    )


def run_constellation(
    config: RunConfig,
    *,
    ephemeris: EphemerisProvider,
    kernels: LoadedKernels,
    rng: np.random.Generator,
    earth_orientation: EarthOrientation | None = None,
    backend: Backend = "numpy",
) -> ConstellationRunResult:
    """Complete constellation run in one process (stages A and B; module docstring).

    Args:
        config: Configuration with a ``constellation`` section.
        ephemeris: Ephemeris provider.
        kernels: The configured kernel set.
        rng: The run's Generator seeded from ``project.seed`` (``random`` ring phases).
        earth_orientation: Earth-fixed -> J2000 model (default SPICE ``IAU_EARTH``).
        backend: Beam backend (:func:`evaluate_links`).
    """
    state = rng.bit_generator.state
    members = constellation_members(config, rng)
    unchanged = rng.bit_generator.state == state
    links = evaluate_links(
        config,
        members,
        ephemeris=ephemeris,
        kernels=kernels,
        earth_orientation=earth_orientation,
        backend=backend,
    )
    grid = time_grid_from_config(config.time, kernels)
    return finish_constellation(
        config,
        members,
        links,
        grid=grid,
        kernel_provenance=tuple(kernels.provenance),
        rng_state_unchanged=bool(unchanged),
        window_end_utc=tdb_s_to_utc(float(grid.epochs_tdb_s[-1]), kernels, decimals=3),
        extra_log=config_log(config),
    )


__all__ = [
    "DELIVERING",
    "LINK_REASONS",
    "SCHEDULER_BITS",
    "ConstellationMember",
    "ConstellationRunResult",
    "LinkSet",
    "ReceptionTimeAggregates",
    "chunk_indices",
    "config_log",
    "constellation_members",
    "duty_cycle_from_config",
    "evaluate_links",
    "finish_constellation",
    "link_reason_index",
    "merge_link_sets",
    "reception_time_aggregates",
    "run_constellation",
    "satellite_reason_index",
    "site_capacities",
]
