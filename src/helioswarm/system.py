"""End-to-end single-satellite, single-receiver run (PDF §11.1 vertical slice).

:func:`run_single_satellite` orchestrates the validated M1-M5 models on the output time grid
(every stage vectorised over epochs) and joins them in the M6 link budget:

1. time grid (UTC -> TDB, SPICE) and orbit: L1 test-particle propagation
   (:func:`~helioswarm.propagate.propagate_l1_from_config`) or the L0 two-body orbit
   (:func:`~helioswarm.propagate.l0_reference_trajectory`), J2000 / SSB;
2. satellite illumination (conical shadow, retarded Sun; M3);
3. solar collection, PV temperature, on-board chain, radiator and energy ledger (M4) at the
   transmitter operating point;
4. access to the receiver: retarded-time targeting, elevation, Earth/Sun/Moon occultation, Sun
   avoidance (M3);
5. atmosphere (terminal segment; M5) and beam capture with pointing statistics (M5) at the
   epochs with geometric access;
6. availability conditions and the PDF §10.1/§10.4 link budget
   (:func:`~helioswarm.link_budget.evaluate_link_budget`).

Unavailability reasons (bitmask :class:`Unavailability`, one ``uint32`` per epoch) are recorded
*all at once* (several can hold together). The blocking reasons make the link unavailable and
``P_grid = 0``; the limiting flags (receiver saturation, partial eclipse) mark epochs whose
power is reduced but whose link is available. Availability = no blocking bit set.

Operating policies (documented in docs/model_assumptions.md, M6):

* **Thermal infeasibility** (``ThermalResult.feasible`` false: PV plate above its limit or a
  configured radiator above its limit, M4) -> the transmitter is not operated (reason
  ``THERMAL``). The schema defines no derating policy, so none is applied. The thermal state is
  evaluated at the full operating point (transmitter on); the off-state temperature is not
  modelled.
* **Eclipse**: zero illumination (umbra) -> ``ECLIPSE`` (no power to transmit); a partial
  eclipse only reduces ``P_PV`` and is flagged ``PARTIAL_ECLIPSE``.
* **Pointing**: ``transmitter.min_pointing_efficiency`` (optional) -> ``POINTING`` where
  ``BeamResult.pointing_loss`` (expected / perfect-pointing capture) is below it (or undefined).
* **Weather**: a hard optical outage of the scenario (zenith transmittance 0) -> ``WEATHER``.
* **Power density**: beam abort (:mod:`helioswarm.link_budget`) -> ``POWER_DENSITY_ABORT``.
* **Receiver saturation**: the receiver input (or heat) limit clips ``P_captured`` ->
  ``RECEIVER_SATURATION`` (limiting flag; the link delivers the clipped power).

The run is deterministic. It contains no stochastic step: pointing jitter enters through the
deterministic quadrature of the Rice offset law (M5), not by sampling. The explicit
``numpy.random.Generator`` argument is the run's single source of randomness (PDF §14.1); the
result records whether it was used (it is not, in M6), so a later stochastic model cannot draw
from an implicit global state.

Only one receiver site is supported here: several sites would require receiver scheduling to
avoid double counting (PDF §10.4, §11.2; milestone M7), so a configuration with more than one
site is rejected rather than summed.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from helioswarm.atmosphere import (
    AtmosphereResult,
    AtmosphereSettings,
    atmosphere_settings_from_config,
    atmospheric_transmission,
)
from helioswarm.beam import BeamResult, BeamSettings, beam_settings_from_config, evaluate_beam
from helioswarm.beam_table import (
    BeamTableDiagnostics,
    beam_table_settings_from_config,
    evaluate_beam_table,
)
from helioswarm.config import RunConfig, SiteConfig, approximations_in_force, config_hash
from helioswarm.ephemeris import EphemerisProvider
from helioswarm.frames import CartesianTrajectory
from helioswarm.ground import EarthOrientation, GroundSite, SpiceEarthOrientation
from helioswarm.kernels import KernelProvenance, LoadedKernels
from helioswarm.link_budget import (
    LinkBudgetResult,
    ReceiverModel,
    evaluate_link_budget,
    receiver_model_for_site,
)
from helioswarm.propagate import (
    L1_MODEL,
    L1Diagnostics,
    check_kernel_set,
    l0_reference_trajectory,
    propagate_l1_from_config,
)
from helioswarm.thermal import (
    PowerThermalModels,
    PowerThermalResult,
    evaluate_power_thermal,
    power_thermal_from_illumination,
    power_thermal_models_from_config,
)
from helioswarm.timescale import TimeGrid, tdb_s_to_utc, time_grid_from_config, utc_to_tdb_s
from helioswarm.visibility import (
    AccessResult,
    AccessSummary,
    IlluminationResult,
    OcculterRadii,
    access_settings_from_config,
    compute_access,
    orbit_drift_m,
    satellite_illumination,
    summarize_access,
)

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
UInt32Array = NDArray[np.uint32]

Backend = Literal["numpy", "numpy_batch", "jax"]
"""Numerical backend of the beam-capture kernel: ``numpy`` (M5 reference, per epoch),
``numpy_batch`` or ``jax`` (batched kernel of :mod:`helioswarm.capture_batch` on the CPU or
compiled by JAX; Airy far field only)."""


class BeamCapture(Protocol):
    """The beam quantities the link budget needs at the epochs with access (shape (M,)):
    :class:`~helioswarm.beam.BeamResult` satisfies it."""

    @property
    def capture_fraction(self) -> Float64Array:
        """Expected capture fraction [1]."""
        ...

    @property
    def peak_irradiance_per_W_m2(self) -> Float64Array:
        """Peak irradiance per watt [m^-2]."""
        ...

    @property
    def pointing_loss(self) -> Float64Array:
        """Expected / perfect-pointing capture [1]."""
        ...


_T = TypeVar("_T")


class Unavailability(enum.IntFlag):
    """Per-epoch link condition bits (``reason_code``; PDF §10.4, §10.5, §11.4).

    Bits 0-9 are **blocking** (link unavailable, ``P_grid = 0``); bits 16-17 are **limiting**
    flags (power reduced, link available). Bits 10-12 are set only by the constellation
    scheduler (M7) on links that were feasible on their own but not scheduled; a
    single-satellite run never sets them (they are not in :data:`BLOCKING_REASONS`).
    """

    NONE = 0
    BELOW_MIN_ELEVATION = 1 << 0
    EARTH_OCCULTATION = 1 << 1
    SUN_OCCULTATION = 1 << 2
    MOON_OCCULTATION = 1 << 3
    SUN_AVOIDANCE = 1 << 4
    ECLIPSE = 1 << 5
    THERMAL = 1 << 6
    WEATHER = 1 << 7
    POINTING = 1 << 8
    POWER_DENSITY_ABORT = 1 << 9
    DUTY_CYCLE = 1 << 10
    RECEIVER_CAPACITY = 1 << 11
    SCHEDULER_NOT_SELECTED = 1 << 12
    RECEIVER_SATURATION = 1 << 16
    PARTIAL_ECLIPSE = 1 << 17


BLOCKING_REASONS: Final[tuple[Unavailability, ...]] = (
    Unavailability.BELOW_MIN_ELEVATION,
    Unavailability.EARTH_OCCULTATION,
    Unavailability.SUN_OCCULTATION,
    Unavailability.MOON_OCCULTATION,
    Unavailability.SUN_AVOIDANCE,
    Unavailability.ECLIPSE,
    Unavailability.THERMAL,
    Unavailability.WEATHER,
    Unavailability.POINTING,
    Unavailability.POWER_DENSITY_ABORT,
)
"""Blocking reasons in *precedence order*: the primary reason of an unavailable epoch is the
first one set (geometry before spacecraft state before link conditions)."""

LIMITING_FLAGS: Final[tuple[Unavailability, ...]] = (
    Unavailability.RECEIVER_SATURATION,
    Unavailability.PARTIAL_ECLIPSE,
)
"""Non-blocking flags (power reduced, link available)."""

BLOCKING_MASK: Final[int] = int(sum(int(r) for r in BLOCKING_REASONS))
"""Bitwise OR of every blocking reason."""


def reason_names(code: int) -> tuple[str, ...]:
    """Names of the bits set in one reason code (blocking first, in precedence order)."""
    return tuple(str(r.name) for r in (*BLOCKING_REASONS, *LIMITING_FLAGS) if int(code) & int(r))


def primary_reason(codes: UInt32Array) -> NDArray[np.str_]:
    """Primary (highest-precedence) blocking reason per epoch, or ``""`` if available."""
    c = np.asarray(codes, dtype=np.uint32)
    out = np.full(c.shape, "", dtype="<U24")
    unset = np.ones(c.shape, dtype=bool)
    for r in BLOCKING_REASONS:
        hit = unset & ((c & np.uint32(int(r))) != 0)
        out[hit] = str(r.name)
        unset &= ~hit
    return out


@dataclass(frozen=True)
class StageTiming:
    """Wall time of one pipeline stage.

    Attributes:
        stage: Stage name.
        wall_s: Elapsed wall-clock time [s] (``time.perf_counter``).
    """

    stage: str
    wall_s: float


@dataclass(frozen=True)
class RunLogEntry:
    """One message for ``logs/run.log`` (warnings, integrator and solver diagnostics).

    Attributes:
        level: ``INFO`` or ``WARNING``.
        stage: Pipeline stage that produced it.
        message: Text.
    """

    level: Literal["INFO", "WARNING"]
    stage: str
    message: str


@dataclass(frozen=True)
class ReferenceLink:
    """The rated (reference) link that defines the capacity-factor denominator (PDF §10.5).

    ``rated_power_W`` is the ``P_grid`` the configured system delivers at reference
    conditions: satellite at the window-mean heliocentric distance, collector at normal
    incidence and fully illuminated, receiver at the window-mean range with the beam at zenith
    (elevation 90 deg: full receiver projection, air mass 1), the configured pointing errors,
    and the receiver input/heat limits applied. Atmosphere at zenith: for the ``scenario``
    model the *clear-sky* scenario values replace the configured weather (``none``: 1); for
    ``itu_r`` the configured ITU-R gas + cloud + rain attenuation exceeded
    ``itu_r_exceedance_percent`` of the time is used (the ITU-R model has no separate
    clear-sky state, so the reference is *not* clear sky there). The run's operating policies
    apply to the reference exactly as to any epoch: if the reference operating point is
    thermally infeasible the transmitter is not operated, and if the reference beam exceeds the
    power-density limit it is aborted; in both cases ``rated_power_W = 0``, the capacity factor
    is undefined (``None``) and the flags below say which condition failed.

    Attributes:
        definition: The sentence above (written to summary.json).
        sun_distance_m: Reference heliocentric distance (window mean) [m].
        range_m: Reference satellite-receiver range (window mean over all epochs) [m].
        elevation_rad: Reference elevation (pi/2) [rad].
        pv_electric_W: ``P_PV`` at reference [W].
        aperture_W: ``P_aperture`` at reference [W].
        eta_atmosphere: Zenith atmospheric transmission at the reference weather [1].
        eta_capture: Expected capture at the reference range, zenith [1].
        rated_power_W: ``P_grid`` at reference [W].
        thermal_feasible: Thermal verdict at the reference operating point (``False`` ->
            transmitter off, ``rated_power_W = 0``).
        power_density_abort: The reference beam triggers the beam abort (``rated_power_W =
            0``).
    """

    definition: str
    sun_distance_m: float
    range_m: float
    elevation_rad: float
    pv_electric_W: float
    aperture_W: float
    eta_atmosphere: float
    eta_capture: float
    rated_power_W: float
    thermal_feasible: bool
    power_density_abort: bool


REFERENCE_DEFINITION: Final[str] = (
    "rated_power_W = P_grid delivered at reference conditions: satellite at the window-mean "
    "heliocentric distance, collector at normal incidence and fully illuminated, receiver at "
    "the window-mean satellite range with the beam at zenith (elevation 90 deg, full receiver "
    "projection, air mass 1), configured pointing errors, receiver limits applied; atmosphere "
    "at zenith: clear-sky scenario values for models.atmosphere=scenario (1 for none), the "
    "configured ITU-R gas+cloud+rain attenuation exceeded itu_r_exceedance_percent of the time "
    "for itu_r (not clear sky); the thermal and power-density policies apply (an infeasible or "
    "aborted reference gives rated_power_W = 0, reference_thermal_feasible / "
    "reference_power_density_abort say which); capacity_factor = mean(P_grid) / "
    "rated_power_W (null if rated_power_W = 0)"
)


@dataclass(frozen=True, eq=False)
class SystemRunResult:
    """Everything one single-satellite run produced (arrays on the output grid, shape (N,)).

    Attributes:
        config: The validated configuration.
        config_hash: :func:`helioswarm.config.config_hash` of ``config``.
        grid: Output time grid (TDB; emission epochs of the satellite).
        trajectory: Satellite states on the grid (J2000 / SSB).
        orbit_model: ``orbit.model``.
        l1_diagnostics: Integrator diagnostics (L1 runs) or ``None``.
        orbit_drift_m: ``|r - r_L0|`` per epoch for L1 runs (planets + SRP effect), else
            ``None`` [m].
        illumination: Satellite illumination (M3).
        power_thermal: Solar, PV, on-board chain, thermal and ledger (M4).
        access: Access geometry (M3) to the single receiver site.
        access_summary: PDF §7.3 diagnostics of ``access`` (incl. the hour-angle gap).
        beam_epoch_index: Grid indices at which the beam and atmosphere were evaluated (the
            epochs with geometric access).
        beam: Beam result on those epochs, or ``None`` if there were none.
        atmosphere: Atmosphere result on those epochs, or ``None``.
        link: Link budget on the full grid.
        reason_code: :class:`Unavailability` bitmask per epoch (``uint32``).
        available: No blocking reason set (equals ``link.transmitting``).
        reference: Rated (reference) link of the capacity factor.
        receiver: Receiver model used.
        backend: Beam-capture backend used.
        kernel_provenance: Verified kernel files (names, URLs, SHA-256).
        source_ids: Ephemeris and Earth-orientation data ids.
        itu_r_versions: ITU-R recommendations used (``itu_r`` atmosphere), else empty.
        itu_r_polarization_tilt_deg: Rain polarisation tilt used (``itu_r``), else ``None``.
        rng_state_unchanged: The supplied Generator was not drawn from (no stochastic step).
        timings: Wall time per stage.
        log: Warnings and diagnostics for ``logs/run.log``.
        window_end_utc: UTC of the last output epoch (SPICE ``et2utc``, milliseconds; for the
            manifest's simulation window only, never used in arithmetic).
        beam_table: Diagnostics of the Chebyshev beam table (``models.beam_evaluation =
            'chebyshev_table'``, M7b), else ``None``.
    """

    config: RunConfig
    config_hash: str
    grid: TimeGrid
    trajectory: CartesianTrajectory
    orbit_model: str
    l1_diagnostics: L1Diagnostics | None
    orbit_drift_m: Float64Array | None
    illumination: IlluminationResult
    power_thermal: PowerThermalResult
    access: AccessResult
    access_summary: AccessSummary
    beam_epoch_index: NDArray[np.int64]
    beam: BeamResult | None
    atmosphere: AtmosphereResult | None
    link: LinkBudgetResult
    reason_code: UInt32Array
    available: BoolArray
    reference: ReferenceLink
    receiver: ReceiverModel
    backend: Backend
    kernel_provenance: tuple[KernelProvenance, ...]
    source_ids: tuple[str, ...]
    itu_r_versions: tuple[str, ...]
    itu_r_polarization_tilt_deg: float | None
    rng_state_unchanged: bool
    timings: tuple[StageTiming, ...] = field(repr=False)
    log: tuple[RunLogEntry, ...] = field(repr=False)
    window_end_utc: str | None = None
    beam_table: BeamTableDiagnostics | None = None

    def __post_init__(self) -> None:
        n = len(self.grid)
        for name in ("reason_code", "available"):
            arr = np.array(getattr(self, name), copy=True)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if len(self.link) != n or len(self.access) != n or len(self.trajectory) != n:
            raise ValueError("link, access and trajectory must be on the output grid")
        if not np.array_equal(self.available, self.link.transmitting):
            raise ValueError("available must equal link.transmitting")
        if np.any(self.link.grid_W[~self.available] != 0.0):
            raise ValueError("P_grid must be 0 at unavailable epochs")

    def __len__(self) -> int:
        return len(self.grid)

    def expand(self, values: Float64Array, fill: float = math.nan) -> Float64Array:
        """Scatter a beam-epoch array onto the full grid (``fill`` elsewhere)."""
        out = np.full(len(self), fill, dtype=np.float64)
        out[self.beam_epoch_index] = values
        return out


def _timed(stage: str, timings: list[StageTiming], fn: Callable[[], _T]) -> _T:
    t0 = time.perf_counter()
    value = fn()
    timings.append(StageTiming(stage=stage, wall_s=time.perf_counter() - t0))
    return value


def _evaluate_beam_backend(
    range_m: Float64Array,
    elevation_rad: Float64Array,
    settings: BeamSettings,
    backend: Backend,
    epochs_tdb_s: Float64Array,
) -> BeamResult:
    if backend == "numpy":
        return evaluate_beam(range_m, elevation_rad, settings, epochs_tdb_s=epochs_tdb_s)
    if backend in ("jax", "numpy_batch"):
        from helioswarm.capture_batch import evaluate_beam_batched

        return evaluate_beam_batched(
            range_m,
            elevation_rad,
            settings,
            backend="jax" if backend == "jax" else "numpy",
            epochs_tdb_s=epochs_tdb_s,
        )
    raise ValueError(f"unknown backend {backend!r}; use 'numpy', 'numpy_batch' or 'jax'")


def _clear_sky(settings: AtmosphereSettings, config: RunConfig) -> AtmosphereSettings:
    atm = config.atmosphere
    return dataclasses.replace(
        settings,
        weather_scenario="clear",
        optical_zenith_transmittance=atm.optical_zenith_transmittance_clear,
        microwave_zenith_attenuation_dB=atm.microwave_zenith_attenuation_clear_dB,
    )


def reference_link(
    config: RunConfig,
    *,
    models: PowerThermalModels,
    beam_settings: BeamSettings,
    atmosphere_settings: AtmosphereSettings,
    receiver: ReceiverModel,
    sun_distance_m: float,
    range_m: float,
    backend: Backend = "numpy",
) -> ReferenceLink:
    """Rated link at reference conditions (:class:`ReferenceLink`, PDF §10.5 denominator).

    Args:
        config: Run configuration (receiver site for ``itu_r``).
        models: M4 model parameters.
        beam_settings: Beam settings.
        atmosphere_settings: Atmosphere settings (the ``scenario`` weather is replaced by
            clear sky; ``itu_r`` is used as configured).
        receiver: Receiver model.
        sun_distance_m: Reference heliocentric distance [m].
        range_m: Reference satellite-receiver range [m].
        backend: Beam backend.

    Returns:
        :class:`ReferenceLink`.
    """
    zenith = 0.5 * math.pi
    pt = evaluate_power_thermal(np.array([sun_distance_m]), 1.0, 1.0, models)
    atm = atmospheric_transmission(
        np.array([zenith]),
        _clear_sky(atmosphere_settings, config),
        site=config.receiver.sites[0],
    )
    beam = _evaluate_beam_backend(
        np.array([range_m]), np.array([zenith]), beam_settings, backend, np.array([0.0])
    )
    link = evaluate_link_budget(
        pt.solar.pv_electric_W,
        models.efficiencies,
        receiver,
        # The run's thermal policy applies to the reference too: infeasible -> not operated.
        link_enabled=np.asarray(pt.thermal.feasible, dtype=bool),
        eta_atmosphere=atm.transmittance,
        eta_capture=beam.capture_fraction,
        peak_irradiance_per_W_m2=beam.peak_irradiance_per_W_m2,
    )
    return ReferenceLink(
        definition=REFERENCE_DEFINITION,
        sun_distance_m=float(sun_distance_m),
        range_m=float(range_m),
        elevation_rad=zenith,
        pv_electric_W=float(pt.solar.pv_electric_W[0]),
        aperture_W=float(pt.chain.aperture_W[0]),
        eta_atmosphere=float(atm.transmittance[0]),
        eta_capture=float(beam.capture_fraction[0]),
        rated_power_W=float(link.grid_W[0]),
        thermal_feasible=bool(pt.thermal.feasible[0]),
        power_density_abort=bool(link.power_density_abort[0]),
    )


def atmosphere_at_access(
    access: AccessResult, settings: AtmosphereSettings, site: SiteConfig
) -> tuple[NDArray[np.int64], AtmosphereResult | None]:
    """Atmospheric transmission (M5) at the epochs with geometric access to one site.

    Returns:
        ``(beam_epoch_index, atmosphere)``: grid indices with ``access.visible`` and the
        transmission there (``None`` if there is no such epoch).

    Raises:
        RuntimeError: the atmosphere model is invalid at an epoch with access (the validator
            prevents this: the minimum elevation is above the model floor).
    """
    vis_idx = np.flatnonzero(access.visible).astype(np.int64)
    if not vis_idx.size:
        return vis_idx, None
    atm = atmospheric_transmission(access.elevation_rad[vis_idx], settings, site=site)
    if not np.all(atm.in_validity):
        raise RuntimeError(
            "atmosphere model invalid at an epoch with access (the configuration validator "
            "should have prevented this: min elevation below the model floor)"
        )
    return vis_idx, atm


def site_link_budget(
    config: RunConfig,
    *,
    access: AccessResult,
    illumination: IlluminationResult,
    power_thermal: PowerThermalResult,
    models: PowerThermalModels,
    receiver: ReceiverModel,
    beam_epoch_index: NDArray[np.int64],
    atmosphere: AtmosphereResult | None,
    beam: BeamCapture | None,
    atmosphere_settings: AtmosphereSettings,
) -> tuple[UInt32Array, LinkBudgetResult, BoolArray]:
    """Availability conditions and link budget of one satellite-to-site link on the grid.

    Sets the :class:`Unavailability` bits (geometry and occultation from ``access``; eclipse
    and partial eclipse from ``illumination``; thermal; weather; pointing) as documented in the
    module docstring, evaluates the PDF §10.1/§10.4 budget
    (:func:`~helioswarm.link_budget.evaluate_link_budget`) on the epochs with no blocking bit,
    then adds the power-density abort and receiver-saturation bits. Shared by the
    single-satellite run and every satellite-site pair of a constellation (M7), so both use
    one implementation.

    Args:
        config: Configuration of the satellite (transmitter pointing condition).
        access: Access geometry to the site.
        illumination: Satellite illumination.
        power_thermal: Solar/thermal result (``P_PV``, thermal verdict).
        models: M4 model parameters (on-board efficiencies).
        receiver: Receiver model of the site.
        beam_epoch_index: Grid indices where ``atmosphere`` and ``beam`` were evaluated.
        atmosphere: Atmosphere at those epochs, or ``None`` (no access).
        beam: Beam capture at those epochs (:class:`BeamCapture`, e.g. a
            :class:`~helioswarm.beam.BeamResult`), or ``None``.
        atmosphere_settings: Atmosphere settings (hard optical outage).

    Returns:
        ``(reason_code, link, available)`` on the full grid.
    """
    n = len(access)
    vis_idx = beam_epoch_index
    code = np.zeros(n, dtype=np.uint32)

    def setbit(mask: BoolArray, reason: Unavailability) -> None:
        code[np.asarray(mask, dtype=bool)] |= np.uint32(int(reason))

    setbit(access.below_min_elevation, Unavailability.BELOW_MIN_ELEVATION)
    setbit(access.earth_blocked, Unavailability.EARTH_OCCULTATION)
    setbit(access.sun_blocked, Unavailability.SUN_OCCULTATION)
    setbit(access.moon_blocked, Unavailability.MOON_OCCULTATION)
    setbit(access.sun_avoidance_violation, Unavailability.SUN_AVOIDANCE)
    frac = illumination.illumination_fraction
    setbit(frac <= 0.0, Unavailability.ECLIPSE)
    setbit((frac > 0.0) & (frac < 1.0), Unavailability.PARTIAL_ECLIPSE)
    setbit(~power_thermal.thermal.feasible, Unavailability.THERMAL)
    if atmosphere_settings.hard_outage:
        setbit(np.ones(n, dtype=bool), Unavailability.WEATHER)
    eta_atm = np.full(n, np.nan)
    eta_cap = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    if beam is not None and atmosphere is not None:
        eta_atm[vis_idx] = atmosphere.transmittance
        weather_bad = np.zeros(n, dtype=bool)
        weather_bad[vis_idx] = ~atmosphere.weather_available
        setbit(weather_bad, Unavailability.WEATHER)
        eta_cap[vis_idx] = beam.capture_fraction
        peak[vis_idx] = beam.peak_irradiance_per_W_m2
        min_pe = config.transmitter.min_pointing_efficiency
        if min_pe is not None:
            pe = beam.pointing_loss
            bad = np.zeros(n, dtype=bool)
            bad[vis_idx] = ~(np.nan_to_num(pe, nan=-1.0) >= min_pe)
            setbit(bad, Unavailability.POINTING)

    enabled = (code & np.uint32(BLOCKING_MASK)) == 0
    link = evaluate_link_budget(
        power_thermal.solar.pv_electric_W,
        models.efficiencies,
        receiver,
        link_enabled=enabled,
        eta_atmosphere=eta_atm,
        eta_capture=eta_cap,
        peak_irradiance_per_W_m2=peak,
    )
    setbit(link.power_density_abort, Unavailability.POWER_DENSITY_ABORT)
    setbit(link.receiver_saturated | link.receiver_heat_limited, Unavailability.RECEIVER_SATURATION)
    available = (code & np.uint32(BLOCKING_MASK)) == 0
    return code, link, available


def _orbit(
    config: RunConfig,
    ephemeris: EphemerisProvider,
    kernels: LoadedKernels,
    log: list[RunLogEntry],
    earth_orientation: EarthOrientation,
) -> tuple[TimeGrid, CartesianTrajectory, L1Diagnostics | None, Float64Array | None]:
    grid = time_grid_from_config(config.time, kernels)
    t0 = utc_to_tdb_s(config.orbit.initial_epoch_utc, kernels)
    l0 = l0_reference_trajectory(
        config.orbit,
        grid.epochs_tdb_s,
        initial_epoch_tdb_s=t0,
        ephemeris=ephemeris,
        earth_orientation=earth_orientation,
    )
    if config.orbit.model != L1_MODEL:
        if config.orbit.is_geocentric:
            log.append(RunLogEntry("INFO", "orbit", _geocentric_orbit_note(config, l0, ephemeris)))
        else:
            log.append(RunLogEntry("INFO", "orbit", "L0 two-body Kepler orbit (closed form)"))
        return grid, l0, None, None
    run = propagate_l1_from_config(config, ephemeris, kernels)
    diag = run.propagation.diagnostics
    for seg in diag.segments:
        log.append(
            RunLogEntry(
                "INFO" if seg.success else "WARNING",
                "orbit",
                f"L1 {seg.direction} segment {seg.t_start_tdb_s:.3f} -> {seg.t_end_tdb_s:.3f} "
                f"TDB s: {seg.n_output} outputs, nfev {seg.nfev}, success {seg.success}, "
                f"message '{seg.message}'",
            )
        )
    log.append(
        RunLogEntry(
            "INFO",
            "orbit",
            f"L1 formulation {diag.formulation}; total nfev {diag.nfev}; heliocentric two-body "
            f"energy relative drift {diag.sun_two_body_energy_rel_drift:.3e}; ephemeris-cache "
            f"step {diag.cache_step_s} s, max midpoint error {diag.cache_max_midpoint_error_m:.3e}"
            " m",
        )
    )
    traj = run.propagation.trajectory
    return run.grid, traj, diag, orbit_drift_m(traj, l0)


def _geocentric_orbit_note(
    config: RunConfig, trajectory: CartesianTrajectory, ephemeris: EphemerisProvider
) -> str:
    """Run-log line of a geocentric L0 orbit (M7b): model, GM, radius range, assumption."""
    earth = ephemeris.positions("EARTH", trajectory.epochs_tdb_s)
    r = np.linalg.norm(np.asarray(trajectory.position_m) - earth, axis=1)
    preset = (
        f"geostationary preset at {config.orbit.geostationary_longitude_deg} deg E (IAU_EARTH)"
        if config.orbit.is_geostationary_preset
        else "configured J2000 equatorial elements"
    )
    return (
        f"L0 two-body Kepler orbit about the Earth's centre (closed form; {preset}; "
        f"GM_earth = {ephemeris.gm_m3_s2('EARTH'):.10e} m^3/s^2); geocentric radius "
        f"{float(np.min(r)):.6f}..{float(np.max(r)):.6f} m; lunisolar, J2 and SRP "
        "perturbations omitted (station-keeping assumed, docs/model_assumptions.md M7b)"
    )


def run_single_satellite(
    config: RunConfig,
    *,
    ephemeris: EphemerisProvider,
    kernels: LoadedKernels,
    rng: np.random.Generator,
    earth_orientation: EarthOrientation | None = None,
    backend: Backend = "numpy",
) -> SystemRunResult:
    """End-to-end run of one satellite and one receiver (module docstring).

    Args:
        config: Validated configuration (exactly one receiver site).
        ephemeris: Ephemeris provider (J2000 / SSB states, GM values).
        kernels: Active kernel-set handle (time conversion, PCK radii, provenance); must be the
            configured ``frames.ephemeris_kernel_set``.
        rng: The run's random Generator (seeded from ``project.seed`` by the caller). M6 has no
            stochastic step; the result records that it was not drawn from.
        earth_orientation: Earth-fixed -> J2000 model (default: SPICE ``IAU_EARTH``).
        backend: Beam-capture backend (``numpy`` reference or ``jax``).

    Returns:
        :class:`SystemRunResult`.

    Raises:
        ValueError: more than one receiver site, or an unknown backend.
        KernelSetMismatchError: ``kernels`` is not the configured kernel set.
        BeamModelValidityError: the beam model is outside its validity range at some epoch.
    """
    if len(config.receiver.sites) != 1:
        raise ValueError(
            f"run_single_satellite supports exactly one receiver site, got "
            f"{len(config.receiver.sites)}; several sites need receiver scheduling (M7) to "
            "avoid double counting"
        )
    if backend not in ("numpy", "numpy_batch", "jax"):
        raise ValueError(f"unknown backend {backend!r}; use 'numpy', 'numpy_batch' or 'jax'")
    check_kernel_set(config, kernels)
    rng_state = rng.bit_generator.state
    timings: list[StageTiming] = []
    log: list[RunLogEntry] = [
        RunLogEntry("WARNING", "config", "named approximation in force: " + a.describe())
        for a in approximations_in_force(config)
    ]
    orientation: EarthOrientation = (
        earth_orientation if earth_orientation is not None else SpiceEarthOrientation(kernels)
    )
    radii = OcculterRadii.from_kernels(kernels)

    grid, traj, l1_diag, drift = _timed(
        "orbit", timings, lambda: _orbit(config, ephemeris, kernels, log, orientation)
    )
    n = len(grid)
    illum = _timed(
        "illumination",
        timings,
        lambda: satellite_illumination(traj, ephemeris=ephemeris, radii=radii),
    )
    models = power_thermal_models_from_config(config)
    pt = _timed("solar_thermal", timings, lambda: power_thermal_from_illumination(illum, models))
    site_cfg = config.receiver.sites[0]
    access = _timed(
        "access",
        timings,
        lambda: compute_access(
            traj,
            GroundSite.from_config(site_cfg),
            ephemeris=ephemeris,
            earth_orientation=orientation,
            settings=access_settings_from_config(config, radii),
        ),
    )
    summary = summarize_access(access, orbit_drift=drift)

    atm_settings = atmosphere_settings_from_config(config)
    beam_settings = beam_settings_from_config(config)
    vis_idx, atm = _timed(
        "atmosphere",
        timings,
        lambda: atmosphere_at_access(access, atm_settings, site_cfg),
    )
    beam: BeamResult | None = None
    table_diag: BeamTableDiagnostics | None = None
    if atm is not None:
        el_v = access.elevation_rad[vis_idx]
        if config.models.beam_evaluation == "chebyshev_table":

            def reference(r: Float64Array, e: Float64Array, t: Float64Array) -> BeamResult:
                return _evaluate_beam_backend(r, e, beam_settings, backend, t)

            beam, table_diag = _timed(
                "beam",
                timings,
                lambda: evaluate_beam_table(
                    access.range_m[vis_idx],
                    el_v,
                    beam_settings,
                    table=beam_table_settings_from_config(config),
                    evaluate=reference,
                    epochs_tdb_s=access.t_emit_tdb_s[vis_idx],
                ),
            )
            log.append(RunLogEntry("INFO", "beam", table_diag.describe()))
        else:
            beam = _timed(
                "beam",
                timings,
                lambda: _evaluate_beam_backend(
                    access.range_m[vis_idx],
                    el_v,
                    beam_settings,
                    backend,
                    access.t_emit_tdb_s[vis_idx],
                ),
            )

    # The site's own capacity overrides (M7) apply to the single-satellite run too: one
    # definition of the receiver for this run and for a one-satellite constellation. Beam slots
    # (max_simultaneous_beams >= 1) cannot bind for one satellite (one beam).
    receiver = receiver_model_for_site(config.receiver, site_cfg)
    code, link, available = _timed(
        "link_budget",
        timings,
        lambda: site_link_budget(
            config,
            access=access,
            illumination=illum,
            power_thermal=pt,
            models=models,
            receiver=receiver,
            beam_epoch_index=vis_idx,
            atmosphere=atm,
            beam=beam,
            atmosphere_settings=atm_settings,
        ),
    )

    ref = _timed(
        "reference_link",
        timings,
        lambda: reference_link(
            config,
            models=models,
            beam_settings=beam_settings,
            atmosphere_settings=atm_settings,
            receiver=receiver,
            sun_distance_m=float(np.mean(illum.sun_distance_m)),
            range_m=float(np.mean(access.range_m)),
            backend=backend,
        ),
    )

    # ---- diagnostics for the run log ----------------------------------------------------
    th = pt.thermal
    if not np.all(th.pv_converged):
        log.append(
            RunLogEntry(
                "WARNING",
                "solar_thermal",
                f"PV temperature solve not converged at {int(np.sum(~th.pv_converged))} epochs",
            )
        )
    if not np.all(th.pv_unique_root):
        log.append(
            RunLogEntry(
                "WARNING",
                "solar_thermal",
                f"PV root uniqueness condition fails at {int(np.sum(~th.pv_unique_root))} epochs",
            )
        )
    led = pt.ledger
    log.append(
        RunLogEntry(
            "INFO" if led.closed else "WARNING",
            "solar_thermal",
            f"energy ledger closed={led.closed}: max relative residual "
            f"{led.max_relative_residual:.3e} (rtol {led.rtol:.1e}); thermal infeasible at "
            f"{int(np.sum(~th.feasible))} of {n} epochs",
        )
    )
    lt = access.light_time
    n_fallback = int(np.count_nonzero(lt.method == "brentq"))
    log.append(
        RunLogEntry(
            "WARNING" if n_fallback or not np.all(lt.converged) else "INFO",
            "access",
            f"light time: max iterations {int(np.max(lt.iterations))}, brentq fallbacks "
            f"{n_fallback}, unconverged {int(np.sum(~lt.converged))}, max residual "
            f"{float(np.max(lt.residual_m)):.3e} m",
        )
    )
    gap = summary.hour_angle_max_gap_rad
    vis = np.asarray(access.visible, dtype=bool)
    has_edges = bool(np.any(vis[1:] != vis[:-1]))
    if gap / (2.0 * math.pi) > ALIASING_WARNING_FRACTION and not has_edges:
        log.append(
            RunLogEntry(
                "INFO",
                "access",
                f"largest unsampled hour-angle gap {math.degrees(gap):.3f} deg, but the access "
                "never changes in the window (no access-window edge, e.g. a geostationary "
                "satellite): no diurnal fraction is quantised by the output step",
            )
        )
    elif gap / (2.0 * math.pi) > ALIASING_WARNING_FRACTION:
        log.append(
            RunLogEntry(
                "WARNING",
                "access",
                f"diurnal aliasing: largest unsampled hour-angle gap {math.degrees(gap):.3f} deg "
                f"(~{gap / (2.0 * math.pi):.4f} of a day per window edge); availability and "
                "access fractions are quantised by the output step "
                f"{config.time.output_step_s} s (docs/model_assumptions.md, M3/M6)",
            )
        )
    if beam is not None:
        for note in beam.validity_notes:
            log.append(RunLogEntry("INFO", "beam", note))
        methods, counts = np.unique(beam.diagnostics.percentile_method, return_counts=True)
        log.append(
            RunLogEntry(
                "INFO",
                "beam",
                "capture percentile methods: "
                + ", ".join(f"{m}={c}" for m, c in zip(methods, counts, strict=True))
                + f"; min offset probability mass "
                f"{float(np.min(beam.diagnostics.offset_probability_mass)):.15f}",
            )
        )
    else:
        log.append(RunLogEntry("WARNING", "beam", "no epoch with geometric access"))
    if atm is not None:
        for note in atm.notes:
            log.append(RunLogEntry("WARNING", "atmosphere", note))
    if not (ref.rated_power_W > 0.0):
        why = [
            name
            for name, bad in (
                ("thermally infeasible", not ref.thermal_feasible),
                ("power-density abort", ref.power_density_abort),
            )
            if bad
        ]
        log.append(
            RunLogEntry(
                "WARNING",
                "reference_link",
                "rated power is 0 at the reference conditions ("
                + (", ".join(why) if why else "zero delivered power")
                + "); the capacity factor is undefined",
            )
        )
    rng_unchanged = rng.bit_generator.state == rng_state
    return SystemRunResult(
        config=config,
        config_hash=config_hash(config),
        grid=grid,
        trajectory=traj,
        orbit_model=config.orbit.model,
        l1_diagnostics=l1_diag,
        orbit_drift_m=drift,
        illumination=illum,
        power_thermal=pt,
        access=access,
        access_summary=summary,
        beam_epoch_index=vis_idx,
        beam=beam,
        atmosphere=atm,
        link=link,
        reason_code=code,
        available=available,
        reference=ref,
        receiver=receiver,
        backend=backend,
        kernel_provenance=tuple(kernels.provenance),
        source_ids=tuple(access.source_ids),
        itu_r_versions=() if atm is None else tuple(atm.recommendation_versions),
        itu_r_polarization_tilt_deg=None if atm is None else atm.polarization_tilt_deg,
        rng_state_unchanged=bool(rng_unchanged),
        timings=tuple(timings),
        log=tuple(log),
        window_end_utc=tdb_s_to_utc(float(grid.epochs_tdb_s[-1]), kernels, decimals=3),
        beam_table=table_diag,
    )


ALIASING_WARNING_FRACTION: Final[float] = 0.005
"""Warn when the largest unsampled hour-angle gap exceeds this fraction of a day (each access
window edge is then resolved only to ~this fraction of the time; M3 measured a 0.020 bias of
the hourly MVP access fraction at gap/2pi = 0.024)."""


__all__ = [
    "ALIASING_WARNING_FRACTION",
    "BLOCKING_MASK",
    "BLOCKING_REASONS",
    "LIMITING_FLAGS",
    "REFERENCE_DEFINITION",
    "Backend",
    "BeamCapture",
    "ReferenceLink",
    "RunLogEntry",
    "StageTiming",
    "SystemRunResult",
    "Unavailability",
    "atmosphere_at_access",
    "primary_reason",
    "reason_names",
    "reference_link",
    "run_single_satellite",
    "site_link_budget",
]
