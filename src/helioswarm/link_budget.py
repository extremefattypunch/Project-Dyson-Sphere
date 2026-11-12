"""End-to-end link budget of one satellite-receiver link (PDF §10.1, §10.4).

One auditable sequence of power stages, each efficiency with exactly one owner::

    P_PV          (M4 solar/thermal: generated PV electrical power)
    P_PV,tx     = P_PV * 1[transmitting]                (operating decision, this module)
    P_dc_to_tx  = P_PV,tx    * eta_power_conditioning   (power_chain.onboard_power_chain)
    P_beam      = P_dc_to_tx * eta_electrical_to_beam   (power_chain.onboard_power_chain)
    P_aperture  = P_beam     * eta_optics_or_rf         (power_chain.onboard_power_chain)
    P_atm       = P_aperture * eta_atmosphere           (atmosphere.atmospheric_transmission)
    P_captured  = P_atm      * eta_capture              (beam.BeamResult.capture_fraction)
    P_accepted  = min(P_captured, P_max_input, Q_max/(1 - eta_rx))   (receiver limits, here)
    P_receiver  = P_accepted * eta_receiver_conversion  (receiver.conversion_efficiency)
    P_grid      = P_receiver * eta_grid_conditioning    (receiver.grid_efficiency)

Single owners, no double counting (PDF §10.1):

* The first three efficiencies are applied only by
  :func:`helioswarm.power_chain.onboard_power_chain` (the M4 radiator uses the same function).
* ``eta_capture`` is ``BeamResult.capture_fraction``: the beam pattern is normalised to the
  power leaving the aperture (M5), so there is **no** extra "aperture efficiency". The aperture
  taper is inside the pattern, Gaussian source spill-over belongs to ``eta_optics_or_rf``
  (M0/M5 definition), and the wavefront (Ruze) and phased-array grating-lobe factors are
  applied once, inside ``capture_fraction`` (``beam.evaluate_beam``). Pointing bias/jitter is
  the expectation over the Rice offset law, also inside ``capture_fraction``.
* ``eta_atmosphere`` is applied once, here, and depends on elevation only (never on range).

Receiver constraints (PDF §10.4) are engineering constraints of the chain, not post-processing:

* ``receiver.max_input_power_W`` bounds ``P_captured`` (M0 field definition: the beam power
  entering the receiver aperture before conversion). Excess power is clipped, reported as
  ``receiver_clipped_W`` and flagged ``receiver_saturated``.
* ``receiver.max_heat_rejection_W`` (optional) bounds the receiver waste heat
  ``Q_rx = P_accepted (1 - eta_rx)``; it is enforced as a second bound on ``P_accepted``
  (``P_accepted <= Q_max / (1 - eta_rx)``) and flagged ``receiver_heat_limited``.
* ``receiver.max_power_density_W_m2`` (optional) is a **beam-abort** condition (the only policy
  the schema defines; there is no throttling option): if the peak incident irradiance on the
  beam-normal plane, ``P_atm * peak_irradiance_per_W`` (the aberration-free pattern peak, an
  upper bound), would exceed the limit, the transmitter is not operated at that epoch
  (``P_PV,tx = 0`` and every downstream stage is 0) and ``power_density_abort`` is set.

Epochs where the link is not enabled (geometry, occultation, weather, thermal, pointing or
eclipse; decided by :mod:`helioswarm.system`) carry ``P_PV,tx = 0``; their atmosphere and
capture factors are recorded as 0 (the transmission of a blocked/unevaluated path), so every
recorded efficiency lies in [0, 1].

Invariants (checked on construction of every :class:`LinkBudgetResult`; PDF §13.1 "power never
increases through passive stages; all efficiencies lie in [0, 1]"): every efficiency is in
[0, 1] and every stage power is <= the previous stage. Floating-point multiplication by a
factor in [0, 1] is monotone (``fl(x * eta) <= x`` exactly), so the checks are exact (no
tolerance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.config import ReceiverConfig, ReceiverTechnology
from helioswarm.power_chain import OnboardEfficiencies, onboard_power_chain

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

STAGE_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("P_PV", "pv_electric_W"),
    ("P_PV,tx", "pv_to_transmitter_W"),
    ("P_dc_to_tx", "dc_to_tx_W"),
    ("P_beam", "beam_W"),
    ("P_aperture", "aperture_W"),
    ("P_atm", "atm_W"),
    ("P_captured", "captured_W"),
    ("P_accepted", "accepted_W"),
    ("P_receiver", "receiver_W"),
    ("P_grid", "grid_W"),
)
"""The PDF §10.1 stage chain in order: (symbol, :class:`LinkBudgetResult` attribute)."""


class LinkBudgetInvariantError(ValueError):
    """A link-budget invariant (efficiency range or stage monotonicity) is violated."""


@dataclass(frozen=True)
class ReceiverModel:
    """Receiver and grid interface of one technology case (PDF §10.4).

    Optical (``laser_photovoltaic``) and microwave (``rectenna``) receivers are different
    technology cases: each configuration states its own conversion efficiency and limits
    (the schema ties the technology to ``transmitter.mode``).

    Attributes:
        technology: ``laser_photovoltaic`` or ``rectenna``.
        conversion_efficiency: ``eta_receiver_conversion = P_receiver / P_accepted`` [1].
        grid_efficiency: ``eta_grid_conditioning = P_grid / P_receiver`` [1].
        max_input_power_W: Bound on ``P_captured`` accepted by the receiver [W].
        max_heat_rejection_W: Bound on ``Q_rx = P_accepted (1 - eta_rx)`` [W] or ``None``.
        max_power_density_W_m2: Beam-abort peak irradiance limit [W m^-2] or ``None``.
    """

    technology: ReceiverTechnology
    conversion_efficiency: float
    grid_efficiency: float
    max_input_power_W: float
    max_heat_rejection_W: float | None = None
    max_power_density_W_m2: float | None = None

    def __post_init__(self) -> None:
        for name in ("conversion_efficiency", "grid_efficiency"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and 0.0 < v <= 1.0):
                raise ValueError(f"ReceiverModel.{name}={v} must lie in (0, 1]")
        for name in ("max_input_power_W", "max_heat_rejection_W", "max_power_density_W_m2"):
            v = getattr(self, name)
            if v is not None and not (math.isfinite(float(v)) and float(v) > 0.0):
                raise ValueError(f"ReceiverModel.{name}={v} must be finite and > 0")

    @property
    def heat_limited_input_W(self) -> float:
        """``Q_max / (1 - eta_rx)``: the accepted power at which the heat limit binds [W]
        (``inf`` if not enforced or if ``eta_rx = 1``)."""
        if self.max_heat_rejection_W is None or self.conversion_efficiency >= 1.0:
            return math.inf
        return self.max_heat_rejection_W / (1.0 - self.conversion_efficiency)


def receiver_model_from_config(receiver: ReceiverConfig) -> ReceiverModel:
    """:class:`ReceiverModel` from the ``receiver`` configuration section."""
    return ReceiverModel(
        technology=receiver.technology,
        conversion_efficiency=receiver.conversion_efficiency,
        grid_efficiency=receiver.grid_efficiency,
        max_input_power_W=receiver.max_input_power_W,
        max_heat_rejection_W=receiver.max_heat_rejection_W,
        max_power_density_W_m2=receiver.max_power_density_W_m2,
    )


def _frozen(values: ArrayLike, dtype: type = np.float64) -> NDArray[np.generic]:
    arr: NDArray[np.generic] = np.array(values, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class LinkBudgetResult:
    """Every stage power and every efficiency of the PDF §10.1 chain per epoch (shape (N,)).

    Attributes:
        pv_electric_W: Generated PV electrical power ``P_PV`` (M4) [W].
        pv_to_transmitter_W: PV power routed to the transmitter, ``P_PV`` when transmitting
            else 0 [W].
        dc_to_tx_W: ``P_dc_to_tx`` [W].
        beam_W: ``P_beam`` [W].
        aperture_W: ``P_aperture`` (power leaving the aperture) [W].
        atm_W: ``P_atm`` (beam power arriving at the receiver plane) [W].
        captured_W: ``P_captured`` (beam power intercepted by the receiver aperture) [W].
        accepted_W: ``P_accepted`` after the receiver input/heat limits [W].
        receiver_W: ``P_receiver`` (receiver electrical output) [W].
        grid_W: ``P_grid`` (accepted by the grid interface) [W].
        eta_power_conditioning, eta_electrical_to_beam, eta_optics_or_rf: On-board stage
            efficiencies [1] (constants of the run).
        eta_atmosphere: ``P_atm / P_aperture`` per epoch (0 where not evaluated) [1].
        eta_capture: ``P_captured / P_atm`` per epoch (0 where not evaluated) [1].
        eta_receiver_acceptance: ``P_accepted / P_captured`` (1 where nothing is captured) [1].
        eta_receiver_conversion: ``P_receiver / P_accepted`` [1].
        eta_grid_conditioning: ``P_grid / P_receiver`` [1].
        link_enabled: All non-receiver availability conditions met (input of the budget).
        transmitting: Transmitter operated (``link_enabled`` and no beam abort).
        power_density_abort: Beam aborted because the peak irradiance would exceed the limit.
        receiver_saturated: ``P_captured > max_input_power_W`` (clipped).
        receiver_heat_limited: ``P_captured`` exceeds the heat-rejection bound (clipped).
        receiver_clipped_W: ``P_captured - P_accepted`` [W].
        receiver_heat_W: Receiver waste heat ``Q_rx = P_accepted - P_receiver`` [W].
        requested_peak_power_density_W_m2: Peak irradiance the beam would produce if operated
            (0 where the link is not enabled) [W m^-2].
        peak_power_density_W_m2: Actual peak incident irradiance (0 when not transmitting)
            [W m^-2].
        receiver: Receiver model used.
    """

    pv_electric_W: Float64Array
    pv_to_transmitter_W: Float64Array
    dc_to_tx_W: Float64Array
    beam_W: Float64Array
    aperture_W: Float64Array
    atm_W: Float64Array
    captured_W: Float64Array
    accepted_W: Float64Array
    receiver_W: Float64Array
    grid_W: Float64Array
    eta_power_conditioning: float
    eta_electrical_to_beam: float
    eta_optics_or_rf: float
    eta_atmosphere: Float64Array
    eta_capture: Float64Array
    eta_receiver_acceptance: Float64Array
    eta_receiver_conversion: float
    eta_grid_conditioning: float
    link_enabled: BoolArray
    transmitting: BoolArray
    power_density_abort: BoolArray
    receiver_saturated: BoolArray
    receiver_heat_limited: BoolArray
    receiver_clipped_W: Float64Array
    receiver_heat_W: Float64Array
    requested_peak_power_density_W_m2: Float64Array
    peak_power_density_W_m2: Float64Array
    receiver: ReceiverModel = field(repr=False)

    def __post_init__(self) -> None:
        shape = np.shape(self.pv_electric_W)
        if len(shape) != 1:
            raise ValueError("LinkBudgetResult arrays must be 1-D")
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                if value.shape != shape:
                    raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
                object.__setattr__(self, name, _frozen(value, value.dtype.type))
        problems = link_budget_violations(self)
        if problems:
            raise LinkBudgetInvariantError("; ".join(problems))

    def __len__(self) -> int:
        return int(self.pv_electric_W.shape[0])

    # PDF p.4 LinkBudgetResult field names (aliases of the stage attributes above).
    @property
    def power_at_transmitter_W(self) -> Float64Array:
        """PDF p.4 ``power_at_transmitter_W``: DC power delivered to the transmitter
        ``P_dc_to_tx`` [W]."""
        return self.dc_to_tx_W

    @property
    def power_at_receiver_W(self) -> Float64Array:
        """PDF p.4 ``power_at_receiver_W``: beam power intercepted by the receiver
        ``P_captured`` [W]."""
        return self.captured_W

    @property
    def atmospheric_factor(self) -> Float64Array:
        """PDF p.4 ``atmospheric_factor``: ``eta_atmosphere`` [1]."""
        return self.eta_atmosphere

    @property
    def rx_electric_W(self) -> Float64Array:
        """PDF p.4 ``rx_electric_W``: receiver electrical output ``P_receiver`` [W]."""
        return self.receiver_W

    def stage(self, attribute: str) -> Float64Array:
        """Stage power array by attribute name (one of :data:`STAGE_COLUMNS`)."""
        if attribute not in {a for _, a in STAGE_COLUMNS}:
            raise KeyError(f"{attribute!r} is not a link-budget stage")
        out: Float64Array = getattr(self, attribute)
        return out


def link_budget_violations(result: LinkBudgetResult) -> tuple[str, ...]:
    """Invariant violations of a link budget (empty tuple if none).

    Checks: every efficiency (scalar or per epoch) is finite and in [0, 1]; every stage power is
    finite, >= 0 and <= the preceding stage (:data:`STAGE_COLUMNS` order); ``P_grid = 0`` where
    not transmitting. Exact comparisons (module docstring).
    """
    problems: list[str] = []
    for name in (
        "eta_power_conditioning",
        "eta_electrical_to_beam",
        "eta_optics_or_rf",
        "eta_atmosphere",
        "eta_capture",
        "eta_receiver_acceptance",
        "eta_receiver_conversion",
        "eta_grid_conditioning",
    ):
        v = np.asarray(getattr(result, name), dtype=np.float64)
        if np.any(~np.isfinite(v)) or np.any(v < 0.0) or np.any(v > 1.0):
            problems.append(f"{name} outside [0, 1] or non-finite")
    prev_name: str | None = None
    prev: Float64Array | None = None
    for _, attr in STAGE_COLUMNS:
        p = np.asarray(getattr(result, attr), dtype=np.float64)
        if np.any(~np.isfinite(p)) or np.any(p < 0.0):
            problems.append(f"{attr} negative or non-finite")
        if prev is not None and np.any(p > prev):
            n_bad = int(np.count_nonzero(p > prev))
            problems.append(f"{attr} exceeds {prev_name} at {n_bad} epoch(s) (passive stage)")
        prev_name, prev = attr, p
    off = ~np.asarray(result.transmitting, dtype=bool)
    if np.any(np.asarray(result.grid_W)[off] != 0.0):
        problems.append("grid_W non-zero at an epoch without transmission")
    return tuple(problems)


def _factor(values: ArrayLike, enabled: BoolArray, name: str) -> Float64Array:
    v = np.broadcast_to(np.asarray(values, dtype=np.float64), enabled.shape)
    bad = enabled & (~np.isfinite(v) | (v < 0.0) | (v > 1.0))
    if np.any(bad):
        raise ValueError(
            f"{name} must be finite and in [0, 1] wherever the link is enabled "
            f"({int(np.count_nonzero(bad))} bad epoch(s))"
        )
    return np.where(enabled, v, 0.0)


def evaluate_link_budget(
    pv_electric_W: ArrayLike,
    efficiencies: OnboardEfficiencies,
    receiver: ReceiverModel,
    *,
    link_enabled: ArrayLike,
    eta_atmosphere: ArrayLike,
    eta_capture: ArrayLike,
    peak_irradiance_per_W_m2: ArrayLike,
) -> LinkBudgetResult:
    """PDF §10.1 stage chain with the PDF §10.4 receiver constraints, per epoch.

    Args:
        pv_electric_W: Generated PV electrical power ``P_PV`` [W], shape (N,).
        efficiencies: On-board stage efficiencies (single owner:
            :func:`~helioswarm.power_chain.onboard_power_chain`).
        receiver: Receiver model (conversion, grid and limits).
        link_enabled: All availability conditions other than the receiver's own met (bool, N).
        eta_atmosphere: ``eta_atmosphere`` per epoch [1]; may be NaN where not enabled.
        eta_capture: ``BeamResult.capture_fraction`` per epoch [1]; may be NaN where not enabled.
        peak_irradiance_per_W_m2: ``BeamResult.peak_irradiance_per_W_m2`` per epoch [m^-2];
            may be NaN where not enabled.

    Returns:
        :class:`LinkBudgetResult` (invariants checked on construction).

    Raises:
        ValueError: inconsistent shapes, a factor outside [0, 1] at an enabled epoch, or
            negative/non-finite PV power.
    """
    p_pv = np.atleast_1d(np.asarray(pv_electric_W, dtype=np.float64))
    if p_pv.ndim != 1:
        raise ValueError("pv_electric_W must be 1-D")
    enabled = np.broadcast_to(np.asarray(link_enabled, dtype=bool), p_pv.shape).copy()
    eta_atm = _factor(eta_atmosphere, enabled, "eta_atmosphere")
    eta_cap = _factor(eta_capture, enabled, "eta_capture")
    peak = np.broadcast_to(np.asarray(peak_irradiance_per_W_m2, dtype=np.float64), p_pv.shape)
    bad_peak = enabled & (~np.isfinite(peak) | (peak < 0.0))
    if np.any(bad_peak):
        raise ValueError("peak_irradiance_per_W_m2 must be finite and >= 0 where enabled")
    peak = np.where(enabled, peak, 0.0)

    # Prospective chain (if the beam were operated at every enabled epoch) for the abort check.
    requested = onboard_power_chain(np.where(enabled, p_pv, 0.0), efficiencies)
    requested_density = requested.aperture_W * eta_atm * peak
    if receiver.max_power_density_W_m2 is None:
        abort = np.zeros(p_pv.shape, dtype=bool)
    else:
        abort = enabled & (requested_density > receiver.max_power_density_W_m2)
    transmitting = enabled & ~abort

    chain = onboard_power_chain(np.where(transmitting, p_pv, 0.0), efficiencies)
    p_atm = chain.aperture_W * eta_atm
    p_cap = p_atm * eta_cap
    heat_bound = receiver.heat_limited_input_W
    accepted = np.minimum(np.minimum(p_cap, receiver.max_input_power_W), heat_bound)
    p_rx = accepted * receiver.conversion_efficiency
    p_grid = p_rx * receiver.grid_efficiency
    with np.errstate(invalid="ignore", divide="ignore"):
        acceptance = np.where(p_cap > 0.0, accepted / np.where(p_cap > 0.0, p_cap, 1.0), 1.0)
    return LinkBudgetResult(
        pv_electric_W=p_pv,
        pv_to_transmitter_W=chain.pv_electric_W,
        dc_to_tx_W=chain.dc_to_tx_W,
        beam_W=chain.beam_W,
        aperture_W=chain.aperture_W,
        atm_W=p_atm,
        captured_W=p_cap,
        accepted_W=accepted,
        receiver_W=p_rx,
        grid_W=p_grid,
        eta_power_conditioning=efficiencies.power_conditioning,
        eta_electrical_to_beam=efficiencies.electrical_to_beam,
        eta_optics_or_rf=efficiencies.optics_or_rf,
        eta_atmosphere=eta_atm,
        eta_capture=eta_cap,
        eta_receiver_acceptance=np.minimum(acceptance, 1.0),
        eta_receiver_conversion=receiver.conversion_efficiency,
        eta_grid_conditioning=receiver.grid_efficiency,
        link_enabled=enabled,
        transmitting=transmitting,
        power_density_abort=abort,
        receiver_saturated=p_cap > receiver.max_input_power_W,
        receiver_heat_limited=p_cap > heat_bound,
        receiver_clipped_W=p_cap - accepted,
        receiver_heat_W=accepted - p_rx,
        requested_peak_power_density_W_m2=requested_density,
        peak_power_density_W_m2=p_atm * peak,
        receiver=receiver,
    )


__all__ = [
    "STAGE_COLUMNS",
    "LinkBudgetInvariantError",
    "LinkBudgetResult",
    "ReceiverModel",
    "evaluate_link_budget",
    "link_budget_violations",
    "receiver_model_from_config",
]
