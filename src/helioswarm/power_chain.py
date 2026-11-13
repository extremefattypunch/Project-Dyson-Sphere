"""On-board part of the PDF §10.1 power chain: PV output -> conditioned DC -> beam -> aperture.

This module is the **single owner** of the three on-board stage efficiencies (PDF §10.1: "each
efficiency factor must have one owner and one definition"). The thermal model (M4, radiator
waste heat) and the end-to-end link budget (M6) both obtain the on-board stage powers from
:func:`onboard_power_chain`, so the same efficiency is never applied (or counted as heat) twice::

    P_dc_to_tx = P_PV       * eta_power_conditioning   (transmitter.power_conditioning_efficiency)
    P_beam     = P_dc_to_tx * eta_electrical_to_beam   (transmitter.electrical_to_beam_efficiency)
    P_aperture = P_beam     * eta_optics_or_rf         (transmitter.optics_or_rf_efficiency)

Each stage loss is the stage input minus the stage output (equal to ``P_in (1 - eta)``):

* ``power_conditioning_loss_W = P_PV - P_dc_to_tx``
* ``electrical_to_beam_loss_W = P_dc_to_tx - P_beam``
* ``optics_or_rf_loss_W = P_beam - P_aperture``

``P_aperture`` leaves the spacecraft as the beam; the three losses are the on-board waste heat
of the chain (M4 assigns all of them to the power-electronics/transmitter radiator node, a
conservative upper bound for the optics loss, part of which may be spill-over radiated to space
rather than absorbed; see docs/model_assumptions.md, M4).

Stages after the aperture (atmosphere, capture, receiver, grid) are not on board and belong to
the M6 link budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.config import TransmitterConfig

Float64Array = NDArray[np.float64]


def _frozen(values: ArrayLike) -> Float64Array:
    arr = np.array(values, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class OnboardEfficiencies:
    """The three on-board stage efficiencies of PDF §10.1 (each in (0, 1]).

    Attributes:
        power_conditioning: ``eta_power_conditioning = P_dc_to_tx / P_PV`` [1].
        electrical_to_beam: ``eta_electrical_to_beam = P_beam / P_dc_to_tx`` [1].
        optics_or_rf: ``eta_transmitter_optics_or_rf = P_aperture / P_beam`` [1]; a separate
            physical loss, not an aperture (diffraction) efficiency.
    """

    power_conditioning: float
    electrical_to_beam: float
    optics_or_rf: float

    def __post_init__(self) -> None:
        for name in ("power_conditioning", "electrical_to_beam", "optics_or_rf"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and 0.0 < value <= 1.0):
                raise ValueError(f"OnboardEfficiencies.{name}={value} must lie in (0, 1]")
            object.__setattr__(self, name, value)

    @property
    def dc_to_aperture(self) -> float:
        """Product of the three stage efficiencies, ``P_aperture / P_PV`` [1]."""
        return self.power_conditioning * self.electrical_to_beam * self.optics_or_rf


def onboard_efficiencies_from_config(transmitter: TransmitterConfig) -> OnboardEfficiencies:
    """Stage efficiencies from the ``transmitter`` configuration section (PDF §10.1)."""
    return OnboardEfficiencies(
        power_conditioning=transmitter.power_conditioning_efficiency,
        electrical_to_beam=transmitter.electrical_to_beam_efficiency,
        optics_or_rf=transmitter.optics_or_rf_efficiency,
    )


@dataclass(frozen=True, eq=False)
class OnboardPowerChain:
    """Stage powers and losses of the on-board chain, vectorised (same shape as ``pv_electric_W``).

    Attributes:
        pv_electric_W: PV electrical output ``P_PV`` (chain input) [W].
        dc_to_tx_W: Conditioned DC power delivered to the transmitter ``P_dc_to_tx`` [W].
        beam_W: Beam (optical or RF) power generated ``P_beam`` [W].
        aperture_W: Power leaving the transmitting aperture ``P_aperture`` [W].
        power_conditioning_loss_W: ``P_PV - P_dc_to_tx`` [W].
        electrical_to_beam_loss_W: ``P_dc_to_tx - P_beam`` [W].
        optics_or_rf_loss_W: ``P_beam - P_aperture`` [W].
        efficiencies: The stage efficiencies used.
    """

    pv_electric_W: Float64Array
    dc_to_tx_W: Float64Array
    beam_W: Float64Array
    aperture_W: Float64Array
    power_conditioning_loss_W: Float64Array
    electrical_to_beam_loss_W: Float64Array
    optics_or_rf_loss_W: Float64Array
    efficiencies: OnboardEfficiencies = field(repr=False)

    def __post_init__(self) -> None:
        shape = np.shape(self.pv_electric_W)
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                if value.shape != shape:
                    raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
                object.__setattr__(self, name, _frozen(value))

    @property
    def total_loss_W(self) -> Float64Array:
        """Sum of the three on-board stage losses [W] (all assigned to the radiator node)."""
        total = self.power_conditioning_loss_W + self.electrical_to_beam_loss_W
        return np.asarray(total + self.optics_or_rf_loss_W, dtype=np.float64)


def onboard_power_chain(
    pv_electric_W: ArrayLike, efficiencies: OnboardEfficiencies
) -> OnboardPowerChain:
    """On-board stage powers of PDF §10.1 from the PV electrical output.

    ``P_dc_to_tx = P_PV eta_pc``, ``P_beam = P_dc_to_tx eta_e2b``, ``P_aperture = P_beam eta_o``;
    each loss is stage input minus stage output (module docstring). Passive stages never increase
    power because every efficiency lies in (0, 1].

    Args:
        pv_electric_W: PV electrical output ``P_PV`` [W], any shape, finite and >= 0.
        efficiencies: Stage efficiencies (single owner, :class:`OnboardEfficiencies`).

    Returns:
        :class:`OnboardPowerChain` with arrays of the shape of ``pv_electric_W``.

    Raises:
        ValueError: negative or non-finite input power.
    """
    p_pv = np.array(pv_electric_W, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(p_pv)) or np.any(p_pv < 0.0):
        raise ValueError("pv_electric_W must be finite and non-negative")
    p_dc = p_pv * efficiencies.power_conditioning
    p_beam = p_dc * efficiencies.electrical_to_beam
    p_ap = p_beam * efficiencies.optics_or_rf
    return OnboardPowerChain(
        pv_electric_W=p_pv,
        dc_to_tx_W=p_dc,
        beam_W=p_beam,
        aperture_W=p_ap,
        power_conditioning_loss_W=p_pv - p_dc,
        electrical_to_beam_loss_W=p_dc - p_beam,
        optics_or_rf_loss_W=p_beam - p_ap,
        efficiencies=efficiencies,
    )


__all__ = [
    "OnboardEfficiencies",
    "OnboardPowerChain",
    "onboard_efficiencies_from_config",
    "onboard_power_chain",
]
