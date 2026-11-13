"""Solar collection and temperature-dependent PV conversion (PDF §8.1-8.2).

Equations (all SI; PDF §8.1-8.2):

* Irradiance at the satellite, inverse-square law::

      S(r) = S_1AU * (AU / r_sun)^2

  ``r_sun`` is the satellite - Sun-centre distance (``IlluminationResult.sun_distance_m``: the
  light-time-retarded Sun from the ephemeris provider, J2000 / SSB states). ``S_1AU`` is
  ``constants.solar_constant_1au_W_m2``. For a spherically symmetric Sun the flux through a
  plane normal to the Sun direction is exactly ``L / (4 pi r^2)`` at any ``r > R_sun`` (energy
  conservation), so no finite-disk correction applies at normal incidence; solar-cycle
  variability of the TSI (~0.1 %) is not modelled.
* Incidence: ``sun_tracking`` keeps the collector normal on the Sun line, ``cos(inc) = 1`` (an
  optimistic bound, PDF §8.1). ``fixed_normal`` uses the inertially fixed J2000 unit normal
  ``n``: ``cos(inc) = max(n . s_hat, 0)`` with ``s_hat`` the satellite -> Sun unit vector. Light
  arriving on the back face (``n . s_hat < 0``) gives **zero** absorbed and electrical power: the
  back face carries no cells, and its absorption of sunlight is *not* modelled (the panel then
  produces nothing; its back-lit heating is a documented limitation). The Sun is treated as a
  point source for the incidence angle (its 4.65e-3 rad radius at 1 au matters only within that
  angle of grazing incidence). Aberration of the sunlight direction and the Doppler factor of
  the flux in the satellite frame are O(v/c) ~ 1e-4 effects and are neglected.
* Illumination: the visible fraction ``f`` of the solar disk (M3 conical shadow) multiplies the
  flux. ``incident_W = S A cos(inc) f`` is the sunlight intercepted by the collector plane.
* Absorbed / reflected split: ``absorbed_W = absorptivity * incident_W`` (PDF §8.1
  ``P_abs = S A alpha cos(inc)``, times ``f``) and ``reflected_W = (1 - absorptivity) *
  incident_W``. The array is opaque, so "reflected" covers everything not absorbed. Reflected
  sunlight leaves the spacecraft and is **not waste heat**. Because ``P_abs`` already excludes it,
  the PDF §8.1 term "explicitly_modelled_reflected_or_transmitted_power" is not subtracted again
  from ``Q_PV``: ``Q_PV = P_abs - P_PV``.
* PV conversion: ``eta_PV`` is defined relative to **absorbed** power (configuration schema,
  M0), ``P_PV = P_abs eta_PV(T)``, with the *local linear approximation* (PDF §8.2)::

      eta_PV(T) = clip(eta_ref * (1 + gamma_T * (T - T_ref)), eta_min, eta_max)

  It is a local linearisation around ``T_ref`` and is not a measured efficiency-temperature
  curve; the clip keeps it in ``[eta_min, eta_max] subset [0, 1]``. Its unclipped validity
  interval is reported by :meth:`PvEfficiencyLaw.linear_range_K`, and every result flags epochs
  at which the clip is active. A technology-specific measured curve should replace it in later
  studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.config import RunConfig
from helioswarm.constants import AU_M
from helioswarm.frames import FrameName, TimeScale

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

IncidenceMode = Literal["sun_tracking", "fixed_normal"]
"""Collector attitude model (``collector.incidence_mode``)."""

UNIT_NORMAL_TOLERANCE = 1.0e-6
"""Accepted ``| |n| - 1 |`` of a fixed collector normal (same tolerance as the schema)."""


def _frozen(values: ArrayLike, dtype: type = np.float64) -> NDArray[np.generic]:
    arr: NDArray[np.generic] = np.array(values, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def _check_finite(name: str, values: Float64Array) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class PvEfficiencyLaw:
    """Clipped linear PV efficiency law (PDF §8.2), efficiency relative to absorbed power.

    ``eta_PV(T) = clip(eta_ref (1 + gamma_T (T - T_ref)), eta_min, eta_max)`` with
    ``eta_ref = reference``, ``gamma_T = temperature_coefficient_per_K``,
    ``T_ref = reference_temperature_K``, ``eta_min = minimum``, ``eta_max = maximum``: a labelled
    *local linear approximation*.

    Attributes:
        reference: ``eta_ref`` at ``reference_temperature_K`` [1].
        reference_temperature_K: ``T_ref`` [K].
        temperature_coefficient_per_K: Relative coefficient ``gamma_T`` [K^-1].
        minimum: Lower clip ``eta_min`` [1], ``0 <= eta_min``.
        maximum: Upper clip ``eta_max`` [1], ``eta_max <= 1``.
    """

    reference: float
    reference_temperature_K: float
    temperature_coefficient_per_K: float
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.reference,
            self.reference_temperature_K,
            self.temperature_coefficient_per_K,
            self.minimum,
            self.maximum,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("PvEfficiencyLaw parameters must be finite")
        if not 0.0 <= self.minimum < self.maximum <= 1.0:
            raise ValueError(
                f"need 0 <= minimum < maximum <= 1 (got {self.minimum}, {self.maximum})"
            )
        if not self.minimum <= self.reference <= self.maximum:
            raise ValueError(f"reference={self.reference} outside [{self.minimum}, {self.maximum}]")
        if self.reference_temperature_K <= 0.0:
            raise ValueError("reference_temperature_K must be > 0")

    def unclipped(self, temperature_K: ArrayLike) -> Float64Array:
        """Linear law without the clip, ``eta_ref (1 + gamma_T (T - T_ref))`` [1]."""
        t = np.asarray(temperature_K, dtype=np.float64)
        return np.asarray(
            self.reference
            * (1.0 + self.temperature_coefficient_per_K * (t - self.reference_temperature_K)),
            dtype=np.float64,
        )

    def efficiency(self, temperature_K: ArrayLike) -> Float64Array:
        """``eta_PV(T)`` [1], clipped to ``[minimum, maximum]`` (see :func:`pv_efficiency`)."""
        return pv_efficiency(temperature_K, self)

    def is_clipped(self, temperature_K: ArrayLike) -> BoolArray:
        """True where the linear law lies outside ``[minimum, maximum]`` (clip active)."""
        raw = self.unclipped(temperature_K)
        return np.asarray((raw < self.minimum) | (raw > self.maximum), dtype=np.bool_)

    def linear_range_K(self) -> tuple[float, float]:
        """Temperature interval [K] on which the linear law is not clipped.

        For ``gamma_T != 0`` the bounds are where ``eta_ref (1 + gamma_T (T - T_ref))`` equals
        ``eta_min`` and ``eta_max``; for ``gamma_T = 0`` (or ``eta_ref = 0``) the law is
        constant and the interval is unbounded (``(-inf, inf)``).
        """
        g = self.temperature_coefficient_per_K
        if g == 0.0 or self.reference == 0.0:
            return (-math.inf, math.inf)
        t_at = [
            self.reference_temperature_K + (eta / self.reference - 1.0) / g
            for eta in (self.minimum, self.maximum)
        ]
        return (min(t_at), max(t_at))


def pv_efficiency(temperature_K: ArrayLike, law: PvEfficiencyLaw) -> Float64Array:
    """PV efficiency relative to absorbed power, PDF §8.2 clipped local linear law.

    ``eta_PV(T) = clip(eta_ref (1 + gamma_T (T - T_ref)), eta_min, eta_max)``; never below
    ``eta_min >= 0`` nor above ``eta_max <= 1``.

    Args:
        temperature_K: Cell temperature ``T`` [K], any shape.
        law: Efficiency law parameters.

    Returns:
        ``eta_PV`` [1], shape of ``temperature_K``.
    """
    return np.asarray(np.clip(law.unclipped(temperature_K), law.minimum, law.maximum))


@dataclass(frozen=True)
class CollectorModel:
    """PV collector parameters used by the solar and thermal models (PDF §8.1-8.2).

    Attributes:
        area_m2: Collector area ``A`` [m^2].
        absorptivity: Solar absorptance ``alpha`` of the front face [1].
        efficiency_law: PV efficiency law (relative to absorbed power).
        incidence_mode: ``sun_tracking`` or ``fixed_normal``.
        fixed_normal_unit_j2000: Unit normal of the front face in J2000 (``fixed_normal``).
        max_operating_temperature_K: PV feasibility limit [K].
        solar_constant_1au_W_m2: ``S_1AU`` [W m^-2].
    """

    area_m2: float
    absorptivity: float
    efficiency_law: PvEfficiencyLaw
    incidence_mode: IncidenceMode
    fixed_normal_unit_j2000: tuple[float, float, float] | None
    max_operating_temperature_K: float
    solar_constant_1au_W_m2: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.area_m2) and self.area_m2 > 0.0):
            raise ValueError("area_m2 must be > 0")
        if not 0.0 < self.absorptivity <= 1.0:
            raise ValueError("absorptivity must lie in (0, 1]")
        if not (math.isfinite(self.solar_constant_1au_W_m2) and self.solar_constant_1au_W_m2 > 0):
            raise ValueError("solar_constant_1au_W_m2 must be > 0")
        if not self.max_operating_temperature_K > 0.0:
            raise ValueError("max_operating_temperature_K must be > 0")
        if self.incidence_mode == "fixed_normal":
            if self.fixed_normal_unit_j2000 is None:
                raise ValueError("fixed_normal requires fixed_normal_unit_j2000")
            norm = math.sqrt(sum(c * c for c in self.fixed_normal_unit_j2000))
            if abs(norm - 1.0) > UNIT_NORMAL_TOLERANCE:
                raise ValueError(f"fixed_normal_unit_j2000 is not a unit vector (|n| = {norm})")
        elif self.incidence_mode == "sun_tracking":
            if self.fixed_normal_unit_j2000 is not None:
                raise ValueError("sun_tracking ignores fixed_normal_unit_j2000; leave it None")
        else:
            raise ValueError(f"unknown incidence_mode {self.incidence_mode!r}")


def collector_model_from_config(config: RunConfig) -> CollectorModel:
    """:class:`CollectorModel` from the ``collector`` and ``constants`` sections."""
    c = config.collector
    return CollectorModel(
        area_m2=c.area_m2,
        absorptivity=c.absorptivity,
        efficiency_law=PvEfficiencyLaw(
            reference=c.pv_efficiency_reference,
            reference_temperature_K=c.reference_temperature_K,
            temperature_coefficient_per_K=c.temperature_coefficient_per_K,
            minimum=c.pv_efficiency_min,
            maximum=c.pv_efficiency_max,
        ),
        incidence_mode=c.incidence_mode,
        fixed_normal_unit_j2000=c.fixed_normal_unit_j2000,
        max_operating_temperature_K=c.max_operating_temperature_K,
        solar_constant_1au_W_m2=config.constants.solar_constant_1au_W_m2,
    )


def solar_flux_W_m2(sun_distance_m: ArrayLike, solar_constant_1au_W_m2: float) -> Float64Array:
    """Solar irradiance at distance ``r_sun`` from the Sun's centre (PDF §8.1).

    ``S(r) = S_1AU (AU / r_sun)^2`` with AU = 149 597 870 700 m (IAU 2012 B2).

    Args:
        sun_distance_m: Satellite - Sun-centre distance ``r_sun`` [m], > 0.
        solar_constant_1au_W_m2: ``S_1AU`` [W m^-2].

    Returns:
        ``S`` [W m^-2], shape of ``sun_distance_m``.
    """
    r = np.asarray(sun_distance_m, dtype=np.float64)
    _check_finite("sun_distance_m", r)
    if np.any(r <= 0.0):
        raise ValueError("sun_distance_m must be > 0")
    return np.asarray(solar_constant_1au_W_m2 * (AU_M / r) ** 2, dtype=np.float64)


def incidence_cosine(collector: CollectorModel, sun_direction_unit: ArrayLike) -> Float64Array:
    """Cosine of the solar incidence angle on the collector front face (module docstring).

    ``sun_tracking``: 1 at every epoch. ``fixed_normal``: ``max(n . s_hat, 0)`` with the J2000
    unit normal ``n`` and the satellite -> Sun unit vector ``s_hat``; back-face illumination
    (``n . s_hat < 0``) gives 0, i.e. zero collected power.

    Args:
        collector: Collector model (attitude mode and normal).
        sun_direction_unit: Satellite -> Sun unit vectors, J2000, shape (N, 3).

    Returns:
        ``cos(inc)`` in [0, 1], shape (N,).
    """
    s = np.asarray(sun_direction_unit, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError(f"sun_direction_unit must have shape (N, 3), got {s.shape}")
    _check_finite("sun_direction_unit", s)
    norms = np.linalg.norm(s, axis=1)
    if np.any(np.abs(norms - 1.0) > UNIT_NORMAL_TOLERANCE):
        raise ValueError("sun_direction_unit must contain unit vectors")
    if collector.incidence_mode == "sun_tracking":
        return np.ones(s.shape[0], dtype=np.float64)
    assert collector.fixed_normal_unit_j2000 is not None  # checked by CollectorModel
    n = np.asarray(collector.fixed_normal_unit_j2000, dtype=np.float64)
    return np.asarray(np.clip(s @ n, 0.0, 1.0), dtype=np.float64)


def incident_solar_power_W(
    sun_distance_m: ArrayLike,
    incidence_cosine: ArrayLike,
    illumination_fraction: ArrayLike,
    collector: CollectorModel,
) -> Float64Array:
    """Sunlight intercepted by the collector plane, ``S(r) A cos(inc) f`` [W] (PDF §8.1).

    Args:
        sun_distance_m: ``r_sun`` [m].
        incidence_cosine: ``cos(inc)`` in [0, 1].
        illumination_fraction: Visible solar-disk fraction ``f`` in [0, 1] (M3).
        collector: Collector model (area and ``S_1AU``).

    Returns:
        Intercepted power [W], broadcast shape of the inputs.
    """
    flux = solar_flux_W_m2(sun_distance_m, collector.solar_constant_1au_W_m2)
    cos_i = np.asarray(incidence_cosine, dtype=np.float64)
    frac = np.asarray(illumination_fraction, dtype=np.float64)
    _check_finite("incidence_cosine", cos_i)
    _check_finite("illumination_fraction", frac)
    if np.any((cos_i < 0.0) | (cos_i > 1.0)):
        raise ValueError("incidence_cosine must lie in [0, 1]")
    if np.any((frac < 0.0) | (frac > 1.0)):
        raise ValueError("illumination_fraction must lie in [0, 1]")
    return np.asarray(flux * collector.area_m2 * cos_i * frac, dtype=np.float64)


def absorbed_solar_power_W(
    sun_distance_m: ArrayLike,
    incidence_cosine: ArrayLike,
    illumination_fraction: ArrayLike,
    collector: CollectorModel,
) -> Float64Array:
    """``P_abs = S(r) A alpha cos(inc) f`` [W] (PDF §8.1, times the illumination fraction)."""
    incident = incident_solar_power_W(
        sun_distance_m, incidence_cosine, illumination_fraction, collector
    )
    return np.asarray(collector.absorptivity * incident, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class SolarPowerResult:
    """Solar collection and PV conversion, vectorised over epochs or distances (PDF p.4).

    Attributes:
        sun_distance_m: ``r_sun`` [m].
        flux_W_m2: ``S(r)`` [W m^-2].
        incidence_cosine: ``cos(inc)`` [1].
        illumination_fraction: Visible solar-disk fraction ``f`` [1].
        illuminated: ``f cos(inc) > 0``.
        incident_W: Sunlight intercepted by the collector plane ``S A cos(inc) f`` [W].
        reflected_W: ``(1 - alpha) incident_W`` [W]; leaves the spacecraft, not waste heat.
        absorbed_W: ``P_abs = alpha incident_W`` [W].
        pv_temperature_K: PV cell (plate) temperature used for the efficiency [K].
        pv_efficiency: ``eta_PV(T)`` relative to absorbed power [1].
        pv_efficiency_clipped: The clip of the linear law is active.
        pv_electric_W: ``P_PV = P_abs eta_PV(T)`` [W].
        pv_heat_W: Heat dissipated in the PV array ``Q_PV = P_abs - P_PV`` [W].
        collector: Collector model used.
        epochs_tdb_s: Epochs [TDB s past J2000] for a time series, or ``None`` for a sweep
            over heliocentric distance.
        frame: Frame of the Sun direction / collector normal (``J2000``).
        time_scale: Time scale of ``epochs_tdb_s`` (``TDB``).
    """

    sun_distance_m: Float64Array
    flux_W_m2: Float64Array
    incidence_cosine: Float64Array
    illumination_fraction: Float64Array
    illuminated: BoolArray
    incident_W: Float64Array
    reflected_W: Float64Array
    absorbed_W: Float64Array
    pv_temperature_K: Float64Array
    pv_efficiency: Float64Array
    pv_efficiency_clipped: BoolArray
    pv_electric_W: Float64Array
    pv_heat_W: Float64Array
    collector: CollectorModel = field(repr=False)
    epochs_tdb_s: Float64Array | None = None
    frame: FrameName = "J2000"
    time_scale: TimeScale = "TDB"

    def __post_init__(self) -> None:
        shape = np.shape(self.sun_distance_m)
        if len(shape) != 1:
            raise ValueError("SolarPowerResult arrays must be 1-D")
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                if value.shape != shape:
                    raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
                object.__setattr__(self, name, _frozen(value, value.dtype.type))
        if self.frame != "J2000" or self.time_scale != "TDB":
            raise ValueError("SolarPowerResult is J2000 / TDB by contract")


def solar_power(
    sun_distance_m: ArrayLike,
    incidence_cosine: ArrayLike,
    illumination_fraction: ArrayLike,
    pv_temperature_K: ArrayLike,
    collector: CollectorModel,
    *,
    epochs_tdb_s: ArrayLike | None = None,
) -> SolarPowerResult:
    """Solar collection and PV conversion at given cell temperatures (PDF §8.1-8.2).

    ``S = S_1AU (AU/r)^2``; ``incident = S A cos(inc) f``; ``P_abs = alpha incident``;
    ``reflected = (1 - alpha) incident``; ``P_PV = P_abs eta_PV(T)``; ``Q_PV = P_abs - P_PV``.
    The self-consistent cell temperature comes from the PV thermal node
    (:func:`helioswarm.thermal.solve_pv_temperature`).

    Args:
        sun_distance_m: ``r_sun`` [m], shape (N,).
        incidence_cosine: ``cos(inc)`` [1], broadcastable to (N,).
        illumination_fraction: ``f`` [1], broadcastable to (N,).
        pv_temperature_K: Cell temperature [K], broadcastable to (N,).
        collector: Collector model.
        epochs_tdb_s: Optional TDB epochs, shape (N,).

    Returns:
        :class:`SolarPowerResult`.
    """
    r = np.atleast_1d(np.asarray(sun_distance_m, dtype=np.float64))
    if r.ndim != 1:
        raise ValueError("sun_distance_m must be 1-D")
    shape = r.shape
    cos_i = np.broadcast_to(np.asarray(incidence_cosine, dtype=np.float64), shape)
    frac = np.broadcast_to(np.asarray(illumination_fraction, dtype=np.float64), shape)
    temp = np.broadcast_to(np.asarray(pv_temperature_K, dtype=np.float64), shape)
    _check_finite("pv_temperature_K", temp)
    if np.any(temp <= 0.0):
        raise ValueError("pv_temperature_K must be > 0")
    flux = solar_flux_W_m2(r, collector.solar_constant_1au_W_m2)
    incident = incident_solar_power_W(r, cos_i, frac, collector)
    absorbed = collector.absorptivity * incident
    reflected = (1.0 - collector.absorptivity) * incident
    eta = pv_efficiency(temp, collector.efficiency_law)
    electric = absorbed * eta
    epochs = None
    if epochs_tdb_s is not None:
        epochs = np.asarray(epochs_tdb_s, dtype=np.float64)
        if epochs.shape != shape:
            raise ValueError(f"epochs_tdb_s has shape {epochs.shape}, expected {shape}")
    return SolarPowerResult(
        sun_distance_m=r,
        flux_W_m2=flux,
        incidence_cosine=np.array(cos_i),
        illumination_fraction=np.array(frac),
        illuminated=np.asarray(cos_i * frac > 0.0),
        incident_W=incident,
        reflected_W=reflected,
        absorbed_W=absorbed,
        pv_temperature_K=np.array(temp),
        pv_efficiency=eta,
        pv_efficiency_clipped=collector.efficiency_law.is_clipped(temp),
        pv_electric_W=electric,
        pv_heat_W=absorbed - electric,
        collector=collector,
        epochs_tdb_s=epochs,
    )


__all__ = [
    "CollectorModel",
    "IncidenceMode",
    "PvEfficiencyLaw",
    "SolarPowerResult",
    "absorbed_solar_power_W",
    "collector_model_from_config",
    "incidence_cosine",
    "incident_solar_power_W",
    "pv_efficiency",
    "solar_flux_W_m2",
    "solar_power",
]
