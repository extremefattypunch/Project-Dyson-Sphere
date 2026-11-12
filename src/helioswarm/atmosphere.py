"""Atmospheric transmission of the terminal (near-Earth) segment of the beam (PDF §10.2-10.3).

The atmosphere is a short terminal segment at the receiver; its factor depends only on the
receiver site, the elevation of the arriving beam, the carrier and the weather model -- never
on the interplanetary range (PDF §10.3: "do not apply terrestrial attenuation formulas to the
interplanetary vacuum segment"). It is kept separate from diffraction capture and receiver
conversion (PDF §10.2) and is the single owner of ``eta_atmosphere`` in the PDF §10.1 chain
``P_atm = P_aperture * eta_atmosphere``.

Models (``models.atmosphere``)
------------------------------
* ``none``: ``eta = 1`` (no atmospheric loss; an upper bound).
* ``scenario`` (PDF §10.2 MVP: explicit clear/cloudy factors and an availability mask). Zenith
  values from the ``atmosphere`` section for ``models.weather_scenario`` are scaled with the
  **plane-parallel air mass** ``X(el) = 1 / sin(el)``:

  - optical: ``eta = tau_zenith^X``. A zenith transmittance of exactly 0 (default cloudy
    optical case) is a **hard outage**: ``eta = 0`` and ``weather_available = False``.
  - microwave: ``A_dB = A_zenith_dB * X``, ``eta = 10^(-A_dB/10)`` (never an outage).

  The plane-parallel ``X`` diverges at the horizon and exceeds the Kasten & Young (1989,
  Appl. Opt. 28, 4735) spherical-atmosphere air mass by 3.1 % at 10 deg, 11 % at 5 deg; the
  schema therefore requires ``receiver.min_elevation_deg >=
  atmosphere.plane_parallel_min_elevation_deg`` (default 10 deg). :func:`kasten_young_airmass`
  is provided as the named comparison path (not used by the scenario model).
* ``itu_r`` (microwave only; optional ``itur`` extra): ITU-R Earth-space slant-path attenuation
  ``A = A_gas + A_cloud + A_rain`` exceeded ``atmosphere.itu_r_exceedance_percent`` of an
  average year at the site, from the ``itur`` package (ITU-Rpy, I. del Portillo, MIT licence):
  P.618-13 (rain, combination), P.676 (gases, Annex 2 approximation), P.840-7 (clouds), with
  the auxiliary P.453, P.835, P.836, P.837, P.838, P.839, P.1510 and P.1511 maps bundled in the
  package (nothing is downloaded). The recommendation versions used are pinned on every call
  (:data:`ITU_R_VERSIONS`) and returned in the result for the run manifest (PDF §10.3).
  Tropospheric **scintillation is excluded**: P.618's scintillation model is defined for
  earth-station antennas of a few metres (antenna-averaging factor), not for kilometre-scale
  rectennas whose aperture averages it out; this is a documented limitation. The attenuation
  is a long-term *statistic* applied deterministically at every epoch (a design level), not a
  weather time series. Site heights are the configured WGS84 ellipsoidal heights passed as
  P.618 "height above mean sea level" (geoid undulation, |N| < ~110 m, neglected). The rain
  polarisation tilt ``tau`` (P.838/P.618; 45 deg = circular) is the configured
  ``atmosphere.itu_r_polarization_tilt_deg`` and is returned in the result for the manifest.

  *Validation status.* The wrapper is validated against the ITU-R SG3 validation examples
  with P.676-11 and P.1511-1 pinned (the revisions of those examples, as bundled with
  ``itur`` 0.4.0). Production runs pin P.676-12 and P.1511-2 (:data:`ITU_R_VERSIONS`), for
  which ``itur`` ships no validation set: the production gas component is regression-pinned,
  not validated (``docs/validation_report.md``).

  *Thread safety.* ``itur`` keeps the selected recommendation revisions as library-global
  state and the warnings capture uses the process-global filter. The wrapper therefore holds a
  module lock (:data:`_ITUR_LOCK`) for the whole call, pins the revisions on entry and
  restores the previous ones on exit, so concurrent wrapper calls in threads are serialised
  and never see each other's revisions. Code that calls ``itur`` directly in other threads
  is not protected.

Elevations below the model's validity floor (or <= 0) give ``NaN`` transmission with
``in_validity = False``: such epochs are below the receiver's minimum elevation anyway (the
schema enforces the floor) and must be masked by the access model, never silently attenuated.
"""

from __future__ import annotations

import math
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.config import AtmosphereModel, RunConfig, SiteConfig, TransmitterMode
from helioswarm.constants import SPEED_OF_LIGHT_M_S

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

WeatherScenario = Literal["clear", "cloudy"]

ITU_R_VERSIONS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "itu618": 13,
        "itu676": 12,
        "itu840": 7,
        "itu453": 13,
        "itu835": 6,
        "itu836": 6,
        "itu837": 7,
        "itu838": 3,
        "itu839": 4,
        "itu1510": 1,
        "itu1511": 2,
    }
)
"""ITU-R recommendation revisions used by ``models.atmosphere = "itu_r"`` (the newest that
``itur`` 0.4.0 implements; e.g. ``itu618: 13`` = ITU-R P.618-13). Read-only. Newer in-force
revisions exist (e.g. P.618-14, P.676-13); see ``docs/data_provenance.md``. P.676-12 and
P.1511-2 have no validation set in ``itur`` 0.4.0 (module docstring)."""

_ITUR_LOCK: Final[threading.Lock] = threading.Lock()
"""Serialises calls into ``itur`` (library-global revision state; module docstring)."""


def plane_parallel_airmass(elevation_rad: ArrayLike) -> Float64Array:
    """Plane-parallel air mass ``X = 1 / sin(el)`` [1] (valid at high elevation only)."""
    el = np.asarray(elevation_rad, dtype=np.float64)
    out: Float64Array = 1.0 / np.sin(el)
    return out


def kasten_young_airmass(elevation_rad: ArrayLike) -> Float64Array:
    """Kasten & Young (1989, Appl. Opt. 28, 4735) relative optical air mass (comparison path).

    ``X = 1 / (sin h + 0.50572 (h_deg + 6.07995)^-1.6364)`` with ``h`` the (apparent) elevation.
    Not used by the ``scenario`` model; provided to quantify the plane-parallel error.
    """
    el = np.asarray(elevation_rad, dtype=np.float64)
    h_deg = np.degrees(el)
    out: Float64Array = 1.0 / (np.sin(el) + 0.50572 * (h_deg + 6.07995) ** -1.6364)
    return out


@dataclass(frozen=True)
class AtmosphereSettings:
    """Inputs of the atmosphere model (one owner of ``eta_atmosphere``).

    Attributes:
        model: ``none``, ``scenario`` or ``itu_r``.
        mode: ``optical`` or ``microwave``.
        weather_scenario: ``clear`` or ``cloudy`` (``scenario`` model only).
        optical_zenith_transmittance: Zenith transmittance of the selected scenario [1].
        microwave_zenith_attenuation_dB: Zenith attenuation of the selected scenario [dB].
        min_valid_elevation_rad: Validity floor of the model's elevation dependence [rad].
        wavelength_m: Carrier wavelength [m].
        itu_r_exceedance_percent: ``p`` [%] for ``itu_r``.
        itu_r_polarization_tilt_deg: Rain polarisation tilt ``tau`` [deg] for ``itu_r``.
    """

    model: AtmosphereModel
    mode: TransmitterMode
    weather_scenario: WeatherScenario
    optical_zenith_transmittance: float
    microwave_zenith_attenuation_dB: float
    min_valid_elevation_rad: float
    wavelength_m: float
    itu_r_exceedance_percent: float = 1.0
    itu_r_polarization_tilt_deg: float = 45.0

    @property
    def hard_outage(self) -> bool:
        """True when the selected scenario blocks the link entirely (optical, tau_zenith = 0)."""
        return (
            self.model == "scenario"
            and self.mode == "optical"
            and self.optical_zenith_transmittance == 0.0
        )


def atmosphere_settings_from_config(config: RunConfig) -> AtmosphereSettings:
    """:class:`AtmosphereSettings` from a validated run configuration."""
    atm = config.atmosphere
    cloudy = config.models.weather_scenario == "cloudy"
    if config.models.atmosphere == "itu_r":
        floor_deg = atm.itu_r_min_elevation_deg
    else:
        floor_deg = atm.plane_parallel_min_elevation_deg
    return AtmosphereSettings(
        model=config.models.atmosphere,
        mode=config.transmitter.mode,
        weather_scenario=config.models.weather_scenario,
        optical_zenith_transmittance=(
            atm.optical_zenith_transmittance_cloudy
            if cloudy
            else atm.optical_zenith_transmittance_clear
        ),
        microwave_zenith_attenuation_dB=(
            atm.microwave_zenith_attenuation_cloudy_dB
            if cloudy
            else atm.microwave_zenith_attenuation_clear_dB
        ),
        min_valid_elevation_rad=math.radians(floor_deg),
        wavelength_m=config.transmitter.wavelength_m,
        itu_r_exceedance_percent=atm.itu_r_exceedance_percent,
        itu_r_polarization_tilt_deg=atm.itu_r_polarization_tilt_deg,
    )


@dataclass(frozen=True, eq=False)
class AtmosphereResult:
    """Atmospheric transmission of the terminal segment, per epoch (shape (N,)).

    Attributes:
        transmittance: ``eta_atmosphere = P_atm / P_aperture`` in [0, 1]; NaN outside validity.
        attenuation_dB: ``-10 log10(transmittance)`` [dB] (inf for a hard outage).
        airmass: Plane-parallel ``1/sin(el)`` (``scenario``); NaN for other models.
        weather_available: False where the weather model makes the link unavailable
            (hard optical outage); True otherwise.
        in_validity: Elevation within the model's validity range.
        elevation_rad: Elevations used [rad].
        model, mode, weather_scenario: Settings used.
        gas_attenuation_dB, cloud_attenuation_dB, rain_attenuation_dB: ITU-R components
            (``itu_r`` only, else ``None``).
        recommendation_versions: ITU-R recommendations used (``itu_r``), else empty.
        polarization_tilt_deg: Rain polarisation tilt ``tau`` used (``itu_r``), else ``None``.
        notes: Warnings raised by the external model (recorded, not suppressed).
    """

    transmittance: Float64Array
    attenuation_dB: Float64Array
    airmass: Float64Array
    weather_available: BoolArray
    in_validity: BoolArray
    elevation_rad: Float64Array
    model: AtmosphereModel
    mode: TransmitterMode
    weather_scenario: WeatherScenario
    gas_attenuation_dB: Float64Array | None = None
    cloud_attenuation_dB: Float64Array | None = None
    rain_attenuation_dB: Float64Array | None = None
    recommendation_versions: tuple[str, ...] = ()
    polarization_tilt_deg: float | None = None
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        shape = np.shape(self.elevation_rad)
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                if value.shape != shape:
                    raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
                arr = np.array(value, copy=True)
                arr.setflags(write=False)
                object.__setattr__(self, name, arr)
        tr = self.transmittance[self.in_validity]
        if np.any(np.isnan(tr)) or np.any(tr < 0.0) or np.any(tr > 1.0):
            raise ValueError("transmittance must lie in [0, 1] wherever in_validity is True")


def atmospheric_transmission(
    elevation_rad: ArrayLike,
    settings: AtmosphereSettings,
    *,
    site: SiteConfig | None = None,
) -> AtmosphereResult:
    """Atmospheric transmission at the given beam elevations (module docstring).

    Args:
        elevation_rad: Apparent elevation of the satellite at the receiver [rad], shape (N,).
        settings: Atmosphere settings.
        site: Receiver site (required for ``itu_r``).

    Returns:
        :class:`AtmosphereResult`. Depends on elevation (and site/carrier) only, never on range.
    """
    el = np.atleast_1d(np.asarray(elevation_rad, dtype=np.float64))
    if np.any(~np.isfinite(el)):
        raise ValueError("elevation_rad must be finite")
    valid = (el >= settings.min_valid_elevation_rad - 1e-15) & (el <= 0.5 * math.pi + 1e-12)
    nan = np.full(el.shape, np.nan)
    if settings.model == "none":
        valid = el > 0.0
        tr = np.where(valid, 1.0, np.nan)
        return AtmosphereResult(
            transmittance=tr,
            attenuation_dB=np.where(valid, 0.0, np.nan),
            airmass=nan,
            weather_available=np.ones(el.shape, dtype=bool),
            in_validity=valid,
            elevation_rad=el,
            model="none",
            mode=settings.mode,
            weather_scenario=settings.weather_scenario,
        )
    if settings.model == "scenario":
        x = np.where(valid, plane_parallel_airmass(np.where(valid, el, 0.5 * math.pi)), np.nan)
        if settings.mode == "optical":
            tau_z = settings.optical_zenith_transmittance
            tr = np.where(valid, tau_z**x if tau_z > 0.0 else 0.0, np.nan)
            with np.errstate(divide="ignore"):
                att = np.where(valid, -10.0 * np.log10(np.where(valid, tr, 1.0)), np.nan)
        else:
            att = np.where(valid, settings.microwave_zenith_attenuation_dB * x, np.nan)
            tr = np.where(valid, 10.0 ** (-att / 10.0), np.nan)
        available = np.full(el.shape, not settings.hard_outage)
        return AtmosphereResult(
            transmittance=tr,
            attenuation_dB=att,
            airmass=x,
            weather_available=available,
            in_validity=valid,
            elevation_rad=el,
            model="scenario",
            mode=settings.mode,
            weather_scenario=settings.weather_scenario,
        )
    if settings.model == "itu_r":
        if site is None:
            raise ValueError("models.atmosphere='itu_r' needs the receiver site")
        comps = itu_r_slant_path_attenuation_dB(
            site.latitude_deg,
            site.longitude_deg,
            site.altitude_m,
            SPEED_OF_LIGHT_M_S / settings.wavelength_m,
            np.where(valid, el, 0.5 * math.pi),
            settings.itu_r_exceedance_percent,
            polarization_tilt_deg=settings.itu_r_polarization_tilt_deg,
        )
        att = np.where(valid, comps.total_dB, np.nan)
        tr = np.where(valid, 10.0 ** (-att / 10.0), np.nan)
        return AtmosphereResult(
            transmittance=tr,
            attenuation_dB=att,
            airmass=nan,
            weather_available=np.ones(el.shape, dtype=bool),
            in_validity=valid,
            elevation_rad=el,
            model="itu_r",
            mode=settings.mode,
            weather_scenario=settings.weather_scenario,
            gas_attenuation_dB=np.where(valid, comps.gas_dB, np.nan),
            cloud_attenuation_dB=np.where(valid, comps.cloud_dB, np.nan),
            rain_attenuation_dB=np.where(valid, comps.rain_dB, np.nan),
            recommendation_versions=comps.recommendation_versions,
            polarization_tilt_deg=settings.itu_r_polarization_tilt_deg,
            notes=comps.notes,
        )
    raise ValueError(f"unknown atmosphere model '{settings.model}'")


@dataclass(frozen=True, eq=False)
class ItuRAttenuation:
    """ITU-R slant-path attenuation components [dB] (``itur``), shape (N,)."""

    gas_dB: Float64Array
    cloud_dB: Float64Array
    rain_dB: Float64Array
    total_dB: Float64Array
    recommendation_versions: tuple[str, ...]
    itur_version: str
    notes: tuple[str, ...]


def itu_r_slant_path_attenuation_dB(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    frequency_Hz: float,
    elevation_rad: ArrayLike,
    exceedance_percent: float,
    *,
    versions: Mapping[str, int] = ITU_R_VERSIONS,
    rainfall_rate_001_mm_h: float | None = None,
    polarization_tilt_deg: float = 45.0,
) -> ItuRAttenuation:
    """Earth-space attenuation ``A_gas + A_cloud + A_rain`` exceeded ``p`` % of an average year.

    Wraps ``itur.atmospheric_attenuation_slant_path`` (signature verified from the installed
    ``itur`` 0.4.0 source: ``(lat, lon, f, el, p, D, hs=None, rho=None, R001=None, eta=0.5,
    T=None, H=None, P=None, hL=1000.0, Ls=None, tau=45, V_t=None, mode='approx',
    return_contributions=False, include_rain=True, include_gas=True,
    include_scintillation=True, include_clouds=True)``; frequency in GHz, elevation in degrees,
    ``hs`` in km; returns ``(A_g, A_c, A_r, A_s, A)`` as astropy dB quantities). Scintillation
    is excluded (module docstring), so ``A = A_g + A_c + A_r`` (P.618-13 §2.5 combination with
    ``A_s = 0``). ``polarization_tilt_deg`` is P.618's tau in [0, 90] deg (45 = circular
    polarisation, 0 = horizontal linear; from ``atmosphere.itu_r_polarization_tilt_deg`` in
    runs). ``rainfall_rate_001_mm_h`` overrides the P.837 rain rate exceeded 0.01 % of the time
    (None = P.837 map).

    ``itur`` stores the selected recommendation revisions as library-global state; under
    :data:`_ITUR_LOCK` this wrapper sets every revision in ``versions``, evaluates, and restores
    the revisions that were selected before the call, so results never depend on call order
    and the caller's ``itur`` state is left unchanged. Warnings emitted by ``itur`` are
    captured and returned in ``notes``.

    Raises:
        ImportError: the optional ``itur`` extra is not installed.
    """
    try:
        import itur
        import itur.models as itu_models
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "models.atmosphere='itu_r' needs the optional dependency 'itur' (uv sync --extra itur)"
        ) from exc
    if not (math.isfinite(polarization_tilt_deg) and 0.0 <= polarization_tilt_deg <= 90.0):
        raise ValueError("polarization_tilt_deg must lie in [0, 90]")
    el_deg = np.degrees(np.atleast_1d(np.asarray(elevation_rad, dtype=np.float64)))
    kwargs: dict[str, object] = {}
    if rainfall_rate_001_mm_h is not None:
        kwargs["R001"] = rainfall_rate_001_mm_h
    with _ITUR_LOCK:
        previous = {name: getattr(itu_models, name).get_version() for name in versions}
        try:
            for name, rev in versions.items():
                getattr(itu_models, name).change_version(rev)
            rev_list = tuple(
                f"ITU-R P.{name.removeprefix('itu')}-{getattr(itu_models, name).get_version()}"
                for name in versions
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                a_g, a_c, a_r, _a_s, a_t = itur.atmospheric_attenuation_slant_path(
                    latitude_deg,
                    longitude_deg,
                    frequency_Hz / 1.0e9,
                    el_deg,
                    exceedance_percent,
                    1.0,
                    hs=altitude_m / 1000.0,
                    tau=polarization_tilt_deg,
                    return_contributions=True,
                    include_scintillation=False,
                    **kwargs,
                )
        finally:
            for name, rev in previous.items():
                getattr(itu_models, name).change_version(rev)

    def _arr(q: object) -> Float64Array:
        out: Float64Array = np.broadcast_to(
            np.asarray(getattr(q, "value", q), dtype=np.float64), el_deg.shape
        ).copy()
        return out

    return ItuRAttenuation(
        gas_dB=_arr(a_g),
        cloud_dB=_arr(a_c),
        rain_dB=_arr(a_r),
        total_dB=_arr(a_t),
        recommendation_versions=rev_list,
        itur_version=str(itur.__version__),
        notes=tuple(sorted({str(w.message) for w in caught})),
    )


__all__ = [
    "ITU_R_VERSIONS",
    "AtmosphereResult",
    "AtmosphereSettings",
    "ItuRAttenuation",
    "atmosphere_settings_from_config",
    "atmospheric_transmission",
    "itu_r_slant_path_attenuation_dB",
    "kasten_young_airmass",
    "plane_parallel_airmass",
]
