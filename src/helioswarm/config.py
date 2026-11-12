"""Run-configuration schema (pydantic v2), YAML loading, resolved dumps and config hashing.

One YAML file defines every physical parameter of a run (PDF §4 engineering rules, §5
configuration contract). This module is the *only* place where configuration-boundary units
(au, degrees, days, dB) appear; every section exposes SI accessors (``*_m``, ``*_s``, ``*_rad``)
for the physics modules.

Schema conventions
------------------
* Every field name carries its unit as a suffix (``_m``, ``_s``, ``_K``, ``_W``, ``_W_m2``,
  ``_deg``, ``_au``, ``_days``, ``_dB``, ``_per_K``, ``_m2_kg``); dimensionless quantities
  (efficiencies, emissivities, eccentricity, ``cr``) carry no suffix.
* Every field carries ``json_schema_extra`` metadata: ``provenance`` (one of ``measured``,
  ``sourced``, ``assumed``, ``swept``; PDF §13.4), ``description``, ``unit`` and ``kind``
  (``physical``, ``numerical``, ``model_choice`` or ``metadata``); ``sourced`` defaults also
  carry ``source``. The provenance in the schema describes the *default*; a run may relabel a
  value through ``provenance_overrides`` (e.g. ``swept`` in a parameter sweep).
* Models are frozen, reject unknown keys (``extra="forbid"``), are strict (no silent
  string-to-number coercion) and reject NaN/inf.
* ``schema_version`` is incremented on any backward-incompatible schema change so that old
  resolved configs remain interpretable.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, Self, cast, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_core import PydanticUndefined

from helioswarm.constants import AU_M, SECONDS_PER_DAY, SOLAR_RADIUS_NOMINAL_M

SCHEMA_VERSION: Final[int] = 1
"""Configuration schema version understood by this package."""

GRID_END_REL_TOL: Final[float] = 8.0 * sys.float_info.epsilon
"""Relative tolerance (of the window length) on ``n * output_step_s == duration_s``: a few
float64 rounding units of the two products, nothing more."""

Provenance = Literal["measured", "sourced", "assumed", "swept"]
FieldKind = Literal["physical", "numerical", "model_choice", "metadata"]

TransmitterMode = Literal["optical", "microwave"]
BeamModel = Literal["airy_far_field", "gaussian_beam", "fresnel_hankel", "aperture_far_field"]
ApertureIllumination = Literal["uniform_circular", "truncated_gaussian", "parabolic_on_pedestal"]
ReceiverTechnology = Literal["laser_photovoltaic", "rectenna"]

OPTICAL_BEAM_MODELS: Final[frozenset[str]] = frozenset(
    {"airy_far_field", "gaussian_beam", "fresnel_hankel"}
)
"""Beam models allowed with ``transmitter.mode == "optical"``."""

MICROWAVE_BEAM_MODELS: Final[frozenset[str]] = frozenset({"aperture_far_field", "fresnel_hankel"})
"""Beam models allowed with ``transmitter.mode == "microwave"``."""

OPTICAL_WAVELENGTH_RANGE_M: Final[tuple[float, float]] = (1.0e-7, 2.0e-5)
"""Wavelengths accepted for optical (laser) transmission: 100 nm (UV) to 20 um (thermal IR)."""

MICROWAVE_WAVELENGTH_RANGE_M: Final[tuple[float, float]] = (1.0e-3, 1.0)
"""Wavelengths accepted for microwave transmission: 1 mm to 1 m (300 GHz to 300 MHz)."""

RECEIVER_TECHNOLOGY_FOR_MODE: Final[Mapping[str, str]] = {
    "optical": "laser_photovoltaic",
    "microwave": "rectenna",
}
"""Receiver technology required by each transmitter mode (PDF §10.4: optical and microwave
receivers are different technology cases)."""


# ---------------------------------------------------------------------------------------------
# Field metadata helpers
# ---------------------------------------------------------------------------------------------


def _field(
    default: Any = PydanticUndefined,
    *,
    unit: str,
    provenance: Provenance,
    description: str,
    kind: FieldKind = "physical",
    source: str | None = None,
    **constraints: Any,
) -> Any:
    """Build a pydantic ``Field`` carrying the helioswarm metadata in ``json_schema_extra``.

    Omitting ``default`` makes the field required.
    """
    extra: dict[str, str] = {
        "provenance": provenance,
        "description": description,
        "unit": unit,
        "kind": kind,
    }
    if source is not None:
        extra["source"] = source
    return Field(
        default,
        description=description,
        json_schema_extra=cast("dict[str, Any]", extra),
        **constraints,
    )


class _Section(BaseModel):
    """Base class of every configuration section: frozen, strict, no unknown keys, finite."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


_ISSUE_SEPARATOR: Final[str] = "\n"
"""Separator between cross-field issues inside one ``ValueError`` (split again for reporting)."""


def _raise_if(issues: list[str]) -> None:
    """Raise one ``ValueError`` carrying every collected cross-field issue."""
    if issues:
        raise ValueError(_ISSUE_SEPARATOR.join(issues))


_UTC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(\.\d{1,6})?Z?$"
)


def _canonical_utc(value: str) -> str:
    """Validate an ISO-8601 UTC epoch string and return its canonical form.

    Accepted: ``YYYY-MM-DDTHH:MM:SS[.ffffff][Z]`` (a space may replace ``T``). Time-zone offsets
    are rejected (the name ``*_utc`` fixes the time scale). Canonical form uses ``T``, no ``Z``,
    and drops an all-zero fraction, so equivalent spellings hash identically.
    """
    match = _UTC_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(
            f"{value!r} is not an ISO-8601 UTC epoch 'YYYY-MM-DDTHH:MM:SS[.ffffff]' "
            "(no time-zone offset; the field is UTC by definition)"
        )
    year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
    fraction = match.group(7) or ""
    if second == 60:
        raise ValueError(f"{value!r}: leap-second epochs (ss=60) are not accepted as config epochs")
    try:
        datetime(year, month, day, hour, minute, second)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not a valid calendar epoch: {exc}") from None
    fraction = fraction.rstrip("0")
    if fraction == ".":
        fraction = ""
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}{fraction}"


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


class ProjectConfig(_Section):
    """Run identification and reproducibility settings."""

    name: str = _field(
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Run/scenario name; used to label outputs (letters, digits, '_', '.', '-').",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        max_length=128,
    )
    seed: int = _field(
        unit="-",
        provenance="assumed",
        kind="numerical",
        description="Seed for the numpy Generator passed explicitly to every stochastic step.",
        ge=0,
    )
    output_dir: str | None = _field(
        None,
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Suggested run output directory; not used by validate-config.",
    )
    description: str = _field(
        "",
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Free-text description of the scenario.",
    )


class TimeConfig(_Section):
    """Simulation time window. Epochs are UTC strings; durations and steps are SI seconds,
    except ``duration_days`` (1 day = 86400 s exactly)."""

    start_utc: str = _field(
        unit="UTC ISO-8601",
        provenance="assumed",
        kind="numerical",
        description="Start epoch of the output time grid (UTC; converted to TDB via SPICE).",
    )
    duration_days: float = _field(
        unit="day",
        provenance="assumed",
        kind="numerical",
        description="Length of the simulated window; 1 day = 86400 s. Must be an integer "
        "multiple of output_step_s.",
        gt=0.0,
        le=36525.0,
    )
    output_step_s: float = _field(
        unit="s",
        provenance="assumed",
        kind="numerical",
        description="Spacing of the saved output time grid.",
        gt=0.0,
    )
    integrator_step_s: float | None = _field(
        None,
        unit="s",
        provenance="assumed",
        kind="numerical",
        description="Maximum step size of the adaptive L1 orbit integrator, passed as "
        "scipy.integrate.solve_ivp(max_step=...); the integrator may take smaller steps. "
        "Required when orbit.model is 'prescribed_ephemeris_perturbed_test_particle' and "
        "rejected otherwise (the analytic two-body model has no integrator). Integration "
        "tolerances (rtol/atol) are added with the L1 propagator (M2).",
        gt=0.0,
    )

    @field_validator("start_utc")
    @classmethod
    def _check_start(cls, value: str) -> str:
        return _canonical_utc(value)

    @model_validator(mode="after")
    def _check_grid(self) -> Self:
        issues: list[str] = []
        duration_s = self.duration_s
        if self.output_step_s > duration_s:
            issues.append(
                f"time.output_step_s={self.output_step_s} exceeds the window length "
                f"time.duration_days={self.duration_days} (= {duration_s} s)"
            )
        else:
            # Compare the reconstructed grid end with the window end in seconds. The tolerance
            # only absorbs float64 rounding of duration_days*86400 and n*output_step_s (each
            # <= 1 ulp of duration_s, i.e. <= eps*duration_s); 8*eps*duration_s is 0.7 us for
            # the longest allowed window (100 yr), so any real mismatch is rejected.
            ratio = duration_s / self.output_step_s
            n = round(ratio)
            residual_s = duration_s - n * self.output_step_s
            if abs(residual_s) > GRID_END_REL_TOL * duration_s:
                issues.append(
                    f"time.duration_days={self.duration_days} (= {duration_s} s) is not an "
                    f"integer multiple of time.output_step_s={self.output_step_s} "
                    f"(ratio {ratio:.6f}; {n} steps end {residual_s:.6g} s before the window "
                    "end); the output grid must end exactly at the window end"
                )
        _raise_if(issues)
        return self

    @property
    def duration_s(self) -> float:
        """Window length [s] = duration_days * 86400."""
        return self.duration_days * SECONDS_PER_DAY

    @property
    def n_output_intervals(self) -> int:
        """Number of output intervals, duration_s / output_step_s (an integer by validation)."""
        return round(self.duration_s / self.output_step_s)


class FramesConfig(_Section):
    """Reference frames and origin. Only the combinations implemented and validated are
    accepted; every state in the code carries its frame and origin explicitly."""

    inertial: Literal["J2000"] = _field(
        "J2000",
        unit="-",
        provenance="sourced",
        kind="model_choice",
        source="NAIF SPICE Frames Required Reading (frame name 'J2000')",
        description="Inertial frame of all internal states (SPICE frame name).",
    )
    origin: Literal["SOLAR SYSTEM BARYCENTER"] = _field(
        "SOLAR SYSTEM BARYCENTER",
        unit="-",
        provenance="sourced",
        kind="model_choice",
        source="NAIF SPICE NAIF IDs Required Reading (body 0)",
        description="Origin of all internal inertial states (SPICE body name).",
    )
    earth_fixed: Literal["IAU_EARTH"] = _field(
        "IAU_EARTH",
        unit="-",
        provenance="sourced",
        kind="model_choice",
        source="NAIF SPICE PCK Required Reading; pck00011.tpc",
        description="Earth-fixed frame used to rotate receiver sites into the inertial frame "
        "(IAU rotation model from the text PCK; lower accuracy than ITRF93).",
    )
    earth_ellipsoid: Literal["WGS84"] = _field(
        "WGS84",
        unit="-",
        provenance="sourced",
        kind="model_choice",
        source="NGA.STND.0036 (WGS 84)",
        description="Reference ellipsoid for geodetic receiver coordinates and Earth blockage.",
    )


class OrbitConfig(_Section):
    """Initial osculating heliocentric orbit of the (single) satellite and force-model flags.

    Elements are Sun-centred and referred to the ECLIPJ2000 frame at ``initial_epoch_utc``;
    they are converted to a J2000/SSB state before propagation (M1/M2).
    """

    model: Literal["two_body_kepler", "prescribed_ephemeris_perturbed_test_particle"] = _field(
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Orbit propagation level: 'two_body_kepler' (L0, analytic Sun-only) or "
        "'prescribed_ephemeris_perturbed_test_particle' (L1, numerical with SPICE bodies).",
    )
    initial_epoch_utc: str = _field(
        unit="UTC ISO-8601",
        provenance="assumed",
        kind="numerical",
        description="Epoch at which the orbital elements are osculating (UTC).",
    )
    elements_frame: Literal["ECLIPJ2000"] = _field(
        "ECLIPJ2000",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Reference plane of the orbital elements (SPICE frame name).",
    )
    elements_center: Literal["SUN"] = _field(
        "SUN",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Central body of the orbital elements.",
    )
    semimajor_axis_au: float = _field(
        unit="au",
        provenance="assumed",
        description="Semimajor axis (1 au = 149597870700 m, IAU 2012 B2).",
        gt=0.0,
        le=100.0,
    )
    eccentricity: float = _field(
        0.0,
        unit="1",
        provenance="assumed",
        description="Eccentricity; elliptic orbits only (0 <= e < 1).",
        ge=0.0,
        lt=1.0,
    )
    inclination_deg: float = _field(
        0.0,
        unit="deg",
        provenance="assumed",
        description="Inclination to the ecliptic of J2000.",
        ge=0.0,
        le=180.0,
    )
    raan_deg: float = _field(
        0.0,
        unit="deg",
        provenance="assumed",
        description="Longitude of the ascending node.",
        ge=0.0,
        lt=360.0,
    )
    arg_periapsis_deg: float = _field(
        0.0,
        unit="deg",
        provenance="assumed",
        description="Argument of periapsis.",
        ge=0.0,
        lt=360.0,
    )
    true_anomaly_deg: float = _field(
        unit="deg",
        provenance="assumed",
        description="True anomaly at initial_epoch_utc.",
        ge=0.0,
        lt=360.0,
    )
    include_planetary_perturbations: bool = _field(
        False,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Include planetary third-body accelerations (L1 model only).",
    )
    include_solar_radiation_pressure: bool = _field(
        False,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Include radial solar-radiation-pressure acceleration (L1 model only).",
    )
    cr: float | None = _field(
        None,
        unit="1",
        provenance="assumed",
        description="Radiation-pressure coefficient C_r of the cannonball/flat-plate-normal "
        "model (1 = fully absorbing, 2 = fully specular normal reflector); required with SRP "
        "and rejected without it.",
        ge=1.0,
        le=2.0,
    )
    area_to_mass_m2_kg: float | None = _field(
        None,
        unit="m^2 kg^-1",
        provenance="assumed",
        description="Sun-facing area to mass ratio A/m used by SRP; required with SRP and "
        "rejected without it.",
        gt=0.0,
    )

    @field_validator("initial_epoch_utc")
    @classmethod
    def _check_epoch(cls, value: str) -> str:
        return _canonical_utc(value)

    @model_validator(mode="after")
    def _check_orbit(self) -> Self:
        issues: list[str] = []
        perihelion_m = self.perihelion_m
        if perihelion_m <= SOLAR_RADIUS_NOMINAL_M:
            issues.append(
                f"orbit perihelion a(1-e) = {perihelion_m:.6g} m "
                f"(orbit.semimajor_axis_au={self.semimajor_axis_au}, "
                f"orbit.eccentricity={self.eccentricity}) is inside the Sun "
                f"(nominal solar radius {SOLAR_RADIUS_NOMINAL_M:.6g} m, IAU 2015 B3)"
            )
        if self.model == "two_body_kepler":
            if self.include_planetary_perturbations:
                issues.append(
                    "orbit.model='two_body_kepler' is Sun-only two-body motion; it contradicts "
                    "orbit.include_planetary_perturbations=true (use "
                    "'prescribed_ephemeris_perturbed_test_particle')"
                )
            if self.include_solar_radiation_pressure:
                issues.append(
                    "orbit.model='two_body_kepler' has no non-gravitational forces; it "
                    "contradicts orbit.include_solar_radiation_pressure=true"
                )
        if self.include_solar_radiation_pressure:
            if self.cr is None:
                issues.append("orbit.cr is required when include_solar_radiation_pressure=true")
            if self.area_to_mass_m2_kg is None:
                issues.append(
                    "orbit.area_to_mass_m2_kg is required when include_solar_radiation_pressure"
                    "=true"
                )
        else:
            for name in ("cr", "area_to_mass_m2_kg"):
                if getattr(self, name) is not None:
                    issues.append(
                        f"orbit.{name} is set but include_solar_radiation_pressure=false "
                        "ignores it; remove it or set include_solar_radiation_pressure=true"
                    )
        _raise_if(issues)
        return self

    @property
    def semimajor_axis_m(self) -> float:
        """Semimajor axis [m]."""
        return self.semimajor_axis_au * AU_M

    @property
    def perihelion_m(self) -> float:
        """Perihelion distance a(1 - e) [m]."""
        return self.semimajor_axis_m * (1.0 - self.eccentricity)

    @property
    def inclination_rad(self) -> float:
        """Inclination [rad]."""
        return math.radians(self.inclination_deg)

    @property
    def raan_rad(self) -> float:
        """Longitude of the ascending node [rad]."""
        return math.radians(self.raan_deg)

    @property
    def arg_periapsis_rad(self) -> float:
        """Argument of periapsis [rad]."""
        return math.radians(self.arg_periapsis_deg)

    @property
    def true_anomaly_rad(self) -> float:
        """True anomaly at the initial epoch [rad]."""
        return math.radians(self.true_anomaly_deg)


class CollectorConfig(_Section):
    """Photovoltaic collector (PDF §8.1-8.2).

    ``P_abs = S(r) * area_m2 * absorptivity * cos(incidence)`` and ``P_PV = P_abs * eta_PV(T)``:
    following PDF §8.1 the PV efficiency is defined relative to *absorbed* power.
    ``eta_PV(T) = clip(eta_ref * (1 + gamma_T * (T - T_ref)), eta_min, eta_max)`` (PDF §8.2) is a
    labelled local linear approximation.
    """

    area_m2: float = _field(unit="m^2", provenance="assumed", description="Collector area.", gt=0.0)
    pv_efficiency_reference: float = _field(
        unit="1",
        provenance="assumed",
        description="PV conversion efficiency eta_ref at reference_temperature_K, defined as "
        "P_PV / P_abs (PDF §8.1).",
        gt=0.0,
        le=1.0,
    )
    reference_temperature_K: float = _field(
        unit="K",
        provenance="assumed",
        description="Cell temperature T_ref at which eta_ref applies.",
        gt=0.0,
    )
    temperature_coefficient_per_K: float = _field(
        unit="K^-1",
        provenance="assumed",
        description="Relative efficiency temperature coefficient gamma_T in "
        "eta_ref*(1+gamma_T*(T-T_ref)); sanity range |gamma_T| <= 0.01 K^-1.",
        ge=-0.01,
        le=0.01,
    )
    pv_efficiency_min: float = _field(
        0.0,
        unit="1",
        provenance="assumed",
        description="Lower clip eta_min of the linear efficiency law (default 0: no "
        "negative efficiency).",
        ge=0.0,
        le=1.0,
    )
    pv_efficiency_max: float = _field(
        1.0,
        unit="1",
        provenance="assumed",
        description="Upper clip eta_max of the linear efficiency law. The default 1.0 is only "
        "the energy-conservation bound; set a technology-specific limit for real studies.",
        gt=0.0,
        le=1.0,
    )
    absorptivity: float = _field(
        unit="1",
        provenance="assumed",
        description="Solar absorptance alpha of the collector front surface.",
        gt=0.0,
        le=1.0,
    )
    incidence_mode: Literal["sun_tracking", "fixed_normal"] = _field(
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="'sun_tracking' (normal always on the Sun line; optimistic bound) or "
        "'fixed_normal' (inertially fixed normal given by fixed_normal_unit_j2000).",
    )
    fixed_normal_unit_j2000: tuple[float, float, float] | None = _field(
        None,
        unit="1",
        provenance="assumed",
        description="Unit collector normal in J2000 for incidence_mode='fixed_normal'.",
        strict=False,
    )

    @model_validator(mode="after")
    def _check_collector(self) -> Self:
        issues: list[str] = []
        if not self.pv_efficiency_min < self.pv_efficiency_max:
            issues.append(
                f"collector.pv_efficiency_min={self.pv_efficiency_min} must be < "
                f"collector.pv_efficiency_max={self.pv_efficiency_max}"
            )
        if not (self.pv_efficiency_min <= self.pv_efficiency_reference <= self.pv_efficiency_max):
            issues.append(
                f"collector.pv_efficiency_reference={self.pv_efficiency_reference} lies outside "
                f"[pv_efficiency_min, pv_efficiency_max] = "
                f"[{self.pv_efficiency_min}, {self.pv_efficiency_max}]"
            )
        if self.incidence_mode == "fixed_normal":
            if self.fixed_normal_unit_j2000 is None:
                issues.append(
                    "collector.fixed_normal_unit_j2000 is required when "
                    "incidence_mode='fixed_normal'"
                )
            else:
                norm = math.sqrt(sum(c * c for c in self.fixed_normal_unit_j2000))
                if abs(norm - 1.0) > 1e-6:
                    issues.append(
                        f"collector.fixed_normal_unit_j2000={self.fixed_normal_unit_j2000} must "
                        f"be a unit vector (|n| = {norm:.12g}; tolerance 1e-6)"
                    )
        elif self.fixed_normal_unit_j2000 is not None:
            issues.append(
                "collector.fixed_normal_unit_j2000 is set but incidence_mode='sun_tracking' "
                "ignores it; remove it or use incidence_mode='fixed_normal'"
            )
        _raise_if(issues)
        return self


class TransmitterConfig(_Section):
    """Power transmitter: DC conditioning, DC-to-beam conversion, aperture and pointing.

    Stage chain (PDF §10.1): ``P_dc_to_tx = P_PV * power_conditioning_efficiency``,
    ``P_beam = P_dc_to_tx * electrical_to_beam_efficiency``,
    ``P_aperture = P_beam * optics_or_rf_efficiency``. The beam/aperture model is normalised to
    the power leaving the aperture, so ``optics_or_rf_efficiency`` must represent a separate
    physical loss and is not an "aperture efficiency".

    ``aperture_illumination`` is the field *across the aperture* (zero outside its edge), and
    ``P_aperture`` is the power in that field. Source power that falls outside the aperture edge
    before it is emitted (e.g. spill-over of a Gaussian laser beam clipped by the aperture,
    fraction ``exp(-2 a^2 / w0^2)``) is a transmitter-optics loss owned by
    ``optics_or_rf_efficiency``; no beam model applies it again.
    """

    mode: TransmitterMode = _field(
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Transmission technology: 'optical' (laser) or 'microwave'.",
    )
    wavelength_m: float = _field(
        unit="m",
        provenance="assumed",
        description="Carrier wavelength (optical: 1e-7..2e-5 m; microwave: 1e-3..1 m).",
        gt=0.0,
    )
    aperture_diameter_m: float = _field(
        unit="m",
        provenance="assumed",
        description="Diameter D of the circular transmitting aperture.",
        gt=0.0,
    )
    power_conditioning_efficiency: float = _field(
        1.0,
        unit="1",
        provenance="assumed",
        description="eta_power_conditioning = P_dc_to_tx / P_PV. Default 1.0 means the stage is "
        "lossless (optimistic); set a value for real studies.",
        gt=0.0,
        le=1.0,
    )
    electrical_to_beam_efficiency: float = _field(
        unit="1",
        provenance="assumed",
        description="eta_electrical_to_beam = P_beam / P_dc_to_tx (laser wall-plug or DC-to-RF).",
        gt=0.0,
        le=1.0,
    )
    optics_or_rf_efficiency: float = _field(
        1.0,
        unit="1",
        provenance="assumed",
        description="eta_transmitter_optics_or_rf = P_aperture / P_beam: separate physical "
        "losses in optics/feed (not diffraction). Default 1.0 = lossless.",
        gt=0.0,
        le=1.0,
    )
    aperture_illumination: ApertureIllumination = _field(
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Aperture field amplitude profile: 'uniform_circular', "
        "'truncated_gaussian' (needs gaussian_waist_radius_m) or 'parabolic_on_pedestal' "
        "(needs edge_taper_dB).",
    )
    gaussian_waist_radius_m: float | None = _field(
        None,
        unit="m",
        provenance="assumed",
        description="1/e^2 intensity radius w0 of the Gaussian aperture field, which is "
        "truncated at the aperture edge a = aperture_diameter_m/2 (truncated_gaussian only); "
        "must not exceed a. models.beam='gaussian_beam' additionally requires "
        "a/w0 >= models.gaussian_beam_min_truncation_ratio.",
        gt=0.0,
    )
    edge_taper_dB: float | None = _field(
        None,
        unit="dB",
        provenance="assumed",
        description="Edge illumination taper (intensity, centre/edge) for "
        "parabolic_on_pedestal only.",
        gt=0.0,
        le=60.0,
    )
    rms_wavefront_error_m: float = _field(
        0.0,
        unit="m",
        provenance="assumed",
        description="RMS wavefront (path-length) error across the aperture, used for the "
        "Ruze/Strehl loss exp(-(2*pi*sigma/lambda)^2). Default 0 = perfect aperture.",
        ge=0.0,
    )
    array_element_spacing_m: float | None = _field(
        None,
        unit="m",
        provenance="assumed",
        description="Microwave only: element spacing of a uniform phased-array lattice filling "
        "the aperture (enables the array factor). None = continuous filled aperture.",
        gt=0.0,
    )
    pointing_sigma_rad: float = _field(
        unit="rad",
        provenance="assumed",
        description="Per-axis 1-sigma random pointing jitter (isotropic 2-D Gaussian; "
        "small-angle model, <= 0.1 rad).",
        ge=0.0,
        le=0.1,
    )
    pointing_bias_rad: float = _field(
        0.0,
        unit="rad",
        provenance="assumed",
        description="Deterministic pointing bias magnitude (small-angle model, <= 0.1 rad).",
        ge=0.0,
        le=0.1,
    )

    @model_validator(mode="after")
    def _check_transmitter(self) -> Self:
        issues: list[str] = []
        lo, hi = (
            OPTICAL_WAVELENGTH_RANGE_M if self.mode == "optical" else MICROWAVE_WAVELENGTH_RANGE_M
        )
        if not lo <= self.wavelength_m <= hi:
            issues.append(
                f"transmitter.wavelength_m={self.wavelength_m} is outside the "
                f"{self.mode} range [{lo}, {hi}] m required by transmitter.mode='{self.mode}'"
            )
        illum = self.aperture_illumination
        if illum == "truncated_gaussian":
            if self.gaussian_waist_radius_m is None:
                issues.append(
                    "transmitter.gaussian_waist_radius_m is required for "
                    "aperture_illumination='truncated_gaussian'"
                )
            elif self.gaussian_waist_radius_m > 0.5 * self.aperture_diameter_m:
                issues.append(
                    f"transmitter.gaussian_waist_radius_m={self.gaussian_waist_radius_m} exceeds "
                    f"the aperture radius aperture_diameter_m/2={0.5 * self.aperture_diameter_m}"
                )
        elif self.gaussian_waist_radius_m is not None:
            issues.append(
                f"transmitter.gaussian_waist_radius_m is only meaningful for "
                f"aperture_illumination='truncated_gaussian' (got '{illum}')"
            )
        if illum == "parabolic_on_pedestal":
            if self.edge_taper_dB is None:
                issues.append(
                    "transmitter.edge_taper_dB is required for "
                    "aperture_illumination='parabolic_on_pedestal'"
                )
        elif self.edge_taper_dB is not None:
            issues.append(
                f"transmitter.edge_taper_dB is only meaningful for "
                f"aperture_illumination='parabolic_on_pedestal' (got '{illum}')"
            )
        if self.array_element_spacing_m is not None:
            if self.mode != "microwave":
                issues.append(
                    "transmitter.array_element_spacing_m (phased-array lattice) requires "
                    f"transmitter.mode='microwave' (got '{self.mode}')"
                )
            elif self.array_element_spacing_m >= self.aperture_diameter_m:
                issues.append(
                    f"transmitter.array_element_spacing_m={self.array_element_spacing_m} must be "
                    f"smaller than aperture_diameter_m={self.aperture_diameter_m}"
                )
        _raise_if(issues)
        return self


class SiteConfig(_Section):
    """One ground receiver site (geodetic coordinates on the WGS84 ellipsoid)."""

    name: str = _field(
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Unique site name.",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        max_length=64,
    )
    latitude_deg: float = _field(
        unit="deg",
        provenance="assumed",
        description="Geodetic latitude (WGS84).",
        ge=-90.0,
        le=90.0,
    )
    longitude_deg: float = _field(
        unit="deg",
        provenance="assumed",
        description="Geodetic longitude, east positive (WGS84).",
        ge=-180.0,
        le=180.0,
    )
    altitude_m: float = _field(
        0.0,
        unit="m",
        provenance="assumed",
        description="Height above the WGS84 ellipsoid.",
        ge=-1000.0,
        le=9000.0,
    )

    @property
    def latitude_rad(self) -> float:
        """Geodetic latitude [rad]."""
        return math.radians(self.latitude_deg)

    @property
    def longitude_rad(self) -> float:
        """Geodetic longitude [rad]."""
        return math.radians(self.longitude_deg)


class ReceiverConfig(_Section):
    """Ground receiver network and receiver/grid conversion (PDF §10.1, §10.4).

    ``P_receiver = P_captured * conversion_efficiency``; ``P_grid = P_receiver * grid_efficiency``.

    Receiver limits (PDF §10.4) act on defined stages of that chain:

    * ``max_input_power_W`` bounds the *captured beam power* ``P_captured`` (optical or RF power
      entering the receiver aperture, before conversion): the receiver accepts
      ``P_accepted = min(P_captured, max_input_power_W)`` and the excess is reported as a
      receiver-saturation loss; then ``P_receiver = P_accepted * conversion_efficiency``.
    * ``max_heat_rejection_W`` bounds the receiver waste heat
      ``Q_rx = P_accepted * (1 - conversion_efficiency)`` (all accepted power not converted to
      electricity is assumed absorbed as heat).
    * ``max_power_density_W_m2`` bounds the peak incident beam power density.
    """

    sites: tuple[SiteConfig, ...] = _field(
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Receiver sites (at least one; unique names).",
        min_length=1,
        strict=False,
    )
    technology: ReceiverTechnology = _field(
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Receiver technology: 'laser_photovoltaic' (optical) or 'rectenna' "
        "(microwave); must match transmitter.mode.",
    )
    radius_m: float = _field(
        unit="m",
        provenance="assumed",
        description="Radius of the circular receiver aperture.",
        gt=0.0,
    )
    min_elevation_deg: float = _field(
        unit="deg",
        provenance="assumed",
        description="Minimum elevation of the satellite above the local horizon for access. "
        "With models.atmosphere='scenario' it must be >= "
        "atmosphere.plane_parallel_min_elevation_deg (validity of the air-mass model).",
        ge=0.0,
        lt=90.0,
    )
    sun_avoidance_deg: float = _field(
        0.0,
        unit="deg",
        provenance="assumed",
        description="Minimum Sun-receiver-satellite angle at the receiver; 0 = no constraint.",
        ge=0.0,
        lt=180.0,
    )
    conversion_efficiency: float = _field(
        unit="1",
        provenance="assumed",
        description="eta_receiver_conversion = P_receiver / P_captured.",
        gt=0.0,
        le=1.0,
    )
    grid_efficiency: float = _field(
        unit="1",
        provenance="assumed",
        description="eta_grid_conditioning = P_grid / P_receiver.",
        gt=0.0,
        le=1.0,
    )
    max_input_power_W: float = _field(
        unit="W",
        provenance="assumed",
        description="Maximum captured beam power P_captured (optical/RF power entering the "
        "receiver, before conversion_efficiency) that the receiver accepts; captured power "
        "above it is clipped and reported as receiver saturation (not electrical output).",
        gt=0.0,
    )
    max_heat_rejection_W: float | None = _field(
        None,
        unit="W",
        provenance="assumed",
        description="Thermal rejection capability of the receiver: maximum waste heat "
        "Q_rx = P_accepted * (1 - conversion_efficiency) it can reject (PDF §10.4). None = not "
        "enforced (Q_rx is still reported).",
        gt=0.0,
    )
    max_power_density_W_m2: float | None = _field(
        None,
        unit="W m^-2",
        provenance="assumed",
        description="Maximum allowed incident beam power density at the receiver (beam-abort "
        "constraint). None = not enforced (peak density is still reported).",
        gt=0.0,
    )

    @field_validator("sites")
    @classmethod
    def _unique_names(cls, sites: tuple[SiteConfig, ...]) -> tuple[SiteConfig, ...]:
        names = [s.name for s in sites]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"receiver.sites names must be unique; duplicated: {dupes}")
        return sites

    @property
    def min_elevation_rad(self) -> float:
        """Minimum elevation [rad]."""
        return math.radians(self.min_elevation_deg)

    @property
    def sun_avoidance_rad(self) -> float:
        """Sun-avoidance angle [rad]."""
        return math.radians(self.sun_avoidance_deg)

    @property
    def area_m2(self) -> float:
        """Receiver aperture area pi * radius_m^2 [m^2]."""
        return math.pi * self.radius_m**2


class ThermalConfig(_Section):
    """Radiator for transmitter/power-electronics waste heat and PV-array thermal node (PDF §8.3).

    ``Q_rad = emissivity * sigma * A_eff * (T_rad^4 - T_space^4)`` with
    ``A_eff = A_physical * radiating_sides * radiator_view_factor``: ``radiating_sides`` is the
    number of faces of a thin flat plate that see space (1: one face insulated; 2: both faces
    exposed); ``radiator_view_factor`` is the view factor from each face to deep space.
    """

    emissivity: float = _field(
        unit="1",
        provenance="assumed",
        description="Hemispherical IR emissivity of the radiator surface.",
        gt=0.0,
        le=1.0,
    )
    radiator_temperature_limit_K: float = _field(
        unit="K",
        provenance="assumed",
        description="Maximum allowed radiator temperature.",
        gt=0.0,
    )
    radiating_sides: Literal[1, 2] = _field(
        unit="1",
        provenance="assumed",
        description="Number of faces of the thin flat radiator plate radiating to space.",
    )
    radiator_view_factor: float = _field(
        unit="1",
        provenance="assumed",
        description="View factor from each radiating face to deep space.",
        gt=0.0,
        le=1.0,
    )
    radiator_area_m2: float | None = _field(
        None,
        unit="m^2",
        provenance="assumed",
        description="Physical radiator plate area. None = size the radiator at the temperature "
        "limit; a value = solve the radiator temperature for that area.",
        gt=0.0,
    )
    space_temperature_K: float = _field(
        2.725,
        unit="K",
        provenance="sourced",
        source="CMB temperature 2.72548 K, Fixsen (2009) ApJ 707, 916",
        description="Radiative sink temperature T_space (deep-space background).",
        gt=0.0,
    )
    pv_front_emissivity: float = _field(
        0.90,
        unit="1",
        provenance="assumed",
        description="IR emissivity of the PV array front (Sun-facing) surface; default equals "
        "the PDF §5 example radiator emissivity, replace with a measured value.",
        gt=0.0,
        le=1.0,
    )
    pv_back_emissivity: float = _field(
        0.90,
        unit="1",
        provenance="assumed",
        description="IR emissivity of the PV array back surface; default as pv_front_emissivity.",
        gt=0.0,
        le=1.0,
    )
    other_heat_load_W: float = _field(
        0.0,
        unit="W",
        provenance="assumed",
        description="Additional waste heat Q_other rejected by the radiator (PDF §8.3).",
        ge=0.0,
    )

    @model_validator(mode="after")
    def _check_thermal(self) -> Self:
        if not self.radiator_temperature_limit_K > self.space_temperature_K:
            raise ValueError(
                f"thermal.radiator_temperature_limit_K={self.radiator_temperature_limit_K} must "
                f"exceed thermal.space_temperature_K={self.space_temperature_K} (a radiator at "
                "or below the sink temperature cannot reject heat)"
            )
        return self


class AtmosphereConfig(_Section):
    """Scenario factors for the short atmospheric terminal segment (PDF §10.2-10.3).

    Used when ``models.atmosphere == "scenario"``. Zenith values are scaled to the link
    elevation with the plane-parallel air mass X = 1/sin(elevation): transmittance
    tau = tau_zenith**X, attenuation A_dB = A_zenith_dB * X. Defaults are lossless/optimistic
    except the cloudy optical case, which is a hard outage (PDF §10.2).

    X = 1/sin(el) diverges at the horizon and overestimates the air mass of a spherical
    atmosphere at low elevation (relative to Kasten & Young 1989, Appl. Opt. 28, 4735:
    +3.1 % at 10 deg, +11 % at 5 deg, +26 % at 3 deg). It is therefore only used above
    ``plane_parallel_min_elevation_deg``, which ``receiver.min_elevation_deg`` must respect.
    """

    optical_zenith_transmittance_clear: float = _field(
        1.0,
        unit="1",
        provenance="assumed",
        description="Optical zenith transmittance in the 'clear' scenario. Default 1.0 = no "
        "clear-sky extinction (optimistic); replace with a sourced value.",
        gt=0.0,
        le=1.0,
    )
    optical_zenith_transmittance_cloudy: float = _field(
        0.0,
        unit="1",
        provenance="assumed",
        description="Optical zenith transmittance in the 'cloudy' scenario. Default 0 = opaque "
        "cloud, i.e. hard outage (PDF §10.2).",
        ge=0.0,
        le=1.0,
    )
    microwave_zenith_attenuation_clear_dB: float = _field(
        0.0,
        unit="dB",
        provenance="assumed",
        description="Microwave zenith attenuation, 'clear' scenario. Default 0 = lossless.",
        ge=0.0,
        le=100.0,
    )
    microwave_zenith_attenuation_cloudy_dB: float = _field(
        0.0,
        unit="dB",
        provenance="assumed",
        description="Microwave zenith attenuation, 'cloudy' scenario. Default 0 = lossless.",
        ge=0.0,
        le=100.0,
    )
    airmass_model: Literal["plane_parallel"] = _field(
        "plane_parallel",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Air-mass model X(elevation) = 1/sin(elevation) (plane-parallel; "
        "valid only above plane_parallel_min_elevation_deg).",
    )
    plane_parallel_min_elevation_deg: float = _field(
        10.0,
        unit="deg",
        provenance="assumed",
        description="Validity floor of the plane-parallel air mass: with "
        "models.atmosphere='scenario', receiver.min_elevation_deg must be >= this. At the "
        "default 10 deg, 1/sin(el) exceeds the Kasten & Young (1989) air mass by 3.1 %.",
        gt=0.0,
        lt=90.0,
    )

    @model_validator(mode="after")
    def _check_scenarios(self) -> Self:
        issues: list[str] = []
        if self.optical_zenith_transmittance_cloudy > self.optical_zenith_transmittance_clear:
            issues.append(
                "atmosphere.optical_zenith_transmittance_cloudy="
                f"{self.optical_zenith_transmittance_cloudy} exceeds the clear-sky value "
                f"{self.optical_zenith_transmittance_clear}"
            )
        if self.microwave_zenith_attenuation_cloudy_dB < self.microwave_zenith_attenuation_clear_dB:
            issues.append(
                "atmosphere.microwave_zenith_attenuation_cloudy_dB="
                f"{self.microwave_zenith_attenuation_cloudy_dB} is below the clear-sky value "
                f"{self.microwave_zenith_attenuation_clear_dB}"
            )
        _raise_if(issues)
        return self


class ModelsConfig(_Section):
    """Model selections and validity thresholds."""

    beam: BeamModel = _field(
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Beam propagation model. Optical: airy_far_field (uniform aperture, "
        "Fraunhofer), gaussian_beam (propagates an untruncated Gaussian of waist w0 normalised "
        "to P_aperture: an approximation of the truncated_gaussian aperture field, accepted "
        "only when a/w0 >= gaussian_beam_min_truncation_ratio), fresnel_hankel (any "
        "illumination, including strongly truncated Gaussians). Microwave: aperture_far_field "
        "(Fraunhofer of the configured illumination), fresnel_hankel.",
    )
    atmosphere: Literal["none", "scenario"] = _field(
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="'none' = no atmospheric loss (upper bound); 'scenario' = use the "
        "atmosphere section factors for models.weather_scenario.",
    )
    weather_scenario: Literal["clear", "cloudy"] = _field(
        "clear",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Deterministic weather scenario applied over the whole window (needs "
        "atmosphere='scenario'; 'cloudy' with atmosphere='none' is rejected).",
    )
    light_time: bool = _field(
        True,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Retarded-time (light-time) targeting (PDF §7.1); false = instantaneous "
        "geometry approximation.",
    )
    sun_occultation: bool = _field(
        True,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Block links whose beam segment passes within the solar limb (+ margin).",
    )
    earth_occultation: bool = _field(
        True,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Block links whose beam segment intersects the Earth ellipsoid.",
    )
    moon_occultation: bool = _field(
        True,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Block links whose beam segment intersects the Moon.",
    )
    sun_occultation_limb_margin_deg: float = _field(
        0.0,
        unit="deg",
        provenance="assumed",
        description="Extra angular margin around the solar limb, seen from the receiver.",
        ge=0.0,
        le=90.0,
    )
    fraunhofer_max_fresnel_number: float = _field(
        0.5,
        unit="1",
        provenance="sourced",
        source="Far-field distance R >= 2 D^2 / lambda (i.e. N_F = D^2/(lambda R) <= 0.5): "
        "IEEE Std 145-2013, Standard for Definitions of Terms for Antennas, 'far-field "
        "region'; C. A. Balanis, Antenna Theory: Analysis and Design, Wiley, ch. 2 'Field "
        "Regions'",
        description="Far-field models raise an error when N_F = D^2/(lambda R) exceeds this.",
        gt=0.0,
        le=1.0,
    )
    gaussian_beam_min_truncation_ratio: float = _field(
        2.5,
        unit="1",
        provenance="assumed",
        description="Validity threshold of models.beam='gaussian_beam': the aperture radius "
        "to waist ratio a/w0 must be >= this. For a Gaussian truncated at a, the far-field "
        "on-axis intensity per unit aperture power is (1-exp(-r^2))^2/(1-exp(-2 r^2)) times "
        "the untruncated value (r = a/w0): 0.9961 at r = 2.5 (0.4 % peak overestimate by "
        "the untruncated model; truncated power fraction exp(-2 r^2) = 3.7e-6), 0.964 at "
        "r = 2, 0.46 at r = 1. Use fresnel_hankel for more strongly truncated apertures.",
        ge=1.0,
        le=10.0,
    )


class ConstantsConfig(_Section):
    """Physical constants that a study may vary (each with a sourced default)."""

    solar_constant_1au_W_m2: float = _field(
        1361.0,
        unit="W m^-2",
        provenance="sourced",
        source="IAU 2015 Resolution B3, nominal total solar irradiance S_sun^N = 1361 W m^-2",
        description="Total solar irradiance at 1 au, S_1AU, in S(r) = S_1AU (au/r)^2.",
        ge=1000.0,
        le=2000.0,
    )


# ---------------------------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------------------------


class RunConfig(_Section):
    """Complete, validated run configuration (single source of every physical parameter)."""

    schema_version: int = _field(
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Configuration schema version (required, no default, so an unversioned "
        "file is never silently read under a newer schema); must equal the version this "
        "package supports.",
    )
    project: ProjectConfig
    time: TimeConfig
    frames: FramesConfig = Field(default_factory=FramesConfig)
    orbit: OrbitConfig
    collector: CollectorConfig
    transmitter: TransmitterConfig
    receiver: ReceiverConfig
    thermal: ThermalConfig
    atmosphere: AtmosphereConfig = Field(default_factory=AtmosphereConfig)
    models: ModelsConfig
    constants: ConstantsConfig = Field(default_factory=ConstantsConfig)
    provenance_overrides: dict[str, Provenance] = _field(
        default_factory=dict,
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Per-run provenance relabelling: dotted leaf path (e.g. "
        "'receiver.radius_m', 'receiver.sites[0].latitude_deg') -> provenance.",
        strict=False,
    )

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value}; this helioswarm understands "
                f"schema_version {SCHEMA_VERSION}"
            )
        return value

    @model_validator(mode="after")
    def _check_cross_section(self) -> Self:
        issues: list[str] = []
        mode = self.transmitter.mode
        beam = self.models.beam
        allowed = OPTICAL_BEAM_MODELS if mode == "optical" else MICROWAVE_BEAM_MODELS
        if beam not in allowed:
            issues.append(
                f"models.beam='{beam}' is not allowed with transmitter.mode='{mode}' "
                f"(allowed: {sorted(allowed)})"
            )
        illum = self.transmitter.aperture_illumination
        if beam == "airy_far_field" and illum != "uniform_circular":
            issues.append(
                "models.beam='airy_far_field' is the pattern of a uniformly illuminated circular "
                f"aperture; it contradicts transmitter.aperture_illumination='{illum}'"
            )
        if beam == "gaussian_beam":
            waist_m = self.transmitter.gaussian_waist_radius_m
            if illum != "truncated_gaussian":
                issues.append(
                    "models.beam='gaussian_beam' requires transmitter.aperture_illumination="
                    f"'truncated_gaussian' with gaussian_waist_radius_m (got '{illum}')"
                )
            elif waist_m is not None:
                ratio = 0.5 * self.transmitter.aperture_diameter_m / waist_m
                limit = self.models.gaussian_beam_min_truncation_ratio
                if ratio < limit:
                    issues.append(
                        "models.beam='gaussian_beam' (untruncated Gaussian approximation) is "
                        f"outside its validity range: aperture radius / waist = {ratio:.6g} "
                        f"(transmitter.aperture_diameter_m/2 / "
                        f"transmitter.gaussian_waist_radius_m) is below "
                        f"models.gaussian_beam_min_truncation_ratio={limit}; use "
                        "models.beam='fresnel_hankel' for a strongly truncated aperture"
                    )
        if self.models.atmosphere == "none" and self.models.weather_scenario != "clear":
            issues.append(
                f"models.weather_scenario='{self.models.weather_scenario}' has no effect with "
                "models.atmosphere='none' (no atmospheric loss); use models.atmosphere="
                "'scenario' or weather_scenario='clear'"
            )
        if self.models.atmosphere == "scenario" and (
            self.receiver.min_elevation_deg < self.atmosphere.plane_parallel_min_elevation_deg
        ):
            issues.append(
                f"receiver.min_elevation_deg={self.receiver.min_elevation_deg} is below "
                "atmosphere.plane_parallel_min_elevation_deg="
                f"{self.atmosphere.plane_parallel_min_elevation_deg}, the validity floor of "
                f"the '{self.atmosphere.airmass_model}' air mass X = 1/sin(el) used by "
                "models.atmosphere='scenario' (X diverges at the horizon)"
            )
        required_tech = RECEIVER_TECHNOLOGY_FOR_MODE[mode]
        if self.receiver.technology != required_tech:
            issues.append(
                f"receiver.technology='{self.receiver.technology}' does not match "
                f"transmitter.mode='{mode}' (requires '{required_tech}')"
            )
        if (
            self.orbit.model == "prescribed_ephemeris_perturbed_test_particle"
            and self.time.integrator_step_s is None
        ):
            issues.append(
                "time.integrator_step_s is required when orbit.model="
                "'prescribed_ephemeris_perturbed_test_particle'"
            )
        if self.orbit.model == "two_body_kepler" and self.time.integrator_step_s is not None:
            issues.append(
                "time.integrator_step_s is set but orbit.model='two_body_kepler' is analytic "
                "and ignores it; remove it or use 'prescribed_ephemeris_perturbed_test_particle'"
            )
        if self.provenance_overrides:
            known = set(leaf_paths(self))
            for key in self.provenance_overrides:
                if key not in known:
                    hint = difflib.get_close_matches(key, sorted(known), n=1)
                    suggestion = f"; did you mean '{hint[0]}'?" if hint else ""
                    issues.append(
                        f"provenance_overrides key '{key}' is not a configuration leaf "
                        f"path{suggestion}"
                    )
        _raise_if(issues)
        return self


# ---------------------------------------------------------------------------------------------
# Field walking and provenance
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldMeta:
    """Metadata attached to one schema field (see module docstring)."""

    unit: str
    provenance: Provenance
    kind: FieldKind
    description: str
    source: str | None


@dataclass(frozen=True, slots=True)
class ProvenanceRow:
    """One configuration leaf value together with its metadata and effective provenance."""

    path: str
    value: Any
    unit: str
    provenance: Provenance
    schema_provenance: Provenance
    kind: FieldKind
    description: str
    source: str | None


def field_meta(model_cls: type[BaseModel], name: str) -> FieldMeta:
    """Return the helioswarm metadata of field ``name`` of ``model_cls``."""
    info = model_cls.model_fields[name]
    extra = info.json_schema_extra
    if not isinstance(extra, dict):
        raise TypeError(f"{model_cls.__name__}.{name} has no helioswarm field metadata")
    source = extra.get("source")
    return FieldMeta(
        unit=str(extra["unit"]),
        provenance=cast("Provenance", extra["provenance"]),
        kind=cast("FieldKind", extra["kind"]),
        description=str(extra["description"]),
        source=None if source is None else str(source),
    )


def _section_type(annotation: Any) -> type[_Section] | None:
    if isinstance(annotation, type) and issubclass(annotation, _Section):
        return annotation
    return None


def _tuple_item_section(annotation: Any) -> type[_Section] | None:
    if get_origin(annotation) is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return _section_type(args[0])
    return None


def _iter_leaves(model: _Section, prefix: str) -> Iterator[tuple[str, Any, FieldMeta]]:
    for name, info in type(model).model_fields.items():
        if name == "provenance_overrides":
            continue
        path = f"{prefix}{name}"
        value = getattr(model, name)
        annotation = info.annotation
        if _section_type(annotation) is not None:
            yield from _iter_leaves(value, f"{path}.")
        elif _tuple_item_section(annotation) is not None:
            for i, item in enumerate(value):
                yield from _iter_leaves(item, f"{path}[{i}].")
        else:
            yield path, value, field_meta(type(model), name)


def leaf_paths(config: RunConfig) -> tuple[str, ...]:
    """Dotted paths of every leaf value of ``config`` (list items as ``name[i]``)."""
    return tuple(path for path, _, _ in _iter_leaves(config, ""))


def provenance_table(config: RunConfig) -> tuple[ProvenanceRow, ...]:
    """Every leaf value with unit, description and effective provenance (PDF §13.4).

    The effective provenance is the schema default unless relabelled in
    ``config.provenance_overrides``.
    """
    rows = []
    for path, value, meta in _iter_leaves(config, ""):
        effective = config.provenance_overrides.get(path, meta.provenance)
        rows.append(
            ProvenanceRow(
                path=path,
                value=value,
                unit=meta.unit,
                provenance=effective,
                schema_provenance=meta.provenance,
                kind=meta.kind,
                description=meta.description,
                source=meta.source,
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """One configuration problem: dotted field path, message and offending value (if any)."""

    path: str
    message: str
    value: Any = None
    has_value: bool = False

    def format(self) -> str:
        """Human-readable one-line form ``path: message (got: value)``."""
        got = f" (got: {self.value!r})" if self.has_value else ""
        return f"{self.path}: {self.message}{got}"


class ConfigError(Exception):
    """Raised when a configuration file cannot be read or fails validation."""

    def __init__(self, source: str, issues: Sequence[ConfigIssue]) -> None:
        self.source = source
        self.issues: tuple[ConfigIssue, ...] = tuple(issues)
        super().__init__(self.format())

    def format(self) -> str:
        """Multi-line report listing every issue."""
        n = len(self.issues)
        lines = [f"invalid configuration {self.source}: {n} problem{'s' if n != 1 else ''}"]
        lines.extend(f"  - {issue.format()}" for issue in self.issues)
        return "\n".join(lines)


def _loc_to_path(loc: Sequence[int | str]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else part
    return out or "<root>"


def _model_at(loc: Sequence[int | str]) -> type[_Section] | None:
    """Section class addressed by ``loc`` inside RunConfig (for unknown-key suggestions)."""
    current: type[_Section] = RunConfig
    for part in loc:
        if isinstance(part, int):
            continue
        info = current.model_fields.get(part)
        if info is None:
            return None
        nxt = _section_type(info.annotation) or _tuple_item_section(info.annotation)
        if nxt is None:
            return None
        current = nxt
    return current


_MESSAGE_PREFIXES: Final[tuple[str, ...]] = ("Value error, ", "Assertion failed, ")


def issues_from_validation_error(exc: ValidationError) -> tuple[ConfigIssue, ...]:
    """Translate a pydantic ``ValidationError`` into readable :class:`ConfigIssue` records."""
    issues: list[ConfigIssue] = []
    for err in exc.errors(include_url=False):
        loc = err["loc"]
        path = _loc_to_path(loc)
        msg = str(err["msg"])
        for prefix in _MESSAGE_PREFIXES:
            if msg.startswith(prefix):
                msg = msg[len(prefix) :]
        etype = err["type"]
        value = err.get("input")
        show_value = not isinstance(value, Mapping | BaseModel)
        if etype == "extra_forbidden":
            parent = _model_at(loc[:-1])
            key = str(loc[-1])
            msg = "unknown key (not in the configuration schema)"
            if parent is not None:
                hint = difflib.get_close_matches(key, list(parent.model_fields), n=1)
                if hint:
                    msg += f"; did you mean '{hint[0]}'?"
        elif etype == "missing":
            msg = "missing required field"
            if path == "schema_version":
                msg += f" (this helioswarm reads schema_version: {SCHEMA_VERSION})"
            show_value = False
        for part in msg.split(_ISSUE_SEPARATOR):
            issues.append(ConfigIssue(path=path, message=part, value=value, has_value=show_value))
    return tuple(issues)


# ---------------------------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------------------------


class _ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader for configs.

    Differences from ``yaml.SafeLoader``: (1) timestamps stay strings (epochs are validated as
    UTC strings), (2) floats without a decimal point or exponent sign (``1e-6``, ``1.0e12``)
    resolve to float as in YAML 1.2 (PyYAML's YAML 1.1 resolver reads them as strings),
    (3) duplicate mapping keys are an error instead of silently keeping the last value.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if isinstance(key_node, yaml.ScalarNode):
                key = key_node.value
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
                seen.add(key)
        return cast("dict[Any, Any]", super().construct_mapping(node, deep=deep))


_ConfigLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def read_yaml(path: str | Path) -> Any:
    """Parse a YAML file with the config loader; raise :class:`ConfigError` on I/O/syntax errors."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(str(p), [ConfigIssue("<file>", f"cannot read file: {exc}")]) from exc
    try:
        return yaml.load(text, Loader=_ConfigLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(str(p), [ConfigIssue("<yaml>", f"YAML syntax error: {exc}")]) from exc


def parse_config(data: Any, source: str = "<mapping>") -> RunConfig:
    """Validate a mapping (e.g. parsed YAML) into a :class:`RunConfig`.

    Raises :class:`ConfigError` listing every problem (field path, message, offending value).
    """
    if not isinstance(data, Mapping):
        raise ConfigError(
            source,
            [
                ConfigIssue(
                    "<root>",
                    f"top level must be a mapping of sections, got {type(data).__name__}",
                )
            ],
        )
    try:
        return RunConfig.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigError(source, issues_from_validation_error(exc)) from exc


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a YAML run configuration file.

    Raises :class:`ConfigError` with aggregated, human-readable issues on any failure.
    """
    return parse_config(read_yaml(path), source=str(path))


def resolved_dict(config: RunConfig) -> dict[str, Any]:
    """JSON-compatible dict of the fully resolved configuration, including all defaults."""
    return config.model_dump(mode="json")


def canonical_json(config: RunConfig) -> str:
    """Canonical JSON of the resolved config: sorted keys, no whitespace, ASCII, shortest
    round-trip float repr. This is the input of :func:`config_hash`."""
    return json.dumps(
        resolved_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(config: RunConfig) -> str:
    """SHA-256 hex digest of :func:`canonical_json` (identifies the physical run definition).

    Equivalent spellings of the same config (key order, omitted defaults, ``1e3`` vs ``1000.0``,
    ``Z``-suffixed epochs) hash identically.
    """
    return hashlib.sha256(canonical_json(config).encode("ascii")).hexdigest()


def dump_resolved_yaml(config: RunConfig) -> str:
    """Resolved configuration as YAML (all defaults explicit), headed by schema version and
    config hash comments. Loading the output with :func:`load_config` reproduces ``config``."""
    header = (
        "# helioswarm resolved configuration (all defaults explicit)\n"
        f"# schema_version: {config.schema_version}\n"
        f"# config_hash_sha256: {config_hash(config)}\n"
    )
    body = yaml.safe_dump(resolved_dict(config), sort_keys=False, allow_unicode=False, width=100)
    return header + body


def write_resolved_yaml(config: RunConfig, path: str | Path) -> Path:
    """Write :func:`dump_resolved_yaml` to ``path`` (parent directories are created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_resolved_yaml(config), encoding="utf-8")
    return p


__all__ = [
    "MICROWAVE_BEAM_MODELS",
    "MICROWAVE_WAVELENGTH_RANGE_M",
    "OPTICAL_BEAM_MODELS",
    "OPTICAL_WAVELENGTH_RANGE_M",
    "SCHEMA_VERSION",
    "AtmosphereConfig",
    "CollectorConfig",
    "ConfigError",
    "ConfigIssue",
    "ConstantsConfig",
    "FieldMeta",
    "FramesConfig",
    "ModelsConfig",
    "OrbitConfig",
    "ProjectConfig",
    "ProvenanceRow",
    "ReceiverConfig",
    "RunConfig",
    "SiteConfig",
    "ThermalConfig",
    "TimeConfig",
    "TransmitterConfig",
    "canonical_json",
    "config_hash",
    "dump_resolved_yaml",
    "field_meta",
    "leaf_paths",
    "load_config",
    "parse_config",
    "provenance_table",
    "read_yaml",
    "resolved_dict",
    "write_resolved_yaml",
]
