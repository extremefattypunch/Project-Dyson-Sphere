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

import copy
import difflib
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Self, cast, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_core import PydanticUndefined

from helioswarm.constants import (
    AU_M,
    CMB_TEMPERATURE_K,
    SECONDS_PER_DAY,
    SOLAR_IRRADIANCE_NOMINAL_1AU_W_M2,
    SOLAR_RADIUS_NOMINAL_M,
    SPEED_OF_LIGHT_M_S,
    WGS84_SEMIMAJOR_AXIS_M,
)

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

ITU_R_FREQUENCY_RANGE_HZ: Final[tuple[float, float]] = (1.0e9, 55.0e9)
"""Carrier frequencies accepted by ``models.atmosphere = "itu_r"``: the ITU-R P.618 Earth-space
rain-attenuation method applies up to 55 GHz, and the P.676 Annex 2 gaseous approximation from
1 GHz (domains as implemented and documented by the ``itur`` package, version 0.4.0)."""

ReceiverOrientation = Literal["horizontal", "tracking"]
PhaseReference = Literal["absolute", "earth_relative"]
"""Meaning of an orbit's phase angle (``orbit.true_anomaly_deg``, ring phases): ``absolute`` = the
true anomaly of the osculating elements; ``earth_relative`` = the offset of the satellite's
heliocentric ecliptic (ECLIPJ2000) longitude from the Earth's at the element epoch, positive
ahead of the Earth in its orbital motion (resolved with the ephemeris at run time,
:func:`helioswarm.propagate.resolve_orbit_phase`)."""

EARTH_RELATIVE_MIN_ABS_COS_INCLINATION: Final[float] = 1.0e-12
"""``earth_relative`` phases need ``|cos i|`` above this: for a polar orbit (i = 90 deg) the
ecliptic longitude of the satellite takes only the two values Omega and Omega + 180 deg, so a
longitude offset does not determine the true anomaly."""
AtmosphereModel = Literal["none", "scenario", "itu_r"]
WeatherSource = Literal["scenario", "era5"]

PerturbingBody = Literal[
    "MERCURY BARYCENTER",
    "VENUS BARYCENTER",
    "EARTH",
    "MOON",
    "MARS BARYCENTER",
    "JUPITER BARYCENTER",
    "SATURN BARYCENTER",
    "URANUS BARYCENTER",
    "NEPTUNE BARYCENTER",
]
"""Bodies (SPICE names) that may perturb the satellite in the L1 model besides the Sun."""

DEFAULT_PERTURBING_BODIES: Final[tuple[PerturbingBody, ...]] = get_args(PerturbingBody)
"""Default L1 perturber set: every planetary-system barycentre in DE440 (de440s), with the
Earth-Moon system split into EARTH (399) and MOON (301)."""

INTEGRATOR_RTOL_MIN: Final[float] = 1.0e-13
"""Lowest accepted L1 integrator rtol: above scipy's floor 100*eps (2.2e-14), below which
``solve_ivp`` silently raises rtol (shared by the schema and ``propagate.IntegratorSettings``)."""

INTEGRATOR_RTOL_MAX: Final[float] = 1.0e-6
"""Highest accepted L1 integrator rtol (looser tolerances give km-level errors per year)."""

EPHEMERIS_CACHE_STEP_MAX_S: Final[float] = 7200.0
"""Largest accepted node spacing of the L1 Hermite ephemeris cache [s]. Measured midpoint
interpolation errors over 2030 (de440s): < 1 m for every body at 7200 s (Mercury barycentre
largest); at 21600 s Mercury reaches ~77 m and at 86400 s ~2e4 m (Moon ~5e3 m), so larger
spacings are rejected. Each run also reports the measured cache error in its diagnostics."""

RECEIVER_TECHNOLOGY_FOR_MODE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "optical": "laser_photovoltaic",
        "microwave": "rectenna",
    }
)
"""Receiver technology required by each transmitter mode (PDF §10.4: optical and microwave
receivers are different technology cases)."""


# ---------------------------------------------------------------------------------------------
# Field metadata helpers
# ---------------------------------------------------------------------------------------------

_OMIT_WHEN_DEFAULT_KEY: Final[str] = "omit_when_default"
"""``json_schema_extra`` key of fields that :func:`resolved_dict` omits at their default."""


def _field(
    default: Any = PydanticUndefined,
    *,
    unit: str,
    provenance: Provenance,
    description: str,
    kind: FieldKind = "physical",
    source: str | None = None,
    omit_when_default: bool = False,
    **constraints: Any,
) -> Any:
    """Build a pydantic ``Field`` carrying the helioswarm metadata in ``json_schema_extra``.

    Omitting ``default`` makes the field required. ``omit_when_default=True`` marks a field
    added after the M7 hash re-pin whose default leaves every earlier configuration's meaning
    unchanged: :func:`resolved_dict` omits it while it holds its default, so the canonical JSON
    and :func:`config_hash` of configurations that do not use it are unchanged (M7b).
    """
    extra: dict[str, str] = {
        "provenance": provenance,
        "description": description,
        "unit": unit,
        "kind": kind,
    }
    if source is not None:
        extra["source"] = source
    if omit_when_default:
        extra[_OMIT_WHEN_DEFAULT_KEY] = "true"
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


def canonical_utc(value: str) -> str:
    """Public form of the configuration's UTC-epoch validator.

    Returns the canonical ``YYYY-MM-DDTHH:MM:SS[.ffffff]`` spelling of an ISO-8601 UTC epoch
    (no time-zone offset, no leap-second ``:60``) or raises ``ValueError``. Used by
    :mod:`helioswarm.timescale` so that configuration epochs and API epochs obey one rule.
    """
    return _canonical_utc(value)


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
        description="Informational only: not read by validate-config or run. `helioswarm "
        "run` writes to --out, else $HELIOSWARM_RUNS_DIR/<project.name>, else "
        "runs/<project.name>.",
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
        "rejected otherwise (the analytic two-body model has no integrator).",
        gt=0.0,
    )
    integrator_rtol: float = _field(
        1.0e-12,
        unit="1",
        provenance="assumed",
        kind="numerical",
        description="Relative tolerance of the L1 DOP853 integrator (solve_ivp rtol). The "
        "lower bound 1e-13 stays above scipy's floor 100*eps (2.2e-14), below which scipy "
        "silently raises rtol. L1 only (rejected when set with orbit.model='two_body_kepler').",
        ge=INTEGRATOR_RTOL_MIN,
        le=INTEGRATOR_RTOL_MAX,
    )
    integrator_atol_position_m: float = _field(
        1.0e-3,
        unit="m",
        provenance="assumed",
        kind="numerical",
        description="Absolute tolerance of the L1 integrator on each position component. At "
        "1 au it is far below rtol*|r| (0.15 m for rtol 1e-12), so the relative tolerance "
        "governs; atol only matters for a component passing through zero. L1 only.",
        gt=0.0,
        le=1.0e3,
    )
    integrator_atol_velocity_m_s: float = _field(
        1.0e-9,
        unit="m s^-1",
        provenance="assumed",
        kind="numerical",
        description="Absolute tolerance of the L1 integrator on each velocity component. At "
        "1 au it is below rtol*|v| (3e-8 m/s for rtol 1e-12), so the relative tolerance "
        "governs. L1 only.",
        gt=0.0,
        le=1.0e-3,
    )
    ephemeris_cache_step_s: float = _field(
        3600.0,
        unit="s",
        provenance="assumed",
        kind="numerical",
        description="Sample spacing of the cubic-Hermite cache of prescribed body positions "
        "used by the L1 integrator (positions and velocities sampled from the ephemeris). "
        "Validated range: up to 7200 s, where the interpolation error stays below 1 m for "
        "every DE440 body used (tested against direct SPICE calls; the error grows as h^4, "
        "e.g. ~2e4 m for the Mercury barycentre at 86400 s). Every L1 run reports the "
        "measured cache error (L1Diagnostics.cache_midpoint_error_m). L1 only.",
        gt=0.0,
        le=EPHEMERIS_CACHE_STEP_MAX_S,
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
    ephemeris_kernel_set: Literal["helioswarm-de440s-v1"] = _field(
        "helioswarm-de440s-v1",
        unit="-",
        provenance="sourced",
        kind="model_choice",
        source="data/kernels/kernels.json kernel_set_id (NAIF generic kernels naif0012.tls, "
        "pck00011.tpc, gm_de440.tpc, de440s.bsp; SHA-256 pinned)",
        description="SPICE kernel set (ephemeris, GM values, leap seconds) the run must use. "
        "SPICE-backed run entry points refuse a loaded kernel set with another kernel_set_id, "
        "so the configuration pins the ephemeris data of the run.",
    )


class OrbitConfig(_Section):
    """Initial osculating orbit of the (single) satellite and force-model flags.

    Heliocentric (``elements_center = SUN``, the default): elements are Sun-centred and
    referred to the ECLIPJ2000 frame at ``initial_epoch_utc``; the L0 model
    (``two_body_kepler``, M1) propagates these Sun-centred ECLIPJ2000 elements directly and the
    resulting trajectory is rotated to J2000 afterwards (``frames.py``); the origin shift
    Sun -> SSB needs the Sun's ephemeris state (M2). The L1 model (M2) converts the initial
    state to J2000/SSB before integration.

    Geocentric (``elements_center = EARTH``, M7b GEO baseline): elements are Earth-centred
    (NAIF 399) and referred to the J2000 *equatorial* frame (inclination relative to the mean
    equator of J2000), ``mu = GM_earth`` from ``gm_de440.tpc``; only the L0 two-body model is
    available (Earth point mass; lunisolar, J2 and SRP perturbations are omitted and assumed
    removed by station-keeping, docs/model_assumptions.md "M7b"). The geostationary preset
    ``geostationary_longitude_deg`` replaces the elements by the circular orbit in the
    IAU_EARTH equatorial plane of the element epoch that co-rotates with the IAU_EARTH frame
    above the given longitude (:func:`helioswarm.geocentric.geostationary_orbit`).
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
    elements_frame: Literal["ECLIPJ2000", "J2000"] = _field(
        "ECLIPJ2000",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Reference plane of the orbital elements (SPICE frame name): ECLIPJ2000 "
        "for Sun-centred elements, J2000 (mean equator of J2000) for Earth-centred elements.",
    )
    elements_center: Literal["SUN", "EARTH"] = _field(
        "SUN",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Central body of the orbital elements (SPICE name): SUN (heliocentric "
        "swarm) or EARTH (geocentric orbit, M7b GEO baseline; L0 two-body only).",
    )
    semimajor_axis_au: float | None = _field(
        None,
        unit="au",
        provenance="assumed",
        description="Semimajor axis (1 au = 149597870700 m, IAU 2012 B2). Required, except "
        "with orbit.geostationary_longitude_deg (then derived from GM_earth and the IAU_EARTH "
        "rotation rate, and must not be set).",
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
    true_anomaly_deg: float | None = _field(
        None,
        unit="deg",
        provenance="assumed",
        description="Phase angle at initial_epoch_utc: the true anomaly when "
        "phase_reference='absolute'; with phase_reference='earth_relative' the offset of the "
        "satellite's heliocentric ecliptic (ECLIPJ2000) longitude from the Earth's at that epoch, "
        "positive ahead of the Earth (resolved to a true anomaly with the ephemeris at run time). "
        "Required, except with orbit.geostationary_longitude_deg (must not be set then).",
        ge=0.0,
        lt=360.0,
    )
    geostationary_longitude_deg: float | None = _field(
        None,
        unit="deg",
        provenance="assumed",
        kind="physical",
        omit_when_default=True,
        description="Geostationary preset (M7b; needs elements_center='EARTH', "
        "elements_frame='J2000', model='two_body_kepler'): east longitude in the IAU_EARTH "
        "frame of the sub-satellite point at initial_epoch_utc. The orbit is circular, in the "
        "IAU_EARTH equatorial plane at that epoch, with radius r = (GM_earth / omega^2)^(1/3) "
        "and inertial velocity omega x r, omega = the IAU_EARTH angular velocity (pck00011) at "
        "the epoch, so it co-rotates with the IAU_EARTH frame. None = no preset (use the "
        "elements). The elements semimajor_axis_au and true_anomaly_deg must then be unset "
        "and eccentricity, inclination_deg, raan_deg, arg_periapsis_deg left at 0.",
        ge=-180.0,
        le=180.0,
    )
    phase_reference: PhaseReference = _field(
        "absolute",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Reference of true_anomaly_deg: 'absolute' (true anomaly of the elements) or "
        "'earth_relative' (heliocentric ecliptic longitude offset from the Earth (NAIF 399, "
        "geometric Sun-centred position) at initial_epoch_utc, positive ahead of the Earth).",
    )
    include_planetary_perturbations: bool = _field(
        False,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Include planetary third-body accelerations (L1 model only). false = the "
        "Sun is the only gravitating body: heliocentric two-body motion (+ SRP), the Sun's "
        "ephemeris used only to place the orbit relative to the SSB (the L1 equations are "
        "integrated heliocentrically with consistent direct and indirect terms).",
    )
    perturbing_bodies: tuple[PerturbingBody, ...] = _field(
        DEFAULT_PERTURBING_BODIES,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Bodies (SPICE names) whose point-mass gravity acts on the satellite in "
        "addition to the Sun when include_planetary_perturbations=true. Default: all planetary "
        "system barycentres of DE440 with the Earth-Moon system split into EARTH and MOON. "
        "Any subset is physically consistent (heliocentric integration with the direct and "
        "indirect terms of exactly these bodies). Rejected when set explicitly with "
        "include_planetary_perturbations=false.",
        min_length=1,
        strict=False,
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
        geo = self.geostationary_longitude_deg is not None
        if geo:
            issues.extend(self._geostationary_issues())
        else:
            for name in ("semimajor_axis_au", "true_anomaly_deg"):
                if getattr(self, name) is None:
                    issues.append(
                        f"orbit.{name} is required (it may be omitted only with the "
                        "geostationary preset orbit.geostationary_longitude_deg)"
                    )
        center_frame = {"SUN": "ECLIPJ2000", "EARTH": "J2000"}[self.elements_center]
        if self.elements_frame != center_frame:
            issues.append(
                f"orbit.elements_center='{self.elements_center}' requires orbit.elements_frame="
                f"'{center_frame}' (got '{self.elements_frame}'): heliocentric elements are "
                "ecliptic, geocentric elements are referred to the J2000 equator"
            )
        if self.elements_center == "EARTH":
            if self.model != "two_body_kepler":
                issues.append(
                    "orbit.elements_center='EARTH' supports only orbit.model='two_body_kepler' "
                    "(L0 Earth point mass + the station-keeping assumption, M7b); the L1 "
                    "propagator integrates heliocentric test particles"
                )
            if self.phase_reference != "absolute":
                issues.append(
                    "orbit.phase_reference='earth_relative' is a heliocentric phase; it is "
                    "undefined for orbit.elements_center='EARTH'"
                )
            if self.semimajor_axis_au is not None:
                perigee_m = self.semimajor_axis_au * AU_M * (1.0 - self.eccentricity)
                if perigee_m <= WGS84_SEMIMAJOR_AXIS_M:
                    issues.append(
                        f"orbit perigee a(1-e) = {perigee_m:.6g} m (orbit.semimajor_axis_au="
                        f"{self.semimajor_axis_au}, orbit.eccentricity={self.eccentricity}) is "
                        f"inside the Earth (WGS 84 equatorial radius {WGS84_SEMIMAJOR_AXIS_M} m)"
                    )
        elif self.semimajor_axis_au is not None:
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
        if (
            self.phase_reference == "earth_relative"
            and abs(math.cos(self.inclination_rad)) < EARTH_RELATIVE_MIN_ABS_COS_INCLINATION
        ):
            issues.append(
                f"orbit.phase_reference='earth_relative' is undefined for a polar orbit "
                f"(orbit.inclination_deg={self.inclination_deg}): the ecliptic longitude of a "
                "polar orbit takes only the values raan and raan + 180 deg"
            )
        # An explicit list equal to the default is the default (a resolved dump writes every
        # field explicitly and must load again); any other explicit list would be ignored.
        bodies_explicit = "perturbing_bodies" in self.model_fields_set and tuple(
            self.perturbing_bodies
        ) != tuple(DEFAULT_PERTURBING_BODIES)
        if not self.include_planetary_perturbations and bodies_explicit:
            issues.append(
                "orbit.perturbing_bodies is set but include_planetary_perturbations=false "
                "ignores it; remove it or set include_planetary_perturbations=true"
            )
        dupes = sorted({b for b in self.perturbing_bodies if self.perturbing_bodies.count(b) > 1})
        if dupes:
            issues.append(f"orbit.perturbing_bodies lists bodies more than once: {dupes}")
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

    def _geostationary_issues(self) -> list[str]:
        """Rules of the geostationary preset (the elements are derived, not configured)."""
        issues: list[str] = []
        if self.elements_center != "EARTH":
            issues.append(
                "orbit.geostationary_longitude_deg needs orbit.elements_center='EARTH' and "
                "orbit.elements_frame='J2000'"
            )
        for name in ("semimajor_axis_au", "true_anomaly_deg"):
            if getattr(self, name) is not None:
                issues.append(
                    f"orbit.{name} is set but the geostationary preset "
                    "(orbit.geostationary_longitude_deg) derives it; remove it"
                )
        for name in ("eccentricity", "inclination_deg", "raan_deg", "arg_periapsis_deg"):
            if getattr(self, name) != 0.0:
                issues.append(
                    f"orbit.{name}={getattr(self, name)} contradicts the geostationary preset "
                    "(circular orbit in the IAU_EARTH equator); leave it at 0"
                )
        return issues

    @property
    def is_geocentric(self) -> bool:
        """Earth-centred elements (``elements_center == 'EARTH'``, M7b)."""
        return self.elements_center == "EARTH"

    @property
    def is_geostationary_preset(self) -> bool:
        """The geostationary preset is used (``geostationary_longitude_deg`` set)."""
        return self.geostationary_longitude_deg is not None

    @property
    def geostationary_longitude_rad(self) -> float:
        """Sub-satellite east longitude of the geostationary preset [rad].

        Raises:
            ValueError: no geostationary preset.
        """
        if self.geostationary_longitude_deg is None:
            raise ValueError("orbit.geostationary_longitude_deg is not set")
        return math.radians(self.geostationary_longitude_deg)

    def _required(self, name: str) -> float:
        value = getattr(self, name)
        if value is None:
            raise ValueError(
                f"orbit.{name} is not configured (geostationary preset: the elements are "
                "derived from the ephemeris at run time, helioswarm.geocentric."
                "geostationary_orbit)"
            )
        return float(value)

    @property
    def semimajor_axis_m(self) -> float:
        """Semimajor axis [m] (raises ``ValueError`` for the geostationary preset)."""
        return self._required("semimajor_axis_au") * AU_M

    @property
    def perihelion_m(self) -> float:
        """Periapsis distance a(1 - e) [m] (perihelion for Sun-centred elements)."""
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
        """True anomaly at the initial epoch [rad] (raises ``ValueError`` for the
        geostationary preset)."""
        return math.radians(self._required("true_anomaly_deg"))


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
    max_operating_temperature_K: float = _field(
        400.0,
        unit="K",
        provenance="assumed",
        description="Maximum allowed PV array (cell) operating temperature; the PV thermal node "
        "is infeasible above it. The 400 K default is an assumed scenario value, not a sourced "
        "qualification limit: replace it with the limit of the chosen cell technology.",
        gt=0.0,
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
    array_element_pattern_exponent: float | None = _field(
        None,
        unit="1",
        provenance="assumed",
        description="Exponent q of the lattice element power pattern cos^q(theta); required "
        "with (and only with) array_element_spacing_m. q = 1 is the ideal unit-cell element "
        "(projected-area pattern); q sets how much power goes into visible grating lobes when "
        "the spacing exceeds the wavelength. Must be >= 1 (large-array lobe model).",
        ge=1.0,
        le=20.0,
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
    min_pointing_efficiency: float | None = _field(
        None,
        unit="1",
        provenance="assumed",
        description="Pointing availability condition (PDF §10.4: a link is unavailable if "
        "pointing uncertainty violates the configured conditions): the link is declared "
        "unavailable (reason 'pointing', P_grid = 0) at epochs where the pointing efficiency "
        "E[capture] / capture(perfect pointing) (BeamResult.pointing_loss) is below this value. "
        "None = not enforced (the efficiency is still applied to the captured power and "
        "reported).",
        gt=0.0,
        le=1.0,
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
        elif self.array_element_pattern_exponent is not None:
            issues.append(
                "transmitter.array_element_pattern_exponent is only meaningful with "
                "transmitter.array_element_spacing_m (phased-array lattice)"
            )
        if self.array_element_spacing_m is not None and self.array_element_pattern_exponent is None:
            issues.append(
                "transmitter.array_element_pattern_exponent (element power pattern cos^q) is "
                "required with transmitter.array_element_spacing_m"
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
    max_simultaneous_beams: int | None = _field(
        None,
        unit="-",
        provenance="assumed",
        description="Beam slots of this site: the most beams (satellites) it accepts at one "
        "epoch (PDF §10.4/§11.2 receiver capacity; enforced by the constellation scheduler). "
        "None = receiver.max_simultaneous_beams.",
        ge=1,
    )
    max_input_power_W: float | None = _field(
        None,
        unit="W",
        provenance="assumed",
        description="Override of receiver.max_input_power_W for this site: bound on the SUM of "
        "the captured beam powers of all beams incident on it. None = receiver value.",
        gt=0.0,
    )
    max_heat_rejection_W: float | None = _field(
        None,
        unit="W",
        provenance="assumed",
        description="Override of receiver.max_heat_rejection_W for this site (bound on the "
        "receiver waste heat of the summed accepted power). None = receiver value.",
        gt=0.0,
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

    ``P_receiver = P_accepted * conversion_efficiency``; ``P_grid = P_receiver * grid_efficiency``,
    with ``P_accepted`` the captured power after the receiver limits below (``P_accepted =
    P_captured`` when no limit binds).

    Receiver limits (PDF §10.4) act on defined stages of that chain:

    * ``max_input_power_W`` bounds the *captured beam power* ``P_captured`` (optical or RF power
      entering the receiver aperture, before conversion): the receiver accepts
      ``P_accepted = min(P_captured, max_input_power_W)`` and the excess is reported as a
      receiver-saturation loss; then ``P_receiver = P_accepted * conversion_efficiency``.
    * ``max_heat_rejection_W`` bounds the receiver waste heat
      ``Q_rx = P_accepted * (1 - conversion_efficiency)`` (all accepted power not converted to
      electricity is assumed absorbed as heat), i.e. ``P_accepted <= Q_max / (1 -
      conversion_efficiency)`` (M6: enforced as a clip, flagged as receiver saturation).
    * ``max_power_density_W_m2`` bounds the peak incident beam power density.

    Several transmitters (constellation runs, PDF §10.4 "enforce receiver power limits and
    prevent double-counting the same receiver capacity"): every limit applies to the *sum* over
    the beams incident on one site (independent, incoherent beams: intensities and powers add,
    fields are never added). ``max_simultaneous_beams`` bounds the number of beams per site;
    each site may override the beam, input-power and heat limits (``SiteConfig``).
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
        description="eta_receiver_conversion = P_receiver / P_accepted, with P_accepted = "
        "P_captured after the receiver input/heat limits (= P_captured when no limit binds).",
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
        "constraint), on the plane normal to the beam. None = not enforced (peak density is "
        "still reported).",
        gt=0.0,
    )
    max_simultaneous_beams: int | None = _field(
        None,
        unit="-",
        provenance="assumed",
        description="Beam slots per site: the most beams (satellites) one receiver site accepts "
        "at one epoch (constellation scheduler). None = unlimited. A single-satellite run "
        "never has more than one beam.",
        ge=1,
    )
    orientation: ReceiverOrientation = _field(
        "horizontal",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Receiver aperture orientation: 'horizontal' (flat field, normal = local "
        "up; projected area pi r^2 sin(elevation), modelled as the circle of equal projected "
        "area) or 'tracking' (normal kept on the beam; projected area pi r^2).",
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

    def site_max_input_power_W(self, site: SiteConfig) -> float:
        """Input-power limit of ``site`` (its override, else ``max_input_power_W``) [W]."""
        return self.max_input_power_W if site.max_input_power_W is None else site.max_input_power_W

    def site_max_heat_rejection_W(self, site: SiteConfig) -> float | None:
        """Heat-rejection limit of ``site`` (its override, else the receiver value) [W]."""
        if site.max_heat_rejection_W is not None:
            return site.max_heat_rejection_W
        return self.max_heat_rejection_W

    def site_max_beams(self, site: SiteConfig) -> int | None:
        """Beam slots of ``site`` (its override, else ``max_simultaneous_beams``; None =
        unlimited)."""
        if site.max_simultaneous_beams is not None:
            return site.max_simultaneous_beams
        return self.max_simultaneous_beams


class ThermalConfig(_Section):
    """Radiator for transmitter/power-electronics waste heat and PV-array thermal node (PDF §8.3).

    ``Q_rad = emissivity * sigma * A_eff * (T_rad^4 - T_space^4)`` with
    ``A_eff = A_physical * radiating_sides * radiator_view_factor``: ``radiating_sides`` is the
    number of faces of a thin flat plate that see space (1: one face insulated; 2: both faces
    exposed); ``radiator_view_factor`` is the view factor from each face to deep space.

    PV array node (M4, ``helioswarm.thermal``): a thin isothermal flat plate whose front face
    faces the Sun, radiating from both faces with ``pv_front_emissivity`` and
    ``pv_back_emissivity`` and view factor ``pv_view_factor`` to deep space:
    ``P_abs - P_PV(T) = (eps_f + eps_b) * sigma * A_collector * F_pv * (T^4 - T_space^4)``.
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
        CMB_TEMPERATURE_K,
        unit="K",
        provenance="sourced",
        source="CMB temperature 2.72548 K, Fixsen (2009) ApJ 707, 916 "
        "(helioswarm.constants.CMB_TEMPERATURE_K)",
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
    pv_view_factor: float = _field(
        1.0,
        unit="1",
        provenance="assumed",
        description="View factor from each face (front and back) of the flat PV array plate "
        "to deep space; the remainder 1 - F of each face's hemisphere is assumed to see "
        "surfaces at the plate temperature (no net exchange). 1.0 = isolated plate.",
        gt=0.0,
        le=1.0,
    )
    radiator_specific_mass_kg_m2: float | None = _field(
        None,
        unit="kg m^-2",
        provenance="assumed",
        description="Radiator areal mass for the radiator mass proxy m = A_rad * this value "
        "(physical plate area). None = no mass proxy reported.",
        gt=0.0,
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
    itu_r_exceedance_percent: float = _field(
        1.0,
        unit="%",
        provenance="assumed",
        description="models.atmosphere='itu_r' only: percentage p of an average year for which "
        "the ITU-R P.618 slant-path attenuation (gas + cloud + rain) is exceeded; the "
        "deterministic attenuation used at every epoch. Range of the P.618 rain method: "
        "0.001-5 %.",
        ge=0.001,
        le=5.0,
    )
    itu_r_polarization_tilt_deg: float = _field(
        45.0,
        unit="deg",
        provenance="assumed",
        description="models.atmosphere='itu_r' only: polarisation tilt angle tau relative to "
        "the horizontal used by the ITU-R P.838/P.618 rain attenuation (45 = circular "
        "polarisation, 0 = horizontal linear, 90 = vertical linear). Default 45: circular "
        "polarisation assumed for the power beam (assumption, not a design value).",
        ge=0.0,
        le=90.0,
    )
    itu_r_min_elevation_deg: float = _field(
        5.0,
        unit="deg",
        provenance="assumed",
        description="models.atmosphere='itu_r' only: receiver.min_elevation_deg must be >= this "
        "(the P.676 Annex 2 slant-path approximation is recommended for 5-90 deg).",
        ge=5.0,
        lt=90.0,
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
    atmosphere: AtmosphereModel = _field(
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="'none' = no atmospheric loss (upper bound); 'scenario' = use the "
        "atmosphere section factors for models.weather_scenario; 'itu_r' (microwave only, "
        "needs the optional 'itur' extra) = ITU-R P.618/P.676/P.840 slant-path attenuation "
        "exceeded atmosphere.itu_r_exceedance_percent of the time at each receiver site.",
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
    light_time_tolerance_s: float = _field(
        1.0e-9,
        unit="s",
        provenance="assumed",
        kind="numerical",
        description="Convergence tolerance |tau_new - tau| of the retarded-time (light-time) "
        "iteration (PDF §7.1). 1e-9 s is 0.3 m of light path; the fixed point contracts by "
        "~|v_site|/c ~ 1e-4 per iteration, so it is reached in 3-4 iterations.",
        gt=0.0,
        le=1.0e-3,
    )
    light_time_max_iterations: int = _field(
        50,
        unit="1",
        provenance="assumed",
        kind="numerical",
        description="Maximum fixed-point iterations of the light-time solution; epochs that do "
        "not converge fall back to a bracketed root solve (brentq) and are flagged.",
        ge=2,
        le=1000,
    )
    earth_occultation_standoff_m: float = _field(
        1.0,
        unit="m",
        provenance="assumed",
        kind="numerical",
        description="Length of the beam segment next to the receiver that the Earth-ellipsoid "
        "blockage test skips, so the receiver end point on the surface does not count as an "
        "intersection (sites below the ellipsoid are tested against the similar ellipsoid "
        "through the site). Blind only for |elevation| < standoff/(2a) ~ 8e-8 rad at 1 m.",
        gt=0.0,
        le=1000.0,
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
    beam_evaluation: Literal["per_epoch", "chebyshev_table"] = _field(
        "per_epoch",
        unit="-",
        provenance="assumed",
        kind="numerical",
        omit_when_default=True,
        description="How the beam model is evaluated over the epochs with access (M7b): "
        "'per_epoch' = the reference evaluation at every epoch; 'chebyshev_table' = an "
        "adaptive Chebyshev-Lobatto table over (range, elevation) interpolated to every epoch "
        "and validated against the reference at check epochs in every run (a failed check "
        "stops the run; helioswarm.beam_table). Same beam physics; for near-constant "
        "geometries (GEO) it replaces up to minutes of Fresnel-Hankel quadrature per epoch.",
    )
    beam_table_max_nodes: int = _field(
        33,
        unit="1",
        provenance="assumed",
        kind="numerical",
        omit_when_default=True,
        description="Largest number of Chebyshev-Lobatto nodes per dimension of the beam table "
        "(2^k + 1; refinement 3 -> 5 -> 9 -> 17 -> 33 ...). chebyshev_table only.",
        ge=3,
        le=129,
    )
    beam_table_check_epochs: int = _field(
        8,
        unit="1",
        provenance="assumed",
        kind="numerical",
        omit_when_default=True,
        description="Evenly spaced check epochs per check set (two disjoint sets, plus the "
        "epochs of extreme range and elevation) at which the table is compared with the "
        "reference. chebyshev_table only.",
        ge=1,
        le=1000,
    )
    beam_table_capture_tolerance: float = _field(
        1.0e-9,
        unit="1",
        provenance="assumed",
        kind="numerical",
        omit_when_default=True,
        description="Largest accepted |table - reference| of the expected, nominal and "
        "bias-only capture fractions at the check epochs (absolute). chebyshev_table only.",
        gt=0.0,
        le=1.0e-3,
    )
    beam_table_percentile_tolerance: float = _field(
        1.0e-6,
        unit="1",
        provenance="assumed",
        kind="numerical",
        omit_when_default=True,
        description="Largest accepted |table - reference| of the capture percentiles p05/p50/"
        "p95 under pointing jitter (diagnostics; the reference switches between percentile "
        "methods, which makes them non-smooth at ~1e-7). chebyshev_table only.",
        gt=0.0,
        le=1.0e-2,
    )
    beam_table_relative_tolerance: float = _field(
        1.0e-6,
        unit="1",
        provenance="assumed",
        kind="numerical",
        omit_when_default=True,
        description="Largest accepted relative difference of the peak irradiance and the other "
        "interpolated beam quantities at the check epochs. chebyshev_table only.",
        gt=0.0,
        le=1.0e-2,
    )

    weather_source: WeatherSource = _field(
        "scenario",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Source of the weather availability (M8a): 'scenario' = the deterministic "
        "models.weather_scenario over the whole window (default; pre-M8a behaviour); 'era5' = "
        "hourly ERA5 total cloud cover at each receiver site (weather section; optical only, "
        "needs atmosphere='scenario' with weather_scenario='clear', whose clear-sky factors "
        "apply at the clear hours).",
    )

    @model_validator(mode="after")
    def _check_beam_table(self) -> Self:
        issues: list[str] = []
        n = self.beam_table_max_nodes
        if n > 1 and math.log2(n - 1) != int(math.log2(n - 1)):
            issues.append(
                f"models.beam_table_max_nodes={n} must be 2^k + 1 (3, 5, 9, 17, 33, 65, 129): "
                "the Chebyshev-Lobatto sets are refined by doubling"
            )
        if self.beam_evaluation == "per_epoch":
            for name in BEAM_TABLE_FIELDS:
                default = ModelsConfig.model_fields[name].default
                if getattr(self, name) != default:
                    issues.append(
                        f"models.{name} is set but models.beam_evaluation='per_epoch' ignores "
                        "it; remove it or use 'chebyshev_table'"
                    )
        _raise_if(issues)
        return self


# ---------------------------------------------------------------------------------------------
# M8a: site weather from ERA5 (models.weather_source = "era5")
# ---------------------------------------------------------------------------------------------


class WeatherConfig(_Section):
    """ERA5 site-weather availability (M8a; PDF §10.2 "later, use site-specific cloud ...
    data to estimate operational availability. Cloud obstruction can be a hard outage for a
    ground optical receiver").

    At each receiver site and arrival epoch the hourly ERA5 total cloud cover ``tcc`` (fraction
    of the ~0.25 deg model grid box covered by cloud, NOT a line-of-sight probability) of the
    mapped historical hour decides a hard optical outage: ``threshold``: outage iff
    ``tcc > cloud_cover_threshold``; ``bernoulli``: clear with probability ``1 - tcc`` (one
    seeded draw per site and historical hour, independent between hours). Simulated UTC
    epochs map to the same calendar month/day/hour of a historical year (``year_mapping``).
    """

    era5_dataset: Literal[
        "reanalysis-era5-single-levels-timeseries", "reanalysis-era5-single-levels"
    ] = _field(
        "reanalysis-era5-single-levels-timeseries",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="CDS collection of the cached files: the ERA5 time-series product (DOI "
        "10.24381/1cf1ad76, default) or the gridded single-levels product (DOI "
        "10.24381/cds.adbb2d47); the same ERA5 analyses (helioswarm.era5).",
    )
    era5_first_year: int = _field(
        2014,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="First calendar year of the cached ERA5 record used.",
        ge=1940,
        le=2100,
    )
    era5_last_year: int = _field(
        2023,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Last calendar year of the cached ERA5 record used.",
        ge=1940,
        le=2100,
    )
    interpolation: Literal["nearest", "bilinear"] = _field(
        "nearest",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Site value from the 0.25 deg grid: nearest node, or bilinear between the "
        "four enclosing nodes.",
    )
    neighbourhood_nodes: int = _field(
        1,
        unit="1",
        provenance="assumed",
        kind="numerical",
        description="Half-width k of the downloaded (2k+1)^2 node box (identifies the cached "
        "files; configs/weather/*.yaml neighbourhood_nodes).",
        ge=1,
        le=4,
    )
    outage_model: Literal["threshold", "bernoulli"] = _field(
        "threshold",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="'threshold': hard outage iff tcc > cloud_cover_threshold; 'bernoulli': "
        "clear line of sight with probability 1 - tcc (seeded draw per site-hour).",
    )
    cloud_cover_threshold: float | None = _field(
        None,
        unit="1",
        provenance="assumed",
        description="Outage threshold on the grid-box cloud fraction (threshold model only; "
        "required there, rejected for bernoulli). An assumption: tcc is an area fraction, not "
        "the cloudiness of the receiver's line of sight.",
        ge=0.0,
        le=1.0,
    )
    year_mapping: Literal["fixed", "sampled"] = _field(
        "fixed",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="'fixed': the first simulated UTC calendar year maps to historical_year, "
        "later ones to the following years (cyclic within the record); 'sampled': each "
        "simulated calendar year maps to a year drawn uniformly from the record (seeded).",
    )
    historical_year: int | None = _field(
        None,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Historical year of the first simulated year ('fixed' only; required "
        "there, rejected for 'sampled').",
        ge=1940,
        le=2100,
    )

    @model_validator(mode="after")
    def _check_weather(self) -> Self:
        issues: list[str] = []
        if self.era5_last_year < self.era5_first_year:
            issues.append(
                f"weather.era5_last_year={self.era5_last_year} < era5_first_year="
                f"{self.era5_first_year}"
            )
        if self.outage_model == "threshold" and self.cloud_cover_threshold is None:
            issues.append("weather.outage_model='threshold' needs weather.cloud_cover_threshold")
        if self.outage_model == "bernoulli" and self.cloud_cover_threshold is not None:
            issues.append(
                "weather.cloud_cover_threshold is set but weather.outage_model='bernoulli' "
                "ignores it; remove it"
            )
        if self.year_mapping == "fixed":
            if self.historical_year is None:
                issues.append("weather.year_mapping='fixed' needs weather.historical_year")
            elif not self.era5_first_year <= self.historical_year <= self.era5_last_year:
                issues.append(
                    f"weather.historical_year={self.historical_year} is outside "
                    f"[{self.era5_first_year}, {self.era5_last_year}]"
                )
        elif self.historical_year is not None:
            issues.append(
                "weather.historical_year is set but weather.year_mapping='sampled' ignores it; "
                "remove it"
            )
        _raise_if(issues)
        return self


BEAM_TABLE_FIELDS: Final[tuple[str, ...]] = (
    "beam_table_max_nodes",
    "beam_table_check_epochs",
    "beam_table_capture_tolerance",
    "beam_table_percentile_tolerance",
    "beam_table_relative_tolerance",
)
"""``models`` fields that only the ``chebyshev_table`` beam evaluation reads."""


def _weather_issues(config: RunConfig) -> list[str]:
    """Cross-section rules of ``models.weather_source`` and the ``weather`` section (M8a)."""
    issues: list[str] = []
    m = config.models
    if m.weather_source == "scenario":
        if config.weather is not None:
            issues.append(
                "the weather section is set but models.weather_source='scenario' ignores it; "
                "set models.weather_source='era5' or remove the section"
            )
        return issues
    if config.weather is None:
        issues.append("models.weather_source='era5' needs the weather section")
    if config.transmitter.mode != "optical":
        issues.append(
            "models.weather_source='era5' (cloud cover as a hard optical outage) requires "
            f"transmitter.mode='optical' (got '{config.transmitter.mode}'); the microwave "
            "path uses the ITU-R cloud/rain statistics (models.atmosphere='itu_r')"
        )
    if m.atmosphere != "scenario" or m.weather_scenario != "clear":
        issues.append(
            "models.weather_source='era5' requires models.atmosphere='scenario' and "
            "models.weather_scenario='clear' (the clear-sky factors apply at clear hours; "
            f"got atmosphere='{m.atmosphere}', weather_scenario='{m.weather_scenario}')"
        )
    if importlib.util.find_spec("xarray") is None:
        issues.append(
            "models.weather_source='era5' needs the optional 'weather' extra (xarray, h5netcdf), "
            "which is not installed (uv sync --extra weather)"
        )
    return issues


class ConstantsConfig(_Section):
    """Physical constants that a study may vary (each with a sourced default)."""

    solar_constant_1au_W_m2: float = _field(
        SOLAR_IRRADIANCE_NOMINAL_1AU_W_M2,
        unit="W m^-2",
        provenance="sourced",
        source="IAU 2015 Resolution B3, nominal total solar irradiance S_sun^N = 1361 W m^-2",
        description="Total solar irradiance at 1 au, S_1AU, in S(r) = S_1AU (au/r)^2.",
        ge=1000.0,
        le=2000.0,
    )


# ---------------------------------------------------------------------------------------------
# Constellation (M7; PDF §11.2-11.4, §10.4)
# ---------------------------------------------------------------------------------------------

PhaseDistribution = Literal["even", "random"]
SchedulerPolicy = Literal["greedy_max_power", "optimal_assignment"]

RING_ELEMENT_FIELDS: Final[tuple[str, ...]] = (
    "semimajor_axis_au",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "arg_periapsis_deg",
)
"""Orbital elements a ring may set (``None`` inherits the value of the ``orbit`` section)."""

RING_TECHNOLOGY_FIELDS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "collector_area_m2": ("collector", "area_m2"),
        "transmitter_aperture_diameter_m": ("transmitter", "aperture_diameter_m"),
        "transmitter_electrical_to_beam_efficiency": (
            "transmitter",
            "electrical_to_beam_efficiency",
        ),
        "transmitter_pointing_sigma_rad": ("transmitter", "pointing_sigma_rad"),
    }
)
"""Per-ring technology override -> (section, field) of the satellite configuration it replaces."""


class RingTechnologyConfig(_Section):
    """Per-ring technology overrides (``None`` = the value of the base configuration).

    Each set value replaces one field of the satellite's configuration
    (:data:`RING_TECHNOLOGY_FIELDS`); the resulting satellite configuration is validated with
    the same schema rules as any run configuration.
    """

    collector_area_m2: float | None = _field(
        None,
        unit="m^2",
        provenance="assumed",
        description="Replaces collector.area_m2 for the satellites of this ring.",
        gt=0.0,
    )
    transmitter_aperture_diameter_m: float | None = _field(
        None,
        unit="m",
        provenance="assumed",
        description="Replaces transmitter.aperture_diameter_m for this ring.",
        gt=0.0,
    )
    transmitter_electrical_to_beam_efficiency: float | None = _field(
        None,
        unit="1",
        provenance="assumed",
        description="Replaces transmitter.electrical_to_beam_efficiency for this ring.",
        gt=0.0,
        le=1.0,
    )
    transmitter_pointing_sigma_rad: float | None = _field(
        None,
        unit="rad",
        provenance="assumed",
        description="Replaces transmitter.pointing_sigma_rad for this ring.",
        ge=0.0,
    )


class RingConfig(_Section):
    """One orbital ring of identical, independently orbiting satellites (PDF §11.2).

    The ring's orbital elements default to those of the ``orbit`` section (``None`` = inherit);
    the force model, orbit model and element epoch always come from ``orbit``. Satellite ``k``
    (``k = 0 .. n_satellites-1``) gets the phase angle ``phi_k = phase_offset_deg + span k / n``
    (``even``) or ``phase_offset_deg + U_k``, ``U_k ~ uniform[0, span)`` drawn from the run's
    Generator seeded with ``project.seed`` (``random``), ``span = phase_span_deg`` (default
    360), reduced mod 360 and interpreted according to
    ``phase_reference`` exactly like ``orbit.true_anomaly_deg``. Even spacing is in true anomaly
    (absolute) or ecliptic longitude (earth_relative): equal spacing in time only for circular
    orbits.
    """

    name: str = _field(
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Unique ring name.",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        max_length=64,
    )
    n_satellites: int = _field(
        unit="-",
        provenance="assumed",
        description="Number of satellites on the ring.",
        ge=1,
        le=100_000,
    )
    semimajor_axis_au: float | None = _field(
        None,
        unit="au",
        provenance="assumed",
        description="Semimajor axis (None = orbit.semimajor_axis_au).",
        gt=0.0,
        le=100.0,
    )
    eccentricity: float | None = _field(
        None,
        unit="1",
        provenance="assumed",
        description="Eccentricity (None = orbit.eccentricity).",
        ge=0.0,
        lt=1.0,
    )
    inclination_deg: float | None = _field(
        None,
        unit="deg",
        provenance="assumed",
        description="Inclination to the ecliptic of J2000 (None = orbit.inclination_deg).",
        ge=0.0,
        le=180.0,
    )
    raan_deg: float | None = _field(
        None,
        unit="deg",
        provenance="assumed",
        description="Longitude of the ascending node (None = orbit.raan_deg).",
        ge=0.0,
        lt=360.0,
    )
    arg_periapsis_deg: float | None = _field(
        None,
        unit="deg",
        provenance="assumed",
        description="Argument of periapsis (None = orbit.arg_periapsis_deg).",
        ge=0.0,
        lt=360.0,
    )
    phase_distribution: PhaseDistribution = _field(
        "even",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="'even' (phi_k = offset + span k/n) or 'random' (offset + uniform[0, span) "
        "from the Generator seeded with project.seed).",
    )
    phase_offset_deg: float | None = _field(
        None,
        unit="deg",
        provenance="assumed",
        description="Phase of satellite 0 (None = orbit.true_anomaly_deg), interpreted with "
        "phase_reference.",
        ge=0.0,
        lt=360.0,
    )
    phase_span_deg: float = _field(
        360.0,
        unit="deg",
        provenance="assumed",
        description="Arc the phases are spread over: phi_k = offset + span k/n ('even') or "
        "offset + U_k, U_k ~ uniform[0, span) ('random'). 360 = the whole ring; a smaller arc "
        "keeps the satellites out of a sector (e.g. near the Earth).",
        gt=0.0,
        le=360.0,
    )
    phase_reference: PhaseReference | None = _field(
        None,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="'absolute' or 'earth_relative' (orbit.phase_reference semantics; None = "
        "orbit.phase_reference).",
    )
    technology: RingTechnologyConfig = Field(default_factory=RingTechnologyConfig)


class SchedulerConfig(_Section):
    """Deterministic, capacity-constrained link scheduler (PDF §11.2, §13.1 scheduler row).

    At every epoch the feasible satellite-to-site links are assigned subject to beam
    exclusivity (one target per satellite), the beam slots of each site, the summed
    accepted-power and heat limits of each site (excess clipped and reported) and the
    power-density interlock on the sum of the incident beams' peak irradiances (incoherent
    sum), and the optional transmitter duty cycle. Retargeting is instantaneous: a satellite
    may serve a different site at the next epoch with no slew or acquisition time (named
    approximation; the number of retargeting events is reported).
    """

    policy: SchedulerPolicy = _field(
        "greedy_max_power",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="'greedy_max_power': links in order (captured power desc, satellite index, "
        "site index), each taken if its satellite is free and its site has capacity; "
        "'optimal_assignment': per-epoch maximum of the delivered power (exact MILP / "
        "assignment, scipy.optimize).",
    )
    duty_cycle_max: float | None = _field(
        None,
        unit="1",
        provenance="assumed",
        description="Transmitter duty-cycle limit: a satellite transmits at most "
        "floor(duty_cycle_max * N_w) of any N_w = floor(duty_cycle_window_s / output_step_s) "
        "consecutive output epochs (causal: enforced epoch by epoch). None = no limit.",
        gt=0.0,
        le=1.0,
    )
    duty_cycle_window_s: float | None = _field(
        None,
        unit="s",
        provenance="assumed",
        description="Length of the sliding duty-cycle window (required with duty_cycle_max).",
        gt=0.0,
    )
    retarget_model: Literal["instantaneous"] = _field(
        "instantaneous",
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Slew/retarget time between targets: 'instantaneous' (ignored; named "
        "approximation, retarget events counted). No finite-time model is implemented.",
    )

    @model_validator(mode="after")
    def _check_duty(self) -> Self:
        if (self.duty_cycle_max is None) != (self.duty_cycle_window_s is None):
            raise ValueError(
                "constellation.scheduler.duty_cycle_max and duty_cycle_window_s must be set "
                "together"
            )
        return self


class MassProxyConfig(_Section):
    """Assumed specific masses of the deployed-mass proxy (PDF §11.4 "mass proxy").

    ``m_sat = bus_mass_kg + collector_areal_mass_kg_m2 A_col + s_rad A_rad
    + transmitter_areal_mass_kg_m2 pi D^2 / 4``, with the radiator areal mass
    ``s_rad = thermal.radiator_specific_mass_kg_m2`` (the existing M4 field; required when a
    mass proxy is configured, so one physical parameter has one field) and ``A_rad`` the
    physical radiator area ``thermal.radiator_area_m2`` if configured, else the largest
    required radiator area of the satellite over the run (M4 sizing at the operating point).
    A *proxy*, not a mass budget: no structure, propellant, attitude control or margins beyond
    what the numbers stand for. No default: every value is a scenario assumption stated in the
    configuration.
    """

    collector_areal_mass_kg_m2: float = _field(
        unit="kg m^-2",
        provenance="assumed",
        description="Mass per collector area (PV blanket and its support).",
        gt=0.0,
    )
    transmitter_areal_mass_kg_m2: float = _field(
        unit="kg m^-2",
        provenance="assumed",
        description="Mass per transmitter aperture area pi D^2/4.",
        gt=0.0,
    )
    bus_mass_kg: float = _field(
        unit="kg",
        provenance="assumed",
        description="Fixed bus mass per satellite.",
        ge=0.0,
    )


class ConstellationConfig(_Section):
    """Constellation of independent satellites on one or more rings (PDF §11.2-11.4).

    The base ``orbit``, ``collector``, ``transmitter`` and ``thermal`` sections define every
    satellite; each ring sets its elements, phases and technology overrides. Every satellite
    is an independent test particle and an independent transmitter: received powers of
    different satellites add (incoherent, PDF §11.3); no coherent combining is modelled.
    """

    rings: tuple[RingConfig, ...] = _field(
        unit="-",
        provenance="assumed",
        kind="metadata",
        description="Rings (at least one; unique names).",
        min_length=1,
        strict=False,
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    mass_proxy: MassProxyConfig | None = _field(
        None,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Assumed specific masses of the mass proxy (None = no mass proxy and no "
        "power per mass).",
    )

    @field_validator("rings")
    @classmethod
    def _unique_rings(cls, rings: tuple[RingConfig, ...]) -> tuple[RingConfig, ...]:
        names = [r.name for r in rings]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"constellation.rings names must be unique; duplicated: {dupes}")
        return rings

    @property
    def n_satellites(self) -> int:
        """Total number of satellites."""
        return sum(r.n_satellites for r in self.rings)


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
    constellation: ConstellationConfig | None = _field(
        None,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="Constellation definition (M7). None = single-satellite run of the orbit "
        "section (exactly one receiver site).",
    )
    weather: WeatherConfig | None = _field(
        None,
        unit="-",
        provenance="assumed",
        kind="model_choice",
        description="ERA5 site-weather settings (M8a); required by and only allowed with "
        "models.weather_source='era5'.",
    )
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
        if self.models.atmosphere != "scenario" and self.models.weather_scenario != "clear":
            issues.append(
                f"models.weather_scenario='{self.models.weather_scenario}' has no effect with "
                f"models.atmosphere='{self.models.atmosphere}' (only the 'scenario' model uses "
                "it); use models.atmosphere='scenario' or weather_scenario='clear'"
            )
        if self.models.atmosphere == "itu_r":
            if importlib.util.find_spec("itur") is None:
                issues.append(
                    "models.atmosphere='itu_r' needs the optional dependency 'itur', which is "
                    "not installed in this environment (install the extra: uv sync --extra itur)"
                )
            if mode != "microwave":
                issues.append(
                    "models.atmosphere='itu_r' (ITU-R P.618 radio-wave attenuation) requires "
                    f"transmitter.mode='microwave' (got '{mode}')"
                )
            else:
                f_hz = SPEED_OF_LIGHT_M_S / self.transmitter.wavelength_m
                f_lo, f_hi = ITU_R_FREQUENCY_RANGE_HZ
                if not f_lo <= f_hz <= f_hi:
                    issues.append(
                        f"models.atmosphere='itu_r' needs a carrier frequency in "
                        f"[{f_lo:.6g}, {f_hi:.6g}] Hz; transmitter.wavelength_m="
                        f"{self.transmitter.wavelength_m} gives {f_hz:.6g} Hz"
                    )
            if self.receiver.min_elevation_deg < self.atmosphere.itu_r_min_elevation_deg:
                issues.append(
                    f"receiver.min_elevation_deg={self.receiver.min_elevation_deg} is below "
                    f"atmosphere.itu_r_min_elevation_deg={self.atmosphere.itu_r_min_elevation_deg}"
                    " required by models.atmosphere='itu_r'"
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
        if not self.collector.max_operating_temperature_K > self.thermal.space_temperature_K:
            issues.append(
                f"collector.max_operating_temperature_K="
                f"{self.collector.max_operating_temperature_K} must exceed "
                f"thermal.space_temperature_K={self.thermal.space_temperature_K} (the PV array "
                "cannot be colder than the radiative sink)"
            )
        if (
            self.orbit.model == "prescribed_ephemeris_perturbed_test_particle"
            and self.time.integrator_step_s is None
        ):
            issues.append(
                "time.integrator_step_s is required when orbit.model="
                "'prescribed_ephemeris_perturbed_test_particle'"
            )
        if self.orbit.model == "two_body_kepler":
            l1_only = (
                "integrator_step_s",
                "integrator_rtol",
                "integrator_atol_position_m",
                "integrator_atol_velocity_m_s",
                "ephemeris_cache_step_s",
            )
            for name in l1_only:
                # An explicit value equal to the schema default is the default (a resolved dump
                # writes every field and must load again, M7); any other value is ignored.
                default = TimeConfig.model_fields[name].default
                explicit = (
                    name in self.time.model_fields_set and getattr(self.time, name) != default
                )
                if (name == "integrator_step_s" and self.time.integrator_step_s is not None) or (
                    name != "integrator_step_s" and explicit
                ):
                    issues.append(
                        f"time.{name} is set but orbit.model='two_body_kepler' is analytic "
                        "and ignores it; remove it or use "
                        "'prescribed_ephemeris_perturbed_test_particle'"
                    )
        if self.constellation is not None:
            issues.extend(_constellation_issues(self))
        issues.extend(_weather_issues(self))
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


def _satellite_mapping(
    config: RunConfig, ring_index: int, phase_angle_deg: float
) -> dict[str, Any]:
    """Mapping of the single-satellite configuration of one ring satellite (see
    :func:`satellite_config`)."""
    if config.constellation is None:
        raise ValueError("configuration has no constellation section")
    ring = config.constellation.rings[ring_index]
    # exclude_unset: keep "explicitly set" exactly as in the source configuration (validators
    # such as orbit.perturbing_bodies distinguish set from defaulted fields).
    data = config.model_dump(mode="json", exclude_unset=True)
    data.pop("constellation", None)
    orbit = data["orbit"]
    for name in RING_ELEMENT_FIELDS:
        value = getattr(ring, name)
        if value is not None:
            orbit[name] = value
    phase = float(phase_angle_deg) % 360.0
    orbit["true_anomaly_deg"] = 0.0 if phase >= 360.0 else phase
    if ring.phase_reference is not None:
        orbit["phase_reference"] = ring.phase_reference
    for name, (section, key) in RING_TECHNOLOGY_FIELDS.items():
        value = getattr(ring.technology, name)
        if value is not None:
            data.setdefault(section, {})[key] = value
    data["provenance_overrides"] = {
        k: v for k, v in config.provenance_overrides.items() if not k.startswith("constellation")
    }
    return data


def duty_cycle_epochs(duty_cycle_max: float, window_s: float, step_s: float) -> tuple[int, int]:
    """Duty-cycle window and allowance in output epochs: ``N_w = floor(window_s / step_s)`` and
    ``m = floor(duty_cycle_max * N_w)``.

    Both floors are taken in exact rational arithmetic on the shortest decimal representation of
    each value (``Fraction(repr(x))``, i.e. the number as written in the configuration), so that
    exact decimal products are not rounded down by binary floating point
    (``0.29 * 100 = 28.999999999999996`` in IEEE 754 would give 28 transmitting epochs, not 29).

    Args:
        duty_cycle_max: Allowed transmitting fraction (0, 1].
        window_s: Sliding window length [s] (> 0).
        step_s: Output step [s] (> 0).

    Returns:
        ``(N_w, m)``.
    """
    if not (duty_cycle_max > 0.0 and window_s > 0.0 and step_s > 0.0):
        raise ValueError("duty_cycle_max, window_s and step_s must be > 0")
    n_w = math.floor(Fraction(repr(float(window_s))) / Fraction(repr(float(step_s))))
    m = math.floor(Fraction(repr(float(duty_cycle_max))) * n_w)
    return int(n_w), int(m)


def ring_phase_offset_deg(config: RunConfig, ring_index: int) -> float:
    """Phase of satellite 0 of a ring: ``phase_offset_deg`` or ``orbit.true_anomaly_deg`` [deg]."""
    if config.constellation is None:
        raise ValueError("configuration has no constellation section")
    ring = config.constellation.rings[ring_index]
    if ring.phase_offset_deg is not None:
        return ring.phase_offset_deg
    if config.orbit.true_anomaly_deg is None:
        raise ValueError("orbit.true_anomaly_deg is not set (geostationary preset)")
    return config.orbit.true_anomaly_deg


def satellite_config(config: RunConfig, ring_index: int, phase_angle_deg: float) -> RunConfig:
    """Single-satellite :class:`RunConfig` of one satellite of ring ``ring_index``.

    The base configuration without its ``constellation`` section, with the ring's elements
    (where set), ``orbit.true_anomaly_deg = phase_angle_deg mod 360`` and
    ``orbit.phase_reference`` of the ring (where set), and the ring's technology overrides;
    provenance overrides of constellation keys are dropped. Validated with the full schema, so a
    ring override that breaks any rule fails exactly as it would in a single-satellite file.

    Raises:
        ConfigError: the satellite configuration is invalid.
    """
    data = _satellite_mapping(config, ring_index, phase_angle_deg)
    ring = config.constellation.rings[ring_index] if config.constellation else None
    return parse_config(data, source=f"constellation ring {ring.name if ring else ring_index}")


def _constellation_issues(config: RunConfig) -> list[str]:
    """Cross-section checks of the constellation section (rings validated as satellites)."""
    assert config.constellation is not None
    c = config.constellation
    issues: list[str] = []
    if config.orbit.elements_center != "SUN":
        return [
            "constellation: geocentric orbits (orbit.elements_center='EARTH') are a "
            "single-satellite GEO baseline (M7b); constellations are heliocentric"
        ]
    for i, ring in enumerate(c.rings):
        data = _satellite_mapping(config, i, ring_phase_offset_deg(config, i))
        try:
            RunConfig.model_validate(data)
        except ValidationError as exc:
            for issue in issues_from_validation_error(exc):
                issues.append(
                    f"constellation.rings[{i}] ('{ring.name}') gives an invalid satellite "
                    f"configuration: {issue.format()}"
                )
    if config.models.beam_evaluation != "per_epoch":
        issues.append(
            "constellation: models.beam_evaluation='chebyshev_table' is implemented for the "
            "single-satellite run only; use 'per_epoch' (or the batched backends)"
        )
    if c.mass_proxy is not None and config.thermal.radiator_specific_mass_kg_m2 is None:
        issues.append(
            "constellation.mass_proxy needs thermal.radiator_specific_mass_kg_m2 (the radiator "
            "areal mass of the proxy)"
        )
    sch = c.scheduler
    if sch.duty_cycle_max is not None and sch.duty_cycle_window_s is not None:
        n_window, n_on = duty_cycle_epochs(
            sch.duty_cycle_max, sch.duty_cycle_window_s, config.time.output_step_s
        )
        if n_window < 1 or n_on < 1:
            issues.append(
                "constellation.scheduler: duty_cycle_max * floor(duty_cycle_window_s / "
                f"time.output_step_s) = {sch.duty_cycle_max} * {n_window} allows no transmitting "
                "epoch per window (need >= 1)"
            )
    return issues


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
    """Section class of a field annotation (``Section`` or ``Section | None``), else None."""
    if isinstance(annotation, type) and issubclass(annotation, _Section):
        return annotation
    args = get_args(annotation)
    if args and type(None) in args:
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1 and isinstance(rest[0], type) and issubclass(rest[0], _Section):
            return rest[0]
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
        if _section_type(annotation) is not None and value is not None:
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


@dataclass(frozen=True, slots=True)
class IdealisedSetting:
    """A configured value that makes a stage lossless or a constraint inactive.

    Attributes:
        path: Dotted configuration path.
        value: Configured value.
        effect: What the value idealises (the result is an upper bound in that respect).
    """

    path: str
    value: Any
    effect: str


def idealised_settings(config: RunConfig) -> tuple[IdealisedSetting, ...]:
    """Active settings that turn a result into an idealised upper bound (PDF §13.4).

    PDF §13.4: "If a model is only an idealized upper bound, label it as such in the figure and
    manuscript." The rules mirror ``docs/model_assumptions.md`` "Optimistic defaults"; a value
    is listed only if it is the lossless / unconstrained one in *this* configuration, so a
    study that overrides it with a sourced value is not labelled. Three groups:

    * lossless stages: every efficiency of the PDF §10.1 chain equal to 1, a perfect
      wavefront, perfect pointing, PV efficiency clipped only at 1, an isolated PV plate;
    * inactive constraints: a disabled Earth/Sun/Moon occultation check, no pointing-
      availability threshold, no receiver power-density or heat-rejection limit, no Sun
      avoidance;
    * weather and atmosphere: no atmosphere at all, or the deterministic scenario model with
      the ``clear`` scenario (one weather state for the whole window, so never a cloud outage:
      availability is a weather upper bound, PDF §10.2/§17; not listed with
      ``models.weather_source = 'era5'``, whose ERA5 cloud cover sets hard outages and the
      clear scenario only the clear-hour factors), and a configured scenario whose zenith
      factor is lossless (optical transmittance 1, microwave attenuation 0 dB; with ERA5 the
      clear-hour factor).

    Not listed: ``models.light_time = false`` (an approximation of either sign, reported by
    :func:`approximations_in_force`), a finite ``receiver.max_input_power_W`` (whether it binds
    is a run outcome, reported as ``receiver_saturation_time_fraction``), the ITU-R statistical
    attenuation (exceeded p % of the time, not an upper bound) and scalar diffraction without
    turbulence (labelled separately in every beam figure).
    """
    c = config
    out: list[IdealisedSetting] = []

    def add(path: str, value: Any, effect: str) -> None:
        out.append(IdealisedSetting(path=path, value=value, effect=effect))

    tx, rx, m = c.transmitter, c.receiver, c.models
    # Lossless stages (PDF §10.1 chain, in chain order).
    if tx.power_conditioning_efficiency == 1.0:
        add("transmitter.power_conditioning_efficiency", 1.0, "lossless power conditioning")
    if tx.electrical_to_beam_efficiency == 1.0:
        add(
            "transmitter.electrical_to_beam_efficiency",
            1.0,
            "lossless electrical-to-beam conversion",
        )
    if tx.optics_or_rf_efficiency == 1.0:
        add("transmitter.optics_or_rf_efficiency", 1.0, "lossless transmit optics / RF feed")
    if tx.rms_wavefront_error_m == 0.0:
        add("transmitter.rms_wavefront_error_m", 0.0, "perfect wavefront (no Ruze/Strehl loss)")
    if tx.pointing_sigma_rad == 0.0 and tx.pointing_bias_rad == 0.0:
        add("transmitter.pointing_sigma_rad", 0.0, "perfect pointing (no bias, no jitter)")
    if rx.conversion_efficiency == 1.0:
        add("receiver.conversion_efficiency", 1.0, "lossless receiver conversion")
    if rx.grid_efficiency == 1.0:
        add("receiver.grid_efficiency", 1.0, "lossless grid conditioning")
    if c.collector.incidence_mode == "sun_tracking":
        add(
            "collector.incidence_mode",
            "sun_tracking",
            "normal solar incidence at every epoch (optimistic bound, PDF §8.1)",
        )
    if c.collector.pv_efficiency_max == 1.0:
        add(
            "collector.pv_efficiency_max",
            1.0,
            "PV efficiency clipped only at the energy-conservation bound",
        )
    if c.thermal.pv_view_factor == 1.0:
        add("thermal.pv_view_factor", 1.0, "isolated PV plate (no view of the spacecraft)")
    # Inactive constraints.
    for body in ("earth", "sun", "moon"):
        if not getattr(m, f"{body}_occultation"):
            add(
                f"models.{body}_occultation",
                False,
                f"{body.capitalize()} occultation not checked (inactive constraint: a blocked "
                "beam counts as delivered)",
            )
    if tx.min_pointing_efficiency is None:
        add(
            "transmitter.min_pointing_efficiency",
            None,
            "no pointing-availability threshold (inactive constraint)",
        )
    if rx.max_power_density_W_m2 is None:
        add("receiver.max_power_density_W_m2", None, "no power-density / beam-abort limit")
    if rx.max_heat_rejection_W is None:
        add("receiver.max_heat_rejection_W", None, "no receiver heat-rejection limit")
    if rx.sun_avoidance_deg == 0.0:
        add("receiver.sun_avoidance_deg", 0.0, "no Sun-avoidance constraint at the receiver")
    # Weather and atmosphere.
    if m.atmosphere == "none":
        add("models.atmosphere", "none", "no atmospheric attenuation and no weather outage")
    elif m.atmosphere == "scenario":
        scenario = m.weather_scenario
        # With ERA5 site weather (M8a) the clear scenario only sets the clear-sky factors; the
        # cloudy hours are hard outages, so the sky is not permanently clear.
        if scenario == "clear" and m.weather_source != "era5":
            add(
                "models.weather_scenario",
                "clear",
                "permanently clear sky: no cloud outage at any epoch (availability is a "
                "weather upper bound)",
            )
        a = c.atmosphere
        if tx.mode == "optical":
            tau = (
                a.optical_zenith_transmittance_clear
                if scenario == "clear"
                else a.optical_zenith_transmittance_cloudy
            )
            if tau == 1.0:
                add(
                    f"atmosphere.optical_zenith_transmittance_{scenario}",
                    1.0,
                    f"no {scenario}-sky optical extinction",
                )
        else:
            att = (
                a.microwave_zenith_attenuation_clear_dB
                if scenario == "clear"
                else a.microwave_zenith_attenuation_cloudy_dB
            )
            if att == 0.0:
                add(
                    f"atmosphere.microwave_zenith_attenuation_{scenario}_dB",
                    0.0,
                    f"no {scenario}-sky microwave attenuation",
                )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class NamedApproximation:
    """A configured model choice that replaces a physical model by a named approximation.

    PDF §17: "Never silently replace a physical model with an approximation: name it, document
    its validity and provide a comparison path."

    Attributes:
        path: Dotted configuration path of the setting that selects it.
        value: Configured value.
        approximation: What is approximated and its validity.
        comparison_path: How to quantify its effect.
    """

    path: str
    value: Any
    approximation: str
    comparison_path: str

    def describe(self) -> str:
        """One-line text ``path = value: approximation; comparison: ...``."""
        return (
            f"{self.path} = {self.value}: {self.approximation}; comparison path: "
            f"{self.comparison_path}"
        )


def approximations_in_force(config: RunConfig) -> tuple[NamedApproximation, ...]:
    """Named approximations selected by this configuration (PDF §17; run log, manifest, report).

    * ``collector.incidence_mode = fixed_normal`` with SRP in the L1 propagator: the SRP force
      stays the radial cannonball law on the full ``A/m`` (PDF §6.3). For a flat plate with
      specular reflectivity ``eps = C_r - 1`` (rest absorbed; Montenbruck & Gill 2000,
      §3.4; ``s`` from the Sun to the satellite) at incidence ``theta`` the true force is
      ``P A cos(theta) [(1 - eps) s + 2 eps cos(theta) n]``, so the radial model overestimates
      the radial component by ``(1 + eps) / (cos(theta) [(1 - eps) + 2 eps cos^2(theta)])``
      (``1/cos(theta)`` only for ``eps = 0``) and omits the non-radial component
      ``2 eps P A cos^2(theta) sin(theta)`` along the plate normal.
    * ``models.light_time = false``: instantaneous geometry (site at the emission epoch).
    * a constellation with more than one satellite (and light time on): the scheduler sums and
      limits the beams of one *emission* epoch at each receiver, although their light times
      differ (emission-epoch aggregation; bounded on every run by the reception-time
      reconstruction, M7 review).
    * ``orbit.elements_center = EARTH`` (M7b GEO baseline): L0 Earth point mass with the
      lunisolar, J2 and SRP perturbations omitted ("L0 + station-keeping"; magnitudes in
      docs/model_assumptions.md, M7b).
    """
    out: list[NamedApproximation] = []
    o = config.orbit
    if o.elements_center == "EARTH":
        out.append(
            NamedApproximation(
                path="orbit.elements_center",
                value="EARTH",
                approximation="L0 two-body Kepler about the Earth's centre (Earth point mass); "
                "lunisolar third-body (~0.84 deg/yr inclination growth in 2030), J2 (+522 m "
                "synchronous radius), tesseral harmonics and SRP (eccentricity circle "
                "~(3/2) a_srp/(n r n_sun): 2.9e-4 at A/m = 0.02 m^2/kg, 8.7e-3 = +-367 km at "
                "the comparison collector's ~0.6 m^2/kg) are omitted: station-keeping is "
                "assumed and its propellant is not in the mass proxy "
                "(docs/model_assumptions.md, M7b)",
                comparison_path="helioswarm.geocentric.propagate_geocentric_point_masses "
                "(Earth + Sun + Moon integration) and the closed-form J2/SRP estimates in "
                "helioswarm.geocentric; docs/model_assumptions.md M7b",
            )
        )
    if (
        config.collector.incidence_mode == "fixed_normal"
        and o.model == "prescribed_ephemeris_perturbed_test_particle"
        and o.include_solar_radiation_pressure
    ):
        out.append(
            NamedApproximation(
                path="collector.incidence_mode",
                value="fixed_normal",
                approximation="radial (cannonball) SRP on the full A/m although the collector "
                "normal is inertially fixed: overestimates the radial SRP by "
                "(1+eps)/(cos(theta)[(1-eps)+2 eps cos^2(theta)]), eps = C_r - 1, and omits "
                "the non-radial component along the plate normal (docs/model_assumptions.md, "
                "M6b)",
                comparison_path="rerun with orbit.include_solar_radiation_pressure = false to "
                "bound the SRP contribution; attitude-dependent SRP (L3) is deferred to M8",
            )
        )
    if not config.models.light_time:
        out.append(
            NamedApproximation(
                path="models.light_time",
                value=False,
                approximation="instantaneous geometry: the receiver site is taken at the "
                "emission epoch (no retarded-time targeting, PDF §7.1); during the light "
                "time range/c the site moves |v_site| range/c (per au of range ~1.5e4 km from "
                "the Earth's orbital motion and up to ~230 km from its rotation)",
                comparison_path="rerun with models.light_time = true (the default)",
            )
        )
    w = config.weather
    if config.models.weather_source == "era5" and w is not None:
        out.append(
            NamedApproximation(
                path="models.weather_source",
                value="era5",
                approximation="ERA5 total cloud cover (fraction of a ~0.25 deg model grid box, "
                f"hourly) used as the optical line-of-sight state ({w.outage_model}"
                + (
                    f", outage iff tcc > {w.cloud_cover_threshold}"
                    if w.outage_model == "threshold"
                    else ", clear with probability 1 - tcc, independent hourly draws"
                )
                + "); cloud optical depth, cloud geometry along the slant path and sub-hour "
                "variability are not modelled; historical weather of a mapped year stands in "
                "for the simulated year",
                comparison_path="helioswarm weather climatology (threshold and Bernoulli clear "
                "fractions per site); rerun with another threshold / outage_model / "
                "historical_year, or with models.weather_source = 'scenario' (clear upper "
                "bound)",
            )
        )
    c = config.constellation
    if c is not None and c.n_satellites > 1 and config.models.light_time:
        out.append(
            NamedApproximation(
                path="constellation",
                value=f"{c.n_satellites} satellites",
                approximation="emission-epoch aggregation: beams scheduled at the same emission "
                "epoch are summed, clipped and limited (beam slots, input/heat limit, density "
                "interlock) together at each receiver although they arrive up to the spread of "
                "their light times (range/c) apart; P_grid per receiver is reported on the "
                "emission grid",
                comparison_path="the reception-time reconstruction written by every run "
                "(receivers.parquet rx_slot_* columns, summary.json reception_time: beams, "
                "density and accepted energy in reception time)",
            )
        )
    return tuple(out)


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


EXTENDS_KEY: Final[str] = "extends"
"""Top-level key of a run-configuration file naming the file it extends (M7b)."""

MAX_EXTENDS_DEPTH: Final[int] = 16
"""Longest accepted ``extends`` chain."""


def merge_config_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge of two configuration mappings: ``override`` wins.

    Mappings are merged key by key, recursively; every other value (scalars, ``null``, lists
    such as ``receiver.sites``) in ``override`` replaces the base value as a whole. Neither
    input is modified.
    """
    out: dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge_config_mappings(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_extends(path: Path, chain: tuple[Path, ...]) -> tuple[Any, tuple[Path, ...]]:
    p = path.resolve()
    if p in chain:
        raise ConfigError(
            str(chain[0]),
            [
                ConfigIssue(
                    EXTENDS_KEY,
                    "circular 'extends' chain: " + " -> ".join(str(c) for c in (*chain, p)),
                )
            ],
        )
    if len(chain) >= MAX_EXTENDS_DEPTH:
        raise ConfigError(
            str(chain[0]),
            [ConfigIssue(EXTENDS_KEY, f"'extends' chain longer than {MAX_EXTENDS_DEPTH} files")],
        )
    data = read_yaml(p)
    chain = (*chain, p)
    if not isinstance(data, Mapping) or EXTENDS_KEY not in data:
        return data, chain
    target = data[EXTENDS_KEY]
    if not isinstance(target, str) or not target:
        raise ConfigError(
            str(p),
            [
                ConfigIssue(
                    EXTENDS_KEY,
                    "must be the path of one configuration file (relative to this file)",
                    value=target,
                    has_value=True,
                )
            ],
        )
    base, chain = _resolve_extends(p.parent / target, chain)
    if not isinstance(base, Mapping):
        raise ConfigError(
            str(p),
            [ConfigIssue(EXTENDS_KEY, f"extended file {target!r} is not a mapping of sections")],
        )
    child = {k: v for k, v in data.items() if k != EXTENDS_KEY}
    return merge_config_mappings(base, child), chain


def read_config_mapping(path: str | Path) -> Any:
    """Read a run-configuration YAML file and resolve its ``extends`` chain (M7b).

    A file may name one other file with a top-level ``extends: <path>`` (relative to the
    file's directory). The extended file is read the same way (chains are allowed; cycles and
    chains longer than :data:`MAX_EXTENDS_DEPTH` are rejected) and the extending file is
    deep-merged over it (:func:`merge_config_mappings`: mappings merge, every other value
    replaces). The result is an ordinary mapping without ``extends``; it is validated as a
    whole, so a file that is only a fragment (e.g. a shared technology block) is never a valid
    run configuration on its own. Used for the harmonised comparison configurations, whose
    shared assumptions are literally the same file.

    Raises:
        ConfigError: unreadable file, YAML error, a bad ``extends`` value or a cycle.
    """
    data, _ = _resolve_extends(Path(path), ())
    return data


def config_sources(path: str | Path) -> tuple[Path, ...]:
    """The files a run configuration is assembled from: ``path`` first, then each file it
    extends (resolved absolute paths; one file when there is no ``extends``)."""
    _, chain = _resolve_extends(Path(path), ())
    return chain


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
    """Read and validate a YAML run configuration file (``extends`` resolved,
    :func:`read_config_mapping`).

    Raises :class:`ConfigError` with aggregated, human-readable issues on any failure.
    """
    return parse_config(read_config_mapping(path), source=str(path))


def omits_when_default(model_cls: type[BaseModel], name: str) -> bool:
    """Whether field ``name`` of ``model_cls`` is omitted from :func:`resolved_dict` at its
    default (``_field(..., omit_when_default=True)``)."""
    extra = model_cls.model_fields[name].json_schema_extra
    return isinstance(extra, dict) and extra.get(_OMIT_WHEN_DEFAULT_KEY) == "true"


def _strip_omitted_defaults(model: _Section, data: dict[str, Any]) -> None:
    """Remove, in place, the ``omit_when_default`` fields of ``model`` that hold their default
    (recursing into sub-sections and tuples of sections)."""
    for name, info in type(model).model_fields.items():
        if name not in data:
            continue
        value = getattr(model, name)
        if omits_when_default(type(model), name):
            default = info.get_default(call_default_factory=True)
            if value == default:
                del data[name]
                continue
        if isinstance(value, _Section) and isinstance(data[name], dict):
            _strip_omitted_defaults(value, data[name])
        elif isinstance(value, tuple) and isinstance(data[name], list):
            for item, item_data in zip(value, data[name], strict=True):
                if isinstance(item, _Section) and isinstance(item_data, dict):
                    _strip_omitted_defaults(item, item_data)


def resolved_dict(config: RunConfig) -> dict[str, Any]:
    """JSON-compatible dict of the fully resolved configuration, including all defaults.

    Exception (M7b): fields declared ``omit_when_default`` are left out while they hold their
    default (:func:`omits_when_default`), so adding them did not change the canonical JSON or
    the hash of any configuration that does not use them. Loading the dict again restores the
    same defaults, so the round trip is unchanged.
    """
    data = config.model_dump(mode="json")
    _strip_omitted_defaults(config, data)
    return data


def _hash_input(config: RunConfig) -> dict[str, Any]:
    """:func:`resolved_dict` without the M8a keys while they are off.

    ``models.weather_source = 'scenario'`` and an absent ``weather`` section are omitted, so
    every configuration that does not use the M8a weather model keeps its pre-M8a hash (they
    select exactly the pre-M8a behaviour). Any other value is hashed.
    """
    data = resolved_dict(config)
    if data["models"].get("weather_source") == "scenario":
        del data["models"]["weather_source"]
    if data.get("weather", 0) is None:
        del data["weather"]
    return data


def canonical_json(config: RunConfig) -> str:
    """Canonical JSON of the resolved config (M8a weather keys omitted while off,
    :func:`_hash_input`): sorted keys, no whitespace, ASCII, shortest round-trip float repr.
    This is the input of :func:`config_hash`."""
    return json.dumps(
        _hash_input(config),
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
    "BEAM_TABLE_FIELDS",
    "DEFAULT_PERTURBING_BODIES",
    "EARTH_RELATIVE_MIN_ABS_COS_INCLINATION",
    "EPHEMERIS_CACHE_STEP_MAX_S",
    "EXTENDS_KEY",
    "INTEGRATOR_RTOL_MAX",
    "INTEGRATOR_RTOL_MIN",
    "MICROWAVE_BEAM_MODELS",
    "MICROWAVE_WAVELENGTH_RANGE_M",
    "OPTICAL_BEAM_MODELS",
    "OPTICAL_WAVELENGTH_RANGE_M",
    "RING_ELEMENT_FIELDS",
    "RING_TECHNOLOGY_FIELDS",
    "SCHEMA_VERSION",
    "AtmosphereConfig",
    "CollectorConfig",
    "ConfigError",
    "ConfigIssue",
    "ConstantsConfig",
    "ConstellationConfig",
    "FieldMeta",
    "FramesConfig",
    "IdealisedSetting",
    "MassProxyConfig",
    "ModelsConfig",
    "NamedApproximation",
    "OrbitConfig",
    "PerturbingBody",
    "PhaseDistribution",
    "PhaseReference",
    "ProjectConfig",
    "ProvenanceRow",
    "ReceiverConfig",
    "RingConfig",
    "RingTechnologyConfig",
    "RunConfig",
    "SchedulerConfig",
    "SchedulerPolicy",
    "SiteConfig",
    "ThermalConfig",
    "TimeConfig",
    "TransmitterConfig",
    "WeatherConfig",
    "WeatherSource",
    "approximations_in_force",
    "canonical_json",
    "canonical_utc",
    "config_hash",
    "config_sources",
    "dump_resolved_yaml",
    "duty_cycle_epochs",
    "field_meta",
    "idealised_settings",
    "leaf_paths",
    "load_config",
    "merge_config_mappings",
    "omits_when_default",
    "parse_config",
    "provenance_table",
    "read_config_mapping",
    "read_yaml",
    "resolved_dict",
    "ring_phase_offset_deg",
    "satellite_config",
    "write_resolved_yaml",
]
