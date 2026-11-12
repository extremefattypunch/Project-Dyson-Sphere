"""Beam evaluation along a trajectory: model selection, validity, capture and ``BeamResult``.

This module joins the propagation models (``aperture.py``, ``optical.py``, ``microwave.py``)
and the capture/pointing statistics (``capture.py``) into the PDF p.4 ``BeamResult``,
vectorised over epochs. It is the single place where the beam-side factors of the PDF §10.1
chain are combined::

    capture_fraction = eta_wavefront * eta_lattice * E[eta_diffraction(d)]        (expected)
    capture_fraction_nominal = eta_wavefront * eta_lattice * eta_diffraction(0)   (perfect pointing)
    pointing_loss = capture_fraction / capture_fraction_nominal

* ``eta_diffraction(d)``: fraction of the beam power (normalised to the power leaving the
  aperture, ``P_aperture``) that falls on the projected receiver disk offset by ``d``
  (``capture.py``); its expectation over the Rice-distributed pointing offset.
* ``eta_wavefront = exp(-(2 pi sigma_w / lambda)^2)`` (single owner
  :func:`helioswarm.aperture.wavefront_efficiency`): the coherent fraction; the scattered
  remainder is conservatively assumed **not** captured (it spreads into a halo much wider than
  the diffraction pattern when the error correlation length is small compared with ``D``).
* ``eta_lattice``: main-lobe fraction of a sparse phased-array lattice (microwave with
  ``array_element_spacing_m``; 1 otherwise), ``microwave.lattice_main_lobe_fraction``.

There is no separate "aperture efficiency" (PDF §10.1): the taper is inside the propagated
pattern (``taper_efficiency`` is reported as a diagnostic only) and spill-over is owned by
``transmitter.optics_or_rf_efficiency`` upstream. The atmosphere (``atmosphere.py``) is applied
by the link budget, not here.

Model selection (``models.beam``; schema-checked against ``transmitter.mode``):

=====================  ========================================================  ==============
model                  pattern                                                   validity check
=====================  ========================================================  ==============
``airy_far_field``     :class:`optical.AiryPattern` (uniform illumination)       N_F <= limit
``gaussian_beam``      :class:`optical.GaussianBeamPattern` (untruncated TEM00)  a/w0 >= ratio
``aperture_far_field`` :class:`aperture.FarFieldAperturePattern` (any taper)     N_F <= limit
``fresnel_hankel``     :class:`aperture.FresnelHankelPattern` (any taper)        paraxial phase
=====================  ========================================================  ==============

Violations raise :class:`helioswarm.aperture.BeamModelValidityError` for the whole call (never
silently replaced). ``N_F = D^2/(lambda R)`` is computed for every epoch and returned.

Receiver geometry: the receiver is replaced by the circle of equal projected area seen along
the beam (``capture.projected_receiver_radius_m``; ``receiver.orientation``). Pointing offsets at
the receiver are ``R * bias`` and ``R * sigma`` in the plane normal to the beam.

Peak irradiance (PDF §10.4, §10.5 "power density"): ``peak_irradiance_per_W_m2`` is the
maximum of the *aberration-free* pattern per watt arriving at the receiver plane (an upper bound
for the safety constraint: the Strehl reduction and pointing jitter are deliberately not used to
lower it). With ``incident_power_W`` (the beam power reaching the receiver plane, i.e. after the
atmosphere) the peak in W m^-2 and the ``receiver.max_power_density_W_m2`` check are added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.aperture import (
    ApertureField,
    BeamModelValidityError,
    FarFieldAperturePattern,
    FresnelHankelPattern,
    RadialBeamPattern,
    aperture_from_config,
    fresnel_number,
    wavefront_efficiency,
)
from helioswarm.capture import (
    RICE_SPAN_SIGMAS,
    ReceiverOrientation,
    capture_fraction_centred_ellipse,
    capture_statistics,
    projected_receiver_radius_m,
)
from helioswarm.config import (
    MICROWAVE_BEAM_MODELS,
    OPTICAL_BEAM_MODELS,
    BeamModel,
    RunConfig,
    TransmitterMode,
)
from helioswarm.microwave import LatticeArray
from helioswarm.optical import AiryPattern, GaussianBeamPattern, airy_first_null_radius_m

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

FAR_FIELD_MODELS: Final[frozenset[str]] = frozenset({"airy_far_field", "aperture_far_field"})
"""Beam models that are Fraunhofer approximations (subject to the Fresnel-number limit)."""

CAPTURE_TOLERANCE: Final[float] = 1.0e-12
"""Slack allowed above 1 on capture fractions (quadrature rounding) before a result is refused."""


@dataclass(frozen=True)
class BeamSettings:
    """Everything the beam model needs, taken from one validated configuration.

    Attributes:
        mode: ``optical`` or ``microwave``.
        beam_model: ``models.beam``.
        wavelength_m: Carrier wavelength [m].
        aperture: Aperture field (diameter, illumination).
        receiver_radius_m: Physical receiver radius [m].
        receiver_orientation: ``horizontal`` or ``tracking``.
        pointing_bias_rad: Deterministic pointing-bias magnitude [rad].
        pointing_sigma_rad: Per-axis 1-sigma isotropic jitter [rad].
        rms_wavefront_error_m: RMS wavefront error [m].
        max_fresnel_number: Far-field validity limit [1].
        gaussian_beam_min_truncation_ratio: ``a/w0`` floor of ``gaussian_beam`` [1].
        lattice: Phased-array lattice (microwave only) or ``None`` (filled aperture).
        max_power_density_W_m2: Beam-abort power-density limit [W m^-2] or ``None``.
        quadrature_refinement: Multiplies all quadrature panel counts (numerics, not physics).
    """

    mode: TransmitterMode
    beam_model: BeamModel
    wavelength_m: float
    aperture: ApertureField
    receiver_radius_m: float
    receiver_orientation: ReceiverOrientation
    pointing_bias_rad: float
    pointing_sigma_rad: float
    rms_wavefront_error_m: float = 0.0
    max_fresnel_number: float = 0.5
    gaussian_beam_min_truncation_ratio: float = 2.5
    lattice: LatticeArray | None = None
    max_power_density_W_m2: float | None = None
    quadrature_refinement: float = 1.0

    def __post_init__(self) -> None:
        allowed = OPTICAL_BEAM_MODELS if self.mode == "optical" else MICROWAVE_BEAM_MODELS
        if self.beam_model not in allowed:
            raise ValueError(f"beam model '{self.beam_model}' not allowed in mode '{self.mode}'")
        illum = self.aperture.illumination
        if self.beam_model == "airy_far_field" and illum != "uniform_circular":
            raise ValueError("airy_far_field requires uniform_circular illumination")
        if self.beam_model == "gaussian_beam":
            if illum != "truncated_gaussian":
                raise ValueError("gaussian_beam requires truncated_gaussian illumination")
            assert self.aperture.gaussian_waist_radius_m is not None
            ratio = self.aperture.radius_m / self.aperture.gaussian_waist_radius_m
            if ratio < self.gaussian_beam_min_truncation_ratio:
                raise BeamModelValidityError(
                    f"gaussian_beam: a/w0 = {ratio:.6g} < {self.gaussian_beam_min_truncation_ratio}"
                    " (truncation too strong for the untruncated Gaussian); use fresnel_hankel"
                )
        if self.lattice is not None and self.mode != "microwave":
            raise ValueError("a phased-array lattice is a microwave geometry")
        for name in ("wavelength_m", "receiver_radius_m", "max_fresnel_number"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"BeamSettings.{name} must be finite and > 0")
        for name in ("pointing_bias_rad", "pointing_sigma_rad", "rms_wavefront_error_m"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"BeamSettings.{name} must be finite and >= 0")

    @property
    def wavefront_efficiency(self) -> float:
        """``exp(-(2 pi sigma_w / lambda)^2)`` (single owner in ``aperture.py``)."""
        return wavefront_efficiency(self.rms_wavefront_error_m, self.wavelength_m)

    @property
    def lattice_main_lobe_fraction(self) -> float:
        """Grating-lobe main-beam fraction (1 for a filled aperture)."""
        return 1.0 if self.lattice is None else self.lattice.main_lobe_fraction(self.wavelength_m)


def beam_settings_from_config(
    config: RunConfig, *, quadrature_refinement: float = 1.0
) -> BeamSettings:
    """:class:`BeamSettings` from a validated run configuration."""
    tx = config.transmitter
    lattice = None
    if tx.array_element_spacing_m is not None:
        lattice = LatticeArray(
            spacing_m=tx.array_element_spacing_m,
            aperture_diameter_m=tx.aperture_diameter_m,
            element_exponent_q=(
                1.0
                if tx.array_element_pattern_exponent is None
                else tx.array_element_pattern_exponent
            ),
        )
    return BeamSettings(
        mode=tx.mode,
        beam_model=config.models.beam,
        wavelength_m=tx.wavelength_m,
        aperture=aperture_from_config(tx),
        receiver_radius_m=config.receiver.radius_m,
        receiver_orientation=config.receiver.orientation,
        pointing_bias_rad=tx.pointing_bias_rad,
        pointing_sigma_rad=tx.pointing_sigma_rad,
        rms_wavefront_error_m=tx.rms_wavefront_error_m,
        max_fresnel_number=config.models.fraunhofer_max_fresnel_number,
        gaussian_beam_min_truncation_ratio=config.models.gaussian_beam_min_truncation_ratio,
        lattice=lattice,
        max_power_density_W_m2=config.receiver.max_power_density_W_m2,
        quadrature_refinement=quadrature_refinement,
    )


def make_beam_pattern(
    settings: BeamSettings, range_m: float, rho_max_m: float
) -> RadialBeamPattern:
    """Beam pattern per watt at ``range_m`` for ``settings.beam_model`` (validity enforced).

    Args:
        settings: Beam settings.
        range_m: Transmitter-receiver distance [m].
        rho_max_m: Largest transverse radius at which the pattern will be evaluated [m]
            (enters the Fresnel paraxial check).

    Raises:
        BeamModelValidityError: model outside its validity range at this range.
    """
    model = settings.beam_model
    ref = settings.quadrature_refinement
    if model == "airy_far_field":
        return AiryPattern(
            aperture_diameter_m=settings.aperture.diameter_m,
            wavelength_m=settings.wavelength_m,
            range_m=range_m,
            max_fresnel_number=settings.max_fresnel_number,
        )
    if model == "gaussian_beam":
        assert settings.aperture.gaussian_waist_radius_m is not None
        return GaussianBeamPattern(
            waist_radius_m=settings.aperture.gaussian_waist_radius_m,
            wavelength_m=settings.wavelength_m,
            range_m=range_m,
        )
    if model == "aperture_far_field":
        return FarFieldAperturePattern(
            aperture=settings.aperture,
            wavelength_m=settings.wavelength_m,
            range_m=range_m,
            max_fresnel_number=settings.max_fresnel_number,
            refinement=ref,
        )
    if model == "fresnel_hankel":
        return FresnelHankelPattern(
            aperture=settings.aperture,
            wavelength_m=settings.wavelength_m,
            range_m=range_m,
            paraxial_rho_max_m=rho_max_m,
            refinement=ref,
        )
    raise ValueError(f"unknown beam model '{model}'")


@dataclass(frozen=True, eq=False)
class BeamFieldDiagnostics:
    """Pattern and quadrature diagnostics of a :class:`BeamResult` (per-epoch arrays shape (N,)).

    Attributes:
        taper_efficiency: Aperture illumination efficiency (diagnostic; already in the pattern).
        source_spillover_fraction: Gaussian source power outside the aperture (owned by
            ``optics_or_rf_efficiency``; never applied here).
        edge_intensity_ratio: ``|A(a)|^2/|A(0)|^2``.
        feature_scale_m: Transverse feature scale of the pattern at the receiver [m].
        first_null_radius_m: Airy first-null radius ``1.21967 lambda R/D`` (uniform
            illumination, far field; NaN otherwise) [m].
        gaussian_beam_radius_m: ``w(z)`` (``gaussian_beam``; NaN otherwise) [m].
        receiver_angular_radius_rad: Projected receiver radius / range [rad].
        receiver_projection_error: Relative error of the equal-area-circle receiver at zero
            pointing offset, ``(eta_circle - eta_ellipse) / eta_ellipse`` with the exact
            centred projected ellipse ``a x a sin(el)``
            (``capture.capture_fraction_centred_ellipse``);
            0 for a tracking receiver. Positive = the circle is optimistic.
        offset_probability_mass: Quadrature mass of the Rice density (1 to ~1e-12).
        percentile_method: Method of the capture percentiles per epoch.
        lattice_element_count: Elements of the lattice (0 for a filled aperture).
    """

    taper_efficiency: float
    source_spillover_fraction: float
    edge_intensity_ratio: float
    feature_scale_m: Float64Array
    first_null_radius_m: Float64Array
    gaussian_beam_radius_m: Float64Array
    receiver_angular_radius_rad: Float64Array
    receiver_projection_error: Float64Array
    offset_probability_mass: Float64Array
    percentile_method: NDArray[np.str_]
    lattice_element_count: int = 0


@dataclass(frozen=True, eq=False)
class BeamResult:
    """Beam propagation and capture per epoch (PDF p.4 ``BeamResult``; arrays shape (N,)).

    All capture fractions are fractions of the beam power reaching the receiver plane
    (``P_atm`` in the PDF §10.1 chain, i.e. ``P_aperture`` times the atmosphere factor, which is
    applied elsewhere) and lie in [0, 1].

    Attributes:
        wavelength_m: Carrier wavelength [m].
        aperture_m: Transmit aperture diameter ``D`` [m].
        range_m: Transmitter-receiver range per epoch [m].
        elevation_rad: Beam elevation at the receiver per epoch [rad].
        beam_model: ``models.beam``.
        mode: ``optical`` or ``microwave``.
        fresnel_number: ``D^2/(lambda R)`` per epoch.
        fresnel_number_limit: Far-field limit applied (``None`` for non-far-field models).
        capture_fraction: Expected capture including pointing, wavefront and lattice factors.
        capture_fraction_nominal: Capture with perfect pointing (no bias, no jitter).
        capture_fraction_bias_only: Capture with the bias and no jitter.
        capture_fraction_p05, capture_fraction_p50, capture_fraction_p95: Percentiles of the
            capture distribution over the pointing jitter.
        pointing_loss: ``capture_fraction / capture_fraction_nominal`` (<= 1 when the pattern
            peaks on axis; can exceed 1 in the Fresnel near field).
        wavefront_efficiency: Ruze/Strehl factor applied [1].
        lattice_main_lobe_fraction: Phased-array grating-lobe factor applied [1].
        projected_receiver_radius_m: Equal-projected-area receiver radius [m].
        pointing_bias_rad, pointing_sigma_rad: Pointing-error model [rad].
        peak_irradiance_per_W_m2: Maximum aberration-free irradiance per watt [m^-2].
        peak_irradiance_W_m2: Peak irradiance for ``incident_power_W`` [W m^-2] (or ``None``).
        exceeds_max_power_density: Peak above ``receiver.max_power_density_W_m2`` (``None`` if
            no power was given or no limit is configured).
        receiver_orientation: ``horizontal`` or ``tracking``.
        diagnostics: :class:`BeamFieldDiagnostics`.
        validity_notes: Assumptions and checks applied.
        scalar_diffraction: Always True (no polarisation / vector effects; PDF §9.2).
        coherent_combining: Always False (independent transmitters; PDF §9.3).
        epochs_tdb_s: Epochs [TDB s] if supplied.
        time_scale: ``TDB``.
    """

    wavelength_m: float
    aperture_m: float
    range_m: Float64Array
    elevation_rad: Float64Array
    beam_model: BeamModel
    mode: TransmitterMode
    fresnel_number: Float64Array
    fresnel_number_limit: float | None
    capture_fraction: Float64Array
    capture_fraction_nominal: Float64Array
    capture_fraction_bias_only: Float64Array
    capture_fraction_p05: Float64Array
    capture_fraction_p50: Float64Array
    capture_fraction_p95: Float64Array
    pointing_loss: Float64Array
    wavefront_efficiency: float
    lattice_main_lobe_fraction: float
    projected_receiver_radius_m: Float64Array
    pointing_bias_rad: float
    pointing_sigma_rad: float
    peak_irradiance_per_W_m2: Float64Array
    peak_irradiance_W_m2: Float64Array | None
    exceeds_max_power_density: BoolArray | None
    receiver_orientation: ReceiverOrientation
    diagnostics: BeamFieldDiagnostics = field(repr=False)
    validity_notes: tuple[str, ...] = ()
    scalar_diffraction: bool = True
    coherent_combining: bool = False
    epochs_tdb_s: Float64Array | None = None
    time_scale: str = "TDB"

    def __post_init__(self) -> None:
        if self.scalar_diffraction is not True:
            raise ValueError("BeamResult models scalar diffraction only (scalar_diffraction=True)")
        if self.coherent_combining is not False:
            raise ValueError(
                "coherent combining across transmitters is not modelled (PDF §9.3): independent "
                "satellites add power incoherently"
            )
        if self.time_scale != "TDB":
            raise ValueError("BeamResult epochs are TDB seconds")
        shape = np.shape(self.range_m)
        for name, value in list(vars(self).items()):
            if isinstance(value, np.ndarray):
                if value.shape != shape:
                    raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
                arr = np.array(value, copy=True)
                arr.setflags(write=False)
                object.__setattr__(self, name, arr)
        for name in (
            "capture_fraction",
            "capture_fraction_nominal",
            "capture_fraction_bias_only",
            "capture_fraction_p05",
            "capture_fraction_p50",
            "capture_fraction_p95",
        ):
            v = getattr(self, name)
            if np.any(~np.isfinite(v)) or np.any(v < 0.0) or np.any(v > 1.0 + CAPTURE_TOLERANCE):
                raise ValueError(f"BeamResult.{name} must lie in [0, 1]")


def evaluate_beam(
    range_m: ArrayLike,
    elevation_rad: ArrayLike,
    settings: BeamSettings,
    *,
    incident_power_W: ArrayLike | None = None,
    epochs_tdb_s: ArrayLike | None = None,
) -> BeamResult:
    """Beam capture, pointing statistics and peak irradiance per epoch (PDF §9).

    Args:
        range_m: Transmitter-receiver range per epoch [m] (e.g. ``AccessResult.range_m``).
        elevation_rad: Beam elevation at the receiver per epoch [rad], in (0, pi/2]
            (``AccessResult.elevation_rad``; only epochs with access should be passed when the
            receiver is horizontal).
        settings: Beam settings (one validated configuration).
        incident_power_W: Beam power reaching the receiver plane per epoch [W] (optional; adds
            the peak irradiance in W m^-2 and the power-density check).
        epochs_tdb_s: Epochs [TDB s] (optional labels).

    Returns:
        :class:`BeamResult`.

    Raises:
        BeamModelValidityError: the selected model is outside its validity range at any epoch.
    """
    r = np.atleast_1d(np.asarray(range_m, dtype=np.float64))
    el = np.atleast_1d(np.asarray(elevation_rad, dtype=np.float64))
    if r.shape != el.shape or r.ndim != 1:
        raise ValueError("range_m and elevation_rad must be 1-D arrays of equal length")
    nf = fresnel_number(settings.aperture.diameter_m, settings.wavelength_m, r)
    a_eff = projected_receiver_radius_m(
        settings.receiver_radius_m, el, settings.receiver_orientation
    )
    n = r.size
    bias = settings.pointing_bias_rad
    sigma = settings.pointing_sigma_rad
    if settings.lattice is not None:
        max_angle = float(np.max(a_eff / r)) + bias + RICE_SPAN_SIGMAS * sigma
        settings.lattice.check_validity(settings.wavelength_m, max_angle)
    eta_wf = settings.wavefront_efficiency
    eta_lat = settings.lattice_main_lobe_fraction
    factor = eta_wf * eta_lat
    nominal = np.empty(n)
    expected = np.empty(n)
    bias_only = np.empty(n)
    p05 = np.empty(n)
    p50 = np.empty(n)
    p95 = np.empty(n)
    peak = np.empty(n)
    feature = np.empty(n)
    mass = np.empty(n)
    proj_err = np.zeros(n)
    methods = np.empty(n, dtype="<U16")
    first_null = np.full(n, np.nan)
    w_gauss = np.full(n, np.nan)
    for i in range(n):
        nu = r[i] * bias
        s = r[i] * sigma
        # Largest transverse radius at which the pattern is evaluated: the offset-capture
        # window, the physical receiver radius (exact-ellipse diagnostic) and the peak search
        # of the Fresnel pattern (a_tx + 4 lambda R / D); enters the Fresnel paraxial check.
        rho_max = max(
            float(a_eff[i]) + nu + RICE_SPAN_SIGMAS * s,
            settings.receiver_radius_m,
            settings.aperture.radius_m
            + 4.0 * settings.wavelength_m * float(r[i]) / settings.aperture.diameter_m,
        )
        pattern = make_beam_pattern(settings, float(r[i]), rho_max)
        stats = capture_statistics(
            pattern, float(a_eff[i]), nu, s, refinement=settings.quadrature_refinement
        )
        nominal[i] = stats.nominal
        expected[i] = stats.expected
        bias_only[i] = stats.bias_only
        p05[i], p50[i], p95[i] = stats.p05, stats.p50, stats.p95
        mass[i] = stats.offset_probability_mass
        if settings.receiver_orientation == "horizontal":
            a_phys = settings.receiver_radius_m
            ellipse = capture_fraction_centred_ellipse(
                pattern, a_phys, a_phys * math.sin(float(el[i])), n_angles=256
            )
            proj_err[i] = (stats.nominal - ellipse) / ellipse if ellipse > 0.0 else 0.0
        methods[i] = stats.percentile_method
        peak[i] = pattern.peak_intensity_m2_inv()
        feature[i] = pattern.feature_scale_m
        if isinstance(pattern, GaussianBeamPattern):
            w_gauss[i] = pattern.beam_radius_m
    if (
        settings.aperture.illumination == "uniform_circular"
        and settings.beam_model in FAR_FIELD_MODELS
    ):
        first_null = airy_first_null_radius_m(
            settings.wavelength_m, r, settings.aperture.diameter_m
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        pointing_loss = np.where(nominal > 0.0, expected / nominal, np.nan)
    peak_w = None
    exceeds = None
    if incident_power_W is not None:
        p_in = np.broadcast_to(np.asarray(incident_power_W, dtype=np.float64), r.shape)
        if np.any(~np.isfinite(p_in)) or np.any(p_in < 0.0):
            raise ValueError("incident_power_W must be finite and >= 0")
        peak_w = p_in * peak
        if settings.max_power_density_W_m2 is not None:
            exceeds = peak_w > settings.max_power_density_W_m2
    notes = [
        "scalar diffraction (no polarisation/vector effects, no thermal distortion)",
        "single transmitter; no coherent combining across satellites",
        f"receiver orientation '{settings.receiver_orientation}': equal-projected-area circle",
        "pointing: bias + isotropic 2-D Gaussian jitter, Rice offset, deterministic quadrature",
        "wavefront loss exp(-(2 pi sigma_w/lambda)^2); scattered power assumed not captured",
        "peak irradiance of the aberration-free pattern (upper bound)",
    ]
    if settings.beam_model in FAR_FIELD_MODELS:
        notes.append(
            f"far-field model valid: max N_F = {float(np.max(nf)):.6g} <= "
            f"{settings.max_fresnel_number}"
        )
    if settings.beam_model == "gaussian_beam":
        notes.append("untruncated Gaussian beam (waist in the aperture plane, collimated)")
    if settings.lattice is not None:
        notes.append("broadside square lattice; grating-lobe main-beam fraction applied")
    diag = BeamFieldDiagnostics(
        taper_efficiency=settings.aperture.taper_efficiency,
        source_spillover_fraction=settings.aperture.source_spillover_fraction,
        edge_intensity_ratio=settings.aperture.edge_intensity_ratio,
        feature_scale_m=feature,
        first_null_radius_m=first_null,
        gaussian_beam_radius_m=w_gauss,
        receiver_angular_radius_rad=a_eff / r,
        receiver_projection_error=proj_err,
        offset_probability_mass=mass,
        percentile_method=methods,
        lattice_element_count=0 if settings.lattice is None else settings.lattice.element_count,
    )
    return BeamResult(
        wavelength_m=settings.wavelength_m,
        aperture_m=settings.aperture.diameter_m,
        range_m=r,
        elevation_rad=el,
        beam_model=settings.beam_model,
        mode=settings.mode,
        fresnel_number=nf,
        fresnel_number_limit=(
            settings.max_fresnel_number if settings.beam_model in FAR_FIELD_MODELS else None
        ),
        capture_fraction=factor * expected,
        capture_fraction_nominal=factor * nominal,
        capture_fraction_bias_only=factor * bias_only,
        capture_fraction_p05=factor * p05,
        capture_fraction_p50=factor * p50,
        capture_fraction_p95=factor * p95,
        pointing_loss=pointing_loss,
        wavefront_efficiency=eta_wf,
        lattice_main_lobe_fraction=eta_lat,
        projected_receiver_radius_m=a_eff,
        pointing_bias_rad=bias,
        pointing_sigma_rad=sigma,
        peak_irradiance_per_W_m2=peak,
        peak_irradiance_W_m2=peak_w,
        exceeds_max_power_density=exceeds,
        receiver_orientation=settings.receiver_orientation,
        diagnostics=diag,
        validity_notes=tuple(notes),
        epochs_tdb_s=(None if epochs_tdb_s is None else np.asarray(epochs_tdb_s, dtype=np.float64)),
    )


__all__ = [
    "FAR_FIELD_MODELS",
    "BeamFieldDiagnostics",
    "BeamResult",
    "BeamSettings",
    "beam_settings_from_config",
    "evaluate_beam",
    "make_beam_pattern",
]
