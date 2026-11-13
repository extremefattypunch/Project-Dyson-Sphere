"""Response curves of one configuration (PDF §8.4 "Required results", §9.5).

``helioswarm characterize CONFIG --out DIR`` evaluates the deterministic M4/M5 models of one
validated configuration over parameter grids (no orbit, no ephemeris, no kernels) and writes

    power_thermal_vs_distance.parquet   PDF §8.4: solar flux and collector power, PV efficiency
                                        and temperature, waste heat and radiator area, net
                                        electrical power to the transmitter, thermal
                                        feasibility and margin, versus heliocentric distance
                                        and transmitter operating point (eta_electrical_to_beam)
    capture_vs_pointing.parquet         PDF §9.5: capture efficiency versus pointing sigma and
                                        receiver radius at one stated link geometry
    capture_vs_wavefront.parquet        PDF §9.2/§9.3/§9.5: capture versus RMS wavefront (phase)
                                        error (Ruze/Strehl factor) at the same geometry
    characterization.json               config hash, grids, geometry, definitions, versions,
                                        idealised (upper-bound) settings of the configuration
    config_resolved.yaml                the configuration with all defaults
    figures/*.png                       drawn only from the saved Parquet files

Every row is one model evaluation of the configuration with exactly one parameter changed
(columns name the changed value). These are model curves, not time series: the link geometry
of the capture curves (range, elevation) is an explicit input recorded with the output, the
collector is at normal incidence and in full sunlight (the Sun-tracking, unshadowed case of
:func:`helioswarm.thermal.power_thermal_vs_distance`), and no availability condition applies.
Outputs are SI with unit-suffixed column names.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from helioswarm import __version__
from helioswarm.aperture import fresnel_number
from helioswarm.beam import beam_settings_from_config, evaluate_beam
from helioswarm.config import RunConfig, config_hash, idealised_settings, write_resolved_yaml
from helioswarm.constants import AU_M
from helioswarm.thermal import power_thermal_models_from_config, power_thermal_vs_distance

CHARACTERIZATION_SCHEMA: Final[str] = "helioswarm-characterization/1"
"""Version tag of ``characterization.json``."""

POWER_THERMAL_FILE: Final[str] = "power_thermal_vs_distance.parquet"
CAPTURE_POINTING_FILE: Final[str] = "capture_vs_pointing.parquet"
CAPTURE_WAVEFRONT_FILE: Final[str] = "capture_vs_wavefront.parquet"
METADATA_FILE: Final[str] = "characterization.json"

DEFAULT_SUN_DISTANCES_AU: Final[tuple[float, ...]] = tuple(
    round(0.3 + 0.05 * k, 2) for k in range(19)
)
"""0.3-1.2 au in 0.05 au steps (the PDF §2 candidate radii lie on this grid)."""

DEFAULT_RECEIVER_RADII_M: Final[tuple[float, ...]] = (1000.0, 5000.0, 10000.0)
"""PDF §2 scenario receiver radii (the configured radius is added)."""

DEFAULT_SIGMA_DIFFRACTION_MULTIPLES: Final[tuple[float, ...]] = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    1.0,
)
"""Pointing sigmas in units of the diffraction angle ``lambda/D`` (configured sigma added)."""

DEFAULT_WAVEFRONT_WAVELENGTH_MULTIPLES: Final[tuple[float, ...]] = (0.0, 0.02, 0.05, 0.1, 0.2)
"""RMS wavefront errors in units of the wavelength (configured value added)."""


@dataclass(frozen=True)
class LinkGeometry:
    """Fixed link geometry of the capture curves (an explicit scenario input).

    Attributes:
        range_m: Transmitter-receiver range [m].
        elevation_rad: Beam elevation at the receiver [rad], in (0, pi/2].
    """

    range_m: float
    elevation_rad: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.range_m) and self.range_m > 0.0):
            raise ValueError("range_m must be finite and > 0")
        if not 0.0 < self.elevation_rad <= 0.5 * math.pi:
            raise ValueError("elevation_rad must lie in (0, pi/2]")


def _with_configured(values: Sequence[float], configured: float) -> tuple[float, ...]:
    return tuple(sorted({float(v) for v in values} | {float(configured)}))


def power_thermal_table(
    config: RunConfig,
    sun_distance_m: Sequence[float],
    electrical_to_beam_efficiencies: Sequence[float] | None = None,
) -> pa.Table:
    """PDF §8.4 curves versus heliocentric distance and transmitter operating point.

    Each operating point replaces ``eta_electrical_to_beam`` of the configured on-board chain
    (the other stage efficiencies, the collector, PV node and radiator are the configured ones)
    and calls :func:`helioswarm.thermal.power_thermal_vs_distance` (normal incidence, full Sun):
    ``S(r) = S_1AU (au/r)^2``, ``P_abs = alpha A S``, self-consistent PV temperature,
    ``P_PV = P_abs eta_PV(T)``, ``P_dc_to_tx = eta_pc P_PV`` (net electrical power available to
    the transmitter), ``Q_waste = P_PV - P_aperture + Q_other`` and the radiator area
    ``A = Q_waste / (eps sigma n F (T_lim^4 - T_space^4))`` (or its temperature for a fixed area).

    Args:
        config: Validated configuration.
        sun_distance_m: Heliocentric distances [m].
        electrical_to_beam_efficiencies: Operating points [1]; default the configured value.

    Returns:
        Long table, one row per (operating point, distance).
    """
    r = np.asarray(sun_distance_m, dtype=np.float64)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("sun_distance_m must be a non-empty 1-D sequence")
    base = power_thermal_models_from_config(config)
    etas = (
        (base.efficiencies.electrical_to_beam,)
        if electrical_to_beam_efficiencies is None
        else tuple(float(e) for e in electrical_to_beam_efficiencies)
    )
    cols: dict[str, list[np.ndarray]] = {}

    def put(name: str, values: object) -> None:
        cols.setdefault(name, []).append(np.asarray(values))

    for eta in etas:
        models = dataclasses.replace(
            base, efficiencies=dataclasses.replace(base.efficiencies, electrical_to_beam=eta)
        )
        res = power_thermal_vs_distance(r, models)
        sol, ch, th = res.solar, res.chain, res.thermal
        put("eta_electrical_to_beam", np.full(r.size, eta))
        put("sun_distance_m", r)
        put("flux_W_m2", sol.flux_W_m2)
        put("incident_W", sol.incident_W)
        put("absorbed_W", sol.absorbed_W)
        put("pv_temperature_K", sol.pv_temperature_K)
        put("pv_efficiency", sol.pv_efficiency)
        put("pv_efficiency_clipped", sol.pv_efficiency_clipped)
        put("pv_electric_W", sol.pv_electric_W)
        put("pv_heat_W", sol.pv_heat_W)
        put("dc_to_tx_W", ch.dc_to_tx_W)
        put("beam_W", ch.beam_W)
        put("aperture_W", ch.aperture_W)
        put("waste_heat_W", th.waste_heat_W)
        put("radiator_area_m2", th.radiator_area_m2)
        put("radiator_temperature_K", th.equilibrium_temperature_K)
        put("pv_margin_K", th.pv_margin_K)
        put("thermal_margin_K", th.margin_K)
        put("thermal_feasible", th.feasible)
    return pa.table({k: pa.array(np.concatenate(v)) for k, v in cols.items()})


def capture_pointing_table(
    config: RunConfig,
    geometry: LinkGeometry,
    pointing_sigma_rad: Sequence[float],
    receiver_radius_m: Sequence[float],
) -> pa.Table:
    """PDF §9.5 "capture efficiency versus pointing sigma and receiver radius".

    Each row is :func:`helioswarm.beam.evaluate_beam` of the configured beam (model,
    wavelength, aperture, bias, wavefront and lattice factors, receiver orientation) at
    ``geometry`` with ``pointing_sigma_rad`` and ``receiver.radius_m`` replaced. ``capture``
    is the expected capture over the Rice-distributed offset (bias + isotropic jitter),
    ``capture_nominal`` the perfect-pointing value and ``p05/p50/p95`` the percentiles over
    the jitter.

    Raises:
        BeamModelValidityError: the beam model is invalid at this geometry.
    """
    base = beam_settings_from_config(config)
    rng = np.array([geometry.range_m])
    el = np.array([geometry.elevation_rad])
    cols: dict[str, list[float]] = {}
    for a in receiver_radius_m:
        for s in pointing_sigma_rad:
            st = dataclasses.replace(base, receiver_radius_m=float(a), pointing_sigma_rad=float(s))
            b = evaluate_beam(rng, el, st)
            row = {
                "receiver_radius_m": float(a),
                "pointing_sigma_rad": float(s),
                "pointing_sigma_over_diffraction": float(s)
                * st.aperture.diameter_m
                / st.wavelength_m,
                "pointing_bias_rad": st.pointing_bias_rad,
                "range_m": geometry.range_m,
                "elevation_rad": geometry.elevation_rad,
                "projected_receiver_radius_m": float(b.projected_receiver_radius_m[0]),
                "fresnel_number": float(b.fresnel_number[0]),
                "capture": float(b.capture_fraction[0]),
                "capture_nominal": float(b.capture_fraction_nominal[0]),
                "capture_p05": float(b.capture_fraction_p05[0]),
                "capture_p50": float(b.capture_fraction_p50[0]),
                "capture_p95": float(b.capture_fraction_p95[0]),
                "pointing_efficiency": float(b.pointing_loss[0]),
            }
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return pa.table({k: pa.array(np.asarray(v, dtype=np.float64)) for k, v in cols.items()})


def capture_wavefront_table(
    config: RunConfig, geometry: LinkGeometry, rms_wavefront_error_m: Sequence[float]
) -> pa.Table:
    """Capture versus RMS wavefront (phase) error at ``geometry`` (PDF §9.2/§9.3/§9.5).

    The wavefront factor is the Ruze/Strehl law ``exp(-(2 pi sigma_w / lambda)^2)`` applied
    once to the far-field pattern (:func:`helioswarm.aperture.wavefront_efficiency`); for a
    phased array it is the phase-error degradation of the main lobe. The configured pointing
    and receiver are kept.
    """
    base = beam_settings_from_config(config)
    rng = np.array([geometry.range_m])
    el = np.array([geometry.elevation_rad])
    cols: dict[str, list[float]] = {}
    for w in rms_wavefront_error_m:
        st = dataclasses.replace(base, rms_wavefront_error_m=float(w))
        b = evaluate_beam(rng, el, st)
        row = {
            "rms_wavefront_error_m": float(w),
            "phase_rms_rad": 2.0 * math.pi * float(w) / st.wavelength_m,
            "wavefront_efficiency": st.wavefront_efficiency,
            "capture": float(b.capture_fraction[0]),
            "range_m": geometry.range_m,
            "elevation_rad": geometry.elevation_rad,
        }
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    return pa.table({k: pa.array(np.asarray(v, dtype=np.float64)) for k, v in cols.items()})


@dataclass(frozen=True)
class CharacterizationOutputs:
    """Files written by :func:`write_characterization`.

    Attributes:
        out_dir: Output directory.
        files: Relative file names written (tables, metadata, resolved config).
        figures: Figure paths.
    """

    out_dir: Path
    files: tuple[str, ...]
    figures: tuple[Path, ...]


def write_characterization(
    config: RunConfig,
    out_dir: Path,
    *,
    geometry: LinkGeometry,
    sun_distance_m: Sequence[float] | None = None,
    electrical_to_beam_efficiencies: Sequence[float] | None = None,
    receiver_radius_m: Sequence[float] | None = None,
    pointing_sigma_rad: Sequence[float] | None = None,
    rms_wavefront_error_m: Sequence[float] | None = None,
) -> CharacterizationOutputs:
    """Evaluate the three curve families and write them into a prepared ``out_dir``.

    Defaults: distances :data:`DEFAULT_SUN_DISTANCES_AU`; the configured operating point;
    receiver radii :data:`DEFAULT_RECEIVER_RADII_M` plus the configured one; pointing sigmas
    :data:`DEFAULT_SIGMA_DIFFRACTION_MULTIPLES` times ``lambda/D`` plus the configured one;
    wavefront errors :data:`DEFAULT_WAVEFRONT_WAVELENGTH_MULTIPLES` times ``lambda`` plus the
    configured one. Figures are drawn afterwards from the saved files only.
    """
    from helioswarm.io import file_sha256, write_json
    from helioswarm.manifest import git_info, package_versions, utc_now_iso
    from helioswarm.plotting import make_characterization_figures

    tx = config.transmitter
    diffraction_rad = tx.wavelength_m / tx.aperture_diameter_m
    r = (
        tuple(au * AU_M for au in DEFAULT_SUN_DISTANCES_AU)
        if sun_distance_m is None
        else tuple(float(v) for v in sun_distance_m)
    )
    radii = _with_configured(
        DEFAULT_RECEIVER_RADII_M if receiver_radius_m is None else receiver_radius_m,
        config.receiver.radius_m,
    )
    sigmas = _with_configured(
        tuple(k * diffraction_rad for k in DEFAULT_SIGMA_DIFFRACTION_MULTIPLES)
        if pointing_sigma_rad is None
        else pointing_sigma_rad,
        tx.pointing_sigma_rad,
    )
    wfe = _with_configured(
        tuple(k * tx.wavelength_m for k in DEFAULT_WAVEFRONT_WAVELENGTH_MULTIPLES)
        if rms_wavefront_error_m is None
        else rms_wavefront_error_m,
        tx.rms_wavefront_error_m,
    )
    start = utc_now_iso()
    tables = {
        POWER_THERMAL_FILE: power_thermal_table(config, r, electrical_to_beam_efficiencies),
        CAPTURE_POINTING_FILE: capture_pointing_table(config, geometry, sigmas, radii),
        CAPTURE_WAVEFRONT_FILE: capture_wavefront_table(config, geometry, wfe),
    }
    digest = config_hash(config)
    files: dict[str, str] = {}
    for name, table in tables.items():
        path = out_dir / name
        meta = {b"config_hash": digest.encode(), b"schema": CHARACTERIZATION_SCHEMA.encode()}
        pq.write_table(table.replace_schema_metadata(meta), path, compression="zstd")
        files[name] = file_sha256(path)
    cfg_path = write_resolved_yaml(config, out_dir / "config_resolved.yaml")
    files["config_resolved.yaml"] = file_sha256(cfg_path)
    git = git_info()
    write_json(
        out_dir / METADATA_FILE,
        {
            "schema": CHARACTERIZATION_SCHEMA,
            "helioswarm_version": __version__,
            "project_name": config.project.name,
            "config_hash": digest,
            "git": dataclasses.asdict(git),
            "packages": package_versions(),
            "wall_clock_start_utc": start,
            "wall_clock_end_utc": utc_now_iso(),
            "link_geometry": {
                "range_m": geometry.range_m,
                "elevation_rad": geometry.elevation_rad,
                "fresnel_number": float(
                    fresnel_number(tx.aperture_diameter_m, tx.wavelength_m, geometry.range_m)
                ),
                "note": "explicit scenario input of the capture curves, not a run geometry",
            },
            "collector_condition": "normal incidence, full Sun (Sun-tracking, unshadowed: "
            "optimistic bound, PDF §8.1)",
            "idealised_settings": [
                {"path": i.path, "value": i.value, "effect": i.effect}
                for i in idealised_settings(config)
            ],
            "diffraction_angle_rad": diffraction_rad,
            "files": files,
            "definitions": {
                POWER_THERMAL_FILE: power_thermal_table.__doc__,
                CAPTURE_POINTING_FILE: capture_pointing_table.__doc__,
                CAPTURE_WAVEFRONT_FILE: capture_wavefront_table.__doc__,
            },
        },
    )
    figures = make_characterization_figures(out_dir)
    return CharacterizationOutputs(
        out_dir=out_dir, files=(*files.keys(), METADATA_FILE), figures=figures
    )


__all__ = [
    "CAPTURE_POINTING_FILE",
    "CAPTURE_WAVEFRONT_FILE",
    "CHARACTERIZATION_SCHEMA",
    "DEFAULT_RECEIVER_RADII_M",
    "DEFAULT_SIGMA_DIFFRACTION_MULTIPLES",
    "DEFAULT_SUN_DISTANCES_AU",
    "DEFAULT_WAVEFRONT_WAVELENGTH_MULTIPLES",
    "METADATA_FILE",
    "POWER_THERMAL_FILE",
    "CharacterizationOutputs",
    "LinkGeometry",
    "capture_pointing_table",
    "capture_wavefront_table",
    "power_thermal_table",
    "write_characterization",
]
