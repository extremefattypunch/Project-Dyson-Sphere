"""Run figures, drawn **only** from the saved run files (PDF §13.4).

Every figure reads ``timeseries.parquet``, ``summary.json`` and ``config_resolved.yaml`` from the
run directory (never in-memory model objects), states the run it belongs to (project name and
configuration hash) and is written to ``figures/``. The object-oriented matplotlib API
(``Figure`` + Agg canvas) is used, so no global pyplot state is touched.

Figures: ``orbit_access.png`` (orbit, Earth range, elevation and access), ``solar_thermal.png``
(PV power, temperatures, radiator area), ``beam_capture.png`` (expected / perfect-pointing
capture, pointing percentile band, pointing efficiency, Fresnel number), ``power_stages.png``
(stage-energy waterfall and ``P_grid(t)``), ``unavailability.png`` (time fractions by primary
reason) and ``composite.png`` (all panels on one page). Idealised settings of the run
(:func:`helioswarm.config.idealised_settings`, :func:`idealised_note`) are named on the
power-stage and composite figures and on the ``characterize`` power/thermal and capture
figures, and the beam figures carry the scalar-diffraction upper-bound note (PDF §13.4).
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from helioswarm.config import RunConfig, idealised_settings, load_config
from helioswarm.constants import AU_M
from helioswarm.link_budget import STAGE_COLUMNS
from helioswarm.system import BLOCKING_REASONS

FIGURE_FILES: Final[tuple[str, ...]] = (
    "orbit_access.png",
    "solar_thermal.png",
    "beam_capture.png",
    "power_stages.png",
    "unavailability.png",
    "composite.png",
)
"""Files written by :func:`make_figures` into ``figures/``."""

UPPER_BOUND_NOTE: Final[str] = (
    "scalar diffraction, no turbulence; peak irradiance of the aberration-free pattern "
    "(idealised upper bound)"
)

_DPI: Final[int] = 110


def idealised_note(config: RunConfig, width: int = 180) -> str:
    """Figure label naming every idealised setting of ``config`` (PDF §13.4).

    The effects of :func:`helioswarm.config.idealised_settings`, joined with ``"; "`` and
    wrapped at ``width`` characters (whitespace only, so each effect stays readable).
    """
    ideal = idealised_settings(config)
    text = (
        "Idealised (upper-bound) settings: " + "; ".join(i.effect for i in ideal)
        if ideal
        else "No idealised stage or constraint settings"
    )
    return textwrap.fill(text, width=width, break_on_hyphens=False, break_long_words=False)


def _num(value: float | None) -> float:
    """``value``, or NaN for an undefined (``None``) summary entry."""
    return math.nan if value is None else value


class _RunData:
    """Columns, summary and configuration of one saved run (read-only)."""

    def __init__(self, run_dir: Path) -> None:
        import pyarrow.parquet as pq

        from helioswarm.io import CONFIG_FILE, SUMMARY_FILE, TIMESERIES_FILE
        from helioswarm.metrics import RunSummary, read_summary

        self.run_dir = run_dir
        table = pq.read_table(run_dir / TIMESERIES_FILE)
        self.cols: dict[str, np.ndarray[Any, Any]] = {
            name: np.asarray(table.column(name).to_numpy(zero_copy_only=False))
            for name in table.column_names
        }
        self.summary: RunSummary = read_summary(run_dir / SUMMARY_FILE)
        self.config: RunConfig = load_config(run_dir / CONFIG_FILE)
        self.days = self.cols["elapsed_s"] / 86400.0
        self.label = (
            f"{self.config.project.name} | config {self.summary.config_hash[:12]} | "
            f"start {self.config.time.start_utc} UTC, step {self.config.time.output_step_s:g} s"
            " | from saved run data"
        )
        self.idealised_note = idealised_note(self.config)

    def col(self, name: str) -> np.ndarray[Any, Any]:
        return self.cols[name]


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=_DPI)
    return path


def _orbit_panels(rd: _RunData, ax_xy: Axes, ax_r: Axes, ax_el: Axes) -> None:
    x = rd.col("sat_x_m") / AU_M
    y = rd.col("sat_y_m") / AU_M
    ax_xy.plot(x, y, lw=1.0)
    ax_xy.plot([0.0], [0.0], marker="*", ms=8, color="orange", ls="none", label="SSB")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("J2000 x [au] (SSB)")
    ax_xy.set_ylabel("J2000 y [au]")
    ax_xy.set_title("Satellite orbit (J2000, SSB)")
    ax_r.plot(rd.days, rd.col("range_m") / AU_M, label="Earth-receiver range")
    ax_r.plot(rd.days, rd.col("sun_distance_m") / AU_M, label="Sun distance")
    ax_r.set_xlabel("time since start [d]")
    ax_r.set_ylabel("distance [au]")
    ax_r.legend(fontsize=7)
    ax_r.set_title("Range and heliocentric distance")
    el = np.degrees(rd.col("elevation_rad"))
    ax_el.plot(rd.days, el, lw=0.6, color="0.4", label="elevation")
    vis = rd.col("visible").astype(bool)
    ax_el.plot(rd.days[vis], el[vis], ".", ms=1.5, label="geometric access")
    ax_el.axhline(rd.config.receiver.min_elevation_deg, color="r", lw=0.8, label="min elevation")
    ax_el.set_xlabel("time since start [d]")
    ax_el.set_ylabel("elevation [deg]")
    ax_el.legend(fontsize=7, loc="lower right")
    ax_el.set_title(f"Access fraction {rd.summary.diurnal_aliasing.access_fraction:.4f}")


def _solar_panels(rd: _RunData, ax_p: Axes, ax_t: Axes) -> None:
    ax_p.plot(rd.days, rd.col("incident_solar_W") / 1e6, label="incident sunlight")
    ax_p.plot(rd.days, rd.col("pv_electric_W") / 1e6, label="P_PV")
    ax_p.plot(rd.days, rd.col("waste_heat_W") / 1e6, label="radiator waste heat")
    ax_p.set_xlabel("time since start [d]")
    ax_p.set_ylabel("power [MW]")
    ax_p.legend(fontsize=7)
    ax_p.set_title(
        "Solar collection (M4)"
        + (
            "; sun-tracking = optimistic bound"
            if rd.config.collector.incidence_mode == "sun_tracking"
            else ""
        ),
        fontsize=8,
    )
    ax_t.plot(rd.days, rd.col("pv_temperature_K"), label="PV temperature")
    ax_t.axhline(
        rd.config.collector.max_operating_temperature_K, color="r", lw=0.8, label="PV limit"
    )
    ax_t.set_xlabel("time since start [d]")
    ax_t.set_ylabel("temperature [K]")
    ax_t.set_title(
        f"Thermal: radiator area max {rd.summary.thermal.radiator_area_m2_max:.4g} m^2, "
        f"feasible {rd.summary.thermal.feasible_time_fraction:.3f}",
        fontsize=8,
    )
    ax_t.legend(fontsize=7)


def _beam_panels(rd: _RunData, ax_c: Axes, ax_pe: Axes) -> None:
    vis = rd.col("visible").astype(bool)
    d = rd.days[vis]
    if vis.any():
        ax_c.fill_between(
            d,
            rd.col("capture_p05")[vis],
            rd.col("capture_p95")[vis],
            alpha=0.3,
            label="p05-p95 (pointing jitter)",
        )
        ax_c.plot(d, rd.col("capture_nominal")[vis], ".", ms=1.5, label="perfect pointing")
        ax_c.plot(d, rd.col("eta_capture")[vis], ".", ms=1.5, label="expected (applied)")
        ax_pe.plot(d, rd.col("pointing_efficiency")[vis], ".", ms=1.5, label="pointing eff.")
        ax_f = ax_pe.twinx()
        ax_f.plot(d, rd.col("fresnel_number")[vis], ".", ms=1.0, color="0.5")
        ax_f.set_ylabel("Fresnel number D^2/(lambda R)", fontsize=7)
    ax_c.set_yscale("log")
    ax_c.set_xlabel("time since start [d]")
    ax_c.set_ylabel("capture fraction")
    ax_c.legend(fontsize=7)
    ax_c.set_title(f"Beam capture ({rd.config.models.beam})")
    ax_pe.set_xlabel("time since start [d]")
    ax_pe.set_ylabel("E[capture]/capture(0)")
    ax_pe.set_title("Pointing efficiency (left) and Fresnel number (right, grey)", fontsize=8)


def _power_panels(rd: _RunData, ax_w: Axes, ax_g: Axes) -> None:
    energy = rd.summary.stage_energy_J
    names = [sym for sym, _ in STAGE_COLUMNS]
    vals = np.array([energy[attr.replace("_W", "_J")] for _, attr in STAGE_COLUMNS]) / 3.6e9
    pos = np.where(vals > 0.0, vals, np.nan)
    ax_w.bar(range(len(names)), pos)
    ax_w.set_yscale("log")
    ax_w.set_xticks(range(len(names)))
    ax_w.set_xticklabels(names, rotation=60, fontsize=7)
    ax_w.set_ylabel("energy over window [MWh]")
    ax_w.set_title("Power-stage waterfall (PDF §10.1)")
    ax_g.plot(rd.days, rd.col("grid_W") / 1e3, lw=0.6)
    ax_g.set_xlabel("time since start [d]")
    ax_g.set_ylabel("P_grid [kW]")
    s = rd.summary
    cf = s.capacity_factor
    ax_g.set_title(
        f"P_grid(t): mean {_num(s.p_grid_W.time_mean) / 1e3:.4g} kW, availability "
        f"{s.availability:.3f}, CF {'n/a' if cf is None else f'{cf:.3f}'}",
        fontsize=8,
    )


def _unavail_panel(rd: _RunData, ax: Axes) -> None:
    blocking = rd.summary.unavailability.blocking
    names = [str(r.name) for r in BLOCKING_REASONS]
    prim = [blocking[n].time_fraction_primary for n in names]
    flagged = [blocking[n].time_fraction_flagged for n in names]
    y = np.arange(len(names))
    ax.barh(y - 0.2, prim, height=0.4, label="primary reason")
    ax.barh(y + 0.2, flagged, height=0.4, label="flagged (overlapping)")
    ax.set_yticks(y)
    ax.set_yticklabels([n.lower() for n in names], fontsize=7)
    ax.set_xlabel("time fraction")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=7)
    ax.set_title(f"Unavailability (availability {rd.summary.availability:.3f})")


def make_figures(run_dir: Path | str) -> tuple[Path, ...]:
    """Write every figure of :data:`FIGURE_FILES` from the saved files of ``run_dir``."""
    rd = _RunData(Path(run_dir))
    out_dir = rd.run_dir / "figures"
    paths: list[Path] = []

    fig = Figure(figsize=(13, 4))
    axs = fig.subplots(1, 3)
    _orbit_panels(rd, axs[0], axs[1], axs[2])
    fig.suptitle(rd.label, fontsize=8)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "orbit_access.png"))

    fig = Figure(figsize=(10, 4))
    axs = fig.subplots(1, 2)
    _solar_panels(rd, axs[0], axs[1])
    fig.suptitle(rd.label, fontsize=8)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "solar_thermal.png"))

    fig = Figure(figsize=(10, 4))
    axs = fig.subplots(1, 2)
    _beam_panels(rd, axs[0], axs[1])
    fig.suptitle(rd.label + "\n" + UPPER_BOUND_NOTE, fontsize=7)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "beam_capture.png"))

    fig = Figure(figsize=(10, 4))
    axs = fig.subplots(1, 2)
    _power_panels(rd, axs[0], axs[1])
    fig.suptitle(rd.label + "\n" + rd.idealised_note, fontsize=6)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "power_stages.png"))

    fig = Figure(figsize=(6, 4))
    ax = fig.subplots(1, 1)
    _unavail_panel(rd, ax)
    fig.suptitle(rd.label, fontsize=6)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "unavailability.png"))

    fig = Figure(figsize=(16, 15))
    axs = fig.subplots(4, 3)
    _orbit_panels(rd, axs[0, 0], axs[0, 1], axs[0, 2])
    _solar_panels(rd, axs[1, 0], axs[1, 1])
    _unavail_panel(rd, axs[1, 2])
    _beam_panels(rd, axs[2, 0], axs[2, 1])
    axs[2, 2].axis("off")
    s = rd.summary
    text = [
        f"P_grid mean {_num(s.p_grid_W.time_mean):.4g} W",
        f"median {s.p_grid_W.median:.4g} W, p95 {s.p_grid_W.p95:.4g} W",
        f"peak {s.p_grid_W.peak:.4g} W",
        f"availability {s.availability:.4f}",
        f"delivered energy {s.delivered_energy_J / 3.6e9:.4g} MWh",
        f"end-to-end efficiency {_num(s.end_to_end_efficiency.energy_ratio):.3e}",
        f"outages {s.outages.count}, longest {s.outages.longest_s / 3600:.1f} h",
        f"diurnal gap {math.degrees(s.diurnal_aliasing.hour_angle_max_gap_rad):.2f} deg"
        + (" (ALIASING WARNING)" if s.diurnal_aliasing.warning else ""),
    ]
    axs[2, 2].text(0.0, 1.0, "\n".join(text), va="top", fontsize=10, family="monospace")
    _power_panels(rd, axs[3, 0], axs[3, 1])
    axs[3, 2].axis("off")
    axs[3, 2].text(
        0.0, 1.0, UPPER_BOUND_NOTE + ".\n" + rd.idealised_note, va="top", fontsize=7, wrap=True
    )
    fig.suptitle(rd.label, fontsize=9)
    fig.tight_layout()
    paths.append(_save(fig, out_dir / "composite.png"))
    return tuple(paths)


CHARACTERIZATION_FIGURE_FILES: Final[tuple[str, ...]] = (
    "power_thermal_vs_distance.png",
    "capture_vs_pointing.png",
    "capture_vs_wavefront.png",
)
"""Files written by :func:`make_characterization_figures` into ``figures/``."""


def make_characterization_figures(out_dir: Path | str) -> tuple[Path, ...]:
    """Draw the ``helioswarm characterize`` figures from the saved Parquet files only.

    Reads ``power_thermal_vs_distance.parquet``, ``capture_vs_pointing.parquet``,
    ``capture_vs_wavefront.parquet``, ``characterization.json`` and ``config_resolved.yaml``
    of ``out_dir`` (PDF §13.4: figures from saved data, stating their configuration).
    """
    import pyarrow.parquet as pq

    from helioswarm.characterize import (
        CAPTURE_POINTING_FILE,
        CAPTURE_WAVEFRONT_FILE,
        METADATA_FILE,
        POWER_THERMAL_FILE,
    )
    from helioswarm.io import read_json

    d = Path(out_dir)
    meta = read_json(d / METADATA_FILE)
    cfg = load_config(d / "config_resolved.yaml")
    geo = meta["link_geometry"]
    label = (
        f"{cfg.project.name} | config {meta['config_hash'][:12]} | from saved characterization"
        f" data | {meta['collector_condition']}"
    )
    paths: list[Path] = []

    def cols(name: str) -> dict[str, np.ndarray[Any, Any]]:
        t = pq.read_table(d / name)
        return {c: np.asarray(t.column(c).to_numpy(zero_copy_only=False)) for c in t.column_names}

    pt = cols(POWER_THERMAL_FILE)
    fig = Figure(figsize=(13, 7.5))
    axs = fig.subplots(2, 3)
    for eta in np.unique(pt["eta_electrical_to_beam"]):
        m = pt["eta_electrical_to_beam"] == eta
        r_au = pt["sun_distance_m"][m] / AU_M
        tag = f"eta_e2b {eta:.3g}"
        axs[0, 0].plot(r_au, pt["flux_W_m2"][m], label=tag)
        axs[0, 1].plot(r_au, pt["pv_electric_W"][m] / 1e6, label=f"P_PV, {tag}")
        axs[0, 1].plot(r_au, pt["dc_to_tx_W"][m] / 1e6, ls="--", label=f"P_dc_to_tx, {tag}")
        axs[0, 2].plot(r_au, pt["pv_efficiency"][m], label=tag)
        axs[1, 0].plot(r_au, pt["pv_temperature_K"][m], label=tag)
        axs[1, 1].plot(r_au, pt["waste_heat_W"][m] / 1e6, label=f"Q_waste, {tag}")
        axs[1, 2].plot(r_au, pt["radiator_area_m2"][m], label=tag)
        bad = m & ~pt["thermal_feasible"].astype(bool)
        axs[1, 0].plot(pt["sun_distance_m"][bad] / AU_M, pt["pv_temperature_K"][bad], "rx", ms=5)
    axs[1, 0].axhline(cfg.collector.max_operating_temperature_K, color="r", lw=0.8)
    titles = (
        ("Solar flux S(r) [W/m^2]", "Collector power [MW]", "PV efficiency eta_PV(T)"),
        (
            "PV temperature [K] (x: thermally infeasible; red: limit)",
            "Radiator waste heat [MW]",
            "Required radiator area [m^2]",
        ),
    )
    for i in range(2):
        for j in range(3):
            axs[i, j].set_title(titles[i][j], fontsize=8)
            axs[i, j].set_xlabel("heliocentric distance [au]")
            axs[i, j].legend(fontsize=6)
    fig.suptitle(label + "\nPDF §8.4 curves\n" + idealised_note(cfg, width=230), fontsize=6)
    fig.tight_layout()
    paths.append(_save(fig, d / "figures" / CHARACTERIZATION_FIGURE_FILES[0]))

    cp = cols(CAPTURE_POINTING_FILE)
    fig = Figure(figsize=(11, 4.5))
    ax_c, ax_p = fig.subplots(1, 2)
    for a in np.unique(cp["receiver_radius_m"]):
        m = cp["receiver_radius_m"] == a
        x = cp["pointing_sigma_over_diffraction"][m]
        ax_c.plot(x, cp["capture"][m], "o-", ms=3, label=f"receiver radius {a:.4g} m")
        ax_c.fill_between(x, cp["capture_p05"][m], cp["capture_p95"][m], alpha=0.15)
        ax_p.plot(x, cp["pointing_efficiency"][m], "o-", ms=3, label=f"{a:.4g} m")
    ax_c.set_xlabel("pointing sigma / (lambda/D)")
    ax_c.set_ylabel("expected capture (band: p05-p95)")
    ax_c.set_yscale("log")
    ax_c.legend(fontsize=6)
    ax_p.set_xlabel("pointing sigma / (lambda/D)")
    ax_p.set_ylabel("E[capture] / capture(perfect pointing)")
    ax_p.legend(fontsize=6)
    fig.suptitle(
        f"{label}\nPDF §9.5: capture vs pointing sigma and receiver radius at R = "
        f"{geo['range_m'] / AU_M:.4g} au, elevation {math.degrees(geo['elevation_rad']):.3g} deg, "
        f"N_F = {geo['fresnel_number']:.3g}; {UPPER_BOUND_NOTE}\n" + idealised_note(cfg, width=200),
        fontsize=5,
    )
    fig.tight_layout()
    paths.append(_save(fig, d / "figures" / CHARACTERIZATION_FIGURE_FILES[1]))

    cw = cols(CAPTURE_WAVEFRONT_FILE)
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots(1, 1)
    ax.plot(cw["phase_rms_rad"], cw["capture"], "o-", ms=3, label="expected capture")
    ax2 = ax.twinx()
    ax2.plot(cw["phase_rms_rad"], cw["wavefront_efficiency"], "s--", ms=3, color="0.5")
    ax2.set_ylabel("Ruze/Strehl factor exp(-phi_rms^2)", fontsize=7)
    ax.set_xlabel("RMS phase error 2 pi sigma_w / lambda [rad]")
    ax.set_ylabel("expected capture")
    ax.legend(fontsize=6)
    fig.suptitle(label + "\nCapture vs wavefront (phase) error", fontsize=5)
    fig.tight_layout()
    paths.append(_save(fig, d / "figures" / CHARACTERIZATION_FIGURE_FILES[2]))
    return tuple(paths)


CONSTELLATION_FIGURE_FILES: Final[tuple[str, ...]] = (
    "constellation_power.png",
    "constellation_counts.png",
    "constellation_unavailability.png",
    "constellation_satellites.png",
)
"""Files written by :func:`make_constellation_figures` into ``figures/``."""


def make_constellation_figures(run_dir: Path | str) -> tuple[Path, ...]:
    """Figures of a constellation run, drawn only from its saved tables (PDF §13.4).

    ``constellation_power.png`` (network and per-receiver ``P_grid(t)``),
    ``constellation_counts.png`` (active satellites, valid and scheduled links),
    ``constellation_unavailability.png`` (satellite-time fractions by primary reason) and
    ``constellation_satellites.png`` (delivered energy per satellite versus its Earth-relative
    ecliptic longitude at the element epoch).
    """
    import pyarrow.parquet as pq

    from helioswarm.constellation_io import (
        NETWORK_FILE,
        REASON_NAMES,
        RECEIVERS_FILE,
        SATELLITES_FILE,
        read_constellation_summary,
    )
    from helioswarm.io import CONFIG_FILE, SUMMARY_FILE

    d = Path(run_dir)
    net = pq.read_table(d / NETWORK_FILE)
    rec = pq.read_table(d / RECEIVERS_FILE)
    sats = pq.read_table(d / SATELLITES_FILE)
    summary = read_constellation_summary(d / SUMMARY_FILE)
    config = load_config(d / CONFIG_FILE)

    def col(table: Any, name: str) -> np.ndarray[Any, Any]:
        return np.asarray(table.column(name).to_numpy(zero_copy_only=False))

    days = col(net, "elapsed_s") / 86400.0
    label = (
        f"{config.project.name} | config {summary.config_hash[:12]} | {summary.n_satellites} "
        f"satellites, {summary.n_sites} sites, scheduler {summary.scheduler.policy} | step "
        f"{config.time.output_step_s:g} s | from saved run data"
    )
    note = idealised_note(config) + "\nIndependent satellites: received powers add "
    note += "incoherently (no coherent combining)."
    out = d / "figures"
    paths: list[Path] = []

    fig = Figure(figsize=(11, 4.5))
    ax = fig.subplots(1, 1)
    n_s = summary.n_sites
    p_site = col(rec, "grid_W").reshape(-1, n_s)
    for j in range(n_s):
        ax.plot(days, p_site[:, j], lw=0.6, label=f"site {summary.receivers[j].site_name}")
    ax.plot(days, col(net, "grid_W"), lw=0.8, color="k", label="network")
    ax.set_xlabel("days since start")
    ax.set_ylabel("P_grid [W]")
    ax.legend(fontsize=7)
    fig.suptitle(label + "\n" + note, fontsize=6)
    fig.tight_layout()
    paths.append(_save(fig, out / CONSTELLATION_FIGURE_FILES[0]))

    fig = Figure(figsize=(11, 4.5))
    ax = fig.subplots(1, 1)
    ax.plot(days, col(net, "valid_link_count"), lw=0.6, label="valid (feasible) links")
    ax.plot(days, col(net, "satellites_with_valid_link"), lw=0.6, label="satellites with a link")
    ax.plot(days, col(net, "active_satellite_count"), lw=0.8, label="active (scheduled)")
    ax.set_xlabel("days since start")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    fig.suptitle(label, fontsize=6)
    fig.tight_layout()
    paths.append(_save(fig, out / CONSTELLATION_FIGURE_FILES[1]))

    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots(1, 1)
    frac = summary.unavailability.satellite_time_fraction_primary
    names = ["DELIVERING", *REASON_NAMES]
    vals = [summary.unavailability.satellite_time_fraction_delivering] + [
        frac[n] for n in REASON_NAMES
    ]
    ax.barh(names, vals, color="0.4")
    ax.set_xlabel("fraction of satellite-time")
    ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(label + "\nprimary reason per satellite-epoch", fontsize=6)
    fig.tight_layout()
    paths.append(_save(fig, out / CONSTELLATION_FIGURE_FILES[2]))

    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots(1, 1)
    lon = col(sats, "earth_relative_longitude_deg")
    ax.plot(lon, col(sats, "delivered_energy_J") / 3.6e9, "o", ms=3)
    ax.set_xlabel("Earth-relative ecliptic longitude at the element epoch [deg]")
    ax.set_ylabel("delivered energy (pro-rata share) [MWh]")
    fig.suptitle(label, fontsize=6)
    fig.tight_layout()
    paths.append(_save(fig, out / CONSTELLATION_FIGURE_FILES[3]))
    return tuple(paths)


__all__ = [
    "CHARACTERIZATION_FIGURE_FILES",
    "CONSTELLATION_FIGURE_FILES",
    "FIGURE_FILES",
    "UPPER_BOUND_NOTE",
    "idealised_note",
    "make_characterization_figures",
    "make_constellation_figures",
    "make_figures",
]
