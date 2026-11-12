"""One-page run report (``report.md``) from the saved run files (PDF §11.1 item 8, §13.4).

The report lists the configuration's key parameters with their provenance labels (measured /
sourced / assumed / swept, PDF §13.4), the headline metrics with their definitions, the
provenance of code and data (from ``manifest.json``) and links to the figures, including the
composite page ``figures/composite.png``. It reads only ``config_resolved.yaml``,
``summary.json`` and ``manifest.json``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Final

from helioswarm.config import load_config, provenance_table
from helioswarm.plotting import FIGURE_FILES, UPPER_BOUND_NOTE

KEY_PARAMETERS: Final[tuple[str, ...]] = (
    "time.start_utc",
    "time.duration_days",
    "time.output_step_s",
    "orbit.model",
    "orbit.semimajor_axis_au",
    "orbit.eccentricity",
    "orbit.inclination_deg",
    "orbit.true_anomaly_deg",
    "collector.area_m2",
    "collector.pv_efficiency_reference",
    "collector.absorptivity",
    "collector.max_operating_temperature_K",
    "transmitter.mode",
    "transmitter.wavelength_m",
    "transmitter.aperture_diameter_m",
    "transmitter.aperture_illumination",
    "transmitter.power_conditioning_efficiency",
    "transmitter.electrical_to_beam_efficiency",
    "transmitter.optics_or_rf_efficiency",
    "transmitter.pointing_sigma_rad",
    "transmitter.pointing_bias_rad",
    "transmitter.rms_wavefront_error_m",
    "receiver.sites[0].latitude_deg",
    "receiver.sites[0].longitude_deg",
    "receiver.technology",
    "receiver.radius_m",
    "receiver.orientation",
    "receiver.min_elevation_deg",
    "receiver.conversion_efficiency",
    "receiver.grid_efficiency",
    "receiver.max_input_power_W",
    "receiver.max_power_density_W_m2",
    "thermal.radiator_temperature_limit_K",
    "models.beam",
    "models.atmosphere",
    "models.weather_scenario",
    "constants.solar_constant_1au_W_m2",
)
"""Configuration leaves listed in the report."""


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == 0.0 or not math.isfinite(v):
            return str(v)
        return f"{v:.6g}"
    return str(v)


def report_markdown(run_dir: Path) -> str:
    """Markdown text of the one-page report of ``run_dir``."""
    from helioswarm.io import CONFIG_FILE, MANIFEST_FILE, SUMMARY_FILE, read_json

    cfg = load_config(run_dir / CONFIG_FILE)
    s = read_json(run_dir / SUMMARY_FILE)
    m = read_json(run_dir / MANIFEST_FILE)
    rows = {r.path: r for r in provenance_table(cfg)}
    lines = [
        f"# Run report: {cfg.project.name}",
        "",
        f"Configuration hash `{s['config_hash']}` (schema {m['config_schema_version']}); "
        f"helioswarm {m['helioswarm_version']}, git `{m['git']['commit']}` "
        f"(dirty: {m['git']['dirty']}); backend `{m['backend']}`; seed {m['seed']} "
        f"(generator used: {not m['rng_state_unchanged']}). Generated from the saved run files "
        "(config_resolved.yaml, summary.json, manifest.json).",
        "",
        f"> {UPPER_BOUND_NOTE}.",
        "",
        "## Key parameters",
        "",
        "| Parameter | Value | Unit | Provenance |",
        "|---|---|---|---|",
    ]
    for key in KEY_PARAMETERS:
        r = rows.get(key)
        if r is None:
            continue
        lines.append(f"| `{key}` | {_fmt(r.value)} | {r.unit} | {r.provenance} |")
    pg = s["p_grid_W"]
    cf = s["capacity_factor"]
    den = s["capacity_factor_denominator"]
    e2e = s["end_to_end_efficiency"]
    out = s["outages"]
    al = s["diurnal_aliasing"]
    cap = s["beam_capture"]["eta_capture_when_available"]
    dens = s["receiver_power_density_W_m2"]["incident_peak_when_transmitting"]
    lines += [
        "",
        "## Metrics (definitions in summary.json and docs/model_assumptions.md, M6)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| P_grid time mean | {_fmt(pg['time_mean'])} W |",
        f"| P_grid median / p05 / p95 / peak | {_fmt(pg['median'])} / {_fmt(pg['p05'])} / "
        f"{_fmt(pg['p95'])} / {_fmt(pg['peak'])} W |",
        f"| P_grid mean when available | {_fmt(pg['mean_when_available'])} W |",
        f"| Delivered energy | {_fmt(s['delivered_energy_J'])} J |",
        f"| Availability | {_fmt(s['availability'])} |",
        f"| Capacity factor | {_fmt(cf)} (rated {_fmt(den['rated_power_W'])} W) |",
        f"| End-to-end efficiency E_grid/E_incident | {_fmt(e2e['energy_ratio'])} |",
        f"| Expected capture when available (median) | {_fmt(cap['median'])} |",
        f"| Peak incident power density when transmitting (max) | {_fmt(dens['max'])} W/m² |",
        f"| Outages: count / longest / median | {out['count']} / {_fmt(out['longest_s'])} s / "
        f"{_fmt(out['median_s'])} s (±{_fmt(out['resolution_s'])} s; {out['censored_count']} "
        "censored) |",
        f"| Receiver saturation time fraction | {_fmt(s['receiver_saturation_time_fraction'])} |",
        f"| Thermal feasible time fraction | {_fmt(s['thermal']['feasible_time_fraction'])} |",
        f"| Diurnal hour-angle gap | {_fmt(math.degrees(al['hour_angle_max_gap_rad']))} deg "
        f"({_fmt(al['effective_distinct_diurnal_phases'])} effective phases; aliasing warning: "
        f"{al['warning']}) |",
        "",
        f"Capacity-factor denominator: {den['definition']}.",
        "",
        "## Unavailability (time fraction by primary reason)",
        "",
        "| Reason | Primary | Flagged |",
        "|---|---|---|",
    ]
    for name, v in s["unavailability"]["blocking"].items():
        if v["time_fraction_flagged"] > 0.0:
            lines.append(
                f"| {name.lower()} | {_fmt(v['time_fraction_primary'])} | "
                f"{_fmt(v['time_fraction_flagged'])} |"
            )
    kernels = ", ".join(f"{k['name']} (sha256 {k['sha256'][:12]}…)" for k in m["kernels"])
    job = m["slurm"].get("SLURM_JOB_ID")
    lines += [
        "",
        "## Provenance",
        "",
        f"- Kernels: {kernels}",
        f"- ITU-R: {', '.join(m['itu_r_versions']) or 'not used'}",
        f"- Host {m['hostname']}, CPU {m['cpu_model']}, Slurm job {job}",
        f"- Wall clock {m['wall_clock_start_utc']} → {m['wall_clock_end_utc']}",
        "",
        "## Figures",
        "",
        "![composite](figures/composite.png)",
        "",
    ]
    lines += [f"- [{name}](figures/{name})" for name in FIGURE_FILES]
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path | str) -> Path:
    """Write ``report.md`` into ``run_dir`` (from saved files only)."""
    p = Path(run_dir)
    path = p / "report.md"
    path.write_text(report_markdown(p), encoding="utf-8")
    return path


__all__ = ["KEY_PARAMETERS", "report_markdown", "write_report"]
