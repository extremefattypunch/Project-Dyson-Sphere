"""M8 (moved from M8a): heliocentric vs GEO economics on the SAME economic boundary, side by side.

Reads the ``economics.json`` files written by ``helioswarm economics`` for the M7b comparison
sweeps (and their ERA5 re-runs) -- heliocentric points costed with configs/economics/{optimistic,
nominal,conservative}.yaml, GEO points with configs/economics/geo_{...}.yaml, which differ only in
the transport Delta-v model (patched conics vs the explicit, sourced LEO->GEO 4.7 km/s; checked by
tests/integration/test_m8_forward.py) -- and writes one table per pair: per sweep point the LCOE
(USD/kWh) and NPV of each scenario and the nominal Monte Carlo p05/p50/p95. Scenario ranges and
intervals only, never one "robust" LCOE (PDF §12.5, §17). Nothing is recomputed.

Ratios (heliocentric / GEO) are reported in two stated ways, never by picking one point by a
favourable scenario: (a) the best point of EACH architecture under EACH criterion separately
(optimistic, nominal, conservative LCOE and the nominal Monte Carlo median), and (b) the range
of the per-point ratio over the points costed in both architectures (same grid point, or the
same weather year for the ERA5 sweeps).

    python scripts/m8_helio_vs_geo_economics.py OUT_DIR LABEL=HELIO_ECON_DIR:GEO_ECON_DIR [...]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def _rows(path: Path, arch: str) -> list[dict[str, Any]]:
    doc = json.loads((path / "economics.json").read_text())
    by_point: dict[str, dict[str, Any]] = {}
    for r in doc["runs"]:
        point = Path(r["run_dir"]).name
        row = by_point.setdefault(
            point,
            {
                "architecture": arch,
                "point": point,
                "project": r["inputs"]["project_name"],
                "annual_energy_kWh": r["inputs"]["annual_energy_kWh"],
                "status": "costed",
            },
        )
        sc = r["scenario"].removeprefix("geo_")
        res = r["result"]
        row[f"lcoe_{sc}"] = res["lcoe_usd_per_kWh"]
        row[f"npv_{sc}"] = res["npv_usd"]
        row[f"mass_kg_{sc}"] = res["mass_total_kg"]
        row[f"delta_v_m_s_{sc}"] = sum(res["delta_v_m_s"]) / len(res["delta_v_m_s"])
        row[f"flags_{sc}"] = ",".join(res["flags"])
        mc = r.get("monte_carlo")
        if mc:
            p = mc["lcoe_percentiles_usd_per_kWh"]
            row[f"mc_{sc}_lcoe_p05"] = p.get("p05")
            row[f"mc_{sc}_lcoe_p50"] = p.get("p50")
            row[f"mc_{sc}_lcoe_p95"] = p.get("p95")
            row[f"mc_{sc}_n_undefined"] = mc["n_undefined_lcoe"]
    unc = doc.get("uncosted_points") or []
    for u in unc:
        by_point.setdefault(u["point"], {"architecture": arch, "point": u["point"]})["status"] = (
            f"not costed: {u['status']}"
        )
    return list(by_point.values())


CRITERIA = ("lcoe_optimistic", "lcoe_nominal", "lcoe_conservative", "mc_nominal_lcoe_p50")


def _ratios(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Heliocentric / GEO LCOE ratios per criterion (module docstring (a), (b))."""
    out: dict[str, dict[str, Any]] = {}
    for c in CRITERIA:
        val = {
            arch: {r["point"]: r[c] for r in rows if r["architecture"] == arch and r.get(c)}
            for arch in ("heliocentric", "GEO")
        }
        best = {
            a: min(v.items(), key=lambda kv: kv[1]) if v else (None, None) for a, v in val.items()
        }
        common = sorted(set(val["heliocentric"]) & set(val["GEO"]))
        same = [val["heliocentric"][p] / val["GEO"][p] for p in common]
        hb, gb = best["heliocentric"][1], best["GEO"][1]
        out[c] = {
            "helio_point": best["heliocentric"][0],
            "helio_best": hb,
            "geo_point": best["GEO"][0],
            "geo_best": gb,
            "best_ratio": hb / gb if hb and gb else None,
            "same_point_min": min(same) if same else None,
            "same_point_max": max(same) if same else None,
            "n_points": len(same),
        }
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "undefined"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main(argv: list[str]) -> int:
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    md = ["# Heliocentric vs GEO economics (same economic boundary; M8)", ""]
    for spec in argv[2:]:
        label, dirs = spec.split("=", 1)
        helio, geo = (Path(d) for d in dirs.split(":"))
        rows = sorted(_rows(helio, "heliocentric"), key=lambda r: r["point"]) + sorted(
            _rows(geo, "GEO"), key=lambda r: r["point"]
        )
        cols = [
            "architecture",
            "point",
            "annual_energy_kWh",
            "lcoe_optimistic",
            "lcoe_nominal",
            "lcoe_conservative",
            "mc_nominal_lcoe_p05",
            "mc_nominal_lcoe_p50",
            "mc_nominal_lcoe_p95",
            "npv_nominal",
            "mass_kg_nominal",
            "delta_v_m_s_nominal",
            "flags_nominal",
            "status",
        ]
        with (out / f"{label}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        md += [
            f"## {label}",
            "",
            f"helio: `{helio}`; GEO: `{geo}`",
            "",
            "| " + " | ".join(cols) + " |",
            "|---" * len(cols) + "|",
        ]
        md += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
        md.append("")
        ratios = _ratios(rows)
        md += [
            "Ratios heliocentric / GEO per criterion: (a) best point of each architecture "
            "selected separately under that criterion; (b) min-max of the ratio at the same "
            "point over the points costed in both.",
            "",
            "| criterion | best helio point | helio LCOE | best GEO point | GEO LCOE | (a) ratio "
            "| (b) same-point ratio min | max | points |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c, r in ratios.items():
            md.append(
                f"| {c} | {r['helio_point'] or '-'} | {_fmt(r['helio_best'])} | "
                f"{r['geo_point'] or '-'} | {_fmt(r['geo_best'])} | {_fmt(r['best_ratio'])} | "
                f"{_fmt(r['same_point_min'])} | {_fmt(r['same_point_max'])} | {r['n_points']} |"
            )
        md.append("")
        summary[label] = {"helio": str(helio), "geo": str(geo), "rows": rows, "ratios": ratios}
    (out / "helio_vs_geo_economics.md").write_text("\n".join(md) + "\n")
    (out / "helio_vs_geo_economics.json").write_text(json.dumps(summary, indent=1))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
