"""M8 review check: is the Airy far field the high-fidelity answer for the re-evaluated designs?

For a saved single-satellite run directory (``config_resolved.yaml`` + ``timeseries.parquet``
of an optical design with a uniformly illuminated aperture), take the epoch with the largest
Fresnel number ``N_F = D^2/(lambda R)`` and compare, at that range, the centred encircled
energy at the receiver radius and the peak irradiance of the Fresnel-Hankel pattern
(:class:`helioswarm.aperture.FresnelHankelPattern`, Goodman eq. 4-17, M5-validated) with the
Airy pattern (:class:`helioswarm.optical.AiryPattern`). Prints one line per run directory.

    uv run python scripts/m8_fresnel_vs_airy_check.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from helioswarm.aperture import ApertureField, FresnelHankelPattern
from helioswarm.optical import AiryPattern


def check(run_dir: Path) -> str:
    """One comparison line for ``run_dir`` (module docstring)."""
    cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text(encoding="utf-8"))
    tx = cfg["transmitter"]
    if tx["aperture_illumination"] != "uniform_circular":
        return f"{run_dir.name}: not a uniform aperture; skipped"
    t = pq.read_table(run_dir / "timeseries.parquet")
    nf = np.asarray(t.column("fresnel_number").to_numpy(zero_copy_only=False), dtype=float)
    rng = np.asarray(t.column("range_m").to_numpy(zero_copy_only=False), dtype=float)
    k = int(np.nanargmax(nf))
    range_m = float(rng[k])
    d = float(tx["aperture_diameter_m"])
    lam = float(tx["wavelength_m"])
    a = float(cfg["receiver"]["radius_m"])
    fresnel = FresnelHankelPattern(
        ApertureField("uniform_circular", d), lam, range_m, paraxial_rho_max_m=a
    )
    airy = AiryPattern(
        aperture_diameter_m=d,
        wavelength_m=lam,
        range_m=range_m,
        max_fresnel_number=float(cfg["models"]["fraunhofer_max_fresnel_number"]),
    )
    ee_f = float(fresnel.encircled_energy(np.array([a]))[0])
    ee_a = float(airy.encircled_energy(np.array([a]))[0])
    peak = fresnel.peak_intensity_m2_inv() / airy.peak_intensity_m2_inv()
    return (
        f"{run_dir.name}: N_F max {nf[k]:.4g} at R = {range_m:.4g} m (D = {d:.4g} m, "
        f"lambda = {lam:.4g} m, a = {a:.5g} m): EE Fresnel {ee_f:.8f}, Airy {ee_a:.8f}, "
        f"relative {(ee_f - ee_a) / ee_a:.3e}; peak irradiance Fresnel/Airy {peak:.6f}"
    )


def main(argv: list[str]) -> int:
    for arg in argv[1:]:
        print(check(Path(arg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
