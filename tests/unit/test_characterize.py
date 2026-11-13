"""PDF §8.4 / §9.5 response curves (helioswarm.characterize) against closed forms.

References are written here independently of the package helpers:

* S(r) = S_1AU (au/r)^2 (PDF §8.1), P_abs = alpha A S, P_PV = P_abs eta_PV(T) with the PDF §8.2
  linear law evaluated here, P_dc_to_tx = eta_pc P_PV, Q_waste = P_PV - P_aperture + Q_other,
  A_rad = Q_waste / (eps sigma n F (T_lim^4 - T_space^4)) with sigma = 2 pi^5 k^4 / (15 h^3 c^2)
  from the exact 2019 SI constants (CODATA's 10-digit sigma differs by 3e-11 relative);
* centred Airy capture of a disk of radius a: 1 - J0(x)^2 - J1(x)^2, x = pi D a / (lambda R)
  (Born & Wolf §8.5.2), at zenith (projected radius = physical radius);
* the Ruze/Strehl factor exp(-(2 pi sigma_w / lambda)^2) multiplies the expected capture once.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.special import j0, j1

from helioswarm.characterize import (
    CAPTURE_POINTING_FILE,
    CAPTURE_WAVEFRONT_FILE,
    METADATA_FILE,
    POWER_THERMAL_FILE,
    LinkGeometry,
    capture_pointing_table,
    capture_wavefront_table,
    power_thermal_table,
)
from helioswarm.cli import main
from helioswarm.config import idealised_settings, load_config, parse_config
from helioswarm.constants import AU_M

from config_helpers import MVP_CONFIG, load_mvp_data, set_path

SIGMA_SI = 2 * math.pi**5 * 1.380649e-23**4 / (15 * 6.62607015e-34**3 * 299792458.0**2)
ZENITH = LinkGeometry(range_m=AU_M, elevation_rad=0.5 * math.pi)


def test_power_thermal_curves_against_closed_forms() -> None:
    cfg = load_config(MVP_CONFIG)
    r = np.array([0.7, 1.0, 1.2]) * AU_M
    t = power_thermal_table(cfg, r, electrical_to_beam_efficiencies=[0.25, 0.5]).to_pydict()
    assert t["eta_electrical_to_beam"] == [0.25] * 3 + [0.5] * 3
    c, tx, th = cfg.collector, cfg.transmitter, cfg.thermal
    s1 = cfg.constants.solar_constant_1au_W_m2
    for k in range(6):
        rk = t["sun_distance_m"][k]
        flux = s1 * (AU_M / rk) ** 2
        assert t["flux_W_m2"][k] == pytest.approx(flux, rel=1e-15)
        assert t["absorbed_W"][k] == pytest.approx(c.absorptivity * c.area_m2 * flux, rel=1e-14)
        temp = t["pv_temperature_K"][k]
        eta = c.pv_efficiency_reference * (
            1 + c.temperature_coefficient_per_K * (temp - c.reference_temperature_K)
        )
        eta = min(max(eta, c.pv_efficiency_min), c.pv_efficiency_max)
        assert t["pv_efficiency"][k] == pytest.approx(eta, rel=1e-14)
        p_pv = t["absorbed_W"][k] * eta
        assert t["pv_electric_W"][k] == pytest.approx(p_pv, rel=1e-14)
        e2b = t["eta_electrical_to_beam"][k]
        assert t["dc_to_tx_W"][k] == pytest.approx(
            tx.power_conditioning_efficiency * p_pv, rel=1e-14
        )
        p_ap = p_pv * tx.power_conditioning_efficiency * e2b * tx.optics_or_rf_efficiency
        q = p_pv - p_ap + th.other_heat_load_W
        assert t["waste_heat_W"][k] == pytest.approx(q, rel=1e-13)
        area = q / (
            th.emissivity * SIGMA_SI * th.radiating_sides * th.radiator_view_factor
            * (th.radiator_temperature_limit_K**4 - th.space_temperature_K**4)
        )  # fmt: skip
        assert t["radiator_area_m2"][k] == pytest.approx(area, rel=1e-9)
        assert t["thermal_feasible"][k] == (temp <= c.max_operating_temperature_K)
        # The returned PV temperature closes the M4 PV-plate balance independently:
        # alpha A S - P_PV = (eps_f + eps_b) sigma F A (T^4 - T_space^4). Tolerance 1e-10 of
        # P_abs: the solver's residual bound is 1e-12 P_abs (thermal.PV_BALANCE_RTOL) with
        # CODATA sigma; the exact-SI sigma used here differs by 3e-11 relative.
        radiated = (
            (th.pv_front_emissivity + th.pv_back_emissivity) * SIGMA_SI * th.pv_view_factor
            * c.area_m2 * (temp**4 - th.space_temperature_K**4)
        )  # fmt: skip
        absorbed = c.absorptivity * c.area_m2 * flux
        assert absorbed - p_pv == pytest.approx(radiated, abs=1e-10 * absorbed)
    # The operating point changes the waste heat but not the PV side (PV node is independent of
    # the radiator): same P_PV, Q_waste(0.25) - Q_waste(0.5) = P_dc (0.5 - 0.25) eta_o.
    np.testing.assert_array_equal(t["pv_electric_W"][:3], t["pv_electric_W"][3:])
    for k in range(3):
        dq = t["waste_heat_W"][k] - t["waste_heat_W"][k + 3]
        expect = t["dc_to_tx_W"][k] * 0.25 * tx.optics_or_rf_efficiency
        assert dq == pytest.approx(expect, rel=1e-12)
    # Default operating point is the configured one.
    one = power_thermal_table(cfg, [AU_M]).to_pydict()
    assert one["eta_electrical_to_beam"] == [tx.electrical_to_beam_efficiency]


def test_power_thermal_curves_with_an_undersized_fixed_radiator() -> None:
    # A fixed radiator area smaller than the required one: the radiator runs above T_lim, so
    # the point is infeasible although the PV plate is within its limit. Closed forms:
    # T_rad = (Q_waste / (eps sigma n F A) + T_space^4)^(1/4), margin = min(T_pv_lim - T_pv,
    # T_lim - T_rad) (thermal.ThermalResult).
    data = load_mvp_data()
    set_path(data, "thermal.radiator_area_m2", 100.0)
    cfg = parse_config(data)
    th, c = cfg.thermal, cfg.collector
    r = np.array([1.0, 1.2]) * AU_M
    t = power_thermal_table(cfg, r).to_pydict()
    for k in range(2):
        q = t["waste_heat_W"][k]
        t_rad = (
            q / (th.emissivity * SIGMA_SI * th.radiating_sides * th.radiator_view_factor * 100.0)
            + th.space_temperature_K**4
        ) ** 0.25
        assert t["radiator_temperature_K"][k] == pytest.approx(t_rad, rel=1e-10)
        assert t_rad > th.radiator_temperature_limit_K  # undersized (A_req ~ 4e2 m^2 at 1 au)
        assert t["pv_temperature_K"][k] < c.max_operating_temperature_K  # PV side feasible
        assert t["thermal_feasible"][k] is False
        margin = min(
            c.max_operating_temperature_K - t["pv_temperature_K"][k],
            th.radiator_temperature_limit_K - t_rad,
        )
        assert t["thermal_margin_K"][k] == pytest.approx(margin, rel=1e-9)
        assert t["thermal_margin_K"][k] < 0.0


def test_capture_curves_centred_airy_and_monotonicity() -> None:
    cfg = load_config(MVP_CONFIG)
    lam, d = cfg.transmitter.wavelength_m, cfg.transmitter.aperture_diameter_m
    radii = [500.0, 1000.0, 5000.0]
    sigmas = [0.0, 0.1 * lam / d, 0.3 * lam / d, 1.0 * lam / d]
    t = capture_pointing_table(cfg, ZENITH, sigmas, radii).to_pydict()
    assert len(t["capture"]) == 12
    for k in range(12):
        x = math.pi * d * t["receiver_radius_m"][k] / (lam * AU_M)
        # Perfect pointing (bias 0 in the MVP): centred encircled energy, any sigma.
        assert t["capture_nominal"][k] == pytest.approx(1 - j0(x) ** 2 - j1(x) ** 2, rel=1e-9)
        if t["pointing_sigma_rad"][k] == 0.0:
            assert t["capture"][k] == pytest.approx(t["capture_nominal"][k], rel=1e-12)
        assert t["projected_receiver_radius_m"][k] == t["receiver_radius_m"][k]  # zenith
        assert t["pointing_efficiency"][k] == pytest.approx(
            t["capture"][k] / t["capture_nominal"][k], rel=1e-12
        )
    cap = np.array(t["capture"]).reshape(3, 4)  # rows: receiver radius, columns: sigma
    # Larger sigma cannot improve the expected capture of a receiver inside the first Airy ring
    # (x <= 1.05 here, inside the first dark ring at x = 3.83; Anderson 1955 for a unimodal
    # pattern), strictly decreasing for sigma > 0.
    assert np.all(np.diff(cap, axis=1) < 0.0)
    # A larger concentric disk contains the smaller one: capture increases with radius at
    # every sigma.
    assert np.all(np.diff(cap, axis=0) > 0.0)


def test_capture_wavefront_curve_applies_the_ruze_factor_once() -> None:
    cfg = load_config(MVP_CONFIG)
    lam = cfg.transmitter.wavelength_m
    w = [0.0, 0.05 * lam, 0.1 * lam]
    t = capture_wavefront_table(cfg, ZENITH, w).to_pydict()
    for k in range(3):
        ruze = math.exp(-((2 * math.pi * w[k] / lam) ** 2))
        assert t["wavefront_efficiency"][k] == pytest.approx(ruze, rel=1e-15)
        assert t["capture"][k] == pytest.approx(t["capture"][0] * ruze, rel=1e-12)
    assert t["phase_rms_rad"][2] == pytest.approx(0.2 * math.pi, rel=1e-15)


def test_link_geometry_validation() -> None:
    with pytest.raises(ValueError, match="elevation"):
        LinkGeometry(range_m=AU_M, elevation_rad=0.0)
    with pytest.raises(ValueError, match="range"):
        LinkGeometry(range_m=-1.0, elevation_rad=1.0)


def test_cli_characterize_writes_tables_metadata_and_figures(tmp_path: Path) -> None:
    out = tmp_path / "char"
    argv = ["characterize", str(MVP_CONFIG), "--out", str(out), "--radii-au", "0.5", "1.0",
            "--receiver-radii-m", "1000", "--pointing-sigmas-rad", "0", "1e-8"]  # fmt: skip
    assert main(argv) == 0
    for name in (POWER_THERMAL_FILE, CAPTURE_POINTING_FILE, CAPTURE_WAVEFRONT_FILE,
                 METADATA_FILE, "config_resolved.yaml"):  # fmt: skip
        assert (out / name).is_file(), name
    assert len(list((out / "figures").glob("*.png"))) == 3
    meta = json.loads((out / METADATA_FILE).read_text())
    assert meta["link_geometry"]["range_m"] == AU_M
    assert (
        meta["config_hash"]
        == pq.read_table(out / POWER_THERMAL_FILE).schema.metadata[b"config_hash"].decode()
    )
    pt = pq.read_table(out / POWER_THERMAL_FILE).to_pydict()
    assert pt["sun_distance_m"] == [0.5 * AU_M, 1.0 * AU_M]
    cp = pq.read_table(out / CAPTURE_POINTING_FILE).to_pydict()
    # Requested sigmas plus the configured 1e-9 rad (receiver radius 1 km is the configured one).
    assert cp["pointing_sigma_rad"] == [0.0, 1e-9, 1e-8]
    first = (out / POWER_THERMAL_FILE).read_bytes()
    assert main(argv) == 1  # refuses to overwrite
    assert main([*argv, "--overwrite"]) == 0
    assert (out / POWER_THERMAL_FILE).read_bytes() == first  # deterministic
    # Figures are redrawn from the saved files only.
    from helioswarm.plotting import make_characterization_figures

    for f in (out / "figures").iterdir():
        f.unlink()
    assert len(make_characterization_figures(out)) == 3
    assert main(["characterize", str(MVP_CONFIG), "--out", str(tmp_path / "x"),
                 "--elevation-deg", "0"]) == 1  # fmt: skip


def test_characterization_records_and_labels_the_idealised_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # PDF §13.4: idealised upper bounds labelled in the figure. characterization.json lists the
    # configuration's idealised settings, and the power/thermal and capture figures name every
    # one of them (figure text captured before saving).
    import helioswarm.plotting as plotting_mod

    out = tmp_path / "char"
    argv = ["characterize", str(MVP_CONFIG), "--out", str(out), "--radii-au", "1.0",
            "--receiver-radii-m", "1000", "--pointing-sigmas-rad", "0"]  # fmt: skip
    assert main(argv) == 0
    ideal = idealised_settings(load_config(MVP_CONFIG))
    meta = json.loads((out / METADATA_FILE).read_text())
    assert [d["path"] for d in meta["idealised_settings"]] == [i.path for i in ideal]
    assert "models.weather_scenario" in {i.path for i in ideal}
    captured: dict[str, str] = {}
    real_save = plotting_mod._save

    def capture(fig, path):  # type: ignore[no-untyped-def]
        texts = [fig._suptitle.get_text()] if fig._suptitle is not None else []
        captured[path.name] = " ".join(" ".join(texts).split())  # undo the line wrapping
        return real_save(fig, path)

    monkeypatch.setattr(plotting_mod, "_save", capture)
    plotting_mod.make_characterization_figures(out)
    for name in ("power_thermal_vs_distance.png", "capture_vs_pointing.png"):
        for i in ideal:
            assert i.effect in captured[name], (name, i.path)
