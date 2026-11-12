"""Solar radiation pressure (SRP), simple radial model (PDF §6.2-6.3, propagation level L3 item
used optionally by L1).

Model (PDF §6.2)::

    P_srp(r) = P_srp_1AU * (AU / r_sun)^2,        P_srp_1AU = S_1AU / c
    a_srp    = P_srp(r) * Cr * (A/m) * s_hat_away_from_sun

with ``r_sun = |r_sat - r_sun|`` the distance from the Sun's centre, ``s_hat`` the unit vector
from the Sun to the satellite, ``S_1AU`` the configured total solar irradiance at 1 au
(``constants.solar_constant_1au_W_m2``), ``c`` the speed of light, ``Cr`` the radiation-pressure
coefficient and ``A/m`` the Sun-facing area-to-mass ratio.

Approximation (named, per the engineering rules): this is the *cannonball* model, equivalently
a flat plate kept normal to the Sun line. ``Cr = 1`` is a perfect absorber, ``Cr = 2`` a perfect
specular reflector at normal incidence (Montenbruck & Gill, *Satellite Orbits*, 2000, §3.4;
Vallado, 4th ed., §8.6.4). Validity and what is NOT modelled:

* the force is purely radial (no attitude-dependent area or off-normal reflection; that is the
  L3 item of PDF §6.2);
* no shadowing (planetary eclipses are not applied to SRP; for heliocentric collectors they
  are rare and short; M3 computes the illumination fraction);
* the Sun is a point source at its centre (no finite-disk or limb-darkening correction,
  negligible beyond a few solar radii), and S_1AU is constant (no solar-cycle variation);
* Poynting-Robertson drag and thermal re-emission are omitted (velocity term ~ v/c ~ 1e-4 of
  the radial force).

Because ``a_srp`` is radial and ``proportional to 1/r^2``, it is conservative and equivalent to
reducing the Sun's GM: ``mu_eff = GM_sun - k`` with ``k = P_srp_1AU AU^2 Cr A/m`` (used as an
exact closed-form check of the propagator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from helioswarm.constants import AU_M, SPEED_OF_LIGHT_M_S

Float64Array = NDArray[np.float64]


def srp_pressure_1au_N_m2(solar_constant_1au_W_m2: float) -> float:
    """Solar radiation pressure at 1 au on a perfect absorber, ``P = S_1AU / c`` [N m^-2].

    For S_1AU = 1361 W m^-2 this is 4.54e-6 N m^-2.
    """
    if not (math.isfinite(solar_constant_1au_W_m2) and solar_constant_1au_W_m2 > 0.0):
        raise ValueError("solar_constant_1au_W_m2 must be finite and > 0")
    return solar_constant_1au_W_m2 / SPEED_OF_LIGHT_M_S


@dataclass(frozen=True)
class SrpModel:
    """Radial (cannonball / Sun-facing flat plate) SRP model.

    Attributes:
        cr: Radiation-pressure coefficient C_r in [1, 2] (1 absorbing, 2 specular normal).
        area_to_mass_m2_kg: Sun-facing area to mass ratio A/m [m^2 kg^-1].
        pressure_1au_N_m2: P_srp_1AU = S_1AU / c [N m^-2].
    """

    cr: float
    area_to_mass_m2_kg: float
    pressure_1au_N_m2: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.cr) and 1.0 <= self.cr <= 2.0):
            raise ValueError(f"cr must be in [1, 2], got {self.cr}")
        if not (math.isfinite(self.area_to_mass_m2_kg) and self.area_to_mass_m2_kg > 0.0):
            raise ValueError("area_to_mass_m2_kg must be finite and > 0")
        if not (math.isfinite(self.pressure_1au_N_m2) and self.pressure_1au_N_m2 > 0.0):
            raise ValueError("pressure_1au_N_m2 must be finite and > 0")

    @classmethod
    def from_solar_constant(
        cls, *, cr: float, area_to_mass_m2_kg: float, solar_constant_1au_W_m2: float
    ) -> SrpModel:
        """Build the model with ``P_srp_1AU = S_1AU / c``."""
        return cls(
            cr=cr,
            area_to_mass_m2_kg=area_to_mass_m2_kg,
            pressure_1au_N_m2=srp_pressure_1au_N_m2(solar_constant_1au_W_m2),
        )

    @property
    def strength_m3_s2(self) -> float:
        """``k = P_srp_1AU AU^2 Cr A/m`` [m^3 s^-2], so that ``|a_srp| = k / r_sun^2``."""
        return self.pressure_1au_N_m2 * AU_M * AU_M * self.cr * self.area_to_mass_m2_kg

    def lightness_number(self, gm_sun_m3_s2: float) -> float:
        """Ratio of SRP to solar gravity, ``beta = k / GM_sun`` [1] (independent of r)."""
        return self.strength_m3_s2 / gm_sun_m3_s2

    def acceleration_m_s2(self, position_m: ArrayLike, sun_position_m: ArrayLike) -> Float64Array:
        """SRP acceleration [m s^-2] (see :func:`srp_acceleration_m_s2`)."""
        return srp_acceleration_m_s2(position_m, sun_position_m, self.strength_m3_s2)


def srp_acceleration_m_s2(
    position_m: ArrayLike, sun_position_m: ArrayLike, strength_m3_s2: float
) -> Float64Array:
    """Radial SRP acceleration ``a = k (r - r_sun) / |r - r_sun|^3`` [m s^-2].

    Equivalent to ``P_srp_1AU (AU/d)^2 Cr (A/m) s_hat`` with ``d = |r - r_sun|`` and ``s_hat``
    pointing from the Sun to the satellite (PDF §6.2).

    Args:
        position_m: Satellite position(s), shape (3,) or (N, 3) [m].
        sun_position_m: Sun position(s), same frame and origin, broadcastable [m].
        strength_m3_s2: ``k = P_srp_1AU AU^2 Cr A/m`` (:attr:`SrpModel.strength_m3_s2`).

    Returns:
        Acceleration with the broadcast shape of the inputs.
    """
    d = np.asarray(position_m, dtype=np.float64) - np.asarray(sun_position_m, dtype=np.float64)
    if d.shape[-1:] != (3,):
        raise ValueError(f"positions must have a trailing dimension of 3, got {d.shape}")
    dist = np.sqrt(np.sum(d * d, axis=-1, keepdims=True))
    if np.any(dist <= 0.0):
        raise ValueError("satellite at the Sun's centre: SRP direction undefined")
    return np.asarray(strength_m3_s2 * d / dist**3, dtype=np.float64)


def srp_along_track_drift_estimate_m(
    lightness_number: float, *, semimajor_axis_m: float, duration_s: float, gm_sun_m3_s2: float
) -> float:
    """Closed-form along-track SRP displacement ``Delta s ~ 2 beta n t a`` [m] (PDF §6.3).

    For a near-circular orbit initialised with the *gravitational* circular speed
    ``v^2 = GM/a``, radial SRP turns ``mu`` into ``mu (1 - beta)``; vis-viva then gives
    ``1/a' = (1 - 2 beta) / ((1 - beta) a)``, i.e. ``a' ~ a (1 + beta)`` and
    ``n' ~ n (1 - 2 beta)``, so the along-track separation from the SRP-free orbit grows as
    ``2 beta n t a`` (valid for ``beta << 1``, ``e << 1`` and ``2 beta n t << 1``; verified
    against the L1 integration of the MVP orbit to 0.1 %, tests/unit/test_l1_propagation.py).

    Args:
        lightness_number: ``beta = k / GM_sun`` (:meth:`SrpModel.lightness_number`).
        semimajor_axis_m: Orbit semimajor axis a [m].
        duration_s: Elapsed time t [s].
        gm_sun_m3_s2: GM of the Sun [m^3 s^-2].
    """
    for name, v in (
        ("semimajor_axis_m", semimajor_axis_m),
        ("gm_sun_m3_s2", gm_sun_m3_s2),
    ):
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be finite and > 0, got {v}")
    if not (math.isfinite(lightness_number) and lightness_number >= 0.0):
        raise ValueError(f"lightness_number must be finite and >= 0, got {lightness_number}")
    if not (math.isfinite(duration_s) and duration_s >= 0.0):
        raise ValueError(f"duration_s must be finite and >= 0, got {duration_s}")
    n = math.sqrt(gm_sun_m3_s2 / semimajor_axis_m**3)
    return 2.0 * lightness_number * n * duration_s * semimajor_axis_m


def srp_material_area_to_mass_m2_kg(
    threshold_m: float,
    *,
    cr: float,
    solar_constant_1au_W_m2: float,
    semimajor_axis_m: float,
    duration_s: float,
    gm_sun_m3_s2: float,
) -> float:
    """Smallest ``A/m`` [m^2 kg^-1] whose SRP along-track drift reaches ``threshold_m``.

    Inverts :func:`srp_along_track_drift_estimate_m` with ``beta = P_srp_1AU AU^2 Cr (A/m) /
    GM_sun`` (PDF §6.3 "report the A/m range for which SRP materially changes the orbit"):
    SRP is material for every ``A/m`` above the returned value.
    """
    if not (math.isfinite(threshold_m) and threshold_m > 0.0):
        raise ValueError(f"threshold_m must be finite and > 0, got {threshold_m}")
    if not (math.isfinite(duration_s) and duration_s > 0.0):
        raise ValueError(f"duration_s must be finite and > 0, got {duration_s}")
    unit = SrpModel.from_solar_constant(
        cr=cr, area_to_mass_m2_kg=1.0, solar_constant_1au_W_m2=solar_constant_1au_W_m2
    )
    drift_per_unit_am = srp_along_track_drift_estimate_m(
        unit.lightness_number(gm_sun_m3_s2),
        semimajor_axis_m=semimajor_axis_m,
        duration_s=duration_s,
        gm_sun_m3_s2=gm_sun_m3_s2,
    )
    return threshold_m / drift_per_unit_am


__all__ = [
    "SrpModel",
    "srp_acceleration_m_s2",
    "srp_along_track_drift_estimate_m",
    "srp_material_area_to_mass_m2_kg",
    "srp_pressure_1au_N_m2",
]
