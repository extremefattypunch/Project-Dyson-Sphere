"""Site weather availability of optical links from ERA5 cloud cover (M8a; PDF §10.2, §17).

``models.weather_source = "era5"`` replaces the deterministic clear/cloudy scenario by an
hourly, site-specific outage mask built from ERA5 total cloud cover ``tcc``
(:mod:`helioswarm.era5`). The deterministic physics (geometry, beam, clear-sky atmosphere) is
unchanged; weather only decides, per receiver site and arrival epoch, whether the optical
link is blocked (bit ``WEATHER``, ``eta_atmosphere = 0``).

Model (all choices are assumptions, ``docs/model_assumptions.md`` M8a):

1. **Time.** Weather is indexed by UTC. The arrival epoch at the receiver (TDB seconds past
   J2000, ``AccessResult.t_arrival_tdb_s``) is converted explicitly,
   ``UTC_posix = TDB - DeltaET(TDB) + 946728000`` with SPICE ``deltet`` (``DeltaET = ET - UTC``
   = TAI-UTC + 32.184 s + TDB-TT; 946728000 = POSIX time of 2000-01-01T12:00:00 UTC; checked
   against ``et2utc`` to 0 s in the tests). ``DeltaET`` is tabulated hourly over the run
   window and interpolated linearly (:class:`UtcOffsetTable`): exact to < 1 ms except within
   one hour of a leap second, where the 1 s step is smeared over that hour.
2. **Hour.** The UTC epoch is rounded to the nearest hour (ERA5 analyses are instantaneous
   at hh:00; an epoch at hh:30:00 rounds up).
3. **Year mapping.** The simulated UTC (month, day, hour) maps to the same calendar position
   in a historical year: ``fixed`` -- simulated calendar year ``Y0 + k`` (``Y0`` = UTC year of
   ``time.start_utc``) maps to the record year with index ``(historical_year - first + k) mod
   n``; ``sampled`` -- each simulated calendar year ``Y0 .. Y_end + 1`` maps to a record year
   drawn uniformly with the weather Generator (in that order). 29 February of a simulated
   leap year maps to 28 February of a non-leap historical year (that day's weather repeats).
4. **Outage.** ``threshold``: blocked iff ``tcc > cloud_cover_threshold`` (``tcc`` equal to
   the threshold is clear). ``bernoulli``: blocked iff ``u < tcc`` with ``u ~ U[0, 1)`` drawn
   once per (site, historical hour) -- clear-line-of-sight probability ``1 - tcc`` under the
   assumption that the cloud-covered fraction of the grid box is randomly placed relative to
   the line of sight; draws are independent between hours (no persistence: an approximation
   that breaks the real autocorrelation of cloud; the threshold model keeps it).
5. **Sites.** Each site uses its own grid node; all sites share the mapped historical hour,
   so the real spatial correlation of the record is kept (joint statistics in
   :func:`joint_statistics`).

Randomness: one dedicated ``numpy.random.Generator`` (:func:`weather_generator`, seeded from
``project.seed`` with a fixed spawn key) passed explicitly; it is consumed only by the
``sampled`` year mapping and the ``bernoulli`` draws, in a fixed order (years, then sites in
configuration order), so every process of a run (Slurm array tasks included) builds the
identical weather.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from helioswarm.atmosphere import AtmosphereResult
from helioswarm.config import RunConfig, SiteConfig, WeatherConfig, canonical_utc
from helioswarm.era5 import (
    ATTRIBUTION_TEMPLATE,
    DATASET_INFO,
    GRID_STEP_DEG,
    LICENCE_SPDX,
    Era5Error,
    Interpolation,
    grid_node,
    load_node_series,
    site_series,
    weather_dir,
)
from helioswarm.kernels import LoadedKernels

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

POSIX_J2000_UTC_S: Final[int] = 946_728_000
"""POSIX time of 2000-01-01T12:00:00 UTC (the zero of SPICE's 'UTC seconds past J2000')."""

WEATHER_STREAM_KEY: Final[int] = 0x5745  # "WE"
"""Spawn key of the weather Generator's SeedSequence (independent of the run Generator)."""

ERA5_VARIABLE: Final[str] = "total_cloud_cover"

WEATHER_FILE: Final[str] = "weather.json"
"""Run-directory file with the weather provenance (listed in the manifest outputs)."""


class WeatherError(RuntimeError):
    """Weather model error (configuration, data coverage)."""


def weather_generator(seed: int) -> np.random.Generator:
    """The weather model's Generator: ``default_rng(SeedSequence(seed, spawn_key=(0x5745,)))``.

    A separate stream so that weather draws never shift the run Generator (random phases).
    """
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(WEATHER_STREAM_KEY,)))


# ---------------------------------------------------------------------------------------------
# Time: TDB -> UTC (explicit) and calendar mapping
# ---------------------------------------------------------------------------------------------


def tdb_to_utc_posix_exact(tdb_s: Float64Array, kernels: LoadedKernels) -> Float64Array:
    """UTC as POSIX seconds of TDB epochs, one SPICE ``deltet`` call per epoch (reference).

    ``UTC_posix = TDB - DeltaET(TDB) + 946728000`` (module docstring, item 1).
    """
    import spiceypy

    kernels.require_active(kind="LSK")
    t = np.asarray(tdb_s, dtype=np.float64)
    out = np.empty_like(t)
    flat_in, flat_out = t.reshape(-1), out.reshape(-1)
    for k, et in enumerate(flat_in):
        flat_out[k] = et - float(spiceypy.deltet(float(et), "ET")) + POSIX_J2000_UTC_S
    return out


@dataclass(frozen=True, eq=False)
class UtcOffsetTable:
    """``DeltaET = TDB - UTC`` tabulated hourly over a window (linear interpolation).

    Attributes:
        tdb_s: Table epochs [TDB s past J2000], hourly, covering the window.
        delta_et_s: ``ET - UTC`` at those epochs [s] (SPICE ``deltet``).
    """

    tdb_s: Float64Array
    delta_et_s: Float64Array

    @classmethod
    def build(cls, tdb_min_s: float, tdb_max_s: float, kernels: LoadedKernels) -> UtcOffsetTable:
        """Table over ``[tdb_min - 1 h, tdb_max + 1 h]`` at a 1 h step."""
        import spiceypy

        kernels.require_active(kind="LSK")
        n = math.ceil((tdb_max_s - tdb_min_s) / 3600.0) + 3
        grid = tdb_min_s - 3600.0 + 3600.0 * np.arange(n, dtype=np.float64)
        delta = np.array([float(spiceypy.deltet(float(e), "ET")) for e in grid])
        return cls(tdb_s=grid, delta_et_s=delta)

    def utc_posix_s(self, tdb_s: Float64Array) -> Float64Array:
        """UTC POSIX seconds of TDB epochs inside the table window."""
        t = np.asarray(tdb_s, dtype=np.float64)
        if t.size and (np.min(t) < self.tdb_s[0] or np.max(t) > self.tdb_s[-1]):
            raise WeatherError("epoch outside the UTC offset table window")
        out: Float64Array = t - np.interp(t, self.tdb_s, self.delta_et_s) + POSIX_J2000_UTC_S
        return out


def nearest_hour_posix(utc_posix_s: Float64Array) -> Int64Array:
    """POSIX second of the nearest whole UTC hour (half hour rounds up)."""
    t = np.asarray(utc_posix_s, dtype=np.float64)
    out: Int64Array = (np.floor((t + 1800.0) / 3600.0) * 3600.0).astype(np.int64)
    return out


def calendar_fields(hour_posix_s: Int64Array) -> tuple[Int64Array, ...]:
    """(year, month, day, hour) of whole-hour POSIX times (proleptic Gregorian, UTC)."""
    h = np.asarray(hour_posix_s, dtype=np.int64).astype("datetime64[s]").astype("datetime64[h]")
    years = h.astype("datetime64[Y]")
    months = h.astype("datetime64[M]")
    days = h.astype("datetime64[D]")
    year = years.astype(np.int64) + 1970
    month = (months - years.astype("datetime64[M]")).astype(np.int64) + 1
    day = (days - months.astype("datetime64[D]")).astype(np.int64) + 1
    hour = (h - days.astype("datetime64[h]")).astype(np.int64)
    return year, month, day, hour


def _posix_from_fields(
    year: Int64Array, month: Int64Array, day: Int64Array, hour: Int64Array
) -> Int64Array:
    ym = (year - 1970) * 12 + (month - 1)
    d = ym.astype("datetime64[M]").astype("datetime64[D]") + (day - 1).astype("timedelta64[D]")
    t = d.astype("datetime64[h]") + hour.astype("timedelta64[h]")
    out: Int64Array = t.astype("datetime64[s]").astype(np.int64)
    return out


def year_map(
    settings: WeatherConfig,
    sim_years: Sequence[int],
    rng: np.random.Generator,
) -> dict[int, int]:
    """Historical year of each simulated calendar year (module docstring, item 3).

    ``sampled`` draws ``rng.integers(n)`` once per simulated year in ascending order.
    """
    first, last = settings.era5_first_year, settings.era5_last_year
    n = last - first + 1
    years = sorted(sim_years)
    if settings.year_mapping == "fixed":
        assert settings.historical_year is not None  # schema rule
        y0 = years[0]
        return {y: first + (settings.historical_year - first + (y - y0)) % n for y in years}
    draws = rng.integers(0, n, size=len(years))
    return {y: first + int(k) for y, k in zip(years, draws, strict=True)}


def historical_hour_posix(utc_posix_s: Float64Array, mapping: Mapping[int, int]) -> Int64Array:
    """POSIX time of the historical hour of each simulated UTC epoch (items 2-3)."""
    hour = nearest_hour_posix(utc_posix_s)
    y, mo, d, h = calendar_fields(hour)
    missing = sorted(set(np.unique(y).tolist()) - set(mapping))
    if missing:
        raise WeatherError(f"no historical year mapped for simulated year(s) {missing}")
    hy = np.array([mapping[int(v)] for v in y], dtype=np.int64)
    feb29 = (mo == 2) & (d == 29)
    not_leap = np.array([not calendar.isleap(int(v)) for v in hy], dtype=bool)
    d = np.where(feb29 & not_leap, 28, d)
    return _posix_from_fields(hy, mo, d, h)


def outage_from_cloud_cover(
    cloud_cover: Float64Array,
    settings: WeatherConfig,
    rng: np.random.Generator | None,
) -> BoolArray:
    """Hard optical outage per hour (item 4). ``rng`` is used by ``bernoulli`` only."""
    tcc = np.asarray(cloud_cover, dtype=np.float64)
    if settings.outage_model == "threshold":
        assert settings.cloud_cover_threshold is not None  # schema rule
        out: BoolArray = tcc > settings.cloud_cover_threshold
        return out
    if rng is None:
        raise WeatherError("the bernoulli outage model needs a Generator")
    u = rng.random(tcc.shape)
    res: BoolArray = u < tcc
    return res


# ---------------------------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SiteWeather:
    """Hourly ERA5 cloud cover and outage mask of one receiver site.

    Attributes:
        site_name: Receiver site.
        latitude_deg, longitude_deg: Site coordinates [deg].
        node_key: ERA5 grid node used.
        interpolation: ``nearest`` or ``bilinear``.
        start_posix_s: POSIX time of the first hour (1 Jan of the first record year, 00 UTC).
        cloud_cover: Hourly ``tcc`` at the site [1], shape (T,).
        outage: Hourly hard-outage flag, shape (T,).
        files: ERA5 files used.
        file_sha256: Their SHA-256.
        cds_request_ids: CDS request ids.
        retrieved_utc: Retrieval times.
        expver: ERA5 experiment versions in the files (``0001`` = final ERA5).
    """

    site_name: str
    latitude_deg: float
    longitude_deg: float
    node_key: str
    interpolation: Interpolation
    start_posix_s: int
    cloud_cover: Float64Array
    outage: BoolArray
    files: tuple[str, ...]
    file_sha256: tuple[str, ...]
    cds_request_ids: tuple[str | None, ...]
    retrieved_utc: tuple[str, ...]
    expver: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("cloud_cover", "outage"):
            arr = np.array(getattr(self, name), copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if self.cloud_cover.shape != self.outage.shape:
            raise ValueError("cloud_cover and outage must have the same shape")

    def index_of(self, hour_posix_s: Int64Array) -> Int64Array:
        """Hour indices of whole-hour POSIX times (must lie inside the record)."""
        idx = (np.asarray(hour_posix_s, dtype=np.int64) - self.start_posix_s) // 3600
        if idx.size and (int(np.min(idx)) < 0 or int(np.max(idx)) >= self.cloud_cover.size):
            raise WeatherError(f"{self.site_name}: historical hour outside the ERA5 record")
        out: Int64Array = idx.astype(np.int64)
        return out


@dataclass(frozen=True, eq=False)
class WeatherProvider:
    """ERA5 weather of every receiver site of a run (module docstring).

    Attributes:
        settings: The ``weather`` configuration section.
        sites: ``{site name: SiteWeather}``.
        year_mapping: ``{simulated year: historical year}``.
        utc_table: TDB -> UTC conversion over the run window.
        weather_dir: Directory the files were read from.
    """

    settings: WeatherConfig
    sites: Mapping[str, SiteWeather]
    year_mapping: Mapping[int, int]
    utc_table: UtcOffsetTable
    weather_dir: str

    def lookup(
        self, site_name: str, arrival_tdb_s: Float64Array
    ) -> tuple[BoolArray, Float64Array, Int64Array]:
        """``(outage, cloud_cover, historical_hour_posix)`` at arrival epochs (TDB s)."""
        sw = self.sites[site_name]
        utc = self.utc_table.utc_posix_s(np.asarray(arrival_tdb_s, dtype=np.float64))
        hist = historical_hour_posix(utc, self.year_mapping)
        idx = sw.index_of(hist)
        return sw.outage[idx], sw.cloud_cover[idx], hist

    def apply(
        self, atmosphere: AtmosphereResult, site_name: str, arrival_tdb_s: Float64Array
    ) -> AtmosphereResult:
        """Atmosphere with the weather outage applied: ``eta = 0``, attenuation ``inf`` and
        ``weather_available = False`` at blocked epochs; unchanged elsewhere."""
        outage, _tcc, _hist = self.lookup(site_name, arrival_tdb_s)
        if outage.shape != atmosphere.transmittance.shape:
            raise ValueError("arrival epochs must match the atmosphere epochs")
        return replace(
            atmosphere,
            transmittance=np.where(outage, 0.0, atmosphere.transmittance),
            attenuation_dB=np.where(outage, np.inf, atmosphere.attenuation_dB),
            weather_available=atmosphere.weather_available & ~outage,
        )

    def provenance(self) -> dict[str, Any]:
        """JSON-compatible record of the weather data used (PDF §14.1; ``weather.json``)."""
        years = sorted({int(y) for y in self.year_mapping.values()})
        return {
            "source": "era5",
            "dataset": self.settings.era5_dataset,
            "dataset_doi": DATASET_INFO[self.settings.era5_dataset][1],
            "citation": DATASET_INFO[self.settings.era5_dataset][2],
            "licence_spdx": LICENCE_SPDX,
            "attribution": ATTRIBUTION_TEMPLATE.format(
                year=max((r[:4] for s in self.sites.values() for r in s.retrieved_utc), default="?")
            ).replace("Generated using", "Contains modified"),
            "variable": ERA5_VARIABLE,
            "spatial_resolution": f"{GRID_STEP_DEG} deg regular lat/lon grid, "
            f"{self.settings.interpolation} interpolation to the site",
            "temporal_resolution": "1 h (instantaneous analyses at hh:00 UTC; epochs rounded "
            "to the nearest hour)",
            "record_years": [self.settings.era5_first_year, self.settings.era5_last_year],
            "year_mapping": self.settings.year_mapping,
            "simulated_to_historical_year": {str(k): v for k, v in self.year_mapping.items()},
            "historical_years_used": years,
            "outage_model": self.settings.outage_model,
            "cloud_cover_threshold": self.settings.cloud_cover_threshold,
            "weather_dir": self.weather_dir,
            "sites": {
                name: {
                    "latitude_deg": s.latitude_deg,
                    "longitude_deg": s.longitude_deg,
                    "node_key": s.node_key,
                    "files": list(s.files),
                    "file_sha256": list(s.file_sha256),
                    "cds_request_ids": list(s.cds_request_ids),
                    "retrieved_utc": sorted(set(s.retrieved_utc)),
                    "expver": list(s.expver),
                    "record_clear_fraction": float(1.0 - np.mean(s.outage)),
                    "record_mean_cloud_cover": float(np.mean(s.cloud_cover)),
                }
                for name, s in self.sites.items()
            },
        }


def simulated_years(config: RunConfig) -> tuple[int, ...]:
    """Simulated UTC calendar years of the run window plus one (arrival epochs may cross the
    year end after the last emission epoch; light time < 1 day)."""
    start = _dt.datetime.fromisoformat(canonical_utc(config.time.start_utc))
    end = start + _dt.timedelta(days=config.time.duration_days + 1.0)
    return tuple(range(start.year, end.year + 1))


def load_site_weather(
    site: SiteConfig,
    settings: WeatherConfig,
    directory: Path,
) -> tuple[SiteWeather, Float64Array]:
    """Read the ERA5 cloud cover of one site (outage filled later); returns the
    :class:`SiteWeather` with an all-False outage placeholder and the cloud cover."""
    node = grid_node(site.latitude_deg, site.longitude_deg)
    years = list(range(settings.era5_first_year, settings.era5_last_year + 1))
    try:
        series = load_node_series(
            directory,
            node,
            ERA5_VARIABLE,
            years,
            neighbourhood_nodes=settings.neighbourhood_nodes,
            dataset=settings.era5_dataset,
        )
    except Era5Error as exc:
        raise WeatherError(f"site '{site.name}': {exc}") from exc
    tcc = np.clip(
        site_series(series, site.latitude_deg, site.longitude_deg, settings.interpolation), 0.0, 1.0
    )
    sw = SiteWeather(
        site_name=site.name,
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        node_key=node.key,
        interpolation=settings.interpolation,
        start_posix_s=int(series.block.time_utc_posix_s[0]),
        cloud_cover=tcc,
        outage=np.zeros(tcc.shape, dtype=bool),
        files=tuple(str(f) for f in series.files),
        file_sha256=series.file_sha256,
        cds_request_ids=series.cds_request_ids,
        retrieved_utc=series.retrieved_utc,
        expver=series.block.expver,
    )
    return sw, tcc


def build_weather(
    config: RunConfig,
    kernels: LoadedKernels,
    *,
    tdb_window_s: tuple[float, float],
    rng: np.random.Generator,
    directory: Path | None = None,
) -> WeatherProvider | None:
    """Weather provider of a run, or ``None`` for ``models.weather_source = 'scenario'``.

    Args:
        config: Run configuration.
        kernels: Loaded kernels (LSK for TDB -> UTC).
        tdb_window_s: ``(min, max)`` of the arrival epochs that will be looked up [TDB s]
            (a margin of one day is added on both sides).
        rng: The weather Generator (:func:`weather_generator`); drawn from only by
            ``sampled`` and ``bernoulli`` in the documented order.
        directory: Weather directory (default :func:`helioswarm.era5.weather_dir`).
    """
    if config.models.weather_source != "era5":
        return None
    settings = config.weather
    if settings is None:  # pragma: no cover - schema rule
        raise WeatherError("models.weather_source='era5' needs the weather section")
    d = weather_dir() if directory is None else directory
    mapping = year_map(settings, simulated_years(config), rng)
    sites: dict[str, SiteWeather] = {}
    for site in config.receiver.sites:
        sw, tcc = load_site_weather(site, settings, d)
        outage = outage_from_cloud_cover(tcc, settings, rng)
        sites[site.name] = replace(sw, outage=outage)
    table = UtcOffsetTable.build(tdb_window_s[0] - 86400.0, tdb_window_s[1] + 86400.0, kernels)
    return WeatherProvider(
        settings=settings,
        sites=sites,
        year_mapping=mapping,
        utc_table=table,
        weather_dir=str(d),
    )


# ---------------------------------------------------------------------------------------------
# Climatology statistics
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteClimatology:
    """Clear-sky statistics of one site over the record for one threshold.

    Attributes:
        site_name: Site.
        latitude_deg, longitude_deg: Coordinates [deg].
        node_key: Grid node.
        threshold: Cloud-cover threshold (clear iff ``tcc <= threshold``).
        n_hours: Hours in the record.
        clear_fraction: Fraction of hours clear [1].
        bernoulli_clear_fraction: ``mean(1 - tcc)`` (expected clear fraction of the
            ``bernoulli`` model) [1].
        mean_cloud_cover: ``mean(tcc)`` [1].
        clear_fraction_by_month: 12 values (January first) [1].
        clear_fraction_by_utc_hour: 24 values [1].
        clear_fraction_by_local_solar_hour: 24 values, local mean solar time
            ``UTC + longitude/15 h`` (whole-hour bins) [1].
        clear_fraction_by_year: ``{year: fraction}`` [1].
        longest_outage_h: Longest run of consecutive cloudy hours [h].
        mean_outage_h: Mean length of cloudy runs [h].
    """

    site_name: str
    latitude_deg: float
    longitude_deg: float
    node_key: str
    threshold: float
    n_hours: int
    clear_fraction: float
    bernoulli_clear_fraction: float
    mean_cloud_cover: float
    clear_fraction_by_month: tuple[float, ...]
    clear_fraction_by_utc_hour: tuple[float, ...]
    clear_fraction_by_local_solar_hour: tuple[float, ...]
    clear_fraction_by_year: Mapping[int, float] = field(default_factory=dict)
    longest_outage_h: int = 0
    mean_outage_h: float = 0.0


def _runs(mask: BoolArray) -> Int64Array:
    """Lengths of runs of True in a 1-D boolean array."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    d = np.diff(m)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out: Int64Array = (ends - starts).astype(np.int64)
    return out


def site_climatology(
    site_name: str,
    latitude_deg: float,
    longitude_deg: float,
    node_key: str,
    time_posix_s: Int64Array,
    cloud_cover: Float64Array,
    threshold: float,
) -> SiteClimatology:
    """Clear-sky statistics (:class:`SiteClimatology`) of an hourly ``tcc`` series."""
    t = np.asarray(time_posix_s, dtype=np.int64)
    tcc = np.asarray(cloud_cover, dtype=np.float64)
    if t.shape != tcc.shape or t.size == 0:
        raise WeatherError("time and cloud cover must be non-empty and of equal shape")
    clear = tcc <= threshold
    year, month, _day, hour = calendar_fields(t)
    by_month = tuple(float(np.mean(clear[month == m])) for m in range(1, 13))
    by_hour = tuple(float(np.mean(clear[hour == h])) for h in range(24))
    local = np.floor((hour + longitude_deg / 15.0) % 24.0).astype(np.int64) % 24
    by_local = tuple(float(np.mean(clear[local == h])) for h in range(24))
    by_year = {int(y): float(np.mean(clear[year == y])) for y in np.unique(year)}
    runs = _runs(~clear)
    return SiteClimatology(
        site_name=site_name,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        node_key=node_key,
        threshold=float(threshold),
        n_hours=int(t.size),
        clear_fraction=float(np.mean(clear)),
        bernoulli_clear_fraction=float(np.mean(1.0 - tcc)),
        mean_cloud_cover=float(np.mean(tcc)),
        clear_fraction_by_month=by_month,
        clear_fraction_by_utc_hour=by_hour,
        clear_fraction_by_local_solar_hour=by_local,
        clear_fraction_by_year=by_year,
        longest_outage_h=int(np.max(runs)) if runs.size else 0,
        mean_outage_h=float(runs.mean()) if runs.size else 0.0,
    )


@dataclass(frozen=True)
class JointStatistics:
    """Network weather statistics of a set of sites at the same hours (one threshold).

    Attributes:
        site_names: Sites.
        threshold: Threshold.
        all_cloudy_fraction: Fraction of hours with every site cloudy (from the data) [1].
        all_cloudy_if_independent: Product of the sites' cloudy fractions (what independence
            would give) [1].
        at_least_one_clear_fraction: ``1 - all_cloudy_fraction`` [1].
        all_clear_fraction: Fraction of hours with every site clear [1].
        pairwise_outage_correlation: ``{"a|b": Pearson correlation of the cloudy flags}``.
    """

    site_names: tuple[str, ...]
    threshold: float
    all_cloudy_fraction: float
    all_cloudy_if_independent: float
    at_least_one_clear_fraction: float
    all_clear_fraction: float
    pairwise_outage_correlation: Mapping[str, float]


def joint_statistics(
    site_names: Sequence[str], cloud_cover: Float64Array, threshold: float
) -> JointStatistics:
    """Joint statistics of ``cloud_cover`` with shape (n_sites, T) on a common hourly axis."""
    tcc = np.asarray(cloud_cover, dtype=np.float64)
    if tcc.ndim != 2 or tcc.shape[0] != len(site_names):
        raise WeatherError("cloud_cover must have shape (n_sites, T)")
    cloudy = tcc > threshold
    all_cloudy = float(np.mean(np.all(cloudy, axis=0)))
    indep = float(np.prod(np.mean(cloudy, axis=1)))
    corr: dict[str, float] = {}
    for a in range(len(site_names)):
        for b in range(a + 1, len(site_names)):
            x, y = cloudy[a].astype(float), cloudy[b].astype(float)
            sx, sy = x.std(), y.std()
            corr[f"{site_names[a]}|{site_names[b]}"] = (
                float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
                if sx > 0 and sy > 0
                else math.nan
            )
    return JointStatistics(
        site_names=tuple(site_names),
        threshold=float(threshold),
        all_cloudy_fraction=all_cloudy,
        all_cloudy_if_independent=indep,
        at_least_one_clear_fraction=1.0 - all_cloudy,
        all_clear_fraction=float(np.mean(np.all(~cloudy, axis=0))),
        pairwise_outage_correlation=corr,
    )


def _json_finite(x: Any) -> Any:
    """``x`` with every non-finite float replaced by ``None`` (strict JSON)."""
    if isinstance(x, float | np.floating):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, Mapping):
        return {str(k): _json_finite(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_finite(v) for v in x]
    return x


def climatology_command(
    config_path: str | Path,
    out_dir: Path,
    *,
    thresholds: Sequence[float] | None,
    first_year: int | None,
    last_year: int | None,
    interpolation: Interpolation,
    directory: Path,
    overwrite: bool,
) -> Path:
    """``helioswarm weather climatology``: per-site and joint statistics from the cache.

    ``config_path`` is an ERA5 download configuration (all its sites) or a run configuration
    (its receiver sites; thresholds and years from its ``weather`` section when present).
    Writes ``climatology.json``, ``climatology_monthly.parquet``,
    ``climatology_hourly.parquet`` and ``climatology.md``. Returns the JSON path.
    """
    import pandas as pd

    from helioswarm.config import load_config, read_yaml
    from helioswarm.era5 import load_download_config

    raw = read_yaml(config_path)
    sites: list[tuple[str, float, float]]
    neighbourhood = 1
    default_thresholds: list[float] = [0.1, 0.3, 0.5]
    if isinstance(raw, dict) and "era5_schema_version" in raw:
        dl = load_download_config(config_path)
        sites = [(s.name, s.latitude_deg, s.longitude_deg) for s in dl.sites]
        fy, ly, neighbourhood = dl.first_year, dl.last_year, dl.neighbourhood_nodes
        dataset = dl.dataset
    else:
        cfg = load_config(config_path)
        sites = [(s.name, s.latitude_deg, s.longitude_deg) for s in cfg.receiver.sites]
        w = cfg.weather
        dataset = WeatherConfig.model_fields["era5_dataset"].default
        if w is not None:
            fy, ly, neighbourhood = w.era5_first_year, w.era5_last_year, w.neighbourhood_nodes
            dataset = w.era5_dataset
            if w.cloud_cover_threshold is not None:
                default_thresholds = [w.cloud_cover_threshold]
        else:
            fy, ly = (
                WeatherConfig.model_fields["era5_first_year"].default,
                (WeatherConfig.model_fields["era5_last_year"].default),
            )
    fy = first_year if first_year is not None else fy
    ly = last_year if last_year is not None else ly
    ths = list(thresholds) if thresholds else default_thresholds
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise WeatherError(f"{out_dir} exists and is not empty (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(fy, ly + 1))
    series: dict[str, tuple[Int64Array, Float64Array, str]] = {}
    files: dict[str, list[str]] = {}
    retrieved: list[str] = []
    for name, lat, lon in sites:
        node = grid_node(lat, lon)
        ns = load_node_series(
            directory,
            node,
            ERA5_VARIABLE,
            years,
            neighbourhood_nodes=neighbourhood,
            dataset=dataset,
        )
        tcc = np.clip(site_series(ns, lat, lon, interpolation), 0.0, 1.0)
        series[name] = (ns.block.time_utc_posix_s, tcc, node.key)
        retrieved += list(ns.retrieved_utc)
        files[name] = [
            f"{Path(f).name} sha256={s}" for f, s in zip(ns.files, ns.file_sha256, strict=True)
        ]
    names = [s[0] for s in sites]
    per_site: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    hourly_rows: list[dict[str, Any]] = []
    for th in ths:
        for name, lat, lon in sites:
            t, tcc, key = series[name]
            c = site_climatology(name, lat, lon, key, t, tcc, th)
            d = dict(c.__dict__)
            d["clear_fraction_by_year"] = {str(k): v for k, v in c.clear_fraction_by_year.items()}
            yearly = np.array(list(c.clear_fraction_by_year.values()))
            d["clear_fraction_year_min"] = float(yearly.min())
            d["clear_fraction_year_max"] = float(yearly.max())
            d["clear_fraction_year_std"] = float(yearly.std(ddof=1)) if yearly.size > 1 else 0.0
            per_site.append(d)
            for m in range(12):
                monthly_rows.append(
                    {
                        "threshold": th,
                        "site": name,
                        "month": m + 1,
                        "clear_fraction": c.clear_fraction_by_month[m],
                    }
                )
            for h in range(24):
                hourly_rows.append(
                    {
                        "threshold": th,
                        "site": name,
                        "hour": h,
                        "clear_fraction_utc": c.clear_fraction_by_utc_hour[h],
                        "clear_fraction_local_solar": c.clear_fraction_by_local_solar_hour[h],
                    }
                )
    joint: list[dict[str, Any]] = []
    groups: list[tuple[str, list[str]]] = [("all_sites", names)]
    if isinstance(raw, dict) and "era5_schema_version" not in raw:
        groups = [("receiver_network", names)]
    for th in ths:
        for label, members in groups:
            if len(members) < 2:
                continue
            stack = np.vstack([series[m][1] for m in members])
            js = joint_statistics(members, stack, th)
            jd = dict(js.__dict__)
            jd["group"] = label
            joint.append(jd)
    result = {
        "config": str(config_path),
        "weather_dir": str(directory),
        "variable": ERA5_VARIABLE,
        "dataset": dataset,
        "years": [fy, ly],
        "interpolation": interpolation,
        "thresholds": ths,
        "definition": "clear iff tcc <= threshold (tcc = ERA5 total cloud cover, grid-box "
        "fraction); bernoulli_clear_fraction = mean(1 - tcc)",
        # Year of the data's retrieval (file provenance), as in a run's weather.json: the text
        # does not change when the statistics are recomputed later.
        "attribution": ATTRIBUTION_TEMPLATE.format(
            year=max((r[:4] for r in retrieved), default="?")
        ).replace("Generated using", "Contains modified"),
        "retrieved_utc": sorted(set(retrieved)),
        "dataset_doi": DATASET_INFO[dataset][1],
        "files": files,
        "sites": per_site,
        "joint": joint,
    }
    path = out_dir / "climatology.json"
    # Standard JSON: undefined statistics (e.g. the correlation of a constant outage flag) -> null.
    path.write_text(
        json.dumps(_json_finite(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    pd.DataFrame(monthly_rows).to_parquet(out_dir / "climatology_monthly.parquet", index=False)
    pd.DataFrame(hourly_rows).to_parquet(out_dir / "climatology_hourly.parquet", index=False)
    lines = [
        f"# ERA5 clear-sky climatology ({fy}-{ly}, {interpolation})",
        "",
        "Generated by `helioswarm weather climatology`; clear iff tcc <= threshold.",
        "",
        "| threshold | site | lat | lon | clear fraction | min/max year | 1 - mean tcc | "
        "longest outage [h] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in per_site:
        lines.append(
            f"| {d['threshold']} | {d['site_name']} | {d['latitude_deg']} | "
            f"{d['longitude_deg']} | {d['clear_fraction']:.3f} | "
            f"{d['clear_fraction_year_min']:.3f}/{d['clear_fraction_year_max']:.3f} | "
            f"{d['bernoulli_clear_fraction']:.3f} | {d['longest_outage_h']} |"
        )
    if joint:
        lines += [
            "",
            "| threshold | group | all cloudy | if independent | all clear |",
            "|---|---|---|---|---|",
        ]
        for j in joint:
            lines.append(
                f"| {j['threshold']} | {j['group']} | {j['all_cloudy_fraction']:.4g} | "
                f"{j['all_cloudy_if_independent']:.4g} | {j['all_clear_fraction']:.4g} |"
            )
    lines += ["", str(result["attribution"]), ""]
    (out_dir / "climatology.md").write_text("\n".join(lines))
    return path


__all__ = [
    "ERA5_VARIABLE",
    "POSIX_J2000_UTC_S",
    "WEATHER_FILE",
    "WEATHER_STREAM_KEY",
    "JointStatistics",
    "SiteClimatology",
    "SiteWeather",
    "UtcOffsetTable",
    "WeatherError",
    "WeatherProvider",
    "build_weather",
    "calendar_fields",
    "climatology_command",
    "historical_hour_posix",
    "joint_statistics",
    "load_site_weather",
    "nearest_hour_posix",
    "outage_from_cloud_cover",
    "simulated_years",
    "site_climatology",
    "tdb_to_utc_posix_exact",
    "weather_generator",
    "year_map",
]
