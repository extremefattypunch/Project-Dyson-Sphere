"""ERA5 hourly single-level data at receiver sites: request builder, downloader, cache, reader.

Products (Copernicus Climate Change Service (C3S) Climate Data Store (CDS); ERA5: Hersbach et
al. 2020, "The ERA5 global reanalysis", Q. J. R. Meteorol. Soc. 146(730), 1999-2049,
doi:10.1002/qj.3803; licence CC-BY-4.0, CDS licence record ``cc-by`` revision 1):

* ``reanalysis-era5-single-levels`` -- *ERA5 hourly data on single levels from 1940 to
  present*, DOI 10.24381/cds.adbb2d47 (gridded; requests limited per year and month block).
* ``reanalysis-era5-single-levels-timeseries`` -- *ERA5 hourly time-series data on single
  levels from 1940 to present*, DOI 10.24381/1cf1ad76: the same ERA5 analyses re-arranged
  for time-series access (an area of at most 2 x 2 deg and any date range in one request,
  delivered as a zip archive holding one NetCDF file). Checked in M8a: for 2014 at the 3 x 3
  nodes around 23.5 N 0 E and 0 N 0 E both products give identical ``tcc`` values (maximum
  absolute difference 0). This is the default because a 10-year request takes about a minute
  instead of ~10 minutes of CDS queue per year.

Required acknowledgement (ECMWF Copernicus Knowledge Base, "How to acknowledge and cite a
Climate Data Store (CDS) catalogue entry and the data published as part of it", step 3):
:data:`ATTRIBUTION_TEMPLATE`.

What this module does (PDF §10.2 "later, use site-specific cloud data", §14.1 provenance,
§14.2 NetCDF only for gridded weather):

* :class:`Era5DownloadConfig` -- one YAML file (``configs/weather/*.yaml``) names the sites,
  variables and years to download.
* :func:`build_requests` -- **pure** request builder. Each site is snapped to the nearest node
  of the native 0.25 deg ERA5 grid (:func:`grid_node`); the request area is the
  ``(2k+1) x (2k+1)``-node box centred on that node (``k = neighbourhood_nodes``, default 1:
  a 3 x 3 box, enough for nearest-node and bilinear interpolation, :func:`site_series`).
  Single-levels requests are chunked per grid node, per calendar year and per block of
  consecutive months so that the CDS cost of each request stays below the limit
  (:func:`request_cost`); time-series requests take the whole period per grid node.
* :func:`download_requests` -- idempotent downloader: a file whose provenance JSON matches the
  request (request SHA-256) and whose bytes match the recorded SHA-256 is skipped; otherwise the
  request is submitted with ``cdsapi``, downloaded to a temporary file, checked (variables,
  hourly time axis, grid) and only then renamed into place next to its provenance JSON
  (:class:`Era5FileProvenance`: dataset, DOI, variables, area, period, the request itself, CDS
  request id, retrieval UTC, cdsapi version, SHA-256, licence and attribution).
* :func:`load_node_series` / :func:`site_series` -- read the cached files of one grid node
  back as an hourly UTC series and interpolate it to the site (nearest node or bilinear).

The network is touched only by :func:`download_requests` through an injectable client
factory; unit tests use a fake client and committed fixtures.

Time: ERA5 ``valid_time`` is UTC (hourly analyses at hh:00); the NetCDF files store it as
seconds since 1970-01-01 (POSIX, no leap seconds). Series are kept as int64 POSIX seconds.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]

DATASET: Final[str] = "reanalysis-era5-single-levels"
"""CDS collection id of the gridded product."""

DATASET_TIMESERIES: Final[str] = "reanalysis-era5-single-levels-timeseries"
"""CDS collection id of the time-series product (default)."""

Era5Dataset = Literal["reanalysis-era5-single-levels", "reanalysis-era5-single-levels-timeseries"]

DATASET_TITLE: Final[str] = "ERA5 hourly data on single levels from 1940 to present"
DATASET_DOI: Final[str] = "10.24381/cds.adbb2d47"
DATASET_CITATION: Final[str] = (
    "Copernicus Climate Change Service (C3S) Climate Data Store (CDS) (2018): ERA5 hourly data "
    "on single levels from 1940 to present. DOI: 10.24381/cds.adbb2d47 (DataCite record: "
    "creator C3S, publication year 2018). Reference paper: Hersbach, H. et al. (2020), The "
    "ERA5 global reanalysis, Q. J. R. Meteorol. Soc. 146(730), 1999-2049, doi:10.1002/qj.3803."
)
DATASET_TIMESERIES_TITLE: Final[str] = (
    "ERA5 hourly time-series data on single levels from 1940 to present"
)
DATASET_TIMESERIES_DOI: Final[str] = "10.24381/1cf1ad76"
DATASET_TIMESERIES_CITATION: Final[str] = (
    "ECMWF / Copernicus Climate Change Service (C3S) Climate Data Store (CDS) (2025): ERA5 "
    "hourly time-series data on single levels from 1940 to present. DOI: 10.24381/1cf1ad76 "
    "(DataCite record: creator C3S, publisher ECMWF, publication year 2025). Reference paper: "
    "Hersbach, H. et al. (2020), The ERA5 global reanalysis, Q. J. R. Meteorol. Soc. 146(730), "
    "1999-2049, doi:10.1002/qj.3803."
)
DATASET_INFO: Final[Mapping[str, tuple[str, str, str]]] = MappingProxyType(
    {
        DATASET: (DATASET_TITLE, DATASET_DOI, DATASET_CITATION),
        DATASET_TIMESERIES: (
            DATASET_TIMESERIES_TITLE,
            DATASET_TIMESERIES_DOI,
            DATASET_TIMESERIES_CITATION,
        ),
    }
)
"""Title, DOI and citation of each supported collection."""

TIMESERIES_MAX_EXTENT_DEG: Final[float] = 2.0
"""Largest latitude/longitude extent of a time-series area request (CDS form, 2026-11-13);
``neighbourhood_nodes <= 4`` keeps the (2k+1)-node box (2k x 0.25 deg) within it."""

LICENCE_SPDX: Final[str] = "CC-BY-4.0"
LICENCE_URL: Final[str] = "https://creativecommons.org/licenses/by/4.0/legalcode"
ATTRIBUTION_TEMPLATE: Final[str] = (
    "Generated using Copernicus Climate Change Service information {year}. Neither the "
    "European Commission nor ECMWF is responsible for any use that may be made of the "
    "Copernicus information or data it contains."
)
"""CDS acknowledgement wording ('Generated using' for unmodified data, 'Contains modified'
for derived data); ``{year}`` = year of the retrieval."""

GRID_STEP_DEG: Final[float] = 0.25
"""Native ERA5 atmospheric grid spacing of the CDS regular lat/lon product [deg]."""

CDS_COST_LIMIT: Final[int] = 121_000
"""Per-request cost limit of the CDS costing endpoint for this collection (measured
2026-11-13 with ``ecmwf.datastores`` ``estimate_costs``: ``{'id': 'size', 'limit': 121000}``)."""

CDS_COST_PER_FIELD: Final[int] = 6
"""Cost units per (variable, hour) field reported by the CDS costing endpoint for this
collection, independent of the area (measured 2026-11-13: 1 variable x 8760 h -> 52560;
1 variable x 8784 h (2020) -> 52704; 6 variables x 744 h -> 26784; area 0.5 deg or 20 deg
box identical). An empirical constant of the service, re-checked by the downloader through the
size limit only (a request above the limit is rejected by the CDS)."""

WEATHER_DIR_ENV: Final[str] = "HELIOSWARM_WEATHER_DIR"
"""Environment variable naming the weather data directory."""

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_WEATHER_DIR: Final[Path] = _REPO_ROOT / "data" / "weather"

VARIABLE_SHORT_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "total_cloud_cover": "tcc",
        "low_cloud_cover": "lcc",
        "medium_cloud_cover": "mcc",
        "high_cloud_cover": "hcc",
        "total_column_water_vapour": "tcwv",
        "total_column_cloud_liquid_water": "tclw",
    }
)
"""CDS variable names accepted by the builder -> NetCDF short names (ECMWF parameter
database: tcc 164, lcc 186, mcc 187, hcc 188, tcwv 137, tclw 78)."""

PROVENANCE_SCHEMA_VERSION: Final = 1
_CHUNK_BYTES: Final[int] = 1 << 20


class Era5Error(RuntimeError):
    """ERA5 configuration, download or cache error."""


# ---------------------------------------------------------------------------------------------
# Download configuration
# ---------------------------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class Era5Site(_Model):
    """A site to download (geodetic WGS84 coordinates; ERA5's grid is treated as geodetic)."""

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=64)
    latitude_deg: float = Field(ge=-89.5, le=89.5)
    longitude_deg: float = Field(ge=-180.0, le=180.0)
    origin: str = Field(
        description="Where the site comes from (a run configuration and site name, or "
        "'candidate' for an example site evaluated from the data)."
    )


class Era5DownloadConfig(_Model):
    """Contents of an ERA5 download configuration file (``configs/weather/*.yaml``)."""

    era5_schema_version: Literal[1]
    dataset: Era5Dataset = "reanalysis-era5-single-levels-timeseries"
    product_type: Literal["reanalysis"] = "reanalysis"
    variables: tuple[str, ...] = Field(min_length=1)
    first_year: int = Field(ge=1940, le=2100)
    last_year: int = Field(ge=1940, le=2100)
    neighbourhood_nodes: int = Field(1, ge=1, le=4)
    max_request_cost: int = Field(CDS_COST_LIMIT, ge=1)
    sites: tuple[Era5Site, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Era5DownloadConfig:
        issues: list[str] = []
        if self.last_year < self.first_year:
            issues.append(f"last_year {self.last_year} < first_year {self.first_year}")
        unknown = [v for v in self.variables if v not in VARIABLE_SHORT_NAMES]
        if unknown:
            issues.append(f"unknown variables {unknown}; known: {sorted(VARIABLE_SHORT_NAMES)}")
        if len(set(self.variables)) != len(self.variables):
            issues.append("duplicate variables")
        names = [s.name for s in self.sites]
        if len(set(names)) != len(names):
            issues.append("duplicate site names")
        if issues:
            raise ValueError("; ".join(issues))
        return self


def load_download_config(path: str | Path) -> Era5DownloadConfig:
    """Read and validate an ERA5 download configuration (YAML)."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    try:
        return Era5DownloadConfig.model_validate(raw, strict=False)
    except ValueError as exc:
        raise Era5Error(f"{path}: invalid ERA5 download configuration: {exc}") from exc


# ---------------------------------------------------------------------------------------------
# Grid and request builder (pure)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class GridNode:
    """A node of the 0.25 deg ERA5 grid, stored as integer multiples of the grid step.

    Attributes:
        lat_index: Latitude / 0.25 deg (integer).
        lon_index: Longitude / 0.25 deg (integer, longitude in [-180, 180)).
    """

    lat_index: int
    lon_index: int

    @property
    def latitude_deg(self) -> float:
        """Node latitude [deg]."""
        return self.lat_index * GRID_STEP_DEG

    @property
    def longitude_deg(self) -> float:
        """Node longitude in [-180, 180) [deg]."""
        return self.lon_index * GRID_STEP_DEG

    @property
    def key(self) -> str:
        """File-system key, e.g. ``N23.50_W120.00``."""
        lat, lon = self.latitude_deg, self.longitude_deg
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        return f"{ns}{abs(lat):05.2f}_{ew}{abs(lon):06.2f}"


def grid_node(latitude_deg: float, longitude_deg: float) -> GridNode:
    """Nearest node of the 0.25 deg ERA5 grid (ties rounded half away from zero).

    Longitude is wrapped into [-180, 180) first. Nearest-node snapping moves a site by at most
    0.125 deg in latitude and longitude (<= ~14 km north-south).
    """
    if not (math.isfinite(latitude_deg) and math.isfinite(longitude_deg)):
        raise ValueError("coordinates must be finite")
    lon = ((longitude_deg + 180.0) % 360.0) - 180.0

    def snap(x: float) -> int:
        q = x / GRID_STEP_DEG
        return int(math.copysign(math.floor(abs(q) + 0.5), q))

    li = snap(lon)
    if li * GRID_STEP_DEG >= 180.0:
        li -= round(360.0 / GRID_STEP_DEG)
    return GridNode(lat_index=snap(latitude_deg), lon_index=li)


def request_area(node: GridNode, neighbourhood_nodes: int) -> tuple[float, float, float, float]:
    """CDS ``area`` ``[N, W, S, E]`` [deg] of the box of ``2k+1`` nodes centred on ``node``.

    Raises:
        Era5Error: the box would cross a pole or the antimeridian (not supported: the CDS
            area semantics across the dateline are not relied upon).
    """
    d = neighbourhood_nodes * GRID_STEP_DEG
    north, south = node.latitude_deg + d, node.latitude_deg - d
    west, east = node.longitude_deg - d, node.longitude_deg + d
    if north > 90.0 or south < -90.0:
        raise Era5Error(f"area around {node.key} crosses a pole")
    if west < -180.0 or east >= 180.0:
        raise Era5Error(f"area around {node.key} crosses the antimeridian (not supported)")
    return (north, west, south, east)


def hours_in_months(year: int, months: Sequence[int]) -> int:
    """Number of hourly analyses in the given months of ``year`` (Gregorian calendar)."""
    return sum(24 * calendar.monthrange(year, m)[1] for m in months)


def request_cost(n_variables: int, year: int, months: Sequence[int]) -> int:
    """CDS cost estimate ``CDS_COST_PER_FIELD * n_variables * n_hours`` of one request."""
    return CDS_COST_PER_FIELD * n_variables * hours_in_months(year, months)


def month_blocks(year: int, n_variables: int, max_cost: int) -> tuple[tuple[int, ...], ...]:
    """Split the 12 months of ``year`` into the fewest blocks of consecutive months whose
    :func:`request_cost` is <= ``max_cost`` (greedy, which is optimal for consecutive
    blocks with positive weights).

    Raises:
        Era5Error: a single month already exceeds ``max_cost``.
    """
    blocks: list[tuple[int, ...]] = []
    current: list[int] = []
    for m in range(1, 13):
        if request_cost(n_variables, year, (m,)) > max_cost:
            raise Era5Error(
                f"one month of {n_variables} variable(s) costs "
                f"{request_cost(n_variables, year, (m,))} > {max_cost}; request fewer variables"
            )
        if current and request_cost(n_variables, year, (*current, m)) > max_cost:
            blocks.append(tuple(current))
            current = []
        current.append(m)
    blocks.append(tuple(current))
    return tuple(blocks)


@dataclass(frozen=True)
class Era5Request:
    """One CDS request: one grid node and a period (single levels: one year and a block of
    consecutive months; time series: whole years ``first_year .. last_year``).

    Attributes:
        dataset: CDS collection id.
        product_type: ``reanalysis``.
        node: Centre grid node.
        neighbourhood_nodes: Half-width ``k`` of the node box.
        area: ``[N, W, S, E]`` [deg].
        year: First calendar year.
        months: Consecutive months (1-12) of each year (all 12 for the time series).
        variables: CDS variable names.
        site_names: Configured sites snapped to ``node`` (metadata, not part of the request).
        last_year: Last calendar year (``None`` = ``year``).
    """

    dataset: str
    product_type: str
    node: GridNode
    neighbourhood_nodes: int
    area: tuple[float, float, float, float]
    year: int
    months: tuple[int, ...]
    variables: tuple[str, ...]
    site_names: tuple[str, ...]
    last_year: int | None = None

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_INFO:
            raise ValueError(f"unsupported dataset {self.dataset!r}")
        if self.last_year is not None and self.last_year < self.year:
            raise ValueError("last_year < year")
        if self.final_year != self.year and self.months != tuple(range(1, 13)):
            raise ValueError("a multi-year request must cover whole years")
        if self.dataset == DATASET and self.final_year != self.year:
            raise ValueError("single-levels requests cover one year")

    @property
    def final_year(self) -> int:
        """Last calendar year covered."""
        return self.year if self.last_year is None else self.last_year

    def as_cds_dict(self) -> dict[str, Any]:
        """The CDS request payload (keys as accepted by the CDS for this collection).

        Single levels: days 01-31 are always listed (the CDS ignores non-existent dates; the
        cost of a February request counts its real days only, measured). Time series: one
        ``date`` range ``YYYY-MM-DD/YYYY-MM-DD``.
        """
        if self.dataset == DATASET_TIMESERIES:
            last_day = calendar.monthrange(self.final_year, self.months[-1])[1]
            return {
                "variable": list(self.variables),
                "area": [float(a) for a in self.area],
                "date": [
                    f"{self.year:04d}-{self.months[0]:02d}-01/"
                    f"{self.final_year:04d}-{self.months[-1]:02d}-{last_day:02d}"
                ],
                "data_format": "netcdf",
            }
        return {
            "product_type": [self.product_type],
            "variable": list(self.variables),
            "year": [f"{self.year:04d}"],
            "month": [f"{m:02d}" for m in self.months],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [float(a) for a in self.area],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

    @property
    def request_sha256(self) -> str:
        """SHA-256 of the canonical JSON of (dataset, request payload)."""
        payload = json.dumps(
            {"dataset": self.dataset, "request": self.as_cds_dict()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    @property
    def n_hours(self) -> int:
        """Hourly analyses covered."""
        return sum(hours_in_months(y, self.months) for y in range(self.year, self.final_year + 1))

    @property
    def period_start_utc(self) -> str:
        """First hour covered (ISO 8601 UTC)."""
        return f"{self.year:04d}-{self.months[0]:02d}-01T00:00:00Z"

    @property
    def period_end_utc(self) -> str:
        """Last hour covered (ISO 8601 UTC)."""
        last = calendar.monthrange(self.final_year, self.months[-1])[1]
        return f"{self.final_year:04d}-{self.months[-1]:02d}-{last:02d}T23:00:00Z"

    @property
    def relative_path(self) -> Path:
        """Cache path relative to the weather directory."""
        var = "-".join(VARIABLE_SHORT_NAMES[v] for v in self.variables)
        if self.dataset == DATASET_TIMESERIES:
            period = f"{self.year:04d}-{self.final_year:04d}"
        else:
            period = f"{self.year:04d}_m{self.months[0]:02d}-{self.months[-1]:02d}"
        name = f"{self.node.key}_k{self.neighbourhood_nodes}_{var}_{period}.nc"
        return Path("era5") / self.dataset / self.node.key / name


def build_requests(config: Era5DownloadConfig) -> tuple[Era5Request, ...]:
    """All requests of a download configuration (pure; deterministic order: node, year,
    months). Sites on the same grid node share requests."""
    by_node: dict[GridNode, list[str]] = {}
    for site in config.sites:
        by_node.setdefault(grid_node(site.latitude_deg, site.longitude_deg), []).append(site.name)
    out: list[Era5Request] = []
    for node in sorted(by_node):
        area = request_area(node, config.neighbourhood_nodes)
        if config.dataset == DATASET_TIMESERIES:
            out.append(
                Era5Request(
                    dataset=config.dataset,
                    product_type=config.product_type,
                    node=node,
                    neighbourhood_nodes=config.neighbourhood_nodes,
                    area=area,
                    year=config.first_year,
                    months=tuple(range(1, 13)),
                    variables=config.variables,
                    site_names=tuple(sorted(by_node[node])),
                    last_year=config.last_year,
                )
            )
            continue
        for year in range(config.first_year, config.last_year + 1):
            for months in month_blocks(year, len(config.variables), config.max_request_cost):
                out.append(
                    Era5Request(
                        dataset=config.dataset,
                        product_type=config.product_type,
                        node=node,
                        neighbourhood_nodes=config.neighbourhood_nodes,
                        area=area,
                        year=year,
                        months=months,
                        variables=config.variables,
                        site_names=tuple(sorted(by_node[node])),
                    )
                )
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# Provenance and cache verification
# ---------------------------------------------------------------------------------------------


class Era5FileProvenance(_Model):
    """Provenance JSON written next to every downloaded file (PDF §14.1)."""

    provenance_schema_version: Literal[1]
    dataset: str
    dataset_title: str
    dataset_doi: str
    citation: str
    licence_spdx: str
    licence_url: str
    attribution: str
    product_type: str
    variables: tuple[str, ...]
    variable_short_names: tuple[str, ...]
    grid_step_deg: float
    node_key: str
    node_latitude_deg: float
    node_longitude_deg: float
    neighbourhood_nodes: int
    area_NWSE_deg: tuple[float, float, float, float]
    period_start_utc: str
    period_end_utc: str
    temporal_resolution: str
    n_hours: int
    request: dict[str, Any]
    request_sha256: str
    cds_request_id: str | None
    retrieved_utc: str
    cdsapi_version: str
    ecmwf_datastores_client_version: str | None
    file_name: str
    file_size_bytes: int
    file_sha256: str
    site_names: tuple[str, ...]
    archive_member: str | None = None
    archive_sha256: str | None = None


def provenance_path(data_path: Path) -> Path:
    """Path of the provenance JSON of a data file (``<file>.provenance.json``)."""
    return data_path.with_name(data_path.name + ".provenance.json")


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def read_provenance(data_path: Path) -> Era5FileProvenance:
    """Read and validate the provenance JSON of a data file."""
    p = provenance_path(data_path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return Era5FileProvenance.model_validate(raw, strict=False)
    except (OSError, ValueError) as exc:
        raise Era5Error(f"{p}: missing or invalid provenance: {exc}") from exc


def verify_cached(data_path: Path, request: Era5Request) -> bool:
    """True if ``data_path`` exists, its provenance describes ``request`` (same request
    SHA-256) and the file's size and SHA-256 equal the recorded ones."""
    if not data_path.is_file() or not provenance_path(data_path).is_file():
        return False
    try:
        prov = read_provenance(data_path)
    except Era5Error:
        return False
    if prov.request_sha256 != request.request_sha256:
        return False
    if data_path.stat().st_size != prov.file_size_bytes:
        return False
    return file_sha256(data_path) == prov.file_sha256


def weather_dir(override: str | Path | None = None) -> Path:
    """Weather data directory: ``override``, else ``$HELIOSWARM_WEATHER_DIR``, else
    ``data/weather/`` of the source checkout."""
    if override is not None:
        return Path(override)
    env = os.environ.get(WEATHER_DIR_ENV)
    return Path(env) if env else DEFAULT_WEATHER_DIR


# ---------------------------------------------------------------------------------------------
# NetCDF content
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Era5Block:
    """Gridded hourly values of one file (or several concatenated).

    Attributes:
        time_utc_posix_s: Valid times, int64 POSIX seconds (UTC, hourly), shape (T,).
        latitude_deg: Node latitudes, descending (the gridded product's order; files stored
            ascending, as the time-series product does, are flipped on reading), shape (Y,).
        longitude_deg: Node longitudes (ascending), shape (X,).
        values: ``{short_name: array (T, Y, X)}`` (float64; fractions or kg m^-2).
        expver: Distinct ERA5 experiment versions in the file (``0001`` = final ERA5,
            ``0005`` = preliminary ERA5T), empty if the file has no ``expver``.
    """

    time_utc_posix_s: Int64Array
    latitude_deg: Float64Array
    longitude_deg: Float64Array
    values: Mapping[str, Float64Array]
    expver: tuple[str, ...] = ()


def read_era5_netcdf(path: Path, short_names: Sequence[str]) -> Era5Block:
    """Read ERA5 CDS NetCDF-4 (``valid_time``, ``latitude``, ``longitude``) with xarray.

    Needs the optional ``weather`` extra (xarray + h5netcdf).
    """
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment without the extra
        raise Era5Error(
            "reading ERA5 files needs the 'weather' extra (uv sync --extra weather)"
        ) from exc
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        times = ds[tname].values.astype("datetime64[s]").astype(np.int64)
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
        vals: dict[str, Float64Array] = {}
        for name in short_names:
            if name not in ds.data_vars:
                raise Era5Error(f"{path}: variable '{name}' missing (has {list(ds.data_vars)})")
            arr = np.asarray(ds[name].transpose(tname, "latitude", "longitude").values)
            vals[name] = arr.astype(np.float64)
        if lat.size > 1 and lat[0] < lat[-1]:  # ascending -> canonical descending order
            lat = lat[::-1].copy()
            vals = {k: v[:, ::-1, :].copy() for k, v in vals.items()}
        expver = (
            tuple(sorted({str(v) for v in np.atleast_1d(ds["expver"].values)}))
            if "expver" in ds.variables
            else ()
        )
    return Era5Block(
        time_utc_posix_s=np.asarray(times, dtype=np.int64),
        latitude_deg=lat,
        longitude_deg=lon,
        values=vals,
        expver=expver,
    )


def check_block(block: Era5Block, request: Era5Request) -> None:
    """Check that a downloaded block is the requested one: hourly UTC times covering exactly
    the requested months, the requested node box, finite values (cloud fractions in [0, 1]).

    Raises:
        Era5Error: on any mismatch.
    """
    t = block.time_utc_posix_s
    start = int(_dt.datetime.fromisoformat(request.period_start_utc).timestamp())
    expected = start + 3600 * np.arange(request.n_hours, dtype=np.int64)
    if t.shape != expected.shape or not np.array_equal(t, expected):
        raise Era5Error(
            f"time axis mismatch: {t.size} times, expected {request.n_hours} hourly from "
            f"{request.period_start_utc}"
        )
    k = request.neighbourhood_nodes
    want_lat = request.node.latitude_deg + GRID_STEP_DEG * np.arange(k, -k - 1, -1)
    want_lon = request.node.longitude_deg + GRID_STEP_DEG * np.arange(-k, k + 1)
    if block.latitude_deg.shape != want_lat.shape or not np.allclose(
        block.latitude_deg, want_lat, atol=1e-6
    ):
        raise Era5Error(f"latitude axis {block.latitude_deg} != requested {want_lat}")
    if block.longitude_deg.shape != want_lon.shape or not np.allclose(
        block.longitude_deg, want_lon, atol=1e-6
    ):
        raise Era5Error(f"longitude axis {block.longitude_deg} != requested {want_lon}")
    for name, arr in block.values.items():
        if not np.all(np.isfinite(arr)):
            raise Era5Error(f"variable '{name}' has non-finite values")
        # ERA5 cloud cover is stored with GRIB packing: allow a packing tolerance.
        cloud = name in ("tcc", "lcc", "mcc", "hcc")
        if cloud and (np.any(arr < -1e-6) or np.any(arr > 1.0 + 1e-6)):
            raise Era5Error(f"cloud fraction '{name}' outside [0, 1]")


# ---------------------------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------------------------


class SubmittedRequest(Protocol):
    """What a CDS client returns for a submitted request (``ecmwf.datastores.Remote``)."""

    @property
    def request_id(self) -> str:
        """CDS request (job) id."""
        ...

    def download(self, target: str | None = None) -> str:
        """Wait for the result and download it to ``target``; returns the path."""
        ...


class CdsClient(Protocol):
    """The part of ``cdsapi.Client`` (``LegacyClient``, cdsapi >= 0.7.7) used here."""

    def retrieve(self, name: str, request: dict[str, Any]) -> SubmittedRequest:
        """Submit ``request`` to collection ``name`` (no waiting)."""
        ...


def default_client() -> CdsClient:
    """``cdsapi.Client`` reading ``~/.cdsapirc`` (never printed), returning the submitted
    request (``wait_until_complete=False``) so that the CDS request id is recorded."""
    import cdsapi

    client: CdsClient = cdsapi.Client(quiet=True, progress=False, wait_until_complete=False)
    return client


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class DownloadOutcome:
    """Result of one request in :func:`download_requests`.

    Attributes:
        request: The request.
        path: Final data file path.
        action: ``present`` (verified cache hit) or ``downloaded``.
        cds_request_id: CDS request id (downloaded only).
        file_size_bytes: Size of the data file [byte].
    """

    request: Era5Request
    path: Path
    action: Literal["present", "downloaded"]
    cds_request_id: str | None
    file_size_bytes: int


def download_requests(
    requests: Iterable[Era5Request],
    directory: Path,
    *,
    client_factory: Callable[[], CdsClient] = default_client,
    log: Callable[[str], None] | None = None,
    now_utc: Callable[[], _dt.datetime] | None = None,
) -> tuple[DownloadOutcome, ...]:
    """Download every request not yet cached and verified (idempotent; module docstring).

    Each file is downloaded to a temporary file in its final directory, checked
    (:func:`check_block`), hashed, and only then renamed into place; the provenance JSON is
    written atomically after the data file. A data file without valid provenance, or whose
    bytes no longer match it, is downloaded again.

    Args:
        requests: Requests (:func:`build_requests`).
        directory: Weather data directory (:func:`weather_dir`).
        client_factory: Creates the CDS client (called once, lazily; tests inject a fake).
        log: Progress messages.
        now_utc: Clock for the retrieval time (tests inject a fixed one).

    Returns:
        One :class:`DownloadOutcome` per request, in order.
    """
    emit = log if log is not None else (lambda _m: None)
    clock = now_utc if now_utc is not None else (lambda: _dt.datetime.now(_dt.UTC))
    client: CdsClient | None = None
    outcomes: list[DownloadOutcome] = []
    for req in requests:
        final = directory / req.relative_path
        if verify_cached(final, req):
            emit(f"present    {req.relative_path}")
            outcomes.append(DownloadOutcome(req, final, "present", None, final.stat().st_size))
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        if client is None:
            client = client_factory()
        emit(f"submit     {req.relative_path}  ({req.n_hours} h x {len(req.variables)} var)")
        fd, tmp_name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".part", dir=final.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        raw = tmp.with_name(tmp.name + ".raw")
        try:
            submitted = client.retrieve(req.dataset, req.as_cds_dict())
            request_id = str(getattr(submitted, "request_id", "")) or None
            submitted.download(str(raw))
            member, archive_sha = _unpack_single_netcdf(raw, tmp)
            short = [VARIABLE_SHORT_NAMES[v] for v in req.variables]
            check_block(read_era5_netcdf(tmp, short), req)
            retrieved = clock().astimezone(_dt.UTC)
            sha = file_sha256(tmp)
            size = tmp.stat().st_size
            title, doi, citation = DATASET_INFO[req.dataset]
            prov = Era5FileProvenance(
                provenance_schema_version=PROVENANCE_SCHEMA_VERSION,
                dataset=req.dataset,
                dataset_title=title,
                dataset_doi=doi,
                citation=citation,
                licence_spdx=LICENCE_SPDX,
                licence_url=LICENCE_URL,
                attribution=ATTRIBUTION_TEMPLATE.format(year=retrieved.year),
                product_type=req.product_type,
                variables=req.variables,
                variable_short_names=tuple(short),
                grid_step_deg=GRID_STEP_DEG,
                node_key=req.node.key,
                node_latitude_deg=req.node.latitude_deg,
                node_longitude_deg=req.node.longitude_deg,
                neighbourhood_nodes=req.neighbourhood_nodes,
                area_NWSE_deg=req.area,
                period_start_utc=req.period_start_utc,
                period_end_utc=req.period_end_utc,
                temporal_resolution="1 h (instantaneous analyses at hh:00 UTC)",
                n_hours=req.n_hours,
                request=req.as_cds_dict(),
                request_sha256=req.request_sha256,
                cds_request_id=request_id,
                retrieved_utc=retrieved.strftime("%Y-%m-%dT%H:%M:%SZ"),
                cdsapi_version=_package_version("cdsapi") or "unknown",
                ecmwf_datastores_client_version=_package_version("ecmwf-datastores-client"),
                file_name=final.name,
                file_size_bytes=size,
                file_sha256=sha,
                site_names=req.site_names,
                archive_member=member,
                archive_sha256=archive_sha,
            )
            tmp.chmod(0o644)
            tmp.replace(final)
            _write_json_atomic(provenance_path(final), prov.model_dump(mode="json"))
        finally:
            tmp.unlink(missing_ok=True)
            raw.unlink(missing_ok=True)
        emit(f"downloaded {req.relative_path}  {size} B  sha256 {sha}  request {request_id}")
        outcomes.append(DownloadOutcome(req, final, "downloaded", request_id, size))
    return tuple(outcomes)


def _unpack_single_netcdf(raw: Path, target: Path) -> tuple[str | None, str | None]:
    """Move the downloaded payload to ``target``: a NetCDF file as is, or the single ``.nc``
    member of a zip archive (the time-series product's delivery). Returns ``(member name,
    archive SHA-256)`` for an archive, ``(None, None)`` otherwise.

    Raises:
        Era5Error: an archive with other than exactly one plain ``.nc`` member.
    """
    import shutil
    import zipfile

    if not zipfile.is_zipfile(raw):
        raw.replace(target)
        return None, None
    with zipfile.ZipFile(raw) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        if len(members) != 1:
            raise Era5Error(f"archive holds {len(members)} files, expected one NetCDF file")
        info = members[0]
        name = info.filename
        if not name.endswith(".nc") or "/" in name or "\\" in name or name.startswith("."):
            raise Era5Error(f"unexpected archive member {name!r}")
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return name, file_sha256(raw)


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp = Path(tmp_name)
    tmp.chmod(0o644)
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------
# Reading the cache back
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class NodeSeries:
    """Hourly gridded series of one grid-node box over whole calendar years.

    Attributes:
        node: Centre grid node.
        variable: CDS variable name.
        short_name: NetCDF short name.
        block: Concatenated data (hourly, gap-free, UTC).
        files: Data files used (in time order).
        file_sha256: Their SHA-256 (from the verified provenance).
        cds_request_ids: CDS request ids (provenance).
        retrieved_utc: Retrieval times (provenance).
        years: Calendar years covered.
    """

    node: GridNode
    variable: str
    short_name: str
    block: Era5Block
    files: tuple[Path, ...]
    file_sha256: tuple[str, ...]
    cds_request_ids: tuple[str | None, ...]
    retrieved_utc: tuple[str, ...]
    years: tuple[int, ...]


def load_node_series(
    directory: Path,
    node: GridNode,
    variable: str,
    years: Sequence[int],
    *,
    neighbourhood_nodes: int = 1,
    dataset: str = DATASET_TIMESERIES,
) -> NodeSeries:
    """Load the cached files of ``dataset`` at ``node`` covering ``years`` for ``variable``.

    Files are found through their provenance JSON (any file of the collection whose provenance
    lists the variable, the node and the neighbourhood and whose period overlaps the years);
    every file's SHA-256 is re-verified, the hours outside the years are dropped, and the
    concatenated time axis must be hourly and gap-free over the whole years.

    Raises:
        Era5Error: missing files, failed verification or a gap.
    """
    if variable not in VARIABLE_SHORT_NAMES:
        raise Era5Error(f"unknown variable {variable!r}")
    short = VARIABLE_SHORT_NAMES[variable]
    if dataset not in DATASET_INFO:
        raise Era5Error(f"unsupported dataset {dataset!r}")
    if sorted(set(years)) != list(range(min(years), max(years) + 1)):
        raise Era5Error(f"years must be consecutive, got {sorted(years)}")
    y0, y1 = min(years), max(years)
    node_dir = directory / "era5" / dataset / node.key
    candidates: list[tuple[str, Path, Era5FileProvenance]] = []
    for prov_file in sorted(node_dir.glob("*.nc.provenance.json")) if node_dir.is_dir() else []:
        data = prov_file.with_name(prov_file.name[: -len(".provenance.json")])
        prov = read_provenance(data)
        if (
            variable in prov.variables
            and prov.neighbourhood_nodes == neighbourhood_nodes
            and int(prov.period_start_utc[:4]) <= y1
            and int(prov.period_end_utc[:4]) >= y0
        ):
            candidates.append((prov.period_start_utc, data, prov))
    candidates.sort(key=lambda c: c[0])
    if not candidates:
        raise Era5Error(
            f"no ERA5 '{variable}' files for node {node.key} in {node_dir} (run "
            "`helioswarm weather download`)"
        )
    blocks: list[Era5Block] = []
    for _, data, prov in candidates:
        if not data.is_file() or data.stat().st_size != prov.file_size_bytes:
            raise Era5Error(f"{data}: missing or size differs from its provenance")
        if file_sha256(data) != prov.file_sha256:
            raise Era5Error(f"{data}: SHA-256 differs from its provenance")
        blocks.append(read_era5_netcdf(data, [short]))
    first = int(_dt.datetime(y0, 1, 1, tzinfo=_dt.UTC).timestamp())
    last = int(_dt.datetime(y1, 12, 31, 23, tzinfo=_dt.UTC).timestamp())
    keep = [(b.time_utc_posix_s >= first) & (b.time_utc_posix_s <= last) for b in blocks]
    t = np.concatenate([b.time_utc_posix_s[k] for b, k in zip(blocks, keep, strict=True)])
    expected = np.arange(first, last + 1, 3600, dtype=np.int64)
    if t.shape != expected.shape or not np.array_equal(t, expected):
        raise Era5Error(
            f"ERA5 '{variable}' at {node.key}: time axis is not hourly and gap-free over "
            f"{min(years)}-{max(years)} ({t.size} hours found, {expected.size} expected)"
        )
    lat, lon = blocks[0].latitude_deg, blocks[0].longitude_deg
    for b in blocks[1:]:
        if not (np.array_equal(b.latitude_deg, lat) and np.array_equal(b.longitude_deg, lon)):
            raise Era5Error(f"grid differs between files of node {node.key}")
    values = np.concatenate([b.values[short][k] for b, k in zip(blocks, keep, strict=True)], axis=0)
    return NodeSeries(
        node=node,
        variable=variable,
        short_name=short,
        block=Era5Block(
            time_utc_posix_s=t,
            latitude_deg=lat,
            longitude_deg=lon,
            values={short: values},
            expver=tuple(sorted({v for b in blocks for v in b.expver})),
        ),
        files=tuple(c[1] for c in candidates),
        file_sha256=tuple(c[2].file_sha256 for c in candidates),
        cds_request_ids=tuple(c[2].cds_request_id for c in candidates),
        retrieved_utc=tuple(c[2].retrieved_utc for c in candidates),
        years=tuple(range(y0, y1 + 1)),
    )


Interpolation = Literal["nearest", "bilinear"]


def site_series(
    series: NodeSeries,
    latitude_deg: float,
    longitude_deg: float,
    interpolation: Interpolation,
) -> Float64Array:
    """Hourly values at a site from a node box (shape (T,)).

    * ``nearest``: the value of the nearest grid node (the box centre by construction). ERA5
      values at a node represent the model grid box around it (~28 km at 0.25 deg).
    * ``bilinear``: bilinear interpolation between the four nodes enclosing the site
      (standard tensor-product linear interpolation in latitude and longitude).

    Raises:
        Era5Error: the site is not inside the box (bilinear) or not nearest to its centre.
    """
    block = series.block
    vals = block.values[series.short_name]
    lat_axis = block.latitude_deg
    lon_axis = block.longitude_deg
    node = grid_node(latitude_deg, longitude_deg)
    if node != series.node:
        raise Era5Error(
            f"site ({latitude_deg}, {longitude_deg}) snaps to {node.key}, not {series.node.key}"
        )
    lon = ((longitude_deg + 180.0) % 360.0) - 180.0
    if interpolation == "nearest":
        i = int(np.argmin(np.abs(lat_axis - node.latitude_deg)))
        j = int(np.argmin(np.abs(lon_axis - node.longitude_deg)))
        out: Float64Array = np.array(vals[:, i, j], dtype=np.float64)
        return out
    if interpolation == "bilinear":
        # lat axis descending (ERA5 order): find i with lat[i] >= lat >= lat[i+1]
        i_candidates = np.flatnonzero(
            (lat_axis[:-1] >= latitude_deg) & (lat_axis[1:] <= latitude_deg)
        )
        j_candidates = np.flatnonzero((lon_axis[:-1] <= lon) & (lon_axis[1:] >= lon))
        if not i_candidates.size or not j_candidates.size:
            raise Era5Error("site outside the downloaded node box")
        i, j = int(i_candidates[0]), int(j_candidates[0])
        u = (lat_axis[i] - latitude_deg) / (lat_axis[i] - lat_axis[i + 1])
        v = (lon - lon_axis[j]) / (lon_axis[j + 1] - lon_axis[j])
        res: Float64Array = (
            (1 - u) * (1 - v) * vals[:, i, j]
            + (1 - u) * v * vals[:, i, j + 1]
            + u * (1 - v) * vals[:, i + 1, j]
            + u * v * vals[:, i + 1, j + 1]
        ).astype(np.float64)
        return res
    raise ValueError(f"unknown interpolation {interpolation!r}")


__all__ = [
    "ATTRIBUTION_TEMPLATE",
    "CDS_COST_LIMIT",
    "CDS_COST_PER_FIELD",
    "DATASET",
    "DATASET_CITATION",
    "DATASET_DOI",
    "DATASET_INFO",
    "DATASET_TIMESERIES",
    "DATASET_TIMESERIES_DOI",
    "DATASET_TITLE",
    "DEFAULT_WEATHER_DIR",
    "GRID_STEP_DEG",
    "LICENCE_SPDX",
    "VARIABLE_SHORT_NAMES",
    "WEATHER_DIR_ENV",
    "CdsClient",
    "DownloadOutcome",
    "Era5Block",
    "Era5Dataset",
    "Era5DownloadConfig",
    "Era5Error",
    "Era5FileProvenance",
    "Era5Request",
    "Era5Site",
    "GridNode",
    "Interpolation",
    "NodeSeries",
    "SubmittedRequest",
    "build_requests",
    "check_block",
    "default_client",
    "download_requests",
    "file_sha256",
    "grid_node",
    "hours_in_months",
    "load_download_config",
    "load_node_series",
    "month_blocks",
    "provenance_path",
    "read_era5_netcdf",
    "read_provenance",
    "request_area",
    "request_cost",
    "site_series",
    "verify_cached",
    "weather_dir",
]
