"""Synthetic ERA5 files for tests (importable as ``era5_helpers``; no network).

:class:`FakeCdsClient` stands in for ``cdsapi.Client``: ``retrieve`` returns a submitted
request whose ``download`` writes a NetCDF-4 file with the layout of the CDS output
(``valid_time`` hourly UTC, ``latitude`` descending, ``longitude`` ascending, float32 ``tcc``)
for exactly the requested months and node box, with values from a user function
``tcc(time_posix_s, lat_deg, lon_deg)``. Running :func:`helioswarm.era5.download_requests`
with it therefore exercises the real downloader, checks and provenance.
"""

from __future__ import annotations

import calendar
import datetime as dt
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from helioswarm.era5 import (
    Era5DownloadConfig,
    Era5Site,
    build_requests,
    download_requests,
)

TccFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def write_era5_like(
    path: Path,
    times_posix_s: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    tcc: np.ndarray,
    *,
    expver: str | None = "0001",
) -> None:
    """Write a CDS-layout NetCDF-4 file (h5netcdf engine)."""
    import xarray as xr

    t = np.asarray(times_posix_s, dtype=np.int64).astype("datetime64[s]").astype("datetime64[ns]")
    coords: dict[str, Any] = {
        "valid_time": t,
        "latitude": np.asarray(lat, dtype=np.float64),
        "longitude": np.asarray(lon, dtype=np.float64),
    }
    if expver is not None:  # the gridded product carries expver, the time series does not
        coords["expver"] = ("valid_time", np.full(t.shape, expver))
    ds = xr.Dataset(
        {"tcc": (("valid_time", "latitude", "longitude"), np.asarray(tcc, dtype=np.float32))},
        coords=coords,
    )
    ds.to_netcdf(path, engine="h5netcdf")


class FakeSubmitted:
    """A submitted request (``request_id`` + ``download``)."""

    def __init__(self, request: dict[str, Any], tcc: TccFunction, counter: list[int]) -> None:
        self.request = request
        self.tcc = tcc
        counter[0] += 1
        self.request_id = f"fake-{counter[0]:04d}"

    def download(self, target: str | None = None) -> str:
        assert target is not None
        n, w, s, e = self.request["area"]
        if "date" in self.request:  # time-series product: date range, zip, ascending latitude
            a, b = self.request["date"][0].split("/")
            start = int(dt.datetime.fromisoformat(a).replace(tzinfo=dt.UTC).timestamp())
            end = int(dt.datetime.fromisoformat(b).replace(tzinfo=dt.UTC).timestamp()) + 82800
            t = np.arange(start, end + 1, 3600, dtype=np.int64)
            lat = np.arange(s, n + 1e-9, 0.25)
        else:
            year = int(self.request["year"][0])
            months = [int(m) for m in self.request["month"]]
            start = int(dt.datetime(year, months[0], 1, tzinfo=dt.UTC).timestamp())
            hours = sum(24 * calendar.monthrange(year, m)[1] for m in months)
            t = start + 3600 * np.arange(hours, dtype=np.int64)
            lat = np.arange(n, s - 1e-9, -0.25)
        lon = np.arange(w, e + 1e-9, 0.25)
        tt, la, lo = np.meshgrid(t, lat, lon, indexing="ij")
        if "date" in self.request:
            inner = Path(target).with_name(Path(target).name + ".member.nc")
            write_era5_like(inner, t, lat, lon, self.tcc(tt, la, lo), expver=None)
            with zipfile.ZipFile(target, "w") as zf:
                zf.write(inner, arcname="reanalysis-era5-single-levels-timeseries-sfcFAKE.nc")
            inner.unlink()
        else:
            write_era5_like(Path(target), t, lat, lon, self.tcc(tt, la, lo))
        return target


class FakeCdsClient:
    """Counts submitted requests; never touches the network."""

    def __init__(self, tcc: TccFunction) -> None:
        self.tcc = tcc
        self.counter = [0]
        self.requests: list[dict[str, Any]] = []

    def retrieve(self, name: str, request: dict[str, Any]) -> FakeSubmitted:
        assert name in ("reanalysis-era5-single-levels", "reanalysis-era5-single-levels-timeseries")
        assert ("date" in request) == name.endswith("timeseries")
        self.requests.append(request)
        return FakeSubmitted(request, self.tcc, self.counter)


def make_weather_dir(
    directory: Path,
    sites: list[tuple[str, float, float]],
    first_year: int,
    last_year: int,
    tcc: TccFunction,
    *,
    dataset: str = "reanalysis-era5-single-levels-timeseries",
) -> FakeCdsClient:
    """Populate ``directory`` with synthetic ERA5 tcc files + provenance for ``sites``."""
    cfg = Era5DownloadConfig.model_validate(
        {
            "era5_schema_version": 1,
            "dataset": dataset,
            "variables": ["total_cloud_cover"],
            "first_year": first_year,
            "last_year": last_year,
            "sites": [
                Era5Site(name=n, latitude_deg=la, longitude_deg=lo, origin="test")
                for n, la, lo in sites
            ],
        },
        strict=False,
    )
    client = FakeCdsClient(tcc)
    download_requests(build_requests(cfg), directory, client_factory=lambda: client)
    return client
