"""ERA5 request builder, downloader cache/provenance and reader (helioswarm.era5; M8a).

No test touches the network: downloads go through :class:`era5_helpers.FakeCdsClient`, and the
real-data checks use the committed one-month fixture ``tests/data/era5`` (README there).
"""

from __future__ import annotations

import calendar
import datetime as dt
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from helioswarm.era5 import (
    CDS_COST_LIMIT,
    DATASET,
    DATASET_TIMESERIES,
    VARIABLE_SHORT_NAMES,
    Era5DownloadConfig,
    Era5Error,
    Era5FileProvenance,
    Era5Request,
    GridNode,
    build_requests,
    check_block,
    download_requests,
    file_sha256,
    grid_node,
    hours_in_months,
    load_download_config,
    load_node_series,
    month_blocks,
    provenance_path,
    read_era5_netcdf,
    read_provenance,
    request_area,
    request_cost,
    site_series,
    verify_cached,
)

from config_helpers import CONFIG_DIR, REPO_ROOT
from era5_helpers import FakeCdsClient, make_weather_dir

pytest.importorskip("xarray")

FIXTURE = REPO_ROOT / "tests" / "data" / "era5" / "N23.50_E000.00_k1_tcc_2014_m02-02.nc"


def _cfg(**kw: object) -> Era5DownloadConfig:
    base: dict[str, object] = {
        "era5_schema_version": 1,
        "dataset": DATASET,
        "variables": ["total_cloud_cover"],
        "first_year": 2015,
        "last_year": 2016,
        "sites": [{"name": "a", "latitude_deg": 23.4, "longitude_deg": 0.0, "origin": "t"}],
    }
    base.update(kw)
    return Era5DownloadConfig.model_validate(base, strict=False)


# ---- grid and area -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lat", "lon", "node_lat", "node_lon"),
    [
        (23.4, 0.0, 23.5, 0.0),  # 23.4 is 0.1 from 23.5, 0.15 from 23.25
        (-23.4, 120.0, -23.5, 120.0),
        (33.0, -120.0, 33.0, -120.0),  # on a node
        (0.125, 0.125, 0.25, 0.25),  # exact tie: half away from zero
        (-0.125, -0.125, -0.25, -0.25),
        (10.0, 190.0, 10.0, -170.0),  # longitude wrapped into [-180, 180)
        (10.0, 179.9, 10.0, -180.0),  # nearest node of 179.9 is 180 == -180
    ],
)
def test_grid_node_snaps_to_nearest_quarter_degree(
    lat: float, lon: float, node_lat: float, node_lon: float
) -> None:
    node = grid_node(lat, lon)
    assert node.latitude_deg == node_lat
    assert node.longitude_deg == node_lon
    # Never further than half a cell (0.125 deg) from the site, modulo the longitude wrap.
    assert abs(node.latitude_deg - lat) <= 0.125 + 1e-12
    assert abs(((node.longitude_deg - lon + 180.0) % 360.0) - 180.0) <= 0.125 + 1e-12


def test_node_key_and_request_area() -> None:
    node = grid_node(-23.4, -70.4)
    assert node.key == "S23.50_W070.50"
    assert request_area(node, 1) == (-23.25, -70.75, -23.75, -70.25)  # [N, W, S, E]
    assert request_area(node, 2) == (-23.0, -71.0, -24.0, -70.0)
    with pytest.raises(Era5Error, match="antimeridian"):
        request_area(GridNode(lat_index=0, lon_index=-720), 1)  # lon -180: west edge -180.25


# ---- request sizing ----------------------------------------------------------------------------


def test_hours_in_months_counts_leap_years() -> None:
    assert hours_in_months(2015, range(1, 13)) == 8760
    assert hours_in_months(2016, range(1, 13)) == 8784
    assert hours_in_months(2016, (2,)) == 29 * 24
    assert hours_in_months(2100, (2,)) == 28 * 24  # 2100 is not a leap year (Gregorian)


def test_request_cost_reproduces_the_measured_cds_costs() -> None:
    # Values returned by the CDS costing endpoint on 2026-11-13 (era5.CDS_COST_PER_FIELD doc).
    assert request_cost(1, 2019, range(1, 13)) == 52560
    assert request_cost(1, 2020, range(1, 13)) == 52704
    assert request_cost(6, 2019, (1,)) == 26784
    assert request_cost(6, 2019, (1, 2, 3, 4)) == 103680
    assert request_cost(2, 2019, range(1, 13)) == 105120


def test_month_blocks_fit_the_limit_and_are_minimal() -> None:
    assert month_blocks(2015, 1, CDS_COST_LIMIT) == (tuple(range(1, 13)),)
    blocks = month_blocks(2016, 6, CDS_COST_LIMIT)
    assert [m for b in blocks for m in b] == list(range(1, 13))  # consecutive, complete
    assert all(request_cost(6, 2016, b) <= CDS_COST_LIMIT for b in blocks)
    # Each block is maximal: adding the next month would exceed the limit.
    for b, nxt in itertools.pairwise(blocks):
        assert request_cost(6, 2016, (*b, nxt[0])) > CDS_COST_LIMIT
    assert len(blocks) == 3  # 6 x 6 x 8784 / 121000 = 2.6 -> at least 3
    with pytest.raises(Era5Error, match="one month"):
        month_blocks(2015, 6, 10_000)


def test_build_requests_payload_and_site_deduplication() -> None:
    cfg = _cfg(
        sites=[
            {"name": "a", "latitude_deg": 23.4, "longitude_deg": 0.0, "origin": "t"},
            {"name": "b", "latitude_deg": 23.45, "longitude_deg": 0.05, "origin": "t"},  # same node
            {"name": "c", "latitude_deg": -23.4, "longitude_deg": 120.0, "origin": "t"},
        ]
    )
    reqs = build_requests(cfg)
    assert len(reqs) == 2 * 2  # two nodes x two years, one block each
    assert [r.node.key for r in reqs] == ["S23.50_E120.00"] * 2 + ["N23.50_E000.00"] * 2
    assert reqs[2].site_names == ("a", "b")
    d = reqs[3].as_cds_dict()
    assert d == {
        "product_type": ["reanalysis"],
        "variable": ["total_cloud_cover"],
        "year": ["2016"],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{x:02d}" for x in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [23.75, -0.25, 23.25, 0.25],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    assert reqs[3].n_hours == 8784
    assert reqs[3].period_start_utc == "2016-01-01T00:00:00Z"
    assert reqs[3].period_end_utc == "2016-12-31T23:00:00Z"
    assert str(reqs[3].relative_path) == (
        f"era5/{DATASET}/N23.50_E000.00/N23.50_E000.00_k1_tcc_2016_m01-12.nc"
    )
    # The request hash identifies the payload: same request -> same hash, any change -> new.
    assert reqs[3].request_sha256 == build_requests(cfg)[3].request_sha256
    assert reqs[3].request_sha256 != reqs[2].request_sha256


def test_download_config_validation_and_shipped_file() -> None:
    with pytest.raises(ValueError, match="unknown variables"):
        _cfg(variables=["cloud"])
    with pytest.raises(ValueError, match="last_year"):
        _cfg(first_year=2020, last_year=2019)
    with pytest.raises(ValueError, match="duplicate site"):
        _cfg(
            sites=[
                {"name": "a", "latitude_deg": 0.0, "longitude_deg": 0.0, "origin": "t"},
                {"name": "a", "latitude_deg": 1.0, "longitude_deg": 0.0, "origin": "t"},
            ]
        )
    shipped = load_download_config(CONFIG_DIR / "weather" / "era5_sites.yaml")
    assert shipped.variables == ("total_cloud_cover",)
    assert (shipped.first_year, shipped.last_year) == (2014, 2023)
    # Every receiver site of the shipped run configurations is in the download list.
    from helioswarm.config import load_config

    have = {(s.latitude_deg, s.longitude_deg) for s in shipped.sites}
    for name in ("mvp_single_satellite.yaml", "constellation_demo.yaml"):
        for site in load_config(CONFIG_DIR / name).receiver.sites:
            assert (site.latitude_deg, site.longitude_deg) in have
    assert set(VARIABLE_SHORT_NAMES) >= set(shipped.variables)


# ---- real fixture --------------------------------------------------------------------------------


def _fixture_request() -> Era5Request:
    node = grid_node(23.4, 0.0)
    return Era5Request(
        dataset=DATASET,
        product_type="reanalysis",
        node=node,
        neighbourhood_nodes=1,
        area=request_area(node, 1),
        year=2014,
        months=(2,),
        variables=("total_cloud_cover",),
        site_names=("site_lon000",),
    )


def test_real_fixture_reads_and_passes_the_content_checks() -> None:
    block = read_era5_netcdf(FIXTURE, ["tcc"])
    assert block.time_utc_posix_s.size == 28 * 24
    assert block.time_utc_posix_s[0] == int(dt.datetime(2014, 2, 1, tzinfo=dt.UTC).timestamp())
    assert np.all(np.diff(block.time_utc_posix_s) == 3600)
    np.testing.assert_array_equal(block.latitude_deg, [23.75, 23.5, 23.25])
    np.testing.assert_array_equal(block.longitude_deg, [-0.25, 0.0, 0.25])
    assert block.expver == ("0001",)  # final ERA5, not ERA5T
    tcc = block.values["tcc"]
    assert tcc.shape == (672, 3, 3)
    assert tcc.min() >= 0.0
    assert tcc.max() <= 1.0
    check_block(block, _fixture_request())
    wrong_month = Era5Request(**{**_fixture_request().__dict__, "months": (3,)})
    with pytest.raises(Era5Error, match="time axis"):
        check_block(block, wrong_month)
    node = grid_node(23.0, 0.0)
    wrong_area = Era5Request(
        **{**_fixture_request().__dict__, "node": node, "area": request_area(node, 1)}
    )
    with pytest.raises(Era5Error, match="latitude"):
        check_block(block, wrong_area)


def test_real_fixture_provenance_schema_and_checksum(tmp_path: Path) -> None:
    prov = read_provenance(FIXTURE)
    assert isinstance(prov, Era5FileProvenance)
    assert prov.dataset_doi == "10.24381/cds.adbb2d47"
    assert prov.licence_spdx == "CC-BY-4.0"
    assert "Copernicus Climate Change Service information" in prov.attribution
    assert "10.1002/qj.3803" in prov.citation
    assert prov.request == _fixture_request().as_cds_dict()
    assert prov.request_sha256 == _fixture_request().request_sha256
    assert prov.cds_request_id
    assert prov.retrieved_utc.endswith("Z")
    assert prov.file_sha256 == file_sha256(FIXTURE)
    assert prov.file_size_bytes == FIXTURE.stat().st_size
    assert prov.n_hours == 672
    assert prov.variable_short_names == ("tcc",)
    # Cache verification: in place -> verified; a modified byte -> not verified.
    target = tmp_path / _fixture_request().relative_path
    target.parent.mkdir(parents=True)
    shutil.copy(FIXTURE, target)
    shutil.copy(provenance_path(FIXTURE), provenance_path(target))
    assert verify_cached(target, _fixture_request())
    other = Era5Request(**{**_fixture_request().__dict__, "months": (3,)})
    assert not verify_cached(target, other)  # provenance of another request
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    assert not verify_cached(target, _fixture_request())
    provenance_path(target).write_text("{}")
    assert not verify_cached(target, _fixture_request())


# ---- downloader ----------------------------------------------------------------------------------


def _tcc_hour_pattern(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """tcc = (hour of day)/23 + small lat/lon gradient, clipped: a known, non-constant field."""
    hour = (t // 3600) % 24
    return np.clip(hour / 23.0 + 0.0 * lat + 0.0 * lon, 0.0, 1.0)


def test_downloader_writes_files_and_provenance_and_is_idempotent(tmp_path: Path) -> None:
    client = make_weather_dir(
        tmp_path, [("a", 23.4, 0.0)], 2015, 2016, _tcc_hour_pattern, dataset=DATASET
    )
    assert client.counter[0] == 2
    cfg = _cfg()
    reqs = build_requests(cfg)
    for r in reqs:
        f = tmp_path / r.relative_path
        prov = read_provenance(f)
        assert prov.file_sha256 == file_sha256(f)
        assert prov.cds_request_id is not None
        assert prov.cds_request_id.startswith("fake-")
        assert prov.request == r.as_cds_dict()
        assert prov.n_hours == r.n_hours
        assert json.loads(provenance_path(f).read_text())["site_names"] == ["a"]
    # Second call: everything verified, nothing submitted.
    again = FakeCdsClient(_tcc_hour_pattern)
    out = download_requests(reqs, tmp_path, client_factory=lambda: again)
    assert [o.action for o in out] == ["present", "present"]
    assert again.counter[0] == 0
    # A corrupted file is downloaded again (and only that one).
    f0 = tmp_path / reqs[0].relative_path
    f0.write_bytes(f0.read_bytes()[:-10])
    out = download_requests(reqs, tmp_path, client_factory=lambda: again)
    assert [o.action for o in out] == ["downloaded", "present"]
    assert again.counter[0] == 1


def test_downloader_rejects_wrong_content_and_leaves_no_file(tmp_path: Path) -> None:
    class WrongYear(FakeCdsClient):
        def retrieve(self, name: str, request: dict[str, object]):  # type: ignore[no-untyped-def]
            bad = dict(request)
            bad["year"] = ["2014"]  # the server returns another year
            return super().retrieve(name, bad)

    reqs = build_requests(_cfg(first_year=2015, last_year=2015))
    client = WrongYear(_tcc_hour_pattern)
    with pytest.raises(Era5Error, match="time axis"):
        download_requests(reqs, tmp_path, client_factory=lambda: client)
    node_dir = (tmp_path / reqs[0].relative_path).parent
    assert list(node_dir.iterdir()) == []  # no final file, no provenance, no .part left


# ---- reading the cache back ----------------------------------------------------------------------


def test_load_node_series_concatenates_whole_years_incl_leap_year(tmp_path: Path) -> None:
    make_weather_dir(tmp_path, [("a", 23.4, 0.0)], 2015, 2016, _tcc_hour_pattern, dataset=DATASET)
    ns = load_node_series(
        tmp_path, grid_node(23.4, 0.0), "total_cloud_cover", [2015, 2016], dataset=DATASET
    )
    t = ns.block.time_utc_posix_s
    assert t.size == 8760 + 8784
    assert t[0] == int(dt.datetime(2015, 1, 1, tzinfo=dt.UTC).timestamp())
    assert np.all(np.diff(t) == 3600)
    hour = (t // 3600) % 24
    np.testing.assert_allclose(ns.block.values["tcc"][:, 1, 1], hour / 23.0, atol=1e-7)
    assert len(ns.files) == 2
    assert ns.cds_request_ids[0] is not None
    assert ns.block.expver == ("0001",)
    with pytest.raises(Era5Error, match="gap-free"):
        load_node_series(
            tmp_path, grid_node(23.4, 0.0), "total_cloud_cover", [2015, 2016, 2017], dataset=DATASET
        )
    with pytest.raises(Era5Error, match="no ERA5"):
        load_node_series(
            tmp_path, grid_node(0.0, 0.0), "total_cloud_cover", [2015], dataset=DATASET
        )
    with pytest.raises(Era5Error, match="no ERA5"):  # the time-series collection is separate
        load_node_series(tmp_path, grid_node(23.4, 0.0), "total_cloud_cover", [2015])
    # A file whose bytes changed after download is refused.
    f = ns.files[0]
    f.write_bytes(f.read_bytes()[:-1] + b"\x00")
    with pytest.raises(Era5Error, match=r"size|SHA-256"):
        load_node_series(
            tmp_path, grid_node(23.4, 0.0), "total_cloud_cover", [2015, 2016], dataset=DATASET
        )


def _linear_field(t: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # tcc linear in latitude and longitude (constant in time) inside [0, 1] on the 3x3 box.
    return 0.5 + 0.4 * (lat - 23.5) + 0.8 * (lon - 0.0)


def test_site_series_nearest_and_bilinear(tmp_path: Path) -> None:
    make_weather_dir(tmp_path, [("a", 23.4, 0.0)], 2015, 2015, _linear_field)
    ns = load_node_series(tmp_path, grid_node(23.4, 0.0), "total_cloud_cover", [2015])
    lat, lon = 23.4, 0.1
    nearest = site_series(ns, lat, lon, "nearest")
    bilinear = site_series(ns, lat, lon, "bilinear")
    # nearest: the node value 0.5; bilinear reproduces a linear field exactly (float32 storage:
    # tolerance 1e-6).
    np.testing.assert_allclose(nearest, 0.5, atol=1e-6)
    np.testing.assert_allclose(bilinear, 0.5 + 0.4 * (lat - 23.5) + 0.8 * lon, atol=1e-6)
    assert not np.allclose(nearest, bilinear)
    with pytest.raises(Era5Error, match="snaps to"):
        site_series(ns, 24.0, 0.0, "nearest")


def test_calendar_assumption_of_fake_matches_cds_days() -> None:
    # The builder lists days 01-31 for every month; the CDS ignores impossible dates. The fake
    # (and check_block) count the real days, which is what the measured cost shows.
    assert sum(calendar.monthrange(2016, m)[1] for m in range(1, 13)) * 24 == 8784


# ---- time-series product -------------------------------------------------------------------------


def test_timeseries_request_payload_one_request_per_node() -> None:
    cfg = _cfg(dataset=DATASET_TIMESERIES, first_year=2014, last_year=2023)
    reqs = build_requests(cfg)
    assert len(reqs) == 1
    r = reqs[0]
    assert r.as_cds_dict() == {
        "variable": ["total_cloud_cover"],
        "area": [23.75, -0.25, 23.25, 0.25],
        "date": ["2014-01-01/2023-12-31"],
        "data_format": "netcdf",
    }
    assert r.n_hours == 87648  # 10 years incl. the leap years 2016 and 2020
    assert r.period_end_utc == "2023-12-31T23:00:00Z"
    assert str(r.relative_path).endswith("N23.50_E000.00_k1_tcc_2014-2023.nc")
    # k <= 4 (schema) keeps the box within the product's 2 x 2 deg area limit.
    assert 2 * 4 * 0.25 <= 2.0
    with pytest.raises(ValueError, match="one year"):
        Era5Request(**{**r.__dict__, "dataset": DATASET})


def test_timeseries_zip_is_unpacked_and_latitude_order_normalised(tmp_path: Path) -> None:
    client = make_weather_dir(tmp_path, [("a", 23.4, 0.0)], 2015, 2016, _linear_field)
    assert client.counter[0] == 1  # both years in one request
    r = build_requests(_cfg(dataset=DATASET_TIMESERIES))[0]
    prov = read_provenance(tmp_path / r.relative_path)
    assert prov.dataset_doi == "10.24381/1cf1ad76"
    assert prov.archive_member is not None
    assert prov.archive_member.endswith(".nc")
    assert prov.archive_sha256 is not None
    assert prov.archive_sha256 != prov.file_sha256
    ns = load_node_series(tmp_path, grid_node(23.4, 0.0), "total_cloud_cover", [2016])
    assert ns.block.time_utc_posix_s.size == 8784  # one year sliced from the 2-year file
    np.testing.assert_array_equal(ns.block.latitude_deg, [23.75, 23.5, 23.25])  # descending
    # The flipped field still has the north row first: tcc rises with latitude (+0.4 / deg).
    v = ns.block.values["tcc"][0]
    assert v[0, 1] > v[1, 1] > v[2, 1]
    np.testing.assert_allclose(site_series(ns, 23.4, 0.1, "bilinear"), 0.5 - 0.04 + 0.08, atol=1e-6)


def test_archive_with_two_members_is_refused(tmp_path: Path) -> None:
    import zipfile

    from helioswarm.era5 import _unpack_single_netcdf

    raw = tmp_path / "x.zip"
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr("a.nc", b"1")
        zf.writestr("b.nc", b"2")
    with pytest.raises(Era5Error, match="2 files"):
        _unpack_single_netcdf(raw, tmp_path / "out.nc")
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr("../evil.nc", b"1")
    with pytest.raises(Era5Error, match="unexpected archive member"):
        _unpack_single_netcdf(raw, tmp_path / "out.nc")
