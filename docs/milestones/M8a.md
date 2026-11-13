# M8a: ERA5 site weather and parameterised economics (LCOE/NPV)

Status: implemented on branch `helioswarm-m8a` (worktree `/n/home05/ianpoon/Project-Dyson-Sphere-m8a`,
based on the M7 commit `e86f8e7`), in parallel with M7b (GEO baseline) in the main checkout.
All gates pass (D). The ERA5 download and every long run were Slurm jobs of this worktree
(`HELIOSWARM_REPO` + `HELIOSWARM_ENV_OVERLAY`; each job log prints `helioswarm from
/n/home05/ianpoon/Project-Dyson-Sphere-m8a/src/helioswarm/__init__.py`). The runs were made on
the uncommitted M8a tree (manifests record `git.dirty = true`). A separate step commits.

PDF basis: §10.2 ("Later, use site-specific cloud and atmospheric data to estimate operational
availability. Cloud obstruction can be a hard outage for a ground optical receiver."), §14.1
(record the weather-data product, retrieval date, variables, resolution, licence/attribution),
§14.2 (NetCDF only for gridded weather), §12.5 (economics as a separate parameterised module;
compare scenarios and uncertainty; never a single speculative LCOE), §17 ("Economic
overclaim"), §19 (ERA5 and the NASA SBSP assessment as references; the NASA reference designs
are not used as designs, but two of its system-level cost numbers are transferred to these runs
as stated, `assumed` proxies: the rectenna/ground-segment cost per m² for the optical receiver
and the operations cost of one GW-class system; F and Review outcomes).

## A. Summary

### Part A: ERA5 site weather

- **Data** (`src/helioswarm/era5.py`). Pure request builder (sites → nearest 0.25° node, 3 × 3
  node box `[N, W, S, E]`; per-node requests; single-levels requests chunked by year and month
  block under the measured CDS cost limit), idempotent downloader (`cdsapi` 0.7.7; temporary
  file → content check → SHA-256 → rename; provenance JSON per file with collection, DOI,
  citation, variables, area, period, the request and its SHA-256, CDS request id, retrieval UTC,
  cdsapi / datastores versions, file and archive SHA-256, licence and attribution; verified
  files are skipped), reader (xarray + h5netcdf; canonical descending latitude), cache loader
  (re-verifies SHA-256, slices whole years, requires a gap-free hourly axis), nearest/bilinear
  site values. CLI `helioswarm weather download|climatology`. New env var
  `HELIOSWARM_WEATHER_DIR` (`scripts/cluster_env.sh` → netscratch).
- **Product switch (deviation, documented).** The first bulk job used the gridded
  `reanalysis-era5-single-levels` product (one request per site-year): ~13 min of CDS queue per
  request, i.e. ~20 h for 120 requests. It was cancelled after 23 min (2 files) and the
  download moved to `reanalysis-era5-single-levels-timeseries` (the same ERA5 analyses,
  re-arranged for time series; DOI 10.24381/1cf1ad76), which accepts the same 3 × 3 area and the
  whole decade in one request (~1 min). `scripts/era5_product_crosscheck.py`: at both nodes with
  gridded 2014 files every one of the 3 × 3 × 8760 `tcc` values is identical between the two
  products (maximum difference 0). Both collections are supported (`weather.era5_dataset`).
- **Download** (job 60341911, 2 min 38 s, 12 sites × 2014–2023, 45 MB on netscratch).
- **Availability model** (`src/helioswarm/weather.py`; `docs/model_assumptions.md` M8a).
  `models.weather_source: era5` + `weather` section: arrival epoch (TDB) → UTC with SPICE
  `deltet` (explicit; checked against `et2utc`, incl. a leap second) → nearest hour → same
  calendar month/day/hour of a historical year (`fixed` cyclic or `sampled`, seeded; 29 Feb →
  28 Feb) → site `tcc` → hard outage (`threshold`: `tcc > c`; `bernoulli`: `u < tcc`, clear
  probability `1 − tcc`). Applied in `system.atmosphere_at_access`, i.e. in the single-satellite
  run and in every constellation stage-A pair (array tasks build identical weather). Sites share
  the historical hour (real spatial correlation kept). Default `scenario` reproduces every
  earlier result and hash (D). Weather provenance is written to `weather.json` in each run and
  listed in the manifest; the named approximation ("tcc is a grid-box fraction, not the line of
  sight") is in the run log and manifest.
- **Climatology and joint statistics** (`helioswarm weather climatology`): clear fraction by
  month, UTC hour and local mean solar hour, per-year spread, longest/mean cloudy spell, the
  Bernoulli expectation `1 − mean(tcc)`, and for a network the all-cloudy fraction vs the
  independence product and pairwise outage correlations.
- **Example receiver sites chosen from the data** (`configs/constellation_demo_era5_clear_sites.yaml`):
  highest clear fraction per longitude sector — atacama (24.6 S 70.4 W, 0.847), aswan
  (23.9 N 32.9 E, 0.889), site_lon120e = "pilbara" (23.4 S 120 E, 0.737).
- **Microwave**: not coupled (ITU-R statistics stay); the builder accepts the cloud-liquid-water
  and water-vapour columns for a later P.840/P.676 coupling.

### Part B: economics

- **Module** `src/helioswarm/economics.py`: `EconomicsConfig` (every input a `CostParameter`
  with `provenance` sourced/derived/assumed and a page-level source), mass from the run's M7
  proxy or specific masses, transport `c_LEO m exp(Δv/(g0 Isp))` with Δv explicit (GEO) or
  impulsive patched conics computed per satellite from its own orbit (ring overrides) and slot
  phase (Hohmann + free phase drift within the deployment time budget, plus a phasing ellipse
  for the residual phase; exactly the co-orbital phasing transfer at 1 au), CAPEX
  (manufacturing, transport, transmitter, technology-specific receiver), construction instalments, degradation, hardware
  replacement with performance renewal, opex, CRF, LCOE (USD/kWh and USD/MWh), NPV, equivalent
  annual cost; zero energy → LCOE undefined (`zero_energy`); near-zero (< 1 MWh/yr) →
  undefined (`negligible_energy`, formal ratio kept for diagnosis); seeded Monte Carlo
  (triangular low/mode/high from the scenarios). CLI `helioswarm economics RUN_OR_SWEEP_DIR
  --config ... [--out]` → `economics.json` + `economics.md`, from saved files only (kernels are
  loaded only to resolve the Earth-relative phase of absolute-phase orbits).
- **Scenarios** `configs/economics/{optimistic,nominal,conservative}.yaml` (+ `geo_nominal.yaml`
  with the explicit LEO→GEO Δv for the M7b comparison after the merge). Sourced values from NASA
  OTPS 2024 (fetched; SHA-256 and PDF pages in `docs/data_provenance.md`).

### Results (real runs; one simulated year 2030; all design values assumed)

**ERA5 climatology** (`$HELIOSWARM_RUNS_DIR/era5_climatology/`; 2014–2023, nearest node; clear
iff `tcc ≤ c`):

| Site | lat, lon | clear c = 0.1 | c = 0.3 | c = 0.5 (min–max year) | 1 − mean tcc | longest cloudy spell at 0.5 [h] |
|---|---|---|---|---|---|---|
| example_site (MVP) | 0, 0 | 0.023 | 0.166 | 0.382 (0.325–0.431) | 0.365 | 213 |
| site_lon000 (demo) | 23.4, 0 | 0.641 | 0.734 | 0.793 (0.737–0.831) | 0.778 | 112 |
| site_lon120e (demo, "pilbara") | −23.4, 120 | 0.535 | 0.648 | 0.737 (0.678–0.765) | 0.720 | 199 |
| site_lon120w (demo) | 33, −120 | 0.353 | 0.448 | 0.522 (0.478–0.618) | 0.514 | 198 |
| atacama | −24.6, −70.4 | 0.651 | 0.764 | 0.847 (0.790–0.903) | 0.815 | 117 |
| sonora_arizona | 32.0, −111.6 | 0.500 | 0.608 | 0.693 (0.619–0.751) | 0.681 | 114 |
| hawaii_summit (grid box) | 19.8, −155.5 | 0.185 | 0.398 | 0.525 (0.404–0.578) | 0.501 | 205 |
| aswan | 23.9, 32.9 | 0.800 | 0.855 | 0.889 (0.849–0.913) | 0.881 | 78 |
| namibia_highland | −23.3, 16.5 | 0.624 | 0.733 | 0.813 (0.778–0.836) | 0.792 | 148 |
| rub_al_khali | 21.0, 50.0 | 0.688 | 0.787 | 0.847 (0.784–0.881) | 0.827 | 145 |
| ladakh | 32.8, 79.0 | 0.347 | 0.491 | 0.588 (0.513–0.653) | 0.585 | 161 |
| south_australia | −29.0, 134.75 | 0.553 | 0.651 | 0.722 (0.670–0.798) | 0.711 | 110 |

Bilinear instead of nearest changes the c = 0.5 clear fraction by ≤ 0.011 (atacama). At c = 0.5
the clear fraction is within 0.032 of `1 − mean(tcc)` at every site (the basis of the default
threshold, A of `docs/model_assumptions.md` M8a). Joint statistics (c = 0.5): demo network
all-cloudy 0.0298 of the hours (0.0260 if independent; pairwise outage correlations −0.026 to
0.024); ERA5-chosen network 0.0069 (0.0045 if independent; correlations 0.003–0.051). The
"hawaii_summit" value is a 28 km box average, not the summit sky (limitation F).

**ERA5 re-runs** (weather of each historical year 2014–2023 mapped onto 2030; threshold 0.5,
nearest node):

| Run | Clear sky (scenario) | ERA5, 10 weather years: P_grid mean (min–max) | ratio to clear | availability (min–max) | outages / longest |
|---|---|---|---|---|---|
| MVP optical, example_site (0 N 0 E) — `mvp_era5_years_sweep` (job 60342908) | 1751.1 W, availability 0.4221 | 735.8 W (653.5–844.9) | 0.37–0.48 | 0.157–0.202 (mean 0.176) | 345–397 / 117–235 h |
| Constellation demo N = 10, demo sites — `constellation_demo_era5_years_sweep` (job 60343733) | 44.21 kW (M7 sweep p0004), availability 1.000 | 31.69 kW (30.14–32.67) | 0.68–0.74 | 0.907–0.948 (mean 0.936) | 150–227 / 11.9–41.1 h |
| Constellation demo N = 10, ERA5-chosen sites — `constellation_demo_era5_clear_sites_years_sweep` (job 60343742) | 42.97 kW (`constellation_demo_clear_sites`, job 60344241), availability 1.000 | 36.62 kW (35.25–37.59) | 0.82–0.88 | 0.971–0.989 (mean 0.983) | 42–87 / 4.3–19.5 h |

Findings, stated plainly:

- **Weather is a first-order term for optical delivery.** The MVP receiver at 0 N 0 E (a
  scenario location in the Gulf of Guinea) is clear only 38 % of the hours; the satellite's
  delivered energy falls to 37–48 % of the clear-sky value and its availability from 0.42 to
  0.16–0.20, depending on the weather year.
- **Site diversity and site choice.** With three sites ~120° apart the demo network keeps
  0.91–0.95 availability under weather (no site alone exceeds 0.62), but delivers only 68–74 %
  of its clear-sky energy; the satellites' weather-blocked share of satellite-time is
  0.21–0.27. Moving the three receivers to the clearest candidates of each longitude sector
  (chosen from the data, not from prior claims) costs 2.8 % of the clear-sky power (geometry)
  but raises the weather-limited delivery to 82–88 % of clear sky, +15.6 % energy and
  availability 0.97–0.99 (outages 42–87 per year, longest ≤ 19.5 h).
- **Year-to-year spread** (aleatory): −11 % to +15 % of the MVP's mean delivered power over
  the ten weather years, −5 % to +3 % for the two networks.
- **Correlation**: all-sites-cloudy hours occur somewhat more often than independence predicts
  (0.0298 vs 0.0260; 0.0069 vs 0.0045), i.e. weather is weakly positively correlated across
  these continents; the model keeps the measured correlation.

**Economics** (`$HELIOSWARM_RUNS_DIR/economics_m8a/*`; USD per kWh delivered to the grid;
30-year system, scenarios optimistic / nominal / conservative; MC = nominal Monte Carlo 5–95 %
over the varied inputs, 2 000 samples; ERA5 rows: lowest p05 – highest p95 over the ten weather
years (median of the p50s). Recomputed after the review fixes: scenario values unchanged to
1e-9, Monte Carlo intervals shifted by ≤ 5 % — the discount-rate range now starts at the
optimistic 0.03 instead of 0.02 and the optical-receiver input adds one draw stream):

| Run | E₁ [kWh/yr] | Mass [t] | mean Δv [km/s] | CAPEX nominal [USD] (mfg / transport / receiver; transmitter 0) | LCOE opt / nom / cons | MC p05–p95 (p50) |
|---|---|---|---|---|---|---|
| MVP optical, clear sky (`mvp_single_satellite_m8a_check`) | 1.535e4 | 16.6 | 4.10 | 4.13e8 (3.3e7 / 3.6e7 / 3.4e8) | 1.25e3 / 2.67e3 / 6.65e3 | 2.25e3–3.75e3 (2.97e3) |
| MVP optical, ERA5 2014–2023 | 5.9e3 (2014) | 16.6 | 4.10 | same | 2.60e3–3.36e3 / 5.53e3–7.15e3 / 1.38e4–1.78e4 | 4.67e3–1.00e4 (7.11e3) |
| MVP microwave (M6 run) | 2.76 | 39 300 | 4.10 | 1.97e11 | **undefined** (`negligible_energy`) | — |
| Demo N = 10, clear sky (M7 p0004) | 3.876e5 | 166 | 4.89 | 1.81e9 (3.3e8 / 4.4e8 / 1.0e9) | 93 / 417 / 1.75e3 | 356–807 (546) |
| Demo N = 10, ERA5 2014–2023 | 2.84e5 (2014) | 166 | 4.89 | same | 126–137 / 564–611 / 2.37e3–2.57e3 | 482–1.18e3 (760) |
| ERA5-chosen sites, clear sky | 3.767e5 | 166 | 4.89 | same | 96 / 429 / 1.80e3 | 366–830 (562) |
| ERA5-chosen sites, ERA5 2014–2023 | 3.27e5 (2014) | 166 | 4.89 | same | 110–117 / 490–523 / 2.06e3–2.20e3 | 419–1.01e3 (655) |
| N = 1000 (M7 merge) | 2.494e6 | 16 620 | 4.80 | 7.7e10 (3.3e10 / 4.3e10 / 1.0e9) | 436 / 3.70e3 / 2.16e4 | 3.08e3–9.77e3 (5.73e3) |
| N = 1–100 (M7 sweep, 12 points) | 2.7e4–1.5e6 | 16.6–1662 | 3.6–4.9 | — | 86–988 / 415–2.89e3 / 973–6.81e3 | — |

Read with PDF §12.5 in mind: these are **scenario ranges for an assumed technology**, not cost
predictions. All cases lie 3–6 orders of magnitude above the 0.05 USD/kWh reference price
(NPV strongly negative in every scenario). What they show robustly: (i) for the MVP the 1 km
receiver and fixed operations dominate — both are **assumed transfers** of NASA microwave-SBSP
numbers (rectenna/ground-segment cost per m² as an optical-receiver proxy; the operations cost
of one GW-class system), so the MVP and demo LCOE levels inherit that proxy (F); (ii) for the
constellation, transport is the largest space-segment term (mass ratio ~3.1 from the ~4.9 km/s
LEO→co-orbital-slot Δv with Isp 450 s) and replacements dominate the PV of costs at N = 1000;
(iii) N = 1000 is ~9× more expensive per kWh than N = 10 because receiver capacity limits the
delivered energy (M7); (iv) weather raises the LCOE by the inverse of the delivered fraction
(1/0.72 for the demo sites, 1/0.85 for the chosen sites). The Monte Carlo median exceeds the
nominal value because the input ranges are skewed (conservative ends farther from nominal).

## B. Files added/changed

Added:

- `src/helioswarm/era5.py`, `src/helioswarm/weather.py`, `src/helioswarm/economics.py`
- `configs/weather/era5_sites.yaml`; `configs/mvp_single_satellite_era5.yaml`,
  `configs/constellation_demo_era5.yaml`, `configs/constellation_demo_era5_clear_sites.yaml`,
  `configs/constellation_demo_clear_sites.yaml`, `configs/mvp_era5_years_sweep.yaml`,
  `configs/constellation_demo_era5_years_sweep.yaml`,
  `configs/constellation_demo_era5_clear_sites_years_sweep.yaml`;
  `configs/economics/{optimistic,nominal,conservative,geo_nominal}.yaml`
- `slurm/era5_download.sbatch`; `scripts/era5_product_crosscheck.py`;
  `scripts/m8a_default_path_pin.py`, `scripts/era5_fixture_request.py` (review)
- `tests/era5_helpers.py` (fake CDS client writing CDS-layout files), `tests/unit/test_era5.py`,
  `tests/unit/test_weather.py`, `tests/unit/test_economics.py`, `tests/unit/test_weather_cli.py`
  (review),
  `tests/integration/test_weather_run.py`, `tests/data/era5/` (74 kB real one-month fixture +
  provenance + README with attribution)
- `docs/milestones/M8a.md`

Changed (kept small and delimited for the merge into `helioswarm-m7-m9`):

- `src/helioswarm/config.py`: `WeatherSource`, `ModelsConfig.weather_source` (default
  `scenario`), `WeatherConfig` + `_weather_issues`, `RunConfig.weather`, one call in the
  cross-section validator, named approximation for ERA5, `_hash_input` (M8a keys not hashed
  while off), `__all__`.
- `src/helioswarm/system.py`: `atmosphere_at_access(..., weather=None)`,
  `run_single_satellite(..., weather=None)`, `SystemRunResult.weather`, `load_run_weather`,
  `describe_weather`; `src/helioswarm/constellation.py`: `evaluate_links(..., weather=None)`,
  `ConstellationRunResult.weather_provenance`; `src/helioswarm/constellation_io.py` and
  `src/helioswarm/io.py`: `weather.json` written and listed in the manifest outputs.
- `src/helioswarm/cli.py`: `weather download|climatology`, `economics` (one delimited block +
  two dispatch lines + docstring).
- `slurm/common.sh`: optional `HELIOSWARM_ENV_OVERLAY` (unset by default: no effect on the main
  checkout); `scripts/cluster_env.sh`: `HELIOSWARM_WEATHER_DIR`.
- `pyproject.toml` (`weather` extra; `cdsapi.*` mypy override; `era5_helpers` isort), `uv.lock`
  (+ cdsapi 0.7.7, ecmwf-datastores-client 0.6.0, xarray 2026.9.0, h5netcdf 1.8.3, h5py 3.16.0,
  multiurl, requests, …), `.github/workflows/ci.yml` (`--extra weather`), `.gitignore`
  (`data/weather/*`).
- `tests/unit/test_config_validity.py` (the M7 re-pin test also drops the two M8a keys; the MVP
  hash pin `a3fafc4d…23af` is unchanged).
- Docs: `docs/model_assumptions.md` (M8a section), `docs/data_provenance.md` (ERA5, economics
  inputs), `docs/compliance_matrix.md` (§10.2, §12.5, §14.1, §17, §15 rows), `README.md`,
  `experiments/README.md` (E10–E12).

## C. Exact commands run

From `/n/home05/ianpoon/Project-Dyson-Sphere-m8a`, every shell with `source
scripts/cluster_env.sh && source /n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/m8a_env.sh`;
jobs submitted with `export HELIOSWARM_REPO=/n/home05/ianpoon/Project-Dyson-Sphere-m8a
HELIOSWARM_ENV_OVERLAY=/n/netscratch/hekstra_lab/Lab/ian_poon/helioswarm/m8a_env.sh`.

```bash
# dependencies (h5netcdf 1.8 needs its h5py extra) and API checks
uv lock && uv sync --locked --extra dev --extra xcheck --extra itur --extra jax --extra opt --extra weather
uv run python -c "import cdsapi, inspect; print(inspect.signature(cdsapi.Client))"      # (url=None, key=None, *args, **kwargs)
#   LegacyClient(..., wait_until_complete=True, ...).retrieve(name, request, target=None) -> Remote
#   (request_id, download(target)) when wait_until_complete=False; read from the installed sources
# (The original session ran these checks as abbreviated interactive one-liners; spelled-out
#  equivalents, re-run in the review on the fixture request:)
uv run python -c "import sys; sys.path.insert(0, 'scripts'); import cdsapi; from era5_fixture_request import fixture_request; r = fixture_request(); c = cdsapi.Client(quiet=True).client; print(c.estimate_costs(r.dataset, r.as_cds_dict())); print([(x.get('id'), x.get('revision')) for x in c.get_licences(scope='dataset')])"
#   -> {'id': 'size', 'cost': 4032.0, 'limit': 121000.0} (= 6 x 1 variable x 672 h); ('cc-by', 1) listed
#   collection metadata/DOIs: cdsapi.Client().client.get_collection('reanalysis-era5-single-levels').json
curl -s https://api.crossref.org/works/10.1002/qj.3803 ; curl -s https://api.datacite.org/dois/10.24381/cds.adbb2d47
curl -s https://api.datacite.org/dois/10.24381/1cf1ad76
curl -sL -o .../otps.pdf https://www.nasa.gov/wp-content/uploads/2024/01/otps-sbsp-report-final-tagged-approved-1-8-24-tagged-v2.pdf
pdftotext -layout otps.pdf otps.txt            # values and pages in docs/data_provenance.md
# smoke download (one month, gridded) -> tests/data/era5 fixture; the request is rebuilt by a
# committed script (offline check: its SHA-256 equals the fixture provenance's request_sha256)
uv run python scripts/era5_fixture_request.py                      # matches fixture: True
uv run python scripts/era5_fixture_request.py --download <dir>     # the download itself
# bulk download: first attempt (gridded product, cancelled after 23 min, 2 files), then the time series
sbatch slurm/era5_download.sbatch configs/weather/era5_sites.yaml 4      # 60337126 (cancelled)
sbatch -t 0-01:00 slurm/era5_download.sbatch configs/weather/era5_sites.yaml 4   # 60341911
uv run python scripts/era5_product_crosscheck.py                       # 2 x OK, max diff 0
uv run helioswarm weather climatology configs/weather/era5_sites.yaml \
    --out $HELIOSWARM_RUNS_DIR/era5_climatology --threshold 0.1 0.3 0.5
uv run helioswarm weather climatology configs/weather/era5_sites.yaml \
    --out $HELIOSWARM_RUNS_DIR/era5_climatology_bilinear --threshold 0.5 --interpolation bilinear
uv run helioswarm weather climatology configs/constellation_demo_era5.yaml \
    --out $HELIOSWARM_RUNS_DIR/era5_climatology_constellation_demo_era5 --threshold 0.1 0.3 0.5
uv run helioswarm weather climatology configs/constellation_demo_era5_clear_sites.yaml \
    --out $HELIOSWARM_RUNS_DIR/era5_climatology_constellation_demo_era5_clear_sites --threshold 0.1 0.3 0.5
# ERA5 re-runs and checks (test partition, one job each: the QOS allows 5 submitted jobs)
sbatch --array=0 --export=ALL,SWEEP_ARRAY_COUNT=1 -t 0-00:30 --mem=1G -J helioswarm-era5-mvp \
    slurm/sweep_array.sbatch configs/mvp_era5_years_sweep.yaml $HELIOSWARM_RUNS_DIR/mvp_era5_years_sweep    # 60342908
sbatch -t 0-00:05 -J helioswarm-mvp-check slurm/run.sbatch configs/mvp_single_satellite.yaml \
    --out $HELIOSWARM_RUNS_DIR/mvp_single_satellite_m8a_check                                               # 60342911
sbatch --array=0 --export=ALL,SWEEP_ARRAY_COUNT=1 -t 0-02:30 --mem=2G -J helioswarm-era5-cdemo \
    slurm/sweep_array.sbatch configs/constellation_demo_era5_years_sweep.yaml \
    $HELIOSWARM_RUNS_DIR/constellation_demo_era5_years_sweep                                                # 60343733
sbatch --array=0 --export=ALL,SWEEP_ARRAY_COUNT=1 -t 0-02:30 --mem=2G -J helioswarm-era5-cclear \
    slurm/sweep_array.sbatch configs/constellation_demo_era5_clear_sites_years_sweep.yaml \
    $HELIOSWARM_RUNS_DIR/constellation_demo_era5_clear_sites_years_sweep                                    # 60343742
sbatch -t 0-00:30 --mem=2G -J helioswarm-cdemo-clearsites slurm/run.sbatch configs/constellation_demo_clear_sites.yaml  # 60344241
#   (60342962, 60342967, 60342995: the same constellation jobs first submitted to `shared`, cancelled
#    while pending (Priority) and resubmitted to `test` as above)
uv run helioswarm sweep configs/<sweep>.yaml --out $HELIOSWARM_RUNS_DIR/<sweep> --collate-only   # x3, in session
# 2-day MVP content hash, M7 code vs M8a code (script committed in the review; pinned by
# tests/integration/test_weather_run.py::test_default_scenario_path_reproduces_the_pre_m8a_content_hash)
mkdir -p $TMPDIR/m7src && git archive e86f8e7 src | tar -x -C $TMPDIR/m7src
PYTHONPATH=$TMPDIR/m7src/src uv run python scripts/m8a_default_path_pin.py   # 34c97c9f...7e2b
uv run python scripts/m8a_default_path_pin.py                                # 34c97c9f...7e2b
# economics (in session, seconds each)
for r in mvp_single_satellite_m8a_check mvp_single_satellite_microwave constellation_demo_sweep_r2 \
         constellation_demo_n1000_r2 mvp_era5_years_sweep constellation_demo_era5_years_sweep \
         constellation_demo_era5_clear_sites_years_sweep constellation_demo_clear_sites; do
  uv run helioswarm economics $HELIOSWARM_RUNS_DIR/$r --config configs/economics/optimistic.yaml \
      --config configs/economics/nominal.yaml --config configs/economics/conservative.yaml \
      --out $HELIOSWARM_RUNS_DIR/economics_m8a/$r --overwrite
done
seff <jobid>   # every job
uv run ruff check . ; uv run ruff format --check . ; uv run mypy src ; uv run pytest -q
```

## D. Tests and results

Gates (final, after all changes): `uv run ruff check .` → All checks passed; `uv run ruff
format --check .` → all files formatted; `uv run mypy src` → Success: no issues found in 39
source files; `uv run pytest -q` → **1278 passed, 2 skipped** (whole suite, incl. SPICE, cross-check and
ITU-R tests; after the review fixes, see Review outcomes; 1264 before).

```
uv run pytest -q
SKIPPED [1] tests/integration/test_constellation.py:544: JAX_PLATFORMS=cpu: no GPU requested (run by slurm/gpu_bench.sbatch)
SKIPPED [1] tests/unit/test_capture_batch.py:139: JAX_PLATFORMS=cpu: no GPU requested (run by slurm/gpu_bench.sbatch)
1278 passed, 2 skipped in 267.91s (0:04:27)
```

(81 test items in the five new test files; the two skips are the real-GPU tests run by the
`gpu_test` benchmark jobs.)

New tests (all network-free; ERA5 tests use the committed fixture or the fake CDS client):

- `tests/unit/test_era5.py` (23): grid snapping (ties, wrap, ±0.125° bound), area box, leap-year
  hours, the measured CDS costs reproduced, minimal month blocks, exact request payloads
  (gridded and time series), request-hash sensitivity, shipped sites cover every configured
  receiver, real fixture read + content checks (wrong month/area rejected), provenance schema +
  SHA-256 + cache verification (byte flip, wrong request), idempotent download (no client call
  on the second pass; truncated file re-downloaded), wrong server content leaves no file, zip
  unpacking (single member only; path traversal refused), latitude-order normalisation, year
  slicing, gap detection, nearest vs bilinear (exact on a linear field).
- `tests/unit/test_weather.py` (15): half-hour rounding, calendar fields vs `datetime`, cyclic
  year mapping, 29 Feb → 28 Feb, seeded year sampling (independent stream), TDB→UTC vs SPICE
  `et2utc` across the 2016-12-31 leap second (< 1 ms away from it, ≤ 1 s within the hour),
  threshold edges (`tcc = c` clear), Bernoulli frequency = tcc (5σ binomial tolerance),
  provider blocks exactly the cloudy hours, climatology counts by month/UTC/local hour/year and
  longest spell, joint statistics vs hand count, configuration rules, hash unchanged while off
  and changed when on, resolved-dump round trip.
- `tests/integration/test_weather_run.py` (5, SPICE): the WEATHER bit hits exactly the epochs
  whose arrival UTC (independent `et2utc` string path) falls in a cloudy hour; clear epochs
  deliver exactly the clear-sky run's power; all-clear ERA5 ≡ scenario run; all-cloudy blocks
  every accessible epoch; threshold strictness; Bernoulli seeded and between the extremes;
  `weather.json` + manifest entry through the CLI writer; constellation stage A blocks exactly
  the links with access at the overcast site and none elsewhere.
- `tests/unit/test_economics.py` (24): CRF = inverse annuity sum (incl. r → 0) and the NASA
  value 0.0510192593; Tsiolkovsky (ln 2 → 2); escape and Hohmann impulses vs vis-viva; phasing
  ellipse gains the phase (Kepler's third law); analytic LCOE = CRF·C0/E + O/E and NPV = 0 at
  p = LCOE; discounted energy with degradation = geometric series; LCOE monotone in d and r;
  replacement cycles (2 in 30 years) with performance renewal and partial replacement;
  zero/near-zero energy → undefined + flag; per-satellite Δv transport; mass sources; config
  validation; shipped scenarios ordered (optimistic < nominal < conservative LCOE, reversed
  NPV) with page-level sources; seeded Monte Carlo inside the scenario range; run inputs from a
  saved constellation directory and the CLI (refuses to overwrite).

Reproduction of earlier results (`scenario` default):

- Configuration hashes unchanged (MVP `a3fafc4d…23af` pinned; all M7 hashes unaffected because
  the M8a keys are not hashed while off).
- 2-day MVP time-series content hash with the M7 code (`git archive e86f8e7`) and the M8a code:
  both `34c97c9fa1f94a5161fd2fe188a4f82cff25375c3a4d164ee4f77bba04147e2b`.
- Full-year MVP (job 60342911): all 73 columns of the saved M6 run
  (`$HELIOSWARM_RUNS_DIR/mvp_single_satellite/`) identical (NaN-aware, bit for bit); delivered
  energy 55 260 935 592.524734 J in both. (Its content hash differs from the M6 run's only because
  M6b added three columns, `transmit_dir_x/y/z`.)

Slurm jobs (right-sizing from `seff`):

| Job | What | Partition, request | Wall | Max RSS | CPU eff. |
|---|---|---|---|---|---|
| 60337126 | ERA5 gridded bulk (cancelled) | test, 1 c, 2 GB, 8 h | 23 min | — | — |
| 60341911 | ERA5 time-series download, 12 sites, 4 parallel | test, 1 c, 2 GB, 1 h | 2 min 38 s | 226 MB | 8.9 % (network-bound) |
| 60342908 | MVP ERA5 sweep, 10 years | test, 1 c, 1 GB | 6 min 34 s | 439 MB | 93 % |
| 60342911 | MVP default (reproduction) | test, 1 c, 512 MB | 1 min 1 s | 161 MB | — |
| 60343733 | Demo N = 10 ERA5 sweep, 10 years | test, 1 c, 2 GB | 25 min 58 s | 638 MB | 98 % |
| 60343742 | ERA5-chosen sites sweep, 10 years | test, 1 c, 2 GB | 25 min 58 s | 644 MB | 98 % |
| 60344241 | ERA5-chosen sites, clear sky | test, 1 c, 2 GB | 2 min 48 s | 420 MB | — |

Right-sizing for reuse: the ERA5 download needs 1 core, 512 MB, 10 min; an N = 10 one-year
constellation point ~2.6 min and 700 MB; an MVP point ~40 s and 450 MB. No GPU was requested:
the weather lookup is a vectorised index per site and epoch (negligible next to the orbit and
beam stages), and economics runs in seconds; profiling gives no batched kernel to accelerate.

Data volume: 45 MB of ERA5 time-series files + 1.4 MB of gridded cross-check files on
netscratch; run outputs as before.

## E. Assumptions and equations implemented (with references)

All equations, validity ranges and neglected effects are in `docs/model_assumptions.md` (M8a);
data, licences and sources in `docs/data_provenance.md` (M8a). In brief:

- ERA5 `tcc` (Hersbach et al. 2020, doi:10.1002/qj.3803; CDS DOIs 10.24381/1cf1ad76,
  10.24381/cds.adbb2d47; CC-BY-4.0, CDS attribution wording). Outage `tcc > c` (c = 0.5
  assumed; justified by the random-overlap reading `P(clear LOS) = 1 − tcc`) or Bernoulli
  `u < tcc`. TDB → UTC: `UTC = TDB − ΔET + 946 728 000 s` (SPICE `deltet`; NAIF Time Required
  Reading). Bilinear: tensor-product linear interpolation.
- Economics: `CRF = r(1+r)^n/((1+r)^n − 1)`; `LCOE = Σ cost_t (1+r)^−t / Σ E_t (1+r)^−t`;
  NREL simple LCOE as used by NASA OTPS 2024 (PDF p. 73–74); Tsiolkovsky `m0/m1 =
  exp(Δv/(g0 Isp))`; escape `Δv = sqrt(v∞² + 2μ/r) − sqrt(μ/r)`; Hohmann impulses; phasing
  `P' = P(1 − φ/(2πN))`, `a' = a(P'/P)^(2/3)` (Kepler's third law; vis-viva).
- Sourced cost inputs: NASA OTPS 2024 (launch 850 USD/kg baseline, 1500 USD/kg Falcon Heavy
  2019, 85 USD/kg derived; 3 % discount rate; 30-year system; 10/15-year hardware; operations
  14.4 M USD/yr per system and ~0.5 M USD/yr per satellite today; ground-system cost per m²
  derived from RD1/RD2; > 4.7 km/s LEO→GEO; 0.02–0.05 USD/kWh terrestrial 2050). Everything
  else assumed and labelled.

## F. Known limitations and next milestone

- `tcc` is a grid-box fraction (~28 km): no cloud optical depth, no slant-path geometry, no
  sub-hour variability; mountain-top sites are represented by their box (hawaii_summit).
  Aerosol/dust extinction stays in the (lossless) clear-sky factor. Ten years (2014–2023) are
  shorter than a 30-year climate normal and carry no trend.
- The Bernoulli mode drops cloud persistence (hours independent); the threshold mode keeps it.
- Microwave links are not coupled to ERA5 (ITU-R statistics); ERA5 cloud liquid water could
  replace the P.840 statistic later.
- Economics: flat manufacturing USD/kg (no learning curve), mixed cost years without
  inflation adjustment, no development cost, decommissioning, land, transmission or storage;
  the time to reach the operational slot (2-year deployment budget plus any phasing
  revolutions) does not delay revenue; stage dry mass ignored in the transport model;
  the **optical receiver cost is the microwave rectenna/ground-segment cost per m² used as a
  proxy** (no sourced laser-PV receiver cost; `optical_receiver_capex_per_m2_usd`, assumed)
  and the fixed operations cost of one GW-class SBSP system is applied unchanged to every run
  (assumed transfer) — both set the level of the MVP and demo LCOEs; the transmitter has no
  separate capex (`transmitter_capex_per_m2_usd = 0`: folded into the flat USD/kg);
  delivered energy is assumed proportional to space-segment performance (`E_k = E₁ P_k`), which
  overstates the degradation loss (LCOE biased high) for receiver-capacity-limited networks
  such as N = 1000; the near-1-au combined Hohmann + phasing Δv is an upper-bound
  approximation (exact at 1 au; the shipped runs are all at exactly 1 au); the mass proxy is the M7 proxy (assumed specific masses;
  the optical areal masses were also applied to the microwave array, whose LCOE is undefined
  anyway).
- **Deferred to M8 after the merge with M7b** (per the parallel-worktree plan): the helio-vs-GEO
  economics comparison (`configs/economics/geo_nominal.yaml` is ready: explicit LEO→GEO Δv) and
  ERA5 re-runs of the M7b comparison configurations; M8 then uses ERA5 years as the aleatory
  weather input and LCOE as an optional objective.
- Merge notes: `config._hash_input` keeps every pre-M8a hash; M7b's new schema keys will change
  hashes on their own terms (their re-pin stands). Conflicts are expected only in the shared
  files listed in B (small, delimited blocks).

## Review outcomes

Independent review of M8a (physics, tests and spec lenses; 18 findings, two pairs of
duplicates). Every finding was first reproduced; none was refuted. Gates after the fixes:
ruff check / ruff format --check / mypy src clean; `uv run pytest -q` → 1278 passed, 2 skipped
(GPU-only). The new tests were checked to fail for the reviewers' mutations (TDB used as UTC,
sign of ΔET flipped, epoch −300 s, emission instead of arrival epoch, sign of the Earth-relative
phase flipped, base-orbit semimajor axis, EAC without the `(1+r)^(T_c−1)` factor, opex per site
instead of per satellite, opex fraction of manufacturing only, window factor dropped).

| # | Finding | Disposition |
|---|---|---|
| 1, 11 (major, physics/spec) | `run_inputs` used the base `orbit` (semimajor axis, circular/ecliptic check) and ignored ring overrides | **Fixed.** `member_orbits()` resolves every satellite's orbit once (`constellation_members`); `RunEconomicInputs.semimajor_axis_m` is per satellite, `circular_ecliptic` requires every member to be circular and ecliptic; Δv is computed per member. Reproduced first: ring at 0.7 au / 10° gave 1 au phasing values. Tests: `test_ring_element_overrides_set_each_satellites_transfer` (0.7 au → Hohmann for all 10; inclined ring refused). |
| 2 (major, physics) | Δv discontinuous at `|a − 1 au| = 1e-6 au` (Hohmann ignored the slot phase) | **Fixed.** The slot phase is part of every transfer: Hohmann + free drift on the arc `θ₀ + (n_a − n_E)[0, N yr]`; outside it a phasing ellipse at `a` for the residual phase, added to both impulses; exactly the co-orbital model at 1 au; continuous in `a` and φ (named upper-bound approximation near 1 au; `docs/model_assumptions.md`). Reviewer's case: 1.0000011 au now 6115.2 m/s (was 3176.4). Tests: continuity across 1 au, far orbits (0.7 au) pure Hohmann, 1.05 au arc interior/ends/opposite. Shipped results (all at 1 au) unchanged. |
| 3 (minor, physics) | EAC = CRF·PV low by `(1+r)^−(T_c−1)` for `T_c > 1` | **Fixed.** `EAC = CRF · PV · (1+r)^(T_c−1)` (reference epoch documented); test vs closed form and `LCOE·E` for `T_c = 1, 3, 5`. Shipped scenarios have `T_c = 1` (unchanged). |
| 4, 14 (minor, physics/spec) | `_needs_ephemeris` ignored ring `phase_reference` | **Fixed.** Inspects the base and every ring's reference (None = inherit); test `test_ring_with_absolute_phase_loads_the_ephemeris`. |
| 5 (major, tests) | No test of the TDB→UTC conversion and arrival epoch in the weather lookup | **Fixed.** Unit test with the real `UtcOffsetTable.build` (ΔET 67.184 s in Jan 2015) probing hh:29:20 / hh:30:40; integration test (parametrised) with start times putting arrivals within 66 s before hh:30 UTC (about 28 s) and emission/arrival on different hours, with the discriminating preconditions asserted; checked against the reviewer's three mutations and the emission-epoch mutation. |
| 6 (major, tests) | Single-run summary branch and absolute-phase branch untested | **Fixed.** SPICE test on a saved single run of the MVP config: phase vs independent `spkpos(EARTH wrt SUN, ECLIPJ2000)` (1e-9 rad; −70.18°), kind, aperture `πd²/4`, radiator area, Δv, CLI on the directory; the flipped sign is shown to change Δv by > 100 m/s. |
| 7 (minor, tests) | MC discount-rate low bound 0.02 below the optimistic 0.03; claim "inside the scenario range" false | **Fixed.** Low bound set to 0.03 (`nominal`, `geo_nominal`); load-time check that each mode equals the scenario value; test checks that the bounds equal the optimistic/conservative values; the "inside the range" statement is now documented and commented as an observation for the shipped seed. MC intervals recomputed (table above). |
| 8 (minor, tests) | Opex terms and window annualisation never checked by value | **Fixed.** Closed-form test with `N = 3`, `O_sat ≠ 0`, opex fraction ≠ 0 and a 30-day window. |
| 9 (minor, tests) | `climatology_command` and `weather download|climatology` CLI untested | **Fixed.** `tests/unit/test_weather_cli.py`: download CLI (dry run, site filters, idempotency with the fake client) and climatology vs hand counts (clear fractions, January, Bernoulli expectation, joint all-cloudy/all-clear, monthly Parquet, null correlation, attribution year). |
| 10 (minor, tests) | Default-path output not pinned | **Fixed.** `test_default_scenario_path_reproduces_the_pre_m8a_content_hash` pins `34c97c9f…7e2b` (re-obtained in the review with `scripts/m8a_default_path_pin.py`). |
| 12 (major, spec) | Optical runs used a NASA microwave rectenna cost labelled `derived`; GW-system opex labelled `sourced` for a single satellite; "not transplanted" claims | **Fixed.** Receiver cost split into `microwave_receiver_capex_per_m2_usd` (derived, as before) and `optical_receiver_capex_per_m2_usd` (same values, provenance `assumed`, source text stating the proxy); `opex_fixed_per_year_usd` relabelled `assumed` (sourced number, assumed transfer). Statements corrected in `docs/data_provenance.md`, this file (header, A, F) and `docs/model_assumptions.md`. Numbers unchanged (a sourced optical-receiver cost was not found; none invented). |
| 13 (minor, spec) | No transmitter capex input | **Fixed.** Optional `transmitter_capex_per_m2_usd` (per m² of aperture; default 0, `assumed`, "folded into USD/kg"), in the CAPEX breakdown and replaced with the space segment; test. |
| 15 (minor, spec) | Integer/structural inputs without provenance; MC ranges unchecked; `with_values` bypassed validation | **Fixed.** `system_lifetime_years`, `construction_years`, `transport.phasing_years` are `IntParameter`s and `min_meaningful_energy_kWh_per_year` a `CostParameter` (all with provenance; `system_lifetime_source` folded in); mode = scenario value and admissible bounds checked on load; `with_values` re-validates each sampled configuration (tests). |
| 16 (minor, spec) | `E_k = E₁ P_k` is an unnamed approximation for saturated networks | **Fixed (documented)** as a named approximation with validity in `docs/model_assumptions.md`, the module docstring and F. The optional correction from a saturated-energy fraction is **deferred**: the run summaries carry no per-year saturation measure from which it could be derived without re-simulation. |
| 17 (minor, spec) | Section C not reproducible (`pin.py`, elided one-liners) | **Fixed.** `scripts/m8a_default_path_pin.py` and `scripts/era5_fixture_request.py` committed (the latter rebuilds the fixture request; its SHA-256 matches the fixture provenance offline); the CDS cost/licence check spelled out and re-run (cost 4032 = 6 × 1 × 672, limit 121 000; `cc-by` rev 1). The original abbreviated lines are marked as such. |
| 18 (minor, spec) | Climatology attribution year from the wall clock; `allow_nan=True` | **Fixed.** Year from the files' `retrieved_utc` (also written as `retrieved_utc`); non-finite values written as `null` with `allow_nan=False`; tested (retrieval year 2019 in the test, null correlations). The shipped climatology outputs contain no NaN and their year (2026) equals the retrieval year, so they were not regenerated. |

Economics outputs were recomputed in session (`$HELIOSWARM_RUNS_DIR/economics_m8a/*`, previous
versions kept in `economics_m8a_prereview/`): every scenario LCOE, CAPEX, NPV, mass and Δv
identical to 1e-9 relative; only the Monte Carlo percentiles changed (table in A).
