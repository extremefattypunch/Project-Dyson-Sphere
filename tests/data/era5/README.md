# ERA5 test fixture (real CDS output, unmodified)

`N23.50_E000.00_k1_tcc_2014_m02-02.nc`: ERA5 hourly total cloud cover (`tcc`), February 2014
(672 hours), 3 x 3 nodes of the 0.25 deg grid around 23.5 N, 0.0 E (area N 23.75, W -0.25,
S 23.25, E 0.25), downloaded on 2026-11-13 by `helioswarm.era5.download_requests` (CDS request
id and SHA-256 in the provenance JSON next to it). Used by `tests/unit/test_era5.py`
(reader, content checks, provenance schema); unit tests never touch the network.

Dataset: ERA5 hourly data on single levels from 1940 to present, Copernicus Climate Change
Service (C3S) Climate Data Store (CDS), DOI 10.24381/cds.adbb2d47; Hersbach et al. (2020),
Q. J. R. Meteorol. Soc. 146(730), 1999-2049, doi:10.1002/qj.3803. Licence: CC-BY-4.0.

Generated using Copernicus Climate Change Service information 2026. Neither the European
Commission nor ECMWF is responsible for any use that may be made of the Copernicus information
or data it contains.
