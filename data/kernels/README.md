# SPICE kernels

SPICE kernels are **not committed**. They are large, regenerable and distributed by NASA/JPL NAIF.
This directory is the portable default kernel location. On the Cannon cluster,
`scripts/cluster_env.sh` sets `HELIOSWARM_KERNEL_DIR` to a netscratch directory instead.

Milestone M2 will add:

- `data/kernels/kernels.json`, which records the kernel file names, source URLs on
  naif.jpl.nasa.gov and SHA-256 checksums;
- `scripts/fetch_kernels.py`, which downloads the kernels idempotently and verifies those
  checksums.

The planned set is `naif0012.tls`, `pck00011.tpc`, `gm_de440.tpc` and `de440s.bsp`. The exact
files, versions and checksums will be recorded in `kernels.json`, in `docs/data_provenance.md`
and in every run manifest.

Status (M0): no kernels are used yet.
