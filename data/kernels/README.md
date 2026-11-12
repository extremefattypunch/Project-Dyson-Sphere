# SPICE kernels

SPICE kernels are **not committed**. They are large, regenerable and distributed publicly by
NASA/JPL NAIF. Only the manifest `kernels.json` is versioned. It records, for each kernel:

- the file name and kind;
- the source URL on naif.jpl.nasa.gov;
- its purpose and size;
- its SHA-256, the NAIF-published MD5 where one exists, the server's `Last-Modified` and the
  retrieval date.

| Kernel | Kind | Purpose |
|---|---|---|
| `naif0012.tls` | LSK | Leap seconds for UTC ↔ TDB (`str2et`, `et2utc`) |
| `pck00011.tpc` | text PCK | IAU rotation models and radii (IAU_EARTH, from M3) |
| `gm_de440.tpc` | text PCK | DE440 GM values (`bodvrd`) |
| `de440s.bsp` | SPK | DE440 short planetary ephemeris, 1849-12-26 to 2150-01-22 TDB |

## Fetching

```bash
uv run helioswarm fetch-kernels          # or: uv run python scripts/fetch_kernels.py
```

The command is idempotent:

- Missing files are downloaded to a temporary file, verified (size, SHA-256, MD5) and only then
  renamed into place.
- Existing files are verified and left alone.
- A file that does not match the manifest is **never** overwritten. The command stops with an
  error instead.

Kernel directory: `--kernel-dir`, else `$HELIOSWARM_KERNEL_DIR`, else this directory. On the
Cannon cluster, `scripts/cluster_env.sh` sets `HELIOSWARM_KERNEL_DIR` to netscratch. Netscratch
is purged after 90 days, so re-run the command after a purge.

## Checksum policy

NAIF does not publish SHA-256 checksums. The SHA-256 values in `kernels.json` were computed on
first retrieval (2026-11-12, trust on first use) and are enforced from then on. For `de440s.bsp`,
NAIF publishes an MD5 in `spk/planets/aa_checksums.txt`. It matched at retrieval and is enforced
as well. The `gm_de440.tpc` SHA-256 also equals the value recorded independently in M1.

## Use in code

`helioswarm.kernels.KernelSet.loaded()` is the only place kernels are loaded. It verifies the
files, loads them into an empty SPICE pool, and unloads and `kclear`s them on exit. Provenance
records (name, path, URL, size, SHA-256, MD5, retrieval date) are exposed for run manifests.
See `docs/data_provenance.md`.
