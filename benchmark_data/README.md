# Benchmark Data

Field discharge records and basin attributes used in the eight-site
benchmark consistency test (main text Section 4, SI Sections S6--S7).

## Directory structure

```
benchmark_data/
  raw/
    {gauge_id}_usgs_instantaneous.csv   # 15-minute discharge (8 sites)
    {gauge_id}_usgs_daily.csv           # Daily mean discharge (8 sites)
    camels_clim.txt                     # CAMELS v2 climate attributes
    camels_geol.txt                     # CAMELS v2 geology attributes
    camels_topo.txt                     # CAMELS v2 topography attributes
```

## Benchmark sites

| Gauge ID | Site name | State | Hydrogeologic role | Instantaneous record |
|----------|-----------|-------|-------------------|---------------------|
| 07014500 | Meramec River near Sullivan | MO | Positive control (Ozark karst) | 2000-01-01 to 2024-12-31 |
| 07067500 | Big Spring near Van Buren | MO | Karst benchmark (Ozark) | 2000-02-09 to 2024-12-31 |
| 02322500 | Ichetucknee River near Fort White | FL | Model-limitation case (Florida Platform) | 2000-01-01 to 2024-12-31 |
| 08155500 | Barton Springs at Austin | TX | Mixed case (Edwards) | 2000-01-01 to 2024-12-31 |
| 08169000 | Comal Springs at New Braunfels | TX | Supplementary Edwards comparator | 2001-10-01 to 2024-12-31 |
| 08171000 | Blanco River at Wimberley | TX | Supplementary Edwards comparator | 2001-10-01 to 2024-12-31 |
| 11148900 | Arroyo Seco near Soledad | CA | Non-carbonate negative control | 2000-01-01 to 2024-12-31 |
| 01013500 | Fish River near Fort Kent | ME | Non-carbonate negative control | 2000-01-03 to 2024-12-31 |

## Data sources and access

### USGS discharge records

All discharge files were downloaded from the **USGS National Water
Information System (NWIS)** web services.

- **Service URL:** https://waterservices.usgs.gov/nwis/
- **Instantaneous values (IV):** 15-minute unit-value discharge
  (`parameterCd=00060`), retrieved via the IV web service.
- **Daily values (DV):** Daily mean discharge (`parameterCd=00060`,
  `statCd=00003`), retrieved via the DV web service.

Example query for Meramec River instantaneous data:

```
https://nwis.waterservices.usgs.gov/nwis/iv/?sites=07014500&parameterCd=00060&startDT=2000-01-01&endDT=2024-12-31&format=rdb
```

**Citation:**
U.S. Geological Survey. USGS Water Data for the Nation. U.S. Geological
Survey National Water Information System (NWIS). Accessed 2025.
https://doi.org/10.5066/F7P55KJN

### CAMELS basin attributes

The three attribute files (`camels_clim.txt`, `camels_geol.txt`,
`camels_topo.txt`) are subsets of the **CAMELS v2** dataset (671 basins).
They are used to identify non-carbonate reference basins for the extended
negative-control analysis (SI Section S17) via the `carbonate_rocks_frac`
field in `camels_geol.txt` and the geographic coordinates in
`camels_topo.txt`.

- **Original source:** https://ral.ucar.edu/solutions/products/camels
- **DOI:** https://doi.org/10.5065/D6MW2F4D

**Citation:**
Addor, N., Newman, A. J., Mizukami, N., and Clark, M. P. (2017). The
CAMELS data set: Catchment attributes and meteorology for large-sample
studies. *Hydrology and Earth System Sciences*, 21, 5293--5313.
https://doi.org/10.5194/hess-21-5293-2017

## File format

### Discharge CSVs

| Column | Description |
|--------|-------------|
| `datetime` or `date` | Timestamp (ISO 8601 with timezone for IV; date-only for DV) |
| `discharge_cfs` | Discharge in cubic feet per second |

Instantaneous files are at the native USGS reporting cadence (typically
15 min; Blanco River reports at 5 min). Daily files contain one row per
calendar day.

### CAMELS attribute files

Semicolon-delimited text files with `gauge_id` as the join key. Field
definitions are documented in Addor et al. (2017), Table 1 and
supplementary material.

## Preprocessing

No preprocessing is applied to these raw files before they enter the
analysis pipeline. Gap masking, smoothing, event extraction, and HI
computation are performed by the scripts in `scripts/` (see
`scripts/README.md`). The raw files archived here are exactly as
downloaded from the respective services.

## Notes

- Discharge values of exactly 0.0 or negative values in the raw NWIS
  records are retained as-is; the event-extraction pipeline handles
  these via the cutoff-fraction filter (`f_cut`).
- The CAMELS attribute files contain all 671 CAMELS basins, not just the
  two used in the benchmark (Arroyo Seco 11148900, Fish River 01013500).
  The extended negative-control analysis (SI Section S17) uses a broader
  subset filtered by `carbonate_rocks_frac = 0`.

