# Long-Term Context — TPU Acoustic Telemetry

## Project Definition
Production client acoustic telemetry data processing, clock synchronization, and 3D positioning pipeline for the TPU Cowlitz River / Cowlitz Falls Dam telemetry project. Ingest 2025 datasets, format standardized SQLite databases, and execute verified legacy `jsats3d` workflows.

## Project Classification
- Type: Production Telemetry Analysis Project (not R&D)
- Primary Client/Study: Tacoma Public Utilities (TPU) Cowlitz Falls Dam / Cowlitz River Acoustic Telemetry

## Workspace Layout
- `.ai_journal/`: Session notes, prompt conventions, long-term rules.
- `config/`: Pipeline parameters and study configurations.
- `data/`: Local staging databases and lookup tables.
- `docs/`: Technical guides, workflow specifications, and manuals.
- `jsats3d/`: Core Python library for telemetry processing and positioning algorithms.
- `notebooks/`: Jupyter exploration notebooks.
- `output/`: Processed figures, CSV exports, and run deliverables.
- `scripts/`: Ingestion adapters, legacy batch drivers, and trial scripts.
- `tests/`: Automated unit tests and smoke checks.

## Deliverables
- Reusable data adapter (`scripts/adapt_2025_to_legacy.py`)
- Standardized SQLite staging databases
- Pipeline validation outputs and diagnostics

## Stable Data Sources
| Dataset | Type | Intended Use | Inspection Status |
|---|---|---|---|
| `cowlitz_2025_AT_config.xlsx` | Excel Workbook | Receiver metadata, hydrophone depths, beacon codes, beacon periods | Inspected |
| `master_df_gps.csv` | CSV | Dynamic & static receiver GPS coordinates (CFD02-CFD09) | Inspected |
| `2025 Master Covariate Table_20251212.csv` | CSV | Water temperature (`tblInterpolatedTemp`) and water level (`tblWSEL`) | Inspected |
| `master_df_test.csv` | CSV | Test tag detections (`FFD3`, 7,550 rows) | Inspected |
| `master_df_study.csv` | CSV | Study tag detections (77M rows) | Inspected |
| `master_df_beacon.csv` | CSV | Beacon tag detections (46M rows) | Inspected |
| `master_df_unknown.csv` | CSV | Unknown tag detections (86M rows) | Inspected |
| `released_v0.csv` | CSV | PTAGIS release metadata | Inspected |
| `collected_v0.csv` | CSV | PTAGIS observation metadata | Inspected |
| `cowlitz_AT_2025_testing_sheets.xlsx` | Excel Workbook | Controlled test track and static hold metadata | Inspected |

## Durable Structural Facts
### `2_AT_detection_datasets`
- Schema: `dateTime`, `tagCode`, `amp`, `receiverName`, `event`
- Raw signal metrics (`SNR`, `NBW`, `FreqOff`, `Pascals`, `Celsius`) absent in 2025 deliverables.
- `master_df_test.csv` contains 7,550 detections for test tag `FFD3`.

### `1_array_metadata/cowlitz_2025_AT_config.xlsx`
- 37 named receivers, 50 beacon entries, 35 beacon periods (typically 60s, some 30s).
- Hydrophone depths present for 31 receivers.
- Coordinates provided in WGS84 latitude/longitude, converted to EPSG:26910 (UTM Zone 10N NAD83).

### `Master Covariate Table/2025 Master Covariate Table_20251212.csv`
- 26,688 5-min intervals (2025-06-17 to 2025-09-17).
- Temperatures in Celsius (`BB_TPU_Surface_t`, `FBS_Surface_t`, etc.).
- Water elevations in feet (`NSC.CZD_WTR_EL.F_CV`).

## Standing Decisions
| Date | Decision | Reason |
|---|---|---|
| 2026-09-10 | Strictly read-only raw data access | Prevent data corruption on network drive |
| 2026-09-10 | Build isolated adapter script `adapt_2025_to_legacy.py` | Avoid breaking legacy `jsats3d.py` |
| 2026-09-10 | Leave missing raw signal metrics as `NULL` | Do not fabricate artificial `SNR`/`NBW` |
| 2026-09-11 | Normalize datetime resolution in `jsats3d.py` | Prevent pandas datetime nanosecond/second integer scale mismatch |

## Established Methods
- Stage single tags or chunked detections into SQLite containing `tblTag`, `tblReceiver`, `tblDetectionRaw`, `tblInterpolatedTemp`, `tblWSEL`, `tblStudyParameters`.
- Project WGS84 coordinates to EPSG:26910 easting/northing.

## Project Conventions
- Coordinates: UTM Zone 10N NAD83 (EPSG:26910), meters.
- Elevations: meters relative to Benchmark (BM) or Water Surface Elevation (WSEL).
- Timestamps: UTC/Local datetime strings converted to float seconds since Unix epoch.

## Persistent Assumptions
- Beacon tag periods in config workbook are authoritative.
- `FFD3` is mobile validation tag, not stationary receiver beacon.

## Known Limitations
- Missing `SNR` and `NBW` prevents legacy ML/DBSCAN multipath filtering from operating unchanged.
- `FFD3` pulse rate not documented in config workbook.
- CHN receivers lack GPS / static coordinates in config.

## Completed Milestones
- Formatted 2025 datasets into legacy SQLite schema (`jsats3d_2025_FFD3_manager_demo.db`).
- Verified legacy database loading and temperature interpolation execution.
