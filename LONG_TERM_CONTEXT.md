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
| 2026-09-13 | Preserve paper/2019/DBSCAN dataset as regression reference | Maintain legacy parity while changing code |
| 2026-09-13 | Use DBSCAN for new ATS multipath filtering | New data lacks legacy classifier features |
| 2026-09-13 | Work only on feature branch | Protect `main` and existing legacy projects |
| 2026-09-14 | Treat FFD3 pulse rate as approximately 3 seconds pending static-hold measurement | PM provided approximate rate; exact rate requires code-based measurement from static holds |
| 2026-09-15 | Use 3.33 seconds as provisional FFD3 pulse rate in staging only | Measured static-hold median; pending PM confirmation and not a final study-wide parameter |
| 2026-09-15 | Store WSEL in source feet with `BM_Elev_Units=feet` | Legacy runtime converts WSEL when output units are meters |
| 2026-09-15 | Stage local beacon rows through an explicit bounded datetime window | Avoid loading all beacon detections and prevent unbounded epoch assumptions |
| 2026-09-15 | Drop incomplete receivers only when explicitly requested | Preserve raw metadata by default and report each dropped receiver/reason |

## Established Methods
- Stage single tags or chunked detections into SQLite containing `tblTag`, `tblReceiver`, `tblDetectionRaw`, `tblInterpolatedTemp`, `tblWSEL`, `tblStudyParameters`.
- Project WGS84 coordinates to EPSG:26910 easting/northing.
- Maintain a known legacy dataset for regression and parity checks.
- Apply pulse-rate blanking before DBSCAN when pulse rate is known.
- Use 3D point clouds, voxel density, plan-view heat maps, depth histograms, and kernel density utilization distributions for fish-space visualization.

## Project Conventions
- Coordinates: UTM Zone 10N NAD83 (EPSG:26910), meters.
- Elevations: meters relative to Benchmark (BM) or Water Surface Elevation (WSEL).
- Timestamps: UTC/Local datetime strings converted to float seconds since Unix epoch.

## Persistent Assumptions
- Beacon tag periods in config workbook are authoritative.
- `FFD3` is mobile validation tag, not stationary receiver beacon.
- Legacy paper/2019/DBSCAN results remain reference outputs until replacement parity is demonstrated.

## Known Limitations
- Missing `SNR` and `NBW` prevents legacy ML/DBSCAN multipath filtering from operating unchanged.
- `FFD3` pulse rate is approximately 3 seconds; exact interval remains pending measurement from static holds.
- CHN receivers lack GPS / static coordinates in config.
- Synchronization parameters remain provisional pending PM guidance on the new synchronization approach.
- FFD3 staging now includes bounded local-beacon rows, but master receiver and synchronization parameters remain unresolved.
- No approved permanent regression dataset selected yet.

## Completed Milestones
- Formatted 2025 datasets into legacy SQLite schema (`jsats3d_2025_FFD3_manager_demo.db`).
- Verified legacy database loading and temperature interpolation execution.
