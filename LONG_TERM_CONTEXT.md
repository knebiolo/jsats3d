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
| `2025_Temp_String_Data_5min_interpolated.csv` | CSV | Authoritative temperature string; use `DD_N_0p5`, `DD_N_1p5`, `DD_N_9`, `DD_N_18` | Inspected |
| `ATS_3017_Internal_Column_Guide.txt` | Text Guide | Decode File Format 2.0 `Internal` clock/status groups | Inspected |
| `pre_diagnostics.py` / `construct_ATS_dfs` | Legacy Python | Prior ATS raw-file parser reference | Inspected |
| `raw_data/` | ATS raw receiver CSVs | Raw File Format 2.0 detections, GPS rows, Internal clock evidence, SigStr, sensor fields | Available; target parser smoke-tested |

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
| 2026-09-16 | Do not use nominal beacon PRI to identify clock jumps or quantify jump magnitude | Beacon PRI has jitter, slight drift, and a catch-up ping approximately every 15.5-16.5 minutes |
| 2026-09-16 | Use ZOI02 as the central clock reference for the planned 2025 synchronization method | Owner-selected reference from clock synchronization meeting |
| 2026-09-16 | Correct ZOI02 time jumps without adjusting ZOI02 drift | Planned reference-clock treatment from clock synchronization meeting |
| 2026-09-16 | Estimate other receiver jumps and drift from beacon TDOA using piecewise regressions | Preserve stable segments between firmware-related clock jumps |
| 2026-09-16 | Parse raw receiver files into legacy-compatible tables | Raw files contain clock synchronization events and one-second jump flags absent from concatenated deliverables |
| 2026-09-16 | Use DD_N temperature-string columns as authoritative temperature inputs | PM specified four DD_N depths: 0.5, 1.5, 9, and 18 |
| 2026-09-16 | Preserve and decode ATS `Internal` during raw parsing | Clock events, counter restarts, offsets, one-second adjustments, and GPS-loss status are encoded there |
| 2026-09-16 | Retain `SigStr` from raw ATS files as a candidate multipath feature | PM preliminary exploration suggests utility; adoption requires Gate 3 validation |
| 2026-09-16 | Preserve the legacy core and remove parallel architecture modules | New work is limited to parsers, adapters, diagnostics, and legacy-table-compatible preprocessing |
| 2026-09-16 | Allow additive ATS columns in legacy tables | Preserve `Internal`, `SigStr`, raw sensor fields, receiver/firmware metadata, and source provenance without changing required legacy columns |
| 2026-09-17 | Restrict raw parsing to exact serials mapped to ZOI01-ZOI11 and CFD01-CFD09 | PM identified 20 receivers relevant to 3D processing; avoid unrelated uploaded receivers |
| 2026-09-17 | Prefer `_cleaned`, then `_recovered`/`_recovery`, over same-stem original files | PM confirmed corrected files remove corrupt ATS lines that otherwise break processing |

## Established Methods
- Stage single tags or chunked detections into SQLite containing `tblTag`, `tblReceiver`, `tblDetectionRaw`, `tblInterpolatedTemp`, `tblWSEL`, `tblStudyParameters`.
- Preserve required legacy `tblDetectionRaw` columns and add ATS-specific columns when present: `Event`, `Internal`, `SigStr`, `RawTemperature`, `Pressure`, `Tilt`, `BatteryVoltage`, `BitPeriod`, `Threshold`, `ReceiverType`, `FirmwareVersion`, `FileFormatVersion`, `SourceFile`, and `SourceRow`.
- Map raw ATS `SigStr` unchanged into legacy `Amplitude` while retaining the original `SigStr` column; do not convert or normalize it during ingestion.
- Project WGS84 coordinates to EPSG:26910 easting/northing.
- Maintain a known legacy dataset for regression and parity checks.
- Apply pulse-rate blanking before DBSCAN when pulse rate is known.
- Use 3D point clouds, voxel density, plan-view heat maps, depth histograms, and kernel density utilization distributions for fish-space visualization.

## Project Conventions
- Coordinates: UTM Zone 10N NAD83 (EPSG:26910), meters.
- Elevations: meters relative to Benchmark (BM) or Water Surface Elevation (WSEL).
- Timestamps: UTC/Local datetime strings converted to float seconds since Unix epoch.

## Persistent Assumptions
- Beacon tag periods in the config workbook describe nominal transmission periods only; they are not authoritative timing references for clock-jump detection or magnitude estimation.
- `FFD3` is mobile validation tag, not stationary receiver beacon.
- Legacy paper/2019/DBSCAN results remain reference outputs until replacement parity is demonstrated.

## Known Limitations
- Missing `SNR` and `NBW` prevents legacy ML/DBSCAN multipath filtering from operating unchanged.
- `FFD3` pulse rate is approximately 3 seconds; exact interval remains pending measurement from static holds.
- CHN receivers lack GPS / static coordinates in config.
- Synchronization parameters remain provisional pending PM guidance on the new synchronization approach.
- FFD3 staging now includes bounded local-beacon rows, but master receiver and synchronization parameters remain unresolved.
- Beacon PRI is irregular: jitter and slight drift are present, with a catch-up ping approximately every 15.5-16.5 minutes.
- Receiver firmware can create time jumps. Raw files identify synchronization events and flagged one-second jumps, but do not report every jump magnitude.
- Multipath in beacon detections complicates TDOA-based jump and drift estimation.
- The authoritative DD_N temperature file contains four complete depth columns but still spans 2025-06-17 through 2025-09-17; it does not cover the 2025-06-10 test day.
- The supplied Internal-column guide applies to File Format 2.0; File Format 3.0 and later require a separate schema.
- Legacy `construct_ATS_dfs` keeps `temp` and `sigStr` but drops `diagCode`/Internal from detection output, so it cannot be reused unchanged for clock-event parsing.
- Experimental `pipeline_mode`, `multipath_interface`, and `sync_readiness` modules were removed; they are not part of the legacy-core approach.
- Target 3D raw receiver set contains 20 SR3017 units, all listed as firmware v10.62F in the configuration workbook.
- June 10 raw folder contains 18 of 20 target serials; CFD05/serial 19033 and ZOI04/serial 20027 are absent from that folder.
- Raw parser supports only verified File Format 2.0 and fails on unsupported formats.
- No approved permanent regression dataset selected yet.

## Completed Milestones
- Formatted 2025 datasets into legacy SQLite schema (`jsats3d_2025_FFD3_manager_demo.db`).
- Verified legacy database loading and temperature interpolation execution.
