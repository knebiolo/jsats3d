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
| `Tag Drag Period/DD_N`, `DD_S` HOBO CSVs | CSV | 10/11-depth temperature, 2025-06-02 to 07-10 local, covers tag drag | Inspected 2026-09-24 |
| `Tag Drag Period/TagDrag_WSE.xlsx` | Excel | Collector entrance S/N levels (ft), 06-05 to 06-17; ~0.9 ft below forebay CZD signal | Inspected 2026-09-24 |
| `ATS_3017_Internal_Column_Guide.txt` | Text Guide | Decode File Format 2.0 `Internal` clock/status groups | Inspected |
| `pre_diagnostics.py` / `construct_ATS_dfs` | Legacy Python | Prior ATS raw-file parser reference | Inspected |
| `raw_data/` | ATS raw receiver CSVs | Raw File Format 2.0 detections, GPS rows, Internal clock evidence, SigStr, sensor fields | Available; target parser smoke-tested |

## Durable Structural Facts
### `output/jsats3d_2025_final.db` (verified 2026-09-24, read-only)
- 22.4 GB; `tblDetectionRaw` 59,892,757 rows; no indexes (every per-tag/receiver query is a full scan).
- Rows stored in contiguous per-receiver-file blocks; bounded slices can use `rowid` ranges.
- 37 tags: 20 local beacons, unassigned `1xxx`/`2xxx` beacons, `7F32`, study tags `FFD3`, `FC36`, `0B0A`, `0AC6`, `493F`. `C0FE`, `7F0D`, `0FC7` are NOT present.
- All `tblTag.TagType='raw'`; legacy branches on `'study'`. `FC36`, `0B0A`, `0AC6`, `493F` have NULL `pulseRate`.
- `tblInterpolatedTemp` is `BB_TPU_Surface_t` (single surface sensor), not the approved DD_N string.
- `tblReceiver.Z` = -(config hydrophone depth ft x 0.3048), depth below surface, labeled `Ref_Elev='BM'` while `BM_Elev` is NULL.
- `BitPeriod` (e.g. `240 03/31`), `Threshold`, `SigStr`, `Event` populated. `Pressure`/`Tilt` NULL. `RawTemperature=99.99` sentinel.
- Array-wide beacon candidates `1F14`, `1F38`, `1F5A`, `1F71`, `1F94`, `1FCD` absent: `load_beacon_registry()` drops config rows without a receiver name.
- ZOI03 and ZOI06 have no raw files ~2025-06-17 to 06-27; CFD05 has data only Aug-Sep.

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
- Open the final DB with `file:...?mode=ro` URI; bound queries by `rowid` range and time; never add indexes to the final DB (use a working copy).
- `conda run` drops piped stdin; run scripts from files, not here-strings.

## Project Conventions
- Time basis (verified 2026-09-24): ATS detection timestamps are local PDT (raw header `-07z`); `tblDetectionRaw.seconds` is local wall time encoded as if UTC. ATS GPS Fix rows are UTC. `Temperature/2025_Temp_String_Data_5min_interpolated.csv`, HOBO exports (GMT-07:00), and PI WSE exports (US/Pacific) are local. Proposed `UTC_Conv=-7` pending owner confirmation.
- Coordinates: UTM Zone 10N NAD83 (EPSG:26910), meters.
- Elevations: meters relative to Benchmark (BM) or Water Surface Elevation (WSEL).
- Timestamps: UTC/Local datetime strings converted to float seconds since Unix epoch.

## Persistent Assumptions
- Beacon tag periods in the config workbook describe nominal transmission periods only; they are not authoritative timing references for clock-jump detection or magnitude estimation.
- `FFD3` is mobile validation tag, not stationary receiver beacon.
- Legacy paper/2019/DBSCAN results remain reference outputs until replacement parity is demonstrated.

## Known Limitations
- Missing `SNR`/`NBW`/`FreqOff` blocks only the legacy `multipath_classifier()` (it filters `SNR > 0`). It does NOT block the legacy (time, DDoA) DBSCAN in `clock_fix()` / `notebooks/dbscan_multipath.ipynb`, which needs timestamps, geometry, and sound speed only.
- Beacon multipath verified (48 h slice 06-20/21): 26% of bursts multi-detection; later-arrival lag median 16 ms, p99 126 ms; first arrival strongest SigStr in 87.5% of multi-detection bursts; BitPeriod differs little between direct and reflected copies.
- Local-beacon inter-burst interval median 61.5 s (IQR 59.1-62.8 s); nominal 60 s is not exact.
- Legacy study-tag epoch rule `round((t - first)/pulseRate)` requires an exact PRI; measured study PRIs vary (~3.02-3.35 s by tag), so a cross-receiver epoch method is required.
- Receiver GPS fixes spread 10-81 m in 48 h; not usable as hydrophone geometry.
- `FFD3` pulse rate is approximately 3 seconds; exact interval remains pending measurement from static holds.
- CHN receivers lack GPS / static coordinates in config.
- Synchronization parameters remain provisional pending PM guidance on the new synchronization approach.
- FFD3 staging now includes bounded local-beacon rows, but master receiver and synchronization parameters remain unresolved.
- Beacon PRI is irregular: jitter and slight drift are present, with a catch-up ping approximately every 15.5-16.5 minutes.
- Receiver firmware can create time jumps. Raw files identify synchronization events and flagged one-second jumps, but do not report every jump magnitude.
- Multipath in beacon detections complicates TDOA-based jump and drift estimation.
- The 4-depth DD_N file spans 2025-06-17 to 09-17. Tag-drag temperature gap resolved 2026-09-24 by `Tag Drag Period` HOBO files (06-02 to 07-10). Tag-drag WSE only as collector-entrance level (~0.9 ft offset from forebay CZD); forebay WSE for 06-05 to 06-16 still missing.
- The supplied Internal-column guide applies to File Format 2.0; File Format 3.0 and later require a separate schema.
- Legacy `construct_ATS_dfs` keeps `temp` and `sigStr` but drops `diagCode`/Internal from detection output, so it cannot be reused unchanged for clock-event parsing.
- Experimental `pipeline_mode`, `multipath_interface`, and `sync_readiness` modules were removed; they are not part of the legacy-core approach.
- Target 3D raw receiver set contains 20 SR3017 units, all listed as firmware v10.62F in the configuration workbook.
- June 10 raw folder contains 18 of 20 target serials; CFD05/serial 19033 and ZOI04/serial 20027 are absent from that folder.
- Raw parser supports only verified File Format 2.0 and fails on unsupported formats.
- No approved permanent regression dataset selected yet.

## Owner Escalations (open)
- 2026-09-24: ZOI02 as central clock reference is contradicted by 48 h evidence. 149/151 >300 ms TDoA steps are common-mode across receivers (originate in ZOI02 timestamps); ZOI02 logged 2,449 one-second-adjustment rows in 48 h; 164 jumps vs 21 (CFD04), 28 (ZOI08), 29 (CFD09). Unexplained common ~-400 ms offset mode on ZOI02. Pairwise differencing between non-reference receivers leaves 84-94% of epochs within 0.5 ms. Decision stays in place until owner review.
- 2026-09-24 follow-up: hybrid option (ZOI02 beacon as metronome source, ToT taken from a cleaner clock such as ZOI09/CFD04/ZOI08 minus d/c) cut median jumps 164 -> 20-24 while keeping ZOI02 beacon coverage. ZOI09 alone as metronome: 13 jumps but only 13-14 receivers hear it and 0.5 ms share drops to 0.68. Only 65% of ZOI02 jumps carry a one-second flag. PM leaning ZOI02 (central) or ZOI09 (for floats); unresolved.
- ZOI05 as detecting receiver of 7D2D is chaotic (77% >5 ms outliers); needs field review.
- 2026-09-24 pairwise DBSCAN (`scripts/beacon_pairwise_dbscan.py`, 06-17 to 07-01): ZOI05 clock free-runs ~+/-30 s (clock fault). CFD units show ~126 us residual RMS and ms-scale sawtooth vs ~7 us for ZOI units. CFD01 has a persistent second mode ~21.5 ms late (steady reflection). Fixed DBSCAN parameters (0.5 ms budget, 2.5-period window, min_samples 3) await owner review.

## Completed Milestones
- Formatted 2025 datasets into legacy SQLite schema (`jsats3d_2025_FFD3_manager_demo.db`).
- Verified legacy database loading and temperature interpolation execution.
