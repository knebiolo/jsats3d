# Session: 2026-09-22 — Final Legacy Database and Data Audit

- Author: Ethan Muhlestein / Copilot
- Date: 2026-09-22
- Tags: #legacy-database #raw-parser #five-tags #data-quality #temperature #tag-drag #cleanup
- Branch: `ENM_jsat3d_edits`

## Purpose

Complete one consolidated legacy-formatted SQLite database for the selected 2025 tags, preserve ATS raw fields in the legacy detection table, audit the resulting data, and remove superseded local database artifacts without touching original data.

## Final Database

- Final path: `output/jsats3d_2025_final.db`
- One SQLite database remains in local `output/`.
- Original K: drive raw files were never modified.
- Intermediate local databases and loose diagnostic outputs were removed after final build validation.
- Existing `output/positioning/` repository content was retained.

## Final Table Counts

- `tblDetectionRaw`: 2,823,380 rows.
- `tblReceiver`: 20 rows.
- `tblTag`: 5 rows.
- `tblInterpolatedTemp`: 26,602 rows.
- `tblWSEL`: 26,602 rows.
- `tblStudyParameters`: 1 row.

Selected tags:

- `FFD3`: 33,471 detections.
- `FC36`: 46,209 detections.
- `C0FE`: 1,955 detections.
- `7F0D`: 2,415,731 detections.
- `0FC7`: 326,014 detections.

Selected receivers:

- ZOI01-ZOI11.
- CFD01-CFD09.
- All 20 target serials were found during final build.
- 285 corrected/original target raw files were processed.

## Legacy Table Contract

`tblDetectionRaw` retains required legacy fields:

- `timeStamp`
- `seconds`
- `Tag_ID`
- `Rec_ID`
- `FreqOff`
- `Amplitude`
- `NBW`
- `SNR`
- `Valid`
- `Pascals`
- `Celsius`

The same table now carries ATS additions:

- Receiver/model metadata: `ReceiverType`, `FirmwareVersion`, `FileFormatVersion`, `SerialNumber`.
- Provenance: `SourceFile`, `SourceRow`.
- Raw ATS clock field: `Internal`.
- Decoded Internal groups, flags, counter, offset, and status.
- `InternalPositionDifferenceSeconds`.
- `OneSecondAdjustmentEvidence`.
- `ClockStatusMarker`.
- `Event`.
- `SigStr`.
- `RawTemperature`, `Pressure`, `Tilt`, `BatteryVoltage`, `BitPeriod`, `Threshold`.
- `OffsetChanged`, `CounterRestart`, `ClockEventReasons`.
- `GPSFixTimeStamp`, `GPSFixLatitude`, `GPSFixLongitude`.

No separate GPS or clock-event table is required for final output. Those additions are stored in `tblDetectionRaw`, as requested. This matches legacy database count and structure while extending columns.

## Parser Changes

- Kept legacy `jsats3d.py` unchanged.
- Added raw ATS File Format 2.0 parser.
- Selected exact receiver serials from configuration workbook.
- Preferred `_cleaned`, then `_recovered`/`_recovery`, over original files.
- Added five-tag filtering.
- Added faster positional raw parsing.
- Replaced per-cell pandas conversions with standard-library parsing.
- Added parallel raw-file processing.
- Added batched SQLite inserts.
- Added metadata-table creation in `--legacy-db` mode.
- Parser preserves original timestamps. No clock correction was applied.

## Build Performance

- Original row-wise approach was too slow for the full raw season.
- Optimized parser reduced the bottleneck substantially.
- Full build completed with all 285 target files.
- Final build wrote one consolidated database.

## Data Audit Findings

- Core detection timestamps, tag IDs, receiver IDs, Internal values, SigStr, and provenance were populated.
- No duplicate source-row groups were found.
- No out-of-range timestamps were found.
- Receiver X/Y/Z geometry was complete for all 20 final receivers.
- All 107 detection days from June 3 through September 17 contained detections.
- `RawTemperature` was `99.99` on every row. This is a no-sensor sentinel, not valid water temperature.
- `Pressure` and `Tilt` were `N/A` in sampled raw SR3017 files.
- `VBatt` contained real receiver battery values.
- `SNR`, `NBW`, and `FreqOff` remain unavailable and NULL.
- Four tags have no configured pulse rate in `tblTag`; only FFD3 has provisional `3.33` seconds.
- CFD05 has only 9 detections, all in August, and requires field/data review.
- GPS fix fields are NULL for some early rows before a file's first GPS fix; this is expected provenance behavior.

## Temperature and Tag-Drag Audit

Temperature string:

- File: `Temperature/2025_Temp_String_Data_5min_interpolated.csv`.
- Four DD_N depths are available: `DD_N_0p5`, `DD_N_1p5`, `DD_N_9`, `DD_N_18`.
- Coverage: June 17 through September 17, 2025.

Tag-drag GPS:

- File: `5_array_testing/array_testing_drag_GPS.csv`.
- Coverage: June 5, 11:50 through June 10, 11:45, 2025.
- All drag tracks occur before temperature-string coverage begins.
- Therefore no drag-test timestamp has authoritative temperature coverage.
- This is a source-data coverage gap, not a parser/date error.
- Early-June temperature/WSEL data or an approved alternate source is still required for sound-speed-based validation.

## Controlled Test Dates

- June 5: FBY-NS-01, FBY-SN-01, FBY-NS-02, FBY-SN-02.
- June 10: ENT-01 through ENT-05 and FBY-D-1 through FBY-D-4.
- CFNSC drag worksheet events also occur June 10 around 16:10-16:21.

## Study Parameters Still Unresolved

`tblStudyParameters` currently has:

- `BM_Elev_Units = feet`.
- `Output_Units = meters`.
- `UTC_Conv = NULL`.
- `BM_Elev = NULL`.
- `masterReceiver = NULL`.
- Synchronization start/end = NULL.

No physical value was invented.

## Clock-Synchronization Context

Meeting decisions remain active:

- Use ZOI02 as central reference.
- Do not use nominal beacon PRI to identify jumps or quantify jump magnitude.
- Beacon PRI has jitter, drift, and catch-up pings around 15.5-16.5 minutes.
- Use raw Internal markers and flags to identify candidate jumps.
- Estimate jump magnitudes using beacon TDOA.
- Correct ZOI02 jumps without adjusting ZOI02 drift.
- Use piecewise regressions for other receivers' jumps and drift.
- Preserve corrected and original times separately when synchronization begins.

## Cleanup

Removed local intermediate/test databases and loose previews. Kept only:

- `output/jsats3d_2025_final.db`.
- Existing `output/positioning/` content.

The K: drive databases generated in earlier sessions were identified as intermediate artifacts. Their removal was left to the user because they reside on the shared K: drive.

## Validation

- Final database table preview generated.
- Final database row counts verified.
- Final database columns verified.
- Full test suite: 11 tests passed after parser optimization.
- Parser compiled successfully.
- `git diff --check` passed.

## Next Steps

1. Provide PM with final database path and table summary.
2. Obtain early-June temperature/WSEL coverage.
3. Confirm pulse rates for the four non-FFD3 tags if epoch processing requires them.
4. Parse/use beacon detections for synchronization while retaining them in the same legacy table contract.
5. Populate owner-approved synchronization parameters.
6. Run ZOI02 reference-clock analysis.
7. Apply clock corrections only after TDOA validation.
8. Run legacy positioning and controlled-test validation.

## Script Cleanup

Removed one-off audit/inspection helpers not needed for the current legacy workflow:

- `audit_detection_schema.py`
- `audit_final_db.py`
- `audit_receiver_inventory.py`
- `inspect_formatted_tables.py`
- `legacy_readiness_audit.py`
- `measure_tag_intervals.py`

Retained production/current workflow scripts:

- `parse_ats_raw_to_legacy.py`
- `adapt_2025_to_legacy.py`
- Legacy processing drivers for temperature, beacon/metronome, clock fixing, positioning, validation, and reporting.

Validation after cleanup: 10 tests passed, retained adapters compiled, and `git diff --check` passed.

## DBSCAN Readiness Update

- Confirmed final study-only DB did not contain receiver-beacon detections.
- Added `--include-config-beacons` to the parser.
- Rebuilt candidate database `output/jsats3d_2025_with_beacons.db` with the five study tags plus configured local receiver-beacon tags.
- Candidate contained 37 tags and 59,892,757 `tblDetectionRaw` rows across the same six legacy tables.
- Beacon rows remain in `tblDetectionRaw`; no separate beacon table was introduced.
- DBSCAN work is now possible diagnostically because beacon detections and `SigStr` are present.
- No DBSCAN filtering was applied, no rows were rejected, and no `eps` or `min_samples` was selected.
- Diagnostic DBSCAN must first construct epochs and features: first-arrival lag, relative `SigStr`, epoch rank, and inter-detection interval.
- Nominal beacon PRI must not be treated as exact because meeting notes document jitter, drift, and catch-up pings.
- Final promoted database now includes the beacon-inclusive build after validation.
- Correction to earlier study-only counts: the current final `output/jsats3d_2025_final.db` is the beacon-inclusive database, not the earlier 2,823,380-row study-only database. Current `tblDetectionRaw` count is 59,892,757 and `tblTag` count is 37.

## Process Discipline

- Journal updated during each meaningful work phase.
- Raw K: files remained read only.
- New algorithm work stops at unresolved physical parameters and owner gates.

## DBSCAN Diagnostic Start

- Added `scripts/extract_dbscan_features.py`.
- This is diagnostic-only: it does not reject detections, change the database, or select DBSCAN parameters.
- It requires an explicit `--period-seconds` value; no PRI is guessed.
- It extracts receiver-local epoch number, lag from epoch first arrival, relative SigStr, epoch rank, and inter-detection interval.
- Added synthetic feature tests.
- Focused tests passed: 9 tests.
- Ran first real pass on FFD3 using measured provisional PRI `3.33` seconds.
- Extracted 33,471 FFD3 rows and 27,317 receiver-local epochs.
- Receiver coverage included all 20 target receivers; CFD05 contributed only 2 FFD3 feature rows, confirming its sparse detection issue.
- Created local diagnostic artifact `output/ffd3_dbscan_features.csv`; it is not a production filter result.
- No `eps`, `min_samples`, rejection threshold, or filtering decision was selected.

## Safe DBSCAN Diagnostic Pass

- Added `scripts/dbscan_diagnostic.py`.
- Added exploratory k-distance output using `min_samples=3` only as a diagnostic neighborhood size; it is not a production setting.
- Exploratory features were standardized using study-derived mean/std solely for visualization/diagnostic comparison. This transform is not approved for production use.
- Ran on FFD3 feature data with explicit PRI `3.33` seconds.
- Complete feature rows: 33,471.
- Receiver-local epochs: 27,317.
- Produced `output/ffd3_dbscan_summary.csv` and `output/ffd3_dbscan_k_distance.csv`.
- No detections were filtered, rejected, or written back to the database.
- No production `eps` was selected.
- Diagnostic output shows lag is usually near zero, with receiver-level 95th-percentile lag generally below approximately 0.054 seconds; relative SigStr lower tails vary by receiver. These are descriptive results, not a filter threshold.
- Safe DBSCAN tests passed: 10 tests total.

## Exploratory DBSCAN Parameter Sweep

- Added exploratory sweep support to `scripts/dbscan_diagnostic.py`.
- Tested `eps` values 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, and 2.0 in standardized diagnostic feature space.
- Tested `min_samples` values 2, 3, and 4.
- Produced `output/ffd3_dbscan_parameter_sweep.csv`.
- Noise fractions ranged from approximately 0.003% to 0.418% across this candidate grid.
- At `eps=0.25`, noise fractions were approximately 0.221% (`min_samples=2`), 0.287% (`3`), and 0.418% (`4`).
- At `eps=1.0`, noise fractions were approximately 0.009% (`2`), 0.027% (`3`), and 0.030% (`4`).
- These results are exploratory only. Standardization used study-derived mean/std and is not a physically approved transform. No candidate pair was selected, no rows were rejected, and the production database was unchanged.

## Beacon DBSCAN Diagnostic Pass

- Ran the same diagnostic extractor on ZOI02 local beacon `7D2D` using its configured 60-second period as a provisional grouping interval only.
- Produced `output/zoi02_7d2d_dbscan_features.csv`.
- Extracted 682,708 rows across 129,483 provisional epochs.
- Epoch size median: 5 detections; 95th percentile: 8; maximum: 14.
- Lag median: approximately 0.056 seconds; 95th percentile: approximately 0.087 seconds; maximum: approximately 28.626 seconds.
- Relative SigStr median: -22; 5th percentile: -61; maximum: 0.
- 15 provisional epochs had maximum lag above 1 second.
- These results show meaningful multipath/epoch complexity, but the nominal 60-second interval is not approved as exact because meeting notes document beacon jitter, drift, and catch-up pings.
- No DBSCAN rows were rejected. No `eps` or production `min_samples` was selected.
