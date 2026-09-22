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
