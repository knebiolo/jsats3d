# Session: 2026-09-17 — Raw ATS Parser Validation

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-17 local workspace date
- Tags: #raw-parser #legacy-tables #clock-events #ZOI02 #milestone-M1 #milestone-M3

## Active Context
- Branch: `ENM_jsat3d_edits`
- Raw root: `K:\Jobs\5662\001\Data\DataTrans\2025_Data\raw_data`
- Raw inputs remained read only.
- Current focus: parse target 3D receivers into legacy-compatible tables while preserving ATS clock and signal evidence.

## PM Direction
- All raw files are uploaded.
- Restrict 3D work to ZOI01-ZOI11 and CFD01-CFD09.
- Select files by exact receiver serials from the configuration workbook.
- Accept `_cleaned`, `_recovered`, and `_recovery` files as corrected source files.
- Corrected variants contain ATS corrupt lines removed by Penny.
- PM requested notification if ATS files show unexpected structure or values.
- PM is available to help narrow the required data before any long full-dataset run.
- PM is unavailable from 12:30-2:00 on 2026-09-17 and available outside that window.

## Target Inventory
- Configuration maps 20 target receivers to SR3017 serials.
- All target rows list firmware v10.62F.
- June 10 array-testing folder contains 18 target serials.
- CFD05/serial 19033 and ZOI04/serial 20027 are absent from that folder.

## Implementation
- Added `scripts/parse_ats_raw_to_legacy.py`.
- Exact filename serial matching prevents partial serial collisions.
- Corrected-file priority: cleaned, recovered/recovery, original.
- Parser supports verified ATS File Format 2.0 only and fails loudly on other versions.
- Parser preserves legacy `tblDetectionRaw` columns and adds ATS fields.
- Parser writes GPS rows to `tblGPSRaw` and clock-event evidence to `tblClockEventRaw`.
- Parser preserves original timestamps; it does not perform clock correction.
- Parser supports `--serial`, `--start`, `--end`, and `--max-files` for bounded runs.

## Preserved ATS Fields
- Raw and decoded Internal groups.
- SigStr, also mapped unchanged to legacy Amplitude.
- Raw temperature, pressure, tilt, battery voltage, bit period, and threshold.
- Receiver model, serial, firmware, and file-format version.
- Source file and source row.
- Offset-change, counter-restart, status-marker, and one-second-adjustment evidence.

## Validation
- SR18076 one-file parse: 150,594 detections, 9,454 GPS rows, 723 clock-event rows.
- ZOI02/serial 18078 bounded June 10 parse: 31,763 detections, 1,884 GPS rows, 209 clock-event rows.
- ZOI02 span: 2025-06-10 00:00:02.269722 through 2025-06-10 12:10:59.248460.
- ZOI02 evidence: 24 offset changes, 44 counter restarts, zero one-second adjustment evidence.
- All ZOI02 detection tag IDs were four-character hexadecimal values.
- SourceFile, SourceRow, Internal, and SigStr were complete for all 31,763 ZOI02 detections.
- Full test suite: 11 passed.
- Parser compilation: passed.
- `git diff --check`: passed.

## Failed or Stopped Run
- Full 18-file June 10 parse was stopped because row-wise parsing was slow and `conda run` buffered progress.
- No incomplete full-run output is accepted as an artifact.
- Full raw five-tag ZOI02 run was stopped because scanning all historical raw files was too slow for the immediate requested database.
- Added multi-tag filtering to `scripts/adapt_2025_to_legacy.py` and built `output/legacy_five_tags_2025.db` from `master_df_test.csv`.
- Final five-tag database contains 21,748 detections, 5 tags, 33 receivers, 26,602 temperature rows, and 26,602 WSEL rows.
- Per-tag counts: FFD3 7,550; FC36 5,058; C0FE 3,947; 7F0D 2,686; 0FC7 2,507.
- This database is complete for five-tag legacy table testing. Raw Internal/GPS/clock-event data remains separate for synchronization work.

## Decisions and Assumptions
- Legacy `jsats3d.py` remains unchanged and authoritative.
- New code is limited to parsing, formatting, diagnostics, and preprocessing that feeds legacy tables.
- Internal markers identify candidate jump boundaries but do not quantify jump magnitude.
- No timestamp correction, drift coefficient, jump magnitude, sound speed, or multipath threshold was selected.

## Next Steps
- Parse remaining target serials in bounded runs.
- Reconcile rows by raw source file.
- Combine parsed detections with legacy metadata/environment tables.
- Review ZOI02 clock-event boundaries against beacon TDOA.
- Obtain missing June 10 temperature/WSEL coverage before accepted sound-speed-dependent analysis.
- Contact PM before launching an opaque or excessively long multi-receiver parse; agree on receiver, tag, and date bounds first.
- Use `output/legacy_five_tags_2025.db` for immediate five-tag legacy workflow testing.
