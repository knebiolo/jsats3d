# Session: 2026-09-16 — Clock Synchronization Meeting

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-16 local workspace date
- Tags: #clock-sync #raw-parser #time-jumps #beacons #tdoa #piecewise-regression #milestone-M3

## Active Context
- Current focus: Record owner-selected synchronization direction before implementation.
- Planned source: raw receiver files, read only.
- Planned output: legacy-compatible SQLite tables and synchronization diagnostics.
- No synchronization code or physical parameter changed during this entry.

## Meeting Findings
- Beacon transmission intervals are not sufficiently consistent for identifying receiver time jumps or quantifying jump magnitude.
- Beacon PRI contains substantial jitter and slight drift.
- Beacon tags emit a catch-up ping approximately every 15.5-16.5 minutes.
- Receiver firmware creates occasional clock time jumps.
- Raw receiver files identify clock synchronization events and contain flags for reported one-second jumps.
- Jump events can therefore be identified from raw-file synchronization events or one-second jump flags, but jump magnitude is not generally reported.
- Multipath in beacon detections adds ambiguity to time-difference-of-arrival measurements.
- Legacy positioning mathematics and legacy table formats remain usable for 2025 after the new parsing and synchronization stages produce suitable inputs.
- PM supplied the authoritative temperature file and specified the four `DD_N` depth columns.
- PM supplied an ATS SR3001/SR3017 Internal-column guide for File Format 2.0.
- PM supplied prior parser code, specifically `construct_ATS_dfs`, as a reference implementation.
- PM noted preliminary evidence that raw `SigStr` may help reject some multipath, but not necessarily all multipath.

## Supplied Source Audit
- Temperature file: `K:\Jobs\5662\001\Data\DataTrans\2025_Data\Temperature\2025_Temp_String_Data_5min_interpolated.csv`.
- Authoritative columns: `DD_N_0p5`, `DD_N_1p5`, `DD_N_9`, and `DD_N_18`.
- All four columns contain 26,662 non-null rows.
- Temperature coverage remains 2025-06-17 through 2025-09-17, so the supplied file does not resolve temperature coverage for the 2025-06-10 test day.
- Internal guide: `K:\Jobs\5662\001\Background\Instrument Information\ATS_3017_Internal_Column_Guide.txt`.
- The guide applies to File Format 2.0 receiver CSVs. Newer File Format 3.0 and later layouts differ and must be audited separately.
- Prior parser: `K:\Jobs\5662\001\Analysis\Scripts\Drew_Scripts\LambdasFucntions_Old\pre_diag_at_detections_chelan\pre_diagnostics.py`.
- `construct_ATS_dfs` splits multiple header/file sections, reads ATS rows, extracts GPS rows, removes GPS/status pseudo-tags from detections, parses timestamps, and trims tag codes.
- The prior parser reads `diagCode`, `temp`, and `sigStr`, but drops `diagCode` from its detection output. It therefore cannot be reused unchanged because the new clock-sync design requires the Internal/diagnostic groups.

## Internal Column Facts
- Internal is 25 characters with six groups for File Format 2.0.
- Group 1 identifies clock/status events including `001111`, `GPS111`, `RTC222`, `0000SL`, GPS-disabled `xxxxGD`, and reset/start markers.
- Group 2 gives SR3017 sub-second position within the 15-second count. An exact one-second mismatch against DateTime identifies a one-second detection-time adjustment, including unflagged 10.63 cases.
- Group 4a carries explicit 10.62 clock-check flags such as `1`, `>`, `F`, `=`, and `A`.
- Group 4b is a record counter; restarts to `000` or `001` identify likely synchronization/reset boundaries.
- Group 5 is the time offset used to construct detection time. Offset changes identify candidate jump boundaries but do not report jump magnitude.
- Group 6 records last sync/reset state. `D` and recovery states require special review; `F` is commonly normal after synchronization.
- GPS position rows do not follow the six-group Internal layout and must be parsed separately using tag values `GPS Fix` or `GPS Clock`.

## SigStr Decision
- Preserve raw `SigStr` in parser output with source units and receiver provenance.
- Treat `SigStr` as a candidate feature only.
- Do not adopt a SigStr threshold or clustering transform without beacon/static-hold validation and recorded before/after metrics.

## Physical Reasoning
Clock correction must separate discontinuous time jumps from continuous clock drift. A time jump shifts all later arrivals abruptly, while oscillator drift accumulates gradually between jumps. Beacon PRI cannot provide a stable clock because transmitter jitter, transmitter drift, and periodic catch-up pings can resemble receiver-clock changes. The planned method therefore uses raw receiver clock-event evidence to locate jump boundaries and acoustic TDOA to estimate correction magnitude. A 1 ms unresolved timing error is approximately 1 m of positioning error, so jump and drift residuals must be reported at sub-millisecond scale.

## Selected Synchronization Direction
1. Obtain and parse raw receiver files.
2. Identify synchronization events and flagged one-second jumps for each receiver.
3. Use ZOI02 as the central clock reference.
4. Use TDOA from the ZOI02 beacon observed at surrounding receivers to estimate ZOI02 time-jump magnitudes.
5. Correct ZOI02 time jumps without adjusting ZOI02 drift.
6. For every other receiver, use beacon TDOA to estimate time-jump magnitudes and clock drift within the periods bounded by jumps.
7. Fit piecewise regressions within those stable periods.
8. Apply corrections to beacon data and assess residuals.
9. Apply accepted corrections to test-tag data for 2025-06-10.
10. Run corrected test-tag data through the 3D positioning workflow and evaluate performance.

## Architecture Decisions
- Raw-file parsing is now required because concatenated detection CSVs omit clock synchronization events and jump flags.
- Parser output should use the legacy table format so existing downstream code and the legacy benchmark remain available.
- Nominal beacon PRI must not drive jump identification or jump-size estimation.
- ZOI02 is the selected central clock reference.
- ZOI02 drift must not be adjusted under the selected plan; only identified time jumps are corrected.
- Other receivers use piecewise jump and drift corrections derived from beacon TDOA.
- Beacon multipath must be addressed or explicitly accounted for before fitting synchronization regressions.

## Legacy-Core Reset
- User confirmed the legacy package must remain the only core processing implementation.
- Removed experimental `jsats3d/pipeline_mode.py`, `jsats3d/multipath_interface.py`, and `jsats3d/sync_readiness.py`.
- Removed the stale `docs/2025_processing_contract.md` architecture document.
- Removed architecture-only tests while retaining adapter-formatting tests.
- New development is limited to raw ATS parsers, table-format adapters, synchronization preprocessing, diagnostics, and tests that produce data in legacy-compatible tables.
- The legacy `jsats3d/jsats3d.py` core was not modified during this reset.

## Legacy Table Extension Contract
- Required legacy detection columns remain unchanged: `timeStamp`, `seconds`, `Tag_ID`, `Rec_ID`, `FreqOff`, `Amplitude`, `NBW`, `SNR`, `Valid`, `Pascals`, `Celsius`, and `TagTypeSource`.
- Added optional ATS columns: `Event`, `Internal`, `SigStr`, `RawTemperature`, `Pressure`, `Tilt`, `BatteryVoltage`, `BitPeriod`, `Threshold`, `ReceiverType`, `FirmwareVersion`, `FileFormatVersion`, `SourceFile`, and `SourceRow`.
- Processed deliveries using `amp` continue to populate legacy `Amplitude`.
- Raw ATS rows using `sigStr` populate legacy `Amplitude` with the unchanged raw value and also retain `SigStr`.
- Missing legacy fields remain NULL.
- Focused adapter tests passed after the reset and raw-field mapping.

## Validation After Legacy-Core Reset
- Full test suite passed: 8 tests.
- Modified adapter and readiness-audit scripts compiled successfully.
- No live source, script, test, or current documentation imports the removed architecture modules.
- Real FFD3 formatter smoke test produced `output/legacy_formatter_smoke.db` from 7,550 detections.
- `tblDetectionRaw` contains all required legacy columns plus the approved additive ATS columns.
- The smoke database staged 23 complete receivers and reported 8 excluded receivers with explicit missing-field reasons.
- `git diff --check` passed.

## 2026-09-17 Raw Data Availability
- PM confirmed all raw data is uploaded under `K:\Jobs\5662\001\Data\DataTrans\2025_Data\raw_data`.
- PM restricted 3D processing to ZOI01-ZOI11 and CFD01-CFD09.
- Configuration workbook maps these names to 20 exact SR3017 serial numbers, all listed as firmware v10.62F.
- Raw filename selection uses exact `SR<serial>` matching to prevent partial matches such as `SR18078` matching `SR18078250610...`.
- Files ending `_cleaned`, `_recovered`, or `_recovery` are accepted corrected inputs. When an original and corrected file share a stem, selection priority is `_cleaned`, then recovered/recovery, then original.
- June 10 array-testing folder contains 18 of the 20 target serials. CFD05/19033 and ZOI04/20027 are absent from that folder.

## Raw Parser Implementation
- Added `scripts/parse_ats_raw_to_legacy.py` for verified ATS File Format 2.0.
- Parser reads raw files without modifying them and rejects unsupported file formats.
- Parser writes legacy-compatible detections to `tblDetectionRaw`.
- Parser writes GPS pseudo-tag rows to `tblGPSRaw`.
- Parser writes identified clock-event evidence to `tblClockEventRaw`.
- Legacy detection columns remain present; raw `SigStr` populates legacy `Amplitude` unchanged while also remaining in `SigStr`.
- Parser preserves raw Internal, decoded Internal groups, clock status markers, offset-change evidence, counter restarts, one-second adjustment evidence, raw sensors, receiver/firmware/file-format metadata, source file, and source row.
- Parser identifies candidate clock-event boundaries but does not estimate jump magnitude or alter timestamps.

## Raw Parser Validation
- One-file SR18076 parse succeeded: 150,594 detections, 9,454 GPS rows, and 723 clock-event rows across the source file span.
- Bounded ZOI02/serial 18078 parse for 2025-06-10 succeeded: 31,763 detections, 1,884 GPS rows, and 209 clock-event rows.
- ZOI02 parsed span was 2025-06-10 00:00:02.269722 through 2025-06-10 12:10:59.248460.
- ZOI02 contained 24 offset changes and 44 counter-restart rows in the bounded interval. No one-second adjustment evidence was found in that interval.
- All 31,763 parsed ZOI02 detections had four-character hexadecimal tag IDs, complete source-file/source-row/Internal provenance, and non-null SigStr.
- Full 18-file June 10 run was stopped because row-wise parsing was too slow and progress was buffered. No result from that incomplete run is treated as an artifact.
- Added bounded `--serial` selection for observable per-receiver runs.
- Full test suite passed: 11 tests. Parser compiled successfully and `git diff --check` passed.

## Next Steps After Raw Parser Validation
- Run bounded per-serial parses for the remaining target receivers with visible progress.
- Reconcile detection, GPS, and clock-event counts per source file.
- Join parser output with legacy `tblTag`, `tblReceiver`, `tblInterpolatedTemp`, `tblWSEL`, and `tblStudyParameters` staging.
- Review ZOI02 offset changes and counter restarts against beacon TDOA before estimating jump magnitudes.
- Do not correct timestamps until event semantics and TDOA estimates pass owner review.

## Parameter Changes With Rationale
- No numeric synchronization parameter was selected.
- The reported 15.5-16.5 minute catch-up interval is observational context, not a filter threshold or epoch parameter.
- No drift coefficient, jump magnitude, regression window, residual threshold, sound speed, or beacon period was invented.

## Blockers & Required Inputs
- Obtain representative raw files for every receiver model used in the selected array.
- Document raw-file schemas, clock-event records, one-second jump flags, timezone representation, and timestamp resolution.
- Confirm exact raw-file field values that distinguish GPS synchronization, clock synchronization events, and one-second jump flags.
- Confirm which surrounding receivers should be paired with the ZOI02 beacon for reference-clock jump estimation.
- Confirm how beacon multipath will be excluded before piecewise regression fitting.
- Obtain authoritative temperature and WSEL coverage for the 2025-06-10 test-tag analysis before sound-speed-dependent residual or positioning acceptance.
- Obtain actual raw receiver files and enumerate their File Format and firmware versions before selecting parser layouts.
- Obtain a guide or representative files for any File Format 3.0+ receivers.
- Clarify whether another DD_N temperature deliverable covers June 10, because the supplied authoritative file starts June 17.

## Validation Plan
- Reconcile parser row counts and event counts against every raw input file.
- Produce per-receiver jump-event tables with source-file provenance.
- Produce per-segment piecewise regression coefficients, sample counts, time spans, and residual distributions.
- Flag residuals exceeding 0.5 ms and continue with traceable warnings.
- Compare corrected beacon residuals before and after each correction stage.
- Preserve uncorrected timestamps alongside corrected timestamps.
- Run 2025-06-10 test-tag positions only after synchronization corrections pass owner review.

## Files Touched
- `LONG_TERM_CONTEXT.md`
- `.ai_journal/session/2026-09-16_clock_sync_meeting.md`

## Next Steps
- Obtain raw receiver files.
- Audit raw schemas by receiver model.
- Design the receiver-type-aware raw parser and clock-event table contract.
- Do not implement jump correction until raw event semantics are verified.