# Session: 2026-09-15 — 2025 Execution Readiness Review

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-15 local workspace date
- Tags: #execution-readiness #metadata #legacy-compatibility #gate-M1 #gate-M2

## Active Context
- Current focus: Determine next steps required before running legacy-compatible 2025 processing.
- Raw source root: `K:\Jobs\5662\001\Data\DataTrans\2025_Data`.
- Existing staging database inspected: `jsats3d_2025_FFD3_manager_demo.db`.

## Findings
- Existing FFD3 staging database contains six tables: `tblDetectionRaw`, `tblInterpolatedTemp`, `tblReceiver`, `tblStudyParameters`, `tblTag`, and `tblWSEL`.
- Existing staging database contains 7,550 detection rows, 26,602 temperature rows, 26,602 WSEL rows, 31 receiver rows, and one tag row.
- It does not yet contain beacon enumeration, multipath-filter, clock-corrected, or Deng-position tables.
- Deployment workbook contains 57 receiver rows and five receiver models: SR3017 (25), WHS4350-L (10), SR3001 (9), SR3000 (7), and WHS4350-S (6).
- Workbook contains 50 non-null beacon tag codes, 35 non-null beacon periods, 33 periods of 60 seconds, and 2 periods of 30 seconds.
- Workbook contains complete latitude/longitude for 30 of 57 rows and hydrophone depth for 31 of 57 rows.

## Decisions & Assumptions
- Do not run legacy metronome, clock-fix, or Deng positioning against current FFD3 staging database.
- Treat workbook hardware counts as an owner-review item because the current inventory is broader than the two-model summary in project context.
- Treat missing coordinates, depths, beacon periods, and synchronization parameters as explicit blockers, not values to infer silently.
- Target legacy-compatible table names and output shapes, not unverified parity with legacy scientific results.

## Parameter Changes With Rationale
- No processing parameter changed.
- No physical value was inferred from incomplete metadata.

## Next Steps
1. Reconcile 57 workbook receiver rows against deployed receiver inventory and raw detections.
2. Resolve coordinate and hydrophone-depth completeness, including vertical datum and benchmark elevation.
3. Resolve all beacon assignments and periods, including receivers without local beacons.
4. Confirm synchronization architecture with PM before populating `masterReceiver` or synchronization windows.
5. Create a validated full beacon staging database after metadata resolution.
6. Produce beacon coverage and epoch reports before filtering or clock correction.

## Validation
- `python -m unittest discover -s tests`: previously passed with 2 tests.
- Raw source files were read only.

## Reconciliation Correction
- The workbook has 57 physical rows but 37 unique receiver names.
- GPS records contain 8 unique receiver names, all present in the configuration workbook.
- The FFD3 test file contains 34 receiver names, including configured receivers without GPS records.
- The earlier statement that the workbook contained 57 receiver rows remains true, but 37 is the correct configured receiver count for receiver-level planning.
- This distinction must be preserved in future reports.
- Added `scripts/audit_receiver_inventory.py` for complete reconciliation, but stopped the full scan before completion because scanning all large detection files was too slow for the immediate receiver-name inventory. No new inventory CSV was produced.
- Existing receiver-name checks remain valid for the test file and initial chunks of the other detection files; rerun the inventory with a deliberately bounded/sample mode if a complete artifact is needed.

## Beacon Coverage Audit
- Scanned the full `master_df_beacon.csv` detection file using only `tagCode` and `receiverName` columns.
- Produced `output/beacon_coverage_audit.csv`.
- Matched 46,139,116 detection rows to 39 unique configured beacon tag codes.
- 37 configured beacon IDs were detected; configured IDs `2008` and `2010` had zero detections.
- Detected beacon coverage ranged from 17 to 35 receivers, with median coverage of 34 receivers.
- Six detected beacon IDs have no local receiver assignment in the configuration workbook and were heard by 32 to 35 receivers. These are array-wide beacon candidates, not yet approved as a synchronization reference.
- All 29 detected beacons with configured 60-second periods and both configured 30-second-period beacons produced detections. Six detected unassigned beacon IDs have missing configured periods.

## Beacon Interpretation Blockers
- Determine whether the six unassigned, broadly detected beacon IDs are the reported high-amplitude array-wide beacon or another beacon class.
- Resolve missing periods and local assignments for unassigned beacon IDs.
- Confirm whether `2008` and `2010` were deployed, renamed, inactive, or absent from the beacon deliverable.
- Do not select a synchronization architecture from coverage alone.

## 3D Receiver Metadata Check
- Verified ZOI01-ZOI04 have latitude, longitude, and hydrophone depth in the workbook and projected X/Y/Z values in the FFD3 staging database.
- The statement that only ZOI01-ZOI04 have the data needed for 3D geometry is not supported by the current configuration workbook.
- Among the 34 active FFD3 receivers, 25 have complete latitude, longitude, and hydrophone-depth metadata: CFD01-CFD09, UPS01-UPS03, UPS05-UPS06, and ZOI01-ZOI11.
- CHN01-CHN05 and DNS01-DNS04 lack complete horizontal/depth metadata in the workbook and are not currently geometry-complete.
- Geometry completeness does not prove positioning readiness. Every candidate receiver still requires verified coordinate datum, vertical datum, clock synchronization, sound-speed handling, and usable detections for a given epoch.
- Treat ZOI01-ZOI04 as a confirmed four-receiver subset, not the complete 3D-capable set, unless the owner supplies a separate rule excluding the other 21 geometry-complete receivers.
- Produced `output/receiver_3d_readiness.csv` from the current FFD3 staging database.
- Current FFD3 staging contains 23 receivers with complete X/Y/Z values, 3 with X/Y only, and 5 with no complete horizontal geometry.
- CFD05 and CFD09 are geometry-complete in the workbook but absent from the FFD3 staging database because FFD3 has no detections at those receivers.
- Therefore, current FFD3 staging has 23 practical 3D geometry candidates, not four.

## FFD3 Controlled-Test Coverage
- Inspected `cowlitz_AT_2025_testing_sheets.xlsx` and extracted windows whose metadata mentions FFD3.
- Produced `output/ffd3_validation_windows.csv`.
- Upstream static holds detected FFD3 on two receivers per window, primarily UPS01-03 and UPS05-06.
- Forebay static holds DB-1 through DB-5 produced 245-335 detections across 17-21 receivers, making them the strongest current candidates for positioning and pulse-interval validation.
- Several Forebay static holds had zero or very few detections.
- Several approach-drag windows had zero detections; later drag windows had only one receiver with detections.
- Do not treat every worksheet window as usable ground truth. Each window requires detection-yield and receiver-geometry checks before positioning validation.
- DB-1 through DB-5 should be prioritized for static-hold validation once synchronization is available.

## 2025 FFD3 Preflight
- Produced `output/2025_preflight_audit.txt`.
- Current FFD3 staging contains 7,550 detections, one tag, 31 receivers, no duplicate rows, and non-null amplitude for every detection.
- Legacy signal fields `SNR`, `NBW`, `FreqOff`, `Pascals`, and `Celsius` are NULL for all 7,550 rows.
- Current staged detections span 2025-06-05 through 2025-06-16.
- Staged temperature and WSEL span 2025-06-17 through 2025-09-17, so they do not cover any current FFD3 detections or the June validation windows.
- This environmental coverage mismatch blocks sound-speed assignment for the current FFD3 positioning validation. Do not extrapolate or silently reuse later temperature values.
- `tblStudyParameters` still has NULL UTC offset, benchmark elevation, master receiver, and synchronization bounds.
- All downstream beacon, filter, clock-fixed, and position tables remain absent.
- Checked separate `Temperature/2025_Temp_String_Data_5min_interpolated.csv`; it contains 26,662 rows from 2025-06-17 through 2025-09-17, confirming the same coverage gap rather than an alternate early-June source.

## Owner Review Artifact
- Added `docs/2025_owner_input_checklist.md`.
- Checklist separates locally confirmed facts from owner-confirmation requests and defines the processing gate before accepted synchronization or positioning.

## Additional Pre-Stop Diagnostics
- Produced `output/ffd3_static_hold_intervals.csv` from DB-1 through DB-5.
- Static-hold receiver-level median intervals had an overall median of approximately 3.326 seconds. DB-1 median was 3.347 seconds; DB-2 through DB-5 ranged approximately 3.301-3.338 seconds.
- Produced `output/ffd3_static_hold_amplitude.csv`.
- DB-1 through DB-5 had no NULL or zero amplitudes in the extracted FFD3 detections. Hold-level amplitude medians ranged 215-218.
- Staged 3D receiver coordinates have matrix rank 3 across 23 receivers. Coordinate spans are approximately 795.6 m X, 423.2 m Y, and 10.1 m Z. The ZOI subset also has rank 3.
- These diagnostics support proceeding with later validation once temperature coverage and synchronization are resolved. They do not authorize sound-speed extrapolation or clock correction.
- Produced `output/receiver_coordinate_audit.csv` after correcting null receiver-name handling.
- Across 37 configured receivers, 8 have dynamic GPS records, 22 rely on configuration latitude/longitude, and 7 lack horizontal coordinates.
- 26 of 37 configured receivers have complete latitude, longitude, and hydrophone-depth metadata; the active FFD3 subset has 25 such receivers before filtering to receivers with detections.
- Coordinate provenance must remain explicit: dynamic GPS medians and configuration coordinates are not interchangeable evidence of the same deployment state.
- FFD3 test data contains an `event` field with 7,373 `True` and 177 `False` values, but its semantics are undocumented and it is not approved as a quality label.
- FFD3 amplitude distribution has median 212, 5th percentile 0, 95th percentile 220, and maximum 222. Receiver-level medians vary widely, reinforcing the need for relative within-epoch amplitude normalization.
- Event-field diagnostic: `event=False` rows have amplitude median 0 and mean approximately 42.4, while `event=True` rows have amplitude median 213 and mean approximately 186.3.
- Event-true fractions vary from 0.0 to 1.0 by receiver, so the field may encode receiver/vendor behavior or a detection condition. It must not be promoted to a universal quality or direct-path label without ATS documentation and validation.
- Produced `output/receiver_coordinate_discrepancy.csv`.
- Compared dynamic GPS medians against projected configuration latitude/longitude for CFD02-CFD09.
- Horizontal discrepancies ranged from approximately 3.5 m to 13.1 m, with median approximately 7.1 m.
- These differences are material to acoustic positioning and may reflect deployment movement, survey timing, coordinate datum, or metadata mismatch. Do not silently choose one source as authoritative.
- GPS movement check shows dynamic coordinate spans up to approximately 261 m east-west and 497 m north-south across CFD02-CFD08 during June 3-September 17.
- GPS records cover the deployment period, so their medians are not necessarily fixed receiver positions. Receiver coordinates must be time-matched to detections or a deployment-state coordinate source must be identified.
- This is an additional positioning blocker, especially for moving or dynamically deployed receivers.
- Produced `output/ffd3_hold_receiver_positions.csv` with GPS records during DB-1 through DB-5 windows.
- GPS coverage during those holds was sparse: 26 receiver/window groups, usually one GPS record each.
- Produced `output/ffd3_hold_coordinate_discrepancy.csv` comparing hold-time GPS positions with projected configuration coordinates.
- Hold-time discrepancies ranged from approximately 1.4 m to 11.9 m, with median approximately 3.0 m. CFD04 was consistently among the largest discrepancies.
- Hold-time results reduce, but do not eliminate, coordinate uncertainty. They are insufficient to declare one coordinate source authoritative.

## Legacy Readiness Boundary
- Current staging tables contain all required base column names for legacy-compatible ingestion.
- Critical staged NULLs remain: `tblTag.pulseRate`; receiver `Tag_ID` for four rows; receiver X/Y for five rows; receiver Z for three rows; and all key study parameters except units.
- Raw detection columns required by ingestion are populated, but legacy signal fields remain entirely NULL.
- All downstream metronome, multipath, clock-fixed, and position tables are absent.
- Independent diagnostics can continue for schemas, coverage, intervals, amplitudes, coordinate provenance, and validation windows.
- Accepted synchronization, filtering, and positioning cannot proceed defensibly until owner inputs resolve environmental coverage, beacon identity/periods, coordinate authority, synchronization architecture, and provenance.

## Legacy Algorithm Boundary
- Prior DBSCAN notebook review produced `output/dbscan_prior_review.txt`.
- Legacy DBSCAN experiment interpolated DDoA on a 37-second grid, calculated Euclidean distances in unscaled time/DDoA space, and used `eps` near 37 with `min_samples=3`. It demonstrates an exploratory method, not a validated 2025 parameter set.
- Legacy `multipath_2()` requires enumerated `transNo`/epoch data and writes rank-based `multipath` flags.
- Legacy `multipath_classifier()` applies `dat = dat[dat.SNR > 0]` before classification. Since every staged 2025 SNR value is NULL, running that classifier against current 2025 data would produce no usable classifier input.
- Legacy clock correction requires populated metronome filtered tables, master receiver, pulse rates, receiver coordinates, WSEL, temperature coverage, and benchmark parameters.
- Legacy Deng positioning requires populated secondary-filter data, corrected timestamps, complete receiver ephemeris, WSEL, benchmark parameters, temperature interpolation, and pulse rate.
- Therefore legacy wrappers can be inspected and contract-tested, but cannot produce accepted 2025 positions by substitution or NULL-field filling.

## Contract and Synthetic Validation Work
- Added `docs/2025_processing_contract.md` covering stage inputs/outputs, provenance, DBSCAN candidate features, rejection accounting, static-hold reports, beacon coverage reports, output compatibility, and hard stops.
- Added `tests/test_2025_contracts.py` with synthetic tests for group-local lag, relative amplitude, epoch rank, epoch-boundary intervals, and preservation of NULL legacy measurements.
- Focused contract tests passed: 3 tests.
- Full test suite passed: 5 tests.
- Modern Python 3 scripts compiled successfully when excluding legacy Python 2 wrappers.
- Full script compilation remains blocked by existing `scripts/projectSetup.py` and `scripts/projectSetup_2018.py` Python 2 print syntax; these are historical wrappers and were not modified.
- No production DBSCAN, synchronization, or positioning parameter was selected.

## Processing Boundary Implementation
- Added `jsats3d/pipeline_mode.py`.
- Added automatic mode detection based on schema and populated legacy signal fields.
- Legacy mode requires populated `SNR`, `NBW`, and `FreqOff` fields.
- ATS-2025 mode accepts the reduced `dateTime`, `tagCode`, `amp`, and `receiverName` schema and reports unavailable legacy fields.
- Explicit legacy requests fail loudly when run against ATS-2025 data.
- Added legacy-shaped output mapping that preserves `NULL` for unavailable metrics.
- Real `master_df_test.csv` sample selects `ats_2025` automatically.
- Synthetic legacy and ATS cases are covered by tests.
- Full suite now passes 10 tests.
- This boundary selects and normalizes modes; it does not yet implement 2025 DBSCAN, synchronization, or positioning.

## Multipath Boundary Implementation
- Added `jsats3d/multipath_interface.py`.
- Added `FilterResult` with method name, input count, retained count, rejected count, and aggregate rejection reason.
- Added `CallableMultipathFilter` for wrapping existing legacy or external filters without hiding their method identity.
- Added `UnfilteredBaseline` for explicit pre-filter diagnostics.
- Added feature-presence validation that fails before filtering when required features are absent.
- Synthetic tests cover baseline accounting, callable-filter accounting, and missing-feature failure.
- Full suite now passes 13 tests.
- No DBSCAN parameters were selected and no legacy algorithm behavior was changed.

## Five-Tag Test Extension
- Extended `scripts/measure_tag_intervals.py` to accept repeated `--tag` arguments.
- Added synthetic multi-tag interval test.
- Selected five high-coverage tags from `master_df_test.csv`: `FFD3`, `FC36`, `C0FE`, `7F0D`, and `0FC7`.
- Produced `output/five_tag_intervals.csv`.
- FFD3 is the only selected tag with confirmed controlled-test metadata in the current validation workbook. The other four tags provide multi-tag coverage diagnostics only, not surveyed-truth validation.
- Full suite passes 14 tests.

## Synchronization Readiness Implementation
- Added `jsats3d/sync_readiness.py`.
- Added preflight checks for required columns, complete geometry, beacon epoch receiver counts, and temperature coverage.
- Validator does not estimate offsets, drift, sound speed, periods, or thresholds.
- Real FFD3 staging result: `ready=False` because no beacon epochs exist and temperature coverage does not span detection times.
- Fixed timestamp conversion by normalizing pandas datetime values to `datetime64[ns]` before Unix-second conversion.
- Focused contract tests pass 14 tests after fix.

## Array-Wide Candidate Analysis
- Active unassigned beacon IDs are `1F14`, `1F38`, `1F5A`, `1F71`, `1F94`, and `1FCD`.
- Workbook contains duplicate rows for these IDs across SR3001, SR3000, and WHS4350 models, but no receiver-name assignment or period.
- `2010` appears on a WHS4350-L row with a configured 60-second period but has zero detections.
- Observed interval analysis for the six active candidates produced 102 tag/receiver summaries. Overall median interval was approximately 62 seconds after excluding gaps over 10 minutes; per-tag median values ranged approximately 68 to 123 seconds.
- Interval results are affected by missed detections and do not justify assigning an exact pulse period.
- Produced `output/arraywide_beacon_intervals.csv`.
- Required owner input: identify these beacon tags, confirm their transmission periods, and explain duplicate model rows and missing receiver assignments.
- Produced `output/beacon_amplitude_audit.csv`.
- Candidate amplitude medians ranged from 167 to 198, while SR3017 local beacon medians generally ranged from 202 to 217. This does not support identifying the six candidates as high-amplitude from the delivered `amp` field alone.
- Several WHS4350 local beacon groups had median amplitude 0 with sparse or receiver-specific detections, so amplitude is not directly comparable across receiver models.
- Amplitude evidence is therefore insufficient to select the array-wide synchronization beacon.