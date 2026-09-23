# Session: 2026-09-23 — Current-Data Diagnostic Plan

- Author: Ethan Muhlestein / Copilot
- Date: 2026-09-23
- Tags: #diagnostics #beacons #clock-events #dbscan #coverage #geometry #provenance
- Branch: `ENM_jsat3d_edits`

## Scope

Execute all diagnostics possible with current data before new PM data arrives. Read final database only. Do not modify the final database, raw K: files, or apply production filtering.

## Ordered Checklist

1. Beacon coverage audit
   - Which receivers hear each beacon.
   - Detection counts.
   - Missing periods.
   - Array-wide beacon candidates.
2. Beacon epoch diagnostics
   - Inter-detection intervals.
   - Burst sizes.
   - Missed transmissions.
   - Catch-up ping behavior.
   - Internal clock-event boundaries.
3. Clock-event inventory
   - Offset changes.
   - Counter restarts.
   - One-second adjustment evidence.
   - GPS-loss/status markers.
   - Events by receiver and date.
4. DBSCAN feature diagnostics
   - First-arrival lag.
   - Relative SigStr.
   - Epoch rank.
   - Inter-detection interval.
   - K-distance distributions.
   - Candidate eps comparisons.
5. Tag/receiver coverage
   - Detections by tag, receiver, and date.
   - Sparse receivers, especially CFD05.
   - Tags with insufficient coverage.
   - Missing days and deployment gaps.
6. Receiver geometry QA
   - X/Y/Z completeness.
   - Convex-hull geometry.
   - Receiver spacing.
   - Coordinate source comparisons.
7. Signal-quality review
   - SigStr distributions.
   - Zero/sentinel values.
   - Receiver-to-receiver differences.
   - Firmware/model differences.
8. Provenance checks
   - Source files.
   - Source rows.
   - Corrected versus original files.
   - File-format and firmware coverage.
9. Legacy compatibility testing
   - Confirm table names.
   - Confirm required columns.
   - Validate SQLite loading.
   - Prepare clock-fixed and filter output schemas.

## Safety Rules

- No DBSCAN rows rejected.
- No production `eps` or `min_samples` selected.
- No timestamps corrected.
- No sound speed guessed.
- No source files modified.
- Every meaningful result appended to this journal.

## Active Step

Beacon coverage audit.

## Step 1 Result — Beacon Coverage

- Produced `output/current_beacon_coverage.csv`.
- Final DB contains 20 configured local receiver-beacon tags and 37 observed tags total.
- No configured local beacon tag is absent from the final DB.
- Several tags are heard by all 20 target receivers, including `7F32`, `0B0A`, `1002`, `2001`, `2004`, `2006`, `2009`, `FC36`, and `FFD3`.
- Broad receiver coverage alone does not identify an array-wide beacon because study tags are also heard by all 20 receivers.
- Beacon identity must come from the configuration/deployment registry, not detection receiver count alone.
- Detection counts and spans are available in the coverage artifact.

## Current Step

Beacon epoch diagnostics: inter-detection intervals, burst sizes, missed transmissions, catch-up behavior, and Internal-event boundaries.

## Step 2 Result — Beacon Epoch Diagnostics

- Scope: ZOI02 local beacon `7D2D`, 682,708 rows.
- Burst intervals <=1 second: 553,210.
- Burst groups: 128,715; median burst size 5; maximum burst size 14.
- Intervals 30-90 seconds: 128,849, representing nominal-period-like gaps but not proof of exact epochs.
- Intervals 90-300 seconds: 469, consistent with missed or unobserved transmissions.
- Intervals 930-1050 seconds: 7, catch-up-ping-like gaps near 15.5-17.5 minutes.
- Intervals >1050 seconds: 29, representing longer outages/gaps.
- Interval percentiles: p01 approximately 0.0019 s, median approximately 0.0172 s, p95 approximately 62.19 s, p99 approximately 62.58 s.
- Internal event evidence was present on 16,390 rows across offset changes, counter restarts, status markers, and one-second-adjustment evidence.
- Conclusion: raw detection rows must be collapsed into burst-aware epochs. Nominal PRI alone cannot define clean epochs.

## Current Step

Clock-event inventory: offset changes, counter restarts, one-second evidence, GPS/status markers, and event dates by receiver.

## Step 3 Result — Clock-Event Inventory

- Produced `output/current_clock_event_inventory.csv`.
- Produced `output/current_clock_event_summary.csv`.
- Inventory covers all 20 target receivers and groups events by receiver, date, and reason.
- Counts are separated into `offset_changes`, `counter_restarts`, `one_second_evidence`, and `status_markers`.
- Event rows are not equivalent to jump rows: status markers dominate several receivers, especially CFD02, ZOI01, ZOI02, ZOI04, ZOI05, ZOI07, ZOI08, and ZOI11.
- Offset changes and counter restarts are the stronger candidate jump-boundary evidence; one-second evidence is a separate flag.
- No timestamp corrections were applied.

## Current Step

DBSCAN feature diagnostics: first-arrival lag, relative SigStr, epoch rank, inter-detection interval, k-distance distributions, and candidate comparisons.

## Step 4 Result — DBSCAN Feature Diagnostics

- Existing FFD3 diagnostic artifacts: `output/ffd3_dbscan_features.csv`, `output/ffd3_dbscan_summary.csv`, `output/ffd3_dbscan_k_distance.csv`, and `output/ffd3_dbscan_parameter_sweep.csv`.
- Existing ZOI02/7D2D artifact: `output/zoi02_7d2d_dbscan_features.csv`.
- No rows were filtered and no production parameters were selected.
- Representative all-tag k-distance attempt was stopped after reaching large beacon B32A; it did not silently sample or produce a misleading artifact.
- Follow-up representative analysis must require explicit tag/receiver scope.

## Current Step

Tag/receiver/date coverage: detections by tag, receiver, date; sparse receivers; insufficient tags; missing days.

## Step 5 Result — Tag/Receiver Coverage

- Produced `output/current_tag_coverage.csv`.
- Produced `output/current_receiver_coverage.csv`.
- Final DB covers 107 calendar days from June 3 through September 17 with no zero-detection days overall.
- Study-tag coverage varies substantially: `0B0A` has 8,187 rows; `FC36` 46,209; `493F` 107,549; `0AC6` 695,444; `FFD3` 33,471.
- Receiver coverage is highly uneven.
- CFD05 is sparse and concentrated in August 14-September 17, requiring deployment/data review.
- Tag/receiver coverage must be considered before DBSCAN or positioning; sparse combinations cannot support stable epoch or geometry conclusions.

## Current Step

Receiver geometry QA: X/Y/Z completeness, convex-hull geometry, spacing, and coordinate-source comparison.

## Step 6 Result — Receiver Geometry QA

- Final DB contains 20 receivers with complete X/Y/Z.
- Geometry rank is 3.
- Coordinate spans: X approximately 103.6 m, Y approximately 103.2 m, Z approximately 10.1 m.
- Pairwise receiver spacing: minimum approximately 7.8 m, median approximately 61.2 m, maximum approximately 141.9 m.
- Convex-hull volume approximately 29,976.2 cubic meters.
- Vertical geometry is shallow relative to horizontal extent; this must be considered in 3D conditioning and precision evaluation.
- No coordinate source was changed during QA.

## Current Step

Signal-quality review: SigStr distributions, sentinel values, receiver differences, and firmware/model differences.

## Step 7 Result — Signal Quality

- Produced `output/current_signal_quality_by_receiver.csv`.
- Final DB has 59,892,757 non-null SigStr/Amplitude values.
- Overall SigStr: median 208, 5th percentile 164, 95th percentile 221, minimum 45, maximum 223.
- SigStr contains no zero, negative, or NULL values in the final DB.
- `RawTemperature=99.99` is present on all rows; it is a no-sensor sentinel, not a sound-speed input.
- Receiver/model distributions differ: receiver-level SigStr medians range approximately 183-218, so absolute cross-receiver comparisons remain unsafe without gain/threshold metadata.
- Firmware variants are present within target receivers, including 10.49 F, 10.52F, 10.56F, and 10.62F. Firmware grouping must remain explicit during later analysis.

## Current Step

Provenance checks: source files, source rows, corrected/original selection, file format, and firmware coverage.

## Step 8 Result — Provenance Attempt

- Initial provenance audit attempted to materialize all 59.9M rows into pandas and did not produce an artifact; it was stopped/ended before completion because this approach is too memory-heavy.
- No database or raw data changed.
- Provenance QA must be rewritten as a streaming/chunked or SQL aggregate audit before rerunning.
- Do not interpret missing provenance artifact as a data failure.

## Current Step

Legacy compatibility testing: table names, required columns, SQLite loading, and output schema preparation.

## Step 9 Result — Legacy Compatibility

- Final DB contains the six required legacy tables: `tblDetectionRaw`, `tblReceiver`, `tblTag`, `tblInterpolatedTemp`, `tblWSEL`, and `tblStudyParameters`.
- Required legacy columns were verified for each table.
- Legacy interpolation call succeeded via `temp_interpolator` on the final DB.
- `tblStudyParameters` still contains NULL metadata for `UTC_Conv`, `masterReceiver`, `synch_time_start`, and `synch_time_end`; clock-fix routines remain blocked by missing sync metadata.
- No production filtering or database writes were applied during QA.

## Step 10 Result — Explicit-Scope DBSCAN

- Ran a bounded representative DBSCAN check on `FFD3` at `ZOI02` only.
- Output: [output/ffd3_zoi02_representative_kdistance.csv](output/ffd3_zoi02_representative_kdistance.csv)
- Result: 1,452 representative epochs from 2,911,260 source rows, no detections filtered.
- This is diagnostic only; it does not select production `eps` or `min_samples`.
- Broad all-tag sweeps remain disallowed.

## Current Step

Finalize provenance artifact and document the remaining unresolved issue: a memory-safe provenance audit must write an explicit CSV without pandas materialization.

## Step 11 Result — Explicit-Scope DBSCAN Comparison

- Completed the next bounded check: `FFD3` at `ZOI01`.
- Output: [output/ffd3_zoi01_representative_kdistance.csv](output/ffd3_zoi01_representative_kdistance.csv)
- Result: 1,811 representative rows from 4,681,435 source rows and 3,154,762 source epochs.
- The run completed successfully with exit code 0 and reported `No detections filtered`.
- This remains diagnostic-only and is not a production parameter selection.
- Comparison set now includes: `FFD3 / ZOI01`, `FFD3 / ZOI02`, and `7D2D / ZOI02`.
- The repeatable pattern is still explicitly bounded to tag/receiver scope; no all-tag sweep has been reopened.

## Step 12 Result — Bounded Comparison Decision

- Compared the three bounded summary artifacts: `FFD3 / ZOI01`, `FFD3 / ZOI02`, and `7D2D / ZOI02`.
- All three have `lag_p95=0`, `relative_sigstr_p05=0`, and `k_distance_p50`, `k_distance_p90`, and `k_distance_p95` equal to zero.
- The `k_distance_p99` value differs materially for `FFD3`: 0.306715 at `ZOI01` versus 1.059077 at `ZOI02`.
- `7D2D / ZOI02` is fully degenerate through the reported k-distance percentiles, with `k_distance_p99=0`.
- Conclusion: this representative feature set does not provide a stable cross-case basis for selecting production `eps` or `min_samples`.
- Production DBSCAN tuning is stopped pending a validated epoch/feature definition and synchronization metadata.
- Corrected the diagnostic source-count printout so it reports scoped source rows/epochs once rather than multiplying them by representative-row count.
- Revalidated `FFD3 / ZOI01`: 1,811 representative rows, 2,585 source rows, 1,742 source epochs, exit code 0, and no detections filtered.

## Paper Workflow Review

- Reviewed the supplied Nebiolo and Meyer (2021) workflow against the current legacy-compatible 2025 project.
- Paper stages are classified here as ready, provisional, diagnostic-only, design-ready/data-blocked, or blocked.
- The paper confirms the controlling order: environment and receiver geometry, beacon epochs, clock synchronization, receiver-coordinate validation, fish multipath, then fish positioning.
- The paper's moving trash-rack receivers and CPDI warning directly reinforce the need for epoch-by-epoch receiver geometry and explicit trash-rack proximity reporting.
- Current data supports database/provenance preparation, clock-event/TDOA staging, validation manifests, geometry-contract design, and synthetic WSEL-relative-Z tests.
- Current data does not support accepted clock correction, production multipath filtering, or accepted positioning.
- Required owner inputs are listed here; no physical values were invented.

## Cleanup — Validation Scope

- Removed abandoned exploratory DBSCAN scripts and generated DBSCAN artifacts because production tuning is stopped and those outputs are not validation deliverables.
- Retained `scripts/dbscan_diagnostic.py` and `scripts/extract_dbscan_features.py` because repository contract tests and diagnostic-only validation depend on them.
- Removed build logs and the database preview text file.
- Kept the final legacy database, provenance/coverage/clock/signal audits, beacon timing artifact, legacy processing drivers, parser/adapter, tag-drag and temperature tools, tests, paper, and journals.
- No source raw files or final database rows were modified.

## Receiver Motion Readiness

- Added `scripts/receiver_motion_readiness.py` as a read-only report generator.
- Produced `output/current_receiver_motion_readiness.csv` for all 20 target receivers.
- Current `tblReceiver` contains fixed BM coordinates only; motion class, reference time/WSEL, hydrophone surface offset, perpendicular spatial offset, deployment depth, and geometry source remain blank pending owner inputs.
- Report status is `blocked_pending_owner_geometry_inputs` for every receiver.
- Final database was read only; no coordinates or WSEL values were changed.

## Clock/TDOA Readiness

- Added `scripts/clock_tdoa_readiness.py` as a read-only SQL aggregate report.
- Produced `output/current_clock_tdoa_readiness.csv` for all 20 receivers.
- Host-beacon mappings come from `tblReceiver.Tag_ID`; cross-receiver beacon counts are staged for future TDOA work.
- Report includes host-beacon rows, other-receiver rows, raw event counts, offset changes, counter restarts, one-second evidence, status markers, and detection time spans.
- Initial implementation was too slow because it rescanned the 60M-row table per receiver; replaced with aggregate passes and reran successfully.
- Status remains `staged_raw_events_pending_tdoa_and_owner_sync_parameters`.
- No timestamps, detections, or database tables were modified.

## Daily Closeout — 2026-09-23

### Accomplished

- Compared three bounded DBSCAN diagnostics and stopped production parameter tuning because the feature distributions were degenerate and inconsistent across tag/receiver cases.
- Corrected bounded DBSCAN source-count reporting before cleanup.
- Reviewed the supplied Nebiolo and Meyer (2021) paper and aligned the project order: environment/geometry, beacon epochs, synchronization, receiver validation, fish multipath, then positioning.
- Removed superseded exploratory DBSCAN scripts, representative outputs, build logs, and preview text.
- Preserved required diagnostic modules because repository contract tests depend on them.
- Created receiver-motion readiness report for all 20 receivers.
- Created clock/TDOA readiness report for all 20 receivers using aggregate SQL.
- Confirmed final database and raw source data remained unchanged.
- Ran 13 repository tests successfully.
- Ran Python compilation and `git diff --check` successfully.

### Current State

- Final legacy database remains available at `output/jsats3d_2025_final.db`.
- Receiver geometry is fixed BM geometry only; moving-receiver inputs remain absent.
- Clock/TDOA evidence is staged but not corrected.
- No production DBSCAN parameters exist.
- No detections were filtered.

### Hard Blockers

- Approved validation/study tag list.
- Moving/trash-rack receiver measurements, WSEL relationship, offsets, depths, and drawings.
- Approved synchronization reference and TDOA rules.
- Validated clock jump/drift model.
- June 5-16 temperature/WSEL coverage.
- Operations data and extended tag-drag validation inputs.

### Next Session

- Review the two readiness CSVs.
- Prepare validation-tag manifest.
- Define synthetic WSEL-relative receiver-Z tests.
- Do not modify final database or begin accepted positioning until owner inputs arrive.

