# Session: 2026-09-14 — PM Guidance on Pulse Rate and Synchronization

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-14 local workspace date; PM response received during current session
- Tags: #manager-guidance #pulse-rate #synchronization #milestone-M2 #milestone-M3

## Active Context
- Files being worked on: `LONG_TERM_CONTEXT.md`, `.ai_journal/session/2026-09-14_pmd_guidance.md`
- Current focus: Incorporate project-manager guidance without advancing blocked synchronization work

## Summary
- PM reports that FFD3 pulse rate is approximately 3 seconds.
- More precise pulse-rate estimation can be obtained with a short code analysis of static holds if required.
- PM expects synchronization parameters may need refinement for the new synchronization approach and will provide more detail later this week.
- PM notes that many requested items are intermediate or calculated from upstream inputs; a meeting should be scheduled once the complete information set is assembled.

## Decisions & Assumptions
- Use approximately 3 seconds as the current FFD3 pulse-rate working value for planning and provisional analysis only.
- Do not treat 3 seconds as an exact blanking interval or silently tune a DBSCAN parameter from it.
- Keep synchronization architecture and parameters provisional until the promised technical details arrive.
- Preserve the requirement for a static-hold measurement if exact FFD3 timing is needed.

## Parameter Changes With Rationale
- No code parameter changed.
- Recorded PM-provided approximate FFD3 pulse rate: 3 seconds.
- Exact pulse interval remains unmeasured; no validation metric exists yet.

## Blockers & Known Limitations
- Exact FFD3 pulse interval is unresolved.
- Synchronization parameters and the new synchronization approach remain unresolved.
- Gate 0 and synchronization validation remain incomplete.

## Next Steps
- Obtain and record PM synchronization details when available.
- Measure FFD3 intervals from static holds if DBSCAN blanking or validation requires higher precision. Added `scripts/measure_tag_intervals.py` for chunked descriptive measurement.
- Assemble required inputs and schedule project meeting before finalizing intermediate calculated products.

## Active Context
- No raw data modified.
- No synchronization or filtering run performed.
- No physical parameter changed; interval analyzer reports observed values without imposing a pulse rate.

## New Run Result
- Ran `scripts/measure_tag_intervals.py` against the read-only raw `master_df_test.csv` for `FFD3`.
- Produced `output/ffd3_intervals.csv`.
- Found detections on 31 receivers.
- Receiver-level median intervals were generally approximately 3.31 to 3.36 seconds where detection counts were substantial.
- Overall result supports PM's approximate 3-second report, with an observed nominal interval near 3.34 seconds.
- Several receivers contain long gaps, producing inflated means and upper quantiles; these represent missed or absent detections and must not be treated as pulse-period estimates.
- `UPS02` includes a sub-3-second interval, so exact blanking behavior still requires static-hold review and epoch-level inspection.

## Validation
- `python -m py_compile scripts/measure_tag_intervals.py`: passed.
- `python -m unittest discover -s tests`: 2 tests passed.

## Gate 0 Audit Result
- Added `scripts/audit_detection_schema.py` as a read-only schema audit.
- Audited all four detection deliverables in sample-only mode: `master_df_test.csv`, `master_df_study.csv`, `master_df_beacon.csv`, and `master_df_unknown.csv`.
- All four files share columns: `dateTime`, `tagCode`, `amp`, `receiverName`, and `event`.
- All sampled timestamps contain 6 fractional digits, consistent with microsecond-formatted output.
- `amp` is the only observed signal field among `amp`, `Amplitude`, `SNR`, `NBW`, `FreqOff`, `Pascals`, and `Celsius`.
- Produced `output/detection_schema_audit_all_samples.txt`.
- Full row counts were intentionally not requested for the large study, beacon, and unknown files; sample-only mode avoids an unbounded scan while preserving schema and timestamp findings.
- Test file full scan found 79,034 rows and 34 receivers; this is total file content, not FFD3-only content.