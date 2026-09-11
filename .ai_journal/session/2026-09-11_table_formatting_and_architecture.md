# Session: 2026-09-11 — Architecture Setup & Table Formatting Verification

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-11 UTC
- Tags: #architecture #system_prompt #verification #legacy_compat #milestone-M1

## Active Context
- Files being worked on: `System_Prompt.txt`, `LONG_TERM_CONTEXT.md`, `.ai_journal/long_term/personality.md`, `jsats3d/jsats3d.py`, `scripts/adapt_2025_to_legacy.py`
- Tests / Demos used: `jsats3d_2025_FFD3_manager_demo.db`
- Current focus: Milestone M1 completion — Standardized context scaffolding, verified legacy module execution, formatted manager demonstration database

## Summary
- Established comprehensive AI journal architecture, System Prompt, and Long-Term Context.
- Adapted 2025 Cowlitz datasets into legacy SQLite schema (`jsats3d_2025_FFD3_manager_demo.db`).
- Resolved datetime conversion issue in `jsats3d/jsats3d.py` to support modern pandas `datetime64[ns]` timestamp integers.
- Verified legacy ingestion and temperature interpolation execution.

## Work Completed
- Created detailed `System_Prompt.txt` with full 8-step pipeline context, staged context loading, and strict operational constraints.
- Created `LONG_TERM_CONTEXT.md` capturing dataset structural facts, coordinate datums, and standing decisions.
- Created `.ai_journal/long_term/personality.md` defining communication standards.
- Re-ran `scripts/adapt_2025_to_legacy.py` to produce `jsats3d_2025_FFD3_manager_demo.db` (7,550 detections, 31 receivers, 26,602 temp/wsel records).
- Fixed `jsats3d.temp_interpolator()` in `jsats3d/jsats3d.py` to properly convert datetime objects to Unix seconds without scale errors.
- Verified successful temperature interpolation at test timestamp (`14.29 °C`).
- Conducted table verification queries across all 6 legacy SQLite tables.

## Decisions & Assumptions
- Decision: Use staged context loading policy (read latest session notes + core long-term files) to prevent context exhaustion.
- Decision: Convert WGS84 lat/long coordinates to EPSG:26910 (UTM Zone 10N NAD83, meters) in receiver table.
- Decision: Populate hydrophone depths as negative Z elevation values in meters relative to benchmark.
- Assumption: `BB_TPU_Surface_t` and `NSC.CZD_WTR_EL.F_CV` in the covariate table serve as primary water temperature and water surface elevation inputs for 2025.

## Blockers & Known Limitations
- Missing raw receiver signal metrics (`SNR`, `NBW`, `FreqOff`, `Pascals`) prevent execution of legacy machine-learning multipath classifiers (`multipath_classifier()`).
- Pulse burst rate for test tag `FFD3` remains unconfirmed.
- True 3D positioning via `position.Deng()` remains blocked until clocks are synchronized and multipath is filtered or bypassed.

## What I Wished I Was Told
- The legacy `jsats3d.temp_interpolator()` assumes integer timestamps are already in nanoseconds; when pandas returns timestamps in microsecond or standard int64 format, division by 1e9 can fail bounds checks unless normalized via `.astype("datetime64[ns]").astype(np.int64) / 1e9`.

## Files Touched
- `System_Prompt.txt` (created / updated)
- `LONG_TERM_CONTEXT.md` (created)
- `.ai_journal/long_term/personality.md` (created)
- `jsats3d/jsats3d.py` (edited — timestamp compatibility)
- `scripts/adapt_2025_to_legacy.py` (edited — projected coords & environment table)
- `.ai_journal/session/2026-09-10_data_inventory_and_adapter.md` (updated)
- `.ai_journal/session/2026-09-11_table_formatting_and_architecture.md` (created / updated)

## Affected Living Artifacts
- System_Prompt.txt
- LONG_TERM_CONTEXT.md
- README.md

## Long-Term Promotions
- Staged SQLite schema requirements: `tblTag`, `tblReceiver`, `tblDetectionRaw`, `tblInterpolatedTemp`, `tblWSEL`, `tblStudyParameters`.
- Coordinate transformation standard: EPSG:4326 -> EPSG:26910.
- `temp_interpolator()` patch promoted to main `jsats3d/jsats3d.py`.

## Next Step
- Determine `FFD3` pulse burst rate or test-tag mapping to proceed with clock synchronization (Milestone M3).
