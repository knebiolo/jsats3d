# Session: 2026-09-10 — 2025 Data Inventory & Adapter Development

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-10 UTC
- Tags: #inventory #adapter #schema #ingestion #milestone-M1

## Active Context
- Files being worked on: `scripts/adapt_2025_to_legacy.py`, `scripts/extract_single_tag.py`
- Tests / Demos used: `master_df_test.csv` (FFD3 test tag subset)
- Current focus: Milestone M1 — Ingest 2025 Cowlitz AT data into legacy SQLite tables without modifying source data

## Summary
- Inventoried 2025 Cowlitz acoustic telemetry deliverables on network drive (`K:`).
- Assessed schema compatibility between 2025 files and legacy `jsats3d` SQLite requirements.
- Developed `scripts/adapt_2025_to_legacy.py` to stage detections into legacy tables without modifying raw files.
- Staged test tag `FFD3` and verified SQLite table schemas.

## Work Completed
- Inspected all datasets in `K:\Jobs\5662\001\Data\DataTrans\2025_Data`.
- Identified 7,550 `FFD3` detections in `master_df_test.csv` and 1,180 in `master_df_study.csv`.
- Discovered receiver metadata in `cowlitz_2025_AT_config.xlsx` (37 receivers, beacon codes, periods, depths).
- Discovered 1.1M dynamic receiver GPS records in `master_df_gps.csv` (CFD02–CFD09).
- Discovered 26,688 environmental records in `2025 Master Covariate Table_20251212.csv`.
- Discovered controlled validation tracks in `cowlitz_AT_2025_testing_sheets.xlsx` referencing `FFD3 @ 15'`.
- Built `scripts/adapt_2025_to_legacy.py` with chunked reading, single-tag filtering, WGS84-to-UTM (EPSG:26910) coordinate projection, and environment table population.
- Generated `jsats3d_2025_FFD3_test.db` and `jsats3d_2025_FFD3_formatted.db`.

## Decisions & Assumptions
- Decision: Keep raw network data strictly read-only (`K:\Jobs\5662\001\Data\DataTrans\2025_Data`) — prevent data corruption.
- Decision: Implement standalone adapter script — preserve `jsats3d/jsats3d.py` intact.
- Decision: Keep missing signal columns (`SNR`, `NBW`, `FreqOff`, `Pascals`) as explicit `NULL` rather than fabricated values.
- Decision: Focus initial pipeline validation on single test tag `FFD3` to avoid slow processing of 180M+ total rows.
- Assumption: Beacon periods in config workbook are authoritative.

## Blockers & Known Limitations
- `FFD3` pulse burst rate not documented in config workbook.
- Raw receiver signal metrics (`SNR`, `NBW`) missing from processed 2025 detection files.
- CHN receivers lack static/GPS coordinates in config.

## What I Wished I Was Told
- The full 2025 detection datasets exceed 180 million rows total across study, beacon, test, and unknown files; running a full unindexed database write takes hours and will appear hung without chunking and single-tag filters.

## Files Touched
- `scripts/adapt_2025_to_legacy.py` (created)
- `scripts/extract_single_tag.py` (inspected)

## Affected Living Artifacts
- LONG_TERM_CONTEXT.md
- README.md

## Long-Term Promotions
- Tag `FFD3` identified as controlled test tag (15-ft hold), not stationary beacon.
- 2025 deliverables lack raw receiver signal metrics (`SNR`, `NBW`).
- Dynamic GPS available for CFD02–CFD09 only; CHN stations lack coordinates.

## Next Step
- Run legacy module smoke test and verify temperature interpolation on staged database.
