# Session: 2026-09-29 — v2 Cleanup and v3 Run-File Commit

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 09:29 local PDT (UTC-7) = 16:29 UTC
- Tags: #cleanup #v3 #run-file #git
- Branch: `ENM_jsat3d_edits`
- New session file: more than 60 min since the last 2026-09-28 entry (System Prompt Section 13). The v3 build, sweep and coverage results are in `2026-09-28_kevin_approvals_clock_resets.md`.

## Active Context
- Database: `output/jsats3d_2025_v3.db` (25.4 GB, 60,410,185 detections, 303 files, 20/20 receivers). This is now the only project database.
- Env: `jsat_3d`. Tests: 28 pass (with the uncommitted parser fix and its test in the working tree).
- Current focus: repo tidy-up after the v3 build.

## Short Summary
- User approved deleting the old databases. Deleted:
  - `output/jsats3d_2025_v2.db` (23.2 GB). It is superseded by v3: same code path plus the 18 ZOI03/ZOI06 daily files.
  - The v2-based output folders `output/dbscan_ddoa_0620/`, `output/dbscan_sweep_0620/` and `output/beacon_coverage/`. Superseded by `output/dbscan_jsats3d_2025_v3/`, `output/dbscan_sweep_v3/` and `output/beacon_coverage_v3/`.
  - All of these were local, gitignored outputs. No raw data (K:) touched.
- Remaining in `output/`: `jsats3d_2025_v3.db`, `jsats3d_2025_v3.run.json`, `jsats3d_2025_v3_build.log`, `dbscan_jsats3d_2025_v3/`, `dbscan_sweep_v3/`, `beacon_coverage_v3/`, `cfd_gps/`, `positioning/` (pre-project, kept).
- Ran the full test suite: 28 pass.
- Committed `config/run_data.toml` (v3 run settings) and the session journals.
- NOT committed, per user (Ethan will commit manually):
  - `scripts/parse_ats_raw_to_legacy.py`: D-file regex and `--workers`.
  - `tests/test_2025_contracts.py`: daily-file test case. Held back with the parser because it fails against the old regex.

## Files Touched
- This journal (new).
- Committed: `config/run_data.toml`, `.ai_journal/session/2026-09-28_kevin_approvals_clock_resets.md`, this file.
- Left uncommitted: `scripts/parse_ats_raw_to_legacy.py`, `tests/test_2025_contracts.py`.

## Decisions & Assumptions
- "Old databases" read as v2 plus the outputs derived from it; v3 is the working database.
- The test change is held with the parser fix so no commit contains a failing test.

## Parameter Changes With Rationale
- None.

## Blockers & Known Limitations
- Unchanged from 2026-09-28: CFD05/CFD09 serial swap unfixed; Kevin approval list outstanding (reference clock, `master_receiver = ZOI08` vs 7D2D coverage ranking, `signal_proxies`, DBSCAN parameters, steady-reflection over-labelling, sound-speed source, Deng in-hull bug); legacy run blocked (no `jsat_legacy` env, `bm_elev` blank).
- Parser worker WARNING capture still open.

## Next Steps
1. Ethan commits the parser fix and its test together.
2. Send the Kevin approval list with the v3 outputs.
3. Confirm the CFD05/CFD09 swap time with the PM.
4. Clock-jump correction (paper step 5) after Kevin's reference-clock decision.
