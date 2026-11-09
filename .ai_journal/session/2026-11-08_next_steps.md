# Next Steps — 2026-11-08

## Task 0 — Setup (read-only)

- Tests: 69 OK
- Disk: 32.3 GB free
- Git: not a repo (owner commits)
- Production Arm B: policy=1, candidates=1,425 accepted / 1,173 unresolved, validation=1 failed / 502 passed / 1,797 unverifiable, no position/filter tables
- PM plots exist: output/11082026_pm_plots/ (v1)
- Diagnostic copy exists: output/11072026_diag_armB/

## Task 1a — Design note: stage-1-only arm "B1"

### Functions read and quoted

**`clock_jump_pre.revert_tdoa_supplement(db, source_db)`** (clock_jump_pre.py:967):
"Takes a Path for source_db, attaches it read-only, restores the post-internal-code state by original rowid, leaves stage-1 ledger rows."
- Attaches source_db as `src` (mode=ro)
- Verifies no jump_offset_s in source (must be unshifted baseline)
- For each Rec_ID with stage-2 events: computes cumulative stage-1 offsets, updates tblDetectionRaw seconds and jump_offset_s
- Verifies compact rows map to source by original rowid
- Replaces tblClockJumpEvents with stage-1 rows only
- Drops METRONOME_TABLES + tblClockJumpCandidates
- Returns {"restored_rows": N, "removed_stage2_events": N}

**`_invalidate_clock_outputs(c)`** (_2025_pipeline.py:814):
"Drops everything derived from detection times so the pipeline rebuilds it after a change to tblDetectionRaw."
- Resets progress rows for __ats_phase2__, __ats_fish_two_stage__, __export__
- Drops DERIVED_TABLES + position tables + tblDeepReceiver_step6_solution
- Removes positions/ and positions_2d/ folders
- Removes drag plot PNGs

**`clock_jumps_done(c)`** (_2025_pipeline.py:229):
"Returns True if apply_clock_jumps is False, or if __ats_clock_jumps__ progress row exists."

**`build_context(args)`** (_2025_pipeline.py:1432):
"If clock_segment_safe=true, requires compact.json with status=complete and target matching DB path. Re-creates the policy table on every run when clock_segment_safe=true."
- Raises ValueError if policy table exists but clock_segment_safe=false
- Raises RuntimeError if compact.json missing or target mismatches
- Creates/replaces tblClockExperimentPolicy when clock_segment_safe=true

### Design: exact command list

1. **Create folder:** `New-Item -ItemType Directory output\11082026_whatif_B1`

2. **Copy DB:** `Copy-Item output\11062026_clocksync\jsats3d_2025_earlyfish.db output\11082026_whatif_B1\jsats3d_2025_B1.db`

3. **Copy and edit compact.json:** Copy `output\11062026_clocksync\jsats3d_2025_earlyfish.compact.json`, edit `target` to point to the new DB path.

4. **Copy and edit run file:** Copy `config\run_data_earlyfish_2026-11-03_clocksync.toml` to `output\11082026_whatif_B1\run_B1.toml`. Change only:
   - `output_db = 'output\11082026_whatif_B1\jsats3d_2025_B1.db'`
   - `clock_tdoa_supplement = false`
   - Keep `clock_segment_safe = true` and `apply_clock_jumps = true`

5. **Revert stage 2 on the copy** (Python snippet):
   ```python
   import clock_jump_pre
   clock_jump_pre.revert_tdoa_supplement(
       r'C:\Dev\jsat3d\jsats3d\output\11082026_whatif_B1\jsats3d_2025_B1.db',
       r'C:\Dev\jsat3d\jsats3d\output\10292026_Test\jsats3d_2025_earlyfish.db'
   )
   ```
   This restores post-stage-1 state. The `__ats_clock_jumps__` progress row is NOT deleted (re-running stage 1 on already-shifted DB raises by design).

6. **Drop derived tables and rerun:** Call `_invalidate_clock_outputs(c)` then run:
   ```
   python scripts/_2025_pipeline.py output/11082026_whatif_B1/run_B1.toml --from metronome --to deep_clocks
   ```

7. **Verify before measuring:**
   - `tblClockJumpEvents` has 696 stage-1 rows, 0 stage-2 rows
   - Temperature table: 30,817 rows
   - `tblDetectionClockFixed` has rows for all 17 receivers

### Constraints met
- Works on a copy: `output/11082026_whatif_B1/jsats3d_2025_B1.db`
- Copies production `.compact.json`, edits only `target`
- Run file: only `output_db` and `clock_tdoa_supplement` changed
- `clock_segment_safe = true` and `apply_clock_jumps = true` preserved (same clock-fix algorithm as production)
- Revert uses Arm A as read-only source_db
- `__ats_clock_jumps__` progress row preserved
- Derived tables dropped, then rerun from metronome

### Details my design must also contain

1. **Context build for `_invalidate_clock_outputs`:**
   ```python
   import sys; sys.path.insert(0, 'scripts'); from types import SimpleNamespace
   import _2025_pipeline as p
   c = p.build_context(SimpleNamespace(run_file=r'C:\Dev\jsat3d\jsats3d\output\11082026_whatif_B1\run_B1.toml', db=None))
   ```
   `build_context` uses `args.run_file` and `args.db` (None means use run file's output_db). It re-creates the policy row on the copy (expected).

2. **Pass `pathlib.Path` to revert, check return value:**
   ```python
   import pathlib
   result = clock_jump_pre.revert_tdoa_supplement(
       pathlib.Path(r'C:\Dev\jsat3d\jsats3d\output\11082026_whatif_B1\jsats3d_2025_B1.db'),
       pathlib.Path(r'C:\Dev\jsat3d\jsats3d\output\10292026_Test\jsats3d_2025_earlyfish.db')
   )
   # result['removed_stage2_events'] must be 1,653
   ```
   Run once. Check return value.

3. **Size guard:** Every step on a segment-safe DB calls `check_capacity(db, 4 GiB free, 6 GiB max)`.
   - Production DB: 5,976,518,656 bytes
   - Limit: 6,442,450,944 bytes
   - If a step raises "Compact database exceeded its size budget": **STOP and report; never raise the limit.**

4. **After rerun, prove recomputed:**
   - Knots per receiver differ from production (e.g. ZOI01 production has 46,816)
   - Stage-1 ledger = 696 rows
   - `tblClockJumpValidation` rewritten

5. **Re-check Arm A and production Arm B unchanged:**
   - Arm A: 2026-11-06 18:34:13, 18,631,671,808 bytes
   - Production Arm B: 2026-11-07 22:16:26, 5,976,518,656 bytes
   - Both must be unchanged after 1b.

