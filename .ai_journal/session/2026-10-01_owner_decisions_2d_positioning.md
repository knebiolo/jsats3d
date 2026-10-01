# Session: 2026-10-01, Branch Review, Owner Decisions, 3D/2D Receiver Split

- Author: Ethan Muhlestein / Claude
- Date: 2026-10-01
- Tags: #review #decisions #dbscan #bm-elev #cfd #2d #deng #2019 #2025
- Branch: `ENM_jsat3d_edits`, base `f919f07` (main untouched at `1e5ec35`)

## Active Context
- Reviewer: Claude, working in a separate clone (macOS, pandas 2.3, local only).
- No K: data and no real database (2019 or 2025 v3) was read or written this session.
- Work was a code review of the latest pushes, then owner decisions, then a 3D/2D build.

## 1. Branch Review (commits `7aae684` and `f919f07`)
### What changed upstream
- `7aae684`: v3 run settings and journals (full season `jsats3d_2025_v3.db`, 60,410,185
  detections, 303 files).
- `f919f07`: legacy environment removed; `jsats3d.py` modernized in place for pandas 3 and
  current scikit learn; GPS based float positions (`tblReceiverGPS`, `position.receiver_position_at`);
  parser D file fix; DBSCAN host passthrough; `clock-fix-check` command; setup and one off
  scripts removed.

### Status recorded from LLM_Prompts.txt and journals
- 2019: local DB rebuilt (28,582,645 detections). End to end run not complete. Recent runs
  failed in surface processing (NaN rejected by DBSCAN in `clock_fix`; missing
  `tblMetronomeFiltered` / `tblMetronomeSecondFiltered`). Cause not established.
- 2025: provisional clock fix done for ZOI08, ZOI09, ZOI11, CFD02 only.

### Review findings (not fixed; flagged)
1. `beacon_epoch.adjacent_receiver_enumeration` vectorization changed the rule. The original
   loop gave a child detection to the LAST host epoch whose window contained it; the new code
   gives it to the NEAREST. They differ whenever host epochs are closer than one pulse period
   (ATS catch up pings do this). The matching edit in `multipath_data_object` kept last write
   wins, so the two sites now disagree.
2. `Deng()` discriminant guard changes output rows. Original: a negative discriminant became
   "negative time of arrival" rows in SolutionA only. Now: a new comment in both A and B.
   "solution found" rows are unchanged, but row counts no longer match the canonical DB.
3. `receiver_position_at`: outside GPS coverage it raises ValueError, which the bare `except`
   in `Deng()` records as "singular matrix" (wrong label). It uses raw 1 minute fixes (3 to 6 m
   noise, 2 to 4 ms) where the 9/28 journal recommended a 15 minute median. It filters and sorts
   the full GPS table four times per receiver set, which will be slow at 1.1M rows.
4. None of the core modernization is validated against the canonical 2019 DB yet.
5. Unverified leads for the 2019 failures:
   - Missing tables: the driver drops derived tables at the start of every run; an orphan
     worker from an earlier run could drop tables while another run reads them.
   - NaN in DBSCAN: `clock_fix` builds `interp1d` on `seconds`; duplicate seconds give NaN.
     Check `tblMetronomeSecondFiltered` for duplicate seconds on the failing receiver.
6. `LLM_Prompts.txt` carries a stale 9/29 "run in R01 Deng" section next to the 10/01
   "not complete" status, and a repeated status block.
7. One test assumed a Windows path separator and failed off Windows (fixed below).

## 2. Step Comparison, 2019 vs 2025 (legacy order)
| Step | Same code? | 2019 | 2025 |
|---|---|---|---|
| 1 Import | No (Teknologic importer vs ATS parser, same tables) | done | done (v3) |
| 2 Temperature, sound speed | Yes (method) | done | done |
| 3 Metronome epochs | No (beacon_epoch vs pairwise DBSCAN) | done in earlier runs | provisional |
| 4 Metronome multipath | No (KNN vs DBSCAN) | done in earlier runs | provisional |
| 5 Surface clock fix | Yes (`clock_fix`) | failing recently | 4 receivers only |
| 6 Deep receivers (Deng) | Yes | reached R01 once | not started |
| 7 Repeat 3 to 5, all receivers | Yes | not reached | not started |
| 8 Study tag multipath | No (2025 method not built) | not reached | not built |
| 9 Deng fish positions | Yes | not reached | not started |
| 10 Load and accuracy | Yes | not reached | not started |
Neither data set has fish positions yet.

## 3. Owner Decisions (project lead, 2026-10-01)
1. Beacon DBSCAN parameters approved as they stand, fixed study wide: 0.5 ms timing budget,
   2.5 period window, min_samples 3, anchor side >= 50% of >= 4 receivers, 250 ms steady
   reflection cap. No per group tuning.
2. `BM_Elev` = 861.5 ft (2019 benchmark, same site and datum). Was provisional; now decided.
3. CFD receivers are included, for 2D positioning only. They are clock fixed with the surface
   set. 3D Deng (deep receivers and fish) uses ZOI receivers only: ZOI01 to ZOI10
   (ZOI11 excluded, PM 2026-09-28).
4. 2D method: fixed fish depth; reuse Deng's exact TDoA equations reduced to 2D.
5. 2D receiver set: ZOI01 to ZOI10 plus CFD02 to CFD09.

## 4. Code Changes
- `jsats3d/jsats3d.py`: new `position.deng_2d_roots` (static) and `position.Deng2D(fixed_z)`.
  Same equations as `Deng()` (R^T S = 0.5 b - c^2 t T0, |S| = c T0) with the known
  S_z = fixed_z - r0_z moved to the right side, so a reference plus two receivers give a
  quadratic in T0 with roots A and B. First arrival per receiver per transmission, every set of
  three receivers, receiver X/Y from `receiver_position_at` (GPS floats), plan view hull flag.
  Rows are collected in lists (no per row concat). Writes `<tag>_2D_solutionA/B.csv`.
  `Deng()` itself is unchanged.
- `scripts/legacy_pipeline.py`: `receiver_sets()` adds `receivers_3d`, `receivers_2d`,
  `fixed_z_2d`. Defaults keep 2019 behaviour (3D = all receivers, no 2D). Deep receivers are
  positioned from the non deep 3D receivers. Fish 3D Deng uses `receivers_3d`; `deng_2d()` runs
  for `receivers_2d`; `load_2d_positions()` writes `tblPositions_Deng2D`, kept apart from
  `tblPositions_Deng` so 2019 comparisons stay clean. Rerun reset drops both tables.
- `config/run_data.toml`: decisions recorded; `receivers_3d`, `receivers_2d`, `fixed_z_2d = ''`.
- `tests/test_2025_contracts.py`: exact 2D recovery test; receiver set test; the legacy command
  test no longer assumes a Windows path separator.
- `LLM_Prompts.txt`: rewritten from scratch as a handoff for another AI: absolute read only data
  rules, project summary, environment, step status table, this session's changes, missing
  information, breaking code, and Task A (full read only inventory of every file in both data
  roots, with outputs only on local C:) followed by Tasks B to G. Old content is in git history.

## Parameter Changes With Rationale
- `BM_Elev` 861.5 ft: status changed from provisional to decided (value unchanged).
- `fixed_z_2d` left blank on purpose: the 2D fish depth is an owner value. The driver refuses to
  run 2D until it is set. Units: tblReceiver Z frame (2025: metres, negative below surface).
- No DBSCAN, pulse rate, timing window, sound speed or geometry value changed.

## Validation
- 35 tests pass (local, pandas 2.3). `git diff --check` clean.
- Synthetic array (9 receivers; ZOI09 stands in for a CFD: clock fixed, 2D only), full legacy
  workflow with `--skip-build`:
  - 3D used ZOI09 in 0 rows; 2D used it in 3,315 of 9,436 rows.
  - Per ping median vs truth: 3D solution B XY 0.43 m (Z 8.3 m; weak vertical geometry without
    the extra receiver); 2D solution B XY 1.05 m with fish Z fixed at -6 m (true Z -6 +/- 2 m).
- Not yet run on Windows, pandas 3, or real data.

## Blockers and Known Limitations
- Owner: value of `fixed_z_2d`.
- 2D combinations grow as C(n,3): 18 receivers give 816 sets per transmission. Expect long runs.
- `process()` still runs the legacy KNN metronome; the approved 2025 path stages DBSCAN metronome
  tables instead. Reconcile before a full 2025 run.
- Review findings 1 to 5 above remain open.

## Next Steps
1. Owner sets `fixed_z_2d`.
2. Fix the 2019 surface clock fix failure (check orphan workers and duplicate seconds first),
   then complete 2019 and compare with the canonical DB.
3. Decide on review findings 1 to 3 (enumeration rule, discriminant rows, GPS interpolation).
4. Reconcile the 2025 metronome path in `process()`, then extend the clock fix to all surface
   receivers and request Gate 2.
5. Build the 2025 study tag multipath step (cross receiver epochs, DBSCAN).
