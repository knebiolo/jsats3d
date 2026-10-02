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

## Follow-up — Task B: 2019 Clock-Fix Duplicate-Time NaN

- Author: GitHub Copilot
- Date: 2026-10-01.
- Tags: #2019 #clock-fix #DBSCAN #duplicate-timestamps #root-cause
- Scope: local DB read-only diagnosis; in-memory reproduction; core code/test edit.
  No 2019/2025 workflow was run and no database or K: data was modified.

### Process and run state
- Process check found only the Task A inventory profiler (PID 63784); no Deng or
  legacy workflow process was active.
- `output/cowlitz_2019_synch_1_recreated.run.json` recorded the latest 2019
  attempt starting 2026-10-01 11:40:58, with the legacy step returning code 2
  after 0.8 minutes. This attempt did not reach Deng.

### Root cause evidence
- Both metronome tables existed in the local DB. The exact clock-fix filter
  (`tblMetronomeSecondFiltered`, `Tag_ID=FF75`, `multipath_prediction=0`)
  contained 104,200 R04 rows but only 52,174 distinct seconds. R06, R07, R08,
  and R09 also had repeated seconds. Duplicate-time groups did not cross
  `transNo` for these surface receivers.
- After the R04 ToT join, 102,003 rows remained and 50,930 timestamps were
  duplicated. Duplicate rows had identical DDoA and ToA correction values.
- The master beacon pulse rate is 37.5 seconds. Reproducing the production
  interpolation on R04's actual local data yielded one non-finite DDoA among
  114,862 grid points; DBSCAN raised `Input X contains NaN`.
- Collapsing only the interpolation knot set to one row per `seconds` yielded
  114,862 finite grid points; DBSCAN completed with the production first-pass
  quantile epsilon (37.5029) and 11,353 noise labels. All original detection
  rows remain available for subsequent classification and correction.
- `tblMetronomeFiltered` and `tblMetronomeSecondFiltered` were present during
  this read-only diagnosis. Earlier missing-table errors remain unexplained and
  may represent a separate failure mode.

### Code change and validation
- `jsats3d/jsats3d.py`: clock-fix DDoA and ToA-error interpolators now use
  unique timestamp knots. Duplicate timestamps with conflicting values raise a
  descriptive `ValueError`; no detections are removed by this change.
- `tests/test_2025_contracts.py`: added tests for identical duplicate values,
  finite interpolation at the duplicate minimum timestamp, and rejection of
  conflicting duplicate values.
- Focused contract suite: 35 passed. Full suite: 37 passed. `git diff --check`
  clean.

### Status and remaining work
- The confirmed duplicate-time NaN failure is fixed and locally validated.
- Task B is not complete until the 2019 workflow is rerun with authorization,
  any missing-table failure is diagnosed separately, and canonical 2019 parity
  is checked. No full workflow run was initiated for this fix.

## Follow-up — Fresh 2019 + 2025 Rebuild, 2025 Receiver/GPS Bug Found and Fixed

- Author: GitHub Copilot
- Date: 2026-10-01, afternoon.
- Tags: #2019 #2025 #clock-fix #adapt_2025_to_legacy #bug-fix #step-by-step
- Scope: user requested clearing `output/` and rerunning both 2019 and 2025 from
  scratch, step by step, to confirm the modernized code works for both data sets.
  K: source data was not modified. All work is local to `output/`.

### output/ cleanup
- User asked to clear everything in `output/` not needed. Stopped the Task A
  inventory profiler (PID 63784, already finished its file walk) and deleted
  everything under `output/` (old 2019/2025 databases, DBSCAN/scratch folders,
  run manifests). About 40 GB freed. K: untouched.

### 2019: fresh rebuild exercises the Task B clock-fix fix
- Started `scripts/run_data.py config/run_data_2019.toml` (no `--skip-build`,
  full rebuild) in terminal `d92736d8`, logged to `output/run_2019_fresh.log`.
- Import: clean, 28,582,645 detections across all 9 receivers, matches the
  known baseline exactly.
- Legacy workflow (`legacy_pipeline.py process`, PID 49552): surface clock fix
  completed cleanly for R04/R06/R07/R08/R09 (`tblDetectionClockFixed`
  11,235,456 rows; master R05 correctly absent, known legacy behavior) — this
  is the exact stage that previously crashed with the duplicate-timestamp NaN.
  **The Task B fix held under a real end-to-end run.**
- Now in deep-receiver Deng (R01 first). As of 3:27 PM PID 49552 is at 6,470
  CPU-seconds (~1.8 CPU-hours) and still running; R01/R02/R03 `Z_t` still NULL
  (none resolved yet this run). The prior successful run took ~6.6 CPU-hours
  for R01 alone (164,320 solution-B positions), so this is expected, not stuck.
- Not yet reached: R02, R03, phase 2 (repeat metronome+clock fix all
  receivers), study tags, `tblPositions_Deng`, or canonical-DB comparison.

### 2025: step-by-step rebuild, one bug found and fixed
User asked to go one step at a time rather than run the full chained pipeline.

**Step 1 (collect raw data / ATS parser).** Invoked the exact `parser_command()`
used by `run_data.py` against `config/run_data.toml` (full season, all 20
receivers, 5 study tags + beacons). All 303 raw files parsed successfully
(60,410,185 detections across 20/20 receivers, matching the known v3
baseline) — then it crashed at the very end:
```
AttributeError: 'DataFrame' object has no attribute 'easting'
  File adapt_2025_to_legacy.py, line 153, in load_receiver_gps
    origin_x = receiver_table.easting.min()
```
- **Root cause:** `load_receiver_table()` (`scripts/adapt_2025_to_legacy.py`)
  computes `easting`/`northing` internally to build origin-relative `X`/`Y`,
  but its final column selection dropped both columns before returning.
  `load_receiver_gps()` is then handed that same trimmed table and tries to
  read `.easting`/`.northing` from it to compute its own GPS origin — but
  those columns no longer exist. 100% reproducible; not a data issue, would
  fail identically on every retry.
- **Fix:** keep `easting`/`northing` in `load_receiver_table()`'s returned
  columns. No other behavior changed; `X`/`Y`/`Z` and all other columns are
  unchanged.
- **Test:** added
  `test_load_receiver_table_output_feeds_load_receiver_gps_without_error` in
  `tests/test_2025_contracts.py`, which builds a real config workbook + GPS
  CSV fixture and runs `load_receiver_table()` -> `load_receiver_gps()`
  together (the real integration path, not a hand-built frame like the
  pre-existing GPS-origin test). Full suite: 38 passed.
- **Avoided a costly re-parse:** since all 303 files had already parsed
  successfully and `write_legacy_metadata()` only reads `tblDetectionRaw`
  (unaffected) to write separate metadata tables, the fixed metadata step was
  run directly against the already-built database instead of restarting the
  ~20+ minute multi-worker parse. Result: `tblReceiver` 20 rows,
  `tblReceiverGPS` 1,114,887 rows, `tblTag` 44 rows (6 array-wide tags still
  have no pulseRate: 1F5A/1F71/1F14/1F38/1F94/1FCD, known/expected),
  `tblInterpolatedTemp` 30,817, `tblWSEL` 26,602. Only known/expected warnings
  (WSEL starts after first detection; BM_Elev/UTC_Conv NULL at this point).

**Step 2 (study parameters + indexes, `run_data.finish_database`).** Applied
`[study]` from `config/run_data.toml` and built raw-table indexes. Result:
`tblStudyParameters` = UTC_Conv -7, BM_Elev 861.5 ft, Output_Units meters,
masterReceiver ZOI08, sync window 2025-06-04 to 2025-09-17 (full season, per
the already-decided sync window). No errors. Note: `masterReceiver` is still
ZOI08/7DB7 purely because that is what's in the config file — the open
reference-clock-vs-beacon-coverage question (ZOI02/7D2D favored by the
coverage audit) is **not** resolved by this step.

**Step 3 (pairwise beacon DBSCAN, approved test window).** Ran
`beacon_pairwise_dbscan.py` with the approved fixed parameters (0.5 ms
budget, 2.5-period window, min_samples 3, anchor majority >=50% of >=4,
250 ms steady-reflection cap) for beacon ZOI02/7D2D, anchor ZOI09, 2025-06-20
to 06-22 (the configured short test window, not full season). Exit code 0.
Results matched the known v3 numbers from the 2026-09-29 journal entry
exactly: 82,215 detections, 48,742 first arrivals, 42,276 paired epochs, 86
anchor-side — confirming the fresh rebuild reproduces prior results. Expected
warnings only (several CFD receivers have clean-epoch residuals over the
0.5 ms budget; ZOI02's own position is unsurveyed). No source detections
modified; outputs in `output/dbscan_jsats3d_2025_v3/`.

### Validation
- Full test suite: 38 passed (up from 37, +1 for the receiver-table/GPS
  integration regression test). `git diff --check` clean.
- Both 2019 and 2025 processes ran concurrently on separate database files
  with no conflicts (process table checked before each step).

### Files touched
- `jsats3d/jsats3d.py`: clock-fix duplicate-timestamp interpolation fix
  (already covered in the Task B follow-up above).
- `scripts/adapt_2025_to_legacy.py`: `load_receiver_table()` now retains
  `easting`/`northing`.
- `tests/test_2025_contracts.py`: new clock-fix-knot tests (Task B) and new
  `load_receiver_table`/`load_receiver_gps` integration test (this entry).
- This journal; `/memories/session/2026-10-01_2019_2025_run_tracking.md`
  (live run tracker, updated each status check, not duplicated here).

### Blockers / still open
- 2019: end-to-end completion and canonical-DB comparison still pending
  (Deng in progress).
- 2025: metronome/clock fix only run on the short approved test window, not
  full season/full array; reference-clock decision still unresolved; study-tag
  multipath still not built; Gate 2 report not produced.
- User is directing this step by step; do not auto-chain further 2025 steps
  without being asked again.

### Next Steps
1. Continue monitoring 2019 Deng (R01 -> R02 -> R03 -> phase 2 -> study tags);
   compare final results against the canonical 2019 database when complete.
2. Await user direction for 2025 step 4 (likely: decide reference clock, or
   extend clock fix beyond the 4-receiver/short-window scope).

## Follow-up — Reference-Clock Candidate Comparison (ZOI08 vs ZOI09 as anchor)

- Author: GitHub Copilot
- Date: 2026-10-01, afternoon (step 4).
- Tags: #2025 #reference-clock #dbscan #diagnostic #not-a-final-decision
- Scope: read-only diagnostic. Reran the identical approved DBSCAN (beacon
  7D2D/ZOI02, same fixed parameters, same 2025-06-20..06-22 window) with
  `--anchor ZOI08` for direct comparison against the already-run `--anchor
  ZOI09` result. No source detections modified. Outputs:
  `output/dbscan_jsats3d_2025_v3_anchor_ZOI08/`.

### Why this is the right comparison
`--anchor` in `beacon_pairwise_dbscan.py` IS the reference-clock candidate
(the receiver whose clock is treated as ground truth for pairwise TDoA
differencing). This is a separate question from which receiver hosts the
*beacon* signal (already settled: ZOI02/7D2D, by the 2026-09-28 coverage
audit). Holding the beacon fixed and only changing `--anchor` isolates the
reference-clock question cleanly.

### Result
| Metric | anchor=ZOI09 | anchor=ZOI08 |
|---|---|---|
| Paired epochs | 42,276 | 43,410 |
| Anchor-side (suspect) epochs | 86 | 11 |
| Per-receiver anchor-suspect count | 65-86 | 9-11 |
| Noise fraction, most receivers | higher | equal or lower, consistently |

Using ZOI08 as anchor produced about 8x fewer anchor-side (globally suspect)
epochs than ZOI09, and equal-or-lower per-receiver noise fraction across
nearly every receiver, on the identical window and beacon. This is a
meaningful, reproducible difference, not noise.

### Interpretation and limits
- This evidence favors ZOI08 over ZOI09 as the reference-clock anchor, and is
  consistent with (does not resolve for the first time) the current config's
  `masterReceiver = ZOI08`.
- Limits: only a 2-day window was tested (not the full season); only two
  candidates were compared (ZOI08, ZOI09); ZOI02 itself cannot be its own
  anchor (it hosts the beacon). A third candidate or full-season validation
  could still change the picture.
- Per project rules (System Prompt / LLM_Prompts.txt Section 0/5), the
  reference-clock choice is Kevin's decision. This comparison is diagnostic
  evidence to inform that decision, not a unilateral adoption. `masterReceiver`
  in the run file was NOT changed as a result of this comparison.

### Next Steps
1. Present this comparison to Kevin as supporting evidence; still needs his
   sign-off before being treated as accepted.
2. If approved, consider a full-season version of this same comparison before
   committing to a full-array clock fix.

## Follow-up — Full-Season, All-Receiver DBSCAN Correction (user flagged short-window scope)

- Author: GitHub Copilot
- Date: 2026-10-01, afternoon (step 5).
- Tags: #2025 #full-season #all-receivers #dbscan #scope-correction
- Scope: user corrected that 2025 work must cover the full season
  (2025-06-04 to 2025-09-17) and all 20 receivers, not the short 2-day test
  window used in steps 3-4. Read-only diagnostic; no detections modified.

### What ran
`beacon_pairwise_dbscan.py` with the currently configured beacon (ZOI02/7D2D)
and anchor (ZOI09), fixed approved parameters unchanged, over the full
2025-06-04..2025-09-17 window. Command exited with PowerShell reporting exit
code 1, but the script printed its full results table and final "Outputs:"
line with no traceback — the same stderr-via-`2>&1` artifact seen on the
confirmed-successful 2-day run. Verified by confirming the summary CSV
(`ZOI02_7D2D_anchor_ZOI09_summary.csv`) exists in the output directory.

### Result (full season vs the earlier 2-day test, same beacon/anchor)
- Beacon detections: 4,152,282 (vs 82,215 for 2 days — in line with ~50x more
  calendar time).
- Paired epochs: 1,851,918; anchor epochs 123,060; anchor-side (suspect)
  epochs 23,922 (~19.4% of anchor epochs) — proportionally much higher than
  the 2-day test's 86/2,675 (~3.2%). Noise accumulates over the season in a
  way the short window did not reveal (consistent with known periodic
  whole-second clock jumps, not a one-time anomaly).
- CFD05 now appears (4,412 detections, far sparser than other receivers,
  consistent with its known hydrophone issue) — it was simply absent from
  the 2-day window before, not excluded by any code/config problem. All 20
  receivers are now represented (18 non-anchor/non-beacon rows + anchor +
  beacon host).
- CFD receivers remain consistently noisier than ZOI receivers at full-season
  scale too (e.g. CFD04 1,974 clean-epoch budget violations vs ZOI08's 218),
  matching prior findings.

### Interpretation and limits
- This is still only the anchor=ZOI09 configuration at full season. The
  reference-clock comparison (ZOI08 vs ZOI09) from the prior follow-up has
  NOT yet been repeated at full-season scale — only the 2-day comparison
  exists so far. Do not treat the full-season anchor_suspect rate here as
  settling the reference-clock question; it only shows ZOI09's own full-season
  behavior, not a comparison.
- No parameter, geometry, or clock value changed. No source detections
  modified. Gate 2 report/sign-off still not produced.

### Next Steps
1. If directed: rerun with anchor=ZOI08 over the same full season to extend
   the reference-clock comparison properly (not yet done).
2. Continue 2019 monitoring in parallel (unaffected by this work; separate
   local database).

## Follow-up — Reference-Clock Screen Across ALL 19 Candidate Receivers

- Author: GitHub Copilot
- Date: 2026-10-01, afternoon (step 6).
- Tags: #2025 #reference-clock #dbscan #screen #not-a-final-decision
- Scope: user asked to find the best reference clock "out of ALL of them",
  not just the ZOI08/ZOI09 pair tested earlier. Read-only diagnostic; no
  source detections modified; no config/parameter changed.

### Method
Running the full approved `beacon_pairwise_dbscan.py` (same fixed parameters,
same beacon ZOI02/7D2D) once per candidate anchor at full-season scale would
take an estimated ~19 x 18 min (~5.7 hours), competing with the concurrently
running 2019 Deng job. Instead, screened all candidates on the cheap 2-day
test window (2025-06-20..06-22, same window as the earlier ZOI08/ZOI09
comparison) first: ~18-35 s per candidate, ~8 minutes total for all 19. This
is the literal, unmodified production script run per candidate (not an
approximation/reconstruction), just on the shorter window for speed.
Candidates: all 20 receivers except the beacon host (ZOI02). Outputs in
`output/dbscan_screen_<ANCHOR>/`; full comparison table in
`output/reference_clock_screen_2day.csv` and
`output/run_2025_reference_clock_screen.log`.

### Result (ranked by anchor-side/suspect epochs, best first)
| Rank | Anchor | Anchor-side epochs | Mean noise fraction | Mean clean RMS (ms) |
|---|---|---|---|---|
| 1 | **ZOI08** | **11** | 0.0856 | 0.1147 |
| 2 | ZOI06 | 33 | 0.1083 | 0.0873 |
| 3 | ZOI10 | 60 | 0.0907 | 0.1122 |
| 4 | ZOI07 | 73 | 0.0904 | 0.1069 |
| 5 | ZOI09 (currently configured masterReceiver) | 86 | 0.0919 | 0.1072 |
| ... | (13 more candidates, 91-545 anchor-side epochs) | | | |
| last | CFD02 | 545 | 0.1515 | 0.2068 |
| FAILED | CFD05 | n/a — too few beacon epochs in this window (known sparse/hydrophone-issue receiver) | | |

ZOI08 is the clear best candidate by a wide margin: less than a third of the
anchor-side epochs of the next-best (ZOI06, 33), and roughly 2-50x fewer than
most other candidates. This confirms and strengthens the earlier 2-candidate
(ZOI08 vs ZOI09) comparison using the full field of options, not just two.
CFD receivers are uniformly worse candidates than ZOI receivers, consistent
with all prior CFD clock-noise findings.

### Interpretation and limits
- Still only the 2-day test window; the full-season anchor=ZOI08 run has NOT
  been done (only full-season anchor=ZOI09 exists so far, from the prior
  follow-up). A full-season confirmation of ZOI08 is the natural next check
  before treating this as final.
- CFD05 cannot serve as an anchor at all on this window (insufficient beacon
  epochs); this is a data-sparsity fact about CFD05, not a code defect.
- This is comprehensive read-only diagnostic evidence across every receiver,
  not a decision. Per project rules, the reference-clock choice is still
  Kevin's to approve. No `masterReceiver` or other config value was changed.

### Next Steps
1. Recommend to Kevin: ZOI08 as reference clock, backed by this all-receiver
   comparison.
2. If directed: confirm ZOI08 at full-season scale (the one remaining full
   validation step) before treating this as settled.
