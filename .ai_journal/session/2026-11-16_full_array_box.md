# 3D Box + Presence Extended to the Entire Receiver Area (incl. CFD) — Session Journal
# Session: 2026-11-16
# Branch: ENM_jsats3d_clock_updates (only branch touched)

## Owner instruction (2026-11-16)
"For this process, we need our spatial filter to go over the entire receiver area
including the CFD ones."

## Decisions
- The 3D window X/Y is now the min/max of ALL positioned receivers in tblReceiver
  (20: ZOI01-11 + CFD01-09) + position_3d_box_margin_m — not the 9 receivers_3d.
  Z stays the [depth floor, 0] plausibility window. New window (margin 10 m):
  X [427.93, 550.80], Y [26.92, 149.41], Z [-16, 0]. (9-receiver window was
  X [427.93, 518.56], Y [69.52, 145.28].)
- Presence (present_unpositioned rows) expanded the same way — confirmed with the
  owner: emissions heard by >=1 of ALL positioned receivers (not just the 9
  receivers_3d) with no final fix. Rationale: the "in-array" area is now the
  whole array, so a fish in the CFD row near the dam face must show up as
  in-array-but-unpositioned, not as a gap that reads as the fish leaving.
- The [legacy] receivers_3d config is unchanged and still drives the pipeline
  positioning set; only the box and presence in the product chain moved to the
  full array.
- position_3d_box = true, margin 10 m, present_unpositioned = true unchanged in
  both earlyfish run files (comments updated to the new window definition).

## Why the 9-receiver box was wrong (validation)
Array layout (tblReceiver, Arm A DB): ZOI01-11 X [437.93, 508.56] Y [73.98,
139.41]; CFD01-09 X [467.36, 540.80] Y [36.92, 124.63]; all 20: X [437.93,
540.80] Y [36.92, 139.41]. The CFD row extends the array ~110 m east and ~45 m
south beyond the ZOI-only extent. Much of what the 9-receiver box rejected as
"off-array" was fish legitimately in/near the CFD row.

Full-study 4-tag re-run (output/pm_figures_2026-11-16/example_fish/):

  tag   box-rej (9rec m10 -> 20rec m10)   kalman (old -> new)     presence (old -> new)
  26E4  238 -> 116                         1373 -> 1388            1604 -> 1867
  9F8C  147 -> 4                            45 -> 65               1612 -> 1995
  1D5A  790 -> 310                         3333 -> 3398           13601 -> 14451
  B5ED  1461 -> 210                        2297 -> 2323            7654 -> 8958

New Kalman counts are essentially the pre-box contract (2026-10-25: 1387 / 65 /
3413 / 2327): the whole-array box keeps everything the legacy chain had except
the genuinely off-array solutions — the intended effect. Presence rows grew
because the CFD row is now counted as in-array.

26E4 jump interval re-validation (position_box_compare_2025.py, whole-array box,
margin 10 m; output/11162026_box_compare/26E4_box_compare.png shown in Review):
- 116 solutions rejected (was 1080 at margin 0 on the 9-receiver box; 238 at
  margin 10).
- Consensus 1398 -> 1396: only 3 fixes dropped, 1 new; matched-fix drift median
  0.0 / p99 0.46 m / max 17.3 m (the 11-14 run moved p99 8.67 m because the
  narrow box stripped real solutions; the wide box barely moves medians).
- 45 with-box fixes sit just outside the STRICT (margin 0) array edge: 3 in the
  CFD row at 23:19-23:20 (2.4-4.8 m east of X=540.8, the fish's first fixes at
  ~(546,67)) and 42 at the north edge 01:06-01:24 (0.1-9 m north of Y=139.4).
  These agree with the WITHOUT-box consensus to within ~0.5 m (matched drift),
  i.e. the consensus robustly places the fish there — a fish a few metres past
  the outermost receiver, exactly what the margin is for. No km-scale or
  100 m-scale excursions remain; the X~570 m and ~150 m-class jump artifacts are
  still rejected (window X max 550.8).

## Files touched
- scripts/fish_figures_2025.py: box + presence now use recs_all = rec[rec
  [X_t,Y_t] notna]; removed the receivers_3d_cfg read/resolution (no longer used
  here); docstring + comments updated.
- scripts/fish_dashboard_v3.py: same box change in main(); EXPECTED_KALMAN
  re-verified to the 2026-11-16 counts (26E4 1388, 9F8C 65, 1D5A 3398, B5ED
  2323) with both previous contracts kept in the comment; docstring/README
  wording updated.
- scripts/position_box_compare_2025.py: box from all positioned receivers
  (removed the run-file receivers_3d resolution + local tomllib import); plan
  view labels all 20 receivers.
- config/run_data_earlyfish_2025.toml + config/run_data_earlyfish_2026-10-28.toml:
  [figures] comments updated (window = all positioned receivers; presence = all
  positioned receivers). Values unchanged (box true, margin 10.0, presence true).
- LONG_TERM_CONTEXT.md: 2026-11-16 row.

## Artifacts
- output/11162026_box_compare/ — 26E4 with/without plot + summary + row CSVs
  (whole-array box, margin 10 m).
- output/pm_figures_2026-11-16/example_fish/ — final 4-tag pass (figures,
  fish_qc.csv, <TAG>_positions.csv incl. presence).
- output/pm_figures_2026-11-16/fish_dashboard_v3/ — dashboard rebuilt on the new
  chain; all four tags pass the self-check.
- 2026-11-15 artifacts (9-receiver-box runs) preserved in
  output/11152026_4tag_margin/ and output/pm_figures_2026-11-15/.

## Tests
python -m unittest discover -s tests -> Ran 21 tests, OK.

## Gotchas hit
- None new. (The compare tool's local `import tomllib` became unused once the
  box stopped reading receivers_3d — removed with the block.)

## Next steps
1. Owner commits on ENM_jsats3d_clock_updates (nothing committed by this
   session, per project rule).
2. PM review: 26E4_box_compare.png (whole-array box), 4-tag fish_qc.csv
   (pm_figures_2026-11-16), dashboard.html.
3. 2027 items unchanged: TDOA jump detection from the frozen clocksync branch as
   the pre-deployment check; present_unpositioned visualization (Kevin: "save
   that for later").

## Addendum: 9 clock-jump tags — new solutions + with/without filter (owner 2026-11-16)

### Request
New folder showing how the pre-median spatial filter helps positioning for fish
during known clock jumps on commonly used receivers (PM sheet):
- ZOI02 2025-07-06 21:12-21:22: B175, 8FF4, 56B7
- ZOI02 2025-07-25 07:35-07:42: 5EDD, 694C, 80FD
- ZOI07 2025-07-01 19:55-21:05: CB75, A360, 7A68

### Discovery: the 9 tags were never staged
None of the 9 is in the Arm A DB (tblTag/tblDetectionRaw/tblDetectionClockFixed
all 43 tags = 4 study fish + 39 config beacons; the 2026-10-29 parse used
`--tag` filters). Raw scan of all 484 files under the OneDrive raw_data root:
all 9 present (study window 06-15..08-06 parse yields 2,362,504 rows:
B175 905,391 / 8FF4 110,966 / 56B7 234,869 / 5EDD 69,448 / 694C 9,693 /
80FD 86,956 / CB75 179 / A360 705,894 / 7A68 239,108). Control: the 4 study
tags were found by the same scan (scan valid). Raw tag code format:
`G72` + 6 hex chars; project tag ID = first 4 chars of the suffix.

### Key mechanism finding (why no clock-chain rerun is needed)
F1 (2026-10-21): for study tags, `multipath_data_object.__init__` reads
tblDetectionRaw and applies the per-receiver clock correction ON THE FLY from
the knot curves stored in tblDetectionClockFixed (beacon-derived, already in
the Arm A DB for all 20 receivers, full window). So new tags only need:
raw staging -> tblTag row (measured pulse rate) -> step 8 multipath -> step 11
Deng (receivers_3d) -> APPEND export to tblPositions_Deng (widen_table does not
drop; positions_data_management appends and does NOT delete its input files).

### New tool: scripts/position_additional_tags_2025.py
Append-only positioning of additional tags on a finished earlyfish DB. Reuses
_2025_pipeline step functions verbatim (fish_multipath_run/positions_run/
export_run) with c.tags overridden and its OWN work folder
(<db stem>_addtags), so the original work folder + 4-tag CSVs and every
existing tag's rows are untouched. Idempotency guard (refuses tags already in
tblTag), source-table tag guard, per-chunk commit + flushed progress.

### INCIDENT (2026-11-16, resolved): duplicate-append loop
First run: the append loop re-created the `pd.read_sql(chunksize=...)` iterator
INSIDE the `while True` body, so each iteration re-read and re-inserted the
first 200k rows forever. The commit sat after the loop, so it was ONE giant
uncommitted transaction; the DB file grew 18.63 -> 34.81 GB in ~40 min
(uncommitted page allocations) while the valid database was unchanged. Killed
the process; the next connection's hot-journal recovery rolled the transaction
back and truncated the file to 18,631,671,808 bytes (= page_count*page_size).
`PRAGMA integrity_check` = OK; per-tag counts match the original build
(1D5A 189,380 / B5ED 95,813 / 26E4 37,981 / 9F8C 14,940; 43 tags; zero
duplicate (SourceFile,SourceRow,Rec_ID) groups). Arm A DB = pristine original.
Fixes in the driver: chunk iterator created once; commit per chunk; source-tag
guard; `python -u` for the run log.

### Pulse rates (median inter-pulse gap in [2.5,4.5] s, pooled per receiver)
B175 3.1360 / 8FF4 3.0509 / 56B7 3.0122 / 5EDD 3.0980 / 694C 2.9689 /
80FD 3.0608 / A360 2.9847 / 7A68 3.0419 s (all consistent with the ~3 s study
tags). CB75: 179 detections, min within-receiver gap 28 s, only 1 detection on
ZOI07 (06-27, not in the 07-01 window) -> no measurable pulse rate -> staged
but EXCLUDED from the positioning chain (no transNo -> Deng unusable); its
comparison row will read "no solutions".

### Compare-tool fix (needed for the narrow jump windows)
position_box_compare_2025.py: `pd.Timestamp(str).timestamp()` is machine-local
(Pacific machine = stored basis + 7 h in July) -> now
`pd.Timestamp(str).tz_localize("UTC").timestamp()` (stored basis = PDT wall
clock as-if-UTC).

### Positioning results (2026-11-16, 23.0 min end-to-end)
Driver re-run succeeded; "existing tags' rows unchanged" verified. New tags in
tblPositions_Deng (root B "solution found" / total rows):

  tag    B found / B rows     A found / A rows
  B175   165,445 / 507,651    35,037 / 507,651
  8FF4    31,160 / 102,132     5,982 / 102,132
  56B7    49,768 / 163,886     5,106 / 163,886
  5EDD     6,892 /  24,762       964 /  24,762
  694C     1,729 /   5,070       157 /   5,070
  80FD     1,587 /  26,079       219 /  26,079
  CB75         0 /         0         0 /        0  (skipped: no pulse rate)
  A360   239,383 / 718,375    39,921 / 718,375
  7A68    84,454 / 201,255    14,475 / 201,255

Status entry "additional_tags" appended to
output/10292026_Test/jsats3d_2025_earlyfish.pipeline_status.json; work folder
output/10292026_Test/jsats3d_2025_earlyfish_addtags/ (original 4-tag work
folder untouched).

### Status / next
- scripts/jump_window_comparisons_2025.py (jump window +/- 1 h context,
  production box, margin 10 m) -> output/11162026_jump_tags/ with
  jump_windows_summary.csv (now includes explicit zero rows + note for tags
  without solutions); representative PNGs shown in the Review pane.
- Run logs: output/11162026_addtags_run.txt, output/11162026_jump_tags_run.txt.

### Jump-window presence map (stored basis = PDT wall clock; settled the time question)
t0 = ToA - T01 is on the stored basis (same chain as the 4 study tags; 26E4
validation 2026-11-16). Raw detections (tblDetectionRaw, stored basis) vs the
three PM-sheet windows:

  tag    window (stored)         last before        first after         verdict
  7A68   ZOI07 7/1 19:55-21:05   continuous 15:00->21:20 (10,459 rows in window)   PRESENT
  80FD   ZOI02 7/25 07:35-07:42  continuous 06:16->10:59 (644 rows in window)      PRESENT
  B175   ZOI02 7/6  21:12-21:22  7/6 14:48 (6.4 h)      7/7 00:10 (2.8 h)  gap
  56B7   ZOI02 7/6  21:12-21:22  7/6 20:35 (0.6 h)      7/7 06:01 (8.7 h)  gap
  8FF4   ZOI02 7/6  21:12-21:22  7/6 17:36 (3.6 h)      7/7 11:05 (13.7 h) gap
  5EDD   ZOI02 7/25 07:35-07:42  7/25 05:52 (1.7 h)     7/25 11:59 (4.3 h) gap
  694C   ZOI02 7/25 07:35-07:42  7/25 07:29:41 (6 min!) 7/25 09:39 (2.0 h) gap
  A360   ZOI07 7/1  19:55-21:05  7/1 12:31 (7.4 h)      7/2 00:12 (3.1 h)  gap
  CB75   ZOI07 7/1  19:55-21:05  7/1 18:28 (1.4 h)      7/1 22:57 (1.9 h)  gap

So only 7A68 and 80FD had root-B solutions in their jump context; the other
seven have raw-detection gaps straddling the windows (nothing to position,
hence nothing for the filter to protect). A360 has 239K found-solutions over
6/25-7/2 but ZERO root-B rows on 7/1 itself.

### With/without results (output/11162026_jump_tags/, whole-array box, margin 10 m)
  tag    solutions   box-rej   consensus off->on   drift   gross off->on
  7A68   3094        0         277 -> 277          0.0 m   0 -> 0
  80FD    177        0         18 -> 18            0.0 m   0 -> 0
  (7 tags: no solutions in window)
Interpretation: both present fish sat inside the array with dense 9-receiver
coverage; the per-transmission median was already robust to the jump-corrupted
solutions (none were outside the array volume), so the pre-median box changed
nothing for them — consistent with the 26E4 result where the box matters when
corrupted solutions land OUTSIDE the array volume (116 rejected, 3 fixes
dropped). The box's protection shows in the 26E4-style edge cases, not in
dense central coverage.

### Plots for all nine tags (owner question 2026-11-17: "where are the plots of all the tags?")
The compare tool only writes a PNG when the tag has solutions in the window, so
the seven gap tags had none. Added:
- B175_box_compare.png — DAY-WIDE comparison (7/6 00:00 -> 7/7 01:00): 27,673
  solutions, 1,729 fixes, 0 box-rejected, 42 gross fixes both arms (all within
  the 10 m margin, i.e. near-edge — same class as 26E4's 45). The plot shows
  the fish's 7/6 track ending ~14:48, well before the 21:12 jump.
- scripts/jump_window_presence_2025.py -> 3 presence figures, one per jump
  event, 3 panels each (all nine tags covered): hourly raw detections (gray
  bars) + root-B found solutions (blue) on the jump day, jump window shaded
  red. presence_ZOI02_2025-07-06.png (B175/8FF4/56B7),
  presence_ZOI02_2025-07-25.png (5EDD/694C/80FD),
  presence_ZOI07_2025-07-01.png (CB75/A360/7A68). The two present fish show
  dense counts through the shaded band; the other seven show their gaps.
- jump_windows_summary.csv: B175 row now carries the day-wide metrics + note.

### Open question for the owner/PM
Seven of the nine sheet rows have no fixes during the window (detection gaps).
Possible explanations: (a) the sheet lists fish that were in the array THAT
DAY (not necessarily during the exact jump interval); (b) the sheet times are
on a different basis than the stored basis (tested true-UTC: only B175 would
then fit the 7/6 window, none of 8FF4/56B7; the 7/25 row fits stored-basis
80FD and no other interpretation fits 5EDD/694C better). Awaiting owner input
before drawing conclusions about the seven.

### Script purge (owner request 2026-11-17)
Owner asked which scripts are exploratory/diagnostic vs needed for the 2025
pipeline while keeping the 2019 code runnable. Verified the full import graph
(run_data -> _2025_pipeline/legacy_pipeline/parse_ats; _2025_pipeline ->
beacon_pairwise_dbscan + clock_jump_pre + ent_analysis subprocess; figures ->
ent_analysis + drag_test_figures helpers; tests). Result:
- DELETED: scripts/adapt_2025_to_legacy.py — early-Sept CSV->legacy adapter,
  superseded by parse_ats_raw_to_legacy.py (the raw-ATS path the pipeline
  uses). The 2026-10-27 journal's "imported by parse_ats" note was stale;
  parse_ats no longer imports it. Only code reference was the one-line
  test_import_adapter in tests/test_smoke.py (removed; docstring updated).
- KEPT per owner: metronome.py, mulitpath.py (Kevin's 2019 drivers, hardcoded
  knebiolo paths; superseded by legacy_pipeline.py functions but preserved
  from the original 2019 tree).
- KEPT: all pipeline scripts, the five 11-16 deliverable tools
  (position_additional_tags, position_box_compare, jump_window_comparisons,
  jump_window_presence, fish_dashboard_v3), jsats3d package, tests.
- NOTE: the 2026-11-03 audit table's "metronome.py 2025 Needed? Yes" was
  wrong (nothing imports it; the 2025 metronome step is
  beacon_pairwise_dbscan.py). Kevin's notebooks stay per the 09-24 legacy-core
  rule. Tests: 20 pass after the change.

### Second purge (owner follow-up 2026-11-17)
Owner: keep the pipeline code + median filter, drop the diagnostic/output
scripts — "we already have figures scripts for all of that." Verified the
three A/B tools are read-only (never touch tblPositions_Deng; referenced only
by each other) and deleted:
- scripts/position_box_compare_2025.py
- scripts/jump_window_comparisons_2025.py
- scripts/jump_window_presence_2025.py
Generated deliverables remain on disk in output/11162026_jump_tags/ (local
only, gitignored). position_additional_tags_2025.py STAYS — it is pipeline
code (appended the 9 tags' positions). If the PM corrects the seven absent
sheet rows, the A/B tool will be rebuilt from this journal (tz_localize("UTC")
window parsing, margin-box rejections vs strict-box gross counts, zero-row
notes) — ~1 h rewrite, reference outputs exist. Tests: 20 pass.
