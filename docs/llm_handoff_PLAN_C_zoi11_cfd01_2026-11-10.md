# Plan C for Qwen 3.6 27B Dense: ZOI11 and CFD01 four arm positioning comparison (2026-11-10)

**Author:** external reviewer (code and document review only, no access to data or outputs).
**Executor:** Qwen 3.6 27B Dense on the Windows machine. **Covers:** Drew's Request C.
**Read first:** `docs/llm_handoff_qwen36_35b_moe_RECOVERY_2026-11-07.md` sections 1 and 6 (rules, traps), `.ai_journal/session/2026-11-01_zoi11_debris_barrier_ab.md` (the earlier two arm test), then this file. **If files disagree, this one wins.**
Work **one task at a time**. Every task ends with **STOP**: report and wait.

The reviewer cannot see `output/`. It is not in git. **The only thing the reviewer will ever read is the report file in Task 7.**

This plan does not touch the clock jump experiment. No gate, no policy table, no `apply_clock_jumps`. Another model is running Plan B at the same time in `output\11112026_stage1_closure\`. Do not read or write that folder. Do not open Arm A, B1 or production Arm B at all.

---

## Revision 2026-11-11. Read this before anything else. It overrides the sections below where they differ.

The Plan B run reported that `output\zoi11_test\armA\zoi11_A.db` is no longer on disk. The plan below builds every arm from that file, so the build route changes. Everything about scoring, plotting, checks and wording stays as written.

**R1. Task 0 adds a source search.** Report each of these as found (path, size, last write) or not found:
- `Get-ChildItem C:\Dev\jsat3d -Recurse -Filter zoi11_A.db -ErrorAction SilentlyContinue` (this also covers `C:\Dev\jsat3d\output_archive_*`)
- the folder `output\zoi11_test\raw_staging` and its file count
- `output\jsats3d_2025_tagdrag_ent_anchorZOI03.db`
- the two raw folders `20250610_Receivers_array_testing` and `20250618_Receivers` under the `raw_root` of `config\run_data_earlyfish_2026-11-03_clocksync.toml`, with file counts

**R2. Choose the route by what exists. Do not choose by preference.**

| Found | Route |
|---|---|
| `zoi11_A.db` anywhere | **Route 1.** Follow the plan below unchanged, with that path as the `--init-from` source. Never write to it. |
| no `zoi11_A.db` | **Route 2.** Build arm N from raw, then make Z, C and ZC from arm N. Steps below. |

**R3. Route 2 steps.**

1. Run file `config\run_data_zoi11cfd01_N.toml` is still a copy of `run_data_zoi11_A.toml`. Allowed differences now: comments, `output_db`, `raw_root`, and `tags`.
   - `raw_root`: keep the staging folder if it still exists with 48 files. Otherwise use the `raw_root` value of the earlyfish clocksync run file. The window in `[selection]` limits what is parsed.
   - `tags = ['FFD3', 'FC36']`. The parse happens anyway, so FC36 is added now. This answers owner question 1 of Task 1 in advance. Do **not** add a pulse rate for FC36 to the run file. After the build, report `SELECT Tag_ID, TagType, pulseRate FROM tblTag WHERE Tag_ID IN ('FFD3','FC36')`. If FC36 has no pulse rate, it cannot be positioned: report that and continue with FFD3 only. Never invent a pulse rate.
2. `python.exe scripts\run_data.py config\run_data_zoi11cfd01_N.toml --dry-run`. Paste the whole output. **STOP and wait for the owner's OK.** The dry run shows the receivers, window, tags and steps without running anything.
3. After the OK: the same command without `--dry-run`, console saved with `*>&1 | Tee-Object output\11112026_zoi11_cfd01\armN\run_log.txt`. This parses raw files (read only), builds the database and runs the full 2025 pipeline. If it stops, paste the last 40 lines and STOP.
4. Now run the Task 1 inventory on `arm_N.db` (the plan's 1a to 1f, with `arm_N.db` in place of `zoi11_A.db`; the arm B checks in 1b are skipped). In 1a confirm `tblReceiver_initial` exists and compare its ZOI05 and ZOI06 rows with `tblReceiver`. If they differ, STOP.
5. Arms Z, C, ZC: the Task 3 command with `--init-from output\11112026_zoi11_cfd01\armN\arm_N.db`. Their run files differ from the N run file only in `output_db`, `surface_receivers`, `receivers_3d`. Show `git diff --no-index` of each against the **N** file.
6. Disk: report the size of `arm_N.db` before making copies. Three more copies must leave at least 3 GB free. If not, STOP.

**R4. Task 4 check 1 changes under Route 2.** There is no old database to compare with. Instead report arm N's FFD3 median error on each forebay hold with GPS, next to the 2026-11-01 journal figures (five holds where ZOI11 heard the tag: arm A mean 21.0 m, median 11.6 m; FBY-S-10 47.6 m; FBY-S-7 205.9 m; FBY-S-4 71.8 m). Exact agreement is not expected because the code has changed. A hold that differs by more than 5 m goes under "Things that looked wrong".

**R5. Rule 3 changes.** There may be no `output\zoi11_test\` to protect. The rule becomes: never write outside `output\11112026_zoi11_cfd01\`, `config\` (four run files), the journal and the report.

---

## Revision 2, 2026-11-11: the PM's written request. Overrides everything below and Revision 1 where they differ.

Source: `docs/pm_request_zoi11_cfd01_evaluation_2026-11-11.md`. Read it. It is the authority on what the PM wants.

**The decision this work serves.** Include or exclude ZOI11 and CFD01 from the production array. The PM's only reason to include either: *it increases positioning resolution underneath the debris barrier without degrading positions in other locations.* The assessment is **qualitative**. Plots come first. Tables support them.

**V1. Arm names.** Use the PM's names on every plot, table and file. The letters stay as short keys in file paths only.

| Key | Name on plots | Colour |
|---|---|---|
| N | 1. Excluded | `#9a9a9a` grey |
| Z | 2. ZOI11 Only | `#1f6fb5` blue |
| C | 3. CFD01 Only | `#dd6b20` orange |
| ZC | 4. Both | `#2e8b57` green |

**V2. Events.** Every static hold whose Test ID contains `-S-` (forebay) or starts with `DB` (debris barrier), and every validation drag line in the GPS file. If Task 1d finds other event types in the workbook, list them and ask. Do not drop any event silently. An event outside the built window is listed as "outside window", not skipped.

**V3. Two plots per event and tag, replacing Task 5b.** Both show all four arms, GPS truth where it exists, and the receivers.

1. **2D plan view, PNG** `plots_2d\<TestID>_<tag>.png`. As Task 5b's left panel, at full width. No bar chart panel. Legend with the four arm names and `n=` fixes per arm.
2. **Interactive 3D, HTML** `plots_3d\<TestID>_<tag>.html`, plotly `Scatter3d`.
   - x, y in local metres, z in metres (negative down). Set `scene.aspectmode="data"`.
   - One trace per arm: per transmission medians for holds, the Kalman track for drags (`mode="markers"` for holds, `"lines+markers"` for drags), marker size 3, the arm's colour. Hover: arm name, local time (PDT), x, y, z, number of solutions.
   - GPS truth: black. A hold is one black diamond; a drag is a black line. Place truth at the tag's known depth, `dtf.TAGZ[tag]`. Say in the hover that the depth is the deployed tag depth, not a GPS measurement.
   - Receivers: small black markers with their names as text, from `dm.xyz` (`X_t, Y_t, Z_t`). ZOI11 and CFD01 larger and red.
   - Write with `fig.write_html(path, include_plotlyjs="directory")`. That puts one shared `plotly.min.js` in the folder, so the files work offline and stay small.
   - Title: `<TestID>  <tag>  <date> <start> to <end> local time (PDT)`. Debris barrier holds add `no GPS truth`.
3. An `index.html` in the output folder: one table row per event and tag, linking the PNG and the HTML, with the fix count per arm.

Open every PNG. Open at least five of the HTML files in a browser, among them one debris barrier hold and one drag, and rotate them. Write one line per file opened.

**V4. What to report, replacing Task 6 tables 1 and 6. Tables 2 to 5 stay.**

*Table V4a, per event, tag and arm:* transmissions with a fix; total solutions; spread as std of x, std of y, std of z of the per transmission medians; median z; and, where GPS exists, median XY error.

*Table V4b, new fixes:* for arms Z, C and ZC against arm N, per event and tag: transmissions that have a fix in the arm and none in arm N; transmissions that have a fix in arm N and none in the arm; transmissions in both. Match on `transNo` (valid here, see section 8).

*Table V4c, where the new fixes are:* for the transmissions counted as new in V4b, the distance from each fix to the nearer of ZOI11 and CFD01 (XY, from `dm.xyz`). Report the count in three bands: under 25 m, 25 to 50 m, over 50 m. These bands only describe the result. They are not pipeline parameters and nothing is filtered by them.

*Table V4d, elsewhere:* for events with GPS, the change in median XY error against arm N for Z, C and ZC, next to the distance from the event's GPS truth (its median point) to the nearer of ZOI11 and CFD01. Sort by that distance. This is the direct look at "without degrading positions in other locations".

**V5. Wording for the summary.** Three short paragraphs, one per added arm. Each states only: how many events gained fixes and how many transmissions (V4b); where those fixes sit (V4c); on how many events with GPS the median error rose by more than 0.5 m, fell by more than 0.5 m, or stayed within 0.5 m, and how far from the two receivers the risers are (V4d). Then one line: "Include or exclude is the PM's decision." Do not recommend.

**V6. Constraints repeated from the PM's request.** No existing output folder or database is modified. New work goes in `output\11112026_zoi11_cfd01\`. Only the receiver lists differ between arms. No commit. No push.

---

## 1. The request

Drew, verbatim: "The hydrophones for ZOI11 and CFD01 should both be 1 ft below the bottom of the debris barrier and therefore useful for positioning, but likely plagued with multipath signal conflicts. Include each of the receivers individually, and both and create plots for all the static holds, validation drags in comparison to the original without each included."

Required product: for **every** static hold and **every** validation drag, **one plot** showing four results together:

| Arm | Name on plots | `receivers_3d` |
|---|---|---|
| N | No New Receivers | the nine: ZOI01, ZOI02, ZOI03, ZOI05, ZOI06, ZOI07, ZOI08, ZOI09, ZOI10 |
| Z | ZOI11 Added | the nine + ZOI11 |
| C | CFD01 Added | the nine + CFD01 |
| ZC | ZOI11/CFD01 Added | the nine + ZOI11 + CFD01 |

## 2. What already exists (2026-11-01)

- `config/run_data_zoi11_A.toml` (arm N) and `config/run_data_zoi11_B.toml` (arm Z). Window 2025-06-10 00:00 to 06-13 23:59, tag `FFD3` only, `apply_clock_jumps = false`.
- `output\zoi11_test\armA\zoi11_A.db` and `armB\zoi11_B.db`, each run through the full 2025 pipeline in about 5 minutes.
- `output\zoi11_test\compare_zoi11_ab.py`, `hold_summary.csv`, `per_transmission.csv`, `zoi11_ab_holds.png`.
- Result then: five forebay holds where ZOI11 heard the tag, mean error 21.0 to 10.8 m. Four holds with no ZOI11 pings were bit identical between arms.

## 3. What is missing

1. Arms C and ZC.
2. Validation drags. The earlier test scored static holds only.
3. Tag FC36. The drag lines used FC36 and FFD3 (`scripts/drag_test_figures_2025.py`, `TAGS`). The existing databases were built with `tags = ['FFD3']`, so FC36 is probably absent. Task 1 checks.
4. One plot per event with all four arms.

## 4. Facts from the code you must respect

- `CFD01` is already in `surface_receivers` in both existing run files, so its clock is already synchronised. Adding it to `receivers_3d` is the only change for arm C. `ZOI11` is in neither list in arm N, so arm Z adds it to both (as the existing B file does).
- `_2025_pipeline.py --init-from SRC --db NEW` makes a new database from a finished one: it copies the file, drops every derived table and the progress table, and restores `tblReceiver` from `tblReceiver_initial` (`init_whatif`, line 1489). It refuses to overwrite an existing file. This is the build method for this plan. (RECOVERY forbids `--init-from` on the clock sync experiment databases. That rule is about those databases. It does not apply to the small `zoi11_test` databases, which have no policy table.)
- Deng only combines receivers that have an arrival for that transmission. A hold where the added receiver heard nothing must come out identical to arm N. That is a built in correctness check, use it (Task 4).
- Scoring chain for drag lines is `line_product` in `scripts/drag_test_figures_2025.py`: root B rows, `legacy_receiver_set_consensus`, `smooth_track(jump_limit_m=10.0, jump_window=11, smooth_window=3)`, `kalman_filter_track(process_noise=2.0, measurement_noise=1.0, z_noise=0.5, max_gap_s=20.0)`, scored against the 1 Hz GPS track. Static holds use `hold_product`: per transmission median of root B solutions inside the hold window. **Import and call these functions. Do not rewrite them. Do not change their parameters.**
- Local frame origin for GPS is taken from each database's own `tblReceiver` (`DataManager.__init__`: median of `easting - X`, `northing - Y`).
- Debris barrier holds DB-1 to DB-5 have **no GPS**. They cannot be scored against truth. They still get a plot (spread and depth only).
- `config/run_data_tagdrag_all.toml` notes that receiver CFD09 (serial 19026) sat at the CFD05 location until about 2025-06-11 12:40. CFD09 is in `surface_receivers` only, not `receivers_3d`, so it does not enter positions. Do not add it.

## 5. Rules

1. First command: `Set-Location C:\Dev\jsat3d\jsats3d`. `git rev-parse --show-toplevel` must print `C:/Dev/jsat3d/jsats3d`. Absolute paths for all writes.
2. Raw data under `K:\` and `...\Documents\jsats_data\` is read only.
3. The existing `output\zoi11_test\armA\` and `armB\` folders are read only. Never overwrite them. New work goes in `C:\Dev\jsat3d\jsats3d\output\11112026_zoi11_cfd01\`.
4. The four arms must differ **only** in `output_db`, `surface_receivers` and `receivers_3d`. Show the diff of every new run file against `run_data_zoi11_A.toml` before running anything. Any other differing line is a defect.
5. No parameter changes anywhere. No receiver coordinate or depth edits. If Drew's "1 ft below the bottom of the debris barrier" does not match the depth in `tblReceiver`, report both numbers and continue with the database value. Changing a depth is the owner's decision.
6. New files allowed: run files in `config\` named exactly as in Task 2, one comparison script inside the output folder, the journal, the Task 7 report. Nothing in `scripts\`. Nothing in the repo root.
7. Never invent values. A hold with no solutions in an arm is reported as zero solutions, not skipped.
8. Show every event and every arm, including the ones that got worse.
9. Never write "validated" or "proven". An arm with lower error on these holds is "lower error on these holds".
10. Time basis: study wall clock, PDT stored as if UTC. Never label "UTC".
11. Journal after every task in `C:\Dev\jsat3d\jsats3d\.ai_journal\session\2026-11-11_zoi11_cfd01_four_arm.md`, then `Get-Item` on that file and paste LastWriteTime and its last 3 lines.
12. Disk: free space was about 4.9 GB on 2026-11-09. Check before every copy. Needed: four copies of the source database plus working folders. If free space after the copies would be below 3 GB, STOP and report the sizes. Do not delete anything to make room.
13. Do not commit. Do not push.

---

## 6. Tasks

### Task 0. Setup (read only)

1. Rule 1 checks, `git status --short`, `git log -1 --format="%h %s"`, test suite summary line, free disk.
2. Record size and last write of `output\zoi11_test\armA\zoi11_A.db`, `armB\zoi11_B.db`, and whether `compare_zoi11_ab.py`, `hold_summary.csv`, `raw_staging\` exist.
3. Create the output folder and the journal.

**STOP. Report with journal proof.**

### Task 1. Inventory (read only). The owner decides scope after this.

Open `zoi11_A.db` and `zoi11_B.db` with `mode=ro` and `PRAGMA query_only=ON`.

**1a. Receivers.** From `tblReceiver` print for ZOI11, CFD01 and the nine: `Rec_ID, Tag_ID, X, Y, Z, X_t, Y_t, Z_t` and any of `HydrophoneDepth_ft, HydrophoneOffset_ft, TotalDepth_ft, MountDescription, DeploymentDate` that exist. Confirm `tblReceiver_initial` exists (needed by `--init-from`). If it does not, STOP.

**1b. Clock coverage.** `SELECT Rec_ID, COUNT(*), MIN(seconds), MAX(seconds) FROM tblDetectionClockFixed GROUP BY Rec_ID`. CFD01 must be present in arm A. ZOI11 must be present in arm B and absent in arm A. Report the two rows and the gap between knots: median and maximum of consecutive `seconds` differences for CFD01 and ZOI11.

**1c. Tags.** `SELECT Tag_ID, COUNT(*) FROM tblDetectionRaw WHERE Tag_ID IN ('FFD3','FC36') GROUP BY Tag_ID`. And per receiver for ZOI11 and CFD01: `SELECT Rec_ID, Tag_ID, COUNT(*) ... WHERE Rec_ID IN ('ZOI11','CFD01') AND Tag_ID IN ('FFD3','FC36') GROUP BY 1,2`.

**1d. Events.** List every test event that has a window inside 2025-06-10 00:00 to 06-13 23:59:
- Static holds: every row of sheet `Forebay_StaticHolds` in the testing workbook (`SHEETS_FILE` in `scripts\drag_test_figures_2025.py`), including the ones `static_holds()` skips. Columns: Test ID, Zone, Date, Start, End, Lat, Lon, has GPS (yes/no).
- Drag lines: every `line_name` in the GPS file (`GPS_FILE`), with first and last timestamp and point count. Also list any other sheet in the workbook that holds drags or holds (print all sheet names).
- For each event and each of FFD3 and FC36: number of raw detections inside the window on ZOI11, on CFD01, and on all nine together.

Report this as one table. Mark each event `in window` or `outside window`.

**1e. Multipath on the two receivers.** For FFD3 (and FC36 if present), inside the union of all event windows: per receiver (ZOI11, CFD01, and for reference ZOI07 and ZOI10) the number of raw detections, the number of bursts (gap of at least half the tag pulse rate starts a new burst), and the share of bursts with more than one detection. This is the first measure of Drew's "plagued with multipath" remark.

**1f. Decision table for the owner.** State plainly:
- Is FC36 in the database? If not, drag lines can only be scored for FFD3 unless the database is parsed again with `tags = ['FFD3','FC36']`.
- Which drags and holds fall outside the window.
- Estimated disk need for four copies.

**STOP. Report the tables. Wait for the owner to answer two questions: (1) FFD3 only, or parse again to add FC36? (2) keep the 06-10 to 06-13 window, or widen it?** Until told otherwise, the default is FFD3 only and the existing window, and Tasks 2 to 6 are written for that default.

### Task 2. Run files and design note (no execution)

Create four run files as copies of `config\run_data_zoi11_A.toml`. Change only what is listed.

| File | `output_db` | `surface_receivers` | `receivers_3d` |
|---|---|---|---|
| `config\run_data_zoi11cfd01_N.toml` | `output\11112026_zoi11_cfd01\armN\arm_N.db` | unchanged | unchanged |
| `config\run_data_zoi11cfd01_Z.toml` | `...\armZ\arm_Z.db` | add `'ZOI11'` after `'ZOI10'` | add `'ZOI11'` at the end |
| `config\run_data_zoi11cfd01_C.toml` | `...\armC\arm_C.db` | unchanged | add `'CFD01'` at the end |
| `config\run_data_zoi11cfd01_ZC.toml` | `...\armZC\arm_ZC.db` | add `'ZOI11'` after `'ZOI10'` | add `'ZOI11', 'CFD01'` at the end |

Replace the comment header of each with two lines: the arm name and "Four arm ZOI11/CFD01 test, 2026-11-11; see docs/llm_handoff_PLAN_C_zoi11_cfd01_2026-11-10.md".

Then show, for each of the four files, the output of `git diff --no-index config\run_data_zoi11_A.toml config\<new file>`. The only differing lines may be comments, `output_db`, `surface_receivers`, `receivers_3d`.

Write the exact commands of Task 3 into the journal with the real paths filled in.

Why arm N is rebuilt and not reused: the pipeline code has changed since 2026-11-01. All four arms must come from the same code. The rebuilt arm N is then compared against the old arm A as a regression check (Task 4).

**STOP. Show the four diffs and the command list. Wait for the owner's OK.**

### Task 3. Build the four arms

For each arm X in N, Z, C, ZC, one at a time (do not run them in parallel; each Deng job needs several GB of memory):

```
python.exe scripts\_2025_pipeline.py config\run_data_zoi11cfd01_X.toml ^
  --db output\11112026_zoi11_cfd01\armX\arm_X.db ^
  --init-from output\zoi11_test\armA\zoi11_A.db --no-drag-plots
```

Create each `armX` folder first. Check free disk before each run. Save the full console output of each run to `armX\run_log.txt` (use `*>&1 | Tee-Object`).

After each run record: every step name with its status from `arm_X.pipeline_status.json`; row counts of `tblDetectionClockFixed` per receiver; row count of `tblPositions_Deng` total and with `comment='solution found' AND solution='B'`; size of the database. If any step failed, paste the last 40 lines of the log and STOP. Do not retry with changed settings.

At the end re-check size and last write of `output\zoi11_test\armA\zoi11_A.db`: unchanged.

**STOP. Report the per arm table with journal proof.**

### Task 4. Correctness checks before any scoring (read only)

1. **Regression.** Compare rebuilt arm N with the old `zoi11_A.db`: number of root B `solution found` rows, and for FFD3 the per transmission median X, Y, Z joined on `transNo` (report how many transmissions match, and median, p99 and maximum absolute difference in metres). If the maximum is above 0.01 m, report it as "code changed results since 2026-11-01" with the numbers. Continue, but put it at the top of the report.
2. **Receiver membership.** In each arm, count root B rows whose receiver columns (`r0, r1, r2, r3`) contain ZOI11, and CFD01. Expected: N has neither; Z has ZOI11 only; C has CFD01 only; ZC has both. Any other pattern: STOP.
3. **Identity where the receiver is silent.** For every event where ZOI11 has zero FFD3 detections in the window, arm Z per transmission medians must equal arm N (maximum difference 0.000 m). The same for CFD01 and arm C. Report the event list and the maximum difference. A non zero difference here means the arms differ in something other than the receiver list: STOP.
4. **Clock sanity for the added receivers.** From `tblDetectionClockFixed` in arm ZC: knot count, median and maximum knot gap for ZOI11 and CFD01 inside each event window. An event with no knot within 120 s of its window for the added receiver is flagged `clock unsupported` in every later table.

**STOP. Report the four checks.**

### Task 5. Score and plot

Write one script, `C:\Dev\jsat3d\jsats3d\output\11112026_zoi11_cfd01\compare_four_arms.py`. It opens the four databases read only and writes only into the output folder. It must reuse project code:

```python
import sys
sys.path.insert(0, r"C:\Dev\jsat3d\jsats3d\scripts")
import drag_test_figures_2025 as dtf      # DataManager, line_product, hold_product
```

`dtf.TAGS` is `("FC36", "FFD3")`. `DataManager` handles a missing tag (its products come back empty). Do not edit the module.

**5a. Per event numbers.** For each arm build `dm = dtf.DataManager(db)`. Check that `dm.ox, dm.oy` are equal across the four arms to 0.001 m. If not, STOP.

- Static holds with GPS: `g = dtf.hold_product(dm, hold, tag, dm.p3b)`. Record n transmissions, n solutions, median error, p90 error, RMSE (`sqrt(mean(err**2))`), median dx, median dy, median z, std of x and of y.
- Static holds without GPS (DB-1 to DB-5 and any other skipped hold that has a date and times): build the hold dict yourself with the window times and `x = y = nan`, call the same grouping (copy the three grouping lines of `hold_product`, do not compute an error). Record n transmissions, n solutions, median x, y, z, std of x and y. No error columns.
- Drag lines: `p = dtf.line_product(dm, line, tag)`. Record candidate transmissions (`len(p["ids"])`), scored fixes, median error, p90, RMSE, yield percent.

Also, for arms Z, C and ZC, the share of root B rows inside the event window whose receiver set contains the added receiver or receivers.

Write `event_scores.csv`: one row per event, tag and arm.

**5b. One plot per event.** Fixed colours, same in every plot: N `#9a9a9a` (grey), Z `#1f6fb5` (blue), C `#dd6b20` (orange), ZC `#2e8b57` (green). Layout:

- Left panel, plan view, equal aspect. Receivers as small black triangles with labels; ZOI11 and CFD01 as larger red triangles. Static hold: per transmission medians of each arm as small dots, the arm's median as a large marker with black edge, truth as a black star with 10 m and 20 m circles (no star and no circles when there is no GPS). Drag line: GPS truth as a black line, each arm's Kalman track as a coloured line with dots.
- Right panel: a horizontal bar chart of median error per arm with p90 as a thin whisker and `n=` written on each bar. For events without GPS show std of x and y instead and title the panel "spread (no GPS truth)".
- Title: test ID, tag, date and time range, "local time (PDT)".
- File: `plots\<TestID>_<tag>.png`, dpi 130.

Open and look at every PNG. For each, write one line in the journal: what you see, and anything that looks wrong (overlapping labels, an arm missing, a track far outside the array).

**5c. Summary plot.** One figure: events on the x axis (holds then drags), median error on the y axis, four coloured markers per event. Events without GPS are left out and listed under the figure.

**STOP. Report file list, `event_scores.csv` row count, and the per event one line notes.**

### Task 6. Tables for Drew (from `event_scores.csv`, no new computation)

1. **Main table.** One row per event and tag. Columns: n for arm N, then median error for N, Z, C, ZC, then which arm is lowest. Mark an arm `=N` when its result is identical to arm N (receiver silent).
2. **Win, loss, tie count** against arm N for each of Z, C, ZC. A tie is a median error within 0.5 m of arm N. Count separately for holds and drags.
3. **Yield.** Solutions and scored transmissions per arm, summed over events.
4. **Multipath.** The Task 1e table next to, per event, the share of solutions that used the added receiver. Then the median error of arm Z transmissions that used ZOI11 against those that did not (same events), and the same for arm C and CFD01. This is the direct test of whether the added receiver's own solutions are worse.
5. **Depth.** Median z per arm for the debris barrier holds against the deployed tag depth recorded in the sheet.
6. **Events that got worse.** Every event where Z, C or ZC has a median error more than 0.5 m above arm N. List them all. For each, the share of solutions using the added receiver and whether it is flagged `clock unsupported`.

**STOP. Report the tables.**

### Task 7. Report file and hand off

Write `C:\Dev\jsat3d\jsats3d\docs\reports\PLAN_C_report_2026-11-11.md` (create the folder if missing). Use exactly these sections:

1. **Run facts.** Git head, branch, test summary line, free disk at start and end, source database size and last write at start and end.
2. **Task 1 inventory.** All tables. The owner's two answers.
3. **Task 2.** The four diffs.
4. **Task 3.** Per arm step status and counts.
5. **Task 4.** The four checks with numbers.
6. **Task 6.** All six tables.
7. **Plots.** File names only, and the one line note for each.
8. **Deviations.** Every difference from this plan and why. "none" if none.
9. **Things that looked wrong.** Tracebacks in full.
10. **Open questions for the reviewer.**

Then: final journal entry with `Get-Item` proof and a proposed commit list for the owner (four run files, the report, the journal). Do not commit.

**STOP. Final report with journal proof.**

---

## 7. What you may say and what you may not

- Say: "On the N holds with GPS where ZOI11 heard the tag, median error went from A to B m." Do not say: "ZOI11 improves positioning."
- Say: "CFD01 solutions have a median error of X m against Y m for solutions without it, on these events." Do not say: "CFD01 is unreliable."
- Fourteen or so short events on two days is a small test. Say the number of events and the number of transmissions with every claim.
- A debris barrier hold has no truth. Tighter spread is not lower error. Say "spread", never "accuracy", for those.
- Whether to add either receiver to production run files is the owner's decision.

## 8. Known traps

- `"26E4"` style tags read as floats; use `dtype=str` for any tag column read from CSV.
- `a.loc[i] - b.loc[j]` aligns on labels. Use `.to_numpy()`.
- Match transmissions across arms on `transNo` only **within this plan**. That is valid here because all four arms share identical raw timestamps and no clock jumps are applied. (In the clock jump work the rule is the opposite.)
- `drag_test_figures_2025.py` prints a baseline check for ENT-01 to ENT-05 and exits non zero when run as a script on a different database. Do not run it as a script. Import it.
- `--init-from` raises if the target exists. Never delete a target to rerun; use a new folder name and report it.
- ZOI05 and ZOI06 coordinates are already swapped in current builds. `--init-from` restores `tblReceiver` from `tblReceiver_initial`. In Task 1a compare ZOI05 and ZOI06 rows between `tblReceiver` and `tblReceiver_initial` in the source database. If they differ, STOP: the restore would undo the swap.
- PowerShell breaks multi line `python -c`. Save snippets as `.py` files in the output folder.
