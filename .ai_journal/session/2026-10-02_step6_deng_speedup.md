# Session: 2026-10-02 — Step 6 (Deng) Speed-Up and Step 6 Runs, 2019 and 2025

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-02 about 09:50 to 10:45 local (Pacific; the file `2026-10-01_parallel_rerun_2019_2025.md` carries the earlier work; entries there between 09:55 and 10:40 on 10-02 are the detailed record). UTC is local plus 7 h while daylight saving applies.
- Tags: #step6 #deng #performance #2019 #2025 #parallel #memory #decisions
- Branch: `ENM_jsat3d_edits`, HEAD `0a8d86c`, uncommitted edits.
- New session file: more than 60 minutes (about 10 h) after the last entry in `2026-10-01_parallel_rerun_2019_2025.md` at 23:50 (System Prompt Section 13).

## Active Context
- Both years follow the 12-step paper order, separate parallel processes, stop where told, 12-step table shown after every reply (project lead rule, see the 10-01 journal).
- Databases: `output/cowlitz_2019_synch_1_recreated.db`, `output/jsats3d_2025_v3.db`. Env `jsat_3d`, pandas 3.0.5.
- Steps 1-5 done for both years (2025 step 5 including the host fallback, see the 10-01 journal).

## Short Summary
- Step 6 for 2019 (R01, old code) had run 14+ hours without output. Measured why: `position.Deng` appended every solution to a growing DataFrame (quadratic) and, for 2025, filtered the 1.1M-row GPS table on every receiver lookup.
- Fixed (Stage A, project lead approved): same results, much faster. Validated identical to the committed code.
- Restarted 2025 step 6 (ZOI01-ZOI03) on the new code; started 2019 R02 on the new code; 2019 R01 old code left running as ground truth. Two jobs stopped for memory.

## Decisions & Assumptions
1. **Run 2025 step 6 now (user):** three parallel processes, one per deep receiver, reference receivers ZOI04-ZOI10 (`receivers_3d` minus deep), solution B median to `tblReceiver` X_t/Y_t/Z_t, no `tag_multipath` (step 5 already in the tables). First attempt failed because `legacy_pipeline.receiver_sets()` raises while `receivers_2d` is set and `fixed_z_2d` is blank (owner item open); bypassed in the inline run by computing the reference list directly; no config or code change for that.
2. **Stage A speed-up (user: "lets do Stage A"; question asked first whether it changes the outcome: answer was no for Stage A, last-digit differences possible for Stage B):** implemented, validated, accepted. Stage B (numpy batching) not started; would need a stated tolerance and the full-size 2019 comparison.
3. **Keep 2019 R01 (old code, PID 37208) running** as full-size ground truth; it writes `FF76_solutionA/B.csv` in `output/cowlitz_2019_synch_1_recreated_legacy/deep_receivers` at the end of its Deng. When it prints its median, stop that process (it would continue into R02/R03 on the old code).
4. **Stopped two 2019 jobs for memory (my decision, nothing written by them):** new-code R01 check and R03, after free RAM fell to 1.1 GB of 31.5 GB. To restart when memory allows.
5. **Restarted the three 2025 jobs** (old code, 40 CPU-min each, nothing written) on the new code.

## Parameter Changes With Rationale
- None. No DBSCAN, clock, sound-speed, geometry or Deng parameter changed. The speed-up changes only how rows are collected and how receiver data is looked up.

## Run Log and Results
- 2019 step 6 old code at 09:50: still R01, 804 CPU-min (13.4 h); at 10:40: 853 CPU-min. No progress output; 2019 DB unchanged since 10-01 18:52.
- Measurement of the slowness (read-only, temp output folder): FF76 into Deng = 530,111 rows, 113,968 epochs (110,273 with 4+ receivers; 78,373 with 5, 31,900 with 4), 423,765 4-receiver combinations. Base cost 0.13 s per epoch (200/400/800 epochs: 25.9/55.1/104.3 s) = about 4.1 h. One `pd.concat` onto a growing frame (pandas 3.0.5, str dtype): 1k rows 2 ms, 50k 9 ms, 100k 19 ms, 200k 29 ms; over 424k combinations about 7.5 h more; R01 total about 12 h, three 2019 receivers about 36 h. Cross-check: an earlier successful pass had 164,320 solutions and took just over an hour, which the model reproduces.
- Validation of the new code against the committed code (`git show HEAD:jsats3d/jsats3d.py` loaded from the Temp folder `jsats3d_old`; frames compared with exact equality including dtypes, plus CSV bytes):
  - 2019 FF76, first 400 epochs: A 1,822 rows and B 1,778 rows, identical; 50.7 s -> 19.0 s.
  - 2025 FFC7, 40 epochs with 5 or more receivers: A 238 and B 202 rows, identical; 177.5 s -> 5.8 s (30x).
  - Full unit test suite: 39 tests OK.
- Speed estimates, new code: 2019 FF76 2,000 epochs 97.9 s, 4,000 epochs 229.2 s (about 0.05-0.06 s per epoch, was 0.13 s and growing) -> about 2 h per deep receiver (was about 12 h). 2025 about 0.15 s per epoch with 7 reference receivers -> about 4-5 h per receiver (old code over 100 h).
- Launches on the new code: 2025 ZOI01 (FFC7), ZOI02 (7D2D), ZOI03 (7DBC) at about 10:33-10:36, logs `output/run_2025_step6_ZOI0*.log`; 2019 R02 (FF74) at 10:37, log `output/run_2019_step6_R02.log` (config `config/run_data_2019.toml`, reference R04-R09, solution B). Each writes `deep_receivers_<REC>` under its `_legacy` work folder and updates `tblReceiver` at the end.
- Memory: the three 2025 jobs hold 4.3-4.8 GB each (all 1.1M `tblReceiverGPS` rows are loaded, though only CFD02-CFD09 have GPS tracks). Free RAM 2.9 GB at 10:40 with old R01, new R02 and three 2025 jobs running.
- Not yet produced: any 2025 step 6 result.

### 2019 step 6 results so far (appended 2026-10-02 about 11:58 local)
- **R01 (old code, finished 11:24 local, about 15.2 h wall, about 924 CPU-min):** 164,203 solution-B positions, median X_t -21.050119, Y_t 9.735925, Z_t 248.104404 (written to `tblReceiver`). Previous run: 164,320 positions, median X -21.050, Y 9.736, Z 248.090. Difference: 117 fewer positions, Z +0.014 m; X and Y match to the printed precision. Cause not investigated (this run rebuilt steps 1-5 from scratch under the current code; the earlier run used an earlier build). Old-code CSVs kept as ground truth: `output/cowlitz_2019_synch_1_recreated_legacy/deep_receivers/FF76_solutionA.csv` (51.8 MB) and `FF76_solutionB.csv` (55.0 MB).
- **R02 (new code, 76.1 min wall):** FF74, 472,165 solution-B rows, 456,227 'solution found'; median X_t 5.119336, Y_t -1.321243, Z_t 252.531168; IQR X 0.311, Y 0.404, Z 8.392 (written to `tblReceiver`). Output in `deep_receivers_R02`. No earlier-run R02 value is recorded in the journals, so no comparison yet. The speed-up prediction (about 2 h per receiver) was conservative: R02 took 76 min against about 12 h on the old code.
- Old-code process (PID 37208) was stopped at 11:55 after R01 finished, because it had started R02 on the old code (nothing from R02 old code was written). R02 old-code CSV partial outputs: none.
- Launched at about 11:58: R03 (FF78, new code, updates `tblReceiver`, log `output/run_2019_step6_R03.log`) and a full-size equivalence check of FF76 on the new code writing to `deep_receivers_R01_newcode_check` with NO database update, comparing SHA-256 of `FF76_solutionA/B.csv` with the old-code files (log `output/run_2019_step6_R01_newcode_check.log`).
- Memory at 11:54: 4.6 GB free with the three 2025 jobs (3.6-4.7 GB each) running.
- 2025 step 6 after 80 CPU-min each: still inside Deng, no output yet (expected 4-5 h wall each).

### Status check 12:26-12:38 local: no jobs running, changes by someone else found (appended 2026-10-02 12:38 local)
- User asked for a status check of 2019 and 2025. Findings, not caused by me:
  - No python process is running. 2025 ZOI01/02/03 (started 10:33-10:36) were stopped by a `Stop-Process -Id 30492,33616,40024` command in a terminal I did not run in this session; nothing was written (`tblReceiver` ZOI01/02/03 still the config coordinates 447.07/124.77/-11.15, 467.60/116.36/-12.40, 486.07/106.06/-10.29). About 2 h of compute lost.
  - 2019 R03 and the R01 equivalence check, launched at about 11:55, are also gone; their logs stop at the 'max timestamp' line (11:55), no output folders with files. Cause unknown (no OOM event found; 18.5 GB RAM free at 12:26). 2019 `tblReceiver`: R01 and R02 hold the new medians, R03 still Z NULL.
  - Two git commits that I did not make appeared: `0a4d92c` 10:47 'Speed up Deng solver and add beacon filter' and `a2e1f2f` 11:03 'Refresh LLM handoff and add equivalence test protocol' (author Ethan Muhlestein). Another tool session (a terminal named Cline) ran a command in this workspace.
  - Uncommitted edit in `jsats3d/jsats3d.py` (not by this session): `position.__init__` loads only the `tblReceiverGPS` rows of `resolved_clocks` (`WHERE Rec_ID IN (...)`) instead of the whole table. Terminal history shows old-vs-new comparisons on FFC7 with CFD receivers: identical frames and CSV bytes, GPS rows 1,114,887 -> 596,733, 37.3 s -> 1.2 s on 15 epochs. This is the optional idea listed above; not yet reviewed by me beyond that output.
  - 2019 DB quick_check 'ok'. A 2025 DB quick_check (47 GB) was started and cancelled as too slow; jobs only read that DB, row counts look normal (`tblDetectionFilterSecondary` 6,462,581).
- Nothing restarted yet; waiting for the project lead's decision.

### Restart of all five step 6 jobs and duplicate-data check (appended 2026-10-02 12:55 local)
- User: run 2019 R03, the R01 check and 2025 ZOI01-03 in parallel, and make sure no data is duplicated.
- Duplicate check before launch (read-only, tblDetectionFilterPrimary/Secondary for the six deep beacons):
  - Row counts per tag match the earlier runs exactly, so nothing was appended twice. 2019 Secondary: FF76 955,672; FF74 931,783; FF78 494,347. 2025 Secondary: FFC7 1,928,399; 7D2D 2,324,804; 7DBC 2,209,378. No `tblPositions*` table exists in either database. Step 6 writes only CSV files in its own folder and one UPDATE of `tblReceiver` X_t/Y_t/Z_t per receiver, so a rerun cannot duplicate rows.
  - No NULL `transNo` in Secondary for any tag.
  - A few exact duplicate rows exist: groups of identical (Rec_ID, seconds_fix, transNo): 2019 FF74 8 rows (4 extra copies); 2025 FFC7 56 rows (34 extra), 7D2D 120 (72 extra), 7DBC 28 (14 extra); 2019 FF76 and FF78 none. They are later arrivals or rejected rows (multipath 1 or prediction 1): zero of them are rows Deng uses (multipath 0 and prediction 0), so step 6 is unaffected. Likely cause: two detections with the same `seconds_fix` in `multipath_2`, where the join on duplicate index labels multiplies the rows (legacy behaviour; not confirmed). Left as is, not deleted, pending the project lead's decision.
- Work folders: removed one empty leftover folder (`deep_receivers_R01`, from a stopped job). Old-code R01 CSVs in `deep_receivers/` and R02 results in `deep_receivers_R02/` kept. Each job resets only its own output folder (`fresh_folder`).
- Launched 12:53-12:54 (current working tree, including the uncommitted GPS-row filter in `position.__init__`): 2019 R03 (FF78, updates `tblReceiver`), 2019 R01 equivalence check (FF76, no database update, SHA-256 of solution A/B CSVs against the old-code files), 2025 ZOI01 (FFC7), ZOI02 (7D2D), ZOI03 (7DBC) (each updates its own `tblReceiver` row at the end). Logs `output/run_2019_step6_R03.log`, `output/run_2019_step6_R01_newcode_check.log`, `output/run_2025_step6_ZOI0*.log`. 12.3 GB RAM free at launch.

## Code Changes
- `jsats3d/jsats3d.py`, `position.Deng`: solution rows collected in lists and turned into frames in 50,000-row chunks, one concatenation at the end (was one `pd.concat` per solution); epochs grouped once with `groupby('transNo')`; receiver Z from a dict built from the ephemeris (first row per receiver, as `.values[0]` did). Same row-building code, same try/except structure, same legacy quirks.
- `jsats3d/jsats3d.py`, `position.receiver_position_at(rec_id, timestamp, z_value, tracks=None)` plus new `_receiver_track(rec_id)`: optional per-receiver cache (static X/Y, sorted and de-duplicated GPS arrays, their min and max). Without `tracks` the behaviour is as before; the existing GPS unit test still passes.
- Earlier today (10-01 journal): `multipath_data_object` host fallback; `beacon_pairwise_dbscan.filter_deep_beacon`.
- No new unit test yet for the `tracks` cache or the host fallback.

## Physics / Method Notes
- Deng's exact solution per 4-receiver combination is unchanged. Legacy quirks left as they are (project lead has not decided): (1) if T_0a <= 0 < T_0b the solution-B point uses the never-assigned `S1a`; the NameError is swallowed by the bare `except:` and that B solution is lost, with a 'singular matrix' row written to SolutionA; (2) the B hull test uses S1a for Y and Z; (3) 'negative time of arrival' rows for B are appended to SolutionA. Effect on the number of 'solution found' rows in B not yet measured; medians use only 'solution found' rows.
- Stage B would change floating-point results in the last digits (about 1e-12) and could flip a borderline row at the discriminant, time-of-arrival sign or hull tolerance tests; to be quantified against the old-code R01 output before use.

## Blockers & Known Limitations
- Memory limits how many Deng jobs can run together; 2019 R03 and the new-code R01 check are waiting.
- `fixed_z_2d` still blank (blocks the full `process()` run, not the inline step 6 runs).
- Kevin sign-off on the 2025 reference clock (provisional ZOI08/7DB7, anchor ZOI10) still open; no post-correction clock residual (Gate 2) report for either year.
- 2025 deep receiver coordinates are unsurveyed config values; step 6 results will be sanity-checked against them only.

## Next Steps
1. When 2019 R01 (old code) finishes: record its median, stop that process, compare `FF76_solutionA/B.csv` with a new-code rerun of FF76 (exact check at full size), then run R03 and the R01 check on the new code.
2. When R02 and the 2025 jobs finish: report medians and spreads per deep receiver against the starting coordinates (2025) and the earlier run (2019 R01 median X_t -21.050, Y_t 9.736, Z_t 248.090); stop for review.
3. Optional: load only the needed GPS rows in `position.__init__` to save 3-4 GB per 2025 job (no expected change in results).
4. Do not start step 7 for either year until the project lead says so.

## Files Touched
- Modified: `jsats3d/jsats3d.py`, `LONG_TERM_CONTEXT.md` (standing decisions 2026-10-02), `.ai_journal/session/2026-10-01_parallel_rerun_2019_2025.md`.
- New: this journal.
- Generated (gitignored): `output/run_2025_step6_ZOI0*.log`, `output/run_2019_step6_R0*.log`, `deep_receivers_*` work folders; Temp folder `jsats3d_old\jsats3d_old.py` (copy of committed code for comparisons).

## Follow-up: LLM_Prompts.txt Refresh and Equivalence Test Protocol

- Author: Ethan Muhlestein / Claude
- Date: 2026-10-02, after commit `0a4d92c`.
- Tags: #handoff #testing #equivalence #modernization

### What changed
- `LLM_Prompts.txt` rewritten to the current state: 12 step status table for both years, decisions
  log, a table of every change to Kevin's `jsats3d.py` (7e3b90e) classed as compatibility, speed,
  behaviour or new, missing information, risky code, task order and data locations.
- New Section 8, test protocol for speed and modernization: tiers T0 (unit), T1 (golden master
  fixtures), T2 (old vs new A/B with fixed seed), T3 (Kevin original oracle in a test only env,
  needs user approval), T4 (canonical 2019 parity report), T5 (synthetic truth), T6 (performance);
  required tiers per change class; tests needed now E1 to E9.

### Findings recorded
- The duplicate timestamps (R04/FF75: 104,200 rows, 52,174 distinct seconds) were reproduced by the
  fresh rebuild, so the pipeline creates them; the clock_fix knot fix hides the symptom. Test E1 is
  first priority.
- Task A inventory output was lost when output/ was cleared; redo and keep findings in docs/.

### Validation
- Documentation only. No code, data or database touched. 39 tests pass.

### 2019 R03 finished: result looks unreliable (appended 2026-10-02 13:15 local)
- R03 (FF78, new code) finished in 19.0 min (job started 12:53): 132,905 solution-B rows, only 59,153 'solution found' (46,540 'negative quadratic discriminant', 27,212 'not enough receivers'). The job then wrote median X_t 19.721, Y_t -3.285, Z_t 270.682 (IQR X 8.665, Y 2.598, Z 44.249) to `tblReceiver` (previous config value X 22.25, Y -1.91, Z NULL).
- Quality problem: the found solutions are bimodal. Quantiles of the found solutions: Z 5% 226.4, 25% 227.1, 50% 270.7, 75% 271.3, 95% 273.2; X 5% 19.56, 25% 19.65, 50% 19.72, 75% 28.32, 95% 28.63; Y 25% -5.85, 50% -3.28, 75% -3.25. Two clusters (about Z 227 and Z 271); the median falls at the boundary, so the written X_t/Y_t/Z_t mixes the clusters and should not be trusted. Extreme outliers also exist (Y min -40,989, Z min -165,232, X max 110,295). `in_hull` is False for every found solution. For comparison R01 and R02 had tight spreads (R02 IQR 0.31/0.40/8.39) and heights 248.1 and 252.5; surface receivers are at Z 258.6-260.6.
- Not yet diagnosed: whether the two clusters are the two Deng roots (T_0a vs T_0b; solution B takes the second root, and the legacy bug drops some B solutions), a particular receiver combination, or poor geometry for FF78. Nothing changed; `tblReceiver` R03 holds the unreliable median until the project lead decides (options: restore R03 to the config value, use solution A for R03, filter to the dominant cluster, or investigate by receiver combination).
- Other jobs at 13:13: ZOI01-03 and the R01 check still inside Deng (about 18 CPU-min each), no output yet.

### Why R03 is bimodal: two Deng roots mirrored about a nearly flat receiver set (appended 2026-10-02 13:35 local)
- User asked to investigate R03 by receiver. Read-only analysis of the saved solution CSVs (`deep_receivers/FF76_*` for R01 old code, `deep_receivers_R02`, `deep_receivers_R03`) and `tblReceiver`; no data or code changed.
- **Receivers actually used:** R05 (the master clock receiver) never appears in any solution for R01, R02 or R03 because its detections are absent from `tblDetectionClockFixed` (known legacy behaviour, see step 4). So every 2019 deep receiver is solved from at most R04, R06, R07, R08, R09. For R03 (FF78) the first-arriving receiver is R06 in 58,542 of 59,153 found solutions.
- **Geometry:** the six surface receivers span X -20.4..24.9 and Y -28.9..27.1 m but only 2.02 m in Z (258.55-260.57). The two combinations that produce nearly all R03 solutions have a Z range of only 1.36 m (R04-R06-R08-R09) and 2.02 m (R04-R06-R07-R09). With an almost flat receiver set, Deng's two roots (T_0a / T_0b, solutions A and B) are close to mirror images across the receiver plane and the height is weakly constrained.
- **R03 mirror evidence:** for combinations with both roots found (58,046), median |(Z_A + Z_B)/2 - receiver-plane Z| is 1.44 m (R01 16.7 m, R02 11.8 m, so R01 and R02 are not near-mirror). R03 solution B: 68.6% of found roots lie ABOVE the receiver plane (R01 1.9%, R02 20.6%). Receivers float at the surface and the deep receiver sits on the bottom, so an above-plane root is not physical.
- **R03 solution B by combination** (found solutions): R04-R06-R08-R09 40,402 solutions, all above the plane, median X 19.68, Y -3.26, Z 271.16, Z IQR 0.76; R04-R06-R07-R09 17,162, all below, X 28.42, Y -5.88, Z 226.81, Z IQR 0.42; R04-R06-R07-R08 935 (wild: X 91, Z 126, IQR 251); R04-R07-R08-R09 421; R06-R07-R08-R09 233. The bimodal median is just these two combinations (Z 271 vs 227), so the written R03 median (19.721, -3.285, 270.682) mixes a mirror (unphysical) root with a physical one. 242 found solutions are gross outliers (|X|, |Y| > 200 or |Z| > 400).
- **in_hull is False for every found solution in R01, R02 and R03** (A and B): the hull is built from the nearly flat surface receivers, so any below-plane point is outside it. The `in_hull` flag is therefore not informative for deep receivers; it is not used by the median.
- **Physical-root test (below-plane root per combination, gross outliers removed; same input rows, no new computation of Deng):**
  - R01: median X -20.974, Y 9.711, Z 249.480; IQR 0.172 / 0.045 / 0.433 (written B median: -21.050, 9.736, 248.104).
  - R02: median X 5.135, Y -1.417, Z 247.501; IQR 0.054 / 0.049 / 0.634 (written B median: 5.119, -1.321, 252.531). X and Y agree within about 0.1 m; Z differs by 1.4 m (R01) and 5.0 m (R02), because the B median is pulled up by above-plane roots.
  - R03: combination R04-R06-R08-R09 40,326 solutions, root A, X 21.04, Y -3.95, Z 244.62 (IQR X 0.19, Z 0.83); combination R04-R06-R07-R09 16,993, root B, X 28.42, Y -5.88, Z 226.81 (IQR X 0.18, Z 0.41); two minor combinations 97 and 68 solutions. Overall median X 21.106, Y -3.988, Z 244.382, IQR 7.330 / 1.924 / 17.962.
- **Conclusion:** (1) the legacy rule 'use solution B' does not reliably pick the physical root when the receivers are nearly coplanar: R03 is the clear case, R02's Z is biased high by about 5 m and R01's by about 1.4 m. (2) For R03 even after choosing the physical root the two main combinations disagree by 7.4 m in X, 1.9 m in Y and 17.8 m in Z although each is precise (IQR under 1 m). That points to a systematic timing or position inconsistency involving R07 versus R08 (the combinations differ by exactly that pair; the combinations that contain both R07 and R08 give erratic solutions). Combination 1 (with R08, no R07) lies 2.3 m from the config X/Y of R03 (22.25, -1.91); combination 2 (with R07) is 6.5 m away. Which of R07/R08 is wrong is not established.
- **2025 implication:** the same flat-array issue is likely for ZOI01-03 (reference ZOI04-ZOI10 at Z -2.29..-3.25, deep receivers at Z -10..-12). The running 2025 jobs will write the legacy solution-B median to `tblReceiver`; their CSVs allow the physical-root version to be recomputed afterwards.
- **Not changed, decisions for the project lead:** (a) rule for choosing the root (B only as in Kevin's `kernels`, or the below-plane root per combination); (b) what to write to `tblReceiver` for R03 now (currently the mixed B median 19.721, -3.285, 270.682), and whether to rewrite R01/R02 Z; (c) investigate R07 versus R08 clock-fix residuals for FF78 times; (d) whether to exclude combinations with both R07 and R08. Suggested next read-only checks: per-receiver clock-fix residual for R07 and R08 on the FF78 detection times, and the same by-combination analysis for the 2025 CSVs when they finish.

### R01 full-size equivalence check PASSED (appended 2026-10-02 13:50 local)
- New Deng code (working tree incl. uncommitted GPS-row filter) rerun on FF76 against the 2019 database, no database update: 50.3 min wall (old code about 15.2 h, about 18x faster at full size, with 4 other jobs running in parallel). A 427,910 rows and B 427,010 rows, same as the old-code run.
- `FF76_solutionA.csv` (51,837,474 bytes) and `FF76_solutionB.csv` (54,957,053 bytes): SHA-256 identical to the old-code files. The speed-up changes nothing in the output at full size, for both the solution tables and (by extension) the median derived from them.
- Output folder `deep_receivers_R01_newcode_check` holds the duplicate copy of these two CSVs (about 107 MB); it can be deleted, the old-code originals in `deep_receivers/` are kept.
