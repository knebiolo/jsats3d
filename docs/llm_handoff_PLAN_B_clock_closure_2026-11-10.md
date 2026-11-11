# Plan B for Qwen 3.6 35B MoE: explain the Stage 1 drift, find the missed jumps, validate (2026-11-10)

**Author:** external reviewer (code and document review only, no access to data or outputs).
**Executor:** Qwen 3.6 35B MoE on the Windows machine. **Covers:** Drew's Request B.
**Read first, in this order:** `docs/llm_handoff_qwen36_35b_moe_RECOVERY_2026-11-07.md` (rules and traps), `docs/llm_handoff_qwen36_35b_moe_STAGE1_TDOA_VIZ_2026-11-09.md` (section 0 facts, Appendix B), then this file. **If files disagree, this one wins.**
Work **one task at a time**. Every task ends with **STOP**: report and wait.

The reviewer cannot see `output/`. It is not in git. **The only thing the reviewer will ever read is the report file in Task 8.** Anything you do not paste into that file did not happen, as far as the review is concerned.

---

## 1. Why this plan exists

The 2026-11-10 analysis document ends with an open question: Stage 1 applies permanent shifts that add up to between -11.5 s and +15.3 s, yet the raw 7DB7 series sits at its path delay and its excursions come back. It asks whether the firmware events are "permanent clock error" or "adjustments GPS absorbs".

The reviewer's reading of the code says this question can be answered with numbers you already have on disk. It is a hypothesis until Task 1 confirms it. **Do not repeat it to Drew as fact before then.**

### 1.1 The arithmetic

For one listener, in original receiver time:

- `raw(t)` = listener first arrival minus ZOI08 first arrival. The beacon and the receivers do not move, so the true value is one constant, the path delay `L0` (11 to 63 ms).
- Every real clock step adds to `raw`. If `raw` returns to `L0`, the real steps since the start add up to zero.
- `B1(t) = raw(t) - cum_offset(t)`. This identity was verified to 0.0000 ms on 2026-11-09.

So wherever `raw` is back at `L0`:

    B1(t) - L0 = -(sum of applied offsets) = (sum of real steps that were NOT applied) - (size errors of applied offsets) - (applied offsets where raw never stepped)

The blue series can only drift for three reasons:

| Term | Meaning | What it says about the method |
|---|---|---|
| **Missed steps** | `raw` stepped, Stage 1 has no line there | Stage 1 is right but incomplete. This is exactly Drew's statement "not all jumps are identified". |
| **Size error** | A line sits on a real step, but its `offset_s` differs from the step | The plus or minus 300 s median measured the wrong level (likely an unstable plateau). |
| **Phantom line** | A line sits where `raw` did not step | The internal code fired without a sustained clock step. |

Supporting facts already in the journals: 672 of 696 Stage 1 offsets agree with the local raw step within 50 ms (2026-11-07 audit). The 2026-11-04 result "curves continuous at 463 applied jumps" says each applied step is locally right. Neither says anything about steps that were never applied. So the two results do not conflict with the drift. The reviewer expects the **missed steps** term to dominate, with the return half of an excursion being the usual missing piece (example in the ZOI05 ledger rows: the level is back near 32 ms before the second event with no event in between).

### 1.2 Corrections to the 2026-11-10 analysis document

Do not carry these statements forward:

1. "DBSCAN implementation not started" is wrong. `clock_jump_pre.tdoa_supplement()` is Stage 2. It ran on 2026-11-03 and produced 1,653 supplementary jumps. Its measured weakness is known: 79.7% agreement with raw TDoA against 96.3% for Stage 1, and 55.6% uncertain boundaries. What is truly not started is the **tdiff** variant.
2. "Fish track validation not started" is wrong. Three example fish were plotted on 2026-11-04 and a second diagnostic ran on 2026-11-08 (B3). What is missing is a version for the current arms with a fixed selection rule.
3. The "key finding" paragraph offers two physical stories. Task 1 replaces it with the three measured terms above.

### 1.3 One idea to test, not to apply

Stage 1 and Stage 2 both add up measured steps. One wrong step then stays in every later timestamp. Because `L0` is a physical constant, a regime's correction can instead be read directly as `regime level - L0`. An error in one regime then cannot leak into the next. The script reports how well this would work as one column (`level_anchored_within_50ms_pct`). It is computed in memory on a copy. **It is a what-if number for the owner. Do not implement it, and do not describe it as a fix.**

---

## 2. Rules

All RECOVERY rules and all rules of the 2026-11-09 handoff apply. The ones that matter most here:

1. First command: `Set-Location C:\Dev\jsat3d\jsats3d`. `git rev-parse --show-toplevel` must print `C:/Dev/jsat3d/jsats3d`. `git branch --show-current` must print `ENM_jsats3d_edits_clocksync`. Absolute paths for every write.
2. Every database opens with `sqlite3.connect("file:%s?mode=ro" % path.replace("\\","/"), uri=True)` and then `PRAGMA query_only=ON`. No database copies in this plan. No database writes at all.
3. All new files go in `C:\Dev\jsat3d\jsats3d\output\11112026_stage1_closure\`. Two exceptions, both text: the journal and the Task 8 report.
4. **Do not edit `stage1_closure_audit.py`.** Copy it and run it. If it fails, paste the full traceback and STOP. Do not fix it, do not write a replacement.
5. No parameter changes. The script's constants are project values (0.5 ms, 50 ms, 250 ms, 300 s). If a result looks odd, report it. Do not adjust a constant to make it look better.
6. Time basis is study wall clock, PDT stored as if UTC. Never write "UTC" on a label.
7. Report only numbers you produced this session. Paste tables whole. Do not summarise a table into a sentence and drop the table.
8. Never write "validated", "proven", "works", or "fixed". Write what the numbers show.
9. 1 ms of timing error is about 1 m of position error. 0.5 ms is the warning budget. 50 ms is the application floor. Do not mix them.
10. Journal after every task in `C:\Dev\jsat3d\jsats3d\.ai_journal\session\2026-11-11_stage1_closure.md`, then `Get-Item` on that file and paste LastWriteTime plus its last 3 lines.
11. Do not commit. Do not push. The owner does both.
12. If free disk is below 3 GB at any point, STOP.

---

## 3. Tasks

### Task 0. Setup (read only)

1. Rule 1 checks. `git status --short`. `git log -1 --format="%h %s"`.
2. Test suite: `python -m pytest tests -q` with the env python (`C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe`). Record the exact summary line.
3. Free disk on C:.
4. Confirm inputs exist and record size and last write of each:
   - `output\11092026_stage1_tdoa_review\series_armA_uncorrected.csv` (expect 1,003,984 data rows)
   - `output\11092026_stage1_tdoa_review\series_b1_corrected.csv` (expect 1,003,970)
   - `output\11092026_stage1_tdoa_review\stage1_correction_lines.csv` (expect 696)
5. Record size and last write of Arm A, B1 and production Arm B (expected values are in the 2026-11-09 handoff section 0).
6. Create the output folder and the journal file. Copy `docs\handoff_tools\stage1_closure_audit.py` into the output folder.
7. Run `python.exe output\11112026_stage1_closure\stage1_closure_audit.py --selftest`. It must print eleven `OK` lines and `SELFTEST PASSED`. If any line says `FAIL`, paste all of the output and STOP.

**STOP. Report with journal proof.**

### Task 1. Run the closure audit (CSV only, no database)

```
python.exe output\11112026_stage1_closure\stage1_closure_audit.py ^
  --series output\11092026_stage1_tdoa_review\series_armA_uncorrected.csv ^
  --lines  output\11092026_stage1_tdoa_review\stage1_correction_lines.csv ^
  --out    output\11112026_stage1_closure\audit
```

It writes `closure_report.md`, `steps_detected.csv`, `line_matches.csv`, `tdiff_classes.csv`.

Checks before you report. Record each as `value -> OK / MISMATCH`:

| Check | Expected | If not |
|---|---|---|
| Totals `lines` | 696 | STOP |
| Receivers in Table A | 18 | STOP |
| Totals `confirmed` | near 672 (the 2026-11-07 audit). Anything from 620 to 696 is plausible | Below 600: STOP. The time basis of the two CSVs may differ. Do not continue |
| ZOI04 and ZOI11 `lines` | 0 | STOP |
| `final_cum_offset_s` for ZOI01 and ZOI03 | about +15.3 and -11.5 | Report the values and continue |
| `baseline_ms` per receiver | between 11 and 63 | Report any outside and continue |
| Table G | a table, or one line starting "Table G not computed" | Either is acceptable. Paste what is there |

What to read in each table:

- **Table A.** `raw_within_50ms_pct` is the share of epochs where the uncorrected clock is already right. `stage1_within_50ms_pct` is the same after Stage 1. If Stage 1 is lower than raw for a receiver, Stage 1 moved more epochs away from the path delay than it brought back. Say so plainly for each such receiver.
- **Table B.** `step_without_line` is the count of jumps Stage 1 never saw. `line_without_step` is the count of lines where the raw series did not step.
- **Table C.** The drift decomposition. For each receiver name the largest of the three terms in absolute value. `unexplained_s` should be small. List every receiver where `|unexplained_s| > 0.2`.
- **Table D.** Whether the 343 "unverifiable" lines behave differently from the 352 "passed" ones.
- **Table E and F.** Drew's tdiff idea with the classes written out. See Task 2.
- **Table H.** Missed steps shared by three or more receivers at the same time and size. These point at the master clock (ZOI08), not at the listeners.
- **Table I.** The forty largest missed steps.

**STOP. Paste the Totals and Tables A, B, C, D in the chat report. All tables go into the Task 8 report file.**

### Task 2. Answer Drew's tdiff question from Tables E, F, G (no new computation)

Drew proposed: compute tdiff for each TDoA value, then cluster into three groups: large (clock jumps), small (multipath), the rest (normal).

Write one short section for the report that states, with the numbers from the tables:

1. How many groups the data shows. Use Table F (where the empty gaps in the histogram are) and Table G (which number of groups has the lowest BIC, and where the group centres sit in ms). State both. If they disagree, say so and do not pick one.
2. Why three groups by size alone is not enough. A clock jump and a one epoch outlier can have the same tdiff size. They differ in what happens next: after a jump the level stays, after an outlier it comes back. Table E separates them: `sustained_step` against `large_spike_reflection_range` and `large_spike_beyond_reflection`. Give the three totals. This is the measured reason Stage 2 mis-sized steps at unstable plateaus.
3. How the size classes line up with physics already in the project: 0.5 ms budget, reflections from about 1.5 ms up to 250 ms, clock steps from about 390 ms.

Do not run DBSCAN with new parameters. Do not tune anything. The existing answer to "how many clusters for the regime DBSCAN" stays as written in RECOVERY section 2.

**STOP. Report the section text.**

### Task 3. Why were the missed steps missed? (read only on B1)

For every row of `steps_detected.csv` with `has_line == False`, look for internal evidence in B1.

Before the loop, check the time basis of the evidence table, because Stage 1 shifted `tblDetectionRaw.seconds` but may not have shifted the evidence table:

```python
cols = {r[1] for r in con.execute("PRAGMA table_info(tblClockEventEvidence)")}
orig = "seconds+COALESCE(jump_offset_s,0)" if "jump_offset_s" in cols else "seconds"
```

This is the same rule as `clock_jump_pre._supplement_evidence` (line 792). Record which branch applied. `steps_detected.csv` times and `tblClockJumpCandidates.jump_time_s` are both original receiver time.

For each missed step, with `lo = t_before_s - 5` and `hi = t_after_s + 5`:

- `n_trigger` = rows in the evidence table for that receiver with `OffsetChanged=1 OR CounterRestart=1` and original time in `[lo, hi]`.
- `n_onesec` = rows with `OneSecondAdjustmentEvidence=1` in `[lo, hi]`.
- `cand` = rows of `tblClockJumpCandidates` for that receiver with `status='unresolved'` and `jump_time_s` in `[t_before_s - 300, t_after_s + 300]`. Keep the `note` text of the nearest one.

Assign exactly one class per missed step, first match wins:

1. `unresolved_candidate`: `cand` is not empty. Stage 1 saw the code but could not measure it. Record the note.
2. `trigger_not_candidate`: `n_trigger > 0`. A code exists inside the bracket and Stage 1 did not list it. This would be a defect. List every such row in full.
3. `one_second_flag_only`: `n_onesec > 0`.
4. `no_internal_evidence`.

Also record `bracket_s` for each. A bracket of one or two beacon periods pins the jump time well. A bracket of hours means the receiver was silent.

Save `missed_step_evidence.csv`. Report a table: receiver by class counts, plus for class 1 the count per distinct `note`. Add the median and 90th percentile of `bracket_s` per class.

Bound every query by receiver and time. Do not scan a table once per step without a `Rec_ID` filter. If the loop is slow, load the evidence rows for one receiver once into pandas and use `numpy.searchsorted`.

**STOP. Report the tables.**

### Task 4. Do the one second per record flags explain the spikes? (read only on B1, three receivers)

Background: the ATS guide says flag `1` means a minus one second adjustment on that record and `>` means plus one second (RECOVERY doc; `clock_jump_pre.firmware_record_adjustments`, line 208). If the large spikes are these flagged records, they are known per record and are not clock jumps.

1. Run `PRAGMA table_info(tblDetectionRaw)` on B1. If there is no `Internal` column, write "Task 4 not possible: no Internal column in B1" and go to STOP.
2. Pick the three receivers with the highest `large_spike_beyond_reflection` count in Table E. State them.
3. For each, read the 7DB7 rows: `SELECT seconds+COALESCE(jump_offset_s,0) AS o, Internal FROM tblDetectionRaw WHERE Rec_ID=? AND Tag_ID='7DB7'`.
4. Decode the flag the same way the project does: `parts = Internal.split()`, `flag = parts[3][0]` when there are at least four parts, else empty. Map `'1' -> -1`, `'>' -> +1`, anything else `0`.
5. Join to that receiver's rows of `series_armA_uncorrected.csv` on `o == t` (tolerance 1e-4 s). Report how many series epochs found their row. If fewer than 99% join, STOP and report.
6. For each joined epoch compute `dev = tdoa_s - median(tdoa_s of the 5 epochs before and 5 after, excluding itself)`. Round `dev` to the nearest whole second: `dev_s = round(dev)`.
7. Report one cross table per receiver: rows `flag` (-1, 0, +1), columns `dev_s` (-2, -1, 0, +1, +2, other), cell = epoch count.

Reading it: if nearly all epochs with `dev_s = +1` carry flag `+1` (and the same for minus), the flags explain the spikes. If the flagged epochs sit at `dev_s = 0`, they do not. If the sign is reversed, say so. Report the three tables and one sentence each. Do not correct anything.

**STOP. Report the tables.**

### Task 5. How does the existing Stage 2 ledger compare? (read only on production Arm B)

Read `SELECT Rec_ID, stage, jump_time_s, offset_s, evidence, boundary_uncertain, window_start_s, window_end_s FROM tblClockJumpEvents WHERE stage=2` from `output\11062026_clocksync\jsats3d_2025_earlyfish.db`. Expect 1,653 rows (49 with `Rec_ID='ZOI08'`).

Match Stage 2 listener events to the missed steps (`has_line == False`) one to one per receiver. A pair matches when the step bracket widened by 300 s on each side overlaps `[window_start_s, window_end_s]` (use `jump_time_s` for both ends when the window is null). Among time matches prefer the smallest `|offset_s - step_s|`.

Report per receiver:

| Column | Meaning |
|---|---|
| `missed_steps` | from Task 1 |
| `stage2_events` | Stage 2 rows for that receiver |
| `both_agree` | time match and `|offset_s - step_s| <= 0.05` |
| `both_disagree` | time match, size differs by more than 0.05. Also give how many of these have a size difference within 0.05 of 0.5, 1.0 or 2.0 s |
| `step_only` | missed step with no Stage 2 event |
| `stage2_only` | Stage 2 event with no detected step. Also give how many have `boundary_uncertain=1` |

Treat the 49 master rows separately: for each, count how many listeners have a missed step within 180 s of opposite sign and equal size within 0.05 s, and compare with Table H.

This table tells the owner whether the simple "single difference plus sustained level" test finds the same jumps as the regime DBSCAN, and which of the two is wrong where they differ. Do not conclude which is right. Report counts.

**STOP. Report the table.**

### Task 6. Example fish around jumps (read only; plots from existing position files)

Drew asked: pick random jumps, find fish in the forebay then, plot both arms for two or three fish.

1. **Inventory first.** List which of these exist, with size and last write: Arm A `*_positions.csv` files under `output\10292026_Test\`; the diagnostic Arm B copy `output\11072026_diag_armB\`; `output\11082026_whatif_B1\B3_diagnostic_figures\`. Read every tag column with `dtype={"tag": str}` (the `26E4` trap). If only one arm has position files, STOP and report. Do not run the positions pipeline in this plan.
2. **Fixed selection rule, no hand picking.** Candidate jumps are all rows of the B1 ledger (`stage1_correction_lines.csv`) on the nine solving receivers. For each candidate count consensus fixes of each tag within plus or minus 300 s of the jump in each arm (match across arms by time, nearest within 1 s, never by transNo). Keep candidates where one tag has at least 15 fixes in both arms. Then `numpy.random.default_rng(20261110).permutation` the kept list and take, in that order, the first candidate of each status (`passed`, `unverifiable`) and then the next one that uses a different tag from both. That gives three. If fewer than three exist, report how many and why.
3. **Plot.** One PNG per chosen jump: plan view, plus or minus 30 minutes, Arm A in grey, the other arm in blue, receivers as black triangles, the jump receiver ringed. Title has tag, receiver, local time (PDT), applied offset in ms and status. Add the word DIAGNOSTIC in the title. Below it a small panel of the XY difference between arms against time with a vertical line at the jump.
4. **Table.** Per chosen jump: tag, receiver, local time, offset ms, status, fixes in each arm within 300 s, median and maximum XY difference within 300 s, and the same two numbers for 300 to 1800 s away.

State clearly which arm the blue series is and that it is a gate bypassed diagnostic. A difference between arms is not an error against truth. There is no GPS truth for these fish.

**STOP. Report the inventory, the selection list with the random order, and the table.**

### Task 7. Were any validation drags or static holds inside a jump period? (read only, scoping)

Truth exists only for the test tags on 2025-06-10 to 06-12. The clock work covers 06-15 onward. This task finds out whether a truth based check of jump correction is possible at all.

1. Check `output\zoi11_test\armA\zoi11_A.db` exists (window 2025-06-10 to 06-13, all 20 receivers parsed). Record size and last write. Open read only.
2. Check whether its `tblDetectionRaw` has the columns `OffsetChanged`, `CounterRestart`, `OneSecondAdjustmentEvidence`. If not, report that and skip step 4's code counts.
3. Extract the 7DB7 series with the existing function, exactly as on 2026-11-09: `clock_jump_pre.beacon_first_arrivals(con, "ZOI08", "7DB7", 60.0, t0-60, t1+60)`. Save it as `series_dragwindow.csv` in the output folder.
4. Find sustained steps with the audit script's own function, do not rewrite it:
   ```python
   import sys; sys.path.insert(0, r"C:\Dev\jsat3d\jsats3d\output\11112026_stage1_closure")
   import stage1_closure_audit as sca
   # per receiver: g = series[series.Rec_ID == rec].sort_values("t"); steps = sca.find_steps(g.t.to_numpy(float), g.tdoa_s.to_numpy(float))
   ```
5. Read the test windows from the testing workbook (path in `scripts\drag_test_figures_2025.py`, `SHEETS_FILE`; drag line windows come from the GPS file as in `DataManager`). List every test ID with date, start, end, tag.
6. Report one table: test ID, start, end, then for each of the solving receivers the number of sustained steps and the number of `OffsetChanged or CounterRestart` rows inside the test window widened by 300 s.

If no test window contains a step, say so in one line: a truth based check of jump correction is not possible with the 2025 test data. If some do, list them. They are the cases for a later A/B build that the owner must approve. Do not build anything.

**STOP. Report the table.**

### Task 8. Report file and hand off

Write `C:\Dev\jsat3d\jsats3d\docs\reports\PLAN_B_report_2026-11-11.md` (create the folder). This file is what the reviewer reads. Use exactly these sections:

1. **Run facts.** Git head, branch, test summary line, free disk at start and end, input file sizes, database sizes and last writes at start and end.
2. **Selftest output.** Pasted whole.
3. **Task 1.** The full text of `closure_report.md`, pasted whole, then the check table.
4. **Task 2.** The tdiff section.
5. **Task 3.** Tables. Every `trigger_not_candidate` row in full.
6. **Task 4.** The three cross tables.
7. **Task 5.** The comparison table.
8. **Task 6.** Inventory, selection list, table. File names of the PNGs.
9. **Task 7.** The table and the one line conclusion.
10. **Deviations.** Every place you did something different from this plan, and why. Write "none" if none.
11. **Things that looked wrong.** Anything odd, even if you are not sure. Tracebacks in full.
12. **Open questions for the reviewer.**

Then: final journal entry with `Get-Item` proof, final size and last write of the three databases, and a proposed commit list for the owner (this plan's report, the journal). Do not commit.

**STOP. Final report with journal proof.**

---

## 4. What you may say and what you may not

| Result | Say | Do not say |
|---|---|---|
| Missed steps term dominates Table C | "The Stage 1 drift is the sum of N jumps that carry no applied correction. Stage 1 offsets that were applied match the raw step in M of 696 cases." | "Stage 1 is wrong", "Stage 1 is validated" |
| Size error term is large on a receiver | "On receiver X, K applied offsets differ from the raw step by more than 50 ms; J of them sit on unstable plateaus." | "The offsets are bad" |
| Phantom term is large | "K applied lines have no sustained raw step within 300 s." | "Firmware codes are unreliable" |
| `unexplained_s` is large | "The decomposition does not close on receiver X by Y s. Cause not identified." | Any guess presented as a cause |
| Level anchored column is high | "A what-if that reads each regime's level against the path delay would put P% of epochs within 50 ms. Not implemented. Owner decision." | "This fixes the drift" |

## 5. Known traps for this plan

- `series_armA_uncorrected.csv` `t` is original receiver time. `stage1_correction_lines.csv` `jump_time_s` is original receiver time. `corrected_event_time_s` is not. The script uses `jump_time_s`. Do not swap them.
- `common_mode` in the lines file is a listener count, not a time (the 2026-11-09 hover bug).
- A value near plus or minus 30 s in `tdoa_s` is a pairing flip at a burst boundary, not a clock state.
- ZOI08 is the master and has no series of its own. A master step shows up on every listener at once (Table H).
- CFD05 has no 7DB7 data. It is absent, not zero.
- `"26E4"` reads as a float unless `dtype={"tag": str}`.
- `a.loc[i] - b.loc[j]` aligns on labels. Use `.to_numpy()`.
- PowerShell breaks multi line `python -c`. Save snippets as `.py` files in the output folder and run them.
