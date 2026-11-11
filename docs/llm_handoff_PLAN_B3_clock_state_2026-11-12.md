# Plan B3 for Qwen 3.6 35B MoE: clock state cost, second beacon test, example fish (2026-11-12)

**Author:** external reviewer (no access to data or outputs). **Executor:** Qwen 3.6 35B MoE.
**Read first:** `docs/reports/REVIEW_stage1_findings_2026-11-12.md` (what the audits showed, and which earlier statements are withdrawn), then `docs/llm_handoff_PLAN_B_clock_closure_2026-11-10.md` section 2 (rules) and `docs/llm_handoff_PLAN_B2_followup_2026-11-11.md` section 2 (extra rules). All of those rules apply. **If files disagree, this one wins.**
One task at a time. Every task ends with **STOP**.

Tasks 1 and 2 use the tested tool. You write no analysis code for them. Task 3 is the only task where you write code, and it has two STOP points.

Output folder: `C:\Dev\jsat3d\jsats3d\output\11132026_stage1_b3\`
Journal: `C:\Dev\jsat3d\jsats3d\.ai_journal\session\2026-11-13_stage1_b3.md`

---

## 1. Corrections to carry into the journal

From the review, section 2.2. Write these three lines into the journal in Task 0:

1. ZOI04 and ZOI11 are outside the run's receiver lists. That is a run setting, not a bug. Stage 1 totals use the 16 covered receivers: 4,414 sustained steps, 3,725 without a line.
2. Stage 1 has no regime detection. "Unresolvable" means the events are too close together for its plus or minus 300 s median.
3. The hypothesis that master flags explain the shared plus one second steps is withdrawn (flags Table O).

---

## 2. Tasks

### Task 0. Setup

1. `Set-Location C:\Dev\jsat3d\jsats3d`; toplevel and branch checks; `git log -1 --format="%h %s"`; test suite summary line; free disk (STOP below 3 GB).
2. Three row database table: Arm A, B1, production Arm B, each with path, size in bytes, last write.
3. Create the output folder and journal. Write the three corrections above.
4. Copy `docs\handoff_tools\stage1_closure_audit.py` and `docs\handoff_tools\stage1_followup_audit.py` into the output folder. **Copy them again even if an older copy exists elsewhere. The follow-up tool changed.**
5. From the output folder run both self tests:
   - `python.exe stage1_closure_audit.py --selftest` : eleven OK lines, `SELFTEST PASSED`.
   - `python.exe stage1_followup_audit.py --selftest` : **fifteen** OK lines, `SELFTEST PASSED`. If you see ten lines, you copied the old file.

**STOP. Report with journal proof.**

### Task 1. How much time and how many fish detections fall in each clock state? (read only on B1)

```
python.exe output\11132026_stage1_b3\stage1_followup_audit.py brackets ^
  --series output\11092026_stage1_tdoa_review\series_armA_uncorrected.csv ^
  --db     output\11082026_whatif_B1\jsats3d_2025_B1.db ^
  --out    output\11132026_stage1_b3
Copy-Item output\11132026_stage1_b3\followup_brackets.md docs\reports\PLAN_B3_brackets_2026-11-13.md
```

The tool reads the study tag rows (26E4, 9F8C, 1D5A, B5ED) of every receiver from B1 in original receiver time, and sorts each one by the kind of beacon interval it falls in. The four kinds are defined at the top of the output file.

Checks:

| Check | Expected | If not |
|---|---|---|
| Receivers in Table R | 18 | STOP |
| The four time columns add to 100 on every row (within 0.5) | yes | STOP |
| `fish_rows` above 0 for ZOI01, ZOI02, ZOI03, ZOI05, ZOI06, ZOI07, ZOI09, ZOI10 | yes | Report which are 0 and continue |

Sentences. Fill in and use only these:

- "Across all receivers, A% of fish detections fall where the receiver clock is at the path delay at both neighbouring beacon epochs, B% where it is offset by the same amount at both, C% where the two neighbouring epochs disagree, and D% in beacon gaps over 300 s (Table R3)."
- "The receiver with the largest ambiguous share is X at C% (Table R2). The receiver with the largest steady offset share is Y at B%."
- "ZOI08 is the master and is not in this table. Its own clock changes show up as changes on every listener."

Do not write that any of these detections are "lost", "bad" or "fixable". The table counts where they fall.

**STOP.**

### Task 2. Does a second beacon see the same clock state? (read only on Arm A)

This is the test of whether an offset is a state of the receiver clock (every detection in that minute is shifted) or belongs to single records (only that detection is shifted).

**2a. Which second beacons exist.** Open Arm A read only and run:
`SELECT r.Rec_ID, r.Tag_ID, t.pulseRate FROM tblReceiver r LEFT JOIN tblTag t ON t.Tag_ID = r.Tag_ID WHERE r.Rec_ID IN ('ZOI01','ZOI02','ZOI03')`
Expected tags: FFC7 at ZOI01, 7D2D at ZOI02, 7DBC at ZOI03. Report the three rows. Use the tag names the query returns, not the expected ones, if they differ. Use the returned `pulseRate` as `--period`. If a pulse rate is NULL, use 60.0 and say so.

**2b. Extract each second beacon against ZOI08.** One command per tag. Each is one scan of Arm A for that tag (minutes). The tool refuses a shifted database and writes only the CSV.

```
python.exe output\11132026_stage1_b3\stage1_followup_audit.py extract ^
  --db output\10292026_Test\jsats3d_2025_earlyfish.db ^
  --tag 7D2D --master ZOI08 --period <pulseRate> ^
  --scripts-dir C:\Dev\jsat3d\jsats3d\scripts ^
  --out-csv output\11132026_stage1_b3\series_7D2D.csv
```

Repeat for the other two tags with their own file names. Paste the console lines of each (raw rows, paired epochs, per receiver counts). If the tool says "No paired epochs", ZOI08 does not hear that beacon: report it and go on to the next tag.

**2c. Compare.** One command per extracted tag:

```
python.exe output\11132026_stage1_b3\stage1_followup_audit.py crossbeacon ^
  --series output\11092026_stage1_tdoa_review\series_armA_uncorrected.csv ^
  --second-series output\11132026_stage1_b3\series_7D2D.csv ^
  --label 7D2D --out output\11132026_stage1_b3
Copy-Item output\11132026_stage1_b3\followup_crossbeacon_7D2D.md docs\reports\PLAN_B3_crossbeacon_7D2D_2026-11-13.md
```

Checks per tag: Arm A size and last write unchanged; receivers compared at least 5 (if fewer, report the number and continue); Totals `paired_stable` above 1,000 (if lower, say the comparison is thin and give the number).

Sentences. Per tag, read Totals `agreement_when_first_shifted_pct` and use exactly one:

| Value | Sentence |
|---|---|
| 90 or more | "When the master beacon shows a receiver shifted by 50 ms or more, beacon T shows the same shift in A% of N comparisons. The shift behaves as a state of the clock." |
| 50 to 90 | "Beacon T shows the same shift in A% of N comparisons. Mixed: part state, part not. Per receiver values range from X to Y (Table S)." |
| below 50 | "Beacon T shows the same shift in only A% of N comparisons. The shift does not behave as a state of the receiver clock." |

Then add, from Totals: the count `only_second_beacon_shifted`. These are epochs where the second beacon is shifted and the master beacon is not. Say the count. Do not explain it.

One caution to write next to the sentences: both series use ZOI08 as reference, so a change in ZOI08's own clock appears in both and counts as agreement. This test cannot separate a listener state from a master state. It separates a state from a single record effect.

**STOP.**

### Task 3. Example fish around jumps, two arms

Arms, fixed. Do not substitute others:
- Arm A, grey: `output\10292026_Test\figures\<tag>_positions.csv`
- Diagnostic Arm B (Stage 1 + Stage 2), blue: `output\11072026_diag_armB\figures\<tag>_positions.csv`

**3a. Look before computing.** For tag 1D5A in each arm: print the column names, the dtypes, the first three rows, the row count, and the distinct values of any column that labels the row type (consensus, smoothed, Kalman or similar). Read with `dtype={"tag": str}` if a tag column exists. State which column is time, which are x and y, and which rows are the consensus fixes. State the time basis you believe the time column uses, and why. Confirm both arms have the same columns.

**STOP. Report this and wait for the owner's OK on the column choice.**

**3b. Select by rule, no hand picking.** Candidate jumps: the rows of `stage1_correction_lines.csv` on ZOI01, ZOI02, ZOI03, ZOI05, ZOI06, ZOI07, ZOI09, ZOI10. For each candidate and each tag, count consensus fixes within 300 s of `jump_time_s` in each arm. Keep candidates where some tag has at least 15 fixes in **both** arms. Shuffle the kept list with `numpy.random.default_rng(20261113).permutation`. Walk the shuffled list and take: the first with status `passed`; the first with status `unverifiable`; then the next whose tag differs from both. Report the count of candidates, the count kept, and the first ten rows of the shuffled list with the three chosen ones marked.

Note that `jump_time_s` is original receiver time and the arms' fix times are in each arm's own corrected time. They differ by at most a few seconds. A 300 s window absorbs that. Say so on the report; do not try to convert.

**STOP. Report the selection table.**

**3c. Plot and tabulate.** One PNG per chosen jump in `output\11132026_stage1_b3\fish_examples\`:
- Top panel, plan view, equal aspect, 30 minutes either side of the jump. Arm A consensus fixes as grey dots joined by a thin grey line. Diagnostic Arm B the same in blue. Receivers from either database's `tblReceiver` (`X`, `Y`) as black triangles with labels; the jump's receiver ringed in red.
- Bottom panel: XY distance between the arms against time, for fixes matched across arms by nearest time within 1 s (never by transNo). Vertical line at the jump.
- Title: `DIAGNOSTIC  <tag>  <receiver>  <local time> (PDT)  applied <offset> ms  <status>`.

Table, one row per chosen jump: tag, receiver, local time, offset ms, status, fixes within 300 s in each arm, matched pairs within 300 s, median and maximum XY distance within 300 s, the same two for 300 to 1,800 s away.

Open each PNG and write one line on what it shows, including anything that looks wrong.

Wording: the blue arm is a gate bypassed diagnostic. A distance between arms is not an error. There is no GPS truth for these fish. Never write "improved" or "degraded"; write the distances.

**STOP.**

### Task 4. Report and hand off

Write `docs\reports\PLAN_B3_report_2026-11-13.md`:

1. **Run facts.** Head, branch, test line, free disk start and end, the three row database table at start and end.
2. **Self tests.** Both outputs whole.
3. **Files copied into `docs\reports\`.** Name and size in bytes of each.
4. **Checks.** Every check from Tasks 1 and 2 with its value.
5. **Sentences.** From Tasks 1 and 2, each with its table.
6. **Task 2a rows and Task 2b console lines.**
7. **Task 3.** The 3a column report, the 3b selection table, the 3c table, PNG names and one line each.
8. **Deviations.** "none" if none.
9. **Things that looked wrong.**
10. **Open questions.**

Final journal entry with `Get-Item` proof. Proposed commit list. Do not commit.

**STOP.**
