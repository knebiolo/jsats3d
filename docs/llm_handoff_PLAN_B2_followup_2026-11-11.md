# Plan B2 for Qwen 3.6 35B MoE: follow-up to the Stage 1 closure audit (2026-11-11)

**Author:** external reviewer (no access to data or outputs). **Executor:** Qwen 3.6 35B MoE.
**Read first:** `docs/llm_handoff_PLAN_B_clock_closure_2026-11-10.md` section 2 (rules, all still apply), then this file. **If files disagree, this one wins.**
One task at a time. Every task ends with **STOP**.

This plan is short on purpose. Three script runs, one existing project command, one inventory. **You do not write analysis code in this plan.** You run tested tools and copy their output files.

---

## 1. Review of the first run (`docs/reports/PLAN_B_report_2026-11-11.md`)

### 1.1 What the first run established

- 671 of 696 Stage 1 lines sit on a real raw step of the same size. All 352 "passed" lines matched. All 25 problem lines are in the "unverifiable" group.
- 7,024 sustained steps exist in the raw series. 6,335 have no Stage 1 line.
- The only large group of tdiff values is centred on 1,000 ms.
- On three receivers, a record carrying the minus one second flag sits one second early in 82 to 95% of cases.

### 1.2 Errors in the first run. Record these in the journal as corrections.

1. **Task 3 read the wrong database.** The notes reported ("supplementary boundary uncertain", "overlapping clean DBSCAN regimes", "residual TDoA step") are Stage 2 notes. They exist in production Arm B, not in B1. The `unresolved_candidate` counts are therefore not Stage 1 results. Task 3 below replaces them.
2. **Task 4 conclusion was backwards.** The tables show the flag is a reliable minus one second marker. The sentence "The flags do NOT explain the large spikes" is not supported. The figure "610/1,576 = 39%" does not match the table (610 flagged of 758 epochs at minus one second).
3. **Task 2 called 20,500 large spikes "transient reflections".** About 9,500 of them are above 250 ms, the project's upper bound for a reflection. They cannot all be reflections.
4. **Task 6 passed a STOP.** Only one arm had position files. The plan said STOP. Single arm plots cannot compare arms.
5. **The report left tables out.** Tables B, E, F, H, I were summarised or missing. Run facts listed one database out of three, with Arm A's timestamp on it. No end of run check was reported.

### 1.3 Errors in the reviewer's own plan. Do not repeat them to anyone as findings.

1. The drift arithmetic in Table C does not close. `unexplained_s` is large on 12 receivers. So Table C cannot be used to say how much drift comes from each cause. Task 1 below looks for where the missing level change sits.
2. The class name `trigger_not_candidate` was described as "a defect". That was wrong. Stage 1 measures every internal code event and silently drops the ones it cannot measure (`quantify_jumps` returns them as `evidence_only` and `unresolvable`, and the applying run does not save them). Task 3 below recovers those two lists.
3. Table A ("93% to 14% within 50 ms") measures absolute level. The later clock synchronisation absorbs a constant offset. **Table A is not evidence that Stage 1 harms positions. Never show it to the PM without that sentence.**
4. The expected final offsets (+15.3 s, -11.5 s) were season extremes, not end values. The mismatch in the first report is the reviewer's error.

### 1.4 The question this plan tests

Positive one second deviations on the listeners carried no listener flag. Many missed steps are shared by 3 to 17 receivers at once. Hypothesis: those are flagged records on the **master** (ZOI08). A minus one second record on the master makes every listener's TDoA read plus one second. If the per record flags on both ends account for most steps, then the answer to Drew's "for some reason not all jumps are identified" is that Stage 1 triggers only on `OffsetChanged` and `CounterRestart`, while the receiver marks its one second adjustments record by record in a different field. **This is a hypothesis. Task 2 tests it. Do not state it as a result before then.**

---

## 2. Extra rules for this plan

1. **No analysis code of your own.** If a tool fails, paste the traceback and STOP.
2. **Reports are copied, not retyped.** Each tool writes a markdown file. You copy that file into `docs\reports\` with `Copy-Item`. You do not paste, shorten, reformat or "tidy" it.
3. **Database facts are a three row table every time**: Arm A, B1, production Arm B, each with path, size in bytes, last write. At the start and at the end.
4. Every sentence of interpretation you write must name the table and column it comes from.
5. Output folder: `C:\Dev\jsat3d\jsats3d\output\11122026_stage1_followup\`. Journal: `C:\Dev\jsat3d\jsats3d\.ai_journal\session\2026-11-12_stage1_followup.md`.

---

## 3. Tasks

### Task 0. Setup

1. `Set-Location C:\Dev\jsat3d\jsats3d`; toplevel and branch checks; `git log -1 --format="%h %s"`; test suite summary line; free disk (STOP below 3 GB).
2. Three row database table (rule 3).
3. Create the output folder and journal. Write the corrections of section 1.2 into the journal.
4. Copy **both** `docs\handoff_tools\stage1_closure_audit.py` and `docs\handoff_tools\stage1_followup_audit.py` into the output folder.
5. Run both self tests from the output folder:
   - `python.exe stage1_closure_audit.py --selftest` must end with `SELFTEST PASSED` (eleven OK lines).
   - `python.exe stage1_followup_audit.py --selftest` must end with `SELFTEST PASSED` (ten OK lines).
6. Publish the first run's full tables, unedited:
   `Copy-Item output\11112026_stage1_closure\audit\closure_report.md docs\reports\PLAN_B_closure_report_full_2026-11-11.md`

**STOP. Report with journal proof.**

### Task 1. Where does the level change sit? (CSV only)

```
python.exe output\11122026_stage1_followup\stage1_followup_audit.py signed ^
  --series output\11092026_stage1_tdoa_review\series_armA_uncorrected.csv ^
  --lines  output\11092026_stage1_tdoa_review\stage1_correction_lines.csv ^
  --out    output\11122026_stage1_followup
Copy-Item output\11122026_stage1_followup\followup_signed.md docs\reports\PLAN_B2_signed_2026-11-12.md
```

Checks: 18 receivers in Table J; in every row `total_s` equals `last_minus_first_s` to 0.001.

Write at most six sentences, each naming its table:
- Table J: for each receiver, which column cancels the `sustained_step_s` column. If it is `normal_s` or `small_multipath_or_jitter_s`, the level moves back gradually (a ramp). If it is a `large_spike` column, the level moves back in jumps too short to count as sustained.
- Table K: whether down steps and up steps balance.
- Table L and L2: what share of the steps without a line are near one or two seconds.
- Table M: how long a typical regime away from the path delay lasts, in minutes.

**STOP.**

### Task 2. Do the per record flags on listener and master account for the steps? (read only on B1)

```
python.exe output\11122026_stage1_followup\stage1_followup_audit.py flags ^
  --series output\11092026_stage1_tdoa_review\series_armA_uncorrected.csv ^
  --lines  output\11092026_stage1_tdoa_review\stage1_correction_lines.csv ^
  --db     output\11082026_whatif_B1\jsats3d_2025_B1.db ^
  --out    output\11122026_stage1_followup
Copy-Item output\11122026_stage1_followup\followup_flags.md docs\reports\PLAN_B2_flags_2026-11-12.md
```

The tool opens the database read only, reads the 7DB7 rows of every receiver, decodes the flag, and joins each series epoch to its listener row and to its master row. It then removes the flagged second from an in-memory copy and counts sustained steps again.

Checks:

| Check | Expected | If not |
|---|---|---|
| Totals `min_joined_pct` | at least 99 | STOP. The series and the database disagree on time |
| Totals `steps_before` | 7,024 | STOP |
| Table N receivers | 18 | STOP |

How to read it. Use these sentences, filled in, and no others:

| Result | Sentence |
|---|---|
| `steps_removed_pct` is 60 or more | "Removing the documented per record one second adjustment on listener and master, in memory, takes sustained steps from A to B (C% fewer). Table O: of the epochs the flags predict at plus or minus one second, D% are observed there." |
| between 20 and 60 | "The per record flags account for part of the steps: A to B (C% fewer)." |
| below 20 | "The per record flags account for few steps: A to B (C% fewer). The hypothesis in section 1.4 is not supported." |
| `plus_flags` is 0 | Add: "No plus one second flag occurs in the 7DB7 rows of any receiver." |
| `steps_after` is higher than `steps_before` on any receiver | List those receivers. Add: "On these receivers the adjustment made the series worse. Cause not identified." |

Also report from Table O the row `predicted 0`: the counts under `observed -1 s` and `observed +1 s` are whole second offsets that no flag explains.

**Do not apply anything to any database. Do not call this a fix.** Whether a per record correction belongs in the pipeline is the owner's decision (it already exists as a diagnostic: `clock_jump_pre.firmware_record_adjustments`).

**STOP.**

### Task 3. What did Stage 1 decide about the events next to each missed step? (read only on Arm A)

Stage 1 has a report mode that writes all three of its lists without touching the database. Run the existing command. It opens the database with `mode=ro`.

```
python.exe scripts\clock_jump_pre.py output\10292026_Test\jsats3d_2025_earlyfish.db ^
  --master-receiver ZOI08 --beacon-tag 7DB7 --period 60.0 ^
  --start "2025-06-15 00:00:00" --end "2025-08-06 23:59:59" ^
  --receivers ZOI05 ZOI06 ZOI07 ZOI08 ZOI09 ZOI10 CFD01 CFD02 CFD03 CFD04 CFD06 CFD07 CFD08 CFD09 ZOI01 ZOI02 ZOI03 ^
  --report output\11122026_stage1_followup\stage1_report *>&1 | Tee-Object output\11122026_stage1_followup\stage1_report_console.txt
```

Do not add `--apply`. Arm A is 18 GB with no indexes, so each receiver costs a full scan. Expect a long run (it may exceed an hour). Do not run `quick_check`. Do not interrupt it and restart with fewer receivers.

Checks:

| Check | Expected | If not |
|---|---|---|
| Console line `Jumps accepted:` | 696 | STOP. Report the three counts. Do not run the next command |
| `jumps.csv` rows | 696 | STOP |
| Arm A size and last write afterwards | unchanged | STOP and report |

Then:

```
python.exe output\11122026_stage1_followup\stage1_followup_audit.py dispositions ^
  --steps output\11112026_stage1_closure\audit\steps_detected.csv ^
  --stage1-report output\11122026_stage1_followup\stage1_report ^
  --out output\11122026_stage1_followup
Copy-Item output\11122026_stage1_followup\followup_dispositions.md docs\reports\PLAN_B2_dispositions_2026-11-12.md
```

Report the three counts from the console line (accepted, evidence only, unresolvable) and at most four sentences from Tables P and Q:
- How many missed steps have no Stage 1 event at all within 300 s (`no_stage1_event`). These are jumps the internal codes never announced.
- How many had an event that Stage 1 could not measure, and the two most common reasons in Table Q.
- How many had an event measured below 50 ms or above 5 s.

Note for Table P: ZOI04 and ZOI11 were not in the Stage 1 receiver list. All of their steps will read `no_stage1_event`. Say so. That is why they have zero lines (question 1 of the first report).

**STOP.**

### Task 4. Is a second arm's position output available? (inventory only)

No plots. No pipeline runs.

1. `Get-ChildItem C:\Dev\jsat3d\jsats3d\output -Recurse -Filter *_positions.csv` : list folder, file, size, last write.
2. `Get-ChildItem C:\Dev\jsat3d\jsats3d\output -Recurse -Filter *.db` : list folder, file, size, last write. For each database under `11072026_diag_armB` and `11082026_whatif_B1`, open read only and report whether `tblPositions_Deng` exists and its row count.
3. Also list any folder matching `C:\Dev\jsat3d\output_archive_*` and whether it contains `zoi11_test`.

**STOP. Report the three listings.** The owner decides which arm, if any, to export for the fish examples.

### Task 5. Report and hand off

Write `docs\reports\PLAN_B2_report_2026-11-12.md` with exactly these sections:

1. **Run facts.** Head, branch, test line, free disk start and end, the three row database table at start and at end.
2. **Self tests.** Both outputs, whole.
3. **Files copied into `docs\reports\`.** Name and size of each.
4. **Checks.** Every check table from Tasks 1 to 3 with the values found.
5. **Sentences.** The sentences from Tasks 1 to 3, each naming its table.
6. **Task 4 listings.**
7. **Deviations.** "none" if none.
8. **Things that looked wrong.**
9. **Open questions.**

Final journal entry with `Get-Item` proof. Proposed commit list: the five files in `docs\reports\` from this plan, the journal. Do not commit.

**STOP.**
