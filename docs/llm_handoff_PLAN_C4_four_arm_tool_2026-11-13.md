# Plan C4: run the four arm tool for the ZOI11 and CFD01 evaluation (2026-11-13)

**Author:** external reviewer (no access to data or outputs). **Executor:** the model the owner assigns.

**Read first, in this order:**
1. `docs/llm_handoff_REPORT_CONTRACT.md`. It says what your report must contain. It is binding.
2. `docs/reports/REVIEW_plan_c3_run_2026-11-12.md`. It says what went wrong before, issue by issue.
3. `docs/pm_request_zoi11_cfd01_evaluation_2026-11-11.md`. It says what the PM wants.

**If files disagree, this one wins.** Plans C, C2 and C3 are closed. Do not follow them.

---

## 1. What is different this time

**You write no code. Not one line. Not a query, not a helper, not a one line check.**

A tested tool does all the analysis: `docs\handoff_tools\zoi11_cfd01_four_arm.py`. You run six commands, copy the files they write, answer fixed questions, and stop where it says STOP.

If you think something needs a script, a query, or a different definition: STOP and write the question into the report. The reviewer will change the tool.

If the tool fails: paste the whole traceback, STOP. Do not edit the tool. Do not work around it.

## 2. Hard rules

1. No code (section 1).
2. No builds. No `run_data.py`, no `_2025_pipeline.py`. The four databases exist.
3. No deletes, no moves, no renames. Nothing is written outside the output folder, `docs\reports\` and the journal.
4. Every task ends with **STOP**. STOP means: write the journal entry with the clock time, show the owner what the task says to show, and do nothing until the owner replies. Quote the reply word for word in the journal.
5. Tool output is copied with `Copy-Item`. It is never retyped, shortened or reformatted.
6. You do not explain results. You do not write "indicates", "suggests", "because". The tool writes the sentences.
7. No commit. No push.

## 3. Paths

| What | Path |
|---|---|
| Repository | `C:\Dev\jsat3d\jsats3d` |
| Tool in the repository | `docs\handoff_tools\zoi11_cfd01_four_arm.py` |
| Output folder (new) | `output\11112026_zoi11_cfd01\c4\` |
| Arm N, "1. Excluded" | `output\11112026_zoi11_cfd01\armN\arm_N.db` |
| Arm Z, "2. ZOI11 Only" | `output\11112026_zoi11_cfd01\armZ_r2\arm_Z.db` |
| Arm C, "3. CFD01 Only" | `output\11112026_zoi11_cfd01\armC\arm_C.db` |
| Arm ZC, "4. Both" | `output\11112026_zoi11_cfd01\armZC_r2\arm_ZC.db` |
| Journal (new) | `.ai_journal\session\2026-11-13_zoi11_cfd01_c4.md` |
| Cover report | `docs\reports\PLAN_C4_report_2026-11-13.md` |

Arms Z and ZC are the `_r2` files, built with `--init-from`. The older `armZ\arm_Z.db` and `armZC\arm_ZC.db` are not used and not touched.

Set these once per PowerShell session. Copy them exactly:

```powershell
Set-Location C:\Dev\jsat3d\jsats3d
$OUT  = "output\11112026_zoi11_cfd01\c4"
$TOOL = "$OUT\zoi11_cfd01_four_arm.py"
$A = @("--scripts-dir","C:\Dev\jsat3d\jsats3d\scripts",
       "--armN","output\11112026_zoi11_cfd01\armN\arm_N.db",
       "--armZ","output\11112026_zoi11_cfd01\armZ_r2\arm_Z.db",
       "--armC","output\11112026_zoi11_cfd01\armC\arm_C.db",
       "--armZC","output\11112026_zoi11_cfd01\armZC_r2\arm_ZC.db",
       "--out",$OUT)
```

Every tool command below is then `python.exe $TOOL <command> @A`. After every command record `$LASTEXITCODE`.

Each command loads the four databases first. Expect several minutes per command. Do not interrupt it.

---

## 4. Tasks

### Task 0. Setup and self test

1. Run identity and machine state as the report contract sections 1 and 2 say. The four databases of section 3 go in the database table. If one is missing, STOP.
2. `New-Item -ItemType Directory $OUT` (if it exists already and is not empty, STOP).
3. `Copy-Item docs\handoff_tools\zoi11_cfd01_four_arm.py $TOOL`
4. `Get-FileHash -Algorithm SHA256 docs\handoff_tools\zoi11_cfd01_four_arm.py, $TOOL`. The two hashes must be equal. Put both in the report.
5. Self test:
   `python.exe $TOOL --selftest --scripts-dir C:\Dev\jsat3d\jsats3d\scripts`

| Check | Expected |
|---|---|
| First line | `zoi11_cfd01_four_arm fourarm-2026-11-12-a` |
| Lines starting `OK` | 23 |
| Lines starting `FAIL` | 0 |
| The word `SKIPPED` | does not appear |
| Last line | `SELFTEST PASSED (23 checks)` |

Paste the whole self test output. The self test also prints `CHECK` lines and one line `IDENTITY DOES NOT HOLD (1 transmissions over 1 mm)`. That line is the test finding a difference that was planted on purpose. It is correct.

**STOP. If any check differs, do not go on.**

### Task 1. Preflight

`python.exe $TOOL preflight @A`

It writes `c4_preflight.md`. If it stops with "GPS file not found" or "testing workbook not found", the `K:` drive is not reachable. STOP and ask the owner for the two paths. They are then added to `$A` as `"--gps-file","<path>","--sheets-file","<path>"`. Do not choose a file yourself. The `gps_csv` in the run files is a different file and is not a substitute.

`Copy-Item $OUT\c4_preflight.md docs\reports\PLAN_C4_preflight_2026-11-13.md`

Checks. The console prints five `CHECK` lines. Copy each into the check table with its result:

| Check | Expected | If FAIL |
|---|---|---|
| same tables in all four arms | PASS | STOP |
| membership: N neither, Z ZOI11 only, C CFD01 only, ZC both | PASS | STOP. A database is at the wrong path. |
| receiver coordinates equal in all arms | PASS | STOP |
| local origin equal in all arms | PASS | STOP |
| transmission numbers aligned | PASS | STOP. Arms cannot be matched on transmission number. |

Then copy these numbers from `c4_preflight.md` section 5 into the cover report, in one table: for each arm and tag, `solution_rows`, `trans_with_solution`, `trans_after_position_screen`. Under it write exactly: "Solution rows rise with each added receiver because the solver writes one row per receiver combination. Transmissions are the unit for coverage."

**STOP. Show the owner the five check lines and that table.**

### Task 2. Identity

`python.exe $TOOL identity @A`

`Copy-Item $OUT\c4_identity.md docs\reports\PLAN_C4_identity_2026-11-13.md`

The console ends with one of two lines. Copy that line into the report.

| Console line | What you do |
|---|---|
| `IDENTITY HOLDS (0 transmissions over 1 mm)` | Go on to Task 3 after the STOP. |
| `IDENTITY DOES NOT HOLD (N transmissions over 1 mm)` | Exit code is 2. That is not a crash. Copy the file, report N, and STOP. **Do not explain it. Do not investigate it.** The owner and the reviewer decide what happens next. |

Also copy into the cover report, for each of the three arms and both tags, the columns `trans_in_arm`, `trans_with_added_receiver_row`, `compared`, `over_1mm`, `over_1mm_with_same_row_count`, `compared_but_heard_by_added`.

**STOP.**

### Task 3. Events

`python.exe $TOOL events @A`

`Copy-Item $OUT\c4_events.md docs\reports\PLAN_C4_events_2026-11-13.md`

Copy into the cover report: the console's count table (kind, scored, events), and the list of events that are not scored with their reasons.

Then answer these from `c4_events.md`, with numbers only:

1. How many forebay holds are scored? How many debris barrier holds? How many drags?
2. For how many scored events is `ZOI11_heard_FFD3` zero? `ZOI11_heard_FC36` zero? The same two for CFD01.
3. Which scored events have `trans_FFD3` and `trans_FC36` both zero? List them.

**STOP. Show the owner the event table. The owner confirms the list is complete, or names what is missing. Quote the reply.**

### Task 4. Three sample plots

`python.exe $TOOL plots --sample @A`

It makes plots for one forebay hold, one debris barrier hold and one drag, and writes `c4_plots_sample.md` and `index_sample.html`.

`Copy-Item $OUT\c4_plots_sample.md docs\reports\PLAN_C4_plots_sample_2026-11-13.md`

Open each PNG in `$OUT\plots_2d\`. For each one answer, in a table with one row per PNG:

| Question | Answer form |
|---|---|
| File name | text |
| Does the legend list four arms, each with `n=`? | yes / no |
| The four `n=` values | four numbers |
| Is GPS truth drawn (star, or black line)? | yes / no / "no GPS truth" in title |
| Are ZOI11 and CFD01 drawn as red triangles with labels? | yes / no |
| Does any legend entry say "off plot"? If yes, the numbers | no / numbers |
| Does the title say "local time (PDT)"? | yes / no |
| Anything that looks wrong | one line, or "nothing" |

Open each HTML in `$OUT\plots_3d\` in a browser. For each one:

| Question | Answer form |
|---|---|
| File name | text |
| Does the page show a 3D scene that rotates with the mouse? | yes / no |
| Does hovering a point show the arm name, a time and x, y, z? | yes / no |
| Are the receivers labelled? | yes / no |
| Anything that looks wrong | one line, or "nothing" |

**STOP. Show the owner the three PNG files and one HTML. The owner says go on, or says what to change. Quote the reply. If the owner wants a change to the plots, STOP the run: the reviewer changes the tool.**

### Task 5. Scores, tables and all plots

```powershell
python.exe $TOOL score @A
python.exe $TOOL plots @A
```

```powershell
Copy-Item $OUT\c4_tables.md docs\reports\PLAN_C4_tables_2026-11-13.md
Copy-Item $OUT\c4_plots.md  docs\reports\PLAN_C4_plots_2026-11-13.md
```

Checks:

| Check | Expected | If not |
|---|---|---|
| `event_scores.csv` rows (console line of `score`) | scored events x 2 tags x 4 arms | STOP |
| `c4_tables.md` has sections V4a, V4b, V4c, V4d and "Summary sentences" | yes | STOP |
| PNG count in `plots_2d\` equals "plots made" on the console | yes | report both numbers |
| HTML count in `plots_3d\` (not counting `plotly.min.js`) equals the PNG count | yes | report both numbers |
| `index.html` exists | yes | STOP |
| Size and last write of the four databases | unchanged from Task 0 | STOP |

Open five PNG files that you did not open in Task 4, among them one debris barrier hold and one drag if they exist. Answer the Task 4 PNG questions for each.

Do **not** copy `event_scores.csv`, `fixes.csv`, the PNG files or the HTML files into `docs\`. They stay in the output folder.

**STOP.**

### Task 6. Cover report and hand off

Write `docs\reports\PLAN_C4_report_2026-11-13.md` with exactly these sections. Each is described in the report contract.

1. **Run identity.**
2. **Machine state**, start and end.
3. **Commands.** One row per command: full command line, exit code, elapsed time.
4. **Console output** of each command (contract section 4).
5. **Files copied to `docs\reports\`.** Source, copy, bytes of each. Then the two tool hashes.
6. **Checks.** Every check of Tasks 0 to 5 as a row: check, expected, observed, result.
7. **Preflight numbers** (the Task 1 table and its fixed sentence).
8. **Identity result** (the console line and the Task 2 columns).
9. **Events** (the Task 3 counts and the three answers).
10. **Plot answers** (the Task 4 and Task 5 tables).
11. **Summary for the PM.** Copy the block "Summary sentences" from `c4_tables.md`, whole, with its last line "Include or exclude is the PM's decision." Add nothing. Change nothing.
12. **Stops.** Each STOP: time, what was shown, the owner's reply quoted.
13. **Deviations.** Both directions, or "none".
14. **Things that looked wrong.** Observations only.
15. **Not done.**
16. **Questions for the reviewer.**

Final journal entry with the `Get-Item` lines of the seven files in `docs\reports\`. Proposed commit list: the seven report files and the journal. Nothing under `output\`.

**STOP. Do not commit.**

---

## 5. What comes back, and what the reviewer will do with each file

| File in `docs\reports\` | Written by | The reviewer uses it to |
|---|---|---|
| `PLAN_C4_report_2026-11-13.md` | you, fixed form | see that the plan was followed: commands, exit codes, hashes, stops, deviations |
| `PLAN_C4_preflight_2026-11-13.md` | tool | confirm the four databases are the right ones and comparable; see the solver row census with the 9999 rows separated |
| `PLAN_C4_identity_2026-11-13.md` | tool | settle whether the arms differ only through the added receiver's rows; if not, read the ten largest differences |
| `PLAN_C4_events_2026-11-13.md` | tool | see which holds and drags exist, which are scored, and how often ZOI11 and CFD01 heard each tag in each event |
| `PLAN_C4_plots_sample_2026-11-13.md` | tool | see fix counts per arm for the three sample events |
| `PLAN_C4_tables_2026-11-13.md` | tool | the result: fixes per event and arm, fixes gained and lost, where the gained fixes are, error change against distance from the two receivers |
| `PLAN_C4_plots_2026-11-13.md` | tool | confirm every event and tag pair has a plot or a stated reason for none |

The reviewer will then write a review and a short text for the PM built from Tables V4b, V4c and V4d.

## 6. Open points for the owner (not for the executor)

1. **The distance bands** in Table V4c (under 25 m, 25 to 50 m, over 50 m from the nearer of ZOI11 and CFD01) are the reviewer's choice. They stand in for "underneath the debris barrier". If the PM has the barrier's line or polygon, the tool can use that.
2. **What counts as a fix.** For holds the tool uses the project's receiver set consensus (two receiver sets within 10 m). The earlier client figures plotted every per transmission median for holds. The consensus form was chosen so that one wild solver row cannot count as a fix. Say so if the PM wants the other form.
3. **The reviewer cannot see the plots.** If a few sample PNG files may be committed under `docs\reports\figures\`, the review of the plots becomes possible. Otherwise it rests on the executor's yes and no answers and on your own look.
4. **Deep receivers** kept their configured positions in every arm (`solutions_used = 0`, C3 report). It affects all four arms alike. Whether that is normal for the 06-10 to 06-13 window is a question for Kevin.

## 7. Known traps

- PowerShell: one command per line. No `^`. No multi line `python -c`.
- `$A` and `$TOOL` are lost when the PowerShell window closes. Set them again from section 3.
- An exit code of 2 from `preflight` or `identity` means a check did not pass. It is a result, not a crash. A traceback is a crash.
- The tool imports `scripts\drag_test_figures_2025.py`. Do not run that file as a script. It exits non zero on these databases by design.
- `plotly.min.js` in `plots_3d\` is shared by all HTML files. It is not a plot.
- Times in every output are local time (PDT). Never write "UTC".
