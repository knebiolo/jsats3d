# Reviewer notes on the Plan C3 run (commit e055ee5)

**Author:** external reviewer. Code and committed reports only. No access to data or outputs.
**Sources:** `docs/reports/PLAN_C3_report_2026-11-12.md`, `docs/reports/PLAN_C2_C3_full_report_2026-11-12.md`, `.ai_journal/session/2026-11-12_zoi11_cfd01_c2.md`, `jsats3d/jsats3d.py`, `scripts/drag_test_figures_2025.py`, `scripts/ent_analysis_2025.py`.

Three runs have now ended without a plot. Each time the cause was the same: the executor wrote its own analysis code and its own definitions. Plan C4 removes that step. A tested tool does the analysis. The executor runs it and copies its output.

---

## 1. What the run established

| Fact | Source | Use |
|---|---|---|
| All 20 receiver coordinates are equal in the four arms. | C3 report Task 1 | The reviewer's deep receiver hypothesis is refuted. |
| The deep receivers kept their configured positions in every arm (`solutions_used = 0`). | C3 report Task 1 | See issue 9. |
| Arms Z and ZC rebuilt with `--init-from` give the same solution row counts as the first run's workaround. | journal | The workaround was equivalent. The old and new databases can both be trusted as builds. |
| Transmissions kept by the position screen: FFD3 976 / 982 / 1,079 / 1,117 and FC36 1,371 / 1,426 / 1,559 / 1,520 for arms N / C / Z / ZC. | full report 7c | The only coverage figure in the report that counts transmissions. |
| Free disk is 20.36 GB. All four databases exist. | journal | Disk no longer blocks the work. |

## 2. Issues, one by one

Each issue names what was written, why it is wrong, and what Plan C4 does about it.

### Issue 1. Solver rows reported as positioned transmissions

**Written:** section 7a is titled "Export-found solutions (positioned transmissions)". Section 13: "ZC provides 2.3x more positioned transmissions than baseline."

**Why it is wrong:** the solver writes one row per receiver combination and per root. A transmission heard by six receivers gives many rows. Adding a receiver raises the count even when no transmission gains a fix. The report's own section 7c shows the transmission counts: 976 against 1,117 for FFD3, 1,371 against 1,520 for FC36. That is 1% to 14% more, not 134% more.

**Plan C4:** the tool prints `solution_rows` and `trans_with_solution` side by side in every table and defines a fix at the top of every output file.

### Issue 2. "Silent" was redefined, so the identity check tested nothing

**Written:** "ZOI11 tag = 1FF1, CFD01 tag = B32A. Both receivers have 0 detections of their tags in `tblDetectionFilterPrimary`. Therefore all FFD3 and FC36 transmissions are CFD01-silent."

**Why it is wrong:** Plan C3 defined a silent transmission as one where no solution row of the arm contains the added receiver. The executor instead asked whether the receiver hears its own beacon tag. That is a different question. A receiver's own beacon is not in the filter table at all, because only the fish tags and the deep beacons are processed. With that definition every transmission counts as silent, so "Z against N, silent only" became "Z against N, everything". Differences are expected there: arm Z has extra rows with ZOI11 in them, and the medians move.

**Consequence:** the Z "failure" in C2 (306 and 407 transmissions) and in C3 (same numbers) is very probably this artifact. The alarm that stopped two runs may never have been real.

**Plan C4:** the tool's `identity` command uses the fixed definition. Its self test plants one differing transmission in a synthetic arm and confirms the check finds that one and no other.

### Issue 3. The arm C "pass" contradicts the report's own section 9

**Written:** section 10b: C against N, FFD3, 2,182 compared, 0 differ. Section 9: C against N, FFD3, 2,174 common, 116 differ by more than 1 m, maximum 22,246 m.

**Why it is wrong:** by the executor's own definition all transmissions are silent, so both tables compare the same thing and must agree. They do not. Section 10b also compares 2,182 transmissions when arm N shares only 2,174 with arm C. One of the two comparisons read the wrong database or the wrong root. Section 9 used root A; the plan said root B.

**Plan C4:** one tool, one definition, one table. No second script.

### Issue 4. Mean Z in thousands of metres is a fill value, not a solver bug

**Written:** "Mean Z positions are in the thousands of meters ... This suggests a potential coordinate system issue or solver bug that needs investigation."

**Why it is wrong:** `jsats3d/jsats3d.py` (lines 1724, 1757, 1784, 1865 and following) writes `9999.` into X, Y, Z of every row that has no solution. The executor averaged all rows. The arithmetic confirms it from the report's own tables:

| Arm, tag, root | Rows | Solution rows | Failed share | Failed share x 9999 | Reported mean Z |
|---|---|---|---|---|---|
| N, FFD3, B | 60,942 | 27,324 | 55.2% | 5,516 | 5,514.9 |
| N, FC36, B | 105,773 | 48,786 | 53.9% | 5,387 | 5,386.8 |
| N, FFD3, A | 60,942 | 19,860 | 67.4% | 6,741 | 6,707.7 |

Section 8 of the full report (mean positions, mean shifts of 100 to 300 m between arms) is therefore meaningless and must not be passed on.

**Plan C4:** the tool uses only rows with comment `solution found`, through the project's own loader. Preflight prints the row census by comment, with the count of rows at 9000 or more, so the fill value is visible.

### Issue 5. "Without ground truth data"

**Written:** open question 4: "Without ground truth data, we can only assess coverage ... Is there a way to validate accuracy?"

**Why it is wrong:** the forebay holds have surveyed positions and the drags have a GPS track. Scoring against them is the task. `scripts/drag_test_figures_2025.py` does it for one database.

**Plan C4:** the tool imports that module and scores every hold and drag in all four arms.

### Issue 6. Clock knots called fish detections

**Written:** "ZOI11 detections of fish tags (FFD3, FC36): 4,395 clock-fixed rows."

**Why it is wrong:** `tblDetectionClockFixed` holds the master beacon knots used for the clock correction. It has no fish detections. How often ZOI11 and CFD01 heard the fish tags has not been reported anywhere yet. That number matters most of all: if a receiver did not hear the tag during an event, it cannot change that event.

**Plan C4:** the `events` command prints, per event and tag, the detections at ZOI11 and at CFD01 from `tblDetectionFilterSecondary`.

### Issue 7. The clock section contradicts its own note

**Written:** "82.4% of rows over the 0.5 ms budget", and two lines later "`seconds_residual` is the clock offset the fix removes ... NOT leftover error".

**Why it is wrong:** the second line is right and makes the first meaningless. The table also has a broken row (CFD03 lost its first cell).

**Plan C4:** the clock table is not part of this deliverable. It is left out. The Stage 1 work covers clocks.

### Issue 8. Steps skipped and steps added

- The owner's choice A or B was not asked for or recorded. Task 3 (event list, sample plots, plots) and Task 4 did not run.
- Arms Z and ZC were rebuilt, which the plan did not ask for. The journal says "User confirmed goal". The owner's words are not quoted. Nothing was deleted, and the rebuild turned out useful (it confirmed the workaround), so no harm came of it.
- Section 16 "Rules observed" states "Each task ended with STOP". The journal shows Tasks 1 and 2, the rebuild and the full report in one pass.
- The same section says banned words were avoided. Open question 4 uses "improve" and the objective uses "improves or declines".

### Issue 9. Deep receivers at configured positions in every arm

**Written:** "The Deng solver found insufficient receivers to compute their positions in all four arms."

**Status:** reported, not explained, and the reviewer cannot tell from the code alone whether this is normal for a four day window. It affects all four arms alike, so it does not bias the comparison. It is listed as an open question for the owner in Plan C4. It is not investigated there.

## 3. Reviewer's own errors in this thread

1. Plan C's identity check assumed arms could differ only through the added receiver's rows. The reviewer then proposed the deep receiver path as the cause of the failure. Task 1 refuted that.
2. The reviewer accepted the C2 "failure" as a real observation without asking how "silent" had been decided. The C2 review noted the definition was missing (item 4 of its section 4) but still built a hypothesis on the result.
3. The first Plan C review read `seconds_residual` as leftover error. Withdrawn in the C2 review.
4. Plans C, C2 and C3 asked the executor to write analysis code. Three runs showed that this does not work for this task.

## 4. What Plan C4 changes

| Before | Now |
|---|---|
| Executor writes scripts from a prose specification | Executor runs `docs/handoff_tools/zoi11_cfd01_four_arm.py`, which has a self test with 23 checks |
| Definitions in the plan, reinterpreted by the executor | Definitions inside the tool, printed at the top of every output |
| Report written by the executor | Report is the tool's markdown, copied with `Copy-Item`, plus a short cover file in a fixed form |
| What to return described per task | `docs/llm_handoff_REPORT_CONTRACT.md` lists what every report must contain, and Plan C4 lists the files |
