# Plan C3: why the arms differ where the added receiver is silent, then the plots (2026-11-12)

**Author:** external reviewer (no access to data or outputs). **Executor:** the model the owner assigns.
**Read first:** `docs/reports/REVIEW_plan_c2_run_2026-11-12.md`, then `docs/llm_handoff_PLAN_C2_rebuild_and_plots_2026-11-12.md` section 1 (hard rules) and section 3 (traps). All of those rules apply. **If files disagree, this one wins.**

One task at a time. Every task ends with **STOP**: journal entry, report, then wait.

Journal: continue `.ai_journal\session\2026-11-12_zoi11_cfd01_c2.md`. Add entries at the end, each with the clock time. Do not reorder old entries.
Report: `docs\reports\PLAN_C3_report_2026-11-12.md`
Scripts you write go in `output\11112026_zoi11_cfd01\c3\` as `.py` files. Every database is opened read only: `sqlite3.connect("file:" + path + "?mode=ro", uri=True)`.

Tasks 1 and 2 build nothing and need almost no disk.

---

## 1. Rules added for this run

1. **Measure, do not explain.** If a number is unexpected, report it. Do not write what it "indicates" unless a sentence template below says so.
2. **Never state the result of a check you did not run.** Write "not run".
3. The correction to carry into the journal first: "`seconds_residual` in `tblDetectionClockFixed` is the clock offset the fix removes (the knots of the correction curve). It is not leftover error and is not compared with the 0.5 ms budget."

---

## 2. Tasks

### Task 0. State (read only)

Branch, `git log -1 --format="%h %s"`, `git status --short`, free disk on `C:` in GB. For each of `armN\arm_N.db`, `armC\arm_C.db`, `armZ\arm_Z.db`, `armZC\arm_ZC.db`: exists, bytes, last write. If any is missing, STOP.

Answer in one line: was disk freed between the Task 0 disk report (1.23 GB free) and the arm N build? If you do not know, write "not known".

**STOP.**

### Task 1. Do the receiver coordinates differ between arms?

**1a.** From each arm: `SELECT Rec_ID, X_t, Y_t, Z_t FROM tblReceiver ORDER BY Rec_ID`. One table: receiver down the side; for each of Z, C, ZC the XY distance in metres from arm N's position (`hypot(dX_t, dY_t)`) and the Z difference. Three decimals. Print all 20 receivers.

**1b.** From each arm: `SELECT * FROM tblDeepReceiver_step6_solution`. Print the four results one under the other.

**1c.** From each arm's status file (`arm_X.pipeline_status.json`, or whatever status JSON sits next to the database; list the files in each arm folder first), copy the notes of the deep receiver step and the adopt step: `found`, `epochs`, `X`, `Y`, `Z`, `shift_m` per deep receiver. If there is no such file, write "no status file".

**1d.** Does the added receiver hear the deep beacons? First get the three beacon tags: `SELECT Rec_ID, Tag_ID FROM tblReceiver WHERE Rec_ID IN ('ZOI01','ZOI02','ZOI03')`. Then in arm ZC: `SELECT Rec_ID, Tag_ID, COUNT(*) FROM tblDetectionFilterSecondary WHERE Tag_ID IN (<the three tags>) AND Rec_ID IN ('ZOI11','CFD01') GROUP BY Rec_ID, Tag_ID`. Print the result. Zero rows is a valid result.

Sentence. Pick one per arm (Z, C, ZC):
- "In arm X the deep receivers sit at the same coordinates as in arm N (largest difference D m)." Use when every difference is at most 0.001 m.
- "In arm X the deep receivers are displaced from their arm N positions: ZOI01 by A m, ZOI02 by B m, ZOI03 by C m. The added receiver is one of the references that locate them (`_2025_pipeline.py` lines 914 and 1485)."

**STOP. Report the tables and the three sentences.**

### Task 2. The identity check, done fully, for Z and for C

Write one script. Definitions, fixed:

- **Rows used:** `tblPositions_Deng` where `comment = 'solution found' AND solution = 'B'`, with `X`, `Y`, `Z` cast to float and finite.
- **Silent transmission** for arm X and its added receiver R (ZOI11 for Z, CFD01 for C): a `(Tag_ID, transNo)` in arm X where no row used has R in `r0, r1, r2, r3`. If those columns have other names, print the column list of `tblPositions_Deng` and say which you used.
- **Compared:** silent transmissions that also have rows in arm N.
- **Per transmission value:** the median X, median Y, median Z of the rows used, and the row count.
- **Difference:** XY distance between arm X's and arm N's medians.

For tags FFD3 and FC36, and for arm Z against N and arm C against N, report:

| Column | Meaning |
|---|---|
| compared | transmissions compared |
| same_rows | of those, how many have the same row count in both arms |
| over_1mm | difference above 0.001 m |
| over_1m | above 1 m |
| over_100m | above 100 m |
| median, p90, p99, max | of the difference, metres |
| uses_deep | of the transmissions over 1 mm, how many have ZOI01, ZOI02 or ZOI03 in some row's receiver columns |
| no_deep_over_1mm | of the transmissions with **no** deep receiver in any row, how many are over 1 mm |

Then for arm Z and tag FFD3, print the ten transmissions with the largest difference: `transNo`, row count in N and in Z, median X, Y, Z in each, and the distinct receiver sets in each.

Sentences. Pick one per arm and tag:
- "`no_deep_over_1mm` is 0. Every difference involves a deep receiver."
- "`no_deep_over_1mm` is K. Differences exist that do not involve a deep receiver."

Do not add an explanation.

**STOP. Report. The owner now chooses how to go on. Do not start Task 3 without the owner's choice quoted in the journal.**

Owner's choices, for the record:

| Choice | Meaning |
|---|---|
| A | Use the four arms as built. The report states that an arm includes the receiver in fish positions **and** in locating the deep receivers, as production would. |
| B | Stop. The arms need a different build that holds the deep receiver positions at arm N's values. That needs a new plan. |

### Task 3. Plots and per event numbers

Plan C2 Task 3, unchanged, including its two owner OKs (event list, first three plots).

Two additions:
1. On every plot, draw the deep receivers ZOI01, ZOI02, ZOI03 once per arm in the arm's colour (small triangles) if Task 1 found them displaced by more than 0.5 m. Otherwise draw them once in black.
2. Per event and tag, also record for each arm how many transmissions survive in `tblPositions_DengDBSCAN` against how many have a root B row in `tblPositions_Deng`. One column pair per arm in `event_scores.csv`.

**STOP after all plots.**

### Task 4. Tables and report

Plan C2 Task 4, with the report named `docs\reports\PLAN_C3_report_2026-11-12.md` and these changes to its sections:

- Section 3 becomes **Task 1 tables and sentences** (receiver coordinates across arms).
- Section 4 becomes **Task 2 table, the ten rows, and sentences**, followed by the owner's choice A or B, quoted.
- Section 5, the clock table, is copied from `PLAN_C2_task2_check_2026-11-12.md` with the column renamed "offset removed, ms" and this line under it: "This column is the clock offset the fix removes. It is not leftover error."
- Section 9, corrections, gains two lines:
  - "The identity check of Plan C assumed that only fish solutions containing the added receiver can change. The added receiver also takes part in locating the deep receivers. Task 1 gives the measured displacement."
  - "The statement that the solver geometry is fundamentally different is withdrawn. No cause was measured in that run."

If the owner chose A, the summary (V5 wording) starts with this line, filled in from Task 1: "Each arm includes its added receiver in fish positions and in locating the deep receivers. Against arm 1, the deep receivers moved by at most D m in arm 2, E m in arm 3 and F m in arm 4."

Final journal entry with `Get-Item` proof. Proposed commit list. Do not commit.

**STOP.**
