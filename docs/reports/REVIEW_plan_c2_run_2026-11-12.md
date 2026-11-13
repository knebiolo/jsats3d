# Reviewer notes on the Plan C2 partial run (commit d896dbf)

**Author:** external reviewer. Code and committed reports only. No access to data or outputs.
**Sources:** `docs/reports/PLAN_C2_task0_disk_2026-11-12.md`, `docs/reports/PLAN_C2_task2_check_2026-11-12.md`, `.ai_journal/session/2026-11-12_zoi11_cfd01_c2.md`, `scripts/_2025_pipeline.py`, `scripts/legacy_pipeline.py`, `jsats3d/jsats3d.py`.

---

## 1. What is now established

| Fact | Source |
|---|---|
| Arm N rebuilt from raw. Solution row counts equal the first run. | journal Task 1a |
| Arm C built with `--init-from`. Solution row counts equal the first run's workaround. | journal Task 1c |
| The workaround dropped nine named tables, the progress table among them. | task0 report |
| All four databases hold the same 17 tables. Shared tables have equal row counts. | task2 report 2a |
| Receiver membership is right in all four arms. | task2 report, check 2 |
| The identity check fails for arm Z: 306 of 2,174 FFD3 and 407 of 2,417 FC36 transmissions differ from arm N by more than 1 mm where ZOI11 is silent. | task2 report, check 3 |

The executor stopped at the failed check, as the plan says. That was right.

**The workaround is very probably equivalent to `--init-from`.** Arm C gave identical counts both ways, and the table census is the same in every arm. The failed check is therefore unlikely to be a build fault of the kept arms.

## 2. Reviewer's error: the identity check rested on a wrong premise

Plan C said: where the added receiver hears nothing, the arm must equal arm N, otherwise "the arms differ in something other than the receiver list".

The code shows a path by which the receiver list itself changes other things:

1. Step 6 (`deep_positions_run`, `_2025_pipeline.py` line 914) finds the position of each deep receiver (ZOI01, ZOI02, ZOI03) from its own beacon. The receivers it uses for that are `c.reference`: every receiver in `receivers_3d` that is not deep (line 1485).
2. In arm Z that reference set contains ZOI11. In arm C it contains CFD01.
3. Step 6b (`adopt_deep_run`, line 968) writes the solved X and Y of the deep receivers into `tblReceiver` and then recomputes the deep clocks, the fish filter and the positions.

So if ZOI11 hears a deep beacon, the deep receivers sit at slightly different coordinates in arm Z than in arm N. Every fish solution that uses a deep receiver then moves, including for transmissions ZOI11 never heard. Poorly conditioned solutions amplify a small receiver shift into a very large one, which would account for the maximum of tens of kilometres.

This is a hypothesis from the code. It is not confirmed. One query confirms or refutes it: compare `X_t, Y_t, Z_t` of the deep receivers across the four arms (Plan C3 Task 1).

A second path exists and is smaller: the optional position screen (`positions_dbscan_run`, line 1345) clusters all transmissions of a tag together and uses acceleration between neighbours. A transmission can be kept in one arm and dropped in another because its neighbours changed. This affects which transmissions survive, not their coordinates.

**What this means for the comparison.** If the hypothesis holds, the arms differ in exactly what production would differ in when a receiver is added: the receiver takes part in fish positions and in locating the deep receivers. That is a fair test of "include or exclude". It is not a test of the receiver's contribution to fish positions alone. The report must say which one it is.

## 3. Reviewer's error: the clock table was misread in the last review

Section 3 of `REVIEW_plan_c_run_2026-11-11.md` treated `seconds_residual` as leftover timing error and compared it with the 0.5 ms budget. The code comment at `jsats3d/jsats3d.py` line 322 describes `tblDetectionClockFixed` as the "piecewise (seconds, seconds_residual) per receiver" curve, and the solver reads those rows as the knots of the correction. So the column is the clock offset that the fix removes, not what is left after it. **That section is withdrawn.**

Read correctly, the Task 2c table says:

| Reading | Numbers |
|---|---|
| ZOI11 and CFD01 offsets are ordinary for this array | medians 1.761 and 3.478 ms; the other receivers range from 0.126 ms (ZOI07) to 8.625 ms (ZOI05), ZOI09 aside |
| ZOI09 sits almost one second from the master for most of the window | median 943.8 ms, p95 975.9 ms |
| Several receivers have a tail near 0.4 s or 0.9 s | p95 of 394 to 405 ms on CFD03, CFD06, CFD07, CFD08, ZOI02, ZOI06; 880 and 918 ms on ZOI11 and CFD01 |

The last two rows are the clock jump families of the Stage 1 work, seen in the June test window, with `apply_clock_jumps = false` in all four run files. Whether the correction curve follows those excursions or smooths across them is not known from this table. ZOI09 is in `receivers_3d` in every arm, so this affects all four arms alike.

The executor's sentences follow the template and are correct as written. The executor did not print the column names or the query, and reported p95 where the plan asked for p90. Minor.

## 4. Executor's errors and gaps

1. "The massive differences ... indicate the solver geometry is fundamentally different." Not shown. No cause was measured.
2. "Arm ZC would fail the same check." Not run. Stated as fact.
3. **Check 3 was not run for arm C.** That was the most useful comparison available: arm C was built by the planned route minutes earlier. A pass would point at the workaround, a fail at the pipeline.
4. Check 3 was done per transmission. The plan said per event. The per transmission form is the stricter and better one, but the report does not say how "ZOI11 silent" was decided or which rows the medians were taken over.
5. The journal is out of order: the disk STOP is written after Tasks 1 and 2. The disk report shows 1.23 GB free and no arm N database; the journal shows arm N (1.36 GB) built afterwards. Either disk was freed in between or the figures are wrong. The journal does not say.
6. The 8.4 GB gap from the first run was not found. The first run's start figure of 12.71 GB free may simply have been wrong.
7. The journal gives the kept databases as "1,468 MB" and "1,514 MB". Those are the byte counts read as MB. The real sizes are 1,400 and 1,444 MB.

What went right: every STOP that mattered was honoured, nothing was deleted, the build commands are quoted, the workaround is now on record, and the sentences use the given templates.

## 5. Next

Plan C3 (`docs/llm_handoff_PLAN_C3_identity_cause_and_plots_2026-11-12.md`):

1. Three read only comparisons to find why the arms differ where the added receiver is silent.
2. An owner decision on what the comparison is called.
3. Then the PM's plots.
