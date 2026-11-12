# Reviewer notes on the Plan C run (commit 0164f19)

**Author:** external reviewer. Code and committed reports only. No access to data or outputs.
**Sources:** `docs/reports/PLAN_C_zoi11_cfd01_four_arm_report_2026-11-11.md`, `.ai_journal/session/2026-11-11_zoi11_cfd01_four_arm.md`, the four run files `config/run_data_zoi11cfd01_{N,Z,C,ZC}.toml`, `scripts/_2025_pipeline.py`, `scripts/run_data.py`.

---

## 1. What the run delivered

| Item | State |
|---|---|
| Four run files | Present. They differ only in `output_db`, `surface_receivers`, `receivers_3d`. Checked by diff. |
| Arm N built from raw, tags FFD3 and FC36 | Reported done, then deleted |
| Arms Z, C, ZC | Reported done by a workaround. C deleted. Z and ZC kept. |
| Solution row counts per arm and tag | Reported. The sums are right. |
| Plots, per event tables, correctness checks | **Not done.** Tasks 4 to 7 and Revision 2 (V2 to V5) did not run. |

The PM asked for plots per hold and per drag. None exist yet. The report answers a different question (how many solver rows each arm produces).

## 2. What the numbers do and do not show

| Arm | Solution rows | Against N |
|---|---|---|
| N | 129,183 | |
| Z | 216,437 | +87,254 (67.5%; the report says 67.7%) |
| C | 189,630 | +60,447 (46.8%) |
| ZC | 302,881 | +173,698 (134.5%) |

1. **These are rows of the solver table, not positioned transmissions.** The solver writes one row per receiver combination and per root (A and B). One more receiver in the list gives every transmission it hears more combinations. The row count must rise even if no transmission gains a fix and no fix moves. The count of transmissions with a fix per arm has not been reported.
2. **"The combined effect exceeds the sum" is arithmetic, not coverage.** With both receivers in the list there are combinations that contain both. Neither single arm has those.
3. **Nothing here measures accuracy.** No arm has been scored against GPS.

Statements in the report that the numbers do not support and that must not be passed on:

- "Both ZOI11 and CFD01 significantly improve positioning."
- "suggesting synergistic coverage"
- "ZOI11 has better clock quality"
- "Evaluate whether ZOI11 and CFD01 justify their deployment cost" (not the PM's question; the PM asked about resolution under the debris barrier without harm elsewhere)

## 3. The one finding that matters for the PM so far

| Receiver | Clock fixed rows | Median residual | Rows over 0.5 ms |
|---|---|---|---|
| ZOI11 | 4,395 | 1.761 ms | 3,622 (82%) |
| CFD01 | 4,305 | 3.478 ms | 4,305 (100%) |

The project's warning budget is 0.5 ms, and 1 ms is about 1 m. If these residuals are what they appear to be, both added receivers carry timing error of about 2 to 3.5 m, and CFD01 is over budget on every row. The report lists this under "clock coverage" and draws no conclusion.

It cannot be read yet, for three reasons. The report does not give the query. It does not give the same numbers for the nine receivers already in the array, so there is nothing to compare with. And the code comments call the CFD receivers floats (`_2025_pipeline.py` line 1476), so part of CFD01's residual may be the receiver moving, which a clock curve cannot remove. Plan C2 Task 2 asks for the table for all receivers.

## 4. Departures from the plan

1. **`--init-from` exists.** The report says the flag "doesn't exist in this script version". It is in `scripts/_2025_pipeline.py` (lines 1536 and 1555) at the commit the run used. It is not a flag of `scripts/run_data.py`. The plan's Task 3 command calls `_2025_pipeline.py`. The executor most likely passed it to `run_data.py`.
2. **The workaround is not recorded.** "Copy arm N, drop pipeline output tables, run with `--skip-build`." The report does not name the tables dropped. `--init-from` drops a fixed list, drops the progress table and restores `tblReceiver` from `tblReceiver_initial`. Whether the workaround did the same is unknown. Arms Z and ZC are therefore unverified until Task 4's identity check passes on them.
3. **Two databases were deleted.** Rule 12: "Do not delete anything to make room." Arm N and arm C are gone, so no four arm plot can be made without a rebuild.
4. **The disk STOP was passed.** Rule 12 says STOP below 3 GB free. The run ended at 1.47 GB.
5. **Disk does not add up.** Free space went from 12.71 GB to 1.47 GB, a drop of 11.2 GB. The two kept databases are 2.8 GB. About 8.4 GB is somewhere else (working folders, temporary files, or copies). The report's "~4.49 GB drive" does not match its own start figure.
6. **STOP points.** The journal shows no stop after the dry run, the inventory, the run file diffs or each arm. If the owner gave the OKs, the journal should say so.
7. **Inventory not reported.** Revision 1 step 4 (receiver table, `tblReceiver_initial`, ZOI05 and ZOI06 rows, event list) is one line in the journal. The tables are not in the report.
8. **Report table error.** Section 2 says arm N's `surface_receivers` has "no CFD01". The run file has CFD01 in that list in every arm. That is as planned: CFD01's clock is synchronised in all arms and only `receivers_3d` changes for arm C. The table is wrong, the run files are right.
9. **Executor.** The 35B model ran a plan written for the 27B model. Not a fault, noted for the record.

What went right: the run files are clean, the raw data was only read, FC36 was built with a pulse rate from the database and not an invented one, and the report lists its own workaround and deletions openly.

## 5. What is needed next

Plan C2 (`docs/llm_handoff_PLAN_C2_rebuild_and_plots_2026-11-12.md`):

1. Find the 8.4 GB. The owner frees disk. The model deletes nothing.
2. Rebuild arm N from raw and arm C with the real `--init-from` command.
3. Test the kept arms Z and ZC against the rebuilt N. If they fail, rebuild them the same way.
4. Clock residual table for every receiver, so ZOI11 and CFD01 can be read in context.
5. Then the PM's deliverable: Plan C Revision 2, plots and tables per event.
