# Plan C2: finish the ZOI11 and CFD01 four arm evaluation (2026-11-12)

**Author:** external reviewer (no access to data or outputs). **Executor:** the model the owner assigns.
**Read first, in this order:**
1. `docs/reports/REVIEW_plan_c_run_2026-11-11.md` (what the first run got wrong)
2. `docs/pm_request_zoi11_cfd01_evaluation_2026-11-11.md` (what the PM wants)
3. `docs/llm_handoff_PLAN_C_zoi11_cfd01_2026-11-10.md`: Revision 2 (V1 to V6), section 5 (rules), section 7 (wording), section 8 (traps), Tasks 4 to 7

All rules of Plan C apply. **If files disagree, this one wins.**
One task at a time. Every task ends with **STOP**. A STOP means: write the journal entry, report, and do nothing more until the owner answers. "The owner's OK" must be quoted in the journal with its time.

Output folder (same as before): `C:\Dev\jsat3d\jsats3d\output\11112026_zoi11_cfd01\`
Journal: `C:\Dev\jsat3d\jsats3d\.ai_journal\session\2026-11-12_zoi11_cfd01_c2.md`
Report: `docs\reports\PLAN_C2_report_2026-11-12.md`

---

## 1. Hard rules for this run

1. **You delete nothing.** No `Remove-Item`, no `del`, no overwrite of a database. If disk is short, STOP. Only the owner frees space.
2. **Disk floor is 3 GB free.** Check before every build. Below the floor: STOP.
3. **`--init-from` is a flag of `scripts\_2025_pipeline.py`. It is not a flag of `scripts\run_data.py`.** Use the commands below exactly. If a command fails, paste the full error and STOP. Do not invent another route.
4. **Do not edit the four run files.** They are committed and correct.
5. **A count of solution rows is not a result.** Never write that an arm "improves positioning" from row counts. Banned words in the report: improve, significant, synergistic, better, worse, justify, validated, proven. Write the numbers.
6. No commit. No push. Raw data is read only.

---

## 2. Tasks

### Task 0. Setup and disk (read only)

1. `Set-Location C:\Dev\jsat3d\jsats3d`; branch check; `git log -1 --format="%h %s"`; `git status --short`; test suite summary line.
2. Free disk on `C:` in GB, two decimals.
3. Size of every folder directly under `output\`, largest first:
   `Get-ChildItem output -Directory | ForEach-Object { [pscustomobject]@{Folder=$_.Name; GB=[math]::Round(((Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum/1GB),2)} } | Sort-Object GB -Descending | Format-Table -AutoSize`
4. The same one level deeper for `output\11112026_zoi11_cfd01\`, and a list of every file there over 100 MB (path, bytes, last write).
5. Also report any file over 500 MB last written on 2026-11-11 anywhere under `C:\Dev\jsat3d` and under `$env:TEMP`.
6. From your memory of the first run, or from `run_log.txt` files if they exist, write down: the exact commands used for the workaround, and the exact names of the tables that were dropped. If this is not recorded anywhere, write "not recorded". Do not guess.

Disk needed for this plan: arm N about 1.4 GB, arm C about 1.4 GB, working space about 2 GB, floor 3 GB. **Free space must be at least 8 GB before Task 1.**

**STOP. Report the tables. If free space is under 8 GB, say so and name the largest items from steps 3 to 5. The owner decides what to remove and removes it.**

### Task 1. Rebuild arm N and arm C

Check first: `Test-Path output\11112026_zoi11_cfd01\armN\arm_N.db` and the same for `armC\arm_C.db`. Both must be False. If either is True, STOP.

**1a. Arm N from raw.**

```
python.exe scripts\run_data.py config\run_data_zoi11cfd01_N.toml --dry-run
```

Paste the whole output. **STOP. Wait for the owner's OK.**

After the OK:

```
python.exe scripts\run_data.py config\run_data_zoi11cfd01_N.toml *>&1 | Tee-Object output\11112026_zoi11_cfd01\armN\run_log.txt
```

If it stops with an error, paste the last 40 lines and STOP.

**1b. Inventory on arm N (read only).** This was skipped last time. Report as tables:
- `SELECT name FROM sqlite_master WHERE type='table' ORDER BY name` with the row count of each table.
- `tblReceiver` rows for ZOI11, CFD01 and the nine receivers of arm N's `receivers_3d`: `Rec_ID, Tag_ID, X, Y, Z, X_t, Y_t, Z_t`.
- Does `tblReceiver_initial` exist? If not, STOP: `--init-from` needs it.
- ZOI05 and ZOI06 rows of `tblReceiver` next to the same rows of `tblReceiver_initial`. If they differ, STOP.
- `SELECT Tag_ID, TagType, pulseRate FROM tblTag WHERE Tag_ID IN ('FFD3','FC36')`.
- Solution rows per tag and root, in the same layout as section 3 of the first report. Expected from that report: FFD3 A 19,860, B 27,324; FC36 A 33,213, B 48,786. Say for each whether it matches. A mismatch is not a STOP. Report it under "Things that looked wrong".

**STOP. Report.**

**1c. Arm C by the planned route.** Free disk check first.

```
python.exe scripts\_2025_pipeline.py config\run_data_zoi11cfd01_C.toml ^
  --db output\11112026_zoi11_cfd01\armC\arm_C.db ^
  --init-from output\11112026_zoi11_cfd01\armN\arm_N.db --no-drag-plots *>&1 | Tee-Object output\11112026_zoi11_cfd01\armC\run_log.txt
```

In PowerShell write the command on one line (the `^` is only for reading here).

Then report: size and last write of `arm_N.db` before and after (unchanged); the table list with row counts for arm C; solution rows per tag and root next to the first report's arm C figures (26,383 / 35,777 / 52,717 / 74,753).

Sentence to use, pick one:
- "Arm C built with `--init-from` gives the same solution row counts as the first run's workaround."
- "Arm C built with `--init-from` gives different counts from the first run's workaround: (numbers). The workaround was not equivalent."

**STOP.**

### Task 2. Are the kept arms Z and ZC usable? Clock table.

**2a. Table census.** For all four databases, one table: table name down the side, row count per arm across. Mark every table that exists in some arms and not in others.

**2b. Plan C Task 4, all four checks,** on N and C (new) and Z and ZC (kept from the first run). The two that decide:
- Check 2, receiver membership: N has neither receiver in any solution row; Z has ZOI11 only; C has CFD01 only; ZC has both.
- Check 3, identity where the receiver is silent: for events where ZOI11 has zero detections of the tag, arm Z per transmission medians equal arm N to 0.000 m. The same for CFD01 and arm C.

| Result | Action |
|---|---|
| All checks pass for Z and ZC | Keep them. Go on. |
| Any check fails for Z or ZC | Do not use that database. STOP and report. The owner will free disk; then build `armZ\arm_Z_r2.db` or `armZC\arm_ZC_r2.db` with the Task 1c command and that arm's run file. Never overwrite the old file. |
| Check 3 fails for arm C | STOP. Something other than the receiver list differs. |

**2c. Clock residuals for every receiver, from arm ZC.** First print the column names of `tblDetectionClockFixed` and write the exact query you use. Then one row per receiver in that table: rows; median of the absolute residual in ms; 90th percentile in ms; rows over 0.5 ms; that share in percent. Sort by median, largest first. Mark ZOI11 and CFD01.

Then the same table from arm N for the receivers it has, to show the shared receivers are unchanged between arms.

Sentences. Fill in and use only these:
- "ZOI11's median clock residual is A ms. The nine receivers of the base array range from B to C ms."
- "CFD01's median clock residual is A ms. P% of its rows are over the 0.5 ms budget. The other CFD receivers range from B to C ms."
- "1 ms is about 1 m. This table is timing only. It is not a position error."

Do not explain why a residual is large.

**STOP.**

### Task 3. Plots and per event numbers (the PM's deliverable)

Do Plan C Task 5 as changed by Revision 2: events by V2, arm names and colours by V1, two plots per event and tag by V3, `index.html`.

Additions for this run:

1. Before plotting, print the event list: Test ID, type (forebay hold, debris barrier hold, drag), date, start, end, GPS truth yes or no, inside the built window yes or no. **STOP. Wait for the owner's OK on the list.**
2. Tags: FFD3 and FC36. An event where a tag has no fixes in any arm still gets a row in the tables with zeros. It gets no plot. List those events.
3. Count fixes as **transmissions with a fix**, never as solution rows. The legend `n=` is transmissions.
4. Make the first three plots only (one forebay hold, one debris barrier hold, one drag). Open them. Write one line each on what you see and anything that looks wrong. **STOP. Wait for the owner's OK on the look.** Then make the rest.

**STOP after all plots. Report file counts for `plots_2d\` and `plots_3d\`, the `event_scores.csv` row count, and the one line notes.**

### Task 4. Tables and report

Plan C Task 6 as changed by V4 (Tables V4a to V4d, and old tables 2 to 5), then Plan C Task 7 with V5 wording.

The report `docs\reports\PLAN_C2_report_2026-11-12.md` has these sections, in this order:

1. **Run facts.** Head, branch, test line, free disk at start and end, every database with path, bytes, last write.
2. **Task 0 disk tables** and the workaround record (or "not recorded").
3. **Arm N inventory** (Task 1b) and the arm C count comparison sentence (Task 1c).
4. **Table census and the four checks** (Task 2a, 2b), each check with its value.
5. **Clock table and its three sentences** (Task 2c).
6. **Event list.**
7. **Tables V4a to V4d.**
8. **Summary by V5.** Three short paragraphs, then: "Include or exclude is the PM's decision."
9. **Corrections to the first report.** Copy these four lines:
   - Solution row counts rise with every receiver added to the list, because each transmission gets more receiver combinations. They are not a count of positioned transmissions.
   - The first report's statements "significantly improve positioning", "synergistic coverage" and "better clock quality" are withdrawn.
   - `--init-from` exists in `scripts\_2025_pipeline.py`. The first run's workaround was not needed.
   - CFD01 is in `surface_receivers` in all four arms. Only `receivers_3d` changes for arm C.
10. **Deviations.** "none" if none.
11. **Things that looked wrong.**
12. **Open questions.**

Final journal entry with `Get-Item` proof for the report, `index.html`, and both plot folders. Proposed commit list (docs, journal, nothing under `output\`). Do not commit.

**STOP.**

---

## 3. Known traps

- `--init-from` refuses an existing target. That is the safety. Never delete a target to rerun. Use a new file name ending `_r2` and report it.
- `arm_N.db` is the source for arm C and for any `_r2` build. Record its size and last write before and after each use. It must not change.
- The run files point `output_db` at the original paths. `--db` on `_2025_pipeline.py` sets the database for that run. Do not edit the run files to match.
- Debris barrier holds have no GPS. Their plots and rows carry spread and depth only.
- The time basis is PDT stored as if UTC. Label times "local time (PDT)". Never "UTC".
- Each Deng job needs several GB of memory. One build at a time.
