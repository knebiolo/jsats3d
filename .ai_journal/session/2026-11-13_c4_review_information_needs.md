# Plan C4 read-back: where the ZOI11/CFD01 evaluation stands, and what I need to advise the next step

**Written:** 2026-11-12 (evening), filed with the C4 series, which is dated 2026-11-13.
**Author:** external reviewer (Claude). Code and committed files only. No access to `output\`, the databases, the `K:` drive, a plot, or the Windows machine.
**Read for this entry:** commit `2f1ca5e` (all eight files), `docs/llm_handoff_PLAN_C4_four_arm_tool_2026-11-13.md`, `docs/llm_handoff_REPORT_CONTRACT.md`, `docs/pm_request_zoi11_cfd01_evaluation_2026-11-11.md`, `docs/reports/REVIEW_plan_c3_run_2026-11-12.md`, `docs/handoff_tools/zoi11_cfd01_four_arm.py`, `scripts/drag_test_figures_2025.py`, `jsats3d/jsats3d.py` (lines 340 to 480), the four `config/run_data_zoi11cfd01_*.toml` files, `.ai_journal/session/2026-11-01_zoi11_debris_barrier_ab.md`.
**Status of this file:** not committed. Nothing else was changed.

Statements about cause carry `[measured: source]` or `[not measured]`, as the report contract asks of the executor.

---

## 1. What we are trying to accomplish

The PM has one decision to make: **include or exclude ZOI11 and CFD01 in the production 3D receiver set.** The PM's stated test is narrow: a receiver earns its place only if it "increases positioning resolution underneath the debris barrier without degrading positions in other locations."

The products the PM asked for are qualitative: for every forebay hold, debris barrier hold and validation drag, one 2D plan plot and one interactive 3D plot showing four arms together (1. Excluded, 2. ZOI11 Only, 3. CFD01 Only, 4. Both), with GPS truth and receivers, plus per event and arm the fix count, the spread, any visible degradation, and whether fixes appear where the baseline has none.

This sits beside the clock work (Plans B, B2, B3). That thread is paused on a PM and Kevin discussion about whether clock jump corrections are needed at all. It touches this evaluation in one place only: all four arms ran with `apply_clock_jumps = false`.

Plans C, C2 and C3 each ended without a plot because the executor wrote its own analysis and redefined terms. Plan C4 replaced that with a tested tool the executor only runs.

## 2. Where things stand after commit 2f1ca5e

**C4 ran end to end. For the first time the PM products exist:** 46 PNG and 46 HTML (23 scored events x 2 tags), `event_scores.csv` (184 rows), `fixes.csv`, and the tables V4a to V4d. They are on the Windows machine under `output\11112026_zoi11_cfd01\c4\`. None of the plots or CSV files are in the repository.

| Item | Result | Source |
|---|---|---|
| Self test | 23 OK, 0 FAIL | C4 report section 4 |
| Tool hash, repository copy against run copy | equal | C4 report section 5 |
| Preflight: tables, membership, receiver coordinates, local origin | PASS, four of five | preflight section 9 |
| Preflight: transmission numbers aligned | **FAIL**: Z and ZC differ from N on 312 FC36 and 297 FFD3 transmissions, up to 2.41 s and 2.76 s. C differs on 0. | preflight section 7 |
| Identity | HOLDS, 0 of 3,229 compared transmission and arm pairs over 1 mm, every one with equal row counts | identity sections 2 to 4 |
| Events | 23 scored (9 forebay holds, 5 debris barrier holds, 9 drags). Not scored: FBY-S-6 (no times), four drags dated 2025-06-05 | events |
| Databases | byte sizes and write times unchanged | C4 report section 2 |

**The run went past a STOP.** The plan says a failed alignment check means "Arms cannot be matched on transmission number" and STOP. The owner authorised "proceed Path A". Tables V4b and V4d match arms on transmission number, so their standing depends on section 4.1 below.

**Not done by the executor:** the 3D HTML files were never opened; the five extra PNG files of Task 5 were not opened; the identity columns were not copied into cover section 8; STOP entries carry no clock times and the owner's replies are not quoted in full; "Path A" is not defined anywhere in the committed files.

## 3. What the committed numbers show

These are my recomputations from `PLAN_C4_tables_2026-11-13.md`. Each can be checked against the table named.

### 3.1 Debris barrier holds, the PM's target (V4a, V4b)

Fixes per hold and tag, arms N / Z / C / ZC:

| Hold | FC36 | FFD3 |
|---|---|---|
| DB-1 | 18 / 19 / 19 / 19 | 18 / 18 / 18 / 18 |
| DB-2 | 21 / 21 / 21 / 21 | 13 / 13 / 13 / 13 |
| DB-3 | 12 / 21 / 19 / 21 | 19 / 19 / 19 / 19 |
| DB-4 | 18 / 19 / 18 / 19 | 19 / 19 / 19 / 19 |
| DB-5 | 0 / 6 / 6 / 10 | 13 / 17 / 14 / 16 |
| **Sum of both tags** | | **151 / 172 / 166 / 175** |

About 210 transmissions fall in those ten windows (events table, `trans_` columns), so the baseline already fixes roughly 72% and the best arm roughly 83%.

- The gain at the barrier is concentrated in two hold and tag pairs: DB-3 FC36 (12 to 21) and DB-5 FC36 (0 to 6, 6, 10). Seven of ten pairs are at their ceiling in the baseline and cannot gain.
- DB-2 FFD3 stays at 13 of 20 in every arm although ZOI11 logged 82 detections in that window. Something other than receiver count limits that pair.
- Spread at the barrier narrows with ZOI11 in most pairs (DB-1 FC36 std_x 2.45 to 1.23 m, DB-4 FC36 std_x 2.00 to 0.29 m, DB-3 FFD3 std_y 2.65 to 0.75 m). With CFD01 alone it widens in three: DB-4 FFD3 std_y 3.91 to 6.81 m, DB-5 FFD3 std_y 7.70 to 19.15 m, DB-3 FC36 std_z 1.95 to 4.26 m.
- **Depth moves by metres between arms with no truth to judge it.** DB-1 FC36 median_z: -6.99, -9.27, -10.43, -12.96 (N, Z, C, ZC). DB-1 FFD3: -6.74, -7.86, -8.50, -8.89. DB-4 FFD3: -1.49, -5.78, -6.16, -6.06. DB-5 FFD3: -12.79, -3.11, -6.26, -6.90.
- **The DB-5 FC36 gained fixes do not agree with each other across arms.** V4c puts the 6 gained in arm Z under 25 m from the nearer receiver, 5 of the 6 gained in arm C over 50 m, and all 10 gained in arm ZC at 25 to 50 m. The tag did not move. Median z is -3.95, +3.03 and -1.58. At least two of those three sets are in the wrong place.

### 3.2 "Without degrading positions in other locations" (V4d)

The summary sentences count pairs where the median error "rose by more than 0.5 m": 2 for ZOI11, 4 for CFD01, 2 for Both. Those counts use the unpaired median, which changes when an arm adds new fixes. The paired column answers the PM's question directly:

| Pair, arm | Unpaired change | Paired change | Fixes N to arm |
|---|---|---|---|
| FBY-S-9 FC36, ZOI11 | +3.04 m | +0.09 m | 4 to 11 |
| FBY-S-9 FC36, CFD01 | +2.11 m | 0.00 m | 4 to 5 |
| FBY-S-1 FC36, ZOI11 | +1.25 m | +0.18 m | 21 to 21 |
| FBY-D-1 FC36, CFD01 | +0.89 m | 0.00 m | 62 to 63 |
| FBY-D-4 FFD3, CFD01 | +0.69 m | 0.00 m | 18 to 25 |
| FBY-S-1 FC36, CFD01 | +0.68 m | 0.00 m | 21 to 21 |

**No pair in V4d has a paired median change above +0.34 m** (the largest is FBY-D-4 FFD3, Both). So the transmissions that already had a fix kept about the same error. Where the unpaired median rose, the new fixes are less accurate than the old ones (FBY-S-9 FC36 p90 goes from 8.05 to 24.78 m with ZOI11).

That is a more useful statement for the PM than the summary sentence, with two limits: a paired median hides the tail, and FBY-S-1 FC36 has all 21 fixes paired and still moves 1.25 m in the unpaired median, which means some of those 21 moved a lot.

Tails the summary sentences do not report (V4a, p90_err): ENT-05 FC36 with CFD01 8.37 to 11.50 m; ENT-03 FC36 with CFD01 12.18 to 15.55 m; FBY-D-4 FFD3 9.07 to 12.83 m (ZOI11) and 14.75 m (CFD01).

### 3.3 Fixes lost (V4b)

Adding a receiver removed 66 (ZOI11), 51 (CFD01) and 64 (Both) fixes that the baseline had. Largest: FBY-D-4 FC36 20 lost with ZOI11; FBY-D-1 FC36 14 lost with ZOI11, a net drop from 62 to 56 fixes. The tables do not say whether the lost fixes were accurate ones or poor ones.

### 3.4 Where the gained fixes are (V4c)

Of 311 fixes gained with ZOI11, 21 are in the debris barrier holds and 290 in forebay holds and drags. "187 over 50 m" in the summary sentence is driven by drag duration: the five barrier holds are five minutes of data, the four FBY-D drags are 47 minutes. The band totals describe where the boat spent time, not where the receiver adds coverage per unit of opportunity.

### 3.5 Built-in controls that came out right

Where the added receiver logged 0 or 1 detections, the arm equals the baseline exactly (FBY-S-3 FFD3 all arms, FBY-S-4 FFD3 all arms, FBY-S-7 FFD3 with ZOI11, FBY-S-5 FFD3 with CFD01). Together with the identity result this says the four databases differ only through rows that contain the added receiver. `[measured: V4a rows, identity sections 2 to 4]`

### 3.6 Baseline errors that are large before any receiver is added

FBY-S-10: 33.11 m (FC36) and 58.62 m (FFD3). FBY-D-1 FC36: 23.60 m. FBY-D-4 FC36: 19.70 m. FBY-S-8 FFD3: 24.31 m on 3 fixes. The biggest "error fell" entries in V4d are these same events. Median z values above the water surface appear in several fix sets (FBY-S-5 FC36 +14.05 m, FBY-S-2 FC36 +3.56 m, FBY-D-1 FFD3 +14.78 m with ZOI11, -54.48 m in the baseline). A fix as the tool defines it has no depth screen.

## 4. What I cannot conclude yet

### 4.1 The failed alignment check: probably a property of the check, not of the arms

My reading of the code:

- Study tag transmission numbers are `round((seconds_fix - first detection) / pulseRate)`, computed over the detections of **every** receiver (`jsats3d/jsats3d.py:412-415`).
- A receiver with no clock curve keeps its raw time: "left uncorrected" (`jsats3d/jsats3d.py:373-378`).
- ZOI11 is in `surface_receivers` only in arms Z and ZC (config diff). CFD01 is in `surface_receivers` in all four.
- The alignment check compares `dm.first`, the minimum `seconds_fix` per transmission number over **all** receivers in `tblDetectionFilterSecondary`, ZOI11 included (`scripts/drag_test_figures_2025.py:263-266`).

So in arms N and C the ZOI11 detections sit on ZOI11's uncorrected clock, and wherever ZOI11's raw time is the earliest in a transmission number the "first arrival" differs from arms Z and ZC. `[not measured]`

Three committed numbers fit that reading:

1. C against N differs on 0 transmissions; Z and ZC differ on the same 312 and 297 with the same maxima. `[measured: preflight section 7]`
2. `tblDetectionClockFixed` has 160,649 rows in N and C and 165,044 in Z and ZC: 4,395 ZOI11 clock knots exist only in Z and ZC. `[measured: preflight section 2]`
3. `det_ZOI11` inside a fixed event window differs between the two groups and agrees inside each: FBY-S-8 FC36 32, 31, 32, 31 and DB-1 FC36 105, 102, 105, 102 for N, Z, C, ZC. The same detections carry different times in the two groups. `[measured: V4a]`

If that is right, the transmission numbers of the nine baseline receivers mean the same thing in all arms and V4b and V4d stand. The identity result supports it: 3,229 exact matches with equal row counts would not happen if numbering had shifted, and 37 + 38 + 15 + 42 of the compared transmissions in Z and ZC were heard by ZOI11.

What stays open:

- The identity check compares only transmissions with no added-receiver row. The 609 flagged transmissions are, by construction, mostly the ones ZOI11 heard, so the two sets overlap little. The match has not been tested on the flagged set itself.
- `first detection` in the numbering formula is the global minimum over all receivers. If that single detection came from ZOI11, the numbering origin differs between the two groups by ZOI11's clock offset. `common_trans` is 2,494 in Z against 2,496 in C, so the number sets are not perfectly equal.
- `dm.first` also decides which transmissions belong to a drag (first arrival inside the GPS window plus or minus 3 s, `drag_test_figures_2025.py:288-295`). In arms N and C that boundary is drawn with the uncorrected ZOI11 times. A few gained or lost fixes at the start and end of each drag may come from that and not from positioning.
- Hold windows and drag errors use `t0 = ToA - T01` from the solver rows, not `dm.first`, so they are not touched. `[measured: drag_test_figures_2025.py:282, 355, 365; four_arm.py:309]`

### 4.2 Whether the gained fixes are good fixes

The PM criterion is resolution under the barrier. The barrier holds have no GPS truth, so every fix gained there is unscored in XY. DB-5 FC36 (section 3.1) shows gained fixes that disagree by tens of metres between arms. For the events with truth, the tool reports the error of the whole arm, never the error of the gained fixes alone.

### 4.3 Depth

The PM asked for "plan view with elevation". The tool reports `median_z` and `std_z` and no depth error, although deployed tag depths exist (`ent_analysis_2025.py:29`, FC36 -1.524 m and FFD3 -4.572 m for the drags; the 2026-11-01 journal gives 20 ft, 6.1 m, for FFD3 at the barrier holds). ZOI11 and CFD01 both hang at -3.048 m, in the same plane as six of the nine baseline receivers, so there is no geometric reason to expect them to help depth. DB-1 depth moves away from 6.1 m with every receiver added (FFD3 -6.74, -7.86, -8.50, -8.89), while DB-4 and DB-5 FFD3 move toward it. I cannot rank the arms on depth without knowing the true depth per event and tag.

### 4.4 Whether "fix" here is what production would deliver

The tool's fix is the receiver-set consensus of the evaluation scripts. The production pipeline carries `tblPositions_DengDBSCAN` forward (`positions_dbscan = true`). On that table the order of arms is different: FC36 transmissions after the screen are 1,371, 1,559, 1,426, 1,520 for N, Z, C, ZC. Adding CFD01 on top of ZOI11 removes 39. The decision is about the production array, so the definition matters.

### 4.5 What the plots look like

Nobody who writes the PM text has seen one. The executor answered yes and no on six PNG files and opened no HTML file.

### 4.6 Whether two tags over two days generalise

All scored events are from 2025-06-10 (forebay, drags, 09:58 to 11:46) and 2025-06-11 (barrier, 12:02 to 12:15). Two boat-held tags at fixed depths.

---

## 5. Information I need, in the order it would change my advice

Each item says what it is, what it decides, and the cheapest way to get it.

### A. Answers only the owner, the PM or Kevin can give

**A1. The debris barrier's location in project coordinates.** A line or polygon (easting and northing, or the local X and Y of the receiver table), or at least the as-held positions of DB-1 to DB-5 from the field notes.
*Decides:* everything about the PM's criterion. The 25 m and 50 m rings around ZOI11 and CFD01 are my stand-in. With the outline, the tool can count fixes under the barrier and elsewhere, which is the PM's own split.
*How:* a CAD or GIS export, a sketch with coordinates, or which receivers are mounted on the barrier (the 2026-11-01 journal says ZOI11 is; is CFD01?).

**A2. True tag depth for each scored event and tag.** Were FC36 and FFD3 at 5 ft and 15 ft during the forebay holds as during the drags? Were both at 20 ft at the barrier holds, or only FFD3? Is there a depth column in the testing workbook?
*Decides:* whether I can add a depth error column, which is the only truth available at the barrier holds, and whether the metre-scale depth shifts in section 3.1 are movement toward or away from the truth.

**A3. Which positions go to the client: the DBSCAN-screened table, the consensus fixes of the figure scripts, or something else?**
*Decides:* whether the C4 tables answer the production question or need a second scoring on `tblPositions_DengDBSCAN` (section 4.4).

**A4. What "resolution" means to the PM.** More transmissions fixed, tighter scatter, or smaller error? And what counts as "degrading": the median of fixes that existed before (paired, section 3.2), the worst case, or the appearance of new poor fixes?
*Decides:* which of V4b, V4c, V4d leads the PM text, and whether the FBY-S-9 pattern (old fixes unchanged, new fixes worse) counts for or against a receiver.

**A5. Is the decision one switch or two?** May the PM include ZOI11 and exclude CFD01?
*Decides:* how the PM text is laid out. The numbers lean differently for the two: ZOI11 narrows barrier spread in most pairs; CFD01 alone widens it in three and carries the larger p90 increases on drags.

**A6. What was "Path A"?** The exact choice the owner made at the failed preflight, and what the other paths were.
*Decides:* nothing in the analysis, but the review must record it and it is not in a committed file.

**A7. Why are four validation drags dated 2025-06-05 (FBY-NS-01, FBY-NS-02, FBY-SN-01, FBY-SN-02)?** Is the date right, and is there receiver data for that day?
*Decides:* whether the PM's "all validation drag lines" is met by 9 of 13. If the data exist, the arms need a second window, which is a rebuild and a disk question (16.80 GB free). If the PM accepts 9, one sentence in the text covers it.

**A8. FBY-S-6.** The docstring in `drag_test_figures_2025.py:301` says the sheet row reads "Replace this with DB-1-5". Confirm it is a placeholder and not a hold.

**A9. Do the large baseline errors have a known field cause?** FBY-S-10 (33 m and 59 m), FBY-D-1 and FBY-D-4 FC36 (24 m and 20 m). Boat drift, a GPS antenna offset, a tag on a long line, a wrong sheet time?
*Decides:* whether the largest "error fell" rows are real gains or events where the truth itself is off. They dominate the favourable side of V4d.

**A10. For Kevin: deep receivers at configured positions (`solutions_used = 0`) in this four day window. Normal?** Carried over from the C3 review. It affects all arms alike.

**A11. Pulse rates of FC36 and FFD3 in `tblTag`.** The run file gives FFD3 = 3.33 s and nothing for FC36; the owner checklist records an observed 3.34 s for FFD3.
*Decides:* how exposed the numbering formula in section 4.1 is to period error. Two numbers from one query, or from the tag specification sheet.

### B. Numbers the tool can produce (a tool revision, then a short rerun)

No rebuild is needed for any of these. Each command took under two minutes in C4.

**B1. An alignment check that leaves the added receivers out.** First arrival per transmission number using only the nine baseline receivers, each arm against N; plus, per baseline receiver, the count of (transmission number, receiver) pairs present in one arm and not the other; plus which receiver holds the global first detection of each tag in each arm.
*Decides:* whether V4b and V4d stand as they are (section 4.1). This is the one item that could invalidate committed tables, so it goes first.

**B2. The same check reported for the flagged set only**, with the count of flagged transmissions whose earliest detection in arm N is a ZOI11 row. If my reading is right that count is 312 and 297.

**B3. Drag membership from the nine baseline receivers** (or from solver `t0`), so the start and end of each drag are cut the same way in every arm.

**B4. Error of the gained fixes and of the lost fixes, separately**, per event, tag and arm: count, median, p90 for gained fixes in the arm, and for lost fixes as the baseline had them.
*Decides:* section 3.3 and 4.2. "Gained 311" means little until I know whether those 311 are 3 m fixes or 30 m fixes.

**B5. Paired change as a distribution**: of the fixes present in both arms, how many moved more than 1 m, 2 m and 5 m closer to truth and how many further.
*Decides:* the "without degrading" sentence for the PM, with the tail visible.

**B6. Depth error** once A2 is answered: median and p90 of z minus deployed depth, per event, tag and arm, including the barrier holds.

**B7. Fixes inside and outside the barrier outline** once A1 is answered, replacing the bands. Until then: the band table split by event kind, and a rate (fixes per transmission available) beside each count, so drag duration stops driving the totals.

**B8. Cross-arm agreement at the barrier holds.** For each barrier hold and tag: the median XY of each arm's fixes and the distance between arms, and the same for gained fixes only.
*Decides:* whether the barrier gains are consistent positions or scatter (DB-5 FC36).

**B9. The same V4a to V4d on `tblPositions_DengDBSCAN`**, if A3 says that table is the product.

**B10. Detections per transmission at ZOI11 and CFD01 per event.** DB-1 FC36 shows 102 ZOI11 detections for about 22 transmissions, near five per transmission. Which arrival the solver uses when a receiver logs several for one transmission is worth one line in the preflight.
*Decides:* whether multipath at the barrier explains the depth shifts and the DB-5 disagreement. `[not measured]`

**B11. Why DB-2 FFD3 stops at 13 of 20 in every arm**: for the 7 transmissions without a fix, the number of receivers that heard them and the solver comment counts.

**B12. A fix count with a depth screen** (z between the bed and the surface), beside the unscreened count, so fixes above the water are visible as such.

### C. Files I need committed so I can see them

The reviewer sees only what is committed. Plan C4 section 6 item 3 left this to the owner.

**C1. `event_scores.csv` and `fixes.csv`.** Small text files. With `fixes.csv` I can compute B4, B5, B8 and B12 myself and check the tool's tables instead of asking for another run. This is the single most useful thing to add.

**C2. Ten PNG files under `docs\reports\figures\`:** DB-1 to DB-5 for FC36 (the barrier), FBY-S-9 FC36 and FBY-S-10 FFD3 (largest changes), FBY-D-1 FC36 (most fixes lost), FBY-D-4 FC36 (most fixes gained), ENT-05 FC36 (CFD01 tail). I can read images.

**C3. `events.csv`** and, if they exist, the build logs of arms N and Z around the line "no clock correction curve for receiver(s)". If arm N's log names ZOI11 there, section 4.1 is measured and B2 becomes a formality.

**C4. The owner's own look at the plots.** Three lines are enough: do the four arms sit on top of each other at the barrier holds, does any arm scatter visibly more, do the 3D pages open and rotate. The executor could not open them.

### D. Process, for the next plan

**D1.** Clock time on every STOP and the owner's reply quoted in full. C4 section 12 has neither.
**D2.** When a check the plan marks STOP fails and the owner goes on, the report needs the owner's reason in the owner's words.
**D3.** HTML files cannot be opened by the executor's harness. The next plan should ask the owner for that look (C4) and drop it from the executor's tasks.
**D4.** The tool should print the Task 2 identity columns to the console so they reach the cover report by copy.

---

## 6. Paths forward

**Path 1, recommended. A short C5 on the existing databases.** I revise the tool (B1 to B5, B7 interim form, B8, B10 to B12; version `fourarm-...-b`, self test extended), the executor reruns four commands, the owner commits `fixes.csv`, `event_scores.csv` and ten PNG files. Cost: one tool revision, about ten minutes of machine time, no rebuild, no disk. It settles whether the committed tables stand and turns "gained 311" into a statement about quality. Then I write the C4/C5 review and the PM text from numbers I have checked.

**Path 2. Write the PM text now from the C4 tables.** Possible today. It would say: the barrier holds gain 21 (ZOI11), 15 net (CFD01) and 24 net (Both) fixes on a baseline of 151; fixes that existed before keep their error within 0.34 m in the paired median; new fixes are in places less accurate than old ones; depth moves by metres at the barrier with no truth to judge it. It would carry a caveat that the tables rest on a transmission matching check that failed and that I believe is an artifact but have not measured. I would not send that to a client-facing PM without B1.

**Path 3. Rebuild arms N and C with ZOI11 clock-synchronised but outside `receivers_3d`.** That removes the uncorrected ZOI11 times at the source. Two builds, disk pressure, and it changes the baseline definition. Only worth it if B1 shows the numbering really differs.

**Middle course if time is short:** commit `fixes.csv`, `event_scores.csv` and the build logs (C1, C3) and answer A1 to A5. I can then do most of Path 1's analysis myself from the CSV, and only B1 to B3 need the machine.

**What I would ask the PM first, if only one message goes out:** A1 (barrier outline), A4 (what resolution and degrading mean) and A5 (one switch or two). Those three shape the deliverable more than any further computation.

## 7. Limits of this entry

- Every statement in section 3 is arithmetic on committed tables. I have not seen a database row or a plot.
- Section 4.1 is a reading of code plus three consistent numbers. It is a hypothesis until B1 or C3 comes back. In this thread I have already proposed one cause that the data refuted (the deep receiver hypothesis, C3 review section 3), so it should be treated the same way.
- The 6.1 m barrier depth comes from an earlier journal by another model and is not confirmed for FC36.
- The self test passes the alignment check on synthetic arms that all share one clock state, so it never exercised the situation the real databases are in. That is a gap in my tool, not in the executor's run.
