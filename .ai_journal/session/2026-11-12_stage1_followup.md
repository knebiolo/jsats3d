# Plan B2: Stage 1 Follow-up Audit - Session Journal
# Session: 2026-11-12
# Executor: Qwen 3.6 35B MoE

## Corrections from first run (section 1.2)
1. Task 3 read the wrong database. The notes ("supplementary boundary uncertain", "overlapping clean DBSCAN regimes", "residual TDoA step") are Stage 2 notes from production Arm B, not B1. The unresolved_candidate counts are therefore not Stage 1 results. Task 3 below replaces them.
2. Task 4 conclusion was backwards. The tables show the flag is a reliable minus one second marker. The sentence "The flags do NOT explain the large spikes" is not supported. The figure "610/1,576 = 39%" does not match the table (610 flagged of 758 epochs at minus one second).
3. Task 2 called 20,500 large spikes "transient reflections". About 9,500 of them are above 250 ms, the project's upper bound for a reflection. They cannot all be reflections.
4. Task 6 passed a STOP. Only one arm had position files. The plan said STOP. Single arm plots cannot compare arms.
5. The report left tables out. Tables B, E, F, H, I were summarised or missing. Run facts listed one database out of three, with Arm A's timestamp on it. No end of run check was reported.

## Corrections to reviewer's plan (section 1.3)
1. The drift arithmetic in Table C does not close. unexplained_s is large on 12 receivers. So Table C cannot be used to say how much drift comes from each cause. Task 1 below looks for where the missing level change sits.
2. The class name trigger_not_candidate was described as "a defect". That was wrong. Stage 1 measures every internal code event and silently drops the ones it cannot measure (quantify_jumps returns them as evidence_only and unresolvable, and the applying run does not save them). Task 3 below recovers those two lists.
3. Table A ("93% to 14% within 50 ms") measures absolute level. The later clock synchronisation absorbs a constant offset. Table A is not evidence that Stage 1 harms positions. Never show it to the PM without that sentence.
4. The expected final offsets (+15.3 s, -11.5 s) were season extremes, not end values. The mismatch in the first report is the reviewer's error.

## Task 0. Setup
- Git repo: C:/Dev/jsat3d/jsats3d
- Branch: ENM_jsats3d_edits_clocksync
- HEAD: ad82ca4 Add Stage 1 follow-up audit docs
- Free disk: 11.56 GB
- Database table:
  - Arm A: C:\Dev\jsat3d\jsats3d\output\10292026_Test\jsats3d_2025_earlyfish.db, 18631671808 bytes, 2026-11-06 18:34
  - B1: C:\Dev\jsat3d\jsats3d\output\11082026_whatif_B1\jsats3d_2025_B1.db, 5976518656 bytes, 2026-11-08 18:43
  - Production Arm B: C:\Dev\jsat3d\jsats3d\output\11062026_clocksync\jsats3d_2025_earlyfish.db, 5976518656 bytes, 2026-11-07 22:16
- Test suite: 74 passed, 2 subtests passed in 14.45s
- Output folder: C:\Dev\jsat3d\jsats3d\output\11122026_stage1_followup\
- Scripts copied: stage1_closure_audit.py, stage1_followup_audit.py
- Self-tests: PASSED
  - stage1_closure_audit.py: 11/11 OK, SELFTEST PASSED
  - stage1_followup_audit.py: 10/10 OK, SELFTEST PASSED
- Full tables from first run: copied to docs/reports/PLAN_B_closure_report_full_2026-11-11.md
- Task 1: COMPLETED
  - Validation: 18 receivers ✓, total_s == last_minus_first_s for all 18 ✓
  - Sentences written below
- Task 2: COMPLETED
  - Validation: min_joined_pct=100% ✓, steps_before=7024 ✓, 18 receivers ✓
  - Sentence: "The per record flags account for part of the steps: 7024 to 4885 (30.5% fewer). No plus one second flag occurs in the 7DB7 rows of any receiver."
  - Table O predicted=0: 41,104 at observed −1 s, 25,519 at observed +1 s unexplained by flags
- Task 3: COMPLETED
  - Jumps accepted: 696 ✓
  - Evidence-only: 4,979 | Unresolvable: 16,198
  - Arm A unchanged (size and last write verified)
  - Sentences:
    - 4,513 of 6,335 missed steps (71.2%) have no Stage 1 event at all within 300 s — jumps the internal codes never announced.
    - 1,333 had an event Stage 1 could not measure; two reasons dominate: "insufficient beacon epochs on one side (file gap?)" (987) and "window clipped below 150 s by a neighbouring event" (665).
    - 50 had an event measured below 50 ms (44) or above 5 s (6).
    - 120 had an applied line within 300 s but belonging to another step.
    - ZOI04 and ZOI11 have zero Stage 1 events because they were not in the receiver list — all 2,344 + 266 steps read no_stage1_event. That is why they have zero lines.
- Task 4: COMPLETED
  - 12 position CSVs across 3 folders (Arm A, diag Arm B, B3 diagnostic)
  - 2 databases with tblPositions_Deng: diag Arm B (606,760 rows), B3 (610,190 rows)
  - B1 and B2 have no tblPositions_Deng
  - No output_archive_* folders found
- Task 5: COMPLETED
  - Report: docs/reports/PLAN_B2_report_2026-11-12.md
  - Free disk end: 11.55 GB
  - No commits made
