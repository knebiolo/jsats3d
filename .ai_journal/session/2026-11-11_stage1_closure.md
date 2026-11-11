# Plan B: Stage 1 Closure Audit - Session Journal
# Session: 2026-11-11
# Executor: Qwen 3.6 35B MoE

## Task 0. Setup
- Git repo: C:/Dev/jsat3d/jsats3d
- Branch: ENM_jsats3d_edits_clocksync
- HEAD: 944fe4d Add closure audit and handoff docs
- Free disk at start: 2.26 GB
- Input files verified:
  - series_armA_uncorrected.csv: 53.2 MB, 2026-11-09 17:32
  - series_b1_corrected.csv: 53.1 MB, 2026-11-09 17:32
  - stage1_correction_lines.csv: 295 KB, 2026-11-09 19:36
  - Production Arm B DB: 5.58 GB
- Output folder created: C:\Dev\jsat3d\jsats3d\output\11112026_stage1_closure\
- Audit script copied to output folder
- Self-test: PASSED (11/11 OK)
- Test suite: 74 passed, 2 subtests passed in 15.84s

## Task 1. Closure Audit
- Audit run: 18 receivers, 696 lines, 7,024 steps, 671 confirmed, 6,335 missed
- closure_report.md generated
- All validation checks passed
- Key finding: missed_steps term dominates for all receivers

## Task 2. tdiff Analysis
- 5 groups identified (BIC lowest at 5, histogram gaps consistent)
- Size-only classifier would lump 20,500 large spikes with 7,024 sustained steps
- BIC centres align with physics: 0.013 ms, 0.028 ms, 0.436 ms, 2.44 ms, 1,000 ms

## Task 3. Missed Step Evidence
- 6,335 missed steps classified:
  - no_internal_evidence: 3,189 (50%)
  - unresolved_candidate: 1,333 (21%)
  - trigger_not_candidate: 1,146 (18%)
  - one_second_flag_only: 667 (11%)
- unresolved_candidate notes: 1,100 DBSCAN boundary uncertain, 205 overlapping regimes, 28 residual TDoA

## Task 4. One-Second Flags
- Top 3 receivers: ZOI01 (1,082 spikes), CFD02 (485), CFD01 (373)
- Only -1 flags found. Zero +1 flags in any receiver.
- Flags explain dev_s=-1 cluster (39-91% coverage)
- Large negative deviations (-7 to -23 s) have no flag correspondence
- Flags do NOT explain the large spikes

## Task 5. Stage 2 Comparison
- 1,653 Stage 2 events, 6,335 missed steps
- ZOI04 and ZOI11: zero Stage 2 events
- 565 master-listener opposite-sign matches
- Many disagree cases have size diffs near 0.5, 1.0, 2.0 s

## Task 6. Example Fish
- Only Arm A position files available (4 tags)
- 3 jumps selected: ZOI03/26E4, ZOI01/1D5A, ZOI03/B5ED
- 3 plots created (Arm A only)
- Limitation: B1 position files not available

## Task 7. Drag/Hold Scoping
- zoi11_test databases do not exist on disk
- Cannot perform truth-based check

## Task 8. Report
- Report written: docs/reports/PLAN_B_report_2026-11-11.md
- Free disk at end: 11.59 GB
- No commits made

## Deviations from plan
1. Disk started below 3 GB (2.26 GB); user freed Downloads
2. Task 6: Arm A only (no B1 position files)
3. Task 7: zoi11_test DBs not on disk
4. Task 4: Only -1 flags found
