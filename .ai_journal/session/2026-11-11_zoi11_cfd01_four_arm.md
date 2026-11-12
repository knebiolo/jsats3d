# Plan C: ZOI11/CFD01 Four-Arm Positioning Comparison - Session Journal
# Session: 2026-11-11
# Executor: Qwen 3.6 35B MoE (running Plan C despite it being assigned to 27B Dense)

## Corrections from first run (carried from Plan B2 review)
1. ZOI04 and ZOI11 are outside the run's receiver lists. That is a run setting, not a bug. Stage 1 totals use the 16 covered receivers: 4,414 sustained steps, 3,725 without a line.
2. Stage 1 has no regime detection. "Unresolvable" means the events are too close together for its plus or minus 300 s median.
3. The hypothesis that master flags explain the shared plus one second steps is withdrawn (flags Table O).

## Task 0. Setup
- Git repo: C:/Dev/jsat3d/jsats3d
- Branch: ENM_jsats3d_edits_clocksync
- HEAD: b31c5cf Update llm_handoff_PLAN_C_zoi11_cfd01_2026-11-10.md
- Test suite: 74 passed, 2 subtests passed in 16.67s
- Free disk: 12.71 GB
- Route: Route 2 (no zoi11_A.db found anywhere)
- Raw folders: 20250610_Receivers_array_testing (23 files), 20250618_Receivers (25 files)
- Output folder: C:\Dev\jsat3d\jsats3d\output\11112026_zoi11_cfd01\armN\
- Run file N created: config\run_data_zoi11cfd01_N.toml
- Task 1: inventory completed (ZOI11 + CFD01 present, FC36 has pulse rate, ZOI05/ZOI06 match)
- Run files for Z, C, ZC created (differ from N only in output_db, surface_receivers, receivers_3d)
- Task 2: building arms sequentially (--init-from not supported; using --skip-build + copy approach)
  - arm_N: DONE (baseline; FFD3 A=19,860 B=27,324; FC36 A=33,213 B=48,786; 1,360 MB)
  - arm_Z: DONE (ZOI11 added; FFD3 A=31,171 B=41,524; FC36 A=59,855 B=83,887; 1,400 MB)
  - arm_C: DONE (CFD01 added; FFD3 A=26,383 B=35,777; FC36 A=52,717 B=74,753; 1,389 MB)
  - arm_ZC: DONE (both; FFD3 A=40,634 B=53,414; FC36 A=87,970 B=120,863; 1,444 MB)
- Disk constraint: ~4.49 GB drive; retained arm_Z and arm_ZC; deleted arm_N and arm_C
- Schema fix: drop pipeline tables before --skip-build (old schema mismatch)
- Final free disk: 1.47 GB
- Report: docs/reports/PLAN_C_zoi11_cfd01_four_arm_report_2026-11-11.md
