# Session: 2026-10-08 Master Beacon Comparison & Deng All Receivers

**Author:** AI assistant
**Timestamp:** 2026-10-08T14:00:00-07:00
**Tags:** master-beacon, clock-sync, deng, tag-drag, 2025
**Status:** In progress

## Files Touched
- `scripts/test_master_beacons.py` — created (ancillary, will delete after use)

## Summary
User wants to improve pipeline accuracy by:
1. Testing every receiver as master beacon candidate
2. Running Deng on all receivers (not just deep receivers)
3. Fixing clock sync issues

## Decisions & Assumptions
- Test harness script `test_master_beacons.py` is ancillary — will be deleted after use per user rule
- Each master beacon candidate gets its own database copy from the legacy-aligned base
- Pipeline runs full legacy-aligned sequence for each candidate
- Clock stats (RMS, max, mean, median correction) are the primary comparison metric
- Drag accuracy against GPS truth is the secondary metric

## Current State (from conversation summary)
- Legacy-aligned DB: `output/jsats3d_2025_tagdrag_ent_legacyaligned.db`
- 141,966 solution rows
- Raw horizontal accuracy (median/p90):
  - ENT-01/FC36: 3.30 / 20.0 m
  - ENT-01/FFD3: 4.29 / 19.3 m
  - ENT-05/FC36: 3.87 / 27.2 m
  - ENT-05/FFD3: 3.95 / 24.3 m
- Worst outliers: 530 m (ENT-05/FC36), 401 m (ENT-05/FFD3)
- ZOI02 clock problem: 334 steps >1 ms, largest 322.9 ms, 154.9 ms RMS
- ZOI09: large but stable offset
- Master beacon: ZOI08
- Deep receivers: ZOI01, ZOI02, ZOI03 (only these get Deng step-6 positioning)
- Surface receivers use config GPS positions

## Test Run Status (2026-10-08 18:00-19:00 PDT)
- Test harness: `scripts/test_master_beacons.py` (ancillary, delete after)
- Candidates: ZOI08, ZOI03, ZOI05
- ZOI08: DB exists from previous run (skipped)
- ZOI03: DB exists from previous run (skipped)
- ZOI05: Running full pipeline, at step 7 (deep_clocks)
- Output: `output/test_master_beacons.log`
- Previous run showed ZOI05 deep positions: ZOI01 (119m shift), ZOI02 (1m shift), ZOI03 (5m shift)
- Test will complete all steps: deep_clocks → fish_multipath → speed_of_sound → positions → export → plot_drag
- Then collect clock stats + drag accuracy for each candidate

## Plan
1. Run `test_master_beacons.py` with candidates: ZOI08 (current), ZOI03 (anchor), ZOI05 (swapped pair)
2. Compare clock stats + drag accuracy
3. Pick best master beacon
4. Rerun full pipeline with best master
5. Extend Deng to all receivers with beacon detections (not just deep receivers)
6. Re-evaluate accuracy against GPS truth

## Blockers & Known Limitations
- K: drive must be accessible for raw data
- Each full pipeline run is time-consuming (metronome + clock fix + Deng for each candidate)
- CFD receivers (CFD02-08) are mobile, GPS positions may be less reliable

## Next Steps
- Run master beacon test with 3 candidates
- Review results, pick best
- Rerun full pipeline with best master + Deng all receivers
- Delete `test_master_beacons.py` after use

## Active Context
- Database: `output/jsats3d_2025_tagdrag_ent_legacyaligned.db`
- Run config: `config/run_data_tagdrag_ent.toml`

## Session Log (2026-10-08)
- 19:00 PDT: Harness v1 invalid. `--init-from` runs the full pipeline with the run-file master (ZOI08) before the DB override, so the "ZOI03" test ran ZOI08. ZOI08 run killed at positions (exit -1). Harness deleted.
- Master is read from two places: run file `[dbscan] beacon_receiver` (metronome) and `tblStudyParameters.masterReceiver` (clock fix, `lp.master`). Candidate runs set both (temp run file + DB update).
- `init_whatif` without `--swap` resets tblReceiver to `tblReceiver_initial`, which is the UNSWAPPED config. Candidate DBs use `--init-from legacyaligned --swap ZOI05 ZOI06`.
- Earlier harness clock stats (ZOI05 -19.3 ms, ZOI06 +11.3 ms) came from unswapped positions. Invalid. Swapped baseline: ZOI05 -6.4 ms, ZOI06 -1.6 ms.
- Base: `output/jsats3d_2025_tagdrag_ent_legacyaligned.db` (live ZOI05/ZOI06 swapped; same positions as swap0506).
- ZOI03 excluded as master: pipeline requires master in surface_receivers with a known position. ZOI03 position comes from step 6, which needs the clock fix first.
- Phase 1 (clock steps only) for all 12 surface receivers. ZOI08 baseline candidate (19:31 PDT) reproduces legacyaligned deep_clocks stats exactly. Batch of 11 others started async.
- Baseline ZOI08 defects: ZOI02 334 steps >1 ms (rms 154.9 ms); ZOI09 constant -946 ms offset (rms 0.99 ms).
- Ancillary scripts deleted: `scripts/test_master_beacons.py`, `scripts/check_test_dbs.py`, `scripts/check_position_data.py`.
- Reference-clock decision is owned by Kevin (2026-10-01 decision row). This test informs it; production master is not changed here.

## Batch Results (19:37 PDT, 6 of 12 done)
- ZOI08 baseline `|residual|` medians (ms): ZOI05 7.0, ZOI06 1.6, ZOI07 0.2, ZOI10 6.2, CFD02-08 1.8-6.4, ZOI09 946 (constant).
- ZOI05 master: ZOI06 54.3 ms, ZOI07 8.8, ZOI08 10.3, CFD02-08 0.9-6.2. Failed at deep_positions: no root-B solutions for ZOI01.
- ZOI06 master: failed at deep_positions (same ZOI01 error).
- ZOI07 master: failed at deep_positions (same ZOI01 error).
- ZOI09 master: failed at deep_beacons (`No objects to concatenate` in `beacon_pairwise_dbscan.cluster`). Sparse shared epochs.
- ZOI10 master: failed at deep_beacons (same error). Sparse shared epochs.
- CFD02 in deep_clocks. CFD03, CFD04, CFD06, CFD07, CFD08 queued.
- ZOI08 is the only viable master so far.

## Plan
1. Phase 1 batch: 11 candidates, compare clock stats per receiver (score = steps >1 ms, rms about line). [running]
2. Full positions/export/plot_drag for top candidate(s). Baseline drag numbers from `output/run_ent_legacyaligned_positions.log`.
3. Deng on static receivers (ZOI05, ZOI06, ZOI07, ZOI09, ZOI10; master excluded), references = other static 3D receivers. Deep receivers unchanged. CFD02-08 excluded: dynamic GPS (floats).
4. Re-evaluate vs GPS truth (plot_drag / `ent_analysis_2025.py --only 12`).

## Decisions (after batch, 19:45 PDT)
- Batch final: ZOI08 is the only static surface master that passes end to end. CFD03/04/07/08 pass the clock stage but are floats with dynamic GPS, so their config position is not a fixed clock reference. CFD02 fails in deep_clocks (SVD), CFD06 fails (no ZOI09 clock rows). ZOI05/06/07 fail at deep_positions (ZOI01 no root-B solutions). ZOI09/10 fail at deep_beacons (no shared epochs).
- Clock metric excluding ZOI02 (about 400 steps with every master, common-mode): ZOI08 median rms 0.99 ms, 0 steps. CFD03 0.92 ms, 1 step (float, rejected). CFD04 1.21 ms (float). CFD07 1.23 ms, 34 steps (float). CFD08 3.36 ms (float).
- Master stays ZOI08. Production master unchanged; the reference-clock decision remains Kevin's.
- Deng-all in `scripts/tagdrag_2025_pipeline.py`: `deng` = deep + static surface 3D receivers, master excluded (ZOI01, ZOI02, ZOI03, ZOI05, ZOI06, ZOI07, ZOI09, ZOI10). Each surface receiver is solved against the other surface receivers at config positions. Deep solves unchanged. Adoption writes XYZ for deep, XY only for surface (surface Z stays configured).
- Test change: `tests/test_resume.py` context gains `deng`. Full suite 55 of 55 pass.
- Run: `output/jsats3d_2025_tagdrag_ent_dengall.db`, log `output/run_ent_dengall.log`, `--init-from` legacy-aligned with `--swap ZOI05 ZOI06`.
- Baseline GPS (legacy-aligned, plot_drag "3D legacy B centroid", horizontal median / p90 m): ENT-01 FC36 3.30 / 20.0; ENT-01 FFD3 4.29 / 19.3; ENT-05 FC36 3.87 / 27.2; ENT-05 FFD3 3.95 / 24.3.
- Open: CFD02-08 excluded from Deng (dynamic GPS). Confirm with owner.

## Deng-all run: first attempt failed, then changed (after 19:45 PDT)
- First run `output/jsats3d_2025_tagdrag_ent_dengall.db`: steps 2-5 ok. Step 6: deep ZOI01-03 solved with static references. ZOI05 beacon (tag 7F91) failed with no root-B solutions.
- Cause: static surface references give 0 transmissions with 4 or more clean receivers for tag 7F91 (392 with any). ZOI09 has 1 clean detection, ZOI10 none. CFD receivers hear it well.
- With CFD receivers as references: 314 transmissions have 4 or more clean receivers.
- Change in `scripts/tagdrag_2025_pipeline.py`: surface Deng references = `receivers_2d` minus deep minus target (CFD via `tblReceiverGPS` tracks, handled by `jsats3d.position`). Deep references unchanged (static). CFD receivers are references only, never targets. A surface receiver with no solutions keeps its configured position and is recorded as skipped.
- Resume run (`--db` on existing DB, metronome/clock_surface/deep_beacons skipped): `output/run_ent_dengall_resume.log`.
- Tests: `test_resume.py` 12 of 12 pass after the change.
- Open: CFD as references (dynamic GPS, unit position offsets 16-30 m possible) needs owner sign-off. Compare GPS results with and without.

## Deng-all result (negative): surface adoption made GPS worse
- Resume run (`output/run_ent_dengall_resume.log`): steps 6-11 ok. Step 12 failed to write the four PNGs but printed the GPS table. Likely cause: ENT-01 FFD3 has only 2 hull-valid fixes.
- Deep ZOI01-03: identical to legacy-aligned baseline (0.0 m). Only surface receivers moved.
- Surface Deng: ZOI05 solved XY shift 16.7 m with solved Z +40.8 m (implausible, config Z kept). ZOI07 shift 26.3 m from only 4 solutions. ZOI09 shift 4.3 m from 9 epochs. ZOI06 and ZOI10 no solutions (configured kept).
- GPS, "3D legacy B centroid", horizontal median / p90 (m), baseline vs Deng-all:
  - ENT-01 FC36: 3.30 / 20.0 vs 18.60 / 58.4 (yield 89% vs 87%)
  - ENT-01 FFD3: 4.29 / 19.3 vs 22.15 / 84.7 (yield 67% vs 49%)
  - ENT-05 FC36: 3.87 / 27.2 vs 10.61 / 86.7 (yield 94% vs 94%)
  - ENT-05 FFD3: 3.95 / 24.3 vs 8.98 / 50.3 (yield 85% vs 83%)
- Hull-valid counts fell: ENT-01 FFD3 13 to 2, ENT-05 FFD3 48 to 22.
- Decision: `deng_surface` flag added to `config/run_data_tagdrag_ent.toml` [legacy], default `false`. Production stays legacy deep-only (`output/jsats3d_2025_tagdrag_ent_legacyaligned.db`). Deng-all DB kept as experiment record, not production.
- Tests: full suite 55 of 55 pass.

## Fast positioning harness (validated)
- Ancillary `scripts/_exp_positions.py` (delete after use): copies the legacy-aligned DB, keeps only the GPS line windows (epochs 1749550800-1749551800 for ENT-01, 1749552300-1749553200 for ENT-05), reruns Deng for FC36/FFD3, exports, scores with `ent_analysis_2025.py --only 12`.
- Validation: reproduces production "3D legacy B centroid" exactly. ENT-01 FC36 3.30/20.0 m, ENT-01 FFD3 4.29/19.3, ENT-05 FC36 3.87/27.2, ENT-05 FFD3 3.95/24.3. Same transmission counts (55, 51, 89, 81).
- Variants and their rules (chosen from clock/physics evidence, not GPS error):
  - `drop02`: exclude ZOI02 entirely (clock RMS 155 ms vs 7 ms or less for the others).
  - `jump5`: exclude ZOI02 detections within 5 s of its clock steps (>1 ms between consecutive rows).
  - `wc`: legacy `water_column` flag on (rejects solutions above WSEL or >4 m below the lowest receiver).
- Caveat: the GPS holdout scores all variants. Picking the best of several on the same holdout makes the gain optimistic. Confirm any choice with the owner.

## Variant results (line windows, horizontal median / p90 m, "3D legacy B centroid")
- Baseline (validated): ENT-01 FC36 3.30/20.0; ENT-01 FFD3 4.29/19.3; ENT-05 FC36 3.87/27.2; ENT-05 FFD3 3.95/24.3.
- `drop02`: 4.23/15.9; 6.09/15.5; 2.91/25.9; 3.73/21.2. Mixed. ENT-01 FFD3 yield falls 67% to 39%. Not adopted.
- `jump5`: identical to baseline. Zero ZOI02 detections within 5 s of a clock step in the windows. The ZOI02 problem is outside the drag windows.
- `wc` (legacy water-column flag): 3.22/15.7; 4.13/16.9; 3.40/23.2; 3.17/12.9. Better median and p90 in all four lines. Yield nearly unchanged (ENT-01 FFD3 34 to 30 scored). Depth estimates mixed.
- Independent check: 9 static holds FBY-S-1..10 from `cowlitz_AT_2025_testing_sheets.xlsx` (Forebay_StaticHolds, 06-10, FBY-S-6 absent). Not used for variant choice.
  - Base (production-equivalent): n=313, horizontal median 5.66 m, p90 30.65 m, depth median -3.00 m.
  - `wc`: n=297, median 4.40 m, p90 28.54 m, depth median +2.23 m. 12 of 18 hold-tag pairs improve. FBY-S-8 FFD3 falls from 9 solutions to 1. Depth sign flips.
  - Verdict: the drag-line gain was partly optimistic (holdout tuning). `wc` is not production-ready on its own.
  - Errors are largest at FBY-S-9 and FBY-S-10 (15-50 m). Both lie south of ZOI08/ZOI09, outside the receiver hull. Geometry is the limit.
- Next diagnostic: CFD02, 03, 04, 06, 07, 08 as dynamic 3D references (tracks in `tblReceiverGPS`, covering June to September). CFD02 and CFD03 sit south of the array.
  - `cfd_drags` result (much worse, horizontal median / p90 m): ENT-01 FC36 9.39/38.3 (baseline 3.30/20.0); ENT-01 FFD3 8.40/31.4 (4.29/19.3); ENT-05 FC36 21.68/101.0 (3.87/27.2); ENT-05 FFD3 23.82/137.1 (3.95/24.3). Rejected. Likely the float unit-position offset (16-30 m) biases the references.
  - `cfd_holds` on full windows ran 72 min without finishing one tag (1365 combinations per transmission). Stopped. Restarted on tight hold windows as `holds_base_tight` (reproduces base) and `cfd_holds_tight`.
  - Tight-window base reproduces exactly: n=313, median 5.66 m, p90 30.65 m, depth -3.00 m.
  - `cfd_holds_tight`: n=335, median 12.94 m, p90 77.31 m, depth -9.15 m. Worse on independent holds too. CFD references rejected for both drags and holds.

## Position DBSCAN diagnostic (diagnostic only, not production)
- Method: DBSCAN (min_samples 5) on Deng root-B XY per tag. Primary eps 5 m, fixed before scoring. eps 3 m and 10 m are sensitivity only.
- Baseline reproduces production per line: ENT-01 FC36 3.31/19.97, ENT-01 FFD3 4.30/19.30, ENT-05 FC36 3.89/27.21, ENT-05 FFD3 3.95/24.30.
- Drags, eps 5 m: pooled median 3.86 to 3.62 m, p90 21.7 to 14.7 m, kept 236 to 211.
- Holds (independent), eps 5 m: median 5.66 to 5.28 m, p90 30.7 to 25.5 m, kept 313 to 297.
- Sensitivity: eps 3 m gives drags p90 15.9 m and holds median 5.04 / p90 24.2 m, but keeps only 67% of drag fixes. eps 10 m gives little.
- Verdict: helps both checks with a pre-set rule. It is an outlier screen the owner removed earlier, so it needs owner sign-off.
- Log: `output/dbscan_diag.log`. Harness `scripts/_dbscan_diag.py` deleted after use.

## DBSCAN implemented as opt-in pipeline step (PM request)
- `scripts/tagdrag_2025_pipeline.py`: new steps `positions_dbscan` (writes `tblPositions_DengDBSCAN`) and `plot_drag_dbscan` (runs `ent_analysis_2025.py --table tblPositions_DengDBSCAN`). Config `[legacy] positions_dbscan = false`, `dbscan_eps_m = 5.0`, `dbscan_min_samples = 5`. Default off, so production is unchanged.
- `scripts/ent_analysis_2025.py`: new `--table` option (choices: `tblPositions_Deng`, `tblPositions_DengDBSCAN`).
- Test added: `test_positions_dbscan_drops_isolated_solutions`. Full suite 56 of 56 pass.
- Run on a copy `output/jsats3d_2025_tagdrag_ent_dbscan.db` (production DB untouched). Plots: `output/2025_review/jsats3d_2025_tagdrag_ent_dbscan_dbscan/`. Baseline plots for comparison: `output/2025_review/legacy_baseline_plots/`.
- Scores (3D legacy B centroid, median / p90 m, fixes baseline to DBSCAN):
  - ENT-01 FC36: 3.30/20.0 to 3.01/8.0 (51 to 42 fixes)
  - ENT-01 FFD3: 4.29/19.3 to 4.17/18.8 (35 to 30)
  - ENT-05 FC36: 3.87/27.2 to 3.46/13.6 (86 to 77)
  - ENT-05 FFD3: 3.95/24.3 to 3.86/18.0 (70 to 67)
- Plots: DBSCAN removes the long spikes into deep Z and far off track. Dense clusters of wrong fixes survive (e.g. near X 500-520 on ENT-01 FC36). Production approval pending owner sign-off.

## Least-squares TDOA diagnostic (not adopted)
- Purpose: test whether least-squares positioning smooths the multipath jumps. No least-squares solver existed in `jsats3d.py`; the class docstring mentions one, but only Deng and Deng2D are implemented. Built as a diagnostic.
- Same transmissions as the Deng baseline, same scoring against drag GPS and static holds.
- Drags, median / p90 m: Deng centroid 3.86 / 21.7. LS linear 6.38 / 56.0. LS robust (soft_l1) 5.29 / 53.9. LS linear warm-started from Deng 5.87 / 40.1. LS robust warm-started from Deng 4.80 / 28.3.
- Static holds, median / p90 m: Deng centroid 5.66 / 30.7. LS linear 12.65 / 251. LS robust 11.43 / 35.9. LS linear warm 11.32 / 69.3. LS robust warm 6.53 / 28.9.
- Verdict: no LS variant beats the Deng centroid on the median. Robust warm-start trims some tail on holds but worsens the median. The jumps are not removed by refinement. The multipath cause is not shown by this test. A per-receiver residual check would test it directly. Not adopted.
- Log: `output/ls_diag.log`. Harness `scripts/_ls_diag.py` deleted after use.

## Handoff status (2026-10-09)
- ENT-02..04 scored and plotted from the legacy-aligned DB (`ent_analysis_2025.py --lines`, default still ENT-01 and ENT-05). Plots: `output/2025_review/ent_all_lines_legacy/`. Legacy B centroid, median / p90 m: ENT-02 FC36 4.80/10.6 (yield 83%), ENT-02 FFD3 4.44/22.7 (79%); ENT-03 FC36 3.41/20.8 (97%), ENT-03 FFD3 2.45/16.4 (93%); ENT-04 FC36 1.90/17.4 (95%), ENT-04 FFD3 2.71/21.8 (88%). The production plot step still covers ENT-01 and ENT-05 only.
- ZOI01-03 positions come from beacon-based Deng (step 6, adopted into tblReceiver), not from config. Config to solved, with 'solutions' = Deng solutions used:
  - ZOI01: (447.07, 124.77, -11.15) to (439.01, 124.01, -13.11), shift 8.1 m, 202 solutions
  - ZOI02: (467.60, 116.36, -12.40) to (468.04, 116.64, -8.11), shift 0.5 m, 775 solutions
  - ZOI03: (486.07, 106.06, -10.29) to (486.82, 108.63, -8.78), shift 2.7 m, 343 solutions
- Earlier unswapped test (before the ZOI05/06 swap was applied) moved ZOI01 by 119 m. So the deep Deng positions depend on the surface positions used as references. Input review should include the swap.
- Clock jumps: ZOI02 has 334 steps over the day. In the drag windows, none fall within 5 s of a study-tag row (1250 rows checked). The drag results do not depend on them.
- Open for next week: input review (config, swap, depth/WSEL datum), ZOI02 clock and one-second flags, whether ENT-02..04 belong in production scoring, DBSCAN approval.
