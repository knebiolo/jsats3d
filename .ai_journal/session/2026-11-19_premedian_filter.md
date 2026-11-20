# 2026-11-19 - Pre-median per-epoch filter: implementation and first validation

## Session metadata
- Author: Copilot, AI assistant using Copilot SDK in VS Code.
- Timestamp: 2026-11-19 ~13:55 PST (UTC-08:00) / ~21:55Z.
- Tags: premedian, clock-jumps, deng, validation, experimental, owner-review.

## Request
"Apply a pre-median geospatial filter to solutions for a given epoch to remove bad solutions that may bias the
median. This is the best solution to the clock jump issue given the lack of information on the timing of those
events." A 3D box filter exists on branch `ENM_jsats3d_clock_updates` (jsats3d.py ~1822, PM 2026-11-16).

## Review of the branch box (no code taken from it)
- Fixed global box, not per epoch. X/Y = receiver min/max rectangle +5 m (not the hull, as its summary says).
- Top bound bug: `z_max = WSELfun.y.max()` = 866.4 ft / 3.28 = 264 m elevation, while receiver Z is local depth
  (Z_t -12.4 to -2.3 m) -> above-surface solutions are never rejected. Legacy `water_column` has the same frame issue.
- Always on, hard-coded margins, in core `position.Deng` -> would change 2019 output. No rejection counts.

## Baseline run
- `config/run_data_premedian_11192026.toml` = 10/29 early-fish run (K:\...\2025_output_results\10292026_run,
  run file run_data_earlyfish_2026-10-28.toml) with paths on C:\jsats_data, plus ZOI11 (surface + 3D) and CFD01 (3D)
  at the project lead's request. Output `output\11192026_premedian\jsats3d_2025_earlyfish.db` (kept read-only).
- 29,912,682 detections (identical to October). ZOI05 deep_clocks numbers identical to October to the digit
  (rms about line 497.457 ms, 703 steps >1 ms, largest 5190.5 ms). Total 46 min (parse 12.5, index 2.7, pipeline
  26.3, figures 4.4) vs ~80 min in October.

## Implementation (only `scripts/_2025_pipeline.py`; no new scripts; jsats3d.py untouched)
- New optional step `premedian` (label 10c) after bathymetry_check, before positions_dbscan.
  Off by default (`[positioning]`/`[legacy]` `premedian_filter = false`) -> done-check returns True, nothing runs,
  existing runs are bit-identical. 2019 runs use legacy_pipeline.py, which never calls _2025_pipeline.py.
- Box gate (switchable: `premedian_box`, `premedian_box_z`): XY within `premedian_box_margin_m` (5) of the 3D
  receivers' convex hull; Z between `premedian_box_z_top_m` (0 = surface in receiver frame) and deepest receiver
  minus `premedian_box_z_pad_m` (3). Non-finite solutions always rejected.
- Per epoch (Tag x root x transNo): start at the solution with most XY neighbours within `premedian_radius_m` (10 m,
  = project receiver-set consensus radius, PM 2026-10-14; mode-seeking, robust when bad solutions are a majority);
  keep those within the radius; fewer than `premedian_min_sets` (2) distinct receiver sets -> whole epoch rejected.
- Rejected rows stay in tblPositions_Deng with comment `premedian box|epoch outlier|epoch unsupported - rejected`;
  earlier premedian flags reset on every run (idempotent). Audit table `tblPreMedianFilter` (per tag/root, per
  receiver rejection share, settings JSON); counts and top receivers printed to the run log.
- Drops `tblPositions_DengDBSCAN` (rebuilt from filtered solutions). With the filter on, positions_dbscan uses the
  ping **median** instead of the mean (gated: off = original mean).
- Bug found and fixed during testing: done-check ignored changed settings; now compares saved settings JSON.
- Tests: 6 passed (jsat_3d env). `--list` shows the step.

## Validation 1 - drag GPS truth (06-10, 17 receivers incl. CFDs; `drag_validation\`)
Copies of `output\tagdrag\gps_smooth_2026-11-19\raw\tagdrag_gps_raw.db`; arms rerun from step 10c; scored with the
three-arm tool (project consensus -> smoothing -> Kalman for drags; receiver-set consensus for holds).
Run files `config/run_data_premedian_drag_B.toml`, `_C.toml`.

| Arm | Drag median-of-median XY err | Drag median p90 | Hold median err | Fixes |
|---|---|---|---|---|
| A no filter | 3.14 m | 7.30 m | 4.58 m | 1892 drag / 351 hold |
| B per-epoch | 3.60 m | 8.81 m | 4.97 m | 1856 / 351 |
| C per-epoch + XY box | 3.93 m | 12.89 m | 4.90 m | 1918 / 349 |

- B rejected 53% of drag solutions (epoch outlier 845,331); highest receiver shares CFD09 74% (static on 06-10, no
  GPS), CFD04 65%, CFD08 58%, ZOI01 55%, ZOI05 54%.
- C: XY box badly hurts edge lines (FBY-D-1 median 3.1 -> 32.8 m; FBY-D-4 7.8 -> 19.3 m).
- Neither arm improves accuracy on GPS truth. Caveat: 06-10 window has few clock jumps (ZOI02: 3 auto-detected,
  0.06-0.29 s) and the scored product already contains a 10 m / 2-set consensus, smoothing and Kalman.

## Validation 2 - fish (`filtered\`, per-epoch only, no truth)
- Full box first tried: rejected 245,605 of 344,404 found (box 236,464). Root B box reasons: Z > 0 30.4%, XY >5 m
  outside hull 21.9%, Z < -15.4 m 9.9%. Lost 6,784 baseline DBSCAN pings (46% above surface, 39% >5 m outside hull):
  box removes the known Z-bias/mirror and outside-hull presence data, not clock-jump artefacts. Not adopted.
- Per-epoch only: rejected 99,699 (outlier 81,861, unsupported 17,838); top receiver shares ZOI06 37%, ZOI01 33%,
  ZOI02 30%, ZOI03 30%, ZOI11 29%. DBSCAN-kept pings: 1D5A 7721 -> 5361, 26E4 1829 -> 1488, 9F8C 120 -> 123,
  B5ED 4484 -> 3644.
- Track plausibility, all root-B pings before DBSCAN (consecutive pings <= 12 s apart):

| Solutions | Ping agg | Pings | Median speed | >2 m/s | >5 m/s | Z>0 |
|---|---|---|---|---|---|---|
| no filter | mean | 16,242 | 1.19 m/s | 35.0% | 14.8% | 37.9% |
| no filter | median | 16,242 | 0.79 | 26.1% | 10.3% | 33.6% |
| per-epoch | mean | 11,206 | 0.57 | 12.1% | 3.6% | 29.9% |
| per-epoch | median | 11,206 | 0.51 | 12.6% | 3.7% | 26.8% |

  The filter, not the median, produces the smoothing; it costs 31% of pings.

## Interpretation (for owner review)
- On fish (long window, real clock jumps incl. ZOI05 5.19 s steps) the per-epoch filter removes most implausible
  speeds, at a 31% ping cost. On drags with truth it slightly worsens accuracy (+0.46 m median, +1.5 m p90).
  Smoother tracks are not proof of more accurate tracks.
- No production parameter changed; filter stays off by default. Adoption, radius and min_sets are owner decisions.

## Parameter changes
- None in production. Experimental defaults: radius 10 m, min_sets 2, box margin 5 m, z top 0, z pad 3 m (box off
  in the recommended arm).

## Files touched
- `scripts/_2025_pipeline.py` (step 10c, DBSCAN aggregation gate).
- Configs: `run_data_premedian_11192026.toml`, `_filtered.toml`, `run_data_premedian_drag_B.toml`, `_C.toml`.
- Outputs (gitignored): `output\11192026_premedian\` (baseline DB, `filtered\`, `drag_validation\` with three DBs,
  `premedian_three_arm.py` copy and `compare\` plots/tables).

## Next steps (proposed)
- Test drags inside a clock-jump interval, or synthetic jump injection into the drag DB (add a known offset to one
  receiver for a period) to measure the filter where it is meant to act.
- Radius sensitivity (e.g. 10 / 20 / 30 m) on drag truth - one change at a time, recorded.
- Score fish end product (consensus -> Kalman) for both arms with fish_figures.

## Clock-jump tests (14:05-14:45 PST)
### Real jump - 26E4 at ZOI02, fish baseline vs per-epoch copy (read-only analysis)
- ZOI02 clock knots in `tblDetectionClockFixed`: correction steps +0.436 s across a 512 s knot gap
  (06-27 23:49:58 -> 23:58:30) and -1.809 s across a 639 s gap (06-28 00:55:02 -> 01:05:41). Fish times inside the
  gaps get a linear ramp (np.interp), i.e. ZOI02 times wrong by up to 0.44 s / 1.8 s.
- Reference = per-ping median of non-ZOI02 root-B solutions (unaffected by ZOI02's clock), pings with >= 3 of them.
- Gap 1 (43 pings): filter rejected 75% of ZOI02 combos (9% outside gaps) - it identifies the jumped receiver.
  But only 1.9% of surviving solutions in the gap contain ZOI02 (large jumps already fail / exceed |XYZ| 1000 m),
  so the unfiltered median was already 0.0 m from the reference; filtered median 1.2 m (it also trims good combos).
  Gap 2: no pings with enough non-ZOI02 solutions.
### Injected jump with GPS truth - drag DB, ZOI02 FC36/FFD3 seconds_fix + offset, 2025-06-10 10:20-11:50
- Six copies of `drag_validation\A_none` (jump 10/50/400 ms x no filter / per-epoch filter), rerun from step 11;
  FBY-S holds (09:58-10:20) untouched as control. `drag_validation\jump_test\compare_XXXms\` (three-arm plots).

| Jump | Clean no filter | Jump no filter | Jump + per-epoch filter (drag median-of-median / median p90) |
|---|---|---|---|
| 10 ms (~15 m) | 3.14 / 7.30 m | 3.65 / 10.80 m | 3.65 / 10.55 m |
| 50 ms (~74 m) | 3.14 / 7.30 m | 3.37 / 7.82 m | 3.44 / 10.58 m |
| 400 ms (~590 m) | 3.14 / 7.30 m | 3.39 / 7.98 m | 3.44 / 10.58 m |

- Holds (control): unchanged without filter (4.58 m); filter 4.97 m in every arm (filter effect alone).
- Small jumps hurt most (they stay plausible); large jumps are largely self-rejected by existing checks.
- The per-epoch filter does not recover any of the jump damage on the scored product (project consensus ->
  smoothing -> Kalman), and worsens p90 at 50/400 ms. Worst line ENT-01 FFD3 (9.3 -> 15.6 -> 33.8 m at 10 ms).

## Conclusion
- The pre-median filter (10 m densest-cluster, >= 2 sets) is implemented, audited and opt-in, but on all available
  evidence it does **not** improve accuracy against clock jumps beyond the existing receiver-set consensus, and it
  costs yield. Not recommended for adoption as configured. Owner decision.
- Damage from jumps comes from uncorrected clock gaps; the 2026-11-03 audit recommendation (do not interpolate
  fish times across clock-knot gaps / carry segment boundaries) targets the cause directly and is the suggested next test.

## Wrap-up (15:02 PST)
- **Decision (project lead, 2026-11-19):** keep the implementation as it is - step 10c in `_2025_pipeline.py`,
  opt-in (`premedian_filter = false` default), box switchable (`premedian_box`, `premedian_box_z`). No production
  run file enables it. The fish test copy (`run_data_premedian_11192026_filtered.toml`) is set to per-epoch only
  (`premedian_box = false`).
- **B vs C (as explained to the project lead):** B = per-epoch only (keep the solutions that agree with each other
  within 10 m of the densest cluster, >= 2 receiver sets, then median). C = XY box first (drop solutions > 5 m outside
  the 3D receiver hull), then B. Same inside the array; at the edge C discards the correct cluster and is pulled
  inward (FBY-D-1 FC36 3.2 -> 32.8 m, FBY-D-4 FC36 9.6 -> 19.3 m, FBY-S-10 FC36 5.8 -> 15.1 m). The box was only
  tested together with the per-epoch step; the branch box as written (rectangle, broken top) was never tested alone.
- **Package for Drew (PM):** `output\11192026_premedian\for_drew\`
  1. `1_summary_error_by_jump_size.png` - drag median error, current method vs per-epoch filter, for no jump and
     10/50/400 ms injected ZOI02 jumps (3.14/3.60, 3.65/3.65, 3.37/3.44, 3.39/3.44 m). Made with a one-off inline
     command (no script added).
  2. `2_example_worst_ENT-01_FFD3_10ms_jump.png`, 3. `3_example_typical_FBY-D-3_FC36_10ms_jump.png` (copies of the
     three-arm plots).
  Email draft given in chat: filter identifies the jumped receiver but does not improve accuracy beyond the existing
  consensus; recommend not enabling it; propose handling clock-knot gaps directly.
- Not committed. Open: box-only jump test (offered, not run); clock-gap handling test (proposed next).

## Pulse-rate calculator fix (15:06 PST)
- `run_data.py` `_calculate_pulse_rates_from_indexed` (runs after indexing in finish_database; median of 1-10 s
  same-receiver gaps per study tag) overwrote every study tag's rate, including run-file `[study] pulse_rates`,
  despite its docstring ("any missing rates"). Today's build: 9F8C 3.05 -> 3.0506, B5ED 2.969 -> 2.9699,
  26E4 2.9862, 1D5A 3.0812 stored (<= 0.001 s from run file; negligible, but a silent override of a set parameter).
- Fix (project lead approved): skip tags whose tblTag.pulseRate is already set (run file or config workbook wins);
  calculation only fills NULL rates. Checked with an in-memory table (set rate kept, NULL filled = 3.000 s);
  6 tests pass. Existing databases unchanged.
