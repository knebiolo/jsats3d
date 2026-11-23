# 2026-11-22 - drag-tag validation of the receiver cap (prior vs cap 8 vs cap 14)

## Request (project lead 2026-11-22 19:11)
- Validate the method used before the cap against the cap method on the tag drags. Plots should be like
  `K:\...\2025_output_results\ZOI11_CFD01_Inclusion` (GPS truth plus each arm, per drag line/hold and tag).
- Arms:
  - prior: all 20 receivers, no cap.
  - cap 8.
  - cap 14.
  - Pre-median filter on for all arms.
  - Cap 20 equals prior with 20 receivers; the project lead confirmed 3 arms.
- Delete everything in `output` first. The 100-fish results had already been copied to K:\...\100 fish cap
  (188 PNGs and 4 CSVs).

## Setup
- Configs `config/run_data_tagdrag_cap_{base,prior,cap8,cap14}.toml`:
  - Generated from `run_data_100chinook_11212026.toml`: all 20 receivers, pre-median 12 m with box, DBSCAN on.
  - Tags FC36 and FFD3.
  - Window 2025-06-10 09:00 to 06-11 13:00, covering:
    - ENT-01..05 and FBY-D-1..4 (GPS lines, 06-10).
    - FBY-S-1..10 holds (06-10, GPS point).
    - DB-1..5 holds (06-11, no GPS point).
  - 2D (known-depth) step off; drag/fish figure steps off.
  - Arms:
    - prior: cap 0, hull filter off.
    - cap8: cap 8 + keep deep + hull 5 m.
    - cap14: cap 14 + keep deep + hull 5 m.
- Flow:
  - base: `run_data.py` with `[legacy] run = false` (build), then `_2025_pipeline.py --to fish_multipath`.
  - Copy the base DB to each arm, then run `_2025_pipeline.py <arm>.toml --from positions --no-drag-plots`.
  - So the arms share the same detections, clock fix and multipath; only the positioning differs.
- Arm comparison tool (first written as a `--arms` mode of `scripts/drag_test_figures_2025.py`; moved 11-23 at the project lead's request to `tests/drag_arm_comparison.py`, and the script was restored unchanged). Usage: `python tests/drag_arm_comparison.py --out DIR prior=DB cap8=DB ...`
  - Plots `<line>_<tag>.png`: GPS truth plus each arm's root-B consensus -> smooth -> Kalman track (the existing
    `line_product` chain), with n, median and p90 in the legend.
  - Plots `<hold>_<tag>.png`: each arm's per-transmission medians plus the arm median (large marker); GPS star and
    10/20 m circles when the sheet has a point.
  - CSVs `qc/arm_drag_errors.csv`, `qc/arm_hold_errors.csv`, `qc/arm_summary.csv` (incl. positions step minutes).
  - `timed_holds` in the test tool also returns timed holds without GPS (DB-1..5).
- Base build launched 2026-11-22 ~19:40 (`output\tagdrag_cap\base_console.log`).

## Run notes
- The first base pipeline run failed at deep_clocks: CFD05 has no data before August. CFD05 was removed from every
  receiver list of the 4 configs (19 receivers; ZOI04 has data only from 06-11 08:18). Resumed; the base finished
  19:48.
- Arms finished 20:04 (all exit 0). Figures at 20:05: 18 drag plots and 28 hold plots, in
  `output\tagdrag_cap\figures\`.

## Results (drag lines: Kalman track vs GPS; holds: per-transmission median vs GPS point)
| arm | positions step | drag pings scored (of 2,426) | median of drag medians | median of drag p90s | median of hold medians |
|---|---|---|---|---|---|
| prior (no cap) | 10.1 min | 1,738 | 4.12 m | 14.5 m | 4.79 m |
| cap 8 | 1.7 min | 1,231 | 4.15 m | 9.3 m | 5.46 m |
| cap 14 | 8.2 min | 1,738 | 4.07 m | 14.5 m | 5.03 m |
- Cap 14 is practically the same as prior: same yield, medians within about 0.5 m, only about 20% faster.
- Cap 8 is about 6x faster but scores about 29% fewer drag pings.
  - In-array lines (ENT, FBY-D-2/3) and most holds are a little worse (0-1 m; ENT-01 FFD3 12.4 vs 9.0 m).
  - FBY-S-2 is 12.6 vs 5.6 m. FBY-S-7/8/9 are worse, with fewer transmissions at S-9, S-10, DB-4 and DB-5.
  - It is much better on FBY-D-1 (5.5 vs 28 m) and FBY-D-4 (7.5 vs 19 m). These drags ran mostly south of
    CFD02/CFD03, outside the array. There, prior/cap14 give confident wrong in-array tracks; cap 8 mostly drops
    those pings (FBY-D-1 FC36 n = 19 vs 84). That is a selection effect: no arm positions the southern part.
- DB-1..5 (no GPS): arm medians agree within about 1-3 m except DB-4/DB-5, where cap 8 has few transmissions and
  differs by 5-18 m.

## 2026-11-22 20:45 - pre-median filter test on the drags (test only, no pipeline code)
- All arms are copies of the prior-arm DB (19 receivers, no cap):
  - pm_off: SQL reset of every 'premedian ...' comment.
  - pm_on: the prior arm as run.
  - pm_xyonly: `_2025_pipeline.py --only premedian` with a temporary run file (session folder, deleted) setting the
    existing option `premedian_box_z = false`.
- Plots and CSVs: `output\tagdrag_cap\figures_premedian\`.
- Horizontal (drag Kalman tracks vs GPS):

| arm | drag pings scored | median of drag medians | median of drag p90s | median of hold medians |
|---|---|---|---|---|
| filter off | 1,911 | 2.82 m | 6.3 m | 5.20 m |
| filter on | 1,738 | 4.12 m | 14.5 m | 4.79 m |
| XY box only | 1,913 | 3.38 m | 12.3 m | 4.89 m |

  - FBY-D-1: 2.5/3.3 m with the filter off vs 28/30 m on.
  - FBY-D-4: 5.9/9.5 m off vs 19/14 m on.
  - The box XY limit removes the true solutions of tags just outside the array, so the remaining wrong in-array
    solutions win. This corrects the cap-test reading: with the filter off, the southern FBY-D parts are positioned.
  - Off is also better or equal on ENT and FBY-D-2/3. Holds are slightly better with the filter on (4.8 vs 5.2 m).
- Depth (median absolute error vs tow depth; FC36 5 ft, FFD3 15 ft; DB holds 10/20 ft):
  - Drags: on 3.8 m, off 4.6 m, XY box only 4.8 m.
  - Holds: on 3.7 m, off 4.7 m, XY box only 3.9 m.
  - All arms are too deep: medians -5 to -9 m vs a true -1.5 / -4.6 m.
- Split idea (XY from one arm, Z from another): XY from "off" plus Z from "on" would be the best combination here.
- No pipeline changes were made.
