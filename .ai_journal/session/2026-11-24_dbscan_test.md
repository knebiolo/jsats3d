# 2026-11-24 - clock DBSCAN validation on the drag tags (5-pass vs segmented)

## Project lead instructions
- Yesterday's step-aware clock work went beyond what was asked and was reverted. The user's rule: only make
  requested changes, and talk first before changing code.
- No receiver cap. Clock jumps split only at a 50 ms residual change; the 3-beacon-period split is removed. Keep the
  rest of Drew's segmented logic.
- Validate on the drag tags FC36 and FFD3 first, with plots in `output\dbscan test`. Then look at 26E4, 4D92, 23C1
  and 45BA (06-24..06-29).

## Code
- Drew's commit 96a1544 had already removed the gap split and set `deng_max_receivers = 0` in the clockskip configs.
- Reverted my cfb7dbb additions (project lead: revert the extras; spike filter "whatever Drew said" = none).
  - `jsats3d/jsats3d.py`, `scripts/{legacy_pipeline,_2025_pipeline,ent_analysis_2025,fish_figures_2025}.py` and
    `README.md` are restored to merge 22990f9.
  - Reapplied the 96a1544 change: `_segmented_clock_dbscan` boundaries = `ToA_error.diff().abs() >= 0.05` only.
  - README wording updated.
  - Deleted `tests/test_clock_segments.py` and `tests/clock_segment_comparison.py`. `tests/test_smoke.py` is Drew's
    96a1544 version. 13 tests pass.

## Run
- Configs `config/run_data_dbscantest_{base,baseline,segmented}.toml`, from `run_data_tagdrag_cap_prior.toml`:
  19 receivers, no cap, no hull filter, pre-median on, window 06-10 09:00..06-11 13:00.
- The arms differ only in `segmented_clock_dbscan`. Plots come from `tests/drag_arm_comparison.py` into
  `output\dbscan test\figures`.

## Results (11:30, both arms exit 0, about 28 min total)
- Drags, summary (both arms the same on median of drag medians):

| arm | drag pings scored (of 2,426) | median of drag medians | median of drag p90s | median of hold medians | hold transmissions |
|---|---|---|---|---|---|
| 5-pass | 1,738 | 4.12 m | 14.5 m | 4.79 m | 340 |
| segmented | 1,782 | 4.12 m | 13.8 m | 3.85 m | 342 |

- Drag detail:
  - ENT lines identical except ENT-01 (+-0.3 m).
  - FBY-D-2/3 slightly better (0.1-0.3 m) with more pings.
  - FBY-D-4 FFD3 13.8 -> 9.2 m (n 64 -> 74).
  - FBY-D-1 still about 29 m in both arms (the pre-median edge effect, not the clock).
- Holds:
  - Better on S-1..S-5 and S-8 FC36 (e.g. S-2 FFD3 5.4 -> 2.9 m, S-3 FFD3 4.9 -> 3.0 m).
  - Worse on S-9 FC36 (6.1 -> 9.5 m) and S-8 FFD3 (6.5 -> 7.4 m).
- No degradation seen in yield. The 06-10 window has few clock steps, so this checks normal-period behaviour more
  than jump recovery.

## 12:00 - plan/event figures
- `tests/clock_segment_comparison.py` restored from cfb7dbb and trimmed (project lead OK): plan view and event figures
  only, no `tblClockSegments` or flags. Clock steps are found from the stored residuals (>= 50 ms).
- Output `output\dbscan test\clock_figures\`:
  - 18 drag-line pairs (tag x line) and 5 clock-step events.
  - `window_summary.csv`: pings per window per arm.
- Main event: ZOI02 +0.34 s from about 11:22 to 11:37 on 06-10 (FBY-D-2/3/4). The baseline ramps the correction
  linearly across both steps; the segmented filter holds levels. FC36 pings in the window: 240 -> 247.
- Drags: same ping counts except FBY-D-3/4 (+7/+9, +2/+8).
