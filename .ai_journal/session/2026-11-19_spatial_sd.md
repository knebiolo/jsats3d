# 2026-11-19 - Spatial SD dataset and maps (approximate spatial error rate)

## Session metadata
- Author: Copilot, AI assistant using Copilot SDK in VS Code.
- Timestamp: 2026-11-19 ~16:55 PST (UTC-08:00) / ~00:55Z 11-20.
- Tags: spatial-error, precision, fish-figures, validation.

## Request
"Create a dataset and associated figures displaying the dataset's SD over the entire area so that it may be used as
an approximation for spatial error rate of the positioning." Background: branch `ENM_jsats3d_edits_clocksync`
(four-arm tool) used event SD across holds/drags; earlier unjournaled outputs in
`output\11112026_zoi11_cfd01\armZC\spatial_error_pnnl\` (no code in repo).

## Decisions (project lead)
- Data: fish run `output\11192026_premedian` (4 tags, 11 positioning receivers).
- SD = per-transmission spread of the agreeing solutions (x_std/y_std/z_std of fish_positions.csv; std of the
  solutions accepted by the receiver-set consensus). Horizontal = sqrt(x_std^2 + y_std^2).
- 1 m grid; smoothed contour version as well, masked to data coverage; validate against drag GPS.
- spatial_sd_grid.csv keeps easting_m/northing_m; fish_positions.csv drops them (project lead).

## Code (`scripts/fish_figures_2025.py` only)
- `write_spatial_sd()`: 1 m grid over the 1st-99th percentile of positioned fixes + 5 m; each node = median and p90
  of SD of fixes within 5 m (moving window, cKDTree); nodes with < 5 fixes left blank (no extrapolation).
  Outputs: `spatial_sd_grid.csv` (x_m, y_m, n_fixes, {x,y,z,h}_sd_median_m, {x,y,z,h}_sd_p90_m, in_convexhull,
  easting_m, northing_m), `spatial_sd_summary.csv` (percentiles per axis: all / inside / outside hull),
  `spatial_sd_{x,y,z,h}.png` (pixel) and `_contour.png`. Colour cap = 95th percentile of the gridded medians
  (stated on colour bar). Receiver hull and positioning receivers (red) drawn.
- `write_fish_positions` now returns (table, hull receiver ids); easting_m/northing_m removed from fish_positions.csv.
- Constants SD_GRID_M = 1, SD_RADIUS_M = 5, SD_MIN_POINTS = 5 documented in code.

## Results (9,317 positioned fixes)
| Region | X median / p90 | Y | Z | Horizontal |
|---|---|---|---|---|
| all | 1.41 / 3.40 m | 1.70 / 4.14 | 2.70 / 4.94 | 2.49 / 5.22 |
| inside hull (6,398) | 1.29 / 3.02 | 1.57 / 3.69 | 2.70 / 4.93 | 2.20 / 4.72 |
| outside hull (2,919) | 1.83 / 4.09 | 2.15 / 4.79 | 2.69 / 4.97 | 3.20 / 5.93 |
- 7,954 grid nodes with data. Lowest SD in the array interior (ZOI01-ZOI03-ZOI08); highest at the edges/corners.

## Validation against drag GPS truth (one-off, read-only)
- Drag DB `output\tagdrag\gps_smooth_2026-11-19\raw\tagdrag_gps_raw.db` (06-10, ENT + FBY-D, FC36/FFD3, 17 receivers
  incl. CFDs), same water gate + consensus; truth from `drag_test_figures_2025.truth` (project GPS timing).
  A first attempt with naive interpolation of the minute-rounded GPS timestamps gave a spurious 14 m median error -
  discarded.
- 2,010 fixes: median horizontal error 4.97 m vs median SD 4.66 m (median error/SD 1.07); error <= SD for 46%,
  <= 2 SD for 72%.
- Per-fix rank correlation SD vs error: 0.19 (weak). By SD quintile median error 4.9 / 3.4 / 3.7 / 5.3 / 9.1 m;
  p90 17-26 m in every quintile. 5 m cells (169 with >= 5 fixes): cell SD vs cell error rank correlation 0.33.
- Interpretation: SD has the right magnitude as a typical error and is a moderate indicator of where error is
  higher, but it measures precision (agreement), not accuracy: low SD does not exclude large errors (consistent but
  biased solutions, e.g. clock offsets shifting all combinations). Use it as an approximate, lower-bound spatial
  error rate; roughly double it for a ~70% bound.

## Parameter changes
- None to positioning. New presentation constants above.

## Files touched
- `scripts/fish_figures_2025.py`, this journal. Figures regenerated in `output\11192026_premedian\figures\`
  (folder had been deleted by accident and was rebuilt; first rerun died silently after 26E4 PNGs, second completed).
- No temporary scripts/files.
