# 2026-11-19 - Single client output CSV with unpositioned epochs (Output_Columns.xlsx)

## Session metadata
- Author: Copilot, AI assistant using Copilot SDK in VS Code.
- Timestamp: 2026-11-19 ~16:00 PST (UTC-08:00) / ~00:00Z 11-20.
- Tags: output-csv, client-deliverable, fish-figures, presence.

## Request
"Update the output csv edits. Please make it so the output CSV has extra rows for epochs where there are detections
but either the algorithm fails or there just are not enough detections for positioning. This will ensure that fish
are not falsely assumed to leave the area when analyzed later on." Project lead: ONE file for all tags following
`K:\Jobs\5662\001\Analysis\Output_Columns.xlsx` (AT_code, datetime_utc, x_m, y_m, z_m, x_std, y_std, z_std,
n_solutions, in_convexhull, in_3Darea), replacing the per-tag `<TAG>_positions.csv`.

## Decisions (project lead, 2026-11-19)
- Format CSV; file `fish_positions.csv` in the fish-figures output folder; per-tag CSVs removed.
- Position = receiver-set consensus per transmission (median of agreeing root-B solutions within 10 m, >= 2 receiver
  sets, after the surface/depth-floor gate) - the figures still show the smoothed/Kalman track.
- Rows = every transmission of the tag after multipath filtering (tblDetectionFilterSecondary); no position ->
  blank x/y/z/std/hull flags and n_solutions = 0.
- datetime_utc = true UTC: study time (local wall clock stored as UTC) minus tblStudyParameters.UTC_Conv h (-7 -> +7 h).
  Positioned rows use the consensus time of transmission (ToA - T01); unpositioned rows use the first clock-fixed
  arrival (differs from transmission time by the travel time, < ~0.1 s). Millisecond precision.
- x_m/y_m/z_m in the local receiver frame (as figures); easting_m/northing_m (UTM 10N, origin = median of
  tblReceiver easting - X_t) appended after the 11 columns ("both needed").
- x_std/y_std/z_std = standard deviation of the agreeing solutions; n_solutions = number of those solutions.
- in_convexhull = inside 2D (XY) convex hull, in_3Darea = inside 3D (XYZ) convex hull of the receivers the solver
  used (any r0-r3 in tblPositions_Deng).

## Code (main scripts only; no new scripts)
- `scripts/ent_analysis_2025.py` `legacy_receiver_set_consensus`: adds x_std/y_std/z_std keys (additive; existing
  columns and values unchanged).
- `scripts/fish_figures_2025.py`: `write_fish_positions()` + OUTPUT_COLUMNS; main stores each tag's consensus and
  writes the single CSV after the loop; per-tag CSV block removed; docstring updated. Fails loudly if UTC_Conv is
  missing or a positioned transmission has no detection row.
- Nothing else reads `<TAG>_positions.csv` (drag_test_figures_2025.py writes its own per-line CSVs).

## Verification (baseline `output\11192026_premedian`, figures regenerated in `figures\`)
- 34,607 rows, 9,317 positioned: 26E4 3,263/1,454; 9F8C 2,078/145; 1D5A 17,874/4,277; B5ED 11,392/3,441.
  Positioned counts equal fish_qc n_consensus.
- 26E4 study gap 06-28 00:54:59-01:05:47 PDT: 217 rows (1 positioned) instead of nothing.
- 26E4 first positioned row 2025-06-28T06:18:41.533+00:00 = study 06-27 23:18:41 + 7 h (correct).
- Unpositioned rows carry no values besides AT_code/time/n_solutions = 0; positioned n_solutions >= 2;
  median std x/y/z 1.41/1.70/2.70 m; in_convexhull 68.7%, in_3Darea 28.4% of positioned rows. Sorted by tag, time.
- 6 tests pass.

## Known limitations (flagged, not changed)
- "Every transmission" includes likely false detections: rows before PTAGIS release 1D5A 4, 9F8C 19, B5ED 16
  (26E4 not in PTAGIS); transmissions heard by a single receiver 1D5A 677, 26E4 419, 9F8C 434, B5ED 1,360.
  Option: add an n_receivers column or restrict to >= 2 receivers (owner decision).
- Clock-gap events (2026-11-19 26E4 isolation) now appear as unpositioned rows, not missing rows.

## Files touched
- `scripts/ent_analysis_2025.py`, `scripts/fish_figures_2025.py`, this journal.
- Output: `output\11192026_premedian\figures\fish_positions.csv` (old per-tag CSVs deleted there).
- No temporary scripts/files created (one-off checks piped to python).

## Follow-up 16:05 PST - status and n_receivers columns (project lead request)
- Appended after easting_m/northing_m: `n_receivers` (distinct receivers that heard the transmission, after multipath
  filtering) and `status`, checked in order: `positioned`; `too few receivers` (< 4 of the positioning receivers -
  the receivers used in tblPositions_Deng - heard it); `solver failed` (no root-B Deng solution); `out of water`
  (every root-B solution has Z > 0 or Z < depth floor); `solutions disagree` (in-water solutions but < 2 receiver
  sets within 10 m).
- `write_fish_positions(..., floor_m)` now takes the resolved depth floor (same gate as the figures).
- Baseline result (34,607 rows): positioned 9,317; too few receivers 11,202; solver failed 7,153;
  solutions disagree 3,790; out of water 3,145. Per tag equals the earlier read-only breakdown.
  Single-receiver rows: 2,890. Positioned rows always have x; non-positioned never. 6 tests pass.
