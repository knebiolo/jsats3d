# 2026-11-20/21 - Deng receiver cap + outside-array filter (all 20 receivers)

## Why
- The project lead wants all 20 receivers (ZOI01-11, CFD01-09) used for 3D positioning, so fish can be positioned
  across the whole array, CFD side included.
- Deng solves every 4-receiver combination of the receivers that heard a ping. That is 330 combinations with
  11 receivers and 4,845 with 20. The uncapped 20-receiver run (11-20 10:05) positioned only 15 tags in 7 h, one of
  them taking 4 h 40 min, and was a disk risk. It was stopped and its positions were cleared.
- Many 20-receiver solutions fall outside the array, where the receivers do not surround the fish (poor geometry,
  i.e. extrapolation).

## Test (11-20 evening, temporary harness, since deleted; no script changes)
- Tags 0C43, A230, 8505, 5AF2 and 70EF (random, seed 20261121, from tags with a pulse rate).
- Each ping keeps its 8 earliest receivers. If none of them is deep (ZOI01-03), the 8th is swapped for the earliest
  deep receiver that heard the ping, because shallow-only sets give poor Z.
- Full season with the cap: Deng took 46-295 s per tag (uncapped, up to 4 h 40 min per tag).
- Same 150 sampled pings, cap-8 vs all combinations:
  - 3-7x fewer combinations.
  - 4-10% fewer pings positioned.
  - Median XY difference 0.3-0.5 m and median Z difference 0.02-0.34 m. The exception, 0C43 (mostly in the
    poor-geometry area), had a 2.5 m median and 15 m p90.
- Many positions fell outside the 20-receiver hull (0C43: 4,295 of 5,428), mostly south-west of ZOI11/CFD01/CFD02.
- Drag GPS validation was not run: the 06-10 drag DB had been deleted, and the project lead said to skip it.
- Outputs: `output\11202026_100fish\cap8_test\` (`cap8_summary.csv`, `cap8_vs_full_sample.csv`, `<TAG>_cap8_plan.png`).

## Code (project lead approved 11-20 22:35; off by default, legacy behaviour unchanged)
- `scripts/_2025_pipeline.py`, step 11 `positions_run`. New `[legacy]`/`[positioning]` options read by `deng_settings`:
  - `deng_max_receivers` (0 = off): `cap_receivers` keeps each transNo's N earliest detections (by seconds_fix).
  - `deng_keep_deep` (default true): with the cap, swap in the earliest deep receiver when none of the N is deep.
  - `deng_hull_filter` (default false) and `deng_hull_margin_m` (5.0): `reject_outside_hull` re-marks solved rows of
    the per-tag Deng CSVs more than the margin (XY) outside the `receivers_3d` hull as
    `outside receiver array - rejected`. Downstream steps all select `comment = 'solution found'`.
- `positions_run` now builds `jsats3d.position` directly instead of calling `legacy_pipeline.deng` (same call,
  `water_column` false), so the cap can be applied to `tag_data`. `jsats3d.py` and the 2019 pipeline are unchanged.
- Per-tag notes record deep_swapped, deng_s, A_found, B_found and outside_array.
- `tests/test_deng_options.py` adds 3 tests (cap order, deep swap, hull marking); 9 of 9 pass.
- Checked on real data: 70EF gives 5 deep swaps and 71,906 combinations, the same as the harness.
- The pre-median box (5 m outside the same hull) overlaps the hull filter. The hull filter also works with the
  pre-median filter off.

## 100-fish rerun with the cap (`config/run_data_100chinook_11202026.toml` + cap 8, keep deep, hull filter 5 m)
- Started 11-20 22:40 with `_2025_pipeline.py --from positions --no-drag-plots` on the existing DB (steps 1-8 kept),
  then `fish_figures_2025.py --out output\11202026_100fish\figures`. Console log: `cap8_rerun_console.log`.
- Positions: 90 tags in 356 min. Only 183 min of that was Deng solving; the rest was reading each tag's detections
  (20 un-indexed queries on the 42 GB DB). An index on tblDetectionFilterSecondary (Tag_ID, Rec_ID) is a possible
  future speed-up (not done).
- Comparison: the 11-19 run with 11 receivers, uncapped, took 388 min for positions.
- 10,749,353 solutions rejected as outside the array.
- Export 15 min (14,540,809 solutions found). Bathymetry check 36 min.
- Pre-median 31 min: 9,505,168 rejected (box 9.09M, mostly its Z limits; epoch outlier 153k; unsupported 261k).
  Highest receiver rejection shares: CFD05 80%, CFD06/07 75%, CFD09/CFD02 74%.
- positions_dbscan 5 min. Pipeline finished 11-21 06:02 (exit 0). Figures were running at 06:08.
- Disk: 41 GB free.

## 2026-11-21 16:50 - full rebuild for the 85 tags from the 11-19 run
- The project lead wanted the same positioning on the tag IDs in K:\...\2025_output_results\100 fish (85 tags from
  the 11-19 run). Only 14 of them were in the 11-20 DB, and the 11-19 DB had been deleted, so the project lead chose
  a full rebuild that replaces the 11-20 DB.
- New run file `config/run_data_100chinook_11212026.toml`: a copy of the 11-20 file with those 85 tags (same settings
  and output DB path).
- Deleted the 11-20 DB, its status/run json and the `_legacy` work folder.
- Old figures moved to `output\11202026_100fish\figures_prev_90tags_cap8_11202026`.
- Launched `run_data.py` detached at 16:47 (parse, index, pipeline, figures). Log: `run85_console.log`.
  108 GB free at launch.
- 11-22 06:15: positions took 538.5 min (85 tags); export 24.4 min (106,905,434 rows, 19,168,869 found). Only 24.6 GB was free, so with the project lead's OK the 13.5 GB of per-tag position CSVs in the _legacy\positions work folder were deleted (already loaded into the DB).
- 11-22 16:30: the bathymetry check (step 10b) had run for about 10 h. Last night's run took 36 min.
  - `jsats3d/bathymetry.check_positions` does 19.2M single-row UPDATEs in one transaction, with no progress output.
    The process had read about 11.5 TB, likely cache thrashing with the 64 GB DB.
  - Its bathy_z / bathymetry_check columns are not used by any later step (pre-median, DBSCAN, figures).
  - The project lead chose to stop it and resume. The uncommitted updates rolled back cleanly; the columns exist, so
    the step counts as done, and the bathymetry flags are empty for this run.
  - Resumed at 16:33 with `_2025_pipeline.py --from premedian --no-drag-plots`, then figures. Log: `run85_resume_console.log`.
  - Possible follow-up (needs approval): vectorize check_positions (KD-tree query on all rows, one bulk update).
- 11-22 18:04 resume outcome:
  - premedian 42.3 min: 13,599,826 of 19,168,869 rejected (box 12.79M, outlier 238k, unsupported 572k). CFD
    rejection shares were 80-81%.
  - positions_dbscan FAILED after 6.0 min: MemoryError inside sklearn `dbscan_inner` (positions_dbscan_run line 922),
    i.e. the neighbourhood lists of one large tag (likely B3A1/A963) did not fit in memory. The DBSCAN table was not
    built. fish_figures does not use it.
  - Figures ok: 85 plan_time + 85 depth_time.
  - fish_positions.csv: 2,779,837 rows, 512,471 positioned. The 11-19 run (same 85 tags, 11 receivers, uncapped)
    had 267,836 positioned.
  - Spatial SD median x/y/z 1.79/2.07/3.30 m. The status json still shows bathymetry_check "running" (it was killed).
