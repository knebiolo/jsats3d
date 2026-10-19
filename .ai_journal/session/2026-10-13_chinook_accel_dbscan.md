# 2026-10-13 — Chinook tracks (5 fish), acceleration DBSCAN, PM feedback

Sessions: 2026-10-12 ~19:30-22:30 (Chinook build) and 2026-10-13 ~19:00-20:30 (plots, acceleration filter), PDT. Branch `ENM_jsat3d_edits`.
Chinook DB `output/jsats3d_2025_chinook_aug.db`; run file `config/run_data_chinook_aug.toml`.

## TL;DR
1. Drew's request: position "a handful" of Chinook with what we have. Five fish ran through steps 1-12 (steps 2-11; drag plots skip themselves): 5939, B7D1, B400, 4CA5, 38B7 (released 08-05 / 08-12, window 2025-08-06 17:00 to 08-09 00:00 UTC basis).
2. Found and fixed three bugs on the way (digit-only Tag_ID, DB lock in step 8, mistyped secondary table). Details below.
3. Found a serious problem: with `adopt_deep_z = 'solved'` the deep receivers landed ABOVE the water (ZOI01 +22.9 m, ZOI02 +2.4, ZOI03 +6.5; configured -11.2/-12.4/-10.3). Surface beacons are coplanar, so deep Z is unobservable. Chinook run file now uses `adopt_deep_z = 'config'`; rebuild from step 6b is running (positions step ~97 min). Earlier Chinook tracks/plots were from the faulty-depth run and were deleted.
4. PM (2026-10-13): journals look fine, agrees to rent an RTK for receiver positions, asked if least squares improved precision/accuracy (answer: no, ties Deng), and set direction: from here DBSCAN position filtering using ACCELERATION, not velocity.
5. Acceleration DBSCAN chosen from tag-drag GPS (ENT-01..05, two DBs): at ~70% of pings kept, XY p90 8.5 m vs 11.2 m for a distance screen at 73%; medians equal (~2.7 m). At the old 84% yield it is no better than the current screen. It is now an option in the pipeline (`dbscan_mode = 'acceleration'`); default stays `'distance'`.

## Chinook selection (discussed)
- 419 candidate Chinook released 07-01..08-30; only tags released from 06-27 have the full array (ZOI03/ZOI06 have no raw files ~06-17 to 06-27).
- Census from `master_df_study.csv` (3.5 GB, read-only on K:): pings heard by >=4 ZOI receivers incl. a deep one, per tag. 353 tags have >=1 such ping; the five chosen have 7,000-12,000 each inside one ~30 h window.
- Pulse periods are NOT in any deliverable (only drag tags had provisional rates). Estimated by regression of multi-receiver ping times on pulse count over contiguous runs (+-0.004 s): 5939 3.1396, B7D1 3.1099, B400 3.0103, 4CA5 3.0806, 38B7 3.1497 s. Written to `[study] pulse_rates`. NEEDS PROJECT LEAD CONFIRMATION.
- Note: master_df times are UTC; the run window ended up applied on the UTC basis (all five tags still fully covered, B7D1 only just: first detection 18:28 UTC).
- ZOI04 left out of `receivers_3d` to match the validated drag setup (it existed by August; could add geometry).

## Bugs found and fixed
1. **Digit-only tag code (5939) read as integer.** `multipath_data_management` (and `epoch_fix_data_management`) use `pd.read_csv` without dtype, so Tag_ID 5939 became an integer and every `Tag_ID = '5939'` query returned nothing (step 11: "no filtered detections for 5939"). Fix: `dtype={'Tag_ID': str}` in both readers in `jsats3d/jsats3d.py`. No effect on alphanumeric codes (2019 tags unchanged). Same bug would hit codes like `1E22` (float).
2. **Database locked in step 8.** `fish_multipath_run` called `table_names(c.db)` while its own connection held a large pending DELETE. Fix in `scripts/tagdrag_2025_pipeline.py`: read the table list before opening the connection.
3. **Mistyped `tblDetectionFilterSecondary`.** The empty 5939 frame from bug 1 created the table with all-TEXT columns, so every time column became text and step 11 failed in `pd.to_datetime(unit='s')`. Dropped the table (local DB only) and reran step 8; fixed at the root by bug 1.

## Outputs and plots
- `output/chinook_aug_tracks_rootB.csv` (per-transmission root-B medians) came from the faulty-depth run: obsolete.
- New `scripts/plot_fish_tracks.py`: one 3D line per fish in time order (smoothed, filtered), start/end markers, receivers shown; writes `output/2025_review/<out>/<tag>_track.png` only. Filter: set has a deep receiver, inside the array box, depth between -15 m and 0, >=2 good solutions per ping; 120 s rolling median; line not drawn across gaps or >40 m jumps.
- User asked to clear ALL files in `output/2025_review/chinook_aug/` (21 files, my own plots/CSVs) and show only the lines; folder is empty until the corrected rebuild finishes.
- Pre-fix observations (not to be trusted for depth): fish depth bimodal (-25 m / +3 m), tags parked near X 445, Y 110 (ZOI01/04/10).

## PM question: did least squares help?
No. Drag lines: fixed-Z NLS 2.83 m median (Deng 3.01 m); static holds in-hull 2.71 m vs 2.55 m (p90 7.35 vs 5.22 m). Free-Z NLS was far worse (13.8 m). Paper's 6 cm is precision, not accuracy. Clocks are not the limit (post-fix <=0.21 ms all receivers, ZOI01-06 ~1-5 us). Remaining limits: receiver/hydrophone positions (units are 16-30 m from ZOI01-03 hydrophones; RTK helps), deep-Z geometry, multipath, towed-tag layback. Draft reply to PM was given.

## Acceleration DBSCAN (tag-drag comparison)
Same per-transmission mean positions for every variant, scored vs GPS on ENT-01..05 (DBs `..._ent_anchorZOI03.db` solved Z and `..._ent_zconfig.db`):

| Filter | kept | XY median | XY p90 | 3D p90 |
|---|---|---|---|---|
| none | 88% | 3.35 | 18.5 | 20.7 |
| current distance DBSCAN (5 m, 5) | 84% | 3.11 | 13.8 | 17.9 |
| distance DBSCAN (5 m, 10) | 73% | 2.75 | 11.2 | 14.5 |
| acceleration DBSCAN (eps 8, 5 pts, accel scale 5, depth weight 1) | 70% | 2.72 | 8.5 | 11.4 |
| acceleration limit <= 2 m/s2 | 68% | 2.71 | 8.3 | 12.0 |

- Acceleration feature = horizontal second difference from the ping before and after (neighbours <= 12 s apart); DBSCAN features = [X, Y, depth * weight, accel * scale].
- Gains are in the tail (about 25% at equal yield); the median floor ~2.7 m includes towed-tag layback and ~1.2 m GPS truth. Settings were tuned and scored on the same drag data (optimistic); both DBs agree.
- A spike also inflates the acceleration of its two neighbours, so each spike costs ~3 pings.

## Code changes (all tracked unless noted)
- `jsats3d/jsats3d.py`: Tag_ID read as text in two data-management readers.
- `scripts/tagdrag_2025_pipeline.py`: step 8 lock fix; `ping_acceleration()`; `positions_dbscan_run` now supports `dbscan_mode = 'distance'` (default, identical to before) or `'acceleration'` with `dbscan_eps_m` (default 8 in acceleration mode), `dbscan_min_samples`, `dbscan_accel_scale` (5), `dbscan_depth_weight` (1), `dbscan_accel_gap_s` (12). Notes record the mode.
- `config/run_data_tagdrag_ent.toml`: `dbscan_mode = 'acceleration'` + tuned values (eps 8). Existing ENT DB still holds the distance-screen table until step 11c is rerun.
- `config/run_data_chinook_aug.toml` (new, untracked): Chinook window/tags/pulse periods, `adopt_deep_z = 'config'`, acceleration DBSCAN.
- `scripts/plot_fish_tracks.py` (new, untracked).
- `tests/test_resume.py`: two new tests (acceleration function; acceleration mode drops a spike and rejects an unknown mode). Full suite: 57 tests OK.
- Script cleanup (user request): deleted six scripts not used by the pipeline or the 2019 run: `clock_fix_serial.py`, `coordinate_with_Deng.py`, `metronome.py`, `mulitpath.py`, `tag_drag_RMSE.py` (original 2022 author drivers with hard-coded paths on another machine; `legacy_pipeline.py` reimplements them and does not call them) and `extract_dbscan_features.py` (diagnostic only; its one test removed from `tests/test_2025_contracts.py`). All recoverable from git history. Remaining scripts: `adapt_2025_to_legacy`, `beacon_pairwise_dbscan`, `ent_analysis_2025`, `legacy_pipeline`, `parse_ats_raw_to_legacy`, `plot_fish_tracks`, `run_data`, `tagdrag_2025_pipeline`. Suite now 56 tests OK. `legacy_pipeline.py` docstrings still name the deleted scripts as provenance.

## 2019 / legacy compatibility
- All new behaviour is opt-in or confined to the 2025 ATS scripts: `tagdrag_2025_pipeline.py` is not used by the 2019 run (`run_data_2019.toml`, `legacy_pipeline.py`); `positions_dbscan` is off unless the run file turns it on, `dbscan_mode` defaults to `'distance'`, `adopt_deep_z` defaults to `'solved'` (legacy behaviour).
- The only change in the shared core (`jsats3d/jsats3d.py`) is reading Tag_ID as text, which keeps 2019 alphanumeric tags unchanged and only fixes numeric-looking codes.
- Not done: a fresh full 2019 rerun. Tests that import 2019 data and the position class pass.

## Decisions still open (owner)
- Confirm Chinook pulse periods; confirm `adopt_deep_z` for production (solved gave deep receivers above the water on Chinook; drag also had ZOI02 4.3 m off).
- RTK survey of ZOI hydrophones (PM agrees); deep-receiver depth survey.
- Adopt acceleration DBSCAN settings for production (eps 8 / 5 points / scale 5 / depth weight 1); retune once RTK positions or more truth exist.
- Whether to add ZOI04 to `receivers_3d`.
- Review `LLM_Prompts.txt` diff (670 lines, still unreviewed).

## Next
1. ~~Wait for Chinook rebuild~~ DONE 2026-10-13 ~21:55: positions step 101.6 min, export ok. Dropped the distance-screen `tblPositions_DengDBSCAN` and reran step 11c in acceleration mode (kept: 5939 8,784/10,811; B7D1 8,853/10,134; B400 5,598/6,893; 4CA5 4,561/6,434; 38B7 4,692/6,399 transmissions).
2. ~~Plots~~ DONE: `scripts/plot_fish_tracks.py` now reads `tblPositions_DengDBSCAN` by default (`--table` to change); five `<tag>_track.png` in `output/2025_review/chinook_aug/`. After the fix the deep receivers sit at -10 to -12 m and fish depths fall between -2 and -14 m. Tracks are tangled loops around the ZOI01/04/10 corner (X ~440-470, Y ~110-140), not clean paths: expect +-3 m median / ~8 m p90 per ping.
3. ~~Rerun ENT step 11c~~ DONE (see below).

## Follow-up (2026-10-13 ~22:00): "do all"
1. **Plots.** `scripts/plot_fish_tracks.py` now draws two panels per fish: the 3D line and a top-down map (both coloured by hours since start, green circle = start, magenta square = end, receivers labelled). Two sets written: `output/2025_review/chinook_aug/` (120 s median) and `output/2025_review/chinook_aug_300s/` (300 s median, easier to read slow movement). 5939 at 300 s: starts near ZOI03 (East ~500, North ~85), wanders through the array interior for ~17 h and ends beside ZOI01/ZOI10 (East ~440, North ~120).
2. **Tag-drag confirmation of the acceleration screen inside the pipeline.** Non-destructive: copied the ENT DB to `output/jsats3d_2025_tagdrag_ent_accel.db`, dropped `tblPositions_DengDBSCAN`, ran `tagdrag_2025_pipeline.py config/run_data_tagdrag_ent.toml --db <copy> --only positions_dbscan plot_drag_dbscan` (log `output/run_ent_accel_dbscan.log`, plots/scores in `output/2025_review/jsats3d_2025_tagdrag_ent_accel_dbscan/`). Original ENT DB and its distance-screen folder untouched. Kept 2,199 of 3,040 transmissions (72%).
   Scores from `item12_pipeline_summary.csv`, "3D legacy B centroid", all ENT-01..05 FC36+FFD3 rows (weighted by scored transmissions):

   | Screen | scored | XY median | XY p90 | 3D p90 |
   |---|---|---|---|---|
   | distance (5 m, 5 pts) | 530 | 3.16 m | 13.7 m | 17.2 m |
   | acceleration (eps 8, 5 pts, scale 5, depth 1) | 451 | 2.98 m | 9.4 m | 12.2 m |

   Weighted means of per-line medians/p90s (approximation), but the direction and size match the offline grid search (p90 down ~30%, median ~0.2 m, ~15% fewer pings kept). Worst row stays ENT-02 FC36 (5.4 m median; towed-tag layback).
3. Open: pulse periods and `adopt_deep_z` still need owner sign-off; RTK survey; the 2019 full rerun was not repeated (legacy code paths untouched by these changes).
4. Extra files created today beyond code: `output/jsats3d_2025_tagdrag_ent_accel.db` (0.38 GB copy), `output/run_chinook_aug_*.log`, `output/run_ent_accel_dbscan.log`; temp helper scripts live in `%TEMP%` only.
