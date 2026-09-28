# Session: 2026-09-28 — Owner Approval List, PM Clock-Reset Question

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-28 09:25 local PDT (UTC-7) = 16:25 UTC
- Tags: #onboarding #owner-review #clock-sync #dbscan #pm-communication #gate-2 #gate-3
- Branch: `ENM_jsat3d_edits` (HEAD `9418edf`, clean; untracked paper PDF not to be committed)

## Active Context
- Database: `output/jsats3d_2025_v2.db` (read-only).
- Env: `jsat_3d`. Last recorded test run: 19 pass (2026-09-25).
- Current focus: consolidate owner (Kevin) approval items before paper step 5 clock correction.

## Onboarding
- Stage A load: System Prompt, `LONG_TERM_CONTEXT.md`, `README.md`, `personality.md`, session files 09-17, 09-22, 09-23 (x2), 09-24, 09-25. No pointer stubs; no degraded digests.
- More than 60 min since last session (2026-09-25): new session.

## PM Message (2026-09-28)
> "wow, sounds like things are behaving. love to hear that. i'm interested in the clock resets and how that lines up the data, because +/- 30 seconds is a huge amount of drift."

### Assessment
- The +/-30 s figure is from the 2026-09-24 first DBSCAN run, where ZOI05 was described as a "free-running clock drifting about +/-30 s". That finding was superseded on 2026-09-25: ZOI05 logs UTC (`00z`), all other target receivers log PDT (`-07z`).
- Mechanism of the artifact: a 7 h (25,200 s) offset is ~402 beacon periods of ~62.7 s. Epoch matching folds the offset into a +/-0.5-period window; small changes in the true beacon period (jitter, drift) move the folded remainder by tens of seconds. This produced the apparent smooth +/-30 s wander. It is not clock drift.
- After the -7 h shift: ZOI05 noise 100% -> 6.0%, clean residual RMS 25 us (two-week run).
- Actual measured clock behaviour (two-week v2 run, 06-17 to 07-01):
  - Within-segment drift: 0.2-0.6 us/s (ZOI07/ZOI08 ~ -0.05 us/s), ~25 us per 60 s ping. Inside the 0.5 ms budget.
  - Clock resets/jumps (ZOI): 0.3-1.1 s steps, 10-24 per day; 78-91% coincide with an ATS Internal flag (chance 3-24%), so jump timing is recoverable from raw files.
  - ZOI02 (beacon host) jumps: mostly +/-1 s, up to +/-5 s, tails near +/-400 ms; only 65% flagged. Cancels in pairwise differencing.
  - Correction approach (paper Eq. 7-10): break series at jumps, fit piecewise linear drift per clean segment. Leave-one-out error ZOI median 4.5-8.3 us, p95 15-20 us.
- Error budget reminder: 1 ms timing error ~ 1 m position error. Uncorrected 1 s jump ~ 1.5 km; corrected residual ~ 20 us ~ 2 cm.
- Action: PM reply must correct the +/-30 s interpretation (earlier messaging called ZOI05 a clock fault).

## Owner (Kevin) Approval List — prepared, not yet sent
1. Reference clock / beacon source: standing ZOI02 decision vs hybrid (ZOI02 beacon, ToT from cleaner clock ZOI09/CFD04/ZOI08) vs ZOI09. ZOI02 is on-bottom and unsurveyed; its position error becomes a constant per-receiver bias. ZOI02 needs a second beacon for its own sync.
2. Clock-correction method: jump location from Internal flags + TDoA magnitude, piecewise-linear drift per segment (paper Eq. 10), output `tblDetectionClockFixed`.
3. Metronome phase-2 multipath: DBSCAN on (t, delta) replacing 2019 KNN. Method substitution vs published workflow.
4. Fixed DBSCAN parameters: TIMING_BUDGET_S 0.5 ms, MIN_SAMPLES 3, TIME_WINDOW_PERIODS 2.5 (changed from 2.0, recorded 09-24), Chebyshev eps = 1 in scaled space, ANCHOR_MAJORITY 0.5, ANCHOR_MIN_RECEIVERS 4, MAX_REFLECTION_DELAY_S 0.25. Note eps derives from timing budget, not System Prompt 7.4 reflection-path derivation (that applies to fish-tag filter).
5. Steady-reflection rule (keep earliest mode per regime; CFD01 ~21.5 ms second mode).
6. CFD receivers: 10-25x noisier clocks, 2.4-8.2% over budget. Include with sigma weighting, or exclude.
7. Sound-speed source: legacy `sos()` comment cites Wikipedia; paper cites Seafloor Systems table.
8. Legacy `temp_interpolator()` datetime change in `jsats3d.py` (2 lines, 2026-09-11): accept or relocate to adapter.
9. Legacy defects (not fixed): `position.Deng()` solution-B in-hull test uses solution-A y/z; `position.__init__` WSEL /3.28084 unconditional; `multipath_classifier()` `SNR > 0` drops all ATS rows. Fix in core, or work around.

## Verification of Approval List and PM Text (09:40 PDT)
- User direction: DBSCAN is the main multipath method, not the legacy KNN `multipath_classifier()`. Item 3 (KNN swap) dropped as already decided.
- Checked against code: `scripts/beacon_pairwise_dbscan.py`, `jsats3d/jsats3d.py`, `git diff 7e3b90e HEAD`. Numbers checked against 09-24/09-25 journals only; the scratch scripts and DBSCAN outputs were deleted 09-25, so no numbers were re-run.
- Confirmed in code: TIMING_BUDGET_S 0.0005, MIN_SAMPLES 3, TIME_WINDOW_PERIODS 2.5, ANCHOR_MAJORITY 0.5 (applied as >=), ANCHOR_MIN_RECEIVERS 4, MAX_REFLECTION_DELAY_S 0.25 (used only by the steady-reflection rule), DBSCAN eps 1.0 Chebyshev on (t / 2.5 periods, delta / 0.5 ms).
- Confirmed: `sos()` cites Wikipedia. A merged comment line disables the ft/s conversion, so `sos()` returns m/s. Unused `sos_apply()` converts with 3.2804 (typo for 3.28084).
- Confirmed: the only `jsats3d.py` change vs `7e3b90e` is the 2-line `temp_interpolator()` datetime conversion (commit `893faea`).
- Confirmed: `Deng()` line 1297 builds the solution-B in-hull test point from `S1b.item(0), S1a.item(1), S1a.item(2)`. The written solution-B coordinates are correct; only its `in_hull` flag is wrong.
- Corrected: WSEL /3.28084 is unconditional only in `position.__init__` (line 1122). `clock_fix_object` (line 782) checks units. The 2025 DB uses feet -> meters, so this has no effect on this study (latent defect only).
- Dropped: `multipath_classifier()` `SNR > 0` is off the DBSCAN path. FYI only.
- Corrected: jumps are detected as steps in the pairwise delta series. Internal flags confirm timing but miss some jumps: ZOI 9-22% missed (ZOI03 62%), CFD 70-83%, ZOI02 35%.
- Corrected: paper Eq. 10 (per 09-25 journal; PDF not re-read, no PDF library in env) is linear interpolation between neighbouring clean beacon epochs within a segment, not one straight line per segment. Straight-line fit RMS 7-11 us (ZOI) vs interpolation LOO median 4.5-8.3 us.
- Corrected: jump count for cleaner receivers is 21-30 (own beacon + clock, 48 h), 20-24 (hybrid), not "about 20".
- Corrected: drift is relative to anchor ZOI09. Long-segment medians 0.04-0.47 us/s in magnitude (not 0.2-0.6), i.e. about 3-30 us per ping.
- Corrected: flag match 78-91% excludes ZOI03 (38%). Jumps/day 9-24.
- Corrected: "after correction" figures are leave-one-out interpolation estimates. No correction has been applied (paper step 5 not started). 20 us x 1,465 m/s = 2.9 cm range, not 2 cm. CFD: median 84-158 us, p95 407-550 us.
- Clarified: the +/-30 s is the edge of the epoch match window (0.5 x 60 s nominal period, `merge_asof` tolerance). The 7 h ZOI05 offset folded into that window.
- CFD/ZOI noise ratio: LOO median 10-35x, depending on the metric.

## Receiver Coordinates Check (config workbook, read-only)
- Source: `K:\...\CowlitzAT2025_Data_Deliverables\1_array_metadata\cowlitz_2025_AT_config.xlsx`, sheet `Configuration_Cowlitz_AT_2025`. Columns include Latitude/Longitude (degrees), Longitudinal Offset (feet), depths. No elevation, coordinate method, datum, or accuracy column.
- All 20 target receivers have lat/long; these are what `tblReceiver` X/Y use (EPSG:26910).
- ZOI01-11: 13-14 decimal places, suggesting they were converted from another coordinate system; method unknown.
- CFD01: 12 decimals. CFD02-09: 5-7 decimals (5 decimals ~1 m), consistent with a handheld/receiver GPS reading at deployment. CFD02-09 also have dynamic GPS (`master_df_gps.csv`).
- Correction to today's PM draft: we have coordinates; the open question is their source and accuracy, not their existence. The "10-81 m spread" refers to the ATS receivers' own GPS rows, not the config coordinates.
- Physics: 1 m position error / 1,465 m/s = 0.68 ms, which exceeds the 0.5 ms budget and looks like a constant clock offset. The paper accuracy test (0.02-0.05 m RMSE) needs survey-grade (cm) coordinates for at least the metronome/reference receiver.

## master_df_gps.csv Check (read-only; scratch in %TEMP%\jsats_gps, deleted)
- File: `4_gps_datasets/master_df_gps.csv`, 1,114,887 rows. Columns: dateTime, receiverName, lat, lon, easting, northing. 60 s fixes, 6-decimal lat/lon (~0.1 m resolution).
- Receivers: CFD02-CFD09 only. No ZOI units, no CFD01. Span 06-03 to 09-17 (CFD09 from 06-11). The drag window 06-05 to 06-10 has CFD02-08.
- Already used: `load_receiver_table()` sets CFD02-09 X/Y to the season median of this file (DB matches to 0.0 m). ZOI01-11 and CFD01 X/Y come from the config lat/long (0.0 m match). Local origin E 567994.003, N 5146184.401 (EPSG:26910).
- Config lat/long vs GPS season median: 3.5-13.1 m apart (CFD).
- Scatter of 1-min fixes about the median: p50 3.3-6.1 m, p95 7.4-15.7 m. Daily-median p95 2.5-10.5 m; daily-median max 8-106 m (likely service/relocation days, not reviewed).
- GPS alone cannot separate real float motion from receiver noise (consistent with 09-25 within-hour sd 0.8-4 m).
- Physics: 0.5 ms budget x 1,465 m/s = 0.73 m. A 3-6 m typical fix error = 2-4 ms, so this GPS supports rough placement and motion screening only, not sync-grade or paper-accuracy geometry.
- Unknowns for PM: GPS device and mounting, antenna-to-hydrophone horizontal offset, whether the large daily jumps are relocations.

## Data Catalog + QC Tracker Review (read-only)
- Files: `CowlitzAT2025_Data_Deliverables/data_catalog_cowlitz_falls_AT_2025.xlsx`; `6_quality_control/data_processing_tracker_cowlitz_2025.xlsx`, `receiver_data_gaps.xlsx`. Provider: Four Peaks (Penny Rowe, Mark Weiland). System Prompt says "Spheros"; confirm naming with owner.
- Config Latitude/Longitude = hydrophone position, WGS84. No survey method or accuracy stated.
- KMZ note: CFD02-09 config positions approximate at deployment; for fine-scale use receiver GPS time series. `master_df_gps.csv` = GPS from the SR3017 receivers on floats. The antenna-to-hydrophone offset is not documented.
- All deliverable detection/GPS timestamps PDT. PTAGIS `collected_v0` Obs Time is PST (1 h offset from PDT); flag for later fish-history work.
- `amp` is raw counts, uncalibrated, not comparable across receivers. Supports the relative-amplitude normalization rule.
- Beacon period "typically 60 s; some high-amplitude tags 30 s". Array-wide beacon periods are still not given per tag.
- "Cleaned" in the tracker = combining split files and removing corrupt/bad-date GPS lines. No multipath removal is described. We parse raw files, so upstream filtering does not affect our DB.
- CFD05: hydrophone connection issue (06-18), bad file (06-26), offline for repair about 06-26 to 08-14, very few study detections. Explains the sparse CFD05 data.
- ZOI05: tracker 06-09 "Discovered the time zone was off for this receiver". Consistent with our UTC finding.

### DEFECT: parser skips ZOI03/ZOI06 daily files (June gap is ours, not a data gap)
- Tracker shows no gap for ZOI03/ZOI06 in June. `raw_data/20250626_Receivers` holds 18 daily files `SR20026D2506xx_*_cleaned.csv` (ZOI03, 9 files, 201 MB) and `SR18084D2506xx_*_cleaned.csv` (ZOI06, 9 files, 76 MB) covering 06-18 to 06-26.
- `SERIAL_PATTERN = ^SR(\d+)(?=_|\.)` in `parse_ats_raw_to_legacy.py` rejects the `D` before the date. `discover_target_files()` skips non-matching files silently (violates fail-loudly).
- Only these 18 files across `raw_data` have this naming.
- Fix requires a parser change plus a DB rebuild or append (owner approval). Not fixed.

### ISSUE: CFD05/CFD09 serial swap during tag drag
- Tracker (06-09/06-10 download): "Config was switched to process CFD05 as 19026 because that's the config during the tag drag - after that we changed CFD05 and CFD09."
- Config: CFD05 = SR19033, CFD09 = SR19026. The parser maps by config serial, so all SR19026 rows are labelled CFD09.
- Supporting evidence: SR19026 new file starts `250611_124001`; `master_df_gps.csv` CFD09 starts 2025-06-11 12:41:20; SR19033 has no 06-10 array-testing file.
- Implication: SR19026 detections before about 06-11 12:40 were likely at the CFD05 location. The DB puts them at CFD09, about 88 m away (tblReceiver X/Y). This is critical for tag-drag validation (06-05 to 06-10).
- Needs PM confirmation of swap time; then a time-dependent serial-to-receiver mapping. Not fixed.

## DDoA Figure (user request)
- Ran `beacon_pairwise_dbscan.py` read-only: ZOI02/7D2D, anchor ZOI09, 2025-06-20 to 06-22 PDT, no rowid bound (full scan). Output `output/dbscan_ddoa_0620/`.
- 72,902 detections, 43,636 first arrivals, 37,319 paired epochs, 85 anchor-side. 15 receivers (ZOI03/ZOI06 absent: parser file-skip defect).
- WARNING (System Prompt 9.2): CFD02/03/04/06/07/08/09 have 35-99 clean epochs with residual > 0.5 ms each. ZOI all 0.
- Figure `ZOI02_7D2D_anchor_ZOI09_DDoA.png`: y = DDoA = c(t_i - t_anchor) in m (legacy `clock_fix()` definition `SoS * TDoA`); dashed line = expected |B-R_i| - |B-anchor| (Euclidean, tblReceiver). Scratch plotting script in %TEMP%.
- Observation: DDoA sits on discrete levels about 1,465 m apart (1 s x c), i.e. whole-second clock jumps. Between jumps, points track the geometric line. ZOI07 spends most of the window about -1,467 m (1 s) off.
- No parameters changed.

## Meeting Notes (2026-09-28, recorded by user)
1. Find which receiver's beacon is heard most across all other receivers.
2. DBSCAN: run a suite of test parameters to find better settings.
3. min_samples 3 = each detection is compared with its neighbours (min_samples 1 = every point is its own group; nothing is noise).
4. Review `notebooks/dbscan_multipath.ipynb`.
5. Correct clock time jumps.
6. Before/after plots, plus plots showing how the parameters work.
7. Order: plots first, then fix time jumps.
8. CFD receivers jump more and are noisier. CFD GPS jumps 50-80 m and needs averaging/filtering: 15-minute window average. Plot GPS metrics through time.
9. Identify the metronome clock; it must be heard by all receivers.

### Policy flag (System Prompt 7.4) — escalate, do not route around
- Kevin's notebook sets `eps` = 99th percentile of consecutive (t, DDoA) Euclidean distances per data set, i.e. per receiver. 7.4 forbids per-group auto-tuned eps.
- A parameter sweep is compatible with 7.4 only if it ends in ONE study-wide value set, chosen against a stated validation metric, recorded with before/after. Proposed validation: leave-one-out clock-interpolation error, noise fraction, segment count, clean > 0.5 ms count, on a fixed set of receivers and windows.
- Notebook mechanics to note: unscaled Euclidean on (seconds, metres). The 37 s grid step dominates, so eps p99 about 40 allows about sqrt(40^2 - 37^2) = 15 m DDoA (about 10 ms) neighbour tolerance. Linear imputation onto a regular grid also creates synthetic points across gaps and jumps.

## Beacon Coverage (meeting item 1)
- One read-only aggregate scan (159 s): per tag x receiver x hour, distinct minutes with detections (about one per ping). Listener operating hours = hours with any detection. Output `output/beacon_coverage/` (matrix CSV, summary CSV, heatmap). Scratch script in %TEMP%.
- CFD05 is effectively deaf: it hears other beacons in only 13-14% of its 820 operating hours (hydrophone issue, per tracker).
- 8 local beacons are heard in >=90% of hours by all 18 other working receivers: B354 (CFD08), B36A (CFD03), FA1B (CFD04), 7DB7 (ZOI08), B35A (CFD09), B2D6 (CFD05), 7D2D (ZOI02), FFC7 (ZOI01).
- Pings per heard hour (max about 57 at 62.7 s): 7D2D (ZOI02) is the most uniform, median 52, minimum 43 (CFD01). B354 median 53 but ZOI06 only 21; FA1B median 51, ZOI06 22. ZOI06 hears most beacons poorly (17-28/h) except 7D2D (47) and FFC7 (45).
- Result: ZOI02's beacon 7D2D best meets "heard by all". This supports ZOI02 as the metronome beacon source; the clock-reference question (ZOI02 jumps) stays open (hybrid option).
- Gate 0 (System Prompt 8): the "array-wide" tags 1F14/1F38/1F5A/1F71/1F94/1FCD are heard in <=2% of hours (max about 30% at CFD06-09/ZOI01-03 for 1FCD/1F94). They are NOT array-wide in the data. 7F32 (30 s, unassigned) is heard >=90% by only 11 receivers, median 27 pings/h.
- Caveats: ZOI03/ZOI06 06-18..06-26 missing (parser defect); SR19026 labelled CFD09 throughout (swap). Hour-level share is coarse; pings/h is the metronome-relevant metric.

## Meeting Items 2/6: DBSCAN Parameter Sweep + Before/After Plots
- Change budget exceeded with user approval ("implement the notes"): 3 script files.
- `scripts/beacon_pairwise_dbscan.py`: `cluster()`, `classify()` and `steady_reflection_labels()` accept `tolerance_s`, `window_periods` and `min_samples`. Defaults = fixed study constants; 19 tests pass; default outputs unchanged.
- New `scripts/dbscan_parameter_sweep.py`: re-clusters an existing `*_epochs.csv` (no DB scan).
  - Grid: tolerance 0.1/0.25/0.5/1/2/5 ms x window 1.5/2/2.5/3/5 pings (min_samples 3), plus min_samples 2/4/5 at window 2.5.
  - Metrics per receiver: noise %, segments, leave-one-out interpolation error p50/p95, % LOO > 0.5 ms, recall of synthetic late arrivals (5% of epochs, log-uniform 1.5-126 ms, seed 0).
  - Legacy notebook rule scored for comparison: per-receiver eps = p99 of consecutive (s, m) distances, Euclidean, no imputation.
- Run on `output/dbscan_ddoa_0620/` epochs (06-20..06-22, 15 receivers). Outputs in `output/dbscan_sweep_0620/`: `sweep_results.csv`, `sweep_summary.csv`, `sweep_heatmaps.png`, `sweep_min_samples.png`, `how_dbscan_works.png`, `before_after.png`.
- Results (group medians, min_samples 3):
  - Window acts in steps because pings are 62.7 s apart: 1.5 = 2.0 (adjacent ping only), 2.5 = 3.0 (bridges one missed ping), 5 bridges more. Going 2.0 -> 2.5 halves rejection (ZOI 7 -> 5%, CFD 28 -> 13%) with the same LOO.
  - ZOI tolerance: LOO p95 flat about 16 us for 0.1-0.5 ms, then jumps to about 457 us at 1 ms (sub-ms steps and small echoes get absorbed). 0.5 ms is the largest safe value.
  - CFD tolerance: a smooth trade-off. Rejection 52/28/13/3% vs LOO p95 108/235/444/738 us at 0.1/0.25/0.5/1 ms. 0.5 ms keeps LOO p95 inside the budget (3.1% of epochs over).
  - Recall of planted echoes is 100% for tolerance <= 1 ms and falls at 2-5 ms (90-94% / 72-80%). It does not discriminate below 1 ms because planted delays are >= 1.5 ms.
  - min_samples: 2 rejects slightly less than 3 (ZOI 3.3 vs 4.5%, CFD 10.3 vs 13.3%) with the same LOO; 4-5 reject more with no gain. Keep 3 (legacy value, needs neighbours on both sides).
  - Legacy notebook rule: per-receiver eps 125-1,466 m (clock jumps of 1 s = 1,465 m dominate the p99). Catches 0-8.5% of planted echoes; LOO p95 1.2 ms (CFD) / 3.9 ms (ZOI). Not usable on 2025 data unless clock jumps are removed first. For Kevin.
- Conclusion: current fixed setting (0.5 ms, 2.5 pings, min_samples 3) sits at the knee for ZOI and inside the budget for CFD. No change proposed.
- Parameter changes: NONE.

## Meeting Item 8: CFD Float GPS (15-min averaging)
- New `scripts/cfd_gps_diagnostics.py`: reads `master_df_gps.csv` (read-only, chunked), bins to 15 min, writes mean and median per bin, within-bin RMS scatter, |mean - median|, bin-to-bin step, and distance from season median. Outputs in `output/cfd_gps/`: `cfd_gps_15min.csv`, `cfd_gps_summary.csv`, `cfd_gps_track.png`, `cfd_gps_quality.png`.
- Within a 15-min bin, GPS scatter is small: RMS p50 0.7-1.8 m, p95 2.1-4.7 m (CFD04 highest). Fixes > 50 m from their bin median: <= 0.03%.
- 15-min median wanders between bins: step p50 1.2-3.0 m, p95 4-10 m; CFD04 has 542 steps > 10 m. Consistent with float motion on the mooring, not GPS noise alone.
- The 50-100 m excursions are event-like, not noise:
  - All floats sit 50-110 m away until about 06-04/05 (pre-deployment or placement; config Deployment Date 06-04). CFD03 moves in gradually until about 06-05. This overlaps the tag-drag window start.
  - Excursions near 09-04 (CFD02/06/08, CFD07) and 08-14 / 07-13 (CFD07) line up with download dates in the QC tracker (service visits).
- |mean - median| p95 0.74-1.43 m, which is above the 0.73 m timing-equivalent budget. The median is the safer 15-min summary. Mean vs median needs owner approval.
- Implication to test: the static season-median position is several metres off most of the time. Float motion of metres = ms of TDoA, which the pairwise DBSCAN absorbs as apparent "clock drift" on CFD receivers. Needs a correlation check before the clock-jump fix.

## Plot Style + Cleanup (user direction)
- User: DBSCAN plots should match the legacy notebook. Before = paired first arrivals before DBSCAN; after = clean (final product). Remove code not needed to run. Do not delete K: data.
- `beacon_pairwise_dbscan.py`: replaced the stacked delta plot with `plot_before_after()`. One PNG per receiver: DDoA (m) = c(t_i - t_anchor) vs seconds, before (black) | after (blue), in `<stem>_before_after/`. Re-run 06-20..06-22: identical counts (37,319 paired epochs, 85 anchor-side).
  - CFD04 after-plot shows the notebook-style drift sawtooth (about -5 to +5 m, ramps about 4 h).
  - ZOI08 shows whole-second levels (about 0 and -1,470 m) = clock jumps. The jump fix is needed for continuous curves.
- `dbscan_parameter_sweep.py`: removed the legacy-notebook-rule comparison (result kept in this journal) and the residual before/after plot (superseded). Kept the grid, scores, heatmaps, min_samples plot and how-it-works plot.
- Deleted: %TEMP% scratch scripts (`ddoa_fig.py`, `beacon_coverage.py`) and superseded local figures (`output/dbscan_sweep_0620/before_after.png`, `output/dbscan_ddoa_0620/*_delta.png`, `*_DDoA.png`). Beacon-coverage CSV/heatmap outputs kept.
- K: drive: read only, nothing written or deleted.
- Parameter changes: NONE.

## LLM_Prompts.txt Rewrite (user request)
- Replaced the old Prompt 001 round-trip log (recoverable from git) with a current project handoff: status, source data and headers, header-to-table mapping and join keys, processing commands, key results, known defects, open questions, next steps.
- Schema verified from the v2 DB (`pragma table_info`) and a raw file header (`SR18084_20250618.csv`, read only).
- Documented the name collision: DB `Event` = clock-event flag from the parser; deliverable `event` = 3+ detections in 180 s.

## Pre-push Review
- Reviewed all new/changed code. `beacon_pairwise_dbscan.py` and `dbscan_parameter_sweep.py`: nothing unused.
- `cfd_gps_diagnostics.py`: removed the unused `--start/--end` plot options, fixed the quality-plot title, and removed a datetime `round` warning from the summary print.
- Checks: py_compile OK for all 3 scripts, 19 tests pass, `git diff --check` clean, GPS script re-run OK (K: read only).
- User will commit and push manually. Do not commit `Nebiolo_Meyer_2021 (003).pdf`.

## PM Items (separate from Kevin)
- Surveyed surface-receiver positions; confirm ZOI01-03 static; deployment-day WSE; forebay WSE 06-05 to 06-16; array-wide beacon identities/periods; Spheros upstream filtering docs; approve `UTC_Conv = -7` and provisional study pulse rates.

## Files Touched
- This journal only.

## Decisions & Assumptions
- None new.

## Parameter Changes With Rationale
- None.

## Blockers & Known Limitations
- Unchanged from 2026-09-25 journal.

## Next Steps
1. Send approval list to Kevin.
2. Reply to PM correcting +/-30 s interpretation.
3. Fix parser worker WARNING capture.
4. Paper step 5 clock correction after approvals.
