# 2026-11-19 - CFD receiver GPS rolling-filter test (5 and 15 min) on the 06-10 tag drags

## Session metadata
- Author: Copilot, AI assistant using Copilot SDK in VS Code.
- Timestamp: 2026-11-19 ~11:45 PST (UTC-08:00) / ~19:45Z.
- Tags: receiver-gps, cfd, tag-drag, validation, pm-request, experiment.

## Request
PM: compare positions from raw interpolated CFD02-CFD09 GPS against a 5-min and a 15-min rolling filter
applied to that GPS first, plotted per validation drag in the ZOI11/CFD01 four-arm style.

## Findings before running
- Receiver GPS enters only the Deng position solve (`jsats3d.position.receiver_position_at`, linear
  interpolation of `tblReceiverGPS`). The existing run file `run_data_tagdrag_ent_c.toml` had no CFD
  receiver in `receivers_3d`, so a GPS filter could not change any position there. Project lead approved
  adding CFD01-CFD09 to the 3D set for all arms (CFD05 omitted: no 2025-06-10 raw file).
- GPS time basis checked: raw CFD03 GPS Fix rows are UTC (18:18:25); `master_df_gps.csv` holds the same fix
  at 11:18:25, i.e. PDT wall clock, same basis as detections. No offset problem.
- CFD01 has no GPS (static). CFD09 GPS starts 2025-06-11 (static on 06-10). Filtered receivers with GPS on
  06-10: CFD02, 03, 04, 06, 07, 08 (~1 fix/min; 1-7 m std over 09:00-13:00).
- Another process was rebuilding `output\tagdrag\jsats3d_2025_tagdrag_ent_anchorZOI03.db` from scratch and
  draft scripts (`scripts/tagdrag_gps_smooth.py`, `gps_smooth_*.py`) rewrite `tblReceiverGPS` in place.
  At the project lead's instruction the rebuild (PID 6464) was stopped; that DB is now an empty 8 KB file and
  must be rebuilt if needed. None of the draft scripts were used.

## Method
- New run file `config/run_data_tagdrag_gps_smooth.toml`: copy of `run_data_tagdrag_ent_c.toml` with
  CFD01-04,06-09 added to `surface_receivers` (clock fix) and `receivers_3d`; `known_depth_diagnostics=false`.
- Arm 1 (raw): full `run_data.py` build -> `output\tagdrag\gps_smooth_2026-11-19\raw\tagdrag_gps_raw.db`.
- Arms 2/3: byte copies of arm 1; `tblPositions_Deng*` dropped; CFD02-CFD09 `tblReceiverGPS` X/Y replaced by a
  **centered, time-window (5 or 15 min) rolling mean, min_periods=1**, per receiver; then
  `_2025_pipeline.py --db <copy> --from positions`. Detections, clock fix, solver settings identical
  (verified: GPS timestamps and non-CFD rows unchanged; secondary detection rows 63,120 in all arms).
- Smoothing shift of CFD positions in the drag window: 5-min median 0.28 m (p95 1.30, max 8.08);
  15-min median 0.62 m (p95 2.45, max 7.69).
- Comparison: `gps_smooth_three_arm.py` (copy of `output/11112026_zoi11_cfd01/c4/zoi11_cfd01_four_arm.py`,
  arms changed only; same project scoring: consensus -> smoothing -> Kalman for drags, receiver-set
  consensus for holds; XY error to GPS truth). Output in `...\gps_smooth_2026-11-19\compare\`.

## Results (36 event x tag pairs: ENT-01..05, FBY-D-1..4, FBY-S-1..10 holds; FC36, FFD3)
- Drag median-of-median XY error: raw 3.1 m, 5-min 3.2 m, 15-min 3.2 m. Holds: 4.6 / 4.6 / 4.7 m.
- 5-min: median error within 0.5 m of raw on 32/36 pairs (rose >0.5 m on 1, fell on 3).
- 15-min: within 0.5 m on 31/36 (rose on 3, fell on 2).
- Fix counts change by a few transmissions either way (2243 / 2240 / 2255).
- Conclusion: neither filter measurably improves or degrades positions. Expected, since the filters move
  the CFD hydrophones by well under a metre at median, below the ~3 m drag error.

## Parameter changes
- None to production. Experimental only: 5 and 15 min centered rolling mean on CFD GPS in copies.

## Run times
- Raw arm: parse + clock sync ~8 min, 3D positions (17 receivers) 6.1 min, total ~16 min.
- Filtered arms: ~9.5 min each (run in parallel).

## Files touched
- Added: `config/run_data_tagdrag_gps_smooth.toml`, this journal.
- Output (gitignored): `output\tagdrag\gps_smooth_2026-11-19\` (3 arm DBs, logs, `gps_smooth_three_arm.py`,
  `compare\gps_tables.md`, `event_scores.csv`, `fixes.csv`, `plots_2d\*.png`, `plots_3d\*.html`, `index.html`);
  pipeline review plots under `output\2025_review\tagdrag_gps_{raw,f5,f15}[_dbscan]`.

## Open items
- 06-05 drags (below) are 2D only; 06-10 drags are 3D. Not directly comparable.
- Untracked draft/diagnostic scripts left by another session in repo root (`check_*.py`, `reset_*.py`,
  `drop_positions.py`, `backup_5min.py`, `verify_5min.py`) and `scripts/` (`gps_smooth_analysis.py`,
  `gps_smooth_compare.py`, `tagdrag_gps_smooth.py`, `check_cfd_gps.py`): not part of either pipeline;
  project rule says remove. Not deleted here (not this session's files) - project lead to confirm.
- `output\tagdrag\jsats3d_2025_tagdrag_ent_anchorZOI03.db` is empty after the stopped rebuild.
- Adding CFDs to `receivers_3d` for the drag runs is itself a change worth a separate A/B if adopted.

## Follow-up 12:10 PST - 2025-06-05 forebay approach drags (FBY-NS-01/02, FBY-SN-01/02)
Request audit found the first pass covered all 06-10 drags (ENT-01..05, FBY-D-1..4) but missed the four 06-05
drags with GPS truth. CFNSC drags have no GPS track (cannot be scored). Upstream holds UP1/UP2 (06-05) are holds,
not drags, and are not read by the comparison tool.

- Run file `config/run_data_tagdrag_gps_smooth_0605.toml`; outputs `output\tagdrag\gps_smooth_2026-11-19_0605\`.
- Window 2025-06-05 10:30-13:30 (sync to 17:00). Parse: 22,039 detections; only 8 receivers recorded:
  ZOI05, ZOI06, CFD02, 04, 06, 07, 08, 09. ZOI08 (master) first raw file starts 06-09 23:00; ZOI01-03 absent.
- **Decisions (project lead, this window only):** clock master **ZOI05** (beacon 7F91 heard by all 8; 551 self,
  >=181 at every other receiver); DBSCAN anchor **CFD07** (most 7F91 detections after ZOI05); deep receivers none.
- **Geometry:** all 8 hydrophones at Z -3.05 to -3.10 m (5 cm span) -> 3D Deng Z undetermined (planar array).
  Scored product is **2D Deng with tag Z fixed at tow depth** (FC36 -1.524 m, FFD3 -4.572 m), XY error vs GPS.
  Comparison script copy gained `--solver 2D` (uses tblPositions_Deng2D root B in place of the 3D table).
- Filtered receivers with GPS on 06-05: CFD02, 04, 06, 07, 08 (CFD09 static). Shift in drag window: 5-min median
  0.43 m (p95 2.02, max 4.88); 15-min median 0.77 m (p95 2.90, max 7.84).
- Results (fixes / median XY error m, raw -> 5 -> 15):
  FBY-NS-01 FC36 2/2/1, 27.4/27.2/16.6; FFD3 19/22/20, 25.0/25.2/24.6.
  FBY-NS-02 FC36 10/10/10, 12.4/12.4/12.4; FFD3 19/18/19, 13.9/14.1/13.5.
  FBY-SN-01 FC36 9/10/11, 14.8/17.0/18.0; FFD3 14/16/21, 8.9/8.2/11.6.
  FBY-SN-02 FC36 2/2/4, 51.5/51.4/40.2; FFD3 1/0/0, 28.1/-/-.
- Reading: errors are 9-50 m and fixes few because the 06-05 drags run along the east edge of an 8-receiver
  planar array (largely outside the hull) with a substitute master. Differences between arms are within the noise
  of a handful of fixes; no filter shows a consistent improvement. Same conclusion as 06-10.
- Plot note: receivers not recording on 06-05 are still drawn (black triangles) at configured positions.
