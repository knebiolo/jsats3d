# Session: 2026-10-09 ENT-01..05 clean rebuild with 2D

**Author:** AI assistant
**Tags:** rebuild, 2d, ent, cleanup
**Status:** In progress (rebuild running)

## Cleared (owner choice)
- Deleted all of `output/` except `cowlitz_2019_synch_1_recreated.db`, `cowlitz_2019_synch_1_recreated_legacy/`, `run_2019_*.log`, `2025_review/` (PM share packet kept).
- Deleted the step-1 base DB too, so step 1 re-parses raw files from K:.

## Code and config changes
- `scripts/tagdrag_2025_pipeline.py`: `DRAG_LINES` now ENT-01..05; `plot_drag` passes `--lines`; new optional step `plot_drag_2d` (12c) scoring `tblPositions_Deng2D` into `output/2025_review/<db stem>_2d/`.
- `scripts/ent_analysis_2025.py`: `--table` accepts `tblPositions_Deng2D`; item12 reads all columns and fills `r3` when missing (2D uses sets of three).
- `config/run_data_tagdrag_ent.toml`: `known_depth_diagnostics = true` (Deng2D with Z fixed to FC36 -1.524 m, FFD3 -4.572 m; scored against GPS, diagnostic only; was off by owner rule).
- No swap flag needed: new builds apply ZOI05/ZOI06 swap in `adapt_2025_to_legacy`.

## Run
- Step 1: `python scripts\run_data.py config\run_data_tagdrag_ent.toml` -> `output/run_ent_step1.log`.
- Steps 2-12: `python scripts\tagdrag_2025_pipeline.py config\run_data_tagdrag_ent.toml` -> `output/run_ent_pipeline.log`.
- DB: `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db`.
- Tests: full suite OK before launch.

## Result (steps 1-12 done, clean rebuild)
- Step 1 ok; steps 2-12 ok after a fix: plot_drag_done and the cleanup loops looked for stale PNG names without the 3D_legacy_B token, so step 12 reported failure although all PNGs existed. Now match item12_pipeline3d_3D_legacy_B_<line>_<tag>.png.
- Timings: 3D positions 12.2 min, Deng2D 27.6 min.
- 3D legacy B centroid reproduces the earlier production numbers exactly (ENT-01 FC36 3.30/20.0, ENT-05 FC36 3.87/27.2 m median/p90).
- 2D (Z fixed, root-B centroid), median / p90 m vs 3D: ENT-01 FC36 12.77/35.6 vs 3.30/20.0; ENT-01 FFD3 16.41/30.9 vs 4.29/19.3; ENT-02 FC36 6.35/37.8 vs 4.80/10.6; ENT-02 FFD3 10.36/18.7 vs 4.44/22.7; ENT-03 FC36 6.43/16.7 vs 3.41/20.8; ENT-03 FFD3 7.23/20.4 vs 2.45/16.4; ENT-04 FC36 7.10/39.0 vs 1.90/17.4; ENT-04 FFD3 9.56/40.6 vs 2.71/21.8; ENT-05 FC36 8.85/94.1 vs 3.87/27.2; ENT-05 FFD3 10.96/47.5 vs 3.95/24.3. 2D yield is higher (80-98%) but median error is 2-4x worse.
- Plots: 3D output/2025_review/jsats3d_2025_tagdrag_ent_anchorZOI03/; 2D output/2025_review/jsats3d_2025_tagdrag_ent_anchorZOI03_2d/.

## 2D retired; accuracy investigation (owner request)
- known_depth_diagnostics = false in run file. Cause of 2D failure: CFD float receivers in the sets-of-three (ENT-01 FC36 median 8.4 m with ZOI-only sets, 12.8/26.4/38.6 m with 1/2/3 CFD receivers; float GPS = unit position, 16-30 m off the hydrophone). Fixed-Z mismatch (3D solves FC36 4.5 m deeper than the assumed -1.524 m) and three-receiver geometry add the rest.
- Per-receiver 3D screen across all five lines: ZOI01 worst (combos 4.89 m vs 3.42 m excluded) BUT removing it from the centroid is a wash per line (helps ENT-04/05, hurts ENT-01/02/03). No receiver cut.
- DBSCAN screen (eps 5 m, min_samples 5) run on the fresh DB, positions_dbscan = true in the run file (separate table tblPositions_DengDBSCAN, legacy tables untouched; owner sign-off still pending for production). plot_drag_dbscan now passes --lines for all five.
- DBSCAN vs raw (legacy B centroid, median/p90 m): ENT-01 FC36 3.01/8.0 (was 3.30/20.0); ENT-01 FFD3 4.17/18.8 (4.29/19.3); ENT-02 FC36 4.80/10.6 (same); ENT-02 FFD3 4.39/21.2 (4.44/22.7); ENT-03 FC36 2.82/6.9 (3.41/20.8); ENT-03 FFD3 2.37/15.0 (2.45/16.4); ENT-04 FC36 1.88/14.5 (1.90/17.4); ENT-04 FFD3 2.47/12.6 (2.71/21.8); ENT-05 FC36 3.46/13.6 (3.87/27.2); ENT-05 FFD3 3.86/18.0 (3.95/24.3). Better or equal on every line; biggest effect in the p90 tail.
- Plots: output/2025_review/jsats3d_2025_tagdrag_ent_anchorZOI03_dbscan/.
- Open accuracy leads: FC36 depth bias -4.5 m (datum/WSEL unconfirmed); remaining dense multipath clusters survive DBSCAN; receiver survey accuracy is the floor (paper needs cm-grade coordinates).

## Datum verified; deep-Z what-if launched
- Datum CHECK: drag-window WSEL 862.27 ft vs 862.46 ft reference = 6 cm; WSEL followers (ZOI07-09) already move with WSEL(t) in receiver_z_at; fixed receivers' deployment-surface frame differs from drag day by ~6 cm. Datum is NOT the cause of the depth bias.
- Proof it is geometry: solver puts FC36 (5 ft) at -6.1 m and FFD3 (15 ft) at -3.2 m; order inverted, impossible for a constant datum offset. All solving receivers except ZOI01-03 sit in the top 3 m: weak vertical geometry.
- Real Z-input defect found: step 6b adopts Deng-solved Z for deep receivers; ZOI02 adopted -8.11 m vs measured hydrophone depth -12.40 m (4.3 m), ZOI03 -8.78 vs -10.29, ZOI01 -13.11 vs -11.15.
- New option [legacy] adopt_deep_z: 'solved' (legacy default, unchanged) | 'config' (keep measured depth, adopt XY only). done-check updated; full tests pass.
- What-if: output/jsats3d_2025_tagdrag_ent_zconfig.db from --init-from fresh DB, temp run file config/_run_ent_zconfig.toml (adopt_deep_z='config'), log output/run_ent_zconfig.log. Temp TOML to delete after.
- Not fixable from the desk: receiver survey accuracy (needs cm-grade survey). Dense multipath clusters: next candidate after this run.

## Deep-Z what-if result (adopt_deep_z='config', DBSCAN-screened, median/p90 m vs base)
- Stale-table bug found and fixed: tblPositions_DengDBSCAN was not dropped by init_whatif / adopt_deep / deep_clocks invalidation, so the what-if first scored the copied base table. Now dropped in all three places.
- Result mixed: pooled median 3.23 -> 3.12, pooled p90 14.1 -> 10.9. Helps every FFD3 line and ENT-05 FC36 (ENT-01 FFD3 4.17/18.8 -> 3.24/9.8); hurts ENT-02 (FC36 4.80/10.6 -> 5.61/25.6) and ENT-01/03 FC36 medians.
- Verdict: not adopted. Default stays adopt_deep_z='solved' (legacy). The option is in the run file for the owner's call; evidence DB kept at output/jsats3d_2025_tagdrag_ent_zconfig.db (log run_ent_zconfig.log).
- Temp run file config/_run_ent_zconfig.toml deleted.
- Summary of the 'fix everything' pass: datum verified correct (6 cm, nothing to fix); deep-Z measured-depth option built and tested (mixed, owner call); DBSCAN screen on and plotted for all 5 lines (clear win, sign-off pending); survey-grade receiver coordinates remain the accuracy floor (field work).
