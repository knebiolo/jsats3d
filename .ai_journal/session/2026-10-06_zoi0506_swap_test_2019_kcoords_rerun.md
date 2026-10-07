# Session: 2026-10-06 — ZOI05/06 coordinate swap test (2025 drag) and 2019 Deng rerun with verified coordinates

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-06 15:02 local (PDT). Continues `2026-10-05_drew_guidance_receiver_positions.md`.

## ZOI05/ZOI06 swap experiment (2025 tag drag, zoi02fix DB)
- Question: do Deng positions improve if ZOI05 and ZOI06 hydrophone coordinates are swapped (crossed-cable hypothesis from 10-04/10-05)?
- DB: `output/jsats3d_2025_tagdrag_all_20250605_zoi02fix.db`. Work dir `..._legacy`.
- Pitfall found: `position.Deng` reads receiver positions from `X_t, Y_t, Z_t`, NOT `X, Y, Z`. First swap attempt only changed X/Y/Z, so swapped Deng output was byte-identical to config. Second pitfall: sequential UPDATEs overwrite each other; must stage originals in a temp table (`tmp_xyzt`) before swapping.
- Final DB state: ZOI05 X_t/Y_t/Z_t = 508.56/136.22/-3.10, ZOI06 = 475.92/135.20/-3.10 (both X,Y,Z and X_t,Y_t,Z_t swapped, consistent per receiver).
- Deng rerun, 10 receivers ZOI01-ZOI10, 6 tags; pickle `deng_solutions_swapped_rerun.pkl`, CSVs `positions_swapped_rerun/`. Baseline: `deng_solutions_config10.pkl` / `positions_config10/`.
- Accuracy vs GPS truth (tblDetectionClockFixed GPSFix lat/lon -> EPSG 26910 -> local, origin E 567994.0029751 / N 5146184.4013067; merge_asof on ToA vs seconds_fix tol 2 s; horizontal error only; NaN rows dropped — Deng Z is local depth, GPS has no usable Z):

| Tag  | cfg med | swap med | Δ med | cfg MAE | swap MAE | Δ MAE |
|------|--------:|---------:|------:|--------:|---------:|------:|
| FC36 | 47.49 | 48.26 | +0.77 | 91.43 | 92.31 | +0.89 |
| FFD3 | 49.50 | 50.36 | +0.86 | 70.39 | 99.94 | +29.56 |
| CC74 | 28.72 | 25.43 | -3.29 | 34.93 | 27.58 | -7.35 |
| 10D3 | 43.30 | 41.72 | -1.57 | 48.69 | 47.37 | -1.32 |
| 5CC5 | 33.09 | 30.51 | -2.58 | 45.68 | 47.58 | +1.90 |
| 7FA2 | 41.37 | 33.00 | -8.37 | 45.97 | 41.41 | -4.55 |

- Verdict: mixed. 4 CFNSC float tags improve (median -1.6 to -8.4 m), FC36/FFD3 slightly worse (+0.8 m median, FFD3 MAE +30 m from outliers). Not conclusive evidence for or against crossed cables; effect small vs the 30-50 m raw-solution medians (note these are unfiltered B-root medians, much coarser than step 12 box-rule numbers).
- NOTE: these medians use ALL B-root solutions merged to all GPS rows, not the box-rule track; direct comparison to step 12 figures (5-8 m) invalid.
- DB is currently LEFT IN SWAPPED STATE (X,Y,Z and X_t,Y_t,Z_t both swapped for ZOI05/ZOI06). Revert from `tblReceiver_initial` before any production run.

## 2019 Deng rerun with verified receiver coordinates (Cowlitz)
- DB: `output/cowlitz_2019_dbscan_20261005.db`. Verified coordinates confirmed in tblReceiver (R03 Z_t 250.63).
- Old run renamed `tblPositions_Deng` -> `tblPositions_Deng_oldrecvr`; earlier pre-fix table kept as `tblPositions_Deng_before_fix`.
- Batch: 149 tags, 8 receivers R01-R04/R06-R09, water_column=True, sorted by kept detections; completed 322 min (~5.4 h). Log `output\run_2019_step11_kcoords.log`.
- Table counts (rows / tags / 'solution found'):
  - tblPositions_Deng (verified coords): 5,508,870 / 149 / 1,159,119 (21.0%)
  - tblPositions_Deng_oldrecvr: 5,508,870 / 149 / 1,503,842 (27.3%)
  - tblPositions_Deng_before_fix: 5,508,870 / 149 / 2,256,927 (41.0%)
- FINDING: solution-found rate DROPPED with verified coordinates (21% vs 27% vs 41%). Row counts identical (same transmissions/combos).
- QUALITY COMPARISON (spread = std of X/Y/Z across receivers per transmission):

| Table | Root | n solutions | spread median | spread mean | spread p95 | in-hull % |
|-------|------|------------:|--------------:|------------:|-----------:|----------:|
| verified (Deng) | A | 136,125 | 0.00 m | 1.83 m | 6.36 m | 34.7% |
| verified (Deng) | B | 1,022,994 | 1.99 m | 1.78 m | 4.21 m | |
| oldrecvr | A | 161,445 | 0.00 m | 1.74 m | 6.31 m | 38.4% |
| oldrecvr | B | 1,342,397 | 0.92 m | 1.11 m | 3.25 m | |
| before_fix | A | 719,223 | 5.91 m | 10.05 m | 33.46 m | 2.2% |
| before_fix | B | 1,537,704 | 0.91 m | 1.26 m | 3.72 m | |

- VERDICT: verified coordinates are WORSE than oldrecvr. Root B spread median 1.99 m vs 0.92 m (+116%), p95 4.21 vs 3.25 m (+29%), in-hull 34.7% vs 38.4%. Root A similar. The "verified" GPS coordinates degraded positioning quality.
- before_fix has terrible Root A spread (5.91 m) but good Root B — confirms Root A is ill-conditioned with bad receiver coords. Oldrecvr is the best of the three.

## Next steps
- Decide whether to revert ZOI05/06 swap in zoi02fix DB (currently swapped).
- Evaluate 2019 verified-coords positions vs oldrecvr (accuracy metric needed; no GPS truth for 2019 fish — compare solution spread/in-hull rates or beacon self-positioning).
- Report swap-test result to project lead/Drew: swap helps CFNSC tags, hurts FC36/FFD3 slightly; not decisive.
