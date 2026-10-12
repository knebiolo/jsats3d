# 2026-10-11 — Error Root Cause Analysis (Deng vs GPS), Paper Parity, Legacy Diff

Session ~21:00-22:15 local (PDT). Branch `ENM_jsat3d_edits`. DB `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db`.

## TL;DR (read this first)
1. Clocks are NOT the accuracy limit. True post-fix clock error is <=0.21 ms at every receiver (target 0.47 ms already met). The first analysis below mistook the APPLIED correction (`seconds_residual`) for leftover error; that section is marked SUPERSEDED.
2. Real error budget (~3.5 m drag median): per-ping combo scatter (~3.2 m, dominant), towed-tag layback (2-5 m bias on ENT-02/03), GPS truth timing (~1.2 m), clocks (<=0.3 m).
3. Static-hold benchmark (cleaner truth): in-hull 2.55 m median, p90 5.2 m; out-of-hull degrades to 13-51 m (S-9/S-10).
4. Static surface receivers cannot position fish or themselves: all sit at -2.3 to -3.3 m (coplanar). Only ZOI01/02/03 give vertical aperture.
5. Paper (Nebiolo & Meyer 2021) fish were solved by overdetermined NLS (Mathematica, never released); Deng was only for deep receivers. We built the NLS equivalent: it only ties Deng (2.83 vs 3.01 m drag) and only works when Deng-seeded + fixed-Z. Solver is not the limit.
6. Paper's "6 cm" is LSQ precision, not accuracy; its own tag-drag RMSE was 9.2/15.6/29.6 m. Its real edge: Trimble-surveyed receivers (10 cm), encircling array, vertical aperture.
7. Diff vs `main`: our branch is a strict superset of the 2019 method with six legacy bugs fixed. Nothing in main to "shift back" to.
8. Biggest fixable software term: receiver position error (ZOI01 -0.89 m systematic residual). Independent fix = RTK survey of ZOI hydrophones (none exists on K:).

## Goal
Answer: why can't Deng solving produce near-perfect GPS-matching tracks, and can we match the 2019 paper?

## Work
- Finished `scripts/error_root_cause2.py` (8-part analysis; replaced broken `error_root_cause.py` draft).
- Fixed bugs: timeStamp dtype merge, receiver-table shadowing by loop var, DataFrame `.str.contains` misuse, empty-CFD guard, cp1252 crash (ran with `PYTHONIOENCODING=utf-8`).
- Results saved: `output/error_root_cause_results.txt`.
- DB: `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db`. Baseline: 278 matched, XY median 3.52 m, mean 5.52 m.

## Initial findings (SUPERSEDED - these used the applied correction, not leftover error; see CORRECTION)
1. ~~Clock residual = dominant error source.~~ WRONG. Per-receiver applied |seconds_residual| during drag window:
   - ZOI09 ≈ 946 ms (catastrophic), ZOI02 ≈ 180 ms ± 193 ms (unstable)
   - ZOI05 ≈ 9.5 ms, CFD08/04 ≈ 6 ms, ZOI03 ≈ 2.8 ms
   - ZOI01 ≈ 1.2 ms, ZOI06 ≈ 0.47 ms, ZOI07 ≈ 0.2 ms, ZOI08 = 0 (master)
2. **Physics**: 1 ms sync error ≈ 1.5 m position error (c_water ≈ 1480 m/s). ZOI03/ZOI05 residuals (2.8–9.5 ms) ≈ 4–14 m — exactly the observed error band.
3. Receiver-set composition confirms: ZOI06-dominated sets → 1.2–2.6 m; ZOI03/ZOI05 sets → 5–25 m (ZOI03-only: 25 m median).
4. Systematic bias small: ~+2 m X, −1.7 m Y (survey-level).
5. Geometry NOT limiting: hull area bins show no strong trend; tag <500 m from receivers.
6. Exclude ZOI09: median 3.52 → 3.15 m (modest; ZOI09 rarely in winning combos).
7. CFD-only solving: 0 positions (fish not heard by 4+ CFD).
8. Spearman avg-clk-resid vs XY error: r=0.152, p=0.011 (weak but significant).

## ~~Conclusion~~ (WITHDRAWN)
The claim "accuracy ceiling = clock surface quality" is false. Items 2, 3, 8 above are only valid as observations of which receivers were corrected most heavily. Exclude-ZOI09 test (3.52 -> 3.15 m) and CFD-only (0 positions) results remain valid.

---

## CORRECTION (same day, follow-up)
First table above measured `seconds_residual` = APPLIED correction, NOT post-fix error
(already documented 2026-10-01; re-verified in jsats3d.py: `seconds_fix = seconds - seconds_residual`).

### True post-fix clock error — `scripts/clock_loo_quality.py`
Leave-one-out prediction at clock-fix knots (`receiver_*_clock_fix.csv`, ToA_error), drag window:
- ZOI01/02/03/05/06: 0.001–0.005 ms (~mm). CFDs: 0.06–0.15 ms. ZOI07/09: ~0.2 ms.
- **ALL receivers ≤0.21 ms — target 0.47 ms already met everywhere. Clocks are NOT the limit.**
- Watch: ZOI02 full-record loo_p95 44.6 ms (one bad segment outside drag); ZOI10 gap 1669 s during drag.

### Layback — `scripts/layback_test.py`
Error decomposed along/cross travel direction per line/tag:
- Along-track: ENT-02 ≈ −5.0 m, ENT-03 ≈ −2.7 m, ENT-01 ≈ 0/+2 m. Cross-track: +0.6 ± 0.7 m.
- Negative along = tag trails GPS antenna = towed layback. Removing per-line bias: 3.52 → 2.63 m median.

### Scatter decomposition — `scripts/scatter_decompose.py`
- Within-ping combo spread (same ping, different 4-receiver subsets): X std median 9.1 m, Y 6.8 m.
- Implied SE of per-ping median ≈ 3.2 m — matches observed 3.5 m error almost exactly. **Dominant term.**
- GPS truth error ~1.2 m (minute-truncated stamps, reconstructed 1 Hz, boat ~0.6 m/s).

### Revised error budget
| Source | Contribution |
|---|---|
| Combo scatter (geometry + receiver positions + multipath) | ~3.2 m (dominant) |
| Towed-tag layback bias | 2–5 m on ENT-02/03 (removable) |
| GPS truth timing | ~1.2 m |
| Clocks (post-fix) | ≤0.3 m (done) |

### Static hold benchmark — `scripts/static_holds_score.py`
Production anchorZOI03 DB vs FBY-S-1..10 (no layback/timing issues; handheld GPS truth ±few m):
- **In-hull (S-1..S-3): 2.1–5.2 m per hold, pooled median 2.55 m, p90 5.22 m** (n=110 pings).
- Out-of-hull degrades with distance: S-4 1.7–3.0 m → S-9/S-10 13–51 m. Z errors blow up out-of-hull (FFD3 S-8 z=−30 m vs −4.6 true).
- Confirms error budget: geometry/GDOP dominates; in-hull floor ~2.5 m ≈ combo-scatter SE (3.2 m) given truth uncertainty.
- Matches 2026-10-04 swap-what-if hold test (S3 in-hull 1.4–2.2 m with fitted receiver positions — fitted positions buy ~1 m more).

### Why no static/surface-only solving — `scripts/why_no_surface_solve.py`, `why_no_surface_receiver_solve.py`, `deep_count_check.py`
- Fish detections live in `tblDetectionFilterSecondary` (NOT tblMetronomeSecondFiltered; that holds only master beacon 7DB7).
- Audibility is NOT the problem: 73% of drag pings have ≥4 clean surface-ZOI receivers; 99% with CFDs.
- Geometry IS the problem: all surface receivers sit at −2.3 to −3.3 m (≈1 m Z spread over ~500 m array) — coplanar. Deep = −8.1/−8.8/−13.1 m.
- Deng quality by # deep receivers in 4-combo: 0 deep → 37% in-bounds, 24% plausible Z; 1 deep → 64%/53%; 2 deep → 71%/43%. Coplanar TDOA cannot resolve Z (mirror/runaway solutions).
- Solving static receiver POSITIONS acoustically: surface beacons have 0 transmissions with ≥4 static refs (only master beacon survives metronome filter; and refs are coplanar too). 2026-10-08 Deng-all test already showed surface adoption makes GPS worse (3.3 → 18.6 m, ZOI05 Z +41 m).

### Path to better tracks (owner-gated)
1. Layback: estimate tow offset per line or model speed×layback; recovers ~0.9 m median.
2. Combo scatter: GDOP-weighted/filtered combo selection, robust location estimate (geometric median), reject high-leverage combos. Receiver position accuracy matters more than clocks now (ZOI01-03 unit-vs-hydrophone offsets 16–30 m known issue; ZOI05/06 swap already applied).
3. GPS truth: use static holds FBY-S-1..10 (known positions, no layback/timing issue) as cleaner accuracy benchmark.

### Paper-parity NLS solver test — `scripts/nls_fish_solver.py`, `scripts/nls_fixedz.py`
Paper (Nebiolo & Meyer 2021) solved fish with overdetermined NLS (Mathematica, never released);
Deng was only for deep receivers. Built scipy equivalent:
- Raw NLS (free Z, soft_l1): FAILS — drag median 13.8 m. Gross outliers + coplanar Z runaway.
- + iterative worst-residual trimming: still 12.0 m, only 554/1657 pings. Bad seeds → local minima.
- + fixed-Z (known tag depth) + **Deng-median seed**: WORKS — 1,469/1,657 pings, fit RMS 1.02 m,
  drag median 2.83 m (Deng 3.01), holds in-hull 2.71 m (Deng 2.55). CFDs mostly trimmed (drift).
- Conclusion: solver choice is NOT the limit; hybrid Deng-seed + NLS-refine ≈ Deng median. The paper's
  edge was surveyed receivers (Trimble 10 cm) + encircling array with vertical aperture, not the solver.
- GOLD: per-receiver median residuals from tight fits now measure receiver position bias directly:
  ZOI01 −0.89 m (largest), others ±0.25 m. This is the self-calibration observable.

### Next (owner-gated)
- Joint receiver-position calibration: solve per-receiver XY offsets minimizing NLS residuals over all
  drag+hold pings, then re-test held-out. Supersedes/confirms the 10-04 fitted-positions result.
- RTK survey of ZOI receivers remains the independent ground truth (paper's own 2019 protocol).

### K: drive GPS census (re-verified today; full inventory was 2026-10-04)
Every GPS source in the 2025 data:
- `array_testing_drag_GPS.csv`: boat antenna during drags; minute-truncated timestamps (1 Hz reconstructed).
- `cowlitz_AT_2025_testing_sheets.xlsx`: static holds, handheld lat/long (+/- few m).
- `4_gps_datasets/master_df_gps.csv`: 1,114,887 rows, 2025-06-03 to 09-17, **CFD02-09 floats only**, ~1/min; feeds `tblReceiverGPS`.
- `cowlitz_2025_AT_config.xlsx`: hand-entered ZOI/CHN/DNS/UPS hydrophone lat/long; catalog flags CFD and CHN/DNS as approximate.
- `cowlitz_2025_deployed_array.kmz`: same 35 points as config, NOT independent.
- `tblDetectionRaw` GPSFix columns: each ATS UNIT position, not hydrophone (ZOI01-03 units 16-30 m from hydrophones).
- Conclusion: no independent survey of any ZOI hydrophone exists on K:. Not exhaustively read: 2-4 GB master_df detection CSVs, every raw_data file (Egnyte slow), PNG/HTML contents, Lotek raw.
- Frame note: `tblReceiverGPS` X/Y are in the same local frame as `tblReceiver.X_t/Y_t` (checked with `scripts/frame_check.py`).

### Paper review: Nebiolo & Meyer 2021 (Anim Biotelemetry 9:20)
- Workflow (Fig. 3): metronome epochs -> metronome multipath -> clock sync at surface -> beacon multipath -> position deep receivers (Deng 2011) -> clock sync at depth -> fish multipath -> fish positions (NLS in Mathematica, not released).
- Accuracy evidence in the paper: stationary beacon R05 vs survey RMSE 0.02/0.05/0.004 m; precision (LSQ sigma) in hull 0.06/0.06/0.12 m, outside 0.25/0.33/0.27 m; tag-drag RMSE 9.2/15.6/29.6 m, blamed on code-only GPS (URE 7.8 m 95%). Authors recommend RTK/total station for the drag boat.
- Array in paper: 9 receivers encircling the area, ~16 m vertical range (3 bottom, 3 trashrack, 3 fixed), surface receivers Trimble Geox7 surveyed to 10 cm. Out-of-hull positions called "effectively extrapolated".
- Parity items we already match: metronome sync, DBSCAN/KNN multipath, Deng for deep receivers, per-epoch SoS from temperature profile.
- Parity gap: fish solver (NLS) and surveyed receivers / geometry (field items).

### Legacy `main` diff review (`git diff main...HEAD -- jsats3d/jsats3d.py`, 298+/94-, saved `output/diff_vs_main.txt`)
Bugs fixed on our branch (legacy main is wrong):
1. Root-B point built as (S1b.x, S1a.y, S1a.z) - mixed roots; fixed to S1b.
2. T_0a <= 0 < T_0b hit undefined S1a, swallowed by bare except, valid B solution lost (also an undefined-name statement in the debug print path).
3. "No B solution" rows appended to SolutionA.
4. Multipath DBSCAN `chunk_df[chunk_df != -1]` filtered elementwise, not by cluster; fixed to `dbscan_cluster != -1`.
5. sqrt of negative discriminant unguarded; now returns "negative quadratic discriminant" rows.
6. Duplicate timestamps fed to `interp1d` in clock fix; now deduplicated with conflict check.
Validated-equivalent performance changes: vectorized epoch enumeration, chunked Deng row collection (identical vs git HEAD per 10-02), `pd.concat` for `.append`, connection closing.
2025-only additions: dynamic CFD XY from tblReceiverGPS (no extrapolation), WSEL-following receiver Z, opt-in `water_column` rejection, `Deng2D` fixed-Z solver, beacon epoch re-hosting when host receiver stops hearing.
No scientific component from 2019 is missing except the unreleased Mathematica NLS fish solver.

## Scripts created today (DELETED at end of session; not pipeline code)
All 21 untracked diagnostic scripts were removed from `scripts/` (no references from pipeline, tests or config). Only tracked pipeline scripts remain. To reproduce any result, rebuild from the method described in the sections above. Scripts that existed:
- `error_root_cause2.py` (8-part error analysis; output `output/error_root_cause_results.txt`; note: its clock section is the superseded one)
- `clock_loo_quality.py` (true post-fix clock error by leave-one-out at knots)
- `layback_test.py`, `scatter_decompose.py`, `static_holds_score.py`
- `why_no_surface_solve.py`, `why_no_surface_receiver_solve.py`, `deep_count_check.py`, `find_fish_tables.py`, `frame_check.py`
- `nls_fish_solver.py`, `nls_fixedz.py` (paper-parity NLS prototypes; diagnostic only; `output/nls_fish_positions_dragwindow.csv` is from the free-Z run)
- Stale draft `error_root_cause.py` (superseded by `error_root_cause2.py`) and earlier debug scripts (`debug_*.py`, `list_tables.py`) can be deleted at owner discretion.

## Scoreboard (all against production anchorZOI03 DB)
| Method | Drag median | Holds in-hull median |
|---|---|---|
| Deng root-B per-ping median (production) | 3.01 m (3.52 m on 278 matched earlier) | 2.55 m |
| NLS free Z | 13.8 m | 12.1 m |
| NLS free Z + trimming | 12.0 m | 10.0 m |
| NLS fixed Z + Deng seed | 2.83 m | 2.71 m |
| Earlier fitted-position what-if (10-04, not adopted) | n/a | 1.4-2.2 m |

## Decisions / actions needing owner or field input
- RTK/Trimble survey of ZOI hydrophone positions (resolves ZOI01 ~8 m west question; validates fitted coordinates). Also confirm ZOI05/06 cable/lat-long swap physically.
- Next deployment: encircle study region, add vertical aperture (receivers at varied depth away from ZOI01-03 corner), RTK-track the drag boat with full-second timestamps, record tow layback.
- Adopt or reject: joint receiver-position calibration from NLS residuals (not run yet); combo-quality filter (>=1 deep receiver, GDOP, Z plausibility; not built yet); out-of-hull quality flag for production fish positions; fixed-Z 2D solving for known-depth tags.
- Nothing in this session changed production code, config, or databases.

## Open items / not done
- Joint receiver-position calibration run + held-out test.
- GDOP/deep-count combo filter prototype and rescore.
- Layback correction estimate per line.
- ZOI02 full-record loo_p95 44.6 ms (bad segment outside drag) and ZOI10 1,669 s beacon gap during drag worth checking.
- Repo memory updated with error budget, NLS result, and paper facts (`/memories/repo/jsats3d_facts.md`).
