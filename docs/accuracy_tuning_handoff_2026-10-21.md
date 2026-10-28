# 2025 Accuracy Tuning — Handoff for Implementation (2026-10-21)

Purpose: verified findings + parameter test results + an ordered implementation plan for
getting the best achievable fish-position accuracy from the 2025 tag-drag pipeline.
Validation truth: drag GPS tracks ENT-01..ENT-05 (`array_testing_drag_GPS.csv`), scored
exactly like `scripts/ent_analysis_2025.py` (median/p90 XY per line, weighted by scored
transmissions). Tag depths: FC36 = -1.524 m, FFD3 = -4.572 m (5 ft / 15 ft, recorded).

**Read first:** `System_Prompt.txt` (operating rules), `LONG_TERM_CONTEXT.md` (standing
decisions, owner escalations). Branch: `ENM_jsat3d_edits`. Conda env:
`C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe`
(run as `python` after `conda activate jsat_3d`; `conda run` drops stdin).

Baseline DB: `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db` (window 2025-06-10 09:00-17:00 wall;
tags FC36/FFD3; 10 receivers_3d = ZOI01,02,03,05,06,07,08,09,10; 13 receivers_2d add CFD01,05,09).

## STATUS (updated 2026-10-21, end of campaign)

| Item | State |
|---|---|
| F1 clock correction in `multipath_data_object` | **APPLIED** (jsats3d.py, study-tag branch). Note: the first implementation (earlier today) was broken **twice** — `np.interp` called with 2 args (missing `fp`), then the query referenced undefined `con` (local is `conn`); both TypeErrors/NameErrors were swallowed by a blanket except → silent no-op. Final form uses per-receiver knot arrays + vectorized `np.interp(t, ts, rs, left/right=endpoint)` and a `sqlite_master` table check (no try/except). Pre-flight verified on the production DB: ZOI09 shift 946.4 ms, ZOI09−ZOI08 residual +924 → −19 ms, 32286/39347 FC36 detections corrected. |
| F2 root-B scoring/plotting | **APPLIED** (ent_analysis_2025.py `legacy_b` + token `3D_legacy_B`; pipeline plot tokens done-check/cleanup/`--plots` all switched A→B, incl. the 2D step that previously matched no token). |
| Smoothing (10, 11, 3) | **APPLIED** (ent_analysis_2025.py call site, with note). |
| Kalman 2.0/1.0/0.5/20 | **NO CHANGE** (96-combo grid flat; noted inline). |
| DBSCAN screen (accel 8/5/5/1) | **NO CHANGE** (best at matched yield; unconstrained grid top is overfit; noted). |
| Validation rerun (production DB) | **PASSED** — `output/run_2026_10_21_verify.log`. F1 active in-pipeline (32286/39347 FC36, 20683/23771 FFD3 corrected). Final per-line B-Kalman: 3D weighted median **3.95 m** (baseline 11.05, −64%), 2D **3.73 m** (baseline 9.55, −61%); 3D B depth medians −2..−6 m (was +6 m class). Root-B PNGs written for all 5 lines × 2 tags in all three review folders. |
| Follow-ups (Steps 4-5: classifier sweep, all-20-receiver clock fix, deep_solution B, etc.) | **OPEN** — see Sections 4-6. |

Full record with before/after metrics: `.ai_journal/session/2026-10-21_accuracy_tuning.md`.

---

## 1. VERIFIED FINDINGS (with evidence)

### F1 (P0, NEW — the other models' list missed this) Fish timestamps are never clock-corrected

**The pipeline computes per-receiver clock corrections (steps 2-7) but never applies them to
study-tag detections.** Trace:

1. `jsats3d/jsats3d.py` `multipath_data_object.__init__` (~line 309): for `tagType == 'study'`
   it reads **`tblDetectionRaw`** (raw clock times) and merely renames `seconds` -> `seconds_fix`
   (comment claims "receivers are already clock-fixed via step 4" — that is false: step 4 only
   corrects the beacon rows that go to `tblDetectionClockFixed`).
2. Epoch assignment `transNo = round((seconds_fix - first)/pulseRate)` is offset-invariant, so
   the error is invisible there.
3. `position.__init__` (~line 1291) reads `tblDetectionFilterSecondary` and Deng uses
   `seconds_fix` directly. TDOA = t_i - t_ref therefore carries the per-receiver clock offset
   (delta_i - delta_ref) into every solution.

**Evidence (test T1, script `_tune_t1_fish_times.py`, pairwise first-arrival TDOA residuals vs
geometry, both drag tags):**

| Receiver | Offset vs main group (ms) | In clock-fix product? |
|---|---|---|
| CFD01 | ~ +940 (late) | **No** — not in surface_receivers, never clock-audited |
| ZOI09 | ~ +924..+960 (late) | "Yes" — but see below |
| ZOI11 | ~ +940 (late) | **No** — excluded from processing |
| ZOI02 | ~ +22..+85 + jumps | Yes (curve, range 8..405 ms) |
| ZOI01..ZOI10 (others) | 0..+45 | Yes (curves -5..+22 ms) |

Main group {CFD02-08, ZOI03-08, ZOI10} internal residuals: 0-45 ms. The ~940 ms late trio
{CFD01, ZOI09, ZOI11} matches the 2026-10-19 beacon audit (ZOI09 = 946,060 us) — that audit
used ZOI09 as reference for one view, which is why CFD01/ZOI11 were never flagged.

**ZOI09's correction is additionally invalidated.** `clock_fix()` (jsats3d.py ~line 1047)
subtracts the 946 ms from the residual curve *before* interpolation, so the stored
`seconds_residual` for ZOI09 is ~0.4 ms (verified in DB) and `seconds_fix ≈ seconds`:
the 946 ms is NOT in the clock-fix product. The 2026-10-19 journal read the resulting
"residual 946 ms -> 0.777 ms" as "945 ms fixed"; it actually means the applied correction
shrank to ~0. (That is also why the 10-19 rebuild changed zero accuracy: fish times were
never corrected either way.)

**Physics:** each receiver clock offset delta_r biases its range by c*delta_r. 946 ms *
~1476 m/s ≈ 1.4 km. Of C(10,4)=210 four-receiver combinations, C(9,3)=84 (40%) include ZOI09;
those solutions are grossly wrong or nonexistent and poison the per-transmission centroid.
This is the single largest error source in the 3D path and explains the catastrophic
Z medians (root A: -42 m median, -191 km extreme; 20% of rows above the water surface).

**Fix (implement in `multipath_data_object` or a new pre-step):** before epoch assignment,
for every study-tag row at receiver r and time t: `t_corr = t - residual_r(t)`, where
residual_r is the piecewise-linear interpolant of `(seconds, seconds_residual)` from
`tblDetectionClockFixed` (beacon basis, 8 h sync window), plus the 946 ms constant for ZOI09
(the documented hardware offset; the curve no longer contains it). A working prototype of
this is `scripts/_tune_t2_apply_clockfix.py` (test DB
`output/jsats3d_2025_tagdrag_ent_clockfix.db`). Test T2 (full pipeline rerun) quantifies the
gain — see Section 3.

**Caveats to keep in the journal:**
- Corrections come from the beacon at ZOI08; ZOI08's position is unsurveyed, so per-receiver
  corrections carry a residual bias (T1 shows 10-45 ms discrepancies vs the fish-derived
  offsets — the unsurveyed host + geometry, not a bug).
- CFD01 (in receivers_2d) and ZOI11 have no correction curve; T2 left them uncorrected.
  CFD01 ≈ +944 ms late (T1). For a clean 2D result, extend the clock-fix run to all 20
  receivers (beacon 7DB7 is heard at 20/20) or derive constants and record them.
- ZOI02's curve contains the jump segments (8..405 ms) — the piecewise fit is per-segment,
  so applying it in time is the right treatment.

### F2 (P0-3, CONFIRMED) Root B beats root A, by a lot

From `tblPositions_Deng` / `tblPositions_Deng2D` in the baseline DB (test T3, all 5 lines,
weighted by scored transmissions):

| 3D variant | scored | med XY (m) | p90 XY (m) |
|---|---|---|---|
| **A centroid (current pipeline product)** | 461 | **55.0** | 346.4 |
| A consensus/Hampel | 172 | 31.1 | 56.7 |
| A Kalman (2.0/1.0/0.5/20) | 126 | 28.5 | 45.9 |
| B centroid | 505 | 39.4 | 231.4 |
| B consensus/Hampel | 295 | 12.2 | 36.8 |
| **B Kalman** | 262 | **11.1** | 25.4 |

| 2D variant | scored | med XY (m) | p90 XY (m) |
|---|---|---|---|
| A centroid | 533 | 70.5 | 428.0 |
| A Kalman | 83 | 21.4 | 28.8 |
| B centroid | 582 | 21.0 | 79.7 |
| **B Kalman** | 422 | **9.6** | 38.9 |

- B has 2.1-2.3x more solutions (3D) and 4.2x (2D); B Z is shallower (FC36 B median +6.1 m
  vs A -42.3 m; tag at -1.5 m) and 61-65% of B rows are above water vs 14-20% of A rows.
- **The current pipeline scores root A** (`ent_analysis_2025.py` line 277 `solution == "A"`,
  and `plot_drag_2d_run` passes `--plots 3D_legacy_A`) — the worst variant available.
- Switching the scoring/plotting to root B + consensus/Hampel/Kalman is a no-resolve win:
  3D 55 m -> 11 m, 2D 21 m -> 9.6 m (weighted median).
- Physics: in a nearly flat array (9 m vertical spread) both quartic roots are ill-conditioned;
  the "A above / B below plane" naming from the 2019 deep-array case does not transfer. B
  empirically concentrates near the true track. This is an observation about this array's
  geometry + residual biases, not a universal property — record it as such.

### F3 (P0-1, PARTIALLY CONFIRMED) Z-filter mismatch is real but low severity

- `scripts/chinook_analysis_2025.py` lines 32-35: `LOCAL_Z_MIN, LOCAL_Z_MAX = -30, 0`.
- `scripts/plot_fish_tracks.py` line 40: `LOCAL_Z_MIN, LOCAL_Z_MAX = -15, 0`.
- These are two standalone diagnostic scripts (plot_fish_tracks is gitignored, not pipeline).
  The ENT scoring path (`ent_analysis_2025.py`) has **no Z filter at all** (only a |XYZ|<1000
  sentinel; its `z_min` at line 327 is a plot-axis limit). No pipeline result is affected.
- If the two scripts are ever used together, reconcile them (recommend -15..0 for forebay
  depths, matching plot_fish_tracks; the deep receivers sit at -10..-12 m).

### F4 (P0-2, BY DESIGN — do not "fix") Deng does not constrain Z

`Deng()` (jsats3d.py line 1385) flags `in_hull` but never rejects above-water/land solutions.
This is the published 2021 methodological commitment (System Prompt 11.0): physically
impossible solutions are retained as presence data. Keep it. The hull-screened scoring variant
already exists (`3D_hull_screened`). Only 0.6-8% of 3D rows are in-hull in this array — a
geometry fact, not a code defect.

### F5 (context) What the other list got right

- P1 position-DBSCAN grid (eps, min_samples, mode, accel_scale, depth_weight) — real, config-
  driven, cheap to test. Results in Section 3.
- P2 SUSPECT_WINDOW=9 / SUSPECT_LATE_S=0.003 s (`tagdrag_2025_pipeline.py` lines 50-51) are
  hardcoded and never tuned — real gap.
- P2 clock jump threshold 0.05 s (jsats3d.py line 1119) unvalidated — real gap; note the
  legacy DBSCAN eps in clock_fix is a *quantile* (90/95 pct of interpolation-step distance,
  lines 994-998), not the fixed 0.5 ms budget — that is legacy-core behaviour, owner review
  before change.
- P3 "smoothing window 30-300 s, jump threshold 20-100 m" — units hallucinated; the actual
  `smooth_track` params are count-based rolling windows (jump_window 7, smooth_window 5) and
  jump_limit_m 15. T3 swept the real params.

---

## 2. TUNING MAP (every tunable knob, file:line, current value, status)

| # | Knob | Location | Current | Notes |
|---|---|---|---|---|
| 1 | Fish clock correction | jsats3d.py `multipath_data_object` ~309 | **ABSENT** | F1. Implement. |
| 2 | ZOI09 constant | jsats3d.py ~1047-1050 | 0.946 s subtracted pre-fit | Keep, but it must also reach fish times (F1). Re-derive from data when ZOI08 host surveyed. |
| 3 | Clock jump threshold | jsats3d.py line 1119 | 0.05 s, 30-epoch windows, 5-min merge | Unvalidated; changing requires full re-clock (expensive). |
| 4 | Legacy DBSCAN eps in clock_fix | jsats3d.py 994-998 | 90/95 pct quantile, min_samples 3 | Legacy core; owner review. |
| 5 | Beacon DBSCAN budget/window/min_samples | beacon_pairwise_dbscan.py 26-31 | 0.5 ms / 2.5 periods / 3 | Fixed by design (System Prompt 7.4); **owner review pending per LONG_TERM_CONTEXT 2026-09-24**. Do not change without owner. |
| 6 | SUSPECT_WINDOW | tagdrag_2025_pipeline.py line 50 | 9 transmissions | ATS fish classifier. Sweep 5/9/13 with T2-style rerun. |
| 7 | SUSPECT_LATE_S | tagdrag_2025_pipeline.py line 51 | 0.003 s | Same. Physics: boat moves ~1.2 ms/ping; reflections sit several ms late (comment). |
| 8 | positions_dbscan mode | run_data_tagdrag_ent.toml [legacy] | acceleration | PM direction 2026-10-13. |
| 9 | dbscan_eps_m | toml | 8.0 | Tuned 2026-10-13 on same truth (optimistic). |
| 10 | dbscan_min_samples | toml | 5 | Same. |
| 11 | dbscan_accel_scale | toml | 5.0 | m of feature per m/s^2. |
| 12 | dbscan_depth_weight | toml | 1.0 | |
| 13 | dbscan_accel_gap_s | toml | 12.0 | |
| 14 | Scoring/plot root | ent_analysis_2025.py line 277; tagdrag plot_drag_2d 541 | A | **Should be B** (F2). |
| 15 | smooth_track jump_limit_m / jump_window / smooth_window / max_gap_s | ent_analysis_2025.py 168 | 15 / 7 / 5 / 20 | T3 swept. |
| 16 | Kalman pn/mn/zn/gap | ent_analysis_2025.py line 300 | 2.0/1.0/0.5/20 | Tuned 2026-10-19 (256-combo grid). T3 swept. |
| 17 | deep_solution root | toml [legacy] | A | Affects step 6 deep-receiver positions (ZOI01-03). Test B. |
| 18 | adopt_deep_z | toml | solved | Drag DB: 'solved'; chinook DB used 'config' (solved put deep recs above water, 2026-10-13). Owner decision. |
| 19 | receivers_3d / receivers_2d | toml | 10 / 13 | ZOI02 drop recommended 2026-10-19 (3x speedup, no accuracy loss) — not applied. |
| 20 | fixed_z_2d_by_tag | toml | -1.524 / -4.572 | Drag depths; "depth reference to be confirmed" (comment). |
| 21 | Sound speed / temp | tblInterpolatedTemp (DD_N 4-depth mean) | mean profile | System Prompt 10: depth-resolved profile is a permitted experiment vs Gate 4 metrics. |
| 22 | metronome beacon/anchor | toml [dbscan] | 7DB7 @ ZOI08, anchor ZOI03 | Owner owns reference decision (2026-09-16/10-01 escalations). |

---

## 3. TEST RESULTS (this session, 2026-10-21)

### T2 — clock corrections applied to fish times (working DB `output/jsats3d_2025_tagdrag_ent_clockfix.db`)

Prep: `scripts/_tune_t2_apply_clockfix.py` (kept). Rerun: `fish_multipath -> positions -> export ->
positions_2d -> positions_dbscan -> plot_drag -> plot_drag_2d -> plot_drag_dbscan` (47 min,
Deng 3D 30.8 min). Scored with the identical A/B x variant matrix as T3.

Weighted median XY (m) over ENT-01..05, FC36+FFD3 (weight = scored transmissions):

| 3D variant | baseline | clock-fixed | change |
|---|---|---|---|
| A centroid | 55.00 | 54.50 | ~0 (centroid still outlier-dominated) |
| A consensus/Hampel | 31.12 | 9.77 | -69% |
| A Kalman | 28.46 | 8.82 | -69% |
| B centroid | 39.37 | 8.75 | -78% |
| B consensus/Hampel | 12.15 | 4.22 | -65% |
| **B Kalman** | **11.05** | **4.07** | **-63%** |

| 2D variant | baseline | clock-fixed | change |
|---|---|---|---|
| A centroid | 70.48 | 68.93 | ~0 |
| A Kalman | 21.41 | 7.15 | -67% |
| B centroid | 21.02 | 7.79 | -63% |
| **B Kalman** | **9.55** | **3.83** | **-60%** |

Yield also improved (3D B-Kalman scored 262 -> 459; 2D 422 -> 487). 3D B depth median moved
from +6.1 m (tag at -1.5/-4.6 m) to -1.0 m.

Per-line 3D B-Kalman median XY after correction (FC36 / FFD3, m):
ENT-01 8.5/8.2, ENT-02 6.4/6.9, ENT-03 4.0/3.4, ENT-04 2.6/2.5, ENT-05 2.9/3.5.
Per-line 2D B-Kalman: ENT-01 5.9/6.5, ENT-02 3.5/3.5, ENT-03 3.8/2.7, ENT-04 3.2/2.4, ENT-05 4.5/4.5.

Caveats: CFD01 (in receivers_2d) was left uncorrected (~944 ms late, no curve) — the 2D
numbers are conservative. ZOI09 got curve + 0.946 s constant. Corrections carry the unsurveyed
ZOI08-host bias (10-45 ms per receiver, T1).

### T3 — parameter grids (baseline DB, uncorrected fish times)

Raw results: `output/tuning_2026_10_21/t3_*.csv` (340 screen combos, 27 smoothing, 96 Kalman).
Script: `scripts/_tune_t3_grids.py` (kept — rerun after Step 1 below; optima may shift).

**P0-3 root A vs B:** see F2 table. B wins every variant, both tables.

**P1 position-DBSCAN screen grid (3D root B, 340 combos):**
- Top grid entries are degenerate: the "best" combos keep only 1-6 scored transmissions
  (wmed 2.0 m) — pure overfit of the screen to the truth track. Do not adopt.
- Current production setting (acceleration, eps 8, ms 5, scale 5, dwt 1): scored 110 (~11%
  yield), wmed FC36 16.51 / FFD3 19.74 m.
- Best at yield >= 40%: acceleration eps 12, ms 5, scale 5, dwt 1 -> scored 211, wmed 15.43,
  wp90 41.6. Best at yield >= 60%: eps 20, ms 10, scale 1, dwt 2 -> wmed 18.34.
- No distance-mode combo reaches >= 200 scored at comparable quality; acceleration mode wins.
- Conclusion: at equal yield the screen cannot beat the B-consensus/Kalman aggregation
  (11.05 m at 52% yield, unscreened). The screen is a QC layer, not the accuracy lever.
  Current setting stays; eps 12/ms 5/scale 5/dwt 1 is the only mild improvement candidate.

**P3 smoothing grid (root-B consensus track, 27 combos):**
- Current (jlimit 15, jwin 7, swin 5): wmed ~10.2. Best (10, 11, 3): wmed 9.20, wp90 19.56,
  scored 223. ~1 m improvement; adopt (10, 11, 3) or keep current — small either way.

**P3 Kalman grid (96 combos):** flat. pn 0.5 marginally best (11.01) vs current pn 2.0 (11.05);
all mn/zn/gap within 0.1 m. The 2026-10-19 tuning is fine; no change needed. Consensus/Hampel
does the work; the Kalman stage is nearly a no-op on this data.

---

## 4. IMPLEMENTATION PLAN (ordered)

**Step 1 — F1: apply clock corrections to study-tag times** (biggest lever: -60..70% error,
+75% yield, measured in T2).
- Where: `jsats3d/jsats3d.py`, `multipath_data_object.__init__`, study-tag branch (~line 309).
- Load `tblDetectionClockFixed` (Rec_ID, seconds, seconds_residual); build per-receiver
  `np.interp` over sorted unique knots (left/right = endpoint values); `t = seconds - residual_r(t)`.
- ZOI09: add the 0.946 s constant (see F1; the curve no longer contains it). Keep the constant
  in one named place with the journal reference; re-derive from data later.
- Receivers without a curve: fail loudly (raise) if they appear in `receivers_3d`/`receivers_2d`;
  otherwise log a warning and leave uncorrected. (No silent no-ops.)
- Prototype: `scripts/_tune_t2_apply_clockfix.py` (kept; it works on a DB copy — the production
  version belongs in the core, applied in-memory in `multipath_data_object`).
- 2019 compatibility: 2019 DBs have the same table shape; verify against the 2019 regression
  dataset before merging (System Prompt: legacy parity is non-negotiable). If 2019 residuals
  are non-trivial, gate the correction behind a run-file flag (e.g.
  `apply_clock_to_study_tags = true` in 2025 configs only).
- Validate: rerun on a DB copy: fish_multipath..plot_drag_dbscan; expect T2 numbers
  (3D B-Kalman wmed ~4 m, 2D ~3.8 m; yield ~460-490). Also check transNo histograms
  before/after (epoch assignment should be stable: constant offsets are offset-invariant;
  ZOI02 jump segments may shift a few epoch edges — report the count).

**Step 2 — F2: score/plot root B.**
- `ent_analysis_2025.py` line 277: `solution == "B"` (rename `legacy_a` -> `legacy_b`,
  variant labels "3D legacy B centroid" etc.); `plot_drag_2d_run` `--plots 3D_legacy_B`.
- Keep root A scored as a diagnostic (cheap) so the comparison stays visible in the summary.
- Validate: step 12 only; expect 3D 55 -> ~4.1 m (B Kalman), 2D 21.4 -> ~3.8 m vs the current
  root-A pipeline product.

**Step 3 — re-run the T3 grids on the corrected DB** (`scripts/_tune_t3_grids.py` kept;
point it at the production DB after Steps 1-2). Update:
- screen grid: current production (8/5/5/1) vs eps-12 candidate; pick by yield-matched median,
  never by the top of an unyield-constrained sort (overfit trap, see T3 results),
- smoothing: current (15/7/5) vs (10/11/3) — ~1 m,
- Kalman: keep 2.0/1.0/0.5/20 (grid was flat).
Log before/after per parameter (System Prompt: no silent tuning).

**Step 4 — sweep the ATS fish classifier** (SUSPECT_LATE_S 0.002/0.003/0.005, SUSPECT_WINDOW
5/9/13) on corrected times, T2 protocol. Pick by yield + median XY jointly; log both.

**Step 5 — (optional, owner-gated)** extend clock fix to all 20 receivers (CFD01/ZOI11/CFD09)
for a clean 2D result (CFD01 ~944 ms late, in receivers_2d); deep_solution A vs B for step 6;
adopt_deep_z decision; depth-resolved sound-speed experiment (System Prompt 10); RTK survey of
the ZOI hydrophones (discussed 2026-10-13) — the residual 3-6 m floor is dominated by receiver
position error + unsurveyed beacon host + towed-tag layback, not by tunable software.

## 5. GUARDRAILS (non-negotiable)

- Raw data on `K:\Jobs\5662\001\...` is strictly read-only; all test DBs are local copies in
  `output/`.
- No fabricated sensor values; missing metrics stay NULL.
- Keep physically-impossible solutions (F4).
- Beacon DBSCAN fixed params (knob 5) and reference clock (knob 22) need owner sign-off.
- Every parameter change: journal entry with rationale + before/after validation metric.
- Max 2 edited files per session unless approved (new test scripts are fine; this campaign's
  `_tune_*.py` scripts get deleted after use).
- Commit to `ENM_jsat3d_edits`, never `main`; no PR while a gate is failed/unreviewed.

## 6. OPEN ITEMS FOR OWNER

1. F1 implementation (Step 1) — new pipeline behaviour; needs Kevin's sign-off.
2. ZOI09 946 ms constant: re-derive from data / survey ZOI08 host (RTK already discussed 10-13).
3. Beacon DBSCAN fixed params (0.5 ms / 2.5 / 3) — owner review still pending since 2026-09-24.
4. Reference clock choice (ZOI02 escalation 2026-09-24; ZOI10 anchor 18% unvalidated).
5. adopt_deep_z for production (solved vs config).
6. ZOI02 in receivers_3d (drop recommended 2026-10-19, unapplied).
7. fixed_z_2d depth reference confirmation (toml comment).
