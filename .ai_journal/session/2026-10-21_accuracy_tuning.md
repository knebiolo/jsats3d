# 2026-10-21 — Accuracy tuning campaign: findings, parameter tests, applied pipeline changes

Session: 2026-10-21 (continuation of onboarding session; see 2026-10-21_onboarding.md).
Branch `ENM_jsat3d_edits`. Working DB `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db`
(window 2025-06-10 09:00–17:00 wall/UTC-encoded; tags FC36 3.038 s / FFD3 3.33 s;
receivers_3d = ZOI01,02,03,05,06,07,08,09,10; receivers_2d adds CFD01,05,09).
Validation truth: drag GPS ENT-01..05 (`K:\...\5_array_testing\array_testing_drag_GPS.csv`,
fallback `data/drag_GPS_2025.csv`), scored exactly as `scripts/ent_analysis_2025.py`
(median/p90 XY per line; weighted aggregates = transmission-weighted).
Env: `C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe`.

## TL;DR
1. **Found and fixed the dominant error source (F1, new — not in the external list):**
   per-receiver clock corrections are computed in steps 2–7 but were **never applied to
   study-tag detections**; fish TDOAs carried raw per-receiver clock offsets (ZOI09 946 ms
   ≈ 1.4 km). Fix (piecewise residual from `tblDetectionClockFixed` + ZOI09 0.946 s constant)
   applied in `multipath_data_object`. Measured (test T2): 3D root-B Kalman weighted median XY
   **11.05 → 4.07 m**, 2D **9.55 → 3.83 m**, scored yield **+75%** (3D 262→459), 3D B depth
   median +6.1 m → −1.0 m (tag at −1.5/−4.6 m).
2. **Root B beats root A everywhere** (F2, confirmed): B has 2.1–2.3× more 3D solutions and
   shallower, realistic Z; A-centroid (what the pipeline was scoring) was the *worst* variant
   (3D 55.0 m weighted median). Scoring/plotting switched A→B.
3. **Smoothing**: (jump_limit, jump_window, smooth_window) = (10, 11, 3) beats (15, 7, 5) by
   ~1 m on the 27-combo grid → applied.
4. **Kalman**: 96-combo grid is flat (all within 0.1 m of the 2026-10-19 values) → **no change**,
   kept 2.0/1.0/0.5/20 with an inline note.
5. **Positions DBSCAN screen**: production setting (acceleration, eps 8, ms 5, scale 5, dwt 1)
   is best at matched yield; the unconstrained grid top is degenerate (1–6 scored transmissions
   = pure overfit) → **no change**.
6. External list's other claims: Z-filter mismatch real but only between two standalone
   diagnostic scripts (−30 vs −15 m), no pipeline impact; Deng not constraining Z is **by design**
   (System Prompt 11.0); its "P3 smoothing 30–300 s / jump 20–100 m" units are hallucinated
   (real params are count windows 15/7/5 + 15 m jump limit).
7. Changes applied to 3 files with inline "why" notes (user-approved this session); validation
   rerun of fish_multipath→…→plot_drag_dbscan launched on the production DB (log
   `output/run_2026_10_21_verify.log`). Expect T2 numbers as the acceptance target.

## Findings (with evidence)

### F1 — Fish timestamps were never clock-corrected (P0, implementation gap)
Trace: `multipath_data_object` (jsats3d.py) reads **`tblDetectionRaw`** for `tagType == 'study'`
and merely renamed `seconds` → `seconds_fix`; `position.__init__` reads
`tblDetectionFilterSecondary`, which stores those uncorrected times; Deng TDOA = t_i − t_ref
therefore carries (δ_i − δ_ref). The clock-fix product (`tblDetectionClockFixed`, beacon basis,
8 h sync window) was only ever used for beacon/deep-receiver tables.

Evidence (test T1, pairwise first-arrival TDOA residuals vs geometry, both drag tags):
- ~940 ms **late** trio {CFD01, ZOI09, ZOI11} vs the main group; main group internal 0–45 ms.
  Matches the 2026-10-19 beacon audit (ZOI09 = 946,060 µs). CFD01/ZOI11 were never flagged in
  10-19 because that audit used ZOI09 as one reference (hides co-lateness).
- CFD01 is in receivers_2d and ZOI11 is excluded; both have **no correction curve** (not in the
  15 clock-audited receivers). CFD01 ≈ +944 ms late → 2D result conservative until a curve exists.

ZOI09 sub-finding: the 2026-10-19 `clock_fix` override subtracts 946 ms from the residual curve
**before** interpolation, so stored `seconds_residual` for ZOI09 is ~0.4 ms and the 946 ms is
*not* in the clock-fix product (verified in DB; pre-fix audit mean 946,060 µs vs post-fix 777 µs
means the applied correction shrank to ~0, not that the data was fixed). This is also why the
10-19 rebuild changed zero accuracy: fish times were never corrected either way.

Physics: range error = c·δ ≈ 1476 m/s × 0.946 s ≈ 1.4 km. Of C(10,4)=210 four-receiver
combinations, C(9,3)=84 (40%) include ZOI09; those solutions were grossly wrong or nonexistent
and poisoned the per-transmission centroid (root-A median Z −42 m, p10 −169 m class; 14–20% of
A rows above water).

Residuals after the fix: corrections carry the unsurveyed ZOI08 beacon-host bias (T1 vs beacon
curve discrepancies 10–45 ms per receiver → 15–65 m class, direction varies); CFD01/CFD09/ZOI11
still uncorrected (no curves). Floor estimate ~2.7–3.3 m (tow layback + GPS 1.2 m + receiver
position error) — measured 4.07 m 3D / 3.83 m 2D is consistent with that floor.

### F2 — Root B beats root A (confirmed, larger than expected)
Baseline DB, weighted medians (see test T3): 3D A-centroid 55.0 / A-Kalman 28.5 /
B-centroid 39.4 / B-Kalman **11.05** m; 2D A-Kalman 21.4 / B-Kalman **9.55** m. B Z shallower
(FC36 +6.1 vs A −42.3; tag −1.5/−4.6), B above-water share 61–65% vs A 14–20%.
The pipeline scored/plotted root A (ent_analysis line 277; plot_drag_2d `--plots 3D_legacy_A`)
— the worst available variant. Note: this is an observation about *this* flat array (9 m
vertical spread) + residual biases, not a universal property; record it as such.

### F3 — Z-filter mismatch (external claim, low severity)
`chinook_analysis_2025.py` (−30..0 m) vs `plot_fish_tracks.py` (−15..0 m, gitignored standalone).
Two separate diagnostic scripts; the ENT scoring path has **no Z filter** (|XYZ|<1000 sentinel
only; its `z_min` is a plot-axis limit). No pipeline result affected. If ever used together,
reconcile to −15..0 (deep receivers at −10..−12 m).

### F4 — Deng does not constrain Z (by design — do not "fix")
`Deng()` flags `in_hull`, never rejects above-water solutions: published 2021 methodology
(System Prompt 11.0) keeps physically impossible solutions as presence data. In this array only
0.6–8% of 3D rows are in-hull (geometry fact). The hull-screened scoring variant exists.

## Tests (all scored vs drag GPS; raw results in `output/tuning_2026_10_21/*.csv`)

### T1 — fish-time clock-offset verification (pairwise TDOA residuals)
Method: first arrivals (det_rank=1, kept) per receiver pair; residual = (t_i−t_ref) −
(|O r_i|−|O r_r|)/c, SoS 1475.7 m/s (median temp 17.91 °C). Origin-dependent geometry term
≤ array span/c (~0.1 ms) — negligible vs ms-scale offsets. Result: the ~940 ms trio above;
beacon curve medians 0–22.4 ms.

### T2 — clock corrections applied to fish times (proof-of-concept DB copy)
Prep: DB copy; per-receiver piecewise `np.interp` of (seconds, seconds_residual) from
tblDetectionClockFixed applied to study-tag `seconds` in tblDetectionRaw; ZOI09 +0.946 s;
CFD01/CFD09/ZOI11 left uncorrected (no curves); position tables dropped; pipeline rerun
`fish_multipath→positions→export→positions_2d→positions_dbscan→plot_drag→plot_drag_2d→plot_drag_dbscan`
(47 min; Deng 3D 30.8 min). Scored with the identical A/B × variant matrix as T3.

| 3D variant | baseline | clock-fixed | Δ |
|---|---|---|---|
| A centroid | 55.00 | 54.50 | ~0 (centroid outlier-dominated) |
| A consensus/Hampel | 31.12 | 9.77 | −69% |
| A Kalman | 28.46 | 8.82 | −69% |
| B centroid | 39.37 | 8.75 | −78% |
| B consensus/Hampel | 12.15 | 4.22 | −65% |
| **B Kalman** | **11.05** | **4.07** | **−63%** |

| 2D variant | baseline | clock-fixed | Δ |
|---|---|---|---|
| A centroid | 70.48 | 68.93 | ~0 |
| A Kalman | 21.41 | 7.15 | −67% |
| B centroid | 21.02 | 7.79 | −63% |
| **B Kalman** | **9.55** | **3.83** | **−60%** |

Per-line 3D B-Kalman median XY after fix (FC36/FFD3 m): ENT-01 8.5/8.2, ENT-02 6.4/6.9,
ENT-03 4.0/3.4, ENT-04 2.6/2.5, ENT-05 2.9/3.5. 2D B-Kalman: 5.9/6.5, 3.5/3.5, 3.8/2.7,
3.2/2.4, 4.5/4.5. Yield: 3D B-Kalman scored 262→459, 2D 422→487.

### T3 — parameter grids (baseline DB, uncorrected times)
- **Root A vs B** (P0-3): F2 table above. B wins every variant, both tables.
- **Screen grid** (P1, 340 combos, 3D root B): unconstrained top = degenerate overfit
  (scored 1–6, wmed 2.0 m — do not adopt). Production (acc 8/5/5/1): scored 110 (~11% yield),
  wmed FC36 16.5 / FFD3 19.7 m. Best at yield ≥40%: acc eps 12/ms 5/scale 5/dwt 1 → scored 211,
  wmed 15.4. Best at ≥60%: eps 20/ms 10/scale 1/dwt 2 → 18.3. No distance-mode combo reaches
  comparable quality at ≥200 scored. At equal yield the unscreened B-consensus/Kalman
  (11.05 m @ 52%) beats the screen → screen is a QC layer, keep production setting.
- **Smoothing** (P3, 27 combos on root-B consensus): best (10, 11, 3) → wmed 9.20, p90 19.56,
  scored 223 vs ~10.2 for current (15, 7, 5) → ~1 m.
- **Kalman** (P3, 96 combos): flat; pn 0.5 marginally best (11.01) vs current 2.0 (11.05);
  all mn/zn/gap within 0.1 m → keep 2026-10-19 values.

## Changes applied this session (files: 3 — user-approved; budget noted per System Prompt)

1. **`jsats3d/jsats3d.py`** — `multipath_data_object.__init__`, study-tag branch (~line 318):
   F1 correction. Builds per-receiver knot arrays from `tblDetectionClockFixed`,
   `seconds_fix = seconds − interp(residual_r(t))` (endpoint hold outside sync window),
   ZOI09 `ZOI09_OFFSET_S = 0.946` added back (documented constant; re-derive after ZOI08
   survey). Missing/empty correction table → explicit NOTE + legacy behavior (2019 parity);
   any other error now **propagates** (no silent no-ops).
   ⚠️ The first implementation of this block (applied earlier today by a parallel session) was
   **broken twice**: (a) `np.interp` was called with only 2 positional args (missing `fp`)
   → TypeError swallowed by a blanket `except` → silent no-op; (b) after fixing (a), the query
   referenced `con` (undefined; the local is `conn`) → NameError swallowed the same way →
   first validation run printed "NOTE: no tblDetectionClockFixed correction rows" and
   proceeded with uncorrected times (caught on a 45 s log check, killed after 2 min). Final
   form: per-receiver knot storage + vectorized `np.interp(t, ts, rs, left=rs[0],
   right=rs[-1])`, table-existence check via `sqlite_master` instead of try/except, and a
   pre-flight construction test before the full rerun (below).
2. **`scripts/ent_analysis_2025.py`** — root A→B scoring (applied earlier today: `legacy_b`,
   label "3D legacy B centroid", token `3D_legacy_B`; Kalman 2.0/1.0/0.5/max_gap 20;
   smooth_track max_gap 8→20 + empty guard). This session: smoothing call
   `smooth_track(..., jump_limit_m=10.0, jump_window=11, smooth_window=3)` with T3 note;
   comment on why Kalman is unchanged (96-combo flat grid).
3. **`scripts/tagdrag_2025_pipeline.py`** — stale root-A plot references → B: `plot_drag_done`
   PNG check, `deep_positions_run` cleanup unlink, `plot_drag_run` blue_source note,
   `plot_drag_2d_run` docstring + `--plots 3D_legacy_B` (was `3D_legacy_A` → matched no variant
   token after the switch; would have produced no 2D plots and a permanently-undone step).
   `plot_drag_dbscan_run` already passed `3D_legacy_B` (latent no-op before root-B scoring
   existed — now works).

Not changed (deliberate): DBSCAN screen config (production setting best at matched yield),
Kalman parameters (flat grid), SUSPECT_WINDOW=9/SUSPECT_LATE_S=0.003 (sweep recommended on
corrected times — follow-up), clock jump threshold 0.05 s (needs full re-clock; owner-gated),
beacon DBSCAN fixed params (owner review pending since 2026-09-24).

## Validation (rerun on production DB, launched 2026-10-21)
Pre-flight (one-off script, deleted): constructed `multipath_data_object('FC36', DB, tmp)` on
the production DB — printed "Applied clock corrections to 32286/39347 fish detections
(3 receivers without curves: CFD01, CFD09, ZOI11)"; ZOI09 raw−corrected shift median
946.4 ms (p5 945.3 / p95 947.1); pairwise ZOI09−ZOI08 TDOA residual on corrected times
median −19.0 ms (p10 −25.4 / p90 −1.1) vs ~+924 ms uncorrected — the −19 ms is the
unsurveyed-ZOI08-host bias class; epoch assignment stable (1993 transmissions, same as T2).
First full attempt aborted (see bug (b) above); relaunched after pre-flight passed.
Prep (one-off `scripts/_verify_prep.py`, deleted after use): dropped tblPositions_Deng/Deng2D/
DengDBSCAN, cleared `__ats_fish_two_stage__` progress, removed work/positions{,_2d}, removed 50
stale root-A review files + stale `_2d` summary. Raw data on K: untouched; tblDetectionRaw
seconds in the production DB stay raw (correction is now in-memory in the core).
Command: `python scripts/tagdrag_2025_pipeline.py config/run_data_tagdrag_ent.toml --only
fish_multipath positions export positions_2d positions_dbscan plot_drag plot_drag_2d
plot_drag_dbscan` (background; log `output/run_2026_10_21_verify.log`).
Acceptance checks: (a) fish_multipath prints "Applied clock corrections to N/N fish
detections" for both tags (N ≈ 39.3k/23.8k; warnings expected only for CFD01/CFD09/ZOI11);
(b) item12 summaries in `output/2025_review/jsats3d_2025_tagdrag_ent_anchorZOI03{,_2d,_dbscan}`
match T2 (3D B-Kalman per-line ≈ 8.5/6.4/4.0/2.6/2.9 FC36; 2D ≈ 5.9/3.5/3.8/3.2/4.5);
(c) root-B PNGs written for all 5 lines × 2 tags.
**RESULT — PASSED (2026-10-21, ~32 min total; positions 25.7 min, positions_2d 4.2 min):**
(a) fish_multipath printed "Applied clock corrections to 20683/23771" (FFD3) and
"32286/39347" (FC36) detections, with the expected warning only for CFD01/CFD09/ZOI11.
(b) Production per-line B-Kalman median XY (m), 3D: ENT-01 8.62/10.67, ENT-02 5.63/6.37,
ENT-03 3.81/3.41, ENT-04 2.62/2.41, ENT-05 3.13/2.95 (FC36/FFD3) → weighted median **3.95 m**
vs 11.05 m baseline (−64%). 2D: 6.11/6.32, 3.66/3.32, 3.49/2.55, 3.27/2.23, 4.41/4.16 →
weighted median **3.73 m** vs 9.55 m baseline (−61%). 3D B depth medians now −2..−6 m
(tag at −1.5/−4.6 m) vs +6 m class before. ENT-01 remains the weak line (geometry at the
drag start + low yield: 9–28 scored transmissions).
(c) Root-B PNGs written for all 5 lines × 2 tags in the 3D, 2D and DBSCAN review folders;
all 16 pipeline steps OK. Summaries: `output/2025_review/jsats3d_2025_tagdrag_ent_anchorZOI03{,_2d,_dbscan}/item12_pipeline_summary.csv`.
Note: production run also uses the new smoothing (10,11,3) — a few lines marginally better
than the T2 proof-of-concept, ENT-01 marginally worse (small n); within noise.

## Cleanup
Deleted: `scripts/_tune_*.py` (6 diagnostics: inspect_db, t1_fish_times, t2_apply_clockfix,
t2_score, t3_grids, t3_check), `scripts/_verify_prep.py`, test DB
`output/jsats3d_2025_tagdrag_ent_clockfix.db`, `output/run_t2_clockfix.log`,
`output/run_t3_grids.log`, `output/2025_review/jsats3d_2025_tagdrag_ent_clockfix*` folders.
Kept: `output/tuning_2026_10_21/*.csv` (test evidence), `docs/accuracy_tuning_handoff_2026-10-21.md`,
`output/run_2026_10_21_verify.log` (validation record).

## Open items for owner
1. F1 sign-off (new core behavior; 2019 DBs get the same treatment — residuals there ~0 so
   ~no-op, but verify against the 2019 regression dataset before merging).
2. ZOI09 946 ms constant: re-derive from data / survey ZOI08 beacon host (RTK discussed 10-13).
3. Extend clock fix to all 20 receivers (CFD01/ZOI11/CFD09 have no curves; CFD01 in
   receivers_2d, ~944 ms late → 2D numbers conservative).
4. Beacon DBSCAN fixed params (0.5 ms / 2.5-period / 3) — owner review pending since 2026-09-24.
5. Reference clock choice (ZOI02 escalation 2026-09-24; ZOI10 anchor 18% unvalidated).
6. SUSPECT_WINDOW/SUSPECT_LATE_S sweep on corrected times (Step 4 in handoff).
7. adopt_deep_z (solved vs config), ZOI02 drop from receivers_3d (3× speedup, no accuracy loss),
   fixed_z_2d depth reference confirmation, depth-resolved sound-speed experiment.
