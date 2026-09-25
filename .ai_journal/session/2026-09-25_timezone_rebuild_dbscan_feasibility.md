# Session: 2026-09-25 — Time-Zone Fix, v2 Rebuild, Pairwise DBSCAN, Clock-Sync Feasibility

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-25, approx. 11:30-14:45 local PDT (UTC-7) = approx. 18:30-21:45 UTC. Rebuild ran 12:33-13:08 PDT (19:33-20:08 UTC).
- Tags: #time-zone #parser #rebuild #dbscan #clock-sync #feasibility #milestone-M1 #milestone-M3 #milestone-M4 #gate-2
- Branch: `ENM_jsat3d_edits`
- Moved from `2026-09-24_onboarding.md` on 2026-09-25 to comply with System Prompt Section 13 (one dated file per session).

## Short Summary
- Found ZOI05 logs UTC (`00z`) while 19 receivers log PDT (`-07z`); earlier "ZOI05 clock fault" finding was wrong.
- Fixed parser (per-file time-zone shift, beacon registry, TagType, GPS UTC labels), adapter (DD_N temperature incl. HOBO, receiver metadata), and DBSCAN script (anchor-side rule, steady-reflection rule, legacy-style outputs, vectorised check).
- Rebuilt `output/jsats3d_2025_v2.db` (59,935,751 detections); deleted superseded `jsats3d_2025_final.db`.
- Two-week pairwise DBSCAN on v2 with all receivers; clock-sync feasibility analysis.
- Verified raw K: data and legacy code untouched this session.

## Files Touched
- `scripts/parse_ats_raw_to_legacy.py`, `scripts/adapt_2025_to_legacy.py`, `scripts/beacon_pairwise_dbscan.py`, `tests/test_2025_contracts.py` (committed in `f7eb0e8`; later vectorisation of `steady_reflection_labels()` uncommitted).
- `LONG_TERM_CONTEXT.md`, this journal, `.ai_journal/session/2026-09-24_onboarding.md` (sections moved out).
- Change budget (2 files) exceeded with user approval (user-directed fix list).

## Active Context
- Database: `output/jsats3d_2025_v2.db` (read-only for analysis).
- Env: `jsat_3d`. Tests: 19 pass.
- Current focus: paper step 5 (clock sync of known-position receivers) — blocked on owner decisions and survey data.

---

## ZOI05 Correction + Paper Workflow Gap Map

### ZOI05 correction (supersedes "clock fault" finding)
- Config `Receiver Time Zone Offset`: ZOI05 `-00z`, all other target receivers `-07z`. Raw headers SR18081: `File Start ... 00z`.
- 48 h slice (06-20/21, 7D2D, anchor ZOI09): unshifted noise 100%, 0 segments; shifted -7 h noise 7.6%, 115 segments, residual RMS 10.2 us.
- All ZOI05 rows in old final DB (study tags included) were 7 h offset. Earlier PM messaging calling ZOI05 a clock fault must be corrected.

### Nebiolo & Meyer (2021) Fig. 3 workflow vs 2025 status
1. Import receivers/HOBO/SCADA -> DB: DONE. Gaps (at time of mapping): temperature was BB_TPU surface sensor; tag-drag HOBO not loaded; WSE 06-05..06-16 only collector level (~0.9 ft offset); UTC_Conv NULL; ZOI05 +7 h; BM_Elev NULL. (Temperature, HOBO, ZOI05 fixed later this session.)
2. Speed of sound (mean of all depths; paper cites Seafloor Systems table, legacy `sos()` comment cites Wikipedia): DD_N mean + legacy `sos()`. Gap: confirm table source with Kevin.
3. Metronome epochs: DONE in pairwise script. True 7D2D spacing 62.7 s.
4. Metronome multipath: phase 1 first arrival DONE; phase 2 = DBSCAN (t, delta) replacing KNN, DIAGNOSTIC.
5. Clock sync of known-position receivers (Eq. 7, piecewise-linear Eq. 10 -> seconds_fix): NOT STARTED.
6. Beacon multipath for receivers at depth (ZOI01-03): NOT STARTED.
7. Position receivers at depth with Deng from own beacons: NOT STARTED.
8. Clock sync receivers at depth: NOT STARTED.
9. Fish tag multipath: NOT STARTED (needs cross-receiver epoch grouping; study PRIs inexact).
10. Position fish, convex hull, retain impossible positions: NOT STARTED.
11. Accuracy (metronome RMSE vs survey), tag drag overlay, precision in/out hull: NOT STARTED.

### Method implication
- Paper master R05 was surface-mounted and surveyed. ZOI02 is on bottom and unsurveyed. Using ZOI02 as beacon source puts its position error into d_Bi - d_Ba, a constant per-receiver offset indistinguishable from clock bias. Either solve ZOI02 position first or use a surveyed surface receiver's beacon.

### Legacy code issues noticed (flag to Kevin, not fixed)
- `position.Deng()`: in-hull test for solution B uses `S1a.item(1)`, `S1a.item(2)` (solution A y/z).
- `position.__init__`: WSEL divided by 3.28084 unconditionally (units check commented out).
- `multipath_classifier()`: `dat[dat.SNR > 0]` drops all ATS rows.

---

## Parser/Adapter/DBSCAN Fixes (bounded validation)

### Parser (`parse_ats_raw_to_legacy.py`)
- Per-file time-zone shift to study basis PDT (UTC-7). Evidence order: GPS-derived (median whole-hour detection-minus-GPS over >=3 pairs) > header `File Start ... NNz` > config `Receiver Time Zone Offset`. Header/config disagreement without GPS raises.
- New columns: `RawDateTime`, `ReceiverUTCOffsetHours`, `TimeShiftHours`, `TimeZoneSource`. `GPSFixTimeStamp` written with `+00:00`.
- `TagType` = `beacon` for config beacon codes, else `study`. `--include-config-beacons` includes array-wide beacons.
- `UTC_Conv` left NULL with warning (owner confirmation pending).

### Adapter (`adapt_2025_to_legacy.py`)
- `load_beacon_registry()` keeps beacons without a receiver name.
- `load_temperature_string()`: mean of all DD_N depths per step (paper method); 10-depth HOBO where complete, 4-depth string after; incomplete steps dropped; `TempSource`, `DepthCount` columns.
- `tblReceiver` adds ZReference, UTCOffset, HydrophoneDepth_ft, HydrophoneOffset_ft, TotalDepth_ft, MountDescription, DeploymentDate. Z unchanged.
- WSE unchanged (forebay CZD only).

### DBSCAN (`beacon_pairwise_dbscan.py`)
- Classes `clean`, `noise`, `steady_reflection`, `anchor_suspect`; residuals on clean only.
- `--output-db` writes legacy-style `tblMetronomeFiltered` / `tblMetronomeSecondFiltered`.

### Validation (scratch)
- `output/scratch_tzfix.db` (06-20/21; CFD09, ZOI02, ZOI05, ZOI08, ZOI09): ZOI05 aligned; array-wide beacons present; TagType beacon.
- DBSCAN on scratch: ZOI05 7.58% noise, 10.2 us; ZOI08 3.48%, 55.3 us; CFD09 9.22%, 223.2 us, 109 steady-reflection epochs, 88 clean >0.5 ms (WARNING). Anchor-side 0 (only 3 detecting receivers).

---

## Full Rebuild (v2) + Two-Week DBSCAN

### Rebuild
- Parser on `K:\...\raw_data`, `--legacy-db --include-config-beacons`, study tags FFD3, FC36, 0B0A, 0AC6, 493F -> `output/jsats3d_2025_v2.db`, ~35 min.
- 285 files, 20/20 serials, 59,935,751 detections, 2,774,924 GPS rows, 943,058 clock-event rows.
- All files resolved by GPS-derived offset. ZOI05 shifted -7 h all season. ZOI04 file `SR20027_250710_140506_recovery_cleaned.csv` (126,426 rows, 07-10..07-16) also logged UTC, shifted -7 h; its override WARNING did not reach the log (worker stdout not captured) — parser logging defect.
- Tag counts identical to old DB for every previously present tag; added 1F14 8,073, 1F38 467, 1F5A 341, 1F71 1,274, 1F94 15,445, 1FCD 15,966, 2010 1,428.
- TagType 39 beacon / 5 study. Temperature DD_N_HOBO 10,936 rows (11.22-18.86 C), DD_N_string 19,881 rows (15.76-22.81 C).

### DBSCAN performance fix
- First v2 run stalled (~18 CPU-min): `steady_reflection_labels()` nested `iterrows` O(k^2). Rewritten with numpy broadcasting; identical outputs at 50 and 300 synthetic clusters; 3.17 s -> 0.013 s at 300. Tests pass.

### Two-week DBSCAN (ZOI02/7D2D, anchor ZOI09, 06-17..07-01)
- 511,208 detections -> 308,960 first arrivals -> 261,254 paired epochs; 18,379 anchor epochs; 1,355 anchor-side epochs set aside.

| Rec_ID | epochs | judged | segments | noise % | steady refl | anchor-side | clean RMS (ms) | clean >0.5 ms |
|---|---|---|---|---|---|---|---|---|
| ZOI08 | 18,351 | 17,000 | 375 | 1.04 | 324 | 1,351 | 0.0505 | 7 |
| ZOI10 | 17,880 | 16,565 | 497 | 2.72 | 225 | 1,315 | 0.0349 | 5 |
| ZOI07 | 18,209 | 16,872 | 526 | 3.44 | 184 | 1,337 | 0.0430 | 7 |
| ZOI04 | 17,875 | 16,569 | 530 | 4.10 | 205 | 1,306 | 0.0452 | 3 |
| ZOI05 | 14,807 | 13,770 | 1,057 | 6.01 | 61 | 1,037 | 0.0250 | 5 |
| ZOI01 | 17,121 | 15,868 | 731 | 8.31 | 159 | 1,253 | 0.0239 | 2 |
| ZOI11 | 17,625 | 16,335 | 725 | 9.38 | 122 | 1,290 | 0.0276 | 2 |
| ZOI06 | 4,192 | 3,883 | 481 | 19.26 | 0 | 309 | 0.0128 | 0 |
| ZOI03 | 7,819 | 7,239 | 513 | 19.66 | 152 | 580 | 0.0299 | 2 |
| CFD04 | 18,025 | 16,693 | 973 | 9.93 | 96 | 1,332 | 0.2291 | 622 |
| CFD09 | 17,255 | 15,998 | 1,321 | 15.26 | 309 | 1,257 | 0.2100 | 422 |
| CFD03 | 15,444 | 14,338 | 1,479 | 18.23 | 199 | 1,106 | 0.1991 | 305 |
| CFD08 | 16,617 | 15,415 | 1,329 | 21.26 | 129 | 1,202 | 0.1991 | 327 |
| CFD06 | 13,928 | 12,914 | 1,432 | 27.70 | 63 | 1,014 | 0.1909 | 126 |
| CFD07 | 16,024 | 14,863 | 1,526 | 30.19 | 122 | 1,161 | 0.1856 | 177 |
| CFD02 | 16,897 | 15,653 | 1,622 | 32.53 | 96 | 1,244 | 0.2058 | 263 |
| CFD01 | 13,185 | 12,227 | 1,464 | 38.26 | 283 | 958 | 0.0405 | 0 |

- vs 2026-09-24 run: ZOI08 noise 7.41% -> 1.04%; ZOI10 8.99% -> 2.72%; ZOI07 9.80% -> 3.44%; CFD04 14.93% -> 9.93%; ZOI05 99.98% -> 6.01%.
- Noise late-share 0.28-0.89: remaining noise predominantly late (multipath at detecting receiver).
- Steady-reflection epochs 61-324 per receiver; not visually audited.
- WARNING (System Prompt 9.2): CFD02/03/04/06/07/08/09 have 126-622 clean epochs >0.5 ms; ZOI units 2-7 each.

---

## PM Geometry Guidance + Clock-Sync/DBSCAN Feasibility

### PM input
- Hydrophone depth = depth at deployment. ZOI04, ZOI05, ZOI06, ZOI10 static; "adjust the other ones". ZOI11 and CFD01 won't help (excluded).
- Interpretation (confirm): static -> fixed elevation = WSE(deployment) - depth; others Z(t) = WSE(t) - depth. ZOI01-03 "On bottom" assumed static.

### Method
- Leave-one-out: each interior clean epoch predicted by linear interpolation (paper Eq. 10) from same-segment neighbours (2x real spacing, conservative).
- Coverage, benign breaks (step <= 0.5 ms), jump intervals (step > 0.5 ms); jump localisation vs Internal flags with chance baseline.
- Precision: TOA covariance with unknown emission time on grid inside hull; sigma = LOO RMS (|err| <= 2 ms); sound speed = legacy sos at window median DD_N (1464.6 m/s).

### Results
| Rec | covered % | benign % | jump-int % | jumps/d | flagged % (chance) | LOO med us | LOO p95 us | >0.5 ms % |
|---|---|---|---|---|---|---|---|---|
| ZOI08 | 91.6 | 5.5 | 2.9 | 10.1 | 81.6 (3.1) | 4.5 | 15.0 | 0.06 |
| ZOI07 | 89.2 | 6.1 | 4.6 | 22.2 | 88.1 (7.4) | 5.0 | 16.0 | 0.05 |
| ZOI10 | 89.6 | 7.2 | 3.1 | 10.9 | 78.3 (2.9) | 5.5 | 18.0 | 0.05 |
| ZOI04 | 87.7 | 6.2 | 6.1 | 16.1 | 85.3 (3.3) | 5.5 | 18.0 | 0.04 |
| ZOI01 | 80.7 | 9.1 | 10.1 | 24.2 | 90.9 (8.1) | 5.4 | 17.5 | 0.02 |
| ZOI05 | 74.1 | 20.1 | 5.8 | 18.9 | 89.4 (5.3) | 6.7 | 20.5 | 0.06 |
| ZOI06 | 17.8 | 11.4 | 70.5 | 9.2 | 79.1 (0.6) | 7.0 | 20.0 | 0.00 |
| ZOI03 | 32.4 | 7.6 | 60.0 | 11.6 | 37.7 (1.9) | 8.3 | 73.7 | 0.13 |
| CFD04 | 80.9 | 8.0 | 11.0 | 45.9 | 17.0 (1.4) | 85.0 | 407.0 | 2.41 |
| CFD09 | 70.8 | 12.7 | 16.5 | 59.0 | 16.8 (2.3) | 83.8 | 482.4 | 4.36 |
| CFD08 | 64.1 | 12.0 | 23.8 | 63.1 | 30.4 (2.3) | 90.0 | 451.5 | 3.58 |
| CFD03 | 61.2 | 13.8 | 25.0 | 69.9 | 26.9 (0.8) | 105.5 | 486.5 | 4.53 |
| CFD06 | 49.4 | 16.1 | 34.4 | 65.2 | 27.5 (1.4) | 158.0 | 524.5 | 6.33 |
| CFD07 | 53.9 | 14.9 | 31.3 | 73.4 | 28.8 (2.0) | 113.7 | 549.4 | 7.16 |
| CFD02 | 54.7 | 13.0 | 32.2 | 83.5 | 19.9 (2.8) | 156.5 | 549.5 | 8.20 |

- ZOI03/ZOI06 low coverage = missing raw files ~06-17..06-27.
- ZOI breaks dominated by 300-1100 ms steps; CFD breaks 48-59% are 0.5-1 ms (noise-induced, bridgeable).
- Within-segment linear-fit RMS: ZOI 7-11 us; CFD 187-238 us (white jitter).
- GPS within-hour sd 0.8-4 m: cannot resolve hydrophone motion.
- Vertical-error sensitivity 38-281 us per m; WSE 1 h change p95 0.029 m -> <= 6 us within a segment. Z model matters for positioning geometry, not sync.
- Timing-only precision inside hull: all ZOI 0.020/0.019/0.110 m; ZOI+CFD 0.025/0.023/0.141 m; random 8 rx 0.08/0.07/0.30; 6 rx 0.12/0.15/1.9; 4 rx 0.39/0.48/6.0. Paper 0.06/0.06/0.12.

### Conclusions
- ZOI clock sync feasible, p95 <= ~20 us, jump times recoverable from flags. CFD feasible but 10-25x noisier; down-weight by sigma.
- Precision can match paper with >= 8 receivers; Z is weak axis.
- Accuracy limited by systematics: receiver positions, unsurveyed ZOI02, static-receiver deployment WSE (before 06-17), sound speed table source.

---

## Integrity Check (end of session)
- K: raw data: only files modified since 2026-09-24 are the PM's Tag Drag Period upload (09-24 14:57), pre-dating first read.
- Legacy code: no commit since 09-16 touches `jsats3d/`, Kevin's drivers, or notebooks. One historical change vs Kevin's `7e3b90e`: `jsats3d.py` `temp_interpolator()` datetime conversion (2 lines, 2026-09-11 session). Flag to Kevin or relocate to adapter.

## Decisions & Assumptions
- Study time basis PDT (UTC-7); `UTC_Conv` stays NULL until owner confirms.
- Study tags for rebuild: FFD3, FC36, 0B0A, 0AC6, 493F (as in prior DB).
- ZOI11 and CFD01 excluded from feasibility per PM.
- ZOI01-03 assumed static (unconfirmed).

## Parameter Changes With Rationale
- Provisional study pulse rates: FC36 3.038 s, 0B0A 3.024 s, 0AC6 3.204 s, 493F 3.155 s (median single-receiver burst spacing, 2026-09-24 audit). Fills NULLs only. PENDING PM APPROVAL.
- DBSCAN additions (fixed, not tuned): `ANCHOR_MAJORITY = 0.5`, `ANCHOR_MIN_RECEIVERS = 4` (3-D solution minimum); `MAX_REFLECTION_DELAY_S = 0.25` (observed later-arrival p99 126 ms; clock steps >= ~390 ms).
- Effect of anchor-side rule (same window): ZOI08 noise 7.41% -> 1.04%, CFD04 14.93% -> 9.93%.
- No change to TIMING_BUDGET_S, TIME_WINDOW_PERIODS, MIN_SAMPLES.

## Blockers & Known Limitations
- Kevin: reference beacon/clock choice; approval of fixed DBSCAN parameters and steady-reflection rule; sound-speed table source; legacy `temp_interpolator` change.
- PM/field: surveyed surface-receiver positions; confirm ZOI01-03 static; deployment-day WSE (06-04/05) for static units; forebay WSE 06-05..06-16; array-wide beacon identities/periods; Spheros upstream filtering documentation; approve UTC_Conv -7 and study pulse rates.
- Parser worker WARNING prints not captured in log.
- ZOI02 not self-synced (beacon host); needs second beacon.

## Cleanup
- Deleted: `output/jsats3d_2025_final.db` (superseded), `output/rebuild_v2.log`, `output/dbscan_v2/`, `output/metronome_v2.db`, `output/current_*.csv`, scratch DBs and DBSCAN folders, `scripts/__pycache__`, `tests/__pycache__`, all %TEMP% scratch scripts.
- Kept: `output/jsats3d_2025_v2.db`, `output/positioning/` (pre-project), all legacy code, all K: data.

## Next Steps
1. Fix parser worker WARNING capture.
2. Obtain Kevin/PM decisions above.
3. Paper step 5 clock correction (Eqs. 7-10) from clean segments, flag-located jumps, per-receiver sigma -> `tblDetectionClockFixed` + Gate 2 residual report.
4. Measure real detections-per-fish-ping to set expected precision.
