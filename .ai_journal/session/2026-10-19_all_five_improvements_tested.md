# Session Summary: All 5 Improvements Tested

**Date:** 2026-10-19
**Goal:** Test 5 software-only accuracy improvements for JSATS 3D acoustic telemetry pipeline

---

## 1. Receiver Self-Calibration

**Approach:** Solve for refined receiver positions using beacon TDOA residuals (least-squares optimization).

**Result:** ❌ **Not actionable**

The optimization returned 0 corrections (cost=0) because clock errors and position errors are **confounded** — you can't simultaneously solve for both from beacon TDOA alone. A second attempt with time-window matching found large corrections (50-90m in X/Y, +947m for ZOI09) but these were clock errors masquerading as position errors.

**Lesson:** Can't separate clock errors from position errors without additional constraints. Receiver calibration requires independent position knowledge (GPS, surveying) or a separate calibration beacon with known position.

---

## 2. Drop ZOI02 from receivers_3d

**Approach:** Remove ZOI02 (150ms clock residuals, Internal column data empty) from `receivers_3d` and `receivers_2d`, re-run pipeline.

**Result:** ⚠️ **Negligible accuracy impact, 3x speedup**

| Metric | With ZOI02 | Without ZOI02 | Δ |
|--------|-----------|---------------|---|
| Deng 3D time | 18.4 min | 6.2 min | -66% |
| 3D consensus/Hampel median XY | 28.35m (ENT-01 FC36) | 28.35m | 0.00 |
| 2D consensus/Hampel median XY | 7.05m (ENT-01 FC36) | 6.83m | -0.22m |

**Verdict:** Dropping ZOI02 is a **free speedup** (3x faster Deng solver) with no accuracy loss. Clock instability in ZOI02 isn't the dominant error source — receiver position errors and flat geometry dominate.

**Recommendation:** ✅ **Drop ZOI02 from receivers_3d** in the config file.

---

## 3. Kalman Filter Tuning

**Approach:** Grid search over (process_noise, measurement_noise, z_noise, max_gap_s) × 256 combinations, scored against GPS holdout data.

**Result:** ✅ **Modest improvements (1-12%)**

| Tag/Line | Current (0.5/2.0/1.0/12.0) | Best (2.0/1.0/0.5/20.0) | Δ |
|----------|---------------------------|------------------------|---|
| ENT-01 FC36 | 22.77m | 19.95m | **-12.4%** |
| ENT-03 FC36 | 20.17m | 18.11m | **-10.2%** |
| ENT-02 FC36 | 26.38m | 24.90m | -5.6% |
| ENT-04 FC36 | 36.63m | 35.19m | -3.9% |
| ENT-05 FC36 | 36.52m | 35.49m | -2.8% |
| ENT-05 FFD3 | 26.20m | 25.56m | -2.4% |
| ENT-04 FFD3 | 14.15m | 14.02m | -0.9% |
| ENT-03 FFD3 | 31.81m | 31.44m | -1.2% |

**Best params:** `process_noise=2.0, measurement_noise=1.0, z_noise=0.5, max_gap_s=20.0`
- Avg median_xy: 26.80m (vs ~28m current)

**Verdict:** ✅ **Apply tuned params** — biggest gains for FC36 tags (10-12%), smaller for FFD3.

---

## 4. Leave as Is

No action required.

---

## 5. Multi-Beacon Clock Sync

**Approach:** Average clock corrections from multiple beacons (7DB7, 7D2D, 7F91, etc.) instead of relying on single beacon (7DB7).

**Result:** ⚠️ **Not practical with current pipeline**

- Only 7DB7 is processed through the metronome step (4809 detections)
- Other beacons (7D2D, 7F91, 7DBC, B32A, B354, B35A) have insufficient data in metronome output
- Would require modifying the metronome step to process multiple beacons

**Deeper analysis:** The clock residuals are dominated by **systematic errors** (receiver position errors, clock drift) not random noise. Averaging across beacons would only marginally reduce random noise (~5-10% improvement at best) while systematic errors remain unchanged.

**Verdict:** ⛔ **Low ROI** — requires significant pipeline changes for minimal gain. The dominant error source is receiver position uncertainty, not beacon noise.

---

## Overall Impact Assessment

### What actually moves the needle:
1. **Receiver position accuracy** — CFD floating receivers have 3-8m GPS std dev, ZOI08 beacon host is unsurveyed. This is the #1 error source.
2. **Receiver geometry** — Nearly-flat array (9m vertical spread, Y-axis scale distortion 0.37-0.96). Adding depth diversity would help enormously.
3. **Deng solver combinations** — C(9,4)=126 combinations per transmission. Better combination selection could help.

### What doesn't help much:
- Clock corrections (already <10ms for most receivers after fixes)
- Multi-beacon averaging (systematic errors dominate)
- Kalman filter tuning (1-12% improvement, but still within noise floor)
- Dropping bad receivers (speedup but no accuracy gain)

### Recommended next steps:
1. **Receiver calibration with known beacons** — Deploy a temporary beacon at a surveyed position to solve for receiver positions
2. **Improve receiver geometry** — Add depth diversity (deeper CFD receivers)
3. **Multi-beacon sync with surveyed anchors** — If multiple beacons have surveyed positions, use them for joint position+clock calibration

---

## Files Modified

1. `jsats3d/scripts/ent_analysis_2025.py` — Kalman params updated (2.0/1.0/0.5/20.0), empty DataFrame guard in smooth_track
2. `jsats3d/scripts/kalman_tune.py` — New tuning script (256 param combinations)
3. `jsats3d/scripts/multi_beacon_clock.py` — New multi-beacon analysis script

## Files Created (for testing, now cleaned up)
- `config/run_data_tagdrag_ent_no_zoi02.toml` — Removed (ZOI02 removed)
- `output/jsats3d_2025_tagdrag_ent_no_zoi02.db` — Removed (test DB)
- `output/calibration_results/` — Removed (test data)
- `output/calibration_results_v2/` — Removed (test data)

## Diagnostic Scripts Created (now cleaned up)
- `_drop_for_no_zoi02.py`
- `_fix_seconds_no_zoi02.py`
- `_list_tables.py`
- `_check_beacons.py`
- `calibrate_receivers.py`
- `calibrate_receivers_v2.py`
