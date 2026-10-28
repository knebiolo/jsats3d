# 2025 Chinook Pipeline Parameter Audit & Optimization Plan

**Date:** 2026-10-21
**Purpose:** Identify every tunable parameter in the 2025 Chinook pipeline, document current values, assess whether they've been optimized, and provide testable hypotheses for a Qwen 3.6 model to validate.

---

## Pipeline Overview

```
Step 1: Parse raw files, build DB (run_data.py)
Step 2-3: Metronome - beacon epoch enumeration + multipath ranking (beacon_pairwise_dbscan.py)
Step 4: Surface clock fix (clock_fix)
Step 5: Deep beacon multipath + positioning (multipath_2, Deng)
Step 6: Deep receiver positions (Deng root median)
Step 6b: Adopt deep positions into tblReceiver
Step 7: Deep clocks (rerun metronome + clock fix with adopted positions)
Step 8: Fish multipath - primary ranking + ATS timing classifier (multipath_2, classify_ats_fish)
Step 9: Speed of sound (temperature interpolation)
Step 11: Fish positions (Deng solver, roots A and B)
Step 10: Export positions to tblPositions_Deng
Step 11c: DBSCAN screen on root-B pings (tblPositions_DengDBSCAN)
Step 12: Plots (plot_fish_tracks.py, chinook_analysis_2025.py)
```

---

## PARAMETER CATEGORY 1: Beacon Pairwise DBSCAN (Step 2-3)

**File:** `jsats3d/scripts/beacon_pairwise_dbscan.py`

| Parameter | Current Value | Location | Tunable? | Optimized? |
|-----------|--------------|----------|----------|------------|
| `TIMING_BUDGET_S` | 0.0005 (0.5 ms) | Line 26 | Hardcoded | NOT optimized - based on system timing budget |
| `TIME_WINDOW_PERIODS` | 2.5 | Line 29 | Hardcoded | NOT optimized - accounts for missed transmissions |
| `MIN_SAMPLES` | 3 | Line 31 | Hardcoded | NOT optimized - legacy clock_fix value |
| `ANCHOR_MAJORITY` | 0.5 | Line 33 | Hardcoded | NOT optimized - majority threshold |
| `ANCHOR_MIN_RECEIVERS` | 4 | Line 34 | Hardcoded | NOT optimized - minimum receivers for anchor |
| `MAX_REFLECTION_DELAY_S` | 0.25 (250 ms) | Line 36 | Hardcoded | Based on p99 observation (126 ms) |
| DBSCAN eps | 1.0 (Chebyshev) | Line 178 | Hardcoded | NOT optimized - normalized space |
| DBSCAN min_samples | 3 | Line 178 | Hardcoded | NOT optimized |
| `merge_tolerance_s` | 0.5 * period | Line 130 | Hardcoded | Based on half-period rule |

**Testable Hypotheses:**
1. `MIN_SAMPLES=3` may be too low for noisy environments - test 3, 4, 5
2. `TIME_WINDOW_PERIODS=2.5` may be too loose - test 2.0, 2.5, 3.0
3. `MAX_REFLECTION_DELAY_S=0.25` is safe (p99=126ms) but may miss longer reflections - test 0.25, 0.5
4. `TIMING_BUDGET_S=0.0005` is the clock fix tolerance - test 0.0005, 0.001

**Validation:** Compare `tblMetronomeFiltered` row counts, check clock residuals in `tblDetectionClockFixed`.

---

## PARAMETER CATEGORY 2: Clock Fix DBSCAN (Step 4, 7)

**File:** `jsats3d/jsats3d.py` lines 970-1011

| Parameter | Current Value | Location | Tunable? | Optimized? |
|-----------|--------------|----------|----------|------------|
| Clock fix iterations | 5 | Line 970 | Hardcoded | NOT optimized |
| Iteration 0 DBSCAN eps | 90th percentile of euclidean distances | Line 994 | Hardcoded | Data-driven but NOT tuned |
| Iteration 1+ DBSCAN eps | 95th percentile | Line 998 | Hardcoded | NOT tuned |
| Clock fix DBSCAN min_samples | 3 | Lines 995, 999 | Hardcoded | NOT optimized |
| ZOI09 hardware offset | 0.946 s | Line 1049 | Hardcoded | Applied as known correction |
| Jump detection window | 300 s (5 min) | Line 1078 | Hardcoded | NOT optimized |
| Auto-jump threshold | 0.05 s (50 ms) | Line 1119 | Hardcoded | NOT optimized |
| Auto-jump window size | 30 epochs | Line 1106 | Hardcoded | NOT optimized |

**Testable Hypotheses:**
1. 5 iterations may be excessive - test 3, 5, 7 iterations
2. eps starting at 90th percentile is aggressive - test 85th, 90th, 95th
3. Jump detection at 50 ms may miss small jumps or catch noise - test 20ms, 50ms, 100ms
4. Jump window of 5 min may be too wide/narrow for clock jumps - test 2min, 5min, 10min
5. ZOI09 946ms offset is known but verify it's still valid for 2025 data

**Validation:** Check `seconds_residual` statistics per receiver in `tblDetectionClockFixed`. Target: median residual < 0.1 ms, p95 < 0.5 ms.

---

## PARAMETER CATEGORY 3: Fish Multipath - Primary Ranking (Step 8)

**File:** `jsats3d/jsats3d.py` lines 417-475

| Parameter | Current Value | Location | Tunable? | Optimized? |
|-----------|--------------|----------|----------|------------|
| Multipath ranking | det_rank > 1 = multipath | Line 459 | Hardcoded | Simple rank-based, NOT optimized |
| Duplicate removal | keep='first' | Line 411 | Hardcoded | Legacy default |

**Testable Hypotheses:**
1. Rank-based multipath detection is simplistic - could benefit from signal features (Amplitude, NBW, SNR)
2. `keep='first'` for duplicates may keep wrong detection - test keep='last', keep='best_SNR'

---

## PARAMETER CATEGORY 4: Fish Multipath - ATS Timing Classifier (Step 8)

**File:** `jsats3d/scripts/tagdrag_2025_pipeline.py` lines 328-347

| Parameter | Current Value | Location | Tunable? | Optimized? |
|-----------|--------------|----------|----------|------------|
| `SUSPECT_WINDOW` | 9 transmissions | Line 50 | Hardcoded | NOT optimized |
| `SUSPECT_LATE_S` | 0.003 (3 ms) | Line 51 | Hardcoded | Based on 1.2 ms/ping boat movement |

**Testable Hypotheses:**
1. `SUSPECT_WINDOW=9` may be too narrow for slow-drift reflections - test 5, 9, 15
2. `SUSPECT_LATE_S=0.003` is 3 ms - test 1ms, 3ms, 5ms
3. The boat movement rate (1.2 ms/ping) suggests 3 ms is ~2.5 ping distances - verify this is correct for fish (not boat)

**Validation:** Check `tblDetectionFilterSecondary` dbscan_class distribution. Target: reasonable suspect_reflection rate (not too many, not too few).

---

## PARAMETER CATEGORY 5: Multipath ML Classifier (Legacy, not used for chinook)

**File:** `jsats3d/jsats3d.py` lines 504-703

| Parameter | Current Value | Location | Tunable? | Optimized? |
|-----------|--------------|----------|----------|------------|
| ML Method | KNN | run_data config | Configurable | NOT optimized |
| KNN n_neighbors | 2 | Line 659 | Hardcoded | NOT optimized |
| SVM C | 1E10 | Line 615 | Hardcoded | NOT optimized |
| DBSCAN eps (classifier) | 95th percentile | Line 564 | Hardcoded | Data-driven |
| DBSCAN min_samples (classifier) | 6 | Line 565 | Hardcoded | NOT optimized |
| train_test_split test_size | 0.3 | Lines 602, 607 | Hardcoded | NOT optimized |
| random_state | 111 | Lines 602, 607 | Hardcoded | NOT optimized |
| Chunk size for large data | 10 | Line 547 | Hardcoded | NOT optimized |
| n_neighbors for NearestNeighbors | 20 | Line 555 | Hardcoded | NOT optimized |
| SNR filter | SNR > 0 | Line 517 | Hardcoded | NOT optimized |

**Note:** This classifier is NOT used for chinook_2025 (legacy `run=false`). But if enabled, these parameters need tuning.

---

## PARAMETER CATEGORY 6: Deng Solver (Step 11)

**File:** `jsats3d/jsats3d.py` lines 1385-1640

| Parameter | Current Value | Location | Tunable? | Optimized? |
|-----------|--------------|----------|----------|------------|
| Hull tolerance | 1e-12 | Line 1389 | Hardcoded | NOT optimized |
| Min receivers for solution | 4 | Line 1425 | Hardcoded | Physical minimum |
| water_column flag | false | run_data config | Configurable | NOT applied (legacy output) |
| Solution selection | A and B both written | Lines 1541-1582 | Hardcoded | Both kept, median used |
| Reference receiver selection | First by seconds_fix | Line 1463 | Hardcoded | NOT optimized - could use best geometry |
| Combination strategy | All 4-combos, median | Lines 1442-1495 | Hardcoded | NOT optimized - no weighting |

**Testable Hypotheses:**
1. `water_column=false` means no Z filtering in solver - solutions can be Z>0 or Z<-30m. Setting to true would filter but changes legacy output.
2. Hull tolerance of 1e-12 is extremely tight - test 1e-10, 1e-8, 1e-6
3. No weighting of combinations by geometry (DOP) - all combos treated equally
4. Reference receiver chosen by earliest seconds_fix - could cause issues if that receiver has timing noise
5. Deng solver does NOT constrain Z values - all filtering is post-hoc

**Validation:** Check `tblPositions_Deng` comment distribution. Target: high "solution found" rate, low "negative discriminant" rate.

---

## PARAMETER CATEGORY 7: Deep Receiver Solutions (Step 5-6)

**File:** `jsats3d/scripts/tagdrag_2025_pipeline.py` lines 179-214

| Parameter | Current Value | Location | Tunable? | Optimized? |
|-----------|--------------|----------|----------|------------|
| `deep_solution` root | 'A' | run_data config | Configurable | NOT optimized - test both A and B |
| `adopt_deep_z` mode | 'solved' | run_data config | Configurable | NOT optimized - test 'config' vs 'solved' |

**Testable Hypotheses:**
1. `deep_solution='A'` for chinook - verify if 'B' gives better deep receiver positions
2. `adopt_deep_z='solved'` adopts Deng median Z - test if 'config' (tape-measured) is more accurate
3. Deep receiver Z was off by 7ft (ZOI01), 14ft (ZOI02), 5ft (ZOI03) vs tape (from legacy_pipeline.py line 54)

---

## PARAMETER CATEGORY 8: Position DBSCAN Filtering (Step 11c)

**File:** `jsats3d/scripts/tagdrag_2025_pipeline.py` lines 571-613
**Config:** `run_data_chinook_2025.toml` lines 54-60

| Parameter | Current Value | Location | Tunable? | Optimized? |
|-----------|--------------|----------|----------|------------|
| `positions_dbscan` | true | Line 54 | Configurable | Enabled |
| `dbscan_mode` | 'acceleration' | Line 55 | Configurable | NOT fully optimized - test 'distance' vs 'acceleration' |
| `dbscan_eps_m` | 8.0 | Line 56 | Configurable | NOT optimized |
| `dbscan_min_samples` | 5 | Line 57 | Configurable | NOT optimized |
| `dbscan_accel_scale` | 5.0 | Line 58 | Configurable | NOT optimized |
| `dbscan_depth_weight` | 1.0 | Line 59 | Configurable | NOT optimized |
| `dbscan_accel_gap_s` | 12.0 | Line 60 | Configurable | NOT optimized |

**Testable Hypotheses:**

### 8a. Mode: distance vs acceleration
- **distance mode:** DBSCAN on per-transmission XY mean only (eps=5.0m default)
- **acceleration mode:** DBSCAN on XY + depth + horizontal acceleration (eps=8.0m default)
- **Hypothesis:** acceleration mode is better for fish (captures physics) but may be too aggressive

### 8b. eps_m sweep
- Current: 8.0m (acceleration), 5.0m (distance)
- **Test:** 3.0, 5.0, 8.0, 12.0, 20.0
- **Expected:** Lower eps = fewer kept positions (tighter clustering), higher eps = more positions (risk of outliers)

### 8c. min_samples sweep
- Current: 5
- **Test:** 3, 5, 7, 10
- **Expected:** Higher min_samples = fewer noise points flagged, more kept positions

### 8d. accel_scale sweep
- Current: 5.0 (meters of feature distance per m/s^2)
- **Test:** 1.0, 3.0, 5.0, 10.0
- **Expected:** Higher scale = acceleration matters more in clustering

### 8e. depth_weight sweep
- Current: 1.0 (equal weight to XY and Z)
- **Test:** 0.0 (ignore Z), 0.5, 1.0, 2.0
- **Expected:** Higher weight = deeper positions more likely to be filtered

### 8f. accel_gap_s sweep
- Current: 12.0 seconds
- **Test:** 6.0, 12.0, 24.0, 60.0
- **Expected:** Higher gap_s = acceleration calculated across longer gaps (smoother but less responsive)

**Validation Metrics:**
- % of transmissions kept per tag
- XY p90 error vs GPS (for drag tags)
- Z distribution (should be -15 to 0m for fish)
- Track continuity (no wild jumps)
- Start/end consistency

---

## PARAMETER CATEGORY 9: Coordinate Filtering (Post-processing)

**File:** `jsats3d/scripts/chinook_analysis_2025.py` lines 33-35
**File:** `jsats3d/scripts/plot_fish_tracks.py` lines 38-42

| Parameter | Analysis Script | Plot Script | Tunable? | Optimized? |
|-----------|----------------|-------------|----------|------------|
| X_MIN | 420 | 420 | Hardcoded | Based on array bounds |
| X_MAX | 560 | 560 | Hardcoded | Based on array bounds |
| Y_MIN | 20 | 20 | Hardcoded | Based on array bounds |
| Y_MAX | 160 | 160 | Hardcoded | Based on array bounds |
| Z_MIN | -30 | -15 | Hardcoded | Z differs between scripts! |
| Z_MAX | 0 | 0 | Hardcoded | Surface |

**Issues Found:**
1. **Z_MIN mismatch:** Analysis uses -30m, plotting uses -15m. This means the two outputs are filtering differently!
2. **Hardcoded bounds:** Not in config file, not documented as tunable
3. **Receiver array extent:** X:437-540, Y:37-140, Z:-12 to -3. Bounds are ~15-20m beyond array edges.

**Testable Hypotheses:**
1. Z_MIN=-30 vs -15: Fish typically swim in upper water column. -15m may be more realistic. Test -10, -15, -20, -30.
2. Expand X bounds to 410-570 to capture edge movements
3. Expand Y bounds to 15-165 for same reason
4. Tighten bounds to X:430-545, Y:35-145 to keep fish strictly within array

---

## PARAMETER CATEGORY 10: Smoothing & Gap Filtering (Plotting Only)

**File:** `jsats3d/scripts/plot_fish_tracks.py`

| Parameter | Current Value | Tunable? | Optimized? |
|-----------|--------------|----------|------------|
| Smoothing window | 120s | CLI arg | NOT optimized |
| Jump threshold | 40m | Hardcoded | NOT optimized |
| Gap threshold | 300s (5 min) | Hardcoded | NOT optimized |

**Testable Hypotheses:**
1. Smoothing window: test 30s, 60s, 120s, 300s
   - Shorter = more detail, noisier
   - Longer = smoother, may oversmooth real movements
2. Jump threshold: test 20m, 40m, 60m, 100m
   - Fish swimming speed ~0.5-2 m/s, so 40m jump in one ping is likely an outlier
3. Gap threshold: test 120s, 300s, 600s
   - Fish may have gaps when outside detection range

---

## PARAMETER CATEGORY 11: Receiver Selection

**File:** `run_data_chinook_2025.toml`

| Parameter | Current Value | Tunable? | Optimized? |
|-----------|--------------|----------|------------|
| `surface_receivers` | 12 receivers | Configurable | NOT optimized |
| `deep_receivers` | ZOI01, ZOI02, ZOI03 | Configurable | NOT optimized |
| `receivers_3d` | 9 receivers | Configurable | NOT optimized |
| `receivers_2d` | 12 receivers | Configurable | NOT optimized |
| `deng_surface` | false | Configurable | NOT optimized - test true |

**Testable Hypotheses:**
1. `deng_surface=false` means only deep receivers used for clock reference. Test true - may improve clock fix.
2. Missing CFD receivers from `receivers_3d` - CFD01, CFD05, CFD09 not in 3D array but in 2D. Verify this is intentional.
3. ZOI04, ZOI06, ZOI10, ZOI11 not in receivers_3d - verify these are surface/floating receivers

---

## PARAMETER CATEGORY 12: Speed of Sound / Temperature

**File:** `jsats3d/jsats3d.py` lines 1501-1507

| Parameter | Current Value | Tunable? | Optimized? |
|-----------|--------------|----------|------------|
| Temp interpolator | 'linear' | Hardcoded | NOT optimized - test 'cubic' |
| SOS calculation | Freshwater table | Hardcoded | Based on temp |

**Testable Hypotheses:**
1. Linear interpolation may be too simple for temperature variations - test cubic
2. SOS table assumes standard freshwater - verify salinity/temp conditions match

---

## PARAMETER CATEGORY 13: Water Column Filtering

**File:** `jsats3d/jsats3d.py` lines 1386-1392, 1624-1631

| Parameter | Current Value | Tunable? | Optimized? |
|-----------|--------------|----------|------------|
| water_column flag | false | Configurable | NOT applied |
| Water surface check | WSEL > 0 | Hardcoded | Uses tblWSEL |
| Z floor | lowest_receiver_Z - 4m | Hardcoded | NOT optimized |

**Testable Hypotheses:**
1. Setting `water_column=true` would flag solutions above surface or >4m below deepest receiver
2. Z floor of -4m below deepest receiver may be too shallow (receivers at -12m, so floor=-16m)
3. Consider: Z floor should be -30m (reasonable water column depth) not -4m below receiver

---

## PARAMETER CATEGORY 14: Pulse Rate Configuration

**File:** `run_data_chinook_2025.toml` line 36

| Parameter | Current Value | Tunable? | Optimized? |
|-----------|--------------|----------|------------|
| `pulse_rates` | {} (empty) | Configurable | Pulse rates added manually during debugging |

**Note:** Pulse rates were added manually during debugging (3.155 Hz for 0FBD, 041D, 5F82). Should be in config file.

---

## SUMMARY: Priority Parameter Optimization Order

### P0 - Must Fix (Wrong or Inconsistent)
1. **Z_MIN mismatch** between analysis (-30) and plotting (-15) scripts - must be consistent
2. **Deng solver water_column filter** not applied - solutions can be physically impossible (Z>0)
3. **Solution selection** - A vs B not explicitly chosen; B has more positions and shallower Z

### P1 - High Impact (Should Test)
4. **DBSCAN eps_m sweep** (3, 5, 8, 12, 20) - biggest impact on position quality
5. **DBSCAN min_samples sweep** (3, 5, 7, 10)
6. **DBSCAN mode** - test 'distance' vs 'acceleration'
7. **accel_scale sweep** (1, 3, 5, 10)
8. **depth_weight sweep** (0, 0.5, 1.0, 2.0)

### P2 - Medium Impact (Test if P1 passes)
9. **deep_solution** - test A vs B for deep receiver positioning
10. **deng_surface** - test false vs true
11. **SUSPECT_WINDOW** (5, 9, 15) and **SUSPECT_LATE_S** (0.001, 0.003, 0.005)
12. **Clock fix jump detection** threshold (20ms, 50ms, 100ms)

### P3 - Low Impact / Plotting Only
13. **Smoothing window** (30, 60, 120, 300) - affects plot quality, not positions
14. **Jump threshold** (20, 40, 60, 100) - affects plot quality, not positions
15. **Coordinate bounds** - tighten or expand

---

## Recommended Testing Strategy

### Phase 1: Quick Validation (1-2 hours)
```python
# Test 1: Verify Z filtering consistency
# - Check Z distributions in both scripts with same bounds
# - Confirm all positions are Z <= 0 and Z >= -30m

# Test 2: A vs B comparison
# - For each tag, compare A and B position counts, Z medians
# - Decide which solution to use as primary

# Test 3: DBSCAN mode comparison
# - Run positions_dbscan with 'distance' and 'acceleration'
# - Compare kept % and position quality
```

### Phase 2: Parameter Sweep (4-8 hours)
```python
# Test 4: Grid search on DBSCAN parameters
# - eps_m: [3, 5, 8, 12]
# - min_samples: [3, 5, 7]
# - For each combo, measure:
#   - % transmissions kept
#   - Position density (positions per hour)
#   - Z distribution (median, p5, p95)
#   - Track smoothness (mean acceleration)
#   - Physical validity (% Z in [-15, 0])
```

### Phase 3: Validation (2-4 hours)
```python
# Test 5: Best parameter set validation
# - Run with best parameters from Phase 2
# - Generate plots and XYZ exports
# - Compare to current output
# - Check start/end consistency
# - Verify fish end near same area
```

---

## Files to Modify for Testing

1. **`config/run_data_chinook_2025.toml`** - Add tunable parameters to config
2. **`scripts/tagdrag_2025_pipeline.py`** - Read params from config instead of hardcoded
3. **`scripts/chinook_analysis_2025.py`** - Fix Z_MIN to match plot script
4. **`scripts/plot_fish_tracks.py`** - Make bounds configurable, add Z_MIN to config
5. **`jsats3d/jsats3d.py`** - Consider adding water_column filter by default

---

## Journal Entry Reference

- Previous journal: `2026-10-13_chinook_accel_dbscan.md`
- Old plotting script: `plot_fish_tracks.py` (deleted, recreated with improved features)
- DB: `output/jsats3d_2025_chinook.db` (13.9 GB, 4 tags)
- Current DBSCAN params: mode='acceleration', eps=8.0, min_samples=5, accel_scale=5.0, depth_weight=1.0, gap_s=12.0
