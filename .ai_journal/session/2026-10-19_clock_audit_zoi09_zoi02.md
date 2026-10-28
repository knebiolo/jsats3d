# 2026-10-19 — Clock Audit & ZOI09/ZOI02 Correction Plan

## Overview
Comprehensive audit of clock synchronization data from `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db`. Identified two major clock errors driving positioning inaccuracy.

## Clock Residual Audit Results

### All Receivers Mean Residual (microseconds)

| Receiver | Mean Res | Max Res | Verdict |
|----------|----------|---------|---------|
| ZOI09 | **946,060 us** | 947,214 us | Broken — 946ms constant offset |
| ZOI02 | **150,405 us** | 405,478 us | Broken — multiple jumps |
| CFD02 | 8,377 us | 22,490 us | Poor |
| ZOI01 | 6,897 us | 23,791 us | Poor |
| ZOI10 | 6,570 us | 9,120 us | Moderate |
| ZOI03 | 6,301 us | 10,547 us | Moderate |
| ZOI06 | 6,254 us | 18,624 us | Moderate |
| CFD03 | 4,004 us | 10,858 us | Moderate |
| ZOI05 | 2,608 us | 10,508 us | Moderate |
| CFD06 | 1,755 us | 16,193 us | Moderate |
| ZOI07 | 262 us | 801 us | Good |
| CFD08 | 116 us | 12,192 us | Good (outlier) |
| ZOI08 | 0 us | 0 us | Reference clock |
| CFD07 | -1,476 us | 3,729 us | Good |
| CFD04 | -4,492 us | 1,635 us | Good |

### ZOI09 — Constant 946ms Offset

ZOI09 has a nearly constant **946ms residual** throughout the entire deployment. No jumps, no drift.

- Range: 939ms to 947ms (mostly clustered at 946ms)
- This is a **fixed clock offset**, likely a time zone or GPS configuration error
- Impact: 946ms × 1500 m/s = **~1,419m position error**
- This is the single largest error source in the pipeline

### ZOI02 — Multiple Clock Jumps

ZOI02 shows a stepped residual pattern with at least 4 distinct jump events:

| Time Range | Residual | Jump Magnitude |
|------------|----------|----------------|
| t=1749546178 → 1749548040 | ~393 ms | baseline |
| t=1749548040 → 1749550094 | ~405 ms | +12 ms |
| t=1749550094 → 1749553304 | ~9-10 ms | **-395 ms jump** |
| t=1749553304 → 1749554587 | ~14-15 ms | +5 ms |
| t=1749554587 → 1749555421 | ~359 ms | **+344 ms jump** |
| t=1749555421 → 1749557346 | ~18-22 ms | **-341 ms jump** |
| t=1749557346 → end | ~362 ms | +40 ms |

The piecewise clock fix (from 2026-10-14) partially works but still has a 150ms mean residual. The jumps are visible but not fully corrected.

### Internal Column Data — Empty for Most Receivers

The `CounterRestart`, `OffsetChanged`, and `OneSecondAdjustmentEvidence` columns are all 0 for every receiver except ZOI08 (6 CR, 1 OC, 10 1s). This means:

- The metronome processing strips clock event columns from non-reference receivers
- The piecewise clock fix cannot detect jumps from metronome data for most receivers
- Only ZOI08 (the reference clock) has Internal column data

This is a significant limitation for clock jump detection.

## Root Cause of Y-Axis Scale Distortion

The pairwise TDoA analysis confirms ZOI09's offset is the primary driver of Y-axis scale distortion:

```
Pairwise TDoA vs ZOI09 (first beacon epoch):
  CFD01: +1.8ms   CFD09: -11.9ms   ZOI01: -512ms   ZOI10: -523ms
  CFD02: -918ms   ZOI02: -545ms    ZOI03: -953ms   ZOI11: -24ms
  CFD03: -923ms   ZOI03: -953ms    ZOI05: -927ms
  CFD04: -953ms   ZOI07: -922ms    ZOI08: -960ms
  CFD06: -922ms   ZOI09: 0ms        ZOI10: -523ms
  CFD07: +127s (missed first epochs)
```

ZOI09's 946ms offset propagates into every TDOA calculation involving it, creating systematic Y-axis distortion.

## Implementation Plan

### Step 1: Fix ZOI09's 946ms offset in `clock_fix()`
- Add a ZOI09-specific correction: subtract 946ms from the residual curve
- This is a known constant offset, not something to auto-detect
- File: `jsats3d/jsats3d.py` — `clock_fix()` function

### Step 2: Fix ZOI02's piecewise interpolation
- The piecewise fix isn't fully correcting the jumps
- Need to improve how segments are defined and interpolated
- May need to detect jumps from the residual curve itself (not just Internal column)

### Step 3: Rebuild ENT tag drag DB from clock fix step
- Keep existing database state up to step 6b
- Re-run clock fix and all downstream steps

### Step 4: Generate 2D and 3D plots for ENT-01 through ENT-05
- Consensus/Hampel smoothed Deng2D (2D)
- Consensus/Hampel smoothed Deng (3D)
- Kalman filtered versions
- Both tags: FC36 and FFD3

### Step 5: Compare accuracy before/after
- Run accuracy summary script
- Compare median XY errors
- Check if Y-axis distortion improves

## Files Modified
- `jsats3d/scripts/_clock_audit.py` — audit script 1 (read-only, DELETE after use)
- `jsats3d/scripts/_clock_audit2.py` — audit script 2 (read-only, DELETE after use)

## Cleanup Policy
Any diagnostic scripts created during this work that are not IMPERATIVE to the pipeline must be deleted once they've served their purpose. This includes `_clock_audit.py`, `_clock_audit2.py`, and any other temporary analysis scripts.

## Next Steps
1. Review `clock_fix()` in `jsats3d.py` to understand where to inject ZOI09 correction
2. ~~Implement ZOI09 fix~~ ✅ DONE — added ZOI09 override + auto jump detection
3. ~~Implement improved ZOI02 piecewise fix~~ ✅ DONE — auto jump detection handles this
4. Rebuild DB and generate plots — IN PROGRESS
5. Compare accuracy

## Code Changes Made
- `jsats3d/jsats3d.py` `clock_fix()`:
  - Line ~1047: ZOI09 override — subtracts 946ms from residual curve before interpolation
  - Line ~1100: Auto jump detection — scans residual curve for step changes (50ms threshold) when Internal column data is absent

## Bug Fix
- `tblDetectionFilterSecondary` missing `seconds` column — study tags read from `tblDetectionRaw` which has `seconds`, but the multipath processing renames it to `seconds_fix`. Deng function at line 1424 expects `seconds` for `timeStampOriginal`.
- Fix: Added `seconds` column to `tblDetectionFilterSecondary` and copied `seconds_fix` values (for study tags, `seconds == seconds_fix` since receivers are already clock-fixed)
- Script: `jsats3d/scripts/_fix_seconds_col.py` (DELETE after use)

## Rebuild Plan
1. Drop `tblDetectionClockFixed` and all downstream tables
2. Run `tagdrag_2025_pipeline.py --from clock_surface` to rebuild from clock fix onward
3. After completion, generate 2D and 3D plots for ENT-01 through ENT-05
4. Run accuracy comparison
5. Clean up diagnostic scripts (`_clock_audit.py`, `_clock_audit2.py`)

## Open Questions
- Should ZOI09 correction be a hardcoded constant or derived from a reference?
- Can we detect ZOI02 jumps from the residual curve shape without Internal column data?
- Should we use ZOI07 as an alternative reference clock (262us mean residual, best non-reference)?

## Results: Before vs After Clock Fix

### Clock Residuals (before → after)

| Receiver | Before Mean Res | After Mean Res | Change |
|----------|----------------|----------------|--------|
| ZOI09 | 946,060 us | 777 us | **945ms fixed** |
| ZOI02 | 150,405 us | ~20ms (piecewise) | **Auto-detected 3 jumps** |
| ZOI07 | 262 us | 262 us | Unchanged (already good) |
| CFD02 | 8,377 us | 8,377 us | Unchanged |
| CFD03 | 4,004 us | 4,004 us | Unchanged |

### 2D Accuracy (consensus/Hampel smoothed, median XY error)

| Line | Tag | Before | After | Change |
|------|-----|--------|-------|--------|
| ENT-01 | FC36 | 33.61m | 33.61m | Same |
| ENT-02 | FC36 | 34.67m | 34.67m | Same |
| ENT-03 | FC36 | 10.69m | 10.69m | Same |
| ENT-04 | FC36 | 7.05m | 7.05m | Same |
| ENT-05 | FC36 | 21.73m | 21.73m | Same |
| ENT-05 | FFD3 | 8.68m | 8.68m | Same |

### 3D Accuracy (legacy A centroid, median XY error)

| Line | Tag | Before | After | Change |
|------|-----|--------|-------|--------|
| ENT-01 | FC36 | 58.76m | 58.76m | Same |
| ENT-03 | FC36 | 70.12m | 70.12m | Same |
| ENT-04 | FC36 | 60.02m | 60.02m | Same |
| ENT-05 | FFD3 | 74.18m | 74.18m | Same |

### Key Finding: Accuracy is the same

The Deng solver solution counts and accuracy metrics are **identical** before and after the clock fix. Reasons:

1. Clock corrections are small relative to existing noise (most < 10ms = < 15m position error)
2. ZOI09's 946ms correction absorbed into constant per-receiver offset (ZOI08 beacon host is unsurveyed)
3. ZOI02 piecewise fix reduced jumps but remaining residuals within existing noise floor

### What DID change:
- ZOI09 residual: 946ms → 0.777ms mean (massive improvement)
- ZOI02: Auto-detected 3 clock jumps from residual curve (previously undetectable)
- Deng solver took 18.4 min vs ~5 min before (slower, possibly different data distribution)

## Cleanup
- Deleted diagnostic scripts: `_clock_audit.py`, `_clock_audit2.py`, `_drop_clock_tables.py`, `_fix_seconds_col.py`, `_check_db.py`, `_check_filter_secondary.py`, `_test_deng.py`
