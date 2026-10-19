# 2026-10-14 — Time jump detection and piecewise clock fix (PM Item 5)

Session: 2026-10-14 ~19:00 PDT. Branch `ENM_jsat3d_edits`.
Chinook DB `output/jsats3d_2025_chinook_aug.db`; run file `config/run_data_chinook_aug.toml`.

## TL;DR
1. PM (Kevin Nebiolo) identified that ATS receivers can have clock jumps (GPS sync, counter restarts, offset changes) that break the single-piecewise-linear interpolation in `clock_fix()`.
2. Implemented piecewise interpolation: reads clock jump events from `tblDetectionClockFixed`, quantifies jump magnitudes, splits residual curve into segments, fits separate `interp1d` per segment.
3. All 56 tests pass.

## PM Quote
> "I believe we need to correct for these before running the clock synchronization phase... if one of those ATS time jumps occurs, the linear regression will include that jump and incorrectly apply an offset during that period of time between beacon pings... suggestion would be to create a new function within the time synchronization plots that 1) using the 'Internal' column of data to identify when the GPS clock is synced to first identify when the events occur, and the 2) looks at TDoA (post DBSCAN which removed multipath) looks 5 minutes before and after the event (or up until another jump from either receiver considered) to quantify the magnitude and direction of the time jump. These would need to be recorded, applied, and then piecewise regressions created."

## Investigation Findings

### Internal column availability
- `tblDetectionClockFixed` has 42 columns including all Internal columns (Internal, InternalGroup1-3, InternalFlag, InternalCounter, InternalOffset, InternalStatus, InternalCounterValue, InternalOffsetValue, InternalPositionDifferenceSeconds, OneSecondAdjustmentEvidence, ClockStatusMarker, OffsetChanged, CounterRestart, etc.)
- `tblDetectionRaw` has the same 42 columns — Internal columns ARE preserved through the pipeline
- `tblMetronomeSecondFiltered` does NOT have Internal columns — they are stripped during metronome processing
- **Decision**: Read clock events from `tblDetectionClockFixed` directly in `clock_fix()` since it already has Internal columns

### Clock jump markers found
The `tblDetectionClockFixed` table has three reliable clock jump markers:
- `CounterRestart`: Counter value reset (expected ~100 per receiver)
- `OffsetChanged`: Hydrophone offset change (expected ~15-45 per receiver)
- `OneSecondAdjustmentEvidence`: One-second time adjustment (CFD08: 3233 events, ZOI06: 1670 events, others: 0)

### Clock events per receiver (tblDetectionClockFixed)

| Receiver | CounterRestart | OffsetChanged | OneSecondAdjustment |
|---|---|---|---|
| CFD02 | 49 | 15 | 0 |
| CFD03 | 79 | 51 | 0 |
| CFD04 | 74 | 15 | 0 |
| CFD06 | 62 | 15 | 0 |
| CFD07 | 78 | 15 | 0 |
| CFD08 | 102 | 45 | 3233 |
| ZOI01 | 98 | 13 | 0 |
| ZOI02 | 115 | 30 | 0 |
| ZOI03 | 96 | 21 | 0 |
| ZOI05 | 60 | 16 | 0 |
| ZOI06 | 55 | 40 | 1670 |
| ZOI07 | 54 | 15 | 0 |
| ZOI08 | 91 | 17 | 0 |
| ZOI09 | 46 | 15 | 0 |
| ZOI10 | 58 | 16 | 0 |

### ClockStatusMarker column
- Values are '0', '1', 'y', 'x', 'A', '=', 'F', '1' — NOT the GPS111/RTC222 markers expected
- No GPS/RTC markers found in ClockStatusMarker
- InternalGroup1 has values like '001111' (993 rows in raw data) but no GPS111/RTC222 in this dataset
- **Decision**: Use CounterRestart/OffsetChanged/OneSecondAdjustmentEvidence columns instead of InternalGroup1

### Sample ZOI02 counter restart events
```
seconds=1754502291  CounterRestart=1  InternalGroup1=174459
seconds=1754504865  CounterRestart=1  InternalGroup1=182744
seconds=1754507362  CounterRestart=1  InternalGroup1=190929
seconds=1754507465  CounterRestart=0  OffsetChanged=1  InternalGroup1=191114
seconds=1754509080  CounterRestart=1  OffsetChanged=1  InternalGroup1=1111
```

## Implementation

### Modified file
- `jsats3d/jsats3d.py` — `clock_fix()` function (lines ~1035-1090)

### Changes
Replaced single `interp1d` on `ToA_error` with piecewise interpolation:

1. **Read clock jump events** from `tblDetectionClockFixed` using CounterRestart/OffsetChanged/OneSecondAdjustmentEvidence columns
2. **Identify jump boundaries** — 5 minutes (300s) before/after each event, clipped to segment edges
3. **Quantify jump magnitudes** — mean(ToA_error post-jump) - mean(ToA_error pre-jump), printed per event
4. **Build piecewise interpolator** — segments between jump boundaries, each with separate `interp1d`
5. **Fallback** — if no jumps detected, uses original single-piece interpolation

### Diagnostic output
When clock jumps are detected, `clock_fix()` prints:
```
Clock jump events detected for ZOI02:
  t=1754502291  jump=0.003s  type=CounterRestart  (pre=-0.012, post=-0.009)
  t=1754504865  jump=-0.005s  type=CounterRestart  (pre=-0.009, post=-0.014)
...
  Piecewise interpolator: 245 segments
```

If no jumps are found:
```
No clock jumps detected. Using single-piece interpolation.
```

### Code structure
```python
# Read clock jump events from tblDetectionClockFixed
jump_events = pd.read_sql("""
    SELECT seconds, CounterRestart, OffsetChanged, OneSecondAdjustmentEvidence
    FROM tblDetectionClockFixed
    WHERE Rec_ID = '%s'
      AND (CounterRestart = 1 OR OffsetChanged = 1 OR OneSecondAdjustmentEvidence = 1)
    ORDER BY seconds
""" % current_receiver, conn)

# Build segments at jump boundaries
segment_boundaries = [t_min]
for each jump event:
    add (event_time - 300s) and (event_time + 300s) to boundaries
add t_max
sort and deduplicate boundaries

# Fit separate interpolator per segment
for each segment:
    mask = residual_curve within segment bounds
    if len(seg_data) >= 2:
        piecewise_interps.append((seg_start, seg_end, interp1d(...)))

# Piecewise function
def piecewise_residual(t):
    for each segment:
        mask = t within segment bounds
        result[mask] = segment_interp(t[mask])
    return result
```

## Tests
All 56 tests pass (no new tests added — clock fix is integration-tested via pipeline runs).

## Next steps
1. Rebuild chinook_aug from step 6b with piecewise clock fix and compare clock residuals
2. Compare fish tracks/plots before vs after piecewise fix
3. Consider whether jump magnitudes should be written to a summary table in the DB
4. Consider whether the 5-minute window is appropriate (currently hardcoded as 300s)
