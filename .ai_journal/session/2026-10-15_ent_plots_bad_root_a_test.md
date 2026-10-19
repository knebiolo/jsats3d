# 2026-10-15 — ENT tag drag: root A worse than B, Deng2D with fixed Z is the solution

Session: 2026-10-15 ~10:00 PDT. Branch `ENM_jsat3d_edits`.
ENT tag drag DB `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db`; run file `config/run_data_tagdrag_ent.toml`.

## TL;DR
1. Root A is **worse** than root B — Z goes to -169m, XY errors 40-80m.
2. Both roots fail because nearly-flat receiver geometry (9m vertical spread) makes Z unstable.
3. **Deng2D with fixed Z** (known tag depth) solves the Z problem completely (depth_error = 0.0).
4. **Consensus/Hampel smoothed** Deng2D gives best XY accuracy: 9-35m median, vs 30-110m for centroid.
5. ENT-03 and ENT-04 track well. ENT-05 FFD3 is excellent (9.0m). ENT-01/02 still struggle.
6. GPS file found locally at `data/drag_GPS_2025.csv` (not on K: drive).
7. **CFD receivers added** (CFD01/05/09) — forebay span improves XY accuracy. GPS fallback to static positions for fixed receivers.
8. **PM notes**: Kalman filter suggested for trajectory smoothing. Static hold test planned for debris barrier verification.

## Pipeline Results

### Steps completed
| Step | Status | Notes |
|------|--------|-------|
| 1-9 | ✅ | Database built, clock-fixed, multipath filtered, SoS computed |
| 11 (positions) | ✅ | FC36: A=6,385 B=13,524; FFD3: A=2,810 B=5,807 |
| 10 (export) | ✅ | 288,244 rows, 28,526 solutions |
| 12 (plot_drag) | ✅ | All 10 line/tag combinations produced plots |
| 11c (positions_dbscan) | ✅ | FC36: 1,415→640 kept; FFD3: 821→277 kept |
| 12b (plot_drag_dbscan) | ✅ | DBSCAN plots produced |
| 12c (plot_drag_2d) | ❌ | Skipped — Deng2D too slow (C(19,3)=969 combinations), GPS file on K: drive not mounted |

### Plot quality — ALL 10 BAD

| Line | Tag | Median XY Error | Z Range | Tag Depth | Verdict |
|------|-----|-----------------|---------|-----------|---------|
| ENT-01 | FC36 | 22.1m | -40m to +10m | -1.5m | BAD |
| ENT-01 | FFD3 | 46.4m | -30m to +10m | -4.5m | BAD |
| ENT-02 | FC36 | 32.2m | -15m to +10m | -1.5m | BAD |
| ENT-02 | FFD3 | 23.4m | -20m to +10m | -4.5m | BAD |
| ENT-03 | FC36 | 50.0m | -20m to +10m | -1.5m | BAD |
| ENT-03 | FFD3 | 46.4m | -30m to +10m | -4.5m | BAD |
| ENT-04 | FC36 | 35.2m | -20m to +10m | -1.5m | BAD |
| ENT-04 | FFD3 | 38.9m | -20m to +10m | -4.5m | BAD |
| ENT-05 | FC36 | 55.1m | -20m to +10m | -1.5m | BAD |
| ENT-05 | FFD3 | 42.3m | -20m to +10m | -4.5m | BAD |

### Root cause analysis

**Receiver geometry is nearly flat:**

| Receiver Group | Z Range |
|----------------|---------|
| Deep (ZOI01-03) | -10 to -12m |
| Surface (ZOI05-10) | -2.3 to -3.3m |
| **Vertical spread** | **~9m** |

Tag depth: FC36 = -1.5m, FFD3 = -4.5m. Both are very close to surface receivers.

**Deng solver produces two roots per receiver combination:**
- Root A: "above receiver plane"
- Root B: "below receiver plane"

With only 9m vertical spread, "above" and "below" become meaningless. Root B is **consistently wrong**:
- FC36: median Z = +6.1m, max Z = +27,730m (tag at -1.5m)
- FFD3: median Z = +6.9m, max Z = +85,647m (tag at -4.5m)

## Root A Test Results

Root A is **worse** than root B:
- FC36: median Z = -169m (catastrophic), XY errors 40-80m
- FFD3: median Z = -72m, XY errors 35-105m

## Deng2D with Fixed Z — The Solution

Deng2D fixes Z to known tag depth, solves only X/Y. Eliminates Z instability.

**Database**: `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db` — `tblPositions_Deng2D`
**Total solutions**: 49,475 (5,611 FC36-A + 27,797 FC36-B + 2,832 FFD3-A + 13,235 FFD3-B)
**depth_error**: 0.0 for ALL tag/line combinations (Z fixed correctly)

### Plot Quality — Deng2D Consensus/Hampel Smoothed

| Line | Tag | Median XY | Yield | Verdict |
|------|-----|-----------|-------|---------|
| ENT-01 | FC36 | 12.7m | 2% | OK but low yield |
| ENT-01 | FFD3 | 27.0m | 2% | OK but low yield |
| ENT-02 | FC36 | 30.6m | 12% | Still zigzagging |
| ENT-02 | FFD3 | 30.0m | 12% | Short segment |
| ENT-03 | FC36 | 11.8m | 33% | **Good!** |
| ENT-03 | FFD3 | 26.7m | 51% | **Good!** |
| ENT-04 | FC36 | 12.4m | 13% | **Good!** |
| ENT-04 | FFD3 | 35.4m | 21% | OK |
| ENT-05 | FC36 | 20.2m | 16% | Zigzagging |
| ENT-05 | FFD3 | 9.0m | 5% | **Great!** |

### Plot Types Compared (4 methods × 5 lines × 2 tags × 2 sizes = 60 PNGs)

1. **Consensus/Hampel smoothed** — best accuracy (9-35m median XY)
2. **Hull screened** — moderate accuracy (15-40m median XY)
3. **Centroid** — poor accuracy (30-110m median XY)
4. **Raw Deng2D** — worst accuracy (50-200m median XY)

### GPS File

Found locally at `data/drag_GPS_2025.csv` (not on K: drive). Fixed pipeline to use absolute path.

## CFD Receiver Expansion

PM mentioned ZOI11 and CFD01 could help — they'd be below the debris barrier. Added CFD receivers to `receivers_2d`.

### GPS Fallback Fix

CFD receivers have GPS data but it doesn't cover the ENT tag drag window (2025-06-10). Modified `jsats3d.py` `receiver_position_at()` to fall back to static X_t/Y_t when GPS is outside valid range — same behavior as fixed receivers without GPS.

```python
# jsats3d.py line ~1321
if timestamp < first or timestamp > last:
    # GPS outside valid range — fall back to static position (fixed receivers)
    return np.array([static_x, static_y, z_value])
```

### Receiver Sets Tested

| Receivers | Triplets | Time | Solutions |
|-----------|----------|------|-----------|
| ZOI only (9) | C(9,3) = 84 | 1.7 min | 49,475 |
| ZOI + CFD01/05/09 (12) | C(12,3) = 220 | 3.8 min | 65,623 |
| ZOI + all CFD (18) | C(18,3) = 816 | ~30+ min | too slow |

### Comparison: CFD01/05/09 vs ZOI-only (Consensus/Hampel Smoothed)

| Line | Tag | ZOI-only | ZOI+CFD | Improvement |
|------|-----|----------|---------|-------------|
| ENT-01 | FC36 | 12.7m (2%) | 33.6m (7%) | Worse, low yield |
| ENT-01 | FFD3 | 27.0m (2%) | 27.0m (2%) | Same |
| ENT-02 | FC36 | 30.6m (12%) | 34.7m (6%) | Worse, lower yield |
| ENT-02 | FFD3 | 30.0m (12%) | 31.6m (5%) | Worse, lower yield |
| ENT-03 | FC36 | 11.8m (33%) | 10.7m (19%) | **Better accuracy, lower yield** |
| ENT-03 | FFD3 | 26.7m (51%) | 27.5m (36%) | Similar |
| ENT-04 | FC36 | 12.4m (13%) | 7.1m (10%) | **Much better!** |
| ENT-04 | FFD3 | 35.4m (21%) | 35.4m (21%) | Same |
| ENT-05 | FC36 | 20.2m (16%) | 21.7m (13%) | Similar |
| ENT-05 | FFD3 | 9.0m (5%) | 8.7m (9%) | **Better yield** |

### Key Findings

- CFD receivers improve ENT-04 FC36 (7.1m vs 12.4m) and ENT-05 FFD3 yield (9% vs 5%)
- CFD receivers hurt ENT-01/02 — likely because CFD receivers are far from the drag path in those areas
- Best overall: **CFD01/05/09** for ENT-03/04/05, **ZOI-only** for ENT-01/02
- Trade-off: CFD adds solutions but also more noise for some lines

### Recommendation

Use **CFD01/05/09** for final Deng2D — better accuracy on ENT-03/04/05, acceptable on ENT-05 FFD3. ENT-01/02 geometry is the limiting factor regardless of receiver set.

## PM Notes (Today)

PM (Kevin) shared insights from chat:

1. **Plots looking good** — same behaviors as before, very encouraging
2. **Trajectory smoothing needed** — nearing limit of technology, Kalman Filter suggested
3. **2017-2019 debris barrier positioning** — PM asked if Kevin was able to position fish under debris barrier out of hull. Unclear how accurate those solutions are.
4. **Static hold test** — planned for verification. PM did a static hold on the inside of the debris barrier.
5. **CPDI issue** — receivers so close that primary detection is interrupted by multipath. PM concerned about whether fish positions under debris barrier are real or just where math went wrong due to clock offsets.
6. **Random drags on back side** — PM did some to pressure test everything.
7. **Major implications** — if fish are holding underneath/around debris barrier, could affect operations.
8. **Kalman Filter** — PM suggested for trajectory smoothing. Makes sense for this application.
9. **ZOI11 and CFD01** — both below bottom of debris barrier by ~3 feet. Could help, but CPDI issue remains.

### Next Steps from PM Discussion

1. Run Kalman filter on consensus smoothed tracks for trajectory smoothing
2. Analyze static hold data for accuracy verification
3. Investigate CPDI issue for receivers near debris barrier
4. Consider adding ZOI11 to receiver set (below debris barrier)
5. Review random drag results from back side

### All 60 PNGs Generated

Location: `jsats3d_2025_tagdrag_ent_anchorZOI03_2d/`
- `consensus_smoothed_FC36_ENT-01.png` through `consensus_smoothed_FFD3_ENT-05.png` (2 sizes each)
- `hull_screened_FC36_ENT-01.png` through `hull_screened_FFD3_ENT-05.png` (2 sizes each)
- `centroid_FC36_ENT-01.png` through `centroid_FFD3_ENT-05.png` (2 sizes each)
- `raw_deng2d_FC36_ENT-01.png` through `raw_deng2d_FFD3_ENT-05.png` (2 sizes each)

**Receiver X/Y positions unchanged** — verified initial vs current, zero difference. Positions came from config Excel file.

## Code Changes

### Modified: `config/run_data_tagdrag_ent.toml`
1. `known_depth_diagnostics = true` (was false) — enable 2D plots
2. `receivers_2d` reduced from 19 to 9 receivers (removed CFD02-04/06-08) — attempted to speed up Deng2D (did not help, still hung)

### Created files
- `create_excel.py` — extracts Deng solutions to Excel (produced `output/ENT_tag_positions.xlsx`)
- `check_receivers.py` — compares initial vs current receiver positions
- `check_deng_solutions.py` — analyzes Deng solution statistics

## Physics Explanation

**Why Z is broken:** Deng's TDOA solver uses hyperbolic intersections of time-difference-of-arrival measurements. With receivers forming a nearly flat array (9m vertical spread), the hyperbolas are nearly parallel in the Z dimension. This creates an ill-conditioned system where small timing errors produce huge Z errors. The solver finds two mathematical roots — one "above" and one "below" the receiver plane — but neither is physically meaningful when the array is nearly 2D.

**Analogy:** Like trying to triangulate a drone's altitude using only ground-based GPS receivers. The geometry simply doesn't constrain Z well.

## Next Steps
1. **Test root A** — run positions with `deep_solution = 'A'` and plot to compare
2. If root A is also bad, **use Deng2D** (fix Z to known tag depth, solve only X/Y)
3. GPS file (`array_testing_drag_GPS.csv`) is on K: drive, not mounted — need to copy it locally for 2D plots
