# 2026-10-24: Early Fish Pipeline Fix + Fish Figures

**Session:** Continuation of 2026-10-22/23 early fish work.
**Model:** dell/coder-fast (Qwen3.6 35B MoE)

## TL;DR
1. **Fixed `deep_beacons_done()`** — was checking for beacon rows in `tblDetectionFilterSecondary`, but `fish_multipath` (step 8) clears that table before writing fish-only rows. This made the step non-idempotent: after step 8 ran, the done check returned False, triggering a re-run that failed with `DatabaseError: Execution failed`. Changed the check to look for `tblDetectionClockFixed` rows for deep receivers instead.
2. **Data integrity confirmed** — all 4 fish have 203,623 root-B 3D solutions (26E4: 19,897, 9F8C: 1,654, 1D5A: 51,859, B5ED: 37,263). DBSCAN filtered: 190,727 rows. Root B Z medians are realistic (-1.6 to 0.1 m). Root A has extreme Z outliers (median -17 to -45 m).
3. **Fish figures generated** — 20 files (5 per tag: plan_time, plan_depth, 3d, track.gif, positions.csv) + fish_qc.csv under `output/pm_figures_2026-10-22/example_fish/`.
4. **Pipeline status.json updated** — deep_beacons marked "ok" with notes about partial beacon coverage (FFC7 at ZOI01 had 659k primary rows; 7D2D/7DBC at ZOI02/ZOI03 had 0 — beacons not detected at surface receivers).

## Deep Beacons Issue (Root Cause Analysis)

### What happened
The initial pipeline run (2026-10-24 16:10-17:01) completed all steps. A later re-run (17:30) attempted `deep_beacons` and failed with `DatabaseError: Execution failed` on the `to_sql(..., if_exists="append")` call in `filter_deep_beacon()`.

### Root cause
`deep_beacons_done()` at line 145-146 of `tagdrag_2025_pipeline.py`:
```python
def deep_beacons_done(c):
    return all(count(c.db, "tblDetectionFilterSecondary", "where Tag_ID = ?", (t,)) > 0
               for t in c.deng_tags.values())
```

This checks for deng_tags (beacon tags: FFC7, 7D2D, 7DBC) in `tblDetectionFilterSecondary`. After `fish_multipath` (step 8) runs, it deletes ALL rows from `tblDetectionFilterSecondary` for study tags and writes fish-only rows. The beacon rows are gone, so `deep_beacons_done()` returns False, triggering a re-run.

The re-run fails because:
- `filter_deep_beacon()` tries to process beacon tags that have no clock-fixed detections (7D2D/7DBC had 0 primary rows)
- OR the `to_sql(..., if_exists="append")` fails due to schema/data conflicts

### Fix applied
Changed `deep_beacons_done()` to check for downstream products instead:
```python
def deep_beacons_done(c):
    return all(count(c.db, "tblDetectionClockFixed", "where Rec_ID = ?", (r,)) > 0
               for r in c.deep)
```

This checks for clock-fixed rows for deep receivers (ZOI01, ZOI02, ZOI03), which are the actual output of the deep_beacons→deep_positions→deep_clocks chain.

### Beacon coverage note
Only FFC7 (ZOI01's beacon) was detected at surface receivers (659,040 primary rows). 7D2D (ZOI02) and 7DBC (ZOI03) had 0 primary rows — their beacons are likely at depths where surface receivers can't detect them. This is expected for deep-bottom beacons.

## Fish Figures Generated

### Output files
- `output/pm_figures_2026-10-22/example_fish/26E4_plan_time.png`
- `output/pm_figures_2026-10-22/example_fish/26E4_plan_depth.png`
- `output/pm_figures_2026-10-22/example_fish/26E4_3d.png`
- `output/pm_figures_2026-10-22/example_fish/26E4_track.gif` (150 frames)
- `output/pm_figures_2026-10-22/example_fish/26E4_positions.csv` (19,897 solution + 1,975 consensus rows)
- Same pattern for 9F8C, 1D5A, B5ED (20 files total)

### Track summaries (FIRST PASS — superseded; see "Follow-up" below for corrected QC)

| Tag | Solutions | Transmissions | Span | Max Speed | Jumps >3 m/s |
|-----|-----------|--------------|------|-----------|-------------|
| 26E4 | 19,897 | 1,975 | 2.2 h | 3,329 m/s* | 227 |
| 9F8C | 1,654 | 182 | 0.3 h | 30.86 m/s | 62 |
| 1D5A | 51,859 | 7,555 | 18.1 h | 539 m/s* | 1,170 |
| B5ED | 37,263 | 4,646 | 12.1 h | 53,351 m/s* | 993 |

*Extreme speeds are from Z-outlier positions (Root B max Z: 2,186 to 872,424 m). These are physically impossible and should be filtered. The DBSCAN-filtered data is much cleaner.

### Data quality notes
- **Root B** is the correct solution (validated by ENT drag tests): realistic Z medians (-1.6 to 0.1 m), more solutions than Root A
- **Root A** has extreme Z outliers (median -17 to -45 m, min -196,013 to -587,980 m) — these are the "above water" / physically impossible solutions
- The fish figures use unfiltered Root-B data. For client delivery, consider using DBSCAN-filtered positions or adding a Z sanity filter (e.g., |Z| < 50 m)
- **9F8C** has very low transmission count (182) — likely a brief detection window or low pulse rate tag
- **26E4** span is only 2.2 hours in the filtered data — most detections may have been filtered by DBSCAN or the tag was only active briefly

## Files Modified
- `scripts/tagdrag_2025_pipeline.py` — `deep_beacons_done()` fix (line 145-152)
- `scripts/_fish_figures_2025.py` — added `X, Y` to receiver SELECT (line 50)
- `output/jsats3d_2025_earlyfish.pipeline_status.json` — corrected deep_beacons status

## Files Created (throwaway — clean up after use)
- `scripts/_check_earlyfish_state.py`
- `scripts/_verify_earlyfish_data.py`
- `scripts/_fix_deep_beacons_done.py`
- `scripts/_probe_receiver_cols.py`

## Files Created (deliverables)
- `output/pm_figures_2026-10-22/example_fish/` — 20 files (4 tags × 5 each)
- `output/pm_figures_2026-10-22/example_fish/fish_qc.csv`

## Follow-up (post-timeout continuation, ~20:30-21:00)

The previous session timed out ("Timed out waiting for response headers") right after
writing this journal — all of its work was in fact on disk. This continuation verified
state, then finished the open "Z-outlier filtering for client delivery" item.

### State verified
- `deep_beacons_done()` fix confirmed in place (tagdrag_2025_pipeline.py L145-154).
- Pipeline status JSON complete; all 13 steps "ok".
- Analysis env identified: `C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d`
  (Python 3.11.15, numpy 2.4.6, pandas 3.0.5, matplotlib 3.11.1) — `python` is NOT on PATH.

### Outlier analysis (why the first-pass figures were not deliverable)
- Receiver array spans only X_t[438..541] x Y_t[37..139] (~100 m box, Z_t -12..-2 m).
- Raw root-B solution extents: 26E4 X[-710..62,510] Y[-8,726..545,273] Z[-9,751..872,424];
  9F8C modest; 1D5A X[-14,050..36,905] Z[-2,186..15,923]; B5ED X[-16,386..64,460]
  Y[-51,652..10,518] Z[-9,805..74,677].
- **Second artifact class found:** 27-51% of raw root-B rows per tag are ABOVE the water
  surface (Z > 0, up to +408 m). Z = 0 is the surface (receivers -2..-12 m; tape-measured
  deep Z 10.3-12.4 m; drag-tag recorded depths -1.5/-4.6 m). These are mirror solutions
  of the 3D solver near the array edge. The item12 chain's 1 km gate does not catch them,
  and its jump check is XY-only, so without a gate 22-48% of the Kalman track ended up
  above the surface.

### Fix: surface-gated item12 chain in scripts/_fish_figures_2025.py
Rewrote the generator (now the persistent fish-figure tool, same convention as
drag_test_figures_2025.py). Product chain:
    raw root-B solutions -> drop Z > 0 (surface gate)
      -> legacy_receiver_set_consensus (imported from ent_analysis_2025; |XYZ|<1000 m,
         10 m radius, >=2 receiver sets)
      -> smooth_track (jump 10 m / window 11 / smooth 3, gap 20 s)
      -> kalman_filter_track (2.0, 1.0, 0.5, gap 20 s)
Figures + QC use the Kalman product; client CSV now carries three products:
`solution` (raw root-B, incl. in_hull), `consensus`, `kalman` — same schema as the
drag-test client CSVs. Also fixed the first-pass jump_summary index misalignment
(zip of shifted Series vs itertuples produced "nan,nan" from-points).

### Regenerated figures — QC comparison (first pass -> now, on final Kalman product)

| Tag  | Max speed m/s | Jumps >3 m/s | Above-surface (raw -> track) |
|------|---------------|--------------|------------------------------|
| 26E4 | 3,329 -> 1.82 | 227 -> 0     | 27.4% -> 0.00%               |
| 9F8C | 30.9 -> 1.89  | 62 -> 0      | 45.9% -> 0.00%               |
| 1D5A | 539 -> 11.5*  | 1,170 -> 4   | 50.6% -> 0.00%               |
| B5ED | 53,351 -> 4.86| 993 -> 9     | 30.7% -> 0.00%               |

* 1D5A max is a sub-second t0 interval between consecutive consensus points; the 4 remaining
flags are 3-4.7 m/s bursts plus sub-second dt artifacts (0-1 m moves) — physically plausible
salmon behavior, documented in example_fish/fish_qc.csv.
Kalman tracks: 100% below surface; Z ranges 26E4 [-23.6, -0.2], 9F8C [-23.3, -1.5],
1D5A [-54.2, -0.1], B5ED [-33.1, -0.7] m; medians -3.1 / -5.8 / -2.5 / -8.1 m.

### Simplified rendering (2nd pass, ~21:15)
Per "simple is best": removed the clutter from all figure types — the 20 labeled
receiver markers, the point-to-point connecting lines (hairball on dense tracks),
and the grid. Figures are now just the track points (colored by time or depth) +
start/end triangles + colorbar; the GIF is the growing colored track + red position
dot + timestamp. Receiver context intentionally dropped (the track area overlaps the
~100 m array box, so the x-marks read as noise).

### Files
- `scripts/_fish_figures_2025.py` — REWRITTEN (persistent tool; no longer throwaway)
- `output/pm_figures_2026-10-22/example_fish/` — all 21 files regenerated 2026-10-24 21:15
- `output/pm_figures_2026-10-22/README.md` — added "Example fish" section + file list
- Cleaned up throwaways (all 8 probe scripts incl. this session's _probe_outliers.py /
  _probe_zsurf.py; 9 probe logs in output/: _exit_test, _pipe_recheck{,_err},
  _probe_exit_full, _rawfish_probe{,2,3,4}, _rawfish_checkpoint.jsonl).
  Kept: _earlyfish_build.log, _earlyfish_pipeline.log (legit run logs).

## Status
- **Pipeline:** All 13 steps pass; re-runs idempotent (deep_beacons fix verified in place)
- **Data:** 203,623 root-B 3D solutions, 190,727 DBSCAN-filtered
- **Figures:** Client-ready — surface-gated item12 chain; 100% underwater tracks;
  phantom speeds eliminated (first-pass raw-consensus figures superseded)
- **Next steps:** 26E4 fate investigation (absent from released_v0.csv; raw detections from
  06/11), ZOI01-03 Deng2D X/Y (currently configured), CFD01/ZOI11 clock curves (~940 ms late,
  no data), ZOI08 RTK survey residual, 9F8C low yield (62 Kalman pts / 17 min span — likely
  real: brief detection window, not a bug)
