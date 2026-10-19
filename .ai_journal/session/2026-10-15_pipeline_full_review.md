# 2026-10-15 Pipeline Full Review

## Overview
Comprehensive end-to-end review of the jsats3d pipeline: entry points, configs, file paths, outputs, legacy compatibility.

## Entry Points & Configs — ALL OK

| Entry Point | Config | Dry-Run | Status |
|---|---|---|---|
| `run_data.py` | `run_data.toml` (2025 ATS) | Pass | OK |
| `run_data.py` | `run_data_2019.toml` (Teknologic) | Pass | OK |
| `tagdrag_2025_pipeline.py` | `run_data_tagdrag_ent.toml` | Pass | OK |
| `legacy_pipeline.py` | any run file | N/A | OK |

All scripts import cleanly. All dry-runs pass.

## Issues Found & Fixed

### FIXED — GPS file path inconsistency

Three GPS file paths were used across the pipeline:

| Function | Before | After |
|---|---|---|
| `plot_drag_run()` (step 12) | `K:/.../array_testing_drag_GPS.csv` | Same (K: drive) |
| `plot_drag_2d_run()` (step 12c) | `data/drag_GPS_2025.csv` | **K: drive, fallback to local** |
| `plot_drag_dbscan_run()` (step 12b) | `K:/.../array_testing_drag_GPS.csv` | Same (K: drive) |

`plot_drag_2d_run` now uses the same K: drive source as the other two, with a fallback to `data/drag_GPS_2025.csv` if K: is unavailable.

**File**: `scripts/tagdrag_2025_pipeline.py` — `plot_drag_2d_run()` function

### FIXED — `speed_of_sound_done()` always returned False

The step always re-ran on every pipeline execution. Now checks for `tblInterpolatedTemp` rows (which always exist after build), so it correctly skips on reruns.

**File**: `scripts/tagdrag_2025_pipeline.py` — `speed_of_sound_done()` function

## Noted — Not Fixed

### K: drive dependency (risk, not bug)

All 2025 configs point to K: drive paths. Pipeline reads from K: (raw data is read-only — good). If K: is not mounted, the build step fails immediately. The 2019 config uses local `data/` paths — no K: dependency.

### `ent_analysis_2025.py` default GPS path

The module-level `GPS_FILE` constant still points to K: drive. This is only used as a default when no `--gps` argument is passed. All pipeline callers pass `--gps` explicitly, so this is harmless.

## All Software Outputs

### SQLite Database Tables (per run)
- `tblStudyParameters` — UTC offset, BM elevation, master receiver, sync window
- `tblTag` — tag codes, pulse rates, types (study/beacon)
- `tblReceiver` — X/Y/Z positions, beacon tag IDs
- `tblReceiver_initial` — config snapshot (before deep position adoption)
- `tblReceiverGPS` — dynamic GPS positions (CFD receivers)
- `tblWSEL` — water surface elevation
- `tblTemp` — temperature readings
- `tblInterpolatedTemp` — interpolated temperature series
- `tblDetectionRaw` — raw detections from all receivers
- `tblMetronomeUnfiltered` — master beacon unfiltered detections
- `tblMetronomeFiltered` — master beacon after primary ranking
- `tblMetronomeSecondFiltered` — master beacon after secondary classifier
- `tblDetectionClockFixed` — clock-corrected detections
- `tblDetectionFilterPrimary` — study tag primary ranking
- `tblDetectionFilterSecondary` — study tag after multipath classifier
- `tblPositions_Deng` — 3D Deng solutions (root A + root B)
- `tblPositions_Deng2D` — 2D Deng solutions (fish Z fixed)
- `tblPositions_DengDBSCAN` — DBSCAN-screened Deng solutions
- `tblDeepReceiver_step6_solution` — deep receiver solved positions
- `tblProcessProgress` — resume tracking

### Scratch Folders (in `output/<db>_legacy/`)
- `clock_fix_scratch/` — per-receiver clock fix CSVs
- `metronome_classifier/` — metronome classification CSVs
- `multipath_primary/` — primary ranking CSVs per tag
- `multipath_secondary/` — secondary classification CSVs per tag
- `deep_receivers/` — deep receiver Deng solution CSVs
- `positions/` — fish tag Deng solution CSVs (deleted by export)
- `positions_2d/` — fish tag Deng2D solution CSVs (deleted by export)
- `positions_export/` — staging for position export
- `figures/` — Deng diagnostic plots per receiver

### Review Plots (in `output/2025_review/<db>/`)
- `ENT-01_positions.png` through `ENT-05_positions.png` — drag trajectory plots
- `item12_pipeline3d_3D_legacy_A_ENT-XX_FC36.png` — GPS truth vs Deng XYZ
- `item12_pipeline3d_3D_legacy_A_ENT-XX_FC36_full_range.png` — full range version
- (same pattern for ENT-01/03/04/05 × FC36/FFD3)

### DBSCAN Output Folders (in `output/`)
- `dbscan_<db>_<beacon>_anchor<anchor>/` — pairwise DBSCAN diagnostic HTML + CSVs

### Other Outputs
- `<db>.pipeline_status.json` — step completion status
- `ENT_tag_positions.xlsx` — Excel export of consensus smoothed positions

## Legacy 2019 Compatibility — OK

The 2019 pipeline uses a separate code path:
- `data_format = 'teknologic'` triggers `import_2019()` in `legacy_pipeline.py`
- Reads `tblTag.csv`, `tblReceiver_Initial.csv`, `tblWSEL.csv`, `tblTemp.csv` from local `data/2019/`
- Calls `jsats3d.acoustic_data_import(rec, "Teknologic", ...)` — the Teknologic importer
- Same workflow: metronome → clock fix → deep receivers → study tags → Deng
- No K: drive dependency
- Dry-run passes

**The shared core (`jsats3d.py`) handles both formats** — the compatibility fixes are in `jsats3d.py` (GPS fallback, piecewise clock fix). Both 2019 and 2025 use the same positioning functions.

## Receiver Settings — OK

- `surface_receivers` — must include `master_receiver` (ZOI08)
- `deep_receivers` — positioned from their own beacons (ZOI01/02/03)
- `receivers_3d` — Deng 3D positioning (ZOI01-ZOI10, no CFD)
- `receivers_2d` — Deng2D positioning (ZOI01-ZOI03 + CFD01/05/09)
- `fixed_z_2d_by_tag` — FC36=-1.524m, FFD3=-4.572m
- Validation: `receiver_sets()` checks all receivers_3d/2d are in surface+deep

## Current Database State

`output/jsats3d_2025_tagdrag_ent_anchorZOI03.db`:
- `tblPositions_Deng`: 28,526 solutions found
- `tblPositions_Deng2D`: 65,623 solutions found
- `tblProcessProgress`: 2 entries (`__ats_phase2__`, `__ats_fish_two_stage__`)
- Pipeline is in a good state — phases 1-2 and fish multipath are marked complete

## Files Modified
- `scripts/tagdrag_2025_pipeline.py` — GPS path fix in `plot_drag_2d_run()`, speed_of_sound fix in `speed_of_sound_done()`

## Next Steps
- No blocking issues remain
- Pipeline is ready for production runs
- K: drive dependency is a known operational constraint
