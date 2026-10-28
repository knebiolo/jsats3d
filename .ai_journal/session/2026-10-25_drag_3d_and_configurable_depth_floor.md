# 2026-10-25 — Drag 3D views, drag Z sanity, project-configurable depth floor

## Request
1. Make sure the tag-drag Z values make sense.
2. Add a simple 3D plot of the drags (the fish already have 3D).
3. The depth floor must **not be hardcoded** — the project works in multiple areas, so it
   must be interchangeable / adjustable per project.
4. Journal + clean up any diagnostic scripts without breaking the pipeline.

## Drag Z sanity (new finding — `qc/drag_line_depths.csv`)
The drag Kalman product (the delivered track) has the **same flat-array mirror artifacts
as the fish**: above-surface and below-floor outliers, plus the known systematic Z bias
(deeper than the known tow depth). Known tow depths: FC36 −1.524 m, FFD3 −4.572 m.

Per line/tag (n, Z median vs known, range, above-surface %, below-floor %):

| line   | tag  | n   | known | median | range            | above | below |
|--------|------|-----|-------|--------|------------------|-------|-------|
| ENT-01 | FC36 | 30  | -1.52 | -5.28  | [-12.3, +15.6]   | 23.3% | 0.0%  |
| ENT-01 | FFD3 | 9   | -4.57 | -8.79  | [-17.4, -3.2]    | 0.0%  | 22.2% |
| ENT-02 | FC36 | 31  | -1.52 | -8.49  | [-11.8, +15.2]   | 25.8% | 0.0%  |
| ENT-02 | FFD3 | 27  | -4.57 | -5.20  | [-9.3, +2.6]     | 11.1% | 0.0%  |
| ENT-03 | FC36 | 59  | -1.52 | -3.18  | [-22.2, +14.2]   | 18.6% | 1.7%  |
| ENT-03 | FFD3 | 50  | -4.57 | -2.17  | [-16.0, +12.9]   | 24.0% | 2.0%  |
| ENT-04 | FC36 | 63  | -1.52 | -3.54  | [-17.5, +2.7]    | 11.1% | 1.6%  |
| ENT-04 | FFD3 | 51  | -4.57 | -1.73  | [-13.1, +7.5]    | 17.6% | 0.0%  |
| ENT-05 | FC36 | 71  | -1.52 | -4.07  | [-10.9, +12.0]   | 23.9% | 0.0%  |
| ENT-05 | FFD3 | 59  | -4.57 | -1.89  | [-13.0, +14.7]   | 35.6% | 0.0%  |
| FBY-D-1| FC36 | 72  | -1.52 | +5.86  | [-5.0, +21.5]    | 61.1% | 0.0%  |
| FBY-D-1| FFD3 | 0   | -4.57 | —      | no scored trans. | —     | —     |
| FBY-D-2| FC36 | 105 | -1.52 | -1.48  | [-15.2, +21.6]   | 38.1% | 0.0%  |
| FBY-D-2| FFD3 | 37  | -4.57 | -6.40  | [-40.9, +27.0]   | 18.9% | 27.0% |
| FBY-D-3| FC36 | 256 | -1.52 | -4.11  | [-10.3, +23.3]   | 18.0% | 0.0%  |
| FBY-D-3| FFD3 | 180 | -4.57 | -1.37  | [-8.8, +24.0]    | 28.3% | 0.0%  |
| FBY-D-4| FC36 | 98  | -1.52 | -1.22  | [-57.4, +24.3]   | 43.9% | 13.3% |
| FBY-D-4| FFD3 | 6   | -4.57 | -7.36  | [-24.7, -6.9]    | 0.0%  | 16.7% |

Reading:
- Medians are roughly at tow depth + the known Z bias (0.2–7 m deeper; best: FBY-D-2 FC36
  −1.48 vs −1.52). FBY-D-1 FC36 is the weak line (median above the water, 61% above-surface
  points) — consistent with its worst XY accuracy (median 35.8 m).
- Ranges reach +27 m / −57 m: mirror artifacts, not real. The 3D view excludes them.
- XY accuracy (the client metric) is untouched by any Z handling — the item12 scoring chain
  is XY-only and was left byte-identical (baseline gate PASS, 10/10 ENT medians diff 0.00 m).

## Changes
### `scripts/drag_test_figures_2025.py` (shared home for the floor logic)
- New helpers (single source of truth; the fish script imports them):
  - `run_file_from_status(db)` — auto-discovers the run file from
    `output/<dbstem>.pipeline_status.json` → `"run_file"`.
  - `figures_section(run_file)` — reads the `[figures]` TOML table ({} when absent).
  - `resolve_depth_floor(db, explicit, run_file, figs)` → `(floor_m, provenance)`.
    Priority: 1) CLI `--depth-floor-m`, 2) run file `[figures] depth_floor_m`,
    3) auto: deepest receiver `Z_t` − `[figures] depth_floor_margin_m` (default 4.0 m).
  - `DEFAULT_DEPTH_FLOOR_MARGIN_M = 4.0`.
- CLI: `--run-file`, `--depth-floor-m`. Prints the resolved floor + provenance at startup.
- **New per-line 3D views** `plot_line_3d()` → `<line>_3d.png` (10 new deliverables:
  ENT-01..05, FBY-D-1..4): GPS track at each tag's known tow depth (black, green start /
  red end markers) + root-B Kalman track per tag (time-colored), **gated to [floor, 0] m**
  so mirror artifacts are not drawn, + 3D receiver reference.
- **New Z QC** `line_depth_qc()` → `qc/drag_line_depths.csv` + log table (the numbers above).
- README: Notes bullet documenting the 3D views, the view-only gate, and where the floor
  comes from; file list picks up the new PNGs automatically.
- Scoring chain, baseline gate, and client CSVs **unchanged**.

### `scripts/fish_figures_2025.py`
- Hardcoded `Z_FLOOR_M = -16.0` **removed** → runtime `resolve_depth_floor()` (same
  priority; same `--run-file` / `--depth-floor-m` CLI).
- `fish_qc.csv` gains a `depth_floor_m` audit column.
- QC numbers unchanged (floor still resolves to −16.0, now from the run file):
  26E4 19897→1387 kalman, 9F8C 1654→65, 1D5A 51859→3413, B5ED 37263→2327.

### Config (per-project, the "interchangeable" part)
Added to `config/run_data_tagdrag_ent.toml` and `config/run_data_earlyfish_2025.toml`:
```toml
[figures]
depth_floor_m        = -16.0
depth_floor_margin_m = 4.0
```
Site justification in the comments: deepest measured TOTAL depth 13.3 m (43.5 ft at ZOI02),
deepest hydrophone −12.40 m → −16.0 = hydrophone + 4 m (chinook 2026-10-21 audit margin).
For a new deployment: set `depth_floor_m` from the local bathymetry, or omit it entirely and
the generators auto-derive (deepest receiver − 4 m). Verified all three priority paths:
auto → −16.40 m, run file → −16.0 m, explicit → −20.0 m. Pipeline `--list` exit 0 on the
updated configs (extra TOML section is ignored by run_data/pipeline).

## Verification
- Drag generator re-run: **BASELINE GATE PASS** (10/10 ENT medians diff 0.00 m), manifest
  46 files (36 prior + 10 `_3d.png` + `drag_line_depths.csv`).
- Fish generator re-run: identical QC to 2026-10-25 journals; 25 deliverables.
- Visual check: ENT-03_3d.png and FBY-D-2_3d.png — GPS at tow depth, both tag tracks inside
  the water column, receivers as reference; loop shape matches the 2D plan views.
- Tests: 14 passed. 2 known pre-existing failures, both confirmed unrelated to this session:
  `tests/test_2025_contracts.py` (imports `scripts/extract_dbscan_features.py`, deleted at
  HEAD 8ca7ba4) and `tests/test_resume.py::test_ats_fish_filters_keep_primary_and_secondary_separate`
  (WinError 32 temp-file lock during teardown; the failing path — tagdrag_2025_pipeline /
  legacy_pipeline — is not touched by this session's edits).

## Cleanup
No diagnostic/throwaway scripts were created this session (all work was edits to the two
persistent figure scripts + two config files). Repo root clean; `scripts/` holds the 10
persistent tools.

## Open items (owner-gated, unchanged)
- Known Z bias (fish medians ~1.6–3.7 m deeper than known tag depths; drag medians similar)
  — sound speed / ZOI08 RTK / uncorrected CFD01–ZOI11 clock class. Separate from artifacts.
- Offer still open: gate the drag client CSV `consensus` rows to [floor, 0] at export time
  (they currently carry the Z artifacts, e.g. +1975 m static-hold medians). Scoring stays XY-only.
- FBY-D-1 FFD3 has no scored transmissions (122 candidates) — pre-existing, documented in QC.
