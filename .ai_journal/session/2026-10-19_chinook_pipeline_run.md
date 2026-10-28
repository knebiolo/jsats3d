# Session: Chinook Fish Pipeline Run

**Date:** 2026-10-19
**Goal:** Run 4 Chinook tags through full 2D/3D positioning pipeline, generate simple plots and XYZ exports

---

## Part 1: Data Verification Summary

Verified all K drive data sources used by the pipeline:

### All 8 config paths exist and are valid
- Raw data: 25.4 GB, 487 files, 25 date folders (Jun 10 – Sep 17)
- Config XLSX, GPS CSV, covariate CSV, temperature CSV, HOBO dir, released file, drag GPS

### Tag code mapping confirmed
- Raw ATS files: `G72C474B6` → `C474` (strips G72 prefix, takes 4 hex chars)
- Pipeline filters by 4-hex codes from config

### GPS data: CFD only (expected)
- `master_df_gps.csv` has only 8 CFD receivers
- ZOI receivers use config XLSX positions (expected behavior)

### Chinook tag selection (from 538 Chinook with acoustic codes)
Selected 4 tags with best ZOI/CFD receiver coverage:

| Tag | Total Detections | Release Date | Key Receivers |
|-----|-----------------|--------------|---------------|
| **0FBD** | 297,994 | 6/17/2025 | ZOI02, CFD07, ZOI01, CFD06, CFD02 |
| **041D** | 126,951 | 7/22/2025 | ZOI03, ZOI01, ZOI02, CFD09, CFD08 |
| **493F** | 101,934 | 6/17/2025 | ZOI02, ZOI01, ZOI07, ZOI04, ZOI05 |
| **5F82** | 95,097 | 6/24/2025 | ZOI01, ZOI02, ZOI07, ZOI03, ZOI04 |

All 4 tags detected by ZOI receivers (needed for 3D positioning).

---

## Part 2: New Files Created

### `config/run_data_chinook_2025.toml`
- Tags: 0FBD, 041D, 493F, 5F82
- Window: 2025-06-17 to 2025-09-17 (full deployment)
- Output DB: `output/jsats3d_2025_chinook.db`
- No fixed depths (unknown fish depth)
- DBSCAN enabled for quality filtering

### `scripts/chinook_analysis_2025.py`
Simple analysis script that generates per-fish:
- `output/chinook_2025/<tag>/2D_plot.png` — XY trajectory with start/end markers
- `output/chinook_2025/<tag>/3D_plot.png` — XYZ trajectory with start/end markers
- `output/chinook_2025/<tag>/xyz_data.xlsx` — X, Y, Z, time, solution columns

Plots are minimal: one line, green start dot, red end dot, light grid, equal aspect.

---

## Part 3: Run Plan

1. **Step 1:** `python scripts/run_data.py config/run_data_chinook_2025.toml` — parse raw → SQLite
2. **Pipeline:** `python scripts/tagdrag_2025_pipeline.py config/run_data_chinook_2025.toml` — metronome, clock fix, deep receivers, Deng positions
3. **Analysis:** `python scripts/chinook_analysis_2025.py --db output/jsats3d_2025_chinook.db` — plots + XYZ export
