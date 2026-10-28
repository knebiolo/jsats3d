# Session: Chinook Fish Pipeline Run - Complete

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
| **0FBD** | 329,632 | 6/17/2025 | ZOI02, CFD07, ZOI01, CFD06, CFD02 |
| **041D** | 123,983 | 7/22/2025 | ZOI03, ZOI01, ZOI02, CFD09, CFD08 |
| **493F** | 120,411 | 6/17/2025 | ZOI02, ZOI01, ZOI07, ZOI04, ZOI05 |
| **5F82** | 102,255 | 6/24/2025 | ZOI01, ZOI02, ZOI07, ZOI03, ZOI04 |

All 4 tags detected by ZOI receivers (needed for 3D positioning).

---

## Part 2: Pipeline Processing Results

### Step 1: Parse Raw Data
- 25.4M raw detections parsed
- 20 receivers, 43 tags (4 study tags + beacons)
- Time window: 2025-06-17 to 2025-07-31

### Step 2-12: Full Pipeline
| Step | Status | Notes |
|------|--------|-------|
| Metronome | ok | 1.3M metronome detections |
| Clock fix | ok | 12 surface receivers fixed |
| Deep beacons | ok | 3 deep beacon tags processed |
| Deep positions | ok | ZOI01/02/03 using configured positions |
| Fish multipath | ok | All 4 tags processed |
| Positions (Deng) | ok | 42.2 min for all 4 tags |
| Export | ok | 795,620 rows, 52,049 solutions |
| DBSCAN | ok | Quality filtering applied |

### Deng Solution Counts
| Tag | Solution A | Solution B | Total |
|-----|-----------|-----------|-------|
| 0FBD | 5,171 | 10,151 | 15,322 |
| 041D | 8,536 | 11,341 | 19,877 |
| 493F | 5,978 | 10,028 | 16,006 |
| 5F82 | 215 | 629 | 844 |

---

## Part 3: Output Files

Generated for EACH fish:
- `output/chinook_2025/<tag>/2D_plot.png` — Simple XY trajectory with start/end markers
- `output/chinook_2025/<tag>/3D_plot.png` — Simple XYZ trajectory with start/end markers
- `output/chinook_2025/<tag>/xyz_data.xlsx` — X, Y, Z, time, solution columns

**FIXED:** Original plots used UTM coordinates from `tblPositions_Deng` (millions of meters).
Corrected to use `tblPositions_DengDBSCAN` with local coordinate filtering (X: 200-800, Y: -200 to 400, Z: -300 to 300).

### File Sizes
| Tag | 2D Plot | 3D Plot | XYZ Excel |
|-----|---------|---------|-----------|
| 0FBD | 32 KB | 204 KB | 7.9 MB |
| 041D | 55 KB | 197 KB | 5.2 MB |
| 493F | 68 KB | 188 KB | 4.3 MB |
| 5F82 | 34 KB | 209 KB | 2.3 MB |

---

## Part 4: Bugs Fixed

1. **pandas attribute access bug** (jsats3d.py line 1424):
   - Changed `tDat.seconds` to `tDat['seconds']` (and `seconds_fix`)
   - Newer pandas versions don't allow column access as attributes

2. **Missing pulse rates** (tblTag):
   - 0FBD, 041D, 5F82 had no pulse rate (None)
   - Added pulse rate 3.155 Hz (V16 default) to enable transmission assignment

3. **Database connection bug** (chinook_analysis_2025.py):
   - Fixed connection being closed before second query

4. **Wrong table + UTM coordinates** (chinook_analysis_2025.py):
   - Original read from `tblPositions_Deng` which has UTM coordinates (millions of meters)
   - Fixed to read from `tblPositions_DengDBSCAN` with local coordinate filtering
   - Valid local range: X: 200-800, Y: -200 to 400, Z: -300 to 300

---

## Part 5: Notes

- Coordinates are in UTM easting/northing (meters) from Deng solver
- 5F82 has fewer solutions (844) due to fewer detections
- All plots are simple: one line, green start dot, red end dot, light grid
- XYZ Excel files contain: X_m, Y_m, Z_m, Time_s, ToA_s, Solution, Transmission
