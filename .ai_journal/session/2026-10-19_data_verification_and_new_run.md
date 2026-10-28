# Session: Data Verification + New Chinook Pipeline Run

**Date:** 2026-10-19
**Goal:** Verify all pipeline data sources, then run 4 Chinook tags for full 2D/3D positioning

---

## Part 1: K Drive Data Verification

Verified all data paths and sources used by `run_data_tagdrag_ent.toml` pipeline.

### ✅ All 8 config paths exist
| Path | File | Size |
|------|------|------|
| raw_root | `raw_data/` folder | 25.4 GB, 487 files, 25 dates |
| config_xlsx | `cowlitz_2025_AT_config.xlsx` | Config metadata |
| gps_csv | `master_df_gps.csv` | 82.9 MB |
| covariate_csv | `2025 Master Covariate Table_20251212.csv` | 24.7 MB |
| temperature_csv | `2025_Temp_String_Data_5min_interpolated.csv` | 7.2 MB |
| hobo_dir | `Tag Drag Period/DD_N/` | 10 files |
| released_file | `released_v0.csv` | 2,992 tags |
| drag_gps | `array_testing_drag_GPS.csv` | 216 KB |

### ✅ Raw receiver data — FFD3/FC36 confirmed
- **Tag code mapping:** `G72C474B6` → `C474` (strips G72 prefix, takes 4 hex chars)
- **FFD3:** ~3,275 detections on June 10 across ZOI/CHN receivers
- **FC36:** ~5,496 detections on June 10 across ZOI/CHN receivers
- **Detections start at 09:56:37** — within the 09:00–13:00 tag drag window
- Pipeline correctly filters by `tags=['FFD3', 'FC36']`

### ⚠️ GPS data — CFD only (expected)
- `master_df_gps.csv` has **only 8 CFD receivers** (CFD02–CFD09)
- All 10 ZOI receivers are **missing** from GPS data
- This is expected — CFD receivers were GPS'd, ZOI receivers use config XLSX positions
- Pipeline falls back to `cowlitz_2025_AT_config.xlsx` for ZOI receiver positions

### ✅ Drag GPS truth data
- 13 lines: ENT-01 through ENT-05 + FBY-D/NS/SN lines
- ENT-01 and ENT-05 data within the tag drag window

### ✅ Tag identification
- FFD3 and FC36 exist in `master_df_test.csv` (7,550 and 5,058 detections)
- Also in `master_df_study.csv` (15 and 17 detections)
- Pipeline reads raw receiver files → strips G72 → filters by 4-hex → correct

### ✅ Other data sources
- Covariates: 105 variables (flow, temp, weather, turbidity, DO)
- Temperature: 7 strings, 4–5 depths each, 5-min intervals
- HOBO: 10 depth files in DD_N/
- Released tags: 2,992 tags with species info

---

## Part 2: New Chinook Pipeline Run (next session)
- Identify 4 Chinook tags from released_v0.csv
- Update config for new tags
- Run pipeline for full 2D/3D positioning
- Output per fish: simple 2D plot, 3D plot, XYZ Excel file
