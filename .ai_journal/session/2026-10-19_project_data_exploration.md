# Project Data Exploration — K Drive Inventory

**Date:** 2026-10-19
**Goal:** Read-only inventory of all project data on K drive

---

## K Drive Structure

```
K:\Jobs\5662\001\
├── Data/
│   ├── DataTrans/
│   │   └── 2025_Data/
│   │       ├── raw_data/                    ← Raw receiver & Lotek files (25.4 GB, 487 files)
│   │       ├── downloaded_data/             ← Downloaded receiver data by date (486 files)
│   │       ├── CowlitzAT2025_Data_Deliverables/  ← Structured deliverables
│   │       ├── Tag Drag Period/             ← Tag drag WSE and depth data
│   │       ├── Temperature/                 ← Temperature string data
│   │       ├── Master Covariate Table/      ← Environmental covariates
│   │       └── 2025_output_results/         ← Previous pipeline outputs (0.57 GB, 28 files)
│   ├── FieldData/                           ← Field data (currently empty)
│   └── From_Drew_OUTDATED/                  ← QC folders (QC1, QC2, QC3)
├── Analysis/
├── Background/
├── Construction/
├── Documents/
├── Drawings/
├── PM/
├── Proposal/
├── Safety/
└── -Shared/
```

---

## 1. Raw Data (25.4 GB, 487 files)

### Receiver Raw Files (`raw_data/*/Receivers*/`)
- **Format:** ATS File Format 2.0 (`.csv`)
- **Dates:** 20250610 through 20250917 (24 date folders)
- **Types:** 
  - `20250610_Receivers_array_testing/` — ZOI/CFD receivers
  - `20250610_Lotek_array_testing/` — Lotek receivers (UPS/CHN/DNS)
- **File size:** 7–28 MB each
- **Columns:** Internal, SiteName, DateTime, TagCode, Tilt, VBatt, Temp, Pressure, SigStr, Bit Period, Threshold, etc.
- **Example:** `SR18081_20250610.csv` (ZOI05, serial 18081, ATS SR3017 firmware v10.49)

### Lotek Raw Files (`raw_data/*/Lotek*/`)
- **Format:** Lotek binary-style CSV
- **Columns:** Time, Fraction, Decimal ID, Hexadecimal ID, Power
- **Example:** `20250610_UPS_05.csv` (157 MB)

---

## 2. Data Deliverables

### 1_array_metadata/
- `cowlitz_2025_AT_config.xlsx` — Receiver/tag configuration
- `cowlitz_2025_deployed_array.kmz` — Array layout (KMZ)

### 2_AT_detection_datasets/
- `master_df_beacon.csv` — 2.17 GB, beacon tag detections
- `master_df_study.csv` — 3.62 GB, study tag detections
- `master_df_test.csv` — 3.67 MB, test tag detections
- `master_df_unknown.csv` — 4.01 GB, unknown tag detections

### 3_PTAGIS_datasets/
- `collected_v0.csv` — 2.09 MB, antenna-detected (collected) fish
- `released_v0.csv` — 0.50 MB, released fish with tag codes

### 4_gps_datasets/
- `master_df_gps.csv` — 84.9 MB, receiver GPS time series

### 5_array_testing/
- `array_testing_drag_GPS.csv` — 216 KB, drag GPS track data (ENT-01 through ENT-05)
- `cowlitz_AT_2025_testing_sheets.xlsx` — 3.28 MB, testing sheets

### 6_quality_control/
- `data_processing_tracker_cowlitz_2025.xlsx` — Processing tracker
- `receiver_data_gaps.xlsx` — Receiver data gap report
- `release_QC_tracker_cowlitz_2025.xlsx` — Release QC tracker
- `beacon_tag_pings_per_hour.html_*` — 7 HTML reports (CFD, CHN, DNS, UPS, ZOI x2)
- `data_gaps_plot_*.html` — 5 HTML gap plots (CFD, CHN, DNS, UPS, ZOI)

---

## 3. Temperature Data

### `Temperature/2025_Temp_String_Data_5min_interpolated.csv` (7.2 MB)
- **Interval:** 5 minutes
- **Strings:** BB_LCPUD (4 depths), BB_TPU (4), DD_N (4), DD_C (1), DD_S (5), FBS (5), PDS (4)
- **Variables:** Temperature at each depth + stratification indices

---

## 4. Master Covariate Table (24.7 MB)

### `2025 Master Covariate Table_20251212.csv`
- **Interval:** 5–15 minutes
- **Categories:**
  - **Hydrology (5 min):** Lake level, tailwater, weir positions, flows (spillway, discharge, diffuser, sluice), unit flows
  - **Turbidity (5 min):** Forebay, Cispus, Upper Cowlitz, Mid-Forebay
  - **Weather (15 min):** Wind dir/speed, rain accumulation, temperature, humidity, dew point, UV, solar radiation
  - **USGS Gages (15 min):** Cowlitz River near Kosmos, Cowlitz at Randle
  - **Water Quality (15 min):** DO, salinity-adjusted DO
  - **Temperature Strings (15 min):** Surface temp + stratification for all strings
  - **Binary:** LCPUD is spilling, occlusion plate presence

### `2025 Master Covariate Table_Definitions.csv`
- 105 variable definitions with units and logging intervals

---

## 5. Tag Drag Period Data

### `Tag Drag Period/TagDrag_WSE.xlsx` (93 KB)
- Water surface elevation during tag drag

### `Tag Drag Period/DD_N/` and `DD_S/`
- Depth data at various depths (0.5m, 1.5m, 3m, 4.5m, 6m, 7.5m, 9m, 12m, 15m, 18m, 21m)
- 21 CSV files, ~342 KB each
- Date: 20250710

---

## 6. Downloaded Receiver Data

- **Dates:** 20250610 through 20250917 (17 date folders)
- **Format:** CSV files per receiver serial number
- **Total:** 486 files (484 CSV, 2 Python, 1 TXT)

---

## 7. Previous Pipeline Outputs

### `2025_output_results/` (0.57 GB, 28 files)
- `jsats3d_2025_tagdrag_ent_legacyaligned.db` — Previous pipeline database
- `archive/` — Archived pipeline runs:
  - `jsats3d_2025_tagdrag_20250610.db` — Old database
  - `jsats3d_2025_tagdrag_20250610.run.json` — Run configuration
  - `dbscan_2025_tagdrag_20250610_ZOI08_anchorZOI0[1-3]/` — DBSCAN output folders (multiple files each)
  - `jsats3d_2025_tagdrag_20250610_legacy/deep_receivers/` — Deep receiver folders

### `downloaded_data/` (14.8 GB, 401 files)
- 15 date folders (Jun 10 – Sep 17)
- `archive/` subfolders in Aug 20 and Aug 28
- Empty dirs: Sep 04, 11, 17
- `.py` scripts: `inputs_prefilter_ats.py`, `inputs_prefilter_lotek.py`
- `.txt` cleaning logs in Jul 02

### Previous DB structure (legacyaligned):
| Table | Rows | Key Columns |
|-------|------|-------------|
| tblDetectionRaw | 147,361 | timeStamp, seconds, Rec_ID, Tag_ID |
| tblTag | 38 | Tag_ID, TagType, pulseRate |
| tblInterpolatedTemp | 30,817 | timeStamp, C, TempSource, DepthCount |
| tblWSEL | 29,924 | timeStamp, WSEL |
| tblReceiver | 20 | Rec_ID, X_t, Y_t, Z_t |
| tblReceiverGPS | 1,114,887 | Rec_ID, dateTime, X, Y |
| tblMetronomeFiltered | 4,809 | Rec_ID, Tag_ID, seconds, transNo |
| tblMetronomeSecondFiltered | 3,375 | + multipath_prediction, dbscan_class |
| tblDetectionClockFixed | 112,713 | + seconds_fix, seconds_residual |
| tblDetectionFilterPrimary | 51,641 | + det_rank |
| tblDetectionFilterSecondary | 51,641 | + multipath_prediction |
| tblPositions_Deng | 247,036 | X, Y, Z, T01, ToA, r0-r3 |
| tblPositions_Deng2D | 1,226,366 | X, Y, Z, T01, ToA |
| tblDeepReceiver_step6_solution | 3 | X, Y, Z |

---

## 8. Current Working Database

### `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db` (local, C:)
- Same structure as previous DB but with clock fixes applied
- 28,824 3D Deng solutions (vs 247,036 in previous — filtered to tag drag window)
- 440,752 2D Deng solutions

---

## Key Observations

1. **Massive raw data:** 25.4 GB of raw receiver files across 24 deployment dates
2. **Rich covariates:** 100+ environmental variables (flow, temperature, weather, turbidity, DO)
3. **Multiple tag types:** Beacon tags (7DB7, 7D2D, etc.), study tags (FC36, FFD3), test tags, unknown tags
4. **GPS ground truth:** Drag GPS tracks for ENT-01 through ENT-05, receiver GPS time series
5. **Temperature diversity:** 7 temp strings with 4-5 depth levels each
6. **Previous outputs:** Multiple archived databases with different configurations
7. **QC infrastructure:** HTML reports for data gaps, beacon ping rates, processing tracker
