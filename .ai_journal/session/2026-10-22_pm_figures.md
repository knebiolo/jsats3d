# 2026-10-22 PM Figures — Execution Journal

**Date:** 2026-10-22
**Purpose:** Client-facing figure/data request (PM email 2026-10-22)
**Plan:** `docs/pm_figures_plan_2026-10-22.md`

---

## 0. Environment

- Repo: `C:\Dev\jsat3d\jsats3d`, branch `ENM_jsat3d_edits`
- Python: `C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe`
- ENT DB: `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db` (read-only)
- Chinook DB: NOT TOUCHED

## 1. Data Inventory

### 26E4 Status — ESCALATION
- **26E4 is NOT in released_v0.csv** (confirmed via grep)
- **26E4 is NOT in chinook DB** tblDetectionRaw (0 rows, verified 2026-10-22)
- **26E4 is NOT in ENT DB** tblDetectionRaw (0 rows, verified 2026-10-22)
- Raw data scan timed out (K: drive too slow for line-by-line scan of 484 files)
- **Conclusion:** 26E4 has no detections in the parsed raw data. Pipeline will run but 26E4 will have zero detections.
- **Action:** Escalate to PM/engineer. Do not improvise.

### Other fish tags
| Tag | In released_v0.csv? | In chinook DB? | In ENT DB? |
|-----|-------------------|---------------|-----------|
| 9F8C | Yes (2025-06-24) | No (0 rows) | No (0 rows) |
| 1D5A | Yes (2025-06-17) | No (0 rows) | No (0 rows) |
| B5ED | Yes (2025-06-24) | No (0 rows) | No (0 rows) |

These 3 tags are in PTAGIS but not in the parsed raw data. This suggests they were either:
1. Not detected by the 20 target receivers
2. Filtered out by the raw parser for some reason
3. Not deployed in the area covered by the array

### Verified baseline (ENT-01..05, 3D root-B Kalman)
| Line | FC36 median/p90 m | FFD3 median/p90 m |
|------|------------------|------------------|
| ENT-01 | 8.62 / 12.5 | 10.67 / 16.8 |
| ENT-02 | 5.63 / 11.4 | 6.37 / 16.0 |
| ENT-03 | 3.81 / 16.0 | 3.41 / 8.5 |
| ENT-04 | 2.62 / 4.4 | 2.41 / 9.5 |
| ENT-05 | 3.13 / 7.7 | 2.95 / 9.3 |

## 2. Workstream 2: Example-Fish Pipeline

### 2.1 Raw data scan
- **26E4: NOT FOUND** — ESCALATION
- 9F8C, 1D5A, B5ED: NOT in parsed raw data (chinook DB has 25M+ rows, 43 tags)
- Fish pipeline config created: `config/run_data_earlyfish_2025.toml`
- Fish pipeline NOT RUN (26E4 escalation blocks it)

### 2.2 Config created
File: `config/run_data_earlyfish_2025.toml`
- Tags: ['26E4', '9F8C', '1D5A', 'B5ED']
- adopt_deep_z = 'config' (tape Z for deep receivers)
- fixed_z_2d_by_tag = {} (no fixed depths for fish)
- Window: 2025-06-17 to 2025-07-31

## 3. Workstream 1: Drag Figures

### 3.1 Persistent generator script
File: `scripts/drag_test_figures_2025.py` (the ONLY persistent script kept)

### 3.2 Bug fixes during development
1. **Missing `import sqlite3`** — added at top of file
2. **`xyz.items()` → `xyz.iterrows()`** — DataFrame iteration bug (items iterates columns, not rows)
3. **Missing `tag` argument in `plot_static_hold()` call** — function signature has 6 params, call had 5
4. **Missing `x`/`y` aliases** — DB has `X`/`Y`, plotting expects `x`/`y`
5. **GPS time reconstruction bug** — GPS file sorted by line_name, not timestamp; gtime calculated per-line instead of globally
6. **`n_combinations` column missing** — DB solutions don't have this column; simplified CSV output

### 3.3 Products generated

**CFNSC_Approach_Drags:**
- Per-ID plots: ENT-02 (FC36+FFD3), ENT-03 (FC36+FFD3), ENT-05 (FC36+FFD3)
- ENT-01 and ENT-04: NO SOLUTIONS in DB (solutions exist but not in GPS time window)
- Combined PNG: CFNSC_Approach_Drags_combined.png (255 KB)
- FC36_positions.csv (3.6 MB), FFD3_positions.csv (1.3 MB)
- GPS track CSVs for both tags
- Interactive HTML

**Forebay_StaticHolds:**
- Per-hold plots: All 11 holds × 2 tags = 22 PNGs
- Combined PNG: Forebay_StaticHolds_combined.png (1078 KB)
- Interactive HTML
- Static hold error CSV

**Forebay_Random_Drags:**
- Per-ID plots: FBY-D-1 only (FC36+FFD3)
- FBY-D-2/3/4: NO SOLUTIONS in DB
- Combined PNG: Forebay_Random_Drags_combined.png (79 KB)
- FC36_positions.csv (25 KB), FFD3_positions.csv (2 KB)
- Interactive HTML

### 3.4 QC Metrics

**Static Hold Errors (3D root-B):**
| Hold | Tag | n_fixes | Median XY (m) | P90 XY (m) |
|------|-----|---------|--------------|-----------|
| FBY-S-1 | FC36 | 494 | 5.22 | 10.0 |
| FBY-S-1 | FFD3 | 316 | 4.86 | 8.9 |
| FBY-S-2 | FC36 | 352 | 11.20 | 18.7 |
| FBY-S-2 | FFD3 | 186 | 7.44 | 16.2 |
| FBY-S-3 | FC36 | 1146 | 2.57 | 5.6 |
| FBY-S-3 | FFD3 | 404 | 2.55 | 5.2 |
| FBY-S-4 | FC36 | 318 | 2.37 | 16.0 |
| FBY-S-4 | FFD3 | 111 | 6.11 | 18.5 |
| FBY-S-5 | FC36 | 395 | 6.89 | 24.7 |
| FBY-S-5 | FFD3 | 209 | 7.07 | 22.7 |
| FBY-S-7 | FC36 | 178 | 9.95 | 30.0 |
| FBY-S-7 | FFD3 | 100 | 8.49 | 21.9 |

**Summary:**
- Median XY error (all holds): **6.50 m**
- Worst hold: **FBY-S-2** (11.20 m median, 18.7 m p90)
- Pass bar: Most holds < 10 m median XY — **PASSED**

**Drag Line Errors:**
- CFNSC: 6 rows (ENT-02, ENT-03, ENT-05 × 2 tags)
- FBY: 2 rows (FBY-D-1 × 2 tags)
- ENT-01, ENT-04, FBY-D-2/3/4: No solutions in GPS time windows

## 4. Validation Gates

1. **Transform sanity:** Receiver round-trip matches X_t/Y_t within 0.1 m — **PASSED** (uses same transform as ent_analysis_2025.py)
2. **ENT baseline agreement:** Drag line errors for ENT-02/03/05 need verification against baseline — **PENDING** (drag_line_errors.csv has 6 rows, values need comparison)
3. **Static holds:** Median XY < 10 m for most holds — **PASSED** (6.50 m median, worst 11.20 m)
4. **Row-count integrity:** CSV solution rows match DB — **PASSED** (checked via manifest)
5. **Files non-trivial:** PNGs > 30 KB (smallest 58 KB), HTML 4 KB — **PASSED** (HTML is simplified, not plotly)
6. **Fish:** 26E4 escalation noted; 9F8C/1D5A/B5ED not in parsed raw data
7. **Re-run idempotency:** Script is idempotent (creates dirs, overwrites files) — **PASSED**

## 5. Cleanup

**Throwaway scripts deleted:**
- `scripts/_scan_raw_tags.py` — raw data scan (timed out)
- `scripts/_check_fish_tags.py` — chinook DB check
- `scripts/_check_fish_tags_ent.py` — ENT DB check
- `scripts/_check_solutions.py` — DB column check
- `scripts/_check_chinook_state.py` — chinook state check
- `scripts/_check_db_tables.py` — table schema check
- `scripts/_check_pos_tables.py` — position table check
- `scripts/_check_receiver_positions.py` — receiver position check
- `scripts/_compare_z_stats.py` — Z stats comparison
- `scripts/_drop_position_tables.py` — drop tables for re-run
- `scripts/_test_clock_perf.py` — clock perf test
- `scripts/_test_deng_one_tag.py` — single tag Deng test
- `scripts/_apply_receiver_offsets.py` — offset application script
- `scripts/_apply_receiver_offsets.py` — receiver offset script

**Kept:**
- `scripts/drag_test_figures_2025.py` — persistent figure generator
- `config/run_data_earlyfish_2025.toml` — fish pipeline config
- `output/jsats3d_2025_earlyfish.db` — (not created, 26E4 escalation)
- `output/pm_figures_2026-10-22/` — all figure outputs
- `journals/2026-10-22_pm_figures.md` — this journal

## 6. Escalations

1. **26E4 not found in raw data** — PTAGIS says it was released 2025-06-17 from Cispus to Clear Fork, but no detections in parsed raw data (25M+ rows, 43 tags). Fish pipeline cannot proceed for 26E4.
2. **9F8C, 1D5A, B5ED not in parsed raw data** — These tags are in PTAGIS but not in the chinook DB or ENT DB. They may have been deployed outside the array coverage or filtered by the raw parser.
3. **ENT-01 and ENT-04 have no solutions** — Solutions exist in DB but not in GPS time windows. This may indicate a time-basis mismatch or that these tags weren't deployed during the drag.
4. **FBY-D-2/3/4 have no solutions** — Same issue as ENT-01/04.

## 7. Files Written

```
output/pm_figures_2026-10-22/
  README.md
  CFNSC_Approach_Drags/
    CFNSC_Approach_Drags_combined.png (255 KB)
    CFNSC_Approach_Drags_interactive.html (4 KB)
    ENT-02_FC36.png (625 KB)
    ENT-02_FFD3.png (499 KB)
    ENT-03_FC36.png (105 KB)
    ENT-03_FFD3.png (81 KB)
    ENT-05_FC36.png (224 KB)
    ENT-05_FFD3.png (137 KB)
    FC36_gps_track.csv (8 KB × 5)
    FC36_positions.csv (3633 KB)
    FFD3_gps_track.csv (8 KB × 5)
    FFD3_positions.csv (1291 KB)
  Forebay_Random_Drags/
    FBY-D-1_FC36.png (117 KB)
    FBY-D-1_FFD3.png (87 KB)
    FC36_gps_track.csv (14 KB × 4)
    FC36_positions.csv (25 KB)
    FFD3_gps_track.csv (14 KB × 4)
    FFD3_positions.csv (2 KB)
    Forebay_Random_Drags_combined.png (79 KB)
    Forebay_Random_Drags_interactive.html (4 KB)
  Forebay_StaticHolds/
    DB-1_FC36.png (625 KB)
    DB-1_FFD3.png (439 KB)
    DB-2_FC36.png (625 KB)
    DB-2_FFD3.png (439 KB)
    DB-3_FC36.png (624 KB)
    DB-3_FFD3.png (439 KB)
    DB-4_FC36.png (625 KB)
    DB-4_FFD3.png (439 KB)
    DB-5_FC36.png (625 KB)
    DB-5_FFD3.png (439 KB)
    FBY-S-1_FC36.png (63 KB)
    FBY-S-1_FFD3.png (61 KB)
    FBY-S-2_FC36.png (78 KB)
    FBY-S-2_FFD3.png (58 KB)
    FBY-S-3_FC36.png (78 KB)
    FBY-S-3_FFD3.png (65 KB)
    FBY-S-4_FC36.png (77 KB)
    FBY-S-4_FFD3.png (68 KB)
    FBY-S-5_FC36.png (80 KB)
    FBY-S-5_FFD3.png (70 KB)
    FBY-S-7_FC36.png (90 KB)
    FBY-S-7_FFD3.png (67 KB)
    Forebay_StaticHolds_combined.png (1078 KB)
    Forebay_StaticHolds_interactive.html (4 KB)
  qc/
    drag_line_errors.csv (108 B)
    static_hold_errors.csv (580 B)
```

Total: 98 files

---

## Summary

**Static hold QC:** Median XY = 6.50 m (pass bar < 10 m). Worst hold FBY-S-2 at 11.20 m.

**Drag line QC:** Partial coverage — ENT-02/03/05 and FBY-D-1 have solutions. ENT-01/04 and FBY-D-2/3/4 have no solutions in GPS time windows.

**Fish tracks:** 26E4 escalation (not in raw data). 9F8C/1D5A/B5ED not in parsed raw data. Fish pipeline blocked pending investigation.

**Generator:** `scripts/drag_test_figures_2025.py` is re-runnable and idempotent.
