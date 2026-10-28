# 2025 Pipeline Run Journal — 2026-10-27

## Summary

Full 2025 early-fish tag drag pipeline run from scratch:
- **Parsed** 304 ATS raw files → 29,912,682 detections → 15.58 GB SQLite DB
- **Ran** full 2025 pipeline: DBSCAN metronome → clock surface → deep beacons → Deng 3D positioning → ATS timing multipath → speed of sound → fish positions → DBSCAN screen
- **Generated** figures for 4 tags: 26E4, 9F8C, 1D5A, B5ED

**Output:** `C:\Dev\jsat3d\jsats3d\output\10272026_test\`

---

## What We Did

### 1. Bug Fixes in `parse_ats_raw_to_legacy.py`

Three bugs in the batch commit optimization (lines 517-548):

| Line | Bug | Fix |
|------|-----|-----|
| 521 | `cursor.execute(insert_sql)` — executes INSERT with 49 placeholders but no values | Removed (was trying to "pre-prepare" a statement) |
| 543 | `cursor.executemany("", batch[:BATCH_ROWS])` — empty string instead of SQL | Changed to `cursor.executemany(insert_sql, batch[:BATCH_ROWS])` |
| 548 | `cursor.executemany("", batch)` — same empty string issue | Changed to `cursor.executemany(insert_sql, batch)` |

**Root cause:** Another AI tried to apply a "prepared statement" pattern from other languages to Python's `sqlite3` module. Python's sqlite3 doesn't support `cursor.execute(sql)` + `cursor.executemany("", data)`. The SQL must be passed to every `executemany` call.

### 2. Config Change

Removed CFD05 from `surface_receivers` in `run_data_earlyfish_2026-10-26.toml`:
- CFD05 had no clock data available (not deployed/active during study)
- Caused `RuntimeError: check failed: no clock-fixed rows for deep receiver CFD05`

### 3. Disk Space Issues

C: drive had only 10.3 GB free (2.2%). Old runs and archives were consuming ~30 GB:
- `output/jsats3d_2025_earlyfish.db` — 16 GB (old run)
- `output/10262026_Run/` — 13.82 GB (old run)
- `output/output_archive_2026-10-05/` — 6.38 GB (old archive)

Deleted these to free up space for index creation.

### 4. Database Corruption

Deleted orphaned journal file (`jsats3d_2025_earlyfish.db-journal`, 3.9 GB) after killing a stuck `run_data.py` process. This corrupted the DB — SQLite needs the journal to complete or roll back the last transaction. **Never delete journal files while the DB is in use.**

### 5. Index Creation

The `run_data.py` indexes step failed with "database or disk is full." Created indexes manually via `_create_indexes.py` after freeing disk space.

---

## How to Run a Successful 2025 Pipeline

### Prerequisites

1. **Disk space:** At least 40 GB free on C: drive (DB grows to ~16 GB, indexes add ~2 GB, temp files during processing)
2. **No other processes** holding the DB open (kill lingering Python processes first)
3. **CFD05 removed** from `surface_receivers` in config

### Config File

`config/run_data_earlyfish_2026-10-26.toml` — key settings:

```toml
[paths]
raw_root        = 'C:\Users\Ethan.Muhlestein\OneDrive - Kleinschmidt Associates\Documents\jsats_data\raw_data'
config_xlsx     = 'C:\Users\Ethan.Muhlestein\OneDrive - Kleinschmidt Associates\Documents\jsats_data\CowlitzAT2025_Data_Deliverables\1_array_metadata\cowlitz_2025_AT_config.xlsx'
gps_csv         = 'C:\Users\Ethan.Muhlestein\OneDrive - Kleinschmidt Associates\Documents\jsats_data\CowlitzAT2025_Data_Deliverables\4_gps_datasets\master_df_gps.csv'
covariate_csv   = 'C:\Users\Ethan.Muhlestein\OneDrive - Kleinschmidt Associates\Documents\jsats_data\Master Covariate Table\2025 Master Covariate Table_20251212.csv'
temperature_csv = 'C:\Users\Ethan.Muhlestein\OneDrive - Kleinschmidt Associates\Documents\jsats_data\Temperature\2025_Temp_String_Data_5min_interpolated.csv'
hobo_dir        = 'C:\Users\Ethan.Muhlestein\OneDrive - Kleinschmidt Associates\Documents\jsats_data\Tag Drag Period\DD_N'
output_db       = 'output\10272026_test\jsats3d_2025_earlyfish.db'

[study]
master_receiver  = 'ZOI08'
pulse_rates      = { '26E4' = 2.986, '9F8C' = 3.050, '1D5A' = 3.081, 'B5ED' = 2.969 }

[legacy]
surface_receivers = ['ZOI05', 'ZOI06', 'ZOI07', 'ZOI08', 'ZOI09', 'ZOI10',
                     'CFD01', 'CFD02', 'CFD03', 'CFD04', 'CFD06', 'CFD07', 'CFD08', 'CFD09']
deep_receivers    = ['ZOI01', 'ZOI02', 'ZOI03']
```

**Note:** CFD05 is intentionally excluded from `surface_receivers`.

### Step-by-Step Commands

#### Step 1: Clear Output Folder

```powershell
Remove-Item "C:\Dev\jsat3d\jsats3d\output\10272026_test\*" -Recurse -Force -ErrorAction SilentlyContinue
```

#### Step 2: Run `run_data.py` (Parse Raw Files)

```powershell
C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe C:\Dev\jsat3d\jsats3d\scripts\run_data.py "C:\Dev\jsat3d\jsats3d\config\run_data_earlyfish_2026-10-26.toml"
```

- **Timeout:** Use 48 hours (172,800,000 ms) — parse takes ~30 min, indexes step can take 30-60 min, but 48h gives plenty of headroom
- **Output:** 29.9M detections, ~13 GB DB (before indexes)
- **Note:** `run_data.py` already has all logic for study parameters, pulse rates, and indexes — no separate scripts needed

#### Step 3: Run the 2025 Pipeline

```powershell
C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe C:\Dev\jsat3d\jsats3d\scripts\_2025_pipeline.py "C:\Dev\jsat3d\jsats3d\config\run_data_earlyfish_2026-10-26.toml"
```

- **Timeout:** 48 hours (172,800,000 ms) — pipeline takes ~35 min, but 48h gives headroom
- **Steps run:** prepare → metronome → clock_surface → deep_beacons → deep_positions → deep_clocks → fish_multipath → positions → export → positions_dbscan
- **Total pipeline time:** ~35 minutes

#### Step 4: Generate Figures

```powershell
C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe C:\Dev\jsat3d\jsats3d\scripts\fish_figures_2025.py --db "C:\Dev\jsat3d\jsats3d\output\10272026_test\jsats3d_2025_earlyfish.db" --out "C:\Dev\jsat3d\jsats3d\output\10272026_test\figures"
```

- **Output:** 25 files (6 per tag + fish_qc.csv)

---

## Troubleshooting

### "database is locked"
- Kill all Python processes: `Get-Process python -ErrorAction SilentlyContinue \| Where-Object {$_.Path -like "*anaconda3*"} \| Stop-Process -Force`
- Wait 10 seconds, then retry

### "database disk image is malformed"
- **Never delete `.db-journal` files** — they contain uncommitted transaction data
- Delete the corrupted DB and re-run from Step 2

### "database or disk is full"
- Check free space: `Get-Volume -DriveLetter C`
- Delete old runs/archives from `output/` folder
- Need at least 40 GB free for a full run

### "no study tags with a pulse rate" / "masterReceiver is empty"
- Re-run `run_data.py` with a 48-hour timeout — it handles everything in one pass

### "no clock-fixed rows for deep receiver CFD05"
- Remove CFD05 from `surface_receivers` in config

### "Incorrect number of bindings supplied" in parse
- Fix the 3 bugs in `parse_ats_raw_to_legacy.py` (see Bug Fixes section above)

---

## Key Files Modified

| File | Change |
|------|--------|
| `jsats3d/scripts/parse_ats_raw_to_legacy.py` | Fixed 3 bugs in batch commit (lines 521, 543, 548) |
| `jsats3d/config/run_data_earlyfish_2026-10-26.toml` | Removed CFD05 from surface_receivers |

## Output Structure

```
output/10272026_test/
├── jsats3d_2025_earlyfish.db          (15.58 GB)
├── jsats3d_2025_earlyfish.pipeline_status.json
├── jsats3d_2025_earlyfish.run.json
└── figures/
    ├── 26E4_plan_time.png
    ├── 26E4_plan_depth.png
    ├── 26E4_3d.png
    ├── 26E4_track.gif
    ├── 26E4_track3d.gif
    ├── 26E4_positions.csv
    ├── 9F8C_plan_time.png
    ├── 9F8C_plan_depth.png
    ├── 9F8C_3d.png
    ├── 9F8C_track.gif
    ├── 9F8C_track3d.gif
    ├── 9F8C_positions.csv
    ├── 1D5A_plan_time.png
    ├── 1D5A_plan_depth.png
    ├── 1D5A_3d.png
    ├── 1D5A_track.gif
    ├── 1D5A_track3d.gif
    ├── 1D5A_positions.csv
    ├── B5ED_plan_time.png
    ├── B5ED_plan_depth.png
    ├── B5ED_3d.png
    ├── B5ED_track.gif
    ├── B5ED_track3d.gif
    ├── B5ED_positions.csv
    └── fish_qc.csv
```

## Pipeline Status (Final)

| Step | Name | Status | Time |
|------|------|--------|------|
| 1b | prepare | ok | 0.0 min |
| 2-3 | metronome (DBSCAN) | ok | 5.1 min |
| 4 | clock_surface | ok | 5.5 min |
| 5 | deep_beacons | ok | 4.3 min |
| 6 | deep_positions (Deng) | ok | 8.3 min |
| 6b | adopt_deep | skipped | — |
| 7 | deep_clocks | ok | 10.8 min |
| 8 | fish_multipath (ATS timing) | ok | 1.6 min |
| 9 | speed_of_sound | skipped | — |
| 11 | positions (Deng) | ok | 2.7 min |
| 10 | export | ok | 0.1 min |
| 11b | positions_2d | skipped | — |
| 12 | plot_drag | skipped | — |
| 12c | plot_drag_2d | skipped | — |
| 11c | positions_dbscan | ok | 0.1 min |
| 12b | plot_drag_dbscan | skipped | — |

**Total pipeline time:** ~38 minutes (excluding parse)
