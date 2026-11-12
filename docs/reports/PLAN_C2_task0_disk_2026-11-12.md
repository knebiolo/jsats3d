# Plan C2: Task 0 — Setup and Disk Report
# Date: 2026-11-12

## Git state
- Repo: C:/Dev/jsat3d/jsats3d
- Branch: ENM_jsats3d_edits_clocksync
- HEAD: 2328357 Document Plan C2 rebuild and review
- Status: (clean — no uncommitted changes)

## Free disk
- C: drive: 1.23 GB free

## Largest folders under output/
| Folder | GB |
|---|---|
| 11062026_clocksync | 18.85 |
| 10292026_Test | 17.75 |
| 11082026_whatif_B1 | 17.40 |
| 11072026_diag_armB | 5.68 |
| 11112026_zoi11_cfd01 | 3.15 |
| 11092026_stage1_tdoa_review | 0.26 |

## output/11112026_zoi11_cfd01/ subfolders
| Subfolder | GB |
|---|---|
| armZC | 1.54 |
| armZ | 1.46 |
| arm_ZC_legacy | 0.13 |
| arm_Z_legacy | 0.09 |
| arm_C_legacy | 0.09 |
| armC | 0.09 |
| armN | 0.06 |
| arm_N_legacy | 0.06 |

## Files over 100 MB under output/11112026_zoi11_cfd01/
| Path | Bytes | LastWrite |
|---|---|---|
| armZ\arm_Z.db | 1,468,461,056 | 2026-11-11 20:37:10 |
| armZC\arm_ZC.db | 1,513,746,432 | 2026-11-11 20:31:40 |

## Files over 500 MB last written 2026-11-11
| Path | MB | LastWrite |
|---|---|---|
| jsats3d\output\11112026_zoi11_cfd01\armZC\arm_ZC.db | 1,443.6 | 2026-11-11 20:31:40 |
| jsats3d\output\11112026_zoi11_cfd01\armZ\arm_Z.db | 1,400.4 | 2026-11-11 20:37:10 |

## First run workaround record
The tables dropped (from `output/11112026_zoi11_cfd01/drop_tables.py`):
1. tblDetectionFilterSecondary
2. tblDetectionFilterPrimary
3. tblDetectionClockFixed
4. tblDeepReceiver_step6_solution
5. tblMetronomeFiltered
6. tblMetronomeSecondFiltered
7. tblPositions_Deng
8. tblPositions_DengDBSCAN
9. tblProcessProgress

Exact commands used (from journal):
1. `Copy-Item arm_N.db arm_Z.db` (then arm_C.db, arm_ZC.db)
2. Drop the 9 tables above via `DROP TABLE IF EXISTS`
3. `python.exe scripts/run_data.py <run_file.toml> --skip-build`

## STOP — Disk check

**Free space is 1.23 GB. The plan requires at least 8 GB before Task 1.**

Largest items consuming space:
- `output/11062026_clocksync/`: 18.85 GB
- `output/10292026_Test/`: 17.75 GB
- `output/11082026_whatif_B1/`: 17.40 GB
- `output/11072026_diag_armB/`: 5.68 GB
- `output/11112026_zoi11_cfd01/armZ/arm_Z.db`: 1.40 GB
- `output/11112026_zoi11_cfd01/armZC/arm_ZC.db`: 1.44 GB

Total of top 3 folders alone: 54.0 GB. The owner decides what to remove.
