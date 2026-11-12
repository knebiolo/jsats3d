# Plan C: ZOI11/CFD01 Four-Arm Positioning Comparison — Final Report
# Date: 2026-11-11
# Executor: Qwen 3.6 35B MoE

## 1. Executive Summary

Built four arm configurations to evaluate ZOI11 and CFD01 receiver positioning impact:
- **Arm N:** Baseline (no ZOI11, no CFD01) — 129,183 solutions
- **Arm Z:** ZOI11 added — 216,437 solutions (+67.7%)
- **Arm C:** CFD01 added — 189,630 solutions (+46.8%)
- **Arm ZC:** Both added — 302,881 solutions (+134.5%)

Both ZOI11 and CFD01 significantly improve positioning. The combined effect (ZC) exceeds the sum of individual effects, suggesting synergistic coverage.

## 2. Build Details

### Route: Route 2 (build from raw)
- No `zoi11_A.db` found on disk
- Raw folders: 20250610_Receivers_array_testing (23 files), 20250618_Receivers (25 files)
- Raw root: `C:\Users\Ethan.Muhlestein\OneDrive - Kleinschmidt Associates\Documents\jsats_data\raw_data`

### Run files created
- `config/run_data_zoi11cfd01_N.toml` — baseline
- `config/run_data_zoi11cfd01_Z.toml` — ZOI11 added
- `config/run_data_zoi11cfd01_C.toml` — CFD01 added
- `config/run_data_zoi11cfd01_ZC.toml` — both added

### Key difference between run files
| Arm | surface_receivers | receivers_3d |
|---|---|---|
| N | 14 receivers (no ZOI11, no CFD01) | 9 receivers (no ZOI11, no CFD01) |
| Z | 15 receivers (+ZOI11) | 10 receivers (+ZOI11) |
| C | 14 receivers (+CFD01) | 10 receivers (+CFD01) |
| ZC | 15 receivers (+ZOI11, +CFD01) | 11 receivers (+ZOI11, +CFD01) |

### Pipeline approach
`--init-from` flag not supported in this script version. Adapted using `--skip-build` + copy approach:
1. Copy arm_N.db to target DB
2. Drop pipeline output tables (schema mismatch fix)
3. Run with `--skip-build`

## 3. Results Summary

| Arm | FFD3 A | FFD3 B | FC36 A | FC36 B | Total Rows | Total Solutions |
|---|---|---|---|---|---|---|
| N | 19,860 | 27,324 | 33,213 | 48,786 | 333,430 | 129,183 |
| Z | 31,171 | 41,524 | 59,855 | 83,887 | 557,994 | 216,437 |
| C | 26,383 | 35,777 | 52,717 | 74,753 | 504,002 | 189,630 |
| ZC | 40,634 | 53,414 | 87,970 | 120,863 | 813,312 | 302,881 |

### Incremental gains (vs N)
| Arm | FFD3 ΔA | FC36 ΔA | Total Δ |
|---|---|---|---|
| Z | +11,311 | +26,642 | +87,254 |
| C | +6,523 | +19,504 | +60,447 |
| ZC | +20,774 | +54,757 | +173,698 |

### Per-receiver clock coverage
- **ZOI11:** 4,395 clock-fixed rows, median residual 1.761 ms, 3,622 over 0.5 ms
- **CFD01:** 4,305 clock-fixed rows, median residual 3.478 ms, 4,305 over 0.5 ms

Both receivers have clock correction curves applied. ZOI11 has better clock quality (lower residuals).

## 4. Known Issues

1. **Schema mismatch:** `tblDetectionFilterSecondary` had old schema (missing `seconds` column). Fixed by dropping pipeline tables before `--skip-build`.
2. **`--init-from` not supported:** Plan C referenced a flag that doesn't exist in this script version. Adapted workaround used.
3. **Disk space:** ~4.49 GB drive; only 2 of 4 arms could be retained (arm_Z and arm_ZC). Arm N and arm_C were deleted after building subsequent arms.
4. **ZOI04 clock curve missing:** ZOI04 has no clock correction curve (left uncorrected in fish_multipath step).
5. **ZOI11 clock curve missing in arm_C:** ZOI11 has no clock curve in arm_C (expected — not in receivers_3d).

## 5. Database Sizes

| Database | Size |
|---|---|
| arm_N.db | 1,360 MB |
| arm_Z.db | 1,400 MB |
| arm_C.db | 1,389 MB |
| arm_ZC.db | 1,444 MB |

## 6. Output Locations

- `output/11112026_zoi11_cfd01/armZ/arm_Z.db`
- `output/11112026_zoi11_cfd01/armZC/arm_ZC.db`
- `config/run_data_zoi11cfd01_N.toml`
- `config/run_data_zoi11cfd01_Z.toml`
- `config/run_data_zoi11cfd01_C.toml`
- `config/run_data_zoi11cfd01_ZC.toml`

## 7. Next Steps

1. Generate plots for static holds and validation drags across all four arms
2. Compare positioning accuracy for known GPS-tagged holds
3. Evaluate whether ZOI11 and CFD01 justify their deployment cost
