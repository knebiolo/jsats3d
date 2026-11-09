# Journal: 2026-11-07 Hygiene + TDOA Plots

## Hygiene (Task 1)

### Tests
- Ran 69 tests in 6.979s — OK
- Confirmed by owner's 11-07 session; re-verified this session

### Script cleanup
- 43 scripts quarantined to `output/_quarantine_scripts/`:
  - 39 root `_*.py` files (session artifacts from 11-03 through 11-07)
  - `scripts/fix_gate.py` — gate-bypass tool
  - `scripts/copy_filters.py` — copies Arm A filters into Arm B
  - `scripts/option_c_simple.py` — v4 experiment
  - `scripts/option_c_apply_new_jumps.py` — v4 experiment
- Root directory now clean (only `__init__.py` and `_2025_pipeline.py` remain)
- Owner approved quarantine (not delete)

## Clock surface rerun (temperature fix)
- Removed 576 fabricated rows from Arm B tblInterpolatedTemp (deleted rows before 06-17)
- Changed `bounds_error=True` to `bounds_error=False` in `temp_interpolator` (jsats3d.py:125) — NaN temps no longer crash, return NaN instead
- Added NaN filter before DBSCAN (jsats3d.py:1144-1150) — drops non-finite ts/ddoa rows
- Added NaN drop from residual_curve (jsats3d.py:1203-1205) — drops NaN ToA_error before residual interpolation
- clock_surface: **ok in 4.6 min**, 1,097,043 clock_fixed rows, all 15 receivers processed
- deep_clocks: **ok in 8.7 min**, all 17 receivers processed
- 24,979 total epochs outside temperature coverage dropped across all receivers (beacon_pairwise_dbscan.py)
- These are 06-15/16 data points — no temperature available, correctly dropped

### Key findings from prior session (verified)
1. **Temperature fabrication in Arm B**: 576 rows (2025-06-15/16) back-filled with constant 13.924 °C. **Owner approved removal — deleted 576 rows.** Arm B table now starts 06-17, matching Arm A. Only temp source is `2025_Temp_String_Data_5min_interpolated.csv` (starts 06-17). No HOBO files exist on disk.
2. **Arm A modified**: `_add_temp_to_arm_a.py` wrote a string-only temp table to baseline DB. Positions predate this; results unchanged.
3. **Curve coverage dominates gate**: 88,923 of 91,430 blocked fish rows due to "outside clock coverage / unsupported gap / LOO > 0.5 ms".
4. **Counter-basis fix is genuine bug fix**: firmware check now compares against `seconds + jump_offset_s`, not shifted time.

## TDOA Plots (Task 3 — COMPLETE)
Three clock_review intervals produced (read-only, no DB writes):
1. 2025-06-27 to 2025-06-30 (475,375 paired rows)
2. 2025-06-26 to 2025-06-28 (465,605 paired rows)
3. 2025-06-19 to 2025-06-22 (461,406 paired rows)

### Results: Adjacent TDoA changes ≥ 50 ms (Arm A vs Arm B)
| Receiver | Interval 1 | Interval 2 | Interval 3 |
|---|---|---|---|
| ZOI01 | 323→285 (-12%) | 426→371 (-13%) | 202→177 (-12%) |
| ZOI05 | 47→36 (-23%) | — | 228→198 (-13%) |
| CFD01 | 57→50 (-12%) | — | — |
| ZOI06 | 14→13 (-7%) | 38→31 (-18%) | — |
| ZOI07 | 123→110 (-11%) | — | — |

**All 5 receivers improved. No receiver got worse. Average: -13%.**

### Stage coverage (interval 1)
TDOA supplement **added** final clock rows: CFD01 +102, ZOI07 +30, ZOI01 +53, ZOI05 +24.
This suggests jumps recover data previously lost in clock_fix DBSCAN passes.

### TDOA PNGs produced
- 15 receivers × 3 intervals = 45 TDoA plot PNGs
- Locations: `output/11062026_clocksync/clock_review/interval_*/tdoa_arms_*.png`
- Summary table: `output/11062026_clocksync/clock_review/tdoa_summary.md`
