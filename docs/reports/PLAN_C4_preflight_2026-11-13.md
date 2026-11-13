# Plan C4 preflight

Tool version fourarm-2026-11-12-a. Times are study wall clock (PDT stored as if UTC).

- **solution row**: a root B row of `tblPositions_Deng` with comment `solution found`. One row per receiver combination. It rises with every receiver added. It is not a count of fixes.
- **fix**: for a static hold, a transmission kept by the project's receiver set consensus (two or more receiver sets within 10 m). For a drag, a point of the project's consensus, smoothing and Kalman track.
- **error**: XY distance from a fix to GPS truth.
- **nearer receiver**: the nearer of ZOI11 and CFD01, in XY.

## 1. Databases

| arm | name | path | bytes | last_write |
|---|---|---|---|---|
| N | 1. Excluded | output\11112026_zoi11_cfd01\armN\arm_N.db | 1,425,960,960 | 2026-11-11 21:24:35 |
| Z | 2. ZOI11 Only | output\11112026_zoi11_cfd01\armZ_r2\arm_Z.db | 1,465,643,008 | 2026-11-12 16:43:20 |
| C | 3. CFD01 Only | output\11112026_zoi11_cfd01\armC\arm_C.db | 1,453,572,096 | 2026-11-11 21:34:03 |
| ZC | 4. Both | output\11112026_zoi11_cfd01\armZC_r2\arm_ZC.db | 1,510,928,384 | 2026-11-12 16:48:27 |

## 2. Table census (rows)

| table | N | Z | C | ZC |
|---|---|---|---|---|
| tblDeepReceiver_step6_solution | 3 | 3 | 3 | 3 |
| tblDetectionClockFixed | 160,649 | 165,044 | 160,649 | 165,044 |
| tblDetectionFilterPrimary | 76,580 | 76,580 | 76,580 | 76,580 |
| tblDetectionFilterSecondary | 76,580 | 76,580 | 76,580 | 76,580 |
| tblDetectionRaw | 2,107,705 | 2,107,705 | 2,107,705 | 2,107,705 |
| tblInterpolatedTemp | 30,817 | 30,817 | 30,817 | 30,817 |
| tblMetronomeFiltered | 108,027 | 108,027 | 108,027 | 108,027 |
| tblMetronomeSecondFiltered | 78,130 | 78,130 | 78,130 | 78,130 |
| tblPositions_Deng | 333,430 | 557,994 | 504,002 | 813,312 |
| tblPositions_DengDBSCAN | 108,789 | 189,460 | 162,717 | 266,271 |
| tblProcessProgress | 2 | 2 | 2 | 2 |
| tblReceiver | 20 | 20 | 20 | 20 |
| tblReceiverGPS | 1,114,887 | 1,114,887 | 1,114,887 | 1,114,887 |
| tblReceiverSwap | absent | 0 | 0 | 0 |
| tblReceiver_initial | 20 | 20 | 20 | 20 |
| tblStudyParameters | 1 | 1 | 1 | 1 |
| tblTag | 41 | 41 | 41 | 41 |
| tblWSEL | 29,924 | 29,924 | 29,924 | 29,924 |

## 3. Solver table

Columns of `tblPositions_Deng`: `Unnamed: 0, transNo, solNo, r0, r1, r2, r3, X, Y, Z, T01, ToA, comment, in_hull, solution, Tag_ID`

Rows by root and comment. Rows that are not `solution found` carry 9999 and are never used.

| arm | root | comment | rows | rows_with_X_9000_or_more |
|---|---|---|---|---|
| N | A | negative quadratic discriminant - no solution | 72,329 | 72,329 |
| N | A | negative time of arrival - no soluiton | 40,612 | 40,612 |
| N | A | not enough receivers for solution | 701 | 701 |
| N | A | solution found | 53,073 | 46 |
| N | B | negative quadratic discriminant - no solution | 72,329 | 72,329 |
| N | B | negative time of arrival - no soluiton | 17,575 | 17,575 |
| N | B | not enough receivers for solution | 701 | 701 |
| N | B | solution found | 76,110 | 11 |
| Z | A | negative quadratic discriminant - no solution | 129,200 | 129,200 |
| Z | A | negative time of arrival - no soluiton | 58,201 | 58,201 |
| Z | A | not enough receivers for solution | 570 | 570 |
| Z | A | solution found | 91,026 | 87 |
| Z | B | negative quadratic discriminant - no solution | 129,200 | 129,200 |
| Z | B | negative time of arrival - no soluiton | 23,816 | 23,816 |
| Z | B | not enough receivers for solution | 570 | 570 |
| Z | B | solution found | 125,411 | 16 |
| C | A | negative quadratic discriminant - no solution | 118,432 | 118,432 |
| C | A | negative time of arrival - no soluiton | 53,886 | 53,886 |
| C | A | not enough receivers for solution | 583 | 583 |
| C | A | solution found | 79,100 | 71 |
| C | B | negative quadratic discriminant - no solution | 118,432 | 118,432 |
| C | B | negative time of arrival - no soluiton | 22,456 | 22,456 |
| C | B | not enough receivers for solution | 583 | 583 |
| C | B | solution found | 110,530 | 15 |
| ZC | A | negative quadratic discriminant - no solution | 200,319 | 200,319 |
| ZC | A | negative time of arrival - no soluiton | 77,278 | 77,278 |
| ZC | A | not enough receivers for solution | 455 | 455 |
| ZC | A | solution found | 128,604 | 119 |
| ZC | B | negative quadratic discriminant - no solution | 200,319 | 200,319 |
| ZC | B | negative time of arrival - no soluiton | 31,605 | 31,605 |
| ZC | B | not enough receivers for solution | 455 | 455 |
| ZC | B | solution found | 174,277 | 24 |

## 4. Receiver membership in solution rows (root B)

| arm | solution_rows | rows_with_ZOI11 | rows_with_CFD01 | receivers_in_rows |
|---|---|---|---|---|
| N | 76,110 | 0 | 0 | ZOI01 ZOI02 ZOI03 ZOI05 ZOI06 ZOI07 ZOI08 ZOI09 ZOI10 |
| Z | 125,411 | 49,301 | 0 | ZOI01 ZOI02 ZOI03 ZOI05 ZOI06 ZOI07 ZOI08 ZOI09 ZOI10 ZOI11 |
| C | 110,530 | 0 | 34,420 | CFD01 ZOI01 ZOI02 ZOI03 ZOI05 ZOI06 ZOI07 ZOI08 ZOI09 ZOI10 |
| ZC | 174,277 | 63,747 | 48,866 | CFD01 ZOI01 ZOI02 ZOI03 ZOI05 ZOI06 ZOI07 ZOI08 ZOI09 ZOI10 ZOI11 |

## 5. Transmissions per tag

`trans_with_solution` is the number of transmissions that have at least one solution row. Compare arms on this column, not on `solution_rows`.

| arm | tag | trans_any_row | solution_rows | trans_with_solution | solution_rows_inside_1000m | trans_after_position_screen |
|---|---|---|---|---|---|---|
| N | FC36 | 2,417 | 48,786 | 2,016 | 48,676 | 1,371 |
| N | FFD3 | 2,174 | 27,324 | 1,487 | 27,268 | 976 |
| Z | FC36 | 2,430 | 83,887 | 2,146 | 83,726 | 1,559 |
| Z | FFD3 | 2,193 | 41,524 | 1,610 | 41,452 | 1,079 |
| C | FC36 | 2,427 | 74,753 | 2,076 | 74,606 | 1,426 |
| C | FFD3 | 2,182 | 35,777 | 1,553 | 35,713 | 982 |
| ZC | FC36 | 2,431 | 120,863 | 2,170 | 120,600 | 1,520 |
| ZC | FFD3 | 2,194 | 53,414 | 1,657 | 53,327 | 1,117 |

## 6. Receiver coordinates (arm N) and XY shift of each arm against arm N

| receiver | X_t | Y_t | Z_t | xy_shift_Z_m | xy_shift_C_m | xy_shift_ZC_m |
|---|---|---|---|---|---|---|
| ZOI01 | 447.072 | 124.773 | -11.150 | 0.000 | 0.000 | 0.000 |
| ZOI02 | 467.600 | 116.365 | -12.395 | 0.000 | 0.000 | 0.000 |
| ZOI03 | 486.071 | 106.060 | -10.287 | 0.000 | 0.000 | 0.000 |
| ZOI04 | 444.396 | 139.409 | -3.251 | 0.000 | 0.000 | 0.000 |
| ZOI05 | 508.560 | 135.279 | -3.099 | 0.000 | 0.000 | 0.000 |
| ZOI06 | 475.918 | 134.263 | -3.099 | 0.000 | 0.000 | 0.000 |
| ZOI07 | 442.712 | 99.448 | -2.286 | 0.000 | 0.000 | 0.000 |
| ZOI08 | 459.682 | 88.602 | -2.844 | 0.000 | 0.000 | 0.000 |
| ZOI09 | 477.667 | 79.520 | -2.261 | 0.000 | 0.000 | 0.000 |
| ZOI10 | 437.928 | 135.027 | -3.251 | 0.000 | 0.000 | 0.000 |
| ZOI11 | 454.690 | 73.978 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD01 | 467.362 | 68.789 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD02 | 496.201 | 38.685 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD03 | 509.285 | 60.290 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD04 | 505.184 | 99.148 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD05 | 539.080 | 36.917 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD06 | 540.803 | 64.058 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD07 | 539.318 | 84.827 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD08 | 530.873 | 101.435 | -3.048 | 0.000 | 0.000 | 0.000 |
| CFD09 | 532.584 | 124.631 | -3.048 | 0.000 | 0.000 | 0.000 |

## 7. Do transmission numbers mean the same transmission in every arm?

First arrival time per transmission number, each arm against arm N.

| tag | arm | common_trans | first_arrival_differs_over_0_5s | max_abs_diff_s |
|---|---|---|---|---|
| FC36 | Z | 2,494 | 312 | 2.4145 |
| FC36 | C | 2,496 | 0 | 0.0000 |
| FC36 | ZC | 2,494 | 312 | 2.4145 |
| FFD3 | Z | 2,303 | 297 | 2.7576 |
| FFD3 | C | 2,305 | 0 | 0.0000 |
| FFD3 | ZC | 2,303 | 297 | 2.7576 |

## 8. Inputs

- GPS file: `K:\Jobs\5662\001\Data\DataTrans\2025_Data\CowlitzAT2025_Data_Deliverables\5_array_testing\array_testing_drag_GPS.csv` (exists: True), 13 drag lines
- Testing workbook: `K:\Jobs\5662\001\Data\DataTrans\2025_Data\CowlitzAT2025_Data_Deliverables\5_array_testing\cowlitz_AT_2025_testing_sheets.xlsx` (exists: True)
- Data window of arm N (first arrivals): 2025-06-10 05:43:25 to 2025-06-13 22:44:29
- Tags: FC36, FFD3

## 9. Checks

- same tables in all four arms (tblReceiverSwap aside): **PASS**
- membership: N neither, Z ZOI11 only, C CFD01 only, ZC both: **PASS**
- receiver coordinates equal in all arms (0.001 m): **PASS**
- local origin equal in all arms (0.001 m): **PASS**
- transmission numbers aligned (no first arrival differs by more than 0.5 s): **FAIL**
