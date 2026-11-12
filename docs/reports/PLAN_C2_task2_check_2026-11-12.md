# Plan C2: Task 2 — Kept Arms Z and ZC Usability Check
# Date: 2026-11-12

## Task 2a: Table census

| Table | N | Z | C | ZC |
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
| tblReceiver_initial | 20 | 20 | 20 | 20 |
| tblStudyParameters | 1 | 1 | 1 | 1 |
| tblTag | 41 | 41 | 41 | 41 |
| tblWSEL | 29,924 | 29,924 | 29,924 | 29,924 |

Only `tblReceiverSwap` exists in arm C (0 rows) but not others. All other tables present in all four arms.

## Task 2b: Receiver membership and identity checks

### Check 2: Receiver membership in receivers_3d
| Arm | ZOI11 | CFD01 | Result |
|---|---|---|---|
| N | False | False | PASS |
| Z | True | False | PASS |
| C | False | True | PASS |
| ZC | True | True | PASS |

### Check 3: Identity where receiver is silent
Comparing arm Z vs arm N for FFD3 transmissions where ZOI11 has zero detections:
- FFD3: 2,174 transmissions compared, 306 different (>1 mm), max diff = 94,966 m
- FC36: 2,417 transmissions compared, 407 different (>1 mm), max diff = 28,169 m

**Check 3 FAILS.** Adding ZOI11 to the receiver list changes Deng solver positions even for transmissions ZOI11 never hears. The massive differences (tens of kilometers) indicate the solver geometry is fundamentally different.

Similarly, arm ZC would fail the same check (it inherits Z's receiver list plus CFD01).

For arm C, the same check would compare CFD01-silent transmissions between arm C and arm N. This was not run because Check 3 already failed for Z.

## Task 2c: Clock residuals (arm ZC)

| Rec_ID | rows | median_ms | p95_ms | over_0.5ms | pct_over |
|---|---|---|---|---|---|
| CFD01 | 4,305 | 3.478 | 917.8 | 4,305 | 100.0% |
| CFD02 | 3,438 | 7.225 | 22.3 | 3,398 | 98.8% |
| CFD03 | 3,256 | 3.528 | 400.7 | 2,956 | 90.8% |
| CFD04 | 2,936 | 4.354 | 6.2 | 2,932 | 99.9% |
| CFD06 | 828 | 5.142 | 401.3 | 760 | 91.8% |
| CFD07 | 2,865 | 3.503 | 398.2 | 2,630 | 91.8% |
| CFD08 | 2,118 | 5.050 | 394.6 | 2,051 | 96.8% |
| CFD09 | 1,912 | 1.688 | 16.2 | 1,588 | 83.1% |
| ZOI01 | 4,364 | 5.714 | 20.1 | 4,302 | 98.6% |
| ZOI02 | 4,414 | 6.259 | 405.3 | 4,414 | 100.0% |
| ZOI03 | 4,516 | 5.540 | 13.6 | 4,516 | 100.0% |
| ZOI05 | 3,541 | 8.625 | 14.7 | 3,445 | 97.3% |
| ZOI06 | 3,099 | 4.468 | 402.7 | 2,837 | 91.5% |
| ZOI07 | 2,398 | 0.126 | 0.6 | 194 | 8.1% |
| ZOI08 | 113,940 | 0.000 | 0.0 | 0 | 0.0% |
| ZOI09 | 1,728 | 943.8 | 975.9 | 1,693 | 98.0% |
| ZOI10 | 991 | 6.436 | 21.1 | 972 | 98.1% |
| ZOI11 | 4,395 | 1.761 | 880.0 | 3,622 | 82.4% |

Sentences:
- "ZOI11's median clock residual is 1.761 ms. The nine receivers of the base array range from 0.126 ms to 943.8 ms."
- "CFD01's median clock residual is 3.478 ms. 100.0% of its rows are over the 0.5 ms budget. The other CFD receivers range from 1.688 ms to 7.225 ms."
- "1 ms is about 1 m. This table is timing only. It is not a position error."

## STOP — Check 3 failed for arm Z

**Arm Z and arm ZC are not usable as-is.** The Deng solver produces different positions when ZOI11 is in the receiver list, even for transmissions ZOI11 never detects. Max position differences are tens of kilometers.

The owner must decide:
1. Rebuild arm Z (and ZC) with a different approach
2. Investigate why the Deng solver geometry changes so dramatically
3. Proceed with arm N and arm C only (if Check 3 passes for C)
