# Plan C4 identity check

Tool version fourarm-2026-11-12-a. Times are study wall clock (PDT stored as if UTC).

- **solution row**: a root B row of `tblPositions_Deng` with comment `solution found`. One row per receiver combination. It rises with every receiver added. It is not a count of fixes.
- **fix**: for a static hold, a transmission kept by the project's receiver set consensus (two or more receiver sets within 10 m). For a drag, a point of the project's consensus, smoothing and Kalman track.
- **error**: XY distance from a fix to GPS truth.
- **nearer receiver**: the nearer of ZOI11 and CFD01, in XY.

A transmission is **compared** when the arm has solution rows for it, none of those rows contains an added receiver, and arm N also has solution rows for it. For such a transmission the arm should hold the same rows as arm N, so the medians should be equal.

`compared_but_heard_by_added` counts compared transmissions that the added receiver did detect (`tblDetectionFilterSecondary`) although no solution row contains it.

## 2. ZOI11 Only against 1. Excluded (added: ZOI11)

| tag | trans_in_arm | trans_with_added_receiver_row | compared | same_row_count | over_1mm | over_1m | over_100m | median_m | p99_m | max_m | over_1mm_with_same_row_count | over_1mm_using_deep_receiver | compared_but_heard_by_added |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FC36 | 2,146 | 1,922 | 224 | 224 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 | 37 |
| FFD3 | 1,610 | 926 | 684 | 684 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 | 38 |

- Tag FC36: all 224 compared transmissions are equal to arm N within 1 mm.

- Tag FFD3: all 684 compared transmissions are equal to arm N within 1 mm.

## 3. CFD01 Only against 1. Excluded (added: CFD01)

| tag | trans_in_arm | trans_with_added_receiver_row | compared | same_row_count | over_1mm | over_1m | over_100m | median_m | p99_m | max_m | over_1mm_with_same_row_count | over_1mm_using_deep_receiver | compared_but_heard_by_added |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FC36 | 2,076 | 1,412 | 664 | 664 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 | 107 |
| FFD3 | 1,553 | 583 | 970 | 970 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 | 40 |

- Tag FC36: all 664 compared transmissions are equal to arm N within 1 mm.

- Tag FFD3: all 970 compared transmissions are equal to arm N within 1 mm.

## 4. Both against 1. Excluded (added: ZOI11, CFD01)

| tag | trans_in_arm | trans_with_added_receiver_row | compared | same_row_count | over_1mm | over_1m | over_100m | median_m | p99_m | max_m | over_1mm_with_same_row_count | over_1mm_using_deep_receiver | compared_but_heard_by_added |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FC36 | 2,170 | 1,995 | 175 | 175 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 | 15 |
| FFD3 | 1,657 | 1,145 | 512 | 512 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0 | 42 |

- Tag FC36: all 175 compared transmissions are equal to arm N within 1 mm.

- Tag FFD3: all 512 compared transmissions are equal to arm N within 1 mm.

## Result

**IDENTITY HOLDS in all three arms.**
