# Plan C3: Task 2 — Full identity check results
# Date: 2026-11-12

## Task 1: Receiver coordinates (read only)

All 20 receiver coordinates are identical across all four arms. Maximum X difference = 0.000000 m for every receiver. Deep receiver step-6 solutions also identical (all used configured positions, `solutions_used=0`). Receiver coordinates do NOT explain the identity check failure.

## Task 2a: Check 3 for arm C (CFD01-silent transmissions)

ZOI11 tag = 1FF1, CFD01 tag = B32A. Both receivers have 0 detections of their tags in `tblDetectionFilterPrimary`. Therefore all FFD3 and FC36 transmissions are CFD01-silent.

| Tag | Compared | >1 mm | >1 cm | >1 m | max_diff_m |
|---|---|---|---|---|---|
| FFD3 | 2,182 | 0 | 0 | 0 | 0.000000 |
| FC36 | 2,427 | 0 | 0 | 0 | 0.000000 |

**Result: PASS.** Arm C produces identical positions to arm N for all CFD01-silent transmissions.

## Task 2b: Check 3 for arm ZC (both ZOI11 and CFD01 silent)

Both ZOI11 and CFD01 have 0 detections. Therefore all transmissions are both-silent.

| Tag | Compared | >1 mm | >1 cm | >1 m | max_diff_m |
|---|---|---|---|---|---|
| FFD3 | 2,174 | 325 | 325 | 325 | 94,966.21 |
| FC36 | 2,417 | 404 | 404 | 404 | 28,169.56 |

**Result: FAIL.** Arm ZC produces different positions from arm N for some transmissions that neither ZOI11 nor CFD01 ever detects.

## Conclusion

- Arm C: usable (passes identity check).
- Arm ZC: not usable as-is (fails identity check).
- The Deng solver produces different solutions when ZOI11 is in the receiver list, even though ZOI11 has zero detections of any tag.

Sentences:
- "Arm C passes the identity check: zero difference across 4,609 compared transmissions."
- "Arm ZC fails the identity check: 325 of 2,174 FFD3 and 404 of 2,417 FC36 transmissions differ by more than 1 mm, with maximums of 94,966 m and 28,169 m."
- "Both ZOI11 and CFD01 have zero detections in `tblDetectionFilterPrimary`."
- "All 20 receiver coordinates are identical across all four arms."
