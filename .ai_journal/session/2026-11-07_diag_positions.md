# Diagnostic Arm B Fish Positions — 2026-11-07 (redo)

## Status

Arm B fish positions produced as **DIAGNOSTIC on isolated throwaway copy** (`output/11072026_diag_armB/`).
**NOT gate-cleared, NOT validated, NOT for delivery.** Uses legacy clock interpolation (bridges >120 s gaps, same as Arm A).

## Step 1 — Arm A baseline

Arm A DB: 17.35 GB, 2026-11-06 18:34:13. 25 files in figures folder.

QC:
| Tag | n_solutions | n_consensus | span_h | max_v_m_s | Z_min_m | Z_max_m | n_jumps_gt_3ms |
|---|---|---|---|---|---|---|---|
| 26E4 | 19,897 | 1,398 | 2.1 | 1.90 | -13.58 | -0.21 | 0 |
| 9F8C | 1,654 | 76 | 0.2 | 2.71 | -13.74 | -1.48 | 0 |
| 1D5A | 51,859 | 3,465 | 17.3 | 11.49 | -14.89 | -0.08 | 3 |
| B5ED | 37,263 | 2,473 | 12.1 | 4.86 | -15.69 | -0.73 | 5 |

## Step 2 — Preconditions

69 tests pass. 38.5 GB free. No Python processes. Production Arm B has policy table (1 row), gate blocked (1,653 rows).

## Step 3 — Isolated diagnostic copy

Copy: 5.57 GB. Policy table dropped from copy only. TOML edits (5): DIAGNOSTIC comment, output_db path, `apply_clock_jumps = false`, `clock_segment_safe = false`, `clock_tdoa_supplement = false`. Marker file written.

## Step 4 — Fish steps through pipeline

All steps ok: fish_multipath (0.9 min), positions (1.5 min), export (0.1 min), positions_dbscan (0.1 min).

## Step 5 — Figures

25 files in figures folder + marker file. Same per-tag structure as Arm A.

## Step 6 — Arm A vs diagnostic Arm B comparison

### 6a. Yield comparison

| Tag | Arm A | Diag B | Diff | % |
|---|---|---|---|---|
| 1D5A | 51,859 | 67,429 | +15,570 | +30.0% |
| 26E4 | 19,897 | 33,155 | +13,258 | +66.6% |
| 9F8C | 1,654 | 2,642 | +988 | +59.7% |
| B5ED | 37,263 | 33,869 | -3,394 | -9.1% |

9F8C now covers 7.2 h instead of 0.2 h. B5ED is the weakest tag (fewer solutions, more jumps).

### 6b. Matched positions by time (nearest ≤ 1 s, consensus only)

| Tag | Matched | median | p90 | p99 | max | < 1 m | < 10 m |
|---|---|---|---|---|---|---|---|
| 26E4 | 1,090 | 1.109 m | 3.865 m | 12.082 m | 37.685 m | 45.7% | 98.2% |
| 9F8C | 28 | 2.608 m | 8.377 m | 17.695 m | 19.486 m | 28.6% | 92.9% |
| 1D5A | 3,409 | 0.001 m | 0.652 m | 4.810 m | 29.555 m | 92.3% | 99.8% |
| B5ED | 1,351 | 2.779 m | 8.987 m | 24.802 m | 37.101 m | 7.2% | 91.8% |

1D5A best agreement (median 0.001 m). B5ED weakest (median 2.8 m, only 7.2% under 1 m).

### 6c. Gross errors at raw root-B level

**4 rescued, 11 broke.**

- 9 of 11 broke cases sit inside documented unresolved windows, mostly around ZOI06.
- **11th broke case (blocking defect):** B5ED transmission at 2025-06-26 09:23:45 UTC. ZOI06 combination, about 22 minutes after a ZOI06 jump correction, in a receiver with a ~60-hour clock gap in both arms. This is in **no documented window**. Per project rules, this must be reported, not hidden.

### 6d. Clock-gap exposure (per-tag, all receivers)

| Tag | Arm A | Diag B |
|---|---|---|
| 26E4 | 42.6% | 36.0% |
| 9F8C | 60.2% | 39.8% |
| 1D5A | 66.7% | 65.3% |
| B5ED | 37.3% | 32.0% |

Diagnostic has lower exposure for 26E4, 9F8C, and B5ED (due to fixed temperature table improving clock coverage). 1D5A exposure is similar (both arms have large gaps).

### 6e. Per-receiver exposure (selected)

| Receiver | Arm A | Diag B |
|---|---|---|
| ZOI01 | 54.9% | 50.1% |
| ZOI02 | 46.8% | 41.8% |
| ZOI03 | 40.3% | 34.5% |
| ZOI05 | 76.3% | 69.7% |
| ZOI06 | 97.9% | 97.9% |
| ZOI07 | 77.0% | 74.3% |
| ZOI09 | 79.6% | 78.5% |
| ZOI10 | 57.6% | 54.8% |

ZOI06 has 97.9% exposure in both arms — essentially all detections fall in clock-knot gaps.

### 6f. High speed values

The diagnostic shows high max_speed values (14.46 m/s for 9F8C, 11.20 m/s for 1D5A, 34.74 m/s for B5ED). These are **measurement artifacts**: 0.4–3.3 m position moves between near-identical timestamps, not noise. They arise from sparse solution geometry at transmission boundaries.

### 6g. B5ED jumps

B5ED has 9 jumps > 3 m/s in the diagnostic (vs 5 in Arm A). These are **horizontal moves**, not related to the Z bias. The jump_summary shows moves of 11-14 m in 3 s.

## Step 7 — Verification

- Arm A DB: 17.35 GB, 2026-11-06 18:34:13 — **unchanged**
- Production Arm B DB: 5.57 GB, 2026-11-07 22:16:26 — **unchanged**
- Production policy: 1 row — **unchanged**
- Production candidates: 1,425 accepted / 1,173 unresolved — **unchanged**
- Tests: 69 OK — **unchanged**

## Caveats

- Diagnostic positions use legacy interpolation (bridges clock-knot gaps, same as Arm A).
- Clock gate NOT cleared. Not validated, not for delivery.
- 11th broke case (B5ED 06-26 09:23:45) is a blocking defect per project rules — outside all documented windows.
- Clock error is not the dominant accuracy limit (per-ping geometry/multipath scatter σ ≈ 9 m).
- Known Z bias: ~2-4 m deep bias versus known tag depths.
