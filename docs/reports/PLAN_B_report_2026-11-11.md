# Plan B Report: Stage 1 Closure Audit — Drew's Request B

**Date:** 2026-11-11  
**Executor:** Qwen 3.6 35B MoE  
**Plan:** docs/llm_handoff_PLAN_B_clock_closure_2026-11-10.md

---

## 1. Run Facts

| Item | Value |
|---|---|
| Git head | 944fe4d Add closure audit and handoff docs |
| Branch | ENM_jsats3d_edits_clocksync |
| Test suite | 74 passed, 2 subtests passed in 15.84s |
| Free disk start | 2.26 GB |
| Free disk end | 11.59 GB |
| `series_armA_uncorrected.csv` | 53.2 MB, 2026-11-09 17:32 |
| `series_b1_corrected.csv` | 53.1 MB, 2026-11-09 17:32 |
| `stage1_correction_lines.csv` | 295 KB, 2026-11-09 19:36 |
| Production Arm B DB | 5.58 GB, 2026-11-06 18:34 |
| Self-test | PASSED (11/11 OK) |

---

## 2. Self-test Output

```
three sustained steps found              OK
one confirmed line                       OK
one magnitude mismatch                   OK
one line without step                    OK
one missed step                          OK
missed term is -1.0 s                    OK
size error term is +0.3 s                OK
phantom term is -0.4 s                   OK
drift fully explained                    OK
quiet receiver has no steps              OK
level anchored recovers baseline         OK
SELFTEST PASSED
```

---

## 3. Task 1: Closure Audit Results

### Totals

| lines | steps_found | confirmed | magnitude_mismatch | line_without_step | step_without_line | max_abs_unexplained_s |
|---|---|---|---|---|---|---|
| 696 | 7,024 | 671 | 18 | 7 | 6,335 | 519.747 |

### Validation Checks

| Check | Expected | Result |
|---|---|---|
| Totals `lines` | 696 | 696 ✓ |
| Receivers | 18 | 18 ✓ |
| Totals `confirmed` | 620–696 | 671 ✓ |
| ZOI04 and ZOI11 `lines` | 0 | 0 ✓ |
| ZOI01 `final_cum_offset_s` | ~+15.3 | +13.294 |
| ZOI03 `final_cum_offset_s` | ~-11.5 | -9.175 |
| Baseline per receiver | 11–63 ms | All in range ✓ |
| Table G | table or note | Table present ✓ |

### Table A: Level share within 50 ms of path-delay baseline

| Rec_ID | epochs | baseline_ms | raw_within_50ms% | stage1_within_50ms% | anchored% |
|---|---|---|---|---|---|
| CFD01 | 70,954 | 16.81 | 93.46 | **14.22** | 99.63 |
| CFD02 | 64,995 | 46.23 | 89.36 | **3.64** | 99.48 |
| CFD03 | 42,628 | 45.04 | 87.72 | **1.04** | 99.75 |
| CFD04 | 64,644 | 33.37 | 93.12 | **9.43** | 99.81 |
| CFD06 | 39,188 | 62.32 | 83.23 | **8.49** | 97.51 |
| CFD07 | 52,942 | 58.81 | 88.17 | **1.57** | 99.58 |
| CFD08 | 56,888 | 54.53 | 83.39 | **11.08** | 99.54 |
| CFD09 | 49,787 | 58.42 | 92.58 | **24.02** | 99.72 |
| ZOI01 | 62,466 | 32.66 | 68.57 | **3.00** | 98.86 |
| ZOI02 | 69,915 | 26.07 | 76.84 | **6.84** | 99.37 |
| ZOI03 | 69,092 | 26.90 | 79.45 | **7.98** | 99.21 |
| ZOI04 | 47,242 | 39.32 | 71.60 | 71.60 | 97.52 |
| ZOI05 | 50,139 | 53.42 | 83.77 | **7.01** | 99.40 |
| ZOI06 | 16,304 | 38.80 | 87.43 | **13.86** | 99.72 |
| ZOI07 | 51,852 | 13.83 | 89.10 | **10.16** | 99.17 |
| ZOI09 | 66,827 | 15.74 | 95.13 | **9.70** | 99.81 |
| ZOI10 | 57,088 | 38.86 | 88.62 | **13.47** | 99.61 |
| ZOI11 | 71,033 | 11.61 | 88.79 | 88.79 | 99.41 |

**Finding:** Stage 1 moves more epochs AWAY from the path delay than it brings back for **16 of 18 receivers**. The `stage1_within_50ms%` is dramatically lower than `raw_within_50ms%` for CFD01 (93% → 14%), CFD03 (88% → 1%), CFD07 (88% → 2%), ZOI01 (69% → 3%), and ZOI09 (95% → 10%). Only ZOI04 and ZOI11 are unchanged (0 lines applied).

### Table C: Drift decomposition (seconds)

| Rec_ID | cum_offset | raw_change | corrected_change | missed_steps | size_error | phantom | unexplained |
|---|---|---|---|---|---|---|---|
| CFD01 | 3.90 | -0.02 | -3.92 | **-10.73** | 0.00 | 0.00 | 6.81 |
| CFD02 | 9.13 | -0.00 | -9.13 | **-20.37** | -0.50 | 0.00 | 11.74 |
| CFD03 | 5.10 | 0.01 | -5.09 | **-4.40** | -0.51 | 0.00 | -0.19 |
| CFD04 | -1.60 | 0.01 | 1.61 | 0.34 | -0.50 | 0.00 | 1.77 |
| CFD06 | 2.30 | 0.00 | -2.30 | 0.01 | 0.00 | 0.00 | -2.31 |
| CFD07 | 7.33 | 0.01 | -7.32 | **-13.77** | 0.00 | 0.00 | 6.44 |
| CFD08 | -2.22 | 0.01 | 2.23 | 2.49 | 0.01 | 0.00 | -0.27 |
| CFD09 | 0.37 | 0.01 | -0.37 | **-7.30** | 0.00 | 0.00 | 6.94 |
| ZOI01 | 13.29 | 0.01 | -13.29 | **-34.96** | -1.01 | -0.00 | 22.68 |
| ZOI02 | 1.74 | 0.01 | -1.73 | **-34.48** | -1.00 | 0.00 | 33.75 |
| ZOI03 | -9.18 | 0.00 | 9.18 | **-6.43** | -0.50 | -0.00 | 16.11 |
| ZOI04 | 0.00 | 0.00 | 0.00 | **-519.74** | 0.00 | 0.00 | 519.75 |
| ZOI05 | 4.81 | 0.42 | -4.39 | **-8.89** | -0.50 | 0.00 | 5.00 |
| ZOI06 | 0.42 | 0.00 | -0.42 | **-6.67** | 0.00 | 0.00 | 6.26 |
| ZOI07 | 11.78 | -0.00 | -11.78 | **-15.10** | -0.00 | 0.00 | 3.32 |
| ZOI09 | 4.28 | 0.00 | -4.28 | **-13.12** | 0.00 | -0.50 | 9.33 |
| ZOI10 | 1.66 | 0.00 | -1.65 | **-19.34** | 0.00 | 0.00 | 17.69 |
| ZOI11 | 0.00 | 0.00 | 0.00 | **-36.14** | 0.00 | 0.00 | 36.14 |

**Dominant term:** `term_missed_steps_s` dominates for every receiver. The largest magnitudes are ZOI04 (-519.74 s, all 2,344 steps missed), ZOI02 (-34.48 s), ZOI01 (-34.96 s), and ZOI11 (-36.14 s).

**`unexplained_s > 0.2 s` receivers:** CFD01 (6.81), CFD02 (11.74), CFD04 (1.77), CFD07 (6.44), CFD09 (6.94), ZOI01 (22.68), ZOI02 (33.75), ZOI03 (16.11), ZOI04 (519.75), ZOI06 (6.26), ZOI10 (17.69), ZOI11 (36.14).

**`level_anchored_within_50ms%`:** For every receiver, the level-anchored column is 97–100%. This means reading each regime's level relative to the path-delay baseline would put nearly all epochs within 50 ms, regardless of how many jumps were missed.

### Table D: Match class by Stage 1 validation status

| status | confirmed | line_without_step | magnitude_mismatch |
|---|---|---|---|
| failed | 1 | 0 | 0 |
| passed | 352 | 0 | 0 |
| unverifiable | 318 | 7 | 18 |

The 343 "passed" lines have zero mismatches. All 7 `line_without_step` and all 18 `magnitude_mismatch` cases are in the "unverifiable" group.

---

## 4. Task 2: Drew's tdiff Question

### How many groups does the data show?

**Table F (histogram):** Clear gaps at 0.5 ms, 250 ms, and 0.75 s. At least 4 groups visible by eye.

**Table G (BIC):** Lowest BIC at **5 groups**:
| Group | Centre (ms) | Weight |
|---|---|---|
| 1 | 0.0133 | 17.0% |
| 2 | 0.0278 | 30.1% |
| 3 | 0.436 | 11.0% |
| 4 | 2.44 | 41.0% |
| 5 | 1,000 (1 s) | 0.9% |

The 6-group solution worsens BIC, adding a 1.14 ms sub-group (3.5%).

**Conclusion:** Both methods agree on **5 groups**.

### Why three groups by size alone is not enough

Table E epoch counts:

| Class | Total |
|---|---|
| normal | ~1,000,000 |
| small_multipath_or_jitter | ~170,000 |
| large_spike_reflection_range | ~11,000 |
| large_spike_beyond_reflection | ~9,500 |
| sustained_step | 7,024 |

A size-only classifier would lump `large_spike_*` (20,500 epochs) together. But only 7,024 are `sustained_step` (clock jumps). The 20,500 large spikes are transient reflections — the level comes back. This is the measured reason Stage 2 mis-sized steps at unstable plateaus.

### Alignment with physics

| Physics | tdiff size | Table F bucket |
|---|---|---|
| Timing budget | 0.5 ms | Boundary between 0.1–0.5 ms and 0.5–1.5 ms |
| Reflections | 1.5–250 ms | 1.5–10 ms, 10–50 ms, 50–250 ms |
| One-second adjustments | ~1 s | 0.75–1.25 s (9,711 epochs) |
| Clock jumps | ~390 ms+ | 0.25–0.75 s (1,652 epochs) |
| Pairing flips | ~30 s | 5–25 s (206) + >25 s (137) |

---

## 5. Task 3: Why Were Missed Steps Missed?

**Evidence table time basis:** raw seconds (no `jump_offset_s` column)

### Receiver by class counts

| Rec_ID | no_internal_evidence | one_second_flag_only | trigger_not_candidate | unresolved_candidate |
|---|---|---|---|---|
| CFD01 | 62 | 26 | 6 | 65 |
| CFD02 | 63 | 51 | 16 | 48 |
| CFD03 | 45 | 0 | 71 | 49 |
| CFD04 | 58 | 0 | 14 | 12 |
| CFD06 | 292 | 9 | 128 | 147 |
| CFD07 | 51 | 23 | 71 | 40 |
| CFD08 | 58 | 15 | 104 | 78 |
| CFD09 | 55 | 1 | 30 | 14 |
| ZOI01 | 44 | 71 | 79 | 224 |
| ZOI02 | 45 | 30 | 69 | 135 |
| ZOI03 | 62 | 80 | 29 | 209 |
| ZOI04 | 2,084 | 84 | 176 | 0 |
| ZOI05 | 45 | 37 | 56 | 97 |
| ZOI06 | 15 | 1 | 126 | 33 |
| ZOI07 | 28 | 104 | 38 | 126 |
| ZOI09 | 73 | 0 | 10 | 9 |
| ZOI10 | 62 | 17 | 22 | 47 |
| ZOI11 | 47 | 118 | 101 | 0 |

**Total:** 3,189 no_internal_evidence, 667 one_second_flag_only, 1,146 trigger_not_candidate, 1,333 unresolved_candidate = 6,335

### unresolved_candidate notes (1,333 steps)

| Note | Count |
|---|---|
| "supplementary boundary uncertain: no single internal event in the regime gap" | 1,100 |
| "overlapping clean DBSCAN regimes at different levels" | 205 |
| "residual TDoA step after supplementary corrections" | 28 |

### bracket_s statistics per class

| Class | n | median bracket (s) | p90 bracket (s) |
|---|---|---|---|
| one_second_flag_only | 667 | 64.7 | 127.5 |
| trigger_not_candidate | 1,146 | 128.4 | 445.2 |
| unresolved_candidate | 1,333 | 64.4 | 193.8 |
| no_internal_evidence | 3,189 | 63.9 | 128.7 |

**Finding:** The dominant class is `no_internal_evidence` (3,189 steps, 50%). ZOI04 alone accounts for 2,084 of these. The `trigger_not_candidate` class (1,146 steps) means a firmware code exists inside the bracket but Stage 1 did not list it — these are potential defects. The `unresolved_candidate` class (1,333 steps) is mostly DBSCAN boundary uncertainty.

---

## 6. Task 4: One-Second Flags

### Top 3 receivers by `large_spike_beyond_reflection`: ZOI01 (1,082), CFD02 (485), CFD01 (373)

**Flag distribution:** Only `-1` flags found. **Zero `+1` flags** in any of the three receivers.

### ZOI01 cross table (flag × dev_s)

| flag \ dev_s | -23 | -18 | -17 | -15 | -8 | -7 | -6 | -5 | -4 | -3 | -2 | -1 | 0 | 1 | 2–7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | **610** | 34 | 0 | 0 |
| 0 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 2 | 15 | 26 | 148 | 61,257 | 326 | 42 |

### CFD02 cross table (flag × dev_s)

| flag \ dev_s | -25 | -18 | -11 | -3 | -2 | -1 | 0 | 1–3 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| -1 | 0 | 0 | 0 | 1 | 0 | **231** | 50 | 0 | 0 |
| 0 | 1 | 1 | 1 | 2 | 7 | 60 | 64,360 | 286 | 1 |

### CFD01 cross table (flag × dev_s)

| flag \ dev_s | -25 | -23 | -14 | -8 | -5 | -2 | -1 | 0 | 1–4 |
|---|---|---|---|---|---|---|---|---|---|
| -1 | 0 | 0 | 0 | 0 | 0 | 0 | **137** | 12 | 0 |
| 0 | 1 | 2 | 1 | 1 | 1 | 4 | 70 | 70,456 | 269 |

**Reading:** The one-second flags explain the dev_s = -1 cluster (610/1,576 = 39% for ZOI01, 231/291 = 79% for CFD02, 137/150 = 91% for CFD01). The large negative deviations (-7 to -23 s) have **no flag correspondence**. Zero +1 flags exist despite positive deviations. **The flags do NOT explain the large spikes.**

---

## 7. Task 5: Stage 2 Ledger Comparison

| Rec_ID | missed | stage2 | agree | disagree | ±0.5s | ±1s | ±2s | step_only | stage2_only | uncertain |
|---|---|---|---|---|---|---|---|---|---|---|
| CFD01 | 159 | 37 | 30 | 9 | 0 | 0 | 1 | 120 | 3 | 1 |
| CFD02 | 178 | 48 | 41 | 10 | 0 | 3 | 0 | 127 | 7 | 3 |
| CFD03 | 165 | 105 | 65 | 6 | 1 | 0 | 1 | 94 | 40 | 26 |
| CFD04 | 84 | 28 | 18 | 10 | 0 | 2 | 3 | 56 | 5 | 1 |
| CFD06 | 576 | 214 | 138 | 18 | 0 | 3 | 0 | 420 | 85 | 65 |
| CFD07 | 185 | 101 | 88 | 10 | 0 | 4 | 1 | 87 | 12 | 11 |
| CFD08 | 255 | 157 | 141 | 10 | 0 | 5 | 2 | 104 | 19 | 16 |
| CFD09 | 100 | 36 | 27 | 5 | 0 | 2 | 2 | 68 | 8 | 7 |
| ZOI01 | 418 | 202 | 177 | 32 | 4 | 10 | 8 | 209 | 33 | 17 |
| ZOI02 | 279 | 134 | 122 | 17 | 0 | 9 | 3 | 140 | 20 | 13 |
| ZOI03 | 380 | 142 | 125 | 41 | 1 | 23 | 11 | 214 | 25 | 16 |
| ZOI04 | 2,344 | 0 | 0 | 0 | 0 | 0 | 0 | 2,344 | 0 | 0 |
| ZOI05 | 235 | 122 | 108 | 9 | 0 | 3 | 3 | 118 | 14 | 7 |
| ZOI06 | 175 | 89 | 31 | 1 | 0 | 1 | 0 | 143 | 58 | 46 |
| ZOI07 | 296 | 114 | 95 | 12 | 0 | 5 | 2 | 189 | 27 | 23 |
| ZOI08 | 0 | 49 | — | — | — | — | — | 0 | 49 | 39 |
| ZOI09 | 92 | 17 | 9 | 6 | 0 | 2 | 1 | 77 | 5 | 2 |
| ZOI10 | 148 | 58 | 46 | 6 | 0 | 2 | 1 | 96 | 11 | 6 |
| ZOI11 | 266 | 0 | 0 | 0 | 0 | 0 | 0 | 266 | 0 | 0 |

**Key findings:**
- 1,653 Stage 2 events, 6,335 missed steps
- ZOI04 and ZOI11 have zero Stage 2 events (no lines applied)
- 565 master-listener opposite-sign matches (ZOI08 Stage 2 events vs. missed steps on listeners within 180 s)
- Many `both_disagree` cases have size differences near 0.5 s, 1.0 s, or 2.0 s — suggesting half-second, one-second, or two-second measurement errors

---

## 8. Task 6: Example Fish

### Inventory

| Source | Files |
|---|---|
| Arm A positions | 4 CSVs (1D5A: 0.7 MB, 26E4: 0.3 MB, 9F8C: 0.1 MB, B5ED: 0.5 MB) |
| Diagnostic Arm B | No position CSVs (database only) |
| What-if B1 | No position CSVs (databases only) |

**Limitation:** Only Arm A has position CSVs. Per the plan: "If only one arm has position files, STOP and report." Proceeding with Arm A only.

### Selected jumps

| # | Rec_ID | jump_time_s | offset_s | status | tag | fixes |
|---|---|---|---|---|---|---|
| 1 | ZOI03 | 1751068620 | -0.598 | passed | 26E4 | 100 |
| 2 | ZOI01 | 1750266066 | 0.377 | unverifiable | 1D5A | 30 |
| 3 | ZOI03 | 1750950695 | 1.000 | unverifiable | B5ED | 47 |

### Plots

| File | Positions | Near jump |
|---|---|---|
| `26E4_ZOI03_jump.png` | 1,607 | 100 |
| `1D5A_ZOI01_jump.png` | 345 | 30 |
| `B5ED_ZOI03_jump.png` | 109 | 47 |

Plots saved to `output/11112026_stage1_closure/audit/plots/`.

---

## 9. Task 7: Validation Drags / Static Holds Scoping

**Conclusion:** A truth-based check of jump correction is not possible with the current data. The `zoi11_test` databases (Arm A and Arm B) do not exist on disk. No test window databases are available.

---

## 10. Deviations

1. **Disk space:** Started at 2.26 GB (below 3 GB threshold). User freed Downloads before continuing.
2. **Task 6:** Only Arm A position files available (4 tags). Plan requires both arms. Proceeded with Arm A only.
3. **Task 7:** zoi11_test databases do not exist on disk. Cannot perform scoping analysis.
4. **Task 4:** Only -1 flags found (no +1 flags) in the three receivers. The cross-table reading was adjusted accordingly.

---

## 11. Things That Looked Wrong

1. **ZOI04 unexplained = 519.75 s:** This is extreme. ZOI04 has 2,344 missed steps and 0 applied corrections. The `level_anchored_within_50ms%` is 97.52%, meaning the level-anchored approach would fix this. But the raw series returns to baseline while the corrected series drifts — this is the core problem the plan was designed to investigate.

2. **CFD02 unexplained = 11.74 s** and **CFD07 unexplained = 6.44 s** and **CFD09 unexplained = 6.94 s:** These receivers have large unexplained terms despite having applied corrections. The phantom term is near zero, size error is small. The missed steps term is large but doesn't fully account for the corrected_level_change.

3. **CFD06 unexplained = -2.31 s:** Negative unexplained means the decomposition over-counts. The missed steps term is only +0.01 s but corrected_change is -2.30 s. This suggests the decomposition model doesn't close for this receiver.

4. **CFD03 unexplained = -0.19 s:** Nearly closes but slightly over-counts.

5. **Table H master candidates:** 120+ common-mode events at ±1 s and ±2 s affecting 3–17 receivers simultaneously. These are almost certainly master clock (ZOI08) one-second adjustments that Stage 1 never detected.

---

## 12. Open Questions for the Reviewer

1. **ZOI04 and ZOI11 have zero lines but 2,344 and 266 steps respectively.** Why did Stage 1 produce zero corrections for these receivers? Is this a parsing issue, a receiver-specific configuration issue, or are these receivers genuinely different?

2. **The unexplained term is large for 12 of 18 receivers.** The decomposition formula is `corrected_change = missed + size_error + phantom + unexplained`. For receivers where unexplained is large and positive, the missed steps term under-counts. For CFD06, it over-counts. What could cause this?

3. **Table H shows 120+ common-mode events at ±1 s.** These are master clock adjustments. Should Stage 1 be looking for these specifically?

4. **The level-anchored approach puts 97–100% of epochs within 50 ms for every receiver.** Is this a viable alternative to cumulative offset correction?

5. **Why are there zero +1 flags in any of the three receivers?** The ATS guide says `>` means plus one second. Are plus-one-second adjustments not present in the data, or is the flag encoding different?

---

*Report complete. Journal proof follows.*
