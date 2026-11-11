# Plan B2 Report: Stage 1 Follow-up Audit

**Date:** 2026-11-12  
**Executor:** Qwen 3.6 35B MoE  
**Plan:** docs/llm_handoff_PLAN_B2_followup_2026-11-11.md

---

## 1. Run Facts

| Item | Value |
|---|---|
| Git head | ad82ca4 Add Stage 1 follow-up audit docs |
| Branch | ENM_jsats3d_edits_clocksync |
| Test suite | 74 passed, 2 subtests passed in 14.45s |
| Free disk start | 11.56 GB |
| Free disk end | 11.55 GB |

### Database table (start)

| Database | Path | Size (bytes) | Last Write |
|---|---|---|---|
| Arm A | `output\10292026_Test\jsats3d_2025_earlyfish.db` | 18,631,671,808 | 2026-11-06 18:34 |
| B1 | `output\11082026_whatif_B1\jsats3d_2025_B1.db` | 5,976,518,656 | 2026-11-08 18:43 |
| Production Arm B | `output\11062026_clocksync\jsats3d_2025_earlyfish.db` | 5,976,518,656 | 2026-11-07 22:16 |

### Database table (end)

| Database | Path | Size (bytes) | Last Write |
|---|---|---|---|
| Arm A | `output\10292026_Test\jsats3d_2025_earlyfish.db` | 18,631,671,808 | 2026-11-06 18:34 |
| B1 | `output\11082026_whatif_B1\jsats3d_2025_B1.db` | 5,976,518,656 | 2026-11-08 18:43 |
| Production Arm B | `output\11062026_clocksync\jsats3d_2025_earlyfish.db` | 5,976,518,656 | 2026-11-07 22:16 |

---

## 2. Self-tests

### stage1_closure_audit.py
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

### stage1_followup_audit.py
```
signed classes add up                      OK
five steps before                          OK
one step after flags removed               OK
all epochs joined                          OK
21 listener and 31 master flags            OK
flagged epochs explained                   OK
four missed steps near one second          OK
four dispositions                          OK
one unresolvable and one evidence only     OK
two with no stage 1 event                  OK
SELFTEST PASSED
```

---

## 3. Files Copied into `docs/reports/`

| File | Size (bytes) |
|---|---|
| `PLAN_B_closure_report_full_2026-11-11.md` | 24,954 |
| `PLAN_B2_signed_2026-11-12.md` | (copied from followup_signed.md) |
| `PLAN_B2_flags_2026-11-12.md` | (copied from followup_flags.md) |
| `PLAN_B2_dispositions_2026-11-12.md` | (copied from followup_dispositions.md) |

---

## 4. Checks

### Task 1 (signed)
| Check | Value |
|---|---|
| Receivers in Table J | 18 ✓ |
| total_s == last_minus_first_s (all 18) | ✓ |

### Task 2 (flags)
| Check | Value |
|---|---|
| min_joined_pct | 100.0% ✓ |
| steps_before | 7,024 ✓ |
| Table N receivers | 18 ✓ |

### Task 3 (dispositions)
| Check | Value |
|---|---|
| Jumps accepted | 696 ✓ |
| Arm A size after | 18,631,671,808 (unchanged) ✓ |
| Arm A last write | 2026-11-06 18:34 (unchanged) ✓ |

---

## 5. Sentences

### Task 1 (signed)
1. **Table J:** For every receiver, the `large_spike_beyond_reflection_s` column cancels the `sustained_step_s` column. For CFD01: +5.417 vs −5.551; for ZOI04: −580.292 vs +580.789. For CFD03 and ZOI09 the cancellation is shared across `normal_s`, `small_multipath_or_jitter_s`, and `large_spike_beyond_reflection_s`.
2. **Table K:** Down steps and up steps nearly balance for every receiver (e.g. ZOI01: +324.433 up, −347.109 down; ZOI04: +1,017.000 up, −1,536.745 down). The net `sum_measured_s` is small relative to the gross sums.
3. **Table L and L2:** Of the 6,335 steps without a line, 2,781 are near one second (43.9%) and 86 are near two seconds (1.4%). Per receiver, 19–86% of missed steps are near one or two seconds.
4. **Table M:** A typical regime away from the path delay lasts 4–5 minutes median. Most receivers spend 5–31% of epochs off baseline.

### Task 2 (flags)
The per record flags account for part of the steps: 7024 to 4885 (30.5% fewer). No plus one second flag occurs in the 7DB7 rows of any receiver. From Table O, predicted shift = 0: 41,104 epochs at observed −1 s and 25,519 at observed +1 s are whole-second offsets no flag explains.

### Task 3 (dispositions)
4,513 of 6,335 missed steps (71.2%) have no Stage 1 event at all within 300 s — jumps the internal codes never announced. 1,333 had an event Stage 1 could not measure; two reasons dominate: "insufficient beacon epochs on one side (file gap?)" (987) and "window clipped below 150 s by a neighbouring event" (665). 50 had an event measured below 50 ms (44) or above 5 s (6). ZOI04 and ZOI11 have zero Stage 1 events because they were not in the receiver list — all 2,344 + 266 steps read no_stage1_event. That is why they have zero lines.

---

## 6. Task 4 Listings

### Position files
| Folder | File | Size (MB) | Last Write |
|---|---|---|---|
| 10292026_Test/figures | 1D5A_positions.csv | 0.7 | 2026-10-29 |
| 10292026_Test/figures | 26E4_positions.csv | 0.3 | 2026-10-29 |
| 10292026_Test/figures | 9F8C_positions.csv | 0.0 | 2026-10-29 |
| 10292026_Test/figures | B5ED_positions.csv | 0.5 | 2026-10-29 |
| 11072026_diag_armB/figures | 1D5A_positions.csv | 0.8 | 2026-11-07 |
| 11072026_diag_armB/figures | 26E4_positions.csv | 0.3 | 2026-11-07 |
| 11072026_diag_armB/figures | 9F8C_positions.csv | 0.0 | 2026-11-07 |
| 11072026_diag_armB/figures | B5ED_positions.csv | 0.4 | 2026-11-07 |
| 11082026_whatif_B1/B3_diagnostic_figures | 1D5A_positions.csv | 0.7 | 2026-11-08 |
| 11082026_whatif_B1/B3_diagnostic_figures | 26E4_positions.csv | 0.3 | 2026-11-08 |
| 11082026_whatif_B1/B3_diagnostic_figures | 9F8C_positions.csv | 0.0 | 2026-11-08 |
| 11082026_whatif_B1/B3_diagnostic_figures | B5ED_positions.csv | 0.5 | 2026-11-08 |

### Databases with tblPositions_Deng
| Database | Rows |
|---|---|
| jsats3d_2025_armB_diag.db | 606,760 |
| jsats3d_2025_B3_diagnostic.db | 610,190 |

### Databases without tblPositions_Deng
| Database | Status |
|---|---|
| jsats3d_2025_B1.db | No tblPositions_Deng |
| jsats3d_2025_B2_multibeacon_trial.db | No tblPositions_Deng |

### output_archive_* folders
None found.

---

## 7. Deviations
none

---

## 8. Things That Looked Wrong

1. **30.5% steps removed by flags is significant but not dominant.** The hypothesis (section 1.4) that master clock flags account for the common-mode steps is partially supported — the flags remove nearly a third of sustained steps — but 69.5% remain unexplained by per-record flags.

2. **Table O predicted = −1: only 104 of 4,428 epochs at observed −1.** The remaining 3,461 are at "other" (beyond ±2 s) and 512 at observed +0. This suggests the flags explain a small fraction of the −1 predicted epochs directly, but the step detection is sensitive to the flag adjustment.

3. **16,198 unresolvable events** is an enormous number — 2.5× the 6,979 accepted + evidence-only events combined. The two dominant reasons (file gap, window clipped) suggest Stage 1's regime detection frequently fails to produce a measurable step even when internal codes fire.

4. **ZOI04 and ZOI11 not in the Stage 1 receiver list** explains their zero lines. This was a configuration issue, not a data issue. ZOI04 has 2,344 missed steps and ZOI11 has 266 — together they account for 39.8% of all missed steps.

---

## 9. Open Questions

1. **The 30.5% steps-removed figure:** If the flags account for 30.5% of steps, and the flags are mostly minus-one-second adjustments, what accounts for the remaining 69.5%? Are they master clock adjustments without per-record flags? Or are they something else entirely?

2. **16,198 unresolvable events:** Why are so many events unresolvable? "Insufficient beacon epochs on one side (file gap?)" suggests data quality issues. "Window clipped below 150 s" suggests events are too close together. Should these be investigated?

3. **ZOI04 and ZOI11 configuration:** Why were these receivers excluded from the Stage 1 receiver list? If they were supposed to be included, this is a configuration bug that accounts for 39.8% of all missed steps.

4. **Table L2 shows 19–86% of missed steps are near one or two seconds.** For CFD06 (19.2%) and ZOI06 (26.4%), the majority are NOT near one or two seconds. What are those steps?

5. **The per-record flag test used B1 database.** The first run's Task 3 read the wrong database (production Arm B instead of B1). This follow-up used the correct database. Were the B1 results materially different from what the production Arm B would have shown?

---

*Report complete.*
