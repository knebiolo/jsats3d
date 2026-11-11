# Reviewer synthesis of the Stage 1 audits (2026-11-12)

**Author:** external reviewer. Code and committed reports only. No access to data or outputs.
**Sources:** `PLAN_B_closure_report_full_2026-11-11.md`, `PLAN_B2_signed_2026-11-12.md`, `PLAN_B2_flags_2026-11-12.md`, `PLAN_B2_dispositions_2026-11-12.md`, `PLAN_B2_report_2026-11-12.md`. Every number below names its table.

Basis for all numbers: beacon 7DB7 at master ZOI08, first arrival per burst, 2025-06-15 to 2025-08-06, study wall clock (PDT stored as if UTC). 1 ms of timing error is about 1 m of position error.

---

## 1. What is established

### 1.1 Stage 1 is right where it acts, and it acts on few jumps

| Fact | Number | Table |
|---|---|---|
| Stage 1 lines that sit on a raw step of the same size | 671 of 696 | closure Totals |
| "Passed" lines confirmed | 352 of 352 | closure D |
| Problem lines (18 wrong size, 7 with no step) | all 25 are "unverifiable" | closure D |
| Sustained raw steps, all 18 listeners | 7,024 | closure Totals |
| Sustained raw steps, the 16 listeners Stage 1 was run on | 4,414 | closure B, less ZOI04 and ZOI11 |
| Share of those steps that carry a Stage 1 line | 671 of 4,414, about 15% | |

ZOI04 and ZOI11 are not in the run file's receiver lists, so Stage 1 never looked at them. That is a run setting, not a fault. Their 2,610 steps must not be counted as Stage 1 misses. On the 16 receivers Stage 1 covered, **3,725 steps have no line**.

### 1.2 Why Stage 1 misses them

Stage 1's own report mode, run read only on Arm A, reproduced the 696 accepted jumps and showed what it did with every internal code event:

| Stage 1 outcome for an internal code event | Count | Source |
|---|---|---|
| Accepted as a jump | 696 | dispositions Inputs |
| Measured, no step of 50 ms to 5 s | 4,979 | dispositions Inputs |
| Could not be measured | 16,198 | dispositions Inputs |

So about 3% of internal code events become corrections. For the 3,725 missed steps on the 16 covered receivers (dispositions P and Q, less ZOI04 and ZOI11):

| Reason | Steps | Share |
|---|---|---|
| No internal code event within 300 s | 1,903 | 51% |
| An event exists but Stage 1 could not measure it | 1,652 | 44% |
| An applied line is nearby but belongs to another step | 120 | 3% |
| An event exists, measured under 50 ms or over 5 s | 50 | 1% |

The two "could not measure" reasons are "insufficient beacon epochs on one side" (987) and "window clipped below 150 s by a neighbouring event" (665).

The reason is visible in signed Table M: a regime away from the path delay lasts a median of 4 to 12 beacon epochs on 15 of 18 receivers, about 3 to 16 minutes. Stage 1 needs 150 to 300 s of quiet on **each** side of an event, with 3 to 4 epochs per side. The measuring window is as long as the thing being measured. Half of the remaining jumps have no code at all.

**Plain statement for the PM:** the internal codes are neither necessary nor sufficient. Most code events are not measurable jumps, and half of the unmeasured jumps have no code.

### 1.3 What the jumps look like

- Size: of the steps without a line, 2,781 of 6,335 are within a quarter second of one second, and 86 near two seconds (signed L). Per receiver the share near one or two seconds is 19% to 86% (signed L2).
- A second family sits 0.6 s away from whole seconds: shared steps of 0.61, 1.60, 2.59, 3.61, 4.61 s (closure H), and the most common regime levels are 0.5, 1.5, 2.5, 3.5 s after rounding to the half second (signed M2). What produces 0.6 s is not known.
- Duration: minutes, not days (signed M). Receivers spend 5% to 31% of epochs away from the path delay.
- Many steps are shared. Closure Table H lists steps of the same size at the same time on up to 17 listeners, often as a plus one second followed by a minus one second a few minutes later. A change seen by every listener at once belongs to the master clock, ZOI08.
- Up and down steps nearly cancel (signed K). The level leaves and comes back.

### 1.4 The per record flag

| Fact | Number | Table |
|---|---|---|
| Listener epochs carrying the minus one second flag | 4,487 | flags Totals |
| Master epochs carrying it | 2,716 | flags Totals |
| Plus one second flags | 0 | flags Totals |
| Sustained steps after removing the flagged second in memory | 7,024 to 4,885, 30.5% fewer | flags Totals |
| Share of epochs within 50 ms of the path delay, before and after | unchanged on every receiver | flags N |
| Flagged listener epochs that sit more than 2.5 s from the path delay | 3,461 of 4,428 | flags O |
| Unflagged epochs at minus one second / plus one second | 41,104 / 25,519 | flags O, row "predicted 0" |

Reading: the flag marks records inside periods when the clock is already seconds off. Removing it merges levels inside those periods, which is why the step count falls. It does not bring anything back to the path delay, and it does not explain the one second regimes.

---

## 2. What was wrong

### 2.1 Reviewer's errors

1. **"Master flags explain the shared plus one second steps" is not supported.** Flags O: of 2,657 epochs where the master record is flagged, 35 sit at plus one second. This hypothesis is withdrawn.
2. **The drift decomposition (closure Table C) does not close** and should not be quoted. Signed Table J shows why: the level leaves in sustained steps and returns in excursions shorter than five epochs, which the first audit classed as spikes. Signed Table K shows the second reason: 9% to 78% of steps per receiver had too few epochs to measure their size.
3. **Closure Table A is not evidence of harm.** It measures absolute level. The later synchronisation removes a constant offset.

### 2.2 Executor's errors in the B2 report

1. "ZOI04 and ZOI11 ... this is a configuration bug that accounts for 39.8% of all missed steps." Not a bug. They are outside the run's receiver lists by the PM's earlier instruction. Totals for Stage 1 must use the 16 covered receivers.
2. "16,198 unresolvable events ... suggest Stage 1's regime detection frequently fails." Stage 1 has no regime detection. It measures a median on each side of an event. The count says the events come too close together for that window.
3. Section 3 of the B2 report lists three copied files without sizes.

The B2 run otherwise followed the plan: all checks reported, files copied unedited, databases unchanged, no deviations.

---

## 3. What this means for Drew's Request B

| Drew asked | What the numbers support |
|---|---|
| A supplementary process for the jumps the codes miss | Needed. On the covered receivers the codes yield a correction for about 15% of sustained steps. |
| DBSCAN, and how many clusters | Stage 2 already does this on regime levels. It agrees with the raw step for 1,261 of its 1,604 listener events and finds no event for roughly 2,260 further steps on the covered receivers (first report, Task 5). Both stages add measured steps together, so one missed or mis-sized step shifts everything after it. With several thousand steps per season that is fragile. |
| tdiff into three groups | By size the data shows: under 0.5 ms; 0.5 ms to 250 ms; and a whole second family centred on 1,000 ms (closure F and G). Size alone cannot tell a lasting change from a one epoch excursion: 7,024 lasting edges against 7,492 short ones above 250 ms (closure E). Persistence must be part of the rule. |
| Arm A against Arm B, fish examples | Position files now confirmed for Arm A and for two diagnostic arms (B2 report section 6). Plan B3 Task 3. |

### One approach to evaluate. Not implemented. Owner decision.

The beacon gives the clock state directly once a minute: raw TDoA minus the path delay. Each interval between two beacon epochs is then one of: clean, steadily offset by a known amount, ambiguous (the two ends differ), or a beacon gap. Nothing is added up, so an error cannot carry forward. The legacy clock fix already works from per epoch knots in this way; the open point is only what to do with detections inside an ambiguous interval.

Two things must be measured before this is worth proposing. Plan B3 measures both:

1. **Cost.** How much time, and how many fish detections, fall in ambiguous intervals.
2. **Whether the offset is a clock state at all.** If a receiver's clock is one second off for five minutes, a second, independent beacon heard by the same receiver must show the same second. If it does not, the offset belongs to single records and no beacon based correction can fix fish detections.

---

## 4. Still unknown

- What the 0.6 s family is.
- Why ZOI04 behaves differently: 2,344 steps, median regime of one epoch, 1,818 steps too short to size, and a net of minus 520 s in measured steps (signed K, M). It is outside the 3D receiver set.
- Whether the shared steps come from ZOI08's clock or from its reception of its own beacon.
- Anything about position accuracy. None of this has been tested against truth. The 2025 test tag data predates the window, and the small test databases are no longer on disk.
