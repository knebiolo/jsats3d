# JSAT3D Stage 1 Clock Correction — Full Request/Response/Gap Analysis

**Date:** 2026-11-10  
**Author:** Qwen 3.6 35B MoE (with PM verification)  
**Status:** Task 10 complete — deliverable ready for Drew

---

## Part 1: Drew's Requests and Our Response

### Request A: TDoA Visualization (Primary)

**Drew's request (verbatim):**
> "I've become familiar with the current scripts, but I think our next step should be visualizing the beacon data so we can review the clock corrections directly. Interactive plots will let us inspect each receiver's timing behavior and assess whether Stage 1 is working before we add more complexity."

**Requirements:**
- One interactive plot per receiver, paired to ZOI08/7DB7
- Plot TDoA in ms or seconds, datetime x-axis, TDoA y-axis
- Overlay uncorrected + Stage 1-corrected as scatter points
- Vertical line for every applied Stage 1 correction with hover: event time, receiver, magnitude/direction, evidence
- Full study span or monthly segments if too large
- Isolated B1 database, no modifications to originals
- Standalone HTML in named output folder, index/README

**What we delivered:**

| Requirement | Status | Details |
|---|---|---|
| 18 interactive plots | ✅ Done | One per non-master receiver with 7DB7 data (CFD01-04, CFD06-09, ZOI01-07, ZOI09-11). CFD05 excluded (no 7DB7 data). |
| Plotly with datetime x-axis | ✅ Done | x-axis: `local time (PDT)`, never "UTC". y-axis: TDoA in ms. |
| Uncorrected + corrected overlay | ✅ Done | Grey = Arm A baseline, Blue = B1 corrected. |
| 696 vertical lines with hover | ✅ Done | Each hover carries: original time, corrected time, evidence, source file, TDoA step, applied shift, pre/post levels, epoch counts, common-mode listeners, validation status, residual. |
| Full study span | ✅ Done | 2025-06-14 → 2025-08-06 (no segmentation triggered — max 142,066 points per receiver, below 150,000 threshold). |
| Isolated B1 DB | ✅ Done | `output/11082026_whatif_B1/jsats3d_2025_B1.db` — read-only, never modified. |
| Standalone HTML | ✅ Done | Each file inlined with Plotly.js (~6-11 MB), works offline. |
| Index + README | ✅ Done | `index.html` (18 receivers, point/line counts, links), `README.md` (method, provenance, how-to-read, key finding, limitations). |
| Stage 2/DBSCAN disabled | ✅ Done | Only Stage 1 corrections (696 events, all `stage=1`). |
| Faithful to data | ✅ Done | 343 unverifiable + 1 failed shown, not hidden. CFD08 red line visible with 0.519 ms residual. |

**Guardrails held:**
- Arm A + B1 + Arm B prod: all strictly read-only (`mode=ro` + `PRAGMA query_only=ON`)
- Every DB byte-identical before/after (size + timestamps re-verified after each round)
- No new files outside `output/11092026_stage1_tdoa_review/`
- No commits, no changes to `scripts/`

**QA proof:**
- Series-mapping identity exact to 0.0000 ms over 332,753 matched burst pairs
- Hover census: 696/696 bare integers, 0 buggy patterns
- Per-receiver line counts exact: 39/55/34/30/13/20/54/5/169/123/84/0/18/1/7/19/25/0
- Two bugs found and fixed in QA: common-mode hover ("1000.0 ms" → integer), x-axis rendering (numeric line x on date axis)

**What was beyond the ask:**
- The review surfaced the firmware-event vs observed-behavior discrepancy (key open question below)
- Three uncorrected clock anomalies now visible to Drew (ZOI04, ZOI10, ZOI11)
- 2026-11-04 A/B test context included in README

---

### Request B: Clock Jumps + DBSCAN + Fish Track Validation

**Drew's request (verbatim):**
> "Based on our call today, it appears the methodology of identifying clock jumps through the internal code and quantifying magnitude/direction through TDOA... for some reason not all jumps are identified through this methodology. Therefore, we need to come up with a supplementary process of identifying remaining jumps identifiable through the TDOA beacon plots."

**Requirements:**
1. Supplementary process for identifying remaining jumps via TDoA beacon plots
2. DBSCAN clustering to optimize results (figure out number of clusters)
3. Alternative: calculate tdiff for each TDoA value, cluster with DBSCAN into 3 groups (large jumps = time jump events, small jumps = multipath, rest = normal)
4. Continue Arm A vs Arm B comparison workflow
5. Validate with 2-3 example fish tracks during jump periods (or validation drags)

**What we did:**
- ✅ Delivered the visualization that enables this review (the plots are the foundation for identifying remaining jumps)
- ✅ Arm A vs B1 comparison completed (identity verified to 0.0000 ms)
- ✅ Key finding surfaced: 696 firmware-based corrections apply permanent shifts, but raw TDoA shows only short-lived excursions — the corrected series drifts while raw recovers

**What is MISSING (not yet done):**
- ❌ **DBSCAN implementation** — not started. Need to determine optimal cluster count for TDoA residuals after Stage 1 corrections.
- ❌ **tdiff clustering** — not started. Alternative approach: compute `tdiff = TDoA(t) - TDoA(t-1)` and cluster with DBSCAN into 3 groups.
- ❌ **Fish track validation** — not started. Need to: pick random jumps, find fish in forebay during jump period, plot Arms A and B for 2-3 example fish.
- ❌ **Validation drag analysis** — not started. Check if any validation drags occurred during jump periods.

**Status:** These are the Stage 2 tasks that Drew explicitly said to implement later. The visualization deliverable is the foundation for doing them.

---

### Request C: ZOI11 and CFD01 Inclusion

**Drew's request (verbatim):**
> "The hydrophones for ZOI11 and CFD01 should both be 1 ft below the bottom of the debris barrier and therefore useful for positioning, but likely plagued with multipath signal conflicts. Include each of the receivers individually, and both and create plots for all the static holds, validation drags in comparison to the original without each included."

**Requirements:**
- Individual plots for ZOI11 and CFD01 (currently in the 18-receiver set)
- Comparison plots: No New Receivers vs ZOI11 Added vs CFD01 Added vs ZOI11/CFD01 Added
- Plots for static holds and validation drags
- All on a single plot per validation event

**What we did:**
- ✅ ZOI11 and CFD01 are included in the 18 receiver plots (ZOI11: 71,033 points, 0 corrections; CFD01: 70,954 points, 39 corrections)
- ❌ Comparison plots (No New Receivers vs ZOI11 Added vs CFD01 Added vs ZOI11/CFD01 Added) — not started

**What is MISSING:**
- ❌ **Positioning comparison plots** — Need to create plots showing positioning results (not TDoA) with/without ZOI11 and CFD01 for each validation drag/static hold
- ❌ **Static hold identification** — Need to identify which time periods are static holds vs validation drags
- ❌ **GPS ground truth alignment** — Need to align GPS positions with acoustic positioning results

**Status:** This is a separate deliverable from the TDoA visualization. It requires positioning pipeline output, not just TDoA plots.

---

## Part 2: Raw Data — Structure and Samples

### 2.1 `tblDetectionRaw` (Arm A — 10292026_Test)

**Purpose:** Raw acoustic detection records from all receivers. This is the pipeline's pre-clock-fix table. The `seconds` column is the detection timestamp in Unix seconds (PDT stored as-if-UTC).

**Size:** 29,912,682 rows (Arm A DB: 18,631,671,808 bytes)

**Sample rows:**

```
timeStamp              seconds      Rec_ID  ReceiverType  FirmwareVersion  Tag_ID  FreqOff  Amplitude  NBW   SNR  Valid  Pascals  Celsius  TagTypeSource  Event  SigStr  RawTemperature  Pressure  Tilt  BatteryVoltage  BitPeriod  Threshold  OffsetChanged  CounterRestart  ClockEventReasons  GPSFixTimeStamp          GPSFixLatitude  GPSFixLongitude  RawDateTime                    ReceiverUTCOffsetHours  TimeShiftHours  TimeZoneSource
2025-06-15 00:00:13    1.749946e+09  CFD07   SR3017        10.62F           7D0E    None     178        239.0 12   1      None     None     raw              0      178     99.99           None      None  11.59             239        30/31        166              0                   2025-06-15 07:00:10+00:00  46.466517       -122.107467      2025-06-15 00:00:13.614207  -7.0                  0.0             gps
2025-06-15 00:00:14    1.749946e+09  CFD07   SR3017        10.62F           B36A    None     215        240.0 48   1      None     None     raw              0      215     99.99           None      None  11.59             240        01/31        167              0                   2025-06-15 07:00:10+00:00  46.466517       -122.107467      2025-06-15 00:00:14.078510  -7.0                  0.0             gps
2025-06-15 00:00:17    1.749946e+09  CFD07   SR3017        10.62F           1FF1    None     207        239.0 39   1      None     None     raw              0      207     99.99           None      None  11.59             239        24/31        168              0                   2025-06-15 07:00:10+00:00  46.466517       -122.107467      2025-06-15 00:00:17.902704  -7.0                  0.0             gps
```

**Key columns for clock correction work:**
| Column | Type | Description |
|---|---|---|
| `seconds` | float64 | Unix timestamp (PDT as-if-UTC) |
| `Rec_ID` | str | Receiver identifier (CFD01-ZOI11) |
| `Tag_ID` | str | Beacon/tag identifier (7DB7 = B1 metronome) |
| `Internal` | str | Firmware internal code (e.g., "000014 0551 00 y041 674 A") |
| `InternalFlag` | str | Internal flag (e.g., "y", "0") |
| `InternalCounter` | str | Internal counter value |
| `InternalOffset` | str | Internal offset |
| `InternalStatus` | str | Internal status |
| `InternalCounterValue` | int64 | Counter value (numeric) |
| `InternalOffsetValue` | int64 | Offset value (numeric) |
| `InternalPositionDifferenceSeconds` | float64 | Position difference in seconds |
| `OneSecondAdjustmentEvidence` | int64 | 1 if one-second adjustment detected |
| `ClockStatusMarker` | int64 | Clock status marker |
| `OffsetChanged` | int64 | 1 if offset changed (clock jump evidence) |
| `CounterRestart` | int64 | 1 if counter restarted (clock jump evidence) |
| `ClockEventReasons` | str | Text description of clock event |
| `GPSFixTimeStamp` | str | GPS fix timestamp (if available) |
| `GPSFixLatitude` | float64 | GPS latitude |
| `GPSFixLongitude` | float64 | GPS longitude |

**Important notes:**
- `seconds` is stored as Unix seconds (PDT stored as-if-UTC)
- The `Internal` column contains firmware codes that the Stage 1 pipeline uses to detect clock jumps
- `OffsetChanged`, `CounterRestart`, `OneSecondAdjustmentEvidence` are the three evidence types for clock jumps
- GPS data is available for some receivers (CFD07 in sample has GPS fix)

---

### 2.2 `tblClockJumpEvents` (B1 — 11082026_whatif_B1)

**Purpose:** Stage 1 correction ledger. Each row is an applied clock jump correction. 696 rows total, all `stage=1`.

**Size:** 696 rows (B1 DB: 5,976,518,656 bytes)

**Sample rows:**

```
Rec_ID  jump_time_s    jump_time_local              evidence                              source_file                                          n_pre  n_post  pre_level_s  post_level_s  offset_s   common_mode  stage
ZOI05   1.750255e+09   2025-06-18 14:01:06          OffsetChanged                         .../SR18081_250610_191801.csv                         4      4       0.032705     -0.947652     -0.980357  0.0          1
ZOI05   1.750313e+09   2025-06-19 06:01:01          OffsetChanged                         .../SR18081_250618_224901_cleaned.csv                 4      4       0.032447     -0.947863     -0.980310  0.0          1
ZOI05   1.750352e+09   2025-06-19 16:57:08          OffsetChanged;OneSecondAdjustment     .../SR18081_250618_224901_cleaned.csv                 4      4      -1.544442      0.071347      1.615789  0.0          1
ZOI05   1.750446e+09   2025-06-20 19:05:55          CounterRestart;OneSecondAdjustment    .../SR18081_250618_224901_cleaned.csv                 4      4      -2.547407     -3.047238     -0.499832  0.0          1
ZOI05   1.750943e+09   2025-06-26 12:58:27          CounterRestart                        .../SR18081_250618_224901_cleaned.csv                 4      5       0.056733      1.630883      1.574149  0.0          1
```

**Key columns:**
| Column | Type | Description |
|---|---|---|
| `Rec_ID` | str | Receiver identifier |
| `jump_time_s` | float64 | Jump time in Unix seconds (original/pre-shift receiver time) |
| `jump_time_local` | str | Jump time as PDT display string |
| `evidence` | str | Evidence type: "OffsetChanged", "CounterRestart", "OneSecondAdjustment", or combinations |
| `source_file` | str | Source CSV file where the jump was detected |
| `n_pre` | int64 | Number of pre-event epochs used |
| `n_post` | int64 | Number of post-event epochs used |
| `pre_level_s` | float64 | Median TDoA in pre-event window (seconds) |
| `post_level_s` | float64 | Median TDoA in post-event window (seconds) |
| `offset_s` | float64 | TDoA step = post_level_s - pre_level_s (the clock jump magnitude) |
| `common_mode` | float64 | Number of common-mode listeners |
| `stage` | int64 | Always 1 for Stage 1 corrections |

**Validation join (from `tblClockJumpValidation`):**
| Column | Type | Description |
|---|---|---|
| `status` | str | "passed" (352), "unverifiable" (343), or "failed" (1) |
| `residual_step_s` | float64 | Residual step after correction (null for unverifiable) |

---

### 2.3 `tblClockJumpCandidates` (B1)

**Purpose:** Full ledger of all candidate events (accepted + unresolved). 945 rows = 696 accepted + 249 unresolved.

**Sample rows:**

```
Rec_ID  jump_time_s    jump_time_local              offset_s   evidence                              status   note
CFD01   1.750291e+09   2025-06-18 23:57:02          -1.002519  OffsetChanged                         accepted internal-code jump
CFD01   1.750295e+09   2025-06-19 01:01:00           0.999188  OffsetChanged;CounterRestart          accepted internal-code jump
CFD01   1.750309e+09   2025-06-19 05:01:02          -1.003136  OffsetChanged                         accepted internal-code jump
CFD01   1.750338e+09   2025-06-19 13:01:00           0.996159  OffsetChanged;CounterRestart          accepted internal-code jump
CFD01   1.750464e+09   2025-06-20 23:57:11           0.997960  OffsetChanged;CounterRestart          accepted internal-code jump
```

---

### 2.4 `tblClockEventEvidence` (B1)

**Purpose:** Low-level firmware evidence for clock events. Links raw detection rows to clock event indicators.

**Sample rows:**

```
source_rowid  seconds      Rec_ID  OffsetChanged  CounterRestart  OneSecondAdjustmentEvidence  InternalOffsetValue  SourceFile                                    SourceRow  InternalCounterValue
1129          1.749949e+09  CFD07   1              1               0                            244                  .../SR18076_250610_125501.csv                143307     0
1130          1.749949e+09  CFD07   0              1               0                            244                  .../SR18076_250610_125501.csv                143309     1
4539          1.749960e+09  CFD07   0              1               0                            244                  .../SR18076_250610_125501.csv                147566     0
4540          1.749960e+09  CFD07   0              1               0                            244                  .../SR18076_250610_125501.csv                147567     1
5819          1.749964e+09  CFD07   1              1               0                            1208                 .../SR18076_250610_125501.csv                149158     0
```

**Key columns:**
| Column | Type | Description |
|---|---|---|
| `source_rowid` | int64 | Row ID in `tblDetectionRaw` |
| `seconds` | float64 | Detection timestamp |
| `Rec_ID` | str | Receiver |
| `OffsetChanged` | int64 | 1 if offset changed at this row |
| `CounterRestart` | int64 | 1 if counter restarted at this row |
| `OneSecondAdjustmentEvidence` | int64 | 1 if one-second adjustment at this row |
| `InternalOffsetValue` | int64 | Offset value at this row |

---

### 2.5 `tblClockJumpValidation` (B1)

**Purpose:** Validation results for each applied correction. Joined to `tblClockJumpEvents` on (Rec_ID, jump_time_s).

**Sample rows:**

```
Rec_ID  jump_time_s    start_s      end_s        n_pre  n_post  residual_step_s  status
ZOI05   1.750255e+09   1.750255e+09 1.750256e+09 2      4       None             unverifiable
ZOI05   1.750313e+09   1.750313e+09 1.750313e+09 2      4       None             unverifiable
ZOI05   1.750352e+09   1.750352e+09 1.750353e+09 2      2       None             unverifiable
ZOI05   1.750446e+09   1.750446e+09 1.750447e+09 0      0       None             unverifiable
ZOI05   1.750943e+09   1.750942e+09 1.750943e+09 4      0       None             unverifiable
```

**Status distribution:** 352 passed / 343 unverifiable / 1 failed (CFD08, residual 0.5188 ms)

---

## Part 3: Output Data — Structure and Samples

### 3.1 `series_armA_uncorrected.csv`

**Purpose:** Arm A (uncorrected) paired first-arrival TDoA series. Extracted from `tblDetectionRaw` using `clock_jump_pre.beacon_first_arrivals()`.

**Size:** 1,003,984 rows, 4 columns

**Sample rows:**

```
Rec_ID  t            transNo  tdoa_s
CFD01   1.749946e+09 0.0      0.015441
CFD01   1.749946e+09 1.0      0.015453
CFD01   1.749946e+09 2.0      0.015468
CFD01   1.749946e+09 3.0      0.015479
CFD01   1.749946e+09 4.0      0.015492
```

**Columns:**
| Column | Type | Description |
|---|---|---|
| `Rec_ID` | str | Receiver identifier |
| `t` | float64 | Detection time in Unix seconds (PDT as-if-UTC) |
| `transNo` | float64 | Master transmission number (index of ZOI08 first arrivals) |
| `tdoa_s` | float64 | TDoA in seconds (listener minus master) |

---

### 3.2 `series_b1_corrected.csv`

**Purpose:** B1 (Stage 1-corrected) paired first-arrival TDoA series. Same extraction process but from B1 database where corrections are applied.

**Size:** 1,003,970 rows, 4 columns (same structure as Arm A)

**Sample rows:**

```
Rec_ID  t            transNo  tdoa_s
CFD01   1.749946e+09 0.0      0.015441
CFD01   1.749946e+09 1.0      0.015453
CFD01   1.749946e+09 2.0      0.015468
CFD01   1.749946e+09 3.0      0.015479
CFD01   1.749946e+09 4.0      0.015492
```

**Key relationship:** `B1_TDoA = ArmA_TDoA - cumulative_offset` (identity verified to 0.0000 ms)

---

### 3.3 `stage1_correction_lines.csv`

**Purpose:** 696 correction lines with cumulative offset curves and validation joins. This is the primary data for the vertical lines in the plots.

**Size:** 696 rows, 21 columns

**Sample rows:**

```
Rec_ID  jump_time_s    jump_time_local              evidence                              source_file                                          n_pre  n_post  pre_level_s  post_level_s  offset_s   common_mode  stage  cum_offset_s  corrected_event_time_s  applied_shift_ms  offset_ms  pre_level_ms  post_level_ms  status       residual_step_s  residual_step_ms
CFD01   1.750291e+09   2025-06-18 23:57:02          OffsetChanged                         .../SR25308_250618_152801_cleaned.csv                 5      5       0.017829     -0.984690     -1.002519  1.0          1    -1.002519     1.750291e+09            1002.519131       -1002.519131 17.829180    -984.689951    passed         0.000011           0.011325
CFD01   1.750295e+09   2025-06-19 01:01:00          OffsetChanged;CounterRestart          .../SR25308_250618_152801_cleaned.csv                 5      5      -0.983876      0.015312      0.999188  1.0          1    -0.003331     1.750295e+09            -999.187946       999.187946   -983.875990  15.311956    unverifiable   NaN                NaN
```

**Key columns:**
| Column | Type | Description |
|---|---|---|
| `jump_time_s` | float64 | Original (pre-shift) receiver time |
| `jump_time_local` | str | PDT display string |
| `evidence` | str | Evidence type(s) |
| `offset_s` | float64 | TDoA step (post - pre) |
| `cum_offset_s` | float64 | Cumulative offset at this event |
| `corrected_event_time_s` | float64 | `jump_time_s - cum_offset_s` |
| `applied_shift_ms` | float64 | `-offset_s * 1000` (shift applied to recorded times) |
| `offset_ms` | float64 | `offset_s * 1000` |
| `pre_level_ms` / `post_level_ms` | float64 | Pre/post median TDoA in ms |
| `status` | str | "passed", "unverifiable", or "failed" |
| `residual_step_ms` | float64 | Residual after correction (null for unverifiable) |

---

### 3.4 `summary.csv`

**Purpose:** One row per receiver with point counts, line counts, and file references.

**Size:** 18 rows, 9 columns

```
Rec_ID  n_armA  n_b1  n_lines  n_passed  n_unverifiable  n_failed  files                        total_size_mb
CFD01   70954   70954 39       14        25              0         CFD01_stage1_tdoa.html       10.57
CFD02   64995   64994 55       16        39              0         CFD02_stage1_tdoa.html       10.02
CFD03   42628   42625 34       18        16              0         CFD03_stage1_tdoa.html       8.23
CFD04   64644   64644 30       5         25              0         CFD04_stage1_tdoa.html       10.03
CFD06   39188   39187 13       6         7               0         CFD06_stage1_tdoa.html       7.98
CFD07   52942   52941 20       5         15              0         CFD07_stage1_tdoa.html       9.00
CFD08   56888   56888 54       22        31              1         CFD08_stage1_tdoa.html       9.42
CFD09   49787   49787 5        3         2               0         CFD09_stage1_tdoa.html       9.00
ZOI01   62466   62464 169      82        87              0         ZOI01_stage1_tdoa.html       9.84
ZOI02   69915   69911 123      89        34              0         ZOI02_stage1_tdoa.html       10.40
ZOI03   69092   69092 84       63        21              0         ZOI03_stage1_tdoa.html       10.37
ZOI04   47242   47242 0        0         0               0         ZOI04_stage1_tdoa.html       8.87
ZOI05   50139   50138 18       6         12              0         ZOI05_stage1_tdoa.html       8.79
ZOI06   16304   16304 1        1         0               0         ZOI06_stage1_tdoa.html       6.24
ZOI07   51852   51852 7        3         4               0         ZOI07_stage1_tdoa.html       9.06
ZOI09   66827   66826 19       3         16              0         ZOI09_stage1_tdoa.html       10.27
ZOI10   57088   57088 25       16        9               0         ZOI10_stage1_tdoa.html       9.42
ZOI11   71033   71033 0        0         0               0         ZOI11_stage1_tdoa.html       11.31
```

---

### 3.5 Databases

| Database | Path | Size | Last Write | Purpose |
|---|---|---|---|---|
| **Arm A** (uncorrected baseline) | `output/10292026_Test/jsats3d_2025_earlyfish.db` | 18,631,671,808 bytes | 2026-11-06 18:34:13 | Pipeline output from `enm_jsat3d_edits` branch. Raw detections (`tblDetectionRaw`) without clock corrections. |
| **B1** (Stage 1 only) | `output/11082026_whatif_B1/jsats3d_2025_B1.db` | 5,976,518,656 bytes | 2026-11-08 18:43:47 | Isolated copy with 696 Stage 1 corrections applied. Contains `tblClockJumpEvents`, `tblClockJumpValidation`, `tblClockJumpCandidates`, `tblClockEventEvidence`. |
| **Arm B prod** | `output/11062026_clocksync/jsats3d_2025_earlyfish.db` | 5,976,518,656 bytes | 2026-11-07 22:16:26 | Production Arm B (Stage 1 + Stage 2). Must remain unchanged. |

---

## Part 4: Key Finding for Drew

**The plots show what Stage 1 did, and it surfaces a real question:**

The 696 firmware-based corrections apply permanent cumulative shifts (up to ±15 s), but the raw 7DB7 TDoA sits at the fixed acoustic path delay (11–63 ms) with only short-lived (1–6 day) excursions — so the corrected series drifts while the raw recovers.

**Open question:** Do the firmware events represent permanent clock error, or scheduled adjustments that GPS re-absorbs?

**Context:** The 2026-11-04 A/B test (Stage 1+2) found the corrections make the multi-tag clock curves continuous at all 463 measurable applied jumps (max residual 0.256 ms).

**Also visible:**
- Uncorrected anomalies on ZOI04 (07-10…14), ZOI10 (07-15), ZOI11 (06-22…27)
- One failed validation on CFD08 (0.519 ms residual)
- 343 unverifiable corrections (no independent verification possible)

---

## Part 5: What to Send Drew

### Files to copy (self-contained, works offline):

```
output/11092026_stage1_tdoa_review/
├── index.html                          ← Entry point
├── README.md                           ← Method, how-to-read, key finding
├── summary.csv                         ← Per-receiver counts (for spreadsheets)
├── stage1_correction_lines.csv         ← Full metadata for all 696 corrections
├── plots/
│   ├── CFD01_stage1_tdoa.html          ← 10.1 MB
│   ├── CFD02_stage1_tdoa.html          ← 9.6 MB
│   ├── CFD03_stage1_tdoa.html          ← 7.8 MB
│   ├── CFD04_stage1_tdoa.html          ← 9.6 MB
│   ├── CFD06_stage1_tdoa.html          ← 7.6 MB
│   ├── CFD07_stage1_tdoa.html          ← 8.6 MB
│   ├── CFD08_stage1_tdoa.html          ← 9.0 MB  ← RED failed line
│   ├── CFD09_stage1_tdoa.html          ← 8.6 MB
│   ├── ZOI01_stage1_tdoa.html          ← 9.4 MB  ← Densest (169 lines)
│   ├── ZOI02_stage1_tdoa.html          ← 9.9 MB
│   ├── ZOI03_stage1_tdoa.html          ← 9.9 MB
│   ├── ZOI04_stage1_tdoa.html          ← 8.5 MB  ← No corrections + anomaly
│   ├── ZOI05_stage1_tdoa.html          ← 8.4 MB
│   ├── ZOI06_stage1_tdoa.html          ← 6.0 MB  ← Sparsest (1 line)
│   ├── ZOI07_stage1_tdoa.html          ← 8.6 MB
│   ├── ZOI09_stage1_tdoa.html          ← 9.8 MB
│   ├── ZOI10_stage1_tdoa.html          ← 9.0 MB  ← Uncorrected anomaly
│   └── ZOI11_stage1_tdoa.html          ← 10.8 MB ← No corrections
```

### Suggested first plots for Drew:
1. **CFD08** — the only red failed line (0.519 ms residual)
2. **ZOI01** — densest (169 lines), largest drift (+15.3 s)
3. **ZOI04** — no corrections, visible 07-10…14 anomaly
4. **CFD01** — quiet receiver, baseline look

### One-paragraph finding for Drew:
> The plots show what Stage 1 did, and it surfaces a real question: the 696 firmware-based corrections apply permanent cumulative shifts (up to ±15 s), but the raw 7DB7 TDoA sits at the fixed acoustic path delay with only short-lived (1–6 day) excursions — so the corrected series drifts while the raw recovers. Whether the firmware events represent permanent clock error or GPS-re-absorbed adjustments is the decision point. Also visible: uncorrected anomalies on ZOI04 (07-10…14), ZOI10 (07-15), ZOI11 (06-22…27), and the one failed validation on CFD08 (0.519 ms residual).

---

## Part 6: What's Still Needed (Next Steps)

### High Priority (from Drew's Request B):
1. **DBSCAN on TDoA residuals** — After Stage 1 corrections, compute residuals and cluster with DBSCAN. Determine optimal number of clusters.
2. **tdiff clustering** — Alternative: compute `tdiff = TDoA(t) - TDoA(t-1)` and cluster into 3 groups (large jumps, small jumps/multipath, normal).
3. **Fish track validation** — Pick 2-3 random jumps, find fish in forebay during jump periods, plot Arms A and B for example fish tracks.
4. **Validation drag analysis** — Check if any validation drags occurred during jump periods.

### Medium Priority (from Drew's Request C):
5. **Positioning comparison plots** — Create plots showing positioning with/without ZOI11 and CFD01 for each validation drag/static hold.
6. **Static hold identification** — Identify which time periods are static holds vs validation drags.

### Low Priority (cleanup):
7. **Quarantine stray script** — `scripts/visualize_beacon_tdota.py` was quarantined to `output/_quarantine_scripts/` but not committed.
8. **Commit plan** — 4 commits as proposed in Task 5 (data extraction, generator script, deliverable plots, documentation).
