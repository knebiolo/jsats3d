# Follow-up: per-record one second flags on listener and master

Tool version followup-2026-11-11-a. Times are study wall clock (PDT stored as if UTC). 1 ms of timing error is about 1 m of position error.

Database time basis: shifted (jump_offset_s added back). Master receiver: ZOI08.

predicted shift = listener flag minus master flag. A flagged listener record should sit one second early (TDoA minus one). A flagged master record should make every listener's TDoA plus one.

## Totals

| steps_before | steps_after | steps_removed_pct | listener_flagged | master_flagged | plus_flags | min_joined_pct |
|---|---|---|---|---|---|---|
| 7024 | 4885 | 30.5 | 4487 | 2716 | 0 | 100.0 |

## Table N. Steps before and after removing the flagged adjustment in memory

| Rec_ID | epochs | joined_pct | listener_flagged | master_flagged | plus_flags | steps_before | steps_after | steps_without_line_before | within_50ms_before_pct | within_50ms_after_pct | flagged_epochs_explained_pct | offset_epochs_flagged_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CFD01 | 70954 | 100.00 | 149 | 197 | 0 | 198 | 66 | 159 | 93.46 | 93.42 | 1.45 | 0.13 |
| CFD02 | 64995 | 100.00 | 282 | 184 | 0 | 233 | 85 | 178 | 89.36 | 89.36 | 0.00 | 0.00 |
| CFD03 | 42628 | 100.00 | 6 | 109 | 0 | 199 | 156 | 165 | 87.72 | 87.71 | 0.00 | 0.00 |
| CFD04 | 64644 | 100.00 | 0 | 166 | 0 | 114 | 53 | 84 | 93.12 | 93.12 | 0.00 | 0.00 |
| CFD06 | 39188 | 100.00 | 38 | 106 | 0 | 589 | 551 | 576 | 83.23 | 83.23 | 0.00 | 0.00 |
| CFD07 | 52942 | 100.00 | 84 | 131 | 0 | 205 | 133 | 185 | 88.17 | 88.21 | 9.39 | 0.46 |
| CFD08 | 56888 | 100.00 | 345 | 139 | 0 | 309 | 228 | 255 | 83.39 | 82.93 | 1.24 | 0.13 |
| CFD09 | 49787 | 100.00 | 72 | 129 | 0 | 105 | 51 | 100 | 92.58 | 92.63 | 14.51 | 0.94 |
| ZOI01 | 62466 | 100.00 | 650 | 167 | 0 | 583 | 351 | 418 | 68.57 | 68.59 | 3.58 | 0.32 |
| ZOI02 | 69915 | 100.00 | 357 | 184 | 0 | 402 | 265 | 279 | 76.84 | 76.78 | 0.19 | 0.01 |
| ZOI03 | 69092 | 100.00 | 938 | 184 | 0 | 462 | 140 | 380 | 79.45 | 79.49 | 1.98 | 0.24 |
| ZOI04 | 47242 | 100.00 | 294 | 138 | 0 | 2344 | 2206 | 2344 | 71.60 | 71.63 | 4.86 | 0.58 |
| ZOI05 | 50139 | 100.00 | 247 | 156 | 0 | 253 | 131 | 235 | 83.77 | 83.74 | 0.25 | 0.03 |
| ZOI06 | 16304 | 100.00 | 2 | 27 | 0 | 176 | 165 | 175 | 87.43 | 87.43 | 0.00 | 0.00 |
| ZOI07 | 51852 | 100.00 | 481 | 142 | 0 | 303 | 88 | 296 | 89.10 | 88.99 | 0.17 | 0.03 |
| ZOI09 | 66827 | 100.00 | 0 | 189 | 0 | 110 | 40 | 92 | 95.13 | 95.13 | 0.00 | 0.00 |
| ZOI10 | 57088 | 100.00 | 149 | 178 | 0 | 173 | 74 | 148 | 88.62 | 88.62 | 0.92 | 0.06 |
| ZOI11 | 71033 | 100.00 | 393 | 190 | 0 | 266 | 102 | 266 | 88.79 | 88.67 | 0.36 | 0.05 |

## Table O. Predicted shift against observed whole-second offset from the path delay (epoch counts)

| predicted_shift_s | observed -2 s | observed -1 s | observed +0 s | observed +1 s | observed +2 s | observed other |
|---|---|---|---|---|---|---|
| -2 | 0 | 0 | 0 | 0 | 0 | 0 |
| -1 | 348 | 104 | 512 | 3 | 0 | 3461 |
| 0 | 6759 | 41104 | 898492 | 25519 | 4945 | 20080 |
| 1 | 5 | 6 | 128 | 35 | 191 | 2292 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 |

