# Follow-up: what Stage 1 decided next to each missed step

Tool version followup-2026-11-11-a. Times are study wall clock (PDT stored as if UTC). 1 ms of timing error is about 1 m of position error.

## Inputs

stage1_accepted must be 696. If it is not, the Stage 1 report run used different settings.

| missed_steps | stage1_accepted | stage1_evidence_only | stage1_unresolvable |
|---|---|---|---|
| 6335 | 696 | 4979 | 16198 |

## Table P. Missed steps by Stage 1 disposition, per receiver

| Rec_ID | accepted | evidence_only | no_stage1_event | unresolvable |
|---|---|---|---|---|
| CFD01 | 10 | 1 | 119 | 29 |
| CFD02 | 10 | 1 | 135 | 32 |
| CFD03 | 7 | 0 | 36 | 122 |
| CFD04 | 8 | 0 | 50 | 26 |
| CFD06 | 2 | 6 | 338 | 230 |
| CFD07 | 0 | 1 | 54 | 130 |
| CFD08 | 4 | 2 | 70 | 179 |
| CFD09 | 0 | 4 | 47 | 49 |
| ZOI01 | 21 | 7 | 190 | 200 |
| ZOI02 | 11 | 10 | 113 | 145 |
| ZOI03 | 20 | 9 | 280 | 71 |
| ZOI04 | 0 | 0 | 2344 | 0 |
| ZOI05 | 8 | 6 | 124 | 97 |
| ZOI06 | 0 | 0 | 11 | 164 |
| ZOI07 | 6 | 0 | 182 | 108 |
| ZOI09 | 7 | 0 | 66 | 19 |
| ZOI10 | 6 | 3 | 88 | 51 |
| ZOI11 | 0 | 0 | 266 | 0 |

## Table Q. Reasons

| disposition | detail | count |
|---|---|---|
| no_stage1_event |  | 4513 |
| unresolvable | insufficient beacon epochs on one side (file gap?) | 987 |
| unresolvable | window clipped below 150 s by a neighbouring event | 665 |
| accepted | an applied line is within 300 s but belongs to another step | 120 |
| evidence_only | measured step below 50 ms | 44 |
| evidence_only | measured step above the 5 s maximum | 6 |

