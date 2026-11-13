# Plan C4 event list

Tool version fourarm-2026-11-12-a. Times are study wall clock (PDT stored as if UTC).

- **solution row**: a root B row of `tblPositions_Deng` with comment `solution found`. One row per receiver combination. It rises with every receiver added. It is not a count of fixes.
- **fix**: for a static hold, a transmission kept by the project's receiver set consensus (two or more receiver sets within 10 m). For a drag, a point of the project's consensus, smoothing and Kalman track.
- **error**: XY distance from a fix to GPS truth.
- **nearer receiver**: the nearer of ZOI11 and CFD01, in XY.

Data window of arm N: 2025-06-10 05:43:25 to 2025-06-13 22:44:29. Workbook sheets: Upstream_StaticHolds, Forebay_StaticHolds, CFNSC_StaticHolds, Forebay_Approach_Drags, CFNSC_Approach_Drags, Forebay_Random_Drags, CFNSC_Drags.

`trans_<tag>` is the number of transmissions of the tag inside the event (arm N). `<receiver>_heard_<tag>` is the number of detections of the tag at that receiver inside the event (arm ZC, `tblDetectionFilterSecondary`).

| event | kind | date | start | end | gps_truth | in_window | scored | note | trans_FC36 | ZOI11_heard_FC36 | CFD01_heard_FC36 | trans_FFD3 | ZOI11_heard_FFD3 | CFD01_heard_FFD3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FBY-S-1 | forebay hold | 2025-06-10 | 09:58:02 | 09:59:02 | yes | yes | yes |  | 22 | 37 | 20 | 20 | 17 | 5 |
| FBY-S-2 | forebay hold | 2025-06-10 | 10:00:22 | 10:01:22 | yes | yes | yes |  | 22 | 39 | 29 | 20 | 20 | 3 |
| FBY-S-3 | forebay hold | 2025-06-10 | 10:02:57 | 10:03:57 | yes | yes | yes |  | 22 | 72 | 17 | 20 | 1 | 0 |
| FBY-S-4 | forebay hold | 2025-06-10 | 10:05:28 | 10:06:32 | yes | yes | yes |  | 23 | 32 | 25 | 21 | 0 | 0 |
| FBY-S-5 | forebay hold | 2025-06-10 | 10:08:24 | 10:09:28 | yes | yes | yes |  | 23 | 23 | 25 | 21 | 21 | 1 |
| FBY-S-6 | forebay hold |  |  |  | no | no | no | times unusable: Unknown datetime string format, unable to parse: NA | 0 | 0 | 0 | 0 | 0 | 0 |
| FBY-S-7 | forebay hold | 2025-06-10 | 10:10:45 | 10:11:45 | yes | yes | yes |  | 22 | 36 | 15 | 20 | 0 | 18 |
| FBY-S-8 | forebay hold | 2025-06-10 | 10:13:09 | 10:14:04 | yes | yes | yes |  | 20 | 31 | 49 | 19 | 0 | 19 |
| FBY-S-9 | forebay hold | 2025-06-10 | 10:15:52 | 10:16:52 | yes | yes | yes |  | 22 | 19 | 3 | 20 | 18 | 11 |
| FBY-S-10 | forebay hold | 2025-06-10 | 10:19:02 | 10:20:02 | yes | yes | yes |  | 22 | 35 | 12 | 19 | 29 | 13 |
| DB-1 | debris barrier hold | 2025-06-11 | 12:14:12 | 12:15:12 | no | yes | yes |  | 21 | 102 | 22 | 21 | 98 | 35 |
| DB-2 | debris barrier hold | 2025-06-11 | 12:02:36 | 12:03:36 | no | yes | yes |  | 22 | 58 | 18 | 20 | 82 | 15 |
| DB-3 | debris barrier hold | 2025-06-11 | 12:04:15 | 12:05:15 | no | yes | yes |  | 22 | 96 | 24 | 20 | 78 | 50 |
| DB-4 | debris barrier hold | 2025-06-11 | 12:06:26 | 12:07:26 | no | yes | yes |  | 22 | 97 | 22 | 20 | 67 | 19 |
| DB-5 | debris barrier hold | 2025-06-11 | 12:09:45 | 12:10:45 | no | yes | yes |  | 22 | 64 | 35 | 20 | 61 | 32 |
| ENT-01 | drag | 2025-06-10 | 10:24:13 | 10:26:55 | yes | yes | yes |  | 55 | 58 | 46 | 50 | 36 | 3 |
| ENT-02 | drag | 2025-06-10 | 10:30:17 | 10:32:34 | yes | yes | yes |  | 47 | 61 | 40 | 43 | 42 | 6 |
| ENT-03 | drag | 2025-06-10 | 10:36:33 | 10:39:50 | yes | yes | yes |  | 66 | 106 | 58 | 61 | 33 | 14 |
| ENT-04 | drag | 2025-06-10 | 10:43:27 | 10:47:24 | yes | yes | yes |  | 80 | 130 | 64 | 73 | 14 | 21 |
| ENT-05 | drag | 2025-06-10 | 10:50:28 | 10:54:53 | yes | yes | yes |  | 89 | 147 | 87 | 81 | 23 | 28 |
| FBY-D-1 | drag | 2025-06-10 | 10:58:46 | 11:05:25 | yes | yes | yes |  | 133 | 203 | 114 | 122 | 61 | 62 |
| FBY-D-2 | drag | 2025-06-10 | 11:05:28 | 11:14:35 | yes | yes | yes |  | 182 | 235 | 172 | 166 | 62 | 65 |
| FBY-D-3 | drag | 2025-06-10 | 11:14:47 | 11:31:51 | yes | yes | yes |  | 339 | 591 | 316 | 309 | 158 | 72 |
| FBY-D-4 | drag | 2025-06-10 | 11:31:56 | 11:45:53 | yes | yes | yes |  | 277 | 396 | 211 | 249 | 169 | 132 |
| FBY-NS-01 | drag | 2025-06-05 | 11:50:21 | 11:52:46 | yes | no | no |  | 0 | 0 | 0 | 0 | 0 | 0 |
| FBY-NS-02 | drag | 2025-06-05 | 11:55:57 | 11:58:46 | yes | no | no |  | 0 | 0 | 0 | 0 | 0 | 0 |
| FBY-SN-01 | drag | 2025-06-05 | 11:53:10 | 11:55:53 | yes | no | no |  | 0 | 0 | 0 | 0 | 0 | 0 |
| FBY-SN-02 | drag | 2025-06-05 | 11:59:09 | 12:01:45 | yes | no | no |  | 0 | 0 | 0 | 0 | 0 | 0 |

## Counts

| kind | scored | events |
|---|---|---|
| debris barrier hold | yes | 5 |
| drag | no | 4 |
| drag | yes | 9 |
| forebay hold | no | 1 |
| forebay hold | yes | 9 |

Events that are not scored, with the reason:

- FBY-S-6 (forebay hold): times unusable: Unknown datetime string format, unable to parse: NA
- FBY-NS-01 (drag): outside the data window
- FBY-NS-02 (drag): outside the data window
- FBY-SN-01 (drag): outside the data window
- FBY-SN-02 (drag): outside the data window
