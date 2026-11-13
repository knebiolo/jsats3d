# Plan C4 tables

Tool version fourarm-2026-11-12-a. Times are study wall clock (PDT stored as if UTC).

- **solution row**: a root B row of `tblPositions_Deng` with comment `solution found`. One row per receiver combination. It rises with every receiver added. It is not a count of fixes.
- **fix**: for a static hold, a transmission kept by the project's receiver set consensus (two or more receiver sets within 10 m). For a drag, a point of the project's consensus, smoothing and Kalman track.
- **error**: XY distance from a fix to GPS truth.
- **nearer receiver**: the nearer of ZOI11 and CFD01, in XY.

## V4a. Per event, tag and arm

`share_rows_added` is the percent of the arm's solution rows in the event that contain an added receiver. `det_ZOI11` and `det_CFD01` are detections of the tag at that receiver in the event. Events without GPS truth have no error columns.

| event | kind | tag | arm_name | solution_rows | trans_with_solution | fixes | std_x | std_y | std_z | median_z | median_err | p90_err | share_rows_added | det_ZOI11 | det_CFD01 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FBY-S-1 | forebay hold | FC36 | 1. Excluded | 781 | 21 | 21 | 1.48 | 2.14 | 1.78 | -5.42 | 2.89 | 6.09 |  | 37 | 20 |
| FBY-S-1 | forebay hold | FC36 | 2. ZOI11 Only | 1,282 | 21 | 21 | 1.84 | 1.47 | 1.91 | -6.77 | 4.13 | 6.29 | 39.08 | 37 | 20 |
| FBY-S-1 | forebay hold | FC36 | 3. CFD01 Only | 1,168 | 21 | 21 | 1.46 | 1.05 | 2.03 | -6.12 | 3.57 | 5.70 | 33.13 | 37 | 20 |
| FBY-S-1 | forebay hold | FC36 | 4. Both | 1,840 | 21 | 21 | 1.79 | 1.19 | 1.69 | -7.20 | 4.13 | 6.20 | 57.55 | 37 | 20 |
| FBY-S-1 | forebay hold | FFD3 | 1. Excluded | 609 | 19 | 19 | 1.16 | 2.39 | 2.78 | -0.84 | 3.13 | 5.10 |  | 17 | 5 |
| FBY-S-1 | forebay hold | FFD3 | 2. ZOI11 Only | 1,084 | 19 | 19 | 1.14 | 2.10 | 2.72 | -3.72 | 2.93 | 4.72 | 43.82 | 17 | 5 |
| FBY-S-1 | forebay hold | FFD3 | 3. CFD01 Only | 659 | 19 | 19 | 1.03 | 2.33 | 2.89 | -0.84 | 3.13 | 4.62 | 7.59 | 17 | 5 |
| FBY-S-1 | forebay hold | FFD3 | 4. Both | 1,154 | 19 | 19 | 1.14 | 2.04 | 2.79 | -3.72 | 2.72 | 4.72 | 47.23 | 17 | 5 |
| FBY-S-2 | forebay hold | FC36 | 1. Excluded | 478 | 21 | 21 | 2.94 | 0.70 | 3.76 | 3.56 | 7.21 | 9.74 |  | 39 | 29 |
| FBY-S-2 | forebay hold | FC36 | 2. ZOI11 Only | 678 | 21 | 21 | 3.68 | 0.67 | 3.43 | 4.80 | 6.96 | 9.71 | 29.50 | 39 | 29 |
| FBY-S-2 | forebay hold | FC36 | 3. CFD01 Only | 668 | 21 | 21 | 3.27 | 0.80 | 3.46 | 5.48 | 6.23 | 9.65 | 28.44 | 39 | 29 |
| FBY-S-2 | forebay hold | FC36 | 4. Both | 975 | 21 | 21 | 3.97 | 0.96 | 3.41 | 2.93 | 6.01 | 9.58 | 50.97 | 39 | 29 |
| FBY-S-2 | forebay hold | FFD3 | 1. Excluded | 203 | 13 | 13 | 3.31 | 0.69 | 9.50 | -7.87 | 7.77 | 12.10 |  | 20 | 3 |
| FBY-S-2 | forebay hold | FFD3 | 2. ZOI11 Only | 328 | 13 | 13 | 3.32 | 0.60 | 9.10 | -5.31 | 6.79 | 11.75 | 38.11 | 20 | 3 |
| FBY-S-2 | forebay hold | FFD3 | 3. CFD01 Only | 207 | 13 | 13 | 3.25 | 0.70 | 9.49 | -7.87 | 7.77 | 11.94 | 1.93 | 20 | 3 |
| FBY-S-2 | forebay hold | FFD3 | 4. Both | 339 | 13 | 13 | 3.25 | 0.64 | 9.07 | -5.31 | 6.79 | 11.30 | 40.12 | 20 | 3 |
| FBY-S-3 | forebay hold | FC36 | 1. Excluded | 734 | 20 | 20 | 2.18 | 1.20 | 1.69 | -5.12 | 2.13 | 3.69 |  | 72 | 17 |
| FBY-S-3 | forebay hold | FC36 | 2. ZOI11 Only | 1,267 | 20 | 20 | 2.28 | 1.24 | 1.94 | -5.97 | 2.12 | 4.45 | 42.07 | 72 | 17 |
| FBY-S-3 | forebay hold | FC36 | 3. CFD01 Only | 1,136 | 20 | 20 | 2.18 | 1.01 | 1.87 | -5.89 | 2.05 | 3.82 | 35.39 | 72 | 17 |
| FBY-S-3 | forebay hold | FC36 | 4. Both | 1,875 | 20 | 20 | 2.39 | 1.02 | 1.88 | -6.51 | 2.03 | 4.60 | 60.85 | 72 | 17 |
| FBY-S-3 | forebay hold | FFD3 | 1. Excluded | 298 | 19 | 17 | 2.21 | 1.02 | 1.82 | -3.82 | 2.18 | 3.72 |  | 1 | 0 |
| FBY-S-3 | forebay hold | FFD3 | 2. ZOI11 Only | 298 | 19 | 17 | 2.21 | 1.02 | 1.82 | -3.82 | 2.18 | 3.72 | 0.00 | 1 | 0 |
| FBY-S-3 | forebay hold | FFD3 | 3. CFD01 Only | 298 | 19 | 17 | 2.21 | 1.02 | 1.82 | -3.82 | 2.18 | 3.72 | 0.00 | 1 | 0 |
| FBY-S-3 | forebay hold | FFD3 | 4. Both | 298 | 19 | 17 | 2.21 | 1.02 | 1.82 | -3.82 | 2.18 | 3.72 | 0.00 | 1 | 0 |
| FBY-S-4 | forebay hold | FC36 | 1. Excluded | 617 | 18 | 16 | 0.68 | 1.62 | 1.58 | 0.03 | 1.30 | 3.08 |  | 32 | 25 |
| FBY-S-4 | forebay hold | FC36 | 2. ZOI11 Only | 1,052 | 18 | 16 | 0.52 | 1.43 | 1.66 | -0.47 | 1.15 | 2.68 | 41.35 | 32 | 25 |
| FBY-S-4 | forebay hold | FC36 | 3. CFD01 Only | 1,100 | 18 | 17 | 0.53 | 1.08 | 1.59 | -0.51 | 0.87 | 2.10 | 43.91 | 32 | 25 |
| FBY-S-4 | forebay hold | FC36 | 4. Both | 1,773 | 18 | 17 | 0.49 | 1.06 | 1.90 | -1.45 | 0.81 | 2.07 | 65.20 | 32 | 25 |
| FBY-S-4 | forebay hold | FFD3 | 1. Excluded | 492 | 20 | 19 | 1.38 | 5.63 | 2.47 | 2.06 | 3.72 | 8.01 |  | 0 | 0 |
| FBY-S-4 | forebay hold | FFD3 | 2. ZOI11 Only | 492 | 20 | 19 | 1.38 | 5.63 | 2.47 | 2.06 | 3.72 | 8.01 | 0.00 | 0 | 0 |
| FBY-S-4 | forebay hold | FFD3 | 3. CFD01 Only | 492 | 20 | 19 | 1.38 | 5.63 | 2.47 | 2.06 | 3.72 | 8.01 | 0.00 | 0 | 0 |
| FBY-S-4 | forebay hold | FFD3 | 4. Both | 492 | 20 | 19 | 1.38 | 5.63 | 2.47 | 2.06 | 3.72 | 8.01 | 0.00 | 0 | 0 |
| FBY-S-5 | forebay hold | FC36 | 1. Excluded | 602 | 22 | 22 | 6.78 | 1.29 | 12.63 | 14.05 | 9.14 | 22.87 |  | 23 | 25 |
| FBY-S-5 | forebay hold | FC36 | 2. ZOI11 Only | 897 | 22 | 22 | 6.77 | 1.85 | 12.57 | -6.45 | 7.73 | 15.00 | 32.89 | 23 | 25 |
| FBY-S-5 | forebay hold | FC36 | 3. CFD01 Only | 876 | 22 | 22 | 7.43 | 1.54 | 12.38 | -4.02 | 7.04 | 22.26 | 31.28 | 23 | 25 |
| FBY-S-5 | forebay hold | FC36 | 4. Both | 1,364 | 22 | 22 | 4.93 | 1.16 | 11.37 | -6.45 | 4.37 | 14.34 | 55.87 | 23 | 25 |
| FBY-S-5 | forebay hold | FFD3 | 1. Excluded | 291 | 19 | 16 | 10.41 | 1.98 | 16.48 | -3.03 | 9.33 | 28.43 |  | 21 | 1 |
| FBY-S-5 | forebay hold | FFD3 | 2. ZOI11 Only | 436 | 19 | 16 | 11.10 | 3.33 | 14.36 | -3.39 | 7.84 | 24.05 | 33.26 | 21 | 1 |
| FBY-S-5 | forebay hold | FFD3 | 3. CFD01 Only | 291 | 19 | 16 | 10.41 | 1.98 | 16.48 | -3.03 | 9.33 | 28.43 | 0.00 | 21 | 1 |
| FBY-S-5 | forebay hold | FFD3 | 4. Both | 436 | 19 | 16 | 11.10 | 3.33 | 14.36 | -3.39 | 7.84 | 24.05 | 33.26 | 21 | 1 |
| FBY-S-7 | forebay hold | FC36 | 1. Excluded | 428 | 19 | 19 | 3.42 | 8.43 | 9.96 | 0.81 | 5.34 | 18.06 |  | 36 | 15 |
| FBY-S-7 | forebay hold | FC36 | 2. ZOI11 Only | 760 | 19 | 19 | 2.29 | 1.89 | 5.91 | -0.12 | 3.00 | 4.28 | 43.68 | 36 | 15 |
| FBY-S-7 | forebay hold | FC36 | 3. CFD01 Only | 697 | 20 | 19 | 2.90 | 7.21 | 9.39 | -5.03 | 3.61 | 9.33 | 38.59 | 36 | 15 |
| FBY-S-7 | forebay hold | FC36 | 4. Both | 1,093 | 20 | 19 | 2.48 | 1.57 | 5.55 | -5.06 | 2.94 | 4.54 | 60.84 | 36 | 15 |
| FBY-S-7 | forebay hold | FFD3 | 1. Excluded | 212 | 14 | 12 | 3.51 | 3.09 | 5.13 | -4.45 | 4.91 | 7.16 |  | 0 | 18 |
| FBY-S-7 | forebay hold | FFD3 | 2. ZOI11 Only | 212 | 14 | 12 | 3.51 | 3.09 | 5.13 | -4.45 | 4.91 | 7.16 | 0.00 | 0 | 18 |
| FBY-S-7 | forebay hold | FFD3 | 3. CFD01 Only | 369 | 15 | 14 | 2.89 | 2.07 | 4.28 | -5.13 | 2.76 | 6.21 | 42.55 | 0 | 18 |
| FBY-S-7 | forebay hold | FFD3 | 4. Both | 369 | 15 | 14 | 2.89 | 2.07 | 4.28 | -5.13 | 2.76 | 6.21 | 42.55 | 0 | 18 |
| FBY-S-8 | forebay hold | FC36 | 1. Excluded | 632 | 18 | 18 | 2.37 | 5.09 | 3.84 | 0.40 | 9.73 | 11.88 |  | 32 | 49 |
| FBY-S-8 | forebay hold | FC36 | 2. ZOI11 Only | 1,164 | 19 | 18 | 2.27 | 2.61 | 2.69 | -0.58 | 3.40 | 5.49 | 45.70 | 31 | 49 |
| FBY-S-8 | forebay hold | FC36 | 3. CFD01 Only | 1,170 | 19 | 18 | 3.03 | 3.54 | 2.49 | 0.54 | 3.78 | 6.55 | 45.98 | 32 | 49 |
| FBY-S-8 | forebay hold | FC36 | 4. Both | 1,927 | 19 | 19 | 2.65 | 2.64 | 4.06 | -0.81 | 3.53 | 5.03 | 67.20 | 31 | 49 |
| FBY-S-8 | forebay hold | FFD3 | 1. Excluded | 20 | 9 | 3 | 1.81 | 3.97 | 13.20 | -18.33 | 24.31 | 30.94 |  | 0 | 19 |
| FBY-S-8 | forebay hold | FFD3 | 2. ZOI11 Only | 20 | 9 | 3 | 1.81 | 3.97 | 13.20 | -18.33 | 24.31 | 30.94 | 0.00 | 0 | 19 |
| FBY-S-8 | forebay hold | FFD3 | 3. CFD01 Only | 53 | 13 | 9 | 2.05 | 7.41 | 2.21 | 4.49 | 2.52 | 8.16 | 62.26 | 0 | 19 |
| FBY-S-8 | forebay hold | FFD3 | 4. Both | 53 | 13 | 9 | 2.05 | 7.41 | 2.21 | 4.49 | 2.52 | 8.16 | 62.26 | 0 | 19 |
| FBY-S-9 | forebay hold | FC36 | 1. Excluded | 63 | 14 | 4 | 3.96 | 4.18 | 12.77 | -8.64 | 4.90 | 8.05 |  | 19 | 3 |
| FBY-S-9 | forebay hold | FC36 | 2. ZOI11 Only | 151 | 20 | 11 | 12.13 | 6.81 | 14.50 | -8.11 | 7.94 | 24.78 | 58.28 | 19 | 3 |
| FBY-S-9 | forebay hold | FC36 | 3. CFD01 Only | 73 | 15 | 5 | 4.02 | 4.10 | 12.93 | -10.14 | 7.01 | 8.01 | 13.70 | 19 | 3 |
| FBY-S-9 | forebay hold | FC36 | 4. Both | 161 | 20 | 12 | 9.90 | 5.81 | 13.71 | -9.29 | 7.01 | 20.22 | 60.87 | 19 | 3 |
| FBY-S-9 | forebay hold | FFD3 | 1. Excluded | 73 | 10 | 7 | 8.57 | 8.53 | 6.63 | -4.06 | 9.04 | 29.32 |  | 17 | 11 |
| FBY-S-9 | forebay hold | FFD3 | 2. ZOI11 Only | 159 | 12 | 11 | 8.33 | 8.51 | 5.59 | -5.29 | 2.69 | 11.54 | 54.09 | 18 | 11 |
| FBY-S-9 | forebay hold | FFD3 | 3. CFD01 Only | 140 | 11 | 9 | 4.27 | 5.23 | 5.71 | -4.34 | 5.81 | 19.95 | 47.86 | 17 | 11 |
| FBY-S-9 | forebay hold | FFD3 | 4. Both | 226 | 12 | 12 | 3.03 | 2.87 | 4.96 | -5.20 | 4.31 | 7.17 | 67.70 | 18 | 11 |
| FBY-S-10 | forebay hold | FC36 | 1. Excluded | 300 | 19 | 12 | 11.32 | 19.30 | 23.40 | -4.01 | 33.11 | 47.83 |  | 34 | 12 |
| FBY-S-10 | forebay hold | FC36 | 2. ZOI11 Only | 599 | 20 | 18 | 9.22 | 9.68 | 7.13 | 1.99 | 10.00 | 28.85 | 49.92 | 35 | 12 |
| FBY-S-10 | forebay hold | FC36 | 3. CFD01 Only | 429 | 19 | 12 | 12.81 | 27.57 | 17.26 | 4.29 | 26.74 | 42.61 | 30.07 | 34 | 12 |
| FBY-S-10 | forebay hold | FC36 | 4. Both | 728 | 20 | 17 | 7.38 | 8.57 | 8.11 | 1.04 | 6.92 | 24.28 | 58.79 | 35 | 12 |
| FBY-S-10 | forebay hold | FFD3 | 1. Excluded | 62 | 17 | 11 | 8.77 | 8.94 | 20.76 | -1.57 | 58.62 | 69.36 |  | 30 | 13 |
| FBY-S-10 | forebay hold | FFD3 | 2. ZOI11 Only | 179 | 17 | 16 | 7.41 | 14.47 | 3.15 | 1.38 | 12.92 | 20.40 | 65.36 | 29 | 13 |
| FBY-S-10 | forebay hold | FFD3 | 3. CFD01 Only | 139 | 17 | 13 | 9.66 | 25.48 | 9.21 | -1.53 | 7.42 | 54.32 | 55.40 | 30 | 13 |
| FBY-S-10 | forebay hold | FFD3 | 4. Both | 256 | 17 | 17 | 7.24 | 14.68 | 3.90 | 3.26 | 9.23 | 13.24 | 75.78 | 29 | 13 |
| DB-1 | debris barrier hold | FC36 | 1. Excluded | 214 | 19 | 18 | 2.45 | 2.87 | 6.02 | -6.99 |  |  |  | 105 | 22 |
| DB-1 | debris barrier hold | FC36 | 2. ZOI11 Only | 511 | 19 | 19 | 1.23 | 1.61 | 2.40 | -9.27 |  |  | 58.12 | 102 | 22 |
| DB-1 | debris barrier hold | FC36 | 3. CFD01 Only | 524 | 19 | 19 | 1.01 | 0.78 | 3.11 | -10.43 |  |  | 59.16 | 105 | 22 |
| DB-1 | debris barrier hold | FC36 | 4. Both | 1,057 | 20 | 19 | 0.91 | 0.99 | 3.12 | -12.96 |  |  | 79.75 | 102 | 22 |
| DB-1 | debris barrier hold | FFD3 | 1. Excluded | 1,185 | 18 | 18 | 0.98 | 1.11 | 0.61 | -6.74 |  |  |  | 98 | 35 |
| DB-1 | debris barrier hold | FFD3 | 2. ZOI11 Only | 2,150 | 18 | 18 | 0.74 | 0.60 | 0.56 | -7.86 |  |  | 44.88 | 98 | 35 |
| DB-1 | debris barrier hold | FFD3 | 3. CFD01 Only | 2,149 | 18 | 18 | 0.66 | 0.44 | 0.97 | -8.50 |  |  | 44.86 | 98 | 35 |
| DB-1 | debris barrier hold | FFD3 | 4. Both | 3,538 | 18 | 18 | 0.82 | 0.42 | 1.40 | -8.89 |  |  | 66.51 | 98 | 35 |
| DB-2 | debris barrier hold | FC36 | 1. Excluded | 741 | 21 | 21 | 0.55 | 0.80 | 0.89 | -3.77 |  |  |  | 61 | 18 |
| DB-2 | debris barrier hold | FC36 | 2. ZOI11 Only | 1,462 | 21 | 21 | 0.45 | 0.65 | 0.82 | -4.81 |  |  | 49.32 | 58 | 18 |
| DB-2 | debris barrier hold | FC36 | 3. CFD01 Only | 1,300 | 21 | 21 | 0.46 | 0.71 | 0.88 | -4.84 |  |  | 43.00 | 61 | 18 |
| DB-2 | debris barrier hold | FC36 | 4. Both | 2,342 | 21 | 21 | 0.38 | 0.41 | 0.98 | -5.13 |  |  | 68.36 | 58 | 18 |
| DB-2 | debris barrier hold | FFD3 | 1. Excluded | 1,058 | 13 | 13 | 0.25 | 0.45 | 0.91 | -6.51 |  |  |  | 82 | 15 |
| DB-2 | debris barrier hold | FFD3 | 2. ZOI11 Only | 1,792 | 14 | 13 | 0.27 | 0.42 | 0.39 | -6.76 |  |  | 40.96 | 82 | 15 |
| DB-2 | debris barrier hold | FFD3 | 3. CFD01 Only | 1,490 | 13 | 13 | 0.25 | 0.39 | 0.72 | -6.71 |  |  | 28.99 | 82 | 15 |
| DB-2 | debris barrier hold | FFD3 | 4. Both | 2,436 | 14 | 13 | 0.28 | 0.16 | 0.44 | -7.10 |  |  | 56.57 | 82 | 15 |
| DB-3 | debris barrier hold | FC36 | 1. Excluded | 148 | 21 | 12 | 0.73 | 3.83 | 1.95 | -3.98 |  |  |  | 96 | 24 |
| DB-3 | debris barrier hold | FC36 | 2. ZOI11 Only | 340 | 21 | 21 | 0.58 | 0.95 | 2.25 | -3.12 |  |  | 56.47 | 96 | 24 |
| DB-3 | debris barrier hold | FC36 | 3. CFD01 Only | 341 | 21 | 19 | 1.39 | 2.74 | 4.26 | -3.13 |  |  | 56.60 | 96 | 24 |
| DB-3 | debris barrier hold | FC36 | 4. Both | 697 | 21 | 21 | 0.72 | 1.58 | 2.70 | -3.15 |  |  | 78.77 | 96 | 24 |
| DB-3 | debris barrier hold | FFD3 | 1. Excluded | 674 | 19 | 19 | 0.64 | 2.65 | 1.52 | -6.01 |  |  |  | 78 | 50 |
| DB-3 | debris barrier hold | FFD3 | 2. ZOI11 Only | 1,345 | 19 | 19 | 0.49 | 0.75 | 0.51 | -6.04 |  |  | 49.89 | 78 | 50 |
| DB-3 | debris barrier hold | FFD3 | 3. CFD01 Only | 1,237 | 19 | 19 | 0.55 | 2.43 | 1.13 | -6.03 |  |  | 45.51 | 78 | 50 |
| DB-3 | debris barrier hold | FFD3 | 4. Both | 2,208 | 19 | 19 | 0.38 | 0.62 | 0.46 | -6.04 |  |  | 69.47 | 78 | 50 |
| DB-4 | debris barrier hold | FC36 | 1. Excluded | 442 | 19 | 18 | 2.00 | 4.80 | 7.36 | -3.23 |  |  |  | 97 | 22 |
| DB-4 | debris barrier hold | FC36 | 2. ZOI11 Only | 840 | 19 | 19 | 0.29 | 1.76 | 1.79 | -3.22 |  |  | 47.38 | 97 | 22 |
| DB-4 | debris barrier hold | FC36 | 3. CFD01 Only | 564 | 19 | 18 | 1.55 | 2.93 | 3.17 | -3.01 |  |  | 21.63 | 97 | 22 |
| DB-4 | debris barrier hold | FC36 | 4. Both | 1,106 | 19 | 19 | 0.24 | 2.29 | 1.54 | -3.31 |  |  | 60.04 | 97 | 22 |
| DB-4 | debris barrier hold | FFD3 | 1. Excluded | 846 | 19 | 19 | 1.40 | 3.91 | 2.83 | -1.49 |  |  |  | 67 | 19 |
| DB-4 | debris barrier hold | FFD3 | 2. ZOI11 Only | 1,503 | 19 | 19 | 0.83 | 3.09 | 1.79 | -5.78 |  |  | 43.71 | 67 | 19 |
| DB-4 | debris barrier hold | FFD3 | 3. CFD01 Only | 1,249 | 19 | 19 | 0.91 | 6.81 | 3.62 | -6.16 |  |  | 32.27 | 67 | 19 |
| DB-4 | debris barrier hold | FFD3 | 4. Both | 2,220 | 19 | 19 | 0.76 | 4.35 | 1.68 | -6.06 |  |  | 61.89 | 67 | 19 |
| DB-5 | debris barrier hold | FC36 | 1. Excluded | 22 | 12 | 0 |  |  |  |  |  |  |  | 64 | 35 |
| DB-5 | debris barrier hold | FC36 | 2. ZOI11 Only | 53 | 15 | 6 | 4.08 | 4.42 | 11.25 | -3.95 |  |  | 58.49 | 64 | 35 |
| DB-5 | debris barrier hold | FC36 | 3. CFD01 Only | 65 | 12 | 6 | 4.37 | 9.32 | 0.37 | 3.03 |  |  | 66.15 | 64 | 35 |
| DB-5 | debris barrier hold | FC36 | 4. Both | 145 | 20 | 10 | 2.18 | 5.20 | 3.78 | -1.58 |  |  | 84.83 | 64 | 35 |
| DB-5 | debris barrier hold | FFD3 | 1. Excluded | 99 | 17 | 13 | 5.90 | 7.70 | 22.66 | -12.79 |  |  |  | 61 | 32 |
| DB-5 | debris barrier hold | FFD3 | 2. ZOI11 Only | 221 | 18 | 17 | 2.92 | 3.99 | 11.30 | -3.11 |  |  | 55.20 | 61 | 32 |
| DB-5 | debris barrier hold | FFD3 | 3. CFD01 Only | 194 | 18 | 14 | 7.41 | 19.15 | 18.88 | -6.26 |  |  | 48.97 | 61 | 32 |
| DB-5 | debris barrier hold | FFD3 | 4. Both | 369 | 18 | 16 | 5.50 | 12.02 | 4.02 | -6.90 |  |  | 73.17 | 61 | 32 |
| ENT-01 | drag | FC36 | 1. Excluded | 688 | 42 | 30 | 17.75 | 3.74 | 8.50 | -5.63 | 8.61 | 13.57 |  | 58 | 46 |
| ENT-01 | drag | FC36 | 2. ZOI11 Only | 1,018 | 49 | 30 | 17.52 | 2.71 | 8.79 | -5.97 | 7.94 | 12.64 | 32.42 | 58 | 46 |
| ENT-01 | drag | FC36 | 3. CFD01 Only | 945 | 49 | 30 | 17.41 | 3.92 | 8.55 | -6.14 | 7.92 | 12.15 | 27.20 | 58 | 46 |
| ENT-01 | drag | FC36 | 4. Both | 1,440 | 50 | 37 | 23.04 | 2.98 | 7.54 | -7.49 | 5.92 | 9.60 | 52.22 | 58 | 46 |
| ENT-01 | drag | FFD3 | 1. Excluded | 109 | 19 | 7 | 9.48 | 1.31 | 3.29 | -5.88 | 9.53 | 11.69 |  | 36 | 3 |
| ENT-01 | drag | FFD3 | 2. ZOI11 Only | 186 | 26 | 9 | 8.34 | 1.01 | 1.99 | -7.79 | 7.81 | 11.17 | 40.32 | 36 | 3 |
| ENT-01 | drag | FFD3 | 3. CFD01 Only | 112 | 19 | 7 | 9.48 | 1.31 | 3.29 | -5.88 | 9.53 | 11.69 | 2.68 | 36 | 3 |
| ENT-01 | drag | FFD3 | 4. Both | 193 | 26 | 10 | 10.74 | 1.02 | 1.94 | -8.32 | 7.52 | 11.00 | 42.49 | 36 | 3 |
| ENT-02 | drag | FC36 | 1. Excluded | 1,339 | 47 | 42 | 23.35 | 10.51 | 6.64 | -8.30 | 6.04 | 9.02 |  | 60 | 40 |
| ENT-02 | drag | FC36 | 2. ZOI11 Only | 2,126 | 47 | 42 | 22.58 | 9.90 | 4.48 | -8.21 | 5.60 | 8.20 | 37.02 | 61 | 40 |
| ENT-02 | drag | FC36 | 3. CFD01 Only | 1,938 | 47 | 43 | 23.69 | 10.83 | 4.93 | -8.55 | 5.58 | 7.29 | 30.91 | 60 | 40 |
| ENT-02 | drag | FC36 | 4. Both | 3,074 | 47 | 44 | 23.23 | 10.32 | 2.46 | -8.29 | 4.95 | 7.96 | 56.44 | 61 | 40 |
| ENT-02 | drag | FFD3 | 1. Excluded | 521 | 35 | 27 | 22.62 | 8.19 | 2.85 | -5.20 | 6.37 | 16.03 |  | 42 | 6 |
| ENT-02 | drag | FFD3 | 2. ZOI11 Only | 904 | 35 | 32 | 20.23 | 8.15 | 3.37 | -7.81 | 5.41 | 7.77 | 42.37 | 42 | 6 |
| ENT-02 | drag | FFD3 | 3. CFD01 Only | 523 | 35 | 28 | 23.15 | 8.72 | 3.54 | -5.51 | 6.37 | 16.03 | 0.38 | 42 | 6 |
| ENT-02 | drag | FFD3 | 4. Both | 911 | 35 | 33 | 20.69 | 8.42 | 3.26 | -7.81 | 5.19 | 7.77 | 42.81 | 42 | 6 |
| ENT-03 | drag | FC36 | 1. Excluded | 3,360 | 65 | 57 | 31.79 | 14.68 | 6.37 | -3.36 | 3.47 | 12.18 |  | 106 | 58 |
| ENT-03 | drag | FC36 | 2. ZOI11 Only | 5,496 | 66 | 60 | 31.21 | 14.27 | 4.90 | -4.44 | 3.18 | 10.57 | 38.61 | 106 | 58 |
| ENT-03 | drag | FC36 | 3. CFD01 Only | 4,826 | 65 | 57 | 32.47 | 14.74 | 6.60 | -4.09 | 3.16 | 15.55 | 30.38 | 106 | 58 |
| ENT-03 | drag | FC36 | 4. Both | 7,551 | 66 | 62 | 32.09 | 14.42 | 4.98 | -4.62 | 3.13 | 9.32 | 55.32 | 106 | 58 |
| ENT-03 | drag | FFD3 | 1. Excluded | 1,783 | 60 | 51 | 30.79 | 14.23 | 6.58 | -1.84 | 2.80 | 8.61 |  | 32 | 14 |
| ENT-03 | drag | FFD3 | 2. ZOI11 Only | 2,583 | 60 | 51 | 29.84 | 13.98 | 4.22 | -2.51 | 2.66 | 6.42 | 30.97 | 33 | 14 |
| ENT-03 | drag | FFD3 | 3. CFD01 Only | 1,970 | 60 | 56 | 30.41 | 13.84 | 4.80 | -2.42 | 2.36 | 4.59 | 9.49 | 32 | 14 |
| ENT-03 | drag | FFD3 | 4. Both | 2,799 | 60 | 57 | 30.48 | 14.10 | 4.18 | -2.54 | 2.32 | 3.93 | 36.30 | 33 | 14 |
| ENT-04 | drag | FC36 | 1. Excluded | 3,481 | 75 | 63 | 30.22 | 16.59 | 3.40 | -3.40 | 2.72 | 4.40 |  | 130 | 64 |
| ENT-04 | drag | FC36 | 2. ZOI11 Only | 5,702 | 79 | 72 | 33.91 | 17.94 | 3.60 | -4.61 | 2.83 | 4.74 | 38.95 | 130 | 64 |
| ENT-04 | drag | FC36 | 3. CFD01 Only | 5,497 | 76 | 67 | 31.73 | 17.32 | 4.26 | -4.34 | 2.68 | 4.47 | 36.67 | 130 | 64 |
| ENT-04 | drag | FC36 | 4. Both | 8,482 | 79 | 73 | 34.63 | 18.30 | 3.87 | -4.90 | 2.92 | 5.38 | 58.96 | 130 | 64 |
| ENT-04 | drag | FFD3 | 1. Excluded | 1,708 | 64 | 51 | 31.84 | 16.35 | 3.68 | -1.75 | 2.51 | 9.51 |  | 14 | 21 |
| ENT-04 | drag | FFD3 | 2. ZOI11 Only | 1,847 | 67 | 52 | 32.90 | 16.75 | 3.76 | -1.70 | 2.52 | 9.54 | 6.98 | 14 | 21 |
| ENT-04 | drag | FFD3 | 3. CFD01 Only | 2,023 | 65 | 54 | 32.46 | 16.34 | 4.93 | -1.82 | 2.50 | 5.31 | 15.57 | 14 | 21 |
| ENT-04 | drag | FFD3 | 4. Both | 2,162 | 67 | 56 | 33.89 | 16.95 | 4.51 | -1.87 | 2.52 | 5.80 | 20.54 | 14 | 21 |
| ENT-05 | drag | FC36 | 1. Excluded | 3,096 | 80 | 60 | 35.06 | 21.49 | 6.33 | -1.95 | 2.82 | 8.37 |  | 147 | 87 |
| ENT-05 | drag | FC36 | 2. ZOI11 Only | 5,401 | 84 | 71 | 34.57 | 20.39 | 6.15 | -2.41 | 2.61 | 6.19 | 42.68 | 147 | 87 |
| ENT-05 | drag | FC36 | 3. CFD01 Only | 5,307 | 84 | 61 | 34.07 | 21.24 | 5.33 | -4.03 | 2.71 | 11.50 | 41.66 | 147 | 87 |
| ENT-05 | drag | FC36 | 4. Both | 8,226 | 85 | 67 | 34.38 | 21.22 | 6.24 | -4.12 | 2.71 | 9.30 | 62.36 | 147 | 87 |
| ENT-05 | drag | FFD3 | 1. Excluded | 1,702 | 70 | 59 | 33.42 | 23.48 | 4.35 | -1.89 | 2.95 | 9.23 |  | 23 | 28 |
| ENT-05 | drag | FFD3 | 2. ZOI11 Only | 1,833 | 72 | 66 | 34.68 | 23.41 | 3.89 | -1.73 | 2.61 | 6.75 | 7.15 | 23 | 28 |
| ENT-05 | drag | FFD3 | 3. CFD01 Only | 1,975 | 74 | 64 | 34.49 | 24.35 | 4.44 | -2.24 | 2.61 | 8.73 | 13.82 | 23 | 28 |
| ENT-05 | drag | FFD3 | 4. Both | 2,106 | 76 | 68 | 34.15 | 23.41 | 4.00 | -1.98 | 2.50 | 6.55 | 19.18 | 23 | 28 |
| FBY-D-1 | drag | FC36 | 1. Excluded | 2,048 | 96 | 62 | 22.66 | 12.07 | 8.16 | -1.35 | 23.60 | 42.23 |  | 203 | 114 |
| FBY-D-1 | drag | FC36 | 2. ZOI11 Only | 3,748 | 112 | 56 | 23.03 | 11.60 | 7.38 | -0.89 | 16.86 | 28.19 | 45.36 | 203 | 114 |
| FBY-D-1 | drag | FC36 | 3. CFD01 Only | 2,368 | 105 | 63 | 31.26 | 15.22 | 10.03 | -0.06 | 24.49 | 41.74 | 13.51 | 203 | 114 |
| FBY-D-1 | drag | FC36 | 4. Both | 4,255 | 114 | 62 | 37.46 | 13.48 | 11.50 | -1.38 | 17.15 | 28.76 | 51.87 | 203 | 114 |
| FBY-D-1 | drag | FFD3 | 1. Excluded | 31 | 19 | 1 |  |  |  | -54.48 | 38.82 | 38.82 |  | 61 | 62 |
| FBY-D-1 | drag | FFD3 | 2. ZOI11 Only | 125 | 43 | 8 | 15.93 | 3.66 | 2.40 | 14.78 | 3.69 | 4.99 | 75.20 | 61 | 62 |
| FBY-D-1 | drag | FFD3 | 3. CFD01 Only | 42 | 22 | 0 |  |  |  |  |  |  | 26.19 | 61 | 62 |
| FBY-D-1 | drag | FFD3 | 4. Both | 136 | 45 | 8 | 15.93 | 3.66 | 2.40 | 14.78 | 3.69 | 4.99 | 77.21 | 61 | 62 |
| FBY-D-2 | drag | FC36 | 1. Excluded | 3,755 | 159 | 97 | 22.16 | 16.56 | 9.44 | -1.21 | 7.39 | 17.12 |  | 235 | 172 |
| FBY-D-2 | drag | FC36 | 2. ZOI11 Only | 6,915 | 166 | 125 | 19.10 | 17.02 | 7.25 | -2.19 | 2.97 | 10.46 | 45.70 | 235 | 172 |
| FBY-D-2 | drag | FC36 | 3. CFD01 Only | 6,554 | 162 | 128 | 20.68 | 21.11 | 7.62 | -1.78 | 4.16 | 13.85 | 42.71 | 235 | 172 |
| FBY-D-2 | drag | FC36 | 4. Both | 10,483 | 167 | 133 | 19.02 | 19.69 | 6.12 | -3.18 | 2.56 | 7.17 | 64.18 | 235 | 172 |
| FBY-D-2 | drag | FFD3 | 1. Excluded | 681 | 105 | 34 | 17.59 | 10.63 | 13.64 | -7.02 | 7.42 | 36.62 |  | 62 | 65 |
| FBY-D-2 | drag | FFD3 | 2. ZOI11 Only | 952 | 113 | 43 | 18.44 | 19.68 | 10.51 | -6.37 | 4.06 | 20.19 | 28.47 | 62 | 65 |
| FBY-D-2 | drag | FFD3 | 3. CFD01 Only | 911 | 113 | 41 | 15.80 | 16.92 | 9.13 | -5.17 | 3.31 | 9.64 | 25.25 | 62 | 65 |
| FBY-D-2 | drag | FFD3 | 4. Both | 1,202 | 119 | 59 | 18.54 | 20.03 | 6.61 | -0.66 | 3.22 | 9.00 | 43.34 | 62 | 65 |
| FBY-D-3 | drag | FC36 | 1. Excluded | 9,008 | 322 | 267 | 28.76 | 21.28 | 6.13 | -2.69 | 3.30 | 9.15 |  | 591 | 316 |
| FBY-D-3 | drag | FC36 | 2. ZOI11 Only | 15,190 | 335 | 298 | 29.93 | 25.51 | 5.54 | -3.60 | 3.33 | 7.93 | 40.70 | 591 | 316 |
| FBY-D-3 | drag | FC36 | 3. CFD01 Only | 13,544 | 330 | 283 | 31.02 | 25.36 | 6.44 | -3.33 | 2.85 | 8.68 | 33.49 | 591 | 316 |
| FBY-D-3 | drag | FC36 | 4. Both | 21,798 | 336 | 300 | 30.15 | 25.29 | 5.78 | -4.32 | 2.79 | 7.64 | 58.68 | 591 | 316 |
| FBY-D-3 | drag | FFD3 | 1. Excluded | 3,219 | 252 | 174 | 28.20 | 20.02 | 5.79 | -0.67 | 3.63 | 9.90 |  | 158 | 72 |
| FBY-D-3 | drag | FFD3 | 2. ZOI11 Only | 4,374 | 261 | 196 | 31.35 | 23.23 | 6.53 | -1.28 | 3.63 | 8.94 | 26.41 | 158 | 72 |
| FBY-D-3 | drag | FFD3 | 3. CFD01 Only | 3,560 | 262 | 184 | 29.39 | 22.09 | 6.18 | -0.67 | 3.67 | 10.01 | 9.58 | 158 | 72 |
| FBY-D-3 | drag | FFD3 | 4. Both | 4,740 | 266 | 206 | 32.28 | 24.85 | 6.42 | -1.58 | 3.84 | 8.73 | 32.09 | 158 | 72 |
| FBY-D-4 | drag | FC36 | 1. Excluded | 1,831 | 214 | 91 | 33.49 | 28.91 | 7.55 | -2.10 | 19.70 | 50.41 |  | 395 | 211 |
| FBY-D-4 | drag | FC36 | 2. ZOI11 Only | 3,892 | 245 | 145 | 34.35 | 30.27 | 8.58 | -3.87 | 13.06 | 36.50 | 52.95 | 396 | 211 |
| FBY-D-4 | drag | FC36 | 3. CFD01 Only | 2,638 | 220 | 113 | 33.86 | 29.09 | 7.79 | -2.63 | 13.79 | 49.56 | 30.59 | 395 | 211 |
| FBY-D-4 | drag | FC36 | 4. Both | 5,092 | 249 | 159 | 33.46 | 30.69 | 7.60 | -3.73 | 13.76 | 35.49 | 64.04 | 396 | 211 |
| FBY-D-4 | drag | FFD3 | 1. Excluded | 355 | 70 | 18 | 34.02 | 27.29 | 3.89 | -7.60 | 4.82 | 9.07 |  | 169 | 132 |
| FBY-D-4 | drag | FFD3 | 2. ZOI11 Only | 671 | 100 | 37 | 34.48 | 38.86 | 7.32 | -9.40 | 4.25 | 12.83 | 47.09 | 169 | 132 |
| FBY-D-4 | drag | FFD3 | 3. CFD01 Only | 493 | 74 | 25 | 36.99 | 34.06 | 3.63 | -9.07 | 5.51 | 14.75 | 27.99 | 169 | 132 |
| FBY-D-4 | drag | FFD3 | 4. Both | 827 | 104 | 40 | 34.16 | 38.54 | 7.97 | -10.54 | 5.29 | 12.13 | 57.07 | 169 | 132 |

## V4b. Fixes gained and lost against 1. Excluded

Matched on transmission number.

| event | tag | arm | fixes_arm1 | fixes_arm | gained | lost | in_both |
|---|---|---|---|---|---|---|---|
| FBY-S-1 | FC36 | 2. ZOI11 Only | 21 | 21 | 0 | 0 | 21 |
| FBY-S-1 | FC36 | 3. CFD01 Only | 21 | 21 | 0 | 0 | 21 |
| FBY-S-1 | FC36 | 4. Both | 21 | 21 | 0 | 0 | 21 |
| FBY-S-1 | FFD3 | 2. ZOI11 Only | 19 | 19 | 0 | 0 | 19 |
| FBY-S-1 | FFD3 | 3. CFD01 Only | 19 | 19 | 0 | 0 | 19 |
| FBY-S-1 | FFD3 | 4. Both | 19 | 19 | 0 | 0 | 19 |
| FBY-S-2 | FC36 | 2. ZOI11 Only | 21 | 21 | 0 | 0 | 21 |
| FBY-S-2 | FC36 | 3. CFD01 Only | 21 | 21 | 0 | 0 | 21 |
| FBY-S-2 | FC36 | 4. Both | 21 | 21 | 0 | 0 | 21 |
| FBY-S-2 | FFD3 | 2. ZOI11 Only | 13 | 13 | 0 | 0 | 13 |
| FBY-S-2 | FFD3 | 3. CFD01 Only | 13 | 13 | 0 | 0 | 13 |
| FBY-S-2 | FFD3 | 4. Both | 13 | 13 | 0 | 0 | 13 |
| FBY-S-3 | FC36 | 2. ZOI11 Only | 20 | 20 | 0 | 0 | 20 |
| FBY-S-3 | FC36 | 3. CFD01 Only | 20 | 20 | 0 | 0 | 20 |
| FBY-S-3 | FC36 | 4. Both | 20 | 20 | 0 | 0 | 20 |
| FBY-S-3 | FFD3 | 2. ZOI11 Only | 17 | 17 | 0 | 0 | 17 |
| FBY-S-3 | FFD3 | 3. CFD01 Only | 17 | 17 | 0 | 0 | 17 |
| FBY-S-3 | FFD3 | 4. Both | 17 | 17 | 0 | 0 | 17 |
| FBY-S-4 | FC36 | 2. ZOI11 Only | 16 | 16 | 0 | 0 | 16 |
| FBY-S-4 | FC36 | 3. CFD01 Only | 16 | 17 | 1 | 0 | 16 |
| FBY-S-4 | FC36 | 4. Both | 16 | 17 | 1 | 0 | 16 |
| FBY-S-4 | FFD3 | 2. ZOI11 Only | 19 | 19 | 0 | 0 | 19 |
| FBY-S-4 | FFD3 | 3. CFD01 Only | 19 | 19 | 0 | 0 | 19 |
| FBY-S-4 | FFD3 | 4. Both | 19 | 19 | 0 | 0 | 19 |
| FBY-S-5 | FC36 | 2. ZOI11 Only | 22 | 22 | 0 | 0 | 22 |
| FBY-S-5 | FC36 | 3. CFD01 Only | 22 | 22 | 0 | 0 | 22 |
| FBY-S-5 | FC36 | 4. Both | 22 | 22 | 0 | 0 | 22 |
| FBY-S-5 | FFD3 | 2. ZOI11 Only | 16 | 16 | 0 | 0 | 16 |
| FBY-S-5 | FFD3 | 3. CFD01 Only | 16 | 16 | 0 | 0 | 16 |
| FBY-S-5 | FFD3 | 4. Both | 16 | 16 | 0 | 0 | 16 |
| FBY-S-7 | FC36 | 2. ZOI11 Only | 19 | 19 | 0 | 0 | 19 |
| FBY-S-7 | FC36 | 3. CFD01 Only | 19 | 19 | 0 | 0 | 19 |
| FBY-S-7 | FC36 | 4. Both | 19 | 19 | 0 | 0 | 19 |
| FBY-S-7 | FFD3 | 2. ZOI11 Only | 12 | 12 | 0 | 0 | 12 |
| FBY-S-7 | FFD3 | 3. CFD01 Only | 12 | 14 | 2 | 0 | 12 |
| FBY-S-7 | FFD3 | 4. Both | 12 | 14 | 2 | 0 | 12 |
| FBY-S-8 | FC36 | 2. ZOI11 Only | 18 | 18 | 0 | 0 | 18 |
| FBY-S-8 | FC36 | 3. CFD01 Only | 18 | 18 | 0 | 0 | 18 |
| FBY-S-8 | FC36 | 4. Both | 18 | 19 | 1 | 0 | 18 |
| FBY-S-8 | FFD3 | 2. ZOI11 Only | 3 | 3 | 0 | 0 | 3 |
| FBY-S-8 | FFD3 | 3. CFD01 Only | 3 | 9 | 6 | 0 | 3 |
| FBY-S-8 | FFD3 | 4. Both | 3 | 9 | 6 | 0 | 3 |
| FBY-S-9 | FC36 | 2. ZOI11 Only | 4 | 11 | 7 | 0 | 4 |
| FBY-S-9 | FC36 | 3. CFD01 Only | 4 | 5 | 1 | 0 | 4 |
| FBY-S-9 | FC36 | 4. Both | 4 | 12 | 8 | 0 | 4 |
| FBY-S-9 | FFD3 | 2. ZOI11 Only | 7 | 11 | 4 | 0 | 7 |
| FBY-S-9 | FFD3 | 3. CFD01 Only | 7 | 9 | 2 | 0 | 7 |
| FBY-S-9 | FFD3 | 4. Both | 7 | 12 | 5 | 0 | 7 |
| FBY-S-10 | FC36 | 2. ZOI11 Only | 12 | 18 | 6 | 0 | 12 |
| FBY-S-10 | FC36 | 3. CFD01 Only | 12 | 12 | 2 | 2 | 10 |
| FBY-S-10 | FC36 | 4. Both | 12 | 17 | 6 | 1 | 11 |
| FBY-S-10 | FFD3 | 2. ZOI11 Only | 11 | 16 | 5 | 0 | 11 |
| FBY-S-10 | FFD3 | 3. CFD01 Only | 11 | 13 | 4 | 2 | 9 |
| FBY-S-10 | FFD3 | 4. Both | 11 | 17 | 6 | 0 | 11 |
| DB-1 | FC36 | 2. ZOI11 Only | 18 | 19 | 1 | 0 | 18 |
| DB-1 | FC36 | 3. CFD01 Only | 18 | 19 | 1 | 0 | 18 |
| DB-1 | FC36 | 4. Both | 18 | 19 | 1 | 0 | 18 |
| DB-1 | FFD3 | 2. ZOI11 Only | 18 | 18 | 0 | 0 | 18 |
| DB-1 | FFD3 | 3. CFD01 Only | 18 | 18 | 0 | 0 | 18 |
| DB-1 | FFD3 | 4. Both | 18 | 18 | 0 | 0 | 18 |
| DB-2 | FC36 | 2. ZOI11 Only | 21 | 21 | 0 | 0 | 21 |
| DB-2 | FC36 | 3. CFD01 Only | 21 | 21 | 0 | 0 | 21 |
| DB-2 | FC36 | 4. Both | 21 | 21 | 0 | 0 | 21 |
| DB-2 | FFD3 | 2. ZOI11 Only | 13 | 13 | 0 | 0 | 13 |
| DB-2 | FFD3 | 3. CFD01 Only | 13 | 13 | 0 | 0 | 13 |
| DB-2 | FFD3 | 4. Both | 13 | 13 | 0 | 0 | 13 |
| DB-3 | FC36 | 2. ZOI11 Only | 12 | 21 | 9 | 0 | 12 |
| DB-3 | FC36 | 3. CFD01 Only | 12 | 19 | 7 | 0 | 12 |
| DB-3 | FC36 | 4. Both | 12 | 21 | 9 | 0 | 12 |
| DB-3 | FFD3 | 2. ZOI11 Only | 19 | 19 | 0 | 0 | 19 |
| DB-3 | FFD3 | 3. CFD01 Only | 19 | 19 | 0 | 0 | 19 |
| DB-3 | FFD3 | 4. Both | 19 | 19 | 0 | 0 | 19 |
| DB-4 | FC36 | 2. ZOI11 Only | 18 | 19 | 1 | 0 | 18 |
| DB-4 | FC36 | 3. CFD01 Only | 18 | 18 | 0 | 0 | 18 |
| DB-4 | FC36 | 4. Both | 18 | 19 | 1 | 0 | 18 |
| DB-4 | FFD3 | 2. ZOI11 Only | 19 | 19 | 0 | 0 | 19 |
| DB-4 | FFD3 | 3. CFD01 Only | 19 | 19 | 0 | 0 | 19 |
| DB-4 | FFD3 | 4. Both | 19 | 19 | 0 | 0 | 19 |
| DB-5 | FC36 | 2. ZOI11 Only | 0 | 6 | 6 | 0 | 0 |
| DB-5 | FC36 | 3. CFD01 Only | 0 | 6 | 6 | 0 | 0 |
| DB-5 | FC36 | 4. Both | 0 | 10 | 10 | 0 | 0 |
| DB-5 | FFD3 | 2. ZOI11 Only | 13 | 17 | 4 | 0 | 13 |
| DB-5 | FFD3 | 3. CFD01 Only | 13 | 14 | 3 | 2 | 11 |
| DB-5 | FFD3 | 4. Both | 13 | 16 | 5 | 2 | 11 |
| ENT-01 | FC36 | 2. ZOI11 Only | 30 | 30 | 0 | 0 | 30 |
| ENT-01 | FC36 | 3. CFD01 Only | 30 | 30 | 0 | 0 | 30 |
| ENT-01 | FC36 | 4. Both | 30 | 37 | 8 | 1 | 29 |
| ENT-01 | FFD3 | 2. ZOI11 Only | 7 | 9 | 2 | 0 | 7 |
| ENT-01 | FFD3 | 3. CFD01 Only | 7 | 7 | 0 | 0 | 7 |
| ENT-01 | FFD3 | 4. Both | 7 | 10 | 3 | 0 | 7 |
| ENT-02 | FC36 | 2. ZOI11 Only | 42 | 42 | 0 | 0 | 42 |
| ENT-02 | FC36 | 3. CFD01 Only | 42 | 43 | 1 | 0 | 42 |
| ENT-02 | FC36 | 4. Both | 42 | 44 | 2 | 0 | 42 |
| ENT-02 | FFD3 | 2. ZOI11 Only | 27 | 32 | 6 | 1 | 26 |
| ENT-02 | FFD3 | 3. CFD01 Only | 27 | 28 | 1 | 0 | 27 |
| ENT-02 | FFD3 | 4. Both | 27 | 33 | 7 | 1 | 26 |
| ENT-03 | FC36 | 2. ZOI11 Only | 57 | 60 | 5 | 2 | 55 |
| ENT-03 | FC36 | 3. CFD01 Only | 57 | 57 | 5 | 5 | 52 |
| ENT-03 | FC36 | 4. Both | 57 | 62 | 8 | 3 | 54 |
| ENT-03 | FFD3 | 2. ZOI11 Only | 51 | 51 | 1 | 1 | 50 |
| ENT-03 | FFD3 | 3. CFD01 Only | 51 | 56 | 5 | 0 | 51 |
| ENT-03 | FFD3 | 4. Both | 51 | 57 | 6 | 0 | 51 |
| ENT-04 | FC36 | 2. ZOI11 Only | 63 | 72 | 9 | 0 | 63 |
| ENT-04 | FC36 | 3. CFD01 Only | 63 | 67 | 4 | 0 | 63 |
| ENT-04 | FC36 | 4. Both | 63 | 73 | 10 | 0 | 63 |
| ENT-04 | FFD3 | 2. ZOI11 Only | 51 | 52 | 1 | 0 | 51 |
| ENT-04 | FFD3 | 3. CFD01 Only | 51 | 54 | 3 | 0 | 51 |
| ENT-04 | FFD3 | 4. Both | 51 | 56 | 5 | 0 | 51 |
| ENT-05 | FC36 | 2. ZOI11 Only | 60 | 71 | 11 | 0 | 60 |
| ENT-05 | FC36 | 3. CFD01 Only | 60 | 61 | 4 | 3 | 57 |
| ENT-05 | FC36 | 4. Both | 60 | 67 | 10 | 3 | 57 |
| ENT-05 | FFD3 | 2. ZOI11 Only | 59 | 66 | 7 | 0 | 59 |
| ENT-05 | FFD3 | 3. CFD01 Only | 59 | 64 | 6 | 1 | 58 |
| ENT-05 | FFD3 | 4. Both | 59 | 68 | 11 | 2 | 57 |
| FBY-D-1 | FC36 | 2. ZOI11 Only | 62 | 56 | 8 | 14 | 48 |
| FBY-D-1 | FC36 | 3. CFD01 Only | 62 | 63 | 3 | 2 | 60 |
| FBY-D-1 | FC36 | 4. Both | 62 | 62 | 13 | 13 | 49 |
| FBY-D-1 | FFD3 | 2. ZOI11 Only | 1 | 8 | 8 | 1 | 0 |
| FBY-D-1 | FFD3 | 3. CFD01 Only | 1 | 0 | 0 | 1 | 0 |
| FBY-D-1 | FFD3 | 4. Both | 1 | 8 | 8 | 1 | 0 |
| FBY-D-2 | FC36 | 2. ZOI11 Only | 97 | 125 | 37 | 9 | 88 |
| FBY-D-2 | FC36 | 3. CFD01 Only | 97 | 128 | 36 | 5 | 92 |
| FBY-D-2 | FC36 | 4. Both | 97 | 133 | 40 | 4 | 93 |
| FBY-D-2 | FFD3 | 2. ZOI11 Only | 34 | 43 | 15 | 6 | 28 |
| FBY-D-2 | FFD3 | 3. CFD01 Only | 34 | 41 | 15 | 8 | 26 |
| FBY-D-2 | FFD3 | 4. Both | 34 | 59 | 29 | 4 | 30 |
| FBY-D-3 | FC36 | 2. ZOI11 Only | 267 | 298 | 39 | 8 | 259 |
| FBY-D-3 | FC36 | 3. CFD01 Only | 267 | 283 | 27 | 11 | 256 |
| FBY-D-3 | FC36 | 4. Both | 267 | 300 | 38 | 5 | 262 |
| FBY-D-3 | FFD3 | 2. ZOI11 Only | 174 | 196 | 26 | 4 | 170 |
| FBY-D-3 | FFD3 | 3. CFD01 Only | 174 | 184 | 14 | 4 | 170 |
| FBY-D-3 | FFD3 | 4. Both | 174 | 206 | 36 | 4 | 170 |
| FBY-D-4 | FC36 | 2. ZOI11 Only | 91 | 145 | 74 | 20 | 71 |
| FBY-D-4 | FC36 | 3. CFD01 Only | 91 | 113 | 26 | 4 | 87 |
| FBY-D-4 | FC36 | 4. Both | 91 | 159 | 87 | 19 | 72 |
| FBY-D-4 | FFD3 | 2. ZOI11 Only | 18 | 37 | 19 | 0 | 18 |
| FBY-D-4 | FFD3 | 3. CFD01 Only | 18 | 25 | 8 | 1 | 17 |
| FBY-D-4 | FFD3 | 4. Both | 18 | 40 | 23 | 1 | 17 |

## V4c. Where the gained fixes are

Distance from each gained fix to the nearer of ZOI11 and CFD01. The bands only describe the result. Nothing is filtered by them.

| event | tag | arm | gained | under 25 m | 25 to 50 m | over 50 m |
|---|---|---|---|---|---|---|
| FBY-S-1 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-1 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-1 | FC36 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-1 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-1 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-1 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-2 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-2 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-2 | FC36 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-2 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-2 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-2 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-3 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-3 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-3 | FC36 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-3 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-3 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-3 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-4 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-4 | FC36 | 3. CFD01 Only | 1 | 1 | 0 | 0 |
| FBY-S-4 | FC36 | 4. Both | 1 | 1 | 0 | 0 |
| FBY-S-4 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-4 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-4 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-5 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-5 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-5 | FC36 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-5 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-5 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-5 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-7 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-7 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-7 | FC36 | 4. Both | 0 | 0 | 0 | 0 |
| FBY-S-7 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-7 | FFD3 | 3. CFD01 Only | 2 | 0 | 2 | 0 |
| FBY-S-7 | FFD3 | 4. Both | 2 | 0 | 2 | 0 |
| FBY-S-8 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-8 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-S-8 | FC36 | 4. Both | 1 | 1 | 0 | 0 |
| FBY-S-8 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| FBY-S-8 | FFD3 | 3. CFD01 Only | 6 | 6 | 0 | 0 |
| FBY-S-8 | FFD3 | 4. Both | 6 | 6 | 0 | 0 |
| FBY-S-9 | FC36 | 2. ZOI11 Only | 7 | 0 | 3 | 4 |
| FBY-S-9 | FC36 | 3. CFD01 Only | 1 | 0 | 0 | 1 |
| FBY-S-9 | FC36 | 4. Both | 8 | 0 | 2 | 6 |
| FBY-S-9 | FFD3 | 2. ZOI11 Only | 4 | 0 | 0 | 4 |
| FBY-S-9 | FFD3 | 3. CFD01 Only | 2 | 0 | 0 | 2 |
| FBY-S-9 | FFD3 | 4. Both | 5 | 0 | 0 | 5 |
| FBY-S-10 | FC36 | 2. ZOI11 Only | 6 | 0 | 5 | 1 |
| FBY-S-10 | FC36 | 3. CFD01 Only | 2 | 0 | 0 | 2 |
| FBY-S-10 | FC36 | 4. Both | 6 | 0 | 4 | 2 |
| FBY-S-10 | FFD3 | 2. ZOI11 Only | 5 | 0 | 4 | 1 |
| FBY-S-10 | FFD3 | 3. CFD01 Only | 4 | 0 | 0 | 4 |
| FBY-S-10 | FFD3 | 4. Both | 6 | 0 | 4 | 2 |
| DB-1 | FC36 | 2. ZOI11 Only | 1 | 0 | 1 | 0 |
| DB-1 | FC36 | 3. CFD01 Only | 1 | 0 | 1 | 0 |
| DB-1 | FC36 | 4. Both | 1 | 0 | 1 | 0 |
| DB-1 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| DB-1 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| DB-1 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| DB-2 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| DB-2 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| DB-2 | FC36 | 4. Both | 0 | 0 | 0 | 0 |
| DB-2 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| DB-2 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| DB-2 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| DB-3 | FC36 | 2. ZOI11 Only | 9 | 9 | 0 | 0 |
| DB-3 | FC36 | 3. CFD01 Only | 7 | 7 | 0 | 0 |
| DB-3 | FC36 | 4. Both | 9 | 9 | 0 | 0 |
| DB-3 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| DB-3 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| DB-3 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| DB-4 | FC36 | 2. ZOI11 Only | 1 | 1 | 0 | 0 |
| DB-4 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| DB-4 | FC36 | 4. Both | 1 | 1 | 0 | 0 |
| DB-4 | FFD3 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| DB-4 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| DB-4 | FFD3 | 4. Both | 0 | 0 | 0 | 0 |
| DB-5 | FC36 | 2. ZOI11 Only | 6 | 6 | 0 | 0 |
| DB-5 | FC36 | 3. CFD01 Only | 6 | 0 | 1 | 5 |
| DB-5 | FC36 | 4. Both | 10 | 0 | 10 | 0 |
| DB-5 | FFD3 | 2. ZOI11 Only | 4 | 4 | 0 | 0 |
| DB-5 | FFD3 | 3. CFD01 Only | 3 | 1 | 1 | 1 |
| DB-5 | FFD3 | 4. Both | 5 | 3 | 1 | 1 |
| ENT-01 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| ENT-01 | FC36 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| ENT-01 | FC36 | 4. Both | 8 | 0 | 0 | 8 |
| ENT-01 | FFD3 | 2. ZOI11 Only | 2 | 0 | 0 | 2 |
| ENT-01 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| ENT-01 | FFD3 | 4. Both | 3 | 0 | 0 | 3 |
| ENT-02 | FC36 | 2. ZOI11 Only | 0 | 0 | 0 | 0 |
| ENT-02 | FC36 | 3. CFD01 Only | 1 | 0 | 0 | 1 |
| ENT-02 | FC36 | 4. Both | 2 | 0 | 0 | 2 |
| ENT-02 | FFD3 | 2. ZOI11 Only | 6 | 0 | 0 | 6 |
| ENT-02 | FFD3 | 3. CFD01 Only | 1 | 0 | 0 | 1 |
| ENT-02 | FFD3 | 4. Both | 7 | 0 | 0 | 7 |
| ENT-03 | FC36 | 2. ZOI11 Only | 5 | 0 | 0 | 5 |
| ENT-03 | FC36 | 3. CFD01 Only | 5 | 0 | 0 | 5 |
| ENT-03 | FC36 | 4. Both | 8 | 0 | 0 | 8 |
| ENT-03 | FFD3 | 2. ZOI11 Only | 1 | 0 | 1 | 0 |
| ENT-03 | FFD3 | 3. CFD01 Only | 5 | 0 | 0 | 5 |
| ENT-03 | FFD3 | 4. Both | 6 | 0 | 1 | 5 |
| ENT-04 | FC36 | 2. ZOI11 Only | 9 | 0 | 0 | 9 |
| ENT-04 | FC36 | 3. CFD01 Only | 4 | 0 | 0 | 4 |
| ENT-04 | FC36 | 4. Both | 10 | 0 | 0 | 10 |
| ENT-04 | FFD3 | 2. ZOI11 Only | 1 | 0 | 0 | 1 |
| ENT-04 | FFD3 | 3. CFD01 Only | 3 | 0 | 0 | 3 |
| ENT-04 | FFD3 | 4. Both | 5 | 0 | 0 | 5 |
| ENT-05 | FC36 | 2. ZOI11 Only | 11 | 0 | 3 | 8 |
| ENT-05 | FC36 | 3. CFD01 Only | 4 | 0 | 2 | 2 |
| ENT-05 | FC36 | 4. Both | 10 | 0 | 4 | 6 |
| ENT-05 | FFD3 | 2. ZOI11 Only | 7 | 0 | 1 | 6 |
| ENT-05 | FFD3 | 3. CFD01 Only | 6 | 0 | 3 | 3 |
| ENT-05 | FFD3 | 4. Both | 11 | 0 | 4 | 7 |
| FBY-D-1 | FC36 | 2. ZOI11 Only | 8 | 3 | 2 | 3 |
| FBY-D-1 | FC36 | 3. CFD01 Only | 3 | 1 | 0 | 2 |
| FBY-D-1 | FC36 | 4. Both | 13 | 3 | 3 | 7 |
| FBY-D-1 | FFD3 | 2. ZOI11 Only | 8 | 0 | 4 | 4 |
| FBY-D-1 | FFD3 | 3. CFD01 Only | 0 | 0 | 0 | 0 |
| FBY-D-1 | FFD3 | 4. Both | 8 | 0 | 4 | 4 |
| FBY-D-2 | FC36 | 2. ZOI11 Only | 37 | 6 | 15 | 16 |
| FBY-D-2 | FC36 | 3. CFD01 Only | 36 | 4 | 13 | 19 |
| FBY-D-2 | FC36 | 4. Both | 40 | 6 | 15 | 19 |
| FBY-D-2 | FFD3 | 2. ZOI11 Only | 15 | 3 | 0 | 12 |
| FBY-D-2 | FFD3 | 3. CFD01 Only | 15 | 3 | 9 | 3 |
| FBY-D-2 | FFD3 | 4. Both | 29 | 5 | 12 | 12 |
| FBY-D-3 | FC36 | 2. ZOI11 Only | 39 | 1 | 6 | 32 |
| FBY-D-3 | FC36 | 3. CFD01 Only | 27 | 2 | 4 | 21 |
| FBY-D-3 | FC36 | 4. Both | 38 | 2 | 7 | 29 |
| FBY-D-3 | FFD3 | 2. ZOI11 Only | 26 | 0 | 3 | 23 |
| FBY-D-3 | FFD3 | 3. CFD01 Only | 14 | 0 | 4 | 10 |
| FBY-D-3 | FFD3 | 4. Both | 36 | 0 | 6 | 30 |
| FBY-D-4 | FC36 | 2. ZOI11 Only | 74 | 35 | 8 | 31 |
| FBY-D-4 | FC36 | 3. CFD01 Only | 26 | 12 | 3 | 11 |
| FBY-D-4 | FC36 | 4. Both | 87 | 44 | 7 | 36 |
| FBY-D-4 | FFD3 | 2. ZOI11 Only | 19 | 0 | 0 | 19 |
| FBY-D-4 | FFD3 | 3. CFD01 Only | 8 | 0 | 0 | 8 |
| FBY-D-4 | FFD3 | 4. Both | 23 | 0 | 0 | 23 |

## V4d. Error change against 1. Excluded, events with GPS truth, sorted by distance from the receivers

`change_m` is the arm's median error minus arm 1's. `paired_median_change_m` uses only transmissions that have a fix in both arms.

| event | tag | arm | truth_to_nearer_receiver_m | median_err_arm1 | median_err_arm | change_m | paired_fixes | paired_median_change_m |
|---|---|---|---|---|---|---|---|---|
| FBY-S-8 | FC36 | 2. ZOI11 Only | 14.49 | 9.73 | 3.40 | -6.33 | 18 | -5.80 |
| FBY-S-8 | FC36 | 3. CFD01 Only | 14.49 | 9.73 | 3.78 | -5.95 | 18 | -4.91 |
| FBY-S-8 | FC36 | 4. Both | 14.49 | 9.73 | 3.53 | -6.20 | 18 | -5.82 |
| FBY-S-8 | FFD3 | 2. ZOI11 Only | 14.49 | 24.31 | 24.31 | 0.00 | 3 | 0.00 |
| FBY-S-8 | FFD3 | 3. CFD01 Only | 14.49 | 24.31 | 2.52 | -21.79 | 3 | -23.73 |
| FBY-S-8 | FFD3 | 4. Both | 14.49 | 24.31 | 2.52 | -21.79 | 3 | -23.73 |
| ENT-05 | FC36 | 2. ZOI11 Only | 20.15 | 2.82 | 2.61 | -0.21 | 60 | -0.44 |
| ENT-05 | FC36 | 3. CFD01 Only | 20.15 | 2.82 | 2.71 | -0.10 | 57 | 0.06 |
| ENT-05 | FC36 | 4. Both | 20.15 | 2.82 | 2.71 | -0.11 | 57 | -0.17 |
| ENT-05 | FFD3 | 2. ZOI11 Only | 20.15 | 2.95 | 2.61 | -0.34 | 58 | 0.00 |
| ENT-05 | FFD3 | 3. CFD01 Only | 20.15 | 2.95 | 2.61 | -0.34 | 57 | 0.00 |
| ENT-05 | FFD3 | 4. Both | 20.15 | 2.95 | 2.50 | -0.45 | 56 | 0.00 |
| FBY-S-4 | FC36 | 2. ZOI11 Only | 21.01 | 1.30 | 1.15 | -0.15 | 16 | -0.14 |
| FBY-S-4 | FC36 | 3. CFD01 Only | 21.01 | 1.30 | 0.87 | -0.43 | 16 | -0.34 |
| FBY-S-4 | FC36 | 4. Both | 21.01 | 1.30 | 0.81 | -0.49 | 16 | -0.36 |
| FBY-S-4 | FFD3 | 2. ZOI11 Only | 21.01 | 3.72 | 3.72 | 0.00 | 19 | 0.00 |
| FBY-S-4 | FFD3 | 3. CFD01 Only | 21.01 | 3.72 | 3.72 | 0.00 | 19 | 0.00 |
| FBY-S-4 | FFD3 | 4. Both | 21.01 | 3.72 | 3.72 | 0.00 | 19 | 0.00 |
| FBY-S-3 | FC36 | 2. ZOI11 Only | 26.72 | 2.13 | 2.12 | -0.01 | 20 | -0.01 |
| FBY-S-3 | FC36 | 3. CFD01 Only | 26.72 | 2.13 | 2.05 | -0.08 | 20 | 0.00 |
| FBY-S-3 | FC36 | 4. Both | 26.72 | 2.13 | 2.03 | -0.10 | 20 | -0.06 |
| FBY-S-3 | FFD3 | 2. ZOI11 Only | 26.72 | 2.18 | 2.18 | 0.00 | 17 | 0.00 |
| FBY-S-3 | FFD3 | 3. CFD01 Only | 26.72 | 2.18 | 2.18 | 0.00 | 17 | 0.00 |
| FBY-S-3 | FFD3 | 4. Both | 26.72 | 2.18 | 2.18 | 0.00 | 17 | 0.00 |
| FBY-D-4 | FC36 | 2. ZOI11 Only | 29.87 | 19.70 | 13.06 | -6.64 | 70 | -2.55 |
| FBY-D-4 | FC36 | 3. CFD01 Only | 29.87 | 19.70 | 13.79 | -5.91 | 86 | -0.00 |
| FBY-D-4 | FC36 | 4. Both | 29.87 | 19.70 | 13.76 | -5.94 | 71 | -4.58 |
| FBY-D-4 | FFD3 | 2. ZOI11 Only | 29.87 | 4.82 | 4.25 | -0.56 | 18 | 0.00 |
| FBY-D-4 | FFD3 | 3. CFD01 Only | 29.87 | 4.82 | 5.51 | 0.69 | 17 | 0.00 |
| FBY-D-4 | FFD3 | 4. Both | 29.87 | 4.82 | 5.29 | 0.48 | 17 | 0.34 |
| ENT-04 | FC36 | 2. ZOI11 Only | 31.44 | 2.72 | 2.83 | 0.11 | 62 | 0.14 |
| ENT-04 | FC36 | 3. CFD01 Only | 31.44 | 2.72 | 2.68 | -0.05 | 62 | -0.05 |
| ENT-04 | FC36 | 4. Both | 31.44 | 2.72 | 2.92 | 0.20 | 62 | 0.01 |
| ENT-04 | FFD3 | 2. ZOI11 Only | 31.44 | 2.51 | 2.52 | 0.01 | 50 | 0.00 |
| ENT-04 | FFD3 | 3. CFD01 Only | 31.44 | 2.51 | 2.50 | -0.01 | 50 | 0.00 |
| ENT-04 | FFD3 | 4. Both | 31.44 | 2.51 | 2.52 | 0.01 | 50 | 0.00 |
| FBY-D-3 | FC36 | 2. ZOI11 Only | 39.20 | 3.30 | 3.33 | 0.03 | 258 | -0.10 |
| FBY-D-3 | FC36 | 3. CFD01 Only | 39.20 | 3.30 | 2.85 | -0.45 | 255 | -0.36 |
| FBY-D-3 | FC36 | 4. Both | 39.20 | 3.30 | 2.79 | -0.50 | 261 | -0.45 |
| FBY-D-3 | FFD3 | 2. ZOI11 Only | 39.20 | 3.63 | 3.63 | 0.00 | 169 | 0.00 |
| FBY-D-3 | FFD3 | 3. CFD01 Only | 39.20 | 3.63 | 3.67 | 0.04 | 169 | 0.00 |
| FBY-D-3 | FFD3 | 4. Both | 39.20 | 3.63 | 3.84 | 0.21 | 169 | 0.00 |
| ENT-03 | FC36 | 2. ZOI11 Only | 39.86 | 3.47 | 3.18 | -0.29 | 54 | -0.05 |
| ENT-03 | FC36 | 3. CFD01 Only | 39.86 | 3.47 | 3.16 | -0.31 | 51 | -0.05 |
| ENT-03 | FC36 | 4. Both | 39.86 | 3.47 | 3.13 | -0.34 | 53 | -0.22 |
| ENT-03 | FFD3 | 2. ZOI11 Only | 39.86 | 2.80 | 2.66 | -0.14 | 48 | -0.07 |
| ENT-03 | FFD3 | 3. CFD01 Only | 39.86 | 2.80 | 2.36 | -0.45 | 49 | 0.00 |
| ENT-03 | FFD3 | 4. Both | 39.86 | 2.80 | 2.32 | -0.48 | 49 | -0.17 |
| FBY-D-2 | FC36 | 2. ZOI11 Only | 44.05 | 7.39 | 2.97 | -4.42 | 87 | -2.72 |
| FBY-D-2 | FC36 | 3. CFD01 Only | 44.05 | 7.39 | 4.16 | -3.23 | 91 | -1.55 |
| FBY-D-2 | FC36 | 4. Both | 44.05 | 7.39 | 2.56 | -4.84 | 92 | -2.86 |
| FBY-D-2 | FFD3 | 2. ZOI11 Only | 44.05 | 7.42 | 4.06 | -3.37 | 27 | -0.00 |
| FBY-D-2 | FFD3 | 3. CFD01 Only | 44.05 | 7.42 | 3.31 | -4.11 | 25 | 0.00 |
| FBY-D-2 | FFD3 | 4. Both | 44.05 | 7.42 | 3.22 | -4.20 | 29 | -0.33 |
| FBY-D-1 | FC36 | 2. ZOI11 Only | 45.01 | 23.60 | 16.86 | -6.75 | 47 | -5.93 |
| FBY-D-1 | FC36 | 3. CFD01 Only | 45.01 | 23.60 | 24.49 | 0.89 | 59 | 0.00 |
| FBY-D-1 | FC36 | 4. Both | 45.01 | 23.60 | 17.15 | -6.45 | 48 | -6.02 |
| FBY-D-1 | FFD3 | 2. ZOI11 Only | 45.01 | 38.82 | 3.69 | -35.13 | 0 |  |
| FBY-D-1 | FFD3 | 3. CFD01 Only | 45.01 | 38.82 |  |  | 0 |  |
| FBY-D-1 | FFD3 | 4. Both | 45.01 | 38.82 | 3.69 | -35.13 | 0 |  |
| FBY-S-7 | FC36 | 2. ZOI11 Only | 47.36 | 5.34 | 3.00 | -2.34 | 19 | -2.28 |
| FBY-S-7 | FC36 | 3. CFD01 Only | 47.36 | 5.34 | 3.61 | -1.73 | 19 | -0.98 |
| FBY-S-7 | FC36 | 4. Both | 47.36 | 5.34 | 2.94 | -2.40 | 19 | -2.40 |
| FBY-S-7 | FFD3 | 2. ZOI11 Only | 47.36 | 4.91 | 4.91 | 0.00 | 12 | 0.00 |
| FBY-S-7 | FFD3 | 3. CFD01 Only | 47.36 | 4.91 | 2.76 | -2.15 | 12 | -1.78 |
| FBY-S-7 | FFD3 | 4. Both | 47.36 | 4.91 | 2.76 | -2.15 | 12 | -1.78 |
| ENT-02 | FC36 | 2. ZOI11 Only | 52.10 | 6.04 | 5.60 | -0.44 | 41 | -0.30 |
| ENT-02 | FC36 | 3. CFD01 Only | 52.10 | 6.04 | 5.58 | -0.46 | 41 | -0.21 |
| ENT-02 | FC36 | 4. Both | 52.10 | 6.04 | 4.95 | -1.09 | 41 | -0.94 |
| ENT-02 | FFD3 | 2. ZOI11 Only | 52.10 | 6.37 | 5.41 | -0.96 | 26 | -0.16 |
| ENT-02 | FFD3 | 3. CFD01 Only | 52.10 | 6.37 | 6.37 | -0.00 | 27 | 0.00 |
| ENT-02 | FFD3 | 4. Both | 52.10 | 6.37 | 5.19 | -1.19 | 26 | -0.16 |
| FBY-S-10 | FC36 | 2. ZOI11 Only | 55.89 | 33.11 | 10.00 | -23.11 | 12 | -24.29 |
| FBY-S-10 | FC36 | 3. CFD01 Only | 55.89 | 33.11 | 26.74 | -6.37 | 10 | 0.00 |
| FBY-S-10 | FC36 | 4. Both | 55.89 | 33.11 | 6.92 | -26.19 | 11 | -29.13 |
| FBY-S-10 | FFD3 | 2. ZOI11 Only | 55.89 | 58.62 | 12.92 | -45.70 | 11 | -45.70 |
| FBY-S-10 | FFD3 | 3. CFD01 Only | 55.89 | 58.62 | 7.42 | -51.20 | 9 | -36.78 |
| FBY-S-10 | FFD3 | 4. Both | 55.89 | 58.62 | 9.23 | -49.39 | 11 | -52.89 |
| FBY-S-2 | FC36 | 2. ZOI11 Only | 56.36 | 7.21 | 6.96 | -0.25 | 21 | -0.17 |
| FBY-S-2 | FC36 | 3. CFD01 Only | 56.36 | 7.21 | 6.23 | -0.97 | 21 | -0.08 |
| FBY-S-2 | FC36 | 4. Both | 56.36 | 7.21 | 6.01 | -1.20 | 21 | -0.39 |
| FBY-S-2 | FFD3 | 2. ZOI11 Only | 56.36 | 7.77 | 6.79 | -0.98 | 13 | -0.19 |
| FBY-S-2 | FFD3 | 3. CFD01 Only | 56.36 | 7.77 | 7.77 | 0.00 | 13 | 0.00 |
| FBY-S-2 | FFD3 | 4. Both | 56.36 | 7.77 | 6.79 | -0.98 | 13 | -0.22 |
| FBY-S-5 | FC36 | 2. ZOI11 Only | 57.24 | 9.14 | 7.73 | -1.41 | 22 | -1.98 |
| FBY-S-5 | FC36 | 3. CFD01 Only | 57.24 | 9.14 | 7.04 | -2.10 | 22 | -1.47 |
| FBY-S-5 | FC36 | 4. Both | 57.24 | 9.14 | 4.37 | -4.77 | 22 | -6.10 |
| FBY-S-5 | FFD3 | 2. ZOI11 Only | 57.24 | 9.33 | 7.84 | -1.49 | 16 | -1.10 |
| FBY-S-5 | FFD3 | 3. CFD01 Only | 57.24 | 9.33 | 9.33 | 0.00 | 16 | 0.00 |
| FBY-S-5 | FFD3 | 4. Both | 57.24 | 9.33 | 7.84 | -1.49 | 16 | -1.10 |
| FBY-S-1 | FC36 | 2. ZOI11 Only | 57.49 | 2.89 | 4.13 | 1.25 | 21 | 0.18 |
| FBY-S-1 | FC36 | 3. CFD01 Only | 57.49 | 2.89 | 3.57 | 0.68 | 21 | 0.00 |
| FBY-S-1 | FC36 | 4. Both | 57.49 | 2.89 | 4.13 | 1.25 | 21 | -0.01 |
| FBY-S-1 | FFD3 | 2. ZOI11 Only | 57.49 | 3.13 | 2.93 | -0.21 | 19 | -0.12 |
| FBY-S-1 | FFD3 | 3. CFD01 Only | 57.49 | 3.13 | 3.13 | 0.00 | 19 | 0.00 |
| FBY-S-1 | FFD3 | 4. Both | 57.49 | 3.13 | 2.72 | -0.41 | 19 | -0.12 |
| FBY-S-9 | FC36 | 2. ZOI11 Only | 60.53 | 4.90 | 7.94 | 3.04 | 4 | 0.09 |
| FBY-S-9 | FC36 | 3. CFD01 Only | 60.53 | 4.90 | 7.01 | 2.11 | 4 | 0.00 |
| FBY-S-9 | FC36 | 4. Both | 60.53 | 4.90 | 7.01 | 2.11 | 4 | 0.09 |
| FBY-S-9 | FFD3 | 2. ZOI11 Only | 60.53 | 9.04 | 2.69 | -6.36 | 7 | -2.87 |
| FBY-S-9 | FFD3 | 3. CFD01 Only | 60.53 | 9.04 | 5.81 | -3.24 | 7 | 0.00 |
| FBY-S-9 | FFD3 | 4. Both | 60.53 | 9.04 | 4.31 | -4.73 | 7 | -6.75 |
| ENT-01 | FC36 | 2. ZOI11 Only | 68.87 | 8.61 | 7.94 | -0.67 | 29 | -0.12 |
| ENT-01 | FC36 | 3. CFD01 Only | 68.87 | 8.61 | 7.92 | -0.69 | 29 | -0.27 |
| ENT-01 | FC36 | 4. Both | 68.87 | 8.61 | 5.92 | -2.70 | 28 | -1.76 |
| ENT-01 | FFD3 | 2. ZOI11 Only | 68.87 | 9.53 | 7.81 | -1.72 | 7 | -0.03 |
| ENT-01 | FFD3 | 3. CFD01 Only | 68.87 | 9.53 | 9.53 | 0.00 | 7 | 0.00 |
| ENT-01 | FFD3 | 4. Both | 68.87 | 9.53 | 7.52 | -2.00 | 7 | -0.03 |

## Event and tag pairs with no fix in any arm

- none

## Summary sentences (copy these, do not rewrite them)

**2. ZOI11 Only.** 25 of 46 event and tag pairs gained fixes (311 transmissions gained, 66 lost). Of the gained fixes, 68 are under 25 m from the nearer of ZOI11 and CFD01, 56 at 25 to 50 m, 187 over 50 m. On 36 pairs with GPS truth and fixes in both arms, the median error rose by more than 0.5 m on 2, fell by more than 0.5 m on 17, and stayed within 0.5 m on 17. The pairs where it rose lie 57 to 61 m from the nearer receiver.

**3. CFD01 Only.** 26 of 46 event and tag pairs gained fixes (193 transmissions gained, 51 lost). Of the gained fixes, 37 are under 25 m from the nearer of ZOI11 and CFD01, 43 at 25 to 50 m, 113 over 50 m. On 35 pairs with GPS truth and fixes in both arms, the median error rose by more than 0.5 m on 4, fell by more than 0.5 m on 13, and stayed within 0.5 m on 18. The pairs where it rose lie 30 to 61 m from the nearer receiver.

**4. Both.** 31 of 46 event and tag pairs gained fixes (405 transmissions gained, 64 lost). Of the gained fixes, 81 are under 25 m from the nearer of ZOI11 and CFD01, 87 at 25 to 50 m, 237 over 50 m. On 36 pairs with GPS truth and fixes in both arms, the median error rose by more than 0.5 m on 2, fell by more than 0.5 m on 21, and stayed within 0.5 m on 13. The pairs where it rose lie 57 to 61 m from the nearer receiver.

Include or exclude is the PM's decision.
