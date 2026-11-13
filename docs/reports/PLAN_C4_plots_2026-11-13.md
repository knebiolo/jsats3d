# Plan C4 plots

Tool version fourarm-2026-11-12-a. Times are study wall clock (PDT stored as if UTC).

- **solution row**: a root B row of `tblPositions_Deng` with comment `solution found`. One row per receiver combination. It rises with every receiver added. It is not a count of fixes.
- **fix**: for a static hold, a transmission kept by the project's receiver set consensus (two or more receiver sets within 10 m). For a drag, a point of the project's consensus, smoothing and Kalman track.
- **error**: XY distance from a fix to GPS truth.
- **nearer receiver**: the nearer of ZOI11 and CFD01, in XY.

Plots made: 46 PNG and 46 HTML. Event and tag pairs without a fix in any arm get no plot: 0.

`off_plot` counts fixes outside the plotted area per arm. They are counted in every table.

| event | kind | tag | fixes_N | fixes_Z | fixes_C | fixes_ZC | png | html | off_plot |
|---|---|---|---|---|---|---|---|---|---|
| FBY-S-1 | forebay hold | FC36 | 21 | 21 | 21 | 21 | plots_2d/FBY-S-1_FC36.png | plots_3d/FBY-S-1_FC36.html |  |
| FBY-S-1 | forebay hold | FFD3 | 19 | 19 | 19 | 19 | plots_2d/FBY-S-1_FFD3.png | plots_3d/FBY-S-1_FFD3.html |  |
| FBY-S-2 | forebay hold | FC36 | 21 | 21 | 21 | 21 | plots_2d/FBY-S-2_FC36.png | plots_3d/FBY-S-2_FC36.html |  |
| FBY-S-2 | forebay hold | FFD3 | 13 | 13 | 13 | 13 | plots_2d/FBY-S-2_FFD3.png | plots_3d/FBY-S-2_FFD3.html |  |
| FBY-S-3 | forebay hold | FC36 | 20 | 20 | 20 | 20 | plots_2d/FBY-S-3_FC36.png | plots_3d/FBY-S-3_FC36.html |  |
| FBY-S-3 | forebay hold | FFD3 | 17 | 17 | 17 | 17 | plots_2d/FBY-S-3_FFD3.png | plots_3d/FBY-S-3_FFD3.html |  |
| FBY-S-4 | forebay hold | FC36 | 16 | 16 | 17 | 17 | plots_2d/FBY-S-4_FC36.png | plots_3d/FBY-S-4_FC36.html |  |
| FBY-S-4 | forebay hold | FFD3 | 19 | 19 | 19 | 19 | plots_2d/FBY-S-4_FFD3.png | plots_3d/FBY-S-4_FFD3.html |  |
| FBY-S-5 | forebay hold | FC36 | 22 | 22 | 22 | 22 | plots_2d/FBY-S-5_FC36.png | plots_3d/FBY-S-5_FC36.html |  |
| FBY-S-5 | forebay hold | FFD3 | 16 | 16 | 16 | 16 | plots_2d/FBY-S-5_FFD3.png | plots_3d/FBY-S-5_FFD3.html |  |
| FBY-S-7 | forebay hold | FC36 | 19 | 19 | 19 | 19 | plots_2d/FBY-S-7_FC36.png | plots_3d/FBY-S-7_FC36.html |  |
| FBY-S-7 | forebay hold | FFD3 | 12 | 12 | 14 | 14 | plots_2d/FBY-S-7_FFD3.png | plots_3d/FBY-S-7_FFD3.html |  |
| FBY-S-8 | forebay hold | FC36 | 18 | 18 | 18 | 19 | plots_2d/FBY-S-8_FC36.png | plots_3d/FBY-S-8_FC36.html |  |
| FBY-S-8 | forebay hold | FFD3 | 3 | 3 | 9 | 9 | plots_2d/FBY-S-8_FFD3.png | plots_3d/FBY-S-8_FFD3.html |  |
| FBY-S-9 | forebay hold | FC36 | 4 | 11 | 5 | 12 | plots_2d/FBY-S-9_FC36.png | plots_3d/FBY-S-9_FC36.html |  |
| FBY-S-9 | forebay hold | FFD3 | 7 | 11 | 9 | 12 | plots_2d/FBY-S-9_FFD3.png | plots_3d/FBY-S-9_FFD3.html |  |
| FBY-S-10 | forebay hold | FC36 | 12 | 18 | 12 | 17 | plots_2d/FBY-S-10_FC36.png | plots_3d/FBY-S-10_FC36.html | C:3 |
| FBY-S-10 | forebay hold | FFD3 | 11 | 16 | 13 | 17 | plots_2d/FBY-S-10_FFD3.png | plots_3d/FBY-S-10_FFD3.html |  |
| DB-1 | debris barrier hold | FC36 | 18 | 19 | 19 | 19 | plots_2d/DB-1_FC36.png | plots_3d/DB-1_FC36.html |  |
| DB-1 | debris barrier hold | FFD3 | 18 | 18 | 18 | 18 | plots_2d/DB-1_FFD3.png | plots_3d/DB-1_FFD3.html |  |
| DB-2 | debris barrier hold | FC36 | 21 | 21 | 21 | 21 | plots_2d/DB-2_FC36.png | plots_3d/DB-2_FC36.html |  |
| DB-2 | debris barrier hold | FFD3 | 13 | 13 | 13 | 13 | plots_2d/DB-2_FFD3.png | plots_3d/DB-2_FFD3.html |  |
| DB-3 | debris barrier hold | FC36 | 12 | 21 | 19 | 21 | plots_2d/DB-3_FC36.png | plots_3d/DB-3_FC36.html |  |
| DB-3 | debris barrier hold | FFD3 | 19 | 19 | 19 | 19 | plots_2d/DB-3_FFD3.png | plots_3d/DB-3_FFD3.html |  |
| DB-4 | debris barrier hold | FC36 | 18 | 19 | 18 | 19 | plots_2d/DB-4_FC36.png | plots_3d/DB-4_FC36.html |  |
| DB-4 | debris barrier hold | FFD3 | 19 | 19 | 19 | 19 | plots_2d/DB-4_FFD3.png | plots_3d/DB-4_FFD3.html |  |
| DB-5 | debris barrier hold | FC36 | 0 | 6 | 6 | 10 | plots_2d/DB-5_FC36.png | plots_3d/DB-5_FC36.html |  |
| DB-5 | debris barrier hold | FFD3 | 13 | 17 | 14 | 16 | plots_2d/DB-5_FFD3.png | plots_3d/DB-5_FFD3.html |  |
| ENT-01 | drag | FC36 | 30 | 30 | 30 | 37 | plots_2d/ENT-01_FC36.png | plots_3d/ENT-01_FC36.html |  |
| ENT-01 | drag | FFD3 | 7 | 9 | 7 | 10 | plots_2d/ENT-01_FFD3.png | plots_3d/ENT-01_FFD3.html |  |
| ENT-02 | drag | FC36 | 42 | 42 | 43 | 44 | plots_2d/ENT-02_FC36.png | plots_3d/ENT-02_FC36.html |  |
| ENT-02 | drag | FFD3 | 27 | 32 | 28 | 33 | plots_2d/ENT-02_FFD3.png | plots_3d/ENT-02_FFD3.html |  |
| ENT-03 | drag | FC36 | 57 | 60 | 57 | 62 | plots_2d/ENT-03_FC36.png | plots_3d/ENT-03_FC36.html |  |
| ENT-03 | drag | FFD3 | 51 | 51 | 56 | 57 | plots_2d/ENT-03_FFD3.png | plots_3d/ENT-03_FFD3.html |  |
| ENT-04 | drag | FC36 | 63 | 72 | 67 | 73 | plots_2d/ENT-04_FC36.png | plots_3d/ENT-04_FC36.html |  |
| ENT-04 | drag | FFD3 | 51 | 52 | 54 | 56 | plots_2d/ENT-04_FFD3.png | plots_3d/ENT-04_FFD3.html |  |
| ENT-05 | drag | FC36 | 60 | 71 | 61 | 67 | plots_2d/ENT-05_FC36.png | plots_3d/ENT-05_FC36.html |  |
| ENT-05 | drag | FFD3 | 59 | 66 | 64 | 68 | plots_2d/ENT-05_FFD3.png | plots_3d/ENT-05_FFD3.html |  |
| FBY-D-1 | drag | FC36 | 62 | 56 | 63 | 62 | plots_2d/FBY-D-1_FC36.png | plots_3d/FBY-D-1_FC36.html | ZC:2 |
| FBY-D-1 | drag | FFD3 | 1 | 8 | 0 | 8 | plots_2d/FBY-D-1_FFD3.png | plots_3d/FBY-D-1_FFD3.html |  |
| FBY-D-2 | drag | FC36 | 97 | 125 | 128 | 133 | plots_2d/FBY-D-2_FC36.png | plots_3d/FBY-D-2_FC36.html |  |
| FBY-D-2 | drag | FFD3 | 34 | 43 | 41 | 59 | plots_2d/FBY-D-2_FFD3.png | plots_3d/FBY-D-2_FFD3.html |  |
| FBY-D-3 | drag | FC36 | 267 | 298 | 283 | 300 | plots_2d/FBY-D-3_FC36.png | plots_3d/FBY-D-3_FC36.html |  |
| FBY-D-3 | drag | FFD3 | 174 | 196 | 184 | 206 | plots_2d/FBY-D-3_FFD3.png | plots_3d/FBY-D-3_FFD3.html |  |
| FBY-D-4 | drag | FC36 | 91 | 145 | 113 | 159 | plots_2d/FBY-D-4_FC36.png | plots_3d/FBY-D-4_FC36.html |  |
| FBY-D-4 | drag | FFD3 | 18 | 37 | 25 | 40 | plots_2d/FBY-D-4_FFD3.png | plots_3d/FBY-D-4_FFD3.html |  |
