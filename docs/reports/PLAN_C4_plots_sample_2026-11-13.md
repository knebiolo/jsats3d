# Plan C4 plots (sample)

Tool version fourarm-2026-11-12-a. Times are study wall clock (PDT stored as if UTC).

- **solution row**: a root B row of `tblPositions_Deng` with comment `solution found`. One row per receiver combination. It rises with every receiver added. It is not a count of fixes.
- **fix**: for a static hold, a transmission kept by the project's receiver set consensus (two or more receiver sets within 10 m). For a drag, a point of the project's consensus, smoothing and Kalman track.
- **error**: XY distance from a fix to GPS truth.
- **nearer receiver**: the nearer of ZOI11 and CFD01, in XY.

Plots made: 6 PNG and 6 HTML. Event and tag pairs without a fix in any arm get no plot: 0.

`off_plot` counts fixes outside the plotted area per arm. They are counted in every table.

| event | kind | tag | fixes_N | fixes_Z | fixes_C | fixes_ZC | png | html | off_plot |
|---|---|---|---|---|---|---|---|---|---|
| FBY-S-1 | forebay hold | FC36 | 21 | 21 | 21 | 21 | plots_2d/FBY-S-1_FC36.png | plots_3d/FBY-S-1_FC36.html |  |
| FBY-S-1 | forebay hold | FFD3 | 19 | 19 | 19 | 19 | plots_2d/FBY-S-1_FFD3.png | plots_3d/FBY-S-1_FFD3.html |  |
| DB-1 | debris barrier hold | FC36 | 18 | 19 | 19 | 19 | plots_2d/DB-1_FC36.png | plots_3d/DB-1_FC36.html |  |
| DB-1 | debris barrier hold | FFD3 | 18 | 18 | 18 | 18 | plots_2d/DB-1_FFD3.png | plots_3d/DB-1_FFD3.html |  |
| ENT-01 | drag | FC36 | 30 | 30 | 30 | 37 | plots_2d/ENT-01_FC36.png | plots_3d/ENT-01_FC36.html |  |
| ENT-01 | drag | FFD3 | 7 | 9 | 7 | 10 | plots_2d/ENT-01_FFD3.png | plots_3d/ENT-01_FFD3.html |  |
