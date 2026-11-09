# PM TDOA Plots for All Receivers — 2026-11-08

## Request
The user asked for the existing PM-style plots to be generated for each receiver under `output/11082026_pm_plots/`.

## Work
- Read existing `output/11082026_pm_plots/` plots and the saved Arm A/Arm B `multi_beacon_relative_timing.csv` for interval 2025-06-27 06:00 through 2025-06-30 06:00 (study wall-clock, PDT stored as-if-UTC), master beacon 7DB7 at ZOI08.
- Confirmed the source contains both arms for 18 listening receivers: CFD01-04, CFD06-09, ZOI01-07, ZOI09-11. ZOI08 is the master and has no self-TDOA series.
- Generated one lightly labeled two-panel plot per receiver in `output/11082026_pm_plots/all_receivers/`, plus `summary.csv` and `README.txt`. Grey = Original, blue = New. Each arm is centered on its own median to emphasize changes; absolute inter-arm offset is not shown. Y outliers beyond the 99.5th-percentile absolute deviation are clipped for readability; per-receiver clip levels are recorded in the CSV.
- Plots preserve existing PM style and interval, and do not overwrite existing plot files. Opened and visually checked ZOI01, ZOI02, CFD06, and ZOI06.

## Results
- 18 PNGs, all 1181x670; no empty/small files; 18 summary rows.
- Aggregate adjacent beacon-TDOA changes >=50 ms: **1,758 -> 1,544** across all 18 receivers. No receiver had more such changes in the New arm.
- The 18 individual counts and epoch counts are in `summary.csv`.
- ZOI01 remains a caveat: 323 -> 285 changes, but prior inspection found its New-arm series has a persistent level offset and a remaining -1 s step. These median-centered plots intentionally emphasize variability/steps, not absolute synchronization.
- Timing plots are not fish-position accuracy evidence and do not clear the clock review gate.

## Decisions and parameters
No code, databases, raw data, physical parameters, DBSCAN settings, thresholds, or run configurations changed. Analysis used saved, read-only plot data. No helper scripts were created. Existing plots and Arm A baseline were not modified.

## Next
User may share `output/11082026_pm_plots/all_receivers/` with the PM as a timing diagnostic, with the offset caveat above. Continue using the blocked production gate; no fish processing was run.
