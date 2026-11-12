# PM Request: ZOI11 / CFD01 Receiver Evaluation Plots

**Date:** 2026-11-11  
**From:** PM (via Ethan)  
**For:** Claude Code 5.2  

---

## Context

The PM has been reviewing the clock jump correction work (Plans B, B2). The analysis shows that not all clock jump events are identified through the Internal data from receivers. If jumps cannot be identified, corrections would require guessing, which could leave data off within the one-minute beacon period.

The PM is now considering whether clock jump corrections are even needed. Alternative ideas being discussed with Kevin:
- Flag periods where jumps occur and carry flags through analysis
- Apply spatial filters on solution clusters before taking the median for XYZ, to avoid biasing positions from uncorrected jumps
- Possibly abandon clock corrections entirely and rely on piecewise regression + spatial filtering

**This is under discussion with Kevin — not yet a decision.**

---

## Action Item: ZOI11 / CFD01 Receiver Evaluation

Move forward with the ZOI11 and CFD01 positioning evaluation. The goal is to **qualitatively decide whether to include or exclude these two receivers** from the production receiver array.

### Only reason to include either receiver:
It increases positioning resolution underneath the debris barrier **without** degrading positions in other locations.

---

## Required Products

### Plots for every event, showing all 4 scenarios together

For **each** of the following events, create plots comparing four arm configurations:

| Arm | Receivers | Receivers 3D |
|---|---|---|
| **1. Excluded** (baseline) | ZOI01, ZOI02, ZOI03, ZOI05, ZOI06, ZOI07, ZOI08, ZOI09, ZOI10 | Same nine |
| **2. ZOI11 Only** | Same nine + ZOI11 | Same nine + ZOI11 |
| **3. CFD01 Only** | Same nine + CFD01 | Same nine + CFD01 |
| **4. Both** | Same nine + ZOI11 + CFD01 | Same nine + ZOI11 + CFD01 |

**Events to plot:**
- **Static holds:** All holds with "S" prefix (forebay holds) and "DB" prefix (debris barrier holds)
- **Validation drags:** All validation drag lines

### Plot specifications

Each event gets plots showing all 4 scenarios. The PM prefers:
- **Interactive 3D plots** (plan view with elevation)
- **Plan view 2D plots** (top-down XY)

Each plot must show:
- The four positioning solutions (one per arm configuration), in distinct colors
- GPS tracks (truth reference) for comparison
- Receivers as reference markers

The plots will be busy — that's expected. The goal is qualitative visual assessment, not pixel-perfect clarity.

---

## What to Report

For each event and each arm, report:
- Number of solutions / fixes
- Position spread (XY and optionally Z)
- Any visible degradation compared to baseline
- Whether ZOI11 or CFD01 adds solutions in areas where the baseline has none (especially near/under the debris barrier)

---

## Constraints

- Do **not** modify any existing output folders or databases.
- New work goes in a fresh output folder.
- Do **not** commit or push.
- No parameter changes to the positioning pipeline — only the receiver lists differ between arms.
- Debris barrier holds (DB-1 through DB-5) have **no GPS truth** — plots for those should show spread and depth only, not error.

---

## Notes

- CFD09 (serial 19026) was at the CFD05 location until about 2025-06-11 12:40. CFD09 is in `surface_receivers` only, not `receivers_3d`, so it should **not** be added.
- CFD01 is already in `surface_receivers` in existing run files, so its clock is already synchronised. Adding it to `receivers_3d` is the only change needed for the CFD01-only arm.
- ZOI11 is in neither list in the baseline, so it must be added to both `surface_receivers` and `receivers_3d` for the ZOI11 arms.
