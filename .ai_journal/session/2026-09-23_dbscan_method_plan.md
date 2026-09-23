# Session: 2026-09-23 — DBSCAN Method Plan

- Author: Ethan Muhlestein / Copilot
- Date: 2026-09-23
- Tags: #dbscan #multipath #epoch-model #parameter-selection #gate-M4
- Branch: `ENM_jsat3d_edits`

## Goal

Develop a defensible DBSCAN multipath workflow for 2025 ATS data without silently tuning parameters or changing the final database.

## Method Boundary

- Final database remains read-only during diagnostics.
- No detection rejection yet.
- No production `eps` or `min_samples` selected yet.
- No legacy core changes.
- All diagnostic runs use bounded tag/receiver scopes or streaming representative epochs.

## Required Order

1. Verify beacon epoch definition before clustering. Nominal PRI values are not automatically exact because meeting notes document jitter, drift, and catch-up pings.
2. Use raw Internal evidence and beacon interval distributions to identify epoch boundaries and anomalies.
3. Define approved feature space: lag from first arrival, relative SigStr, epoch rank, and inter-detection interval.
4. Define one global feature scaling method. Do not scale separately per receiver or group.
5. Derive physical lag bounds from approved geometry, depth, and sound-speed inputs.
6. Generate exploratory k-distance diagnostics using deterministic representative epoch points.
7. Compare fixed candidate `min_samples` values using beacon/static-hold validation.
8. Select one fixed study-wide `eps` and `min_samples` only after metrics and owner review.
9. Apply filtering on a copy, preserve original rows, and report rejection accounting.

## First Bounded Diagnostic

- Scope: ZOI02 local beacon `7D2D`, then one or two additional representative beacon receivers.
- Preserve all source rows in diagnostic outputs.
- Report interval distributions, epoch-size distributions, Internal event boundaries, lag, and relative SigStr.
- Do not infer production PRI from a single percentile.
- Do not use old legacy `eps` values.

## Known Blockers

- Beacon PRI is jittery and includes catch-up pings.
- Temperature string begins June 17; June 5-16 validation lacks authoritative temperature.
- Reflection-boundary geometry for physical `eps` derivation is not yet documented.
- Upstream multipath provenance remains unresolved.

## Validation Requirements

- Track input rows, complete-feature rows, epoch counts, representative-row rules, and rejected rows.
- Compare candidate behavior across receiver and tag groups.
- Keep diagnostic artifacts separate from final database.
- Record every parameter change and rationale.
- Do not call diagnostic output a production filter result.

## Next Step

Inspect beacon interval and Internal-event structure for a bounded ZOI02 beacon slice before expanding DBSCAN scope.

## 2021 Paper Lessons

- The published workflow confirms the correct broad order: import SQLite data, enumerate beacon epochs, remove beacon multipath, synchronize clocks, position submerged receivers, remove fish-tag multipath, then solve positions.
- Four or more receivers are required for a 3D position. Positions outside the receiver convex hull are extrapolations and are less reliable.
- First-arrival retention is physically justified because reflected paths are longer and normally arrive later. However, the paper explicitly notes that the direct path can be missed, so first-arrival filtering is only a primary filter, not a complete solution.
- The paper's second-stage multipath work used biased first-arrival labels, an unsupervised k-means step, then SVC, NB, CART, and KNN classifiers. KNN was the practical choice, but it depended on SNR, NBW, FreqOff, and amplitude fields unavailable in the current ATS concatenated data.
- The paper reports CPDI near reflective infrastructure as unresolved. Filtering cannot replace receiver repositioning or baffling.
- Clock synchronization used a known reference receiver, beacon TDOA, known receiver geometry, and sound speed. It fit piecewise linear interpolation to receiver clock bias between beacon epochs, which supports the current ZOI02-reference and piecewise-regression direction.
- The paper warns that missed reference-beacon epochs create long gaps where linear drift interpolation becomes less reliable. This directly supports treating the current beacon jitter, catch-up pings, missing detections, and firmware jumps as separate epoch-quality issues.
- Temperature was measured at multiple depths and averaged at each time to derive sound speed. The current project has four DD_N columns but lacks coverage for the June 5-16 validation window, so the paper's sound-speed method cannot yet be applied to the tag-drag dates.
- The paper retained physically impossible positions rather than silently discarding them and assessed precision separately inside and outside the convex hull. This remains a required reporting rule.
- Published benchmark values remain the acceptance reference: inside-hull precision approximately 0.06 m, 0.06 m, 0.12 m; outside-hull precision approximately 0.25 m, 0.33 m, 0.27 m; metronome accuracy RMSE approximately 0.02 m, 0.05 m, 0.004 m.

## 2025 Implications

- Reuse legacy mathematics, SQLite-centered staging, epoch concepts, first-arrival physics, piecewise clock-bias modeling, convex-hull labeling, and RMSE/precision reporting.
- Do not reuse legacy KNN inputs or parameter values when SNR/NBW/FreqOff are absent.
- DBSCAN must be treated as a new diagnostic/filter method, with lag and relative SigStr as the strongest available inputs.
- Validate DBSCAN first on beacon data, because beacon-derived first-arrival structure provides the closest available multipath reference before applying it to study tags.
- Keep original detections and add flags; do not overwrite raw or final source rows.
- The paper strengthens the case for solving synchronization before positioning and for separating jump correction from drift interpolation.

## ZOI02 Bounded Result

- Added `scripts/inspect_beacon_epoch_timing.py`.
- Scope: tag `7D2D`, receiver `ZOI02`, streamed from final DB.
- Rows: 682,708.
- Median row interval: 0.017 seconds, showing dense burst/multipath rows.
- 553,210 intervals were <= 1 second.
- 128,849 intervals were between 30 and 90 seconds.
- 633 intervals exceeded 90 seconds; maximum gap was approximately 8.1 days.
- 99th-percentile row interval was approximately 62.6 seconds.
- Internal evidence included 24 offset changes, 280 counter-restart combinations, and 1,729 one-second-adjustment evidence rows.
- Created `output/zoi02_7d2d_timing.csv`.
- Interpretation: nominal beacon period cannot be used directly on raw rows. Epoch construction must collapse bursts and handle outages/clock events before DBSCAN features are treated as valid.
- No detections filtered and no DBSCAN parameters selected.

## Closeout Update — 2026-09-23

- Removed the exploratory timing script and generated timing artifact during validation-scope cleanup.
- Retained DBSCAN feature and diagnostic modules because repository tests depend on them.
- Final decision remains unchanged: no production `eps` or `min_samples` selected.
- Future DBSCAN work waits for validated epoch construction, receiver geometry, synchronization, and approved validation tags.
