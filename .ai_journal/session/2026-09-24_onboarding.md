# Session: 2026-09-24 — Onboarding and Project Readiness

- Author: Ethan Muhlestein / Copilot
- Date: 2026-09-24
- Tags: #onboarding #read-only #legacy-core #clock-sync #dbscan #project-context
- Branch: `ENM_jsat3d_edits`

## Scope
Review the project context, long-term decisions, and recent session records before any implementation work begins.

## Work Completed
- Read the project system prompt and confirmed the project is a 2025 ATS receiver study, not a 2019 re-run.
- Read the repo-level onboarding context in [README.md](../../README.md).
- Read the durable project context in [LONG_TERM_CONTEXT.md](../../LONG_TERM_CONTEXT.md).
- Read the long-term pattern file in [.ai_journal/long_term/personality.md](../long_term/personality.md).
- Read the latest session notes from [.ai_journal/session](./) covering database readiness, parser work, synchronization planning, and DBSCAN constraints.

## Core Working Constraints
- Raw data remains read-only; transformed outputs go to local staging/output databases.
- Legacy `jsats3d` remains the authoritative runtime; new work must feed that core without replacing it.
- No physical constants or assumptions are to be invented without explicit owner validation.
- No silent parameter tuning; any `eps`, `min_samples`, or threshold change must be documented with rationale and validation evidence.
- Journal entries must record all meaningful findings and parameter changes.

## Current Project Position
- The repo is positioned around a 2025 ATS receiver study using 2025 raw files and a final legacy-compatible database already built in `output/`.
- A final DB exists with 20 target receivers, five selected tags, and legacy table structure; the project is past raw-ingestion scaffolding and into synchronization and multipath design.
- `SNR`, `NBW`, and `FreqOff` remain unavailable in current ATS data, which blocks direct reuse of the legacy classifier and forces a new DBSCAN-first design.
- The project has not yet selected a final synchronization architecture; ZOI02 is treated as the central reference candidate, but owner confirmation and residual verification remain required.
- Temperature coverage for the June test window remains a major blocker. The authoritative temperature string begins on 2025-06-17, while the June validation windows begin earlier.
- Beacon and clock-event diagnostics are active, while production `eps` and `min_samples` remain intentionally unselected.

## Active Decisions Carried Forward
- Preserve legacy tables and add ATS-only columns rather than rewriting the core model.
- Use a receiver-aware parsing layer and downstream adapter logic.
- Keep DBSCAN feature design limited to physically justified metrics: lag from first arrival, relative amplitude, epoch rank, and inter-detection interval.
- Use fixed, physically derived clustering parameters only after validation and owner review.
- Report residuals and warnings explicitly if timing drift exceeds the project threshold.

## Immediate Next Step
- Begin with the project-specific task currently in motion: synchronization verification and DBSCAN-ready epoch diagnostics, with raw data and final DB treated as read-only artifacts.

---

## Full DBSCAN Readiness Audit — 2026-09-24

- Tags: #dbscan #multipath #clock-sync #audit #gate-2 #gate-3
- Env: `jsat_3d`. DB `output/jsats3d_2025_final.db` (22.4 GB) opened `mode=ro`. K: raw untouched.
- Scratch scripts/outputs only in `%TEMP%\jsats_audit\` (a01-a10). No repo code, DB, or raw file modified.
- Sources read: all 12 session journals, LONG_TERM_CONTEXT, Nebiolo & Meyer (2021) full text, `jsats3d.py` (beacon_epoch, multipath_data_object, multipath_2, multipath_classifier, sos, clock_fix, position.Deng), `notebooks/dbscan_multipath.ipynb`, our scripts (parse/adapt/extract_dbscan_features/dbscan_diagnostic/clock_tdoa_readiness/receiver_motion_readiness), legacy drivers.

### Corrections to earlier statements (this session and prior)
- Earlier chat claim that DBSCAN is blocked because ATS lacks SNR/NBW/FreqOff is WRONG for Kevin's method. Legacy has two DBSCANs:
  1. `multipath_classifier()` DBSCAN on (Amplitude, NBW, SNR) — needs absent fields; also first line `dat[dat.SNR > 0]` drops all 2025 rows.
  2. `clock_fix()` / notebook DBSCAN on (time, DDoA) of the metronome clock-bias series — needs only timestamps, receiver geometry, sound speed. Directly applicable to ATS.
- Our 09-23 diagnostic DBSCAN (lag, relative SigStr per receiver) is a third, new design; its degenerate k-distance percentiles are consistent with ~74% of bursts being single-detection.
- Onboarding note above said "five selected tags": final DB actually holds 37 tags (20 local beacons, 12 unassigned `1xxx/2xxx` beacons, 7F32, and study tags FFD3, FC36, 0B0A, 0AC6, 493F). C0FE, 7F0D, 0FC7 from the 09-22 journal are NOT present.

### Data facts verified
- `tblDetectionRaw` 59,892,757 rows, no indexes; every legacy per-tag query is a full 22 GB scan. Rows stored in contiguous per-file blocks, so rowid ranges give cheap bounded slices.
- `BitPeriod` populated (e.g. `240 03/31`, 238.06-241.97); `Threshold` populated; `SigStr` populated; `Event` 0/1 present. Pressure/Tilt NULL; RawTemperature 99.99 sentinel.
- `tblInterpolatedTemp` = `BB_TPU_Surface_t` (single surface sensor) from covariate table, NOT the approved DD_N temperature string. Coverage 06-17 to 09-17.
- `tblReceiver`: Z = -(config hydrophone depth ft x 0.3048), i.e., depth below surface, but `Ref_Elev='BM'` and `BM_Elev` NULL. WSEL range 856.9-866.4 ft (2.9 m). Legacy treats BM receivers as fixed Z; true vertical reference unresolved.
- `tblTag.TagType='raw'` for all; legacy branches on 'study'. Study tags FC36/0B0A/0AC6/493F have NULL pulseRate.
- Array-wide beacon candidates 1F14/1F38/1F5A/1F71/1F94/1FCD absent: `load_beacon_registry()` drops config rows without a receiver name, so parser never selected them.
- ZOI03, ZOI06 have no raw files ~06-17 to 06-27; CFD05 sparse (Aug-Sep only).

### Bounded diagnostics (06-20 00:00 to 06-22 00:00 UTC, 17 receivers, 1,017,205 local-beacon rows)
- Multipath structure: 732,275 bursts (1 s split, diagnostic only). 26.2% multi-detection. Later-arrival lag median 16.4 ms, p95 74.8 ms, p99 126 ms (about 24/111/187 m extra path at ~1480 m/s). 23 of ~285k later arrivals >200 ms.
- First arrival is strongest SigStr in 87.5% of multi-detection bursts; later arrivals median rel SigStr -16. SigStr carries real discriminating information.
- BitPeriod: later arrivals differ from burst-first by median 0.03, p99 0.26 units; weak discriminator at rank 1-3.
- Own-beacon inter-burst interval: median 61.5 s, IQR 59.1-62.8 s, 5-95% 29.8-68.2 s. Nominal 60 s is not exact.
- Metronome TDoA (legacy clock_fix construction, ZOI02 beacon 7D2D): 2,518 host epochs; 70-100% matched at 16 receivers. Apparent bias dominated by whole-second jumps (p01-p99 range ~4.6-7.6 s) plus a common ~-400 ms mode.
- Common-mode test: 149 of 151 >300 ms steps in ZOI08 occur at the same epochs in CFD09 and ZOI01 -> errors originate in ZOI02 (reference) timestamps. ZOI02 raised 2,449 one-second-adjustment rows in 48 h.
- After pairwise differencing (ZOI08-CFD09), 83.9% of epochs within 0.5 ms of rolling median; >2 ms outliers 0.4%.
- Alternative references (median over receivers, 48 h): ZOI02 164 jumps >300 ms; CFD04 21; ZOI08 28; CFD09 29; ZOI10 30; ZOI11 47; ZOI01/ZOI07 70. CFD04/ZOI08/CFD09 had zero one-second-adjustment rows.
- ZOI05 as a detecting receiver of 7D2D is chaotic (77% >5 ms outliers); flag for review.
- Study-tag PRIs measurable: FFD3 ~3.35 s, FC36 ~3.04 s, 0B0A ~3.02 s, 0AC6 ~3.20 s, 493F ~3.16 s; 13-25% of study bursts multi-detection.
- Receiver GPS fixes spread 10-81 m (p05-p95) in 48 h; interpreted as antenna/fix noise or antenna placement, not hydrophone position. Not usable as geometry.

### Conclusion
Not blocked by missing signal fields. Blocked by: (1) reference-clock whole-second/offset jumps not yet corrected; (2) unapproved reference choice given ZOI02 performance; (3) sound speed source (DD_N not loaded; June 5-16 uncovered); (4) vertical reference/receiver Z; (5) study-tag epoch assignment (legacy transNo rounding requires exact PRI); (6) array-wide beacons not parsed; (7) no ground truth / validation windows prepared; (8) upstream Spheros filtering undocumented.

### Parameter changes with rationale
- None. The 1 s burst split, 30 s match tolerance, 15-epoch rolling median, and 0.5/2/5 ms bins are diagnostic-only and not proposed as production parameters.

### Contradiction for owner (escalated, not routed around)
- Standing decision "ZOI02 central reference" contradicted by 48 h evidence. Needs owner review before sync design proceeds.

### Next steps
1. Owner review of this audit; decide reference receiver(s) and jump-handling approach.
2. Load DD_N temperature string into a copy DB; request June 5-16 temperature/WSEL.
3. Confirm hydrophone vertical reference (fixed vs surface-following) per receiver.
4. Fix beacon registry to include unassigned array-wide candidates; re-parse bounded slice.
5. Build pairwise-differenced (t, DDoA) series per receiver pair and apply fixed-eps DBSCAN derived from 0.5 ms budget; validate on held-out beacon epochs.
6. Add indexes on a working copy DB (never the final DB) before full-season runs.

---

## Cleanup — 2026-09-24

- Owner (user) requested removal of ancillary code and preservation of findings. Change budget exceeded with approval: 3 files edited plus deletions.
- Removed with `git rm` (recoverable from commit `e71b4ec`):
  - `scripts/dbscan_diagnostic.py` — study-derived standardization and eps sweeps conflict with fixed physically derived eps policy (System Prompt 7.4); superseded by audit.
  - `scripts/clock_tdoa_readiness.py` — status-stub CSV; superseded by audit TDoA findings.
  - `scripts/receiver_motion_readiness.py` — wrote blank placeholder fields only.
  - `scripts/extract_single_tag.py` — loaded full CSV into memory and wrote output into the K: raw data folder (violates read-only raw rule); adapter `--tag` filter replaces it.
- Removed tests `test_dbscan_diagnostic_does_not_filter_rows` and `test_dbscan_sweep_reports_without_filtering_source` with their imports in `tests/test_2025_contracts.py`.
- Deleted ignored artifacts: `output/zoi02_7d2d_timing.csv` (09-23 journal said removed; it was not), `output/current_clock_tdoa_readiness.csv`, `output/current_receiver_motion_readiness.csv`, stale `pycache` for removed modules, and `%TEMP%\jsats_audit\` scratch (a01-a10 scripts, slice pickles). Numeric results are preserved in the audit section above.
- Kept: legacy core and Kevin's drivers/notebooks (legacy-core rule), `parse_ats_raw_to_legacy.py`, `adapt_2025_to_legacy.py`, `extract_dbscan_features.py` (matches approved feature spec 7.2), remaining `output/current_*.csv` artifacts cited by 09-23 journal.
- Not removed, owner call: tracked `notebooks/dbscan_multipath-checkpoint.ipynb` (byte-identical to main notebook), `jsats3d_Project_Notebook-checkpoint.ipynb` (differs), tracked py37/py38 `.pyc` files, `.spyproject/`. All are Kevin-origin and ignored by `.gitignore` but tracked.
- Untracked `Nebiolo_Meyer_2021 (003).pdf` in repo root: do not commit (publisher copy).
- Validation: 11 tests passed (was 13; 2 removed). `py_compile` passed for adapter, parser, feature extractor, tests. `git diff --check` passed.
- LONG_TERM_CONTEXT.md updated: final-DB structural facts, corrected SNR/NBW limitation, multipath/PRI/GPS findings, owner escalation on ZOI02, read-only query methods.

---

## Reference Receiver Test — PM question (ZOI02 central vs ZOI09 for floats)

- Same 48 h slice (06-20/21), read-only, temp scratch deleted after run. No parameters changed.
- Distance to array centroid: ZOI03 12.3 m, CFD04 17.6, ZOI09 20.4, ZOI08 30.0, ZOI02 30.2. ZOI02 is not the most central by geometry.
- Local beacon heard by (>=200 bursts): ZOI02 17, CFD04 17, ZOI08 17, ZOI09 14 receivers. ZOI02 beacon has full slice coverage.
- Median over receivers (jumps >300 ms / share within 0.5 ms of rolling median):
  - ZOI02 beacon + ZOI02 clock: 164 / 0.841 (16 receivers)
  - ZOI09 beacon + ZOI09 clock: 13 / 0.678 (13 receivers)
  - ZOI02 beacon + ZOI09 clock: 20 / 0.830 (15)
  - ZOI02 beacon + CFD04 clock: 20 / 0.843 (15)
  - ZOI02 beacon + ZOI08 clock: 24 / 0.859 (15)
- Hybrid method: ToT = clean receiver's detection of ZOI02 beacon minus d/c. Beacon source and clock reference need not be the same receiver.
- ZOI02 common-mode steps: 187. With one-second-adjustment flag in preceding interval: 121 (65%); offset change: 36; counter restart: 12. About one-third are unflagged; magnitudes mostly +/-1 s, up to +/-5 s, tails near +/-400 ms. Flags alone cannot correct ZOI02; TDoA-based estimation still required.
- Residual ~20 jumps in hybrid runs are the detecting receivers' own jumps, expected to be handled by per-segment correction.
- Open: PM meaning of "floats" (surface ZOI units vs GPS-tracked CFD units) not confirmed.

---

## Pairwise Beacon DBSCAN — first run (M3/M4 diagnostic, no rejection)

- User directed: proceed with DBSCAN using available receivers.
- Added `scripts/beacon_pairwise_dbscan.py`; added test `test_pairwise_dbscan_flags_late_outliers_and_splits_on_jump`. 12 tests pass.
- Physics: delta_ia = (t_i - t_a) - (d_Bi - d_Ba)/c = (eps_i - eps_a) + (m_i - m_a). Beacon host clock cancels, so ZOI02 jumps drop out. Clean epochs are piecewise-smooth clock segments; multipath is isolated outliers (late if at receiver i, early if at anchor).
- Inputs: beacon ZOI02/7D2D, anchor ZOI09 (PM options combined). Bursts split at 0.5 x pulseRate (legacy rule). Sound speed = legacy `jsats3d.sos()` on mean of DD_N_0p5/1p5/9/18 (2019 method), read-only from K: temperature CSV. Distances from `tblReceiver` X/Y/Z (vertical reference unresolved).
- DBSCAN: features (t / (W x period), delta / 0.5 ms), Chebyshev metric, eps = 1, min_samples = 3. Neighbour = within W periods AND within 0.5 ms.

### Parameter derivation and change (required record)
- TIMING_BUDGET_S = 0.5 ms: System Prompt 9.2 budget; below shortest observed reflection delay (p05 1.47 ms). Fixed.
- MIN_SAMPLES = 3: legacy `clock_fix()` value; needs a neighbour on each side. Fixed.
- TIME_WINDOW_PERIODS: 2.0 -> 2.5. Rationale: 7D2D true epoch spacing 62.7 s vs 60 s nominal; one missed ping = 125.4 s > 120 s window, fragmenting segments. 2.5 covers one miss if true period <= 1.25 x nominal. Before/after on identical 48 h (06-20/21):
  - ZOI08 segments 143 -> 19; noise 5.47% -> 3.48%
  - CFD04 segments 214 -> 65; noise 11.17% -> 5.85%
  - ZOI10 195 -> 41; 8.18% -> 5.22%. CFD01 176 -> 215; 59.3% -> 31.5%.
  - Held fixed study-wide from this point pending owner review.

### Two-week run (2025-06-17 to 07-01, rowid 1,450,000-15,400,000)
- 511,257 detections -> 308,999 first arrivals -> 260,917 paired epochs, 17 receivers. Nothing rejected.

| Rec_ID | epochs | segments | noise % | clustered resid RMS (ms) | clustered >0.5 ms |
|---|---|---|---|---|---|
| ZOI08 | 18,351 | 357 | 7.41 | 0.0536 | 9 |
| ZOI10 | 17,880 | 490 | 8.99 | 0.0368 | 7 |
| ZOI07 | 18,209 | 523 | 9.80 | 0.0435 | 7 |
| ZOI04 | 17,875 | 526 | 10.33 | 0.0461 | 4 |
| ZOI01 | 17,121 | 732 | 14.32 | 0.0259 | 3 |
| ZOI11 | 17,625 | 731 | 15.36 | 0.0304 | 3 |
| ZOI03 | 7,819 | 561 | 25.21 | 0.0310 | 2 |
| ZOI06 | 4,192 | 482 | 25.07 | 0.0130 | 0 |
| CFD04 | 18,025 | 999 | 14.93 | 0.2335 | 684 |
| CFD09 | 17,255 | 1,374 | 19.54 | 0.2150 | 485 |
| CFD03 | 15,444 | 1,524 | 22.17 | 0.2026 | 336 |
| CFD08 | 16,617 | 1,377 | 25.22 | 0.2019 | 361 |
| CFD06 | 13,928 | 1,483 | 30.64 | 0.1962 | 166 |
| CFD07 | 16,024 | 1,587 | 32.76 | 0.1891 | 210 |
| CFD02 | 16,897 | 1,693 | 34.98 | 0.2098 | 309 |
| CFD01 | 13,185 | 1,548 | 42.32 | 0.0406 | 0 |
| ZOI05 | 14,470 | 1 | 99.98 | 0.0144 | 0 |

- Noise attribution (share of noise epochs late / in multi-detection bursts): ZOI08 0.060/0.863, ZOI07 0.165/0.850, ZOI10 0.205/0.165, ZOI04 0.337/0.745, ZOI01 0.362/0.655, ZOI11 0.479/0.421, CFD01 0.596/0.226, CFD02 0.586/0.386, CFD07 0.616/0.235.
- Long segments (>=30 epochs) median hours / drift us/s: ZOI08 1.37/-0.068, ZOI07 1.12/-0.038, ZOI04 1.04/0.375, ZOI10 1.06/0.467, ZOI01 0.94/0.417, ZOI11 0.91/0.116, CFD04 0.91/0.219, CFD09 0.86/0.296, CFD08 0.75/0.402. CFD01 has no long segments.
- Outputs (gitignored): `output/dbscan_pairwise/ZOI02_7D2D_anchor_ZOI09_{epochs,segments,summary}.csv`, `_delta.png`.
- ZOI receivers: segment residual RMS median 6.7 us; noise 7-15% (ZOI03/ZOI06 25%, partial coverage). Clocks resolvable at microsecond level.
- CFD receivers: residual RMS median 126 us; 150-690 clustered epochs >0.5 ms per receiver (flagged); ms-scale sawtooth ramps; long segments ~0.7-0.9 h median, suggesting periodic re-sync. Heavier fragmentation (999-1693 segments).
- Drift within long segments 0.2-0.6 us/s (ZOI07/ZOI08 ~ -0.05); ~25 us per ping, well inside budget.
- CFD01: two persistent modes ~21.5 ms apart (steady reflection, ~32 m extra path). A steady reflection is dense, so DBSCAN density alone does not label it noise. Needs design decision (keep earliest-mode cluster per regime).
- ZOI05: 99.98% noise; free-running clock drifting about +/-30 s smoothly and wrapping the +/-0.5 period match window. Clock fault, not multipath. Exclude/flag pending field review.
- Validation signal: 1,536 epochs where >=50% of >=4 receivers are noise simultaneously -> anchor-side (ZOI09) multipath/miss candidates, consistent with physics (anchor multipath makes delta early for all receivers). Noise late-share for ZOI07/ZOI08 only 6-17%, consistent with noise dominated by anchor-side events.
- Warnings: temperature CSV has no timezone, treated as UTC (UTC_Conv unresolved).

### Next steps
1. Owner review of fixed parameters and of CFD01 dual-mode handling rule.
2. Flag anchor-side epochs (common-mode noise) and remove them before judging receiver noise.
3. Map CFD segment breaks against Internal offset/one-second flags to confirm re-sync origin.
4. Repeat with a second beacon (e.g. CFD04/FA1B) as independent check.
5. ZOI05 field review; ZOI03/ZOI06 June gap confirmation.

---

## Tag Drag Period Environmental Data + Time-Zone Audit

- PM delivered `K:\...\2025_Data\Tag Drag Period\` (read-only): 21 HOBO CSVs and `TagDrag_WSE.xlsx`.
- Temperature: DD_N 10 depths (0.5, 1.5, 3, 4.5, 6, 7.5, 9, 12, 15, 18 m), DD_S 11 depths (adds 21 m). 5 min, no gaps >10 min, local 2025-06-02 13:45 -> 07-10 13:00, header `Date Time, GMT-07:00`. Covers the June 5-10 tag-drag window fully (1,477 complete DD_N rows).
- Drag-window stratification DD_N 0.5 m minus 18 m: median 0.36 C, max 1.44 C.
- Existing 4-depth file `Temperature/2025_Temp_String_Data_5min_interpolated.csv` is local PDT, not UTC: shifted +7 h it matches HOBO DD_N_0.5 exactly (100% of 6,781 overlap rows); as UTC match 2.3%.
- 10-depth vs 4-depth DD_N mean: median -0.03 C (p05 -0.11, p95 0.05); sound speed -0.11 m/s (p95 abs 0.40); timing effect over 100 m path median 5.7 us, p95 18.7 us. Negligible vs 0.5 ms budget for the 2019 mean-temperature baseline.
- DD_N vs DD_S full-profile mean differ median 0.21 C (p95 abs 0.45): spatial variation exceeds depth-sampling effect.
- WSE: signals `NSC.COL_S_ENT_WTR_LVL.F_CV` and `NSC.COL_N_ENT_WTR_LVL.F_CV` (collector entrance levels, ft, US/Pacific), 2025-06-05 11:15 -> 06-17 11:10, 5 min interpolated grid. Not the forebay signal `NSC.CZD_WTR_EL.F_CV` used in `tblWSEL`. 11 h overlap on 06-17: CZD minus S median +0.90 ft, CZD minus N +0.88 ft, one excursion to -0.68 ft. Cannot splice without owner-approved offset or the CZD signal.
- ATS raw time basis: header `File Start: 06/10/2025 08:59:00 -07z`; detection DateTime local PDT; GPS Fix rows UTC (16:00 rows between 09:00 detections). `tblDetectionRaw.seconds` = local wall time encoded as if UTC. `GPSFixTimeStamp` column is UTC while `timeStamp` is local.
- Consequence: detections, 4-depth temperature, HOBO, and WSE are all local and mutually consistent; pairwise DBSCAN output unchanged. Only labels were wrong. `scripts/beacon_pairwise_dbscan.py` relabeled (local start/end, `start_local`), timezone warning removed; re-run reproduced identical counts (260,917 epochs, 1,536 anchor-side candidates). 12 tests pass.
- Proposed `UTC_Conv = -7` (PDT; study window June-Sept has no DST change). Owner confirmation required before populating `tblStudyParameters`.
- Unverified: time zone of Master Covariate Table (`tblInterpolatedTemp`, `tblWSEL`).

### Parameter changes
- None. Time-zone relabel only.

### Answer for Kevin/PM (request full high-res temperature?)
- Not needed for the baseline mean sound speed (<=19 us p95 effect). Useful for the permitted System Prompt Section 10 depth-resolved experiment and for DD_S/DD_N spatial comparison. Low cost to request; not blocking.
- Blocking instead: forebay WSE (`NSC.CZD_WTR_EL.F_CV`) for June 5-16, or confirmation that collector-entrance level with ~0.9 ft offset is acceptable.

---

## 2026-09-25 entries moved

All 2026-09-25 work (ZOI05 correction, paper gap map, parser/adapter/DBSCAN fixes, v2 rebuild, two-week DBSCAN, clock-sync feasibility) is in `2026-09-25_timezone_rebuild_dbscan_feasibility.md` per System Prompt Section 13.
