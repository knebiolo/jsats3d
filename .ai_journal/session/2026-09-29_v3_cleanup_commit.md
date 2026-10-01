# Session: 2026-09-29 — v2 Cleanup and v3 Run-File Commit

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 09:29 local PDT (UTC-7) = 16:29 UTC
- Tags: #cleanup #v3 #run-file #git
- Branch: `ENM_jsat3d_edits`
- New session file: more than 60 min since the last 2026-09-28 entry (System Prompt Section 13). The v3 build, sweep and coverage results are in `2026-09-28_kevin_approvals_clock_resets.md`.

## Active Context
- Database: `output/jsats3d_2025_v3.db` (25.4 GB, 60,410,185 detections, 303 files, 20/20 receivers). This is now the only project database.
- Env: `jsat_3d`. Tests: 28 pass (with the uncommitted parser fix and its test in the working tree).
- Current focus: repo tidy-up after the v3 build.

## Short Summary
  - `output/jsats3d_2025_v2.db` (23.2 GB). It is superseded by v3: same code path plus the 18 ZOI03/ZOI06 daily files.
  - The v2-based output folders `output/dbscan_ddoa_0620/`, `output/dbscan_sweep_0620/` and `output/beacon_coverage/`. Superseded by `output/dbscan_jsats3d_2025_v3/`, `output/dbscan_sweep_v3/` and `output/beacon_coverage_v3/`.
  - All of these were local, gitignored outputs. No raw data (K:) touched.
  - `scripts/parse_ats_raw_to_legacy.py`: D-file regex and `--workers`.
  - `tests/test_2025_contracts.py`: daily-file test case. Held back with the parser because it fails against the old regex.

## Files Touched

## Decisions & Assumptions
- The test change is held with the parser fix so no commit contains a failing test.

- Unchanged from 2026-09-28: CFD05/CFD09 serial swap unfixed; Kevin approval list outstanding (reference clock, `master_receiver = ZOI08` vs 7D2D coverage ranking, `signal_proxies`, DBSCAN parameters, steady-reflection over-labelling, sound-speed source, Deng in-hull bug); legacy run blocked (no `jsat_legacy` env, `bm_elev` blank).

## Next Steps
2. Send the Kevin approval list with the v3 outputs.
3. Confirm the CFD05/CFD09 swap time with the PM.
4. Clock-jump correction (paper step 5) after Kevin's reference-clock decision.

## Follow-up — Shared Runtime and Coding Standards

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 11:04:02 PDT (UTC-7) = 2026-09-29 18:04:02 UTC.
- Tags: #shared-runtime #2019 #2025 #compatibility #coding-standards #validation
- Files touched: `jsats3d/jsats3d.py`, `scripts/run_data.py`, `scripts/legacy_pipeline.py`,
  `scripts/projectSetup.py`, `config/run_data_2019.toml`, `environment_legacy.yml`,
  `tests/test_2025_contracts.py`.

### Short Summary

- Preserved the 2019 legacy algorithm and routed both 2019 Teknologic and 2025 ATS
  runs through the same active `jsat_3d` Python environment.
- Replaced removed pandas `DataFrame.append` calls with equivalent `pd.concat` calls.
- Replaced positional `to_sql` arguments with keyword arguments.
- Closed SQLite connections in shared import/setup paths. This fixed Windows temporary
  database cleanup and prevents leaked handles during production runs.
- Added a temporary-fixture regression test proving 2019 import creates the required
  legacy tables under the shared runtime. Existing 2025 contract coverage remains active.
- Marked `projectSetup.py` and `environment_legacy.yml` as deprecated historical/
  compatibility artifacts. They remain available for reference and are not the supported path.

### Decisions and Rationale

- One supported runtime is preferred because the legacy algorithms can now load under
  current pandas and scikit-learn after compatibility-only substitutions. No timing,
  sound-speed, geometry, DBSCAN, or classifier parameter was changed.
- Legacy scientific behavior remains the authority. Compatibility edits were limited
  to API removal and resource ownership; no algorithm rewrite was attempted.
- NASA-style coding is now a standing rule: keep functions at or below 60 lines for
  new or touched code where practical, use explicit resource cleanup, fail loudly,
  preserve traceable behavior, and validate every edit with focused tests.

### Validation

- Full suite: 29 tests passed in `jsat_3d`.
- Current-runtime 2019 fixture import: passed; required legacy tables created.
- Core database smoke test: passed.
- Python compilation and `git diff --check`: passed.
- Raw K: data was not modified.

### Parameter Changes With Rationale

- None. No physical or algorithmic parameter changed.

### Blockers and Known Limitations

- Real 2019 data is not available in this workspace, so published-result parity is
  not yet demonstrated.
- Several pre-existing legacy/core functions exceed the 60-line standard. They are
  intentionally deferred to avoid a broad algorithmic refactor that could alter the
  published 2019 behavior.
- The 2019 fixture emits an existing empty-temperature warning because its minimal
  synthetic interval does not cover the interpolation edges.

### Next Steps

1. Keep new and touched functions within the 60-line standard.
2. Add focused tests before any legacy-core refactor.
3. Obtain and run the real 2019 validation dataset before claiming parity.

## Follow-up — GPS Interpolation and Interactive Diagnostics

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 11:08:47 PDT (UTC-7) = 2026-09-29 18:08:47 UTC.
- Tags: #gps #interpolation #plotting #diagnostics #2025
- Files touched: `scripts/gps_position_interpolator.py`, `scripts/cfd_gps_diagnostics.py`,
  `scripts/beacon_pairwise_dbscan.py`, `tests/test_2025_contracts.py`.

### Short Summary

- Added reusable piecewise receiver-position interpolation with `linear` and `cubic`
  methods. Both refuse extrapolation and return `NaN` outside each receiver's observed
  GPS interval. Duplicate timestamps keep the last fix.
- Extended GPS diagnostics for selected receivers, spatial tracks colored by elapsed time,
  interpolation comparison plots, piecewise position CSV output, and a self-contained
  Plotly HTML map using 15-minute points.
- Added a zoomable interactive DBSCAN before/after HTML review. Browser-only views are
  downsampled; complete CSV/static artifacts remain unchanged.
- Installed Plotly in the `jsat_3d` environment. Static Matplotlib outputs remain available.

### Real-Data Result

- Ran the updated diagnostic against the read-only `master_df_gps.csv`.
- Input: 1,114,887 GPS fixes.
- GPS exists for CFD02-CFD09 only. ZOI05, ZOI06, ZOI07, ZOI08, and ZOI09 have no rows
  in this source file; the script emitted a warning and produced no fabricated tracks.
- New outputs: `output/cfd_gps_v4/`, including `cfd_gps_spatial_tracks.png`,
  `cfd_gps_interpolation_comparison.png`, `cfd_gps_piecewise_positions.csv`, and
  `cfd_gps_interactive.html`.
- Raw K: data was read only.

### Decisions and Rationale

- 15-minute query spacing is used for diagnostics to control browser size, not as a
  replacement for detection-time coordinates. The interpolation API accepts arbitrary
  detection timestamps for later positioning integration.
- Linear and cubic are comparison methods, not an adopted production method. Cubic can
  overshoot between noisy fixes; selection requires validation against receiver motion
  evidence and positioning residuals.
- No receiver coordinates, clock values, sound speed, or DBSCAN parameters changed.

### Validation

- Piecewise interpolation test: passed for linear, cubic, endpoint recovery, and no extrapolation.
- Synthetic GPS diagnostics: generated static plots, CSV, and interactive HTML successfully.
- Synthetic interactive DBSCAN plot: generated successfully.
- Full suite: 30 tests passed.
- Python compilation, diagnostics, and `git diff --check`: passed.

### Blockers and Known Limitations

- The new interpolator is not yet connected to `position.Deng`; integrating time-varying
  coordinates into the legacy solver requires a deliberate schema/core design and validation.
- ZOI GPS comparison requested by Drew and Kevin is impossible from `master_df_gps.csv`;
  obtain the source file containing those receivers before interpreting them as stationary
  GPS controls.

### Next Steps

1. Decide whether dynamic GPS positions belong in an additive SQLite table or a solver input adapter.
2. Validate linear versus cubic on controlled receiver-motion or surveyed-position data.
3. Connect detection-time interpolation to positioning only after the schema and acceptance test are approved.

## Follow-up — Dynamic GPS Geometry Integration

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 11:19:32 PDT (UTC-7) = 2026-09-29 18:19:32 UTC.
- Tags: #gps #positioning #schema #legacy-core #validation
- Files touched: `scripts/adapt_2025_to_legacy.py`, `scripts/parse_ats_raw_to_legacy.py`,
  `jsats3d/jsats3d.py`, `tests/test_2025_contracts.py`, `config/run_data.toml`.

### Short Summary

- Added additive `tblReceiverGPS` staging to both 2025 ingestion paths. It stores raw
  GPS fix times and local-origin X/Y coordinates while leaving `tblReceiver` unchanged.
- Updated `position.Deng` to use linearly interpolated GPS X/Y at each detection time
  when that receiver has GPS fixes. Receivers without GPS retain static legacy X/Y.
- Z continues through the existing legacy WSEL/depth path. Cubic interpolation remains
  diagnostic-only and is not silently adopted for production positioning.
- Out-of-range GPS queries raise a prominent error rather than extrapolating.

### Decisions and Rationale

- Additive table chosen to preserve 2019 schemas and legacy tables. 2019 databases do
  not contain `tblReceiverGPS`, so their static behavior is unchanged.
- Linear interpolation is the production candidate because it is piecewise, local,
  transparent, and does not overshoot noisy GPS fixes. Cubic remains an experiment.
- No physical, clock, sound-speed, DBSCAN, or rejection parameters changed.

### Validation

- GPS table origin/filter fixture: passed.
- Dynamic solver lookup fixture: passed for linear interpolation, static fallback, and
  out-of-range refusal.
- Full suite: 32 tests passed.
- Python compilation, diagnostics, and `git diff --check`: passed.

### Blockers and Known Limitations

- Existing `output/jsats3d_2025_v3.db` predates `tblReceiverGPS`; it must be rebuilt or
  regenerated before production positioning can use dynamic coordinates.
- Convex-hull labeling still uses the static receiver hull. Dynamic hull labeling needs
  a separate validation step before changing reportable in-hull/out-of-hull results.
- Real GPS source contains CFD02-CFD09 only; no ZOI GPS was available for the requested
  stationary comparison.

### Next Steps

1. Rebuild a bounded 2025 database with `tblReceiverGPS` and verify row counts.
2. Add dynamic-hull validation before changing in-hull labels.
3. Compare linear/cubic positioning residuals on controlled data before any cubic adoption.

## Follow-up — Result-Preserving Legacy Performance Work

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 14:45:42 PDT (UTC-7) = 2026-09-29 21:45:42 UTC.
- Tags: #performance #sqlite #legacy #validation
- Files touched: `scripts/run_data.py`, `scripts/legacy_pipeline.py`.

### Change

- Added conditional composite indexes for existing raw detection query patterns:
  `(Tag_ID, Rec_ID, seconds)` and `(Tag_ID, seconds)`.
- Added conditional indexes for derived metronome tables after they are created.
- Indexes change query access paths only; no algorithmic or physical parameter changed.

### Validation

- Full suite: 32 tests passed.
- SQLite confirmed use of `idx_raw_tag_rec_seconds` for the beacon query.
- No Python process remained active after validation.

### Important Data-State Finding

- The local 2019 database currently contains only R01-R06 (20,838,274 rows), not the
  complete nine-receiver baseline (28,582,645 rows). The database file timestamp shows
  an interrupted rebuild during R06. This is unrelated to index creation; indexes do
  not delete rows.
- Legacy processing is stopped. No speed comparison or scientific result is claimed
  from this incomplete database.

### Next Steps

1. Remove/rebuild only the local database from all nine read-only K: receiver folders.
2. Verify 28,582,645 rows and nine receivers before processing.
3. Benchmark indexed and unindexed phase timings on complete local data.

## Follow-up — Vectorized Legacy Metronome Optimization

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 15:47:08 PDT (UTC-7) = 2026-09-29 22:47:08 UTC.
- Tags: #performance #vectorization #legacy #parity
- Files touched: `jsats3d/jsats3d.py`.

### Change

- Replaced row-wise host metronome enumeration with a cumulative lag-boundary calculation.
- Replaced repeated full child-dataframe scans with vectorized nearest epoch-window matching.
- Enabled all-core neighbor searches for `NearestNeighbors` and `KNeighborsClassifier`.
- No pulse-rate, DBSCAN, KNN neighbor-count, clock, geometry, or rejection parameter changed.

### Validation

- Full suite: 32 tests passed.
- Synthetic vectorized epoch assignment matched the legacy half-period window result.
- Current indexed local workflow is running against the complete 28,582,645-row database.
- Final speed and scientific parity comparison remain pending workflow completion.

### Safety

- Only local C: database/scratch outputs are written.
- K: 2019 source remains read-only.

## Follow-up — Local C: Cleanup

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 15:59:55 PDT (UTC-7) = 2026-09-29 22:59:55 UTC.
- Tags: #cleanup #local-only #legacy #products

### Removed Tracked Extras

- Deprecated setup/environment artifacts: `scripts/projectSetup.py`,
  `scripts/projectSetup_2018.py`, `environment_legacy.yml`.
- Unused legacy extras: `scripts/mulitpath_experiment_with_kats.py`,
  `scripts/temperature_assessment.py`, `scripts/temp_and_uncertainty.py`.
- Retired meeting-only diagnostics: `scripts/dbscan_parameter_sweep.py`,
  `scripts/beacon_coverage_report.py`.
- Active legacy core, 2019/2025 adapters, DBSCAN production diagnostic, GPS diagnostic,
  GPS interpolator, Kevin's positioning drivers, tests, and notebooks were preserved.

### Removed Local Generated Products

- Failed/incomplete 2019 recreation database, run manifest, and legacy scratch folder.
- Retired GPS diagnostic folders `output/cfd_gps/` and `output/cfd_gps_v4/`.
- Retired sweep and beacon-coverage folders `output/dbscan_sweep_v3/` and
  `output/beacon_coverage_v3/`.

### Preserved Canonical Products

- `output/jsats3d_2025_v3.db`, its run manifest, and build log.
- `output/dbscan_jsats3d_2025_v3/`.
- `output/positioning/`.
- Supplied paper PDF remains preserved as reference material.

### Safety and Validation

- No K: path was included in any deletion command.
- Full suite: 32 tests passed.
- Active path compiled successfully.
- `git diff --check` passed.

## Follow-up — 2019 RuntimeWarning Cleanup

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 18:52:40 PDT (UTC-7) = 2026-09-30 01:52:40 UTC.
- Tags: #warnings #2019 #temperature #validation

- Cause: the 2019 temperature interpolator evaluates 0.1 seconds beyond the
  observed profile range, producing all-NaN edge values and repeated NumPy
  `Mean of empty slice` RuntimeWarnings.
- Change: preserve the resulting NaN when no finite temperature exists, but
  check for finite values before calling `nanmean`.
- Validation: focused 2019 import test and full 32-test suite passed. Remaining
  output warnings are explicit timezone/HOBO data warnings, not RuntimeWarnings.

## Follow-up — Deng Negative-Discriminant Warning Fix

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 18:56:16 PDT (UTC-7) = 2026-09-30 01:56:16 UTC.
- Tags: #2019 #Deng #numerical-stability #warnings #validation

- Cause: Deng's quadratic time-of-arrival equation produced negative or invalid
  discriminants, then called `sqrt`, generating repeated `invalid value encountered
  in sqrt` RuntimeWarnings and NaN candidate solutions.
- Change: check the discriminant and quadratic coefficient before `sqrt`. For a
  negative/invalid discriminant or zero coefficient, retain paired traceable
  no-solution rows with sentinel coordinates and continue. No fabricated position
  is produced and no solution is silently discarded.
- Validation: core compilation, diagnostics, and full 32-test suite passed.
- K: data was not accessed or modified by this fix.

## Follow-up — 2019 Deng Checkpoint and Handoff Update

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 19:01:14 PDT (UTC-7) = 2026-09-30 02:01:14 UTC.
- Tags: #2019 #Deng #checkpoint #handoff #journaling

### Current State

- Local 2019 database was rebuilt from all nine K: receiver folders and verified
  at 28,582,645 detections.
- Surface metronome/KNN and clock fixing completed. Local checkpoint contains
  approximately 11.2M clock-fixed rows and 955,672 R01/FF76 secondary rows.
- Final deep Deng positioning has not completed. `tblPositions_Deng` does not
  yet exist, so no final 2019 parity claim is allowed.
- A prior parent run exited while an orphan Python worker remained active. Stale
  workers must be identified and stopped before any retry to prevent concurrent
  writes to the local C: database.

### Handoff Update

- Updated `LLM_Prompts.txt` with this checkpoint, numerical fixes, active-worker
  safety procedure, and the mandatory rule to journal every fix, warning,
  optimization, failed run, and cleanup action.

### Safety

- No K: data was modified. All database and scratch writes remain local.

## Follow-up — R01 Deng Success and Second Scalar-Access Fix (R02)

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 13:16:20 PDT (UTC-7) = 2026-09-30 20:16:20 UTC.
- Tags: #2019 #Deng #deep-receivers #pandas-compatibility #validation

### Milestone

- Deep receiver R01 (beacon FF76) completed Deng positioning for the first time:
  164,320 solution-B positions; median written to tblReceiver
  (X_t -21.050, Y_t 9.736, Z_t 248.090).

### Failure and Fix

- R02 (beacon FF74) stopped with pandas `Invalid call for scalar access
  (setting)!` — the same `.at` MultiIndex incompatibility fixed earlier in
  `host_receiver_enumeration`, in its second location: the beacon-tag epoch
  branch of `multipath_data_object.__init__`.
- Fix: replaced the row-wise `.at` loop with the equivalent vectorized logic:
  cumulative half-pulse-rate lag boundaries for host epochs, positional
  assignment of host transNo, `seconds.min()` per epoch, and ascending
  idempotent window writes preserving the original last-write-wins semantics.
- No pulse rate, window width, or epoch rule changed.

### Validation

- Core compilation and full 32-test suite passed.
- Full workflow restarted from the intact local database (no K: rebuild).


## Architecture Decision — One Modernized Legacy Code Path

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-29 15:56:59 PDT (UTC-7) = 2026-09-29 22:56:59 UTC.
- Tags: #architecture #owner-direction #legacy #2019 #2025 #parity

### Decision

- Kevin/Drew's intended product is one software path: the existing legacy `jsats3d`
  implementation, modernized in place to run both 2019 Teknologic and 2025 ATS data.
- Do not create a parallel 2025 solver, duplicate legacy algorithm, or separate product
  pipeline that can drift scientifically.
- Existing ingestion/adapters may remain format-specific where hardware schemas differ,
  but they must produce the legacy table contracts consumed by the same core functions.
- Performance changes must preserve 2019 outputs. Every optimization requires a parity
  check against the canonical 2019 database before adoption.
- New code must follow the project NASA-style rules: bounded functions, explicit resource
  cleanup, fail-loudly behavior, focused tests, and journaled rationale.

### Consequence

- Current speed work in `jsats3d.py`, `run_data.py`, and `legacy_pipeline.py` is aligned
  with this decision. The new GPS diagnostic module is analysis support only; dynamic
  GPS integration must remain an additive input to the existing positioning core.
