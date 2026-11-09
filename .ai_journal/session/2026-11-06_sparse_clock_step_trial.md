# 2026-11-06 - Sparse clock evidence and isolated ZOI06 small-step trial

## Metadata
- Author: GitHub Copilot.
- Local timestamp: 2026-11-06 18:14:42 PST (UTC-08:00).
- UTC timestamp: 2026-11-07 02:14:42Z.
- Tags: clock-sync, sparse-evidence, source-boundaries, diagnostic-only, holdout, owner-review.
- Files touched: scripts/clock_jump_pre.py, scripts/_2025_pipeline.py, tests/test_clock_jump_pre.py, and this session journal. Expanded change-budget approval remains applicable.
- Continuity: follows 2026-11-06_clock_sync_request_status.md. A new session file records the substantially later timestamp instead of silently treating the whole day as a short session.

## Request, approvals, and constraints
The user requested continuing toward defensible clock corrections and better fish positions. Continued with sparse-evidence validation rather than rerunning an unchanged median method. The existing median estimator and production defaults remain unchanged.

Explained the physical rule: independent beacon streams must show the same source-bounded arrival-time step, without competing reference events, before assigning a correction. Timing agreement remains at the existing 0.5-ms budget. Approximately 1 ms timing error can contribute about 1 m position error, with the actual effect dependent on geometry.

After finding a supported approximately 0.907-ms step that the existing 50-ms application minimum rejects, explicitly asked whether to test it in isolation. The user selected **Approve isolated correction validation**, using the existing 0.5-ms budget as the sparse-consensus minimum while leaving production defaults unchanged. Approval is for bounded in-memory timing validation, not production application, gate bypass, fish processing, or acceptance sign-off.

## Implementation
- Extended rapid_reference_consensus with include_isolated=False. Opt-in isolated source events can use the existing three-distinct-beacon consensus rule; ordinary rapid diagnostics remain unchanged by default. Brackets must remain source-bounded, contain no other target event or reference offset/restart, and span no more than two actual configured beacon periods. Three supported reference steps must agree within 0.5 ms. No automatic cluster-count tuning or fabricated arrivals were introduced.
- Extended the existing clock_rapid_reference_review to expose include_isolated and an explicit consensus_min_step=0.05 diagnostic argument. Recovered counter-only episodes can enter the opt-in isolated diagnostic, while CounterContinuation records remain excluded. Reports distinguish original and firmware-normalized coordinates.
- Added sparse_correction_trial in the existing clock pre-step module. It requires exactly the target receiver's observations and one independently supported source event. It copies the observations, subtracts the measured step from post-boundary TDoA values based on original SourceFile/SourceRow, and repeats the same reference checks. It never changes raw timestamps, database rows, or caller-owned dataframes. Its output expressly states that fitting-reference continuity is not independent timing truth or fish accuracy.
- Added tests for default-disabled isolated consensus, competing reference rejection, explicit sub-50-ms minimum, copied-coordinate non-mutation/reversibility, and refusing unsupported trials.
- A first integrated check caught an indentation error in the changed return block. Repaired it immediately and reran the same reference-producer guard test successfully. No unrelated code was changed.

## Bounded data and source safeguards
- Source database: output/10292026_Test/jsats3d_2025_earlyfish.db, opened SQLite mode=ro with query_only for every real-data query.
- Receiver/time interval: ZOI06, 2025-06-27 23:20 through 2025-06-28 01:20, in study PDT wall-time encoding.
- Original source file: C:/Users/Ethan.Muhlestein/OneDrive - Kleinschmidt Associates/Documents/jsats_data/raw_data/20250702_Receivers/SR18084_20250702.csv.
- Four reconstructed episodes: SourceRows87897,89151,91437,91910. Raw sources and all existing study databases remain unchanged.
- Initial deep-beacon references ZOI01/ZOI02/ZOI03 leave all four isolated events unresolved: all twelve nearest before/after reference brackets exceed the existing two-period gap rule. This is not solved by merely lowering the epoch-count requirement.
- Examined reception coverage of all nineteen existing non-target receiver beacon hosts in this bounded interval, then selected nearby references using recorded coverage rather than fitted step agreement. No diagnostic reference was added to the solving geometry.
- Nearby ZOI05/CFD01/ZOI04 references independently show no measurable jump at23:24:51 / SourceRow87897: median9.775161743164062e-06 s, between-reference range2.47955322265625e-05 s, all three brackets uncontested. This is evidence-only under the tested rule, not a production ledger update.
- ZOI11 was subsequently evaluated as another well-heard independent beacon. Every source reference was recovered from its original all-tag file, including competing resets omitted from the selected-tag database.
- Source-event guards reject contaminated or excessive-gap brackets. For example, CFD01 has its own events inside the23:57 bracket, and multiple other references miss timely brackets. Those measurements were not promoted into corrections.

## Supported small-step finding
The event at **2025-06-28 01:12:01.292634**, ZOI06 SourceRow91910, has three uncontested reference brackets:

| Reference | Measured step |
|---|---:|
| ZOI05 /7F91 | 0.0009069442749023438 s |
| CFD01 /B32A | 0.0009310245513916016 s |
| ZOI11 /1FF1 | 0.0009069442749023438 s |

- Median measured step: **0.0009069442749023993 s** (approximately0.907 ms).
- Between-reference range: **0.0000240802764892578 s** (approximately24.08 microseconds).
- Each reference has zero competing source offset/restart events in its supported bracket.
- This step exceeds the existing0.5-ms timing budget but is below the ordinary0.05-second (50-ms) application minimum. The minimums have different purposes and are not interchangeable.

### Parameter decision and measured effect
- **Production changes: none.** Existing ordinary/supplement minimum remains0.05 s.
- **Explicitly approved isolated diagnostic change:** consensus minimum0.05 ->0.0005 s for this sparse timing experiment only. Timing agreement/budget remains0.0005 s; max magnitude, source guards, and gap limits are unchanged.
- Rationale: test a directly supported timing-budget exceedance that the large-jump application threshold necessarily misses; require three independent uncontested source brackets, not a lower raw-DBSCAN rejection threshold.
- Diagnostic classification with ZOI05/CFD01/ZOI11: one supported small-step measurement, three unresolved events. This is not acceptance of those three unresolved events.
- Original and firmware-normalized measurements at this selected event agree; the target paired records have no per-record firmware adjustments. Other ZOI11 diagnostic observations carry documented adjustments and remain explicitly labeled.

## In-memory timing trial
- Applied the fixed0.0009069442749023993-second step only to a copied paired-observation dataframe, SourceFile matching the original file and SourceRow>=91910. Changed28 copied TDoA observations; source rows, original timestamps, and source databases remain intact.
- Fitting-reference maximum absolute step: **0.0009310245513916016 ->0.0000240802764892023 s** (0.931 ->0.024 ms).
- All three estimation-reference corrected brackets are within the unchanged0.5-ms budget. This is a local continuity test on the fitting references, not independent accuracy.

### Unused-beacon holdout
- Screened unused recorded CFD04/CFD08/CFD09 beacon hosts without changing the fitted offset. CFD08 provides an eligible118.057298-second bracket at the selected event; CFD04's185.325222-second bracket exceeds the unchanged two-period rule and was not promoted.
- Independently recovered CFD08's original source states. Its accepted holdout bracket has zero competing reference events and no per-record firmware adjustments.
- Scored the **original fixed offset**, not the offset from a new fit including CFD08:
  - Unused CFD08 observed step: **+0.0004627704620361 s**.
  - Post-trial step: **-0.00044417381286629925 s**.
  - Absolute step improves slightly and remains within0.0005 s.
- The CFD08 bracket was not used to fit the original offset. A separately saved three-reference diagnostic containing CFD08 is only a source-guard check; it does not redefine the scored original estimate.
- This supports local timing plausibility, but one holdout bracket does not establish full-season synchronization, absolute clock truth, or better fish positions.

## Artifacts
All new outputs remain under output/11062026_clock_recheck/:
- zoi06_sparse/: initial deep-reference isolated consensus and support CSVs.
- zoi06_sparse_nearby/: nearby-reference evidence-only episode and support reports.
- zoi06_sparse_zoi11/: independent ZOI11 reference review.
- **zoi06_small_step_trial/**: original-source evidence, original/normalized consensus CSVs, trial_before_events.csv, trial_after_events.csv, trial_before_support.csv, trial_after_support.csv, **trial_summary.json**, **holdout_cfd08_summary.json**, and **tdoa_small_step_before_after.png**.
- zoi06_holdout_cfd08/: original-source CFD08 evidence and its guarded bracketing observations.

The PNG was inspected directly. It displays actual bracketing observations in milliseconds, before versus in-memory correction, and clearly labels the existing0.5-ms budget, study PDT time, and absence of production/fish-position changes. It is not a full Arm A/B track comparison.

## Verification
- Full clock pre-step test file: **39 tests pass**.
- Full integrated clock gate/source-review test file: **11 tests pass**.
- Combined touched behavioral slices: **50 tests pass** on final runs.
- Project-conda py_compile passes for scripts/clock_jump_pre.py, scripts/_2025_pipeline.py, and tests/test_clock_jump_pre.py.
- Editor diagnostics report no errors in those three files.
- Source queries and scans are local/read-only; no commits, branches, source database updates, raw data writes, or standalone diagnostic scripts were created.
- C: capacity check during this work showed6,776,680,448 bytes free (approximately6.31GiB). This does not certify capacity for a new full database copy plus existing reserve/rollback requirements. No production copy/rebuild was attempted and no files were deleted.

## Blockers and next steps
- One previously missed **small** source-supported clock step now has a reversible local timing experiment and one unused-beacon check. Do not report a successful full clock repair or fish-position improvement.
- ZOI06's23:57 and01:01 episodes remain unsupported by three uncontested timely references. Other previously documented full-run timing issues remain unresolved; production gates have not cleared.
- Next accepted work is broader independent validation of the small-step evidence rule and source-order application on a new isolated database only after storage capacity and applicable gate/owner approvals are met. Preserve existing defaults unless adoption is explicitly approved with validation metrics.
- A clean full Arm B reconstruction, fresh fish filtering, and matched fish A/B accuracy/yield comparison remain unrun. Do not reuse Arm A filter tables or bypass unresolved timing to claim success.