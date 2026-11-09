# 2026-11-03 - Clock-only experiment storage preflight

- Author: Copilot, AI assistant using Copilot SDK in VS Code.
- Timestamp: 2026-11-03, session follow-up beginning 17:35 PST (UTC-08:00);
  corresponding UTC date/time begins 2026-11-04 01:35Z.
- Tags: clocksync, storage, read-only, experiment-preparation.
- Branch: `ENM_jsat3d_edits`.

## Approved direction
Full June 15-August 6 clock-only comparison of 1D5A, 26E4, 9F8C and B5ED.
Keep baseline receiver geometry and non-clock settings unchanged. Original database
and [baseline figures](../../output/10292026_Test/figures) remain untouched.
New outputs belong under [11032026_clocksync](../../output/11032026_clocksync).

The user approved more than two necessary implementation edits. Storage instructions:
investigate a smaller isolated database; do not delete anything. Stop for review
before solving when unresolved timing intervals affect fish.

## Measurements
- Available C: space: approximately 11.41 GiB at the sizing check.
- Required four study tags, master 7DB7, and deep beacons FFC7, 7D2D, 7DBC:
  **7,492,659 raw rows**.
- Actual 1,000-row samples per tag were inserted into in-memory SQLite using the
  baseline raw schema and four raw indexes.
- Extrapolated raw-plus-index footprint: **4.005 GiB**.
- This is a sample estimate, not a guaranteed bound. It excludes additional event
  evidence, reference-check beacons, metadata, generated tables, figures and temporary
  disk requirements.
- Existing receiver-GPS table has 1,114,887 rows in the donor database; metadata
  must be included in the capacity budget, not assumed negligible.

## Missing dependency and safe recovery
The unshifted Arm A database lacks `tblInterpolatedTemp`. Arm B retains 30,817
temperature rows covering 2025-06-02 13:45 through 2025-09-17 13:45.
The earlier audit verified reconstructed temperatures/sound speeds against saved
baseline values. Recovery can use a read-only donor or the original temperature
inputs; do not repair or modify the baseline.

Internal-event evidence on omitted tags must remain available to the experimental
quantifier. Merely selecting eight tags would alter event discovery and is insufficient.
Reference attribution may require more beacon data than the minimal selection.

## Failure and alternative
SQLite `dbstat` is unavailable in the current interpreter. The initial physical-table
size query failed explicitly with `no such table: dbstat`; read-only tag counts succeeded.
The in-memory physical-page sample was then used as a bounded alternative.

Current pipeline `--init-from` performs a full file copy followed by VACUUM.
That path is not appropriate with available storage. A compact destination would
need streaming construction with explicit capacity checks, preserved raw provenance,
reconciled counts and minimal necessary indexes, without a full-copy intermediate.

## Decisions and parameter changes
- No production parameter changes.
- No clock offsets, source databases, raw files or figures modified.
- No clock pipeline or positioning run started.
- Compact storage is promising, but total run capacity remains unverified.
- Do not claim the working database or clock fixes have been implemented.

## Files touched, verification and next steps
- Added this journal and
  [storage_preflight.json](../../output/11032026_clocksync/storage_preflight.json).
- Source queries used URI `mode=ro` and `PRAGMA query_only=ON`; sampling wrote only
  to in-memory SQLite.
- Next: account for retained event evidence/reference data and generated-output
  overhead; implement/test safe compact construction and the clock-only fixes.
- Test the baseline figure chain, then event grouping and segment-safe corrections.
- Apply the agreed clock-validation stop before any fish solutions.
- No commit or PR created.

## Implementation follow-up - compact construction
The earlier status above describes the initial sizing phase; this follow-up supersedes
the implementation status, but does not imply that a full run or comparison is finished.
- Added `scripts/compact_clock_study.py` and focused tests. This is a preparation
  command, not a clock or solver run and not a replacement pipeline.
- It streams the eight required raw tags into a new SQLite database, preserving
  original raw rowids and values, all metadata including GPS, and the donor temperature
  table. A separate `tblClockEventEvidence` retains Internal triggers and corroborating
  evidence from every original tag so tag reduction does not hide event messages.
- Source databases are opened read-only in stable read transactions. Existing targets
  and shifted source databases are refused. A completion manifest and destination row
  reconciliation/quick_check identify interrupted builds.
- Construction guard: reserve 4 GiB free and stop if the destination exceeds 6 GiB.
  These are explicit disk safety budgets, not physical or filtering parameters. They
  bound construction against roughly 11.4 GiB free; they do not certify sufficient
  capacity for generated solutions. Further capacity checks are required before runs.
- Only raw indexes needed for tag/receiver/time and receiver lookups are created;
  full-copy and VACUUM are not used.
- Three focused tests pass: rowid/event/metadata preservation and byte-identical source,
  capacity refusal, and previously shifted-source refusal. Initial Windows test teardown
  failed because sqlite connection context managers do not close handles; tests now
  explicitly close handles and pass.
- No clock offsets or solutions have been applied/run at this stage.

## Experimental clock changes and review policy
- `clock_jump_pre.find_events` now reads the preserved all-tag evidence table when
  present; ordinary full databases retain their original query path.
- Added opt-in same-file/same-InternalOffset trigger grouping with an explicit 1 second
  maximum **total** span (not chained). This is a conservative review convenience for
  sub-second duplicate messages, not proof that they represent one physical event.
  Counter restarts and missing provenance are never grouped. Start/end uncertainty,
  trigger counts, window endpoints and plateau spreads are exported in the candidate
  ledger. Grouped offsets are **not automatically applied**.
- The conservative policy also withholds common-mode events for reference attribution,
  unstable plateaus above the existing 0.5 ms warning budget, and above-maximum steps.
  Measured steps below the 50 ms application threshold but above the 0.5 ms warning
  budget are unresolved, not clean evidence-only. These thresholds have distinct roles.
- Backward jumps are also withheld: a timestamp-only SQL threshold can incorrectly
  include pre-event fish rows when the clock jumps backward and the two sides overlap.
  Mapping corrected validation windows alone does not repair source-order attribution.
  Source-file/row boundary reconstruction must be reviewed before such offsets are
  applied; the experiment does not pretend this ambiguity is resolved.
- A report-only event pass writes `tblClockJumpCandidates`, checks original-time fish
  overlap on the actual nine 3D receivers, and stops for review **before applying any
  offsets** if unresolved evidence affects fish. This is earlier than the requested
  pre-solver stop and avoids acting on uncertain boundaries.
- Approved-event application refuses an already shifted raw table. Raw source data
  remains in the original database, unchanged.
- New opt-in `jsats3d.clock_segments` maps both before/after sides of an applied jump
  into corrected receiver time, including overlaps from backward jumps. Residual
  validation uses the existing 0.5 ms warning budget and labels missing sides
  unverifiable instead of printing an apparent success.
- Experimental residual/fish interpolation no longer bridges >2 nominal beacon-period
  gaps or >=50 ms adjacent steps, and does not hold endpoint corrections outside clock
  coverage. Exact measured knots remain available. Legacy behavior stays the default.
- Five legacy adaptive DBSCAN passes remain unchanged. The experiment saves the
  prefilter clock data for reproducible filtering-loss diagnosis rather than silently
  disabling the filter.
- A later pre-fish pipeline gate also checks coverage, mapped-event validations and
  neighbor leave-one-out timing warnings above 0.5 ms. This is a warning screen,
  **not independent timing truth** or proof of physical fish accuracy.
- Standalone study-tag loading independently rejects unsupported corrections on
  configured solving receivers. Non-solving receivers are not added to the geometry.
- New configuration: `config/run_data_earlyfish_2026-11-03_clocksync.toml`.
  Baseline study, selection, receiver sets, solver/filter, figure and output settings
  were compared programmatically and match. The conda environment name, destination,
  overwrite protection and clock options intentionally differ. Water-column/bathymetry
  filtering is explicitly disabled. All experimental review outputs remain under the
  requested destination.
- Direct runner:
  `python scripts/_2025_pipeline.py config/run_data_earlyfish_2026-11-03_clocksync.toml --only clock_jumps`
  after the compact manifest reports complete. Do not use `run_data.py` rebuilding or
  full-copy `--init-from` for this experiment.
- Focused tests at this point: 17 clock/segment/gate tests plus 3 compact-copy tests
  pass. Python compilation and Pylance syntax checks pass for the changed core/pipeline;
  Pylance resolved callers remain compatible with extended jump APIs.
- Full database construction is in progress. No synchronization, fish solutions,
  baseline figure reproduction or result comparison has run yet. Remaining gating and
  reference-attribution work must be reported rather than bypassed.
- Before any accepted offsets are applied, the runner additionally requires the 4 GiB
  free reserve plus one database-sized rollback-journal allowance. If unavailable, it
  stops without applying offsets; the construction budget is not an update-space budget.

## Completed construction and actual review stop
- Compact construction completed: **4,166,221,824 bytes (3.88 GiB)**,
  7,492,659 raw rows, 159,893 all-tag evidence rows, 30,817 temperature rows,
  preserved metadata counts, and `PRAGMA quick_check = ok`.
- Ran the full-interval `--only clock_jumps` stage with the new configuration.
  The conservative review gate stopped as designed before any timestamp correction,
  synchronization or solutions. The pipeline status calls this `FAILED` because
  review stops use an explicit RuntimeError; this is a data-review block, not a crash.
- Candidate classification: **126 accepted candidates, 1,473 evidence-only,
  20,274 unresolved candidates** across the configured synchronization receivers.
  These are evidence-message hypotheses, not that many proven physical clock jumps.
  In particular, master receiver evidence cannot be quantified against itself and
  remains unresolved pending independent reference attribution.
- On the nine actual solving receivers, **367 unresolved windows overlap fish**;
  deduplicated overlap is **29,934 raw fish detections** in the study interval.
  These are potential timing exposure, not bad positions. Counting by window overlaps
  double-counts some detections; the distinct raw-rowid SQL count avoids that.
- All nine solving receivers have some flagged exposure, including the master.
- Verified the working raw table still has no `jump_offset_s` column and there is no
  `tblClockJumpEvents` applied ledger. No candidate offsets were applied.
- Persistent review results:
  `output/11032026_clocksync/clock_review/clock_jump_candidates.csv`,
  `clock_jump_summary.json`, `clock_review_gate.csv`, `clock_review_gate.json`;
  completion manifest and pipeline status are beside the working database.
- Stop here for user review as agreed. Next safe work is source-order/reference
  attribution and resolving evidence/filtering gaps, not automatically applying
  candidates or presenting a fish comparison that has not been run.

## Follow-up: user requested "review and compare"
- Compared the **actual Arm B applied ledger**, not the standalone report CSV:
  the database contains 696 applied events while `jump_report_2026-10-29/jumps.csv`
  contains 772 report candidates. These are different scopes and must not be conflated.
- All 696 applied events match the new candidates by exact receiver/time. Their
  measured offsets are identical within floating-point serialization
  (maximum difference 4.44e-16 seconds). The new acceptance policy retains 126 and
  withholds 570. This is a policy/evidence comparison, not a changed-offset experiment.
- Important implementation limitation: the 1 second grouping rule grouped **zero**
  real events. It excludes all restart flags, so it does not yet solve the dominant
  000/001 repeated-message case. Passing the synthetic grouping test did not establish
  the required real-data outcome; the intended root-cause fix remains incomplete.
- Reviewed the parser's actual rule: `InternalCounter in ("000", "001")` marks both
  consecutive counter messages as `CounterRestart`. This is not transition-based
  restart detection. Distinct triggering rows do not imply distinct physical jumps.
- Among 3,381 adjacent sub-second trigger pairs in all retained evidence, 3,377
  have the same source file/offset and counters 0 then 1. This audit covers the
  retained evidence span, not exclusively the fish study interval.
- Concrete ZOI01 June 16 09:01 example from original source provenance:
  rowid841077 counter1708/offset1208, rowid841078 counter0/offset2164
  at 09:01:05.770104, rowid841079 counter1/same offset2164
  at 09:01:06.162663, then counters2/3/4 with the same offset. SourceRows337368
  and337369 are adjacent. The first trigger belongs to omitted tag7F91 and survives
  correctly in the all-tag evidence table. Treating 0 and1 as separate events clips
  both surrounding beacon windows instead of measuring the outer plateaus.
- Most fish-overlapping blocks are missing/clipped measurement windows:
  161 clipped windows plus133 insufficient-epoch windows =294 of367.
  The remaining73 are measured timing-budget, backward-boundary or common-mode
  attribution warnings. They should not all be described as proven jumps.
- Master ZOI08 has1,335 unresolved candidates because no listener-versus-master
  series exists for the master itself. This is absent independent reference evidence,
  not proof of1,335 master jumps. Blanket master warnings contribute to the stop.
- Saved read-only outputs under `clock_review`:
  `prior_applied_vs_review.csv`, `close_trigger_source_pairs.csv`,
  `zoi01_cluster_source_example.csv`, `distinct_fish_timing_exposure.csv`,
  `clock_review_comparison.json`. A session-only helper generated these without
  database writes and was removed afterward.
- No new synchronized clocks or fish positions exist to compare with baseline figures.
  The gate was not bypassed. No source or production-code edits were made in this
  follow-up review. Recommended next implementation: source-order-aware 0->1 restart
  episode consolidation, retaining uncertainty/provenance; then independent
  master/reference attribution and rerun the review before the solver.

## Continued read-only investigation (explicit user choice)
The user selected "Continue read-only investigation without code changes". No
production code, configuration, raw data or database was changed during this stage.
Only diagnostics and this requested journal were written.

### Restart-episode hypothesis
- Tested consolidation **in memory**, requiring adjacent source rows, counter0->1,
  same file/offset, and <=1 second between recorded timestamps. This did not modify
  the parser or the experimental grouping rule.
- Found3,326 such episodes in the study interval. Using the episode's outer sides,
  clipped at the neighboring remaining episodes, gave1,990 measurable cases.
- Of these,1,239 measured steps were <=0.5 ms (plateau stability is a separate test).
  There were89 measurable >=50 ms steps, but only30 also satisfied both <=0.5 ms
  plateau-spread and <=5 second magnitude screens.
- The30 stable application-size hypotheses include4 on ZOI04/11, which are outside
  the configured comparison geometry. Only17 are on actual solving receivers.
- **None of those17 has a selected fish detection in its +/-300 second outer
  measurement window.** This means no immediate fish exposure in that window,
  not no later effect: a cumulative raw correction can affect later detections.
- The earlier ZOI01 June16 example reproduces a -0.406231 second aggregate step
  using4 pre/5 post beacon epochs and0.108/0.150 ms plateau spreads. This supports
  the episode interpretation, but does not certify source-order timing truth.
- Output: `restart_episode_outer_plateaus.csv`, `restart_episode_fish_exposure.csv`.

### Backward-jump boundary concern: tested, not assumed
- Reviewed335 previously applied negative jumps. Matched every ledger event to
  original source provenance, then checked selected fish rows on the same source
  file between neighboring trigger rows for disagreement between source order and
  the SQL timestamp update boundary.
- Found **zero selected-fish row mismatches** in this bounded check.
  Thus the overlap risk is real in the algorithm, but this investigation does
  **not** demonstrate that it damaged the four example fish. The earlier concern
  must remain a potential bug, not an observed explanation for their weird tracks.
- This check excludes other tags, cross-file ambiguity and wider event attribution;
  it is not a full proof of timestamp correctness.
- Output: `prior_backward_source_boundary_mismatches.csv` (empty, headers retained).

### Master/reference hypotheses
- Screened each master episode's +/-300 second outer beacon plateaus on every
  available listener.227 stable >=50 ms listener-step rows occurred around134 master
  episodes. Most listeners also had their own nearby Internal episodes.
- Three priority master/reference hypotheses had>=3 same-sign listener steps,
  <=150 ms between-listener magnitude range, and no listener's own episode in the
  measurement window:
  - June23 18:52:49: CFD01/ZOI01/ZOI03/ZOI11, median+1.000106 seconds.
  - July17 13:01:01: ZOI01/ZOI03/ZOI05, median+0.373520 seconds.
  - July24 13:08:04: CFD01/ZOI02/ZOI03/ZOI11, median+1.399309 seconds.
- These are **reference-attribution candidates**, not proof that the master jumped.
  A shared listener/reference event can look similar in a single-reference TDoA
  network. ZOI11 is independent diagnostic evidence only, not a new solving receiver.
- Outputs: `master_event_listener_plateau_hypotheses.csv`,
  `master_reference_priority_hypotheses.csv`, `clock_boundary_review_summary.json`.

### Most immediately fish-relevant review target
- Ranked the73 **measured** warning windows that overlap fish, separately from the
 294 missing/clipped-window warnings.
- Top window is ZOI02 June28 **00:56-01:06**, with382 raw26E4 detections and unstable
  beacon plateaus. This independently returns to the interval identified in the
  earlier three-day audit (remaining ~2 second transition with missing final knots).
- Follow-up priorities include ZOI01 June28 01:13:35-01:23:35 (35726E4 rows),
  ZOI03 June27 23:52-June28 00:02 (35526E4 rows), and ZOI02 in the same interval
  (35226E4 rows). Window counts overlap and do not measure bad solutions.
- Output: `fish_overlapping_measured_warning_priorities.csv`.
- Conclusion: repeated restart-message consolidation is supported but is not by
  itself the demonstrated fix for the fish-affected June28 problem. Prioritize
  matched multi-receiver/reference timing and upstream-vs-final filtering loss
  in that interval before applying a new correction or claiming better positions.
- The temporary session diagnostic helper was removed after execution. New fish
  solutions and a positional A/B comparison still have not been run.

## Pipeline consolidation and resolving blockers before positions
- User asked to generate positions/figures, and reiterated that diagnostic helpers
  must be removed and reusable changes must live in existing 2025/2019 pipeline files.
  Asked explicitly how to handle the unresolved timing; user selected
  **"Resolve timing blockers before generating positions"** rather than retaining or
  excluding unresolved detections in a diagnostic positional run.
- Removed the three new standalone implementation helpers and their compiled caches:
  `scripts/compact_clock_study.py`, `scripts/clock_study_gate.py`,
  `jsats3d/clock_segments.py`.
  Their reusable functionality now lives in existing `scripts/run_data.py`,
  `scripts/_2025_pipeline.py`, and `jsats3d/jsats3d.py`, respectively.
  The existing `scripts/clock_jump_pre.py` remains the pre-sync pipeline step.
  Regression tests remain under `tests/`; no standalone exploratory Python files remain.
- Compact preparation now uses:
  `python scripts/run_data.py config/run_data_earlyfish_2026-11-03_clocksync.toml --compact-from output/10292026_Test/jsats3d_2025_earlyfish.db --temperature-source output/ab_B_2026-10-29/jsats3d_2025_earlyfish.db --reserve-gib 4 --max-database-gib 6`.
  The existing destination is protected against overwrite; **do not rerun this command
  against the already completed database**. Experimental rebuilding/metadata proxy
  rewriting through ordinary `run_data.py` paths is explicitly refused.
- Fixed the real restart-episode grouping case: source-adjacent0->1 counters with
  the same file/offset and bounded total recorded-time span. Other restart types,
  file/offset changes and nonadjacent source rows remain distinct.
  Counter provenance absent from the existing compact evidence table is recovered
  by original rowid from the read-only baseline registered in the experiment policy.
  New compact builds retain these counter values directly.
- Actual full-interval rerun consolidated **2,813 restart episodes** on configured
  synchronization receivers (the earlier3,326 diagnostic hypotheses included extra
  nonconfigured receivers). Candidate counts now:124 accepted,2,057 evidence-only,
  16,879 unresolved. Fish-overlapping windows reduced367->316; deduplicated potential
  raw-fish exposure reduced29,934->27,492. This is less false event fragmentation,
  not a positional accuracy improvement. No offsets were applied.
- All25 available tests pass, including restart provenance guards, legacy jump
  behavior, compact safety, coverage gate and explicit2019 run-plan checks.
- Independently compared current clock_fix outputs with the Git HEAD core on
  synthetic2019 inputs, with no experiment policy, real five-pass DBSCAN and normal
  interpolation: exact CSV/dataframe parity for150 masterR05 rows and143 listenerR04
  rows. This does not certify a full historical2019 rerun. Initial test cleanup exposed
  uncollected SQLite handles; the corrected test invokes garbage collection and all
  explicitly identified temporary test directories were cleaned.

### Integrated June28 multi-reference review
- Added an optional `clock_review` step **inside the existing2025 pipeline**, disabled
  unless a diagnostic interval is supplied. It reads original/prior databases through
  read-only connections, creates no corrections, and saves relative timing plus stage
  coverage and four comparison PNGs under the requested output destination.
- Executed:
  `python scripts/_2025_pipeline.py config/run_data_earlyfish_2026-11-03_clocksync.toml --only clock_review --clock-review-interval "2025-06-27 23:45:00" "2025-06-28 01:30:00" --compare-clock-db output/ab_B_2026-10-29/jsats3d_2025_earlyfish.db`.
  Completed in2.8 minutes, exporting10,740 paired observations.
- Folder:
  `output/11032026_clocksync/clock_review/interval_1751067900_1751074200`.
  Four beacon hosts:ZOI08/7DB7, ZOI01/FFC7, ZOI02/7D2D, ZOI03/7DBC.
  Plots show each arm's unfit TDoA relative to its initial four-epoch median;
  arms have different timestamp coordinates/sample sets. They are **diagnostic clock
  plots, not new fish figures or synchronized timing accuracy**.
- Crucial finding: the remaining ~2-second transition is not exclusive toZOI02.
  Around01:05, nearly all listeners of theZOI08 beacon show an approximately
  -2-second step. In the opposite direction,ZOI08 as listener of the three deep
  beacons shows +2.000037,+2.000035,+2.000054-second changes.
  This strongly localizes a shared timing transition to the master/reference side;
  it is not adequately modeled as another isolatedZOI02 correction.
- Prior corrected curves retain this transition. ExampleZOI02 versusZOI08:
  +0.999985 seconds at01:03:23, -1.000043 at01:04:27, and -2.000033 at01:05:31
  (corrected receiver time). Multiple rapid master-side transitions complicate the
  original four-epochs-per-side300-second median windows.
- Verified original master Internal evidence:
 01:01:00.217196 offset244;01:02:01.741300 offset248;
 01:03:00.770119 offset252;01:04:01.647607 offset256;
 01:05:03.238021 offset260, with GPS/status evidence and offset-change flags.
- Read a bounded original source-file neighborhood aroundSourceRow1384388.
  The source offset changes100->104 (hex) betweenSourceRows1384382/1384384,
  immediately after a GPS Fix line. The first post-change raw source detection is
  omitted tag3664 at01:05:00.697801; the baseline's first retained evidence is26E4
  at01:05:03.238021/counter6/SourceRow1384388.
  This demonstrates why retained detection timestamps are not always the exact
  source-file boundary. No original source file was altered.
- In the105-minute review interval,ZOI02 prior corrected clock processing had94
  accepted upstream epochs with valid master joins but only67 final clock rows;
  the unshifted arm had93/80. These are different retained sample sets and do not
  prove the filter alone caused the discrepancy, but confirm unsupported evidence
  loss remains.
- Remaining implementation needs: multi-reference master-event reconstruction at
  true source-file boundaries, plus supported downstream knot recovery/validation.
  Rapid60-second events do not satisfy the current minimum150-second clipped side
  rule. Do not silently lower those requirements or invent a master correction from
  a smoothed300-second median. Independent-beacon consensus would be a new evidence
  method requiring explicit tests and a documented decision.
- The timing gate remains blocked. **No new fish positions or positional figures
  have been generated**, consistent with the user's chosen stop policy.
- Final validation after adding the integrated diagnostic regression: **26 tests pass**
  plus compilation and diff whitespace checks. The new test verifies that the optional
  reference diagnostic is disabled by default and leaves source database bytes unchanged.
  Final checks confirm no correction column/applied ledger and no temporary session
  diagnostic Python scripts remain.

## Master resolution attempt: retained median method
- User requested resolution. Offered a strict opt-in three-beacon consensus evidence
  method for rapid master events while preserving2019 behavior. User chose
  **"Keep the existing median method and continue investigating"**.
  No alternative estimator, lower minimum sample count or timing threshold was applied.
- Added source-state recovery to the existing optional2025 `clock_review` step.
  It scans relevant original source files read-only, including tags omitted by the
  baseline import. It reports exact source row/state transitions and previous source
  rows. Internal position differences remain diagnostics, not calibrated offsets.
- Recovered the original master offset-state boundary at
  June28 **01:05:00.697801**, SourceRow1384384, tag3664, versus the retained trigger
  01:05:03.238021, SourceRow1384388, tag26E4. Added a regression proving omitted-tag
  source transitions are retained by this diagnostic.
- The three deep-reference single-epoch changes around01:05 are
  +2.000037,+2.000035,+2.000054 seconds. Their19-microsecond range supports the
  master-side interpretation but is **not sufficient under the retained median rule**.
- Earlier master transitions disagree across beacons. ZOI01's master-listener
  change at01:03 is about0.000039 seconds, whileZOI02/03 show about-1.000 seconds.
  Original reference evidence also containsZOI01/02 offset changes at01:01 andZOI03
  at01:02. Independent-reference stability cannot be assumed across that episode.
- Executed the existing median quantifier separately against each of the three
  deep references for the00:55-01:12 interval, without changing its rules.
  **All15 measurements (five master events times three references) are unresolved**:
  rapid events are about60 seconds apart, so windows clip below the required150 seconds.
  Interior events typically have only1 pre/1 post beacon epoch, versus the normal
 3 clipped/4 full minimum. The01:05 event has1 pre/5 post on each reference.
- Thus widening windows would combine multiple events, whereas lowering the minimum
  or accepting one-epoch consensus would change the user-retained method. Neither
  was done. These observations cannot become a defensible per-event correction by
  additional reruns of the unchanged quantifier.
- Saved integrated diagnostic results:
  `output/11032026_clocksync/clock_review/interval_1751072100_1751073120/`
  including `master_existing_median_review.csv`,
  `ZOI08_source_state_boundaries.csv`, timing PNGs and manifest.
- Final **28 tests pass**, including the rapid-event rejection and all-tag source
  boundary regression, plus compilation/whitespace checks. Existing2019 and median
  behavior remain covered; no standalone diagnostic scripts were created.
- **Timing is not resolved and no offsets/positions have been applied.**
  To resolve these rapid events while keeping the median method requires additional
  sufficiently frequent stable timing evidence or independently calibrated receiver
  clock/GPS records. Otherwise an explicitly approved alternative evidence method is
  required. Do not label the recovered source boundary as a completed physical correction.

## Approved rapid-event evidence extension
- User requested completion of the PM experiment and explicitly approved:
  **"Approve the opt-in rapid-event method, validate it, and proceed to comparison
  only when supported"**.
- Isolated events keep the existing median method. Rapid-event measurements may
  use independent-beacon agreement at recovered source-file boundaries, with competing
  reference events, missing samples and disagreement left blocked. No baseline or
  2019 default change is authorized. This approval does not waive the final timing gate.
- Implementation/validation is in progress; do not claim new corrections, fish
  solutions, improved accuracy, or a completed comparison before execution verifies them.

### Rapid-method validation and original-reference safeguards
- Implemented the approved opt-in three-beacon rapid-event measurement in the existing
  `clock_jump_pre.py` and optional `_2025_pipeline.py` review step. It is currently
  **measurement/review only**, not an applied master correction or a replacement for
  the ordinary `clock_jumps` candidate/application stage.
- First real June28 00:55-01:12 run classified five master offset-state events:
  - 01:01:00.217196, SourceRow1383966: unresolved; no uncontested three-reference bracket.
  - 01:02:00.125277, SourceRow1384077: unresolved; only one uncontested reference bracket.
  - 01:03:00.770119, SourceRow1384190: unresolved; ZOI01 measured about +0.000039 seconds
    while ZOI02/03 measured about -0.999974/-0.999969 seconds.
  - 01:04:01.647607, SourceRow1384283: accepted measurement +1.000019 seconds;
    independent reference steps +1.000016/+1.000031/+1.000019 seconds.
  - 01:05:00.697801, SourceRow1384384: accepted measurement +2.000037 seconds;
    independent reference steps +2.000037/+2.000035/+2.000054 seconds.
  Accepted means agreement under the experimental evidence rule, **not applied,
  independently certified clock accuracy, or improved fish positioning**.
- Tightened the producer to scan original all-tag source files for each independent
  reference, rather than relying only on retained-tag database flags. Exact offset
  transitions and parser-standard counter restarts now guard the reference brackets.
  Added a regression proving an omitted reference event blocks otherwise unanimous
  beacon measurements.
- The first stricter trial also treated every InternalGroup3 state transition as a
  competing event and rejected all five events. Inspection showed these uncalibrated
  states change across the reference minute boundaries; this is not itself a
  parser-standard clock-jump candidate. Removed that unsupported classification,
  retaining group-three transitions in the diagnostic source-state CSVs.
  No offsets were applied during either trial.
- A further real rerun with complete original-reference offset/restart checks returned
  **two accepted measurements and three unresolved events** again. Original master
  recovery retained 1,653 state/restart rows, including unchanged-state counter restarts.
  Original receiver files and source databases were opened read-only.
- Target beacon brackets now also reject additional counter-only target episodes.
  The existing verified source-adjacent counter0->1 grouping prevents treating the
  second message of the same bounded restart episode as a separate jump.
  Missing source files/parseable evidence are explicitly unavailable, not a clean pass.
- Reference-support reports now retain pre/post source rows and reference times even
  when a competing event blocks acceptance, plus the competing-event count/timestamps.
  Review manifests include per-reference original-source coverage.
- **33 tests pass**, including new omitted-reference producer, unchanged-state restart,
  and additional-target-event regressions. Pylance syntax checks passed for both modified
  pipeline files; `git diff --check` passed. No standalone diagnostic scripts were added.
- The experiment remains **blocked before application and fish solving**. The three
  unresolved rapid events cannot be approved by widening brackets across other events,
  dropping a disagreeing reference, or relabeling the source offset-code increment as a
  calibrated time jump. Source-order application/master-listener reconciliation and
  the full timing/coverage gate are still prerequisites for an actual correction run.
  The four-tag positional comparison and unaffected controls have **not** been produced.

## Continuation: documented per-record firmware adjustment
- User requested finishing the experiment to see results. Continued investigation
  under the existing no-unsupported-fish-timing rule; no baseline or raw timestamps changed.
- Found a concrete within-minute artifact in the original master file. At SourceRow1384082
  (01:02:01.056416), flag `1` appears and the raw DateTime position becomes approximately
  one second earlier than Internal group2. The state ends at SourceRow1384097
  (01:02:08.101728). The ZOI01 beacon at SourceRow1384084 falls inside it; the other
  two reference-beacon measurements do not. This makes raw adjacent beacon steps disagree.
- Re-read the supplied authoritative instrument guide at
  `K:\Jobs\5662\001\Background\Instrument Information\ATS_3017_Internal_Column_Guide.txt`,
  groups2/4a. For SR3017 firmware10.62, `1` documents an individual detection-time
  adjustment of -1 second; `>` documents +1 second. Group2 is only a hundredth-second
  counter-position cross-check, not a sub-millisecond magnitude estimator.
  Group3 is the minute of the most recent GPS/clock reading, not a calibrated jump.
- Added a strict diagnostic decoder inside existing `clock_jump_pre.py`. It requires
  SR3017, 10.62-family firmware, file format2.0 and agreement between the flag and
  circular15-second counter position within the counter's truncation precision.
  It rejects inconsistent provenance/flags instead of guessing. Counter differences
  of -0.99 seconds are consistent with truncation; reversal is exactly the documented
  one second, not a noisy counter-derived fractional correction.
- Existing optional2025 review now exports **both** unchanged raw measurements and
  a separately labeled firmware-normalized diagnostic. This reverses documented
  per-record transformations in the diagnostic TDoA coordinate only; reference-event
  and source-row guards remain in their original coordinates. It does not declare
  raw GPS timestamps wrong or authorize automatically reversing firmware corrections
  in the production raw table.
- Real June28 remeasurement changes one paired observation:
  ZOI08 listening to ZOI01 at SourceRow1384084, raw TDoA-1.572075 seconds to
  diagnostic TDoA-0.572075 seconds. The01:03 master measurements become
  -0.999961/-0.999974/-0.999969 seconds, a13-microsecond range.
  Firmware-normalized diagnostic classifications: **three accepted measurements,
  two unresolved**, versus two/three on unchanged raw measurements.
  The01:04/+1.000019 and01:05/+2.000037 measurements remain unchanged.
- The01:01 event still has competing reference resets and a missing source-bounded
  third beacon epoch. The01:02 event still lacks a third source-bounded epoch and
  the available normalized steps are not unanimous. These events cannot be marked
  resolved by this diagnostic.
- Corrected a further diagnostic guard false positive: original source-adjacent
  counter0->1 rows with identical offset and sync-status are the same restart
  continuation even when the next detection arrives more than one second later.
  This is tracked explicitly as `CounterContinuation`, not a changed global grouping
  threshold. Nonadjacent rows, changed offset/status, and missing first records remain
  distinct candidates. Verified the real01:02 continuation is1.616 seconds later.
  Existing conservative median/grouping behavior is unchanged.
- Read-only same-transmission differencing against other listening receivers was also
  checked. Around the unresolved events it does not yield unanimous independent
  sub-0.5ms evidence; it was not promoted to an applied estimator.
- Persisted diagnostic observation/event/reference CSVs and
  `ZOI08_firmware_adjustment_diagnostic.png` beneath the existing June28 review folder.
  The figure shows original versus normalized TDoA, **not synchronized clocks or fish
  positions**. Inspected the generated figure and reran the integrated review successfully.
- **35 tests pass**, including documented adjustment direction, circular counter wrap,
  incompatible firmware/model/format, inconsistent counter evidence, and restart
  continuation offset/status/source-adjacency guards. Pylance syntax checks passed;
  both production rapid-consensus call sites are signature-compatible; whitespace
  validation passed. Reverified no `jump_offset_s` column in the experiment database.
- Full-period fish solving remains blocked. A restricted supported-data comparison
  would change the user's previous no-exclusions decision and requires confirmation;
  no silent exclusions, relaxed timing budget or fabricated clock offsets were used.

### Full-period stop rule retained; broader existing-beacon search
- Offered an explicitly restricted supported-data pilot versus retaining the original
  full-period stop rule. User selected **`full_resolution_only`**. No excluded-period
  pilot or fish solution was started.
- Searched the other configured beacon hosts using indexed original detections and
  the compact all-tag candidate evidence as a **provisional screen only**. All19
  nonmaster receiver beacon hosts were considered, not just the three deep hosts.
  The screen did not produce three uncontested agreeing source-bounded references
  for either remaining June28 event.
- At01:01, provisional uncontested measurements included ZOI05/7F91 at-1.681966s
  and CFD03/B36A at-1.593299s, which disagree by about88.7ms, far above0.5ms.
  Several otherwise similar approximately-1.599s measurements belong to receivers
  with their own competing reset evidence and cannot be accepted as stable references.
  Several other hosts have missing paired beacon observations in the required brackets.
- At01:02, provisional normalized measurements included ZOI01 +0.000036s
  (its own competing candidate evidence), ZOI02 +0.017330s, ZOI04 +0.000052s,
  CFD04 +0.001265s, CFD08 +0.001283s and CFD09 +0.000366s.
  These do not form three uncontested measurements agreeing within0.5ms.
  This screen is not complete original-source certification for every host; it is
  already insufficient to promote additional offsets.
- Recovered full original all-tag source states for potential listening-clock controls:
  CFD04 has its own offset/reset at01:01:00.021898; ZOI07 at01:01:00.220432
  and01:02:00.625700; ZOI09 at01:01:00.191422 and a separate retained counter1
  candidate at01:01:01.829759. These cannot simply be declared unaffected clocks.
- Also checked all39 configured beacon tags for original master detections in the
  27-minute padded review interval. Besides the configured receiver beacons,7F32
  has29 master detections but no host in the20-receiver table. Prior records identify
  it as the CHN03 beacon, not an extra surveyed/local clock host that can be silently
  added. No geometry, raw selection or unknown host mapping was changed.
- A first per-host screen was stopped after query-plan inspection showed repeated
  full receiver flag scans were unnecessarily expensive. The replacement screen used
  the already preserved all-tag compact evidence for provisional flags and original
  indexed beacon detections; it completed. The stopped/read-only helper changed no data.
- The approved single-event three-reference evidence rule still cannot support
  full-period application from these observations. Additional independent clock
  evidence or a separately reviewed reconstruction method would be needed; repeated
  reruns or silent threshold relaxation are not a resolution.

## Joint simultaneous-event reconstruction (continued November4)
- User approved **`joint_reconstruction`**, then requested continuation on November4.
  The full-period no-exclusions/stop rule remains in effect.
- Added a read-only joint diagnostic in the existing `clock_jump_pre.py` and existing
  optional2025 review step. The opt-in experiment config enables it; no2019 default
  or existing raw-application path was changed.
- The model shares source-bounded event magnitudes and relative linear drift across
  the four beacon networks. Per-receiver/beacon intercepts absorb constant acoustic
  delays. Fixing one receiver's linear drift only chooses the common-drift coordinate;
  it does **not** declare that receiver's event magnitudes zero.
- Before interpreting a fitted magnitude, the diagnostic checks the design's null
  space for that event. Individually unidentifiable events are reported as unresolved
  with a missing magnitude, never as an acceptable minimum-norm solution.
- Each beacon network is omitted in turn. A supported joint measurement requires
  identifiable estimates whose heldout-beacon range is at most0.5ms, at least three
  usable heldout beacon brackets, and every usable receiver bracket residual at most
 0.5ms. A median cannot hide a disagreeing receiver bracket.
  Brackets cannot cross another target event and cannot exceed two beacon periods.
- The robust fit uses the established0.5ms budget as its soft-loss scale; this does
  not lower the acceptance threshold or certify rejected/multipath observations.
  Optimizes only the identifiable parameter subspace to avoid redundant null-space
  fitting. Failure to converge is explicit.
- Added synthetic tests for known simultaneous network reconstruction, independent
  beacon holdouts, common-simultaneous-event ambiguity, heldout disagreement, hidden
  receiver disagreement and missing/ambiguous source provenance.
  The initial complete suite passed39 tests; the subsequent hidden-disagreement
  regression also passed in the focused21-test clock pre-step suite.
- Started the full configured synchronization-receiver June28 diagnostic, including
  original all-tag source scans and complete source/firmware provenance for all four
  beacon networks. Results are pending; no jump offsets or fish positions were applied.

### Joint validation results and shared self-reception delay check
- The original full17-receiver joint run completed in16.2 minutes:1,316 paired
  observations,109 parameters, rank105. All27 interval events remained unresolved
  under independent-beacon validation, including the five master candidates.
  The01:01/01:02 master estimates changed by8.390/8.548ms across beacon holdouts,
  above the0.5ms budget. Full-fit residual p95 was6.606ms.
- Observations showed that a host's own first beacon reception can contribute a
  common per-transmission delay to every listener TDoA. This is a model hypothesis,
  not a proven physical explanation for each residual. Added explicit free
  per-transmission nuisance terms and validated with **same-transmission listener
  differences**, which cancel those terms. They are not raw clock corrections.
  Synthetic regression verifies a shared17ms self-reception delay is not attributed
  to a receiver jump. A host's own withheld beacon cannot alone validate its clock
  jump when self-reception delay is unknown.
- Reran against the saved verified-source observations, avoiding repeated source
  scans. Updated event/holdout CSVs, model summary and review manifest consistently.
  Model now has211 parameters, rank203, residual p95 **1.797ms**.
  This smaller residual is not proof of correct timing or positional improvement.
- Master estimates in this diagnostic coordinate:
 01:01/-1.593198s,01:02/-0.000001s,01:03/-0.999856s,
 01:04/+1.000000s,01:05/+2.000084s.
  All five remain **unresolved joint measurements**. The first two have only two
  independently validating external-beacon brackets, not the required three.
  Later events have three external beacons but some heldout listener differences
  exceed0.5ms. These do not override earlier three-beacon rapid-method evidence.
- Added worst-control receiver and exact pre/post source rows to heldout reports,
  preserving the evidence needed to investigate failed brackets rather than masking
  them with a median. Examples:
  - ZOI08 versus ZOI07 on ZOI03 beacon: a19.871ms heldout discrepancy at01:03.
  - ZOI08 versus CFD08 on ZOI02 beacon:54.339ms at01:04 and54.476ms at01:05.
  - ZOI08 versus ZOI03 on ZOI01 beacon:3.546ms at01:01.
- Inspected original raw candidate arrivals for these exact controls. At CFD08,
  ZOI02-beacon transmission12 has arrivals1.444118s and about1.498863s relative
  to the host, but transmission13 has only the later~1.498503s arrival; transmission14
  returns to~1.443930s. This is consistent with a missing early arrival/late reflection,
  not proof of another receiver clock jump. It cannot be repaired by inventing an
  absent early timestamp. ZOI07's ZOI03-beacon observation changes~20ms between
  transmissions11/12 with only one retained arrival each.
- Diagnostic receiver-subset checks (existing solving receivers, then a smaller
  local subset) did not pass all event holdouts. They changed no geometry or fish
  selection and are not a proposal to discard fish periods.
- A test originally injected an inconsistent observation on an event's own hosted
  beacon and expected that host event's external-listener validation to fail.
  With host-reception delay explicitly unknown, that assertion did not test the
  intended condition. Corrected the fixture to inject disagreement on the target
  receiver's independently hosted beacon; rejection and worst-row traceability pass.
- **41 tests pass**, including host-delay separation, unknown common simultaneous
  jumps, independent holdouts, incompatible source evidence and legacy regressions.
  The experiment still has no applied offsets or new fish positions. Resolving
  individual beacon reflections must preserve normal multipath filtering and cannot
  be achieved by relabeling every inconsistent sample as a clock event.

### Continuation: unchanged event-segment DBSCAN and an additional beacon
- User requested continued clock corrections at10:19 November4. Added an independent
  support diagnostic in existing `clock_jump_pre.py`, wired into the existing joint
  review. It uses the existing `beacon_pairwise_dbscan.cluster` and its unchanged
  0.5ms/three-sample defaults on same-transmission listener differences.
  It never takes fitted clock offsets as inputs, never crosses either listener's
  original event regime, and never fills absent arrivals.
- Exported `beacon_event_segment_support.csv`:8,006 pair observations classified
  as4,959 clean,2,962 noise and85 steady-reflection observations. These are overlapping
  pair comparisons, not unique detections or fish positions. Noise is not proof
  of reflection; most rapid regimes have only one observation and cannot independently
  satisfy the existing three-sample filter.
- Specifically, the failed01:02 beacon brackets on CFD02/CFD07 and01:03 brackets
  on ZOI07/ZOI03 are one-observation regimes classified as noise. The54ms CFD08
  bracket at01:04 is likewise one observation per side. Filtering cannot create
  the missing independent evidence without relaxing requirements or importing data.
- Tested the already recorded extra beacon7F32 with existing receivers as acoustic
  time references, without assigning an unknown CHN03 clock or adding geometry.
  ZOI07 had insufficient reception coverage; repeated the bounded diagnostic with
  CFD08/CFD04, which have better recorded7F32 coverage. These add219/171 matched
  observations, respectively, to the original joint networks.
- Neither alternative provides any usable heldout7F32 bracket across the five master
  boundaries under the existing source-bounded/two-period rule. Original master7F32
  detections jump from00:59:58 to01:02:25, and from01:03:23 to01:05:25; these cannot
  independently isolate every intervening reset. All five joint master measurements
  remain unresolved. No extra beacon was added to the compact database or production
  correction configuration.
- Retained `additional_7F32_reference_review.csv`, the CFD08/CFD04 observation and
  holdout CSVs in the existing June28 review folder. Helpers reuse existing pipeline
  functions and were executed inline; no standalone exploratory scripts were created.
- Added regression tests proving event-segment diagnostics cancel common host
  reception delay and leave sparse regimes unclassified. Added explicit rejection
  of invalid transmission IDs in the joint diagnostic.
- No offsets, synchronized clocks or new fish positions have been applied.
  The full-period no-exclusions rule remains enforced. Current disk free is about
 6.29GB, also below the existing4GiB-plus-database rollback-reserve requirement for
  eventual application; no existing data was deleted and no reserve was waived.

### Original downloaded master and missing host self-receptions
- Located the original uncleaned master file:
  `K:\Jobs\5662\001\Data\DataTrans\2025_Data\downloaded_data\20250702_Receivers\SR25311_250618_153601.csv`.
  Read it without modification. Exported the1,962 all-tag detections in the17-minute
  interval and compared the440 selected-tag detections with the compact experiment.
  All440 match exactly on timestamp/tag/Internal, with no missing or extra selected
  detections. This interval's missing support is not attributable to cleaning the
  master file. Original/cleaned source row numbers differ by one; their row coordinates
  must not be interchanged.
- Important refinement: the master **does** hear the ZOI03 beacon at01:01:41.748547.
  Its conventional host-based pairing omitted that transmission because a matching
  ZOI03 self-reception was unavailable. Therefore "missing source-bounded beacon
  epoch" in the original diagnostic means missing usable **paired** epoch, not
  necessarily an absent master detection.
- Re-paired the same7DBC transmissions against existing receiverZOI05, with its own
  source-bounded clock events retained as joint unknowns. No beacon host identity,
  survey geometry, fish selection or raw timestamp changed. This recovered370
  paired network observations and a third independent beacon bracket for the first
  two master events. ZOI07 was also tested as a reference but did not recover those
  early brackets because its own reception coverage was insufficient.
- All five master joint measurements still fail the unchanged0.5ms validation:
  worst7DBC control discrepancies are9.356ms at01:01 and7.347ms at01:02.
  Later discrepancies remain19.871/1.105/3.380ms for01:03/04/05.
  These paired observations improve evidence coverage, not demonstrated correction
  accuracy; a missing host self-reception cannot justify forcing another control
  receiver's reflected/noisy observation to pass.
- Saved alternate7DBC ZOI05/ZOI07 observation/event/holdout CSVs in the existing review
  folder. These are labeled alternate-reference diagnostics, not the production jump
  ledger. The default pipeline reference/geometry configuration remains unchanged.
- Final complete suite: **43 tests pass**; syntax and whitespace checks pass.
  Reverified no experimental raw correction column, applied ledger or position tables.

## November4 scope reset and supplied original-data inventory
- User expressed frustration and restated the goal: identify clock jumps and correct
  receiver times so they are synchronized. Inspected the supplied local
  `...\Documents\jsats_data` root, its original `downloaded_data`, cleaned `raw_data`,
  data catalog, processing tracker, detection deliveries, GPS and testing folders.
- The original downloaded-data readme confirms receiver files precede Spheros/TPU
  cleaning. The detection deliverables include full beacon/study/test/unknown CSVs;
  the catalog identifies detection timestamps as PDT and event labels as occurrence
  classifications, not calibrated receiver clock jumps.
- `master_df_gps.csv` contains dateTime/receiverName/lat/lon/easting/northing.
  It is a receiver position time series, not an independent sub-millisecond clock
  calibration. The raw GPS messages likewise cannot be assumed to provide a
  detection-processor arrival-time calibration without a documented mapping.
  The data-output workbook contains release/positioning/metrics summaries, not
  clock calibration records.
- The processing tracker and receiver-gap workbook are available for distinguishing
  file/download gaps from clock-fit/paired-observation gaps. No original data was
  changed or uploaded elsewhere.
- Important implementation status: all rapid/joint extensions remain diagnostics.
  The actual experimental application step still uses conservative median candidates;
  supported joint/rapid magnitudes have not been reconciled into that ledger.
  Source-order application, corrected-master/listener recomputation and subsequent
  drift synchronization/coverage validation remain unfinished. Continuing only
  diagnostic variations cannot by itself deliver the PM experiment.
- Required implementation sequence remains: correct paired-observation coverage;
  classify individual beacon observations separately from clock events; estimate
  uniquely supported source-bounded jumps; persist transactional timestamp/audit
  corrections; rebuild drift synchronization without bridging unsupported event
  boundaries; then validate timing and run the four-tag affected/control comparison.
- Disk availability during this check is about6.01GB onC:.K: reports the same free
  capacity and must not be presumed independent storage. No existing data deletion
  or reduced transaction reserve was authorized.

## PM direction (November4 afternoon): supplementary TDoA/DBSCAN jump detection
- PM notes: internal-code identification plus TDoA magnitude/direction helps but misses
  some jumps. Add a supplementary process that identifies remaining jumps visible in
  the TDoA beacon plots, possibly with DBSCAN. Iterative for-loop corrections are not
  needed; the current methodology is fine. Continue Arm A (original) versus Arm B (new)
  comparisons, including TDoA plots, to verify no degradation.
- Read-only prototype on prior Arm B, June27 06:00-June30 06:00, using the existing
  `beacon_pairwise_dbscan.cluster` (0.5ms budget, 2.5 periods, 3 samples): 48,742 clean,
  8,021 noise, 1,174 steady-reflection epochs; 196 remaining steps >=50ms across 18
  receivers. ZOI02 had 9, including -2.000s at June28 01:05 and +2.975/-2.600s near
  05:00. Many steps occurred on most listeners simultaneously (14 receivers at01:05),
  i.e. master-side jumps that per-listener correction would misattribute.
- DBSCAN cluster count is not a tuning input: the number of stable regimes is inferred
  from the existing timing budget and time window; each jump starts a new regime.
- User decisions (ask_user): **keep the strict stop-before-unresolved-fish-timing gate**
  and **reuse the uncorrected `11032026_clocksync` compact database**.
- Implemented (existing files only, opt-in `clock_tdoa_supplement`):
  - `clock_jump_pre.tdoa_supplement`: after Arm B internal-code corrections, recompute
    master-beacon TDoA, cluster with the unchanged DBSCAN, measure level changes between
    consecutive clean regimes (medians of four edge epochs). Steps identical within1ms
    on >=3 listeners and >=1/3 of listeners with clean coverage are corrected on the
    master (offset = -listener step); then remaining 50ms-5s steps per listener.
    Internal OffsetChanged/CounterRestart episodes in the regime gap pin the boundary;
    several mean a rapid episode (net step at last event, window flagged); none means
    the gap is flagged. Uncertain windows are `unresolved` candidates for the strict gate.
    Corrections use original receiver time (`seconds + jump_offset_s`), accumulate in
    `jump_offset_s`, append stage-2 rows to `tblClockJumpEvents` with corrected-side
    times, and run in one transaction with a second-application refusal.
  - Core `_clock_mapped_event_bounds` uses explicit corrected sides when present
    (legacy cumulative mapping otherwise); `clock_fix` reads all ledger columns. Only
    the segment-safe experiment validates; 2019/legacy paths unchanged.
  - `_2025_pipeline._clock_jumps_supplement_run`: stage 1 (unchanged internal-code
    method, `apply=True`), stage 2 supplement, ledger/candidate CSV export, pre-sync gate
    reported (fish steps are stopped later by the existing pre-fish gate). Disk guard
    for in-place update: database size + 1GiB rollback reserve (reproducible compact copy).
- Tests: synthetic uncoded master jump (+1.0s, single internal code) and uncoded listener
  jump (+0.5s, no code) are both removed; fish inside the uncoded gap is unshifted and
  flagged; three-listener requirement; second-application refusal; explicit corrected
  sides. **48 tests pass**.
- Started `--from clock_jumps --to deep_clocks` on the compact database.
- Stage 1 reproduced prior Arm B exactly: 696 internal-code jumps on the same16 receivers.
- **Bug found in the first full-period supplement**: 1,890 receiver corrections and
  **zero** master corrections (1,086 uncertain). Over52 days, sub-50ms DBSCAN fragments
  chained overlapping windows into huge components and only one value group per
  component was evaluated, so obvious master jumps were attributed to every listener
  (leaving ZOI08's own fish detections uncorrected). Stopped the run during metronome,
  before any clock sync was built on it.
- Fixes: only steps >=50ms participate in master attribution; groups are formed from
  steps whose first post-jump epochs fall within three periods, agree within1ms and
  share a common transmission bracket; one master jump per bracket (largest group
  wins); a bracket already corrected on the master cannot take another master jump in
  later passes (prevented a 4-pass oscillation in a new test); per-listener passes may
  repeat until no step remains (max4 passes).
- Read-only check on prior Arm B June27-30: master jumps -1.000104/+0.999887s
  (June27 08:51/08:54, 7/7 listeners), **+2.000025s at June28 01:05 (6/7, range0.09ms)**,
  -2.974550/+2.600235s at05:01/05:04. The01:05 value matches the independent
  three-beacon rapid measurement (+2.000037s).
- Added `revert_tdoa_supplement` and pipeline flag `--revert-tdoa-supplement`: restores
  rows from the unshifted source by original rowid using apply_offsets arithmetic, so the
  post-Stage-1 state is bit-identical (verified rowids map 200k/200k sample). Tests cover
  exact revert, long listener gaps/small fragments, and the simultaneous-listener group.
  **51 tests pass**. Reverted and reran Stage 2 plus clock steps.
- First revert attempt: correlated per-WHEN source lookups made it slow (~45 min), then
  `DROP TABLE IF EXISTS tblMetronome...` resolved to the **attached read-only Arm A**
  (the experiment had no such tables) and failed. Read-only mode blocked the write; Arm A
  verified intact. pandas `to_sql` had already committed the row restore and stage-1
  ledger together. Verified **all 7,492,659 rows bit-identical** (seconds and
  jump_offset_s) to prior Arm B by rowid. Fixed: drops qualified `main.`; revert uses a
  single `UPDATE ... FROM` join; regression test puts a metronome table in the source;
  `--revert-tdoa-supplement` is idempotent (resets clock_jumps when nothing to revert).
- Disabled the completed June28 rapid/joint diagnostics in the experiment config so
  full-period A/B reviews do not rescan all source files. `clock_review` now adds the new
  arm as a third source when the experiment is corrected and writes
  `tdoa_arms_<receiver>.png` (Arm A / prior Arm B / Arm B new, pre-sync master-beacon
  TDoA with counts of >=50ms adjacent changes).

### Arm B new results (November4 evening)
- Stage 2 after fixes: 1,653 supplementary jumps (49 master, 1,604 receiver; 830 with
  uncertain boundaries), 249 residual steps reported unresolved; 5 passes (3 master
  passes, then listeners). Clock steps completed: metronome, clock_surface,
  deep_beacons, deep_positions (configured deep positions, as in Arm A), deep_clocks
  (mapped jump validation 40 passed / 170 unverifiable / 1 failed).
- Strict gate (pre-sync, original-time windows): 630 issues, 3,541 overlapping raw fish
  rows flagged (ZOI06 2,734), versus 27,492 under the earlier conservative ledger.
- Final clock-correction curves, Arm A vs new Arm B: fewer steps >1ms on every receiver;
  largest step CFD03 3.6s->37ms, CFD04 3.6s->27ms, CFD06 4.0s->66ms, ZOI06 2.0s->47ms,
  ZOI03 4.6s->1.0s; several receivers now max exactly ~1.0s. ZOI01/02/05/07 and CFD09
  still have 5-7.6s maxima and higher scatter than Arm A - not yet explained.
- PM TDoA plots (June27 06:00-June30 06:00, `clock_review/interval_1751004000_1751263200/
  tdoa_arms_<rec>.png`): >=50ms adjacent changes 1,758 (A) -> 1,544 (B new); fewer on every
  receiver. ZOI02 is flat apart from isolated reflections and an early June27 toggling
  period. ZOI01 still shows one uncorrected -1s plateau change near June28 13:00 and many
  isolated points exactly +/-1s off the line - consistent with SR3017 per-record firmware
  adjustments (flag `1`/`>`), which also affect fish detections. Not yet corrected.
- Prior Arm B folder `output/ab_B_2026-10-29` no longer exists (user freed ~20GB); I did
  not delete it. C: free space dropped transiently to 0.17GB from external sync software
  (Egnyte Connect cache ~7.5GB / OneDrive), then recovered to ~21.6GB; no pipeline cause.
- No fish positions yet; no commits (user commits only).
