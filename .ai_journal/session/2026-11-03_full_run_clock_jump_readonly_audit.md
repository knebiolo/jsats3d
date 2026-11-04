# 2026-11-03 - Full-run read-only clock-jump audit and correction candidates

## Session metadata
- Author: Copilot, AI assistant using Copilot SDK in VS Code.
- Journal timestamp: 2026-11-03 16:46:27 PST (UTC-08:00).
- UTC timestamp: 2026-11-04 00:46:27Z.
- Tags: clock-jumps, read-only-audit, experimental, validation, interpolation-gaps, owner-review.
- Branch: `ENM_jsat3d_edits`.

## Request and scope
The project lead revisited the PM's request to correct receiver clock jumps from
Internal evidence and beacon TDoA before clock synchronization, compare affected fish
positions, and check for adverse changes elsewhere. The lead pointed to the three-day
plots in [output/clock_jump_ddoa_plots](../../output/clock_jump_ddoa_plots), which were
missing from the previous journal's account of the visual comparison.

The lead first requested investigation without code changes, then explicitly selected
**all receivers across the entire existing early-fish A/B run**, rather than only the
three-day interval. This session did not approve or apply any correction.

- Run interval: 2025-06-15 00:00:00 through 2025-08-06 23:59:59.
- Receiver roster: 20; paired listener timing evidence available for 18, plus the
  ZOI08 master. CFD05 has no master-beacon detections in this interval.
- Master beacon: `7DB7`, hosted at `ZOI08`; configured nominal period 60 seconds.
- Arm A: `output\10292026_Test\jsats3d_2025_earlyfish.db`.
- Arm B: `output\ab_B_2026-10-29\jsats3d_2025_earlyfish.db`.
- Arm B configuration: [run_data_earlyfish_2026-10-29_abB.toml](../../config/run_data_earlyfish_2026-10-29_abB.toml).
- Study time basis: PDT wall time encoded as if UTC. Dates below are **study wall
  time**, not true UTC. Decoding an epoch with a UTC renderer does not change this basis.

## Summary and revised conclusion
The large negative bands in Arm B's three-day DDoA plots are explained by accumulated
pre-sync timestamp corrections. They are not physical receiver distances or final
post-sync timing errors. However, the full-run investigation finds substantial unresolved
level transitions, loss of otherwise joinable beacon observations during clock fixing,
and fish-time interpolation across those gaps.

The offset arithmetic checks out on matched retained beacon epochs. That does **not**
establish that each offset was physically attributed correctly, that every jump was
found, or that fish accuracy improved without regressions.

The prior broad claim that the experiment "does not harm the rest of the data" is not
supported by the old within-2-km validity test. A rescued centroid still had Z about
+27.86 m in the saved comparison; being near the array in XY is not proof of a
physically plausible 3D solution. No production acceptance gate or owner sign-off was
established by this audit.

## Methods and safeguards
- Used the project's explicit `jsat_3d` Python executable:
  `C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe`.
- Opened source SQLite databases with URI `mode=ro` and `PRAGMA query_only=ON`.
- Read existing receiver clock CSVs, applied-event ledger, prior unresolved-event
  reports, upstream metronome tables, and final clock knots.
- Loaded **1,639,714 raw master-beacon detections** with indexed, tag/receiver/time-bounded
  queries. No full raw-detection table scan or index creation was used.
- Enumerated adjacent clock-error changes, interpolation gaps, and raw first-arrival
  level candidates. These are observed candidate transitions, not a census of true
  firmware clock jumps.
- Used leave-one-out neighbor interpolation as a diagnostic: withhold one retained
  beacon knot and predict its error from the two adjacent retained knots. This is not
  independent timing truth, a complete pipeline rerun, or fish-position accuracy.
- Traced large-level Arm B gaps through upstream master-transmission availability,
  final knot presence, and indexed raw study-tag counts.
- Created temporary audit helpers only in the Copilot session folder; removed both
  helpers after successful validation. CSV/JSON results remain there.
- No raw source files, source databases, production scripts, configurations, or physical
  parameters were modified. The only repository edit for this request is this journal.

## Verified findings

### 1. Applied timestamp arithmetic agrees with the recorded ledger
Across **584,109 matched retained beacon epochs on all 16 corrected listeners**:
- `B.seconds = A.seconds - cumulative_recorded_offset(A.seconds)` within
  **0.596 microseconds maximum**.
- Matched master transmission times (`ToT`) are identical, maximum difference 0.
- Arm B's applied ledger contains **696 events**.

This verifies arithmetic on the matched retained epochs only, not every raw row in the
databases or the correctness of the original magnitude estimates.

For the June 27-29 ZOI02 subset, 2,931 epochs matched and showed the same arithmetic
agreement. The accumulated +9.031501-second correction around the June 27 23:57 event
explains roughly a 13-km shift when TDoA is displayed as distance using the saved sound
speed. Source code computes `DDoA = SoS * TDoA` before fitting the clock residual.

### 2. Large remaining transitions occur across missing final clock knots
Across Arm B's 16 listener curves:
- **1,194 intervals** have adjacent clock-error changes of at least 50 ms.
- Every such interval is longer than the diagnostic 120-second gap split.
- **1,167** also have a persistent four-epoch median level difference of at least 50 ms.
- **1,157 of 1,194** have retained upstream beacon epochs whose transmission numbers
  have master `ToT` matches, but **no final clock knots inside the interval**.
- All 1,194 intervals have zero final clock knots inside their saved-curve endpoints.

Because those 1,157 intervals have valid upstream master joins, absent raw data or a
missing master transmission is not a sufficient explanation. The post-join clock-fix
filtering path, especially its five DBSCAN passes, is the primary next investigation.
No in-run filter trace was recorded in this audit, so the exact rejecting iteration and
reason remain unverified.

These intervals are not necessarily 1,194 independent clock events. A retained gap can
span multiple changes, pairing errors, or a genuine outage.

### 3. Fish detections occur within risky interpolation intervals
**128** of those intervals contain **40,290 raw study-tag detections** in Arm B.
The study-tag path constructs receiver knots from `tblDetectionClockFixed` and applies
`np.interp`, with endpoint holds outside the curve. It does not carry segment/gap
boundaries into that fish-time interpolation.

| Receiver | Remaining large-level intervals | Raw fish detections inside |
|---|---:|---:|
| CFD01 | 24 | 80 |
| CFD02 | 30 | 1,018 |
| CFD03 | 87 | 1,658 |
| CFD04 | 25 | 351 |
| CFD06 | 159 | 6,560 |
| CFD07 | 84 | 7,251 |
| CFD08 | 141 | 4,194 |
| CFD09 | 38 | 5,215 |
| ZOI01 | 149 | 2,347 |
| ZOI02 | 109 | 3,382 |
| ZOI03 | 70 | 710 |
| ZOI05 | 99 | 2,859 |
| ZOI06 | 63 | 967 |
| ZOI07 | 59 | 2,508 |
| ZOI09 | 18 | 295 |
| ZOI10 | 39 | 895 |
| **Total** | **1,194** | **40,290** |

These counts are receiver detections, not unique transmissions, positions, or proven
bad fixes. CFD detections are not automatically included in this run's configured
9-receiver 3D Deng solution. Their presence still matters to a complete clock audit.

### 4. Closely spaced Internal triggers can make otherwise measurable events unresolved
The existing quantifier clips each event's before/after window at neighboring
`OffsetChanged` or `CounterRestart` candidates. Related evidence rows can occur fractions
of a second apart, leaving one record with only pre-event epochs and another with only
post-event epochs. Both are then rejected as windows shorter than 150 seconds.

Diagnostic grouping found:
- **172 unapplied clustered-trigger candidates** with a measurable aggregate before/after
  first-arrival TDoA difference.
- **64** have before and after plateau peak-to-peak spreads each within 0.5 ms, and an
  aggregate step within the existing maximum magnitude of 5 seconds.
- The remaining candidates include mixed/noisy plateaus or out-of-range differences.
  They must not be treated as one offset.

Concrete example: **ZOI01, June 16 09:01:05-09:01:06**:
- Two triggers span about **0.393 seconds**.
- Historical report: first trigger `n_pre=4, n_post=0`; second `n_pre=0, n_post=5`.
  Both report "window clipped below 150 s by a neighbouring event."
- Grouped before/after raw beacon levels differ by **-0.406231 seconds**.
- Pre/post spreads: approximately **0.108/0.150 ms**.

This is strong evidence that event grouping merits investigation. It does not yet prove
which exact timestamp within the group should be corrected or which clock caused the step.

### 5. The applied-jump gate mixes original and corrected timestamp coordinates
The applied ledger stores original `jump_time_s`, while Arm B beacon `seconds` have
cumulative offsets subtracted. `clock_fix` centers its event validation window on the
original event timestamp instead of mapping the boundary into corrected coordinates.

- All 696 event centers differ from the post-event corrected boundary.
- Absolute center displacement: median **3.64 s**, p95 **13.56 s**, maximum **15.27 s**.
- None exceeds the gate's +/-300-second window, so this is not proof that all previous
  validation results are invalid. It can change pre/post membership near an event.
- The before-event and after-event mappings differ when a jump compresses or overlaps
  the corrected timeline; one naive center shift is not sufficient in every case.

The gate also accepts residual median steps below **50 ms**, whereas the project's
timing-warning budget is **0.5 ms**. These serve different purposes: jump detection is
not timing acceptance. Current code inspection shows piecewise **linear interpolation**
through residual observations, rather than fitted piecewise regressions.

### 6. Full-run leave-one-out diagnostics do not show uniform improvement
| Diagnostic exceedances above 0.5 ms | Arm A | Arm B |
|---|---:|---:|
| All retained listener knots | 27,509 | 27,707 |
| Both neighboring gaps <=120 s | 7,452 | 8,384 |

The arms retain different knots and have different sample counts, so these totals are
not a matched accuracy score and must not be interpreted as a quantified production
regression. They do demonstrate why a local continuity success cannot certify the
entire run.

For ZOI02 on June 27-29 alone, exceedances fell from 87 to 53, with worst diagnostic
errors still about 202/201 ms. That subset's exceedances all bordered longer gaps.
The all-receiver, full-run audit also finds short-neighbor exceedances; the earlier
gap-only conclusion applies to that ZOI02 three-day subset, not all receivers.

## June 28 ZOI02 case: updated trace
The original matched-A/B subset showed a roughly 2-second transition across 639 seconds.
Tracing Arm B's own complete retained curve gives tighter bounding observations:

- Master/study wall-time endpoints: **2025-06-28 00:56:05 to 01:05:42**.
- Corrected receiver-time gap: **574.289 seconds**.
- Clock-error endpoint change: **-2.000031 seconds**.
- Eight retained upstream beacon observations have valid master transmission matches.
- Zero final clock knots remain inside the interval.
- **369 raw 26E4 detections** occur inside the interval.
- The interval contains one mapped applied-event boundary and multiple raw/common-mode
  candidates. Determine reference attribution and pairing before adding another offset.

The two gap lengths describe different sample sets, not contradictory measurements.
The audit has not yet established the timing of every change within this gap or
computed the resulting error of each 26E4 solution.

## Correction priorities - recommendations only
1. Trace post-join DBSCAN rejection for intervals where good upstream beacon evidence
   disappears; do not merely add more offsets or cosmetically re-center plots.
2. Resolve related Internal triggers as candidate event groups, with explicit bounds,
   stable before/after evidence, and reference attribution.
3. Carry supported segment boundaries through fish-time correction. Do not bridge an
   unresolved discontinuity as if it were smooth clock drift.
4. Validate in consistent original/corrected time coordinates. Separately report
   0.5-ms timing-budget exceedances instead of relying on the 50-ms jump gate.
5. Investigate common-mode candidates with independent listener-pair differences or
   a second beacon before assigning a shared step to each receiver.
6. Re-run paired fish comparisons around verified events and away from events; report
   trajectory plausibility, yield, hull status, vertical artifacts, and known-truth
   errors where available. A within-2-km test is insufficient.

The remaining-gap ledger routes 776 intervals to unflagged-step/epoch-mismatch
investigation, 284 to common-mode/reference attribution, 103 to clustered-trigger
re-quantification, and 31 to applied-event magnitude/coordinate validation. These are
heuristic investigation categories, not causal classifications or an offset schedule.

## Parameters, decisions, and assumptions
- **Production parameter changes: none.**
- No new speed of sound, geometry, hydrophone depth, PRI, or datum was assumed.
- Diagnostic thresholds: existing 50-ms jump threshold; existing 0.5-ms timing budget;
  120-second gap split; existing 150-second minimum window and four-epoch full-window
  minimum; +/-300-second before/after summaries.
- Common-mode screening used +/-120 seconds and +/-150 ms diagnostically; at least four
  peer listeners. Cluster/reference overlap screening used +/-30 seconds, half the
  configured nominal beacon period. These were not installed as production filters.
- Exact event boundaries and common-mode causes remain hypotheses. Nearest-epoch
  first-arrival pairing can be wrong when clock offsets or dropouts create ambiguity.
- `apply_clock_jumps` remains an experimental opt-in; this audit neither enables it nor
  changes an existing run file.

## Artifacts and continuity
Detailed results are outside the repository in the persistent Copilot session folder:

`C:\Users\Ethan.Muhlestein\.copilot\session-state\3c64a55f-0bfa-4d5d-aad8-99995dbff311\files`

Key deliverables:
- `clock_audit_summary.json`: scope, per-receiver counts, criteria, caveats, validation.
- `clock_remaining_gap_corrections.csv`: 1,194 intervals, master/receiver endpoints,
  level changes, upstream joins, final knots, fish counts, investigation recommendations.
- `clock_clustered_trigger_candidates.csv`: 172 grouped-event hypotheses with 64
  stable-plateau priorities explicitly marked.
- `clock_ledger_verification.csv`: arithmetic comparison of all 16 corrected listeners.
- `clock_gate_coordinate_audit.csv`: original/corrected boundary mappings for 696 events.
- `clock_raw_transition_candidates.csv`: 10,158 raw first-arrival level candidates with
  source-file/source-row provenance; not a list of verified jumps.
- `clock_common_mode_candidates.csv`: 2,238 candidate rows, not distinct events.
- `clock_receiver_summary.csv`, `clock_curve_transition_candidates.csv`,
  `clock_interpolation_gaps.csv`, `clock_raw_coverage.csv`,
  `clock_upstream_stage_summary.csv`: supporting diagnostics.

No audit artifacts were copied into the repository. This journal preserves the key
findings if the session artifact directory is unavailable later.

Related repository records:
- [Previous A/B journal](2026-10-30_clock_jump_pre_sync_ab.md).
- [Previous A/B report](../../output/jump_ab_report_2026-10-29/REPORT.md).
- [Three-day plot folder](../../output/clock_jump_ddoa_plots).
- [Pre-sync event quantifier](../../scripts/clock_jump_pre.py).
- [Clock fixing and study-tag correction](../../jsats3d/jsats3d.py).
- [Pairwise beacon processing](../../scripts/beacon_pairwise_dbscan.py).
- [Raw Internal decoding](../../scripts/parse_ats_raw_to_legacy.py).

## Verification, failures, and limitations
- Audit validation passed: report counts reconcile; 16 matched listener ledgers agree
  within one microsecond; matched master ToT differences are zero; all 1,194 risk
  intervals have no final interior clock knots.
- An initial audit helper failed on CFD05's empty beacon result because pandas assigned
  an object dtype to the empty timestamp column. The helper was corrected to report
  absent coverage explicitly, then the complete audit finished successfully.
- Raw querying was narrowed to covering-index columns, with source provenance recovered
  by candidate rowid, rather than repeatedly fetching all raw fields.
- No pipeline, solver, automated test suite, or source database rebuild was run.
- Saved beacon interpolation errors at fitted knots were zero by construction; that is
  not independent validation of fish-time interpolation.
- ZOI04/ZOI11 have raw beacon evidence but no A/B clock curves in this run; CFD05 has no
  master-beacon evidence. ZOI08 self-TDoA cannot independently verify the master clock.
- Historical runs recorded uncommitted code. Current source inspection cannot certify
  that every present source line was identical during those runs.
- Git status changed externally during the read-only audit. No changes by other actors
  were reverted or attributed to this investigation.
- True event completeness, exact causes of all observed transitions, physical accuracy,
  and a safe production correction schedule remain unresolved.

## Files touched and next steps
- Repository file added: this journal only.
- Temporary session helpers: created, run, then deleted.
- Session CSV/JSON audit artifacts: retained.
- Source data/databases/configuration/code: no writes by this audit.
- No commit or PR created; no validation gate declared accepted.

Next owner-gated task: isolate the ZOI02 June 28 interval and representative high-yield
gaps, trace precisely which filtering stage removes each beacon observation, and validate
candidate grouped events with listener-pair evidence before implementing changes.
Active context is the A/B databases, the saved receiver curves and event reports, and the
session artifact ledgers listed above.
