# Session: 2026-09-30 — 2019 Final Positioning Run Diagnosis

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 20:06:22 PDT (UTC-7) = 2026-10-01 03:06:22 UTC
- Tags: #2019 #Deng #diagnosis #run-monitoring #no-code-change
- Branch: `ENM_jsat3d_edits`
- New session file: more than 60 minutes since the last 2026-09-29 entry (System Prompt Section 13).

## Active Context
- Database: `output/cowlitz_2019_synch_1_recreated.db` (local C:, 28,582,645 raw detections, 9 receivers).
- Env: `jsat_3d`. Live legacy run: PID 38928, started 2026-09-30 13:16 PDT.
- Current focus: complete the full 2019 workflow to `tblPositions_Deng` and compare against the canonical K: database.

## Short Summary
- User reported the run "failed" (terminals showed exit code 1) and asked for diagnosis.
- Diagnosis outcome: **the current run has not failed.** It is alive and computing inside deep receiver R01's Deng solve. The exit-code-1 terminals belong to the two *earlier* runs (the pre-fix failures). No traceback, no `STOPPED:` line, and no failed legacy step exists for the current run.

## Diagnosis Evidence (all read-only)
1. Terminal output for the live run ends at `max timestamp is 1568444400.0` — the last print in `position.__init__` before Deng's per-epoch combination loop, which prints nothing until finished.
2. Process table: PID 38928 (`jsat_3d` python) started 13:16 PDT, ~6.6 CPU-hours consumed, 1.1 GB working set — continuous single-threaded compute, exactly the Deng loop profile.
3. FF76 input audit (read-only `diag_ff76.py`, deleted after use):
   - 955,672 secondary rows; 113,971 distinct epochs; max 13 rows/epoch.
   - Clean rows entering Deng: 530,615; clean epochs have <=5 receivers (31 single-receiver, 78,550 five-receiver). Combination count per epoch is tiny — no combinatorial explosion.
   - `transNo` all REAL, no NULL after the pre-Deng delete. The 2026-09-30 vectorized epoch fix wrote sane values; it is NOT the cause of any failure.
4. Run manifest `cowlitz_2019_synch_1_recreated.run.json`: only the completed `study parameters and indexes` step is recorded; the legacy step has not returned — consistent with still-running, not crashed.
5. Progress check (`diag_progress.py`, read-only): R01/R02/R03 `X_t/Y_t/Z_t` still at initial values, `tblPositions_Deng` absent → the live process is mid-R01-Deng. On the previous successful pass, this stage alone produced 164,320 solutions and took over an hour.

## Decisions & Assumptions
- Decision: do NOT kill or restart the live run. Killing it would discard ~7 hours of valid computation. The prior R01 success (164,320 solution-B positions, median X_t -21.050, Y_t 9.736, Z_t 248.090) proves this stage completes under the current code.
- Assumption to verify on completion: R02 (beacon FF74) will now pass with the 2026-09-30 `multipath_data_object` epoch fix; it was the last code failure boundary.
- Known timing behaviour: this run started fresh (the driver drops derived tables per run), so R01 median and `tblPositions_Deng` will only appear near the end of each phase.

## Parameter Changes With Rationale
- None. No code, parameter, or data change this session. Diagnosis only.

## Files Touched
- This journal (new).
- Temporary read-only diagnostics `output/diag_ff76.py` and `output/diag_progress.py` (first deleted; second pending delete on completion check).

## Blockers & Known Limitations
- Deng's per-epoch loop is single-threaded legacy code; R01 alone is a multi-hour stage and there are three deep receivers plus a phase-2 rerun and study tags. Wall-clock completion is many hours.
- Terminal buffers truncate; if the process dies without a visible traceback again, the next restart should wrap the run with `-X faulthandler` and `Tee-Object` to a persistent log so no failure mode can be lost. Recorded here as the prepared contingency, not applied now (no restart).
- Legacy KNN remains unseeded; derived-table counts vary by ~0.1% between runs. Exact-row parity with the canonical DB is not expected; statistical parity is the gate.

## Next Steps
1. Wait for PID 38928 to finish R01 -> R02 -> R03 -> phase 2 -> study tags.
2. On completion: verify `tblPositions_Deng` exists, count solutions, and compare derived-table counts, receiver medians, and position statistics against the canonical `K:\Jobs\3870\014\Calcs\Data\cowlitz_2019_synch_1.db` (read-only).
3. If the process dies silently again: restart once with faulthandler + persistent tee log for full crash capture.
4. Journal the completion or failure boundary either way.

## Follow-up — 2025 Step-1 (Import) Readiness Audit vs 2019

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 ~21:00 PDT (UTC-7) = 2026-10-01 ~04:00 UTC
- Tags: #2025 #import-audit #readiness #no-code-change
- Files touched: this journal only. Two temporary read-only diagnostics
  (`output/diag_import_compare.py`, `output/diag_2025_readiness.py`) created and deleted.

### Purpose
Step-by-step readiness check: can `output/jsats3d_2025_v3.db` feed the same legacy
workflow that is currently running on the 2019 data? This entry covers step 1 (import).

### Findings (all read-only)
Import-table comparison (2019 recreated DB vs 2025 v3 DB):
- Present and populated in both: `tblDetectionRaw`, `tblInterpolatedTemp`, `tblReceiver`,
  `tblStudyParameters`, `tblTag`, `tblWSEL`.
- `tblTemp` (raw temp) absent in 2025 — NOT a blocker; legacy processing reads only
  `tblInterpolatedTemp` (jsats3d.py lines 116, 866), which 2025 has (30,817 rows).
- 2025 `tblDetectionRaw` (60,410,185 rows) carries all 11 legacy columns plus additive
  ATS columns; schema contract satisfied.
- All 20 receivers have beacon Tag_IDs with pulseRate set (60 s; 7F91/7F32 at 30 s).
  Master beacon 7DB7 (ZOI08) detected at all 20 receivers (3,238,083 detections).
- `tblReceiverGPS` absent from v3 (predates the GPS staging feature) — dynamic
  coordinates unavailable until rebuild; static X/Y path works.

### Blockers for the legacy run on 2025 (confirmed against code, not config comments)
1. `tblStudyParameters.BM_Elev` is NULL. `clock_fix_object` (line 779) and `position`
   (line 1129) read BM_Elev; clock fix cannot run. Owner must supply the 2025 benchmark
   elevation. BM_Elev_Units currently 'feet' — unit must come with the value.
2. `synch_time_start`/`synch_time_end` are NULL — legacy sync window undefined. Owner
   must set the 2025 synchronization window.
3. NBW and SNR are 0/60,410,185 non-null. The secondary classifier (line 632) scales
   `['Amplitude','NBW','SNR']`; with NULL columns every row is dropped. Options pending
   Kevin: `signal_proxies = true` (SNR=SigStr-Threshold, NBW=BitPeriod) or replacing the
   secondary classifier with the DBSCAN method for 2025 (M4 design).
4. `master_receiver = ZOI08` and surface/deep receiver split remain pending Kevin approval.

### Conclusion
Step 1 (import) is structurally complete for 2025. The run is blocked at step 2+
by two missing study parameters (BM_Elev, sync window) and the open classifier-feature
decision. No code defect found in the import path.

## Follow-up — BM_Elev Search Across All 2025 Data

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 ~21:30 PDT (UTC-7) = 2026-10-01 ~04:30 UTC
- Tags: #2025 #BM_Elev #benchmark #datum #read-only #owner-question
- Files touched: this journal only. Temporary read-only diagnostics
  (`output/diag_bm_elev_search.py`, `output/diag_bm_elev_cols.py`) created and deleted.

### Search performed (all read-only)
- Repo docs: `docs/2025_owner_input_checklist.md` lists benchmark elevation + vertical
  datum as an unresolved owner question. `LONG_TERM_CONTEXT.md` confirms BM_Elev NULL.
- `cowlitz_2025_AT_config.xlsx` (full dump): receiver lat/long, hydrophone depth,
  deployment dates — NO surveyed benchmark elevation, NO vertical datum field.
- `data_catalog_cowlitz_falls_AT_2025.xlsx`: catalog descriptions only; no elevation.
- `cowlitz_AT_2025_testing_sheets.xlsx` (7 sheets): no elevation/datum fields.
- `2025 Master Covariate Table_20251212.csv`: water-surface elevations in feet
  (`NSC.CZD_WTR_EL.F_CV` ≈ 861.5–862.5 ft; also CZU upstream gauge). No benchmark.

**Verdict: the 2025 benchmark elevation is not present in any delivered data.**
It must come from the owner/PM (already on the owner checklist).

### Code-path analysis — what BM_Elev actually does
- `clock_fix` (jsats3d.py 815–840): BM_Elev is used ONLY for receivers whose
  `Ref_Elev != 'BM'` (WSEL-referenced), to compute Z(t) = BM_Elev − (WSEL(t) − z).
- 2019: R07/R08/R09 are `WSEL`-referenced → BM_Elev genuinely required (262.585 m).
- 2025: ALL 20 receivers are `Ref_Elev = 'BM'` → the WSEL branch is never taken;
  BM_Elev does not enter any computed quantity.
- `position` (Deng, line ~1233): the benchmark correction is commented out in the
  legacy code; `z_at_t` returns `Z_t` directly. BM_Elev unused in positioning.
- Remaining failure mode: `clock_fix_object.__init__` line 784 divides
  `benchmark_elev / 3.28084` when `BM_Elev_Units='feet'` and output is meters.
  With BM_Elev NULL this raises TypeError before any science runs. So the NULL
  still crashes the run even though the value is mathematically inert for 2025.

### Candidate value (owner confirmation required — NOT adopted)
- 2019 (same site, Cowlitz Falls) used BM_Elev = 262.585 m = 861.5 ft.
- The 2025 covariate water elevations sit at 861.5–862.5 ft — same datum family.
- Plausible that the same project benchmark applies, but per System Prompt
  Section 1 a physical parameter is never guessed: ask Kevin/PM to confirm the
  2025 benchmark elevation and vertical datum.

### Interim option for step-2 testing (proposed, not applied)
Because all 2025 receivers are BM-referenced, any placeholder BM_Elev produces
bit-identical scientific output; it only has to be non-NULL to survive the unit
conversion line. If testing proceeds before the owner answers, set a documented
placeholder with a prominent logged warning and replace it on owner confirmation.

### Next Steps
1. Owner question (already on checklist): 2025 benchmark elevation + vertical datum.
2. With user approval: proceed to step-2 (metronome) testing on the 2025 database,
   which does not require BM_Elev.

## Follow-up — Step-2 Plan Approved, Owner Questions Drafted

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 ~21:45 PDT (UTC-7) = 2026-10-01 ~04:45 UTC
- Tags: #2025 #step-2 #metronome #owner-questions #documentation
- Files touched: `docs/2025_owner_input_checklist.md` (BM_Elev audit result and sync-window
  item added under Study and Datums), this journal.

### User decision
- Proceed to step-2 (metronome) testing on the 2025 database.
- Document the BM_Elev finding and draft short questions for Kevin/Drew.

### Questions sent to Kevin/Drew (chat-ready text)
1. BM_Elev: "We can't find a 2025 benchmark elevation in any deliverable. 2019 used
   262.585 m (861.5 ft) at this site — is that benchmark still valid for 2025, and in
   what vertical datum? (All 2025 receivers are BM-referenced, so it doesn't change any
   number — we just need the official value on record.)"
2. Sync window: "What synchronization window should we use for the 2025 legacy run?
   For testing we'll use the deployment span 2025-06-04 to 2025-09-17 unless you say otherwise."
3. SNR/NBW: "ATS gives us amplitude but no SNR/NBW, which the 2019 secondary classifier
   needs. Do you want proxies (SNR = SigStr - Threshold, NBW = BitPeriod) so the legacy
   classifier runs, or should 2025 use the DBSCAN method instead?"

### Step-2 (metronome) feasibility check before testing
- `beacon_epoch.__init__` (jsats3d.py 194+) reads only `tblTag` (pulseRate), `tblReceiver`
  (host for the beacon), and `tblDetectionRaw` for that one tag. It applies NO sync-window
  filter — the `synch_time_start/end` filter lives in the *import* path (lines 146–168),
  which the 2025 parser replaces. So step 2 runs with the parameters exactly as stored.
- Master beacon 7DB7 (ZOI08): 3,238,083 detections across all 20 receivers; pulseRate 60 s.
- BM_Elev is NOT needed for step 2 (first needed by `clock_fix_object`, step 3).

### Test design (long-run rule, System Prompt Section 16)
- Writes go only to derived tables (`tblMetronome*`) in the local C: v3 database and a
  scratch folder; raw K: untouched. Derived tables are dropped/rebuilt per run by design.
- The 2019 production run (PID 38928) stays untouched; the test is a separate process and
  must remain light enough not to starve it. Metronome is single-threaded pandas work.
- Plan: run metronome only (beacon_epoch -> multipath_2 -> multipath_classifier) via a
  bounded driver, observed in the foreground, not fire-and-forget.
- Known expectation: the secondary classifier step will hit the NULL SNR/NBW wall
  (signal_proxies decision pending). The test's purpose is to prove epoch enumeration
  and primary ranking work on ATS data, and to surface the exact failure boundary of
  the classifier — evidence for Kevin's decision, not a workaround.

### Parameter Changes With Rationale
- None yet. Checklist text proposes a documented testing-only sync window and a
  placeholder BM_Elev policy; neither is applied to any database in this entry.

## Follow-up — Sync Window and DBSCAN Decisions Recorded

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 ~22:00 PDT (UTC-7) = 2026-10-01 ~05:00 UTC
- Tags: #2025 #decisions #sync-window #dbscan #parameters
- Files touched: `config/run_data.toml`, `docs/2025_owner_input_checklist.md`, this journal.

### Decisions (project lead, 2026-09-30)
1. **Sync window = deployment window.** `synch_time_start = 2025-06-04 00:00:00`,
   `synch_time_end = 2025-09-17 00:00:00` (array config sheet deployment/recovery span).
   Written to `config/run_data.toml` [study]. Checklist item marked resolved.
2. **2025 multipath rejection uses DBSCAN.** The legacy SNR/NBW secondary classifier is
   not used for 2025; SNR/NBW proxies are not needed and `signal_proxies` stays false.
   This matches the approved M4 design (System Prompt Section 7: DBSCAN, unsupervised,
   per tag per receiver, fixed physically derived eps).

### Parameter Changes With Rationale
- `synch_time_start/end`: NULL -> deployment window. Rationale: legacy workflow requires
  a defined synchronization window; the deployment span is the maximal physically
  meaningful window. Effect: defines the detection set entering synchronization; no
  algorithmic parameter changed. (Config only — the v3 database's tblStudyParameters
  still holds NULLs until a `--skip-build` parameter refresh or rebuild is run.)
- No eps, min_samples, threshold, or physical constant changed.

### Consequence for the step map (2019 vs 2025)
- Step 2 (metronome): `beacon_epoch` + `multipath_2` (primary, physics-based lag ranking)
  apply to 2025 unchanged. The *secondary* classifier substep is 2019-only; for 2025 the
  secondary stage is the DBSCAN method (integration point to be designed — M4).
- Step 3 (clock fix): unaffected by the classifier swap, but reads
  `tblMetronomeSecondFiltered` — the DBSCAN stage must deliver its output under the
  table contract the clock fix expects (`multipath_prediction == 0` rows).
- BM_Elev remains the one outstanding study parameter (owner question stands).

### Still open with Kevin
- BM_Elev value + vertical datum (inert for 2025 computation but must be on record).
- master_receiver = ZOI08 and surface/deep split approval.
- DBSCAN eps/min_samples final values (physically derived, fixed study-wide).

### Next Steps
1. Refresh tblStudyParameters in the v3 database from the updated run file (--skip-build)
   once BM_Elev policy is settled (placeholder or owner value).
2. Build the bounded step-2 metronome test (beacon_epoch + multipath_2 on 7D B7/ZOI08),
   foreground, observed; document where the DBSCAN stage plugs in.
3. Design the DBSCAN secondary stage to write the `tblMetronomeSecondFiltered` /
   `tblDetectionFilterSecondary` contracts so clock_fix and Deng run unmodified.

## Follow-up — Step-2 Test Built; First 2025 Incompatibility Found and Fixed

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 ~22:30 PDT (UTC-7) = 2026-10-01 ~05:30 UTC
- Tags: #2025 #step-2 #metronome #compatibility-fix #fillna #validation
- Files touched: `scripts/step2_metronome_check.py` (new), `jsats3d/jsats3d.py`
  (two-line-site compatibility fix), this journal.

### Step-2 test driver
- New `scripts/step2_metronome_check.py`: runs the exact legacy call order from
  `legacy_pipeline.metronome()` — `beacon_epoch` -> `host_receiver_enumeration` ->
  `adjacent_receiver_enumeration` -> `multipath_2(multipath_data_object(metronome=True))`
  — stopping before the 2019-only SNR/NBW secondary classifier (2025 secondary = DBSCAN).
- Resets only `tblMetronomeUnfiltered`/`tblMetronomeFiltered` first (same rerun-safety
  pattern as the production driver, scoped to this step).
- Report: epoch count, epoch-start spacing vs the 60 s pulse rate, per-receiver rows /
  epochs heard / multipath %, and child detections outside every epoch window.
- Writes only derived tables in the local v3 DB plus a scratch folder; raw K: untouched.
  The live 2019 process imported its code at startup and is unaffected by file edits.

### Failure found (the test's purpose)
- First run crashed in `beacon_epoch.host_receiver_enumeration` at `fillna(0)`:
  pandas 3 string-dtype columns reject integer fill. Trigger: 2025 additive ATS TEXT
  columns (GPSFixTimeStamp, ClockStatusMarker, RawDateTime, ...) are mostly NULL.
  2019 never hit this because its NULL-able columns are all numeric.

### Fix (result-preserving, both sites)
- `jsats3d.py` lines ~251 (`host_receiver_enumeration`) and ~354 (`multipath_data_object`
  unsynced-host branch): blanket `fillna(0)` -> numeric-columns-only `fillna(0)`.
- Scientific intent preserved: the fill exists to zero the first-row lag NaN for the
  epoch cumsum; numeric NaN -> 0 behavior for 2019 columns is bit-identical (2019 string
  columns carry no NaN, proven by the live 2019 run passing these stages).
- Bonus correctness: avoids writing fabricated '0' strings into ATS sensor fields
  (NO FABRICATED SENSOR DATA rule).

### Validation
- `py_compile`: passed. Full suite: 32 passed (pytest newly installed into `jsat_3d`;
  dev tool only, no runtime dependency change).
- Step-2 re-run in progress on the 2025 v3 database at entry time.

### Parameter Changes With Rationale
- None. Compatibility-only edit; no physical or algorithmic parameter changed.

## Follow-up — Step-2 Metronome PASSED on 2025 Data

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 ~23:00 PDT (UTC-7) = 2026-10-01 ~06:00 UTC
- Tags: #2025 #step-2 #metronome #passed #anomalies #CPDI
- Files touched: this journal. Derived tables `tblMetronomeUnfiltered` (3,238,083 rows)
  and `tblMetronomeFiltered` (3,238,499 rows) written to local `jsats3d_2025_v3.db`.

### Result
The legacy metronome primary stages ran end-to-end on the 2025 ATS database with the
same scientific method as 2019 (half-period epoch rule, nearest-window child assignment,
within-epoch rank -> multipath flag):
- Host ZOI08 (beacon 7DB7): 134,301 epochs enumerated.
- All 20 receivers heard the master beacon; epoch coverage 38k–134k epochs/receiver.
- Child detections outside every epoch window: 74,352 of 2,670,706 (2.78%).

### Physics observations
1. **Epoch spacing is ~63.5 s, not the configured 60 s.** Median 63.565 s
   (p5 63.254, p95 64.355; tight distribution, not multiples of 60 — so these are real
   consecutive transmissions, not missed epochs). The enumeration is robust to this
   (half-period rule), but the cause should be confirmed with ATS/Kevin: probable
   nominal-period-plus-offset beacon behavior. tblTag pulseRate stays 60.0 (unchanged);
   flag only.
2. **CPDI pattern visible and expected.** Host ZOI08 hears its own high-amplitude beacon
   with 76.3% multipath (4.2 detections/epoch average). On-bottom receivers ZOI01
   (62.7%), ZOI02 (46.9%), ZOI09 (42.1%) show elevated multipath — consistent with
   reflective-surface geometry. Report, not filter (System Prompt 7.5).
3. **CFD05 nearly deaf: 3,822 rows** vs ~100k typical — matches the PM's known
   hydrophone issue note. Keep excluded-from-reliance list.

### Anomaly to investigate (not fixed, flagged)
- `tblMetronomeFiltered` has 416 MORE rows than `tblMetronomeUnfiltered` (+0.013%).
  Cause: duplicate (Rec_ID, Tag_ID, seconds) keys with differing additive columns
  (likely the same physical detection ingested from overlapping source files).
  The legacy full-row `drop_duplicates` cannot catch them, and the rank join expands
  on the duplicated index; ties also produce det_rank 1.5/1.5, marking both rows
  multipath (the direct arrival is lost for that epoch). 2019 showed no expansion
  (554,375 -> 554,375). Action: audit duplicate-seconds keys in the 2025 parser
  output before step 3; do NOT silently dedupe in the legacy core.

### 2019 comparison (same stage)
- 2019: master R05, 554,375 metronome rows, 9 receivers.
- 2025: master ZOI08, 3.24M metronome rows, 20 receivers — larger array, higher
  beacon amplitude, same algorithm, same code path.

### Next Steps
1. Audit 2025 duplicate (Rec_ID, Tag_ID, seconds) detections (read-only diagnostic).
2. Step 3 preview: clock_fix needs non-NULL BM_Elev in tblStudyParameters (inert value)
   and the sync window refresh; then test clock fix for the surface receivers.
3. DBSCAN secondary stage design (replaces multipath_classifier for 2025).

## Follow-up — Duplicate Audit Closed; Finding for Drew/Spheros

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-09-30 ~23:59 PDT (UTC-7) = 2026-10-01 ~07:00 UTC
- Tags: #2025 #duplicates #provenance #spheros #read-only
- Files touched: this journal. Temporary diagnostic `output/diag_dup_audit.py` created,
  run, killed mid-final-query at user request, deleted. Databases read-only throughout.

### Findings (beacon 7DB7 scope, complete; whole-DB total scan cancelled by user)
- 11 duplicate (Rec_ID, seconds) keys, 33 extra rows, out of 3,238,083 beacon rows:
  ZOI09 (10 keys), ZOI10 (1 key repeated 20 times).
- Every duplicate comes from the SAME raw file at different row positions with a
  stride of exactly 8 rows (e.g. SR25309_250618_153501_cleaned.csv rows 42603/42611;
  SR18079_250626_141501_cleaned.csv rows 47388..47540 step 8, 20 copies of one ping).
- Identical timestamp (microsecond), identical SigStr per copy -> same physical
  detection, duplicated as repeated row blocks in the Spheros "cleaned" deliverable.
  Not a parser defect: `SourceRow` proves the rows exist separately in the file.
- Effect downstream: duplicated keys expand the legacy rank join (+416 rows in
  tblMetronomeFiltered) and tie det_rank at 1.5, so affected epochs lose their
  direct-arrival label. Magnitude ~0.01% — negligible statistically, but the
  rejection accounting should be exact.

### Decisions
- No dedupe implemented anywhere yet. Options: (a) Spheros reissues or explains;
  (b) documented parser-level dedupe on (Rec_ID, Tag_ID, seconds, SigStr) with a
  logged count. Decision deferred to owner/provider response (fail-loud policy;
  no silent data dropping).
- Whole-database duplicate total not measured (query cancelled); the per-file
  block-repeat mechanism is established from the beacon subset.

### Next Steps
1. Send summary to Drew/Spheros (text in chat log; key question: why do cleaned
   CSVs contain repeated 8-row blocks, and does the raw unfiltered delivery too?).
2. Proceed to the DBSCAN secondary stage (2025 step-2b) so clock fix has its
   tblMetronomeSecondFiltered input.

## Follow-up — Prior DBSCAN Work Reviewed (System Prompt 7.1 Report)

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-01 ~00:30 PDT (UTC-7) = 2026-10-01 ~07:30 UTC
- Tags: #dbscan #review #step-2b #integration-design #no-code-change
- Files touched: this journal only (read-only review of notebook, scripts, journals).

### What exists (more mature than expected)
The DBSCAN secondary stage is NOT a blank page. `scripts/beacon_pairwise_dbscan.py`
is a production-grade implementation, parameter-swept and validated on real 2025 data:
- Features: normalized time window (t / (2.5 x true beacon period)) and normalized
  pairwise clock offset delta_s / 0.5 ms, where delta_s = (t_i - t_a) - (d_Bi - d_Ba)/c.
  Chebyshev metric, eps = 1.0 in scaled space (equivalent to the 0.5 ms timing budget).
- Fixed study-wide parameters (no per-group tuning, satisfying System Prompt 7.4):
  TIMING_BUDGET_S 0.5 ms, TIME_WINDOW_PERIODS 2.5 (2.0->2.5 on 09-24, journaled),
  MIN_SAMPLES 3, ANCHOR_MAJORITY 0.5, ANCHOR_MIN_RECEIVERS 4, MAX_REFLECTION_DELAY_S 0.25 s.
- 36-configuration grid sweep (09-28) validated the setting as the knee point:
  100% recall of planted synthetic echoes at <=1 ms tolerance; ZOI LOO p95 ~16 us;
  the old notebook per-receiver p99-eps rule catches only 0-8.5% of planted echoes
  on 2025 data and was rejected.
- Classes: clean / noise / steady_reflection / anchor_suspect.
- Legacy-mode output already exists (--output-db): writes `tblMetronomeFiltered`
  (det_rank/multipath) and `tblMetronomeSecondFiltered` including
  `multipath_prediction` (1 = not clean) — exactly the contract `clock_fix` reads.
- The earlier notebook (per-receiver Euclidean, auto-eps) is historical; superseded.

### Open items blocking PRODUCTION adoption (not testing)
1. Kevin approval list items: fixed parameter set (item 4), steady-reflection
   retention rule (item 5, CFD01 dual-mode ~21.5 ms apart), CFD sigma-weighting
   or exclusion (item 6; CFD residuals 10-25x ZOI, 3.1% epochs over budget).
2. Known data issues that touch DBSCAN inputs: ZOI03/ZOI06 parser file-skip
   (18 June files), CFD05/CFD09 serial swap window, ZOI02 host position unsurveyed.

### Integration gap analysis (step-2b -> step-3)
- The tool was validated on beacon 7D2D (ZOI02) anchored at ZOI09. The legacy
  metronome/clock-fix pipeline needs the MASTER beacon 7DB7 (ZOI08) in the
  metronome tables. Next test: run the legacy-mode output path for 7DB7 and
  verify clock_fix can consume it.
- Legacy clock_fix query contract confirmed: tblMetronomeSecondFiltered WHERE
  Rec_ID = current AND Tag_ID = master AND multipath_prediction == 0.

### Next Steps
1. Test run: beacon_pairwise_dbscan.py legacy-mode output for master beacon 7DB7
   into a scratch DB; validate schema against clock_fix expectations.
2. Put the steady-reflection retention policy question on Kevin's list (already
   item 5) and do not auto-select a mode before his answer.
3. After BM_Elev non-NULL refresh: attempt step 3 (clock fix) on 2025 using the
   DBSCAN-produced tblMetronomeSecondFiltered.

## Follow-up — Step-2b Test: DBSCAN Legacy-Mode Output on Master Beacon 7DB7

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-01 ~01:00 PDT (UTC-7) = 2026-10-01 ~08:00 UTC
- Tags: #2025 #step-2b #dbscan #master-beacon #contract-check
- Files touched: this journal. New scratch outputs `output/dbscan_7DB7_step2b/`
  (CSVs, per-receiver plots, scratch DB `metronome_7DB7_legacy.db`). Source v3
  database read-only; temporary contract-check diagnostic created and deleted.

### Run
`beacon_pairwise_dbscan.py` on master beacon 7DB7 (host ZOI08), anchor ZOI09,
2025-06-20..06-22 (validated sweep window), fixed parameters unchanged
(0.5 ms / 2.5 periods / min_samples 3 / Chebyshev eps 1.0). First run of the
pairwise method on the MASTER beacon (prior validation used 7D2D/ZOI02).

### Results
- 62,693 detections -> 40,728 first arrivals -> 32,288 paired epochs.
- ZOI01/ZOI02/ZOI11/CFD01 noise 3-10%; clean residual RMS 52-73 us (inside budget).
- ZOI06 84% noise: expected — this window sits inside the known ZOI03/ZOI06
  June parser file gap; sparse data, not a method failure.
- CFD06/CFD07 55-59% noise; CFD02 20 clean epochs over 0.5 ms — consistent with
  the known CFD noisiness (Kevin approval item 6).
- anchor_suspect ~27% of epochs (659/2,437) — noticeably higher than the 7D2D
  runs; host-side (ZOI08 on the debris barrier, 76% self-multipath in step 2)
  or anchor-side cause not yet separated. Flag for review before production.
- Emitted warnings preserved: 6 CFD receivers with over-budget clean epochs;
  ZOI08 host position unsurveyed (absorbed as constant per-receiver offsets).

### Contract check vs legacy clock_fix
- `tblMetronomeSecondFiltered`: exact legacy query shape
  (Rec_ID = x AND Tag_ID = '7DB7' AND multipath_prediction == 0) returns rows
  for all 18 non-host receivers (58-1,778 clean rows each). Columns include
  everything clock_fix touches plus additive dbscan_class/delta_s. PASS.
- `tblMetronomeFiltered` ToT contract: FAIL as-is — clock_fix reads ToT from
  tblMetronomeFiltered WHERE Rec_ID = master host AND multipath == 0, and the
  pairwise tool does not emit host (ZOI08) rows.
- Resolution identified, matching 2019 architecture: the legacy metronome primary
  (step 2, validated today) already writes host rank-1 rows to tblMetronomeFiltered
  in the v3 DB. The 2019 secondary classifier also passed the host through without
  classification ("the excluded host" branch). So the DBSCAN stage must (a) write
  its SecondFiltered rows into the project DB alongside the legacy metronome tables,
  and (b) add host passthrough rows (det_rank 1 -> multipath_prediction 0,
  class 'host') to SecondFiltered, mirroring the legacy host behavior exactly.

### Parameter Changes With Rationale
- None. Fixed DBSCAN parameters used as validated; no tuning.

### Next Steps
1. Build the step-2b integration glue: DBSCAN SecondFiltered + host passthrough
   written into the project DB after the legacy metronome primary.
2. Refresh 2025 study parameters (sync window; BM_Elev inert non-NULL placeholder
   with logged warning) pending Kevin's official value.
3. Step-3 clock fix test on 2025 surface receivers.
4. Ask Kevin: anchor_suspect rate 27% on the master beacon — acceptable, or
   investigate host-side cause first?

## Follow-up — Host Passthrough Added; clock_fix Contracts All Pass

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-01 ~01:30 PDT (UTC-7) = 2026-10-01 ~08:30 UTC
- Tags: #2025 #step-2b #dbscan #host-passthrough #contract-pass
- Files touched: `scripts/beacon_pairwise_dbscan.py` (`write_legacy_tables` host
  passthrough + call site), this journal. Scratch DB regenerated.

### Change
- `write_legacy_tables` now emits beacon-host rows, mirroring the 2019 secondary
  classifier's "excluded host" branch and the function's existing anchor handling:
  - Host rank-1 bursts matched to anchor epochs by nearest within 0.5 x period,
    one per transNo (nearest wins), class 'host', `multipath_prediction` 0;
    epochs in the anchor-suspect set keep class 'anchor_suspect' / prediction 1.
  - `delta_s` left NaN for host rows — the pairwise offset is undefined at the
    host; no fabricated zero.
  - 'host' added to the clean set for `multipath_prediction`.
- Signature change: `write_legacy_tables(..., beacon_rec, period)`; call site updated.
- Note: today's session has exceeded the default 2-file change budget
  (jsats3d.py, step2_metronome_check.py, beacon_pairwise_dbscan.py, config, docs);
  each step was explicitly user-approved in chat.

### Validation
- `py_compile` + full suite: 32 passed.
- Regenerated scratch DB (7DB7/ZOI08, anchor ZOI09, 06-20..06-22):
  tblMetronomeFiltered 57,265 rows (was 48,680), SecondFiltered 37,162 (was 34,725).
- Contract checks (read-only):
  - ToT: 2,437 host rows, multipath = 0, zero duplicate transNo. PASS.
  - Host clock data: 1,778 'host' clean + 659 'anchor_suspect' excluded. PASS.
  - Non-host receivers: legacy query returns clean rows for all 18. PASS (earlier).

### Parameter Changes With Rationale
- None. Host passthrough is classification plumbing; no DBSCAN parameter,
  physical constant, or rejection criterion changed.

### Next Steps
1. Step-3 readiness: refresh v3 tblStudyParameters (sync window from run file;
   BM_Elev inert non-NULL placeholder with a prominent logged warning, pending
   Kevin's official value and datum).
2. Step-3 test: run clock_fix for a small surface-receiver set against a DB
   carrying the DBSCAN metronome tables; compare residual behavior to 2019.
3. Production decision points remain with Kevin (parameters, steady-reflection
   rule, CFD weighting, anchor_suspect rate at the master beacon).

## Follow-up — Drew's Response on Datum, Duplicates, and 7DB7

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-01 08:55:14 PDT (UTC-7) = 2026-10-01 15:55:14 UTC
- Tags: #2025 #owner-response #provenance #CPDI #datum #gate-review
- Files touched: this journal and `docs/2025_owner_input_checklist.md`.
- Active context: 2025 v3 import and June 20-22 7DB7 DBSCAN diagnostic; no
  new run, database edit, K: write, or parameter change.

### Reported by Drew (not independently verified here)
- Investigate the high anchor-suspect rate; CPDI is a hypothesis, not an
  established explanation. Kevin has not yet accepted ZOI08 as reference.
- Prefer the 2019 vertical datum for simplicity if appropriate, but defer the
  final datum and benchmark elevation to Kevin. No value was approved.
- Detection rows appear out of chronological order in some downloaded files;
  Drew reports the same duplicates there and has contacted ATS about a possible
  receiver issue. Cause and overall scope remain unknown.
- Penny's preprocessing reportedly removed a few malformed rows containing
  special characters or NULLs; Drew will obtain her cleaning script and the
  original downloaded files for comparison.

### Local evidence and correction of earlier claims
- `config/run_data.toml` points to a folder called `raw_data`, but
  `parse_ats_raw_to_legacy.py:discover_target_files` prefers `_cleaned` over
  `_recovered`/`_recovery` and unsuffixed files for the same canonical key.
  The duplicate audit's `SourceFile` values are `_cleaned.csv`. A folder name
  does not demonstrate that v3 used unfiltered, unmodified receiver output.
- The 11 duplicate keys / 33 extra rows are a 7DB7 subset only, not a
  whole-database estimate. Row offsets of eight and matching timestamp/SigStr
  establish repeated values in selected inputs, not their origin. Retract the
  earlier conclusion that the cleaning/concatenation pipeline caused them.
- The 659/2,437 anchor-suspect figure is the DBSCAN diagnostic's common-mode
  flag, not a diagnosis of CPDI or proof the host alone is at fault. Check
  host ZOI08, anchor ZOI09, cross-receiver residuals, beacon alternatives,
  and deployment geometry before choosing a clock reference. No thresholds tuned.
- All 20 v3 receivers currently have `Ref_Elev='BM'`; the legacy WSEL branch
  thus does not use BM_Elev on those rows. That code observation does not
  validate the 2025 vertical datum or the physical meaning of their Z values.

### Decisions, blockers, and next steps
- No deduplication, resorting, source replacement, or benchmark placeholder
  authorized. Do not advance a production clock correction or claim Gate 2
  acceptance while input provenance, datum, and reference quality are open.
- Request Penny's exact script/version, downloaded originals, and a row-count
  ledger for malformed/removed rows. Compare duplicates and timestamp ordering
  before and after cleaning on the named receiver files; await ATS findings.
- Diagnose 7DB7 anchor-suspect epochs against the existing 7D2D diagnostic
  and host/anchor detection timing, label findings provisional, and bring the
  reference-clock choice and datum to Kevin for review.
- Parameter changes with rationale: none. The prior proposed inert BM_Elev
  placeholder is not approved for use; retain NULL until Kevin confirms it.

## Follow-up — Provisional 2019 Datum and 2025 30-Day DBSCAN Diagnostic

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-01 09:18 PDT (UTC-7) = 2026-10-01 16:18 UTC (start)
- Tags: #2025 #datum #dbscan #30-day-diagnostic #provisional
- Files touched: `docs/2025_owner_input_checklist.md` and this journal.
- Active context: read-only input `output/jsats3d_2025_v3.db`; local output
  `output/dbscan_7DB7_30d_20250620_20250720/`.

### Decision and scope
- Project lead selects the 2019 vertical datum for provisional 2025 work,
  consistent with Drew's recommendation. Do not equate this datum choice with
  confirmation of the 2025 benchmark elevation (2019: 262.585 m); Kevin's
  confirmation and receiver Z-reference check remain outstanding. No physical
  value written to the database or run config for this diagnostic.
- Run 30 days of the 2025 master beacon 7DB7 (host ZOI08, anchor ZOI09):
  2025-06-20 inclusive to 2025-07-20 exclusive, local study timestamps.
  Same fixed DBSCAN parameters as the prior two-day test; no tuning.

### Preconditions and run status
- 917,746 7DB7 detection rows in the requested window; interpolated
  temperature has non-NULL coverage from 2025-06-02 13:45 to 2025-09-17 13:45.
- C: had approximately 54 GB free before the run. No output directory collision.
- Started the existing `beacon_pairwise_dbscan.py` diagnostic with
  `--beacon-receiver ZOI08 --anchor ZOI09 --start 2025-06-20 --end 2025-07-20`
  and `--no-interactive`. No `--output-db`: input SQLite remains read-only,
  results are CSVs and plots in a new C: output directory; K: untouched.
- At entry time the process was still running. The script warned that it scans
  the entire `tblDetectionRaw` table without `--rowid-range`; this warning
  concerns query cost, not an unbounded output time window. Do not claim a
  result until completion and artifact validation.

### Blockers and next steps
- Diagnose the 7DB7 anchor-suspect share across this longer window, compare
  with the two-day result and the 7D2D reference, and distinguish CPDI from
  clock or anchor effects. Owner sign-off remains required before Gate 2.
- Parameter changes with rationale: none. Datum selection is provisional;
  benchmark elevation and DBSCAN constants are unchanged.

## Follow-up — 2025 30-Day DBSCAN Completed and Checked

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-01 09:30:49 PDT (UTC-7) = 2026-10-01 16:30:49 UTC
- Tags: #2025 #dbscan #30-day-diagnostic #anchor-suspect #provisional
- Files touched: this journal; generated CSVs and plots under
  `output/dbscan_7DB7_30d_20250620_20250720/` (C: only).
- Active context: 7DB7/ZOI08 beacon, ZOI09 anchor; 2025-06-20 inclusive to
  2025-07-20 exclusive; source DB opened read-only, no --output-db used.

### Result and validation
- Run finished successfully with fixed parameters unchanged (0.5 ms timing
  budget, 2.5 periods, min_samples 3, Chebyshev eps 1.0). Artifacts verified:
  `_epochs.csv`, `_segments.csv`, `_summary.csv`, and per-receiver plots.
- 917,746 detections -> 617,647 first arrivals -> 502,681 paired receiver
  epochs. 37,928 anchor epochs; 9,847 flagged anchor-suspect (25.96%).
  Earlier June 20-22 diagnostic: 659/2,437 (27.04%); the longer window does
  not make the anomaly disappear.
- Daily anchor-suspect share: min 18.21%, median 25.76%, max 35.90%; highest
  June 22 (439/1,223), lowest July 11 (234/1,285). Flags occur throughout
  the window; this does not distinguish CPDI from an anchor or clock problem.
- Selected receiver metrics from `_summary.csv`: ZOI01/ZOI02/ZOI11 clean
  residual RMS 0.0582/0.0564/0.0565 ms; CFD02 0.1741 ms with 273 clean
  epochs beyond 0.5 ms. CFD06 noise fraction 0.6206; ZOI06 0.8759, affected
  by previously identified missing June files. These are diagnostic measures,
  not acceptance claims; plot and source-coverage review remain necessary.
- The script printed over-budget warnings for CFD02/03/04/06/07/08/09 and
  ZOI02/07/10, plus the unsurveyed beacon-host position warning. Include
  per-receiver warnings in Kevin's review; 1 ms timing error is roughly
  1 m positional error.

### Decisions and next steps
- No benchmark value, DBSCAN setting, raw detection, or source file changed.
  Gate 2 is not passed; no accepted clock solution or CPDI diagnosis claimed.
- Compare the timing and receiver concurrence of suspect epochs across 7DB7
  and 7D2D, inspect host ZOI08 and anchor ZOI09 direct/late arrival evidence,
  and review geometry and source gaps with Kevin before reference selection.
- Parameter changes with rationale: none; exactly the earlier fixed settings.

## Follow-up — Step 3 Provisional Clock Readiness and Residual Review

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-01 09:45:10 PDT (UTC-7) = 2026-10-01 16:45:10 UTC
- Tags: #2025 #step-3 #clock #read-only #provisional #gate-2-blocked
- Files touched: this journal only. Existing 30-day DBSCAN epoch CSV read;
  neither SQLite database nor receiver timestamps changed.
- Active context: `output/dbscan_7DB7_30d_20250620_20250720/` (7DB7 from
  ZOI08; ZOI09 anchor; June 20 to July 20, 2025).

### Why this is a diagnostic rather than clock correction
- Pairwise beacon arrivals minus expected geometric travel-time differences
  yield *relative* clock/propagation offsets. A beacon-host position error
  appears as a constant receiver-dependent offset; a 1 ms timing error is
  roughly 1 m of positional error. Do not treat the offsets as calibrated
  absolute clock bias without a verified reference and receiver geometry.
- `legacy_pipeline.metronome()` still calls the 2019 KNN secondary classifier;
  no 2025 integration has been run. `clock_fix()` itself fits five iterations
  of DBSCAN using a per-receiver p90/p95 distance to select eps (legacy core
  lines ~930-960), conflicting with the fixed study-wide eps rule for 2025.
  Calling it unmodified would silently change the 2025 filter; do not do so.
- The v3 DB currently has `tblMetronomeUnfiltered` and
  `tblMetronomeFiltered` from the step-2 primary test, but no
  `tblMetronomeSecondFiltered` or `tblDetectionClockFixed`. BM_Elev and sync
  dates are NULL in that DB. The scratch two-day DBSCAN table contract test
  is not a successful execution of clock_fix in the project database.

### 30-day relative-clock diagnostic (existing classified epochs only)
- 267,113 clean paired receiver epochs for 17 non-host/non-anchor receivers;
  all have fitted `resid_s`. This is an *in-sample* residual, not an independent
  post-correction validation and not a per-receiver GPS-clock audit.
- Across clean epochs, median pairwise offset to ZOI09:
  ZOI05 +16.949 ms (15,268 clean epochs); ZOI06 -8.860 ms (631 clean
  epochs); CFD03 +3.200 ms (11,909 clean epochs); CFD04 -2.084 ms
  (13,989 clean epochs). Causes undetermined: clock offsets, source matching,
  geometry, and propagation cannot yet be separated.
- Clean absolute-residual p95: ZOI05 0.153 ms; ZOI06 0.157 ms;
  CFD02 0.362 ms; CFD07 0.303 ms. Despite p95 values below 0.5 ms,
  the 0.5 ms budget is exceeded on some clean epochs: CFD02 273,
  CFD03 40, CFD04 50, CFD06 14, CFD07 63, CFD08 68,
  CFD09 60, ZOI02 1, ZOI07 7, ZOI10 1. Prominent warnings required in
  any gate report and downstream traceability.
- Clean points span about 29-30 days on each reported receiver, but segment
  counts range from 190 (ZOI06) to 3,289 (CFD02). The current CSV provides
  per-cluster drift (`_segments.csv`), not a validated continuous clock-drift
  model. Do not infer one drift rate over all 30 days.

### Decisions, blockers, next steps
- No correction applied; raw K: and local detection tables unchanged. No
  physical parameter, threshold, sound-speed model, or datum value changed.
- Compare 7DB7 anchor-suspect periods to host/anchor arrivals and 7D2D before
  selecting a common time reference. Kevin must decide the reference and
  review fixed DBSCAN settings, source provenance, hydrophone coordinates/Z,
  and the 2025 benchmark elevation. June 5-16 temperature/WSEL coverage
  remains unresolved for controlled-test positioning.
- Only after those decisions: arrange a *fixed-parameter* 2025 clock-fix path
  without legacy per-receiver eps tuning, produce per-receiver offset, drift,
  residual distributions AND residual plots for every receiver (including
  GPS-synced units), flag >0.5 ms, then request Gate 2 sign-off. Do not run
  Deng positioning before this review.

## Follow-up — Step 3 Clock Fix EXECUTED on 2025 (Provisional) + Session Cleanup

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-01 ~10:30 PDT (UTC-7) = 2026-10-01 ~17:30 UTC
- Tags: #2025 #step-3 #clock-fix #executed #provisional #cleanup
- Files touched: `config/run_data.toml` (bm_elev 861.5 provisional),
  `scripts/step3_clockfix_check.py` (new driver), v3 database derived tables,
  this journal. Deleted: `scripts/step2_metronome_check.py`,
  `output/dbscan_7DB7_step2b/`, `output/jsats3d_2025_v3_step2_scratch/`,
  v3 `tblMetronomeUnfiltered`.

### Parameter changes with rationale
- `BM_Elev` NULL -> 861.5 (feet; = 262.585 m, the 2019 benchmark at the same
  site). PROVISIONAL per project lead + Drew 2026-10-01; Kevin confirmation
  pending. Mathematically inert for 2025 (all receivers BM-referenced; value
  only survives the feet->meters conversion line). Recorded in config comment.
- `synch_time_start/end` written to v3 tblStudyParameters (deployment window,
  decision journaled 2026-09-30). No other parameter changed.

### Step-3 execution (first 2025 clock fix)
- Staged coherent DBSCAN metronome tables into v3 via `beacon_pairwise_dbscan`
  (30-day window, master beacon 7DB7/ZOI08, anchor ZOI09):
  tblMetronomeFiltered 862,092 rows; tblMetronomeSecondFiltered 578,528 rows.
- Ran preserved legacy `clock_fix` (unchanged, including its internal
  iterative DBSCAN refinement — 2019 behavior) via the production driver's
  stage for master ZOI08 + ZOI09, ZOI11, CFD02.
- ToT join: 37,919 master transmissions; clock-data rows match the DBSCAN
  clean-epoch counts exactly (ZOI09 28,081; ZOI11 24,644; CFD02 19,944);
  per-receiver joins lost <=6 rows.
- `tblDetectionClockFixed` written: ZOI08 3,899,607 rows (residual 0,
  master passthrough — correct); ZOI09 614,737; ZOI11 1,354,726;
  CFD02 649,299.
- INTERPRETATION (important): `seconds_residual` is the APPLIED correction
  (estimated clock offset at detection time), not post-correction error.
  Median |correction| 2.1-5.6 ms matches the pairwise diagnostic's relative
  offsets — the clocks genuinely needed ms-scale correction. p95 values of
  1.0-2.8 s indicate periods of large offset (clock jumps >=~390 ms and
  window-edge interpolation) that must be reviewed in the Gate 2 report.
  Post-correction residual validation (corrected arrivals vs expected) is a
  separate, still-outstanding artifact.

### Cleanup (superseded items removed; all reproducible)
- `scripts/step2_metronome_check.py` deleted: purpose served (step-2 gate
  evidence journaled 2026-09-30); its table reset would now clobber the staged
  DBSCAN metronome tables if rerun — removed as a footgun.
- `output/dbscan_7DB7_step2b/` (2-day scratch DB + CSVs) deleted: superseded
  by the 30-day artifacts and the v3-integrated tables.
- `output/jsats3d_2025_v3_step2_scratch/` deleted (empty scratch).
- v3 `tblMetronomeUnfiltered` dropped: produced by the deleted step-2 test
  with legacy host-enumeration epoch numbering, inconsistent with the staged
  DBSCAN anchor `transNo`; nothing in the 2025 path reads it.
- KEPT: `output/dbscan_7DB7_30d_20250620_20250720/` (current diagnostic),
  `output/dbscan_jsats3d_2025_v3/` (7D2D reference for the anchor-suspect
  comparison), `output/jsats3d_2025_v3_step3_scratch/figures/` (clock-drift
  plots + per-receiver clock-fix CSVs — Gate 2 review artifacts),
  `scripts/step3_clockfix_check.py` (active driver).

### Validation
- Full suite after cleanup: 32 passed. v3 table set verified. K: untouched.

### Next steps
1. Post-correction residual validation and per-receiver residual plots for
   the full surface set; quantify corrected-arrival error vs the 0.5 ms budget.
2. Kevin review: BM_Elev value, reference clock (anchor-suspect 26%),
   steady-reflection rule, CFD inclusion; then full-array clock fix.
3. Only then: step 4 (deep receivers via Deng) on 2025.

## Follow-up — No-New-Scripts Consolidation and Handoff Correction

- Author: Copilot
- Date: 2026-10-01 (time not recorded).
- Tags: #workflow #no-new-scripts #tests #documentation
- User constraint: do not add new scripts; improve existing project files instead.

### Changes
- Consolidated the piecewise GPS interpolation helpers into the existing
  `scripts/cfd_gps_diagnostics.py`; updated the existing contract test import.
  Removed the standalone `scripts/gps_position_interpolator.py` module.
- Removed the standalone `scripts/step3_clockfix_check.py` driver. Added an
  opt-in `clock-fix-check` command to the existing `scripts/legacy_pipeline.py`.
  It refuses to run if `tblDetectionClockFixed` already exists, rather than
  dropping prior results. This is newly added diagnostic behavior, not a
  pre-existing legacy workflow step, and should be retained only if approved.
- Added a contract test verifying the clock-fix check refuses existing results
  and leaves their row intact.
- Corrected `LLM_Prompts.txt`: 2019 run status reflects the observed failures,
  and the cause is not asserted as proven. No edits were made to the attached
  owner-input checklist during this follow-up.
- The checklist's provisional datum wording is documented earlier in this
  journal under "Drew's Response on Datum, Duplicates, and 7DB7"; the project
  lead's provisional datum choice is not equivalent to confirmed benchmark
  elevation or Gate 2 approval.

### Validation and Limits
- Focused contract suite: 31 tests passed.
- Full suite: 33 tests passed.
- `git diff --check` passed; no untracked Python scripts remained afterward.
- No 2019 or 2025 positioning/clock-fix workflow was run during this follow-up.
- No database or K: source data was accessed or changed during these edits.
- The previously recorded "active driver" note above is historical; the
  standalone driver was subsequently removed and its guarded diagnostic was
  consolidated into the existing pipeline as described here.

### Open Decision
- User asked whether the added clock-fix diagnostic could be removed. No
  removal was requested yet; `clock-fix-check` and its test remain pending
  user decision and must not be represented as required legacy behavior.
