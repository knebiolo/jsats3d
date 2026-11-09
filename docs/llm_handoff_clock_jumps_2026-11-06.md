# LLM Handoff — Clock-Jump Correction Workstream (for Qwen 3.6 35B MoE)

**Date:** 2026-11-06. **Author:** previous AI assistant (GitHub Copilot), reviewed state as of this date.
**Audience:** a successor AI assistant taking over this workstream.
**Read this entire file before touching any code or data.** Then follow the staged
context-load policy in `System_Prompt.txt` Section 12 (read the latest
`.ai_journal/session/` files, `LONG_TERM_CONTEXT.md`, `README.md`).

---

## 1. The PM request you are fulfilling (verbatim intent)

1. **Keep the current two-stage methodology.** Clock jumps are identified from the
   receivers' internal status codes, and each jump's magnitude/direction is quantified
   from master-beacon TDOA. This works but **misses some jumps** (not every jump carries
   an internal code).
2. **Add a supplementary process** that finds the remaining jumps that are visible in
   the TDOA beacon plots. DBSCAN is the suggested tool; the PM asked "how many clusters
   are required to optimize results" — see Section 4.2 for the answer we adopted.
3. **Do NOT build an iterative for-loop correction through time.** The PM explicitly
   withdrew that idea; the current methodology stands. (Note: the implemented supplement
   uses a small bounded number of re-measurement passes, max 4, which is not the
   withdrawn chronological for-loop design — but do not extend it toward one.)
4. **Always validate with Arm A (original) vs Arm B (new) comparisons**, especially
   TDOA plots per receiver, to prove the modifications do not degrade results.

**Ultimate goal:** fully synchronized receiver clocks → trusted fish positions from the
Deng solver → a clean Arm A vs Arm B fish comparison showing no degradation and ideally
improvement, with every correction traceable to recorded evidence.

---

## 2. Non-negotiable project rules (summarized from System_Prompt.txt — read it)

- **1 ms of timing error ≈ 1 m of position error.** State this whenever discussing timing.
- **Timing warning budget: 0.5 ms.** Residuals above it are flagged loudly, never hidden.
- **STRICT READ-ONLY RAW DATA.** Never write to anything under
  `...\Documents\jsats_data\` or `K:\...`. Open SQLite sources with
  `mode=ro` URI + `PRAGMA query_only=ON` for audits.
- **Never fabricate physical values.** No invented timestamps, arrivals, sensor values,
  speeds of sound, or "plausible" offsets. Missing = NULL/unresolved, loudly.
- **No silent parameter tuning.** Any changed threshold/eps/window requires a journal
  entry with rationale and a before/after validation metric. This has been enforced all
  along — continue it.
- **Validation gates are hard stops.** When the clock review gate reports unresolved
  timing affecting fish, the pipeline RAISES. Do not weaken gates to "warn and proceed"
  — that was done once in a throwaway experiment (`output/11032026_clocksync_v4`) and
  had to be reverted; treat it as a defect pattern.
- **Journal everything** under `.ai_journal/session/YYYY-MM-DD_topic.md` (full prose,
  all fields: files touched, decisions, parameter changes + rationale, blockers, next
  steps). Chat replies to the user are terse (3–6 word sentences); journals are not.
- **Conda env `jsat_3d`.** Run tests:
  `conda run -n jsat_3d python -m unittest discover -s tests`.
  `conda run` drops piped stdin — write inline `python -c "..."` one-liners or use
  existing scripts; never leave one-off script files behind.
- **Owner commits; you do not.** Work stays on branch `ENM_jsat3d_edits`.
- Delete every throwaway diagnostic script when done; keep their CSV/JSON/PNG outputs
  in `output/`.

---

## 3. Architecture you are working in

- Legacy core: `jsats3d/jsats3d.py` (metronome, `clock_fix` piecewise regression with
  five adaptive DBSCAN passes, Deng closed-form positioning, study-tag correction via
  `np.interp` over per-receiver `(seconds, seconds_residual)` knots from
  `tblDetectionClockFixed`).
- 2025 pipeline driver: `scripts/_2025_pipeline.py` (steps: prepare → `clock_jumps`
  (step 1c, opt-in) → metronome → clock_surface → deep_beacons → deep_positions →
  deep_clocks → fish_multipath → positions → export → positions_dbscan). Config via
  TOML run files in `config/`.
- Pre-sync jump module: `scripts/clock_jump_pre.py` — everything below lives here.
- Beacon DBSCAN utility: `scripts/beacon_pairwise_dbscan.py` — `cluster()` with FIXED
  parameters: 0.5 ms budget, 2.5-period time window, min_samples 3, Chebyshev metric.
- Master beacon: tag `7DB7` hosted on `ZOI08` (nominal 60 s period). Deep beacons:
  `FFC7`@ZOI01, `7D2D`@ZOI02, `7DBC`@ZOI03. Every receiver also hosts its own beacon.
- Study window for the early-fish dataset: 2025-06-15 → 2025-08-06; four fish tags
  `26E4, 9F8C, 1D5A, B5ED`. Time basis: **PDT wall clock stored as-if-UTC epoch**.
- Key databases (all under `output/`):
  - `10292026_Test/jsats3d_2025_earlyfish.db` — **Arm A** (unshifted baseline).
  - `11032026_clocksync/jsats3d_2025_earlyfish.db` — **Arm B** compact experiment DB
    (3.88 GiB, 8 tags incl. beacons, all-tag `tblClockEventEvidence`, built by
    `scripts/run_data.py --compact-from`). Its manifest is the `.compact.json` beside it.
  - Run file: `config/run_data_earlyfish_2026-11-03_clocksync.toml`
    (`apply_clock_jumps=true`, `clock_segment_safe=true`, `clock_tdoa_supplement=true`).

---

## 4. What has been implemented and validated so far

### 4.1 Stage 1 — internal-code jumps (the PM's "current methodology")
`clock_jump_pre.run()` / `quantify_jumps()`: candidates = `OffsetChanged` or
`CounterRestart` rows in `tblDetectionRaw` (dense `OneSecondAdjustmentEvidence` only
corroborates, never triggers). Magnitude = median master-beacon first-arrival TDOA
±300 s per side, clipped at neighbouring candidates; accept 0.05 ≤ |offset| ≤ 5.0 s;
≥4 epochs per full side (3 clipped, ≥150 s). Applied with cumulative per-receiver
`UPDATE` on `tblDetectionRaw.seconds` + audit column `jump_offset_s` + ledger
`tblClockJumpEvents`; downstream tables invalidated. **696 jumps** applied on the
early-fish data (2026-10-30 A/B: rescued 1 gross error, broke 0, curves continuous at
all measurable jump times).

### 4.2 Stage 2 — the TDOA-DBSCAN supplement (the PM's new ask — DONE, with caveats)
`clock_jump_pre.tdoa_supplement()`: after stage 1, recompute each listener's
master-beacon TDOA series, cluster with the **unchanged** `beacon_pairwise_dbscan.cluster`.

**Answer to the PM's "how many clusters" question — give this answer, do not tune a
cluster count:** DBSCAN has no k. The number of stable regimes is *data-driven*: the
fixed eps (0.5 ms budget × 2.5-period window, physically derived, Section 7.4 of the
system prompt) means every genuine clock step starts a new cluster, so "cluster count"
= number of stable clock regimes the data actually contains. Optimizing a cluster count
would be silent per-group parameter tuning, which this project forbids.

Level changes ≥50 ms between consecutive clean regimes = jumps without a usable internal
code. Identical simultaneous steps on ≥3 listeners (and ≥⅓ of covered listeners) are
attributed to the **master** (offset = −listener step); remaining steps per listener.
Internal evidence inside the regime gap pins the boundary; otherwise
`boundary_uncertain=1` → unresolved candidate for the strict gate. Runs in one
transaction, max 4+1 passes, refuses double application; `revert_tdoa_supplement()`
restores bit-identical pre-stage-2 state from the unshifted source.
**Result on real data: 1,653 supplementary jumps (49 master, 1,604 receiver; 830
boundary-uncertain; 249 residual steps unresolved).** PM TDOA plots improved on every
receiver (≥50 ms adjacent changes 1,758 → 1,544 across the June 27–30 interval).

### 4.3 Arm A / Arm B validation (the PM's continuing requirement — pattern established)
- Report folders: `output/jump_ab_report_2026-10-29/` (journal
  `2026-10-30_clock_jump_pre_sync_ab.md`) and the 2026-11-04 full validation
  (journal `journals/2026-11-04_ab_fish_validation_clocksync.md`; its report folder
  `output/jump_ab_report_2026-11-04/` is **no longer on disk** — the journal is the record).
- 2026-11-04 headline: Arm B curves continuous at all 463 measurable applied jumps
  (max residual step 0.256 ms vs Arm A's ~420 ms); raw root-B rescued 6 / broke 8
  (all 8 inside or adjacent to documented unresolved windows); +1,064 consensus fixes
  (+14.4%); no-impact matched fixes p50 8.3 cm. **This was a diagnostic on a throwaway
  copy with the policy table dropped — it is NOT production acceptance.**
- Comparison recipe (reuse it): match transmissions **by time (nearest ≤1 s), never by
  transNo** across arms; validity = root-B median ≤2 km from array centroid at the raw
  level (consensus level is degenerately in-array by construction); affected =
  ±300 s of an applied jump; always report rescued/broke/no-impact + curve steps at
  jump times. Per-receiver TDOA arm plots are produced by the pipeline's optional
  `clock_review` step (`tdoa_arms_<rec>.png`).
- Known pandas traps already hit (avoid repeating): tag `26E4` parses as float
  (`dtype={"tag": str}`); `.loc` label alignment makes cross-frame arithmetic NaN
  (use `.to_numpy()`); `datetime64[us]` breaks ns-based epoch math.

### 4.4 The strict review gate (current state: ENFORCED)
`_2025_pipeline.review_gate()` writes `clock_review_gate.csv/.json` then **raises
RuntimeError** when unresolved candidate windows, failed/unverifiable applied-jump
validations, or unsupported/missing corrected-time coverage overlap fish detections on
the solving receivers. `jsats3d.py`'s experimental study-tag loader likewise raises on
unsupported detections / missing curves when the experiment policy table is present
(`_clock_experiment_enabled`). Legacy databases without the policy table are untouched
(2019 parity preserved). The production Arm B gate is **blocked: 630 issues** —
fish steps have never legitimately run on it.

### 4.5 2026-11-06 session work (latest; journals
`2026-11-06_clock_sync_request_status.md` and `2026-11-06_sparse_clock_step_trial.md`)
- Restored the gate raises (they had been weakened to warnings by the v4 experiment).
- Supplement boundary honesty fix: `_supplement_evidence` no longer collapses distinct
  internal triggers within 1 s into one "certain" boundary (only exact-duplicate
  timestamps dedupe). Read-only audit: 10 of 223 ZOI06/ZOI02 stage-2 "certain"
  boundaries actually contain multiple distinct triggers → they must be re-derived,
  not trusted.
- A/B diagnostic no longer double-counts the same DB as two arms.
- **Original-source boundary reconstruction** (`_2025_pipeline.clock_source_boundary_review`
  + `clock_source_median_review`): re-reads the raw ATS CSV (read-only) to find the TRUE
  first new-state detection. Real finding: ZOI06's 23:57 reset actually starts at
  23:57:00.698508 (SourceRow 89151), **12.88 s earlier** than the retained DB trigger;
  counter 0→1 across a GPS-Fix-only line is ONE reset episode (`CounterContinuation`,
  `ContinuationAcrossGPS`), collapsing 5 apparent ZOI06 events to 4.
- **Sparse three-beacon consensus** (`rapid_reference_consensus(..., include_isolated=True)`):
  an isolated event may be measured from 3 distinct beacon hosts' source-bounded
  brackets (≤2 periods, no competing internal event in any bracket, agreement ≤0.5 ms).
  Reference hosts are chosen by recorded reception coverage (ZOI06's best: ZOI05, CFD01,
  ZOI04, ZOI11 — ZOI11 is diagnostic-only, never solving geometry).
- Real results on ZOI06 June 28 (fish-heavy interval): 23:24 event = **no step**
  (3 refs agree within 25 µs of zero → evidence_only). 23:57 and 01:01 events remain
  **unresolved** (contaminated/over-gap brackets). 01:12:01 event = **supported
  0.907 ms step** (ZOI05/CFD01/ZOI11 agree within 24 µs) — below the 50 ms application
  floor but above the 0.5 ms budget.
- **Owner-approved isolated trial** (diagnostic `consensus_min_step=0.0005`, production
  defaults untouched): applying 0.907 ms in a copied in-memory TDOA coordinate made the
  fitting-reference step 0.931 → 0.024 ms, and an **unused holdout beacon** (CFD08,
  uncontested 118 s bracket) went +0.463 → −0.444 ms (within budget).
  Artifacts: `output/11062026_clock_recheck/zoi06_small_step_trial/`
  (`trial_summary.json`, `holdout_cfd08_summary.json`, `tdoa_small_step_before_after.png`).
- Test state: ~50 focused tests green across `tests/test_clock_jump_pre.py` and
  `tests/test_clock_study_gate.py`; plus `test_clock_segments.py` (11) covering the
  restored strict loader guards.

---

## 5. What is NOT done — the remaining plan, in order

Work top-down; each step has a hard exit criterion. Ask the owner before any step that
applies offsets, changes a default, or needs >2 edited files.

### Step 1 — Resolve the gate's issue list with evidence, not thresholds
The production gate blocks on 630 issues (519 boundary-uncertain, 92 overlapping-regime,
19 residual-step; receivers ZOI01–03, 05–08). For each fish-overlapping window, in
priority order of raw fish rows (the June 27–28 ZOI01/ZOI02/ZOI03/ZOI06 cluster first):
1. Run `clock_source_boundary_review` on the receiver's interval → true source-order
   boundaries (this alone fixed ZOI06's false 5th event and 12.88 s boundary error).
2. Run `clock_source_median_review` (default master reference first, then
   `reference_host`/`anchor_receiver` alternates) under the UNCHANGED median rules.
3. Where medians can't resolve, run `clock_rapid_reference_review(...,
   include_isolated=True)` — 3 uncontested source-bounded beacon brackets, ≤0.5 ms
   agreement. Choose references by recorded coverage, verify each reference's own
   original source states (competing events disqualify the bracket).
4. Classify every event: supported step / supported no-step (evidence_only) /
   genuinely evidence-limited (stays unresolved and blocked). **Do not lower epoch
   minimums, widen brackets across competing events, or average disagreeing references.**
Exit: every fish-overlapping window is either supported or documented evidence-limited.

### Step 2 — Decide the sub-50 ms policy with the owner
The 01:12 trial proves the ordinary 50 ms application floor misses real >0.5 ms steps
that three independent beacons can certify. Present to the owner: adopt a
sparse-consensus application path (min step = 0.5 ms, three-reference rule, holdout
check) as an explicit opt-in, or record such steps as flagged warnings only. Record the
decision + rationale in the journal and `LONG_TERM_CONTEXT.md`. Do not change the
default without sign-off.

### Step 3 — Re-derive the 10 falsely-certain stage-2 boundaries
On a fresh compact copy (never the already-shifted Arm B), re-run stage 1 + stage 2 so
the corrected `_supplement_evidence` yields honest `boundary_uncertain` flags, and the
source-order boundaries from Step 1 pin events where possible. Verify with
`revert_tdoa_supplement` round-trip equality if iterating.

### Step 4 — Clean Arm B rebuild and full pipeline run
Check disk first (the compact DB needs ~4 GiB + 4 GiB reserve + rollback allowance;
C: has hovered at 6–12 GiB free and transiently hit zero from Egnyte/OneDrive caches —
a test run failed on `database or disk is full` and succeeded on retry).
Build: `python scripts/run_data.py config/<new>.toml --compact-from
output/10292026_Test/jsats3d_2025_earlyfish.db --temperature-source <donor> ...`, then
`python scripts/_2025_pipeline.py <toml> --from clock_jumps --to deep_clocks`, then —
**only if the gate passes** — the fish steps. The gate raising is the designed stop,
not a crash; never drop `tblClockExperimentPolicy` or copy Arm A filter tables into
Arm B (both happened in discarded experiments and invalidate the comparison).

### Step 5 — Full Arm A vs Arm B validation for the PM
Reuse the 2026-11-04 recipe (Section 4.3) plus:
- per-receiver pre-sync TDOA arm plots (`clock_review` step) — the PM explicitly wants
  these;
- curve-step table at all applied jump times (measurable count, median/p99/max, ≥50 ms
  count);
- raw root-B rescued/broke/no-impact with every broke case traced to a documented
  window or escalated as a regression (a clean-region broke case is a **blocking
  defect**, Section 11 of the system prompt);
- fish-track examples around applied jumps for 2–3 fish;
- positional yield accounting (detections in → positions out, rejections by stage).
Exit: written gate report + owner sign-off before any PR.

### Step 6 — Acceptance context (do not skip when reporting "better positions")
Clock error is currently NOT the dominant accuracy limit — per-ping geometry/multipath
scatter is (journal 2026-10-11: post-fix clock error ≤0.21 ms on all receivers; X-scatter
σ≈9 m). Set expectations with the PM accordingly: clock-jump completeness buys yield,
curve continuity, and removal of gross errors (548 km → 20 m class), not centimetre
accuracy. Stationary-hold RMSE and drag-track overlays remain the acceptance metrics
(system prompt Section 11).

---

## 6. Operational gotchas (hard-won; do not rediscover)

- `conda run -n jsat_3d` returns exit 1 on stderr output even when tests pass; read the
  unittest summary line, and it drops piped stdin (no here-strings).
- PowerShell mangles multi-line `python -c` with nested quotes; use single-line
  one-liners with walrus/listcomps, or existing scripts.
- SQLite `DROP TABLE IF EXISTS x` can resolve into an ATTACHed read-only DB — qualify
  `main.x`. `dbstat` is unavailable in this build.
- pandas `.get(key, row[other])` evaluates the fallback eagerly → KeyError when the
  fallback column is absent; use explicit `if key in row` (this bit us in
  `_evidence_string`).
- An empty beacon result gives an object-dtype timestamp column (CFD05 has no master
  detections) — handle explicitly.
- The raw-file `Internal` decode guide is
  `K:\Jobs\5662\001\Background\Instrument Information\ATS_3017_Internal_Column_Guide.txt`
  (File Format 2.0, firmware 10.62): flag `1` = −1 s per-record adjustment, `>` = +1 s;
  group 2 = 15 s counter position (hundredths, cross-check only); group 3 = GPS minute,
  NOT a calibrated jump. `firmware_record_adjustments()` implements this with strict
  provenance checks — diagnostic coordinate only, never a raw repair.
- ZOI05 logs UTC (+7 h vs others) — already normalized in the v2+ databases; don't
  re-shift. ZOI05/ZOI06 hydrophone coordinates were confirmed swapped and are already
  corrected in current builds — never swap twice.
- `ZOI09` carries a constant +0.946 s hardware offset added back at fish-correction
  time (`ZOI09_OFFSET_S` in jsats3d.py) — leave it unless the owner re-derives it.
- Each Deng job loads ~4.5 GB RAM; ≤5 parallel. Full compact build ~24 min; clock steps
  ~45 min total on the compact DB.

## 7. Where to look first (file map)

| What | Where |
|---|---|
| Stage 1 + stage 2 + sparse consensus + trial | `scripts/clock_jump_pre.py` |
| Pipeline steps, gate, source-boundary/median/rapid/joint reviews | `scripts/_2025_pipeline.py` |
| Strict fish-loader guards, clock_fix, segment-safe interp | `jsats3d/jsats3d.py` |
| Fixed DBSCAN (eps/budget/window/min_samples) | `scripts/beacon_pairwise_dbscan.py` |
| Tests (keep green; add for every behavior change) | `tests/test_clock_jump_pre.py`, `tests/test_clock_study_gate.py`, `tests/test_clock_segments.py` |
| Arm B run config | `config/run_data_earlyfish_2026-11-03_clocksync.toml` |
| Latest findings | `.ai_journal/session/2026-11-06_*.md`, `journals/2026-11-04_ab_fish_validation_clocksync.md` |
| Durable facts/decisions | `LONG_TERM_CONTEXT.md` |
| Current diagnostic artifacts | `output/11062026_clock_recheck/`, `output/11032026_clocksync/clock_review/` |

**First actions for the successor:** (1) staged context load per System_Prompt §12;
(2) `conda run -n jsat_3d python -m unittest discover -s tests` to confirm green;
(3) read the gate CSVs in `output/11032026_clocksync/clock_review/`; (4) start
Step 1 of Section 5 on the highest fish-exposure window; (5) journal before you stop.
