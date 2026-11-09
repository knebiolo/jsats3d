# Handoff for Qwen 3.6 35B MoE — Clock-Jump Workstream (state as of 2026-11-07)

> **SUPERSEDED.** Use `docs/llm_handoff_qwen36_35b_moe_RECOVERY_2026-11-07.md` instead. This file predates the
> Arm A coverage comparison (the >120 s gaps are a legacy property, not a regression) and the audit corrections.

Supersedes `docs/llm_handoff_clock_jumps_2026-11-06.md` for current status. That file is
still the reference for background, architecture, and the Arm A/B comparison method.
Read it, then read this one. **This file wins if they disagree.**

How to use this file: work **one numbered task at a time**. Each task has a **STOP** line.
When you reach a STOP, report to the user and wait. Do not combine tasks.

---

## 0. Rules you must follow (short form — full text in `System_Prompt.txt`)

1. **1 ms timing error ≈ 1 m position error.** Say this when you discuss timing.
2. **0.5 ms** is the timing warning budget. **50 ms** is the jump-application minimum.
   They are different things. Do not mix them.
3. **Raw data is read-only.** Never write under `...\Documents\jsats_data\` or `K:\`.
4. **Never invent values.** No back-filled temperatures, no copied filter tables,
   no made-up timestamps. Missing data stays missing and is reported.
5. **Never weaken a gate.** Never set candidate rows to `accepted`, validation rows to
   `passed`, or gate JSON to `passed` by hand. Gates change only through code the
   owner approved, with tests.
6. **Arm A is the baseline. Do not write to `output/10292026_Test/`.** Open it with
   `mode=ro` and `PRAGMA query_only=ON`.
7. **Ask before deleting files.** Ask before any change to defaults or thresholds.
8. **Journal every session** in `.ai_journal/session/YYYY-MM-DD_topic.md`, full prose.
9. Chat replies to the user: short, 3–6 word sentences.
10. Env: `conda run -n jsat_3d ...`. Tests:
    `conda run -n jsat_3d python -m unittest discover -s tests`.
    Exit code 1 is normal for `conda run`; read the `Ran N tests ... OK` line.

---

## 1. The PM request (what "done" means)

| # | PM ask | Status today |
|---|---|---|
| 1 | Keep internal-code jump detection + TDOA magnitude | **Done.** 696 stage-1 jumps. |
| 2 | Supplementary method for jumps the codes miss (DBSCAN suggested; "how many clusters?") | **Implemented.** 1,653 stage-2 jumps (49 master, 1,604 receiver). Boundary quality is the open problem. |
| 3 | No iterative for-loop correction | **Respected.** Supplement uses ≤4 bounded re-measure passes; do not expand it. |
| 4 | Keep comparing Arm A vs Arm B, especially TDOA plots | **Behind.** No A/B TDOA plots or fish comparison exist for the current Arm B build. |

**Answer to "how many clusters" (give this, do not tune k):** DBSCAN has no cluster
count. With the fixed physical eps (0.5 ms × 2.5 beacon periods, min_samples 3), each
real clock step starts a new cluster. The number of clusters is the number of stable
clock regimes in the data. Tuning a count per receiver would be hidden parameter tuning,
which the project forbids.

**Ultimate goal:** synchronized clocks → fish positions from Arm B → Arm A vs Arm B
comparison that shows no degradation → PM report with TDOA plots and fish tracks.

---

## 2. Verified current state (checked 2026-11-07 by reading files and DBs)

**Databases**
- Arm A: `output/10292026_Test/jsats3d_2025_earlyfish.db` (baseline, unshifted).
- Arm B: `output/11062026_clocksync/jsats3d_2025_earlyfish.db` (compact, 8 tags,
  2,349 applied jumps: 696 stage 1 + 1,653 stage 2). Run file:
  `config/run_data_earlyfish_2026-11-03_clocksync.toml`.
- Older folders `output/11032026_clocksync/`, `output/11062026_clock_recheck/`,
  `output/jump_ab_report_2026-11-04/` are **gone from disk**. Journals are the record.

**Arm B gate (real, enforced):** blocked, 1,652 issue rows, all 9 solving receivers.
- 572 rows "supplementary boundary uncertain" → 3,541 fish rows.
- 961 rows "applied jump validation unverifiable" → 8,880 fish rows.
- 8 rows "outside clock coverage / unsupported gap / LOO > 0.5 ms" → **88,923 fish rows**.
- 92 overlapping-regime + 19 residual-step rows → 0 fish rows.
- DB state verified: candidates 1,425 accepted / 1,173 unresolved; validation
  502 passed / 1 failed / 1,797 unverifiable; no `tblPositions*` tables; no
  "owner approved" notes remain.

**Code (uncommitted on branch `ENM_jsat3d_edits`) — verified correct:**
- `review_gate` raises when unresolved timing touches fish (`scripts/_2025_pipeline.py`).
- Fish loader raises on unsupported timing when the experiment policy table exists
  (`jsats3d/jsats3d.py`). Legacy/2019 DBs are unaffected.
- `firmware_record_adjustments(..., counter_seconds=...)` now certifies the group-2
  counter against original time (`seconds + jump_offset_s`). This fixed a real bug:
  30,486 of 35,336 flagged rows failed only because jumps were applied in place.
- Full suite: **Ran 69 tests ... OK** (verified).

**Step 1 result (from the 11-07 session, verified):** all 36 fish-overlapping
boundary-uncertain windows are `evidence_limited`, 0 supported. Report:
`output/11062026_clocksync/clock_review/step1_resolution_report_2026-11-07.md`.

---

## 3. Problems found that you must know about

### 3.1 Arm B temperature contains fabricated values (rule 4 violation)
`tblInterpolatedTemp` in Arm B was rebuilt on 11-06 by `_fix_temp_v9.py` from the
temperature string only, then `reindex().ffill().bfill()`:
- **576 rows (2025-06-15 00:00 to 06-16 23:55) are a back-filled constant 13.924 °C**,
  labeled `DD_N_string`. The string starts 06-17. Real data for those days exists in
  the HOBO files (Arm A recorded 11.5–14.0 °C there).
- Measured Arm B minus Arm A recorded temperature (CFD01 clock epochs):
  - before 06-17: mean **+1.26 °C**, max +2.41 °C (≈ +4 to +8 m/s sound speed);
  - 06-17 to 07-10 (HOBO vs string): mean +0.03 °C, max 0.32 °C (≈ ≤1 m/s);
  - after 07-10: identical.
- 8 fish detections fall before 06-17. The main effect is on clock curves for
  06-15/16 and on A/B comparability.
- The 11-07 journal called this table "no fabrication". That is wrong.

### 3.2 Arm A was written to
`_add_temp_to_arm_a.py` added a string-only `tblInterpolatedTemp` (26,662 rows from
06-17) to Arm A. Arm A's existing positions were computed earlier with a HOBO+string
table (30,817 rows from 06-02), so its results are unchanged, but the DB is no longer
pristine. Do not write to it again. Record this.

### 3.3 Leftover scripts (39 at repo root + several in `scripts/`)
Root `_*.py` files from the 11-06 evening session, and untracked `scripts/fix_gate.py`,
`scripts/copy_filters.py`, `scripts/option_c_*.py` (v4 experiment). `fix_gate.py` and
`copy_filters.py` are **gate-bypass / data-mixing tools**. Never run them.
Ask the owner before deleting; propose moving them to `output/_quarantine_scripts/`.

### 3.4 The gate is dominated by curve coverage, not jump boundaries
Step 1 can unblock at most 3,541 of 91,430 blocked fish rows. The 88,923-row
curve-coverage class decides whether fish can ever run. The 11-03 audit found the
cause area: 1,157 gaps had valid upstream beacon joins but **no final clock knots** —
epochs are being dropped inside `clock_fix` (five adaptive DBSCAN passes).

---

## 4. The plan — do these in order

### Task 1 — Hygiene (no science changes)
1. Run the full test suite; confirm `Ran 69 tests ... OK`.
2. List leftover scripts (root `_*.py`, `scripts/fix_gate.py`, `scripts/copy_filters.py`,
   `scripts/option_c_*.py`, other untracked `scripts/check_*.py`, `compare_*.py`).
3. Ask the owner: quarantine-move or delete? Do what they choose.
4. Journal the inventory and decision.

**STOP. Report: tests result, script list, owner choice.**

### Task 2 — Fix Arm B temperature (needs owner OK, then a clock rerun)
Explain first: sound speed depends on temperature; a 1 °C error ≈ 3–4 m/s, which
scales every range and the clock-fix distance terms.
1. Ask the owner to approve replacing Arm B `tblInterpolatedTemp`.
2. Rebuild it with the project loader `adapt_2025_to_legacy.load_temperature_string`
   (HOBO `Tag Drag Period/DD_N` + string, as in the 10-30 journal). Paths are in the
   run TOML. **No ffill/bfill outside data coverage.**
3. Verify against Arm A's recorded `avg_C` in
   `output/10292026_Test/jsats3d_2025_earlyfish_legacy/figures/receiver_*_clock_fix.csv`:
   expected max |ΔC| ≈ 0 for all periods.
4. Rerun clock steps only:
   `python scripts/_2025_pipeline.py config/run_data_earlyfish_2026-11-03_clocksync.toml --from metronome --to deep_clocks`
   (check C: free space first: need DB size + 4 GiB).
5. Re-run the gate (it will still block). Record new issue counts.

**STOP. Report: ΔC check, new gate counts.**

### Task 3 — PM deliverable: Arm A vs Arm B TDOA plots (read-only)
This is the part of the PM request that is behind. Do it before more method work.
1. Use the pipeline's existing review step:
   `python scripts/_2025_pipeline.py <run toml> --only clock_review --clock-review-interval "2025-06-27 06:00:00" "2025-06-30 06:00:00"`
   It writes `tdoa_arms_<receiver>.png` (Arm A unshifted vs Arm B) plus CSVs under
   `output/11062026_clocksync/clock_review/interval_<start>_<end>/`.
2. Repeat for 2 more intervals with heavy fish presence (pick from the gate CSV,
   highest `raw_fish_rows`), e.g. a ZOI06 window and a ZOI01 window.
3. Make one summary table: per receiver, count of adjacent TDOA changes ≥ 50 ms,
   Arm A vs Arm B. Fewer in B = jumps removed. More in B anywhere = investigate.
4. Look at every PNG. Write what you see in the journal.

**STOP. Report: the table and which receivers improved or got worse.**

### Task 4 — Owner decisions (ask, do not decide)
Present these with numbers. Use the question tool. One decision per question.
1. **Sub-50 ms policy.** A 0.907 ms step was found by 3 beacons agreeing within 24 µs
   (journal `2026-11-06_sparse_clock_step_trial.md`). Options: opt-in sparse-consensus
   application (min 0.5 ms, 3 references, holdout check) / flag only.
2. **Gate mode.** Today one bad window blocks the whole fish run. Option: a
   "exclude-and-flag" mode — fish detections inside unresolved windows or unsupported
   curve spans are excluded from Arm B positioning, counted, and the **same detections
   are excluded from Arm A** for a matched comparison. Needs code + tests.
3. **Validation-unverifiable (961 rows / 8,880 fish).** Block, or treat by gate mode.
4. **Provenance guard scope** for the 3 ZOI06 windows refused by duplicate re-logged
   rows in ZOI09/ZOI10 (not target or reference receivers): keep strict, or scope the
   duplicate check to the receivers used in the measurement.

**STOP. Record each answer in the journal and `LONG_TERM_CONTEXT.md` Standing Decisions.**

### Task 5 — Improve curve coverage (the biggest lever)
Goal: fewer "no final clock knots" gaps, so fewer fish rows are blocked.
1. Pick 3 gaps from `clock_curve_warnings.csv` that overlap many fish rows.
2. For each, trace beacon epochs through the stages: raw first arrivals →
   `tblMetronomeSecondFiltered` (multipath_prediction=0) → master join →
   `tblDetectionClockFixed`. Record which stage drops each epoch.
3. If the five `clock_fix` DBSCAN passes drop clean epochs, write that up with counts.
   **Do not change eps/min_samples/passes without owner approval and an A/B metric.**
4. Propose a fix with expected effect. Implement only after approval, with tests.

**STOP after the trace report.**

### Task 6 — Clean fish run and full A/B (only when the gate allows it)
1. Run fish steps through the pipeline main loop only (it runs the gate):
   `--from fish_multipath --to positions_dbscan`. Never call step functions directly
   to skip the gate. Never drop `tblClockExperimentPolicy`. Never copy Arm A filters.
2. Compare with the established recipe (2026-11-04 journal):
   match transmissions by **time** (nearest ≤ 1 s), not `transNo`; raw root-B
   validity ≤ 2 km of array centroid; affected = ±300 s of an applied jump; report
   rescued / broke / no-impact; curve steps at jump times; 2–3 fish track plots
   around jumps; yield (detections in → positions out).
3. Any "broke" case outside a documented window = **blocking defect**. Report it.

**STOP. Write the PM report draft. Owner reviews before any PR.**

---

## 5. Known traps (do not repeat)
- `"26E4"` parses as a float in pandas. Use `dtype={"tag": str}`.
- `a.loc[i] - b.loc[j]` aligns by label → NaN. Use `.to_numpy()`.
- `pd.to_datetime(...).astype('int64')` unit bugs: use explicit `datetime64[ms]`.
- `row.get(k, row[other])` evaluates the fallback first → KeyError.
- `clock_rapid_reference_review` overwrites `<rec>_source_state_boundaries.csv` in its
  folder. Give each run its own subfolder.
- PowerShell breaks multi-line `python -c` and nested quotes. Use one-line commands or
  existing scripts. Do not leave new script files behind.
- C: drive free space swings (OneDrive/Egnyte). Check before any DB write.
- Study time = PDT wall clock stored as if UTC.

## 6. File map
| What | Where |
|---|---|
| Jump stages, consensus, firmware decode | `scripts/clock_jump_pre.py` |
| Pipeline, gate, reviews | `scripts/_2025_pipeline.py` |
| Fish loader guards, clock_fix | `jsats3d/jsats3d.py` |
| Fixed DBSCAN | `scripts/beacon_pairwise_dbscan.py` |
| Temperature loader | `scripts/adapt_2025_to_legacy.py` (`load_temperature_string`) |
| Tests | `tests/test_clock_jump_pre.py`, `tests/test_clock_study_gate.py`, `tests/test_clock_segments.py` |
| Latest journals | `.ai_journal/session/2026-11-06_*.md`, `2026-11-07_gate_tamper_restore_step1.md` |
| Gate outputs | `output/11062026_clocksync/clock_review/` |
