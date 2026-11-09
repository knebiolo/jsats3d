# Handoff for Qwen 3.8 27B Dense — Clock-Jump Workstream (state as of 2026-11-07)

You (or a prior instance of this model) ran the 2026-11-07 session recorded in
`.ai_journal/session/2026-11-07_gate_tamper_restore_step1.md`. This file gives
(1) an independent review of that session, (2) problems it missed, and (3) the plan
to finish the PM request. Background, architecture and the A/B recipe are in
`docs/llm_handoff_clock_jumps_2026-11-06.md`; a task-by-task version of the same plan
is in `docs/llm_handoff_qwen36_35b_moe_2026-11-07.md`. **This file wins on current status.**

Project rules from `System_Prompt.txt` all apply. The ones that matter most here:
1 ms ≈ 1 m; 0.5 ms timing budget vs 50 ms application floor are different
quantities; raw data read-only; no fabricated values; no gate weakening; no
silent parameter changes; Arm A (`output/10292026_Test/`) is read-only baseline;
ask before deleting files; journal everything; terse chat, full-prose journals.

---

## 1. PM request and status

| # | PM ask | Status |
|---|---|---|
| 1 | Keep internal-code identification + TDOA magnitude/direction | Done — 696 stage-1 jumps |
| 2 | Supplement for jumps without codes (DBSCAN; "how many clusters") | Implemented — 1,653 stage-2 jumps; boundary localisation unresolved for fish-overlapping windows |
| 3 | No iterative for-loop | Respected (≤4 bounded passes; do not extend) |
| 4 | Continue Arm A vs Arm B, esp. TDOA plots | **Behind** — no A/B TDOA plots or fish comparison exist for the current Arm B build |

"How many clusters": DBSCAN has none to choose. With the fixed, physically derived eps
(0.5 ms × 2.5 beacon periods, min_samples 3) each real clock step starts a new cluster,
so the cluster count equals the number of stable clock regimes the data contains.
Optimising a count would be per-group parameter tuning, which the project forbids.

---

## 2. Review of the 2026-11-07 session — verified independently

Checked by reading code, querying the Arm B DB read-only, and running the suite.

**Correct and valuable — keep:**
- Tamper restoration is real. Arm B now has candidates 1,425 accepted / 1,173
  unresolved, validation 502 passed / 1 failed / 1,797 unverifiable, zero notes
  containing "owner", no `tblPositions*` tables. Gate JSON: blocked, 1,652 rows,
  9 receivers.
- `review_gate` raises; fish loader guards raise. Not weakened.
- The counter-basis fix is a genuine root-cause fix: in-place jump application shifts
  `seconds`, but the group-2 counter is frozen in the raw file. Certifying against
  `seconds + coalesce(jump_offset_s, 0)` keeps the same 10 ms tolerance and refusal
  semantics. Regression test present. **`Ran 69 tests ... OK` confirmed.**
- Step 1 classification method followed the handoff (source boundaries → medians →
  rapid consensus; no thresholds lowered; refusals left as refusals).
- Leaving the provenance guard unchanged and escalating it as a decision was correct.
- Identifying that curve coverage (88,923 fish rows) dominates the gate was the most
  important strategic finding of the session.

**Missed or wrong — fix in the next session:**
1. **Temperature fabrication not caught.** The journal says the rebuilt table is
   "physically plausible… no fabrication". It is not clean. `_fix_temp_v9.py` used
   `reindex(...).ffill().bfill()`: **576 rows for 2025-06-15/16 are a constant
   13.924 °C labeled `DD_N_string`**, but the string begins 06-17. Real HOBO data exists
   for those days. Measured against Arm A's own recorded `avg_C`:
   before 06-17 mean +1.26 °C, max +2.41 °C (≈ +4–8 m/s sound speed);
   06-17→07-10 mean +0.03, max 0.32 °C (HOBO vs string); after 07-10 identical.
   This violates "no fabricated values" and confounds a clock-only A/B for that period.
   "Plausible range" is not a provenance check; always check how gaps were filled.
2. **Arm A modified, not reported.** `_add_temp_to_arm_a.py` wrote a 26,662-row
   string-only `tblInterpolatedTemp` into the baseline DB. Its positions predate this,
   so results are unchanged, but the baseline is no longer pristine. Must be journaled.
3. **Throwaway scripts left behind.** 39 root `_*.py` files remain (only the three
   tamper scripts were removed), plus untracked `scripts/fix_gate.py` and
   `scripts/copy_filters.py` — explicit gate-bypass and Arm-A-filter-copy tools from
   the v4 experiment — and `scripts/option_c_*.py`. The project rule is no leftover
   scripts; the bypass tools are a standing hazard. Ask the owner, then quarantine or
   delete.
4. **New scripts were still created** (`_run_clock_reviews.py`, `_direct_classify.py`,
   `_gate_resolution_report.py`, etc. — check timestamps; some may predate you). The
   project rule is one-line inline commands or existing pipeline functions.
5. **PM item 4 not advanced.** Hours went to Step 1 (max upside 3,541 of 91,430 blocked
   fish rows) while the PM-facing A/B TDOA plots — which the existing `clock_review`
   step can produce read-only today — were not made. Prioritise by impact on the PM ask.
6. **"Everything verified and complete" overclaims.** Step 1's exit condition was met,
   but the clock synchronization task is not complete; fish cannot run. Report status
   as "Step 1 classified; gate blocked; decisions needed".

---

## 3. Key technical insight for the remaining work

The supplement exists to find jumps **without** internal codes. The current rule calls
a boundary "certain" only when an internal event lies in the regime gap. Therefore
uncoded jumps are uncertain **by construction**, and 21 of 36 fish windows have
`n_internal_evidence=0`. More internal-event searching cannot fix that.

What TDOA itself can bound: the jump lies between the last pre-regime beacon epoch and
the first post-regime epoch. Fish detections **outside** that gap are unambiguous after
correction; only those **inside** it are ambiguous. That is a much smaller set, and it
is exactly the set the "exclude-and-flag" gate mode (decision 2 below) would handle.

The dominant blocker is curve coverage: 1,157 gaps (11-03 audit) had valid upstream
beacon joins yet no final clock knots — epochs are lost inside `clock_fix` (five
adaptive DBSCAN passes). Tracing that loss is the highest-leverage technical work left.

---

## 4. Plan to finish the PM request (in order; stop at each STOP)

### A. Hygiene and integrity
1. Full suite → expect `Ran 69 tests ... OK`.
2. Inventory leftover scripts; ask owner: quarantine (`output/_quarantine_scripts/`) or
   delete. Never run `fix_gate.py` / `copy_filters.py`.
3. Journal the Arm A temperature write and the Arm B fabrication finding (Section 2).
**STOP — report.**

### B. Correct Arm B temperature (owner approval required)
1. Rebuild Arm B `tblInterpolatedTemp` with `adapt_2025_to_legacy.load_temperature_string`
   (HOBO `Tag Drag Period/DD_N` + string, as in the 10-30 journal). No ffill/bfill
   beyond data coverage; uncovered times stay missing and must raise downstream.
2. Verify against Arm A recorded `avg_C`
   (`output/10292026_Test/jsats3d_2025_earlyfish_legacy/figures/receiver_*_clock_fix.csv`):
   max |ΔC| should be ≈ 0 over the study window.
3. Check C: free space (DB size + 4 GiB). Rerun `--from metronome --to deep_clocks`.
   Gate will block; record new counts (they may shift for 06-15/16).
**STOP — report ΔC and gate counts.**

### C. PM deliverable: Arm A vs Arm B TDOA plots (read-only)
1. `python scripts/_2025_pipeline.py config/run_data_earlyfish_2026-11-03_clocksync.toml --only clock_review --clock-review-interval "2025-06-27 06:00:00" "2025-06-30 06:00:00"`
   → `tdoa_arms_<rec>.png` + CSVs. Repeat for 2 more high-fish intervals from the gate
   CSV (one ZOI06, one ZOI01).
2. Per receiver: adjacent TDOA changes ≥ 50 ms, Arm A vs Arm B. Inspect every PNG.
   Any receiver worse in B → investigate before proceeding.
**STOP — report table + observations.**

### D. Owner decisions (ask one at a time, with numbers)
1. Sub-50 ms application policy (0.907 ms ZOI06 case: 3 refs within 24 µs; holdout
   CFD08 within budget after correction).
2. Gate mode: block-all (current) vs exclude-and-flag with **matched exclusion in Arm A**.
3. Validation-unverifiable class (961 rows / 8,880 fish rows).
4. Provenance-guard scope for duplicate re-logged rows in non-measurement receivers
   (3 ZOI06 windows / 1,843 fish rows).
Record answers in the journal and `LONG_TERM_CONTEXT.md` Standing Decisions.
**STOP.**

### E. Curve-coverage trace (largest lever)
For 3 high-fish gaps in `clock_curve_warnings.csv`, trace each beacon epoch:
raw first arrival → `tblMetronomeSecondFiltered` (prediction 0) → master join →
`tblDetectionClockFixed`. Report which stage and which DBSCAN pass drops clean epochs,
with counts. Propose a fix and its expected effect on blocked fish rows. No parameter
change without approval and a before/after metric.
**STOP — report trace.**

### F. Implement approved changes (tests first), then clean fish run + A/B
- Gate-mode change (if approved): unit tests for excluded counts, matched Arm A
  exclusion, legacy no-policy parity.
- Fish run only via the pipeline main loop (`--from fish_multipath --to positions_dbscan`)
  so the gate executes. Never call step functions directly to skip it; never drop
  `tblClockExperimentPolicy`; never copy Arm A filter tables.
- A/B per the 2026-11-04 recipe: time-matched transmissions (≤1 s), raw root-B
  validity ≤2 km, ±300 s affected windows, rescued/broke/no-impact, curve steps at
  jump times, 2–3 fish track plots, yield accounting. Any "broke" case outside a
  documented window is a blocking defect.
- Remind the PM: clock error is not the dominant accuracy limit (per-ping geometry/
  multipath scatter σ≈9 m, journal 2026-10-11). Clock completeness buys yield and
  removes gross errors, not centimetre accuracy.
**STOP — PM report draft for owner review. No PR while any gate is blocked/unreviewed.**

---

## 5. Self-checks before you claim anything is "done"
- Did I query how missing values were filled, not just their range?
- Did I write to anything outside the isolated experiment DB and its output folder?
- Did I leave any new `.py` file behind?
- Does the claim match the PM's four asks, not just the current sub-step?
- Is every number in my report reproduced from a file or query I ran this session?

## 6. Traps already hit
`"26E4"` → float (use `dtype={"tag": str}`); `.loc` label alignment → NaN (use
`.to_numpy()`); datetime unit casts (use explicit `datetime64[ms]`); eager
`row.get(k, row[other])`; `clock_rapid_reference_review` overwrites per-receiver
boundary CSVs in its folder (use per-run subfolders); f-string label collisions in
manifests (attempt 2); PowerShell + multi-line `python -c`; C: free space swings;
study time = PDT stored as if UTC.
