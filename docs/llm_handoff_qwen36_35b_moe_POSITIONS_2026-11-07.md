# Positions Handoff for Qwen 3.6 35B MoE — Diagnostic Arm B fish positions + figures per tag (2026-11-07)

**Read first:** `docs/llm_handoff_qwen36_35b_moe_RECOVERY_2026-11-07.md` (rules, traps, verified facts). This file
**replaces its Task 7** and overrides its Task 6 "message for Kevin" step (no longer needed to produce diagnostic plots).
If any file disagrees with this one, this one wins. Work **one numbered step at a time**; every step ends with **STOP**
(report numbers, wait). Chat replies short; journal entries full prose.

---

## 0. What the user wants and what is authorized

The project lead (the user) wants **fish positions and figures for each tag, soon**, done the **safest** way, **without
damaging existing positions**. Authorized on 2026-11-07 by the project lead's explicit instruction:

1. **Plot the existing Arm A positions now** (read-only; they are the validated baseline).
2. **Produce Arm B positions as a DIAGNOSTIC on an isolated throwaway copy**, with figures for each of the four tags
   (26E4, 9F8C, 1D5A, B5ED), clearly labeled **"DIAGNOSTIC — clock gate NOT cleared"**.

**Why this is allowed, and what it is NOT.** The clock gate protects what we *call validated and deliver*. A labeled
diagnostic on a copy does not pass the gate and does not change production. These Arm B positions are **not** production
results, **not** PM deliverables, and must **never** be described as "validated" or "gate-passed". Signing off the
gate (Gate 2 in `System_Prompt.txt`) belongs to the project owner Kevin Nebiolo; nothing here asks you to bypass it in
production. Do not ask the user the old Kevin questions as a precondition for this work.

**What the diagnostic really is:** Arm B's fish steps on the copy use the *legacy* clock-correction path (linear
interpolation across knots, exactly as Arm A did). That path silently bridges the >120 s knot gaps that the strict gate
refuses. So these positions are **comparable to Arm A** but carry the same exposure Arm A always had. This is the same
recipe used and documented on 2026-11-04 (`journals/2026-11-04_ab_fish_validation_clocksync.md`).

## 1. Hard rules for this task (breaking any is a defect)

1. **Never write to** `output/10292026_Test/` (Arm A) or `output/11062026_clocksync/` (production Arm B) — databases,
   gate files, figures, anything. Open them only with `file:...?mode=ro` and `PRAGMA query_only=ON`.
2. All new output goes under **`output/11072026_diag_armB/`** (folder name keeps the word DIAGNOSTIC in its marker file).
3. **Never** drop `tblClockExperimentPolicy`, flip statuses, edit `review_gate`, `_clock_supported_interpolation`, or any guard
   **in the production DB or in code**. The only table drop allowed is `tblClockExperimentPolicy` **inside the diagnostic copy**.
4. **Never** copy Arm A filter/position tables into the copy. Never call step functions directly; use the pipeline main loop.
5. No new `.py` files; use one-off snippets. No thresholds or parameters changed. The owner commits; you do not.
6. If any step raises an error you did not expect, **STOP and report**; do not patch code to get past it.
7. Disk: need ≥ 20 GiB free before copying (the DB is ~6 GB). Check first; it swings.
8. Journal each step in `.ai_journal/session/2026-11-07_diag_positions.md` (new file): commands, numbers, decisions, parameter
   changes (none), blockers, next steps. **The previous Qwen session wrote no journal entry — that is a defect. Do not repeat it.**

## 2. Corrections to your own previous reporting (do not repeat these)

- Task 4 was reported as "all 19 receivers improved". **False.** There are 18 listener receivers per interval and 53 of 54
  improved; ZOI04 went 67 → 68 (not a corrected receiver). Always state counts exactly as computed.
- The gate files you cited were stale (19:28, before your 22:00 rerun). Always regenerate or check timestamps before quoting.
- `pipeline_status.json` in the production folder still shows `fish_multipath: running` from 19:27 — stale; ignore it. The diagnostic
  copy gets its own fresh status file.
- Report **only numbers you reproduced in this session from a file or query.** Say "not verified" otherwise.
- The Kevin draft (`output/11072026_recovery/kevin_message_draft.md`) was corrected by another assistant; leave it as is.

## 3. Verified facts you can rely on (checked 2026-11-07 night)

- Arm A (`output/10292026_Test/`): complete. `tblPositions_Deng` root-B "solution found": **1D5A 51,859; 26E4 19,897; 9F8C 1,654;
  B5ED 37,263**. Figures folder `output/10292026_Test/figures/` has 25 files: per tag `<TAG>_plan_time.png`, `_plan_depth.png`,
  `_3d.png`, `_track.gif`, `_track3d.gif`, `_positions.csv`, plus `fish_qc.csv`.
- Production Arm B (`output/11062026_clocksync/jsats3d_2025_earlyfish.db`, last written 22:16): temperature table fixed (30,817 rows,
  matches Arm A), clock chain rerun through `deep_clocks`, **no fish filter or position tables yet**, `tblClockExperimentPolicy`
  present, gate blocked (1,653 rows). Free disk was ~41 GiB.
- `fish_figures_2025.py` reads a DB read-only and writes to any `--out` folder; `--tags` defaults to all study tags;
  `--depth-floor-m` defaults come from the run file `[figures] depth_floor_m` (**-16.0** in both run files).
- `build_context` (in `scripts/_2025_pipeline.py`) **raises** if a DB has the segment-safe policy table but the run file says
  `clock_segment_safe=false`. It **re-creates** the policy table on every run if the run file says `clock_segment_safe=true`.
  Therefore the diagnostic copy needs BOTH: policy table dropped AND a run file with `clock_segment_safe=false`.
- The 2026-11-04 diagnostic took ~0.8 min (fish_multipath) + 1.6 (positions) + 0.1 (export) + 0.1 (positions_dbscan).

---

## 4. Steps

### Step 1 — Baseline plots from Arm A (zero risk; do this FIRST so the user can plot immediately)
1. Record `Get-Item` size + LastWriteTime of the Arm A DB, the production Arm B DB, and `output/10292026_Test/figures/*` (needed in Step 7).
2. Read `output/10292026_Test/figures/fish_qc.csv`. Make a per-tag table: n_solutions, n_above_surface, n_below_floor, depth_floor_m,
   n_consensus, n_kalman, first_obs, last_obs, span_h, max_speed_m_s, z_min_m, z_max_m, n_jumps_gt_3ms. Report it.
3. Open **every** PNG for each tag (plan_time, plan_depth, 3d) and write 2-3 plain sentences per tag: where the fish is, what the
   track looks like, anything odd (jumps, depth outside 0 to -16 m). The GIFs: list names and sizes only.
4. Give the user the folder path `output/10292026_Test/figures/` as the "validated baseline plots, available now".
**STOP. Report the table, the per-tag descriptions, and the path.**

### Step 2 — Preconditions for the diagnostic copy
1. `conda run -n jsat_3d python -m unittest discover -s tests` → expect `Ran 69 tests ... OK`.
2. Confirm no python pipeline process is running; free disk ≥ 20 GiB.
3. Confirm production Arm B still has the policy row and gate still blocked (use Appendix D of the RECOVERY doc, read-only).
**STOP if any check fails.**

### Step 3 — Create the isolated diagnostic copy
1. `New-Item -ItemType Directory output\11072026_diag_armB`
2. `Copy-Item output\11062026_clocksync\jsats3d_2025_earlyfish.db output\11072026_diag_armB\jsats3d_2025_armB_diag.db`
   (do **not** copy the `.compact.json`, the backup `pre_tempfix.db`, or anything else). A distinct file stem keeps work folders,
   status files, and `output/2025_review/<stem>` from colliding with production.
3. On the **copy only** (normal connection to the copy path): `DROP TABLE tblClockExperimentPolicy`. Verify by query that the copy has
   no such table **and that production Arm B still has its 1 policy row**.
4. Copy the run file: `Copy-Item config\run_data_earlyfish_2026-11-03_clocksync.toml output\11072026_diag_armB\run_diag.toml`, then edit
   only these lines in the copy: first line comment `# DIAGNOSTIC copy of Arm B - clock gate NOT cleared - not for delivery`;
   `output_db = 'output\11072026_diag_armB\jsats3d_2025_armB_diag.db'`; `[legacy] apply_clock_jumps = false`;
   `clock_segment_safe = false`; `clock_tdoa_supplement = false`. Change nothing else (study window, tags, receivers, DBSCAN,
   `depth_floor_m = -16.0` stay).
5. Write a one-line marker file `output\11072026_diag_armB\DIAGNOSTIC_NOT_GATE_CLEARED.txt`:
   `Arm B diagnostic copy. Legacy interpolation bridges clock-knot gaps >120 s (same as Arm A). Clock gate NOT cleared. Not validated, not for delivery.`
**STOP. Report copy size, that the policy table is gone from the copy only, and the exact TOML diff (`Compare-Object`).**

### Step 4 — Run fish steps through the pipeline main loop (copy only)
1. `python scripts/_2025_pipeline.py output/11072026_diag_armB/run_diag.toml --db output/11072026_diag_armB/jsats3d_2025_armB_diag.db --list`
   and read the step order. If drag-plot steps sit between `fish_multipath` and `positions_dbscan`, add `--no-drag-plots`.
2. Run: `python scripts/_2025_pipeline.py output/11072026_diag_armB/run_diag.toml --db output/11072026_diag_armB/jsats3d_2025_armB_diag.db --from fish_multipath --to positions_dbscan`
   (let the user watch it). Expected: all steps `ok`; no "Clock review required" errors (the gate does not run because the policy table is gone).
3. Query the copy (read-only): per tag, count of `tblPositions_Deng` rows with `comment='solution found'` for solutions A and B.
   Report next to Arm A's counts (1D5A 51,859; 26E4 19,897; 9F8C 1,654; B5ED 37,263 root B).
**STOP. Report step timings, counts per tag, and any warning text.**

### Step 5 — Figures for each tag (diagnostic)
1. First read Arm A's `fish_qc.csv` column `depth_floor_m`; use the **same** value (expected -16.0).
2. `python scripts/fish_figures_2025.py --db output/11072026_diag_armB/jsats3d_2025_armB_diag.db --out output/11072026_diag_armB/figures --run-file output/11072026_diag_armB/run_diag.toml --depth-floor-m -16.0`
3. Confirm 25 files: for 26E4, 9F8C, 1D5A, B5ED each `_plan_time.png`, `_plan_depth.png`, `_3d.png`, `_track.gif`, `_track3d.gif`,
   `_positions.csv`, plus `fish_qc.csv`. Copy the marker file into the `figures` folder as well.
4. Open every PNG. Per tag, 2-3 plain sentences. Flag anything physically impossible (Z above 0 or below -16 m, speeds > 3 m/s sustained, jumps).
**STOP. Report the Arm B `fish_qc.csv` table in the same layout as Step 1.**

### Step 6 — Arm A vs diagnostic Arm B comparison (read-only; this is the "no degradation" evidence)
Per `journals/2026-11-04_ab_fish_validation_clocksync.md` recipe:
1. Per tag, side by side: n_solutions, n_consensus, n_kalman, span_h, max_speed, n_jumps_gt_3ms, z range, share above surface.
2. Match transmissions **by time** (nearest pair within 1 s), **never by transNo**. For matched final consensus positions report XY
   difference: median, p90, p99, max, and share < 1 m / < 10 m. Load position CSVs with `dtype={"tag": str}` (`26E4` parses as a float).
3. Raw root-B level (median per transmission, "valid" = within 2 km of array centroid (467.0, 113.3 m)): rescued / broke / no-impact.
   For every "broke" case state the receiver set, time, and whether it lies inside a window listed in
   `output/11062026_clocksync/clock_review/clock_review_gate.csv`. A "broke" case **outside** every documented window is a blocking defect: report it prominently.
4. **Exposure:** per tag, the share of that tag's detections (per receiver) lying in >120 s clock-knot gaps, for Arm A and Arm B
   (adapt Appendix A of the RECOVERY doc, grouping by tag). This tells the user how much of each track rests on bridged clock gaps.
5. Pick 2-3 fish examples around an applied jump (use `tblClockJumpEvents` in the copy, read-only) and describe Arm A vs Arm B tracks.
**STOP. Report the tables and examples; state plainly if Arm B is better, equal, or worse per tag.**

### Step 7 — Prove nothing else was touched, then journal
1. Re-check size + LastWriteTime of Arm A DB, production Arm B DB, and Arm A figures against Step 1 → must be **identical**. Production Arm B
   must still have the policy row, `tblClockJumpCandidates` 1,425/1,173, validation 1/502/1,797, and no `tblPositions*`/`tblDetectionFilter*` tables.
2. Full test suite again → `Ran 69 tests ... OK`. `git status --short` → no new files in `scripts/` or repo root; `scripts/` and `jsats3d/` unchanged.
3. Write `.ai_journal/session/2026-11-07_diag_positions.md` (full prose, all fields). State in its first paragraph that the Arm B positions are
   diagnostic and not gate-cleared.
**STOP. Final report to the user (short): where each tag's figures are (Arm A baseline and Arm B diagnostic), the comparison table, and caveats.**

---

## 5. How to talk about the results (the user will show these to others)

- Arm A figures: "original pipeline, validated by use; baseline."
- Arm B figures: "diagnostic; same legacy clock interpolation as Arm A; strict clock gate not cleared; not for delivery."
- Never say "no degradation proven." Say what was measured: yield, differences between matched positions, rescued/broke counts, exposure.
- Remember: 1 ms of clock error ≈ 1 m of position error, but clock error is not the dominant accuracy limit here — per-ping geometry/multipath
  scatter (σ ≈ 9 m, journal 2026-10-11) is. Clock work buys yield and removes gross errors (e.g. 548 km → 20 m), not centimetre accuracy.
- Known weak axis: Z (flat array; fish solutions show a ~2-4 m deep bias versus known tag depths, journal 2026-10-25).

## 6. If something goes wrong

- `build_context` raises "segment-safe clock policy": the policy table is still in the copy, or the run file still has `clock_segment_safe=true`. Fix the **copy / copied TOML only**.
- `database or disk is full`: stop, free space (ask before deleting anything), retry once.
- Fish step raises "Clock review required": you are running against the production DB or the policy table is still present. **Stop; do not edit code.**
- A step is skipped as "already done": read its `*_done()` check in `scripts/_2025_pipeline.py`; do not force it by editing tables in production.
- Anything unclear: STOP and report with the exact error text.
