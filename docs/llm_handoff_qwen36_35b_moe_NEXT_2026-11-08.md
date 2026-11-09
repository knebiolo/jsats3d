# Next-Steps Plan for Qwen 3.6 35B MoE — improve Arm B, then refresh the PM plots (2026-11-08)

**Read first (in this order):** `docs/llm_handoff_qwen36_35b_moe_RECOVERY_2026-11-07.md` (rules, traps),
`docs/llm_handoff_qwen36_35b_moe_POSITIONS_2026-11-07.md` (diagnostic recipe), then this file.
**If files disagree, this one wins.** Work **one task at a time**; each ends with **STOP** (report, wait).

## 0. State at the start (verified 2026-11-08; re-verify, do not assume)

- PM request status: (1) internal codes + TDOA **done**; (2) DBSCAN supplement **built but not accurate enough** (stage 2: 79.7% of measurable
  events agree with raw TDOA within 50 ms vs 96.6% for stage 1; 55.6% of stage-2 events have an uncertain boundary); (3) no for-loop **respected**;
  (4) Arm A vs Arm B comparison **partly done**: TDOA plots improved (3 days: 1,758 -> 1,544 jumps >= 50 ms, 12% fewer; 53 of 54 receiver-intervals better),
  diagnostic fish positions comparable for 1D5A/26E4, weaker for B5ED, no truth-based test yet.
- Production Arm B `output/11062026_clocksync/jsats3d_2025_earlyfish.db`: gate **blocked** (1,653 rows). Do not change it.
- Diagnostic copy: `output/11072026_diag_armB/` (not gate-cleared; positions identical on rerun).
- Provisional decisions (LONG_TERM_CONTEXT.md, 2026-11-07) still stand: no change to what production applies; sub-50 ms steps flagged only; gate block-all;
  anchor ZOI03; provenance guard strict. **What-ifs on isolated copies are allowed and requested below; changing production is not.**
- First PM plot set already exists: `output/11082026_pm_plots/` (1_timing_jumps_by_receiver, 2_example_ZOI02, 3_fish_detections_on_clock_gaps, 4_positions_per_fish, extra_ZOI01_less_improved).

## 1. Rules for this plan (all of the recovery rules apply; these are the ones that failed before)

0. **Working directory and file locations (a previous run broke this).** The repo is `C:\Dev\jsat3d\jsats3d`, **not** `C:\Dev\jsat3d`. Your first command in every session must be
   `Set-Location C:\Dev\jsat3d\jsats3d`, and `git rev-parse --show-toplevel` must print `C:/Dev/jsat3d/jsats3d` (if it says "not a git repository", you are in the wrong folder and every relative path is wrong).
   Use **absolute paths** for all writes. The journal folder is `C:\Dev\jsat3d\jsats3d\.ai_journal\session\`. Earlier journals landed in `C:\Dev\jsat3d\.ai_journal\session\`
   (one level too high): `2026-11-07_diag_positions.md`, `2026-11-07_hygiene_tdoad.md`, `2026-11-07_jump_audit.md`, `2026-11-08_next_steps.md`. In Task 0, **copy** (do not move or delete) those four
   into the repo journal folder and continue the 11-08 journal there.
1. **Journal proof.** After every task append to `C:\Dev\jsat3d\jsats3d\.ai_journal\session\2026-11-08_next_steps.md` (full prose: commands, numbers, decisions,
   parameter changes (none unless approved), blockers). Then run `Get-Item` on **that file's full path** (not the folder) and **paste its LastWriteTime and its last 3 lines in your report.**
   A report without this proof is incomplete.
2. **Ask before deleting or overwriting anything** in `output/` (the last run overwrote a folder without asking). Use new, dated folder names instead.
3. Never write to Arm A (`output/10292026_Test/`) or production Arm B. Open them `mode=ro` + `PRAGMA query_only=ON`. Never bypass or edit the gate.
4. No new `.py` files. One-off snippets only. No threshold/eps/window/anchor changes. The owner commits.
5. Time basis is study wall-clock (PDT stored as if UTC). **Never label times "UTC".**
6. Report only numbers you reproduced this session. State counts exactly (18 receivers, not 19; B5ED matched pairs 1,344 under one-to-one matching).
7. Disk: check free space before any copy (about 34 GiB at last check; a DB copy is about 6 GB). Stop if below 15 GiB.

## 2. Tasks

### Task 0 — Setup (read-only)
Full test suite (`Ran 69 tests ... OK`), `git status --short` (no new files in `scripts/` or repo root), disk space, confirm production gate inputs
(1,425 accepted / 1,173 unresolved; validation 1 failed / 502 passed / 1,797 unverifiable; policy row present; no `tblPositions*`). Create the journal file.
**STOP. Report with journal proof.**

### Task 1 — What-if: does stage 2 help or hurt? (stage-1-only arm "B1")
Question for the PM: *are the supplementary (DBSCAN) corrections an improvement over internal-code corrections alone?*
Measure three arms side by side: **A** (original), **B1** (stage 1 only), **B** (stage 1 + 2, existing).

**1a. Design note first (no execution).** Read and quote the exact behavior of: `clock_jump_pre.revert_tdoa_supplement(db, source_db)` (takes a **Path** for
`source_db`, attaches it read-only, restores the post-stage-1 state by original rowid, leaves stage-1 ledger rows), `_invalidate_clock_outputs(c)`,
`clock_jumps_done`, `build_context` (needs `<db>.compact.json` with `status: complete` and `target` equal to the DB path; it re-creates the policy table when
`clock_segment_safe=true`), and the step order. Then write the exact command list into the journal. Constraints your design must meet:
- Work on a **copy**: `output/11082026_whatif_B1/jsats3d_2025_B1.db` (copy of production Arm B). Copy the production `.compact.json` and edit only `target` to the new path.
  Copy the run file to `output/11082026_whatif_B1/run_B1.toml` and change only: `output_db`, `clock_tdoa_supplement = false`. Keep `clock_segment_safe = true` and
  `apply_clock_jumps = true` so the **same clock-fix algorithm** as production Arm B is used (otherwise the arms are not comparable).
- Revert stage 2 on the copy by calling the revert function inline with Arm A's path as `source_db` (Arm A is only read). Do **not** delete the `__ats_clock_jumps__`
  progress row (re-running stage 1 on an already-shifted DB raises by design). Drop derived tables as the pipeline does, rerun `--from metronome --to deep_clocks` on the copy.
- Expected after revert: `tblClockJumpEvents` has 696 stage-1 rows and no stage-2 rows; sum of raw `seconds` + `jump_offset_s` unchanged versus the source rows.
- **Details your design must also contain** (a first design note missed them):
  1. A runnable way to build the context `c` for `_invalidate_clock_outputs`: `import sys; sys.path.insert(0, 'scripts'); from types import SimpleNamespace; import _2025_pipeline as p;
     c = p.build_context(SimpleNamespace(run_file=r'C:\Dev\jsat3d\jsats3d\output\11082026_whatif_B1\run_B1.toml', db=None))` (it re-creates the policy row on the copy, which is expected). Read `build_context`
     first and confirm which `args` attributes it uses.
  2. Pass `pathlib.Path` objects to `revert_tdoa_supplement`, run it **once**, and check its return value: `removed_stage2_events` must be 1,653.
  3. **Size guard.** In the main loop, every step on a segment-safe DB calls `check_capacity(db, 4 GiB free, 6 GiB max)`. The production DB is 5,976,518,656 bytes; the limit is 6,442,450,944. If a step
     raises "Compact database exceeded its size budget", **STOP and report; never raise the limit.**
  4. After the rerun, prove it recomputed: knots per receiver differ from production (e.g. ZOI01 production has 46,816), stage-1 ledger = 696 rows, `tblClockJumpValidation` rewritten.
  5. Afterwards re-check size and LastWriteTime of Arm A (`2026-11-06 18:34:13`, 18,631,671,808 bytes) and production Arm B (`2026-11-07 22:16:26`, 5,976,518,656 bytes): both must be unchanged.
**STOP. Show the design note; wait for the user's OK.**

**1b. Execute** exactly the approved design. Verify before measuring: ledger counts, temperature table 30,817 rows, clock-fixed rows present for all 17 receivers.

**1c. Measure (read-only), report as one table per metric, A vs B1 vs B:**
1. TDOA: adjacent steps >= 50 ms per receiver for the 3 existing intervals (`clock_review` outputs for A and B exist; compute B1 from the copy the same way: first-arrival
   times minus nearest ZOI08 `7DB7` first arrival, in **stored (corrected) time**, 5-epoch rule not needed, count `abs(diff) >= 0.05`). Also the 1st-99th percentile level range per receiver.
2. Ledger audit (Appendix B of the RECOVERY doc) on B1 (expected stage 1 only: 672/696 agree within 50 ms).
3. Clock-knot gap exposure per tag (fish detections in >120 s gaps or outside the knot span; 8 non-master solving receivers): A 26E4 46.7 / 9F8C 66.3 / 1D5A 74.1 / B5ED 40.3 %;
   B 39.5 / 43.8 / 72.5 / 34.6 %.
4. Diagnostic fish positions on a copy of B1 using the POSITIONS recipe (drop policy table on the **new** copy only, `clock_segment_safe=false`, `apply_clock_jumps=false`,
   `clock_tdoa_supplement=false`, depth floor -16.0, label DIAGNOSTIC): yield per tag, time-matched XY agreement vs A, raw rescued/broke (and for every "broke": inside an unresolved window?).
**STOP. Report the tables and a plain verdict:** stage 2 helps / is neutral / hurts, per metric. Do not change production.

### Task 2 — What-if: take ZOI06 out (read-only, post hoc)
ZOI06 hears the master beacon rarely (about 16k bursts vs 50-70k for others; 2,992 knots; ~60 h gap), is fully exposed in both arms, and is in most "broke" cases.
Using only existing tables (`tblPositions_Deng`, columns `r0..r3`, `solution`, `comment`) in Arm A and in the diagnostic copy, **drop every combination that contains ZOI06**, then re-run
the same final-product chain as `scripts/fish_figures_2025.py` (import and call `legacy_receiver_set_consensus`, `smooth_track`, `kalman_filter_track` from `scripts/`, depth floor -16.0, same
parameters it uses). Write only to `output/11082026_zoi06_excluded/`. Report per tag, A vs B (and B1 if Task 1 is done): positions, time-matched XY agreement, raw rescued/broke,
and whether the B5ED 06-26 09:23:45 "broke" case disappears. Also report how many combinations/transmissions were lost by removing ZOI06 (yield cost).
**STOP. Report the table.** Whether to drop ZOI06 from the receiver set is the owner's decision; only present numbers.

### Task 3 — Scoping only: truth-based accuracy test (tag drag / static holds)
Goal: position error against known truth for Arm A vs Arm B, the project's real acceptance test (System_Prompt section 11).
**Do not build or run anything in this task.** Inventory (read-only): which databases/outputs exist for the 2025-06-10..06-12 drag window (check `output/`,
`C:\Dev\jsat3d\output_archive_*`, `config/run_data_tagdrag*.toml`, `data/drag_GPS_2025.csv`, `scripts/drag_test_figures_2025.py`, `scripts/ent_analysis_2025.py`); the truth sources and their limits
(GPS minute-truncated; towed-tag layback; static holds FBY-S-1..10); whether the raw files for that window are reachable (K: is slow; read only a few targeted files); how many internal-code
events and jumps fall inside the window (count from raw `Internal` evidence or an existing ledger); and a cost estimate (disk, hours) for an Arm A vs Arm B build for that window.
Write a one-page design with the exact commands you would run and the metrics (median XY error per line, RMSE on static holds, yield).
**STOP. Report the inventory and design; wait for the user's OK.**

### Task 4 — Refresh the PM plots (only with measured data)
Make a new folder `output/11082026_pm_plots/v2/` (do not overwrite v1). Same style as v1: two colours (grey = Original, blue = New), almost no text, no gridlines, one short title,
legend only. Rules that keep it honest:
- Always show **every** tag/receiver in the chart; never drop one that got worse.
- The example time-series receiver is chosen by a **fixed rule** (largest relative drop in jumps among solving receivers with >= 50 jumps in the original); also save the least-improved
  big receiver as an extra file. Plot TDOA relative to the receiver's own median in each arm.
- If Task 1 is done, add a third colour (light blue) = "Internal codes only" to plots 1 and 3. If Task 2 is done, add one plot "Same comparison without ZOI06".
- Plots show clock-level and yield metrics. **They are not position-accuracy results**; say so in the message to the user, not on the plots.
Use the recipe in Appendix A. View every PNG before reporting; check for overlapping legends.
**STOP. Report file paths and a 3-line plain summary per plot, including anything that looks worse.**

### Task 5 — Housekeeping and hand-off to the owner
1. `pipeline_status.json` in production still says `fish_multipath: running` (stale). Do not edit it; list it in the journal.
2. Update `output/11072026_recovery/kevin_message_draft.md` with the results of Tasks 1-3 (numbers only, plain language, the open question about the 11th "broke" case).
3. Propose a commit plan (the owner commits): group the uncommitted changes into logical commits (e.g. clock_jump_pre stage-1/2 + tests; `_2025_pipeline` review/gate; `jsats3d.py` strict loader + `temp_interpolator`
   revert; `run_data.py` compact copy; docs/handoffs; journals; run files). List files per group. Do not run `git commit`.
4. Add to `LONG_TERM_CONTEXT.md` Standing Decisions only what Tasks 1-3 actually established, marked provisional.
**STOP. Final report with journal proof.**

## 3. Interpreting results (what to say, what not to say)
- Say "plausibly not degrading" only where TDOA and fish metrics agree. Never say "proven" or "validated".
- If B1 is as good as or better than B on TDOA, fish agreement and exposure, state that stage 2 adds risk without clear benefit and let the owner decide; do not remove stage 2 yourself.
- If stage 2 is clearly better on TDOA but worse on mismatches at unstable plateaus, say so; a selective variant (skip stage-2 offsets where the plateau spread is > 0.5 s or the boundary is uncertain)
  would need code and tests and the owner's approval; **do not implement it in this plan.**
- 1 ms of timing error is about 1 m of position error; clock error is not the dominant accuracy limit here (per-ping geometry/multipath scatter about 9 m).

## Appendix A — plot recipe (adapt; run as a snippet)
```python
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold'})
GRAY, BLUE = '#9a9a9a', '#1f6fb5'
base = Path('output/11062026_clocksync/clock_review/interval_1751004000_1751263200')   # 06-27 06:00 to 06-30 06:00
x = pd.read_csv(base / 'multi_beacon_relative_timing.csv', usecols=['arm', 'host', 'Rec_ID', 't', 'tdoa_s', 'adjacent_step_s'])
x = x[x.host == 'ZOI08']                                  # arms: 'unshifted' = Original, 'tdoa_supplement' = New
cnt = x.assign(big=x.adjacent_step_s.abs() >= 0.05).groupby(['Rec_ID', 'arm']).big.sum().unstack('arm')
rec = ['ZOI01','ZOI02','ZOI03','ZOI05','ZOI06','ZOI07','ZOI09','ZOI10']; c = cnt.loc[rec]; i = np.arange(len(rec)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.2)); ax.bar(i - w/2, c.unshifted, w, color=GRAY, label='Original'); ax.bar(i + w/2, c.tdoa_supplement, w, color=BLUE, label='New')
ax.set_xticks(i); ax.set_xticklabels(rec, fontsize=10); ax.set_ylabel('Timing jumps'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('output/11082026_pm_plots/v2/1_timing_jumps_by_receiver.png', dpi=160)
```
Known v1 values to reproduce before changing anything: totals 1,758 -> 1,544; ZOI02 72 -> 46; ZOI01 323 -> 285; exposure 46.7/66.3/74.1/40.3 -> 39.5/43.8/72.5/34.6; final positions
1,398/76/3,465/2,473 -> 1,774/134/4,231/2,337 (26E4, 9F8C, 1D5A, B5ED). If you cannot reproduce them, stop and report.
