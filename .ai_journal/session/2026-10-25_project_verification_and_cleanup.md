# 2026-10-25: Independent Verification of Recent Work + Diagnostic Cleanup

**Session:** Independent test pass over the Oct 21-25 workstreams (accuracy tuning, ENT drag
figures, early-fish pipeline, fish figures, Z analysis). User pasted the prior sessions'
summaries and asked for: testing, result validation/comparison, journal, and diagnostic
script removal.
**Env:** `C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe`

## TL;DR
1. **Pipeline re-run (idempotency) PASSES** — `tagdrag_2025_pipeline.py` on the earlyfish DB
   skipped all 16 steps in seconds, exit 0. The 2026-10-24 `deep_beacons_done()` fix is
   verified end-to-end: step 5 is "already done, skipped" where it previously failed with
   `DatabaseError` on re-run.
2. **Every claimed number in the 10-24/10-25 journals reproduces from the DB** (row counts,
   per-tag/root solution counts, DBSCAN keeps, Z medians, above-water fractions). The
   "DBSCAN Filtered" Z p50s in the Z-accuracy journal use the Z ∈ [-30, 0] realistic-range
   gate (1D5A -4.74, 9F8C -12.81, B5ED -10.64 vs journal -4.7 / -12.8 / -10.6).
3. **Both figure generators re-run cleanly and reproducibly.**
   - `drag_test_figures_2025.py`: **BASELINE GATE PASS**, all 10 ENT medians diff 0.00 m vs
     the verified 2026-10-21 item12 baseline; 36 files regenerated; QC CSV identical.
   - `fish_figures_2025.py`: `fish_qc.csv` **byte-identical** after regeneration; 25 files.
4. **Bug found and fixed: README clobbering.** `drag_test_figures_2025.py::_write_readme`
   rewrote the shared `output/pm_figures_2026-10-22/README.md` without the Example-fish
   section (added by the 10-25 session), so any drag-only re-run silently deleted the fish
   documentation. Fixed: the README writer now emits the Example-fish section (chain
   description, live QC table from `fish_qc.csv`, file list, re-run line) whenever
   `example_fish/` exists. Verified by re-running the generator.
5. **Pre-existing test-suite break at HEAD (not from this session):**
   `tests/test_2025_contracts.py` imports `scripts/extract_dbscan_features.py`, which commit
   8ca7ba4 (HEAD, "Refine 2025 telemetry pipeline behavior") deleted → ImportError at load.
   `tests/test_resume.py::test_ats_fish_filters_keep_primary_and_secondary_separate` fails
   with a Windows temp-file lock (`WinError 32` in `rmtree` of a temp sqlite) even in
   isolated runs — environmental, not a code regression. 14/16 tests pass.
6. **Cleanup done:** 25 root-level ad-hoc diagnostic scripts removed (all gitignored
   scratch; `analyze_current_accuracy.py` was tracked — restore with `git checkout -- analyze_current_accuracy.py`
   if ever needed), `scripts/plot_fish_tracks.py` (gitignored standalone; the 10-19 journal
   already recorded it as removed), and this session's throwaway `scripts/_validate_earlyfish.py`.
   `scripts/` now holds exactly 10 persistent tools (7 pipeline + chinook/drag/fish figure
   generators).

## Tests run (this session)

### T1 — pipeline idempotency (the 10-24 fix)
- Pre-flight: evaluated every `done()` check in `tagdrag_2025_pipeline.STEPS` against the
  earlyfish DB via a throwaway script — 15/16 done; only `plot_drag_dbscan` (12b) returns
  False because its done-check looks for `item12_pipeline_summary.csv` in
  `output/2025_review/<stem>_dbscan/`, which is never written for non-drag DBs (no
  FC36/FFD3). Its `run()` is a no-op skip (`"ENT truth is available only for FC36 and FFD3"`),
  so the re-run costs nothing and writes no data. Bookkeeping wart only (see Open items).
- Real run: `python scripts/tagdrag_2025_pipeline.py config/run_data_earlyfish_2025.toml`
  → all 16 steps "already done, skipped" except the 12b no-op; exit 0; seconds wall time.
  **deep_beacons = skipped** (previously: re-run + `DatabaseError`). PASS.
- The run rewrote step statuses in the status JSON ("done (skipped, output exists)"); the
  original build record (10-24 16:10-17:01 run) was restored from backup afterwards.

### T2 — data integrity vs journal claims (earlyfish DB, 16 GB)
| Claim (journals) | Measured | |
|---|---|---|
| 609,374 position rows / 203,623 solutions | 609,374 / 203,623 | OK |
| B-found: 26E4 19,897 / 9F8C 1,654 / 1D5A 51,859 / B5ED 37,263 | identical | OK |
| A-found: 26E4 16,327 / 9F8C 1,285 / 1D5A 44,707 / B5ED 30,631 | identical | OK |
| DBSCAN-filtered 190,727 rows | 190,727 | OK |
| DBSCAN kept trans: 1D5A 6,866 / 26E4 1,802 / 9F8C 80 / B5ED 3,950 | identical | OK |
| status JSON notes vs DB (positions/export/dbscan) | all OK | OK |
| Z p50 realistic[-30,0]: 1D5A -3.1 / 9F8C -8.2 / B5ED -8.3 | -3.12 / -8.23 / -8.28 | OK |
| realistic %: 1D5A 48.6 / 9F8C 51.9 / B5ED 66.3 (26E4 71.9) | identical | OK |
| above-water %: 26E4 27.4 / 9F8C 45.9 / 1D5A 50.6 / B5ED 30.7 | identical | OK |
| < -30 m %: 26E4 0.7 / 9F8C 2.2 / 1D5A 0.8 / B5ED 3.0 | identical | OK |
| DBSCAN Z p50 (realistic-gated): 1D5A -4.7 / 9F8C -12.8 / B5ED -10.6 | -4.74 / -12.81 / -10.64 | OK |

F1 clock correction confirmed in code: `jsats3d/jsats3d.py` `multipath_data_object.__init__`
study-tag branch (piecewise `np.interp` of `tblDetectionClockFixed` residuals + ZOI09 0.946 s
add-back, table-existence check via `sqlite_master`, errors propagate).

### T3 — figure deliverables re-run and compared
- Drag: 36 files regenerated; baseline gate PASS (10/10 lines diff 0.00 m);
  `qc/drag_line_errors.csv` regenerated content identical to the pre-run file.
- Fish: 25 files regenerated; `fish_qc.csv` byte-identical (26E4 1.82 m/s max 0 jumps,
  9F8C 1.89/0, 1D5A 11.49/4, B5ED 4.86/9; 100% underwater tracks).

### T4 — tag-drag status vs older runs (user question)
Reproduced today, post-fix state (3D root-B consensus+Kalman vs drag GPS):
- Weighted median XY **3.95 m** vs **11.05 m** pre-F1 baseline (-64%); 2D **3.73 m** vs
  **9.55 m** (-61%). Root A scoring (what older pipeline runs used) was the worst variant:
  A-Kalman 28.5 m 3D. Per-line ENT-01..05 medians: 8.62/10.67, 5.63/6.37, 3.81/3.41,
  2.62/2.41, 3.13/2.95 (FC36/FFD3) — exactly the verified 10-21 baseline, reproduced today.
- Static holds: overall 3D median XY 7.64 m; FBY-S-1..7 mostly <11 m; FBY-S-8..10 degraded
  (16-74 m, undiagnosed, flagged in the README).

## Files changed (this session)
- `scripts/drag_test_figures_2025.py` — `_write_readme()`: new "Example fish" section +
  `### example_fish` file list, emitted when `example_fish/` exists (fixes README
  clobbering on drag-only re-runs). The only code change of the session.
- `output/pm_figures_2026-10-22/README.md` — regenerated by the fixed generator
  (drag QC + Example-fish section + full file list; 10 KB).
- `output/pm_figures_2026-10-22/{CFNSC_Approach_Drags,Forebay_Random_Drags,Forebay_StaticHolds,qc}`
  — 36 files regenerated (content-identical QC).
- `output/pm_figures_2026-10-22/example_fish/` — 25 files regenerated (QC byte-identical).
- `output/jsats3d_2025_earlyfish.pipeline_status.json` — temporarily rewritten by the T1
  re-run test; original build record restored (deep_beacons "ok" with 10-24 notes).

## Cleanup
Deleted (diagnostics, per user request + project convention):
- Root-level scratch (25): `_check_clock_events.py`, `_check_clock_markers.py`, `_check_db.py`,
  `_check_db2.py`, `_check_raw_gps.py`, `check_cfd_gps_cols.py`, `check_cfd_gps_range.py`,
  `check_cfd_receivers.py`, `check_clocked.py`, `check_db.py`, `check_db2.py`, `check_db3.py`,
  `check_deng_solutions.py`, `check_multipath.py`, `check_raw_cols.py`, `check_receivers.py`,
  `check_tables.py`, `check_transno.py`, `clear_all.py`, `clear_tables.py`, `clear_tables2.py`,
  `create_excel.py`, `drop_pos.py`, `drop_tables.py`, `analyze_current_accuracy.py` (tracked;
  restore via git if needed). None were referenced by any script, test, doc, or notebook.
- `scripts/plot_fish_tracks.py` — gitignored standalone diagnostic (recorded as removed in
  the 2026-10-19 journal; reappeared; not imported anywhere).
- `scripts/_validate_earlyfish.py` — this session's throwaway validation script.
Kept: `_earlyfish_build.log`, `_earlyfish_pipeline.log` (official build record), all
`output/` deliverables, `chinook_analysis_2025.py` (persistent chinook tool).

## Status
- **Verified:** deep_beacons idempotency fix; earlyfish data integrity; Z-accuracy numbers;
  ENT drag baseline (diff 0.00 m); fish figure reproducibility; F1 fix in core.
- **Fixed:** README clobbering in the drag figure generator.
- **Open items (unchanged from 10-24/25 journals):** 26E4 fate (absent from released_v0;
  detections 06/11), ZOI01-03 Deng2D X/Y (configured, not solved), CFD01/ZOI11 clock curves
  (~940 ms late), ZOI08 RTK survey residual, 9F8C short real detection window.
- **New open items:**
  1. `tests/test_2025_contracts.py` broken at HEAD: imports `scripts/extract_dbscan_features.py`
     (deleted in commit 8ca7ba4). Owner decision: restore the module or update/remove the
     test (the 10-09 journal records the module as "kept by owner choice" — it has since been
     removed in the latest commit without updating the test).
  2. `plot_drag_dbscan` done-check wart: for non-drag DBs it is permanently "not done"
     (harmless no-op re-run); a tags guard in `plot_drag_dbscan_done()` would make the
     status file stable.
  3. `test_resume` flaky `WinError 32` temp-file lock on this machine (environmental).
