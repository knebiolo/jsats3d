# 2026-11-01 — ZOI11 A/B: debris-barrier static holds (DB-1..DB-5) + forebay holds (FBY-S-*)

## Objective
Determine whether receiver ZOI11 (on the debris barrier, deployed 2025-06-10) helps position the
FFD3 test tag near the debris barrier: positioning with vs. without ZOI11, using the five
2025-06-11 debris-barrier holds (no GPS) plus the nine 2025-06-10 forebay holds (known GPS).

## Changes (additive; no pipeline code touched)
- Added `config/run_data_zoi11_A.toml` (control, 9-receiver Deng) and `config/run_data_zoi11_B.toml`
  (treatment, + ZOI11 in `surface_receivers` and `receivers_3d`, 10-receiver Deng). Both:
  window 2025-06-10 00:00 → 2025-06-13 23:59, `tags = ['FFD3']`, `apply_clock_jumps = false`,
  drag/figure outputs off.
- All outputs under `output/zoi11_test/` (C: drive only, per standing rule):
  - `raw_staging/` — 48 **hard links** to `raw_data\20250610_Receivers_array_testing` (23) +
    `raw_data\20250618_Receivers` (25); zero disk cost, originals untouched.
  - `armA/zoi11_A.db` — parsed once via `parse_ats_raw_to_legacy.py` CLI (2,063,206 detections,
    20/20 target serials, temperature HOBO coverage 06-02→07-10 spans the window).
  - `armB/zoi11_B.db` — byte-copy of the parsed armA DB (identical raw rows by construction).
  - Both arms then `run_data.py --skip-build` (finish_database + full 2025 pipeline,
    --no-drag-plots). All steps ok in ~5 min each.
  - `dbscan_zoi11_{A,B}_ZOI08_anchorZOI03/` (metronome byproducts) moved from `output/` into
    `zoi11_test/` so everything for this test lives in one folder.
- Comparison: `output/zoi11_test/compare_zoi11_ab.py` (read-only on the two arm DBs) →
  `hold_summary.csv`, `per_transmission.csv`, `zoi11_summary.json`, `zoi11_ab_holds.png`.
- Local frame for the FBY GPS conversion: origin recovered from each arm's own `tblReceiver`
  (median of easting−X / northing−Y; 13 of 20 receivers carry no hydrophone offset), UTM 26910.

## Key results
- **FBY holds (GPS truth), 5 holds where ZOI11 heard the tag:** mean error 21.0 → 10.8 m,
  median 11.6 → 8.0 m (B vs A). Best single: FBY-S-10 47.6 → 16.6 m.
- **4 holds with 0 (or 1) ZOI11 pings: arms bit-identical** (Deng only combines receivers with a ToA).
- **DB holds (no truth):** spread std tighter in 4 of 5 holds; depth vs 20 ft (6.1 m) deployment:
  arm B within 1.1 m for all five, arm A DB-5 10.2 m off (−16.3 m); solutions 1.7–1.8× more in B.
- ZOI11 carries 15–18% of all FFD3 detections at the barrier holds (56–93 pings/60 s vs ~25 avg
  per other receiver) — it is the strongest receiver there.
- ZOI11 clock (arm B): 4,766 sync pings, 6 auto-detected jumps (0.106–0.402 s), 13-segment fit,
  0.82 s offset era through most of 06-10 morning; handled by the legacy path (no
  apply_clock_jumps).
- Bad holds in BOTH arms (unrelated): FBY-S-7 (205.9 m), FBY-S-4 (71.8 m) — tag motion/multipath.

## Verdict
**ZOI11 helps near the debris barrier.** Recommend adding it to `surface_receivers` +
`receivers_3d` for runs covering the barrier/forebay area. No algorithm change needed.

## Cleanup
- Deleted temp probe scripts from `%TEMP%\opencode` (probe_*.py, read_test_sheets.py,
  test_sheets_media\) — standing rule: no diagnostic scripts left behind.
- Committed to `ENM_jsat3d_edits`: `tests/test_clock_jump_pre.py` (missed in a5b53a0) and the two
  zoi11 tomls + this journal. `output/` is gitignored (data stays local).

## Open items
- None for this test. Production tomls (earlyfish etc.) intentionally untouched — the ZOI11
  addition is a run-file change the team can adopt when they want it.
