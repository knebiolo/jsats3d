# 2026-10-30 — Pre-clock-sync jump correction (experimental) + A/B experiment

Session: 2026-10-29/30. Branch `ENM_jsat3d_edits`. Implements Kevin's 2026-10-14 spec
(verbatim quote in `2026-10-14_piecewise_clock_fix.md`): jump events from the per-file
internal offset-change codes, magnitudes from master-beacon TDoA, offsets applied to the
data files BEFORE the piecewise-regression clock sync. The 10-14 piecewise code is kept
as the legacy path (auto-detect for jumps that carry no internal flag).

## TL;DR
1. New `scripts/clock_jump_pre.py` (pipeline step 1c, off by default): finds internal
   events in `tblDetectionRaw`, quantifies each with the master-beacon (7DB7 @ ZOI08)
   first-arrival TDoA median ±300 s, applies `seconds - offset` per receiver with audit
   column `jump_offset_s`, records `tblClockJumpEvents`, invalidates downstream tables.
2. `clock_fix()` dead jump branch replaced by a loud validation gate: residual step at
   each applied-jump time must be < 50 ms, else RuntimeError. Auto-detect unchanged and
   now always runs (covers unflagged jumps in both arms).
3. A/B: arm A = `output\10292026_Test` (legacy, 2026-10-29 run); arm B =
   `output\ab_B_2026-10-29` (`apply_clock_jumps = true`, what-if copy via `--init-from`).
   Comparison: `scripts/_exp_jump_ab.py` → `output/jump_ab_report_2026-10-29/`.

## Design decisions
- **Candidates** = `OffsetChanged` OR `CounterRestart` rows (sparse; median gap ~1.8 h).
  `OneSecondAdjustmentEvidence` is dense (135,349 rows, median gap ~2 s in this DB) so it
  only corroborates a nearby candidate (±120 s), it does not trigger one.
- **Quantification**: per-burst first arrival (burst = gap ≥ 0.5 × 60 s, same pairing
  rule as the metronome), paired to the master's transmissions (nearest, ≤ 0.5 period,
  closest pair per epoch). offset = median(TDoA post) − median(TDoA pre); ±300 s per
  side, each side clipped at the neighbouring candidate. Accept 0.05 ≤ |offset| ≤ 5.0 s.
  MIN_EPOCHS_PER_SIDE = 4 full window / 3 clipped (≥150 s side); below → unresolvable.
- **Application**: one `UPDATE ... SET seconds = CASE ... END` per affected receiver
  (cumulative offsets, latest-jump-first WHEN order), `jump_offset_s` audit column
  (DEFAULT 0.0), then drop of the metronome + all downstream tables so the pipeline
  rebuilds on corrected times.
- **Gate in `clock_fix`**: for receivers with rows in `tblClockJumpEvents`, the residual
  curve must be continuous at each applied-jump time (±300 s, excluding other applied
  jumps' windows; < 3 epochs per side → "unverifiable", no raise). A mis-measured offset
  must stop the run, not be smoothed.
- **ZOI09 946 ms hack unchanged in both arms** (constant offset, not an in-file jump) —
  keeps the A/B isolated to the jump treatment.
- **ZOI04/ZOI11/CFD05 not processed** (not in surface+deep): identical in both arms;
  their jumps are listed in the pre-scan report but not applied (documented limitation).
- Common-mode: same-time same-sign step clusters are diagnostic-only in v1
  (`common_mode` column); the master (ZOI08) had 0 accepted jumps, so no master-side
  attribution was needed for this dataset.

## Pre-scan (read-only, arm A DB, final code)
- 159,893 internal evidence rows scanned → **772 jumps accepted** (696 on the 17
  processed receivers; 76 on ZOI04/ZOI11 not applied), 5,682 evidence-only, 18,255
  unresolvable (window clipped < 150 s by a neighbouring candidate, or file gap).
- Magnitudes: 4 × 0.05–0.15 s, 143 × 0.15–0.4 s, 500 × 0.4–1.2 s, 125 × ≥1.2 s (max 4.98 s).
  Median per receiver ≈ ±0.4 s (the firmware 0.4-s sync step) or ±1.0 s.
- Evidence: 723/772 (93%) carry OffsetChanged; 49 are CounterRestart-only (25 with a
  nearby OneSecondAdjustment corroboration). Confirms the candidate split in the design.
- Cross-check vs arm A's saved correction curves: 190/239 of the first pass (and 772 in
  the final pass) — where arm A's curve exists on both sides, median |Δstep| = 0 ms.
- Reverse cross-check: arm A's saved curves contain **3,018 level steps > 100 ms**;
  only **28%** have an internal event within ±120 s (flag coverage: CFD 9–30%,
  ZOI 3–46%). So the event-driven fix corrects the flagged subset; the remaining
  unflagged jumps are still handled by the legacy auto-detect in both arms (identical
  treatment → does not contaminate the A/B comparison).
- 379 of the 3,018 steps sit in arm A's ±5-min gap windows (auto-detected → data
  dropped → fish times bridged by np.interp: up to |J|/2 ≈ 200–550 m for 0.4–1.1 s);
  ~1,600 are smoothed into the curve (≤ |J| within the ~60 s ping interval straddling
  the jump).

## Code changes (all uncommitted on ENM_jsat3d_edits)
- NEW `scripts/clock_jump_pre.py` — pre-step module + CLI (`--report` read-only, `--apply`).
- CHG `scripts/_2025_pipeline.py` — step `clock_jumps` (1c) between prepare and metronome;
  `[legacy] apply_clock_jumps` (default false → step skipped, runs bit-identical),
  `clock_jump_window_s` (300), `clock_jump_min_step_s` (0.05), `clock_jump_max_step_s` (5.0);
  `_invalidate_clock_outputs()` drops derived/positions/step6/progress after an apply.
- CHG `jsats3d/jsats3d.py` `clock_fix()` — dead `tblDetectionClockFixed` jump branch
  (read a table that never carried those columns in the 2025 pipeline) replaced by the
  `tblClockJumpEvents` validation gate; auto-detect now runs unconditionally (was
  `elif`).
- NEW `config/run_data_earlyfish_2026-10-29_abB.toml` — arm B run file.
- NEW `scripts/_exp_jump_ab.py` — A/B comparison (affected windows, no-impact check,
  curve re-check, summary JSON).
- NEW `tests/test_clock_jump_pre.py` — synthetic two-jump tests (all pass).

## Arm B run state (2026-10-30)
- Step 1c `clock_jumps` ran on real data: **33.3 min**, **696 jumps applied** (134,042
  events scanned on the 17 receivers; 4,979 evidence-only; 16,198 unresolvable),
  **22,489,540 rows shifted** across 16 receivers (ZOI08 master: 0 jumps).
- **Bit-exact verification vs arm A** (rowid join; B is a byte-copy + in-place UPDATEs):
  all 29,912,682 rows of the 17 processed receivers satisfy
  `B.seconds = A.seconds − cum(A.seconds)` with max |Δ| = 2.4e-07 s (float64 rounding),
  and the `jump_offset_s` audit column equals cum exactly (0.0 everywhere). PASS.
- Metronome initially failed: the first version of `_invalidate_clock_outputs` wrongly
  dropped `tblInterpolatedTemp` (a parse artifact built from the external temperature
  files, NOT a speed_of_sound output — step 9 only reads it). Fixed in
  `_2025_pipeline.py`; table rebuilt into arm B with the parse's own loader
  (`adapt_2025_to_legacy.load_temperature_string`, same CSV + HOBO DD_N inputs).
- Rebuild verification: arm A's clock_fix legacy CSVs (computed with the original
  table) record per-epoch `avg_C`/`SoS`; the rebuilt table reproduces 6,000 sampled
  values across CFD01/ZOI01/ZOI09 with **max |ΔT| = 0.00000 °C, max |ΔSoS| = 0.0000 m/s**.
  SoS at window start/middle/end: 1453.36 / 1476.31 / 1474.12 m/s.
- Note: arm A's DB no longer contains `tblInterpolatedTemp` (file mtime 2026-10-30
  09:42 — a stray write whose source process was not identified; the table is the only
  parse artifact missing, and nothing in the A/B comparison reads it from arm A).
  Arm A's positions/clock curves/raw rows are intact and were spot-checked.
- Arm B pipeline resumed (steps 1c/prepare skipped; metronome onward re-running).

## Results (arm B complete; A/B comparison in `output\jump_ab_report_2026-10-29\REPORT.md`)
Arm B pipeline: all steps ok (metronome 7.3, clock_surface 6.6, deep_beacons 4.7,
deep_positions 9.0, deep_clocks 14.0, fish_multipath 1.8, positions 2.8, export 0.1,
positions_dbscan 0.1 min; 1c/prepare/speed_of_sound/adopt_deep skipped). The in-run
`clock_fix` validation gate passed — no residual step ≥ 50 ms at any verifiable applied
jump time (a violation would have raised and stopped the run).

**Mechanism (saved correction curves, gate-equivalent window):**
- Arm B: 305 of 696 applied-jump times measurable (the rest sit inside raw-data gaps,
  < 3 epochs/side) — median |step| **0.03 ms**, p99 0.21 ms, **max 0.91 ms**, zero ≥ 50 ms.
  The jumps are gone; the curves are continuous.
- Arm A: only 4 times measurable (A's own auto-detect gaps the curve at most detected
  jumps) — all 4 carry the jump: |step| 392–406 ms.
- A wider-window re-check's 2.6 s "steps" in B were traced to **unflagged** common-mode
  events near (but outside) the applied-jump windows (e.g. CFD01 +2997 ms jump at
  2025-06-28 23:57 sits in a 445 s beacon gap; the −2.6 s level change across that gap is
  a separate unflagged event also on CFD02/ZOI02/ZOI03/ZOI10). Treated identically in
  both arms — out of scope. A-vs-B row-level check at one such site confirms B = A − cum
  exactly (pre-gap Δ = cum, post-gap Δ = cum + applied offset).

**A/B positioning (SoS 1476.3 m/s; valid = fix ≤ 2 km from array center):**
| Set | n | both valid | rescued (A bogus→B valid) | broke (A valid→B bogus) | valid % A→B |
|---|---|---|---|---|---|
| Affected (in any applied-jump window) | 578 | 577 | **1** | **0** | 99.83→**100.00** |
| No-impact (all other transmissions) | 13,529 | 13,527 | 0 | **0** | 99.99→99.99 |

- The one rescue: 26E4 #323274 (2025-06-27 23:59:33, 149 s after the ZOI02 +435.8 ms
  jump). Arm A solved one of its 10 detections with a 703 s tag-time misassignment
  (T01=703.19 s → 548 km garbage point), dragging the per-transmission centroid to
  54.9 km; arm B's 9 detections are all in-array (centroid 20 m from the array). The
  same transmission appears in both the ZOI02 and ZOI03 window rows (4 s common-mode
  jump pair). All other affected transmissions agree (ZOI02 window: mean 12 m, p90 20 m).
- No-impact: 13,347 matched clean pairs — 65.3 % bit-identical (<1 cm), 88.9 % <1 m,
  97.0 % <10 m. The difference **decays with distance to the nearest applied jump**
  (p99: 151 m at 360–600 s → 33/44/16 m at 10 min–2 h) — the expected second-order
  piecewise re-fit (removing a jump changes that receiver's regression segments; fixes
  not using a jumping receiver's detections are bit-identical). Not a global bias.
- Confounders excluded: deep receiver geometry identical in both arms (both found=0,
  configured positions; `tblReceiver` byte-equal for ZOI01–03); temperature table
  identical (rebuilt values reproduce arm A's recorded avg_C/SoS to the digit); ZOI08
  master data untouched (0 jumps); 946 ms ZOI09 hack unchanged in both arms.
- Solution counts (arm B run, Deng roots A/B): 1D5A 46428/54042, 26E4 17922/21729,
  9F8C 1334/1714, B5ED 30537/37163.

**Verdict:** the pre-sync jump correction does what it was designed to do — the flagged
jumps are removed from the clock curves (sub-ms continuity where arm A carries ~400 ms
steps), it rescues one spurious far-field transmission in a jump window, and it breaks
zero valid fixes anywhere; the residual no-impact difference is small, local, and
decays with distance from the jumps. Feature stays off by default
(`apply_clock_jumps=false`) so existing runs remain bit-identical.

## Run notes / gotchas
- `--init-from` vacuum can fail on a full C: ("database or disk is full", ~32 GB free vs
  18.6 GB copy + vacuum temp). Workaround: the what-if DB is fully valid after the
  drops/restore (commit precedes vacuum); rerun the pipeline with `--db` only.
- PowerShell mangles inline `python -c` quotes; use script files.
- `tblInterpolatedTemp` and `tblWSEL` are parse artifacts (external temperature/
  covariate files) — never drop them in an invalidation; the metronome reads the
  temperature table before step 9 exists to rebuild it (and step 9 never rebuilds it).

## Follow-up (same day): A/B figures in legacy ClockDrift style + cleanup
- User asked for arm A / arm B figures in the project's own ClockDrift style
  (see `output\10292026_Test\jsats3d_2025_earlyfish_legacy\figures\ClockDrift_*.png`,
  generated by `clock_fix` in jsats3d.py: top = red DDoA vs beacon transmission
  number, bottom = array map). Wrote
  `output\jump_ab_report_2026-10-29\armA_ZOI02_clockdrift.png` and
  `armB_ZOI02_clockdrift.png`: same layout, window 3 h before / 1 h after the ZOI02
  +435.8 ms jump (2025-06-27 23:57:04). Arm A shows the 641 m step; arm B is
  continuous across the jump. (Arm B's curve level sits at −12.6 km because the
  applied pre-sync offsets had accumulated +9.03 s by that time; the piecewise
  clock-sync fit re-determines the per-segment level, so the absolute level is
  irrelevant — only continuity at the jump time is, and the gate verified it.)
- Data facts learned while making the figures:
  - The raw data around the jump is continuous (no raw gap); the 7-beacon gap in
    the SAVED clock-fix curve (transNo 16110–16115) is identical in both arms —
    legacy data handling (ToT join / master saved-table behavior), NOT the
    pre-sync correction and NOT auto-detect (arm B had no step left to detect).
  - The saved ZOI02 curve additionally drops +630 m → −2020 m between transNo
    16169 (~00:55) and 16178 (≥01:01:40), around the flagged 01:01:00 +191 ms
    jump, while the pre-scan post-level there was +0.62 s and the ZOI08 master
    raw data is continuous through that time. Legacy-pipeline artifact, identical
    in both arms, outside the figure window, no impact on the A/B verdict.
- Deleted per user request: all one-off diagnostic/search/plot scripts —
  `C:\Users\Ethan.Muhlestein\AppData\Local\Temp\opencode\*` (including
  plot_clockdrift_ab.py and far_points.py),
  `C:\Users\Ethan.Muhlestein\AppData\Local\Temp\attach_test.db`, and the repo's
  `scripts/_exp_jump_ab.py` (its outputs — CSVs/summary/REPORT.md — are kept in
  `output\jump_ab_report_2026-10-29\`). Also deleted the two earlier
  trajectory PNGs (superseded by the ClockDrift-style figures).
- Repo now holds only pipeline feature code: `scripts/clock_jump_pre.py`,
  `scripts/_2025_pipeline.py` (step 1c), `jsats3d/jsats3d.py` (validation gate —
  no-op when `tblClockJumpEvents` is absent, so 2019 runs are unaffected),
  `tests/test_clock_jump_pre.py`, `config/run_data_earlyfish_2026-10-29_abB.toml`,
  and this journal entry.
