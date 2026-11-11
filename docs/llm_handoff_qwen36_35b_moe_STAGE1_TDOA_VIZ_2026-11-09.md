# Handoff for Qwen 3.6 35B MoE — Stage 1 beacon TDoA visualization for PM review (2026-11-09)

**Purpose.** Implement Drew's request: interactive per-receiver plots of beacon TDoA (beacon 7DB7 at master ZOI08) comparing **uncorrected** vs **Stage 1-corrected** timing, with a vertical line for every **applied** Stage 1 correction. Stage 2 / DBSCAN supplement stays disabled. Deliverable: standalone Plotly HTML files + index + README in a dated output folder.

**Read first (in this order):** `docs/llm_handoff_qwen36_35b_moe_RECOVERY_2026-11-07.md` (rules, traps), then this file. **If files disagree, this one wins.** Work **one task at a time**; each ends with **STOP** (report, wait).

## Final decisions (locked 2026-11-09 by the owner; do not change mid-task)

- **D1 — Uncorrected source:** Arm A (`output/10292026_Test/jsats3d_2025_earlyfish.db`) read-only. True unshifted baseline; its 7DB7 rows are identical to B1's (verified 2026-11-09: same per-receiver counts, 1,639,714 total; no `jump_offset_s` column; no `tblClockJumpEvents`).
- **D2 — Shared x-axis = Stage 1-corrected time.** Corrected series plotted at its own (B1 stored) time. Uncorrected series' x positions mapped through that receiver's cumulative offset curve (rule in Appendix B.2). Master ZOI08 is unshifted (verified: `jump_offset_s` = 0 on all 281,219 ZOI08 7DB7 rows), so the basis is unambiguous. Never label times "UTC" — axis label is "local time (PDT)".
- **D3 — Segmentation:** one full-span (2025-06-15 → 2025-08-06) file per receiver using `scattergl`. Auto-split into calendar-month segments **only if** a receiver's combined point count (both arms) exceeds 150,000. None qualify today (max 142,066, ZOI11).
- **D4 — Applied corrections only.** Vertical lines for the 696 rows of `tblClockJumpEvents` (all `stage=1`). The 249 unresolved Stage 1 candidates in `tblClockJumpCandidates` are NOT plotted (they were never applied).
- **D5 — Generator script location.** The single new `.py` file lives **inside the output folder** (`output/11092026_stage1_tdoa_review/make_stage1_tdoa_plots.py`), next to its deliverables. Nothing is added to `scripts/` or the repo root (standing "no new scripts" rule); the owner can move it into `scripts/` at commit time if wanted.

## 0. State at the start (verified 2026-11-09; re-verify in Task 0, do not assume)

- Tests: `74 passed, 2 subtests passed in 15.8s` via `python -m pytest tests -q` from `C:\Dev\jsat3d\jsats3d`.
- Env python (call it directly; `conda` is NOT on PATH): `C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe` (Python 3.11.15, plotly 6.9.0, pandas 3.0.5, numpy 2.4.6).
- Git HEAD `55eb0d6` (2026-11-08 "Add experimental clock sync safeguards"); working tree clean.
- **B1** (the Stage 1-only isolated arm) `C:\Dev\jsat3d\jsats3d\output\11082026_whatif_B1\jsats3d_2025_B1.db`:
  - 5,976,518,656 bytes; last write 2026-11-08 18:13:52 (per 11-08 journal — re-verify).
  - `tblClockJumpEvents`: **696 rows, all `stage=1`**, 16 receivers (CFD01 39, CFD02 55, CFD03 34, CFD04 30, CFD06 13, CFD07 20, CFD08 54, CFD09 5, ZOI01 169, ZOI02 123, ZOI03 84, ZOI05 18, ZOI06 1, ZOI07 7, ZOI09 19, ZOI10 25). Offsets −3,618.85 ms … +4,599.64 ms (median +381.09 ms).
  - `tblClockJumpValidation`: 696 rows — **352 passed / 343 unverifiable / 1 failed**. The failure: CFD08, `jump_time_s=1753487823.677804`, residual step 0.5188 ms.
  - `tblClockJumpCandidates`: 945 rows = 696 accepted + 249 unresolved (Stage 1 review ledger).
  - 7DB7: 1,639,714 rows across all 20 receivers; window 1749945638.87 → 1754524779.52 (= 2025-06-15 00:00:39 → 2025-08-06 23:59:39 local PDT stored as-if-UTC).
  - `tblDetectionRaw`: 7,492,659 rows. ZOI08 `jump_offset_s` = 0.0 on every 7DB7 row.
- **Arm A** (uncorrected baseline) `C:\Dev\jsat3d\jsats3d\output\10292026_Test\jsats3d_2025_earlyfish.db`:
  - 18,631,671,808 bytes; last write 2026-11-06 18:34:13.
  - 7DB7 per-receiver counts **identical to B1** (verified 2026-11-09; totals 1,639,714, same min/max). No `jump_offset_s` column, no `tblClockJumpEvents`. `tblDetectionRaw` holds 29,912,682 rows (superset DB — always bound queries by the 7DB7 window).
  - `PRAGMA quick_check` on this DB takes >10 min — do NOT run it; the 7DB7 aggregation scan takes ~3–5 min, that is expected.
- **Production Arm B** `C:\Dev\jsat3d\jsats3d\output\11062026_clocksync\jsats3d_2025_earlyfish.db`: 5,976,518,656 bytes, last write 2026-11-07 22:16:26. Must be unchanged at the end.
- B1 paired first-arrival epoch counts per listener (computed 2026-11-09 with `clock_jump_pre.beacon_first_arrivals`, period 60.0): total **1,003,970** — CFD01 70,954; CFD02 64,994; CFD03 42,625; CFD04 64,644; CFD06 39,187; CFD07 52,941; CFD08 56,888; CFD09 49,787; ZOI01 62,464; ZOI02 69,911; ZOI03 69,092; ZOI04 47,242; ZOI05 50,138; ZOI06 16,304; ZOI07 51,852; ZOI09 66,826; ZOI10 57,088; ZOI11 71,033.
- Receivers to plot: the **18 non-master receivers with 7DB7 data** (CFD01-04, CFD06-09, ZOI01-07, ZOI09-11). ZOI04, ZOI11, CFD05 have **no applied corrections** → their plots have no vertical lines (that is informative, keep them).

## 1. Rules (all RECOVERY rules apply; these are the ones that failed before)

1. **Working directory.** First command: `Set-Location C:\Dev\jsat3d\jsats3d`; `git rev-parse --show-toplevel` must print `C:/Dev/jsat3d/jsats3d`. Absolute paths for all writes.
2. **Journal proof.** After every task append full prose (commands, numbers, decisions, blockers) to `C:\Dev\jsat3d\jsats3d\.ai_journal\session\2026-11-09_stage1_tdoa_visualization.md`, then run `Get-Item` on **that file's full path** and paste its LastWriteTime and last 3 lines in the report. A report without this proof is incomplete.
3. **Read-only databases, always.** Arm A, B1, and production Arm B open as `sqlite3.connect("file:%s?mode=ro" % path.replace("\\","/"), uri=True)` **plus** `con.execute("PRAGMA query_only=ON")`. No DB copies. No writes anywhere except `C:\Dev\jsat3d\jsats3d\output\11092026_stage1_tdoa_review\`.
4. **No new files outside the output folder** — not in `scripts/`, not in the repo root. The one new `.py` file is the generator inside the output folder (D5). No new package installs (plotly 6.9.0 already present); if anything is missing, STOP and report.
5. **No parameter changes.** Period 60.0 s, pairing tolerance 0.5 × period, min step 0.05 s, max step 5.0 s. No thresholds, colors, or layout choices invented beyond Appendix C.
6. **Time basis.** Study wall-clock (PDT) stored as-if-UTC. Convert stored seconds to display strings with `pd.to_datetime(t, unit="s", utc=True)` and then **strip the offset** (pass naive `YYYY-MM-DD HH:MM:SS` strings to Plotly). Axis title: `local time (PDT)`. Never label anything "UTC".
7. **Report only numbers you reproduced this session.** State counts exactly (18 receivers, not "about 18"; 696 lines, not "several hundred").
8. **Ask before deleting or overwriting anything in `output/`.** New dated folder only.
9. **Disk.** Check free space in Task 0; the deliverable is ~150–250 MB total. Stop and report if free space < 5 GB.
10. **`conda run` drops piped stdin** (standing decision) — run all Python from files or `-c` strings via the env `python.exe`, never here-strings through `conda run`.

## 2. Tasks

### Task 0 — Setup (read-only)

1. Working-directory check (Rule 1). `git status --short` (expect clean).
2. Full test suite (Rule: `74 passed, 2 subtests passed`).
3. Free disk space.
4. Re-verify the §0 facts that bound this task (B1 size/last-write, 696 events all stage=1, validation 352/343/1, ZOI08 `jump_offset_s` all 0.0, 7DB7 total 1,639,714, Arm A size/last-write, production B size/last-write). Record each as `value → OK / MISMATCH`.
5. Create `C:\Dev\jsat3d\jsats3d\output\11092026_stage1_tdoa_review\` with subfolder `plots\`. Create the journal file (Rule 2).
**STOP. Report with journal proof.**

### Task 1 — Extract both TDoA series (read-only; CSVs into the output folder)

Run the snippet below (save it to a `.py` file in the output folder named `extract_series.py` and execute it). It writes `series_armA_uncorrected.csv` and `series_b1_corrected.csv`, each with columns `Rec_ID, t, transNo, tdoa_s`.

```python
import sqlite3, sys
import pandas as pd
sys.path.insert(0, r"C:\Dev\jsat3d\jsats3d\scripts")
import clock_jump_pre as cpj

OUT = r"C:\Dev\jsat3d\jsats3d\output\11092026_stage1_tdoa_review"

def extract(db_path, out_csv, label):
    con = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    con.execute("PRAGMA query_only=ON")
    span = pd.read_sql("SELECT MIN(seconds) t0, MAX(seconds) t1 FROM tblDetectionRaw WHERE Tag_ID='7DB7'", con)
    t0, t1 = float(span.t0.iloc[0]) - 60.0, float(span.t1.iloc[0]) + 60.0
    arr = cpj.beacon_first_arrivals(con, "ZOI08", "7DB7", 60.0, t0, t1)
    con.close()
    arr.to_csv(out_csv, index=False)
    print(label, "epochs:", len(arr), " window: %.2f .. %.2f" % (t0, t1))
    return arr

a = extract(r"C:\Dev\jsat3d\jsats3d\output\10292026_Test\jsats3d_2025_earlyfish.db", OUT + r"\series_armA_uncorrected.csv", "ArmA")
b = extract(r"C:\Dev\jsat3d\jsats3d\output\11082026_whatif_B1\jsats3d_2025_B1.db", OUT + r"\series_b1_corrected.csv", "B1")
ca, cb = a.groupby("Rec_ID").size(), b.groupby("Rec_ID").size()
print(pd.DataFrame({"armA": ca, "b1": cb, "delta": cb - ca}).to_string())
```

Verify and report:
- B1 total epochs **1,003,970** and the per-receiver table matches §0 exactly. Arm A total should be within ±200 per receiver of B1 (pairing can flip at a jump near a burst boundary); report the exact delta table and STOP if any |delta| > 500.
- Both windows identical to 1749945638.87 → 1754524779.52 (±60 s padding).
- No NaN in `tdoa_s`; `transNo` starts at 0 in both arms.
- Re-check Arm A and B1 size + last-write: unchanged.
**STOP. Report tables with journal proof.**

### Task 2 — Offset curves, line positions, spot-checks (pure computation)

Build the per-receiver cumulative offset curves from the B1 ledger and the 696 line records. Save to a `.py` file in the output folder named `build_lines.py`; it writes `stage1_correction_lines.csv` with columns:

`Rec_ID, jump_time_s, cum_offset_s, corrected_event_time_s, offset_ms, applied_shift_ms, evidence, source_file, pre_level_ms, post_level_ms, n_pre, n_post, common_mode, status, residual_step_ms`

Semantics (Appendix B.2): `cum_offset_s` = cumsum of that receiver's `offset_s` in `jump_time_s` order; `corrected_event_time_s = jump_time_s - cum_offset_s` (the event's own row is shifted, so its own offset is included); `applied_shift_ms = -offset_s × 1000` (the shift actually applied to recorded times); `status` and `residual_step_ms` joined from `tblClockJumpValidation` on (Rec_ID, jump_time_s).

Then spot-check **three events — one passed, one unverifiable, one failed (the CFD08 one)** — by recomputing the uncorrected TDoA median in `[t_ev − 300, t_ev)` and `(t_ev, t_ev + 300]` for that receiver from `series_armA_uncorrected.csv` and comparing to the ledger `pre_level_s`/`post_level_s`. Tolerance: ±20 ms (windows here are not clipped by neighbouring events; if one is, say so and relax to ±100 ms with an explanation). Also verify: for the 3 event-free receivers (ZOI04, ZOI11, CFD05) the mapping is the identity.
**STOP. Report the three spot-checks (ledger vs recomputed, ms) with journal proof.**

### Task 3 — Write the generator script (write only to the output folder)

Write `C:\Dev\jsat3d\jsats3d\output\11092026_stage1_tdoa_review\make_stage1_tdoa_plots.py` exactly per **Appendix C** (traces, colors, hover content, naming, segmentation rule, index + README generation). It must:
- read the two series CSVs and `stage1_correction_lines.csv` (all in the output folder) — **no database access at all**;
- write only `plots/*.html`, `index.html`, `README.md`, `summary.csv` inside the output folder;
- print a per-receiver table: n_armA, n_b1, n_lines (passed/unverifiable/failed), file name(s), file size MB;
- apply the D3 segmentation rule (150,000 combined points → monthly segments; expected: no splits).

Self-check before running: `--help` works; a `--selftest` mode that renders ZOI06 (smallest) to a temp file in the output folder and deletes nothing.
**STOP. Show the script's argument list, main function signatures, and the selftest result. Wait for the user's OK.**

### Task 4 — Generate all plots

Run the generator (full run). Verify:
- 18 files in `plots/` (plus any monthly segments, none expected); every file > 3.6 MB (inlined plotly.js ~3.5 MB + data).
- Per file: exactly 2 `scattergl` traces; number of vertical line traces equals that receiver's line count (count `"yref":"paper"` occurrences); ZOI04/ZOI11/CFD05 files have 0 line traces.
- CFD08 file contains the red (failed) line and hover text mentioning `0.5188` (residual ms) — this is the one failed validation; it must be visible.
- `index.html` links resolve (every linked file exists).
- **Browser spot-check (required — you have browser access for this task):** open `plots/CFD08_stage1_tdoa.html`, `plots/ZOI01_stage1_tdoa.html` (densest, 169 lines), and `plots/ZOI06_stage1_tdoa.html` (sparsest, 1 line) in browser tabs. Confirm by eye: at each line the grey series steps and the blue series does not (or steps by the residual where the validation failed); report what you see, including anything that looks wrong. If your browser tooling can save screenshots, save a zoomed view around one line per file as `plots/<REC>_spotcheck.png` and list them in the report; if not, the written description stands.
- Re-check Arm A, B1, production B size + last-write: unchanged.
**STOP. Report the per-receiver table with journal proof.**

### Task 5 — Index/README/summary already generated; finalize and hand off

1. Read `README.md` and `index.html`; verify they contain everything Appendix C.5 requires (sources with sizes/last-writes, method, time-basis statement, color legend, segmentation policy, regeneration command, limitations). Fix the generator if anything is missing, rerun, re-verify.
2. `summary.csv` present and matches the Task 4 table.
3. Add one line to `LONG_TERM_CONTEXT.md` **Completed Milestones**:
   `2026-11-09: Stage 1 TDoA review plots (18 receivers, Arm A uncorrected vs B1 corrected, 696 applied-correction lines) in output/11092026_stage1_tdoa_review/; deliverable for PM review, not a validation result.`
4. Journal: full session record; final `Get-Item` proof.
5. Propose (do not run) a commit plan for the owner: new output folder contents + this handoff doc + the journal + the LONG_TERM_CONTEXT line.
**STOP. Final report with journal proof.**

## 3. Interpreting results (what to say, what not to say)

- The plots are a **review aid**. Never say Stage 1 "works", "is validated", or "is wrong". Say what is visible: where lines sit on visible grey steps (consistent), where a line sits with no step or a much smaller step (questionable — flag it), where the blue series still steps (jumps that remain), and count any ±60 s outliers (pairing flips at burst boundaries).
- The 1 failed validation (CFD08, residual 0.5188 ms) and the 343 unverifiable lines are **part of the data**, not defects to hide.
- 1 ms of timing error ≈ 1 m of position error; this is a timing review, not a position-accuracy review.
- Time basis is study wall-clock (PDT stored as-if-UTC). **Never label times "UTC".**

## Appendix A — Reference tables (reproduce before changing anything)

7DB7 raw rows per receiver (Arm A = B1, verified 2026-11-09):

| Rec | rows | Rec | rows | Rec | rows |
|---|---|---|---|---|---|
| CFD01 | 86,408 | ZOI01 | 160,793 | ZOI09 | 105,977 |
| CFD02 | 66,115 | ZOI02 | 123,224 | ZOI10 | 62,104 |
| CFD03 | 43,401 | ZOI03 | 109,431 | ZOI11 | 115,827 |
| CFD04 | 86,981 | ZOI04 | 52,699 | total | 1,639,714 |
| CFD05 | 0 | ZOI05 | 50,668 | | |
| CFD06 | 42,612 | ZOI06 | 17,463 | | |
| CFD07 | 57,658 | ZOI07 | 64,660 | | |
| CFD08 | 60,349 | ZOI08 (master) | 281,219 | | |
| CFD09 | 52,125 | | | | |

Paired first-arrival epochs per listener (B1): CFD01 70,954; CFD02 64,994; CFD03 42,625; CFD04 64,644; CFD06 39,187; CFD07 52,941; CFD08 56,888; CFD09 49,787; ZOI01 62,464; ZOI02 69,911; ZOI03 69,092; ZOI04 47,242; ZOI05 50,138; ZOI06 16,304; ZOI07 51,852; ZOI09 66,826; ZOI10 57,088; ZOI11 71,033; total 1,003,970.

Applied Stage 1 events per receiver: CFD01 39; CFD02 55; CFD03 34; CFD04 30; CFD06 13; CFD07 20; CFD08 54; CFD09 5; ZOI01 169; ZOI02 123; ZOI03 84; ZOI05 18; ZOI06 1; ZOI07 7; ZOI09 19; ZOI10 25; total 696. Offset range −3,618.85 … +4,599.64 ms, median +381.09 ms.

## Appendix B — Code references (reuse, do not reinvent)

**B.1 `scripts/clock_jump_pre.py:174` — `beacon_first_arrivals(con, master_rec, beacon_tag, period, t0, t1)`**
"Step 2 input: first arrival per burst per receiver, paired to master transmissions. Same pairing rule as beacon_pairwise_dbscan.pairwise_series (nearest master epoch within 0.5 × period, closest pair per epoch) so the magnitude is consistent with the metronome. `tdoa_s = t_listener - t_master`; a receiver clock step of J seconds after the event appears as a +J step in this series."
Mechanics: a burst starts when the gap since the previous arrival on the same receiver is ≥ 0.5 × period; `transNo` = index of the master's (ZOI08) first arrivals in time order; each listener arrival is `merge_asof`-matched to the nearest master `transNo` within 0.5 × period, one closest pair per `transNo`. Returns columns `Rec_ID, t, transNo, tdoa_s` (no row for the master itself). It reads `tblDetectionRaw` with `Tag_ID = ?` and `seconds >= t0 AND seconds < t1`, so pass the window with ±60 s padding.

**B.2 `scripts/clock_jump_pre.py:746` — `apply_offsets(db, jumps)` — the mapping rule**
For each receiver, jumps sorted by `jump_time_s`, cumulative `s_k = o_1+…+o_k`. The applied shift is `WHEN seconds >= t_k THEN seconds - s_k` (a row at exactly `t_k` IS shifted). Therefore, for original-time `t`: applied shift `s(t) = s_k` where `k = max{i : t_i <= t}` (0 if none); corrected time `x = t - s(t)`. Line position for event k: `x_k = t_k - s_k`. `jump_offset_s` in `tblDetectionRaw` stores exactly `s(t)` — use it only as a cross-check (e.g. on ZOI08 it must be 0 everywhere).

**B.3 `scripts/clock_jump_pre.py:631` — `quantify_jumps(...)` — what the ledger fields mean**
For each internal-code candidate event: `pre` = first-arrival epochs in `[event_start − 300 s, event_start)`, `post` in `(t_ev, t_ev + 300 s]` (clipped at neighbouring candidate events); `pre_level_s`/`post_level_s` = medians; **`offset_s = post_level_s − pre_level_s`** (the TDoA step in uncorrected time, i.e. the direction and size of the receiver clock error); accepted when `0.05 <= |offset_s| <= 5.0`. `evidence` is a `;`-joined string from `OffsetChanged`, `CounterRestart`, `OneSecondAdjustment`. `jump_time_s` is in **original (pre-shift) receiver time**; `jump_time_local` is its PDT display form.

**B.4 Ledger and validation schemas (B1, verified 2026-11-09)**
`tblClockJumpEvents`: `Rec_ID, jump_time_s, jump_time_local, evidence, source_file, n_pre, n_post, pre_level_s, post_level_s, offset_s, common_mode, stage`.
`tblClockJumpValidation`: `Rec_ID, jump_time_s, start_s, end_s, n_pre, n_post, residual_step_s, status` with `status` ∈ {passed, unverifiable, failed}. Join key: (Rec_ID, jump_time_s).

**B.5 pandas 3.x notes:** `pd.read_sql` with a raw `sqlite3` connection works. Nothing in the snippets uses removed APIs. If any deprecation warning breaks a step, STOP and report — do not silently "fix" it.

## Appendix C — Generator spec (Appendix C.1–C.5 are binding)

**C.1 Inputs / outputs.** Inputs: `series_armA_uncorrected.csv`, `series_b1_corrected.csv`, `stage1_correction_lines.csv` (all in the output folder). Outputs, all in `C:\Dev\jsat3d\jsats3d\output\11092026_stage1_tdoa_review\`: `plots\<REC>_stage1_tdoa.html` (or `plots\<REC>_stage1_tdoa_<YYYY-MM>.html` for monthly segments), `index.html`, `README.md`, `summary.csv`. No database access. No writes anywhere else.

**C.2 Data preparation per receiver.**
- Corrected series: `x = t` (B1 stored seconds), `y = tdoa_s × 1000` (ms).
- Uncorrected series: `x = t − s(t)` with `s(t)` from that receiver's offset curve (B.2; receivers with no events: identity), `y = tdoa_s × 1000` (ms).
- x values to Plotly: naive `YYYY-MM-DD HH:MM:SS` strings derived with `pd.to_datetime(x, unit="s", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")` (Rule 6).
- Segmentation (D3): if `len(armA) + len(b1) > 150000`, split both series by calendar month of the corrected x; each segment is a separate file whose title carries the segment label (`2025-06`, `2025-07`, `2025-08 (partial)`).

**C.3 Traces and styling per file.**
1. Grey trace: `go.Scattergl(x=…, y=…, mode="markers", name="Uncorrected (Arm A)", marker=dict(size=2.5, color="#9a9a9a", opacity=0.55), hoverinfo="x+y")`.
2. Blue trace: `go.Scattergl(x=…, y=…, mode="markers", name="Stage 1 corrected (B1)", marker=dict(size=2.5, color="#1f6fb5", opacity=0.8), hoverinfo="x+y")`.
3. One line trace per applied correction: `go.Scatter(x=[xk, xk], y=[0, 1], yref="paper", mode="lines", line=dict(color=<status color>, width=1.5), hoverinfo="text", hovertext=[<hover html>], name=<status>)`. Status colors: passed `#2e8b57`, unverifiable `#e8a020`, failed `#d62728`. (Line traces are separate traces so each is individually hoverable; a receiver's legend should collapse to one entry per status — set the same `name` per status group.)
4. Hover html per line (exact fields, in this order):
   `<b>{Rec_ID}</b> — Stage 1 applied correction<br>Event time (original): {jump_time_local}<br>Event time (corrected): {xk as PDT string}<br>Evidence: {evidence}<br>Source file: {basename(source_file)}<br>TDoA step (uncorrected): {offset_ms:+.1f} ms<br>Shift applied to recorded times: {applied_shift_ms:+.1f} ms<br>Pre / post level: {pre_level_ms:.1f} / {post_level_ms:.1f} ms<br>Epochs used: {n_pre} pre / {n_post} post<br>Common-mode listeners: {common_mode}<br>Validation: {status}` plus, when `residual_step_ms` is not null: `<br>Residual step after correction: {residual_step_ms:+.3f} ms`.
5. Layout: `title = "7DB7 @ ZOI08 vs {REC} — Stage 1 only (B1){, segment label if split}"`; `xaxis_title="local time (PDT)"`; `yaxis_title="TDoA (ms) — listener minus master, 7DB7 first arrivals"`; `width=1200, height=650, autosize=True`; light grid; legend top-right.
6. `fig.write_html(path, include_plotlyjs=True)` — inlined on every file (standalone, offline-capable). Expected file size ≈ 3.5 MB (plotly.js) + points × ~45 B + lines.

**C.4 `index.html`.** Self-contained page, one table: receiver | uncorrected pts | corrected pts | lines (passed/unverifiable/failed) | file link(s) | size MB. Sort receivers CFD01…CFD09, ZOI01…ZOI11. Header states: source DBs (B1 stage-1-only; Arm A unshifted), master 7DB7 @ ZOI08, window 2025-06-15 → 2025-08-06 (local PDT), "review aid, not a validation result".

**C.5 `README.md` (generated, so it can never drift).** Must contain: (a) purpose and Drew's request in one paragraph; (b) source databases with full paths, sizes, last-write times, and "opened read-only"; (c) method — first arrival per burst (burst gap ≥ 30 s), nearest-master pairing within 30 s, `transNo` alignment, uncorrected x mapped through the receiver's cumulative offset curve (B.2), master unshifted (verified); (d) time basis — study wall-clock PDT stored as-if-UTC, x-axis = Stage 1-corrected time, never "UTC"; (e) color legend (grey/blue/green/amber/red) with meanings; (f) segmentation policy (150,000-point threshold, monthly, none triggered this run); (g) regeneration command (exact `python.exe make_stage1_tdoa_plots.py` invocation); (h) limitations — ±60 s outliers are pairing flips at burst boundaries, ZOI04/ZOI11/CFD05 have no applied corrections, point hover is time+TDoA only (detail lives in the lines), plots are a review aid not a validation.

**C.6 `summary.csv`.** One row per receiver: `Rec_ID, n_armA, n_b1, n_lines, n_passed, n_unverifiable, n_failed, files, total_size_mb`.

---

# FOLLOW-UP (appended 2026-11-09 after owner review of the Tasks 0–5 output)

**Context.** The owner independently reviewed the completed run (read-only). Verdict: deliverable structure, x/y mapping, line placement, and labels are all verified correct (at transNo-matched bursts, `(yB1 − yArmA) + cum_offset(t) = 0.0000 ms` across 332,753 pairs on 6 receivers; all 18 line counts exact; no "UTC" in user strings). **Three defects remain.** Tasks 6–8 fix them. All §1 rules still apply; D1–D5 remain locked; do not re-extract series (Tasks 1–2 CSVs stay, except the lines CSV rebuilt in Task 6).

**Pre-approved overwrites (owner-approved exception to Rule 8):** `stage1_correction_lines.csv`, `plots/*.html` (18 files), `index.html`, `README.md`, `summary.csv` in the output folder — and only those. Everything else in `output/` is read-only. **Do not touch `scripts/visualize_beacon_tdota.py`** (pre-existing stray script; the owner handles it).

## Task 6 — Fix the common-mode hover bug and regenerate

The hover on every one of the 696 lines currently shows a count scaled and mis-united, e.g. `Common-mode listeners: 1000.0 ms` (meaning: 1 listener). Two-line fix:

1. `build_lines.py` line 47: **delete** `grp["common_mode"] = grp["common_mode"] * 1000` (the ledger `common_mode` is a listener count, not a time).
2. `make_stage1_tdoa_plots.py` line 135: `common_mode = int(round(line["common_mode"])) if pd.notna(line.get("common_mode")) else 0`; line 148: `f"Common-mode listeners: {common_mode}<br>"`.

Then rerun, in order, via the env python: `build_lines.py` (rebuilds the lines CSV) and `make_stage1_tdoa_plots.py` (regenerates all 18 plots + index + README + summary). No database access in either.

Verify before STOP:
- `stage1_correction_lines.csv`: `common_mode` column now holds raw counts (CFD08 failed event = 1).
- Hover census: occurrences of `Common-mode listeners: ` across the 18 HTML files = 696; occurrences of the buggy pattern `Common-mode listeners: [\d.]+ ms` = **0**.
- Re-run the full 18-file structural check: exactly 2 scattergl traces per file; line trace counts 39/55/34/30/13/20/54/5/169/123/84/0/18/1/7/19/25/0; x-axis title `local time (PDT)`; CFD08 failed-line hover still shows `Validation: failed` and `Residual step after correction: +0.519 ms`.
- File sizes in the same 6.0–11.3 MB band; all 18 `index.html` links resolve.
- Arm A, B1, production B sizes + last-writes unchanged.
**STOP. Report the census numbers with journal proof.**

## Task 7 — Browser spot-check (required; you have browser access)

Open in browser tabs: `plots/CFD08_stage1_tdoa.html` (1 red failed line), `plots/ZOI01_stage1_tdoa.html` (169 lines, densest), `plots/ZOI06_stage1_tdoa.html` (1 line, sparsest).

Confirm by eye and report:
- At each correction line the **grey** series steps and the **blue** series does not — except at the CFD08 failed line, where blue may still show a ~0.5 ms residual step (invisible at this scale; that is the point of the red line).
- Legend, titles, and the `local time (PDT)` axis label render correctly.
- Amber lines (unverifiable) visibly outnumber green (passed) — expected (343 vs 352 overall; e.g. ZOI01 is 87 amber vs 82 green).
- Anything that looks wrong (missing series, broken hovers, clipped ranges) — report it verbatim.

Save a zoomed screenshot around one line per file as `plots/<REC>_spotcheck.png` if your browser tooling supports screenshots; otherwise the written description stands.
**STOP. Report observations + screenshot paths with journal proof.**

## Task 8 — Journal corrections and final handoff

Append a "Corrections" section to the session journal:
1. **B1 last-write:** the Task 0 "MISMATCH (fractional seconds)" explanation was wrong. B1's true last write 2026-11-08 18:43:47 is the documented stage-1 candidate restore (945 rows) recorded in the follow-up section of `2026-11-08_stage2_whatif_B1.md`, after the 18:13:52 snapshot that handoff §0 cited. Data integrity unaffected (size identical; contents re-verified: 945 = 696 accepted + 249 unresolved).
2. **Task 4 proof:** the pasted 5:40 PM `Get-Item` predates actual plot generation (5:48 PM); attach a fresh `Get-Item` proof with the current timestamp for this follow-up.
3. **Task 3 gate:** record that the user approved the Task 3 design before Task 4 ran.

Then:
- Re-confirm the 3 database sizes + last-writes one final time.
- Confirm the commit plan (journal, Task 5) still lists exactly the right files; note the corrected files have identical names.
- Final report: what changed in this follow-up (2 code lines + regenerated artifacts + spot-check results), what did not change (series CSVs, mapping, all other hover fields), and the one remaining owner action (quarantine of `scripts/visualize_beacon_tdota.py` — not yours to do).
**STOP. Final report with journal proof.**
