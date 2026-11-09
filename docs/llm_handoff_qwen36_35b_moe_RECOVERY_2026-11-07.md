# Recovery Handoff for Qwen 3.6 35B MoE — Clock-Jump Workstream (2026-11-07, late)

**Supersedes** `docs/llm_handoff_qwen36_35b_moe_2026-11-07.md` (written before several findings
below). Background and architecture are still in `docs/llm_handoff_clock_jumps_2026-11-06.md`.
**If any file disagrees with this one, this one wins.**

Work **one numbered task at a time**. Every task ends with **STOP**: report to the user and wait.
Do not start the next task in the same turn. Chat replies: short sentences. Journal entries: full prose.

---

## 1. Rules (violating any of these is a defect)

1. **1 ms of timing error ≈ 1 m of position error.** Say so whenever you discuss timing.
2. **Two different numbers:** 0.5 ms = timing warning budget. 50 ms = jump-application minimum. Never mix them.
3. **Raw data is read-only:** nothing under `...\Documents\jsats_data\` or `K:\`.
4. **Arm A is the baseline and is read-only:** `output/10292026_Test/jsats3d_2025_earlyfish.db`.
   Open every database for analysis with `file:...?mode=ro` and `PRAGMA query_only=ON`.
5. **Never invent values.** No forward/back-filled temperatures, no copied filter tables, no made-up timestamps.
6. **Never weaken or bypass a gate.** Never set candidate rows to `accepted`, validation rows to `passed`,
   gate JSON to `passed`, or drop `tblClockExperimentPolicy`. Never edit `_clock_supported_interpolation`
   or `review_gate` to be more permissive. Do not run anything in `output/_quarantine_scripts/`.
7. **No threshold, eps, window, anchor, or epoch-count change** without the owner's approval and a
   before/after metric. Record every parameter change and its rationale in the journal.
8. **No new script files.** Run analysis as one-off snippets (see Appendix). Output CSV/JSON/PNG is fine
   under a new folder `output/11072026_recovery/`.
9. **Ask before deleting anything.** **The owner commits; you do not.** Branch: `ENM_jsat3d_edits`.
10. **Journal each session** in `.ai_journal/session/YYYY-MM-DD_topic.md`: files touched, commands, results,
    decisions, parameter changes (none unless approved), blockers, next steps.
11. Env: `conda run -n jsat_3d ...`. Tests: `conda run -n jsat_3d python -m unittest discover -s tests`.
    Exit code 1 from `conda run` is normal; read the `Ran N tests ... OK` line. `conda run` drops piped stdin.
12. PowerShell breaks multi-line `python -c`. Use the Python-snippet tool or single-line commands.

---

## 2. The PM request (what "done" means)

1. Keep internal-code jump identification + TDOA magnitude/direction. **Done (stage 1).**
2. Add a supplementary method for jumps the codes miss (DBSCAN suggested; "how many clusters?"). **Implemented (stage 2).**
   Answer for the PM: DBSCAN has no cluster count to choose. With the fixed physical eps (0.5 ms × 2.5 beacon
   periods, min_samples 3), each real clock step starts a new cluster; the count equals the number of stable
   clock regimes in the data. Tuning a count per receiver would be hidden parameter tuning.
3. No iterative for-loop correction (PM withdrew it). Stage 2 uses ≤4 bounded re-measure passes; do not extend.
4. **Compare Arm A (original) vs Arm B (new), especially TDOA plots, to show no degradation. NOT YET DONE for the current Arm B build.**

---

## 3. Verified facts (checked 2026-11-07 by queries on the real databases)

**Roles (unchanged between arms):** ZOI08 = master/beacon host (metronome, tag `7DB7`); ZOI03 = DBSCAN anchor.
`scripts/beacon_pairwise_dbscan.py`, `legacy_pipeline.py`, `adapt_2025_to_legacy.py` have no changes vs HEAD.

**Databases:** Arm A = `output/10292026_Test/...db`. Arm B = `output/11062026_clocksync/jsats3d_2025_earlyfish.db`
(compact, 8 tags, 2,349 applied jumps: 696 stage 1 + 1,653 stage 2). Run file for Arm B:
`config/run_data_earlyfish_2026-11-03_clocksync.toml`. Old Arm B folders from 11-03/11-04 are gone from disk.
Free disk ≈ 44 GiB (it fluctuates; recheck before any DB copy or rerun).

**Arm B gate (real): blocked**, 1,652 rows, all 9 solving receivers. Gate inputs untouched: candidates
1,425 accepted / 1,173 unresolved; validation 493 passed / 2 failed / 1,805 unverifiable; policy row present;
no `tblPositions*` tables. No fish step has ever legitimately run on Arm B.

**Tests:** full suite = `Ran 69 tests ... OK`.

### 3.1 What is NOT broken (do not "fix" these)
- **The >120 s final clock-knot gaps exist in Arm A too and are slightly worse there.** Fish rows in gaps >2 periods,
  Arm A → Arm B: ZOI01 15,035 → 13,749; ZOI02 14,807 → 13,026; ZOI03 11,031 → 9,287; ZOI05 14,558 → 13,381;
  ZOI06 5,494 → 5,493; ZOI07 17,843 → 17,241; ZOI09 5,904 → 5,931; ZOI10 9,983 → 9,708; CFD04 11,561 → 11,483.
  Median knot spacing identical (63.6 s). The legacy fish path bridged these gaps silently with `np.interp`.
  The strict gate only makes a long-standing property visible. It is **not** a regression from the jump work.
- The metronome, anchor, and pairwise DBSCAN code are unchanged. Do not touch them.
- The gate's 8 curve-coverage rows are caused by those >120 s gaps, not by the 50 ms step rule
  (0 fish rows fail on the 50 ms step test alone) and not by temperature.
- Stage 1 (internal codes) is largely correct: 96% of measured events agree with raw TDOA within 50 ms.

### 3.2 What IS wrong (this is your work)
1. **Fabricated temperatures in Arm B.** `tblInterpolatedTemp` was rebuilt on 11-06 from the 4-depth string only,
   then `reindex().ffill().bfill()`: **576 rows (2025-06-15 00:00 → 06-16 23:55) are a constant back-filled 13.924 °C**
   labeled `DD_N_string`; the string starts 06-17. Real HOBO data exists from 06-02. Versus Arm A's recorded
   `avg_C` (CFD01 clock-fix epochs): before 06-17 mean +1.26 °C, max +2.41 °C; 06-17→07-10 (HOBO vs string)
   mean +0.03, max 0.32 °C; after 07-10 identical. A 1 °C error ≈ 3–4 m/s sound speed.
2. **Core change that affects every run (uncommitted):** `temp_interpolator` in `jsats3d/jsats3d.py` changed
   `bounds_error=True` → `False`, so times outside the temperature table silently become NaN instead of raising.
   It was added to cope with the bad temperature table. Violates "fail loudly".
3. **Arm A was modified** by an earlier session: a string-only `tblInterpolatedTemp` (26,662 rows from 06-17) was
   added to Arm A. Arm A's saved positions/clock curves predate it (computed with a HOBO+string table from 06-02),
   so results are unchanged, but the DB is not pristine. Do not write to it again.
4. **Stage-2 ledger errors.** A valid audit (Appendix B) of 2,148 measurable listener events: 85% agree within 50 ms
   (77% within 0.5 ms). Stage 1: 96.3% agree. **Stage 2: 79.7% agree (ZOI01/02/03/05/07: 147 of 690 mismatch = 21%;
   27% where `boundary_uncertain=1`).** Mismatch errors are quantized (±1.0 s, ±0.5 s, ±2.0 s); 88% sit at unstable
   plateaus (5-epoch spread > 0.5 s) vs 31% of matches — consistent with ±1 s per-record firmware toggling. Offsets
   are cumulative, so a few wrong ±1 s steps accumulate: ZOI01 net applied offset is −11.5 s from 06-27 on while its
   raw TDOA stays flat (~0.03 s). 41 of 49 master (ZOI08) stage-2 corrections are mirrored on listeners, i.e. real.
5. **Leftover scripts: resolved.** The 43 root-level throwaway scripts are in `output/_quarantine_scripts/`; the 16 untracked
   v4-experiment scripts from `scripts/` were moved to `output/_quarantine_scripts/scripts_v4_experiment/` (2026-11-07; verified
   untracked, unreferenced, 69 tests still pass). Never run anything in the quarantine folder. Never delete it without asking.

### 3.3 A trap that already misled a previous session
`TDoA - TDoA_error` in `receiver_*_clock_fix.csv` **equals `TDoA_expected`** (geometry/sound-speed expectation;
checked: max difference 1e-6 s; value 0.0266 s, std 0.0001 s). It is flat by construction and is NOT raw TDOA.
Any audit built on it is invalid. Raw TDOA = a listener's first-arrival time minus the nearest ZOI08 `7DB7`
first-arrival time, both in **original** receiver time (`seconds + coalesce(jump_offset_s, 0)`).

---

## 4. Task list

### Task 1 — Orientation and baseline (read-only)
1. Read, in order: this file; `System_Prompt.txt` sections 1, 2, 9, 11; the last three sections of
   `.ai_journal/session/2026-11-07_gate_tamper_restore_step1.md` (headings "Curve-coverage trace", "Ledger audit",
   "Is the code broken?"); `docs/llm_handoff_clock_jumps_2026-11-06.md` sections 3–5.
2. Run the full test suite → expect `Ran 69 tests ... OK`.
3. `git status --short` → confirm modified files are only `LONG_TERM_CONTEXT.md, jsats3d/jsats3d.py,
   scripts/_2025_pipeline.py, scripts/clock_jump_pre.py, scripts/run_data.py, tests/test_clock_jump_pre.py,
   tests/test_smoke.py` plus untracked journals/docs/config. There should be **no untracked files in `scripts/`**. Report anything else.
4. Check free disk space on C:.
5. Confirm gate inputs are intact (Appendix D one-liner) and still blocked.
**STOP. Report: test result, git state, disk space, gate counts.**

### Task 2 — Fix Arm B temperature (owner has delegated this; it needs a clock rerun)
Why: sound speed depends on temperature; fabricated 06-15/16 values bias range terms in the clock fix.
1. **Backup first:** copy the Arm B DB to `output/11062026_clocksync/jsats3d_2025_earlyfish.pre_tempfix.db`
   (≈4.2 GB; check disk first). Do not delete the backup.
2. Build the correct table **in memory** with the project loader, no filling beyond coverage:
   `adapt_2025_to_legacy.load_temperature_string(temperature_csv, hobo_dir)` using the two paths in the run TOML
   (`[paths] temperature_csv` and `hobo_dir`). Expected columns: `timeStamp, C, TempSource, DepthCount`;
   HOBO rows from 2025-06-02, string rows after the last complete HOBO step.
3. **Verify before writing** (Appendix C): compare against Arm A's recorded `avg_C` in
   `output/10292026_Test/jsats3d_2025_earlyfish_legacy/figures/receiver_CFD01_clock_fix.csv`. Expected after the fix:
   |ΔC| ≈ 0 (well under 0.01 °C) in all three periods (before 06-17, 06-17→07-10, after 07-10). If not, STOP and report.
4. Write only that table into Arm B: `DROP TABLE tblInterpolatedTemp` then `to_sql(..., if_exists="replace")` on a normal
   (non read-only) connection to Arm B only. Confirm row count, min/max timestamp, and no NaN in `C`.
5. Force a rebuild of everything derived from temperature. The pipeline skips a step whenever its `*_done()` check finds
   output. Read `metronome_done`, `clock_surface_done`, `deep_beacons_done`, `deep_positions_done`, `deep_clocks_done`
   and `_invalidate_clock_outputs` in `scripts/_2025_pipeline.py` and `DERIVED_TABLES` in `scripts/legacy_pipeline.py`.
   Reset the progress rows and drop the derived tables those checks look at (the stage-1/2 jump ledger and
   `jump_offset_s` must NOT be touched — they do not use temperature). If the reset is unclear, STOP and ask.
6. Rerun: `python scripts/_2025_pipeline.py config/run_data_earlyfish_2026-11-03_clocksync.toml --from metronome --to deep_clocks`
   (long; run it so the user can watch it; check disk first). Never use `--init-from`, `run_data.py` rebuild, or `--compact-from` again.
7. Re-run the gate through the pipeline function path (Appendix D). **It is expected to still block** (the >120 s gaps are
   a legacy property). Record new counts per reason. Repeat Appendix A to compare gap rows with Arm A.
**STOP. Report: ΔC table, new gate counts, new gap-row counts vs Arm A.**

### Task 3 — Revert the silent-NaN temperature change
1. In `jsats3d/jsats3d.py::temp_interpolator`, change `bounds_error = False` back to `bounds_error = True`
   (the original behavior in HEAD). Leave the NaN-drop guards in `clock_fix` alone (harmless when temperature covers the data).
2. Run the full suite (`Ran 69 tests ... OK`) and a quick run of the affected step to confirm no `ValueError` from
   out-of-range temperatures. If a legitimate time falls outside the table, report it; do not re-widen the interpolator.
3. Journal it as a parameter/behavior change with rationale: restore fail-loud behavior.
**STOP. Report: diff, test result, step result.**

### Task 4 — PM deliverable: Arm A vs Arm B TDOA plots (read-only; PM item 4)
1. Generate with the existing pipeline review step, three intervals (each its own run):
   `python scripts/_2025_pipeline.py config/run_data_earlyfish_2026-11-03_clocksync.toml --only clock_review --clock-review-interval "2025-06-27 06:00:00" "2025-06-30 06:00:00"`
   Pick two more intervals with heavy fish presence from `output/11062026_clocksync/clock_review/clock_review_gate.csv`
   (largest `raw_fish_rows`; one ZOI06 window, one ZOI01 window). Output lands in `clock_review/interval_<start>_<end>/`
   (`tdoa_arms_<receiver>.png` plus CSVs; Arm A "unshifted" vs Arm B "tdoa_supplement").
2. Build one table per interval: per receiver, count of adjacent TDOA changes ≥ 50 ms, Arm A vs Arm B.
3. **Open and look at every PNG.** Write what you see. Any receiver that is worse in Arm B → list it and investigate before moving on.
**STOP. Report the table and the receivers that got worse (if any).**

### Task 5 — Stage-2 ledger audit report (read-only; evidence for an owner decision)
1. Run the valid audit (Appendix B) on all receivers; save per-event results to
   `output/11072026_recovery/ledger_audit_events.csv` (rec, stage, jump time, applied offset, measured raw step,
   plateau spreads, boundary_uncertain, agree flags) and a summary table by receiver/stage.
2. Add: how many applied stage-2 offsets sit at unstable plateaus (spread > 0.5 s) or `boundary_uncertain=1`;
   how many applied offsets have no measurable raw step; per-receiver accumulated error.
3. **Do not change how offsets are applied.** Whether to stop applying stage-2 offsets at unstable plateaus is the owner's
   decision and needs an A/B metric. Prepare the numbers only.
**STOP. Report the summary table.**

### Task 6 — Decisions: ALREADY MADE (conservative defaults). Do NOT ask the user these again.
The user said they do not know the answers and told the assistant to pick the safest option. These are **provisional**:
they are safe defaults, not scientific sign-off. Kevin Nebiolo (project owner) / the PM must confirm before any of them is
relaxed. Your job in this task is to (a) follow them, and (b) write the short message to Kevin in item (b) below.

| # | Question | Decision (safest) | Consequence for you |
|---|---|---|---|
| 1 | Apply stage-2 offsets at unstable-plateau (spread > 0.5 s) or `boundary_uncertain=1` events? | **Make no change to what is applied.** Do not edit the ledger or the application code. The gate already blocks any fish use of these events, so nothing downstream consumes them. Evaluate only as a **what-if on an isolated copy** if the user later asks. | Task 5 produces numbers only. |
| 2 | Apply certified sub-50 ms steps (e.g. the 0.907 ms ZOI06 step)? | **Flag only.** Report them as timing warnings. Do not apply them and do not lower `clock_jump_min_step_s` (stays 0.05). | No code change. |
| 3 | Gate mode: block-all vs exclude-and-flag? | **Keep block-all.** Do not implement exclude-and-flag now. | Fish steps stay blocked. Do not look for a way around it. |
| 4 | Metronome anchor | **Keep ZOI03** (identical to Arm A). | No change. |
| 5 | Provenance guard (3 ZOI06 windows) | **Keep strict as written.** | No change. |
| 6 | Untracked scripts in `scripts/` | **Quarantine (move, never delete).** Already done 2026-11-07: 16 files in `output/_quarantine_scripts/scripts_v4_experiment/`. `scripts/` has no untracked scripts left. | None. Never run them. |

(b) **Message for Kevin / the PM** — draft it in the journal, plain language, with numbers, and give it to the user to send:
- Fish positions for Arm B are blocked by design: 1,652 gate rows, all 9 solving receivers.
- The main blocker (8 rows, ~88,900 fish rows) is >120 s gaps between clock knots. **Arm A has the same gaps** (e.g. ZOI01 15,035 vs 13,749
  fish rows); the original pipeline bridged them silently by interpolation. So "no degradation" cannot be shown by fish positions
  until Kevin decides how gaps should be treated.
- Question for Kevin: should an A/B fish comparison use a matched exclusion (the same fish detections inside unsupported clock
  spans removed from both arms), clearly labeled diagnostic and not production? This needs a small code change plus tests.
- Question for Kevin: should supplement (stage 2) offsets at unstable plateaus (21% mismatch with raw TDOA on ZOI01/02/03/05/07)
  be applied, or reported as unresolved? A what-if run on a copy can quantify the effect first.
- Question for Kevin: should certified sub-50 ms steps (0.5-50 ms; 1 ms ≈ 1 m) be corrected?
**STOP. Report that decisions are recorded and the message is drafted.**

### Task 7 — REPLACED by `docs/llm_handoff_qwen36_35b_moe_POSITIONS_2026-11-07.md`
The project lead (user) instructed on 2026-11-07 to produce positions and per-tag figures soon, by the safest route and without
damaging existing positions. Follow the POSITIONS file: plot Arm A (validated baseline) now, and produce Arm B as a labeled
**diagnostic on an isolated copy**. The old Task 7 text below is kept only for the A/B recipe.

### (old) Task 7 — Only after Task 6: implement approved changes (tests first), then clean fish run and A/B
- Fish steps only through the pipeline main loop (it runs the gate): `--from fish_multipath --to positions_dbscan`.
  Never call step functions directly to skip the gate. Never copy Arm A filter tables into Arm B.
- A/B recipe (see `journals/2026-11-04_ab_fish_validation_clocksync.md`): match transmissions **by time** (nearest ≤ 1 s),
  never by `transNo`; raw root-B validity = median within 2 km of array centroid (467.0, 113.3 m); "affected" = within ±300 s
  of an applied jump; report rescued / broke / no-impact; curve steps at applied jump times; 2–3 fish track plots around
  jumps; yield accounting (detections in → positions out). Any "broke" case outside a documented window is a blocking defect.
- Tell the PM plainly: clock error is not the dominant accuracy limit (per-ping geometry/multipath scatter σ ≈ 9 m,
  journal 2026-10-11). Clock completeness buys yield and removes gross errors (e.g. 548 km → 20 m), not centimetre accuracy.
**STOP. Draft the PM report for owner review. No PR while any gate is blocked or unreviewed.**

---

## 5. Appendix — paste-ready snippets (run with the Python-snippet tool; workspace root `c:\Dev\jsat3d\jsats3d`)

### A. Fish rows falling in clock-knot gaps (compare arms)
```python
import sqlite3, numpy as np, pandas as pd
from pathlib import Path
ro = lambda p: sqlite3.connect(Path(p).resolve().as_uri() + '?mode=ro', uri=True)
A = ro('output/10292026_Test/jsats3d_2025_earlyfish.db'); B = ro('output/11062026_clocksync/jsats3d_2025_earlyfish.db')
tags = ('26E4', '9F8C', '1D5A', 'B5ED'); P = 60.0; rows = []
for rec in ['ZOI01','ZOI02','ZOI03','ZOI05','ZOI06','ZOI07','ZOI09','ZOI10']:
    r = {'rec': rec}
    for n, c in (('A', A), ('B', B)):
        k = pd.read_sql("SELECT seconds FROM tblDetectionClockFixed WHERE Rec_ID=? ORDER BY seconds", c, params=[rec]).seconds.drop_duplicates().to_numpy(float)
        f = np.array([x[0] for x in c.execute("SELECT seconds FROM tblDetectionRaw WHERE Rec_ID=? AND Tag_ID IN (?,?,?,?)", (rec, *tags))], float)
        right = np.clip(np.searchsorted(k, f), 1, len(k) - 1)
        r[n + '_fish_in_gaps'] = int(((f < k[0]) | (f > k[-1]) | ((k[right] - k[right-1]) > 2*P)).sum()); r[n + '_knots'] = len(k)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
```

### B. Valid ledger audit (applied offset vs measured raw TDOA step, original time)
```python
import sqlite3, numpy as np, pandas as pd
from pathlib import Path
b = sqlite3.connect(Path('output/11062026_clocksync/jsats3d_2025_earlyfish.db').resolve().as_uri() + '?mode=ro', uri=True)
P = 60.0
def load(rec):
    d = pd.read_sql("SELECT seconds+coalesce(jump_offset_s,0) AS o FROM tblDetectionRaw WHERE Rec_ID=? AND Tag_ID='7DB7'", b, params=[rec]).sort_values('o')
    g = d.o.diff(); return d[g.isna() | (g >= 0.5*P)].o.to_numpy()
mo = load('ZOI08')
led = pd.read_sql("SELECT Rec_ID, stage, jump_time_s, offset_s, window_start_s, window_end_s, boundary_uncertain FROM tblClockJumpEvents", b)
rows = []
for rec in sorted(set(led.Rec_ID) - {'ZOI08'}):
    xo = load(rec); i = np.searchsorted(mo, xo); l = np.clip(i-1, 0, len(mo)-1); r = np.clip(i, 0, len(mo)-1)
    pick = np.where(np.abs(xo-mo[l]) <= np.abs(xo-mo[r]), l, r); raw = xo - mo[pick]; ok = np.abs(raw) < 0.5*P; t, v = xo[ok], raw[ok]
    for e in led[led.Rec_ID == rec].itertuples():
        lo = e.window_start_s if pd.notna(e.window_start_s) else e.jump_time_s; hi = e.window_end_s if pd.notna(e.window_end_s) else e.jump_time_s
        pi = np.flatnonzero(t < lo)[-5:]; po = np.flatnonzero(t > hi)[:5]
        if len(pi) < 3 or len(po) < 3 or t[pi[0]] < lo-900 or t[po[-1]] > hi+900:
            rows.append(dict(rec=rec, stage=e.stage, t=e.jump_time_s, offset=e.offset_s, raw_step=np.nan, unc=e.boundary_uncertain)); continue
        rows.append(dict(rec=rec, stage=e.stage, t=e.jump_time_s, offset=e.offset_s, raw_step=float(np.median(v[po]) - np.median(v[pi])),
                         pre_spread=float(np.ptp(v[pi])), post_spread=float(np.ptp(v[po])), unc=e.boundary_uncertain))
m = pd.DataFrame(rows); k = m.dropna(subset=['raw_step']).copy(); k['err'] = (k.offset - k.raw_step).abs()
k['agree_50ms'] = k.err <= 0.05; k['agree_0p5ms'] = k.err <= 0.0005
print(k.groupby(['stage']).agg(n=('err', 'size'), agree_50ms=('agree_50ms', 'sum'), agree_0p5ms=('agree_0p5ms', 'sum')).to_string())
print(k.groupby('rec').agg(n=('err', 'size'), agree_50ms=('agree_50ms', 'sum')).to_string())
```
Expected on the current Arm B: 2,148 measurable; 1,829 agree ≤ 50 ms; 1,656 agree ≤ 0.5 ms (stage 1: 672/696, stage 2: 1,157/1,452).
Note: after Task 2 the ledger itself does not change (temperature is not used by stages 1–2), so these numbers should stay the same.

### C. Temperature table vs Arm A's recorded `avg_C`
```python
import sqlite3, numpy as np, pandas as pd
from pathlib import Path
d = pd.read_csv('output/10292026_Test/jsats3d_2025_earlyfish_legacy/figures/receiver_CFD01_clock_fix.csv', usecols=['seconds', 'avg_C']).dropna()
b = sqlite3.connect(Path('output/11062026_clocksync/jsats3d_2025_earlyfish.db').resolve().as_uri() + '?mode=ro', uri=True)
t = pd.read_sql('SELECT timeStamp, C FROM tblInterpolatedTemp', b)
ts = pd.to_datetime(t.timeStamp).astype('datetime64[ms]').astype('int64') / 1000.0
d['dC'] = np.interp(d.seconds, ts, t.C) - d.avg_C
e1, e2 = 1750118400, 1752152400   # 2025-06-17 00:00 and 2025-07-10 ~13:00, study wall-clock as epoch
d['period'] = np.where(d.seconds < e1, '1_before_0617', np.where(d.seconds < e2, '2_0617_to_0710', '3_after_0710'))
print(d.groupby('period').dC.agg(['count', 'mean', 'min', 'max', lambda s: s.abs().quantile(.95)]).round(4).to_string())
print('rows', len(t), t.timeStamp.min(), t.timeStamp.max(), 'NaN in C:', int(t.C.isna().sum()))
```
Before the fix this prints mean +1.257 / +0.031 / 0.000. After the fix all three means must be ≈ 0.

### D. Gate state (read-only) and re-running the real gate
```python
import sqlite3, json, pandas as pd
from pathlib import Path
b = sqlite3.connect(Path('output/11062026_clocksync/jsats3d_2025_earlyfish.db').resolve().as_uri() + '?mode=ro', uri=True)
print(b.execute('SELECT status,count(1) FROM tblClockJumpCandidates GROUP BY status').fetchall())    # 1425 accepted / 1173 unresolved
print(b.execute('SELECT status,count(1) FROM tblClockJumpValidation GROUP BY status').fetchall())   # changes after Task 2 rerun
print(json.loads(Path('output/11062026_clocksync/clock_review/clock_review_gate.json').read_text()))
g = pd.read_csv('output/11062026_clocksync/clock_review/clock_review_gate.csv'); g['kind'] = g.reason.str.slice(0, 40)
print(g.groupby('kind').agg(rows=('Rec_ID', 'size'), fish=('raw_fish_rows', 'sum')).to_string())
```
To regenerate the gate files use the real function (it raises when blocked — that is correct; catch nothing, just read the CSV it wrote):
`conda run -n jsat_3d python -c "import sys; sys.path.insert(0,'scripts'); from _2025_pipeline import review_gate; from clock_jump_pre import naive_seconds; review_gate('output/11062026_clocksync/jsats3d_2025_earlyfish.db', ['26E4','9F8C','1D5A','B5ED'], ['ZOI01','ZOI02','ZOI03','ZOI05','ZOI06','ZOI07','ZOI08','ZOI09','ZOI10'], naive_seconds('2025-06-15 00:00:00'), naive_seconds('2025-08-06 23:59:59'), 'output/11062026_clocksync/clock_review', check_curves=True)"`

---

## 6. Known traps
- `"26E4"` parses as a float in pandas → use `dtype={"tag": str}` when reading tag columns.
- `a.loc[i] - b.loc[j]` aligns on labels → NaN. Use `.to_numpy()`.
- Use explicit `datetime64[ms]` for epoch conversion; pandas 2.x unit changes silently break ns/us math.
- `row.get(key, row[other])` evaluates the fallback eagerly → KeyError. Use an explicit `if key in row`.
- `clock_rapid_reference_review` overwrites `<rec>_source_state_boundaries.csv` in its folder → give each run its own subfolder.
- `DROP TABLE IF EXISTS x` can resolve into an ATTACHed read-only DB → qualify `main.x`.
- Study time basis = PDT wall clock stored as if UTC. ZOI05 logs UTC (already normalized in current builds). ZOI05/ZOI06 hydrophone
  coordinates are already swapped in current builds — never swap again. ZOI09 carries a +0.946 s hardware offset handled in `jsats3d.py`.
- C: free space swings (OneDrive/Egnyte caches). Check before any DB write or copy. A test once failed with `database or disk is full` and passed on retry.
- Report only numbers you reproduced this session from a file or query. "Everything verified and complete" is not an acceptable summary
  while the gate is blocked and fish cannot run.

## 7. File map
| What | Where |
|---|---|
| Stage 1/2, consensus, firmware decode | `scripts/clock_jump_pre.py` |
| Pipeline steps, gate, reviews | `scripts/_2025_pipeline.py` |
| clock_fix, fish loader guards, `temp_interpolator` | `jsats3d/jsats3d.py` |
| Temperature loader | `scripts/adapt_2025_to_legacy.py::load_temperature_string` |
| Fixed DBSCAN (do not tune) | `scripts/beacon_pairwise_dbscan.py` |
| Arm B run file | `config/run_data_earlyfish_2026-11-03_clocksync.toml` |
| Tests | `tests/test_clock_jump_pre.py`, `tests/test_clock_study_gate.py`, `tests/test_clock_segments.py`, `tests/test_smoke.py` |
| Findings and decisions | `.ai_journal/session/2026-11-06_*.md`, `2026-11-07_gate_tamper_restore_step1.md`, `LONG_TERM_CONTEXT.md` |
| Quarantined scripts (never run) | `output/_quarantine_scripts/` |
