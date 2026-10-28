# 2025 Deng Pipeline Performance — Implementation Instructions for Qwen 3.6 MoE (2026-10-26, v2)

v2 replaces the 2026-10-26 v1 plan. The project lead added two hard
constraints after v1: **(1) the speed-ups must not change how the current
pipeline works, and (2) the 2019 data must still run exactly as the legacy
code does.** v1 phases that touched shared legacy code beyond `Deng()`
(clock_fix dead code, `widen_table`) are withdrawn. v2 scope is minimal:
`position.Deng()` internals only, plus read-only/working-copy database
preparation.

v2.1 (2026-10-26, same day): 2019 baseline made concrete (data confirmed
present: 7.77 GB raw, 316 study tags + 9 beacons; staged run procedure;
tiered 2019 gates), 0.0 disk pre-flight added, four dead per-transmission
pandas assignments identified (skipped in the fast path only), Section 9
session decisions appended. No code changed; this is still an instruction
set for the Qwen 3.6 MoE session.

**Read first:** `System_Prompt.txt` (operating rules), `LONG_TERM_CONTEXT.md`,
`docs/accuracy_tuning_handoff_2026-10-21.md` (F1/F2 context, T2/T3 targets).

Conda env: `jsat_3d`. Work from the repo root `C:\Dev\jsat3d\jsats3d`.
All line numbers refer to `jsats3d/jsats3d.py` at baseline commit `8ca7ba4`
(branch `ENM_jsat3d_edits`, clean tree).

---

## 0. HARD CONSTRAINTS (every instruction below enforces these)

**HC-1 — Behavior preservation.** The pipeline must produce *identical*
results: same rows, same values, same row order, same comment strings, same
CSV bytes, same tables, same derived quantities. What counts as a change:
any difference in any output byte, any difference in which rows exist, any
difference in a value at the least significant bit, any dropped or reordered
row. Extra log lines are allowed. Fewer/different log lines are not.

**HC-2 — 2019 parity.** After every code change, the 2019 workflow
(`config/run_data_2019.toml`) must still run end-to-end and produce artifacts
byte-identical to the baseline run of commit `8ca7ba4`. The 2019 gate is
mandatory for any edit to `jsats3d.py` (the shared core). Note: `Deng()` is
shared — 2019 uses it for deep-receiver beacons (R01-R03) and all study tags.

**HC-3 — Single code path.** No year/DB-conditional branches, no runtime
flags that switch math. The fast path IS the pipeline. Rollback mechanism is
git (one commit per sub-item), not a legacy-mode switch.

**HC-4 — Scope fence.** The only file that may be edited is
`jsats3d/jsats3d.py`, and only the body of `position.Deng()` (lines 1434-1686).
New scratch scripts `scripts/_perf_*.py` are allowed (deleted at final
sign-off). Working-copy database preparation (index on a *copy*) is allowed.
Everything else is out of scope — see Section 8.

**Baseline definition.** "Current behavior" = code at commit `8ca7ba4`. This
includes the F1 clock correction (lines 318-365), which also affects 2019
study-tag runs when `tblDetectionClockFixed` exists. Whether F1 should apply
to 2019 is an existing open owner item (2026-10-21 handoff §6.1) — do NOT
touch it, do NOT "fix" it; the performance baseline simply includes it.

---

## 1. PHASE 0 — Baselines, harness, profile (no code changes)

Everything in Phase 0 is read-only against production data plus new scratch
scripts. Nothing under `jsats3d/` or `scripts/` (existing files) is modified.

### 0.0 Pre-flight (do first, before any other Phase 0 step)

- `git status` must show a clean tree at `8ca7ba4` on `ENM_jsat3d_edits`.
  If dirty, stop and report before doing anything else.
- 2019 data confirmed present (2026-10-26): `data/2019/Synch_1/` (7.77 GB
  raw Teknologic DET, 9 receiver folders R01-R09), `data/2019/tblTag.csv`
  (316 study tags, all `pulseRate=3`; 9 beacons FF74-FF81 + 7DD2),
  `tblReceiver_Initial.csv`, `tblWSEL.csv`, `tblTemp.csv`, and
  `config/run_data_2019.toml`.
- **Disk space — user decision point.** Measured 2026-10-26: C: has
  ~20.8 GB free. The 2019 baseline DB is expected to need ~15-30 GB (exact
  size unknown until import; the import log shows progress) and the Phase 3
  chinook working copy needs ~25 GB (22 GB DB + index). C: alone is not
  enough for both. Measure every local drive
  (`Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3"` — report
  DeviceID, FreeSpace, Size) and present the numbers to the user. K: is the
  network raw-data drive: read-only, never a target. If no local drive has
  >= 40 GB free, STOP and ask the user where the two artifacts live (free
  up space, another drive, or split). Do not guess.
  If the 2019 baseline DB must live off the default
  `output/cowlitz_2019_synch_1_recreated.db`: copy
  `config/run_data_2019.toml` to a NEW file (e.g.
  `config/run_data_2019_baseline.toml`) changing only `output_db`, and run
  the baseline from the copy. Never edit the original toml in place.
- Execution order for Phase 0:
  0.0 -> 0.2a (2019 dry-run + import, foreground) -> 0.2b (2019 process,
  background with log) -> 0.1 (2025 drag baseline, foreground) -> 0.3
  (profile + invariant scan + ingest timing, foreground) -> 0.4 (journal
  gate; 2025 portion as soon as it is ready, 2019 portion when the run
  finishes). The 2019 and 2025 work use separate DBs and may run
  concurrently. **Phase 1 may not start until both baselines exist.**

### 0.1 2025 drag baseline

- DB: `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db` (do not modify; if a
  step needs writes, work on a copy).
- Receivers (from `config/run_data_tagdrag_ent.toml`):
  `receivers_3d = [ZOI01, ZOI02, ZOI03, ZOI05, ZOI06, ZOI07, ZOI08, ZOI09, ZOI10]`
  (10 receivers), tags FC36 and FFD3.
- Create `scripts/_perf_harness.py` with subcommands (design below). Run the
  `baseline2025` subcommand:
  1. For each tag in (FC36, FFD3): `pos = jsats3d.position(tag, receivers_3d,
     db, out_folder, fig_folder); pos.Deng(water_column=False)` — exactly the
     call `legacy_pipeline.deng()` makes. `out_folder` = a fresh scratch dir
     (e.g. `output/perf_baseline_2025/positions`). Record per-tag wall time.
  2. Save byte-exact copies of the 4 CSVs (`FC36_solutionA.csv`,
     `FC36_solutionB.csv`, `FFD3_solutionA.csv`, `FFD3_solutionB.csv`) under
     `output/perf_baseline_2025/`.
  3. Save frame dumps: for each `pos.DengSolutionA_unfiltered` /
     `B_unfiltered`: `df.to_pickle(...)` plus a text dump of
     `df.dtypes` and `repr(df.head(5))`. (Pickle of the full frame is the
     comparison artifact.)
  4. ENT scoring: make a copy of the DB, load the baseline CSVs into it via
     the same path the pipeline uses (`lp.widen_table` +
     `jsats3d.positions_data_management("Deng", staging, db)` — the legacy
     loader deletes its input, so stage copies), then run
     `python scripts/ent_analysis_2025.py --db <copy> --only 12 --out
     perf_baseline_2025 --lines ENT-01 ENT-02 ENT-03 ENT-04 ENT-05 --table
     tblPositions_Deng`. Save the summary CSVs it writes.
     **Expected (T2, 2026-10-21): 3D root-B Kalman weighted median ~4.07 m,
     scored 459.** (2D/Deng2D numbers are sanity only — `Deng2D` is out of
     scope and cannot change.)
- Record: per-tag Deng wall time, total, CSV hashes (SHA-256), ENT summary.

### 0.2 2019 baseline (one-time; mandatory before any Phase 1 commit)

Scale (from the data, 2026-10-26): 316 study tags (all `pulseRate=3`) +
9 beacons, 9 receivers (surface R04-R09, deep R01-R03), sync window
2019-04-19 14:05:22 to 2019-05-24 12:25:00 (~35 days), 7.77 GB raw. On the
current (unoptimized) code this is a **multi-hour to multi-day** job:
deep-receiver Deng alone runs ~2 h per receiver (2026-10-02 journal), plus
two metronome/clock-fix phases and per-tag multipath + Deng for all 316
study tags. Plan for it to run unattended; do not block 0.1/0.3 on it
(separate DBs).

Staged procedure (each stage observed before the next starts):

1. **Dry run (seconds):**
   `python scripts/run_data.py config/run_data_2019.toml --dry-run`
   — validates the run file and prints the step plan. (Use the baseline
   toml copy if 0.0 moved `output_db`.)
2. **Import (foreground, watch it; ~1-3 h for 7.8 GB):**
   `python scripts/legacy_pipeline.py import-2019 config/run_data_2019.toml`
   — creates the DB (`tblDetectionRaw`, `tblTag`, `tblReceiver`,
   `tblWSEL`, `tblInterpolatedTemp`) and prints "Imported receiver R0X"
   per receiver. All 9 must import without errors. If it fails: journal
   the error, change nothing under `jsats3d/`, report to the user.
3. **Full process (background with log; the long part):**
   `python scripts/run_data.py config/run_data_2019.toml --skip-build`
   — reuses the imported DB, applies [study] parameters + indexes, then
   runs the legacy workflow: metronome -> clock fix -> deep receivers ->
   phase 2 -> 316 study tags -> export. Per System Prompt 16 the user
   launches and observes it from their own terminal, e.g.:
   `python scripts/run_data.py config/run_data_2019.toml --skip-build *> output/perf_2019_baseline.log`
   (PowerShell `*>` redirects all streams). The implementing session may
   background it only with explicit user OK.
   - **Health checkpoints before leaving it unattended:** the log shows
     `Metronome: beacon ...` completing, `Clock fix: receiver R04`
     finishing, and the first deep-receiver step starting.
   - `run_data.py` writes `output/cowlitz_2019_synch_1_recreated.run.json`
     (per-step wall times + git commit) — keep it as the wall-time record.
   - **Resume after a crash:** `run_data.py` does not pass `--resume`, so
     resume directly with
     `python scripts/legacy_pipeline.py process config/run_data_2019.toml --resume`
     (skips finished tags via `tblProcessProgress`; if phases 1-2 are
     incomplete it restarts the phases).
   - **Partial-DB gotcha:** the toml has `overwrite = false`; a partial DB
     from a failed attempt blocks a fresh run. To restart clean, delete
     the partial `output/cowlitz_2019_synch_1_recreated.db` (+ `.run.json`
     + work folder) and journal the deletion.

Save (under `output/perf_baseline_2019/`):
  1. All position CSVs from the work folder
     (`output/cowlitz_2019_synch_1_recreated_legacy/positions/`,
     `*_solutionA.csv`, `*_solutionB.csv`) byte-exact, plus a manifest:
     - deep beacons R01, R02, R03: positioned from the **surface** set
       `[R04, R05, R06, R07, R08, R09]` (see `position_deep_receivers` in
       `legacy_pipeline.py`);
     - every study tag: positioned from **all 9** receivers. Save the exact
       tag list in run order:
       `SELECT Tag_ID FROM tblTag WHERE TagType = 'study' AND pulseRate IS NOT NULL`
       against the baseline DB (the same auto-selection `process()` uses;
       all 316 qualify).
     - Per tag, record which comment strings appear in its CSVs ('solution
       found', 'singular matrix encountered - no soluiton', 'negative
       quadratic discriminant - no solution', 'negative time of arrival -
       no soluiton', 'not enough receivers for solution') — used for the
       Phase 1 subset-gate selection (Section 5).
  2. Dump of `tblPositions_Deng` (pickled frame + dtypes).
  3. Wall time per step (from `.run.json` + log step headers).
- Do not delete the 2019 DB or work folder — the fast 2019 gate in Phase 1
  re-runs only the Deng step against it (all derived tables are already
  populated).
- If the run fails or produces anomalies (crash, zero solutions everywhere,
  suspicious F1 interaction), STOP: journal it and escalate. Do not "fix"
  2019 behavior inside this work (HC-4).

### 0.3 Profile + data-invariant scan

- `cProfile` one drag-tag Deng run (FC36) on the drag DB; save the profile.
  Record the split between: per-combo pandas work, matrix inverses, position
  lookups, temperature interpolation, row construction, `to_numeric`, hull.
- Ingest timing (read-only SELECTs against production DBs are fine):
  - Pre-check: `PRAGMA index_list(tblDetectionRaw)` on
    `output/jsats3d_2025_chinook.db`. The standard `run_data.py` build adds
    four indexes (`idx_raw_tag_rec`, `idx_raw_rec`, `idx_raw_tag_rec_seconds`,
    `idx_raw_tag_seconds`). If the production DB already has them, the
    "no indexes" assumption is stale — report it; Phase 3 may be moot for
    ingest (measure, then decide with the user).
  - `SELECT * FROM tblDetectionRaw WHERE Tag_ID = 'FC36'` on
    `output/jsats3d_2025_chinook.db` as-is (measure seconds).
  - Repeat on a scratch copy with the standard four indexes (measure
    seconds).
  - Verify on 3 tags that the indexed read returns **identical rows in
    identical order** to the as-is read (compare frames). If order ever
    differs, Phase 3 is blocked (escalate; do not ship an order-changing
    "optimization").
- Tie/duplicate scan (determines whether the Phase 1 guard ever fires): for
  the drag DB and the 2019 DB, over all study-tag + deep-beacon rows in
  `tblDetectionFilterSecondary` that `position` would read: per transmission
  (Tag_ID, transNo), report (a) count where a Rec_ID appears more than once,
  (b) count where two rows share the same `seconds_fix`. Expected: 0 and 0.
  Report actuals in the journal.

### 0.4 Phase 0 gate

Journal entry with: baseline hashes, wall times, profile summary, invariant
scan results, ENT baseline numbers. No commits (harness scripts stay
uncommitted or committed as `scripts/_perf_harness.py` — decide with the user;
campaign convention is to delete `_perf_*`/`_tune_*` scripts after use, but
this one must survive until Phase 2 sign-off).

---

## 2. PHASE 1 — `Deng()` micro-optimization (the only code edit)

One file: `jsats3d/jsats3d.py`. One function: `position.Deng()`.
Implement as sub-items S1..S5, each its own commit, each gated before the
next starts. After each sub-item: `python -m py_compile jsats3d/jsats3d.py`,
then the two gates of Section 5.

### 2.1 What stays byte-for-byte (do not touch inside `Deng()`)

- Lines 1434-1445: signature, `water_column` comment, `point_in_hull`,
  `Solution_Cols`.
- Lines 1447-1465: `SolutionA/B` init, `pending`/`chunks`, `add_row`
  **except** the one-line extension in S4 below, `collect_rows`, `tracks`,
  `z_by_rec`.
- Lines 1466-1471: `tag_data` sort, `by_trans`, `no_rows`, `tSteps`, loop
  header, `tDat` slice.
- Line 1486: `tDat.sort_values(by='seconds_fix')` — kept in **both**
  branches (the fast path's index order IS this time order).
- Lines 1472-1473 (`timeStampFix` / `timeStampOriginal`) and 1487-1488
  (`rank`, `set_index`): **dead work — skip in the fast path, keep verbatim
  in the legacy branch.** The columns are never read anywhere downstream
  (tDat is a per-transmission copy from fancy `.iloc`). This is not a
  micro-optimism: `pd.to_datetime` runs twice per transmission; for a deep
  beacon with ~80k transmissions it is a large fraction of the current
  ~2 h per-receiver time (2026-10-02 journal). Skipping them in the fast
  path changes no observable output (HC-1: outputs identical; only wasted
  work dropped). In the legacy fallback branch they are assigned in the same
  relative order as today (sort -> timeStamp* -> rank -> set_index ->
  combos), so that path stays byte-identical to today.
- Lines 1647-1652: the `else` branch ("not enough receivers" row) — keep the
  current per-row `pd.DataFrame` construction there (one row per
  transmission; negligible). Note: today the `timeStamp*` lines run for
  len<4 transmissions too; after the restructure they run only inside the
  legacy branch — no observable change (columns unread).
- Lines 1654-1686: `collect_rows`, `distF` (dead but keep), the
  `pd.to_numeric` block, the `water_column` block, CSV writes.

### 2.2 The restructure + guard (S1)

The per-transmission block is restructured so the guard runs right after
the time sort, the four dead assignments (2.1) move into the legacy branch
only, and the legacy body is preserved verbatim under the guard:

```python
            if len(tDat) >= 4:
                if print_output == True:
                    print (len(tDat))
                tDat.sort_values(by = 'seconds_fix', axis = 0, ascending = True, inplace = True)
                # PERF 2026-10-26 (docs/deng_perf_handoff_2026-10-26.md v2.1):
                # fast path requires unique receivers and strictly increasing
                # times within the transmission. Violations fall back to the
                # legacy loop below (byte-identical to today's behavior,
                # including the four dead column assignments). Invariant scan
                # (Phase 0.3) found zero violations in 2019 and 2025 data.
                recs_arr = tDat.Rec_ID.values
                times_arr = tDat.seconds_fix.to_numpy(dtype = float)
                fast_ok = (len(np.unique(recs_arr)) == len(recs_arr)) and bool(np.all(np.diff(times_arr) > 0))
                if not fast_ok:
                    tDat['timeStampFix'] = pd.to_datetime(tDat['seconds_fix'], unit = 's')
                    tDat['timeStampOriginal'] = pd.to_datetime(tDat['seconds_fix'], unit = 's')
                    tDat['rank'] = tDat.seconds_fix.rank()
                    tDat.set_index('Rec_ID', drop = False, inplace = True)
                    <current lines 1489-1644 VERBATIM, re-indented one level>
                else:
                    <fast path: Section 2.3>
            else:
                <current lines 1647-1652 VERBATIM (unchanged)>
```

The **entire current legacy loop body (lines 1489-1644: `tested = []`, the
`combinations(tDat.Rec_ID.values, 4)` loop, everything inside) is kept
verbatim under the guard** — nothing in it is modified. For S1, the fast
branch (`else:`) temporarily contains a COPY of the same legacy body (so
S1 is a pure structural refactor — identical behavior by construction,
including which dead work runs, because both branches are identical).
S2-S5 then transform the fast branch only, per Section 2.3.

Gate S1 on both harnesses (expect trivial pass; it validates the
restructure and the harness itself).

### 2.3 The fast path (S2-S5, implemented inside `else:` of the guard)

Reference implementation (port exactly; keep the `print_output` prints with
the same format strings where values match — they are dead in production
because `lp.deng` never sets `print_output=True`):

```python
                else:
                    k = len(recs_arr)
                    # Per-transmission precompute: same functions and
                    # arguments as the legacy per-combo calls, executed once
                    # per receiver instead of once per combo.
                    pos_arr = np.empty((k, 3))
                    for idx in range(k):
                        rec = recs_arr[idx]
                        t = times_arr[idx]
                        z = self.receiver_z_at(rec, t, z_by_rec[rec])
                        pos_arr[idx] = self.receiver_position_at(rec, t, z, tracks)
                    # Sound speed cached per distinct t_ref (legacy: one
                    # interpolator call per combo; distinct t_ref values per
                    # transmission <= k). Same function, same scalar input.
                    sos_cache = {}
                    def sos_at(t):
                        v = sos_cache.get(t)
                        if v is None:
                            v = sos(self.interpolator(t))
                            sos_cache[t] = v
                        return v
                    tested = set()
                    sol_no = 0
                    for (i0, i1, i2, i3) in combinations(range(k), 4):
                        recs = (recs_arr[i0], recs_arr[i1], recs_arr[i2], recs_arr[i3])
                        if recs in tested:
                            continue  # unreachable under fast_ok; guard kept
                        ref, r1, r2, r3 = recs
                        t_ref, t1, t2, t3 = times_arr[i0], times_arr[i1], times_arr[i2], times_arr[i3]
                        r0Pos, r1Pos, r2Pos, r3Pos = pos_arr[i0], pos_arr[i1], pos_arr[i2], pos_arr[i3]
                        self.z_translation = r0Pos[2]
                        if print_output == True:
                            print ("The position of receiver %s is %s"%(ref,r0Pos))
                        R = np.array([[r1Pos[0]-r0Pos[0], r2Pos[0]-r0Pos[0], r3Pos[0]-r0Pos[0]],
                                      [r1Pos[1]-r0Pos[1], r2Pos[1]-r0Pos[1], r3Pos[1]-r0Pos[1]],
                                      [r1Pos[2]-r0Pos[2], r2Pos[2]-r0Pos[2], r3Pos[2]-r0Pos[2]]])
                        if print_output == True:
                            print ("Posisition Matrix R \n %s"%(R))
                        tdoa_1 = np.round(t1 - t_ref, 6)
                        tdoa_2 = np.round(t2 - t_ref, 6)
                        tdoa_3 = np.round(t3 - t_ref, 6)
                        tv = np.array([tdoa_1, tdoa_2, tdoa_3])
                        if print_output == True:
                            print ("Print T matrix \n %s"%(tv))
                        SoS = sos_at(t_ref)
                        if print_output == True:
                            print ("Current Speed of Sound = %s"%(SoS))
                        b1 = np.linalg.norm(r1Pos-r0Pos)**2 - (SoS**2 * tdoa_1**2)
                        b2 = np.linalg.norm(r2Pos-r0Pos)**2 - (SoS**2 * tdoa_2**2)
                        b3 = np.linalg.norm(r3Pos-r0Pos)**2 - (SoS**2 * tdoa_3**2)
                        bv = np.array([b1, b2, b3])
                        if print_output == True:
                            print ("B matrix \n %s"%(bv))
                        try:
                            Rinv = np.linalg.inv(R)
                            RinvT = Rinv.T
                            mt = (tv @ Rinv) @ RinvT
                            a = SoS**4 * (mt @ tv) - SoS**2
                            p = -0.5 * SoS**2 * (mt @ bv)
                            mb = (bv @ Rinv) @ RinvT
                            q = 0.25 * (mb @ bv)
                            if print_output == True:
                                print ('a = %s, p = %s, q = %s'%(a,p,q))
                            discriminant = float(p**2 - a*q)
                            coefficient = float(a)
                            if not np.isfinite(discriminant) or discriminant < 0 or coefficient == 0:
                                add_row('A', (j, sol_no, ref, r1, r2, r3, 9999.,9999.,9999.,9999.,9999.,
                                              'negative quadratic discriminant - no solution',''))
                                add_row('B', (j, sol_no, ref, r1, r2, r3, 9999.,9999.,9999.,9999.,9999.,
                                              'negative quadratic discriminant - no solution',''))
                                sol_no = sol_no + 1
                                tested.add(recs)
                                continue
                            sq = np.sqrt(p**2 - a*q)
                            T_0a = (-p + sq)/a
                            T_0b = (-p - sq)/a
                            if print_output == True:
                                print ('T_0a = %s, T_0b = %s'%(T_0a, T_0b))
                            if np.sign(T_0a) > 0:
                                S1a = RinvT @ (0.5 * bv - SoS**2 * (tv * T_0a))
                                point = np.array([r0Pos[0] + S1a[0], r0Pos[1] + S1a[1], r0Pos[2] + S1a[2]])
                                in_hull = point_in_hull(point, self.convex_hull)
                                add_row('A', (j, sol_no, ref, r1, r2, r3,
                                              r0Pos[0] + S1a[0], r0Pos[1] + S1a[1], r0Pos[2] + S1a[2],
                                              T_0a, times_arr[0], 'solution found', in_hull))
                                if print_output == True:
                                    print ("Solution Found for fish %s at transmission %s"%(self.tag,j))
                            else:
                                if print_output == True:
                                    print ("No solution A found time step %s"%(j))
                                add_row('A', (j, sol_no, ref, r1, r2, r3, 9999.,9999.,9999.,9999.,9999.,
                                              'negative time of arrival - no soluiton',''))
                            if np.sign(T_0b) > 0:
                                S1b = RinvT @ (0.5 * bv - SoS**2 * (tv * T_0b))
                                point = np.array([r0Pos[0] + S1b[0], r0Pos[1] + S1b[1], r0Pos[2] + S1b[2]])
                                in_hull = point_in_hull(point, self.convex_hull)
                                add_row('B', (j, sol_no, ref, r1, r2, r3,
                                              r0Pos[0] + S1b[0], r0Pos[1] + S1b[1], r0Pos[2] + S1b[2],
                                              T_0b, times_arr[0], 'solution found', in_hull))
                                if print_output == True:
                                    print ("Solution Found for fish %s at transmission %s"%(self.tag,j))
                            else:
                                if print_output == True:
                                    print ("No solution B found time step %s"%(j))
                                add_row('B', (j, sol_no, ref, r1, r2, r3, 9999.,9999.,9999.,9999.,9999.,
                                              'negative time of arrival - no soluiton',''))
                        except:
                            if print_output == True:
                                print ("Singular matrix encountered, no solution at transmission %s"%(j))
                            add_row('A', (j, sol_no, ref, r1, r2, r3, 9999.,9999.,9999.,9999.,9999.,
                                          'singular matrix encountered - no soluiton',''))
                            add_row('B', (j, sol_no, ref, r1, r2, r3, 9999.,9999.,9999.,9999.,9999.,
                                          'singular matrix encountered - no soluiton',''))
                        sol_no = sol_no + 1
                        tested.add(recs)
```

Sub-item split (each commit + gate; the reference above is the FINAL form —
land it incrementally):
- **S2** = precompute in the fast branch (`pos_arr` once per receiver,
  `sos_at` cache) + keep the *legacy per-combo body* in the fast branch too
  (i.e. fast branch = legacy loop body but reading `pos_arr`/`sos_at` where
  the legacy body called `receiver_position_at`/`interpolator`, and the four
  dead assignments dropped per 2.1). This isolates the caching change.
- **S3**: replace the fast branch's per-combo pandas (masks, `pd.concat`,
  `sort_values`, `recs` tuple) with the index-array version above, and
  `tested` list -> `set` (membership semantics identical: append-after-check
  placement preserved).
- **S4**: replace the fast branch's `np.matrix` math with the `np.array` +
  single-`inv` + `@` form above; extend `add_row` to accept a tuple:
  ```python
  def add_row(which, row):
      if isinstance(row, pd.DataFrame):
          row = row.to_numpy()[0]
      pending[which].append(row)
      if len(pending[which]) >= 50000:
          ...  # unchanged
  ```
  and switch the fast branch's row construction from per-row DataFrames to
  the 13-tuples above (the legacy branch keeps passing DataFrames —
  `add_row` handles both).
- **S5**: final cleanup pass — remove now-dead local variables in the fast
  branch, verify comment strings, run the full gate set, record final
  before/after wall times. (Do not delete the legacy branch — it is the
  permanent guard for unexpected data.)

### 2.4 Exact-string checklist (a typo change is a behavior change)

Preserve verbatim, including the legacy misspellings:
- `'negative quadratic discriminant - no solution'`
- `'negative time of arrival - no soluiton'` (sic)
- `'singular matrix encountered - no soluiton'` (sic)
- `'not enough receivers for solution'`
- `'solution found'`
- The `9999.` sentinel pattern (five 9999. for found-solution failure rows;
  ten 9999. for the not-enough-receivers row — that row is untouched anyway).
- Column semantics: `T01` = the root value (`T_0a`/`T_0b`); `ToA` =
  `times_arr[0]` — the FIRST arrival of the whole transmission, NOT `t_ref`.
- `in_hull` is a real bool only on 'solution found' rows; `''` everywhere
  else.
- `sol_no` increments exactly once per processed combination (all three
  exit paths: discriminant-continue, normal, except).
- `self.z_translation` is assigned per combo from `r0Pos[2]`, in combo order
  (last write wins — check external readers with a grep before finishing).

### 2.5 Why this is bit-identical (for the journal)

- `np.linalg.inv` on the same 3x3 float64 array is what `np.matrix.I` calls
  internally — same LAPACK path, same bits.
- `R.T.I` == `inv(R).T` exactly.
- The a/p/q expressions keep the legacy association:
  `(tv @ Rinv) @ RinvT` then `@ tv` / `@ bv` — the same dot products in the
  same summation order as the `np.matrix` chain.
- `mt` is computed once and reused for a and p; the legacy code recomputed
  the identical product — deterministic, so reuse is bit-identical.
- `tv * T_0a` (elementwise) == the legacy `(3x1) * (1x1)` matrix product
  (each output is a single product, same value).
- `np.round(tdoa, 6)` unchanged (microsecond convention is part of the
  method — HC-4/Section 8).
- Row tuples carry the same python/numpy scalar values; the `dtype=str`
  chunk construction and trailing `pd.to_numeric` block are unchanged, so
  CSV bytes follow.
- If any sub-item FAILS the byte-diff: do not force it. Print the differing
  intermediate (a, p, q, T_0a, T_0b, S1) for one failing transmission in
  debug runs on both code versions, identify which sub-expression diverged,
  keep the legacy form for that expression in the fast path, and document
  the divergence in the journal.

### 2.6 2019-specific notes for the implementer

- The 2019 run exercises the same `Deng()` with 6- or 9-receiver sets;
  receiver names are R01-R09. Nothing in the fast path may assume 2025
  receiver names, WSEL-followers, or GPS positions (the `receiver_position_at`
  call is generic — keep it exactly).
- 2019 deep beacons: receivers = surface list; 2019 study tags: all 9.
  (Phase 0.2 manifest records the exact sets — use it.)
- The 2019 DB may contain transNo gaps/NaNs that the 2025 drag DB doesn't —
  the `for j in sorted(tSteps)` loop structure is unchanged, so no new
  exposure, but the 2019 gate catches anything anyway.

---

## 3. PHASE 2 — Batched Deng (CONDITIONAL; owner-informed; do not start by default)

**Trigger:** only if the Phase 1 full-season estimate (Phase 0.3 ingest
times + Phase 1 per-tag Deng times, extrapolated to 44 tags) exceeds the
project lead's threshold (default proposal: > 6 hours total). Present the
number and get the decision before writing code.

Design (all per transmission, k receivers, time-ordered arrays `T` (k,),
`P` (k,3)):
- `idx = np.array(list(combinations(range(k), 4)))` -> (n,4);
  `I0,I1,I2,I3 = idx.T`.
- `t_ref = T[I0]`; `tdoa = np.round(np.stack([T[I1],T[I2],T[I3]], axis=1) - t_ref[:,None], 6)`.
- `R = np.stack([P[I1]-P[I0][:,None], P[I2]-P[I0][:,None], P[I3]-P[I0][:,None]], axis=1)` (n,3,3).
- `SoS` via one vectorized `self.interpolator(t_ref)` call (same
  bounds_error failure mode as the per-combo version).
- **Singular-combo handling (must be behavior-exact):** compute
  `Rinv = np.linalg.inv(R)` in chunks of 1000 combos. `np.linalg.inv` on a
  stacked array raises if ANY member is singular. On `LinAlgError`: recompute
  that chunk combo-by-combo with the Phase 1 per-combo try/except (identical
  rows to legacy). No singularity threshold is ever introduced — legacy
  detects singularity by LAPACK failure, so must the batch.
- `a, p, q` as batched reductions; no-solution mask
  `(~np.isfinite(disc)) | (disc < 0) | (a == 0)` — the legacy code funnels
  all three conditions into the same comment string; keep that.
- `T_0a/T_0b`, root validity `np.sign(...) > 0`,
  `S1 = np.einsum('nij,nj->ni', Rinv.T, 0.5*bv - SoS**2[:,None]*(tv*T0[:,None]))`.
- `in_hull` vectorized: hull planes `(A (m,3), c (m,))` ->
  `np.all(points @ A.T + c <= 1e-12, axis=1)` (same tolerance 1e-12 as
  `point_in_hull`).
- Row assembly: build the A-row and B-row arrays in combo order (one row per
  combo per table, same as legacy), feed `add_row`/`collect_rows` through
  the same `dtype=str` chunking.

**Tolerance gate (write these numbers in the journal BEFORE running):**
row counts, comments, transNo/solNo sequences identical; |delta| on X, Y, Z,
T01 <= 1e-12 relative for all 'solution found' rows; ENT wmed/p90/scored
identical to Phase 0 baseline. Expectation is actually byte-identical CSVs
(same LAPACK, same summation orders); if the byte-diff fails while within
tolerance, stop and escalate — do not ship a tolerance-passing
non-identical core change without the owner's explicit acceptance recorded in
`LONG_TERM_CONTEXT.md`.

**2019 after Phase 2:** full end-to-end `run_data.py config/run_data_2019.toml`
re-run on a fresh DB (not just the Deng step) and byte-compare against the
Phase 0.2 baseline.

---

## 4. PHASE 3 — Full-season ingest speed-up (working copy only; zero core changes)

- `copy` `output/jsats3d_2025_chinook.db` to
  `output/working/jsats3d_2025_chinook_work.db` (22 GB — use
  `sqlite3 .backup` or a verified file copy; the production DB is never
  touched — read-only rule).
- On the copy only: `CREATE INDEX IF NOT EXISTS idx_drtag ON tblDetectionRaw(Tag_ID)`.
- Verify (Phase 0.3 already did the 3-tag check): indexed reads return
  identical rows/order for a sample of tags; then run the full season against
  the copy and compare per-tag `tag_data` frames for at least 3 tags against
  the unindexed path (pickled frame compare).
- The production DB stays index-free (documented pattern: "never add indexes
  to the final DB; use a working copy").
- 2019: unaffected (small DB, unindexed, identical code path — HC-2 holds by
  construction; the Phase 1/2 gates already cover the shared code).
- Optional (only with project-lead approval): add the copy+index step as a
  `prepare` helper in `scripts/tagdrag_2025_pipeline.py` / the chinook
  runner. That is 2025-driver-only code; the 2019 driver is untouched.

---

## 5. GATES (exact, per sub-item)

After every Phase 1 commit (S1..S5) and after Phase 2:

1. `python -m py_compile jsats3d/jsats3d.py`
2. **2025 drag gate:** harness `check2025` — re-run Deng for FC36/FFD3 on
   `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db` into a fresh folder;
   SHA-256 of the 4 CSVs must equal the Phase 0.1 baseline hashes; pickled
   frames must `pandas.testing.assert_frame_equal(..., check_exact=True)`
   against baseline (dtypes included).
3. **2019 fast gate:** harness `check2019` — against
   `output/cowlitz_2019_synch_1_recreated.db` (baseline DB, unmodified).
   Re-running all 316 study tags per sub-item would take days, so the gate
   is tiered:
   - **Subset (after S1-S4):** R01-R03 (surface receiver set) + the first 10
     study tags in the Phase 0.2 manifest order + 2 edge-case tags chosen
     from the manifest comment inventory: one whose baseline CSV contains
     'singular matrix encountered' rows, one containing 'negative quadratic
     discriminant' rows; if 2019 has none of a type, record that and fall
     back in order to: a tag with zero 'solution found' rows, then the tag
     with the most transmissions. SHA-256 of every subset CSV must equal
     the manifest hashes.
   - **Full (after S5, and after Phase 2 only):** R01-R03 + all 316 study
     tags (all-9 set); every CSV byte-identical to the manifest.
   - If any subset gate fails: run the full gate immediately before
     diagnosing, so the failure scope is known.
4. **ENT gate (S5 and Phase 2 only):** re-run the ENT scoring of Phase 0.1;
   wmed/p90/scored per line and per variant must equal baseline exactly
   (3D B-Kalman 4.07 m / 459 scored expected).
5. Record per-tag wall times before/after in the journal (this is the
   "before/after effect on a validation metric" the System Prompt requires —
   here the metric is wall time plus the unchanged accuracy numbers).

Rollback: `git revert <commit>` (each sub-item is its own commit on
`ENM_jsat3d_edits`), then re-run the two CSV gates to confirm restoration.

---

## 6. JOURNAL + COMMIT RULES FOR THIS WORK

- Branch: `ENM_jsat3d_edits`. Never `main`. No PR while a gate is failed or
  unreviewed (System Prompt git workflow).
- Commit messages: `perf(deng): S3 - index-array combos, tested set [gate: 2025 byte-identical, 2019 byte-identical]` style.
- Journal entry per phase (minimum fields per System Prompt 13): author,
  timestamps (UTC + local), tags, files touched, summary, decisions &
  assumptions, parameter changes (expected: NONE — if you find you need to
  change any constant, STOP and escalate; no silent tuning), blockers, next
  steps, active context. Include: gate results (hashes, wall times), the
  Phase 0.3 invariant-scan numbers, and the Phase 0.2 2019 tag/receiver
  manifest.
- `scripts/_perf_*.py` scratch scripts: keep until final sign-off, delete
  after (campaign convention), note the deletion in the final journal entry.

---

## 7. EXPECTED OUTCOME (fill from gates; report to project lead)

| Item | Baseline (Phase 0) | After Phase 1 | After Phase 3 |
|---|---|---|---|
| 2025 drag Deng wall (FC36+FFD3, 8-h window) | 30.8 min (T2 record) | measure | measure |
| 2025 drag CSV bytes | baseline hashes | identical (required) | identical (required) |
| 2019 artifacts | baseline hashes | identical (required) | identical (required) |
| ENT 3D B-Kalman wmed / scored | 4.07 m / 459 | identical (required) | identical (required) |
| Per-tag tblDetectionRaw ingest (chinook) | measure (no index) | unchanged | measure (indexed working copy) |

Success = every "identical (required)" cell holds AND wall time improves.
If wall time does not improve meaningfully (e.g. < 2x on Deng), report the
profile evidence and stop — do not chase Phase 2 without the project
lead's decision.

**Wall-time estimates (estimates only — replace with Phase 0 measurements):**
- 2019 import: ~1-3 h (7.77 GB raw).
- 2019 full process: ~2 h per deep receiver (Deng, 2026-10-02 journal) +
  metronome/clock-fix phases + 316 study tags at minutes each on current
  code -> plan for ~1-3 days unattended.
- 2025 drag baseline: ~1 h (30.8 min recorded Deng + overhead + ENT).
- Phase 1 expected effect on the Deng portion: per-transmission dead pandas
  (`to_datetime` x2) and per-combo pandas/inverses removed -> order-of-
  magnitude gains expected (e.g. ~2 h deep beacon -> minutes; 30.8 min drag
  -> single-digit minutes). Report measured values; the gates use bytes,
  not estimates.

---

## 8. OUT OF SCOPE (explicit; do not touch in this work)

- The F1 clock-correction block (`multipath_data_object.__init__` lines
  318-365) — validated; owner sign-off for standing pipeline behaviour is a
  separate open item.
- `multipath_classifier` (legacy 2019 path; not called by any 2025 pipeline;
  its `dat[dat.SNR > 0]` would drop all ATS rows — known limitation, not a
  defect to fix here).
- `classify_ats_fish` and its `SUSPECT_WINDOW` / `SUSPECT_LATE_S` parameters
  (owner-gated sweep, separate open item).
- `clock_fix` in any form — including the dead temperature load at lines
  965-972 (v1 Phase 4 withdrawn under HC-1/HC-4), the 5-iteration DBSCAN,
  the quantile eps (owner review pending since 2026-09-24).
- `widen_table` (v1 Phase 5 withdrawn; negligible cost, shared driver code).
- `Deng2D`, `deng_2d_roots` (2025-only diagnostics; not the bottleneck).
- Beacon DBSCAN fixed parameters (0.5 ms / 2.5 periods / min_samples 3) —
  owner sign-off pending.
- Microsecond quantization (`np.round(tdoa, 6)`, CSV `float_format='%.6f'`).
- Legacy Deng quirks (bare `except` at line 1633, Solution-B handling,
  hull-test asymmetries) — owner decision pending; HC-1 forbids changing
  which rows are produced.
- `position.__init__` WSEL/temp loads and the line-1384
  `WSEL/3.28084` unconditional division (legacy quirk — flag, don't fix).
- Reference clock / metronome choices (owner-owned escalations).
- Any change to `tblDetectionRaw` content, any index on a production DB,
  any write to raw data.
- `scripts/run_data.py` and all existing `config/*.toml` files (if the
  2019 baseline DB must move drives, use a NEW toml copy — 0.0).

## 9. SESSION DECISIONS 2026-10-26 (implementer Q&A)

Decisions from the planning session, 2026-10-26. These override any looser
reading of earlier sections.

- **D1 — Phase 0 first.** Confirmed: no code before both baselines exist.
  Order: 0.0 pre-flight -> 0.2a dry-run + import (foreground) -> 0.2b 2019
  process (background, user-launched, log) -> 0.1 2025 drag baseline -> 0.3
  profile/invariants/ingest timing -> 0.4 journal gate. 2019 and 2025 may
  run concurrently (separate DBs).
- **D2 — Phase 2 threshold.** Keep the 6-hour default, defined on the 2025
  full-season production run (44 tags: ingest + Deng + tag loop), decided
  with the Phase 1 measured extrapolation in hand. The threshold does NOT
  apply to the 2019 reference run (one-time parity artifact, not
  production).
- **D3 — Harness.** Commit `scripts/_perf_harness.py` as its own first
  Phase 0 commit (`chore(perf): add Phase 0/1 harness`). Keep it until
  final sign-off; delete with a final commit + journal note. (Deviates from
  the delete-after-use campaign convention because this work spans
  sessions — record that in the journal.)
- **D4 — np.matrix -> np.array safety.** No permanent debug flag (HC-3).
  Before the S4 commit, run a throwaway side-by-side script
  (`scripts/_perf_s4_compare.py`, deleted afterward): on the drag DB, for
  the first 200 transmissions with k >= 4 (deterministic run order), for
  every combo, compute the legacy np.matrix expressions and the new
  np.array expressions for (Rinv, mt, a, p, q, T_0a, T_0b, S1a, S1b, X, Y,
  Z, T01) and compare bit-exact (`==`). Expected: zero mismatches. Any
  mismatch -> Section 2.5 procedure (keep the legacy form for that
  expression, journal it). The S4 byte-diff gate remains the final
  authority.
- **D5 — 2019 run mode.** Staged (0.2): dry-run -> import in foreground
  (observed) -> full process in the BACKGROUND with a log, launched by the
  user from their own terminal (System Prompt 16). Checkpoints before
  leaving it unattended: metronome done, first clock-fix receiver done.
  Resume: `python scripts/legacy_pipeline.py process config/run_data_2019.toml --resume`.
  Partial-DB gotcha: delete the partial DB + journal before a clean
  restart (`overwrite = false` blocks re-runs).
- **D6 — Disk.** C: had ~20.8 GB free at 2026-10-26 ~15:00 local — not
  enough for the 2019 baseline DB (~15-30 GB expected) plus the Phase 3
  chinook working copy (~25 GB). 0.0 pre-flight measures all drives;
  placement is a user decision (see 0.0). K: is read-only raw data, never
  a target.
