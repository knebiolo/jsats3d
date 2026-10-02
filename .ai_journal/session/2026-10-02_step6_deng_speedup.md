# Session: 2026-10-02 — Step 6 (Deng) Speed-Up and Step 6 Runs, 2019 and 2025

- Author: Ethan Muhlestein / Copilot
- Timestamp: 2026-10-02 about 09:50 to 10:45 local (Pacific; the file `2026-10-01_parallel_rerun_2019_2025.md` carries the earlier work; entries there between 09:55 and 10:40 on 10-02 are the detailed record). UTC is local plus 7 h while daylight saving applies.
- Tags: #step6 #deng #performance #2019 #2025 #parallel #memory #decisions
- Branch: `ENM_jsat3d_edits`, HEAD `0a8d86c`, uncommitted edits.
- New session file: more than 60 minutes (about 10 h) after the last entry in `2026-10-01_parallel_rerun_2019_2025.md` at 23:50 (System Prompt Section 13).

## Active Context
- Both years follow the 12-step paper order, separate parallel processes, stop where told, 12-step table shown after every reply (project lead rule, see the 10-01 journal).
- Databases: `output/cowlitz_2019_synch_1_recreated.db`, `output/jsats3d_2025_v3.db`. Env `jsat_3d`, pandas 3.0.5.
- Steps 1-5 done for both years (2025 step 5 including the host fallback, see the 10-01 journal).

## Short Summary
- Step 6 for 2019 (R01, old code) had run 14+ hours without output. Measured why: `position.Deng` appended every solution to a growing DataFrame (quadratic) and, for 2025, filtered the 1.1M-row GPS table on every receiver lookup.
- Fixed (Stage A, project lead approved): same results, much faster. Validated identical to the committed code.
- Restarted 2025 step 6 (ZOI01-ZOI03) on the new code; started 2019 R02 on the new code; 2019 R01 old code left running as ground truth. Two jobs stopped for memory.

## Decisions & Assumptions
1. **Run 2025 step 6 now (user):** three parallel processes, one per deep receiver, reference receivers ZOI04-ZOI10 (`receivers_3d` minus deep), solution B median to `tblReceiver` X_t/Y_t/Z_t, no `tag_multipath` (step 5 already in the tables). First attempt failed because `legacy_pipeline.receiver_sets()` raises while `receivers_2d` is set and `fixed_z_2d` is blank (owner item open); bypassed in the inline run by computing the reference list directly; no config or code change for that.
2. **Stage A speed-up (user: "lets do Stage A"; question asked first whether it changes the outcome: answer was no for Stage A, last-digit differences possible for Stage B):** implemented, validated, accepted. Stage B (numpy batching) not started; would need a stated tolerance and the full-size 2019 comparison.
3. **Keep 2019 R01 (old code, PID 37208) running** as full-size ground truth; it writes `FF76_solutionA/B.csv` in `output/cowlitz_2019_synch_1_recreated_legacy/deep_receivers` at the end of its Deng. When it prints its median, stop that process (it would continue into R02/R03 on the old code).
4. **Stopped two 2019 jobs for memory (my decision, nothing written by them):** new-code R01 check and R03, after free RAM fell to 1.1 GB of 31.5 GB. To restart when memory allows.
5. **Restarted the three 2025 jobs** (old code, 40 CPU-min each, nothing written) on the new code.

## Parameter Changes With Rationale
- None. No DBSCAN, clock, sound-speed, geometry or Deng parameter changed. The speed-up changes only how rows are collected and how receiver data is looked up.

## Run Log and Results
- 2019 step 6 old code at 09:50: still R01, 804 CPU-min (13.4 h); at 10:40: 853 CPU-min. No progress output; 2019 DB unchanged since 10-01 18:52.
- Measurement of the slowness (read-only, temp output folder): FF76 into Deng = 530,111 rows, 113,968 epochs (110,273 with 4+ receivers; 78,373 with 5, 31,900 with 4), 423,765 4-receiver combinations. Base cost 0.13 s per epoch (200/400/800 epochs: 25.9/55.1/104.3 s) = about 4.1 h. One `pd.concat` onto a growing frame (pandas 3.0.5, str dtype): 1k rows 2 ms, 50k 9 ms, 100k 19 ms, 200k 29 ms; over 424k combinations about 7.5 h more; R01 total about 12 h, three 2019 receivers about 36 h. Cross-check: an earlier successful pass had 164,320 solutions and took just over an hour, which the model reproduces.
- Validation of the new code against the committed code (`git show HEAD:jsats3d/jsats3d.py` loaded from the Temp folder `jsats3d_old`; frames compared with exact equality including dtypes, plus CSV bytes):
  - 2019 FF76, first 400 epochs: A 1,822 rows and B 1,778 rows, identical; 50.7 s -> 19.0 s.
  - 2025 FFC7, 40 epochs with 5 or more receivers: A 238 and B 202 rows, identical; 177.5 s -> 5.8 s (30x).
  - Full unit test suite: 39 tests OK.
- Speed estimates, new code: 2019 FF76 2,000 epochs 97.9 s, 4,000 epochs 229.2 s (about 0.05-0.06 s per epoch, was 0.13 s and growing) -> about 2 h per deep receiver (was about 12 h). 2025 about 0.15 s per epoch with 7 reference receivers -> about 4-5 h per receiver (old code over 100 h).
- Launches on the new code: 2025 ZOI01 (FFC7), ZOI02 (7D2D), ZOI03 (7DBC) at about 10:33-10:36, logs `output/run_2025_step6_ZOI0*.log`; 2019 R02 (FF74) at 10:37, log `output/run_2019_step6_R02.log` (config `config/run_data_2019.toml`, reference R04-R09, solution B). Each writes `deep_receivers_<REC>` under its `_legacy` work folder and updates `tblReceiver` at the end.
- Memory: the three 2025 jobs hold 4.3-4.8 GB each (all 1.1M `tblReceiverGPS` rows are loaded, though only CFD02-CFD09 have GPS tracks). Free RAM 2.9 GB at 10:40 with old R01, new R02 and three 2025 jobs running.
- Not yet produced: any step 6 result (no medians yet for either year).

## Code Changes
- `jsats3d/jsats3d.py`, `position.Deng`: solution rows collected in lists and turned into frames in 50,000-row chunks, one concatenation at the end (was one `pd.concat` per solution); epochs grouped once with `groupby('transNo')`; receiver Z from a dict built from the ephemeris (first row per receiver, as `.values[0]` did). Same row-building code, same try/except structure, same legacy quirks.
- `jsats3d/jsats3d.py`, `position.receiver_position_at(rec_id, timestamp, z_value, tracks=None)` plus new `_receiver_track(rec_id)`: optional per-receiver cache (static X/Y, sorted and de-duplicated GPS arrays, their min and max). Without `tracks` the behaviour is as before; the existing GPS unit test still passes.
- Earlier today (10-01 journal): `multipath_data_object` host fallback; `beacon_pairwise_dbscan.filter_deep_beacon`.
- No new unit test yet for the `tracks` cache or the host fallback.

## Physics / Method Notes
- Deng's exact solution per 4-receiver combination is unchanged. Legacy quirks left as they are (project lead has not decided): (1) if T_0a <= 0 < T_0b the solution-B point uses the never-assigned `S1a`; the NameError is swallowed by the bare `except:` and that B solution is lost, with a 'singular matrix' row written to SolutionA; (2) the B hull test uses S1a for Y and Z; (3) 'negative time of arrival' rows for B are appended to SolutionA. Effect on the number of 'solution found' rows in B not yet measured; medians use only 'solution found' rows.
- Stage B would change floating-point results in the last digits (about 1e-12) and could flip a borderline row at the discriminant, time-of-arrival sign or hull tolerance tests; to be quantified against the old-code R01 output before use.

## Blockers & Known Limitations
- Memory limits how many Deng jobs can run together; 2019 R03 and the new-code R01 check are waiting.
- `fixed_z_2d` still blank (blocks the full `process()` run, not the inline step 6 runs).
- Kevin sign-off on the 2025 reference clock (provisional ZOI08/7DB7, anchor ZOI10) still open; no post-correction clock residual (Gate 2) report for either year.
- 2025 deep receiver coordinates are unsurveyed config values; step 6 results will be sanity-checked against them only.

## Next Steps
1. When 2019 R01 (old code) finishes: record its median, stop that process, compare `FF76_solutionA/B.csv` with a new-code rerun of FF76 (exact check at full size), then run R03 and the R01 check on the new code.
2. When R02 and the 2025 jobs finish: report medians and spreads per deep receiver against the starting coordinates (2025) and the earlier run (2019 R01 median X_t -21.050, Y_t 9.736, Z_t 248.090); stop for review.
3. Optional: load only the needed GPS rows in `position.__init__` to save 3-4 GB per 2025 job (no expected change in results).
4. Do not start step 7 for either year until the project lead says so.

## Files Touched
- Modified: `jsats3d/jsats3d.py`, `LONG_TERM_CONTEXT.md` (standing decisions 2026-10-02), `.ai_journal/session/2026-10-01_parallel_rerun_2019_2025.md`.
- New: this journal.
- Generated (gitignored): `output/run_2025_step6_ZOI0*.log`, `output/run_2019_step6_R0*.log`, `deep_receivers_*` work folders; Temp folder `jsats3d_old\jsats3d_old.py` (copy of committed code for comparisons).
