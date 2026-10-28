# 2026-10-22 — PM figure request: plan written for Qwen 3.6 execution

## Task
PM email 2026-10-22 (forwarded by user): client-ready figures + CSVs for (a) three tag-drag
categories from `cowlitz_AT_2025_testing_sheets.xlsx` (CFNSC_Approach_Drags,
Forebay_StaticHolds, Forebay_Random_Drags — GPS tracks + solutions, plan view per ID +
interactive plots, "a script that runs all this anyhow that we will continue to re-run")
and (b) four example fish (26E4, 9F8C, 1D5A, B5ED — fates: CFNSC entry, turbine routes,
spillway flumes, reservoir return; early-year near-isothermal fish; color scale + points
showing jumps; optional animation). User asked for a detailed plan for Qwen 3.6 to execute:
run → analyze → report plot links back for review; no plot/diagnostic scripts may remain in
the pipeline; journal everything.

## Data inventory (verified this session, read-only)
- GPS file `5_array_testing/array_testing_drag_GPS.csv` line_names: ENT-01..05 (Jun 10
  10:24–10:54), FBY-D-1..4 (Jun 10 10:58–11:45), FBY-NS-01/02 + FBY-SN-01/02 (Jun 05,
  11:50–12:01). No CFNSC-named lines.
- Testing-sheets xlsx sheets: Upstream_StaticHolds, Forebay_StaticHolds, CFNSC_StaticHolds,
  Forebay_Approach_Drags, CFNSC_Approach_Drags, Forebay_Random_Drags, CFNSC_Drags.
  **CFNSC_Approach_Drags sheet lists Test IDs ENT-1..5 = ENT-01..05 in the GPS file.**
  **Forebay_StaticHolds sheet = FBY-S-1..7 (Jun 10, 09:58:02–10:11:45, ~60 s each), each with
  ONE known Lat/Lon point; FBY-S-6 row says "Replace this with DB-1-5 (below)" → full sheet
  has more rows (DB-1-5 group) that the executor must read.**
  FBY-NS/SN (Jun 5) belong to Forebay_Approach_Drags — NOT the static holds.
- All three requested drag categories fall inside the ENT DB window (2025-06-10 09:00–13:00,
  `config/run_data_tagdrag_ent.toml`): **no new parse needed for drags**; positions are in
  `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db` (last written 2026-10-22 16:46, F1 clock
  fix active).
- Baseline re-verified against `output/2025_review/.../item12_pipeline_summary.csv`:
  3D root-B Kalman medians ENT-01 8.62/10.67, ENT-02 5.63/6.37, ENT-03 3.81/3.41,
  ENT-04 2.62/2.41, ENT-05 3.13/2.95 (FC36/FFD3); weighted median = 3.95 m — matches the
  2026-10-21 PASSED validation. Table now in the plan §1.
- Example fish: **not in any local DB** (checked chinook.db, chinook_backup.db,
  tagdrag DBs, output_archive). `jsats3d_2025_chinook.db` (14 GB, written 2026-10-22 16:59)
  contains only 0FBD/041D/493F/5F82, Jun 17–Jul 31, and has NO position tables (clock-fix
  stage only) — active parallel work, marked do-not-touch in the plan.
- `released_v0.csv`: 1D5A (rel. 2025-06-17 11:33), 9f8c (2025-06-24 10:35), b5ed
  (2025-06-24 10:35) found; **26E4 NOT found** → flagged as escalation trigger if also
  absent from raw data.
- Raw data `K:\...\2025_Data\raw_data\`: 487 files, per-date folders from 20250618 on.
- plotly 6.9.0 verified installed in the jsat_3d env.
- Pipeline command pattern re-verified from source: `run_data.py` = step 1 parse only;
  `tagdrag_2025_pipeline.py` = steps 1b–11 (+drag plot steps). STEPS list confirmed
  (prepare … positions_dbscan, plot_drag*). For the fish run the plan uses both commands
  with `--only` excluding the three plot_drag* steps (drag-GPS specific).

## Decisions
- WS order: kick off the multi-hour fish build first (background), build drag figures while
  it runs.
- Script rule interpretation: exactly ONE persistent generator (`scripts/
  drag_test_figures_2025.py`) remains — it is NOT a pipeline step (never invoked by
  run_data.py / tagdrag_2025_pipeline.py) and the PM explicitly wants it re-runnable;
  precedent = `chinook_analysis_2025.py`, `ent_analysis_2025.py`. All other helpers are
  `_`-prefixed throwaways, deleted after use. Flagged to user for confirmation.
- Fish product: new DB `output/jsats3d_2025_earlyfish.db`, window 2025-06-17 → 2025-07-31,
  tags 26E4/9F8C/1D5A/B5ED; 3D root B only (no fixed depths → 2D step no-op).
- Outputs: `output/pm_figures_2026-10-22/` (drags/, example_fish/, qc/, README.md index).
- Validation gates in plan: transform round-trip, ENT baseline agreement (~1 m),
  static-hold median error table (pass bar < 10 m), row-count integrity, file sizes,
  fish jump check (>3 m/s flags), idempotent re-run.

## Files
- Created: `docs/pm_figures_plan_2026-10-22.md` (self-contained execution plan for Qwen 3.6,
  incl. verified baseline table, exact commands, escalation triggers, report-back format,
  reviewer checklist).
- Created (this journal): this file.
- Deleted: `scripts/_plan_probe.py` (DB probe used during inventory).
- Pipeline files modified: none. K: untouched (read-only).

## Status
PLAN WRITTEN. Awaiting user confirmation of the persistent-script decision, then Qwen 3.6
executes `docs/pm_figures_plan_2026-10-22.md` and reports file links back for review.

## Takeover (2026-10-22 ~23:30 -> 10-23 ~00:15): Qwen 3.6 audit, fixes, completion
User instruction: look at Qwen 3.6's mistakes, fix them, run what the plan said to run,
and report results + locations.

### Qwen 3.6 audit (evidence-based)
WORKING:
- Static-hold FBY-S-1..7 plots + error rows (median 6.50 m): transform + DB access sound.
- Throwaway cleanup done; ENT/chinook DBs untouched (mtimes predate Qwen's first artifact).
BROKEN (generator v1 + its artifacts):
1. Drag scoring used DBSCAN-screened per-transmission MEANS, not the reference
   consensus/smooth/Kalman chain on tblPositions_Deng -> FBY-D-1 reported 91 m and
   ENT-01/04 + FBY-D-2/3/4 reported "no solutions in GPS window" (FALSE: window probe
   shows 16-226 candidate transmits per window in both Deng and DBSCAN tables).
2. Per-ID plots drew all-day solution clouds (window filter not effective on the
   plotted set); error circles hard-coded at (500, 100).
3. Static holds read from an 8-row preview: FBY-S-8/9/10 missing; DB-1..5 (2025-06-11,
   outside the ENT DB window 2025-06-10, and no GPS collected - DB-1 "Not Collected",
   DB-2..5 blank) produced junk all-day "drag" plots and no error rows.
4. "Interactive" HTML was a 3.7-4 KB SVG stub, not plotly (journal admits it).
5. README headline "Median XY (all lines): 93.22 m" computed from the broken 2-row QC
   CSV - a broken product that must not reach the client.
6. Fish workstream blocked on a FALSE PREMISE: 9F8C/1D5A/B5ED "not in parsed raw"
   inferred from absence in chinook.db / ENT DB - neither DB selects those tags
   (parser filters on the config tag list), so absence proves nothing. Its line-by-line
   raw scan "timed out" and it gave up, blocking all four tags instead of escalating
   26E4 and continuing with the other three (plan instruction).
7. config/run_data_earlyfish_2025.toml was invalid: triple [paths] tables (TOML error),
   invented [filter]/[sync]/[positions] sections (run_data.py schema is
   [selection]/[study]/[legacy]/[dbscan]), hobo_dir missing \DD_N, anchor ZOI10
   (chinook config uses ZOI03). It could never have parsed.
8. Journal/README inconsistencies (journal "6 CFNSC rows" vs 2 in CSV vs 0 later).

### Fixes applied (this session)
- scripts/drag_test_figures_2025.py REWRITTEN (the one persistent generator):
  * Scoring = ent_analysis_2025 item12 chain verbatim: tblPositions_Deng root B ->
    legacy_receiver_set_consensus(10 m, >=2 sets) -> smooth_track(10,11,3) ->
    kalman_filter_track(2.0,1.0,0.5,20 s); window matching on first-arrival epoch
    (min seconds_fix per Tag_ID/transNo from tblDetectionFilterSecondary, the same
    source the baseline uses) +/- 3 s.
  * Static holds read from the FULL Forebay_StaticHolds sheet; valid only with numeric
    Lat/Lon + Date/Start/End -> FBY-S-1,2,3,4,5,7,8,9,10 (FBY-S-6 + DB-1..5 excluded,
    reasons documented in README).
  * Per-ID figures: GPS truth (black, reference 1 Hz reconstruction) + per-tag Kalman
    track with time-colored points; error circles anchored at GPS start.
  * Interactive HTML: real plotly (inline, hover, receivers, stars for holds).
  * Client CSVs: per category x tag with product in {solution, consensus} +
    <category>_gps_tracks.csv; QC: qc/drag_line_errors.csv, qc/static_hold_errors.csv.
  * Baseline gate: ENT-01..05 must reproduce item12 within 1 m else exit 1.
  * README written ASCII-safe + explicit utf-8 (v1 mangled en/em dashes via cp1252;
    also fixed pandas r.product method-collision that printed "<bound method>").
- config/run_data_earlyfish_2025.toml REWRITTEN from the chinook template: tags
  26E4/9F8C/1D5A/B5ED, window 2025-06-17 -> 2025-07-31, output jsats3d_2025_earlyfish.db,
  anchor ZOI03, hobo_dir\DD_N, adopt_deep_z='config' (tape-measured deep Z).
- Qwen's stale output/pm_figures_2026-10-22/ deleted before regeneration.

### Verification (before -> after)
- Baseline gate PASS: all 10 ENT lines reproduce item12 EXACTLY (diff 0.00 m):
  ENT-01 8.62/12.5 + 10.67/16.8; ENT-02 5.63/11.4 + 6.37/16.0; ENT-03 3.81/16.0 +
  3.41/8.5; ENT-04 2.62/4.4 + 2.41/9.5; ENT-05 3.13/7.7 + 2.95/9.3 (median/p90 m,
  FC36/FFD3); n_scored identical to item12 (28/9/31/27/56/49/62/50/70/58).
- New results (missing/broken in v1): FBY-D-1 FC36 35.83/47.9 m (n=71); FBY-D-1 FFD3
  no scoreable transmits (only 4 FFD3 solution transmits exist in the window - real
  yield gap); FBY-D-2 7.42 + 11.53; FBY-D-3 2.98 + 3.25; FBY-D-4 24.60 + 4.69 (n=6).
- Static holds (3D per-transmission median): S-1 2.71/3.10, S-2 7.98/11.12, S-3
  2.04/2.13, S-4 0.89/3.88, S-5 9.00/11.26, S-7 5.47/4.01 m (pass bar met);
  S-8 7.31/22.85, S-9 40.36 (one 427.6 m outlier transmission, p90 427.6)/16.03,
  S-10 32.50/56.74 - late holds (10:13-10:20) degrade, offsets mainly +Y (north).
  Real finding, not a code bug (chain proven by ENT reproduction); documented in the
  README as a follow-up (candidates: late-morning clock drift or boat drift off the
  recorded hold point).
- Visual checks: ENT-02 + FBY-D-3 tracks hug the GPS line; FBY-D-1 shows the final
  ~40 s of the line ~15-30 m north of the GPS endpoint (boat final-stop artifact,
  documented in README).
- Deliverable: 36 files under output/pm_figures_2026-10-22/ (README index).

### Fish workstream (in progress)
- First raw pre-check (64 byte-pattern membership scans per file) stalled ~90 min with
  no progress (64 passes x ~45 MB per file over K:); killed. Rewritten probe: single
  pass per file, comma-anchored case-insensitive regex (,|^)(G72)?<code> , ; folders
  20250618..20250806 only (fish window + one recovery safety net); exact column-5
  confirmation on hit files; reports raw casing (parser tag match is CASE-SENSITIVE and
  run_data.py uppercases config tags - if raw casing differs, the config must match).
- Pending probe: run_data.py (parse only) + tagdrag_2025_pipeline.py --only prepare
  metronome clock_surface deep_beacons deep_positions adopt_deep deep_clocks
  fish_multipath speed_of_sound positions export positions_2d positions_dbscan
  (background, logged), then fish figures (plan views, color scale, points, GIF
  animation - no ffmpeg on this machine, GIF via PillowWriter) + CSVs in example_fish/.
- 26E4 still absent from released_v0.csv -> escalation to user per plan; raw scan is
  the definitive check; continue with the other three regardless.

### Files
- Rewritten: scripts/drag_test_figures_2025.py, config/run_data_earlyfish_2025.toml.
- Regenerated: output/pm_figures_2026-10-22/ (36 files) + README.
- Throwaway (delete after use): scripts/_probe_rawfish.py, output/_rawfish_probe.log,
  output/_drag_gen_run.log, scripts/_probe_gtime.py, scripts/_probe_windows.py.
- K: untouched (read-only). ENT DB + chinook DB untouched.

### Status
DRAG DELIVERABLE COMPLETE + VERIFIED (baseline gate PASS). Fish build pending raw
pre-check (running in background).

## Raw-scan correction (2026-10-23 ~01:30)
CRITICAL FIND - earlier raw pre-checks were INVALID (method bug, caught by validator):
- Raw tag fields are G72 + 8 hex digits (e.g. G72FC3617), NOT G72 + 4. The DB 4-char
  code is chars [3:7] (parser parse_tag_code), so the 4-hex code is the FIRST 4 of the
  8 suffix digits.
- My pre-pass regex required the 4-hex code to END at the comma (bare field or
  G72+4), so it could never match G72<code><4 more hex>, - both the "75/166, 0 hits"
  run and the "166 files, NOT FOUND" run used the broken pattern. Their negative
  results are VOID.
- Validator confirmed the fix: same June file now yields FC36=211, FFD3=416 detections
  (matches independent byte counts).
- FIRST CORRECTED RESULT (1 file, 20250618_Receivers/SR18076_250610_125501.csv):
  26E4 PRESENT - tag G7226E443, 2 detections, raw time 2025-06-16 19:51 (receiver-local,
  unclock-corrected). 26E4 was detected ~a day BEFORE the 1D5A release (06-17 11:33)
  and ~8 days before 9F8C/B5ED (06-24) - consistent with it being absent from
  released_v0.csv (released earlier, or pre-release deployment).
- Action: config window extended 2025-06-17 -> 2025-06-15 start (selection + study
  sync) so 26E4's early detections are inside the study window; corrected full 166-file
  rescan launched (checkpoint cleared; per-file heartbeat; validator on first 3 files).
- Files: _probe_rawfish.py + _probe_rawfish2.py (throwaway), _probe_format.py
  (throwaway, raw format inspector), output/_rawfish_probe{,2,3}.log,
  output/_rawfish_checkpoint.jsonl (deleted + regenerated by the corrected run).

## Raw scan FINAL (2026-10-23 ~02:00) + fish build launched
Corrected scan (G72 + 6-hex format, validated: FC36/FFD3 counts match field-level
exactly) over all 166 window files - ALL FOUR FISH PRESENT:
- 26E4: n=37,987 detections, raw 06/11 04:40 -> 08/06 08:33, 19 receivers
- 9F8C: n=14,944,  raw 06/11 02:26 -> 08/05 15:39, 19 receivers
- 1D5A: n=189,380, raw 06/12 03:43 -> 08/06 02:22, 19 receivers
- B5ED: n=95,820,  raw 06/11 07:15 -> 08/06 10:25, 19 receivers
(Raw times are receiver-local, unclock-corrected; first detections 06/11 predate the
06/17 and 06/24 released_v0 dates - same pattern as 26E4's missing release record;
clock sync in the pipeline will give study times.)
Window decision: extended 2025-06-15 00:00 -> 2025-08-06 23:59:59 (selection, study
sync, dbscan) to cover the full raw span so no fate is clipped (parse cost is the same
file set; only row count grows).
Launched (background, output/_earlyfish_build.log): run_data.py (parse, 4 tags +
beacons) && tagdrag_2025_pipeline.py --only prepare metronome clock_surface
deep_beacons deep_positions adopt_deep deep_clocks fish_multipath speed_of_sound
positions export positions_2d positions_dbscan. ETA hours (chinook precedent ~4 h).
On completion: verify positions per tag -> scripts/_fish_figures_2025.py ->
example_fish/ figures + README fates -> final report.

## Fish build: parse OK, pipeline pulse-rate fix (2026-10-23 ~03:10)
- run_data.py parse COMPLETED: output/jsats3d_2025_earlyfish.db, 304 files, 20/20
  receivers, 29,912,682 detections, 43 tags (4 study + 39 beacons/other).
- Pipeline first launch FAILED: "no study tags with a pulse rate" - parser left
  pulseRate NULL for all 4 fish. Root cause: released_v0.csv has NO pulse-rate
  column (cols: Tag Code, Acoustic Tag Value, Species, Life Stage, Mark Time,
  Release Site/Date/Time, Length, Weight), and PROVISIONAL_STUDY_PULSE_RATES in
  adapt_2025_to_legacy.py only lists FC36/0B0A/0AC6/493F. (Beacons 1F14/1F38/
  1F5A/1F71/1F94/1FCD also NULL - not needed by build_context, chinook had same.)
- FIX: measured pulse periods from the parsed detections (median inter-pulse gap
  in [2.5,4.5] s, pooled over all 20 receivers; p10-p90 spread <=0.02 s = tight):
  26E4=2.986, 9F8C=3.050, 1D5A=3.081, B5ED=2.969 s. Set in config [study]
  pulse_rates; run_data.py --skip-build applied them to tblTag (no re-parse).
  Cross-check: ENT drag tags FC36=3.038/FFD3=3.330 - same order of magnitude.
- Pipeline relaunched (background, output/_earlyfish_pipeline.log): prepare
  metronome clock_surface deep_beacons deep_positions adopt_deep deep_clocks
  fish_multipath speed_of_sound positions export positions_2d positions_dbscan.
  ETA: hours (chinook precedent ~4 h for the full chain; parse already done so
  expect a bit less).
- Throwaway probes: scripts/_probe_pulserate.py (delete after use).
