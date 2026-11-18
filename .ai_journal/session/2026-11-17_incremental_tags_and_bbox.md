# Session 2026-11-17 — Incremental tag addition + user-set box in the 2025 pipeline

## Owner requests
1. "which scripts are diagnostic/searching scripts ... Do they actually do the pipeline
   for positioning? I want the median filter and what we worked on in the pipeline code,
   but i dont want all the outputs" -> deleted the three read-only A/B/diagnostic tools
   (position_box_compare_2025.py, jump_window_comparisons_2025.py, jump_window_presence_2025.py).
   Their generated deliverables remain in output/11162026_jump_tags/ (local, gitignored).
2. "the 2025 pipeline should already apply the spatial median filter that the user can set
   as needed (bbox) and run the position solving of either a few or all tags as needed"
   -> incorporated (NOT a new script) into the existing pipeline:
   (a) incremental tag addition on a finished DB,
   (b) optional explicit box override in [figures].
   "i dont want new scripts i want incorperations."

## Design
The parser (parse_ats_raw_to_legacy.py) is a from-scratch builder: it DELETES its output
database and writes every table with replace semantics (verified lines 510-511). So
incremental staging parses the new tags into a TEMPORARY database, then appends only the
new tags' rows. Division of labor:
- run_data.py: staging + registration (the "build" side)
- _2025_pipeline.py: per-tag positioning (the "solve" side)

### run_data.py
- staged_tags(db): tags with rows in tblDetectionRaw (None when the db is absent).
- build_steps (ats):
  - no db            -> full build exactly as before (parser direct, beacons included)
  - db exists + new  -> step "parse + stage new tags: X, Y" = stage_new_tags(...)
  - db exists + all  -> parse skipped (prints note; deleting the db is the only full-rebuild
  selected tags                 path — a rerun can no longer wipe a finished database)
  - --skip-build + un-staged selected tags -> hard error (before, they would have been
    silently unpositioned)
- stage_new_tags(output_db, paths, start, end, new_tags):
  parser (tag filter = new tags only, NO beacons — already staged) -> temp db
  <db stem>_addtags_parse.db; guards: every requested tag present in the temp parse,
  source table contains only the new tags, output db has all parser columns;
  append tblDetectionRaw in 200k-row chunks with a COMMIT PER CHUNK + flushed progress
  (the 2026-11-16 incident lesson); row-count check vs parsed; insert-or-ignore tblTag
  rows for the new tags; temp db deleted in finally.
- finish_database(..., selection_tags=()): after the run-file pulse rates, staged tags
  still with pulseRate NULL get a MEASURED rate (median inter-pulse gap in [2.5, 4.5] s
  pooled over receivers, the documented run-file method); <50 gaps -> reported as not
  measurable, tag stays staged but is excluded from positioning (CB75 case).
- measure_pulse_rate(raw, tag, lo, hi): the shared measurement helper (returns None when
  unmeasurable; the old standalone driver's raising variant).

### _2025_pipeline.py
- Per-tag completion markers in tblProcessProgress: __mp__<TAG> (multipath) and
  __pos__<TAG> (Deng + export). tag_markers(db, tags) backfills them once from
  finished-stage evidence (secondary rows / solution rows) so pre-existing finished
  databases light up correctly; idempotent.
- build_context: c.pending_mp / c.pending_pos; prints backfill + pending status; notes
  [selection] tags without a tblTag row/pulse rate ("stage with run_data.py first").
- fish_multipath_done/run: pending_mp only; __mp__ marked per tag INSIDE the loop
  (crash mid-loop resumes at the next pending tag).
- positions_done/run: pending_pos only.
- export_done/run: pending_pos only; DELETEs the pending tags' tblPositions_Deng rows
  first (crashed-half-export reruns cannot duplicate), then loads the pending CSVs
  (the fresh positions folder contains only pending tags' files), then marks __pos__.
- bathymetry_check_done: column exists AND no NULL values (rows appended by an
  incremental addition are NULL until the check reruns; the check is whole-table and
  idempotent).
- positions_dbscan_done: screened table covers every tag with solutions in
  tblPositions_Deng (a new solution tag triggers a whole-table rescreen; the screen
  writes its own table and never touches the product rows).
- Run-level steps (prepare, clock_jumps, metronome, clock_surface, deep_*, speed_of_sound)
  unchanged: receiver-level work is tag-independent and skipped on a finished DB.
- pipeline_status.json gains pending_multipath / pending_positions.

### ent_analysis_2025.py + figure scripts (the bbox override)
- New shared helper spatial_box(figs, receivers, floor_m, box_margin_m) ->
  (box, effective_margin, source):
  - default: receiver volume (min/max of ALL positioned receivers, entire array incl.
    the CFD row) + margin (exactly today's behavior),
  - [figures] position_3d_box_xy = [x0, x1, y0, y1]: explicit user window, margin NOT
    applied (the coordinates are the final window); Z is always [depth_floor_m, 0].
- fish_figures_2025.py and fish_dashboard_v3.py both call it (single source of truth;
  the old duplicated construction removed). Documented in both earlyfish run files
  (key present, commented out, with the strict array volume as example).

## Absorbed + deleted
position_additional_tags_2025.py deleted: its staging/pulse-rate/chunked-append logic
moved into run_data.py, its multipath/Deng/export reuse is now the pipeline's own
pending-tag path. No new scripts were created (tests/test_incremental_pipeline.py is a
test module, not a script).

## Tests (37 pass: 20 pre-existing + 17 new)
tests/test_incremental_pipeline.py: measure_pulse_rate (median/sparse/out-of-window),
staged_tags (absent db / with rows / no raw table), tag_markers backfill + idempotence,
pending-based step gates, spatial_box (receiver-derived / explicit / bad length / empty
receivers), TestBuildStepsStaging (late-binding closure regression: the staging step must
carry exactly the missing tags even with the study-key `missing` reassignment in play;
no stage step when all staged; stage_new_tags refuses an empty list).

## BUG FOUND + FIXED (classification error from the first purge)
adapt_2025_to_legacy.py WAS deleted in the first purge on the (wrong) claim that parse_ats
no longer imports it. It DOES: seven metadata functions (load_beacon_registry,
load_receiver_table, load_receiver_gps, load_temperature_string, load_environment,
tag_types, apply_tag_pulse_rates) are imported via an INDENTED try/except import that a
line-anchored `^import` grep missed. Caught when the new test imported run_data ->
parse_ats -> ModuleNotFoundError. Restored with `git checkout --` (script + the
test_smoke.py line). See the CORRECTION note in 2026-11-16_full_array_box.md.
Lesson: import greps must allow leading whitespace (try/except import blocks).

## Real-data validation (80FD re-add on a throwaway copy)
Plan: copy the finished Arm A DB to a temp dir, delete 80FD's rows (raw, tblTag,
filters, positions), run `python scripts/run_data.py run_verify_80fd.toml` (80FD added to
[selection] tags, outputs = positions only), then verify 80FD is re-positioned to exactly
the source rows and every other tag's rows are untouched.
Run file + verify script: C:\Users\Ethan.Muhlestein\AppData\Local\Temp\opencode\verif_incremental\
(disposable, outside the repo).

### INCIDENT (caught by the validation itself): late-binding closure -> UNFILTERED parse -> disk full
First validation attempt: the log showed "Parsing 0 new tags into temporary database ..."
while the step name said "parse + stage new tags: 80FD". The temporary parse database
grew to 14.93 GB (0 rows after rollback — the parser commits once at the end), filling
the C: drive (35 GB free -> 0.02 GB); the run died, the 0-byte log lost all output, and
the temp DB had to be deleted by hand to recover the disk.

Root cause: in run_data.build_steps the staging step was
`lambda: stage_new_tags(output_db, paths, start, end, missing)`. The lambda captures the
NAME `missing`, which is REASSIGNED later in the same function
(`missing = [key for key in ("bm_elev", "master_receiver") ...]` — the study-key check;
empty list because both keys are set). At call time the lambda therefore saw [] ->
stage_new_tags([]) -> parser_command with no tags -> NO --tag filter -> full unfiltered
parse of the entire raw dataset into the temp DB. (A standalone parser run with
`--tag 80FD` reproduces the CORRECT result: exactly 86,956 detections, 94.8 MB — so the
parser and the command builder were fine; only the closure was broken.)

Fix (run_data.py):
- `def stage_missing(missing=missing): ...` — default argument binds the VALUE at
  definition time; comment documents the reassignment below.
- stage_new_tags now REFUSES an empty new_tags list (ValueError) instead of degrading
  to an unfiltered parse — the failure mode that ate 15 GB can no longer happen.
- Regression test TestBuildStepsStaging (tests/test_incremental_pipeline.py): builds the
  real steps with the study-key reassignment in play and asserts the staged step carries
  exactly the missing tags (and that no stage step is built when all tags are staged).
  Suite: 37 tests OK.

State after incident: temp parse DB deleted, disk recovered (14.5 GB free), copy DB
verified byte-identical to source and matching the saved baseline exactly
(check_baseline.py PASS — no partial append reached the copy).

### Second incident (validation attempt 2): deep_beacons done-check gap (pre-existing)
With the closure fixed, the staging side worked end-to-end:
- "Detections: 86956", chunked append (1 chunk of 86,956), temp db deleted,
- "pulseRate 80FD = 3.0608 s (measured: median inter-pulse gap, p10-p90 3.0564-3.0620, n=43027)" — matches the 2026-11-16 measured value,
- "backfilled per-tag progress markers: __mp__/__pos__ for the 4 original tags",
- "incremental run: multipath pending 80FD | Deng+export pending 80FD",
- steps 1b/1c/2-3/4 (prepare, clock_jumps, metronome, clock_surface) all skipped.
Then step 5 deep_beacons RAN (not skipped) and crashed:
"table tblDetectionFilterSecondary has no column named seconds".

Root cause (PRE-EXISTING, not caused by this session's changes):
- deep_beacons_done checked that each deep receiver's beacon tag has rows in
  tblDetectionFilterSecondary. But in ANY database built by this pipeline, those rows
  are GONE by the end of the build: step 5 writes the beacon rows, step 6 consumes
  them, and step 8's per-fish multipath reload REBUILDS tblDetectionFilterSecondary
  (verified on the original Arm A DB: 2,700,543 secondary rows, all 12 fish tags,
  ZERO rows for beacons FFC7/7D2D/7DBC). The row-presence check can therefore never
  be true on a finished DB — the original workflow never noticed because the pipeline
  was only ever run ONCE per database.
- The re-run then crashed on a second latent quirk: filter_deep_beacon appends a
  DataFrame with a `seconds` column to a table whose schema has `seconds_fix`
  (the table was created by the legacy loader, not by that append). Unreachable once
  the done check is fixed; the legacy loader (beacon_pairwise_dbscan / jsats3d
  monolith) was deliberately left untouched.

Fix (_2025_pipeline.deep_beacons_done): if the beacon rows are present -> done (as
before); otherwise fall back to deep_positions_done (the beacons are the INPUT of
step 6, so with every deep receiver positioned, step 5 is done). Fresh builds are
unaffected: at step-5 time neither the beacon rows nor the deep positions exist, so
the step still runs. Suite: 37 tests OK.
Note: the interrupted attempt left partial FFC7 primary rows on the first copy, so
validation was restarted on a FRESH copy with the fixed code.

### FINAL RESULT: PASS (fresh copy, both fixes in place)
`run_data.py run_verify_80fd.toml` exited 0. Step trace: parse+stage (86,956 rows, 1
chunk), pulseRate 80FD = 3.0608 s (measured, n=43027), backfill of the 4 original tags,
pending = {80FD}. Every run-level step skipped — INCLUDING step 5 deep_beacons (the new
deep_positions fallback) and step 6 deep_positions. Only steps 8 (fish_multipath), 11
(positions/Deng), 10 (export), 11c (positions_dbscan rescreen) ran, for 80FD only.

verif_80fd.py verify — ALL checks PASS:
- 80FD raw 86,956 / pos 52,158 / found 1,806 / rate 3.0608 / TagType study — all match source.
- All 50 other tags' raw row counts identical; all 11 other fish tags' pos rows/found identical.
- No temporary parse DB left behind.
rowhash_80fd.py (multiset MD5 of Tag_ID, solution, comment, X, Y, Z for 80FD):
- source 26989f35622ec77f2b3da1932098979b == copy 26989f35622ec77f2b3da1932098979b.
=> The incrementally re-positioned tag is BIT-IDENTICAL to the original; the finished
   tags were untouched. Incremental tag addition is validated end-to-end on real data.

Cleanup: the throwaway copy + temp artifacts under
C:\Users\Ethan.Muhlestein\AppData\Local\Temp\opencode\verif_incremental\ are disposable;
deleted after validation to reclaim the ~22.7 GB.
