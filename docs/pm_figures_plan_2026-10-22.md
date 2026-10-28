# PM Figure Request 2026-10-22 — Execution Plan for Qwen 3.6

You are executing a client-facing figure/data request from the PM (email 2026-10-22).
This plan is self-contained. Read `System_Prompt.txt` and `LONG_TERM_CONTEXT.md` first —
the operating rules (journaling, change budget, escalate-don't-improvise, K: read-only)
apply in full.

**Your end state:** (1) client-ready plots + CSVs for 3 tag-drag categories and 4 example
fish, (2) QC/validation metrics computed and reported, (3) one-off scripts deleted,
(4) everything journaled, (5) a report-back message listing every file path so the
requesting engineer can forward the links for review.

---

## 0. Environment & hard rules

- Repo: `C:\Dev\jsat3d\jsats3d` · Branch `ENM_jsat3d_edits` (never `main`, no PRs).
- Python (the ONLY working env): `C:\Users\Ethan.Muhlestein\AppData\Local\anaconda3\envs\jsat_3d\python.exe`
  (Python 3.11.15, pandas 3.0.5, numpy 2.4.6, sklearn 1.9.0, scipy 1.17.1,
  plotly 6.9.0 — verified installed 2026-10-22). Run from the repo folder. `conda run`
  drops stdin — call the exe directly.
- **K: drive is strictly read-only.** Never write to `K:\Jobs\5662\...`. All outputs go to
  `C:\Dev\jsat3d\jsats3d\output\`.
- **Do not touch** `output/jsats3d_2025_chinook.db` or `jsats3d_2025_chinook_backup.db`
  (active parallel work, modified 2026-10-22). Read-only queries OK.
- **Do not edit existing pipeline files** for this task. The only new code is: one config
  (WS2), the persistent figure generator, and throwaway `_`-prefixed helpers (deleted after
  use, per convention). The ENT DB is used **read-only**.
- Journal to `.ai_journal/session/2026-10-22_pm_figures.md` as you go (create at start):
  decisions, metrics before/after, cleanup list, escalations. No silent parameter tuning.
- Long jobs (WS2 parse/pipeline): run in background with a log file, journal the command.

## 1. Data inventory (already verified — trust, but re-check cheaply as you go)

**Drag work (all three PM categories are ALREADY positioned in the current ENT DB):**

| PM category | Test IDs | Date / time (wall clock, UTC-encoded) | GPS truth source |
|---|---|---|---|
| CFNSC_Approach_Drags | **ENT-01..ENT-05** (sheet Test IDs "ENT-1..5") | 2025-06-10 10:24–10:55 | `array_testing_drag_GPS.csv` line_name ENT-01..05 |
| Forebay_StaticHolds | **FBY-S-1,2,3,4,5,7** (+ "DB-1-5" rows — read full sheet; FBY-S-6 row is marked "Replace this with DB-1-5 (below)" → exclude, use the DB rows) | 2025-06-10 09:58:02–10:11:45, ~60 s each | **No GPS track** — each hold has ONE known point (Lat/Lon) + start/end times in the sheet |
| Forebay_Random_Drags | **FBY-D-1..4** | 2025-06-10 10:59:03–11:46:00 | `array_testing_drag_GPS.csv` line_name FBY-D-1..4 |

- Positions DB: `output/jsats3d_2025_tagdrag_ent_anchorZOI03.db` (current production DB,
  last written 2026-10-22; F1 clock fix active — its item12 baseline was verified
  2026-10-21 after the clock-fix re-run; tags FC36 @ −1.524 m and FFD3 @ −4.572 m; tables
  `tblPositions_Deng` (3D, roots A/B), `tblPositions_Deng2D` (2D fixed-Z, roots A/B),
  `tblPositions_DengDBSCAN` (screened root-B), `tblReceiver` (20 receivers, X_t/Y_t/Z_t +
  easting/northing), `tblWSEL`).
  **Use root B** for all client products (root A is known-bad on this array — see
  `docs/accuracy_tuning_handoff_2026-10-21.md`).
- GPS/sheet files (read-only):
  - `K:\Jobs\5662\001\Data\DataTrans\2025_Data\CowlitzAT2025_Data_Deliverables\5_array_testing\array_testing_drag_GPS.csv`
    (columns: line_name, lat, long, timestamp; ~1 Hz rows)
  - `K:\Jobs\5662\001\Data\DataTrans\2025_Data\CowlitzAT2025_Data_Deliverables\5_array_testing\cowlitz_AT_2025_testing_sheets.xlsx`
    (sheets: Upstream_StaticHolds, Forebay_StaticHolds, CFNSC_StaticHolds,
    Forebay_Approach_Drags, CFNSC_Approach_Drags, Forebay_Random_Drags, CFNSC_Drags.
    Forebay_StaticHolds columns: Test ID, Date, Start Time, End Time, Zone, Lat, Lon, Comments.
    READ THE FULL sheet — rows beyond the first 8 exist, incl. the "DB-1-5" replacement rows.)
- Coordinate transform (use EXACTLY this, same as `scripts/ent_analysis_2025.py`):
  WGS84 → EPSG:26910 (pyproj, always_xy) → subtract (easting−X).median() / (northing−Y).median()
  from `tblReceiver` → local array meters (array ≈ X 437–540, Y 37–140, Z −12…−2).
- GPS time reconstruction (copy verbatim from the `Data` class in `scripts/ent_analysis_2025.py`):
  1 Hz per-minute blocks; `gtime = minute_start + 60 − block_count + arange(n)`.
- Verified baseline for these same ENT-01..05 lines (source of truth:
  `output/2025_review/jsats3d_2025_tagdrag_ent_anchorZOI03/item12_pipeline_summary.csv`,
  "3D Kalman filtered" rows = root B; verified 2026-10-21 after the F1 clock-fix re-run;
  weighted median over all lines/tags = 3.95 m):

  | Line   | FC36 median/p90 m | FFD3 median/p90 m |
  |--------|-------------------|-------------------|
  | ENT-01 | 8.62 / 12.5       | 10.67 / 16.8      |
  | ENT-02 | 5.63 / 11.4       | 6.37 / 16.0       |
  | ENT-03 | 3.81 / 16.0       | 3.41 / 8.5        |
  | ENT-04 | 2.62 / 4.4        | 2.41 / 9.5        |
  | ENT-05 | 3.13 / 7.7        | 2.95 / 9.3        |

  (2D root-B Kalman medians, same folder with `_2d` suffix: ENT-01 6.11/6.32,
  ENT-02 3.66/3.32, ENT-03 3.49/2.55, ENT-04 3.27/2.23, ENT-05 4.41/4.16 — FC36/FFD3.)
  Your drag-line QC numbers must land in this neighborhood; a big deviation
  means a bug (usually time-basis or transform).

**Example-fish work (NOT yet positioned — full pipeline run required):**

- Tags: **26E4, 9F8C, 1D5A, B5ED** (chinook juveniles, Cispus/Clear Fork releases).
  PM's stated fates: 26E4 → entered CFNSC; 9F8C → turbine routes; 1D5A → spillway flumes;
  B5ED → returned to reservoir. (PM selected early-year fish = near-isothermal, short residence.)
- Release metadata `...\3_PTAGIS_datasets\released_v0.csv` (column `Acoustic Tag Value`,
  case-mixed): **1D5A** released 2025-06-17 11:33; **9F8C** 2025-06-24 10:35;
  **B5ED** 2025-06-24 10:35 — all from "COWLR2 - Cispus River to Clear Fork (km 135-202)".
  ⚠️ **26E4 is NOT in released_v0.csv** — verify it exists in the raw data (step 2.1); if it
  has no detections at all, STOP the fish workstream and escalate (do not improvise).
- Raw data: `K:\Jobs\5662\001\Data\DataTrans\2025_Data\raw_data\` — per-date folders
  (20250618_Receivers, 20250619_Lotek, … 487 files). Fish detections begin ≥ 2025-06-17.
- No existing DB contains these four tags (verified 2026-10-22 against all local DBs).

## 2. Workstream 2 first (longest pole): example-fish pipeline run

Do this BEFORE the drag figures so the multi-hour job runs while you build drag figures.

### 2.1 Verify tags in raw (read-only, ~10 min)
Scan the raw files for the four tag codes (case-insensitive; raw uses the acoustic values,
e.g. `9f8c`, `b5ed`). Confirm each tag has detections and note first/last detection per tag.
If 26E4 is absent → escalate and continue with the other three (note in journal + README).

### 2.2 New config `config/run_data_earlyfish_2025.toml`
Copy `config/run_data_chinook_2025.toml` and change:
- `output_db = 'output\jsats3d_2025_earlyfish.db'`
- `tags = ['26E4', '9F8C', '1D5A', 'B5ED']`
- `start = '2025-06-17 00:00:00'`, `end = '2025-07-31 23:59:59'` (same sync window; extend
  `end` only if 2.1 shows detections after Jul 31)
- keep `adopt_deep_z = 'config'` (field team 2026-10-22: tape-measured Z for ZOI01-03),
  `deep_solution`, receivers lists, `hobo_dir`, dbscan block — identical to the chinook config.
- keep `fixed_z_2d_by_tag = {}` (fish have no fixed depth — the 2D fixed-Z step is a no-op).
- Add a header comment: purpose, PM request 2026-10-22, near-isothermal early-year fish.

### 2.3 Run the full build (two commands, background, log `output/run_earlyfish_2026_10_22.log`)
Command 1 (parse raw → DB):
    python scripts/run_data.py config/run_data_earlyfish_2025.toml
Command 2 (positioning chain, paper steps 1b–11; skip the three `plot_drag*` steps —
they need the drag-GPS truth and are drag-specific; the fish figures come from WS1):
    python scripts/tagdrag_2025_pipeline.py config/run_data_earlyfish_2025.toml --only prepare metronome clock_surface deep_beacons deep_positions adopt_deep deep_clocks fish_multipath speed_of_sound positions export positions_2d positions_dbscan
This is the same two-command pattern that built `jsats3d_2025_chinook.db` (see
`.ai_journal/session/2026-10-19_chinook_pipeline_run.md` for the exact precedent).
Expect **hours** (that comparable 4-tag 45-day build is 14 GB). `positions_2d` is a
no-op for fish (no fixed depths configured) — expected; the fish product is 3D root B.
Journal the commands + start time. Do NOT poll — continue to Workstream 1; check only
when the background job notifies completion.
Gate failures (duplicate clock groups, no solutions, etc.) → stop, journal, escalate.

### 2.4 On completion: fish QC data
For each tag, from the new DB (root B):
- `tblPositions_DengDBSCAN` rows with the local-coord water-column filter
  (X 420–560, Y 20–160, Z −30…0 — same bounds as `scripts/chinook_analysis_2025.py`),
  ordered by ToA (absolute time), with t0 = ToA − T01 as the transmission time.
- Per-transmission consensus track (median of root-B solutions per transNo within ≤10 m,
  ≥2 distinct receiver sets — same logic as `legacy_receiver_set_consensus` in
  `scripts/ent_analysis_2025.py`) for the "recommended" product.
- Jump check: speed between consecutive consensus fixes; flag > 3 m/s (chinook burst swim
  is ~2–3 m/s) as candidate multipath/epoch errors → `qc/fish_jump_checks.csv`.
- Fate plausibility: for each tag, report first/last fix (t, X, Y, Z) in the README and
  state whether the track is consistent with the PM's stated fate (qualitative; the
  CFNSC/turbine/spillway regions are identifiable from the array map + the CFNSC_Drags
  sheet). Do NOT force a classification — report observations.

## 3. Workstream 1: drag figures + CSVs (data is ready — build now)

Create ONE persistent, well-documented generator script
`scripts/drag_test_figures_2025.py` (NOT a pipeline step; never called by the pipeline;
header comment: "Client figure generator for the 2026-10-22 PM request; re-run as methods
improve; inputs: ENT DB + GPS csv + testing-sheets xlsx"). The PM explicitly wants a script
they can re-run as methods refine — this is the ONE script from this task that remains
(deliverable tool, like `chinook_analysis_2025.py`); every other helper is throwaway.
It must be idempotent, read-only on all inputs, and print a manifest of what it wrote.

### 3.1 Products per drag category (all root B)

**A. Static holds (Forebay_StaticHolds)** — the headline QC product:
- For each valid hold (FBY-S-1..5, 7, DB-1-5 rows per the sheet): transform the sheet
  Lat/Lon to local XY; take tag fixes (3D root-B and 2D root-B separately) with t0 inside
  [Start Time, End Time]; compute n_fixes, median XY error, p90 XY error vs the hold point.
- Figure per hold: plan view, hold point as black star, receiver markers, tag fixes as
  colored points (color = time) + connecting line, 10 m / 20 m error circles, title
  "FBY-S-1 · 2025-06-10 09:58:02–09:59:02 · FC36 & FFD3". One figure per hold, both tags
  in one figure (FC36 blue points, FFD3 orange points; legend).
- Combined figure: all holds on one plan view, each hold's mean position as a marker
  offset from its true point by the error vector (classic "error arrow" plot).
- `qc/static_hold_errors.csv`: hold_id, tag, product(3D/2D), n_fixes, median_xy_m,
  p90_xy_m, median_dx_m, median_dy_m. **Pass bar: median XY < 10 m for most holds**
  (near-isothermal early season); report anything worse prominently.

**B. Drag tracks (CFNSC_Approach_Drags = ENT-01..05, Forebay_Random_Drags = FBY-D-1..4):**
- Per test ID: GPS track reconstructed per §1 (black line + small black points, 1 Hz),
  solution points (root-B per-transmission consensus, color = time, viridis + colorbar)
  connected by a line, receiver markers (labeled), start/end markers, title with
  test ID + date/time + tags, equal aspect, grid, legend. One figure per test ID with
  BOTH tags (FC36 blue-based, FFD3 orange-based color ramps) — plus one combined
  category figure (all IDs of the category, GPS thin, solutions as points only).
- Plan view = the deliverable ("a plan view of each ID"). Add a small 3D inset or a
  separate side view (Y vs Z) per ID as a bonus — optional, low effort.
- `drag_line_errors.csv`: line, tag, product, n_scored, median_xy_m, p90_xy_m
  (scored = fixes within the GPS track time span; same method as ent_analysis §1).
  These must agree with the §1 verified baseline for ENT-01..05 (± a meter).

**C. Interactive (plotly HTML):**
- One HTML per category (3 files): plan view, GPS black line, solution points colored by
  time with hover text (tag, t, X, Y, Z, receiver set, n_combinations), receiver markers,
  dropdown or legend to toggle tags, static-hold true points as stars (static category).
  Self-contained single file (inline JSON; no CDN dependence is preferred — if plotly
  requires `include_plotlyjs='inline'`, file size ~3–5 MB is fine).

**D. Client CSVs — "all the positions of the tags" (one CSV per tag, 2 files):**
`drags/<category>/<TAG>_positions.csv` with columns:
`tag, category, test_id, t_iso_utc, X_m, Y_m, Z_m, product, root, in_hull, receiver_set, n_combinations`
where `product` ∈ {`solution` (every root-B receiver-set solution row, 3D and 2D — include
both with a `dim` column 3D/2D), `consensus` (per-transmission median track)`.
Also copy `drags/<category>/<TAG>_gps_track.csv` (t, x, y local) from the GPS csv.
Round to 3 decimals; ISO-8601 times on the UTC-encoded wall basis (document the basis in
the README — receivers log PDT, ZOI05 logs UTC, DB `seconds` = wall time encoded as UTC).

### 3.2 Example-fish figures (after WS2 completes)
Per tag (`example_fish/<TAG>/`):
- **Plan view PNG**: consensus track as line + ALL points (PM: "points added so we can see
  when the fish jumps"), color scale = time (viridis, colorbar), release-location note,
  receiver markers, title with tag + fate (PM's stated fate as a label, flagged "PM-stated").
  Color-by-Z variant as second PNG (depth scale) — the "color scale like before" ask.
- **3D PNG** (X, Y, Z) same style (Z range ~ 0…−30 m).
- **Interactive HTML** (plotly, 3D scatter + time slider or frame buttons; hover = t, XYZ).
- **Animation** (the "extra time" ask — do it if the others are done): matplotlib
  `FuncAnimation` → `track.mp4` (or GIF if ffmpeg unavailable: save as `.gif` via
  PillowWriter; note the format in README). Frames = consensus fixes subsampled to ≤200
  frames; moving point + fading trail + fixed receiver context.
- `<TAG>_positions.csv`: same schema as drag CSVs (3D root B only — no 2D product for
  fish; product = solution/consensus; no GPS).

## 4. Output layout & README

```
output/pm_figures_2026-10-22/
  README.md            <- INDEX: every file, one line each, the key QC metrics table,
                          coordinate/time-basis notes, how to re-run the generator
  drags/
    CFNSC_Approach_Drags/   (per-ID PNGs, combined PNG, category HTML, FC36/FFD3 CSVs)
    Forebay_StaticHolds/    (per-hold PNGs, error-arrow PNG, category HTML, CSVs)
    Forebay_Random_Drags/   (as CFNSC)
  example_fish/
    26E4/ 9F8C/ 1D5A/ B5ED/  (plan PNG, 3D PNG, depth-color PNG, HTML, mp4/gif, CSV)
  qc/
    static_hold_errors.csv, drag_line_errors.csv, fish_jump_checks.csv
```
README must state: positions are root-B; static-hold median errors (the headline number);
drag-line median/p90 table; fish first/last fixes + fate observations; the time basis;
"generated by scripts/drag_test_figures_2025.py, re-runnable".

## 5. Validation gates (do all; journal each result)

1. **Transform sanity:** known receiver easting/northing → local round-trip matches
   X_t/Y_t within 0.1 m (catches pyproj/axis mistakes).
2. **ENT baseline agreement:** drag_line_errors for ENT-01..05 within ~1 m of the §1
   baseline table (3D root-B Kalman medians, per tag).
3. **Static holds:** median XY error per hold (pass bar §3.1A); report the worst hold.
4. **Row-count integrity:** every CSV's `solution` rows == DB row count (root B, same
   filters); `consensus` rows == distinct transNo count.
5. **Files non-trivial:** every PNG > 30 KB, HTML > 100 KB, animation > 10 frames.
6. **Fish:** jump-check flags listed (not hidden); fate observations in README; tags with
   < 20 fixes get a "sparse" flag in README.
7. **Re-run check:** run the generator a second time → identical manifest (idempotent).

## 6. Cleanup & journaling (System_Prompt.txt rules)

- **Delete after use:** all `_`-prefixed throwaway scripts (raw-scan probe, ad-hoc checks),
  temp files. Keep: `scripts/drag_test_figures_2025.py` (the re-runnable generator),
  `config/run_data_earlyfish_2025.toml`, the new fish DB (data product — note size),
  everything under `output/pm_figures_2026-10-22/`.
- **Journal** `.ai_journal/session/2026-10-22_pm_figures.md`: data inventory findings
  (incl. 26E4 status), WS2 command + timings + gates, every QC metric with the gate result,
  files kept/deleted, escalations. No parameter changes were made to the pipeline for this
  task (state that explicitly).
- **Escalate (stop + journal + report) if:** 26E4 absent from raw; any pipeline gate
  failure; static-hold errors > 25 m (would indicate a time-basis or transform bug, not a
  method issue — re-check against the ENT baseline before blaming physics); raw-folder
  gaps for dates you need.

## 7. Report-back format (your final message)

```
PM figures 2026-10-22 — DONE / DONE-WITH-CAVEATS / BLOCKED
Headline QC: static holds median XY = X m (n holds), drag lines median/p90 = Y/Z m,
fish: first/last fixes per tag + fate observation, jump flags = N
Files:
  <full path to README.md>
  <every PNG/HTML/CSV/MP4 path, grouped by category>
DB: output/jsats3d_2025_earlyfish.db (size), log: output/run_earlyfish_2026_10_22.log
Journal: .ai_journal/session/2026-10-22_pm_figures.md
Escalations: <none | list>
```
The requesting engineer forwards the file links for review — make every path absolute.

## 8. Reviewer's checklist (what will be checked on your report-back)

- Static-hold errors table present and plausible; ENT baseline numbers reproduced.
- Plots: GPS vs solutions visually consistent (no 1-hour offsets, no mirrored axes,
  receivers in the right places, colorbars labeled).
- CSVs open cleanly; time basis documented; root B only; 3D+2D both present.
- Fish fates observations match the PM's stated fates (or the mismatch is flagged).
- No leftover throwaway scripts; no pipeline files modified; K: untouched.
