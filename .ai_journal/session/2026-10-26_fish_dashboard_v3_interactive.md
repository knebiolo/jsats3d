# 2026-10-26 — Fish Dashboard v3 (filterable) build, debugging, and verification

## Request (owner, verbatim intent)
Upgrade the fish dashboard to a **filterable** one: select tag type / species (e.g. only
chinook), restrict to a time period, select by **water-column temperature** (with time), and
**combine** those filters. Likes the 3D view but "all those dots all at once is kind of hard
to conceptualize" — figure out **one tag or all tags in 3D**. Data-wise: tag name, species,
release date, recapture date if possible — "items a fisheries biologist / hydropower project
manager would need to see". "Go ahead and build and test." "Make sure we can see the labels
clearly and they don't overlap anything." "Simple enough anyone who looks at it can make
deductions about what is going on."

Constraints (carried): single self-contained Plotly HTML, no server; one documented time
basis for all axes; tag-generic so the full-season / chinook DB can be dropped in later;
DB read-only; pipeline + existing deliverables untouched; journal every session; delete
all throwaway scripts.

## What was built
`scripts/fish_dashboard_v3.py` (built by me directly, not the builder AI) →
`output/pm_figures_2026-10-22/fish_dashboard_v3/`:
- `dashboard.html` (6.2 MB, self-contained; Plotly 3.7.0 inlined via `plotly.offline`)
- `fish_roster.csv` — per-fish metadata: tag, species, release date/time, length, weight,
  first/last detection (local + UTC), fate, closest approach to CFD centroid
- `fish_tracks.csv` (7192 rows) — per-point track with `t_local`, `t_utc`, X/Y/Z, speed,
  water temp at fish depth, surface (0.5 m) temp
- `README.md`

Reuses the verified `fish_figures_2025.py` track chain (gate → consensus → smooth →
Kalman) with a **hard self-check** against verified `fish_qc.csv` counts — matched exactly:
26E4 1387, 9F8C 65, 1D5A 3413, B5ED 2327. Closest-approach = distance to CFD centroid,
reproduces v2 values exactly (8.5 / 9.5 / 14.1 / 11.9 m).

Features (all verified in browser):
- Combinable filters: species, tag (All + each), time window (datetime inputs + 2 range
  sliders, synced), water-temp min/max at a selectable reference depth
  (fish depth / 0.5 / 1.5 / 9 / 18 m), station select (DD_N, FBS, PDS, DD_S, BB_*),
  show-points toggle, reset.
- Clickable roster table (biologist-facing: species, release, length, weight, fate).
- 3D: line + endpoints + leader-line labels per selected fish; receiver markers;
  surface + floor planes; hover shows local + UTC (+7 h) + temp + speed.
  One fish by default; "All" shows all four with non-overlapping labels.
- Water-column panel: 5-min HOBO series (4 depths at selected station) downsampled 3×,
  WSEL on secondary axis, per-fish observation-window bands + tag annotations +
  release-time lines.
- Depth-vs-time panel: single tag → temp-coloured (colorbar); all tags → one line per fish.
- Plain-English notes footer (array is upstream of the dam; "passage" = leaving array
  footprint; 1D5A PIT recapture at CFD; 26E4 not in PTAGIS; Z bias; sparse early windows).

## Time basis (confirmed again this session)
Study basis is **PDT wall clock encoded as-UTC** (stored epoch rendered as UTC = wall
clock; true UTC = epoch + 7 h). Dashboard displays local (PDT) everywhere; hover shows
UTC = +7 h. Two non-obvious fixes were needed for the *axes* (below).

## Bugs found and fixed (this session)
1. **Plotly 3.7.0 `surfacecolor` gradient array** — `surfacecolor:[[0,"#7fb3d5"]]` on a
   `surface` trace throws `Cannot read properties of undefined (reading '0')` inside
   Plotly's surface colorization (minified `getImageData` path). This was the "mystery
   first render fails" — actually **render3d throwing deterministically, every time**,
   while the 2D panels were fine all along (isolated with a minimal 3D probe: text traces,
   markers+text, legend names OK; surface+gradient THROWS). Fix: plain color string.
   (Earlier in the same debugging arc: `line.color` arrays in 2D scatter also throw in
   3.7.0 — use `marker.color` + colorscale; and epoch seconds must be embedded as
   float64 b64, since Float32 quantizes ~128 s at 1.75e9.)
2. **pandas `datetime64[us]` resolution corrupted `wallclock_epoch`** —
   `pd.to_datetime` on the temp/WSEL strings returns `datetime64[us]` in this pandas
   version, so `astype("int64") // 10**9` (ns divisor) silently returned wrong values
   (t0 off by /1000, dt = 0). Impact: **every per-point fish temperature was wrong**
   (all points got the last-row 2025-09-17 profile via `np.interp` clamping at idx=inf),
   and the temp/WSEL curves sat at epoch ~0 on the axis. Fix: unit-safe conversion via
   `dt.to_numpy().astype("datetime64[ms]").astype(np.int64) / 1000.0`.
   Independently verified after fix: 26E4 first point temp 15.09 °C @ 6.31 m and surface
   15.38 °C match manual interpolation of the raw CSV.
3. **`parse_local_datetime` used `Timestamp.timestamp()`** — naive timestamps resolve
   against the *machine's* timezone (MDT here, 1 h off PDT). Fix: treat parsed fields as
   wall clock directly (as-UTC epoch), unit-safe.
4. **Plotly 3.x dropped the axis `tz` attribute** (verified: `tz`/`timezone`/template-tz
   all ignored; date axes always render in the viewer's local tz). Since stored epochs
   are wall-clock-as-UTC, axis labels were off by the viewer's offset. Fix: compensate at
   runtime — `ax(e) = e*1000 + new Date(...).getTimezoneOffset()*60000` — so the axes
   read study wall clock **regardless of where the file is opened** (Jun–Sep range is
   DST-stable).
5. **First-render robustness** (defensive, environment-dependent): per-step try/catch in
   `renderAll` (one failing panel must not block the others), timer-based retry loop
   (rAF does not fire in headless/hidden tabs), initial render gated on `load` +
   `document.fonts.ready` with a 1.5 s cap.

## Verification (browser, final build)
- Fresh load: **zero errors**, all three figures render without manual intervention.
- Counts exact: default 3413 (1D5A); All tags 7192 for 4 fish (1387+65+3413+2327).
- Axis alignment exact: 1D5A window renders 06-18 17:02 → 06-19 10:18 (verified local
  window); release line at 06-17 11:33; 14-day tick cadence 06-15 → 09-07 (2025);
  WSEL axis 06-05 → 09-18.
- Label non-overlap (owner requirement): "All" mode label anchor pairwise distances
  38.9–80.2 m — all far apart at 13 px font.
- Filter suite: All → 7192; temp 14.5–15.0 @ fish depth → 2824; @ 0.5 m ref → 2806;
  time window 06-18 16:00 → 06-19 04:00 on 1D5A → 139; species+tag combinations correct;
  reset restores defaults.
- Hover: local "06-27 23:19" / UTC "06-28 06:19" (+7 h) / "15.1 C" / "0.00 m/s".
- Screenshots unavailable in this headless browser (no visible window) — verified at
  DOM/state level instead (trace data, shape coordinates, SVG text, label positions).

## Data findings (for the biologist)
- Per-track water temp (corrected): 1D5A 14.1–15.0 °C (earliest fish, mid-June — coldest),
  26E4 14.5–15.4, 9F8C 15.8–16.1, B5ED 15.7–16.3 (released 06-24) — a sensible
  seasonal warming gradient across the cohort.
- Roster: 3 confirmed juvenile Chinook (1D5A, 9F8C, B5ED; PTAGIS released_v0), all
  released upstream into COWLR2 (km 135–202) and tracked downstream toward the dam.
  **26E4 not in PTAGIS** — species unconfirmed (labeled "26E4 - unconfirmed").
- 1D5A = **REACHED DAM**: PIT-antenna detection at CFD 06-19/2025 10:06 local, 12 min
  before its last acoustic detection (10:18). Passage not array-confirmed (array sits
  entirely upstream — all CFD receivers' Study Zone = "Forebay").
- 9F8C / B5ED / 26E4 = lost in forebay (last seen mid-channel, no further detections).

## Cleanup
- Deleted 8 throwaway explorers: `_tmp_explore_db.py`, `_tmp_explore2.py` … `_tmp_explore8.py`.
- Stopped the localhost:8791 HTTP server (was only for browser testing).
- Pipeline, figure scripts, and existing deliverables untouched (DB opened `mode=ro`).

## Notes / follow-ups
- The builder AI's claimed journal `2026-10-25_fish_dashboard_v2_corrected.md` is
  **still missing** from `.ai_journal/session/` (flagged in the previous session too).
  The v2 deliverables themselves were verified correct; only the journal never materialized.
- The v2 dashboard (`fish_dashboard/`) remains as-is; v3 lives alongside it.
- Drop-in readiness for the full-season chinook DB: the script auto-discovers the run
  file and depth floor, reads tags from the DB, and joins PTAGIS when present —
  species/release columns simply populate when available.
- Owner-gated open items (unchanged): gate drag client CSV consensus Z at export;
  full-season fate analysis; Z bias; CFD01/ZOI11 clock; ZOI08 RTK residual;
  ZOI01-03 Deng2D X/Y; `plot_drag_dbscan` done-check wart.
