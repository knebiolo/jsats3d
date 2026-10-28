# 2026-10-25: Fish Figures — Receivers for Spatial Reference + 3D Animations

**Session:** Continuation of the 2026-10-24 early-fish figure work (see
`2026-10-24_earlyfish_fix_and_figures.md`).
**Request:** Show receivers (ZOI + CFD) on the plots and animations so the client has
a spatial reference; add a 3D animation; journal everything; remove diagnostics.

## TL;DR
1. **Receivers restored on every figure** — all 20 (ZOI01-11, CFD01-09) as small gray
   x marks with labels: plan views (time + depth), static 3D, 2D track GIF, 3D track GIF.
   This time they read as reference, not clutter, because the hairball connecting lines
   that made them noisy were already removed.
2. **New deliverable: `<TAG>_track3d.gif`** — 3D animated track (receivers as static
   reference + growing colored track + red current-position dot + timestamp, 10 fps,
   PillowWriter). 150 / 150 / 62 / 150 frames for 26E4 / 1D5A / 9F8C / B5ED.
3. **Renamed** `scripts/_fish_figures_2025.py` -> `scripts/fish_figures_2025.py`
   (it is now a persistent deliverable tool; project convention: no underscore prefix).
4. **Files:** `example_fish/` went 21 -> 25 files (4 tags x 6 each + fish_qc.csv):
   plan_time.png, plan_depth.png, 3d.png, track.gif, **track3d.gif** (new), positions.csv.
5. **QC unchanged** — data and the surface-gated item12 chain are identical to the
   2026-10-24 final pass; only rendering changed. fish_qc.csv numbers still hold
   (max speeds 1.82 / 1.89 / 11.5* / 4.86 m/s; 0 / 0 / 4 / 9 jump flags; 100% underwater).

## Changes in scripts/fish_figures_2025.py
- Re-added `tblReceiver` query (Rec_ID, X_t, Y_t, Z_t).
- New helpers `draw_receivers_2d(ax, rec, zorder)` and `draw_receivers_3d(ax, rec)`:
  gray x (edgecolor k, size 6) + label (fontsize 6.5, color 0.35) at each receiver.
- plan_view and static 3D: receivers drawn with the track (zorder 4 in 2D).
- 2D GIF: receivers drawn at zorder 1 (background, below the growing track + head).
- 3D GIF: new FuncAnimation block — fixed axis limits (data + receiver extents,
  Z +/- 3 m margin), `line3.set_data/set_3d_properties` for the growing track,
  `head3` red dot, `text2D` timestamp; same frame indexing as the 2D GIF.
- Docstring updated (product list + receiver note; usage line after rename).

## Docs
- `output/pm_figures_2026-10-22/README.md` — example-fish section: figure bullets now
  describe the 2D + 3D animations and the receiver reference; file list adds
  `{tags}_track3d.gif`; re-run line and footer updated to the new script name
  (regenerated 2026-10-25 00:30).

## Verification (visual, extracted frames)
- 26E4 plan_time: labeled ZOI/CFD x's + time-colored track + start/end triangles — clean.
- 9F8C 2D GIF last frame: head dot at the final position (443, 72 = last_obs), track
  drawn from the start (516, 80) — chronology correct.
- 26E4 / 1D5A 3D GIF frames: receivers visible as reference; 1D5A final frame shows the
  full 17 h track including the ~-55 m dive; red head at the final position.

## Cleanup
- No diagnostic/throwaway scripts remain in `scripts/` (the 2026-10-24 pass already
  removed all `_probe_*` scripts; this pass renamed the last underscore file to its
  persistent name). `scripts/` now holds 11 persistent tools.
- `output/` keeps only the two legitimate run logs: `_earlyfish_build.log`
  (2026-10-23 parse/build) and `_earlyfish_pipeline.log` (2026-10-24 full pipeline run) —
  the official record of the ~2.5 h build; referenced by the 10-24 journal.
- Temp GIF-frame extractions (used for visual verification) deleted from
  `%TEMP%\opencode`.

## Status
- **Figures:** 25 files, client-ready: receivers on every figure + animation for
  spatial reference; 2D + 3D animations per tag; QC table unchanged.
- **Open items (unchanged):** 26E4 fate, ZOI01-03 Deng2D X/Y, CFD01/ZOI11 clock curves,
  ZOI08 RTK residual, 9F8C's short real detection window.
