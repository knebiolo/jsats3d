"""Generator for the PM's 2026-10-22 example-fish figures (early fish, 4 tags).

Persistent deliverable tool (NOT a pipeline step; never invoked by run_data.py or
_2025_pipeline.py). Re-run as methods improve:

    python scripts/fish_figures_2025.py [--db DB_PATH] [--out OUT_DIR]

Input (all read-only):
    output/jsats3d_2025_earlyfish.db (tblPositions_Deng, tblReceiver)

Products (under output/pm_figures_2026-10-22/example_fish/):
    <TAG>_plan_time.png    plan view, track points colored by time
    <TAG>_plan_depth.png   plan view, colored by depth (Z)
    <TAG>_3d.png           3D scatter colored by time
    <TAG>_track.gif        2D animated track (PillowWriter; no ffmpeg on this machine)
    <TAG>_track3d.gif      3D animated track (receivers + growing track + red head)
    <TAG>_positions.csv    client schema, products: consensus | kalman
    fish_qc.csv            per-tag summary: span, max speed, >3 m/s jump flags

All figures show the 20 receivers (ZOI01-11, CFD01-09) as small gray x marks with
labels -- the client's spatial reference for the array layout.

Track product = the verified item12 chain (identical to drag_test_figures_2025.py,
2026-10-21 baseline gate), preceded by a physical plausibility gate:
    root-B Deng 3D solutions
      -> drop Z > 0 (fish are underwater; Z = 0 is the water surface)
      -> drop Z < -16 m (fish cannot be below the bay bottom)
      -> legacy_receiver_set_consensus (|XYZ| < 1000 m, 10 m radius, >=2 receiver sets)
      -> smooth_track (jump 10 m / window 11 / smooth 3, gap 20 s)
      -> kalman_filter_track (2.0, 1.0, 0.5, gap 20 s)

The 2026-10-24 first pass plotted raw per-transmission medians of ALL root-B
solutions; diverged-solution outliers (raw extents up to ~872 km in Z for 26E4
against a ~100 m receiver box) produced phantom speeds (max 53,351 m/s for B5ED).
The chain above is what rejects them. The surface gate additionally removes the
smaller-scale above-surface mirror solutions (27-51% of raw root-B rows per tag)
that the chain's 1 km gate does not catch.

The _positions.csv files contain only the final validated products (consensus and
kalman). Raw Deng solutions are not included -- they are processing artifacts, not
part of the final dataset. Rejection counts are in fish_qc.csv.

Depth floor: the same flat-array geometry produces mirror/parallax solutions BELOW
the bottom (documented 2026-10-02: "Z of -25.7 m is deeper than the bottom ... the
vertical component is poorly constrained by this nearly flat reference set"). The
floor is project-interchangeable (resolve_depth_floor in drag_test_figures_2025.py):
    1. CLI --depth-floor-m
    2. run file [figures] depth_floor_m
    3. auto: deepest receiver Z_t minus [figures] depth_floor_margin_m (default 4 m)
At this site the deepest measured TOTAL bay depth is 13.3 m (43.5 ft at ZOI02,
config workbook, journals 2026-10-04 / LONG_TERM_CONTEXT.md) and the run file sets
depth_floor_m = -16.0 (4 m below the deepest hydrophone, the chinook 2026-10-21
audit margin). Without it, delivered tracks dived to -23 m (26E4), -23 m (9F8C),
-54 m (1D5A) and -33 m (B5ED) - below the bay floor.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ent_analysis_2025 import legacy_receiver_set_consensus  # noqa: E402  (reference consensus)
from drag_test_figures_2025 import (smooth_track, kalman_filter_track,  # noqa: E402  (item12 chain copies)
                                    resolve_depth_floor, run_file_from_status)

REPO = HERE.parent
DEFAULT_DB = REPO / "output" / "jsats3d_2025_earlyfish.db"
DEFAULT_OUT = REPO / "output" / "pm_figures_2026-10-22" / "example_fish"
JUMP_LIMIT_MS = 3.0
JUMP_MAX_DT_S = 120.0
# Depth floor (Z below which solutions are unphysical) is resolved at run time by
# resolve_depth_floor(): project-settable via CLI --depth-floor-m or run file
# [figures] depth_floor_m; auto default = deepest receiver Z minus margin.
TAG_ORDER = ("26E4", "9F8C", "1D5A", "B5ED")
TAG_COLORS = {"26E4": "#1f77b4", "9F8C": "#d62728", "1D5A": "#2ca02c", "B5ED": "#9467bd"}


def iso_utc(t):
    return pd.to_datetime(t, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def draw_receivers_2d(ax, rec, zorder=4):
    """Small gray x marks + labels: client spatial reference for the array."""
    for rid, row in rec.iterrows():
        ax.plot(row.X_t, row.Y_t, "kx", markersize=6, markeredgewidth=1.2, zorder=zorder)
        ax.annotate(rid, (row.X_t + 1.5, row.Y_t + 1.5), fontsize=6.5, color="0.35", zorder=zorder)


def draw_receivers_3d(ax, rec):
    for rid, row in rec.iterrows():
        ax.scatter([row.X_t], [row.Y_t], [row.Z_t], marker="x", color="k", s=25,
                   depthshade=False)
        ax.text(row.X_t + 1, row.Y_t + 1, row.Z_t, rid, fontsize=6.5, color="0.35")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--run-file", default=None,
                    help="run file for [figures] options (default: auto from <db>.pipeline_status.json)")
    ap.add_argument("--depth-floor-m", type=float, default=None,
                    help="Z (m) below which solutions are unphysical (default: [figures] depth_floor_m, "
                         "else deepest receiver Z minus margin)")
    args = ap.parse_args()
    db, out = Path(args.db), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_file = Path(args.run_file) if args.run_file else run_file_from_status(db)
    floor_m, floor_src = resolve_depth_floor(db, args.depth_floor_m, run_file)
    print("depth floor: %.2f m (%s)%s" % (floor_m, floor_src,
          ("  run file: %s" % run_file) if run_file else ""))
    con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)
    rec = pd.read_sql("select Rec_ID, X_t, Y_t, Z_t from tblReceiver", con).set_index("Rec_ID")
    p = pd.read_sql("select * from tblPositions_Deng where comment = 'solution found'", con)
    p["t0"] = pd.to_numeric(p.ToA, errors="coerce") - pd.to_numeric(p.T01, errors="coerce")
    for c in ("X", "Y", "Z", "t0"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[(p.solution == "B") & p[["X", "Y", "Z", "t0"]].notna().all(axis=1)].copy()
    tags = [t for t in TAG_ORDER if (p.Tag_ID == t).any()]
    extra = sorted(set(p.Tag_ID.unique()) - set(TAG_ORDER))
    if extra:
        print("note: other root-B tags in DB (not requested, ignored): %s" % extra)
    if not tags:
        print("NO study-tag root-B positions in %s" % db.name)
        con.close()
        return 1

    qc_rows = []
    for tag in tags:
        print("=" * 64)
        print("TAG %s" % tag)
        raw = p[p.Tag_ID == tag].copy()
        # ---- physical plausibility gate: fish are underwater AND above the bay bottom.
        # Z = 0 is the water surface (receivers at Z -2..-12 m; tape-measured deep
        # Z 10.3-12.4 m; deepest measured TOTAL depth 13.3 m at ZOI02). Raw root-B
        # solutions show 27-51% above-surface mirror artifacts near the array edge
        # and below-bottom mirror/parallax artifacts (flat array; journal 2026-10-02);
        # dropping both before consensus keeps them from pulling the per-transmission
        # median. The raw rows are NOT written to the client CSV -- they are processing
        # artifacts. Rejection counts are in fish_qc.csv.
        n_above = int((raw.Z > 0).sum())
        n_below = int((raw.Z < floor_m).sum())
        sub = raw[(raw.Z <= 0) & (raw.Z >= floor_m)].copy()
        # ---- verified item12 chain: consensus -> smooth -> Kalman
        g = legacy_receiver_set_consensus(sub)
        gs, n_rej = smooth_track(g, jump_limit_m=10.0, jump_window=11, smooth_window=3)
        gk, _ = kalman_filter_track(gs, process_noise=2.0, measurement_noise=1.0,
                                    z_noise=0.5, max_gap_s=20.0)
        if gk.empty:
            print("  no filtered track for %s (n_solutions=%d, n_consensus=%d, smooth_rejected=%d)"
                  % (tag, len(sub), len(g), n_rej))
            continue
        # ---- QC: speed / jumps on the final (Kalman) product
        dt = gk.t0.diff()
        d = np.hypot(gk.x.diff(), gk.y.diff())
        v = np.where((dt > 0) & (dt < JUMP_MAX_DT_S), d / dt, np.nan)
        jump_idx = gk.index[(dt > 0) & (dt < JUMP_MAX_DT_S) & (d / dt > JUMP_LIMIT_MS)]
        summary_parts = []
        for i in jump_idx:
            prev, cur = gk.iloc[i - 1], gk.iloc[i]
            summary_parts.append("%s @%.1f,%.1f->%.1f,%.1f (%.0f m in %.0f s)" % (
                pd.Timestamp(cur.t0, unit="s", tz="UTC").strftime("%m-%d %H:%M"),
                prev.x, prev.y, cur.x, cur.y,
                float(np.hypot(cur.x - prev.x, cur.y - prev.y)), float(cur.t0 - prev.t0)))
        t0, t1 = gk.t0.iloc[0], gk.t0.iloc[-1]
        span_h = (t1 - t0) / 3600.0
        qc_rows.append(dict(tag=tag, n_solutions=int(len(raw)), n_above_surface=int(n_above),
                            n_below_floor=int(n_below), depth_floor_m=round(float(floor_m), 2),
                            n_consensus=int(len(g)), n_smooth_rejected=int(n_rej),
                            n_kalman=int(len(gk)),
                            first_obs=iso_utc(pd.Series([t0])).iloc[0],
                            last_obs=iso_utc(pd.Series([t1])).iloc[0], span_h=round(span_h, 1),
                            first_X_m=round(float(gk.x.iloc[0]), 1), first_Y_m=round(float(gk.y.iloc[0]), 1),
                            last_X_m=round(float(gk.x.iloc[-1]), 1), last_Y_m=round(float(gk.y.iloc[-1]), 1),
                            max_speed_m_s=round(float(np.nanmax(v)) if np.isfinite(v).any() else np.nan, 2),
                            z_min_m=round(float(gk.z.min()), 2), z_max_m=round(float(gk.z.max()), 2),
                            n_jumps_gt_3ms=int(len(jump_idx)),
                            jump_summary="; ".join(summary_parts)[:500]))
        print("  solutions=%d (above-surface dropped=%d, below-floor<%.2f m dropped=%d)  consensus=%d  smooth_rejected=%d  kalman=%d"
              % (len(raw), n_above, floor_m, n_below, len(g), n_rej, len(gk)))
        print("  span=%.1f h  first=%s (%.0f,%.0f)  last=%s (%.0f,%.0f)  max_v=%.2f m/s  jumps>%g m/s: %d  Z range [%.1f, %.1f] m"
              % (span_h, qc_rows[-1]["first_obs"], gk.x.iloc[0], gk.y.iloc[0],
                 qc_rows[-1]["last_obs"], gk.x.iloc[-1], gk.y.iloc[-1],
                 qc_rows[-1]["max_speed_m_s"] or 0, JUMP_LIMIT_MS, len(jump_idx),
                 float(gk.z.min()), float(gk.z.max())))

        norm_t = (gk.t0 - t0) / max(t1 - t0, 1e-9)

        def plan_view(ax, cmap, values, title):
            ax.scatter(gk.x, gk.y, c=values, cmap=cmap, s=18, alpha=0.8, zorder=3)
            draw_receivers_2d(ax, rec)
            ax.plot(gk.x.iloc[0], gk.y.iloc[0], "g>", markersize=11, zorder=5)
            ax.plot(gk.x.iloc[-1], gk.y.iloc[-1], "r<", markersize=11, zorder=5)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(title)
            ax.set_aspect("equal")

        fig, ax = plt.subplots(figsize=(10, 8))
        plan_view(ax, "viridis", norm_t,
                  "%s - plan view, points colored by time (green start, red end)" % tag)
        fig.colorbar(ax.collections[0], ax=ax, label="time (fraction of span)")
        fig.savefig(out / ("%s_plan_time.png" % tag), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        plan_view(ax, "twilight", gk.z, "%s - plan view, points colored by depth Z (m)" % tag)
        fig.colorbar(ax.collections[0], ax=ax, label="Z (m)")
        fig.savefig(out / ("%s_plan_depth.png" % tag), dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 8))
        ax3 = fig.add_subplot(111, projection="3d")
        sc = ax3.scatter(gk.x, gk.y, gk.z, c=norm_t, cmap="viridis", s=12, alpha=0.8)
        draw_receivers_3d(ax3, rec)
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_zlabel("Z (m)")
        ax3.set_title("%s - 3D, points colored by time" % tag)
        fig.colorbar(sc, ax=ax3, label="time (fraction of span)")
        fig.savefig(out / ("%s_3d.png" % tag), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- animated track (GIF): growing colored track + red head + timestamp
        n_frames = min(150, max(30, len(gk)))
        idx = np.unique(np.linspace(0, len(gk) - 1, n_frames).round().astype(int))
        fig, ax = plt.subplots(figsize=(9, 7.5))
        draw_receivers_2d(ax, rec, zorder=1)
        trail, = ax.plot([], [], "-", color=TAG_COLORS.get(tag, "C0"), linewidth=1.5,
                         alpha=0.85, zorder=2)
        head, = ax.plot([], [], "o", color="red", ms=8, zorder=3)
        tlab = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
                       bbox=dict(facecolor="white", alpha=0.7))
        xlo = min(gk.x.min(), rec.X_t.min()) - 15
        xhi = max(gk.x.max(), rec.X_t.max()) + 15
        ylo = min(gk.y.min(), rec.Y_t.min()) - 15
        yhi = max(gk.y.max(), rec.Y_t.max()) + 15
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.set_title("%s - track over time (red = current position)" % tag)

        def frame(i):
            k = idx[i]
            trail.set_data(gk.x.iloc[:k + 1], gk.y.iloc[:k + 1])
            head.set_data([gk.x.iloc[k]], [gk.y.iloc[k]])
            tlab.set_text(pd.Timestamp(gk.t0.iloc[k], unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"))
            return trail, head, tlab

        from matplotlib.animation import FuncAnimation, PillowWriter
        anim = FuncAnimation(fig, frame, frames=len(idx), interval=90, blit=False)
        anim.save(str(out / ("%s_track.gif" % tag)), writer=PillowWriter(fps=10))
        plt.close(fig)
        print("  gif: %s frames" % len(idx))

        # ---- 3D animated track (GIF): receivers + growing track + red head
        fig = plt.figure(figsize=(10, 8))
        ax3 = fig.add_subplot(111, projection="3d")
        draw_receivers_3d(ax3, rec)
        line3, = ax3.plot([], [], [], "-", color=TAG_COLORS.get(tag, "C0"),
                          linewidth=1.5, alpha=0.85)
        head3, = ax3.plot([], [], [], "o", color="red", ms=9)
        tlab3 = ax3.text2D(0.02, 0.98, "", transform=ax3.transAxes, va="top", fontsize=10,
                           bbox=dict(facecolor="white", alpha=0.7))
        ax3.set_xlim(min(gk.x.min(), rec.X_t.min()) - 15, max(gk.x.max(), rec.X_t.max()) + 15)
        ax3.set_ylim(min(gk.y.min(), rec.Y_t.min()) - 15, max(gk.y.max(), rec.Y_t.max()) + 15)
        ax3.set_zlim(min(gk.z.min(), rec.Z_t.min()) - 3, max(gk.z.max(), rec.Z_t.max()) + 3)
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_zlabel("Z (m)")
        ax3.set_title("%s - 3D track over time (red = current position)" % tag)
        xs_all, ys_all, zs_all = (gk.x.to_numpy(), gk.y.to_numpy(), gk.z.to_numpy())

        def frame3d(i):
            k = idx[i]
            line3.set_data(xs_all[:k + 1], ys_all[:k + 1])
            line3.set_3d_properties(zs_all[:k + 1])
            head3.set_data([xs_all[k]], [ys_all[k]])
            head3.set_3d_properties([zs_all[k]])
            tlab3.set_text(pd.Timestamp(gk.t0.iloc[k], unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"))
            return line3, head3

        anim3 = FuncAnimation(fig, frame3d, frames=len(idx), interval=90, blit=False)
        anim3.save(str(out / ("%s_track3d.gif" % tag)), writer=PillowWriter(fps=10))
        plt.close(fig)
        print("  gif3d: %s frames" % len(idx))

        # ---- client CSV (consensus + kalman only; raw solutions are NOT included)
        con_df = pd.DataFrame({
            "tag": tag, "t_iso_utc": iso_utc(g.t0), "t_s": g.t0.round(3),
            "X_m": g.x.round(3), "Y_m": g.y.round(3), "Z_m": g.z.round(3),
            "product": "consensus", "root": "B", "dim": "3D", "in_hull": "",
            "transNo": g.transNo, "r0": "", "r1": "", "r2": "", "r3": "",
            "receiver_set": "", "receiver_sets": g.receiver_sets, "n_combinations": g.n_combinations})
        kal_df = pd.DataFrame({
            "tag": tag, "t_iso_utc": iso_utc(gk.t0), "t_s": gk.t0.round(3),
            "X_m": gk.x.round(3), "Y_m": gk.y.round(3), "Z_m": gk.z.round(3),
            "product": "kalman", "root": "B", "dim": "3D", "in_hull": "",
            "transNo": gk.transNo, "r0": "", "r1": "", "r2": "", "r3": "",
            "receiver_set": "", "receiver_sets": "", "n_combinations": ""})
        pd.concat([con_df, kal_df], ignore_index=True, sort=False).to_csv(
            out / ("%s_positions.csv" % tag), index=False)
        print("  csv: %s_positions.csv (%d consensus + %d kalman rows)"
              % (tag, len(con_df), len(kal_df)))

    pd.DataFrame(qc_rows).to_csv(out / "fish_qc.csv", index=False)
    con.close()
    print("=" * 64)
    print("wrote %d tags to %s" % (len(tags), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
