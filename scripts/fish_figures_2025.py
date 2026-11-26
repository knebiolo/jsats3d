"""Generator for the PM's 2026-10-22 example-fish figures (early fish, 4 tags).

Persistent deliverable tool (NOT a pipeline step; never invoked by run_data.py or
_2025_pipeline.py). Re-run as methods improve:

    python scripts/fish_figures_2025.py [--db DB_PATH] [--out OUT_DIR]

Input (all read-only):
    output/jsats3d_2025_earlyfish.db (tblPositions_Deng, tblReceiver)

Products (under output/pm_figures_2026-10-22/example_fish/):
    <TAG>_plan_time.png    plan view, track points colored by time
    <TAG>_plan_depth.png   plan view, colored by depth (Z)
    <TAG>_depth_time.png   fish depth (Z) over time (track line + points)
    <TAG>_3d.png           3D scatter colored by time
    <TAG>_track.gif        2D animated track (PillowWriter; no ffmpeg on this machine)
    <TAG>_track3d.gif      3D animated track (receivers + growing track + red head)
    fish_positions.csv     ONE client CSV for all tags (Output_Columns.xlsx): one row per transmission heard after
                           multipath filtering; consensus position or blank when not positioned (n_solutions = 0)
    fish_qc.csv            per-tag summary: span, max speed, >3 m/s jump flags
    spatial_sd_grid.csv    1 m grid of the median per-transmission SD (x, y, z, horizontal) within 5 m; blank < 5 fixes
    spatial_sd_summary.csv SD percentiles per axis, all / inside / outside the receiver hull
    spatial_sd_<x|y|z|h>[_contour].png   SD maps (pixel and contoured)

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

fish_positions.csv holds the consensus position per transmission (the figures show the
smoothed/Kalman track). Raw Deng solutions are not included -- they are processing artifacts,
not part of the final dataset. Rejection counts are in fish_qc.csv.

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
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ent_analysis_2025 import legacy_receiver_set_consensus  # noqa: E402  (reference consensus)
from drag_test_figures_2025 import (smooth_track, kalman_filter_track,  # noqa: E402  (item12 chain copies)
                                    resolve_depth_floor, run_file_from_status, figures_section)

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
FISH_PRODUCTS = ("plan_time", "plan_depth", "depth_time", "3d", "gifs")


def iso_utc(t):
    return pd.to_datetime(t, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


# Client output (Output_Columns.xlsx, 2026-11-19): one CSV for all tags, one row per transmission (epoch) heard after
# multipath filtering. Epochs without a consensus position (solver failed, too few receivers, or rejected by the
# plausibility gate / receiver-set consensus) keep blank x/y/z/std and n_solutions = 0, so a fish that is heard but
# not positioned is not mistaken for one that left. Appended after the 11 columns: n_receivers (distinct receivers
# that heard the transmission) and status (positioned, or why not).
OUTPUT_FILE = "fish_positions.csv"
OUTPUT_COLUMNS = ["AT_code", "datetime_utc", "x_m", "y_m", "z_m", "x_std", "y_std", "z_std", "n_solutions",
                  "in_2Dconvexhull", "in_3Dconvexhull", "n_receivers", "status"]
# status of each row; the four non-positioned reasons are checked in this order
STATUS_POSITIONED = "positioned"
STATUS_FEW = "too few receivers"          # < 4 of the positioning receivers heard it (3D Deng needs 4)
STATUS_FAILED = "solver failed"           # no root-B Deng solution
STATUS_OUT = "out of water"               # every solution above the surface (Z > 0) or below the depth floor
STATUS_DISAGREE = "solutions disagree"    # in-water solutions, but < 2 receiver sets within 10 m (consensus)


def write_fish_positions(con, out, tags, consensus, floor_m):
    """Write OUTPUT_FILE. consensus: {tag: legacy_receiver_set_consensus frame (transNo, t0, x, y, z, *_std,
    n_combinations)}. Time = consensus time of transmission, else first clock-fixed arrival; converted from the
    study basis (local wall clock stored as UTC) to true UTC with tblStudyParameters.UTC_Conv."""
    from scipy.spatial import Delaunay
    conv = pd.read_sql("select UTC_Conv from tblStudyParameters", con).UTC_Conv.dropna()
    if conv.empty:
        raise ValueError("tblStudyParameters.UTC_Conv is empty: cannot write datetime_utc as true UTC")
    shift_s = -float(conv.iloc[0]) * 3600.0          # UTC_Conv = -7 (PDT) -> UTC = study time + 7 h
    tags_sql = ",".join("'%s'" % t for t in tags)
    det = pd.read_sql("select Tag_ID, transNo, min(seconds_fix) t_first, count(distinct Rec_ID) n_receivers "
                      "from tblDetectionFilterSecondary "
                      "where Tag_ID in (%s) and transNo is not null group by Tag_ID, transNo" % tags_sql, con)
    # Hulls of the receivers the solver actually used (any r0-r3 in tblPositions_Deng).
    used = pd.read_sql("select distinct r0 r from tblPositions_Deng union select distinct r1 from tblPositions_Deng "
                       "union select distinct r2 from tblPositions_Deng union select distinct r3 from tblPositions_Deng",
                       con).r.dropna().astype(str)
    rec = pd.read_sql("select * from tblReceiver", con).set_index("Rec_ID")
    rec = rec.loc[[r for r in rec.index if r in set(used)]]
    # Per transmission: positioning receivers that heard it, root-B solutions found, and those inside the water gate.
    heard3d = pd.read_sql("select Tag_ID, transNo, count(distinct Rec_ID) n3d from tblDetectionFilterSecondary "
                          "where Tag_ID in (%s) and transNo is not null and Rec_ID in (%s) group by Tag_ID, transNo"
                          % (tags_sql, ",".join("'%s'" % r for r in rec.index)), con)
    sol = pd.read_sql("select Tag_ID, transNo, Z from tblPositions_Deng where Tag_ID in (%s) and solution = 'B' "
                      "and comment = 'solution found'" % tags_sql, con)
    sol["Z"] = pd.to_numeric(sol.Z, errors="coerce")
    sol["wet"] = (sol.Z <= 0) & (sol.Z >= floor_m)
    sol = sol.groupby(["Tag_ID", "transNo"]).agg(n_found=("Z", "size"), n_wet=("wet", "sum")).reset_index()
    for frame in (det, heard3d, sol):
        frame["transNo"] = pd.to_numeric(frame.transNo).round(0)
    det = det.merge(heard3d, on=["Tag_ID", "transNo"], how="left").merge(sol, on=["Tag_ID", "transNo"], how="left")
    det[["n3d", "n_found", "n_wet"]] = det[["n3d", "n_found", "n_wet"]].fillna(0)
    hull2 = Delaunay(rec[["X_t", "Y_t"]].astype(float).to_numpy())
    hull3 = Delaunay(rec[["X_t", "Y_t", "Z_t"]].astype(float).to_numpy())
    frames = []
    for tag in tags:
        d = det[det.Tag_ID == tag][["transNo", "t_first", "n_receivers", "n3d", "n_found", "n_wet"]].copy()
        g = consensus.get(tag)
        cols = ["transNo", "t0", "x", "y", "z", "x_std", "y_std", "z_std", "n_combinations"]
        g = g[cols].copy() if g is not None and len(g) else pd.DataFrame(columns=cols)
        g["transNo"] = pd.to_numeric(g.transNo).round(0)
        missing = set(g.transNo) - set(d.transNo)
        if missing:
            raise ValueError("%s: %d positioned transmissions have no detection row" % (tag, len(missing)))
        m = d.merge(g, on="transNo", how="left")
        pos = m.x.notna()
        t = np.where(pos, m.t0, m.t_first).astype(float) + shift_s
        xyz2 = m.loc[pos, ["x", "y"]].astype(float).to_numpy()
        xyz3 = m.loc[pos, ["x", "y", "z"]].astype(float).to_numpy()
        in2 = pd.Series(pd.NA, index=m.index, dtype="boolean")
        in3 = pd.Series(pd.NA, index=m.index, dtype="boolean")
        if pos.any():
            in2[pos] = hull2.find_simplex(xyz2) >= 0
            in3[pos] = hull3.find_simplex(xyz3) >= 0
        status = np.select([pos.to_numpy(), (m.n3d < 4).to_numpy(), (m.n_found == 0).to_numpy(), (m.n_wet == 0).to_numpy()],
                           [STATUS_POSITIONED, STATUS_FEW, STATUS_FAILED, STATUS_OUT], default=STATUS_DISAGREE)
        frames.append(pd.DataFrame({
            "AT_code": tag,
            "datetime_utc": pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%S").str[:-3] + "+00:00",
            "x_m": m.x.astype(float).round(1), "y_m": m.y.astype(float).round(1),
            "z_m": m.z.astype(float).round(1),
            "x_std": m.x_std.astype(float).round(1), "y_std": m.y_std.astype(float).round(1),
            "z_std": m.z_std.astype(float).round(1),
            "n_solutions": m.n_combinations.fillna(0).astype(int),
            "in_2Dconvexhull": in2, "in_3Dconvexhull": in3,
            "n_receivers": m.n_receivers.astype(int), "status": status,
            "_t": t}))
        print("  %s: %d epochs heard, %d positioned; %s" % (tag, len(m), int(pos.sum()),
              ", ".join("%s %d" % (k, v) for k, v in pd.Series(status).value_counts().items() if k != STATUS_POSITIONED)))
    table = pd.concat(frames, ignore_index=True).sort_values(["AT_code", "_t"]).drop(columns="_t")
    table[OUTPUT_COLUMNS].to_csv(out / OUTPUT_FILE, index=False)
    print("wrote %s: %d rows (%d positioned) for %d tags; hull receivers: %s"
          % (OUTPUT_FILE, len(table), int(table.x_m.notna().sum()), len(tags), ", ".join(rec.index)))
    return table, list(rec.index)


# Spatial SD map (2026-11-19 request: SD over the area as an approximation of the spatial error rate).
# SD = per-transmission spread of the agreeing Deng solutions (x_std/y_std/z_std in OUTPUT_FILE), i.e. the precision
# of each fix. Gridded at SD_GRID_M: each node takes the median SD of the positioned fixes within SD_RADIUS_M (moving
# window); nodes with fewer than SD_MIN_POINTS fixes in reach stay blank (no values invented outside data coverage).
SD_GRID_M = 1.0
SD_RADIUS_M = 5.0
SD_MIN_POINTS = 5
SD_AXES = (("x", "x_std", "X"), ("y", "y_std", "Y"), ("z", "z_std", "Z"), ("h", "h_std", "horizontal"))


def write_spatial_sd(table, out, rec, hull_ids):
    """spatial_sd_grid.csv, spatial_sd_summary.csv and SD maps (pixel + contour) for X, Y, Z and horizontal."""
    from scipy.spatial import cKDTree, ConvexHull
    p = table[table.x_m.notna()].copy()
    if len(p) < SD_MIN_POINTS:
        print("spatial SD: only %d positioned fixes, maps skipped" % len(p))
        return None
    p["h_std"] = np.hypot(p.x_std, p.y_std)
    xy = p[["x_m", "y_m"]].to_numpy(float)
    # Grid over the data (1st-99th percentile plus the window), so far outliers do not inflate the map.
    lo = np.floor(np.percentile(xy, 1, axis=0) - SD_RADIUS_M)
    hi = np.ceil(np.percentile(xy, 99, axis=0) + SD_RADIUS_M)
    gx = np.arange(lo[0], hi[0] + SD_GRID_M, SD_GRID_M)
    gy = np.arange(lo[1], hi[1] + SD_GRID_M, SD_GRID_M)
    GX, GY = np.meshgrid(gx, gy)
    nodes = np.column_stack([GX.ravel(), GY.ravel()])
    hits = cKDTree(xy).query_ball_point(nodes, r=SD_RADIUS_M)
    n = np.array([len(h) for h in hits])
    ok = n >= SD_MIN_POINTS
    grid = pd.DataFrame({"x_m": nodes[:, 0], "y_m": nodes[:, 1], "n_fixes": n})
    vals = {col: p[col].to_numpy(float) for _, col, _ in SD_AXES}
    for key, col, _ in SD_AXES:
        med = np.full(len(nodes), np.nan)
        p90 = np.full(len(nodes), np.nan)
        for i in np.flatnonzero(ok):
            v = vals[col][hits[i]]
            med[i], p90[i] = np.median(v), np.percentile(v, 90)
        grid["%s_sd_median_m" % key] = np.round(med, 3)
        grid["%s_sd_p90_m" % key] = np.round(p90, 3)
    hull_rec = rec.loc[[r for r in hull_ids if r in rec.index]]
    hull_xy = hull_rec[["X_t", "Y_t"]].astype(float).to_numpy()
    from scipy.spatial import Delaunay
    grid["in_2Dconvexhull"] = Delaunay(hull_xy).find_simplex(nodes) >= 0
    if {"easting", "northing"} <= set(rec.columns):
        grid["easting_m"] = (grid.x_m + float((rec.easting.astype(float) - rec.X_t.astype(float)).median())).round(3)
        grid["northing_m"] = (grid.y_m + float((rec.northing.astype(float) - rec.Y_t.astype(float)).median())).round(3)
    grid = grid[ok].reset_index(drop=True)
    grid.to_csv(out / "spatial_sd_grid.csv", index=False)
    # Fix-level summary inside / outside the 2D receiver hull.
    rows = []
    for name, sel in (("all", p.index), ("inside hull", p.index[p.in_2Dconvexhull == True]),  # noqa: E712
                      ("outside hull", p.index[p.in_2Dconvexhull == False])):  # noqa: E712
        for key, col, label in SD_AXES:
            v = p.loc[sel, col].dropna()
            rows.append({"region": name, "axis": label, "n_fixes": len(v),
                         "median_m": round(float(v.median()), 3) if len(v) else np.nan,
                         "p75_m": round(float(v.quantile(.75)), 3) if len(v) else np.nan,
                         "p90_m": round(float(v.quantile(.9)), 3) if len(v) else np.nan,
                         "p95_m": round(float(v.quantile(.95)), 3) if len(v) else np.nan})
    pd.DataFrame(rows).to_csv(out / "spatial_sd_summary.csv", index=False)
    # Maps: shared colour cap per axis = 95th percentile of the gridded medians (stated on the colour bar).
    hull_poly = hull_xy[ConvexHull(hull_xy).vertices]
    hull_poly = np.vstack([hull_poly, hull_poly[:1]])
    for key, col, label in SD_AXES:
        zvals = np.full(len(nodes), np.nan)
        zvals[ok] = grid["%s_sd_median_m" % key].to_numpy()
        Z = zvals.reshape(GX.shape)
        cap = float(np.nanpercentile(Z, 95))
        inside = float(p.loc[p.in_2Dconvexhull == True, col].median())  # noqa: E712
        for kind in ("pixel", "contour"):
            fig, ax = plt.subplots(figsize=(9.5, 8))
            if kind == "pixel":
                im = ax.pcolormesh(GX, GY, np.ma.masked_invalid(Z), cmap="viridis", vmin=0, vmax=cap, shading="auto")
            else:
                levels = np.linspace(0, cap, 13)
                im = ax.contourf(GX, GY, np.clip(np.ma.masked_invalid(Z), 0, cap), levels=levels, cmap="viridis")
            cb = fig.colorbar(im, ax=ax, extend="max")
            cb.set_label("%s SD of agreeing solutions (m), median within %.0f m; colour capped at %.1f m"
                         % (label, SD_RADIUS_M, cap), fontsize=9)
            ax.plot(hull_poly[:, 0], hull_poly[:, 1], "w--", lw=1.2, label="receiver hull (positioning receivers)")
            for rid, row in rec.iterrows():
                used = rid in hull_ids
                ax.plot(row.X_t, row.Y_t, "^", color="red" if used else "0.4", markersize=7 if used else 5,
                        markeredgecolor="k", markeredgewidth=0.5)
                ax.annotate(rid, (row.X_t + 1.5, row.Y_t + 1.5), fontsize=7, color="k" if used else "0.4")
            ax.set_aspect("equal")
            ax.set_facecolor("0.92")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("Positioning SD, %s: median %.2f m (inside hull %.2f m), %d fixes, %d tags\n"
                         "%.0f m grid; blank = fewer than %d fixes within %.0f m%s"
                         % (label, float(p[col].median()), inside, len(p), p.AT_code.nunique(), SD_GRID_M,
                            SD_MIN_POINTS, SD_RADIUS_M, "" if kind == "pixel" else "; contoured"), fontsize=10)
            ax.legend(loc="lower left", fontsize=8)
            fig.savefig(out / ("spatial_sd_%s%s.png" % (key, "" if kind == "pixel" else "_contour")), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
    print("spatial SD: %d grid nodes with data (%.0f m grid, %.0f m window, >= %d fixes); median SD x/y/z %.2f/%.2f/%.2f m"
          % (len(grid), SD_GRID_M, SD_RADIUS_M, SD_MIN_POINTS, p.x_std.median(), p.y_std.median(), p.z_std.median()))
    return grid


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
    ap.add_argument("--tags", nargs="+", default=None,
                    help="tags to generate figures for (default: all study tags in DB)")
    args = ap.parse_args()
    db, out = Path(args.db), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_file = Path(args.run_file) if args.run_file else run_file_from_status(db)
    floor_m, floor_src = resolve_depth_floor(db, args.depth_floor_m, run_file)
    print("depth floor: %.2f m (%s)%s" % (floor_m, floor_src,
          ("  run file: %s" % run_file) if run_file else ""))
    # Per-fish figures to make: run file [figures] fish_products (default all). fish_positions.csv, fish_qc.csv and
    # the spatial SD outputs are always written.
    products = set(figures_section(run_file).get("fish_products", FISH_PRODUCTS)) if run_file else set(FISH_PRODUCTS)
    unknown = products - set(FISH_PRODUCTS)
    if unknown:
        raise SystemExit("[figures] fish_products: unknown %s (choose from %s)" % (sorted(unknown), list(FISH_PRODUCTS)))
    print("per-fish figures: %s" % ", ".join(p for p in FISH_PRODUCTS if p in products))
    con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)
    rec = pd.read_sql("select Rec_ID, X_t, Y_t, Z_t from tblReceiver", con).set_index("Rec_ID")
    # Root-B solved rows, only the columns the chain uses (full table can be tens of millions of rows).
    p = pd.read_sql("select Tag_ID, transNo, solution, r0, r1, r2, r3, X, Y, Z, ToA, T01, in_hull "
                    "from tblPositions_Deng where comment = 'solution found' and solution = 'B'", con)
    p["t0"] = pd.to_numeric(p.ToA, errors="coerce") - pd.to_numeric(p.T01, errors="coerce")
    for c in ("X", "Y", "Z", "t0"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p[(p.solution == "B") & p[["X", "Y", "Z", "t0"]].notna().all(axis=1)].copy()
    # Determine which tags to process
    if args.tags:
        tags = [t for t in args.tags if (p.Tag_ID == t).any()]
        missing = [t for t in args.tags if t not in p.Tag_ID.values]
        if missing:
            print("note: tags not in DB (ignored): %s" % missing)
    else:
        # Every study tag in the DB with root-B solutions (the four example fish first, as before).
        study = set(pd.read_sql("select Tag_ID from tblTag where TagType = 'study'", con).Tag_ID.astype(str))
        found = set(p.Tag_ID.astype(str)) & study
        tags = [t for t in TAG_ORDER if t in found] + sorted(found - set(TAG_ORDER))
        extra = sorted(set(p.Tag_ID.astype(str)) - study)
        if extra:
            print("note: non-study root-B tags in DB (ignored): %s" % extra)
    if not tags:
        print("NO study-tag root-B positions in %s" % db.name)
        con.close()
        return 1

    qc_rows = []
    consensus = {}
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
        consensus[tag] = g
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

        # ---- Combined figure: plan view (top) + depth over time (bottom) ----
        if "plan_time" in products or "depth_time" in products:
            fig, (ax_plan, ax_depth) = plt.subplots(
                2, 1, figsize=(14, 12),
                gridspec_kw={"height_ratios": [2, 1]},
                constrained_layout=True,
            )

            # ---- Top: plan view (square) ----
            ax_plan.scatter(gk.x, gk.y, c=norm_t, cmap="viridis", s=14, alpha=0.8, zorder=3)
            draw_receivers_2d(ax_plan, rec)

            # First detection = black square, last = red circle
            ax_plan.plot(gk.x.iloc[0], gk.y.iloc[0], "ks", markersize=9, zorder=5,
                         label="First Detection")
            ax_plan.plot(gk.x.iloc[-1], gk.y.iloc[-1], "ro", markersize=9, zorder=5,
                         label="Last Detection")

            ax_plan.set_xlabel("X (m)")
            ax_plan.set_ylabel("Y (m)")
            ax_plan.set_title(tag, fontweight="bold", fontsize=12)
            ax_plan.set_aspect("equal")

            span_h = round((t1 - t0) / 3600.0, 1)
            cbar = fig.colorbar(ax_plan.collections[0], ax=ax_plan, label="Fraction of Time Span (Time Span = %.1f Hours)" % span_h, pad=0.02)

            ax_plan.legend(loc="lower left", fontsize=8)

            # ---- Bottom: depth over time ----
            tt = pd.to_datetime(gk.t0, unit="s")
            ax_depth.plot(tt, -gk.z, "o", color="black", markersize=2, zorder=3,
                          label="Raw Depth Positions")

            # 5-minute rolling median
            rolling = pd.DataFrame({"z": -gk.z, "t": tt}).set_index("t")
            rolling_med = rolling.z.rolling("5min", center=True).median()
            ax_depth.plot(rolling_med.index, rolling_med.values, "--", color="red", linewidth=2.5,
                          alpha=0.8, label="5-Min Rolling Median", zorder=4)

            ax_depth.set_ylabel("Depth (m)")
            ax_depth.set_xlabel("Local Time (PDT) – 2025")
            ax_depth.set_title("")
            ax_depth.invert_yaxis()
            ax_depth.grid(True, alpha=0.3)

            # Date formatting
            ax_depth.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            fig.autofmt_xdate()

            ax_depth.legend(loc="lower right", fontsize=8)

            # Shared font styling
            for ax in (ax_plan, ax_depth):
                ax.tick_params(labelsize=8)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(8)
                ax.xaxis.label.set_fontsize(9)
                ax.yaxis.label.set_fontsize(9)
                ax.xaxis.label.set_fontweight("bold")
                ax.yaxis.label.set_fontweight("bold")
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.label.set_fontsize(9)
            cbar.ax.yaxis.label.set_fontweight("bold")

            fig.savefig(out / ("%s.png" % tag), dpi=150, bbox_inches="tight")
            plt.close(fig)

        if "3d" in products:
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

        if "gifs" not in products:
            continue
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

    pd.DataFrame(qc_rows).to_csv(out / "fish_qc.csv", index=False)
    # Single client CSV for all tags (replaces the former per-tag <TAG>_positions.csv files), then the spatial SD map.
    table, hull_ids = write_fish_positions(con, out, tags, consensus, floor_m)
    write_spatial_sd(table, out, pd.read_sql("select * from tblReceiver", con).set_index("Rec_ID"), hull_ids)
    con.close()
    print("=" * 64)
    print("wrote %d tags to %s" % (len(tags), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
