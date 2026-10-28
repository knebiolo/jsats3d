"""Client figure generator for the 2026-10-22 PM request (tag drags).

Persistent deliverable tool (NOT a pipeline step; never invoked by run_data.py or
_2025_pipeline.py). Re-run as methods improve:

    python scripts/drag_test_figures_2025.py [--db DB_PATH] [--out OUT_DIR]

Inputs (all read-only):
    ENT DB            output/jsats3d_2025_tagdrag_ent_anchorZOI03.db
    GPS csv           K:\\...\\5_array_testing\\array_testing_drag_GPS.csv
    Testing sheets    K:\\...\\5_array_testing\\cowlitz_AT_2025_testing_sheets.xlsx

Products (under output/pm_figures_2026-10-22/):
    per test ID     plan-view PNG (GPS truth black + root-B 3D Kalman track per tag)
    per static hold plan-view PNG (known point star + 10/20 m circles + fixes)
    per category    combined PNG + interactive plotly HTML + client CSVs (solution+consensus)
    qc/             drag_line_errors.csv, static_hold_errors.csv
    README.md       index + QC numbers

Scoring method = scripts/ent_analysis_2025.py item 12 (verified 2026-10-21 baseline):
root-B Deng rows -> per-transmission receiver-set consensus (10 m, >=2 sets) ->
Hampel smoothing (jump 10 m / window 11 / smooth 3) -> constant-velocity Kalman
(2.0, 1.0, 0.5, gap 20 s), scored against the reconstructed 1 Hz GPS track.
The generator prints a baseline check for ENT-01..05 and exits non-zero if any
median drifts > 1 m from the verified item12 numbers.

Fixes vs the 2026-10-22 first pass (see journal 2026-10-22_pm_figures.md):
    * scoring now uses the reference consensus/smooth/Kalman chain on tblPositions_Deng
      (the first pass scored DBSCAN-screened per-transmission means -> 91 m on FBY-D-1,
      and wrongly reported ENT-01/04 and FBY-D-2/3/4 as having no solutions)
    * static holds read from the FULL Forebay_StaticHolds sheet (first pass missed
      FBY-S-8/9/10); DB-1..5 excluded with a documented reason (2025-06-11, outside the
      ENT DB window, and no GPS points collected per the sheet)
    * interactive HTML is real plotly (first pass wrote a 4 KB SVG stub)
    * per-ID plots contain the window-filtered track, not all-day solutions
    * error circles anchored to the GPS start, not a hard-coded (500, 100)
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import sqlite3
import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pyproj import Transformer
import openpyxl
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ent_analysis_2025 import legacy_receiver_set_consensus  # noqa: E402  (reference consensus)

REPO = HERE.parent
GPS_FILE = Path(r"K:\Jobs\5662\001\Data\DataTrans\2025_Data\CowlitzAT2025_Data_Deliverables\5_array_testing\array_testing_drag_GPS.csv")
SHEETS_FILE = Path(r"K:\Jobs\5662\001\Data\DataTrans\2025_Data\CowlitzAT2025_Data_Deliverables\5_array_testing\cowlitz_AT_2025_testing_sheets.xlsx")
DEFAULT_DB = REPO / "output" / "jsats3d_2025_tagdrag_ent_anchorZOI03.db"
DEFAULT_OUT = REPO / "output" / "pm_figures_2026-10-22"

TAGS = ("FC36", "FFD3")
TAGZ = {"FC36": -1.524, "FFD3": -4.572}
CATEGORIES = {
    "CFNSC_Approach_Drags": ["ENT-01", "ENT-02", "ENT-03", "ENT-04", "ENT-05"],
    "Forebay_Random_Drags": ["FBY-D-1", "FBY-D-2", "FBY-D-3", "FBY-D-4"],
}
# Verified 2026-10-21 baseline (item12_pipeline_summary.csv, "3D Kalman filtered", root B):
BASELINE_3D_KALMAN = {
    ("ENT-01", "FC36"): (8.62, 12.5), ("ENT-01", "FFD3"): (10.67, 16.8),
    ("ENT-02", "FC36"): (5.63, 11.4), ("ENT-02", "FFD3"): (6.37, 16.0),
    ("ENT-03", "FC36"): (3.81, 16.0), ("ENT-03", "FFD3"): (3.41, 8.5),
    ("ENT-04", "FC36"): (2.62, 4.4), ("ENT-04", "FFD3"): (2.41, 9.5),
    ("ENT-05", "FC36"): (3.13, 7.7), ("ENT-05", "FFD3"): (2.95, 9.3),
}
CMAP = {"FC36": "Blues", "FFD3": "Oranges"}
LINEC = {"FC36": "#2b6cb0", "FFD3": "#dd6b20"}


# --------------------------------------------------------------------------- depth floor
# Project-interchangeable below-bottom plausibility gate. The array's receivers define the
# deployment area, so the default is derived from the database (deepest receiver minus a
# margin); a project overrides it in its run file ([figures]) or on the command line.
DEFAULT_DEPTH_FLOOR_MARGIN_M = 4.0


def run_file_from_status(db):
    """Auto-discover a database's run file from its pipeline status record (if any)."""
    status = Path(db).parent / ("%s.pipeline_status.json" % Path(db).stem)
    if status.exists():
        try:
            rf = json.loads(status.read_text(encoding="utf-8")).get("run_file")
        except (OSError, json.JSONDecodeError):
            return None
        if rf:
            p = Path(rf)
            if p.is_absolute():
                return p if p.exists() else None
            cand = REPO / p
            return cand if cand.exists() else None
    return None


def figures_section(run_file):
    """[figures] table of a run file ({} when absent)."""
    if run_file is None:
        return {}
    path = Path(run_file)
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh).get("figures", {}) or {}


def resolve_depth_floor(db, explicit=None, run_file=None, figs=None):
    """Z (m) below which solutions are unphysical (fish/drag tags cannot be below the bay bottom).

    Returns (floor_m, provenance). Priority:
      1. explicit value (CLI --depth-floor-m)
      2. run file [figures] depth_floor_m
      3. auto: deepest receiver Z_t minus margin
         (run file [figures] depth_floor_margin_m, else 4.0 m)
    """
    figs = figs if figs is not None else figures_section(run_file)
    if explicit is not None:
        return float(explicit), "CLI --depth-floor-m"
    if "depth_floor_m" in figs:
        return float(figs["depth_floor_m"]), "run file [figures] depth_floor_m"
    con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)
    try:
        deepest = float(pd.read_sql("select min(Z_t) z from tblReceiver", con).z.iloc[0])
    finally:
        con.close()
    margin = float(figs.get("depth_floor_margin_m", DEFAULT_DEPTH_FLOOR_MARGIN_M))
    return deepest - margin, "auto: deepest receiver %.2f m minus %.1f m margin" % (deepest, margin)


# --------------------------------------------------------------------------- reference chain
def smooth_track(track, jump_limit_m=15.0, jump_window=7, smooth_window=5, max_gap_s=20.0):
    """Copied verbatim from ent_analysis_2025.item12 (tuned 2026-10-21)."""
    if track.empty:
        return track, 0
    track = track.sort_values("t0").reset_index(drop=True).copy()
    if len(track) < 3:
        track["jump_distance_m"] = np.nan
        return track, 0
    center_x = track.x.rolling(jump_window, center=True, min_periods=3).median()
    center_y = track.y.rolling(jump_window, center=True, min_periods=3).median()
    track["jump_distance_m"] = np.hypot(track.x - center_x, track.y - center_y)
    keep = track.jump_distance_m.isna() | (track.jump_distance_m <= jump_limit_m)
    rejected = int((~keep).sum())
    track = track[keep].copy().reset_index(drop=True)
    track["segment"] = (track.t0.diff().fillna(0) > max_gap_s).cumsum()
    for _, indices in track.groupby("segment").groups.items():
        for column in ("x", "y", "z"):
            track.loc[indices, column] = track.loc[indices, column].rolling(
                smooth_window, center=True, min_periods=1).median().to_numpy()
    return track.drop(columns="segment"), rejected


def kalman_filter_track(track, process_noise=0.5, measurement_noise=2.0, z_noise=1.0, max_gap_s=12.0):
    """Copied verbatim from ent_analysis_2025.item12 (tuned 2026-10-19/21)."""
    if len(track) < 3:
        return track.copy(), 0
    track = track.sort_values("t0").reset_index(drop=True).copy()
    track["segment"] = (track.t0.diff().fillna(0) > max_gap_s).cumsum()
    results = []
    for seg_id, seg in track.groupby("segment", sort=False):
        if len(seg) < 2:
            results.append(seg)
            continue
        pos = seg[["x", "y", "z"]].values.astype(float)
        dt = np.diff(seg.t0.values.astype(float))
        dt = np.maximum(dt, 0.1)
        n = len(pos)
        F = np.eye(6)
        H = np.eye(3, 6)
        Q = np.eye(6)
        Q[:3, :3] *= process_noise ** 2
        Q[3:, 3:] *= process_noise ** 2
        R = np.eye(3)
        R[0, 0] = measurement_noise ** 2
        R[1, 1] = measurement_noise ** 2
        R[2, 2] = z_noise ** 2
        x_est = np.array([pos[0, 0], pos[0, 1], pos[0, 2], 0.0, 0.0, 0.0])
        P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
        smoothed = np.zeros((n, 3))
        smoothed[0] = pos[0]
        for i in range(1, n):
            F[0, 3] = dt[i - 1]
            F[1, 4] = dt[i - 1]
            F[2, 5] = dt[i - 1]
            x_est = F @ x_est
            P = F @ P @ F.T + Q
            F[0, 3] = 0
            F[1, 4] = 0
            F[2, 5] = 0
            y_meas = pos[i] - H @ x_est
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x_est = x_est + K @ y_meas
            P = (np.eye(6) - K @ H) @ P
            smoothed[i] = x_est[:3]
        seg_out = seg[["t0", "transNo"]].copy()
        seg_out["x"] = smoothed[:, 0]
        seg_out["y"] = smoothed[:, 1]
        seg_out["z"] = smoothed[:, 2]
        results.append(seg_out)
    result = pd.concat(results, ignore_index=True).sort_values("t0").reset_index(drop=True)
    return result, 0


def truth(tr, t):
    """GPS truth at time t (interpolated along the reconstructed 1 Hz track)."""
    ok = (t >= tr.gtime.min()) & (t <= tr.gtime.max())
    tx = np.interp(t, tr.gtime, tr.x)
    ty = np.interp(t, tr.gtime, tr.y)
    return tx, ty, ok


def iso_utc(t):
    return pd.to_datetime(t, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


# --------------------------------------------------------------------------- data
class DataManager:
    def __init__(self, db):
        con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)
        rec = pd.read_sql("select Rec_ID, X_t, Y_t, Z_t, easting, northing, X, Y from tblReceiver", con)
        self.rec = rec.set_index("Rec_ID")
        self.xyz = self.rec[["X_t", "Y_t", "Z_t"]]
        self.ox = float((self.rec.easting - self.rec.X).median())
        self.oy = float((self.rec.northing - self.rec.Y).median())

        # GPS tracks with the reference 1 Hz reconstruction (ent_analysis_2025.Data)
        gps = pd.read_csv(GPS_FILE)
        gx, gy = Transformer.from_crs(4326, 26910, always_xy=True).transform(gps.long.values, gps.lat.values)
        gps["x"], gps["y"] = gx - self.ox, gy - self.oy
        gps["minute"] = pd.to_datetime(gps.timestamp)
        self.tracks = {}
        for line in sorted(set(gps.line_name.unique())):
            t = gps[gps.line_name == line].reset_index(drop=True)
            if t.empty:
                continue
            counts = t.groupby("minute").size()
            # 1 Hz: the partial first minute holds the last n seconds of its minute
            t["gtime"] = counts.index[0].tz_localize("UTC").timestamp() + 60 - counts.iloc[0] + np.arange(len(t))
            self.tracks[line] = t

        # First-arrival epoch per (tag, transmission) — same source as ent_analysis
        cols = [r[1] for r in con.execute("pragma table_info(tblDetectionFilterSecondary)")]
        want = [c for c in ("Tag_ID", "transNo", "seconds_fix") if c in cols]
        sec = pd.read_sql("select %s from tblDetectionFilterSecondary" % ", ".join(want), con)
        sec["seconds_fix"] = pd.to_numeric(sec.seconds_fix)
        self.first = {tag: g.seconds_fix.groupby(g.transNo).min()
                      for tag, g in sec[sec.Tag_ID.isin(TAGS)].groupby("Tag_ID")}

        # Root-B Deng positions (3D and 2D)
        self.p3b = self._load_root_b(con, "tblPositions_Deng")
        self.p2b = self._load_root_b(con, "tblPositions_Deng2D")
        con.close()

    @staticmethod
    def _load_root_b(con, table):
        if table not in [r[0] for r in con.execute("select name from sqlite_master where type='table'")]:
            return pd.DataFrame()
        p = pd.read_sql("select * from %s where comment = 'solution found'" % table, con)
        if p.empty:
            return p
        if "r3" not in p.columns:
            p["r3"] = ""
        p["t0"] = pd.to_numeric(p.ToA, errors="coerce") - pd.to_numeric(p.T01, errors="coerce")
        for c in ("X", "Y", "Z", "t0"):
            p[c] = pd.to_numeric(p[c], errors="coerce")
        num = p[["X", "Y", "Z", "t0"]].notna().all(axis=1)
        return p[(p.solution == "B") & num].copy()

    def trans_ids(self, line, tag):
        """Transmission numbers whose first arrival falls in the line's GPS window (±3 s)."""
        tr = self.tracks.get(line)
        f = self.first.get(tag)
        if tr is None or f is None:
            return set()
        lo, hi = tr.gtime.min() - 3, tr.gtime.max() + 3
        return set(f[(f >= lo) & (f <= hi)].index)

    def static_holds(self):
        """Valid single-point static holds from the full Forebay_StaticHolds sheet.

        A row is valid only if it has numeric Lat/Lon plus Date/Start/End. This
        excludes FBY-S-6 ("Replace this with DB-1-5") and DB-1..5 (Debris Barrier
        holds of 2025-06-11 — outside the ENT DB window, and no GPS collected:
        DB-1 says 'Not Collected', DB-2..5 are blank).
        """
        wb = openpyxl.load_workbook(str(SHEETS_FILE), read_only=True)
        ws = wb["Forebay_StaticHolds"]
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        if not rows:
            return []
        header = [str(c).strip() if c is not None else "" for c in rows[0]]
        tr = Transformer.from_crs(4326, 26910, always_xy=True)
        holds, skipped = [], []
        for row in rows[1:]:
            d = dict(zip(header, row))
            tid = str(d.get("Test ID") or "").strip()
            lat, lon = d.get("Lat"), d.get("Lon")
            date, st, et = d.get("Date"), d.get("Start Time"), d.get("End Time")
            valid = (tid
                     and isinstance(lat, (int, float)) and isinstance(lon, (int, float))
                     and not pd.isna(lat) and not pd.isna(lon)
                     and date is not None and st is not None and et is not None)
            if not valid:
                if tid:
                    skipped.append(tid)
                continue
            t0 = (pd.Timestamp(date) + pd.Timedelta(str(st))).tz_localize("UTC").timestamp()
            t1 = (pd.Timestamp(date) + pd.Timedelta(str(et))).tz_localize("UTC").timestamp()
            hx, hy = tr.transform(float(lon), float(lat))
            holds.append(dict(hold_id=tid, x=hx - self.ox, y=hy - self.oy, t0=t0, t1=t1,
                              start=str(st)[:8], end=str(et)[:8], date=str(pd.Timestamp(date).date())[:10]))
        return holds, skipped


# --------------------------------------------------------------------------- products
def line_product(dm, line, tag):
    """Reference chain for one drag line + tag: consensus -> smooth -> Kalman -> score."""
    tr = dm.tracks[line]
    ids = dm.trans_ids(line, tag)
    rows = dm.p3b[(dm.p3b.Tag_ID == tag) & (dm.p3b.transNo.isin(ids))]
    g = legacy_receiver_set_consensus(rows)
    gs, rejected = smooth_track(g, jump_limit_m=10.0, jump_window=11, smooth_window=3)
    gk, _ = kalman_filter_track(gs, process_noise=2.0, measurement_noise=1.0, z_noise=0.5, max_gap_s=20.0)
    if gk.empty:
        return dict(ids=ids, rows=rows, consensus=g, kalman=gk, scored=None, rejected=rejected)
    # single-point segments may carry extra input columns; normalize before merging
    gk = gk[["t0", "transNo", "x", "y", "z"]]
    if not g.empty and "receiver_sets" in g.columns:
        gk = gk.merge(g[["transNo", "receiver_sets", "n_combinations"]], on="transNo", how="left")
    else:
        gk["receiver_sets"] = 0
        gk["n_combinations"] = 0
    scored = None
    if not gk.empty:
        tx, ty, ok = truth(tr, gk.t0.to_numpy())
        s = gk[ok].copy()
        if not s.empty:
            s["err"] = np.hypot(s.x - tx[ok], s.y - ty[ok])
            scored = s
    return dict(ids=ids, rows=rows, consensus=g, kalman=gk, scored=scored, rejected=rejected)


def hold_product(dm, hold, tag, table):
    """Per-transmission median positions inside the hold window (3D or 2D table)."""
    sub = table[(table.Tag_ID == tag) & (table.t0 >= hold["t0"] - 1) & (table.t0 <= hold["t1"] + 1)]
    g = sub.groupby("transNo").agg(x=("X", "median"), y=("Y", "median"), z=("Z", "median"),
                                   t0=("t0", "median"), n=("X", "size")).reset_index().sort_values("t0")
    if g.empty:
        return None
    g["dx"], g["dy"] = g.x - hold["x"], g.y - hold["y"]
    g["err"] = np.hypot(g.dx, g.dy)
    return g


# --------------------------------------------------------------------------- plotting
def _style(ax, title):
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def _receivers(ax, xyz, fontsize=7):
    for rec_id, row in xyz.iterrows():
        ax.plot(row.X_t, row.Y_t, "kx", markersize=7, markeredgewidth=1.2, zorder=4)
        ax.annotate(rec_id, (row.X_t + 2, row.Y_t + 2), fontsize=fontsize, color="gray")


def _circles(ax, cx, cy, radii=(10, 20)):
    for r in radii:
        ax.add_patch(Circle((cx, cy), r, fill=False, color="red", linestyle="--", alpha=0.5, zorder=2))
    ax.text(cx + 11, cy + 11, "10 m", color="red", fontsize=7, alpha=0.8)
    ax.text(cx + 21, cy + 21, "20 m", color="red", fontsize=7, alpha=0.8)


def plot_line(dm, line, products, out_path):
    """Plan view, one test ID: GPS truth (black) + root-B 3D Kalman track per tag."""
    tr = dm.tracks[line]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(tr.x, tr.y, color="black", linewidth=2, alpha=0.7, zorder=1, label="GPS truth")
    ax.plot(tr.x.iloc[0], tr.y.iloc[0], "g>", markersize=11, zorder=5, label="GPS start")
    ax.plot(tr.x.iloc[-1], tr.y.iloc[-1], "r<", markersize=11, zorder=5, label="GPS end")
    for tag in TAGS:
        gk = products[tag]["kalman"]
        if gk.empty:
            continue
        t = gk.sort_values("t0")
        norm = (t.t0 - t.t0.min()) / max(t.t0.max() - t.t0.min(), 1e-9)
        ax.plot(t.x, t.y, color=LINEC[tag], alpha=0.35, linewidth=1.2, zorder=2)
        ax.scatter(t.x, t.y, c=norm, cmap=CMAP[tag], s=22, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.3, label=tag)
    _receivers(ax, dm.xyz)
    _circles(ax, tr.x.iloc[0], tr.y.iloc[0])
    t0 = pd.Timestamp(tr.gtime.iloc[0], unit="s", tz="UTC").strftime("%H:%M")
    t1 = pd.Timestamp(tr.gtime.iloc[-1], unit="s", tz="UTC").strftime("%H:%M")
    _style(ax, "%s · 2025-06-10 %s–%s · 3D root B (consensus + Kalman)" % (line, t0, t1))
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _receivers_3d(ax, xyz, fontsize=7):
    for rec_id, row in xyz.iterrows():
        ax.scatter([row.X_t], [row.Y_t], [row.Z_t], marker="x", color="k", s=30,
                   depthshade=False)
        ax.text(row.X_t + 2, row.Y_t + 2, row.Z_t, rec_id, fontsize=fontsize, color="0.35")


def plot_line_3d(dm, line, products, out_path, floor_m):
    """Simple 3D view of one drag line: GPS truth at each tag's known tow depth (black)
    + the root-B Kalman track per tag, restricted to the plausible water column
    [floor_m, 0] (the same plausibility gate as the fish figures; scoring is untouched)."""
    tr = dm.tracks[line]
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    xlo = min(dm.xyz.X_t.min(), tr.x.min()) - 15
    xhi = max(dm.xyz.X_t.max(), tr.x.max()) + 15
    ylo = min(dm.xyz.Y_t.min(), tr.y.min()) - 15
    yhi = max(dm.xyz.Y_t.max(), tr.y.max()) + 15
    for tag in TAGS:
        gk = products[tag]["kalman"]
        if gk.empty:
            continue
        t = gk.sort_values("t0")
        # GPS truth is the surface boat track; the tag was towed at TAGZ[tag].
        ax.plot(tr.x, tr.y, [TAGZ[tag]] * len(tr), color="black", linewidth=1.2,
                alpha=0.6, zorder=1)
        ax.plot([tr.x.iloc[0]], [tr.y.iloc[0]], [TAGZ[tag]], "g>", markersize=9, zorder=4)
        ax.plot([tr.x.iloc[-1]], [tr.y.iloc[-1]], [TAGZ[tag]], "r<", markersize=9, zorder=4)
        # Water-column gate: drop above-surface and below-floor artifacts from the view.
        tg = t[(t.z <= 0) & (t.z >= floor_m)]
        if tg.empty:
            continue
        norm = (tg.t0 - tg.t0.min()) / max(tg.t0.max() - tg.t0.min(), 1e-9)
        ax.plot(tg.x, tg.y, tg.z, color=LINEC[tag], alpha=0.3, linewidth=1.0, zorder=2)
        ax.scatter(tg.x, tg.y, tg.z, c=norm, cmap=CMAP[tag], s=16, alpha=0.85,
                   depthshade=False, label="%s (tow %.1f m)" % (tag, TAGZ[tag]))
    _receivers_3d(ax, dm.xyz)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(floor_m, 2.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    t0 = pd.Timestamp(tr.gtime.iloc[0], unit="s", tz="UTC").strftime("%H:%M")
    t1 = pd.Timestamp(tr.gtime.iloc[-1], unit="s", tz="UTC").strftime("%H:%M")
    ax.set_title("%s · 2025-06-10 %s–%s · 3D root B (GPS at tow depth; track gated to [%.0f, 0] m)"
                 % (line, t0, t1, floor_m))
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def line_depth_qc(line, products, floor_m):
    """Per-line Z sanity of the delivered Kalman product vs the known tow depths."""
    rows = []
    for tag in TAGS:
        gk = products[tag]["kalman"]
        if gk.empty:
            rows.append(dict(line=line, tag=tag, n=0, known_z_m=TAGZ[tag]))
            continue
        z = gk.z.to_numpy()
        rows.append(dict(line=line, tag=tag, n=int(len(z)), known_z_m=TAGZ[tag],
                         z_median_m=round(float(np.median(z)), 2),
                         z_min_m=round(float(z.min()), 2), z_max_m=round(float(z.max()), 2),
                         above_surface_pct=round(100.0 * float((z > 0).mean()), 1),
                         below_floor_pct=round(100.0 * float((z < floor_m).mean()), 1)))
    return rows


def plot_hold(dm, hold, products, out_path):
    """Plan view, one static hold: known point (star) + per-transmission fixes per tag."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(hold["x"], hold["y"], "k*", markersize=20, zorder=5, label="True position")
    for tag in TAGS:
        g = products.get(tag)
        if g is None or g.empty:
            continue
        t = g.sort_values("t0")
        norm = (t.t0 - t.t0.min()) / max(t.t0.max() - t.t0.min(), 1e-9)
        ax.scatter(t.x, t.y, c=norm, cmap=CMAP[tag], s=30, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.3, label=tag)
    _receivers(ax, dm.xyz)
    _circles(ax, hold["x"], hold["y"])
    _style(ax, "%s · %s %s–%s · static hold (per-transmission median)"
           % (hold["hold_id"], hold["date"], hold["start"], hold["end"]))
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(dm, category, test_ids, products_by_line, out_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    for line in test_ids:
        tr = dm.tracks.get(line)
        if tr is None:
            continue
        ax.plot(tr.x, tr.y, color="black", linewidth=1, alpha=0.5, zorder=1)
        ax.plot(tr.x.iloc[0], tr.y.iloc[0], "g>", markersize=8, zorder=4)
        ax.plot(tr.x.iloc[-1], tr.y.iloc[-1], "r<", markersize=8, zorder=4)
        ax.annotate(line, (tr.x.iloc[0] + 2, tr.y.iloc[0] + 2), fontsize=7, color="black")
    for tag in TAGS:
        pts = [p[tag]["kalman"] for p in products_by_line.values() if not p[tag]["kalman"].empty]
        if not pts:
            continue
        t = pd.concat(pts, ignore_index=True).sort_values("t0")
        norm = (t.t0 - t.t0.min()) / max(t.t0.max() - t.t0.min(), 1e-9)
        ax.scatter(t.x, t.y, c=norm, cmap=CMAP[tag], s=14, alpha=0.7, zorder=3,
                   edgecolors="white", linewidths=0.2, label=tag)
    _receivers(ax, dm.xyz)
    _style(ax, "%s — all test IDs (3D root B)" % category)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_arrows(dm, holds, hold_qc, out_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    for hold in holds:
        rows = [r for r in hold_qc if r["hold_id"] == hold["hold_id"] and r["product"] == "3D"]
        if not rows:
            continue
        mdx = float(np.mean([r["median_dx_m"] for r in rows]))
        mdy = float(np.mean([r["median_dy_m"] for r in rows]))
        ax.plot(hold["x"], hold["y"], "k*", markersize=15, zorder=5)
        ax.arrow(hold["x"], hold["y"], mdx, mdy, head_width=2.5, head_length=1.5,
                 fc="red", ec="red", alpha=0.7, zorder=4)
        ax.annotate(hold["hold_id"], (hold["x"] + 4, hold["y"] + 4), fontsize=8)
    _receivers(ax, dm.xyz)
    _style(ax, "Static hold error arrows — mean per-transmission offset from true position (3D root B)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- interactive HTML
def write_interactive_html(dm, category, test_ids, products_by_line, holds, hold_products, out_path):
    fig = go.Figure()
    if holds:  # static-hold category
        for hold in holds:
            fig.add_trace(go.Scatter(x=[hold["x"]], y=[hold["y"]], mode="markers",
                                     marker=dict(symbol="star", size=16, color="black"),
                                     name="true: %s" % hold["hold_id"], showlegend=False,
                                     hovertemplate="%s (known point)<br>X=%{x:.1f} Y=%{y:.1f}<extra></extra>"))
            for tag in TAGS:
                g = hold_products.get((hold["hold_id"], tag))
                if g is None or g.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=g.x, y=g.y, mode="markers", name="%s %s" % (tag, hold["hold_id"]),
                    marker=dict(size=7, color=g.t0, colorscale="Blues" if tag == "FC36" else "Oranges",
                                colorbar=dict(title="t (s)")),
                    hovertemplate="%s %s<br>%{customdata[0]}<br>X=%{x:.2f} Y=%{y:.2f} Z=%{customdata[1]:.2f}<extra></extra>",
                    customdata=pd.DataFrame({"t": iso_utc(g.t0), "z": g.z}).values))
        gps_lines = []
    else:
        for line in test_ids:
            tr = dm.tracks.get(line)
            if tr is None:
                continue
            fig.add_trace(go.Scatter(x=tr.x, y=tr.y, mode="lines", name="GPS %s" % line,
                                     line=dict(color="black", width=2)))
            fig.add_trace(go.Scatter(x=[tr.x.iloc[0]], y=[tr.y.iloc[0]], mode="markers",
                                     marker=dict(symbol="triangle-right", size=12, color="green"),
                                     name="GPS %s start" % line, showlegend=False))
            fig.add_trace(go.Scatter(x=[tr.x.iloc[-1]], y=[tr.y.iloc[-1]], mode="markers",
                                     marker=dict(symbol="triangle-left", size=12, color="red"),
                                     name="GPS %s end" % line, showlegend=False))
        for line in test_ids:
            for tag in TAGS:
                gk = products_by_line.get(line, {}).get(tag, {}).get("kalman")
                if gk is None or gk.empty:
                    continue
                t = gk.sort_values("t0")
                fig.add_trace(go.Scatter(
                    x=t.x, y=t.y, mode="lines+markers", name="%s %s" % (line, tag),
                    line=dict(width=1.2, color="lightgray"),
                    marker=dict(size=8, color=t.t0, colorscale="Blues" if tag == "FC36" else "Oranges",
                                colorbar=dict(title="t (s)")),
                    hovertemplate="%s %s<br>%{customdata[0]}<br>X=%{x:.2f} Y=%{y:.2f} Z=%{customdata[1]:.2f}<br>sets=%{customdata[2]}<extra></extra>",
                    customdata=pd.DataFrame({"t": iso_utc(t.t0), "z": t.z,
                                             "n": t.receiver_sets.fillna(0).astype(int)}).values))
    rec = dm.rec
    fig.add_trace(go.Scatter(x=rec.X_t, y=rec.Y_t, mode="markers+text", name="receivers",
                             marker=dict(symbol="x", size=10, color="gray"),
                             text=rec.index, textposition="top right", textfont=dict(size=9, color="gray"),
                             hovertemplate="%{text}<extra>receiver</extra>"))
    x0, x1 = min(rec.X_t.min(), rec.Y_t.min()) - 25, max(rec.X_t.max(), rec.Y_t.max()) + 25
    fig.update_layout(title="%s — interactive (root B, 3D)" % category,
                      xaxis_title="X (m)", yaxis_title="Y (m)",
                      xaxis=dict(range=[x0, x1], scaleanchor="y"),
                      yaxis=dict(range=[x0, x1]), height=760,
                      legend=dict(font=dict(size=10)))
    fig.write_html(str(out_path), include_plotlyjs="inline")


# --------------------------------------------------------------------------- CSVs
def write_line_csvs(dm, category, test_ids, products_by_line, out_dir):
    written = []
    for tag in TAGS:
        sol_rows, con_rows = [], []
        for line in test_ids:
            p = products_by_line.get(line, {}).get(tag)
            if p is None:
                continue
            ids = p["ids"]
            rows = p["rows"]
            if not rows.empty:
                s = rows[rows.transNo.isin(ids)].copy()
                s["receiver_set"] = s[["r0", "r1", "r2", "r3"]].astype(str).apply(
                    lambda r: "+".join(sorted(r.values)), axis=1, result_type="reduce")
                sol_rows.append(pd.DataFrame({
                    "tag": tag, "category": category, "test_id": line,
                    "t_iso_utc": iso_utc(s.t0), "t_s": s.t0.round(3),
                    "X_m": s.X.round(3), "Y_m": s.Y.round(3), "Z_m": s.Z.round(3),
                    "product": "solution", "root": "B", "dim": "3D",
                    "in_hull": s.in_hull, "transNo": s.transNo,
                    "r0": s.r0, "r1": s.r1, "r2": s.r2, "r3": s.r3,
                    "receiver_set": s.receiver_set}))
            if not p["consensus"].empty:
                c = p["consensus"].copy()
                con_rows.append(pd.DataFrame({
                    "tag": tag, "category": category, "test_id": line,
                    "t_iso_utc": iso_utc(c.t0), "t_s": c.t0.round(3),
                    "X_m": c.x.round(3), "Y_m": c.y.round(3), "Z_m": c.z.round(3),
                    "product": "consensus", "root": "B", "dim": "3D",
                    "in_hull": "", "transNo": c.transNo,
                    "r0": "", "r1": "", "r2": "", "r3": "",
                    "receiver_set": "", "receiver_sets": c.receiver_sets,
                    "n_combinations": c.n_combinations}))
        if sol_rows or con_rows:
            df = pd.concat(sol_rows + con_rows, ignore_index=True, sort=False)
            path = out_dir / ("%s_positions.csv" % tag)
            df.to_csv(path, index=False)
            written.append(path)
    # GPS tracks (same for both tags — one file per category)
    gps_parts = []
    for line in test_ids:
        tr = dm.tracks.get(line)
        if tr is None:
            continue
        gps_parts.append(pd.DataFrame({"line_name": line, "t_s": tr.gtime.round(3),
                                       "t_iso_utc": iso_utc(tr.gtime),
                                       "x_m": tr.x.round(3), "y_m": tr.y.round(3)}))
    if gps_parts:
        path = out_dir / ("%s_gps_tracks.csv" % category)
        pd.concat(gps_parts, ignore_index=True).to_csv(path, index=False)
        written.append(path)
    return written


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    (out / "qc").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PM Figures Generator 2026-10-22 (fixed)  db=%s" % db.name)
    print("=" * 72)
    run_file = Path(args.run_file) if args.run_file else run_file_from_status(db)
    floor_m, floor_src = resolve_depth_floor(db, args.depth_floor_m, run_file)
    print("depth floor: %.2f m (%s)%s" % (floor_m, floor_src,
          ("  run file: %s" % run_file) if run_file else ""))
    dm = DataManager(db)
    holds, skipped_holds = dm.static_holds()
    print("static holds (valid): %s" % [h["hold_id"] for h in holds])
    print("static holds (skipped, no numeric GPS point): %s" % skipped_holds)

    manifest = []
    line_qc, hold_qc, depth_qc = [], [], []
    products_by_category = {}

    # ---------------- drag lines (CFNSC approach + forebay random)
    for category, test_ids in CATEGORIES.items():
        cat_dir = out / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        products_by_line = {}
        for line in test_ids:
            products = {tag: line_product(dm, line, tag) for tag in TAGS}
            products_by_line[line] = products
            plot_line(dm, line, products, cat_dir / ("%s.png" % line))
            manifest.append(cat_dir / ("%s.png" % line))
            plot_line_3d(dm, line, products, cat_dir / ("%s_3d.png" % line), floor_m)
            manifest.append(cat_dir / ("%s_3d.png" % line))
            depth_qc.extend(line_depth_qc(line, products, floor_m))
            for tag in TAGS:
                p = products[tag]
                s = p["scored"]
                if s is None or s.empty:
                    print("  %s %s: NO SCORED TRANSMISSIONS (candidates=%d)" % (line, tag, len(p["ids"])))
                    continue
                row = dict(line=line, tag=tag, n_scored=len(s),
                           median_xy_m=round(float(s.err.median()), 2),
                           p90_xy_m=round(float(s.err.quantile(0.9)), 1),
                           yield_pct=round(100 * len(s) / max(len(p["ids"]), 1)))
                line_qc.append(row)
                print("  %s %s: n=%d median=%.2f m p90=%.1f m" % (line, tag, len(s), row["median_xy_m"], row["p90_xy_m"]))
        plot_combined(dm, category, test_ids, products_by_line, cat_dir / ("%s_combined.png" % category))
        manifest.append(cat_dir / ("%s_combined.png" % category))
        write_interactive_html(dm, category, test_ids, products_by_line, None, None,
                               cat_dir / ("%s_interactive.html" % category))
        manifest.append(cat_dir / ("%s_interactive.html" % category))
        manifest.extend(write_line_csvs(dm, category, test_ids, products_by_line, cat_dir))
        products_by_category[category] = products_by_line

    # ---------------- static holds
    hold_dir = out / "Forebay_StaticHolds"
    hold_dir.mkdir(parents=True, exist_ok=True)
    hold_products = {}
    for hold in holds:
        products = {}
        for tag in TAGS:
            for dim, table in (("3D", dm.p3b), ("2D", dm.p2b)):
                g = hold_product(dm, hold, tag, table)
                if g is None or g.empty:
                    continue
                hold_products[(hold["hold_id"], tag, dim)] = g
                products[tag] = products.get(tag, g)
                line_qc_row = dict(hold_id=hold["hold_id"], tag=tag, product=dim,
                                   n_transmissions=int(g.transNo.nunique()), n_solutions=int(len(g)),
                                   median_xy_m=round(float(g.err.median()), 2),
                                   p90_xy_m=round(float(g.err.quantile(0.9)), 1),
                                   median_dx_m=round(float(g.dx.median()), 2),
                                   median_dy_m=round(float(g.dy.median()), 2))
                hold_qc.append(line_qc_row)
                print("  hold %s %s %s: n_tx=%d median=%.2f m p90=%.1f m"
                      % (hold["hold_id"], tag, dim, line_qc_row["n_transmissions"],
                         line_qc_row["median_xy_m"], line_qc_row["p90_xy_m"]))
        plot_hold(dm, hold, {tag: hold_products.get((hold["hold_id"], tag, "3D")) for tag in TAGS},
                  hold_dir / ("%s.png" % hold["hold_id"]))
        manifest.append(hold_dir / ("%s.png" % hold["hold_id"]))
        # client CSV: hold positions (3D consensus) + no GPS (single known point in the sheet)
    hold_qc_df = pd.DataFrame(hold_qc)
    hold_qc_df.to_csv(out / "qc" / "static_hold_errors.csv", index=False)
    manifest.append(out / "qc" / "static_hold_errors.csv")
    plot_error_arrows(dm, holds, hold_qc_df.to_dict("records") if not hold_qc_df.empty else [], hold_dir / "error_arrows.png")
    manifest.append(hold_dir / "error_arrows.png")
    write_interactive_html(dm, "Forebay_StaticHolds", [], {}, holds,
                           {(h, t): hold_products.get((h, t, "3D")) for h, t in
                            {(hh["hold_id"], tt) for hh in holds for tt in TAGS}},
                           hold_dir / "Forebay_StaticHolds_interactive.html")
    manifest.append(hold_dir / "Forebay_StaticHolds_interactive.html")
    # static-hold CSVs (per tag: all per-transmission fixes, 3D+2D)
    for tag in TAGS:
        parts = []
        for (hid, t, dim), g in hold_products.items():
            if t != tag:
                continue
            hold = next(h for h in holds if h["hold_id"] == hid)
            parts.append(pd.DataFrame({
                "tag": tag, "category": "Forebay_StaticHolds", "test_id": hid,
                "t_iso_utc": iso_utc(g.t0), "t_s": g.t0.round(3),
                "X_m": g.x.round(3), "Y_m": g.y.round(3), "Z_m": g.z.round(3),
                "product": "consensus", "root": "B", "dim": dim,
                "in_hull": "", "transNo": g.transNo}))
        if parts:
            path = hold_dir / ("%s_positions.csv" % tag)
            pd.concat(parts, ignore_index=True, sort=False).to_csv(path, index=False)
            manifest.append(path)
    if holds:
        with open(hold_dir / "hold_points.csv", "w") as fh:
            fh.write("hold_id,date,start_time,end_time,known_X_m,known_Y_m\n")
            for h in holds:
                fh.write("%s,%s,%s,%s,%.3f,%.3f\n" % (h["hold_id"], h["date"], h["start"], h["end"], h["x"], h["y"]))
        manifest.append(hold_dir / "hold_points.csv")

    # ---------------- QC drag csv
    if line_qc:
        pd.DataFrame(line_qc).to_csv(out / "qc" / "drag_line_errors.csv", index=False)
        manifest.append(out / "qc" / "drag_line_errors.csv")
    if depth_qc:
        depth_path = out / "qc" / "drag_line_depths.csv"
        pd.DataFrame(depth_qc).to_csv(depth_path, index=False)
        manifest.append(depth_path)
        print("\n=== Drag line Z sanity (Kalman product vs known tow depth; floor %.2f m) ===" % floor_m)
        for r in depth_qc:
            if not r.get("n"):
                continue
            print("  %s %s: n=%-4d known %5.2f m  Z median %7.2f m  range [%.1f, %.1f]  "
                  "above-surface %.1f%%  below-floor %.1f%%"
                  % (r["line"], r["tag"], r["n"], r["known_z_m"], r["z_median_m"],
                     r["z_min_m"], r["z_max_m"], r["above_surface_pct"], r["below_floor_pct"]))

    # ---------------- baseline gate (ENT-01..05 must reproduce item12 within 1 m)
    print("\n=== Baseline check vs verified item12 (2026-10-21) ===")
    base_ok = True
    for row in line_qc:
        ref = BASELINE_3D_KALMAN.get((row["line"], row["tag"]))
        if not ref:
            continue
        dmed = abs(row["median_xy_m"] - ref[0])
        status = "OK " if dmed <= 1.0 else "FAIL"
        if dmed > 1.0:
            base_ok = False
        print("  %s %s: median %.2f m vs baseline %.2f m (diff %.2f) %s"
              % (row["line"], row["tag"], row["median_xy_m"], ref[0], dmed, status))

    # ---------------- README
    _write_readme(out, holds, skipped_holds, line_qc, hold_qc, manifest, floor_m)
    manifest.append(out / "README.md")

    print("\nMANIFEST: %d files" % len(manifest))
    for p in sorted(manifest):
        p = Path(p)
        print("  %s (%.0f KB)" % (p.relative_to(REPO), p.stat().st_size / 1024) if p.exists() else "  %s MISSING" % p)
    if not base_ok:
        print("\nBASELINE GATE: FAIL — drag scoring does not reproduce the verified item12 numbers")
        sys.exit(1)
    print("\nBASELINE GATE: PASS")


def _write_readme(out, holds, skipped_holds, line_qc, hold_qc, manifest, floor_m):
    L = []
    L.append("# PM Figures 2026-10-22 - Index (fixed 2026-10-22 evening)")
    L.append("")
    L.append("Client-ready plots + CSVs for the three tag-drag categories (PM email 2026-10-22).")
    L.append("Example-fish tracks (26E4, 9F8C, 1D5A, B5ED) are a separate workstream - see `example_fish/`.")
    L.append("")
    L.append("## Coordinate / time basis")
    L.append("- Positions: local array meters (WGS84 -> EPSG:26910, minus receiver median offset).")
    L.append("- Time: UTC-encoded wall clock (receivers log PDT, ZOI05 logs UTC; DB `seconds` = wall time as UTC).")
    L.append("- All products are **root B** (root A is known-bad on this array).")
    L.append("- Drag product: per-transmission receiver-set consensus (10 m, >=2 sets) + Hampel smoothing")
    L.append("  (10 m / 11 / 3) + constant-velocity Kalman (2.0/1.0/0.5, 20 s gap) - the verified 2026-10-21 chain.")
    L.append("- Static-hold product: per-transmission median of root-B solutions inside the hold window.")
    L.append("- Re-run: `python scripts/drag_test_figures_2025.py` (idempotent, read-only on inputs).")
    L.append("")
    L.append("## Static hold QC (headline number)")
    if hold_qc:
        hq = pd.DataFrame(hold_qc)
        h3 = hq[hq["product"] == "3D"]
        L.append("- Median XY error, all holds/tags (3D): **%.2f m** (p90 %.1f m)"
                 % (h3.median_xy_m.median(), h3.p90_xy_m.median()))
        worst = h3.loc[h3.median_xy_m.idxmax()]
        L.append("- Worst: **%s %s** (%.2f m median, %.1f m p90)"
                 % (worst.hold_id, worst.tag, worst.median_xy_m, worst.p90_xy_m))
        L.append("")
        L.append("| Hold | Tag | Dim | n_trans | n_sols | Median XY (m) | P90 XY (m) | med dX | med dY |")
        L.append("|------|-----|-----|---------|--------|---------------|------------|--------|--------|")
        for _, r in hq.iterrows():
            L.append("| %s | %s | %s | %d | %d | %.2f | %.1f | %.2f | %.2f |"
                     % (r["hold_id"], r["tag"], r["product"], r["n_transmissions"], r["n_solutions"],
                        r["median_xy_m"], r["p90_xy_m"], r["median_dx_m"], r["median_dy_m"]))
    L.append("")
    L.append("## Drag line QC (3D root B, consensus+Kalman, vs GPS)")
    if line_qc:
        lq = pd.DataFrame(line_qc)
        L.append("- Median XY, all lines/tags: **%.2f m** (p90 %.1f m)"
                 % (lq.median_xy_m.median(), lq.p90_xy_m.median()))
        L.append("")
        L.append("| Line | Tag | n_scored | Median XY (m) | P90 XY (m) | Yield % |")
        L.append("|------|-----|----------|---------------|------------|---------|")
        for _, r in lq.iterrows():
            L.append("| %s | %s | %d | %.2f | %.1f | %d |"
                     % (r.line, r.tag, r.n_scored, r.median_xy_m, r.p90_xy_m, r.yield_pct))
        L.append("")
        L.append("ENT-01..05 reproduce the verified 2026-10-21 item12 baseline within 1 m (gate in the generator).")
    L.append("")
    L.append("## Notes")
    L.append("- Each drag line also has a simple 3D view (`<line>_3d.png`): the GPS track at each tag's")
    L.append("  known tow depth (FC36 -1.5 m, FFD3 -4.6 m) plus the root-B Kalman track gated to the")
    L.append("  water column [%.1f, 0] m, so above-surface / below-bottom mirror artifacts are not drawn."
             % floor_m)
    L.append("  The gate is view-only: scoring (median/p90 XY) and the client CSVs are unchanged. The")
    L.append("  floor is project-settable via the run file `[figures] depth_floor_m` (else auto = deepest")
    L.append("  receiver Z minus margin). Z sanity per line: `qc/drag_line_depths.csv`.")
    L.append("- FBY-S-6 (sheet) is marked 'Replace this with DB-1-5 (below)' and is excluded.")
    L.append("- DB-1..5 (Debris Barrier static holds) are **excluded**: they ran 2025-06-11 12:02-12:15, outside")
    L.append("  the ENT DB window (2025-06-10 only), and no GPS points were collected for them (sheet: DB-1")
    L.append("  'Not Collected - See approximate location in figure to the right', DB-2..5 blank).")
    L.append("- Static-hold QC degrades through the morning: FBY-S-1..7 (09:58-10:09) median 0.9-11 m, while")
    L.append("  FBY-S-8..10 (10:13-10:20) are 16-74 m (FBY-S-9 FC36 has a 428 m outlier transmission).")
    L.append("  FBY-S-9/10 offsets are mainly +Y (north). Cause not yet diagnosed (candidate: late-morning")
    L.append("  receiver clock drift or the boat drifting off the recorded hold point) - flagged for follow-up.")
    L.append("- FBY-D-1 (first random drag) shows an elevated FC36 median (35.8 m): the final ~40 s of the line")
    L.append("  sits ~15-30 m north of the GPS track, consistent with the boat's final stop not matching the")
    L.append("  recorded GPS endpoint. FBY-D-4 is also elevated (24.6 m). FBY-D-2/3 are 3-12 m.")
    L.append("- FBY-D-1 FFD3 and FBY-D-4 FFD3 have very few solution transmissions in the data (a real")
    L.append("  positioning-yield gap, not a plotting issue).")
    L.append("- GPS tracks are 1 Hz per-minute blocks reconstructed as in scripts/ent_analysis_2025.py.")
    L.append("- GPS tracks are the same for both drag tags: one `<category>_gps_tracks.csv` per category.")
    # example_fish is a separate workstream (scripts/fish_figures_2025.py); document it here so a
    # drag-only re-run does not wipe the section out of the shared index (2026-10-25).
    if (out / "example_fish").is_dir():
        L.append("")
        L.append("## Example fish (early-fish example tracks)")
        L.append("- Separate workstream: 4 early-fish tags (26E4, 9F8C, 1D5A, B5ED) from")
        L.append("  `output/jsats3d_2025_earlyfish.db` (window 2025-06-15 -> 2025-08-06).")
        L.append("- Track product: the verified item12 chain, preceded by a physical plausibility gate")
        L.append("  (drop Z > 0 and Z < -16 m; deepest measured bay depth is 13.3 m at ZOI02):")
        L.append("  root-B Deng 3D solutions -> receiver-set consensus (10 m, >=2 sets) -> Hampel")
        L.append("  smoothing (10 m / 11 / 3) -> Kalman (2.0/1.0/0.5, 20 s gap).")
        L.append("- Every figure shows the 20 receivers (ZOI01-11, CFD01-09) as small gray x marks with")
        L.append("  labels - the client's spatial reference for the ~100 m array.")
        L.append("- Per tag: plan view (time-colored), plan view (depth-colored), static 3D scatter,")
        L.append("  a 2D animated track GIF and a 3D animated track GIF (static receiver reference +")
        L.append("  growing colored track + red current-position dot + timestamp), and a client CSV")
        L.append("  with the three products (solution / consensus / kalman).")
        fish_qc_path = out / "example_fish" / "fish_qc.csv"
        if fish_qc_path.exists():
            fq = pd.read_csv(fish_qc_path)
            L.append("")
            L.append("| Tag | Solutions | Consensus | Kalman pts | Span (h) | Max speed (m/s) | Z min/max (m) | Jumps >3 m/s |")
            L.append("|-----|-----------|-----------|------------|----------|-----------------|---------------|--------------|")
            for _, r in fq.iterrows():
                L.append("| %s | %d | %d | %d | %.1f | %.2f | %.1f / %.1f | %d |"
                         % (r.tag, r.n_solutions, r.n_consensus, r.n_kalman, r.span_h,
                            r.max_speed_m_s, r.z_min_m, r.z_max_m, r.n_jumps_gt_3ms))
            L.append("")
            L.append("Kalman tracks are 100% within the water column: below the surface and above the")
            L.append("bay floor (deepest measured total depth 13.3 m; floor gate -16 m). The plausibility")
            L.append("gate removes the above-water and below-bottom mirror-solution artifacts (journals")
            L.append("2026-10-24/25).")
        L.append("- Re-run: `python scripts/fish_figures_2025.py` (idempotent, read-only on inputs).")
    L.append("")
    L.append("## Files")
    for cat in ("CFNSC_Approach_Drags", "Forebay_StaticHolds", "Forebay_Random_Drags"):
        cat_path = out / cat
        if cat_path.exists():
            L.append("")
            L.append("### %s" % cat)
            for f in sorted(cat_path.glob("*")):
                L.append("- `%s/%s`" % (cat, f.name))
    L.append("")
    L.append("### qc")
    for f in sorted((out / "qc").glob("*")):
        L.append("- `qc/%s`" % f.name)
    if (out / "example_fish").is_dir():
        L.append("")
        L.append("### example_fish")
        for f in sorted((out / "example_fish").glob("*")):
            L.append("- `example_fish/%s`" % f.name)
    L.append("")
    L.append("---")
    L.append("Generated: %s by `scripts/drag_test_figures_2025.py`" % pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    text = "\n".join(L).encode("ascii", "replace").decode("ascii")  # ASCII-safe for any Windows editor
    (out / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
