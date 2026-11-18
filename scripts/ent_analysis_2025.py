"""Read-only step 12: legacy Deng XYZ against ENT-01/ENT-05 GPS and recorded depths.

    python scripts/ent_analysis_2025.py --db output/<db>.db --only 12

Blue is the root-B per-transmission mean XYZ from tblPositions_Deng, matching legacy trajectory plotting.
Black is holdout GPS at the recorded tag depth. Truth never enters position solving.
Zoom and full-range plots plus horizontal, depth and XYZ scores go under output/2025_review/.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

REPO = Path(__file__).resolve().parents[1]
GPS_FILE = Path(r"K:\Jobs\5662\001\Data\DataTrans\2025_Data\CowlitzAT2025_Data_Deliverables\5_array_testing\array_testing_drag_GPS.csv")
TAGZ = {"FC36": -1.524, "FFD3": -4.572}
LINES = ("ENT-01", "ENT-05")


# ------------------------------------------------------------------------------------------------ data

class Data:
    def __init__(self, db, gps_file=GPS_FILE):
        con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)
        self.rec = pd.read_sql("select Rec_ID, X_t, Y_t, Z_t, easting, northing, X, Y from tblReceiver", con).set_index("Rec_ID")
        ox, oy = (self.rec.easting - self.rec.X).median(), (self.rec.northing - self.rec.Y).median()
        self.xyz = self.rec[["X_t", "Y_t", "Z_t"]]
        gps = pd.read_csv(gps_file)
        gx, gy = Transformer.from_crs(4326, 26910, always_xy=True).transform(gps.long.values, gps.lat.values)
        gps["x"], gps["y"] = gx - ox, gy - oy
        gps["minute"] = pd.to_datetime(gps.timestamp)
        self.tracks = {}
        for line in LINES:
            t = gps[gps.line_name == line].reset_index(drop=True)
            counts = t.groupby("minute").size()
            # 1 Hz: the partial first minute holds the last n seconds of its minute
            t["gtime"] = counts.index[0].tz_localize("UTC").timestamp() + 60 - counts.iloc[0] + np.arange(len(t))
            self.tracks[line] = t
        cols = [r[1] for r in con.execute("pragma table_info(tblDetectionFilterSecondary)")]
        want = [c for c in ("Tag_ID", "transNo", "Rec_ID", "seconds_fix", "det_rank", "delta_s", "multipath_prediction", "Amplitude") if c in cols]
        self.sec = pd.read_sql("select %s from tblDetectionFilterSecondary" % ", ".join(want), con)
        con.close()
        self.sec["seconds_fix"] = pd.to_numeric(self.sec.seconds_fix)
        fish = self.sec[self.sec.Tag_ID.isin(TAGZ)]
        self.fa = fish[fish.det_rank == 1].drop_duplicates(["Tag_ID", "transNo", "Rec_ID"])
        first = fish.groupby(["Tag_ID", "transNo"]).seconds_fix.min()
        self.tx = {}
        for line, t in self.tracks.items():
            for tag in TAGZ:
                f = first.loc[tag]
                self.tx[(line, tag)] = set(f[(f >= t.gtime.min() - 3) & (f <= t.gtime.max() + 3)].index)

    def truth(self, line, t):
        tr = self.tracks[line]
        ok = (t >= tr.gtime.min()) & (t <= tr.gtime.max())
        return np.interp(t, tr.gtime, tr.x), np.interp(t, tr.gtime, tr.y), ok


def legacy_receiver_set_consensus(positions, radius_m=10.0, minimum_sets=2):
    """Diagnostic only: retain positions backed by distinct nearby receiver sets."""
    if positions.empty:
        return positions.copy()
    required = {"transNo", "r0", "r1", "r2", "r3", "X", "Y", "Z", "t0", "in_hull"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError("Deng rows missing columns: %s" % sorted(missing))
    # PM 2026-10-14: no hull filter, keep both roots A and B
    valid = positions[np.isfinite(positions[["X", "Y", "Z", "t0"]]).all(axis=1)].copy()
    valid = valid[valid[["X", "Y", "Z"]].abs().lt(1000).all(axis=1)]
    valid["receiver_set"] = valid[["r0", "r1", "r2", "r3"]].apply(
        lambda row: "+".join(sorted(row.astype(str))), axis=1)
    accepted = []
    for trans_no, group in valid.groupby("transNo", sort=True):
        center = group[["X", "Y"]].median()
        nearby = group[np.hypot(group.X - center.X, group.Y - center.Y) <= radius_m]
        if nearby.receiver_set.nunique() < minimum_sets:
            continue
        med = nearby[["X", "Y", "Z", "t0"]].median()
        accepted.append({"transNo": trans_no, "x": med.X, "y": med.Y, "z": med.Z, "t0": med.t0,
                         "receiver_sets": int(nearby.receiver_set.nunique()),
                         "candidate_sets": int(group.receiver_set.nunique()), "n_combinations": int(len(nearby))})
    return pd.DataFrame(accepted).sort_values("t0").reset_index(drop=True) if accepted else pd.DataFrame()


# --------------------------------------------------------------------------- 3D window screen
# Clock-jump artifacts land kilometres from the array (worst documented: 548 km),
# while the 3D receiver volume is ~100 m across. An axis-aligned window around the
# receiver volume rejects them BEFORE the per-transmission consensus median, instead
# of relying on the median to survive gross errors. X/Y bounds = receivers_3d min/max
# (a fish inside the array's coverage cannot be far outside the receiver extent);
# the Z bound is NOT the receiver Z extent (fish swim above the shallowest hydrophone)
# but the existing surface/floor plausibility window (0 .. depth_floor_m), so the two
# already-validated Z thresholds are reused unchanged (PM 2026-11-14: drop clock-jump
# corrections, filter solutions in 3D before the median).


def receiver_3d_box(receivers):
    """Axis-aligned receiver volume: (xmin, xmax, ymin, ymax, zmin, zmax) from X_t/Y_t/Z_t.

    receivers: DataFrame with X_t, Y_t, Z_t columns (usually a tblReceiver slice for
    the configured receivers_3d). zmin/zmax describe hydrophone placement, not the
    fish's depth range -- callers pass the physical Z window (surface/floor) in.
    """
    cols = pd.DataFrame(receivers[["X_t", "Y_t", "Z_t"]]).dropna()
    if cols.empty:
        raise ValueError("no receivers with finite X_t/Y_t/Z_t")
    return (float(cols.X_t.min()), float(cols.X_t.max()),
            float(cols.Y_t.min()), float(cols.Y_t.max()),
            float(cols.Z_t.min()), float(cols.Z_t.max()))


def three_d_box_mask(positions, box, margin_m=0.0):
    """Boolean mask: True where a Deng row lies inside the axis-aligned 3D window.

    box = (xmin, xmax, ymin, ymax, zmin, zmax); margin_m widens every axis.
    Non-finite coordinates are rejected (False) -- missing stays missing, never kept.
    """
    m = float(margin_m)
    x0, x1, y0, y1, z0, z1 = (float(v) for v in box)
    x = pd.to_numeric(positions["X"], errors="coerce").to_numpy()
    y = pd.to_numeric(positions["Y"], errors="coerce").to_numpy()
    z = pd.to_numeric(positions["Z"], errors="coerce").to_numpy()
    return (np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            & (x >= x0 - m) & (x <= x1 + m) & (y >= y0 - m) & (y <= y1 + m)
            & (z >= z0 - m) & (z <= z1 + m))


def spatial_box(figs, receivers, floor_m, box_margin_m):
    """The 3D window box for the pre-consensus-median screen: (box, effective_margin, source).

    box = (xmin, xmax, ymin, ymax, zmin, z1) with Z always the [depth floor, 0]
    plausibility window. By default the X/Y extent is the receiver volume
    (min/max of ALL positioned receivers) and effective_margin = box_margin_m
    (widened by the mask). [figures] position_3d_box_xy = [x0, x1, y0, y1]
    overrides the extent with an explicit user-set window; the margin is then 0
    (the given coordinates ARE the final window).
    """
    explicit = figs.get("position_3d_box_xy") or None
    if explicit:
        if len(explicit) != 4:
            raise ValueError("position_3d_box_xy must be [x0, x1, y0, y1], got %r" % (explicit,))
        x0, x1, y0, y1 = (float(v) for v in explicit)
        return (x0, x1, y0, y1, float(floor_m), 0.0), 0.0, "explicit position_3d_box_xy"
    x0, x1, y0, y1, _rzmin, _rzmax = receiver_3d_box(receivers)
    return (x0, x1, y0, y1, float(floor_m), 0.0), float(box_margin_m), \
        ("min/max of all positioned receivers (entire array incl. the CFD row) + %.1f m margin; "
         "%d receivers" % (float(box_margin_m), len(receivers)))


def present_unpositioned_rows(detections, solutions, consensus, box_rejected=()):
    """Emissions of one tag heard inside the array but without a final position.

    detections: one tag's 3D-receiver detections; columns transNo, Rec_ID, seconds_fix.
    solutions: the tag's root-B 'solution found' rows (any transNo that ever solved).
    consensus: the final consensus frame (transNo that got a position).
    box_rejected: iterable of transNo whose EVERY solution fell outside the 3D window
        (only meaningful when the window screen is on).

    Returns DataFrame(transNo, t0, n_receivers, reason) sorted by t0, where
        t0           = median detection seconds_fix of the emission
        n_receivers  = distinct 3D receivers hearing the emission (presence evidence)
        reason       = 'outside_3d_box'  (solutions existed, all outside the window)
                     | 'not_consensus'   (solutions existed, none survived consensus)
                     | 'no_solution'     (no root-B solution at all)
    Emissions with a consensus position are omitted.
    """
    cols = ["transNo", "t0", "n_receivers", "reason"]
    if detections.empty:
        return pd.DataFrame(columns=cols)
    d = (detections.dropna(subset=["transNo"])
         .groupby("transNo")
         .agg(t0=("seconds_fix", "median"), n_receivers=("Rec_ID", "nunique"))
         .reset_index())
    fixed = (set(consensus["transNo"].astype(float)) if consensus is not None and not consensus.empty else set())
    solved = (set(pd.to_numeric(solutions["transNo"], errors="coerce").dropna())
              if solutions is not None and not solutions.empty else set())
    rejected = set(float(t) for t in box_rejected)
    rows = []
    for r in d.itertuples(index=False):
        t = float(r.transNo)
        if t in fixed:
            continue
        if t in rejected:
            reason = "outside_3d_box"
        elif t in solved:
            reason = "not_consensus"
        else:
            reason = "no_solution"
        rows.append((t, float(r.t0), int(r.n_receivers), reason))
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("t0").reset_index(drop=True) if len(out) else out


# ------------------------------------------------------------------------------------------------ solver













# ------------------------------------------------------------------------------------------------ items





























def item12(d, db, table="tblPositions_Deng", plot_tokens=None):
    """Legacy Deng root-B centroid XYZ, scored against holdout drag GPS and depths."""
    import matplotlib.pyplot as plt

    def score(line, tag, frame, method, expected_count):
        if frame.empty:
            return None, None
        tx, ty, ok = d.truth(line, frame.t0.to_numpy())
        result = frame.copy()
        result["gps_x"], result["gps_y"] = tx, ty
        result["err"] = np.hypot(result.x - tx, result.y - ty)
        result = result[ok].copy()
        if result.empty:
            return None, None
        depth_error = result.z - TAGZ[tag]
        row = dict(line=line, tag=tag, method=method, scored_transmissions=len(result),
                   yield_pct=round(100 * len(result) / max(expected_count, 1)),
               horizontal_error_median_m=round(result.err.median(), 2),
               horizontal_error_p90_m=round(result.err.quantile(.9), 1),
               estimated_z_median_m=round(result.z.median(), 1),
               depth_error_median_m=round(depth_error.median(), 1),
               absolute_depth_error_p90_m=round(depth_error.abs().quantile(.9), 1),
               xyz_error_median_m=round(np.hypot(result.err, depth_error).median(), 2),
               xyz_error_p90_m=round(np.hypot(result.err, depth_error).quantile(.9), 1))
        return row, result

    def smooth_track(track, jump_limit_m=15.0, jump_window=7, smooth_window=5, max_gap_s=8.0):
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
        """Constant-velocity Kalman filter on consensus smoothed positions.

        State: [x, y, z, vx, vy, vz] (6D).
        Process noise controls how much the filter trusts the model vs measurements.
        measurement_noise is XY uncertainty; z_noise is depth uncertainty.
        Splits into segments when time gap exceeds max_gap_s.
        """
        if len(track) < 3:
            return track.copy(), 0
        track = track.sort_values("t0").reset_index(drop=True).copy()

        # Split into segments by time gap
        track["segment"] = (track.t0.diff().fillna(0) > max_gap_s).cumsum()

        results = []
        for seg_id, seg in track.groupby("segment", sort=False):
            if len(seg) < 2:
                results.append(seg)
                continue
            pos = seg[["x", "y", "z"]].values.astype(float)
            dt = np.diff(seg.t0.values.astype(float))
            dt = np.maximum(dt, 0.1)  # minimum 0.1s to avoid division issues

            n = len(pos)
            # State: [x, y, z, vx, vy, vz]
            F = np.eye(6)
            H = np.eye(3, 6)  # observe only position

            # Process noise covariance (Q)
            Q = np.eye(6)
            Q[:3, :3] *= process_noise ** 2  # position uncertainty
            Q[3:, 3:] *= process_noise ** 2  # velocity uncertainty

            # Measurement noise covariance (R)
            R = np.eye(3)
            R[0, 0] = measurement_noise ** 2
            R[1, 1] = measurement_noise ** 2
            R[2, 2] = z_noise ** 2

            # Initial state
            x_est = np.array([pos[0, 0], pos[0, 1], pos[0, 2], 0.0, 0.0, 0.0])
            P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])  # initial covariance

            smoothed = np.zeros((n, 3))
            smoothed[0] = pos[0]

            for i in range(1, n):
                # Predict
                F[0, 3] = dt[i - 1]
                F[1, 4] = dt[i - 1]
                F[2, 5] = dt[i - 1]
                x_est = F @ x_est
                P = F @ P @ F.T + Q

                # Reset F for next iteration
                F[0, 3] = 0
                F[1, 4] = 0
                F[2, 5] = 0

                # Update
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

    print("\n=== ITEM 12: Legacy Deng pipeline XYZ vs holdout drag GPS and recorded depths (PNGs in %s) ===" % OUT)
    con = sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)
    p3 = pd.read_sql("select * from %s where comment = 'solution found'" % table, con)
    con.close()
    if "r3" not in p3:  # Deng2D uses sets of three
        p3["r3"] = ""
    p3["t0"] = pd.to_numeric(p3.ToA) - pd.to_numeric(p3.T01)
    numeric_cols = ["X", "Y", "Z", "t0"]
    for c in numeric_cols:
        if c in p3.columns:
            p3[c] = pd.to_numeric(p3[c], errors="coerce")
    legacy_a = p3[(p3.solution == "A") & p3[numeric_cols].notna().all(axis=1)].copy()

    rows = []
    for line in LINES:
        for tag in TAGZ:
            ids, zt = d.tx[(line, tag)], TAGZ[tag]
            legacy_rows = legacy_a[(legacy_a.Tag_ID == tag) & legacy_a.transNo.isin(ids)]
            g_legacy = legacy_rows.groupby("transNo").agg(
                x=("X", "mean"), y=("Y", "mean"), z=("Z", "mean"),
                t0=("t0", "median"), n=("X", "size")).reset_index().sort_values("t0")
            hull_rows = legacy_rows[
                legacy_rows.in_hull.astype(str).str.lower().isin(["1", "true"])
                & np.isfinite(legacy_rows[["X", "Y", "Z"]]).all(axis=1)
                & legacy_rows[["X", "Y", "Z"]].abs().lt(1000).all(axis=1)
            ].copy()
            hull_rows["receiver_set"] = hull_rows[["r0", "r1", "r2", "r3"]].astype(str).apply(
                lambda row: "+".join(sorted(row.values)), axis=1, result_type="reduce")
            g_hull = hull_rows.groupby("transNo").agg(
                x=("X", "median"), y=("Y", "median"), z=("Z", "median"),
                t0=("t0", "median"), n=("X", "size"),
                receiver_sets=("receiver_set", "nunique")).reset_index().sort_values("t0")
            g_supported = legacy_receiver_set_consensus(legacy_rows)
            g_smooth, rejected = smooth_track(g_supported)
            g_kalman, _ = kalman_filter_track(g_smooth, process_noise=0.5, measurement_noise=2.0, z_noise=1.0)
            if g_legacy.empty:
                print("  %s %s: no pipeline solutions" % (line, tag))
                continue
            variants = (("3D legacy A centroid", g_legacy, "3D_legacy_A"),
                        ("3D hull/sentinel screened median", g_hull, "3D_hull_screened"),
                        ("3D independent-set consensus", g_supported, "3D_set_consensus"),
                        ("3D consensus/Hampel smoothed", g_smooth, "3D_consensus_smoothed"),
                        ("3D Kalman filtered", g_kalman, "3D_kalman"))
            scored = {}
            for name, g, file_token in variants:
                if g.empty:
                    continue
                row, scored[name] = score(line, tag, g, name, len(ids))
                if row:
                    row["outliers_removed"] = rejected if "smoothed" in name else 0
                    row["candidate_transmissions"] = int(g_legacy.transNo.nunique())
                    if "receiver_sets" in g:
                        row["median_supporting_receiver_sets"] = float(g.receiver_sets.median())
                    rows.append(row)
                    scored[name].assign(line=line, tag=tag).to_csv(
                        OUT / ("item12_pipeline_%s_%s_%s.csv" % (file_token, line, tag)), index=False)
            tr = d.tracks[line]
            x_min = min(float(tr.x.min()), float(d.xyz.X_t.min())) - 15.0
            x_max = max(float(tr.x.max()), float(d.xyz.X_t.max())) + 15.0
            y_min = min(float(tr.y.min()), float(d.xyz.Y_t.min())) - 15.0
            y_max = max(float(tr.y.max()), float(d.xyz.Y_t.max())) + 15.0
            z_min, z_max = min(float(d.xyz.Z_t.min()), zt) - 5.0, 2.0
            for name, _, file_token in variants:
                if name not in scored or (plot_tokens and file_token not in plot_tokens):
                    continue
                blue_track = scored[name]
                fig = plt.figure(figsize=(9, 7))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(tr.x, tr.y, np.full(len(tr), zt), color="black", linestyle="-", linewidth=3.5,
                        zorder=3, label="GPS truth at recorded depth")
                ax.plot(blue_track.x, blue_track.y, blue_track.z, color="blue", linestyle="-", linewidth=2.5,
                        label="Pipeline Deng XYZ estimate")
                summary = next(r for r in rows if r["line"] == line and r["tag"] == tag and r["method"] == name)
                ax.set_title("%s | %s (%s)\n%d/%d fixes; median XY error %.1f m" %
                         (line, tag, name, summary["scored_transmissions"], len(ids),
                          summary["horizontal_error_median_m"]), fontsize=11)
                ax.set_xlabel("X (m)", labelpad=3)
                ax.set_ylabel("Y (m)", labelpad=3)
                ax.set_zlabel("Depth Z (m)", labelpad=3)
                ax.grid(False)
                ax.legend(loc="upper left", fontsize=8, frameon=False)
                ax.view_init(elev=35, azim=-60)
                full_png = OUT / ("item12_pipeline3d_%s_%s_%s_full_range.png" % (file_token, line, tag))
                zoom_png = OUT / ("item12_pipeline3d_%s_%s_%s.png" % (file_token, line, tag))
                fig.savefig(full_png, dpi=150, bbox_inches="tight")
                ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
                ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
                fig.savefig(zoom_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                # 2D top-down plot
                fig2 = plt.figure(figsize=(9, 7))
                ax2 = fig2.add_subplot(111)
                ax2.plot(tr.x, tr.y, color="black", linestyle="-", linewidth=3.5,
                         zorder=3, label="GPS truth")
                ax2.plot(blue_track.x, blue_track.y, color="blue", linestyle="-", linewidth=2.5,
                         zorder=2, label="Pipeline Deng XY estimate")
                ax2.set_title("%s | %s (%s)\n%d/%d fixes; median XY error %.1f m" %
                         (line, tag, name, summary["scored_transmissions"], len(ids),
                          summary["horizontal_error_median_m"]), fontsize=11)
                ax2.set_xlabel("X (m)")
                ax2.set_ylabel("Y (m)")
                ax2.set_aspect("equal")
                ax2.grid(True)
                ax2.legend(loc="upper left", fontsize=8, frameon=False)
                ax2.set_xlim(x_min, x_max)
                ax2.set_ylim(y_min, y_max)
                full_2d = OUT / ("item12_pipeline2d_%s_%s_%s_full_range.png" % (file_token, line, tag))
                zoom_2d = OUT / ("item12_pipeline2d_%s_%s_%s.png" % (file_token, line, tag))
                fig2.savefig(full_2d, dpi=150, bbox_inches="tight")
                fig2.savefig(zoom_2d, dpi=150, bbox_inches="tight")
                plt.close(fig2)
            print("  %s %s: raw %d, hull-valid %d, Hampel removed %d" %
                  (line, tag, len(g_legacy), len(g_hull), rejected))
    tab = pd.DataFrame(rows)
    print("\nDefinitions: scored_transmissions = fixes with matching GPS time; yield_pct = scored / GPS-window transmissions; "
          "horizontal_error_* = XY distance to GPS; depth_error = estimated Z minus recorded depth (negative means too deep); "
          "xyz_error_* combines horizontal and depth error. Filtered/smoothed variants are diagnostics, not pipeline outputs.")
    print(tab.to_string(index=False))
    tab.to_csv(OUT / "item12_pipeline_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(REPO / "output" / "jsats3d_2025_tagdrag_ent_pipeline.db"))
    ap.add_argument("--gps", default=str(GPS_FILE), help="holdout GPS file; never used to solve positions")
    ap.add_argument("--only", nargs="+", type=int, choices=[12], default=[12])
    ap.add_argument("--out", default="analysis", help="folder name under output/2025_review for the CSVs")
    ap.add_argument("--table", default="tblPositions_Deng", choices=["tblPositions_Deng", "tblPositions_DengDBSCAN", "tblPositions_Deng2D"])
    ap.add_argument("--plots", nargs="+", default=None, help="plot file tokens to draw, e.g. 3D_legacy_B (default all)")
    ap.add_argument("--lines", nargs="+", default=["ENT-01", "ENT-05"], help="drag lines to score and plot")
    args = ap.parse_args()
    global OUT
    OUT = REPO / "output" / "2025_review" / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    global LINES
    LINES = tuple(args.lines)
    d = Data(args.db, args.gps)
    print("database %s | tags %s | lines %s | transmissions in the line windows: %s" % (args.db, list(TAGZ), LINES, {k: len(v) for k, v in d.tx.items()}))
    if 12 in args.only:
        item12(d, args.db, args.table, args.plots)


if __name__ == "__main__":
    main()
