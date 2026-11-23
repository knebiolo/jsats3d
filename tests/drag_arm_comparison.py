"""Drag-tag method comparison (test tool, not a pipeline step; not collected by pytest).

Compares positioning methods on the same tag drags and static holds: one database per arm, all copies of one base
database (same detections, clock fix and multipath; only the positioning or filtering differs). Uses the scoring
chain of scripts/drag_test_figures_2025.py unchanged (root-B receiver-set consensus -> smooth -> Kalman vs GPS for
drag lines; per-transmission medians for holds).

    python tests/drag_arm_comparison.py --out OUT_DIR prior=PATH\\prior.db cap8=PATH\\cap8.db cap14=PATH\\cap14.db

Writes <line>_<tag>.png and <hold>_<tag>.png (GPS truth plus each arm; holds without a GPS point, DB-1..5, show only
the arms), qc/arm_drag_errors.csv, qc/arm_hold_errors.csv and qc/arm_summary.csv.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import drag_test_figures_2025 as dtf  # noqa: E402

ARM_STYLE = (("#7f7f7f", "o"), ("#1f77b4", "s"), ("#ff7f0e", "^"), ("#2ca02c", "D"), ("#9467bd", "v"))


def timed_holds(dm):
    """Every Forebay_StaticHolds row with Date/Start/End; x = y = None when the sheet has no GPS point (DB-1..5)."""
    wb = openpyxl.load_workbook(str(dtf.SHEETS_FILE), read_only=True)
    rows = list(wb["Forebay_StaticHolds"].iter_rows(values_only=True))
    wb.close()
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    tr = Transformer.from_crs(4326, 26910, always_xy=True)
    holds = []
    for row in rows[1:]:
        d = dict(zip(header, row))
        tid = str(d.get("Test ID") or "").strip()
        lat, lon = d.get("Lat"), d.get("Lon")
        date, st, et = d.get("Date"), d.get("Start Time"), d.get("End Time")
        if not (tid and isinstance(date, datetime.datetime) and isinstance(st, datetime.time)
                and isinstance(et, datetime.time)):
            continue
        located = isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and not pd.isna(lat) and not pd.isna(lon)
        x = y = None
        if located:
            hx, hy = tr.transform(float(lon), float(lat))
            x, y = hx - dm.ox, hy - dm.oy
        holds.append(dict(hold_id=tid, x=x, y=y,
                          t0=(pd.Timestamp(date) + pd.Timedelta(str(st))).tz_localize("UTC").timestamp(),
                          t1=(pd.Timestamp(date) + pd.Timedelta(str(et))).tz_localize("UTC").timestamp(),
                          start=str(st)[:8], end=str(et)[:8], date=str(pd.Timestamp(date).date())))
    return holds


def positions_minutes(db):
    """Minutes the positions step took for this DB (from its pipeline status record), else None."""
    status = Path(db).parent / ("%s.pipeline_status.json" % Path(db).stem)
    try:
        steps = json.loads(status.read_text(encoding="utf-8")).get("steps", {})
    except (OSError, json.JSONDecodeError):
        return None
    return (steps.get("positions", {}) if isinstance(steps, dict) else {}).get("minutes")


def draw_receivers(ax, xyz):
    for rec_id, row in xyz.iterrows():
        ax.plot(row.X_t, row.Y_t, "k^", markersize=6, zorder=4)
        ax.annotate(rec_id, (row.X_t + 2, row.Y_t + 2), fontsize=7, color="0.3")


def finish(ax, title, path):
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def compare_arms(arms, out):
    out = Path(out)
    (out / "qc").mkdir(parents=True, exist_ok=True)
    dms = {}
    for name, db in arms.items():
        print("loading arm %s: %s" % (name, db), flush=True)
        dms[name] = dtf.DataManager(db)
    ref = next(iter(dms.values()))
    style = {name: ARM_STYLE[i % len(ARM_STYLE)] for i, name in enumerate(arms)}
    holds = timed_holds(ref)
    lines = [l for ids in dtf.CATEGORIES.values() for l in ids if l in ref.tracks]
    drag_rows, hold_rows = [], []

    for line in lines:
        tr = ref.tracks[line]
        for tag in dtf.TAGS:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(tr.x, tr.y, "-", color="black", linewidth=2.5, zorder=2, label="GPS truth")
            for name, dm in dms.items():
                p = dtf.line_product(dm, line, tag)
                gk, s = p["kalman"], p["scored"]
                n_ids = len(p["ids"])
                row = dict(test_id=line, tag=tag, arm=name, candidate_pings=n_ids, n_track=int(len(gk)),
                           n_scored=0 if s is None else int(len(s)),
                           yield_pct=round(100.0 * (0 if s is None else len(s)) / max(n_ids, 1), 1),
                           median_xy_m=np.nan if s is None or s.empty else round(float(s.err.median()), 2),
                           p90_xy_m=np.nan if s is None or s.empty else round(float(s.err.quantile(0.9)), 2))
                drag_rows.append(row)
                if gk.empty:
                    continue
                col, mk = style[name]
                t = gk.sort_values("t0")
                ax.plot(t.x, t.y, "-", color=col, linewidth=0.8, alpha=0.6, zorder=3)
                ax.scatter(t.x, t.y, color=col, marker=mk, s=22, alpha=0.8, zorder=4,
                           label="%s  n=%d  median %.1f m  p90 %.1f m" % (name, row["n_scored"], row["median_xy_m"],
                                                                         row["p90_xy_m"]))
            draw_receivers(ax, ref.xyz)
            start = pd.Timestamp(tr.gtime.min(), unit="s").strftime("%Y-%m-%d %H:%M:%S")
            end = pd.Timestamp(tr.gtime.max(), unit="s").strftime("%H:%M:%S")
            finish(ax, "%s  %s  %s to %s local time (PDT)\nerrors: Kalman track vs GPS truth" % (line, tag, start, end),
                   out / ("%s_%s.png" % (line, tag)))

    for hold in holds:
        known = hold["x"] is not None
        for tag in dtf.TAGS:
            fig, ax = plt.subplots(figsize=(10, 10))
            if known:
                ax.plot(hold["x"], hold["y"], "k*", markersize=24, zorder=6, label="GPS truth")
                for r in (10, 20):
                    ax.add_patch(Circle((hold["x"], hold["y"]), r, fill=False, color="0.4", linestyle="--", zorder=1))
            for name, dm in dms.items():
                g = dtf.hold_product(dm, dict(hold, x=hold["x"] or 0.0, y=hold["y"] or 0.0), tag, dm.p3b)
                row = dict(test_id=hold["hold_id"], tag=tag, arm=name, gps_point=known,
                           n_transmissions=0 if g is None else int(len(g)))
                if g is None or g.empty:
                    hold_rows.append(row)
                    continue
                mx, my, mz = float(g.x.median()), float(g.y.median()), float(g.z.median())
                row.update(median_x_m=round(mx, 2), median_y_m=round(my, 2), median_z_m=round(mz, 2),
                           spread_xy_m=round(float(np.hypot(g.x - mx, g.y - my).median()), 2))
                if known:
                    row.update(median_xy_m=round(float(g.err.median()), 2), p90_xy_m=round(float(g.err.quantile(0.9)), 2),
                               arm_median_error_m=round(float(np.hypot(mx - hold["x"], my - hold["y"])), 2))
                hold_rows.append(row)
                col, mk = style[name]
                ax.scatter(g.x, g.y, color=col, marker=mk, s=18, alpha=0.6, zorder=3,
                           label="%s  n=%d%s" % (name, len(g), ("  median %.1f m" % row["median_xy_m"]) if known else ""))
                ax.scatter([mx], [my], color=col, marker=mk, s=260, edgecolors="black", linewidths=1.5, zorder=5)
            draw_receivers(ax, ref.xyz)
            sub = ("large markers: arm medians; circles: 10 m and 20 m" if known
                   else "no GPS truth; large markers: arm medians")
            finish(ax, "%s  %s  %s %s to %s local time (PDT)\n%s" % (hold["hold_id"], tag, hold["date"], hold["start"],
                                                                     hold["end"], sub),
                   out / ("%s_%s.png" % (hold["hold_id"], tag)))

    drag = pd.DataFrame(drag_rows)
    hold_df = pd.DataFrame(hold_rows)
    drag.to_csv(out / "qc" / "arm_drag_errors.csv", index=False)
    hold_df.to_csv(out / "qc" / "arm_hold_errors.csv", index=False)
    summary = []
    for name, db in arms.items():
        d = drag[drag.arm == name]
        h = hold_df[(hold_df.arm == name) & hold_df.gps_point] if not hold_df.empty else hold_df
        summary.append(dict(arm=name, db=str(db), positions_step_min=positions_minutes(db),
                            drag_scored=int(d.n_scored.sum()), drag_candidate_pings=int(d.candidate_pings.sum()),
                            drag_median_of_medians_m=round(float(d.median_xy_m.median()), 2),
                            drag_median_p90_m=round(float(d.p90_xy_m.median()), 2),
                            hold_median_of_medians_m=round(float(h.median_xy_m.median()), 2) if "median_xy_m" in h else np.nan,
                            hold_transmissions=int(h.n_transmissions.sum()) if not h.empty else 0))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "qc" / "arm_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== drag lines (Kalman track vs GPS) ===")
    print(drag.pivot_table(index=["test_id", "tag"], columns="arm", values=["median_xy_m", "n_scored"],
                           sort=False).to_string())
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\nwrote %d drag and %d hold plots to %s" % (len(lines) * len(dtf.TAGS), len(holds) * len(dtf.TAGS), out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", metavar="NAME=DB", help="one database per arm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    compare_arms({k: Path(v) for k, v in (a.split("=", 1) for a in args.arms)}, Path(args.out))


if __name__ == "__main__":
    main()
