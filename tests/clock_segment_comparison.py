"""Clock-filter comparison figures (test tool, not a pipeline step; not collected by pytest).
Baseline (original 5-pass clock DBSCAN) vs segmented clock DBSCAN ("midpoint holds"); both databases must be copies
of one base build. Restored and trimmed from commit cfb7dbb (2026-11-24).

    python tests/clock_segment_comparison.py --baseline DB --segmented DB --out DIR

For each drag line (GPS window) and tag, and for each clock-step event in the data:
    <tag>_<label>_plan.png   2x2 plan view: per-transmission median X/Y of root-B solutions (before position DBSCAN,
                             depth filtering or smoothing); red rings = pings inside the highlighted window
    <tag>_<label>_event.png  ping coverage (root B, any solution), median ping depth, receiver clock residual/correction
Clock steps are found from the stored residuals (tblDetectionClockFixed) as consecutive changes >= 50 ms, the rule
the segmented filter uses.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from jsats3d.jsats3d import _interpolate_clock_segments  # noqa: E402
import drag_test_figures_2025 as dtf  # noqa: E402

FLOOR_M = -16.0
STEP_S = 0.05
PAD_S = 300.0
ARMS = ("baseline", "segmented")
LABEL = {"baseline": "Baseline", "segmented": "Midpoint holds"}
COLOR = {"baseline": "tab:orange", "segmented": "tab:green"}


def hhmm(t):
    return pd.to_datetime(t, unit="s")


def connect(db):
    return sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)


def pings(db):
    """Per transmission: root-B solutions found and their median X/Y/Z/t0."""
    con = connect(db)
    p = pd.read_sql("select Tag_ID, transNo, X, Y, Z, ToA, T01 from tblPositions_Deng where solution = 'B' "
                    "and comment = 'solution found'", con)
    con.close()
    for c in ("transNo", "X", "Y", "Z", "ToA", "T01"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["t0"] = p.ToA - p.T01
    p = p[np.isfinite(p[["X", "Y", "Z", "t0"]]).all(axis=1)]
    return p.groupby(["Tag_ID", "transNo"]).agg(x=("X", "median"), y=("Y", "median"), z=("Z", "median"),
                                                t0=("t0", "median"), n=("X", "size")).reset_index().sort_values("t0")


def clock_rows(db):
    con = connect(db)
    cols = {r[1] for r in con.execute("pragma table_info(tblDetectionClockFixed)")}
    extra = "".join(", " + c for c in ("clock_segment", "clock_segment_start", "clock_segment_end") if c in cols)
    k = pd.read_sql("select Rec_ID, seconds, seconds_residual%s from tblDetectionClockFixed" % extra, con)
    con.close()
    return k


def correction(knots, rec, t):
    """Fish clock correction of a receiver at times t, as multipath_data_object applies it."""
    k = knots[knots.Rec_ID == rec].dropna(subset=["seconds", "seconds_residual"])
    if k.empty:
        return np.full(len(t), np.nan)
    if "clock_segment" in k.columns:
        return _interpolate_clock_segments(k, np.asarray(t, float))
    k = k.sort_values("seconds").drop_duplicates("seconds")
    return np.interp(t, k.seconds, k.seconds_residual, left=k.seconds_residual.iloc[0], right=k.seconds_residual.iloc[-1])


def clock_steps(knots):
    """Consecutive residual changes >= STEP_S per receiver: (Rec_ID, t_before, t_after, step_s)."""
    rows = []
    for rec, k in knots.dropna(subset=["seconds_residual"]).groupby("Rec_ID"):
        k = k.sort_values("seconds").drop_duplicates("seconds")
        d = k.seconds_residual.diff()
        for i in np.flatnonzero(d.abs().to_numpy() >= STEP_S):
            rows.append(dict(Rec_ID=rec, t_before=float(k.seconds.iloc[i - 1]), t_after=float(k.seconds.iloc[i]),
                             step_s=float(d.iloc[i])))
    return pd.DataFrame(rows, columns=["Rec_ID", "t_before", "t_after", "step_s"])


def plan_figure(p, rec, tag, t0, t1, win, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw={"height_ratios": [1.5, 1]})
    xl = (rec.X_t.min() - 25, rec.X_t.max() + 25)
    yl = (rec.Y_t.min() - 25, rec.Y_t.max() + 25)
    sub = {a: p[a][(p[a].Tag_ID == tag) & (p[a].t0 >= t0) & (p[a].t0 <= t1)] for a in ARMS}
    both = pd.concat(sub.values())
    full = ((both.x.min() - 10, both.x.max() + 10), (both.y.min() - 10, both.y.max() + 10)) if len(both) else (xl, yl)
    sc = None
    for col, a in enumerate(ARMS):
        d = sub[a]
        inwin = (d.t0 >= win[0]) & (d.t0 <= win[1])
        for row, (lx, ly) in enumerate(((xl, yl), full)):
            ax = axes[row, col]
            sc = ax.scatter(d.x, d.y, c=d.t0, cmap="viridis", vmin=t0, vmax=t1, s=14, alpha=0.75, zorder=3)
            ax.scatter(d.x[inwin], d.y[inwin], s=60, facecolors="none", edgecolors="red", linewidths=1.2, zorder=4)
            for rid, r in rec.iterrows():
                ax.plot(r.X_t, r.Y_t, "x", color="0.4", ms=5, zorder=2)
                if row == 0:
                    ax.annotate(rid, (r.X_t + 1.5, r.Y_t + 1.5), fontsize=6.5, color="0.4")
            ax.set_xlim(*lx)
            ax.set_ylim(*ly)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.set_xlabel("Array-local X (m)")
            ax.set_ylabel("Array-local Y (m)")
            if row == 0:
                outside = int((~d.x.between(*xl) | ~d.y.between(*yl)).sum())
                ax.set_title("%s - array view\n%d pings; %d in window; %d outside this view"
                             % (LABEL[a], len(d), int(inwin.sum()), outside))
            else:
                ax.set_title("%s - full range (all plotted pings)" % LABEL[a])
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    ticks = np.linspace(t0, t1, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([hhmm(t).strftime("%H:%M") for t in ticks])
    cb.set_label("%s study wall time" % hhmm(t0).strftime("%Y-%m-%d"))
    fig.suptitle("%s plan view: baseline vs midpoint holds | %s\nMedian X/Y of successful root-B solutions per "
                 "transmission; before position DBSCAN, depth filtering or smoothing\nRed rings: %s %s-%s. "
                 "Identical axes per row; zoom limits are not data filtering."
                 % (tag, hhmm(t0).strftime("%B %d, %H:%M") + "-" + hhmm(t1).strftime("%H:%M"), title,
                    hhmm(win[0]).strftime("%H:%M:%S"), hhmm(win[1]).strftime("%H:%M:%S")), fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def event_figure(p, knots, tag, rec_id, t0, t1, win, path, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    grid = np.arange(t0, t1, 5.0)
    for i, a in enumerate(ARMS):
        d = p[a][(p[a].Tag_ID == tag) & (p[a].t0 >= t0) & (p[a].t0 <= t1)]
        axes[0].plot(hhmm(d.t0), np.full(len(d), len(ARMS) - 1 - i), "o", ms=3, color=COLOR[a], label=LABEL[a])
        axes[1].plot(hhmm(d.t0), d.z, "o", ms=3, alpha=0.6, color=COLOR[a], label=LABEL[a])
        k = knots[a][(knots[a].Rec_ID == rec_id) & (knots[a].seconds >= t0) & (knots[a].seconds <= t1)]
        axes[2].plot(hhmm(grid), correction(knots[a], rec_id, grid), "-", color=COLOR[a], lw=1.4,
                     label="%s correction" % LABEL[a])
        axes[2].plot(hhmm(k.seconds), k.seconds_residual, "o", ms=4, color=COLOR[a])
    axes[0].set_yticks(range(len(ARMS)))
    axes[0].set_yticklabels([LABEL[a] for a in reversed(ARMS)])
    axes[0].set_title("%s: transmission coverage (root B, any solution)" % tag)
    axes[1].axhline(0, ls=":", color="0.3")
    axes[1].axhline(FLOOR_M, ls=":", color="0.3")
    axes[1].set_ylim(-25, 10)
    axes[1].set_ylabel("Median ping depth Z (m)")
    axes[2].set_ylabel("%s clock residual (s)" % rec_id)
    axes[2].set_xlabel("%s study wall time" % hhmm(t0).strftime("%Y-%m-%d"))
    for ax in axes:
        ax.axvspan(hhmm(win[0]), hhmm(win[1]), color="red", alpha=0.12)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title + "\nJump times are estimated; increased coverage is not an accuracy validation", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def pick_receiver(knots, used, lo, hi):
    """Positioning receiver whose correction differs most between the arms inside [lo, hi]."""
    grid = np.arange(lo, hi, 5.0)
    best, score = sorted(used)[0], -1.0
    for r in sorted(used):
        diff = np.abs(correction(knots["baseline"], r, grid) - correction(knots["segmented"], r, grid))
        s = float(np.nanmax(diff)) if np.isfinite(diff).any() else -1.0
        if s > score:
            best, score = r, s
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=6, help="clock-step events to plot")
    ap.add_argument("--no-drags", action="store_true", help="skip the drag-line windows (fish databases)")
    ap.add_argument("--window", nargs=5, action="append", default=[], metavar=("TAG", "START", "END", "HL_START", "HL_END"),
                    help="extra named window: plot START..END, highlight HL_START..HL_END (e.g. a known gap)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dbs = {"baseline": Path(args.baseline), "segmented": Path(args.segmented)}
    p = {a: pings(db) for a, db in dbs.items()}
    knots = {a: clock_rows(db) for a, db in dbs.items()}
    con = connect(dbs["baseline"])
    used = set(pd.read_sql("select r0 r from tblPositions_Deng union select r1 from tblPositions_Deng union "
                           "select r2 from tblPositions_Deng union select r3 from tblPositions_Deng", con).r.dropna())
    rec = pd.read_sql("select Rec_ID, X_t, Y_t from tblReceiver", con).set_index("Rec_ID")
    con.close()
    rec = rec.loc[[r for r in rec.index if r in used]]
    tags = sorted(set(p["baseline"].Tag_ID) | set(p["segmented"].Tag_ID))
    rows = []

    def window_row(kind, label, tag, lo, hi):
        row = dict(kind=kind, label=label, tag=tag, start=str(hhmm(lo)), end=str(hhmm(hi)))
        for a in ARMS:
            d = p[a][(p[a].Tag_ID == tag) & (p[a].t0 >= lo) & (p[a].t0 <= hi)]
            row["%s_pings" % a] = int(len(d))
            row["%s_in_water_pct" % a] = round(100 * float(d.z.between(FLOOR_M, 0).mean()), 1) if len(d) else np.nan
        return row

    # ---- named windows (e.g. a known gap)
    for tag, start, end, hs, he in args.window:
        lo, hi = pd.Timestamp(start).timestamp(), pd.Timestamp(end).timestamp()
        win = (pd.Timestamp(hs).timestamp(), pd.Timestamp(he).timestamp())
        label = "window_%s" % hhmm(lo).strftime("%Y%m%d_%H%M")
        r = pick_receiver(knots, used, *win)
        plan_figure(p, rec, tag, lo, hi, win, "window", out / ("%s_%s_plan.png" % (tag, label)))
        event_figure(p, knots, tag, r, lo, hi, win, out / ("%s_%s_event.png" % (tag, label)),
                     "%s: baseline (5-pass clock DBSCAN) vs midpoint holds; shaded = highlighted window" % tag)
        rows.append(window_row("window", label, tag, *win))

    # ---- drag lines (GPS window)
    dm = None if args.no_drags else dtf.DataManager(dbs["baseline"])
    for line in [] if dm is None else [l for ids in dtf.CATEGORIES.values() for l in ids if l in dm.tracks]:
        g0, g1 = float(dm.tracks[line].gtime.min()), float(dm.tracks[line].gtime.max())
        lo, hi = g0 - PAD_S, g1 + PAD_S
        r = pick_receiver(knots, used, lo, hi)
        for tag in tags:
            plan_figure(p, rec, tag, lo, hi, (g0, g1), "%s GPS window" % line, out / ("%s_%s_plan.png" % (tag, line)))
            event_figure(p, knots, tag, r, lo, hi, (g0, g1), out / ("%s_%s_event.png" % (tag, line)),
                         "%s %s: baseline (5-pass clock DBSCAN) vs midpoint holds; shaded = GPS window" % (tag, line))
            rows.append(window_row("drag", line, tag, g0, g1))

    # ---- clock steps (either arm, positioning receivers); one figure per event (steps within 10 min grouped)
    steps = pd.concat([clock_steps(knots[a]) for a in ARMS]).drop_duplicates()
    steps = steps[steps.Rec_ID.isin(used)].sort_values("t_after")
    events, cur = [], None
    for s in steps.itertuples():
        if cur is None or s.t_before > cur["t_after"] + 600:
            cur = dict(t_before=s.t_before, t_after=s.t_after, recs={s.Rec_ID}, big=(abs(s.step_s), s.Rec_ID))
            events.append(cur)
        else:
            cur["t_after"] = max(cur["t_after"], s.t_after)
            cur["recs"].add(s.Rec_ID)
            cur["big"] = max(cur["big"], (abs(s.step_s), s.Rec_ID))
    print("clock-step events on positioning receivers: %d" % len(events))
    plotted = 0
    for e in sorted(events, key=lambda e: -len(e["recs"])):
        lo, hi = e["t_before"] - 1200, e["t_after"] + 1200
        counts = {t: sum(int(((p[a].Tag_ID == t) & p[a].t0.between(lo, hi)).sum()) for a in ARMS) for t in tags}
        tag = max(counts, key=counts.get)
        if plotted >= args.max_steps or counts[tag] == 0:
            continue
        label = "step_%s" % hhmm(e["t_before"]).strftime("%Y%m%d_%H%M")
        win = (e["t_before"], e["t_after"])
        plan_figure(p, rec, tag, lo, hi, win, "clock step", out / ("%s_%s_plan.png" % (tag, label)))
        event_figure(p, knots, tag, e["big"][1], lo, hi, win, out / ("%s_%s_event.png" % (tag, label)),
                     "%s: clock step on %d receiver(s) (%s); largest %s %.3f s"
                     % (tag, len(e["recs"]), ", ".join(sorted(e["recs"])), e["big"][1], e["big"][0]))
        rows.append(window_row("clock step", label, tag, *win))
        plotted += 1
    pd.DataFrame(rows).to_csv(out / "window_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
