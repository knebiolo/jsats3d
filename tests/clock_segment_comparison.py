"""Segmented-clock validation: baseline (five-pass clock filter) vs segmented clock with midpoint holds.
Test tool, not a pipeline step (not collected by pytest). Both databases must be copies of one base build.

    python tests/clock_segment_comparison.py --baseline DB --segmented DB --out DIR

Figures
    <tag>_<yyyymmdd_hhmm>_plan.png   plan view, baseline vs midpoint holds (per-transmission median X/Y of root-B
                                     solutions, before position DBSCAN / depth filtering / smoothing)
    <tag>_<yyyymmdd_hhmm>_event.png  coverage (root B, any solution), median ping depth, receiver clock residual
CSVs
    gap_recovery.csv        transmissions positioned in each event window, per arm, after DBSCAN, plausibility
    jump_events.csv         held clock steps (tblClockSegments) and positioned transmissions around each
    normal_periods.csv      per tag outside every jump window: yield, lost transmissions, position differences
    receiver_timing.csv     per receiver: |correction difference| between arms outside jump windows, unsupported share
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jsats3d.jsats3d import _interpolate_clock_segments  # noqa: E402

FLOOR_M = -16.0
EVENT_PAD_S = 600.0          # window around a held step treated as jump-affected
ARMS = ("baseline", "segmented")
LABEL = {"baseline": "Baseline", "segmented": "Midpoint holds"}
COLOR = {"baseline": "tab:orange", "segmented": "tab:green"}


def ts(text):
    return pd.Timestamp(text).timestamp()


def hhmm(t):
    return pd.to_datetime(t, unit="s")


def connect(db):
    return sqlite3.connect("file:%s?mode=ro" % str(db).replace("\\", "/"), uri=True)


def pings(db, table="tblPositions_Deng"):
    """Per transmission: root-B solutions found, median X/Y/Z/t0 and the clock flag (if present)."""
    con = connect(db)
    cols = {r[1] for r in con.execute("pragma table_info(%s)" % table)}
    if not cols:
        con.close()
        return pd.DataFrame()
    flag = ", clock_low_support" if "clock_low_support" in cols else ""
    p = pd.read_sql("select Tag_ID, transNo, X, Y, Z, ToA, T01%s from %s where solution = 'B' "
                    "and comment = 'solution found'" % (flag, table), con)
    con.close()
    for c in ("transNo", "X", "Y", "Z", "ToA", "T01"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["t0"] = p.ToA - p.T01
    p = p[np.isfinite(p[["X", "Y", "Z", "t0"]]).all(axis=1)]
    agg = dict(x=("X", "median"), y=("Y", "median"), z=("Z", "median"), t0=("t0", "median"), n=("X", "size"))
    if flag:
        p["clock_low_support"] = pd.to_numeric(p.clock_low_support, errors="coerce").fillna(0)
        agg["flagged"] = ("clock_low_support", "max")
    return p.groupby(["Tag_ID", "transNo"]).agg(**agg).reset_index().sort_values("t0")


def clock_rows(db):
    con = connect(db)
    cols = {r[1] for r in con.execute("pragma table_info(tblDetectionClockFixed)")}
    extra = "".join(", " + c for c in ("clock_segment", "clock_segment_start", "clock_segment_end",
                                       "clock_segment_support") if c in cols)
    k = pd.read_sql("select Rec_ID, seconds, seconds_residual%s from tblDetectionClockFixed" % extra, con)
    con.close()
    return k


def correction(knots, rec, t):
    """Fish clock correction of a receiver at times t, exactly as multipath_data_object applies it."""
    k = knots[knots.Rec_ID == rec].dropna(subset=["seconds", "seconds_residual"])
    if k.empty:
        return np.full(len(t), np.nan)
    if "clock_segment" in k.columns:
        return _interpolate_clock_segments(k, np.asarray(t, float))
    k = k.sort_values("seconds").drop_duplicates("seconds")
    return np.interp(t, k.seconds, k.seconds_residual, left=k.seconds_residual.iloc[0], right=k.seconds_residual.iloc[-1])


def receivers(db):
    con = connect(db)
    r = pd.read_sql("select Rec_ID, X_t, Y_t from tblReceiver", con).set_index("Rec_ID")
    con.close()
    return r


def plan_figure(p, rec, tag, t0, t1, gap, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw={"height_ratios": [1.5, 1]})
    xl = (rec.X_t.min() - 25, rec.X_t.max() + 25)
    yl = (rec.Y_t.min() - 25, rec.Y_t.max() + 25)
    sub = {a: p[a][(p[a].Tag_ID == tag) & (p[a].t0 >= t0) & (p[a].t0 <= t1)] for a in ARMS}
    both = pd.concat(sub.values())
    full = ((both.x.min() - 10, both.x.max() + 10), (both.y.min() - 10, both.y.max() + 10)) if len(both) else (xl, yl)
    sc = None
    for col, a in enumerate(ARMS):
        d = sub[a]
        ingap = (d.t0 >= gap[0]) & (d.t0 <= gap[1])
        for row, (lx, ly) in enumerate(((xl, yl), full)):
            ax = axes[row, col]
            sc = ax.scatter(d.x, d.y, c=d.t0, cmap="viridis", vmin=t0, vmax=t1, s=14, alpha=0.75, zorder=3)
            ax.scatter(d.x[ingap], d.y[ingap], s=60, facecolors="none", edgecolors="red", linewidths=1.2, zorder=4)
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
                ax.set_title("%s - array view\n%d pings; %d in original gap; %d outside this view"
                             % (LABEL[a], len(d), int(ingap.sum()), outside))
            else:
                ax.set_title("%s - full range (all plotted pings)" % LABEL[a])
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    ticks = np.linspace(t0, t1, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([hhmm(t).strftime("%H:%M") for t in ticks])
    cb.set_label("%s study wall time" % hhmm(t0).strftime("%Y-%m-%d"))
    fig.suptitle("%s plan view: baseline vs midpoint holds | %s, %s-%s\nMedian X/Y of successful root-B solutions per "
                 "transmission; before position DBSCAN, depth filtering or smoothing\nRed rings: original gap %s-%s. "
                 "Identical axes per row; zoom limits are not data filtering."
                 % (tag, hhmm(t0).strftime("%B %d"), hhmm(t0).strftime("%H:%M"), hhmm(t1).strftime("%H:%M"),
                    hhmm(gap[0]).strftime("%H:%M:%S"), hhmm(gap[1]).strftime("%H:%M:%S")), fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def event_figure(p, knots, tag, rec_id, t0, t1, gap, path, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})
    for i, a in enumerate(ARMS):
        d = p[a][(p[a].Tag_ID == tag) & (p[a].t0 >= t0) & (p[a].t0 <= t1)]
        axes[0].plot(hhmm(d.t0), np.full(len(d), len(ARMS) - 1 - i), "o", ms=3, color=COLOR[a], label=LABEL[a])
        axes[1].plot(hhmm(d.t0), d.z, "o", ms=3, alpha=0.6, color=COLOR[a], label=LABEL[a])
        if "flagged" in d.columns:
            f = d[d.flagged > 0]
            axes[1].plot(hhmm(f.t0), f.z, "o", ms=7, mfc="none", mec="k", mew=0.7,
                         label="low clock support (< 3 beacons)" if len(f) else None)
        k = knots[a][(knots[a].Rec_ID == rec_id) & (knots[a].seconds >= t0) & (knots[a].seconds <= t1)]
        grid = np.arange(t0, t1, 5.0)
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
        ax.axvspan(hhmm(gap[0]), hhmm(gap[1]), color="red", alpha=0.12)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title + "\nJump times are estimated; increased coverage is not an accuracy validation", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def window_stats(p, pd_dbscan, tag, lo, hi):
    row = {}
    for a in ARMS:
        d = p[a][(p[a].Tag_ID == tag) & (p[a].t0 >= lo) & (p[a].t0 <= hi)]
        k = pd_dbscan[a]
        kd = k[(k.Tag_ID == tag) & (k.t0 >= lo) & (k.t0 <= hi)] if len(k) else k
        wet = d.z.between(FLOOR_M, 0)
        dt, dist = np.diff(d.t0), np.hypot(np.diff(d.x), np.diff(d.y))
        fast = (dist / np.where(dt > 0, dt, np.nan) > 3.0) & (dt <= 30)
        row.update({"%s_positioned" % a: int(len(d)), "%s_after_dbscan" % a: int(len(kd)),
                    "%s_in_water_pct" % a: round(100 * wet.mean(), 1) if len(d) else np.nan,
                    "%s_jumps_gt_3ms" % a: int(np.nansum(fast)),
                    "%s_flagged" % a: int(d.flagged.sum()) if "flagged" in d.columns else 0})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--segmented", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="26E4")
    ap.add_argument("--receiver", default="ZOI02")
    ap.add_argument("--start", default="2025-06-28 00:40:00")
    ap.add_argument("--end", default="2025-06-28 01:20:00")
    ap.add_argument("--gap", nargs=2, default=["2025-06-28 00:54:59", "2025-06-28 01:05:47"])
    ap.add_argument("--events", type=int, default=4, help="additional jump events to plot")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dbs = {"baseline": Path(args.baseline), "segmented": Path(args.segmented)}
    p = {a: pings(db) for a, db in dbs.items()}
    pdb = {a: pings(db, "tblPositions_DengDBSCAN") for a, db in dbs.items()}
    knots = {a: clock_rows(db) for a, db in dbs.items()}
    rec = receivers(dbs["baseline"])
    con = connect(dbs["segmented"])
    seg = pd.read_sql("select * from tblClockSegments", con)
    used = set(pd.read_sql("select r0 r from tblPositions_Deng union select r1 from tblPositions_Deng union "
                           "select r2 from tblPositions_Deng union select r3 from tblPositions_Deng", con).r.dropna())
    con.close()
    rec3d = rec.loc[[r for r in rec.index if r in used]]

    # ---- 26E4 known gap
    t0, t1, gap = ts(args.start), ts(args.end), (ts(args.gap[0]), ts(args.gap[1]))
    stem = "%s_%s" % (args.tag, hhmm(t0).strftime("%Y%m%d_%H%M"))
    plan_figure(p, rec3d, args.tag, t0, t1, gap, out / ("%s_plan.png" % stem))
    event_figure(p, knots, args.tag, args.receiver, t0, t1, gap, out / ("%s_event.png" % stem),
                 "%s known gap: midpoint holds at jumps; linear interpolation within stable segments" % args.tag)
    rows = [dict(event="known gap", tag=args.tag, receiver=args.receiver, start=args.gap[0], end=args.gap[1],
                 **window_stats(p, pdb, args.tag, *gap))]

    # ---- additional held jumps on positioning receivers, with fish present
    steps = seg[(seg.boundary_before == "step") & seg.Rec_ID.isin(rec3d.index)].copy()
    steps["jump_t"] = steps.clock_segment_start
    ev = []
    for s in steps.itertuples():
        lo, hi = s.jump_t - EVENT_PAD_S, s.jump_t + EVENT_PAD_S
        for tag in sorted(set(p["baseline"].Tag_ID) | set(p["segmented"].Tag_ID)):
            st = window_stats(p, pdb, tag, lo, hi)
            if st["baseline_positioned"] + st["segmented_positioned"] == 0:
                continue
            ev.append(dict(Rec_ID=s.Rec_ID, jump_time=hhmm(s.jump_t), step_s=round(float(s.step_before_s), 3),
                           gap_s=round(float(s.gap_before_s), 1), support=int(s.support), method=s.method,
                           internal_event=getattr(s, "internal_event", None),
                           cross_receiver_steps=getattr(s, "cross_receiver_steps", None), tag=tag, **st))
    ev = pd.DataFrame(ev)
    if len(ev):
        ev["recovered"] = ev.segmented_positioned - ev.baseline_positioned
        ev = ev.sort_values("recovered", ascending=False)
    ev.to_csv(out / "jump_events.csv", index=False)
    known = (gap[0] - EVENT_PAD_S, gap[1] + EVENT_PAD_S)
    plotted, shown = 0, []
    for e in ev.itertuples() if len(ev) else []:
        jt = e.jump_time.timestamp()
        # one figure per event: a master-clock step shows on every receiver at about the same time
        if plotted >= args.events or (known[0] <= jt <= known[1] and e.tag == args.tag) or \
                any(abs(jt - t) < 2 * EVENT_PAD_S for t in shown):
            continue
        shown.append(jt)
        lo, hi = jt - 1200, jt + 1200
        event_figure(p, knots, e.tag, e.Rec_ID, lo, hi, (jt - e.gap_s / 2, jt + e.gap_s / 2),
                     out / ("%s_%s_%s_event.png" % (e.tag, e.jump_time.strftime("%Y%m%d_%H%M"), e.Rec_ID)),
                     "%s, %s step %+.3f s at %s (estimated)" % (e.tag, e.Rec_ID, e.step_s, e.jump_time))
        rows.append(dict(event="jump", tag=e.tag, receiver=e.Rec_ID, start=str(hhmm(jt - EVENT_PAD_S)),
                         end=str(hhmm(jt + EVENT_PAD_S)), **window_stats(p, pdb, e.tag, jt - EVENT_PAD_S, jt + EVENT_PAD_S)))
        plotted += 1
    pd.DataFrame(rows).to_csv(out / "gap_recovery.csv", index=False)

    # ---- normal periods: everything outside +/- EVENT_PAD_S of any held or unbridged step on a 3D receiver
    all_steps = seg[seg.boundary_before.isin(["step", "step_outage"]) & seg.Rec_ID.isin(rec3d.index)]
    windows = np.column_stack([all_steps.first_obs - all_steps.gap_before_s - EVENT_PAD_S,
                               all_steps.first_obs + EVENT_PAD_S]) if len(all_steps) else np.empty((0, 2))

    def normal(t):
        t = np.asarray(t, float)
        bad = np.zeros(len(t), bool)
        for lo, hi in windows:
            bad |= (t >= lo) & (t <= hi)
        return ~bad

    norm_rows = []
    for tag in sorted(set(p["baseline"].Tag_ID)):
        b = p["baseline"][(p["baseline"].Tag_ID == tag)]
        s = p["segmented"][(p["segmented"].Tag_ID == tag)]
        b, s = b[normal(b.t0)], s[normal(s.t0)]
        # transNo is enumerated per arm from each arm's corrected first detection; match pings by time instead
        m = pd.merge_asof(b.sort_values("t0"), s.sort_values("t0"), on="t0", direction="nearest", tolerance=1.0,
                          suffixes=("_b", "_s")).dropna(subset=["x_s"])
        d = np.hypot(m.x_b - m.x_s, m.y_b - m.y_s)
        norm_rows.append(dict(tag=tag, baseline_positioned=len(b), segmented_positioned=len(s), matched=len(m),
                              baseline_only=len(b) - len(m), segmented_only=len(s) - len(m),
                              median_xy_diff_m=round(float(d.median()), 3) if len(d) else np.nan,
                              p95_xy_diff_m=round(float(d.quantile(0.95)), 3) if len(d) else np.nan,
                              share_diff_over_1m=round(float((d > 1).mean()), 4) if len(d) else np.nan,
                              median_z_diff_m=round(float((m.z_b - m.z_s).abs().median()), 3) if len(d) else np.nan,
                              segmented_flagged=int(s.flagged.sum()) if "flagged" in s.columns else 0))
    pd.DataFrame(norm_rows).to_csv(out / "normal_periods.csv", index=False)

    # ---- receiver timing consistency
    lo = min(k.seconds.min() for k in knots.values())
    hi = max(k.seconds.max() for k in knots.values())
    grid = np.arange(lo, hi, 60.0)
    keep = normal(grid)
    tim = []
    for r in sorted(set(knots["baseline"].Rec_ID)):
        cb, cs = correction(knots["baseline"], r, grid), correction(knots["segmented"], r, grid)
        diff = np.abs(cb - cs)[keep & np.isfinite(cs)] * 1e3
        tim.append(dict(Rec_ID=r, normal_minutes=int(keep.sum()),
                        segmented_unsupported_pct=round(100 * float((~np.isfinite(cs[keep])).mean()), 2),
                        median_abs_diff_ms=round(float(np.median(diff)), 4) if len(diff) else np.nan,
                        p99_abs_diff_ms=round(float(np.quantile(diff, 0.99)), 3) if len(diff) else np.nan,
                        max_abs_diff_ms=round(float(diff.max()), 3) if len(diff) else np.nan,
                        segments=int(((seg.Rec_ID == r) & (seg.method != "isolated_multipath")).sum()),
                        held_steps=int(((seg.Rec_ID == r) & (seg.boundary_before == "step")).sum()),
                        fallback=int(((seg.Rec_ID == r) & seg.method.isin(["linear", "constant"])).sum()),
                        unsupported_segments=int(((seg.Rec_ID == r) & (seg.method == "none")).sum()),
                        isolated_multipath=int(((seg.Rec_ID == r) & (seg.method == "isolated_multipath")).sum())))
    pd.DataFrame(tim).to_csv(out / "receiver_timing.csv", index=False)
    pd.set_option("display.width", 250)
    for name in ("gap_recovery", "normal_periods", "receiver_timing"):
        print("\n=== %s ===" % name)
        print(pd.read_csv(out / ("%s.csv" % name)).to_string(index=False))
    print("\n=== jump_events (top 15 by recovered) ===")
    print(ev.head(15).to_string(index=False) if len(ev) else "none")


if __name__ == "__main__":
    main()
