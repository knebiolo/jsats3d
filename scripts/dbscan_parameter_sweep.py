"""DBSCAN parameter sweep on a pairwise beacon series (diagnostic; rejects nothing).

Input is the *_epochs.csv written by beacon_pairwise_dbscan.py. Every setting is scored on the
same receivers and window so ONE study-wide choice can be made (System Prompt 7.4):
  noise %            share of judged epochs rejected (the cost)
  LOO p50/p95 (us)   each interior clean epoch predicted from its clean neighbours (clock-sync quality)
  recall %           synthetic late arrivals caught (multipath-rejection quality, non-circular)
Before/after DDoA plots come from beacon_pairwise_dbscan.py itself.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beacon_pairwise_dbscan import MIN_SAMPLES, TIME_WINDOW_PERIODS, TIMING_BUDGET_S, classify

TOLERANCES_MS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
WINDOWS = [1.5, 2.0, 2.5, 3.0, 5.0]
MIN_SAMPLES_GRID = [2, 3, 4, 5]
INJECT_FRACTION = 0.05
# Later-arrival delays observed at beacons (2026-09-24 audit): p05 1.47 ms, p99 126 ms.
INJECT_DELAY_RANGE_S = (0.0015, 0.126)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("epochs_csv")
    p.add_argument("--period", type=float, required=True, help="Beacon nominal period used by the DBSCAN run (s)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--receivers", nargs="+", default=["ZOI08", "CFD04"],
                   help="Receivers drawn in the how-it-works figure")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def inject(series, rng):
    s = series.copy()
    pick = rng.random(len(s)) < INJECT_FRACTION
    lo, hi = np.log(INJECT_DELAY_RANGE_S)
    s.loc[pick, "delta_s"] = s.loc[pick, "delta_s"] + np.exp(rng.uniform(lo, hi, int(pick.sum())))
    s["injected"] = pick
    return s


def loo_errors(res):
    c = res[res.dbscan_class == "clean"].sort_values(["Rec_ID", "label", "t_anchor"])
    g = c.groupby(["Rec_ID", "label"])
    t0, d0, t1, d1 = g.t_anchor.shift(1), g.delta_s.shift(1), g.t_anchor.shift(-1), g.delta_s.shift(-1)
    pred = d0 + (d1 - d0) * (c.t_anchor - t0) / (t1 - t0)
    return pd.DataFrame({"Rec_ID": c.Rec_ID, "err": c.delta_s - pred}).dropna()


def score(res, inj):
    err = loo_errors(res)
    rows = []
    for rid, g in res.groupby("Rec_ID"):
        judged = g[g.dbscan_class != "anchor_suspect"]
        e = err.loc[err.Rec_ID == rid, "err"].abs()
        planted = inj[(inj.Rec_ID == rid) & inj.injected]
        rows.append(dict(Rec_ID=rid, group=rid[:3], epochs=len(g),
                         noise_pct=100 * (judged.dbscan_class != "clean").mean(),
                         segments=g.loc[g.dbscan_class == "clean", "label"].nunique(),
                         loo_p50_us=1e6 * e.median() if len(e) else np.nan,
                         loo_p95_us=1e6 * e.quantile(0.95) if len(e) else np.nan,
                         loo_over_budget_pct=100 * (e > TIMING_BUDGET_S).mean() if len(e) else np.nan,
                         recall_pct=100 * (planted.dbscan_class != "clean").mean() if len(planted) else np.nan))
    return pd.DataFrame(rows)


def run_grid(series, injected, period):
    combos = [(t, w, 3) for t, w in itertools.product(TOLERANCES_MS, WINDOWS)]
    combos += [(t, TIME_WINDOW_PERIODS, m) for t, m in itertools.product(TOLERANCES_MS, MIN_SAMPLES_GRID) if m != 3]
    frames = []
    for i, (tol, win, ms) in enumerate(combos, 1):
        params = dict(tolerance_s=tol / 1e3, window_periods=win, min_samples=ms)
        base, _ = classify(series, period, **params)
        inj, _ = classify(injected, period, **params)
        frames.append(score(base, inj).assign(tolerance_ms=tol, window_periods=win, min_samples=ms))
        print("  [%d/%d] tol %.2f ms, window %.1f, min_samples %d" % (i, len(combos), tol, win, ms), flush=True)
    return pd.concat(frames, ignore_index=True)


def plot_heatmaps(summary, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    sub = summary[summary.min_samples == 3]
    metrics = [("noise_pct", "Rejected %", "Reds"), ("loo_p95_us", "Clock LOO error p95 (us)", "Blues"),
               ("recall_pct", "Planted echoes caught %", "Greens")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for r, grp in enumerate(["ZOI", "CFD"]):
        for c, (col, title, cmap) in enumerate(metrics):
            piv = sub[sub.group == grp].pivot(index="window_periods", columns="tolerance_ms", values=col)
            ax = axes[r, c]
            ax.imshow(piv.values, origin="lower", aspect="auto", cmap=cmap)
            for (i, j), v in np.ndenumerate(piv.values):
                ax.text(j, i, "%.0f" % v, ha="center", va="center", fontsize=8)
            ax.set_xticks(range(piv.shape[1]), ["%g" % v for v in piv.columns])
            ax.set_yticks(range(piv.shape[0]), ["%g" % v for v in piv.index])
            j, i = list(piv.columns).index(TIMING_BUDGET_S * 1e3), list(piv.index).index(TIME_WINDOW_PERIODS)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=2.5))
            ax.set_title("%s receivers (median): %s" % (grp, title), fontsize=9)
            ax.set_xlabel("Tolerance (ms)")
            ax.set_ylabel("Time window (pings)")
    fig.suptitle("DBSCAN sweep, min_samples = 3. Black box = current setting (0.5 ms, 2.5 pings)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_min_samples(summary, path):
    import matplotlib.pyplot as plt
    sub = summary[summary.window_periods == TIME_WINDOW_PERIODS]
    metrics = [("noise_pct", "Rejected %"), ("loo_p95_us", "Clock LOO error p95 (us)"), ("recall_pct", "Planted echoes caught %")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for r, grp in enumerate(["ZOI", "CFD"]):
        for c, (col, title) in enumerate(metrics):
            ax = axes[r, c]
            for ms, g in sub[sub.group == grp].groupby("min_samples"):
                ax.plot(g.tolerance_ms, g[col], "o-", lw=2.5 if ms == MIN_SAMPLES else 1, label="min_samples %d" % ms)
            ax.axvline(TIMING_BUDGET_S * 1e3, color="0.6", ls=":")
            ax.set_xscale("log")
            ax.set_xlabel("Tolerance (ms)")
            ax.set_title("%s: %s" % (grp, title), fontsize=9)
            if col == "loo_p95_us":
                ax.set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Effect of min_samples (window fixed at %.1f pings). Dotted line = current tolerance" % TIME_WINDOW_PERIODS, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_how_it_works(series, period, receivers, path):
    """Draws each point's neighbourhood box: +/- window in time, +/- tolerance in delta."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    tols = [0.1, TIMING_BUDGET_S * 1e3, 2.0]
    colours = {"clean": "tab:blue", "noise": "k", "steady_reflection": "tab:red", "anchor_suspect": "0.6"}
    fig, axes = plt.subplots(len(receivers), len(tols), figsize=(15, 3.2 * len(receivers)), squeeze=False)
    ref, _ = classify(series, period)
    for r, rid in enumerate(receivers):
        g = ref[(ref.Rec_ID == rid) & (ref.dbscan_class == "clean")]
        if g.empty:
            continue
        longest = g.label.value_counts().idxmax()
        seg = g[g.label == longest]
        t_mid = seg.t_anchor.median()
        lo, hi = t_mid - 1.5 * 3600, t_mid + 1.5 * 3600
        for c, tol in enumerate(tols):
            res, _ = classify(series, period, tolerance_s=tol / 1e3)
            s = res[(res.Rec_ID == rid) & res.t_anchor.between(lo, hi)]
            y0 = s.loc[s.dbscan_class == "clean", "delta_s"].median()
            if np.isnan(y0):
                y0 = s.delta_s.median()
            x, y = (s.t_anchor - lo) / 60, (s.delta_s - y0) * 1e3
            ax = axes[r, c]
            for cls, col in colours.items():
                m = s.dbscan_class == cls
                ax.scatter(x[m], y[m], s=10, c=col, label=cls)
            k = s.iloc[len(s) // 2]
            ax.add_patch(Rectangle(((k.t_anchor - lo) / 60 - TIME_WINDOW_PERIODS * period / 60, (k.delta_s - y0) * 1e3 - tol),
                                   2 * TIME_WINDOW_PERIODS * period / 60, 2 * tol, fill=False, ec="tab:orange", lw=1.5))
            ax.set_ylim(-max(3, 4 * tol), max(10, 8 * tol))
            ax.set_title("%s, tolerance %.2f ms: %.0f%% rejected" % (rid, tol, 100 * (s.dbscan_class != "clean").mean()), fontsize=9)
            ax.set_xlabel("Minutes into 3 h slice")
            ax.set_ylabel("delta - median (ms)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("How DBSCAN decides: orange box = one point's neighbourhood (+/-%.1f pings x +/-tolerance). "
                 "A point needs >= %d points (itself included) inside its box to seed a cluster." % (TIME_WINDOW_PERIODS, MIN_SAMPLES), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    series = pd.read_csv(args.epochs_csv, usecols=["Rec_ID", "t_anchor", "delta_s"]).reset_index(drop=True)
    missing = sorted(set(args.receivers) - set(series.Rec_ID))
    if missing:
        raise ValueError("Receivers not in epochs file: %s" % missing)
    injected = inject(series, np.random.default_rng(args.seed))
    print("%d epochs, %d receivers; planted %d synthetic late arrivals (%.0f%%, %.1f-%.0f ms, seed %d)"
          % (len(series), series.Rec_ID.nunique(), int(injected.injected.sum()), 100 * INJECT_FRACTION,
             INJECT_DELAY_RANGE_S[0] * 1e3, INJECT_DELAY_RANGE_S[1] * 1e3, args.seed))
    results = run_grid(series, injected, args.period)
    results.to_csv(os.path.join(args.output_dir, "sweep_results.csv"), index=False, float_format="%.4f")
    keys = ["tolerance_ms", "window_periods", "min_samples", "group"]
    metrics = ["noise_pct", "segments", "loo_p50_us", "loo_p95_us", "loo_over_budget_pct", "recall_pct"]
    summary = results.groupby(keys)[metrics].median().reset_index()
    summary.to_csv(os.path.join(args.output_dir, "sweep_summary.csv"), index=False, float_format="%.3f")
    plot_heatmaps(summary, os.path.join(args.output_dir, "sweep_heatmaps.png"))
    plot_min_samples(summary, os.path.join(args.output_dir, "sweep_min_samples.png"))
    plot_how_it_works(series, args.period, args.receivers, os.path.join(args.output_dir, "how_dbscan_works.png"))
    cur = summary[(summary.tolerance_ms == TIMING_BUDGET_S * 1e3) & (summary.window_periods == TIME_WINDOW_PERIODS)
                  & (summary.min_samples == MIN_SAMPLES)]
    print("Current setting (group medians):\n" + cur.round(2).to_string(index=False))
    print("No detections modified. Outputs in %s" % args.output_dir)


if __name__ == "__main__":
    main()
