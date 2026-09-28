"""CFD float GPS diagnostics: 15-minute averaging and GPS quality through time (read-only).

Why: receiver position error enters TDoA directly (1 m ~ 0.68 ms at 1465 m/s), and single 60 s
fixes scatter by metres with occasional tens-of-metres spikes. Each 15-minute bin is summarised
by mean and median; the median resists spikes, the mean is what was proposed. Both are written.
Timestamps are PDT (data catalog). Antenna-to-hydrophone offset is undocumented and not applied.
"""
import argparse
import os

import numpy as np
import pandas as pd

BIN = "15min"
SPIKE_THRESHOLDS_M = [10, 25, 50]
MAP_HALF_WIDTH_M = 15
TRACK_YMAX_M = 20


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gps_csv", help="master_df_gps.csv (read-only)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--map-day", default="2025-06-20", help="Day (PDT) drawn in the raw-vs-average map")
    return p.parse_args()


def load(path):
    cols = ["dateTime", "receiverName", "easting", "northing"]
    g = pd.concat(pd.read_csv(path, usecols=cols, chunksize=500_000, parse_dates=["dateTime"]), ignore_index=True)
    bad = g[cols].isna().any(axis=1)
    if bad.any():
        print("WARNING: %d GPS rows with missing fields dropped" % int(bad.sum()))
    return g[~bad].rename(columns={"receiverName": "Rec_ID"}).sort_values(["Rec_ID", "dateTime"])


def bin_positions(g):
    g = g.assign(bin=g.dateTime.dt.floor(BIN))
    b = g.groupby(["Rec_ID", "bin"]).agg(n=("easting", "size"), E_mean=("easting", "mean"), N_mean=("northing", "mean"),
                                          E_median=("easting", "median"), N_median=("northing", "median")).reset_index()
    g = g.merge(b[["Rec_ID", "bin", "E_median", "N_median"]], on=["Rec_ID", "bin"])
    g["dev_m"] = np.hypot(g.easting - g.E_median, g.northing - g.N_median)
    spread = g.groupby(["Rec_ID", "bin"]).dev_m.agg(sd_m=lambda d: float(np.sqrt(np.mean(d ** 2))), max_dev_m="max")
    b = b.merge(spread.reset_index(), on=["Rec_ID", "bin"])
    b["mean_minus_median_m"] = np.hypot(b.E_mean - b.E_median, b.N_mean - b.N_median)
    ref = g.groupby("Rec_ID")[["easting", "northing"]].median()
    for k in ("mean", "median"):
        b["dist_%s_from_season_m" % k] = np.hypot(b["E_%s" % k] - b.Rec_ID.map(ref.easting), b["N_%s" % k] - b.Rec_ID.map(ref.northing))
    b["step_median_m"] = b.groupby("Rec_ID")[["E_median", "N_median"]].diff().pipe(lambda d: np.hypot(d.E_median, d.N_median))
    g["dist_from_season_m"] = np.hypot(g.easting - g.Rec_ID.map(ref.easting), g.northing - g.Rec_ID.map(ref.northing))
    return g, b


def summarize(g, b):
    rows = []
    for rid, d in g.groupby("Rec_ID"):
        bb = b[b.Rec_ID == rid]
        row = dict(Rec_ID=rid, fixes=len(d), start=d.dateTime.min(), end=d.dateTime.max(), bins=len(bb),
                   fix_dev_p50_m=d.dev_m.median(), fix_dev_p95_m=d.dev_m.quantile(0.95),
                   bin_sd_p50_m=bb.sd_m.median(), bin_sd_p95_m=bb.sd_m.quantile(0.95),
                   mean_minus_median_p95_m=bb.mean_minus_median_m.quantile(0.95),
                   bin_step_p50_m=bb.step_median_m.median(), bin_step_p95_m=bb.step_median_m.quantile(0.95),
                   bin_steps_over_10m=int((bb.step_median_m > 10).sum()))
        for t in SPIKE_THRESHOLDS_M:
            row["fixes_over_%dm_pct" % t] = 100 * (d.dev_m > t).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_map(g, b, path, day):
    """One day of raw 1-min fixes vs 15-min medians, in metres around each float's usual (season-median) spot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = sorted(g.Rec_ID.unique())
    ref = g.groupby("Rec_ID")[["easting", "northing"]].median()
    start, end = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
    g = g[(g.dateTime >= start) & (g.dateTime < end)]
    b = b[(b.bin >= start) & (b.bin < end)]
    fig, axes = plt.subplots(2, (len(recs) + 1) // 2, figsize=(16, 8), squeeze=False)
    for ax, rid in zip(axes.flat, recs):
        d, bb = g[g.Rec_ID == rid], b[b.Rec_ID == rid]
        e0, n0 = ref.loc[rid, "easting"], ref.loc[rid, "northing"]
        ax.scatter(d.easting - e0, d.northing - n0, s=3, c="0.7", label="Raw GPS (every minute)")
        ax.plot(bb.E_median - e0, bb.N_median - n0, "o", ms=4, c="tab:blue", label="15-minute average")
        ax.set_xlim(-MAP_HALF_WIDTH_M, MAP_HALF_WIDTH_M)
        ax.set_ylim(-MAP_HALF_WIDTH_M, MAP_HALF_WIDTH_M)
        ax.set_aspect("equal")
        ax.set_title(rid if len(d) else "%s (no data)" % rid)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
    for ax in list(axes.flat)[len(recs):]:
        ax.axis("off")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Float GPS on %s: raw vs 15-minute average" % day)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_track(b, path):
    """Distance of each 15-min average from the float's usual spot; big moves pinned at the top in red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = sorted(b.Rec_ID.unique())
    fig, axes = plt.subplots(len(recs), 1, figsize=(13, 1.6 * len(recs)), sharex=True, squeeze=False)
    for ax, rid in zip(axes[:, 0], recs):
        # Reindex to a full 15-min grid so data gaps show as breaks, not straight lines.
        bb = b[b.Rec_ID == rid].set_index("bin").dist_median_from_season_m
        bb = bb.reindex(pd.date_range(bb.index.min(), bb.index.max(), freq=BIN))
        big = bb > TRACK_YMAX_M
        ax.plot(bb.index, bb.clip(upper=TRACK_YMAX_M), lw=0.7, c="tab:blue")
        ax.scatter(bb.index[big], np.full(big.sum(), TRACK_YMAX_M), c="tab:red", s=8, zorder=3)
        ax.set_ylim(0, TRACK_YMAX_M + 2)
        ax.set_ylabel(rid, rotation=0, labelpad=20)
    fig.suptitle("How far each float is from its usual spot (15-minute average, metres)\n"
                 "Red = moved more than %d m" % TRACK_YMAX_M)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    g = load(args.gps_csv)
    print("%d fixes, receivers %s" % (len(g), sorted(g.Rec_ID.unique())))
    g, b = bin_positions(g)
    b.to_csv(os.path.join(args.output_dir, "cfd_gps_15min.csv"), index=False, float_format="%.3f")
    s = summarize(g, b)
    s.to_csv(os.path.join(args.output_dir, "cfd_gps_summary.csv"), index=False, float_format="%.3f")
    plot_map(g, b, os.path.join(args.output_dir, "cfd_gps_raw_vs_15min.png"), args.map_day)
    plot_track(b, os.path.join(args.output_dir, "cfd_gps_movement.png"))
    print(s.round({c: 2 for c in s.select_dtypes("number").columns}).to_string(index=False))
    print("Raw GPS not modified. Outputs in %s" % args.output_dir)


if __name__ == "__main__":
    main()
