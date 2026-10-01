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
from scipy.interpolate import interp1d

BIN = "15min"
SPIKE_THRESHOLDS_M = [10, 25, 50]
MAP_HALF_WIDTH_M = 15
TRACK_YMAX_M = 20
DEFAULT_RECEIVERS = ["CFD04", "ZOI05", "ZOI06", "ZOI07", "ZOI08", "ZOI09"]
GPS_COLUMNS = ("Rec_ID", "dateTime", "easting", "northing")


def _validate_interpolation_inputs(gps, query):
    missing_gps = set(GPS_COLUMNS) - set(gps.columns)
    missing_query = {"Rec_ID", "dateTime"} - set(query.columns)
    if missing_gps:
        raise ValueError("GPS data missing columns: %s" % sorted(missing_gps))
    if missing_query:
        raise ValueError("Query data missing columns: %s" % sorted(missing_query))


def _timestamp_seconds(values):
    timestamps = pd.to_datetime(values)
    return timestamps.astype("int64").to_numpy(dtype=float) / 1.0e9


def _receiver_interpolators(gps, method):
    if method not in ("linear", "cubic"):
        raise ValueError("method must be 'linear' or 'cubic', got %r" % method)
    data = gps.sort_values("dateTime").drop_duplicates("dateTime", keep="last")
    if len(data) < 2:
        raise ValueError("at least two GPS fixes are required")
    if method == "cubic" and len(data) < 4:
        raise ValueError("cubic interpolation requires at least four GPS fixes")
    seconds = _timestamp_seconds(data.dateTime)
    east = interp1d(seconds, data.easting, kind=method, bounds_error=False, fill_value=np.nan)
    north = interp1d(seconds, data.northing, kind=method, bounds_error=False, fill_value=np.nan)
    return east, north, seconds[0], seconds[-1]


def interpolate_positions(gps, query, method="linear"):
    """Return GPS positions at query times without extrapolating beyond fixes."""
    _validate_interpolation_inputs(gps, query)
    result = query[["Rec_ID", "dateTime"]].copy()
    result["dateTime"] = pd.to_datetime(result["dateTime"])
    result["easting"], result["northing"] = np.nan, np.nan
    for rec_id, indices in result.groupby("Rec_ID").groups.items():
        receiver_gps = gps[gps.Rec_ID == rec_id]
        if receiver_gps.empty:
            continue
        east, north, start, end = _receiver_interpolators(receiver_gps, method)
        times = _timestamp_seconds(result.loc[indices, "dateTime"])
        in_range = (times >= start) & (times <= end)
        result.loc[indices, "easting"] = np.where(in_range, east(times), np.nan)
        result.loc[indices, "northing"] = np.where(in_range, north(times), np.nan)
    return result


def grid_queries(gps, frequency="15min"):
    """Build per-receiver query times over the observed GPS interval."""
    rows = []
    for rec_id, data in gps.groupby("Rec_ID"):
        for timestamp in pd.date_range(data.dateTime.min(), data.dateTime.max(), freq=frequency):
            rows.append((rec_id, timestamp))
    return pd.DataFrame(rows, columns=["Rec_ID", "dateTime"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gps_csv", help="master_df_gps.csv (read-only)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--map-day", default="2025-06-20", help="Day (PDT) drawn in the raw-vs-average map")
    p.add_argument("--receivers", nargs="+", default=DEFAULT_RECEIVERS,
                   help="Receivers included in plots and interpolation outputs")
    p.add_argument("--interpolation-methods", nargs="+", choices=("linear", "cubic"),
                   default=["linear", "cubic"])
    p.add_argument("--no-interactive", action="store_true", help="Skip the Plotly HTML plot")
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


def plot_map(g, b, path, day, receivers):
    """One day of raw 1-min fixes vs 15-min medians, in metres around each float's usual (season-median) spot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [rid for rid in receivers if rid in set(g.Rec_ID)]
    ref = g.groupby("Rec_ID")[["easting", "northing"]].median()
    start, end = pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)
    g = g[(g.dateTime >= start) & (g.dateTime < end)]
    b = b[(b.bin >= start) & (b.bin < end)]
    if not recs:
        raise ValueError("none of the requested receivers have GPS rows")
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


def plot_track(b, path, receivers):
    """Distance of each 15-min average from the float's usual spot; big moves pinned at the top in red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [rid for rid in receivers if rid in set(b.Rec_ID)]
    if not recs:
        raise ValueError("none of the requested receivers have binned GPS rows")
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


def plot_spatial_tracks(g, path, receivers):
    """Show whether each selected receiver follows a path or jitters spatially."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [rid for rid in receivers if rid in set(g.Rec_ID)]
    fig, axes = plt.subplots(2, (len(recs) + 1) // 2, figsize=(16, 8), squeeze=False)
    for ax, rid in zip(axes.flat, recs):
        data = g[g.Rec_ID == rid].sort_values("dateTime")
        elapsed = (data.dateTime - data.dateTime.min()).dt.total_seconds() / 3600
        points = ax.scatter(data.easting, data.northing, c=elapsed, s=5, cmap="viridis")
        ax.plot(data.easting, data.northing, color="0.6", lw=0.4, alpha=0.7)
        ax.scatter(data.easting.iloc[0], data.northing.iloc[0], c="tab:green", s=30, label="start")
        ax.scatter(data.easting.iloc[-1], data.northing.iloc[-1], c="tab:red", s=30, label="end")
        ax.set_title(rid)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        fig.colorbar(points, ax=ax, label="Hours since first fix")
    for ax in list(axes.flat)[len(recs):]:
        ax.axis("off")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("GPS movement through time: path versus jitter")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def interpolation_outputs(g, output_dir, receivers, methods):
    selected = g[g.Rec_ID.isin(receivers)]
    queries = grid_queries(selected, frequency=BIN)
    outputs = []
    for method in methods:
        positions = interpolate_positions(selected, queries, method=method)
        positions["method"] = method
        outputs.append(positions)
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(os.path.join(output_dir, "cfd_gps_piecewise_positions.csv"), index=False, float_format="%.3f")
    return result


def plot_interpolation_comparison(positions, path):
    """Compare piecewise methods on the same 15-minute query grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = sorted(positions.Rec_ID.unique())
    fig, axes = plt.subplots(2, (len(recs) + 1) // 2, figsize=(16, 8), squeeze=False)
    for ax, rid in zip(axes.flat, recs):
        data = positions[positions.Rec_ID == rid]
        for method, method_data in data.groupby("method"):
            ax.plot(method_data.easting, method_data.northing, lw=1, label=method)
        ax.set_title(rid)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
    for ax in list(axes.flat)[len(recs):]:
        ax.axis("off")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Piecewise GPS interpolation comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_interactive_tracks(b, path, receivers):
    """Write a compact, zoomable Plotly map from 15-minute GPS summaries."""
    import plotly.graph_objects as go
    figure = go.Figure()
    for rid in receivers:
        data = b[b.Rec_ID == rid].sort_values("bin")
        if data.empty:
            continue
        figure.add_trace(go.Scattergl(
            x=data.E_mean, y=data.N_mean, mode="lines+markers", name=rid,
            customdata=data[["bin", "sd_m", "mean_minus_median_m"]],
            hovertemplate=("%{fullData.name}<br>%{customdata[0]}<br>"
                           "E %{x:.2f} m, N %{y:.2f} m<br>"
                           "within-bin RMS %{customdata[1]:.2f} m<br>"
                           "mean-median %{customdata[2]:.2f} m<extra></extra>")))
    figure.update_layout(title="15-minute GPS movement: zoom and inspect time",
                         xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                         yaxis_scaleanchor="x", template="plotly_white")
    figure.write_html(path, include_plotlyjs=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    g = load(args.gps_csv)
    print("%d fixes, receivers %s" % (len(g), sorted(g.Rec_ID.unique())))
    g, b = bin_positions(g)
    selected = [rid for rid in args.receivers if rid in set(g.Rec_ID)]
    missing = sorted(set(args.receivers) - set(selected))
    if missing:
        print("WARNING: requested receivers without GPS rows: %s" % missing)
    b.to_csv(os.path.join(args.output_dir, "cfd_gps_15min.csv"), index=False, float_format="%.3f")
    s = summarize(g, b)
    s.to_csv(os.path.join(args.output_dir, "cfd_gps_summary.csv"), index=False, float_format="%.3f")
    plot_map(g, b, os.path.join(args.output_dir, "cfd_gps_raw_vs_15min.png"), args.map_day, selected)
    plot_track(b, os.path.join(args.output_dir, "cfd_gps_movement.png"), selected)
    plot_spatial_tracks(g[g.Rec_ID.isin(selected)], os.path.join(args.output_dir, "cfd_gps_spatial_tracks.png"), selected)
    positions = interpolation_outputs(g, args.output_dir, selected, args.interpolation_methods)
    plot_interpolation_comparison(positions, os.path.join(args.output_dir, "cfd_gps_interpolation_comparison.png"))
    if not args.no_interactive:
        plot_interactive_tracks(b, os.path.join(args.output_dir, "cfd_gps_interactive.html"), selected)
    print(s.round({c: 2 for c in s.select_dtypes("number").columns}).to_string(index=False))
    print("Raw GPS not modified. Outputs in %s" % args.output_dir)


if __name__ == "__main__":
    main()
