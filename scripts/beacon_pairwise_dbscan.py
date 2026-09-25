"""Pairwise beacon TDoA DBSCAN diagnostic (read-only on the source DB; rejects nothing).

For beacon B heard at receiver i and anchor a:
    delta_ia = (t_i - t_a) - (d_Bi - d_Ba) / c = (eps_i - eps_a) + (m_i - m_a)
The beacon host clock cancels. Clean epochs form piecewise-smooth clock segments;
multipath (m >= 0, observed >= ~1.5 ms) appears as isolated outliers or, when the
reflection is persistent, as a parallel cluster offset later in time.
Errors in the beacon host position enter d_Bi and appear as constant per-receiver
offsets, indistinguishable from clock bias without a surveyed host or a second beacon.
"""
import argparse
import datetime as dt
import os
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from jsats3d import sos  # legacy freshwater sound-speed table (m/s), cubic interpolation

# System Prompt 9.2 timing budget; also below shortest observed reflection delay (p05 1.47 ms, 2026-09-24 audit).
TIMING_BUDGET_S = 0.0005
# Neighbour window in nominal periods: one missed transmission when true period is up to 25% above
# nominal (7D2D measured 62.7 s vs 60 s nominal; 2.0 fragmented segments, see journal 2026-09-24).
TIME_WINDOW_PERIODS = 2.5
# Legacy clock_fix() value: a point needs neighbours on both sides to be core.
MIN_SAMPLES = 3
# Anchor-side epoch: a majority of receivers are simultaneously noise, and at least 4 (3-D solution minimum) report.
ANCHOR_MAJORITY = 0.5
ANCHOR_MIN_RECEIVERS = 4
# Upper bound on a reflection delay: observed later arrivals p99 126 ms (23 of ~285k > 200 ms); clock steps are >= ~390 ms.
MAX_REFLECTION_DELAY_S = 0.25
DD_N_COLUMNS = ["DD_N_0p5", "DD_N_1p5", "DD_N_9", "DD_N_18"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("database")
    p.add_argument("--beacon-receiver", required=True, help="Receiver hosting the beacon (e.g. ZOI02)")
    p.add_argument("--anchor", required=True, help="Clock anchor receiver (e.g. ZOI09)")
    p.add_argument("--start", required=True, help="Receiver-local start (PDT, UTC-7), e.g. 2025-06-17")
    p.add_argument("--end", required=True, help="Receiver-local end (exclusive)")
    p.add_argument("--temperature-csv", help="DD_N string CSV; default reads tblInterpolatedTemp (rebuilt DB only)")
    p.add_argument("--rowid-range", nargs=2, type=int, help="Bound the scan to a rowid block")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-db", help="Write legacy-style tblMetronomeFiltered / tblMetronomeSecondFiltered here")
    return p.parse_args()


def naive_seconds(text):
    # tblDetectionRaw.seconds holds study-basis wall time (PDT, UTC-7) encoded as if UTC.
    return pd.Timestamp(text, tz="UTC").timestamp()


def connect_ro(database):
    return sqlite3.connect("file:%s?mode=ro" % os.path.abspath(database).replace("\\", "/"), uri=True)


def load(args):
    con = connect_ro(args.database)
    rec = pd.read_sql("select Rec_ID, Tag_ID, X, Y, Z from tblReceiver", con).set_index("Rec_ID")
    for r in (args.beacon_receiver, args.anchor):
        if r not in rec.index:
            raise ValueError("Receiver %s not in tblReceiver" % r)
    tag = rec.loc[args.beacon_receiver, "Tag_ID"]
    period = pd.read_sql("select pulseRate from tblTag where Tag_ID = ?", con, params=[tag]).pulseRate
    if period.empty or pd.isna(period.iloc[0]):
        raise ValueError("No pulseRate for beacon %s; refusing to guess" % tag)
    query = "select seconds, Rec_ID from tblDetectionRaw where Tag_ID = ? and seconds >= ? and seconds < ?"
    params = [tag, naive_seconds(args.start), naive_seconds(args.end)]
    if args.rowid_range:
        query += " and rowid between ? and ?"
        params += args.rowid_range
    else:
        warnings.warn("No --rowid-range: full-table scan of tblDetectionRaw")
    det = pd.read_sql(query, con, params=params)
    con.close()
    if det.empty:
        raise ValueError("No detections for %s in window" % tag)
    return rec, tag, float(period.iloc[0]), det


def load_temperature(path=None, database=None):
    if path:
        t = pd.read_csv(path, usecols=["DateTime"] + DD_N_COLUMNS)
        t["s"] = pd.to_datetime(t.DateTime, format="mixed").astype("datetime64[ns]").astype("int64") / 1e9
        t["C"] = t[DD_N_COLUMNS].mean(axis=1)
        # Verified 2026-09-24: this file is local PDT (matches HOBO GMT-07:00 exports at +7 h), same basis as detections.
        return t.dropna(subset=["C"])[["s", "C"]].sort_values("s")
    con = connect_ro(database)
    t = pd.read_sql("select * from tblInterpolatedTemp", con)
    con.close()
    if "TempSource" not in t:
        raise ValueError("tblInterpolatedTemp lacks TempSource (pre-DD_N build); pass --temperature-csv")
    t["s"] = pd.to_datetime(t.timeStamp).astype("datetime64[ns]").astype("int64") / 1e9
    return t.dropna(subset=["C"])[["s", "C"]].sort_values("s")


def assign_bursts(det, period):
    det = det.sort_values(["Rec_ID", "seconds"]).copy()
    gap = det.groupby("Rec_ID").seconds.diff()
    det["burst"] = (gap.isna() | (gap >= 0.5 * period)).groupby(det.Rec_ID).cumsum()
    det["det_rank"] = det.groupby(["Rec_ID", "burst"]).cumcount() + 1
    return det


def first_arrivals(det):
    return det.groupby(["Rec_ID", "burst"]).agg(t=("seconds", "min"), burst_n=("seconds", "size")).reset_index()


def pairwise_series(first, rec, beacon_rec, anchor, period, temp):
    a = first[first.Rec_ID == anchor][["t", "burst_n"]].rename(columns={"t": "t_anchor", "burst_n": "anchor_burst_n"})
    a = a.sort_values("t_anchor").reset_index(drop=True)
    if len(a) < MIN_SAMPLES:
        raise ValueError("Anchor %s has too few beacon epochs" % anchor)
    a["transNo"] = np.arange(1, len(a) + 1)
    t_lo, t_hi = temp.s.min(), temp.s.max()
    xyz = rec[["X", "Y", "Z"]]
    d_ba = np.linalg.norm(xyz.loc[beacon_rec] - xyz.loc[anchor])
    rows = []
    for rid in sorted(first.Rec_ID.unique()):
        if rid in (beacon_rec, anchor):
            continue
        j = first[first.Rec_ID == rid][["t", "burst", "burst_n"]].sort_values("t")
        m = pd.merge_asof(j, a, left_on="t", right_on="t_anchor", direction="nearest", tolerance=0.5 * period)
        m = m.dropna(subset=["t_anchor"])
        m = m.assign(err=(m.t - m.t_anchor).abs()).sort_values("err").drop_duplicates("t_anchor").sort_values("t_anchor")
        outside = (m.t_anchor < t_lo) | (m.t_anchor > t_hi)
        if outside.any():
            warnings.warn("%s: %d epochs outside temperature coverage dropped" % (rid, int(outside.sum())))
            m = m[~outside]
        c = sos(np.interp(m.t_anchor, temp.s, temp.C))
        d_bi = np.linalg.norm(xyz.loc[beacon_rec] - xyz.loc[rid])
        m["Rec_ID"] = rid
        m["sound_speed"] = c
        m["delta_s"] = (m.t - m.t_anchor) - (d_bi - d_ba) / c
        rows.append(m.drop(columns="err"))
    return pd.concat(rows, ignore_index=True), a


def _fit_residuals(g, clean):
    g["resid_s"] = np.nan
    for _, s in g[clean].groupby("label"):
        if len(s) >= 2:
            coef = np.polyfit(s.t_anchor - s.t_anchor.iloc[0], s.delta_s, 1)
            g.loc[s.index, "resid_s"] = s.delta_s - np.polyval(coef, s.t_anchor - s.t_anchor.iloc[0])
    other = ~clean
    seg = g[clean].set_index("t_anchor").delta_s
    if other.any() and len(seg):
        g.loc[other, "resid_s"] = g.loc[other, "delta_s"] - np.interp(g.loc[other, "t_anchor"], seg.index.values, seg.values)
    return g


def steady_reflection_labels(g):
    """Clusters overlapping in time with an earlier-delta cluster by (budget, MAX_REFLECTION_DELAY_S]."""
    spans = g[g.label >= 0].groupby("label").agg(t0=("t_anchor", "min"), t1=("t_anchor", "max"), d=("delta_s", "median"))
    if len(spans) < 2:
        return set()
    t0, t1, d = spans.t0.values, spans.t1.values, spans.d.values
    # Row i = earlier (direct) cluster, column j = candidate later (reflected) cluster.
    overlap = (t0[:, None] <= t1[None, :]) & (t0[None, :] <= t1[:, None])
    gap = d[None, :] - d[:, None]
    late = overlap & (gap > TIMING_BUDGET_S) & (gap <= MAX_REFLECTION_DELAY_S)
    np.fill_diagonal(late, False)
    return set(spans.index.values[late.any(axis=0)])


def cluster(series, period):
    out = []
    for rid, g in series.groupby("Rec_ID"):
        g = g.sort_values("t_anchor").copy()
        x = np.column_stack([g.t_anchor / (TIME_WINDOW_PERIODS * period), g.delta_s / TIMING_BUDGET_S])
        g["label"] = DBSCAN(eps=1.0, min_samples=MIN_SAMPLES, metric="chebyshev").fit_predict(x)
        late = steady_reflection_labels(g)
        g["dbscan_class"] = np.where(g.label < 0, "noise", np.where(g.label.isin(late), "steady_reflection", "clean"))
        out.append(_fit_residuals(g, (g.dbscan_class == "clean").values))
    return pd.concat(out, ignore_index=True)


def anchor_suspect_epochs(res):
    by_epoch = res.groupby("t_anchor").agg(n=("Rec_ID", "size"), noise=("dbscan_class", lambda v: (v == "noise").mean()))
    return set(by_epoch[(by_epoch.n >= ANCHOR_MIN_RECEIVERS) & (by_epoch.noise >= ANCHOR_MAJORITY)].index)


def classify(series, period):
    """Two passes: find anchor-side epochs, exclude them, then re-cluster every receiver."""
    suspects = anchor_suspect_epochs(cluster(series, period))
    kept = cluster(series[~series.t_anchor.isin(suspects)], period)
    dropped = series[series.t_anchor.isin(suspects)].copy()
    dropped["label"] = -1
    dropped["dbscan_class"] = "anchor_suspect"
    dropped["resid_s"] = np.nan
    return pd.concat([kept, dropped], ignore_index=True).sort_values(["Rec_ID", "t_anchor"]), suspects


def summarize(res, anchor):
    seg_rows, sum_rows = [], []
    for rid, g in res.groupby("Rec_ID"):
        clean = g[g.dbscan_class == "clean"]
        for lab, s in clean.groupby("label"):
            span = s.t_anchor.max() - s.t_anchor.min()
            slope = np.polyfit(s.t_anchor - s.t_anchor.min(), s.delta_s, 1)[0] if len(s) >= 2 else np.nan
            seg_rows.append(dict(Rec_ID=rid, segment=lab, epochs=len(s),
                                 start_local=dt.datetime.fromtimestamp(s.t_anchor.min(), dt.timezone.utc).replace(tzinfo=None),
                                 hours=round(span / 3600, 3), drift_us_per_s=round(slope * 1e6, 4),
                                 resid_rms_ms=round(float(np.sqrt(np.nanmean(s.resid_s ** 2))) * 1e3, 4),
                                 resid_over_budget=int((s.resid_s.abs() > TIMING_BUDGET_S).sum())))
        noise = g[g.dbscan_class == "noise"]
        judged = g[g.dbscan_class != "anchor_suspect"]
        rms = float(np.sqrt(np.nanmean(clean.resid_s ** 2))) * 1e3 if clean.resid_s.notna().any() else np.nan
        sum_rows.append(dict(Rec_ID=rid, anchor=anchor, epochs=len(g), judged_epochs=len(judged),
                             segments=clean.label.nunique(),
                             noise_fraction=round(len(noise) / max(len(judged), 1), 4),
                             steady_reflection_epochs=int((g.dbscan_class == "steady_reflection").sum()),
                             anchor_suspect_epochs=int((g.dbscan_class == "anchor_suspect").sum()),
                             noise_late_share=round(float((noise.resid_s > 0).mean()), 3) if len(noise) else np.nan,
                             clean_resid_rms_ms=round(rms, 4),
                             clean_over_budget=int((clean.resid_s.abs() > TIMING_BUDGET_S).sum())))
    return pd.DataFrame(seg_rows), pd.DataFrame(sum_rows)


def write_legacy_tables(path, det, res, anchor_epochs, tag, anchor):
    """Legacy-style metronome tables. transNo = anchor epoch number; seconds are uncorrected."""
    first_map = res[["Rec_ID", "burst", "transNo", "dbscan_class", "delta_s"]]
    a = anchor_epochs.assign(Rec_ID=anchor)
    anchor_class = np.where(a.t_anchor.isin(set(res.loc[res.dbscan_class == "anchor_suspect", "t_anchor"])), "anchor_suspect", "anchor")
    anchor_bursts = det[(det.Rec_ID == anchor) & (det.det_rank == 1)][["Rec_ID", "burst", "seconds"]]
    anchor_bursts = anchor_bursts.merge(a[["t_anchor", "transNo"]].assign(dbscan_class=anchor_class, delta_s=0.0),
                                        left_on="seconds", right_on="t_anchor").drop(columns=["seconds", "t_anchor"])
    first_map = pd.concat([first_map, anchor_bursts.assign(Rec_ID=anchor)], ignore_index=True)
    rows = det.merge(first_map, on=["Rec_ID", "burst"], how="inner")
    rows["Tag_ID"] = tag
    rows["timeStamp"] = pd.to_datetime(rows.seconds, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    rows["multipath"] = (rows.det_rank > 1).astype(int)
    primary = rows[["Rec_ID", "Tag_ID", "timeStamp", "seconds", "transNo", "det_rank", "multipath"]]
    second = rows[rows.multipath == 0].copy()
    second["multipath_prediction"] = (~second.dbscan_class.isin(["clean", "anchor"])).astype(int)
    second = second[["Rec_ID", "Tag_ID", "timeStamp", "seconds", "transNo", "det_rank", "multipath",
                     "multipath_prediction", "dbscan_class", "delta_s"]]
    con = sqlite3.connect(path)
    try:
        primary.to_sql("tblMetronomeFiltered", con, if_exists="replace", index=False)
        second.to_sql("tblMetronomeSecondFiltered", con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()
    return len(primary), len(second)


def plot(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = sorted(res.Rec_ID.unique())
    fig, axes = plt.subplots(len(recs), 1, figsize=(10, 1.6 * len(recs)), sharex=True)
    for ax, rid in zip(np.atleast_1d(axes), recs):
        g = res[res.Rec_ID == rid]
        t = pd.to_datetime(g.t_anchor, unit="s")
        clean = g.dbscan_class == "clean"
        ax.scatter(t[clean], g.delta_s[clean] * 1e3, s=1, c=g.label[clean] % 10, cmap="tab10")
        for cls, colour in (("noise", "k"), ("steady_reflection", "r"), ("anchor_suspect", "0.7")):
            sel = g.dbscan_class == cls
            ax.scatter(t[sel], g.delta_s[sel] * 1e3, s=2, c=colour)
        ax.set_ylabel(rid, rotation=0, labelpad=20)
    np.atleast_1d(axes)[0].set_title(title + "  (colour=segment, black=noise, red=steady reflection, grey=anchor-side; y=delta ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    rec, tag, period, det = load(args)
    temp = load_temperature(args.temperature_csv, args.database)
    det = assign_bursts(det, period)
    first = first_arrivals(det)
    series, anchor_epochs = pairwise_series(first, rec, args.beacon_receiver, args.anchor, period, temp)
    res, suspects = classify(series, period)
    segments, summary = summarize(res, args.anchor)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, "%s_%s_anchor_%s" % (args.beacon_receiver, tag, args.anchor))
    res.to_csv(stem + "_epochs.csv", index=False, float_format="%.6f")
    segments.to_csv(stem + "_segments.csv", index=False)
    summary.to_csv(stem + "_summary.csv", index=False)
    plot(res, stem + "_delta.png", "%s beacon %s vs anchor %s" % (args.beacon_receiver, tag, args.anchor))
    print("Beacon %s (%s), period %.1f s, anchor %s" % (tag, args.beacon_receiver, period, args.anchor))
    print("Fixed parameters: budget %.1f ms, window %.1f periods, min_samples %d, anchor majority %.2f of >=%d, "
          "max reflection delay %.0f ms" % (TIMING_BUDGET_S * 1e3, TIME_WINDOW_PERIODS, MIN_SAMPLES,
                                            ANCHOR_MAJORITY, ANCHOR_MIN_RECEIVERS, MAX_REFLECTION_DELAY_S * 1e3))
    print("Detections %d; first arrivals %d; paired epochs %d; anchor epochs %d; anchor-side epochs %d"
          % (len(det), len(first), len(res), len(anchor_epochs), len(suspects)))
    print(summary.to_string(index=False))
    for r in summary[summary.clean_over_budget > 0].itertuples():
        print("WARNING: %s has %d clean epochs with residual > %.1f ms" % (r.Rec_ID, r.clean_over_budget, TIMING_BUDGET_S * 1e3))
    print("WARNING: beacon host %s position is unsurveyed; its error is absorbed as constant per-receiver offsets"
          % args.beacon_receiver)
    if args.output_db:
        n1, n2 = write_legacy_tables(args.output_db, det, res, anchor_epochs, tag, args.anchor)
        print("Wrote %s: tblMetronomeFiltered %d rows, tblMetronomeSecondFiltered %d rows" % (args.output_db, n1, n2))
    print("No source detections modified. Outputs: %s_*" % stem)


if __name__ == "__main__":
    main()
