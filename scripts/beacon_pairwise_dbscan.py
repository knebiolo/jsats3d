"""Pairwise beacon TDoA DBSCAN diagnostic (read-only; writes CSV/PNG only, rejects nothing).

For beacon B heard at receiver i and anchor a:
    delta_ia = (t_i - t_a) - (d_Bi - d_Ba) / c = (eps_i - eps_a) + (m_i - m_a)
The beacon host clock cancels. Clean epochs form piecewise-smooth clock segments;
multipath (m >= 0, observed >= ~1.5 ms) appears as isolated outliers.
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
DD_N_COLUMNS = ["DD_N_0p5", "DD_N_1p5", "DD_N_9", "DD_N_18"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("database")
    p.add_argument("--beacon-receiver", required=True, help="Receiver hosting the beacon (e.g. ZOI02)")
    p.add_argument("--anchor", required=True, help="Clock anchor receiver (e.g. ZOI09)")
    p.add_argument("--start", required=True, help="Receiver-local start (PDT, UTC-7), e.g. 2025-06-17")
    p.add_argument("--end", required=True, help="Receiver-local end (exclusive)")
    p.add_argument("--temperature-csv", required=True)
    p.add_argument("--rowid-range", nargs=2, type=int, help="Bound the scan to a rowid block")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def naive_seconds(text):
    # tblDetectionRaw.seconds holds receiver-local wall time (raw header "-07z") encoded as if UTC.
    return pd.Timestamp(text, tz="UTC").timestamp()


def load(args):
    con = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(args.database).replace("\\", "/"), uri=True)
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


def load_temperature(path):
    t = pd.read_csv(path, usecols=["DateTime"] + DD_N_COLUMNS)
    t["s"] = pd.to_datetime(t.DateTime, format="mixed").astype("datetime64[ns]").astype("int64") / 1e9
    t["C"] = t[DD_N_COLUMNS].mean(axis=1)
    # Verified 2026-09-24: this file is local PDT (matches HOBO GMT-07:00 exports at +7 h), same basis as detections.
    return t.dropna(subset=["C"])[["s", "C"]].sort_values("s")


def first_arrivals(det, period):
    det = det.sort_values(["Rec_ID", "seconds"])
    gap = det.groupby("Rec_ID").seconds.diff()
    det["burst"] = (gap.isna() | (gap >= 0.5 * period)).groupby(det.Rec_ID).cumsum()
    first = det.groupby(["Rec_ID", "burst"]).agg(t=("seconds", "min"), burst_n=("seconds", "size")).reset_index()
    return first


def pairwise_series(first, rec, beacon_rec, anchor, period, temp):
    a = first[first.Rec_ID == anchor][["t", "burst_n"]].rename(columns={"t": "t_anchor", "burst_n": "anchor_burst_n"})
    a = a.sort_values("t_anchor")
    if len(a) < MIN_SAMPLES:
        raise ValueError("Anchor %s has too few beacon epochs" % anchor)
    t_lo, t_hi = temp.s.min(), temp.s.max()
    xyz = rec[["X", "Y", "Z"]]
    d_ba = np.linalg.norm(xyz.loc[beacon_rec] - xyz.loc[anchor])
    rows = []
    for rid in sorted(first.Rec_ID.unique()):
        if rid in (beacon_rec, anchor):
            continue
        j = first[first.Rec_ID == rid][["t", "burst_n"]].sort_values("t")
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
    return pd.concat(rows, ignore_index=True)


def cluster(series, period):
    out = []
    for rid, g in series.groupby("Rec_ID"):
        g = g.sort_values("t_anchor").copy()
        x = np.column_stack([g.t_anchor / (TIME_WINDOW_PERIODS * period), g.delta_s / TIMING_BUDGET_S])
        g["label"] = DBSCAN(eps=1.0, min_samples=MIN_SAMPLES, metric="chebyshev").fit_predict(x)
        g["resid_s"] = np.nan
        for lab, s in g[g.label >= 0].groupby("label"):
            if len(s) >= 2:
                coef = np.polyfit(s.t_anchor - s.t_anchor.iloc[0], s.delta_s, 1)
                g.loc[s.index, "resid_s"] = s.delta_s - np.polyval(coef, s.t_anchor - s.t_anchor.iloc[0])
        noise = g.label < 0
        if noise.any():
            seg = g[~noise].set_index("t_anchor").delta_s
            if len(seg):
                ref = np.interp(g.loc[noise, "t_anchor"], seg.index.values, seg.values)
                g.loc[noise, "resid_s"] = g.loc[noise, "delta_s"] - ref
        out.append(g)
    return pd.concat(out, ignore_index=True)


def summarize(res, anchor):
    seg_rows, sum_rows = [], []
    for rid, g in res.groupby("Rec_ID"):
        segs = g[g.label >= 0].groupby("label")
        for lab, s in segs:
            span = s.t_anchor.max() - s.t_anchor.min()
            slope = np.polyfit(s.t_anchor - s.t_anchor.min(), s.delta_s, 1)[0] if len(s) >= 2 else np.nan
            seg_rows.append(dict(Rec_ID=rid, segment=lab, epochs=len(s),
                                 start_local=dt.datetime.fromtimestamp(s.t_anchor.min(), dt.timezone.utc).replace(tzinfo=None),
                                 hours=round(span / 3600, 3), drift_us_per_s=round(slope * 1e6, 4),
                                 resid_rms_ms=round(float(np.sqrt(np.nanmean(s.resid_s ** 2))) * 1e3, 4),
                                 resid_over_budget=int((s.resid_s.abs() > TIMING_BUDGET_S).sum())))
        noise = g[g.label < 0]
        clustered = g[g.label >= 0]
        sum_rows.append(dict(Rec_ID=rid, anchor=anchor, epochs=len(g), segments=int(g.label.max() + 1),
                             noise_epochs=len(noise), noise_fraction=round(len(noise) / len(g), 4),
                             noise_late_share=round(float((noise.resid_s > 0).mean()), 3) if len(noise) else np.nan,
                             noise_multi_burst_share=round(float((noise.burst_n > 1).mean()), 3) if len(noise) else np.nan,
                             clustered_multi_burst_share=round(float((clustered.burst_n > 1).mean()), 3),
                             clustered_resid_rms_ms=round(float(np.sqrt(np.nanmean(clustered.resid_s ** 2))) * 1e3, 4),
                             clustered_over_budget=int((clustered.resid_s.abs() > TIMING_BUDGET_S).sum())))
    summary = pd.DataFrame(sum_rows)
    epoch_noise = res.groupby("t_anchor").label.agg(lambda v: (v < 0).mean())
    receivers = res.groupby("t_anchor").Rec_ID.size()
    common = epoch_noise[(epoch_noise >= 0.5) & (receivers >= 4)]
    return pd.DataFrame(seg_rows), summary, common


def plot(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = sorted(res.Rec_ID.unique())
    fig, axes = plt.subplots(len(recs), 1, figsize=(10, 1.6 * len(recs)), sharex=True)
    for ax, rid in zip(np.atleast_1d(axes), recs):
        g = res[res.Rec_ID == rid]
        t = pd.to_datetime(g.t_anchor, unit="s")
        ax.scatter(t[g.label >= 0], g.delta_s[g.label >= 0] * 1e3, s=1, c=g.label[g.label >= 0] % 10, cmap="tab10")
        ax.scatter(t[g.label < 0], g.delta_s[g.label < 0] * 1e3, s=2, c="k")
        ax.set_ylabel(rid, rotation=0, labelpad=20)
    np.atleast_1d(axes)[0].set_title(title + "  (colour = segment, black = noise; y = delta ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    rec, tag, period, det = load(args)
    temp = load_temperature(args.temperature_csv)
    first = first_arrivals(det, period)
    series = pairwise_series(first, rec, args.beacon_receiver, args.anchor, period, temp)
    res = cluster(series, period)
    segments, summary, common = summarize(res, args.anchor)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, "%s_%s_anchor_%s" % (args.beacon_receiver, tag, args.anchor))
    res.to_csv(stem + "_epochs.csv", index=False, float_format="%.6f")
    segments.to_csv(stem + "_segments.csv", index=False)
    summary.to_csv(stem + "_summary.csv", index=False)
    plot(res, stem + "_delta.png", "%s beacon %s vs anchor %s" % (args.beacon_receiver, tag, args.anchor))
    print("Beacon %s (%s), period %.1f s, anchor %s" % (tag, args.beacon_receiver, period, args.anchor))
    print("Fixed parameters: budget %.1f ms, window %.1f periods, min_samples %d"
          % (TIMING_BUDGET_S * 1e3, TIME_WINDOW_PERIODS, MIN_SAMPLES))
    print("Detections %d; first arrivals %d; paired epochs %d" % (len(det), len(first), len(res)))
    print(summary.to_string(index=False))
    print("Epochs where >=50%% of >=4 receivers are noise (anchor-side candidates): %d" % len(common))
    over = summary[summary.clustered_over_budget > 0]
    for r in over.itertuples():
        print("WARNING: %s has %d clustered epochs with residual > %.1f ms" % (r.Rec_ID, r.clustered_over_budget, TIMING_BUDGET_S * 1e3))
    print("No detections rejected. Outputs: %s_*" % stem)


if __name__ == "__main__":
    main()
