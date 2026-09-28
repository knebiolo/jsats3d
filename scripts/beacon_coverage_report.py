"""Beacon coverage report: which beacon is heard most uniformly across all other receivers
(diagnostic; read-only; rejects nothing). Answers meeting items 1 and 9: identify the
metronome candidate, defined as the beacon heard by the most receivers, most consistently.
"""
import argparse
import os
import sqlite3
import time

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("database")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def connect_ro(database):
    return sqlite3.connect("file:%s?mode=ro" % os.path.abspath(database).replace("\\", "/"), uri=True)


def scan(database):
    con = connect_ro(database)
    tags = pd.read_sql("select Tag_ID, TagType, pulseRate from tblTag", con)
    rec = pd.read_sql("select Rec_ID, Tag_ID from tblReceiver", con)
    t0 = time.time()
    # Per tag x receiver x hour: distinct minutes with a detection (about one per ping;
    # multipath copies of the same ping share the minute, so this does not double count).
    hourly = pd.read_sql(
        "select Tag_ID, Rec_ID, cast(seconds/3600 as int) as hr, count(distinct cast(seconds/60 as int)) as minutes "
        "from tblDetectionRaw group by Tag_ID, Rec_ID, hr", con)
    con.close()
    print("Full-table scan: %.0f s, %d tag x receiver x hour groups" % (time.time() - t0, len(hourly)))
    return tags, rec, hourly


def build_report(tags, rec, hourly):
    op_hours = hourly.groupby("Rec_ID").hr.nunique()
    host = rec.dropna(subset=["Tag_ID"]).set_index("Tag_ID").Rec_ID
    beacons = tags[tags.TagType == "beacon"].Tag_ID
    b = hourly[hourly.Tag_ID.isin(beacons)]
    m = b.groupby(["Tag_ID", "Rec_ID"]).agg(hours=("hr", "nunique"), minutes=("minutes", "sum")).reset_index()
    m["share_of_listener_hours"] = m.hours / m.Rec_ID.map(op_hours)
    m["pings_per_heard_hour"] = m.minutes / m.hours
    m["host"] = m.Tag_ID.map(host).fillna("array-wide/unassigned")
    other = m[m.Rec_ID != m.host]
    share = other.pivot(index="Tag_ID", columns="Rec_ID", values="share_of_listener_hours").fillna(0)
    rate = other.pivot(index="Tag_ID", columns="Rec_ID", values="pings_per_heard_hour")
    summary = pd.DataFrame({
        "host": share.index.map(lambda t: host.get(t, "array-wide/unassigned")),
        "nominal_period_s": share.index.map(tags.set_index("Tag_ID").pulseRate),
        "listeners_ge90pct_hours": (share >= 0.9).sum(axis=1),
        "listeners_total": share.notna().sum(axis=1),
        "median_pings_per_heard_hour": rate.median(axis=1),
        "min_pings_per_heard_hour": rate.min(axis=1),
        # A metronome must be reliably heard by every receiver, so rank on the worst-case
        # (minimum) rate, not the typical (median) one; a high median with one weak receiver
        # is not "heard by all".
    }).sort_values(["listeners_ge90pct_hours", "min_pings_per_heard_hour"], ascending=False)
    return m, share, summary, op_hours


def plot(share, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = summary.index
    fig, ax = plt.subplots(figsize=(12, 0.28 * len(order) + 2))
    im = ax.imshow(share.loc[order].values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["%s (%s)" % (t, summary.loc[t, "host"]) for t in order], fontsize=7)
    ax.set_xticks(range(share.shape[1]))
    ax.set_xticklabels(share.columns, rotation=90, fontsize=7)
    ax.set_xlabel("Listening receiver")
    ax.set_title("Beacon coverage: share of each listener's operating hours with >=1 detection\n"
                 "(full season; darker = worse coverage; metronome candidate = most uniform bright row)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.02, label="share of listener hours")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tags, rec, hourly = scan(args.database)
    matrix, share, summary, op_hours = build_report(tags, rec, hourly)
    matrix.to_csv(os.path.join(args.output_dir, "beacon_coverage_matrix.csv"), index=False, float_format="%.4f")
    summary.to_csv(os.path.join(args.output_dir, "beacon_coverage_summary.csv"), float_format="%.3f")
    plot(share, summary, os.path.join(args.output_dir, "beacon_coverage_heatmap.png"))
    print("Listener operating hours (any detection):\n" + op_hours.to_string())
    print(summary.round(3).to_string())
    best = summary.index[0]
    print("Best metronome candidate (most receivers >=90%% of hours, best WORST-CASE pings/hour "
          "since a metronome must be heard by every receiver, not just most): %s (host %s)"
          % (best, summary.loc[best, "host"]))
    print("No detections modified. Outputs in %s" % args.output_dir)


if __name__ == "__main__":
    main()
