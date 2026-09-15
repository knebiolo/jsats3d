"""Measure observed detection intervals for one tag from a CSV file.

This is descriptive only. It does not infer a pulse rate or apply blanking.
"""
import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("--tag", action="append", dest="tags")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--output-csv")
    return parser.parse_args()


def measure_intervals(input_csv, tag_codes, chunksize):
    tag_codes = set(tag_codes)
    required = {"dateTime", "tagCode", "receiverName"}
    detections = []
    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError("Missing required columns: %s" % sorted(missing))
        selected = chunk[chunk["tagCode"].astype(str).str.strip().isin(tag_codes)].copy()
        if selected.empty:
            continue
        selected["timeStamp"] = pd.to_datetime(selected["dateTime"], errors="coerce")
        selected = selected.dropna(subset=["timeStamp", "receiverName"])
        selected["receiverName"] = selected["receiverName"].astype(str).str.strip()
        detections.append(selected[["tagCode", "receiverName", "timeStamp"]])

    if not detections:
        raise ValueError("No valid detections found for tags %s" % sorted(tag_codes))

    detections = pd.concat(detections, ignore_index=True)
    detections["tagCode"] = detections.get("tagCode", pd.Series(index=detections.index, dtype="string"))
    detections = detections.sort_values(["tagCode", "receiverName", "timeStamp"])
    detections["interval_seconds"] = (
        detections.groupby(["tagCode", "receiverName"])["timeStamp"].diff().dt.total_seconds()
    )
    intervals = detections.dropna(subset=["interval_seconds"])
    intervals = intervals[intervals["interval_seconds"] > 0]
    if intervals.empty:
        raise ValueError("No positive intervals found for tags %s" % sorted(tag_codes))

    summary = intervals.groupby(["tagCode", "receiverName"])["interval_seconds"].agg(
        detection_count="count",
        median="median",
        mean="mean",
        minimum="min",
        maximum="max",
        p05=lambda values: values.quantile(0.05),
        p95=lambda values: values.quantile(0.95),
    ).reset_index()
    return summary


def main():
    args = parse_args()
    tags = args.tags or ["FFD3"]
    summary = measure_intervals(args.input_csv, tags, args.chunksize)
    output_csv = args.output_csv
    if output_csv is None:
        stem, _ = os.path.splitext(args.input_csv)
        label = "_".join(tags)
        output_csv = "%s_%s_intervals.csv" % (stem, label)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print("Created: %s" % output_csv)
    print("Tags: %s" % ", ".join(tags))
    print("Tag/receivers: %s" % len(summary))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()