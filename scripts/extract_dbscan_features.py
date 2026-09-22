"""Extract diagnostic multipath features without filtering detections."""
import argparse
import os
import sqlite3

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--period-seconds", type=float, required=True)
    parser.add_argument("--receiver")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def extract_features(database, tag, period_seconds, receiver=None):
    if period_seconds <= 0:
        raise ValueError("period-seconds must be positive")
    connection = sqlite3.connect(database)
    query = "select rowid as source_rowid, * from tblDetectionRaw where Tag_ID = ?"
    params = [tag]
    if receiver is not None:
        query += " and Rec_ID = ?"
        params.append(receiver)
    data = pd.read_sql_query(query, connection, params=params)
    connection.close()
    if data.empty:
        raise ValueError("No detections found for tag %s" % tag)
    data["time_seconds"] = pd.to_numeric(data["seconds"], errors="coerce")
    data["amplitude_value"] = pd.to_numeric(data["SigStr"], errors="coerce")
    data = data.dropna(subset=["time_seconds", "amplitude_value", "Rec_ID"])
    data = data.sort_values(["Rec_ID", "time_seconds"]).copy()
    group = data.groupby("Rec_ID", sort=False)
    data["between_detection_seconds"] = group["time_seconds"].diff()
    data["new_epoch"] = data["between_detection_seconds"].isna() | (
        data["between_detection_seconds"] >= period_seconds / 2.0
    )
    data["epoch_number"] = data.groupby("Rec_ID")["new_epoch"].cumsum().astype("int64")
    epoch_group = data.groupby(["Rec_ID", "epoch_number"], sort=False)
    data["epoch_first_seconds"] = epoch_group["time_seconds"].transform("min")
    data["lag_seconds"] = data["time_seconds"] - data["epoch_first_seconds"]
    data["epoch_max_sigstr"] = epoch_group["amplitude_value"].transform("max")
    data["relative_sigstr"] = data["amplitude_value"] - data["epoch_max_sigstr"]
    data["epoch_rank"] = epoch_group.cumcount()
    data["inter_detection_seconds"] = data["between_detection_seconds"].where(
        ~data["new_epoch"]
    )
    return data[[
        "source_rowid", "Tag_ID", "Rec_ID", "timeStamp", "seconds", "SigStr",
        "epoch_number", "lag_seconds", "relative_sigstr", "epoch_rank",
        "inter_detection_seconds",
    ]]


def main():
    args = parse_args()
    features = extract_features(
        args.database, args.tag, args.period_seconds, args.receiver
    )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    features.to_csv(args.output, index=False)
    print("Created: %s" % args.output)
    print("Rows: %s" % len(features))
    print("Epochs: %s" % features[["Rec_ID", "epoch_number"]].drop_duplicates().shape[0])


if __name__ == "__main__":
    main()