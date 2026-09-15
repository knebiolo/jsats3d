"""Audit detection CSV schemas and timestamp precision without modifying inputs."""
import argparse
import csv
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detection_dir")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "master_df_test.csv",
            "master_df_study.csv",
            "master_df_beacon.csv",
            "master_df_unknown.csv",
        ],
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Inspect headers and first 1,000 rows without scanning full files",
    )
    parser.add_argument("--output", default="output/detection_schema_audit.txt")
    return parser.parse_args()


def timestamp_digits(values):
    digits = []
    for value in values.dropna().astype(str):
        if "." in value:
            fraction = value.split(".", 1)[1]
            fraction = fraction.split("+", 1)[0].split("Z", 1)[0]
            digits.append(len(fraction.rstrip()))
    return sorted(set(digits))


def audit_file(path, chunksize, sample_only=False):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        columns = next(reader)

    timestamp_column = next(
        (column for column in ("dateTime", "timeStamp", "DateTime") if column in columns),
        None,
    )
    sample = pd.read_csv(path, nrows=1000)
    report = ["File: %s" % path, "Columns: %s" % ", ".join(columns)]
    report.append("Timestamp column: %s" % timestamp_column)
    report.append("Sample timestamp fractional digits: %s" % (
        timestamp_digits(sample[timestamp_column]) if timestamp_column else "unavailable"
    ))
    report.append("Rows scanned: 0")
    report.append("Tag count: unavailable")
    report.append("Receiver count: unavailable")
    report.append("Observed signal fields: %s" % ", ".join(
        field for field in ("amp", "Amplitude", "SNR", "NBW", "FreqOff", "Pascals", "Celsius")
        if field in columns
    ))

    if sample_only:
        return report

    rows = 0
    tags = set()
    receivers = set()
    for chunk in pd.read_csv(path, chunksize=chunksize):
        rows += len(chunk)
        if "tagCode" in chunk:
            tags.update(chunk["tagCode"].dropna().astype(str).str.strip().unique())
        if "receiverName" in chunk:
            receivers.update(chunk["receiverName"].dropna().astype(str).str.strip().unique())
    report[4] = "Rows scanned: %s" % rows
    report[5] = "Tag count: %s" % len(tags)
    report[6] = "Receiver count: %s" % len(receivers)
    return report


def main():
    args = parse_args()
    report = ["Detection Schema Audit", "", "Input directory: %s" % args.detection_dir, ""]
    for filename in args.files:
        path = os.path.join(args.detection_dir, filename)
        if not os.path.exists(path):
            report.extend(["File: %s" % path, "STATUS: MISSING", ""])
            continue
        report.extend(audit_file(path, args.chunksize, args.sample_only))
        report.append("")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        stream.write("\n".join(report))
    print("Created: %s" % args.output)


if __name__ == "__main__":
    main()