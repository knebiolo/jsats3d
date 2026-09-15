"""Reconcile configured, GPS, and detected receiver inventories."""
import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-xlsx", required=True)
    parser.add_argument("--gps-csv", required=True)
    parser.add_argument("--detection-dir", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--output", default="output/receiver_inventory_audit.csv")
    return parser.parse_args()


def receiver_names(values):
    return set(values.dropna().astype(str).str.strip())


def main():
    args = parse_args()
    config = pd.read_excel(args.config_xlsx)
    config.columns = config.columns.astype(str).str.strip()
    config["Receiver Name"] = config["Receiver Name"].astype(str).str.strip()
    configured = config.drop_duplicates("Receiver Name").set_index("Receiver Name")

    gps = pd.read_csv(args.gps_csv, usecols=["receiverName"])
    gps_names = receiver_names(gps["receiverName"])
    detected_by_file = {}
    detected = set()
    for filename in args.files:
        path = os.path.join(args.detection_dir, filename)
        names = set()
        for chunk in pd.read_csv(path, usecols=["receiverName"], chunksize=args.chunksize):
            names.update(receiver_names(chunk["receiverName"]))
        detected_by_file[filename] = names
        detected.update(names)

    rows = []
    for name in sorted(configured.index):
        rows.append({
            "receiverName": name,
            "configured": True,
            "gps_records": name in gps_names,
            "detected_any_file": name in detected,
            "detected_files": ";".join(
                filename for filename, names in detected_by_file.items() if name in names
            ),
            "receiver_model": configured.at[name, "Receiver Model"],
            "beacon_tag": configured.at[name, "Beacon Tag Code"],
            "beacon_period_seconds": configured.at[name, "Beacon Tag Period (sec)"],
            "latitude": configured.at[name, "Latitude (degrees)"],
            "longitude": configured.at[name, "Longitude (degrees)"],
            "hydrophone_depth_feet": configured.at[name, "Hydrophone Depth (feet)"],
        })

    inventory = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    inventory.to_csv(args.output, index=False)
    print("Created: %s" % args.output)
    print("Configured receivers: %s" % len(configured))
    print("GPS receivers: %s" % len(gps_names))
    print("Detected receivers: %s" % len(detected))
    print("Detected but not configured: %s" % sorted(detected - set(configured.index)))


if __name__ == "__main__":
    main()