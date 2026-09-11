"""Stage 2025 processed detections in legacy jsats3d SQLite tables.

This adapter keeps jsats3d.py unchanged. Missing legacy measurements remain
NULL and are reported instead of being replaced with invented values.
"""
import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
from pyproj import Transformer


DETECTION_COLUMNS = {
    "dateTime": "timeStamp",
    "tagCode": "Tag_ID",
    "amp": "Amplitude",
    "receiverName": "Rec_ID",
}


def parse_args():
    root = r"K:\Jobs\5662\001\Data\DataTrans\2025_Data"
    detection_dir = os.path.join(
        root, "CowlitzAT2025_Data_Deliverables", "2_AT_detection_datasets"
    )
    default_output = os.path.join(root, "jsats3d_2025_staging.db")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-dir", default=detection_dir)
    parser.add_argument("--output-db", default=default_output)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--tag", help="Only stage one tag code")
    parser.add_argument("--config-xlsx", default=os.path.join(root, "CowlitzAT2025_Data_Deliverables", "1_array_metadata", "cowlitz_2025_AT_config.xlsx"))
    parser.add_argument("--covariate-csv", default=os.path.join(root, "Master Covariate Table", "2025 Master Covariate Table_20251212.csv"))
    parser.add_argument(
        "--files",
        nargs="+",
        default=["master_df_test.csv", "master_df_study.csv", "master_df_beacon.csv", "master_df_unknown.csv"],
    )
    return parser.parse_args()


def load_receiver_table(gps_path, config_path):
    gps = pd.read_csv(gps_path, usecols=["receiverName", "easting", "northing"])
    gps = gps.dropna(subset=["receiverName", "easting", "northing"])
    gps = gps.groupby("receiverName", as_index=False)[["easting", "northing"]].median()
    config = pd.read_excel(config_path)
    config = config.rename(columns={
        "Receiver Name": "Rec_ID",
        "Beacon Tag Code": "BeaconTag_ID",
        "Hydrophone Depth (feet)": "Depth_ft",
        "Latitude (degrees)": "latitude",
        "Longitude (degrees)": "longitude",
    })
    config = config[config["Rec_ID"].notna()].copy()
    config = config.drop_duplicates("Rec_ID")
    if gps.empty and config.empty:
        return pd.DataFrame(columns=["Rec_ID", "Tag_ID", "Ref_Elev", "X", "Y", "Z", "X_t", "Y_t", "Z_t"])

    gps = gps.rename(columns={"receiverName": "Rec_ID"})
    receivers = config.merge(gps, on="Rec_ID", how="left")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26910", always_xy=True)
    config_x, config_y = transformer.transform(
        pd.to_numeric(receivers["longitude"], errors="coerce"),
        pd.to_numeric(receivers["latitude"], errors="coerce"),
    )
    receivers["easting"] = receivers["easting"].fillna(pd.Series(config_x, index=receivers.index))
    receivers["northing"] = receivers["northing"].fillna(pd.Series(config_y, index=receivers.index))
    receivers["Tag_ID"] = receivers["BeaconTag_ID"]
    receivers["Ref_Elev"] = "BM"
    receivers["Z"] = -pd.to_numeric(receivers["Depth_ft"], errors="coerce") * 0.3048
    receivers["Z_t"] = receivers["Z"]
    origin_x = receivers["easting"].min()
    origin_y = receivers["northing"].min()
    receivers["X"] = receivers["easting"] - origin_x
    receivers["Y"] = receivers["northing"] - origin_y
    receivers["X_t"] = receivers["X"]
    receivers["Y_t"] = receivers["Y"]
    receivers = receivers[["Rec_ID", "Tag_ID", "Ref_Elev", "X", "Y", "Z", "X_t", "Y_t", "Z_t"]]
    return receivers


def load_environment(covariate_path):
    columns = ["DateTime", "BB_TPU_Surface_t", "FBS_Surface_t", "NSC.CZD_WTR_EL.F_CV"]
    covariates = pd.read_csv(covariate_path, usecols=columns)
    covariates["timeStamp"] = pd.to_datetime(covariates["DateTime"], errors="coerce")
    covariates["C"] = pd.to_numeric(covariates["BB_TPU_Surface_t"], errors="coerce")
    covariates["WSEL"] = pd.to_numeric(covariates["NSC.CZD_WTR_EL.F_CV"], errors="coerce")
    covariates = covariates.dropna(subset=["timeStamp"])
    temperature = covariates[["timeStamp", "C"]].dropna().drop_duplicates("timeStamp")
    wsel = covariates[["timeStamp", "WSEL"]].dropna().drop_duplicates("timeStamp")
    return temperature, wsel


def normalize_detection(chunk, tag_type, tag_filter=None):
    missing = set(DETECTION_COLUMNS) - set(chunk.columns)
    if missing:
        raise ValueError("Missing detection columns: %s" % sorted(missing))
    result = chunk.rename(columns=DETECTION_COLUMNS).copy()
    if tag_filter is not None:
        result = result[result["Tag_ID"].astype(str).str.strip() == tag_filter]
    if result.empty:
        return pd.DataFrame(columns=[
            "timeStamp", "seconds", "Tag_ID", "Rec_ID", "FreqOff", "Amplitude",
            "NBW", "SNR", "Valid", "Pascals", "Celsius", "TagTypeSource",
        ])
    result["timeStamp"] = pd.to_datetime(result["timeStamp"], errors="coerce")
    result = result.dropna(subset=["timeStamp", "Tag_ID", "Rec_ID"])
    result["seconds"] = result["timeStamp"].astype("int64") / 1e9
    result["Tag_ID"] = result["Tag_ID"].astype(str).str.strip()
    result["Rec_ID"] = result["Rec_ID"].astype(str).str.strip()
    result["Amplitude"] = pd.to_numeric(result["Amplitude"], errors="coerce")
    result["NBW"] = np.nan
    result["SNR"] = np.nan
    result["FreqOff"] = np.nan
    result["Valid"] = True
    result["Pascals"] = np.nan
    result["Celsius"] = np.nan
    result["TagTypeSource"] = tag_type
    return result[[
        "timeStamp", "seconds", "Tag_ID", "Rec_ID", "FreqOff", "Amplitude",
        "NBW", "SNR", "Valid", "Pascals", "Celsius", "TagTypeSource",
    ]]


def write_detection_tables(connection, detection_dir, chunksize, tag_filter=None, filenames=None):
    tag_types = {
        "master_df_test.csv": "study",
        "master_df_study.csv": "study",
        "master_df_beacon.csv": "beacon",
        "master_df_unknown.csv": "unknown",
    }
    files = [(filename, tag_types.get(filename, "unknown")) for filename in filenames]
    tags = []
    receivers = set()
    total = 0
    for filename, tag_type in files:
        path = os.path.join(detection_dir, filename)
        if not os.path.exists(path):
            print("Skipped missing file: %s" % path)
            continue
        first_chunk = True
        for chunk in pd.read_csv(path, chunksize=chunksize):
            normalized = normalize_detection(chunk, tag_type, tag_filter)
            if normalized.empty:
                continue
            normalized.to_sql(
                "tblDetectionRaw", connection, if_exists="append", index=False
            )
            tags.append(normalized[["Tag_ID", "TagTypeSource"]])
            receivers.update(normalized["Rec_ID"].unique())
            total += len(normalized)
            if first_chunk:
                print("Imported %s" % filename)
                first_chunk = False
    if not tags:
        raise ValueError("No detection data imported")
    return pd.concat(tags, ignore_index=True).drop_duplicates(), receivers, total


def main():
    args = parse_args()
    gps_path = os.path.join(
        os.path.dirname(args.detection_dir), "4_gps_datasets", "master_df_gps.csv"
    )
    if os.path.exists(args.output_db):
        os.remove(args.output_db)
    os.makedirs(os.path.dirname(args.output_db), exist_ok=True)
    connection = sqlite3.connect(args.output_db)
    try:
        tags, detected_receivers, row_count = write_detection_tables(
            connection, args.detection_dir, args.chunksize, args.tag, args.files
        )
        receiver_table = load_receiver_table(gps_path, args.config_xlsx)
        receiver_table = receiver_table[receiver_table.Rec_ID.isin(detected_receivers)]
        receiver_table.to_sql("tblReceiver", connection, if_exists="replace", index=False)

        tags = tags.rename(columns={"TagTypeSource": "TagType"})
        tags["pulseRate"] = np.nan
        tags.to_sql("tblTag", connection, if_exists="replace", index=False)

        temperature, wsel = load_environment(args.covariate_csv)
        temperature.to_sql("tblInterpolatedTemp", connection, if_exists="replace", index=False)
        wsel.to_sql("tblWSEL", connection, if_exists="replace", index=False)
        study_parameters = pd.DataFrame([{
            "UTC_Conv": np.nan,
            "BM_Elev": np.nan,
            "BM_Elev_Units": "unknown",
            "Output_Units": "meters",
            "masterReceiver": None,
            "synch_time_start": None,
            "synch_time_end": None,
        }])
        study_parameters.to_sql("tblStudyParameters", connection, if_exists="replace", index=False)
        connection.commit()
    finally:
        connection.close()

    print("Created: %s" % args.output_db)
    print("Detection rows: %s" % row_count)
    print("Tags: %s" % len(tags))
    print("Receivers with GPS: %s" % len(receiver_table))
    print("Tag filter: %s" % (args.tag or "all tags"))
    print("Warnings: pulseRate and receiver signal fields remain unavailable")


if __name__ == "__main__":
    main()