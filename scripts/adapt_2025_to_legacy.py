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
    "event": "Event",
    "diagCode": "Internal",
    "temp": "RawTemperature",
    "pressure": "Pressure",
    "tilt": "Tilt",
    "vBatt": "BatteryVoltage",
    "bitPeriod": "BitPeriod",
    "threshold": "Threshold",
    "receiverType": "ReceiverType",
    "firmwareVersion": "FirmwareVersion",
    "fileFormatVersion": "FileFormatVersion",
    "sourceFile": "SourceFile",
    "sourceRow": "SourceRow",
}

LEGACY_DETECTION_COLUMNS = [
    "timeStamp", "seconds", "Tag_ID", "Rec_ID", "FreqOff", "Amplitude",
    "NBW", "SNR", "Valid", "Pascals", "Celsius", "TagTypeSource",
]

ATS_EXTENSION_COLUMNS = [
    "Event", "Internal", "SigStr", "RawTemperature", "Pressure", "Tilt",
    "BatteryVoltage", "BitPeriod", "Threshold", "ReceiverType",
    "FirmwareVersion", "FileFormatVersion", "SourceFile", "SourceRow",
]


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
    parser.add_argument(
        "--beacon-window",
        nargs=2,
        metavar=("START", "END"),
        help="Stage configured local beacon detections in this datetime window",
    )
    parser.add_argument(
        "--drop-incomplete-receivers",
        action="store_true",
        help="Exclude receivers missing beacon Tag_ID or complete X/Y/Z geometry",
    )
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
    config.columns = config.columns.astype(str).str.strip()
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


def load_beacon_registry(config_path):
    config = pd.read_excel(config_path)
    config.columns = config.columns.astype(str).str.strip()
    config = config.rename(columns={
        "Receiver Name": "Rec_ID",
        "Beacon Tag Code": "Tag_ID",
        "Beacon Tag Period (sec)": "pulseRate",
    })
    registry = config[["Rec_ID", "Tag_ID", "pulseRate"]].copy()
    registry["Rec_ID"] = registry["Rec_ID"].astype("string").str.strip()
    registry["Tag_ID"] = registry["Tag_ID"].astype("string").str.strip()
    registry["pulseRate"] = pd.to_numeric(registry["pulseRate"], errors="coerce")
    registry = registry.dropna(subset=["Rec_ID", "Tag_ID"])
    return registry.drop_duplicates("Tag_ID")


def parse_beacon_window(beacon_window):
    if beacon_window is None:
        return None
    start, end = pd.to_datetime(beacon_window, errors="raise")
    if end < start:
        raise ValueError("Beacon window END precedes START")
    return start, end


def drop_incomplete_receivers(receiver_table):
    required = ("Tag_ID", "X", "Y", "Z")
    incomplete = receiver_table[receiver_table[list(required)].isna().any(axis=1)].copy()
    dropped_rows = []
    for row in incomplete.itertuples(index=False):
        missing = [field for field in required if pd.isna(getattr(row, field))]
        dropped_rows.append({"Rec_ID": row.Rec_ID, "reason": "missing " + ", ".join(missing)})
    kept = receiver_table.dropna(subset=list(required)).copy()
    return kept, pd.DataFrame(dropped_rows, columns=["Rec_ID", "reason"])


def apply_tag_pulse_rates(tags, beacon_registry, ffd3_rate=3.33):
    result = tags.merge(beacon_registry[["Tag_ID", "pulseRate"]], on="Tag_ID", how="left")
    result["pulseRate"] = pd.to_numeric(result["pulseRate"], errors="coerce")
    result.loc[result["Tag_ID"] == "FFD3", "pulseRate"] = ffd3_rate
    return result


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


def normalize_detection(chunk, tag_type, tag_filter=None, beacon_window=None, beacon_tags=None):
    required_input = {"dateTime", "tagCode", "receiverName"}
    missing = required_input - set(chunk.columns)
    if missing:
        raise ValueError("Missing detection columns: %s" % sorted(missing))
    if "amp" not in chunk and "sigStr" not in chunk:
        raise ValueError("Missing detection signal column: expected amp or sigStr")
    result = chunk.rename(columns=DETECTION_COLUMNS).copy()
    if "sigStr" in result:
        result["SigStr"] = pd.to_numeric(result["sigStr"], errors="coerce")
        if "Amplitude" not in result:
            result["Amplitude"] = result["SigStr"]
    if tag_filter is not None:
        result = result[result["Tag_ID"].astype(str).str.strip() == tag_filter]
    if beacon_tags is not None:
        result = result[result["Tag_ID"].astype(str).str.strip().isin(beacon_tags)]
    if beacon_window is not None:
        start, end = beacon_window
        parsed = pd.to_datetime(result["timeStamp"], errors="coerce")
        result = result[(parsed >= start) & (parsed <= end)]
    if result.empty:
        return pd.DataFrame(columns=LEGACY_DETECTION_COLUMNS + ATS_EXTENSION_COLUMNS)
    result["timeStamp"] = pd.to_datetime(result["timeStamp"], errors="coerce")
    result = result.dropna(subset=["timeStamp", "Tag_ID", "Rec_ID"])
    result["seconds"] = result["timeStamp"].astype("datetime64[ns]").astype("int64") / 1e9
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
    for column in ATS_EXTENSION_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    return result[LEGACY_DETECTION_COLUMNS + ATS_EXTENSION_COLUMNS]


def write_detection_tables(
    connection,
    detection_dir,
    chunksize,
    tag_filter=None,
    filenames=None,
    beacon_window=None,
    beacon_tags=None,
):
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
            file_tag_filter = None if filename == "master_df_beacon.csv" else tag_filter
            file_beacon_tags = beacon_tags if filename == "master_df_beacon.csv" else None
            file_beacon_window = beacon_window if filename == "master_df_beacon.csv" else None
            normalized = normalize_detection(
                chunk,
                tag_type,
                file_tag_filter,
                file_beacon_window,
                file_beacon_tags,
            )
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
    beacon_window = parse_beacon_window(args.beacon_window)
    beacon_registry = load_beacon_registry(args.config_xlsx)
    local_beacons = set(beacon_registry["Tag_ID"])
    if beacon_window is not None and "master_df_beacon.csv" not in args.files:
        args.files = list(args.files) + ["master_df_beacon.csv"]
    gps_path = os.path.join(
        os.path.dirname(args.detection_dir), "4_gps_datasets", "master_df_gps.csv"
    )
    if os.path.exists(args.output_db):
        os.remove(args.output_db)
    os.makedirs(os.path.dirname(args.output_db), exist_ok=True)
    connection = sqlite3.connect(args.output_db)
    try:
        tags, detected_receivers, row_count = write_detection_tables(
            connection,
            args.detection_dir,
            args.chunksize,
            args.tag,
            args.files,
            beacon_window,
            local_beacons if beacon_window is not None else None,
        )
        receiver_table = load_receiver_table(gps_path, args.config_xlsx)
        receiver_table = receiver_table[receiver_table.Rec_ID.isin(detected_receivers)]
        dropped_receivers = pd.DataFrame(columns=["Rec_ID", "reason"])
        if args.drop_incomplete_receivers:
            receiver_table, dropped_receivers = drop_incomplete_receivers(receiver_table)
        receiver_table.to_sql("tblReceiver", connection, if_exists="replace", index=False)

        tags = tags.rename(columns={"TagTypeSource": "TagType"})
        tags = apply_tag_pulse_rates(tags, beacon_registry)
        tags.to_sql("tblTag", connection, if_exists="replace", index=False)

        temperature, wsel = load_environment(args.covariate_csv)
        temperature.to_sql("tblInterpolatedTemp", connection, if_exists="replace", index=False)
        wsel.to_sql("tblWSEL", connection, if_exists="replace", index=False)
        study_parameters = pd.DataFrame([{
            "UTC_Conv": np.nan,
            "BM_Elev": np.nan,
            "BM_Elev_Units": "feet",
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
    print("Receivers staged: %s" % len(receiver_table))
    print("Tag filter: %s" % (args.tag or "all tags"))
    print("Dropped receivers: %s" % len(dropped_receivers))
    if not dropped_receivers.empty:
        print(dropped_receivers.to_string(index=False))
    print("Warnings: UTC_Conv, BM_Elev, masterReceiver, sync window, and unavailable signal fields remain unresolved")


if __name__ == "__main__":
    main()