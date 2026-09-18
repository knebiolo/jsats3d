"""Parse ATS File Format 2.0 raw files into legacy-compatible SQLite tables.

Raw inputs are read only. Original timestamps and Internal values are preserved;
this parser identifies clock-event evidence but does not correct timestamps.
All detection additions are written into ``tblDetectionRaw``.
"""
import argparse
import csv
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd

try:
    from adapt_2025_to_legacy import (
        apply_tag_pulse_rates,
        load_beacon_registry,
        load_environment,
        load_receiver_table,
    )
except ModuleNotFoundError:
    from scripts.adapt_2025_to_legacy import (
        apply_tag_pulse_rates,
        load_beacon_registry,
        load_environment,
        load_receiver_table,
    )


RAW_COLUMNS = [
    "Internal", "SiteName1", "SiteName2", "SiteName3", "DateTime",
    "TagCode", "Tilt", "VBatt", "Temp", "Pressure", "SigStr",
    "BitPeriod", "Threshold", "Blank",
]
INTERNAL_PATTERN = re.compile(
    r"^(\S{6}) (\S{4}) (\S{2}) (\S)(\S{3}) (\S{3}) (\S)$"
)
SERIAL_PATTERN = re.compile(r"^SR(\d+)(?=_|\.)", re.IGNORECASE)
CORRECTED_SUFFIXES = ("_cleaned", "_recovered", "_recovery")
STATUS_MARKERS = {
    "GPS111", "RTC222", "001111", "006600", "007700", "0000SL",
    "999999", "636363",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root")
    parser.add_argument("--config-xlsx", required=True)
    parser.add_argument("--output-db", required=True)
    parser.add_argument("--legacy-db", action="store_true", help="Add legacy metadata tables")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--serial",
        action="append",
        dest="serials",
        help="Restrict parsing to one or more receiver serial numbers",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Keep one or more four-character tag IDs",
    )
    return parser.parse_args()


def load_target_receivers(config_path):
    config = pd.read_excel(config_path)
    config.columns = config.columns.astype(str).str.strip()
    names = config["Receiver Name"].astype("string").str.strip()
    target = config[names.str.match(r"^(ZOI0[1-9]|ZOI1[01]|CFD0[1-9])$", na=False)].copy()
    target["Receiver Name"] = target["Receiver Name"].astype(str).str.strip()
    target["Receiver Serial Number"] = (
        pd.to_numeric(target["Receiver Serial Number"], errors="coerce")
        .astype("Int64").astype("string")
    )
    target = target.dropna(subset=["Receiver Serial Number"])
    return target.drop_duplicates("Receiver Serial Number").set_index("Receiver Serial Number")


def corrected_priority(path):
    stem = path.stem.lower()
    if stem.endswith("_cleaned"):
        return 3
    if stem.endswith("_recovered") or stem.endswith("_recovery"):
        return 2
    return 1


def canonical_file_key(path):
    stem = path.stem
    for suffix in CORRECTED_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return str(path.parent).lower(), stem.lower()


def discover_target_files(raw_root, target_serials):
    selected = {}
    for path in Path(raw_root).rglob("*.csv"):
        match = SERIAL_PATTERN.match(path.name)
        if not match or match.group(1) not in target_serials:
            continue
        key = canonical_file_key(path)
        current = selected.get(key)
        if current is None or corrected_priority(path) > corrected_priority(current):
            selected[key] = path
    return sorted(selected.values())


def read_file_metadata(path):
    metadata = {"SerialNumber": None, "FirmwareVersion": None, "FileFormatVersion": None}
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for _ in range(12):
            line = stream.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped.startswith("Serial Number:"):
                metadata["SerialNumber"] = stripped.split(":", 1)[1].split(",", 1)[0].strip()
            elif "Firmware" in stripped:
                metadata["FirmwareVersion"] = stripped.rsplit("Firmware", 1)[1].split(",", 1)[0].strip().lstrip("v")
            elif stripped.startswith("File Format Version:"):
                metadata["FileFormatVersion"] = stripped.split(":", 1)[1].split(",", 1)[0].strip()
    return metadata


def parse_number(value):
    value = value.strip()
    if value in ("", "N/A"):
        return None
    return pd.to_numeric(value, errors="coerce")


def parse_tag_code(value):
    value = value.strip()
    if value in ("GPS Fix", "GPS Clock"):
        return value
    if value.startswith("G72") and len(value) >= 7:
        return value[3:7]
    return value


def parse_internal(value, timestamp):
    value = value.strip()
    match = INTERNAL_PATTERN.match(value)
    if not match:
        return None
    group1, group2, group3, flag, counter, offset, status = match.groups()
    group2_seconds = int(group2, 16) / 100.0
    timestamp_position = timestamp.second % 15 + timestamp.microsecond / 1e6
    position_difference = round(timestamp_position - group2_seconds, 2)
    return {
        "InternalGroup1": group1,
        "InternalGroup2": group2,
        "InternalGroup3": group3,
        "InternalFlag": flag,
        "InternalCounter": counter,
        "InternalOffset": offset,
        "InternalStatus": status,
        "InternalCounterValue": int(counter, 16),
        "InternalOffsetValue": int(offset, 16),
        "InternalPositionDifferenceSeconds": position_difference,
        "OneSecondAdjustmentEvidence": abs(position_difference) == 1.0 or flag in ("1", ">"),
        "ClockStatusMarker": group1 in STATUS_MARKERS or group1.startswith("H"),
    }


def parse_gps_coordinates(value):
    match = re.match(r"^(\d{2})(\d{2}\.\d+)\s+([NS])\s+(\d{3})(\d{2}\.\d+)\s+([EW])$", value.strip())
    if not match:
        return None, None
    lat_deg, lat_min, lat_hemi, lon_deg, lon_min, lon_hemi = match.groups()
    latitude = int(lat_deg) + float(lat_min) / 60.0
    longitude = int(lon_deg) + float(lon_min) / 60.0
    if lat_hemi == "S":
        latitude *= -1
    if lon_hemi == "W":
        longitude *= -1
    return latitude, longitude


def parse_raw_file(path, receiver_name, receiver_type, start=None, end=None, tags=None):
    metadata = read_file_metadata(path)
    if metadata["FileFormatVersion"] != "2.0":
        raise ValueError("Unsupported File Format Version in %s: %s" % (path, metadata["FileFormatVersion"]))
    detections = []
    gps_rows = []
    last_gps = {"timeStamp": None, "Latitude": None, "Longitude": None}
    previous_offset = None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        for source_row, values in enumerate(csv.reader(stream), start=1):
            if len(values) < 13:
                continue
            values = values[:14] + [""] * max(0, 14 - len(values))
            row = dict(zip(RAW_COLUMNS, values))
            timestamp = pd.to_datetime(row["DateTime"].strip(), errors="coerce")
            if pd.isna(timestamp):
                continue
            if start is not None and timestamp < start:
                continue
            if end is not None and timestamp > end:
                continue
            raw_tag = row["TagCode"].strip()
            common = {
                "timeStamp": timestamp,
                "seconds": timestamp.value / 1e9,
                "Rec_ID": receiver_name,
                "ReceiverType": receiver_type,
                "FirmwareVersion": metadata["FirmwareVersion"],
                "FileFormatVersion": metadata["FileFormatVersion"],
                "SerialNumber": metadata["SerialNumber"],
                "SourceFile": str(path),
                "SourceRow": source_row,
                "Internal": row["Internal"].strip(),
            }
            if raw_tag in ("GPS Fix", "GPS Clock"):
                latitude, longitude = parse_gps_coordinates(row["Internal"])
                last_gps = {"timeStamp": timestamp, "Latitude": latitude, "Longitude": longitude}
                gps_rows.append({**common, "RecordType": raw_tag, "Latitude": latitude, "Longitude": longitude})
                continue
            parsed_tag = parse_tag_code(raw_tag)
            if tags is not None and parsed_tag not in tags:
                continue
            internal = parse_internal(row["Internal"], timestamp)
            if internal is None:
                continue
            offset_changed = previous_offset is not None and internal["InternalOffset"] != previous_offset
            previous_offset = internal["InternalOffset"]
            counter_restart = internal["InternalCounter"] in ("000", "001")
            event_reasons = []
            if offset_changed:
                event_reasons.append("offset_change")
            if counter_restart:
                event_reasons.append("counter_restart")
            if internal["ClockStatusMarker"]:
                event_reasons.append("status_marker")
            if internal["OneSecondAdjustmentEvidence"]:
                event_reasons.append("one_second_adjustment_evidence")
            detection = {
                **common,
                **internal,
                "Tag_ID": parsed_tag,
                "FreqOff": None,
                "Amplitude": parse_number(row["SigStr"]),
                "NBW": None,
                "SNR": None,
                "Valid": True,
                "Pascals": None,
                "Celsius": None,
                "TagTypeSource": "raw",
                "Event": bool(event_reasons),
                "SigStr": parse_number(row["SigStr"]),
                "RawTemperature": parse_number(row["Temp"]),
                "Pressure": parse_number(row["Pressure"]),
                "Tilt": parse_number(row["Tilt"]),
                "BatteryVoltage": parse_number(row["VBatt"]),
                "BitPeriod": row["BitPeriod"].strip(),
                "Threshold": parse_number(row["Threshold"]),
                "OffsetChanged": offset_changed,
                "CounterRestart": counter_restart,
                "ClockEventReasons": ";".join(event_reasons),
                "GPSFixTimeStamp": last_gps["timeStamp"],
                "GPSFixLatitude": last_gps["Latitude"],
                "GPSFixLongitude": last_gps["Longitude"],
            }
            detections.append(detection)
    return pd.DataFrame(detections), pd.DataFrame(gps_rows)


def append_frame(connection, table, frame):
    if not frame.empty:
        frame.to_sql(table, connection, if_exists="append", index=False)


def write_legacy_metadata(connection, config_path, gps_path, covariate_path, receivers):
    receiver_table = load_receiver_table(gps_path, config_path)
    receiver_table = receiver_table[receiver_table["Rec_ID"].isin(receivers)]
    receiver_table.to_sql("tblReceiver", connection, if_exists="replace", index=False)
    print("Created tblReceiver: %s rows" % len(receiver_table))

    raw_tags = pd.read_sql("select distinct Tag_ID, TagTypeSource from tblDetectionRaw", connection)
    raw_tags = raw_tags.rename(columns={"TagTypeSource": "TagType"})
    beacon_registry = load_beacon_registry(config_path)
    raw_tags = apply_tag_pulse_rates(raw_tags, beacon_registry)
    raw_tags.to_sql("tblTag", connection, if_exists="replace", index=False)
    print("Created tblTag: %s rows" % len(raw_tags))

    temperature, wsel = load_environment(covariate_path)
    temperature.to_sql("tblInterpolatedTemp", connection, if_exists="replace", index=False)
    wsel.to_sql("tblWSEL", connection, if_exists="replace", index=False)
    print("Created tblInterpolatedTemp/tblWSEL: %s/%s rows" % (len(temperature), len(wsel)))
    pd.DataFrame([{
        "UTC_Conv": None,
        "BM_Elev": None,
        "BM_Elev_Units": "feet",
        "Output_Units": "meters",
        "masterReceiver": None,
        "synch_time_start": None,
        "synch_time_end": None,
    }]).to_sql("tblStudyParameters", connection, if_exists="replace", index=False)
    print("Created tblStudyParameters")


def main():
    args = parse_args()
    targets = load_target_receivers(args.config_xlsx)
    if args.serials:
        requested = set(args.serials)
        unknown = sorted(requested - set(targets.index))
        if unknown:
            raise ValueError("Requested serials are not target receivers: %s" % unknown)
        targets = targets.loc[sorted(requested)]
    files = discover_target_files(args.raw_root, set(targets.index))
    if args.max_files is not None:
        files = files[:args.max_files]
    if not files:
        raise ValueError("No target receiver files found")
    found_serials = {SERIAL_PATTERN.match(path.name).group(1) for path in files}
    missing_serials = sorted(set(targets.index) - found_serials)
    start = pd.to_datetime(args.start, errors="raise") if args.start else None
    end = pd.to_datetime(args.end, errors="raise") if args.end else None
    tags = set(args.tags) if args.tags else None
    output = Path(args.output_db)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    totals = {"detections": 0, "gps": 0, "clock_events": 0}
    try:
        for path in files:
            serial = SERIAL_PATTERN.match(path.name).group(1)
            receiver = targets.loc[serial]
            detections, gps_rows = parse_raw_file(
                path,
                receiver["Receiver Name"],
                receiver["Receiver Model"],
                start,
                end,
                tags,
            )
            append_frame(connection, "tblDetectionRaw", detections)
            totals["detections"] += len(detections)
            totals["gps"] += len(gps_rows)
            totals["clock_events"] += int(detections["Event"].sum()) if not detections.empty else 0
            print("Parsed %s: %s detections" % (path.name, len(detections)))
        if args.legacy_db:
            gps_path = os.path.join(
                os.path.dirname(args.config_xlsx), "..", "4_gps_datasets", "master_df_gps.csv"
            )
            covariate_path = os.path.join(
                Path(args.config_xlsx).parents[2], "Master Covariate Table",
                "2025 Master Covariate Table_20251212.csv",
            )
            write_legacy_metadata(
                connection,
                args.config_xlsx,
                os.path.abspath(gps_path),
                covariate_path,
                set(targets["Receiver Name"]),
            )
        connection.commit()
    finally:
        connection.close()
    print("Created: %s" % output)
    print("Files: %s" % len(files))
    print("Target serials found: %s/%s" % (len(found_serials), len(targets)))
    print("Missing target serials: %s" % missing_serials)
    print("Detections: %s" % totals["detections"])
    print("GPS rows: %s" % totals["gps"])
    print("Clock-event rows: %s" % totals["clock_events"])


if __name__ == "__main__":
    main()
