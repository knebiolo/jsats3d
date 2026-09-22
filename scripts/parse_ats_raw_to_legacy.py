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
from datetime import datetime
from functools import partial
from multiprocessing import Pool
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

EPOCH = datetime(1970, 1, 1)
DT_IDX, TAG_IDX, INTERNAL_IDX = 4, 5, 0
TILT_IDX, VBATT_IDX, TEMP_IDX, PRESSURE_IDX = 6, 7, 8, 9
SIGSTR_IDX, BITPERIOD_IDX, THRESHOLD_IDX = 10, 11, 12

DETECTION_DB_COLUMNS = [
    "timeStamp", "seconds", "Rec_ID", "ReceiverType", "FirmwareVersion",
    "FileFormatVersion", "SerialNumber", "SourceFile", "SourceRow", "Internal",
    "InternalGroup1", "InternalGroup2", "InternalGroup3", "InternalFlag",
    "InternalCounter", "InternalOffset", "InternalStatus", "InternalCounterValue",
    "InternalOffsetValue", "InternalPositionDifferenceSeconds",
    "OneSecondAdjustmentEvidence", "ClockStatusMarker", "Tag_ID", "FreqOff",
    "Amplitude", "NBW", "SNR", "Valid", "Pascals", "Celsius", "TagTypeSource",
    "Event", "SigStr", "RawTemperature", "Pressure", "Tilt", "BatteryVoltage",
    "BitPeriod", "Threshold", "OffsetChanged", "CounterRestart",
    "ClockEventReasons", "GPSFixTimeStamp", "GPSFixLatitude", "GPSFixLongitude",
]


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
    parser.add_argument(
        "--include-config-beacons",
        action="store_true",
        help="Add configured local receiver-beacon tag IDs to the tag filter",
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


def _fast_number(value):
    value = value.strip()
    if value == "" or value == "N/A":
        return None
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return None


def _fast_timestamp(value):
    value = value.strip()
    try:
        return datetime.strptime(value, "%m/%d/%Y %H:%M:%S.%f")
    except ValueError:
        try:
            return datetime.strptime(value, "%m/%d/%Y %H:%M:%S")
        except ValueError:
            return None


def parse_detections(path, receiver_name, receiver_type, tags=None, start=None, end=None):
    """Return legacy detection rows as tuples in DETECTION_DB_COLUMNS order."""
    metadata = read_file_metadata(path)
    if metadata["FileFormatVersion"] != "2.0":
        raise ValueError("Unsupported File Format Version in %s: %s" % (path, metadata["FileFormatVersion"]))
    firmware = metadata["FirmwareVersion"]
    file_format = metadata["FileFormatVersion"]
    serial = metadata["SerialNumber"]
    path_str = str(path)
    keep_all = tags is None
    rows = []
    gps_count = 0
    clock_count = 0
    previous_offset = None
    gps_ts = gps_lat = gps_lon = None
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        for source_row, values in enumerate(csv.reader(stream), start=1):
            if len(values) < 13:
                continue
            raw_tag = values[TAG_IDX].strip()
            if raw_tag == "GPS Fix" or raw_tag == "GPS Clock":
                timestamp = _fast_timestamp(values[DT_IDX])
                if timestamp is None:
                    continue
                gps_lat, gps_lon = parse_gps_coordinates(values[INTERNAL_IDX])
                gps_ts = timestamp
                gps_count += 1
                continue
            if len(raw_tag) >= 7 and raw_tag[:3] == "G72":
                parsed_tag = raw_tag[3:7]
            else:
                parsed_tag = raw_tag
            if not keep_all and parsed_tag not in tags:
                continue
            timestamp = _fast_timestamp(values[DT_IDX])
            if timestamp is None:
                continue
            if start is not None and timestamp < start:
                continue
            if end is not None and timestamp > end:
                continue
            internal = parse_internal(values[INTERNAL_IDX], timestamp)
            if internal is None:
                continue
            offset = internal["InternalOffset"]
            offset_changed = previous_offset is not None and offset != previous_offset
            previous_offset = offset
            counter_restart = internal["InternalCounter"] in ("000", "001")
            reasons = []
            if offset_changed:
                reasons.append("offset_change")
            if counter_restart:
                reasons.append("counter_restart")
            if internal["ClockStatusMarker"]:
                reasons.append("status_marker")
            if internal["OneSecondAdjustmentEvidence"]:
                reasons.append("one_second_adjustment_evidence")
            event = 1 if reasons else 0
            clock_count += event
            signal = _fast_number(values[SIGSTR_IDX])
            rows.append((
                timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                (timestamp - EPOCH).total_seconds(),
                receiver_name, receiver_type, firmware, file_format, serial,
                path_str, source_row, values[INTERNAL_IDX].strip(),
                internal["InternalGroup1"], internal["InternalGroup2"],
                internal["InternalGroup3"], internal["InternalFlag"],
                internal["InternalCounter"], internal["InternalOffset"],
                internal["InternalStatus"], internal["InternalCounterValue"],
                internal["InternalOffsetValue"],
                internal["InternalPositionDifferenceSeconds"],
                1 if internal["OneSecondAdjustmentEvidence"] else 0,
                1 if internal["ClockStatusMarker"] else 0,
                parsed_tag, None, signal, None, None, 1, None, None, "raw",
                event, signal, _fast_number(values[TEMP_IDX]),
                _fast_number(values[PRESSURE_IDX]), _fast_number(values[TILT_IDX]),
                _fast_number(values[VBATT_IDX]), values[BITPERIOD_IDX].strip(),
                _fast_number(values[THRESHOLD_IDX]),
                1 if offset_changed else 0, 1 if counter_restart else 0,
                ";".join(reasons),
                gps_ts.strftime("%Y-%m-%d %H:%M:%S.%f") if gps_ts else None,
                gps_lat, gps_lon,
            ))
    return rows, gps_count, clock_count


def _parse_worker(task, tags=None, start=None, end=None):
    path_str, receiver_name, receiver_type = task
    try:
        rows, gps_count, clock_count = parse_detections(
            Path(path_str), receiver_name, receiver_type, tags, start, end
        )
        return Path(path_str).name, rows, gps_count, clock_count, None
    except Exception as error:  # loud-but-continue: report and skip the file
        return Path(path_str).name, [], 0, 0, str(error)



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
    start = datetime.fromisoformat(args.start) if args.start else None
    end = datetime.fromisoformat(args.end) if args.end else None
    tags = set(args.tags) if args.tags else set()
    if args.include_config_beacons:
        tags.update(load_beacon_registry(args.config_xlsx)["Tag_ID"].dropna().astype(str))
    tags = frozenset(tags) if tags else None

    tasks = []
    for path in files:
        serial = SERIAL_PATTERN.match(path.name).group(1)
        receiver = targets.loc[serial]
        tasks.append((str(path), receiver["Receiver Name"], receiver["Receiver Model"]))

    output = Path(args.output_db)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    connection.execute(
        "CREATE TABLE tblDetectionRaw (%s)" % ", ".join(DETECTION_DB_COLUMNS)
    )
    insert_sql = "INSERT INTO tblDetectionRaw VALUES (%s)" % ", ".join(
        ["?"] * len(DETECTION_DB_COLUMNS)
    )
    totals = {"detections": 0, "gps": 0, "clock_events": 0}
    worker = partial(_parse_worker, tags=tags, start=start, end=end)
    workers = min(8, (os.cpu_count() or 2))
    try:
        with Pool(processes=workers) as pool:
            for name, rows, gps_count, clock_count, error in pool.imap_unordered(worker, tasks):
                if error is not None:
                    print("WARNING %s: %s" % (name, error))
                    continue
                if rows:
                    connection.executemany(insert_sql, rows)
                    connection.commit()
                totals["detections"] += len(rows)
                totals["gps"] += gps_count
                totals["clock_events"] += clock_count
                print("Parsed %s: %s detections" % (name, len(rows)))
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
