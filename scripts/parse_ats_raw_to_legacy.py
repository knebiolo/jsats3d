"""Parse ATS File Format 2.0 raw files into legacy-compatible SQLite tables.

Raw inputs are read only. Internal values are preserved; this parser identifies
clock-event evidence but does not correct clock drift or jumps. Detection times are
shifted from each receiver's logging time zone to the study basis (PDT, UTC-7);
the original wall time is kept in ``RawDateTime``.
All detection additions are written into ``tblDetectionRaw``.
"""
import argparse
import csv
import os
import re
import sqlite3
from datetime import datetime, timedelta
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
        load_temperature_string,
        tag_types,
    )
except ModuleNotFoundError:
    from scripts.adapt_2025_to_legacy import (
        apply_tag_pulse_rates,
        load_beacon_registry,
        load_environment,
        load_receiver_table,
        load_temperature_string,
        tag_types,
    )

# Study time basis: 19 of 20 receivers, temperature strings, and PI WSE exports are PDT (verified 2026-09-24).
STUDY_UTC_OFFSET_HOURS = -7.0
UTC_OFFSET_PATTERN = re.compile(r"^([+-]?)(\d{2})z$")
FILE_START_PATTERN = re.compile(r"File Start:\s+\S+\s+\S+\s+([+-]?\d{2}z)")


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
    "RawDateTime", "ReceiverUTCOffsetHours", "TimeShiftHours", "TimeZoneSource",
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
        help="Add all configured beacon tag IDs (local and array-wide) to the tag filter",
    )
    parser.add_argument("--temperature-csv", help="Delivered DD_N temperature string CSV (default under 2025_Data)")
    parser.add_argument("--hobo-dir", help="Folder of DD_N HOBO exports (default: Tag Drag Period/DD_N)")
    return parser.parse_args()


def parse_utc_offset(value):
    match = UTC_OFFSET_PATTERN.match(str(value).strip())
    if not match:
        raise ValueError("Unrecognized ATS time zone offset: %r" % value)
    sign, hours = match.groups()
    return -float(hours) if sign == "-" else float(hours)


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
    metadata = {"SerialNumber": None, "FirmwareVersion": None, "FileFormatVersion": None, "UTCOffset": None}
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
            elif stripped.startswith("File Start:"):
                match = FILE_START_PATTERN.search(stripped)
                if match:
                    metadata["UTCOffset"] = parse_utc_offset(match.group(1))
    return metadata


def resolve_time_shift(header_offset, config_offset, path, gps_offset=None):
    """Return (receiver offset h, hours to add to reach study basis, evidence source).

    GPS rows are UTC, so detection-minus-GPS time is the file's true offset; this
    outranks header and config (a recovery file was found with a wrong 00z header).
    """
    if gps_offset is not None:
        for label, value in (("header", header_offset), ("config", config_offset)):
            if value is not None and value != gps_offset:
                print("WARNING %s: GPS-derived offset %+g h overrides %s offset %+g h" % (path, gps_offset, label, value))
        return gps_offset, STUDY_UTC_OFFSET_HOURS - gps_offset, "gps"
    if header_offset is not None and config_offset is not None and header_offset != config_offset:
        raise ValueError("%s: header offset %+g h disagrees with config offset %+g h and no GPS rows" % (path, header_offset, config_offset))
    offset = header_offset if header_offset is not None else config_offset
    if offset is None:
        raise ValueError("%s: no time zone offset in GPS rows, header, or config" % path)
    return offset, STUDY_UTC_OFFSET_HOURS - offset, "header" if header_offset is not None else "config"


def infer_offset_from_gps(path, max_rows=200_000, pairs_needed=25):
    """Median whole-hour offset of detection time minus the following GPS (UTC) time; None if too few pairs.

    Median over many pairs because startup GPS rows can be corrupt (e.g. "8:11:00" for 18:11:00).
    """
    last_detection = None
    offsets = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        for index, values in enumerate(csv.reader(stream)):
            if index > max_rows or len(offsets) >= pairs_needed:
                break
            if len(values) < 13:
                continue
            tag = values[TAG_IDX].strip()
            stamp = _fast_timestamp(values[DT_IDX])
            if stamp is None:
                continue
            if tag in ("GPS Fix", "GPS Clock"):
                if last_detection is not None:
                    offsets.append((last_detection - stamp).total_seconds() / 3600.0)
                    last_detection = None
            elif tag[:3] == "G72":
                last_detection = stamp
    if len(offsets) < 3:
        return None
    median = float(pd.Series(offsets).median())
    if abs(median - round(median)) > 0.25:
        raise ValueError("%s: median detection-GPS difference %.3f h is not a whole-hour offset" % (path, median))
    return float(round(median))


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


def parse_detections(path, receiver_name, receiver_type, tags=None, start=None, end=None, config_offset=None):
    """Return legacy detection rows as tuples in DETECTION_DB_COLUMNS order."""
    metadata = read_file_metadata(path)
    if metadata["FileFormatVersion"] != "2.0":
        raise ValueError("Unsupported File Format Version in %s: %s" % (path, metadata["FileFormatVersion"]))
    receiver_offset, shift_hours, offset_source = resolve_time_shift(
        metadata["UTCOffset"], config_offset, path, infer_offset_from_gps(path))
    shift = timedelta(hours=shift_hours)
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
            study_time = timestamp + shift
            if start is not None and study_time < start:
                continue
            if end is not None and study_time > end:
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
                study_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                (study_time - EPOCH).total_seconds(),
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
                # ATS GPS rows are UTC regardless of the receiver's detection time zone.
                gps_ts.strftime("%Y-%m-%d %H:%M:%S+00:00") if gps_ts else None,
                gps_lat, gps_lon,
                timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"), receiver_offset, shift_hours, offset_source,
            ))
    return rows, gps_count, clock_count


def _parse_worker(task, tags=None, start=None, end=None):
    path_str, receiver_name, receiver_type, config_offset = task
    try:
        rows, gps_count, clock_count = parse_detections(
            Path(path_str), receiver_name, receiver_type, tags, start, end, config_offset
        )
        return Path(path_str).name, rows, gps_count, clock_count, None
    except Exception as error:  # loud-but-continue: report and skip the file
        return Path(path_str).name, [], 0, 0, str(error)



def write_legacy_metadata(connection, config_path, gps_path, covariate_path, receivers, temperature_csv, hobo_dir):
    receiver_table = load_receiver_table(gps_path, config_path)
    receiver_table = receiver_table[receiver_table["Rec_ID"].isin(receivers)]
    receiver_table.to_sql("tblReceiver", connection, if_exists="replace", index=False)
    print("Created tblReceiver: %s rows" % len(receiver_table))

    beacon_registry = load_beacon_registry(config_path)
    raw_tags = pd.read_sql("select distinct Tag_ID from tblDetectionRaw", connection)
    raw_tags["TagType"] = tag_types(raw_tags["Tag_ID"], beacon_registry)
    raw_tags = apply_tag_pulse_rates(raw_tags, beacon_registry)
    raw_tags.to_sql("tblTag", connection, if_exists="replace", index=False)
    print("Created tblTag: %s rows" % len(raw_tags))
    missing_rate = raw_tags[raw_tags.pulseRate.isna()].Tag_ID.tolist()
    if missing_rate:
        print("WARNING: tags with no pulseRate (legacy epoch code will fail for them): %s" % missing_rate)

    temperature = load_temperature_string(temperature_csv, hobo_dir)
    _, wsel = load_environment(covariate_path)
    temperature.to_sql("tblInterpolatedTemp", connection, if_exists="replace", index=False)
    wsel.to_sql("tblWSEL", connection, if_exists="replace", index=False)
    print("Created tblInterpolatedTemp/tblWSEL: %s/%s rows" % (len(temperature), len(wsel)))
    print("Temperature sources: %s" % temperature.groupby("TempSource").timeStamp.agg(["min", "max", "size"]).to_dict("index"))
    first_detection = pd.read_sql("select min(timeStamp) as t from tblDetectionRaw", connection).t.iloc[0]
    for name, table in (("tblInterpolatedTemp", temperature), ("tblWSEL", wsel)):
        if first_detection is not None and str(table.timeStamp.min()) > str(first_detection):
            print("WARNING: %s starts %s, after first detection %s" % (name, table.timeStamp.min(), first_detection))
    print("WARNING: BM_Elev unresolved (NULL); receiver Z is depth below surface at deployment")
    print("WARNING: UTC_Conv left NULL pending owner confirmation; detections are on study basis UTC%+g h" % STUDY_UTC_OFFSET_HOURS)
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
        config_offset = parse_utc_offset(receiver["Receiver Time Zone Offset"])
        tasks.append((str(path), receiver["Receiver Name"], receiver["Receiver Model"], config_offset))

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
            data_root = Path(args.config_xlsx).parents[2]
            write_legacy_metadata(
                connection,
                args.config_xlsx,
                os.path.abspath(gps_path),
                covariate_path,
                set(targets["Receiver Name"]),
                args.temperature_csv or str(data_root / "Temperature" / "2025_Temp_String_Data_5min_interpolated.csv"),
                args.hobo_dir or str(data_root / "Tag Drag Period" / "DD_N"),
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
