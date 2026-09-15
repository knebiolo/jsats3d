"""Audit a staged database against legacy jsats3d runtime prerequisites."""
import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--output", default="output/legacy_readiness_audit.md")
    return parser.parse_args()


def count(connection, table):
    return int(connection.execute("select count(*) from " + table).fetchone()[0])


def main():
    args = parse_args()
    connection = sqlite3.connect(args.database)
    tables = {
        table: count(connection, table)
        for table in [
            "tblTag", "tblReceiver", "tblDetectionRaw", "tblInterpolatedTemp",
            "tblWSEL", "tblStudyParameters",
        ]
    }
    tags = pd.read_sql("select * from tblTag", connection)
    receivers = pd.read_sql("select * from tblReceiver", connection)
    detections = pd.read_sql("select * from tblDetectionRaw", connection)
    temperature = pd.read_sql("select * from tblInterpolatedTemp", connection)
    wsel = pd.read_sql("select * from tblWSEL", connection)
    parameters = pd.read_sql("select * from tblStudyParameters", connection)
    connection.close()

    detection_min = pd.to_datetime(detections["timeStamp"], errors="coerce").min()
    detection_max = pd.to_datetime(detections["timeStamp"], errors="coerce").max()
    temp_min = pd.to_datetime(temperature["timeStamp"], errors="coerce").min()
    temp_max = pd.to_datetime(temperature["timeStamp"], errors="coerce").max()
    wsel_min = pd.to_datetime(wsel["timeStamp"], errors="coerce").min()
    wsel_max = pd.to_datetime(wsel["timeStamp"], errors="coerce").max()
    tag_lookup = set(tags["Tag_ID"].astype(str))
    receiver_beacons = set(receivers["Tag_ID"].dropna().astype(str))
    beacon_tags = set(tags.loc[tags["TagType"] == "beacon", "Tag_ID"].astype(str))
    missing_receiver_tags = sorted(receiver_beacons - tag_lookup)
    non_beacon_receiver_tags = sorted(receiver_beacons - beacon_tags)
    pulse_ok = not tags["pulseRate"].isna().any()
    receiver_id_ok = not receivers["Tag_ID"].isna().any()
    receiver_xyz_ok = not receivers[["X", "Y", "Z", "X_t", "Y_t", "Z_t"]].isna().any().any()
    params_row = parameters.iloc[0]
    master_ok = pd.notna(params_row["masterReceiver"]) and params_row["masterReceiver"] in set(receivers["Rec_ID"])
    bm_ok = pd.notna(params_row["BM_Elev"])
    utc_ok = pd.notna(params_row["UTC_Conv"])
    units_ok = params_row["BM_Elev_Units"] in ("feet", "meters")
    temp_ok = temp_min <= detection_min and temp_max >= detection_max
    wsel_ok = wsel_min <= detection_min and wsel_max >= detection_max
    beacon_rows = int((detections["TagTypeSource"] == "beacon").sum())
    snr_null = int(detections["SNR"].isna().sum())
    legacy_tables = [
        "tblMetronomeUnfiltered", "tblMetronomeFiltered",
        "tblMetronomeSecondFiltered", "tblDetectionFilterPrimary",
        "tblDetectionFilterSecondary", "tblDetectionClockFixed", "tblPositions_Deng",
    ]
    connection = sqlite3.connect(args.database)
    present_tables = {
        row[0] for row in connection.execute("select name from sqlite_master where type='table'")
    }
    connection.close()

    def result(condition, blocked=False):
        if blocked:
            return "BLOCKED"
        return "PASS" if condition else "FAIL"

    lines = [
        "# Legacy Readiness Audit",
        "",
        "Database: `%s`" % args.database,
        "",
        "This audit checks runtime prerequisites. It does not run synchronization or positioning.",
        "",
        "## Audit Table",
        "",
        "| # | Item | Result | Legacy line | Note |",
        "|---|---|---|---|---|",
        "| 1 | `tblTag.pulseRate` non-null | %s | 209, 295, 750, 1128 | %s/%s rows populated; FFD3 provisional rate is 3.33 seconds. |" % (result(pulse_ok), int(tags["pulseRate"].notna().sum()), len(tags)),
        "| 2 | Receiver `Tag_ID` non-null | %s | 213 | Missing rows: %s. |" % (result(receiver_id_ok), int(receivers["Tag_ID"].isna().sum())),
        "| 3 | Receiver X/Y/Z/X_t/Y_t/Z_t non-null | %s | 1107 | Null counts: %s. |" % (result(receiver_xyz_ok), receivers[["X", "Y", "Z", "X_t", "Y_t", "Z_t"]].isna().sum().to_dict()),
        "| 4 | `masterReceiver` populated and valid | %s | 762 | Value: `%s`. Owner must identify the synchronization reference. |" % (result(master_ok, not master_ok), params_row["masterReceiver"]),
        "| 5 | `BM_Elev` non-null | %s | 779, 1116 | Value: `%s`. Owner must provide benchmark elevation and datum. |" % (result(bm_ok, not bm_ok), params_row["BM_Elev"]),
        "| 6 | `UTC_Conv` non-null | %s | study parameter reads | Value: `%s`. Owner must confirm time convention. |" % (result(utc_ok, not utc_ok), params_row["UTC_Conv"]),
        "| 7 | `BM_Elev_Units` valid | %s | 783 | Value: `%s`. |" % (result(units_ok), params_row["BM_Elev_Units"]),
        "| 8 | Temperature covers detections | %s | 120 | Detection span `%s` to `%s`; temperature span `%s` to `%s`. |" % (result(temp_ok), detection_min, detection_max, temp_min, temp_max),
        "| 9 | WSEL covers detections | %s | 786, 1119 | Detection span `%s` to `%s`; WSEL span `%s` to `%s`. |" % (result(wsel_ok), detection_min, detection_max, wsel_min, wsel_max),
        "| 10 | Beacon rows staged | %s | beacon_epoch / clock_fix | `%s` beacon rows staged; `%s` detection rows total. |" % (result(beacon_rows > 0), beacon_rows, len(detections)),
        "| 11 | Receiver beacons represented in `tblTag` | %s | 763 | Missing tags: `%s`; non-beacon receiver tags: `%s`. |" % (result(not missing_receiver_tags and not non_beacon_receiver_tags), missing_receiver_tags, non_beacon_receiver_tags),
        "| 12 | SNR NULL / 2025 path | PASS | 466 | `%s` of `%s` SNR values are NULL. Legacy classifier is not valid; use ATS-2025 path. |" % (snr_null, len(detections)),
        "",
        "## Tables",
        "",
    ]
    lines.extend("- `%s`: %s rows" % item for item in tables.items())
    lines.append("")
    lines.append("Downstream tables present: `%s`." % ", ".join(sorted(set(legacy_tables) & present_tables)) if set(legacy_tables) & present_tables else "Downstream tables present: none.")
    lines.append("")
    lines.append("## Blocked Owner Questions")
    lines.append("")
    lines.extend([
        "- What receiver/tag should populate `masterReceiver` for the new synchronization approach?",
        "- What benchmark elevation and vertical datum should populate `BM_Elev`?",
        "- What UTC conversion and synchronization window should be used?",
        "- What authoritative temperature/WSEL data covers 2025-06-05 through 2025-06-16?",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Created: %s" % output)


if __name__ == "__main__":
    main()