"""Create a read-only clock/TDOA input readiness report."""
import argparse
import csv
import sqlite3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("output")
    return parser.parse_args()


def main():
    args = parse_args()
    connection = sqlite3.connect(args.database)
    receivers = connection.execute(
        "select Rec_ID, Tag_ID from tblReceiver order by Rec_ID"
    ).fetchall()
    receiver_stats = {
        row[0]: row[1:]
        for row in connection.execute(
            "select Rec_ID, count(*), "
            "sum(case when OffsetChanged = 1 then 1 else 0 end), "
            "sum(case when CounterRestart = 1 then 1 else 0 end), "
            "sum(case when OneSecondAdjustmentEvidence = 1 then 1 else 0 end), "
            "sum(case when ClockStatusMarker is not null and ClockStatusMarker != '' then 1 else 0 end), "
            "min(seconds), max(seconds) from tblDetectionRaw group by Rec_ID"
        )
    }
    tag_receiver_counts = {
        (row[0], row[1]): row[2]
        for row in connection.execute(
            "select Tag_ID, Rec_ID, count(*) from tblDetectionRaw "
            "group by Tag_ID, Rec_ID"
        )
    }
    columns = [
        "Rec_ID", "host_beacon_tag", "host_beacon_rows", "other_receiver_rows",
        "event_rows", "offset_changes", "counter_restarts",
        "one_second_evidence", "status_markers", "first_seconds", "last_seconds",
        "status",
    ]
    with open(args.output, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        for receiver_id, beacon_tag in receivers:
            host_rows = tag_receiver_counts.get((beacon_tag, receiver_id), 0)
            beacon_total = sum(
                count for (tag_id, _), count in tag_receiver_counts.items()
                if tag_id == beacon_tag
            )
            other_rows = beacon_total - host_rows
            event_row = receiver_stats[receiver_id][:5]
            span = receiver_stats[receiver_id][5:]
            writer.writerow([
                receiver_id,
                beacon_tag,
                host_rows,
                other_rows,
                *[value or 0 for value in event_row],
                *span,
                "staged_raw_events_pending_tdoa_and_owner_sync_parameters",
            ])
    connection.close()
    print("Created: %s" % args.output)
    print("Receiver rows: %s" % len(receivers))
    print("Status: staged_raw_events_pending_tdoa_and_owner_sync_parameters")


if __name__ == "__main__":
    main()