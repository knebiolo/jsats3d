"""Create a read-only readiness report for epoch-dependent receiver geometry."""
import argparse
import csv
import sqlite3


REQUIRED_FIELDS = [
    "movement_class",
    "reference_time",
    "reference_wsel",
    "hydrophone_offset_from_surface",
    "spatial_offset_perpendicular_to_surface",
    "deployment_depth",
    "geometry_source",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("output")
    return parser.parse_args()


def main():
    args = parse_args()
    connection = sqlite3.connect(args.database)
    rows = connection.execute(
        "select Rec_ID, Tag_ID, Ref_Elev, X, Y, Z, X_t, Y_t, Z_t "
        "from tblReceiver order by Rec_ID"
    ).fetchall()
    connection.close()

    columns = [
        "Rec_ID", "Tag_ID", "Ref_Elev", "X", "Y", "Z", "X_t", "Y_t", "Z_t",
        *REQUIRED_FIELDS, "status",
    ]
    with open(args.output, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([
                *row,
                *([""] * len(REQUIRED_FIELDS)),
                "blocked_pending_owner_geometry_inputs",
            ])
    print("Created: %s" % args.output)
    print("Receivers: %s" % len(rows))
    print("Status: blocked_pending_owner_geometry_inputs")


if __name__ == "__main__":
    main()