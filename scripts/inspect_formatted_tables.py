"""Print representative rows from formatted SQLite tables."""
import argparse
import sqlite3

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    tables = pd.read_sql(
        "select name from sqlite_master where type='table' order by name",
        connection,
    )["name"].tolist()
    print("TABLE COUNTS")
    for table in tables:
        count = connection.execute("select count(*) from " + table).fetchone()[0]
        print("%-28s %s" % (table, count))
    for table in tables:
        print("\n%s COLUMNS" % table)
        columns = [row[1] for row in connection.execute("pragma table_info(%s)" % table)]
        print(", ".join(columns))
        print("%s SAMPLE" % table)
        print(pd.read_sql("select * from %s limit 3" % table, connection).to_string(index=False))
    connection.close()


if __name__ == "__main__":
    main()
