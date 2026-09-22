"""Read-only data-quality audit of the final jsats3d database."""
import argparse
import sqlite3
from datetime import date, timedelta


def one(cur, sql):
    return cur.execute(sql).fetchone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    args = parser.parse_args()
    c = sqlite3.connect(args.database)

    print("=== TABLE ROW COUNTS ===")
    for t in ["tblDetectionRaw", "tblReceiver", "tblTag", "tblInterpolatedTemp", "tblWSEL", "tblStudyParameters"]:
        print(f"  {t}: {one(c, 'select count(*) from ' + t)[0]:,}")

    print("\n=== tblDetectionRaw NULL / BAD VALUES ===")
    total = one(c, "select count(*) from tblDetectionRaw")[0]
    for col in ["timeStamp", "seconds", "Tag_ID", "Rec_ID", "SigStr", "Amplitude", "Internal", "GPSFixLatitude"]:
        n = one(c, f"select count(*) from tblDetectionRaw where {col} is null")[0]
        print(f"  NULL {col}: {n:,}")
    neg = one(c, "select count(*) from tblDetectionRaw where SigStr < 0")[0]
    print(f"  SigStr < 0 (e.g. -99 no-signal sentinel): {neg:,}")
    sentinel_t = one(c, "select count(*) from tblDetectionRaw where RawTemperature = 99.99")[0]
    print(f"  RawTemperature = 99.99 (receiver no-temp sentinel): {sentinel_t:,} of {total:,}")
    dupes = one(c, """select count(*) from (
        select Rec_ID, Tag_ID, timeStamp, SourceFile, SourceRow, count(*) k
        from tblDetectionRaw group by Rec_ID, Tag_ID, timeStamp, SourceFile, SourceRow having k > 1)""")[0]
    print(f"  duplicate (Rec_ID,Tag_ID,timeStamp,SourceFile,SourceRow) groups: {dupes:,}")
    outlo = one(c, "select count(*) from tblDetectionRaw where timeStamp < '2025-06-01'")[0]
    outhi = one(c, "select count(*) from tblDetectionRaw where timeStamp > '2025-09-30'")[0]
    print(f"  timestamps before 2025-06-01: {outlo:,}; after 2025-09-30: {outhi:,}")

    print("\n=== DETECTIONS PER RECEIVER ===")
    for rec, n, lo, hi in c.execute("""select Rec_ID, count(*), min(timeStamp), max(timeStamp)
        from tblDetectionRaw group by Rec_ID order by Rec_ID"""):
        print(f"  {rec}: {n:,}  [{lo[:10]} .. {hi[:10]}]")

    print("\n=== DETECTIONS PER TAG ===")
    for tag, n in c.execute("select Tag_ID, count(*) from tblDetectionRaw group by Tag_ID order by Tag_ID"):
        print(f"  {tag}: {n:,}")

    print("\n=== DAILY COVERAGE GAPS (days in span with zero detections) ===")
    lo, hi = one(c, "select min(timeStamp), max(timeStamp) from tblDetectionRaw")
    days = {r[0] for r in c.execute("select distinct substr(timeStamp,1,10) from tblDetectionRaw")}
    d0 = date.fromisoformat(lo[:10])
    d1 = date.fromisoformat(hi[:10])
    missing = []
    d = d0
    while d <= d1:
        if d.isoformat() not in days:
            missing.append(d.isoformat())
        d += timedelta(days=1)
    print(f"  span {d0} .. {d1} ({(d1 - d0).days + 1} days); days with detections: {len(days)}; missing: {len(missing)}")
    if missing:
        print("  missing days:", ", ".join(missing))

    print("\n=== RECEIVER GEOMETRY ===")
    for col in ["Tag_ID", "X", "Y", "Z"]:
        n = one(c, f"select count(*) from tblReceiver where {col} is null")[0]
        print(f"  NULL {col}: {n}")

    print("\n=== tblTag ===")
    for tag, tt, pr in c.execute("select Tag_ID, TagType, pulseRate from tblTag order by Tag_ID"):
        print(f"  {tag}: type={tt} pulseRate={pr}")

    print("\n=== ENVIRONMENTAL COVERAGE vs DETECTIONS ===")
    t_lo, t_hi = one(c, "select min(timeStamp), max(timeStamp) from tblInterpolatedTemp")
    w_lo, w_hi = one(c, "select min(timeStamp), max(timeStamp) from tblWSEL")
    print(f"  detections span: {lo} .. {hi}")
    print(f"  temperature span: {t_lo} .. {t_hi}")
    print(f"  WSEL span:        {w_lo} .. {w_hi}")
    before_temp = one(c, f"select count(*) from tblDetectionRaw where timeStamp < '{t_lo}'")[0]
    print(f"  detections BEFORE temperature coverage starts: {before_temp:,} of {total:,}")
    dd = {r[0] for r in c.execute(f"select distinct substr(timeStamp,1,10) from tblDetectionRaw where timeStamp < '{t_lo}'")}
    print(f"  uncovered detection days: {len(dd)} -> {', '.join(sorted(dd))}")

    print("\n=== tblStudyParameters ===")
    row = one(c, "select UTC_Conv,BM_Elev,BM_Elev_Units,masterReceiver,synch_time_start,synch_time_end from tblStudyParameters")
    print("  UTC_Conv=%s BM_Elev=%s Units=%s masterReceiver=%s synch=%s..%s" % row)
    c.close()


if __name__ == "__main__":
    main()
