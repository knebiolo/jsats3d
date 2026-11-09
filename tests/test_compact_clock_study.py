from pathlib import Path
from contextlib import closing
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_data as compact


class TestCompactClockStudy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.db"
        self.target = self.root / "compact.db"
        with closing(sqlite3.connect(self.source)) as con:
            con.execute(
                "CREATE TABLE tblDetectionRaw (Tag_ID TEXT,Rec_ID TEXT,seconds REAL,"
                "OffsetChanged INTEGER,CounterRestart INTEGER,OneSecondAdjustmentEvidence INTEGER,"
                "InternalOffsetValue INTEGER,SourceFile TEXT,SourceRow INTEGER)")
            con.executemany("INSERT INTO tblDetectionRaw(rowid) VALUES(?)", [(2,), (8,)])
            con.execute("UPDATE tblDetectionRaw SET Tag_ID='BC',Rec_ID='R',seconds=10 WHERE rowid=2")
            con.execute(
                "UPDATE tblDetectionRaw SET Tag_ID='OTHER',Rec_ID='R',seconds=11,"
                "OffsetChanged=1,InternalOffsetValue=2,SourceFile='raw.csv' WHERE rowid=8")
            for table in compact.COMPACT_METADATA_TABLES + ("tblInterpolatedTemp",):
                con.execute("CREATE TABLE " + table + " (value INTEGER)")
                con.execute("INSERT INTO " + table + " VALUES (1)")
            con.commit()

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return compact.build_compact(self.source, self.target, ["BC"], self.source, 1, 10**8)

    def test_preserves_rowids_metadata_and_omitted_tag_events(self):
        before = self.source.read_bytes()
        result = self.build()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(before, self.source.read_bytes())
        with closing(sqlite3.connect(self.target)) as con:
            self.assertEqual(con.execute("SELECT rowid,Tag_ID FROM tblDetectionRaw").fetchall(), [(2, "BC")])
            self.assertEqual(con.execute(
                "SELECT source_rowid,OffsetChanged FROM tblClockEventEvidence").fetchall(), [(8, 1)])
            self.assertEqual(con.execute("SELECT * FROM tblInterpolatedTemp").fetchall(), [(1,)])
        with self.assertRaises(FileExistsError):
            self.build()

    def test_refuses_capacity_before_creating_database(self):
        with self.assertRaises(RuntimeError):
            compact.build_compact(self.source, self.target, ["BC"], self.source, 10**30, 10**8)
        self.assertFalse(self.target.exists())

    def test_refuses_shifted_source(self):
        with closing(sqlite3.connect(self.source)) as con:
            con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN jump_offset_s REAL")
            con.commit()
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(self.target.exists())
