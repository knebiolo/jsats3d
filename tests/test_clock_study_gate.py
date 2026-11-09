from contextlib import closing
import json
from pathlib import Path
import sqlite3
import csv
import pandas as pd
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _2025_pipeline import (
    review_gate, clock_review_run, clock_review_done, clock_source_boundary_review, clock_rapid_reference_review,
    clock_source_median_review,
)


class TestClockStudyGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.db"
        self.output = Path(self.tmp.name) / "review"
        with closing(sqlite3.connect(self.db)) as con:
            con.executescript("""
                CREATE TABLE tblDetectionRaw (Rec_ID TEXT,Tag_ID TEXT,seconds REAL,jump_offset_s REAL);
                INSERT INTO tblDetectionRaw VALUES ('R','F',90,2);
                CREATE TABLE tblStudyParameters (masterReceiver TEXT);
                INSERT INTO tblStudyParameters VALUES ('M');
                CREATE TABLE tblReceiver (Rec_ID TEXT,Tag_ID TEXT);
                INSERT INTO tblReceiver VALUES ('M','BC');
                CREATE TABLE tblTag (Tag_ID TEXT,pulseRate REAL);
                INSERT INTO tblTag VALUES ('BC',60);
                CREATE TABLE tblClockJumpCandidates
                    (Rec_ID TEXT,status TEXT,window_start_s REAL,window_end_s REAL,note TEXT);
                CREATE TABLE tblDetectionClockFixed (Rec_ID TEXT,seconds REAL,seconds_residual REAL);
                INSERT INTO tblDetectionClockFixed VALUES ('R',0,0),('R',60,0),('R',240,0);
            """)

    def tearDown(self):
        self.tmp.cleanup()

    def gate(self, curves):
        return review_gate(self.db, ["F"], ["R"], 0, 300, self.output, check_curves=curves)

    def test_unresolved_event_uses_original_not_corrected_time(self):
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("INSERT INTO tblClockJumpCandidates VALUES ('R','unresolved',91,93,'review')")
            con.commit()
        with self.assertRaisesRegex(RuntimeError, "review required"):
            self.gate(False)
        summary = json.loads((self.output / "clock_review_gate.json").read_text())
        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(summary["receivers_with_fish_impact"], ["R"])

    def test_curve_gap_blocks_without_changing_raw(self):
        before = self.db.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "review required"):
            self.gate(True)
        self.assertEqual(self.db.read_bytes(), before)
        self.assertTrue((self.output / "clock_review_gate.csv").is_file())

    def test_supported_fish_passes(self):
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("UPDATE tblDetectionRaw SET seconds=30")
            con.commit()
        self.assertEqual(self.gate(True)["status"], "passed")

    def test_missing_candidate_ledger_does_not_pass(self):
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("DROP TABLE tblClockJumpCandidates")
        with self.assertRaisesRegex(RuntimeError, "ledger is missing"):
            self.gate(False)

    def test_integrated_reference_review_is_readonly_and_disabled_by_default(self):
        self.assertTrue(clock_review_done(SimpleNamespace(clock_review_interval=None)))
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("ALTER TABLE tblDetectionClockFixed ADD COLUMN Tag_ID TEXT")
            con.execute("UPDATE tblDetectionClockFixed SET Tag_ID='BC'")
            for t in (0, 60, 120):
                con.execute("INSERT INTO tblDetectionRaw VALUES ('M','BC',?,0)", (t,))
                con.execute("INSERT INTO tblDetectionRaw VALUES ('R','BC',?,0)", (t+0.01,))
            con.commit()
        self.db.with_suffix(".compact.json").write_text(json.dumps({"source": str(self.db)}))
        context = SimpleNamespace(
            db=self.db, clock_review_interval=["1970-01-01 00:00:00", "1970-01-01 00:03:00"],
            window_start="1970-01-01 00:00:00", window_end="1970-01-01 00:03:00",
            compare_clock_db=None, deep=[], surface=["M", "R"],
            run={"study": {"master_receiver": "M"}, "legacy": {}})
        before = self.db.read_bytes()
        summary = clock_review_run(context)
        self.assertEqual(summary["status"], "diagnostic_only")
        self.assertEqual(summary["paired_rows"], 3)
        self.assertEqual(self.db.read_bytes(), before)
        self.assertTrue((Path(summary["folder"]) / "multi_beacon_relative_timing.csv").is_file())

    def test_source_boundary_includes_tag_omitted_from_database(self):
        source = Path(self.tmp.name) / "receiver.csv"
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["0000SL 0578 03 A063 100 A", "ZOI", "08", "08",
                             "06/28/2025 01:04:59.000000", "G72FA1B63"] + [""]*7)
            writer.writerow(["0000SL 0045 05 y002 104 A", "ZOI", "08", "08",
                             "06/28/2025 01:05:00.697801", "G72366423"] + [""]*7)
            writer.writerow(["0000SL 0143 05 y006 104 A", "ZOI", "08", "08",
                             "06/28/2025 01:05:03.238021", "G7226E443"] + [""]*7)
            writer.writerow(["0000SL 0143 05 y000 104 A", "ZOI", "08", "08",
                             "06/28/2025 01:05:03.238021", "G72366423"] + [""]*7)
            writer.writerow(["0000SL 0255 05 y001 104 A", "ZOI", "08", "08",
                             "06/28/2025 01:05:06.000000", "G72366423"] + [""]*7)
            for internal in ("0000SL 0255 05 y000 104 A", "0000SL 0255 05 y001 104 F",
                             "0000SL 0255 05 y000 104 A", "0000SL 0255 05 y001 108 A",
                             "0000SL 0255 05 y000 108 A"):
                writer.writerow([internal, "ZOI", "08", "08",
                                 "06/28/2025 01:05:07.000000", "G72366423"] + [""]*7)
            writer.writerow(["", "ZOI", "08", "08", "06/28/2025 08:05:10", "GPS Fix"] + [""]*7)
            writer.writerow(["0000SL 0255 05 y001 108 A", "ZOI", "08", "08",
                             "06/28/2025 01:05:08.000000", "G72366423"] + [""]*7)
            writer.writerow(["0000SL 0255 05 y000 108 A", "ZOI", "08", "08",
                             "06/28/2025 01:05:09.000000", "G72366423"] + [""]*7)
            writer.writerow(["unparseable", "ZOI", "08", "08",
                             "06/28/2025 01:05:09.500000", "G72366423"] + [""]*7)
            writer.writerow(["0000SL 0255 05 y001 108 A", "ZOI", "08", "08",
                             "06/28/2025 01:05:10.000000", "G72366423"] + [""]*7)
        start = pd.Timestamp("2025-06-28 01:05:00", tz="UTC").timestamp()
        self.output.mkdir()
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN SourceFile TEXT")
            con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN TimeShiftHours REAL")
            con.execute("UPDATE tblDetectionRaw SET Rec_ID='M',seconds=?,SourceFile=?,TimeShiftHours=0",
                        (start+3.238021, str(source)))
            con.commit()
            summary = clock_source_boundary_review(con, "M", start, start+60, self.output)
        self.assertEqual(summary["status"], "reviewed")
        report = pd.read_csv(self.output / "M_source_state_boundaries.csv")
        transition = report[report.InternalOffsetValue == 260].iloc[0]
        self.assertEqual(transition.Tag_ID, "3664")
        self.assertEqual(transition.SourceRow, 2)
        self.assertAlmostEqual(transition.seconds, start+0.697801, places=5)
        restart = report[report.SourceRow == 4].iloc[0]
        self.assertEqual(restart.CounterRestart, 1)
        self.assertEqual(restart.OffsetChanged, 0)
        continuation = report[report.SourceRow == 5].iloc[0]
        self.assertEqual(continuation.CounterRestart, 1)
        self.assertEqual(continuation.CounterContinuation, 1)
        across_gps = report[report.SourceRow == 12].iloc[0]
        self.assertEqual(across_gps.CounterContinuation, 1)
        self.assertEqual(across_gps.ContinuationAcrossGPS, 1)
        for source_row in (7, 9, 15):
            separate = report[report.SourceRow == source_row].iloc[0]
            self.assertEqual(separate.CounterRestart, 1)
            self.assertEqual(separate.CounterContinuation, 0)

    def test_source_median_review_is_readonly_and_keeps_unresolved_measurements(self):
        self.output.mkdir()
        pd.DataFrame({
            "Rec_ID": ["R"], "SourceFile": ["receiver.csv"], "SourceRow": [10],
            "seconds": [100.], "OffsetChanged": [1], "CounterRestart": [0],
            "CounterContinuation": [0], "InternalOffsetValue": [1],
            "InternalCounterValue": [0], "one_sec_nearby": [False],
        }).to_csv(self.output / "R_source_state_boundaries.csv", index=False)
        with closing(sqlite3.connect(self.db)) as con:
            for time in (0, 60, 120, 180, 240):
                con.execute("INSERT INTO tblDetectionRaw VALUES ('M','BC',?,0)", (time,))
                con.execute("INSERT INTO tblDetectionRaw VALUES ('R','BC',?,0)", (time+0.01,))
            con.commit()
            before = self.db.read_bytes()
            with patch("_2025_pipeline.clock_source_boundary_review", return_value={"status": "reviewed"}):
                summary = clock_source_median_review(con, "R", 0, 300, self.output)
        self.assertEqual(self.db.read_bytes(), before)
        self.assertEqual(summary["status"], "diagnostic_only")
        self.assertEqual(summary["measurement_status_counts"], {"unresolved": 1})
        report = pd.read_csv(summary["report"])
        self.assertEqual(report.source_row.tolist(), [10])

    def test_source_median_review_refuses_unavailable_original_evidence(self):
        with closing(sqlite3.connect(self.db)) as con:
            with patch("_2025_pipeline.clock_source_boundary_review", return_value={"status": "unavailable"}):
                with self.assertRaisesRegex(RuntimeError, "Original source boundaries unavailable"):
                    clock_source_median_review(con, "R", 0, 300, self.output)

    def test_source_median_review_refuses_self_reference(self):
        self.output.mkdir()
        pd.DataFrame({"seconds": [100]}).to_csv(
            self.output / "R_source_state_boundaries.csv", index=False)
        with closing(sqlite3.connect(self.db)) as con:
            with patch("_2025_pipeline.clock_source_boundary_review", return_value={"status": "reviewed"}):
                with self.assertRaisesRegex(ValueError, "not self-comparison"):
                    clock_source_median_review(con, "R", 0, 300, self.output, reference_host="R")

    def test_source_median_review_rejects_competing_alternate_reference_event(self):
        self.output.mkdir()
        sources = pd.DataFrame({
            "Rec_ID": ["R"], "SourceFile": ["receiver.csv"], "SourceRow": [10],
            "seconds": [100.], "OffsetChanged": [1], "CounterRestart": [0],
            "CounterContinuation": [0], "InternalOffsetValue": [1],
            "InternalCounterValue": [0], "one_sec_nearby": [False],
        })
        sources.to_csv(self.output / "R_source_state_boundaries.csv", index=False)
        sources.assign(Rec_ID="A", seconds=110).to_csv(
            self.output / "A_source_state_boundaries.csv", index=False)
        measurement = {"Rec_ID": "R", "jump_time_s": 100., "source_row": 10,
                       "reference_start_s": 60., "reference_end_s": 180., "offset_s": 1.}
        with closing(sqlite3.connect(self.db)) as con:
            with patch("_2025_pipeline.clock_source_boundary_review", return_value={"status": "reviewed"}), \
                    patch("clock_jump_pre.quantify_jumps", return_value=([measurement], [], [], 1)) as quantify:
                summary = clock_source_median_review(
                    con, "R", 0, 300, self.output, reference_host="M", anchor_receiver="A")
        self.assertEqual(quantify.call_args.args[2:4], ("A", "BC"))
        self.assertEqual(summary["measurement_status_counts"], {"unresolved": 1})
        report = pd.read_csv(summary["report"])
        self.assertEqual(report.competing_anchor_events.tolist(), [1.])

    def test_rapid_producer_checks_omitted_reference_source_events(self):
        self.output.mkdir()
        states = pd.DataFrame({
            "SourceFile": ["master.csv"]*3, "SourceRow": [10, 20, 30],
            "seconds": [100., 160., 220.], "previous_offset": [0, 1, 2],
            "InternalOffsetValue": [1, 2, 3], "OffsetChanged": [1]*3,
            "CounterRestart": [0]*3, "one_sec_nearby": [False]*3,
            "CounterContinuation": [0]*3,
        })
        states.to_csv(self.output / "M_source_state_boundaries.csv", index=False)

        def recover_source(con, host, start, end, folder):
            pd.DataFrame({
                "Rec_ID": [host], "seconds": [150.],
                "previous_offset": [0], "InternalOffsetValue": [1 if host == "B" else 0],
                "CounterRestart": [0],
                "CounterContinuation": [0],
            }).to_csv(folder / ("%s_source_state_boundaries.csv" % host), index=False)
            return {"status": "reviewed"}

        paired = pd.DataFrame({"Rec_ID": ["M"]*2, "t": [130., 192.],
                               "tdoa_s": [0., 2.]})
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN SourceFile TEXT")
            con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN SourceRow INTEGER")
            for name in ("Internal", "ReceiverType", "FirmwareVersion", "FileFormatVersion"):
                con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN %s TEXT" % name)
            for host in ("A", "B", "C"):
                con.execute("INSERT INTO tblReceiver VALUES (?,?)", (host, host))
                con.execute("INSERT INTO tblTag VALUES (?,60)", (host,))
                for time, row in ((130., 15), (192., 25)):
                    con.execute("INSERT INTO tblDetectionRaw (Rec_ID,Tag_ID,seconds,SourceFile,SourceRow) "
                                "VALUES ('M',?,?,'master.csv',?)", (host, time, row))
                for time in (130., 190.):
                    con.execute("INSERT INTO tblDetectionRaw (Rec_ID,Tag_ID,seconds,SourceFile,SourceRow) "
                                "VALUES (?,?,?,?,?)", (host, host, time, host+".csv", int(time)))
            with patch("_2025_pipeline.clock_source_boundary_review", side_effect=recover_source), \
                    patch("clock_jump_pre.beacon_first_arrivals", return_value=paired), \
                    patch("clock_jump_pre.find_events", return_value=(
                        pd.DataFrame(columns=["seconds"]), pd.DataFrame())):
                summary = clock_rapid_reference_review(con, "M", ["A", "B", "C"], 90, 230, self.output)
            self.assertEqual(summary["status"], "measured_not_applied")
            support = pd.read_csv(self.output / "M_rapid_consensus_references.csv")
            blocked = support[(support.source_row == 20) & (support.reference_host == "B")].iloc[0]
            self.assertIn("reference has its own Internal event", blocked.note)
            self.assertEqual(blocked.n_competing_reference_events, 1)
            self.assertEqual(blocked.pre_source_row, 15)
            self.assertEqual(blocked.post_source_row, 25)
            result = pd.read_csv(self.output / "M_rapid_consensus_events.csv")
            self.assertEqual(result[result.source_row == 20].iloc[0].status, "unresolved")


if __name__ == "__main__":
    unittest.main()
