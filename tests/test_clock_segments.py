import unittest
from contextlib import closing
import sqlite3
from unittest.mock import patch
import numpy as np
import pandas as pd
from jsats3d.jsats3d import (
    multipath_data_object,
    _clock_supported_interpolation as supported_interpolation,
    _clock_mapped_event_bounds as mapped_event_bounds,
    _clock_validate_applied_events as validate_applied_events,
)


class TestClockSegments(unittest.TestCase):
    def test_no_gap_bridge_or_endpoint_hold(self):
        times = [-1, 0, 30, 60, 90, 240, 270, 300, 301, np.nan]
        values, supported = supported_interpolation(
            times, [0, 60, 240, 300], [0, 0.001, 0.002, 0.003], 60)
        self.assertEqual(supported.tolist(),
                         [False, True, True, True, False, True, True, True, False, False])
        self.assertTrue(np.isnan(values[4]))
        self.assertAlmostEqual(values[2], 0.0005)

    def test_step_is_unsupported_even_with_short_gap(self):
        values, supported = supported_interpolation([0, 30, 60], [0, 60], [0, 1], 60)
        self.assertEqual(supported.tolist(), [True, False, True])
        self.assertTrue(np.isnan(values[1]))

    def test_forward_and_backward_events_map_both_sides(self):
        low, high = mapped_event_bounds({"jump_time_s": [100, 200], "offset_s": [1, -2]})
        np.testing.assert_array_equal(low, [99, 199])
        np.testing.assert_array_equal(high, [100, 201])

    def test_explicit_corrected_sides_override_cumulative_mapping(self):
        low, high = mapped_event_bounds({"jump_time_s": [100, 200], "offset_s": [1, -2],
                                         "corrected_before_s": [97, 196], "corrected_after_s": [96, 198]})
        np.testing.assert_array_equal(low, [96, 196])
        np.testing.assert_array_equal(high, [97, 198])
        with self.assertRaises(ValueError):
            mapped_event_bounds({"jump_time_s": [1], "offset_s": [1],
                                 "corrected_before_s": [np.nan], "corrected_after_s": [0]})

    def test_bad_knots_and_period_fail(self):
        for knots, period in (([0, 0], 60), ([0, 60], 0)):
            with self.assertRaises(ValueError):
                supported_interpolation([30], knots, [0, 1], period)

    def test_validation_missing_data_is_not_success_and_budget_is_half_ms(self):
        events = pd.DataFrame({"jump_time_s": [1000], "offset_s": [10]})
        knots = pd.DataFrame({"seconds": [700, 800, 900, 1100, 1200, 1300],
                              "ToA_error": [0, 0, 0, 0.001, 0.001, 0.001]})
        result = validate_applied_events(knots, events, "R")
        self.assertEqual(result[0]["status"], "failed")
        self.assertEqual(result[0]["start_s"], 690)
        result = validate_applied_events(knots.iloc[:3], events, "R")
        self.assertEqual(result[0]["status"], "unverifiable")

    def load_study(self, clock_rows=None, segment_safe=True, detection_time=30):
        with closing(sqlite3.connect(":memory:")) as con:
            con.executescript("""
                CREATE TABLE tblTag (Tag_ID TEXT, pulseRate REAL, TagType TEXT);
                INSERT INTO tblTag VALUES ('F',3,'study'),('BC',60,'beacon');
                CREATE TABLE tblStudyParameters (masterReceiver TEXT);
                INSERT INTO tblStudyParameters VALUES ('M');
                CREATE TABLE tblReceiver (Rec_ID TEXT, Tag_ID TEXT);
                INSERT INTO tblReceiver VALUES ('M','BC');
                CREATE TABLE tblDetectionRaw (Rec_ID TEXT, Tag_ID TEXT, seconds REAL);
            """)
            con.execute("INSERT INTO tblDetectionRaw VALUES ('R','F',?)", (detection_time,))
            if segment_safe:
                con.execute("CREATE TABLE tblClockExperimentPolicy (segment_safe INTEGER, receivers_3d TEXT)")
                con.execute("INSERT INTO tblClockExperimentPolicy VALUES (1,'M;R')")
            if clock_rows is not None:
                con.execute("CREATE TABLE tblDetectionClockFixed "
                            "(Rec_ID TEXT, seconds REAL, seconds_residual REAL)")
                con.executemany("INSERT INTO tblDetectionClockFixed VALUES (?,?,?)", clock_rows)
            with patch("jsats3d.jsats3d.sqlite3.connect", return_value=con):
                return multipath_data_object("F", ":memory:", "")

    def test_standalone_study_loader_blocks_unsupported_gap(self):
        with self.assertRaisesRegex(RuntimeError, "review required: 1 unsupported"):
            self.load_study([("R", 0, 0), ("R", 180, 0)], detection_time=90)

    def test_standalone_study_loader_blocks_missing_receiver_curve(self):
        with self.assertRaisesRegex(RuntimeError, "review required: no clock curve for R"):
            self.load_study([("OTHER", 0, 0), ("OTHER", 60, 0)])

    def test_standalone_study_loader_blocks_absent_or_empty_clock_table(self):
        for clock_rows in (None, []):
            with self.subTest(clock_rows=clock_rows):
                with self.assertRaisesRegex(RuntimeError, "review required: experimental clock curves are missing"):
                    self.load_study(clock_rows)

    def test_standalone_study_loader_keeps_supported_correction(self):
        study = self.load_study([("R", 0, 0.001), ("R", 60, 0.001)])
        self.assertAlmostEqual(study.data.seconds_fix.iloc[0], 29.999)

    def test_legacy_study_loader_keeps_missing_curve_behavior(self):
        study = self.load_study(segment_safe=False)
        self.assertEqual(study.data.seconds_fix.iloc[0], 30)


if __name__ == "__main__":
    unittest.main()
