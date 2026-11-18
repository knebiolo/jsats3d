"""Tests for the incremental tag-addition machinery (2026-11-17).

Covers the pure logic that lets run_data.py + _2025_pipeline.py add new tags to a
finished database without redoing the finished ones:
  - measure_pulse_rate (run file method: median inter-pulse gap in [2.5, 4.5] s)
  - staged_tags (which selected tags already have raw rows)
  - tag_markers (per-tag __mp__/__pos__ markers, backfilled from finished-stage rows)
  - the pending-based step gates (fish_multipath_done / positions_done / export_done)
  - spatial_box (explicit position_3d_box_xy override vs receiver-derived extent)
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import _2025_pipeline as p25  # noqa: E402
import legacy_pipeline as lp  # noqa: E402
import run_data as rd  # noqa: E402
from ent_analysis_2025 import spatial_box  # noqa: E402


def make_db(tmpdir, name="t.db"):
    path = os.path.join(tmpdir, name)
    con = sqlite3.connect(path)
    con.execute("create table tblDetectionRaw (Tag_ID TEXT, Rec_ID TEXT, seconds REAL)")
    con.close()
    return path


class TestMeasurePulseRate(unittest.TestCase):
    def _raw(self, tag, receivers, period, n_pings, jitter=0.005):
        rows = []
        for r in receivers:
            for i in range(n_pings):
                rows.append((tag, r, 1000.0 + i * period + jitter * (i % 3)))
        return pd.DataFrame(rows, columns=["Tag_ID", "Rec_ID", "seconds"])

    def test_median_gap(self):
        raw = self._raw("AAAA", ["R1", "R2"], 3.0, 300)
        med, p10, p90, n = rd.measure_pulse_rate(raw, "AAAA")
        self.assertIsNotNone(med)
        self.assertAlmostEqual(med, 3.0, delta=0.02)
        self.assertGreater(n, 500)

    def test_sparse_returns_none(self):
        raw = self._raw("BBBB", ["R1"], 3.0, 30)  # 29 gaps < 50
        self.assertIsNone(rd.measure_pulse_rate(raw, "BBBB"))

    def test_out_of_window_gaps_excluded(self):
        rows = [("CCCC", "R1", 1000.0 + 0.5 * i) for i in range(200)]  # 0.5 s gaps: outside [2.5, 4.5]
        raw = pd.DataFrame(rows, columns=["Tag_ID", "Rec_ID", "seconds"])
        self.assertIsNone(rd.measure_pulse_rate(raw, "CCCC"))


class TestStagedTags(unittest.TestCase):
    def test_missing_db_is_none(self):
        self.assertIsNone(rd.staged_tags(rd.Path("/nonexistent/nope.db")))

    def test_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = make_db(tmp)
            con = sqlite3.connect(db)
            con.executemany("insert into tblDetectionRaw values (?,?,?)",
                            [("AAAA", "R1", 1.0), ("BBBB", "R1", 2.0), ("BBBB", "R2", 3.0)])
            con.commit()
            con.close()
            self.assertEqual(rd.staged_tags(rd.Path(db)), {"AAAA", "BBBB"})

    def test_db_without_raw_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "nodb.db")
            con = sqlite3.connect(db)
            con.execute("create table tblTag (Tag_ID TEXT)")
            con.commit()
            con.close()
            self.assertEqual(rd.staged_tags(rd.Path(db)), set())


class TestTagMarkers(unittest.TestCase):
    def test_backfill_from_finished_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = make_db(tmp)
            con = sqlite3.connect(db)
            con.execute("create table tblDetectionFilterSecondary (Tag_ID TEXT, n INT)")
            con.execute("insert into tblDetectionFilterSecondary values ('AAAA', 1)")
            con.execute("create table tblPositions_Deng (Tag_ID TEXT, comment TEXT)")
            con.execute("insert into tblPositions_Deng values ('AAAA', 'solution found')")
            con.commit()
            con.close()
            marked, backfilled = p25.tag_markers(db, ["AAAA", "BBBB"])
            self.assertIn("__mp__AAAA", marked)
            self.assertIn("__pos__AAAA", marked)
            self.assertNotIn("__mp__BBBB", marked)
            self.assertNotIn("__pos__BBBB", marked)
            self.assertEqual(sorted(backfilled), ["__mp__AAAA", "__pos__AAAA"])
            # idempotent: second call backfills nothing
            marked2, backfilled2 = p25.tag_markers(db, ["AAAA", "BBBB"])
            self.assertEqual(marked2, marked)
            self.assertEqual(backfilled2, [])

    def test_fresh_db_no_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = make_db(tmp)
            marked, backfilled = p25.tag_markers(db, ["AAAA"])
            self.assertEqual(marked, set())
            self.assertEqual(backfilled, [])


class TestPendingGates(unittest.TestCase):
    def _c(self, **kw):
        base = {"pending_mp": [], "pending_pos": []}
        base.update(kw)
        return SimpleNamespace(**base)

    def test_done_when_no_pending(self):
        self.assertTrue(p25.fish_multipath_done(self._c()))
        self.assertTrue(p25.positions_done(self._c()))
        self.assertTrue(p25.export_done(self._c()))

    def test_not_done_with_pending(self):
        c = self._c(pending_mp=["AAAA"], pending_pos=["AAAA", "BBBB"])
        self.assertFalse(p25.fish_multipath_done(c))
        self.assertFalse(p25.positions_done(c))
        self.assertFalse(p25.export_done(c))


class TestBuildStepsStaging(unittest.TestCase):
    """Regression test (2026-11-17): the staging step's callable must carry the missing-tag
    list by VALUE. The first version used `lambda: stage_new_tags(..., missing)` and a later
    `missing = [...]` reassignment in build_steps (study keys) silently emptied it, so the
    parser ran UNFILTERED and wrote a ~15 GB temporary database (disk full)."""

    def _build(self, db):
        """Build the steps with stage_new_tags recorded (patch restored on return)."""
        calls = []
        original_stage = rd.stage_new_tags
        rd.stage_new_tags = lambda output_db, paths, start, end, new_tags: calls.append(list(new_tags))
        try:
            args = SimpleNamespace(skip_build=False)
            # legacy run=True forces the `missing` reassignment below the staging branch
            # (study keys check) - that is what broke the naive closure.
            steps = rd.build_steps(args, Path(db), "ats", {"config_xlsx": "unused.xlsx"}, {},
                                   {"bm_elev": 861.5, "master_receiver": "ZOI08"},
                                   {"run": True}, {}, {}, Path(db),
                                   "2025-06-15 00:00:00", "2025-08-06 23:59:59",
                                   {"26E4", "80FD"})
            for name, step in steps:
                if "stage" in name and callable(step):
                    step()
            return steps, calls
        finally:
            rd.stage_new_tags = original_stage

    def test_stage_step_captures_missing_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "main.db")
            con = sqlite3.connect(db)
            con.execute("create table tblDetectionRaw (Tag_ID TEXT, Rec_ID TEXT, seconds REAL)")
            con.execute("insert into tblDetectionRaw values ('26E4', 'R1', 1.0)")
            con.commit()
            con.close()
            steps, calls = self._build(db)
            name = [n for n, _ in steps if "stage" in n]
            self.assertEqual(name, ["parse + stage new tags: 80FD"])
            self.assertEqual(calls, [["80FD"]])  # not [] (late binding) and not both tags

    def test_no_stage_step_when_all_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "main.db")
            con = sqlite3.connect(db)
            con.execute("create table tblDetectionRaw (Tag_ID TEXT, Rec_ID TEXT, seconds REAL)")
            con.executemany("insert into tblDetectionRaw values (?,?,?)",
                            [("26E4", "R1", 1.0), ("80FD", "R1", 2.0)])
            con.commit()
            con.close()
            steps, calls = self._build(db)
            self.assertFalse(any("stage" in n for n, _ in steps))
            self.assertEqual(calls, [])

    def test_stage_new_tags_refuses_empty(self):
        with self.assertRaises(ValueError):
            rd.stage_new_tags(Path("unused.db"), {}, None, None, [])


class TestDeepBeaconsDone(unittest.TestCase):
    """The deep beacons' rows in tblDetectionFilterSecondary are wiped by the per-fish
    multipath reload (step 8) on every finished DB, so the done check must fall back to
    the downstream evidence: deep receivers positioned (step 6 consumed the beacons)."""

    def _c(self, db):
        return SimpleNamespace(db=Path(db), run={"legacy": {"deep_solution": "A"}},
                               deng=["ZOI01"], deng_tags={"ZOI01": "FFC7"})

    def _mk(self, tmp, secondary_tags=(), deep_solutions=()):
        db = os.path.join(tmp, "deep.db")
        con = sqlite3.connect(db)
        con.execute("create table tblDetectionFilterSecondary (Tag_ID TEXT, n INT)")
        for t in secondary_tags:
            con.execute("insert into tblDetectionFilterSecondary values (?, 1)", (t,))
        con.execute("create table tblDeepReceiver_step6_solution (Rec_ID TEXT, solution_root TEXT)")
        for rec, root in deep_solutions:
            con.execute("insert into tblDeepReceiver_step6_solution values (?,?)", (rec, root))
        con.commit()
        con.close()
        return db

    def test_fresh_db_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(p25.deep_beacons_done(self._c(self._mk(tmp))))

    def test_beacon_rows_present_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(p25.deep_beacons_done(self._c(self._mk(tmp, secondary_tags=("FFC7",)))))

    def test_fallback_on_positioned_deep_receivers(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._mk(tmp, deep_solutions=(("ZOI01", "A"),))
            self.assertTrue(p25.deep_beacons_done(self._c(db)))

    def test_fallback_wrong_root_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self._mk(tmp, deep_solutions=(("ZOI01", "B"),))
            self.assertFalse(p25.deep_beacons_done(self._c(db)))


class TestSpatialBox(unittest.TestCase):
    def _recs(self):
        return pd.DataFrame(
            {"X_t": [10.0, 50.0, 30.0], "Y_t": [5.0, 5.0, 40.0], "Z_t": [-12.0, -4.0, -8.0]},
            index=["R1", "R2", "R3"])

    def test_receiver_derived(self):
        box, margin, src = spatial_box({}, self._recs(), floor_m=-16.0, box_margin_m=10.0)
        self.assertEqual(box, (10.0, 50.0, 5.0, 40.0, -16.0, 0.0))
        self.assertEqual(margin, 10.0)
        self.assertIn("receivers", src)

    def test_explicit_override(self):
        box, margin, src = spatial_box({"position_3d_box_xy": [100.0, 200.0, 30.0, 90.0]},
                                       self._recs(), floor_m=-16.0, box_margin_m=10.0)
        self.assertEqual(box, (100.0, 200.0, 30.0, 90.0, -16.0, 0.0))
        self.assertEqual(margin, 0.0)
        self.assertIn("explicit", src)

    def test_bad_explicit_rejected(self):
        with self.assertRaises(ValueError):
            spatial_box({"position_3d_box_xy": [1.0, 2.0, 3.0]}, self._recs(), -16.0, 10.0)

    def test_empty_receivers_raises(self):
        with self.assertRaises(ValueError):
            spatial_box({}, pd.DataFrame({"X_t": [np.nan], "Y_t": [np.nan], "Z_t": [np.nan]}), -16.0, 10.0)


if __name__ == "__main__":
    unittest.main()
