"""Resume logic of scripts/legacy_pipeline.py on tiny temporary databases (no real data)."""
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "scripts")))
import legacy_pipeline as lp  # noqa: E402


def make_db(folder, clock_receivers=(), secondary_tags=()):
    db = Path(folder) / "t.db"
    con = sqlite3.connect(db)
    con.execute("create table tblDetectionClockFixed (Rec_ID TEXT, Tag_ID TEXT)")
    con.execute("create table tblDetectionFilterPrimary (Tag_ID TEXT, transNo REAL)")
    con.execute("create table tblDetectionFilterSecondary (Tag_ID TEXT, transNo REAL)")
    con.executemany("insert into tblDetectionClockFixed values (?, 'x')", [(r,) for r in clock_receivers])
    for tag in secondary_tags:
        con.execute("insert into tblDetectionFilterPrimary values (?, 1)", (tag,))
        con.execute("insert into tblDetectionFilterSecondary values (?, 1)", (tag,))
    con.commit()
    con.close()
    return db


class TestResume(unittest.TestCase):
    def test_ats_fish_filters_keep_primary_and_secondary_separate(self):
        import tagdrag_2025_pipeline as pipeline
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "two_filters.db"
            con = sqlite3.connect(db)
            con.execute("create table tblTag (Tag_ID TEXT, TagType TEXT, pulseRate REAL)")
            con.execute("insert into tblTag values ('T1', 'study', 3.0)")
            con.execute("create table tblStudyParameters (masterReceiver TEXT)")
            con.execute("insert into tblStudyParameters values ('R1')")
            rows = []
            for trans_no in range(21):
                for rec in ("R1", "R2"):
                    time = 1.75e9 + trans_no * 3.0 + (0.02 if rec == "R2" and trans_no == 10 else 0.0)
                    for delay in (0.0, 0.1):
                        rows.append({"Rec_ID": rec, "Tag_ID": "T1", "seconds_fix": time + delay,
                                     "seconds": time + delay})
            pd.DataFrame(rows).to_sql("tblDetectionClockFixed", con, index=False)
            con.close()
            context = SimpleNamespace(db=db, work=Path(folder), tags=["T1"])
            pipeline.fish_multipath_run(context)
            pipeline.fish_multipath_run(context)
            primary = lp.query(db, "select * from tblDetectionFilterPrimary")
            secondary = lp.query(db, "select * from tblDetectionFilterSecondary")
            self.assertEqual(len(primary), 84)
            self.assertEqual(len(secondary), 84)
            self.assertNotIn("multipath_prediction", primary.columns)
            self.assertEqual(int((primary.multipath == 1).sum()), 42)
            rejected_first = secondary[(secondary.multipath == 0) & (secondary.multipath_prediction == 1)]
            self.assertEqual(rejected_first.Rec_ID.tolist(), ["R2"])
            self.assertEqual(rejected_first.transNo.tolist(), [10.0])
            self.assertTrue(pipeline.fish_multipath_done(context))

    def test_ats_deep_receiver_uses_configured_legacy_root_median(self):
        import tagdrag_2025_pipeline as pipeline
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "deep.db"
            con = sqlite3.connect(db)
            pd.DataFrame({"Rec_ID": ["D1"], "Tag_ID": ["B1"], "X_t": [1.0],
                          "Y_t": [2.0], "Z_t": [-3.0]}).to_sql("tblReceiver", con, index=False)
            con.close()
            root_b = pd.DataFrame({"comment": ["solution found"] * 3, "transNo": [1, 2, 3],
                                   "X": [10.0, 12.0, 14.0], "Y": [20.0, 22.0, 24.0],
                                   "Z": [-8.0, -10.0, -12.0]})
            result = SimpleNamespace(DengSolutionB_unfiltered=root_b,
                                     DengSolutionA_unfiltered=root_b.assign(X=1000.0))
            context = SimpleNamespace(db=db, work=Path(folder), deep=["D1"], reference=["S1"],
                                      run={"legacy": {"deep_solution": "B"}})
            with patch.object(lp, "deng", return_value=result):
                pipeline.deep_positions_run(context)
            solution = lp.query(db, "select * from tblDeepReceiver_step6_solution").iloc[0]
            self.assertEqual((solution.X, solution.Y, solution.Z), (12.0, 22.0, -10.0))
            self.assertEqual(solution.solution_root, "B")
            self.assertTrue(pipeline.deep_positions_done(context))

    def test_ats_phase2_reruns_metronome_before_all_receiver_clocks(self):
        import tagdrag_2025_pipeline as pipeline
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            db = make_db(folder)
            context = SimpleNamespace(db=db, work=Path(folder), review=Path(folder),
                                      surface=["S1", "MASTER"], deep=["D1"])
            calls = []
            def query_result(database, sql, params=()):
                if "having count(*)" in sql:
                    return pd.DataFrame({"n": [0]})
                return pd.DataFrame({"seconds": [100.0, 101.0], "seconds_fix": [99.9, 100.9]})
            with patch.object(pipeline, "metronome_run", side_effect=lambda ctx: calls.append("metronome")), \
                 patch.object(lp, "clock_fix", side_effect=lambda database, recs, work: calls.append(tuple(recs))), \
                 patch.object(lp, "query", side_effect=query_result):
                pipeline.deep_clocks_run(context)
            self.assertEqual(calls, ["metronome", ("S1", "MASTER", "D1")])
            con = sqlite3.connect(db)
            marker = con.execute("select Tag_ID from tblProcessProgress").fetchall()
            con.close()
            self.assertIn(("__ats_phase2__",), marker)

    def test_phases_incomplete_without_all_receivers(self):
        with tempfile.TemporaryDirectory() as folder:
            db = make_db(folder, clock_receivers=["R04"], secondary_tags=["A"])
            self.assertFalse(lp.phases_complete(db, ["R04", "R05"], ["R01"], "R05"))

    def test_phases_complete_from_old_database_marks_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            db = make_db(folder, clock_receivers=["R04", "R01"], secondary_tags=["A"])
            self.assertTrue(lp.phases_complete(db, ["R04", "R05"], ["R01"], "R05"))
            marked = set(lp.query(db, "select Tag_ID from %s" % lp.PROGRESS_TABLE).Tag_ID)
            self.assertIn("__phases__", marked)

    def test_phases_not_complete_before_the_tag_loop_starts(self):
        with tempfile.TemporaryDirectory() as folder:
            db = make_db(folder, clock_receivers=["R04", "R01"])
            self.assertFalse(lp.phases_complete(db, ["R04", "R05"], ["R01"], "R05"))

    def test_old_database_redoes_only_the_last_started_tag(self):
        with tempfile.TemporaryDirectory() as folder:
            db = make_db(folder, secondary_tags=["A", "B", "C"])
            positions, positions_2d = Path(folder, "p"), Path(folder, "p2")
            positions.mkdir()
            positions_2d.mkdir()
            for tag in "ABC":
                (positions / ("%s_solutionA.csv" % tag)).write_text("x")
            done = lp.prepare_resume(db, ["A", "B", "C", "D"], positions, positions_2d)
            self.assertEqual(done, {"A", "B"})
            self.assertEqual(set(lp.query(db, "select distinct Tag_ID from tblDetectionFilterSecondary").Tag_ID), {"A", "B"})
            self.assertEqual(sorted(p.name for p in positions.glob("*.csv")), ["A_solutionA.csv", "B_solutionA.csv"])

    def test_progress_table_is_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            db = make_db(folder, secondary_tags=["A", "B"])
            lp.mark_done(db, "A")
            lp.mark_done(db, "B")
            positions, positions_2d = Path(folder, "p"), Path(folder, "p2")
            positions.mkdir()
            positions_2d.mkdir()
            done = lp.prepare_resume(db, ["A", "B", "C"], positions, positions_2d)
            self.assertEqual(done, {"A", "B"})
            self.assertEqual(set(lp.query(db, "select distinct Tag_ID from tblDetectionFilterSecondary").Tag_ID), {"A", "B"})

    def test_nothing_started_means_nothing_finished(self):
        with tempfile.TemporaryDirectory() as folder:
            db = make_db(folder)
            positions, positions_2d = Path(folder, "p"), Path(folder, "p2")
            positions.mkdir()
            positions_2d.mkdir()
            self.assertEqual(lp.prepare_resume(db, ["A", "B"], positions, positions_2d), set())


class TestNoBlockingPlots(unittest.TestCase):
    def test_pipeline_scripts_use_a_non_gui_matplotlib_backend(self):
        import subprocess
        for script in ("tagdrag_2025_pipeline", "legacy_pipeline"):
            code = "import sys; sys.path.insert(0, r'%s'); import %s, matplotlib; print(matplotlib.get_backend().lower())" % (
                os.path.abspath(os.path.join(HERE, "..", "scripts")), script)
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stdout.strip().splitlines()
            self.assertEqual(out[-1] if out else "", "agg", "%s would open plot windows and block the run" % script)


class TestWhatIfSwap(unittest.TestCase):
    def test_confirmed_swap_moves_positions_only(self):
        import pandas as pd
        import adapt_2025_to_legacy as ad
        table = pd.DataFrame({"Rec_ID": ["ZOI05", "ZOI06", "ZOI07"], "Tag_ID": ["t5", "t6", "t7"], "MountDescription": ["m5", "m6", "m7"],
                              **{c: [1.0 + i, 2.0 + i, 3.0 + i] for i, c in enumerate(ad.POSITION_COLUMNS)}})
        out = ad.swap_positions(table, ad.CONFIRMED_POSITION_SWAPS).set_index("Rec_ID")
        before = table.set_index("Rec_ID")
        self.assertEqual(ad.CONFIRMED_POSITION_SWAPS, (("ZOI05", "ZOI06"),))
        self.assertEqual(out.loc["ZOI05", ad.POSITION_COLUMNS].tolist(), before.loc["ZOI06", ad.POSITION_COLUMNS].tolist())
        self.assertEqual(out.loc["ZOI06", ad.POSITION_COLUMNS].tolist(), before.loc["ZOI05", ad.POSITION_COLUMNS].tolist())
        self.assertEqual(out.loc["ZOI05", ["Tag_ID", "MountDescription"]].tolist(), ["t5", "m5"])
        self.assertEqual(out.loc["ZOI07", ad.POSITION_COLUMNS].tolist(), before.loc["ZOI07", ad.POSITION_COLUMNS].tolist())

    def test_swap_exchanges_coordinates_only_and_keeps_config_copy(self):
        import tagdrag_2025_pipeline as tp
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / "src.db", Path(folder) / "dst.db"
            con = sqlite3.connect(src)
            con.execute("create table tblReceiver (Rec_ID TEXT, Tag_ID TEXT, X REAL, Y REAL, Z REAL, X_t REAL, Y_t REAL, Z_t REAL, easting REAL, northing REAL)")
            con.executemany("insert into tblReceiver values (?,?,?,?,?,?,?,?,?,?)", [("ZOI05", "t5", 1, 2, -3, 1, 2, -3, 10, 20), ("ZOI06", "t6", 4, 5, -6, 4, 5, -6, 40, 50)])
            con.execute("create table tblReceiver_initial as select * from tblReceiver")
            con.execute("create table tblDetectionClockFixed (x)")
            con.commit()
            con.close()
            tp.init_whatif(src, dst, [["ZOI05", "ZOI06"]])
            con = sqlite3.connect(dst)
            rows = {r[0]: r for r in con.execute("select Rec_ID, Tag_ID, X, Y, Z, X_t, Y_t, Z_t, easting, northing from tblReceiver")}
            tables = {r[0] for r in con.execute("select name from sqlite_master where type = 'table'")}
            initial = con.execute("select X from tblReceiver_initial where Rec_ID = 'ZOI05'").fetchone()[0]
            con.close()
            self.assertEqual(rows["ZOI05"], ("ZOI05", "t5", 4, 5, -6, 4, 5, -6, 40, 50))
            self.assertEqual(rows["ZOI06"], ("ZOI06", "t6", 1, 2, -3, 1, 2, -3, 10, 20))
            self.assertEqual(initial, 1)
            self.assertNotIn("tblDetectionClockFixed", tables)
            self.assertIn("tblReceiverSwap", tables)
            with self.assertRaises(SystemExit):
                tp.init_whatif(src, dst, [])


if __name__ == "__main__":
    unittest.main()
