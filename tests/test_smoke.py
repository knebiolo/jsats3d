"""Basic tests for jsats3d package and data adapters."""
import sys
import os
import unittest
from pathlib import Path
import sqlite3
from contextlib import closing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestJsats3dBasic(unittest.TestCase):
    def test_import_jsats3d(self):
        import jsats3d
        self.assertTrue(hasattr(jsats3d, 'create_project_db'))
        self.assertTrue(hasattr(jsats3d, 'temp_interpolator'))

    def test_import_adapter(self):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
        import adapt_2025_to_legacy
        self.assertTrue(hasattr(adapt_2025_to_legacy, 'normalize_detection'))

    def test_2019_run_plan_keeps_legacy_import_and_processing(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import run_data
        config = Path(__file__).resolve().parents[1] / "config" / "run_data_2019.toml"
        run = run_data.read_run_file(config)
        args = run_data.parse_args([str(config), "--dry-run"])
        self.assertIsNone(args.compact_from)
        self.assertFalse(run["legacy"].get("clock_segment_safe", False))
        start, end = run_data.check_time(run["selection"])
        steps = run_data.build_steps(
            args, config, run["data_format"], run["paths"], run["selection"], run["study"],
            run["legacy"], run["dbscan"], run["outputs"], run["paths"]["output_db"],
            start, end, set())
        self.assertEqual([name for name, _ in steps],
                         ["legacy 2019 import", "study parameters and indexes", "legacy workflow"])
        self.assertIn("import-2019", steps[0][1])
        self.assertIn("process", steps[2][1])

    def test_legacy_database_does_not_enable_experimental_clock_rules(self):
        from jsats3d.jsats3d import _clock_experiment_enabled
        with closing(sqlite3.connect(":memory:")) as con:
            self.assertFalse(_clock_experiment_enabled(con))
            con.execute("CREATE TABLE tblClockExperimentPolicy (segment_safe INTEGER)")
            con.execute("INSERT INTO tblClockExperimentPolicy VALUES (0)")
            self.assertFalse(_clock_experiment_enabled(con))

if __name__ == '__main__':
    unittest.main()
