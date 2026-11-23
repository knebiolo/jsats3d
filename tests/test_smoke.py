"""Basic tests for jsats3d package and data adapters."""
import sys
import os
import unittest

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


class TestSegmentedClockExperiment(unittest.TestCase):
    def test_stepped_levels_keep_all_endpoints(self):
        import numpy as np
        import pandas as pd
        from jsats3d.jsats3d import _segmented_clock_dbscan
        tdoa = np.repeat([0.01, 0.5, 0.1], [10, 5, 3])
        observations = pd.DataFrame({"seconds": np.arange(18) * 60.0,
                                     "ToA_error": tdoa, "DDoA": tdoa * 1500})
        result = _segmented_clock_dbscan(observations, 60.0)
        self.assertEqual(result.seconds.tolist(), observations.seconds.tolist())
        self.assertEqual(result.clock_segment.nunique(), 3)

    def test_interpolation_does_not_bridge_step_or_extrapolate(self):
        import numpy as np
        import pandas as pd
        from jsats3d.jsats3d import _interpolate_clock_segments
        knots = pd.DataFrame({"seconds": [0.0, 60.0, 180.0, 240.0],
                              "seconds_residual": [0.01, 0.01, 0.5, 0.5],
                              "clock_segment": [0, 0, 1, 1]})
        result = _interpolate_clock_segments(knots, np.array([-1, 0, 30, 120, 210, 240, 241]))
        np.testing.assert_allclose(result, [np.nan, 0.01, 0.01, np.nan, 0.5, 0.5, np.nan], equal_nan=True)

    def test_single_knot_level_is_held_to_midpoints_but_outage_is_not_filled(self):
        import numpy as np
        import pandas as pd
        from jsats3d.jsats3d import _segmented_clock_dbscan, _interpolate_clock_segments
        observations = pd.DataFrame({"seconds": [0, 60, 120, 180, 240, 300, 600, 660, 720],
                                     "ToA_error": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                     "DDoA": [0, 0, 0, 1500, 0, 0, 0, 0, 0]})
        result = _segmented_clock_dbscan(observations, 60)
        self.assertEqual(result.seconds.tolist(), observations.seconds.tolist())
        result["seconds_residual"] = result.ToA_error
        corrected = _interpolate_clock_segments(result, np.array([149, 150, 180, 209, 210, 450]))
        np.testing.assert_allclose(corrected, [0, 1, 1, 1, 0, np.nan], equal_nan=True)

    def test_multi_point_segment_keeps_linear_interpolation(self):
        import numpy as np
        import pandas as pd
        from jsats3d.jsats3d import _segmented_clock_dbscan, _interpolate_clock_segments
        observations = pd.DataFrame({"seconds": [0, 60, 120, 240, 300, 360],
                                     "ToA_error": [0, 0.01, 0.02, 1, 1.01, 1.02]})
        observations["DDoA"] = observations.ToA_error * 1500
        result = _segmented_clock_dbscan(observations, 60)
        result["seconds_residual"] = result.ToA_error
        corrected = _interpolate_clock_segments(result, np.array([30, 150, 179, 180, 210, 270]))
        np.testing.assert_allclose(corrected, [0.005, 0.02, 0.02, 1, 1, 1.005])

if __name__ == '__main__':
    unittest.main()
