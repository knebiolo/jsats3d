"""Segmented clock filter (2025 option): jump handling, fallback support and low-support position flags."""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from jsats3d.jsats3d import _interpolate_clock_segments, _segmented_clock_dbscan  # noqa: E402

P = 64.0  # master beacon period (s)


def beacons(times, errors):
    d = pd.DataFrame({"seconds": np.asarray(times, float), "ToA_error": np.asarray(errors, float)})
    d["DDoA"] = d.ToA_error * 1500
    return d


def fit(obs, **kw):
    summary = []
    out = _segmented_clock_dbscan(obs, P, summary=summary, **kw)
    out["seconds_residual"] = out.ToA_error
    return out, pd.DataFrame(summary)


class SegmentedClock(unittest.TestCase):
    def test_26e4_midpoint_holds(self):
        # 26E4 ZOI02 2025-06-28: +0.410 to 01:00:22, +1.601 at 01:03:32, +0.601 at 01:04:36, -1.4 from 01:05:47
        pre = 3622 - P * np.arange(9, -1, -1)
        post = 3947 + P * np.arange(10)
        obs = beacons(np.r_[pre, 3812, 3876, post], np.r_[[0.410] * 10, 1.601, 0.601, [-1.4] * 10])
        out, seg = fit(obs)
        self.assertEqual(sorted(out.seconds), sorted(obs.seconds))      # endpoints and single knots retained
        t = np.array([3716, 3718, 3843, 3845, 3911, 3912])
        corr, support = _interpolate_clock_segments(out, t, return_support=True)
        np.testing.assert_allclose(corr, [0.410, 1.601, 1.601, 0.601, 0.601, -1.4])  # 01:01:57 and 01:04:04
        np.testing.assert_allclose(support, [10, 1, 1, 1, 1, 10])
        single = seg[seg.n_observed == 1]
        self.assertEqual(single.method.tolist(), ["constant", "constant"])
        self.assertTrue(single.start_estimated.all() and single.end_estimated.all())
        self.assertEqual(seg[seg.n_observed == 10].method.tolist(), ["supported", "supported"])

    def test_two_point_jump_segment_is_linear(self):
        obs = beacons(np.r_[0, 64, 128, 192, 256, 320, 384, 448],
                      [0, 0, 0, 1.0, 1.02, 2.0, 2.0, 2.0])
        out, seg = fit(obs)
        self.assertEqual(seg[seg.n_observed == 2].method.item(), "linear")
        np.testing.assert_allclose(_interpolate_clock_segments(out, np.array([224.0])), [1.01])

    def test_isolated_spike_removed_not_held(self):
        obs = beacons(P * np.arange(8), [0, 0, 0, 0.8, 0, 0, 0, 0])
        out, seg = fit(obs)
        self.assertNotIn(3 * P, out.seconds.tolist())
        self.assertEqual(seg.method.tolist().count("isolated_multipath"), 1)
        np.testing.assert_allclose(_interpolate_clock_segments(out, np.array([3 * P])), [0.0])

    def test_step_free_long_gaps_remain_in_one_segment(self):
        obs = beacons([0, 64, 128, 1000, 1064, 2000, 2064, 2128], [0] * 8)
        out, seg = fit(obs)
        self.assertEqual(out.seconds.tolist(), obs.seconds.tolist())
        self.assertEqual(seg.method.tolist(), ["supported"])
        corr, support = _interpolate_clock_segments(out, np.array([1032.0, 64.0]), return_support=True)
        np.testing.assert_allclose(corr, [0.0, 0.0])
        np.testing.assert_allclose(support, [8, 8])

    def test_subthreshold_change_is_interpolated_across_long_gap(self):
        obs = beacons([0, 64, 128, 1000, 1064, 1128], [0, 0, 0, 0.049, 0.049, 0.049])
        out, seg = fit(obs)
        self.assertEqual(len(seg), 1)
        np.testing.assert_allclose(_interpolate_clock_segments(out, np.array([564.0])), [0.0245])

    def test_step_across_long_gap_is_not_bridged(self):
        obs = beacons([0, 64, 128, 1500, 1564, 1628], [0, 0, 0, 1, 1, 1])
        out, seg = fit(obs, max_jump_gap_s=600)
        self.assertIn("step_outage", seg.boundary_after.tolist())
        self.assertFalse(seg.end_estimated.any())
        corr = _interpolate_clock_segments(out, np.array([128.0, 800.0, 1500.0]))
        np.testing.assert_allclose(corr, [0, np.nan, 1], equal_nan=True)


class ClockSupportFlags(unittest.TestCase):
    def test_flags_follow_receivers_used_and_consensus(self):
        import _2025_pipeline as p25
        from ent_analysis_2025 import legacy_receiver_set_consensus
        cols = ["transNo", "solNo", "r0", "r1", "r2", "r3", "X", "Y", "Z", "T01", "ToA", "comment", "in_hull"]
        sets = [("A", "B", "C", "D"), ("A", "B", "C", "F"), ("A", "B", "D", "F"), ("A", "B", "C", "E"), ("A", "B", "D", "E")]
        xs = {1.0: [10, 10.5, 11, 300, 400],      # trans 1: the E solutions are outliers, not in the consensus
              2.0: [10, 10.5, 11, 11.5, 12]}      # trans 2: the E solutions agree and contribute
        rows = [[t, i, *s, xs[t][i], 5.0, -3.0, 0.1, 100.0 * t + 0.1, "solution found", 1]
                for t in (1.0, 2.0) for i, s in enumerate(sets)]
        # F has no support value (e.g. a receiver outside the segmented run): not flagged
        tag_data = pd.DataFrame({"transNo": [1.0] * 5 + [2.0] * 5, "Rec_ID": list("ABCDE") * 2,
                                 "clock_support": [10, 10, 10, 10, 1, 10, 10, 10, 10, 2]})
        with tempfile.TemporaryDirectory() as tmp:
            for root in "AB":
                pd.DataFrame(rows, columns=cols).astype(str).to_csv(os.path.join(tmp, "T001_solution%s.csv" % root))
            flagged = p25.flag_clock_support(tmp, "T001", tag_data)
            d = pd.read_csv(os.path.join(tmp, "T001_solutionA.csv"), index_col=0, keep_default_na=False)
        self.assertEqual(flagged, 4)
        self.assertEqual(d.clock_low_support.tolist(), [0, 0, 0, 1, 1] * 2)
        self.assertEqual(d.clock_low_support_receivers.tolist(), ["", "", "", "E", "E"] * 2)
        self.assertEqual(d.clock_min_support.tolist(), [10, 10, 10, 1, 1, 10, 10, 10, 2, 2])
        d["t0"] = d.ToA - d.T01
        d["clock_low_support_receivers"] = d.clock_low_support_receivers.replace("", np.nan)
        g = legacy_receiver_set_consensus(d).set_index("transNo")
        self.assertEqual(int(g.loc[1.0, "clock_low_support"]), 0)        # flagged solutions did not contribute
        self.assertEqual(g.loc[1.0, "clock_min_support"], 10)
        self.assertEqual(int(g.loc[2.0, "clock_low_support"]), 1)
        self.assertEqual(g.loc[2.0, "clock_low_support_receivers"], "E")
        self.assertEqual(int(g.loc[2.0, "clock_low_support_solutions"]), 2)
        self.assertEqual(g.loc[2.0, "clock_min_support"], 2)


if __name__ == "__main__":
    unittest.main()
