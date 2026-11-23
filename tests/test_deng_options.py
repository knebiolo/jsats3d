"""Tests for the step 11 Deng options in scripts/_2025_pipeline.py (receiver cap, outside-array filter)."""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import _2025_pipeline as p25  # noqa: E402


def pings(spec):
    """spec: {transNo: [Rec_ID in arrival order]}"""
    rows = [(t, r, 1000.0 * t + 0.001 * i) for t, recs in spec.items() for i, r in enumerate(recs)]
    return pd.DataFrame(rows, columns=["transNo", "Rec_ID", "seconds_fix"]).sample(frac=1, random_state=1)


class CapReceivers(unittest.TestCase):
    def test_keeps_earliest_and_swaps_in_deep(self):
        shallow = ["S%d" % i for i in range(6)]
        df = pings({1: ["D1"] + shallow,                 # deep already among the first 4
                    2: shallow[:5] + ["D2", "D3"],        # deep only late: earliest deep (D2) replaces 4th
                    3: shallow[:3]})                      # fewer than the cap: unchanged
        out, swapped = p25.cap_receivers(df, 4, deep=("D1", "D2", "D3"))
        got = {t: g.sort_values("seconds_fix").Rec_ID.tolist() for t, g in out.groupby("transNo")}
        self.assertEqual(got[1], ["D1", "S0", "S1", "S2"])
        self.assertEqual(got[2], ["S0", "S1", "S2", "D2"])
        self.assertEqual(got[3], ["S0", "S1", "S2"])
        self.assertEqual(swapped, 1)
        self.assertTrue(out.seconds_fix.is_monotonic_increasing)

    def test_no_deep_swap_when_disabled(self):
        df = pings({1: ["S0", "S1", "S2", "S3", "D1"]})
        out, swapped = p25.cap_receivers(df, 4, deep=())
        self.assertEqual(sorted(out.Rec_ID), ["S0", "S1", "S2", "S3"])
        self.assertEqual(swapped, 0)


class OutsideHull(unittest.TestCase):
    def test_marks_found_rows_beyond_margin(self):
        from scipy.spatial import ConvexHull
        eq = ConvexHull(np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)).equations
        d = pd.DataFrame({"X": ["50", "103", "120", "9999."], "Y": ["50", "50", "50", "9999."],
                          "comment": ["solution found"] * 3 + ["singular matrix encountered - no soluiton"]})
        with tempfile.TemporaryDirectory() as tmp:
            d.to_csv(os.path.join(tmp, "ABCD_solutionB.csv"))
            self.assertEqual(p25.reject_outside_hull(tmp, "ABCD", eq, 5.0), 1)
            back = pd.read_csv(os.path.join(tmp, "ABCD_solutionB.csv"), index_col=0)
        self.assertEqual(back.comment.tolist(), ["solution found", "solution found", p25.OUTSIDE_ARRAY,
                                                 "singular matrix encountered - no soluiton"])


if __name__ == "__main__":
    unittest.main()
