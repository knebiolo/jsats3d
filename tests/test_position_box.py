"""Tests for the 3D window screen and present-unpositioned helpers (PM 2026-11-14)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from ent_analysis_2025 import (receiver_3d_box, three_d_box_mask,  # noqa: E402
                               present_unpositioned_rows, legacy_receiver_set_consensus)


def deng_frame(rows):
    """Build a minimal Deng-style frame from (transNo, X, Y, Z, t0, set) tuples."""
    cols = ["transNo", "r0", "r1", "r2", "r3", "X", "Y", "Z", "t0", "in_hull"]
    out = []
    for (t, x, y, z, t0, r0, r1, r2, r3) in rows:
        out.append([t, r0, r1, r2, r3, x, y, z, t0, True])
    return pd.DataFrame(out, columns=cols)


class TestReceiverBox(unittest.TestCase):
    def test_min_max(self):
        rec = pd.DataFrame({"X_t": [10.0, 50.0, 20.0], "Y_t": [5.0, 9.0, 100.0],
                            "Z_t": [-12.0, -3.0, -2.5]})
        self.assertEqual(receiver_3d_box(rec), (10.0, 50.0, 5.0, 100.0, -12.0, -2.5))

    def test_drops_nan_rows(self):
        rec = pd.DataFrame({"X_t": [10.0, np.nan, 50.0], "Y_t": [5.0, 9.0, 9.0],
                            "Z_t": [-12.0, -3.0, -2.5]})
        self.assertEqual(receiver_3d_box(rec), (10.0, 50.0, 5.0, 9.0, -12.0, -2.5))

    def test_empty_raises(self):
        rec = pd.DataFrame({"X_t": [np.nan], "Y_t": [np.nan], "Z_t": [np.nan]})
        with self.assertRaises(ValueError):
            receiver_3d_box(rec)


class TestBoxMask(unittest.TestCase):
    BOX = (10.0, 50.0, 5.0, 100.0, -16.0, 0.0)

    def _frame(self, **kw):
        base = dict(X=30.0, Y=50.0, Z=-8.0)
        base.update(kw)
        return pd.DataFrame([base])

    def test_inside_outside(self):
        self.assertTrue(bool(three_d_box_mask(self._frame(), self.BOX)[0]))
        self.assertFalse(bool(three_d_box_mask(self._frame(X=51.0), self.BOX)[0]))
        self.assertFalse(bool(three_d_box_mask(self._frame(X=9.0), self.BOX)[0]))
        self.assertFalse(bool(three_d_box_mask(self._frame(Y=101.0), self.BOX)[0]))
        self.assertFalse(bool(three_d_box_mask(self._frame(Z=0.5), self.BOX)[0]))
        self.assertFalse(bool(three_d_box_mask(self._frame(Z=-17.0), self.BOX)[0]))

    def test_boundaries_are_inclusive(self):
        for kw in (dict(X=10.0), dict(X=50.0), dict(Y=5.0), dict(Y=100.0),
                   dict(Z=-16.0), dict(Z=0.0)):
            self.assertTrue(bool(three_d_box_mask(self._frame(**kw), self.BOX)[0]))

    def test_margin_widens(self):
        self.assertFalse(bool(three_d_box_mask(self._frame(X=55.0), self.BOX)[0]))
        self.assertTrue(bool(three_d_box_mask(self._frame(X=55.0), self.BOX, margin_m=5.0)[0]))
        self.assertTrue(bool(three_d_box_mask(self._frame(Z=1.0), self.BOX, margin_m=2.0)[0]))

    def test_nan_is_rejected(self):
        for kw in (dict(X=np.nan), dict(Y=np.nan), dict(Z=np.nan)):
            self.assertFalse(bool(three_d_box_mask(self._frame(**kw), self.BOX)[0]))


class TestConsensusWithFarOffSolution(unittest.TestCase):
    """Clock-jump-class solutions land km-scale off in XY but can carry a PLAUSIBLE Z
    (the existing surface/floor Z gate does not catch them). When the jumped receiver
    poisons a MAJORITY of the 4-receiver sets for a transmission, the consensus center
    snaps to the gross cluster and the 10 m radius + >=2 sets rule accepts it. The 3D
    window screen, applied before the median, removes the gross cluster first."""

    # 2 clean sets near (30, 50, -8); 3 gross sets ~500 m away with plausible Z
    ROWS = [(1, 30.0, 50.0, -8.0, 100.0, "A", "B", "C", "D"),
            (1, 31.0, 51.0, -8.2, 100.0, "A", "B", "C", "E"),
            (1, 523.0, 139.0, -9.0, 100.0, "B", "C", "E", "F"),
            (1, 525.0, 140.0, -8.5, 100.0, "A", "D", "E", "F"),
            (1, 527.0, 142.0, -7.0, 100.0, "A", "D", "C", "F")]
    BOX = (10.0, 50.0, 5.0, 100.0, -16.0, 0.0)

    def test_majority_gross_breaks_consensus_without_screen(self):
        g = legacy_receiver_set_consensus(deng_frame(self.ROWS))
        self.assertEqual(len(g), 1)
        # consensus snapped to the gross cluster, ~500 m from the true position
        self.assertGreater(np.hypot(g.x.iloc[0] - 30.5, g.y.iloc[0] - 50.5), 400.0)

    def test_box_screen_restores_clean_consensus(self):
        df = deng_frame(self.ROWS)
        kept = df[three_d_box_mask(df, self.BOX)]
        self.assertEqual(len(kept), 2)
        g = legacy_receiver_set_consensus(kept)
        self.assertEqual(len(g), 1)
        self.assertAlmostEqual(g.x.iloc[0], 30.5, places=6)
        self.assertAlmostEqual(g.y.iloc[0], 50.5, places=6)

    def test_single_outlier_needs_no_screen(self):
        # sanity: with only 1 gross set of 5, the median + 10 m radius already survive it
        rows = [(1, 30.0, 50.0, -8.0, 100.0, "A", "B", "C", "D"),
                (1, 31.0, 51.0, -8.2, 100.0, "A", "B", "C", "E"),
                (1, 29.0, 49.0, -7.8, 100.0, "A", "B", "D", "E"),
                (1, 30.5, 50.5, -8.1, 100.0, "A", "B", "E", "F"),
                (1, 530.0, 150.0, -8.0, 100.0, "A", "B", "C", "F")]
        g = legacy_receiver_set_consensus(deng_frame(rows))
        self.assertEqual(len(g), 1)
        self.assertLess(np.hypot(g.x.iloc[0] - 30.0, g.y.iloc[0] - 50.0), 10.0)


class TestConsensusLegacyUnchanged(unittest.TestCase):
    """box=None (default) must be bit-identical to the pre-screen behavior."""

    def _rows(self):
        return [(1, 30.0, 50.0, -8.0, 100.0, "A", "B", "C", "D"),
                (1, 31.0, 51.0, -8.2, 100.0, "A", "B", "C", "E"),
                (2, 200.0, 300.0, -5.0, 200.0, "A", "B", "C", "D"),
                (2, 201.0, 299.0, -4.9, 200.0, "A", "B", "C", "E"),
                (3, 40.0, 60.0, -6.0, 300.0, "A", "B", "C", "D")]  # only 1 set -> dropped

    def test_default_call_unchanged(self):
        df = deng_frame(self._rows())
        g = legacy_receiver_set_consensus(df)
        self.assertEqual(g.transNo.tolist(), [1, 2])
        self.assertAlmostEqual(g.loc[g.transNo == 1, "x"].iloc[0], 30.5)
        self.assertAlmostEqual(g.loc[g.transNo == 2, "x"].iloc[0], 200.5)


class TestPresentUnpositioned(unittest.TestCase):
    def _det(self, rows):
        return pd.DataFrame(rows, columns=["transNo", "Rec_ID", "seconds_fix"])

    def _sol(self, trans):
        return pd.DataFrame({"transNo": list(trans), "X": [1.0] * len(trans),
                             "Y": [1.0] * len(trans), "Z": [1.0] * len(trans)})

    def _cons(self, trans):
        return pd.DataFrame({"transNo": list(trans), "t0": [1.0] * len(trans),
                             "x": [1.0] * len(trans), "y": [1.0] * len(trans),
                             "z": [1.0] * len(trans)})

    def test_reasons(self):
        det = self._det([(10, "A", 100.0), (10, "B", 101.0),      # fixed -> omitted
                         (11, "A", 200.0), (11, "B", 201.0),      # all solutions outside box
                         (12, "A", 300.0), (12, "B", 301.0),      # solved, not consensus
                         (13, "A", 400.0)])                       # no solution at all
        sol = self._sol([11, 12])
        cons = self._cons([10])
        out = present_unpositioned_rows(det, sol, cons, box_rejected=[11])
        self.assertEqual(out.transNo.tolist(), [11.0, 12.0, 13.0])
        self.assertEqual(out.reason.tolist(), ["outside_3d_box", "not_consensus", "no_solution"])
        self.assertEqual(out.n_receivers.tolist(), [2, 2, 1])
        self.assertEqual(out.t0.tolist(), [200.5, 300.5, 400.0])

    def test_fixed_emissions_omitted(self):
        det = self._det([(10, "A", 100.0), (10, "B", 101.0)])
        out = present_unpositioned_rows(det, self._sol([10]), self._cons([10]))
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), ["transNo", "t0", "n_receivers", "reason"])

    def test_empty_detections(self):
        out = present_unpositioned_rows(pd.DataFrame(columns=["transNo", "Rec_ID", "seconds_fix"]),
                                        self._sol([1]), self._cons([]))
        self.assertTrue(out.empty)

    def test_box_rejected_priority_over_no_solution(self):
        # emission with NO solutions but flagged box-rejected can't happen from the
        # pipeline (rejected implies solutions existed); the helper must still not crash
        # and must prefer the more specific reason when both apply.
        det = self._det([(11, "A", 200.0)])
        out = present_unpositioned_rows(det, self._sol([11]), self._cons([]), box_rejected=[11])
        self.assertEqual(out.reason.tolist(), ["outside_3d_box"])


if __name__ == "__main__":
    unittest.main()
