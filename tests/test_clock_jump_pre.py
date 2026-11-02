"""Synthetic-data tests for scripts/clock_jump_pre.py (pre-clock-sync jump correction).

Builds a small SQLite DB with a master beacon receiver and two listeners: one with a
+1.0 s clock jump (flagged by an OffsetChanged row on the first post-jump beacon row) and
one with a flagged sync but no step (evidence-only). Verifies quantification and the
applied offsets.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import clock_jump_pre as cjp  # noqa: E402

T0 = cjp.naive_seconds("2025-07-01 00:00:00")
WIN_START, WIN_END = "2025-07-01 00:00:00", "2025-07-02 00:00:00"
PERIOD = 60.0
JUMP1_AT, JUMP1_SIZE = T0 + 34 * PERIOD, 1.0    # exactly on the beacon epoch grid
JUMP2_AT, JUMP2_SIZE = T0 + 38 * PERIOD, 0.5    # second jump, +0.5 s


def build_db(path):
    """Master MSTR (beacon BC every 60 s, no jump). Listener L1 jumps +1.0 s at JUMP1_AT
    and +0.5 s at JUMP2_AT (flagged on the first post-jump beacon row of each). Listener
    L2 has a flagged sync, no step."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE tblDetectionRaw (Rec_ID TEXT, Tag_ID TEXT, seconds REAL,
        OffsetChanged INTEGER DEFAULT 0, CounterRestart INTEGER DEFAULT 0,
        OneSecondAdjustmentEvidence INTEGER DEFAULT 0, InternalOffsetValue TEXT,
        SourceFile TEXT)""")
    con.execute("CREATE TABLE tblMetronomeFiltered (x INTEGER)")
    con.execute("INSERT INTO tblMetronomeFiltered VALUES (1)")
    rows = []
    for k in range(60):  # 60 beacon epochs, 3600 s
        t = T0 + k * PERIOD
        rows.append(("MSTR", "BC", t, 0, 0, 0, "100", "mstr.txt"))
        # L1: clock runs +1.0 s after JUMP1_AT, +1.5 s after JUMP2_AT
        shift = 0.0
        if t >= JUMP1_AT:
            shift += JUMP1_SIZE
        if t >= JUMP2_AT:
            shift += JUMP2_SIZE
        flag = 1 if t in (JUMP1_AT, JUMP2_AT) else 0
        rows.append(("L1", "BC", t + shift, flag, 0, 0, "200", "l1.txt"))
        # L2: no jump, one flagged sync row
        rows.append(("L2", "BC", t, 1 if t == JUMP1_AT else 0, 0, 0, "300", "l2.txt"))
    # a fish row on L1 after both jumps (recorded +1.5 s)
    rows.append(("L1", "FISH", T0 + 3_000 + JUMP1_SIZE + JUMP2_SIZE, 0, 0, 0, "200", "l1.txt"))
    con.executemany("INSERT INTO tblDetectionRaw VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


class TestClockJumpPre(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jsats3d_jump_pre_")
        self.db = os.path.join(self.tmp.name, "test.db")
        build_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_find_events_candidates_and_corroboration(self):
        con = sqlite3.connect("file:%s?mode=ro" % self.db, uri=True)
        cands, all_ev = cjp.find_events(con, "L1", T0, T0 + 10_000)
        con.close()
        self.assertEqual(len(cands), 2)
        # the jump-2 flag sits on the first post-jump row, already shifted by BOTH jumps
        self.assertEqual([float(x) for x in cands.seconds],
                         [JUMP1_AT + JUMP1_SIZE, JUMP2_AT + JUMP1_SIZE + JUMP2_SIZE])
        self.assertTrue(bool(cands.iloc[0]["candidate"]))
        self.assertEqual(len(all_ev), 2)

    def test_quantify_accepts_jump_and_evidence_only(self):
        con = sqlite3.connect("file:%s?mode=ro" % self.db, uri=True)
        jumps, evidence_only, unresolvable, n_events = cjp.quantify_jumps(
            con, ["L1", "L2", "MSTR"], "MSTR", "BC", PERIOD, T0, T0 + 10_000,
            window_s=300.0, min_step=0.05, max_step=5.0)
        con.close()
        self.assertGreater(n_events, 0)
        self.assertEqual(len(jumps), 2)
        offsets = {j["jump_time_s"]: j["offset_s"] for j in jumps}
        self.assertAlmostEqual(offsets[JUMP1_AT + JUMP1_SIZE], JUMP1_SIZE, places=3)
        self.assertAlmostEqual(offsets[JUMP2_AT + JUMP1_SIZE + JUMP2_SIZE], JUMP2_SIZE, places=3)
        for j in jumps:
            self.assertEqual(j["Rec_ID"], "L1")
            self.assertGreaterEqual(j["n_pre"], 3)
            self.assertGreaterEqual(j["n_post"], 3)
        self.assertEqual(len(evidence_only), 1)
        self.assertEqual(evidence_only[0]["Rec_ID"], "L2")
        self.assertEqual(len(unresolvable), 0)

    def test_apply_offsets_shifts_post_jump_rows_only(self):
        con = sqlite3.connect("file:%s?mode=ro" % self.db, uri=True)
        jumps, _, _, _ = cjp.quantify_jumps(con, ["L1", "L2", "MSTR"], "MSTR", "BC", PERIOD,
                                            T0, T0 + 10_000, 300.0, 0.05, 5.0)
        con.close()
        self.assertEqual(len(jumps), 2)
        rows, affected = cjp.apply_offsets(self.db, jumps)
        self.assertEqual(affected, ["L1"])
        self.assertEqual(rows, 27)  # beacons k=34..59 (26) + the fish row
        con = sqlite3.connect("file:%s?mode=ro" % self.db, uri=True)
        try:
            # rows between the jumps shifted by jump 1 only; after both, by the cumulative 1.5 s
            between = con.execute("SELECT seconds FROM tblDetectionRaw WHERE Rec_ID='L1' AND Tag_ID='BC' AND seconds >= ? AND seconds < ?",
                                  (JUMP1_AT, JUMP2_AT)).fetchall()
            self.assertAlmostEqual(min(x[0] for x in between), JUMP1_AT, places=6)
            post2 = con.execute("SELECT seconds FROM tblDetectionRaw WHERE Rec_ID='L1' AND Tag_ID='BC' AND seconds >= ?",
                                (JUMP2_AT,)).fetchall()
            self.assertAlmostEqual(min(x[0] for x in post2), JUMP2_AT, places=6)
            # pre-jump rows untouched, jump_offset_s = 0
            pre = con.execute("SELECT seconds FROM tblDetectionRaw WHERE Rec_ID='L1' AND Tag_ID='BC' AND seconds < ?",
                              (JUMP1_AT,)).fetchall()
            self.assertAlmostEqual(pre[-1][0], JUMP1_AT - PERIOD, places=6)
            pre_off = con.execute("SELECT DISTINCT jump_offset_s FROM tblDetectionRaw WHERE Rec_ID='L1' AND seconds < ?",
                                  (JUMP1_AT,)).fetchall()
            self.assertEqual(pre_off, [(0.0,)])
            # fish row after both jumps shifted back by the cumulative offset
            fish = con.execute("SELECT seconds, jump_offset_s FROM tblDetectionRaw WHERE Rec_ID='L1' AND Tag_ID='FISH'").fetchone()
            self.assertAlmostEqual(fish[0], T0 + 3_000, places=6)
            self.assertAlmostEqual(fish[1], JUMP1_SIZE + JUMP2_SIZE, places=6)
            # master rows untouched (zero offset, times unchanged)
            mstr = con.execute("SELECT count(*), max(abs(jump_offset_s)), max(seconds) FROM tblDetectionRaw WHERE Rec_ID='MSTR'").fetchone()
            self.assertEqual(mstr[0], 60)
            self.assertEqual(mstr[1], 0.0)
            self.assertAlmostEqual(mstr[2], T0 + 59 * PERIOD, places=6)
            # event table written
            ev = con.execute("SELECT Rec_ID, offset_s FROM tblClockJumpEvents ORDER BY jump_time_s").fetchall()
            self.assertEqual(len(ev), 2)
            self.assertAlmostEqual(ev[0][1], JUMP1_SIZE, places=3)
            self.assertAlmostEqual(ev[1][1], JUMP2_SIZE, places=3)
            # metronome tables invalidated
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("tblMetronomeFiltered", tables)
        finally:
            con.close()

    def test_no_jumps_is_noop(self):
        # master-only receiver list: no listener has a step, so nothing is applied
        notes, jumps = cjp.run(self.db, ["MSTR"], "MSTR", "BC", PERIOD,
                               WIN_START, WIN_END, apply=True)
        self.assertEqual(jumps, [])
        self.assertEqual(notes["n_jumps"], 0)
        self.assertEqual(notes["rows_shifted"], 0)
        con = sqlite3.connect("file:%s?mode=ro" % self.db, uri=True)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("tblClockJumpEvents", tables)
        self.assertIn("tblMetronomeFiltered", tables)  # untouched when nothing is applied
        con.close()


if __name__ == "__main__":
    unittest.main()
