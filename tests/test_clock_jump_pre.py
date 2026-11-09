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
from contextlib import closing
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import clock_jump_pre as cjp  # noqa: E402
import beacon_pairwise_dbscan as bpd  # noqa: E402

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
    con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN SourceRow INTEGER")
    con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN InternalCounterValue INTEGER")
    con.execute("UPDATE tblDetectionRaw SET SourceRow=rowid*2, "
                "InternalCounterValue=CASE WHEN OffsetChanged=1 THEN 0 ELSE 2 END")
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

    def test_compact_evidence_matches_original_even_when_tag_is_omitted(self):
        with closing(sqlite3.connect(self.db)) as con:
            original, evidence = cjp.find_events(con, "L1", T0, T0 + 10000)
            con.execute("CREATE TABLE tblClockEventEvidence AS SELECT * FROM tblDetectionRaw "
                        "WHERE OffsetChanged=1 OR CounterRestart=1 OR OneSecondAdjustmentEvidence=1")
            con.execute("DELETE FROM tblDetectionRaw WHERE OffsetChanged=1")
            compact, retained = cjp.find_events(con, "L1", T0, T0 + 10000)
            pd.testing.assert_frame_equal(original, compact)
            pd.testing.assert_frame_equal(evidence, retained)

    def test_grouping_preserves_boundaries_and_does_not_chain_or_merge_restarts(self):
        with closing(sqlite3.connect(self.db)) as con:
            candidates, _ = cjp.find_events(con, "L1", T0, T0 + 10000)
        row = candidates.iloc[0].to_dict()
        rows = [dict(row, seconds=T0 + d) for d in (0, 0.4, 0.8, 1.2)]
        rows.append(dict(row, seconds=T0 + 1.3, CounterRestart=1))
        rows.append(dict(row, seconds=T0 + 1.4, InternalOffsetValue=None))
        groups = cjp.group_repeated_triggers(pd.DataFrame(rows), 1.0)
        self.assertEqual(groups.trigger_count.tolist(), [3, 1, 1, 1])
        self.assertAlmostEqual(groups.iloc[0].event_start_s, T0)
        self.assertAlmostEqual(groups.iloc[0].event_end_s, T0 + 0.8)

    def test_refuses_second_application(self):
        with closing(sqlite3.connect(self.db)) as con:
            jumps, _, _, _ = cjp.quantify_jumps(
                con, ["L1"], "MSTR", "BC", PERIOD, T0, T0 + 10000)
        cjp.apply_offsets(self.db, jumps)
        with self.assertRaisesRegex(RuntimeError, "second application"):
            cjp.apply_offsets(self.db, jumps)

    def test_source_boundary_measurement_recovers_omitted_triggers_readonly(self):
        with closing(sqlite3.connect(self.db)) as con:
            sources = pd.read_sql("SELECT * FROM tblDetectionRaw WHERE Rec_ID='L1' AND OffsetChanged=1", con)
            sources["CounterContinuation"] = 0
            sources["one_sec_nearby"] = False
            con.execute("UPDATE tblDetectionRaw SET OffsetChanged=0")
            con.commit()
            before = con.total_changes
            missing, _, _, _ = cjp.quantify_jumps(
                con, ["L1"], "MSTR", "BC", PERIOD, T0, T0 + 10000)
            recovered, _, _, _ = cjp.quantify_jumps(
                con, ["L1"], "MSTR", "BC", PERIOD, T0, T0 + 10000, source_events=sources)
            self.assertEqual(con.total_changes, before)
        self.assertEqual(missing, [])
        self.assertEqual(len(recovered), 2)
        self.assertEqual([row["source_row"] for row in recovered], sources.SourceRow.tolist())
        np.testing.assert_allclose([row["offset_s"] for row in recovered], [JUMP1_SIZE, JUMP2_SIZE])
        self.assertTrue(all(row["boundary_method"] == "original_source_state" for row in recovered))

    def test_source_boundary_measurement_refuses_missing_or_duplicate_provenance(self):
        with closing(sqlite3.connect(self.db)) as con:
            with self.assertRaisesRegex(ValueError, "lack original boundary provenance"):
                cjp.quantify_jumps(con, ["L1"], "MSTR", "BC", PERIOD, T0, T0 + 10000,
                                   source_events=pd.DataFrame())
            sources = pd.read_sql("SELECT * FROM tblDetectionRaw WHERE Rec_ID='L1' AND OffsetChanged=1", con)
            sources["CounterContinuation"] = 0
            sources["one_sec_nearby"] = False
            with self.assertRaisesRegex(ValueError, "duplicate boundary rows"):
                cjp.quantify_jumps(con, ["L1"], "MSTR", "BC", PERIOD, T0, T0 + 10000,
                                   source_events=pd.concat([sources, sources], ignore_index=True))

    def test_conservative_report_does_not_shift_raw_and_marks_grouped_boundary_unresolved(self):
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("INSERT INTO tblDetectionRaw SELECT Rec_ID,'OTHER',seconds+0.4,"
                        "OffsetChanged,CounterRestart,OneSecondAdjustmentEvidence,InternalOffsetValue,SourceFile "
                        ",SourceRow+1,1 "
                        "FROM tblDetectionRaw WHERE Rec_ID='L1' AND OffsetChanged=1")
            con.commit()
            before = con.execute("SELECT sum(seconds) FROM tblDetectionRaw").fetchone()[0]
        notes, jumps = cjp.run(
            self.db, ["L1"], "MSTR", "BC", PERIOD, WIN_START, WIN_END,
            apply=False, group_span_s=1.0, conservative=True, persist_diagnostics=True)
        self.assertEqual(jumps, [])
        self.assertEqual(notes["unresolvable"], 2)
        with closing(sqlite3.connect(self.db)) as con:
            self.assertEqual(con.execute("SELECT sum(seconds) FROM tblDetectionRaw").fetchone()[0], before)
            unresolved = con.execute(
                "SELECT status,trigger_count,n_pre,n_post,note FROM tblClockJumpCandidates").fetchall()
            self.assertTrue(all(r[0] == "unresolved" and r[1] == 2 and min(r[2:4]) >= 3 for r in unresolved))
            self.assertTrue(all("boundary is uncertain" in r[4] for r in unresolved))

    def test_restart_episode_requires_adjacent_source_rows_and_zero_to_one(self):
        with closing(sqlite3.connect(self.db)) as con:
            candidates, _ = cjp.find_events(con, "L1", T0, T0+10000, include_provenance=True)
        first = dict(candidates.iloc[0], CounterRestart=1, SourceRow=100, InternalCounterValue=0)
        second = dict(first, seconds=first["seconds"]+0.4, SourceRow=101, InternalCounterValue=1)
        grouped = cjp.group_repeated_triggers(pd.DataFrame([first, second]), 1)
        self.assertEqual(grouped.trigger_count.tolist(), [2])
        self.assertEqual(grouped.group_kind.tolist(), ["restart_0_to_1"])
        self.assertEqual(grouped.iloc[0].event_start_s, first["seconds"])
        for changes in ({"SourceRow": 102}, {"InternalCounterValue": 0}, {"SourceFile": "other.csv"},
                        {"InternalOffsetValue": "changed"}):
            separate = cjp.group_repeated_triggers(pd.DataFrame([first, dict(second, **changes)]), 1)
            self.assertEqual(separate.trigger_count.tolist(), [1, 1])

    def test_rapid_events_remain_unresolved_under_existing_median_rule(self):
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("UPDATE tblDetectionRaw SET OffsetChanged=1 "
                        "WHERE Rec_ID='L1' AND Tag_ID='BC' AND seconds>? AND seconds<?",
                        (JUMP1_AT+JUMP1_SIZE, JUMP2_AT+JUMP1_SIZE+JUMP2_SIZE))
            con.commit()
            jumps, evidence, unresolved, _ = cjp.quantify_jumps(
                con, ["L1"], "MSTR", "BC", PERIOD, T0, T0+10000)
        self.assertEqual(jumps, [])
        self.assertEqual(evidence, [])
        self.assertEqual(len(unresolved), 5)
        self.assertTrue(all("window clipped below 150 s" in row["note"] for row in unresolved))

    def rapid_fixture(self, step=2.0):
        events = pd.DataFrame({"SourceFile": ["master.csv"]*3,
                               "SourceRow": [10, 20, 30], "seconds": [100., 160., 220.]})
        observations = pd.DataFrame([
            {"reference_host": host, "SourceFile": "master.csv", "SourceRow": source_row,
             "tdoa_s": value, "reference_seconds": time, "period": 60.}
            for host in ("A", "B", "C")
            for source_row, value, time in ((15, 0., 130.), (25, step, 190.))])
        references = pd.DataFrame(columns=["Rec_ID", "seconds"])
        return events, observations, references

    def test_rapid_consensus_accepts_both_directions_at_source_boundary(self):
        for step in (2., -2.):
            events, observations, references = self.rapid_fixture(step)
            result, support = cjp.rapid_reference_consensus(
                events, observations, references, "M", ["A", "B", "C"])
            middle = result[result.source_row == 20].iloc[0]
            self.assertEqual(middle.status, "accepted")
            self.assertEqual(middle.offset_s, step)
            self.assertEqual(middle.n_references, 3)
            self.assertEqual(len(support[support.source_row == 20]), 3)

    def test_sparse_isolated_consensus_is_opt_in_and_keeps_reference_guards(self):
        events, observations, references = self.rapid_fixture()
        events = events.iloc[[1]].copy()
        disabled, _ = cjp.rapid_reference_consensus(
            events, observations, references, "M", ["A", "B", "C"])
        self.assertTrue(disabled.empty)
        result, _ = cjp.rapid_reference_consensus(
            events, observations, references, "M", ["A", "B", "C"], include_isolated=True)
        self.assertEqual(result.status.tolist(), ["accepted"])
        self.assertEqual(result.method.tolist(), ["source_three_beacon_consensus"])
        references = pd.DataFrame({"Rec_ID": ["B"], "seconds": [150.]})
        blocked, _ = cjp.rapid_reference_consensus(
            events, observations, references, "M", ["A", "B", "C"], include_isolated=True)
        self.assertEqual(blocked.status.tolist(), ["unresolved"])
        self.assertEqual(blocked.n_references.tolist(), [2])

    def test_sparse_small_step_requires_explicit_timing_budget_minimum(self):
        events, observations, references = self.rapid_fixture(step=0.000907)
        events = events.iloc[[1]].copy()
        default, _ = cjp.rapid_reference_consensus(
            events, observations, references, "M", ["A", "B", "C"], include_isolated=True)
        self.assertEqual(default.status.tolist(), ["unresolved"])
        trial, _ = cjp.rapid_reference_consensus(
            events, observations, references, "M", ["A", "B", "C"],
            include_isolated=True, min_step=0.0005)
        self.assertEqual(trial.status.tolist(), ["accepted"])
        self.assertAlmostEqual(trial.offset_s.iloc[0], 0.000907)

    def test_sparse_trial_is_reversible_and_never_mutates_observations(self):
        events, observations, references = self.rapid_fixture(step=0.000907)
        events = events.iloc[[1]].copy()
        observations["Rec_ID"] = "M"
        original = observations.copy(deep=True)
        trial = cjp.sparse_correction_trial(
            events, observations, references, "M", ["A", "B", "C"], "master.csv", 20)
        pd.testing.assert_frame_equal(observations, original)
        self.assertEqual(trial["summary"]["status"], "timing_trial_passed")
        self.assertLess(trial["summary"]["after_max_step_s"], 1e-12)
        self.assertEqual(trial["summary"]["observations_shifted"], 3)
        np.testing.assert_array_equal(trial["observations"].SourceRow, original.SourceRow)

    def test_sparse_trial_refuses_uncontested_reference_shortfall(self):
        events, observations, references = self.rapid_fixture(step=0.000907)
        events = events.iloc[[1]].copy()
        observations["Rec_ID"] = "M"
        references = pd.DataFrame({"Rec_ID": ["B"], "seconds": [150.]})
        with self.assertRaisesRegex(RuntimeError, "independently supported"):
            cjp.sparse_correction_trial(
                events, observations, references, "M", ["A", "B", "C"], "master.csv", 20)

    def test_rapid_disagreement_or_competing_reference_blocks(self):
        events, observations, references = self.rapid_fixture()
        observations.loc[(observations.reference_host == "C") & (observations.SourceRow == 25), "tdoa_s"] += .001
        result, _ = cjp.rapid_reference_consensus(events, observations, references, "M", ["A", "B", "C"])
        self.assertEqual(result[result.source_row == 20].iloc[0].status, "unresolved")
        events, observations, _ = self.rapid_fixture()
        references = pd.DataFrame({"Rec_ID": ["B"], "seconds": [150.]})
        result, _ = cjp.rapid_reference_consensus(events, observations, references, "M", ["A", "B", "C"])
        middle = result[result.source_row == 20].iloc[0]
        self.assertEqual(middle.status, "unresolved")
        self.assertEqual(middle.n_references, 2)

    def test_rapid_does_not_reuse_epoch_across_intermediate_event(self):
        events, observations, references = self.rapid_fixture()
        observations.loc[observations.SourceRow == 15, "SourceRow"] = 5
        result, _ = cjp.rapid_reference_consensus(events, observations, references, "M", ["A", "B", "C"])
        self.assertEqual(result[result.source_row == 20].iloc[0].status, "unresolved")
        with self.assertRaises(ValueError):
            cjp.rapid_reference_consensus(events, observations, references, "M", ["A", "A", "C"])

    def test_rapid_counter_only_target_event_blocks_bracket(self):
        events, observations, references = self.rapid_fixture()
        target_events = pd.concat([
            events, pd.DataFrame({"SourceFile": ["master.csv"], "SourceRow": [22], "seconds": [170.]})
        ], ignore_index=True)
        result, support = cjp.rapid_reference_consensus(
            events, observations, references, "M", ["A", "B", "C"], target_events=target_events)
        self.assertEqual(result[result.source_row == 20].iloc[0].status, "unresolved")
        self.assertTrue(all("another target" in note for note in support[support.source_row == 20].note))

    def test_documented_firmware_adjustments_require_counter_and_provenance(self):
        raw = pd.DataFrame({
            "Internal": ["0000SL 01E2 02 1007 0F8 A", "0000SL 01E2 02 >007 0F8 A",
                         "0000SL 01E2 02 A007 0F8 A"],
            "seconds": [123.823362, 125.823362, 124.823362],
            "ReceiverType": ["SR3017"]*3, "FirmwareVersion": ["10.62F"]*3,
            "FileFormatVersion": ["2.0"]*3,
        })
        self.assertEqual(cjp.firmware_record_adjustments(raw).tolist(), [-1., 1., 0.])
        for column, value in (("ReceiverType", "SR3001"), ("FirmwareVersion", "10.63"),
                              ("FileFormatVersion", "3.0"), ("seconds", 124.823362)):
            invalid = raw.copy()
            invalid.loc[0, column] = value
            with self.assertRaises(ValueError):
                cjp.firmware_record_adjustments(invalid)
        with self.assertRaisesRegex(ValueError, "lack.*provenance"):
            cjp.firmware_record_adjustments(raw.drop(columns="ReceiverType"))
        self.assertEqual(cjp.firmware_record_adjustments(raw.iloc[0:0]).tolist(), [])

    def test_documented_firmware_adjustment_handles_fifteen_second_wrap(self):
        raw = pd.DataFrame({
            "Internal": ["0000SL 0032 02 1007 0F8 A"],
            "seconds": [119.5], "ReceiverType": ["SR3017"], "FirmwareVersion": ["10.62F"],
            "FileFormatVersion": ["2.0"],
        })
        self.assertEqual(cjp.firmware_record_adjustments(raw).tolist(), [-1.])

    def test_documented_firmware_adjustment_certifies_original_time_after_in_place_jump(self):
        # In-place jump application (jump_offset_s) shifts the stored seconds away from the
        # raw file time; the group-2 counter was recorded against the ORIGINAL time.
        original = pd.DataFrame({
            "Internal": ["0000SL 01E2 02 1007 0F8 A", "0000SL 01E2 02 >007 0F8 A"],
            "seconds": [123.823362, 125.823362],
            "ReceiverType": ["SR3017"]*2, "FirmwareVersion": ["10.62F"]*2,
            "FileFormatVersion": ["2.0"]*2,
        })
        self.assertEqual(cjp.firmware_record_adjustments(original).tolist(), [-1., 1.])
        jump = 3.676344
        shifted = original.copy()
        shifted["seconds"] = shifted.seconds - jump
        with self.assertRaisesRegex(ValueError, "15-second counter"):
            cjp.firmware_record_adjustments(shifted)
        counter = shifted.seconds + jump
        self.assertEqual(
            cjp.firmware_record_adjustments(shifted, counter_seconds=counter).tolist(),
            [-1., 1.])

    def joint_fixture(self, simultaneous=False):
        receivers = ["M", "A", "B", "C"]
        boundaries = {"M": 120., "A": 120. if simultaneous else 240.,
                      "B": 120. if simultaneous else 360., "C": 120. if simultaneous else 480.}
        jumps = {"M": 1., "A": -.4, "B": .2, "C": -.1}
        events = pd.DataFrame([
            {"Rec_ID": rec, "SourceFile": rec+".csv", "SourceRow": int(boundaries[rec]),
             "seconds": boundaries[rec]} for rec in receivers])
        rows = []
        for host in receivers:
            for rec in receivers:
                if rec == host:
                    continue
                for time in range(30, 691, 30):
                    rows.append({
                        "Rec_ID": rec, "SourceFile": rec+".csv", "SourceRow": time,
                        "reference_host": host, "reference_file": host+".csv", "reference_row": time,
                        "beacon_tag": host+"_beacon", "normalized_t": float(time),
                        "transNo": time//30,
                        "normalized_reference_t": float(time), "period": 30.,
                        "value": .001*(receivers.index(rec)-receivers.index(host))
                                 + jumps[rec]*(time >= boundaries[rec])-jumps[host]*(time >= boundaries[host]),
                    })
        return events, pd.DataFrame(rows), jumps

    def test_joint_recovers_identifiable_steps_with_independent_beacon_holdouts(self):
        events, observations, jumps = self.joint_fixture()
        result, support, notes = cjp.joint_event_reconstruction(events, observations, "M")
        self.assertEqual(set(result.status), {"supported_joint_measurement"})
        for row in result.itertuples():
            self.assertAlmostEqual(row.offset_s, jumps[row.Rec_ID], places=6)
            self.assertGreaterEqual(row.validated_beacon_brackets, 3)
        self.assertTrue(support.identifiable.all())
        self.assertEqual(notes["status"], "diagnostic_only_not_applied")

    def test_joint_never_labels_common_simultaneous_steps_identifiable(self):
        events, observations, _ = self.joint_fixture(simultaneous=True)
        result, _, _ = cjp.joint_event_reconstruction(events, observations, "M")
        self.assertFalse(result.identifiable.any())
        self.assertTrue(result.offset_s.isna().all())
        self.assertEqual(set(result.status), {"unresolved"})

    def test_joint_disagreeing_heldout_beacon_blocks(self):
        events, observations, _ = self.joint_fixture()
        observations.loc[(observations.Rec_ID == "M") & (observations.beacon_tag == "A_beacon")
                         & (observations.SourceRow >= 120), "value"] += .01
        result, _, _ = cjp.joint_event_reconstruction(events, observations, "M")
        self.assertEqual(result[result.Rec_ID == "M"].iloc[0].status, "unresolved")

    def test_joint_does_not_hide_disagreement_behind_median_of_receivers(self):
        events, observations, _ = self.joint_fixture()
        observations.loc[(observations.Rec_ID == "M") & (observations.beacon_tag == "A_beacon")
                         & (observations.SourceRow >= 120), "value"] += .003
        result, support, _ = cjp.joint_event_reconstruction(events, observations, "M")
        failed = support[(support.Rec_ID == "M") & (support.heldout_max_residual_step_s > .0005)]
        self.assertFalse(failed.empty)
        self.assertTrue(failed.worst_control_receiver.notna().all())
        self.assertTrue((failed.worst_pre_target_source_row < failed.source_row).all())
        self.assertTrue((failed.worst_post_target_source_row >= failed.source_row).all())
        self.assertEqual(result[result.Rec_ID == "M"].iloc[0].status, "unresolved")

    def test_joint_shared_host_reception_delay_is_not_a_receiver_jump(self):
        events, observations, jumps = self.joint_fixture()
        observations.loc[(observations.beacon_tag == "A_beacon") & (observations.transNo == 3), "value"] -= .017
        result, _, _ = cjp.joint_event_reconstruction(events, observations, "M")
        self.assertEqual(set(result.status), {"supported_joint_measurement"})
        for row in result.itertuples():
            self.assertAlmostEqual(row.offset_s, jumps[row.Rec_ID], places=6)

    def test_segment_support_uses_existing_filter_without_fitting_clock_jumps(self):
        events, observations, _ = self.joint_fixture()
        baseline = cjp.beacon_event_segment_support(events, observations)
        observations.loc[(observations.beacon_tag == "A_beacon") & (observations.transNo == 3), "value"] += .017
        shared = cjp.beacon_event_segment_support(events, observations)
        pd.testing.assert_frame_equal(baseline, shared, check_exact=False, atol=1e-12, rtol=1e-12)
        self.assertIn("clean", set(baseline.dbscan_class))

    def test_segment_support_does_not_cluster_across_short_event_regimes(self):
        events, observations, _ = self.joint_fixture()
        short = observations[observations.transNo.isin([2, 3, 4, 5, 6])]
        classified = cjp.beacon_event_segment_support(events, short)
        sparse = classified[classified.segment_observations < 3]
        self.assertFalse(sparse.empty)
        self.assertEqual(set(sparse.dbscan_class), {"noise"})

    def test_joint_rejects_missing_and_ambiguous_provenance(self):
        events, observations, _ = self.joint_fixture()
        for column, value in (("SourceFile", "other.csv"), ("SourceRow", float("nan")),
                              ("value", float("inf")), ("period", 0.), ("transNo", float("nan"))):
            invalid = observations.copy()
            invalid.loc[0, column] = value
            with self.assertRaises(ValueError):
                cjp.joint_event_reconstruction(events, invalid, "M")
        with self.assertRaises(ValueError):
            cjp.joint_event_reconstruction(pd.concat([events, events.iloc[:1]]), observations, "M")

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


def build_supplement_db(path):
    """Master M jumps +1.0 s with one internal code; listener L1 jumps +0.5 s with none."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE tblDetectionRaw (Rec_ID TEXT, Tag_ID TEXT, seconds REAL, "
                "OffsetChanged INTEGER DEFAULT 0, CounterRestart INTEGER DEFAULT 0)")
    master_jump_k, listener_jump_k = 100, 160
    rows = []
    for k in range(240):
        t = T0 + k * PERIOD
        master = t + (1.0 if k > master_jump_k else 0.0)
        rows.append(("M", "BC", master, 0, 0))
        for i, rec in enumerate(["L1", "L2", "L3", "L4"]):
            own = 0.5 if rec == "L1" and k > listener_jump_k else 0.0
            rows.append((rec, "BC", t + 0.01 * (i + 1) + own, 0, 0))
    rows.append(("M", "X", T0 + master_jump_k * PERIOD + 31.0, 1, 0))
    rows.append(("L1", "FISH", T0 + listener_jump_k * PERIOD + 30.0, 0, 0))
    rows.append(("L1", "FISH", T0 + 200 * PERIOD + 0.5, 0, 0))
    con.executemany("INSERT INTO tblDetectionRaw VALUES (?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return master_jump_k, listener_jump_k


class TestTdoaSupplement(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="jsats3d_tdoa_supplement_")
        self.db = os.path.join(self.tmp.name, "test.db")
        self.master_k, self.listener_k = build_supplement_db(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def supplement(self):
        return cjp.tdoa_supplement(self.db, ["M", "L1", "L2", "L3", "L4"], "M", "BC", PERIOD,
                                   WIN_START, WIN_END)

    def test_dbscan_eps_controls_data_driven_regime_count(self):
        series = pd.DataFrame({
            "Rec_ID": ["L1"] * 6,
            "t_anchor": T0 + np.arange(6) * PERIOD,
            "original_s": T0 + np.arange(6) * PERIOD,
            "master_original_s": T0 + np.arange(6) * PERIOD,
            "delta_s": [0.0, 0.0, 0.0, 0.00075, 0.00075, 0.00075],
        })
        original = series.copy(deep=True)
        baseline = bpd.cluster(series, PERIOD)
        merged = bpd.cluster(series, PERIOD, eps=2.0)
        baseline_regimes = baseline[baseline.dbscan_class == "clean"].label.nunique()
        merged_regimes = merged[merged.dbscan_class == "clean"].label.nunique()
        self.assertEqual(baseline_regimes, 2)
        self.assertEqual(merged_regimes, 1)
        pd.testing.assert_frame_equal(series, original)

    def test_sensitivity_reports_regimes_without_mutating_input(self):
        series = pd.DataFrame({
            "Rec_ID": ["L1"] * 6,
            "t_anchor": T0 + np.arange(6) * PERIOD,
            "original_s": T0 + np.arange(6) * PERIOD,
            "master_original_s": T0 + np.arange(6) * PERIOD,
            "delta_s": [0.0, 0.0, 0.0, 0.00075, 0.00075, 0.00075],
        })
        original = series.copy(deep=True)
        known = pd.DataFrame({"Rec_ID": ["L1"], "jump_time_s": [T0+2.5*PERIOD], "offset_s": [0.00075]})
        result = cjp.tdoa_supplement_sensitivity(
            series, PERIOD, [1.0, 2.0], [3], min_step=0.0005, known_events=known)
        self.assertEqual(result.clean_regimes.tolist(), [2, 1])
        self.assertEqual(result.in_range_step_candidates.tolist(), [1, 0])
        self.assertEqual(result.known_events_matched.tolist(), [1, 0])
        pd.testing.assert_frame_equal(series, original)

    def test_multibeacon_fallback_requires_heldout_supported_clean_segment(self):
        times = T0 + np.arange(40, dtype=float) * PERIOD
        master = np.linspace(0.01, 0.02, len(times))
        series = pd.DataFrame({
            "t": times, "label": [0]*40, "delta_s": master + 0.003,
            "master_residual": master, "master_supported": [True]*40,
            "dbscan_class": ["clean"]*40,
        })
        estimate, summary = cjp.validated_multibeacon_curve(series, [times[10], times[35]])
        self.assertEqual(len(summary), 1)
        self.assertTrue(np.isfinite(estimate).all())
        self.assertAlmostEqual(summary[0]["holdout_p95_ms"], 0.0, places=8)
        unsupported = series.assign(master_supported=False)
        missing, no_summary = cjp.validated_multibeacon_curve(unsupported, [times[10]])
        self.assertTrue(np.isnan(missing[0]))
        self.assertEqual(no_summary, [])

    def test_multibeacon_consensus_requires_two_within_budget(self):
        estimates = np.array([[0.0100, 0.0100, 0.0100, np.nan],
                      [0.0102, 0.0110, np.nan, np.nan],
                      [np.nan, 0.0101, np.nan, np.nan]])
        center, accepted, counts, spread = cjp.combine_multibeacon_estimates(estimates)
        self.assertEqual(accepted.tolist(), [True, False, False, False])
        self.assertEqual(counts.tolist(), [2, 3, 1, 0])
        self.assertAlmostEqual(center[0], 0.0101)
        self.assertTrue(np.isnan(center[1:]).all())

    def test_master_then_listener_jumps_without_codes_are_removed(self):
        notes = self.supplement()
        self.assertEqual((notes["master_jumps"], notes["receiver_jumps"]), (1, 1))
        self.assertEqual(notes["dbscan"], {
            "eps": 1.0, "min_samples": 3, "window_periods": 2.5, "budget_s": 0.0005})
        self.assertEqual(notes["unresolved_residual"], 0)
        with closing(sqlite3.connect(self.db)) as con:
            arrivals = cjp.beacon_first_arrivals(con, "M", "BC", PERIOD, T0 - 10, T0 + 10 ** 6)
            ledger = pd.read_sql("SELECT * FROM tblClockJumpEvents ORDER BY jump_time_s", con)
            fish = con.execute("SELECT seconds, jump_offset_s FROM tblDetectionRaw "
                               "WHERE Tag_ID='FISH' ORDER BY seconds").fetchall()
        for rec, g in arrivals.groupby("Rec_ID"):
            self.assertLess(np.ptp(g.tdoa_s), 1e-6, rec)
        master, listener = ledger.iloc[0], ledger.iloc[1]
        self.assertEqual((master.Rec_ID, listener.Rec_ID), ("M", "L1"))
        self.assertAlmostEqual(master.offset_s, 1.0, places=6)
        self.assertAlmostEqual(listener.offset_s, 0.5, places=6)
        self.assertAlmostEqual(master.jump_time_s, T0 + self.master_k * PERIOD + 31.0)
        self.assertFalse(bool(master.boundary_uncertain))
        self.assertTrue(bool(listener.boundary_uncertain))
        self.assertAlmostEqual(master.corrected_after_s, master.jump_time_s - 1.0)
        # Fish inside the uncoded gap is not shifted; fish after it is corrected.
        self.assertEqual(fish[0][1], 0.0)
        self.assertAlmostEqual(fish[1][0], T0 + 200 * PERIOD)

    def test_uncertain_boundary_blocks_fish_in_strict_gate_window(self):
        self.supplement()
        with closing(sqlite3.connect(self.db)) as con:
            unresolved = pd.read_sql("SELECT * FROM tblClockJumpCandidates WHERE status='unresolved'", con)
        self.assertEqual(unresolved.Rec_ID.tolist(), ["L1"])
        fish = T0 + self.listener_k * PERIOD + 30.0
        self.assertTrue(unresolved.window_start_s.iloc[0] < fish < unresolved.window_end_s.iloc[0])

    def test_close_distinct_triggers_do_not_certify_a_single_boundary(self):
        low = T0 + self.listener_k * PERIOD
        with closing(sqlite3.connect(self.db)) as con:
            con.executemany("INSERT INTO tblDetectionRaw VALUES ('L1','X',?,1,0)",
                            [(low + 30.0,), (low + 30.4,)])
            step = {"original_start_s": low, "original_end_s": low + PERIOD,
                    "step_s": 0.5}
            record = cjp._supplement_record(con, "L1", step, "receiver", 1)
        self.assertTrue(record["boundary_uncertain"])
        self.assertEqual(record["n_internal_evidence"], 2)
        self.assertAlmostEqual(record["window_start_s"], low + 30.0)
        self.assertAlmostEqual(record["window_end_s"], low + 30.4)

    def test_trigger_evidence_uses_original_time_after_corrections(self):
        low = T0 + self.listener_k * PERIOD
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN jump_offset_s REAL DEFAULT 0")
            con.execute("INSERT INTO tblDetectionRaw VALUES ('L1','X',?,1,0,2)", (low + 28,))
            evidence = cjp._supplement_evidence(con, "L1", low + 29, low + 31)
        np.testing.assert_array_equal(evidence, [low + 30])

    def test_refuses_second_application(self):
        self.supplement()
        with self.assertRaisesRegex(RuntimeError, "already applied"):
            self.supplement()

    def test_common_mode_requires_three_agreeing_listeners(self):
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("DELETE FROM tblDetectionRaw WHERE Rec_ID IN ('L3','L4')")
            con.commit()
        notes = self.supplement()
        self.assertEqual(notes["master_jumps"], 0)

    def test_simultaneous_listener_group_cannot_be_a_second_master_jump(self):
        # L2-L4 also jump together with the master but by a different amount (own +0.3 s).
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("INSERT INTO tblDetectionRaw SELECT 'L5',Tag_ID,seconds+0.04,0,0 "
                        "FROM tblDetectionRaw WHERE Rec_ID='L4' AND Tag_ID='BC'")
            con.execute("INSERT INTO tblDetectionRaw SELECT 'L6',Tag_ID,seconds+0.01,0,0 "
                        "FROM tblDetectionRaw WHERE Rec_ID='L5' AND Tag_ID='BC'")
            con.execute("INSERT INTO tblDetectionRaw SELECT 'L7',Tag_ID,seconds+0.01,0,0 "
                        "FROM tblDetectionRaw WHERE Rec_ID='L6' AND Tag_ID='BC'")
            con.execute("UPDATE tblDetectionRaw SET seconds=seconds+0.3 WHERE Rec_ID IN ('L2','L3','L4') "
                        "AND Tag_ID='BC' AND seconds>?", (T0 + self.master_k * PERIOD + 30,))
            con.commit()
        notes = cjp.tdoa_supplement(self.db, ["M", "L1", "L2", "L3", "L4", "L5", "L6", "L7"], "M", "BC",
                                    PERIOD, WIN_START, WIN_END)
        self.assertEqual(notes["master_jumps"], 1)
        with closing(sqlite3.connect(self.db)) as con:
            ledger = pd.read_sql("SELECT Rec_ID, offset_s FROM tblClockJumpEvents", con)
        self.assertAlmostEqual(ledger[ledger.Rec_ID == "M"].offset_s.iloc[0], 1.0, places=6)
        self.assertEqual(sorted(ledger.Rec_ID), ["L1", "L2", "L3", "L4", "M"])

    def test_long_gap_and_small_fragments_do_not_hide_master_jump(self):
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("DELETE FROM tblDetectionRaw WHERE Rec_ID='L4' AND Tag_ID='BC' AND seconds BETWEEN ? AND ?",
                        (T0 + 60 * PERIOD - 1, T0 + 140 * PERIOD + 1))
            con.execute("UPDATE tblDetectionRaw SET seconds=seconds+0.01 WHERE Rec_ID='L2' AND seconds>?",
                        (T0 + 50 * PERIOD + 1,))
            con.commit()
        notes = self.supplement()
        self.assertEqual((notes["master_jumps"], notes["receiver_jumps"]), (1, 1))
        with closing(sqlite3.connect(self.db)) as con:
            corrected = {r[0] for r in con.execute("SELECT Rec_ID FROM tblClockJumpEvents")}
        self.assertEqual(corrected, {"M", "L1"})

    def test_revert_restores_unshifted_rows_exactly(self):
        import shutil
        source = os.path.join(self.tmp.name, "source.db")
        shutil.copy2(self.db, source)
        with closing(sqlite3.connect(source)) as con:
            con.execute("CREATE TABLE tblMetronomeFiltered (x INTEGER)")
            con.commit()
        self.supplement()
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("INSERT INTO tblClockJumpCandidates (Rec_ID,stage,status,note) "
                        "VALUES ('L2',NULL,'accepted','stage-1 review candidate')")
            con.commit()
            self.assertGreater(con.execute(
                "SELECT count(*) FROM tblClockJumpCandidates WHERE stage=2").fetchone()[0], 0)
        result = cjp.revert_tdoa_supplement(self.db, source)
        self.assertEqual(result["removed_stage2_events"], 2)
        with closing(sqlite3.connect(self.db)) as con:
            con.execute("ATTACH DATABASE ? AS src", (source,))
            self.assertEqual(con.execute(
                "SELECT count(*) FROM tblDetectionRaw d JOIN src.tblDetectionRaw s ON s.rowid=d.rowid "
                "WHERE d.seconds != s.seconds OR d.jump_offset_s != 0").fetchone()[0], 0)
            self.assertEqual(con.execute("SELECT count(*) FROM tblClockJumpEvents").fetchone()[0], 0)
            self.assertTrue(con.execute(
                "SELECT 1 FROM src.sqlite_master WHERE name='tblMetronomeFiltered'").fetchone())
            self.assertEqual(con.execute(
                "SELECT Rec_ID,status,note FROM tblClockJumpCandidates WHERE stage IS NULL"
            ).fetchall(), [("L2", "accepted", "stage-1 review candidate")])
            self.assertEqual(con.execute(
                "SELECT count(*) FROM tblClockJumpCandidates WHERE stage=2").fetchone()[0], 0)
        self.supplement()
        with self.assertRaisesRegex(RuntimeError, "nothing to revert"):
            cjp.revert_tdoa_supplement(source, source)

    def test_supplement_accepts_explicit_dbscan_parameters(self):
        notes = cjp.tdoa_supplement(
            self.db, ["M", "L1", "L2", "L3", "L4"], "M", "BC", PERIOD,
            WIN_START, WIN_END, dbscan_eps=1.25, dbscan_min_samples=4,
            dbscan_window_periods=3.0)
        self.assertEqual(notes["dbscan"], {
            "eps": 1.25, "min_samples": 4, "window_periods": 3.0, "budget_s": 0.0005})


if __name__ == "__main__":
    unittest.main()
