"""Tests for formatting 2025 inputs into legacy-compatible tables."""
import unittest
import tempfile
from pathlib import Path

import pandas as pd

from scripts.adapt_2025_to_legacy import (
    apply_tag_pulse_rates,
    drop_incomplete_receivers,
    normalize_detection,
    parse_beacon_window,
)
from scripts.parse_ats_raw_to_legacy import (
    discover_target_files,
    parse_gps_coordinates,
    parse_internal,
)
from scripts.extract_dbscan_features import extract_features
from scripts.beacon_pairwise_dbscan import cluster


class Test2025Adapter(unittest.TestCase):
    def test_adapter_sets_provisional_ffd3_rate_and_registry_rates(self):
        tags = pd.DataFrame({"Tag_ID": ["FFD3", "B1"], "TagTypeSource": ["study", "beacon"]})
        registry = pd.DataFrame({"Tag_ID": ["B1"], "pulseRate": [60.0]})
        result = apply_tag_pulse_rates(tags, registry)
        self.assertEqual(result.set_index("Tag_ID").loc["FFD3", "pulseRate"], 3.33)
        self.assertEqual(result.set_index("Tag_ID").loc["B1", "pulseRate"], 60.0)

    def test_adapter_beacon_window_selects_only_requested_tags_and_time(self):
        chunk = pd.DataFrame({
            "dateTime": ["2025-06-05 00:00:00", "2025-06-05 00:00:02", "2025-06-05 00:00:04"],
            "tagCode": ["B1", "B2", "B1"], "amp": [1, 2, 3],
            "receiverName": ["R1", "R1", "R1"], "event": [True, True, True],
        })
        result = normalize_detection(
            chunk,
            "beacon",
            beacon_window=parse_beacon_window(["2025-06-05 00:00:01", "2025-06-05 00:00:03"]),
            beacon_tags={"B2"},
        )
        self.assertEqual(result.Tag_ID.tolist(), ["B2"])

    def test_adapter_preserves_ats_extension_fields(self):
        chunk = pd.DataFrame({
            "dateTime": ["2025-06-05 00:00:00"],
            "tagCode": ["B1"], "amp": [210], "receiverName": ["R1"],
            "event": [True], "diagCode": ["GPS111 01A5 32 y000 7C0 F"],
            "sigStr": [210],
        })
        result = normalize_detection(chunk, "beacon")
        self.assertEqual(result.loc[0, "Internal"], "GPS111 01A5 32 y000 7C0 F")
        self.assertEqual(result.loc[0, "SigStr"], 210)
        self.assertTrue(result.loc[0, "Event"])
        self.assertTrue(pd.isna(result.loc[0, "SNR"]))

    def test_adapter_maps_raw_sigstr_without_losing_raw_fields(self):
        chunk = pd.DataFrame({
            "dateTime": ["2025-06-05 00:00:00.123456"],
            "tagCode": ["B1"], "sigStr": [207], "receiverName": ["ZOI02"],
            "diagCode": ["GPS111 01A5 32 y000 7C0 F"], "temp": [14.2],
            "pressure": [3.1], "tilt": [0.5], "vBatt": [12.4],
            "bitPeriod": [1.0], "threshold": [100],
            "receiverType": ["SR3017"], "firmwareVersion": ["10.62F"],
            "fileFormatVersion": ["2.0"], "sourceFile": ["raw.csv"],
            "sourceRow": [42],
        })
        result = normalize_detection(chunk, "beacon")
        self.assertEqual(result.loc[0, "Amplitude"], 207)
        self.assertEqual(result.loc[0, "SigStr"], 207)
        self.assertEqual(result.loc[0, "RawTemperature"], 14.2)
        self.assertEqual(result.loc[0, "SourceRow"], 42)
        self.assertTrue(pd.isna(result.loc[0, "SNR"]))

    def test_adapter_drops_incomplete_receivers_with_reasons(self):
        receivers = pd.DataFrame({
            "Rec_ID": ["R1", "R2"], "Tag_ID": ["B1", None],
            "X": [1.0, 2.0], "Y": [1.0, 2.0], "Z": [1.0, None],
            "X_t": [1.0, 2.0], "Y_t": [1.0, 2.0], "Z_t": [1.0, None],
            "Ref_Elev": ["BM", "BM"],
        })
        kept, dropped = drop_incomplete_receivers(receivers)
        self.assertEqual(kept.Rec_ID.tolist(), ["R1"])
        self.assertEqual(dropped.Rec_ID.tolist(), ["R2"])
        self.assertIn("Tag_ID", dropped.reason.iloc[0])

    def test_raw_parser_prefers_cleaned_file_and_exact_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "SR18078_250606.csv"
            cleaned = root / "SR18078_250606_cleaned.csv"
            false_match = root / "SR18078250610_121101_recovery.csv"
            for path in (original, cleaned, false_match):
                path.write_text("", encoding="utf-8")
            result = discover_target_files(root, {"18078"})
            self.assertEqual(result, [cleaned])

    def test_raw_parser_decodes_internal_groups(self):
        timestamp = pd.Timestamp("2025-06-06 08:54:07.952711")
        result = parse_internal("085443 031B 54 F000 858 F", timestamp)
        self.assertEqual(result["InternalCounter"], "000")
        self.assertEqual(result["InternalOffset"], "858")
        self.assertEqual(result["InternalStatus"], "F")
        self.assertFalse(result["OneSecondAdjustmentEvidence"])

    def test_raw_parser_converts_gps_coordinates(self):
        latitude, longitude = parse_gps_coordinates("4628.0191 N 12206.4983 W")
        self.assertAlmostEqual(latitude, 46.466985, places=6)
        self.assertAlmostEqual(longitude, -122.108305, places=6)

    def test_dbscan_features_are_diagnostic_and_epoch_local(self):
        import sqlite3
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as handle:
            path = handle.name
        try:
            connection = sqlite3.connect(path)
            data = pd.DataFrame({
                "Tag_ID": ["FFD3"] * 3,
                "Rec_ID": ["R01"] * 3,
                "seconds": [10.0, 10.2, 13.5],
                "timeStamp": ["t1", "t2", "t3"],
                "SigStr": [180, 210, 190],
            })
            data.to_sql("tblDetectionRaw", connection, index=False)
            connection.close()
            features = extract_features(path, "FFD3", 3.33)
            self.assertEqual(len(features), 3)
            self.assertEqual(features.epoch_number.tolist(), [1, 1, 2])
            self.assertEqual(features.epoch_rank.tolist(), [0, 1, 0])
            self.assertAlmostEqual(features.lag_seconds.iloc[1], 0.2)
            self.assertAlmostEqual(features.relative_sigstr.iloc[1], 0.0)
        finally:
            import os
            os.remove(path)

    def test_pairwise_dbscan_flags_late_outliers_and_splits_on_jump(self):
        t = 1.75e9 + 62.7 * pd.Series(range(40), dtype=float)
        delta = pd.Series([0.001 + 1e-6 * i for i in range(40)])
        delta.iloc[20:] += 1.0
        delta.iloc[[5, 30]] += 0.003
        series = pd.DataFrame({"Rec_ID": "R1", "t_anchor": t, "delta_s": delta, "burst_n": 1})
        result = cluster(series, 60.0)
        self.assertEqual(sorted(result.index[result.label < 0]), [5, 30])
        self.assertEqual(result.loc[result.label >= 0, "label"].nunique(), 2)
        self.assertEqual(len(result), 40)


if __name__ == "__main__":
    unittest.main()