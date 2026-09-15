"""Synthetic contract tests for 2025 feature and accounting invariants."""
import unittest

import numpy as np
import pandas as pd

from jsats3d.pipeline_mode import select_pipeline_mode, to_legacy_detection_shape
from jsats3d.multipath_interface import (
    CallableMultipathFilter,
    UnfilteredBaseline,
    require_features,
)
from scripts.measure_tag_intervals import measure_intervals
from scripts.adapt_2025_to_legacy import (
    apply_tag_pulse_rates,
    drop_incomplete_receivers,
    normalize_detection,
    parse_beacon_window,
)
from jsats3d.sync_readiness import assess_sync_readiness


def add_epoch_features(detections):
    """Build parameter-free diagnostic features for synthetic contract tests."""
    result = detections.sort_values(["tag_id", "receiver_id", "epoch_id", "timestamp"]).copy()
    group = result.groupby(["tag_id", "receiver_id", "epoch_id"], sort=False)
    result["lag_seconds"] = group["timestamp"].transform(lambda values: values - values.min())
    result["relative_amplitude"] = group["amplitude"].transform(lambda values: values - values.max())
    result["epoch_rank"] = group.cumcount()
    result["inter_detection_seconds"] = group["timestamp"].diff()
    return result


class Test2025Contracts(unittest.TestCase):
    def setUp(self):
        self.detections = pd.DataFrame({
            "tag_id": ["FFD3"] * 4,
            "receiver_id": ["R01"] * 4,
            "epoch_id": [1, 1, 1, 2],
            "timestamp": [10.0, 10.2, 10.8, 13.3],
            "amplitude": [180.0, 210.0, 190.0, 200.0],
        })

    def test_features_are_group_local(self):
        features = add_epoch_features(self.detections)
        epoch_one = features[features.epoch_id == 1]
        self.assertTrue(np.allclose(epoch_one.lag_seconds, [0.0, 0.2, 0.8]))
        self.assertTrue(np.allclose(epoch_one.relative_amplitude, [-30.0, 0.0, -20.0]))
        self.assertEqual(epoch_one.epoch_rank.tolist(), [0, 1, 2])

    def test_inter_detection_interval_preserves_epoch_boundary_as_missing(self):
        features = add_epoch_features(self.detections)
        boundary = features[features.epoch_id == 2].inter_detection_seconds.iloc[0]
        self.assertTrue(pd.isna(boundary))

    def test_missing_legacy_measurements_remain_null(self):
        detections = self.detections.assign(SNR=np.nan, NBW=np.nan, FreqOff=np.nan)
        self.assertEqual(int(detections.SNR.notna().sum()), 0)
        self.assertEqual(int(detections.NBW.notna().sum()), 0)
        self.assertEqual(int(detections.FreqOff.notna().sum()), 0)

    def test_auto_mode_selects_ats_2025_for_reduced_schema(self):
        data = pd.DataFrame({
            "dateTime": ["2025-06-11 12:00:00.000001"],
            "tagCode": ["FFD3"],
            "amp": [212],
            "receiverName": ["ZOI01"],
        })
        mode = select_pipeline_mode(data)
        self.assertEqual(mode.name, "ats_2025")
        self.assertIn("SNR", mode.missing_fields)

    def test_populated_legacy_fields_select_legacy(self):
        data = pd.DataFrame({
            "dateTime": ["2025-06-11 12:00:00.000001"],
            "tagCode": ["TAG1"],
            "amp": [212],
            "receiverName": ["R01"],
            "SNR": [10.0],
            "NBW": [2.0],
            "FreqOff": [0.1],
        })
        self.assertEqual(select_pipeline_mode(data).name, "legacy")

    def test_actual_legacy_column_names_select_legacy(self):
        data = pd.DataFrame({
            "timeStamp": ["2019-06-11 12:00:00.000001"],
            "Tag_ID": ["TAG1"],
            "Rec_ID": ["R01"],
            "SNR": [10.0],
            "NBW": [2.0],
            "FreqOff": [0.1],
        })
        self.assertEqual(select_pipeline_mode(data).name, "legacy")

    def test_legacy_request_fails_for_ats_data(self):
        data = self.detections.rename(columns={
            "timestamp": "dateTime", "tag_id": "tagCode", "amplitude": "amp",
            "receiver_id": "receiverName",
        })
        with self.assertRaises(ValueError):
            select_pipeline_mode(data, requested="legacy")

    def test_2025_data_maps_to_legacy_shape_without_metrics(self):
        data = pd.DataFrame({
            "dateTime": ["2025-06-11 12:00:00.000001"],
            "tagCode": ["FFD3"], "amp": [212], "receiverName": ["ZOI01"],
        })
        result = to_legacy_detection_shape(data)
        self.assertEqual(result.loc[0, "Tag_ID"], "FFD3")
        self.assertEqual(result.loc[0, "Rec_ID"], "ZOI01")
        self.assertTrue(pd.isna(result.loc[0, "SNR"]))

    def test_interval_measurement_supports_multiple_tags(self):
        path = "tests/_synthetic_detections.csv"
        data = pd.DataFrame({
            "dateTime": ["2025-01-01 00:00:00", "2025-01-01 00:00:03", "2025-01-01 00:00:00", "2025-01-01 00:00:04"],
            "tagCode": ["A", "A", "B", "B"],
            "receiverName": ["R1", "R1", "R1", "R1"],
        })
        data.to_csv(path, index=False)
        try:
            summary = measure_intervals(path, ["A", "B"], 10)
        finally:
            import os
            os.remove(path)
        self.assertEqual(set(summary["tagCode"]), {"A", "B"})

    def test_unfiltered_baseline_is_explicit_and_reconciles_counts(self):
        result = UnfilteredBaseline().filter(self.detections)
        self.assertEqual(result.method, "unfiltered_baseline")
        self.assertEqual(result.input_count, result.retained_count)
        self.assertEqual(result.rejected_count, 0)

    def test_callable_filter_reconciles_rejections(self):
        filter_stage = CallableMultipathFilter(
            lambda data: data[data.amplitude >= 190], "synthetic_filter"
        )
        result = filter_stage.filter(self.detections)
        self.assertEqual(result.input_count, 4)
        self.assertEqual(result.retained_count, 3)
        self.assertEqual(result.rejected_count, 1)

    def test_missing_features_fail_before_filtering(self):
        with self.assertRaises(ValueError):
            require_features(self.detections, ["lag_seconds", "relative_amplitude"])

    def test_sync_readiness_rejects_uncovered_temperature(self):
        detections = pd.DataFrame({"Tag_ID": ["FFD3"], "Rec_ID": ["R01"], "seconds": [100.0]})
        receivers = pd.DataFrame({"Rec_ID": ["R01", "R02", "R03", "R04"], "X": [0, 1, 0, 1], "Y": [0, 0, 1, 1], "Z": [0, 0, 0, 1]})
        temperature = pd.DataFrame({"timeStamp": [pd.Timestamp("1970-01-01 00:02:00")], "C": [14.0]})
        epochs = pd.DataFrame({"Tag_ID": ["B1"] * 4, "Rec_ID": ["R01", "R02", "R03", "R04"], "seconds": [100.0] * 4, "transNo": [1] * 4})
        result = assess_sync_readiness(detections, receivers, temperature, epochs)
        self.assertFalse(result.ready)
        self.assertIn("temperature coverage does not span detection times", result.reasons)

    def test_sync_readiness_accepts_complete_synthetic_inputs(self):
        detections = pd.DataFrame({"Tag_ID": ["FFD3"], "Rec_ID": ["R01"], "seconds": [100.0]})
        receivers = pd.DataFrame({"Rec_ID": ["R01", "R02", "R03", "R04"], "X": [0, 1, 0, 1], "Y": [0, 0, 1, 1], "Z": [0, 0, 0, 1]})
        temperature = pd.DataFrame({"timeStamp": [pd.Timestamp("1970-01-01 00:01:00"), pd.Timestamp("1970-01-01 00:03:00")], "C": [14.0, 14.1]})
        epochs = pd.DataFrame({"Tag_ID": ["B1"] * 4, "Rec_ID": ["R01", "R02", "R03", "R04"], "seconds": [100.0] * 4, "transNo": [1] * 4})
        result = assess_sync_readiness(detections, receivers, temperature, epochs)
        self.assertTrue(result.ready)

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


if __name__ == "__main__":
    unittest.main()