"""Tests for formatting 2025 inputs into legacy-compatible tables."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.adapt_2025_to_legacy import (
    apply_tag_pulse_rates,
    drop_incomplete_receivers,
    load_temperature_string,
    load_receiver_gps,
    normalize_detection,
    parse_beacon_window,
    tag_types,
)
from scripts.parse_ats_raw_to_legacy import (
    DETECTION_DB_COLUMNS,
    discover_target_files,
    parse_detections,
    parse_gps_coordinates,
    parse_internal,
    parse_utc_offset,
    resolve_time_shift,
)
from scripts.extract_dbscan_features import extract_features
from scripts.beacon_pairwise_dbscan import classify, cluster
from scripts.legacy_pipeline import clock_fix_check, import_2019, widen_table
from scripts.cfd_gps_diagnostics import interpolate_positions
from jsats3d import position
from scripts.run_data import (
    check_inputs,
    check_output,
    finish_database,
    legacy_command,
    load_species_tags,
    normalize_acoustic_code,
    normalize_tags,
    parser_command,
)


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
            daily = root / "SR18078D250619_000101_cleaned.csv"
            for path in (original, cleaned, false_match, daily):
                path.write_text("", encoding="utf-8")
            result = discover_target_files(root, {"18078"})
            self.assertEqual(result, sorted([cleaned, daily]))

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

    def test_pairwise_dbscan_labels_steady_late_reflection(self):
        t = 1.75e9 + 62.7 * pd.Series(range(60), dtype=float)
        delta = pd.Series([0.001 + 0.0215 * (i % 2) for i in range(60)])
        series = pd.DataFrame({"Rec_ID": "R1", "t_anchor": t, "delta_s": delta, "burst_n": 1})
        result = cluster(series, 60.0)
        late = result[result.delta_s > 0.01]
        self.assertTrue((late.dbscan_class == "steady_reflection").all())
        self.assertTrue((result[result.delta_s < 0.01].dbscan_class == "clean").all())

    def test_pairwise_dbscan_sets_aside_anchor_side_epochs(self):
        t = 1.75e9 + 62.7 * pd.Series(range(30), dtype=float)
        frames = []
        for k in range(5):
            delta = pd.Series([0.001 * k] * 30)
            delta.iloc[12] -= 0.004
            frames.append(pd.DataFrame({"Rec_ID": "R%d" % k, "t_anchor": t, "delta_s": delta, "burst_n": 1}))
        result, suspects = classify(pd.concat(frames, ignore_index=True), 60.0)
        self.assertEqual(suspects, {t.iloc[12]})
        self.assertTrue((result[result.t_anchor == t.iloc[12]].dbscan_class == "anchor_suspect").all())
        self.assertEqual(int((result.dbscan_class == "noise").sum()), 0)

    def test_raw_parser_time_zone_offsets(self):
        self.assertEqual(parse_utc_offset("-07z"), -7.0)
        self.assertEqual(parse_utc_offset("-00z"), 0.0)
        self.assertEqual(parse_utc_offset("00z"), 0.0)
        self.assertEqual(resolve_time_shift(0.0, 0.0, "f"), (0.0, -7.0, "header"))
        self.assertEqual(resolve_time_shift(None, -7.0, "f"), (-7.0, 0.0, "config"))
        self.assertEqual(resolve_time_shift(0.0, -7.0, "f", gps_offset=-7.0), (-7.0, 0.0, "gps"))
        with self.assertRaises(ValueError):
            resolve_time_shift(0.0, -7.0, "f")
        with self.assertRaises(ValueError):
            resolve_time_shift(None, None, "f")

    def test_raw_parser_shifts_utc_receiver_to_study_basis(self):
        lines = [
            "Site Name: ZOI,05,05", "Serial Number: 18081", "ATS Sonic Receiver SR3017 Firmware v10.62F",
            "File Format Version: 2.0  ", "File Start: 06/18/2025 22:49:50 00z  *22EC+0647224930 5OFF", "",
            "4628.0190 N 12206.4862 W ,ZOI,05,05, 06/18/2025 22:50:10       , GPS Fix  ,  N/A, 13.16,  99.99,  N/A  , -99, 000 00/31, 000, ",
            "224930 04B5 49 09CF 3B4 B,ZOI,05,05, 06/18/2025 22:49:57.054972, G727F91E4,  N/A, 13.16,  99.99,  N/A  , 212, 240 13/31, 160, ",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SR18081_test.csv"
            path.write_text("\n".join(lines), encoding="utf-8")
            rows, gps, _ = parse_detections(path, "ZOI05", "SR3017", config_offset=0.0)
        row = dict(zip(DETECTION_DB_COLUMNS, rows[0]))
        self.assertEqual(row["timeStamp"], "2025-06-18 15:49:57.054972")
        self.assertEqual(row["RawDateTime"], "2025-06-18 22:49:57.054972")
        self.assertEqual(row["TimeShiftHours"], -7.0)
        self.assertEqual(row["GPSFixTimeStamp"], "2025-06-18 22:50:10+00:00")
        self.assertEqual(row["TimeZoneSource"], "header")
        self.assertEqual(gps, 1)

    def test_raw_parser_gps_offset_overrides_wrong_header(self):
        lines = [
            "Serial Number: 18078", "ATS Sonic Receiver SR3017 Firmware v10.62F", "File Format Version: 2.0  ",
            "File Start:  07/24/2025 13:02:20 00z  *FFFF+0747201029 5OFF", "",
            "4628.0219 N 12206.4979 W ,ZOI,2,2, 07/24/2025 8:03:00       , GPS Fix  ,  N/A,13.12,99.99,  N/A  ,-99,0      ,0, ",
        ]
        for second in range(10, 14):
            lines += [
                "130259 05D3 02 00EE C34 F,ZOI,2,2, 07/24/2025 13:03:%02d.913697, G725A8583,  N/A,13.12,99.99,  N/A  ,203,240  4/31,147, " % second,
                "4628.0219 N 12206.4979 W ,ZOI,2,2, 07/24/2025 20:03:%02d       , GPS Fix  ,  N/A,13.12,99.99,  N/A  ,-99,0      ,0, " % (second + 1),
            ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SR18078_test.csv"
            path.write_text("\n".join(lines), encoding="utf-8")
            rows, _, _ = parse_detections(path, "ZOI02", "SR3017", config_offset=-7.0)
        row = dict(zip(DETECTION_DB_COLUMNS, rows[0]))
        self.assertEqual(row["timeStamp"], "2025-07-24 13:03:10.913697")
        self.assertEqual(row["TimeZoneSource"], "gps")

    def test_adapter_tag_types_and_provisional_study_rates(self):
        registry = pd.DataFrame({"Rec_ID": ["ZOI02", None], "Tag_ID": ["7D2D", "1F14"], "pulseRate": [60.0, None]})
        tags = pd.DataFrame({"Tag_ID": ["7D2D", "1F14", "FC36", "FFD3"]})
        tags["TagType"] = tag_types(tags.Tag_ID, registry)
        self.assertEqual(tags.TagType.tolist(), ["beacon", "beacon", "study", "study"])
        rates = apply_tag_pulse_rates(tags, registry).set_index("Tag_ID").pulseRate
        self.assertEqual(rates["FC36"], 3.038)
        self.assertTrue(pd.isna(rates["1F14"]))

    def test_temperature_string_prefers_complete_hobo_then_string_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hobo = root / "DD_N"
            hobo.mkdir()
            for depth, value in (("0.5", 12.0), ("18", 10.0)):
                body = ["Plot Title: x", '#,"Date Time, GMT-07:00","Temp, C"',
                        "1,6/2/2025 13:45,%s" % value, "2,6/2/2025 13:50,%s" % value]
                if depth == "18":
                    body[3] = "2,6/2/2025 13:50,"
                (hobo / ("x_DD_N_%s.csv" % depth)).write_text("\n".join(body), encoding="latin-1")
            string = root / "string.csv"
            string.write_text("DateTime,DD_N_0p5,DD_N_1p5,DD_N_9,DD_N_18\n"
                              "2025-06-02 13:45:00,1,1,1,1\n2025-06-02 13:55:00,4,4,4,4\n", encoding="utf-8")
            result = load_temperature_string(str(string), str(hobo))
        self.assertEqual(result.C.tolist(), [11.0, 4.0])
        self.assertEqual(result.TempSource.tolist(), ["DD_N_HOBO", "DD_N_string"])


class TestRunData(unittest.TestCase):
    def test_clock_fix_check_refuses_to_replace_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "clock_fix.db"
            connection = sqlite3.connect(database)
            connection.execute("create table tblDetectionClockFixed (Rec_ID text)")
            connection.execute("insert into tblDetectionClockFixed values ('R01')")
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(ValueError, "already exists"):
                clock_fix_check(database, ["R01"])
            connection = sqlite3.connect(database)
            rows = connection.execute("select Rec_ID from tblDetectionClockFixed").fetchall()
            connection.close()
        self.assertEqual(rows, [("R01",)])

    def test_2019_import_uses_shared_runtime_and_legacy_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw" / "R01").mkdir(parents=True)
            pd.DataFrame({"Tag_ID": ["A1B2"], "TagType": ["study"], "pulseRate": [3.0]}).to_csv(root / "tags.csv", index=False)
            pd.DataFrame({"Rec_ID": ["R01"], "Type": ["Teknologic"], "Tag_ID": ["A1B2"], "Ref_Elev": ["BM"],
                          "X": [0.0], "Y": [0.0], "Z": [0.0], "X_t": [0.0], "Y_t": [0.0], "Z_t": [0.0]}).to_csv(root / "receivers.csv", index=False)
            pd.DataFrame({"timeStamp": ["2018-06-07 13:17:08"], "WSEL": [1.0]}).to_csv(root / "wsel.csv", index=False)
            pd.DataFrame({"meas_dt": ["2018-06-07 13:17:08", "2018-06-07 13:18:08"], "location": ["L", "L"],
                          "depth_ft": ["1", "1"], "temp_f": [50.0, 50.0]}).to_csv(root / "temp.csv", index=False)
            output_db = root / "out.db"
            study = {"utc_conv": -7, "bm_elev": 262.585, "bm_elev_units": "meters", "output_units": "meters",
                     "master_receiver": "R01", "synch_time_start": "2018-06-07 13:17:08",
                     "synch_time_end": "2018-06-07 13:18:08"}
            import_2019({"study": study}, {"raw_root": root / "raw", "tag_csv": root / "tags.csv",
                         "receiver_csv": root / "receivers.csv", "wsel_csv": root / "wsel.csv",
                         "temp_csv": root / "temp.csv", "output_db": output_db})
            connection = sqlite3.connect(output_db)
            tables = {row[0] for row in connection.execute("select name from sqlite_master")}
            connection.close()
        self.assertTrue({"tblTag", "tblReceiver", "tblWSEL", "tblInterpolatedTemp", "tblStudyParameters"} <= tables)

    def test_piecewise_gps_interpolation_supports_linear_cubic_and_no_extrapolation(self):
        gps = pd.DataFrame({"Rec_ID": ["R1"] * 4, "dateTime": pd.date_range("2025-01-01", periods=4, freq="h"),
                            "easting": [0.0, 1.0, 4.0, 9.0], "northing": [0.0, 0.0, 0.0, 0.0]})
        query = pd.DataFrame({"Rec_ID": ["R1"] * 3, "dateTime": pd.to_datetime(
            ["2024-12-31 23:00", "2025-01-01 01:30", "2025-01-01 05:00"])})
        linear = interpolate_positions(gps, query, method="linear")
        cubic = interpolate_positions(gps, query, method="cubic")
        self.assertTrue(linear.easting.iloc[[0, 2]].isna().all())
        self.assertAlmostEqual(linear.easting.iloc[1], 2.5)
        self.assertAlmostEqual(cubic.easting.iloc[1], 2.25)

    def test_receiver_gps_table_uses_receiver_coordinate_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gps.csv"
            pd.DataFrame({"receiverName": ["CFD04", "CFD04", "ZOI01"],
                          "dateTime": ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 00:00"],
                          "easting": [100.0, 110.0, 200.0], "northing": [1000.0, 1010.0, 2000.0]}).to_csv(path, index=False)
            receivers = pd.DataFrame({"Rec_ID": ["CFD04"], "easting": [90.0], "northing": [990.0]})
            result = load_receiver_gps(str(path), receivers)
        self.assertEqual(result.Rec_ID.tolist(), ["CFD04", "CFD04"])
        self.assertEqual(result.X.tolist(), [10.0, 20.0])

    def test_position_uses_dynamic_gps_and_static_fallback(self):
        solver = position.__new__(position)
        solver.ephemeris = pd.DataFrame({"Rec_ID": ["CFD04", "ZOI01"], "X_t": [0.0, 50.0],
                                         "Y_t": [0.0, 60.0], "Z_t": [-3.0, -4.0]})
        solver.dynamic_positions = pd.DataFrame({"Rec_ID": ["CFD04", "CFD04"], "seconds": [100.0, 200.0],
                                                  "X": [10.0, 30.0], "Y": [20.0, 40.0]})
        self.assertEqual(solver.receiver_position_at("CFD04", 150.0, -3.0).tolist(), [20.0, 30.0, -3.0])
        self.assertEqual(solver.receiver_position_at("ZOI01", 150.0, -4.0).tolist(), [50.0, 60.0, -4.0])
        with self.assertRaises(ValueError):
            solver.receiver_position_at("CFD04", 250.0, -3.0)

    def test_acoustic_codes_are_upper_cased_and_excel_damage_reversed_only_when_unique(self):
        self.assertEqual(normalize_acoustic_code("27a0"), ("27A0", None))
        self.assertEqual(normalize_acoustic_code("74")[0], "0074")
        self.assertEqual(normalize_acoustic_code("4.10E+01")[0], "41E0")
        self.assertEqual(normalize_acoustic_code("1.80E+05")[0], "18E4")
        self.assertEqual(normalize_acoustic_code("2.00E+39")[0], "2E39")
        self.assertEqual(normalize_acoustic_code("7.20E+07")[0], "72E6")
        self.assertIsNone(normalize_acoustic_code("0.00E+00")[0])
        self.assertIsNone(normalize_acoustic_code("1.00E+05")[0])  # 1E05, 01E5 and 10E4 all equal 1e5
        self.assertIsNone(normalize_acoustic_code("")[0])

    def test_species_lookup_uses_acoustic_column_case_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            released = Path(directory) / "released_v0.csv"
            released.write_text(
                "Tag Code,Acoustic Tag Value,Species Name\n"
                "3DD.1,27a0,Chinook\n3DD.2,,Coho\n3DD.3,4.10E+01,Chinook\n"
                "3DD.4,0.00E+00,Chinook\n3DD.5,B175,Coho\n", encoding="utf-8")
            codes, repaired, unresolved = load_species_tags(released, ["chinook"])
            self.assertEqual(codes, {"27A0", "41E0"})
            self.assertEqual(repaired, [("4.10E+01", "41E0")])
            self.assertEqual(len(unresolved), 1)
            with self.assertRaises(ValueError):
                load_species_tags(released, ["Steelhead"])

    def test_tags_must_be_four_hex(self):
        self.assertEqual(normalize_tags(["ffd3", " FC36 "]), {"FFD3", "FC36"})
        with self.assertRaises(ValueError):
            normalize_tags(["G72FFD3"])

    def test_output_refused_inside_raw_data_or_over_existing_db(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "2025_Data"
            paths = {}
            for name in ("raw_root", "hobo_dir"):
                paths[name] = data / name
                paths[name].mkdir(parents=True)
            for name in ("config_xlsx", "gps_csv", "covariate_csv", "temperature_csv", "released_file"):
                paths[name] = data / name / "file.csv"
                paths[name].parent.mkdir(parents=True)
                paths[name].write_text("x")
            with self.assertRaises(ValueError):
                check_output(data / "out.db", paths, overwrite=False)
            existing = Path(directory) / "repo" / "old.db"
            existing.parent.mkdir()
            existing.write_text("x")
            with self.assertRaises(ValueError):
                check_output(existing, paths, overwrite=False)
            check_output(existing, paths, overwrite=True)
            check_output(Path(directory) / "repo" / "new.db", paths, overwrite=False)

    def test_parser_command_adds_beacons_only_when_filtering_tags(self):
        paths = {name: Path(name) for name in (
            "raw_root", "config_xlsx", "output_db", "temperature_csv", "hobo_dir", "gps_csv", "covariate_csv")}
        filtered = parser_command(paths, "2025-06-17", "2025-07-01", ["18084"], {"FFD3", "27A0"})
        self.assertIn("--include-config-beacons", filtered)
        self.assertEqual([filtered[i + 1] for i, v in enumerate(filtered) if v == "--tag"], ["27A0", "FFD3"])
        self.assertEqual(filtered[filtered.index("--serial") + 1], "18084")
        everything = parser_command(paths, None, None, [], set())
        self.assertNotIn("--include-config-beacons", everything)
        self.assertNotIn("--tag", everything)
        self.assertNotIn("--start", everything)

    def test_teknologic_format_needs_its_own_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Aggregated").mkdir()
            paths = {"raw_root": root / "Aggregated", "output_db": root / "out.db"}
            for name in ("tag_csv", "receiver_csv", "wsel_csv", "temp_csv"):
                paths[name] = root / (name + ".csv")
                paths[name].write_text("x")
            check_inputs(paths, "teknologic")
            with self.assertRaises(ValueError):
                check_inputs(paths, "ats")

    def test_legacy_command_uses_env_name_or_folder(self):
        by_name = legacy_command({"env": "jsat_legacy"}, "process", "run.toml")
        self.assertTrue(by_name[1].endswith("scripts\\legacy_pipeline.py"))
        self.assertEqual(by_name[-2:], ["process", "run.toml"])
        by_path = legacy_command({"env": "C:\\envs\\jsat_legacy"}, "process", "run.toml")
        self.assertEqual(by_path, by_name)

    def test_finish_database_sets_study_rates_and_provisional_signal_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "t.db"
            con = sqlite3.connect(db)
            con.execute("create table tblDetectionRaw (Tag_ID, Rec_ID, SigStr, Threshold, BitPeriod, SNR, NBW)")
            con.execute("insert into tblDetectionRaw values ('A1B2','ZOI01',212,160,'240 13/31',NULL,NULL)")
            con.execute("create table tblTag (Tag_ID, TagType, pulseRate)")
            con.execute("insert into tblTag values ('A1B2','study',NULL)")
            con.commit()
            con.close()
            study = {"utc_conv": -7, "bm_elev": "", "master_receiver": "ZOI08"}
            finish_database(db, study, "2025-06-20", "2025-06-21", True, {"a1b2": 3.0})
            con = sqlite3.connect(db)
            self.assertEqual(con.execute("select SNR, NBW from tblDetectionRaw").fetchone(), (52, 240.0))
            self.assertEqual(con.execute("select pulseRate from tblTag").fetchone(), (3.0,))
            row = con.execute("select * from tblStudyParameters").fetchone()
            con.close()
        self.assertEqual(row, (-7, None, "feet", "meters", "ZOI08", "2025-06-20", "2025-06-21"))

    def test_widen_table_takes_every_csv_column_before_legacy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "csv"
            folder.mkdir()
            (folder / "a.csv").write_text("Rec_ID,multipath_prediction\n")
            (folder / "b.csv").write_text("Rec_ID,amp_s,multipath_prediction\nR1,0.5,0\n")
            db = str(Path(directory) / "t.db")
            widen_table(db, "tblMetronomeSecondFiltered", folder)
            con = sqlite3.connect(db)
            columns = [row[1] for row in con.execute("PRAGMA table_info(tblMetronomeSecondFiltered)")]
            con.close()
        self.assertEqual(columns, ["Rec_ID", "multipath_prediction", "amp_s"])


if __name__ == "__main__":
    unittest.main()