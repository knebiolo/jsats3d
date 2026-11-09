"""2025 tag drag workflow after the build: paper steps 2-11 on a legacy-format database, one command, resumable.

    python scripts/run_data.py config/run_data_tagdrag_ent.toml                   step 1 (parse raw files, study parameters)
    python scripts/_2025_pipeline.py config/run_data_tagdrag_ent.toml      everything after step 1
    options: --db PATH (other database, own work folder)  --only NAME ...  --from NAME  --to NAME  --list
             --init-from DB [--swap REC_A REC_B]  what-if copy of a finished database, optionally with two hydrophone positions exchanged

Run order and names (paper step in brackets):
    prepare (1b)  clock_jumps (1c, experimental: apply_clock_jumps)  metronome (2-3)  clock_surface (4)
    deep_beacons (5)  deep_positions (6)  adopt_deep (6b)  deep_clocks (7)
    fish_multipath (8)  speed_of_sound (9)  positions (11)  export (10)  positions_2d (optional)  plot_drag (12)
    plot_drag_2d (12c, optional)
A step whose output tables already exist is skipped, so a rerun resumes. Each step checks its own result and a status
record is written to output/<database stem>.pipeline_status.json. K: is only read (water level file); everything is written
to C:. Deep coordinates use the configured legacy Deng root median (default B), then phase 2 reruns metronome and every
receiver clock. Fish use legacy primary ranking followed by a separate ATS timing classifier (ATS lacks NBW/SNR).
Known-depth Deng2D is optional and is not used for the final unknown-depth accuracy plots.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # legacy code calls plt.show() per receiver; a GUI backend blocks the run

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import closing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beacon_pairwise_dbscan as bpd  # noqa: E402
import legacy_pipeline as lp  # noqa: E402
sys.path.insert(0, str(lp.REPO))  # for `from jsats3d import bathymetry`

warnings.filterwarnings("ignore")
jsats3d = lp.jsats3d
from jsats3d.jsats3d import _clock_experiment_enabled, _clock_supported_interpolation

# Forebay minus collector-entrance mean(S, N): median over the overlap 2025-06-17 00:05-11:10 (journal 2026-10-02).
WSE_OFFSET_FT = 0.8905
# Step 8 screen: a first arrival later than this against the rolling median of its own (tag, receiver) series is a
# suspected reflection (window in transmissions). The boat moves range about 1.2 ms per ping; reflections sit several ms late.
SUSPECT_WINDOW = 9
SUSPECT_LATE_S = 0.003


# Drag truth is used only after Deng has solved XYZ.
DRAG_LINES = ("ENT-01", "ENT-02", "ENT-03", "ENT-04", "ENT-05")


def table_names(db):
    return set(lp.query(db, "select name from sqlite_master where type = 'table'").name)


def count(db, table, where="", params=()):
    if table not in table_names(db):
        return 0
    return int(lp.query(db, "select count(*) n from %s %s" % (table, where), params).n.iloc[0])


def need_rows(db, table, where="", params=(), minimum=1):
    n = count(db, table, where, params)
    if n < minimum:
        raise RuntimeError("check failed: %s has %d rows %s (expected at least %d)" % (table, n, where, minimum))
    return n










# ----------------------------------------------------------------------------------------------- steps

def prepare_done(c):
    t = table_names(c.db)
    first = lp.query(c.db, "select min(timeStamp) t from tblWSEL").t.iloc[0] if "tblWSEL" in t else None
    return "tblReceiver_initial" in t and first is not None and str(first) <= c.window_start


def prepare_run(c):
    """Keep the config receiver table and give the window a water level (tblWSEL starts 2025-06-17 in the raw build)."""
    if "tblReceiver_initial" not in table_names(c.db):
        lp.execute(c.db, "create table tblReceiver_initial as select * from tblReceiver")
    first = lp.query(c.db, "select min(timeStamp) t from tblWSEL").t.iloc[0]
    wse = Path(c.paths["hobo_dir"]).parent / "TagDrag_WSE.xlsx"
    sheet = pd.read_excel(wse, sheet_name="Samples")
    sheet.columns = ["ts", "S", "N"]
    sheet["WSEL"] = sheet[["S", "N"]].mean(axis=1, skipna=False) + WSE_OFFSET_FT
    sheet = sheet.dropna(subset=["WSEL"])
    sheet = sheet[sheet.ts < pd.Timestamp(first)]
    rows = list(zip(sheet.ts.dt.strftime("%Y-%m-%d %H:%M:%S"), sheet.WSEL.astype(float)))
    con = sqlite3.connect(c.db)
    con.executemany("insert into tblWSEL(timeStamp, WSEL) values (?,?)", rows)
    con.commit()
    con.close()
    dup = lp.query(c.db, "select count(*) - count(distinct timeStamp) n from tblWSEL").n.iloc[0]
    in_window = need_rows(c.db, "tblWSEL", "where timeStamp >= '%s' and timeStamp <= '%s'" % (c.window_start, c.window_end))
    if dup:
        raise RuntimeError("check failed: %d duplicate tblWSEL timestamps" % dup)
    return {"wsel_rows_inserted": len(rows), "wsel_rows_in_window": in_window}


def review_gate(db, tags, receivers, start, end, output, check_curves=True):
    """Check original event windows and corrected-time coverage without changing data.

    Counts are potentially affected raw detections, not bad positions. A conservative
    event window only localizes evidence; it does not certify a jump's exact boundary.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    curve_warnings = []
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "tblClockJumpCandidates" not in tables:
            raise RuntimeError("Experimental jump candidate ledger is missing; run clock_jumps before timing review")
        master = con.execute("SELECT masterReceiver FROM tblStudyParameters").fetchone()[0]
        cols = {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}
        original = "seconds+coalesce(jump_offset_s,0)" if "jump_offset_s" in cols else "seconds"
        placeholders = ",".join("?" for _ in tags)
        detections = pd.read_sql(
            "SELECT Rec_ID,Tag_ID,seconds,%s AS original_seconds FROM tblDetectionRaw "
            "WHERE Tag_ID IN (%s) AND %s>=? AND %s<=?"
            % (original, placeholders, original, original), con, params=[*tags, start, end])
        unresolved = (pd.read_sql("SELECT * FROM tblClockJumpCandidates WHERE status='unresolved'", con)
                      if "tblClockJumpCandidates" in tables else pd.DataFrame())
        validation = (pd.read_sql("SELECT * FROM tblClockJumpValidation WHERE status!='passed'", con)
                      if "tblClockJumpValidation" in tables else pd.DataFrame())
        for rec in receivers:
            local = detections[detections.Rec_ID == rec]
            for event in unresolved[unresolved.Rec_ID == rec].to_dict("records") if not unresolved.empty else []:
                affected = local[(local.original_seconds >= event["window_start_s"])
                                 & (local.original_seconds <= event["window_end_s"])]
                rows.append({
                    "Rec_ID": rec, "reason": event["note"], "time_basis": "original_receiver",
                    "start_s": event["window_start_s"], "end_s": event["window_end_s"],
                    "raw_fish_rows": len(affected),
                    "tags": ";".join(sorted(affected.Tag_ID.unique())),
                })
            if not check_curves or local.empty or rec == master:
                continue
            for event in validation[validation.Rec_ID == rec].to_dict("records") if not validation.empty else []:
                affected = local[(local.seconds >= event["start_s"]) & (local.seconds <= event["end_s"])]
                rows.append({
                    "Rec_ID": rec, "reason": "applied jump validation " + event["status"],
                    "time_basis": "corrected_receiver", "start_s": event["start_s"],
                    "end_s": event["end_s"], "raw_fish_rows": len(affected),
                    "tags": ";".join(sorted(affected.Tag_ID.unique())),
                })
            clock = (pd.read_sql(
                "SELECT seconds,seconds_residual FROM tblDetectionClockFixed "
                "WHERE Rec_ID=? ORDER BY seconds", con, params=[rec])
                if "tblDetectionClockFixed" in tables else pd.DataFrame())
            if clock.empty or len(clock) < 2:
                bad = local
                reason = "missing clock coverage"
            else:
                clock = clock.drop_duplicates("seconds")
                if len(clock) < 2:
                    bad = local
                    reason = "insufficient distinct clock knots"
                else:
                    period = con.execute(
                        "SELECT t.pulseRate FROM tblTag t JOIN tblReceiver r ON t.Tag_ID=r.Tag_ID "
                        "JOIN tblStudyParameters s ON r.Rec_ID=s.masterReceiver"
                    ).fetchone()
                    if period is None:
                        raise ValueError("Master beacon period is missing")
                    _, supported = _clock_supported_interpolation(
                        local.seconds.to_numpy(), clock.seconds.to_numpy(),
                        clock.seconds_residual.to_numpy(), float(period[0]))
                    ts = clock.seconds.to_numpy(float)
                    rs = clock.seconds_residual.to_numpy(float)
                    bad_knots = np.zeros(len(ts), dtype=bool)
                    if len(ts) >= 3:
                        predicted = rs[:-2] + (rs[2:] - rs[:-2]) * (ts[1:-1] - ts[:-2]) / (ts[2:] - ts[:-2])
                        error = rs[1:-1] - predicted
                        bad_knots[1:-1] = np.abs(error) > 0.0005
                        for k in np.flatnonzero(bad_knots):
                            curve_warnings.append({
                                "Rec_ID": rec, "seconds": ts[k], "start_s": ts[k-1], "end_s": ts[k+1],
                                "neighbor_loo_error_s": float(error[k-1]),
                                "note": "Timing warning, not independent truth or an approved correction.",
                            })
                    right = np.clip(np.searchsorted(ts, local.seconds.to_numpy()), 1, len(ts)-1)
                    supported &= ~(bad_knots[right-1] | bad_knots[right])
                    bad = local[~supported]
                    reason = "outside clock coverage, unsupported gap/step, or neighbor LOO warning above 0.5 ms"
            if not bad.empty:
                rows.append({
                    "Rec_ID": rec, "reason": reason, "time_basis": "corrected_receiver",
                    "start_s": float(bad.seconds.min()), "end_s": float(bad.seconds.max()),
                    "raw_fish_rows": len(bad), "tags": ";".join(sorted(bad.Tag_ID.unique())),
                })
    columns = ["Rec_ID", "reason", "time_basis", "start_s", "end_s", "raw_fish_rows", "tags"]
    report = pd.DataFrame(rows, columns=columns)
    report.to_csv(output / "clock_review_gate.csv", index=False)
    pd.DataFrame(curve_warnings, columns=["Rec_ID", "seconds", "start_s", "end_s",
                                        "neighbor_loo_error_s", "note"]).to_csv(
        output / "clock_curve_warnings.csv", index=False)
    summary = {
        "status": "blocked" if any(r["raw_fish_rows"] for r in rows) else "passed",
        "check_curves": check_curves, "issues": len(rows),
        "receivers_with_fish_impact": sorted({r["Rec_ID"] for r in rows if r["raw_fish_rows"]}),
        "note": "Raw-detection counts overlap between issues and are not counts of inaccurate solutions. "
                "Curve issue start/end summarize affected-row ranges, not a single continuous gap.",
    }
    (output / "clock_review_gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["status"] == "blocked":
        raise RuntimeError("Clock review required: unresolved timing affects fish; see %s"
                           % (output / "clock_review_gate.csv"))
    return summary

def clock_jumps_done(c):
    if not c.run["legacy"].get("apply_clock_jumps", False):
        return True
    return count(c.db, lp.PROGRESS_TABLE, "where Tag_ID = ?", ("__ats_clock_jumps__",)) > 0


def clock_review_done(c):
    return c.clock_review_interval is None


def clock_source_boundary_review(con, receiver, start, end, folder):
    """Recover all-tag source state transitions, including rows omitted during parsing.

    Internal position differences are diagnostic only: they are not calibrated offsets.
    """
    import csv
    from datetime import timedelta
    import clock_jump_pre
    from parse_ats_raw_to_legacy import (
        INTERNAL_IDX, DT_IDX, TAG_IDX, _fast_timestamp, parse_internal, parse_tag_code,
    )
    columns = {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}
    if not {"SourceFile", "TimeShiftHours"} <= columns:
        return {"status": "unavailable", "reason": "database lacks source-file/time-shift provenance"}
    files = con.execute(
        "SELECT DISTINCT SourceFile,TimeShiftHours FROM tblDetectionRaw "
        "WHERE Rec_ID=? AND seconds BETWEEN ? AND ?", (receiver, start-300, end+300)).fetchall()
    shifts = {}
    for filename, shift in files:
        if filename in shifts and shifts[filename] != shift:
            raise ValueError("Conflicting raw time shifts for %s" % filename)
        shifts[filename] = shift
    if not shifts:
        return {"status": "unavailable", "reason": "no original source files cover the interval"}
    rows = []
    for filename, shift in shifts.items():
        path = Path(filename)
        if not path.is_file() or shift is None:
            raise ValueError("Raw source file or time-shift provenance missing: %s" % path)
        prior = None
        gps_only_since_prior = False
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for source_row, values in enumerate(csv.reader(stream), start=1):
                if len(values) < 13 or not values[TAG_IDX].strip().startswith("G72"):
                    gps_only_since_prior = (
                        len(values) >= 13 and values[TAG_IDX].strip() == "GPS Fix"
                        and prior is not None
                        and (prior["SourceRow"] + 1 == source_row or gps_only_since_prior))
                    continue
                stamp = _fast_timestamp(values[DT_IDX])
                if stamp is None:
                    gps_only_since_prior = False
                    continue
                internal = parse_internal(values[INTERNAL_IDX], stamp)
                if internal is None:
                    gps_only_since_prior = False
                    continue
                seconds = clock_jump_pre.naive_seconds(stamp + timedelta(hours=float(shift)))
                state = (internal["InternalOffsetValue"], internal["InternalGroup3"],
                         internal["InternalPositionDifferenceSeconds"], internal["InternalFlag"])
                changed = prior is None or prior["state"] != state
                offset_changed = prior is not None and prior["state"][0] != state[0]
                counter_restart = internal["InternalCounter"] in ("000", "001")
                continuation = (internal["InternalCounterValue"] == 1 and prior is not None
                                and prior["counter"] == 0
                                and (prior["SourceRow"] + 1 == source_row or gps_only_since_prior)
                                and prior["state"][0] == state[0]
                                and prior["status"] == internal["InternalStatus"])
                if start-300 <= seconds <= end+300 and (changed or counter_restart):
                    rows.append({
                        "Rec_ID": receiver, "SourceFile": filename, "SourceRow": source_row,
                        "seconds": seconds, "wall_time": str(stamp + timedelta(hours=float(shift))),
                        "Tag_ID": parse_tag_code(values[TAG_IDX]), "Internal": values[INTERNAL_IDX],
                        "InternalOffsetValue": state[0], "InternalGroup3": state[1],
                        "position_difference_s": state[2], "flag": state[3],
                        "counter": internal["InternalCounterValue"],
                        "InternalCounterValue": internal["InternalCounterValue"],
                        "OffsetChanged": int(offset_changed), "CounterRestart": int(counter_restart),
                        "CounterContinuation": int(continuation),
                        "ContinuationAcrossGPS": int(continuation and gps_only_since_prior),
                        "one_sec_nearby": bool(internal["OneSecondAdjustmentEvidence"]),
                        "previous_SourceRow": prior["SourceRow"] if prior else None,
                        "previous_seconds": prior["seconds"] if prior else None,
                        "previous_offset": prior["state"][0] if prior else None,
                        "previous_group3": prior["state"][1] if prior else None,
                        "previous_position_difference_s": prior["state"][2] if prior else None,
                        "note": "Source state evidence only; position difference is not an approved clock offset.",
                    })
                prior = {"state": state, "SourceRow": source_row, "seconds": seconds,
                         "counter": internal["InternalCounterValue"], "status": internal["InternalStatus"]}
                gps_only_since_prior = False
    if not rows:
        return {"status": "unavailable", "reason": "no parseable original Internal evidence covers the interval"}
    pd.DataFrame(rows).to_csv(folder / ("%s_source_state_boundaries.csv" % receiver), index=False)
    return {"status": "reviewed", "files": len(shifts), "state_boundaries": len(rows)}


def clock_source_median_review(con, receiver, start, end, folder, reference_host=None,
                               anchor_receiver=None):
    """Measure original source events against a recorded beacon without applying offsets."""
    import clock_jump_pre
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source = clock_source_boundary_review(con, receiver, start, end, folder)
    if source["status"] != "reviewed":
        raise RuntimeError("Original source boundaries unavailable for %s: %s" % (receiver, source))
    events = pd.read_csv(folder / ("%s_source_state_boundaries.csv" % receiver))
    if reference_host is None:
        reference_host = con.execute("SELECT masterReceiver FROM tblStudyParameters").fetchone()[0]
    anchor_receiver = reference_host if anchor_receiver is None else anchor_receiver
    if receiver == anchor_receiver:
        raise ValueError("Master events require independent beacon evidence, not self-comparison")
    metadata = con.execute(
        "SELECT r.Tag_ID,t.pulseRate FROM tblReceiver r JOIN tblTag t ON t.Tag_ID=r.Tag_ID "
        "WHERE r.Rec_ID=?", (reference_host,)).fetchone()
    if metadata is None:
        raise ValueError("Missing recorded beacon metadata for %s" % reference_host)
    tag, period = metadata
    anchor_events = None
    if anchor_receiver != reference_host:
        anchor_review = clock_source_boundary_review(con, anchor_receiver, start, end, folder)
        if anchor_review["status"] != "reviewed":
            raise RuntimeError("Alternate reference source boundaries unavailable for %s" % anchor_receiver)
        anchor_states = pd.read_csv(folder / ("%s_source_state_boundaries.csv" % anchor_receiver))
        anchor_events = anchor_states[(anchor_states.OffsetChanged == 1)
                                      | ((anchor_states.CounterRestart == 1)
                                         & (anchor_states.CounterContinuation == 0))]
    measured, evidence, unresolved, count = clock_jump_pre.quantify_jumps(
        con, [receiver], anchor_receiver, tag, float(period), start-300, end+300,
        group_span_s=1.0, source_events=events)
    records = ([dict(row, measurement_status="measured") for row in measured]
               + [dict(row, measurement_status="evidence_only") for row in evidence]
               + [dict(row, measurement_status="unresolved") for row in unresolved])
    report = pd.DataFrame(records)
    if not report.empty:
        report = report[(report.jump_time_s >= start) & (report.jump_time_s <= end)]
        report["reference_host"], report["beacon_tag"] = reference_host, tag
        report["anchor_receiver"] = anchor_receiver
        if anchor_events is not None:
            for index, row in report.iterrows():
                if pd.notna(row.get("reference_start_s")) and pd.notna(row.get("reference_end_s")):
                    competing = anchor_events[anchor_events.seconds.between(
                        row.reference_start_s, row.reference_end_s, inclusive="both")]
                    report.loc[index, "competing_anchor_events"] = len(competing)
                    if not competing.empty:
                        report.loc[index, "measurement_status"] = "unresolved"
                        report.loc[index, "note"] = "alternate listening reference has source events inside bracket"
    path = folder / ("%s_source_median_review.csv" % receiver)
    report.to_csv(path, index=False)
    summary = {"status": "diagnostic_only", "receiver": receiver, "reference_host": reference_host,
               "anchor_receiver": anchor_receiver,
               "source_review": source, "source_state_rows": count,
               "measurement_status_counts": report.measurement_status.value_counts().to_dict()
                   if not report.empty else {}, "report": str(path),
               "warning": "Measurements are not approved offsets. Reference events, plateau stability, "
                          "source-order application and fish impact still require validation."}
    (folder / ("%s_source_median_manifest.json" % receiver)).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _clock_paired_provenance(con, paired, tag, host, start, end):
    import clock_jump_pre
    columns = {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}
    offset_expr = "coalesce(jump_offset_s,0)" if "jump_offset_s" in columns else "0"
    raw = pd.read_sql(
        "SELECT Rec_ID,seconds,%s AS jump_offset_s,SourceRow,SourceFile,Internal,"
        "ReceiverType,FirmwareVersion,FileFormatVersion "
        "FROM tblDetectionRaw WHERE Tag_ID=? AND seconds BETWEEN ? AND ?" % offset_expr,
        con, params=[tag, start, end])
    # The group-2 counter is recorded in the receiver's original time basis; in-place
    # jump application shifts the stored seconds, so certify against the original time.
    raw["record_adjustment_s"] = clock_jump_pre.firmware_record_adjustments(
        raw, counter_seconds=raw.seconds + raw.jump_offset_s)
    raw = raw.rename(columns={"seconds": "t"}).drop_duplicates()
    if raw.duplicated(["Rec_ID", "t"]).any():
        raise RuntimeError("Ambiguous source provenance for %s beacon" % tag)
    paired = paired.merge(raw, on=["Rec_ID", "t"], how="left", validate="one_to_one")
    if paired.SourceRow.isna().any() or paired.SourceFile.isna().any():
        raise RuntimeError("Paired beacon lacks source provenance")
    paired["reference_seconds"] = paired.t-paired.tdoa_s
    reference = raw[raw.Rec_ID == host][["t", "SourceFile", "SourceRow", "record_adjustment_s"]].rename(
        columns={"t": "reference_seconds", "SourceFile": "reference_file", "SourceRow": "reference_row",
                 "record_adjustment_s": "reference_record_adjustment_s"})
    paired = pd.merge_asof(
        paired.sort_values("reference_seconds"), reference.sort_values("reference_seconds"),
        on="reference_seconds", direction="nearest", tolerance=0.000001)
    if paired[["reference_row", "reference_file", "reference_record_adjustment_s"]].isna().any().any():
        raise RuntimeError("Reference beacon lacks exact source/firmware provenance")
    paired["reference_host"], paired["beacon_tag"] = host, tag
    paired["normalized_t"] = paired.t-paired.record_adjustment_s
    paired["normalized_reference_t"] = paired.reference_seconds-paired.reference_record_adjustment_s
    paired["value"] = paired.tdoa_s-paired.record_adjustment_s+paired.reference_record_adjustment_s
    return paired


def clock_rapid_reference_review(con, receiver, references, start, end, folder,
                                 include_isolated=False, consensus_min_step=0.05):
    """Build approved rapid-event evidence from raw provenance; no timestamp writes."""
    import clock_jump_pre
    source_path = folder / ("%s_source_state_boundaries.csv" % receiver)
    if not source_path.is_file():
        raise RuntimeError("Run source-boundary recovery before rapid-reference review")
    states = pd.read_csv(source_path)
    events = states[states.previous_offset.notna()
                    & (states.InternalOffsetValue != states.previous_offset)].copy()
    target_events = states[(states.OffsetChanged == 1)
                           | ((states.CounterRestart == 1) & (states.CounterContinuation == 0))]
    if include_isolated:
        events = target_events.copy()
    observations, normalized_observations, competing, source_reviews = [], [], [], {}
    adjustment_counts = {}
    for host in references:
        source_review = clock_source_boundary_review(con, host, start, end, folder)
        source_reviews[host] = source_review
        if source_review["status"] != "reviewed":
            raise RuntimeError("Independent reference source boundaries cannot be recovered for %s" % host)
        host_states = pd.read_csv(folder / ("%s_source_state_boundaries.csv" % host))
        original_reference_events = host_states[
            (host_states.previous_offset.notna()
             & (host_states.InternalOffsetValue != host_states.previous_offset))
            | ((host_states.CounterRestart == 1)
               & (host_states.CounterContinuation == 0))][["Rec_ID", "seconds"]]
        competing.append(original_reference_events)
        metadata = con.execute(
            "SELECT r.Tag_ID,t.pulseRate FROM tblReceiver r JOIN tblTag t ON r.Tag_ID=t.Tag_ID WHERE r.Rec_ID=?",
            (host,)).fetchone()
        if metadata is None:
            raise ValueError("Missing independent beacon metadata for %s" % host)
        tag, period = metadata
        paired = clock_jump_pre.beacon_first_arrivals(
            con, host, tag, float(period), start-300, end+300)
        paired = paired[paired.Rec_ID == receiver].copy()
        paired = _clock_paired_provenance(con, paired, tag, host, start-300, end+300)
        paired["period"] = float(period)
        observations.append(paired)
        normalized = paired.copy()
        # Keep original source/time coordinates for event guards. Only the diagnostic
        # TDoA measurement reverses the documented per-record firmware transformation.
        normalized["raw_tdoa_s"] = normalized.tdoa_s
        normalized["tdoa_s"] = normalized.value
        normalized_observations.append(normalized)
        adjustment_counts[host] = {
            "paired_target_adjustments": int((normalized.record_adjustment_s != 0).sum()),
            "paired_reference_adjustments": int((normalized.reference_record_adjustment_s != 0).sum()),
        }
        flags, _ = clock_jump_pre.find_events(con, host, start-300, end+300)
        flags = flags.copy()
        flags["Rec_ID"] = host
        competing.append(flags)
    result, support = clock_jump_pre.rapid_reference_consensus(
        events, pd.concat(observations, ignore_index=True),
        pd.concat(competing, ignore_index=True), receiver, references, target_events=target_events,
        include_isolated=include_isolated, min_step=consensus_min_step)
    if not result.empty:
        result = result[result.jump_time_s.between(start, end)]
        support = support[support.jump_time_s.between(start, end)]
    result.to_csv(folder / ("%s_rapid_consensus_events.csv" % receiver), index=False)
    support.to_csv(folder / ("%s_rapid_consensus_references.csv" % receiver), index=False)
    normalized_result, normalized_support = clock_jump_pre.rapid_reference_consensus(
        events, pd.concat(normalized_observations, ignore_index=True),
        pd.concat(competing, ignore_index=True), receiver, references, target_events=target_events,
        include_isolated=include_isolated, min_step=consensus_min_step)
    if not normalized_result.empty:
        normalized_result = normalized_result[normalized_result.jump_time_s.between(start, end)].copy()
        normalized_support = normalized_support[normalized_support.jump_time_s.between(start, end)]
        normalized_result["method"] = ("firmware_normalized_source_three_beacon_consensus"
                           if include_isolated else "firmware_normalized_rapid_three_beacon_consensus")
    normalized_result.to_csv(folder / ("%s_firmware_normalized_rapid_events.csv" % receiver), index=False)
    normalized_support.to_csv(folder / ("%s_firmware_normalized_rapid_references.csv" % receiver), index=False)
    pd.concat(normalized_observations, ignore_index=True).to_csv(
        folder / ("%s_firmware_normalized_observations.csv" % receiver), index=False)
    return {"status": "measured_not_applied", "events": len(result),
            "include_isolated": include_isolated,
            "consensus_min_step_s": consensus_min_step,
            "reference_source_reviews": source_reviews,
            "firmware_diagnostic": {
                "status": "diagnostic_only_not_applied",
                "adjustment_counts": adjustment_counts,
                "classification": normalized_result.status.value_counts().to_dict()
                    if not normalized_result.empty else {},
                "warning": "Reversing documented per-record firmware adjustments defines a diagnostic clock coordinate; "
                           "it is not an approved absolute-time repair or a raw timestamp change.",
            },
            "classification": result.status.value_counts().to_dict() if not result.empty else {}}


def clock_joint_reference_review(c, start, end, folder):
    """Run approved joint-event diagnostics on isolated raw data, without timestamp writes."""
    import clock_jump_pre
    from run_data import _compact_readonly
    receivers = list(dict.fromkeys(c.surface + c.deep))
    master = c.run["study"]["master_receiver"]
    states, observations, source_reviews = [], [], {}
    with closing(_compact_readonly(c.db)) as con:
        if "jump_offset_s" in {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}:
            raise RuntimeError("Joint reconstruction requires unshifted raw data")
        for rec in receivers:
            source_reviews[rec] = clock_source_boundary_review(con, rec, start, end, folder)
            if source_reviews[rec]["status"] != "reviewed":
                raise RuntimeError("Joint reconstruction lacks original source evidence for %s" % rec)
            source = pd.read_csv(folder / ("%s_source_state_boundaries.csv" % rec))
            states.append(source[(source.OffsetChanged == 1)
                                 | ((source.CounterRestart == 1) & (source.CounterContinuation == 0))])
        for host in [master] + c.deep:
            tag, period = con.execute(
                "SELECT r.Tag_ID,t.pulseRate FROM tblReceiver r JOIN tblTag t ON r.Tag_ID=t.Tag_ID WHERE r.Rec_ID=?",
                (host,)).fetchone()
            paired = clock_jump_pre.beacon_first_arrivals(con, host, tag, float(period), start-300, end+300)
            paired = paired[paired.Rec_ID.isin(receivers)]
            paired = _clock_paired_provenance(con, paired, tag, host, start-300, end+300)
            paired["period"] = float(period)
            observations.append(paired)
    data = pd.concat(observations, ignore_index=True)
    events = pd.concat(states, ignore_index=True)
    segment_support = clock_jump_pre.beacon_event_segment_support(events, data)
    segment_support.to_csv(folder / "beacon_event_segment_support.csv", index=False)
    measurements, support, notes = clock_jump_pre.joint_event_reconstruction(
        events, data, master)
    measurements = measurements[measurements.jump_time_s.between(start, end)]
    support = support.merge(
        measurements[["Rec_ID", "source_file", "source_row"]],
        on=["Rec_ID", "source_file", "source_row"], validate="many_to_one")
    data.to_csv(folder / "joint_clock_observations.csv", index=False)
    measurements.to_csv(folder / "joint_clock_event_measurements.csv", index=False)
    support.to_csv(folder / "joint_clock_beacon_holdouts.csv", index=False)
    notes.update(source_reviews=source_reviews,
                 independent_segment_support=segment_support.dbscan_class.value_counts().to_dict(),
                 classification=measurements.status.value_counts().to_dict(),
                 master_classification=measurements[measurements.Rec_ID == master].status.value_counts().to_dict())
    (folder / "joint_clock_model_summary.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return notes


def clock_review_run(c):
    """Read-only multi-beacon timing and upstream/final coverage comparison; never solve or apply offsets."""
    import clock_jump_pre
    import matplotlib.pyplot as plt
    from run_data import _compact_readonly
    start, end = [clock_jump_pre.naive_seconds(value) for value in c.clock_review_interval]
    if start >= end or start < clock_jump_pre.naive_seconds(c.window_start) or end > clock_jump_pre.naive_seconds(c.window_end):
        raise ValueError("Clock review interval must be increasing and inside the study window")
    manifest = json.loads(c.db.with_suffix(".compact.json").read_text())
    sources = [("unshifted", Path(manifest["source"]))]
    if c.compare_clock_db is not None:
        if not c.compare_clock_db.is_file() or c.compare_clock_db.resolve() == c.db.resolve():
            raise ValueError("Comparison must be an existing prior database, not the experiment destination")
        sources.append(("prior_corrected", c.compare_clock_db))
    with closing(_compact_readonly(c.db)) as con:
        if ("jump_offset_s" in {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}
                and c.db.resolve() not in {db.resolve() for _, db in sources}):
            sources.append(("tdoa_supplement", c.db))
    folder = c.db.parent / "clock_review" / ("interval_%s_%s" % (int(start), int(end)))
    folder.mkdir(parents=True, exist_ok=True)
    curves, coverage, source_reviews, master_medians, rapid_reviews = [], [], {}, [], {}
    for arm, db in sources:
        with closing(_compact_readonly(db)) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            master_rec, master_tag = con.execute(
                "SELECT r.Rec_ID,r.Tag_ID FROM tblReceiver r JOIN tblStudyParameters s ON r.Rec_ID=s.masterReceiver"
            ).fetchone()
            if arm == "unshifted" and c.run["legacy"].get("clock_segment_safe", False):
                source_reviews[master_rec] = clock_source_boundary_review(
                    con, master_rec, start, end, folder)
                if c.run["legacy"].get("clock_rapid_event_consensus", False):
                    rapid_reviews[master_rec] = clock_rapid_reference_review(
                        con, master_rec, c.deep, start, end, folder)
            for host in [master_rec] + c.deep:
                beacon = con.execute(
                    "SELECT r.Tag_ID,t.pulseRate FROM tblReceiver r JOIN tblTag t ON r.Tag_ID=t.Tag_ID WHERE r.Rec_ID=?",
                    (host,)).fetchone()
                if beacon is None:
                    raise ValueError("Missing beacon metadata for %s" % host)
                tag, period = beacon
                if arm == "unshifted" and host != master_rec:
                    measured, evidence, unresolved, _ = clock_jump_pre.quantify_jumps(
                        con, [master_rec], host, tag, float(period), start, end,
                        window_s=float(c.run["legacy"].get("clock_jump_window_s", 300)),
                        min_step=float(c.run["legacy"].get("clock_jump_min_step_s", 0.05)),
                        max_step=float(c.run["legacy"].get("clock_jump_max_step_s", 5.0)))
                    for status, records in (("measured", measured), ("evidence_only", evidence),
                                            ("unresolved", unresolved)):
                        master_medians.extend(dict(record, reference_host=host, beacon_tag=tag,
                                                   median_status=status) for record in records)
                data = clock_jump_pre.beacon_first_arrivals(con, host, tag, float(period), start, end)
                if data.empty:
                    raise RuntimeError("No paired %s beacon observations in %s" % (tag, arm))
                data = data.copy()
                data["arm"], data["host"], data["beacon_tag"] = arm, host, tag
                data["adjacent_step_s"] = data.groupby("Rec_ID").tdoa_s.diff()
                data["gap_s"] = data.groupby("Rec_ID").t.diff()
                curves.append(data)
            for rec in dict.fromkeys(c.surface + c.deep):
                upstream = 0
                if "tblMetronomeSecondFiltered" in tables:
                    upstream = con.execute(
                        "SELECT count(*) FROM tblMetronomeSecondFiltered WHERE Rec_ID=? AND Tag_ID=? "
                        "AND seconds BETWEEN ? AND ? AND multipath_prediction=0",
                        (rec, master_tag, start, end)).fetchone()[0]
                joined = 0
                if {"tblMetronomeSecondFiltered", "tblMetronomeFiltered"} <= tables:
                    joined = con.execute(
                        "SELECT count(*) FROM tblMetronomeSecondFiltered l WHERE l.Rec_ID=? AND l.Tag_ID=? "
                        "AND l.seconds BETWEEN ? AND ? AND l.multipath_prediction=0 AND l.transNo!=0 "
                        "AND EXISTS(SELECT 1 FROM tblMetronomeFiltered m WHERE m.Rec_ID=? AND m.Tag_ID=? "
                        "AND m.transNo=l.transNo AND m.multipath=0)",
                        (rec, master_tag, start, end, master_rec, master_tag)).fetchone()[0]
                final = 0
                if "tblDetectionClockFixed" in tables:
                    final = con.execute(
                        "SELECT count(*) FROM tblDetectionClockFixed WHERE Rec_ID=? AND Tag_ID=? AND seconds BETWEEN ? AND ?",
                        (rec, master_tag, start, end)).fetchone()[0]
                coverage.append({"arm": arm, "Rec_ID": rec, "upstream_accepted": upstream,
                                 "master_join_supported": joined, "final_clock_rows": final})
    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_csv(folder / "multi_beacon_relative_timing.csv", index=False)
    pd.DataFrame(coverage).to_csv(folder / "clock_stage_coverage.csv", index=False)
    pd.DataFrame(master_medians).to_csv(folder / "master_existing_median_review.csv", index=False)
    for rec in rapid_reviews:
        normalized = pd.read_csv(folder / ("%s_firmware_normalized_observations.csv" % rec))
        fig, axes = plt.subplots(len(c.deep), 1, figsize=(11, 3 * len(c.deep)), squeeze=False)
        for ax, host in zip(axes[:, 0], c.deep):
            local = normalized[(normalized.reference_host == host)
                               & normalized.t.between(start, end)].sort_values("t")
            ax.plot(pd.to_datetime(local.t, unit="s"), local.raw_tdoa_s, ".-", label="Original TDoA")
            ax.plot(pd.to_datetime(local.t, unit="s"), local.tdoa_s, "x--",
                    label="Documented per-record adjustment reversed (diagnostic)")
            ax.set_title("%s listening to %s beacon" % (rec, host))
            ax.set_ylabel("TDoA (s)")
            ax.legend(fontsize="small")
        fig.suptitle("Firmware diagnostic only - no raw offsets or fish positions applied")
        fig.tight_layout()
        fig.savefig(folder / ("%s_firmware_adjustment_diagnostic.png" % rec))
        plt.close(fig)
    for rec in c.deep + [c.run["study"]["master_receiver"]]:
        fig, axes = plt.subplots(4, len(sources), figsize=(7 * len(sources), 10), squeeze=False)
        for column, (arm, _) in enumerate(sources):
            for row, host in enumerate([c.run["study"]["master_receiver"]] + c.deep):
                ax = axes[row, column]
                data = all_curves[(all_curves.arm == arm) & (all_curves.host == host)
                                  & (all_curves.Rec_ID == rec)].sort_values("t")
                if not data.empty:
                    baseline = data.tdoa_s.head(4).median()
                    ax.plot(pd.to_datetime(data.t, unit="s"), data.tdoa_s-baseline, ".-")
                ax.set_title("%s: listener %s vs beacon host %s" % (arm, rec, host))
                ax.set_ylabel("TDoA minus first-4 median (s)")
                ax.tick_params(axis="x", rotation=20)
        fig.suptitle("Diagnostic relative timing only - not synchronized residuals or fish accuracy")
        fig.tight_layout()
        fig.savefig(folder / ("%s_multi_beacon_timing.png" % rec))
        plt.close(fig)
    arm_labels = {"unshifted": "Arm A (original)", "prior_corrected": "Arm B (internal codes, 10-29)",
                  "tdoa_supplement": "Arm B new (internal codes + TDoA DBSCAN)"}
    master_host = c.run["study"]["master_receiver"]
    pre_sync = all_curves[all_curves.host == master_host]
    for rec in sorted(pre_sync.Rec_ID.unique()):
        fig, axes = plt.subplots(len(sources), 1, figsize=(12, 3.2 * len(sources)), sharex=True, squeeze=False)
        for ax, (arm, _) in zip(axes[:, 0], sources):
            data = pre_sync[(pre_sync.arm == arm) & (pre_sync.Rec_ID == rec)].sort_values("t")
            ax.plot(pd.to_datetime(data.t, unit="s"), data.tdoa_s, ".", ms=1.5)
            steps = int((data.adjacent_step_s.abs() >= 0.05).sum())
            ax.set_title("%s: %s vs master %s beacon (%d adjacent changes >= 50 ms)"
                         % (arm_labels.get(arm, arm), rec, master_host, steps))
            ax.set_ylabel("TDoA (s)")
        axes[-1, 0].tick_params(axis="x", rotation=20)
        fig.suptitle("Pre-synchronization beacon TDoA; flat = no clock jumps (reflections appear as isolated points)")
        fig.tight_layout()
        fig.savefig(folder / ("tdoa_arms_%s.png" % rec), dpi=110)
        plt.close(fig)
    joint = (clock_joint_reference_review(c, start, end, folder)
             if c.run["legacy"].get("clock_joint_event_reconstruction", False) else None)
    notes = {"status": "diagnostic_only", "interval_wall": c.clock_review_interval,
             "sources": {arm: str(db) for arm, db in sources}, "paired_rows": len(all_curves),
             "folder": str(folder), "source_boundary_review": source_reviews,
             "rapid_reference_review": rapid_reviews,
             "joint_reference_review": joint,
             "master_median_status_counts": pd.DataFrame(master_medians).median_status.value_counts().to_dict()
                 if master_medians else {},
             "warning": "Arms use their own receiver timestamp coordinates and paired samples. "
                        "Relative changes and stage counts are diagnostic, not independent timing truth or matched-fish comparison."}
    (folder / "review_manifest.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return notes


def clock_jumps_run(c):
    """Step 1c (experimental; [legacy] apply_clock_jumps = true): correct receiver clock
    jumps BEFORE clock sync (scripts/clock_jump_pre.py). Events from the per-file internal
    offset-change codes in tblDetectionRaw (OffsetChanged/CounterRestart), magnitudes from
    the master-beacon TDoA median +/- clock_jump_window_s per side, applied to
    tblDetectionRaw.seconds with the audit column jump_offset_s, recorded in
    tblClockJumpEvents. Downstream tables are dropped so the pipeline rebuilds them on the
    corrected times. Jumps without an internal code are left to the legacy auto-detect."""
    import clock_jump_pre
    legacy = c.run["legacy"]
    safe = legacy.get("clock_segment_safe", False)
    master_rec, master_tag = lp.master(c.db)
    period = float(lp.query(c.db, "select pulseRate from tblTag where Tag_ID = ?", (master_tag,)).pulseRate.iloc[0])
    if legacy.get("clock_tdoa_supplement", False):
        return _clock_jumps_supplement_run(c, master_rec, master_tag, period)
    notes, jumps = clock_jump_pre.run(
        str(c.db), list(dict.fromkeys(c.surface + c.deep)), master_rec, master_tag, period,
        c.window_start, c.window_end,
        window_s=float(legacy.get("clock_jump_window_s", 300)),
        min_step=float(legacy.get("clock_jump_min_step_s", 0.05)),
        max_step=float(legacy.get("clock_jump_max_step_s", 5.0)),
        apply=not safe,
        group_span_s=float(legacy["clock_trigger_group_span_s"]) if safe else None,
        conservative=safe, persist_diagnostics=safe)
    if safe:
        review_dir = c.db.parent / "clock_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        lp.query(c.db, "SELECT * FROM tblClockJumpCandidates ORDER BY Rec_ID,jump_time_s").to_csv(
            review_dir / "clock_jump_candidates.csv", index=False)
        (review_dir / "clock_jump_summary.json").write_text(
            json.dumps(notes, indent=2, default=str), encoding="utf-8")
        review_gate(c.db, c.tags, c.receivers_3d,
                    clock_jump_pre.naive_seconds(c.window_start),
                    clock_jump_pre.naive_seconds(c.window_end),
                    review_dir, check_curves=False)
        if jumps:
            from run_data import check_capacity
            check_capacity(c.db, 4 * 2**30 + c.db.stat().st_size, 6 * 2**30)
        rows, affected = clock_jump_pre.apply_offsets(str(c.db), jumps)
        notes.update(rows_shifted=rows, affected_receivers=affected)
        (review_dir / "clock_jump_summary.json").write_text(
            json.dumps(notes, indent=2, default=str), encoding="utf-8")
    if jumps:
        _invalidate_clock_outputs(c)
    lp.ensure_progress(c.db)
    lp.mark_done(c.db, "__ats_clock_jumps__")
    return notes


def _clock_jumps_supplement_run(c, master_rec, master_tag, period):
    """Internal-code corrections (Arm B method), then the TDoA/DBSCAN supplement.

    The strict review gate still runs before fish steps; uncertain supplementary
    boundaries are unresolved candidates there, not silently accepted.
    """
    import clock_jump_pre
    from run_data import check_capacity
    legacy = c.run["legacy"]
    if not legacy.get("clock_segment_safe", False):
        raise ValueError("clock_tdoa_supplement requires the segment-safe strict-gate experiment")
    receivers = list(dict.fromkeys(c.surface + c.deep))
    min_step = float(legacy.get("clock_jump_min_step_s", 0.05))
    max_step = float(legacy.get("clock_jump_max_step_s", 5.0))
    dbscan_eps = float(legacy.get("clock_tdoa_dbscan_eps", 1.0))
    dbscan_min_samples = int(legacy.get("clock_tdoa_dbscan_min_samples", 3))
    dbscan_window_periods = float(legacy.get("clock_tdoa_dbscan_window_periods", 2.5))
    # In-place update of the isolated, reproducible compact copy; the reserve covers a
    # worst-case rollback journal of the whole database.
    check_capacity(c.db, c.db.stat().st_size + 2**30, 6 * 2**30)
    with closing(sqlite3.connect(c.db)) as con:
        shifted = "jump_offset_s" in {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}
        has_ledger = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tblClockJumpEvents'").fetchone()
    if shifted and not has_ledger:
        raise RuntimeError("Raw timestamps are shifted without an applied-event ledger; rebuild the compact copy")
    notes = {}
    if not shifted:
        notes, _ = clock_jump_pre.run(
            str(c.db), receivers, master_rec, master_tag, period, c.window_start, c.window_end,
            window_s=float(legacy.get("clock_jump_window_s", 300)), min_step=min_step, max_step=max_step,
            apply=True, group_span_s=None, conservative=False)
    notes["tdoa_supplement"] = clock_jump_pre.tdoa_supplement(
        str(c.db), receivers, master_rec, master_tag, period, c.window_start, c.window_end,
        min_step=min_step, max_step=max_step, dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples, dbscan_window_periods=dbscan_window_periods)
    review_dir = c.db.parent / "clock_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    lp.query(c.db, "SELECT * FROM tblClockJumpEvents ORDER BY Rec_ID,jump_time_s").to_csv(
        review_dir / "clock_jump_applied_ledger.csv", index=False)
    lp.query(c.db, "SELECT * FROM tblClockJumpCandidates ORDER BY Rec_ID,jump_time_s").to_csv(
        review_dir / "clock_jump_candidates.csv", index=False)
    try:
        review_gate(c.db, c.tags, c.receivers_3d, clock_jump_pre.naive_seconds(c.window_start),
                    clock_jump_pre.naive_seconds(c.window_end), review_dir, check_curves=False)
        notes["pre_sync_gate"] = "passed"
    except RuntimeError as error:
        # Reported here; the pre-fish gate re-checks after synchronization and stops fish steps.
        notes["pre_sync_gate"] = "blocked: %s" % error
    (review_dir / "clock_jump_summary.json").write_text(json.dumps(notes, indent=2, default=str), encoding="utf-8")
    _invalidate_clock_outputs(c)
    lp.ensure_progress(c.db)
    lp.mark_done(c.db, "__ats_clock_jumps__")
    return notes


def _invalidate_clock_outputs(c):
    """Drop everything derived from detection times so the pipeline rebuilds it after a
    change to tblDetectionRaw (same pattern as deep_clocks_run / adopt_deep_run).
    tblInterpolatedTemp and tblWSEL are parse artifacts (external temperature/covariate
    files, independent of detection times) and must be kept: the metronome reads the
    temperature table before step 9 runs."""
    con = sqlite3.connect(c.db)
    if lp.PROGRESS_TABLE in table_names(c.db):
        con.execute("delete from %s where Tag_ID in ('__ats_phase2__', '__ats_fish_two_stage__', '__export__')" % lp.PROGRESS_TABLE)
    for table in lp.DERIVED_TABLES + ["tblPositions_Deng", "tblPositions_Deng2D", "tblPositions_DengDBSCAN",
                                      "tblPositions_Deng3DConsensus", "tblPositions_Deng2DJoint",
                                      "tblPositions_Deng2DJointQuality", "tblPositions_Deng3DJoint",
                                      "tblPositions_Deng3DJointQuality", "tblDeepReceiver_step6_solution"]:
        con.execute("drop table if exists %s" % table)
    con.commit()
    con.close()
    for folder in ("positions", "positions_2d"):
        shutil.rmtree(c.work / folder, ignore_errors=True)
    for line in DRAG_LINES:
        (c.review / ("%s_positions.png" % line.replace("-", ""))).unlink(missing_ok=True)
        for tag in ("FC36", "FFD3"):
            for suffix in ("", "_full_range"):
                (c.review / ("item12_pipeline3d_3D_legacy_A_%s_%s%s.png" % (line, tag, suffix))).unlink(missing_ok=True)
                (c.review / ("item12_pipeline3d_3D_legacy_B_%s_%s%s.png" % (line, tag, suffix))).unlink(missing_ok=True)


def metronome_done(c):
    return count(c.db, "tblMetronomeFiltered") > 0 and count(c.db, "tblMetronomeSecondFiltered") > 0


def metronome_run(c):
    """Steps 2-3: pairwise beacon DBSCAN on the master beacon (run_data.py does not pass --output-db, so call it here)."""
    d = c.run["dbscan"]
    out = lp.resolve_path(Path("output") / ("dbscan_%s_%s_anchor%s" % (c.db.stem, d["beacon_receiver"], d["anchor"])))
    cmd = [sys.executable, str(HERE / "beacon_pairwise_dbscan.py"), str(c.db),
           "--beacon-receiver", d["beacon_receiver"], "--anchor", d["anchor"],
           "--start", str(d.get("start") or c.window_start), "--end", str(d.get("end") or c.window_end),
           "--output-dir", str(out), "--output-db", str(c.db), "--no-interactive"]
    done = subprocess.run(cmd, cwd=lp.REPO)
    if done.returncode != 0:
        raise RuntimeError("beacon_pairwise_dbscan.py exited with %s" % done.returncode)
    return {"tblMetronomeFiltered": need_rows(c.db, "tblMetronomeFiltered"),
            "tblMetronomeSecondFiltered": need_rows(c.db, "tblMetronomeSecondFiltered")}


def clock_surface_done(c):
    return count(c.db, "tblDetectionClockFixed") > 0


def clock_surface_run(c):
    """Step 4: surface clock fix against the master beacon (config positions)."""
    if c.run["legacy"].get("clock_segment_safe", False):
        if "tblDetectionClockFixed" in table_names(c.db):
            raise ValueError("Clock-fixed table already exists; refusing duplicate experimental clock run")
        lp.clock_fix(c.db, c.surface, c.work)
    else:
        lp.clock_fix_check(c.db, c.surface)
    n = need_rows(c.db, "tblDetectionClockFixed")
    missing = [r for r in c.surface if count(c.db, "tblDetectionClockFixed", "where Rec_ID = ?", (r,)) == 0]
    return {"clock_fixed_rows": n, "surface_receivers_without_rows": missing}


def deep_beacons_done(c):
    return all(count(c.db, "tblDetectionFilterSecondary", "where Tag_ID = ?", (t,)) > 0 for t in c.deng_tags.values())


def deep_beacons_run(c):
    """Step 5: first-arrival ranking of each deep receiver's beacon, then the stationary-beacon DBSCAN filter."""
    notes = {}
    for rec, tag in c.deng_tags.items():
        primary = lp.fresh_folder(c.work / "multipath_primary")
        data = jsats3d.multipath_data_object(tag, str(c.db), primary)
        if data.empty:
            raise RuntimeError("check failed: no clock-fixed detections for deep beacon %s (%s)" % (tag, rec))
        jsats3d.multipath_2(data)
        lp.widen_table(c.db, "tblDetectionFilterPrimary", primary)
        jsats3d.multipath_data_management(primary, str(c.db), primary=True)
        anchor, _summary, dropped, suspects = bpd.filter_deep_beacon(str(c.db), tag)
        lp.execute(c.db, "delete from tblDetectionFilterSecondary where Tag_ID = '%s' and transNo is null" % tag)
        notes[tag] = {"receiver": rec, "primary": need_rows(c.db, "tblDetectionFilterPrimary", "where Tag_ID = ?", (tag,)),
                      "secondary": need_rows(c.db, "tblDetectionFilterSecondary", "where Tag_ID = ?", (tag,)),
                      "anchor": anchor, "anchor_suspect_epochs": int(suspects), "dropped_no_transNo": int(dropped)}
    return notes


def deep_positions_done(c):
    if "tblDeepReceiver_step6_solution" not in table_names(c.db):
        return False
    columns = {row[1] for row in lp.query(c.db, "pragma table_info(tblDeepReceiver_step6_solution)").itertuples(index=False, name=None)}
    if "solution_root" not in columns:
        return False
    root = c.run["legacy"].get("deep_solution", "B")
    return all(count(c.db, "tblDeepReceiver_step6_solution", "where Rec_ID = ? and solution_root = ?",
                     (receiver, root)) == 1 for receiver in c.deng)


def deep_positions_run(c):
    """Step 6: legacy Deng configured-root median for each Deng receiver beacon (deep, then static surface)."""
    rec = lp.query(c.db, "select Rec_ID, Tag_ID, X_t, Y_t, Z_t from tblReceiver").set_index("Rec_ID")
    root = c.run["legacy"].get("deep_solution", "B")
    if root not in ("A", "B"):
        raise ValueError("deep_solution must be A or B")
    out = Path(lp.fresh_folder(c.work / "deep_receivers"))
    rows, notes = [], {}
    for r in c.deng:
        refs = [x for x in (c.reference if r in c.deep else c.reference_2d) if x != r]
        pos = lp.deng(c.db, rec.at[r, "Tag_ID"], refs, out, c.work / "figures")
        sols = getattr(pos, "DengSolution%s_unfiltered" % root) if pos is not None else pd.DataFrame({"comment": []})
        sols = sols[sols.comment == "solution found"]
        if sols.empty and r in c.deep:
            # Not enough receivers to solve — keep configured position
            notes[r] = {"found": 0, "references": len(refs), "skipped": "insufficient receivers, using configured position"}
            rows.append((r, rec.at[r, "Tag_ID"], float(rec.at[r, "X_t"]), float(rec.at[r, "Y_t"]), float(rec.at[r, "Z_t"]), 0, 0, root))
            continue
        if sols.empty:
            # surface beacon not positionable from its references: configured position kept
            rows.append((r, rec.at[r, "Tag_ID"], float(rec.at[r, "X_t"]), float(rec.at[r, "Y_t"]), float(rec.at[r, "Z_t"]), 0, 0, root))
            notes[r] = {"found": 0, "references": len(refs), "skipped": "configured position kept"}
            continue
        med = sols[["X", "Y", "Z"]].median()
        rows.append((r, rec.at[r, "Tag_ID"], float(med.X), float(med.Y), float(med.Z), int(len(sols)), int(sols.transNo.nunique()), root))
        notes[r] = {"found": int(len(sols)), "solution_root": root,
                    "epochs": int(sols.transNo.nunique()), "X": round(float(med.X), 2), "Y": round(float(med.Y), 2),
                    "Z": round(float(med.Z), 2)}
    con = sqlite3.connect(c.db)
    con.execute("drop table if exists tblDeepReceiver_step6_solution")
    con.execute("create table tblDeepReceiver_step6_solution (Rec_ID TEXT, Tag_ID TEXT, X REAL, Y REAL, Z REAL, "
                "solutions_used INTEGER, epochs_used INTEGER, solution_root TEXT)")
    con.executemany("insert into tblDeepReceiver_step6_solution values (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return notes


def adopt_deep_z_mode(c):
    return str(c.run["legacy"].get("adopt_deep_z", "solved")).lower()


def adopt_deep_done(c):
    if "tblDeepReceiver_step6_solution" not in table_names(c.db):
        return False
    sol = lp.query(c.db, "select Rec_ID, X, Y, Z from tblDeepReceiver_step6_solution").set_index("Rec_ID")
    rec = lp.query(c.db, "select Rec_ID, X_t, Y_t, Z_t from tblReceiver").set_index("Rec_ID")
    z_solved = adopt_deep_z_mode(c) == "solved"
    solved = {r: [sol.at[r, "X"], sol.at[r, "Y"], sol.at[r, "Z"] if (r in c.deep and z_solved) else rec.at[r, "Z_t"]]
              for r in sol.index}
    return all(r in solved and np.allclose(rec.loc[r, ["X_t", "Y_t", "Z_t"]].astype(float).values, solved[r], atol=1e-6)
               for r in c.deng)


def adopt_deep_run(c):
    """Step 6b (legacy phase 2): write the solved deep positions into tblReceiver, then invalidate every output that
    was computed with the config positions (deep clocks, fish multipath, positions, export, plots).
    [legacy] adopt_deep_z: 'solved' adopts the Deng median Z of deep receivers (legacy); 'config' keeps measured depth."""
    sol = lp.query(c.db, "select Rec_ID, X, Y, Z from tblDeepReceiver_step6_solution").set_index("Rec_ID")
    cfg = lp.query(c.db, "select Rec_ID, X_t, Y_t, Z_t from tblReceiver").set_index("Rec_ID")
    z_solved = adopt_deep_z_mode(c) == "solved"
    notes = {}
    for r in c.deng:
        x, y = (float(sol.at[r, k]) for k in ("X", "Y"))
        # surface Z stays configured; Deng XY only
        z = float(sol.at[r, "Z"]) if (r in c.deep and z_solved) else float(cfg.at[r, "Z_t"])
        notes[r] = {"X": round(x, 2), "Y": round(y, 2), "Z": round(z, 2),
                    "shift_m": round(float(np.hypot(x - cfg.at[r, "X_t"], y - cfg.at[r, "Y_t"])), 2)}
        lp.execute(c.db, "update tblReceiver set X_t = %r, Y_t = %r, Z_t = %r where Rec_ID = '%s'" % (x, y, z, r))
    tables = table_names(c.db)
    con = sqlite3.connect(c.db)
    if lp.PROGRESS_TABLE in tables:
        con.execute("delete from %s where Tag_ID in ('__ats_phase2__', '__ats_fish_two_stage__')" % lp.PROGRESS_TABLE)
    if "tblDetectionClockFixed" in tables:
        con.execute("delete from tblDetectionClockFixed where Rec_ID in (%s)" % ",".join("'%s'" % r for r in c.deng))
    inl = ",".join("'%s'" % t for t in c.tags)
    for t in ("tblDetectionFilterPrimary", "tblDetectionFilterSecondary"):
        if t in tables:
            con.execute("delete from %s where Tag_ID in (%s)" % (t, inl))
    for t in ("tblPositions_Deng", "tblPositions_Deng2D", "tblPositions_DengDBSCAN", "tblPositions_Deng3DConsensus",
              "tblPositions_Deng2DJoint", "tblPositions_Deng2DJointQuality",
              "tblPositions_Deng3DJoint", "tblPositions_Deng3DJointQuality"):
        con.execute("drop table if exists %s" % t)
    con.commit()
    con.close()
    for folder in ("positions", "positions_2d"):
        shutil.rmtree(c.work / folder, ignore_errors=True)
    for line in DRAG_LINES:
        png = c.review / ("%s_positions.png" % line.replace("-", ""))
        if png.exists():
            png.unlink()
        for tag in ("FC36", "FFD3"):
            for suffix in ("", "_full_range"):
                (c.review / ("item12_pipeline3d_3D_legacy_A_%s_%s%s.png" % (line, tag, suffix))).unlink(missing_ok=True)
    return notes


def deep_clocks_done(c):
    if lp.PROGRESS_TABLE not in table_names(c.db):
        return False
    complete = count(c.db, lp.PROGRESS_TABLE, "where Tag_ID = ?", ("__ats_phase2__",)) > 0
    return complete and all(count(c.db, "tblDetectionClockFixed", "where Rec_ID = ?", (r,)) > 0
                            for r in dict.fromkeys(c.surface + c.deep))


def deep_clocks_run(c):
    """Legacy phase 2: rerun metronome and every receiver clock using adopted deep coordinates."""
    lp.ensure_progress(c.db)
    lp.execute(c.db, "delete from %s where Tag_ID = '__ats_phase2__'" % lp.PROGRESS_TABLE,
               "delete from %s where Tag_ID = '__ats_fish_two_stage__'" % lp.PROGRESS_TABLE,
               *["drop table if exists %s" % table for table in lp.DERIVED_TABLES],
               "drop table if exists tblPositions_Deng", "drop table if exists tblPositions_Deng2D",
               "drop table if exists tblPositions_DengDBSCAN")
    for folder in ("positions", "positions_2d"):
        shutil.rmtree(c.work / folder, ignore_errors=True)
    for line in DRAG_LINES:
        (c.review / ("%s_positions.png" % line.replace("-", ""))).unlink(missing_ok=True)
        for tag in ("FC36", "FFD3"):
            for suffix in ("", "_full_range"):
                (c.review / ("item12_pipeline3d_3D_legacy_B_%s_%s%s.png" % (line, tag, suffix))).unlink(missing_ok=True)
    metronome_run(c)
    receivers = list(dict.fromkeys(c.surface + c.deep))
    lp.clock_fix(c.db, receivers, c.work)
    dup = int(lp.query(c.db, "select count(*) n from (select 1 from tblDetectionClockFixed group by Rec_ID, Tag_ID, seconds "
                              "having count(*) > 1)").n.iloc[0])
    if dup:
        raise RuntimeError("check failed: %d duplicate clock-fixed groups" % dup)
    notes = {}
    for rec in receivers:
        d = lp.query(c.db, "select seconds, seconds_fix from tblDetectionClockFixed where Rec_ID = ? order by seconds", (rec,))
        if d.empty:
            raise RuntimeError("check failed: no clock-fixed rows for deep receiver %s" % rec)
        corr = (d.seconds_fix - d.seconds).to_numpy()
        t0 = d.seconds.to_numpy() - d.seconds.min()
        slope, icpt = np.polyfit(t0, corr, 1)
        steps = np.abs(np.diff(corr))
        notes[rec] = {"rows": int(len(d)), "median_correction_ms": round(float(np.median(corr)) * 1e3, 3),
                      "drift_us_per_s": round(float(slope) * 1e6, 3),
                      "rms_about_line_ms": round(float(np.sqrt(np.mean((corr - (slope * t0 + icpt)) ** 2))) * 1e3, 3),
                      "steps_over_1ms": int((steps > 1e-3).sum()), "largest_step_ms": round(float(steps.max()) * 1e3, 1)}
    lp.mark_done(c.db, "__ats_phase2__")
    return notes


def fish_multipath_done(c):
    return (count(c.db, lp.PROGRESS_TABLE, "where Tag_ID = ?", ("__ats_fish_two_stage__",)) > 0
            and all(count(c.db, "tblDetectionFilterSecondary", "where Tag_ID = ?", (t,)) > 0 for t in c.tags))


def classify_ats_fish(primary, pulse_rate):
    """ATS secondary classifier: detect late primary candidates relative to each receiver's local pulse-phase trend."""
    if not np.isfinite(pulse_rate) or pulse_rate <= 0:
        raise ValueError("ATS secondary classification requires a finite positive pulseRate")
    secondary = primary.copy().reset_index(drop=True)
    secondary["multipath_prediction"] = secondary.multipath.astype(int)
    secondary["dbscan_class"] = np.where(secondary.multipath == 1, "later_arrival", "unclassified_primary")
    secondary["delta_s"] = np.nan
    for _, group in secondary[secondary.multipath == 0].groupby("Rec_ID"):
        group = group.sort_values("seconds_fix")
        phase = group.seconds_fix - group.transNo * pulse_rate
        trend = phase.rolling(SUSPECT_WINDOW, center=True, min_periods=3).median()
        residual = phase - trend
        judged = group.index[residual.notna()]
        rejected = group.index[residual > SUSPECT_LATE_S]
        secondary.loc[judged, "dbscan_class"] = "clean"
        secondary.loc[group.index, "delta_s"] = residual
        secondary.loc[rejected, "multipath_prediction"] = 1
        secondary.loc[rejected, "dbscan_class"] = "suspect_reflection"
    return secondary


def fish_multipath_run(c):
    """Step 8: legacy pulse-rate enumeration and primary ranking, then a distinct ATS timing classifier."""
    lp.ensure_progress(c.db)
    lp.execute(c.db, "delete from %s where Tag_ID = '__ats_fish_two_stage__'" % lp.PROGRESS_TABLE)
    notes = {}
    for tag in c.tags:
        primary_folder = lp.fresh_folder(c.work / "multipath_primary")
        data = jsats3d.multipath_data_object(tag, str(c.db), primary_folder)
        if data.empty:
            raise RuntimeError("no clock-fixed detections for study tag %s" % tag)
        jsats3d.multipath_2(data)
        existing = table_names(c.db)  # read before the delete: a large delete locks other connections
        con = sqlite3.connect(c.db)
        try:
            for table in ("tblDetectionFilterPrimary", "tblDetectionFilterSecondary"):
                if table in existing:
                    con.execute("delete from %s where Tag_ID = ?" % table, (tag,))
            con.commit()
        finally:
            con.close()
        lp.widen_table(c.db, "tblDetectionFilterPrimary", primary_folder)
        jsats3d.multipath_data_management(primary_folder, str(c.db), primary=True)
        primary = lp.query(c.db, "select * from tblDetectionFilterPrimary where Tag_ID = ?", (tag,))
        rate = float(lp.query(c.db, "select pulseRate from tblTag where Tag_ID = ?", (tag,)).pulseRate.iloc[0])
        secondary = classify_ats_fish(primary[primary.transNo.notna()], rate)
        con = sqlite3.connect(c.db)
        try:
            secondary.to_sql("tblDetectionFilterSecondary", con, if_exists="append", index=False, chunksize=1000)
            con.commit()
        finally:
            con.close()
        notes[tag] = {"primary_rows": int(len(primary)), "secondary_rows": int(len(secondary)),
                      "transmissions": int(secondary.transNo.nunique()),
                      "primary_later_arrivals": int((primary.multipath == 1).sum()),
                      "secondary_suspect_first_arrivals": int((secondary.dbscan_class == "suspect_reflection").sum()),
                      "unclassified_first_arrivals": int((secondary.dbscan_class == "unclassified_primary").sum())}
    lp.mark_done(c.db, "__ats_fish_two_stage__")
    return notes


def speed_of_sound_done(c):
    """Check if speed-of-sound diagnostic has been computed."""
    return count(c.db, "tblInterpolatedTemp") > 0


def speed_of_sound_run(c):
    """Step 9: temperature to speed of sound at the window start, middle and end (read-only check of the inputs)."""
    ti = jsats3d.temp_interpolator(str(c.db), "linear")
    start, end = pd.Timestamp(c.window_start), pd.Timestamp(c.window_end) - pd.Timedelta(minutes=1)
    notes = {}
    for label, ts in (("start", start), ("middle", start + (end - start) / 2), ("end", end)):
        temp = float(ti(pd.Timestamp(ts, tz="UTC").timestamp()))
        sos = float(jsats3d.sos(temp))
        if not (np.isfinite(temp) and 1300 < sos < 1600):
            raise RuntimeError("check failed: temperature %s at %s gives speed of sound %s" % (temp, ts, sos))
        notes[label] = {"C": round(temp, 3), "m_per_s": round(sos, 2)}
    return notes


def positions_done(c):
    if "tblPositions_Deng" not in table_names(c.db):
        return False
    if any((c.work / "positions").glob("*.csv")):
        return True
    # Table exists but may have zero solutions - check for actual Deng solutions
    n = count(c.db, "tblPositions_Deng", "where comment = ?", ("solution found",))
    return n > 0


def positions_run(c):
    """Step 11: Deng for each fish tag with the configured 3D receivers (roots A and B written per tag)."""
    out = Path(lp.fresh_folder(c.work / "positions"))
    notes = {}
    for tag in c.tags:
        pos = lp.deng(c.db, tag, c.receivers_3d, out, c.work / "figures")
        if pos is None:
            raise RuntimeError("check failed: no filtered detections for %s" % tag)
        notes[tag] = {"A_found": int((pos.DengSolutionA_unfiltered.comment == "solution found").sum()),
                      "B_found": int((pos.DengSolutionB_unfiltered.comment == "solution found").sum())}
    if not any(out.glob("*.csv")):
        raise RuntimeError("check failed: Deng wrote no position files")
    return notes


def export_done(c):
    return "tblPositions_Deng" in table_names(c.db)


def export_run(c):
    """Step 10: load the position files into tblPositions_Deng (the legacy loader deletes the files)."""
    positions = c.work / "positions"
    lp.widen_table(c.db, "tblPositions_Deng", positions, extra=("solution", "Tag_ID"))
    jsats3d.positions_data_management("Deng", str(positions), str(c.db))
    total = need_rows(c.db, "tblPositions_Deng")
    found = count(c.db, "tblPositions_Deng", "where comment = 'solution found'")
    return {"rows": total, "solution_found": found,
            "by_tag_root": lp.query(c.db, "select Tag_ID, solution, count(*) n, sum(comment = 'solution found') found "
                                          "from tblPositions_Deng group by Tag_ID, solution").to_dict("records")}


def bathymetry_check_done(c):
    """Bathymetry check is done when the check columns exist in tblPositions_Deng."""
    if "tblPositions_Deng" not in table_names(c.db):
        return False
    cols = {r[1] for r in lp.query(c.db, "pragma table_info(tblPositions_Deng)").itertuples(index=False, name=None)}
    return "bathymetry_check" in cols


def bathymetry_check_run(c):
    """Step 10b: apply bathymetry-based water column filtering to position solutions.

    Reads [legacy] bathymetry settings:
        bathy_path          – path to bathymetry XYZ/CSV/GeoTIFF
        bathy_input_crs     – CRS of the input file (default EPSG:26910)
        water_column        – if true and no bathy data, apply WSEL-only check
    """
    legacy = c.run["legacy"]
    bathy_path = legacy.get("bathy_path")
    water_column = legacy.get("water_column", False)

    if not bathy_path and not water_column:
        return {"status": "skipped", "reason": "no bathy_path, water_column = false"}

    from jsats3d import bathymetry as bth

    if bathy_path:
        bathy_path = lp.resolve_path(bathy_path)
        input_crs = legacy.get("bathy_input_crs", "EPSG:26910")
        print(f"Bathymetry: loading {bathy_path} (CRS: {input_crs})")
        n = bth.load_xyz(str(c.db), bathy_path, input_crs=input_crs)
        print("Bathymetry: checking position solutions")
        bth.check_positions(str(c.db), wsel_tolerance_m=0.05)
        return {"status": "bathymetry", "points_loaded": n}
    else:
        print("Bathymetry: no bathy data; applying WSEL-only water column check")
        bth.check_positions_simple(str(c.db), water_column=True)
        return {"status": "water_column_only"}






def positions_2d_done(c):
    if not c.run["legacy"].get("known_depth_diagnostics", False):
        return True
    known_depth_tags = [tag for tag in c.tags if tag in c.fixed_z_2d]
    return not known_depth_tags or "tblPositions_Deng2D" in table_names(c.db)


def positions_2d_run(c):
    """Legacy Deng2D (every receiver set of three, roots A and B) with fish Z fixed per tag, over receivers_2d."""
    if not c.run["legacy"].get("known_depth_diagnostics", False):
        return {"skipped": "known-depth diagnostic disabled; final positions use legacy 3D Deng"}
    known_depth_tags = [tag for tag in c.tags if tag in c.fixed_z_2d]
    if not known_depth_tags:
        return {"skipped": "no configured known tag depths"}
    if not c.receivers_2d:
        raise RuntimeError("check failed: Deng2D needs receivers_2d")
    out = Path(lp.fresh_folder(c.work / "positions_2d"))
    notes = {}
    for tag in known_depth_tags:
        pos = lp.deng_2d(c.db, tag, c.receivers_2d, out, c.work / "figures", c.fixed_z_2d[tag])
        if pos is None:
            raise RuntimeError("check failed: no filtered detections for %s at the 2D receivers" % tag)
        notes[tag] = {"fixed_z": c.fixed_z_2d[tag], "A_found": int((pos.Deng2DSolutionA.comment == "solution found").sum()),
                      "B_found": int((pos.Deng2DSolutionB.comment == "solution found").sum())}
    found = lp.load_2d_positions(c.db, out, c.fixed_z_2d)
    if found < 1:
        raise RuntimeError("check failed: tblPositions_Deng2D has no solutions")
    notes["solution_found"] = found
    return notes


















def plot_drag_done(c):
    if not {"FC36", "FFD3"} <= set(c.tags):
        return True
    return all((c.review / ("item12_pipeline3d_3D_legacy_A_%s_%s.png" % (line, tag))).exists()
               for line in DRAG_LINES for tag in ("FC36", "FFD3"))


def plot_drag_run(c):
    """Step 12: four black-GPS/blue-Deng XYZ plots; truth is used only for scoring."""
    if not {"FC36", "FFD3"} <= set(c.tags):
        return {"skipped": "ENT truth is available only for FC36 and FFD3"}
    gps_file = Path(c.paths["config_xlsx"]).parents[1] / "5_array_testing" / "array_testing_drag_GPS.csv"
    cmd = [sys.executable, str(HERE / "ent_analysis_2025.py"), "--db", str(c.db),
           "--out", c.db.stem, "--only", "12", "--gps", str(gps_file), "--lines", *DRAG_LINES]
    result = subprocess.run(cmd, cwd=lp.REPO)
    if result.returncode != 0 or not plot_drag_done(c):
        raise RuntimeError("step 12 failed to generate the four legacy Deng comparison plots")
    return {"folder": str(c.review), "blue_source": "tblPositions_Deng root A mean XYZ",
            "gps_and_tag_depth_used_for_solving": False}


def plot_drag_2d_folder(c):
    return lp.REPO / "output" / "2025_review" / (c.db.stem + "_2d")


def plot_drag_2d_done(c):
    return not c.run["legacy"].get("known_depth_diagnostics", False) or (plot_drag_2d_folder(c) / "item12_pipeline_summary.csv").exists()


def plot_drag_2d_run(c):
    """Optional: scores and plots Deng2D (fish Z fixed) root-A centroids against the drag GPS."""
    if not c.run["legacy"].get("known_depth_diagnostics", False):
        return {"skipped": "known_depth_diagnostics = false"}
    if not {"FC36", "FFD3"} <= set(c.tags):
        return {"skipped": "ENT truth is available only for FC36 and FFD3"}
    # Use the same GPS source as plot_drag_run for consistency; fallback to local copy if K: unavailable
    gps_file = Path(c.paths["config_xlsx"]).parents[1] / "5_array_testing" / "array_testing_drag_GPS.csv"
    if not gps_file.exists():
        gps_file = lp.REPO / "data" / "drag_GPS_2025.csv"
    cmd = [sys.executable, str(HERE / "ent_analysis_2025.py"), "--db", str(c.db), "--out", c.db.stem + "_2d",
           "--only", "12", "--gps", str(gps_file), "--table", "tblPositions_Deng2D", "--plots", "3D_legacy_A",
           "--lines", *DRAG_LINES]
    result = subprocess.run(cmd, cwd=lp.REPO)
    if result.returncode != 0 or not plot_drag_2d_done(c):
        raise RuntimeError("2D plot step did not write its summary")
    return {"folder": str(plot_drag_2d_folder(c)), "pngs": sorted(p.name for p in plot_drag_2d_folder(c).glob("item12_pipeline3d_*.png"))}


def positions_dbscan_enabled(c):
    return bool(c.run["legacy"].get("positions_dbscan", False))


def positions_dbscan_done(c):
    return not positions_dbscan_enabled(c) or "tblPositions_DengDBSCAN" in table_names(c.db)


def ping_acceleration(t, xy, gap_s):
    """Horizontal acceleration (m/s^2) at each ping from the pings before and after it; NaN at the ends and across gaps over gap_s."""
    a = np.full(len(t), np.nan)
    if len(t) < 3:
        return a
    dt1, dt2 = t[1:-1] - t[:-2], t[2:] - t[1:-1]
    ok = (dt1 > 0) & (dt2 > 0) & (dt1 <= gap_s) & (dt2 <= gap_s)
    v1 = (xy[1:-1] - xy[:-2]) / np.where(dt1 > 0, dt1, 1)[:, None]
    v2 = (xy[2:] - xy[1:-1]) / np.where(dt2 > 0, dt2, 1)[:, None]
    accel = np.linalg.norm(v2 - v1, axis=1) / (0.5 * (dt1 + dt2))
    a[1:-1][ok] = accel[ok]
    return a


def positions_dbscan_run(c):
    """Optional screen of root-B pings; keeps every root row of the transmissions that fall in a cluster.
    [legacy] dbscan_mode 'distance' (default): DBSCAN on per-transmission XY.
    'acceleration': DBSCAN on XY, depth * dbscan_depth_weight and horizontal acceleration * dbscan_accel_scale (m per m/s^2)."""
    if not positions_dbscan_enabled(c):
        return {"skipped": "positions_dbscan = false (legacy steps unchanged)"}
    from sklearn.cluster import DBSCAN
    legacy = c.run["legacy"]
    mode = str(legacy.get("dbscan_mode", "distance")).lower()
    if mode not in ("distance", "acceleration"):
        raise ValueError("dbscan_mode must be 'distance' or 'acceleration', got %r" % mode)
    eps = float(legacy.get("dbscan_eps_m", 8.0 if mode == "acceleration" else 5.0))
    min_samples = int(legacy.get("dbscan_min_samples", 5))
    accel_scale = float(legacy.get("dbscan_accel_scale", 5.0))
    depth_weight = float(legacy.get("dbscan_depth_weight", 1.0))
    gap_s = float(legacy.get("dbscan_accel_gap_s", 12.0))
    pos = lp.query(c.db, "select * from tblPositions_Deng where comment = 'solution found'")
    root = pos[pos.solution == "B"]
    if mode == "acceleration":
        if "T01" not in root:
            raise ValueError("acceleration mode needs T01 in tblPositions_Deng")
        root = root.assign(**{col: pd.to_numeric(root[col], errors="coerce") for col in ("X", "Y", "Z", "ToA", "T01")})
        root = root.assign(t0=root.ToA - root.T01)
        root = root[np.isfinite(root[["X", "Y", "Z", "t0"]]).all(axis=1) & root[["X", "Y", "Z"]].abs().lt(1000).all(axis=1)]
    kept, notes = [], {}
    for tag, rows in root.groupby("Tag_ID"):
        if mode == "distance":
            ping = rows.groupby("transNo")[["X", "Y"]].mean()
            features = ping.to_numpy(dtype=float)
        else:
            ping = rows.groupby("transNo").agg(X=("X", "mean"), Y=("Y", "mean"), Z=("Z", "mean"), t0=("t0", "median")).sort_values("t0")
            accel = ping_acceleration(ping.t0.to_numpy(), ping[["X", "Y"]].to_numpy(), gap_s)
            accel = np.where(np.isnan(accel), np.nanmedian(accel) if np.isfinite(accel).any() else 0.0, accel)
            features = np.column_stack([ping.X, ping.Y, depth_weight * ping.Z, accel_scale * accel])
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)
        kept_trans = set(ping.index[labels != -1])
        notes[tag] = {"mode": mode, "transmissions": int(len(ping)), "kept": int(len(kept_trans)), "eps_m": eps, "min_samples": min_samples}
        kept.append(pos[(pos.Tag_ID == tag) & pos.transNo.isin(kept_trans)])
    con = sqlite3.connect(c.db)
    pd.concat(kept).to_sql("tblPositions_DengDBSCAN", con, if_exists="replace", index=False)
    con.commit()
    con.close()
    return notes


def plot_dbscan_folder(c):
    return lp.REPO / "output" / "2025_review" / (c.db.stem + "_dbscan")


def plot_drag_dbscan_done(c):
    return not positions_dbscan_enabled(c) or (plot_dbscan_folder(c) / "item12_pipeline_summary.csv").exists()


def plot_drag_dbscan_run(c):
    """Optional: step-12 plots and scores from the DBSCAN-screened positions table."""
    if not positions_dbscan_enabled(c):
        return {"skipped": "positions_dbscan = false"}
    if not {"FC36", "FFD3"} <= set(c.tags):
        return {"skipped": "ENT truth is available only for FC36 and FFD3"}
    gps_file = Path(c.paths["config_xlsx"]).parents[1] / "5_array_testing" / "array_testing_drag_GPS.csv"
    cmd = [sys.executable, str(HERE / "ent_analysis_2025.py"), "--db", str(c.db), "--out", c.db.stem + "_dbscan",
           "--only", "12", "--gps", str(gps_file), "--table", "tblPositions_DengDBSCAN", "--plots", "3D_legacy_B",
           "--lines", *DRAG_LINES]
    result = subprocess.run(cmd, cwd=lp.REPO)
    if result.returncode != 0 or not plot_drag_dbscan_done(c):
        raise RuntimeError("DBSCAN plot step did not write its summary")
    return {"folder": str(plot_dbscan_folder(c)), "pngs": sorted(p.name for p in plot_dbscan_folder(c).glob("item12_pipeline3d_*.png"))}


STEPS = [("prepare", "1b", prepare_done, prepare_run),
         ("clock_review", "1r", clock_review_done, clock_review_run),
         ("clock_jumps", "1c", clock_jumps_done, clock_jumps_run),
         ("metronome", "2-3", metronome_done, metronome_run),
         ("clock_surface", "4", clock_surface_done, clock_surface_run),
         ("deep_beacons", "5", deep_beacons_done, deep_beacons_run),
         ("deep_positions", "6", deep_positions_done, deep_positions_run),
         ("adopt_deep", "6b", adopt_deep_done, adopt_deep_run),
         ("deep_clocks", "7", deep_clocks_done, deep_clocks_run),
         ("fish_multipath", "8", fish_multipath_done, fish_multipath_run),
         ("speed_of_sound", "9", speed_of_sound_done, speed_of_sound_run),
         ("positions", "11", positions_done, positions_run), ("export", "10", export_done, export_run),
         ("bathymetry_check", "10b", bathymetry_check_done, bathymetry_check_run),
         ("positions_2d", "11b", positions_2d_done, positions_2d_run),
         ("plot_drag", "12", plot_drag_done, plot_drag_run),
         ("plot_drag_2d", "12c", plot_drag_2d_done, plot_drag_2d_run),
         ("positions_dbscan", "11c", positions_dbscan_done, positions_dbscan_run),
         ("plot_drag_dbscan", "12b", plot_drag_dbscan_done, plot_drag_dbscan_run)]


# ----------------------------------------------------------------------------------------------- driver

def build_context(args):
    run, paths = lp.read_run_file(args.run_file)
    db = lp.resolve_path(args.db) if args.db else paths["output_db"]
    if not db.exists():
        raise SystemExit("database %s not found; run scripts/run_data.py with this run file first" % db)
    legacy, study, selection = run["legacy"], run["study"], run["selection"]
    with closing(sqlite3.connect(db)) as existing_policy_con:
        if _clock_experiment_enabled(existing_policy_con) and not legacy.get("clock_segment_safe", False):
            raise ValueError("This database uses segment-safe clock policy; use its experimental run configuration")
    if legacy.get("clock_segment_safe", False):
        manifest_path = db.with_suffix(".compact.json")
        if not manifest_path.exists():
            raise RuntimeError("Segment-safe experiment requires an isolated compact-copy manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or Path(manifest["target"]).resolve() != db.resolve():
            raise RuntimeError("Compact experiment database is incomplete or manifest target mismatches")
        if not legacy.get("apply_clock_jumps", False):
            raise ValueError("Segment-safe experiment requires apply_clock_jumps and its candidate ledger")
        with closing(sqlite3.connect(db)) as policy_con:
            policy_con.execute("CREATE TABLE IF NOT EXISTS tblClockExperimentPolicy "
                               "(segment_safe INTEGER, receivers_3d TEXT)")
            policy_columns = {r[1] for r in policy_con.execute("PRAGMA table_info(tblClockExperimentPolicy)")}
            if "source_database" not in policy_columns:
                policy_con.execute("ALTER TABLE tblClockExperimentPolicy ADD COLUMN source_database TEXT")
            policy_con.execute("DELETE FROM tblClockExperimentPolicy")
            policy_con.execute("INSERT INTO tblClockExperimentPolicy "
                               "(segment_safe,receivers_3d,source_database) VALUES (1,?,?)",
                               (";".join(legacy["receivers_3d"]), manifest["source"]))
            policy_con.commit()
    surface, deep = list(legacy["surface_receivers"]), list(legacy["deep_receivers"])
    receivers_3d = list(legacy["receivers_3d"])
    master_rec, _ = lp.master(db)
    if master_rec not in surface:
        raise SystemExit("master receiver %s must be in surface_receivers" % master_rec)
    study_tags = lp.query(db, "select Tag_ID from tblTag where TagType = 'study' and pulseRate is not null").Tag_ID.tolist()
    tags = [t for t in (selection.get("tags") or study_tags) if t in study_tags]
    if not tags:
        raise SystemExit("no study tags with a pulse rate in %s" % db)
    work = lp.resolve_path(db.parent / ("%s_legacy" % db.stem))
    work.mkdir(parents=True, exist_ok=True)
    deng = deep + ([r for r in receivers_3d if r in surface and r != master_rec] if legacy.get("deng_surface", False) else [])  # static only; CFD floats never targets
    deng_tags = {r: lp.query(db, "select Tag_ID from tblReceiver where Rec_ID = ?", (r,)).Tag_ID.iloc[0] for r in deng}
    drag_plots = not getattr(args, "no_drag_plots", False)
    return SimpleNamespace(run=run, paths=paths, db=db, work=work, surface=surface, deep=deep, receivers_3d=receivers_3d,
                           clock_review_interval=getattr(args, "clock_review_interval", None),
                           compare_clock_db=lp.resolve_path(args.compare_clock_db) if getattr(args, "compare_clock_db", None) else None,
                           receivers_2d=list(legacy.get("receivers_2d") or []), fixed_z_2d=dict(legacy.get("fixed_z_2d_by_tag") or {}),
                           review=(db.parent / "figures" if legacy.get("clock_segment_safe", False)
                                   else lp.REPO / "output" / "2025_review" / db.stem),
                           reference=[r for r in receivers_3d if r not in deep], reference_2d=[r for r in (legacy.get("receivers_2d") or receivers_3d) if r not in deep], tags=tags, deng=deng, deng_tags=deng_tags,
                           window_start=str(study["synch_time_start"]), window_end=str(study["synch_time_end"]), drag_plots=drag_plots)


def init_whatif(source, target, swaps):
    """New database from a finished one: derived tables dropped, tblReceiver back at config, then the listed hydrophone positions swapped.

    Only coordinates move (X, Y, Z, X_t, Y_t, Z_t, easting, northing); clock, tag and mount stay with the receiver id.
    tblReceiver_initial keeps the config rows and tblReceiverSwap records what was done."""
    if target.exists():
        raise SystemExit("%s exists; --init-from only creates a new database" % target)
    shutil.copy2(source, target)
    con = sqlite3.connect(target)
    for table in lp.DERIVED_TABLES + ["tblPositions_Deng", "tblPositions_Deng2D", "tblPositions_DengDBSCAN",
                                     "tblPositions_Deng3DConsensus",
                                     "tblPositions_Deng2DJoint", "tblPositions_Deng2DJointQuality",
                                     "tblPositions_Deng3DJoint", "tblPositions_Deng3DJointQuality",
                                     "tblDeepReceiver_step6_solution", lp.PROGRESS_TABLE]:
        con.execute("drop table if exists %s" % table)
    con.execute("drop table tblReceiver")
    con.execute("create table tblReceiver as select * from tblReceiver_initial")
    con.execute("drop table if exists tblReceiverSwap")
    con.execute("create table tblReceiverSwap (Rec_A TEXT, Rec_B TEXT)")
    columns = ["X", "Y", "Z", "X_t", "Y_t", "Z_t", "easting", "northing"]
    for a, b in swaps:
        rows = {r: con.execute("select %s from tblReceiver where Rec_ID = ?" % ", ".join(columns), (r,)).fetchone() for r in (a, b)}
        if None in rows.values():
            raise SystemExit("swap %s %s: receiver not in tblReceiver" % (a, b))
        for rec, other in ((a, b), (b, a)):
            con.execute("update tblReceiver set %s where Rec_ID = ?" % ", ".join("%s = ?" % c for c in columns), (*rows[other], rec))
        con.execute("insert into tblReceiverSwap values (?, ?)", (a, b))
    con.commit()
    con.execute("vacuum")
    con.close()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=lp.REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_file")
    parser.add_argument("--db", help="database to use instead of [paths] output_db (own work folder, own status file)")
    parser.add_argument("--only", nargs="+", metavar="NAME")
    parser.add_argument("--from", dest="first", metavar="NAME")
    parser.add_argument("--to", dest="last", metavar="NAME")
    parser.add_argument("--list", action="store_true", help="show the steps and exit")
    parser.add_argument("--init-from", metavar="DB", help="what-if: create --db as a reset copy of DB (derived tables dropped, config receivers)")
    parser.add_argument("--swap", nargs=2, action="append", metavar=("REC_A", "REC_B"), help="with --init-from: exchange two hydrophone positions")
    parser.add_argument("--no-drag-plots", action="store_true", help="skip drag plot steps (12, 12b, 12c)")
    parser.add_argument("--clock-review-interval", nargs=2, metavar=("START", "END"),
                        help="Opt-in clock_review diagnostic interval, study wall-clock timestamps")
    parser.add_argument("--compare-clock-db", help="Read-only prior corrected database for clock_review")
    parser.add_argument("--revert-tdoa-supplement", action="store_true",
                        help="Restore the experiment to internal-code corrections only, then rerun clock_jumps")
    args = parser.parse_args(argv)
    if args.compare_clock_db and not args.clock_review_interval:
        parser.error("--compare-clock-db requires --clock-review-interval")
    names = [s[0] for s in STEPS]
    if args.list:
        for name, label, _, run in STEPS:
            print("%-15s step %-4s %s" % (name, label, (run.__doc__ or "").strip().splitlines()[0]))
        return 0
    for given in (args.only or []) + [x for x in (args.first, args.last) if x]:
        if given not in names:
            parser.error("unknown step %r; choose from %s" % (given, names))
    if args.init_from:
        if not args.db:
            parser.error("--init-from needs --db for the new database")
        init_whatif(lp.resolve_path(args.init_from), lp.resolve_path(args.db), args.swap or [])
    elif args.swap:
        parser.error("--swap needs --init-from (it never changes an existing database)")
    c = build_context(args)
    if args.revert_tdoa_supplement:
        import clock_jump_pre
        if not c.run["legacy"].get("clock_tdoa_supplement", False):
            parser.error("--revert-tdoa-supplement requires clock_tdoa_supplement in the run file")
        source = json.loads(c.db.with_suffix(".compact.json").read_text())["source"]
        try:
            print("Reverted TDoA supplement: %s" % clock_jump_pre.revert_tdoa_supplement(c.db, source), flush=True)
        except RuntimeError as error:
            if "nothing to revert" not in str(error):
                raise
            print("No TDoA supplement applied; resetting clock_jumps only", flush=True)
        _invalidate_clock_outputs(c)
        with closing(sqlite3.connect(c.db)) as con:
            con.execute("DELETE FROM %s WHERE Tag_ID = ?" % lp.PROGRESS_TABLE, ("__ats_clock_jumps__",))
            con.commit()
    selected = args.only or names[names.index(args.first or names[0]):names.index(args.last or names[-1]) + 1]
    status_path = c.db.parent / ("%s.pipeline_status.json" % c.db.stem)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update({"database": str(c.db), "run_file": str(Path(args.run_file)), "git_head": git_head(), "tags": c.tags})
    steps = status.setdefault("steps", {})
    print("Database %s\nTags %s | master/anchor from run file | window %s to %s | drag_plots=%s" % (c.db, c.tags, c.window_start, c.window_end, c.drag_plots))
    failed = None
    for name, label, done, run in STEPS:
        # Skip drag plot steps if --no-drag-plots was passed
        if not c.drag_plots and name.startswith("plot_drag"):
            print("\n=== step %s: %s ===" % (label, name), flush=True)
            print("skipped (--no-drag-plots)", flush=True)
            steps.setdefault(name, {})["status"] = "done (skipped, --no-drag-plots)"
            continue
        if name not in selected:
            continue
        print("\n=== step %s: %s ===" % (label, name), flush=True)
        if done(c):
            print("already done, skipped", flush=True)
            steps.setdefault(name, {})["status"] = "done (skipped, output exists)"
            continue
        began = time.time()
        steps[name] = {"label": label, "status": "running", "started": datetime.now().isoformat(timespec="seconds")}
        status_path.write_text(json.dumps(status, indent=2, default=str))
        try:
            with closing(sqlite3.connect(c.db)) as gate_con:
                safe = _clock_experiment_enabled(gate_con)
            if safe:
                from run_data import check_capacity
                check_capacity(c.db, 4 * 2**30, 6 * 2**30)
            if safe and name in ("fish_multipath", "positions", "positions_2d", "positions_dbscan"):
                from clock_jump_pre import naive_seconds
                review_gate(c.db, c.tags, c.receivers_3d,
                            naive_seconds(c.window_start), naive_seconds(c.window_end),
                            c.db.parent / "clock_review")
            notes = run(c)
            steps[name].update(status="ok", minutes=round((time.time() - began) / 60, 1), notes=notes)
            print("ok in %.1f min: %s" % ((time.time() - began) / 60, json.dumps(notes, default=str)[:1500]), flush=True)
        except Exception as error:  # a failed check must stop the run, not be skipped
            trace = traceback.format_exc()
            steps[name].update(status="FAILED", minutes=round((time.time() - began) / 60, 1),
                               error="%s: %s" % (type(error).__name__, error), traceback=trace.splitlines()[-12:])
            failed = name
            print("FAILED: %s: %s" % (type(error).__name__, error), flush=True)
            print("FULL TRACEBACK:")
            print(trace, flush=True)
        status_path.write_text(json.dumps(status, indent=2, default=str))
        if failed:
            break
    print("\nStatus (%s)" % status_path)
    print("| step | name | status | minutes |\n|---|---|---|---|")
    for name, label, _, _ in STEPS:
        s = steps.get(name, {})
        print("| %s | %s | %s | %s |" % (label, name, s.get("status", "not run"), s.get("minutes", "")))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
