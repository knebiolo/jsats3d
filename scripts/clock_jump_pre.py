"""Pre-clock-sync jump correction (experimental, 2026-10-29; Kevin's 2026-10-14 spec).

Correct receiver clock jumps BEFORE the clock synchronization (piecewise regression):

  1. EVENTS   -- the internal code: rows in tblDetectionRaw where the receiver's own file
                 reports an offset change (OffsetChanged = 1, first detection after an
                 InternalOffset change within that file), with CounterRestart and
                 OneSecondAdjustmentEvidence as corroborating evidence (parser:
                 parse_ats_raw_to_legacy.py). These give WHEN a jump can occur.
  2. MAGNITUDE-- the TDoA of the master-beacon detections (post-metronome pairing rule,
                 first arrival per burst vs the master receiver, same nearest-epoch
                 tolerance as beacon_pairwise_dbscan): offset = median(t_i - t_master)
                 AFTER the event minus BEFORE it, +/- clock_jump_window_s per side, each
                 side clipped at the neighbouring event. A robust median over the beacon
                 epochs is insensitive to the late multipath echoes (>= ~1.5 ms).
                 This gives HOW MUCH and the direction.
  3. APPLY    -- record in tblClockJumpEvents, then shift the data file:
                 UPDATE tblDetectionRaw SET seconds = seconds - <applied> for every row of
                 that receiver at/after the jump (beacon AND fish rows), with the
                 additive audit column jump_offset_s. Metronome tables are invalidated so
                 the pipeline rebuilds them on the corrected times.

An event with no TDoA step at/above the threshold is recorded as evidence-only (a sync
happened, the clock did not measurably step). Nothing is written to the DB in --report
mode; the pipeline calls run(..., apply=True).

Sign convention: offset_s > 0 means the receiver's recorded time ran LATER after the
event (clock jumped forward); the correction is seconds - offset_s = true time.
"""
import argparse
from contextlib import closing
import datetime as dt
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Beacon epochs needed per side of an event to resolve the median level. The beacon pings
# every ~60-63 s, so a full 300 s window holds ~4-5 per side (grid offset); clipped
# windows are allowed to shrink to ~3 (a >= 150 s window) before the event is reported
# unresolvable. A 4-epoch median is robust for steps >= 50 ms vs ~0.2-0.3 ms TDoA noise.
MIN_EPOCHS_FULL = 4
MIN_EPOCHS_CLIPPED = 3
CLIPPED_MIN_WINDOW_S = 150.0
# A co-occurring step is "similar" within this offset tolerance (ms).
COMMON_MODE_MATCH_MS = 150.0
# OneSecondAdjustmentEvidence rows within this distance of a candidate are merged into its
# evidence (the flag is dense -- median gap ~2 s, 2026-10-29 audit -- so it corroborates,
# it does not trigger).
CORROBORATION_S = 120.0

# Tables built from uncorrected detection times; dropped after the offsets are applied.
METRONOME_TABLES = ("tblMetronomeUnfiltered", "tblMetronomeFiltered", "tblMetronomeSecondFiltered")


def naive_seconds(text):
    """Study-basis wall time (PDT encoded as if UTC) -> unix seconds, same convention as
    tblDetectionRaw.seconds (beacon_pairwise_dbscan.naive_seconds)."""
    return pd.Timestamp(str(text), tz="UTC").timestamp()


def _local_ts(seconds):
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).replace(tzinfo=None)


def find_events(con, receiver, t0, t1, include_provenance=False):
    """Step 1: internal evidence rows for one receiver in [t0, t1).

    Jump candidates (triggers) are OffsetChanged (the file's own 'clock offset changed'
    code: first detection after an InternalOffset change within the file) and
    CounterRestart rows. OneSecondAdjustmentEvidence is dense (fires on most pings;
    median gap ~2 s, 2026-10-29 audit of the 2025 early-fish DB) so it corroborates a
    nearby candidate instead of triggering one.

    Returns (candidates, all_evidence): the sparse trigger rows (each with a
    one_sec_nearby corroboration flag) and every evidence row (for counting).
    """
    evidence_table = ("tblClockEventEvidence" if con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tblClockEventEvidence'"
    ).fetchone() else "tblDetectionRaw")
    provenance_sql = ""
    if include_provenance:
        provenance_sql = (", source_rowid, SourceRow" if evidence_table == "tblClockEventEvidence"
                          else ", rowid AS source_rowid, SourceRow")
    ev = pd.read_sql(
        """SELECT seconds, OffsetChanged, CounterRestart, OneSecondAdjustmentEvidence,
                  InternalOffsetValue, SourceFile%s
           FROM %s
           WHERE Rec_ID = ? AND seconds >= ? AND seconds < ?
             AND (OffsetChanged = 1 OR CounterRestart = 1 OR OneSecondAdjustmentEvidence = 1)
           ORDER BY seconds""" % (provenance_sql, evidence_table), con, params=[receiver, t0, t1])
    if ev.empty:
        return ev, ev
    if include_provenance:
        columns = {r[1] for r in con.execute("PRAGMA table_info(%s)" % evidence_table)}
        if "InternalCounterValue" in columns:
            counter = pd.read_sql(
                "SELECT %s AS source_rowid,InternalCounterValue FROM %s "
                "WHERE Rec_ID=? AND seconds>=? AND seconds<?"
                % ("source_rowid" if evidence_table == "tblClockEventEvidence" else "rowid", evidence_table),
                con, params=[receiver, t0, t1])
        else:
            source = con.execute("SELECT source_database FROM tblClockExperimentPolicy").fetchone()
            if source is None or not Path(source[0]).is_file():
                raise RuntimeError("Original source database is required to recover omitted-tag counter provenance")
            counter_rows = []
            with closing(sqlite3.connect(Path(source[0]).resolve().as_uri() + "?mode=ro", uri=True)) as original:
                ids = ev.source_rowid.tolist()
                for i in range(0, len(ids), 500):
                    batch = ids[i:i+500]
                    counter_rows.extend(original.execute(
                        "SELECT rowid,InternalCounterValue FROM tblDetectionRaw WHERE rowid IN (%s)"
                        % ",".join("?" for _ in batch), batch).fetchall())
            counter = pd.DataFrame(counter_rows, columns=["source_rowid", "InternalCounterValue"])
        ev = ev.merge(counter, on="source_rowid", how="left", validate="one_to_one", sort=False)
        if ev.InternalCounterValue.isna().any():
            raise RuntimeError("Missing original counter provenance; restart episodes cannot be certified")
    one_sec = ev[ev.OneSecondAdjustmentEvidence == 1].seconds.to_numpy()
    s = ev.seconds.to_numpy()
    idx = np.searchsorted(one_sec, s)
    nearby = np.zeros(len(ev), dtype=bool)
    right = idx < len(one_sec)
    nearby[right] = one_sec[idx[right]] - s[right] <= CORROBORATION_S
    left = idx > 0
    nearby[left] = nearby[left] | (s[left] - one_sec[idx[left] - 1] <= CORROBORATION_S)
    ev = ev.copy()
    ev["one_sec_nearby"] = nearby | (ev.OneSecondAdjustmentEvidence == 1).to_numpy()
    ev["candidate"] = ((ev.OffsetChanged == 1) | (ev.CounterRestart == 1)).to_numpy()
    return ev[ev.candidate], ev


def group_repeated_triggers(candidates, max_span_s):
    """Group repeated same-file/offset messages; preserve an uncertain boundary interval.

    Source-adjacent counter 0->1 messages with the same file/offset are one restart
    episode. Other restarts and missing provenance are never grouped. Spans do not chain.
    """
    if not np.isfinite(max_span_s) or max_span_s < 0:
        raise ValueError("Trigger grouping span must be finite and nonnegative")
    groups = []
    for _, row in candidates.sort_values("seconds", kind="stable").iterrows():
        complete = all(pd.notna(row[key]) and str(row[key]).strip()
                       for key in ("SourceFile", "InternalOffsetValue"))
        same_episode = (groups and row["CounterRestart"] == 1
                        and groups[-1]["CounterRestart"] == 1
                        and groups[-1].get("InternalCounterValue") == 0
                        and row.get("InternalCounterValue") == 1
                        and pd.notna(row.get("SourceRow"))
                        and row.get("SourceRow") == groups[-1].get("SourceRow", -2) + 1
                        and groups[-1]["trigger_count"] == 1)
        repeat_offset = groups and not row["CounterRestart"] and not groups[-1]["CounterRestart"]
        merge = (groups and complete and (same_episode or repeat_offset)
                 and row["SourceFile"] == groups[-1]["SourceFile"]
                 and row["InternalOffsetValue"] == groups[-1]["InternalOffsetValue"]
                 and float(row["seconds"]) - groups[-1]["event_start_s"] <= max_span_s)
        if merge:
            group = groups[-1]
            group["seconds"] = float(row["seconds"])
            group["event_end_s"] = float(row["seconds"])
            group["trigger_count"] += 1
            group["group_kind"] = "restart_0_to_1" if same_episode else "repeated_offset"
            group["one_sec_nearby"] = bool(group["one_sec_nearby"] or row["one_sec_nearby"])
        else:
            group = row.to_dict()
            group.update(event_start_s=float(row["seconds"]),
                         event_end_s=float(row["seconds"]), trigger_count=1, group_kind="single")
            groups.append(group)
    return pd.DataFrame(groups)


def beacon_first_arrivals(con, master_rec, beacon_tag, period, t0, t1):
    """Step 2 input: first arrival per burst per receiver, paired to master transmissions.

    Same pairing rule as beacon_pairwise_dbscan.pairwise_series (nearest master epoch
    within 0.5 * period, closest pair per epoch) so the magnitude is consistent with the
    metronome. tdoa_s = t_listener - t_master; a receiver clock step of J seconds after
    the event appears as a +J step in this series.
    """
    det = pd.read_sql(
        "SELECT Rec_ID, seconds FROM tblDetectionRaw WHERE Tag_ID = ? AND seconds >= ? AND seconds < ?",
        con, params=[beacon_tag, t0, t1])
    if det.empty:
        return pd.DataFrame(columns=["Rec_ID", "t", "transNo", "tdoa_s"])
    det = det.sort_values(["Rec_ID", "seconds"]).copy()
    gap = det.groupby("Rec_ID")["seconds"].diff()
    det["burst"] = (gap.isna() | (gap >= 0.5 * period)).groupby(det.Rec_ID).cumsum()
    first = det.groupby(["Rec_ID", "burst"], as_index=False)["seconds"].min().rename(columns={"seconds": "t"})
    master = (first[first.Rec_ID == master_rec][["t"]]
              .rename(columns={"t": "t_m"}).sort_values("t_m").reset_index(drop=True))
    master["transNo"] = np.arange(len(master))
    rows = []
    for rid, g in first.groupby("Rec_ID"):
        if rid == master_rec:
            continue
        m = pd.merge_asof(g.sort_values("t"), master, left_on="t", right_on="t_m",
                          direction="nearest", tolerance=0.5 * period)
        m = m.dropna(subset=["transNo"])
        m["err"] = (m["t"] - m["t_m"]).abs()
        m = m.sort_values("err").drop_duplicates("transNo").sort_values("t_m")
        m = m.assign(Rec_ID=rid, tdoa_s=m["t"] - m["t_m"])
        rows.append(m[["Rec_ID", "t", "transNo", "tdoa_s"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Rec_ID", "t", "transNo", "tdoa_s"])


def firmware_record_adjustments(detections, counter_seconds=None):
    """Decode documented SR3017 10.62 per-record adjustments, not offset-code increments.

    counter_seconds is the time basis the group-2 counter was recorded against. It must
    be the receiver's ORIGINAL recorded time: the 15-second counter is frozen in the raw
    file, while in-place clock-jump application (jump_offset_s) shifts the stored seconds.
    Builds without applied jumps pass the stored seconds (jump offset 0).
    """
    parts = detections.Internal.fillna("").astype(str).str.split()
    flags = parts.str[3].fillna("").astype(str).str[0]
    adjustment = flags.map({"1": -1.0, ">": 1.0}).fillna(0.0)
    selected = adjustment != 0
    if not selected.any():
        return adjustment
    required = {"ReceiverType", "FirmwareVersion", "FileFormatVersion", "seconds"}
    if not required.issubset(detections.columns):
        raise ValueError("Firmware-adjusted detections lack receiver/version provenance")
    rows = detections.loc[selected]
    if not ((rows.ReceiverType == "SR3017")
            & rows.FirmwareVersion.str.match(r"^10\.62(?:[A-Z])?$", na=False)
            & (rows.FileFormatVersion.astype(str) == "2.0")).all():
        raise ValueError("Per-record adjustment is not certified for this receiver/firmware/file format")
    counter = (counter_seconds if counter_seconds is not None
               else detections["seconds"]).loc[selected]
    position = parts.loc[selected].str[1].map(lambda value: int(value, 16) / 100.0)
    difference = (counter % 15 - position + 7.5) % 15 - 7.5
    # Group 2 is truncated to hundredths; it cannot measure sub-millisecond offsets.
    if not ((difference-adjustment.loc[selected]).abs() <= 0.010001).all():
        raise ValueError("Firmware flag disagrees with the detection's 15-second counter position")
    return adjustment


def rapid_reference_consensus(events, observations, reference_events, receiver,
                              references, min_step=0.05, max_step=5.0, target_events=None,
                              include_isolated=False):
    """Measure source-bounded rapid events using three distinct beacon references.

    One epoch per side is allowed only with three-reference agreement within the
    existing 0.5 ms budget. No reference event or intermediate target event may lie
    inside a measurement bracket. This opt-in evidence rule is not a clock-fit model.
    """
    if len(set(references)) != 3 or len(references) != 3 or receiver in references:
        raise ValueError("Rapid consensus requires exactly three distinct beacon hosts")
    records, supports = [], []
    budget = 0.0005
    for filename, source_events in events.groupby("SourceFile"):
        source_events = source_events.sort_values("SourceRow").reset_index(drop=True)
        if source_events.SourceRow.duplicated().any():
            raise ValueError("Duplicate source-event boundaries")
        for i, event in source_events.iterrows():
            boundary = int(event.SourceRow)
            lower = int(source_events.iloc[i-1].SourceRow) if i else -np.inf
            upper = int(source_events.iloc[i+1].SourceRow) if i+1 < len(source_events) else np.inf
            # Only rapid episodes enter this rule; isolated events retain median evidence.
            rapid = ((i > 0 and abs(event.seconds-source_events.iloc[i-1].seconds) < CLIPPED_MIN_WINDOW_S)
                     or (i+1 < len(source_events)
                         and abs(source_events.iloc[i+1].seconds-event.seconds) < CLIPPED_MIN_WINDOW_S))
            if not rapid and not include_isolated:
                continue
            local_support = []
            for host in references:
                local = observations[(observations.reference_host == host)
                                     & (observations.SourceFile == filename)]
                pre = local[(local.SourceRow >= lower) & (local.SourceRow < boundary)].sort_values("SourceRow")
                post = local[(local.SourceRow >= boundary) & (local.SourceRow < upper)].sort_values("SourceRow")
                row = {"Rec_ID": receiver, "source_file": filename, "source_row": boundary,
                       "jump_time_s": float(event.seconds), "reference_host": host,
                       "status": "unsupported", "note": ""}
                if pre.empty or post.empty:
                    row["note"] = "missing source-bounded beacon epoch"
                else:
                    before, after = pre.iloc[-1], post.iloc[0]
                    row.update(pre_tdoa_s=float(before.tdoa_s), post_tdoa_s=float(after.tdoa_s),
                               pre_reference_s=float(before.reference_seconds),
                               post_reference_s=float(after.reference_seconds),
                               pre_source_row=int(before.SourceRow), post_source_row=int(after.SourceRow))
                    if target_events is not None:
                        intervening = target_events[
                            (target_events.SourceFile == filename)
                            & (target_events.SourceRow > before.SourceRow)
                            & (target_events.SourceRow <= after.SourceRow)
                            & (target_events.SourceRow != boundary)]
                        if not intervening.empty:
                            row["note"] = "another target Internal event inside bracket"
                            local_support.append(row)
                            continue
                    dt_host = float(after.reference_seconds-before.reference_seconds)
                    if (not np.isfinite(dt_host) or dt_host <= 0 or dt_host > 2*float(before.period)
                            or after.period != before.period):
                        row["note"] = "invalid or over-two-period reference bracket"
                    else:
                        competing = reference_events[
                            (reference_events.Rec_ID == host)
                            & reference_events.seconds.between(before.reference_seconds,
                                                               after.reference_seconds, inclusive="both")]
                        row["n_competing_reference_events"] = len(competing)
                        row["competing_reference_event_times_s"] = ";".join(
                            str(float(time)) for time in competing.seconds)
                        if not competing.empty:
                            row["note"] = "reference has its own Internal event inside bracket"
                        else:
                            step = float(after.tdoa_s-before.tdoa_s)
                            row.update(status="supported", offset_s=step,
                                       note="source-bounded independent beacon bracket")
                local_support.append(row)
            supports.extend(local_support)
            supported = [row for row in local_support if row["status"] == "supported"]
            result = {"Rec_ID": receiver, "jump_time_s": float(event.seconds),
                      "jump_time_local": _local_ts(event.seconds).strftime("%Y-%m-%d %H:%M:%S"),
                      "source_file": filename, "source_row": boundary,
                      "method": "source_three_beacon_consensus" if include_isolated else "rapid_three_beacon_consensus",
                      "n_references": len(supported),
                      "status": "unresolved", "offset_s": np.nan,
                      "window_start_s": float(event.seconds)-300, "window_end_s": float(event.seconds)+300}
            if len(supported) != 3:
                result["note"] = "fewer than three uncontested reference brackets"
            else:
                steps = np.array([row["offset_s"] for row in supported])
                result["reference_range_s"] = float(np.ptp(steps))
                if not np.all(np.isfinite(steps)) or np.ptp(steps) > budget:
                    result["note"] = "independent beacon steps disagree above 0.5 ms"
                else:
                    offset = float(np.median(steps))
                    result["offset_s"] = offset
                    if abs(offset) <= budget:
                        result.update(status="evidence_only", note="three references show no measurable step")
                    elif min_step <= abs(offset) <= max_step:
                        result.update(status="accepted", note="three independent reference steps agree within 0.5 ms")
                    else:
                        result["note"] = "agreed step outside configured application range"
            records.append(result)
    return pd.DataFrame(records), pd.DataFrame(supports)


def sparse_correction_trial(events, observations, reference_events, receiver, references,
                            source_file, source_row, target_events=None):
    """Test one supported step in a copied TDoA coordinate, never in raw timestamps."""
    if "Rec_ID" not in observations or set(observations.Rec_ID) != {receiver}:
        raise ValueError("Sparse trial requires observations of exactly the target receiver")
    before, before_support = rapid_reference_consensus(
        events, observations, reference_events, receiver, references,
        min_step=0.0005, target_events=target_events, include_isolated=True)
    selected = before[(before.source_file == source_file) & (before.source_row == source_row)]
    if len(selected) != 1 or selected.status.iloc[0] != "accepted":
        raise RuntimeError("Sparse trial requires one independently supported source event")
    offset = float(selected.offset_s.iloc[0])
    corrected = observations.copy(deep=True)
    post = (corrected.SourceFile == source_file) & (corrected.SourceRow >= source_row)
    corrected.loc[post, "tdoa_s"] = corrected.loc[post, "tdoa_s"] - offset
    after, after_support = rapid_reference_consensus(
        events, corrected, reference_events, receiver, references,
        min_step=0.0005, target_events=target_events, include_isolated=True)
    checked = after[(after.source_file == source_file) & (after.source_row == source_row)]
    before_local = before_support[(before_support.source_file == source_file)
                                  & (before_support.source_row == source_row)]
    after_local = after_support[(after_support.source_file == source_file)
                                & (after_support.source_row == source_row)]
    supported = len(checked) == 1 and checked.status.iloc[0] == "evidence_only"
    summary = {"status": "timing_trial_passed" if supported else "timing_trial_failed",
               "receiver": receiver, "source_file": source_file, "source_row": int(source_row),
               "offset_s": offset, "timing_budget_s": 0.0005,
               "before_max_step_s": float(before_local.offset_s.abs().max()),
               "after_max_step_s": float(after_local.offset_s.abs().max()),
               "observations_shifted": int(post.sum()),
               "warning": "In-memory TDoA trial on estimation references, not independent clock truth "
                          "or fish accuracy. No raw timestamps or production offsets changed."}
    return {"summary": summary, "observations": corrected,
            "before_events": before, "after_events": after,
            "before_support": before_support, "after_support": after_support}


def beacon_event_segment_support(events, observations):
    """Classify same-transmission listener differences within original event regimes.

    Uses the existing beacon DBSCAN unchanged. Neither fitted jump magnitudes nor
    nominal beacon emission intervals are inputs. Sparse rapid regimes stay noise;
    this diagnostic never fills missing arrivals or changes timestamps.
    """
    from beacon_pairwise_dbscan import cluster

    data = observations.copy()
    keys = ["Rec_ID", "reference_host", "beacon_tag", "transNo"]
    if data.empty or data.duplicated(keys).any():
        raise ValueError("Segment support requires unique paired beacon observations")
    data["event_regime"] = 0
    for (receiver, filename), local in data.groupby(["Rec_ID", "SourceFile"]):
        boundaries = events[(events.Rec_ID == receiver) & (events.SourceFile == filename)].SourceRow
        if boundaries.duplicated().any():
            raise ValueError("Duplicate source-event boundaries in segment support")
        data.loc[local.index, "event_regime"] = np.searchsorted(
            np.sort(boundaries.to_numpy(float)), local.SourceRow.to_numpy(float), side="right")
    results = []
    for (host, tag), network in data.groupby(["reference_host", "beacon_tag"]):
        receivers = sorted(network.Rec_ID.unique())
        if network.period.nunique() != 1 or not np.isfinite(network.period).all() or (network.period <= 0).any():
            raise ValueError("Beacon network lacks a single positive period")
        for i, receiver in enumerate(receivers):
            left = network[network.Rec_ID == receiver]
            for peer in receivers[i+1:]:
                right = network[network.Rec_ID == peer]
                paired = left.merge(right, on="transNo", suffixes=("_left", "_right"), validate="one_to_one")
                if paired.empty:
                    continue
                if (paired.reference_row_left != paired.reference_row_right).any():
                    raise ValueError("Same-transmission listener observations disagree on reference provenance")
                paired["t_anchor"] = paired.normalized_reference_t_left
                paired["delta_s"] = paired.value_left-paired.value_right
                paired["Rec_ID"] = receiver
                for (left_regime, right_regime), segment in paired.groupby(
                        ["event_regime_left", "event_regime_right"]):
                    classified = cluster(segment, float(network.period.iloc[0]))
                    results.append(pd.DataFrame({
                        "reference_host": host, "beacon_tag": tag,
                        "receiver": receiver, "peer_receiver": peer,
                        "transNo": classified.transNo,
                        "reference_seconds": classified.t_anchor,
                        "receiver_source_file": classified.SourceFile_left,
                        "receiver_source_row": classified.SourceRow_left,
                        "peer_source_file": classified.SourceFile_right,
                        "peer_source_row": classified.SourceRow_right,
                        "receiver_event_regime": left_regime, "peer_event_regime": right_regime,
                        "listener_difference_s": classified.delta_s,
                        "dbscan_class": classified.dbscan_class,
                        "segment_observations": len(segment),
                        "note": "Independent beacon support only; noise is not proof of a reflected arrival.",
                    }))
    if not results:
        raise ValueError("No same-transmission listener pairs for segment support")
    return pd.concat(results, ignore_index=True)


def joint_event_reconstruction(events, observations, anchor):
    """Diagnose identifiable clock steps jointly; never apply minimum-norm guesses.

    Pair intercepts absorb constant acoustic delays; per-transmission terms absorb
    a host's variable self-reception delay. Relative linear drift and source-bounded
    steps are shared across beacon networks. Beacon holdouts use same-transmission
    listener differences, so an unseen host-reception delay cannot masquerade as a
    clock jump. The anchor fixes only the common linear-drift gauge.
    """
    from scipy.optimize import least_squares

    budget = 0.0005
    events = events.sort_values(["Rec_ID", "SourceFile", "SourceRow"]).reset_index(drop=True).copy()
    data = observations.reset_index(drop=True).copy()
    event_keys = ["Rec_ID", "SourceFile", "SourceRow"]
    if events.empty or data.empty or events.duplicated(event_keys).any():
        raise ValueError("Joint reconstruction requires observations and unique source-event boundaries")
    numeric = ["normalized_t", "normalized_reference_t", "value", "SourceRow", "reference_row", "period", "transNo"]
    if not np.isfinite(data[numeric].to_numpy(float)).all() or (data.period <= 0).any():
        raise ValueError("Joint observations contain invalid time, source row or beacon period")
    receivers = sorted(set(data.Rec_ID) | set(data.reference_host))
    if anchor not in receivers or (data.Rec_ID == data.reference_host).any():
        raise ValueError("Invalid joint clock anchor or self-paired observation")
    for rec in receivers:
        files = set(data.loc[data.Rec_ID == rec, "SourceFile"]) | set(
            data.loc[data.reference_host == rec, "reference_file"])
        if len(files) != 1 or any(pd.isna(file) for file in files):
            raise ValueError("Joint review requires one unambiguous source file per receiver in its interval")
        local_events = events[events.Rec_ID == rec]
        if not set(local_events.SourceFile).issubset(files):
            raise ValueError("Joint source-event files disagree with observation provenance")
    if not set(events.Rec_ID).issubset(receivers):
        raise ValueError("Joint event receiver lacks beacon observations")
    if not np.isfinite(events[["seconds", "SourceRow"]].to_numpy(float)).all():
        raise ValueError("Joint events contain invalid source rows or timestamps")
    beacons = sorted(data.beacon_tag.unique())
    if len(beacons) < 3:
        raise ValueError("Joint reconstruction requires at least three beacon networks")
    center = float(data.normalized_reference_t.median())
    columns, names = [], []
    for key in data[["Rec_ID", "reference_host", "beacon_tag"]].drop_duplicates().itertuples(index=False, name=None):
        rec, host, beacon = key
        columns.append(((data.Rec_ID == rec) & (data.reference_host == host)
                        & (data.beacon_tag == beacon)).to_numpy(float))
        names.append(("pair", key))
    for rec in receivers:
        if rec != anchor:
            columns.append((data.Rec_ID.eq(rec) * (data.normalized_t-center) / 600
                            - data.reference_host.eq(rec) * (data.normalized_reference_t-center) / 600
                            ).to_numpy(float))
            names.append(("drift", rec))
    event_columns = []
    for i, event in events.iterrows():
        left = (data.Rec_ID.eq(event.Rec_ID) & data.SourceFile.eq(event.SourceFile)
                & data.SourceRow.ge(event.SourceRow)).to_numpy(float)
        right = (data.reference_host.eq(event.Rec_ID) & data.reference_file.eq(event.SourceFile)
                 & data.reference_row.ge(event.SourceRow)).to_numpy(float)
        event_columns.append(len(columns))
        columns.append(left-right)
        names.append(("event", i))
    if data.duplicated(["Rec_ID", "reference_host", "beacon_tag", "transNo"]).any():
        raise ValueError("Joint beacon transmissions have ambiguous paired receiver observations")
    for host, beacon, transmission in data[
            ["reference_host", "beacon_tag", "transNo"]].drop_duplicates().itertuples(index=False, name=None):
        columns.append(((data.reference_host == host) & (data.beacon_tag == beacon)
                        & (data.transNo == transmission)).to_numpy(float))
        names.append(("reception", host, beacon, transmission))
    design, values = np.column_stack(columns), data.value.to_numpy(float)

    def fit(mask):
        matrix, response = design[mask], values[mask]
        _, singular, vectors = np.linalg.svd(matrix, full_matrices=len(matrix) < matrix.shape[1])
        tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
        rank = int((singular > tolerance).sum())
        null = vectors[rank:]
        basis = vectors[:rank].T
        reduced = matrix @ basis
        initial = np.linalg.lstsq(reduced, response, rcond=None)[0]
        fitted = least_squares(
            lambda x: reduced @ x-response, initial, jac=lambda x: reduced.copy(),
            loss="soft_l1", f_scale=budget, max_nfev=200,
            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not fitted.success:
            raise RuntimeError("Joint clock diagnostic did not converge: %s" % fitted.message)
        return basis @ fitted.x, null, rank

    full, full_null, rank = fit(np.ones(len(data), dtype=bool))
    records, support, folds = [], [], {}
    for beacon in beacons:
        folds[beacon] = fit(data.beacon_tag.ne(beacon).to_numpy())
    for i, event in events.iterrows():
        column = event_columns[i]
        identifiable = np.linalg.norm(full_null[:, column]) < 1e-8
        local_events = events[(events.Rec_ID == event.Rec_ID) & (events.SourceFile == event.SourceFile)]
        lower = local_events.loc[local_events.SourceRow < event.SourceRow, "SourceRow"].max()
        upper = local_events.loc[local_events.SourceRow > event.SourceRow, "SourceRow"].min()
        lower = -np.inf if pd.isna(lower) else lower
        upper = np.inf if pd.isna(upper) else upper
        local = data[(data.Rec_ID == event.Rec_ID) & (data.SourceFile == event.SourceFile)].copy()
        local["event_row"] = local.SourceRow
        estimates, residual_steps, maximum_errors = [], [], []
        for beacon, (fitted, null, _) in folds.items():
            fold_identifiable = np.linalg.norm(null[:, column]) < 1e-8
            differences, brackets = [], []
            heldout = local[local.beacon_tag == beacon]
            for _, pair in heldout.groupby(["Rec_ID", "reference_host"]):
                pre = pair[pair.event_row.between(lower, event.SourceRow, inclusive="left")].sort_values("event_row")
                post = pair[pair.event_row.between(event.SourceRow, upper, inclusive="left")].sort_values("event_row")
                if pre.empty or post.empty:
                    continue
                before, after = pre.iloc[-1], post.iloc[0]
                gap = float(after.normalized_reference_t-before.normalized_reference_t)
                if gap <= 0 or gap > 2*float(before.period):
                    continue
                controls = data[(data.reference_host == before.reference_host)
                                & (data.beacon_tag == beacon) & (data.Rec_ID != event.Rec_ID)]
                for _, control in controls.groupby("Rec_ID"):
                    control_pre = control[control.transNo == before.transNo]
                    control_post = control[control.transNo == after.transNo]
                    if len(control_pre) != 1 or len(control_post) != 1:
                        continue
                    before_control, after_control = control_pre.iloc[0], control_post.iloc[0]
                    delta = (design[after.name]-design[before.name]
                             - design[after_control.name]+design[before_control.name])
                    if np.linalg.norm(null @ delta) >= 1e-8:
                        continue
                    difference = (values[after.name]-values[before.name]
                                  - values[after_control.name]+values[before_control.name])
                    residual_step = float(difference-delta @ fitted)
                    differences.append(residual_step)
                    brackets.append({
                        "control_receiver": before_control.Rec_ID,
                        "pre_target_source_row": int(before.SourceRow),
                        "post_target_source_row": int(after.SourceRow),
                        "pre_control_source_row": int(before_control.SourceRow),
                        "post_control_source_row": int(after_control.SourceRow),
                        "pre_transmission": int(before.transNo), "post_transmission": int(after.transNo),
                        "residual_step_s": residual_step,
                    })
            residual = float(np.median(differences)) if differences else np.nan
            maximum_error = float(np.max(np.abs(differences))) if differences else np.nan
            worst = brackets[int(np.argmax(np.abs(differences)))] if differences else {}
            estimate = float(fitted[column]) if fold_identifiable else np.nan
            support.append({
                "Rec_ID": event.Rec_ID, "source_file": event.SourceFile, "source_row": int(event.SourceRow),
                "heldout_beacon": beacon, "identifiable": fold_identifiable,
                "offset_s": estimate, "n_heldout_brackets": len(differences),
                "heldout_residual_step_s": residual,
                "heldout_max_residual_step_s": maximum_error,
                "worst_control_receiver": worst.get("control_receiver"),
                "worst_pre_target_source_row": worst.get("pre_target_source_row"),
                "worst_post_target_source_row": worst.get("post_target_source_row"),
                "worst_pre_control_source_row": worst.get("pre_control_source_row"),
                "worst_post_control_source_row": worst.get("post_control_source_row"),
            })
            estimates.append(estimate)
            residual_steps.append(residual)
            maximum_errors.append(maximum_error)
        estimates, residual_steps = np.array(estimates), np.array(residual_steps)
        spread = float(np.ptp(estimates)) if np.isfinite(estimates).all() else np.nan
        residual_support = np.isfinite(residual_steps)
        passed = (identifiable and np.isfinite(spread) and spread <= budget
                  and residual_support.sum() >= 3
                  and (np.array(maximum_errors)[residual_support] <= budget).all())
        records.append({
            "Rec_ID": event.Rec_ID, "source_file": event.SourceFile, "source_row": int(event.SourceRow),
            "jump_time_s": float(event.seconds), "identifiable": identifiable,
            "offset_s": float(full[column]) if identifiable else np.nan,
            "beacon_holdout_range_s": spread, "validated_beacon_brackets": int(residual_support.sum()),
            "status": "supported_joint_measurement" if passed else "unresolved",
            "note": "Relative-clock diagnostic only; not independently certified absolute time or an applied correction.",
        })
    residuals = values-design @ full
    notes = {"status": "diagnostic_only_not_applied", "observations": len(data),
             "parameters": design.shape[1], "rank": rank, "linear_drift_gauge": anchor,
             "residual_p95_s": float(np.quantile(np.abs(residuals), .95)),
             "validation": "heldout_beacon_same_transmission_listener_differences",
             "host_reception_delays": "free_per_beacon_transmission_not_clock_corrections",
             "events_supported": sum(row["status"] == "supported_joint_measurement" for row in records),
             "events_unresolved": sum(row["status"] == "unresolved" for row in records),
             "warning": "Robust fitting does not certify multipath, clock model adequacy or fish accuracy."}
    return pd.DataFrame(records), pd.DataFrame(support), notes


def _evidence_string(row):
    parts = [name for name, flag in (("OffsetChanged", row["OffsetChanged"]),
                                     ("CounterRestart", row["CounterRestart"])) if int(flag)]
    if int(row["one_sec_nearby"] if "one_sec_nearby" in row else row["OneSecondAdjustmentEvidence"]):
        parts.append("OneSecondAdjustment")
    return ";".join(parts)


def quantify_jumps(con, receivers, master_rec, beacon_tag, period, t0, t1,
                   window_s=300.0, min_step=0.05, max_step=5.0,
                   group_span_s=None, source_events=None):
    """Steps 1-2: for every internal event of every receiver, measure the beacon-TDoA step.

    Returns (jumps, evidence_only, unresolvable, n_events) as lists of dicts and a count.
    Windows are clipped at the neighbouring CANDIDATE events only (the dense
    OneSecondAdjustment rows never clip), per Kevin's 2026-10-14 spec ("5 minutes before
    and after the event, or up until another jump").
    """
    arrivals = beacon_first_arrivals(con, master_rec, beacon_tag, period, t0, t1)
    jumps, evidence_only, unresolvable = [], [], []
    n_events = 0
    for rec in receivers:
        if source_events is None:
            cands, all_ev = find_events(con, rec, t0, t1, include_provenance=group_span_s is not None)
        else:
            required = {"Rec_ID", "seconds", "SourceFile", "SourceRow", "OffsetChanged",
                        "CounterRestart", "CounterContinuation", "InternalOffsetValue",
                        "InternalCounterValue", "one_sec_nearby"}
            if not required.issubset(source_events.columns):
                raise ValueError("Source events lack original boundary provenance")
            all_ev = source_events[(source_events.Rec_ID == rec)
                                   & (source_events.seconds >= t0) & (source_events.seconds < t1)].copy()
            if all_ev[["SourceFile", "SourceRow", "seconds"]].isna().any().any():
                raise ValueError("Source events contain missing boundary provenance")
            if all_ev.duplicated(["SourceFile", "SourceRow"]).any():
                raise ValueError("Source events contain duplicate boundary rows")
            cands = all_ev[(all_ev.OffsetChanged == 1)
                           | ((all_ev.CounterRestart == 1) & (all_ev.CounterContinuation == 0))]
            cands = cands.sort_values("seconds", kind="stable")
        n_events += int(len(all_ev))
        if cands.empty:
            continue
        if group_span_s is not None:
            cands = group_repeated_triggers(cands, group_span_s)
        local = arrivals[arrivals.Rec_ID == rec]
        cands = cands.reset_index(drop=True)
        for i, row in cands.iterrows():
            t_ev = float(row["seconds"])
            event_start = float(row.get("event_start_s", t_ev))
            t_start = event_start - window_s
            if i > 0:
                t_start = max(t_start, float(cands.loc[i - 1, "seconds"]))
            t_end = t_ev + window_s
            if i + 1 < len(cands):
                t_end = min(t_end, float(cands.loc[i + 1].get(
                    "event_start_s", cands.loc[i + 1, "seconds"])))
            pre = local[(local.t >= t_start) & (local.t < event_start)]
            post = local[(local.t > t_ev) & (local.t <= t_end)]
            base = {"Rec_ID": rec, "jump_time_s": t_ev, "jump_time_local": _local_ts(t_ev).strftime("%Y-%m-%d %H:%M:%S"),
                    "evidence": _evidence_string(row), "source_file": row["SourceFile"],
                    "n_pre": int(len(pre)), "n_post": int(len(post))}
            if source_events is not None:
                base.update(source_row=int(row["SourceRow"]),
                            boundary_method="original_source_state",
                            window_start_s=t_start, window_end_s=t_end)
                used = pd.concat([pre, post], ignore_index=True)
                if not used.empty:
                    reference_times = used.t - used.tdoa_s
                    base.update(reference_start_s=float(reference_times.min()),
                                reference_end_s=float(reference_times.max()))
            if group_span_s is not None:
                base.update(event_start_s=event_start, event_end_s=t_ev,
                            trigger_count=int(row["trigger_count"]),
                            group_kind=row["group_kind"],
                            source_rowid=row.get("source_rowid"), source_row=row.get("SourceRow"),
                            window_start_s=t_start, window_end_s=t_end)
            # A full +/-window_s side holds ~window_s/period epochs; clipped sides need
            # fewer, but below CLIPPED_MIN_WINDOW_S the median level is not resolvable.
            min_epochs = MIN_EPOCHS_FULL
            if (event_start - t_start) < 250.0 or (t_end - t_ev) < 250.0:
                if min(event_start - t_start, t_end - t_ev) < CLIPPED_MIN_WINDOW_S:
                    unresolvable.append(dict(base, offset_s=np.nan,
                                             note="window clipped below %.0f s by a neighbouring event" % CLIPPED_MIN_WINDOW_S))
                    continue
                min_epochs = MIN_EPOCHS_CLIPPED
            if len(pre) < min_epochs or len(post) < min_epochs:
                unresolvable.append(dict(base, offset_s=np.nan,
                                         note="insufficient beacon epochs on one side (file gap?)"))
                continue
            pre_level = float(np.median(pre["tdoa_s"]))
            post_level = float(np.median(post["tdoa_s"]))
            base.update(pre_level_s=pre_level, post_level_s=post_level, offset_s=post_level - pre_level)
            if group_span_s is not None:
                base.update(pre_spread_s=float(pre.tdoa_s.max() - pre.tdoa_s.min()),
                            post_spread_s=float(post.tdoa_s.max() - post.tdoa_s.min()))
            if min_step <= abs(base["offset_s"]) <= max_step:
                jumps.append(base)
            else:
                evidence_only.append(base)
    _annotate_common_mode(jumps, evidence_only, receivers, master_rec)
    return jumps, evidence_only, unresolvable, n_events


def _annotate_common_mode(jumps, evidence_only, receivers, master_rec):
    """Diagnostic only: flag listener steps that many receivers share within seconds.

    A step on the MASTER clock shows up in every listener's TDoA with the SAME sign and
    size. If the master has no Internal event at that time, the event-driven spec does
    not attribute it (it is reported here for owner review instead of being guessed).
    Same-time, mixed-sign clusters are independent receiver GPS syncs, not common mode.
    """
    for j in jumps + evidence_only:
        if j["Rec_ID"] == master_rec:
            j["common_mode"] = 0
            continue
        similar = sum(1 for k in jumps
                      if k is not j and k["Rec_ID"] != j["Rec_ID"]
                      and abs(k["jump_time_s"] - j["jump_time_s"]) <= 60.0
                      and np.sign(k["offset_s"]) == np.sign(j["offset_s"])
                      and abs(abs(k["offset_s"]) - abs(j["offset_s"])) * 1e3 <= COMMON_MODE_MATCH_MS)
        j["common_mode"] = int(similar)


def apply_offsets(db, jumps):
    """Step 3: record tblClockJumpEvents and apply the cumulative per-receiver offset to
    tblDetectionRaw (one CASE scan per affected receiver), then invalidate the metronome
    tables. Returns (rows_shifted, affected_receivers)."""
    if not jumps:
        return 0, []
    con = sqlite3.connect(db)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}
        if "jump_offset_s" in cols:
            raise RuntimeError("Raw timestamps already have a jump correction column; refusing a second application")
        if "jump_offset_s" not in cols:
            # DEFAULT 0.0: rows before the first applied jump (or receivers without jumps)
            # read 0.0 rather than NULL
            con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN jump_offset_s REAL DEFAULT 0.0")
        con.execute("DROP TABLE IF EXISTS tblClockJumpEvents")
        pd.DataFrame(jumps).to_sql("tblClockJumpEvents", con, index=False, chunksize=1000)
        rows_shifted = 0
        by_rec = {}
        for j in jumps:
            by_rec.setdefault(j["Rec_ID"], []).append(j)
        for rec, rec_jumps in sorted(by_rec.items()):
            rec_jumps.sort(key=lambda x: x["jump_time_s"])
            # CASE checks WHENs top-down, so list the latest jump first; the shift paired
            # with jump k is the cumulative sum of jumps 1..k (a row at time s >= t_k has
            # had exactly jumps 1..k applied to it).
            cumulative = [float(x) for x in np.cumsum([j["offset_s"] for j in rec_jumps])]
            when = [float(j["jump_time_s"]) for j in reversed(rec_jumps)]
            shift = list(reversed(cumulative))
            cases = " ".join("WHEN seconds >= %r THEN seconds - %r" % (t, s) for t, s in zip(when, shift))
            shift_cases = " ".join("WHEN seconds >= %r THEN %r" % (t, s) for t, s in zip(when, shift))
            rows_shifted += con.execute(
                "UPDATE tblDetectionRaw SET seconds = CASE %s ELSE seconds END, "
                "jump_offset_s = CASE %s ELSE 0.0 END WHERE Rec_ID = ? AND seconds >= %r"
                % (cases, shift_cases, rec_jumps[0]["jump_time_s"]), (rec,)).rowcount
        for table in METRONOME_TABLES:
            con.execute("DROP TABLE IF EXISTS %s" % table)
        con.commit()
    finally:
        con.close()
    return rows_shifted, sorted(by_rec)


SUPPLEMENT_EDGE_EPOCHS = 4


def _supplement_evidence(con, receiver, low, high):
    """Original-time OffsetChanged/CounterRestart episodes in (low, high]."""
    table = ("tblClockEventEvidence" if con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tblClockEventEvidence'").fetchone()
        else "tblDetectionRaw")
    columns = {r[1] for r in con.execute("PRAGMA table_info(%s)" % table)}
    original = "seconds+COALESCE(jump_offset_s,0)" if "jump_offset_s" in columns else "seconds"
    times = np.array([r[0] for r in con.execute(
        "SELECT %s AS o FROM %s WHERE Rec_ID=? AND (OffsetChanged=1 OR CounterRestart=1) "
        "AND %s>? AND %s<=? ORDER BY o" % (original, table, original, original),
        (receiver, low, high))], dtype=float)
    return np.unique(times)


def _supplement_series(con, master_rec, beacon_tag, period, t0, t1):
    arrivals = beacon_first_arrivals(con, master_rec, beacon_tag, period, t0, t1)
    if arrivals.empty:
        raise ValueError("No paired master-beacon observations for the TDoA supplement")
    columns = {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}
    offset = "COALESCE(jump_offset_s,0)" if "jump_offset_s" in columns else "0"
    raw = pd.read_sql(
        "SELECT Rec_ID, seconds AS t, seconds+%s AS original_s FROM tblDetectionRaw "
        "WHERE Tag_ID=? AND seconds>=? AND seconds<?" % offset, con, params=[beacon_tag, t0, t1])
    raw = raw.drop_duplicates(["Rec_ID", "t"])
    arrivals = arrivals.merge(raw, on=["Rec_ID", "t"], how="left", validate="many_to_one")
    master = (raw[raw.Rec_ID == master_rec][["t", "original_s"]]
              .rename(columns={"t": "t_anchor", "original_s": "master_original_s"}).sort_values("t_anchor"))
    arrivals["t_anchor"] = arrivals.t - arrivals.tdoa_s
    arrivals = pd.merge_asof(arrivals.sort_values("t_anchor"), master, on="t_anchor",
                             direction="nearest", tolerance=1e-4)
    if arrivals[["original_s", "master_original_s"]].isna().any().any():
        raise RuntimeError("Paired beacon observations lack original-time provenance")
    arrivals["delta_s"] = arrivals.tdoa_s
    return arrivals


def _dbscan_steps(clustered, budget):
    """Level changes between consecutive clean DBSCAN regimes of each listener."""
    steps, ambiguous = [], []
    for rec, g in clustered[clustered.dbscan_class == "clean"].groupby("Rec_ID"):
        segments = sorted((s.sort_values("t_anchor") for _, s in g.groupby("label") if len(s) >= MIN_EPOCHS_CLIPPED),
                          key=lambda s: s.t_anchor.iloc[0])
        for pre, post in zip(segments[:-1], segments[1:]):
            window = {"Rec_ID": rec, "anchor_start_s": float(pre.t_anchor.iloc[-1]),
                      "anchor_end_s": float(post.t_anchor.iloc[0]),
                      "master_original_start_s": float(pre.master_original_s.iloc[-1]),
                      "master_original_end_s": float(post.master_original_s.iloc[0]),
                      "original_start_s": float(pre.original_s.iloc[-1]),
                      "original_end_s": float(post.original_s.iloc[0]),
                      "n_pre": len(pre), "n_post": len(post)}
            if post.t_anchor.iloc[0] <= pre.t_anchor.iloc[-1]:
                window.update(original_start_s=float(post.original_s.iloc[0]),
                              original_end_s=float(pre.original_s.iloc[-1]))
                ambiguous.append(dict(window, note="overlapping clean DBSCAN regimes at different levels"))
                continue
            before, after = pre.tail(SUPPLEMENT_EDGE_EPOCHS), post.head(SUPPLEMENT_EDGE_EPOCHS)
            step = float(after.delta_s.median() - before.delta_s.median())
            if abs(step) > budget:
                steps.append(dict(window, step_s=step, pre_spread_s=float(np.ptp(before.delta_s)),
                                  post_spread_s=float(np.ptp(after.delta_s))))
    return steps, ambiguous


def _common_mode_steps(steps, clustered, period, budget, min_step, prior=()):
    """Attribute identical simultaneous listener steps to the master clock.

    A master jump of J shifts every listener's TDoA by -J at the same transmission.
    Steps of at least min_step whose first post-jump epochs fall within three periods,
    agree within two timing budgets and share a common bracket form a group; at least
    three receivers and a third of listeners with clean coverage on both sides are
    attributed to the master. Sub-threshold DBSCAN fragments never participate. A
    master-original bracket already corrected (``prior``) cannot take a second jump.
    """
    if not steps:
        return [], []
    frame = pd.DataFrame(steps).sort_values("anchor_end_s").reset_index(drop=True)
    candidates = frame[frame.step_s.abs() >= min_step]
    clean = clustered[clustered.dbscan_class == "clean"]
    groups = []
    for seed in candidates.itertuples():
        near = candidates[((candidates.anchor_end_s - seed.anchor_end_s).abs() <= 3 * period)
                          & ((candidates.step_s - seed.step_s).abs() <= 2 * budget)]
        near = near.assign(width=near.anchor_end_s - near.anchor_start_s).sort_values("width")
        keep, start, stop = [], -np.inf, np.inf
        for row in near.itertuples():
            if row.Rec_ID in {near.loc[k, "Rec_ID"] for k in keep}:
                continue
            if max(start, row.anchor_start_s) < min(stop, row.anchor_end_s):
                keep.append(row.Index)
                start, stop = max(start, row.anchor_start_s), min(stop, row.anchor_end_s)
        if len(keep) >= 3:
            groups.append((tuple(sorted(keep)), start, stop))
    master, used, taken = [], set(), []
    # The master cannot jump twice at one instant: keep the largest agreeing group per time.
    for keep, start, stop in sorted(set(groups), key=lambda g: (-len(g[0]), g[1])):
        if used.intersection(keep) or any(start < b and a < stop for a, b in taken):
            continue
        agree = candidates.loc[list(keep)]
        original_start = float(agree.loc[agree.anchor_start_s.idxmax(), "master_original_start_s"])
        original_end = float(agree.loc[agree.anchor_end_s.idxmin(), "master_original_end_s"])
        if any(original_start < b and a < original_end for a, b in prior):
            continue
        eligible = sum(1 for _, g in clean.groupby("Rec_ID")
                       if ((g.t_anchor > start - 3 * period) & (g.t_anchor <= start)).any()
                       and ((g.t_anchor >= stop) & (g.t_anchor < stop + 3 * period)).any())
        if 3 * len(agree) < eligible:
            continue
        master.append({
            "step_s": -float(agree.step_s.median()),
            "original_start_s": original_start, "original_end_s": original_end,
            "n_agree": int(len(agree)), "n_eligible": int(eligible),
            "agree_range_s": float(np.ptp(agree.step_s)), "listeners": ";".join(sorted(agree.Rec_ID)),
            "n_pre": int(agree.n_pre.min()), "n_post": int(agree.n_post.min())})
        used.update(keep)
        taken.append((start, stop))
    return master, frame.drop(index=list(used)).to_dict("records")


def tdoa_supplement_sensitivity(series, period, eps_values, min_samples_values,
                               budget=0.0005, window_periods=2.5,
                               min_step=0.05, max_step=5.0, known_events=None):
    """Compare DBSCAN regimes and candidate counts without mutating input data."""
    from beacon_pairwise_dbscan import cluster
    if not np.isfinite(period) or period <= 0 or series.empty:
        raise ValueError("Sensitivity analysis requires nonempty beacon data and a positive period")
    eps_values = tuple(float(value) for value in eps_values)
    min_samples_values = tuple(int(value) for value in min_samples_values)
    if not eps_values or any(not np.isfinite(value) or value <= 0 for value in eps_values):
        raise ValueError("DBSCAN eps values must be finite and positive")
    if not min_samples_values or any(value < 1 for value in min_samples_values):
        raise ValueError("DBSCAN min_samples values must be positive integers")
    if known_events is not None and not {"Rec_ID", "jump_time_s", "offset_s"} <= set(known_events.columns):
        raise ValueError("Known-event labels require Rec_ID, jump_time_s, and offset_s")
    records = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            clustered = cluster(series, period, tolerance_s=budget,
                                window_periods=window_periods, min_samples=min_samples, eps=eps)
            steps, ambiguous = _dbscan_steps(clustered, budget)
            in_range_steps = [row for row in steps
                              if min_step <= abs(row["step_s"]) <= max_step]
            matched_events = set()
            if known_events is not None:
                labels = known_events.sort_values("jump_time_s").to_dict("records")
                for step in sorted(in_range_steps, key=lambda row: row["original_end_s"] - row["original_start_s"]):
                    matches = [i for i, event in enumerate(labels)
                               if i not in matched_events and event["Rec_ID"] == step["Rec_ID"]
                               and step["original_start_s"] <= event["jump_time_s"] <= step["original_end_s"]
                               and np.sign(event["offset_s"]) == np.sign(step["step_s"])
                               and abs(float(event["offset_s"]) - step["step_s"]) <= budget]
                    if matches:
                        matched_events.add(matches[0])
            clean = clustered[clustered.dbscan_class == "clean"]
            classes = clustered.dbscan_class.value_counts()
            records.append({
                "eps": eps,
                "min_samples": min_samples,
                "clean_regimes": int(clean.groupby(["Rec_ID", "label"]).ngroups),
                "clean_epochs": int(len(clean)),
                "noise_epochs": int(classes.get("noise", 0)),
                "steady_reflection_epochs": int(classes.get("steady_reflection", 0)),
                "anchor_suspect_epochs": int(classes.get("anchor_suspect", 0)),
                "steps_over_budget": len(steps),
                "in_range_step_candidates": len(in_range_steps),
                "ambiguous_boundaries": len(ambiguous),
                "known_events": len(known_events) if known_events is not None else np.nan,
                "known_events_matched": len(matched_events) if known_events is not None else np.nan,
                "known_event_recall": (len(matched_events) / len(known_events)
                                       if known_events is not None and len(known_events) else np.nan),
            })
    return pd.DataFrame(records)


def validated_multibeacon_curve(classified, target_times, budget=0.0005,
                                 max_gap_s=120.0, min_overlap=20):
    """Estimate fallback residuals only from clean segments that pass a held-out check.

    `classified` must contain one receiver/beacon series with `t`, `label`, `delta_s`,
    `dbscan_class`, `master_residual`, and `master_supported` columns. The returned
    estimates are diagnostic candidates; callers must require independent-beacon
    agreement before considering any fallback.
    """
    required = {"t", "label", "delta_s", "dbscan_class", "master_residual", "master_supported"}
    if not required <= set(classified.columns):
        raise ValueError("Classified series missing required fallback columns")
    targets = np.asarray(target_times, dtype=float)
    if not np.isfinite(targets).all() or budget <= 0 or max_gap_s <= 0 or min_overlap < 4:
        raise ValueError("Invalid target times or fallback validation settings")
    estimates = []
    summaries = []
    clean = classified[classified.dbscan_class == "clean"]
    for label, segment in clean.groupby("label"):
        segment = segment.sort_values("t")
        times = segment.t.to_numpy(float)
        values = segment.delta_s.to_numpy(float)
        supported = segment.master_supported.to_numpy(bool) & np.isfinite(segment.master_residual.to_numpy(float))
        overlap = np.flatnonzero(supported & np.isfinite(values))
        if len(overlap) < min_overlap:
            continue
        overlap_order = overlap[np.argsort(times[overlap])]
        split = len(overlap_order) // 2
        train, holdout = overlap_order[:split], overlap_order[split:]
        if len(train) < max(2, min_overlap // 2) or len(holdout) < max(2, min_overlap // 2):
            continue
        master_residual = segment.master_residual.to_numpy(float)
        bias = float(np.median(values[train] - master_residual[train]))
        holdout_error = (values[holdout] - bias - master_residual[holdout]) * 1000.0
        p95 = float(np.quantile(np.abs(holdout_error), 0.95))
        if p95 > budget * 1000.0:
            continue
        right = np.searchsorted(times, targets, side="left")
        exact = (right < len(times)) & (times[np.clip(right, 0, len(times) - 1)] == targets)
        bracketed = (right > 0) & (right < len(times)) & ((times[np.clip(right, 0, len(times) - 1)]
                     - times[np.clip(right - 1, 0, len(times) - 1)]) <= max_gap_s)
        eligible = exact | bracketed
        candidate = np.full(len(targets), np.nan)
        if eligible.any():
            candidate[eligible] = np.interp(targets[eligible], times, values - bias)
        estimates.append(candidate)
        summaries.append({"segment": label, "overlap_n": int(len(overlap)),
                          "holdout_n": int(len(holdout)), "holdout_p95_ms": p95,
                          "target_rows": int(np.isfinite(candidate).sum())})
    if not estimates:
        return np.full(len(targets), np.nan), summaries
    stacked = np.vstack(estimates)
    per_target = []
    for column in range(stacked.shape[1]):
        values = stacked[:, column]
        values = values[np.isfinite(values)]
        per_target.append(float(np.median(values))
                          if len(values) == 1 or (len(values) >= 2 and np.ptp(values) <= budget)
                          else np.nan)
    return np.asarray(per_target), summaries


def combine_multibeacon_estimates(estimates, budget=0.0005, min_beacons=2):
    """Return consensus candidates where enough independent beacon curves agree."""
    values = np.asarray(estimates, dtype=float)
    if values.ndim != 2 or budget <= 0 or min_beacons < 2:
        raise ValueError("Expected a 2-D estimate array and at least two agreeing beacons")
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    spread = np.full(values.shape[1], np.inf)
    center = np.full(values.shape[1], np.nan)
    for column in np.flatnonzero(counts >= min_beacons):
        local = values[finite[:, column], column]
        spread[column] = np.ptp(local)
        if spread[column] <= budget:
            center[column] = float(np.median(local))
    return center, (counts >= min_beacons) & (spread <= budget), counts, spread


def _supplement_record(con, receiver, step, kind, pass_no):
    low, high = step["original_start_s"], step["original_end_s"]
    evidence = _supplement_evidence(con, receiver, low, high)
    if len(evidence) == 1:
        boundary, uncertain, start, stop = evidence[0], False, evidence[0], evidence[0]
    elif len(evidence) > 1:
        # Rapid episode: the net step is measured, intermediate states are not.
        boundary, uncertain, start, stop = evidence[-1], True, evidence[0], evidence[-1]
    else:
        boundary, uncertain, start, stop = high, True, low, high
    return dict({k: v for k, v in step.items() if k not in ("Rec_ID", "step_s")},
                Rec_ID=receiver, jump_time_s=float(boundary),
                jump_time_local=_local_ts(boundary).strftime("%Y-%m-%d %H:%M:%S"),
                offset_s=float(step["step_s"]), evidence="tdoa_dbscan_" + kind, stage=2, pass_no=pass_no,
                boundary_uncertain=bool(uncertain), n_internal_evidence=int(len(evidence)),
                window_start_s=float(start), window_end_s=float(stop))


def _apply_supplement_pass(con, records):
    """Shift rows at/after each boundary in ORIGINAL receiver time; offsets accumulate."""
    if "jump_offset_s" not in {r[1] for r in con.execute("PRAGMA table_info(tblDetectionRaw)")}:
        con.execute("ALTER TABLE tblDetectionRaw ADD COLUMN jump_offset_s REAL DEFAULT 0.0")
    shifted = 0
    for rec in sorted({r["Rec_ID"] for r in records}):
        local = sorted((r for r in records if r["Rec_ID"] == rec), key=lambda r: r["jump_time_s"])
        cumulative = np.cumsum([r["offset_s"] for r in local])
        cases = " ".join("WHEN seconds+COALESCE(jump_offset_s,0) >= %r THEN %r" % (r["jump_time_s"], float(s))
                         for r, s in zip(reversed(local), reversed(cumulative)))
        shifted += con.execute(
            "UPDATE tblDetectionRaw SET seconds = seconds - (CASE %s END), "
            "jump_offset_s = COALESCE(jump_offset_s,0) + (CASE %s END) "
            "WHERE Rec_ID = ? AND seconds+COALESCE(jump_offset_s,0) >= ?" % (cases, cases),
            (rec, local[0]["jump_time_s"])).rowcount
    return shifted


def _ledger_corrected_sides(ledger):
    """Corrected receiver time just before/after each applied original-time boundary."""
    ledger = ledger.copy()
    ledger["corrected_before_s"] = np.nan
    ledger["corrected_after_s"] = np.nan
    for _, local in ledger.groupby("Rec_ID"):
        first = local[local.stage == 1].sort_values("jump_time_s")
        times, cumulative = first.jump_time_s.to_numpy(float), np.cumsum(first.offset_s.to_numpy(float))
        second = local[local.stage == 2]

        def total(original, side):
            k = np.searchsorted(times, original, side=side) - 1
            value = cumulative[k] if k >= 0 else 0.0
            later = second.jump_time_s <= original if side == "right" else second.jump_time_s < original
            return value + float(second.loc[later, "offset_s"].sum())
        for index, row in local.iterrows():
            ledger.loc[index, "corrected_before_s"] = row.jump_time_s - total(row.jump_time_s, "left")
            ledger.loc[index, "corrected_after_s"] = row.jump_time_s - total(row.jump_time_s, "right")
    return ledger


def revert_tdoa_supplement(db, source_db):
    """Restore the post-internal-code state exactly, removing stage-2 corrections.

    Rows are recomputed from the unshifted source by original rowid with the same
    arithmetic as apply_offsets, so the result is bit-identical to stage 1 alone.
    """
    source_db = Path(source_db).resolve()
    con = sqlite3.connect(Path(db).resolve().as_uri(), uri=True)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='tblClockJumpEvents'").fetchone():
            raise RuntimeError("No TDoA supplement is applied; nothing to revert")
        ledger = pd.read_sql("SELECT * FROM tblClockJumpEvents", con)
        if "stage" not in ledger or not (ledger.stage == 2).any():
            raise RuntimeError("No TDoA supplement is applied; nothing to revert")
        con.execute("ATTACH DATABASE ? AS src", (source_db.as_uri() + "?mode=ro",))
        if "jump_offset_s" in {r[1] for r in con.execute("PRAGMA src.table_info(tblDetectionRaw)")}:
            raise ValueError("Revert source must be the unshifted baseline")
        first = ledger[ledger.stage == 1]
        if first.empty:
            first = pd.DataFrame(columns=["Rec_ID", "jump_time_s", "offset_s", "evidence", "stage"])
        restored = 0
        for rec in sorted(ledger.loc[ledger.stage == 2, "Rec_ID"].unique()):
            local = first[first.Rec_ID == rec].sort_values("jump_time_s")
            if len(local):
                cumulative = np.cumsum(local.offset_s.to_numpy(float))
                shift = " ".join("WHEN s.seconds >= %r THEN %r" % (float(t), float(c)) for t, c in
                                 zip(reversed(local.jump_time_s.to_numpy(float)), reversed(cumulative)))
                expression = "CASE %s ELSE 0.0 END" % shift
            else:
                expression = "0.0"
            restored += con.execute(
                "UPDATE tblDetectionRaw SET seconds = s.seconds - (%s), jump_offset_s = %s "
                "FROM src.tblDetectionRaw AS s WHERE s.rowid = tblDetectionRaw.rowid "
                "AND tblDetectionRaw.Rec_ID = ?" % (expression, expression), (rec,)).rowcount
        if con.execute("SELECT count(*) FROM tblDetectionRaw d LEFT JOIN src.tblDetectionRaw s "
                       "ON s.rowid = d.rowid WHERE s.rowid IS NULL OR s.Rec_ID != d.Rec_ID").fetchone()[0]:
            raise RuntimeError("Compact rows do not map to the source by original rowid")
        original_columns = [c for c in first.columns if c not in ("corrected_before_s", "corrected_after_s")]
        stage_one = first[original_columns]
        (stage_one.dropna(axis=1, how="all") if len(stage_one) else stage_one).to_sql(
            "tblClockJumpEvents", con, if_exists="replace", index=False)
        candidate_table = con.execute(
            "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='tblClockJumpCandidates'"
        ).fetchone()
        if candidate_table:
            candidate_columns = {row[1] for row in con.execute("PRAGMA main.table_info(tblClockJumpCandidates)")}
            if "stage" not in candidate_columns:
                raise RuntimeError("Cannot safely retain stage-1 candidates without a stage column")
            con.execute("DELETE FROM main.tblClockJumpCandidates WHERE stage=2")
        for table in METRONOME_TABLES:
            # Qualified: an unqualified DROP would resolve to the attached source.
            con.execute("DROP TABLE IF EXISTS main.%s" % table)
        con.commit()
        con.execute("DETACH DATABASE src")
    finally:
        con.close()
    return {"restored_rows": restored, "removed_stage2_events": int((ledger.stage == 2).sum())}


def tdoa_supplement(db, receivers, master_rec, master_tag, period, window_start, window_end,
                    min_step=0.05, max_step=5.0, budget=0.0005, max_passes=4,
                    dbscan_eps=1.0, dbscan_min_samples=3, dbscan_window_periods=2.5):
    """Supplementary jumps from master-beacon TDoA after internal-code corrections.

    The existing beacon DBSCAN (0.5 ms budget, 2.5 periods, 3 samples) splits each
    listener's TDoA into stable regimes; its cluster count is data-driven. Level changes
    between consecutive regimes are jumps without a usable internal code. Simultaneous
    identical steps are corrected on the master, then remaining steps per listener.
    Internal evidence inside the regime gap pins the boundary; otherwise the gap is an
    unresolved window. One transaction: nothing is committed unless every pass succeeds.
    """
    from beacon_pairwise_dbscan import cluster
    t0, t1 = naive_seconds(window_start), naive_seconds(window_end)
    con = sqlite3.connect(db)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "tblClockJumpEvents" in tables:
            existing = pd.read_sql("SELECT * FROM tblClockJumpEvents", con)
            if "stage" in existing and (existing.stage == 2).any():
                raise RuntimeError("TDoA supplement already applied; refusing a second application")
        else:
            existing = pd.DataFrame(columns=["Rec_ID", "jump_time_s", "offset_s", "evidence"])
        applied, unresolved, passes = [], [], []
        for pass_no in range(1, max_passes + 2):
            series = _supplement_series(con, master_rec, master_tag, period, t0, t1)
            clustered = cluster(series[series.Rec_ID.isin(receivers)], period,
                                tolerance_s=budget, window_periods=dbscan_window_periods,
                                min_samples=dbscan_min_samples, eps=dbscan_eps)
            steps, ambiguous = _dbscan_steps(clustered, budget)
            master_steps, remaining = _common_mode_steps(
                steps, clustered, period, budget, min_step,
                prior=[(r["original_start_s"], r["original_end_s"]) for r in applied
                       if r["evidence"] == "tdoa_dbscan_master"])
            sized = [s for s in remaining if min_step <= abs(s["step_s"]) <= max_step]
            passes.append({"pass": pass_no, "steps_over_budget": len(steps), "master_common_mode": len(master_steps),
                           "listener_steps_in_range": len(sized), "ambiguous": len(ambiguous),
                           "classes": clustered.dbscan_class.value_counts().to_dict()})
            if pass_no <= max_passes and master_steps:
                records = [_supplement_record(con, master_rec, s, "master", pass_no) for s in master_steps
                           if abs(s["step_s"]) <= max_step]
            elif pass_no <= max_passes and sized:
                records = [_supplement_record(con, s["Rec_ID"], s, "receiver", pass_no) for s in sized]
            else:
                records = []
            if not records:
                for s in steps:
                    if abs(s["step_s"]) >= min_step:
                        unresolved.append(dict(s, status="unresolved", window_start_s=s["original_start_s"],
                                               window_end_s=s["original_end_s"], jump_time_s=s["original_end_s"],
                                               note="residual TDoA step after supplementary corrections"))
                for s in ambiguous:
                    unresolved.append(dict(s, status="unresolved", window_start_s=s["original_start_s"],
                                           window_end_s=s["original_end_s"], jump_time_s=s["original_end_s"]))
                break
            passes[-1]["rows_shifted"] = _apply_supplement_pass(con, records)
            applied.extend(records)
        ledger = pd.concat([existing.assign(stage=existing.get("stage", 1)), pd.DataFrame(applied)],
                           ignore_index=True)
        if len(ledger):
            ledger = _ledger_corrected_sides(ledger)
            ledger.to_sql("tblClockJumpEvents", con, if_exists="replace", index=False)
        candidates = ([dict(r, status="unresolved" if r["boundary_uncertain"] else "accepted",
                            note="supplementary boundary uncertain: no single internal event in the regime gap"
                            if r["boundary_uncertain"] else "supplementary jump pinned to internal evidence")
                       for r in applied] + unresolved
                      + [dict(r, status="accepted", window_start_s=r["jump_time_s"], window_end_s=r["jump_time_s"],
                              note="internal-code jump") for r in existing.to_dict("records")])
        report = pd.DataFrame(candidates) if candidates else pd.DataFrame(
            columns=["Rec_ID", "jump_time_s", "status", "window_start_s", "window_end_s", "note"])
        report.to_sql("tblClockJumpCandidates", con, if_exists="replace", index=False)
        for table in METRONOME_TABLES:
            con.execute("DROP TABLE IF EXISTS %s" % table)
        con.commit()
    finally:
        con.close()
    return {"applied": len(applied), "master_jumps": sum(r["evidence"] == "tdoa_dbscan_master" for r in applied),
            "receiver_jumps": sum(r["evidence"] == "tdoa_dbscan_receiver" for r in applied),
            "boundary_uncertain": sum(r["boundary_uncertain"] for r in applied),
            "dbscan": {"eps": dbscan_eps, "min_samples": dbscan_min_samples,
                       "window_periods": dbscan_window_periods, "budget_s": budget},
            "unresolved_residual": len(unresolved), "passes": passes}


def run(db, receivers, master_rec, master_tag, period, window_start, window_end,
        window_s=300.0, min_step=0.05, max_step=5.0, apply=True,
        group_span_s=None, conservative=False, persist_diagnostics=False):
    """Full pre-step. apply=True writes tblClockJumpEvents, shifts tblDetectionRaw and
    invalidates the metronome tables. Returns (notes dict, jumps list)."""
    if conservative and group_span_s is None:
        raise ValueError("Conservative mode requires an explicit trigger grouping span")
    t0, t1 = naive_seconds(window_start), naive_seconds(window_end)
    con = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(db).replace("\\", "/"), uri=True)
    try:
        if conservative and any(r[1] == "jump_offset_s" for r in con.execute("PRAGMA table_info(tblDetectionRaw)")):
            raise RuntimeError("Conservative quantification requires unshifted raw timestamps and original event evidence")
        jumps, evidence_only, unresolvable, n_events = quantify_jumps(
            con, receivers, master_rec, master_tag, period, t0, t1, window_s, min_step, max_step,
            group_span_s=group_span_s)
    finally:
        con.close()
    rows_shifted, affected = (0, [])
    if conservative:
        accepted = []
        for event in jumps:
            reasons = []
            if event["common_mode"]:
                reasons.append("common-mode reference attribution requires review")
            if max(event["pre_spread_s"], event["post_spread_s"]) > 0.0005:
                reasons.append("beacon plateau spread exceeds the 0.5 ms timing warning budget")
            if event["trigger_count"] > 1:
                reasons.append("grouped trigger boundary is uncertain; review before applying")
            if event["offset_s"] < 0:
                reasons.append("backward jump can overlap pre-event timestamps; source-order boundary review required")
            if reasons:
                unresolvable.append(dict(event, note="; ".join(reasons)))
            else:
                accepted.append(event)
        jumps = accepted
        clean_evidence = []
        for event in evidence_only:
            if abs(event["offset_s"]) > max_step:
                unresolvable.append(dict(event, note="step exceeds configured maximum; not evidence-only"))
            elif (abs(event["offset_s"]) > 0.0005
                  or max(event["pre_spread_s"], event["post_spread_s"]) > 0.0005):
                unresolvable.append(dict(event, note="sub-threshold step or unstable plateau exceeds 0.5 ms warning budget"))
            else:
                clean_evidence.append(event)
        evidence_only = clean_evidence
        if apply or persist_diagnostics:
            with closing(sqlite3.connect(db)) as report_con:
                records = ([dict(e, status="accepted") for e in jumps]
                           + [dict(e, status="evidence_only") for e in evidence_only]
                           + [dict(e, status="unresolved") for e in unresolvable])
                report = (pd.DataFrame(records) if records else pd.DataFrame(columns=[
                    "Rec_ID", "jump_time_s", "status", "window_start_s", "window_end_s", "note"]))
                report.to_sql("tblClockJumpCandidates", report_con, if_exists="replace", index=False)
    if apply and jumps:
        rows_shifted, affected = apply_offsets(db, jumps)
    notes = {"events": n_events,
             "n_jumps": len(jumps), "evidence_only": len(evidence_only), "unresolvable": len(unresolvable),
             "rows_shifted": rows_shifted, "affected_receivers": affected,
             "window_s": window_s, "min_step_s": min_step, "max_step_s": max_step,
             "group_span_s": group_span_s, "conservative": conservative,
             "jumps_by_receiver": _counts(jumps), "largest_jump_ms":
             round(max((abs(j["offset_s"]) for j in jumps), default=0.0) * 1e3, 1),
             "jumps": [{k: j[k] for k in ("Rec_ID", "jump_time_local", "offset_s", "evidence",
                                          "n_pre", "n_post", "common_mode")} for j in jumps]}
    return notes, jumps


def _counts(jumps):
    out = {}
    for j in jumps:
        out[j["Rec_ID"]] = out.get(j["Rec_ID"], 0) + 1
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database")
    parser.add_argument("--master-receiver", required=True)
    parser.add_argument("--beacon-tag", required=True)
    parser.add_argument("--period", type=float, required=True, help="Beacon pulse rate (s)")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--receivers", nargs="*", default=None, help="Default: all receivers with events")
    parser.add_argument("--window", type=float, default=300.0, help="Seconds per side of each event")
    parser.add_argument("--min-step", type=float, default=0.05, help="Accept a jump at/above this (s)")
    parser.add_argument("--max-step", type=float, default=5.0, help="Reject a jump above this (s)")
    parser.add_argument("--report", help="Write CSV report to this folder (read-only on the DB)")
    parser.add_argument("--apply", action="store_true",
                        help="APPLY the offsets to tblDetectionRaw (default: report only)")
    args = parser.parse_args(argv)

    con = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(args.database).replace("\\", "/"), uri=True)
    t0, t1 = naive_seconds(args.start), naive_seconds(args.end)
    if args.receivers:
        receivers = list(args.receivers)
    else:
        receivers = [r[0] for r in con.execute("SELECT DISTINCT Rec_ID FROM tblDetectionRaw ORDER BY Rec_ID")]
    jumps, evidence_only, unresolvable, n_events = quantify_jumps(
        con, receivers, args.master_receiver, args.beacon_tag, args.period, t0, t1,
        args.window, args.min_step, args.max_step)
    con.close()
    print("Internal event rows scanned: %d" % n_events)

    print("Pre-sync jump scan: master %s beacon %s (period %.3f s), %d receivers, %s .. %s"
          % (args.master_receiver, args.beacon_tag, args.period, len(receivers), args.start, args.end))
    print("Jumps accepted: %d | evidence-only (no step): %d | unresolvable: %d"
          % (len(jumps), len(evidence_only), len(unresolvable)))
    for j in jumps:
        print("  %s  %s  offset=%+8.1f ms  n=%d/%d  %s%s" % (
            j["Rec_ID"], j["jump_time_local"], j["offset_s"] * 1e3, j["n_pre"], j["n_post"],
            j["evidence"], "  [common-mode x%d]" % j["common_mode"] if j.get("common_mode") else ""))
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        pd.DataFrame(jumps).to_csv(os.path.join(args.report, "jumps.csv"), index=False)
        pd.DataFrame(evidence_only).to_csv(os.path.join(args.report, "evidence_only.csv"), index=False)
        pd.DataFrame(unresolvable).to_csv(os.path.join(args.report, "unresolvable.csv"), index=False)
        print("Report written to %s" % args.report)
    if args.apply:
        if not jumps:
            print("No jumps to apply.")
            return 0
        rows, affected = apply_offsets(args.database, jumps)
        print("Applied %d jumps on %s to tblDetectionRaw (%d rows shifted); metronome tables invalidated."
              % (len(jumps), affected, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
